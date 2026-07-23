#!/usr/bin/env python3
"""Run one commit-bound P2 pilot variant with atomic checkpoint evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import time
from pathlib import Path

import torch

from douzero._version import environment_info, git_sha
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    P2_PILOT_PROTOCOL,
    P2_PILOT_SCHEMA,
    P2_VARIANTS,
    _sha256,
    collect_real_pilot_episode,
    create_pilot_learner,
    slice_pilot_batch,
    train_pilot_batch,
    write_pilot_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen P2 real-environment V3 pilot variant."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--command-record", default="")
    return parser.parse_args()


def _finite_metrics(value):
    if isinstance(value, dict):
        return {str(key): _finite_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_metrics(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("pilot metrics contain NaN or Inf")
        return value
    return _finite_metrics(vars(value))


def main() -> int:
    args = _parse_args()
    formal = load_formal_config(args.config)
    if formal.variant not in P2_VARIANTS:
        raise SystemExit("P2 pilot requires one of the six frozen V3 variants")
    budget = formal.budgets["pilot"]
    max_seconds = float(args.max_seconds or budget.wall_clock_seconds)
    max_samples = int(args.max_samples or budget.sample_budget)
    max_steps = int(args.max_optimizer_steps or budget.optimizer_step_budget)
    checkpoint_every = int(
        args.checkpoint_every or formal.runtime.checkpoint_cadence_updates
    )
    if min(max_seconds, max_samples, max_steps, checkpoint_every) <= 0:
        raise SystemExit("pilot limits must be positive")
    source_sha = git_sha()
    if len(source_sha) != 40:
        raise SystemExit("P2 evidence requires a full commit-bound Git SHA")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    learner, resolved = create_pilot_learner(formal)
    resumed_from_samples = 0
    resumed_from_steps = 0
    if args.resume:
        if not checkpoint.is_file():
            raise SystemExit("--resume requires the existing latest.pt manifest target")
        learner.load_checkpoint(checkpoint)
        resumed_from_samples = learner.samples_consumed
        resumed_from_steps = learner.eligible_updates
    elif checkpoint.exists():
        raise SystemExit("fresh pilot refuses to overwrite an existing checkpoint")

    stopped = False
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stopped, stop_signal
        stopped = True
        stop_signal = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    seed = formal.seeds.training[0]
    environment_seed = args.environment_seed if args.environment_seed is not None else seed
    action_seed = args.action_seed if args.action_seed is not None else seed + 1
    action_rng = random.Random(action_seed)
    episodes = 0
    decisions = 0
    skipped_long_episodes = 0
    winners = {"landlord": 0, "farmer": 0}
    last_metrics = None
    started_wall = time.time()
    started = time.monotonic()
    status = "completed"
    failure = None
    try:
        while (
            not stopped
            and time.monotonic() - started < max_seconds
            and learner.samples_consumed < max_samples
            and learner.eligible_updates < max_steps
        ):
            batch = collect_real_pilot_episode(
                learner,
                episode_number=episodes,
                environment_seed=environment_seed,
                action_rng=action_rng,
                epsilon=args.epsilon,
            )
            episodes += 1
            decisions += batch.decisions
            winners[batch.winner_team] += 1
            batch_size = formal.runtime.batch_size
            if batch.trajectories is not None:
                if len(batch.transitions) > batch_size:
                    skipped_long_episodes += 1
                    continue
                pieces = (batch,)
            else:
                pieces = tuple(
                    slice_pilot_batch(batch, index, min(index + batch_size, len(batch.transitions)))
                    for index in range(0, len(batch.transitions), batch_size)
                )
            for piece in pieces:
                if (
                    stopped
                    or learner.samples_consumed + len(piece.transitions) > max_samples
                    or learner.eligible_updates >= max_steps
                ):
                    break
                last_metrics = train_pilot_batch(learner, piece).as_dict()
                if learner.eligible_updates % checkpoint_every == 0:
                    learner.save_checkpoint(checkpoint)
        if stopped:
            status = "stopped"
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        learner.save_checkpoint(checkpoint)
        elapsed = time.monotonic() - started
        env = environment_info()
        env.update({
            "hostname": socket.gethostname(),
            "image_digest": args.image_digest,
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "command_record": args.command_record or None,
        })
        summary = {
            "schema": P2_PILOT_SCHEMA,
            "protocol": P2_PILOT_PROTOCOL,
            "source_git_sha": source_sha,
            "formal_config_sha256": formal.identity_dict()["config_sha256"],
            "training_semantics_hash": formal.identity_dict()["training_semantics_hash"],
            "variant": formal.variant,
            "ruleset": formal.ruleset["id"],
            "seed": seed,
            "status": status,
            "started_at": started_wall,
            "finished_at": time.time(),
            "wall_clock_seconds": elapsed,
            "samples": learner.samples_consumed,
            "optimizer_steps": learner.eligible_updates,
            "episodes": episodes,
            "decisions": decisions,
            "metrics": {
                "last_step": _finite_metrics(last_metrics),
                "samples_per_second": learner.samples_consumed / max(elapsed, 1e-9),
                "optimizer_steps_per_second": learner.eligible_updates / max(elapsed, 1e-9),
                "games_per_second": episodes / max(elapsed, 1e-9),
                "skipped_long_cooperation_episodes": skipped_long_episodes,
                "winner_counts": winners,
                "resolved_config_hash": resolved.stable_hash(),
                "model_hash": learner.model.config.stable_hash(),
                "policy_version": learner.policy_version,
            },
            "resume": {
                "requested": bool(args.resume),
                "from_samples": resumed_from_samples,
                "from_optimizer_steps": resumed_from_steps,
                "continued_update": learner.eligible_updates > resumed_from_steps,
                "stop_signal": stop_signal,
            },
            "evaluation": {
                "paired_deals": 0,
                "status": "not_executed_by_training_runner",
                "playing_strength_claim": False,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "saved": True,
            },
            "environment": env,
            "release_candidate": "NONE",
            "release_status": "NOT READY",
            "playing_strength": "NOT MEASURED",
            "failure": failure,
        }
        write_pilot_summary(output / "training-summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
