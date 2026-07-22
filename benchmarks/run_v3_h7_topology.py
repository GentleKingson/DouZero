"""Run one checkpoint-enabled H7 topology benchmark repetition."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import torch

from douzero._version import git_sha
from douzero.env.rules import RuleSet
from douzero.observation.schema import build_v2_schema
from douzero.v3_hybrid import V3HybridModel
from douzero.v3_hybrid.benchmark import (
    H7_BENCHMARK_SCHEMA,
    V3H7BenchmarkProtocol,
)
from douzero.v3_hybrid.h7_smoke import build_v3_h7_smoke_config
from douzero.v3_hybrid.runtime import (
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
    V3SingleProcessTrainer,
)
from douzero.v3_hybrid.support_matrix import (
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
)
from douzero.v3_hybrid.training.h6_learner import V3H6Learner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--topology", choices=("single_process", "async_4x4", "async_8x4"),
        required=True,
    )
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_protocol(path: Path) -> V3H7BenchmarkProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema", None) != H7_BENCHMARK_SCHEMA:
        raise ValueError("H7 benchmark protocol file schema mismatch")
    seeds = payload.get("seeds")
    if isinstance(seeds, list):
        payload["seeds"] = tuple(seeds)
    protocol = V3H7BenchmarkProtocol(**payload)
    if git_sha() != protocol.source_git_sha:
        raise ValueError("H7 benchmark source SHA does not match the frozen protocol")
    return protocol


def _shared_memory_bytes(trainer) -> int:
    if not getattr(trainer, "_runtime_started", False):
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for owner in (
        trainer._coordinator,
        trainer._coordinator.slots,
        trainer._replay_slots,
        trainer._replay_slots.observations,
    ):
        for value in vars(owner).values():
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def _run_until(trainer, deadline: float, *, episodes: int) -> tuple[int, int]:
    steps = 0
    max_lag = 0
    while time.monotonic() < deadline:
        trainer.collect_episodes(episodes)
        if trainer.step() is None:
            continue
        steps += 1
        max_lag = max(max_lag, trainer.policy_step - trainer._snapshot_step)
    return steps, max_lag


def main() -> None:
    args = _parser().parse_args()
    protocol = _load_protocol(args.protocol)
    seed = protocol.seeds[args.repeat]
    resolved = build_v3_h7_smoke_config()
    if resolved.stable_hash() != protocol.config_hash:
        raise ValueError("H7 benchmark config hash mismatch")
    if resolved.model.stable_hash() != protocol.model_identity_hash:
        raise ValueError("H7 benchmark model hash mismatch")

    if args.topology == "single_process":
        topology = TOPOLOGY_SINGLE_PROCESS
        actors, games, episodes = 1, 1, 4
        trainer_type = V3SingleProcessTrainer
    else:
        topology = TOPOLOGY_ASYNC_SINGLE_GPU
        actors = 4 if args.topology == "async_4x4" else 8
        games, episodes = 4, 4
        trainer_type = V3AsyncSingleGPUTrainer
    runtime_config = V3H7RuntimeConfig(
        topology=topology,
        num_actors=actors,
        games_per_actor=games,
        batch_size=32,
        replay_capacity=4096,
        target_microbatch=4,
        environment_seed=seed,
        action_seed=seed + 1,
    )
    model = V3HybridModel(build_v2_schema(), resolved.model)
    learner = V3H6Learner(model, ruleset=RuleSet.legacy(), config=resolved)
    trainer = trainer_type(learner, resolved, runtime_config)
    try:
        _run_until(
            trainer, time.monotonic() + protocol.warmup_seconds,
            episodes=episodes,
        )
        trainer.quiesce_cycle_boundary()
        torch.cuda.reset_peak_memory_stats(trainer.device)
        before = {
            "games": trainer.stats.games_collected,
            "decisions": trainer.stats.decisions_collected,
            "transitions": trainer.stats.transitions_collected,
            "samples": trainer.stats.learner_cardplay_samples,
            "steps": trainer.stats.optimizer_steps,
        }
        parameter_snapshot = trainer._parameter_update_snapshot()
        started = time.monotonic()
        steps, max_lag = _run_until(
            trainer, started + protocol.measurement_seconds,
            episodes=episodes,
        )
        elapsed = time.monotonic() - started
        boundary = trainer.quiesce_cycle_boundary()
        after = {
            "games": trainer.stats.games_collected,
            "decisions": trainer.stats.decisions_collected,
            "transitions": trainer.stats.transitions_collected,
            "samples": trainer.stats.learner_cardplay_samples,
            "steps": trainer.stats.optimizer_steps,
        }
        shared_memory = _shared_memory_bytes(trainer)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_training_checkpoint(
            str(args.checkpoint),
            long_running_state={
                "benchmark_protocol_hash": protocol.stable_hash(),
                "topology": args.topology,
                "repeat": args.repeat,
            },
        )
        changed = trainer._parameters_changed_since(parameter_snapshot)
        cpu_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        shutdown_started = time.monotonic()
        trainer.shutdown()
        shutdown_seconds = time.monotonic() - shutdown_started
        trainer = None
        record = {
            "schema": H7_BENCHMARK_SCHEMA,
            "protocol_hash": protocol.stable_hash(),
            "topology": args.topology,
            "repeat": args.repeat,
            "seed": seed,
            "measurement_seconds": elapsed,
            "checkpoint_path": str(args.checkpoint),
            "parameter_update_observed": changed and steps > 0,
            "active_slots": int(boundary["active_slots"]),
            "in_flight": int(boundary["in_flight_slots"]),
            "pending": int(boundary["pending_requests"]),
            "games_per_second": (after["games"] - before["games"]) / elapsed,
            "decisions_per_second": (
                after["decisions"] - before["decisions"]
            ) / elapsed,
            "transitions_per_second": (
                after["transitions"] - before["transitions"]
            ) / elapsed,
            "learner_samples_per_second": (
                after["samples"] - before["samples"]
            ) / elapsed,
            "optimizer_steps_per_second": (
                after["steps"] - before["steps"]
            ) / elapsed,
            "requests_per_microbatch": float(boundary["requests_per_microbatch"]),
            "legal_actions_per_batch": float(boundary["actions_per_microbatch"]),
            "queue_wait_seconds": float(boundary["claim_wait_seconds"]),
            "slot_read_seconds": float(boundary["slot_read_seconds"]),
            "collate_seconds": float(boundary["collate_seconds"]),
            "h2d_seconds": float(boundary["h2d_seconds"]),
            "forward_seconds": float(boundary["forward_seconds"]),
            "d2h_seconds": float(boundary["d2h_seconds"]),
            "publish_seconds": float(boundary["publish_seconds"]),
            "replay_drain_seconds": float(boundary["replay_drain_seconds"]),
            "learner_throttle_seconds": float(boundary["learner_throttle_seconds"]),
            "actor_blocked_ratio": float(boundary["actor_blocked_ratio"]),
            "learner_data_wait_ratio": float(boundary["learner_data_wait_ratio"]),
            "policy_lag_max": float(max_lag),
            "cpu_ram_bytes": float(cpu_ram),
            "shared_memory_bytes": float(shared_memory),
            "vram_bytes": float(torch.cuda.max_memory_allocated()),
            "shutdown_seconds": shutdown_seconds,
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in record.values()
        ):
            raise ValueError("H7 benchmark produced a non-finite metric")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    finally:
        if trainer is not None:
            trainer.shutdown()


if __name__ == "__main__":
    main()
