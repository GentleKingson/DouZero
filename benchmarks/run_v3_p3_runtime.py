#!/usr/bin/env python3
"""Run one committed, checkpoint-enabled P3 matched-runtime repetition."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
if not sys.path or sys.path[0] != str(_ROOT):
    sys.path.insert(0, str(_ROOT))

import torch

from douzero._version import git_sha
from douzero.training.v2_buffer import Episode
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    _sha256,
    collect_real_pilot_episode,
    create_pilot_learner,
    train_pilot_batch,
)
from douzero.v3_hybrid.runtime import (
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
    V3SingleProcessTrainer,
)
from douzero.v3_hybrid.runtime_decision import (
    P3_RUNTIME_SCHEMA,
    P3_SEGMENTS,
    P3RuntimeProtocol,
)
from douzero.v3_hybrid.support_matrix import (
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
)
from tools.run_v3_pilot import (
    _HOST_PROC,
    _DOCKER_SOCKET,
    _attest_clean_source,
    _attest_current_container,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument(
        "--topology",
        choices=(
            "base_single_process",
            "base_async_4x4",
            "base_async_8x4",
            "full_hybrid_single_process",
        ),
        required=True,
    )
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_protocol(path: Path) -> P3RuntimeProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema", None) != P3_RUNTIME_SCHEMA:
        raise ValueError("P3 runtime protocol schema mismatch")
    payload["seeds"] = tuple(payload["seeds"])
    return P3RuntimeProtocol(**payload)


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SegmentProfiler:
    """Low-overhead semantic wall timers; nested segments may overlap."""

    def __init__(self) -> None:
        self.values = {name: 0.0 for name in P3_SEGMENTS}

    @contextlib.contextmanager
    def measure(self, name: str):
        if name not in self.values:
            raise KeyError(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] += time.perf_counter() - started

    def wrap(self, name: str, function):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)

        return measured


def _counter_payload(stats) -> dict[str, int]:
    return {
        "games": int(stats.games_collected),
        "decisions": int(stats.decisions_collected),
        "transitions": int(stats.transitions_collected),
        "learner_samples": int(stats.learner_cardplay_samples),
        "optimizer_steps": int(stats.optimizer_steps),
    }


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


def _run_runtime_until(trainer, deadline: float, episodes: int) -> int:
    max_lag = 0
    while time.monotonic() < deadline:
        trainer.collect_episodes(episodes)
        trainer.step()
        max_lag = max(max_lag, trainer.policy_step - trainer._snapshot_step)
    return max_lag


def _strict_runtime_reload(
    trainer_type,
    formal,
    runtime_config,
    checkpoint: Path,
) -> bool:
    learner, resolved = create_pilot_learner(formal)
    restored = trainer_type(learner, resolved, runtime_config)
    try:
        restored.load_training_checkpoint(checkpoint)
        return True
    finally:
        restored.shutdown()


def _run_base(
    protocol: P3RuntimeProtocol,
    formal,
    topology: str,
    seed: int,
    checkpoint: Path,
):
    if topology == "base_single_process":
        runtime_topology = TOPOLOGY_SINGLE_PROCESS
        trainer_type = V3SingleProcessTrainer
        actors, games, episodes = 1, 1, 4
    else:
        runtime_topology = TOPOLOGY_ASYNC_SINGLE_GPU
        trainer_type = V3AsyncSingleGPUTrainer
        actors = 4 if topology == "base_async_4x4" else 8
        games, episodes = 4, 4
    learner, resolved = create_pilot_learner(formal)
    runtime_config = V3H7RuntimeConfig(
        topology=runtime_topology,
        num_actors=actors,
        games_per_actor=games,
        batch_size=protocol.batch_size,
        replay_capacity=formal.runtime.replay_capacity,
        target_microbatch=4,
        environment_seed=seed,
        action_seed=seed + 1,
        max_policy_lag=protocol.max_policy_lag,
    )
    trainer = trainer_type(learner, resolved, runtime_config)
    checkpoint_seconds = 0.0
    shutdown_seconds = 0.0
    try:
        _run_runtime_until(
            trainer, time.monotonic() + protocol.warmup_seconds, episodes
        )
        trainer.quiesce_cycle_boundary()
        torch.cuda.reset_peak_memory_stats(trainer.device)
        before = _counter_payload(trainer.stats)
        parameter_before = _model_digest(trainer.model)
        started = time.monotonic()
        max_lag = _run_runtime_until(
            trainer, started + protocol.measurement_seconds, episodes
        )
        elapsed = time.monotonic() - started
        boundary = trainer.quiesce_cycle_boundary()
        after = _counter_payload(trainer.stats)
        parameter_changed = parameter_before != _model_digest(trainer.model)
        shared_memory = _shared_memory_bytes(trainer)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started = time.perf_counter()
        trainer.save_training_checkpoint(
            str(checkpoint),
            long_running_state={
                "p3_protocol_hash": protocol.stable_hash(),
                "topology": topology,
                "seed": seed,
            },
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        cpu_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        vram = int(torch.cuda.max_memory_allocated())
        shutdown_started = time.monotonic()
        trainer.shutdown()
        shutdown_seconds = time.monotonic() - shutdown_started
        trainer = None
        strict_reload = _strict_runtime_reload(
            trainer_type, formal, runtime_config, checkpoint
        )
        segments = {name: 0.0 for name in P3_SEGMENTS}
        segments.update({
            "public_model_forward": float(boundary["forward_seconds"]),
            "collate": float(boundary["collate_seconds"]),
            "h2d": float(boundary["h2d_seconds"]),
            "d2h": float(boundary["d2h_seconds"]),
            "queue": float(boundary["claim_wait_seconds"]),
            "replay": float(boundary["replay_drain_seconds"]),
            "checkpoint": checkpoint_seconds,
        })
        return {
            "before": before,
            "after": after,
            "elapsed": elapsed,
            "segments": segments,
            "parameter_changed": parameter_changed,
            "checkpoint_reload": strict_reload,
            "policy_lag": max_lag,
            "actor_blocked": float(boundary["actor_blocked_ratio"]),
            "data_wait": float(boundary["learner_data_wait_ratio"]),
            "cpu_ram": cpu_ram,
            "shared_memory": shared_memory,
            "vram": vram,
            "active": int(boundary["active_slots"]),
            "in_flight": int(boundary["in_flight_slots"]),
            "pending": int(boundary["pending_requests"]),
            "shutdown": shutdown_seconds,
            "skipped": 0,
        }
    finally:
        if trainer is not None:
            trainer.shutdown()


def _full_counter(state: Mapping[str, int]) -> dict[str, int]:
    return {
        "games": state["games"],
        "decisions": state["decisions"],
        "transitions": state["transitions"],
        "learner_samples": state["samples"],
        "optimizer_steps": state["steps"],
    }


def _run_full_until(learner, *, seed: int, deadline: float, state, profiler) -> None:
    batch_size = learner.config.learner.base.base.base.public.batch_size
    while time.monotonic() < deadline:
        with profiler.measure("collate"):
            batch = collect_real_pilot_episode(
                learner,
                episode_number=state["games"],
                root_seed=seed,
                worker_id=0,
                epsilon=0.01,
                segment_profiler=profiler,
            )
        state["games"] += 1
        state["decisions"] += batch.decisions
        state["transitions"] += len(batch.transitions)
        if batch.trajectories is not None and len(batch.transitions) > batch_size:
            state["skipped"] += 1
            continue
        before_samples = learner.samples_consumed
        before_steps = learner.eligible_updates
        train_pilot_batch(learner, batch)
        state["samples"] += learner.samples_consumed - before_samples
        state["steps"] += learner.eligible_updates - before_steps


def _strict_full_reload(formal, checkpoint: Path) -> bool:
    restored, _resolved = create_pilot_learner(formal)
    restored.load_checkpoint(checkpoint)
    return True


def _run_full(protocol, formal, seed: int, checkpoint: Path):
    learner, _resolved = create_pilot_learner(formal)
    profiler = SegmentProfiler()
    h3 = learner.base.base.base
    belief_model = learner.base.base.belief_model
    patches = [
        patch.object(
            learner.model,
            "forward_observation",
            profiler.wrap(
                "public_model_forward", learner.model.forward_observation
            ),
        ),
        patch.object(
            h3,
            "train_batch",
            profiler.wrap("oracle_forward_backward", h3.train_batch),
        ),
        patch.object(
            Episode,
            "label_strategy_auxiliary",
            profiler.wrap(
                "strategy_features", Episode.label_strategy_auxiliary
            ),
        ),
    ]
    if learner.model.style_encoder is not None:
        patches.append(patch.object(
            learner.model.style_encoder,
            "forward",
            profiler.wrap("style", learner.model.style_encoder.forward),
        ))
    belief_total = [0.0]
    belief_logits = [0.0]
    if belief_model is not None:
        original_forward = belief_model.forward
        original_logits = belief_model._forward_logits

        def timed_forward(*args, **kwargs):
            started = time.perf_counter()
            result = original_forward(*args, **kwargs)
            belief_total[0] += time.perf_counter() - started
            return result

        def timed_logits(*args, **kwargs):
            started = time.perf_counter()
            result = original_logits(*args, **kwargs)
            belief_logits[0] += time.perf_counter() - started
            return result

        patches.extend([
            patch.object(belief_model, "forward", timed_forward),
            patch.object(belief_model, "_forward_logits", timed_logits),
        ])
    stack = contextlib.ExitStack()
    for item in patches:
        stack.enter_context(item)
    state = {
        "games": 0,
        "decisions": 0,
        "transitions": 0,
        "samples": 0,
        "steps": 0,
        "skipped": 0,
    }
    try:
        _run_full_until(
            learner,
            seed=seed,
            deadline=time.monotonic() + protocol.warmup_seconds,
            state=state,
            profiler=profiler,
        )
        before = _full_counter(state)
        profiler.values = {name: 0.0 for name in P3_SEGMENTS}
        belief_total[0] = 0.0
        belief_logits[0] = 0.0
        torch.cuda.reset_peak_memory_stats(learner.device)
        parameter_before = _model_digest(learner.model)
        started = time.monotonic()
        _run_full_until(
            learner,
            seed=seed,
            deadline=started + protocol.measurement_seconds,
            state=state,
            profiler=profiler,
        )
        elapsed = time.monotonic() - started
        after = _full_counter(state)
        parameter_changed = parameter_before != _model_digest(learner.model)
        profiler.values["belief_logits"] = belief_logits[0]
        profiler.values["exact_dp"] = max(0.0, belief_total[0] - belief_logits[0])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_started = time.perf_counter()
        learner.save_checkpoint(checkpoint)
        profiler.values["checkpoint"] = time.perf_counter() - checkpoint_started
        cpu_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        vram = int(torch.cuda.max_memory_allocated())
        strict_reload = _strict_full_reload(formal, checkpoint)
        return {
            "before": before,
            "after": after,
            "elapsed": elapsed,
            "segments": dict(profiler.values),
            "parameter_changed": parameter_changed,
            "checkpoint_reload": strict_reload,
            "policy_lag": 0,
            "actor_blocked": 0.0,
            "data_wait": 0.0,
            "cpu_ram": cpu_ram,
            "shared_memory": 0,
            "vram": vram,
            "active": 0,
            "in_flight": 0,
            "pending": 0,
            "shutdown": 0.0,
            "skipped": state["skipped"],
        }
    finally:
        stack.close()


def main() -> None:
    args = _parser().parse_args()
    protocol = _load_protocol(args.protocol)
    if git_sha() != protocol.source_git_sha:
        raise SystemExit("P3 source SHA does not match the frozen protocol")
    source_tree = _attest_clean_source(protocol.source_git_sha)
    if source_tree != protocol.source_tree:
        raise SystemExit("P3 source tree does not match the frozen protocol")
    _container_id, image_digest = _attest_current_container(
        _DOCKER_SOCKET, _HOST_PROC
    )
    if image_digest != protocol.image_digest:
        raise SystemExit("P3 container image does not match the frozen protocol")
    if not torch.cuda.is_available():
        raise SystemExit("P3 runtime benchmark requires CUDA")

    full = args.topology == "full_hybrid_single_process"
    formal = load_formal_config(args.full_config if full else args.base_config)
    identity = formal.identity_dict()
    expected_config = (
        protocol.full_config_hash if full else protocol.base_config_hash
    )
    if identity["config_sha256"] != expected_config:
        raise SystemExit("P3 formal config does not match the frozen protocol")
    learner, resolved = create_pilot_learner(formal)
    del learner
    expected_model = protocol.model_identity_hashes[
        "full_hybrid" if full else "base"
    ]
    if resolved.model.stable_hash() != expected_model:
        raise SystemExit("P3 model identity does not match the frozen protocol")

    seed = protocol.seeds[args.repeat]
    result = (
        _run_full(protocol, formal, seed, args.checkpoint)
        if full
        else _run_base(
            protocol, formal, args.topology, seed, args.checkpoint
        )
    )
    elapsed = float(result["elapsed"])
    rates = {
        f"{name}_per_second": (
            result["after"][name] - result["before"][name]
        ) / elapsed
        for name in result["before"]
    }
    record = {
        "schema": P3_RUNTIME_SCHEMA,
        "protocol_hash": protocol.stable_hash(),
        "topology": args.topology,
        "repeat": args.repeat,
        "seed": seed,
        "source_git_sha": protocol.source_git_sha,
        "source_tree": protocol.source_tree,
        "image_digest": protocol.image_digest,
        "config_hash": expected_config,
        "model_identity_hash": expected_model,
        "measurement_seconds": elapsed,
        "counters_before": result["before"],
        "counters_after": result["after"],
        "rates": rates,
        "segments_seconds": result["segments"],
        "parameter_update_observed": result["parameter_changed"],
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "saved": True,
            "strict_reload": result["checkpoint_reload"],
        },
        "policy_lag_max": result["policy_lag"],
        "actor_blocked_ratio": result["actor_blocked"],
        "learner_data_wait_ratio": result["data_wait"],
        "cpu_ram_bytes": result["cpu_ram"],
        "shared_memory_bytes": result["shared_memory"],
        "vram_bytes": result["vram"],
        "active_slots": result["active"],
        "in_flight": result["in_flight"],
        "pending": result["pending"],
        "shutdown_seconds": result["shutdown"],
        "skipped_long_cooperation_episodes": result["skipped"],
    }
    json.dumps(record, allow_nan=False)
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in record.values()
    ):
        raise FloatingPointError("P3 record contains non-finite metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
