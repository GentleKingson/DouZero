#!/usr/bin/env python3
"""Run one committed, checkpoint-enabled P3 matched-runtime repetition."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
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
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_protocol(path: Path) -> P3RuntimeProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema", None) != P3_RUNTIME_SCHEMA:
        raise ValueError("P3 runtime protocol schema mismatch")
    payload["seeds"] = tuple(payload["seeds"])
    return P3RuntimeProtocol(**payload)


def _seed_for_repeat(protocol: P3RuntimeProtocol, repeat: int) -> int:
    if (
        isinstance(repeat, bool)
        or not isinstance(repeat, int)
        or repeat < 0
        or repeat >= protocol.repetitions
    ):
        raise ValueError(
            f"P3 repeat must be in [0, {protocol.repetitions})"
        )
    return protocol.seeds[repeat]


def _module_digest(components: Mapping[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for component, module in sorted(components.items()):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(component.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _learner_digest(learner) -> str:
    h5 = learner.base
    h4 = h5.base
    h3 = h4.base
    components = {"public_model": learner.model}
    for name, module in (
        ("oracle", h3.oracle),
        ("belief", h4.belief_model),
        ("cooperation", h5.cooperation),
    ):
        if module is not None:
            components[name] = module
    return _module_digest(components)


def _live_hardware_identity() -> dict[str, str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "-i",
            str(torch.cuda.current_device()),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(query) != 1 or not query[0].strip():
        raise RuntimeError("P3 could not determine one live NVIDIA driver")
    return {
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "driver": query[0].strip(),
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cpu": platform.machine(),
    }


def _verify_hardware_identity(protocol: P3RuntimeProtocol) -> None:
    observed = _live_hardware_identity()
    for name, value in observed.items():
        if value != getattr(protocol, name):
            raise SystemExit(
                f"P3 live {name} does not match the frozen protocol: "
                f"{value!r} != {getattr(protocol, name)!r}"
            )


class CheckpointCadence:
    """Save at every crossed eligible-update boundary."""

    def __init__(
        self,
        cadence: int,
        initial_updates: int,
        save: Callable[[], None],
    ) -> None:
        if cadence < 1 or initial_updates < 0:
            raise ValueError("P3 checkpoint cadence state is invalid")
        self.cadence = cadence
        self._last_updates = initial_updates
        self._next_update = (initial_updates // cadence + 1) * cadence
        self._save = save
        self.saves = 0
        self.seconds = 0.0

    def observe(self, updates: int) -> None:
        if updates < self._last_updates:
            raise ValueError("P3 checkpoint update counter regressed")
        while updates >= self._next_update:
            started = time.perf_counter()
            self._save()
            self.seconds += time.perf_counter() - started
            self.saves += 1
            self._next_update += self.cadence
        self._last_updates = updates


class SegmentProfiler:
    """Synchronized semantic wall timers; nested segments may overlap."""

    def __init__(
        self,
        *,
        synchronize: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.values = {name: 0.0 for name in P3_SEGMENTS}
        self._synchronize = synchronize or (lambda: None)
        self._clock = clock

    @contextlib.contextmanager
    def measure(self, name: str):
        if name not in self.values:
            raise KeyError(name)
        self._synchronize()
        started = self._clock()
        try:
            yield
        finally:
            self._synchronize()
            self.values[name] += self._clock() - started

    def wrap(self, name: str, function):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)

        return measured

    def time_call(self, accumulator: list[float], function, *args, **kwargs):
        self._synchronize()
        started = self._clock()
        try:
            return function(*args, **kwargs)
        finally:
            self._synchronize()
            accumulator[0] += self._clock() - started


def _episodes_per_cycle(topology: str, actors: int, games: int) -> int:
    if topology == "base_single_process":
        return 4
    if topology not in {"base_async_4x4", "base_async_8x4"}:
        raise ValueError("P3 base topology is unknown")
    return actors * games


def _learner_updates_per_cycle(
    episodes: int, episodes_per_learner_update: int
) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (episodes, episodes_per_learner_update)
    ):
        raise ValueError("P3 learner cadence requires positive integer counts")
    updates, remainder = divmod(episodes, episodes_per_learner_update)
    if remainder:
        raise ValueError("P3 collection cycle is not divisible by learner cadence")
    return updates


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


def _process_rss_bytes(pid: int, proc_root: Path) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError("P3 process PID must be a positive int")
    status = (proc_root / str(pid) / "status").read_text(encoding="utf-8")
    rows = [
        line.split()
        for line in status.splitlines()
        if line.startswith("VmRSS:")
    ]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != "kB":
        raise RuntimeError(f"P3 could not read VmRSS for process {pid}")
    try:
        rss_kib = int(rows[0][1])
    except ValueError as error:
        raise RuntimeError(f"P3 process {pid} VmRSS is invalid") from error
    if rss_kib < 0:
        raise RuntimeError(f"P3 process {pid} VmRSS is negative")
    return rss_kib * 1024


def _aggregate_runtime_rss_bytes(
    trainer,
    *,
    proc_root: Path = Path("/proc"),
    parent_pid: int | None = None,
) -> int:
    """Measure parent plus every live async actor before worker shutdown."""

    pids = [os.getpid() if parent_pid is None else parent_pid]
    if trainer is not None and getattr(trainer, "_runtime_started", False):
        for worker in trainer._workers:
            if worker.pid is None or not worker.is_alive():
                raise RuntimeError("P3 async actor is not live during RSS measurement")
            pids.append(int(worker.pid))
    if len(set(pids)) != len(pids):
        raise RuntimeError("P3 runtime RSS process identities are not unique")
    return sum(_process_rss_bytes(pid, proc_root) for pid in pids)


def _run_runtime_until(
    trainer,
    deadline: float,
    episodes: int,
    episodes_per_learner_update: int,
    checkpoint_cadence: CheckpointCadence,
) -> int:
    updates = _learner_updates_per_cycle(
        episodes, episodes_per_learner_update
    )
    max_lag = 0
    while time.monotonic() < deadline:
        trainer.collect_episodes(episodes)
        before_updates = trainer.stats.optimizer_steps
        trainer.optimize(updates)
        if trainer.stats.optimizer_steps - before_updates != updates:
            raise RuntimeError("P3 runtime learner cadence did not advance exactly")
        checkpoint_cadence.observe(trainer.stats.optimizer_steps)
        max_lag = max(max_lag, trainer.policy_step - trainer._snapshot_step)
    return max_lag


def _strict_runtime_reload(
    trainer_type,
    formal,
    runtime_config,
    checkpoint: Path,
    *,
    episodes: int,
    episodes_per_learner_update: int,
    checkpoint_state: Mapping[str, object],
) -> dict[str, bool]:
    learner, resolved = create_pilot_learner(formal)
    restored = trainer_type(learner, resolved, runtime_config)
    try:
        resumed_state = restored.load_training_checkpoint(checkpoint)
        if resumed_state != dict(checkpoint_state):
            raise RuntimeError("P3 runtime resume sidecar state mismatch")
        parameter_before = _learner_digest(restored.learner)
        updates_before = restored.stats.optimizer_steps
        restored.collect_episodes(episodes)
        restored.optimize(_learner_updates_per_cycle(
            episodes, episodes_per_learner_update
        ))
        boundary = restored.quiesce_cycle_boundary()
        resumed_update = (
            restored.stats.optimizer_steps > updates_before
            and _learner_digest(restored.learner) != parameter_before
        )
        resume_quiesced = all(
            int(boundary[name]) == 0
            for name in (
                "active_slots",
                "in_flight_slots",
                "pending_requests",
            )
        )
        if not resumed_update or not resume_quiesced:
            raise RuntimeError("P3 runtime resume exercise did not complete cleanly")
        return {
            "strict_reload": True,
            "resumed_update": resumed_update,
            "resume_quiesced": resume_quiesced,
        }
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
        actors, games = 1, 1
    else:
        runtime_topology = TOPOLOGY_ASYNC_SINGLE_GPU
        trainer_type = V3AsyncSingleGPUTrainer
        actors = 4 if topology == "base_async_4x4" else 8
        games = 4
    episodes = _episodes_per_cycle(topology, actors, games)
    learner, resolved = create_pilot_learner(formal)
    runtime_config = V3H7RuntimeConfig(
        topology=runtime_topology,
        num_actors=actors,
        games_per_actor=games,
        batch_size=protocol.batch_size,
        replay_capacity=formal.runtime.replay_capacity,
        target_microbatch=4,
        environment_seed=seed,
        environment_seed_derivation=protocol.deal_seed_derivation,
        action_seed=seed + 1,
        max_policy_lag=protocol.max_policy_lag,
    )
    trainer = trainer_type(learner, resolved, runtime_config)
    checkpoint_seconds = 0.0
    shutdown_seconds = 0.0
    checkpoint_state = {
        "p3_protocol_hash": protocol.stable_hash(),
        "topology": topology,
        "seed": seed,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cadence = CheckpointCadence(
        protocol.checkpoint_cadence_updates,
        trainer.stats.optimizer_steps,
        lambda: trainer.save_training_checkpoint(
            str(checkpoint),
            long_running_state=checkpoint_state,
        ),
    )
    try:
        _run_runtime_until(
            trainer,
            time.monotonic() + protocol.warmup_seconds,
            episodes,
            protocol.episodes_per_learner_update,
            cadence,
        )
        trainer.quiesce_cycle_boundary()
        torch.cuda.reset_peak_memory_stats(trainer.device)
        before = _counter_payload(trainer.stats)
        phase_before = trainer.learner.base.base.base.schedule_state()
        parameter_before = _learner_digest(trainer.learner)
        cadence_seconds_before = cadence.seconds
        started = time.monotonic()
        max_lag = _run_runtime_until(
            trainer,
            started + protocol.measurement_seconds,
            episodes,
            protocol.episodes_per_learner_update,
            cadence,
        )
        elapsed = time.monotonic() - started
        boundary = trainer.quiesce_cycle_boundary()
        after = _counter_payload(trainer.stats)
        phase_after = trainer.learner.base.base.base.schedule_state()
        parameter_changed = parameter_before != _learner_digest(trainer.learner)
        shared_memory = _shared_memory_bytes(trainer)
        checkpoint_started = time.perf_counter()
        trainer.save_training_checkpoint(
            str(checkpoint),
            long_running_state=checkpoint_state,
        )
        checkpoint_seconds = (
            cadence.seconds
            - cadence_seconds_before
            + time.perf_counter()
            - checkpoint_started
        )
        cpu_ram = _aggregate_runtime_rss_bytes(trainer)
        vram = int(torch.cuda.max_memory_allocated())
        shutdown_started = time.monotonic()
        trainer.shutdown()
        shutdown_seconds = time.monotonic() - shutdown_started
        trainer = None
        del learner
        del resolved
        _release_cuda_graph()
        resume = _strict_runtime_reload(
            trainer_type,
            formal,
            runtime_config,
            checkpoint,
            episodes=episodes,
            episodes_per_learner_update=(
                protocol.episodes_per_learner_update
            ),
            checkpoint_state=checkpoint_state,
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
            "checkpoint_reload": resume["strict_reload"],
            "resumed_update": resume["resumed_update"],
            "resume_quiesced": resume["resume_quiesced"],
            "training_phase": {
                "before": phase_before.phase,
                "after": phase_after.phase,
                "learner_update_before": phase_before.learner_update,
                "learner_update_after": phase_after.learner_update,
            },
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
            "skipped_incomplete": 0,
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


def _prime_full_hybrid_phase(learner, target_update: int):
    """Advance only schedule counters to the frozen guided-phase boundary."""

    h5 = learner.base
    h4 = h5.base
    h3 = h4.base
    if (
        target_update != h3.config.schedule.warmup_updates
        or h3.config.schedule.at(target_update).phase != "guided"
    ):
        raise ValueError("P3 full-hybrid guided phase boundary is invalid")
    counters = (
        learner.eligible_updates,
        h5.eligible_updates,
        h4.eligible_updates,
        h3.learner_updates,
        learner.samples_consumed,
        h5.samples_consumed,
        h4.samples_consumed,
        h3.samples_consumed,
    )
    if any(value != 0 for value in counters):
        raise RuntimeError("P3 full-hybrid phase priming requires a fresh learner")
    if any(
        statistics.steps != 0
        for statistics in (
            learner.statistics,
            h5.statistics,
            h4.statistics,
            h3.statistics,
        )
    ):
        raise RuntimeError("P3 full-hybrid phase statistics are not fresh")

    h3.learner_updates = target_update
    h3.statistics.steps = target_update
    h4.eligible_updates = target_update
    h4.statistics.steps = target_update
    h4.statistics.base_updates = target_update
    h5.eligible_updates = target_update
    h5.statistics.steps = target_update
    learner.eligible_updates = target_update
    learner.statistics.steps = target_update
    state = h3.schedule_state()
    if state.phase != "guided" or state.learner_update != target_update:
        raise RuntimeError("P3 full-hybrid phase priming did not take effect")
    return state


def _run_full_episode(
    learner,
    *,
    seed: int,
    state,
    profiler,
    checkpoint_cadence: CheckpointCadence,
) -> bool:
    batch_size = learner.config.learner.base.base.base.public.batch_size
    with profiler.measure("collate"):
        batch = collect_real_pilot_episode(
            learner,
            episode_number=state["games"],
            root_seed=seed,
            worker_id=0,
            epsilon=0.01,
            segment_profiler=profiler,
            skip_forced_actions=True,
        )
    state["games"] += 1
    state["decisions"] += batch.decisions
    state["transitions"] += len(batch.transitions)
    if batch.cooperation_skip_reason is not None:
        if batch.cooperation_skip_reason != "missing_nonforced_farmer_role":
            raise RuntimeError("P3 encountered an unknown cooperation skip reason")
        state["skipped_incomplete"] += 1
        return False
    if batch.trajectories is not None and len(batch.transitions) > batch_size:
        state["skipped"] += 1
        return False
    before_samples = learner.samples_consumed
    before_steps = learner.eligible_updates
    train_pilot_batch(learner, batch)
    sample_delta = learner.samples_consumed - before_samples
    step_delta = learner.eligible_updates - before_steps
    if step_delta != 1 or sample_delta < 1:
        raise RuntimeError("P3 full-hybrid batch did not advance exactly once")
    state["samples"] += sample_delta
    state["steps"] += step_delta
    checkpoint_cadence.observe(learner.eligible_updates)
    return True


def _run_full_until(
    learner,
    *,
    seed: int,
    deadline: float,
    state,
    profiler,
    checkpoint_cadence: CheckpointCadence,
) -> None:
    while time.monotonic() < deadline:
        _run_full_episode(
            learner,
            seed=seed,
            state=state,
            profiler=profiler,
            checkpoint_cadence=checkpoint_cadence,
        )


def _strict_full_reload(
    protocol,
    formal,
    checkpoint: Path,
    *,
    seed: int,
    episode_number: int,
) -> dict[str, bool]:
    restored, _resolved = create_pilot_learner(formal)
    restored.load_checkpoint(checkpoint)
    h3 = restored.base.base.base
    if h3.schedule_state().phase != protocol.full_hybrid_phase:
        raise RuntimeError("P3 restored full-hybrid phase drifted")
    parameter_before = _learner_digest(restored)
    updates_before = restored.eligible_updates
    state = {
        "games": episode_number,
        "decisions": 0,
        "transitions": 0,
        "samples": 0,
        "steps": 0,
        "skipped": 0,
        "skipped_incomplete": 0,
    }
    cadence = CheckpointCadence(
        max(protocol.checkpoint_cadence_updates, 1000000),
        restored.eligible_updates,
        lambda: None,
    )
    profiler = SegmentProfiler(
        synchronize=lambda: torch.cuda.synchronize(restored.device)
    )
    for _ in range(128):
        if _run_full_episode(
            restored,
            seed=seed,
            state=state,
            profiler=profiler,
            checkpoint_cadence=cadence,
        ):
            break
    else:
        raise RuntimeError("P3 full-hybrid resume found no trainable episode")
    resumed_update = (
        restored.eligible_updates == updates_before + 1
        and _learner_digest(restored) != parameter_before
    )
    if not resumed_update:
        raise RuntimeError("P3 full-hybrid resume did not update parameters")
    return {
        "strict_reload": True,
        "resumed_update": True,
        "resume_quiesced": True,
    }


def _measure_full(protocol, formal, seed: int, checkpoint: Path):
    learner, _resolved = create_pilot_learner(formal)
    _prime_full_hybrid_phase(
        learner, protocol.full_hybrid_phase_update
    )
    profiler = SegmentProfiler(
        synchronize=lambda: torch.cuda.synchronize(learner.device)
    )
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
            return profiler.time_call(
                belief_total, original_forward, *args, **kwargs
            )

        def timed_logits(*args, **kwargs):
            return profiler.time_call(
                belief_logits, original_logits, *args, **kwargs
            )

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
        "skipped_incomplete": 0,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cadence = CheckpointCadence(
        protocol.checkpoint_cadence_updates,
        learner.eligible_updates,
        lambda: learner.save_checkpoint(checkpoint),
    )
    try:
        _run_full_until(
            learner,
            seed=seed,
            deadline=time.monotonic() + protocol.warmup_seconds,
            state=state,
            profiler=profiler,
            checkpoint_cadence=cadence,
        )
        before = _full_counter(state)
        phase_before = h3.schedule_state()
        profiler.values = {name: 0.0 for name in P3_SEGMENTS}
        belief_total[0] = 0.0
        belief_logits[0] = 0.0
        torch.cuda.reset_peak_memory_stats(learner.device)
        parameter_before = _learner_digest(learner)
        cadence_seconds_before = cadence.seconds
        started = time.monotonic()
        _run_full_until(
            learner,
            seed=seed,
            deadline=started + protocol.measurement_seconds,
            state=state,
            profiler=profiler,
            checkpoint_cadence=cadence,
        )
        elapsed = time.monotonic() - started
        after = _full_counter(state)
        phase_after = h3.schedule_state()
        parameter_changed = parameter_before != _learner_digest(learner)
        profiler.values["belief_logits"] = belief_logits[0]
        profiler.values["exact_dp"] = max(0.0, belief_total[0] - belief_logits[0])
        checkpoint_started = time.perf_counter()
        learner.save_checkpoint(checkpoint)
        profiler.values["checkpoint"] = (
            cadence.seconds
            - cadence_seconds_before
            + time.perf_counter()
            - checkpoint_started
        )
        cpu_ram = _aggregate_runtime_rss_bytes(None)
        vram = int(torch.cuda.max_memory_allocated())
        return {
            "before": before,
            "after": after,
            "elapsed": elapsed,
            "segments": dict(profiler.values),
            "parameter_changed": parameter_changed,
            "training_phase": {
                "before": phase_before.phase,
                "after": phase_after.phase,
                "learner_update_before": phase_before.learner_update,
                "learner_update_after": phase_after.learner_update,
            },
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
            "skipped_incomplete": state["skipped_incomplete"],
            "_resume_episode_number": state["games"],
        }
    finally:
        stack.close()


def _release_cuda_graph() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _run_full(protocol, formal, seed: int, checkpoint: Path):
    result = _measure_full(protocol, formal, seed, checkpoint)
    resume_episode_number = result.pop("_resume_episode_number")
    cleanup_started = time.monotonic()
    _release_cuda_graph()
    result["shutdown"] = time.monotonic() - cleanup_started
    try:
        resume = _strict_full_reload(
            protocol,
            formal,
            checkpoint,
            seed=seed,
            episode_number=resume_episode_number,
        )
    finally:
        _release_cuda_graph()
    result["checkpoint_reload"] = resume["strict_reload"]
    result["resumed_update"] = resume["resumed_update"]
    result["resume_quiesced"] = resume["resume_quiesced"]
    return result


def main() -> None:
    args = _parser().parse_args()
    protocol = _load_protocol(args.protocol)
    try:
        seed = _seed_for_repeat(protocol, args.repeat)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    _verify_hardware_identity(protocol)

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
        "deal_seed_derivation": protocol.deal_seed_derivation,
        "measurement_seconds": elapsed,
        "counters_before": result["before"],
        "counters_after": result["after"],
        "rates": rates,
        "segments_seconds": result["segments"],
        "parameter_update_observed": result["parameter_changed"],
        "training_phase": result["training_phase"],
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "saved": True,
            "strict_reload": result["checkpoint_reload"],
            "resumed_update": result["resumed_update"],
            "resume_quiesced": result["resume_quiesced"],
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
        "skipped_incomplete_cooperation_episodes": result[
            "skipped_incomplete"
        ],
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
