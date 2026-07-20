"""Low-overhead shared metrics for the opt-in legacy training profiler."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


POSITIONS = ("landlord", "landlord_up", "landlord_down")
MAX_LEGAL_ACTIONS = 20000

ACTOR_COUNTERS = (
    "decisions",
    "games",
    "transitions",
    "single_legal_actions",
    "env_step_ns",
    "legal_actions_ns",
    "observation_ns",
    "inference_ns",
    "rollout_write_ns",
    "free_queue_wait_ns",
    "full_queue_put_ns",
)

LEARNER_COUNTERS = (
    "updates",
    "profile_samples",
    "frames",
    "batch_wait_ns",
    "batch_assembly_ns",
    "pin_memory_ns",
    "h2d_ns",
    "forward_ns",
    "backward_ns",
    "grad_clip_ns",
    "optimizer_ns",
    "log_write_ns",
    "log_writes",
    "snapshot_publish_ns",
    "snapshot_publishes",
    "snapshot_skips",
)


class LegacyMetricStore:
    """Multiprocess counters with actor-side batching and an action histogram."""

    def __init__(self, mp_context) -> None:
        self._lock = mp_context.Lock()
        self._generation = mp_context.Value("q", 0, lock=False)
        self._actor = mp_context.Array("q", len(ACTOR_COUNTERS), lock=False)
        self._learner = mp_context.Array("q", len(LEARNER_COUNTERS), lock=False)
        self._role_wait_ns = mp_context.Array("q", len(POSITIONS), lock=False)
        self._role_updates = mp_context.Array("q", len(POSITIONS), lock=False)
        self._role_mean_lag_sum = mp_context.Array(
            "d", len(POSITIONS), lock=False
        )
        self._role_max_lag = mp_context.Array("q", len(POSITIONS), lock=False)
        self._legal_hist = mp_context.Array(
            "q", MAX_LEGAL_ACTIONS + 1, lock=False
        )
        self._legal_max = mp_context.Value("q", 0, lock=False)
        self._started_ns = mp_context.Value("q", time.perf_counter_ns(), lock=False)

    @property
    def generation(self) -> int:
        return int(self._generation.value)

    def reset(self) -> None:
        """Atomically begin a new measurement window after benchmark warmup."""
        with self._lock:
            for values in (
                self._actor,
                self._learner,
                self._role_wait_ns,
                self._role_updates,
                self._role_mean_lag_sum,
                self._role_max_lag,
                self._legal_hist,
            ):
                for index in range(len(values)):
                    values[index] = 0
            self._legal_max.value = 0
            self._started_ns.value = time.perf_counter_ns()
            self._generation.value += 1

    def add_actor(
        self,
        counters: dict[str, int],
        legal_hist: dict[int, int],
        *,
        generation: int | None = None,
    ) -> bool:
        with self._lock:
            if generation is not None and generation != self._generation.value:
                return False
            for name, value in counters.items():
                self._actor[ACTOR_COUNTERS.index(name)] += int(value)
            for count, frequency in legal_hist.items():
                bucket = min(int(count), MAX_LEGAL_ACTIONS)
                self._legal_hist[bucket] += int(frequency)
                self._legal_max.value = max(self._legal_max.value, int(count))
            return True

    def add_learner(
        self,
        counters: dict[str, int],
        *,
        position: str | None = None,
        queue_wait_ns: int = 0,
        mean_policy_lag: float = 0.0,
        max_policy_lag: int = 0,
    ) -> None:
        with self._lock:
            for name, value in counters.items():
                self._learner[LEARNER_COUNTERS.index(name)] += int(value)
            if position is not None:
                role = POSITIONS.index(position)
                self._role_wait_ns[role] += int(queue_wait_ns)
                self._role_updates[role] += 1
                self._role_mean_lag_sum[role] += float(mean_policy_lag)
                self._role_max_lag[role] = max(
                    self._role_max_lag[role], int(max_policy_lag)
                )

    @staticmethod
    def _percentile(histogram: list[int], percentile: float) -> int | None:
        total = sum(histogram)
        if total == 0:
            return None
        target = max(1, math.ceil(total * percentile))
        seen = 0
        for value, count in enumerate(histogram):
            seen += count
            if seen >= target:
                return value
        return len(histogram) - 1

    def snapshot(self) -> dict:
        with self._lock:
            actor = {
                name: int(self._actor[index])
                for index, name in enumerate(ACTOR_COUNTERS)
            }
            learner = {
                name: int(self._learner[index])
                for index, name in enumerate(LEARNER_COUNTERS)
            }
            role_wait = {
                position: int(self._role_wait_ns[index])
                for index, position in enumerate(POSITIONS)
            }
            role_updates = {
                position: int(self._role_updates[index])
                for index, position in enumerate(POSITIONS)
            }
            role_mean_lag_sum = {
                position: float(self._role_mean_lag_sum[index])
                for index, position in enumerate(POSITIONS)
            }
            role_max_lag = {
                position: int(self._role_max_lag[index])
                for index, position in enumerate(POSITIONS)
            }
            histogram = [int(value) for value in self._legal_hist]
            legal_max = int(self._legal_max.value)
            elapsed_ns = max(1, time.perf_counter_ns() - int(self._started_ns.value))

        elapsed_s = elapsed_ns / 1e9
        decisions = actor["decisions"]
        updates = learner["updates"]
        frames = learner["frames"]

        def mean_ms(total_name: str, count: int, source: dict) -> float | None:
            return (
                source[total_name] / count / 1e6
                if count > 0 else None
            )

        return {
            "schema_version": "legacy-training-metrics-v2",
            "measurement_seconds": elapsed_s,
            "rates": {
                "frames_per_second": frames / elapsed_s,
                "learner_updates_per_second": updates / elapsed_s,
                "games_per_second": actor["games"] / elapsed_s,
                "decisions_per_second": decisions / elapsed_s,
                "transitions_per_second": actor["transitions"] / elapsed_s,
            },
            "counts": {"actor": actor, "learner": learner},
            "actor_timing_mean_ms": {
                name.removesuffix("_ns"): mean_ms(name, decisions, actor)
                for name in ACTOR_COUNTERS if name.endswith("_ns")
            },
            "learner_timing_mean_ms": {
                name.removesuffix("_ns"): mean_ms(
                    name,
                    (
                        updates if name in {
                            "batch_wait_ns", "batch_assembly_ns", "pin_memory_ns"
                        }
                        else learner["log_writes"] if name == "log_write_ns"
                        else (
                            learner["snapshot_publishes"]
                            + learner["snapshot_skips"]
                        ) if name == "snapshot_publish_ns"
                        else learner["profile_samples"]
                    ),
                    learner,
                )
                for name in LEARNER_COUNTERS if name.endswith("_ns")
            },
            "queue_starvation_mean_ms": {
                position: (
                    role_wait[position] / role_updates[position] / 1e6
                    if role_updates[position] else None
                )
                for position in POSITIONS
            },
            "policy_lag": {
                position: {
                    "mean_updates": (
                        role_mean_lag_sum[position] / role_updates[position]
                        if role_updates[position] else None
                    ),
                    "max_updates": role_max_lag[position],
                }
                for position in POSITIONS
            },
            "legal_actions": {
                "single_ratio": (
                    actor["single_legal_actions"] / decisions if decisions else None
                ),
                "p50": self._percentile(histogram, 0.50),
                "p95": self._percentile(histogram, 0.95),
                "max": legal_max if decisions else None,
                "overflow_count": histogram[-1],
            },
        }


class ActorMetricRecorder:
    """Actor-local accumulator that avoids a shared lock on every decision."""

    def __init__(self, store: LegacyMetricStore | None, flush_every: int = 128) -> None:
        self.store = store
        self.flush_every = flush_every
        self.generation = store.generation if store is not None else 0
        self.counters = {name: 0 for name in ACTOR_COUNTERS}
        self.legal_hist: dict[int, int] = {}

    def _check_generation(self) -> None:
        if self.store is not None and self.generation != self.store.generation:
            self.counters = {name: 0 for name in ACTOR_COUNTERS}
            self.legal_hist = {}
            self.generation = self.store.generation

    def add(self, **values: int) -> None:
        if self.store is None:
            return
        self._check_generation()
        for name, value in values.items():
            self.counters[name] += int(value)
        if self.counters["decisions"] >= self.flush_every:
            self.flush()

    def legal_actions(self, count: int) -> None:
        if self.store is None:
            return
        self._check_generation()
        self.legal_hist[count] = self.legal_hist.get(count, 0) + 1

    def flush(self) -> None:
        if self.store is None:
            return
        self._check_generation()
        if any(self.counters.values()) or self.legal_hist:
            self.store.add_actor(
                self.counters,
                self.legal_hist,
                generation=self.generation,
            )
        self.counters = {name: 0 for name in ACTOR_COUNTERS}
        self.legal_hist = {}


def write_metrics(path: str, payload: dict) -> None:
    """Write profiler JSON atomically so interrupted runs are never mistaken for data."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
