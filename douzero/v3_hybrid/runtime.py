"""H7 bounded single-GPU runtime over the existing async V2 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import torch

from douzero.training.async_single_gpu import (
    AsyncRequestCoordinator,
    PendingRequestScheduler,
    PinnedObservationBatchStager,
    SharedReplaySlots,
    async_actor_main,
)

from .adaptive_dmc import ADMC_DISABLED
from .integration_config import V3H6ResolvedConfig
from .replay import V3ReplayTransition
from .support_matrix import (
    RULESET_LEGACY,
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
    validate_capability_support,
)
from .training.h6_learner import V3H6Learner

V3_H7_RUNTIME_VERSION = "v3-hybrid-h7-async-runtime-v1"
V3_H7_CHECKPOINT_FORMAT = "v3-hybrid-h7-runtime-checkpoint-v1"
V3_H7_REQUEST_PROTOCOL = "v2-shared-slots-v3-dmc-q-v1"
V3_H7_REPLAY_PROTOCOL = "v3-public-selected-action-q-old-v1"


def _stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H7RuntimeConfig:
    num_actors: int = 4
    games_per_actor: int = 4
    batch_size: int = 32
    replay_capacity: int = 4096
    max_actions: int = 4096
    target_microbatch: int = 4
    microbatch_delay_ms: float = 2.0
    request_timeout_seconds: float = 30.0
    max_policy_lag: int = 128
    environment_seed: int = 1
    action_seed: int = 2
    epsilon: float = 0.01
    max_steps_per_episode: int = 1000
    snapshot_semantics: str = "game-boundary-quiescent-copy-v1"
    request_protocol: str = V3_H7_REQUEST_PROTOCOL
    replay_protocol: str = V3_H7_REPLAY_PROTOCOL

    def __post_init__(self) -> None:
        positive = (
            "num_actors", "games_per_actor", "batch_size", "replay_capacity",
            "max_actions", "target_microbatch", "max_policy_lag",
            "max_steps_per_episode",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"H7 runtime {name} must be a positive int")
        for name in ("environment_seed", "action_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"H7 runtime {name} must be a non-negative int")
        for name in ("microbatch_delay_ms", "request_timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"H7 runtime {name} must be positive and finite")
        if not math.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("H7 runtime epsilon must be in [0, 1]")
        if self.request_protocol != V3_H7_REQUEST_PROTOCOL:
            raise ValueError("unknown H7 request protocol")
        if self.replay_protocol != V3_H7_REPLAY_PROTOCOL:
            raise ValueError("unknown H7 replay protocol")

    def identity(self) -> dict[str, object]:
        return {"version": V3_H7_RUNTIME_VERSION, **asdict(self)}

    def stable_hash(self) -> str:
        return _stable_hash(self.identity())


@dataclass
class V3H7RuntimeStats:
    games_collected: int = 0
    episodes_completed: int = 0
    transitions_collected: int = 0
    decisions_collected: int = 0
    optimizer_steps: int = 0
    episodes_per_team: dict[str, int] = field(
        default_factory=lambda: {"landlord": 0, "farmer": 0}
    )


class V3AsyncSingleGPUTrainer:
    """Base V3+ADMC async trainer; later privileged capabilities fail closed."""

    def __init__(
        self,
        learner: V3H6Learner,
        resolved_config: V3H6ResolvedConfig,
        runtime_config: V3H7RuntimeConfig,
    ) -> None:
        if not isinstance(learner, V3H6Learner):
            raise TypeError("H7 runtime requires V3H6Learner")
        if learner.config != resolved_config:
            raise ValueError("H7 learner and resolved config disagree")
        topology = resolved_config.learner.topology
        features = resolved_config.learner.features
        if topology.topology != TOPOLOGY_SINGLE_PROCESS:
            raise ValueError(
                "H7 runtime requires a validated single_process learner; "
                "the outer runtime owns async topology identity"
            )
        if topology.ruleset != RULESET_LEGACY:
            raise NotImplementedError(
                "H7 async runtime currently supports legacy card-play rules only"
            )
        enabled = set(features.enabled_capabilities())
        unsupported = enabled - {"role_model", "adaptive_dmc", "public_export"}
        if unsupported:
            raise NotImplementedError(
                "H7 async runtime rejects unsupported capabilities before worker "
                f"startup: {sorted(unsupported)}"
            )
        for capability in enabled:
            validate_capability_support(
                capability,
                topology=TOPOLOGY_ASYNC_SINGLE_GPU,
                ruleset=topology.ruleset,
                checkpoint_resume=topology.checkpoint_resume,
                export=topology.export,
                deployment=topology.deployment,
                search=False,
            )
        if not features.adaptive_dmc:
            raise ValueError("H7 async replay requires Adaptive DMC q_old provenance")
        if (
            resolved_config.learner.base.base.base.public.adaptive_dmc.mode
            == ADMC_DISABLED
        ):
            raise ValueError("H7 async runtime cannot use disabled Adaptive DMC")
        if learner.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("H7 async runtime requires CUDA and never falls back")
        self.learner = learner
        self.model = learner.model
        self.resolved_config = resolved_config
        self.config = runtime_config
        self.device = learner.device
        self.stats = V3H7RuntimeStats()
        self.buffer: deque[V3ReplayTransition] = deque(
            maxlen=runtime_config.replay_capacity
        )
        self._rng = random.Random(runtime_config.action_seed)
        self._runtime_started = False
        self._snapshot_step = learner.policy_version
        self._reset_metrics()
        self.inference_model = copy.deepcopy(self.model).to(self.device).eval()
        self.runtime_identity = {
            "runtime": runtime_config.identity(),
            "runtime_hash": runtime_config.stable_hash(),
            "training_hash": learner.compatibility_hash,
            "model_hash": self.model.config.stable_hash(),
            "ruleset": learner.ruleset.identity(),
        }
        self.runtime_hash = _stable_hash(self.runtime_identity)

    @property
    def policy_step(self) -> int:
        return int(self.learner.policy_version)

    @property
    def policy_version(self) -> str:
        return f"v3_hybrid:{self.model.config.stable_hash()[:16]}"

    def _reset_metrics(self) -> None:
        self._segments = {
            name: 0.0 for name in (
                "claim_wait", "slot_read", "collate", "h2d", "forward",
                "d2h", "publish", "replay_drain", "learner", "data_wait",
            )
        }
        self._requests = 0
        self._actions = 0
        self._microbatches = 0
        self._batch_histogram: dict[str, int] = {}
        self._bucket_histogram: dict[str, int] = {}
        self._queue_latencies_ms: list[float] = []

    @staticmethod
    def _increment(histogram: dict[str, int], value: object) -> None:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1

    def _start_runtime(self) -> None:
        import multiprocessing as mp
        from douzero.training.decision_policy import DecisionConfig

        context = mp.get_context("spawn")
        cfg = self.config
        self._tasks = context.Queue()
        self._events = context.Queue()
        self._policy_step = context.Value("q", self.policy_step, lock=True)
        slots = max(2, cfg.num_actors * cfg.games_per_actor)
        self._coordinator = AsyncRequestCoordinator(
            self.model.schema,
            num_slots=slots,
            max_actions=cfg.max_actions,
            output_width=6,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )
        self._replay_slots = SharedReplaySlots(
            self.model.schema,
            num_slots=max(slots * 2, min(cfg.batch_size * 2, 64)),
            max_actions=cfg.max_actions,
            v3_provenance=True,
        )
        self._scheduler = PendingRequestScheduler(
            max_batch_size=slots,
            target_batch_size=cfg.target_microbatch,
            max_delay_seconds=cfg.microbatch_delay_ms / 1000.0,
        )
        self._stagers: dict[int, PinnedObservationBatchStager] = {}
        self._workers = []
        for actor_id in range(cfg.num_actors):
            process = context.Process(
                target=async_actor_main,
                args=(
                    actor_id, self._tasks, self._events,
                    self._coordinator, self._replay_slots,
                ),
                kwargs={
                    "environment_seed": cfg.environment_seed,
                    "action_rng_seed": cfg.action_seed,
                    "epsilon": cfg.epsilon,
                    "max_steps": cfg.max_steps_per_episode,
                    "decision_config": DecisionConfig(),
                    "ruleset": None,
                    "feature_schema_hash": self.model.schema.stable_hash(),
                    "policy_version": self.policy_version,
                    "policy_step": self._policy_step,
                    "games_per_actor": cfg.games_per_actor,
                    "runtime_kind": "v3_hybrid",
                },
                name=f"douzero-v3-actor-{actor_id}",
            )
            process.start()
            self._workers.append(process)
        self._runtime_started = True

    def _service_requests(self, wait_seconds: float = 0.001) -> int:
        started = time.perf_counter()
        requests = self._coordinator.claim_ready(
            max_items=self.config.num_actors * self.config.games_per_actor,
            wait_seconds=wait_seconds,
        )
        self._segments["claim_wait"] += time.perf_counter() - started
        for request in requests:
            if request.policy_snapshot != self._snapshot_step:
                raise RuntimeError("H7 request references an unpublished snapshot")
        self._scheduler.add(requests)
        scheduled = self._scheduler.pop_ready()
        if scheduled is None:
            return 0
        (_snapshot, bucket), group = scheduled
        capacity = (
            int(bucket)
            if isinstance(bucket, int)
            else min(
                self.config.max_actions,
                1 << (max(row.action_count for row in group) - 1).bit_length(),
            )
        )
        stager = self._stagers.get(capacity)
        if stager is None:
            stager = PinnedObservationBatchStager(
                self._coordinator.slots,
                max_batch_size=self.config.num_actors * self.config.games_per_actor,
                action_capacity=capacity,
            )
            self._stagers[capacity] = stager
        started = time.perf_counter()
        size = stager.gather_slots([row.slot_id for row in group])
        self._segments["slot_read"] += time.perf_counter() - started
        started = time.perf_counter()
        batch = stager.batch_view(size, self.model.schema.stable_hash())
        self._segments["collate"] += time.perf_counter() - started
        started = time.perf_counter()
        batch.to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)
        self._segments["h2d"] += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.inference_model.forward_batched(
                batch.state_card_vectors,
                batch.state_context_flat,
                batch.context_card_vectors,
                batch.context_flat,
                batch.history_tokens,
                batch.history_key_padding_mask,
                batch.action_features,
                batch.action_mask,
                batch.acting_role,
            )
            packed = torch.stack((
                output.win_logit.squeeze(-1),
                output.score_if_win.squeeze(-1),
                output.score_if_loss.squeeze(-1),
                output.p_win.squeeze(-1),
                output.score_mean.squeeze(-1),
                output.dmc_q.squeeze(-1),
            ), dim=-1).float().contiguous()
        torch.cuda.synchronize(self.device)
        self._segments["forward"] += time.perf_counter() - started
        started = time.perf_counter()
        packed_cpu = stager.stage_outputs(packed)
        torch.cuda.synchronize(self.device)
        self._segments["d2h"] += time.perf_counter() - started
        started = time.perf_counter()
        for row_index, request in enumerate(group):
            count = request.action_count
            self._coordinator.slots.output_values[
                request.slot_id, :count
            ].copy_(packed_cpu[row_index, :count])
            self._coordinator.complete(request.slot_id)
            self._queue_latencies_ms.append(
                (time.monotonic_ns() - request.submitted_ns) / 1_000_000.0
            )
        self._segments["publish"] += time.perf_counter() - started
        self._requests += len(group)
        self._actions += sum(row.action_count for row in group)
        self._microbatches += 1
        self._increment(self._batch_histogram, len(group))
        self._increment(self._bucket_histogram, bucket)
        return len(group)

    def _drain_replay(self) -> int:
        started = time.perf_counter()
        rows = self._replay_slots.read_ready_v3(
            feature_schema_hash=self.model.schema.stable_hash(),
            target_transform=self.model.config.dmc_target_transform,
            ruleset_identity=self.learner.ruleset.identity(),
        )
        self.buffer.extend(rows)
        self._segments["replay_drain"] += time.perf_counter() - started
        return len(rows)

    def _publish_snapshot(self) -> None:
        if self.policy_step - self._snapshot_step > self.config.max_policy_lag:
            raise RuntimeError("H7 policy lag exceeded its configured bound")
        if self._runtime_started:
            self._coordinator.quiesce()
        self.inference_model.load_state_dict(self.model.state_dict(), strict=True)
        self.inference_model.eval()
        torch.cuda.synchronize(self.device)
        if self._runtime_started:
            with self._policy_step.get_lock():
                self._policy_step.value = self.policy_step
        self._snapshot_step = self.policy_step

    def collect_episodes(self, num_episodes: int | None = None) -> None:
        target = int(num_episodes or 0)
        if target < 0:
            raise ValueError("H7 episode target must be non-negative")
        if target == 0:
            return
        if not self._runtime_started:
            self._start_runtime()
        self._publish_snapshot()
        for episode_id in range(
            self.stats.games_collected,
            self.stats.games_collected + target,
        ):
            self._tasks.put(episode_id)
        completed = expected = received = 0
        deadline = time.monotonic() + self.config.request_timeout_seconds * max(1, target)
        while completed < target or received < expected:
            self._coordinator._raise_if_failed()
            self._service_requests()
            received += self._drain_replay()
            for process in self._workers:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"H7 actor {process.name} exited with {process.exitcode}"
                    )
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("H7 async collection timed out")
                continue
            if event[0] == "failed":
                raise RuntimeError(event[2])
            if event[0] == "started":
                self._coordinator.active_games += 1
            elif event[0] == "completed":
                completed += 1
                self._coordinator.active_games -= 1
                count = int(event[3])
                expected += count
                self.stats.games_collected += 1
                self.stats.episodes_completed += 1
                self.stats.transitions_collected += count
                self.stats.decisions_collected += int(event[6])
                team = "landlord" if int(event[4]) == 0 else "farmer"
                self.stats.episodes_per_team[team] += 1

    def optimize(self, num_steps: int) -> None:
        if num_steps < 0:
            raise ValueError("H7 optimizer steps must be non-negative")
        for _ in range(num_steps):
            if len(self.buffer) < self.config.batch_size:
                raise ValueError("H7 replay has fewer rows than batch_size")
            started = time.perf_counter()
            indices = self._rng.sample(range(len(self.buffer)), self.config.batch_size)
            rows = [self.buffer[index] for index in indices]
            self.learner.train_batch(rows)
            self.stats.optimizer_steps += 1
            self._segments["learner"] += time.perf_counter() - started

    def step(self):
        """Run one learner update for the shared long-running controller."""
        if len(self.buffer) < self.config.batch_size:
            return None
        indices = self._rng.sample(range(len(self.buffer)), self.config.batch_size)
        rows = [self.buffer[index] for index in indices]
        started = time.perf_counter()
        metrics = self.learner.train_batch(rows)
        self.stats.optimizer_steps += 1
        self._segments["learner"] += time.perf_counter() - started
        return metrics

    def quiesce_cycle_boundary(self) -> dict[str, object]:
        if self._runtime_started:
            self._drain_replay()
            counts = self._coordinator.quiesce()
        else:
            counts = {"writing": 0, "ready": 0, "running": 0}
        lag = self.policy_step - self._snapshot_step
        if lag > self.config.max_policy_lag:
            raise RuntimeError("H7 policy lag exceeded its configured bound")
        latencies = sorted(self._queue_latencies_ms)

        def percentile(fraction: float) -> float:
            if not latencies:
                return 0.0
            return latencies[int((len(latencies) - 1) * fraction)]

        result = {
            "active_slots": counts["writing"] + counts["ready"] + counts["running"],
            "in_flight_slots": counts["ready"] + counts["running"],
            "pending_requests": self._scheduler.pending_count if self._runtime_started else 0,
            "replay_occupancy": len(self.buffer),
            "requests_per_microbatch": self._requests / max(1, self._microbatches),
            "actions_per_microbatch": self._actions / max(1, self._microbatches),
            "inference_queue_p50_ms": percentile(0.50),
            "inference_queue_p95_ms": percentile(0.95),
            "inference_queue_p99_ms": percentile(0.99),
            "policy_lag": lag,
            "microbatch_size_histogram": dict(self._batch_histogram),
            "inference_bucket_histogram": dict(self._bucket_histogram),
            **{f"{name}_seconds": value for name, value in self._segments.items()},
        }
        self._reset_metrics()
        return result

    def clear_replay(self) -> None:
        self.buffer.clear()

    def save_training_checkpoint(self, path: str, *, long_running_state) -> None:
        with tempfile.TemporaryDirectory(prefix="douzero-h7-save-") as temporary:
            inner_path = Path(temporary) / "h6.pt"
            self.learner.save_checkpoint(inner_path)
            bundle = {
                "format": V3_H7_CHECKPOINT_FORMAT,
                "artifact_access": "privileged_training_only",
                "runtime_identity": self.runtime_identity,
                "runtime_hash": self.runtime_hash,
                "h6_checkpoint": torch.load(
                    inner_path, map_location="cpu", weights_only=True
                ),
                "stats": asdict(self.stats),
                "rng_state": self._rng.getstate(),
                "snapshot_step": self._snapshot_step,
                "long_running_state": dict(long_running_state),
            }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(bundle, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_training_checkpoint(self, path: str | Path):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format", "artifact_access", "runtime_identity", "runtime_hash",
            "h6_checkpoint", "stats", "rng_state", "snapshot_step",
            "long_running_state",
        }
        if not isinstance(bundle, dict) or set(bundle) != expected:
            raise ValueError("H7 checkpoint fields mismatch")
        if bundle["format"] != V3_H7_CHECKPOINT_FORMAT:
            raise ValueError("H7 checkpoint format mismatch")
        if bundle["artifact_access"] != "privileged_training_only":
            raise ValueError("H7 checkpoint access class mismatch")
        if bundle["runtime_hash"] != self.runtime_hash:
            raise ValueError("H7 runtime identity mismatch")
        if bundle["runtime_identity"] != self.runtime_identity:
            raise ValueError("H7 runtime identity payload mismatch")
        previous_stats = copy.deepcopy(self.stats)
        previous_rng = self._rng.getstate()
        previous_snapshot = self._snapshot_step
        with tempfile.TemporaryDirectory(prefix="douzero-h7-load-") as temporary:
            inner_path = Path(temporary) / "h6.pt"
            torch.save(bundle["h6_checkpoint"], inner_path)
            try:
                self.learner.load_checkpoint(inner_path)
                self.stats = V3H7RuntimeStats(**bundle["stats"])
                self._rng.setstate(bundle["rng_state"])
                self._snapshot_step = int(bundle["snapshot_step"])
                if self._snapshot_step > self.policy_step:
                    raise ValueError("H7 checkpoint snapshot is newer than learner")
                if self.policy_step - self._snapshot_step > self.config.max_policy_lag:
                    raise ValueError("H7 checkpoint policy lag exceeds its bound")
            except Exception:
                self.stats = previous_stats
                self._rng.setstate(previous_rng)
                self._snapshot_step = previous_snapshot
                raise
        return bundle["long_running_state"]

    def shutdown(self) -> None:
        if not self._runtime_started:
            return
        error = None
        try:
            self._coordinator.request_shutdown()
            for _ in self._workers:
                self._tasks.put(None)
            deadline = time.monotonic() + 5.0
            for process in self._workers:
                process.join(max(0.0, deadline - time.monotonic()))
            alive = [process.name for process in self._workers if process.is_alive()]
            if alive:
                error = RuntimeError(f"H7 actor shutdown timed out: {alive}")
            from douzero.training.async_single_gpu import SlotState

            active_slots = sum(
                int((self._coordinator.states == int(state)).sum().item())
                for state in (SlotState.WRITING, SlotState.READY, SlotState.RUNNING)
            )
            if active_slots:
                error = RuntimeError("H7 shutdown left active inference slots")
            if self._scheduler.pending_count:
                error = RuntimeError("H7 shutdown left pending requests")
        finally:
            self._coordinator.shutdown()
            self._replay_slots.close()
            self._tasks.close()
            self._events.close()
            self._runtime_started = False
        if error is not None:
            raise error


__all__ = [
    "V3_H7_CHECKPOINT_FORMAT",
    "V3_H7_REPLAY_PROTOCOL",
    "V3_H7_REQUEST_PROTOCOL",
    "V3_H7_RUNTIME_VERSION",
    "V3AsyncSingleGPUTrainer",
    "V3H7RuntimeConfig",
    "V3H7RuntimeStats",
]
