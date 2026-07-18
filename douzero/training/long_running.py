"""Cycle-boundary orchestration for unattended V2 training.

Replay is intentionally ephemeral.  Every completed cycle ends at an empty
replay boundary, whether or not that boundary publishes a checkpoint.  This
makes a stopped-and-resumed N+M run equivalent to an uninterrupted run with
the same seed and cycle configuration without pretending to serialize replay
objects.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, current_thread, main_thread
from typing import Callable

from douzero.runtime.cleanup import run_cleanup_steps


_RUN_STATE_VERSION = 3


def _peak_ram_bytes() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


class _TrainerLifecycle:
    """Compatibility adapter while keeping controller code buffer-agnostic."""

    def __init__(self, trainer) -> None:
        self._trainer = trainer

    def quiesce_cycle_boundary(self):
        method = getattr(self._trainer, "quiesce_cycle_boundary", None)
        return method() if method is not None else {}

    def clear_replay(self) -> None:
        method = getattr(self._trainer, "clear_replay", None)
        if method is not None:
            method()
            return
        # Checkpoint-era test doubles predate the public lifecycle contract.
        for name in ("buffer", "bidding_buffer"):
            buffer = getattr(self._trainer, name, None)
            if buffer is not None:
                buffer.clear()

    def shutdown(self) -> None:
        method = getattr(self._trainer, "shutdown", None)
        if method is not None:
            method()


@dataclass(frozen=True)
class LongRunningConfig:
    episodes_per_cycle: int
    optimizer_steps_per_cycle: int
    max_cycles: int = 0
    max_total_episodes: int = 0
    max_total_optimizer_steps: int = 0
    max_wall_time_minutes: float = 0.0
    checkpoint_every_cycles: int = 1
    checkpoint_every_steps: int = 0
    checkpoint_every_minutes: float = 0.0
    keep_last_checkpoints: int = 3
    save_on_interrupt: bool = True
    eval_every_cycles: int = 0
    eval_fail_fast: bool = True
    v2_training_mode: str = "single_process"
    num_actors: int = 1
    replay_schema_version: int = 1
    snapshot_publication_semantics: str = "cycle_quiescent_atomic_copy_v1"
    request_ordering_semantics: str = "policy_bucket_role_fifo_microbatch_v1"
    actor_rng_resume_semantics: str = "restart_from_configured_seeds_v1"

    def __post_init__(self) -> None:
        integer_nonnegative = (
            "max_cycles", "max_total_episodes", "max_total_optimizer_steps",
            "checkpoint_every_cycles", "checkpoint_every_steps", "eval_every_cycles",
        )
        if self.episodes_per_cycle < 1:
            raise ValueError("episodes_per_cycle must be >= 1")
        if self.optimizer_steps_per_cycle < 0:
            raise ValueError("optimizer_steps_per_cycle must be >= 0")
        for name in integer_nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("max_wall_time_minutes", "checkpoint_every_minutes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be >= 1")

    def resume_identity(self) -> dict[str, int | str]:
        """Return fields that affect the mathematical N+M trajectory."""
        return {
            "episodes_per_cycle": self.episodes_per_cycle,
            "optimizer_steps_per_cycle": self.optimizer_steps_per_cycle,
            "v2_training_mode": self.v2_training_mode,
            "num_actors": self.num_actors,
            "replay_schema_version": self.replay_schema_version,
            "snapshot_publication_semantics": self.snapshot_publication_semantics,
            "request_ordering_semantics": self.request_ordering_semantics,
            "actor_rng_resume_semantics": self.actor_rng_resume_semantics,
        }


@dataclass
class LongRunningState:
    cycle: int = 0
    total_episodes: int = 0
    total_transitions: int = 0
    total_optimizer_steps: int = 0
    policy_version: str = "current"
    policy_step: int = 0
    resume_source: str = ""
    checkpoint_sequence: int = 0
    total_wall_seconds: float = 0.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_state_version: int = _RUN_STATE_VERSION
    cycle_identity: dict[str, int | str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "LongRunningState":
        if not isinstance(payload, dict):
            raise ValueError("checkpoint long_running_state must be a dict")
        migrated = dict(payload)
        if migrated.get("run_state_version") in {1, 2}:
            previous_version = migrated["run_state_version"]
            migrated["run_state_version"] = _RUN_STATE_VERSION
            if previous_version == 1:
                migrated.setdefault("run_id", "legacy")
            migrated.setdefault("total_wall_seconds", 0.0)
        state = cls(**migrated)
        if state.run_state_version != _RUN_STATE_VERSION:
            raise ValueError("unsupported long-running checkpoint state version")
        if not isinstance(state.run_id, str) or not state.run_id:
            raise ValueError("long-running state run_id must be non-empty text")
        if not all(character.isalnum() or character in "-_" for character in state.run_id):
            raise ValueError("long-running state run_id contains unsafe characters")
        for name in (
            "cycle", "total_episodes", "total_transitions",
            "total_optimizer_steps", "policy_step", "checkpoint_sequence",
        ):
            value = getattr(state, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"long-running state {name} must be a non-negative int")
        if (
            isinstance(state.total_wall_seconds, bool)
            or not isinstance(state.total_wall_seconds, (int, float))
            or not math.isfinite(state.total_wall_seconds)
            or state.total_wall_seconds < 0
        ):
            raise ValueError(
                "long-running state total_wall_seconds must be finite and non-negative"
            )
        state.total_wall_seconds = float(state.total_wall_seconds)
        return state


class StopController:
    """Turn SIGINT/SIGTERM into a boundary-stop request."""

    def __init__(self, event: Event | None = None) -> None:
        self.event = event or Event()
        self.reason = ""
        self._previous: dict[int, object] = {}

    def request(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
        self.event.set()

    def _handle(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        self.request(name.lower())

    def install(self) -> None:
        if current_thread() is not main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


class CheckpointSeries:
    """Publish immutable cycle files plus an atomically replaced latest manifest."""

    def __init__(self, base_path: str, keep_last: int) -> None:
        self.base = Path(base_path)
        self.keep_last = keep_last
        suffix = self.base.suffix or ".pt"
        stem = self.base.name[:-len(suffix)] if self.base.name.endswith(suffix) else self.base.name
        self.prefix = stem
        self.suffix = suffix
        self.latest_manifest = self.base.with_name(f"{self.prefix}-latest.json")
        self.pending_manifest = self.base.with_name(f".{self.prefix}-pending.json")
        self.lock_path = self.base.with_name(f".{self.prefix}.lock")
        self.last_rotation_error = ""
        self._lock_token = ""

    def acquire(self, run_id: str) -> None:
        """Atomically own this series until ``release`` is called."""
        if self._lock_token:
            raise RuntimeError("checkpoint series lock is already held")
        self.base.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        payload = {
            "schema_version": "v2-checkpoint-series-lock-v1",
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at_unix": time.time(),
            "token": token,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(
                f"checkpoint series is locked: {self.lock_path}; verify the owner "
                "has stopped before explicitly removing a stale lock"
            ) from exc
        try:
            encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("checkpoint series lock write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        self._lock_token = token

    def release(self) -> None:
        """Release this process's lock without deleting another owner's lock."""
        if not self._lock_token:
            return
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if payload.get("token") != self._lock_token:
                raise RuntimeError("checkpoint series lock ownership changed")
            self.lock_path.unlink()
        finally:
            self._lock_token = ""

    def _require_lock(self) -> None:
        if not self._lock_token:
            raise RuntimeError("checkpoint series operation requires ownership lock")

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, state: LongRunningState,
                        keep_last: int) -> "CheckpointSeries":
        """Derive and validate the series base encoded in a checkpoint name."""
        source = Path(checkpoint)
        marker = f"-run-{state.run_id}-seq-{state.checkpoint_sequence:06d}"
        if marker not in source.name:
            raise ValueError("resume checkpoint filename does not match its run state")
        prefix, encoded = source.name.split(marker, 1)
        suffix = source.suffix or ".pt"
        expected_tail = (
            f"-cycle-{state.cycle:06d}-step-"
            f"{state.total_optimizer_steps:012d}{suffix}"
        )
        if encoded != expected_tail or not prefix:
            raise ValueError("resume checkpoint filename does not match its counters")
        return cls(str(source.with_name(f"{prefix}{suffix}")), keep_last)

    def _manifest_payload(self, state: LongRunningState, checkpoint: Path) -> dict:
        return {
            "schema_version": 2,
            "latest": checkpoint.name,
            "cycle": state.cycle,
            "total_episodes": state.total_episodes,
            "total_optimizer_steps": state.total_optimizer_steps,
            "policy_version": state.policy_version,
            "policy_step": state.policy_step,
            "checkpoint_sequence": state.checkpoint_sequence,
            "total_wall_seconds": state.total_wall_seconds,
            "run_id": state.run_id,
        }

    @staticmethod
    def _validate_manifest(payload: dict) -> dict:
        if isinstance(payload, dict) and payload.get("schema_version") == 1:
            old_required = {
                "schema_version", "latest", "cycle", "total_episodes",
                "total_optimizer_steps", "policy_version", "policy_step",
                "checkpoint_sequence", "run_id",
            }
            if set(payload) == old_required:
                payload = dict(payload)
                payload["schema_version"] = 2
                payload["total_wall_seconds"] = 0.0
        required = {
            "schema_version", "latest", "cycle", "total_episodes",
            "total_optimizer_steps", "policy_version", "policy_step",
            "checkpoint_sequence", "total_wall_seconds", "run_id",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("latest manifest has an invalid field set")
        if payload["schema_version"] != 2:
            raise ValueError("latest manifest has an unsupported schema_version")
        if (
            not isinstance(payload["latest"], str)
            or not payload["latest"]
            or Path(payload["latest"]).name != payload["latest"]
        ):
            raise ValueError("latest manifest has an unsafe checkpoint name")
        for name in (
            "cycle", "total_episodes", "total_optimizer_steps", "policy_step",
            "checkpoint_sequence",
        ):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"latest manifest {name} must be a non-negative int")
        if not isinstance(payload["policy_version"], str):
            raise ValueError("latest manifest policy_version must be text")
        wall_seconds = payload["total_wall_seconds"]
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds)
            or wall_seconds < 0
        ):
            raise ValueError(
                "latest manifest total_wall_seconds must be finite and non-negative"
            )
        payload["total_wall_seconds"] = float(wall_seconds)
        run_id = payload["run_id"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or not all(character.isalnum() or character in "-_" for character in run_id)
        ):
            raise ValueError("latest manifest run_id is invalid")
        return payload

    def bind_resume(self, state: LongRunningState, checkpoint: str | Path) -> None:
        """Bind resume to this series and reconcile one committed orphan."""
        self._require_lock()
        source = Path(checkpoint)
        expected = self.cycle_path(
            state.run_id,
            state.checkpoint_sequence,
            state.cycle,
            state.total_optimizer_steps,
        )
        if source.name != expected.name or source.parent.resolve() != expected.parent.resolve():
            raise ValueError("resume checkpoint does not belong to checkpoint series")
        if not source.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {source}")

        all_series_files = list(
            self.base.parent.glob(
                f"{self.prefix}-run-*-seq-*-cycle-*-step-*{self.suffix}"
            )
        )
        run_files = self._indexed_checkpoints(state.run_id)

        if not self.latest_manifest.exists():
            if (
                state.checkpoint_sequence != 1
                or len(all_series_files) != 1
                or run_files != {1: source}
                or not self.pending_manifest.is_file()
            ):
                raise ValueError(
                    "long-running resume without a latest manifest requires exactly "
                    "one matching sequence-1 checkpoint and publication intent"
                )
            pending = self._validate_manifest(
                json.loads(self.pending_manifest.read_text(encoding="utf-8"))
            )
            if pending != self._manifest_payload(state, source):
                raise ValueError(
                    "initial checkpoint publication intent does not match resume state"
                )
            self._write_json_atomic(
                self.latest_manifest, self._manifest_payload(state, source)
            )
            self.pending_manifest.unlink(missing_ok=True)
            return

        payload = self._validate_manifest(
            json.loads(self.latest_manifest.read_text(encoding="utf-8"))
        )
        if payload["run_id"] != state.run_id:
            raise ValueError("checkpoint series run_id mismatch")
        manifest_sequence = payload["checkpoint_sequence"]
        if not isinstance(manifest_sequence, int) or isinstance(manifest_sequence, bool):
            raise ValueError("latest manifest checkpoint_sequence must be an int")
        higher_sequences = sorted(
            sequence for sequence in run_files if sequence > manifest_sequence
        )
        if higher_sequences and higher_sequences != [manifest_sequence + 1]:
            raise ValueError(
                "checkpoint series contains multiple, non-contiguous, or skipped orphans"
            )
        if higher_sequences and state.checkpoint_sequence != higher_sequences[0]:
            raise ValueError(
                "refusing to ignore a newer orphan checkpoint; resume the latest manifest"
            )
        if manifest_sequence > state.checkpoint_sequence:
            raise ValueError("refusing to resume an older checkpoint than latest")
        if manifest_sequence < state.checkpoint_sequence - 1:
            raise ValueError("checkpoint sequence is not contiguous with latest")
        if manifest_sequence == state.checkpoint_sequence - 1:
            if run_files.get(state.checkpoint_sequence) != source:
                raise ValueError("orphan checkpoint does not uniquely match resume state")
            self._write_json_atomic(
                self.latest_manifest, self._manifest_payload(state, source)
            )
            self.pending_manifest.unlink(missing_ok=True)
            return

        state.total_wall_seconds = max(
            state.total_wall_seconds, payload["total_wall_seconds"]
        )
        expected_payload = self._manifest_payload(state, source)
        if payload != expected_payload:
            raise ValueError("latest manifest does not match checkpoint state")
        self.pending_manifest.unlink(missing_ok=True)

    def _indexed_checkpoints(self, run_id: str) -> dict[int, Path]:
        """Return this run's files indexed by sequence, rejecting ambiguity."""
        pattern = (
            f"{self.prefix}-run-{run_id}-seq-*-cycle-*-step-*{self.suffix}"
        )
        indexed: dict[int, Path] = {}
        for candidate in self.base.parent.glob(pattern):
            sequence = self.checkpoint_sequence(candidate, run_id=run_id)
            if sequence in indexed:
                raise ValueError(
                    f"checkpoint series contains duplicate sequence {sequence}"
                )
            indexed[sequence] = candidate
        return indexed

    def ensure_available(self, state: LongRunningState) -> None:
        """Fail closed when a fresh run would reuse an existing series."""
        self._require_lock()
        if state.checkpoint_sequence:
            source = state.resume_source
            if not source and self.latest_manifest.exists():
                payload = self._validate_manifest(
                    json.loads(self.latest_manifest.read_text(encoding="utf-8"))
                )
                source = str(self.latest_manifest.parent / payload["latest"])
            if not source:
                raise ValueError("resumed long-running state has no checkpoint source")
            self.bind_resume(state, source)
            return
        legacy = list(self.base.parent.glob(f"{self.prefix}-seq-*{self.suffix}"))
        current = list(self.base.parent.glob(f"{self.prefix}-run-*{self.suffix}"))
        if self.latest_manifest.exists() or legacy or current:
            raise FileExistsError(
                f"checkpoint series already exists for {self.base}; resume from "
                f"{self.latest_manifest} or choose a new --checkpoint_path"
            )

    def cycle_path(
        self, run_id: str, sequence: int, cycle: int, steps: int
    ) -> Path:
        return self.base.with_name(
            f"{self.prefix}-run-{run_id}-seq-{sequence:06d}-cycle-{cycle:06d}"
            f"-step-{steps:012d}{self.suffix}"
        )

    def checkpoint_sequence(self, path: str | Path, *, run_id: str) -> int:
        pattern = re.compile(
            rf"{re.escape(self.prefix)}-run-{re.escape(run_id)}-seq-(\d+)-"
            rf"cycle-\d+-step-\d+{re.escape(self.suffix)}"
        )
        match = pattern.fullmatch(Path(path).name)
        if match is None:
            raise ValueError("checkpoint filename does not match its series")
        return int(match.group(1))

    def publish(self, trainer, state: LongRunningState) -> Path:
        self._require_lock()
        self.last_rotation_error = ""
        sequence = state.checkpoint_sequence + 1
        destination = self.cycle_path(
            state.run_id, sequence, state.cycle, state.total_optimizer_steps
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {destination}")
        checkpoint_state = asdict(state)
        checkpoint_state["checkpoint_sequence"] = sequence
        manifest_state = LongRunningState.from_dict(checkpoint_state)
        manifest = self._manifest_payload(manifest_state, destination)
        self._write_json_atomic(self.pending_manifest, manifest)
        trainer.save_training_checkpoint(str(destination), long_running_state=checkpoint_state)
        self._write_json_atomic(self.latest_manifest, manifest)
        self.pending_manifest.unlink(missing_ok=True)
        state.checkpoint_sequence = sequence
        try:
            self._rotate(destination)
        except OSError as exc:
            self.last_rotation_error = type(exc).__name__
        return destination

    def update_runtime_state(
        self, state: LongRunningState, checkpoint: str | Path
    ) -> None:
        """Atomically persist boundary work accounted after model publication."""
        self._require_lock()
        path = Path(checkpoint)
        expected = self.cycle_path(
            state.run_id,
            state.checkpoint_sequence,
            state.cycle,
            state.total_optimizer_steps,
        )
        if path.resolve() != expected.resolve() or not path.is_file():
            raise ValueError("runtime state checkpoint does not match checkpoint series")
        self._write_json_atomic(
            self.latest_manifest, self._manifest_payload(state, path)
        )

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _rotate(self, current: Path) -> None:
        marker = f"{self.prefix}-run-"
        run_id = current.name[len(marker):].split("-seq-", 1)[0]
        pattern = f"{self.prefix}-run-{run_id}-seq-*-cycle-*-step-*{self.suffix}"
        checkpoints = sorted(
            self.base.parent.glob(pattern),
            key=lambda path: self.checkpoint_sequence(path, run_id=run_id),
            reverse=True,
        )
        for old in checkpoints[self.keep_last:]:
            if old != current:
                old.unlink()


class RunMetricsWriter:
    """Append cycle records once and atomically maintain a constant-size summary."""

    _SUMMARY_FIELDS = {
        "schema_version", "run_id", "status", "stop_reason", "error",
        "cycles_path", "state", "latest_cycle",
    }

    @staticmethod
    def output_paths(summary_path: str | Path) -> tuple[Path, Path]:
        summary = Path(summary_path)
        suffix = summary.suffix or ".json"
        stem = (
            summary.name[:-len(suffix)]
            if summary.name.endswith(suffix) else summary.name
        )
        return summary, summary.with_name(f"{stem}-cycles.jsonl")

    @classmethod
    def validate_checkpoint_paths(
        cls, summary_path: str | Path, series: CheckpointSeries
    ) -> None:
        """Reject metrics paths inside the checkpoint series namespace."""
        summary, cycles = cls.output_paths(summary_path)
        outputs = {summary.resolve(), cycles.resolve()}
        protected = {
            series.base.resolve(),
            series.latest_manifest.resolve(),
            series.pending_manifest.resolve(),
            series.lock_path.resolve(),
        }
        if outputs & protected:
            raise ValueError("metrics output conflicts with checkpoint series paths")
        cycle_pattern = re.compile(
            rf"{re.escape(series.prefix)}-run-.+-seq-\d+-cycle-\d+-step-\d+"
            rf"{re.escape(series.suffix)}"
        )
        if any(
            path.parent == series.base.resolve().parent
            and cycle_pattern.fullmatch(path.name)
            for path in outputs
        ):
            raise ValueError(
                "metrics output conflicts with checkpoint cycle namespace"
            )

    def __init__(self, summary_path: str, *, run_id: str, resume: bool) -> None:
        self.summary_path, self.cycles_path = self.output_paths(summary_path)
        self.run_id = run_id
        self._latest_cycle: dict | None = None
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and (self.summary_path.exists() or self.cycles_path.exists()):
            raise FileExistsError(
                f"metrics output already exists for {self.summary_path}; resume "
                "the run or choose a new --metrics_path"
            )
        if resume and not self.summary_path.exists() and self.cycles_path.exists():
            raise ValueError(
                "metrics cycle history exists without its run summary"
            )
        if resume and self.summary_path.exists():
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != self._SUMMARY_FIELDS:
                raise ValueError("metrics summary has an invalid field set")
            if payload.get("schema_version") != "v2-long-running-run-v2":
                raise ValueError("metrics summary has an unsupported schema_version")
            if payload.get("run_id") != run_id:
                raise ValueError(
                    "metrics run_id does not match the resumed checkpoint"
                )
            if payload.get("cycles_path") != self.cycles_path.name:
                raise ValueError("metrics summary cycles_path does not match output")
            if not all(
                isinstance(payload[name], str)
                for name in ("status", "stop_reason", "error")
            ):
                raise ValueError("metrics summary status fields must be text")
            if payload["state"] is not None and not isinstance(payload["state"], dict):
                raise ValueError("metrics summary state must be an object or null")
            if (
                payload["latest_cycle"] is not None
                and not isinstance(payload["latest_cycle"], dict)
            ):
                raise ValueError(
                    "metrics summary latest_cycle must be an object or null"
                )
            latest_cycle = payload.get("latest_cycle")
            if isinstance(latest_cycle, dict):
                self._latest_cycle = latest_cycle

    def write_cycle(self, record: dict) -> None:
        self._latest_cycle = dict(record)
        encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with self.cycles_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._write_summary(
            status="running",
            stop_reason=str(record.get("stop_reason", "")),
            state=None,
            error="",
            latest_cycle=self._latest_cycle,
        )

    def finalize(
        self,
        *,
        status: str,
        stop_reason: str,
        state: LongRunningState | None,
        error: str = "",
        latest_cycle: dict | None = None,
    ) -> None:
        self._write_summary(
            status=status,
            stop_reason=stop_reason,
            state=state,
            error=error,
            latest_cycle=(
                latest_cycle if latest_cycle is not None else self._latest_cycle
            ),
        )

    def _write_summary(
        self,
        *,
        status: str,
        stop_reason: str,
        state: LongRunningState | None,
        error: str,
        latest_cycle: dict | None,
    ) -> None:
        state_payload = asdict(state) if state is not None else None
        if state_payload is not None and state_payload["resume_source"]:
            state_payload["resume_source"] = Path(
                state_payload["resume_source"]
            ).name
        error_type = error.split(":", 1)[0].strip() if error else ""
        if error_type and not error_type.replace("_", "").isalnum():
            error_type = "Error"
        payload = {
            "schema_version": "v2-long-running-run-v2",
            "run_id": self.run_id,
            "status": status,
            "stop_reason": stop_reason,
            "error": error_type,
            "cycles_path": self.cycles_path.name,
            "state": state_payload,
            "latest_cycle": latest_cycle,
        }
        CheckpointSeries._write_json_atomic(self.summary_path, payload)


class LongRunningTrainer:
    """Run collect/optimize cycles and stop only at empty-replay boundaries."""

    def __init__(
        self,
        trainer,
        config: LongRunningConfig,
        checkpoint_series: CheckpointSeries,
        *,
        state: LongRunningState | None = None,
        stop_controller: StopController | None = None,
        evaluator: Callable[[Path, int], None] | None = None,
        metric_sink: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        peak_memory: Callable[[], int | None] | None = None,
        collect_records: bool = False,
    ) -> None:
        self.trainer = trainer
        self.lifecycle = _TrainerLifecycle(trainer)
        self.config = config
        self.checkpoints = checkpoint_series
        self.stop = stop_controller or StopController()
        self.evaluator = evaluator
        self.metric_sink = metric_sink or (lambda _record: None)
        self.clock = clock
        self.peak_memory = peak_memory or (lambda: None)
        self.collect_records = collect_records
        self.last_stop_reason = ""
        self.state = state or LongRunningState(
            total_episodes=int(trainer.stats.episodes_completed),
            total_transitions=int(trainer.stats.transitions_collected),
            total_optimizer_steps=int(trainer.stats.optimizer_steps),
            policy_version=str(trainer.policy_version),
            policy_step=int(trainer.policy_step),
            cycle_identity=config.resume_identity(),
        )
        legacy_cycle_identity = {
            "episodes_per_cycle": config.episodes_per_cycle,
            "optimizer_steps_per_cycle": config.optimizer_steps_per_cycle,
        }
        if (
            self.state.cycle_identity == legacy_cycle_identity
            and config.v2_training_mode == "single_process"
            and config.num_actors == 1
            and config.replay_schema_version == 1
        ):
            # Explicit v3 single-process migration: the old identity predates
            # topology fields but already guaranteed an empty cycle boundary.
            self.state.cycle_identity = config.resume_identity()
        if self.state.cycle_identity != config.resume_identity():
            raise ValueError(
                "long-running resume identity mismatch: checkpoint has "
                f"{self.state.cycle_identity!r}, runtime expects {config.resume_identity()!r}"
            )
        self._validate_totals()
        self._validate_reachable_limits()
        self.checkpoints.acquire(self.state.run_id)
        try:
            self.checkpoints.ensure_available(self.state)
        except BaseException:
            self.checkpoints.release()
            raise

    def _validate_reachable_limits(self) -> None:
        cfg = self.config
        optimizer_limit_pending = (
            cfg.max_total_optimizer_steps > self.state.total_optimizer_steps
        )
        alternate_limit = bool(
            cfg.max_cycles
            or cfg.max_total_episodes
            or cfg.max_wall_time_minutes
        )
        if (
            cfg.optimizer_steps_per_cycle == 0
            and optimizer_limit_pending
            and not alternate_limit
        ):
            raise ValueError(
                "max_total_optimizer_steps is unreachable when "
                "optimizer_steps_per_cycle is 0 without another stop limit"
            )

    def _validate_totals(self) -> None:
        pairs = (
            ("total_episodes", self.trainer.stats.episodes_completed),
            ("total_transitions", self.trainer.stats.transitions_collected),
            ("total_optimizer_steps", self.trainer.stats.optimizer_steps),
            ("policy_step", self.trainer.policy_step),
        )
        for name, actual in pairs:
            if getattr(self.state, name) != int(actual):
                raise ValueError(
                    f"long-running state {name} does not match trainer checkpoint stats"
                )
        if self.state.policy_version != str(self.trainer.policy_version):
            raise ValueError("long-running state policy_version does not match trainer")

    def _limit_reason(self, total_wall_seconds: float) -> str:
        cfg, state = self.config, self.state
        if cfg.max_cycles and state.cycle >= cfg.max_cycles:
            return "max_cycles"
        if cfg.max_total_episodes and state.total_episodes >= cfg.max_total_episodes:
            return "max_total_episodes"
        if (cfg.max_total_optimizer_steps and
                state.total_optimizer_steps >= cfg.max_total_optimizer_steps):
            return "max_total_optimizer_steps"
        if (
            cfg.max_wall_time_minutes
            and total_wall_seconds >= cfg.max_wall_time_minutes * 60.0
        ):
            return "max_wall_time"
        return ""

    def run(self) -> tuple[LongRunningState, str, list[dict]]:
        started = self.clock()
        base_wall_seconds = self.state.total_wall_seconds
        last_checkpoint_time = started
        last_checkpoint_steps = self.state.total_optimizer_steps
        last_checkpointed_cycle = (
            self.state.cycle if self.state.checkpoint_sequence else 0
        )
        records: list[dict] = []
        reason = ""
        try:
            self.stop.install()
            while True:
                elapsed = self.clock() - started
                self.state.total_wall_seconds = base_wall_seconds + elapsed
                reason = self._limit_reason(self.state.total_wall_seconds)
                if self.stop.event.is_set():
                    reason = self.stop.reason or "stop_event"
                if reason:
                    self.last_stop_reason = reason
                    interrupt = reason in {"sigint", "sigterm", "stop_event"}
                    save_dirty_boundary = (
                        self.state.cycle > last_checkpointed_cycle
                        and (not interrupt or self.config.save_on_interrupt)
                    )
                    if save_dirty_boundary:
                        checkpoint_path = None
                        checkpoint_status = "failed"
                        checkpoint_error = ""
                        try:
                            checkpoint_path = self.checkpoints.publish(
                                self.trainer, self.state
                            )
                            checkpoint_status = (
                                "saved_rotation_failed"
                                if self.checkpoints.last_rotation_error else "saved"
                            )
                            checkpoint_error = self.checkpoints.last_rotation_error
                            last_checkpointed_cycle = self.state.cycle
                        except Exception as exc:
                            checkpoint_error = type(exc).__name__
                        record = {
                            "schema_version": "v2-long-running-cycle-v1",
                            "event": "late_stop_checkpoint",
                            "cycle": self.state.cycle,
                            "total_episodes": self.state.total_episodes,
                            "total_transitions": self.state.total_transitions,
                            "total_optimizer_steps": self.state.total_optimizer_steps,
                            "policy_version": self.state.policy_version,
                            "policy_step": self.state.policy_step,
                            "cycle_wall_seconds": 0.0,
                            "collection_seconds": 0.0,
                            "optimization_seconds": 0.0,
                            "amp_fallback": 0,
                            "checkpoint_path": (
                                checkpoint_path.name if checkpoint_path else ""
                            ),
                            "checkpoint_status": checkpoint_status,
                            "checkpoint_error": checkpoint_error,
                            "resume_source": (
                                Path(self.state.resume_source).name
                                if self.state.resume_source else ""
                            ),
                            "peak_memory_bytes": self.peak_memory(),
                            "peak_ram_bytes": _peak_ram_bytes(),
                            "evaluation_status": "not_due",
                            "evaluation_error": "",
                            "stop_reason": reason,
                        }
                        if self.collect_records:
                            records.append(record)
                        self.metric_sink(record)
                        if checkpoint_path is None:
                            raise RuntimeError(
                                f"checkpoint publication failed: {checkpoint_error}"
                            )
                        self.state.total_wall_seconds = (
                            base_wall_seconds + self.clock() - started
                        )
                        try:
                            self.checkpoints.update_runtime_state(
                                self.state, checkpoint_path
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "checkpoint runtime-state publication failed: "
                                f"{type(exc).__name__}"
                            ) from exc
                    break

                episodes = self.config.episodes_per_cycle
                if self.config.max_total_episodes:
                    episodes = min(
                        episodes,
                        self.config.max_total_episodes - self.state.total_episodes,
                    )
                steps = self.config.optimizer_steps_per_cycle
                if self.config.max_total_optimizer_steps:
                    steps = min(
                        steps,
                        self.config.max_total_optimizer_steps
                        - self.state.total_optimizer_steps,
                    )

                cycle_started = self.clock()
                amp_before = int(self.trainer.stats.amp_fallbacks)
                transitions_before = int(self.trainer.stats.transitions_collected)
                decisions_before = int(getattr(
                    self.trainer.stats,
                    "decisions_collected",
                    transitions_before,
                ))
                collection_started = self.clock()
                self.trainer.collect_episodes(episodes)
                collection_seconds = self.clock() - collection_started
                optimization_started = self.clock()
                steps_taken = 0
                for _ in range(steps):
                    if self.trainer.step() is not None:
                        steps_taken += 1
                if steps and steps_taken != steps:
                    raise RuntimeError(
                        f"requested {steps} optimizer steps in cycle but took {steps_taken}"
                    )
                optimization_seconds = self.clock() - optimization_started
                quiesce_started = self.clock()
                boundary_status = self.lifecycle.quiesce_cycle_boundary() or {}
                quiesce_seconds = self.clock() - quiesce_started

                self.state.cycle += 1
                self.state.total_episodes = int(self.trainer.stats.episodes_completed)
                self.state.total_transitions = int(self.trainer.stats.transitions_collected)
                self.state.total_optimizer_steps = int(self.trainer.stats.optimizer_steps)
                self.state.policy_step = int(self.trainer.policy_step)
                self.state.policy_version = str(self.trainer.policy_version)

                now = self.clock()
                self.state.total_wall_seconds = base_wall_seconds + now - started
                boundary_reason = self._limit_reason(self.state.total_wall_seconds)
                if self.stop.event.is_set():
                    boundary_reason = self.stop.reason or "stop_event"
                if boundary_reason:
                    self.last_stop_reason = boundary_reason
                cycle_due = bool(
                    self.config.checkpoint_every_cycles
                    and self.state.cycle % self.config.checkpoint_every_cycles == 0
                )
                steps_due = bool(
                    self.config.checkpoint_every_steps
                    and self.state.total_optimizer_steps - last_checkpoint_steps
                    >= self.config.checkpoint_every_steps
                )
                time_due = bool(
                    self.config.checkpoint_every_minutes
                    and now - last_checkpoint_time
                    >= self.config.checkpoint_every_minutes * 60.0
                )
                evaluation_due = bool(
                    self.config.eval_every_cycles
                    and self.state.cycle % self.config.eval_every_cycles == 0
                )
                interrupt = boundary_reason in {"sigint", "sigterm", "stop_event"}
                stop_save_due = bool(
                    boundary_reason
                    and (not interrupt or self.config.save_on_interrupt)
                )
                scheduled_due = cycle_due or steps_due or time_due
                due = scheduled_due or evaluation_due or stop_save_due
                checkpoint_path: Path | None = None
                checkpoint_status = "not_due"
                checkpoint_error = ""
                if due:
                    try:
                        checkpoint_path = self.checkpoints.publish(self.trainer, self.state)
                        checkpoint_status = (
                            "saved_rotation_failed"
                            if self.checkpoints.last_rotation_error else "saved"
                        )
                        checkpoint_error = self.checkpoints.last_rotation_error
                        last_checkpoint_time = self.clock()
                        last_checkpoint_steps = self.state.total_optimizer_steps
                        last_checkpointed_cycle = self.state.cycle
                    except Exception as exc:
                        checkpoint_status = "failed"
                        checkpoint_error = type(exc).__name__
                # Saving clears replay. Non-save boundaries deliberately do so too.
                self.lifecycle.clear_replay()

                eval_status = "not_due"
                eval_error = ""
                if evaluation_due:
                    if self.evaluator is None:
                        raise ValueError("eval_every_cycles requires an evaluator")
                    if checkpoint_status == "failed":
                        eval_status = "skipped_checkpoint_failed"
                        eval_error = checkpoint_error
                    else:
                        try:
                            if checkpoint_path is None:
                                raise RuntimeError(
                                    "periodic evaluation requires a saved checkpoint"
                                )
                            self.evaluator(checkpoint_path, self.state.cycle)
                            eval_status = "passed"
                        except Exception as exc:
                            if self.stop.event.is_set():
                                eval_status = "interrupted"
                                boundary_reason = self.stop.reason or "stop_event"
                                self.last_stop_reason = boundary_reason
                            else:
                                eval_status = "failed"
                                eval_error = type(exc).__name__

                # Checkpoint publication and evaluation are part of the cumulative
                # controller wall budget. The model checkpoint remains immutable;
                # the latest manifest carries this post-publication run state.
                self.state.total_wall_seconds = (
                    base_wall_seconds + self.clock() - started
                )
                post_service_reason = self._limit_reason(
                    self.state.total_wall_seconds
                )
                if self.stop.event.is_set():
                    post_service_reason = self.stop.reason or "stop_event"
                if post_service_reason:
                    boundary_reason = post_service_reason
                    self.last_stop_reason = boundary_reason

                record = {
                    "schema_version": "v2-long-running-cycle-v1",
                    "event": "cycle",
                    "cycle": self.state.cycle,
                    "total_episodes": self.state.total_episodes,
                    "total_transitions": self.state.total_transitions,
                    "total_optimizer_steps": self.state.total_optimizer_steps,
                    "policy_version": self.state.policy_version,
                    "policy_step": self.state.policy_step,
                    "cycle_wall_seconds": round(self.clock() - cycle_started, 6),
                    "collection_seconds": round(collection_seconds, 6),
                    "optimization_seconds": round(optimization_seconds, 6),
                    "cycle_quiesce_seconds": round(quiesce_seconds, 6),
                    "replay_occupancy": int(boundary_status.get("replay_occupancy", 0)),
                    "active_slots": int(boundary_status.get("active_slots", 0)),
                    "in_flight_slots": int(boundary_status.get("in_flight_slots", 0)),
                    "transitions_per_second": round(
                        (self.state.total_transitions - transitions_before)
                        / max(collection_seconds, 1e-12), 6
                    ),
                    "decisions_per_second": round(
                        (
                            int(getattr(
                                self.trainer.stats,
                                "decisions_collected",
                                self.state.total_transitions,
                            ))
                            - decisions_before
                        )
                        / max(collection_seconds, 1e-12), 6
                    ),
                    "trainable_decisions_per_second": round(
                        (self.state.total_transitions - transitions_before)
                        / max(collection_seconds, 1e-12), 6
                    ),
                    "learner_steps_per_second": round(
                        steps_taken / max(optimization_seconds, 1e-12), 6
                    ),
                    "requests_per_microbatch": round(
                        float(boundary_status.get("requests_per_microbatch", 0.0)), 6
                    ),
                    "actions_per_microbatch": round(
                        float(boundary_status.get("actions_per_microbatch", 0.0)), 6
                    ),
                    "inference_queue_p50_ms": round(
                        float(boundary_status.get("inference_queue_p50_ms", 0.0)), 6
                    ),
                    "inference_queue_p95_ms": round(
                        float(boundary_status.get("inference_queue_p95_ms", 0.0)), 6
                    ),
                    "inference_gpu_seconds": round(
                        float(boundary_status.get("inference_gpu_seconds", 0.0)), 6
                    ),
                    "learner_gpu_seconds": round(
                        float(boundary_status.get("learner_gpu_seconds", 0.0)), 6
                    ),
                    "policy_lag": int(boundary_status.get("policy_lag", 0)),
                    "amp_fallback": int(self.trainer.stats.amp_fallbacks) - amp_before,
                    "checkpoint_path": checkpoint_path.name if checkpoint_path else "",
                    "checkpoint_status": checkpoint_status,
                    "checkpoint_error": checkpoint_error,
                    "resume_source": (
                        Path(self.state.resume_source).name
                        if self.state.resume_source else ""
                    ),
                    "peak_memory_bytes": self.peak_memory(),
                    "peak_ram_bytes": _peak_ram_bytes(),
                    "peak_vram_bytes": self.peak_memory(),
                    "evaluation_status": eval_status,
                    "evaluation_error": eval_error,
                    "stop_reason": boundary_reason,
                }
                if self.collect_records:
                    records.append(record)
                self.metric_sink(record)
                self.state.total_wall_seconds = (
                    base_wall_seconds + self.clock() - started
                )
                final_boundary_reason = self._limit_reason(
                    self.state.total_wall_seconds
                )
                if self.stop.event.is_set():
                    final_boundary_reason = self.stop.reason or "stop_event"
                if final_boundary_reason:
                    # A request arriving inside metric emission still needs the
                    # normal late-stop path to publish this dirty boundary.
                    if checkpoint_path is not None or boundary_reason:
                        boundary_reason = final_boundary_reason
                        self.last_stop_reason = boundary_reason
                if checkpoint_path is not None and checkpoint_status != "failed":
                    try:
                        self.checkpoints.update_runtime_state(
                            self.state, checkpoint_path
                        )
                    except Exception as exc:
                        checkpoint_status = "failed"
                        checkpoint_error = type(exc).__name__
                if checkpoint_status == "failed":
                    raise RuntimeError(f"checkpoint publication failed: {checkpoint_error}")
                if eval_status == "failed" and self.config.eval_fail_fast:
                    raise RuntimeError(f"periodic evaluation failed: {eval_error}")
                if boundary_reason:
                    reason = boundary_reason
                    break
        finally:
            run_cleanup_steps(
                (
                    self.stop.restore,
                    self.lifecycle.shutdown,
                    self.checkpoints.release,
                ),
                preserve_active_exception=sys.exc_info()[0] is not None,
            )
        return self.state, reason, records


def command_evaluator(
    command: str,
    *,
    windows: bool | None = None,
    stop_controller: StopController | None = None,
    poll_seconds: float = 0.1,
    terminate_timeout_seconds: float = 5.0,
) -> Callable[[Path, int], None]:
    """Build a no-shell callback for an existing evaluation CLI command."""
    import shlex

    stripped = command.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list) or not all(
            isinstance(part, str) and part for part in parsed
        ):
            raise ValueError("JSON eval_command must be a non-empty string array")
        argv = parsed
    else:
        use_windows = os.name == "nt" if windows is None else windows
        argv = shlex.split(command, posix=not use_windows)
        if use_windows:
            argv = [
                part[1:-1]
                if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
                else part
                for part in argv
            ]
    if not argv:
        raise ValueError("eval_command must not be empty")
    if poll_seconds <= 0 or terminate_timeout_seconds < 0:
        raise ValueError("evaluator polling and termination timeouts are invalid")

    def stop_process(process: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except (AttributeError, OSError):
                try:
                    process.terminate()
                except OSError:
                    process.wait()
                    return
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.wait()
                return
        try:
            process.wait(timeout=terminate_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.wait()
                return
        process.wait()

    def evaluate(checkpoint: Path, cycle: int) -> None:
        expanded = [
            part.replace("{checkpoint}", str(checkpoint)).replace("{cycle}", str(cycle))
            for part in argv
        ]
        if stop_controller is not None and stop_controller.event.is_set():
            raise RuntimeError("periodic evaluation interrupted by stop request")
        popen_kwargs = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        process = subprocess.Popen(expanded, **popen_kwargs)
        while True:
            if stop_controller is not None and stop_controller.event.is_set():
                stop_process(process)
                raise RuntimeError("periodic evaluation interrupted by stop request")
            try:
                return_code = process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                continue
        if return_code:
            raise subprocess.CalledProcessError(return_code, expanded)

    return evaluate
