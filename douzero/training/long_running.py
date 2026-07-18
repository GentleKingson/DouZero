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
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, current_thread, main_thread
from typing import Callable


_RUN_STATE_VERSION = 2


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

    def resume_identity(self) -> dict[str, int]:
        """Return fields that affect the mathematical N+M trajectory."""
        return {
            "episodes_per_cycle": self.episodes_per_cycle,
            "optimizer_steps_per_cycle": self.optimizer_steps_per_cycle,
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
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_state_version: int = _RUN_STATE_VERSION
    cycle_identity: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "LongRunningState":
        if not isinstance(payload, dict):
            raise ValueError("checkpoint long_running_state must be a dict")
        migrated = dict(payload)
        if migrated.get("run_state_version") == 1:
            migrated["run_state_version"] = _RUN_STATE_VERSION
            migrated.setdefault("run_id", "legacy")
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

    def ensure_available(self, state: LongRunningState) -> None:
        """Fail closed when a fresh run would reuse an existing series."""
        if state.cycle or state.checkpoint_sequence:
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

    def publish(self, trainer, state: LongRunningState) -> Path:
        sequence = state.checkpoint_sequence + 1
        destination = self.cycle_path(
            state.run_id, sequence, state.cycle, state.total_optimizer_steps
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {destination}")
        checkpoint_state = asdict(state)
        checkpoint_state["checkpoint_sequence"] = sequence
        trainer.save_training_checkpoint(str(destination), long_running_state=checkpoint_state)
        manifest = {
            "schema_version": 1,
            "latest": destination.name,
            "cycle": state.cycle,
            "total_episodes": state.total_episodes,
            "total_optimizer_steps": state.total_optimizer_steps,
            "policy_version": state.policy_version,
            "policy_step": state.policy_step,
            "checkpoint_sequence": sequence,
            "run_id": state.run_id,
        }
        self._write_json_atomic(self.latest_manifest, manifest)
        state.checkpoint_sequence = sequence
        self._rotate(destination)
        return destination

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
            self.base.parent.glob(pattern), key=lambda path: path.name, reverse=True
        )
        for old in checkpoints[self.keep_last:]:
            if old != current:
                old.unlink()


class RunMetricsWriter:
    """Append cycle records once and atomically maintain a constant-size summary."""

    def __init__(self, summary_path: str, *, run_id: str, resume: bool) -> None:
        self.summary_path = Path(summary_path)
        self.run_id = run_id
        suffix = self.summary_path.suffix or ".json"
        stem = (
            self.summary_path.name[:-len(suffix)]
            if self.summary_path.name.endswith(suffix)
            else self.summary_path.name
        )
        self.cycles_path = self.summary_path.with_name(f"{stem}-cycles.jsonl")
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and (self.summary_path.exists() or self.cycles_path.exists()):
            raise FileExistsError(
                f"metrics output already exists for {self.summary_path}; resume "
                "the run or choose a new --metrics_path"
            )
        if resume and self.summary_path.exists():
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
            if payload.get("run_id") != run_id:
                raise ValueError(
                    "metrics run_id does not match the resumed checkpoint"
                )

    def write_cycle(self, record: dict) -> None:
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
            latest_cycle=record,
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
            latest_cycle=latest_cycle,
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
        payload = {
            "schema_version": "v2-long-running-run-v2",
            "run_id": self.run_id,
            "status": status,
            "stop_reason": stop_reason,
            "error": error,
            "cycles_path": self.cycles_path.name,
            "state": asdict(state) if state is not None else None,
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
    ) -> None:
        self.trainer = trainer
        self.config = config
        self.checkpoints = checkpoint_series
        self.stop = stop_controller or StopController()
        self.evaluator = evaluator
        self.metric_sink = metric_sink or (lambda _record: None)
        self.clock = clock
        self.peak_memory = peak_memory or (lambda: None)
        self.state = state or LongRunningState(
            total_episodes=int(trainer.stats.episodes_completed),
            total_transitions=int(trainer.stats.transitions_collected),
            total_optimizer_steps=int(trainer.stats.optimizer_steps),
            policy_version=str(trainer.policy_version),
            policy_step=int(trainer.policy_step),
            cycle_identity=config.resume_identity(),
        )
        if self.state.cycle_identity != config.resume_identity():
            raise ValueError(
                "long-running resume identity mismatch: checkpoint has "
                f"{self.state.cycle_identity!r}, runtime expects {config.resume_identity()!r}"
            )
        self._validate_totals()
        self.checkpoints.ensure_available(self.state)

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

    def _limit_reason(self, elapsed: float) -> str:
        cfg, state = self.config, self.state
        if cfg.max_cycles and state.cycle >= cfg.max_cycles:
            return "max_cycles"
        if cfg.max_total_episodes and state.total_episodes >= cfg.max_total_episodes:
            return "max_total_episodes"
        if (cfg.max_total_optimizer_steps and
                state.total_optimizer_steps >= cfg.max_total_optimizer_steps):
            return "max_total_optimizer_steps"
        if cfg.max_wall_time_minutes and elapsed >= cfg.max_wall_time_minutes * 60.0:
            return "max_wall_time"
        return ""

    def run(self) -> tuple[LongRunningState, str, list[dict]]:
        started = self.clock()
        last_checkpoint_time = started
        last_checkpoint_steps = self.state.total_optimizer_steps
        last_checkpointed_cycle = (
            self.state.cycle if self.state.checkpoint_sequence else 0
        )
        records: list[dict] = []
        reason = ""
        self.stop.install()
        try:
            while True:
                elapsed = self.clock() - started
                reason = self._limit_reason(elapsed)
                if self.stop.event.is_set():
                    reason = self.stop.reason or "stop_event"
                if reason:
                    interrupt = reason in {"sigint", "sigterm", "stop_event"}
                    save_dirty_boundary = (
                        self.state.cycle > last_checkpointed_cycle
                        and (not interrupt or self.config.save_on_interrupt)
                    )
                    if save_dirty_boundary:
                        checkpoint_path = None
                        checkpoint_error = ""
                        try:
                            checkpoint_path = self.checkpoints.publish(
                                self.trainer, self.state
                            )
                            last_checkpointed_cycle = self.state.cycle
                        except Exception as exc:
                            checkpoint_error = f"{type(exc).__name__}: {exc}"
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
                            "checkpoint_path": str(checkpoint_path or ""),
                            "checkpoint_status": (
                                "saved" if checkpoint_path else "failed"
                            ),
                            "checkpoint_error": checkpoint_error,
                            "resume_source": self.state.resume_source,
                            "peak_memory_bytes": self.peak_memory(),
                            "evaluation_status": "not_due",
                            "evaluation_error": "",
                            "stop_reason": reason,
                        }
                        records.append(record)
                        self.metric_sink(record)
                        if checkpoint_error:
                            raise RuntimeError(
                                f"checkpoint publication failed: {checkpoint_error}"
                            )
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

                self.state.cycle += 1
                self.state.total_episodes = int(self.trainer.stats.episodes_completed)
                self.state.total_transitions = int(self.trainer.stats.transitions_collected)
                self.state.total_optimizer_steps = int(self.trainer.stats.optimizer_steps)
                self.state.policy_step = int(self.trainer.policy_step)
                self.state.policy_version = str(self.trainer.policy_version)

                now = self.clock()
                boundary_reason = self._limit_reason(now - started)
                if self.stop.event.is_set():
                    boundary_reason = self.stop.reason or "stop_event"
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
                        checkpoint_status = "saved"
                        last_checkpoint_time = self.clock()
                        last_checkpoint_steps = self.state.total_optimizer_steps
                        last_checkpointed_cycle = self.state.cycle
                    except Exception as exc:
                        checkpoint_status = "failed"
                        checkpoint_error = f"{type(exc).__name__}: {exc}"
                # Saving clears replay. Non-save boundaries deliberately do so too.
                self.trainer.buffer.clear()
                self.trainer.bidding_buffer.clear()

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
                            eval_status = "failed"
                            eval_error = f"{type(exc).__name__}: {exc}"

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
                    "amp_fallback": int(self.trainer.stats.amp_fallbacks) - amp_before,
                    "checkpoint_path": str(checkpoint_path or ""),
                    "checkpoint_status": checkpoint_status,
                    "checkpoint_error": checkpoint_error,
                    "resume_source": self.state.resume_source,
                    "peak_memory_bytes": self.peak_memory(),
                    "evaluation_status": eval_status,
                    "evaluation_error": eval_error,
                    "stop_reason": boundary_reason,
                }
                records.append(record)
                self.metric_sink(record)
                if checkpoint_status == "failed":
                    raise RuntimeError(f"checkpoint publication failed: {checkpoint_error}")
                if eval_status == "failed" and self.config.eval_fail_fast:
                    raise RuntimeError(f"periodic evaluation failed: {eval_error}")
                if boundary_reason:
                    reason = boundary_reason
                    break
        finally:
            self.stop.restore()
        return self.state, reason, records


def command_evaluator(command: str) -> Callable[[Path, int], None]:
    """Build a no-shell callback for an existing evaluation CLI command."""
    import shlex

    argv = shlex.split(command)
    if not argv:
        raise ValueError("eval_command must not be empty")

    def evaluate(checkpoint: Path, cycle: int) -> None:
        expanded = [
            part.replace("{checkpoint}", str(checkpoint)).replace("{cycle}", str(cycle))
            for part in argv
        ]
        subprocess.run(expanded, check=True)

    return evaluate
