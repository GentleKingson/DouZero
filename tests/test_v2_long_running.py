from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event

import pytest
import torch

from douzero.training.long_running import (
    CheckpointSeries,
    LongRunningConfig,
    LongRunningState,
    LongRunningTrainer,
    RunMetricsWriter,
    StopController,
    command_evaluator,
)


@dataclass
class _Stats:
    episodes_completed: int = 0
    transitions_collected: int = 0
    optimizer_steps: int = 0
    amp_fallbacks: int = 0


class _Buffer(list):
    def clear(self):
        super().clear()


class _DeterministicTrainer:
    """Small optimizer-bearing trainer used to pin orchestration semantics."""

    def __init__(self, seed: int = 9, stop: StopController | None = None):
        torch.manual_seed(seed)
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.8
        )
        self.rng = random.Random(seed)
        self.stats = _Stats()
        self.policy_version = "test-policy"
        self.policy_step = 0
        self.buffer = _Buffer()
        self.bidding_buffer = _Buffer()
        self.stop = stop
        self.fail_save = False

    def collect_episodes(self, count: int):
        for _ in range(count):
            value = self.rng.random()
            self.buffer.append(value)
            self.stats.episodes_completed += 1
            self.stats.transitions_collected += 1
        if self.stop is not None:
            self.stop.request("stop_event")

    def step(self):
        if not self.buffer:
            return None
        value = self.buffer[self.stats.optimizer_steps % len(self.buffer)]
        loss = (self.model(torch.tensor([[value]])).sum() - value) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.stats.optimizer_steps += 1
        self.policy_step += 1
        return loss

    def save_training_checkpoint(self, path: str, *, long_running_state=None):
        if self.fail_save:
            raise OSError("injected checkpoint failure")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rng": self.rng.getstate(),
                "stats": asdict(self.stats),
                "policy_version": self.policy_version,
                "policy_step": self.policy_step,
                "long_running_state": long_running_state,
            },
            temporary,
        )
        os.replace(temporary, destination)
        self.buffer.clear()
        self.bidding_buffer.clear()

    def load_training_checkpoint(self, path: str):
        payload = torch.load(path, weights_only=True)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.rng.setstate(payload["rng"])
        self.stats = _Stats(**payload["stats"])
        self.policy_version = payload["policy_version"]
        self.policy_step = payload["policy_step"]
        return payload["long_running_state"]


def _run(tmp_path, trainer, *, cycles, state=None, stop=None, keep=3):
    config = LongRunningConfig(
        episodes_per_cycle=2,
        optimizer_steps_per_cycle=2,
        max_cycles=cycles,
        checkpoint_every_cycles=1,
        keep_last_checkpoints=keep,
    )
    runner = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "train.pt"), keep),
        state=state,
        stop_controller=stop,
        collect_records=True,
    )
    return runner.run()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_long_loop_counts_metrics_and_rotates_atomically(tmp_path):
    trainer = _DeterministicTrainer()
    state, reason, records = _run(tmp_path, trainer, cycles=4, keep=2)
    assert reason == "max_cycles"
    assert (state.cycle, state.total_episodes, state.total_optimizer_steps) == (4, 8, 8)
    assert state.total_transitions == 8
    assert state.policy_step == 8
    assert len(records) == 4
    assert all(record["checkpoint_status"] == "saved" for record in records)
    checkpoints = list(tmp_path.glob("train-run-*-seq-*-cycle-*-step-*.pt"))
    assert len(checkpoints) == 2
    latest = json.loads((tmp_path / "train-latest.json").read_text())
    assert latest["cycle"] == 4
    assert (tmp_path / latest["latest"]).is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_episode_and_optimizer_limits_clip_final_cycle(tmp_path):
    trainer = _DeterministicTrainer()
    config = LongRunningConfig(
        episodes_per_cycle=2,
        optimizer_steps_per_cycle=2,
        max_total_episodes=3,
        max_total_optimizer_steps=3,
    )
    state, reason, _ = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "limited.pt"), 3),
    ).run()
    assert reason in {"max_total_episodes", "max_total_optimizer_steps"}
    assert state.total_episodes == 3
    assert state.total_optimizer_steps == 3


def test_wall_time_stops_after_current_boundary(tmp_path):
    class AdvancingClock:
        value = -0.2

        def __call__(self):
            self.value += 0.2
            return self.value

    trainer = _DeterministicTrainer()
    config = LongRunningConfig(
        episodes_per_cycle=1,
        optimizer_steps_per_cycle=1,
        max_wall_time_minutes=0.01,
    )
    state, reason, records = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "timed.pt"), 3),
        clock=AdvancingClock(),
        collect_records=True,
    ).run()
    assert reason == "max_wall_time"
    assert state.cycle == 1
    assert records[-1]["stop_reason"] == "max_wall_time"
    assert records[-1]["checkpoint_status"] == "saved"


def test_wall_time_budget_accumulates_across_resume(tmp_path):
    class AdvancingClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            self.value += 1.0
            return self.value

    trainer = _DeterministicTrainer()
    first = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_cycles=1,
            max_wall_time_minutes=10,
        ),
        CheckpointSeries(str(tmp_path / "wall.pt"), 2),
        clock=AdvancingClock(),
    )
    state, reason, _ = first.run()
    assert reason == "max_cycles"
    saved_wall_seconds = state.total_wall_seconds
    manifest = json.loads((tmp_path / "wall-latest.json").read_text())
    checkpoint = tmp_path / manifest["latest"]

    resumed = _DeterministicTrainer(seed=999)
    resumed_state = LongRunningState.from_dict(
        resumed.load_training_checkpoint(str(checkpoint))
    )
    resumed_state.resume_source = str(checkpoint)
    second = LongRunningTrainer(
        resumed,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_wall_time_minutes=(saved_wall_seconds + 0.5) / 60.0,
        ),
        CheckpointSeries(str(tmp_path / "wall.pt"), 2),
        state=resumed_state,
        clock=AdvancingClock(),
    )
    final_state, reason, _ = second.run()
    assert reason == "max_wall_time"
    assert final_state.cycle == 1
    assert final_state.total_wall_seconds > saved_wall_seconds


def test_wall_time_persists_checkpoint_and_evaluation_work(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    class ManualClock:
        value = 0.0

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds

    clock = ManualClock()
    trainer = _DeterministicTrainer()
    original_save = trainer.save_training_checkpoint

    def slow_save(*args, **kwargs):
        original_save(*args, **kwargs)
        clock.advance(5.0)

    def slow_evaluate(_checkpoint, _cycle):
        clock.advance(7.0)

    trainer.save_training_checkpoint = slow_save
    state, reason, _ = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_cycles=1,
            eval_every_cycles=1,
        ),
        CheckpointSeries(str(tmp_path / "services.pt"), 2),
        evaluator=slow_evaluate,
        clock=clock,
    ).run()
    assert reason == "max_cycles"
    assert state.total_wall_seconds == 12.0
    manifest_path = tmp_path / "services-latest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["total_wall_seconds"] == 12.0
    resolved = _resolve_resume_checkpoint(str(manifest_path))
    assert resolved.total_wall_seconds == 12.0


def test_unreachable_optimizer_only_limit_fails_closed(tmp_path):
    trainer = _DeterministicTrainer()
    with pytest.raises(ValueError, match="unreachable"):
        LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=1,
                optimizer_steps_per_cycle=0,
                max_total_optimizer_steps=1,
                checkpoint_every_cycles=0,
            ),
            CheckpointSeries(str(tmp_path / "unreachable.pt"), 1),
        )


def test_zero_optimizer_steps_is_allowed_with_reachable_cycle_limit(tmp_path):
    state, reason, _ = LongRunningTrainer(
        _DeterministicTrainer(),
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=0,
            max_total_optimizer_steps=1,
            max_cycles=1,
        ),
        CheckpointSeries(str(tmp_path / "reachable.pt"), 1),
    ).run()
    assert reason == "max_cycles"
    assert state.cycle == 1


def test_stop_event_finishes_cycle_and_saves_boundary(tmp_path):
    stop = StopController(Event())
    trainer = _DeterministicTrainer(stop=stop)
    state, reason, records = _run(tmp_path, trainer, cycles=10, stop=stop)
    assert reason == "stop_event"
    assert state.cycle == 1
    assert state.total_optimizer_steps == 2
    assert records[-1]["checkpoint_status"] == "saved"
    assert not trainer.buffer and not trainer.bidding_buffer


def test_stop_arriving_in_metric_sink_saves_completed_dirty_boundary(tmp_path):
    stop = StopController(Event())
    trainer = _DeterministicTrainer()
    records = []

    def sink(record):
        records.append(record)
        if record["event"] == "cycle":
            stop.request("stop_event")

    runner = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            checkpoint_every_cycles=0,
        ),
        CheckpointSeries(str(tmp_path / "late.pt"), 2),
        stop_controller=stop,
        metric_sink=sink,
        collect_records=True,
    )
    state, reason, _ = runner.run()
    assert reason == "stop_event"
    assert state.cycle == 1
    assert records[-1]["event"] == "late_stop_checkpoint"
    assert records[-1]["checkpoint_status"] == "saved"
    latest = json.loads((tmp_path / "late-latest.json").read_text())
    assert latest["cycle"] == 1


def test_no_save_on_interrupt_does_not_cancel_scheduled_checkpoint(tmp_path):
    stop = StopController(Event())
    trainer = _DeterministicTrainer(stop=stop)
    config = LongRunningConfig(
        episodes_per_cycle=1,
        optimizer_steps_per_cycle=1,
        checkpoint_every_cycles=1,
        save_on_interrupt=False,
    )
    _state, reason, records = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "scheduled.pt"), 2),
        stop_controller=stop,
        collect_records=True,
    ).run()
    assert reason == "stop_event"
    assert records[-1]["checkpoint_status"] == "saved"


def test_no_save_on_interrupt_still_saves_checkpoint_required_for_eval(tmp_path):
    stop = StopController(Event())
    trainer = _DeterministicTrainer(stop=stop)
    evaluated = []
    config = LongRunningConfig(
        episodes_per_cycle=1,
        optimizer_steps_per_cycle=0,
        checkpoint_every_cycles=0,
        save_on_interrupt=False,
        eval_every_cycles=1,
    )
    _state, reason, records = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "evaluated.pt"), 2),
        stop_controller=stop,
        evaluator=lambda checkpoint, cycle: evaluated.append((checkpoint, cycle)),
        collect_records=True,
    ).run()
    assert reason == "stop_event"
    assert records[-1]["checkpoint_status"] == "saved"
    assert records[-1]["evaluation_status"] == "passed"
    assert evaluated[0][1] == 1


@pytest.mark.parametrize("fail_fast", [True, False])
def test_evaluation_failure_is_recorded_and_configurable(tmp_path, fail_fast):
    trainer = _DeterministicTrainer()
    config = LongRunningConfig(
        episodes_per_cycle=1,
        optimizer_steps_per_cycle=0,
        max_cycles=1,
        eval_every_cycles=1,
        eval_fail_fast=fail_fast,
    )
    records = []

    def failing_eval(_checkpoint, _cycle):
        raise OSError("evaluation unavailable")

    runner = LongRunningTrainer(
        trainer,
        config,
        CheckpointSeries(str(tmp_path / "eval.pt"), 2),
        evaluator=failing_eval,
        metric_sink=records.append,
    )
    if fail_fast:
        with pytest.raises(RuntimeError, match="periodic evaluation failed"):
            runner.run()
    else:
        _state, reason, _ = runner.run()
        assert reason == "max_cycles"
    assert records[-1]["evaluation_status"] == "failed"
    assert records[-1]["evaluation_error"] == "OSError"


def test_signal_during_evaluation_stops_without_fail_fast_error(tmp_path):
    stop = StopController(Event())

    def interrupted_evaluation(_checkpoint, _cycle):
        stop.request("sigint")
        raise OSError("evaluation subprocess interrupted")

    state, reason, records = LongRunningTrainer(
        _DeterministicTrainer(),
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=0,
            max_cycles=10,
            eval_every_cycles=1,
            eval_fail_fast=True,
        ),
        CheckpointSeries(str(tmp_path / "eval-signal.pt"), 2),
        stop_controller=stop,
        evaluator=interrupted_evaluation,
        collect_records=True,
    ).run()
    assert reason == "sigint"
    assert state.cycle == 1
    assert records[-1]["evaluation_status"] == "interrupted"
    assert records[-1]["evaluation_error"] == ""


def test_resume_identity_mismatch_fails_closed(tmp_path):
    trainer = _DeterministicTrainer()
    state = LongRunningState(
        policy_version=trainer.policy_version,
        cycle_identity={"episodes_per_cycle": 99, "optimizer_steps_per_cycle": 2},
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(tmp_path, trainer, cycles=1, state=state)


def test_checkpoint_failure_keeps_previous_valid_checkpoint_and_manifest(tmp_path):
    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path, trainer, cycles=1)
    manifest_path = tmp_path / "train-latest.json"
    previous_manifest = manifest_path.read_bytes()
    previous_checkpoint = tmp_path / json.loads(previous_manifest)["latest"]
    trainer.fail_save = True
    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        _run(tmp_path, trainer, cycles=2, state=state)
    assert manifest_path.read_bytes() == previous_manifest
    assert previous_checkpoint.is_file()


def test_checkpoint_failure_retains_boundary_stop_reason(tmp_path):
    trainer = _DeterministicTrainer()
    trainer.fail_save = True
    runner = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_cycles=1,
        ),
        CheckpointSeries(str(tmp_path / "failed.pt"), 1),
    )
    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        runner.run()
    assert runner.last_stop_reason == "max_cycles"


def test_manifest_failure_is_reconciled_from_the_next_orphan(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path, trainer, cycles=1)
    manifest_path = tmp_path / "train-latest.json"
    previous_manifest = manifest_path.read_bytes()
    series = CheckpointSeries(str(tmp_path / "train.pt"), 3)
    original_write = series._write_json_atomic

    def fail_manifest(path, payload):
        if path == series.latest_manifest:
            raise OSError("injected manifest failure")
        original_write(path, payload)

    series._write_json_atomic = fail_manifest
    runner = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=2,
            optimizer_steps_per_cycle=2,
            max_cycles=2,
            checkpoint_every_cycles=1,
        ),
        series,
        state=state,
    )
    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        runner.run()
    assert manifest_path.read_bytes() == previous_manifest

    resolved = _resolve_resume_checkpoint(str(manifest_path))
    assert resolved.checkpoint != str(
        tmp_path / json.loads(previous_manifest)["latest"]
    )
    resumed = _DeterministicTrainer(seed=999)
    payload = resumed.load_training_checkpoint(resolved.checkpoint)
    resumed_state = LongRunningState.from_dict(payload)
    resumed_state.resume_source = resolved.checkpoint
    final_state, reason, _ = LongRunningTrainer(
        resumed,
        LongRunningConfig(
            episodes_per_cycle=2,
            optimizer_steps_per_cycle=2,
            max_cycles=2,
        ),
        CheckpointSeries(resolved.series_base, 3),
        state=resumed_state,
    ).run()
    assert reason == "max_cycles"
    assert final_state.checkpoint_sequence == 2
    assert json.loads(manifest_path.read_text())["checkpoint_sequence"] == 2


def test_first_manifest_failure_recovers_unique_initial_checkpoint(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    trainer = _DeterministicTrainer()
    series = CheckpointSeries(str(tmp_path / "initial.pt"), 3)
    original_write = series._write_json_atomic

    def fail_latest(path, payload):
        if path == series.latest_manifest:
            raise OSError("injected initial manifest failure")
        original_write(path, payload)

    series._write_json_atomic = fail_latest
    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=1,
                optimizer_steps_per_cycle=1,
                max_cycles=1,
            ),
            series,
        ).run()

    checkpoint = next(tmp_path.glob("initial-run-*-seq-000001-*.pt"))
    assert not (tmp_path / "initial-latest.json").exists()
    resolved = _resolve_resume_checkpoint(str(checkpoint))
    resumed = _DeterministicTrainer(seed=999)
    state = LongRunningState.from_dict(
        resumed.load_training_checkpoint(resolved.checkpoint)
    )
    state.resume_source = resolved.checkpoint
    LongRunningTrainer(
        resumed,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_cycles=1,
        ),
        CheckpointSeries(resolved.series_base, 3),
        state=state,
    )
    manifest = json.loads((tmp_path / "initial-latest.json").read_text())
    assert manifest["checkpoint_sequence"] == 1


def test_rotation_failure_does_not_invalidate_published_checkpoint(tmp_path):
    trainer = _DeterministicTrainer()
    series = CheckpointSeries(str(tmp_path / "rotation.pt"), 1)

    def fail_rotation(_current):
        raise OSError("injected rotation failure")

    series._rotate = fail_rotation
    state, reason, records = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=1,
            max_cycles=1,
        ),
        series,
        collect_records=True,
    ).run()
    assert reason == "max_cycles"
    assert state.checkpoint_sequence == 1
    assert records[-1]["checkpoint_status"] == "saved_rotation_failed"
    assert records[-1]["checkpoint_error"] == "OSError"
    latest = json.loads((tmp_path / "rotation-latest.json").read_text())
    assert (tmp_path / latest["latest"]).is_file()


def test_fresh_run_refuses_to_reuse_existing_checkpoint_series(tmp_path):
    trainer = _DeterministicTrainer()
    _run(tmp_path, trainer, cycles=1)
    manifest_path = tmp_path / "train-latest.json"
    previous_manifest = manifest_path.read_bytes()
    previous_checkpoint = tmp_path / json.loads(previous_manifest)["latest"]
    previous_bytes = previous_checkpoint.read_bytes()

    with pytest.raises(FileExistsError, match="checkpoint series already exists"):
        _run(tmp_path, _DeterministicTrainer(), cycles=1)

    assert manifest_path.read_bytes() == previous_manifest
    assert previous_checkpoint.read_bytes() == previous_bytes


def test_resume_rejects_checkpoint_from_a_different_series(tmp_path):
    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path / "a", trainer, cycles=1)
    manifest = json.loads((tmp_path / "a" / "train-latest.json").read_text())
    checkpoint = tmp_path / "a" / manifest["latest"]
    state.resume_source = str(checkpoint)
    with pytest.raises(ValueError, match="does not belong"):
        LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=2,
                optimizer_steps_per_cycle=2,
                max_cycles=2,
            ),
            CheckpointSeries(str(tmp_path / "b" / "train.pt"), 3),
            state=state,
        )


def test_copied_checkpoint_without_publication_intent_cannot_fork_series(tmp_path):
    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path / "source", trainer, cycles=1)
    manifest = json.loads(
        (tmp_path / "source" / "train-latest.json").read_text()
    )
    original = tmp_path / "source" / manifest["latest"]
    copied = tmp_path / "copy" / original.name
    copied.parent.mkdir()
    copied.write_bytes(original.read_bytes())
    state.resume_source = str(copied)
    with pytest.raises(ValueError, match="publication intent"):
        LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=2,
                optimizer_steps_per_cycle=2,
                max_cycles=2,
            ),
            CheckpointSeries(str(tmp_path / "copy" / "train.pt"), 3),
            state=state,
        )


def test_resume_rejects_rollback_behind_latest_sequence(tmp_path):
    trainer = _DeterministicTrainer()
    _run(tmp_path, trainer, cycles=2)
    checkpoints = sorted(tmp_path.glob("train-run-*-seq-*-cycle-*-step-*.pt"))
    older = checkpoints[0]
    resumed = _DeterministicTrainer(seed=999)
    state = LongRunningState.from_dict(resumed.load_training_checkpoint(str(older)))
    state.resume_source = str(older)
    with pytest.raises(ValueError, match="older checkpoint"):
        LongRunningTrainer(
            resumed,
            LongRunningConfig(
                episodes_per_cycle=2,
                optimizer_steps_per_cycle=2,
                max_cycles=3,
            ),
            CheckpointSeries(str(tmp_path / "train.pt"), 3),
            state=state,
        )


def test_direct_old_checkpoint_resolves_unique_newer_orphan(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path, trainer, cycles=1)
    old = tmp_path / json.loads(
        (tmp_path / "train-latest.json").read_text()
    )["latest"]
    series = CheckpointSeries(str(tmp_path / "train.pt"), 3)
    original_write = series._write_json_atomic

    def fail_latest(path, payload):
        if path == series.latest_manifest:
            raise OSError("injected manifest failure")
        original_write(path, payload)

    series._write_json_atomic = fail_latest
    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=2,
                optimizer_steps_per_cycle=2,
                max_cycles=2,
            ),
            series,
            state=state,
        ).run()
    orphan = next(tmp_path.glob("train-run-*-seq-000002-*.pt"))
    resolved = _resolve_resume_checkpoint(str(old))
    assert Path(resolved.checkpoint) == orphan


def test_duplicate_checkpoint_sequence_fails_closed(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    _run(tmp_path, _DeterministicTrainer(), cycles=1)
    manifest_path = tmp_path / "train-latest.json"
    payload = json.loads(manifest_path.read_text())
    original = tmp_path / payload["latest"]
    duplicate = original.with_name(
        original.name.replace("cycle-000001", "cycle-000002")
    )
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="duplicate sequence"):
        _resolve_resume_checkpoint(str(manifest_path))


def test_manifest_counters_must_match_checkpoint_state(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    _run(tmp_path, _DeterministicTrainer(), cycles=1)
    manifest_path = tmp_path / "train-latest.json"
    payload = json.loads(manifest_path.read_text())
    payload["total_episodes"] += 1
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match checkpoint state"):
        _resolve_resume_checkpoint(str(manifest_path))


def test_manifest_resume_reuses_series_when_checkpoint_path_is_omitted(tmp_path):
    from train_v2 import (
        _resolve_resume_checkpoint,
        _select_long_running_checkpoint_path,
    )

    _run(tmp_path, _DeterministicTrainer(), cycles=1)
    resolved = _resolve_resume_checkpoint(str(tmp_path / "train-latest.json"))
    assert _select_long_running_checkpoint_path("", resolved) == str(
        tmp_path / "train.pt"
    )
    with pytest.raises(ValueError, match="does not match"):
        _select_long_running_checkpoint_path(str(tmp_path / "other.pt"), resolved)


def test_sequence_ordering_crosses_one_million_numerically(tmp_path):
    trainer = _DeterministicTrainer()
    state = LongRunningState(
        cycle=999_998,
        checkpoint_sequence=999_998,
        policy_version=trainer.policy_version,
        cycle_identity={"episodes_per_cycle": 1, "optimizer_steps_per_cycle": 0},
    )
    series = CheckpointSeries(str(tmp_path / "large.pt"), 2)
    first = series.publish(trainer, state)
    assert "seq-999999" in first.name

    state.cycle += 1
    original_write = series._write_json_atomic

    def fail_manifest(path, payload):
        if path == series.latest_manifest:
            raise OSError("injected manifest failure")
        original_write(path, payload)

    series._write_json_atomic = fail_manifest
    with pytest.raises(OSError, match="manifest failure"):
        series.publish(trainer, state)
    orphan = next(tmp_path.glob("large-run-*-seq-1000000-*.pt"))
    assert series.checkpoint_sequence(orphan, run_id=state.run_id) == 1_000_000

    from train_v2 import _resolve_resume_checkpoint

    resolved = _resolve_resume_checkpoint(str(tmp_path / "large-latest.json"))
    assert Path(resolved.checkpoint) == orphan


def test_rotation_orders_sequences_numerically_past_six_digits(tmp_path):
    series = CheckpointSeries(str(tmp_path / "rotate-large.pt"), 1)
    run_id = "numeric"
    older = series.cycle_path(run_id, 999_999, 1, 1)
    newest = series.cycle_path(run_id, 1_000_000, 2, 2)
    older.write_bytes(b"older")
    newest.write_bytes(b"newest")
    series._rotate(newest)
    assert newest.is_file()
    assert not older.exists()


def test_binary_json_checkpoint_is_not_misclassified_as_manifest(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    checkpoint = tmp_path / "model-latest.json"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint)
    resolved = _resolve_resume_checkpoint(str(checkpoint))
    assert resolved.checkpoint == str(checkpoint)
    assert not resolved.manifest


def test_one_shot_v2_checkpoint_with_json_suffix_roundtrips(tmp_path):
    from train_v2 import _resolve_resume_checkpoint

    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    config = ModelV2Config(hidden_size=16, history_layers=1, history_heads=1)
    trainer_config = TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=8)
    trainer = V2Trainer(ModelV2(build_v2_schema(), config), config=trainer_config)
    checkpoint = tmp_path / "one-shot.json"
    trainer.save_training_checkpoint(str(checkpoint))
    resolved = _resolve_resume_checkpoint(str(checkpoint))
    restored = V2Trainer(
        ModelV2(build_v2_schema(), config), config=trainer_config
    )
    identity = restored.load_training_checkpoint(resolved.checkpoint)
    assert identity["long_running_state"] is None


def test_n_plus_m_resume_matches_uninterrupted_model_and_optimizer(tmp_path):
    continuous = _DeterministicTrainer(seed=71)
    continuous_state, _, _ = _run(tmp_path / "continuous", continuous, cycles=4)

    first = _DeterministicTrainer(seed=71)
    first_state, _, _ = _run(tmp_path / "split", first, cycles=2)
    manifest = json.loads((tmp_path / "split" / "train-latest.json").read_text())
    checkpoint = tmp_path / "split" / manifest["latest"]

    resumed = _DeterministicTrainer(seed=999)
    state_payload = resumed.load_training_checkpoint(str(checkpoint))
    resumed_state = LongRunningState.from_dict(state_payload)
    resumed_state.resume_source = str(checkpoint)
    final_state, _, _ = _run(
        tmp_path / "split", resumed, cycles=4, state=resumed_state
    )

    final_payload = asdict(final_state)
    continuous_payload = asdict(continuous_state)
    final_payload["resume_source"] = ""
    final_payload["run_id"] = continuous_payload["run_id"]
    final_payload["total_wall_seconds"] = continuous_payload["total_wall_seconds"]
    assert final_payload == continuous_payload
    _assert_nested_equal(continuous.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(
        continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
    )


def test_v2_checkpoint_roundtrip_carries_cycle_state(tmp_path):
    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    cfg = ModelV2Config(hidden_size=16, history_layers=1, history_heads=1)
    trainer_cfg = TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=4)
    trainer = V2Trainer(ModelV2(build_v2_schema(), cfg), config=trainer_cfg)
    state = LongRunningState(
        policy_version=trainer.policy_version,
        cycle_identity={"episodes_per_cycle": 2, "optimizer_steps_per_cycle": 1},
    )
    checkpoint = tmp_path / "v2.pt"
    trainer.save_training_checkpoint(
        str(checkpoint), long_running_state=asdict(state)
    )
    restored = V2Trainer(ModelV2(build_v2_schema(), cfg), config=trainer_cfg)
    identity = restored.load_training_checkpoint(str(checkpoint))
    assert LongRunningState.from_dict(identity["long_running_state"]) == state


def test_real_v2_trainer_n_plus_m_resume_matches_uninterrupted(tmp_path):
    import numpy as np

    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    model_config = ModelV2Config(hidden_size=16, history_layers=1, history_heads=1)
    trainer_config = TrainerConfig(
        max_episodes=0,
        optimizer_steps=0,
        batch_size=1,
        buffer_capacity=512,
        rng_seed=37,
        exp_epsilon=1.0,
    )

    def make_trainer(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return V2Trainer(
            ModelV2(build_v2_schema(), model_config), config=trainer_config
        )

    def run_cycles(directory, trainer, cycles, state=None):
        return LongRunningTrainer(
            trainer,
            LongRunningConfig(
                episodes_per_cycle=1,
                optimizer_steps_per_cycle=1,
                max_cycles=cycles,
                checkpoint_every_cycles=1,
            ),
            CheckpointSeries(str(directory / "train.pt"), 3),
            state=state,
        ).run()[0]

    continuous = make_trainer(101)
    continuous_state = run_cycles(tmp_path / "continuous", continuous, 2)

    first = make_trainer(101)
    run_cycles(tmp_path / "split", first, 1)
    manifest = json.loads((tmp_path / "split" / "train-latest.json").read_text())
    checkpoint = tmp_path / "split" / manifest["latest"]
    resumed = make_trainer(999)
    identity = resumed.load_training_checkpoint(str(checkpoint))
    resumed_state = LongRunningState.from_dict(identity["long_running_state"])
    resumed_state.resume_source = str(checkpoint)
    final_state = run_cycles(tmp_path / "split", resumed, 2, resumed_state)

    assert final_state.cycle == continuous_state.cycle == 2
    assert final_state.total_episodes == continuous_state.total_episodes == 2
    assert (
        final_state.total_optimizer_steps
        == continuous_state.total_optimizer_steps
        == 2
    )
    assert final_state.policy_step == continuous_state.policy_step
    _assert_nested_equal(continuous.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(
        continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["max_wall_time_minutes", "checkpoint_every_minutes"])
def test_time_configuration_rejects_nonfinite_values(field, value):
    arguments = {
        "episodes_per_cycle": 1,
        "optimizer_steps_per_cycle": 0,
        field: value,
    }
    with pytest.raises(ValueError, match="finite"):
        LongRunningConfig(**arguments)


def test_metrics_are_append_only_across_resume_and_finalize_failure(tmp_path):
    summary = tmp_path / "metrics.json"
    writer = RunMetricsWriter(str(summary), run_id="run-1", resume=False)
    writer.write_cycle({"cycle": 1, "stop_reason": ""})
    state = LongRunningState(
        cycle=1, resume_source=str(tmp_path / "private" / "checkpoint.pt")
    )
    writer.finalize(status="stopped", stop_reason="max_cycles", state=state)

    resumed = RunMetricsWriter(str(summary), run_id="run-1", resume=True)
    resumed.write_cycle({"cycle": 2, "stop_reason": ""})
    resumed.finalize(
        status="failed",
        stop_reason="",
        state=LongRunningState(
            cycle=2, resume_source=str(tmp_path / "private" / "checkpoint.pt")
        ),
        error="RuntimeError: injected",
    )

    lines = (tmp_path / "metrics-cycles.jsonl").read_text().splitlines()
    assert [json.loads(line)["cycle"] for line in lines] == [1, 2]
    payload = json.loads(summary.read_text())
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError"
    assert "cycles" not in payload
    assert payload["state"]["resume_source"] == "checkpoint.pt"
    assert payload["latest_cycle"]["cycle"] == 2


def test_fresh_metrics_writer_refuses_to_overwrite_existing_history(tmp_path):
    summary = tmp_path / "metrics.json"
    writer = RunMetricsWriter(str(summary), run_id="run-1", resume=False)
    writer.write_cycle({"cycle": 1})
    with pytest.raises(FileExistsError, match="metrics output already exists"):
        RunMetricsWriter(str(summary), run_id="run-2", resume=False)


def test_resumed_metrics_writer_rejects_a_different_run(tmp_path):
    summary = tmp_path / "metrics.json"
    writer = RunMetricsWriter(str(summary), run_id="run-1", resume=False)
    writer.write_cycle({"cycle": 1})
    with pytest.raises(ValueError, match="run_id"):
        RunMetricsWriter(str(summary), run_id="run-2", resume=True)


def test_production_loop_streams_metrics_without_retaining_history(tmp_path):
    trainer = _DeterministicTrainer()
    streamed = 0

    def sink(_record):
        nonlocal streamed
        streamed += 1

    state, reason, records = LongRunningTrainer(
        trainer,
        LongRunningConfig(
            episodes_per_cycle=1,
            optimizer_steps_per_cycle=0,
            max_cycles=2000,
            checkpoint_every_cycles=0,
        ),
        CheckpointSeries(str(tmp_path / "bounded.pt"), 1),
        metric_sink=sink,
    ).run()
    assert reason == "max_cycles"
    assert state.cycle == streamed == 2000
    assert records == []


def test_cycle_metrics_do_not_expose_absolute_paths(tmp_path):
    trainer = _DeterministicTrainer()
    state, _, _ = _run(tmp_path, trainer, cycles=1)
    manifest = json.loads((tmp_path / "train-latest.json").read_text())
    checkpoint = tmp_path / manifest["latest"]
    resumed = _DeterministicTrainer(seed=999)
    payload = resumed.load_training_checkpoint(str(checkpoint))
    resumed_state = LongRunningState.from_dict(payload)
    resumed_state.resume_source = str(checkpoint)
    _state, _reason, records = _run(
        tmp_path, resumed, cycles=2, state=resumed_state
    )
    record = records[-1]
    assert record["checkpoint_path"] == Path(record["checkpoint_path"]).name
    assert record["resume_source"] == checkpoint.name


def test_windows_evaluator_preserves_backslashes_and_quoted_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "douzero.training.long_running.subprocess.run",
        lambda argv, check: calls.append((argv, check)),
    )
    evaluator = command_evaluator(
        '"C:\\Program Files\\Python\\python.exe" '
        '--baseline "C:\\models\\baseline.pt" --candidate {checkpoint}',
        windows=True,
    )
    evaluator(Path("candidate.pt"), 3)
    argv, check = calls[0]
    assert argv[0] == "C:\\Program Files\\Python\\python.exe"
    assert argv[2] == "C:\\models\\baseline.pt"
    assert argv[-1] == "candidate.pt"
    assert check is True


def test_evaluator_accepts_json_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "douzero.training.long_running.subprocess.run",
        lambda argv, check: calls.append(argv),
    )
    evaluator = command_evaluator(
        '["python", "evaluate.py", "--candidate", "{checkpoint}"]'
    )
    evaluator(Path("candidate.pt"), 1)
    assert calls == [["python", "evaluate.py", "--candidate", "candidate.pt"]]
