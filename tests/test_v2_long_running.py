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
    ).run()
    assert reason == "max_wall_time"
    assert state.cycle == 1
    assert records[-1]["stop_reason"] == "max_wall_time"
    assert records[-1]["checkpoint_status"] == "saved"


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
    assert "evaluation unavailable" in records[-1]["evaluation_error"]


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
    state = LongRunningState(cycle=1)
    writer.finalize(status="stopped", stop_reason="max_cycles", state=state)

    resumed = RunMetricsWriter(str(summary), run_id="run-1", resume=True)
    resumed.write_cycle({"cycle": 2, "stop_reason": ""})
    resumed.finalize(
        status="failed",
        stop_reason="",
        state=LongRunningState(cycle=2),
        error="RuntimeError: injected",
    )

    lines = (tmp_path / "metrics-cycles.jsonl").read_text().splitlines()
    assert [json.loads(line)["cycle"] for line in lines] == [1, 2]
    payload = json.loads(summary.read_text())
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: injected"
    assert "cycles" not in payload


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
