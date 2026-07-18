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
    checkpoints = list(tmp_path.glob("train-seq-*-cycle-*-step-*.pt"))
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

    assert asdict(final_state) | {"resume_source": ""} == asdict(continuous_state)
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
