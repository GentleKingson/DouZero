"""P06 V2 trainer smoke test (CPU).

Verifies the P06 acceptance criterion: "极短训练能完成一次优化且参数变化"
(a very short training run completes one optimizer step and changes the
parameters). Also exercises the buffer, the team-perspective labels, and
the multi-objective loss end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    DecisionConfig,
    Episode,
    LossConfig,
    TrainerConfig,
    Transition,
    V2ReplayBuffer,
    V2Trainer,
)


def _build_model():
    torch.manual_seed(1234)
    schema = build_v2_schema()
    return ModelV2(schema, ModelV2Config())


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def test_buffer_sample_returns_none_when_insufficient():
    buf = V2ReplayBuffer(capacity_transitions=10)
    # Build a tiny episode with one fake transition (no real obs needed
    # because we only test the size guard here).
    fake_transition = Transition(
        obs=None,  # type: ignore[arg-type]
        action_index=0,
        position="landlord",
        target_win=1.0,
        target_score=2.0,
        target_log_score=1.0,
    )
    ep = Episode(transitions=[fake_transition], terminal_result={})
    buf.add_episode(ep)
    import random

    assert buf.sample_minibatch(batch_size=2, rng=random.Random(0)) is None


def test_buffer_capacity_eviction():
    buf = V2ReplayBuffer(capacity_transitions=2)
    t1 = Transition(obs=None, action_index=0, position="landlord", target_win=1.0, target_score=1.0, target_log_score=0.5)  # type: ignore[arg-type]
    t2 = Transition(obs=None, action_index=0, position="landlord_up", target_win=0.0, target_score=-1.0, target_log_score=-0.5)  # type: ignore[arg-type]
    t3 = Transition(obs=None, action_index=0, position="landlord_down", target_win=1.0, target_score=1.0, target_log_score=0.5)  # type: ignore[arg-type]
    buf.add_episode(Episode(transitions=[t1], terminal_result={}))
    buf.add_episode(Episode(transitions=[t2, t3], terminal_result={}))
    # Capacity is 2; the first episode should be evicted.
    assert len(buf) == 2


# --------------------------------------------------------------------------- #
# Trainer end-to-end smoke
# --------------------------------------------------------------------------- #
def test_trainer_runs_one_optimizer_step_and_changes_params(seed_factory):
    seed_factory(2024)
    model = _build_model()
    before = next(model.parameters()).detach().clone()
    trainer = V2Trainer(
        model,
        ruleset=None,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        decision_config=DecisionConfig(mode="pure_win"),
        config=TrainerConfig(
            seed=2024,
            rng_seed=2024,
            max_episodes=3,
            max_steps_per_episode=400,
            exp_epsilon=0.5,  # mix exploration so decisions vary
            batch_size=8,
            learning_rate=1e-3,
            max_grad_norm=40.0,
            optimizer_steps=1,
            buffer_capacity=512,
        ),
    )
    stats = trainer.train()
    after = next(model.parameters()).detach().clone()
    assert stats.episodes_completed >= 1
    assert stats.optimizer_steps == 1
    # The optimizer step only happens if enough transitions were collected.
    assert stats.transitions_collected >= 8 or stats.optimizer_steps == 0
    # The defining acceptance criterion: parameters changed.
    assert not torch.equal(before, after), (
        "V2 trainer completed an optimizer step without changing parameters"
    )
    # Loss / gradient diagnostics were recorded.
    assert "loss_total" in stats.last_loss
    assert stats.grad_norm_last_step >= 0.0


def test_trainer_handles_empty_buffer_gracefully(seed_factory):
    """If too few transitions are collected, step() returns None."""
    seed_factory(2024)
    model = _build_model()
    trainer = V2Trainer(
        model,
        config=TrainerConfig(
            seed=2024,
            rng_seed=2024,
            max_episodes=0,  # collect nothing
            batch_size=8,
        ),
    )
    # collect_episodes(0) is a no-op.
    trainer.collect_episodes(0)
    assert trainer.step() is None
    assert trainer.stats.optimizer_steps == 0


def test_trainer_logs_team_distribution(seed_factory):
    seed_factory(2024)
    model = _build_model()
    trainer = V2Trainer(
        model,
        config=TrainerConfig(
            seed=2024,
            rng_seed=2024,
            max_episodes=3,
            batch_size=4,
            optimizer_steps=0,  # just collect
        ),
    )
    trainer.collect_episodes()
    total_wins = sum(trainer.stats.episodes_per_team.values())
    assert total_wins == trainer.stats.episodes_completed
