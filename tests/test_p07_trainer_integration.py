"""P07 V2Trainer belief integration tests (review round 3 blocker).

Closes the training-loop gap: a ``belief_enabled=True`` value model must be
trainable end-to-end via :class:`V2Trainer` with a frozen
:class:`BeliefModel`. The trainer computes the constrained posterior features
from each ``obs.public`` and fuses them at both the collection and optimizer
call sites.

Acceptance (the headline of this round):
- one ``V2Trainer.train()`` run completes an optimizer step;
- ``belief_proj`` parameters CHANGE (it is the trained component);
- the frozen BeliefModel parameters are UNCHANGED (frozen feature source);
- constructing a belief-enabled trainer without a belief_model fails fast;
- a belief_model paired with a belief-disabled value model fails fast.
"""

from __future__ import annotations

import numpy as np
import torch

from douzero.belief import BeliefConfig, BeliefModel
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    DecisionConfig,
    LossConfig,
    TrainerConfig,
    V2Trainer,
)


def _belief_enabled_value_model():
    torch.manual_seed(1234)
    schema = build_v2_schema()
    cfg = ModelV2Config(
        belief_enabled=True, hidden_size=32, history_heads=4,
        history_layers=2, nan_guard=False,
    )
    return ModelV2(schema, cfg)


def _belief_model():
    torch.manual_seed(5678)
    return BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))


def _trainer_config():
    return TrainerConfig(
        seed=2024, rng_seed=2024, max_episodes=3, max_steps_per_episode=400,
        exp_epsilon=0.5, batch_size=8, learning_rate=1e-3, max_grad_norm=40.0,
        optimizer_steps=1, buffer_capacity=512,
    )


class TestV2TrainerBeliefIntegration:
    def test_belief_enabled_trainer_trains_belief_proj_freezes_belief(
        self, seed_factory
    ):
        seed_factory(2024)
        value = _belief_enabled_value_model()
        belief = _belief_model()
        # Snapshots BEFORE training.
        belief_proj_before = [
            p.detach().clone() for p in value.belief_proj.parameters()
        ]
        belief_params_before = [
            p.detach().clone() for p in belief.parameters()
        ]
        # The frozen belief model must have requires_grad=False + be in eval.
        trainer = V2Trainer(
            value,
            ruleset=None,
            loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
            decision_config=DecisionConfig(mode="pure_win"),
            config=_trainer_config(),
            belief_model=belief,
        )
        assert all(not p.requires_grad for p in belief.parameters())
        assert not belief.training

        stats = trainer.train()
        # An optimizer step ran.
        assert stats.optimizer_steps >= 1

        # belief_proj changed (it is the trained fusion component).
        belief_proj_after = [
            p.detach().clone() for p in value.belief_proj.parameters()
        ]
        assert any(
            not torch.equal(b, a)
            for b, a in zip(belief_proj_after, belief_proj_before)
        ), "belief_proj did not change during the belief-enabled training step"

        # The frozen BeliefModel parameters are byte-identical.
        belief_params_after = [
            p.detach().clone() for p in belief.parameters()
        ]
        for i, (b, a) in enumerate(
            zip(belief_params_before, belief_params_after)
        ):
            assert torch.equal(b, a), (
                f"frozen BeliefModel parameter[{i}] changed during training"
            )

    def test_belief_enabled_without_belief_model_rejected(self):
        value = _belief_enabled_value_model()
        with __import__("pytest").raises(ValueError, match="belief_enabled=True"):
            V2Trainer(
                value, ruleset=None,
                loss_config=LossConfig(lambda_win=1.0),
                decision_config=DecisionConfig(mode="pure_win"),
                config=_trainer_config(),
            )

    def test_belief_model_with_belief_disabled_value_rejected(self):
        torch.manual_seed(1)
        schema = build_v2_schema()
        value = ModelV2(
            schema,
            ModelV2Config(belief_enabled=False, hidden_size=32,
                          history_heads=4, history_layers=2),
        )
        belief = _belief_model()
        with __import__("pytest").raises(ValueError, match="belief_enabled=False"):
            V2Trainer(
                value, ruleset=None,
                loss_config=LossConfig(lambda_win=1.0),
                decision_config=DecisionConfig(mode="pure_win"),
                config=_trainer_config(),
                belief_model=belief,
            )

    def test_belief_disabled_trainer_runs_unchanged(self, seed_factory):
        """A belief-disabled trainer must NOT require or touch the belief path."""
        seed_factory(2024)
        torch.manual_seed(1234)
        schema = build_v2_schema()
        value = ModelV2(
            schema,
            ModelV2Config(belief_enabled=False, hidden_size=32,
                          history_heads=4, history_layers=2, nan_guard=False),
        )
        assert value.belief_proj is None
        before = next(value.parameters()).detach().clone()
        trainer = V2Trainer(
            value, ruleset=None,
            loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
            decision_config=DecisionConfig(mode="pure_win"),
            config=_trainer_config(),
        )
        assert trainer.belief_model is None
        stats = trainer.train()
        after = next(value.parameters()).detach().clone()
        assert stats.optimizer_steps >= 1
        assert not torch.equal(before, after)
