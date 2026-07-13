"""P08 prior head, listwise BC loss, BC config, and pure_prior decision tests."""

from __future__ import annotations

import pytest
import torch

from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.heads import PriorHead
from douzero.models_v2.model import ModelV2
from douzero.models_v2.output import ModelOutput
from douzero.observation.schema import build_v2_schema
from douzero.training.bc_loss import (
    BCLossComponents,
    BCLossConfig,
    BCLossError,
    average_bc_losses,
    listwise_bc_loss,
)
from douzero.training.decision_policy import (
    DecisionConfig,
    SUPPORTED_DECISION_MODES,
    select_action,
)


# --------------------------------------------------------------------------- #
# PriorHead
# --------------------------------------------------------------------------- #
class TestPriorHead:
    def test_forward_shape(self):
        head = PriorHead(hidden_size=8)
        fused = torch.randn(5, 8)
        out = head(fused)
        assert out.shape == (5, 1)

    def test_rejects_wrong_hidden_size(self):
        head = PriorHead(hidden_size=8)
        with pytest.raises(ValueError):
            head(torch.randn(3, 16))

    def test_rejects_non_positive_hidden(self):
        with pytest.raises(ValueError):
            PriorHead(hidden_size=0)


# --------------------------------------------------------------------------- #
# ModelV2 wiring (prior head gated by human_prior_enabled)
# --------------------------------------------------------------------------- #
class TestModelPriorHead:
    def _build_model(self, human_prior: bool) -> ModelV2:
        cfg = ModelV2Config(
            hidden_size=32,
            history_layers=1,
            history_heads=4,
            history_encoder="lstm",
            human_prior_enabled=human_prior,
            nan_guard=False,
        )
        return ModelV2(build_v2_schema(), cfg)

    def test_prior_head_absent_when_disabled(self):
        m = self._build_model(human_prior=False)
        assert m.prior_head is None

    def test_prior_head_present_when_enabled(self):
        m = self._build_model(human_prior=True)
        assert m.prior_head is not None
        counts = m.parameter_count()
        assert "prior_head" in counts
        assert counts["prior_head"] > 0

    def test_forward_emits_prior_logit_when_enabled(self):
        """A full forward through the model produces a prior_logit tensor."""
        from douzero.models_v2.batch import observation_to_model_inputs
        from douzero.observation.encode_v2 import get_obs_v2

        # Build a tiny legacy env and grab one observation.
        from douzero.env.env import Env

        env = Env("adp")
        env.reset()
        infoset = env.infoset
        # Force at least 2 legal actions for a meaningful decision.
        legal = list(infoset.legal_actions)
        if len(legal) < 2:
            pytest.skip("env yielded a trivial single-action decision")
        obs = get_obs_v2(infoset)
        bundle = observation_to_model_inputs(obs)

        m = self._build_model(human_prior=True)
        m.eval()
        with torch.inference_mode():
            out = m(
                state_card_vectors=bundle.state_card_vectors,
                state_context_flat=bundle.state_context_flat,
                context_card_vectors=bundle.context_card_vectors,
                context_flat=bundle.context_flat,
                history_tokens=bundle.history_tokens,
                history_key_padding_mask=bundle.history_key_padding_mask,
                action_features=bundle.action_features,
                action_mask=bundle.action_mask,
                acting_role=bundle.acting_role,
            )
        assert isinstance(out, ModelOutput)
        assert out.prior_logit is not None
        assert out.prior_logit.shape[0] == out.num_actions
        assert out.prior_logit.shape[-1] == 1

    def test_forward_omits_prior_logit_when_disabled(self):
        from douzero.models_v2.batch import observation_to_model_inputs
        from douzero.observation.encode_v2 import get_obs_v2
        from douzero.env.env import Env

        env = Env("adp")
        env.reset()
        infoset = env.infoset
        legal = list(infoset.legal_actions)
        if len(legal) < 2:
            pytest.skip("env yielded a trivial single-action decision")
        obs = get_obs_v2(infoset)
        bundle = observation_to_model_inputs(obs)

        m = self._build_model(human_prior=False)
        m.eval()
        with torch.inference_mode():
            out = m(
                state_card_vectors=bundle.state_card_vectors,
                state_context_flat=bundle.state_context_flat,
                context_card_vectors=bundle.context_card_vectors,
                context_flat=bundle.context_flat,
                history_tokens=bundle.history_tokens,
                history_key_padding_mask=bundle.history_key_padding_mask,
                action_features=bundle.action_features,
                action_mask=bundle.action_mask,
                acting_role=bundle.acting_role,
            )
        assert out.prior_logit is None


# --------------------------------------------------------------------------- #
# ModelOutput prior helpers + pure_prior decision mode
# --------------------------------------------------------------------------- #
class TestModelOutputPrior:
    def _make_output(self, prior_logits, mask):
        n = len(prior_logits)
        return ModelOutput(
            win_logit=torch.zeros(n, 1),
            score_if_win=torch.zeros(n, 1),
            score_if_loss=torch.zeros(n, 1),
            p_win=torch.zeros(n, 1),
            score_mean=torch.zeros(n, 1),
            action_mask=torch.tensor(mask, dtype=torch.bool),
            prior_logit=torch.tensor(prior_logits, dtype=torch.float32).reshape(n, 1),
        )

    def test_argmax_prior_picks_highest_valid(self):
        out = self._make_output([0.1, 0.9, 0.5], [True, True, True])
        assert out.argmax_prior() == 1

    def test_argmax_prior_skips_padded(self):
        # index 1 has the highest logit but is padded; must pick index 2.
        out = self._make_output([0.1, 0.99, 0.5], [True, False, True])
        assert out.argmax_prior() == 2

    def test_selected_prior_logit_raises_when_absent(self):
        out = ModelOutput(
            win_logit=torch.zeros(3, 1),
            score_if_win=torch.zeros(3, 1),
            score_if_loss=torch.zeros(3, 1),
            p_win=torch.zeros(3, 1),
            score_mean=torch.zeros(3, 1),
            action_mask=torch.ones(3, dtype=torch.bool),
            prior_logit=None,
        )
        with pytest.raises(ValueError):
            out.argmax_prior()

    def test_pure_prior_in_supported_modes(self):
        assert "pure_prior" in SUPPORTED_DECISION_MODES

    def test_select_action_pure_prior(self):
        out = self._make_output([0.1, 0.9, 0.5], [True, True, True])
        idx = select_action(out, DecisionConfig(mode="pure_prior"))
        assert idx == 1

    def test_select_action_pure_prior_raises_without_head(self):
        out = ModelOutput(
            win_logit=torch.zeros(3, 1),
            score_if_win=torch.zeros(3, 1),
            score_if_loss=torch.zeros(3, 1),
            p_win=torch.zeros(3, 1),
            score_mean=torch.zeros(3, 1),
            action_mask=torch.ones(3, dtype=torch.bool),
            prior_logit=None,
        )
        with pytest.raises(ValueError):
            select_action(out, DecisionConfig(mode="pure_prior"))


# --------------------------------------------------------------------------- #
# Listwise BC loss
# --------------------------------------------------------------------------- #
class TestListwiseBCLoss:
    def _make(self, logits, mask):
        n = len(logits)
        return (
            torch.tensor(logits, dtype=torch.float32).reshape(n, 1),
            torch.tensor(mask, dtype=torch.bool),
        )

    def test_loss_decreases_when_correct_action_logit_rises(self):
        mask = [True, True, True]
        idx = 1
        logits_low, m = self._make([0.0, 0.0, 0.0], mask)
        logits_high, _ = self._make([0.0, 5.0, 0.0], mask)
        loss_low, _ = listwise_bc_loss(logits_low, m, idx)
        loss_high, _ = listwise_bc_loss(logits_high, m, idx)
        assert loss_high.item() < loss_low.item()

    def test_top1_correct_flag(self):
        logits, m = self._make([0.0, 3.0, 0.0], [True, True, True])
        _, hit = listwise_bc_loss(logits, m, 1)
        assert hit is True
        _, hit2 = listwise_bc_loss(logits, m, 0)
        assert hit2 is False

    def test_padded_action_masked_out(self):
        # The padded action has the highest logit but must be ignored.
        logits, m = self._make([0.0, 0.5, 9.0], [True, True, False])
        loss, hit = listwise_bc_loss(logits, m, 1)
        assert hit is True
        # The masked logit (9.0 at index 2) does not affect softmax.
        logits2, m2 = self._make([0.0, 0.5, 0.0], [True, True, False])
        loss2, _ = listwise_bc_loss(logits2, m2, 1)
        assert torch.allclose(loss, loss2)

    def test_temperature_sharpens(self):
        idx = 0
        logits, m = self._make([1.0, 0.0], [True, True])
        low_t, _ = listwise_bc_loss(logits, m, idx, temperature=1.0)
        high_t, _ = listwise_bc_loss(logits, m, idx, temperature=0.1)
        # Sharper temperature on the correct action -> lower loss.
        assert high_t.item() < low_t.item()

    def test_weight_scales_loss(self):
        logits, m = self._make([0.0, 0.0], [True, True])
        base, _ = listwise_bc_loss(logits, m, 0)
        doubled, _ = listwise_bc_loss(logits, m, 0, weight=2.0)
        assert torch.allclose(doubled, base * 2.0)

    def test_target_at_padded_action_rejected(self):
        logits, m = self._make([0.0, 0.0], [True, False])
        with pytest.raises(BCLossError):
            listwise_bc_loss(logits, m, 1)

    def test_out_of_range_target_rejected(self):
        logits, m = self._make([0.0, 0.0], [True, True])
        with pytest.raises(BCLossError):
            listwise_bc_loss(logits, m, 5)

    def test_all_false_mask_rejected(self):
        logits, m = self._make([0.0, 0.0], [False, False])
        with pytest.raises(BCLossError):
            listwise_bc_loss(logits, m, 0)

    def test_zero_actions_rejected(self):
        z = torch.zeros(0, 1)
        m = torch.zeros(0, dtype=torch.bool)
        with pytest.raises(BCLossError):
            listwise_bc_loss(z, m, 0)

    def test_negative_weight_rejected(self):
        logits, m = self._make([0.0, 0.0], [True, True])
        with pytest.raises(BCLossError):
            listwise_bc_loss(logits, m, 0, weight=-1.0)

    def test_gradient_flows_to_logits(self):
        logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
        m = torch.tensor([True, True])
        loss, _ = listwise_bc_loss(logits, m, 0)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestAverageBCLosses:
    def test_empty_returns_zero_loss(self):
        comps = average_bc_losses([])
        assert isinstance(comps, BCLossComponents)
        assert comps.num_decisions == 0
        assert comps.cross_entropy == 0.0
        assert torch.allclose(comps.total, torch.tensor(0.0))

    def test_average_over_decisions(self):
        # Two decisions with known per-decision losses.
        l1 = torch.tensor(2.0)
        l2 = torch.tensor(4.0)
        comps = average_bc_losses([(l1, True), (l2, False)])
        assert comps.num_decisions == 2
        assert comps.top1_correct == 1
        assert comps.cross_entropy == pytest.approx(3.0)


class TestBCLossConfig:
    def test_defaults(self):
        cfg = BCLossConfig()
        assert cfg.lambda_bc == 0.0
        assert cfg.temperature == 1.0

    def test_rejects_negative_lambda(self):
        with pytest.raises(BCLossError):
            BCLossConfig(lambda_bc=-1.0)

    def test_rejects_zero_temperature(self):
        with pytest.raises(BCLossError):
            BCLossConfig(temperature=0.0)

    def test_to_dict(self):
        d = BCLossConfig(lambda_bc=0.5).to_dict()
        assert d["lambda_bc"] == 0.5


# --------------------------------------------------------------------------- #
# Config plumbing (BC block + loss.lambda_bc)
# --------------------------------------------------------------------------- #
class TestBCConfigPlumbing:
    def test_loss_config_carries_lambda_bc(self):
        from douzero.config.schemas import LossConfig

        assert LossConfig().lambda_bc == 0.0
        assert LossConfig(lambda_bc=0.3).lambda_bc == 0.3

    def test_training_config_has_bc_default_disabled(self):
        from douzero.config.schemas import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.bc.enabled is False
        assert cfg.bc.lambda_bc == 0.0

    def test_yaml_loads_bc_block(self, tmp_path):
        from douzero.config.loader import load_config

        yaml = tmp_path / "bc.yaml"
        yaml.write_text(
            "xpid: bc_test\nobjective: adp\n"
            "feature_version: legacy\nruleset: legacy\nmodel_version: legacy\n"
            "bc:\n"
            "  enabled: true\n"
            "  lambda_bc: 0.25\n"
            "  schedule: linear_decay\n"
            "  schedule_steps: 1000\n"
            "  schedule_floor: 0.05\n",
            encoding="utf-8",
        )
        cfg = load_config(str(yaml))
        assert cfg.bc.enabled is True
        assert cfg.bc.lambda_bc == 0.25
        assert cfg.bc.schedule == "linear_decay"
        assert cfg.bc.schedule_steps == 1000
        assert cfg.bc.schedule_floor == 0.05

    def test_yaml_rejects_unknown_bc_key(self, tmp_path):
        from douzero.config.loader import load_config

        yaml = tmp_path / "bad.yaml"
        yaml.write_text(
            "xpid: bc_test\nobjective: adp\n"
            "feature_version: legacy\nruleset: legacy\nmodel_version: legacy\n"
            "bc:\n  not_a_field: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown bc config keys"):
            load_config(str(yaml))

    def test_enhanced_yaml_loads_with_bc_block(self):
        from douzero.config.loader import load_config

        cfg = load_config("configs/enhanced.yaml")
        assert cfg.bc.enabled is False  # default off; legacy path preserved
        assert cfg.loss.lambda_bc == 0.0


# --------------------------------------------------------------------------- #
# Imperfect-information boundary: BC label never reaches the deployment path
# --------------------------------------------------------------------------- #
class TestBCLeakageBoundary:
    def test_bc_sample_kind_is_privileged(self):
        from douzero.human_data.sample import BC_SAMPLE_KIND

        assert BC_SAMPLE_KIND == "bc_sample"

    def test_pure_prior_does_not_consume_hidden_hands(self):
        """The prior decision mode reads only prior_logit + action_mask; it
        has no access to hidden-hand fields by construction (ModelOutput has
        no hidden-hand field)."""
        out = ModelOutput(
            win_logit=torch.zeros(3, 1),
            score_if_win=torch.zeros(3, 1),
            score_if_loss=torch.zeros(3, 1),
            p_win=torch.zeros(3, 1),
            score_mean=torch.zeros(3, 1),
            action_mask=torch.ones(3, dtype=torch.bool),
            prior_logit=torch.tensor([[0.0], [1.0], [0.0]]),
        )
        idx = select_action(out, DecisionConfig(mode="pure_prior"))
        assert idx == 1
        # No hidden-hand field exists on ModelOutput.
        assert not hasattr(out, "all_handcards")
        assert not hasattr(out, "hidden_hands")
