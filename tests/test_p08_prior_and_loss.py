"""P08 prior head, listwise BC loss, BC config, and pure_prior decision tests."""

from __future__ import annotations

import math

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

    def test_label_smoothing_with_padded_mask_is_finite(self):
        """Medium #2: label smoothing must remain finite when some actions are
        padded (-inf logits). PyTorch's F.cross_entropy would put smoothing
        mass on padded classes and yield inf; the mask-aware implementation
        redistributes over valid classes only."""
        # 4 actions, 1 padded (index 3); smoothing > 0.
        logits = torch.tensor([[0.2], [0.5], [-0.3], [9.0]], requires_grad=True)
        mask = torch.tensor([True, True, True, False])
        loss, hit = listwise_bc_loss(
            logits, mask, 1, label_smoothing=0.1
        )
        assert torch.isfinite(loss)
        assert hit  # index 1 has the highest valid logit
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        # The padded row (index 3) receives NO gradient (mass redistributed
        # over valid classes only).
        assert logits.grad[3].abs() < 1e-9

    def test_label_smoothing_zero_equals_plain_ce(self):
        """label_smoothing=0 must reproduce plain cross-entropy exactly."""
        torch.manual_seed(0)
        logits = torch.randn(5, 1)
        mask = torch.tensor([True, True, True, True, True])
        idx = 2
        plain, _ = listwise_bc_loss(logits, mask, idx, label_smoothing=0.0)
        # Hand-computed plain CE: -log_softmax at idx.
        import torch.nn.functional as F

        log_softmax = F.log_softmax(logits.squeeze(-1), dim=-1)
        expected = -log_softmax[idx]
        assert torch.allclose(plain, expected, atol=1e-6)

    def test_label_smoothing_reduces_loss_on_wrong_prediction(self):
        """Label smoothing should reduce the loss ceiling when the model is
        confidently wrong (the whole point of smoothing)."""
        # Confidently wrong: high logit on index 0, target is index 1.
        logits = torch.tensor([[5.0], [0.0]])
        mask = torch.tensor([True, True])
        plain, _ = listwise_bc_loss(logits, mask, 1, label_smoothing=0.0)
        smooth, _ = listwise_bc_loss(logits, mask, 1, label_smoothing=0.2)
        assert smooth.item() < plain.item()

    def test_label_smoothing_target_distribution_sums_to_one(self):
        """Blocker 2: the smoothed target distribution MUST sum to exactly 1
        (the earlier bug dropped the target's ε/K share, giving 1 − ε/K)."""
        # Reconstruct the target the loss builds internally and check its sum.
        # 4 valid actions, label_smoothing=0.2 -> each valid gets 0.05, target
        # gets 0.05 + 0.8 = 0.85. Sum = 0.85 + 3*0.05 = 1.0.
        import torch.nn.functional as F

        logits = torch.tensor([[0.0], [0.5], [-0.3], [0.2]])
        mask = torch.tensor([True, True, True, True])
        eps = 0.2
        num_valid = 4
        target_idx = 1
        # Hand-compute the expected target probs (mirrors the fix).
        expected = torch.full((4,), eps / num_valid)
        expected[target_idx] += 1.0 - eps
        assert expected.sum().item() == pytest.approx(1.0)
        assert expected[target_idx].item() == pytest.approx(1.0 - eps + eps / num_valid)
        # The loss equals -sum(expected * log_softmax(logits)).
        log_sm = F.log_softmax(logits.squeeze(-1), dim=-1)
        expected_loss = -(expected * log_sm).sum()
        actual_loss, _ = listwise_bc_loss(logits, mask, target_idx, label_smoothing=eps)
        assert torch.allclose(actual_loss, expected_loss, atol=1e-6)

    def test_label_smoothing_sum_independent_of_action_count(self):
        """Blocker 2: the target sum is 1 regardless of K (the bug made the
        shortfall depend on K). Compare K=2 vs K=8 at the same eps."""
        for k in (2, 3, 5, 8):
            logits = torch.zeros(k, 1)
            mask = torch.ones(k, dtype=torch.bool)
            _, _ = listwise_bc_loss(logits, mask, 0, label_smoothing=0.2)
            # If the sum were < 1 the loss would be artificially shrunk; with
            # uniform logits the CE for a proper 1-sum target is exactly
            # -((1-eps+eps/k)*log(p_target) + (k-1)*(eps/k)*log(p_other))
            # where p = 1/k for all. Verify it matches the closed form.
            import torch.nn.functional as F

            eps = 0.2
            p = 1.0 / k
            target_prob = 1.0 - eps + eps / k
            other_prob = eps / k
            expected = -(target_prob * math.log(p) + (k - 1) * other_prob * math.log(p))
            actual, _ = listwise_bc_loss(logits, mask, 0, label_smoothing=eps)
            assert actual.item() == pytest.approx(expected, abs=1e-5)


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

    # ------------------------------------------------------------------ #
    # Blocker 3 (round 3): label_smoothing range validation
    # ------------------------------------------------------------------ #
    def test_listwise_bc_loss_rejects_label_smoothing_above_one(self):
        logits = torch.tensor([[0.0], [0.0]])
        mask = torch.tensor([True, True])
        with pytest.raises(BCLossError, match="label_smoothing"):
            listwise_bc_loss(logits, mask, 0, label_smoothing=1.0)

    def test_listwise_bc_loss_rejects_label_smoothing_negative(self):
        logits = torch.tensor([[0.0], [0.0]])
        mask = torch.tensor([True, True])
        with pytest.raises(BCLossError, match="label_smoothing"):
            listwise_bc_loss(logits, mask, 0, label_smoothing=-0.1)

    def test_listwise_bc_loss_rejects_nan_label_smoothing(self):
        logits = torch.tensor([[0.0], [0.0]])
        mask = torch.tensor([True, True])
        with pytest.raises(BCLossError, match="label_smoothing"):
            listwise_bc_loss(logits, mask, 0, label_smoothing=float("nan"))

    def test_listwise_bc_loss_rejects_inf_label_smoothing(self):
        logits = torch.tensor([[0.0], [0.0]])
        mask = torch.tensor([True, True])
        with pytest.raises(BCLossError, match="label_smoothing"):
            listwise_bc_loss(logits, mask, 0, label_smoothing=float("inf"))

    def test_bcconfig_rejects_invalid_label_smoothing(self):
        from douzero.config.schemas import BCConfig

        with pytest.raises(ValueError, match="label_smoothing"):
            BCConfig(label_smoothing=2.0)
        with pytest.raises(ValueError, match="label_smoothing"):
            BCConfig(label_smoothing=1.0)
        with pytest.raises(ValueError, match="label_smoothing"):
            BCConfig(label_smoothing=-0.1)

    def test_yaml_rejects_invalid_label_smoothing(self, tmp_path):
        from douzero.config.loader import load_config

        yaml = tmp_path / "bad_ls.yaml"
        yaml.write_text(
            "xpid: t\nobjective: adp\n"
            "feature_version: legacy\nruleset: legacy\nmodel_version: legacy\n"
            "bc:\n  label_smoothing: 2.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_config(str(yaml))


# --------------------------------------------------------------------------- #
# Config plumbing (BC block + loss.lambda_bc)
# --------------------------------------------------------------------------- #
class TestBCConfigPlumbing:
    def test_loss_config_carries_lambda_bc(self):
        from douzero.config.schemas import LossConfig

        assert LossConfig().lambda_bc == 0.0
        assert LossConfig(lambda_bc=0.3).lambda_bc == 0.3

    def test_training_config_has_bc_defaults(self):
        """Blocker 3: BCConfig no longer has enabled/lambda_bc (single source
        of truth is loss.lambda_bc). Verify the remaining fields default
        cleanly and the enable condition is loss.lambda_bc."""
        from douzero.config.schemas import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.bc.data_path == ""
        assert cfg.bc.temperature == 1.0
        assert cfg.bc.label_smoothing == 0.0
        assert cfg.loss.lambda_bc == 0.0  # the sole enable condition

    def test_yaml_loads_bc_block(self, tmp_path):
        from douzero.config.loader import load_config

        yaml = tmp_path / "bc.yaml"
        yaml.write_text(
            "xpid: bc_test\nobjective: adp\n"
            "feature_version: legacy\nruleset: legacy\nmodel_version: legacy\n"
            "loss:\n  lambda_bc: 0.25\n"
            "bc:\n"
            "  data_path: /tmp/x.jsonl\n"
            "  temperature: 0.8\n"
            "  label_smoothing: 0.05\n"
            "  schedule: linear_decay\n"
            "  schedule_steps: 1000\n"
            "  schedule_floor: 0.05\n",
            encoding="utf-8",
        )
        cfg = load_config(str(yaml))
        assert cfg.loss.lambda_bc == 0.25  # the sole enable/weight source
        assert cfg.bc.data_path == "/tmp/x.jsonl"
        assert cfg.bc.temperature == 0.8
        assert cfg.bc.label_smoothing == 0.05
        assert cfg.bc.schedule == "linear_decay"
        assert cfg.bc.schedule_steps == 1000
        assert cfg.bc.schedule_floor == 0.05

    def test_yaml_rejects_removed_bc_enabled_and_lambda_bc(self, tmp_path):
        """Blocker 3: bc.enabled and bc.lambda_bc were removed (single source
        of truth); a YAML still setting them must fail loudly."""
        from douzero.config.loader import load_config

        yaml = tmp_path / "bad.yaml"
        yaml.write_text(
            "xpid: bc_test\nobjective: adp\n"
            "feature_version: legacy\nruleset: legacy\nmodel_version: legacy\n"
            "bc:\n  enabled: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown bc config keys"):
            load_config(str(yaml))

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
        # BC is off by default (loss.lambda_bc == 0); legacy path preserved.
        assert cfg.loss.lambda_bc == 0.0

    def test_doc_yaml_example_loads(self, tmp_path):
        """Blocker 1 (round 4): the YAML example in docs/human_data_and_bc.md
        must actually load via load_config (no removed fields like bc.enabled
        or bc.lambda_bc)."""
        from douzero.config.loader import load_config

        yaml = tmp_path / "doc_example.yaml"
        yaml.write_text(
            "xpid: doc_test\nobjective: adp\n"
            "feature_version: v2\nruleset: legacy\nmodel_version: v2\n"
            "model:\n"
            "  version: v2\n"
            "  hidden_size: 64\n"
            "  history_encoder: lstm\n"
            "  history_layers: 1\n"
            "  history_heads: 4\n"
            "  role_embedding_dim: 16\n"
            "  human_prior_enabled: true\n"
            "loss:\n"
            "  lambda_bc: 0.3\n"
            "bc:\n"
            "  data_path: /path/to/validated.jsonl\n"
            "  temperature: 1.0\n"
            "  label_smoothing: 0.0\n"
            "  skill_weight_clip: 10.0\n"
            "  schedule: constant\n"
            "  schedule_steps: 0\n"
            "  schedule_floor: 0.0\n",
            encoding="utf-8",
        )
        cfg = load_config(str(yaml))
        assert cfg.loss.lambda_bc == 0.3
        assert cfg.model.human_prior_enabled is True
        assert cfg.bc.data_path == "/path/to/validated.jsonl"


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
