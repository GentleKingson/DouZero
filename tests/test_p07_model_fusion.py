"""P07 Model V2 belief-fusion tests.

Verifies that the belief feature projection (``belief_enabled=True``):

- is gated solely by ``belief_enabled`` (architecture delta captured by the
  existing identity axis, so belief-disabled checkpoints are unaffected),
- changes the output when belief features are supplied vs. zeroed,
- detaches the belief input under ``belief_stop_gradient=True`` (value loss
  does NOT flow into the belief features) and allows flow when False,
- rejects belief_features passed to a belief-disabled model,
- preserves the imperfect-information boundary (belief features derive from a
  public posterior; no hidden hand reaches the value model).

The belief-disabled regression (existing P05/P06 behaviour is byte-identical)
is covered by the existing ``tests/test_model_v2.py`` suite, which runs
unchanged on this branch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from douzero.belief import BELIEF_FEATURE_DIM, BeliefConfig, BeliefModel
from douzero.belief.model import belief_features_from_probs
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.env.rules import RuleSet


def _model_inputs(num_actions: int = 3, *, schema=None):
    """Build minimal valid ModelV2 inputs for ``num_actions`` actions."""
    schema = schema or build_v2_schema()
    from douzero.models_v2.batch import observation_to_model_inputs
    from douzero.observation.encode_v2 import get_obs_v2

    from douzero.env.env import Env

    np.random.seed(0)
    env = Env("adp")
    env.reset()
    obs = get_obs_v2(env.infoset, schema=schema)
    bundle = observation_to_model_inputs(obs)
    # Slice the action block to the requested number of actions (keep at least
    # one so the model has something to score).
    n = min(num_actions, bundle.action_features.shape[0])
    return (
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features[:n],
        torch.ones(n, dtype=torch.bool),
        bundle.acting_role,
        obs,
    )


class TestBeliefFusionGate:
    def test_belief_disabled_has_no_belief_proj(self):
        model = ModelV2(build_v2_schema(), ModelV2Config(belief_enabled=False))
        assert model.belief_proj is None

    def test_belief_enabled_creates_belief_proj(self):
        cfg = ModelV2Config(belief_enabled=True, hidden_size=32,
                            history_heads=4, history_layers=2)
        model = ModelV2(build_v2_schema(), cfg)
        assert model.belief_proj is not None
        assert model.belief_proj.in_features == BELIEF_FEATURE_DIM
        assert model.belief_proj.out_features == cfg.hidden_size

    def test_belief_enabled_is_identity_axis(self):
        """belief_enabled changes the model-config hash (checkpoint axis)."""
        off = ModelV2Config(belief_enabled=False)
        on = ModelV2Config(belief_enabled=True)
        assert off.stable_hash() != on.stable_hash()

    def test_belief_features_rejected_when_disabled(self):
        model = ModelV2(build_v2_schema(),
                        ModelV2Config(belief_enabled=False, hidden_size=32,
                                      history_heads=4, history_layers=2))
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        feats = torch.zeros(BELIEF_FEATURE_DIM)
        with pytest.raises(ValueError):
            model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                  belief_features=feats)

    def test_parameter_count_includes_belief_proj(self):
        off = ModelV2(build_v2_schema(),
                      ModelV2Config(belief_enabled=False, hidden_size=32,
                                    history_heads=4, history_layers=2))
        on = ModelV2(build_v2_schema(),
                     ModelV2Config(belief_enabled=True, hidden_size=32,
                                   history_heads=4, history_layers=2))
        off_pc = off.parameter_count()
        on_pc = on.parameter_count()
        assert "belief_proj" not in off_pc
        assert "belief_proj" in on_pc
        assert on_pc["total"] == off_pc["total"] + on_pc["belief_proj"]


class TestBeliefFusionForward:
    def _belief_enabled_model(self):
        cfg = ModelV2Config(belief_enabled=True, hidden_size=32,
                            history_heads=4, history_layers=2, nan_guard=False)
        return ModelV2(build_v2_schema(), cfg), cfg

    def test_missing_belief_features_fails_closed_by_default(self):
        """A belief-enabled model without features must NOT silently degrade."""
        model, _ = self._belief_enabled_model()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        with pytest.raises(ValueError):
            model(scv, scf, ccv, cf, ht, hmask, af, am, role)

    def test_missing_belief_features_allowed_with_explicit_flag(self):
        model, _ = self._belief_enabled_model()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        with torch.no_grad():
            out_zero = model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                             allow_missing_belief_features=True)
            out_explicit_zero = model(
                scv, scf, ccv, cf, ht, hmask, af, am, role,
                belief_features=torch.zeros(BELIEF_FEATURE_DIM),
            )
        # The explicit-zero path == the allow-missing zero-vector path.
        np.testing.assert_allclose(
            out_zero.win_logit.numpy(),
            out_explicit_zero.win_logit.numpy(), atol=1e-6,
        )

    def test_nonzero_belief_features_change_output(self):
        model, _ = self._belief_enabled_model()
        model.eval()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        feats = torch.randn(BELIEF_FEATURE_DIM)
        with torch.no_grad():
            out_zero = model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                             belief_features=torch.zeros(BELIEF_FEATURE_DIM))
            out_on = model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                           belief_features=feats)
        assert not np.allclose(out_zero.win_logit.numpy(),
                               out_on.win_logit.numpy(), atol=1e-5)

    def test_stop_gradient_default_blocks_flow_into_belief_features(self):
        """Default path: belief features are detached (frozen-features path).

        The belief_proj layer is still trainable, but the (frozen) belief
        features themselves receive no gradient.
        """
        model, _ = self._belief_enabled_model()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        feats = torch.randn(BELIEF_FEATURE_DIM, requires_grad=True)
        out = model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                    belief_features=feats, belief_stop_gradient=True)
        out.win_logit.sum().backward()
        # Detached inside -> no grad reached the leaf belief feature tensor.
        assert feats.grad is None
        # But the belief_proj layer DID receive gradient (it is trainable).
        assert model.belief_proj.weight.grad is not None

    def test_stop_gradient_false_preserves_feature_graph(self):
        """The opt-in joint path propagates value loss to belief features."""
        model, _ = self._belief_enabled_model()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        features = torch.randn(BELIEF_FEATURE_DIM, requires_grad=True)
        output = model(
            scv, scf, ccv, cf, ht, hmask, af, am, role,
            belief_features=features,
            belief_stop_gradient=False,
        )
        output.win_logit.sum().backward()
        assert features.grad is not None
        assert bool(torch.isfinite(features.grad).all())
        assert float(features.grad.abs().sum()) > 0

    def test_rejects_wrong_belief_feature_shape(self):
        """Exact shape (BELIEF_FEATURE_DIM,) is required, not just trailing dim."""
        model, _ = self._belief_enabled_model()
        (scv, scf, ccv, cf, ht, hmask, af, am, role, _obs) = _model_inputs()
        with pytest.raises(ValueError):
            model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                  belief_features=torch.zeros(BELIEF_FEATURE_DIM + 1))
        # A 2-D tensor with the right trailing dim must also be rejected.
        with pytest.raises(ValueError):
            model(scv, scf, ccv, cf, ht, hmask, af, am, role,
                  belief_features=torch.zeros(1, BELIEF_FEATURE_DIM))


class TestBeliefValueIntegration:
    """End-to-end: a frozen BeliefModel feeds features into ModelV2."""

    def test_belief_model_features_feed_value_model(self):
        from douzero.belief import build_belief_input
        from douzero.observation.encode_v2 import get_obs_v2

        from douzero.env.env import Env

        np.random.seed(1)
        torch.manual_seed(1)
        env = Env("adp")
        env.reset()
        obs = get_obs_v2(env.infoset)
        binput = build_belief_input(obs.public)

        belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        belief.eval()
        with torch.no_grad():
            bout = belief([binput])
            feats = torch.from_numpy(belief_features_from_probs(
                bout.probs.numpy(),
                bout.opponent_a_total,
                np.stack([binput.unseen_counts]),
            )[0].astype(np.float32))

        cfg = ModelV2Config(belief_enabled=True, hidden_size=32,
                            history_heads=4, history_layers=2, nan_guard=False)
        value = ModelV2(build_v2_schema(), cfg)
        value.eval()
        from douzero.models_v2.batch import observation_to_model_inputs

        bundle = observation_to_model_inputs(obs)
        with torch.no_grad():
            out = value(
                bundle.state_card_vectors, bundle.state_context_flat,
                bundle.context_card_vectors, bundle.context_flat,
                bundle.history_tokens, bundle.history_key_padding_mask,
                bundle.action_features, bundle.action_mask,
                bundle.acting_role,
                belief_features=feats,
            )
        assert out.win_logit.shape[0] == bundle.action_features.shape[0]
        assert bool(torch.isfinite(out.win_logit).all())
        # The belief features are a public posterior; the value model consumed
        # NO privileged field (the public observation is the only input source
        # for both models).
        assert obs.is_privileged is False


class TestDeepAgentV2BeliefDeployment:
    """A belief-enabled value model deploys via DeepAgentV2 (review blocker #2).

    The agent holds a frozen BeliefModel, computes the constrained posterior
    features from the public observation, and fuses them into the value model.
    A belief-enabled value model WITHOUT a belief_model is rejected with a
    precise error (no silent crash at inference, no false 'degraded' claim).
    """

    def _build_agent_inputs(self):
        from douzero.env.env import Env
        from douzero.observation.encode_v2 import get_obs_v2

        np.random.seed(7)
        torch.manual_seed(7)
        env = Env("adp")
        env.reset()
        obs = get_obs_v2(env.infoset)
        return obs

    def _belief_enabled_value_model(self):
        cfg = ModelV2Config(belief_enabled=True, hidden_size=32,
                            history_heads=4, history_layers=2, nan_guard=False)
        return ModelV2(build_v2_schema(), cfg)

    def test_belief_enabled_agent_runs_end_to_end(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        value = self._belief_enabled_value_model()
        belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        agent = DeepAgentV2("landlord", value, RuleSet.legacy(),
                            belief_model=belief)
        obs = self._build_agent_inputs()
        chosen = agent.act_v2(obs)
        assert chosen in obs.public.legal_actions

    def test_belief_enabled_without_belief_model_rejected(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        value = self._belief_enabled_value_model()
        with pytest.raises(ValueError, match="belief_enabled=True"):
            DeepAgentV2("landlord", value, RuleSet.legacy())

    def test_belief_model_with_belief_disabled_value_rejected(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        value = ModelV2(build_v2_schema(),
                        ModelV2Config(belief_enabled=False, hidden_size=32,
                                      history_heads=4, history_layers=2))
        belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        with pytest.raises(ValueError, match="belief_enabled=False"):
            DeepAgentV2("landlord", value, RuleSet.legacy(),
                        belief_model=belief)

    def test_belief_disabled_agent_ignores_belief_path(self):
        """A belief-disabled agent must not compute or require belief features."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        value = ModelV2(build_v2_schema(),
                        ModelV2Config(belief_enabled=False, hidden_size=32,
                                      history_heads=4, history_layers=2))
        assert value.belief_proj is None
        agent = DeepAgentV2("landlord", value, RuleSet.legacy())
        assert agent.belief_model is None
        obs = self._build_agent_inputs()
        chosen = agent.act_v2(obs)
        assert chosen in obs.public.legal_actions

    def test_act_from_infoset_path_also_fuses_belief(self):
        """The legacy act(infoset) entry must also wire belief features."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        value = self._belief_enabled_value_model()
        belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        agent = DeepAgentV2("landlord", value, RuleSet.legacy(),
                            belief_model=belief)
        from douzero.env.env import Env

        np.random.seed(9)
        env = Env("adp")
        env.reset()
        chosen = agent.act(env.infoset)
        assert chosen in env.infoset.legal_actions
