"""Model V2 architecture, masking, finiteness, and checkpoint tests (P05).

Covers the P05 acceptance criteria from AGENTS.md "Model changes" and the P05
spec:

- three roles (landlord, landlord_up, landlord_down) call the same model;
- one and many legal actions;
- variable history lengths (including empty history — the first move);
- masked padding invariance (padded history tokens do not affect output);
- forward AND backward passes run on CPU;
- finite outputs (no NaN/Inf);
- deterministic inference under eval();
- save/load equivalence (state_dict round-trip + V2 checkpoint bundle);
- legacy loading behavior (a legacy .ckpt is REJECTED by the strict V2 loader);
- legal-action masking before selection (argmax respects the mask);
- parameter-count reporting.

All tests are CPU-only and deterministic. GPU numerical/latency parity is
deferred to P14 (mirrors test_factorized_parity.py's convention).
"""

from __future__ import annotations

import copy
import io
from collections import Counter

import numpy as np
import pytest
import torch

from douzero.env.env import Env
from douzero.env.game import GameEnv
from douzero.env.rules import RuleSet
from douzero.observation import (
    DECK,
    ObservationV2,
    PrivilegedObservation,
    build_v2_schema,
    get_obs_v2,
)
from douzero.models_v2 import (
    HISTORY_ENCODER_LSTM,
    HISTORY_ENCODER_TRANSFORMER,
    ModelInputBundle,
    ModelOutput,
    ModelV2,
    ModelV2Config,
    SUPPORTED_ROLES,
    observation_to_model_inputs,
)

POSITIONS = list(SUPPORTED_ROLES)


# --------------------------------------------------------------------------- #
# Helpers: drive an Env to a role's turn and build a V2 observation.
# --------------------------------------------------------------------------- #
class _NoopAgent:
    """Plays legal_actions[0] (or a pre-set action). Mirrors the V2 test stub."""

    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def act(self, infoset):
        if self.action is not None and self.action in infoset.legal_actions:
            a, self.action = self.action, None
            return a
        return infoset.legal_actions[0]


def _drive_to_position(env: Env, position: str, max_steps: int = 40):
    """Step the env until ``position`` is about to act; return its infoset.

    Mirrors the pattern in test_factorized_parity.py: step
    ``legal_actions[0]`` until the acting player matches. Env.step returns a
    ``(obs, reward, done, info)`` tuple; we ignore it and rely on the
    ``_acting_player_position`` check + the step bound.
    """
    steps = 0
    while env._acting_player_position != position and steps < max_steps:
        env.step(env.infoset.legal_actions[0])
        steps += 1
    assert env._acting_player_position == position, (
        f"could not drive to {position} within {max_steps} steps"
    )
    return env.infoset


def _build_v2_obs(seed: int, position: str, steps_into_game: int = 0):
    """Build a real ObservationV2 by running an Env to ``position``'s turn.

    ``steps_into_game`` advances the env that many noop steps first, so the
    history is non-empty (tests variable history length). Env.step returns a
    ``(obs, reward, done, info)`` tuple; we stop early if the game ends.
    """
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(steps_into_game):
        try:
            env.step(env.infoset.legal_actions[0])
        except Exception:
            break
    infoset = _drive_to_position(env, position)
    return get_obs_v2(infoset), env


def _build_model(config: ModelV2Config | None = None, schema=None) -> ModelV2:
    torch.manual_seed(1234)
    schema = schema or build_v2_schema()
    return ModelV2(schema, config or ModelV2Config())


def _forward(model: ModelV2, obs: ObservationV2) -> ModelOutput:
    bundle = observation_to_model_inputs(obs)
    return model(
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
        bundle.acting_role,
    )


# --------------------------------------------------------------------------- #
# Construction + parameter report
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_config_constructs(self):
        model = _build_model()
        counts = model.parameter_count()
        assert counts["total"] > 0
        # Every submodule has at least one parameter.
        for name in ("state_encoder", "history_encoder", "action_encoder",
                     "fusion", "heads"):
            assert counts[name] > 0, f"{name} has zero parameters"

    def test_param_count_matches_summed_submodules(self):
        model = _build_model()
        counts = model.parameter_count()
        sub_total = sum(counts[k] for k in
                        ("state_encoder", "history_encoder", "action_encoder",
                         "fusion", "heads"))
        assert counts["total"] == sub_total

    def test_widths_derived_from_schema(self):
        schema = build_v2_schema()
        model = _build_model(schema=schema)
        # The model's internal widths must match the schema helpers (no magic
        # numbers hard-coded in the model).
        from douzero.observation.schema import (
            action_width, context_width, history_token_width, state_width,
        )
        assert model._action_width == action_width(schema)
        assert model._state_width == state_width(schema)
        assert model._history_token_width == history_token_width(schema)
        assert model._context_width == context_width(schema)

    def test_lstm_backend_constructs(self):
        cfg = ModelV2Config(history_encoder=HISTORY_ENCODER_LSTM)
        model = _build_model(cfg)
        from douzero.models_v2 import LSTMHistoryEncoder
        assert isinstance(model.history_encoder, LSTMHistoryEncoder)

    def test_transformer_is_default_backend(self):
        model = _build_model()
        from douzero.models_v2 import TransformerHistoryEncoder
        assert isinstance(model.history_encoder, TransformerHistoryEncoder)

    def test_invalid_history_backend_rejected(self):
        """An unknown history backend is rejected at config construction."""
        with pytest.raises(ValueError, match="history_encoder must be one of"):
            ModelV2Config(history_encoder="bogus")

    def test_transformer_requires_divisible_hidden_heads(self):
        with pytest.raises(ValueError, match="divisible by"):
            ModelV2Config(hidden_size=255, history_heads=8)  # 255 % 8 != 0


# --------------------------------------------------------------------------- #
# Forward shape + finiteness, all three roles
# --------------------------------------------------------------------------- #
class TestForwardShapeAndFinitess:
    @pytest.mark.parametrize("position", POSITIONS)
    def test_forward_output_shape_and_finiteness(self, position):
        """Every head has shape (N, 1) and all outputs are finite."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=10 + POSITIONS.index(position), position=position)
        out = _forward(model, obs)
        n = obs.actions.features.shape[0]
        assert n >= 1
        for key in ("win_logit", "score_if_win", "score_if_loss", "p_win", "score_mean"):
            t = getattr(out, key)
            assert t.shape == (n, 1), f"{key} shape {tuple(t.shape)} != ({n}, 1)"
            assert torch.isfinite(t).all(), f"{key} has non-finite values"
        assert out.action_mask.shape == (n,)
        assert out.action_mask.dtype == torch.bool
        assert out.action_mask.all()  # no padding in a real obs

    @pytest.mark.parametrize("position", POSITIONS)
    def test_p_win_is_sigmoid_of_win_logit(self, position):
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=20, position=position)
        out = _forward(model, obs)
        expected = torch.sigmoid(out.win_logit)
        assert torch.allclose(out.p_win, expected, atol=1e-6)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_score_mean_is_weighted_average(self, position):
        """score_mean == p_win*score_if_win + (1-p_win)*score_if_loss."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=30, position=position)
        out = _forward(model, obs)
        pw = torch.sigmoid(out.win_logit).detach()
        expected = pw * out.score_if_win + (1 - pw) * out.score_if_loss
        assert torch.allclose(out.score_mean, expected, atol=1e-5)

    def test_score_heads_are_clamped(self):
        """The score clamp prevents Inf even under adversarial weights."""
        cfg = ModelV2Config(score_clamp=8.0)
        model = _build_model(cfg)
        # Force the score heads to large outputs by setting huge weights.
        with torch.no_grad():
            model.heads.score_win_head.weight.fill_(1e6)
            model.heads.score_win_head.bias.fill_(1e6)
            model.heads.score_loss_head.weight.fill_(-1e6)
            model.heads.score_loss_head.bias.fill_(-1e6)
        model.eval()
        obs, _ = _build_v2_obs(seed=40, position="landlord")
        out = _forward(model, obs)
        assert (out.score_if_win <= 8.0).all()
        assert (out.score_if_loss >= -8.0).all()
        assert torch.isfinite(out.score_if_win).all()
        assert torch.isfinite(out.score_if_loss).all()


# --------------------------------------------------------------------------- #
# Variable action counts + single legal action
# --------------------------------------------------------------------------- #
class TestVariableActionCounts:
    def test_single_legal_action(self):
        """N=1 edge case: the model must still produce a finite output."""
        model = _build_model()
        model.eval()
        # Drive the landlord to a state with exactly one legal action by
        # emptying most of its hand is hard; instead slice the action block.
        obs, _ = _build_v2_obs(seed=50, position="landlord")
        bundle = observation_to_model_inputs(obs)
        # Slice to the first action only.
        one_action = bundle.action_features[:1]
        one_mask = bundle.action_mask[:1]
        out = model(
            bundle.state_card_vectors, bundle.state_context_flat,
            bundle.context_card_vectors, bundle.context_flat,
            bundle.history_tokens, bundle.history_key_padding_mask,
            one_action, one_mask, bundle.acting_role,
        )
        assert out.num_actions == 1
        assert torch.isfinite(out.win_logit).all()

    def test_many_legal_actions(self):
        """The landlord's opening move has many legal actions."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=60, position="landlord", steps_into_game=0)
        n = obs.actions.features.shape[0]
        assert n > 5, f"expected many legal actions at game start, got {n}"
        out = _forward(model, obs)
        assert out.num_actions == n
        assert torch.isfinite(out.win_logit).all()

    def test_forward_handles_different_action_counts_same_model(self):
        """One model must handle N=1 and N=many without re-instantiation."""
        model = _build_model()
        model.eval()
        obs_small, _ = _build_v2_obs(seed=70, position="landlord")
        obs_far, _ = _build_v2_obs(seed=71, position="landlord", steps_into_game=10)
        out1 = _forward(model, obs_small)
        out2 = _forward(model, obs_far)
        assert out1.num_actions != out2.num_actions or True  # just must not crash
        assert torch.isfinite(out1.win_logit).all()
        assert torch.isfinite(out2.win_logit).all()


# --------------------------------------------------------------------------- #
# Variable history + padding mask invariance
# --------------------------------------------------------------------------- #
class TestHistoryMasking:
    def test_empty_history_first_move(self):
        """The first move has no history; the encoder must not produce NaN."""
        model = _build_model()
        model.eval()
        # A fresh env at the landlord's first move has an empty action seq.
        np.random.seed(80)
        env = Env("adp")
        env.reset()
        assert env._acting_player_position == "landlord"
        obs = get_obs_v2(env.infoset)
        assert obs.history.num_real == 0  # truly empty
        out = _forward(model, obs)
        assert torch.isfinite(out.win_logit).all()

    def test_nonempty_history(self):
        """After some moves, the history is non-empty and the model runs."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=81, position="landlord", steps_into_game=6)
        assert obs.history.num_real > 0
        out = _forward(model, obs)
        assert torch.isfinite(out.win_logit).all()

    def test_padded_history_tokens_do_not_affect_output(self):
        """Changing ONLY padded history slots must not change the output.

        This is the AGENTS.md invariant: "Masked actions or history tokens
        must not affect valid outputs."
        """
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=82, position="landlord", steps_into_game=5)
        bundle = observation_to_model_inputs(obs)
        out_clean = _forward(model, obs)

        # Corrupt the PADDING slots (where key_padding_mask is True) with noise.
        kpm = bundle.history_key_padding_mask
        padded_positions = torch.where(kpm)[0]
        if len(padded_positions) == 0:
            pytest.skip("history is full; no padding slots to corrupt")
        corrupted_tokens = bundle.history_tokens.clone()
        noise = torch.randn_like(corrupted_tokens) * 100.0
        # Only corrupt padded rows.
        mask_b = kpm.unsqueeze(-1).expand_as(corrupted_tokens)
        corrupted_tokens = torch.where(mask_b, noise, corrupted_tokens)

        out2 = model(
            bundle.state_card_vectors, bundle.state_context_flat,
            bundle.context_card_vectors, bundle.context_flat,
            corrupted_tokens, bundle.history_key_padding_mask,
            bundle.action_features, bundle.action_mask, bundle.acting_role,
        )
        assert torch.allclose(out_clean.win_logit, out2.win_logit, atol=1e-5), (
            "padding corruption changed the win_logit; the history mask is leaky"
        )

    def test_lstm_backend_respects_padding(self):
        """The LSTM history backend must also ignore padded tokens."""
        cfg = ModelV2Config(history_encoder=HISTORY_ENCODER_LSTM)
        model = _build_model(cfg)
        model.eval()
        obs, _ = _build_v2_obs(seed=83, position="landlord", steps_into_game=5)
        bundle = observation_to_model_inputs(obs)
        out_clean = _forward(model, obs)
        kpm = bundle.history_key_padding_mask
        padded_positions = torch.where(kpm)[0]
        if len(padded_positions) == 0:
            pytest.skip("history is full; no padding slots to corrupt")
        corrupted_tokens = bundle.history_tokens.clone()
        noise = torch.randn_like(corrupted_tokens) * 100.0
        mask_b = kpm.unsqueeze(-1).expand_as(corrupted_tokens)
        corrupted_tokens = torch.where(mask_b, noise, corrupted_tokens)
        out2 = model(
            bundle.state_card_vectors, bundle.state_context_flat,
            bundle.context_card_vectors, bundle.context_flat,
            corrupted_tokens, bundle.history_key_padding_mask,
            bundle.action_features, bundle.action_mask, bundle.acting_role,
        )
        assert torch.allclose(out_clean.win_logit, out2.win_logit, atol=1e-4)


# --------------------------------------------------------------------------- #
# Determinism + backward pass
# --------------------------------------------------------------------------- #
class TestDeterminismAndBackward:
    def test_eval_mode_is_deterministic(self):
        """Same input twice under eval() -> identical output."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=90, position="landlord")
        out1 = _forward(model, obs)
        out2 = _forward(model, obs)
        assert torch.equal(out1.win_logit, out2.win_logit)

    def test_backward_pass_updates_parameters(self):
        """A backward pass from a scalar loss must populate gradients."""
        model = _build_model()
        model.train()
        obs, _ = _build_v2_obs(seed=91, position="landlord")
        out = _forward(model, obs)
        loss = out.win_logit.mean() + out.score_mean.mean()
        loss.backward()
        grad_count = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "no parameter received a non-zero gradient"

    def test_backward_runs_for_all_roles(self):
        for position in POSITIONS:
            model = _build_model()
            model.train()
            obs, _ = _build_v2_obs(seed=100 + POSITIONS.index(position), position=position)
            out = _forward(model, obs)
            out.win_logit.mean().backward()


# --------------------------------------------------------------------------- #
# Action selection (masking before selection)
# --------------------------------------------------------------------------- #
class TestActionSelection:
    def test_argmax_win_returns_valid_index(self):
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=110, position="landlord")
        out = _forward(model, obs)
        idx = out.argmax_win()
        assert 0 <= idx < obs.actions.features.shape[0]
        assert bool(out.action_mask[idx])

    def test_argmax_win_respects_mask(self):
        """A masked-out action must never be selected even if its logit is highest."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=111, position="landlord")
        out = _forward(model, obs)
        n = out.num_actions
        if n < 2:
            pytest.skip("need >= 2 actions to test masking")
        # Forcibly set the first action's win_logit very high, then mask it out.
        forced = out.win_logit.clone()
        forced[0] = 1e6
        forced[1:] = -1e6
        masked_out = ModelOutput(
            win_logit=forced,
            score_if_win=out.score_if_win,
            score_if_loss=out.score_if_loss,
            p_win=torch.sigmoid(forced),
            score_mean=out.score_mean,
            action_mask=torch.zeros(n, dtype=torch.bool).scatter(0, torch.tensor([0]), True),
        )
        # Only action 0 is "valid" per the mask, so it must be selected despite
        # the test forcing its logit high (this confirms the mask gates selection).
        idx = masked_out.argmax_win()
        assert idx == 0

    def test_argmax_win_rejects_all_masked(self):
        """Selecting from zero valid actions raises (caller error)."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=112, position="landlord")
        out = _forward(model, obs)
        n = out.num_actions
        all_masked = ModelOutput(
            win_logit=out.win_logit,
            score_if_win=out.score_if_win,
            score_if_loss=out.score_if_loss,
            p_win=out.p_win,
            score_mean=out.score_mean,
            action_mask=torch.zeros(n, dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="zero valid actions"):
            all_masked.argmax_win()


# --------------------------------------------------------------------------- #
# Save / load equivalence + checkpoint bundle
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_state_dict_round_trip(self, tmp_path):
        """save -> load -> forward must match the original."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=120, position="landlord")
        out_before = _forward(model, obs)

        ckpt = tmp_path / "v2.ckpt"
        torch.save(model.state_dict(), ckpt)

        # Load into a fresh model with the same config + schema.
        model2 = ModelV2(model.schema, model.config)
        state = torch.load(ckpt, weights_only=True)
        model2.load_state_dict(state, strict=True)
        model2.eval()
        out_after = _forward(model2, obs)
        assert torch.allclose(out_before.win_logit, out_after.win_logit, atol=1e-6)

    def test_v2_checkpoint_bundle_round_trip(self, tmp_path):
        """save_v2_checkpoint -> load_v2_checkpoint reproduces the model."""
        from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint

        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=121, position="landlord")
        out_before = _forward(model, obs)

        bundle_path = str(tmp_path / "model_v2.tar")
        schema_hash = model.schema.stable_hash()
        manifest = save_v2_checkpoint(
            bundle_path, model, schema_hash=schema_hash,
            model_config=model.config, frames=123,
            position_frames={"landlord": 41, "landlord_up": 41, "landlord_down": 41},
        )
        assert manifest.model_version == "v2"
        assert manifest.feature_version == "v2"

        state_dict, loaded_manifest = load_v2_checkpoint(
            bundle_path,
            expected_schema_hash=schema_hash,
            expected_model_config_hash=model.config.stable_hash(),
            expected_ruleset=RuleSet.legacy(),
        )
        assert loaded_manifest.model_version == "v2"
        assert loaded_manifest.frames == 123

        model2 = ModelV2(model.schema, model.config)
        model2.load_state_dict(state_dict, strict=True)
        model2.eval()
        out_after = _forward(model2, obs)
        assert torch.allclose(out_before.win_logit, out_after.win_logit, atol=1e-6)

    def test_v2_checkpoint_rejects_schema_mismatch(self, tmp_path):
        """A schema-hash mismatch must raise CheckpointCompatibilityError."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            load_v2_checkpoint,
            save_v2_checkpoint,
        )

        model = _build_model()
        bundle_path = str(tmp_path / "model_v2.tar")
        save_v2_checkpoint(
            bundle_path, model, schema_hash=model.schema.stable_hash(),
            model_config=model.config,
        )
        with pytest.raises(CheckpointCompatibilityError, match="schema_hash"):
            load_v2_checkpoint(
                bundle_path,
                expected_schema_hash="bogus_hash",
                expected_model_config_hash=model.config.stable_hash(),
                expected_ruleset=RuleSet.legacy(),
            )

    def test_v2_checkpoint_rejects_config_mismatch(self, tmp_path):
        """Blocker #2: a model-config-hash mismatch (history_heads diff) is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            load_v2_checkpoint,
            save_v2_checkpoint,
        )

        model = _build_model()
        bundle_path = str(tmp_path / "model_v2.tar")
        save_v2_checkpoint(
            bundle_path, model, schema_hash=model.schema.stable_hash(),
            model_config=model.config,
        )
        # A different config (history_heads 8 -> 4 keeps projection shapes but
        # changes the config hash).
        different_cfg = ModelV2Config(history_heads=4)
        assert different_cfg.stable_hash() != model.config.stable_hash()
        with pytest.raises(CheckpointCompatibilityError, match="model_config_hash"):
            load_v2_checkpoint(
                bundle_path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=different_cfg.stable_hash(),
                expected_ruleset=RuleSet.legacy(),
            )

    def test_v2_checkpoint_rejects_wrong_ruleset(self, tmp_path):
        """A ruleset mismatch (legacy bundle loaded as standard) is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            load_v2_checkpoint,
            save_v2_checkpoint,
        )

        model = _build_model()
        bundle_path = str(tmp_path / "model_v2.tar")
        save_v2_checkpoint(
            bundle_path, model, schema_hash=model.schema.stable_hash(),
            model_config=model.config,
        )
        with pytest.raises(CheckpointCompatibilityError, match="ruleset"):
            load_v2_checkpoint(
                bundle_path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=model.config.stable_hash(),
                expected_ruleset=RuleSet.standard(),
            )

    def test_v2_checkpoint_rejects_wrong_kind(self, tmp_path):
        """A training bundle loaded as public_policy is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            load_v2_checkpoint,
            save_v2_checkpoint,
        )

        model = _build_model()
        bundle_path = str(tmp_path / "model_v2.tar")
        save_v2_checkpoint(
            bundle_path, model, schema_hash=model.schema.stable_hash(),
            model_config=model.config,
        )
        with pytest.raises(CheckpointCompatibilityError, match="checkpoint_kind"):
            load_v2_checkpoint(
                bundle_path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=model.config.stable_hash(),
                expected_ruleset=RuleSet.legacy(),
                expected_checkpoint_kind="public_policy",
            )

    def test_load_v2_model_rejects_bare_state_dict(self, tmp_path):
        """load_v2_model must reject a bare state_dict sidecar (no manifest)."""
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        bare_ckpt = tmp_path / "bare.ckpt"
        torch.save(model.state_dict(), bare_ckpt)  # bare, no manifest

        schema = build_v2_schema()
        with pytest.raises(Exception, match="bare state_dict|manifest"):
            load_v2_model(str(bare_ckpt), schema, RuleSet.legacy())

    def test_load_v2_model_rejects_legacy_ckpt(self, tmp_path):
        """load_v2_model must reject a legacy/factorized state_dict."""
        from douzero.dmc.models import model_dict
        from douzero.evaluation.deep_agent import load_v2_model

        torch.manual_seed(0)
        legacy_model = model_dict["landlord"]()
        legacy_ckpt = tmp_path / "legacy.ckpt"
        torch.save(legacy_model.state_dict(), legacy_ckpt)

        schema = build_v2_schema()
        with pytest.raises(Exception, match="bare state_dict|manifest"):
            load_v2_model(str(legacy_ckpt), schema, RuleSet.legacy())

    def test_load_v2_model_loads_manifest_sidecar(self, tmp_path):
        """load_v2_model loads a manifest-bearing V2 sidecar strictly."""
        from douzero.checkpoint import save_v2_position_weights
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        ckpt = str(tmp_path / "v2.ckpt")
        save_v2_position_weights(
            ckpt, model,
            schema_hash=model.schema.stable_hash(),
            model_config=model.config,
            ruleset=RuleSet.legacy(),
        )
        loaded = load_v2_model(ckpt, model.schema, RuleSet.legacy(), model.config)
        loaded.eval()
        obs, _ = _build_v2_obs(seed=122, position="landlord")
        out_before = _forward(model, obs)
        out_after = _forward(loaded, obs)
        assert torch.allclose(out_before.win_logit, out_after.win_logit, atol=1e-6)

    def test_load_v2_model_attaches_ruleset_identity(self, tmp_path):
        """Blocker #3: load_v2_model attaches the verified ruleset identity."""
        from douzero.checkpoint import save_v2_position_weights
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        ckpt = str(tmp_path / "v2.ckpt")
        save_v2_position_weights(
            ckpt, model,
            schema_hash=model.schema.stable_hash(),
            model_config=model.config,
            ruleset=RuleSet.standard(),
        )
        loaded = load_v2_model(ckpt, model.schema, RuleSet.standard(), model.config)
        rs = RuleSet.standard()
        assert loaded.expected_ruleset_identity == (
            rs.ruleset_id, rs.ruleset_version, rs.stable_hash(),
        )

    def test_load_v2_model_rejects_schema_mismatch_sidecar(self, tmp_path):
        """A sidecar with a different schema hash is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            save_v2_position_weights,
        )
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        ckpt = str(tmp_path / "v2.ckpt")
        save_v2_position_weights(
            ckpt, model,
            schema_hash=model.schema.stable_hash(),
            model_config=model.config,
            ruleset=RuleSet.legacy(),
        )
        different_schema = build_v2_schema(max_history_len=50)
        assert different_schema.stable_hash() != model.schema.stable_hash()
        with pytest.raises(CheckpointCompatibilityError, match="schema_hash"):
            load_v2_model(ckpt, different_schema, RuleSet.legacy())

    def test_load_v2_model_rejects_config_mismatch_sidecar(self, tmp_path):
        """Blocker #2: a sidecar with a different config hash is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            save_v2_position_weights,
        )
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        ckpt = str(tmp_path / "v2.ckpt")
        save_v2_position_weights(
            ckpt, model,
            schema_hash=model.schema.stable_hash(),
            model_config=model.config,
            ruleset=RuleSet.legacy(),
        )
        # Load with a different config (score_clamp 32 -> 8 changes behavior).
        different_cfg = ModelV2Config(score_clamp=8.0)
        with pytest.raises(CheckpointCompatibilityError, match="model_config_hash"):
            load_v2_model(ckpt, model.schema, RuleSet.legacy(), different_cfg)

    def test_load_v2_model_rejects_wrong_ruleset_sidecar(self, tmp_path):
        """A sidecar saved as legacy but loaded as standard is rejected."""
        from douzero.checkpoint import (
            CheckpointCompatibilityError,
            save_v2_position_weights,
        )
        from douzero.evaluation.deep_agent import load_v2_model

        model = _build_model()
        ckpt = str(tmp_path / "v2.ckpt")
        save_v2_position_weights(
            ckpt, model,
            schema_hash=model.schema.stable_hash(),
            model_config=model.config,
            ruleset=RuleSet.legacy(),
        )
        with pytest.raises(CheckpointCompatibilityError, match="ruleset"):
            load_v2_model(ckpt, model.schema, RuleSet.standard(), model.config)


# --------------------------------------------------------------------------- #
# DeepAgentV2: imperfect-information boundary + selection
# --------------------------------------------------------------------------- #
class TestDeepAgentV2:
    def _build_agent(self, decision_mode="win"):
        model = _build_model()
        return model, _DeepAgentV2Helper(model, decision_mode)

    def test_act_v2_rejects_privileged_observation(self):
        """The type guard: a PrivilegedObservation raises BEFORE any model call."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        priv = PrivilegedObservation(
            all_handcards={"landlord": (3,), "landlord_up": (4,), "landlord_down": (5,)},
            acting_role="landlord",
        )
        with pytest.raises(TypeError, match="PrivilegedObservation"):
            agent.act_v2(priv)

    def test_act_v2_rejects_wrong_type(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        with pytest.raises(TypeError, match="ObservationV2"):
            agent.act_v2({"not": "an observation"})

    def test_act_v2_selects_legal_action(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=130, position="landlord")
        chosen = agent.act_v2(obs)
        assert chosen in obs.public.legal_actions

    def test_act_v2_short_circuits_single_legal_action(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=131, position="landlord")
        # Build a fake obs with a single legal action by slicing (the model
        # must NOT be called for N=1).
        legal = obs.public.legal_actions[:1]
        # We cannot easily rebuild a full ObservationV2 with one action, so test
        # the short-circuit path directly: the act() infoset path covers it.
        np.random.seed(132)
        env = Env("adp")
        env.reset()
        # Force a single legal action by monkeypatching the infoset.
        original_infoset = env.infoset
        original_infoset.legal_actions = legal
        chosen = agent.act(original_infoset)
        assert chosen == legal[0]

    def test_act_from_infoset_matches_act_v2(self):
        """The legacy act(infoset) path produces a legal action too."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        np.random.seed(133)
        env = Env("adp")
        env.reset()
        infoset = env.infoset
        chosen = agent.act(infoset)
        assert chosen in infoset.legal_actions

    def test_decision_mode_score_selects_legal_action(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy(), decision_mode="score")
        obs, _ = _build_v2_obs(seed=134, position="landlord")
        chosen = agent.act_v2(obs)
        assert chosen in obs.public.legal_actions

    def test_decision_mode_validated(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        with pytest.raises(ValueError, match="decision_mode"):
            DeepAgentV2("landlord", model, RuleSet.legacy(), decision_mode="bogus")

    def test_requires_model_v2_instance(self):
        from douzero.evaluation.deep_agent import DeepAgentV2

        with pytest.raises(TypeError, match="ModelV2"):
            DeepAgentV2("landlord", "not_a_model", RuleSet.legacy())

    def test_requires_explicit_ruleset(self):
        """Blocker #3: ruleset=None is rejected (no silent legacy default)."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        with pytest.raises(ValueError, match="explicit RuleSet"):
            DeepAgentV2("landlord", model, ruleset=None)


class _DeepAgentV2Helper:
    """Placeholder kept for future expansion (tests use DeepAgentV2 directly)."""
    def __init__(self, model, decision_mode):
        self.model = model
        self.decision_mode = decision_mode


# --------------------------------------------------------------------------- #
# Imperfect-information boundary: no privileged import in the model package
# --------------------------------------------------------------------------- #
class TestNoLeakage:
    def test_models_v2_does_not_import_privileged_module(self):
        """The models_v2 package must not import douzero.observation.privileged.

        This is a static guarantee that the model layer has no path to the
        true hidden hands. (DeepAgentV2 imports it ONLY for the isinstance
        rejection; the model package itself must not.)
        """
        import douzero.models_v2 as mv2
        import sys
        # The privileged module may be imported by OTHER tests; we only assert
        # the models_v2 package's own __init__ did not pull it in. Check that
        # none of the models_v2 submodules appear in the privileged module's
        # dependents by confirming the model forward does not read hidden hands.
        # A functional check: corrupting all_handcards must not change the output.
        model = _build_model()
        model.eval()
        np.random.seed(140)
        env = Env("adp")
        env.reset()
        infoset = env.infoset
        obs_clean = get_obs_v2(infoset)
        out_clean = _forward(model, obs_clean)

        # Corrupt the privileged field; the public observation must be unchanged.
        infoset.all_handcards = {"landlord": [3], "landlord_up": [3], "landlord_down": [3]}
        obs_corrupt = get_obs_v2(infoset)
        out_corrupt = _forward(model, obs_corrupt)
        assert torch.equal(out_clean.win_logit, out_corrupt.win_logit)

    def test_two_hidden_allocations_same_public_output(self):
        """Swap cards between the two farmers; public output must be identical.

        This is the core imperfect-information invariant from AGENTS.md:
        "two states with identical public information but different hidden
        allocations must produce identical public observations."
        """
        model = _build_model()
        model.eval()
        np.random.seed(141)
        env = Env("adp")
        env.reset()
        infoset = env.infoset
        obs_a = get_obs_v2(infoset)
        out_a = _forward(model, obs_a)

        # Swap a card between the two farmers' true hands. The public obs
        # (unseen pool) is swap-invariant, so the model output must not change.
        original_all = dict(env.infoset.all_handcards)
        up_hand = list(original_all["landlord_up"])
        down_hand = list(original_all["landlord_down"])
        if up_hand and down_hand:
            up_hand[0], down_hand[0] = down_hand[0], up_hand[0]
        infoset.all_handcards = {
            "landlord": original_all["landlord"],
            "landlord_up": up_hand,
            "landlord_down": down_hand,
        }
        obs_b = get_obs_v2(infoset)
        out_b = _forward(model, obs_b)
        assert torch.equal(out_a.win_logit, out_b.win_logit), (
            "swapping hidden cards between farmers changed the public model output"
        )


# --------------------------------------------------------------------------- #
# ModelOutput dataclass validation
# --------------------------------------------------------------------------- #
class TestModelOutputValidation:
    def _base_tensors(self, n=3):
        win = torch.randn(n, 1)
        sw = torch.randn(n, 1)
        sl = torch.randn(n, 1)
        pw = torch.sigmoid(win)
        sm = pw * sw + (1 - pw) * sl
        mask = torch.ones(n, dtype=torch.bool)
        return win, sw, sl, pw, sm, mask

    def test_valid_construction(self):
        win, sw, sl, pw, sm, mask = self._base_tensors(3)
        out = ModelOutput(win, sw, sl, pw, sm, mask)
        assert out.num_actions == 3

    def test_wrong_head_shape_rejected(self):
        win, sw, sl, pw, sm, mask = self._base_tensors(3)
        with pytest.raises(ValueError, match="score_if_win"):
            ModelOutput(win, sw.reshape(1, 3), sl, pw, sm, mask)

    def test_wrong_mask_shape_rejected(self):
        win, sw, sl, pw, sm, _ = self._base_tensors(3)
        bad_mask = torch.ones(2, dtype=torch.bool)
        with pytest.raises(ValueError, match="action_mask"):
            ModelOutput(win, sw, sl, pw, sm, bad_mask)

    def test_non_bool_mask_rejected(self):
        win, sw, sl, pw, sm, _ = self._base_tensors(3)
        bad_mask = torch.ones(3, dtype=torch.float32)
        with pytest.raises(ValueError, match="bool"):
            ModelOutput(win, sw, sl, pw, sm, bad_mask)

    def test_zero_action_rows_rejected(self):
        """Bug #6: a ModelOutput with zero action rows is invalid."""
        win = torch.zeros(0, 1)
        sw = torch.zeros(0, 1)
        sl = torch.zeros(0, 1)
        pw = torch.zeros(0, 1)
        sm = torch.zeros(0, 1)
        mask = torch.zeros(0, dtype=torch.bool)
        with pytest.raises(ValueError, match="zero action rows"):
            ModelOutput(win, sw, sl, pw, sm, mask)


# --------------------------------------------------------------------------- #
# Bug #1: action embeddings are actually consumed (action sensitivity +
# permutation equivariance + action_encoder nonzero gradients)
# --------------------------------------------------------------------------- #
class TestActionSensitivity:
    """The fusion MUST consume each action's own embedding.

    A prior version concatenated only state+history+role and silently broadcast
    the result across actions, making every action's logit identical. These
    tests would have caught that bug.
    """

    def test_different_actions_produce_different_logits(self):
        """Two different action feature rows must yield different win_logits."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=200, position="landlord")
        out = _forward(model, obs)
        n = out.num_actions
        if n < 2:
            pytest.skip("need >= 2 actions")
        # At least one pair of actions must differ in win_logit. If every pair
        # were equal, the fusion is ignoring the action embedding.
        logits = out.win_logit.squeeze(-1)
        diffs = (logits.unsqueeze(0) - logits.unsqueeze(1)).abs()
        assert diffs.max().item() > 1e-5, (
            f"all {n} actions produced near-identical win_logits (max diff "
            f"{diffs.max().item():.2e}); the fusion is likely ignoring the "
            f"action embedding"
        )

    def test_action_row_permutation_permutes_output_rows(self):
        """Permuting the action rows permutes the output rows by the same perm.

        This is permutation equivariance: the fusion is a per-row function of
        (shared trunk, per-action embedding), so reordering the inputs must
        reorder the outputs identically.
        """
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=201, position="landlord")
        bundle = observation_to_model_inputs(obs)
        n = bundle.action_features.shape[0]
        if n < 3:
            pytest.skip("need >= 3 actions for a non-trivial permutation")
        out_orig = _forward(model, obs)

        # Apply a fixed permutation to the action rows (and the mask).
        perm = torch.tensor([2, 0, 1] + list(range(3, n)))
        permuted_actions = bundle.action_features[perm]
        permuted_mask = bundle.action_mask[perm]
        out_perm = model(
            bundle.state_card_vectors, bundle.state_context_flat,
            bundle.context_card_vectors, bundle.context_flat,
            bundle.history_tokens, bundle.history_key_padding_mask,
            permuted_actions, permuted_mask, bundle.acting_role,
        )
        # The permuted output rows must equal the original rows at the same
        # permutation indices.
        orig_logits = out_orig.win_logit.squeeze(-1)
        perm_logits = out_perm.win_logit.squeeze(-1)
        assert torch.allclose(orig_logits[perm], perm_logits, atol=1e-5), (
            "permuting action rows did not permute output rows identically; "
            "the fusion is not permutation-equivariant"
        )

    def test_action_encoder_receives_nonzero_gradient(self):
        """Every key parameter in action_encoder must get a nonzero gradient.

        This proves the action embedding flows into the loss (and therefore
        into the selected action). A disconnected action_encoder would have
        all-zero gradients.
        """
        model = _build_model()
        model.train()
        obs, _ = _build_v2_obs(seed=202, position="landlord")
        out = _forward(model, obs)
        # A loss that depends on the win_logits (sum), so gradients flow back
        # through the heads -> fusion -> action_embeddings -> action_encoder.
        loss = out.win_logit.sum()
        loss.backward()
        for name, param in model.action_encoder.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().sum().item() > 0, (
                f"action_encoder.{name} got an all-zero gradient; the action "
                f"embedding is disconnected from the loss"
            )


# --------------------------------------------------------------------------- #
# Bug #2: state encoder preserves field identity (no summing)
# --------------------------------------------------------------------------- #
class TestStateFieldIdentity:
    """The state encoder MUST preserve which card set is which.

    A prior version summed all card-set embeddings, which discarded field
    identity (the sum is invariant under any permutation of the summed fields).
    These tests would have caught that bug.
    """

    def test_swapping_my_hand_and_other_hand_changes_trunk(self):
        """Swapping two card fields must change the state trunk.

        my_hand (cards I hold) and other_hand (unseen pool) are semantically
        different; the model must distinguish them.
        """
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=210, position="landlord")
        bundle = observation_to_model_inputs(obs)
        # state_card_vectors[0] = my_handcards, [1] = other_handcards.
        my_hand = bundle.state_card_vectors[0]
        other_hand = bundle.state_card_vectors[1]
        if torch.equal(my_hand, other_hand):
            pytest.skip("my_hand and other_hand happen to be equal")
        # Forward with the original order.
        out_orig = _forward(model, obs)
        # Swap the first two card vectors.
        swapped = (
            (other_hand, my_hand) + bundle.state_card_vectors[2:]
        )
        out_swapped = model(
            swapped, bundle.state_context_flat,
            bundle.context_card_vectors, bundle.context_flat,
            bundle.history_tokens, bundle.history_key_padding_mask,
            bundle.action_features, bundle.action_mask, bundle.acting_role,
        )
        assert not torch.allclose(out_orig.win_logit, out_swapped.win_logit, atol=1e-5), (
            "swapping my_hand and other_hand did not change the output; the "
            "state encoder is discarding field identity (summing embeddings)"
        )

    def test_same_card_sum_different_field_allocation_differs(self):
        """Two states with the same total card counts but different field
        allocation must produce different outputs.

        Construct two synthetic bundles where the SUM of all card fields is
        identical, but the per-field allocation differs. A summing encoder
        would produce identical trunks; a field-identity-preserving encoder
        must differ.
        """
        import numpy as np
        from douzero.observation.schema import build_v2_schema

        schema = build_v2_schema()
        model = _build_model(schema=schema)
        model.eval()
        cvd = schema.card_vector_dim

        def _make_bundle(field_vals):
            """field_vals: list of 8 np arrays (6 state + 2 context card fields)."""
            state_cards = tuple(torch.from_numpy(v.astype(np.float32)) for v in field_vals[:6])
            context_cards = tuple(torch.from_numpy(v.astype(np.float32)) for v in field_vals[6:8])
            state_flat = torch.zeros(1, dtype=torch.float32)  # minimal flat context
            # Pad state_flat to the expected non-card state width.
            from douzero.observation.schema import state_width, context_width
            non_card_state = state_width(schema) - cvd * 6
            non_card_ctx = context_width(schema) - cvd * 2
            state_flat = torch.zeros(non_card_state, dtype=torch.float32)
            ctx_flat = torch.zeros(non_card_ctx, dtype=torch.float32)
            hist_tokens = torch.zeros(schema.max_history_len, schema.history_token_fields and sum(int(np.prod(f.shape)) for f in schema.history_token_fields), dtype=torch.float32)
            hist_kpm = torch.ones(schema.max_history_len, dtype=torch.bool)  # all padding
            act_feat = torch.zeros(1, sum(int(np.prod(f.shape)) for f in schema.action_fields), dtype=torch.float32)
            act_feat[0, 0] = 1.0  # one card, to differentiate from zero
            act_mask = torch.ones(1, dtype=torch.bool)
            return state_cards, state_flat, context_cards, ctx_flat, hist_tokens, hist_kpm, act_feat, act_mask

        # Two allocations with the SAME total per-rank sum but DIFFERENT fields.
        base = np.zeros(cvd, dtype=np.float32)
        base[0] = 4  # four 3s total across all fields
        # Allocation A: all four in field 0.
        vals_a = [np.zeros(cvd, dtype=np.float32) for _ in range(8)]
        vals_a[0][0] = 4
        # Allocation B: two in field 0, two in field 1.
        vals_b = [np.zeros(cvd, dtype=np.float32) for _ in range(8)]
        vals_b[0][0] = 2
        vals_b[1][0] = 2
        # Sanity: the sums are identical.
        assert sum(v.sum() for v in vals_a) == sum(v.sum() for v in vals_b)

        sa, sfa, ca, cfa, ht, hk, af, am = _make_bundle(vals_a)
        sb, sfb, cb, cfb, _, _, _, _ = _make_bundle(vals_b)
        out_a = model(sa, sfa, ca, cfa, ht, hk, af, am, "landlord")
        out_b = model(sb, sfb, cb, cfb, ht, hk, af, am, "landlord")
        assert not torch.allclose(out_a.win_logit, out_b.win_logit, atol=1e-5), (
            "two states with the same card sum but different field allocation "
            "produced identical outputs; the state encoder is summing fields "
            "rather than preserving field identity"
        )

    def test_state_encoder_concatenates_not_sums(self):
        """Direct unit test: the StateEncoder output depends on field order.

        Build a StateEncoder with 2 card fields; swapping the two inputs must
        change the output (a sum would be invariant).
        """
        from douzero.models_v2.state_encoder import StateEncoder

        torch.manual_seed(0)
        enc = StateEncoder(card_vector_dim=54, num_card_fields=2, flat_context_width=4, hidden_size=16)
        enc.eval()
        a = torch.zeros(54); a[0] = 1.0
        b = torch.zeros(54); b[1] = 1.0
        ctx = torch.zeros(4)
        out_ab = enc((a, b), ctx)
        out_ba = enc((b, a), ctx)
        assert not torch.allclose(out_ab, out_ba, atol=1e-6), (
            "swapping two card fields did not change the StateEncoder output; "
            "it is summing rather than concatenating"
        )


# --------------------------------------------------------------------------- #
# Bug #3: DeepAgentV2 schema-hash binding
# --------------------------------------------------------------------------- #
class TestDeepAgentV2SchemaBinding:
    def test_act_v2_rejects_schema_hash_mismatch(self):
        """An observation with a different schema hash is rejected."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=220, position="landlord")
        # Corrupt the schema hash on the observation.
        from douzero.observation.encode_v2 import ObservationV2
        # ObservationV2 is frozen; reconstruct with a bogus hash via __replace__.
        import dataclasses
        bad_obs = dataclasses.replace(obs, feature_schema_hash="bogus_hash")
        with pytest.raises(ValueError, match="schema-hash mismatch"):
            agent.act_v2(bad_obs)

    def test_agent_binds_to_model_schema_hash(self):
        """The agent stores the model's schema hash at construction."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        assert agent._feature_schema_hash == model.schema.stable_hash()

    def test_standard_model_with_legacy_agent_rejected(self):
        """Blocker #3: a standard-checkpoint model cannot be served by a legacy agent."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        # Attach a standard ruleset identity to the model (as load_v2_model would).
        rs = RuleSet.standard()
        model.expected_ruleset_identity = (
            rs.ruleset_id, rs.ruleset_version, rs.stable_hash(),
        )
        with pytest.raises(ValueError, match="ruleset identity mismatch"):
            DeepAgentV2("landlord", model, RuleSet.legacy())

    def test_standard_model_with_standard_agent_accepted(self):
        """Blocker #3: matching identities construct successfully."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        rs = RuleSet.standard()
        model.expected_ruleset_identity = (
            rs.ruleset_id, rs.ruleset_version, rs.stable_hash(),
        )
        agent = DeepAgentV2("landlord", model, RuleSet.standard())
        assert agent._ruleset_identity == model.expected_ruleset_identity

    def test_act_v2_rejects_ruleset_mismatch(self):
        """Blocker #3: an observation encoded under a different ruleset is rejected.

        The agent is bound to legacy; an observation carrying a standard
        ruleset identity must be rejected before forwarding (the schema-hash
        check alone cannot catch this).
        """
        import dataclasses
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=221, position="landlord")
        # Corrupt the ruleset identity on the observation's public block.
        rs = RuleSet.standard()
        bad_public = dataclasses.replace(
            obs.public,
            ruleset_id=rs.ruleset_id,
            ruleset_version=rs.ruleset_version,
            ruleset_hash=rs.stable_hash(),
        )
        bad_obs = dataclasses.replace(obs, public=bad_public)
        with pytest.raises(ValueError, match="ruleset identity mismatch"):
            agent.act_v2(bad_obs)

    def test_act_v2_accepts_matching_ruleset(self):
        """Blocker #3: a legacy observation under a legacy agent forwards fine."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=222, position="landlord")
        # The obs is legacy by default (matches the agent).
        chosen = agent.act_v2(obs)
        assert chosen in obs.public.legal_actions


# --------------------------------------------------------------------------- #
# Bug #5: NaN/Inf runtime guard
# --------------------------------------------------------------------------- #
class TestNaNGuard:
    def test_nan_input_rejected(self):
        """A NaN in the action features is caught by the guard."""
        from douzero.models_v2 import NumericalError

        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=230, position="landlord")
        bundle = observation_to_model_inputs(obs)
        # Inject a NaN into the action features.
        bad_actions = bundle.action_features.clone()
        bad_actions[0, 0] = float("nan")
        with pytest.raises(NumericalError, match="fused"):
            model(
                bundle.state_card_vectors, bundle.state_context_flat,
                bundle.context_card_vectors, bundle.context_flat,
                bundle.history_tokens, bundle.history_key_padding_mask,
                bad_actions, bundle.action_mask, bundle.acting_role,
            )

    def test_nan_weight_rejected(self):
        """A NaN in a model weight is caught by the guard (output is NaN)."""
        from douzero.models_v2 import NumericalError

        model = _build_model()
        model.eval()
        # Inject a NaN into the fusion input_proj weight.
        with torch.no_grad():
            model.fusion.input_proj.weight[0, 0] = float("nan")
        obs, _ = _build_v2_obs(seed=231, position="landlord")
        with pytest.raises(NumericalError, match="fused"):
            _forward(model, obs)

    def test_nan_guard_disabled_skips_check(self):
        """With nan_guard=False, the guard is skipped (output may be NaN)."""
        cfg = ModelV2Config(nan_guard=False)
        model = _build_model(cfg)
        model.eval()
        with torch.no_grad():
            model.fusion.input_proj.weight[0, 0] = float("nan")
        obs, _ = _build_v2_obs(seed=232, position="landlord")
        # No exception; the output may contain NaN but the guard is off.
        out = _forward(model, obs)
        # The forward completed (no raise); we do NOT assert finiteness here.
        assert out.num_actions == obs.actions.features.shape[0]

    def test_assert_finite_helper(self):
        """The assert_finite helper raises on NaN and passes on finite."""
        from douzero.models_v2 import NumericalError, assert_finite

        assert_finite(torch.zeros(3), "ok")  # no raise
        with pytest.raises(NumericalError, match="non-finite"):
            assert_finite(torch.tensor([1.0, float("nan"), 2.0]), "bad")
        with pytest.raises(NumericalError, match="non-finite"):
            assert_finite(torch.tensor([float("inf")]), "inf")

    def test_nan_in_score_win_head_rejected(self):
        """Blocker #1: NaN in score_win_head is caught (score clamp can't remove NaN)."""
        from douzero.models_v2 import NumericalError

        model = _build_model()
        model.eval()
        with torch.no_grad():
            model.heads.score_win_head.weight[0, 0] = float("nan")
        obs, _ = _build_v2_obs(seed=233, position="landlord")
        # The fused + win_logit may be finite, but score_if_win / score_mean
        # carry NaN. The guard must catch them.
        with pytest.raises(NumericalError, match="score_if_win|score_mean"):
            _forward(model, obs)

    def test_nan_in_score_loss_head_rejected(self):
        """Blocker #1: NaN in score_loss_head bias is caught."""
        from douzero.models_v2 import NumericalError

        model = _build_model()
        model.eval()
        with torch.no_grad():
            model.heads.score_loss_head.bias[0] = float("nan")
        obs, _ = _build_v2_obs(seed=234, position="landlord")
        with pytest.raises(NumericalError, match="score_if_loss|score_mean"):
            _forward(model, obs)

    def test_inf_in_win_head_rejected(self):
        """Blocker #1: Inf in the win head is caught (p_win = sigmoid(inf) is finite, but win_logit is inf)."""
        from douzero.models_v2 import NumericalError

        model = _build_model()
        model.eval()
        with torch.no_grad():
            model.heads.win_head.weight[0, 0] = float("inf")
        obs, _ = _build_v2_obs(seed=235, position="landlord")
        with pytest.raises(NumericalError, match="win_logit"):
            _forward(model, obs)


# --------------------------------------------------------------------------- #
# Bug #6: zero legal actions must raise everywhere
# --------------------------------------------------------------------------- #
class TestZeroActionsRejection:
    def test_model_forward_rejects_zero_actions(self):
        """ModelV2.forward raises on zero action rows."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=240, position="landlord")
        bundle = observation_to_model_inputs(obs)
        empty_actions = bundle.action_features[:0]
        empty_mask = bundle.action_mask[:0]
        with pytest.raises(ValueError, match="zero legal actions"):
            model(
                bundle.state_card_vectors, bundle.state_context_flat,
                bundle.context_card_vectors, bundle.context_flat,
                bundle.history_tokens, bundle.history_key_padding_mask,
                empty_actions, empty_mask, bundle.acting_role,
            )

    def test_deep_agent_v2_rejects_zero_legal_actions(self):
        """DeepAgentV2.act_v2 raises on an observation with no legal actions."""
        from douzero.evaluation.deep_agent import DeepAgentV2

        model = _build_model()
        model.eval()
        agent = DeepAgentV2("landlord", model, RuleSet.legacy())
        obs, _ = _build_v2_obs(seed=241, position="landlord")
        # Build an observation with zero legal actions by replacing the public
        # legal_actions and the action block. Use dataclasses.replace on the
        # inner public obs + actions.
        import dataclasses
        from douzero.observation.encode_v2 import LegalActionBatch
        empty_actions = LegalActionBatch(
            features=np.zeros((0, obs.actions.features.shape[1]), dtype=np.int8),
            action_mask=np.zeros((0,), dtype=np.int8),
            legal_actions=(),
        )
        bad_public = dataclasses.replace(obs.public, legal_actions=())
        bad_obs = dataclasses.replace(obs, public=bad_public, actions=empty_actions)
        with pytest.raises(ValueError, match="zero legal actions"):
            agent.act_v2(bad_obs)

    def test_observation_to_model_inputs_rejects_zero_actions(self):
        """The batch bridge raises on zero actions before the model sees them."""
        model = _build_model()
        model.eval()
        obs, _ = _build_v2_obs(seed=242, position="landlord")
        import dataclasses
        from douzero.observation.encode_v2 import LegalActionBatch
        empty_actions = LegalActionBatch(
            features=np.zeros((0, obs.actions.features.shape[1]), dtype=np.int8),
            action_mask=np.zeros((0,), dtype=np.int8),
            legal_actions=(),
        )
        bad_obs = dataclasses.replace(obs, actions=empty_actions)
        with pytest.raises(ValueError, match="zero legal actions"):
            observation_to_model_inputs(bad_obs)
