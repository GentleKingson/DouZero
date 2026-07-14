"""P06 r7 tests: close remaining identity and loss-boundary gaps.

Covers the three blockers from the r6 review:

1. **Unknown identity version fail-closed**: values like ``0``, ``3``,
   ``"2"``, ``True``, ``None`` must not silently enter the P05 migration
   path. Only absent / ``1`` → migration; ``2`` → strict; everything
   else → reject.

2. **Signed-log clamp consistency**: the loss target is clamped to
   ``score_clamp`` in BOTH modes (r5 made the clamp universal), so a
   model with ``score_clamp=8`` paired with a loss ``score_clamp=32``
   must be rejected regardless of the transform.

3. **Action-index/mask validation**: the public loss API's
   ``_gather_action()`` must reject negative indices, out-of-bounds
   indices, non-integer dtypes, and indices pointing at padded
   (``action_mask=False``) rows.

Plus the non-blocking fix: ``IDENTITY_VERSION`` declared as
``ClassVar[int]`` so the dataclass machinery does not treat it as an
instance field.
"""

from __future__ import annotations

import math

import pytest
import torch

from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.output import ModelOutput


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_model_output(n_actions: int, mask: list[bool] | None = None) -> ModelOutput:
    """Build a minimal ModelOutput with ``n_actions`` rows."""
    if mask is None:
        mask = [True] * n_actions
    return ModelOutput(
        win_logit=torch.randn(n_actions, 1),
        score_if_win=torch.randn(n_actions, 1),
        score_if_loss=torch.randn(n_actions, 1),
        p_win=torch.rand(n_actions, 1),
        score_mean=torch.randn(n_actions, 1),
        action_mask=torch.tensor(mask, dtype=torch.bool),
    )


def _build_v2_model(**cfg_kwargs):
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    torch.manual_seed(42)
    return ModelV2(build_v2_schema(), ModelV2Config(**cfg_kwargs))


# --------------------------------------------------------------------------- #
# Blocker 1: Unknown identity version fail-closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_version", [0, 4, 999, "2", True, False, "absent"])
def test_unknown_identity_version_rejected(tmp_path, bad_version):
    """Identity versions other than absent/1/2/3 are rejected (fail-closed).

    ``"absent"`` is in the list to verify the absent-key path is treated as
    version 1 (P05 migration), NOT rejected — it should succeed under raw.
    """
    from douzero.checkpoint import load_v2_checkpoint
    from douzero.checkpoint.io import CheckpointCompatibilityError
    from douzero.env.rules import RuleSet

    model = _build_v2_model(score_target_transform="raw")

    # For "absent" (P05 migration), we also need to write the v1 hash so
    # the migration succeeds. For bad versions, we keep the v2 hash.
    from douzero.checkpoint import save_v2_checkpoint
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet

    path = str(tmp_path / f"mangle_{bad_version}.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    bundle = torch.load(path, weights_only=False)
    if bad_version == "absent":
        bundle.pop(_MODEL_CONFIG_IDENTITY_VERSION_KEY, None)
        bundle["model_config_hash"] = model.config.stable_hash_v1()
    else:
        bundle[_MODEL_CONFIG_IDENTITY_VERSION_KEY] = bad_version
    torch.save(bundle, path)

    if bad_version == "absent":
        # Absent key → P05 migration → should succeed under raw.
        state_dict, _ = load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=model.config.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=model.config,
        )
        assert isinstance(state_dict, dict)
    elif bad_version is True or bad_version is False:
        # bool is always rejected regardless of value.
        with pytest.raises(CheckpointCompatibilityError, match="bool"):
            load_v2_checkpoint(
                path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=model.config.stable_hash(),
                expected_ruleset=ruleset,
                runtime_model_config=model.config,
            )
    else:
        with pytest.raises(CheckpointCompatibilityError, match="unsupported"):
            load_v2_checkpoint(
                path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=model.config.stable_hash(),
                expected_ruleset=ruleset,
                runtime_model_config=model.config,
            )


def test_identity_version_1_treated_as_p05(tmp_path):
    """Identity version 1 is treated as P05 and loads under raw."""
    from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet

    model = _build_v2_model(score_target_transform="raw")
    path = str(tmp_path / "v1_id.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    # Set identity version 1 and the v1 hash for the migration to succeed.
    bundle = torch.load(path, weights_only=False)
    bundle[_MODEL_CONFIG_IDENTITY_VERSION_KEY] = 1
    bundle["model_config_hash"] = model.config.stable_hash_v1()
    torch.save(bundle, path)

    state_dict, _ = load_v2_checkpoint(
        path,
        expected_schema_hash=model.schema.stable_hash(),
        expected_model_config_hash=model.config.stable_hash(),
        expected_ruleset=ruleset,
        runtime_model_config=model.config,
    )
    assert isinstance(state_dict, dict)


def test_current_identity_version_strict_match(tmp_path):
    """The current identity version with matching hash loads."""
    from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet

    model = _build_v2_model()
    path = str(tmp_path / "v2_id.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    # The current identity version is stamped by save_v2_checkpoint.
    bundle = torch.load(path, weights_only=False)
    assert bundle[_MODEL_CONFIG_IDENTITY_VERSION_KEY] == 3
    torch.save(bundle, path)

    state_dict, _ = load_v2_checkpoint(
        path,
        expected_schema_hash=model.schema.stable_hash(),
        expected_model_config_hash=model.config.stable_hash(),
        expected_ruleset=ruleset,
        runtime_model_config=model.config,
    )
    assert isinstance(state_dict, dict)


# --------------------------------------------------------------------------- #
# Blocker 2: signed-log clamp consistency
# --------------------------------------------------------------------------- #
def test_trainer_rejects_clamp_mismatch_signed_log():
    """score_clamp mismatch is rejected in signed_log mode (P06 r7)."""
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(score_clamp=8.0, score_target_transform="signed_log"),
    )
    loss_cfg = LossConfig(score_clamp=32.0, score_target_transform="signed_log")
    with pytest.raises(ValueError, match="score_clamp"):
        V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))


def test_trainer_accepts_matching_clamp_signed_log():
    """Matching score_clamp is accepted in signed_log mode."""
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(score_clamp=16.0, score_target_transform="signed_log"),
    )
    loss_cfg = LossConfig(score_clamp=16.0, score_target_transform="signed_log")
    trainer = V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))
    assert trainer is not None


def test_trainer_rejects_clamp_mismatch_raw():
    """score_clamp mismatch is rejected in raw mode (unchanged from r2)."""
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(score_clamp=8.0, score_target_transform="raw"),
    )
    loss_cfg = LossConfig(score_clamp=32.0, score_target_transform="raw")
    with pytest.raises(ValueError, match="score_clamp"):
        V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))


# --------------------------------------------------------------------------- #
# Blocker 3: Action-index validation in _gather_action
# --------------------------------------------------------------------------- #
def test_gather_action_rejects_negative_index():
    """A negative action index must not wrap to the tail."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="negative"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([-1])})


def test_gather_action_rejects_out_of_bounds():
    """An index >= num_actions is rejected."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="out-of-bounds"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([4])})
    with pytest.raises(ValueError, match="out-of-bounds"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([5])})


def test_gather_action_rejects_float_dtype():
    """Float action indices are rejected (would be silently truncated)."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(TypeError, match="integer dtype"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([0.0])})


def test_gather_action_rejects_bool_dtype():
    """Bool action indices are rejected (True/False → 1/0 silently)."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(TypeError, match="torch.bool"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([True])})


def test_gather_action_rejects_padded_row():
    """An index pointing at a padded (action_mask=False) row is rejected."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    # 4 rows; row 3 is padding.
    output = _make_model_output(4, mask=[True, True, True, False])
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="padded"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([3])})


def test_gather_action_accepts_valid_index():
    """A valid index on a real (non-padded) row produces a finite loss."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
        "action_indices": torch.tensor([2]),
    }
    components = fn.forward(output, labels)
    # win is a scalar loss component; verify it is finite.
    assert math.isfinite(float(components.win))


def test_gather_action_rejects_empty_indices():
    """An empty action_indices tensor is rejected."""
    from douzero.training.losses import MultiObjectiveLoss, LossConfig

    output = _make_model_output(4)
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="empty"):
        fn.forward(output, {**labels, "action_indices": torch.tensor([], dtype=torch.long)})


# --------------------------------------------------------------------------- #
# Non-blocking: IDENTITY_VERSION as ClassVar
# --------------------------------------------------------------------------- #
def test_identity_version_is_classvar_not_instance_field():
    """IDENTITY_VERSION is a ClassVar, not a dataclass field, so it cannot
    be passed as a constructor argument."""
    from dataclasses import fields

    field_names = {f.name for f in fields(ModelV2Config)}
    assert "IDENTITY_VERSION" not in field_names, (
        "IDENTITY_VERSION should be ClassVar, not a dataclass field"
    )
    assert ModelV2Config.IDENTITY_VERSION == 3


def test_identity_version_not_constructable():
    """Passing IDENTITY_VERSION=999 to the constructor should raise TypeError
    (unexpected keyword argument) since it is a ClassVar."""
    with pytest.raises(TypeError):
        ModelV2Config(IDENTITY_VERSION=999)  # type: ignore[call-arg]
