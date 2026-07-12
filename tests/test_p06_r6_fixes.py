"""P06 r6 tests: model identity closure, P05 checkpoint migration, and
replay transition role/label alignment.

Covers the three blockers from the r5 review:

1. **Nested model identity**: ``model_version`` (top-level) and
   ``model.version`` (nested ``model:`` block) can contradict each other.
   The loader must cross-validate them, and the identity gate must check
   both.

2. **P05 checkpoint compatibility**: adding ``score_target_transform`` to
   ``ModelV2Config.compatibility_dict()`` changed the model-config hash,
   silently breaking all P05-format checkpoints. The loader must migrate
   P05 checkpoints (identity version 1 / absent) by computing the v1 hash
   and allowing load only when runtime ``score_target_transform == "raw"``.

3. **Transition role/label alignment**: ``Transition.position`` selects
   the team-perspective terminal label, but was not checked against
   ``obs.public.acting_role``. A farmer observation with a landlord
   position would silently train on mismatched labels.

Plus the non-blocking fix: ``_UNSUPPORTED_LEGACY_FIELDS`` now includes
``xpid``, ``save_interval``, and ``savedir``.
"""

from __future__ import annotations

import pytest
import torch

from douzero.models_v2.config import ModelV2Config


# --------------------------------------------------------------------------- #
# Blocker 1: nested model identity closure
# --------------------------------------------------------------------------- #
def test_loader_rejects_top_v2_nested_legacy():
    """Top-level model_version=v2 + nested model.version=legacy is rejected."""
    from douzero.config.loader import _build_training_config

    raw = {
        "feature_version": "v2",
        "ruleset": "legacy",
        "model_version": "v2",
        "objective": "adp",
        "model": {"version": "legacy"},
    }
    with pytest.raises(ValueError, match="model.version.*must match model_version"):
        _build_training_config(raw)


def test_loader_rejects_top_v2_nested_bogus():
    """Nested model.version='unknown-model' is rejected (unsupported set)."""
    from douzero.config.loader import _build_training_config

    raw = {
        "feature_version": "v2",
        "ruleset": "legacy",
        "model_version": "v2",
        "objective": "adp",
        "model": {"version": "unknown-model"},
    }
    # Both the mismatch and the unsupported-set check fire; the cross-
    # validation runs first.
    with pytest.raises(ValueError):
        _build_training_config(raw)


def test_loader_no_model_block_defaults_nested_to_top():
    """Without a model: block, model.version is set to model_version."""
    from douzero.config.loader import _build_training_config

    raw = {
        "feature_version": "v2",
        "ruleset": "legacy",
        "model_version": "v2",
        "objective": "adp",
    }
    cfg = _build_training_config(raw)
    assert cfg.model.version == "v2"
    assert cfg.model_version == "v2"


def test_loader_both_v2_passes():
    """Top-level v2 + nested v2 is accepted."""
    from douzero.config.loader import _build_training_config

    raw = {
        "feature_version": "v2",
        "ruleset": "legacy",
        "model_version": "v2",
        "objective": "adp",
        "model": {"version": "v2", "hidden_size": 128},
    }
    cfg = _build_training_config(raw)
    assert cfg.model.version == "v2"
    assert cfg.model_version == "v2"
    assert cfg.model.hidden_size == 128


def test_train_v2_identity_gate_checks_nested_model_version():
    """train_v2._assert_v2_identity rejects a config whose nested model.version
    is not 'v2', even if model_version is 'v2'."""
    import train_v2

    class FakeCfg:
        feature_version = "v2"
        model_version = "v2"
        ruleset = "legacy"
        objective = "adp"

        class model:
            version = "legacy"

    with pytest.raises(ValueError, match="model.version.*v2"):
        train_v2._assert_v2_identity(FakeCfg())


# --------------------------------------------------------------------------- #
# Blocker 2: P05 V2 checkpoint migration
# --------------------------------------------------------------------------- #
def _build_v2_model(**cfg_kwargs):
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    torch.manual_seed(42)
    return ModelV2(build_v2_schema(), ModelV2Config(**cfg_kwargs))


def test_v1_hash_excludes_score_target_transform():
    """compatibility_dict_v1() does not include score_target_transform."""
    cfg = ModelV2Config(score_target_transform="raw")
    v1 = cfg.compatibility_dict_v1()
    assert "score_target_transform" not in v1


def test_v1_hash_same_for_raw_and_signed_log():
    """v1 hash is the same regardless of score_target_transform (P05 identity)."""
    cfg_raw = ModelV2Config(score_target_transform="raw")
    cfg_log = ModelV2Config(score_target_transform="signed_log")
    assert cfg_raw.stable_hash_v1() == cfg_log.stable_hash_v1()


def test_v2_hash_different_for_raw_and_signed_log():
    """v2 hash (current) differs between raw and signed_log."""
    cfg_raw = ModelV2Config(score_target_transform="raw")
    cfg_log = ModelV2Config(score_target_transform="signed_log")
    assert cfg_raw.stable_hash() != cfg_log.stable_hash()


def test_v1_hash_differs_from_v2_hash():
    """The v1 and v2 hashes are different for the same config."""
    cfg = ModelV2Config()
    assert cfg.stable_hash_v1() != cfg.stable_hash()


def test_save_then_load_round_trip_v2(tmp_path):
    """Save and load a P06 r6 checkpoint: identity version 2, strict match."""
    from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
    from douzero.env.rules import RuleSet

    model = _build_v2_model()
    path = str(tmp_path / "model_v2.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    state_dict, manifest = load_v2_checkpoint(
        path,
        expected_schema_hash=model.schema.stable_hash(),
        expected_model_config_hash=model.config.stable_hash(),
        expected_ruleset=ruleset,
        runtime_model_config=model.config,
    )
    assert isinstance(state_dict, dict)


def test_p05_checkpoint_loads_under_raw_transform(tmp_path):
    """A P05-format checkpoint (no identity version key) loads when runtime
    score_target_transform='raw'."""
    from douzero.checkpoint import save_v2_checkpoint, load_v2_checkpoint
    from douzero.checkpoint.v2 import (
        _MANIFEST_KEY,
        _MODEL_CONFIG_HASH_KEY,
        _MODEL_CONFIG_IDENTITY_VERSION_KEY,
        _SCHEMA_HASH_KEY,
        _V2_STATE_DICT_KEY,
    )
    from douzero.env.rules import RuleSet

    model = _build_v2_model(score_target_transform="raw")
    path = str(tmp_path / "p05_model_v2.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    # Simulate a P05 checkpoint: rewrite the bundle without the identity
    # version key and with the v1 hash.
    bundle = torch.load(path, weights_only=False)
    v1_hash = model.config.stable_hash_v1()
    bundle[_MODEL_CONFIG_HASH_KEY] = v1_hash
    bundle.pop(_MODEL_CONFIG_IDENTITY_VERSION_KEY, None)
    torch.save(bundle, path)

    # Load under raw: should succeed.
    state_dict, _ = load_v2_checkpoint(
        path,
        expected_schema_hash=model.schema.stable_hash(),
        expected_model_config_hash=model.config.stable_hash(),
        expected_ruleset=ruleset,
        runtime_model_config=model.config,
    )
    assert isinstance(state_dict, dict)


def test_p05_checkpoint_rejected_under_signed_log_transform(tmp_path):
    """A P05-format checkpoint is rejected when runtime requests signed_log."""
    from douzero.checkpoint import save_v2_checkpoint, load_v2_checkpoint
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet
    from douzero.checkpoint.io import CheckpointCompatibilityError

    model = _build_v2_model(score_target_transform="raw")
    path = str(tmp_path / "p05_signed_log.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    # Strip identity version and write the v1 hash.
    bundle = torch.load(path, weights_only=False)
    bundle["model_config_hash"] = model.config.stable_hash_v1()
    bundle.pop(_MODEL_CONFIG_IDENTITY_VERSION_KEY, None)
    torch.save(bundle, path)

    # Runtime requests signed_log → must reject.
    runtime_cfg = ModelV2Config(score_target_transform="signed_log")
    with pytest.raises(CheckpointCompatibilityError, match="P05.*raw"):
        load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=runtime_cfg.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=runtime_cfg,
        )


def test_p05_checkpoint_rejected_when_v1_hash_mismatches(tmp_path):
    """A P05-format checkpoint with a wrong v1 hash is rejected even under raw."""
    from douzero.checkpoint import save_v2_checkpoint, load_v2_checkpoint
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet
    from douzero.checkpoint.io import CheckpointCompatibilityError

    model = _build_v2_model()
    path = str(tmp_path / "p05_bad_hash.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    # Strip identity version and write a garbage hash.
    bundle = torch.load(path, weights_only=False)
    bundle["model_config_hash"] = "0" * 64
    bundle.pop(_MODEL_CONFIG_IDENTITY_VERSION_KEY, None)
    torch.save(bundle, path)

    with pytest.raises(CheckpointCompatibilityError, match=r"\(P05 migration\)"):
        load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=model.config.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=model.config,
        )


def test_v2_checkpoint_rejected_on_hash_mismatch(tmp_path):
    """A P06 r6 checkpoint (identity version 2) rejects a hash mismatch."""
    from douzero.checkpoint import save_v2_checkpoint, load_v2_checkpoint
    from douzero.env.rules import RuleSet
    from douzero.checkpoint.io import CheckpointCompatibilityError

    model = _build_v2_model()
    path = str(tmp_path / "v2_mismatch.tar")
    ruleset = RuleSet.legacy()
    save_v2_checkpoint(path, model, ruleset=ruleset)

    wrong_cfg = ModelV2Config(hidden_size=128)
    with pytest.raises(CheckpointCompatibilityError, match="model_config_hash mismatch"):
        load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=wrong_cfg.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=wrong_cfg,
        )


def test_sidecar_p05_migration_round_trip(tmp_path):
    """Save→load a deployment sidecar with P05 format + raw transform."""
    from douzero.checkpoint import save_v2_position_weights, load_v2_position_weights
    from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
    from douzero.env.rules import RuleSet

    model = _build_v2_model(score_target_transform="raw")
    path = str(tmp_path / "p05_sidecar.ckpt")
    ruleset = RuleSet.legacy()
    save_v2_position_weights(path, model, ruleset=ruleset)

    # Simulate P05 by stripping the identity version and writing v1 hash.
    bundle = torch.load(path, weights_only=False)
    bundle["model_config_hash"] = model.config.stable_hash_v1()
    bundle.pop(_MODEL_CONFIG_IDENTITY_VERSION_KEY, None)
    torch.save(bundle, path)

    state_dict, _ = load_v2_position_weights(
        path,
        expected_schema_hash=model.schema.stable_hash(),
        expected_model_config_hash=model.config.stable_hash(),
        expected_ruleset=ruleset,
        runtime_model_config=model.config,
    )
    assert isinstance(state_dict, dict)


# --------------------------------------------------------------------------- #
# Blocker 3: Transition position vs observation acting_role
# --------------------------------------------------------------------------- #
def _fake_obs_for_role(role: str):
    """Build a real ObservationV2 for the given acting role."""
    from douzero.env.env import Env
    from douzero.observation.encode_v2 import get_obs_v2

    import numpy as np

    np.random.seed(999)
    env = Env("adp")
    env.reset()
    while env._acting_player_position != role:
        env.step(env.infoset.legal_actions[0])
    return get_obs_v2(env.infoset)


def test_transition_rejects_farmer_obs_with_landlord_position():
    """Transition.validate rejects obs.acting_role=landlord_up +
    position=landlord."""
    from douzero.training.v2_buffer import Transition

    obs = _fake_obs_for_role("landlord_up")
    tr = Transition(
        obs=obs,
        action_index=0,
        position="landlord",
        target_win=1.0,
        target_score=1.0,
        target_log_score=0.5,
    )
    with pytest.raises(ValueError, match="position.*does not match.*acting_role"):
        tr.validate()


def test_transition_rejects_landlord_obs_with_farmer_position():
    """Transition.validate rejects obs.acting_role=landlord +
    position=landlord_up."""
    from douzero.training.v2_buffer import Transition

    obs = _fake_obs_for_role("landlord")
    tr = Transition(
        obs=obs,
        action_index=0,
        position="landlord_up",
        target_win=1.0,
        target_score=1.0,
        target_log_score=0.5,
    )
    with pytest.raises(ValueError, match="position.*does not match.*acting_role"):
        tr.validate()


def test_transition_accepts_matching_role_and_position():
    """Transition.validate passes when position == obs.acting_role."""
    from douzero.training.v2_buffer import Transition

    for role in ("landlord", "landlord_up", "landlord_down"):
        obs = _fake_obs_for_role(role)
        tr = Transition(
            obs=obs,
            action_index=0,
            position=role,
            target_win=1.0,
            target_score=1.0,
            target_log_score=0.5,
        )
        tr.validate()  # must not raise


def test_buffer_rejects_mismatched_transition():
    """V2ReplayBuffer.add_episode rejects a transition whose position does not
    match the observation's acting_role."""
    from douzero.training.v2_buffer import Episode, Transition, V2ReplayBuffer

    obs = _fake_obs_for_role("landlord")
    tr = Transition(
        obs=obs,
        action_index=0,
        position="landlord_up",  # mismatch!
        target_win=1.0,
        target_score=1.0,
        target_log_score=0.5,
    )
    buf = V2ReplayBuffer(capacity_transitions=10)
    with pytest.raises(ValueError, match="position.*does not match.*acting_role"):
        buf.add_episode(Episode(transitions=[tr], terminal_result={}))


# --------------------------------------------------------------------------- #
# Non-blocking: _UNSUPPORTED_LEGACY_FIELDS completeness
# --------------------------------------------------------------------------- #
def test_unsupported_legacy_fields_includes_xpid_save_interval_savedir():
    """The warning set covers all legacy multiprocess fields the V2 trainer
    ignores."""
    from train_v2 import _UNSUPPORTED_LEGACY_FIELDS

    for field_name in ("xpid", "save_interval", "savedir"):
        assert field_name in _UNSUPPORTED_LEGACY_FIELDS, (
            f"{field_name} should be in _UNSUPPORTED_LEGACY_FIELDS so a "
            f"non-default value triggers a visible warning"
        )


def test_enhanced_yaml_triggers_xpid_warning(capsys):
    """configs/enhanced.yaml sets xpid to a non-default value, so the startup
    warning fires."""
    import train_v2
    from douzero.config import load_config

    cfg = load_config("configs/enhanced.yaml")
    train_v2._warn_unsupported_legacy_fields(cfg)
    captured = capsys.readouterr()
    assert "xpid" in captured.err
