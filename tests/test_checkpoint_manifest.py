"""Tests for versioned checkpoint manifests (P01 Slice 3, consolidated).

Acceptance gates:
  - new-format model.tar round-trips with a complete, consistent manifest;
  - legacy model.tar (no manifest) still loads via the compat path;
  - incompatible schema/model_version/feature_version/ruleset_id/checkpoint_kind
    each raise a precise CheckpointCompatibilityError;
  - unknown model_version is rejected;
  - git_sha is always a string ("unknown" when git is unavailable);
  - the legacy per-position .ckpt sidecar path still works (DeepAgent);
  - the strict position loader rejects key-set and shape mismatches;
  - manifest is stored as a plain dict (weights_only-loadable).
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

from douzero.checkpoint import (
    CURRENT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointManifest,
    build_manifest,
    load_checkpoint,
    load_legacy_model_tar,
    load_legacy_position_ckpt,
    load_position_state_dict_strict,
    save_checkpoint,
)
from douzero.dmc.models import Model, model_dict

POSITIONS = ["landlord", "landlord_up", "landlord_down"]


def _make_models_and_optimizers(seed: int = 900):
    torch.manual_seed(seed)
    learner = Model(device="cpu")
    optimizers = {p: torch.optim.RMSprop(learner.parameters(p)) for p in POSITIONS}
    return learner, optimizers


def _ns(**kw) -> argparse.Namespace:
    """Build a flags Namespace with the version fields defaulted to legacy."""
    base = dict(feature_version="legacy", ruleset="legacy", model_version="legacy",
                savedir="/tmp", xpid="t")
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #
def test_build_manifest_git_sha_is_string():
    m = build_manifest(_ns(), frames=0, position_frames={p: 0 for p in POSITIONS})
    assert isinstance(m.git_sha, str)
    assert m.git_sha  # non-empty


def test_build_manifest_defaults_to_legacy_identity():
    m = build_manifest(None, frames=0, position_frames={p: 0 for p in POSITIONS})
    assert m.feature_version == "legacy"
    assert m.ruleset_id == "legacy"
    assert m.model_version == "legacy"
    assert m.checkpoint_kind == "training_checkpoint"
    assert m.schema_version == CURRENT_SCHEMA_VERSION


def test_build_manifest_rejects_unknown_kind():
    with pytest.raises(ValueError, match="checkpoint_kind"):
        build_manifest(_ns(), frames=0,
                       position_frames={p: 0 for p in POSITIONS},
                       checkpoint_kind="not_a_real_kind")


def test_manifest_to_dict_round_trip():
    m = build_manifest(_ns(), frames=3, position_frames={p: 3 for p in POSITIONS})
    d = m.to_dict()
    # Stored form is a plain dict of primitives (no dataclass), so it is
    # loadable under torch weights_only=True.
    assert isinstance(d, dict)
    assert "checkpoint_kind" in d and "model_version" in d
    m2 = CheckpointManifest.from_dict(d)
    assert m2 == m


# --------------------------------------------------------------------------- #
# save_checkpoint / load_checkpoint round-trip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_preserves_tensors_and_manifest(tmp_path, seed_factory):
    seed_factory(901)
    learner, optimizers = _make_models_and_optimizers()
    stats = {"loss_landlord": 1.5}
    position_frames = {"landlord": 10, "landlord_up": 10, "landlord_down": 10}

    path = str(tmp_path / "model.tar")
    manifest = save_checkpoint(
        path, learner.get_models(), optimizers, stats, _ns(), 10, position_frames
    )

    bundle, loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.frames == 10
    assert loaded.position_frames == position_frames
    assert loaded.checkpoint_kind == "training_checkpoint"
    for key in ("model_state_dict", "optimizer_state_dict", "stats", "flags", "frames", "position_frames"):
        assert key in bundle
    # manifest stored as plain dict in the bundle
    assert isinstance(bundle["manifest"], dict)

    reloaded = Model(device="cpu")
    for p in POSITIONS:
        reloaded.get_model(p).load_state_dict(bundle["model_state_dict"][p])
        for a, b in zip(learner.get_model(p).parameters(), reloaded.get_model(p).parameters()):
            assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Legacy compatibility (no manifest key)
# --------------------------------------------------------------------------- #
def test_legacy_model_tar_without_manifest_loads(tmp_path, seed_factory):
    seed_factory(902)
    learner, optimizers = _make_models_and_optimizers()
    path = str(tmp_path / "legacy.tar")
    torch.save(
        {
            "model_state_dict": {k: learner.get_model(k).state_dict() for k in POSITIONS},
            "optimizer_state_dict": {k: optimizers[k].state_dict() for k in POSITIONS},
            "stats": {"loss_landlord": 0.0},
            "flags": {"xpid": "legacy"},
            "frames": 5,
            "position_frames": {p: 5 for p in POSITIONS},
        },
        path,
    )
    bundle, manifest = load_checkpoint(path)
    assert manifest is None
    assert bundle["frames"] == 5


def test_load_legacy_model_tar_directly(tmp_path):
    path = str(tmp_path / "x.tar")
    torch.save({"frames": 7, "model_state_dict": {}}, path)
    bundle, manifest = load_legacy_model_tar(path)
    assert manifest is None
    assert bundle["frames"] == 7


# --------------------------------------------------------------------------- #
# Incompatible manifest -> precise error, never silent
# --------------------------------------------------------------------------- #
def _save_with(tmp_path, **version_kw):
    learner, optimizers = _make_models_and_optimizers()
    path = str(tmp_path / "v.tar")
    save_checkpoint(path, learner.get_models(), optimizers, {}, _ns(**version_kw), 0,
                    {p: 0 for p in POSITIONS})
    return path


def test_incompatible_feature_version_raises(tmp_path):
    path = _save_with(tmp_path, feature_version="v2")
    with pytest.raises(CheckpointCompatibilityError, match="feature_version"):
        load_checkpoint(path, expected_feature_version="legacy")


def test_incompatible_ruleset_raises(tmp_path):
    path = _save_with(tmp_path, ruleset="standard")
    with pytest.raises(CheckpointCompatibilityError, match="ruleset_id"):
        load_checkpoint(path, expected_ruleset_id="legacy")


def test_incompatible_model_version_raises(tmp_path):
    path = _save_with(tmp_path, model_version="v2")
    with pytest.raises(CheckpointCompatibilityError, match="model_version"):
        load_checkpoint(path, expected_model_version="legacy")


def test_incompatible_schema_version_raises(tmp_path):
    path = _save_with(tmp_path)
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        load_checkpoint(path, expected_schema_version=CURRENT_SCHEMA_VERSION + 1)


def test_incompatible_checkpoint_kind_raises(tmp_path):
    """A training_checkpoint must not load where a position_weights is expected."""
    path = _save_with(tmp_path)
    with pytest.raises(CheckpointCompatibilityError, match="checkpoint_kind"):
        load_checkpoint(path, expected_checkpoint_kind="position_weights")


def test_unknown_model_version_is_rejected(tmp_path):
    """An unknown model_version value raises (not silently accepted)."""
    path = _save_with(tmp_path, model_version="v9_unknown")
    # Loading against a different expected value must raise.
    with pytest.raises(CheckpointCompatibilityError, match="model_version"):
        load_checkpoint(path, expected_model_version="legacy")


# --------------------------------------------------------------------------- #
# Position-weights sidecar: permissive (legacy) + strict (new)
# --------------------------------------------------------------------------- #
def test_load_legacy_position_ckpt_roundtrip(tmp_path, seed_factory):
    seed_factory(907)
    torch.manual_seed(907)
    model = model_dict["landlord"]()
    path = str(tmp_path / "landlord.ckpt")
    torch.save(model.state_dict(), path)
    loaded = load_legacy_position_ckpt(path)
    assert set(loaded.keys()) == set(model.state_dict().keys())


def test_strict_position_load_accepts_matching_ckpt(tmp_path, seed_factory):
    seed_factory(908)
    torch.manual_seed(908)
    model = model_dict["landlord"]()
    path = str(tmp_path / "landlord.ckpt")
    torch.save(model.state_dict(), path)
    loaded = load_position_state_dict_strict(path, model.state_dict())
    assert set(loaded.keys()) == set(model.state_dict().keys())


def test_strict_position_load_rejects_missing_key(tmp_path, seed_factory):
    seed_factory(909)
    torch.manual_seed(909)
    model = model_dict["landlord"]()
    sd = model.state_dict()
    # Drop one key to simulate a partial state_dict.
    partial = {k: v for k, v in sd.items() if k != "dense6.weight"}
    path = str(tmp_path / "partial.ckpt")
    torch.save(partial, path)
    with pytest.raises(CheckpointCompatibilityError, match="key-set mismatch"):
        load_position_state_dict_strict(path, sd)


def test_strict_position_load_rejects_extra_key(tmp_path, seed_factory):
    seed_factory(910)
    torch.manual_seed(910)
    model = model_dict["landlord"]()
    sd = dict(model.state_dict())
    sd["bogus_extra_key"] = torch.zeros(2)
    path = str(tmp_path / "extra.ckpt")
    torch.save(sd, path)
    fresh = model_dict["landlord"]().state_dict()
    with pytest.raises(CheckpointCompatibilityError, match="key-set mismatch"):
        load_position_state_dict_strict(path, fresh)


# --------------------------------------------------------------------------- #
# git_sha string contract via environment simulation
# --------------------------------------------------------------------------- #
def test_manifest_git_sha_unknown_when_no_git(monkeypatch):
    monkeypatch.delenv("DOUZERO_GIT_SHA", raising=False)
    import douzero._version as v

    monkeypatch.setattr(v, "_run_git", lambda *a, **k: None)
    monkeypatch.setattr(v.git_sha, "_cached", v._SENTINEL, raising=False)

    m = build_manifest(_ns(), frames=0, position_frames={p: 0 for p in POSITIONS})
    assert m.git_sha == "unknown"


# --------------------------------------------------------------------------- #
# Security: weights_only=True by default; unsafe pickle opt-in (item 3)
# --------------------------------------------------------------------------- #
def test_safe_training_checkpoint_round_trip(tmp_path, seed_factory):
    """A P01 training checkpoint must load under the safe default (weights_only=True).

    This is the core item-3 gate: the default load_checkpoint() call uses no
    allow_unsafe_pickle flag and must succeed for a locally-saved P01 bundle.
    """
    seed_factory(920)
    learner, optimizers = _make_models_and_optimizers()
    position_frames = {p: 5 for p in POSITIONS}
    path = str(tmp_path / "safe.tar")
    save_checkpoint(path, learner.get_models(), optimizers, {"loss_landlord": 1.0},
                    _ns(), 5, position_frames)
    # Default call: no allow_unsafe_pickle -> weights_only=True.
    bundle, manifest = load_checkpoint(path)
    assert manifest is not None
    assert manifest.frames == 5
    for p in POSITIONS:
        assert p in bundle["model_state_dict"]


def test_unsafe_pickle_rejected_by_default(tmp_path):
    """A checkpoint with a non-safe global must be REFUSED by the safe default.

    We craft a bundle containing a datetime object (a common, benign-but-not-
    weights-only-safe global) and assert the default load raises. The
    allow_unsafe_pickle=True path succeeds UNLESS the environment has set
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 (a hard security override that rejects
    weights_only=False even when the caller opts in).
    """
    import datetime as _dt

    bundle = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "stats": {"created_at": _dt.datetime(2026, 1, 1)},  # non-safe object
        "frames": 0,
        "position_frames": {},
        "manifest": {
            "schema_version": 1, "model_version": "legacy",
            "feature_version": "legacy", "ruleset_id": "legacy",
            "checkpoint_kind": "training_checkpoint",
            "git_sha": "unknown", "python_version": "3.11",
            "torch_version": "2.13", "effective_config": {},
            "frames": 0, "position_frames": {},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    }
    path = str(tmp_path / "unsafe.tar")
    torch.save(bundle, path)

    # Default: weights_only=True -> must refuse (UnpicklingError).
    with pytest.raises(Exception):
        load_checkpoint(path)

    # Explicit opt-in: weights_only=False -> succeeds, unless the environment
    # hard-forces weights_only (a security override we cannot disable).
    force_safe = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") == "1"
    if force_safe:
        with pytest.raises(Exception):
            load_checkpoint(path, allow_unsafe_pickle=True)
    else:
        bundle2, manifest = load_checkpoint(path, allow_unsafe_pickle=True)
        assert manifest is not None
        assert isinstance(bundle2["stats"]["created_at"], _dt.datetime)


def test_no_trusted_constant_or_weights_only_false_default():
    """The TRAINING_CHECKPOINT_TRUSTED pseudo-gate must be gone (item 3)."""
    import douzero.checkpoint as ckpt_mod

    assert not hasattr(ckpt_mod, "TRAINING_CHECKPOINT_TRUSTED")
    assert not hasattr(ckpt_mod.io, "TRAINING_CHECKPOINT_TRUSTED")


def test_legacy_model_tar_safe_default_round_trip(tmp_path):
    """A legacy model.tar (no manifest) with safe contents loads under the default."""
    path = str(tmp_path / "legacy_safe.tar")
    torch.save(
        {"model_state_dict": {}, "frames": 3, "stats": {"a": 1}, "flags": {"x": 1}},
        path,
    )
    bundle, manifest = load_checkpoint(path)
    assert manifest is None
    assert bundle["frames"] == 3


def test_legacy_model_tar_unsafe_default_refused(tmp_path):
    """A legacy model.tar with a non-safe object is refused under the default.

    The allow_unsafe_pickle=True opt-in succeeds unless the environment hard-
    forces weights_only (TORCH_FORCE_WEIGHTS_ONLY_LOAD=1).
    """
    import datetime as _dt

    path = str(tmp_path / "legacy_unsafe.tar")
    torch.save(
        {"model_state_dict": {}, "frames": 3, "stats": {"ts": _dt.datetime(2026, 1, 1)}},
        path,
    )
    # Default (safe) must raise.
    with pytest.raises(Exception):
        load_checkpoint(path)
    # Opt-in succeeds (or is hard-refused under weights_only forcing).
    force_safe = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") == "1"
    if force_safe:
        with pytest.raises(Exception):
            load_checkpoint(path, allow_unsafe_pickle=True)
    else:
        bundle, manifest = load_checkpoint(path, allow_unsafe_pickle=True)
        assert manifest is None
        assert bundle["frames"] == 3


def test_manifest_torch_version_is_plain_str(tmp_path, seed_factory):
    """The manifest torch_version must be a native str (not TorchVersion).

    A TorchVersion object (str subclass) breaks weights_only=True loading, so
    the manifest must coerce it. This test guards the regression.
    """
    seed_factory(921)
    learner, optimizers = _make_models_and_optimizers()
    path = str(tmp_path / "ver.tar")
    manifest = save_checkpoint(path, learner.get_models(), optimizers, {},
                               _ns(), 0, {p: 0 for p in POSITIONS})
    assert type(manifest.torch_version) is str  # exact type, not a subclass
    # And it loads under the safe default (the real proof).
    _, loaded = load_checkpoint(path)
    assert loaded is not None
    assert type(loaded.torch_version) is str
