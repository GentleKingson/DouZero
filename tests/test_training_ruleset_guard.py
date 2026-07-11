"""Tests for the training ruleset and feature_version guards (P02/P03).

Verifies that:
- train() rejects ruleset='standard' with a precise error
- train() rejects feature_version='v2' with a precise error, BEFORE any init
- train() accepts the legacy defaults (does not raise the guard error)
- the --ruleset / --feature_version CLI flags show their choices in --help
- checkpoint manifest records the correct ruleset_id
"""

from __future__ import annotations

import argparse

import pytest


# --------------------------------------------------------------------------- #
# Training guard — ruleset (P02)
# --------------------------------------------------------------------------- #
def test_train_rejects_standard_ruleset():
    """train() must raise ValueError when ruleset='standard'."""
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        ruleset='standard',
        actor_device_cpu=True,
        training_device='cpu',
    )
    with pytest.raises(ValueError, match="Training does not yet support"):
        train(flags)


def test_train_does_not_reject_legacy_ruleset():
    """train() must NOT raise the ruleset guard for legacy.

    It may fail later for other reasons (e.g. missing FileWriter setup), but
    the specific 'Training does not yet support' error must not be raised.
    """
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        ruleset='legacy',
        actor_device_cpu=True,
        training_device='cpu',
    )
    try:
        train(flags)
    except ValueError as e:
        if "Training does not yet support" in str(e):
            pytest.fail("Legacy ruleset was rejected by the guard")
        # Other ValueErrors are fine (the train function will fail on
        # incomplete flags, but the guard must not be the cause).
    except Exception:
        # Any non-ValueError exception is also fine — the guard passed.
        pass


def test_train_rejects_missing_ruleset_defaults_to_legacy():
    """If ruleset attribute is missing, it defaults to 'legacy' (no guard)."""
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        actor_device_cpu=True,
        training_device='cpu',
    )
    try:
        train(flags)
    except ValueError as e:
        if "Training does not yet support" in str(e):
            pytest.fail("Missing ruleset should default to legacy, not be rejected")
    except Exception:
        pass


def test_train_standard_guard_has_no_side_effects(tmp_path):
    """The standard-ruleset guard must fail BEFORE any CUDA/model/process/
    checkpoint initialization — no subprocess, no checkpoint, no side effects.

    We verify this by checking that after the guard raises:
    - No checkpoint directory was created.
    - No subprocess was spawned (we check via a mock).
    """
    from unittest.mock import patch
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        ruleset='standard',
        actor_device_cpu=True,
        training_device='cpu',
        savedir=str(tmp_path),
        xpid='test_no_side_effect',
    )
    # Patch multiprocessing.Process to detect if any subprocess would be spawned.
    with patch('torch.multiprocessing.Process') as mock_proc:
        with pytest.raises(ValueError, match="Training does not yet support"):
            train(flags)
        # No subprocess should have been started.
        assert mock_proc.call_count == 0, "train() spawned a subprocess before the guard"
    # No checkpoint directory should have been created.
    assert not (tmp_path / 'test_no_side_effect').exists(), \
        "train() created a checkpoint directory before the guard"


# --------------------------------------------------------------------------- #
# Training guard — feature_version (P03)
# --------------------------------------------------------------------------- #
def test_train_rejects_v2_feature_version():
    """train() must raise ValueError when feature_version='v2'.

    The V2 observation schema is accepted by configuration but not yet wired
    into the actor/learner, so training must refuse it up front rather than
    silently producing a checkpoint stamped with the wrong feature identity.
    """
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        feature_version='v2',
        ruleset='legacy',
        actor_device_cpu=True,
        training_device='cpu',
    )
    with pytest.raises(ValueError, match="feature_version"):
        train(flags)


def test_train_does_not_reject_legacy_feature_version():
    """train() must NOT raise the feature_version guard for legacy."""
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        feature_version='legacy',
        ruleset='legacy',
        actor_device_cpu=True,
        training_device='cpu',
    )
    try:
        train(flags)
    except ValueError as e:
        if "feature_version" in str(e):
            pytest.fail("Legacy feature_version was rejected by the guard")
    except Exception:
        pass


def test_train_v2_guard_runs_before_any_initialization(tmp_path):
    """The feature_version guard must fire BEFORE CUDA/FileWriter/checkpoint/
    model/buffer/actor init — no subprocess, no checkpoint dir, no side effects.

    This is the critical ordering property: an identity-mismatched run must fail
    fast without spawning actors or writing artifacts that would need cleanup.
    """
    from unittest.mock import patch
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        feature_version='v2',
        ruleset='legacy',
        actor_device_cpu=True,
        training_device='cpu',
        savedir=str(tmp_path),
        xpid='test_v2_no_side_effect',
    )
    with patch('douzero.dmc.file_writer.FileWriter') as mock_fw, \
         patch('torch.multiprocessing.Process') as mock_proc:
        with pytest.raises(ValueError, match="feature_version"):
            train(flags)
        # No FileWriter, no subprocess before the guard.
        assert mock_fw.call_count == 0, \
            "train() constructed a FileWriter before the feature_version guard"
        assert mock_proc.call_count == 0, \
            "train() spawned a subprocess before the feature_version guard"
    assert not (tmp_path / 'test_v2_no_side_effect').exists(), \
        "train() created a checkpoint directory before the feature_version guard"


def test_train_feature_version_guard_precedes_cuda_check():
    """The feature_version guard must run before the CUDA-availability check.

    We force a CUDA-required config (actor_device_cpu=False) with a v2 feature
    version; the guard must raise ValueError (feature_version), NOT the
    AssertionError about CUDA. This proves the guard precedes CUDA init.
    """
    from douzero.dmc.dmc import train

    flags = argparse.Namespace(
        feature_version='v2',
        ruleset='legacy',
        actor_device_cpu=False,  # would trigger the CUDA check
        training_device='0',
    )
    with pytest.raises(ValueError, match="feature_version"):
        train(flags)


# --------------------------------------------------------------------------- #
# CLI --help shows both ruleset choices
# --------------------------------------------------------------------------- #
def test_train_help_shows_both_ruleset_choices():
    """--ruleset must accept both 'legacy' and 'standard'."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "legacy" in result.stdout
    assert "standard" in result.stdout


def test_cli_ruleset_choices():
    from douzero.dmc.arguments import parser

    ns = parser.parse_args(["--ruleset", "standard"])
    assert ns.ruleset == "standard"
    ns = parser.parse_args(["--ruleset", "legacy"])
    assert ns.ruleset == "legacy"
    ns = parser.parse_args([])
    assert ns.ruleset == "legacy"


# --------------------------------------------------------------------------- #
# Checkpoint manifest ruleset_id
# --------------------------------------------------------------------------- #
def test_manifest_records_legacy_ruleset_id():
    """build_manifest with ruleset='legacy' must produce ruleset_id='legacy'."""
    from douzero.checkpoint.manifest import build_manifest

    flags = argparse.Namespace(ruleset='legacy')
    manifest = build_manifest(flags, frames=0, position_frames={})
    assert manifest.ruleset_id == 'legacy'


def test_manifest_records_standard_ruleset_id():
    """build_manifest with ruleset='standard' must produce ruleset_id='standard'."""
    from douzero.checkpoint.manifest import build_manifest

    flags = argparse.Namespace(ruleset='standard')
    manifest = build_manifest(flags, frames=0, position_frames={})
    assert manifest.ruleset_id == 'standard'


def test_manifest_defaults_to_legacy_when_ruleset_missing():
    """build_manifest defaults ruleset_id to 'legacy' when flags lack it."""
    from douzero.checkpoint.manifest import build_manifest

    flags = argparse.Namespace()
    manifest = build_manifest(flags, frames=0, position_frames={})
    assert manifest.ruleset_id == 'legacy'


# --------------------------------------------------------------------------- #
# Checkpoint compatibility validation for standard
# --------------------------------------------------------------------------- #
def test_checkpoint_validation_accepts_standard(tmp_path):
    """load_checkpoint with expected_ruleset_id='standard' must accept a standard checkpoint."""
    import torch
    from douzero.checkpoint.io import load_checkpoint, save_checkpoint
    from douzero.checkpoint.manifest import build_manifest

    # Build a minimal standard checkpoint.
    flags = argparse.Namespace(ruleset='standard')
    manifest = build_manifest(flags, frames=100, position_frames={'landlord': 100})
    bundle = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'stats': {},
        'flags': vars(flags),
        'frames': 100,
        'position_frames': {'landlord': 100},
        'manifest': manifest.to_dict(),
    }
    ckpt_path = str(tmp_path / "standard_model.tar")
    torch.save(bundle, ckpt_path)

    loaded_bundle, loaded_manifest = load_checkpoint(
        ckpt_path, expected_ruleset_id='standard'
    )
    assert loaded_manifest.ruleset_id == 'standard'


def test_checkpoint_validation_rejects_ruleset_mismatch(tmp_path):
    """A legacy checkpoint must be rejected when expected_ruleset_id='standard'."""
    import torch
    from douzero.checkpoint.io import CheckpointCompatibilityError, load_checkpoint
    from douzero.checkpoint.manifest import build_manifest

    flags = argparse.Namespace(ruleset='legacy')
    manifest = build_manifest(flags, frames=0, position_frames={})
    bundle = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'stats': {},
        'flags': vars(flags),
        'frames': 0,
        'position_frames': {},
        'manifest': manifest.to_dict(),
    }
    ckpt_path = str(tmp_path / "legacy_model.tar")
    torch.save(bundle, ckpt_path)

    with pytest.raises(CheckpointCompatibilityError, match="ruleset_id"):
        load_checkpoint(ckpt_path, expected_ruleset_id='standard')


def test_checkpoint_manifest_records_ruleset_version_and_hash():
    """Manifest must record ruleset_version and ruleset_hash."""
    from douzero.checkpoint.manifest import build_manifest
    from douzero.env.rules import RuleSet

    # Legacy manifest.
    flags = argparse.Namespace(ruleset='legacy')
    manifest = build_manifest(flags, frames=0, position_frames={})
    assert manifest.ruleset_version == 'legacy-v1'
    assert manifest.ruleset_hash == RuleSet.legacy().stable_hash()

    # Standard manifest.
    flags = argparse.Namespace(ruleset='standard')
    manifest = build_manifest(flags, frames=0, position_frames={})
    assert manifest.ruleset_version == 'standard-v1'
    assert manifest.ruleset_hash == RuleSet.standard().stable_hash()


def test_checkpoint_same_id_different_hash_rejected(tmp_path):
    """A checkpoint with same ruleset_id but different hash must be rejected."""
    import torch
    from douzero.checkpoint.io import CheckpointCompatibilityError, load_checkpoint
    from douzero.checkpoint.manifest import CheckpointManifest

    # Create a manifest with standard ID but a wrong hash.
    manifest = CheckpointManifest(
        schema_version=1,
        model_version='legacy',
        feature_version='legacy',
        ruleset_id='standard',
        ruleset_version='standard-v1',
        ruleset_hash='0' * 64,  # wrong hash
        checkpoint_kind='training_checkpoint',
        git_sha='unknown',
        python_version='3.11',
        torch_version='2.0',
        effective_config={},
        frames=0,
        position_frames={},
        created_at='2026-01-01T00:00:00+00:00',
    )
    bundle = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'stats': {},
        'flags': {},
        'frames': 0,
        'position_frames': {},
        'manifest': manifest.to_dict(),
    }
    ckpt_path = str(tmp_path / "wrong_hash.tar")
    torch.save(bundle, ckpt_path)

    with pytest.raises(CheckpointCompatibilityError, match="ruleset_hash"):
        load_checkpoint(ckpt_path, expected_ruleset_id='standard')


def test_p01_checkpoint_without_ruleset_version_loads(tmp_path):
    """A P01 manifest (no ruleset_version/hash) must load with legacy defaults."""
    import torch
    from douzero.checkpoint.io import load_checkpoint

    # Simulate a P01 manifest: no ruleset_version/ruleset_hash fields.
    p01_manifest = {
        "schema_version": 1,
        "model_version": "legacy",
        "feature_version": "legacy",
        "ruleset_id": "legacy",
        "checkpoint_kind": "training_checkpoint",
        "git_sha": "unknown",
        "python_version": "3.11",
        "torch_version": "2.0",
        "effective_config": {},
        "frames": 0,
        "position_frames": {},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    bundle = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'stats': {},
        'flags': {},
        'frames': 0,
        'position_frames': {},
        'manifest': p01_manifest,
    }
    ckpt_path = str(tmp_path / "p01.tar")
    torch.save(bundle, ckpt_path)

    loaded_bundle, loaded_manifest = load_checkpoint(
        ckpt_path, expected_ruleset_id='legacy'
    )
    # Backfilled defaults.
    assert loaded_manifest.ruleset_version == 'legacy-v1'
    from douzero.env.rules import RuleSet
    assert loaded_manifest.ruleset_hash == RuleSet.legacy().stable_hash()
