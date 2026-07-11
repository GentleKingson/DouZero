"""Tests for the training ruleset guard (P02 Slice 5).

Verifies that:
- train() rejects ruleset='standard' with a precise error
- train() accepts ruleset='legacy' (does not raise the guard error)
- the --ruleset CLI flag shows both choices in --help
- checkpoint manifest records the correct ruleset_id
"""

from __future__ import annotations

import argparse

import pytest


# --------------------------------------------------------------------------- #
# Training guard
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
