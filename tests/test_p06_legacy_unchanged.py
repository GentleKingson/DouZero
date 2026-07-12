"""P06 legacy training path regression: legacy ``train.py`` is untouched.

Asserts that:

- The legacy ``compute_loss`` in ``douzero.dmc.dmc`` is still the plain-MSE
  single-target loss (the multi-objective losses live in
  :mod:`douzero.training.losses` only).
- The legacy ``TrainingConfig`` defaults are unchanged when the new ``loss``
  and ``decision_policy`` blocks are absent (all zeros -> legacy MSE path).
- ``configs/legacy.yaml`` loads cleanly with the new nested blocks.
- The V2 / multi-objective modules do NOT get imported by the legacy
  training path's top-level imports.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from douzero.config import load_config
from douzero.config.schemas import LossConfig as SchemaLossConfig


def test_legacy_compute_loss_is_plain_mse():
    from douzero.dmc.dmc import compute_loss

    import torch

    logits = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([0.0, 1.0, 2.0])
    loss = compute_loss(logits, targets)
    expected = ((logits.squeeze(-1) - targets) ** 2).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_legacy_config_defaults_have_zero_loss_weights():
    """The new LossConfig defaults preserve the legacy path (all zeros)."""
    cfg = SchemaLossConfig()
    assert cfg.lambda_win == 0.0
    assert cfg.lambda_score == 0.0
    assert cfg.lambda_uncertainty == 0.0
    # score_target_transform defaults to "raw" (the heads fit the raw team
    # score; this is the conservative default for legacy-equivalent scoring).
    assert cfg.score_target_transform == "raw"


def test_legacy_yaml_loads_with_new_blocks(tmp_path):
    cfg = load_config("configs/legacy.yaml")
    # All legacy defaults preserved.
    assert cfg.objective == "adp"
    assert cfg.feature_version == "legacy"
    assert cfg.ruleset == "legacy"
    assert cfg.model_version == "legacy"
    # New blocks present with zero-default weights.
    assert cfg.loss.lambda_win == 0.0
    assert cfg.decision_policy.mode == "pure_win"


def test_legacy_dmc_imports_avoid_training_v2_modules():
    """Importing douzero.dmc.dmc must NOT pull in the P06 training package.

    Uses a subprocess so the test's sys.modules manipulation does not
    corrupt the module identity for subsequent tests in the same session
    (deleting douzero.training.* from sys.modules would cause later tests
    to see a fresh, duplicate DecisionConfig class).
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import importlib\n"
        "# Ensure the training package is not loaded before the check.\n"
        "for m in list(sys.modules):\n"
        "    if m.startswith('douzero.training'):\n"
        "        del sys.modules[m]\n"
        "importlib.import_module('douzero.dmc.dmc')\n"
        "assert 'douzero.training' not in sys.modules, "
        "'douzero.dmc.dmc transitively imports douzero.training'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=__import__("os").path.dirname(__import__("os").path.dirname(__file__)),
    )
    assert result.returncode == 0, (
        f"subprocess check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_dmc_train_gate_still_rejects_v2():
    """The hard training gate on v2/standard must remain (P06 uses train_v2)."""
    import argparse

    from douzero.dmc.dmc import train

    # A v2-model-version flags namespace must be rejected.
    flags = argparse.Namespace(
        feature_version="legacy",
        ruleset="legacy",
        model_version="v2",
        actor_device_cpu=True,
        gpu_devices="",
        num_actor_devices=1,
        num_actors=1,
        training_device="0",
        xpid="x",
        save_interval=1,
        total_frames=1,
        batch_size=1,
        unroll_length=1,
        num_buffers=1,
        num_threads=1,
        learning_rate=1e-4,
        alpha=0.99,
        momentum=0.0,
        epsilon=1e-5,
        objective="adp",
        disable_checkpoint=True,
        savedir=str(__import__("tempfile").mkdtemp()),
        load_model=False,
        exp_epsilon=0.0,
        max_grad_norm=40.0,
        seed=0,
        deterministic=False,
        config="",
    )
    with pytest.raises(ValueError):
        train(flags)
