"""P06 r5 tests: score-output semantic identity, signed-log clamping,
model YAML wiring, and loss-boundary label validation.

Covers the four blockers from the r4 review:

- The public loss API silently treats target_win=NaN as a "loss sample"
  via ``NaN >= 0.5 → False`` in per-sample head selection.
- score_target_transform (raw vs signed_log) changes what the model's
  score heads MEAN, but was not recorded in the checkpoint identity.
- signed-log targets are not guaranteed to be representable by the head
  clamp.
- The ``model:`` YAML block documented by ModelV2Config was not wired
  through the config loader.
"""

from __future__ import annotations

import math

import pytest
import torch

from douzero.models_v2.config import ModelV2Config


# --------------------------------------------------------------------------- #
# Blocker 1: loss-boundary label validation
# --------------------------------------------------------------------------- #
def test_loss_rejects_target_win_nan_at_boundary():
    """The public loss API rejects target_win=NaN before any computation."""
    from douzero.training import LossConfig, MultiObjectiveLoss

    fn = MultiObjectiveLoss(LossConfig(lambda_win=0.0, lambda_score=1.0))
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        fn.forward_gathered(
            torch.tensor([[0.5]], requires_grad=True),
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[-1.0]], requires_grad=True),
            {"target_win": torch.tensor([float("nan")]),
             "target_score": torch.tensor([1.0])},
        )


def test_loss_rejects_target_win_non_binary_at_boundary():
    from douzero.training import LossConfig, MultiObjectiveLoss

    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    with pytest.raises(ValueError, match="target_win must be binary"):
        fn.forward_gathered(
            torch.tensor([[0.5]], requires_grad=True),
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[-1.0]], requires_grad=True),
            {"target_win": torch.tensor([0.7]),
             "target_score": torch.tensor([1.0])},
        )


def test_loss_rejects_target_score_inf_when_score_active():
    from douzero.training import LossConfig, MultiObjectiveLoss

    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=0.5))
    with pytest.raises(ValueError, match="target_score contains non-finite"):
        fn.forward_gathered(
            torch.tensor([[0.5]], requires_grad=True),
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[-1.0]], requires_grad=True),
            {"target_win": torch.tensor([1.0]),
             "target_score": torch.tensor([float("inf")])},
        )


def test_loss_rejects_mismatched_target_win_length():
    from douzero.training import LossConfig, MultiObjectiveLoss

    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0))
    with pytest.raises(ValueError, match="target_win length"):
        fn.forward_gathered(
            torch.tensor([[0.5], [0.6]], requires_grad=True),
            torch.tensor([[1.0], [2.0]], requires_grad=True),
            torch.tensor([[-1.0], [-2.0]], requires_grad=True),
            {"target_win": torch.tensor([1.0]),  # length 1, not 2
             "target_score": torch.tensor([1.0, 2.0])},
        )


# --------------------------------------------------------------------------- #
# Blocker 2: score_target_transform in checkpoint identity
# --------------------------------------------------------------------------- #
def test_different_score_transforms_produce_different_config_hashes():
    """A model trained with 'raw' and one with 'signed_log' have different
    model-config hashes, so the checkpoint loader rejects cross-semantics
    loading."""
    cfg_raw = ModelV2Config(score_target_transform="raw")
    cfg_log = ModelV2Config(score_target_transform="signed_log")
    assert cfg_raw.stable_hash() != cfg_log.stable_hash()


def test_model_config_rejects_bad_score_target_transform():
    with pytest.raises(ValueError, match="score_target_transform"):
        ModelV2Config(score_target_transform="bogus")


def test_trainer_rejects_score_transform_mismatch():
    """The trainer rejects a model/loss score_target_transform mismatch."""
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(build_v2_schema(), ModelV2Config(score_target_transform="raw"))
    loss_cfg = LossConfig(score_target_transform="signed_log", score_clamp=32.0)
    with pytest.raises(ValueError, match="score_target_transform.*does not match"):
        V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))


def test_trainer_accepts_matching_score_transform():
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(score_target_transform="signed_log", score_clamp=32.0),
    )
    loss_cfg = LossConfig(score_target_transform="signed_log", score_clamp=32.0)
    trainer = V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))
    assert trainer is not None


# --------------------------------------------------------------------------- #
# Blocker 3: signed-log target clamped to score_clamp
# --------------------------------------------------------------------------- #
def test_signed_log_target_clamped_to_score_clamp():
    """A signed-log target above the model's score_clamp is clamped so it
    stays inside the representable range."""
    from douzero.training import LossConfig, MultiObjectiveLoss

    fn = MultiObjectiveLoss(
        LossConfig(
            lambda_win=0.0,
            lambda_score=1.0,
            score_target_transform="signed_log",
            score_clamp=0.5,  # very small; log1p(1.0)≈0.693 exceeds this
        )
    )
    # target_score=1.0 → signed_log(1.0)=0.693 → clamped to 0.5
    # head value=0.5 → Huber(0.5, 0.5)=0 → near-perfect fit
    win_logit = torch.tensor([[0.0]], requires_grad=True)
    score_if_win = torch.tensor([[0.5]], requires_grad=True)
    score_if_loss = torch.tensor([[0.5]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([1.0]),
    }
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    # The clamped target (0.5) matches the head (0.5), so loss is ~0.
    assert comps.score < 0.01


# --------------------------------------------------------------------------- #
# Blocker 4: model: YAML block wiring
# --------------------------------------------------------------------------- #
def test_enhanced_yaml_includes_model_block():
    """configs/enhanced.yaml should carry a model: block and load cleanly."""
    from douzero.config import load_config

    cfg = load_config("configs/enhanced.yaml")
    assert cfg.model.version == "v2"
    assert cfg.model.hidden_size == 256
    assert cfg.model.history_encoder == "transformer"


def test_model_yaml_drives_model_construction(tmp_path):
    """A YAML config with model.hidden_size=128 drives a smaller model."""
    import importlib.util
    import os

    yaml_path = tmp_path / "custom_model.yaml"
    yaml_path.write_text(
        "feature_version: v2\nruleset: legacy\nmodel_version: v2\nobjective: adp\n"
        "seed: 42\n"
        "model:\n  version: v2\n  hidden_size: 128\n  history_heads: 4\n"
        "loss:\n  lambda_win: 1.0\n  lambda_score: 0.5\n"
        "  lambda_uncertainty: 0.0\n  score_delta: 1.0\n"
        "  score_target_transform: raw\n  score_clamp: 32.0\n"
        "decision_policy:\n  mode: pure_win\n  abs_tol: 0.0\n"
        "  rel_tol: 0.0\n  risk_penalty: 0.0\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "train_v2_model_check",
        os.path.join(os.path.dirname(__file__), "..", "train_v2.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from douzero.config import load_config

    cfg = load_config(str(yaml_path))
    model_cfg = module._build_model_cfg(cfg)
    assert model_cfg.hidden_size == 128
    assert model_cfg.history_heads == 4
    # score_clamp and score_target_transform come from the loss block.
    assert model_cfg.score_clamp == 32.0
    assert model_cfg.score_target_transform == "raw"


def test_enhanced_yaml_cli_runs_with_model_block():
    """train_v2.py --config configs/enhanced.yaml runs end-to-end with the
    model: block driving model construction."""
    import os
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "train_v2.py"),
            "--config", os.path.join(repo_root, "configs", "enhanced.yaml"),
            "--episodes", "0",
            "--optimizer_steps", "0",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    # The model should have the YAML-configured hidden_size (256).
    assert "params=" in result.stdout
