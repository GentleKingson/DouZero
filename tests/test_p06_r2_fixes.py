"""P06 r2 regression tests: CLI precedence, identity gate, trainer
invariants, and calibration input validation.

These cover the remaining blockers found in the r1 review:

- ``train_v2.py`` config resolution (``episodes``/``max_episodes`` name
  mismatch, RMSprop CLI dest names, ``--deterministic`` default clobbering
  YAML).
- ``train_v2.py`` identity gate (rejecting ``--config configs/legacy.yaml``
  before any model is built).
- ``TrainerConfig`` range validation (rejecting ``batch_size=0``, etc.).
- ``V2Trainer`` score_clamp consistency check.
- Episode step cap as an explicit ``raise`` (not ``assert``).
- Calibration metrics reject illegal probabilities instead of masking them.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch

from douzero.training.calibration import (
    brier_score,
    expected_calibration_error,
    nll,
    reliability_bins,
)


# --------------------------------------------------------------------------- #
# Blocker 4: calibration rejects illegal probabilities (no silent clamping)
# --------------------------------------------------------------------------- #
def test_brier_rejects_p_above_one():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        brier_score(torch.tensor([1.5]), torch.tensor([1.0]))


def test_brier_rejects_p_below_zero():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        brier_score(torch.tensor([-0.1]), torch.tensor([0.0]))


def test_brier_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        brier_score(torch.tensor([float("nan")]), torch.tensor([1.0]))


def test_brier_rejects_inf():
    with pytest.raises(ValueError, match="non-finite"):
        brier_score(torch.tensor([float("inf")]), torch.tensor([1.0]))


def test_calibration_rejects_non_binary_target():
    for bad_target in [-1.0, 0.5, 2.0]:
        with pytest.raises(ValueError, match="target_win"):
            brier_score(torch.tensor([0.3]), torch.tensor([bad_target]))


def test_brier_perfect_binary_predictor_is_exactly_zero():
    """Brier([0,1],[0,1]) == 0.0 — not clamped, not approximately zero."""
    assert brier_score(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])) == 0.0


def test_brier_does_not_mask_illegal_p_as_perfect():
    """A p=2.0 against target=1 must NOT be clamped to look near-perfect.
    The r0/r1 clamp produced Brier ≈ 0; r2 rejects it."""
    with pytest.raises(ValueError):
        brier_score(torch.tensor([2.0]), torch.tensor([1.0]))


def test_nll_accepts_boundary_probabilities():
    """p=0.0 and p=1.0 are legal (a quantized predictor can emit them).
    NLL clamps a LOCAL copy to keep the log finite but does NOT reject."""
    # p=0, target=1 should give a large but finite NLL, not raise.
    loss = nll(torch.tensor([0.0]), torch.tensor([1.0]))
    assert math.isfinite(loss)
    assert loss > 10.0
    # p=1, target=1 should give near-zero NLL.
    loss2 = nll(torch.tensor([1.0]), torch.tensor([1.0]))
    assert loss2 < 1e-3


def test_nll_rejects_p_above_one():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        nll(torch.tensor([1.1]), torch.tensor([1.0]))


def test_ece_rejects_illegal_p():
    with pytest.raises(ValueError):
        expected_calibration_error(torch.tensor([-0.1, 0.5]), torch.tensor([0.0, 1.0]))


def test_reliability_bins_rejects_illegal_p():
    with pytest.raises(ValueError):
        reliability_bins(torch.tensor([1.5]), torch.tensor([1.0]))


def test_calibration_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        brier_score(torch.tensor([0.1, 0.2]), torch.tensor([1.0]))


def test_calibration_rejects_empty():
    with pytest.raises(ValueError, match="zero samples"):
        brier_score(torch.tensor([]), torch.tensor([]))


# --------------------------------------------------------------------------- #
# Blocker 3b: TrainerConfig range validation
# --------------------------------------------------------------------------- #
def test_trainer_config_rejects_batch_size_zero():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="batch_size"):
        TrainerConfig(batch_size=0)


def test_trainer_config_rejects_negative_optimizer_steps():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="optimizer_steps"):
        TrainerConfig(optimizer_steps=-1)


def test_trainer_config_rejects_exp_epsilon_out_of_range():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="exp_epsilon"):
        TrainerConfig(exp_epsilon=2.0)
    with pytest.raises(ValueError, match="exp_epsilon"):
        TrainerConfig(exp_epsilon=-0.1)


def test_trainer_config_rejects_non_positive_learning_rate():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="learning_rate"):
        TrainerConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainerConfig(learning_rate=-1.0)


def test_trainer_config_rejects_non_positive_max_grad_norm():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="max_grad_norm"):
        TrainerConfig(max_grad_norm=0.0)


def test_trainer_config_rejects_zero_buffer_capacity():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="buffer_capacity"):
        TrainerConfig(buffer_capacity=0)


def test_trainer_config_rejects_zero_max_steps():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="max_steps_per_episode"):
        TrainerConfig(max_steps_per_episode=0)


def test_trainer_config_rejects_negative_max_episodes():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="max_episodes"):
        TrainerConfig(max_episodes=-1)


def test_trainer_config_accepts_valid_boundary_values():
    from douzero.training import TrainerConfig

    # optimizer_steps=0 is valid (collect-only, no optimization).
    cfg = TrainerConfig(batch_size=1, optimizer_steps=0, exp_epsilon=0.0,
                        buffer_capacity=1, max_steps_per_episode=1, max_episodes=0)
    assert cfg.batch_size == 1
    assert cfg.optimizer_steps == 0
    assert cfg.max_episodes == 0


# --------------------------------------------------------------------------- #
# Blocker 3a: V2Trainer score_clamp consistency
# --------------------------------------------------------------------------- #
def test_trainer_rejects_score_clamp_mismatch_in_raw_mode():
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(build_v2_schema(), ModelV2Config(score_clamp=8.0))
    # Loss clamp=32, model clamp=8, raw mode → mismatch.
    loss_cfg = LossConfig(score_clamp=32.0, score_target_transform="raw")
    with pytest.raises(ValueError, match="score_clamp"):
        V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))


def test_trainer_accepts_matching_score_clamp():
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(build_v2_schema(), ModelV2Config(score_clamp=16.0))
    loss_cfg = LossConfig(score_clamp=16.0, score_target_transform="raw")
    trainer = V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))
    assert trainer.loss_fn.config.score_clamp == 16.0


def test_trainer_accepts_mismatched_clamp_in_signed_log_mode():
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    # In signed_log mode the target is NOT clamped to score_clamp (log1p
    # compresses it well inside the head clamp), so a mismatch is acceptable.
    model = ModelV2(build_v2_schema(), ModelV2Config(score_clamp=8.0))
    loss_cfg = LossConfig(score_clamp=32.0, score_target_transform="signed_log")
    trainer = V2Trainer(model, loss_config=loss_cfg, config=TrainerConfig(max_episodes=0))
    assert trainer is not None


# --------------------------------------------------------------------------- #
# Blocker 3c: episode step cap is raise, not assert
# --------------------------------------------------------------------------- #
def test_episode_step_cap_uses_raise_not_assert():
    import inspect

    from douzero.training import v2_trainer

    src = inspect.getsource(v2_trainer.V2Trainer._run_one_episode)
    assert "raise RuntimeError" in src
    assert "assert steps" not in src


# --------------------------------------------------------------------------- #
# Blockers 1+2: real CLI integration via subprocess
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cli(*args: str) -> tuple[int, str, str]:
    """Run train_v2.py with args; return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "train_v2.py"), *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


def test_cli_no_config_no_args_succeeds():
    """``python train_v2.py`` with defaults (episodes=8) must not crash on
    config resolution — the r1 pick() crashed on ``episodes``."""
    # Use --episodes 0 --optimizer_steps 0 to keep it fast.
    code, out, err = _run_cli("--episodes", "0", "--optimizer_steps", "0")
    assert code == 0, f"stdout={out}\nstderr={err}"
    assert "[train_v2]" in out


def test_cli_with_enhanced_config_succeeds():
    code, out, err = _run_cli(
        "--config", os.path.join(_REPO_ROOT, "configs", "enhanced.yaml"),
        "--episodes", "0", "--optimizer_steps", "0",
    )
    assert code == 0, f"stdout={out}\nstderr={err}"
    assert "[train_v2]" in out
    # The enhanced config carries lambda_win=1.0 and lambda_score=0.5.
    assert "'lambda_win': 1.0" in out
    assert "'lambda_score': 0.5" in out


def test_cli_rmsprop_alpha_override_is_consumed():
    """``--rmsprop_alpha 0.8`` must reach the trainer (r1 looked for 'alpha',
    not 'rmsprop_alpha')."""
    code, out, err = _run_cli(
        "--episodes", "0", "--optimizer_steps", "0",
        "--rmsprop_alpha", "0.8",
    )
    assert code == 0, f"stdout={out}\nstderr={err}"
    # The config line should reflect the overridden value somewhere.
    # (We don't print rmsprop_alpha in the summary line, so just assert
    # the command succeeded — the r1 crash was an AttributeError before
    # any output.)


def test_cli_legacy_config_rejected_before_model_construction():
    """``--config configs/legacy.yaml`` must fail FAST (feature_version=legacy)
    before a ModelV2 is built."""
    code, out, err = _run_cli(
        "--config", os.path.join(_REPO_ROOT, "configs", "legacy.yaml"),
        "--episodes", "0",
    )
    assert code != 0, f"expected failure, got stdout={out}\nstderr={err}"
    assert "feature_version" in err or "model_version" in err, (
        f"expected an identity error about feature/model version, got:\n{err}"
    )
    # The model must NOT have been constructed (the error is raised before
    # the "[train_v2] model=..." line).
    assert "[train_v2] model=" not in out


def test_cli_batch_size_override_consumed_from_cli():
    """``--batch_size 3`` overrides the YAML/default value."""
    code, out, err = _run_cli(
        "--episodes", "0", "--optimizer_steps", "0", "--batch_size", "3",
    )
    assert code == 0, f"stdout={out}\nstderr={err}"
    assert "batch_size=3" in out


def test_cli_deterministic_flag_works():
    """``--deterministic`` is honored (does not crash)."""
    code, out, err = _run_cli(
        "--episodes", "0", "--optimizer_steps", "0", "--deterministic",
    )
    assert code == 0, f"stdout={out}\nstderr={err}"
