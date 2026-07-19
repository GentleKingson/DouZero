"""P06 r3 regression tests: seed semantics, inference mode, target NaN,
RMSprop ranges, enhanced YAML detection, and replay buffer label integrity.

These cover the remaining blockers found in the r2 review:

- ``seed=0`` must be a true no-op (Python/NumPy/Torch/local-RNG all
  unseeded), not a mixed state where Torch is seeded but the deal shuffle
  is not.
- The V2 trainer's model must be in ``eval()`` mode during self-play so
  ``history_dropout`` / ``mlp_dropout`` do not silently randomize action
  selection when ``exp_epsilon=0``.
- Calibration must reject ``target_win=NaN`` (and Inf), not just
  ``p_win=NaN``.
- ``TrainerConfig`` validates RMSprop parameter ranges.
- ``configs/enhanced.yaml`` legacy multiprocess fields are visibly warned
  when set to non-default values (not silently ignored).
"""

from __future__ import annotations

import io
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
# Blocker 1: seed=0 is a true no-op (no mixed reproducibility)
# --------------------------------------------------------------------------- #
def test_trainer_rng_seed_0_uses_system_entropy():
    """When rng_seed=0, the trainer's local RNG should NOT be seeded to 0;
    it should use system entropy so two trainers with seed=0 behave
    differently (matching the unseeded deal shuffle and model init).

    P06 r4: the r3 version of this test had ``or True`` which made it a
    tautology. This version checks the construction path directly via the
    RNG's internal state: a random.Random(0) has a deterministic internal
    state, while random.Random() (system entropy) does not."""
    import random as _random

    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    torch.manual_seed(999)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    trainer = V2Trainer(model, config=TrainerConfig(rng_seed=0, max_episodes=0))

    # random.Random(0) has a specific internal state tuple; system entropy
    # does not. Compare against a fresh Random(0).
    seeded_state = _random.Random(0).getstate()
    trainer_state = trainer.rng.getstate()
    # The state tuples differ in the third element (the shuffled internal
    # array) when the RNG was initialized differently. The first two
    # elements (version, gusano) may coincidentally match, so compare the
    # full tuple.
    assert trainer_state != seeded_state, (
        "trainer RNG with rng_seed=0 has the same state as random.Random(0); "
        "it should use system entropy (random.Random() with no argument)."
    )


def test_trainer_rng_seed_nonzero_is_reproducible():
    """When rng_seed=42, two trainers produce the same action-sampling
    sequence."""
    import random as _random

    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    torch.manual_seed(999)
    m1 = ModelV2(build_v2_schema(), ModelV2Config())
    t1 = V2Trainer(m1, config=TrainerConfig(rng_seed=42, max_episodes=0))
    torch.manual_seed(999)
    m2 = ModelV2(build_v2_schema(), ModelV2Config())
    t2 = V2Trainer(m2, config=TrainerConfig(rng_seed=42, max_episodes=0))
    vals1 = [t1.rng.random() for _ in range(5)]
    vals2 = [t2.rng.random() for _ in range(5)]
    assert vals1 == vals2


# --------------------------------------------------------------------------- #
# Blocker 2: model.eval() during self-play (dropout determinism)
# --------------------------------------------------------------------------- #
def test_trainer_puts_model_in_eval_mode():
    """V2Trainer.__init__ must call model.eval() so self-play collection runs
    in evaluation mode (dropout off)."""
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import TrainerConfig, V2Trainer

    torch.manual_seed(42)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    # A freshly-constructed nn.Module is in training mode by default.
    assert model.training
    trainer = V2Trainer(model, config=TrainerConfig(max_episodes=0))
    # The trainer must have switched it to eval mode.
    assert not model.training


def test_eval_mode_makes_dropout_deterministic_with_zero_epsilon():
    """A model with history_dropout > 0 must produce identical action
    selections when exp_epsilon=0, because model.eval() disables dropout."""
    from douzero.env.env import Env
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.encode_v2 import get_obs_v2
    from douzero.observation.schema import build_v2_schema
    from douzero.training import DecisionConfig, TrainerConfig, V2Trainer

    torch.manual_seed(42)
    # Build a model with non-zero dropout.
    model = ModelV2(
        build_v2_schema(), ModelV2Config(history_dropout=0.3, mlp_dropout=0.3)
    )
    trainer = V2Trainer(
        model,
        decision_config=DecisionConfig(mode="pure_win"),
        config=TrainerConfig(
            exp_epsilon=0.0,  # no exploration; selection must be deterministic
            max_episodes=0,
        ),
    )
    # Drive the env to a position with >= 2 legal actions.
    import numpy as np

    np.random.seed(123)
    env = Env("adp")
    env.reset()
    steps = 0
    while env._acting_player_position != "landlord" or len(env.infoset.legal_actions) < 2:
        if steps > 40:
            pytest.skip("could not find a multi-action position")
        env.step(env.infoset.legal_actions[0])
        steps += 1
    obs = get_obs_v2(env.infoset)
    # Select twice; with model.eval() the results must be identical.
    idx1 = trainer._choose_action_index(obs)
    idx2 = trainer._choose_action_index(obs)
    assert idx1 == idx2, (
        f"dropout-active self-play produced different actions ({idx1} vs {idx2}) "
        f"with exp_epsilon=0; model.eval() was not called."
    )


# --------------------------------------------------------------------------- #
# Blocker 3: target_win=NaN is rejected
# --------------------------------------------------------------------------- #
def test_brier_rejects_target_nan():
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        brier_score(torch.tensor([0.5]), torch.tensor([float("nan")]))


def test_brier_rejects_target_inf():
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        brier_score(torch.tensor([0.5]), torch.tensor([float("inf")]))


def test_nll_rejects_target_nan():
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        nll(torch.tensor([0.5]), torch.tensor([float("nan")]))


def test_ece_rejects_target_nan():
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        expected_calibration_error(
            torch.tensor([0.3, 0.7]), torch.tensor([1.0, float("nan")])
        )


def test_reliability_bins_rejects_target_nan():
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        reliability_bins(
            torch.tensor([0.3, 0.7]), torch.tensor([float("nan"), 0.0])
        )


# --------------------------------------------------------------------------- #
# Non-blocking: RMSprop range validation
# --------------------------------------------------------------------------- #
def test_trainer_config_rejects_rmsprop_alpha_out_of_range():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="rmsprop_alpha"):
        TrainerConfig(rmsprop_alpha=1.5)
    with pytest.raises(ValueError, match="rmsprop_alpha"):
        TrainerConfig(rmsprop_alpha=-0.1)


def test_trainer_config_rejects_rmsprop_momentum_negative():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="rmsprop_momentum"):
        TrainerConfig(rmsprop_momentum=-0.5)


def test_trainer_config_rejects_rmsprop_epsilon_nonpositive():
    from douzero.training import TrainerConfig

    with pytest.raises(ValueError, match="rmsprop_epsilon"):
        TrainerConfig(rmsprop_epsilon=0.0)
    with pytest.raises(ValueError, match="rmsprop_epsilon"):
        TrainerConfig(rmsprop_epsilon=-1e-6)


def test_trainer_config_accepts_valid_rmsprop():
    from douzero.training import TrainerConfig

    cfg = TrainerConfig(
        rmsprop_alpha=0.95, rmsprop_momentum=0.9, rmsprop_epsilon=1e-8, max_episodes=0
    )
    assert cfg.rmsprop_alpha == 0.95


# --------------------------------------------------------------------------- #
# Blocker 4: enhanced YAML warns about ignored legacy fields
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_warns_on_nondefault_legacy_fields(tmp_path):
    """A YAML config that sets an unconsumed legacy multiprocess field
    to a non-default value should produce a visible warning."""
    import importlib.util

    yaml_path = tmp_path / "warn.yaml"
    yaml_path.write_text(
        "feature_version: v2\nruleset: legacy\nmodel_version: v2\nobjective: adp\n"
        "gpu_devices: '3'\n"
        "loss:\n  lambda_win: 1.0\n  lambda_score: 0.5\n"
        "  lambda_uncertainty: 0.0\n  score_delta: 1.0\n"
        "  score_target_transform: raw\n  score_clamp: 32.0\n"
        "decision_policy:\n  mode: pure_win\n  abs_tol: 0.0\n"
        "  rel_tol: 0.0\n  risk_penalty: 0.0\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "train_v2_warn_check",
        os.path.join(_REPO_ROOT, "train_v2.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from douzero.config import load_config

    cfg = load_config(str(yaml_path))
    # Capture stderr.
    old_stderr = sys.stderr
    sys.stderr = captured = io.StringIO()
    try:
        module._warn_unsupported_legacy_fields(cfg)
    finally:
        sys.stderr = old_stderr
    output = captured.getvalue()
    assert "gpu_devices" in output
    assert "WARNING" in output


def test_cli_no_warning_on_default_legacy_fields():
    """The default enhanced.yaml should NOT warn (all legacy fields at defaults)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_v2_nowarn_check",
        os.path.join(_REPO_ROOT, "train_v2.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from douzero.config import load_config

    cfg = load_config(os.path.join(_REPO_ROOT, "configs", "enhanced.yaml"))
    old_stderr = sys.stderr
    sys.stderr = captured = io.StringIO()
    try:
        module._warn_unsupported_legacy_fields(cfg)
    finally:
        sys.stderr = old_stderr
    # The enhanced.yaml sets some fields to values that differ from the
    # TrainingConfig defaults (e.g. savedir: douzero_v2_checkpoints vs
    # douzero_checkpoints). Those specific fields are NOT in our check list.
    # The fields we DO check (num_actors, gpu_devices, etc.) are at defaults.
    output = captured.getvalue()
    assert "num_actors" not in output
    assert "gpu_devices" not in output


# --------------------------------------------------------------------------- #
# Non-blocking: replay buffer label completeness
# --------------------------------------------------------------------------- #
def test_transition_has_labels_checks_all_three_fields():
    """has_labels() must return False if ANY of target_win, target_score, or
    target_log_score is NaN (P06 r3: previously only checked target_win)."""
    from douzero.training.v2_buffer import Transition

    # All present → True
    t = Transition(obs=None, action_index=0, position="landlord", target_win=1.0, target_score=2.0, target_log_score=0.7)  # type: ignore[arg-type]
    assert t.has_labels()
    # target_score NaN → False
    t = Transition(obs=None, action_index=0, position="landlord", target_win=1.0, target_score=float("nan"), target_log_score=0.7)  # type: ignore[arg-type]
    assert not t.has_labels()
    # target_log_score NaN → False
    t = Transition(obs=None, action_index=0, position="landlord", target_win=1.0, target_score=2.0, target_log_score=float("nan"))  # type: ignore[arg-type]
    assert not t.has_labels()
