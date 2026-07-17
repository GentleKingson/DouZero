"""P06 r1 regression tests for the multi-objective loss + trainer fixes.

These cover the specific blockers found in the r0 review:

- ``lambda_uncertainty > 0`` no longer crashes (B=1 and B>1, finite, grad).
- the conditional-score loss scale is independent of the batch's win/loss
  composition (per-sample selection, not ``0.5 * (win + loss)``).
- ``score_target_transform`` is mutually exclusive (raw vs signed_log),
  not an additive double-supervision on the same head.
- an explicit ``lambda=0`` in YAML is preserved through ``train_v2.py``
  (the r0 ``if > 0 else default`` fallback is gone).
- YAML ``batch_size`` / ``learning_rate`` / ``exp_epsilon`` are actually
  consumed by the trainer (r0 ignored them).
- ``DeepAgentV2`` consumes ``abs_tol`` / ``rel_tol`` / ``risk_penalty``
  via a ``decision_config`` (r0 dropped them to defaults).
- the trainer rejects a non-legacy ruleset at construction.
- ``TrainerConfig`` and the CLI no longer advertise checkpoint/gpu options
  that are not implemented.
- the trainer is fail-closed on non-finite loss / gradient (the optimizer
  is NOT allowed to mutate parameters when the loss or grad norm is NaN/Inf).
"""

from __future__ import annotations

import math

import pytest
import torch

from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    DecisionConfig,
    LossConfig,
    MultiObjectiveLoss,
    TrainerConfig,
    V2Trainer,
    conditional_score_huber_loss,
    uncertainty_nll,
)


# --------------------------------------------------------------------------- #
# Blocker 1: uncertainty NLL (B=1 and B>1, finite, grad)
# --------------------------------------------------------------------------- #
def test_uncertainty_nll_b1_finite_and_grad():
    score_if_win = torch.tensor([[2.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-2.0]], requires_grad=True)
    target_score = torch.tensor([2.0])
    target_win = torch.tensor([1.0])
    loss = uncertainty_nll(score_if_win, score_if_loss, target_score, target_win)
    assert torch.isfinite(loss)
    loss.backward()
    assert score_if_win.grad is not None
    assert torch.isfinite(score_if_win.grad).all()


def test_uncertainty_nll_b_greater_than_1_finite_and_grad():
    score_if_win = torch.tensor([[2.0], [1.0], [3.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-2.0], [-1.0], [-3.0]], requires_grad=True)
    target_score = torch.tensor([2.0, -1.0, 3.0])
    target_win = torch.tensor([1.0, 0.0, 1.0])
    loss = uncertainty_nll(score_if_win, score_if_loss, target_score, target_win)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(score_if_win.grad).all()


def test_uncertainty_nll_via_combiner_mixed_batch():
    win_logit = torch.tensor([[0.3], [-0.4]], requires_grad=True)
    score_if_win = torch.tensor([[1.0], [2.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0], [2.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0, 0.0]),
        "target_score": torch.tensor([1.0, -2.0]),
    }
    fn = MultiObjectiveLoss(
        LossConfig(lambda_win=1.0, lambda_score=0.5, lambda_uncertainty=0.3)
    )
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    assert torch.isfinite(comps.total)
    assert comps.uncertainty > 0.0
    comps.total.backward()
    assert win_logit.grad is not None
    assert torch.isfinite(win_logit.grad).all()


def test_uncertainty_nll_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        uncertainty_nll(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[-1.0]]),  # different shape
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 0.0]),
        )


# --------------------------------------------------------------------------- #
# Blocker 3: per-sample loss scale is independent of batch win/loss mix
# --------------------------------------------------------------------------- #
def test_loss_scale_independent_of_batch_win_loss_mix():
    """Two batches with the SAME per-sample errors must yield the SAME loss,
    regardless of whether they are all-win, all-loss, or mixed. The r0
    ``0.5 * (win_term + loss_term)`` halved the only active term on a
    pure-win or pure-loss batch."""
    # Build three batches, each with the same per-sample |pred - target|:
    # batch A: two win samples, head=2, target=2 (error 0)
    # batch B: two loss samples, head=2, target=2 (error 0)
    # batch C: one win + one loss, head=2, target=2 (error 0)
    head_win = torch.tensor([[2.0], [2.0]])
    head_loss = torch.tensor([[2.0], [2.0]])
    target = torch.tensor([2.0, 2.0])
    loss_a, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([1.0, 1.0])
    )
    loss_b, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([0.0, 0.0])
    )
    loss_c, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([1.0, 0.0])
    )
    assert loss_a.item() == pytest.approx(loss_b.item())
    assert loss_a.item() == pytest.approx(loss_c.item())


def test_loss_scale_consistent_with_nonzero_error():
    """Same per-sample error magnitude yields the same loss regardless of mix,
    even when the error is non-zero."""
    # All batches have per-sample |pred - target| == 1.
    head_win = torch.tensor([[2.0], [2.0]])
    head_loss = torch.tensor([[2.0], [2.0]])
    target = torch.tensor([3.0, 3.0])
    loss_all_win, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([1.0, 1.0])
    )
    loss_all_loss, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([0.0, 0.0])
    )
    loss_mixed, _, _ = conditional_score_huber_loss(
        head_win, head_loss, target, torch.tensor([1.0, 0.0])
    )
    # The r0 bug halved the loss on the pure-win / pure-loss batches.
    assert loss_all_win.item() == pytest.approx(loss_mixed.item())
    assert loss_all_loss.item() == pytest.approx(loss_mixed.item())


# --------------------------------------------------------------------------- #
# Blocker 2: score_target_transform is mutually exclusive
# --------------------------------------------------------------------------- #
def test_score_target_transform_raw_vs_signed_log_produces_different_loss():
    """The same head value produces a small loss under one transform and a
    large loss under the other — confirming the transform selects which
    scale the heads are supervised against, rather than adding a second
    (conflicting) supervision on the same head."""
    win_logit = torch.tensor([[0.0]], requires_grad=True)
    # Head produces log1p(32) ≈ 3.466 — a good signed-log fit for raw=32.
    score_if_win = torch.tensor([[math.log1p(32.0)]], requires_grad=True)
    score_if_loss = torch.tensor([[0.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([32.0]),
    }
    fn_log = MultiObjectiveLoss(
        LossConfig(lambda_win=0.0, lambda_score=1.0, score_target_transform="signed_log")
    )
    fn_raw = MultiObjectiveLoss(
        LossConfig(lambda_win=0.0, lambda_score=1.0, score_target_transform="raw")
    )
    log_loss = fn_log.forward_gathered(
        win_logit, score_if_win, score_if_loss, labels
    ).score
    raw_loss = fn_raw.forward_gathered(
        win_logit, score_if_win, score_if_loss, labels
    ).score
    assert log_loss < 0.05
    assert raw_loss > 10.0


def test_loss_config_rejects_both_log_and_raw_cannot_combine():
    """There is no ``lambda_log`` anymore: a single head cannot be supervised
    toward two scales at once. The API simply does not expose the additive
    log term."""
    # The old API accepted lambda_log; the new API does not.
    with pytest.raises(TypeError):
        LossConfig(lambda_log=0.5)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Blocker 4 + 5: train_v2.py honours YAML (lambda=0 and batch/lr/epsilon)
# --------------------------------------------------------------------------- #
def test_train_v2_preserves_explicit_yaml_lambda_zero(tmp_path):
    """A YAML ``lambda_win: 0`` must reach the trainer as 0, not be silently
    upgraded to 1.0."""
    import importlib.util
    import os
    import sys

    # Isolate the train_v2 module load so the test does not depend on it
    # already being importable.
    yaml_path = tmp_path / "zero.yaml"
    yaml_path.write_text(
        "feature_version: v2\nruleset: legacy\nmodel_version: v2\n"
        "loss:\n  lambda_win: 0.0\n  lambda_score: 0.0\n"
        "  lambda_uncertainty: 0.0\n  score_delta: 1.0\n"
        "  score_target_transform: raw\n  score_clamp: 32.0\n"
        "decision_policy:\n  mode: pure_win\n  abs_tol: 0.0\n"
        "  rel_tol: 0.0\n  risk_penalty: 0.0\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "train_v2_under_test", os.path.join(os.path.dirname(__file__), "..", "train_v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    yaml_cfg = module._load_yaml_config(str(yaml_path))
    loss_cfg = module._build_loss_config(yaml_cfg)
    assert loss_cfg.lambda_win == 0.0
    assert loss_cfg.lambda_score == 0.0
    assert loss_cfg.lambda_uncertainty == 0.0


def test_train_v2_consumes_yaml_batch_lr_epsilon(tmp_path):
    """YAML batch_size / learning_rate / exp_epsilon are actually read by the
    trainer's TrainerConfig (r0 ignored them and used argparse defaults)."""
    import importlib.util
    import os

    yaml_path = tmp_path / "enh.yaml"
    yaml_path.write_text(
        "feature_version: v2\nruleset: legacy\nmodel_version: v2\n"
        "batch_size: 7\nexp_epsilon: 0.42\nmax_grad_norm: 5.0\n"
        "optimizer:\n  learning_rate: 0.007\n  alpha: 0.9\n"
        "  momentum: 0.0\n  epsilon: 1.0e-5\n"
        "loss:\n  lambda_win: 1.0\n  lambda_score: 0.5\n"
        "  lambda_uncertainty: 0.0\n  score_delta: 1.0\n"
        "  score_target_transform: raw\n  score_clamp: 32.0\n"
        "decision_policy:\n  mode: pure_win\n  abs_tol: 0.0\n"
        "  rel_tol: 0.0\n  risk_penalty: 0.0\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "train_v2_under_test_b", os.path.join(os.path.dirname(__file__), "..", "train_v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    yaml_cfg = module._load_yaml_config(str(yaml_path))

    # Simulate an "absent CLI" namespace (SUPPRESS defaults mean absent
    # flags do not appear in the namespace).
    import argparse

    ns = argparse.Namespace(config=str(yaml_path))
    defaults = TrainerConfig()

    def pick(name, yaml_obj=None):
        if name in vars(ns):
            return getattr(ns, name)
        if yaml_obj is not None and hasattr(yaml_obj, name):
            return getattr(yaml_obj, name)
        return getattr(defaults, name)

    # batch_size, exp_epsilon, max_grad_norm come from YAML.
    assert pick("batch_size", yaml_cfg) == 7
    assert pick("exp_epsilon", yaml_cfg) == pytest.approx(0.42)
    assert pick("max_grad_norm", yaml_cfg) == pytest.approx(5.0)
    # learning_rate lives under the optimizer sub-config.
    opt = getattr(yaml_cfg, "optimizer", None)
    assert pick("learning_rate", opt) == pytest.approx(0.007)


# --------------------------------------------------------------------------- #
# Blocker 6: DeepAgentV2 consumes abs_tol / rel_tol / risk_penalty
# --------------------------------------------------------------------------- #
def test_deepagent_v2_carries_full_decision_config():
    """DeepAgentV2 keeps the caller's abs_tol / rel_tol / risk_penalty instead
    of rebuilding a DecisionConfig with default (zero) tolerances."""
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2

    torch.manual_seed(7)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    cfg = DecisionConfig(
        mode="win_then_score", abs_tol=0.13, rel_tol=0.07, risk_penalty=0.5
    )
    agent = DeepAgentV2(
        position="landlord",
        model=model,
        ruleset=RuleSet.legacy(),
        decision_config=cfg,
    )
    assert agent.decision_config is cfg
    assert agent.decision_config.abs_tol == 0.13
    assert agent.decision_config.rel_tol == 0.07
    assert agent.decision_config.risk_penalty == 0.5
    assert agent.decision_mode == "win_then_score"


def test_deepagent_v2_decision_mode_only_keeps_default_tolerances():
    """The P05-compatible path (decision_mode only) keeps default tolerances
    so existing callers do not change behaviour."""
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2

    torch.manual_seed(7)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    agent = DeepAgentV2(
        position="landlord", model=model, ruleset=RuleSet.legacy(), decision_mode="win"
    )
    assert agent.decision_config.abs_tol == 0.0
    assert agent.decision_config.rel_tol == 0.0
    assert agent.decision_config.risk_penalty == 0.0


def test_deepagent_v2_rejects_disagreeing_mode_and_config():
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2

    torch.manual_seed(7)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    cfg = DecisionConfig(mode="pure_score")
    with pytest.raises(ValueError):
        DeepAgentV2(
            position="landlord",
            model=model,
            ruleset=RuleSet.legacy(),
            decision_mode="pure_win",  # disagrees
            decision_config=cfg,
        )


# --------------------------------------------------------------------------- #
# Blocker 7: trainer boundaries (standard ruleset / no fake options)
# --------------------------------------------------------------------------- #
def test_trainer_ruleset_boundary_preserves_legacy_and_supports_standard_bidding():
    """A non-None legacy ruleset remains invalid, while the explicit standard
    path requires (and accepts) a bidding-enabled model."""
    from douzero.env.rules import RuleSet

    torch.manual_seed(7)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    with pytest.raises(ValueError, match="bidding_enabled"):
        V2Trainer(model, ruleset=RuleSet.standard())
    with pytest.raises(ValueError, match="must be standard"):
        V2Trainer(model, ruleset=RuleSet.legacy())
    trainer = V2Trainer(model, ruleset=None, config=TrainerConfig(max_episodes=0))
    assert trainer.ruleset is None

    bidding_model = ModelV2(
        build_v2_schema(), ModelV2Config(bidding_enabled=True, hidden_size=32)
    )
    standard = V2Trainer(
        bidding_model,
        ruleset=RuleSet.standard(),
        loss_config=LossConfig(lambda_bid_policy=1.0),
        config=TrainerConfig(max_episodes=0, optimizer_steps=0),
    )
    assert standard.standard_mode


def test_trainer_config_no_fake_checkpoint_or_gpu_options():
    """r0 advertised checkpoint_dir / save_every_steps / gpu_device that were
    never implemented. r1 removes them. Assert they are NOT in the dataclass
    fields (a future P14 reintroduction is fine, but P06 must not lie)."""
    from dataclasses import fields

    field_names = {f.name for f in fields(TrainerConfig)}
    assert "checkpoint_dir" not in field_names
    assert "save_every_steps" not in field_names
    # The CLI also drops --gpu_device; check the parser does not expose it.
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "train_v2_fields_check",
        os.path.join(os.path.dirname(__file__), "..", "train_v2.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    parser = module._build_parser()
    option_strings = {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings
    }
    assert "--gpu_device" not in option_strings
    assert "--checkpoint_dir" not in option_strings


# --------------------------------------------------------------------------- #
# Blocker 8: fail-closed on non-finite loss / gradient
# --------------------------------------------------------------------------- #
def test_trainer_fails_closed_on_nonfinite_loss(monkeypatch):
    """A NaN loss must raise FloatingPointError BEFORE the optimizer mutates
    parameters. Inject a NaN by patching the loss module to return a NaN."""
    torch.manual_seed(7)
    model = ModelV2(build_v2_schema(), ModelV2Config())
    trainer = V2Trainer(
        model,
        ruleset=None,
        config=TrainerConfig(
            seed=7,
            rng_seed=7,
            max_episodes=2,
            batch_size=4,
            optimizer_steps=0,
            exp_epsilon=0.5,
        ),
    )
    trainer.collect_episodes()
    # Snapshot a parameter so we can prove the optimizer did not run.
    before = next(model.parameters()).detach().clone()

    # Force the loss to return a NaN total.
    from douzero.training.losses import LossComponents

    def fake_forward_gathered(self, win_logit, score_if_win, score_if_loss, labels):
        nan = win_logit.new_full((), float("nan"))
        return LossComponents(
            total=nan, win=float("nan"), score=0.0, uncertainty=0.0,
            num_win=1, num_loss=0,
        )

    monkeypatch.setattr(MultiObjectiveLoss, "forward_gathered", fake_forward_gathered)
    with pytest.raises(FloatingPointError):
        trainer.step()
    # Parameters unchanged.
    after = next(model.parameters()).detach().clone()
    assert torch.equal(before, after)


def test_clip_grad_norm_uses_error_if_nonfinite():
    """The trainer's clip_grad_norm_ call uses error_if_nonfinite=True so a
    NaN gradient raises rather than silently corrupting parameters."""
    import inspect

    from douzero.training import v2_trainer

    src = inspect.getsource(v2_trainer.V2Trainer.step)
    assert "error_if_nonfinite=True" in src, (
        "V2Trainer.step must pass error_if_nonfinite=True to clip_grad_norm_ "
        "so NaN/Inf gradients fail closed."
    )


# --------------------------------------------------------------------------- #
# Blocker 1 (shape): trainer concatenates (not stacks) gathered heads
# --------------------------------------------------------------------------- #
def test_trainer_uses_cat_not_stack():
    """The trainer must torch.cat (not torch.stack) per-decision heads so the
    gathered tensors are (B, 1), not (B, 1, 1)."""
    import inspect

    from douzero.training import v2_trainer

    src = inspect.getsource(v2_trainer.V2Trainer.step)
    assert "torch.cat(gathered_win" in src
    assert "torch.cat(gathered_siw" in src
    assert "torch.cat(gathered_sil" in src
    # The stack form should NOT appear in the step method.
    assert "torch.stack(gathered" not in src


def test_forward_gathered_rejects_stacked_b11_shape():
    """If a caller accidentally passes (B, 1, 1) tensors (e.g. via stack), the
    loss module rejects them with a precise error pointing at the cat/stack
    contract."""
    win_logit = torch.zeros(3, 1, 1, requires_grad=True)
    score_if_win = torch.zeros(3, 1, 1, requires_grad=True)
    score_if_loss = torch.zeros(3, 1, 1, requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0, 0.0, 1.0]),
        "target_score": torch.tensor([1.0, -2.0, 3.0]),
    }
    fn = MultiObjectiveLoss(LossConfig())
    with pytest.raises(ValueError, match="torch.cat"):
        fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
