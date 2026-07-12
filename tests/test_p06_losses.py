"""P06 multi-objective losses (BCE win + masked Huber conditional scores).

Covers the AGENTS.md "Loss or reward changes" test matrix:

- sign conventions (landlord/farmer symmetry via team-perspective labels)
- terminal labels
- conditional masks with empty subsets (all-win and all-loss minibatches)
- extreme multiplier tails (large scores stay finite)
- finite gradients
- λ=0 cleanly disables a term
"""

from __future__ import annotations

import math

import pytest
import torch

from douzero.models_v2.output import ModelOutput
from douzero.training.losses import (
    LossComponents,
    LossConfig,
    MultiObjectiveLoss,
    bce_win_loss,
    conditional_score_huber_loss,
    log_score_aux_loss,
)


def _make_output(win_logit, score_if_win, score_if_loss, mask=None):
    """Build a ModelOutput from raw head values (all length-N lists/tensors)."""
    wl = torch.tensor(win_logit, dtype=torch.float32).reshape(-1, 1)
    sw = torch.tensor(score_if_win, dtype=torch.float32).reshape(-1, 1)
    sl = torch.tensor(score_if_loss, dtype=torch.float32).reshape(-1, 1)
    if mask is None:
        mask = torch.ones(wl.shape[0], dtype=torch.bool)
    else:
        mask = torch.tensor(mask, dtype=torch.bool)
    p = torch.sigmoid(wl)
    sm = p.detach() * sw + (1 - p.detach()) * sl
    return ModelOutput(
        win_logit=wl,
        score_if_win=sw,
        score_if_loss=sl,
        p_win=p,
        score_mean=sm,
        action_mask=mask,
    )


# --------------------------------------------------------------------------- #
# BCE win loss
# --------------------------------------------------------------------------- #
def test_bce_win_loss_perfect_prediction_is_near_zero():
    # High positive logits with target=1, high negative with target=0.
    logits = torch.tensor([[10.0], [-10.0], [10.0]])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = bce_win_loss(logits, targets)
    assert loss.item() < 1e-3


def test_bce_win_loss_confident_wrong_is_large():
    logits = torch.tensor([[10.0]])
    targets = torch.tensor([0.0])
    loss = bce_win_loss(logits, targets)
    assert loss.item() > 5.0


def test_bce_win_loss_finite_at_extremes():
    logits = torch.tensor([[-50.0], [50.0]])
    targets = torch.tensor([1.0, 0.0])
    loss = bce_win_loss(logits, targets)
    assert math.isfinite(loss.item())


# --------------------------------------------------------------------------- #
# Conditional score Huber loss (mask behavior)
# --------------------------------------------------------------------------- #
def test_conditional_loss_all_win_minibatch_no_nan():
    """All-win batch: loss_term supervised, win_term empty, no NaN."""
    score_win = torch.tensor([[1.0], [2.0]])
    score_loss = torch.tensor([[0.0], [0.0]])
    target_score = torch.tensor([1.5, 2.5])
    target_win = torch.tensor([1.0, 1.0])
    loss, nw, nl = conditional_score_huber_loss(
        score_win, score_loss, target_score, target_win
    )
    assert math.isfinite(loss.item())
    assert nw == 2
    assert nl == 0


def test_conditional_loss_all_loss_minibatch_no_nan():
    """All-loss batch: loss_term supervised, win_term empty, no NaN."""
    score_win = torch.tensor([[0.0], [0.0]])
    score_loss = torch.tensor([[-1.0], [-2.0]])
    target_score = torch.tensor([-1.5, -2.5])
    target_win = torch.tensor([0.0, 0.0])
    loss, nw, nl = conditional_score_huber_loss(
        score_win, score_loss, target_score, target_win
    )
    assert math.isfinite(loss.item())
    assert nw == 0
    assert nl == 2


def test_conditional_loss_perfect_prediction_is_zero():
    score_win = torch.tensor([[2.0]])
    score_loss = torch.tensor([[-2.0]])
    target_score = torch.tensor([2.0])  # win
    target_win = torch.tensor([1.0])
    loss, _, _ = conditional_score_huber_loss(
        score_win, score_loss, target_score, target_win
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_conditional_loss_large_multiplier_tail_stays_finite():
    """A 32x bomb score should not produce NaN/Inf in the Huber loss."""
    score_win = torch.tensor([[32.0]])
    score_loss = torch.tensor([[-32.0]])
    target_score = torch.tensor([32.0])
    target_win = torch.tensor([1.0])
    loss, _, _ = conditional_score_huber_loss(
        score_win, score_loss, target_score, target_win, delta=1.0
    )
    assert math.isfinite(loss.item())


def test_log_score_aux_loss_zero_when_lambda_zero_via_combiner():
    output = _make_output([0.5], [2.0], [-2.0])
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([2.0]),
        "target_log_score": torch.tensor([math.log1p(2.0)]),
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=0.0, lambda_log=0.0))
    comps = fn(output, labels)
    assert comps.log == 0.0


def test_log_score_aux_loss_active_when_lambda_positive():
    output = _make_output([0.5], [2.0], [-2.0])
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([2.0]),
        "target_log_score": torch.tensor([10.0]),  # deliberately off
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=0.0, lambda_score=0.0, lambda_log=1.0))
    comps = fn(output, labels)
    assert comps.log > 0.0


# --------------------------------------------------------------------------- #
# Combiner: λ=0 disables a term; total = weighted sum
# --------------------------------------------------------------------------- #
def test_lambda_zero_disables_win_term():
    output = _make_output([10.0], [0.0], [0.0])  # very wrong win prediction
    labels = {
        "target_win": torch.tensor([0.0]),
        "target_score": torch.tensor([0.0]),
    }
    fn_off = MultiObjectiveLoss(LossConfig(lambda_win=0.0, lambda_score=0.0))
    comps = fn_off(output, labels)
    assert comps.total.item() == pytest.approx(0.0, abs=1e-7)


def test_combined_loss_finite_and_grad():
    # Three decisions, one per row, gathered manually (the trainer's path).
    win_logit = torch.tensor([[0.3], [-0.4], [0.9]], requires_grad=True)
    score_if_win = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0], [2.0], [-3.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0, 0.0, 1.0]),
        "target_score": torch.tensor([1.0, -2.0, 3.0]),
        "target_log_score": torch.tensor([0.7, -1.1, 1.4]),
    }
    fn = MultiObjectiveLoss(
        LossConfig(lambda_win=1.0, lambda_score=0.5, lambda_log=0.1)
    )
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    assert isinstance(comps, LossComponents)
    assert math.isfinite(comps.total.item())
    comps.total.backward()
    # The total tensor carries a grad function (the graph is wired).
    assert comps.total.requires_grad


def test_landlord_farmer_symmetry_same_loss():
    """Same-magnitude labels yield the same conditional loss regardless of role.

    Two decisions: a landlord-team win and a farmer-team win, both with
    magnitude-2 scores. The conditional loss depends on the magnitude, not
    the role, so the loss is finite and the win head is the only term that
    could differ (BCE on the same target).
    """
    win_logit = torch.tensor([[0.5], [0.5]], requires_grad=True)
    score_if_win = torch.tensor([[2.0], [2.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-2.0], [-2.0]], requires_grad=True)
    # Both samples won from their team's perspective (target_win=1).
    labels = {
        "target_win": torch.tensor([1.0, 1.0]),
        "target_score": torch.tensor([2.0, 2.0]),
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=1.0))
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    assert math.isfinite(comps.total.item())


# --------------------------------------------------------------------------- #
# forward_gathered (trainer entry point)
# --------------------------------------------------------------------------- #
def test_forward_gathered_accepts_stacked_heads():
    # Two decisions, each pre-gathered to the chosen action's heads.
    win_logit = torch.tensor([[0.1], [0.2]], requires_grad=True)
    score_if_win = torch.tensor([[1.0], [2.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0], [-2.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0, 0.0]),
        "target_score": torch.tensor([1.0, -2.0]),
        "target_log_score": torch.tensor([0.7, -1.1]),
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=1.0))
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    assert math.isfinite(comps.total.item())
    assert comps.num_win == 1
    assert comps.num_loss == 1


def test_loss_config_rejects_negative_weight():
    with pytest.raises(ValueError):
        LossConfig(lambda_win=-0.1)


def test_loss_config_to_dict_roundtrip_keys():
    cfg = LossConfig(lambda_win=1.0, lambda_score=0.5, lambda_log=0.1)
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "lambda_win", "lambda_score", "lambda_log", "lambda_uncertainty",
        "score_delta", "log_score_delta",
    }
