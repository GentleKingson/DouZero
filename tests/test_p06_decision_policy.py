"""P06 multi-objective decision policy.

Verifies that :func:`douzero.training.decision_policy.select_action` picks
the expected action under each of the five modes, that the additive
tolerance band is sign-safe (negative scores behave identically), and that
the ``win`` / ``score`` aliases resolve to ``pure_win`` / ``pure_score``.
"""

from __future__ import annotations

import pytest
import torch

from douzero.models_v2.output import ModelOutput
from douzero.training.decision_policy import (
    DecisionConfig,
    SUPPORTED_DECISION_MODES,
    canonical_mode,
    select_action,
)


def _make_output(p_wins, score_means, score_if_win=None, score_if_loss=None, mask=None):
    """Build a ModelOutput with chosen p_win and score_mean per action.

    win_logit is recovered as logit(p_win) so the model's p_win is exact.
    """
    n = len(p_wins)
    p = torch.tensor(p_wins, dtype=torch.float32).reshape(-1, 1)
    wl = torch.log(p / (1.0 - p)).clamp(-50.0, 50.0)
    sm = torch.tensor(score_means, dtype=torch.float32).reshape(-1, 1)
    if score_if_win is None:
        score_if_win = sm.clone()
    else:
        score_if_win = torch.tensor(score_if_win, dtype=torch.float32).reshape(-1, 1)
    if score_if_loss is None:
        score_if_loss = sm.clone()
    else:
        score_if_loss = torch.tensor(score_if_loss, dtype=torch.float32).reshape(-1, 1)
    if mask is None:
        mask = torch.ones(n, dtype=torch.bool)
    else:
        mask = torch.tensor(mask, dtype=torch.bool)
    return ModelOutput(
        win_logit=wl,
        score_if_win=score_if_win,
        score_if_loss=score_if_loss,
        p_win=p,
        score_mean=sm,
        action_mask=mask,
    )


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #
def test_win_alias_resolves_to_pure_win():
    assert canonical_mode("win") == "pure_win"


def test_score_alias_resolves_to_pure_score():
    assert canonical_mode("score") == "pure_score"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        canonical_mode("greedy")


def test_all_supported_modes_select_valid_action():
    output = _make_output([0.4, 0.6, 0.5], [-1.0, 0.5, 0.0])
    for mode in SUPPORTED_DECISION_MODES:
        if mode in ("pure_prior", "uncertainty_gated_prior"):
            # pure_prior (P08) requires a prior head; the default _make_output
            # has no prior_logit. Build one with a prior head and verify it
            # selects a valid action, and separately that a prior-less output
            # raises (covered in test_p08_prior_and_loss).
            prior_out = ModelOutput(
                win_logit=output.win_logit,
                score_if_win=output.score_if_win,
                score_if_loss=output.score_if_loss,
                p_win=output.p_win,
                score_mean=output.score_mean,
                action_mask=output.action_mask,
                prior_logit=torch.tensor([[0.0], [0.7], [0.2]]),
            )
            idx = select_action(prior_out, DecisionConfig(mode=mode, risk_penalty=0.1))
            assert 0 <= idx < 3
            continue
        cfg = DecisionConfig(mode=mode, risk_penalty=0.1)
        idx = select_action(output, cfg)
        assert 0 <= idx < 3


# --------------------------------------------------------------------------- #
# pure_win / pure_score
# --------------------------------------------------------------------------- #
def test_pure_win_picks_highest_p_win():
    output = _make_output([0.2, 0.9, 0.5], [0.0, 0.0, 0.0])
    assert select_action(output, DecisionConfig(mode="pure_win")) == 1


def test_pure_score_picks_highest_expected_score():
    output = _make_output([0.5, 0.5, 0.5], [-1.0, 2.0, 0.5])
    assert select_action(output, DecisionConfig(mode="pure_score")) == 1


def test_pure_win_respects_mask():
    output = _make_output([0.2, 0.99, 0.5], [0.0, 0.0, 0.0], mask=[True, False, True])
    # Action 1 has the highest p_win but is masked out.
    assert select_action(output, DecisionConfig(mode="pure_win")) == 2


def test_select_action_rejects_all_masked():
    output = _make_output([0.2, 0.9], [0.0, 0.0], mask=[False, False])
    with pytest.raises(ValueError):
        select_action(output, DecisionConfig(mode="pure_win"))


# --------------------------------------------------------------------------- #
# Lexicographic modes (tolerance band)
# --------------------------------------------------------------------------- #
def test_win_then_score_breaks_ties_by_score():
    # Two actions with nearly-equal p_win; the second has the better score.
    output = _make_output([0.50, 0.51], [1.0, 5.0])
    idx = select_action(
        output,
        DecisionConfig(mode="win_then_score", abs_tol=0.05),
    )
    assert idx == 1


def test_win_then_score_keeps_clear_winner_when_outside_band():
    # Two actions well-separated in p_win; the band does not include the
    # second action even though it has the better score.
    output = _make_output([0.90, 0.50], [1.0, 5.0])
    idx = select_action(
        output,
        DecisionConfig(mode="win_then_score", abs_tol=0.05),
    )
    assert idx == 0


def test_score_then_win_breaks_ties_by_win():
    # Two actions with nearly-equal score; the second has the better p_win.
    output = _make_output([0.40, 0.80], [1.00, 1.01])
    idx = select_action(
        output,
        DecisionConfig(mode="score_then_win", abs_tol=0.05),
    )
    assert idx == 1


def test_tolerance_band_negative_safe():
    """Additive tolerance must NOT widen for negative best values.

    A multiplicative threshold ``|x - best| <= rel * |best|`` would shrink
    toward zero as ``best -> 0`` and behave inconsistently across the
    negative/positive score range. The additive band keeps the same width
    on both sides of zero.
    """
    # Two negative scores near -1.0; the band of 0.1 should include both.
    output = _make_output([0.5, 0.5], [-1.00, -1.05])
    idx = select_action(
        output,
        DecisionConfig(mode="score_then_win", abs_tol=0.1, rel_tol=0.0),
    )
    # Both are in the band; the tie-break (p_win) is 0.5 for both, so the
    # lowest index wins.
    assert idx == 0


def test_relative_tolerance_scales_with_max_one_abs_best():
    """rel_tol uses max(1, |best|) so it stays meaningful near zero."""
    # best p_win ~0.5; rel_tol=0.1 adds 0.1*max(1, 0.5) = 0.1 to the band.
    output = _make_output([0.50, 0.42], [1.0, 2.0])
    idx = select_action(
        output,
        DecisionConfig(mode="win_then_score", abs_tol=0.0, rel_tol=0.1),
    )
    # 0.42 is within 0.5 - 0.1 = 0.4? No (0.42 > 0.4), so it's in the band.
    # The second action has the better score, so it wins the tie-break.
    assert idx == 1


# --------------------------------------------------------------------------- #
# risk_aware (default off)
# --------------------------------------------------------------------------- #
def test_risk_aware_with_zero_penalty_equals_pure_score():
    output = _make_output([0.5, 0.5], [1.0, 2.0])
    idx_risk = select_action(
        output, DecisionConfig(mode="risk_aware", risk_penalty=0.0)
    )
    assert idx_risk == 1  # matches pure_score


def test_risk_aware_penalty_prefers_low_uncertainty():
    # Two actions with identical score_mean but different p_win spread.
    # Action 0: p=0.9 (certain win), score=1.0
    # Action 1: p=0.5 (uncertain),   score=1.0  (but score_if_win/score_if_loss differ widely)
    output = _make_output(
        [0.9, 0.5],
        [1.0, 1.0],
        score_if_win=[1.0, 2.0],
        score_if_loss=[1.0, 0.0],
    )
    idx = select_action(
        output, DecisionConfig(mode="risk_aware", risk_penalty=2.0)
    )
    # The penalty pushes action 1 below action 0.
    assert idx == 0


def test_decision_config_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        DecisionConfig(abs_tol=-0.1)


def test_decision_config_to_dict_roundtrip():
    cfg = DecisionConfig(mode="win_then_score", abs_tol=0.1, rel_tol=0.05, risk_penalty=0.2)
    d = cfg.to_dict()
    assert d["mode"] == "win_then_score"
    assert d["abs_tol"] == 0.1
    assert d["rel_tol"] == 0.05
    assert d["risk_penalty"] == 0.2
