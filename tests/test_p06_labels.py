"""P06 team-perspective labels (AGENTS.md sign convention).

Verifies that :mod:`douzero.training.labels` derives ``target_win`` /
``target_score`` / ``target_log_score`` from a terminal GameResult-like
dict consistently from the acting player's team perspective, with no
scattered farmer negation. The same GameResult yields the same farmer-team
label for both farmer roles (the shared-utility invariant).
"""

from __future__ import annotations

import math

import pytest

from douzero.training.labels import (
    ALL_POSITIONS,
    LogScoreTransform,
    team_target_log_score,
    team_target_score,
    team_target_win,
    team_targets,
)


def _landlord_win_result(bomb_count: int = 2, base: int = 1) -> dict:
    """A GameResult-like dict where the landlord team won.

    ``base`` is the per-farmer magnitude at 0 bombs; landlord plays for two
    so ``landlord_score = +2*base*magnitude``, ``farmer_score = -base*magnitude``.
    """
    magnitude = base * (2 ** bomb_count)
    return {
        "winner_team": "landlord",
        "winner_position": "landlord",
        "bomb_count": bomb_count,
        "landlord_score": int(2 * magnitude),
        "farmer_score": int(-magnitude),
    }


def _farmer_win_result(bomb_count: int = 1, base: int = 1) -> dict:
    magnitude = base * (2 ** bomb_count)
    return {
        "winner_team": "farmer",
        "winner_position": "landlord_up",
        "bomb_count": bomb_count,
        "landlord_score": int(-2 * magnitude),
        "farmer_score": int(magnitude),
    }


# --------------------------------------------------------------------------- #
# target_win
# --------------------------------------------------------------------------- #
def test_target_win_landlord_perspective():
    r = _landlord_win_result()
    assert team_target_win(r, "landlord") == 1
    assert team_target_win(r, "landlord_up") == 0
    assert team_target_win(r, "landlord_down") == 0


def test_target_win_farmer_perspective_shared():
    r = _farmer_win_result()
    # Both farmer roles share the team win — no scattered negation.
    assert team_target_win(r, "landlord_up") == 1
    assert team_target_win(r, "landlord_down") == 1
    assert team_target_win(r, "landlord") == 0


def test_target_win_rejects_unknown_position():
    with pytest.raises(ValueError):
        team_target_win(_landlord_win_result(), "bystander")


def test_target_win_rejects_bad_winner_team():
    bad = {"winner_team": "draw", "landlord_score": 0, "farmer_score": 0}
    with pytest.raises(ValueError):
        team_target_win(bad, "landlord")


# --------------------------------------------------------------------------- #
# target_score (sign + magnitude + conservation)
# --------------------------------------------------------------------------- #
def test_target_score_signs_landlord_win():
    r = _landlord_win_result(bomb_count=2)  # magnitude = 4
    # Landlord team won: landlord score positive (plays for two).
    assert team_target_score(r, "landlord") == 8.0
    # Farmer team lost: score negative (per farmer).
    assert team_target_score(r, "landlord_up") == -4.0
    assert team_target_score(r, "landlord_down") == -4.0


def test_target_score_signs_farmer_win():
    r = _farmer_win_result(bomb_count=1)  # magnitude = 2
    assert team_target_score(r, "landlord") == -4.0
    assert team_target_score(r, "landlord_up") == 2.0
    assert team_target_score(r, "landlord_down") == 2.0


def test_score_conservation_invariant():
    """landlord_score + 2*farmer_score == 0 from any terminal result."""
    for r in (_landlord_win_result(0), _landlord_win_result(3),
              _farmer_win_result(0), _farmer_win_result(4)):
        ls = r["landlord_score"]
        fs = r["farmer_score"]
        assert ls + 2 * fs == 0, f"conservation violated: {r}"


def test_farmer_team_score_identical_for_both_roles():
    """No scattered negation: both farmers see the same team score."""
    r = _farmer_win_result(bomb_count=2)
    assert team_target_score(r, "landlord_up") == team_target_score(r, "landlord_down")


# --------------------------------------------------------------------------- #
# target_log_score (sign-preserving log transform)
# --------------------------------------------------------------------------- #
def test_log_score_zero_is_zero():
    r = {"winner_team": "landlord", "landlord_score": 0, "farmer_score": 0}
    for pos in ALL_POSITIONS:
        assert team_target_log_score(r, pos) == 0.0


def test_log_score_sign_and_magnitude():
    r = _landlord_win_result(bomb_count=2)  # landlord_score = 8, farmer = -4
    ls = team_target_log_score(r, "landlord")
    fs = team_target_log_score(r, "landlord_up")
    assert ls > 0
    assert fs < 0
    assert ls == pytest.approx(math.copysign(math.log1p(8.0), 1.0))
    assert fs == pytest.approx(math.copysign(math.log1p(4.0), -1.0))


def test_log_score_transform_callable_matches_helper():
    r = _landlord_win_result(bomb_count=1)
    transform = LogScoreTransform()
    assert transform(team_target_score(r, "landlord")) == pytest.approx(
        team_target_log_score(r, "landlord")
    )


def test_team_targets_keys_and_consistency():
    r = _farmer_win_result(bomb_count=0)
    for pos in ALL_POSITIONS:
        t = team_targets(r, pos)
        assert set(t.keys()) == {"target_win", "target_score", "target_log_score"}
        # target_win in {0, 1}
        assert t["target_win"] in (0.0, 1.0)
        # log-score sign matches score sign
        if t["target_score"] > 0:
            assert t["target_log_score"] > 0
        elif t["target_score"] < 0:
            assert t["target_log_score"] < 0
        else:
            assert t["target_log_score"] == 0.0


def test_team_targets_accepts_gameresult_object():
    """The helpers accept a GameResult-like object too (attribute access)."""
    from douzero.env.scoring import GameResult

    gr = GameResult(
        winner_team="landlord",
        winner_position="landlord",
        bid_value=0,
        bomb_count=1,
        rocket_count=0,
        spring=False,
        anti_spring=False,
        landlord_score=4,
        farmer_score=-2,
    )
    assert team_target_win(gr, "landlord") == 1
    assert team_target_score(gr, "landlord") == 4.0
    assert team_target_score(gr, "landlord_up") == -2.0
