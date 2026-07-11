"""Tests for the scoring engine and GameResult (P02 Slice 2).

Covers:
- legacy scoring parity with game.py:78-95 (compute_player_utility +
  update_num_wins_scores)
- standard scoring across bid/bomb/rocket/spring combinations
- score conservation (landlord_score + 2 * farmer_score == 0)
- spring/anti-spring detection from action counts
- max_multiplier capping
- multiplier_breakdown completeness
"""

from __future__ import annotations

import pytest

from douzero.env.rules import RuleSet
from douzero.env.scoring import (
    GameResult,
    compute_game_result,
    detect_spring_from_action_counts,
)


# --------------------------------------------------------------------------- #
# Legacy scoring parity with game.py
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bomb_num", [0, 1, 2, 3])
@pytest.mark.parametrize("landlord_wins", [True, False])
def test_legacy_scoring_matches_game_py(bomb_num, landlord_wins):
    """compute_game_result in legacy mode must match game.py's scoring exactly.

    game.py: landlord_score = ±2 * 2**bomb_num, farmer_score = ∓1 * 2**bomb_num,
    where bomb_num counts bombs + rocket.
    """
    rs = RuleSet.legacy()
    winner_position = "landlord" if landlord_wins else "landlord_up"

    # Split bomb_num into bomb_count and rocket_count in a few ways; the
    # legacy result must be the same regardless (rocket is counted as a bomb).
    for bomb_count, rocket_count in [(bomb_num, 0), (bomb_num - 1, 1)] if bomb_num > 0 else [(0, 0)]:
        result = compute_game_result(
            played_cards={},
            action_counts={},
            winner_position=winner_position,
            bomb_count=bomb_count,
            rocket_count=rocket_count,
            bid_value=0,
            ruleset=rs,
        )
        expected_multiplier = 2 ** bomb_num
        assert result.total_multiplier == expected_multiplier
        if landlord_wins:
            assert result.landlord_score == 2 * expected_multiplier
            assert result.farmer_score == -expected_multiplier
        else:
            assert result.landlord_score == -2 * expected_multiplier
            assert result.farmer_score == expected_multiplier
        # Score conservation.
        assert result.landlord_score + 2 * result.farmer_score == 0
        # No spring in legacy.
        assert result.spring is False
        assert result.anti_spring is False
        assert result.bid_value == 0


def test_legacy_scoring_no_spring_even_if_farmers_never_played():
    """Legacy ruleset has spring_multiplier=0, so spring is never detected."""
    rs = RuleSet.legacy()
    result = compute_game_result(
        played_cards={"landlord": [3], "landlord_up": [], "landlord_down": []},
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=0,
        ruleset=rs,
    )
    assert result.spring is False
    assert result.anti_spring is False


# --------------------------------------------------------------------------- #
# Standard scoring
# --------------------------------------------------------------------------- #
def test_standard_basic_no_bombs_no_spring():
    """bid=1, no bombs, no spring: landlord wins → +2, farmer -1."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.landlord_score == 2
    assert result.farmer_score == -1
    assert result.total_multiplier == 1
    assert result.spring is False
    assert result.anti_spring is False
    assert result.bid_value == 1


def test_standard_farmer_win_no_bombs():
    """bid=1, no bombs, farmer wins: landlord -2, farmer +1."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord_up",
        bomb_count=0,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.landlord_score == -2
    assert result.farmer_score == 1


@pytest.mark.parametrize("bid", [1, 2, 3])
def test_standard_bid_multiplies_base(bid):
    """bid_value multiplies the base score: total = bid * 1 * 2."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=bid,
        ruleset=rs,
    )
    # base = 1 * bid, multiplier = 1, total = bid, landlord = 2*bid.
    assert result.landlord_score == 2 * bid
    assert result.farmer_score == -bid


@pytest.mark.parametrize("bomb_count", [0, 1, 2, 3])
def test_standard_bomb_multiplier(bomb_count):
    """Each bomb doubles: multiplier = 2**bomb_count."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=bomb_count,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    expected_mult = 2 ** bomb_count
    assert result.total_multiplier == expected_mult
    assert result.landlord_score == 2 * expected_mult
    assert result.farmer_score == -expected_mult


def test_standard_rocket_multiplier():
    """Rocket adds a x2 multiplier on top of bombs."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=1,
        bid_value=1,
        ruleset=rs,
    )
    assert result.total_multiplier == 2
    assert result.landlord_score == 4
    assert result.farmer_score == -2


def test_standard_bomb_and_rocket_combine():
    """1 bomb + rocket: multiplier = 2 * 2 = 4."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=1,
        bid_value=1,
        ruleset=rs,
    )
    assert result.total_multiplier == 4
    assert result.landlord_score == 8
    assert result.farmer_score == -4


def test_standard_bid_bomb_rocket_combined():
    """bid=3, 2 bombs, rocket: total_multiplier = 3*4*2=24, total=1*24=24, landlord=48."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=2,
        rocket_count=1,
        bid_value=3,
        ruleset=rs,
    )
    # total_multiplier = bid(3) × bombs(4) × rocket(2) = 24.
    expected_total_mult = 3 * 4 * 2
    assert result.total_multiplier == expected_total_mult
    expected_total = 1 * expected_total_mult  # base_score=1
    assert result.landlord_score == 2 * expected_total
    assert result.landlord_score == 2 * expected_total
    assert result.farmer_score == -expected_total


# --------------------------------------------------------------------------- #
# Score conservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bid", [1, 2, 3])
@pytest.mark.parametrize("bomb_count", [0, 1, 2])
@pytest.mark.parametrize("rocket_count", [0, 1])
@pytest.mark.parametrize("landlord_wins", [True, False])
def test_score_conservation_all_combinations(
    bid, bomb_count, rocket_count, landlord_wins
):
    """landlord_score + 2 * farmer_score == 0 for all standard combinations."""
    rs = RuleSet.standard()
    winner = "landlord" if landlord_wins else "landlord_down"
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position=winner,
        bomb_count=bomb_count,
        rocket_count=rocket_count,
        bid_value=bid,
        ruleset=rs,
    )
    assert result.landlord_score + 2 * result.farmer_score == 0, (
        f"Conservation violated: bid={bid}, bombs={bomb_count}, "
        f"rocket={rocket_count}, landlord_wins={landlord_wins}, "
        f"landlord={result.landlord_score}, farmer={result.farmer_score}"
    )
    # Winner always has positive score.
    if landlord_wins:
        assert result.landlord_score > 0
        assert result.farmer_score < 0
    else:
        assert result.landlord_score < 0
        assert result.farmer_score > 0


# --------------------------------------------------------------------------- #
# Spring / anti-spring detection
# --------------------------------------------------------------------------- #
def test_spring_landlord_wins_farmers_never_played():
    """Spring: landlord wins, both farmers have 0 valid actions."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord",
        ruleset=rs,
    )
    assert spring is True
    assert anti is False


def test_spring_false_if_farmer_played():
    """Not spring if a farmer played at least one valid action."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 5, "landlord_up": 1, "landlord_down": 0},
        winner_position="landlord",
        ruleset=rs,
    )
    assert spring is False
    assert anti is False


def test_spring_false_if_farmer_wins():
    """Spring only applies when the landlord wins."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord_up",
        ruleset=rs,
    )
    assert spring is False


def test_anti_spring_farmer_wins_landlord_one_action():
    """Anti-spring: farmers win, landlord played exactly 1 valid action."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 1, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord_down",
        ruleset=rs,
    )
    assert spring is False
    assert anti is True


def test_anti_spring_false_if_landlord_played_multiple():
    """Not anti-spring if landlord played more than one action."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 2, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord_up",
        ruleset=rs,
    )
    assert anti is False


def test_anti_spring_false_if_landlord_wins():
    """Anti-spring only applies when farmers win."""
    rs = RuleSet.standard()
    spring, anti = detect_spring_from_action_counts(
        action_counts={"landlord": 1, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        ruleset=rs,
    )
    assert anti is False


def test_spring_doubles_score():
    """Spring x2: bid=1, no bombs → multiplier=2, landlord=4, farmer=-2."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.spring is True
    assert result.total_multiplier == 2
    assert result.landlord_score == 4
    assert result.farmer_score == -2


def test_anti_spring_doubles_score():
    """Anti-spring x2: bid=1, no bombs → multiplier=2, landlord=-4, farmer=2."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 1, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord_up",
        bomb_count=0,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.anti_spring is True
    assert result.total_multiplier == 2
    assert result.landlord_score == -4
    assert result.farmer_score == 2


# --------------------------------------------------------------------------- #
# max_multiplier capping
# --------------------------------------------------------------------------- #
def test_max_multiplier_caps_total():
    """A max_multiplier caps the TOTAL multiplier (including bid, not just events)."""
    rs = RuleSet.from_dict({
        "ruleset_id": "standard",
        "max_multiplier": 4,
    })
    # bid=1, 3 bombs → event_multiplier=8, total_multiplier=1*8=8, capped to 4.
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=3,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.total_multiplier == 4
    assert result.landlord_score == 2 * 1 * 4  # 2 * base_score * capped_mult
    assert result.farmer_score == -1 * 4


def test_max_multiplier_caps_total_including_bid():
    """max_multiplier caps bid × event_multiplier, not just event_multiplier.

    bid=3, 1 bomb → total_multiplier = 3 * 2 = 6, capped to 4.
    Without bid in the cap, it would be event=2 < 4 (uncapped) → total=6.
    With the fix, total_multiplier = min(6, 4) = 4.
    """
    rs = RuleSet.from_dict({
        "ruleset_id": "standard",
        "max_multiplier": 4,
    })
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=0,
        bid_value=3,
        ruleset=rs,
    )
    # total_multiplier = min(3*2, 4) = 4.
    assert result.total_multiplier == 4
    assert result.multiplier_breakdown["uncapped_total_multiplier"] == 6


def test_max_multiplier_does_not_affect_below_cap():
    """Below the cap, the multiplier is unchanged."""
    rs = RuleSet.from_dict({
        "ruleset_id": "standard",
        "max_multiplier": 8,
    })
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    assert result.total_multiplier == 2


# --------------------------------------------------------------------------- #
# multiplier_breakdown completeness
# --------------------------------------------------------------------------- #
def test_multiplier_breakdown_contains_all_components():
    """breakdown should list bid, bomb, rocket, spring when present."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord",
        bomb_count=2,
        rocket_count=1,
        bid_value=3,
        ruleset=rs,
    )
    bd = result.multiplier_breakdown
    assert "bid" in bd and bd["bid"] == 3
    assert "bomb" in bd and bd["bomb"] == 4  # 2**2
    assert "rocket" in bd and bd["rocket"] == 2
    assert "spring" in bd and bd["spring"] == 2
    assert "total_multiplier" in bd


def test_multiplier_breakdown_empty_when_no_components():
    """No bombs, no rocket, no spring: breakdown has only total_multiplier=1."""
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    bd = result.multiplier_breakdown
    # bid is present (bid_multiplier=True, bid_value=1).
    assert "bid" in bd and bd["bid"] == 1
    # No bomb/rocket/spring keys.
    assert "bomb" not in bd
    assert "rocket" not in bd
    assert "spring" not in bd
    assert "anti_spring" not in bd
    assert bd["total_multiplier"] == 1


# --------------------------------------------------------------------------- #
# GameResult serialisation
# --------------------------------------------------------------------------- #
def test_game_result_to_dict_round_trip():
    rs = RuleSet.standard()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 0, "landlord_down": 0},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=1,
        bid_value=2,
        ruleset=rs,
    )
    d = result.to_dict()
    assert d["winner_team"] == "landlord"
    assert d["winner_position"] == "landlord"
    assert d["bid_value"] == 2
    assert d["bomb_count"] == 1
    assert d["rocket_count"] == 1
    assert d["spring"] is True
    assert d["landlord_score"] > 0
    assert isinstance(d["multiplier_breakdown"], dict)
    # Rule identity stamped.
    assert d["ruleset_id"] == "standard"
    assert d["ruleset_version"] == "standard-v1"
    assert len(d["ruleset_hash"]) == 64  # full SHA-256


def test_game_result_legacy_has_legacy_identity():
    """Legacy GameResult must have ruleset_id='legacy'."""
    rs = RuleSet.legacy()
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=0,
        rocket_count=0,
        bid_value=0,
        ruleset=rs,
    )
    assert result.ruleset_id == "legacy"
    assert result.ruleset_version == "legacy-v1"
    assert len(result.ruleset_hash) == 64  # full SHA-256


# --------------------------------------------------------------------------- #
# Invalid winner
# --------------------------------------------------------------------------- #
def test_invalid_winner_position_raises():
    rs = RuleSet.standard()
    with pytest.raises(ValueError, match="winner_position"):
        compute_game_result(
            played_cards={},
            action_counts={},
            winner_position="nobody",
            bomb_count=0,
            rocket_count=0,
            bid_value=1,
            ruleset=rs,
        )
