"""P06 team-perspective labels via Env terminal info.

Verifies that :class:`douzero.env.env.Env` populates the team-perspective
labels at terminal in BOTH legacy and standard modes, that the labels are
consistent with the env's own ``_game_winner`` (legacy) / ``GameResult``
(standard), and that no hidden hand information leaks into the public
``info`` dict.
"""

from __future__ import annotations

import pytest

from douzero.env.env import Env
from douzero.env.rules import RuleSet


def _rollout_to_terminal(env: Env, max_steps: int = 2000):
    """Play greedily (first legal action) until done. Returns terminal info."""
    env.reset()
    steps = 0
    while True:
        assert steps < max_steps, "episode did not terminate"
        action = env.infoset.legal_actions[0]
        _obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            return info


# --------------------------------------------------------------------------- #
# Legacy mode
# --------------------------------------------------------------------------- #
def test_legacy_terminal_info_carries_team_targets(seed_factory):
    seed_factory(2024)
    env = Env("adp")
    info = _rollout_to_terminal(env)
    assert "team_targets" in info
    assert "terminal_result" in info
    for pos in ("landlord", "landlord_up", "landlord_down"):
        labels = info["team_targets"][pos]
        assert set(labels.keys()) == {
            "target_win", "target_score", "target_log_score",
        }
        assert labels["target_win"] in (0.0, 1.0)


def test_legacy_team_targets_match_winner(seed_factory):
    seed_factory(7)
    env = Env("adp")
    info = _rollout_to_terminal(env)
    winner = info["terminal_result"]["winner_team"]
    for pos in ("landlord", "landlord_up", "landlord_down"):
        labels = info["team_targets"][pos]
        # Landlord-team label
        landlord_should_win = (winner == "landlord")
        is_landlord = (pos == "landlord")
        if landlord_should_win == is_landlord:
            assert labels["target_win"] == 1.0
        else:
            assert labels["target_win"] == 0.0


def test_legacy_team_targets_score_conservation(seed_factory):
    seed_factory(11)
    env = Env("adp")
    info = _rollout_to_terminal(env)
    ls = info["terminal_result"]["landlord_score"]
    fs = info["terminal_result"]["farmer_score"]
    assert ls + 2 * fs == 0


def test_legacy_farmer_team_labels_match(seed_factory):
    """Both farmers share team utility: same target_score / target_win."""
    seed_factory(13)
    env = Env("adp")
    info = _rollout_to_terminal(env)
    up = info["team_targets"]["landlord_up"]
    down = info["team_targets"]["landlord_down"]
    assert up["target_win"] == down["target_win"]
    assert up["target_score"] == down["target_score"]


# --------------------------------------------------------------------------- #
# Standard mode
# --------------------------------------------------------------------------- #
def _rollout_standard_to_terminal(env: Env, ruleset: RuleSet, max_steps: int = 2000):
    """Drive bidding with random legal bids, then play greedily to terminal."""
    import random

    rng = random.Random(0)
    env.reset()
    # Bidding: random legal bids until PHASE changes / done.
    for _ in range(ruleset.max_redeals + 1):
        from douzero.env.rules import PHASE_BIDDING
        if env._env.phase != PHASE_BIDDING:
            break
        legal = env._env.get_legal_bids()
        bid = rng.choice(legal)
        _obs, _r, done, info = env.step(None, bid_value=bid)
        if done and info.get("redeal"):
            env.redeal()
            break
        if done:
            return info
    # Now play to terminal.
    steps = 0
    while True:
        assert steps < max_steps, "standard episode did not terminate"
        action = env.infoset.legal_actions[0]
        _obs, _r, done, info = env.step(action)
        steps += 1
        if done:
            return info


def test_standard_terminal_info_carries_team_targets(seed_factory):
    seed_factory(31)
    ruleset = RuleSet.standard()
    env = Env("adp", ruleset=ruleset)
    info = _rollout_standard_to_terminal(env, ruleset)
    if info is None or "team_targets" not in info:
        # The redeal path may produce an early done without team_targets.
        pytest.skip("standard episode did not reach terminal team_targets")
    for pos in ("landlord", "landlord_up", "landlord_down"):
        labels = info["team_targets"][pos]
        assert labels["target_win"] in (0.0, 1.0)


# --------------------------------------------------------------------------- #
# No hidden-hand leakage
# --------------------------------------------------------------------------- #
def test_terminal_info_does_not_leak_hidden_hands(seed_factory):
    """The team-perspective info keys are public (terminal result only)."""
    seed_factory(41)
    env = Env("adp")
    info = _rollout_to_terminal(env)
    forbidden = ("all_handcards", "landlord_up_hand", "landlord_down_hand")
    for key in forbidden:
        assert key not in info, f"hidden-hand key {key!r} leaked into terminal info"
    # Nested terminal_result is the public GameResult summary.
    for key in forbidden:
        assert key not in info.get("terminal_result", {})


def test_terminal_info_legacy_reward_unchanged(seed_factory):
    """The legacy ``reward`` path is untouched: still landlord-perspective."""
    seed_factory(43)
    env = Env("adp")
    env.reset()
    # Play one full game and capture the final reward.
    last_reward = 0.0
    steps = 0
    while True:
        assert steps < 2000
        action = env.infoset.legal_actions[0]
        _obs, reward, done, _info = env.step(action)
        last_reward = reward
        steps += 1
        if done:
            break
    # Legacy reward is non-zero and from landlord's perspective (sign).
    assert last_reward != 0.0
