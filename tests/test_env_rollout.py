"""Integration tests: drive ``Env`` from reset to terminal and assert invariants.

These exercise the full card-play loop (shuffle -> deal -> play -> terminal)
with a deterministic policy (always the first legal action). They guard the
domain invariants listed in AGENTS.md: turn order, hand conservation, terminal
scoring, and the three objective reward conventions.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from douzero.env.env import Env, _cards2array, deck


def _rollout_to_terminal(env: Env, max_steps: int = 2000):
    """Play greedily (first legal action) until done. Returns terminal info."""
    obs = env.reset()
    steps = 0
    while True:
        assert steps < max_steps, "episode did not terminate (infinite loop?)"
        action = env.infoset.legal_actions[0]
        assert action in env.infoset.legal_actions
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            return {
                "steps": steps,
                "reward": reward,
                "winner": env._game_winner,
                "bomb_num": env._game_bomb_num,
                "info": info,
            }


def _played_and_remaining(env: Env):
    """Sum of all cards accounted for across hands and played piles."""
    hands = env._env.info_sets
    total = 0
    for pos in ["landlord", "landlord_up", "landlord_down"]:
        total += len(hands[pos].player_hand_cards)
    for pos in ["landlord", "landlord_up", "landlord_down"]:
        total += len(env._env.played_cards[pos])
    return total


@pytest.mark.parametrize("objective", ["adp", "wp", "logadp"])
def test_rollout_terminates_and_returns_terminal_reward(objective, seed_factory):
    seed_factory(101)
    env = Env(objective)
    result = _rollout_to_terminal(env)
    assert result["winner"] in {"landlord", "farmer"}
    # P06 attaches team-perspective labels to terminal info; the legacy
    # ``reward`` path itself is unchanged. Assert the keys P06 added.
    assert "team_targets" in result["info"]
    assert "terminal_result" in result["info"]
    for pos in ("landlord", "landlord_up", "landlord_down"):
        labels = result["info"]["team_targets"][pos]
        assert labels["target_win"] in (0.0, 1.0)
    # Terminal reward is non-zero and from the landlord's perspective.
    bomb = result["bomb_num"]
    if result["winner"] == "landlord":
        assert result["reward"] > 0
        if objective == "adp":
            assert result["reward"] == 2.0 ** bomb
        elif objective == "logadp":
            assert result["reward"] == bomb + 1.0
        else:
            assert result["reward"] == 1.0
    else:
        assert result["reward"] < 0
        if objective == "adp":
            assert result["reward"] == -(2.0 ** bomb)
        elif objective == "logadp":
            assert result["reward"] == -(bomb + 1.0)
        else:
            assert result["reward"] == -1.0


def test_terminal_step_returns_none_obs(seed_factory):
    seed_factory(102)
    env = Env("adp")
    env.reset()
    done = False
    last_obs = None
    steps = 0
    while not done:
        action = env.infoset.legal_actions[0]
        last_obs, _, done, _ = env.step(action)
        steps += 1
        assert steps < 2000
    assert last_obs is None, "terminal obs must be None"


def test_non_terminal_steps_have_zero_reward(seed_factory):
    seed_factory(103)
    env = Env("adp")
    env.reset()
    # Play a few steps and confirm reward is exactly 0.0 until terminal.
    for _ in range(5):
        action = env.infoset.legal_actions[0]
        _, reward, done, _ = env.step(action)
        if done:
            return
        assert reward == 0.0


def test_card_conservation_across_episode(seed_factory):
    """landlord + up + down hands + all played cards == 54 at every step."""
    seed_factory(104)
    env = Env("adp")
    env.reset()
    assert _played_and_remaining(env) == 54
    done = False
    steps = 0
    while not done:
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        steps += 1
        assert _played_and_remaining(env) == 54, "cards leaked/created mid-game"
        assert steps < 2000
    # At terminal, the winner's hand is empty but total is still 54.
    assert _played_and_remaining(env) == 54


def test_hand_counts_never_negative(seed_factory):
    seed_factory(105)
    env = Env("adp")
    env.reset()
    done = False
    steps = 0
    while not done:
        for pos in ["landlord", "landlord_up", "landlord_down"]:
            assert len(env._env.info_sets[pos].player_hand_cards) >= 0
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        steps += 1
        assert steps < 2000


def test_exactly_one_team_wins_at_terminal(seed_factory):
    seed_factory(106)
    env = Env("adp")
    result = _rollout_to_terminal(env)
    # winner is a single team label.
    assert result["winner"] in {"landlord", "farmer"}


def test_turn_order_is_landlord_down_up(seed_factory):
    """First four actors must be landlord, landlord_down, landlord_up, landlord."""
    seed_factory(107)
    env = Env("adp")
    env.reset()
    seen = [env._acting_player_position]
    for _ in range(3):
        action = env.infoset.legal_actions[0]
        env.step(action)
        seen.append(env._acting_player_position)
    assert seen == ["landlord", "landlord_down", "landlord_up", "landlord"]


def test_reset_reseeds_deck(seed_factory):
    """Two resets with different seeds should usually yield different deals."""
    seed_factory(11)
    env = Env("adp")
    env.reset()
    hand_a = list(env._env.info_sets["landlord"].player_hand_cards)

    seed_factory(999)
    env.reset()
    hand_b = list(env._env.info_sets["landlord"].player_hand_cards)
    assert hand_a != hand_b


def test_deck_is_54_and_well_formed():
    counts = Counter(deck)
    for rank in range(3, 15):
        assert counts[rank] == 4
    assert counts[17] == 4
    assert counts[20] == 1
    assert counts[30] == 1
    assert sum(counts.values()) == 54


def test_three_landlord_cards_belong_to_landlord_after_deal(seed_factory):
    """The 3 bottom cards must be a subset of the landlord's 20-card hand."""
    seed_factory(108)
    env = Env("adp")
    env.reset()
    bottom = set(env._env.three_landlord_cards)
    landlord_hand = Counter(env._env.info_sets["landlord"].player_hand_cards)
    for card in bottom:
        assert landlord_hand[card] >= 1


def test_action_must_be_in_legal_actions(seed_factory):
    seed_factory(109)
    env = Env("adp")
    env.reset()
    # An illegal action (a card not in hand) must be rejected by the assertion.
    illegal = [99]
    with pytest.raises(AssertionError):
        env.step(illegal)
