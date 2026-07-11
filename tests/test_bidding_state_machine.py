"""Tests for the bidding state machine and standard-mode GameEnv (P02 Slice 3).

Covers:
- legacy path is byte-for-byte unchanged (GameEnv without ruleset)
- standard DEAL→BIDDING→REVEAL→PLAYING→TERMINAL phase transitions
- illegal phase / illegal action errors
- three-player bidding: highest bidder becomes landlord
- tie-breaking by bidding order
- all-pass redeal
- bottom card reveal (3 cards added to landlord hand)
- bottom card entity identity preservation
- bidding observation contains no privileged info
- full standard game rollout terminates and produces GameResult
- score conservation at terminal
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from douzero.env.env import Env
from douzero.env.game import GameEnv, IllegalActionError, IllegalPhaseError
from douzero.env.rules import (
    PHASE_BIDDING,
    PHASE_DEAL,
    PHASE_PLAYING,
    PHASE_REVEAL_BOTTOM,
    PHASE_TERMINAL,
    RuleSet,
)


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
class _StubAgent:
    """Minimal agent stub for driving GameEnv directly."""

    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def act(self, infoset):
        if self.action is not None and self.action in infoset.legal_actions:
            a, self.action = self.action, None
            return a
        return infoset.legal_actions[0]


def _make_standard_deck():
    """A fixed 54-card deck for deterministic testing."""
    d = []
    for rank in range(3, 15):
        d.extend([rank] * 4)
    d.extend([17] * 4)
    d.extend([20, 30])
    return d


def _deal_standard(deck_order):
    """Slice a deck into 17+17+17+3 (standard dealing)."""
    return {
        'landlord': sorted(deck_order[:17]),
        'landlord_up': sorted(deck_order[17:34]),
        'landlord_down': sorted(deck_order[34:51]),
        'three_landlord_cards': sorted(deck_order[51:54]),
    }


# --------------------------------------------------------------------------- #
# Legacy path is unchanged
# --------------------------------------------------------------------------- #
def test_legacy_gameenv_has_no_phase_field():
    """GameEnv without ruleset must not set phase (legacy path unchanged)."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    assert env.ruleset is None
    assert env.phase is None
    assert env.bidding_history == []
    assert env.landlord_position is None
    assert env.game_result is None


def test_legacy_gameenv_step_unchanged():
    """Legacy GameEnv.step() must work exactly as before (no ruleset)."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    deck = _make_standard_deck()
    np.random.seed(42)
    np.random.shuffle(deck)
    data = {
        'landlord': sorted(deck[:20]),
        'landlord_up': sorted(deck[20:37]),
        'landlord_down': sorted(deck[37:54]),
        'three_landlord_cards': sorted(deck[17:20]),
    }
    env.card_play_init(data)
    assert env.acting_player_position == 'landlord'
    # Play a few steps — must not raise.
    for _ in range(5):
        env.step()


def test_legacy_env_constructor_unchanged():
    """Env(objective) without ruleset must work as before."""
    env = Env("adp")
    assert env.ruleset is None
    assert env.bidding_obs is None
    np.random.seed(42)
    obs = env.reset()
    # Legacy reset returns a get_obs dict, not a bidding obs.
    assert 'x_batch' in obs
    assert 'z_batch' in obs


# --------------------------------------------------------------------------- #
# Standard mode phase transitions
# --------------------------------------------------------------------------- #
def test_standard_env_reset_enters_bidding_phase():
    """Env with standard ruleset must enter BIDDING after reset."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    obs = env.reset()
    assert env._env.phase == PHASE_BIDDING
    assert obs['phase'] == 'bidding'
    assert 'my_handcards' in obs
    assert 'bidding_history' in obs
    assert 'bid_values' in obs
    assert obs['bid_values'] == [0, 1, 2, 3]


def test_standard_initial_deal_gives_17_cards_each():
    """Standard mode deals 17 cards per player, 3 bottom cards separate."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    for pos in ["landlord", "landlord_up", "landlord_down"]:
        assert len(env._env.info_sets[pos].player_hand_cards) == 17
    # Bottom cards are stored but not yet in any hand.
    assert len(env._env.three_landlord_cards) == 3
    assert len(env._env.bottom_cards_revealed) == 0


# --------------------------------------------------------------------------- #
# Illegal phase / action errors
# --------------------------------------------------------------------------- #
def test_bidding_in_wrong_phase_raises():
    """step_bidding must raise if not in BIDDING phase."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players, ruleset=RuleSet.standard())
    deck = _make_standard_deck()
    data = _deal_standard(deck)
    env.card_play_init_standard(data)
    # Manually force phase to PLAYING.
    env.phase = PHASE_PLAYING
    with pytest.raises(IllegalPhaseError, match="phase"):
        env.step_bidding(1)


def test_illegal_bid_value_raises():
    """A bid value not in ruleset.bid_values must raise."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players, ruleset=RuleSet.standard())
    deck = _make_standard_deck()
    data = _deal_standard(deck)
    env.card_play_init_standard(data)
    with pytest.raises(IllegalActionError, match="not in allowed"):
        env.step_bidding(4)
    with pytest.raises(IllegalActionError, match="not in allowed"):
        env.step_bidding(-1)


def test_env_step_bidding_requires_bid_value():
    """Env.step during BIDDING with bid_value=None must raise."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    with pytest.raises(IllegalActionError, match="bid_value"):
        env.step(None, bid_value=None)


# --------------------------------------------------------------------------- #
# Bidding flow: highest bidder becomes landlord
# --------------------------------------------------------------------------- #
def test_highest_bidder_becomes_landlord():
    """Bid [1, 2, 3] → the bidder of 3 is the landlord, bid_value=3."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)

    env.step(None, bid_value=1)  # first bidder
    assert env._env.phase == PHASE_BIDDING
    env.step(None, bid_value=2)  # second bidder
    assert env._env.phase == PHASE_BIDDING
    env.step(None, bid_value=3)  # third bidder → bidding complete

    # After bidding, phase transitions to PLAYING.
    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 3
    assert env._env.landlord_position == bidding_order[2]


def test_tie_break_by_bidding_order():
    """Bid [2, 0, 2] → first bidder of 2 is the landlord."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)

    env.step(None, bid_value=2)
    env.step(None, bid_value=0)
    env.step(None, bid_value=2)

    assert env._env.bid_value == 2
    assert env._env.landlord_position == bidding_order[0]


def test_all_pass_triggers_redeal():
    """Bid [0, 0, 0] with all_pass_redeal=True → done=True, info={'redeal': True}."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    obs, reward, done, info = env.step(None, bid_value=0)

    assert done is True
    assert info.get('redeal') is True
    assert env.need_redeal is True


def test_single_bidder_becomes_landlord():
    """Bid [0, 0, 3] → the third bidder is the landlord."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)

    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    env.step(None, bid_value=3)

    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 3
    assert env._env.landlord_position == bidding_order[2]


# --------------------------------------------------------------------------- #
# Bottom card reveal
# --------------------------------------------------------------------------- #
def test_bottom_cards_added_to_landlord_hand():
    """After bidding, the 3 bottom cards must be in the landlord's hand (20 total)."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    env.step(None, bid_value=1)
    env.step(None, bid_value=2)
    env.step(None, bid_value=3)

    landlord = env._env.landlord_position
    assert len(env._env.info_sets[landlord].player_hand_cards) == 20
    assert len(env._env.bottom_cards_revealed) == 3


def test_bottom_card_identity_preserved():
    """bottom_cards_revealed must record the exact 3 bottom card identities."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    expected_bottom = sorted(env._env.three_landlord_cards_initial)

    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)

    revealed = sorted(env._env.bottom_cards_revealed)
    assert revealed == expected_bottom


def test_bottom_cards_are_subset_of_landlord_hand():
    """The revealed bottom cards must be in the landlord's hand after reveal."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)

    landlord = env._env.landlord_position
    hand = Counter(env._env.info_sets[landlord].player_hand_cards)
    for card in env._env.bottom_cards_revealed:
        assert hand[card] >= 1


# --------------------------------------------------------------------------- #
# Bidding observation: no privileged info
# --------------------------------------------------------------------------- #
def test_bidding_obs_has_no_opponent_hands():
    """Bidding obs must not contain other players' hand cards."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    obs = env.reset()

    # The obs should only have the bidder's own hand, not opponents'.
    assert 'my_handcards' in obs
    # No field that looks like opponent hands.
    for key in obs:
        assert 'other' not in key.lower()
        assert 'opponent' not in key.lower()
        assert 'all_hand' not in key.lower()


def test_bidding_obs_has_no_bottom_cards():
    """Bidding obs must not reveal the bottom cards."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    obs = env.reset()

    for key in obs:
        assert 'bottom' not in key.lower()
        assert 'three_landlord' not in key.lower()
    # my_handcards should be exactly 17 cards (not 20).
    assert len(obs['my_handcards']) == 17


def test_bidding_obs_shows_public_history():
    """After a bid, the bidding history must be public in the next obs."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    env.step(None, bid_value=2)
    obs = env.bidding_obs if env.bidding_obs else env._env.get_bidding_obs()
    assert len(obs['bidding_history']) == 1
    assert obs['bidding_history'][0][1] == 2


# --------------------------------------------------------------------------- #
# Full standard game rollout
# --------------------------------------------------------------------------- #
def test_standard_full_game_terminates_and_produces_game_result():
    """A full standard game must terminate and produce a GameResult."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    # Bidding: all bid 1 → first bidder is landlord.
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)

    assert env._env.phase == PHASE_PLAYING

    # Play to terminal (greedy: first legal action).
    steps = 0
    done = False
    while not done:
        assert steps < 2000, "standard game did not terminate"
        action = env.infoset.legal_actions[0]
        obs, reward, done, info = env.step(action)
        steps += 1

    assert done
    assert env._env.phase == PHASE_TERMINAL
    assert env._env.game_result is not None
    result = env._env.game_result
    assert result.winner_team in ("landlord", "farmer")
    assert result.winner_position in ("landlord", "landlord_up", "landlord_down")
    # Score conservation.
    assert result.landlord_score + 2 * result.farmer_score == 0
    # info dict should contain the GameResult fields.
    assert 'winner_team' in info
    assert 'landlord_score' in info


def test_standard_full_game_card_conservation():
    """All 54 cards must be accounted for at every step."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    # Bidding.
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)

    def total_cards():
        t = 0
        for pos in ["landlord", "landlord_up", "landlord_down"]:
            t += len(env._env.info_sets[pos].player_hand_cards)
        for pos in ["landlord", "landlord_up", "landlord_down"]:
            t += len(env._env.played_cards[pos])
        return t

    # After reveal: 20 + 17 + 17 = 54.
    assert total_cards() == 54

    done = False
    steps = 0
    while not done:
        assert steps < 2000
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        assert total_cards() == 54
        steps += 1


def test_standard_reward_is_nonzero_at_terminal():
    """Terminal reward must be non-zero and from the landlord's perspective."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()

    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)

    done = False
    reward = 0.0
    steps = 0
    while not done:
        assert steps < 2000
        action = env.infoset.legal_actions[0]
        _, reward, done, _ = env.step(action)
        steps += 1

    assert reward != 0.0
    result = env._env.game_result
    if result.winner_team == "landlord":
        assert reward > 0
    else:
        assert reward < 0
    # Reward equals landlord_score.
    assert reward == float(result.landlord_score)


# --------------------------------------------------------------------------- #
# All-pass without redeal
# --------------------------------------------------------------------------- #
def test_all_pass_without_redeal_assigns_first_bidder():
    """If all_pass_redeal=False, all-pass assigns landlord to first bidder."""
    rs = RuleSet.from_dict({"ruleset_id": "standard", "all_pass_redeal": False})
    env = Env("adp", ruleset=rs)
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)

    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    obs, reward, done, info = env.step(None, bid_value=0)

    # No redeal; landlord is the first bidder with minimum bid.
    assert done is False
    assert env._env.phase == PHASE_PLAYING
    assert env._env.landlord_position == bidding_order[0]
    assert env._env.bid_value == 1
