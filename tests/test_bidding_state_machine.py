"""Tests for the bidding state machine and standard-mode GameEnv (P02 Slice 3).

Covers:
- legacy path is byte-for-byte unchanged (GameEnv without ruleset)
- standard DEAL→BIDDING→REVEAL_BOTTOM→PLAYING→TERMINAL phase transitions
- neutral seat labels ("0", "1", "2") during BIDDING
- role remapping after landlord determined (all 3 seats tested as landlord)
- illegal phase / illegal action errors
- bid 3 (max) ends bidding immediately
- post-terminal step raises IllegalPhaseError
- three-player bidding: highest bidder becomes landlord
- tie-breaking by bidding order
- all-pass redeal
- max_redeals guard
- bottom card reveal (3 cards added to landlord hand)
- bottom card entity identity preservation
- bidding observation contains no privileged info
- get_legal_bids interface
- full standard game rollout terminates and produces GameResult
- score conservation at terminal
- reset clears all bidding/spring/result fields
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
    for _ in range(5):
        env.step()


def test_legacy_env_constructor_unchanged():
    """Env(objective) without ruleset must work as before."""
    env = Env("adp")
    assert env.ruleset is None
    assert env.bidding_obs is None
    np.random.seed(42)
    obs = env.reset()
    assert 'x_batch' in obs
    assert 'z_batch' in obs


# --------------------------------------------------------------------------- #
# Standard mode phase transitions + neutral seats
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


def test_standard_uses_neutral_seats_during_bidding():
    """Bidding phase must use neutral seat labels "0", "1", "2"."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    obs = env.reset()
    assert env._env.bidding_order == ["0", "1", "2"]
    assert obs['position'] in ("0", "1", "2")
    # No role labels during bidding.
    assert "landlord" not in env._env.bidding_order


def test_standard_initial_deal_gives_17_cards_each():
    """Standard mode deals 17 cards per player, 3 bottom cards separate."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    for seat in ["0", "1", "2"]:
        assert len(env._env._seat_infosets[seat].player_hand_cards) == 17
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


def test_post_terminal_step_raises_illegal_phase():
    """Stepping after the game is over must raise IllegalPhaseError."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    done = False
    steps = 0
    while not done:
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        steps += 1
        assert steps < 2000
    assert env._env.phase == PHASE_TERMINAL
    with pytest.raises(IllegalPhaseError, match="game is over"):
        env.step(env.infoset.legal_actions[0] if env.infoset else [3])


# --------------------------------------------------------------------------- #
# Bid 3 ends bidding immediately
# --------------------------------------------------------------------------- #
def test_bid_3_ends_bidding_immediately():
    """A bid of 3 (the maximum) must end bidding immediately."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)
    env.step(None, bid_value=3)
    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 3
    assert env._env.landlord_position == bidding_order[0]
    # Only 1 bid in history.
    assert len(env._env.bidding_history) == 1


def test_bid_3_by_second_seat_ends_bidding():
    """Bid 3 by the second seat also ends bidding immediately."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)
    env.step(None, bid_value=0)  # seat 0 passes
    env.step(None, bid_value=3)  # seat 1 bids 3 → ends
    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 3
    assert env._env.landlord_position == bidding_order[1]
    assert len(env._env.bidding_history) == 2


# --------------------------------------------------------------------------- #
# Bidding flow: highest bidder becomes landlord
# --------------------------------------------------------------------------- #
def test_highest_bidder_becomes_landlord():
    """Bid [1, 2, 3] → the bidder of 3 is the landlord, bid_value=3.

    Note: bid 3 ends immediately, so only [1, 2, 3] with 3 as last works.
    Actually bid 3 by seat 2 ends bidding. Let's test [1, 2, 0] → seat 1 wins.
    """
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)
    env.step(None, bid_value=1)
    env.step(None, bid_value=2)
    env.step(None, bid_value=0)
    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 2
    assert env._env.landlord_position == bidding_order[1]


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
    """Bid [0, 0, 1] → the third bidder is the landlord."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    bidding_order = list(env._env.bidding_order)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    env.step(None, bid_value=1)
    assert env._env.phase == PHASE_PLAYING
    assert env._env.bid_value == 1
    assert env._env.landlord_position == bidding_order[2]


# --------------------------------------------------------------------------- #
# All-pass redeal with max_redeals guard
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
    assert done is False
    assert env._env.phase == PHASE_PLAYING
    assert env._env.landlord_position == bidding_order[0]
    assert env._env.bid_value == 1


def test_max_redeals_guard_forces_landlord_assignment():
    """After max_redeals all-pass redeals, the landlord is force-assigned.

    The max_redeals guard is in the Env wrapper: each all-pass redeal
    increments _redeal_count. When it exceeds max_redeals, the landlord is
    force-assigned to the first bidder. We simulate this by forcing all-pass
    and calling redeal() (which preserves _redeal_count).
    """
    rs = RuleSet.from_dict({"ruleset_id": "standard", "max_redeals": 2})
    env = Env("adp", ruleset=rs)
    np.random.seed(42)
    env.reset()
    # Force all-pass redeals until max_redeals is exceeded.
    for _attempt in range(rs.max_redeals + 1):
        # All three bidders pass.
        env.step(None, bid_value=0)
        env.step(None, bid_value=0)
        obs, reward, done, info = env.step(None, bid_value=0)
        if not info.get('redeal'):
            # max_redeals exceeded → landlord force-assigned, phase=PLAYING.
            assert env._env.phase == PHASE_PLAYING
            assert env._env.landlord_position is not None
            assert info.get('max_redeals_exceeded') is True
            return
        # Redeal for the next attempt (preserves _redeal_count).
        env.redeal()
    pytest.fail("max_redeals guard did not trigger")


# --------------------------------------------------------------------------- #
# Neutral seat → role remapping (all 3 seats as landlord)
# --------------------------------------------------------------------------- #
def _bidding_to_make_seat_landlord(env, target_seat):
    """Bid so that the target seat becomes the landlord.

    bidding_order is ["0", "1", "2"]. We bid 0 for non-target seats and
    1 for the target seat. If the target is not the last bidder, a later
    bidder could overbid, so we bid 0 for all seats after the target.
    """
    order = env._env.bidding_order
    for seat in order:
        if seat == target_seat:
            env.step(None, bid_value=1)
        else:
            env.step(None, bid_value=0)


@pytest.mark.parametrize("target_seat", ["0", "1", "2"])
def test_all_seats_can_become_landlord(target_seat):
    """Each of the 3 neutral seats can become the landlord.

    After role remapping:
    - 'landlord' role has 20 cards (17 + 3 bottom)
    - 'landlord_down' and 'landlord_up' roles have 17 cards each
    - The landlord acts first
    - Total cards = 54
    """
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    _bidding_to_make_seat_landlord(env, target_seat)

    assert env._env.phase == PHASE_PLAYING
    assert env._env.landlord_position == target_seat
    # Role remap: 'landlord' key has 20 cards.
    assert len(env._env.info_sets['landlord'].player_hand_cards) == 20
    assert len(env._env.info_sets['landlord_up'].player_hand_cards) == 17
    assert len(env._env.info_sets['landlord_down'].player_hand_cards) == 17
    # Landlord acts first.
    assert env._env.acting_player_position == 'landlord'
    # Total = 54.
    total = sum(len(env._env.info_sets[p].player_hand_cards)
                for p in ['landlord', 'landlord_up', 'landlord_down'])
    assert total == 54


@pytest.mark.parametrize("target_seat", ["0", "1", "2"])
def test_turn_order_correct_for_each_landlord(target_seat):
    """After role remap, turn order is landlord → landlord_down → landlord_up.

    The legacy get_acting_player_position cycles landlord→down→up→landlord.
    After remap, the 'landlord' key holds the actual landlord's data, so
    this cycle is correct regardless of which seat won the bid.
    """
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    _bidding_to_make_seat_landlord(env, target_seat)

    seen = [env._env.acting_player_position]
    for _ in range(3):
        action = env.infoset.legal_actions[0]
        env.step(action)
        seen.append(env._env.acting_player_position)
    assert seen == ['landlord', 'landlord_down', 'landlord_up', 'landlord']


@pytest.mark.parametrize("target_seat", ["0", "1", "2"])
def test_full_game_works_for_each_landlord(target_seat):
    """A full standard game must terminate for each possible landlord seat."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    _bidding_to_make_seat_landlord(env, target_seat)

    done = False
    steps = 0
    while not done:
        assert steps < 2000, f"game did not terminate for seat {target_seat}"
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        steps += 1
    assert env._env.game_result is not None
    assert env._env.game_result.landlord_score + 2 * env._env.game_result.farmer_score == 0


# --------------------------------------------------------------------------- #
# Bottom card reveal
# --------------------------------------------------------------------------- #
def test_bottom_cards_added_to_landlord_hand():
    """After bidding, the 3 bottom cards must be in the landlord's hand (20 total)."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    assert len(env._env.info_sets['landlord'].player_hand_cards) == 20
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
    hand = Counter(env._env.info_sets['landlord'].player_hand_cards)
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
    assert 'my_handcards' in obs
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
# get_legal_bids interface
# --------------------------------------------------------------------------- #
def test_get_legal_bids_returns_bid_values():
    """GameEnv.get_legal_bids must return the ruleset's bid_values."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players, ruleset=RuleSet.standard())
    deck = _make_standard_deck()
    data = _deal_standard(deck)
    env.card_play_init_standard(data)
    bids = env.get_legal_bids()
    assert bids == [0, 1, 2, 3]


def test_get_legal_bids_raises_outside_bidding():
    """get_legal_bids must raise if not in BIDDING phase."""
    players = {p: _StubAgent() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players, ruleset=RuleSet.standard())
    env.phase = PHASE_PLAYING
    with pytest.raises(IllegalPhaseError):
        env.get_legal_bids()


# --------------------------------------------------------------------------- #
# Full standard game rollout
# --------------------------------------------------------------------------- #
def test_standard_full_game_terminates_and_produces_game_result():
    """A full standard game must terminate and produce a GameResult."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    assert env._env.phase == PHASE_PLAYING
    steps = 0
    done = False
    while not done:
        assert steps < 2000
        action = env.infoset.legal_actions[0]
        obs, reward, done, info = env.step(action)
        steps += 1
    assert done
    assert env._env.phase == PHASE_TERMINAL
    assert env._env.game_result is not None
    result = env._env.game_result
    assert result.winner_team in ("landlord", "farmer")
    assert result.winner_position in ("landlord", "landlord_up", "landlord_down")
    assert result.landlord_score + 2 * result.farmer_score == 0
    assert 'winner_team' in info
    assert 'landlord_score' in info
    # Rule identity stamped.
    assert info['ruleset_id'] == 'standard'
    assert 'ruleset_hash' in info


def test_standard_full_game_card_conservation():
    """All 54 cards must be accounted for at every step."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
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
    assert reward == float(result.landlord_score)


# --------------------------------------------------------------------------- #
# reset clears all fields
# --------------------------------------------------------------------------- #
def test_reset_clears_all_bidding_fields():
    """reset() must clear all bidding, spring, and result fields."""
    env = Env("adp", ruleset=RuleSet.standard())
    np.random.seed(42)
    env.reset()
    env.step(None, bid_value=1)
    env.step(None, bid_value=0)
    env.step(None, bid_value=0)
    # Play to terminal.
    done = False
    steps = 0
    while not done:
        action = env.infoset.legal_actions[0]
        _, _, done, _ = env.step(action)
        steps += 1
        assert steps < 2000
    assert env._env.game_result is not None
    assert env._env.bomb_count >= 0

    # Reset.
    env.reset()
    assert env._env.phase == PHASE_BIDDING
    assert env._env.bidding_history == []
    assert env._env.landlord_position is None
    assert env._env.bid_value == 0
    assert env._env.bomb_count == 0
    assert env._env.rocket_count == 0
    assert env._env.game_result is None
    assert env._env.bottom_cards_revealed == []
    assert env._env.action_counts == {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
    assert env._redeal_count == 0
    assert env.need_redeal is False
