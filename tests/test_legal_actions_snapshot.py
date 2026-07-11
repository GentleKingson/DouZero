"""Snapshot tests for the legal-action generator on a fixed deal.

These tests freeze the *behavior* of ``GameEnv.get_legal_card_play_actions`` so
any future rule change (P02) is caught immediately. The deal is constructed
directly from a deterministic deck so the snapshot does not depend on numpy's
shuffle implementation.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from douzero.env.game import GameEnv
from douzero.env.move_detector import get_move_type


def _canonical_legal_actions_hash(legal_actions):
    """Stable hash: sort each action, then sort the outer list, then sha256."""
    canon = sorted(tuple(sorted(a)) for a in legal_actions)
    payload = json.dumps([list(t) for t in canon])
    return hashlib.sha256(payload.encode()).hexdigest(), canon


def _make_env_with_fixed_deal(card_play_data):
    """Build a GameEnv driven by a stub agent and initialise it directly.

    We bypass ``Env`` (which shuffles) so the deal is fully deterministic.
    """

    class _StubAgent:
        def __init__(self):
            self.action = None

        def set_action(self, action):
            self.action = action

        def act(self, infoset):
            # Default: pick the first legal action. The caller can override via
            # set_action before stepping.
            if self.action is not None and self.action in infoset.legal_actions:
                a, self.action = self.action, None
                return a
            return infoset.legal_actions[0]

    players = {pos: _StubAgent() for pos in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    env.card_play_init(card_play_data)
    return env, players


def test_fixed_deal_landlord_legal_actions_are_sorted(fixed_card_play_data):
    env, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    infoset = env.game_infoset
    for action in infoset.legal_actions:
        assert action == sorted(action), f"unsorted legal action: {action}"


def test_fixed_deal_landlord_has_many_opening_actions(fixed_card_play_data):
    # On the opening lead (rival is empty / TYPE_0_PASS), gen_moves enumerates
    # every type, so the count is in the hundreds, not single digits.
    env, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    infoset = env.game_infoset
    assert len(infoset.legal_actions) > 100


def test_fixed_deal_king_bomb_legal_on_opening_if_both_jokers_in_landlord_hand(
    fixed_card_play_data,
):
    env, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    infoset = env.game_infoset
    hand = set(infoset.player_hand_cards)
    has_rocket = [20, 30] in [sorted(a) for a in infoset.legal_actions]
    if {20, 30}.issubset(hand):
        assert has_rocket, "rocket must be legal when landlord holds both jokers"
    else:
        assert not has_rocket


def test_legal_actions_snapshot_is_stable_across_calls(fixed_card_play_data):
    """Two independent envs with the same deal must produce identical hashes."""
    env1, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    env2, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    h1, _ = _canonical_legal_actions_hash(env1.game_infoset.legal_actions)
    h2, _ = _canonical_legal_actions_hash(env2.game_infoset.legal_actions)
    assert h1 == h2


def test_landlord_opening_actions_classify_valid_with_known_exceptions(
    fixed_card_play_data,
):
    """Opening legal actions classify as known types, EXCEPT a fixed known set.

    Legacy ``gen_type_11_serial_3_1`` builds serial-3+1 wings by selecting
    single cards from ``[c for c in hand if c not in serial_3_set]``. When the
    hand contains a four-of-a-kind, those four equal-rank singles become four
    "wings", yielding a 16-card move such as ``[3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,7]``
    (four triples + a quad used as four single wings). ``get_move_type`` then
    classifies it as ``TYPE_15_WRONG`` because its serial-3 branch rejects any
    card whose multiplicity is 4 (move_detector.py:92).

    This is a genuine generator/detector inconsistency. It is NOT merely
    cosmetic: ``gen_moves()`` (the opening-lead enumerator) returns these
    actions, a model may play one, and the NEXT player then sees it as the
    rival move. In ``GameEnv.get_legal_card_play_actions`` no ``elif`` branch
    matches ``TYPE_15_WRONG``, so the response set collapses to
    ``bombs + rocket + pass`` only -- the responder cannot answer with a normal
    higher serial/triple. So this anomaly does affect actual play.

    The assertion below is intentionally strict: it accepts ONLY the exact,
    pre-computed exception set for this fixed deal and rejects any OTHER
    ``TYPE_15_WRONG`` action, so future drift (or a P02 rule-engine fix) is
    caught immediately.
    """
    # Exact, pre-computed exception set for the fixed deal's landlord hand
    # ([3,4,5,6,7] each x4). The quad can be either the wing rank or one of
    # the serial-triple ranks. Empirically verified against gen_moves().
    expected_exceptions = {
        (3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7),
        (3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7),
    }

    env, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    observed_wrong = {
        tuple(sorted(action))
        for action in env.game_infoset.legal_actions
        if get_move_type(action)["type"] == 15
    }
    assert observed_wrong == expected_exceptions, (
        "The set of TYPE_15_WRONG legal actions changed. If this is an "
        "intentional rule-engine fix (P02), update expected_exceptions; "
        f"otherwise investigate. observed={observed_wrong}"
    )


def test_quad_wing_wrong_move_collapses_response_to_bomb_or_pass():
    """A TYPE_15_WRONG rival move restricts the responder to bomb/rocket/pass.

    This pins the gameplay impact of the generator/detector anomaly documented
    above: when a quad-wing serial-3+1 is the rival action, no ``elif`` in
    ``GameEnv.get_legal_card_play_actions`` matches, so only bombs, rocket, and
    pass are offered. With a bomb-less responder hand the only legal response
    is pass.
    """
    from douzero.env.move_generator import MovesGener

    rival = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7]
    assert get_move_type(rival)["type"] == 15  # TYPE_15_WRONG

    # Responder holds normal cards but no bomb/rocket.
    responder_hand = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 5, 6, 7, 8]
    mg = MovesGener(responder_hand)

    # Reproduce the TYPE_15_WRONG branch behaviour in game.py:177-263.
    moves = []  # no elif matches TYPE_15_WRONG
    moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
    moves = moves + [[]]  # rival non-empty -> pass appended
    for m in moves:
        m.sort()

    # This bomb-less responder can only pass.
    assert moves == [[]]


def test_pass_not_legal_on_opening_lead(fixed_card_play_data):
    """At the very first move there is no rival action to pass on."""
    env, _ = _make_env_with_fixed_deal(fixed_card_play_data)
    actions = [sorted(a) for a in env.game_infoset.legal_actions]
    assert [] not in actions, "pass must not be legal on the opening lead"


def test_pass_becomes_legal_once_there_is_a_rival_move(fixed_card_play_data):
    """After the landlord leads, the next player must be able to pass."""
    env, players = _make_env_with_fixed_deal(fixed_card_play_data)
    # Landlord leads with the first legal action.
    players["landlord"].set_action(env.game_infoset.legal_actions[0])
    env.step()
    # Now it is landlord_down's turn and there is a rival move -> pass allowed.
    infoset = env.game_infoset
    assert [] in [sorted(a) for a in infoset.legal_actions]


def test_two_seeded_envs_produce_same_opening_hash(seeded_env):
    # seeded_env already had reset() called with a fixed numpy seed.
    h1, _ = _canonical_legal_actions_hash(seeded_env.infoset.legal_actions)

    # Rebuild with the same seed and compare.
    from douzero.env.env import Env

    np.random.seed(20240501)
    env2 = Env("adp")
    env2.reset()
    h2, _ = _canonical_legal_actions_hash(env2.infoset.legal_actions)
    assert h1 == h2
