"""Imperfect-information leakage guards (P00 preventive baseline).

AGENTS.md: "two states with identical public information but different hidden
allocations must produce identical public observations."

The legacy code keeps a privileged ``InfoSet`` containing ``all_handcards`` and
``other_hand_cards`` (the opponents' true hands). The *deployment* observation
returned by ``get_obs`` must NOT echo these privileged fields, and two deals
that differ only in how the two farmers' hidden hands are split -- while
keeping the landlord's hand and public action history identical -- must yield
the same public observation for the landlord.

This test class records that contract today so P03 (Observation V2) has a
concrete target to preserve and so any accidental leak is caught immediately.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from douzero.env.env import Env, get_obs
from douzero.env.game import GameEnv


PRIVILEGED_KEYS = {
    "all_handcards",
    "other_hand_cards",
}


def test_get_obs_output_has_no_privileged_keys(seed_factory):
    """The obs dict returned by get_obs must not carry privileged field names."""
    seed_factory(500)
    env = Env("adp")
    env.reset()
    obs = get_obs(env.infoset)
    # Public observation keys (frozen legacy contract).
    assert set(obs.keys()) == {
        "position",
        "x_batch",
        "z_batch",
        "legal_actions",
        "x_no_action",
        "z",
    }
    # None of the privileged InfoSet field names may leak into the obs dict.
    assert not (set(obs.keys()) & PRIVILEGED_KEYS)


def test_infoset_contains_privileged_fields_but_obs_does_not(seed_factory):
    """The InfoSet holds privileged data; get_obs must not expose it by name."""
    seed_factory(501)
    env = Env("adp")
    env.reset()
    infoset = env.infoset
    # Privileged fields DO exist on the infoset (training/teacher use).
    assert hasattr(infoset, "all_handcards")
    assert hasattr(infoset, "other_hand_cards")
    obs = get_obs(infoset)
    for key in PRIVILEGED_KEYS:
        assert key not in obs


def test_landlord_obs_invariant_under_farmer_hand_reallocation(fixed_card_play_data):
    """Same landlord hand + same public history -> identical landlord obs.

    We take the fixed deal, then SWAP a card between the two farmers. From the
    landlord's perspective the public information (its own hand, the bottom
    cards, action history -- empty here, both farmers' *counts*) is unchanged,
    so the landlord's public observation must be byte-identical.

    Note: legacy ``other_hand_cards`` (the union of opponent cards) is also
    invariant under a swap, which is why this property holds for the landlord
    role specifically. This is the strongest invariant the legacy encoder
    satisfies and the one P03 must preserve.
    """

    def build_landlord_obs(data):
        players = {pos: _NoopAgent() for pos in ["landlord", "landlord_up", "landlord_down"]}
        env = GameEnv(players)
        env.card_play_init(data)
        infoset = env.game_infoset
        # On card_play_init the acting player is the landlord (opening lead).
        assert infoset.player_position == "landlord"
        return get_obs(infoset)

    data_a = copy.deepcopy(fixed_card_play_data)
    data_b = copy.deepcopy(fixed_card_play_data)

    # Swap one card between the two farmers (keep counts identical).
    up_b = data_b["landlord_up"]
    down_b = data_b["landlord_down"]
    # Find a card present in both and swap distinct ranks.
    a_card = up_b[0]
    b_card = down_b[0]
    if a_card != b_card:
        up_b[0], down_b[0] = b_card, a_card
        data_b["landlord_up"] = sorted(up_b)
        data_b["landlord_down"] = sorted(down_b)

    obs_a = build_landlord_obs(data_a)
    obs_b = build_landlord_obs(data_b)

    # The landlord's own hand and the bottom cards are identical.
    assert data_a["landlord"] == data_b["landlord"]
    assert data_a["three_landlord_cards"] == data_b["three_landlord_cards"]

    # Public observation for the landlord must be identical.
    np.testing.assert_array_equal(obs_a["x_batch"], obs_b["x_batch"])
    np.testing.assert_array_equal(obs_a["z_batch"], obs_b["z_batch"])
    np.testing.assert_array_equal(obs_a["x_no_action"], obs_b["x_no_action"])
    # Legal actions depend only on the landlord's hand + rival move (empty) ->
    # identical.
    assert obs_a["legal_actions"] == obs_b["legal_actions"]


def test_other_hand_cards_is_union_of_opponents(seed_factory):
    """other_hand_cards (the privileged union) must equal the two opponents' cards.

    This documents what the legacy encoder actually does: it pools both hidden
    opponents into one set, which is exactly why the swap-invariant property
    above holds. P03 will split this into a belief model.
    """
    seed_factory(502)
    env = Env("adp")
    env.reset()
    infoset = env.infoset
    up = set(env._env.info_sets["landlord_up"].player_hand_cards)
    down = set(env._env.info_sets["landlord_down"].player_hand_cards)
    other = set(infoset.other_hand_cards)
    # The union of opponent multiset equals other_hand_cards multiset.
    from collections import Counter

    up_c = Counter(env._env.info_sets["landlord_up"].player_hand_cards)
    down_c = Counter(env._env.info_sets["landlord_down"].player_hand_cards)
    other_c = Counter(infoset.other_hand_cards)
    assert up_c + down_c == other_c


class _NoopAgent:
    """Minimal agent stub for driving GameEnv directly in leakage tests."""

    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def act(self, infoset):
        if self.action is not None and self.action in infoset.legal_actions:
            a, self.action = self.action, None
            return a
        return infoset.legal_actions[0]
