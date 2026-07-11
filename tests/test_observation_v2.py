"""Observation V2 schema, encoding, and imperfect-information boundary (P03).

Covers the P03 acceptance criteria:

- legacy snapshots still pass (the legacy encoder is unchanged);
- PublicObservation serialisation/deserialisation is stable;
- two states with identical public information but different hidden allocations
  produce IDENTICAL public observations (the leakage invariant);
- PrivilegedObservation changes when the hidden allocation changes, and the
  public encoder's return type cannot be a PrivilegedObservation;
- all field shapes are derived from the schema (no magic 319/373/430/484);
- card conservation holds: my + unseen + played + bottom_unplayed == deck.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from douzero.env.env import Env, _cards2array, get_obs
from douzero.env.game import GameEnv
from douzero.env.rules import RuleSet
from douzero.observation import (
    BIG_JOKER,
    CARD_VECTOR_DIM,
    DECK,
    FEATURE_VERSION_V2,
    PRIVILEGED_KIND,
    PUBLIC_KIND,
    SMALL_JOKER,
    FeatureSchemaManifest,
    HistoryMove,
    ObservationV2,
    PrivilegedObservation,
    PublicObservation,
    SEAT_LANDLORD,
    SEAT_NEXT,
    SEAT_PREVIOUS,
    SEAT_SELF,
    SEAT_TEAMMATE,
    build_public_observation,
    build_v2_schema,
    cards_to_vector,
    compute_unseen_pool,
    encode_history,
    get_obs_v2,
    is_privileged,
    legacy_observation_from_v2,
    relative_seat,
    seats_from,
)


# --------------------------------------------------------------------------- #
# Cards encoding
# --------------------------------------------------------------------------- #
class TestCardsEncoding:
    def test_vector_dim_is_54(self):
        assert CARD_VECTOR_DIM == 54

    def test_deck_has_54_cards(self):
        assert len(DECK) == 54

    def test_empty_hand_is_zero_vector(self):
        assert cards_to_vector([]).shape == (54,)
        assert cards_to_vector([]).sum() == 0

    def test_parity_with_legacy_encoder(self):
        """cards_to_vector must reproduce _cards2array exactly."""
        samples = [
            [],
            [3], [3, 3], [3, 3, 3], [3, 3, 3, 3],
            [20], [30], [20, 30],
            [3, 4, 5, 6, 7],
            [17, 17, 17, 17, 14, 14],
            [3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
            [10, 10, 10, 11, 11, 11, 12, 12],
        ]
        for cards in samples:
            a = cards_to_vector(cards)
            b = _cards2array(cards)
            np.testing.assert_array_equal(a, b, err_msg=f"mismatch for {cards}")

    def test_joker_offsets(self):
        vec = cards_to_vector([SMALL_JOKER, BIG_JOKER])
        from douzero.observation.cards import BIG_JOKER_OFFSET, SMALL_JOKER_OFFSET
        assert vec[SMALL_JOKER_OFFSET] == 1
        assert vec[BIG_JOKER_OFFSET] == 1

    def test_rejects_duplicate_joker(self):
        with pytest.raises(ValueError):
            cards_to_vector([SMALL_JOKER, SMALL_JOKER])

    def test_full_deck_encodes_to_full_vector(self):
        """Encoding the entire deck sets every slot to its max multiplicity."""
        vec = cards_to_vector(DECK)
        # 13 numeric ranks × 4 = 52 ones + 2 joker slots = 54 ones.
        assert vec.sum() == 54


# --------------------------------------------------------------------------- #
# Relative seats
# --------------------------------------------------------------------------- #
class TestRelativeSeats:
    def test_landlord_sees_next_and_previous_farmers(self):
        seats = seats_from("landlord")
        assert seats["landlord"] == SEAT_SELF
        assert seats["landlord_down"] == SEAT_NEXT
        assert seats["landlord_up"] == SEAT_PREVIOUS

    def test_farmer_sees_landlord_and_teammate(self):
        seats = seats_from("landlord_up")
        assert seats["landlord_up"] == SEAT_SELF
        assert seats["landlord"] == SEAT_LANDLORD
        assert seats["landlord_down"] == SEAT_TEAMMATE

    def test_landlord_down_teammate_is_up(self):
        seats = seats_from("landlord_down")
        assert seats["landlord_up"] == SEAT_TEAMMATE

    def test_next_previous_are_inverses(self):
        for role in ("landlord", "landlord_up", "landlord_down"):
            from douzero.observation.seats import next_seat, previous_seat
            assert next_seat(previous_seat(role)) == role

    def test_relative_seat_self(self):
        assert relative_seat("landlord", "landlord") == SEAT_SELF


# --------------------------------------------------------------------------- #
# Schema / shapes (no magic numbers)
# --------------------------------------------------------------------------- #
class TestSchema:
    def test_v2_schema_version(self):
        s = build_v2_schema()
        assert s.feature_version == FEATURE_VERSION_V2
        assert s.schema_version == "v2-1"

    def test_state_width_derived_from_constants(self):
        """The flat state width must equal the sum of schema field shapes."""
        s = build_v2_schema()
        expected = sum(np.prod(spec.shape) for spec in s.state_fields)
        # Sanity: build a real obs and compare.
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.state.to_vector().shape == (int(expected),)

    def test_action_width_matches_features(self):
        s = build_v2_schema()
        expected = sum(np.prod(spec.shape) for spec in s.action_fields)
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.actions.features.shape[1] == int(expected)

    def test_history_token_width_matches_tensor(self):
        s = build_v2_schema()
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        # token width = sum of all history_token_fields shapes (incl. valid mask)
        expected = sum(np.prod(spec.shape) for spec in s.history_token_fields)
        assert obs.history.tokens.shape[1] == int(expected)

    def test_no_magic_widths_in_schema(self):
        """Schema field shapes must never be the legacy magic widths."""
        s = build_v2_schema()
        magic = {319, 373, 430, 484}
        for spec in (*s.state_fields, *s.action_fields, *s.history_token_fields):
            for d in spec.shape:
                assert d not in magic

    def test_max_history_len_configurable(self):
        s = build_v2_schema(max_history_len=20)
        assert s.max_history_len == 20
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset, schema=s)
        assert obs.history.tokens.shape[0] == 20

    def test_manifest_is_serializable(self):
        s = build_v2_schema()
        d = s.to_dict()
        # Round-trips through JSON.
        j = json.dumps(d)
        s2_dict = json.loads(j)
        assert s2_dict["schema_version"] == s.schema_version
        assert len(s2_dict["state_fields"]) == len(s.state_fields)

    def test_manifest_deterministic(self):
        """Two schemas with the same config are identical."""
        assert build_v2_schema().to_dict() == build_v2_schema().to_dict()


# --------------------------------------------------------------------------- #
# History tokens
# --------------------------------------------------------------------------- #
class TestHistory:
    def test_history_padding_mask(self):
        """Padding slots have a zero mask; real slots have a one mask."""
        s = build_v2_schema(max_history_len=10)
        moves = [
            HistoryMove("landlord", (3,), False, 1, 3, 1, 1, 19, False, 1),
            HistoryMove("landlord_down", (), True, 0, 0, 0, 0, 17, False, 1),
        ]
        batch = encode_history(moves, s)
        assert batch.num_real == 2
        assert batch.mask[:2].tolist() == [1, 1]
        assert batch.mask[2:].tolist() == [0] * 8
        # Padded tokens are all-zero.
        assert batch.tokens[2:].sum() == 0

    def test_history_drops_old_tokens_beyond_cap(self):
        s = build_v2_schema(max_history_len=3)
        moves = [
            HistoryMove("landlord", (r,), False, 1, r, 1, 1, 19, False, 1)
            for r in (3, 4, 5, 6, 7)
        ]
        batch = encode_history(moves, s)
        assert batch.num_real == 3  # only last 3 kept


# --------------------------------------------------------------------------- #
# Imperfect-information boundary (the core P03 safety test)
# --------------------------------------------------------------------------- #
class TestLeakageBoundary:
    def _build_landlord_infoset(self, data):
        players = {pos: _NoopAgent() for pos in
                   ["landlord", "landlord_up", "landlord_down"]}
        env = GameEnv(players)
        env.card_play_init(data)
        return env.game_infoset

    def test_public_obs_kind_is_public(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.public.kind == PUBLIC_KIND
        assert not is_privileged(obs)
        assert not is_privileged(obs.public)

    def test_observation_v2_is_not_privileged(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert not isinstance(obs, PrivilegedObservation)
        assert obs.is_privileged is False

    def test_public_obs_invariant_under_hidden_reallocation(self, fixed_card_play_data):
        """Same landlord hand + same public history -> identical public obs.

        Swap a card between the two farmers. From the landlord's perspective the
        public information is unchanged, so the encoded public observation must
        be byte-identical. This is the strongest imperfect-information invariant
        and the headline P03 test.
        """
        data_a = copy.deepcopy(fixed_card_play_data)
        data_b = copy.deepcopy(fixed_card_play_data)

        up_b = data_b["landlord_up"]
        down_b = data_b["landlord_down"]
        if up_b[0] != down_b[0]:
            up_b[0], down_b[0] = down_b[0], up_b[0]
            data_b["landlord_up"] = sorted(up_b)
            data_b["landlord_down"] = sorted(down_b)

        obs_a = get_obs_v2(self._build_landlord_infoset(data_a))
        obs_b = get_obs_v2(self._build_landlord_infoset(data_b))

        # Full encoded state + actions + history must match byte-for-byte.
        np.testing.assert_array_equal(
            obs_a.state.to_vector(), obs_b.state.to_vector())
        np.testing.assert_array_equal(
            obs_a.actions.features, obs_b.actions.features)
        np.testing.assert_array_equal(
            obs_a.history.tokens, obs_b.history.tokens)
        # The public dict serialisation must match too.
        assert obs_a.public.to_dict() == obs_b.public.to_dict()

    def test_public_feature_hash_stable_under_reallocation(self, fixed_card_play_data):
        """A content hash of the public features is identical under a swap."""
        data_a = copy.deepcopy(fixed_card_play_data)
        data_b = copy.deepcopy(fixed_card_play_data)
        up_b = data_b["landlord_up"]
        down_b = data_b["landlord_down"]
        if up_b[0] != down_b[0]:
            up_b[0], down_b[0] = down_b[0], up_b[0]
            data_b["landlord_up"] = sorted(up_b)
            data_b["landlord_down"] = sorted(down_b)

        def public_hash(data):
            obs = get_obs_v2(self._build_landlord_infoset(data))
            payload = (
                obs.state.to_vector().tobytes()
                + obs.actions.features.tobytes()
                + obs.history.tokens.tobytes()
            )
            return hashlib.sha256(payload).hexdigest()

        assert public_hash(data_a) == public_hash(data_b)

    def test_privileged_observation_changes_under_reallocation(self, fixed_card_play_data):
        """PrivilegedObservation must DIFFER when the hidden allocation differs."""
        data_a = copy.deepcopy(fixed_card_play_data)
        data_b = copy.deepcopy(fixed_card_play_data)
        up_b = data_b["landlord_up"]
        down_b = data_b["landlord_down"]
        if up_b[0] != down_b[0]:
            up_b[0], down_b[0] = down_b[0], up_b[0]
            data_b["landlord_up"] = sorted(up_b)
            data_b["landlord_down"] = sorted(down_b)

        priv_a = PrivilegedObservation(
            all_handcards={
                "landlord": tuple(sorted(data_a["landlord"])),
                "landlord_up": tuple(sorted(data_a["landlord_up"])),
                "landlord_down": tuple(sorted(data_a["landlord_down"])),
            },
            acting_role="landlord",
        )
        priv_b = PrivilegedObservation(
            all_handcards={
                "landlord": tuple(sorted(data_b["landlord"])),
                "landlord_up": tuple(sorted(data_b["landlord_up"])),
                "landlord_down": tuple(sorted(data_b["landlord_down"])),
            },
            acting_role="landlord",
        )
        assert priv_a.all_handcards != priv_b.all_handcards
        assert priv_a.to_dict() != priv_b.to_dict()

    def test_privileged_kind_marker(self):
        priv = PrivilegedObservation(
            all_handcards={"landlord": (3,), "landlord_up": (), "landlord_down": ()},
            acting_role="landlord",
        )
        assert priv.kind == PRIVILEGED_KIND
        assert is_privileged(priv)
        assert is_privileged(priv.to_dict())

    def test_public_encoder_does_not_read_all_handcards(self, fixed_card_play_data):
        """get_obs_v2 must produce the same output even if all_handcards is wrong.

        We corrupt the privileged all_handcards field on the infoset; the public
        encoder ignores it (it recomputes the unseen pool from public info), so
        the encoded observation is unchanged.
        """
        infoset = self._build_landlord_infoset(fixed_card_play_data)
        obs_clean = get_obs_v2(infoset)

        # Corrupt the privileged field.
        infoset.all_handcards = {"landlord": [3], "landlord_up": [3], "landlord_down": [3]}
        infoset.other_hand_cards = [3, 3, 3]
        obs_corrupt = get_obs_v2(infoset)

        np.testing.assert_array_equal(
            obs_clean.state.to_vector(), obs_corrupt.state.to_vector())


class _NoopAgent:
    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def act(self, infoset):
        if self.action is not None and self.action in infoset.legal_actions:
            a, self.action = self.action, None
            return a
        return infoset.legal_actions[0]


# --------------------------------------------------------------------------- #
# Card conservation invariants (property-style)
# --------------------------------------------------------------------------- #
class TestCardConservation:
    @pytest.mark.parametrize("seed", range(20))
    def test_legacy_deck_partition(self, seed):
        """my + unseen + played == deck (no negatives), for each acting role.

        In legacy mode the bottom cards are part of the landlord's hand, so they
        are already accounted for in ``my_handcards`` (for the landlord) or in
        the unseen pool (for a farmer). The conservation identity is therefore
        ``my + unseen + played == deck`` — the bottom must NOT be added again.
        """
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
        obs = get_obs_v2(env.infoset)
        total = Counter(obs.public.my_handcards)
        total += Counter(obs.public.other_handcards)
        for role in ("landlord", "landlord_up", "landlord_down"):
            total += Counter(obs.public.played_cards.get(role, ()))
        assert total == Counter(DECK)
        assert all(v >= 0 for v in total.values())

    def test_unseen_pool_matches_legacy_other_hand_cards(self):
        """The V2 unseen pool must equal the legacy other_hand_cards union."""
        np.random.seed(42)
        env = Env("adp")
        env.reset()
        obs = get_obs_v2(env.infoset)
        legacy_other = Counter(env.infoset.other_hand_cards)
        v2_pool = Counter(obs.public.other_handcards)
        assert v2_pool == legacy_other

    def test_belief_pool_excludes_bottom_for_farmer(self):
        """The belief unknown pool removes public bottom cards (farmer only)."""
        my_hand = [3, 4, 5]
        played = {"landlord": [], "landlord_up": [], "landlord_down": []}
        bottom = [20, 30]
        from douzero.observation import compute_belief_unknown_pool
        farmer_pool = compute_belief_unknown_pool(
            my_hand, played, bottom, acting_role="landlord_up")
        # The two joker bottom cards are removed for a farmer.
        assert SMALL_JOKER not in farmer_pool
        assert BIG_JOKER not in farmer_pool

    def test_unseen_pool_violation_raises(self):
        """A card appearing more times than in the deck must raise."""
        with pytest.raises(ValueError):
            compute_unseen_pool([3, 3, 3, 3, 3], {}, [])  # five 3s > 4 in deck


# --------------------------------------------------------------------------- #
# Serialisation stability
# --------------------------------------------------------------------------- #
class TestSerialisation:
    def test_public_observation_round_trips_json(self):
        env = Env("adp")
        np.random.seed(7)
        env.reset()
        obs = get_obs_v2(env.infoset)
        d = obs.public.to_dict()
        j = json.dumps(d)
        d2 = json.loads(j)
        assert d == d2

    def test_public_observation_stable_across_calls(self):
        """Encoding the same infoset twice yields identical public dicts."""
        env = Env("adp")
        np.random.seed(11)
        env.reset()
        a = get_obs_v2(env.infoset).public.to_dict()
        b = get_obs_v2(env.infoset).public.to_dict()
        assert a == b


# --------------------------------------------------------------------------- #
# State encoded once per decision
# --------------------------------------------------------------------------- #
class TestEncodeOnce:
    def test_state_vector_has_no_batch_dim(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        # The state vector is 1-D (no per-action batch dim).
        assert obs.state.to_vector().ndim == 1

    def test_legal_action_batch_has_one_row_per_action(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        n = len(env.infoset.legal_actions)
        assert obs.actions.features.shape[0] == n
        assert obs.actions.action_mask.shape == (n,)


# --------------------------------------------------------------------------- #
# Standard-mode (bidding + public bottom cards)
# --------------------------------------------------------------------------- #
class TestStandardMode:
    def _play_to_playing(self, seed=1):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(seed)
        env.reset()
        env.step(None, bid_value=1)
        env.step(None, bid_value=2)
        env.step(None, bid_value=3)  # max bid ends bidding
        return env

    def test_bottom_cards_owned_by_landlord(self):
        env = self._play_to_playing()
        obs = get_obs_v2(
            env.infoset,
            ruleset_id="standard",
            bid_value=env._env.bid_value,
            bidding_history=env._env.bidding_history,
            phase="playing",
        )
        assert obs.public.bottom_cards.owner == "landlord"
        assert obs.public.bottom_cards.all_played is False

    def test_bottom_cards_excluded_from_belief_unknown_pool(self):
        """The belief-model unknown pool must exclude public bottom cards (farmer).

        The landlord's own observation keeps the bottom cards in its hand (so its
        belief pool equals the parity pool). A farmer's belief pool is strictly
        smaller: the 3 public bottom cards are removed because they are known to
        belong to the landlord, not "unknown".
        """
        env = self._play_to_playing()
        from douzero.observation import (
            compute_belief_unknown_pool,
            compute_unseen_pool,
        )
        from douzero.observation.seats import is_landlord

        bottom = env._env.three_landlord_cards
        # Landlord (acting first in the playing phase).
        played = {
            role: list(env._env.played_cards.get(role, []))
            for role in ("landlord", "landlord_up", "landlord_down")
        }
        landlord_hand = list(env._env.info_sets["landlord"].player_hand_cards)
        parity_landlord = compute_unseen_pool(landlord_hand, played, bottom)
        belief_landlord = compute_belief_unknown_pool(
            landlord_hand, played, bottom, acting_role="landlord")
        assert len(belief_landlord) == len(parity_landlord)  # bottom in hand

        # Farmer (landlord_up). Its belief pool drops the 3 bottom cards.
        farmer_hand = list(env._env.info_sets["landlord_up"].player_hand_cards)
        parity_farmer = compute_unseen_pool(farmer_hand, played, bottom)
        belief_farmer = compute_belief_unknown_pool(
            farmer_hand, played, bottom, acting_role="landlord_up")
        assert len(belief_farmer) == len(parity_farmer) - len(bottom)
        assert is_landlord("landlord") and not is_landlord("landlord_up")

    def test_standard_conservation(self):
        env = self._play_to_playing()
        obs = get_obs_v2(
            env.infoset,
            ruleset_id="standard",
            bid_value=env._env.bid_value,
            phase="playing",
        )
        # my + unseen + played == deck (bottom cards are inside the landlord's
        # my_handcards at this point, so they are not added separately).
        total = Counter(obs.public.my_handcards)
        total += Counter(obs.public.other_handcards)
        for role in ("landlord", "landlord_up", "landlord_down"):
            total += Counter(obs.public.played_cards.get(role, ()))
        assert total == Counter(DECK)
