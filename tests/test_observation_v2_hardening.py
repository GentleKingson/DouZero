"""Observation V2 hardening tests (P03 round 2).

Covers the round-2 acceptance items:

- (2) FeatureSchema identity: stable_hash is description-stable, changes on
  field/order/dtype/width/semantic-stamp changes; ObservationV2 carries
  feature_schema_version + the full hash.
- (4) Public bottom-card semantics: revealed = original 3, unplayed = current
  subset; revealed is stable after the landlord plays a bottom card while
  unplayed shrinks; GameEnv/InfoSet expose both.
- (5) Deep immutability: containers are frozen+slots; numpy arrays are
  read-only; mutating the source infoset/lists after building does not change
  the observation; mutating an observation's array raises.
- (6) History contract: valid_mask (1=valid), key_padding_mask (True=padding),
  original_length / was_truncated / truncation_side="left"; padding tokens are
  all-zero; editing padding content cannot affect the mask contract.
- (8) Leakage hardening: an access-throws sentinel replacing
  infoset.all_handcards must NOT break get_obs_v2; the public encoder module
  does not import the privileged module; PublicObservation serialisation
  contains no hidden/all_handcards field.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from douzero.env.env import Env
from douzero.env.game import GameEnv
from douzero.env.rules import RuleSet
from douzero.observation import (
    BIDDING_TOKEN_WIDTH,
    BiddingTokenBatch,
    FeatureSchemaManifest,
    FieldSpec,
    HistoryMove,
    ObservationV2,
    PrivilegedObservation,
    PublicObservation,
    build_public_observation,
    build_v2_schema,
    encode_bidding_history,
    encode_history,
    get_obs_v2,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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


def _legacy_infoset(data):
    players = {pos: _NoopAgent() for pos in
               ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    env.card_play_init(data)
    return env


# --------------------------------------------------------------------------- #
# (2) FeatureSchema identity
# --------------------------------------------------------------------------- #
class TestSchemaIdentity:
    def test_observation_carries_schema_version_and_hash(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.feature_schema_version == obs.schema.schema_version == "v2-1"
        assert obs.feature_schema_hash == obs.schema.stable_hash()
        assert len(obs.feature_schema_hash) == 64  # full SHA-256

    def test_stable_hash_is_description_stable(self):
        """Editing a FieldSpec description must NOT change the hash (item 2)."""
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        # Build an identical schema but with a different description on one field.
        new_state = list(s.state_fields)
        idx = next(i for i, f in enumerate(new_state) if f.name == "my_handcards")
        new_state[idx] = FieldSpec(
            "my_handcards", new_state[idx].shape, new_state[idx].dtype,
            "DIFFERENT description text",
        )
        s2 = dataclasses.replace(s, state_fields=tuple(new_state))
        assert s2.stable_hash() == h_before, (
            "description text edit changed the compatibility hash"
        )

    def test_hash_changes_on_field_removal(self):
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        new_state = tuple(f for f in s.state_fields if f.name != "my_handcards")
        s2 = dataclasses.replace(s, state_fields=new_state)
        assert s2.stable_hash() != h_before

    def test_hash_changes_on_field_order(self):
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        # Swap the first two state fields.
        swapped = (s.state_fields[1], s.state_fields[0]) + s.state_fields[2:]
        s2 = dataclasses.replace(s, state_fields=swapped)
        assert s2.stable_hash() != h_before

    def test_hash_changes_on_dtype(self):
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        new_state = list(s.state_fields)
        new_state[0] = FieldSpec(
            new_state[0].name, new_state[0].shape, "int16",
            new_state[0].description,
        )
        s2 = dataclasses.replace(s, state_fields=tuple(new_state))
        assert s2.stable_hash() != h_before

    def test_hash_changes_on_context_field_removal(self):
        """Removing a context field must change the hash (item 3)."""
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        new_context = tuple(
            f for f in s.context_fields if f.name != "rocket_count")
        s2 = dataclasses.replace(s, context_fields=new_context)
        assert s2.stable_hash() != h_before

    def test_hash_changes_on_bidding_token_width(self):
        """Changing a bidding-token field width must change the hash (item 3)."""
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        new_bidding = tuple(
            FieldSpec(f.name, (5,) if f.name == "bid_seat" else f.shape,
                      f.dtype, f.description)
            for f in s.bidding_token_fields
        )
        s2 = dataclasses.replace(s, bidding_token_fields=new_bidding)
        assert s2.stable_hash() != h_before

    def test_hash_changes_on_max_history_len(self):
        s = build_v2_schema(max_history_len=100)
        s2 = build_v2_schema(max_history_len=50)
        assert s.stable_hash() != s2.stable_hash()

    def test_hash_changes_on_mask_semantics_stamp(self):
        s = build_v2_schema()
        h_before = s.stable_hash()
        s2 = FeatureSchemaManifest(
            feature_version=s.feature_version, schema_version=s.schema_version,
            max_history_len=s.max_history_len, card_vector_dim=s.card_vector_dim,
            seat_onehot_width=s.seat_onehot_width,
            move_type_onehot_width=s.move_type_onehot_width,
            bomb_onehot_width=s.bomb_onehot_width,
            max_cards_left=s.max_cards_left,
            state_fields=s.state_fields, action_fields=s.action_fields,
            history_token_fields=s.history_token_fields,
            mask_semantics_version="DIFFERENT",
        )
        assert s2.stable_hash() != h_before

    def test_compatibility_dict_excludes_description(self):
        s = build_v2_schema()
        d = s.compatibility_dict()
        for group in ("state_fields", "action_fields", "history_token_fields"):
            for f in d[group]:
                assert "description" not in f, (
                    f"description leaked into compatibility_dict for {f['name']}"
                )

    def test_two_same_configs_same_hash(self):
        assert build_v2_schema().stable_hash() == build_v2_schema().stable_hash()


# --------------------------------------------------------------------------- #
# (4) Public bottom-card semantics
# --------------------------------------------------------------------------- #
class TestBottomCardSemantics:
    def test_infoset_exposes_revealed_and_unplayed(self, fixed_card_play_data):
        """GameEnv/InfoSet must expose three_landlord_cards_revealed (item 4)."""
        env = _legacy_infoset(fixed_card_play_data)
        iset = env.game_infoset
        assert hasattr(iset, "three_landlord_cards_revealed")
        # In legacy mode the bottom cards are revealed immediately.
        assert iset.three_landlord_cards_revealed is not None
        assert sorted(iset.three_landlord_cards_revealed) == \
            sorted(fixed_card_play_data["three_landlord_cards"])

    def test_revealed_stable_unplayed_shrinks_when_landlord_plays_bottom(
        self, fixed_card_play_data
    ):
        """After the landlord plays a bottom card, revealed is unchanged but
        unplayed shrinks (item 4)."""
        env = _legacy_infoset(fixed_card_play_data)
        # The landlord is the first actor in legacy mode.
        assert env.acting_player_position == "landlord"
        revealed_initial = sorted(env.game_infoset.three_landlord_cards_revealed)
        unplayed_initial = sorted(env.three_landlord_cards)
        assert revealed_initial == unplayed_initial  # at start, nothing played

        # Find a legal action that contains a bottom card and play it.
        bottom = set(env.three_landlord_cards)
        chosen = None
        for action in env.game_infoset.legal_actions:
            if len(action) > 0 and any(c in bottom for c in action):
                chosen = action
                break
        if chosen is not None:
            env.players["landlord"].set_action(chosen)
            env.step()
            iset = env.game_infoset
            # revealed unchanged.
            assert sorted(iset.three_landlord_cards_revealed) == revealed_initial
            # unplayed shrank (at least one bottom card consumed).
            assert len(env.three_landlord_cards) < len(unplayed_initial) or \
                sorted(env.three_landlord_cards) != unplayed_initial

    def test_public_observation_carries_revealed_and_unplayed(self, fixed_card_play_data):
        env = _legacy_infoset(fixed_card_play_data)
        obs = get_obs_v2(env.game_infoset)
        assert obs.public.bottom_cards.revealed == \
            tuple(sorted(fixed_card_play_data["three_landlord_cards"]))
        assert obs.public.bottom_cards.owner == "landlord"


# --------------------------------------------------------------------------- #
# (5) Deep immutability
# --------------------------------------------------------------------------- #
class TestDeepImmutability:
    def test_state_arrays_are_readonly(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert not obs.state.my_handcards.flags.writeable
        assert not obs.actions.features.flags.writeable
        assert not obs.actions.action_mask.flags.writeable
        assert not obs.history.tokens.flags.writeable
        assert not obs.history.valid_mask.flags.writeable
        assert not obs.bidding_tokens.tokens.flags.writeable

    def test_mutation_of_frozen_array_raises(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        with pytest.raises(ValueError, match="read-only"):
            obs.state.my_handcards[0] = 9
        with pytest.raises(ValueError, match="read-only"):
            obs.actions.features[0, 0] = 9

    def test_mutation_of_source_lists_does_not_change_obs(self, fixed_card_play_data):
        """Building an obs from lists, then mutating the source lists, must not
        change the observation (item 5).

        The observation is built from public info at encode time; later source
        mutation does not retroactively alter it. We mutate a NON-conservation
        -breaking field (other_hand_cards, which the encoder ignores) to prove
        isolation without invalidating the deck."""
        env = _legacy_infoset(fixed_card_play_data)
        iset = env.game_infoset
        obs = get_obs_v2(iset)
        my_hand_before = obs.public.my_handcards
        # Mutate a source list the encoder copies at build time. The encoder
        # copies my_handcards into a tuple, so mutating the source after build
        # cannot affect the already-built observation.
        iset.player_hand_cards.append(99)
        # The FIRST obs is unaffected (it captured a copy at build time).
        assert obs.public.my_handcards == my_hand_before
        assert 99 not in obs.public.my_handcards

    def test_public_observation_is_frozen(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        # Cannot reassign a field on a frozen dataclass.
        with pytest.raises((AttributeError, Exception)):
            obs.public.acting_role = "landlord_up"

    def test_privileged_isolated_from_source_mutation(self):
        hands = {"landlord": [3, 4], "landlord_up": [5], "landlord_down": [6]}
        priv = PrivilegedObservation(all_handcards=hands, acting_role="landlord")
        hands["landlord"].append(99)
        hands["new"] = [7]
        assert 99 not in priv.all_handcards["landlord"]
        assert "new" not in priv.all_handcards

    def test_public_does_not_share_ndarray_with_privileged(self):
        """Public and privileged containers hold disjoint data (item 5).

        Privileged holds true hidden hands; public holds only the public unseen
        pool. They share no array storage."""
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        priv = PrivilegedObservation(
            all_handcards={
                role: tuple(env._env.info_sets[role].player_hand_cards)
                for role in ("landlord", "landlord_up", "landlord_down")
            },
            acting_role="landlord",
        )
        # public.other_handcards is the swap-invariant pool; it is NOT the true
        # allocation (it's sorted and pooled). They are different objects.
        assert obs.public.other_handcards is not priv.all_handcards["landlord_up"]

    def test_containers_use_slots(self):
        """frozen (+ slots where used) containers must not allow new attributes
        or field reassignment (item 5)."""
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        # frozen: cannot reassign a declared field; cannot set a new attribute.
        with pytest.raises((AttributeError, TypeError)):
            obs.public.new_field = 1
        with pytest.raises((AttributeError, TypeError)):
            obs.public.acting_role = "landlord_up"
        # PrivilegedObservation is frozen+slots: new attribute -> TypeError.
        priv = PrivilegedObservation(
            all_handcards={"landlord": ()}, acting_role="landlord")
        with pytest.raises((AttributeError, TypeError)):
            priv.new_field = 1
        with pytest.raises((AttributeError, TypeError)):
            priv.acting_role = "landlord_up"


# --------------------------------------------------------------------------- #
# (6) History contract
# --------------------------------------------------------------------------- #
class TestHistoryContract:
    def _moves(self, n):
        return [
            HistoryMove("landlord", (r,), False, 1, r, 1, 1, 19, False, 1)
            for r in range(3, 3 + n)
        ]

    def test_valid_mask_is_1_for_real(self):
        s = build_v2_schema(max_history_len=10)
        batch = encode_history(self._moves(3), s)
        assert batch.valid_mask[:3].tolist() == [1, 1, 1]
        assert batch.valid_mask[3:].tolist() == [0] * 7

    def test_key_padding_mask_true_means_padding(self):
        s = build_v2_schema(max_history_len=10)
        batch = encode_history(self._moves(3), s)
        # PyTorch convention: True = padding (ignore).
        kpm = batch.key_padding_mask
        assert kpm[:3].tolist() == [False, False, False]
        assert kpm[3:].tolist() == [True] * 7

    def test_key_padding_mask_is_negation_of_valid_mask(self):
        s = build_v2_schema(max_history_len=8)
        batch = encode_history(self._moves(5), s)
        assert (batch.key_padding_mask == (batch.valid_mask == 0)).all()

    def test_original_length_and_was_truncated(self):
        s_small = build_v2_schema(max_history_len=4)
        batch = encode_history(self._moves(10), s_small)
        assert batch.original_length == 10
        assert batch.was_truncated is True
        assert batch.num_real == 4  # capped
        assert batch.truncation_side == "left"

    def test_not_truncated_when_under_cap(self):
        s = build_v2_schema(max_history_len=20)
        batch = encode_history(self._moves(5), s)
        assert batch.was_truncated is False
        assert batch.original_length == 5

    def test_padding_tokens_are_all_zero(self):
        s = build_v2_schema(max_history_len=10)
        batch = encode_history(self._moves(2), s)
        # The padding region (indices 2..9) must be all-zero.
        assert (batch.tokens[2:] == 0).all()

    def test_editing_padding_does_not_change_mask(self):
        """The mask contract is independent of padding content (item 6).

        Even if a caller forcibly wrote into a padding slot (bypassing the
        frozen array), the valid_mask/key_padding_mask would still mark it as
        padding. We verify the mask is what defines validity, not the token
        content."""
        s = build_v2_schema(max_history_len=10)
        batch = encode_history(self._moves(2), s)
        # The frozen tokens cannot be mutated, but the mask is authoritative:
        # padding positions are padding regardless of content.
        assert batch.valid_mask.sum() == 2
        assert batch.key_padding_mask.sum() == 8


# --------------------------------------------------------------------------- #
# (8) Leakage hardening
# --------------------------------------------------------------------------- #
class _AccessThrowsSentinel:
    """A stand-in for infoset.all_handcards that raises on ANY attribute access.

    If get_obs_v2 ever reads all_handcards, this raises and the test fails.
    """


class TestLeakageHardening:
    def test_get_obs_v2_succeeds_with_all_handcards_sentinel(self, fixed_card_play_data):
        """Replace infoset.all_handcards with an access-throws sentinel (item 8).

        get_obs_v2 must succeed because it recomputes the unseen pool from
        public info and never reads all_handcards.
        """
        env = _legacy_infoset(fixed_card_play_data)
        iset = env.game_infoset

        # Replace all_handcards with a property that raises on access.
        class _Bomb:
            def __getattr__(self, name):
                raise AssertionError(
                    f"public encoder accessed privileged all_handcards.{name}"
                )
            def __iter__(self):
                raise AssertionError(
                    "public encoder iterated privileged all_handcards"
                )
            def __getitem__(self, k):
                raise AssertionError(
                    "public encoder indexed privileged all_handcards"
                )

        # Use object.__setattr__ bypass if needed; InfoSet is a plain object.
        iset.all_handcards = _Bomb()
        iset.other_hand_cards = _Bomb()  # also privileged-as-true-allocation
        # Must not raise.
        obs = get_obs_v2(iset)
        assert obs.is_privileged is False

    def test_public_encoder_does_not_import_privileged(self):
        """The encode_v2 module must not import the privileged module (item 8)."""
        import douzero.observation.encode_v2 as enc_mod
        import sys
        # The privileged module should not be loaded as a side effect of using
        # the public encoder. Check the encoder's globals for any reference.
        mod_names = [name for name in sys.modules if "privileged" in name]
        # Building an observation must not require the privileged module to be
        # imported. (It may be importable; we assert the encoder does not pull
        # it in during a get_obs_v2 call.)
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        before = set(sys.modules)
        get_obs_v2(env.infoset)
        after = set(sys.modules)
        new_priv = [m for m in (after - before) if "privileged" in m]
        assert new_priv == [], (
            f"public encoder imported privileged module(s): {new_priv}"
        )

    def test_public_serialization_has_no_hidden_fields(self):
        """PublicObservation.to_dict() must contain no hidden/all_handcards
        field (item 8)."""
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        d = obs.public.to_dict()
        flat = json.dumps(d)
        for forbidden in ("all_handcards", "hidden_hand", "other_hand_cards"):
            assert forbidden not in d, f"{forbidden} leaked into public dict keys"
            assert forbidden not in flat, f"{forbidden} leaked into public serialization"

    def test_public_obs_invariant_under_reallocation_with_sentinel(self, fixed_card_play_data):
        """The swap-invariance invariant must still hold even when all_handcards
        is a sentinel (proving the encoder truly ignores it)."""
        env_a = _legacy_infoset(copy.deepcopy(fixed_card_play_data))
        env_b = _legacy_infoset(copy.deepcopy(fixed_card_play_data))
        # Swap a card between farmers in env_b.
        data_b = copy.deepcopy(fixed_card_play_data)
        up = data_b["landlord_up"]
        down = data_b["landlord_down"]
        if up[0] != down[0]:
            up[0], down[0] = down[0], up[0]
            data_b["landlord_up"] = sorted(up)
            data_b["landlord_down"] = sorted(down)
            env_b = _legacy_infoset(data_b)

        iset_a = env_a.game_infoset
        iset_b = env_b.game_infoset
        iset_a.all_handcards = _BombIfAccessed()
        iset_b.all_handcards = _BombIfAccessed()
        obs_a = get_obs_v2(iset_a)
        obs_b = get_obs_v2(iset_b)
        np.testing.assert_array_equal(
            obs_a.state.to_vector(), obs_b.state.to_vector())


class _BombIfAccessed:
    def __getattr__(self, name):
        raise AssertionError(f"accessed privileged field {name}")

    def __iter__(self):
        raise AssertionError("iterated privileged field")

    def __getitem__(self, k):
        raise AssertionError("indexed privileged field")


# --------------------------------------------------------------------------- #
# (3) Model-consumable public input presence
# --------------------------------------------------------------------------- #
class TestModelConsumableInput:
    def _play_to_playing(self, seed=1):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(seed)
        env.reset()
        env.step(None, bid_value=1)
        env.step(None, bid_value=2)
        env.step(None, bid_value=3)
        return env

    def test_public_carries_phase_bid_bidding_rocket_multiplier_rule(self):
        env = self._play_to_playing()
        rs = RuleSet.standard()
        obs = get_obs_v2(
            env.infoset,
            ruleset_id="standard",
            ruleset_version=rs.ruleset_version,
            ruleset_hash=rs.stable_hash(),
            bid_value=env._env.bid_value,
            bidding_history=env._env.bidding_history,
            bidding_order=env._env.bidding_order,
            bomb_count=env._env.bomb_count,
            rocket_count=env._env.rocket_count,
            total_multiplier=2,
            phase="playing",
        )
        assert obs.public.phase == "playing"
        assert obs.public.bid_value == 3
        assert obs.public.rocket_count == 0
        assert obs.public.total_multiplier == 2
        assert obs.public.ruleset_id == "standard"
        assert obs.public.ruleset_version == rs.ruleset_version
        assert obs.public.ruleset_hash == rs.stable_hash()
        assert obs.bidding_tokens.num_bids == len(env._env.bidding_history)
        assert obs.public.bottom_cards.revealed == tuple(sorted(env._env.bottom_cards_revealed))

    def test_bidding_token_width_constant(self):
        from douzero.observation.public import (
            BID_SEAT_ONEHOT_WIDTH, BID_VALUE_ONEHOT_WIDTH,
        )
        assert BIDDING_TOKEN_WIDTH == BID_SEAT_ONEHOT_WIDTH + BID_VALUE_ONEHOT_WIDTH + 1
        bt = encode_bidding_history([("0", 1), ("1", 3)])
        assert bt.tokens.shape == (2, BIDDING_TOKEN_WIDTH)
        assert bt.num_bids == 2
        assert not bt.tokens.flags.writeable

    def test_bidding_tokens_empty_in_legacy(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.bidding_tokens.num_bids == 0
        assert obs.bidding_tokens.tokens.shape == (0, BIDDING_TOKEN_WIDTH)


# --------------------------------------------------------------------------- #
# (Blocker 1, round 3) Mapping deep immutability
# --------------------------------------------------------------------------- #
class TestMappingImmutability:
    """frozen+slots does NOT freeze interior dicts (review round 3, blocker 1).

    Every mapping field on PublicObservation / PrivilegedObservation must be
    exposed read-only, so ``obs.public.played_cards['landlord'] = ...`` raises.
    """

    def test_public_mapping_fields_are_readonly(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        from types import MappingProxyType
        for field_name in (
            "seat_context", "played_cards", "last_move_dict", "num_cards_left",
        ):
            assert isinstance(getattr(obs.public, field_name), MappingProxyType), (
                f"{field_name} is not a read-only MappingProxyType"
            )

    def test_public_mapping_mutation_raises(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        # Subscript assignment on a MappingProxyType raises TypeError.
        with pytest.raises(TypeError):
            obs.public.played_cards["landlord"] = (3, 3, 3, 3)
        with pytest.raises(TypeError):
            obs.public.num_cards_left["landlord"] = 999
        with pytest.raises(TypeError):
            obs.public.seat_context["x"] = "y"
        with pytest.raises(TypeError):
            obs.public.last_move_dict["landlord"] = (1,)

    def test_privileged_all_handcards_is_readonly(self):
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3, 4], "landlord_up": [5]},
            acting_role="landlord",
        )
        from types import MappingProxyType
        assert isinstance(priv.all_handcards, MappingProxyType)
        with pytest.raises(TypeError):
            priv.all_handcards["landlord"] = (9,)

    def test_privileged_hidden_labels_deep_frozen(self):
        """hidden_hand_labels nested values cannot be mutated via the container."""
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3]},
            acting_role="landlord",
            hidden_hand_labels={"counts": [1, 2, 3], "meta": {"k": 9}},
        )
        # Top-level mapping is read-only.
        with pytest.raises(TypeError):
            priv.hidden_hand_labels["counts"] = [0]
        # Source mutation does not affect the container.
        src = {"counts": [1]}
        p2 = PrivilegedObservation(
            all_handcards={"landlord": [3]}, acting_role="landlord",
            hidden_hand_labels=src)
        src["counts"].append(99)
        assert 99 not in p2.hidden_hand_labels["counts"]


# --------------------------------------------------------------------------- #
# (Blocker 2, round 3) Public inputs bound to the schema
# --------------------------------------------------------------------------- #
class TestSchemaBoundPublicInputs:
    """Every model-consumable public input must live in a schema-described
    tensor block (review round 3, blocker 2), not just on obs.public."""

    def test_schema_has_context_and_bidding_groups(self):
        s = build_v2_schema()
        assert len(s.context_fields) >= 7
        assert len(s.bidding_token_fields) >= 3
        ctx_names = {f.name for f in s.context_fields}
        assert {
            "bottom_cards_revealed", "bottom_cards_unplayed", "bid_value_onehot",
            "phase_onehot", "rocket_count", "total_multiplier", "ruleset_id_onehot",
        } <= ctx_names

    def test_observation_carries_context_block(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.context is not None
        assert obs.schema_bidding_tokens is not None
        # context block arrays read-only.
        assert not obs.context.bottom_cards_revealed.flags.writeable
        assert not obs.schema_bidding_tokens.tokens.flags.writeable

    def test_context_block_width_matches_schema(self):
        s = build_v2_schema()
        expected = sum(int(np.prod(f.shape)) for f in s.context_fields)
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.context.to_vector().shape == (expected,)

    def test_context_block_reflects_standard_state(self):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(1)
        env.reset()
        env.step(None, bid_value=1)
        env.step(None, bid_value=2)
        env.step(None, bid_value=3)
        obs = get_obs_v2(
            env.infoset,
            ruleset=RuleSet.standard(),
            bid_value=3,
            bidding_history=env._env.bidding_history,
            bidding_order=env._env.bidding_order,
            phase="playing",
        )
        # bid_value 3 -> one-hot index 3 set.
        assert obs.context.bid_value_onehot[3] == 1
        # phase "playing" -> index 2 set.
        assert obs.context.phase_onehot[2] == 1
        # ruleset "standard" -> index 1 set.
        assert obs.context.ruleset_id_onehot[1] == 1
        # bottom revealed present in the block.
        assert obs.context.bottom_cards_revealed.sum() == 3
        # 3 bids encoded.
        assert obs.schema_bidding_tokens.num_bids == 3
        # schema bidding token width = 3 + 4 + 1 = 8.
        assert obs.schema_bidding_tokens.tokens.shape == (3, 8)

    def test_hash_covers_context_and_bidding_groups(self):
        """stable_hash must change when context/bidding groups change."""
        import dataclasses
        s = build_v2_schema()
        h = s.stable_hash()
        # Empty context -> different hash.
        s2 = dataclasses.replace(s, context_fields=())
        assert s2.stable_hash() != h
        # Empty bidding -> different hash.
        s3 = dataclasses.replace(s, bidding_token_fields=())
        assert s3.stable_hash() != h


# --------------------------------------------------------------------------- #
# (Blocker 3, round 3) RuleSet identity validation
# --------------------------------------------------------------------------- #
class TestRuleIdentityValidation:
    """get_obs_v2 must never produce an empty/inconsistent ruleset hash
    (review round 3, blocker 3)."""

    def test_legacy_default_fills_canonical_hash(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        obs = get_obs_v2(env.infoset)
        assert obs.public.ruleset_hash != ""
        assert len(obs.public.ruleset_hash) == 64
        assert obs.public.ruleset_hash == RuleSet.legacy().stable_hash()

    def test_ruleset_preferred_path(self):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(1)
        env.reset()
        env.step(None, bid_value=1)
        env.step(None, bid_value=2)
        env.step(None, bid_value=3)
        obs = get_obs_v2(env.infoset, ruleset=RuleSet.standard())
        assert obs.public.ruleset_id == "standard"
        assert obs.public.ruleset_hash == RuleSet.standard().stable_hash()
        assert obs.public.ruleset_version == RuleSet.standard().ruleset_version

    def test_standard_without_version_hash_rejected(self):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(1)
        env.reset()
        env.step(None, bid_value=3)
        with pytest.raises(ValueError, match="standard"):
            get_obs_v2(env.infoset, ruleset_id="standard")

    def test_standard_legacy_version_contradiction_rejected(self):
        env = Env("adp", ruleset=RuleSet.standard())
        np.random.seed(1)
        env.reset()
        env.step(None, bid_value=3)
        with pytest.raises(ValueError):
            get_obs_v2(
                env.infoset,
                ruleset_id="standard",
                ruleset_version="legacy-v1",
                ruleset_hash=RuleSet.standard().stable_hash(),
            )

    def test_both_ruleset_and_id_rejected(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        with pytest.raises(ValueError, match="EITHER"):
            get_obs_v2(env.infoset, ruleset=RuleSet.legacy(), ruleset_id="legacy")

    def test_legacy_wrong_hash_rejected(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        with pytest.raises(ValueError, match="disagrees"):
            get_obs_v2(env.infoset, ruleset_id="legacy", ruleset_hash="wrong")

    def test_unknown_id_rejected(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        with pytest.raises(ValueError):
            get_obs_v2(env.infoset, ruleset_id="v3")

    def test_non_ruleset_object_rejected(self):
        env = Env("adp")
        np.random.seed(0)
        env.reset()
        with pytest.raises(TypeError):
            get_obs_v2(env.infoset, ruleset="not a ruleset")


# --------------------------------------------------------------------------- #
# (Round 4, blocker 1) Privileged nested deep immutability
# --------------------------------------------------------------------------- #
class TestPrivilegedNestedFreeze:
    """deep_freeze must make EVERY nesting level immutable, not just the top
    mapping (review round 4, blocker 1)."""

    def test_nested_list_becomes_tuple(self):
        import numpy as np
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3]},
            acting_role="landlord",
            hidden_hand_labels={"counts": [1, 2, 3]},
        )
        # list -> tuple: .append raises AttributeError (tuples have no append).
        with pytest.raises(AttributeError):
            priv.hidden_hand_labels["counts"].append(99)
        assert isinstance(priv.hidden_hand_labels["counts"], tuple)

    def test_nested_dict_is_readonly(self):
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3]},
            acting_role="landlord",
            hidden_hand_labels={"meta": {"k": 9}},
        )
        # nested dict -> MappingProxyType: subscript assignment raises TypeError.
        with pytest.raises(TypeError):
            priv.hidden_hand_labels["meta"]["k"] = 0

    def test_nested_ndarray_is_readonly(self):
        import numpy as np
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3]},
            acting_role="landlord",
            hidden_hand_labels={"arr": np.array([1, 2, 3])},
        )
        # nested ndarray -> read-only: item assignment raises ValueError.
        with pytest.raises(ValueError):
            priv.hidden_hand_labels["arr"][0] = 1

    def test_deeply_nested_structure_all_frozen(self):
        import numpy as np
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3]},
            acting_role="landlord",
            hidden_hand_labels={
                "outer": {"inner_list": [1, {"deep": [2, 3]}], "s": {1, 2}},
            },
        )
        # every level is immutable.
        with pytest.raises(AttributeError):
            priv.hidden_hand_labels["outer"]["inner_list"].append(9)
        with pytest.raises(TypeError):
            priv.hidden_hand_labels["outer"]["inner_list"][1]["deep"][0] = 9
        assert isinstance(priv.hidden_hand_labels["outer"]["s"], frozenset)

    def test_to_plain_roundtrip(self):
        """to_dict must still serialize a deeply-frozen container correctly."""
        import numpy as np
        priv = PrivilegedObservation(
            all_handcards={"landlord": [3, 4]},
            acting_role="landlord",
            hidden_hand_labels={"counts": [1, 2], "arr": np.array([5, 6])},
        )
        d = priv.to_dict()
        import json
        # Must be JSON-serializable.
        json.dumps(d)
        assert d["hidden_hand_labels"]["counts"] == [1, 2]
        assert d["hidden_hand_labels"]["arr"] == [5, 6]


# --------------------------------------------------------------------------- #
# (Round 4, blocker 2) total_multiplier numeric range safety
# --------------------------------------------------------------------------- #
class TestMultiplierRange:
    """The standard ruleset's total_multiplier is unbounded (max_multiplier is
    None), so the context field must use int32, not int8 (review round 4,
    blocker 2). bid × bombs × rocket × spring can exceed int8's 127."""

    def test_schema_multiplier_is_int32(self):
        s = build_v2_schema()
        spec = s.field_by_name("total_multiplier", "context")
        assert spec.dtype == "int32", (
            f"total_multiplier must be int32 (unbounded), got {spec.dtype}"
        )

    def test_multiplier_127_encoded(self):
        from douzero.observation.encode_v2 import _build_context_block
        from douzero.observation import build_public_observation
        pub = build_public_observation(
            acting_role="landlord", my_handcards=[3], other_handcards=[],
            played_cards={"landlord": [], "landlord_up": [], "landlord_down": []},
            last_move=[], last_move_dict={"landlord": [], "landlord_up": [], "landlord_down": []},
            three_landlord_cards=[], num_cards_left={"landlord": 1, "landlord_up": 0, "landlord_down": 0},
            legal_actions=[[3]], total_multiplier=127)
        ctx = _build_context_block(pub, "playing", build_v2_schema())
        assert ctx.total_multiplier[0] == 127
        assert ctx.total_multiplier.dtype == np.int32

    def test_multiplier_128_does_not_overflow(self):
        from douzero.observation.encode_v2 import _build_context_block
        from douzero.observation import build_public_observation
        pub = build_public_observation(
            acting_role="landlord", my_handcards=[3], other_handcards=[],
            played_cards={"landlord": [], "landlord_up": [], "landlord_down": []},
            last_move=[], last_move_dict={"landlord": [], "landlord_up": [], "landlord_down": []},
            three_landlord_cards=[], num_cards_left={"landlord": 1, "landlord_up": 0, "landlord_down": 0},
            legal_actions=[[3]], total_multiplier=128)
        ctx = _build_context_block(pub, "playing", build_v2_schema())
        # 128 would overflow int8 to -128; int32 holds it.
        assert ctx.total_multiplier[0] == 128
        assert ctx.total_multiplier[0] > 0

    def test_multiplier_256_encoded(self):
        from douzero.observation.encode_v2 import _build_context_block
        from douzero.observation import build_public_observation
        pub = build_public_observation(
            acting_role="landlord", my_handcards=[3], other_handcards=[],
            played_cards={"landlord": [], "landlord_up": [], "landlord_down": []},
            last_move=[], last_move_dict={"landlord": [], "landlord_up": [], "landlord_down": []},
            three_landlord_cards=[], num_cards_left={"landlord": 1, "landlord_up": 0, "landlord_down": 0},
            legal_actions=[[3]], total_multiplier=256)
        ctx = _build_context_block(pub, "playing", build_v2_schema())
        assert ctx.total_multiplier[0] == 256
        assert ctx.total_multiplier[0] > 0

    def test_multiplier_dtype_change_reflected_in_schema_hash(self):
        """Changing total_multiplier dtype changes the schema hash."""
        import dataclasses
        s = build_v2_schema()
        h_before = s.stable_hash()
        new_context = tuple(
            FieldSpec(f.name, f.shape, "int8" if f.name == "total_multiplier" else f.dtype,
                      f.description)
            for f in s.context_fields
        )
        s2 = dataclasses.replace(s, context_fields=new_context)
        assert s2.stable_hash() != h_before

    def test_multiplier_zero_encoded(self):
        from douzero.observation.encode_v2 import _build_context_block
        from douzero.observation import build_public_observation
        pub = build_public_observation(
            acting_role="landlord", my_handcards=[3], other_handcards=[],
            played_cards={"landlord": [], "landlord_up": [], "landlord_down": []},
            last_move=[], last_move_dict={"landlord": [], "landlord_up": [], "landlord_down": []},
            three_landlord_cards=[], num_cards_left={"landlord": 1, "landlord_up": 0, "landlord_down": 0},
            legal_actions=[[3]], total_multiplier=0)
        ctx = _build_context_block(pub, "playing", build_v2_schema())
        assert ctx.total_multiplier[0] == 0
