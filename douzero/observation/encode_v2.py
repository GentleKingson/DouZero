"""Observation V2 encoder (P03).

:func:`get_obs_v2` is the V2 counterpart of the legacy ``get_obs``. It produces
an :class:`ObservationV2` container holding:

- a **state** block encoded **once per decision** (not per legal action);
- a :class:`LegalActionBatch` (one feature row per legal action);
- a :class:`HistoryTokenBatch` (the full public action history, padded).

The legacy encoder duplicates the entire state across every legal-action row
and runs the LSTM once per action. V2 encodes the state once and the actions
once, so a factorized model (P04) can score candidates without recomputing the
shared state/history.

This encoder reads ONLY public fields off the infoset. It never touches
``infoset.all_handcards`` or treats ``other_hand_cards`` as a true allocation —
``other_hand_cards`` is the public swap-invariant union, recomputed here from
public information via :func:`compute_unseen_pool` to guarantee the leakage
invariant holds regardless of how the infoset was populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from douzero.env import move_detector as md

from .cards import CARD_VECTOR_DIM, cards_to_vector
from .history import (
    HistoryMove,
    HistoryTokenBatch,
    encode_history,
)
from .public import (
    PublicObservation,
    build_public_observation,
    compute_unseen_pool,
)
from .schema import (
    BOMB_ONEHOT_WIDTH,
    MAX_CARDS_LEFT,
    MOVE_TYPE_ONEHOT_WIDTH,
    SEAT_ONEHOT_WIDTH,
    FeatureSchemaManifest,
    build_v2_schema,
)
from .seats import ALL_ROLES, LANDLORD_ROLE

#: Bombs and the rocket, matching ``douzero.env.game.bombs``. Kept here so the
#: encoder can flag bomb/rocket actions without importing the game module.
_BOMB_SET: frozenset[tuple[int, ...]] = frozenset(
    tuple(sorted(b)) for b in (
        [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13],
         [14, 14, 14, 14], [17, 17, 17, 17], [20, 30]]
    )
)


@dataclass
class StateBlock:
    """Encoded per-decision state features (no legal-action batch dim).

    Each attribute is an int8 numpy array whose shape matches the corresponding
    :class:`FeatureSchemaManifest` state field. The encoder concatenates them
    in schema order to form the flat state vector.
    """

    my_handcards: np.ndarray
    other_handcards: np.ndarray
    landlord_played: np.ndarray
    landlord_down_played: np.ndarray
    landlord_up_played: np.ndarray
    last_move: np.ndarray
    num_cards_left_landlord: np.ndarray
    num_cards_left_landlord_down: np.ndarray
    num_cards_left_landlord_up: np.ndarray
    bomb_num: np.ndarray
    acting_role: np.ndarray

    def to_vector(self) -> np.ndarray:
        """Concatenate all fields in schema order into one int8 vector."""
        return np.concatenate([
            self.my_handcards,
            self.other_handcards,
            self.landlord_played,
            self.landlord_down_played,
            self.landlord_up_played,
            self.last_move,
            self.num_cards_left_landlord,
            self.num_cards_left_landlord_down,
            self.num_cards_left_landlord_up,
            self.bomb_num,
            self.acting_role,
        ]).astype(np.int8)


@dataclass
class LegalActionBatch:
    """Encoded legal-action features. Shape ``(N, action_width)`` int8.

    ``N == len(legal_actions)``. ``action_mask`` is all-ones (every row is a
    real action); it exists so a batched/padded representation can reuse the
    same array shape with a mask, per the variable-legal-action contract.
    """

    features: np.ndarray  # (N, action_width)
    action_mask: np.ndarray  # (N,)
    legal_actions: tuple[tuple[int, ...], ...]


@dataclass
class ObservationV2:
    """Full V2 observation for one decision. PUBLIC ONLY.

    Holds the encoded state block (once), the legal-action batch, and the
    history-token batch, plus the originating :class:`PublicObservation` for
    inspection / serialisation. No privileged field is present or reachable.
    """

    schema: FeatureSchemaManifest
    public: PublicObservation
    state: StateBlock
    actions: LegalActionBatch
    history: HistoryTokenBatch
    feature_version: str

    @property
    def is_privileged(self) -> bool:
        """Always False — an ObservationV2 is public-only by construction."""
        return False


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def _cards_left_onehot(count: int) -> np.ndarray:
    """One-hot encode a remaining-card count over ``MAX_CARDS_LEFT`` slots.

    Matches the legacy ``_get_one_hot_array`` convention: slot ``count-1`` is
    set (count is 1-indexed). A zero count yields the all-zero vector (no slot
    for "0 cards" because the game ends when any hand empties).
    """
    vec = np.zeros(MAX_CARDS_LEFT, dtype=np.int8)
    if count < 0 or count > MAX_CARDS_LEFT:
        raise ValueError(f"cards-left count {count} out of range [0, {MAX_CARDS_LEFT}]")
    if count >= 1:
        vec[count - 1] = 1
    return vec


def _bomb_onehot(bomb_num: int) -> np.ndarray:
    """One-hot encode the bomb counter over ``BOMB_ONEHOT_WIDTH`` slots.

    Matches the legacy ``_get_one_hot_bomb``. Counts at or beyond the width
    saturate to the last slot (legacy behaviour indexes ``one_hot[bomb_num]``,
    which would overflow; we saturate defensively).
    """
    vec = np.zeros(BOMB_ONEHOT_WIDTH, dtype=np.int8)
    idx = min(int(bomb_num), BOMB_ONEHOT_WIDTH - 1)
    vec[idx] = 1
    return vec


def _acting_role_onehot(role: str) -> np.ndarray:
    """One-hot encode the acting role over ``SEAT_ONEHOT_WIDTH`` slots.

    The first three slots are the absolute roles in canonical order; the rest
    are zero (reserved for relative-seat labels used elsewhere).
    """
    vec = np.zeros(SEAT_ONEHOT_WIDTH, dtype=np.int8)
    if role in ALL_ROLES:
        vec[ALL_ROLES.index(role)] = 1
    return vec


def _encode_action_row(action, schema: FeatureSchemaManifest) -> np.ndarray:
    """Encode one legal action into a flat int8 feature row."""
    cards_vec = cards_to_vector(action)
    is_pass = len(action) == 0
    move_info = md.get_move_type(list(action))
    move_type = int(move_info["type"])
    main_rank = int(move_info.get("rank", 0))
    length = int(move_info.get("len", len(action)))
    action_key = tuple(sorted(action))
    is_bomb = action_key in _BOMB_SET

    move_type_oh = np.zeros(MOVE_TYPE_ONEHOT_WIDTH, dtype=np.int8)
    if 0 <= move_type < MOVE_TYPE_ONEHOT_WIDTH:
        move_type_oh[move_type] = 1

    row = np.concatenate([
        cards_vec,
        move_type_oh,
        np.array([main_rank], dtype=np.int8),
        np.array([length], dtype=np.int8),
        np.array([1 if is_pass else 0], dtype=np.int8),
        np.array([1 if is_bomb else 0], dtype=np.int8),
    ]).astype(np.int8)
    return row


# --------------------------------------------------------------------------- #
# History reconstruction
# --------------------------------------------------------------------------- #
def _reconstruct_history_moves(
    infoset,
    num_cards_left_now: dict[str, int],
    phase_code: int,
) -> list[HistoryMove]:
    """Reconstruct the public action history as :class:`HistoryMove` tokens.

    Walks ``infoset.card_play_action_seq`` (the public, ordered action list)
    and assigns each action to its absolute actor using the canonical turn
    order. The turn order starts at the landlord and proceeds
    ``landlord -> landlord_down -> landlord_up``; passes are included in the
    sequence (they consume a turn).

    ``cards_left_after`` is reconstructed by replaying the hand counts
    backward from the current counts. This is fully determined by public
    information (the action sequence + current counts), so it is swap-invariant.
    """
    seq = list(infoset.card_play_action_seq or [])
    if not seq:
        return []

    # Reconstruct per-role remaining counts before each action by walking
    # backward from the current counts.
    # current counts are AFTER all actions; subtract each action's cards.
    counts_now = {role: num_cards_left_now.get(role, 0) for role in ALL_ROLES}
    counts_after = [None] * len(seq)
    counts = dict(counts_now)
    for i in range(len(seq) - 1, -1, -1):
        # `counts` currently holds counts AFTER action i (for i = last) or
        # after action i+1. Subtract action i's cards to get counts BEFORE i,
        # then counts_after[i] = counts before subtraction = after action i.
        counts_after[i] = dict(counts)
        action = seq[i]
        # Determine actor for position i to know whose count to increment.
        actor = _actor_at(i)
        for c in action:
            counts[actor] = counts.get(actor, 0) + 1

    moves: list[HistoryMove] = []
    for i, action in enumerate(seq):
        actor = _actor_at(i)
        move_info = md.get_move_type(list(action))
        is_bomb = tuple(sorted(action)) in _BOMB_SET
        moves.append(HistoryMove(
            actor_role=actor,
            cards=tuple(sorted(action)),
            is_pass=(len(action) == 0),
            move_type=int(move_info["type"]),
            main_rank=int(move_info.get("rank", 0)),
            length=int(move_info.get("len", len(action))),
            card_count=len(action),
            cards_left_after=int(counts_after[i].get(actor, 0)),
            is_bomb=is_bomb,
            phase=phase_code,
        ))
    return moves


def _actor_at(turn_index: int) -> str:
    """Return the absolute role that acts at ``turn_index`` (0 = landlord lead).

    Turn order: landlord(0) -> landlord_down(1) -> landlord_up(2) -> landlord...
    Passes consume a turn (they appear in ``card_play_action_seq``), so the
    index alone determines the actor.
    """
    return ALL_ROLES[turn_index % len(ALL_ROLES)]


# --------------------------------------------------------------------------- #
# Public encoder
# --------------------------------------------------------------------------- #
def get_obs_v2(
    infoset,
    *,
    schema: FeatureSchemaManifest | None = None,
    ruleset_id: str = "legacy",
    bid_value: int = 0,
    bidding_history=None,
    bomb_count: int | None = None,
    rocket_count: int | None = None,
    total_multiplier: int = 1,
    phase: str = "playing",
) -> ObservationV2:
    """Encode an infoset into a public :class:`ObservationV2`.

    Parameters
    ----------
    infoset
        The legacy ``InfoSet`` (or any object with the same public attributes:
        ``player_position``, ``player_hand_cards``, ``played_cards``,
        ``last_move``, ``last_move_dict``, ``three_landlord_cards``,
        ``num_cards_left_dict``, ``card_play_action_seq``, ``legal_actions``,
        ``bomb_num``). Privileged fields (``all_handcards``,
        ``other_hand_cards``) are IGNORED — the unseen pool is recomputed from
        public information.
    schema
        Optional prebuilt schema (controls ``max_history_len``). Defaults to
        :func:`build_v2_schema` with the default cap.
    ruleset_id, bid_value, bidding_history, bomb_count, rocket_count, ...
        Public rule/bidding state. In legacy mode these default to the legacy
        identity (no bidding, ``bomb_num`` conflates bombs+rocket). In standard
        mode the caller passes the real values from the ``GameEnv``.

    The state is encoded **once**; legal actions are encoded into a
    :class:`LegalActionBatch`; history is encoded into a padded
    :class:`HistoryTokenBatch`. No privileged data is read or returned.
    """
    if schema is None:
        schema = build_v2_schema()

    acting_role = infoset.player_position
    my_hand = list(infoset.player_hand_cards or [])
    played = {
        role: list((infoset.played_cards or {}).get(role, []))
        for role in ALL_ROLES
    }
    bottom_unplayed = list(infoset.three_landlord_cards or [])
    # Public unseen pool — recomputed from public info (swap-invariant).
    other_hand = list(compute_unseen_pool(my_hand, played, bottom_unplayed))

    num_left = dict(infoset.num_cards_left_dict or {})

    public = build_public_observation(
        acting_role=acting_role,
        my_handcards=my_hand,
        other_handcards=other_hand,
        played_cards=played,
        last_move=list(infoset.last_move or []),
        last_move_dict={
            role: list((infoset.last_move_dict or {}).get(role, []))
            for role in ALL_ROLES
        },
        three_landlord_cards=bottom_unplayed,
        num_cards_left=num_left,
        legal_actions=infoset.legal_actions,
        phase=phase,
        ruleset_id=ruleset_id,
        bid_value=bid_value,
        bidding_history=bidding_history,
        bomb_count=int(bomb_count if bomb_count is not None else infoset.bomb_num),
        rocket_count=int(rocket_count if rocket_count is not None else 0),
        total_multiplier=total_multiplier,
    )

    # --- State block (encoded once) ---
    state = StateBlock(
        my_handcards=cards_to_vector(public.my_handcards),
        other_handcards=cards_to_vector(public.other_handcards),
        landlord_played=cards_to_vector(public.played_cards.get("landlord", ())),
        landlord_down_played=cards_to_vector(public.played_cards.get("landlord_down", ())),
        landlord_up_played=cards_to_vector(public.played_cards.get("landlord_up", ())),
        last_move=cards_to_vector(public.last_move),
        num_cards_left_landlord=_cards_left_onehot(
            public.num_cards_left.get("landlord", 0)),
        num_cards_left_landlord_down=_cards_left_onehot(
            public.num_cards_left.get("landlord_down", 0)),
        num_cards_left_landlord_up=_cards_left_onehot(
            public.num_cards_left.get("landlord_up", 0)),
        bomb_num=_bomb_onehot(public.bomb_count),
        acting_role=_acting_role_onehot(public.acting_role),
    )

    # --- Legal action batch ---
    legal = list(infoset.legal_actions or [])
    if legal:
        rows = [_encode_action_row(action, schema) for action in legal]
        features = np.stack(rows, axis=0).astype(np.int8)
    else:
        # No legal actions should not happen (pass is always available when
        # there is a rival move), but guard defensively with an empty array.
        features = np.zeros((0, _action_width(schema)), dtype=np.int8)
    action_mask = np.ones(features.shape[0], dtype=np.int8)
    actions = LegalActionBatch(
        features=features,
        action_mask=action_mask,
        legal_actions=tuple(tuple(sorted(a)) for a in legal),
    )

    # --- History batch ---
    phase_code = _phase_code(phase)
    moves = _reconstruct_history_moves(infoset, num_left, phase_code)
    history = encode_history(moves, schema)

    return ObservationV2(
        schema=schema,
        public=public,
        state=state,
        actions=actions,
        history=history,
        feature_version=schema.feature_version,
    )


def _action_width(schema: FeatureSchemaManifest) -> int:
    width = 0
    for spec in schema.action_fields:
        w = 1
        for d in spec.shape:
            w *= d
        width += w
    return width


def _phase_code(phase: str) -> int:
    from .history import (
        PHASE_CODE_BIDDING,
        PHASE_CODE_PLAYING,
        PHASE_CODE_REVEAL_BOTTOM,
    )
    mapping = {
        "bidding": PHASE_CODE_BIDDING,
        "playing": PHASE_CODE_PLAYING,
        "reveal_bottom": PHASE_CODE_REVEAL_BOTTOM,
    }
    return mapping.get(phase, PHASE_CODE_PLAYING)
