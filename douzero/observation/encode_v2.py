"""Observation V2 encoder (P03, hardened round 2).

:func:`get_obs_v2` is the V2 counterpart of the legacy ``get_obs``. It produces
an :class:`ObservationV2` container holding:

- a **state** block encoded **once per decision** (not per legal action);
- a :class:`LegalActionBatch` (one feature row per legal action);
- a :class:`HistoryTokenBatch` (the full public action history, bounded +
  left-truncated, with an explicit padding mask);
- the public bottom-card identity, bidding history, phase, rocket count, total
  multiplier, and rule identity (item 3 — every public input a V2 model needs,
  so P05 never reaches into ``obs.public`` ad hoc);
- the originating schema identity (``feature_schema_version`` +
  ``feature_schema_hash``) so a checkpoint can reject an incompatible schema
  (item 2);
- the raw public action sequence, so the legacy adapter can reconstruct the
  legacy tensors from the V2 container alone (item 7).

Deep immutability (item 5): :class:`ObservationV2`, :class:`StateBlock`, and
:class:`LegalActionBatch` are frozen+slots; their numpy arrays are frozen
(``write=False``); caller inputs are copied.

The legacy encoder duplicates the entire state across every legal-action row
and runs the LSTM once per action. V2 encodes the state once and the actions
once, so a factorized model (P04) can score candidates without recomputing the
shared state/history.

This encoder reads ONLY public fields off the infoset. It never touches
``infoset.all_handcards`` or treats ``other_hand_cards`` as a true allocation —
the leakage test replaces ``all_handcards`` with an access-throws sentinel and
asserts ``get_obs_v2`` still succeeds (item 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from douzero.env import move_detector as md

from .cards import CARD_VECTOR_DIM, cards_to_vector
from .history import (
    HistoryMove,
    HistoryTokenBatch,
    encode_history,
)
from .public import (
    BiddingTokenBatch,
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
from .seats import ALL_ROLES

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
_ROCKET_KEY: tuple[int, int] = (20, 30)


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a read-only numpy array, PRESERVING its dtype.

    The context block's ``total_multiplier`` is encoded as int32 (it is
    unbounded in the standard ruleset: bid × bombs × rocket × spring can
    exceed int8's 127), so the freezer must NOT force every array to int8.
    Each caller passes an already-typed array (int8 for one-hots/card vectors,
    int32 for the multiplier); this function only copies + sets read-only.
    """
    out = np.asarray(arr)
    if out.flags.writeable:
        out = out.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class StateBlock:
    """Encoded per-decision state features (no legal-action batch dim).

    Each attribute is a read-only int8 numpy array whose shape matches the
    corresponding :class:`FeatureSchemaManifest` state field. The encoder
    concatenates them in schema order to form the flat state vector.

    This is the legacy-parity state block (hand/played/last-move/counts/bomb/
    acting-role). The additional public inputs (bottom-card identity, bid,
    phase, rocket count, total multiplier, rule identity, bidding tokens) live
    in the separate :class:`PublicContextBlock` / :class:`SchemaBiddingTokenBatch`
    (item 3) — both schema-described — so P05 consumes every public input from
    versioned tensor blocks rather than reaching into ``obs.public`` ad hoc.
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

    def __post_init__(self) -> None:
        for arr_name in (
            "my_handcards", "other_handcards", "landlord_played",
            "landlord_down_played", "landlord_up_played", "last_move",
            "num_cards_left_landlord", "num_cards_left_landlord_down",
            "num_cards_left_landlord_up", "bomb_num", "acting_role",
        ):
            arr = getattr(self, arr_name)
            if arr.flags.writeable:
                object.__setattr__(self, arr_name, _freeze(arr))

    def to_vector(self) -> np.ndarray:
        """Concatenate all fields in schema order into one int8 vector.

        Returns a fresh, writable copy (the source arrays are frozen).
        """
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


@dataclass(frozen=True)
class LegalActionBatch:
    """Encoded legal-action features. Shape ``(N, action_width)`` int8.

    ``N == len(legal_actions)``. ``action_mask`` is all-ones (every row is a
    real action); it exists so a batched/padded representation can reuse the
    same array shape with a mask, per the variable-legal-action contract.

    Deep immutability: ``features`` and ``action_mask`` are frozen
    (``write=False``).
    """

    features: np.ndarray  # (N, action_width)
    action_mask: np.ndarray  # (N,)
    legal_actions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.features.flags.writeable:
            object.__setattr__(self, "features", _freeze(self.features))
        if self.action_mask.flags.writeable:
            object.__setattr__(self, "action_mask", _freeze(self.action_mask))


@dataclass(frozen=True)
class PublicContextBlock:
    """Schema-described public-context features (item 3), encoded once.

    Carries the public bottom-card identity (revealed + unplayed), the final
    bid one-hot, the phase one-hot, the rocket count, the total multiplier, and
    the ruleset-id one-hot. Every field's name/shape/dtype is described by
    :attr:`FeatureSchemaManifest.context_fields`, so a model consumes it from a
    versioned tensor block rather than reaching into ``obs.public`` ad hoc.

    Deep immutability: all arrays are frozen (``write=False``).
    """

    bottom_cards_revealed: np.ndarray
    bottom_cards_unplayed: np.ndarray
    bid_value_onehot: np.ndarray
    phase_onehot: np.ndarray
    rocket_count: np.ndarray
    total_multiplier: np.ndarray
    ruleset_id_onehot: np.ndarray

    def __post_init__(self) -> None:
        for arr_name in (
            "bottom_cards_revealed", "bottom_cards_unplayed", "bid_value_onehot",
            "phase_onehot", "rocket_count", "total_multiplier", "ruleset_id_onehot",
        ):
            arr = getattr(self, arr_name)
            if arr.flags.writeable:
                object.__setattr__(self, arr_name, _freeze(arr))

    def to_vector(self) -> np.ndarray:
        """Concatenate all fields in schema order into a single vector.

        Fields are a mix of int8 (one-hots, card vectors, counts) and int32
        (the unbounded ``total_multiplier``). numpy promotes the concatenation
        to int32 so the multiplier does not overflow; the result is read-only
        is irrelevant here (it is a fresh concatenation copy). Models that
        prefer a packed dtype may cast per-field using the schema.
        """
        return np.concatenate([
            self.bottom_cards_revealed,
            self.bottom_cards_unplayed,
            self.bid_value_onehot,
            self.phase_onehot,
            self.rocket_count,
            self.total_multiplier,
            self.ruleset_id_onehot,
        ])


@dataclass(frozen=True)
class SchemaBiddingTokenBatch:
    """Schema-described bidding-token tensor (item 3).

    ``tokens`` has shape ``(num_bids, bidding_token_width)`` int8 (read-only),
    matching :attr:`FeatureSchemaManifest.bidding_token_fields`:
    ``[bid_seat(3), bid_value(4), is_pass(1)]`` per row. ``num_bids`` is the
    real bid count (0 in legacy). This is the schema-bound counterpart of
    :class:`~douzero.observation.public.BiddingTokenBatch`; both are kept so
    the legacy-agnostic public container and the schema-described model input
    each have a single owner, and they are built consistently from the same
    bidding history.
    """

    tokens: np.ndarray
    num_bids: int

    def __post_init__(self) -> None:
        if self.tokens.flags.writeable:
            object.__setattr__(self, "tokens", _freeze(self.tokens))


@dataclass(frozen=True)
class ObservationV2:
    """Full V2 observation for one decision. PUBLIC ONLY.

    Holds the encoded state block (once), the legal-action batch, the
    history-token batch, the public bottom-card identity, the bidding tokens,
    the originating :class:`PublicObservation`, and the schema identity. No
    privileged field is present or reachable.

    Schema identity (item 2): ``feature_schema_version`` and
    ``feature_schema_hash`` bind this observation to the exact schema contract
    it was encoded against, so a checkpoint/model can reject a mismatch.

    Adapter support (item 7): ``card_play_action_seq`` stores the raw public
    action sequence (a tuple of tuples, immutable) so the legacy adapter can
    reconstruct the legacy ``z`` tensor from this container alone.
    """

    schema: FeatureSchemaManifest
    public: PublicObservation
    state: StateBlock
    actions: LegalActionBatch
    history: HistoryTokenBatch
    bidding_tokens: BiddingTokenBatch
    context: PublicContextBlock
    schema_bidding_tokens: SchemaBiddingTokenBatch
    feature_version: str
    feature_schema_version: str
    feature_schema_hash: str
    card_play_action_seq: tuple[tuple[int, ...], ...] = ()

    @property
    def is_privileged(self) -> bool:
        """Always False — an ObservationV2 is public-only by construction."""
        return False


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def _cards_left_onehot(count: int) -> np.ndarray:
    """One-hot encode a remaining-card count over ``MAX_CARDS_LEFT`` slots."""
    vec = np.zeros(MAX_CARDS_LEFT, dtype=np.int8)
    if count < 0 or count > MAX_CARDS_LEFT:
        raise ValueError(f"cards-left count {count} out of range [0, {MAX_CARDS_LEFT}]")
    if count >= 1:
        vec[count - 1] = 1
    return vec


def _bomb_onehot(bomb_num: int) -> np.ndarray:
    """One-hot encode the bomb counter over ``BOMB_ONEHOT_WIDTH`` slots."""
    vec = np.zeros(BOMB_ONEHOT_WIDTH, dtype=np.int8)
    idx = min(int(bomb_num), BOMB_ONEHOT_WIDTH - 1)
    vec[idx] = 1
    return vec


def _acting_role_onehot(role: str) -> np.ndarray:
    """One-hot encode the acting role over ``SEAT_ONEHOT_WIDTH`` slots."""
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
    action_seq: Sequence[Sequence[int]],
    num_cards_left_now: dict[str, int],
    phase_code: int,
) -> list[HistoryMove]:
    """Reconstruct the public action history as :class:`HistoryMove` tokens.

    Walks the public, ordered action list and assigns each action to its
    absolute actor using the canonical turn order. ``cards_left_after`` is
    reconstructed by replaying the hand counts backward from the current counts
    (fully determined by public information, so swap-invariant).
    """
    seq = [list(a) for a in (action_seq or [])]
    if not seq:
        return []

    counts_now = {role: num_cards_left_now.get(role, 0) for role in ALL_ROLES}
    counts_after: list[dict[str, int]] = [None] * len(seq)  # type: ignore[list-item]
    counts = dict(counts_now)
    for i in range(len(seq) - 1, -1, -1):
        counts_after[i] = dict(counts)
        action = seq[i]
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
    """Return the absolute role that acts at ``turn_index`` (0 = landlord lead)."""
    return ALL_ROLES[turn_index % len(ALL_ROLES)]


# --------------------------------------------------------------------------- #
# Public encoder
# --------------------------------------------------------------------------- #
def get_obs_v2(
    infoset,
    *,
    schema: FeatureSchemaManifest | None = None,
    ruleset: "RuleSet | None" = None,
    ruleset_id: str | None = None,
    ruleset_version: str | None = None,
    ruleset_hash: str | None = None,
    bid_value: int = 0,
    bidding_history=None,
    bidding_order=None,
    bomb_count: int | None = None,
    rocket_count: int | None = None,
    total_multiplier: int = 1,
    phase: str = "playing",
) -> ObservationV2:
    """Encode an infoset into a public :class:`ObservationV2`.

    Parameters
    ----------
    infoset
        The legacy ``InfoSet`` (or any object with the same public attributes).
        Privileged fields (``all_handcards``, ``other_hand_cards``) are IGNORED
        — the unseen pool is recomputed from public information, and the
        leakage test replaces ``all_handcards`` with an access-throws sentinel.
    schema
        Optional prebuilt schema (controls ``max_history_len`` and the identity
        hash). Defaults to :func:`build_v2_schema`.
    ruleset
        Preferred way to supply the public rule identity (item 3 / Blocker 3).
        When provided, ``ruleset_id`` / ``ruleset_version`` / ``ruleset_hash``
        are derived from it and the three-argument form MUST NOT be passed
        (passing both raises). The encoder never has to guess a hash.
    ruleset_id, ruleset_version, ruleset_hash
        Alternative (validated) rule-identity arguments. These are checked for
        consistency: legacy auto-fills the canonical legacy hash when the hash
        is omitted; standard requires version+hash and rejects ``standard`` +
        ``legacy-v1``; a provided hash that disagrees with the id/version is
        rejected. At least one of ``ruleset`` or ``ruleset_id`` must be supplied
        (omitting both defaults to legacy with the canonical hash).
    bid_value, bidding_history, bidding_order
        Public bidding state. In legacy mode these default to no bidding.
    bomb_count, rocket_count, total_multiplier
        Public multiplier state (item 3). ``rocket_count`` is separate from
        ``bomb_count`` (the legacy ``bomb_num`` conflates both).
    phase
        Game phase string ("bidding"/"playing"/"reveal_bottom").

    The state is encoded **once**; legal actions are encoded into a
    :class:`LegalActionBatch`; history is encoded into a bounded, padded
    :class:`HistoryTokenBatch`; the public-context block and schema bidding
    tokens are encoded once (item 3). No privileged data is read or returned.
    """
    if schema is None:
        schema = build_v2_schema()

    # Resolve the rule identity consistently (Blocker 3): never an empty hash.
    rs_id, rs_version, rs_hash = _resolve_rule_identity(
        ruleset, ruleset_id, ruleset_version, ruleset_hash
    )

    acting_role = infoset.player_position
    my_hand = list(infoset.player_hand_cards or [])
    played = {
        role: list((infoset.played_cards or {}).get(role, []))
        for role in ALL_ROLES
    }
    bottom_unplayed = list(infoset.three_landlord_cards or [])
    # Public revealed bottom identity (item 4). Prefer the explicit revealed
    # field; fall back to the current unplayed set.
    bottom_revealed = getattr(
        infoset, "three_landlord_cards_revealed", None)
    if bottom_revealed is None:
        bottom_revealed = bottom_unplayed
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
        three_landlord_cards_revealed=bottom_revealed,
        num_cards_left=num_left,
        legal_actions=infoset.legal_actions,
        phase=phase,
        ruleset_id=rs_id,
        ruleset_version=rs_version,
        ruleset_hash=rs_hash,
        bid_value=bid_value,
        bidding_history=bidding_history,
        bidding_order=bidding_order,
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
        features = np.zeros((0, _action_width(schema)), dtype=np.int8)
    action_mask = np.ones(features.shape[0], dtype=np.int8)
    actions = LegalActionBatch(
        features=features,
        action_mask=action_mask,
        legal_actions=tuple(tuple(sorted(a)) for a in legal),
    )

    # --- History batch ---
    phase_code = _phase_code(phase)
    raw_action_seq = [list(a) for a in (infoset.card_play_action_seq or [])]
    moves = _reconstruct_history_moves(raw_action_seq, num_left, phase_code)
    history = encode_history(moves, schema)

    # --- Bidding tokens (public container + schema-described model block) ---
    bidding_tokens = encode_bidding_tokens(bidding_history, bidding_order)
    schema_bidding_tokens = _encode_schema_bidding_tokens(
        bidding_history, bidding_order, schema)

    # --- Public context block (item 3, schema-described) ---
    context = _build_context_block(public, phase, schema)

    # Raw public action sequence (immutable) for the legacy adapter (item 7).
    action_seq_tuple = tuple(tuple(sorted(a)) for a in raw_action_seq)

    return ObservationV2(
        schema=schema,
        public=public,
        state=state,
        actions=actions,
        history=history,
        bidding_tokens=bidding_tokens,
        context=context,
        schema_bidding_tokens=schema_bidding_tokens,
        feature_version=schema.feature_version,
        feature_schema_version=schema.schema_version,
        feature_schema_hash=schema.stable_hash(),
        card_play_action_seq=action_seq_tuple,
    )


def encode_bidding_tokens(
    bidding_history, bidding_order
) -> BiddingTokenBatch:
    """Build the :class:`BiddingTokenBatch` from a bidding history."""
    from .public import encode_bidding_history
    return encode_bidding_history(bidding_history or (), bidding_order)


def _resolve_rule_identity(
    ruleset,
    ruleset_id: str | None,
    ruleset_version: str | None,
    ruleset_hash: str | None,
) -> tuple[str, str, str]:
    """Resolve a consistent (id, version, hash) rule identity (Blocker 3).

    Preferred path: pass a ``RuleSet``; its ``identity()`` is authoritative and
    the three-argument form must NOT also be passed. Fallback path: validate
    the three arguments for consistency (legacy auto-fills its canonical hash;
    standard requires version+hash and rejects ``standard``+``legacy-v1``; a
    disagreeing hash is rejected). The returned hash is NEVER empty.
    """
    from douzero.env.rules import RuleSet

    if ruleset is not None:
        if not isinstance(ruleset, RuleSet):
            raise TypeError(
                f"ruleset must be a RuleSet instance, got {type(ruleset).__name__}"
            )
        if any(v is not None for v in (ruleset_id, ruleset_version, ruleset_hash)):
            raise ValueError(
                "Pass EITHER ruleset= OR (ruleset_id/ruleset_version/"
                "ruleset_hash), not both. Mixing them is ambiguous."
            )
        ident = ruleset.identity()
        return ident["ruleset_id"], ident["ruleset_version"], ident["ruleset_hash"]

    # Fallback: derive/validate from the three-argument form.
    rid = ruleset_id if ruleset_id is not None else "legacy"
    if rid == "legacy":
        canonical = RuleSet.legacy()
        expected_version = canonical.ruleset_version
        expected_hash = canonical.stable_hash()
        if ruleset_version is not None and ruleset_version != expected_version:
            raise ValueError(
                f"legacy ruleset_version must be {expected_version!r}, got "
                f"{ruleset_version!r}"
            )
        if ruleset_hash is not None and ruleset_hash != expected_hash:
            raise ValueError(
                "ruleset_hash disagrees with the canonical legacy ruleset hash"
            )
        return rid, expected_version, expected_hash

    if rid == "standard":
        if ruleset_version is None or ruleset_hash is None:
            raise ValueError(
                "ruleset='standard' requires explicit ruleset_version AND "
                "ruleset_hash (derive them from RuleSet.standard().identity()). "
                "Pass ruleset=RuleSet.standard() to get them automatically."
            )
        canonical = RuleSet.standard()
        if ruleset_version != canonical.ruleset_version:
            raise ValueError(
                f"standard ruleset_version must be "
                f"{canonical.ruleset_version!r}, got {ruleset_version!r}"
            )
        if ruleset_hash != canonical.stable_hash():
            raise ValueError(
                "ruleset_hash disagrees with the canonical standard ruleset hash"
            )
        return rid, ruleset_version, ruleset_hash

    raise ValueError(
        f"ruleset_id must be 'legacy' or 'standard', got {rid!r}"
    )


def _phase_onehot(phase: str) -> np.ndarray:
    """One-hot encode the game phase over PHASE_ONEHOT_WIDTH slots."""
    from .schema import PHASE_ONEHOT_WIDTH
    vec = np.zeros(PHASE_ONEHOT_WIDTH, dtype=np.int8)
    mapping = {"bidding": 0, "reveal_bottom": 1, "playing": 2}
    idx = mapping.get(phase, 3)  # 3 = reserved/unknown
    vec[idx] = 1
    return vec


def _bid_value_onehot(value: int) -> np.ndarray:
    from .schema import BID_VALUE_ONEHOT_WIDTH
    vec = np.zeros(BID_VALUE_ONEHOT_WIDTH, dtype=np.int8)
    if 0 <= value < BID_VALUE_ONEHOT_WIDTH:
        vec[value] = 1
    return vec


def _ruleset_id_onehot(ruleset_id: str) -> np.ndarray:
    from .schema import RULESET_ID_ONEHOT_WIDTH
    vec = np.zeros(RULESET_ID_ONEHOT_WIDTH, dtype=np.int8)
    mapping = {"legacy": 0, "standard": 1}
    idx = mapping.get(ruleset_id)
    if idx is not None:
        vec[idx] = 1
    return vec


def _build_context_block(public, phase: str, schema) -> "PublicContextBlock":
    """Build the schema-described :class:`PublicContextBlock` (item 3)."""
    return PublicContextBlock(
        bottom_cards_revealed=cards_to_vector(public.bottom_cards.revealed),
        bottom_cards_unplayed=cards_to_vector(public.bottom_cards.unplayed),
        bid_value_onehot=_bid_value_onehot(public.bid_value),
        phase_onehot=_phase_onehot(phase),
        rocket_count=np.array([int(public.rocket_count)], dtype=np.int8),
        total_multiplier=np.array([int(public.total_multiplier)], dtype=np.int32),
        ruleset_id_onehot=_ruleset_id_onehot(public.ruleset_id),
    )


def _encode_schema_bidding_tokens(bidding_history, bidding_order, schema):
    """Build the schema-described :class:`SchemaBiddingTokenBatch` (item 3).

    Each row is ``[bid_seat(3), bid_value(4), is_pass(1)]`` matching
    :attr:`FeatureSchemaManifest.bidding_token_fields`.
    """
    history = list(bidding_history or [])
    seat_to_index: dict[str, int] = {}
    if bidding_order is not None:
        for i, seat in enumerate(bidding_order):
            seat_to_index[str(seat)] = i
    rows = []
    next_index = 0
    for seat, value in history:
        seat_str = str(seat)
        if seat_str not in seat_to_index:
            seat_to_index[seat_str] = next_index
            next_index += 1
        idx = seat_to_index[seat_str]
        seat_oh = np.zeros(3, dtype=np.int8)
        if 0 <= idx < 3:
            seat_oh[idx] = 1
        rows.append(np.concatenate([
            seat_oh,
            _bid_value_onehot(int(value)),
            np.array([1 if int(value) == 0 else 0], dtype=np.int8),
        ]).astype(np.int8))
    if rows:
        tokens = np.stack(rows, axis=0).astype(np.int8)
    else:
        tokens = np.zeros((0, 8), dtype=np.int8)  # 3 + 4 + 1
    return SchemaBiddingTokenBatch(tokens=tokens, num_bids=len(rows))


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
