"""Public observation V2 (P03, hardened round 2).

The public observation is the ONLY thing a deployment model may see. It is
built entirely from public information:

- the acting role and the relative-seat context,
- the acting player's own hand,
- public bottom cards (revealed identity + current unplayed subset, both with
  documented landlord ownership),
- bidding history (as a token batch) and the final bid,
- each role's remaining-card count,
- each role's cumulative played cards,
- the complete public action history,
- the current move to beat (the last valid action),
- public bomb / rocket / multiplier / phase / rule-identity state,
- the current legal actions.

Deep immutability (item 5): the public container is ``frozen`` + ``slots``;
caller-supplied lists/dicts are copied; numpy arrays are frozen
(``write=False``); no mutable dict/list is exposed by reference; the public
container shares no ndarray with any privileged container.

It MUST NOT contain true hidden hands, ``all_handcards``, or
``other_hand_cards``-as-true-allocation. The unseen-card pool here is the
*public* union of cards not in the acting hand and not publicly accounted for —
it is swap-invariant (two hidden allocations with the same public footprint
produce the same pool), which is the property the leakage test enforces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .cards import DECK
from .seats import seats_from

#: Marker stamped on every public container so a guard can accept it without
#: introspection (the counterpart to ``privileged.PRIVILEGED_KIND``).
PUBLIC_KIND: str = "public"

#: Maximum number of bidders (three-player DouDizhu).
MAX_BIDDERS: int = 3

#: One-hot width for a bid value. The standard ruleset uses 0/1/2/3 (4 values);
#: we reserve a little headroom without exploding the width.
BID_VALUE_ONEHOT_WIDTH: int = 4

#: One-hot width for the bidder seat index (0/1/2).
BID_SEAT_ONEHOT_WIDTH: int = MAX_BIDDERS

#: Width of one bidding token: seat one-hot + bid-value one-hot + is_pass flag.
BIDDING_TOKEN_WIDTH: int = BID_SEAT_ONEHOT_WIDTH + BID_VALUE_ONEHOT_WIDTH + 1


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a read-only int8 numpy array (copies if needed)."""
    out = np.asarray(arr, dtype=np.int8)
    if out.flags.writeable:
        out = out.copy()
    out.setflags(write=False)
    return out


def _bid_value_onehot(value: int) -> np.ndarray:
    vec = np.zeros(BID_VALUE_ONEHOT_WIDTH, dtype=np.int8)
    if 0 <= value < BID_VALUE_ONEHOT_WIDTH:
        vec[value] = 1
    return vec


def _bid_seat_onehot(seat_index: int) -> np.ndarray:
    vec = np.zeros(BID_SEAT_ONEHOT_WIDTH, dtype=np.int8)
    if 0 <= seat_index < BID_SEAT_ONEHOT_WIDTH:
        vec[seat_index] = 1
    return vec


@dataclass(frozen=True)
class BiddingTokenBatch:
    """Encoded bidding history as a fixed-width token tensor.

    ``tokens`` has shape ``(num_bids, BIDDING_TOKEN_WIDTH)`` int8 (read-only),
    one row per bid in chronological order. Each row is
    ``[seat_one_hot(3), bid_value_one_hot(4), is_pass(1)]``.

    ``num_bids`` is the number of real bids (0 in legacy mode, which has no
    bidding). There is no padding dimension because the bid count is small and
    bounded by :data:`MAX_BIDDERS` (plus redeals); a model consumes the
    variable length via ``num_bids``.

    Deep immutability: ``tokens`` is frozen (``write=False``).
    """

    tokens: np.ndarray
    num_bids: int

    def __post_init__(self) -> None:
        if self.tokens.shape != (self.num_bids, BIDDING_TOKEN_WIDTH):
            raise ValueError(
                f"bidding tokens shape {self.tokens.shape} != "
                f"({self.num_bids}, {BIDDING_TOKEN_WIDTH})"
            )
        if self.tokens.flags.writeable:
            object.__setattr__(self, "tokens", _freeze_array(self.tokens))


def encode_bidding_history(
    bidding_history: Sequence[tuple[str | int, int]],
    bidding_order: Sequence[str] | None = None,
) -> BiddingTokenBatch:
    """Encode a bidding history into a :class:`BiddingTokenBatch`.

    ``bidding_history`` is a sequence of ``(seat, bid_value)`` pairs (the
    ``GameEnv.bidding_history`` shape). ``bidding_order`` maps a seat to its
    index; when omitted, seats are indexed by first appearance.
    """
    seat_to_index: dict[str, int] = {}
    if bidding_order is not None:
        for i, seat in enumerate(bidding_order):
            seat_to_index[str(seat)] = i
    history = list(bidding_history or [])
    rows = []
    next_index = 0
    for seat, value in history:
        seat_str = str(seat)
        if seat_str not in seat_to_index:
            seat_to_index[seat_str] = next_index
            next_index += 1
        idx = seat_to_index[seat_str]
        rows.append(np.concatenate([
            _bid_seat_onehot(idx),
            _bid_value_onehot(int(value)),
            np.array([1 if int(value) == 0 else 0], dtype=np.int8),
        ]))
    if rows:
        tokens = np.stack(rows, axis=0).astype(np.int8)
    else:
        tokens = np.zeros((0, BIDDING_TOKEN_WIDTH), dtype=np.int8)
    return BiddingTokenBatch(tokens=tokens, num_bids=len(rows))


@dataclass(frozen=True)
class PublicBottomCards:
    """The three revealed bottom cards and their public ownership.

    ``revealed`` is the ORIGINAL entity-identity list of the three bottom cards
    (public once the landlord is determined; never mutated after reveal).
    ``unplayed`` is the current subset NOT yet played by the landlord (tracked
    on a first-match removal basis — see module doc). ``all_played`` is True
    once the landlord has played all three. The bottom cards always belong to
    the landlord; ``owner`` records that fact.

    Deep immutability: the tuples are immutable by construction; the container
    is frozen+slots.
    """

    revealed: tuple[int, ...]
    unplayed: tuple[int, ...]
    all_played: bool
    owner: str = "landlord"

    def to_dict(self) -> dict[str, Any]:
        return {
            "revealed": list(self.revealed),
            "unplayed": list(self.unplayed),
            "all_played": self.all_played,
            "owner": self.owner,
        }


@dataclass(frozen=True, slots=True)
class PublicObservation:
    """Everything a deployment model is allowed to see for one decision.

    All fields are public. ``kind`` is always :data:`PUBLIC_KIND`. The unseen
    card pool (``other_handcards``) is the public union of cards the acting
    player cannot account for from its own hand and the public history; it is
    invariant under any re-allocation of hidden cards among the opponents.

    Cardinality / conservation invariant (checked in tests):

    ``my_handcards ∪ other_handcards ∪ all_played == DECK``

    (the bottom cards are inside the landlord's hand for the landlord, or
    inside the opponent pool for a farmer, so they are not added separately).

    Deep immutability (item 5): ``frozen`` + ``slots``. Caller-supplied
    lists/dicts are copied at construction (see
    :func:`build_public_observation`). Every mapping field is exposed as a
    read-only :class:`types.MappingProxyType`, so ``obs.played_cards[...] = x``
    raises ``TypeError``; the encoded tensor batches hold frozen numpy arrays.
    No mutable dict/list field is exposed by reference.

    Model-consumable public input (item 3): in addition to the legacy
    hand/played/last-move features, this carries the public bottom-card
    identity (revealed + unplayed), the final bid, the bidding history (as a
    token batch), the game phase, the rocket count, the total multiplier, and
    the rule identity — everything a V2 model needs without reaching into
    ``obs.public`` ad-hoc (item 3 forbids P05 from bypassing the schema).
    """

    # Identity
    acting_role: str
    seat_context: Mapping[str, str]  # read-only {absolute_role: relative_seat}
    kind: str = field(default=PUBLIC_KIND, init=False)

    # Phase / ruleset public state
    phase: str = "playing"
    ruleset_id: str = "legacy"
    ruleset_version: str = "legacy-v1"
    ruleset_hash: str = ""
    bid_value: int = 0
    bomb_count: int = 0
    rocket_count: int = 0
    total_multiplier: int = 1

    # Cards (public)
    my_handcards: tuple[int, ...] = ()
    other_handcards: tuple[int, ...] = ()  # public unseen pool (swap-invariant)
    played_cards: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    # Public count of non-pass card-play actions per role. Unlike
    # ``played_cards`` this counts moves, not card quantity, and is the
    # authoritative public input for spring/anti-spring features.
    non_pass_action_counts: Mapping[str, int] = field(default_factory=dict)
    last_move: tuple[int, ...] = ()
    last_move_dict: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    # Complete public card-play sequence. P11 consumes this only behind the
    # optional style flag; it contains no player identity or hidden cards.
    action_history: tuple[tuple[int, ...], ...] = ()
    bottom_cards: PublicBottomCards = field(
        default_factory=lambda: PublicBottomCards((), (), True)
    )
    num_cards_left: Mapping[str, int] = field(default_factory=dict)

    # Bidding history (encoded token batch; empty in legacy mode).
    bidding_history: tuple[tuple[str, int], ...] = ()
    bidding_tokens: BiddingTokenBatch = field(
        default_factory=lambda: BiddingTokenBatch(
            tokens=np.zeros((0, BIDDING_TOKEN_WIDTH), dtype=np.int8), num_bids=0)
    )

    # Legal actions (the candidate moves; public)
    legal_actions: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        # Wrap every mapping field in a read-only MappingProxyType so the
        # container is deeply immutable (frozen only blocks field reassignment,
        # not in-place mutation of an exposed dict — item 5).
        object.__setattr__(self, "seat_context",
                           MappingProxyType(dict(self.seat_context)))
        object.__setattr__(self, "played_cards",
                           MappingProxyType(dict(self.played_cards)))
        object.__setattr__(
            self,
            "non_pass_action_counts",
            MappingProxyType(dict(self.non_pass_action_counts)),
        )
        object.__setattr__(self, "last_move_dict",
                           MappingProxyType(dict(self.last_move_dict)))
        object.__setattr__(self, "num_cards_left",
                           MappingProxyType(dict(self.num_cards_left)))
        if self.kind != PUBLIC_KIND:  # defensive; default already set
            object.__setattr__(self, "kind", PUBLIC_KIND)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (no numpy arrays).

        Used for serialisation-stability tests and for stamping a compact
        representation into logs. Deliberately omits the encoded tensor batches
        (cards vectors, bidding tokens) — those live only in the encoded
        tensors. No hidden-hand field is ever present (item 8).
        """
        return {
            "kind": self.kind,
            "acting_role": self.acting_role,
            "seat_context": dict(self.seat_context),
            "phase": self.phase,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_hash": self.ruleset_hash,
            "bid_value": self.bid_value,
            "bomb_count": self.bomb_count,
            "rocket_count": self.rocket_count,
            "total_multiplier": self.total_multiplier,
            "my_handcards": list(self.my_handcards),
            "other_handcards": list(self.other_handcards),
            "played_cards": {k: list(v) for k, v in self.played_cards.items()},
            "non_pass_action_counts": dict(self.non_pass_action_counts),
            "last_move": list(self.last_move),
            "last_move_dict": {k: list(v) for k, v in self.last_move_dict.items()},
            "bottom_cards": self.bottom_cards.to_dict(),
            "num_cards_left": dict(self.num_cards_left),
            "bidding_history": [list(b) for b in self.bidding_history],
            "legal_actions": [list(a) for a in self.legal_actions],
        }


def _safe_tuple(value: Any) -> tuple[int, ...]:
    """Coerce a list of ints (or None) into a sorted-int tuple (a copy)."""
    if value is None:
        return ()
    return tuple(sorted(int(c) for c in value))


def build_public_observation(
    *,
    acting_role: str,
    my_handcards,
    other_handcards,
    played_cards: dict[str, list[int]],
    non_pass_action_counts: dict[str, int] | None = None,
    last_move,
    last_move_dict: dict[str, list[int]],
    three_landlord_cards,
    three_landlord_cards_revealed=None,
    num_cards_left: dict[str, int],
    legal_actions,
    action_history=None,
    phase: str = "playing",
    ruleset_id: str = "legacy",
    ruleset_version: str = "legacy-v1",
    ruleset_hash: str = "",
    bid_value: int = 0,
    bidding_history=None,
    bidding_order=None,
    bomb_count: int = 0,
    rocket_count: int = 0,
    total_multiplier: int = 1,
) -> PublicObservation:
    """Assemble a :class:`PublicObservation` from raw game state.

    Copies every caller-supplied list/dict so the returned container is deep
    -immutable and mutation of the source cannot affect it (item 5).

    Public bottom-card semantics (item 4):

    - ``revealed`` = the ORIGINAL three bottom cards
      (``three_landlord_cards_revealed``, or ``three_landlord_cards`` when the
      original is unavailable — backward compatible). Never mutated after
      reveal.
    - ``unplayed`` = the current subset not yet played by the landlord
      (``three_landlord_cards``, which GameEnv.step reduces as the landlord
      plays them).
    """
    seat_context = seats_from(acting_role)

    # Bottom-card identity: prefer the explicit "revealed" original; fall back
    # to the current three_landlord_cards for backward compatibility.
    revealed_src = three_landlord_cards_revealed
    if revealed_src is None:
        revealed_src = three_landlord_cards
    revealed = tuple(sorted(int(c) for c in (revealed_src or ())))
    unplayed = tuple(sorted(int(c) for c in (three_landlord_cards or ())))
    bottom = PublicBottomCards(
        revealed=revealed,
        unplayed=unplayed,
        all_played=(len(unplayed) == 0),
        owner="landlord",
    )

    # Encode the bidding history into an immutable token batch.
    bidding_tokens = encode_bidding_history(bidding_history or (), bidding_order)

    # Deep-copy all caller-supplied mutable inputs.
    played_tuples = {
        role: _safe_tuple(cards)
        for role, cards in (played_cards or {}).items()
    }
    action_counts = {
        role: int(count)
        for role, count in (non_pass_action_counts or {}).items()
    }
    legal_tuples = tuple(
        tuple(sorted(int(c) for c in a)) for a in (legal_actions or [])
    )
    last_move_t = _safe_tuple(last_move)
    last_move_d = {
        k: _safe_tuple(v) for k, v in (last_move_dict or {}).items()
    }
    action_history_t = tuple(
        _safe_tuple(action) for action in (action_history or ())
    )
    num_left = {k: int(v) for k, v in (num_cards_left or {}).items()}
    bidding_hist_t = tuple(
        (str(pos), int(val)) for pos, val in (bidding_history or ())
    )

    return PublicObservation(
        acting_role=acting_role,
        seat_context=seat_context,
        phase=phase,
        ruleset_id=ruleset_id,
        ruleset_version=ruleset_version,
        ruleset_hash=ruleset_hash,
        bid_value=int(bid_value),
        bomb_count=int(bomb_count),
        rocket_count=int(rocket_count),
        total_multiplier=int(total_multiplier),
        my_handcards=_safe_tuple(my_handcards),
        other_handcards=_safe_tuple(other_handcards),
        played_cards=played_tuples,
        non_pass_action_counts=action_counts,
        last_move=last_move_t,
        last_move_dict=last_move_d,
        action_history=action_history_t,
        bottom_cards=bottom,
        num_cards_left=num_left,
        bidding_history=bidding_hist_t,
        bidding_tokens=bidding_tokens,
        legal_actions=legal_tuples,
    )


def compute_unseen_pool(
    my_handcards,
    played_cards: dict[str, list[int]],
    bottom_unplayed=None,
) -> tuple[int, ...]:
    """Compute the public unseen-card pool from public information only.

    The parity pool = full deck − (my hand) − (all publicly played cards). This
    reproduces the legacy ``other_hand_cards`` union exactly:

    - For the landlord: the 34 farmer cards.
    - For a farmer: the 37 cards held by the landlord + teammate (the landlord's
      3 public bottom cards are included because they are part of the landlord's
      holdings, which the farmer cannot distinguish from the landlord's hidden
      cards at the feature level).

    ``bottom_unplayed`` is accepted (for call-site compatibility) but ignored;
    use :func:`compute_belief_unknown_pool` for the belief-model pool that
    excludes public bottom cards.
    """
    del bottom_unplayed  # accepted for compatibility; see docstring
    full = Counter(DECK)
    full.subtract(Counter(int(c) for c in (my_handcards or [])))
    for cards in (played_cards or {}).values():
        full.subtract(Counter(int(c) for c in (cards or [])))
    pool: list[int] = []
    for card, count in full.items():
        if count < 0:
            raise ValueError(
                f"Card conservation violated: rank {card} has negative count "
                f"{count} in the unseen pool"
            )
        pool.extend([card] * count)
    return tuple(sorted(pool))


def compute_belief_unknown_pool(
    my_handcards,
    played_cards: dict[str, list[int]],
    bottom_unplayed,
    *,
    acting_role: str,
) -> tuple[int, ...]:
    """Compute the belief-model unknown pool (excludes public bottom cards).

    This is the pool a hidden-hand belief model (P07) must distribute among the
    two opponents. It is strictly smaller than :func:`compute_unseen_pool` for a
    farmer, because the farmer knows the 3 unplayed public bottom cards belong
    to the landlord and therefore they are not "unknown".

    For the landlord, the bottom cards are already part of its own hand, so the
    belief pool equals the parity pool.
    """
    parity = compute_unseen_pool(my_handcards, played_cards)
    from .seats import is_landlord
    if is_landlord(acting_role):
        return parity
    pool_counter = Counter(parity)
    pool_counter.subtract(Counter(int(c) for c in (bottom_unplayed or [])))
    out: list[int] = []
    for card, count in sorted(pool_counter.items()):
        if count < 0:
            raise ValueError(
                f"Belief pool conservation violated: rank {card} has negative "
                f"count {count}; bottom cards not a subset of the parity pool"
            )
        out.extend([card] * count)
    return tuple(out)
