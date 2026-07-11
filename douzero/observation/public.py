"""Public observation V2 (P03).

The public observation is the ONLY thing a deployment model may see. It is
built entirely from public information:

- the acting role and the relative-seat context,
- the acting player's own hand,
- public bottom cards and their played/unplayed status (with documented
  ownership — the bottom cards always belong to the landlord),
- bidding history and the final bid,
- each role's remaining-card count,
- each role's cumulative played cards,
- the complete public action history,
- the current move to beat (the last valid action),
- public bomb / multiplier / rule state,
- the current legal actions.

It MUST NOT contain true hidden hands, ``all_handcards``, or
``other_hand_cards``-as-true-allocation. The unseen-card pool here is the
*public* union of cards not in the acting hand and not publicly accounted for —
it is swap-invariant (two hidden allocations with the same public footprint
produce the same pool), which is the property the leakage test enforces.

Public bottom-card handling (P03 spec point 6):

- Farmers know the bottom cards belong to the landlord.
- Unplayed public bottom cards are NOT part of the unknown-card pool; they are
  accounted as landlord-owned.
- When the landlord plays a card whose rank matches a bottom card, the entity
  identity cannot be distinguished, so we record a consistent ownership policy:
  bottom cards are removed from the unplayed set on a first-match basis (this
  mirrors the legacy ``GameEnv.step`` removal order and is verifiable).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .cards import DECK, cards_to_vector
from .seats import seats_from

#: Marker stamped on every public container so a guard can accept it without
#: introspection (the counterpart to ``privileged.PRIVILEGED_KIND``).
PUBLIC_KIND: str = "public"


@dataclass(frozen=True)
class PublicBottomCards:
    """The three revealed bottom cards and their public ownership.

    ``revealed`` is the entity-identity list of the three bottom cards (public
    once the landlord is determined). ``unplayed`` is the subset not yet played
    by the landlord (tracked on a first-match removal basis — see module doc).
    ``all_played`` is True once the landlord has played all three. The bottom
    cards always belong to the landlord; ``owner`` records that fact.

    In legacy mode there is no bidding, but ``three_landlord_cards`` is still
    public (it is the 3-card slice the landlord received). We treat it the same
    way: revealed from the start, owned by the landlord.
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


@dataclass(frozen=True)
class PublicObservation:
    """Everything a deployment model is allowed to see for one decision.

    All fields are public. ``kind`` is always :data:`PUBLIC_KIND`. The unseen
    card pool (``other_handcards``) is the public union of cards the acting
    player cannot account for from its own hand and the public history; it is
    invariant under any re-allocation of hidden cards among the opponents.

    Cardinality / conservation invariants (checked in tests):

    ``my_handcards ∪ other_handcards ∪ all_played ∪ bottom_unplayed == DECK``

    where ``all_played`` is the union of the three roles' cumulative played
    cards and ``bottom_unplayed`` are the bottom cards not yet played by the
    landlord.
    """

    # Identity
    acting_role: str
    seat_context: dict[str, str]  # {absolute_role: relative_seat}
    kind: str = field(default=PUBLIC_KIND, init=False)

    # Phase / ruleset public state
    phase: str = "playing"
    ruleset_id: str = "legacy"
    bid_value: int = 0
    bidding_history: tuple[tuple[str, int], ...] = ()
    bomb_count: int = 0
    rocket_count: int = 0
    total_multiplier: int = 1

    # Cards (public)
    my_handcards: tuple[int, ...] = ()
    other_handcards: tuple[int, ...] = ()  # public unseen pool (swap-invariant)
    played_cards: dict[str, tuple[int, ...]] = field(default_factory=dict)
    last_move: tuple[int, ...] = ()
    last_move_dict: dict[str, tuple[int, ...]] = field(default_factory=dict)
    bottom_cards: PublicBottomCards = field(
        default_factory=lambda: PublicBottomCards((), (), True)
    )
    num_cards_left: dict[str, int] = field(default_factory=dict)

    # Legal actions (the candidate moves; public)
    legal_actions: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.kind != PUBLIC_KIND:  # defensive; default already set
            object.__setattr__(self, "kind", PUBLIC_KIND)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (no numpy arrays).

        Used for serialisation-stability tests and for stamping a compact
        representation into logs. Card vectors are NOT included here; they live
        only in the encoded tensors produced by the encoder.
        """
        return {
            "kind": self.kind,
            "acting_role": self.acting_role,
            "seat_context": dict(self.seat_context),
            "phase": self.phase,
            "ruleset_id": self.ruleset_id,
            "bid_value": self.bid_value,
            "bidding_history": [list(b) for b in self.bidding_history],
            "bomb_count": self.bomb_count,
            "rocket_count": self.rocket_count,
            "total_multiplier": self.total_multiplier,
            "my_handcards": list(self.my_handcards),
            "other_handcards": list(self.other_handcards),
            "played_cards": {k: list(v) for k, v in self.played_cards.items()},
            "last_move": list(self.last_move),
            "last_move_dict": {k: list(v) for k, v in self.last_move_dict.items()},
            "bottom_cards": self.bottom_cards.to_dict(),
            "num_cards_left": dict(self.num_cards_left),
            "legal_actions": [list(a) for a in self.legal_actions],
        }


def _safe_tuple(value: Any) -> tuple[int, ...]:
    """Coerce a list of ints (or None) into a sorted-int tuple."""
    if value is None:
        return ()
    return tuple(sorted(int(c) for c in value))


def build_public_observation(
    *,
    acting_role: str,
    my_handcards,
    other_handcards,
    played_cards: dict[str, list[int]],
    last_move,
    last_move_dict: dict[str, list[int]],
    three_landlord_cards,
    num_cards_left: dict[str, int],
    legal_actions,
    phase: str = "playing",
    ruleset_id: str = "legacy",
    bid_value: int = 0,
    bidding_history=None,
    bomb_count: int = 0,
    rocket_count: int = 0,
    total_multiplier: int = 1,
) -> PublicObservation:
    """Assemble a :class:`PublicObservation` from raw game state.

    This is a thin, validation-light constructor used by the encoder. It
    coerces lists to sorted tuples and builds the seat context from the acting
    role. The unseen-card pool (``other_handcards``) must already be the public
    swap-invariant union (the encoder computes it from the infoset).
    """
    seat_context = seats_from(acting_role)

    played_tuples = {role: _safe_tuple(cards) for role, cards in played_cards.items()}
    bottom_revealed = tuple(sorted(int(c) for c in (three_landlord_cards or ())))
    # The unplayed bottom cards are those not yet consumed by the landlord's
    # plays. The caller passes the *current* three_landlord_cards (already
    # reduced by GameEnv.step as the landlord plays them), so the unplayed set
    # is exactly its current contents. all_played is True when it is empty.
    bottom_unplayed = tuple(sorted(int(c) for c in (three_landlord_cards or ())))
    bottom = PublicBottomCards(
        revealed=bottom_revealed,
        unplayed=bottom_unplayed,
        all_played=(len(bottom_unplayed) == 0),
        owner="landlord",
    )

    legal_tuples = tuple(tuple(sorted(int(c) for c in a)) for a in legal_actions)

    return PublicObservation(
        acting_role=acting_role,
        seat_context=seat_context,
        phase=phase,
        ruleset_id=ruleset_id,
        bid_value=int(bid_value),
        bidding_history=tuple(
            (str(pos), int(val)) for pos, val in (bidding_history or ())
        ),
        bomb_count=int(bomb_count),
        rocket_count=int(rocket_count),
        total_multiplier=int(total_multiplier),
        my_handcards=_safe_tuple(my_handcards),
        other_handcards=_safe_tuple(other_handcards),
        played_cards=played_tuples,
        last_move=_safe_tuple(last_move),
        last_move_dict={k: _safe_tuple(v) for k, v in last_move_dict.items()},
        bottom_cards=bottom,
        num_cards_left={k: int(v) for k, v in num_cards_left.items()},
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

    The bottom cards are NOT subtracted here because they are already accounted
    for in the acting player's hand (the landlord's 20 includes the bottom) or
    in the opponent pool (for a farmer). This keeps the pool swap-invariant and
    byte-identical to the legacy ``other_hand_cards``.

    ``bottom_unplayed`` is accepted (for call-site compatibility) but ignored;
    use :func:`compute_belief_unknown_pool` for the belief-model pool that
    excludes public bottom cards.

    This is purely public: it depends only on the acting hand and the cumulative
    played cards. Two hidden allocations with the same public footprint yield
    the same pool — the swap-invariance the leakage test checks.
    """
    del bottom_unplayed  # accepted for compatibility; see docstring
    full = Counter(DECK)
    full.subtract(Counter(int(c) for c in (my_handcards or [])))
    for cards in played_cards.values():
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

    P03 spec point 6: "unplayed public bottom cards must not enter the
    unknown-card pool".

    For the landlord, the bottom cards are already part of its own hand, so the
    belief pool equals the parity pool (the bottom is not double-counted).
    """
    parity = compute_unseen_pool(my_handcards, played_cards)
    from .seats import is_landlord
    if is_landlord(acting_role):
        # The bottom cards are in the landlord's own hand; nothing to exclude.
        return parity
    # Farmer: remove the public bottom cards from the unknown pool.
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
