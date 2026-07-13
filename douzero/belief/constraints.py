"""Card-conservation constraints for the joint hidden-hand belief model (P07).

This module is the single source of truth for:

- the canonical 15-rank belief category set (13 numeric + 2 jokers),
- the canonical opponent A (the NEXT seat) the model predicts, and opponent B
  (PREVIOUS seat) derived by subtraction,
- the legal-count mask ``[15, 5]`` that zeros out impossible (rank, count)
  slots before softmax, and
- pure helpers that turn a hand (or the public unseen pool) into the per-rank
  count vectors the DP, the loss, and the label builder all consume.

Everything here is public-information-only and deterministic; it never reads
true hidden hands. The true hidden allocation enters only through
:mod:`douzero.belief.labels` (privileged, training-only).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np

from douzero.observation.cards import (
    BIG_JOKER,
    JOKERS,
    NUMERIC_RANKS,
    SMALL_JOKER,
)
from douzero.observation.seats import (
    LANDLORD_ROLE,
    next_seat,
    previous_seat,
)

# --------------------------------------------------------------------------- #
# Canonical rank categories (15 = 13 numeric + 2 jokers)
# --------------------------------------------------------------------------- #
#: The 15 belief rank categories in fixed canonical order: the 13 numeric ranks
#: (3..14, 17 — rank 16 unused) followed by (small joker, big joker). This is
#: the concatenation of :data:`douzero.observation.cards.NUMERIC_RANKS` and
#: :data:`douzero.observation.cards.JOKERS`, kept here so the belief package is
#: self-contained and so logits/labels/masks share one index space.
BELIEF_RANKS: tuple[int, ...] = tuple(NUMERIC_RANKS) + tuple(JOKERS)  # 15 entries

#: Number of belief rank categories.
NUM_BELIEF_RANKS: int = len(BELIEF_RANKS)  # 15

#: Number of count slots per rank in the ``[15, 5]`` logit/mask tensor. Numeric
#: ranks use slots 0..4; joker slots 2..4 are always masked (a joker appears at
#: most once). Keeping one uniform width simplifies the tensor contract; the
#: legal mask encodes the per-rank cap.
NUM_COUNT_SLOTS: int = 5

#: Maximum multiplicity of a numeric rank (four copies in a deck).
NUMERIC_MAX_COUNT: int = 4

#: Maximum multiplicity of a joker (one small + one big, each at most once).
JOKER_MAX_COUNT: int = 1

#: Maximum cards a single opponent can hold (a farmer starts with 17; the
#: landlord with 20). Used only to size count-left one-hot vectors.
MAX_OPPONENT_CARDS: int = 20

#: Map a card code point to its belief-rank index (0..14). Raises for an
#: unknown card so a malformed input fails at the boundary.
BELIEF_RANK_INDEX: dict[int, int] = {rank: i for i, rank in enumerate(BELIEF_RANKS)}


def _rank_index(card: int) -> int:
    """Return the belief-rank index of ``card``, raising on unknown cards."""
    try:
        return BELIEF_RANK_INDEX[int(card)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown card code point {card!r}; not in BELIEF_RANKS {BELIEF_RANKS}"
        ) from exc


def _max_count_for_rank_index(rank_index: int) -> int:
    """Return the per-rank multiplicity cap (4 for numeric, 1 for joker)."""
    if rank_index < len(NUMERIC_RANKS):
        return NUMERIC_MAX_COUNT
    return JOKER_MAX_COUNT


def is_joker_rank(rank_index: int) -> bool:
    """Return True if ``rank_index`` is a joker slot (small/big)."""
    return rank_index >= len(NUMERIC_RANKS)


# --------------------------------------------------------------------------- #
# Canonical opponents
# --------------------------------------------------------------------------- #
def canonical_opponent(acting_role: str) -> str:
    """Return the canonical opponent A role (the NEXT seat, clockwise).

    Opponent A is the prediction target. Choosing a fixed canonical seat keeps
    the belief output's semantics stable across roles: index ``i`` always
    describes "the next player to act after me", regardless of whether the
    actor is the landlord or a farmer.
    """
    return next_seat(acting_role)


def canonical_opponent_b(acting_role: str) -> str:
    """Return the canonical opponent B role (the PREVIOUS seat).

    Opponent B's allocation is fully determined by subtraction:
    ``count_B[rank] = unseen_count[rank] - count_A[rank]``.
    """
    return previous_seat(acting_role)


def opponent_cards_left(
    num_cards_left: Mapping[str, int], opponent_role: str
) -> int:
    """Return a role's public remaining-card count, defaulting to 0.

    Accepts the ``PublicObservation.num_cards_left`` mapping directly. A
    missing role yields 0 (defensive; the public observation always carries all
    three roles during play).
    """
    if not isinstance(num_cards_left, Mapping):
        raise TypeError(
            f"num_cards_left must be a mapping, got {type(num_cards_left).__name__}"
        )
    return int(num_cards_left.get(opponent_role, 0))


def opponent_unknown_total(
    num_cards_left: Mapping[str, int],
    opponent_role: str,
    bottom_unplayed,
) -> int:
    """Return opponent ``opponent_role``'s *predictable* (hidden) card count.

    The belief model distributes the **unknown** pool between the two
    opponents. The public ``num_cards_left`` counts a player's FULL hand, but
    the unplayed public bottom cards are *known* landlord property — they are
    not part of the unknown pool and must not be predicted. So when the
    opponent is the landlord, its predictable total is its public remaining
    count MINUS the number of unplayed public bottom cards.

    For a non-landlord opponent (or when there are no unplayed bottom cards)
    this equals :func:`opponent_cards_left`. The result is clamped at 0.

    This is the total constraint the dynamic program enforces
    (``sum(count_A) == opponent_unknown_total(A)``).
    """
    total = opponent_cards_left(num_cards_left, opponent_role)
    if opponent_role == LANDLORD_ROLE:
        total -= int(len(list(bottom_unplayed or ())))
    return max(0, total)


# --------------------------------------------------------------------------- #
# Per-rank count vectors
# --------------------------------------------------------------------------- #
def per_rank_counts(cards: Iterable[int]) -> np.ndarray:
    """Return a ``(15,)`` int array of per-rank counts for ``cards``.

    ``cards`` is any iterable of card code points. The result is indexed by
    :data:`BELIEF_RANK_INDEX`. Used to encode the unseen pool, the public
    played cards, and (via :mod:`labels`) the true hidden allocation.
    """
    counts = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
    counter = Counter(int(c) for c in (cards or ()))
    for card, n in counter.items():
        counts[_rank_index(card)] += n
    return counts


def per_rank_counts_from_hand(cards: Iterable[int]) -> np.ndarray:
    """Alias for :func:`per_rank_counts` (semantically: a player's hand)."""
    return per_rank_counts(cards)


def unseen_counts_per_rank(unknown_pool: Iterable[int]) -> np.ndarray:
    """Return the per-rank counts of the belief unknown pool.

    ``unknown_pool`` is the public pool of cards held by the two opponents,
    i.e. the output of
    :func:`douzero.observation.public.compute_belief_unknown_pool`. It already
    excludes the public unplayed bottom cards for farmers, so the constraint
    ``sum(counts) == opponent_A + opponent_B`` holds by construction.
    """
    return per_rank_counts(unknown_pool)


# --------------------------------------------------------------------------- #
# Legal mask
# --------------------------------------------------------------------------- #
def legal_mask(unseen_counts: np.ndarray) -> np.ndarray:
    """Build the ``[15, 5]`` boolean legal-count mask for opponent A.

    Slot ``(r, k)`` is True iff opponent A may legally hold ``k`` copies of
    rank ``r``:

    - ``k <= unseen_counts[r]`` (cannot hold more than exist in the unknown
      pool), and
    - ``k <= max_count(r)`` (4 for numeric ranks, 1 for jokers).

    The mask is applied to logits (illegal slots set to ``-inf``) before any
    softmax/DP, guaranteeing every decoded or sampled allocation is
    card-conservative at the per-rank level. The total-count constraint
    (``sum == opponent_A_cards_left``) is enforced separately by the dynamic
    program.

    Parameters
    ----------
    unseen_counts:
        A length-15 int array (e.g. from :func:`unseen_counts_per_rank`).

    Returns
    -------
    numpy.ndarray
        ``dtype=bool``, shape ``(15, 5)``.
    """
    unseen = np.asarray(unseen_counts, dtype=np.int64)
    if unseen.shape != (NUM_BELIEF_RANKS,):
        raise ValueError(
            f"unseen_counts must have shape ({NUM_BELIEF_RANKS},), got {unseen.shape}"
        )
    if np.any(unseen < 0):
        raise ValueError(
            f"unseen_counts must be non-negative, got {unseen.tolist()}"
        )
    slots = np.arange(NUM_COUNT_SLOTS, dtype=np.int64)[None, :]  # (1, 5)
    cap = np.array(
        [_max_count_for_rank_index(r) for r in range(NUM_BELIEF_RANKS)],
        dtype=np.int64,
    )  # (15,)
    # (15, 5): legal iff slot k <= min(unseen_count, cap) for that rank.
    limit = np.minimum(unseen[:, None], cap[:, None])  # (15, 1) broadcast
    return slots <= limit


# --------------------------------------------------------------------------- #
# Posterior-derived helpers (shared by the model output and the value fusion)
# --------------------------------------------------------------------------- #
def expected_counts_from_probs(probs: np.ndarray) -> np.ndarray:
    """Return the per-rank expected count ``E[k]`` from a probability tensor.

    Parameters
    ----------
    probs:
        ``(..., 15, 5)`` array of per-rank count probabilities (already
        masked + normalized along the last axis).

    Returns
    -------
    numpy.ndarray
        ``(..., 15)`` expected counts.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape[-2:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"probs must end with shape ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {arr.shape}"
        )
    counts = np.arange(NUM_COUNT_SLOTS, dtype=np.float64)
    return (arr * counts).sum(axis=-1)


def total_entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """Return the summed per-rank entropy (in nats) from a probability tensor.

    A scalar-ish diagnostic: low entropy ⇒ a peaked posterior (the model is
    confident about the split); high entropy ⇒ a diffuse posterior. Computed
    over the count axis per rank, then summed across the 15 ranks.

    Parameters
    ----------
    probs:
        ``(..., 15, 5)`` normalized probabilities.

    Returns
    -------
    numpy.ndarray
        ``(...,)`` total entropy (nats), non-negative.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape[-2:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"probs must end with shape ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {arr.shape}"
        )
    flat = arr.reshape(-1, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
    # Avoid log(0): only sum where p > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_rank = -np.where(flat > 0, flat * np.log(flat), 0.0).sum(axis=-1)
    return per_rank.sum(axis=-1).reshape(arr.shape[:-2])
