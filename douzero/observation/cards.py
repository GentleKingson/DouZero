"""Versioned card and rank constants for the observation V2 schema (P03).

The legacy encoder (``douzero/env/env.py``) hard-codes a 54-dim card layout:
a Fortran-flattened ``4×13`` rank matrix (ranks ``3..14`` and ``17``) followed
by two joker slots ``[small=20, big=30]``. Rank ``16`` is intentionally unused.

This module is the single source of truth for that layout, derived from named
constants rather than scattered magic numbers. The legacy encoder's
``_cards2array`` and ``Card2Column`` map are reproduced exactly here so a V2
observation can round-trip into the legacy ``x_batch`` / ``z_batch`` tensors
without re-deriving the offsets.

AGENTS.md: "Derive dimensions from a schema or named constants. Do not scatter
magic widths such as role-specific flattened feature sizes across files."
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Canonical rank set (13 ranks: 3..14 and 17; rank 16 is unused)
# --------------------------------------------------------------------------- #
#: The numeric ranks that appear as ordinary cards, ascending. The "2" is
#: represented by ``17`` (the legacy convention); ``16`` is deliberately absent.
NUMERIC_RANKS: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17)

#: Small joker. Uses the legacy code point.
SMALL_JOKER: int = 20

#: Big joker. Uses the legacy code point.
BIG_JOKER: int = 30

#: The two joker code points, in canonical (small, big) order.
JOKERS: tuple[int, int] = (SMALL_JOKER, BIG_JOKER)

#: Number of distinct numeric ranks (each appears up to 4 times).
NUM_NUMERIC_RANKS: int = len(NUMERIC_RANKS)  # 13

#: Maximum multiplicity of any numeric rank (a full deck has four copies).
MAX_NUMERIC_MULTIPLICITY: int = 4

#: Number of joker slots (small + big; each appears at most once).
NUM_JOKER_SLOTS: int = len(JOKERS)  # 2

#: Total card-vector width: 13 ranks × 4 copies + 2 jokers = 54.
CARD_VECTOR_DIM: int = NUM_NUMERIC_RANKS * MAX_NUMERIC_MULTIPLICITY + NUM_JOKER_SLOTS  # 54

#: Column index of each numeric rank in the 4-wide rank matrix. Matches the
#: legacy ``Card2Column`` (env.py) exactly.
RANK_TO_COLUMN: dict[int, int] = {rank: i for i, rank in enumerate(NUMERIC_RANKS)}

#: Inverse of :data:`RANK_TO_COLUMN`.
COLUMN_TO_RANK: dict[int, int] = {i: rank for rank, i in RANK_TO_COLUMN.items()}

#: Offset of the small-joker slot within the 54-dim vector.
SMALL_JOKER_OFFSET: int = NUM_NUMERIC_RANKS * MAX_NUMERIC_MULTIPLICITY  # 52

#: Offset of the big-joker slot within the 54-dim vector.
BIG_JOKER_OFFSET: int = SMALL_JOKER_OFFSET + 1  # 53

#: The canonical full 54-card deck, ascending. Matches ``env.py`` ``deck``.
DECK: tuple[int, ...] = tuple(
    card for rank in NUMERIC_RANKS for card in (rank,) * MAX_NUMERIC_MULTIPLICITY
) + (SMALL_JOKER, BIG_JOKER)

#: One-hot multiplicity vectors, indexed by count 0..4. Matches legacy
#: ``NumOnes2Array`` (env.py): count ``k`` becomes a length-4 vector with the
#: first ``k`` entries set to 1.
_MULTIPLICITY_VECTORS: tuple[np.ndarray, ...] = tuple(
    np.array([1] * k + [0] * (MAX_NUMERIC_MULTIPLICITY - k), dtype=np.int8)
    for k in range(MAX_NUMERIC_MULTIPLICITY + 1)
)


def cards_to_vector(cards: Iterable[int]) -> np.ndarray:
    """Encode a multiset of cards into the canonical 54-dim int8 vector.

    Reproduces the legacy ``_cards2array`` layout exactly:
    each numeric rank occupies a 4-wide column whose entries are the
    multiplicity one-hot (``NumOnes2Array``); the trailing two slots are the
    small and big joker indicators. An empty input yields an all-zero vector.

    The result is Fortran-flattened from a ``4×13`` matrix so rank ``r`` lives
    at offsets ``RANK_TO_COLUMN[r]*4 .. +3`` — identical to the legacy encoder.
    """
    vec = np.zeros(CARD_VECTOR_DIM, dtype=np.int8)
    counts = Counter(cards)
    for card, count in counts.items():
        if card < 20:
            col = RANK_TO_COLUMN.get(card)
            if col is None:
                raise ValueError(f"Unknown numeric card rank {card!r}")
            if count < 0 or count > MAX_NUMERIC_MULTIPLICITY:
                raise ValueError(
                    f"Rank {card} multiplicity {count} out of range "
                    f"[0, {MAX_NUMERIC_MULTIPLICITY}]"
                )
            vec[col * MAX_NUMERIC_MULTIPLICITY:(col + 1) * MAX_NUMERIC_MULTIPLICITY] = (
                _MULTIPLICITY_VECTORS[count]
            )
        elif card == SMALL_JOKER:
            if count > 1:
                raise ValueError(f"Small joker multiplicity {count} exceeds 1")
            vec[SMALL_JOKER_OFFSET] = 1
        elif card == BIG_JOKER:
            if count > 1:
                raise ValueError(f"Big joker multiplicity {count} exceeds 1")
            vec[BIG_JOKER_OFFSET] = 1
        else:
            raise ValueError(f"Unknown card {card!r}")
    return vec


def is_valid_card(card: int) -> bool:
    """Return True if ``card`` is a canonical deck member."""
    if card == SMALL_JOKER or card == BIG_JOKER:
        return True
    return card in RANK_TO_COLUMN
