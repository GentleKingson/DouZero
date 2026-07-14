"""Build the public-only belief input vector from a PublicObservation (P07).

The belief model consumes a compact, fixed-width feature vector assembled
entirely from public information. The most important components are the
per-rank multiplicity one-hot of the *belief unknown pool* (the cards the two
opponents hold between them — public bottom cards excluded for farmers) and
the two opponents' public remaining-card counts. Those two alone determine the
hard constraints the dynamic program enforces; the remaining fields (each
opponent's cumulative played cards, the revealed bottom cards, the last move,
the acting role) give the encoder evidence to shape the posterior.

Imperfect-information boundary
------------------------------
This module reads ONLY from :class:`~douzero.observation.public.PublicObservation`
(via :func:`~douzero.observation.public.compute_belief_unknown_pool`). It never
imports :mod:`douzero.observation.privileged`, never reads ``all_handcards``,
and produces an input that is invariant under any re-allocation of hidden cards
among the opponents (the leakage test asserts this). Belief training labels
are constructed separately in :mod:`douzero.belief.labels`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from douzero.observation.public import compute_belief_unknown_pool
from douzero.style.features import STYLE_FEATURE_WIDTH

from .constraints import (
    BELIEF_RANKS,
    MAX_OPPONENT_CARDS,
    NUM_BELIEF_RANKS,
    NUM_COUNT_SLOTS,
    canonical_opponent,
    canonical_opponent_b,
    opponent_cards_left,
    opponent_unknown_total,
    per_rank_counts,
)

#: Width of one per-rank multiplicity one-hot block (15 ranks × 5 count slots).
_RANK_ONEHOT_WIDTH: int = NUM_BELIEF_RANKS * NUM_COUNT_SLOTS  # 75

#: Width of one opponent cards-left one-hot (0..MAX_OPPONENT_CARDS inclusive).
_CARDS_LEFT_ONEHOT_WIDTH: int = MAX_OPPONENT_CARDS + 1  # 21

#: Width of one per-rank raw-count block (15 ranks).
_RANK_COUNT_WIDTH: int = NUM_BELIEF_RANKS  # 15

#: Acting-role one-hot width (3 roles).
_ROLE_ONEHOT_WIDTH: int = 3

#: Total belief input vector width. Derived from the named block widths above
#: so a schema change surfaces as a shape mismatch, not a silent reordering:
#:   unseen-pool one-hot (75)
#: + opponent-A cards-left one-hot (21)
#: + opponent-B cards-left one-hot (21)
#: + opponent-A played per-rank (15)
#: + opponent-B played per-rank (15)
#: + revealed-bottom per-rank (15)
#: + last-move per-rank (15)
#: + acting-role one-hot (3)
BELIEF_INPUT_DIM: int = (
    _RANK_ONEHOT_WIDTH
    + _CARDS_LEFT_ONEHOT_WIDTH * 2
    + _RANK_COUNT_WIDTH * 4
    + _ROLE_ONEHOT_WIDTH
)  # 75 + 42 + 60 + 3 = 180


def _multiplicity_onehot(counts: np.ndarray) -> np.ndarray:
    """Expand a ``(15,)`` int count vector into a ``(75,)`` one-hot block."""
    counts = np.asarray(counts, dtype=np.int64)
    onehot = np.zeros(_RANK_ONEHOT_WIDTH, dtype=np.float32)
    for r in range(NUM_BELIEF_RANKS):
        k = int(counts[r])
        if 0 <= k < NUM_COUNT_SLOTS:
            onehot[r * NUM_COUNT_SLOTS + k] = 1.0
    return onehot


def _cards_left_onehot(count: int) -> np.ndarray:
    """One-hot encode a remaining-card count over ``[0, MAX_OPPONENT_CARDS]``."""
    vec = np.zeros(_CARDS_LEFT_ONEHOT_WIDTH, dtype=np.float32)
    c = int(count)
    if c < 0:
        raise ValueError(f"cards-left count must be non-negative, got {c}")
    idx = min(c, MAX_OPPONENT_CARDS)
    vec[idx] = 1.0
    return vec


def _role_onehot(role: str) -> np.ndarray:
    """One-hot encode the acting role over the three absolute roles."""
    from douzero.observation.seats import ALL_ROLES

    vec = np.zeros(_ROLE_ONEHOT_WIDTH, dtype=np.float32)
    if role in ALL_ROLES:
        vec[ALL_ROLES.index(role)] = 1.0
    return vec


def _per_rank_raw(counts: np.ndarray) -> np.ndarray:
    """Return a ``(15,)`` float copy of a per-rank count vector."""
    return np.asarray(counts, dtype=np.float32)


@dataclass(frozen=True)
class BeliefInput:
    """The assembled public belief input for one decision.

    ``feature_vector`` is the ``(BELIEF_INPUT_DIM,)`` float32 tensor the
    :class:`~douzero.belief.model.BeliefModel` consumes. The remaining fields
    are the derived constraint quantities the model/DP need at decode time:
    ``unseen_counts`` (the per-rank unknown-pool counts) and
    ``opponent_a_total`` / ``opponent_b_total`` (the public remaining-card
    totals). Keeping them on the input object means the model never has to
    recompute the constraints from the opaque flat vector.

    All fields are PUBLIC. No hidden hand is referenced.
    """

    feature_vector: np.ndarray  # (BELIEF_INPUT_DIM,) float32
    unseen_counts: np.ndarray  # (15,) int64
    opponent_a_total: int
    opponent_b_total: int
    opponent_a_role: str
    opponent_b_role: str
    acting_role: str
    style_features: np.ndarray = field(
        default_factory=lambda: np.zeros(STYLE_FEATURE_WIDTH, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        if self.feature_vector.shape != (BELIEF_INPUT_DIM,):
            raise ValueError(
                f"feature_vector must have shape ({BELIEF_INPUT_DIM},), got "
                f"{self.feature_vector.shape}"
            )
        if self.unseen_counts.shape != (NUM_BELIEF_RANKS,):
            raise ValueError(
                f"unseen_counts must have shape ({NUM_BELIEF_RANKS},), got "
                f"{self.unseen_counts.shape}"
            )
        if self.style_features.shape != (STYLE_FEATURE_WIDTH,):
            raise ValueError(
                f"style_features must have shape ({STYLE_FEATURE_WIDTH},), got "
                f"{self.style_features.shape}"
            )
        # Freeze the arrays so the public input is immutable in spirit.
        for name in ("feature_vector", "unseen_counts", "style_features"):
            arr = getattr(self, name)
            if arr.flags.writeable:
                arr.setflags(write=False)


def build_belief_input(public: Any) -> BeliefInput:
    """Assemble a :class:`BeliefInput` from a ``PublicObservation``.

    Parameters
    ----------
    public:
        A :class:`~douzero.observation.public.PublicObservation` (or any object
        exposing the same public attributes: ``acting_role``, ``my_handcards``,
        ``played_cards``, ``bottom_cards.unplayed``, ``num_cards_left``,
        ``last_move``).

    The unseen pool is recomputed via
    :func:`~douzero.observation.public.compute_belief_unknown_pool`, which
    excludes the public unplayed bottom cards for farmers (the farmer knows
    those belong to the landlord). For the landlord the belief pool equals the
    parity pool.
    """
    from douzero.style.features import build_style_features

    acting_role = public.acting_role
    bottom_unplayed = tuple(public.bottom_cards.unplayed)
    bottom_revealed = tuple(public.bottom_cards.revealed)
    played: Mapping[str, tuple[int, ...]] = public.played_cards
    my_hand = tuple(public.my_handcards)

    # The belief unknown pool (public bottom cards excluded for farmers).
    unknown_pool = compute_belief_unknown_pool(
        my_hand, dict(played), bottom_unplayed, acting_role=acting_role
    )
    unseen = per_rank_counts(unknown_pool)

    opp_a = canonical_opponent(acting_role)
    opp_b = canonical_opponent_b(acting_role)
    # Opponent totals for the DP constraint. The landlord's public card-count
    # includes the known public bottom cards, which are NOT part of the unknown
    # pool; subtract them so ``sum(count_A)`` equals the predictable hidden
    # count (and ``A + B == pool.sum()`` holds). See opponent_unknown_total.
    opp_a_total = opponent_unknown_total(public.num_cards_left, opp_a, bottom_unplayed)
    opp_b_total = opponent_unknown_total(public.num_cards_left, opp_b, bottom_unplayed)
    # The cards-left one-hot is still the PUBLIC count (the encoder sees the
    # public remaining count, not the adjusted hidden count); the adjustment
    # only governs the DP total constraint stored below.
    opp_a_public = opponent_cards_left(public.num_cards_left, opp_a)
    opp_b_public = opponent_cards_left(public.num_cards_left, opp_b)

    # Assemble the flat feature vector.
    blocks: list[np.ndarray] = []
    blocks.append(_multiplicity_onehot(unseen))
    blocks.append(_cards_left_onehot(opp_a_public))
    blocks.append(_cards_left_onehot(opp_b_public))
    blocks.append(_per_rank_raw(per_rank_counts(played.get(opp_a, ()))))
    blocks.append(_per_rank_raw(per_rank_counts(played.get(opp_b, ()))))
    blocks.append(_per_rank_raw(per_rank_counts(bottom_revealed)))
    blocks.append(_per_rank_raw(per_rank_counts(tuple(public.last_move))))
    blocks.append(_role_onehot(acting_role))
    feature_vector = np.concatenate(blocks).astype(np.float32)

    # Sanity: the unknown pool total must equal the two opponents' *hidden*
    # totals (card conservation, bottom cards excluded).
    pool_total = int(unseen.sum())
    if pool_total != opp_a_total + opp_b_total:
        raise ValueError(
            "Belief input conservation violated: unknown pool total "
            f"{pool_total} != opponent A hidden ({opp_a_total}) + opponent B "
            f"hidden ({opp_b_total}). The public observation is inconsistent."
        )

    return BeliefInput(
        feature_vector=feature_vector,
        unseen_counts=unseen,
        opponent_a_total=opp_a_total,
        opponent_b_total=opp_b_total,
        opponent_a_role=opp_a,
        opponent_b_role=opp_b,
        acting_role=acting_role,
        style_features=build_style_features(public),
    )


def build_belief_feature_vector(public: Any) -> np.ndarray:
    """Convenience wrapper returning only the flat feature vector."""
    return build_belief_input(public).feature_vector
