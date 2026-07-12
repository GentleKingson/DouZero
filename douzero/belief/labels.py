"""Privileged belief training labels (P07).

AGENTS.md imperfect-information boundary:

    "Belief labels come from privileged training data only. Belief predictions
    may be supplied to the public value model as posterior means, probabilities,
    entropy, or legal samples."

This module is the ONLY place the true hidden allocation is read to construct a
belief training target. The label describes the canonical opponent A's (NEXT
seat's) per-rank count allocation. The public value model and the deployment
``DeepAgentV2`` never import this module and never see these labels; the belief
model's training entry point (``train_belief.py``) is the sole consumer.

The label is represented in two forms:

- :class:`BeliefLabel` — a small immutable dataclass with the ``(15,)`` int
  target allocation, the target total, and the per-rank count-slot one-hot
  ``[15, 5]`` used by the masked cross-entropy loss.
- :func:`target_allocation_tensor` — the ``(B, 15, 5)`` float one-hot tensor a
  batched loss consumes, built from a batch of allocations.

The label is deterministically derived from ``infoset.all_handcards`` (the
privileged per-role hands) and the public unseen counts; the same public state
with a different true allocation yields a different label but the same model
INPUT — which is exactly what the leakage test enforces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constraints import (
    NUM_BELIEF_RANKS,
    NUM_COUNT_SLOTS,
    canonical_opponent,
    opponent_unknown_total,
    per_rank_counts,
)


@dataclass(frozen=True)
class BeliefLabel:
    """The training-only target for one belief decision.

    ``allocation`` is opponent A's true per-rank count vector (sum ==
    opponent A's public remaining-card count, by card conservation).
    ``count_onehot`` is the ``[15, 5]`` one-hot of ``allocation`` for the
    cross-entropy loss. ``opponent_a_role`` records which role was predicted so
    the label can be cross-checked against the public observation.

    Carrying ``unseen_counts`` lets the loss/decoder reconstruct opponent B by
    subtraction and verify conservation without re-reading privileged data.
    """

    allocation: np.ndarray  # (15,) int64
    count_onehot: np.ndarray  # (15, 5) float32
    opponent_a_role: str
    opponent_a_total: int
    unseen_counts: np.ndarray  # (15,) int64

    def __post_init__(self) -> None:
        if self.allocation.shape != (NUM_BELIEF_RANKS,):
            raise ValueError(
                f"allocation must have shape ({NUM_BELIEF_RANKS},), got "
                f"{self.allocation.shape}"
            )
        if self.count_onehot.shape != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
            raise ValueError(
                f"count_onehot must have shape ({NUM_BELIEF_RANKS}, "
                f"{NUM_COUNT_SLOTS}), got {self.count_onehot.shape}"
            )
        for arr_name in ("allocation", "count_onehot", "unseen_counts"):
            arr = getattr(self, arr_name)
            if arr.flags.writeable:
                arr.setflags(write=False)


def _allocation_onehot(allocation: np.ndarray) -> np.ndarray:
    """Return the ``[15, 5]`` one-hot of a per-rank count allocation."""
    onehot = np.zeros((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), dtype=np.float32)
    for r in range(NUM_BELIEF_RANKS):
        k = int(allocation[r])
        if 0 <= k < NUM_COUNT_SLOTS:
            onehot[r, k] = 1.0
    return onehot


def build_belief_label(
    *,
    acting_role: str,
    all_handcards: Mapping[str, Iterable[int]],
    unseen_counts: np.ndarray,
    num_cards_left: Mapping[str, int],
    bottom_unplayed,
) -> BeliefLabel:
    """Construct the privileged belief label for one decision.

    Parameters
    ----------
    acting_role:
        The acting player's role (determines the canonical opponent A).
    all_handcards:
        ``{role: cards}`` of every role's TRUE hand (privileged). This is
        ``infoset.all_handcards``. MUST NOT be available at inference.
    unseen_counts:
        The public per-rank unknown-pool counts (from
        :func:`~douzero.belief.features.build_belief_input`), carried here so
        the label can verify ``allocation == opponent A's true hidden hand``
        without recomputing the public pool.
    num_cards_left:
        The public remaining-card counts (for the opponent-A total check).
    bottom_unplayed:
        The current unplayed public bottom cards (public). When opponent A is
        the landlord, these are *known* and excluded from the predicted
        allocation: the label describes only the landlord's HIDDEN portion
        (true hand minus the unplayed bottom cards), matching what the belief
        model predicts over the unknown pool.

    Returns
    -------
    BeliefLabel
        The immutable training target.

    Raises
    ------
    ValueError
        If the true allocation violates conservation (the privileged hand does
        not match the public unseen pool / remaining count). A mismatch means
        the infoset itself is inconsistent.
    """
    from douzero.observation.seats import LANDLORD_ROLE

    opp_a = canonical_opponent(acting_role)
    if opp_a not in all_handcards:
        raise ValueError(
            f"all_handcards is missing opponent A role {opp_a!r}; cannot build "
            f"the belief label for acting role {acting_role!r}."
        )
    allocation = per_rank_counts(all_handcards[opp_a])
    unseen = np.asarray(unseen_counts, dtype=np.int64)
    if allocation.shape != unseen.shape:
        raise ValueError(
            f"unseen_counts shape {unseen.shape} != allocation shape "
            f"{allocation.shape}"
        )
    # If opponent A is the landlord, subtract the known public bottom cards:
    # the belief model predicts only the HIDDEN portion over the unknown pool.
    if opp_a == LANDLORD_ROLE:
        allocation = allocation - per_rank_counts(bottom_unplayed or ())
        if np.any(allocation < 0):
            raise ValueError(
                "Belief label conservation violated: opponent A (landlord) "
                "true hand has fewer cards than the public bottom cards at "
                f"ranks {np.where(allocation < 0)[0].tolist()}."
            )
    # Conservation checks against the (bottom-adjusted) hidden total.
    hidden_total = opponent_unknown_total(num_cards_left, opp_a, bottom_unplayed)
    if int(allocation.sum()) != hidden_total:
        raise ValueError(
            f"Belief label conservation violated: opponent A hidden allocation "
            f"has {int(allocation.sum())} cards but the predictable hidden "
            f"total is {hidden_total}. The infoset is inconsistent."
        )
    if np.any(allocation > unseen):
        # Opponent A cannot hold more of a rank than exists in the unknown
        # pool. (Opponent B = unseen - allocation must be non-negative.)
        bad = np.where(allocation > unseen)[0].tolist()
        raise ValueError(
            f"Belief label conservation violated: opponent A holds more than "
            f"the unseen pool at rank indices {bad}. allocation="
            f"{allocation.tolist()} unseen={unseen.tolist()}"
        )
    return BeliefLabel(
        allocation=allocation,
        count_onehot=_allocation_onehot(allocation),
        opponent_a_role=opp_a,
        opponent_a_total=int(allocation.sum()),
        unseen_counts=unseen,
    )


def target_allocation_tensor(
    allocations: Sequence[np.ndarray],
) -> np.ndarray:
    """Build a ``(B, 15, 5)`` float one-hot target tensor from allocations.

    Each ``allocation`` is a ``(15,)`` int vector. Used by the batched masked
    cross-entropy loss.
    """
    if len(allocations) == 0:
        raise ValueError("target_allocation_tensor needs at least one allocation")
    rows = []
    for alloc in allocations:
        rows.append(_allocation_onehot(np.asarray(alloc, dtype=np.int64)))
    return np.stack(rows, axis=0).astype(np.float32)
