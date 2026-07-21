"""P07: joint hidden-hand belief model with exact card conservation.

The belief model infers, from PUBLIC information alone, how the unknown cards
are split between the two opponents. It is the imperfect-information-safe
replacement for "reading the true hidden hands": at deployment it predicts a
posterior; at training time the labels come from
:class:`~douzero.observation.privileged.PrivilegedObservation` only.

AGENTS.md "Belief-model rules" (the spec this package implements):

    - per-rank allocations cannot exceed unseen counts
    - joker counts are at most one each
    - each opponent's total equals the public remaining-card count
    - the two opponent hands sum exactly to the unseen pool
    - known public bottom cards are assigned correctly
    - decoding and sampling cannot rely on unbounded rejection loops

Representation
--------------
From the acting player's view, every unknown card is held by exactly one of two
opponents. We pick a *canonical opponent A* (the NEXT seat, clockwise) as the
prediction target; opponent B (PREVIOUS seat) is then fully determined by

    count_B[rank] = unseen_count[rank] - count_A[rank]

with ``sum_r count_A[rank] == opponent_A_cards_left`` enforced exactly by the
dynamic program. There are 15 rank categories (13 numeric ranks + 2 jokers);
the model emits logits of shape ``[B, 15, 5]`` (count 0..4 per rank), masked to
``[0, unseen_count[rank]]`` and to ``[0, 1]`` for jokers.

Public modules:

- :mod:`constraints`  — canonical opponent, legal masks, label construction.
- :mod:`dynamic_programming` — exact MAP decoder + forward-filter/backward-sample.
- :mod:`features`      — build the belief input vector from a PublicObservation.
- :mod:`model`         — :class:`BeliefModel` (encoder + ``[B,15,5]`` head).
- :mod:`losses`        — masked cross-entropy + optional regularizers + metrics.
- :mod:`labels`        — privileged training labels from ``all_handcards``.
"""

from __future__ import annotations

from importlib import import_module

from .constraints import (
    BELIEF_RANKS,
    BELIEF_RANK_INDEX,
    JOKER_MAX_COUNT,
    NUMERIC_MAX_COUNT,
    NUM_BELIEF_RANKS,
    NUM_COUNT_SLOTS,
    canonical_opponent,
    canonical_opponent_b,
    expected_counts_from_probs,
    legal_mask,
    opponent_cards_left,
    opponent_unknown_total,
    per_rank_counts,
    per_rank_counts_from_hand,
    total_entropy_from_probs,
    unseen_counts_per_rank,
)
from .dynamic_programming import (
    BeliefDPError,
    constrained_marginals,
    decode_map,
    sample_allocation,
)
from .torch_dynamic_programming import constrained_marginals_torch
from .features import (
    BELIEF_INPUT_DIM,
    BeliefInput,
    build_belief_feature_vector,
    build_belief_input,
)
from .model import (
    BELIEF_FEATURE_DIM,
    BeliefConfig,
    BeliefModel,
    BeliefOutput,
    belief_features_from_probs,
    belief_features_from_torch_probs,
)

_TRAINING_ONLY_EXPORTS = {
    "BeliefLabel": ("douzero.belief.labels", "BeliefLabel"),
    "build_belief_label": ("douzero.belief.labels", "build_belief_label"),
    "target_allocation_tensor": (
        "douzero.belief.labels", "target_allocation_tensor"
    ),
    "BeliefLossComponents": (
        "douzero.belief.losses", "BeliefLossComponents"
    ),
    "belief_loss": ("douzero.belief.losses", "belief_loss"),
    "belief_metrics": ("douzero.belief.losses", "belief_metrics"),
    "JointCheckpointManifest": (
        "douzero.belief.joint_checkpoint", "JointCheckpointManifest"
    ),
    "load_joint_checkpoint": (
        "douzero.belief.joint_checkpoint", "load_joint_checkpoint"
    ),
    "save_joint_checkpoint": (
        "douzero.belief.joint_checkpoint", "save_joint_checkpoint"
    ),
}


def __getattr__(name: str):
    """Load privileged-label and trainer helpers only when explicitly requested."""

    target = _TRAINING_ONLY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value

__all__ = [
    "BELIEF_RANKS",
    "BELIEF_RANK_INDEX",
    "JOKER_MAX_COUNT",
    "NUMERIC_MAX_COUNT",
    "NUM_BELIEF_RANKS",
    "NUM_COUNT_SLOTS",
    "canonical_opponent",
    "canonical_opponent_b",
    "expected_counts_from_probs",
    "legal_mask",
    "opponent_cards_left",
    "opponent_unknown_total",
    "per_rank_counts",
    "per_rank_counts_from_hand",
    "total_entropy_from_probs",
    "unseen_counts_per_rank",
    "BeliefDPError",
    "constrained_marginals",
    "decode_map",
    "sample_allocation",
    "constrained_marginals_torch",
    "BELIEF_INPUT_DIM",
    "BeliefInput",
    "build_belief_feature_vector",
    "build_belief_input",
    "BeliefLabel",
    "build_belief_label",
    "target_allocation_tensor",
    "BeliefLossComponents",
    "belief_loss",
    "belief_metrics",
    "JointCheckpointManifest",
    "load_joint_checkpoint",
    "save_joint_checkpoint",
    "BELIEF_FEATURE_DIM",
    "BeliefConfig",
    "BeliefModel",
    "BeliefOutput",
    "belief_features_from_probs",
    "belief_features_from_torch_probs",
]
