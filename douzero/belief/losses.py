"""Masked cross-entropy loss + metrics for the belief model (P07).

The belief loss is a per-rank count cross-entropy, masked so each rank's
softmax only runs over the *legal* count slots (those allowed by the public
unseen-pool cap and the joker multiplicity cap). Two optional regularizers are
provided (count-total and entropy), default-off; and three diagnostic metrics
(rank accuracy, exact-match, count MAE) that ``evaluate_belief.py`` reports.

Numerical stability
-------------------
Illegal logit slots are set to a finite ``-1e30`` sentinel before softmax
(not ``-inf``) so masked softmax produces exact zeros for illegal slots
without ``exp(-inf) = 0`` masking edge cases, and the cross-entropy gather of
a legal target never reads an ``-inf`` logit. The loss returns a clean zero
for a rank whose target is the only legal slot (the model is forced to be
right there — no gradient and no NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .constraints import NUM_BELIEF_RANKS, NUM_COUNT_SLOTS

#: Finite sentinel for masked logits (see module docstring).
_MASK_LOGIT: float = -1e30


def _masked_log_softmax(
    logits: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Log-softmax over the count axis, restricted to legal slots.

    Parameters
    ----------
    logits:
        ``(..., 15, 5)`` raw logits.
    legal:
        ``(..., 15, 5)`` bool tensor; ``True`` where the slot is legal.

    Returns
    -------
    torch.Tensor
        ``(..., 15, 5)`` log-softmax; illegal slots are exactly 0.0 (so they
        contribute nothing when gathered, and ``exp`` of them is 0).
    """
    masked = logits.masked_fill(~legal, _MASK_LOGIT)
    return F.log_softmax(masked, dim=-1)


def belief_loss(
    logits: torch.Tensor,
    target_onehot: torch.Tensor,
    legal: torch.Tensor,
    *,
    lambda_count_reg: float = 0.0,
    lambda_entropy_reg: float = 0.0,
) -> "BeliefLossComponents":
    """Masked per-rank count cross-entropy for the belief head.

    Parameters
    ----------
    logits:
        ``(B, 15, 5)`` raw belief logits.
    target_onehot:
        ``(B, 15, 5)`` one-hot target allocation (from
        :func:`~douzero.belief.labels.target_allocation_tensor`).
    legal:
        ``(B, 15, 5)`` bool legal-count mask (from
        :func:`~douzero.belief.constraints.legal_mask`, batched).
    lambda_count_reg:
        Optional L2 penalty on ``(expected_count - target_total_per_rank)``.
        Default 0.0 (disabled).
    lambda_entropy_reg:
        Optional entropy regularizer weight. Positive encourages a smoother
        posterior (exploration); negative (discouraged) sharpens it. Default
        0.0 (disabled).

    Returns
    -------
    BeliefLossComponents
        ``total`` is the gradient-bearing scalar; the remaining fields are
        python floats for logging.
    """
    if logits.shape != target_onehot.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != target_onehot shape "
            f"{tuple(target_onehot.shape)}"
        )
    if logits.shape != legal.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != legal mask shape "
            f"{tuple(legal.shape)}"
        )
    if logits.shape[-2:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"logits must end with ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {tuple(logits.shape)}"
        )
    for name, val in (("lambda_count_reg", lambda_count_reg),
                      ("lambda_entropy_reg", lambda_entropy_reg)):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")

    logp = _masked_log_softmax(logits.float(), legal.bool())
    # Per-rank cross-entropy = -sum_k target[r,k] * logp[r,k]. Gather only
    # legal targets (a target slot outside the legal mask is a label bug; we
    # still compute it but it reads the masked logit's log_softmax, which is
    # finite, so no NaN — and the metrics will flag the mismatch).
    ce_per_rank = -(target_onehot.float() * logp).sum(dim=-1)  # (B, 15)
    # Mean over ranks and batch.
    ce = ce_per_rank.mean()

    total = ce
    count_term = logits.new_zeros(())
    entropy_term = logits.new_zeros(())

    if lambda_count_reg != 0.0:
        counts = torch.arange(NUM_COUNT_SLOTS, dtype=logits.dtype, device=logits.device)
        probs = logp.exp()
        expected = (probs * counts).sum(dim=-1)  # (B, 15)
        target_counts = (target_onehot.float() * counts).sum(dim=-1)  # (B, 15)
        count_term = ((expected - target_counts) ** 2).mean()
        total = total + lambda_count_reg * count_term

    if lambda_entropy_reg != 0.0:
        probs = logp.exp()
        # Per-rank entropy in nats; only legal slots contribute (illegal are 0).
        entropy_per_rank = -(probs * logp).sum(dim=-1)  # (B, 15)
        entropy_term = entropy_per_rank.mean()
        # Positive lambda_entropy_reg *encourages* entropy (adds -lambda*H to
        # the loss); negative sharpens. Documented in the docstring.
        total = total - lambda_entropy_reg * entropy_term

    return BeliefLossComponents(
        total=total,
        cross_entropy=float(ce.detach().float().item()),
        count_reg=float(count_term.detach().float().item()),
        entropy_reg=float(entropy_term.detach().float().item()),
    )


def belief_metrics(
    probs: np.ndarray,
    target_allocation: np.ndarray,
    legal: np.ndarray,
) -> dict[str, float]:
    """Return evaluation metrics for a batch of belief predictions.

    Parameters
    ----------
    probs:
        ``(B, 15, 5)`` normalized probabilities (masked slots should be ~0).
    target_allocation:
        ``(B, 15)`` int true allocations.
    legal:
        ``(B, 15, 5)`` bool legal mask.

    Returns
    -------
    dict[str, float]
        ``rank_accuracy`` (fraction of ranks whose argmax count matches the
        target), ``exact_match`` (fraction of samples where ALL 15 ranks
        match), and ``count_mae`` (mean absolute per-rank count error).
    """
    p = np.asarray(probs, dtype=np.float64)
    tgt = np.asarray(target_allocation, dtype=np.int64)
    lg = np.asarray(legal, dtype=bool)
    if p.shape[-2:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"probs must end with ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {p.shape}"
        )
    if tgt.shape != p.shape[:-1]:
        raise ValueError(
            f"target_allocation shape {tgt.shape} != probs leading {p.shape[:-1]}"
        )
    # Argmax restricted to legal slots: set illegal prob to -1.
    masked_p = np.where(lg, p, -1.0)
    pred = masked_p.argmax(axis=-1)  # (B, 15)
    rank_match = (pred == tgt)
    rank_accuracy = float(rank_match.mean())
    exact_match = float(rank_match.all(axis=-1).mean())
    count_mae = float(np.abs(pred - tgt).mean())
    return {
        "rank_accuracy": rank_accuracy,
        "exact_match": exact_match,
        "count_mae": count_mae,
    }


@dataclass
class BeliefLossComponents:
    """Return type of :func:`belief_loss`.

    ``total`` is the gradient-bearing scalar; the rest are detached floats
    for logging.
    """

    total: torch.Tensor
    cross_entropy: float
    count_reg: float
    entropy_reg: float

    def as_log_dict(self) -> dict[str, float]:
        return {
            "belief_loss_total": float(self.total.detach().float().item()),
            "belief_cross_entropy": float(self.cross_entropy),
            "belief_count_reg": float(self.count_reg),
            "belief_entropy_reg": float(self.entropy_reg),
        }
