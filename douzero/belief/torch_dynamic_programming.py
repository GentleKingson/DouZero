"""Differentiable constrained belief marginals in pure PyTorch.

The exact NumPy/Python dynamic program remains the source for MAP decoding,
sampling, and evaluation.  This module implements the same log-space
forward-backward recurrence with tensor operations so a value loss can flow
through constrained posterior features into :class:`BeliefModel`.

The recurrence always runs in float32, including under autocast.  Belief
partition functions are numerically sensitive and the state space is tiny
(``15 * 21 * 5``), so reduced precision offers little benefit here.
"""

from __future__ import annotations

import torch

from .constraints import NUM_BELIEF_RANKS, NUM_COUNT_SLOTS
from .dynamic_programming import BeliefDPError

_NEG_INF: float = -1e30
_INFEASIBLE_THRESHOLD: float = -1e20


def _advance_log_partition(
    previous: torch.Tensor,
    rank_logits: torch.Tensor,
) -> torch.Tensor:
    """Advance one bounded-count log-partition row.

    ``previous`` is ``(B, T + 1)`` and ``rank_logits`` is ``(B, 5)``.
    A finite negative sentinel is used instead of ``-inf``: all-impossible
    ``logsumexp`` cells otherwise have undefined softmax gradients and can
    introduce NaNs even when the requested final total is feasible.
    """

    batch, width = previous.shape
    terms: list[torch.Tensor] = []
    for count in range(NUM_COUNT_SLOTS):
        if count >= width:
            shifted = previous.new_full((batch, width), _NEG_INF)
        elif count == 0:
            shifted = previous
        else:
            padding = previous.new_full((batch, count), _NEG_INF)
            shifted = torch.cat((padding, previous[:, : width - count]), dim=1)
        terms.append(shifted + rank_logits[:, count : count + 1])
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def constrained_marginals_torch(
    logits: torch.Tensor,
    totals: torch.Tensor | int,
    legal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return differentiable marginals conditioned on an exact card total.

    Parameters
    ----------
    logits:
        ``(B, 15, 5)`` or ``(15, 5)`` factor logits.
    totals:
        Length-``B`` integer totals, or one integer for a single sample.
    legal:
        Boolean tensor with the same shape as ``logits``.  Illegal count slots
        receive exactly zero probability.  When omitted, every finite slot is
        treated as legal.

    Returns
    -------
    torch.Tensor
        Float32 constrained marginals with the same rank as ``logits``.  Each
        rank row sums to one and ``sum_r E[c_r]`` equals the requested total
        up to float32 rounding.  Gradients flow to legal ``logits``.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    squeeze = logits.ndim == 2
    batched = logits.unsqueeze(0) if squeeze else logits
    if batched.ndim != 3 or tuple(batched.shape[-2:]) != (
        NUM_BELIEF_RANKS,
        NUM_COUNT_SLOTS,
    ):
        raise ValueError(
            "logits must have shape (15, 5) or (B, 15, 5), got "
            f"{tuple(logits.shape)}"
        )
    if legal is None:
        legal_batched = torch.isfinite(batched)
    else:
        if not isinstance(legal, torch.Tensor):
            raise TypeError(f"legal must be a torch.Tensor, got {type(legal).__name__}")
        legal_batched = legal.unsqueeze(0) if squeeze and legal.ndim == 2 else legal
        if tuple(legal_batched.shape) != tuple(batched.shape):
            raise ValueError(
                f"legal shape {tuple(legal.shape)} does not match logits shape "
                f"{tuple(logits.shape)}"
            )
        legal_batched = legal_batched.to(device=batched.device, dtype=torch.bool)

    batch_size = batched.shape[0]
    if isinstance(totals, bool):
        raise TypeError("totals must contain integers, not bool")
    totals_tensor = torch.as_tensor(totals, device=batched.device)
    if totals_tensor.ndim == 0:
        totals_tensor = totals_tensor.reshape(1)
    if totals_tensor.ndim != 1 or totals_tensor.numel() != batch_size:
        raise ValueError(
            f"totals must contain one value per batch item ({batch_size}), got "
            f"shape {tuple(totals_tensor.shape)}"
        )
    if totals_tensor.dtype == torch.bool or totals_tensor.is_floating_point():
        raise TypeError("totals must be an integer tensor or integer sequence")
    totals_tensor = totals_tensor.to(dtype=torch.long)
    if bool((totals_tensor < 0).any().item()):
        raise ValueError("totals must be non-negative")

    # Explicit float32 island: safe under FP16/BF16 autocast and differentiable
    # back to the source dtype through the cast operation.
    work_logits = batched.float().masked_fill(~legal_batched, _NEG_INF)
    max_total = int(totals_tensor.detach().max().item())

    initial = work_logits.new_full((batch_size, max_total + 1), _NEG_INF)
    initial[:, 0] = 0.0
    alpha: list[torch.Tensor] = [initial]
    for rank in range(NUM_BELIEF_RANKS):
        alpha.append(_advance_log_partition(alpha[-1], work_logits[:, rank]))

    terminal = alpha[-1].gather(1, totals_tensor[:, None]).squeeze(1)
    infeasible = terminal <= _INFEASIBLE_THRESHOLD
    if bool(infeasible.any().item()):
        bad = torch.nonzero(infeasible, as_tuple=False).flatten().tolist()
        requested = totals_tensor[infeasible].detach().cpu().tolist()
        raise BeliefDPError(
            "Differentiable belief DP found no feasible allocation for batch "
            f"indices {bad} with totals {requested}."
        )

    beta: list[torch.Tensor | None] = [None] * (NUM_BELIEF_RANKS + 1)
    beta[-1] = initial
    for rank in range(NUM_BELIEF_RANKS - 1, -1, -1):
        beta[rank] = _advance_log_partition(beta[rank + 1], work_logits[:, rank])

    prefix_totals = torch.arange(max_total + 1, device=batched.device)
    rank_marginals: list[torch.Tensor] = []
    for rank in range(NUM_BELIEF_RANKS):
        count_weights: list[torch.Tensor] = []
        suffix = beta[rank + 1]
        if suffix is None:  # pragma: no cover - construction invariant
            raise RuntimeError("belief backward table is incomplete")
        for count in range(NUM_COUNT_SLOTS):
            suffix_total = totals_tensor[:, None] - count - prefix_totals[None, :]
            valid_total = (suffix_total >= 0) & (suffix_total <= max_total)
            suffix_value = suffix.gather(1, suffix_total.clamp(0, max_total))
            convolution = torch.where(
                valid_total,
                alpha[rank] + suffix_value,
                alpha[rank].new_full((), _NEG_INF),
            )
            other_log_partition = torch.logsumexp(convolution, dim=1)
            count_weights.append(
                work_logits[:, rank, count] + other_log_partition
            )
        unnormalized = torch.stack(count_weights, dim=-1)
        probs = torch.softmax(unnormalized, dim=-1)
        probs = probs.masked_fill(~legal_batched[:, rank], 0.0)
        # Masking is exact; renormalization absorbs the last few ULPs without
        # weakening the total constraint or cutting the graph.
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        rank_marginals.append(probs)

    result = torch.stack(rank_marginals, dim=1)
    return result.squeeze(0) if squeeze else result


__all__ = ["constrained_marginals_torch"]
