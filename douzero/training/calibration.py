"""Win-probability calibration metrics (P06).

A value model's ``p_win`` head is only useful for action selection if its
probabilities are calibrated — a predicted 0.7 should win about 70% of the
time. This module provides the standard scalar calibration diagnostics
mandated by AGENTS.md "Evaluation requirements" and the P06 task:

- :func:`brier_score` — mean squared error between ``p_win`` and the 0/1
  outcome. Lower is better; a perfect predictor scores 0; a constant-0.5
  predictor scores 0.25.
- :func:`nll` — mean negative log-likelihood under a Bernoulli model. Lower
  is better; heavily penalizes over-confident wrong predictions.
- :func:`expected_calibration_error` (ECE) — the weighted mean absolute
  gap between the predicted confidence and the empirical accuracy inside
  fixed-width probability bins.
- :func:`reliability_bins` — the per-bin ``(accuracy, confidence, count)``
  table behind the ECE, so a reliability diagram can be drawn.

All functions are pure (no global state), CPU-friendly, and operate on
torch tensors so they can be computed either on the training minibatch or
on an evaluation buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Epsilon used to keep the NLL finite when p_win is exactly 0 or 1 (which
#: the sigmoid can never quite produce, but a clamped / quantized predictor
#: might).
_EPS: float = 1e-7


def _coerce_predictions(p_win: torch.Tensor, target_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten and dtype-coerce predictions and labels to 1-D float tensors."""
    p = p_win.float().reshape(-1).clamp(_EPS, 1.0 - _EPS)
    t = target_win.float().reshape(-1)
    if p.shape != t.shape:
        raise ValueError(
            f"p_win and target_win must have the same shape after flatten, "
            f"got {tuple(p.shape)} vs {tuple(t.shape)}"
        )
    if t.shape[0] == 0:
        raise ValueError("cannot compute calibration metrics on zero samples")
    return p, t


def brier_score(p_win: torch.Tensor, target_win: torch.Tensor) -> float:
    """Return the mean Brier score (lower is better; 0 is perfect)."""
    p, t = _coerce_predictions(p_win, target_win)
    return float(((p - t) ** 2).mean().item())


def nll(p_win: torch.Tensor, target_win: torch.Tensor) -> float:
    """Return the mean Bernoulli negative log-likelihood (lower is better)."""
    p, t = _coerce_predictions(p_win, target_win)
    # The clamping in _coerce_predictions keeps the log finite.
    loss = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
    return float(loss.mean().item())


@dataclass
class BinStat:
    """Per-bin reliability statistics (for diagramming / logging)."""

    low: float
    high: float
    accuracy: float
    confidence: float
    count: int


def reliability_bins(
    p_win: torch.Tensor,
    target_win: torch.Tensor,
    n_bins: int = 15,
) -> list[BinStat]:
    """Return per-bin ``(low, high, accuracy, confidence, count)`` statistics.

    Bins are equal-width intervals over ``[0, 1]``. Empty bins are kept in
    the output (with ``count == 0``) so plotting code can rely on a fixed
    bin count. ``accuracy`` is the empirical win rate in the bin;
    ``confidence`` is the mean predicted ``p_win`` in the bin.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    p, t = _coerce_predictions(p_win, target_win)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    out: list[BinStat] = []
    for i in range(n_bins):
        low = float(edges[i].item())
        high = float(edges[i + 1].item())
        if i == n_bins - 1:
            in_bin = (p >= low) & (p <= high)
        else:
            in_bin = (p >= low) & (p < high)
        count = int(in_bin.sum().item())
        if count == 0:
            acc = 0.0
            conf = 0.0
        else:
            acc = float(t[in_bin].mean().item())
            conf = float(p[in_bin].mean().item())
        out.append(
            BinStat(
                low=low,
                high=high,
                accuracy=acc,
                confidence=conf,
                count=count,
            )
        )
    return out


def expected_calibration_error(
    p_win: torch.Tensor,
    target_win: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Return the Expected Calibration Error (ECE).

    ECE is the count-weighted mean absolute gap between each bin's mean
    predicted confidence and its empirical accuracy. Lower is better; a
    perfectly calibrated predictor scores 0. With ``n_bins=1`` ECE reduces
    to ``|mean(p_win) - mean(target_win)|`` over the whole dataset.
    """
    bins = reliability_bins(p_win, target_win, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("cannot compute ECE over zero samples")
    ece = 0.0
    for b in bins:
        if b.count == 0:
            continue
        ece += (b.count / total) * abs(b.accuracy - b.confidence)
    return float(ece)
