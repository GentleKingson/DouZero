"""Win-probability calibration metrics (P06).

A value model's ``p_win`` head is only useful for action selection if its
probabilities are calibrated — a predicted 0.7 should win about 70% of the
time. This module provides the standard scalar calibration diagnostics
mandated AGENTS.md "Evaluation requirements" and the P06 task:

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

Input validation (P06 r2)
-------------------------
Every public function calls :func:`_validate_predictions`, which rejects:

- shape mismatches between ``p_win`` and ``target_win``;
- empty inputs;
- non-finite ``p_win`` (NaN/Inf);
- ``p_win`` outside ``[0, 1]`` (an illegal probability — a sigmoid head
  can never produce these, so their presence indicates a bug);
- ``target_win`` outside ``{0, 1}``` (a non-binary label).

Illegal probabilities are REJECTED, not silently clamped. The r0/r1 code
clamped ``p_win`` to ``[_EPS, 1-_EPS]`` inside :func:`_coerce_predictions`,
which masked a ``p_win=2.0`` as a near-perfect prediction instead of
surfacing it as a model bug. Only :func:`nll` clamps a LOCAL copy of the
already-validated predictions, purely to keep ``log(0)`` finite at the
exact-boundary values 0.0 and 1.0 that a quantized predictor can emit.

All functions are pure (no global state), CPU-friendly, and operate on
torch tensors so they can be computed either on the training minibatch or
on an evaluation buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Epsilon used ONLY inside :func:`nll` to keep ``log`` finite when a
#: quantized predictor emits exactly 0.0 or 1.0. This is NOT a general-
#: purpose clamp — :func:`_validate_predictions` rejects out-of-range
#: probabilities before this is applied.
_EPS: float = 1e-7

#: Tolerance for the ``target_win ∈ {0, 1}`` check (float labels from a
#: serialized dataset may carry tiny rounding error).
_TARGET_TOL: float = 1e-6


def _validate_predictions(
    p_win: torch.Tensor,
    target_win: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten, dtype-coerce, and validate ``p_win`` / ``target_win``.

    Returns the validated but UNCLAMPED ``(p, t)`` 1-D float tensors.
    Raises :class:`ValueError` for any of the illegal cases listed in the
    module docstring. Does NOT clamp — clamping would mask illegal
    probabilities as near-perfect predictions (P06 r2 fix).
    """
    p = p_win.float().reshape(-1)
    t = target_win.float().reshape(-1)
    if p.shape != t.shape:
        raise ValueError(
            f"p_win and target_win must have the same shape after flatten, "
            f"got {tuple(p.shape)} vs {tuple(t.shape)}"
        )
    if p.shape[0] == 0:
        raise ValueError("cannot compute calibration metrics on zero samples")
    if not bool(torch.isfinite(p).all()):
        raise ValueError(
            f"p_win contains non-finite values (NaN/Inf); calibration metrics "
            f"require finite probabilities in [0, 1]."
        )
    if bool((p < 0.0).any()) or bool((p > 1.0).any()):
        raise ValueError(
            f"p_win contains values outside [0, 1]; a sigmoid head can never "
            f"produce these, so their presence indicates a model or encoding "
            f"bug. min={float(p.min().item()):.6g}, "
            f"max={float(p.max().item()):.6g}."
        )
    # target_win must be binary {0, 1}. Allow a tiny float tolerance.
    t_min = float(t.min().item())
    t_max = float(t.max().item())
    if t_min < -_TARGET_TOL or t_max > 1.0 + _TARGET_TOL:
        raise ValueError(
            f"target_win must be in {{0, 1}}, got range [{t_min:.6g}, {t_max:.6g}]."
        )
    # Check that every value is close to 0 or 1.
    dist_to_binary = (t - t.round().clamp(0.0, 1.0)).abs()
    if bool((dist_to_binary > _TARGET_TOL).any()):
        raise ValueError(
            f"target_win must be binary {{0, 1}}, got non-binary values. "
            f"max distance to nearest of {{0,1}} = {float(dist_to_binary.max().item()):.6g}."
        )
    return p, t


def brier_score(p_win: torch.Tensor, target_win: torch.Tensor) -> float:
    """Return the mean Brier score (lower is better; 0 is perfect).

    Uses the validated but UNCLAMPED ``p_win`` so an illegal probability
    (e.g. 2.0) produces a large Brier score rather than being masked to a
    near-perfect 1.0.
    """
    p, t = _validate_predictions(p_win, target_win)
    return float(((p - t) ** 2).mean().item())


def nll(p_win: torch.Tensor, target_win: torch.Tensor) -> float:
    """Return the mean Bernoulli negative log-likelihood (lower is better).

    Validates first (rejecting illegal probabilities), then clamps a LOCAL
    copy to ``[_EPS, 1-_EPS]`` purely to keep ``log`` finite at the exact
    boundary values 0.0 and 1.0 that a quantized predictor can emit.
    """
    p, t = _validate_predictions(p_win, target_win)
    p_clamped = p.clamp(_EPS, 1.0 - _EPS)
    loss = -(t * torch.log(p_clamped) + (1.0 - t) * torch.log(1.0 - p_clamped))
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
    p, t = _validate_predictions(p_win, target_win)
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
