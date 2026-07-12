"""Numerical-safety guards for Model V2 (P05).

A real runtime NaN/Inf guard (bug #5 fix). The model's forward is wrapped so a
NaN/Inf in an INPUT or a WEIGHT-driven output is caught and raised as a
descriptive error, rather than silently propagating into the multi-objective
loss (P06) where it would poison gradients and produce a misleading "training
diverged" symptom.

Design
------
- :func:`assert_finite` checks a tensor and raises :class:`NumericalError`
  with the offending tensor's name, shape, and the count of non-finite values.
- The guard is OPT-IN via ``ModelV2Config.nan_guard`` (default on). A caller
  that has already validated inputs (e.g. a batched training loop with its own
  sanity checks) may disable it for speed; the default keeps deployment safe.
- The guard runs on the model OUTPUT (the fused representation and the head
  outputs), which catches both bad inputs AND bad weights (a NaN weight
  produces a NaN output regardless of the input). Checking only inputs would
  miss weight corruption.

This is NOT a substitute for the score-head clamp (which prevents Inf from a
wild linear projection). The clamp keeps well-behaved weights finite; the
guard catches the residual cases (NaN weights, NaN inputs that bypass the
clamp because they originate upstream).
"""

from __future__ import annotations

import torch


class NumericalError(RuntimeError):
    """Raised when a model tensor contains NaN or Inf despite the guards.

    The message includes the tensor name, shape, and the non-finite count so
    the failure is actionable (which tensor, how bad).
    """


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    """Raise :class:`NumericalError` if ``tensor`` contains any NaN or Inf.

    Uses ``torch.isfinite`` which is False for both NaN and +/-Inf. The check
    is a single elementwise op + a reduction, so it is cheap relative to the
    forward pass.
    """
    if not torch.isfinite(tensor).all():
        non_finite = int((~torch.isfinite(tensor)).sum().item())
        raise NumericalError(
            f"Model V2 numerical guard failed: tensor {name!r} "
            f"(shape {tuple(tensor.shape)}) contains {non_finite} non-finite "
            f"value(s) (NaN or Inf). This usually indicates corrupted weights "
            f"or a corrupted input; refusing to propagate non-finite values "
            f"into the loss / decision policy."
        )
