"""Numerically guarded mixed-precision optimizer steps."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerStepResult:
    """Detached diagnostics for one completed optimizer step."""

    loss: torch.Tensor
    grad_norm: torch.Tensor
    amp_used: bool
    fell_back: bool


class SafeMixedPrecision:
    """Run autocast/GradScaler with a one-time float32 anomaly fallback."""

    def __init__(self, device: torch.device, *, enabled: bool = False,
                 dtype: str = "float16", fallback_on_nonfinite: bool = True) -> None:
        self.device = torch.device(device)
        if dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp dtype must be 'float16' or 'bfloat16'")
        if self.device.type == "cpu" and enabled and dtype != "bfloat16":
            raise ValueError("CPU autocast is opt-in bfloat16 only")
        if self.device.type not in {"cpu", "cuda"} and enabled:
            raise ValueError(f"autocast is unsupported on {self.device.type!r}")
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.enabled = bool(enabled)
        self.fallback_on_nonfinite = bool(fallback_on_nonfinite)
        self.fallback_count = 0
        self._scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.enabled and self.device.type == "cuda"
        )

    def autocast(self, *, enabled: bool | None = None):
        """Return the configured autocast context."""
        active = self.enabled if enabled is None else enabled
        if not active:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def step(self, loss_closure: Callable[[], torch.Tensor],
             optimizer: torch.optim.Optimizer,
             parameters: Iterable[nn.Parameter], *,
             max_grad_norm: float,
             clip_grad_norm: Callable | None = None) -> OptimizerStepResult:
        """Take one finite optimizer step, retrying once in float32 after AMP."""
        params = list(parameters)
        attempted_amp = self.enabled
        try:
            return self._attempt(loss_closure, optimizer, params,
                                 max_grad_norm, attempted_amp, clip_grad_norm)
        except FloatingPointError:
            if not attempted_amp or not self.fallback_on_nonfinite:
                raise
            self.enabled = False
            self.fallback_count += 1
            optimizer.zero_grad(set_to_none=True)
            result = self._attempt(loss_closure, optimizer, params,
                                   max_grad_norm, False, clip_grad_norm)
            return OptimizerStepResult(result.loss, result.grad_norm, False, True)

    def _attempt(self, closure, optimizer, params, max_grad_norm, use_amp,
                 clip_grad_norm):
        optimizer.zero_grad(set_to_none=True)
        with self.autocast(enabled=use_amp):
            loss = closure()
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"non-finite loss: {loss.detach().float().item()!r}")
        if self._scaler.is_enabled() and use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
        else:
            loss.backward()
        clip = clip_grad_norm or nn.utils.clip_grad_norm_
        grad_norm = clip(params, max_grad_norm, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm.detach()).all():
            raise FloatingPointError(
                f"non-finite gradient norm: {grad_norm.detach().float().item()!r}"
            )
        if self._scaler.is_enabled() and use_amp:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        return OptimizerStepResult(loss.detach(), grad_norm.detach(), use_amp, False)
