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

    def step(
        self,
        loss_closure: Callable[[], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
        *,
        max_grad_norm: float,
        clip_grad_norm: Callable | None = None,
        collective_all_true: Callable[[bool], bool] | None = None,
        synchronize_abandoned_backward: bool = False,
        capture_retry_state: Callable[[], object] | None = None,
        restore_retry_state: Callable[[object], None] | None = None,
    ) -> OptimizerStepResult:
        """Take one finite optimizer step, retrying once in float32 after AMP."""
        if (capture_retry_state is None) != (restore_retry_state is None):
            raise ValueError(
                "capture_retry_state and restore_retry_state must be provided together"
            )
        params = list(parameters)
        attempted_amp = self.enabled
        retry_state = (
            capture_retry_state()
            if attempted_amp
            and self.fallback_on_nonfinite
            and capture_retry_state is not None
            else None
        )
        try:
            return self._attempt(loss_closure, optimizer, params,
                                 max_grad_norm, attempted_amp, clip_grad_norm,
                                 collective_all_true,
                                 synchronize_abandoned_backward)
        except FloatingPointError:
            if not attempted_amp or not self.fallback_on_nonfinite:
                raise
            self.enabled = False
            self.fallback_count += 1
            optimizer.zero_grad(set_to_none=True)
            if restore_retry_state is not None:
                restore_retry_state(retry_state)
            result = self._attempt(loss_closure, optimizer, params,
                                   max_grad_norm, False, clip_grad_norm,
                                   collective_all_true,
                                   synchronize_abandoned_backward)
            return OptimizerStepResult(result.loss, result.grad_norm, False, True)

    def _attempt(self, closure, optimizer, params, max_grad_norm, use_amp,
                 clip_grad_norm, collective_all_true,
                 synchronize_abandoned_backward):
        optimizer.zero_grad(set_to_none=True)
        with self.autocast(enabled=use_amp):
            loss = closure()
        local_loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        loss_finite = (
            collective_all_true(local_loss_finite)
            if collective_all_true is not None else local_loss_finite
        )
        if not loss_finite:
            if synchronize_abandoned_backward:
                # DDP prepared its reducer during forward. Even though this
                # loss must never reach the optimizer, every rank must finish
                # that reducer iteration before a coordinated retry. Replacing
                # non-finite scalar values produces a synchronization-only
                # backward; its gradients are discarded immediately.
                torch.nan_to_num(
                    loss,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).backward()
            detail = (
                f"non-finite loss: {loss.detach().float().item()!r}"
                if not local_loss_finite else "non-finite loss on a peer rank"
            )
            raise FloatingPointError(detail)
        if self._scaler.is_enabled() and use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
        else:
            loss.backward()
        clip = clip_grad_norm or nn.utils.clip_grad_norm_
        grad_norm = clip(params, max_grad_norm, error_if_nonfinite=False)
        local_grad_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
        grad_finite = (
            collective_all_true(local_grad_finite)
            if collective_all_true is not None else local_grad_finite
        )
        if not grad_finite:
            detail = (
                f"non-finite gradient norm: {grad_norm.detach().float().item()!r}"
                if not local_grad_finite
                else "non-finite gradient norm on a peer rank"
            )
            raise FloatingPointError(detail)
        if self._scaler.is_enabled() and use_amp:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        return OptimizerStepResult(loss.detach(), grad_norm.detach(), use_amp, False)
