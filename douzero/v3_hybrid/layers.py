"""Shared fusion and role-specific residual layers for H1."""

from __future__ import annotations

import torch
from torch import nn


class PreNormResidualMLP(nn.Module):
    """Pre-norm residual MLP with no batch-dependent state."""

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.norm(value)
        residual = self.fc1(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        residual = self.dropout(residual)
        return value + residual


class SharedStateActionFusion(nn.Module):
    """Fuse once-per-decision state/history with each legal action."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(hidden_size * 3, hidden_size)
        self.blocks = nn.ModuleList(
            PreNormResidualMLP(hidden_size, dropout) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def _finish(self, fused: torch.Tensor) -> torch.Tensor:
        fused = self.input_projection(fused)
        for block in self.blocks:
            fused = block(fused)
        return self.output_norm(fused)

    def forward(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape != (self.hidden_size,) or history.shape != (self.hidden_size,):
            raise ValueError("scalar state and history must have shape (hidden_size,)")
        if actions.ndim != 2 or actions.shape[-1] != self.hidden_size:
            raise ValueError("scalar actions must have shape (A, hidden_size)")
        count = actions.shape[0]
        if count < 1:
            raise ValueError("at least one legal action is required")
        fused = torch.cat(
            (
                state.unsqueeze(0).expand(count, -1),
                history.unsqueeze(0).expand(count, -1),
                actions,
            ),
            dim=-1,
        )
        return self._finish(fused)

    def forward_batched(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.hidden_size:
            raise ValueError("batched state must have shape (B, hidden_size)")
        if history.shape != state.shape:
            raise ValueError("batched history shape must match state")
        if actions.ndim != 3 or actions.shape[0] != state.shape[0]:
            raise ValueError("batched actions must have shape (B, A, hidden_size)")
        if actions.shape[-1] != self.hidden_size or actions.shape[1] < 1:
            raise ValueError("batched actions have invalid width or zero actions")
        count = actions.shape[1]
        fused = torch.cat(
            (
                state.unsqueeze(1).expand(-1, count, -1),
                history.unsqueeze(1).expand(-1, count, -1),
                actions,
            ),
            dim=-1,
        )
        return self._finish(fused)


class ChannelGate(nn.Module):
    """Action-local SE-style channel gate preserving action permutation order."""

    def __init__(self, hidden_size: int, reduction: int) -> None:
        super().__init__()
        reduced = hidden_size // reduction
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, reduced)
        self.up = nn.Linear(reduced, hidden_size)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.norm(value)
        gate = self.down(gate)
        gate = self.activation(gate)
        gate = torch.sigmoid(self.up(gate))
        return value * gate


class RoleAdapter(nn.Module):
    """Independent residual specialization for one physical role."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        *,
        channel_gate: bool,
        gate_reduction: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            PreNormResidualMLP(hidden_size, dropout) for _ in range(num_layers)
        )
        self.gate = (
            ChannelGate(hidden_size, gate_reduction) if channel_gate else None
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        if self.gate is not None:
            value = value + self.gate(value)
        return self.output_norm(value)


class RoleValueHeads(nn.Module):
    """Independent DMC Q, win, and conditional-score heads for one role."""

    def __init__(
        self,
        hidden_size: int,
        *,
        score_clamp: float,
        dmc_clamp: float,
    ) -> None:
        super().__init__()
        self.dmc_head = nn.Linear(hidden_size, 1)
        self.win_head = nn.Linear(hidden_size, 1)
        self.score_win_head = nn.Linear(hidden_size, 1)
        self.score_loss_head = nn.Linear(hidden_size, 1)
        self.score_clamp = float(score_clamp)
        self.dmc_clamp = float(dmc_clamp)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        dmc_q = torch.clamp(
            self.dmc_head(value), -self.dmc_clamp, self.dmc_clamp
        )
        win_logit = self.win_head(value)
        score_if_win = torch.clamp(
            self.score_win_head(value), -self.score_clamp, self.score_clamp
        )
        score_if_loss = torch.clamp(
            self.score_loss_head(value), -self.score_clamp, self.score_clamp
        )
        p_win = torch.sigmoid(win_logit)
        score_mean = (
            p_win.detach() * score_if_win
            + (1.0 - p_win.detach()) * score_if_loss
        )
        return {
            "dmc_q": dmc_q,
            "win_logit": win_logit,
            "score_if_win": score_if_win,
            "score_if_loss": score_if_loss,
            "p_win": p_win,
            "score_mean": score_mean,
        }
