"""State-action fusion + role conditioning for Model V2 (P05).

Combines, for each legal action, the shared state trunk, the shared history
summary, the per-action embedding, and the acting-role embedding into a fused
representation that the heads (:mod:`douzero.models_v2.heads`) score.

Design
------
This is a "gated MLP" fusion rather than bilinear or cross-attention. Reasons:

- It is simple, fully testable on CPU, and avoids the quadratic memory of a
  bilinear layer over a wide trunk.
- It keeps the per-action cost linear in the number of legal actions (the
  shared state/history are computed once, then broadcast/expanded and
  concatenated with each action embedding — exactly the P04 factorized
  contract, generalized to the richer V2 inputs).
- Cross-attention is a plausible P05+ alternative; the spec allows choosing a
  "simple, testable scheme", and gated MLP is that.

Role conditioning
-----------------
A learned ``nn.Embedding`` maps the acting role (landlord / landlord_up /
landlord_down) to a ``role_embedding_dim`` vector. This preserves the
landlord/farmer positional differences (AGENTS.md "Model rules") without
maintaining three separate models. The embedding is concatenated into every
fused action vector so the heads see the role at scoring time.

The fusion stack is pre-norm residual MLP blocks with LayerNorm (no BatchNorm —
see AGENTS.md "Model rules" and config.py for the rationale).
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + MLP(LayerNorm(x))``."""

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class StateActionFusion(nn.Module):
    """Fuse the shared state+history trunk with each action + role.

    Parameters
    ----------
    hidden_size:
        Trunk / action embedding width (the encoders all project into this).
    role_embedding_dim:
        Width of the learned role embedding table. 0 disables role conditioning
        (not recommended; the farmer positional difference would be lost).
    num_layers:
        Number of residual MLP blocks applied to the fused representation.
    dropout:
        Dropout in the residual blocks. 0 by default (deterministic eval).
    """

    def __init__(
        self,
        hidden_size: int,
        role_embedding_dim: int,
        num_roles: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if role_embedding_dim < 0:
            raise ValueError(f"role_embedding_dim must be non-negative, got {role_embedding_dim}")
        if num_roles <= 0:
            raise ValueError(f"num_roles must be positive, got {num_roles}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_size = hidden_size
        self.role_embedding_dim = role_embedding_dim
        self.num_roles = num_roles
        self.num_layers = num_layers

        if role_embedding_dim > 0:
            self.role_embed = nn.Embedding(num_roles, role_embedding_dim)
            fused_width = hidden_size * 2 + role_embedding_dim
        else:
            self.role_embed = None
            fused_width = hidden_size * 2

        self.input_proj = nn.Linear(fused_width, hidden_size)
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(hidden_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        state_trunk: torch.Tensor,
        history_summary: torch.Tensor,
        action_embeddings: torch.Tensor,
        role_index: int,
    ) -> torch.Tensor:
        """Fuse the shared trunk with each action under the acting role.

        Parameters
        ----------
        state_trunk:
            Shape ``(hidden_size,)`` — the shared state trunk (one per decision).
        history_summary:
            Shape ``(hidden_size,)`` — the shared history summary (one per
            decision).
        action_embeddings:
            Shape ``(N, hidden_size)`` — one embedding per legal action.
        role_index:
            Integer index of the acting role into the role embedding table.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden_size)`` — the fused, role-conditioned action
            representations, ready for the heads.
        """
        n = action_embeddings.shape[0]
        if n == 0:
            raise ValueError("action_embeddings has zero rows (no legal actions)")

        # Broadcast the shared trunk to every action. expand is a view (no copy).
        state_b = state_trunk.unsqueeze(0).expand(n, -1)
        history_b = history_summary.unsqueeze(0).expand(n, -1)

        if self.role_embed is not None:
            if not (0 <= role_index < self.num_roles):
                raise ValueError(
                    f"role_index {role_index} out of range [0, {self.num_roles})"
                )
            role_vec = self.role_embed.weight[role_index]  # (role_embedding_dim,)
            role_b = role_vec.unsqueeze(0).expand(n, -1)
            fused = torch.cat([state_b, history_b, role_b], dim=-1)
        else:
            fused = torch.cat([state_b, history_b], dim=-1)

        h = self.input_proj(fused)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)
