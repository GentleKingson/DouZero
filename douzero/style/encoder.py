"""Trainable encoder for public opponent-style statistics."""

from __future__ import annotations

import torch
from torch import nn

from .features import (
    STYLE_FEATURE_WIDTH,
    STYLE_NUM_OPPONENTS,
    STYLE_PER_OPPONENT_WIDTH,
)


class StyleEncoder(nn.Module):
    """Encode two opponent rows with learned unknown/cold-start embeddings."""

    def __init__(self, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if (
            isinstance(output_dim, bool)
            or not isinstance(output_dim, int)
            or output_dim <= 0
        ):
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        hidden = max(16, output_dim // 2) if hidden_dim is None else hidden_dim
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden}")
        self.output_dim = output_dim
        self.per_opponent = nn.Sequential(
            nn.Linear(STYLE_PER_OPPONENT_WIDTH, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.unknown_embedding = nn.Parameter(
            torch.zeros(STYLE_NUM_OPPONENTS, hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(STYLE_NUM_OPPONENTS * hidden, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode ``(..., STYLE_FEATURE_WIDTH)`` public statistic vectors."""

        if features.shape[-1] != STYLE_FEATURE_WIDTH:
            raise ValueError(
                f"style features last dim {features.shape[-1]} != "
                f"STYLE_FEATURE_WIDTH {STYLE_FEATURE_WIDTH}"
            )
        rows = features.reshape(
            *features.shape[:-1], STYLE_NUM_OPPONENTS, STYLE_PER_OPPONENT_WIDTH
        )
        encoded = self.per_opponent(rows)
        observed = rows[..., :1] > 0.5
        unknown = self.unknown_embedding
        while unknown.ndim < encoded.ndim:
            unknown = unknown.unsqueeze(0)
        encoded = torch.where(observed, encoded, unknown)
        return self.fusion(encoded.flatten(start_dim=-2))
