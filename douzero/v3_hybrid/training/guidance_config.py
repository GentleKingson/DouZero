"""Public-safe configuration for H3 listwise Oracle guidance losses."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleGuidanceLossConfig:
    top_k: int = 4
    ranking_margin: float = 0.1
    lambda_kl: float = 1.0
    lambda_ranking: float = 0.25
    lambda_chosen_value: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive int")
        for name in (
            "ranking_margin", "lambda_kl", "lambda_ranking", "lambda_chosen_value"
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
