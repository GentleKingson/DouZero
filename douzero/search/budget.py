"""Cooperative hard budgets shared by rollout and endgame search."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised internally when a configured hard search limit is exhausted."""


@dataclass(frozen=True)
class SearchConfig:
    """Deployment search settings.

    Search is disabled by default. A zero node, rollout, or time budget also
    forces the exact base-policy fallback.
    """

    enabled: bool = False
    top_k: int = 3
    belief_samples: int = 8
    rollout_depth: int = 12
    endgame_cards_threshold: int = 12
    max_nodes: int = 20_000
    max_rollouts: int = 64
    max_milliseconds: int = 100
    risk_penalty: float = 0.0
    selection_mode: str = "win_then_score"
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("SearchConfig.enabled must be bool")
        for name in (
            "top_k", "belief_samples", "rollout_depth",
            "endgame_cards_threshold", "max_nodes", "max_rollouts",
            "max_milliseconds", "seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"SearchConfig.{name} must be a non-negative int")
        if self.top_k < 1 or self.belief_samples < 1:
            raise ValueError("SearchConfig.top_k and belief_samples must be positive")
        if self.selection_mode not in ("win", "score", "win_then_score"):
            raise ValueError(
                "SearchConfig.selection_mode must be win, score, or win_then_score"
            )
        if (
            isinstance(self.risk_penalty, bool)
            or not isinstance(self.risk_penalty, (int, float))
            or not math.isfinite(self.risk_penalty)
            or self.risk_penalty < 0.0
        ):
            raise ValueError("SearchConfig.risk_penalty must be non-negative finite")


class SearchBudget:
    """Mutable cooperative budget with node, rollout, and wall-clock limits."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.nodes = 0
        self.rollouts = 0
        self.started_at = time.monotonic()

    @property
    def elapsed_milliseconds(self) -> float:
        return (time.monotonic() - self.started_at) * 1000.0

    def check(self) -> None:
        """Raise before more work when any hard limit has been exhausted."""
        if self.config.max_nodes == 0:
            raise BudgetExceeded("max_nodes is zero")
        if self.config.max_rollouts == 0:
            raise BudgetExceeded("max_rollouts is zero")
        if self.config.max_milliseconds == 0:
            raise BudgetExceeded("max_milliseconds is zero")
        if self.elapsed_milliseconds >= self.config.max_milliseconds:
            raise BudgetExceeded("max_milliseconds exhausted")

    def visit_node(self) -> None:
        """Charge one expanded search node."""
        self.check()
        if self.nodes >= self.config.max_nodes:
            raise BudgetExceeded("max_nodes exhausted")
        self.nodes += 1

    def start_rollout(self) -> None:
        """Charge one candidate/sample rollout."""
        self.check()
        if self.rollouts >= self.config.max_rollouts:
            raise BudgetExceeded("max_rollouts exhausted")
        self.rollouts += 1
