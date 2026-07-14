"""Configuration and identity for the optional P09 strategy feature layer."""

from __future__ import annotations

from dataclasses import dataclass

STRATEGY_FEATURE_VERSION: str = "strategy_v1"


@dataclass(frozen=True)
class StrategyFeatureConfig:
    """Ablation controls for deterministic public strategy features.

    Disabled groups retain their fixed feature columns as zeros.  This makes
    ablations comparable without changing tensor widths, while the model
    checkpoint identity still records every toggle.
    """

    hand_enabled: bool = True
    structure_enabled: bool = True
    control_enabled: bool = True
    cooperation_enabled: bool = True
    risk_enabled: bool = True
    node_budget: int = 500
    time_budget_ms: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.node_budget, bool) or self.node_budget <= 0:
            raise ValueError(f"node_budget must be a positive int, got {self.node_budget!r}")
        if isinstance(self.time_budget_ms, bool) or self.time_budget_ms < 0:
            raise ValueError(
                f"time_budget_ms must be a non-negative int, got {self.time_budget_ms!r}"
            )
