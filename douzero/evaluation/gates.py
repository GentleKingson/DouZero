"""Configurable, audit-friendly regression gates for P15 reports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class RegressionGateConfig:
    """Thresholds chosen before evaluation, never tuned by this module."""

    max_p95_latency_ms: float | None = None
    max_brier: float | None = None
    max_ece: float | None = None
    min_overall_win_percentage: float | None = None
    min_role_win_percentage: Mapping[str, float] = field(default_factory=dict)
    required_checks: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.min_role_win_percentage, Mapping):
            raise TypeError("min_role_win_percentage must be a mapping")
        if not isinstance(self.required_checks, Mapping):
            raise TypeError("required_checks must be a mapping")
        for name in ("max_p95_latency_ms", "max_brier", "max_ece"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be non-negative finite or null")
        if self.min_overall_win_percentage is not None:
            if not 0 <= self.min_overall_win_percentage <= 1:
                raise ValueError("min_overall_win_percentage must be in [0, 1]")
        for role, value in self.min_role_win_percentage.items():
            if not 0 <= value <= 1:
                raise ValueError(f"minimum WP for {role} must be in [0, 1]")


def evaluate_regression_gates(
    metrics: Mapping[str, object], config: RegressionGateConfig
) -> dict[str, object]:
    """Evaluate all configured gates without mutating metrics or thresholds."""
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, observed, threshold) -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "threshold": threshold,
        })

    for name, passed in config.required_checks.items():
        add(f"required:{name}", bool(passed), bool(passed), True)

    if config.min_overall_win_percentage is not None:
        observed = float(metrics["overall_win_percentage"])
        add(
            "overall_win_percentage",
            observed >= config.min_overall_win_percentage,
            observed,
            {"min": config.min_overall_win_percentage},
        )
    by_role = metrics["by_role"]
    for role, minimum in config.min_role_win_percentage.items():
        role_metrics = by_role.get(role, {})
        observed = role_metrics.get("win_percentage")
        add(
            f"role_win_percentage:{role}",
            observed is not None and float(observed) >= minimum,
            observed,
            {"min": minimum},
        )
    if config.max_p95_latency_ms is not None:
        observed = float(metrics["inference_latency_ms"]["p95"])
        add(
            "inference_p95_ms",
            math.isfinite(observed) and observed <= config.max_p95_latency_ms,
            observed if math.isfinite(observed) else None,
            {"max": config.max_p95_latency_ms},
        )
    calibration = metrics["calibration"]["overall"]
    for metric_name, maximum in (
        ("brier", config.max_brier),
        ("ece", config.max_ece),
    ):
        if maximum is None:
            continue
        observed = float(calibration[metric_name])
        add(
            f"calibration_{metric_name}",
            math.isfinite(observed) and observed <= maximum,
            observed if math.isfinite(observed) else None,
            {"max": maximum},
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
