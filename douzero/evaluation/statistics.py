"""Deal-clustered statistics for the P15 paired evaluation protocol."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate and percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    confidence_level: float
    paired_deals: int
    bootstrap_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence_level": self.confidence_level,
            "paired_deals": self.paired_deals,
            "bootstrap_samples": self.bootstrap_samples,
        }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def deal_cluster_means(
    observations: Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Collapse legs/seat rotations to one value per independent deal."""
    grouped: dict[str, list[float]] = {}
    for deal_id, value in observations:
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap observations must be finite")
        grouped.setdefault(str(deal_id), []).append(float(value))
    return {
        deal_id: sum(values) / len(values)
        for deal_id, values in grouped.items()
    }


def paired_bootstrap_ci(
    deal_values: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap deal-level paired values, never individual actions/seats.

    ``deal_values`` must already contain exactly one scalar per deal. Call
    :func:`deal_cluster_means` when a deal has multiple mirrored legs or seat
    rotations. Resampling those legs independently would produce falsely tight
    intervals and is intentionally impossible through this API.
    """
    if not deal_values:
        raise ValueError("paired bootstrap requires at least one deal")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    values = [float(value) for value in deal_values.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired bootstrap values must be finite")

    estimate = sum(values) / len(values)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        boot.append(sum(rng.choice(values) for _ in values) / len(values))
    alpha = (1.0 - confidence_level) / 2.0
    low = _quantile(boot, alpha)
    high = _quantile(boot, 1.0 - alpha)
    return ConfidenceInterval(
        estimate=estimate,
        # Downstream promotion records require the reported interval to
        # contain its point estimate. With very few bootstrap draws, Monte
        # Carlo quantiles can otherwise exclude it by sampling accident.
        low=min(low, estimate),
        high=max(high, estimate),
        confidence_level=confidence_level,
        paired_deals=len(values),
        bootstrap_samples=samples,
    )


def percentile(values: Sequence[float], probability: float) -> float:
    """Public percentile helper used for latency reporting."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return _quantile(values, probability)
