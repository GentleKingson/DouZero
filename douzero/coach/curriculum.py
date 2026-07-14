"""Deterministic opening samplers and an auditable curriculum schedule."""

from __future__ import annotations

import json
import math
import os
import random
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from douzero.env.rules import RuleSet

from .records import OpeningRecord, random_opening


TRUE_RANDOM = "true_random"
BALANCED = "balanced"
HARD_FOR_ROLE = "hard_for_role"
MIXTURE = "mixture"
GUIDED_MODES = frozenset({BALANCED, HARD_FOR_ROLE})
SAMPLING_MODES = frozenset({TRUE_RANDOM, BALANCED, HARD_FOR_ROLE, MIXTURE})


class CoachPredictor(Protocol):
    """Minimal interface required by :class:`OpeningSampler`."""

    def predict(self, opening: OpeningRecord, policy_version: str) -> float: ...


def _validate_proportions(name: str, values: Mapping[str, float]) -> dict[str, float]:
    expected = {TRUE_RANDOM, BALANCED, HARD_FOR_ROLE}
    if set(values) != expected:
        raise ValueError(f"{name} proportions must have keys {sorted(expected)}")
    result = {key: float(value) for key, value in values.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"{name} proportions must be finite and non-negative")
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise ValueError(f"{name} proportions must sum to 1.0")
    return result


@dataclass(frozen=True)
class CurriculumSchedule:
    """Three-phase coach mixture with an enforced real-deal floor."""

    early_until: float = 0.30
    mid_until: float = 0.70
    early: Mapping[str, float] = None  # type: ignore[assignment]
    middle: Mapping[str, float] = None  # type: ignore[assignment]
    late: Mapping[str, float] = None  # type: ignore[assignment]
    min_true_random_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.early is None:
            object.__setattr__(
                self, "early",
                {TRUE_RANDOM: 0.20, BALANCED: 0.70, HARD_FOR_ROLE: 0.10},
            )
        if self.middle is None:
            object.__setattr__(
                self, "middle",
                {TRUE_RANDOM: 0.50, BALANCED: 0.30, HARD_FOR_ROLE: 0.20},
            )
        if self.late is None:
            object.__setattr__(
                self, "late",
                {TRUE_RANDOM: 0.90, BALANCED: 0.05, HARD_FOR_ROLE: 0.05},
            )
        if not 0.0 <= self.early_until < self.mid_until <= 1.0:
            raise ValueError("schedule boundaries must satisfy 0 <= early < mid <= 1")
        if not math.isfinite(self.min_true_random_ratio) or not 0.0 <= self.min_true_random_ratio <= 1.0:
            raise ValueError("min_true_random_ratio must be in [0, 1]")
        for name in ("early", "middle", "late"):
            validated = _validate_proportions(name, getattr(self, name))
            if validated[TRUE_RANDOM] < self.min_true_random_ratio:
                raise ValueError(
                    f"{name}.true_random ({validated[TRUE_RANDOM]}) is below "
                    f"min_true_random_ratio ({self.min_true_random_ratio})"
                )
            object.__setattr__(self, name, validated)

    def phase(self, progress: float) -> str:
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be finite and in [0, 1]")
        if progress < self.early_until:
            return "early"
        if progress < self.mid_until:
            return "middle"
        return "late"

    def proportions(self, progress: float) -> dict[str, float]:
        """Return a copy of the configured proportions at ``progress``."""

        return dict(getattr(self, self.phase(progress)))


@dataclass(frozen=True)
class SamplingRecord:
    """One reconstructable training-time opening selection decision."""

    sample_index: int
    opening_id: str
    requested_mode: str
    selected_strategy: str
    phase: str
    progress: float
    configured_proportions: dict[str, float]
    strategy_probability: float
    predicted_landlord_win: float | None
    policy_version: str
    ruleset_hash: str
    cumulative_counts: dict[str, int]
    cumulative_distribution: dict[str, float]
    hard_role: str


class CurriculumAuditLogger:
    """Durable JSONL logger for replaying the actual opening distribution."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def append(self, record: SamplingRecord) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("CurriculumAuditLogger is single-process")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True, ensure_ascii=True) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


class OpeningSampler:
    """Sample true, balanced, role-hard, or scheduled-mixture openings."""

    def __init__(
        self,
        *,
        ruleset: RuleSet | None = None,
        policy_version: str,
        coach: CoachPredictor | None = None,
        mode: str = MIXTURE,
        schedule: CurriculumSchedule | None = None,
        hard_role: str = "landlord",
        candidate_pool_size: int = 16,
        seed: int = 0,
        logger: CurriculumAuditLogger | None = None,
    ) -> None:
        if mode not in SAMPLING_MODES:
            raise ValueError(f"unsupported opening sampling mode {mode!r}")
        if not policy_version:
            raise ValueError("policy_version must be non-empty")
        if hard_role not in ("landlord", "farmer"):
            raise ValueError("hard_role must be 'landlord' or 'farmer'")
        if (
            isinstance(candidate_pool_size, bool)
            or not isinstance(candidate_pool_size, int)
            or candidate_pool_size < 1
        ):
            raise ValueError("candidate_pool_size must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        if mode in GUIDED_MODES | {MIXTURE} and coach is None:
            raise ValueError(f"sampling mode {mode!r} requires a coach")
        self.ruleset = ruleset or RuleSet.legacy()
        self.policy_version = policy_version
        self.coach = coach
        self.mode = mode
        self.schedule = schedule or CurriculumSchedule()
        self.hard_role = hard_role
        self.candidate_pool_size = candidate_pool_size
        self.rng = random.Random(seed)
        self.logger = logger
        self.counts: Counter[str] = Counter()
        self.sample_index = 0

    def sample(self, *, progress: float = 0.0) -> tuple[OpeningRecord, SamplingRecord]:
        """Select one opening and return it with complete audit provenance."""

        phase = self.schedule.phase(progress)
        if self.mode == MIXTURE:
            proportions = self.schedule.proportions(progress)
            selected = self._weighted_strategy(proportions)
            strategy_probability = proportions[selected]
        elif self.mode in GUIDED_MODES:
            # Fixed guided modes remain useful for ablations, but production
            # collection still enforces the configured real-deal floor.
            real_ratio = self.schedule.min_true_random_ratio
            proportions = {
                TRUE_RANDOM: real_ratio,
                BALANCED: (1.0 - real_ratio) if self.mode == BALANCED else 0.0,
                HARD_FOR_ROLE: (
                    (1.0 - real_ratio) if self.mode == HARD_FOR_ROLE else 0.0
                ),
            }
            selected = self._weighted_strategy(proportions)
            strategy_probability = proportions[selected]
        else:
            selected = self.mode
            proportions = {
                TRUE_RANDOM: float(selected == TRUE_RANDOM),
                BALANCED: float(selected == BALANCED),
                HARD_FOR_ROLE: float(selected == HARD_FOR_ROLE),
            }
            strategy_probability = 1.0

        if selected == TRUE_RANDOM:
            opening = random_opening(self.rng, self.ruleset)
        else:
            opening = self._guided_opening(selected)

        probability = None
        if self.coach is not None:
            probability = float(self.coach.predict(opening, self.policy_version))
            if not 0.0 <= probability <= 1.0:
                raise ValueError("coach prediction must be in [0, 1]")

        self.counts[selected] += 1
        self.sample_index += 1
        total = sum(self.counts.values())
        cumulative_counts = {
            key: self.counts[key] for key in (TRUE_RANDOM, BALANCED, HARD_FOR_ROLE)
        }
        record = SamplingRecord(
            sample_index=self.sample_index - 1,
            opening_id=opening.opening_id,
            requested_mode=self.mode,
            selected_strategy=selected,
            phase=phase,
            progress=progress,
            configured_proportions=proportions,
            strategy_probability=strategy_probability,
            predicted_landlord_win=probability,
            policy_version=self.policy_version,
            ruleset_hash=self.ruleset.stable_hash(),
            cumulative_counts=cumulative_counts,
            cumulative_distribution={
                key: count / total for key, count in cumulative_counts.items()
            },
            hard_role=self.hard_role,
        )
        if self.logger is not None:
            self.logger.append(record)
        return opening, record

    def _weighted_strategy(self, proportions: Mapping[str, float]) -> str:
        value = self.rng.random()
        cumulative = 0.0
        for strategy in (TRUE_RANDOM, BALANCED, HARD_FOR_ROLE):
            cumulative += proportions[strategy]
            if value < cumulative:
                return strategy
        return HARD_FOR_ROLE

    def _guided_opening(self, strategy: str) -> OpeningRecord:
        if self.coach is None:
            raise RuntimeError("guided sampling requires a coach")
        candidates = [
            random_opening(self.rng, self.ruleset)
            for _ in range(self.candidate_pool_size)
        ]
        predictions = [
            float(self.coach.predict(candidate, self.policy_version))
            for candidate in candidates
        ]
        if any(not 0.0 <= value <= 1.0 for value in predictions):
            raise ValueError("coach prediction must be in [0, 1]")
        if strategy == BALANCED:
            index = min(range(len(candidates)), key=lambda i: abs(predictions[i] - 0.5))
        elif strategy == HARD_FOR_ROLE:
            key = (lambda i: predictions[i]) if self.hard_role == "landlord" else (
                lambda i: -predictions[i]
            )
            index = min(range(len(candidates)), key=key)
        else:  # pragma: no cover - internal invariant
            raise RuntimeError(f"unexpected guided strategy {strategy}")
        return candidates[index]
