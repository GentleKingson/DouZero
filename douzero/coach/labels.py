"""Durable, policy-versioned outcome labels for the opening coach."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .records import OpeningRecord


@dataclass(frozen=True)
class CoachLabel:
    """Observed self-play outcome for one opening and exact policy version."""

    opening: OpeningRecord
    policy_version: str
    policy_step: int
    landlord_win: float
    landlord_score: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported coach label schema_version")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")
        if isinstance(self.policy_step, bool) or not isinstance(self.policy_step, int):
            raise TypeError("policy_step must be an int")
        if self.policy_step < 0:
            raise ValueError("policy_step must be non-negative")
        if self.landlord_win not in (0.0, 1.0):
            raise ValueError("landlord_win must be 0.0 or 1.0")
        if not math.isfinite(self.landlord_score):
            raise ValueError("landlord_score must be finite")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["opening"] = self.opening.to_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping) -> "CoachLabel":
        required = {
            "opening", "policy_version", "policy_step", "landlord_win",
            "landlord_score", "schema_version",
        }
        if set(raw) != required:
            raise ValueError(
                f"coach label fields must be {sorted(required)}, got {sorted(raw)}"
            )
        return cls(
            opening=OpeningRecord.from_dict(raw["opening"]),
            policy_version=str(raw["policy_version"]),
            policy_step=int(raw["policy_step"]),
            landlord_win=float(raw["landlord_win"]),
            landlord_score=float(raw["landlord_score"]),
            schema_version=int(raw["schema_version"]),
        )

    @classmethod
    def from_terminal(
        cls,
        opening: OpeningRecord,
        terminal_result: Mapping,
        *,
        policy_version: str,
        policy_step: int,
    ) -> "CoachLabel":
        """Build a coach label from public terminal self-play results."""

        winner_team = str(terminal_result.get("winner_team", ""))
        if winner_team not in ("landlord", "farmer"):
            raise ValueError("terminal_result is missing a valid winner_team")
        landlord_target = terminal_result.get("team_targets", {}).get("landlord", {})
        if "target_score" not in landlord_target:
            raise ValueError("terminal_result is missing landlord target_score")
        return cls(
            opening=opening,
            policy_version=policy_version,
            policy_step=policy_step,
            landlord_win=float(winner_team == "landlord"),
            landlord_score=float(landlord_target["target_score"]),
        )


class CoachLabelStore:
    """Append-only JSONL store with single-process ownership and fsync."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def append(self, label: CoachLabel) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("CoachLabelStore is single-process")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(label.to_dict(), sort_keys=True, ensure_ascii=True) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def load_fresh(
        self,
        *,
        policy_version: str,
        current_policy_step: int,
        max_age_steps: int,
    ) -> list[CoachLabel]:
        """Load labels for one policy, excluding future and stale outcomes."""

        if current_policy_step < 0 or max_age_steps < 0:
            raise ValueError("current_policy_step and max_age_steps must be non-negative")
        if not self.path.exists():
            return []
        labels: list[CoachLabel] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    label = CoachLabel.from_dict(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid coach label at {self.path}:{line_number}: {exc}"
                    ) from exc
                age = current_policy_step - label.policy_step
                if (
                    label.policy_version == policy_version
                    and 0 <= age <= max_age_steps
                ):
                    labels.append(label)
        return labels


def calibration_metrics(
    probabilities: Iterable[float], targets: Iterable[float], *, bins: int = 10
) -> dict[str, float]:
    """Compute coach Brier score and expected calibration error."""

    probs = list(probabilities)
    truth = list(targets)
    if len(probs) != len(truth) or not probs:
        raise ValueError("probabilities and targets must have equal non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    if any(not 0.0 <= value <= 1.0 for value in probs + truth):
        raise ValueError("probabilities and targets must be in [0, 1]")
    brier = sum((p - y) ** 2 for p, y in zip(probs, truth)) / len(probs)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            i for i, p in enumerate(probs)
            if low <= p < high or (index == bins - 1 and p == 1.0)
        ]
        if members:
            confidence = sum(probs[i] for i in members) / len(members)
            accuracy = sum(truth[i] for i in members) / len(members)
            ece += len(members) / len(probs) * abs(confidence - accuracy)
    return {"brier": brier, "ece": ece, "count": float(len(probs))}
