"""P15-paired-evaluation promotion gate with auditable thresholds."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PromotionEvaluation:
    candidate_policy_id: str
    incumbent_policy_id: str
    paired_games: int
    estimate: float
    ci_low: float
    ci_high: float
    evaluator_protocol: str = "p15_paired_v1"
    deal_set_id: str = ""

    def __post_init__(self) -> None:
        if self.paired_games < 0:
            raise ValueError("paired_games must be non-negative")
        if not all(math.isfinite(value) for value in (
            self.estimate, self.ci_low, self.ci_high
        )):
            raise ValueError("promotion estimate and confidence interval must be finite")
        if not self.ci_low <= self.estimate <= self.ci_high:
            raise ValueError("promotion confidence interval does not contain estimate")


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    evaluation: PromotionEvaluation
    min_pairs: int
    min_ci_lower_bound: float
    decided_at: str


class PromotionGate:
    """Promote only from P15 paired results whose lower CI clears the gate."""

    def __init__(
        self,
        *,
        min_pairs: int,
        min_ci_lower_bound: float,
        audit_path: str | None = None,
    ) -> None:
        if min_pairs < 1:
            raise ValueError("min_pairs must be positive")
        if not math.isfinite(min_ci_lower_bound):
            raise ValueError("min_ci_lower_bound must be finite")
        self.min_pairs = min_pairs
        self.min_ci_lower_bound = float(min_ci_lower_bound)
        self.audit_path = Path(audit_path) if audit_path else None

    def decide(self, evaluation: PromotionEvaluation) -> PromotionDecision:
        if evaluation.evaluator_protocol != "p15_paired_v1":
            promoted = False
            reason = "evaluation did not use the P15 paired protocol"
        elif evaluation.paired_games < self.min_pairs:
            promoted = False
            reason = "paired sample count is below the configured minimum"
        elif evaluation.ci_low < self.min_ci_lower_bound:
            promoted = False
            reason = "confidence-interval lower bound did not clear the threshold"
        else:
            promoted = True
            reason = "P15 paired confidence interval cleared the promotion gate"
        decision = PromotionDecision(
            promoted=promoted,
            reason=reason,
            evaluation=evaluation,
            min_pairs=self.min_pairs,
            min_ci_lower_bound=self.min_ci_lower_bound,
            decided_at=datetime.now(timezone.utc).isoformat(),
        )
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as handle:
                json.dump(asdict(decision), handle, sort_keys=True)
                handle.write("\n")
        return decision
