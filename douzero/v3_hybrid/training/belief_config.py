"""Identity-bound H4 belief loss and optimizer phase schedule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

BELIEF_MODE_DISABLED = "disabled"
BELIEF_MODE_AUXILIARY = "auxiliary"
BELIEF_MODE_ALTERNATING = "alternating"

BELIEF_PHASE_DISABLED = "disabled"
BELIEF_PHASE_AUXILIARY = "auxiliary"
BELIEF_PHASE_POLICY = "policy"
BELIEF_PHASE_SUPERVISED = "belief"
BELIEF_PHASE_SHARED = "joint_shared_encoder"


@dataclass(frozen=True)
class V3H4BeliefTrainingConfig:
    """H4 loss graph, optimizer ownership, and learner-update schedule."""

    enabled: bool = False
    mode: str = BELIEF_MODE_DISABLED
    lambda_belief: float = 0.0
    learning_rate: float = 1e-4
    max_grad_norm: float = 40.0
    lambda_count_reg: float = 0.0
    lambda_entropy_reg: float = 0.0
    policy_updates_per_cycle: int = 1
    belief_updates_per_cycle: int = 1
    shared_updates_per_cycle: int = 0
    shared_encoder_updates: bool = False

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("belief enabled must be bool")
        if self.mode not in {
            BELIEF_MODE_DISABLED,
            BELIEF_MODE_AUXILIARY,
            BELIEF_MODE_ALTERNATING,
        }:
            raise ValueError("unsupported H4 belief training mode")
        if not isinstance(self.shared_encoder_updates, bool):
            raise TypeError("shared_encoder_updates must be bool")
        for name in (
            "lambda_belief",
            "learning_rate",
            "max_grad_norm",
            "lambda_count_reg",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.lambda_entropy_reg, bool)
            or not isinstance(self.lambda_entropy_reg, (int, float))
            or not math.isfinite(self.lambda_entropy_reg)
        ):
            raise ValueError("lambda_entropy_reg must be finite")
        for name in (
            "policy_updates_per_cycle",
            "belief_updates_per_cycle",
            "shared_updates_per_cycle",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not self.enabled:
            if self.mode != BELIEF_MODE_DISABLED or self.lambda_belief != 0.0:
                raise ValueError(
                    "disabled H4 belief must use mode='disabled' and zero loss"
                )
            if self.shared_encoder_updates or self.shared_updates_per_cycle:
                raise ValueError("disabled H4 belief cannot own shared parameters")
            return
        if self.mode == BELIEF_MODE_DISABLED:
            raise ValueError("enabled H4 belief requires an active mode")
        if self.lambda_belief <= 0.0:
            raise ValueError("enabled H4 belief requires lambda_belief > 0")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("enabled H4 belief optimizer values must be positive")
        if self.mode == BELIEF_MODE_AUXILIARY:
            if self.shared_updates_per_cycle != 0:
                raise ValueError("auxiliary mode updates shared encoders in the same step")
        else:
            if self.policy_updates_per_cycle < 1 or self.belief_updates_per_cycle < 1:
                raise ValueError("alternating mode requires policy and belief phases")
            if self.shared_encoder_updates != (self.shared_updates_per_cycle > 0):
                raise ValueError(
                    "alternating shared phase count must match shared_encoder_updates"
                )

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            **asdict(self),
            "loss_normalization": "real_decisions_role_weight_once_v1",
            "posterior_policy_gradient": "always_detached_v1",
            "phase_clock": "eligible_learner_updates_v1",
        }

    def stable_hash(self) -> str:
        encoded = json.dumps(
            self.compatibility_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def phase_at(self, eligible_update: int) -> str:
        if isinstance(eligible_update, bool) or not isinstance(eligible_update, int):
            raise TypeError("eligible_update must be an int")
        if eligible_update < 0:
            raise ValueError("eligible_update must be non-negative")
        if not self.enabled:
            return BELIEF_PHASE_DISABLED
        if self.mode == BELIEF_MODE_AUXILIARY:
            return BELIEF_PHASE_AUXILIARY
        cycle = (
            self.policy_updates_per_cycle
            + self.belief_updates_per_cycle
            + self.shared_updates_per_cycle
        )
        offset = eligible_update % cycle
        if offset < self.policy_updates_per_cycle:
            return BELIEF_PHASE_POLICY
        offset -= self.policy_updates_per_cycle
        if offset < self.belief_updates_per_cycle:
            return BELIEF_PHASE_SUPERVISED
        return BELIEF_PHASE_SHARED


__all__ = [name for name in globals() if name.startswith("BELIEF_")] + [
    "V3H4BeliefTrainingConfig"
]
