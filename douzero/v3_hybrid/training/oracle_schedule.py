"""Learner-update schedules for H3 online privileged Oracle guiding."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

ORACLE_PHASE_DISABLED = "disabled"
ORACLE_PHASE_WARMUP = "oracle_warmup"
ORACLE_PHASE_GUIDED = "guided"
ORACLE_PHASE_PUBLIC_FINETUNE = "public_finetune"
ORACLE_PHASE_COMPLETE = "complete"


def _finite_nonnegative(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class OracleGuidingScheduleConfig:
    """Complete identity-bound three-phase H3 schedule.

    The guided phase anneals privileged influence to zero. Every update after
    that boundary is exactly public-only, including calls and data dependency.
    """

    enabled: bool = False
    warmup_updates: int = 0
    guided_updates: int = 0
    finetune_updates: int = 0
    oracle_weight_start: float = 1.0
    oracle_weight_end: float = 0.0
    guidance_weight_start: float = 1.0
    guidance_weight_end: float = 0.0
    temperature_start: float = 2.0
    temperature_end: float = 1.0
    privileged_gate_start: float = 1.0
    privileged_gate_end: float = 0.0

    IDENTITY_VERSION = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name in ("warmup_updates", "guided_updates", "finetune_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name in (
            "oracle_weight_start",
            "oracle_weight_end",
            "guidance_weight_start",
            "guidance_weight_end",
            "temperature_start",
            "temperature_end",
            "privileged_gate_start",
            "privileged_gate_end",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.temperature_start <= 0.0 or self.temperature_end <= 0.0:
            raise ValueError("Oracle temperatures must be positive")
        for name in ("privileged_gate_start", "privileged_gate_end"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.enabled:
            if self.guided_updates < 1:
                raise ValueError("enabled Oracle guiding requires guided_updates >= 1")
            if (
                self.oracle_weight_end != 0.0
                or self.guidance_weight_end != 0.0
                or self.privileged_gate_end != 0.0
            ):
                raise ValueError(
                    "enabled Oracle guiding must reach zero Oracle, guidance, and gate "
                    "before public-only finetune"
                )
        elif any(
            (self.warmup_updates, self.guided_updates, self.finetune_updates)
        ):
            raise ValueError("disabled Oracle schedule must have zero phase lengths")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "phase_semantics": "bounded_finetune_scheduled_noop_tick_v2",
            **asdict(self),
        }

    def stable_hash(self) -> str:
        encoded = json.dumps(
            self.compatibility_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OracleGuidingScheduleConfig":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Oracle schedule fields mismatch")
        return cls(**dict(payload))

    @staticmethod
    def _linear(start: float, end: float, index: int, length: int) -> float:
        if length <= 1:
            return float(end)
        fraction = min(max(index / float(length - 1), 0.0), 1.0)
        return float(start + fraction * (end - start))

    def at(self, learner_update: int) -> "OracleScheduleState":
        if isinstance(learner_update, bool) or not isinstance(learner_update, int):
            raise TypeError("learner_update must be an int")
        if learner_update < 0:
            raise ValueError("learner_update must be non-negative")
        if not self.enabled:
            return OracleScheduleState(
                learner_update=learner_update,
                phase=ORACLE_PHASE_DISABLED,
                phase_update=learner_update,
                oracle_weight=0.0,
                guidance_weight=0.0,
                temperature=self.temperature_end,
                privileged_gate=0.0,
                public_training=True,
                privileged_required=False,
            )
        if learner_update < self.warmup_updates:
            return OracleScheduleState(
                learner_update=learner_update,
                phase=ORACLE_PHASE_WARMUP,
                phase_update=learner_update,
                oracle_weight=float(self.oracle_weight_start),
                guidance_weight=0.0,
                temperature=float(self.temperature_start),
                privileged_gate=float(self.privileged_gate_start),
                public_training=False,
                privileged_required=True,
            )
        guided_index = learner_update - self.warmup_updates
        if guided_index < self.guided_updates:
            guidance = self._linear(
                self.guidance_weight_start,
                self.guidance_weight_end,
                guided_index,
                self.guided_updates,
            )
            gate = self._linear(
                self.privileged_gate_start,
                self.privileged_gate_end,
                guided_index,
                self.guided_updates,
            )
            return OracleScheduleState(
                learner_update=learner_update,
                phase=ORACLE_PHASE_GUIDED,
                phase_update=guided_index,
                oracle_weight=self._linear(
                    self.oracle_weight_start,
                    self.oracle_weight_end,
                    guided_index,
                    self.guided_updates,
                ),
                guidance_weight=guidance,
                temperature=self._linear(
                    self.temperature_start,
                    self.temperature_end,
                    guided_index,
                    self.guided_updates,
                ),
                privileged_gate=gate,
                public_training=True,
                privileged_required=True,
            )
        finetune_index = guided_index - self.guided_updates
        if finetune_index < self.finetune_updates:
            return OracleScheduleState(
                learner_update=learner_update,
                phase=ORACLE_PHASE_PUBLIC_FINETUNE,
                phase_update=finetune_index,
                oracle_weight=0.0,
                guidance_weight=0.0,
                temperature=float(self.temperature_end),
                privileged_gate=0.0,
                public_training=True,
                privileged_required=False,
            )
        return OracleScheduleState(
            learner_update=learner_update,
            phase=ORACLE_PHASE_COMPLETE,
            phase_update=finetune_index - self.finetune_updates,
            oracle_weight=0.0,
            guidance_weight=0.0,
            temperature=float(self.temperature_end),
            privileged_gate=0.0,
            public_training=False,
            privileged_required=False,
        )


@dataclass(frozen=True)
class OracleScheduleState:
    learner_update: int
    phase: str
    phase_update: int
    oracle_weight: float
    guidance_weight: float
    temperature: float
    privileged_gate: float
    public_training: bool
    privileged_required: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
