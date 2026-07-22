"""H6 tensor-level loss composition with explicit masks and commit semantics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .contract import V3_HYBRID_LOSS_TERMS
from .model import V3_HYBRID_ROLES

LOSS_NAMES = tuple(V3_HYBRID_LOSS_TERMS)
SCHEDULE_CONSTANT = "constant"
SCHEDULE_LINEAR = "linear"


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class LossTermSchedule:
    """Multiplier schedule clocked by this term's eligible sample batches."""

    kind: str = SCHEDULE_CONSTANT
    start: float = 1.0
    end: float = 1.0
    updates: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {SCHEDULE_CONSTANT, SCHEDULE_LINEAR}:
            raise ValueError("loss schedule kind must be constant or linear")
        for name in ("start", "end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"loss schedule {name} must be finite and non-negative")
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("loss schedule updates must be a non-negative int")
        if self.kind == SCHEDULE_CONSTANT:
            if self.start != self.end or self.updates != 0:
                raise ValueError("constant loss schedule requires start=end and updates=0")
        elif self.updates < 1:
            raise ValueError("linear loss schedule requires updates >= 1")

    def value_at(self, eligible_step: int) -> float:
        if isinstance(eligible_step, bool) or not isinstance(eligible_step, int):
            raise TypeError("eligible_step must be an int")
        if eligible_step < 0:
            raise ValueError("eligible_step must be non-negative")
        if self.kind == SCHEDULE_CONSTANT or eligible_step >= self.updates:
            return float(self.end if self.kind == SCHEDULE_LINEAR else self.start)
        if self.updates == 1:
            return float(self.start)
        fraction = eligible_step / float(self.updates - 1)
        return float(self.start + fraction * (self.end - self.start))


def _default_schedules() -> Mapping[str, LossTermSchedule]:
    return MappingProxyType({name: LossTermSchedule() for name in LOSS_NAMES})


@dataclass(frozen=True)
class V3HybridLossComposerConfig:
    lambda_dmc: float = 0.0
    lambda_win: float = 0.0
    lambda_score: float = 0.0
    lambda_oracle: float = 0.0
    lambda_belief: float = 0.0
    lambda_coop: float = 0.0
    lambda_bc: float = 0.0
    lambda_strategy: float = 0.0
    lambda_bidding: float = 0.0
    landlord_weight: float = 1.0
    landlord_up_weight: float = 1.0
    landlord_down_weight: float = 1.0
    schedules: Mapping[str, LossTermSchedule] = field(default_factory=_default_schedules)

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        for name in V3_HYBRID_LOSS_TERMS.values():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("landlord_weight", "landlord_up_weight", "landlord_down_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if sum(self.role_weights.values()) <= 0.0:
            raise ValueError("at least one V3 loss role weight must be positive")
        if not isinstance(self.schedules, Mapping) or set(self.schedules) != set(LOSS_NAMES):
            raise ValueError("loss composer schedules must contain every canonical term")
        copied = {}
        for name in LOSS_NAMES:
            schedule = self.schedules[name]
            if not isinstance(schedule, LossTermSchedule):
                raise TypeError(f"loss schedule {name} has an invalid type")
            copied[name] = schedule
        object.__setattr__(self, "schedules", MappingProxyType(copied))

    @property
    def role_weights(self) -> dict[str, float]:
        return {
            "landlord": float(self.landlord_weight),
            "landlord_up": float(self.landlord_up_weight),
            "landlord_down": float(self.landlord_down_weight),
        }

    def weight(self, name: str) -> float:
        try:
            field_name = V3_HYBRID_LOSS_TERMS[name]
        except KeyError as exc:
            raise ValueError(f"unknown V3 loss term {name!r}") from exc
        return float(getattr(self, field_name))

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "weights": {name: self.weight(name) for name in LOSS_NAMES},
            "role_weights": self.role_weights,
            "schedules": {
                name: asdict(self.schedules[name]) for name in LOSS_NAMES
            },
            "normalization": "real_valid_samples_role_weight_once_v1",
            "counter_clock": "eligible_term_batch_commit_after_success_v1",
            "disabled_term": "no_input_no_loss_exact_noop_v1",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3HybridLossComposerConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("loss composer config must be a mapping")
        expected = {
            *V3_HYBRID_LOSS_TERMS.values(),
            "landlord_weight",
            "landlord_up_weight",
            "landlord_down_weight",
            "schedules",
        }
        if set(payload) != expected:
            raise ValueError("loss composer config fields mismatch")
        schedules = payload["schedules"]
        if not isinstance(schedules, Mapping) or set(schedules) != set(LOSS_NAMES):
            raise ValueError("loss composer schedule fields mismatch")
        parsed = {}
        for name, value in schedules.items():
            if not isinstance(value, Mapping) or set(value) != {"kind", "start", "end", "updates"}:
                raise ValueError(f"loss composer schedule {name} fields mismatch")
            parsed[name] = LossTermSchedule(**dict(value))
        values = dict(payload)
        values["schedules"] = parsed
        return cls(**values)


@dataclass(frozen=True)
class LossTermTensor:
    """Unreduced per-decision loss values and their authoritative validity."""

    values: torch.Tensor
    valid_mask: torch.Tensor
    roles: Sequence[str]
    sample_ids: Sequence[str]
    gradient_owner: str = "composer"
    schedule_override: float | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("loss term values must have shape (N,)")
        if self.valid_mask.shape != self.values.shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("loss valid_mask must be bool with shape (N,)")
        count = self.values.shape[0]
        if len(self.roles) != count or len(self.sample_ids) != count:
            raise ValueError("loss roles/sample_ids must align with values")
        if any(role not in V3_HYBRID_ROLES for role in self.roles):
            raise ValueError("loss term contains an unsupported role")
        valid_ids = [
            sample_id
            for sample_id, valid in zip(self.sample_ids, self.valid_mask.tolist())
            if valid
        ]
        if any(not isinstance(value, str) or not value for value in valid_ids):
            raise ValueError("valid loss samples require stable non-empty identities")
        if len(set(valid_ids)) != len(valid_ids):
            raise ValueError("loss term repeats a valid sample identity")
        if self.gradient_owner not in {"composer", "external"}:
            raise ValueError("loss gradient_owner must be composer or external")
        if self.schedule_override is not None and (
            isinstance(self.schedule_override, bool)
            or not isinstance(self.schedule_override, (int, float))
            or not math.isfinite(self.schedule_override)
            or self.schedule_override < 0.0
        ):
            raise ValueError("loss schedule_override must be finite and non-negative")


@dataclass(frozen=True)
class LossTermMetrics:
    raw_loss: float
    weighted_loss: float
    valid_samples: int
    valid_mask_count: int
    schedule_weight: float
    configured_weight: float
    eligible_step: int
    role_valid_samples: dict[str, int]
    role_effective_weights: dict[str, float]
    role_raw_loss: dict[str, float | None]
    phase: str


@dataclass(frozen=True)
class LossComposition:
    total: torch.Tensor | None
    terms: Mapping[str, LossTermMetrics]
    eligible_terms: tuple[str, ...]
    optimizer_step_required: bool


class V3HybridLossComposer:
    """Pure composition followed by an explicit post-success counter commit."""

    STATE_FORMAT = "v3-hybrid-h6-loss-composer-state-v1"

    def __init__(self, config: V3HybridLossComposerConfig) -> None:
        if not isinstance(config, V3HybridLossComposerConfig):
            raise TypeError("loss composer requires V3HybridLossComposerConfig")
        self.config = config
        self.eligible_steps = {name: 0 for name in LOSS_NAMES}

    def compose(self, terms: Mapping[str, LossTermTensor]) -> LossComposition:
        if not isinstance(terms, Mapping):
            raise TypeError("loss terms must be a mapping")
        unknown = set(terms) - set(LOSS_NAMES)
        if unknown:
            raise ValueError(f"unknown V3 loss inputs: {sorted(unknown)}")
        disabled_inputs = sorted(
            name for name in terms if self.config.weight(name) == 0.0
        )
        if disabled_inputs:
            raise ValueError(
                "disabled V3 loss terms reject input data: "
                f"{disabled_inputs}"
            )
        metrics: dict[str, LossTermMetrics] = {}
        weighted_tensors: list[torch.Tensor] = []
        eligible: list[str] = []
        for name in LOSS_NAMES:
            configured = self.config.weight(name)
            item = terms.get(name)
            if configured == 0.0:
                if item is not None:
                    raise ValueError(f"disabled V3 loss term {name!r} rejects input data")
                metrics[name] = self._disabled_metrics(name)
                continue
            if not isinstance(item, LossTermTensor):
                raise ValueError(f"enabled V3 loss term {name!r} requires tensor input")
            valid = item.valid_mask
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                metrics[name] = self._empty_metrics(name, configured)
                continue
            values = item.values[valid]
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError(f"V3 loss term {name!r} contains NaN or Inf")
            roles = [role for role, keep in zip(item.roles, valid.tolist()) if keep]
            role_weights = values.new_tensor([
                self.config.role_weights[role] for role in roles
            ])
            denominator = role_weights.sum()
            if not bool(denominator > 0):
                metrics[name] = self._empty_metrics(name, configured)
                continue
            raw = (values * role_weights).sum() / denominator
            multiplier = (
                item.schedule_override
                if item.schedule_override is not None
                else self.config.schedules[name].value_at(self.eligible_steps[name])
            )
            effective_weight = configured * multiplier
            weighted = raw * effective_weight
            if not bool(torch.isfinite(weighted)):
                raise FloatingPointError(f"weighted V3 loss term {name!r} is non-finite")
            eligible.append(name)
            if effective_weight > 0.0 and item.gradient_owner == "composer":
                weighted_tensors.append(weighted)
            role_counts = {
                role: sum(value == role for value in roles) for role in V3_HYBRID_ROLES
            }
            role_effective = {
                role: role_counts[role] * self.config.role_weights[role]
                for role in V3_HYBRID_ROLES
            }
            role_raw: dict[str, float | None] = {}
            for role in V3_HYBRID_ROLES:
                indices = [index for index, value in enumerate(roles) if value == role]
                role_raw[role] = (
                    None
                    if not indices
                    else float(values[indices].mean().detach().cpu().item())
                )
            metrics[name] = LossTermMetrics(
                raw_loss=float(raw.detach().cpu().item()),
                weighted_loss=float(weighted.detach().cpu().item()),
                valid_samples=valid_count,
                valid_mask_count=valid_count,
                schedule_weight=float(multiplier),
                configured_weight=configured,
                eligible_step=self.eligible_steps[name],
                role_valid_samples=role_counts,
                role_effective_weights=role_effective,
                role_raw_loss=role_raw,
                phase=(
                    "external_applied"
                    if item.gradient_owner == "external" and effective_weight > 0.0
                    else "active"
                    if effective_weight > 0.0
                    else "scheduled_zero"
                ),
            )
        total = torch.stack(weighted_tensors).sum() if weighted_tensors else None
        return LossComposition(
            total=total,
            terms=MappingProxyType(metrics),
            eligible_terms=tuple(eligible),
            optimizer_step_required=total is not None,
        )

    def _disabled_metrics(self, name: str) -> LossTermMetrics:
        return LossTermMetrics(
            0.0, 0.0, 0, 0, 0.0, 0.0, self.eligible_steps[name],
            {role: 0 for role in V3_HYBRID_ROLES},
            {role: 0.0 for role in V3_HYBRID_ROLES},
            {role: None for role in V3_HYBRID_ROLES},
            "disabled",
        )

    def _empty_metrics(self, name: str, configured: float) -> LossTermMetrics:
        return LossTermMetrics(
            0.0, 0.0, 0, 0,
            self.config.schedules[name].value_at(self.eligible_steps[name]),
            configured,
            self.eligible_steps[name],
            {role: 0 for role in V3_HYBRID_ROLES},
            {role: 0.0 for role in V3_HYBRID_ROLES},
            {role: None for role in V3_HYBRID_ROLES},
            "no_valid_targets",
        )

    def commit(self, composition: LossComposition) -> None:
        if not isinstance(composition, LossComposition):
            raise TypeError("loss composer commit requires LossComposition")
        for name in composition.eligible_terms:
            expected = composition.terms[name].eligible_step
            if expected != self.eligible_steps[name]:
                raise RuntimeError("stale V3 loss composition cannot be committed")
        for name in composition.eligible_terms:
            self.eligible_steps[name] += 1

    def apply(
        self,
        composition: LossComposition,
        optimizer: torch.optim.Optimizer,
        parameters: Sequence[torch.nn.Parameter],
        *,
        max_grad_norm: float,
    ) -> float:
        """Apply one finite optimizer step; counters advance only after success."""

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("loss composer apply requires a Torch optimizer")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive and finite")
        owned = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        if not composition.optimizer_step_required:
            self.commit(composition)
            return 0.0
        assert composition.total is not None
        try:
            composition.total.backward()
            gradients = [parameter.grad for parameter in owned if parameter.grad is not None]
            if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
                raise FloatingPointError("V3 loss composer gradient is missing or non-finite")
            norm = torch.nn.utils.clip_grad_norm_(
                owned, max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        self.commit(composition)
        return float(norm.detach().cpu().item())

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.STATE_FORMAT,
            "config_hash": self.config.stable_hash(),
            "eligible_steps": dict(self.eligible_steps),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "format", "config_hash", "eligible_steps"
        }:
            raise ValueError("loss composer state fields mismatch")
        if payload["format"] != self.STATE_FORMAT:
            raise ValueError("loss composer state format mismatch")
        if payload["config_hash"] != self.config.stable_hash():
            raise ValueError("loss composer config identity mismatch")
        counters = payload["eligible_steps"]
        if not isinstance(counters, Mapping) or set(counters) != set(LOSS_NAMES):
            raise ValueError("loss composer counter fields mismatch")
        parsed = {}
        for name, value in counters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid loss composer counter {name}")
            parsed[name] = value
        self.eligible_steps = parsed


__all__ = [
    "LOSS_NAMES",
    "LossComposition",
    "LossTermMetrics",
    "LossTermSchedule",
    "LossTermTensor",
    "SCHEDULE_CONSTANT",
    "SCHEDULE_LINEAR",
    "V3HybridLossComposer",
    "V3HybridLossComposerConfig",
]
