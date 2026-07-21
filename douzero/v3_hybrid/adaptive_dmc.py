"""Adaptive DMC tensor formulas for the H2 public-policy learner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from .config import DMC_TARGET_RAW, DMC_TARGET_SIGNED_LOG

ADMC_DISABLED = "disabled"
ADMC_PAPER_RATIO = "paper_ratio"
ADMC_SAFE_HYBRID = "safe_hybrid"

_ADMC_MODES = frozenset({ADMC_DISABLED, ADMC_PAPER_RATIO, ADMC_SAFE_HYBRID})


@dataclass(frozen=True)
class AdaptiveDMCConfig:
    """Identity-bound H2 clipping configuration.

    ``gamma`` follows a learner-update linear schedule. ``epsilon`` and
    ``delta`` are used only by ``safe_hybrid`` but remain identity fields so a
    checkpoint can never be relabelled across ablations.
    """

    mode: str = ADMC_DISABLED
    gamma_start: float = 0.20
    gamma_end: float = 0.05
    gamma_schedule_updates: int = 100_000
    epsilon: float = 1e-3
    delta: float = 0.10

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if self.mode not in _ADMC_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_ADMC_MODES)}, got {self.mode!r}"
            )
        for name in ("gamma_start", "gamma_end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            isinstance(self.gamma_schedule_updates, bool)
            or not isinstance(self.gamma_schedule_updates, int)
            or self.gamma_schedule_updates < 0
        ):
            raise ValueError("gamma_schedule_updates must be a non-negative int")
        for name in ("epsilon", "delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")

    @property
    def enabled(self) -> bool:
        return self.mode != ADMC_DISABLED

    def gamma_at(self, learner_update: int) -> float:
        if isinstance(learner_update, bool) or not isinstance(learner_update, int):
            raise TypeError("learner_update must be an int")
        if learner_update < 0:
            raise ValueError("learner_update must be non-negative")
        if self.gamma_schedule_updates == 0:
            return float(self.gamma_end)
        progress = min(1.0, learner_update / self.gamma_schedule_updates)
        return float(
            self.gamma_start + progress * (self.gamma_end - self.gamma_start)
        )

    def compatibility_dict(self) -> dict[str, object]:
        return {"identity_version": self.IDENTITY_VERSION, **asdict(self)}

    def stable_hash(self) -> str:
        encoded = json.dumps(
            self.compatibility_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AdaptiveDMCConfig":
        if not isinstance(payload, dict):
            raise TypeError("Adaptive DMC config must be a dict")
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing or unknown:
            raise ValueError(
                "Adaptive DMC config fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**payload)


@dataclass(frozen=True)
class AdaptiveDMCResult:
    """Per-sample H2 values before valid-sample and role normalization."""

    loss_per_sample: torch.Tensor
    constrained_q: torch.Tensor
    target: torch.Tensor
    ratio: torch.Tensor
    ratio_clipped: torch.Tensor
    near_zero_fallback: torch.Tensor
    target_clamped: torch.Tensor
    non_finite_fallback: torch.Tensor
    gamma: float

    def __post_init__(self) -> None:
        shape = self.loss_per_sample.shape
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError("Adaptive DMC result must contain a non-empty vector")
        for name in ("constrained_q", "target", "ratio"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape must match loss_per_sample")
        for name in (
            "ratio_clipped",
            "near_zero_fallback",
            "target_clamped",
            "non_finite_fallback",
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be bool with the loss shape")


def _vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1 or value.numel() < 1:
        raise ValueError(f"{name} must have shape (B,) or (B, 1)")
    if not value.is_floating_point():
        value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def transform_dmc_target(
    mc_return: torch.Tensor,
    *,
    transform: str,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform raw team-perspective MC returns and report clamp incidence."""

    value = _vector("mc_return", mc_return)
    if transform == DMC_TARGET_RAW:
        transformed = value
    elif transform == DMC_TARGET_SIGNED_LOG:
        transformed = torch.sign(value) * torch.log1p(value.abs())
    else:
        raise ValueError(f"unsupported DMC target transform {transform!r}")
    if (
        isinstance(clamp, bool)
        or not isinstance(clamp, (int, float))
        or not math.isfinite(clamp)
        or clamp <= 0.0
    ):
        raise ValueError("DMC target clamp must be positive and finite")
    clamped = transformed.abs() > float(clamp)
    return transformed.clamp(-float(clamp), float(clamp)), clamped


def adaptive_dmc_loss(
    q_new: torch.Tensor,
    mc_return: torch.Tensor,
    *,
    config: AdaptiveDMCConfig,
    target_transform: str,
    target_clamp: float,
    learner_update: int,
    q_old: torch.Tensor | None = None,
) -> AdaptiveDMCResult:
    """Return ordinary, paper-ratio, or safe-hybrid per-sample DMC MSE.

    The function performs no reduction. The learner is therefore responsible
    for gathering real selected actions and applying role weights exactly once.
    """

    if not isinstance(config, AdaptiveDMCConfig):
        raise TypeError("config must be AdaptiveDMCConfig")
    current = _vector("q_new", q_new)
    target, target_clamped = transform_dmc_target(
        mc_return, transform=target_transform, clamp=target_clamp
    )
    if target.shape != current.shape:
        raise ValueError("q_new and mc_return batch shapes must match")

    gamma = config.gamma_at(learner_update)
    zeros = torch.zeros_like(current, dtype=torch.bool)
    ratio = torch.ones_like(current)
    ratio_clipped = zeros
    near_zero = zeros
    non_finite = zeros

    if config.mode == ADMC_DISABLED:
        constrained = current
    else:
        if q_old is None:
            raise ValueError(f"{config.mode} requires actor-snapshot q_old")
        previous = _vector("q_old", q_old).to(
            device=current.device, dtype=current.dtype
        )
        if previous.shape != current.shape:
            raise ValueError("q_old and q_new batch shapes must match")
        lower = 1.0 - gamma
        upper = 1.0 + gamma

        if config.mode == ADMC_PAPER_RATIO:
            ratio = current / previous
            clipped_ratio = ratio.clamp(lower, upper)
            candidate = clipped_ratio * previous
            non_finite = ~torch.isfinite(candidate)
            constrained = torch.where(non_finite, current, candidate)
            ratio_clipped = torch.isfinite(ratio) & (ratio != clipped_ratio)
        else:
            near_zero = previous.abs() < config.epsilon
            stable = ~near_zero
            denominator = torch.where(stable, previous, torch.ones_like(previous))
            ratio = torch.where(
                stable, current / denominator, torch.ones_like(current)
            )
            clipped_ratio = ratio.clamp(lower, upper)
            ratio_candidate = clipped_ratio * previous
            delta_candidate = previous + (current - previous).clamp(
                -config.delta, config.delta
            )
            candidate = torch.where(near_zero, delta_candidate, ratio_candidate)
            non_finite = ~torch.isfinite(candidate)
            constrained = torch.where(non_finite, current, candidate)
            ratio_clipped = stable & (ratio != clipped_ratio)

        constrained = constrained.clamp(-float(target_clamp), float(target_clamp))

    loss = (constrained - target) ** 2
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Adaptive DMC loss contains non-finite values")
    return AdaptiveDMCResult(
        loss_per_sample=loss,
        constrained_q=constrained,
        target=target,
        ratio=ratio,
        ratio_clipped=ratio_clipped,
        near_zero_fallback=near_zero,
        target_clamped=target_clamped,
        non_finite_fallback=non_finite,
        gamma=gamma,
    )
