"""H2-only Adaptive DMC learner, metrics, and strict resume checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from douzero._version import git_sha
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import model_input_bundles_to_batch

from .adaptive_dmc import (
    ADMC_DISABLED,
    ADMC_PAPER_RATIO,
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    AdaptiveDMCResult,
    adaptive_dmc_loss,
)
from .checkpoint import h1_compatibility_identity
from .config import V3HybridModelConfig
from .model import V3_HYBRID_ROLES, V3HybridModel
from .replay import (
    V3_H2_REPLAY_SCHEMA_VERSION,
    V3_H2_REPLAY_SEMANTICS,
    V3ReplayTransition,
)

V3_H2_TRAINER_CHECKPOINT_FORMAT = "v3-hybrid-h2-trainer-v1"
V3_H2_TRAINING_CONTRACT = "oadmcdou-ratio-safe-hybrid-v1"

_FORBIDDEN_STATE_NAMES = (
    "privileged", "teacher", "oracle", "all_handcards", "hidden_hand"
)
_CHECKPOINT_KEYS = frozenset({
    "format",
    "source_git_sha",
    "model_state_dict",
    "optimizer_state_dict",
    "feature_schema_hash",
    "model_config",
    "model_config_hash",
    "ruleset_identity",
    "learner_config",
    "learner_config_hash",
    "training_identity",
    "training_identity_hash",
    "policy_version",
    "counters",
    "schedule_state",
    "adaptive_statistics",
    "rng",
    "replay_resume_policy",
})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _positive_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class V3H2LearnerConfig:
    """Identity-bound standalone H2 learner configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    rmsprop_alpha: float = 0.99
    rmsprop_momentum: float = 0.0
    rmsprop_epsilon: float = 1e-5
    max_grad_norm: float = 40.0
    lambda_dmc: float = 1.0
    landlord_weight: float = 1.0
    landlord_up_weight: float = 1.0
    landlord_down_weight: float = 1.0
    device: str = "cpu"
    seed: int = 0
    initial_policy_version: int = 0
    adaptive_dmc: AdaptiveDMCConfig = field(default_factory=AdaptiveDMCConfig)

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        for name in ("batch_size", "seed", "initial_policy_version"):
            value = getattr(self, name)
            minimum = 1 if name == "batch_size" else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an int >= {minimum}")
        _positive_finite("learning_rate", self.learning_rate)
        _positive_finite("rmsprop_epsilon", self.rmsprop_epsilon)
        _positive_finite("max_grad_norm", self.max_grad_norm)
        if (
            not math.isfinite(self.rmsprop_alpha)
            or not 0.0 <= self.rmsprop_alpha < 1.0
        ):
            raise ValueError("rmsprop_alpha must be finite and in [0, 1)")
        if not math.isfinite(self.rmsprop_momentum) or self.rmsprop_momentum < 0.0:
            raise ValueError("rmsprop_momentum must be finite and non-negative")
        for name in (
            "lambda_dmc", "landlord_weight", "landlord_up_weight",
            "landlord_down_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if sum(self.role_weights.values()) <= 0.0:
            raise ValueError("at least one role weight must be positive")
        if self.device != "cpu" and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if not isinstance(self.adaptive_dmc, AdaptiveDMCConfig):
            raise TypeError("adaptive_dmc must be AdaptiveDMCConfig")

    @property
    def role_weights(self) -> dict[str, float]:
        return {
            "landlord": float(self.landlord_weight),
            "landlord_up": float(self.landlord_up_weight),
            "landlord_down": float(self.landlord_down_weight),
        }

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            **{
                key: value
                for key, value in asdict(self).items()
                if key != "adaptive_dmc"
            },
            "adaptive_dmc": self.adaptive_dmc.compatibility_dict(),
            "optimizer": "rmsprop_v1",
            "loss_reduction": "selected_real_samples_role_weight_once_v1",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H2LearnerConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("H2 learner config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing or unknown:
            raise ValueError(
                "H2 learner config fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        values = dict(payload)
        values["adaptive_dmc"] = AdaptiveDMCConfig.from_dict(
            dict(values["adaptive_dmc"])
        )
        return cls(**values)


@dataclass(frozen=True)
class V3H2StepMetrics:
    learner_update: int
    policy_version: int
    mode: str
    gamma: float
    loss_dmc: float
    loss_total: float
    gradient_norm: float
    samples: int
    q_new_mean: float
    q_new_std: float
    q_new_min: float
    q_new_max: float
    q_old_mean: float | None
    q_old_std: float | None
    q_old_min: float | None
    q_old_max: float | None
    q_drift_mean_abs: float | None
    ratio_mean: float | None
    ratio_min: float | None
    ratio_max: float | None
    ratio_clip_fraction: float
    near_zero_fallback_fraction: float
    target_clamp_fraction: float
    non_finite_fallback_fraction: float
    max_policy_lag: int
    role_samples: dict[str, int]
    role_effective_weights: dict[str, float]
    role_losses: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class AdaptiveDMCCumulativeStats:
    """Checkpointable finite running sums for H2 diagnostics."""

    _SCALAR_FIELDS = (
        "steps", "samples", "q_new_sum", "q_new_square_sum", "q_old_sum",
        "q_old_square_sum", "q_old_samples", "q_drift_sum", "ratio_sum",
        "ratio_samples", "ratio_clip_count", "near_zero_count",
        "target_clamp_count", "non_finite_fallback_count", "gradient_norm_sum",
    )

    def __init__(self) -> None:
        self.steps = 0
        self.samples = 0
        self.q_new_sum = 0.0
        self.q_new_square_sum = 0.0
        self.q_old_sum = 0.0
        self.q_old_square_sum = 0.0
        self.q_old_samples = 0
        self.q_drift_sum = 0.0
        self.ratio_sum = 0.0
        self.ratio_samples = 0
        self.ratio_clip_count = 0
        self.near_zero_count = 0
        self.target_clamp_count = 0
        self.non_finite_fallback_count = 0
        self.gradient_norm_sum = 0.0
        self.role_samples = {role: 0 for role in V3_HYBRID_ROLES}
        self.role_loss_sum = {role: 0.0 for role in V3_HYBRID_ROLES}
        self.role_effective_weight_sum = {role: 0.0 for role in V3_HYBRID_ROLES}

    def update(
        self,
        metrics: V3H2StepMetrics,
        *,
        q_new: torch.Tensor,
        q_old: torch.Tensor | None,
        result: AdaptiveDMCResult,
    ) -> None:
        current = q_new.detach().double().reshape(-1).cpu()
        self.steps += 1
        self.samples += metrics.samples
        self.q_new_sum += float(current.sum().item())
        self.q_new_square_sum += float((current * current).sum().item())
        if q_old is not None:
            previous = q_old.detach().double().reshape(-1).cpu()
            self.q_old_samples += int(previous.numel())
            self.q_old_sum += float(previous.sum().item())
            self.q_old_square_sum += float((previous * previous).sum().item())
            self.q_drift_sum += float((current - previous).abs().sum().item())
            finite_ratio = result.ratio.detach().double().reshape(-1).cpu()
            finite_ratio = finite_ratio[torch.isfinite(finite_ratio)]
            self.ratio_samples += int(finite_ratio.numel())
            self.ratio_sum += float(finite_ratio.sum().item())
        self.ratio_clip_count += int(result.ratio_clipped.sum().item())
        self.near_zero_count += int(result.near_zero_fallback.sum().item())
        self.target_clamp_count += int(result.target_clamped.sum().item())
        self.non_finite_fallback_count += int(
            result.non_finite_fallback.sum().item()
        )
        self.gradient_norm_sum += metrics.gradient_norm
        for role in V3_HYBRID_ROLES:
            self.role_samples[role] += metrics.role_samples[role]
            self.role_loss_sum[role] += (
                metrics.role_losses[role] * metrics.role_samples[role]
            )
            self.role_effective_weight_sum[role] += (
                metrics.role_effective_weights[role]
            )

    def state_dict(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in self._SCALAR_FIELDS},
            "role_samples": dict(self.role_samples),
            "role_loss_sum": dict(self.role_loss_sum),
            "role_effective_weight_sum": dict(self.role_effective_weight_sum),
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, object]
    ) -> "AdaptiveDMCCumulativeStats":
        expected = {
            *cls._SCALAR_FIELDS,
            "role_samples", "role_loss_sum", "role_effective_weight_sum",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("Adaptive DMC statistics fields mismatch")
        stats = cls()
        integer_fields = {
            "steps", "samples", "q_old_samples", "ratio_samples",
            "ratio_clip_count", "near_zero_count", "target_clamp_count",
            "non_finite_fallback_count",
        }
        for name in cls._SCALAR_FIELDS:
            value = payload[name]
            if name in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"Adaptive DMC statistic {name} is invalid")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Adaptive DMC statistic {name} is invalid")
            setattr(stats, name, value)
        for name in ("role_samples", "role_loss_sum", "role_effective_weight_sum"):
            values = payload[name]
            if not isinstance(values, Mapping) or set(values) != set(V3_HYBRID_ROLES):
                raise ValueError(f"Adaptive DMC statistic {name} roles mismatch")
            converted = {}
            for role, value in values.items():
                if name == "role_samples":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("Adaptive DMC role sample count is invalid")
                    converted[role] = value
                else:
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Adaptive DMC role statistic is invalid")
                    converted[role] = float(value)
            setattr(stats, name, converted)
        return stats

    def validate_invariants(
        self,
        *,
        mode: str,
        learner_updates: int,
        samples_consumed: int,
        batch_size: int,
        role_weights: Mapping[str, float],
    ) -> None:
        """Reject internally inconsistent diagnostic state before resume."""

        if mode not in {ADMC_DISABLED, ADMC_PAPER_RATIO, ADMC_SAFE_HYBRID}:
            raise ValueError("Adaptive DMC statistics mode is unsupported")
        if self.steps != learner_updates or self.samples != samples_consumed:
            raise ValueError("Adaptive DMC statistics/counter drift")
        if self.steps == 0:
            if self.samples != 0:
                raise ValueError("Adaptive DMC zero-step statistics contain samples")
        elif not self.steps <= self.samples <= self.steps * batch_size:
            raise ValueError("Adaptive DMC statistics batch cardinality is impossible")
        if sum(self.role_samples.values()) != self.samples:
            raise ValueError("Adaptive DMC role samples do not sum to total samples")

        nonnegative_values = {
            "q_new_square_sum": self.q_new_square_sum,
            "q_old_square_sum": self.q_old_square_sum,
            "q_drift_sum": self.q_drift_sum,
            "gradient_norm_sum": self.gradient_norm_sum,
        }
        if any(value < 0.0 for value in nonnegative_values.values()):
            raise ValueError("Adaptive DMC cumulative magnitude is negative")
        for role in V3_HYBRID_ROLES:
            if self.role_loss_sum[role] < 0.0:
                raise ValueError("Adaptive DMC cumulative role loss is negative")
            expected_weight = self.role_samples[role] * float(role_weights[role])
            if not math.isclose(
                self.role_effective_weight_sum[role],
                expected_weight,
                rel_tol=1e-6,
                abs_tol=1e-7 * max(1.0, abs(expected_weight)),
            ):
                raise ValueError("Adaptive DMC cumulative role weight is inconsistent")
            if self.role_samples[role] == 0 and self.role_loss_sum[role] != 0.0:
                raise ValueError("Adaptive DMC empty role has a non-zero loss sum")

        event_counts = {
            "ratio_clip_count": self.ratio_clip_count,
            "near_zero_count": self.near_zero_count,
            "target_clamp_count": self.target_clamp_count,
            "non_finite_fallback_count": self.non_finite_fallback_count,
        }
        if any(count > self.samples for count in event_counts.values()):
            raise ValueError("Adaptive DMC event count exceeds total samples")
        if self.ratio_samples > self.q_old_samples:
            raise ValueError("Adaptive DMC ratio samples exceed q_old samples")
        if self.ratio_clip_count > self.ratio_samples:
            raise ValueError("Adaptive DMC clipped ratios exceed finite ratios")

        if mode == ADMC_DISABLED:
            disabled_values = (
                self.q_old_sum,
                self.q_old_square_sum,
                self.q_old_samples,
                self.q_drift_sum,
                self.ratio_sum,
                self.ratio_samples,
                self.ratio_clip_count,
                self.near_zero_count,
                self.non_finite_fallback_count,
            )
            if any(value != 0 for value in disabled_values):
                raise ValueError("ordinary DMC statistics contain adaptive-only state")
        else:
            if self.q_old_samples != self.samples:
                raise ValueError("Adaptive DMC q_old sample count is incomplete")
            if mode == ADMC_PAPER_RATIO and self.near_zero_count != 0:
                raise ValueError("paper-ratio statistics contain safe-hybrid fallbacks")
            if (
                mode == ADMC_SAFE_HYBRID
                and self.ratio_clip_count > self.samples - self.near_zero_count
            ):
                raise ValueError("safe-hybrid ratio and near-zero counts overlap")


def h2_training_identity(
    model: V3HybridModel,
    ruleset: RuleSet,
    config: V3H2LearnerConfig,
) -> dict[str, object]:
    """Bind the H1 public graph to the complete H2 training contract."""

    public_identity = h1_compatibility_identity(model.config, ruleset)
    return {
        "identity_version": 1,
        "training_contract": V3_H2_TRAINING_CONTRACT,
        "public_policy_compatibility_hash": public_identity.stable_hash(),
        "public_policy_compatibility_identity": public_identity.compatibility_dict(),
        "feature_flags": {
            "adaptive_dmc": config.adaptive_dmc.enabled,
            "oracle": False,
            "belief": False,
            "cooperation": False,
            "bidding": False,
        },
        "replay": {
            "schema_version": V3_H2_REPLAY_SCHEMA_VERSION,
            "semantics": V3_H2_REPLAY_SEMANTICS,
            "adaptive_q_old_required": config.adaptive_dmc.enabled,
        },
        "loss": {
            "lambda_dmc": float(config.lambda_dmc),
            "adaptive_dmc": config.adaptive_dmc.compatibility_dict(),
            "target_transform": model.config.dmc_target_transform,
            "target_clamp": float(model.config.dmc_target_clamp),
            "normalization": "selected_real_samples_role_weight_once_v1",
            "win_score": "unchanged_not_integrated_h2",
        },
        "optimizer": {
            "kind": "rmsprop",
            "learning_rate": float(config.learning_rate),
            "alpha": float(config.rmsprop_alpha),
            "momentum": float(config.rmsprop_momentum),
            "epsilon": float(config.rmsprop_epsilon),
            "max_grad_norm": float(config.max_grad_norm),
        },
        "role_weights": config.role_weights,
        "trainer": config.compatibility_dict(),
        "topology": "standalone_selected_batch_h2_no_actors_v1",
        "replay_resume_policy": "flushed_checkpoint_boundary_v1",
    }


class V3H2Learner:
    """A minimal V3-only selected-action learner for the H2 algorithm."""

    def __init__(
        self,
        model: V3HybridModel,
        *,
        ruleset: RuleSet,
        config: V3H2LearnerConfig | None = None,
    ) -> None:
        if not isinstance(model, V3HybridModel):
            raise TypeError("H2 learner requires a V3HybridModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("H2 learner requires a RuleSet")
        self.config = config or V3H2LearnerConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA H2 learner requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.model.train()
        self.ruleset = ruleset
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha,
            momentum=self.config.rmsprop_momentum,
            eps=self.config.rmsprop_epsilon,
        )
        self.rng = random.Random(self.config.seed)
        self.learner_updates = 0
        self.samples_consumed = 0
        self.policy_version = self.config.initial_policy_version
        self.statistics = AdaptiveDMCCumulativeStats()
        self.compatibility_identity = h2_training_identity(
            self.model, self.ruleset, self.config
        )
        self.compatibility_hash = _canonical_hash(self.compatibility_identity)

    def _empty_metrics(self) -> V3H2StepMetrics:
        zeros_i = {role: 0 for role in V3_HYBRID_ROLES}
        zeros_f = {role: 0.0 for role in V3_HYBRID_ROLES}
        return V3H2StepMetrics(
            learner_update=self.learner_updates,
            policy_version=self.policy_version,
            mode=self.config.adaptive_dmc.mode,
            gamma=self.config.adaptive_dmc.gamma_at(self.learner_updates),
            loss_dmc=0.0,
            loss_total=0.0,
            gradient_norm=0.0,
            samples=0,
            q_new_mean=0.0,
            q_new_std=0.0,
            q_new_min=0.0,
            q_new_max=0.0,
            q_old_mean=None,
            q_old_std=None,
            q_old_min=None,
            q_old_max=None,
            q_drift_mean_abs=None,
            ratio_mean=None,
            ratio_min=None,
            ratio_max=None,
            ratio_clip_fraction=0.0,
            near_zero_fallback_fraction=0.0,
            target_clamp_fraction=0.0,
            non_finite_fallback_fraction=0.0,
            max_policy_lag=0,
            role_samples=zeros_i,
            role_effective_weights=zeros_f,
            role_losses=dict(zeros_f),
        )

    def train_batch(
        self, transitions: Sequence[V3ReplayTransition] | None
    ) -> V3H2StepMetrics:
        """Run one selected-action update, or an exact no-op at lambda zero."""

        if self.config.lambda_dmc == 0.0:
            return self._empty_metrics()
        if transitions is None or not transitions:
            raise ValueError("enabled DMC training requires a non-empty batch")
        if len(transitions) > self.config.batch_size:
            raise ValueError("H2 batch exceeds the configured batch_size")
        adaptive_required = self.config.adaptive_dmc.mode != ADMC_DISABLED
        max_policy_lag = 0
        for transition in transitions:
            if not isinstance(transition, V3ReplayTransition):
                raise TypeError("H2 learner accepts only V3ReplayTransition")
            transition.validate(
                expected_schema_hash=self.model.schema.stable_hash(),
                expected_target_transform=self.model.config.dmc_target_transform,
                expected_ruleset_identity=self.ruleset.identity(),
                adaptive_required=adaptive_required,
            )
            if adaptive_required:
                version = transition.adaptive_provenance.policy_version
                if version > self.policy_version:
                    raise ValueError("replay policy version is newer than the learner")
                max_policy_lag = max(max_policy_lag, self.policy_version - version)

        chosen = [transition.selected_action_index for transition in transitions]
        inputs = model_input_bundles_to_batch(
            [transition.model_inputs for transition in transitions], chosen
        )
        output = self.model.forward_input_batch(inputs)
        gathered = output.gather_chosen(inputs.chosen_action_index)
        q_new = gathered["dmc_q"].squeeze(-1)
        returns = q_new.new_tensor([transition.mc_return for transition in transitions])
        q_old = None
        if adaptive_required:
            q_old = q_new.new_tensor([
                transition.adaptive_provenance.q_old
                for transition in transitions
            ])

        result = adaptive_dmc_loss(
            q_new,
            returns,
            config=self.config.adaptive_dmc,
            target_transform=self.model.config.dmc_target_transform,
            target_clamp=self.model.config.dmc_target_clamp,
            learner_update=self.learner_updates,
            q_old=q_old,
        )
        sample_weights = q_new.new_tensor([
            self.config.role_weights[transition.role]
            for transition in transitions
        ])
        denominator = sample_weights.sum()
        if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
            raise ValueError("H2 batch has zero effective role weight")
        dmc_loss = (result.loss_per_sample * sample_weights).sum() / denominator
        total = dmc_loss * self.config.lambda_dmc
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("H2 total loss is non-finite")

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        gradient_norm_value = float(gradient_norm.detach().float().item())
        self.optimizer.step()

        batch_size = len(transitions)
        role_samples: dict[str, int] = {}
        role_weights: dict[str, float] = {}
        role_losses: dict[str, float] = {}
        for role in V3_HYBRID_ROLES:
            mask = q_new.new_tensor(
                [transition.role == role for transition in transitions],
                dtype=torch.bool,
            )
            count = int(mask.sum().item())
            role_samples[role] = count
            role_weights[role] = float(sample_weights[mask].sum().item())
            role_losses[role] = (
                float(result.loss_per_sample[mask].mean().detach().item())
                if count
                else 0.0
            )

        def distribution(value: torch.Tensor) -> tuple[float, float, float, float]:
            detached = value.detach().float()
            return (
                float(detached.mean().item()),
                float(detached.std(unbiased=False).item()),
                float(detached.min().item()),
                float(detached.max().item()),
            )

        q_new_dist = distribution(q_new)
        q_old_dist = distribution(q_old) if q_old is not None else None
        finite_ratio = result.ratio.detach()[torch.isfinite(result.ratio.detach())]
        ratio_dist = distribution(finite_ratio) if finite_ratio.numel() else None
        self.learner_updates += 1
        self.samples_consumed += batch_size
        self.policy_version += 1
        metrics = V3H2StepMetrics(
            learner_update=self.learner_updates,
            policy_version=self.policy_version,
            mode=self.config.adaptive_dmc.mode,
            gamma=result.gamma,
            loss_dmc=float(dmc_loss.detach().float().item()),
            loss_total=float(total.detach().float().item()),
            gradient_norm=gradient_norm_value,
            samples=batch_size,
            q_new_mean=q_new_dist[0],
            q_new_std=q_new_dist[1],
            q_new_min=q_new_dist[2],
            q_new_max=q_new_dist[3],
            q_old_mean=None if q_old_dist is None else q_old_dist[0],
            q_old_std=None if q_old_dist is None else q_old_dist[1],
            q_old_min=None if q_old_dist is None else q_old_dist[2],
            q_old_max=None if q_old_dist is None else q_old_dist[3],
            q_drift_mean_abs=(
                None
                if q_old is None
                else float((q_new.detach() - q_old).abs().mean().item())
            ),
            ratio_mean=None if ratio_dist is None else ratio_dist[0],
            ratio_min=None if ratio_dist is None else ratio_dist[2],
            ratio_max=None if ratio_dist is None else ratio_dist[3],
            ratio_clip_fraction=float(result.ratio_clipped.float().mean().item()),
            near_zero_fallback_fraction=float(
                result.near_zero_fallback.float().mean().item()
            ),
            target_clamp_fraction=float(result.target_clamped.float().mean().item()),
            non_finite_fallback_fraction=float(
                result.non_finite_fallback.float().mean().item()
            ),
            max_policy_lag=max_policy_lag,
            role_samples=role_samples,
            role_effective_weights=role_weights,
            role_losses=role_losses,
        )
        self.statistics.update(metrics, q_new=q_new, q_old=q_old, result=result)
        return metrics

    @staticmethod
    def _encode_numpy_rng_state(state) -> dict[str, object]:
        return {
            "algorithm": state[0],
            "keys": state[1].tolist(),
            "position": int(state[2]),
            "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }

    @staticmethod
    def _decode_numpy_rng_state(payload: Mapping[str, object]):
        expected = {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("NumPy RNG fields mismatch")
        return (
            str(payload["algorithm"]),
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save strict model/optimizer/schedule/stats/RNG state."""

        source_sha = git_sha()
        if (
            len(source_sha) != 40
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise RuntimeError("H2 trainer checkpoints require a full source Git SHA")
        state_dict = self.model.state_dict()
        forbidden = sorted(
            name for name in state_dict
            if any(token in name.lower() for token in _FORBIDDEN_STATE_NAMES)
        )
        if forbidden:
            raise RuntimeError(f"H2 public model contains forbidden state keys: {forbidden}")
        bundle = {
            "format": V3_H2_TRAINER_CHECKPOINT_FORMAT,
            "source_git_sha": source_sha,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_schema_hash": self.model.schema.stable_hash(),
            "model_config": asdict(self.model.config),
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "policy_version": self.policy_version,
            "counters": {
                "learner_updates": self.learner_updates,
                "samples_consumed": self.samples_consumed,
            },
            "schedule_state": {
                "learner_update": self.learner_updates,
                "gamma": self.config.adaptive_dmc.gamma_at(self.learner_updates),
            },
            "adaptive_statistics": self.statistics.state_dict(),
            "rng": {
                "learner": self.rng.getstate(),
                "python": random.getstate(),
                "numpy": self._encode_numpy_rng_state(np.random.get_state()),
                "torch": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
            },
            "replay_resume_policy": "flushed_checkpoint_boundary_v1",
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """Strictly resume H2 state into an identically configured learner."""

        try:
            # Keep CPU RNG tensors on CPU even when resuming a CUDA learner.
            # Model and optimizer loaders copy their tensors to parameter devices.
            bundle = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"unable to safely load H2 trainer checkpoint: {exc}"
            ) from exc
        if not isinstance(bundle, dict) or set(bundle) != _CHECKPOINT_KEYS:
            actual = set(bundle) if isinstance(bundle, dict) else set()
            raise CheckpointCompatibilityError(
                "H2 checkpoint envelope mismatch: "
                f"missing={sorted(_CHECKPOINT_KEYS - actual)}, "
                f"extra={sorted(actual - _CHECKPOINT_KEYS)}"
            )
        expected_scalars = {
            "format": V3_H2_TRAINER_CHECKPOINT_FORMAT,
            "feature_schema_hash": self.model.schema.stable_hash(),
            "model_config": asdict(self.model.config),
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "replay_resume_policy": "flushed_checkpoint_boundary_v1",
        }
        for name, expected in expected_scalars.items():
            if bundle[name] != expected:
                raise CheckpointCompatibilityError(f"H2 checkpoint {name} mismatch")
        source_sha = bundle["source_git_sha"]
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 40
            or any(character not in "0123456789abcdef" for character in source_sha)
        ):
            raise CheckpointCompatibilityError("H2 checkpoint source Git SHA is invalid")
        if source_sha != git_sha():
            raise CheckpointCompatibilityError("H2 checkpoint source Git SHA mismatch")

        counters = bundle["counters"]
        if not isinstance(counters, dict) or set(counters) != {
            "learner_updates", "samples_consumed"
        }:
            raise CheckpointCompatibilityError("H2 checkpoint counters mismatch")
        learner_updates = counters["learner_updates"]
        samples_consumed = counters["samples_consumed"]
        policy_version = bundle["policy_version"]
        for name, value in (
            ("learner_updates", learner_updates),
            ("samples_consumed", samples_consumed),
            ("policy_version", policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointCompatibilityError(f"H2 checkpoint {name} is invalid")
        if policy_version != self.config.initial_policy_version + learner_updates:
            raise CheckpointCompatibilityError("H2 checkpoint policy version drift")
        schedule = bundle["schedule_state"]
        expected_schedule = {
            "learner_update": learner_updates,
            "gamma": self.config.adaptive_dmc.gamma_at(learner_updates),
        }
        if schedule != expected_schedule:
            raise CheckpointCompatibilityError("H2 checkpoint schedule state mismatch")

        try:
            statistics = AdaptiveDMCCumulativeStats.from_state_dict(
                bundle["adaptive_statistics"]
            )
            statistics.validate_invariants(
                mode=self.config.adaptive_dmc.mode,
                learner_updates=learner_updates,
                samples_consumed=samples_consumed,
                batch_size=self.config.batch_size,
                role_weights=self.config.role_weights,
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"H2 checkpoint cumulative statistics mismatch: {exc}"
            ) from exc
        state_dict = bundle["model_state_dict"]
        if not isinstance(state_dict, dict) or set(state_dict) != set(self.model.state_dict()):
            raise CheckpointCompatibilityError("H2 checkpoint model state keys mismatch")
        forbidden = sorted(
            name for name in state_dict
            if any(token in name.lower() for token in _FORBIDDEN_STATE_NAMES)
        )
        if forbidden:
            raise CheckpointCompatibilityError("H2 checkpoint contains forbidden state")
        optimizer_state = bundle["optimizer_state_dict"]
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
            "state", "param_groups"
        }:
            raise CheckpointCompatibilityError(
                "H2 checkpoint optimizer state envelope mismatch"
            )
        rng = bundle["rng"]
        if not isinstance(rng, dict) or set(rng) != {
            "learner", "python", "numpy", "torch", "cuda"
        }:
            raise CheckpointCompatibilityError("H2 checkpoint RNG fields mismatch")
        try:
            learner_probe = random.Random()
            learner_probe.setstate(rng["learner"])
            python_probe = random.Random()
            python_probe.setstate(rng["python"])
            numpy_state = self._decode_numpy_rng_state(rng["numpy"])
            numpy_probe = np.random.RandomState()
            numpy_probe.set_state(numpy_state)
            torch_probe = torch.Generator(device="cpu")
            torch_probe.set_state(rng["torch"])
            if self.device.type == "cuda":
                if rng["cuda"] is None:
                    raise ValueError("CUDA H2 checkpoint is missing CUDA RNG state")
                cuda_probe = torch.Generator(device=self.device)
                cuda_probe.set_state(rng["cuda"])
            elif rng["cuda"] is not None:
                raise ValueError("CPU H2 checkpoint unexpectedly contains CUDA RNG state")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CheckpointCompatibilityError(
                f"H2 checkpoint RNG state mismatch: {exc}"
            ) from exc

        model_backup = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        optimizer_backup = copy.deepcopy(self.optimizer.state_dict())
        rng_backup = {
            "learner": self.rng.getstate(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }
        try:
            self.model.load_state_dict(state_dict, strict=True)
            self.optimizer.load_state_dict(optimizer_state)
            self.rng.setstate(rng["learner"])
            random.setstate(rng["python"])
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(rng["torch"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], self.device)
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            self.model.load_state_dict(model_backup, strict=True)
            self.optimizer.load_state_dict(optimizer_backup)
            self.rng.setstate(rng_backup["learner"])
            random.setstate(rng_backup["python"])
            np.random.set_state(rng_backup["numpy"])
            torch.random.set_rng_state(rng_backup["torch"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(rng_backup["cuda"], self.device)
            raise CheckpointCompatibilityError(
                f"H2 checkpoint state restore failed transactionally: {exc}"
            ) from exc

        self.learner_updates = learner_updates
        self.samples_consumed = samples_consumed
        self.policy_version = policy_version
        self.statistics = statistics
