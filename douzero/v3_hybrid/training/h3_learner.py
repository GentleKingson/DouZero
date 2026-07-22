"""H3 online Oracle learner with strict public/privileged separation."""

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
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from douzero._version import git_sha
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import ModelInputBundle, model_input_bundles_to_batch

from ..adaptive_dmc import ADMC_DISABLED, adaptive_dmc_loss, transform_dmc_target
from ..config import BELIEF_FEEDBACK_NONE
from ..h2_learner import (
    V3H2LearnerConfig,
    _validate_optimizer_param_groups,
    h2_training_identity,
)
from ..model import V3_HYBRID_ROLES, V3HybridModel
from ..output import BatchedV3HybridModelOutput, V3HybridModelOutput
from ..replay import V3ReplayTransition
from .guidance_config import OracleGuidanceLossConfig
from .oracle_schedule import (
    ORACLE_PHASE_COMPLETE,
    OracleGuidingScheduleConfig,
    OracleScheduleState,
)

if TYPE_CHECKING:
    from douzero.distillation.dataset import OfflineDistillationSample

V3_H3_TRAINER_CHECKPOINT_FORMAT = "v3-hybrid-h3-oracle-trainer-v1"
V3_H3_TRAINING_CONTRACT = "online-privileged-oracle-annealed-public-v1"

_FORBIDDEN_PUBLIC_NAMES = (
    "privileged", "teacher", "oracle", "all_handcards", "hidden_hand"
)
_CHECKPOINT_KEYS = frozenset({
    "format",
    "artifact_access",
    "source_git_sha",
    "student_state_dict",
    "student_optimizer_state_dict",
    "oracle_state_dict",
    "oracle_optimizer_state_dict",
    "feature_schema_hash",
    "model_config",
    "model_config_hash",
    "ruleset_identity",
    "learner_config",
    "learner_config_hash",
    "training_identity",
    "training_identity_hash",
    "counters",
    "schedule_state",
    "statistics",
    "rng",
    "replay_resume_policy",
})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _nonnegative(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _select_real_actions(
    output: BatchedV3HybridModelOutput, index: int, action_count: int
) -> V3HybridModelOutput:
    """Remove batch padding before applying per-decision Oracle guidance."""

    selected = output.select(index)
    if not 0 < action_count <= selected.num_actions:
        raise ValueError("Oracle action count is outside the student output")
    if not bool(selected.action_mask[:action_count].all()) or bool(
        selected.action_mask[action_count:].any()
    ):
        raise ValueError("student batch padding does not match Oracle actions")
    return V3HybridModelOutput(
        dmc_q=selected.dmc_q[:action_count],
        win_logit=selected.win_logit[:action_count],
        score_if_win=selected.score_if_win[:action_count],
        score_if_loss=selected.score_if_loss[:action_count],
        p_win=selected.p_win[:action_count],
        score_mean=selected.score_mean[:action_count],
        action_mask=selected.action_mask[:action_count],
    )


@dataclass(frozen=True)
class V3H3LearnerConfig:
    """All H3 graph, loss, optimizer, and schedule identity axes."""

    public: V3H2LearnerConfig = field(default_factory=V3H2LearnerConfig)
    schedule: OracleGuidingScheduleConfig = field(
        default_factory=OracleGuidingScheduleConfig
    )
    guidance: OracleGuidanceLossConfig = field(
        default_factory=OracleGuidanceLossConfig
    )
    oracle_hidden_size: int = 128
    oracle_value_delta_clamp: float = 32.0
    oracle_learning_rate: float = 1e-4

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.public, V3H2LearnerConfig):
            raise TypeError("public must be V3H2LearnerConfig")
        if not isinstance(self.schedule, OracleGuidingScheduleConfig):
            raise TypeError("schedule must be OracleGuidingScheduleConfig")
        if not isinstance(self.guidance, OracleGuidanceLossConfig):
            raise TypeError("guidance must be OracleGuidanceLossConfig")
        if (
            isinstance(self.oracle_hidden_size, bool)
            or not isinstance(self.oracle_hidden_size, int)
            or self.oracle_hidden_size < 1
        ):
            raise ValueError("oracle_hidden_size must be a positive int")
        for name in ("oracle_value_delta_clamp", "oracle_learning_rate"):
            value = getattr(self, name)
            _nonnegative(name, value)
            if value == 0.0:
                raise ValueError(f"{name} must be positive")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "public": self.public.compatibility_dict(),
            "schedule": self.schedule.compatibility_dict(),
            "guidance": asdict(self.guidance),
            "oracle": {
                "hidden_size": self.oracle_hidden_size,
                "value_delta_clamp": float(self.oracle_value_delta_clamp),
                "learning_rate": float(self.oracle_learning_rate),
                "optimizer": "rmsprop_v1",
            },
            "normalization": "real_decisions_role_weight_once_v1",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H3LearnerConfig":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("H3 learner config fields mismatch")
        values = dict(payload)
        values["public"] = V3H2LearnerConfig.from_dict(values["public"])
        values["schedule"] = OracleGuidingScheduleConfig.from_dict(values["schedule"])
        guidance = values["guidance"]
        if not isinstance(guidance, Mapping) or set(guidance) != set(
            OracleGuidanceLossConfig.__dataclass_fields__
        ):
            raise ValueError("H3 guidance config fields mismatch")
        values["guidance"] = OracleGuidanceLossConfig(**dict(guidance))
        return cls(**values)


@dataclass(frozen=True)
class V3H3StepMetrics:
    learner_update: int
    policy_version: int
    phase: str
    temperature: float
    privileged_gate: float
    guidance_weight: float
    oracle_weight: float
    samples: int
    public_updated: bool
    oracle_updated: bool
    loss_dmc: float
    loss_oracle: float
    loss_kl: float
    loss_ranking: float
    loss_chosen_value: float
    loss_total: float
    public_gradient_norm: float
    oracle_gradient_norm: float
    action_agreement: float | None
    value_error_abs: float | None
    role_samples: dict[str, int]
    role_effective_weights: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class H3CumulativeStats:
    _FIELDS = (
        "steps", "decisions", "public_updates", "oracle_updates", "agreement_sum",
        "agreement_decisions", "value_error_abs_sum", "kl_sum", "loss_sum",
        "dmc_updates", "dmc_decisions", "q_drift_abs_sum", "ratio_clip_count",
        "near_zero_count", "target_clamp_count", "non_finite_fallback_count",
    )

    def __init__(self) -> None:
        self.steps = 0
        self.decisions = 0
        self.public_updates = 0
        self.oracle_updates = 0
        self.agreement_sum = 0.0
        self.agreement_decisions = 0
        self.value_error_abs_sum = 0.0
        self.kl_sum = 0.0
        self.loss_sum = 0.0
        self.dmc_updates = 0
        self.dmc_decisions = 0
        self.q_drift_abs_sum = 0.0
        self.ratio_clip_count = 0
        self.near_zero_count = 0
        self.target_clamp_count = 0
        self.non_finite_fallback_count = 0

    def update(
        self,
        metrics: V3H3StepMetrics,
        *,
        dmc_result=None,
        q_new: torch.Tensor | None = None,
        q_old: torch.Tensor | None = None,
    ) -> None:
        self.steps += 1
        self.decisions += metrics.samples
        self.public_updates += int(metrics.public_updated)
        self.oracle_updates += int(metrics.oracle_updated)
        if metrics.action_agreement is not None:
            self.agreement_sum += metrics.action_agreement * metrics.samples
            self.value_error_abs_sum += float(metrics.value_error_abs) * metrics.samples
            self.agreement_decisions += metrics.samples
        self.kl_sum += metrics.loss_kl * metrics.samples
        self.loss_sum += metrics.loss_total
        if dmc_result is not None:
            if q_new is None:
                raise ValueError("H3 DMC statistics require q_new")
            self.dmc_updates += 1
            self.dmc_decisions += int(q_new.numel())
            if q_old is not None:
                self.q_drift_abs_sum += float(
                    (q_new.detach() - q_old.detach()).abs().sum().cpu().item()
                )
            self.ratio_clip_count += int(dmc_result.ratio_clipped.sum().item())
            self.near_zero_count += int(dmc_result.near_zero_fallback.sum().item())
            self.target_clamp_count += int(dmc_result.target_clamped.sum().item())
            self.non_finite_fallback_count += int(
                dmc_result.non_finite_fallback.sum().item()
            )

    def state_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "H3CumulativeStats":
        if not isinstance(payload, Mapping) or set(payload) != set(cls._FIELDS):
            raise ValueError("H3 statistics fields mismatch")
        result = cls()
        integer = {
            "steps", "decisions", "public_updates", "oracle_updates",
            "agreement_decisions", "dmc_updates", "dmc_decisions",
            "ratio_clip_count", "near_zero_count", "target_clamp_count",
            "non_finite_fallback_count",
        }
        for name in cls._FIELDS:
            value = payload[name]
            if name in integer:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"H3 statistic {name} is invalid")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"H3 statistic {name} is invalid")
            setattr(result, name, value)
        if result.agreement_decisions > result.decisions:
            raise ValueError("H3 agreement count exceeds decisions")
        if result.public_updates > result.steps or result.oracle_updates > result.steps:
            raise ValueError("H3 optimizer updates exceed learner steps")
        if result.dmc_updates > result.public_updates or result.dmc_decisions > result.decisions:
            raise ValueError("H3 DMC statistics exceed public training state")
        for name in (
            "ratio_clip_count", "near_zero_count", "target_clamp_count",
            "non_finite_fallback_count",
        ):
            if getattr(result, name) > result.dmc_decisions:
                raise ValueError("H3 DMC event count exceeds DMC decisions")
        return result


def h3_training_identity(
    model: V3HybridModel, ruleset: RuleSet, config: V3H3LearnerConfig
) -> dict[str, object]:
    return {
        "identity_version": 1,
        "training_contract": V3_H3_TRAINING_CONTRACT,
        "h2_public_identity": h2_training_identity(model, ruleset, config.public),
        "feature_flags": {
            "adaptive_dmc": (
                config.public.lambda_dmc > 0.0
                and config.public.adaptive_dmc.enabled
            ),
            "oracle": config.schedule.enabled,
            "belief": False,
            "cooperation": False,
            "bidding": False,
        },
        "oracle_graph": (
            {
                "access": "privileged_training_only",
                "hidden_size": config.oracle_hidden_size,
                "value_delta_clamp": float(config.oracle_value_delta_clamp),
                "separate_public_backbone": True,
            }
            if config.schedule.enabled
            else {"version": "disabled_no_parameters"}
        ),
        "schedule": config.schedule.compatibility_dict(),
        "loss": {
            "lambda_oracle": "scheduled_oracle_weight",
            "guidance": asdict(config.guidance),
            "public_dmc": config.public.adaptive_dmc.compatibility_dict(),
            "normalization": "real_decisions_role_weight_once_v1",
        },
        "trainer": config.compatibility_dict(),
        "topology": "standalone_online_oracle_h3_no_actors_v1",
        "public_export": "h1_public_sidecar_student_only_v1",
        "replay_resume_policy": "flushed_checkpoint_boundary_v1",
    }


def _same_public_bundle(left: ModelInputBundle, right: ModelInputBundle) -> bool:
    scalar_names = (
        "state_context_flat", "context_flat", "history_tokens",
        "history_key_padding_mask", "action_features", "action_mask",
    )
    return (
        left.acting_role == right.acting_role
        and left.feature_schema_hash == right.feature_schema_hash
        and left.strategy_features is None
        and right.strategy_features is None
        and left.style_features is None
        and right.style_features is None
        and len(left.state_card_vectors) == len(right.state_card_vectors)
        and len(left.context_card_vectors) == len(right.context_card_vectors)
        and all(torch.equal(a, b) for a, b in zip(left.state_card_vectors, right.state_card_vectors))
        and all(torch.equal(a, b) for a, b in zip(left.context_card_vectors, right.context_card_vectors))
        and all(torch.equal(getattr(left, name), getattr(right, name)) for name in scalar_names)
    )


class V3H3Learner:
    """Combined public DMC/ADMC and separately optimized privileged Oracle."""

    def __init__(
        self,
        model: V3HybridModel,
        *,
        ruleset: RuleSet,
        config: V3H3LearnerConfig | None = None,
        oracle=None,
        _allow_h4_belief_feedback: bool = False,
    ) -> None:
        if not isinstance(model, V3HybridModel):
            raise TypeError("H3 learner requires a V3HybridModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("H3 learner requires a RuleSet")
        self.config = config or V3H3LearnerConfig()
        if model.config.belief_feedback != BELIEF_FEEDBACK_NONE:
            if not _allow_h4_belief_feedback:
                raise ValueError(
                    "belief-feedback models require the H4 learner"
                )
            if self.config.schedule.enabled:
                raise ValueError(
                    "H4 belief feedback and H3 Oracle combine only in H6"
                )
        self.device = torch.device(self.config.public.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA H3 learner requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.ruleset = ruleset
        ruleset_identity = (
            ruleset.ruleset_id, ruleset.ruleset_version, ruleset.stable_hash()
        )
        existing = getattr(model, "expected_ruleset_identity", None)
        if existing is not None and existing != ruleset_identity:
            raise ValueError("H3 model and learner ruleset identities differ")
        self.model.expected_ruleset_identity = ruleset_identity
        self.model.train()
        public = self.config.public
        self.student_optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=public.learning_rate,
            alpha=public.rmsprop_alpha,
            momentum=public.rmsprop_momentum,
            eps=public.rmsprop_epsilon,
        )
        self.oracle = None
        self.oracle_optimizer = None
        if self.config.schedule.enabled:
            from .oracle import V3OracleConfig, V3PrivilegedOracle

            expected_config = V3OracleConfig(
                hidden_size=self.config.oracle_hidden_size,
                value_delta_clamp=self.config.oracle_value_delta_clamp,
            )
            self.oracle = oracle or V3PrivilegedOracle(self.model, expected_config)
            if not isinstance(self.oracle, V3PrivilegedOracle):
                raise TypeError("enabled H3 learner requires a V3PrivilegedOracle")
            if self.oracle.config != expected_config:
                raise ValueError("Oracle graph does not match H3 learner config")
            if self.oracle.public_model_config != self.model.config:
                raise ValueError("Oracle and student public graph configs differ")
            self.oracle.to(self.device).train()
            self.oracle_optimizer = torch.optim.RMSprop(
                self.oracle.parameters(),
                lr=self.config.oracle_learning_rate,
                alpha=public.rmsprop_alpha,
                momentum=public.rmsprop_momentum,
                eps=public.rmsprop_epsilon,
            )
        elif oracle is not None:
            raise ValueError("Oracle object supplied while H3 Oracle is disabled")
        self.rng = random.Random(public.seed)
        self.learner_updates = 0
        self.samples_consumed = 0
        self.policy_version = public.initial_policy_version
        self.statistics = H3CumulativeStats()
        self.compatibility_identity = h3_training_identity(
            self.model, self.ruleset, self.config
        )
        self.compatibility_hash = _canonical_hash(self.compatibility_identity)

    def schedule_state(self) -> OracleScheduleState:
        return self.config.schedule.at(self.learner_updates)

    def _guidance_enabled(self) -> bool:
        guidance = self.config.guidance
        return any(
            value > 0.0
            for value in (
                guidance.lambda_kl,
                guidance.lambda_ranking,
                guidance.lambda_chosen_value,
            )
        )

    def _privileged_needed(self, state: OracleScheduleState) -> bool:
        return state.privileged_required and (
            state.oracle_weight > 0.0
            or (state.guidance_weight > 0.0 and self._guidance_enabled())
        )

    def _validate_batch(
        self,
        transitions: Sequence[V3ReplayTransition],
        oracle_samples: Sequence["OfflineDistillationSample"] | None,
        state: OracleScheduleState,
        effective_policy_version: int,
    ) -> tuple[list[float], Sequence["OfflineDistillationSample"] | None]:
        if not transitions or len(transitions) > self.config.public.batch_size:
            raise ValueError("H3 requires a non-empty batch within configured batch_size")
        dmc_enabled = state.public_training and self.config.public.lambda_dmc > 0.0
        adaptive_configured = self.config.public.adaptive_dmc.mode != ADMC_DISABLED
        adaptive_consumed = dmc_enabled and adaptive_configured
        for transition in transitions:
            if not isinstance(transition, V3ReplayTransition):
                raise TypeError("H3 learner accepts only V3ReplayTransition")
            # Warmup and guidance-only phases preserve actor provenance without
            # consuming q_old. A disabled Adaptive-DMC configuration still
            # rejects it, while an active DMC update requires it on every row.
            adaptive_required = adaptive_consumed or (
                adaptive_configured
                and transition.adaptive_provenance is not None
            )
            transition.validate(
                expected_schema_hash=self.model.schema.stable_hash(),
                expected_target_transform=self.model.config.dmc_target_transform,
                expected_ruleset_identity=self.ruleset.identity(),
                adaptive_required=adaptive_required,
            )
            if adaptive_consumed:
                version = transition.adaptive_provenance.policy_version
                if version > effective_policy_version:
                    raise ValueError("H3 replay policy version is newer than learner")
        if self._privileged_needed(state):
            from douzero.distillation.dataset import OfflineDistillationSample

            if oracle_samples is None or len(oracle_samples) != len(transitions):
                raise ValueError("current H3 phase requires one privileged sample per transition")
            for transition, sample in zip(transitions, oracle_samples):
                if not isinstance(sample, OfflineDistillationSample):
                    raise TypeError("H3 reuses OfflineDistillationSample for Oracle data")
                if sample.action_index != transition.selected_action_index:
                    raise ValueError("Oracle/public selected action index mismatch")
                if not math.isclose(sample.target_score, transition.mc_return, rel_tol=0.0, abs_tol=0.0):
                    raise ValueError("Oracle/public terminal target mismatch")
                if not _same_public_bundle(sample.public_inputs, transition.model_inputs):
                    raise ValueError("Oracle/public tensor bundle mismatch")
        elif oracle_samples is not None:
            raise ValueError("public-only H3 phase rejects privileged samples")
        return [self.config.public.role_weights[t.role] for t in transitions], oracle_samples

    def train_batch(
        self,
        transitions: Sequence[V3ReplayTransition],
        *,
        oracle_samples: Sequence["OfflineDistillationSample"] | None = None,
        belief_features: torch.Tensor | None = None,
        external_policy_version_offset: int = 0,
    ) -> V3H3StepMetrics:
        if (
            isinstance(external_policy_version_offset, bool)
            or not isinstance(external_policy_version_offset, int)
            or external_policy_version_offset < 0
        ):
            raise ValueError("external policy version offset must be non-negative")
        effective_policy_version = (
            self.policy_version + external_policy_version_offset
        )
        state = self.schedule_state()
        if state.phase == ORACLE_PHASE_COMPLETE:
            raise RuntimeError("H3 training schedule is complete")
        role_weights, samples = self._validate_batch(
            transitions, oracle_samples, state, effective_policy_version
        )
        effective_weight = math.fsum(role_weights)
        if effective_weight == 0.0:
            return self._empty_metrics(state, effective_policy_version)
        normalized = torch.tensor(
            [weight / effective_weight for weight in role_weights],
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        public_needed = state.public_training and (
            self.config.public.lambda_dmc > 0.0
            or (state.guidance_weight > 0.0 and self._guidance_enabled())
        )
        output = None
        dmc_loss = normalized.sum() * 0.0
        dmc_result = None
        q_new = None
        q_old = None
        if public_needed:
            inputs = model_input_bundles_to_batch(
                [transition.model_inputs for transition in transitions],
                [transition.selected_action_index for transition in transitions],
            ).to(self.device)
            output = self.model.forward_input_batch(
                inputs, belief_features=belief_features
            )
            gathered = output.gather_chosen(inputs.chosen_action_index)
            q_new = gathered["dmc_q"].squeeze(-1)
            if self.config.public.lambda_dmc > 0.0:
                returns = q_new.new_tensor([
                    transition.mc_return for transition in transitions
                ])
                if self.config.public.adaptive_dmc.mode != ADMC_DISABLED:
                    q_old = q_new.new_tensor([
                        transition.adaptive_provenance.q_old for transition in transitions
                    ])
                dmc_result = adaptive_dmc_loss(
                    q_new,
                    returns,
                    config=self.config.public.adaptive_dmc,
                    target_transform=self.model.config.dmc_target_transform,
                    target_clamp=self.model.config.dmc_target_clamp,
                    learner_update=self.learner_updates,
                    q_old=q_old,
                )
                dmc_loss = (dmc_result.loss_per_sample * normalized).sum()

        oracle_losses = []
        guidance_losses = []
        if self._privileged_needed(state):
            assert samples is not None and self.oracle is not None
            for index, sample in enumerate(samples):
                oracle_output = self.oracle(
                    sample.public_inputs,
                    sample.privileged_observation,
                    action_keys=sample.action_keys,
                    privileged_gate=state.privileged_gate,
                )
                target, _ = transform_dmc_target(
                    oracle_output.action_logits.new_tensor([sample.target_score]),
                    transform=self.model.config.dmc_target_transform,
                    clamp=self.model.config.dmc_target_clamp,
                )
                oracle_losses.append(
                    F.mse_loss(
                        oracle_output.action_logits[sample.action_index].reshape(1), target
                    )
                )
                if state.guidance_weight > 0.0 and self._guidance_enabled():
                    from .oracle_loss import oracle_guidance_loss

                    assert output is not None
                    guidance_losses.append(
                        oracle_guidance_loss(
                            _select_real_actions(
                                output, index, len(sample.action_keys)
                            ),
                            sample.action_keys,
                            oracle_output,
                            chosen_action_index=sample.action_index,
                            temperature=state.temperature,
                            config=self.config.guidance,
                        )
                    )

        zero = next(self.model.parameters()).sum() * 0.0
        oracle_loss = zero
        if oracle_losses:
            oracle_loss = torch.stack(oracle_losses).mul(normalized).sum()
        kl = zero
        ranking = zero
        chosen_value = zero
        agreement = None
        value_error = None
        if guidance_losses:
            kl = torch.stack([item.kl for item in guidance_losses]).mul(normalized).sum()
            ranking = torch.stack([item.ranking for item in guidance_losses]).mul(normalized).sum()
            chosen_value = torch.stack([item.chosen_value for item in guidance_losses]).mul(normalized).sum()
            guidance_total = torch.stack([item.total for item in guidance_losses]).mul(normalized).sum()
            agreement = math.fsum(
                item.agreement * weight for item, weight in zip(guidance_losses, normalized.tolist())
            )
            value_error = math.fsum(
                item.value_error_abs * weight for item, weight in zip(guidance_losses, normalized.tolist())
            )
        else:
            guidance_total = zero
        student_total = (
            self.config.public.lambda_dmc * dmc_loss
            + state.guidance_weight * guidance_total
        )
        oracle_total = state.oracle_weight * oracle_loss
        combined = student_total + oracle_total
        if not bool(torch.isfinite(combined)):
            raise FloatingPointError("H3 total loss is NaN or Inf")

        public_updated = bool(public_needed and student_total.requires_grad)
        oracle_updated = bool(oracle_losses and state.oracle_weight > 0.0)
        public_grad = 0.0
        oracle_grad = 0.0
        if public_updated:
            self.student_optimizer.zero_grad(set_to_none=True)
        if oracle_updated:
            self.oracle_optimizer.zero_grad(set_to_none=True)
        if public_updated or oracle_updated:
            combined.backward()
        if public_updated:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.public.max_grad_norm,
                error_if_nonfinite=True,
            )
            public_grad = float(norm.detach().cpu().item())
            self.student_optimizer.step()
            self.policy_version += 1
        if oracle_updated:
            norm = torch.nn.utils.clip_grad_norm_(
                self.oracle.parameters(), self.config.public.max_grad_norm,
                error_if_nonfinite=True,
            )
            oracle_grad = float(norm.detach().cpu().item())
            self.oracle_optimizer.step()

        if (
            not public_updated
            and not oracle_updated
            and not self.config.schedule.enabled
        ):
            return self._empty_metrics(state, effective_policy_version)
        count = len(transitions)
        self.learner_updates += 1
        self.samples_consumed += count
        role_samples = {
            role: sum(transition.role == role for transition in transitions)
            for role in V3_HYBRID_ROLES
        }
        role_effective = {
            role: math.fsum(
                weight for transition, weight in zip(transitions, role_weights)
                if transition.role == role
            )
            for role in V3_HYBRID_ROLES
        }
        metrics = V3H3StepMetrics(
            learner_update=self.learner_updates,
            policy_version=self.policy_version + external_policy_version_offset,
            phase=state.phase,
            temperature=state.temperature,
            privileged_gate=state.privileged_gate,
            guidance_weight=state.guidance_weight,
            oracle_weight=state.oracle_weight,
            samples=count,
            public_updated=public_updated,
            oracle_updated=oracle_updated,
            loss_dmc=float(dmc_loss.detach().cpu().item()),
            loss_oracle=float(oracle_loss.detach().cpu().item()),
            loss_kl=float(kl.detach().cpu().item()),
            loss_ranking=float(ranking.detach().cpu().item()),
            loss_chosen_value=float(chosen_value.detach().cpu().item()),
            loss_total=float(combined.detach().cpu().item()),
            public_gradient_norm=public_grad,
            oracle_gradient_norm=oracle_grad,
            action_agreement=agreement,
            value_error_abs=value_error,
            role_samples=role_samples,
            role_effective_weights=role_effective,
        )
        self.statistics.update(
            metrics, dmc_result=dmc_result, q_new=q_new, q_old=q_old
        )
        return metrics

    def _empty_metrics(
        self, state: OracleScheduleState, policy_version: int
    ) -> V3H3StepMetrics:
        return V3H3StepMetrics(
            learner_update=self.learner_updates,
            policy_version=policy_version,
            phase=state.phase,
            temperature=state.temperature,
            privileged_gate=state.privileged_gate,
            guidance_weight=state.guidance_weight,
            oracle_weight=state.oracle_weight,
            samples=0,
            public_updated=False,
            oracle_updated=False,
            loss_dmc=0.0,
            loss_oracle=0.0,
            loss_kl=0.0,
            loss_ranking=0.0,
            loss_chosen_value=0.0,
            loss_total=0.0,
            public_gradient_norm=0.0,
            oracle_gradient_norm=0.0,
            action_agreement=None,
            value_error_abs=None,
            role_samples={role: 0 for role in V3_HYBRID_ROLES},
            role_effective_weights={role: 0.0 for role in V3_HYBRID_ROLES},
        )

    @staticmethod
    def _numpy_state(state) -> dict[str, object]:
        return {
            "algorithm": state[0], "keys": state[1].tolist(), "position": int(state[2]),
            "has_gauss": int(state[3]), "cached_gaussian": float(state[4]),
        }

    @staticmethod
    def _decode_numpy(payload: Mapping[str, object]):
        expected = {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("NumPy RNG fields mismatch")
        return (
            str(payload["algorithm"]), np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]), int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )

    def save_checkpoint(self, path: str | Path) -> None:
        source_sha = git_sha()
        if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
            raise RuntimeError("H3 checkpoints require a full source Git SHA")
        student_state = self.model.state_dict()
        if any(
            token in name.lower()
            for name in student_state
            for token in _FORBIDDEN_PUBLIC_NAMES
        ):
            raise RuntimeError("H3 public student contains forbidden state keys")
        bundle = {
            "format": V3_H3_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": source_sha,
            "student_state_dict": student_state,
            "student_optimizer_state_dict": self.student_optimizer.state_dict(),
            "oracle_state_dict": None if self.oracle is None else self.oracle.state_dict(),
            "oracle_optimizer_state_dict": (
                None if self.oracle_optimizer is None else self.oracle_optimizer.state_dict()
            ),
            "feature_schema_hash": self.model.schema.stable_hash(),
            "model_config": asdict(self.model.config),
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "counters": {
                "learner_updates": self.learner_updates,
                "samples_consumed": self.samples_consumed,
                "policy_version": self.policy_version,
            },
            "schedule_state": self.schedule_state().as_dict(),
            "statistics": self.statistics.state_dict(),
            "rng": {
                "learner": self.rng.getstate(),
                "python": random.getstate(),
                "numpy": self._numpy_state(np.random.get_state()),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            },
            "replay_resume_policy": "flushed_checkpoint_boundary_v1",
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointCompatibilityError(f"unable to safely load H3 checkpoint: {exc}") from exc
        if not isinstance(bundle, dict) or set(bundle) != _CHECKPOINT_KEYS:
            raise CheckpointCompatibilityError("H3 checkpoint envelope mismatch")
        expected_scalars = {
            "format": V3_H3_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": git_sha(),
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
                raise CheckpointCompatibilityError(f"H3 checkpoint {name} mismatch")
        counters = bundle["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "learner_updates", "samples_consumed", "policy_version"
        }:
            raise CheckpointCompatibilityError("H3 checkpoint counters mismatch")
        try:
            learner_updates = counters["learner_updates"]
            samples_consumed = counters["samples_consumed"]
            policy_version = counters["policy_version"]
            for name, value in (
                ("learner_updates", learner_updates),
                ("samples_consumed", samples_consumed),
                ("policy_version", policy_version),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid H3 {name}")
            if bundle["schedule_state"] != self.config.schedule.at(learner_updates).as_dict():
                raise ValueError("H3 schedule state drift")
            statistics = H3CumulativeStats.from_state_dict(bundle["statistics"])
            if statistics.steps != learner_updates or statistics.decisions != samples_consumed:
                raise ValueError("H3 statistics/counter drift")
            expected_policy_version = (
                self.config.public.initial_policy_version
                + statistics.public_updates
            )
            if policy_version != expected_policy_version:
                raise ValueError("H3 policy version drift")
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(f"H3 checkpoint state mismatch: {exc}") from exc
        student_state = bundle["student_state_dict"]
        if not isinstance(student_state, dict) or set(student_state) != set(self.model.state_dict()):
            raise CheckpointCompatibilityError("H3 student state keys mismatch")
        if any(
            token in name.lower()
            for name in student_state
            for token in _FORBIDDEN_PUBLIC_NAMES
        ):
            raise CheckpointCompatibilityError("H3 public student contains forbidden state")
        oracle_state = bundle["oracle_state_dict"]
        oracle_optimizer_state = bundle["oracle_optimizer_state_dict"]
        if self.oracle is None:
            if oracle_state is not None or oracle_optimizer_state is not None:
                raise CheckpointCompatibilityError("disabled H3 checkpoint contains Oracle state")
        elif (
            not isinstance(oracle_state, dict)
            or set(oracle_state) != set(self.oracle.state_dict())
            or not isinstance(oracle_optimizer_state, dict)
        ):
            raise CheckpointCompatibilityError("enabled H3 checkpoint Oracle state mismatch")
        student_optimizer_state = bundle["student_optimizer_state_dict"]
        if not isinstance(student_optimizer_state, dict) or set(student_optimizer_state) != {
            "state", "param_groups"
        }:
            raise CheckpointCompatibilityError("H3 student optimizer state mismatch")
        _validate_optimizer_param_groups(
            student_optimizer_state["param_groups"],
            self.student_optimizer.state_dict()["param_groups"],
        )
        if self.oracle_optimizer is not None:
            if set(oracle_optimizer_state) != {"state", "param_groups"}:
                raise CheckpointCompatibilityError("H3 Oracle optimizer envelope mismatch")
            _validate_optimizer_param_groups(
                oracle_optimizer_state["param_groups"],
                self.oracle_optimizer.state_dict()["param_groups"],
            )
        rng = bundle["rng"]
        if not isinstance(rng, Mapping) or set(rng) != {"learner", "python", "numpy", "torch", "cuda"}:
            raise CheckpointCompatibilityError("H3 checkpoint RNG fields mismatch")
        try:
            numpy_state = self._decode_numpy(rng["numpy"])
            random.Random().setstate(rng["learner"])
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(numpy_state)
            torch.Generator(device="cpu").set_state(rng["torch"])
            if self.device.type == "cuda":
                if rng["cuda"] is None:
                    raise ValueError("CUDA H3 checkpoint lacks CUDA RNG")
                torch.Generator(device=self.device).set_state(rng["cuda"])
            elif rng["cuda"] is not None:
                raise ValueError("CPU H3 checkpoint contains CUDA RNG")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CheckpointCompatibilityError(f"H3 checkpoint RNG mismatch: {exc}") from exc

        backup = {
            "student": copy.deepcopy(self.model.state_dict()),
            "student_optimizer": copy.deepcopy(self.student_optimizer.state_dict()),
            "oracle": None if self.oracle is None else copy.deepcopy(self.oracle.state_dict()),
            "oracle_optimizer": (
                None if self.oracle_optimizer is None else copy.deepcopy(self.oracle_optimizer.state_dict())
            ),
            "learner_rng": self.rng.getstate(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }
        try:
            self.model.load_state_dict(student_state, strict=True)
            self.student_optimizer.load_state_dict(student_optimizer_state)
            if self.oracle is not None:
                self.oracle.load_state_dict(oracle_state, strict=True)
                self.oracle_optimizer.load_state_dict(oracle_optimizer_state)
            self.rng.setstate(rng["learner"])
            random.setstate(rng["python"])
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(rng["torch"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], self.device)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.model.load_state_dict(backup["student"], strict=True)
            self.student_optimizer.load_state_dict(backup["student_optimizer"])
            if self.oracle is not None:
                self.oracle.load_state_dict(backup["oracle"], strict=True)
                self.oracle_optimizer.load_state_dict(backup["oracle_optimizer"])
            self.rng.setstate(backup["learner_rng"])
            random.setstate(backup["python_rng"])
            np.random.set_state(backup["numpy_rng"])
            torch.random.set_rng_state(backup["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(backup["cuda_rng"], self.device)
            raise CheckpointCompatibilityError(f"H3 checkpoint restore failed: {exc}") from exc
        self.learner_updates = learner_updates
        self.samples_consumed = samples_consumed
        self.policy_version = policy_version
        self.statistics = statistics
