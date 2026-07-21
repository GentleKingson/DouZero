"""H4 conservative belief auxiliary training and detached policy feedback."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from douzero._version import git_sha
from douzero.belief.features import BeliefInput, build_belief_input
from douzero.belief.labels import BeliefLabel, build_belief_label
from douzero.belief.losses import belief_loss
from douzero.belief.model import BeliefModel, belief_features_from_probs
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import (
    ModelInputBundle,
    model_input_bundles_to_batch,
    observation_to_model_inputs,
)
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.privileged import PrivilegedObservation

from ..config import BELIEF_FEEDBACK_NONE
from ..h2_learner import _validate_optimizer_param_groups
from ..model import V3_HYBRID_ROLES, V3HybridModel
from ..replay import V3ReplayTransition
from .belief_config import (
    BELIEF_MODE_ALTERNATING,
    BELIEF_PHASE_AUXILIARY,
    BELIEF_PHASE_DISABLED,
    BELIEF_PHASE_POLICY,
    BELIEF_PHASE_SHARED,
    BELIEF_PHASE_SUPERVISED,
    V3H4BeliefTrainingConfig,
)
from .h3_learner import (
    V3H3Learner,
    V3H3LearnerConfig,
    V3H3StepMetrics,
    _same_public_bundle,
)

V3_H4_TRAINER_CHECKPOINT_FORMAT = "v3-hybrid-h4-joint-belief-trainer-v3"
V3_H4_TRAINING_CONTRACT = "conservative-belief-detached-policy-feedback-v3"

_CHECKPOINT_KEYS = frozenset({
    "format",
    "artifact_access",
    "source_git_sha",
    "model_config_hash",
    "belief_model_config_hash",
    "ruleset_identity",
    "learner_config",
    "learner_config_hash",
    "training_identity",
    "training_identity_hash",
    "h3_checkpoint",
    "belief_state_dict",
    "belief_optimizer_state_dict",
    "counters",
    "phase",
    "statistics",
})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


class _Digest(Protocol):
    def update(self, value: bytes, /) -> None: ...


def _update_array_identity(
    digest: _Digest, name: str, value: np.ndarray | torch.Tensor
) -> None:
    array = (
        value.detach().cpu().contiguous().numpy()
        if isinstance(value, torch.Tensor)
        else np.ascontiguousarray(value)
    )
    metadata = json.dumps(
        {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest.update(metadata.encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _h4_source_state_identity(
    public_inputs: ModelInputBundle, belief_input: BeliefInput
) -> str:
    """Fingerprint both public encodings produced from one observation."""

    digest = hashlib.sha256(b"v3-h4-source-state-binding-v1")
    for index, value in enumerate(public_inputs.state_card_vectors):
        _update_array_identity(digest, f"state_card_vectors.{index}", value)
    _update_array_identity(
        digest, "state_context_flat", public_inputs.state_context_flat
    )
    for index, value in enumerate(public_inputs.context_card_vectors):
        _update_array_identity(digest, f"context_card_vectors.{index}", value)
    for name in (
        "context_flat",
        "history_tokens",
        "history_key_padding_mask",
        "action_features",
        "action_mask",
    ):
        _update_array_identity(digest, name, getattr(public_inputs, name))
    for name in ("strategy_features", "style_features"):
        value = getattr(public_inputs, name)
        if value is None:
            digest.update(f"{name}=none".encode("ascii"))
        else:
            _update_array_identity(digest, name, value)
    digest.update(public_inputs.acting_role.encode("ascii"))
    digest.update(public_inputs.feature_schema_hash.encode("ascii"))

    for name in ("feature_vector", "unseen_counts", "style_features"):
        _update_array_identity(
            digest, f"belief.{name}", getattr(belief_input, name)
        )
    belief_metadata = {
        "acting_role": belief_input.acting_role,
        "opponent_a_role": belief_input.opponent_a_role,
        "opponent_b_role": belief_input.opponent_b_role,
        "opponent_a_total": belief_input.opponent_a_total,
        "opponent_b_total": belief_input.opponent_b_total,
    }
    digest.update(
        json.dumps(
            belief_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class V3H4LearnerConfig:
    base: V3H3LearnerConfig = field(default_factory=V3H3LearnerConfig)
    belief: V3H4BeliefTrainingConfig = field(
        default_factory=V3H4BeliefTrainingConfig
    )

    IDENTITY_VERSION = 3

    def __post_init__(self) -> None:
        if not isinstance(self.base, V3H3LearnerConfig):
            raise TypeError("H4 base must be V3H3LearnerConfig")
        if not isinstance(self.belief, V3H4BeliefTrainingConfig):
            raise TypeError("H4 belief must be V3H4BeliefTrainingConfig")
        if self.belief.enabled and self.base.schedule.enabled:
            raise ValueError(
                "H4 belief and H3 Oracle integration is deferred to H6"
            )
        if (
            self.belief.enabled
            and self.belief.mode == BELIEF_MODE_ALTERNATING
            and self.base.public.lambda_dmc == 0.0
        ):
            raise ValueError(
                "alternating H4 policy phases require lambda_dmc > 0"
            )

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "base": self.base.compatibility_dict(),
            "belief": self.belief.compatibility_dict(),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H4LearnerConfig":
        if not isinstance(payload, Mapping) or set(payload) != {"base", "belief"}:
            raise ValueError("H4 learner config fields mismatch")
        belief = payload["belief"]
        if not isinstance(belief, Mapping) or set(belief) != set(
            V3H4BeliefTrainingConfig.__dataclass_fields__
        ):
            raise ValueError("H4 belief config fields mismatch")
        return cls(
            base=V3H3LearnerConfig.from_dict(payload["base"]),
            belief=V3H4BeliefTrainingConfig(**dict(belief)),
        )


@dataclass(frozen=True)
class V3H4BeliefSample:
    """Public belief input plus an optional training-only true-hand label."""

    public_inputs: ModelInputBundle
    belief_input: BeliefInput
    source_state_identity: str
    label: BeliefLabel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.public_inputs, ModelInputBundle):
            raise TypeError("H4 public_inputs must be a ModelInputBundle")
        if not isinstance(self.belief_input, BeliefInput):
            raise TypeError("H4 belief_input must be a BeliefInput")
        if self.label is not None and not isinstance(self.label, BeliefLabel):
            raise TypeError("H4 label must be BeliefLabel or None")
        if (
            not isinstance(self.source_state_identity, str)
            or len(self.source_state_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_state_identity
            )
        ):
            raise ValueError("H4 source-state identity must be a full SHA-256")
        if self.source_state_identity != _h4_source_state_identity(
            self.public_inputs, self.belief_input
        ):
            raise ValueError("H4 belief sample source-state identity mismatch")
        if self.public_inputs.acting_role != self.belief_input.acting_role:
            raise ValueError("H4 public and belief acting roles differ")
        if self.label is not None:
            if self.label.opponent_a_role != self.belief_input.opponent_a_role:
                raise ValueError("H4 public and label opponent roles differ")
            if self.label.opponent_a_total != self.belief_input.opponent_a_total:
                raise ValueError("H4 public and label opponent totals differ")
            if not np.array_equal(
                self.label.unseen_counts, self.belief_input.unseen_counts
            ):
                raise ValueError("H4 public and label unseen pools differ")


def build_v3_h4_belief_sample(
    observation: ObservationV2,
    privileged: PrivilegedObservation | None = None,
) -> V3H4BeliefSample:
    """Bind one public policy input to an optional privileged belief label."""

    if not isinstance(observation, ObservationV2):
        raise TypeError("H4 sample requires ObservationV2")
    binput = build_belief_input(observation.public)
    label = None
    if privileged is not None:
        if not isinstance(privileged, PrivilegedObservation):
            raise TypeError("H4 privileged input must be PrivilegedObservation")
        if privileged.acting_role != observation.public.acting_role:
            raise ValueError("H4 public and privileged acting roles differ")
        public = observation.public
        label = build_belief_label(
            acting_role=public.acting_role,
            all_handcards=privileged.all_handcards,
            unseen_counts=binput.unseen_counts,
            num_cards_left=public.num_cards_left,
            bottom_unplayed=public.bottom_cards.unplayed,
        )
    public_inputs = observation_to_model_inputs(observation)
    return V3H4BeliefSample(
        public_inputs=public_inputs,
        belief_input=binput,
        source_state_identity=_h4_source_state_identity(public_inputs, binput),
        label=label,
    )


@dataclass(frozen=True)
class V3H4StepMetrics:
    eligible_update: int
    phase: str
    samples: int
    policy_updated: bool
    belief_updated: bool
    shared_encoder_updated: bool
    labels_consumed: int
    loss_belief: float
    belief_cross_entropy: float
    rank_accuracy: float | None
    map_exact_match: float | None
    posterior_calibration_error: float | None
    conservation_max_error: float | None
    belief_gradient_norm: float
    shared_gradient_norm: float
    dp_latency_ms: float
    role_samples: dict[str, int]
    role_effective_weights: dict[str, float]
    base: V3H3StepMetrics | None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base"] = None if self.base is None else self.base.as_dict()
        return payload


class H4CumulativeStats:
    _FIELDS = (
        "steps",
        "decisions",
        "policy_updates",
        "oracle_updates",
        "base_updates",
        "base_decisions",
        "belief_updates",
        "shared_updates",
        "labels_consumed",
        "belief_loss_sum",
        "dp_latency_ms_sum",
    )

    def __init__(self) -> None:
        self.steps = 0
        self.decisions = 0
        self.policy_updates = 0
        self.oracle_updates = 0
        self.base_updates = 0
        self.base_decisions = 0
        self.belief_updates = 0
        self.shared_updates = 0
        self.labels_consumed = 0
        self.belief_loss_sum = 0.0
        self.dp_latency_ms_sum = 0.0

    def update(self, metrics: V3H4StepMetrics) -> None:
        self.steps += 1
        self.decisions += metrics.samples
        self.policy_updates += int(metrics.policy_updated)
        base_updated = bool(
            metrics.base
            and (metrics.base.public_updated or metrics.base.oracle_updated)
        )
        self.oracle_updates += int(bool(metrics.base and metrics.base.oracle_updated))
        self.base_updates += int(base_updated)
        self.base_decisions += 0 if metrics.base is None else metrics.base.samples
        self.belief_updates += int(metrics.belief_updated)
        self.shared_updates += int(metrics.shared_encoder_updated)
        self.labels_consumed += metrics.labels_consumed
        self.belief_loss_sum += metrics.loss_belief
        self.dp_latency_ms_sum += metrics.dp_latency_ms

    def state_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "H4CumulativeStats":
        if not isinstance(payload, Mapping) or set(payload) != set(cls._FIELDS):
            raise ValueError("H4 statistics fields mismatch")
        result = cls()
        for name in cls._FIELDS:
            value = payload[name]
            if name in {
                "steps", "decisions", "policy_updates", "oracle_updates",
                "base_updates", "base_decisions", "belief_updates",
                "shared_updates", "labels_consumed",
            }:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid H4 statistic {name}")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid H4 statistic {name}")
            setattr(result, name, value)
        if max(result.policy_updates, result.belief_updates, result.shared_updates) > result.steps:
            raise ValueError("H4 optimizer counts exceed learner steps")
        if max(result.policy_updates, result.oracle_updates) > result.base_updates:
            raise ValueError("H4 nested update counts exceed base updates")
        if result.base_updates > result.steps or result.base_decisions > result.decisions:
            raise ValueError("H4 nested progress exceeds learner progress")
        if result.labels_consumed > result.decisions:
            raise ValueError("H4 label count exceeds decisions")
        return result


def h4_training_identity(
    model: V3HybridModel,
    belief_model: BeliefModel | None,
    ruleset: RuleSet,
    config: V3H4LearnerConfig,
) -> dict[str, object]:
    return {
        "identity_version": 3,
        "training_contract": V3_H4_TRAINING_CONTRACT,
        "model_config_hash": model.config.stable_hash(),
        "belief_model_config_hash": (
            None if belief_model is None else belief_model.config.stable_hash()
        ),
        "ruleset": ruleset.identity(),
        "learner": config.compatibility_dict(),
        "belief_layout": "public_joint_rank_count_conservative_dp_v1",
        "feedback": model.config.belief_feedback,
        "policy_posterior": "exact_constrained_and_detached_v1",
        "supervision": "raw_logits_privileged_label_side_channel_v1",
        "shared_encoder": "state_history_context_optional_v1",
        "replay_protocol": (
            "h2_admc_coupled_public_policy_source_bound_h4_labels_v3"
        ),
        "topology": "single_process_h4_reference_v1",
    }


class V3H4Learner:
    """Schedule H3 public updates and conservative belief supervision."""

    def __init__(
        self,
        model: V3HybridModel,
        *,
        ruleset: RuleSet,
        config: V3H4LearnerConfig | None = None,
        belief_model: BeliefModel | None = None,
    ) -> None:
        if not isinstance(model, V3HybridModel):
            raise TypeError("H4 learner requires V3HybridModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("H4 learner requires RuleSet")
        self.config = config or V3H4LearnerConfig()
        cfg = self.config.belief
        if not cfg.enabled:
            if belief_model is not None:
                raise ValueError("disabled H4 must not construct a belief model")
            if model.config.belief_feedback != BELIEF_FEEDBACK_NONE:
                raise ValueError("disabled H4 requires a belief-disabled V3 graph")
        else:
            if not isinstance(belief_model, BeliefModel):
                raise TypeError("enabled H4 requires the existing BeliefModel")
            expected_context = model.config.hidden_size if cfg.shared_encoder_updates else 0
            if belief_model.config.shared_context_dim != expected_context:
                raise ValueError(
                    "BeliefConfig.shared_context_dim does not match H4 shared encoder mode"
                )
        self.base = V3H3Learner(
            model,
            ruleset=ruleset,
            config=self.config.base,
            _allow_h4_belief_feedback=cfg.enabled,
        )
        self.model = self.base.model
        self.ruleset = ruleset
        self.device = self.base.device
        self.belief_model = None if belief_model is None else belief_model.to(self.device)
        self.belief_optimizer = None
        if self.belief_model is not None:
            self.belief_model.train()
            parameters = list(self.belief_model.parameters())
            if cfg.shared_encoder_updates:
                parameters.extend(self.model.state_encoder.parameters())
                parameters.extend(self.model.history_encoder.parameters())
            public = self.config.base.public
            self.belief_optimizer = torch.optim.RMSprop(
                parameters,
                lr=cfg.learning_rate,
                alpha=public.rmsprop_alpha,
                momentum=public.rmsprop_momentum,
                eps=public.rmsprop_epsilon,
            )
        self.eligible_updates = 0
        self.samples_consumed = 0
        self.statistics = H4CumulativeStats()
        self.compatibility_identity = h4_training_identity(
            self.model, self.belief_model, ruleset, self.config
        )
        self.compatibility_hash = _canonical_hash(self.compatibility_identity)

    def phase(self) -> str:
        return self.config.belief.phase_at(self.eligible_updates)

    def _validate_samples(
        self,
        transitions: Sequence[V3ReplayTransition],
        samples: Sequence[V3H4BeliefSample] | None,
        *,
        needs_public_belief: bool,
        needs_labels: bool,
    ) -> Sequence[V3H4BeliefSample] | None:
        if not needs_public_belief and not needs_labels:
            if samples is not None:
                raise ValueError("current H4 phase does not consume belief data")
            return None
        if samples is None or len(samples) != len(transitions):
            raise ValueError("H4 requires one aligned belief sample per transition")
        for transition, sample in zip(transitions, samples):
            if not isinstance(sample, V3H4BeliefSample):
                raise TypeError("H4 belief samples have an invalid type")
            if not _same_public_bundle(sample.public_inputs, transition.model_inputs):
                raise ValueError("H4 belief/public replay tensors differ")
            if sample.source_state_identity != _h4_source_state_identity(
                sample.public_inputs, sample.belief_input
            ):
                raise ValueError("H4 belief sample source-state identity mismatch")
            if sample.belief_input.acting_role != transition.role:
                raise ValueError("H4 belief/public replay roles differ")
            if needs_labels and sample.label is None:
                raise ValueError("current H4 phase requires privileged belief labels")
        return samples

    def _belief_forward(
        self,
        transitions: Sequence[V3ReplayTransition],
        samples: Sequence[V3H4BeliefSample],
        *,
        shared_gradient: bool,
    ):
        assert self.belief_model is not None
        shared_context = None
        if self.belief_model.config.shared_context_dim:
            batch = model_input_bundles_to_batch(
                [sample.public_inputs for sample in samples],
                [transition.selected_action_index for transition in transitions],
            )
            shared_context = self.model.encode_input_batch_context(batch)
            if not shared_gradient:
                shared_context = shared_context.detach()
        started = time.perf_counter()
        output = self.belief_model(
            [sample.belief_input for sample in samples],
            shared_context=shared_context,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return output, latency_ms

    def _policy_features(
        self,
        transitions: Sequence[V3ReplayTransition],
        samples: Sequence[V3H4BeliefSample],
    ) -> tuple[torch.Tensor, float]:
        assert self.belief_model is not None
        training = self.belief_model.training
        self.belief_model.eval()
        try:
            with torch.no_grad():
                output, latency_ms = self._belief_forward(
                    transitions, samples, shared_gradient=False
                )
                features = belief_features_from_probs(
                    output.constrained_probs,
                    output.opponent_a_total,
                    np.stack([
                        sample.belief_input.unseen_counts for sample in samples
                    ]),
                )
        finally:
            self.belief_model.train(training)
        parameter = next(self.model.parameters())
        return (
            torch.from_numpy(features).to(
                device=self.device, dtype=parameter.dtype
            ),
            latency_ms,
        )

    @staticmethod
    def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        values = [
            parameter.grad.detach().float().norm(2).square()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not values:
            return 0.0
        return float(torch.stack(values).sum().sqrt().cpu().item())

    def train_batch(
        self,
        transitions: Sequence[V3ReplayTransition],
        *,
        belief_samples: Sequence[V3H4BeliefSample] | None = None,
    ) -> V3H4StepMetrics:
        if not transitions or len(transitions) > self.config.base.public.batch_size:
            raise ValueError("H4 requires a non-empty batch within configured size")
        phase = self.phase()
        policy_phase = phase in {
            BELIEF_PHASE_DISABLED, BELIEF_PHASE_AUXILIARY, BELIEF_PHASE_POLICY
        }
        belief_phase = phase in {
            BELIEF_PHASE_AUXILIARY, BELIEF_PHASE_SUPERVISED, BELIEF_PHASE_SHARED
        }
        shared_phase = phase == BELIEF_PHASE_SHARED or (
            phase == BELIEF_PHASE_AUXILIARY
            and self.config.belief.shared_encoder_updates
        )
        feedback = self.model.config.belief_feedback != BELIEF_FEEDBACK_NONE
        samples = self._validate_samples(
            transitions,
            belief_samples,
            needs_public_belief=policy_phase and feedback,
            needs_labels=belief_phase,
        )
        role_weights = [
            self.config.base.public.role_weights[transition.role]
            for transition in transitions
        ]
        effective = math.fsum(role_weights)
        if effective == 0.0:
            return self._metrics_noop(phase)
        normalized = torch.tensor(
            [weight / effective for weight in role_weights],
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        dp_latency_ms = 0.0
        features = None
        if policy_phase and feedback:
            assert samples is not None
            features, latency = self._policy_features(transitions, samples)
            dp_latency_ms += latency
        base_metrics = None
        if policy_phase:
            base_metrics = self.base.train_batch(
                transitions, belief_features=features
            )

        belief_total = next(self.model.parameters()).sum() * 0.0
        cross_entropy = 0.0
        rank_accuracy = None
        exact_match = None
        calibration = None
        conservation_error = None
        belief_grad = 0.0
        shared_grad = 0.0
        labels_consumed = 0
        belief_updated = False
        if belief_phase:
            assert samples is not None and self.belief_model is not None
            output, latency = self._belief_forward(
                transitions, samples, shared_gradient=shared_phase
            )
            dp_latency_ms += latency
            losses = []
            ce_values = []
            for index, sample in enumerate(samples):
                assert sample.label is not None
                target = torch.from_numpy(sample.label.count_onehot.copy()).to(
                    device=self.device, dtype=output.logits.dtype
                ).unsqueeze(0)
                item = belief_loss(
                    output.logits[index : index + 1],
                    target,
                    output.legal[index : index + 1],
                    lambda_count_reg=self.config.belief.lambda_count_reg,
                    lambda_entropy_reg=self.config.belief.lambda_entropy_reg,
                )
                losses.append(item.total)
                ce_values.append(item.cross_entropy)
            belief_total = torch.stack(losses).mul(normalized).sum()
            weighted = self.config.belief.lambda_belief * belief_total
            if not bool(torch.isfinite(weighted)):
                raise FloatingPointError("H4 belief loss is NaN or Inf")
            self.belief_optimizer.zero_grad(set_to_none=True)
            weighted.backward()
            belief_parameters = list(self.belief_model.parameters())
            shared_parameters = (
                list(self.model.state_encoder.parameters())
                + list(self.model.history_encoder.parameters())
                if shared_phase
                else []
            )
            belief_grad = self._gradient_norm(belief_parameters)
            shared_grad = self._gradient_norm(shared_parameters)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in self.belief_optimizer.param_groups
                    for parameter in group["params"]
                ],
                self.config.belief.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.belief_optimizer.step()
            belief_updated = True
            labels_consumed = len(samples)
            cross_entropy = math.fsum(
                value * weight
                for value, weight in zip(ce_values, normalized.tolist())
            )

            targets = np.stack([sample.label.allocation for sample in samples])
            decoded = self.belief_model.decode_map(output)
            matches = decoded == targets
            rank_accuracy = float(matches.mean())
            exact_match = float(matches.all(axis=1).mean())
            probabilities = output.constrained_probs
            predictions = probabilities.argmax(axis=-1)
            confidence = probabilities.max(axis=-1)
            correctness = (predictions == targets).astype(np.float64)
            calibration = float(np.abs(confidence - correctness).mean())
            expected = output.expected_counts.sum(axis=-1)
            map_totals = decoded.sum(axis=-1)
            totals = output.opponent_a_total.astype(np.float64)
            complement = np.stack([
                sample.belief_input.unseen_counts for sample in samples
            ]) - decoded
            complement_totals = complement.sum(axis=-1)
            opponent_b = np.array([
                sample.belief_input.opponent_b_total for sample in samples
            ])
            conservation_error = float(max(
                np.abs(expected - totals).max(),
                np.abs(map_totals - totals).max(),
                np.abs(complement_totals - opponent_b).max(),
                max(0, int(-complement.min())),
            ))
            if conservation_error > 2e-4:
                raise RuntimeError("H4 belief posterior violated exact conservation")

        consumed_update = belief_updated or bool(
            base_metrics
            and (base_metrics.public_updated or base_metrics.oracle_updated)
        )
        if not consumed_update:
            return self._metrics_noop(phase, base=base_metrics)

        count = len(transitions)
        self.eligible_updates += 1
        self.samples_consumed += count
        role_samples = {
            role: sum(transition.role == role for transition in transitions)
            for role in V3_HYBRID_ROLES
        }
        role_effective = {
            role: math.fsum(
                weight
                for transition, weight in zip(transitions, role_weights)
                if transition.role == role
            )
            for role in V3_HYBRID_ROLES
        }
        metrics = V3H4StepMetrics(
            eligible_update=self.eligible_updates,
            phase=phase,
            samples=count,
            policy_updated=bool(base_metrics and base_metrics.public_updated),
            belief_updated=belief_updated,
            shared_encoder_updated=bool(belief_updated and shared_phase),
            labels_consumed=labels_consumed,
            loss_belief=float(belief_total.detach().cpu().item()),
            belief_cross_entropy=cross_entropy,
            rank_accuracy=rank_accuracy,
            map_exact_match=exact_match,
            posterior_calibration_error=calibration,
            conservation_max_error=conservation_error,
            belief_gradient_norm=belief_grad,
            shared_gradient_norm=shared_grad,
            dp_latency_ms=dp_latency_ms,
            role_samples=role_samples,
            role_effective_weights=role_effective,
            base=base_metrics,
        )
        self.statistics.update(metrics)
        return metrics

    def _metrics_noop(
        self, phase: str, *, base: V3H3StepMetrics | None = None
    ) -> V3H4StepMetrics:
        return V3H4StepMetrics(
            eligible_update=self.eligible_updates,
            phase=phase,
            samples=0,
            policy_updated=False,
            belief_updated=False,
            shared_encoder_updated=False,
            labels_consumed=0,
            loss_belief=0.0,
            belief_cross_entropy=0.0,
            rank_accuracy=None,
            map_exact_match=None,
            posterior_calibration_error=None,
            conservation_max_error=None,
            belief_gradient_norm=0.0,
            shared_gradient_norm=0.0,
            dp_latency_ms=0.0,
            role_samples={role: 0 for role in V3_HYBRID_ROLES},
            role_effective_weights={role: 0.0 for role in V3_HYBRID_ROLES},
            base=base,
        )

    def _inner_bundle(self) -> dict[str, object]:
        descriptor, name = tempfile.mkstemp(suffix=".h3.pt")
        os.close(descriptor)
        path = Path(name)
        try:
            self.base.save_checkpoint(path)
            return torch.load(path, map_location="cpu", weights_only=True)
        finally:
            path.unlink(missing_ok=True)

    def _load_inner_bundle(self, bundle: Mapping[str, object]) -> None:
        descriptor, name = tempfile.mkstemp(suffix=".h3.pt")
        os.close(descriptor)
        path = Path(name)
        try:
            torch.save(dict(bundle), path)
            self.base.load_checkpoint(path)
        finally:
            path.unlink(missing_ok=True)

    def save_checkpoint(self, path: str | Path) -> None:
        source_sha = git_sha()
        if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
            raise RuntimeError("H4 checkpoints require a full source Git SHA")
        bundle = {
            "format": V3_H4_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": source_sha,
            "model_config_hash": self.model.config.stable_hash(),
            "belief_model_config_hash": (
                None
                if self.belief_model is None
                else self.belief_model.config.stable_hash()
            ),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "h3_checkpoint": self._inner_bundle(),
            "belief_state_dict": (
                None if self.belief_model is None else self.belief_model.state_dict()
            ),
            "belief_optimizer_state_dict": (
                None
                if self.belief_optimizer is None
                else self.belief_optimizer.state_dict()
            ),
            "counters": {
                "eligible_updates": self.eligible_updates,
                "samples_consumed": self.samples_consumed,
            },
            "phase": self.phase(),
            "statistics": self.statistics.state_dict(),
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
            raise CheckpointCompatibilityError(
                f"unable to safely load H4 checkpoint: {exc}"
            ) from exc
        if not isinstance(bundle, dict) or set(bundle) != _CHECKPOINT_KEYS:
            raise CheckpointCompatibilityError("H4 checkpoint envelope mismatch")
        expected = {
            "format": V3_H4_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": git_sha(),
            "model_config_hash": self.model.config.stable_hash(),
            "belief_model_config_hash": (
                None
                if self.belief_model is None
                else self.belief_model.config.stable_hash()
            ),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
        }
        for name, value in expected.items():
            if bundle[name] != value:
                raise CheckpointCompatibilityError(f"H4 checkpoint {name} mismatch")
        counters = bundle["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "eligible_updates", "samples_consumed"
        }:
            raise CheckpointCompatibilityError("H4 checkpoint counters mismatch")
        try:
            eligible = counters["eligible_updates"]
            consumed = counters["samples_consumed"]
            for name, value in (("eligible_updates", eligible), ("samples_consumed", consumed)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid H4 {name}")
            if bundle["phase"] != self.config.belief.phase_at(eligible):
                raise ValueError("H4 phase drift")
            statistics = H4CumulativeStats.from_state_dict(bundle["statistics"])
            if statistics.steps != eligible or statistics.decisions != consumed:
                raise ValueError("H4 statistics/counter drift")
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"H4 checkpoint state mismatch: {exc}"
            ) from exc
        belief_state = bundle["belief_state_dict"]
        optimizer_state = bundle["belief_optimizer_state_dict"]
        if self.belief_model is None:
            if belief_state is not None or optimizer_state is not None:
                raise CheckpointCompatibilityError(
                    "disabled H4 checkpoint contains belief state"
                )
        elif (
            not isinstance(belief_state, dict)
            or set(belief_state) != set(self.belief_model.state_dict())
            or not isinstance(optimizer_state, dict)
            or set(optimizer_state) != {"state", "param_groups"}
        ):
            raise CheckpointCompatibilityError("H4 belief state envelope mismatch")
        if self.belief_optimizer is not None:
            try:
                _validate_optimizer_param_groups(
                    optimizer_state["param_groups"],
                    self.belief_optimizer.state_dict()["param_groups"],
                )
            except CheckpointCompatibilityError as exc:
                raise CheckpointCompatibilityError(
                    f"H4 belief optimizer mismatch: {exc}"
                ) from exc
        inner = bundle["h3_checkpoint"]
        if not isinstance(inner, Mapping):
            raise CheckpointCompatibilityError("H4 nested H3 checkpoint is invalid")
        inner_counters = inner.get("counters")
        inner_statistics = inner.get("statistics")
        if not isinstance(inner_counters, Mapping) or not isinstance(
            inner_statistics, Mapping
        ):
            raise CheckpointCompatibilityError(
                "H4 nested H3 progress envelope mismatch"
            )
        nested_progress = {
            "learner_updates": statistics.base_updates,
            "samples_consumed": statistics.base_decisions,
            "policy_version": (
                self.config.base.public.initial_policy_version
                + statistics.policy_updates
            ),
            "steps": statistics.base_updates,
            "decisions": statistics.base_decisions,
            "public_updates": statistics.policy_updates,
            "oracle_updates": statistics.oracle_updates,
        }
        for name, expected_value in nested_progress.items():
            source = inner_counters if name in {
                "learner_updates", "samples_consumed", "policy_version"
            } else inner_statistics
            if source.get(name) != expected_value:
                raise CheckpointCompatibilityError(
                    f"H4 nested H3 {name} does not match H4 progress"
                )
        backup_inner = self._inner_bundle()
        backup_belief = (
            None
            if self.belief_model is None
            else copy.deepcopy(self.belief_model.state_dict())
        )
        backup_optimizer = (
            None
            if self.belief_optimizer is None
            else copy.deepcopy(self.belief_optimizer.state_dict())
        )
        try:
            self._load_inner_bundle(inner)
            if self.belief_model is not None:
                self.belief_model.load_state_dict(belief_state, strict=True)
                self.belief_optimizer.load_state_dict(optimizer_state)
        except (KeyError, TypeError, ValueError, RuntimeError, CheckpointCompatibilityError) as exc:
            self._load_inner_bundle(backup_inner)
            if self.belief_model is not None:
                self.belief_model.load_state_dict(backup_belief, strict=True)
                self.belief_optimizer.load_state_dict(backup_optimizer)
            raise CheckpointCompatibilityError(
                f"H4 checkpoint restore failed: {exc}"
            ) from exc
        self.eligible_updates = eligible
        self.samples_consumed = consumed
        self.statistics = statistics


__all__ = [
    "H4CumulativeStats",
    "V3_H4_TRAINER_CHECKPOINT_FORMAT",
    "V3_H4_TRAINING_CONTRACT",
    "V3H4BeliefSample",
    "V3H4Learner",
    "V3H4LearnerConfig",
    "V3H4StepMetrics",
    "build_v3_h4_belief_sample",
    "h4_training_identity",
]
