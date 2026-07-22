"""H5 sequential farmer cooperation learner and strict resume contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch

from douzero._version import git_sha
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import model_input_bundles_to_batch

from ..config import BELIEF_FEEDBACK_NONE
from ..h2_learner import _validate_optimizer_param_groups
from ..model import V3HybridModel
from ..replay import V3ReplayTransition
from .cooperation import (
    FARMER_ROLES,
    MIXER_DISABLED,
    MIXER_PRIVILEGED,
    FarmerCooperationModule,
    V3H5CooperationConfig,
    V3H5FarmerTrajectory,
    validate_farmer_pairs,
)
from .h4_learner import (
    V3H4BeliefSample,
    V3H4Learner,
    V3H4LearnerConfig,
    V3H4StepMetrics,
)

V3_H5_TRAINER_CHECKPOINT_FORMAT = "v3-hybrid-h5-farmer-cooperation-trainer-v1"
V3_H5_TRAINING_CONTRACT = "sequential-farmer-team-credit-training-sidecar-v1"

_CHECKPOINT_KEYS = frozenset({
    "format",
    "artifact_access",
    "source_git_sha",
    "model_config_hash",
    "ruleset_identity",
    "learner_config",
    "learner_config_hash",
    "training_identity",
    "training_identity_hash",
    "h4_checkpoint",
    "cooperation_state_dict",
    "cooperation_optimizer_state_dict",
    "counters",
    "statistics",
})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H5LearnerConfig:
    base: V3H4LearnerConfig = field(default_factory=V3H4LearnerConfig)
    cooperation: V3H5CooperationConfig = field(
        default_factory=V3H5CooperationConfig
    )

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.base, V3H4LearnerConfig):
            raise TypeError("H5 base must be V3H4LearnerConfig")
        if not isinstance(self.cooperation, V3H5CooperationConfig):
            raise TypeError("H5 cooperation must be V3H5CooperationConfig")
        if self.cooperation.enabled:
            if self.base.belief.enabled:
                raise ValueError("H4 joint-belief/H5 integration is deferred to H6")
            if self.base.base.schedule.enabled:
                raise ValueError("H3 Oracle/H5 integration is deferred to H6")
            if self.base.base.public.lambda_dmc <= 0.0:
                raise ValueError("enabled H5 requires ordinary public DMC updates")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "base": self.base.compatibility_dict(),
            "cooperation": self.cooperation.compatibility_dict(),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H5LearnerConfig":
        if not isinstance(payload, Mapping) or set(payload) != {"base", "cooperation"}:
            raise ValueError("H5 learner config fields mismatch")
        return cls(
            base=V3H4LearnerConfig.from_dict(payload["base"]),
            cooperation=V3H5CooperationConfig.from_dict(payload["cooperation"]),
        )


@dataclass(frozen=True)
class V3H5StepMetrics:
    eligible_update: int
    samples: int
    farmer_samples: int
    episodes: int
    cooperation_updated: bool
    schedule_weight: float
    loss_team_value: float
    loss_trajectory_consistency: float
    loss_mixer: float
    loss_cooperation: float
    cooperation_gradient_norm: float
    public_gradient_norm: float
    mixer_weight_min: float | None
    pass_samples: int
    role_samples: dict[str, int]
    role_effective_weights: dict[str, float]
    teammate_policy_pairs: tuple[tuple[str, str], ...]
    base: V3H4StepMetrics

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["base"] = self.base.as_dict()
        return result


class H5CumulativeStats:
    _FIELDS = (
        "steps",
        "samples",
        "farmer_samples",
        "episodes",
        "cooperation_updates",
        "team_value_loss_sum",
        "trajectory_loss_sum",
        "mixer_loss_sum",
        "gradient_norm_sum",
    )

    def __init__(self) -> None:
        self.steps = 0
        self.samples = 0
        self.farmer_samples = 0
        self.episodes = 0
        self.cooperation_updates = 0
        self.team_value_loss_sum = 0.0
        self.trajectory_loss_sum = 0.0
        self.mixer_loss_sum = 0.0
        self.gradient_norm_sum = 0.0

    def update(self, metrics: V3H5StepMetrics) -> None:
        self.steps += 1
        self.samples += metrics.samples
        self.farmer_samples += metrics.farmer_samples
        self.episodes += metrics.episodes
        self.cooperation_updates += int(metrics.cooperation_updated)
        self.team_value_loss_sum += metrics.loss_team_value
        self.trajectory_loss_sum += metrics.loss_trajectory_consistency
        self.mixer_loss_sum += metrics.loss_mixer
        self.gradient_norm_sum += metrics.cooperation_gradient_norm

    def state_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "H5CumulativeStats":
        if not isinstance(payload, Mapping) or set(payload) != set(cls._FIELDS):
            raise ValueError("H5 statistics fields mismatch")
        result = cls()
        integer = {
            "steps", "samples", "farmer_samples", "episodes",
            "cooperation_updates",
        }
        for name in cls._FIELDS:
            value = payload[name]
            if name in integer:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid H5 statistic {name}")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"invalid H5 statistic {name}")
            setattr(result, name, value)
        if result.cooperation_updates > result.steps:
            raise ValueError("H5 cooperation updates exceed learner steps")
        return result


def h5_training_identity(
    model: V3HybridModel,
    ruleset: RuleSet,
    config: V3H5LearnerConfig,
) -> dict[str, object]:
    return {
        "identity_version": 1,
        "training_contract": V3_H5_TRAINING_CONTRACT,
        "model_config_hash": model.config.stable_hash(),
        "ruleset": ruleset.identity(),
        "learner": config.compatibility_dict(),
        "local_q": "independent_role_specific_dmc_head_v1",
        "team_value": "training_sidecar_farmer_role_heads_v1",
        "trajectory": "ordered_unequal_public_sequence_gru_v1",
        "mixer_export": "forbidden_from_public_artifacts_v1",
        "replay_protocol": "h2_public_rows_plus_h5_public_side_channel_v1",
        "topology": "single_process_h5_reference_v1",
    }


class V3H5Learner:
    """Run H4-compatible public learning plus an opt-in sequential sidecar."""

    def __init__(
        self,
        model: V3HybridModel,
        *,
        ruleset: RuleSet,
        config: V3H5LearnerConfig | None = None,
    ) -> None:
        if not isinstance(model, V3HybridModel):
            raise TypeError("H5 learner requires V3HybridModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("H5 learner requires RuleSet")
        self.config = config or V3H5LearnerConfig()
        if self.config.cooperation.enabled and model.config.belief_feedback != BELIEF_FEEDBACK_NONE:
            raise ValueError("H5 policy belief feedback integration is deferred to H6")
        self.base = V3H4Learner(model, ruleset=ruleset, config=self.config.base)
        self.model = self.base.model
        self.ruleset = ruleset
        self.device = self.base.device
        self.cooperation = None
        self.cooperation_optimizer = None
        if self.config.cooperation.enabled:
            self.cooperation = FarmerCooperationModule(
                model.config.hidden_size, self.config.cooperation
            ).to(self.device)
            public = self.config.base.base.public
            self.cooperation_optimizer = torch.optim.RMSprop(
                self.cooperation.parameters(),
                lr=self.config.cooperation.learning_rate,
                alpha=public.rmsprop_alpha,
                momentum=public.rmsprop_momentum,
                eps=public.rmsprop_epsilon,
            )
        self.eligible_updates = 0
        self.samples_consumed = 0
        self.statistics = H5CumulativeStats()
        self.compatibility_identity = h5_training_identity(
            self.model, ruleset, self.config
        )
        self.compatibility_hash = _canonical_hash(self.compatibility_identity)

    @property
    def _public_optimizer(self):
        return self.base.base.student_optimizer

    @staticmethod
    def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        values = [
            parameter.grad.detach().float().norm(2).square()
            for parameter in parameters
            if parameter.grad is not None
        ]
        return 0.0 if not values else float(torch.stack(values).sum().sqrt().item())

    def train_batch(
        self,
        transitions: Sequence[V3ReplayTransition],
        *,
        trajectories: Sequence[V3H5FarmerTrajectory] | None = None,
        belief_samples: Sequence[V3H4BeliefSample] | None = None,
        privileged_mixer_state: torch.Tensor | None = None,
    ) -> V3H5StepMetrics:
        if not transitions or len(transitions) > self.config.base.base.public.batch_size:
            raise ValueError("H5 requires a non-empty batch within configured size")
        cfg = self.config.cooperation
        if not cfg.enabled:
            if trajectories is not None or privileged_mixer_state is not None:
                raise ValueError("disabled H5 rejects cooperation data")
            base_metrics = self.base.train_batch(
                transitions, belief_samples=belief_samples
            )
            metrics = self._disabled_metrics(base_metrics)
            if base_metrics.samples:
                self.eligible_updates += 1
                self.samples_consumed += len(transitions)
                self.statistics.update(metrics)
            return metrics
        if belief_samples is not None:
            raise ValueError("enabled H5 does not consume privileged belief labels")
        if trajectories is None:
            raise ValueError("enabled H5 requires complete farmer trajectories")
        pairs = validate_farmer_pairs(trajectories)
        ordered = tuple(item for pair in pairs for item in pair)
        farmer_rows = tuple(row for trajectory in ordered for row in trajectory.transitions)
        if len({id(row) for row in farmer_rows}) != len(farmer_rows):
            raise ValueError("H5 trajectory batch repeats a replay transition")
        transition_ids = {id(row) for row in transitions}
        if any(id(row) not in transition_ids for row in farmer_rows):
            raise ValueError("H5 trajectories are not aligned to the learner batch")
        if cfg.mixer_mode == MIXER_PRIVILEGED:
            expected_shape = (len(pairs), cfg.privileged_state_dim)
            if (
                privileged_mixer_state is None
                or privileged_mixer_state.shape != expected_shape
            ):
                raise ValueError("privileged mixer state shape mismatch")
            if not bool(torch.isfinite(privileged_mixer_state).all()):
                raise FloatingPointError("privileged mixer state contains NaN or Inf")
        elif privileged_mixer_state is not None:
            kind = "disabled" if cfg.mixer_mode == MIXER_DISABLED else "public"
            raise ValueError(f"{kind} mixer rejects privileged state")

        base_metrics = self.base.train_batch(transitions)
        schedule_weight = cfg.schedule_weight(self.eligible_updates)
        if schedule_weight == 0.0:
            metrics = self._scheduled_noop(base_metrics, ordered, pairs)
            self.eligible_updates += 1
            self.samples_consumed += len(transitions)
            self.statistics.update(metrics)
            return metrics

        chosen = [row.selected_action_index for row in farmer_rows]
        inputs = model_input_bundles_to_batch(
            [row.model_inputs for row in farmer_rows], chosen
        )
        adapted = self.model.encode_input_batch_actions(inputs)
        rows = torch.arange(len(farmer_rows), device=self.device)
        chosen_index = inputs.chosen_action_index
        chosen_embedding = adapted[rows, chosen_index]
        local_q = chosen_embedding.new_zeros((len(farmer_rows),))
        offset = 0
        for trajectory in ordered:
            count = len(trajectory.transitions)
            role_values = self.model.role_heads[trajectory.role](
                chosen_embedding[offset : offset + count]
            )["dmc_q"].squeeze(-1)
            local_q[offset : offset + count] = role_values
            offset += count
        if not cfg.update_public_model:
            chosen_embedding = chosen_embedding.detach()
            local_q = local_q.detach()

        max_steps = max(len(trajectory.transitions) for trajectory in ordered)
        padded_embedding = chosen_embedding.new_zeros(
            (len(ordered), max_steps, chosen_embedding.shape[-1])
        )
        padded_features = chosen_embedding.new_zeros(
            (len(ordered), max_steps, ordered[0].public_features.shape[-1])
        )
        padded_q = chosen_embedding.new_zeros((len(ordered), max_steps))
        mask = torch.zeros(
            (len(ordered), max_steps), device=self.device, dtype=torch.bool
        )
        role_index = torch.tensor(
            [0, 1] * len(pairs), device=self.device, dtype=torch.long
        )
        targets = chosen_embedding.new_tensor(
            [trajectory.team_return for trajectory in ordered]
        )
        offset = 0
        for index, trajectory in enumerate(ordered):
            count = len(trajectory.transitions)
            padded_embedding[index, :count] = chosen_embedding[offset : offset + count]
            padded_features[index, :count] = trajectory.public_features.to(
                device=self.device, dtype=chosen_embedding.dtype
            )
            padded_q[index, :count] = local_q[offset : offset + count]
            mask[index, :count] = True
            offset += count

        output = self.cooperation(
            padded_embedding,
            padded_features,
            padded_q,
            mask,
            role_index,
            privileged_mixer_state=(
                None
                if privileged_mixer_state is None
                else privileged_mixer_state.to(
                    device=self.device, dtype=chosen_embedding.dtype
                )
            ),
        )
        role_weights = chosen_embedding.new_tensor([
            self.config.base.base.public.role_weights[trajectory.role]
            for trajectory in ordered
        ])
        step_weights = role_weights.unsqueeze(1) * mask.to(role_weights.dtype)
        effective = step_weights.sum()
        if not bool(effective > 0):
            raise ValueError("H5 farmer role weights exclude every real sample")
        team_error = (output.team_value - targets.unsqueeze(1)).square()
        loss_team = (team_error * step_weights).sum() / effective
        trajectory_weight = role_weights / role_weights.sum()
        trajectory_target = (
            (output.trajectory_value - targets).square() * trajectory_weight
        ).sum()
        agreement = (
            output.trajectory_value.reshape(-1, 2)[:, 0]
            - output.trajectory_value.reshape(-1, 2)[:, 1]
        ).square().mean()
        loss_trajectory = 0.5 * (trajectory_target + agreement)
        loss_mixer = chosen_embedding.sum() * 0.0
        if output.mixed_team_value is not None:
            episode_targets = targets.reshape(-1, 2)[:, 0]
            loss_mixer = (output.mixed_team_value - episode_targets).square().mean()
        total = schedule_weight * (
            cfg.lambda_team_value * loss_team
            + cfg.lambda_trajectory_consistency * loss_trajectory
            + cfg.lambda_mixer * loss_mixer
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("H5 cooperation loss is NaN or Inf")
        self.cooperation_optimizer.zero_grad(set_to_none=True)
        if cfg.update_public_model:
            self._public_optimizer.zero_grad(set_to_none=True)
        total.backward()
        cooperation_parameters = list(self.cooperation.parameters())
        public_parameters = list(self.model.parameters()) if cfg.update_public_model else []
        cooperation_grad = self._gradient_norm(cooperation_parameters)
        public_grad = self._gradient_norm(public_parameters)
        torch.nn.utils.clip_grad_norm_(
            cooperation_parameters, cfg.max_grad_norm, error_if_nonfinite=True
        )
        if public_parameters:
            torch.nn.utils.clip_grad_norm_(
                public_parameters, cfg.max_grad_norm, error_if_nonfinite=True
            )
        self.cooperation_optimizer.step()
        if public_parameters:
            self._public_optimizer.step()

        role_samples = {
            role: sum(
                len(trajectory.transitions)
                for trajectory in ordered
                if trajectory.role == role
            )
            for role in FARMER_ROLES
        }
        role_effective = {
            role: role_samples[role]
            * float(self.config.base.base.public.role_weights[role])
            for role in FARMER_ROLES
        }
        metrics = V3H5StepMetrics(
            eligible_update=self.eligible_updates,
            samples=len(transitions),
            farmer_samples=len(farmer_rows),
            episodes=len(pairs),
            cooperation_updated=True,
            schedule_weight=schedule_weight,
            loss_team_value=float(loss_team.detach().cpu()),
            loss_trajectory_consistency=float(loss_trajectory.detach().cpu()),
            loss_mixer=float(loss_mixer.detach().cpu()),
            loss_cooperation=float(total.detach().cpu()),
            cooperation_gradient_norm=cooperation_grad,
            public_gradient_norm=public_grad,
            mixer_weight_min=(
                None
                if output.mixer_weights is None
                else float(output.mixer_weights.detach().min().cpu())
            ),
            pass_samples=sum(
                int(trajectory.selected_action_is_pass.sum().item())
                for trajectory in ordered
            ),
            role_samples=role_samples,
            role_effective_weights=role_effective,
            teammate_policy_pairs=tuple(
                (up.policy_id, down.policy_id) for up, down in pairs
            ),
            base=base_metrics,
        )
        self.eligible_updates += 1
        self.samples_consumed += len(transitions)
        self.statistics.update(metrics)
        return metrics

    def _disabled_metrics(self, base: V3H4StepMetrics) -> V3H5StepMetrics:
        return V3H5StepMetrics(
            eligible_update=self.eligible_updates,
            samples=base.samples,
            farmer_samples=0,
            episodes=0,
            cooperation_updated=False,
            schedule_weight=0.0,
            loss_team_value=0.0,
            loss_trajectory_consistency=0.0,
            loss_mixer=0.0,
            loss_cooperation=0.0,
            cooperation_gradient_norm=0.0,
            public_gradient_norm=0.0,
            mixer_weight_min=None,
            pass_samples=0,
            role_samples={role: 0 for role in FARMER_ROLES},
            role_effective_weights={role: 0.0 for role in FARMER_ROLES},
            teammate_policy_pairs=(),
            base=base,
        )

    def _scheduled_noop(self, base, ordered, pairs) -> V3H5StepMetrics:
        role_samples = {
            role: sum(
                len(item.transitions) for item in ordered if item.role == role
            )
            for role in FARMER_ROLES
        }
        return V3H5StepMetrics(
            eligible_update=self.eligible_updates,
            samples=base.samples,
            farmer_samples=sum(role_samples.values()),
            episodes=len(pairs),
            cooperation_updated=False,
            schedule_weight=0.0,
            loss_team_value=0.0,
            loss_trajectory_consistency=0.0,
            loss_mixer=0.0,
            loss_cooperation=0.0,
            cooperation_gradient_norm=0.0,
            public_gradient_norm=0.0,
            mixer_weight_min=None,
            pass_samples=sum(
                int(item.selected_action_is_pass.sum().item()) for item in ordered
            ),
            role_samples=role_samples,
            role_effective_weights={
                role: role_samples[role]
                * float(self.config.base.base.public.role_weights[role])
                for role in FARMER_ROLES
            },
            teammate_policy_pairs=tuple(
                (up.policy_id, down.policy_id) for up, down in pairs
            ),
            base=base,
        )

    def _inner_bundle(self) -> dict[str, object]:
        descriptor, name = tempfile.mkstemp(suffix=".h4.pt")
        os.close(descriptor)
        path = Path(name)
        try:
            self.base.save_checkpoint(path)
            return torch.load(path, map_location="cpu", weights_only=True)
        finally:
            path.unlink(missing_ok=True)

    def _load_inner_bundle(self, bundle: Mapping[str, object]) -> None:
        descriptor, name = tempfile.mkstemp(suffix=".h4.pt")
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
            raise RuntimeError("H5 checkpoints require a full source Git SHA")
        bundle = {
            "format": V3_H5_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "training_only",
            "source_git_sha": source_sha,
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "h4_checkpoint": self._inner_bundle(),
            "cooperation_state_dict": (
                None if self.cooperation is None else self.cooperation.state_dict()
            ),
            "cooperation_optimizer_state_dict": (
                None
                if self.cooperation_optimizer is None
                else self.cooperation_optimizer.state_dict()
            ),
            "counters": {
                "eligible_updates": self.eligible_updates,
                "samples_consumed": self.samples_consumed,
            },
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
                f"unable to safely load H5 checkpoint: {exc}"
            ) from exc
        if not isinstance(bundle, dict) or set(bundle) != _CHECKPOINT_KEYS:
            raise CheckpointCompatibilityError("H5 checkpoint envelope mismatch")
        expected = {
            "format": V3_H5_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "training_only",
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": asdict(self.config),
            "learner_config_hash": self.config.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
        }
        for name, value in expected.items():
            if bundle[name] != value:
                raise CheckpointCompatibilityError(f"H5 checkpoint {name} mismatch")
        counters = bundle["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "eligible_updates", "samples_consumed"
        }:
            raise CheckpointCompatibilityError("H5 checkpoint counters mismatch")
        eligible = counters["eligible_updates"]
        consumed = counters["samples_consumed"]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (eligible, consumed)
        ):
            raise CheckpointCompatibilityError("H5 checkpoint counters are invalid")
        try:
            statistics = H5CumulativeStats.from_state_dict(bundle["statistics"])
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"H5 checkpoint statistics mismatch: {exc}"
            ) from exc
        if statistics.steps != eligible or statistics.samples != consumed:
            raise CheckpointCompatibilityError("H5 checkpoint progress mismatch")
        state = bundle["cooperation_state_dict"]
        optimizer = bundle["cooperation_optimizer_state_dict"]
        if self.cooperation is None:
            if state is not None or optimizer is not None:
                raise CheckpointCompatibilityError("disabled H5 contains sidecar state")
        else:
            if not isinstance(state, dict) or set(state) != set(self.cooperation.state_dict()):
                raise CheckpointCompatibilityError("H5 cooperation state mismatch")
            if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
                raise CheckpointCompatibilityError("H5 optimizer envelope mismatch")
            _validate_optimizer_param_groups(
                optimizer["param_groups"],
                self.cooperation_optimizer.state_dict()["param_groups"],
            )
        inner = bundle["h4_checkpoint"]
        if not isinstance(inner, Mapping):
            raise CheckpointCompatibilityError("H5 nested H4 checkpoint is invalid")
        backup_inner = self._inner_bundle()
        backup_state = None if self.cooperation is None else copy.deepcopy(
            self.cooperation.state_dict()
        )
        backup_optimizer = None if self.cooperation_optimizer is None else copy.deepcopy(
            self.cooperation_optimizer.state_dict()
        )
        try:
            self._load_inner_bundle(inner)
            if self.base.eligible_updates != eligible or self.base.samples_consumed != consumed:
                raise ValueError("nested H4 progress does not match H5")
            if self.cooperation is not None:
                self.cooperation.load_state_dict(state, strict=True)
                self.cooperation_optimizer.load_state_dict(optimizer)
        except (KeyError, TypeError, ValueError, RuntimeError, CheckpointCompatibilityError) as exc:
            self._load_inner_bundle(backup_inner)
            if self.cooperation is not None:
                self.cooperation.load_state_dict(backup_state, strict=True)
                self.cooperation_optimizer.load_state_dict(backup_optimizer)
            raise CheckpointCompatibilityError(
                f"H5 checkpoint restore failed: {exc}"
            ) from exc
        self.eligible_updates = eligible
        self.samples_consumed = consumed
        self.statistics = statistics


__all__ = [
    "H5CumulativeStats",
    "V3_H5_TRAINER_CHECKPOINT_FORMAT",
    "V3_H5_TRAINING_CONTRACT",
    "V3H5Learner",
    "V3H5LearnerConfig",
    "V3H5StepMetrics",
    "h5_training_identity",
]
