"""H6 single-process integration over the H1-H5 component learners."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from douzero._version import git_sha
from douzero.belief.model import BeliefModel
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.human_data.sample import BCSample
from douzero.models_v2.batch import (
    bidding_observations_to_model_input,
    model_input_bundles_to_batch,
)
from douzero.training.bc_loss import listwise_bc_loss
from douzero.training.bidding import BiddingMinibatch
from douzero.training.losses import resolve_score_target

from ..config import BELIEF_FEEDBACK_NONE
from ..integration_config import V3H6ResolvedConfig
from ..loss_composer import (
    LOSS_NAMES,
    LossComposition,
    LossTermTensor,
    V3HybridLossComposer,
)
from ..model import V3_HYBRID_ROLES, V3HybridModel
from ..replay import V3ReplayTransition
from .belief_config import (
    BELIEF_PHASE_AUXILIARY,
    BELIEF_PHASE_DISABLED,
    BELIEF_PHASE_POLICY,
)
from .h4_learner import V3H4BeliefSample
from .h5_learner import V3H5Learner, V3H5StepMetrics
from .cooperation import V3H5FarmerTrajectory

V3_H6_TRAINER_CHECKPOINT_FORMAT = "v3-hybrid-h6-trainer-v1"
V3_H6_TRAINING_CONTRACT = "atomic-sequential-component-loss-integration-v1"

_CHECKPOINT_KEYS = frozenset({
    "format",
    "artifact_access",
    "source_git_sha",
    "model_config",
    "model_config_hash",
    "ruleset_identity",
    "learner_config",
    "learner_config_hash",
    "training_identity",
    "training_identity_hash",
    "h5_checkpoint",
    "loss_composer",
    "counters",
    "statistics",
})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H6StepMetrics:
    eligible_update: int
    samples: int
    public_aux_updated: bool
    public_aux_gradient_norm: float
    policy_version: int
    loss_total: float
    losses: dict[str, dict[str, object]]
    base: V3H5StepMetrics

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base"] = self.base.as_dict()
        return payload


class H6CumulativeStats:
    _FIELDS = (
        "steps",
        "samples",
        "public_aux_updates",
        "public_aux_loss_sum",
        "public_aux_gradient_norm_sum",
    )

    def __init__(self) -> None:
        self.steps = 0
        self.samples = 0
        self.public_aux_updates = 0
        self.public_aux_loss_sum = 0.0
        self.public_aux_gradient_norm_sum = 0.0

    def update(self, metrics: V3H6StepMetrics) -> None:
        self.steps += 1
        self.samples += metrics.samples
        self.public_aux_updates += int(metrics.public_aux_updated)
        self.public_aux_loss_sum += metrics.loss_total
        self.public_aux_gradient_norm_sum += metrics.public_aux_gradient_norm

    def state_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "H6CumulativeStats":
        if not isinstance(payload, Mapping) or set(payload) != set(cls._FIELDS):
            raise ValueError("H6 statistics fields mismatch")
        result = cls()
        for name in cls._FIELDS:
            value = payload[name]
            if name in {"steps", "samples", "public_aux_updates"}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid H6 statistic {name}")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"invalid H6 statistic {name}")
            setattr(result, name, value)
        if result.public_aux_updates > result.steps:
            raise ValueError("H6 public auxiliary updates exceed learner steps")
        return result


def h6_training_identity(
    model: V3HybridModel,
    ruleset: RuleSet,
    config: V3H6ResolvedConfig,
) -> dict[str, object]:
    return {
        "identity_version": 1,
        "training_contract": V3_H6_TRAINING_CONTRACT,
        "model_config_hash": model.config.stable_hash(),
        "ruleset": ruleset.identity(),
        "resolved_config_hash": config.stable_hash(),
        "learner": config.learner.compatibility_dict(),
        "loss_execution_owners": {
            "dmc": "h3_public_phase",
            "oracle": "h3_oracle_phase",
            "belief": "h4_belief_phase",
            "cooperation": "h5_cooperation_phase",
            "win": "h6_public_aux_phase",
            "score": "h6_public_aux_phase",
            "bc": "h6_public_aux_phase",
            "strategy": "h6_public_aux_phase",
            "bidding": "h6_public_aux_phase",
        },
        "failure_atomicity": "outer_strict_checkpoint_rollback_before_commit_v1",
        "public_policy_version": "h3_plus_h5_plus_h6_public_updates_v1",
        "privileged_serialization": "separate_training_sidecars_never_public_replay_v1",
    }


def _sample_ids(transitions: Sequence[V3ReplayTransition], prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}:{row.deal_id}:{row.episode_id}:{index}:{row.selected_action_index}"
        for index, row in enumerate(transitions)
    )


class V3H6Learner:
    """Integrate stable H1-H5 phases plus public H6 auxiliary objectives."""

    def __init__(
        self,
        model: V3HybridModel,
        *,
        ruleset: RuleSet,
        config: V3H6ResolvedConfig,
        belief_model: BeliefModel | None = None,
    ) -> None:
        if not isinstance(config, V3H6ResolvedConfig):
            raise TypeError("H6 learner requires a resolved H6 config")
        # This performs the support-matrix check before constructing CUDA
        # modules, optimizers, replay, or checkpoint readers.
        config.validate_startup()
        if not isinstance(model, V3HybridModel):
            raise TypeError("H6 learner requires V3HybridModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("H6 learner requires RuleSet")
        if model.config != config.model:
            raise ValueError("H6 model graph differs from the resolved config")
        if ruleset.ruleset_id != config.learner.topology.ruleset:
            raise ValueError("H6 ruleset differs from the support-matrix request")
        self.config = config
        self.base = V3H5Learner(
            model,
            ruleset=ruleset,
            config=config.learner.base,
            belief_model=belief_model,
        )
        self.model = self.base.model
        self.ruleset = ruleset
        self.device = self.base.device
        self.composer = V3HybridLossComposer(config.learner.losses)
        self.eligible_updates = 0
        self.samples_consumed = 0
        self.public_aux_updates = 0
        self.statistics = H6CumulativeStats()
        self.base._bind_h6_policy_version_owner(self)
        self.compatibility_identity = h6_training_identity(
            self.model, ruleset, config
        )
        self.compatibility_hash = _canonical_hash(self.compatibility_identity)

    @property
    def policy_version(self) -> int:
        return self.base.policy_version

    def _external_term(
        self,
        name: str,
        transitions: Sequence[V3ReplayTransition],
        raw_loss: float,
        *,
        valid: bool,
        schedule_override: float | None = None,
    ) -> LossTermTensor:
        parameter = next(self.model.parameters())
        values = parameter.new_full((len(transitions),), float(raw_loss))
        mask = torch.full(
            (len(transitions),), valid, device=parameter.device, dtype=torch.bool
        )
        return LossTermTensor(
            values=values,
            valid_mask=mask,
            roles=tuple(row.role for row in transitions),
            sample_ids=_sample_ids(transitions, name),
            gradient_owner="external",
            schedule_override=schedule_override,
        )

    def _base_terms(
        self,
        transitions: Sequence[V3ReplayTransition],
        metrics: V3H5StepMetrics,
    ) -> dict[str, LossTermTensor]:
        terms: dict[str, LossTermTensor] = {}
        h4 = metrics.base
        h3 = h4.base
        losses = self.config.learner.losses
        if losses.lambda_dmc > 0.0:
            terms["dmc"] = self._external_term(
                "dmc",
                transitions,
                0.0 if h3 is None else h3.loss_dmc,
                valid=bool(h3 and h3.public_updated),
            )
        if losses.lambda_oracle > 0.0:
            scheduled_oracle = 0.0
            valid = False
            if h3 is not None and h3.oracle_updated:
                scheduled_oracle = max(
                    0.0,
                    h3.loss_total
                    - self.config.learner.losses.lambda_dmc * h3.loss_dmc,
                )
                valid = True
            terms["oracle"] = self._external_term(
                "oracle", transitions, scheduled_oracle, valid=valid
            )
        if losses.lambda_belief > 0.0:
            terms["belief"] = self._external_term(
                "belief",
                transitions,
                h4.loss_belief,
                valid=h4.belief_updated,
            )
        if losses.lambda_coop > 0.0:
            cfg = self.config.learner.base.cooperation
            raw = (
                cfg.lambda_team_value * metrics.loss_team_value
                + cfg.lambda_trajectory_consistency
                * metrics.loss_trajectory_consistency
                + cfg.lambda_mixer * metrics.loss_mixer
            )
            multiplier = (
                0.0
                if cfg.lambda_coop == 0.0
                else metrics.schedule_weight / cfg.lambda_coop
            )
            terms["cooperation"] = self._external_term(
                "cooperation",
                transitions,
                raw,
                valid=metrics.cooperation_updated,
                schedule_override=multiplier,
            )
        return terms

    def _public_value_terms(
        self,
        transitions: Sequence[V3ReplayTransition],
        *,
        belief_samples: Sequence[V3H4BeliefSample] | None,
        strategy_targets: Sequence[Mapping[str, object]] | None,
        enabled: bool,
    ) -> tuple[dict[str, LossTermTensor], object | None]:
        losses = self.config.learner.losses
        need_forward = enabled and any((
            losses.lambda_win > 0.0,
            losses.lambda_score > 0.0,
            losses.lambda_strategy > 0.0,
        ))
        if not need_forward or not transitions:
            empty = {}
            for name in ("win", "score", "strategy"):
                if losses.weight(name) > 0.0:
                    empty[name] = self._external_term(
                        name, transitions, 0.0, valid=False
                    )
            return empty, None
        inputs = model_input_bundles_to_batch(
            [row.model_inputs for row in transitions],
            [row.selected_action_index for row in transitions],
        )
        belief_features = None
        if self.model.config.belief_feedback != BELIEF_FEEDBACK_NONE:
            if belief_samples is None:
                raise ValueError("H6 public auxiliaries require aligned public belief samples")
            belief_features, _latency = self.base.base._policy_features(
                transitions, belief_samples
            )
        output = self.model.forward_input_batch(
            inputs, belief_features=belief_features
        )
        gathered = output.gather_chosen(inputs.chosen_action_index)
        roles = tuple(row.role for row in transitions)
        ids = _sample_ids(transitions, "public")
        targets = gathered["win_logit"].new_tensor([
            row.mc_return for row in transitions
        ])
        target_win = (targets > 0.0).to(targets.dtype)
        terms: dict[str, LossTermTensor] = {}
        mask = torch.ones(len(transitions), device=self.device, dtype=torch.bool)
        if losses.lambda_win > 0.0:
            values = F.binary_cross_entropy_with_logits(
                gathered["win_logit"].squeeze(-1), target_win, reduction="none"
            )
            terms["win"] = LossTermTensor(values, mask, roles, ids)
        if losses.lambda_score > 0.0:
            score_target = resolve_score_target(
                targets,
                score_target_transform=self.config.learner.auxiliary.score_target_transform,
                score_clamp=self.model.config.score_clamp,
            )
            prediction = torch.where(
                target_win.bool(),
                gathered["score_if_win"].squeeze(-1),
                gathered["score_if_loss"].squeeze(-1),
            )
            values = F.huber_loss(
                prediction,
                score_target,
                delta=self.config.learner.auxiliary.score_delta,
                reduction="none",
            )
            terms["score"] = LossTermTensor(values, mask, roles, ids)
        if losses.lambda_strategy > 0.0:
            terms["strategy"] = self._strategy_term(
                gathered, transitions, roles, ids, strategy_targets
            )
        return terms, output

    def _strategy_term(
        self,
        gathered: Mapping[str, torch.Tensor],
        transitions: Sequence[V3ReplayTransition],
        roles: Sequence[str],
        ids: Sequence[str],
        strategy_targets: Sequence[Mapping[str, object]] | None,
    ) -> LossTermTensor:
        target_names = (
            "min_turns_after",
            "min_turns_exact_mask",
            "regain_initiative",
            "teammate_finish",
            "teammate_finish_mask",
            "spring_probability",
            "structure_cost",
        )
        if strategy_targets is None or len(strategy_targets) != len(transitions):
            raise ValueError("H6 strategy training requires aligned trajectory labels")
        targets = []
        for labels in strategy_targets:
            if not isinstance(labels, Mapping) or set(labels) != set(target_names):
                raise ValueError(
                    "H6 strategy training requires aligned trajectory labels"
                )
            for name in target_names:
                value = labels[name]
                if not isinstance(value, numbers.Real) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError(
                        f"H6 strategy label {name} must be finite"
                    )
            for name in (
                "min_turns_exact_mask",
                "regain_initiative",
                "teammate_finish",
                "teammate_finish_mask",
                "spring_probability",
            ):
                if float(labels[name]) not in (0.0, 1.0):
                    raise ValueError(f"H6 strategy label {name} must be binary")
            for name in ("min_turns_after", "structure_cost"):
                if float(labels[name]) < 0.0:
                    raise ValueError(
                        f"H6 strategy label {name} must be non-negative"
                    )
            targets.append(labels)
        device = gathered["min_turns_after"].device
        dtype = gathered["min_turns_after"].dtype
        stacked = {
            name: torch.tensor(
                [labels[name] for labels in targets], device=device,
                dtype=torch.bool if name.endswith("mask") else dtype,
            )
            for name in target_names
        }
        component_values = {
            "min_turns": F.huber_loss(
                gathered["min_turns_after"].squeeze(-1),
                stacked["min_turns_after"], reduction="none",
            ),
            "regain": F.binary_cross_entropy_with_logits(
                gathered["regain_initiative_logit"].squeeze(-1),
                stacked["regain_initiative"], reduction="none",
            ),
            "teammate": F.binary_cross_entropy_with_logits(
                gathered["teammate_finish_logit"].squeeze(-1),
                stacked["teammate_finish"], reduction="none",
            ),
            "spring": F.binary_cross_entropy_with_logits(
                gathered["spring_probability_logit"].squeeze(-1),
                stacked["spring_probability"], reduction="none",
            ),
            "structure": F.huber_loss(
                gathered["structure_cost"].squeeze(-1),
                stacked["structure_cost"], reduction="none",
            ),
        }
        masks = {
            "min_turns": stacked["min_turns_exact_mask"],
            "regain": torch.ones_like(stacked["min_turns_exact_mask"]),
            "teammate": stacked["teammate_finish_mask"],
            "spring": torch.ones_like(stacked["min_turns_exact_mask"]),
            "structure": torch.ones_like(stacked["min_turns_exact_mask"]),
        }
        aux = self.config.learner.auxiliary
        component_weights = {
            "min_turns": aux.strategy_lambda_min_turns,
            "regain": aux.strategy_lambda_regain_initiative,
            "teammate": aux.strategy_lambda_teammate_finish,
            "spring": aux.strategy_lambda_spring,
            "structure": aux.strategy_lambda_structure,
        }
        role_weight = component_values["min_turns"].new_tensor([
            self.config.learner.losses.role_weights[role] for role in roles
        ])
        total_denominator = role_weight.sum()
        values = torch.zeros_like(component_values["min_turns"])
        for name, component in component_values.items():
            valid = masks[name]
            denominator = (role_weight * valid.to(role_weight.dtype)).sum()
            if component_weights[name] == 0.0 or not bool(denominator > 0):
                continue
            values = values + (
                component_weights[name]
                * component
                * valid.to(component.dtype)
                * total_denominator
                / denominator
            )
        return LossTermTensor(
            values,
            torch.ones_like(masks["min_turns"]),
            roles,
            ids,
        )

    def _bc_term(self, samples: Sequence[BCSample] | None) -> LossTermTensor:
        parameter = next(self.model.parameters())
        if not samples:
            return LossTermTensor(
                parameter.new_empty((0,)),
                torch.empty((0,), device=self.device, dtype=torch.bool),
                (),
                (),
            )
        terms = []
        roles = []
        ids = []
        for index, sample in enumerate(samples):
            if not isinstance(sample, BCSample):
                raise TypeError("H6 BC input must reuse BCSample")
            sample.validate()
            output = self.model.forward_observation(sample.obs)
            if output.prior_logit is None:
                raise RuntimeError("H6 BC prior head is absent")
            loss, _correct = listwise_bc_loss(
                output.prior_logit,
                output.action_mask,
                sample.human_action_index,
                weight=sample.sample_weight,
                temperature=self.config.learner.auxiliary.bc_temperature,
                label_smoothing=self.config.learner.auxiliary.bc_label_smoothing,
            )
            terms.append(loss)
            roles.append(sample.position)
            ids.append(f"bc:{sample.game_id}:{index}:{sample.human_action_index}")
        values = torch.stack(terms)
        return LossTermTensor(
            values,
            torch.ones(len(terms), device=values.device, dtype=torch.bool),
            tuple(roles),
            tuple(ids),
        )

    def _bidding_term(
        self, batch: BiddingMinibatch | None
    ) -> LossTermTensor:
        parameter = next(self.model.parameters())
        if batch is None:
            return LossTermTensor(
                parameter.new_empty((0,)),
                torch.empty((0,), device=self.device, dtype=torch.bool),
                (),
                (),
            )
        if not isinstance(batch, BiddingMinibatch) or not batch.transitions:
            raise ValueError("H6 bidding input must be a non-empty BiddingMinibatch")
        inputs = bidding_observations_to_model_input([
            row.obs for row in batch.transitions
        ])
        output = self.model.forward_bidding_batched(inputs)
        targets = batch.to_targets(output.bid_logits.device, dtype=output.bid_logits.dtype)
        aux = self.config.learner.auxiliary
        count = len(batch.transitions)
        roles = tuple(row.actor_role for row in batch.transitions)
        logits = output.bid_logits.float()
        rows = torch.arange(count, device=logits.device)
        actions = targets.actions
        if not bool(((actions >= 0) & (actions < logits.shape[1])).all()):
            raise ValueError("H6 bidding action is outside the action schema")
        if not bool(output.bid_action_mask[rows, actions].all()):
            raise ValueError("H6 bidding action is not legal")
        masked_logits = logits.masked_fill(
            ~output.bid_action_mask, torch.finfo(logits.dtype).min
        )
        imitation = F.cross_entropy(masked_logits, actions, reduction="none")
        selected = logits.gather(1, actions[:, None]).squeeze(1)
        behavior = F.binary_cross_entropy_with_logits(
            selected, targets.actor_win.float(), reduction="none"
        )
        policy = torch.where(targets.imitation_mask, imitation, behavior)
        credit = targets.policy_credit_mask
        win = F.binary_cross_entropy_with_logits(
            output.landlord_win_logit.float(),
            targets.landlord_win.float(),
            reduction="none",
        )
        score_target = resolve_score_target(
            targets.landlord_score.float(),
            score_target_transform=aux.score_target_transform,
            score_clamp=self.model.config.score_clamp,
        )
        score = F.huber_loss(
            output.expected_landlord_score.float(),
            score_target,
            delta=aux.score_delta,
            reduction="none",
        )
        role_weights = logits.new_tensor([
            self.config.learner.losses.role_weights[role] for role in roles
        ])
        total_weight = role_weights.sum()
        policy_weight = (role_weights * credit.to(role_weights.dtype)).sum()
        values = (
            aux.bidding_lambda_landlord_win * win
            + aux.bidding_lambda_landlord_score * score
        )
        if aux.bidding_lambda_policy > 0.0 and bool(policy_weight > 0):
            # The composer applies role weights once. This rescaling only
            # gives the masked policy component its own valid denominator.
            values = values + (
                aux.bidding_lambda_policy
                * policy
                * credit.to(policy.dtype)
                * total_weight
                / policy_weight
            )
        ids = tuple(
            f"bid:{row.policy_version}:{row.obs.redeal_count}:"
            f"{row.obs.current_seat}:{index}:{row.bid_action}"
            for index, row in enumerate(batch.transitions)
        )
        return LossTermTensor(
            values,
            torch.ones(count, device=values.device, dtype=torch.bool),
            roles,
            ids,
        )

    def _empty_base_metrics(self) -> V3H5StepMetrics:
        h3 = self.base.base.base
        state = h3.schedule_state()
        h3_metrics = h3._empty_metrics(state, h3.policy_version)
        h4_metrics = self.base.base._metrics_noop(
            self.base.base.phase(), base=h3_metrics
        )
        return self.base._disabled_metrics(h4_metrics)

    def _rollback_snapshot(self, directory: Path) -> tuple[Path, dict[str, object]]:
        path = directory / "h5-rollback.pt"
        self.base.save_checkpoint(path)
        state = {
            "composer": copy.deepcopy(self.composer.state_dict()),
            "eligible_updates": self.eligible_updates,
            "samples_consumed": self.samples_consumed,
            "public_aux_updates": self.public_aux_updates,
            "statistics": copy.deepcopy(self.statistics.state_dict()),
        }
        return path, state

    def _restore_rollback(self, path: Path, state: Mapping[str, object]) -> None:
        self.base.load_checkpoint(path)
        self.composer.load_state_dict(state["composer"])
        self.eligible_updates = state["eligible_updates"]
        self.samples_consumed = state["samples_consumed"]
        self.public_aux_updates = state["public_aux_updates"]
        self.statistics = H6CumulativeStats.from_state_dict(state["statistics"])
        self.base._h6_policy_version_owner = self
        for parameter in self.model.parameters():
            parameter.grad = None
        self.model.train()

    def train_batch(
        self,
        transitions: Sequence[V3ReplayTransition],
        *,
        trajectories: Sequence[V3H5FarmerTrajectory] | None = None,
        belief_samples: Sequence[V3H4BeliefSample] | None = None,
        oracle_samples=None,
        privileged_mixer_state: torch.Tensor | None = None,
        bc_samples: Sequence[BCSample] | None = None,
        bidding_batch: BiddingMinibatch | None = None,
        strategy_targets: Sequence[Mapping[str, object]] | None = None,
    ) -> V3H6StepMetrics:
        auxiliary_samples = len(bc_samples or ()) + (
            0 if bidding_batch is None else len(bidding_batch.transitions)
        )
        if not transitions:
            if auxiliary_samples == 0:
                raise ValueError("H6 requires at least one valid training target")
            if any(value is not None for value in (
                trajectories,
                belief_samples,
                oracle_samples,
                privileged_mixer_state,
                strategy_targets,
            )):
                raise ValueError(
                    "H6 auxiliary-only batches reject card-play sidecars"
                )
        oracle_state = self.base.base.base.schedule_state()
        belief_phase = self.base.base.phase()
        public_phase = oracle_state.public_training and belief_phase in {
            BELIEF_PHASE_DISABLED,
            BELIEF_PHASE_AUXILIARY,
            BELIEF_PHASE_POLICY,
        }
        if not public_phase and any(
            value is not None for value in (bc_samples, bidding_batch, strategy_targets)
        ):
            raise ValueError("non-public H6 phase rejects public auxiliary data")
        if self.config.learner.losses.lambda_strategy == 0.0 and strategy_targets is not None:
            raise ValueError("disabled H6 strategy loss rejects labels")
        with tempfile.TemporaryDirectory(prefix="douzero-h6-rollback-") as temporary:
            snapshot, state = self._rollback_snapshot(Path(temporary))
            try:
                base_metrics = (
                    self.base.train_batch(
                        transitions,
                        trajectories=trajectories,
                        belief_samples=belief_samples,
                        oracle_samples=oracle_samples,
                        privileged_mixer_state=privileged_mixer_state,
                    )
                    if transitions
                    else self._empty_base_metrics()
                )
                terms = self._base_terms(transitions, base_metrics)
                public_terms, _output = self._public_value_terms(
                    transitions,
                    belief_samples=belief_samples,
                    strategy_targets=strategy_targets,
                    enabled=public_phase,
                )
                terms.update(public_terms)
                losses = self.config.learner.losses
                if losses.lambda_bc > 0.0:
                    terms["bc"] = self._bc_term(bc_samples if public_phase else None)
                elif bc_samples is not None:
                    raise ValueError("disabled H6 BC rejects samples")
                if losses.lambda_bidding > 0.0:
                    terms["bidding"] = self._bidding_term(
                        bidding_batch if public_phase else None
                    )
                elif bidding_batch is not None:
                    raise ValueError("disabled H6 bidding rejects samples")
                composition = self.composer.compose(terms)
                aux_required = composition.optimizer_step_required
                gradient_norm = self.composer.apply(
                    composition,
                    self.base._public_optimizer,
                    list(self.model.parameters()),
                    max_grad_norm=self.config.learner.base.base.base.public.max_grad_norm,
                )
                if aux_required:
                    self.public_aux_updates += 1
                total = math.fsum(
                    item.weighted_loss for item in composition.terms.values()
                )
                loss_metrics = {
                    name: asdict(composition.terms[name]) for name in LOSS_NAMES
                }
                metrics = V3H6StepMetrics(
                    eligible_update=self.eligible_updates,
                    samples=len(transitions) + auxiliary_samples,
                    public_aux_updated=aux_required,
                    public_aux_gradient_norm=gradient_norm,
                    policy_version=self.policy_version,
                    loss_total=total,
                    losses=loss_metrics,
                    base=base_metrics,
                )
                self.eligible_updates += 1
                self.samples_consumed += len(transitions) + auxiliary_samples
                self.statistics.update(metrics)
                return metrics
            except Exception:
                self._restore_rollback(snapshot, state)
                raise

    def _inner_bundle(self) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="douzero-h6-save-") as temporary:
            path = Path(temporary) / "h5.pt"
            self.base.save_checkpoint(path)
            return torch.load(path, map_location="cpu", weights_only=True)

    def save_checkpoint(self, path: str | Path) -> None:
        bundle = {
            "format": V3_H6_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": git_sha(),
            "model_config": self.model.config.to_dict(),
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": self.config.learner.compatibility_dict(),
            "learner_config_hash": self.config.learner.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
            "h5_checkpoint": self._inner_bundle(),
            "loss_composer": self.composer.state_dict(),
            "counters": {
                "eligible_updates": self.eligible_updates,
                "samples_consumed": self.samples_consumed,
                "public_aux_updates": self.public_aux_updates,
                "policy_version": self.policy_version,
            },
            "statistics": self.statistics.state_dict(),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"unable to safely load H6 checkpoint: {exc}"
            ) from exc
        if not isinstance(bundle, Mapping) or set(bundle) != _CHECKPOINT_KEYS:
            raise CheckpointCompatibilityError("H6 checkpoint envelope mismatch")
        expected = {
            "format": V3_H6_TRAINER_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "source_git_sha": git_sha(),
            "model_config": self.model.config.to_dict(),
            "model_config_hash": self.model.config.stable_hash(),
            "ruleset_identity": self.ruleset.identity(),
            "learner_config": self.config.learner.compatibility_dict(),
            "learner_config_hash": self.config.learner.stable_hash(),
            "training_identity": self.compatibility_identity,
            "training_identity_hash": self.compatibility_hash,
        }
        for name, value in expected.items():
            if bundle[name] != value:
                raise CheckpointCompatibilityError(f"H6 checkpoint {name} mismatch")
        counters = bundle["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "eligible_updates", "samples_consumed", "public_aux_updates", "policy_version"
        }:
            raise CheckpointCompatibilityError("H6 checkpoint counter fields mismatch")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters.values()
        ):
            raise CheckpointCompatibilityError("H6 checkpoint counters are invalid")
        statistics = H6CumulativeStats.from_state_dict(bundle["statistics"])
        if (
            statistics.steps != counters["eligible_updates"]
            or statistics.samples != counters["samples_consumed"]
            or statistics.public_aux_updates != counters["public_aux_updates"]
        ):
            raise CheckpointCompatibilityError("H6 checkpoint counter/statistic mismatch")
        composer = V3HybridLossComposer(self.config.learner.losses)
        composer.load_state_dict(bundle["loss_composer"])
        inner = bundle["h5_checkpoint"]
        with tempfile.TemporaryDirectory(prefix="douzero-h6-load-") as temporary:
            inner_path = Path(temporary) / "h5.pt"
            torch.save(inner, inner_path)
            self.base.load_checkpoint(inner_path)
        expected_policy = (
            self.base.base.base.policy_version
            + self.base.statistics.public_updates
            + counters["public_aux_updates"]
        )
        if expected_policy != counters["policy_version"]:
            raise CheckpointCompatibilityError("H6 checkpoint policy version mismatch")
        self.composer = composer
        self.eligible_updates = counters["eligible_updates"]
        self.samples_consumed = counters["samples_consumed"]
        self.public_aux_updates = counters["public_aux_updates"]
        self.statistics = statistics
        self.base._h6_policy_version_owner = self


__all__ = [
    "H6CumulativeStats",
    "V3_H6_TRAINER_CHECKPOINT_FORMAT",
    "V3_H6_TRAINING_CONTRACT",
    "V3H6Learner",
    "V3H6StepMetrics",
    "h6_training_identity",
]
