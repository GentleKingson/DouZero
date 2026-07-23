"""P2 low-cost pilot runner contracts and real self-play batch construction.

This module is intentionally not a second trainer.  It adapts the immutable P1
experiment identity to the H6 learner and supplies the training-only sidecars
that the existing H3-H6 learners require.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.distillation.dataset import DistillationSample, OfflineDistillationSample
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation.encode_v2 import ObservationV2, get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.observation.schema import build_v2_schema
from douzero.training.v2_buffer import Episode, Transition

from .adaptive_dmc import AdaptiveDMCConfig
from .config import V3HybridModelConfig
from .formal_config import FormalExperimentConfig
from .h2_learner import V3H2LearnerConfig
from .integration_config import (
    V3H6FeatureFlags,
    V3H6LearnerConfig,
    V3H6ResolvedConfig,
    V3H6TopologyConfig,
)
from .loss_composer import LossTermSchedule, V3HybridLossComposerConfig
from .model import V3HybridModel
from .replay import (
    AdaptiveSnapshotProvenance,
    PendingV3Transition,
    V3ReplayTransition,
    capture_plain_transition,
)
from .training.belief_config import V3H4BeliefTrainingConfig
from .training.cooperation import (
    V3H5CooperationConfig,
    V3H5FarmerDecision,
    V3H5FarmerTrajectory,
    build_h5_public_features,
)
from .training.h3_learner import V3H3LearnerConfig
from .training.h4_learner import (
    V3H4BeliefSample,
    V3H4LearnerConfig,
    build_v3_h4_belief_sample,
)
from .training.h5_learner import V3H5LearnerConfig
from .training.h6_learner import V3H6Learner
from .training.oracle_schedule import OracleGuidingScheduleConfig

P2_PILOT_SCHEMA = "v3-p2-pilot-evidence-v2"
P2_PILOT_PROTOCOL = "real-env-single-process-checkpoint-resume-v2"
P2_VARIANTS = (
    "v3_role",
    "v3_admc",
    "v3_oracle",
    "v3_belief",
    "v3_farmer_cooperation",
    "v3_full_hybrid",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pilot_resolved_config(
    formal: FormalExperimentConfig,
) -> V3H6ResolvedConfig:
    """Convert a frozen P1 V3 config to the executable H6 config exactly once."""

    if formal.variant not in P2_VARIANTS:
        raise ValueError("P2 runner only accepts the six frozen V3 pilot variants")
    if formal.ruleset["id"] != "legacy":
        raise ValueError("P2 first-pass pilot is frozen to legacy card-play")
    model = formal.model["config"]
    model_config = V3HybridModelConfig.from_dict(dict(model))
    features = formal.features
    feature_configs = formal.feature_configs
    weights = formal.losses["weights"]
    roles = formal.losses["role_weights"]
    admc = feature_configs["adaptive_dmc"]
    public = V3H2LearnerConfig(
        batch_size=formal.runtime.batch_size,
        learning_rate=1e-4,
        max_grad_norm=40.0,
        lambda_dmc=float(weights["dmc"]),
        landlord_weight=float(roles["landlord"]),
        landlord_up_weight=float(roles["landlord_up"]),
        landlord_down_weight=float(roles["landlord_down"]),
        device=formal.runtime.device,
        seed=formal.initialization.seed,
        adaptive_dmc=AdaptiveDMCConfig(
            mode=str(admc["mode"]),
            gamma_start=float(admc["gamma_start"]),
            gamma_end=float(admc["gamma_end"]),
            gamma_schedule_updates=int(admc["gamma_schedule_updates"]),
            epsilon=float(admc["epsilon"]),
            delta=float(admc["delta"]),
        ),
    )
    oracle = feature_configs["oracle"]
    schedule = OracleGuidingScheduleConfig(
        enabled=bool(oracle["enabled"]),
        warmup_updates=int(oracle["warmup_updates"]),
        guided_updates=int(oracle["guided_updates"]),
        finetune_updates=int(oracle["finetune_updates"]),
        guidance_weight_start=float(oracle["guidance_weight_start"]),
        guidance_weight_end=float(oracle["guidance_weight_end"]),
    )
    belief = feature_configs["belief"]
    belief_config = V3H4BeliefTrainingConfig(
        enabled=bool(belief["enabled"]),
        mode=str(belief["mode"]),
        lambda_belief=float(weights["belief"]),
        policy_updates_per_cycle=int(belief["policy_updates_per_cycle"]),
        belief_updates_per_cycle=int(belief["belief_updates_per_cycle"]),
        shared_updates_per_cycle=int(belief["shared_updates_per_cycle"]),
        shared_encoder_updates=int(belief["shared_updates_per_cycle"]) > 0,
    )
    cooperation = feature_configs["cooperation"]
    cooperation_config = V3H5CooperationConfig(
        enabled=bool(cooperation["enabled"]),
        lambda_coop=float(weights["cooperation"]),
        mixer_mode=(
            "disabled" if cooperation["mixer"] == "disabled"
            else str(cooperation["mixer"])
        ),
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(
            base=V3H3LearnerConfig(public=public, schedule=schedule),
            belief=belief_config,
        ),
        cooperation=cooperation_config,
    )
    schedules = {
        name: LossTermSchedule(**dict(value))
        for name, value in formal.losses["schedules"].items()
    }
    losses = V3HybridLossComposerConfig(
        lambda_dmc=float(weights["dmc"]),
        lambda_win=float(weights["win"]),
        lambda_score=float(weights["score"]),
        lambda_oracle=float(weights["oracle"]),
        lambda_belief=float(weights["belief"]),
        lambda_coop=float(weights["cooperation"]),
        lambda_bc=float(weights["bc"]),
        lambda_strategy=float(weights["strategy"]),
        lambda_bidding=float(weights["bidding"]),
        landlord_weight=float(roles["landlord"]),
        landlord_up_weight=float(roles["landlord_up"]),
        landlord_down_weight=float(roles["landlord_down"]),
        schedules=schedules,
    )
    flags = V3H6FeatureFlags(
        adaptive_dmc=bool(features["adaptive_dmc"]),
        oracle=bool(features["oracle"]),
        belief=bool(features["belief"]),
        cooperation=bool(features["cooperation"]),
        human_bc=bool(features["human_bc"]),
        strategy=bool(features["strategy"]),
        style=bool(features["style"]),
        bidding=bool(features["bidding"]),
        selective_search=bool(features["search"]),
    )
    return V3H6ResolvedConfig(
        model=model_config,
        learner=V3H6LearnerConfig(
            base=base,
            losses=losses,
            features=flags,
            topology=V3H6TopologyConfig(ruleset="legacy"),
        ),
    )


@dataclass(frozen=True)
class PilotDecision:
    observation: ObservationV2
    privileged: PrivilegedObservation
    selected_action_index: int
    trace_index: int
    pending: PendingV3Transition
    v2_transition: Transition


@dataclass(frozen=True)
class PilotBatch:
    transitions: tuple[V3ReplayTransition, ...]
    belief_samples: tuple[V3H4BeliefSample, ...] | None
    oracle_samples: tuple[OfflineDistillationSample, ...] | None
    trajectories: tuple[V3H5FarmerTrajectory, ...] | None
    strategy_targets: tuple[Mapping[str, float], ...] | None
    decisions: int
    winner_team: str


def unique_legal_actions(actions) -> list[list[int]]:
    """Drop only rank-identical engine duplicates while preserving order."""

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for action in actions:
        key = tuple(action)
        if key not in seen:
            seen.add(key)
            result.append(list(action))
    if not result:
        raise ValueError("environment returned no legal action")
    return result


def _strategy_target(row: Transition) -> dict[str, float]:
    return {
        "min_turns_after": row.target_min_turns_after,
        "min_turns_exact_mask": row.target_min_turns_exact_mask,
        "regain_initiative": row.target_regain_initiative,
        "teammate_finish": row.target_teammate_finish,
        "teammate_finish_mask": row.target_teammate_finish_mask,
        "spring_probability": row.target_spring_probability,
        "structure_cost": row.target_structure_cost,
    }


def collect_real_pilot_episode(
    learner: V3H6Learner,
    *,
    episode_number: int,
    environment_seed: int,
    action_rng: random.Random,
    epsilon: float,
) -> PilotBatch:
    """Collect one real rules-engine episode and build aligned training sidecars."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("pilot epsilon must be in [0, 1]")
    np.random.seed((environment_seed + episode_number) % (1 << 32))
    env = Env("adp")
    env.reset()
    episode_id = f"p2-episode-{episode_number}"
    deal_id = f"p2-deal-{episode_number}"
    decisions: list[PilotDecision] = []
    action_trace: list[tuple[str, tuple[int, ...]]] = []
    features = learner.config.learner.features
    strategy_config = learner.model.strategy_feature_config()
    while True:
        infoset = env.infoset
        # The legacy move generator can emit rank-identical rows through
        # different decomposition paths. They are the same environment action,
        # but H3's offline-compatible action keys are intentionally unique.
        # Remove only exact duplicates, preserving first-seen engine order.
        public_infoset = copy.deepcopy(infoset)
        public_infoset.legal_actions = unique_legal_actions(infoset.legal_actions)
        observation = get_obs_v2(public_infoset, ruleset=learner.ruleset)
        privileged = PrivilegedObservation(
            all_handcards=copy.deepcopy(infoset.all_handcards),
            acting_role=infoset.player_position,
        )
        belief_features = None
        if learner.model.config.belief_feedback != "none":
            belief_features, _ = (
                learner.base.base.policy_features_from_public_observations(
                    [observation]
                )
            )
        with torch.inference_mode():
            output = learner.model.forward_observation(
                observation,
                belief_features=(
                    None if belief_features is None else belief_features[0]
                ),
            )
        mask = output.action_mask.bool()
        valid = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if action_rng.random() < epsilon:
            selected = int(action_rng.choice(valid))
        else:
            selected = int(torch.argmax(
                output.dmc_q[:, 0].masked_fill(~mask, float("-inf"))
            ).item())
        pending = capture_plain_transition(
            observation,
            selected_action_index=selected,
            episode_id=episode_id,
            deal_id=deal_id,
            target_transform=learner.model.config.dmc_target_transform,
            strategy_config=strategy_config,
            style_enabled=learner.model.config.style_enabled,
        )
        if features.adaptive_dmc:
            pending = dataclasses.replace(
                pending,
                adaptive_provenance=AdaptiveSnapshotProvenance(
                    q_old=float(output.dmc_q[selected, 0].item()),
                    policy_version=learner.policy_version,
                    snapshot_slot=0,
                    owner_id=0,
                    generation=episode_number + 1,
                ),
            )
        action = tuple(observation.actions.legal_actions[selected])
        transition = Transition(
            obs=observation,
            action_index=selected,
            position=observation.public.acting_role,
            trace_index=len(action_trace),
            policy_id="p2-current",
            teammate_policy_id=(
                None if observation.public.acting_role == "landlord"
                else "p2-current"
            ),
            policy_version=f"v3_hybrid:{learner.model.config.stable_hash()[:16]}",
            policy_step=learner.policy_version,
        )
        decisions.append(PilotDecision(
            observation, privileged, selected, len(action_trace), pending, transition
        ))
        action_trace.append((transition.position, action))
        _obs, _reward, done, info = env.step(list(action))
        if done:
            break
    terminal = info or {}
    v2_episode = Episode(
        transitions=[item.v2_transition for item in decisions],
        terminal_result=terminal,
        action_trace=action_trace,
    )
    v2_episode.label_from_terminal()
    if features.strategy:
        v2_episode.label_strategy_auxiliary(
            node_budget=learner.model.config.strategy_node_budget,
            time_budget_ms=learner.model.config.strategy_time_budget_ms,
        )
    rows = tuple(
        item.pending.finalize(item.v2_transition.target_score)
        for item in decisions
    )
    belief_samples = None
    if features.belief:
        belief_samples = tuple(
            build_v3_h4_belief_sample(
                item.observation,
                item.privileged,
                strategy_config=strategy_config,
                style_enabled=learner.model.config.style_enabled,
            )
            for item in decisions
        )
    oracle_samples = None
    if features.oracle:
        oracle_samples = tuple(
            DistillationSample(
                public_observation=item.observation,
                privileged_observation=item.privileged,
                action_index=item.selected_action_index,
                target_win=item.v2_transition.target_win,
                target_score=item.v2_transition.target_score,
                sample_id=f"p2:{episode_id}:{index}",
            ).tensorize(
                strategy_config,
                style_enabled=learner.model.config.style_enabled,
            )
            for index, item in enumerate(decisions)
        )
    trajectories = None
    if features.cooperation:
        trajectory_rows = []
        for role in ("landlord_up", "landlord_down"):
            selected_rows = [
                (index, item, row)
                for index, (item, row) in enumerate(zip(decisions, rows))
                if row.role == role
            ]
            if not selected_rows:
                raise RuntimeError("real pilot episode omitted one farmer role")
            trajectory_rows.append(V3H5FarmerTrajectory(
                episode_id=episode_id,
                deal_id=deal_id,
                role=role,
                policy_id="p2-current",
                teammate_policy_id="p2-current",
                decisions=tuple(
                    V3H5FarmerDecision(
                        trace_index=item.trace_index,
                        transition=row,
                        public_features=build_h5_public_features(
                            item.observation, item.selected_action_index
                        ),
                        selected_action_is_pass=(
                            len(item.observation.actions.legal_actions[
                                item.selected_action_index
                            ]) == 0
                        ),
                    )
                    for _index, item, row in selected_rows
                ),
                team_return=float(selected_rows[0][1].v2_transition.target_score),
            ))
        trajectories = tuple(trajectory_rows)
    return PilotBatch(
        transitions=rows,
        belief_samples=belief_samples,
        oracle_samples=oracle_samples,
        trajectories=trajectories,
        strategy_targets=(
            tuple(_strategy_target(item.v2_transition) for item in decisions)
            if features.strategy else None
        ),
        decisions=len(decisions),
        winner_team=str(terminal["winner_team"]),
    )


def slice_pilot_batch(batch: PilotBatch, start: int, end: int) -> PilotBatch:
    """Slice a non-cooperation batch while preserving sidecar alignment."""

    if batch.trajectories is not None:
        raise ValueError("cooperation pilot batches must stay episode-atomic")
    return PilotBatch(
        transitions=batch.transitions[start:end],
        belief_samples=(
            None if batch.belief_samples is None else batch.belief_samples[start:end]
        ),
        oracle_samples=(
            None if batch.oracle_samples is None else batch.oracle_samples[start:end]
        ),
        trajectories=None,
        strategy_targets=(
            None if batch.strategy_targets is None else batch.strategy_targets[start:end]
        ),
        decisions=end - start,
        winner_team=batch.winner_team,
    )


def train_pilot_batch(learner: V3H6Learner, batch: PilotBatch):
    oracle_state = learner.base.base.base.schedule_state()
    strategy_targets = (
        batch.strategy_targets if oracle_state.public_training else None
    )
    oracle_samples = (
        batch.oracle_samples
        if oracle_state.privileged_required
        and (oracle_state.oracle_weight > 0.0 or oracle_state.guidance_weight > 0.0)
        else None
    )
    return learner.train_batch(
        batch.transitions,
        trajectories=batch.trajectories,
        belief_samples=batch.belief_samples,
        oracle_samples=oracle_samples,
        strategy_targets=strategy_targets,
    )


def create_pilot_learner(
    formal: FormalExperimentConfig,
) -> tuple[V3H6Learner, V3H6ResolvedConfig]:
    resolved = build_pilot_resolved_config(formal)
    torch.manual_seed(formal.initialization.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(formal.initialization.seed)
    model = V3HybridModel(build_v2_schema(), resolved.model)
    belief_model = None
    if resolved.learner.features.belief:
        belief_model = BeliefModel(BeliefConfig())
    learner = V3H6Learner(
        model,
        ruleset=RuleSet.legacy(),
        config=resolved,
        belief_model=belief_model,
    )
    return learner, resolved


def validate_pilot_summary(payload: Mapping[str, object]) -> None:
    required = {
        "schema", "protocol", "source_git_sha", "formal_config_sha256",
        "training_semantics_hash", "variant", "ruleset", "seed", "collection", "status",
        "started_at", "finished_at", "wall_clock_seconds", "samples",
        "optimizer_steps", "episodes", "decisions", "metrics", "resume",
        "evaluation", "checkpoint", "environment", "release_candidate",
        "release_status", "playing_strength", "failure",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("P2 pilot summary fields mismatch")
    if payload["schema"] != P2_PILOT_SCHEMA or payload["protocol"] != P2_PILOT_PROTOCOL:
        raise ValueError("P2 pilot summary schema or protocol mismatch")
    for field in ("formal_config_sha256", "training_semantics_hash"):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"P2 pilot {field} must be a full SHA-256")
    source_sha = payload["source_git_sha"]
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or any(c not in "0123456789abcdef" for c in source_sha)
    ):
        raise ValueError("P2 pilot source_git_sha must be a full Git SHA")
    if payload["variant"] not in P2_VARIANTS or payload["ruleset"] != "legacy":
        raise ValueError("P2 pilot variant/ruleset mismatch")
    collection = payload["collection"]
    if not isinstance(collection, Mapping) or set(collection) != {
        "environment_seed", "action_seed", "epsilon"
    }:
        raise ValueError("P2 pilot collection fields mismatch")
    for field in ("environment_seed", "action_seed"):
        value = collection[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"P2 pilot collection {field} is invalid")
    epsilon = collection["epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise ValueError("P2 pilot collection epsilon is invalid")
    if payload["status"] not in {"completed", "stopped", "failed"}:
        raise ValueError("P2 pilot status is invalid")
    for field in ("wall_clock_seconds", "samples", "optimizer_steps", "episodes", "decisions"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"P2 pilot {field} is invalid")
    if payload["release_candidate"] != "NONE" or payload["release_status"] != "NOT READY" or payload["playing_strength"] != "NOT MEASURED":
        raise ValueError("P2 pilot cannot declare release readiness or measured strength")
    checkpoint = payload["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256", "saved"}:
        raise ValueError("P2 pilot checkpoint fields mismatch")
    if checkpoint["saved"] and (
        not isinstance(checkpoint["sha256"], str)
        or len(checkpoint["sha256"]) != 64
    ):
        raise ValueError("P2 saved checkpoint requires SHA-256")
    resume = payload["resume"]
    metrics = payload["metrics"]
    if not isinstance(resume, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("P2 pilot resume and metrics must be objects")
    if set(resume) != {
        "requested", "from_samples", "from_optimizer_steps", "from_episodes",
        "from_decisions", "checkpoint_sha256", "continued_update", "stop_signal",
    }:
        raise ValueError("P2 pilot resume fields mismatch")
    for field in (
        "from_samples", "from_optimizer_steps", "from_episodes", "from_decisions"
    ):
        value = resume[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"P2 pilot resume {field} is invalid")
    resumed_checkpoint = resume["checkpoint_sha256"]
    if bool(resume["requested"]):
        if (
            not isinstance(resumed_checkpoint, str)
            or len(resumed_checkpoint) != 64
            or any(c not in "0123456789abcdef" for c in resumed_checkpoint)
        ):
            raise ValueError("P2 resumed run requires checkpoint SHA-256")
    elif resumed_checkpoint is not None:
        raise ValueError("fresh P2 run cannot declare a resumed checkpoint")
    environment = payload["environment"]
    if (
        not isinstance(environment, Mapping)
        or not isinstance(environment.get("container_id"), str)
        or not environment["container_id"]
    ):
        raise ValueError("P2 pilot requires a container identity")
    for field in ("source_tree",):
        value = environment.get(field)
        if (
            not isinstance(value, str) or len(value) != 40
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"P2 pilot environment {field} is invalid")
    for field in ("gpu", "driver_version", "torch_version", "cuda_runtime", "machine"):
        value = environment.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"P2 pilot environment {field} is invalid")
    if environment.get("cuda_available") is not True:
        raise ValueError("P2 pilot requires CUDA evidence")
    elapsed = float(payload["wall_clock_seconds"])
    expected_samples_rate = (
        float(payload["samples"]) - float(resume["from_samples"])
    ) / max(elapsed, 1e-9)
    expected_steps_rate = (
        float(payload["optimizer_steps"])
        - float(resume["from_optimizer_steps"])
    ) / max(elapsed, 1e-9)
    for name, expected in (
        ("samples_per_second", expected_samples_rate),
        ("optimizer_steps_per_second", expected_steps_rate),
    ):
        observed = metrics.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=1e-12)
        ):
            raise ValueError(f"P2 pilot {name} is inconsistent with raw counters")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if not encoded:
        raise AssertionError("canonical P2 summary unexpectedly empty")


def write_pilot_summary(path: str | Path, payload: Mapping[str, object]) -> None:
    validate_pilot_summary(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


__all__ = [
    "P2_PILOT_PROTOCOL", "P2_PILOT_SCHEMA", "P2_VARIANTS", "PilotBatch",
    "build_pilot_resolved_config", "collect_real_pilot_episode",
    "create_pilot_learner", "slice_pilot_batch", "train_pilot_batch",
    "unique_legal_actions", "validate_pilot_summary", "write_pilot_summary",
]
