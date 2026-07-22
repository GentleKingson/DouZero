"""H5 sequential farmer credit, leakage, no-op, and resume tests."""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.belief.model import BELIEF_FEATURE_DIM, BeliefConfig, BeliefModel
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2 import ModelV2Config
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.distillation.dataset import DistillationSample
from douzero.v3_hybrid import (
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    BELIEF_FEEDBACK_FARMERS,
    V3H2LearnerConfig,
    V3HybridModel,
    V3HybridModelConfig,
    capture_plain_transition,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
)
from douzero.v3_hybrid.replay import AdaptiveSnapshotProvenance
from douzero.v3_hybrid.training.belief_config import V3H4BeliefTrainingConfig
from douzero.v3_hybrid.training.belief_config import (
    BELIEF_MODE_ALTERNATING,
    BELIEF_MODE_AUXILIARY,
)
from douzero.v3_hybrid.training.cooperation import (
    FARMER_ROLES,
    H5_PUBLIC_FEATURE_DIM,
    MIXER_PRIVILEGED,
    MIXER_PUBLIC,
    FarmerCooperationModule,
    MonotonicSequentialMixer,
    V3H5CooperationConfig,
    V3H5FarmerDecision,
    V3H5FarmerTrajectory,
    build_h5_public_features,
    teammate_belief_summary,
    validate_farmer_pairs,
)
from douzero.v3_hybrid.training.h3_learner import V3H3LearnerConfig
from douzero.v3_hybrid.training.h4_learner import (
    V3H4Learner,
    V3H4LearnerConfig,
    build_v3_h4_belief_sample,
)
from douzero.v3_hybrid.training.h5_learner import (
    V3H5Learner,
    V3H5LearnerConfig,
)
from douzero.v3_hybrid.training.oracle_schedule import (
    OracleGuidingScheduleConfig,
)


def _model(*, belief_feedback: str = "none") -> V3HybridModel:
    torch.manual_seed(20260722)
    return V3HybridModel(
        build_v2_schema(),
        V3HybridModelConfig(
            hidden_size=16,
            history_layers=1,
            history_heads=4,
            shared_fusion_layers=1,
            landlord_adapter_layers=1,
            farmer_adapter_layers=1,
            belief_feedback=belief_feedback,
        ),
    )


def _observation(role: str, seed: int):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(120):
        if env._acting_player_position == role:
            infoset = copy.deepcopy(env.infoset)
            infoset.legal_actions = infoset.legal_actions[:4]
            return get_obs_v2(infoset, ruleset=RuleSet.legacy())
        action = next(
            (item for item in env.infoset.legal_actions if item),
            env.infoset.legal_actions[0],
        )
        _obs, _reward, done, _info = env.step(action)
        if done:
            env.reset()
    raise AssertionError(f"could not reach {role}")


def _transition(
    role: str,
    seed: int,
    *,
    episode_id: str = "episode-1",
    deal_id: str = "deal-1",
    team_return: float = 2.0,
):
    observation = _observation(role, seed)
    pending = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=episode_id,
        deal_id=deal_id,
        target_transform="raw",
    )
    return observation, pending.finalize(team_return)


def _pair(
    *,
    up_length: int = 2,
    down_length: int = 1,
    team_return: float = 2.0,
    episode_id: str = "episode-1",
    deal_id: str = "deal-1",
):
    trajectories = []
    all_rows = []
    policies = {"landlord_up": "current-v7", "landlord_down": "league-v3"}
    for role, count, seed in (
        ("landlord_up", up_length, 100),
        ("landlord_down", down_length, 200),
    ):
        rows = tuple(
            _transition(
                role,
                seed + index,
                episode_id=episode_id,
                deal_id=deal_id,
                team_return=team_return,
            )[1]
            for index in range(count)
        )
        features = torch.linspace(
            0.0, 1.0, steps=count * H5_PUBLIC_FEATURE_DIM
        ).reshape(count, H5_PUBLIC_FEATURE_DIM)
        teammate = "landlord_down" if role == "landlord_up" else "landlord_up"
        trajectory = V3H5FarmerTrajectory(
            episode_id=episode_id,
            deal_id=deal_id,
            role=role,
            policy_id=policies[role],
            teammate_policy_id=policies[teammate],
            decisions=tuple(
                V3H5FarmerDecision(
                    trace_index=trace_index,
                    transition=transition,
                    public_features=features[index],
                    selected_action_is_pass=index % 2 == 0,
                )
                for index, (trace_index, transition) in enumerate(
                    zip(
                        range(
                            0 if role == "landlord_up" else 1,
                            count * 2,
                            2,
                        ),
                        rows,
                    )
                )
            ),
            team_return=team_return,
        )
        trajectories.append(trajectory)
        all_rows.extend(rows)
    return tuple(all_rows), tuple(trajectories)


def _pair_with_training_sidecars():
    rows = []
    trajectories = []
    belief_samples = []
    oracle_samples = []
    policies = {"landlord_up": "current-v7", "landlord_down": "league-v3"}
    for trace_index, (role, seed) in enumerate((
        ("landlord_up", 501),
        ("landlord_down", 502),
    )):
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
        for _ in range(120):
            if env._acting_player_position == role:
                infoset = copy.deepcopy(env.infoset)
                infoset.legal_actions = infoset.legal_actions[:4]
                break
            action = next(
                (item for item in env.infoset.legal_actions if item),
                env.infoset.legal_actions[0],
            )
            _obs, _reward, done, _info = env.step(action)
            if done:
                env.reset()
        else:
            raise AssertionError(f"could not reach {role}")
        observation = get_obs_v2(infoset, ruleset=RuleSet.legacy())
        privileged = PrivilegedObservation(
            all_handcards=infoset.all_handcards,
            acting_role=infoset.player_position,
        )
        row = capture_plain_transition(
            observation,
            selected_action_index=0,
            episode_id="episode-sidecars",
            deal_id="deal-sidecars",
            target_transform="raw",
        ).finalize(2.0)
        rows.append(row)
        belief_samples.append(
            build_v3_h4_belief_sample(observation, privileged)
        )
        oracle_samples.append(
            DistillationSample(
                public_observation=observation,
                privileged_observation=privileged,
                action_index=0,
                target_win=1.0,
                target_score=2.0,
                sample_id=f"oracle-{role}",
            ).tensorize()
        )
        teammate = "landlord_down" if role == "landlord_up" else "landlord_up"
        trajectories.append(V3H5FarmerTrajectory(
            episode_id="episode-sidecars",
            deal_id="deal-sidecars",
            role=role,
            policy_id=policies[role],
            teammate_policy_id=policies[teammate],
            decisions=(V3H5FarmerDecision(
                trace_index=trace_index,
                transition=row,
                public_features=torch.linspace(0.0, 1.0, H5_PUBLIC_FEATURE_DIM),
                selected_action_is_pass=False,
            ),),
            team_return=2.0,
        ))
    return (
        tuple(rows),
        tuple(trajectories),
        tuple(belief_samples),
        tuple(oracle_samples),
    )


def _stamp_policy_version(rows, trajectories, version: int):
    replacements = {}
    stamped_rows = []
    for index, row in enumerate(rows):
        stamped = dataclasses.replace(
            row,
            adaptive_provenance=AdaptiveSnapshotProvenance(
                q_old=0.1,
                policy_version=version,
                snapshot_slot=index,
                owner_id=1,
                generation=version,
            ),
        )
        replacements[id(row)] = stamped
        stamped_rows.append(stamped)
    stamped_trajectories = tuple(
        dataclasses.replace(
            trajectory,
            decisions=tuple(
                dataclasses.replace(
                    decision, transition=replacements[id(decision.transition)]
                )
                for decision in trajectory.decisions
            ),
        )
        for trajectory in trajectories
    )
    return tuple(stamped_rows), stamped_trajectories


def _base(*, batch_size: int = 8, device: str = "cpu") -> V3H4LearnerConfig:
    return V3H4LearnerConfig(
        base=V3H3LearnerConfig(
            public=V3H2LearnerConfig(
                batch_size=batch_size,
                learning_rate=1e-3,
                device=device,
                adaptive_dmc=AdaptiveDMCConfig(mode="disabled"),
            )
        )
    )


def _coop(**changes) -> V3H5CooperationConfig:
    values = {
        "enabled": True,
        "hidden_size": 12,
        "lambda_coop": 0.5,
        "lambda_team_value": 1.0,
        "lambda_trajectory_consistency": 0.25,
        "learning_rate": 1e-3,
    }
    values.update(changes)
    return V3H5CooperationConfig(**values)


def _learner(*, device: str = "cpu", **coop_changes) -> V3H5Learner:
    return V3H5Learner(
        _model(),
        ruleset=RuleSet.legacy(),
        config=V3H5LearnerConfig(
            base=_base(device=device), cooperation=_coop(**coop_changes)
        ),
    )


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before, module, prefix: str = "") -> bool:
    return any(
        name.startswith(prefix) and not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def test_public_import_graph_excludes_h5_and_privileged_training_modules():
    code = (
        "import sys; import douzero.v3_hybrid; "
        "assert 'douzero.v3_hybrid.training.cooperation' not in sys.modules; "
        "assert 'douzero.v3_hybrid.training.h5_learner' not in sys.modules; "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_is_identity_bound_scheduled_and_fail_closed():
    baseline = _coop(warmup_updates=2, ramp_updates=2)
    assert baseline.schedule_weight(0) == 0.0
    assert baseline.schedule_weight(1) == 0.0
    assert baseline.schedule_weight(2) == pytest.approx(0.25)
    assert baseline.schedule_weight(3) == pytest.approx(0.5)
    for name, value in {
        "hidden_size": 13,
        "trajectory_layers": 2,
        "dropout": 0.1,
        "lambda_team_value": 0.5,
        "lambda_trajectory_consistency": 0.5,
        "warmup_updates": 3,
        "ramp_updates": 3,
        "learning_rate": 2e-3,
        "max_grad_norm": 5.0,
        "update_public_model": False,
    }.items():
        assert dataclasses.replace(baseline, **{name: value}).stable_hash() != baseline.stable_hash()
    assert V3H5CooperationConfig().enabled is False
    with pytest.raises(ValueError, match="disabled H5"):
        V3H5CooperationConfig(lambda_coop=1.0)
    with pytest.raises(ValueError, match="enabled mixer"):
        _coop(mixer_mode=MIXER_PUBLIC)
    with pytest.raises(ValueError, match="privileged_state_dim"):
        _coop(mixer_mode=MIXER_PRIVILEGED, lambda_mixer=1.0)
    with pytest.raises(ValueError, match="active sidecar loss"):
        _coop(lambda_team_value=0.0, lambda_trajectory_consistency=0.0)


def test_h6_combination_identity_and_remaining_graph_guards():
    belief_combined = V3H5LearnerConfig(
        base=dataclasses.replace(
            _base(), belief=V3H4BeliefTrainingConfig(
                enabled=True, mode=BELIEF_MODE_AUXILIARY, lambda_belief=1.0
            )
        ),
        cooperation=_coop(),
    )
    assert belief_combined.compatibility_dict()["identity_version"] == 2
    oracle = dataclasses.replace(
        _base().base,
        schedule=OracleGuidingScheduleConfig(
            enabled=True, guided_updates=1, finetune_updates=1
        ),
    )
    oracle_combined = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=oracle), cooperation=_coop()
    )
    assert oracle_combined.compatibility_dict()["identity_version"] == 2
    with pytest.raises(ValueError, match="disabled H4"):
        V3H5Learner(
            _model(belief_feedback=BELIEF_FEEDBACK_FARMERS),
            ruleset=RuleSet.legacy(),
            config=V3H5LearnerConfig(base=_base(), cooperation=_coop()),
        )
    for role in FARMER_ROLES:
        public = dataclasses.replace(
            _base().base.public, **{f"{role}_weight": 0.0}
        )
        with pytest.raises(ValueError, match="positive weights for both farmers"):
            V3H5LearnerConfig(
                base=V3H4LearnerConfig(
                    base=dataclasses.replace(_base().base, public=public)
                ),
                cooperation=_coop(warmup_updates=1),
            )


def test_h6_belief_feedback_and_cooperation_train_and_resume(tmp_path):
    rows, trajectories, belief_samples, _oracle_samples = (
        _pair_with_training_sidecars()
    )
    config = V3H5LearnerConfig(
        base=dataclasses.replace(
            _base(),
            belief=V3H4BeliefTrainingConfig(
                enabled=True,
                mode=BELIEF_MODE_AUXILIARY,
                lambda_belief=0.5,
                learning_rate=1e-3,
            ),
        ),
        cooperation=_coop(),
    )

    def build_learner():
        return V3H5Learner(
            _model(belief_feedback=BELIEF_FEEDBACK_FARMERS),
            ruleset=RuleSet.legacy(),
            config=config,
            belief_model=BeliefModel(
                BeliefConfig(hidden_size=16, num_layers=1)
            ),
        )

    learner = build_learner()
    metrics = learner.train_batch(
        rows,
        trajectories=trajectories,
        belief_samples=belief_samples,
    )
    assert metrics.cooperation_updated
    assert metrics.public_updated
    assert metrics.base.belief_updated

    checkpoint = tmp_path / "belief-cooperation.pt"
    learner.save_checkpoint(checkpoint)
    restored = build_learner()
    restored.load_checkpoint(checkpoint)
    assert restored.eligible_updates == learner.eligible_updates == 1
    assert restored.policy_version == learner.policy_version


def test_alternating_belief_only_phase_skips_cooperation_and_public_updates():
    rows, trajectories, belief_samples, _oracle_samples = (
        _pair_with_training_sidecars()
    )
    config = V3H5LearnerConfig(
        base=dataclasses.replace(
            _base(),
            belief=V3H4BeliefTrainingConfig(
                enabled=True,
                mode=BELIEF_MODE_ALTERNATING,
                lambda_belief=0.5,
                learning_rate=1e-3,
                policy_updates_per_cycle=1,
                belief_updates_per_cycle=1,
            ),
        ),
        cooperation=_coop(),
    )
    learner = V3H5Learner(
        _model(belief_feedback=BELIEF_FEEDBACK_FARMERS),
        ruleset=RuleSet.legacy(),
        config=config,
        belief_model=BeliefModel(BeliefConfig(hidden_size=16, num_layers=1)),
    )
    first = learner.train_batch(
        rows,
        trajectories=trajectories,
        belief_samples=belief_samples,
    )
    assert first.cooperation_updated
    cooperation_before = _state(learner.cooperation)
    public_before = _state(learner.model)
    policy_version = learner.policy_version

    second = learner.train_batch(
        rows,
        trajectories=trajectories,
        belief_samples=belief_samples,
    )
    assert second.base.phase == "belief"
    assert second.base.belief_updated
    assert not second.cooperation_updated
    assert not second.public_updated
    assert second.schedule_weight == 0.0
    assert learner.policy_version == policy_version
    assert not _changed(cooperation_before, learner.cooperation)
    assert not _changed(public_before, learner.model)


def test_h6_oracle_warmup_cooperation_checkpoint_accepts_distinct_counters(
    tmp_path,
):
    rows, trajectories, _belief_samples, oracle_samples = (
        _pair_with_training_sidecars()
    )
    oracle = dataclasses.replace(
        _base().base,
        schedule=OracleGuidingScheduleConfig(
            enabled=True,
            warmup_updates=2,
            guided_updates=1,
            finetune_updates=1,
        ),
    )
    config = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=oracle),
        cooperation=_coop(),
    )

    def build_learner():
        return V3H5Learner(
            _model(), ruleset=RuleSet.legacy(), config=config
        )

    learner = build_learner()
    metrics = learner.train_batch(
        rows,
        trajectories=trajectories,
        oracle_samples=oracle_samples,
    )
    assert metrics.cooperation_updated
    assert not metrics.public_updated
    assert not metrics.base.policy_updated
    assert learner.statistics.cooperation_updates == 1
    assert learner.statistics.public_updates == 0

    checkpoint = tmp_path / "oracle-warmup-cooperation.pt"
    learner.save_checkpoint(checkpoint)
    restored = build_learner()
    restored.load_checkpoint(checkpoint)
    assert restored.statistics.cooperation_updates == 1
    assert restored.statistics.public_updates == 0
    assert restored.policy_version == learner.policy_version


def test_public_feature_builder_uses_existing_strategy_and_conservative_belief():
    observation = _observation("landlord_up", 41)
    values = np.zeros(BELIEF_FEATURE_DIM, dtype=np.float32)
    values[:15] = 1.0
    values[30:45] = 0.8
    values[-3:] = [15.0, 15.0, 3.0]
    unseen = np.full(15, 2.0, dtype=np.float32)
    teammate_a = teammate_belief_summary(
        values, unseen, opponent_a_is_teammate=True
    )
    teammate_b = teammate_belief_summary(
        values, unseen, opponent_a_is_teammate=False
    )
    assert np.allclose(teammate_a, teammate_b)
    features = build_h5_public_features(
        observation,
        0,
        belief_features=values,
        unseen_counts=unseen,
        opponent_a_role="landlord_down",
    )
    assert features.shape == (H5_PUBLIC_FEATURE_DIM,)
    assert torch.isfinite(features).all()
    for invalid_role in ("landlord_up", "typo", ""):
        with pytest.raises(ValueError, match="teammate or landlord"):
            build_h5_public_features(
                observation,
                0,
                belief_features=values,
                unseen_counts=unseen,
                opponent_a_role=invalid_role,
            )
    with pytest.raises(TypeError, match="ObservationV2"):
        build_h5_public_features({"all_handcards": {}}, 0)


def test_farmer_pair_contract_binds_reward_perspective_and_league_teammates():
    _rows, trajectories = _pair(team_return=-2.0)
    pairs = validate_farmer_pairs(trajectories)
    assert pairs[0][0].team_return == pairs[0][1].team_return == -2.0
    assert pairs[0][0].policy_id != pairs[0][1].policy_id
    original = trajectories[0]
    canonical = dataclasses.replace(
        original,
        decisions=tuple(reversed(original.decisions)),
    )
    assert canonical.decision_indices == original.decision_indices
    assert all(
        actual is expected
        for actual, expected in zip(canonical.transitions, original.transitions)
    )
    assert torch.equal(canonical.public_features, original.public_features)
    assert torch.equal(
        canonical.selected_action_is_pass, original.selected_action_is_pass
    )
    with pytest.raises(ValueError):
        validate_farmer_pairs(
            (trajectories[0], dataclasses.replace(trajectories[1], team_return=2.0))
        )
    with pytest.raises(ValueError, match="teammate policy"):
        validate_farmer_pairs(
            (
                trajectories[0],
                dataclasses.replace(
                    trajectories[1], teammate_policy_id="wrong-checkpoint"
                ),
            )
        )
    with pytest.raises(ValueError, match="farmer-only"):
        dataclasses.replace(trajectories[0], role="landlord")


def test_monotonic_mixer_has_nonnegative_derivatives():
    torch.manual_seed(1)
    mixer = MonotonicSequentialMixer(state_dim=5, hidden_size=7)
    local = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
    state = torch.randn(2, 5)
    mixed, weights = mixer(local, state)
    mixed.sum().backward()
    assert torch.all(weights > 0)
    assert torch.all(local.grad >= 0)
    assert torch.allclose(local.grad, weights)


def test_module_handles_unequal_sequences_padding_and_early_finish():
    module = FarmerCooperationModule(8, _coop())
    embedding = torch.randn(2, 3, 8)
    features = torch.randn(2, 3, H5_PUBLIC_FEATURE_DIM)
    local_q = torch.randn(2, 3)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    embedding[1, 1:] = float("nan")
    features[1, 1:] = float("nan")
    local_q[1, 1:] = float("nan")
    output = module(
        embedding, features, local_q, mask, torch.tensor([0, 1])
    )
    assert output.team_value.shape == (2, 3)
    assert torch.equal(output.team_value[1, 1:], torch.zeros(2))
    assert torch.isfinite(output.trajectory_embedding).all()
    nonprefix = torch.tensor([[True, False, True], [True, False, False]])
    with pytest.raises(ValueError, match="true prefix"):
        module(embedding, features, local_q, nonprefix, torch.tensor([0, 1]))
    with pytest.raises(FloatingPointError, match="real trajectory"):
        module(
            embedding,
            features,
            local_q,
            torch.ones_like(mask),
            torch.tensor([0, 1]),
        )


def test_team_value_only_updates_sidecar_and_farmer_public_paths():
    rows, trajectories = _pair()
    learner = _learner()
    model_before = _state(learner.model)
    sidecar_before = _state(learner.cooperation)
    metrics = learner.train_batch(rows, trajectories=trajectories)
    assert metrics.cooperation_updated
    assert metrics.public_updated
    assert learner.base.base.policy_version == 1
    assert learner.policy_version == 2
    assert metrics.policy_version == learner.policy_version
    assert metrics.farmer_samples == 3 and metrics.episodes == 1
    assert metrics.role_samples == {"landlord_up": 2, "landlord_down": 1}
    assert metrics.pass_samples == 2
    assert metrics.teammate_policy_pairs == (("current-v7", "league-v3"),)
    assert metrics.mixer_weight_min is None
    assert _changed(sidecar_before, learner.cooperation)
    assert _changed(model_before, learner.model, "role_adapters.landlord_up")
    assert not _changed(model_before, learner.model, "role_adapters.landlord.")
    # Team-value heads are a training sidecar; the public local-Q heads stay
    # independent and are changed only by the ordinary DMC base loss.
    assert learner.cooperation.team_value_heads["landlord_up"] is not learner.model.role_heads["landlord_up"].dmc_head


def test_adaptive_replay_accepts_the_aggregate_h5_policy_version():
    public = dataclasses.replace(
        _base().base.public,
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID),
    )
    learner = V3H5Learner(
        _model(),
        ruleset=RuleSet.legacy(),
        config=V3H5LearnerConfig(
            base=V3H4LearnerConfig(
                base=dataclasses.replace(_base().base, public=public)
            ),
            cooperation=_coop(),
        ),
    )
    rows, trajectories = _pair()
    rows, trajectories = _stamp_policy_version(rows, trajectories, 0)
    first = learner.train_batch(rows, trajectories=trajectories)
    assert first.policy_version == learner.policy_version == 2

    rows, trajectories = _pair(episode_id="episode-2", deal_id="deal-2")
    rows, trajectories = _stamp_policy_version(rows, trajectories, 2)
    second = learner.train_batch(rows, trajectories=trajectories)
    assert learner.base.base.policy_version == 2
    assert second.base.base.policy_version == 3
    assert second.policy_version == learner.policy_version == 4
    with pytest.raises(TypeError, match="external_policy_version_offset"):
        learner.base.base.train_batch(
            rows, external_policy_version_offset=10_000
        )


def test_mixer_training_uses_unequal_local_q_and_public_or_privileged_state():
    rows, trajectories = _pair(up_length=3, down_length=1)
    public = _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0)
    metrics = public.train_batch(rows, trajectories=trajectories)
    assert metrics.mixer_weight_min is not None and metrics.mixer_weight_min > 0
    with pytest.raises(ValueError, match="rejects privileged"):
        public.cooperation(
            torch.randn(2, 1, 16),
            torch.randn(2, 1, H5_PUBLIC_FEATURE_DIM),
            torch.randn(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
            torch.tensor([0, 1]),
            privileged_mixer_state=torch.randn(1, 3),
        )
    privileged = _learner(
        mixer_mode=MIXER_PRIVILEGED,
        lambda_mixer=1.0,
        privileged_state_dim=3,
    )
    metrics = privileged.train_batch(
        rows,
        trajectories=trajectories,
        privileged_mixer_state=torch.randn(1, 3),
    )
    assert metrics.cooperation_updated
    with pytest.raises(ValueError, match="shape mismatch"):
        privileged.train_batch(
            rows,
            trajectories=trajectories,
            privileged_mixer_state=torch.randn(1, 2),
        )


def test_disabled_h5_is_exact_h4_and_allocates_no_sidecar():
    _observation_value, transition = _transition("landlord_up", 88)
    left_model = _model()
    right_model = copy.deepcopy(left_model)
    h5 = V3H5Learner(
        left_model,
        ruleset=RuleSet.legacy(),
        config=V3H5LearnerConfig(base=_base()),
    )
    h4 = V3H4Learner(right_model, ruleset=RuleSet.legacy(), config=_base())
    assert h5.cooperation is None and h5.cooperation_optimizer is None
    h5_metrics = h5.train_batch([transition])
    h4_metrics = h4.train_batch([transition])
    assert h5_metrics.base.as_dict() == h4_metrics.as_dict()
    for name, value in h5.model.state_dict().items():
        assert torch.equal(value, h4.model.state_dict()[name])
    assert ModelV2Config() == ModelV2Config()
    with pytest.raises(ValueError, match="rejects cooperation data"):
        h5.train_batch([transition], trajectories=())


def test_checkpoint_resume_is_strict_and_schedule_continues(tmp_path):
    rows, trajectories = _pair()
    learner = _learner(warmup_updates=1, ramp_updates=2)
    first = learner.train_batch(rows, trajectories=trajectories)
    assert first.schedule_weight == 0.0
    second = learner.train_batch(rows, trajectories=trajectories)
    assert second.schedule_weight == pytest.approx(0.25)
    path = tmp_path / "h5.pt"
    learner.save_checkpoint(path)
    restored = _learner(warmup_updates=1, ramp_updates=2)
    restored.load_checkpoint(path)
    assert restored.eligible_updates == learner.eligible_updates == 2
    assert restored.policy_version == learner.policy_version == 3
    assert restored.samples_consumed == learner.samples_consumed
    assert restored.statistics.state_dict() == learner.statistics.state_dict()
    for name, value in restored.cooperation.state_dict().items():
        assert torch.equal(value, learner.cooperation.state_dict()[name])
    third_left = learner.train_batch(rows, trajectories=trajectories)
    third_right = restored.train_batch(rows, trajectories=trajectories)
    assert third_left.schedule_weight == third_right.schedule_weight == pytest.approx(0.5)
    assert restored.policy_version == learner.policy_version == 5
    for name, value in restored.cooperation.state_dict().items():
        assert torch.equal(value, learner.cooperation.state_dict()[name])


def test_checkpoint_accepts_only_the_enabled_sidecar_parameter_state(tmp_path):
    rows, trajectories = _pair()
    learner = _learner(lambda_trajectory_consistency=0.0)
    learner.train_batch(rows, trajectories=trajectories)
    payload = learner.cooperation_optimizer.state_dict()
    parameter_ids = [
        value for group in payload["param_groups"] for value in group["params"]
    ]
    parameter_names = [
        name for name, _parameter in learner.cooperation.named_parameters()
    ]
    active_names = {
        name
        for parameter_id, name in zip(parameter_ids, parameter_names)
        if parameter_id in payload["state"]
    }
    assert active_names
    assert all(name.startswith("team_value_heads.") for name in active_names)

    path = tmp_path / "team-value-only.pt"
    learner.save_checkpoint(path)
    restored = _learner(lambda_trajectory_consistency=0.0)
    restored.load_checkpoint(path)
    resumed = restored.train_batch(rows, trajectories=trajectories)
    assert resumed.cooperation_updated
    assert restored.eligible_updates == 2


def test_checkpoint_resume_replays_multilayer_gru_dropout_rng(tmp_path):
    rows, trajectories = _pair()
    learner = _learner(trajectory_layers=2, dropout=0.25)
    learner.train_batch(rows, trajectories=trajectories)
    path = tmp_path / "dropout-rng.pt"
    learner.save_checkpoint(path)

    learner.train_batch(rows, trajectories=trajectories)
    uninterrupted_model = _state(learner.model)
    uninterrupted_sidecar = _state(learner.cooperation)

    restored = _learner(trajectory_layers=2, dropout=0.25)
    restored.load_checkpoint(path)
    restored.train_batch(rows, trajectories=trajectories)
    assert restored.policy_version == learner.policy_version
    for name, value in restored.model.state_dict().items():
        assert torch.equal(value, uninterrupted_model[name])
    for name, value in restored.cooperation.state_dict().items():
        assert torch.equal(value, uninterrupted_sidecar[name])


def test_checkpoint_identity_drift_and_public_package_exclusion(tmp_path):
    rows, trajectories = _pair()
    learner = _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0)
    learner.train_batch(rows, trajectories=trajectories)
    training_path = tmp_path / "training.pt"
    learner.save_checkpoint(training_path)
    checkpoint = torch.load(training_path, map_location="cpu", weights_only=True)
    optimizer_state = checkpoint["cooperation_optimizer_state_dict"]["state"]
    assert len(optimizer_state) == len(learner.cooperation.state_dict())
    for suffix, mutate in (
        ("empty", lambda state: state.clear()),
        ("partial", lambda state: state.pop(next(iter(state)))),
    ):
        corrupted = copy.deepcopy(checkpoint)
        mutate(corrupted["cooperation_optimizer_state_dict"]["state"])
        corrupted_path = tmp_path / f"training-{suffix}.pt"
        torch.save(corrupted, corrupted_path)
        with pytest.raises(
            CheckpointCompatibilityError, match="missing or incomplete"
        ):
            _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0).load_checkpoint(
                corrupted_path
            )
    corrupted = copy.deepcopy(checkpoint)
    corrupted["counters"]["policy_version"] += 1
    policy_path = tmp_path / "training-policy-version.pt"
    torch.save(corrupted, policy_path)
    with pytest.raises(CheckpointCompatibilityError, match="policy version"):
        _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0).load_checkpoint(
            policy_path
        )
    sidecar_name = next(iter(checkpoint["cooperation_state_dict"]))
    sidecar_value = checkpoint["cooperation_state_dict"][sidecar_name]
    invalid_sidecar_values = {
        "nonfinite": torch.full_like(sidecar_value, float("nan")),
        "shape": sidecar_value.reshape(-1)[:-1],
        "dtype": sidecar_value.to(torch.float64),
    }
    for suffix, invalid in invalid_sidecar_values.items():
        corrupted = copy.deepcopy(checkpoint)
        corrupted["cooperation_state_dict"][sidecar_name] = invalid
        corrupted_path = tmp_path / f"training-sidecar-{suffix}.pt"
        torch.save(corrupted, corrupted_path)
        with pytest.raises(
            CheckpointCompatibilityError, match="incompatible or non-finite"
        ):
            _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0).load_checkpoint(
                corrupted_path
            )
    student_state = checkpoint["h4_checkpoint"]["h3_checkpoint"][
        "student_state_dict"
    ]
    student_name = next(iter(student_state))
    student_value = student_state[student_name]
    invalid_student_values = {
        "nonfinite": torch.full_like(student_value, float("inf")),
        "shape": student_value.reshape(-1)[:-1],
        "dtype": student_value.to(torch.float64),
    }
    for suffix, invalid in invalid_student_values.items():
        corrupted = copy.deepcopy(checkpoint)
        corrupted["h4_checkpoint"]["h3_checkpoint"]["student_state_dict"][
            student_name
        ] = invalid
        corrupted_path = tmp_path / f"training-student-{suffix}.pt"
        torch.save(corrupted, corrupted_path)
        with pytest.raises(
            CheckpointCompatibilityError, match="incompatible or non-finite"
        ):
            _learner(mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0).load_checkpoint(
                corrupted_path
            )
    drifted = _learner(
        mixer_mode=MIXER_PUBLIC,
        lambda_mixer=1.0,
        hidden_size=13,
    )
    with pytest.raises(CheckpointCompatibilityError, match="config"):
        drifted.load_checkpoint(training_path)
    with pytest.raises(CheckpointCompatibilityError):
        load_v3_hybrid_public_checkpoint(
            training_path,
            schema=build_v2_schema(),
            ruleset=RuleSet.legacy(),
            config=learner.model.config,
        )

    public_path = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(
        public_path, learner.model, ruleset=RuleSet.legacy()
    )
    payload = torch.load(public_path, map_location="cpu", weights_only=True)
    serialized_names = " ".join(payload["state_dict"]).lower()
    assert "mixer" not in serialized_names
    assert "cooperation" not in serialized_names
    assert "trajectory" not in serialized_names
    loaded = load_v3_hybrid_public_checkpoint(
        public_path,
        schema=build_v2_schema(),
        ruleset=RuleSet.legacy(),
        config=learner.model.config,
    )
    assert isinstance(loaded, V3HybridModel)


def test_public_scalar_and_batched_forward_remain_equal_after_h5_update():
    rows, trajectories = _pair()
    learner = _learner()
    learner.train_batch(rows, trajectories=trajectories)
    observations = [_observation(role, 900 + index) for index, role in enumerate(FARMER_ROLES)]
    scalar = [learner.model.forward_observation(item) for item in observations]
    batched = learner.model.forward_observation_batch(observations)
    for index, expected in enumerate(scalar):
        actual = batched.select(index)
        count = expected.num_actions
        assert torch.allclose(actual.dmc_q[:count], expected.dmc_q, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_parameter_update_checkpoint_resume_and_second_update(tmp_path):
    rows, trajectories = _pair()
    learner = _learner(device="cuda", mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0)
    before = _state(learner.cooperation)
    first = learner.train_batch(rows, trajectories=trajectories)
    assert first.cooperation_updated and _changed(before, learner.cooperation)
    assert all(parameter.is_cuda for parameter in learner.model.parameters())
    assert all(parameter.is_cuda for parameter in learner.cooperation.parameters())
    checkpoint = tmp_path / "cuda-h5.pt"
    learner.save_checkpoint(checkpoint)
    restored = _learner(
        device="cuda", mixer_mode=MIXER_PUBLIC, lambda_mixer=1.0
    )
    restored.load_checkpoint(checkpoint)
    resumed_before = _state(restored.cooperation)
    second = restored.train_batch(rows, trajectories=trajectories)
    assert second.cooperation_updated
    assert _changed(resumed_before, restored.cooperation)
    assert restored.eligible_updates == 2


def test_nonfinite_and_padding_contracts_fail_closed():
    rows, trajectories = _pair()
    decision = trajectories[0].decisions[0]
    broken = decision.public_features.clone()
    broken[0] = float("inf")
    with pytest.raises(ValueError, match="finite CPU row"):
        dataclasses.replace(decision, public_features=broken)
    duplicate = dataclasses.replace(
        trajectories[0].decisions[1], trace_index=decision.trace_index
    )
    with pytest.raises(ValueError, match="trace indices must be unique"):
        dataclasses.replace(
            trajectories[0], decisions=(decision, duplicate)
        )
    with pytest.raises(ValueError, match="both farmer trajectories"):
        _learner().train_batch(
            trajectories[0].transitions, trajectories=(trajectories[0],)
        )
    _observation_value, omitted = _transition(
        "landlord_up", 303, episode_id="episode-1", deal_id="deal-1"
    )
    with pytest.raises(ValueError, match="every farmer transition"):
        _learner().train_batch((*rows, omitted), trajectories=trajectories)
