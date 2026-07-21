"""H4 safe joint-belief training, deployment, and resume tests."""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.v3_hybrid import (
    AdaptiveDMCConfig,
    BELIEF_FEEDBACK_FARMERS,
    BELIEF_FEEDBACK_NONE,
    V3BeliefPolicy,
    V3H2LearnerConfig,
    V3HybridModel,
    V3HybridModelConfig,
    capture_plain_transition,
    load_v3_h4_public_checkpoint,
    load_v3_hybrid_public_checkpoint,
    save_v3_h4_public_checkpoint,
)
from douzero.v3_hybrid.training.belief_config import (
    BELIEF_MODE_ALTERNATING,
    BELIEF_MODE_AUXILIARY,
    BELIEF_PHASE_AUXILIARY,
    BELIEF_PHASE_POLICY,
    BELIEF_PHASE_SHARED,
    BELIEF_PHASE_SUPERVISED,
    V3H4BeliefTrainingConfig,
)
from douzero.v3_hybrid.training.h3_learner import (
    V3H3Learner,
    V3H3LearnerConfig,
)
from douzero.v3_hybrid.training.h4_learner import (
    V3H4Learner,
    V3H4LearnerConfig,
    build_v3_h4_belief_sample,
)
from douzero.v3_hybrid.training.oracle_schedule import (
    OracleGuidingScheduleConfig,
)


def _model(feedback: str = BELIEF_FEEDBACK_NONE) -> V3HybridModel:
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
            belief_feedback=feedback,
        ),
    )


def _decision(seed: int = 31, *, advance: int = 0):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(advance):
        action = next(
            (value for value in env.infoset.legal_actions if value),
            env.infoset.legal_actions[0],
        )
        _obs, _reward, done, _info = env.step(action)
        assert not done
    infoset = copy.deepcopy(env.infoset)
    infoset.legal_actions = infoset.legal_actions[:4]
    observation = get_obs_v2(infoset, ruleset=RuleSet.legacy())
    privileged = PrivilegedObservation(
        all_handcards=infoset.all_handcards,
        acting_role=infoset.player_position,
    )
    transition = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=f"h4-episode-{seed}-{advance}",
        deal_id=f"h4-deal-{seed}",
        target_transform="raw",
    ).finalize(2.0)
    sample = build_v3_h4_belief_sample(observation, privileged)
    return observation, transition, sample


def _base() -> V3H3LearnerConfig:
    return V3H3LearnerConfig(
        public=V3H2LearnerConfig(
            batch_size=4,
            learning_rate=1e-3,
            adaptive_dmc=AdaptiveDMCConfig(mode="disabled"),
        )
    )


def _belief_config(
    *,
    mode: str = BELIEF_MODE_AUXILIARY,
    shared: bool = False,
) -> V3H4BeliefTrainingConfig:
    return V3H4BeliefTrainingConfig(
        enabled=True,
        mode=mode,
        lambda_belief=0.5,
        learning_rate=1e-3,
        policy_updates_per_cycle=1,
        belief_updates_per_cycle=1,
        shared_updates_per_cycle=(1 if shared and mode == BELIEF_MODE_ALTERNATING else 0),
        shared_encoder_updates=shared,
    )


def _learner(
    *,
    feedback: str = BELIEF_FEEDBACK_NONE,
    mode: str = BELIEF_MODE_AUXILIARY,
    shared: bool = False,
) -> V3H4Learner:
    model = _model(feedback)
    belief = BeliefModel(
        BeliefConfig(
            hidden_size=16,
            num_layers=1,
            shared_context_dim=(16 if shared else 0),
        )
    )
    return V3H4Learner(
        model,
        ruleset=RuleSet.legacy(),
        config=V3H4LearnerConfig(
            base=_base(), belief=_belief_config(mode=mode, shared=shared)
        ),
        belief_model=belief,
    )


def _state(module):
    return {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }


def _changed(before, module) -> bool:
    return any(
        not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def test_public_import_graph_lazily_excludes_labels_and_privileged_modules():
    code = (
        "import sys; import douzero.v3_hybrid; "
        "assert 'douzero.belief.labels' not in sys.modules; "
        "assert 'douzero.observation.privileged' not in sys.modules; "
        "assert 'douzero.v3_hybrid.training.h4_learner' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_disabled_h4_is_exact_h3_and_creates_no_belief_graph():
    _, transition, _ = _decision()
    left_model = _model()
    right_model = copy.deepcopy(left_model)
    h4 = V3H4Learner(
        left_model,
        ruleset=RuleSet.legacy(),
        config=V3H4LearnerConfig(base=_base()),
    )
    h3 = V3H3Learner(right_model, ruleset=RuleSet.legacy(), config=_base())
    assert h4.belief_model is None and h4.belief_optimizer is None
    assert all("belief_projection" not in name for name in h4.model.state_dict())
    h4_metrics = h4.train_batch([transition])
    h3_metrics = h3.train_batch([transition])
    assert h4_metrics.base.as_dict() == h3_metrics.as_dict()
    for name, tensor in h4.model.state_dict().items():
        assert torch.equal(tensor, h3.model.state_dict()[name])


def test_unsupported_h4_combinations_fail_before_training_initialization():
    oracle = dataclasses.replace(
        _base(),
        schedule=OracleGuidingScheduleConfig(
            enabled=True, guided_updates=1, finetune_updates=1
        ),
    )
    with pytest.raises(ValueError, match="deferred to H6"):
        V3H4LearnerConfig(base=oracle, belief=_belief_config())
    with pytest.raises(ValueError, match="shared_context_dim"):
        V3H4Learner(
            _model(BELIEF_FEEDBACK_FARMERS),
            ruleset=RuleSet.legacy(),
            config=V3H4LearnerConfig(
                base=_base(), belief=_belief_config(shared=True)
            ),
            belief_model=BeliefModel(BeliefConfig(hidden_size=16, num_layers=1)),
        )
    with pytest.raises(ValueError, match="require the H4 learner"):
        V3H3Learner(
            _model(BELIEF_FEEDBACK_FARMERS),
            ruleset=RuleSet.legacy(),
            config=_base(),
        )


def test_auxiliary_belief_updates_are_conservative_and_role_weighted_once():
    _, transition, sample = _decision()
    learner = _learner()
    belief_before = _state(learner.belief_model)
    metrics = learner.train_batch([transition], belief_samples=[sample])
    assert metrics.phase == BELIEF_PHASE_AUXILIARY
    assert metrics.policy_updated and metrics.belief_updated
    assert not metrics.shared_encoder_updated
    assert metrics.labels_consumed == 1
    assert metrics.belief_gradient_norm > 0.0
    assert metrics.conservation_max_error <= 2e-4
    assert metrics.role_effective_weights[transition.role] == pytest.approx(1.0)
    assert _changed(belief_before, learner.belief_model)


def test_alternating_schedule_detaches_policy_and_updates_shared_only_in_phase():
    _, transition, sample = _decision(advance=1)
    learner = _learner(
        feedback=BELIEF_FEEDBACK_FARMERS,
        mode=BELIEF_MODE_ALTERNATING,
        shared=True,
    )
    belief_before = _state(learner.belief_model)
    policy = learner.train_batch([transition], belief_samples=[sample])
    assert policy.phase == BELIEF_PHASE_POLICY and policy.policy_updated
    assert not policy.belief_updated
    assert not _changed(belief_before, learner.belief_model)

    model_before = _state(learner.model)
    belief = learner.train_batch([transition], belief_samples=[sample])
    assert belief.phase == BELIEF_PHASE_SUPERVISED and belief.belief_updated
    assert not belief.shared_encoder_updated
    assert not _changed(model_before, learner.model)

    shared_before = _state(learner.model.state_encoder)
    shared = learner.train_batch([transition], belief_samples=[sample])
    assert shared.phase == BELIEF_PHASE_SHARED
    assert shared.shared_encoder_updated and shared.shared_gradient_norm > 0.0
    assert _changed(shared_before, learner.model.state_encoder)
    assert learner.phase() == BELIEF_PHASE_POLICY


def test_h4_checkpoint_resume_preserves_phase_optimizers_and_policy_version(tmp_path):
    _, transition, sample = _decision(advance=1)
    learner = _learner(
        feedback=BELIEF_FEEDBACK_FARMERS,
        mode=BELIEF_MODE_ALTERNATING,
        shared=True,
    )
    learner.train_batch([transition], belief_samples=[sample])
    learner.train_batch([transition], belief_samples=[sample])
    path = tmp_path / "h4-trainer.pt"
    learner.save_checkpoint(path)

    restored = _learner(
        feedback=BELIEF_FEEDBACK_FARMERS,
        mode=BELIEF_MODE_ALTERNATING,
        shared=True,
    )
    restored.load_checkpoint(path)
    assert restored.eligible_updates == learner.eligible_updates == 2
    assert restored.phase() == learner.phase() == BELIEF_PHASE_SHARED
    assert restored.base.policy_version == learner.base.policy_version
    assert restored.statistics.state_dict() == learner.statistics.state_dict()
    assert restored.belief_optimizer.state_dict()["param_groups"] == (
        learner.belief_optimizer.state_dict()["param_groups"]
    )
    for name, tensor in learner.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[name])
    for name, tensor in learner.belief_model.state_dict().items():
        assert torch.equal(tensor, restored.belief_model.state_dict()[name])

    corrupted = torch.load(path, map_location="cpu", weights_only=True)
    corrupted["counters"] = dict(corrupted["counters"])
    corrupted["counters"]["eligible_updates"] += 1
    bad = tmp_path / "h4-corrupted.pt"
    torch.save(corrupted, bad)
    with pytest.raises(CheckpointCompatibilityError, match="phase drift"):
        _learner(
            feedback=BELIEF_FEEDBACK_FARMERS,
            mode=BELIEF_MODE_ALTERNATING,
            shared=True,
        ).load_checkpoint(bad)


def test_public_policy_uses_only_public_posterior_and_strict_coupled_checkpoint(tmp_path):
    observation, _, sample = _decision(advance=1)
    model = _model(BELIEF_FEEDBACK_FARMERS).eval()
    belief = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1)).eval()
    policy = V3BeliefPolicy(model, belief, ruleset=RuleSet.legacy())
    before = policy.forward_observation(observation)
    assert before.num_actions == len(observation.actions.legal_actions)

    # A privileged label is not an input to either public module and changing
    # or dropping it cannot affect the public posterior/policy path.
    public_only = dataclasses.replace(sample, label=None)
    assert public_only.belief_input is sample.belief_input
    after = policy.forward_observation(observation)
    assert torch.equal(before.dmc_q, after.dmc_q)

    path = tmp_path / "h4-public.pt"
    save_v3_h4_public_checkpoint(path, policy)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    flattened = repr(bundle).lower()
    assert "all_handcards" not in flattened
    assert "training_labels" not in flattened
    assert bundle["artifact_access"] == "public"
    restored = load_v3_h4_public_checkpoint(
        path,
        schema=build_v2_schema(),
        ruleset=RuleSet.legacy(),
        model_config=model.config,
        belief_config=belief.config,
    )
    restored_output = restored.forward_observation(observation)
    assert torch.equal(before.dmc_q, restored_output.dmc_q)
    with pytest.raises(CheckpointCompatibilityError, match="H1 public loader"):
        load_v3_hybrid_public_checkpoint(
            path,
            schema=build_v2_schema(),
            ruleset=RuleSet.legacy(),
            config=model.config,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h4_cuda_policy_belief_shared_backward_and_checkpoint(tmp_path):
    _, transition, sample = _decision(advance=1)
    model = _model(BELIEF_FEEDBACK_FARMERS)
    belief = BeliefModel(
        BeliefConfig(hidden_size=16, num_layers=1, shared_context_dim=16)
    )
    base = dataclasses.replace(
        _base(), public=dataclasses.replace(_base().public, device="cuda")
    )
    learner = V3H4Learner(
        model,
        ruleset=RuleSet.legacy(),
        config=V3H4LearnerConfig(
            base=base,
            belief=_belief_config(
                mode=BELIEF_MODE_ALTERNATING, shared=True
            ),
        ),
        belief_model=belief,
    )
    learner.train_batch([transition], belief_samples=[sample])
    learner.train_batch([transition], belief_samples=[sample])
    metrics = learner.train_batch([transition], belief_samples=[sample])
    assert metrics.shared_gradient_norm > 0.0
    path = tmp_path / "h4-cuda.pt"
    learner.save_checkpoint(path)
    restored = V3H4Learner(
        _model(BELIEF_FEEDBACK_FARMERS),
        ruleset=RuleSet.legacy(),
        config=learner.config,
        belief_model=BeliefModel(belief.config),
    )
    restored.load_checkpoint(path)
    assert restored.eligible_updates == 3
