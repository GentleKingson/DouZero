"""H3 Oracle schedule, training, resume, and public-boundary tests."""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.distillation.dataset import DistillationSample
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.runtime.policy_snapshot import PolicyLease
from douzero.v3_hybrid import (
    AdaptiveDMCConfig,
    V3H2LearnerConfig,
    V3HybridModel,
    V3HybridModelConfig,
    capture_adaptive_transition,
    capture_plain_transition,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
)
from douzero.v3_hybrid.training.guidance_config import OracleGuidanceLossConfig
from douzero.v3_hybrid.training.h3_learner import (
    V3H3Learner,
    V3H3LearnerConfig,
)
from douzero.v3_hybrid.training.oracle import V3OracleConfig, V3PrivilegedOracle
from douzero.v3_hybrid.training.oracle_loss import oracle_guidance_loss
from douzero.v3_hybrid.training.oracle_schedule import (
    ORACLE_PHASE_COMPLETE,
    ORACLE_PHASE_GUIDED,
    ORACLE_PHASE_PUBLIC_FINETUNE,
    ORACLE_PHASE_WARMUP,
    OracleGuidingScheduleConfig,
)


def _model() -> V3HybridModel:
    torch.manual_seed(20260721)
    return V3HybridModel(
        build_v2_schema(),
        V3HybridModelConfig(
            hidden_size=16,
            history_layers=1,
            history_heads=4,
            shared_fusion_layers=1,
            landlord_adapter_layers=1,
            farmer_adapter_layers=1,
        ),
    )


def _decision(seed: int = 3, action_count: int = 4):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    infoset = copy.deepcopy(env.infoset)
    infoset.legal_actions = infoset.legal_actions[:action_count]
    observation = get_obs_v2(infoset, ruleset=RuleSet.legacy())
    privileged = PrivilegedObservation(
        all_handcards=infoset.all_handcards,
        acting_role=infoset.player_position,
    )
    pending = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=f"episode-{seed}",
        deal_id=f"deal-{seed}",
        target_transform="raw",
    )
    transition = pending.finalize(2.0)
    sample = DistillationSample(
        public_observation=observation,
        privileged_observation=privileged,
        action_index=0,
        target_win=1.0,
        target_score=2.0,
        sample_id=f"sample-{seed}",
    ).tensorize()
    return observation, privileged, transition, sample


def _schedule() -> OracleGuidingScheduleConfig:
    return OracleGuidingScheduleConfig(
        enabled=True,
        warmup_updates=1,
        guided_updates=2,
        finetune_updates=3,
        guidance_weight_start=1.0,
        guidance_weight_end=0.0,
        temperature_start=2.0,
        temperature_end=1.0,
        privileged_gate_start=1.0,
        privileged_gate_end=0.0,
    )


def _learner(model=None, **overrides) -> V3H3Learner:
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        adaptive_dmc=AdaptiveDMCConfig(mode="disabled"),
    )
    values = {
        "public": public,
        "schedule": _schedule(),
        "guidance": OracleGuidanceLossConfig(
            lambda_kl=1.0,
            lambda_ranking=0.25,
            lambda_chosen_value=0.5,
        ),
        "oracle_hidden_size": 16,
        "oracle_learning_rate": 1e-3,
    }
    values.update(overrides)
    return V3H3Learner(
        model or _model(), ruleset=RuleSet.legacy(), config=V3H3LearnerConfig(**values)
    )


def _state_copy(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _state_changed(before, module) -> bool:
    return any(not torch.equal(before[name], value) for name, value in module.state_dict().items())


def test_schedule_is_update_based_and_reaches_exact_public_only_state():
    schedule = _schedule()
    warmup = schedule.at(0)
    guided_start = schedule.at(1)
    guided_end = schedule.at(2)
    finetune = schedule.at(3)
    complete = schedule.at(6)
    assert warmup.phase == ORACLE_PHASE_WARMUP
    assert warmup.public_training is False and warmup.privileged_required is True
    assert guided_start.phase == ORACLE_PHASE_GUIDED
    assert guided_start.guidance_weight == pytest.approx(1.0)
    assert guided_start.temperature == pytest.approx(2.0)
    assert guided_end.guidance_weight == 0.0
    assert guided_end.privileged_gate == 0.0
    assert guided_end.oracle_weight == 0.0
    assert finetune.phase == ORACLE_PHASE_PUBLIC_FINETUNE
    assert finetune.privileged_required is False
    assert finetune.oracle_weight == 0.0
    assert finetune.guidance_weight == 0.0
    assert complete.phase == ORACLE_PHASE_COMPLETE
    assert complete.public_training is False
    assert schedule.stable_hash() != dataclasses.replace(
        schedule, temperature_start=3.0
    ).stable_hash()

    single = dataclasses.replace(schedule, warmup_updates=0, guided_updates=1)
    only_guided = single.at(0)
    assert only_guided.phase == ORACLE_PHASE_GUIDED
    assert only_guided.oracle_weight == pytest.approx(single.oracle_weight_start)
    assert only_guided.guidance_weight == pytest.approx(single.guidance_weight_start)
    assert only_guided.temperature == pytest.approx(single.temperature_start)
    assert only_guided.privileged_gate == pytest.approx(
        single.privileged_gate_start
    )
    assert single.at(1).phase == ORACLE_PHASE_PUBLIC_FINETUNE


def test_disabled_h3_import_and_training_need_no_privileged_module_or_object():
    code = (
        "import sys; "
        "from douzero.v3_hybrid.training.h3_learner import V3H3LearnerConfig; "
        "assert 'douzero.observation.privileged' not in sys.modules; "
        "assert V3H3LearnerConfig().schedule.enabled is False"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    _, _, transition, _ = _decision()
    config = V3H3LearnerConfig(
        public=V3H2LearnerConfig(
            batch_size=2,
            adaptive_dmc=AdaptiveDMCConfig(mode="disabled"),
        )
    )
    learner = V3H3Learner(_model(), ruleset=RuleSet.legacy(), config=config)
    assert learner.oracle is None and learner.oracle_optimizer is None
    metrics = learner.train_batch([transition])
    assert metrics.public_updated is True
    assert metrics.oracle_updated is False


def test_oracle_reuses_p10_action_alignment_and_guidance_tensor_formula():
    observation, privileged, _, sample = _decision()
    student = _model()
    oracle = V3PrivilegedOracle(student, V3OracleConfig(hidden_size=16))
    public_output = student.forward_observation(observation)
    original = oracle(
        sample.public_inputs,
        privileged,
        action_keys=sample.action_keys,
    )
    reverse = torch.arange(len(sample.action_keys) - 1, -1, -1)
    teacher = type(original)(
        action_keys=tuple(reversed(original.action_keys)),
        win_logit=original.win_logit.index_select(0, reverse),
        p_win=original.p_win.index_select(0, reverse),
        score_if_win=original.score_if_win.index_select(0, reverse),
        score_if_loss=original.score_if_loss.index_select(0, reverse),
        expected_score=original.expected_score.index_select(0, reverse),
        action_logits=original.action_logits.index_select(0, reverse),
        action_mask=original.action_mask.index_select(0, reverse),
    )
    # A correctly reordered teacher result must align back to the student's
    # environment legal-action order rather than relying on row position.
    loss = oracle_guidance_loss(
        public_output,
        sample.action_keys,
        teacher,
        chosen_action_index=0,
        temperature=2.0,
        config=OracleGuidanceLossConfig(lambda_ranking=0.0, lambda_chosen_value=0.0),
    )
    teacher_logits = original.action_logits.squeeze(-1).detach()
    student_logits = public_output.dmc_q.squeeze(-1)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student_logits / 2.0, dim=0),
        torch.softmax(teacher_logits / 2.0, dim=0),
        reduction="sum",
    ) * 4.0
    assert torch.allclose(loss.kl, expected)


def test_guided_batch_trims_padding_to_each_real_legal_action_count():
    _, _, short_transition, short_sample = _decision(seed=3, action_count=2)
    _, _, long_transition, long_sample = _decision(seed=4, action_count=4)
    learner = _learner(
        schedule=dataclasses.replace(_schedule(), warmup_updates=0)
    )
    metrics = learner.train_batch(
        [short_transition, long_transition],
        oracle_samples=[short_sample, long_sample],
    )
    assert metrics.phase == ORACLE_PHASE_GUIDED
    assert metrics.public_updated is True
    assert metrics.action_agreement is not None


def test_disabled_dmc_has_no_q_old_dependency_and_scheduled_noops_advance():
    _, _, transition, _ = _decision()
    public = V3H2LearnerConfig(
        batch_size=2,
        lambda_dmc=0.0,
        adaptive_dmc=AdaptiveDMCConfig(mode="safe_hybrid"),
    )
    guidance_only = _learner(public=public, schedule=dataclasses.replace(
        _schedule(), warmup_updates=0
    ))
    # A plain transition has no q_old. Guidance needs privileged data, but the
    # disabled DMC term must not inspect adaptive replay provenance.
    _, _, _, sample = _decision()
    metrics = guidance_only.train_batch([transition], oracle_samples=[sample])
    assert metrics.public_updated is True
    assert guidance_only.statistics.dmc_updates == 0

    no_losses = _learner(
        public=public,
        schedule=dataclasses.replace(
            _schedule(), warmup_updates=0, oracle_weight_start=0.0
        ),
        guidance=OracleGuidanceLossConfig(
            lambda_kl=0.0, lambda_ranking=0.0, lambda_chosen_value=0.0
        ),
    )
    before = _state_copy(no_losses.model)
    no_op = no_losses.train_batch([transition])
    assert no_op.samples == 1
    assert no_op.public_updated is False and no_op.oracle_updated is False
    assert no_losses.learner_updates == 1
    assert not _state_changed(before, no_losses.model)
    for _ in range(4):
        no_losses.train_batch([transition])
    assert no_losses.schedule_state().phase == ORACLE_PHASE_COMPLETE
    with pytest.raises(RuntimeError, match="schedule is complete"):
        no_losses.train_batch([transition])


def test_oracle_warmup_accepts_unconsumed_adaptive_provenance():
    observation, _, _, sample = _decision()
    snapshot = _model().eval()
    lease = PolicyLease(
        slot=0,
        version=0,
        model=snapshot,
        owner_id=1,
        generation=1,
    )
    transition = capture_adaptive_transition(
        lease,
        observation,
        selected_action_index=0,
        episode_id="adaptive-warmup",
        deal_id="adaptive-warmup",
        target_transform="raw",
    ).finalize(2.0)
    learner = _learner(
        public=V3H2LearnerConfig(
            batch_size=2,
            adaptive_dmc=AdaptiveDMCConfig(mode="safe_hybrid"),
        )
    )
    metrics = learner.train_batch([transition], oracle_samples=[sample])
    assert metrics.phase == ORACLE_PHASE_WARMUP
    assert metrics.oracle_updated is True and metrics.public_updated is False

    guidance_only = _learner(
        public=V3H2LearnerConfig(
            batch_size=2,
            lambda_dmc=0.0,
            adaptive_dmc=AdaptiveDMCConfig(mode="safe_hybrid"),
        ),
        schedule=dataclasses.replace(_schedule(), warmup_updates=0),
    )
    guided = guidance_only.train_batch([transition], oracle_samples=[sample])
    assert guided.phase == ORACLE_PHASE_GUIDED
    assert guided.public_updated is True and guided.oracle_updated is True


def test_public_student_is_invariant_to_hidden_swap_and_oracle_is_separate():
    observation, privileged, _, sample = _decision()
    model = _model().eval()
    before = model.forward_observation(observation).dmc_q.detach().clone()
    hands = {role: list(cards) for role, cards in privileged.all_handcards.items()}
    left, right = hands["landlord_up"], hands["landlord_down"]
    pair = next((i, j) for i, a in enumerate(left) for j, b in enumerate(right) if a != b)
    left[pair[0]], right[pair[1]] = right[pair[1]], left[pair[0]]
    swapped = PrivilegedObservation(all_handcards=hands, acting_role=privileged.acting_role)
    after = model.forward_observation(observation).dmc_q.detach()
    assert torch.equal(before, after)
    assert all("privileged" not in name and "oracle" not in name for name in model.state_dict())
    oracle = V3PrivilegedOracle(model, V3OracleConfig(hidden_size=16)).eval()
    assert all(
        public.data_ptr() != private.data_ptr()
        for public, private in zip(model.parameters(), oracle.public_backbone.parameters())
    )
    original = oracle(sample.public_inputs, privileged, action_keys=sample.action_keys)
    changed = oracle(sample.public_inputs, swapped, action_keys=sample.action_keys)
    assert not torch.equal(original.action_logits, changed.action_logits)


def test_three_phase_training_keeps_student_independent_then_finishes_public_only():
    _, _, transition, sample = _decision()
    learner = _learner()
    student_before = _state_copy(learner.model)
    oracle_before = _state_copy(learner.oracle)
    warmup = learner.train_batch([transition], oracle_samples=[sample])
    assert warmup.phase == ORACLE_PHASE_WARMUP
    assert warmup.public_updated is False and warmup.oracle_updated is True
    assert not _state_changed(student_before, learner.model)
    assert _state_changed(oracle_before, learner.oracle)

    guided = learner.train_batch([transition], oracle_samples=[sample])
    assert guided.phase == ORACLE_PHASE_GUIDED
    assert guided.public_updated and guided.oracle_updated
    assert guided.action_agreement is not None
    assert guided.loss_kl >= 0.0 and guided.value_error_abs is not None
    annealed = learner.train_batch([transition])
    assert annealed.phase == ORACLE_PHASE_GUIDED
    assert annealed.oracle_weight == 0.0
    assert annealed.guidance_weight == 0.0
    oracle_before_finetune = _state_copy(learner.oracle)
    finetune = learner.train_batch([transition])
    assert finetune.phase == ORACLE_PHASE_PUBLIC_FINETUNE
    assert finetune.public_updated is True and finetune.oracle_updated is False
    assert not _state_changed(oracle_before_finetune, learner.oracle)
    with pytest.raises(ValueError, match="rejects privileged"):
        learner.train_batch([transition], oracle_samples=[sample])
    learner.train_batch([transition])
    learner.train_batch([transition])
    assert learner.schedule_state().phase == ORACLE_PHASE_COMPLETE
    final_student = _state_copy(learner.model)
    with pytest.raises(RuntimeError, match="schedule is complete"):
        learner.train_batch([transition])
    assert not _state_changed(final_student, learner.model)


def test_h3_checkpoint_resume_preserves_schedule_optimizers_policy_and_rng(tmp_path):
    _, _, transition, sample = _decision()
    learner = _learner()
    learner.train_batch([transition], oracle_samples=[sample])
    learner.train_batch([transition], oracle_samples=[sample])
    path = tmp_path / "h3-trainer.pt"
    learner.save_checkpoint(path)
    expected_state = learner.schedule_state().as_dict()
    restored = _learner()
    restored.load_checkpoint(path)
    assert restored.learner_updates == learner.learner_updates == 2
    assert restored.policy_version == learner.policy_version
    assert restored.schedule_state().as_dict() == expected_state
    assert restored.statistics.state_dict() == learner.statistics.state_dict()
    assert restored.student_optimizer.state_dict()["param_groups"] == learner.student_optimizer.state_dict()["param_groups"]
    assert restored.oracle_optimizer.state_dict()["param_groups"] == learner.oracle_optimizer.state_dict()["param_groups"]

    bad = _learner(
        guidance=OracleGuidanceLossConfig(lambda_kl=0.0, lambda_ranking=0.25)
    )
    with pytest.raises(CheckpointCompatibilityError, match="learner_config"):
        bad.load_checkpoint(path)

    corrupted_path = tmp_path / "h3-policy-drift.pt"
    corrupted = torch.load(path, map_location="cpu", weights_only=True)
    corrupted["counters"] = dict(corrupted["counters"])
    corrupted["counters"]["policy_version"] += 7
    torch.save(corrupted, corrupted_path)
    with pytest.raises(CheckpointCompatibilityError, match="policy version drift"):
        _learner().load_checkpoint(corrupted_path)


def test_public_replay_and_export_exclude_oracle_and_public_loader_rejects_trainer(tmp_path):
    _, _, transition, sample = _decision()
    serialized = transition.state_dict()
    flattened = repr(serialized).lower()
    assert "all_handcards" not in flattened
    assert "privileged" not in flattened
    learner = _learner()
    learner.train_batch([transition], oracle_samples=[sample])
    public_path = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(
        public_path, learner.model, ruleset=RuleSet.legacy()
    )
    public_bundle = torch.load(public_path, map_location="cpu", weights_only=True)
    assert all(
        "oracle" not in name and "privileged" not in name
        for name in public_bundle["state_dict"]
    )
    trainer_path = tmp_path / "trainer.pt"
    learner.save_checkpoint(trainer_path)
    trainer_bundle = torch.load(trainer_path, map_location="cpu", weights_only=True)
    assert trainer_bundle["artifact_access"] == "privileged_training_only"
    with pytest.raises(CheckpointCompatibilityError, match="envelope"):
        load_v3_hybrid_public_checkpoint(
            trainer_path,
            schema=build_v2_schema(),
            ruleset=RuleSet.legacy(),
            config=learner.model.config,
        )


def test_deployment_import_graph_does_not_import_oracle_or_privileged():
    code = (
        "import sys; "
        "import douzero.v3_hybrid; "
        "import douzero.evaluation.deep_agent; "
        "import douzero.search; "
        "assert 'douzero.v3_hybrid.training.oracle' not in sys.modules; "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h3_cuda_forward_backward_and_public_finetune():
    _, _, transition, sample = _decision()
    public = V3H2LearnerConfig(
        batch_size=2,
        learning_rate=1e-3,
        device="cuda",
        adaptive_dmc=AdaptiveDMCConfig(mode="disabled"),
    )
    learner = _learner(public=public)
    learner.train_batch([transition], oracle_samples=[sample])
    metrics = learner.train_batch([transition], oracle_samples=[sample])
    assert metrics.public_gradient_norm > 0.0
    assert metrics.oracle_gradient_norm > 0.0
