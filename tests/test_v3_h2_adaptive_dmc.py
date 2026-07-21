"""H2 Adaptive DMC formulas, replay provenance, learner, and resume tests."""

from __future__ import annotations

import copy
import dataclasses
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.runtime.policy_snapshot import PolicyLease
from douzero.v3_hybrid import (
    ADMC_DISABLED,
    ADMC_PAPER_RATIO,
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    AdaptiveSnapshotProvenance,
    V3H2Learner,
    V3H2LearnerConfig,
    V3HybridModel,
    V3HybridModelConfig,
    V3ReplayBuffer,
    V3ReplayTransition,
    adaptive_dmc_loss,
    capture_adaptive_transition,
    capture_plain_transition,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
    transform_dmc_target,
)


def _config(**overrides) -> V3HybridModelConfig:
    values = {
        "hidden_size": 32,
        "history_layers": 1,
        "history_heads": 4,
        "shared_fusion_layers": 1,
        "landlord_adapter_layers": 1,
        "farmer_adapter_layers": 1,
    }
    values.update(overrides)
    return V3HybridModelConfig(**values)


def _model(**overrides) -> V3HybridModel:
    torch.manual_seed(20260721)
    return V3HybridModel(build_v2_schema(), _config(**overrides))


def _observation(seed: int, role: str):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(80):
        if env._acting_player_position == role:
            return get_obs_v2(env.infoset)
        env.step(env.infoset.legal_actions[0])
    raise AssertionError(f"could not reach {role}")


def _plain_transition(seed: int, role: str, mc_return: float = 1.0):
    observation = _observation(seed, role)
    return capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=f"episode-{seed}",
        deal_id=f"deal-{seed}",
        target_transform="raw",
    ).finalize(mc_return)


def _adaptive_transition(
    snapshot: V3HybridModel,
    seed: int,
    role: str,
    *,
    policy_version: int = 0,
    mc_return: float = 1.0,
):
    lease = PolicyLease(
        slot=1,
        version=policy_version,
        model=snapshot,
        owner_id=3,
        generation=5,
    )
    return capture_adaptive_transition(
        lease,
        _observation(seed, role),
        selected_action_index=0,
        episode_id=f"episode-{seed}",
        deal_id=f"deal-{seed}",
        target_transform=snapshot.config.dmc_target_transform,
    ).finalize(mc_return)


def _learner(mode: str, *, model=None, **config_overrides) -> V3H2Learner:
    model = model or _model()
    adaptive = config_overrides.pop(
        "adaptive_dmc",
        AdaptiveDMCConfig(
            mode=mode,
            gamma_start=0.2,
            gamma_end=0.1,
            gamma_schedule_updates=10,
            epsilon=0.01,
            delta=0.2,
        ),
    )
    config = V3H2LearnerConfig(
        batch_size=8,
        adaptive_dmc=adaptive,
        **config_overrides,
    )
    return V3H2Learner(model, ruleset=RuleSet.legacy(), config=config)


def test_admc_config_identity_and_schedule_are_complete():
    baseline = AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID)
    assert baseline.gamma_at(0) == pytest.approx(0.2)
    assert baseline.gamma_at(50_000) == pytest.approx(0.125)
    assert baseline.gamma_at(100_000) == pytest.approx(0.05)
    assert baseline.gamma_at(200_000) == pytest.approx(0.05)
    assert AdaptiveDMCConfig.from_dict(dataclasses.asdict(baseline)) == baseline
    for name, value in {
        "mode": ADMC_PAPER_RATIO,
        "gamma_start": 0.3,
        "gamma_end": 0.1,
        "gamma_schedule_updates": 20,
        "epsilon": 0.02,
        "delta": 0.3,
    }.items():
        assert dataclasses.replace(baseline, **{name: value}).stable_hash() != baseline.stable_hash()


def test_paper_ratio_matches_independent_tensor_oracle():
    q_old = torch.tensor([2.0, -2.0])
    q_new = torch.tensor([3.0, -1.0], requires_grad=True)
    target = torch.tensor([2.0, -2.0])
    result = adaptive_dmc_loss(
        q_new,
        target,
        config=AdaptiveDMCConfig(
            mode=ADMC_PAPER_RATIO,
            gamma_start=0.2,
            gamma_end=0.2,
            gamma_schedule_updates=0,
        ),
        target_transform="raw",
        target_clamp=32.0,
        learner_update=9,
        q_old=q_old,
    )
    expected_ratio = torch.tensor([1.5, 0.5])
    expected_prediction = torch.tensor([2.4, -1.6])
    expected_loss = torch.tensor([(2.4 - 2.0) ** 2, (-1.6 + 2.0) ** 2])
    assert torch.allclose(result.ratio, expected_ratio)
    assert torch.allclose(result.constrained_q, expected_prediction)
    assert torch.allclose(result.loss_per_sample, expected_loss)
    assert result.ratio_clipped.tolist() == [True, True]


def test_safe_hybrid_matches_independent_near_zero_oracle():
    q_old = torch.tensor([2.0, -2.0, 0.0001, -0.0001])
    q_new = torch.tensor([3.0, -1.0, 0.5, -0.5], requires_grad=True)
    result = adaptive_dmc_loss(
        q_new,
        torch.zeros(4),
        config=AdaptiveDMCConfig(
            mode=ADMC_SAFE_HYBRID,
            gamma_start=0.2,
            gamma_end=0.2,
            gamma_schedule_updates=0,
            epsilon=0.01,
            delta=0.2,
        ),
        target_transform="raw",
        target_clamp=32.0,
        learner_update=0,
        q_old=q_old,
    )
    expected = torch.tensor([2.4, -1.6, 0.2001, -0.2001])
    assert torch.allclose(result.constrained_q, expected, atol=1e-6)
    assert result.near_zero_fallback.tolist() == [False, False, True, True]
    assert result.ratio_clipped.tolist() == [True, True, False, False]


def test_disabled_is_exact_ordinary_mse_without_q_old_dependency():
    q_new = torch.tensor([1.5, -0.5], requires_grad=True)
    target = torch.tensor([1.0, -1.0])
    result = adaptive_dmc_loss(
        q_new,
        target,
        config=AdaptiveDMCConfig(mode=ADMC_DISABLED),
        target_transform="raw",
        target_clamp=32.0,
        learner_update=0,
    )
    assert torch.equal(result.constrained_q, q_new)
    assert torch.allclose(result.loss_per_sample, (q_new - target) ** 2)
    result.loss_per_sample.mean().backward()
    assert torch.allclose(q_new.grad, torch.tensor([0.5, 0.5]))


def test_signed_log_target_and_clamp_match_independent_values():
    target, clamped = transform_dmc_target(
        torch.tensor([0.0, 3.0, -3.0, 1_000.0]),
        transform="signed_log",
        clamp=2.0,
    )
    expected = torch.tensor([0.0, math.log(4.0), -math.log(4.0), 2.0])
    assert torch.allclose(target, expected)
    assert clamped.tolist() == [False, False, False, True]


def test_negative_q_old_sign_is_not_destroyed_by_ratio_clipping():
    result = adaptive_dmc_loss(
        torch.tensor([-4.0]),
        torch.tensor([-3.0]),
        config=AdaptiveDMCConfig(
            mode=ADMC_PAPER_RATIO,
            gamma_start=0.2,
            gamma_end=0.2,
            gamma_schedule_updates=0,
        ),
        target_transform="raw",
        target_clamp=32.0,
        learner_update=0,
        q_old=torch.tensor([-2.0]),
    )
    assert result.constrained_q.item() == pytest.approx(-2.4)


def test_nonfinite_inputs_fail_closed_and_paper_zero_records_fallback():
    with pytest.raises(FloatingPointError, match="q_new"):
        adaptive_dmc_loss(
            torch.tensor([float("nan")]),
            torch.tensor([0.0]),
            config=AdaptiveDMCConfig(),
            target_transform="raw",
            target_clamp=32.0,
            learner_update=0,
        )
    result = adaptive_dmc_loss(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        config=AdaptiveDMCConfig(mode=ADMC_PAPER_RATIO),
        target_transform="raw",
        target_clamp=32.0,
        learner_update=0,
        q_old=torch.tensor([0.0]),
    )
    assert result.non_finite_fallback.item()
    assert result.constrained_q.item() == 0.0


def test_q_old_is_captured_from_the_exact_eval_snapshot():
    snapshot = _model().eval()
    observation = _observation(301, "landlord_up")
    lease = PolicyLease(
        slot=2, version=7, model=snapshot, owner_id=4, generation=9
    )
    expected = float(snapshot.forward_observation(observation).dmc_q[0, 0].item())
    pending = capture_adaptive_transition(
        lease,
        observation,
        selected_action_index=0,
        episode_id="episode-301",
        deal_id="deal-301",
        target_transform="raw",
    )
    assert pending.adaptive_provenance.q_old == pytest.approx(expected)
    assert pending.adaptive_provenance.policy_version == 7
    with torch.no_grad():
        next(snapshot.parameters()).add_(100.0)
    transition = pending.finalize(-2.0)
    assert transition.adaptive_provenance.q_old == pytest.approx(expected)

    with pytest.raises(ValueError, match="eval mode"):
        capture_adaptive_transition(
            dataclasses.replace(lease, model=_model().train()),
            observation,
            selected_action_index=0,
            episode_id="episode-302",
            deal_id="deal-302",
            target_transform="raw",
        )


def test_replay_modes_serialization_and_old_schema_fail_closed():
    plain = _plain_transition(302, "landlord")
    ordinary = V3ReplayBuffer(
        4,
        feature_schema_hash=build_v2_schema().stable_hash(),
        target_transform="raw",
        adaptive_required=False,
    )
    ordinary.add(plain)
    restored = V3ReplayBuffer.from_state_dict(ordinary.state_dict())
    assert len(restored) == 1
    assert restored.sample(1, rng=__import__("random").Random(3))[0].deal_id == plain.deal_id

    old = ordinary.state_dict()
    old["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported"):
        V3ReplayBuffer.from_state_dict(old)

    adaptive = V3ReplayBuffer(
        4,
        feature_schema_hash=build_v2_schema().stable_hash(),
        target_transform="raw",
        adaptive_required=True,
    )
    with pytest.raises(ValueError, match="missing actor-snapshot"):
        adaptive.add(plain)
    snapshot = _model().eval()
    adaptive_row = _adaptive_transition(snapshot, 303, "landlord_down")
    adaptive.add(adaptive_row)
    with pytest.raises(ValueError, match="must not depend"):
        ordinary.add(adaptive_row)


def test_replay_import_and_payload_remain_public_only():
    probe = (
        "import sys; "
        "assert 'douzero.observation.privileged' not in sys.modules; "
        "import douzero.v3_hybrid.replay; "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
    payload = _plain_transition(304, "landlord_up").state_dict()
    encoded = repr(payload).lower()
    assert "all_handcards" not in encoded
    assert "hidden_hand" not in encoded
    assert "privileged" not in encoded


def test_ordinary_learner_updates_selected_actions_and_reports_roles_once():
    learner = _learner(
        ADMC_DISABLED,
        landlord_weight=2.0,
        landlord_up_weight=3.0,
        landlord_down_weight=4.0,
    )
    transitions = [
        _plain_transition(310, "landlord", 2.0),
        _plain_transition(311, "landlord_up", -1.0),
        _plain_transition(312, "landlord_down", 1.0),
    ]
    before = {name: value.detach().clone() for name, value in learner.model.named_parameters()}
    metrics = learner.train_batch(transitions)
    assert metrics.samples == 3
    assert metrics.role_samples == {
        "landlord": 1, "landlord_up": 1, "landlord_down": 1
    }
    assert metrics.role_effective_weights == pytest.approx({
        "landlord": 2.0, "landlord_up": 3.0, "landlord_down": 4.0
    })
    assert metrics.q_old_mean is None
    assert metrics.gradient_norm > 0.0
    assert learner.learner_updates == 1
    assert learner.policy_version == 1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in learner.model.named_parameters()
    )


def test_safe_hybrid_learner_metrics_and_policy_lag():
    model = _model()
    snapshot = copy.deepcopy(model).eval()
    learner = _learner(ADMC_SAFE_HYBRID, model=model, initial_policy_version=3)
    records = [
        _adaptive_transition(snapshot, 320, "landlord", policy_version=1),
        _adaptive_transition(snapshot, 321, "landlord_up", policy_version=2),
    ]
    records[0] = dataclasses.replace(
        records[0],
        adaptive_provenance=dataclasses.replace(
            records[0].adaptive_provenance, q_old=0.0
        ),
    )
    metrics = learner.train_batch(records)
    assert metrics.near_zero_fallback_fraction == pytest.approx(0.5)
    assert metrics.max_policy_lag == 2
    assert metrics.q_old_mean is not None
    assert learner.statistics.near_zero_count == 1
    assert learner.statistics.steps == 1


def test_lambda_dmc_zero_is_an_exact_noop_without_replay_dependency():
    learner = _learner(ADMC_SAFE_HYBRID, lambda_dmc=0.0)
    before = copy.deepcopy(learner.model.state_dict())
    optimizer_before = copy.deepcopy(learner.optimizer.state_dict())
    metrics = learner.train_batch(None)
    assert metrics.samples == 0
    assert learner.learner_updates == 0
    assert learner.policy_version == 0
    assert learner.optimizer.state_dict() == optimizer_before
    assert all(torch.equal(before[name], value) for name, value in learner.model.state_dict().items())


def test_checkpoint_resume_preserves_schedule_optimizer_policy_rng_and_stats(tmp_path):
    model = _model()
    snapshot = copy.deepcopy(model).eval()
    learner = _learner(ADMC_SAFE_HYBRID, model=model)
    batch = [
        _adaptive_transition(snapshot, 330, "landlord", policy_version=0),
        _adaptive_transition(snapshot, 331, "landlord_up", policy_version=0),
    ]
    learner.train_batch(batch)
    path = tmp_path / "h2.ckpt"
    learner.save_checkpoint(path)

    resumed = _learner(ADMC_SAFE_HYBRID, model=_model())
    resumed.load_checkpoint(path)
    assert resumed.learner_updates == learner.learner_updates == 1
    assert resumed.samples_consumed == learner.samples_consumed == 2
    assert resumed.policy_version == learner.policy_version == 1
    assert resumed.statistics.state_dict() == learner.statistics.state_dict()
    assert resumed.config.adaptive_dmc.gamma_at(1) == learner.config.adaptive_dmc.gamma_at(1)
    assert resumed.optimizer.state_dict()["param_groups"] == learner.optimizer.state_dict()["param_groups"]
    assert all(
        torch.equal(learner.model.state_dict()[name], resumed.model.state_dict()[name])
        for name in learner.model.state_dict()
    )


def test_checkpoint_identity_partial_load_and_public_loader_fail_closed(tmp_path):
    learner = _learner(ADMC_SAFE_HYBRID)
    path = tmp_path / "h2.ckpt"
    learner.save_checkpoint(path)

    changed = _learner(
        ADMC_SAFE_HYBRID,
        model=_model(),
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID, epsilon=0.02),
    )
    with pytest.raises(CheckpointCompatibilityError, match="learner_config"):
        changed.load_checkpoint(path)

    bundle = torch.load(path, weights_only=True)
    bundle["surprise"] = torch.zeros(1)
    poisoned = tmp_path / "partial.ckpt"
    torch.save(bundle, poisoned)
    with pytest.raises(CheckpointCompatibilityError, match="envelope"):
        learner.load_checkpoint(poisoned)

    with pytest.raises(CheckpointCompatibilityError, match="envelope"):
        load_v3_hybrid_public_checkpoint(
            path,
            schema=build_v2_schema(),
            ruleset=RuleSet.legacy(),
            config=_config(),
        )


def test_h2_trained_model_exports_as_strict_public_h1_sidecar(tmp_path):
    learner = _learner(ADMC_DISABLED)
    learner.train_batch([_plain_transition(340, "landlord", 1.0)])
    path = tmp_path / "public.ckpt"
    save_v3_hybrid_public_checkpoint(path, learner.model, ruleset=RuleSet.legacy())
    payload = torch.load(path, weights_only=True)
    assert "optimizer_state_dict" not in payload
    assert "adaptive_statistics" not in payload
    assert all(
        not any(token in name.lower() for token in ("privileged", "teacher", "oracle"))
        for name in payload["state_dict"]
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_safe_hybrid_forward_backward_and_checkpoint(tmp_path):
    model = _model()
    snapshot = copy.deepcopy(model).eval()
    learner = _learner(ADMC_SAFE_HYBRID, model=model, device="cuda")
    batch = [
        _adaptive_transition(snapshot, 350, "landlord", policy_version=0),
        _adaptive_transition(snapshot, 351, "landlord_down", policy_version=0),
    ]
    metrics = learner.train_batch(batch)
    assert metrics.gradient_norm > 0.0
    assert next(learner.model.parameters()).is_cuda
    path = tmp_path / "cuda-h2.ckpt"
    learner.save_checkpoint(path)
    resumed = _learner(ADMC_SAFE_HYBRID, model=_model(), device="cuda")
    resumed.load_checkpoint(path)
    assert resumed.policy_version == learner.policy_version
