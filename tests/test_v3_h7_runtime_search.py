"""H7 async protocol and public-only selective-search regressions."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel, build_belief_input
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.search import SearchConfig
from douzero.training.async_single_gpu import (
    AsyncRequestCoordinator,
    PinnedObservationBatchStager,
    SharedObservationSlots,
)
from douzero.v3_hybrid.runtime import (
    V3_H7_REPLAY_PROTOCOL,
    V3_H7_REQUEST_PROTOCOL,
    V3H7RuntimeConfig,
    V3AsyncSingleGPUTrainer,
)
from douzero.v3_hybrid import (
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    V3H2LearnerConfig,
    V3HybridLossComposerConfig,
    V3HybridModel,
    V3HybridModelConfig,
)
from douzero.v3_hybrid.integration_config import (
    V3H6FeatureFlags,
    V3H6LearnerConfig,
    V3H6ResolvedConfig,
    V3H6TopologyConfig,
)
from douzero.v3_hybrid.training.h3_learner import V3H3LearnerConfig
from douzero.v3_hybrid.training.h4_learner import V3H4LearnerConfig
from douzero.v3_hybrid.training.h5_learner import V3H5LearnerConfig
from douzero.v3_hybrid.training.h6_learner import V3H6Learner
from douzero.v3_hybrid.benchmark import (
    H7_BENCHMARK_SCHEMA,
    H7_TOPOLOGIES,
    V3H7BenchmarkProtocol,
    validate_h7_benchmark_evidence,
)
from douzero.v3_hybrid.selective_search import (
    V3H7SearchGateConfig,
    V3SelectiveSearch,
)
from douzero.v3_hybrid.support_matrix import (
    RULESET_LEGACY,
    TOPOLOGY_ASYNC_SINGLE_GPU,
    validate_capability_support,
)


def _observation():
    env = Env("adp")
    env.reset()
    return get_obs_v2(env.infoset, ruleset=RuleSet.legacy())


def _output(actions: int):
    p_win = torch.linspace(0.6, 0.4, actions).unsqueeze(-1)
    score = torch.linspace(0.2, -0.2, actions).unsqueeze(-1)
    return SimpleNamespace(
        dmc_q=torch.linspace(0.05, 0.0, actions).unsqueeze(-1),
        win_logit=torch.logit(p_win),
        score_if_win=torch.ones_like(score),
        score_if_loss=-torch.ones_like(score),
        p_win=p_win,
        score_mean=score,
        action_mask=torch.ones(actions, dtype=torch.bool),
        num_actions=actions,
    )


def test_h7_runtime_identity_binds_protocol_and_topology_fields():
    config = V3H7RuntimeConfig()
    changed = replace(config, games_per_actor=5)
    assert config.request_protocol == V3_H7_REQUEST_PROTOCOL
    assert config.replay_protocol == V3_H7_REPLAY_PROTOCOL
    assert config.stable_hash() != changed.stable_hash()
    with pytest.raises(ValueError, match="unknown H7 request protocol"):
        replace(config, request_protocol="unknown")


def test_h7_support_matrix_enables_only_base_async_capabilities():
    for capability in ("role_model", "adaptive_dmc", "public_export"):
        validate_capability_support(
            capability,
            topology=TOPOLOGY_ASYNC_SINGLE_GPU,
            ruleset=RULESET_LEGACY,
            checkpoint_resume=True,
            export=True,
            deployment=True,
            search=False,
        )
    with pytest.raises(ValueError, match="does not support async_single_gpu"):
        validate_capability_support(
            "oracle",
            topology=TOPOLOGY_ASYNC_SINGLE_GPU,
            ruleset=RULESET_LEGACY,
            checkpoint_resume=True,
            export=False,
            deployment=False,
            search=False,
        )


def test_shared_protocol_keeps_v2_width_and_adds_explicit_v3_dmc_q():
    schema = build_v2_schema()
    v2 = SharedObservationSlots(schema, 1, 4)
    v3 = SharedObservationSlots(schema, 1, 4, output_width=6)
    assert v2.output_values.shape[-1] == 5
    assert v2.output_dmc_q is None
    assert v3.output_values.shape[-1] == 6
    assert v3.output_dmc_q.shape == (1, 4)
    with pytest.raises(ValueError, match="output width"):
        SharedObservationSlots(schema, 1, 4, output_width=7)


def test_v3_pinned_stager_preserves_six_output_channels(monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs["dtype"]))
    slots = SharedObservationSlots(build_v2_schema(), 2, 8, output_width=6)
    stager = PinnedObservationBatchStager(
        slots, max_batch_size=2, action_capacity=8
    )
    assert stager.output_values.shape == (2, 8, 6)


def test_coordinator_rejects_unknown_output_layout_before_workers():
    with pytest.raises(ValueError, match="output width"):
        AsyncRequestCoordinator(
            build_v2_schema(), num_slots=1, output_width=4
        )


def test_selective_search_disabled_is_exact_base_noop():
    observation = _observation()
    wrapper = V3SelectiveSearch(
        V3H7SearchGateConfig(enabled=False),
        SearchConfig(enabled=False),
        RuleSet.legacy(),
        search_compatible=True,
    )
    record = wrapper.select(
        observation=observation,
        model_output=_output(len(observation.actions.legal_actions)),
        base_action_index=1,
        belief_model=object(),
    )
    assert record.selected_action_index == 1
    assert record.fallback_reason == "disabled"
    assert wrapper.metrics.snapshot()["trigger_rate"] == 0.0
    changed = V3SelectiveSearch(
        V3H7SearchGateConfig(enabled=False, max_own_cards=9),
        SearchConfig(enabled=False),
        RuleSet.legacy(),
        search_compatible=True,
    )
    assert wrapper.stable_hash() != changed.stable_hash()


def test_package_flag_and_global_stop_fail_to_exact_base_action():
    observation = _observation()
    config = V3H7SearchGateConfig(enabled=True)
    search = SearchConfig(enabled=True)
    incompatible = V3SelectiveSearch(
        config, search, RuleSet.legacy(), search_compatible=False
    )
    record = incompatible.select(
        observation=observation,
        model_output=_output(len(observation.actions.legal_actions)),
        base_action_index=0,
        belief_model=object(),
    )
    assert record.selected_action_index == 0
    assert record.fallback_reason == "package_not_search_compatible"

    stopped = V3SelectiveSearch(
        config,
        search,
        RuleSet.legacy(),
        search_compatible=True,
        stop_requested=lambda: True,
    )
    assert stopped.select(
        observation=observation,
        model_output=_output(len(observation.actions.legal_actions)),
        base_action_index=0,
        belief_model=object(),
    ).fallback_reason == "global_stop"


def test_nonconserved_belief_fails_before_search():
    observation = _observation()
    belief_model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
    belief_output = belief_model([build_belief_input(observation.public)])
    belief_output.expected_counts = belief_output.expected_counts.copy()
    belief_output.expected_counts[0, 0] += 1.0
    wrapper = V3SelectiveSearch(
        V3H7SearchGateConfig(enabled=True),
        SearchConfig(enabled=True),
        RuleSet.legacy(),
        search_compatible=True,
    )
    record = wrapper.select(
        observation=observation,
        model_output=_output(len(observation.actions.legal_actions)),
        base_action_index=0,
        belief_model=belief_model,
        belief_output=belief_output,
    )
    assert record.selected_action_index == 0
    assert record.fallback_reason == "belief_not_conserved"


def test_search_exception_is_reported_and_returns_base(monkeypatch):
    observation = _observation()
    reported = []
    wrapper = V3SelectiveSearch(
        V3H7SearchGateConfig(
            enabled=True, max_total_cards=100, max_own_cards=100
        ),
        SearchConfig(enabled=True),
        RuleSet.legacy(),
        search_compatible=True,
        exception_reporter=reported.append,
    )
    monkeypatch.setattr(
        wrapper.search,
        "select",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("controlled")),
    )
    record = wrapper.select(
        observation=observation,
        model_output=_output(len(observation.actions.legal_actions)),
        base_action_index=2,
        belief_model=object(),
    )
    assert record.selected_action_index == 2
    assert record.fallback_reason == "search_exception:RuntimeError:controlled"
    assert isinstance(reported[0], RuntimeError)
    assert wrapper.metrics.snapshot()["fallback_counts"] == {
        "search_exception:RuntimeError:controlled": 1
    }


def test_nonfinite_public_value_falls_back_before_search():
    observation = _observation()
    output = _output(len(observation.actions.legal_actions))
    output.dmc_q[0] = float("nan")
    wrapper = V3SelectiveSearch(
        V3H7SearchGateConfig(enabled=True),
        SearchConfig(enabled=True),
        RuleSet.legacy(),
        search_compatible=True,
    )
    record = wrapper.select(
        observation=observation,
        model_output=output,
        base_action_index=0,
        belief_model=object(),
    )
    assert record.selected_action_index == 0
    assert record.fallback_reason == "non_finite_value"


def test_public_gate_is_invariant_to_hidden_hand_swap():
    env = Env("adp")
    env.reset()
    observation = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    output = _output(len(observation.actions.legal_actions))
    wrapper = V3SelectiveSearch(
        V3H7SearchGateConfig(enabled=True),
        SearchConfig(enabled=True, max_nodes=0),
        RuleSet.legacy(),
        search_compatible=True,
    )
    before = wrapper._gate_reasons(observation, output, None)
    hands = env.infoset.all_handcards
    hands["landlord_up"], hands["landlord_down"] = (
        hands["landlord_down"], hands["landlord_up"]
    )
    after = wrapper._gate_reasons(observation, output, None)
    assert before == after


def test_search_metrics_zero_denominators_are_finite():
    metrics = V3SelectiveSearch(
        V3H7SearchGateConfig(),
        SearchConfig(),
        RuleSet.legacy(),
        search_compatible=False,
    ).metrics.snapshot()
    assert metrics["trigger_rate"] == 0.0
    assert metrics["change_rate"] == 0.0
    assert all(
        np.isfinite(metrics[name])
        for name in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
    )


def _benchmark_protocol():
    return V3H7BenchmarkProtocol(
        source_git_sha="a" * 40,
        image_digest="sha256:" + "b" * 64,
        config_hash="c" * 64,
        model_identity_hash="d" * 64,
        trainer_identity_hash="e" * 64,
        replay_protocol_hash="f" * 64,
        gpu="gpu",
        driver="driver",
        pytorch="torch",
        cuda="cuda",
        cpu="cpu",
    )


def _benchmark_record(protocol, topology, repeat):
    return {
        "schema": H7_BENCHMARK_SCHEMA,
        "protocol_hash": protocol.stable_hash(),
        "topology": topology,
        "repeat": repeat,
        "seed": protocol.seeds[repeat],
        "measurement_seconds": 300.0,
        "checkpoint_path": f"checkpoint-{topology}-{repeat}.pt",
        "parameter_update_observed": True,
        "active_slots": 0,
        "in_flight": 0,
        "pending": 0,
        "games_per_second": 1.0,
        "decisions_per_second": 2.0,
        "transitions_per_second": 2.0,
        "learner_samples_per_second": 3.0,
        "optimizer_steps_per_second": 0.1,
        "requests_per_microbatch": 2.0,
        "legal_actions_per_batch": 16.0,
        "queue_wait_seconds": 1.0,
        "slot_read_seconds": 1.0,
        "collate_seconds": 1.0,
        "h2d_seconds": 1.0,
        "forward_seconds": 1.0,
        "d2h_seconds": 1.0,
        "publish_seconds": 1.0,
        "replay_drain_seconds": 1.0,
        "learner_throttle_seconds": 1.0,
        "actor_blocked_ratio": 0.1,
        "learner_data_wait_ratio": 0.2,
        "policy_lag_max": 2.0,
        "cpu_ram_bytes": 1.0,
        "shared_memory_bytes": 1.0,
        "vram_bytes": 1.0,
        "shutdown_seconds": 1.0,
    }


def test_h7_benchmark_requires_three_matched_repeats_per_topology():
    protocol = _benchmark_protocol()
    records = [
        _benchmark_record(protocol, topology, repeat)
        for topology in H7_TOPOLOGIES
        for repeat in range(3)
    ]
    validate_h7_benchmark_evidence(records, protocol)
    with pytest.raises(ValueError, match="all topology repetitions"):
        validate_h7_benchmark_evidence(records[:-1], protocol)
    corrupted = [dict(record) for record in records]
    corrupted[0]["in_flight"] = 1
    with pytest.raises(ValueError, match="did not quiesce"):
        validate_h7_benchmark_evidence(corrupted, protocol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_h7_cuda_async_update_checkpoint_resume_and_shutdown(tmp_path):
    model_config = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
    )
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        device="cuda",
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID),
    )
    h5 = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    learner_config = V3H6LearnerConfig(
        base=h5,
        losses=V3HybridLossComposerConfig(lambda_dmc=1.0),
        features=V3H6FeatureFlags(adaptive_dmc=True),
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    resolved = V3H6ResolvedConfig(model=model_config, learner=learner_config)
    runtime_config = V3H7RuntimeConfig(
        num_actors=1,
        games_per_actor=2,
        batch_size=4,
        replay_capacity=256,
        target_microbatch=2,
        environment_seed=123,
        action_seed=456,
    )
    model = V3HybridModel(build_v2_schema(), model_config)
    learner = V3H6Learner(
        model, ruleset=RuleSet.legacy(), config=resolved
    )
    runtime = V3AsyncSingleGPUTrainer(learner, resolved, runtime_config)
    checkpoint = tmp_path / "h7.pt"
    try:
        runtime.collect_episodes(1)
        before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        assert runtime.step() is not None
        assert any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        )
        runtime.save_training_checkpoint(
            str(checkpoint), long_running_state={"cycle": 1}
        )
        corrupted = tmp_path / "corrupted.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["stats"] = dict(payload["stats"])
        payload["stats"]["optimizer_steps"] = -1
        torch.save(payload, corrupted)
        before_rejection = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        with pytest.raises(ValueError, match="statistic optimizer_steps"):
            runtime.load_training_checkpoint(corrupted)
        assert all(
            torch.equal(before_rejection[name], value)
            for name, value in model.state_dict().items()
        )
        status = runtime.quiesce_cycle_boundary()
        assert status["active_slots"] == 0
        assert status["in_flight_slots"] == 0
        assert status["pending_requests"] == 0
    finally:
        runtime.shutdown()

    resumed_model = V3HybridModel(build_v2_schema(), model_config)
    resumed_learner = V3H6Learner(
        resumed_model, ruleset=RuleSet.legacy(), config=resolved
    )
    resumed = V3AsyncSingleGPUTrainer(
        resumed_learner, resolved, runtime_config
    )
    assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
    assert resumed.policy_step == runtime.policy_step
    assert resumed.stats.optimizer_steps == 1
    resumed.shutdown()
    assert not list(tmp_path.glob("*.tmp"))
