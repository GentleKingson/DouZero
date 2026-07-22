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
)
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
