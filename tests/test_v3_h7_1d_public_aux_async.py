from __future__ import annotations

import copy
import inspect
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.strategy.config import StrategyFeatureConfig
from douzero.training.async_single_gpu import (
    SharedObservationSlots,
    SharedReplaySlots,
    async_actor_main,
)
from douzero.training.v2_buffer import Transition
from douzero.v3_hybrid.runtime import (
    V3_H71D_REPLAY_PROTOCOL,
    V3_H71D_REQUEST_PROTOCOL,
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.benchmark import (
    H7_TOPOLOGIES,
    V3H7BenchmarkProtocol,
    validate_h7_benchmark_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _observation(seed: int = 712):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    infoset = copy.deepcopy(env.infoset)
    infoset.legal_actions = infoset.legal_actions[:4]
    return get_obs_v2(infoset, ruleset=RuleSet.legacy())


def _resolved():
    from douzero.v3_hybrid.integration_config import load_v3_hybrid_config

    return load_v3_hybrid_config(
        ROOT / "configs/v3_hybrid_h7_1d_public_aux.yaml"
    )


def _runtime(**changes):
    return replace(
        V3H7RuntimeConfig(
            batch_size=8,
            public_aux_runtime_enabled=True,
            request_protocol=V3_H71D_REQUEST_PROTOCOL,
            replay_protocol=V3_H71D_REPLAY_PROTOCOL,
        ),
        **changes,
    )


def test_public_aux_runtime_fails_closed_before_cuda_or_worker_start():
    resolved = _resolved()
    validate_v3_h7_runtime_config(resolved, _runtime())
    with pytest.raises(ValueError, match="public auxiliary transport"):
        validate_v3_h7_runtime_config(
            resolved, V3H7RuntimeConfig(batch_size=8)
        )
    with pytest.raises(ValueError, match="request protocol"):
        V3H7RuntimeConfig(public_aux_runtime_enabled=True)
    with pytest.raises(NotImplementedError, match="combined async"):
        V3H7RuntimeConfig(
            belief_runtime_enabled=True,
            public_aux_runtime_enabled=True,
            request_protocol=V3_H71D_REQUEST_PROTOCOL,
            replay_protocol=V3_H71D_REPLAY_PROTOCOL,
        )


def test_shared_slots_are_exact_noop_when_public_aux_is_disabled():
    slots = SharedObservationSlots(build_v2_schema(), 2, max_actions=8)
    assert slots.strategy_features is None
    assert slots.style_features is None
    plain = observation_to_model_inputs(_observation())
    slots.write(0, plain)
    restored = slots.read_bundle(0, plain.feature_schema_hash)
    assert restored.strategy_features is None
    assert restored.style_features is None

    enriched = observation_to_model_inputs(
        _observation(),
        StrategyFeatureConfig(),
        style_enabled=True,
    )
    with pytest.raises(ValueError, match="strategy feature contract"):
        slots.write(1, enriched)


def test_public_feature_cache_and_strategy_labels_round_trip_exactly():
    observation = _observation()
    bundle = observation_to_model_inputs(
        observation,
        StrategyFeatureConfig(),
        style_enabled=True,
    )
    slots = SharedReplaySlots(
        build_v2_schema(),
        2,
        max_actions=8,
        v3_provenance=True,
        strategy_features=True,
        style_features=True,
    )
    transition = Transition(
        obs=observation,
        action_index=0,
        position=observation.public.acting_role,
        trace_index=3,
    )
    transition.actor_q_old = 0.25
    transition.actor_id = 2
    transition.episode_id = 9
    transition.target_win = 1.0
    transition.target_score = 2.0
    transition.target_log_score = 2.0
    transition.target_min_turns_after = 3.0
    transition.target_min_turns_exact_mask = 1.0
    transition.target_regain_initiative = 0.0
    transition.target_teammate_finish = 1.0
    transition.target_teammate_finish_mask = 1.0
    transition.target_spring_probability = 0.0
    transition.target_structure_cost = 4.0
    slots.write_transition(transition, bundle, 7, 1.0)
    try:
        deadline = time.monotonic() + 1.0
        records = []
        while not records and time.monotonic() < deadline:
            records = slots.read_ready_v3_aligned(
                feature_schema_hash=bundle.feature_schema_hash,
                target_transform="raw",
                ruleset_identity=RuleSet.legacy().identity(),
                include_strategy_targets=True,
            )
    finally:
        slots.close()
    assert len(records) == 1
    row, key, targets = records[0]
    assert key.actor_id == 2
    assert key.episode_id == 9
    assert torch.equal(
        row.model_inputs.strategy_features, bundle.strategy_features
    )
    assert torch.equal(row.model_inputs.style_features, bundle.style_features)
    assert targets == {
        "min_turns_after": 3.0,
        "min_turns_exact_mask": 1.0,
        "regain_initiative": 0.0,
        "teammate_finish": 1.0,
        "teammate_finish_mask": 1.0,
        "spring_probability": 0.0,
        "structure_cost": 4.0,
    }


def test_actor_reuses_served_public_bundle_and_never_reads_hidden_style_data():
    source = inspect.getsource(async_actor_main)
    assert "transition.public_model_inputs = public_inputs" in source
    assert "replay_slots.write_transition" in source
    assert "build_style_features" not in source
    assert "all_handcards" not in inspect.getsource(
        __import__(
            "douzero.style.features", fromlist=["build_style_features"]
        ).build_style_features
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h71d_cuda_update_checkpoint_resume_and_shutdown(tmp_path):
    from douzero.observation.schema import build_v2_schema
    from douzero.v3_hybrid import V3HybridModel
    from douzero.v3_hybrid.training.h6_learner import V3H6Learner

    resolved = _resolved()
    runtime_config = _runtime(
        batch_size=32,
        num_actors=1,
        games_per_actor=1,
        replay_capacity=256,
    )
    checkpoint = tmp_path / "public-aux-async.pt"

    def build():
        model = V3HybridModel(build_v2_schema(), resolved.model)
        learner = V3H6Learner(
            model,
            ruleset=RuleSet.legacy(),
            config=resolved,
        )
        return learner, V3AsyncSingleGPUTrainer(
            learner, resolved, runtime_config
        )

    learner, trainer = build()
    try:
        trainer.collect_episodes(8)
        before = trainer._parameter_update_snapshot()
        metrics = trainer.step()
        assert metrics is not None
        assert metrics.public_aux_updated
        assert trainer.stats.strategy_optimizer_steps == 1
        assert (
            trainer.stats.strategy_labels_collected
            == trainer.stats.transitions_collected
        )
        assert trainer._parameters_changed_since(before)
        boundary = trainer.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["in_flight_slots"] == 0
        assert boundary["pending_requests"] == 0
        assert boundary["public_aux_parameter_vram_bytes"] > 0
        trainer.save_training_checkpoint(
            str(checkpoint), long_running_state={"cycle": 1}
        )
    finally:
        trainer.shutdown()

    resumed_learner, resumed = build()
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert resumed.stats.strategy_optimizer_steps == 1
        resumed.collect_episodes(8)
        metrics = resumed.step()
        assert metrics is not None
        assert metrics.public_aux_updated
        assert resumed.stats.strategy_optimizer_steps == 2
        boundary = resumed.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["in_flight_slots"] == 0
        assert boundary["pending_requests"] == 0
    finally:
        resumed.shutdown()
    assert not list(tmp_path.glob("*.tmp"))


def test_primary_h7_cli_selects_public_aux_protocol_before_cuda(
    monkeypatch, tmp_path
):
    import train_v3_h7

    captured = {}

    def capture(_resolved, runtime):
        captured["runtime"] = runtime
        raise RuntimeError("validated-before-cuda")

    monkeypatch.setattr(train_v3_h7, "load_v3_hybrid_config", lambda _path: _resolved())
    monkeypatch.setattr(train_v3_h7, "validate_v3_h7_runtime_config", capture)
    monkeypatch.setattr(sys, "argv", [
        "train_v3_h7.py",
        "--config",
        str(tmp_path / "public-aux.yaml"),
        "--checkpoint-path",
        str(tmp_path / "checkpoint"),
    ])
    with pytest.raises(RuntimeError, match="validated-before-cuda"):
        train_v3_h7.main()
    runtime = captured["runtime"]
    assert runtime.public_aux_runtime_enabled is True
    assert runtime.request_protocol == V3_H71D_REQUEST_PROTOCOL
    assert runtime.replay_protocol == V3_H71D_REPLAY_PROTOCOL


def test_public_aux_benchmark_requires_positive_current_protocol_metrics():
    protocol = V3H7BenchmarkProtocol(
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
        formal_config_hash=None,
        oracle_enabled=False,
        public_aux_enabled=True,
        measurement_seconds=1.0,
    )
    records = []
    for topology in H7_TOPOLOGIES:
        for repeat, seed in enumerate(protocol.seeds):
            records.append({
                "schema": protocol.identity()["schema"],
                "protocol_hash": protocol.stable_hash(),
                "topology": topology,
                "repeat": repeat,
                "seed": seed,
                "measurement_seconds": 1.0,
                "checkpoint_path": f"{topology}-{repeat}.pt",
                "parameter_update_observed": True,
                "active_slots": 0,
                "in_flight": 0,
                "pending": 0,
                "games_per_second": 1.0,
                "decisions_per_second": 1.0,
                "transitions_per_second": 1.0,
                "learner_samples_per_second": 1.0,
                "optimizer_steps_per_second": 1.0,
                "oracle_samples_per_second": 0.0,
                "oracle_optimizer_steps_per_second": 0.0,
                "oracle_parameter_vram_bytes": 0.0,
                "cooperation_samples_per_second": 0.0,
                "cooperation_episodes_per_second": 0.0,
                "cooperation_optimizer_steps_per_second": 0.0,
                "cooperation_parameter_vram_bytes": 0.0,
                "strategy_samples_per_second": 1.0,
                "strategy_optimizer_steps_per_second": 1.0,
                "public_aux_parameter_vram_bytes": 1.0,
                "requests_per_microbatch": 2.0,
                "legal_actions_per_batch": 8.0,
                "queue_wait_seconds": 0.1,
                "slot_read_seconds": 0.1,
                "collate_seconds": 0.1,
                "h2d_seconds": 0.1,
                "forward_seconds": 0.1,
                "d2h_seconds": 0.1,
                "publish_seconds": 0.1,
                "replay_drain_seconds": 0.1,
                "learner_throttle_seconds": 0.0,
                "actor_blocked_ratio": 0.1,
                "learner_data_wait_ratio": 0.1,
                "policy_lag_max": 1.0,
                "cpu_ram_bytes": 1.0,
                "shared_memory_bytes": 1.0,
                "vram_bytes": 1.0,
                "shutdown_seconds": 0.1,
            })
    validate_h7_benchmark_evidence(records, protocol)
    records[0]["strategy_samples_per_second"] = 0.0
    with pytest.raises(ValueError, match="strategy_samples_per_second"):
        validate_h7_benchmark_evidence(records, protocol)
