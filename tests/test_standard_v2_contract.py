"""M0 regression gates for the Standard V2 single-GPU production plan."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from benchmarks.bench_standard_v2 import build_unified_benchmark
from benchmarks.standard_v2_reference import build_standard_v2_reference
from douzero.config import load_config
from douzero.env.rules import RuleSet
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    BiddingPolicyConfig,
    Episode,
    TrainerConfig,
    TrainerStats,
    V2Trainer,
)
from douzero.training.standard_v2_contract import (
    BASE_ASYNC_PROTOCOL_VERSION,
    STANDARD_ASYNC_PROTOCOL_VERSION,
    STANDARD_V2_R1_CONFIG_HASH,
    resolved_standard_v2_config_identity,
    stable_identity_hash,
    standard_v2_version_contract,
    validate_standard_v2_r1_config,
)
from train_v2 import (
    _build_decision_config,
    _build_loss_config,
    _build_model_cfg,
    _build_training_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    ROOT / "benchmarks" / "baselines" / "standard_v2_r1_reference.json"
)
GPU_BASELINE_PATH = (
    ROOT / "benchmarks" / "baselines" / "standard_v2_r1_single_gpu.json"
)


def _tiny_model() -> ModelV2:
    return ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=16, history_layers=1, history_heads=1),
    )


def _resolved_r1_config_identity(
    *,
    trainer_overrides: dict | None = None,
    loss_overrides: dict | None = None,
    bidding_overrides: dict | None = None,
) -> dict:
    config = load_config(str(ROOT / "configs" / "standard_v2.yaml"))
    trainer = TrainerConfig(
        seed=config.seed,
        batch_size=config.batch_size,
        exp_epsilon=config.exp_epsilon,
        learning_rate=config.optimizer.learning_rate,
        rmsprop_alpha=config.optimizer.alpha,
        rmsprop_momentum=config.optimizer.momentum,
        rmsprop_epsilon=config.optimizer.epsilon,
        max_grad_norm=config.max_grad_norm,
        amp_enabled=config.amp_enabled,
        amp_dtype=config.amp_dtype,
        amp_fallback_on_nonfinite=config.amp_fallback_on_nonfinite,
        first_bidder_mode=config.first_bidder_mode,
    )
    trainer = replace(trainer, **(trainer_overrides or {}))
    loss = replace(_build_loss_config(config), **(loss_overrides or {}))
    bidding = BiddingPolicyConfig(
        policy=config.bidding.policy,
        warm_start_policy=config.bidding.warm_start_policy,
        learned_probability=config.bidding.learned_probability,
    )
    bidding = replace(bidding, **(bidding_overrides or {}))
    return resolved_standard_v2_config_identity(
        trainer_config=trainer,
        model_config=_build_model_cfg(config),
        loss_config=loss,
        decision_config=_build_decision_config(config),
        bidding_config=bidding,
        ruleset=config.ruleset,
        feature_version=config.feature_version,
        model_version=config.model_version,
        deterministic=config.deterministic,
        ddp_enabled=config.ddp_enabled,
        ddp_backend=config.ddp_backend,
        compile_model=config.compile_model,
        bidding_enabled=config.bidding.enabled,
    )


def test_standard_v2_r1_yaml_matches_frozen_contract():
    identity = validate_standard_v2_r1_config(
        load_config(str(ROOT / "configs" / "standard_v2.yaml"))
    )
    assert stable_identity_hash(identity) == STANDARD_V2_R1_CONFIG_HASH
    assert _resolved_r1_config_identity() == identity


def test_standard_v2_reference_matches_checked_in_golden():
    expected = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    actual = build_standard_v2_reference()
    assert actual == expected
    digest_payload = dict(actual)
    digest = digest_payload.pop("reference_digest")
    assert stable_identity_hash(digest_payload) == digest
    assert actual["coverage"] == {
        "all_pass": True,
        "anti_spring": True,
        "bomb_or_rocket": True,
        "first_bidders": ["0", "1", "2"],
        "max_redeal_guard": True,
        "normal_auction": True,
        "redeal": True,
        "spring": True,
    }


def test_version_registry_names_current_and_reserved_protocols_distinctly():
    versions = standard_v2_version_contract()
    current = versions["current_base_async"]
    reserved = versions["reserved_standard_async"]
    assert current["protocol_version"] == BASE_ASYNC_PROTOCOL_VERSION == 1
    assert reserved["protocol_version"] == STANDARD_ASYNC_PROTOCOL_VERSION == 2
    assert current["episode_task_semantics"] != reserved["episode_task_semantics"]
    assert current["episode_commit_semantics"] != reserved["episode_commit_semantics"]
    assert current["compact_bidding_replay_schema_version"] == 0
    assert reserved["compact_bidding_replay_schema_version"] == 1
    with pytest.raises(ValueError, match="unknown async protocol"):
        TrainerConfig(async_protocol_version=STANDARD_ASYNC_PROTOCOL_VERSION)


def test_cap_guard_stats_include_all_bidding_decisions(monkeypatch):
    trainer = V2Trainer(
        ModelV2(
            build_v2_schema(),
            ModelV2Config(
                hidden_size=16,
                history_encoder="lstm",
                history_layers=1,
                history_heads=1,
                bidding_enabled=True,
                bidding_hidden_size=12,
            ),
        ),
        ruleset=RuleSet.standard(),
        config=TrainerConfig(max_episodes=1, optimizer_steps=0, batch_size=1),
    )
    episode = Episode(
        redeal_count=2,
        max_redeals_exceeded=True,
        abandoned_bidding_transitions=6,
        action_trace=[("landlord", (3,)), ("landlord_down", ())],
    )
    monkeypatch.setattr(trainer, "_run_one_episode", lambda: episode)
    trainer.collect_episodes(1)
    assert trainer.stats.games_collected == 1
    assert trainer.stats.decisions_collected == 2
    assert trainer.stats.bidding_decisions_collected == 6
    assert trainer.stats.abandoned_bidding_transitions == 6
    assert trainer.stats.redeals == 2
    assert trainer.stats.max_redeals_exceeded == 1
    assert trainer.stats.episodes_completed == 0


def test_unified_metric_shape_separates_decisions_and_trainable_samples():
    class Stats:
        games_collected = 3
        episodes_completed = 2
        decisions_collected = 90
        transitions_collected = 60
        bidding_decisions_collected = 12
        bidding_transitions_collected = 7
        abandoned_bidding_transitions = 5
        learner_cardplay_samples = 32
        learner_bidding_samples = 32
        optimizer_steps = 4
        redeals = 1
        max_redeals_exceeded = 1
        belief_supervised_steps = 0
        amp_fallbacks = 0

    report = _build_training_metrics(
        Stats(),
        config_identity=_resolved_r1_config_identity(),
        training_wall_seconds=2.0,
        device_type="cuda",
        peak_memory_bytes=8 * 1024 * 1024,
        peak_reserved_memory_bytes=10 * 1024 * 1024,
        world_size=1,
        parameters_changed=True,
        runtime_metrics={
            "inference_queue_p50_ms": 1.0,
            "inference_queue_p95_ms": 2.0,
            "inference_gpu_seconds": 0.2,
            "learner_gpu_seconds": 0.4,
            "collate_seconds": 0.1,
            "h2d_seconds": 0.2,
        },
    )
    standard = report["standard_v2"]
    assert standard["counts"] == {
        "games": 3,
        "cardplay_decisions": 90,
        "bidding_decisions": 12,
        "play_transitions": 60,
        "bid_transitions": 7,
        "abandoned_bidding_transitions": 5,
        "learner_cardplay_samples": 32,
        "learner_bidding_samples": 32,
        "learner_samples": 64,
        "learner_steps": 4,
    }
    assert standard["rates"]["games_per_second"] == 1.5
    assert standard["rates"]["bid_transitions_per_second"] == 3.5
    assert standard["rates"]["learner_samples_per_second"] == 32.0
    assert standard["staging_seconds"] == 0.3
    unified = build_unified_benchmark(training_metrics=report)
    assert unified["reference_digest"] == build_standard_v2_reference()[
        "reference_digest"
    ]
    assert set(unified["performance"]["rates"]) == {
        "games_per_second",
        "cardplay_decisions_per_second",
        "bidding_decisions_per_second",
        "play_transitions_per_second",
        "bid_transitions_per_second",
        "learner_samples_per_second",
        "learner_steps_per_second",
    }


def _benchmark_training_report(config_identity: dict | None = None) -> dict:
    class Stats:
        games_collected = 3
        episodes_completed = 2
        decisions_collected = 90
        transitions_collected = 60
        bidding_decisions_collected = 12
        bidding_transitions_collected = 7
        abandoned_bidding_transitions = 5
        learner_cardplay_samples = 32
        learner_bidding_samples = 32
        optimizer_steps = 4
        redeals = 1
        max_redeals_exceeded = 1
        belief_supervised_steps = 0
        amp_fallbacks = 0
        metrics_history_complete = True
        metrics_history_source = "native"

    return _build_training_metrics(
        Stats(),
        config_identity=config_identity or _resolved_r1_config_identity(),
        training_wall_seconds=2.0,
        device_type="cuda",
        peak_memory_bytes=8 * 1024 * 1024,
        peak_reserved_memory_bytes=10 * 1024 * 1024,
        world_size=1,
        parameters_changed=True,
        runtime_metrics={
            "collection_seconds": 1.5,
            "optimization_seconds": 0.5,
            "inference_queue_p50_ms": 1.0,
            "inference_queue_p95_ms": 2.0,
            "inference_gpu_seconds": 0.2,
            "learner_gpu_seconds": 0.4,
            "staging_seconds": 0.3,
        },
    )


@pytest.mark.parametrize(
    "identity_overrides",
    [
        {"trainer_overrides": {"batch_size": 1}},
        {"trainer_overrides": {"exp_epsilon": 0.2}},
        {"trainer_overrides": {"amp_enabled": True}},
        {"trainer_overrides": {"first_bidder_mode": "seeded_random"}},
        {"loss_overrides": {"lambda_win": 0.75}},
        {"bidding_overrides": {"policy": "rule"}},
    ],
    ids=(
        "batch-size",
        "epsilon",
        "amp",
        "first-bidder",
        "loss-weight",
        "bidding-policy",
    ),
)
def test_non_r1_resolved_config_cannot_emit_r1_metrics(identity_overrides):
    identity = _resolved_r1_config_identity(**identity_overrides)
    report = _benchmark_training_report(identity)
    assert report["benchmark_identity"] == {
        "schema_version": "standard-v2-r1-benchmark-v1",
        "contract_version": "standard-v2-r1-contract-v1",
        "config_hash": stable_identity_hash(identity),
        "reference_digest": None,
        "qualification": "non_r1",
    }
    assert "standard_v2" not in report


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "schema",
        "hash",
        "missing-count",
        "nan",
        "inf",
        "no-update",
        "incomplete-history",
    ],
)
def test_unified_benchmark_rejects_unbound_or_incomplete_training_input(case):
    report = _benchmark_training_report()
    if case == "empty":
        report = {}
    elif case == "schema":
        report["schema_version"] = "unknown"
    elif case == "hash":
        report["benchmark_identity"]["config_hash"] = "0" * 64
    elif case == "missing-count":
        del report["standard_v2"]["counts"]["games"]
    elif case == "nan":
        report["standard_v2"]["rates"]["games_per_second"] = float("nan")
    elif case == "inf":
        report["standard_v2"]["rates"]["games_per_second"] = float("inf")
    elif case == "no-update":
        report["parameter_update_observed"] = False
    elif case == "incomplete-history":
        report["metrics_history"]["complete"] = False
    with pytest.raises(ValueError):
        build_unified_benchmark(training_metrics=report)


def _benchmark_cycle_record() -> dict:
    identity = _benchmark_training_report()["benchmark_identity"]
    return {
        "schema_version": "v2-long-running-cycle-v2",
        "event": "cycle",
        "benchmark_identity": identity,
        "metrics_history": {"complete": True, "source": "native"},
        "device_type": "cuda",
        "amp": {
            "enabled": False,
            "dtype": "float16",
            "fallback_on_nonfinite": True,
        },
        "compile": {"enabled": False},
        "distributed": {"enabled": False, "world_size": 1},
        "parameter_update_observed": True,
        "cycle_games": 3,
        "cycle_cardplay_decisions": 90,
        "cycle_bidding_decisions": 12,
        "cycle_play_transitions": 60,
        "cycle_bid_transitions": 7,
        "cycle_abandoned_bidding_transitions": 5,
        "cycle_learner_cardplay_samples": 32,
        "cycle_learner_bidding_samples": 32,
        "cycle_learner_samples": 64,
        "cycle_learner_steps": 4,
        "cycle_wall_seconds": 2.0,
        "collection_seconds": 1.5,
        "optimization_seconds": 0.5,
        "games_per_second": 2.0,
        "cardplay_decisions_per_second": 60.0,
        "bidding_decisions_per_second": 8.0,
        "transitions_per_second": 40.0,
        "bid_transitions_per_second": 4.666667,
        "learner_samples_per_second": 128.0,
        "learner_steps_per_second": 8.0,
        "inference_queue_p50_ms": 1.0,
        "inference_queue_p95_ms": 2.0,
        "inference_gpu_seconds": 0.2,
        "learner_gpu_seconds": 0.4,
        "staging_seconds": 0.3,
        "peak_vram_bytes": 8 * 1024 * 1024,
        "amp_fallback": 0,
    }


def test_unified_benchmark_accepts_identity_bound_cycle_metrics():
    unified = build_unified_benchmark(cycle_metrics=_benchmark_cycle_record())
    assert unified["performance"]["status"] == "measured"
    assert unified["performance"]["counts"]["learner_steps"] == 4
    assert unified["performance"]["measurement"]["device_type"] == "cuda"


def test_unified_benchmark_rejects_cycle_identity_drift():
    cycle = _benchmark_cycle_record()
    cycle["benchmark_identity"]["config_hash"] = "f" * 64
    with pytest.raises(ValueError, match="frozen Standard V2 R1"):
        build_unified_benchmark(cycle_metrics=cycle)


def test_checked_in_gpu_baseline_is_bound_to_the_golden_reference():
    baseline = json.loads(GPU_BASELINE_PATH.read_text(encoding="utf-8"))
    reference = build_standard_v2_reference()
    assert baseline["schema_version"] == "standard-v2-r1-benchmark-v1"
    assert baseline["config_hash"] == STANDARD_V2_R1_CONFIG_HASH
    assert baseline["reference_digest"] == reference["reference_digest"]
    assert baseline["performance"]["status"] == "measured"
    assert baseline["performance"]["counts"]["games"] > 0
    assert baseline["performance"]["peak_vram_mib"] > 0


def test_historical_v3_stats_are_marked_as_partial_metrics_history():
    trainer = V2Trainer(
        _tiny_model(),
        config=TrainerConfig(max_episodes=0, optimizer_steps=0, batch_size=2),
    )
    stats = asdict(TrainerStats(episodes_completed=3, optimizer_steps=2))
    for name in (
        "games_collected",
        "bidding_decisions_collected",
        "abandoned_bidding_transitions",
        "learner_cardplay_samples",
        "learner_bidding_samples",
        "metrics_history_complete",
        "metrics_history_source",
    ):
        stats.pop(name)
    restored = trainer._restore_checkpoint_stats(stats, checkpoint_version=3)
    assert restored.games_collected == 3
    assert restored.learner_cardplay_samples == 4
    assert restored.metrics_history_complete is False
    assert restored.metrics_history_source == "migrated_v3_partial"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_async_checkpoint_v5_binds_protocol_and_loads_v4(tmp_path):
    config = TrainerConfig(
        max_episodes=0,
        optimizer_steps=0,
        batch_size=1,
        buffer_capacity=4,
        v2_training_mode="async_single_gpu",
        num_actors=1,
        games_per_actor=1,
        device="cuda",
    )
    trainer = V2Trainer(_tiny_model(), config=config)
    checkpoint = tmp_path / "async-v5.pt"
    identity = trainer.save_training_checkpoint(str(checkpoint))
    assert identity["checkpoint_version"] == 5
    assert identity["async_protocol_version"] == BASE_ASYNC_PROTOCOL_VERSION
    assert identity["compact_bidding_replay_schema_version"] == 0

    bundle = torch.load(checkpoint, map_location="cuda", weights_only=True)
    v4_config, v4_hash = trainer._v4_trainer_config_identity()
    bundle["checkpoint_version"] = 4
    bundle["trainer_config"] = v4_config
    bundle["trainer_config_hash"] = v4_hash
    bundle["stats"].update({
        "episodes_completed": 3,
        "transitions_collected": 12,
        "decisions_collected": 12,
        "optimizer_steps": 2,
    })
    bundle["policy_step"] = 2
    for name in (
        "games_collected",
        "bidding_decisions_collected",
        "abandoned_bidding_transitions",
        "learner_cardplay_samples",
        "learner_bidding_samples",
        "metrics_history_complete",
        "metrics_history_source",
    ):
        bundle["stats"].pop(name)
    for name in (
        "async_protocol_version",
        "compact_bidding_replay_schema_version",
        "episode_task_semantics",
        "episode_commit_semantics",
    ):
        bundle.pop(name)
    v4_path = tmp_path / "async-v4.pt"
    torch.save(bundle, v4_path)
    restored = V2Trainer(_tiny_model(), config=config)
    assert restored.load_training_checkpoint(str(v4_path))["checkpoint_version"] == 4
    assert restored.stats.games_collected == 3
    assert restored.stats.learner_cardplay_samples == 2
    assert restored.stats.learner_bidding_samples == 0
    assert restored.stats.metrics_history_complete is True
    assert restored.stats.metrics_history_source == "migrated_v4_exact"

    v5_bundle = torch.load(checkpoint, map_location="cuda", weights_only=True)
    v5_bundle["async_protocol_version"] = 999
    bad_path = tmp_path / "async-bad-protocol.pt"
    torch.save(v5_bundle, bad_path)
    from douzero.checkpoint.io import CheckpointCompatibilityError

    with pytest.raises(CheckpointCompatibilityError, match="async_protocol_version"):
        restored.load_training_checkpoint(str(bad_path))
