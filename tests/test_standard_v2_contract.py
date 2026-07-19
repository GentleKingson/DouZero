"""M0 regression gates for the Standard V2 single-GPU production plan."""

from __future__ import annotations

import json
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
from douzero.training import Episode, TrainerConfig, V2Trainer
from douzero.training.standard_v2_contract import (
    BASE_ASYNC_PROTOCOL_VERSION,
    STANDARD_ASYNC_PROTOCOL_VERSION,
    STANDARD_V2_R1_CONFIG_HASH,
    stable_identity_hash,
    standard_v2_version_contract,
    validate_standard_v2_r1_config,
)
from train_v2 import _build_training_metrics


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


def test_standard_v2_r1_yaml_matches_frozen_contract():
    identity = validate_standard_v2_r1_config(
        load_config(str(ROOT / "configs" / "standard_v2.yaml"))
    )
    assert stable_identity_hash(identity) == STANDARD_V2_R1_CONFIG_HASH


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
        training_wall_seconds=2.0,
        device_type="cuda",
        peak_memory_bytes=8 * 1024 * 1024,
        peak_reserved_memory_bytes=10 * 1024 * 1024,
        amp_enabled=False,
        amp_dtype="float16",
        amp_fallback_on_nonfinite=True,
        compile_enabled=False,
        ddp_enabled=False,
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


def test_checked_in_gpu_baseline_is_bound_to_the_golden_reference():
    baseline = json.loads(GPU_BASELINE_PATH.read_text(encoding="utf-8"))
    reference = build_standard_v2_reference()
    assert baseline["schema_version"] == "standard-v2-r1-benchmark-v1"
    assert baseline["config_hash"] == STANDARD_V2_R1_CONFIG_HASH
    assert baseline["reference_digest"] == reference["reference_digest"]
    assert baseline["performance"]["status"] == "measured"
    assert baseline["performance"]["counts"]["games"] > 0
    assert baseline["performance"]["peak_vram_mib"] > 0


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

    v5_bundle = torch.load(checkpoint, map_location="cuda", weights_only=True)
    v5_bundle["async_protocol_version"] = 999
    bad_path = tmp_path / "async-bad-protocol.pt"
    torch.save(v5_bundle, bad_path)
    from douzero.checkpoint.io import CheckpointCompatibilityError

    with pytest.raises(CheckpointCompatibilityError, match="async_protocol_version"):
        restored.load_training_checkpoint(str(bad_path))
