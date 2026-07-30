from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation.bidding import get_bidding_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.training.async_single_gpu import (
    AsyncRequestCoordinator,
    RequestKind,
    SharedBiddingSlots,
    async_actor_main,
)
from douzero.v3_hybrid.benchmark import (
    H7_TOPOLOGIES,
    V3H7BenchmarkProtocol,
    validate_h7_benchmark_evidence,
)
from douzero.v3_hybrid.integration_config import load_v3_hybrid_config
from douzero.v3_hybrid.runtime import (
    V3_H71E_REPLAY_PROTOCOL,
    V3_H71E_REQUEST_PROTOCOL,
    V3H7RuntimeConfig,
    V3SingleProcessTrainer,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.support_matrix import (
    RULESET_STANDARD,
    TOPOLOGY_ASYNC_SINGLE_GPU,
    validate_capability_support,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolved():
    return load_v3_hybrid_config(
        ROOT / "configs/v3_hybrid_h7_1e_standard_bidding.yaml"
    )


def _runtime(**changes):
    return replace(
        V3H7RuntimeConfig(
            batch_size=8,
            bidding_batch_size=4,
            bidding_runtime_enabled=True,
            request_protocol=V3_H71E_REQUEST_PROTOCOL,
            replay_protocol=V3_H71E_REPLAY_PROTOCOL,
        ),
        **changes,
    )


def _bidding_observation():
    ruleset = RuleSet.standard()
    env = Env("adp", ruleset=ruleset)
    env.reset(bidding_order=["1", "2", "0"])
    return get_bidding_obs_v2(
        env.bidding_obs,
        ruleset=ruleset,
        redeal_count=env._redeal_count,
    )


def test_h71e_support_and_startup_fail_closed_before_cuda():
    resolved = _resolved()
    validate_v3_h7_runtime_config(resolved, _runtime())
    validate_capability_support(
        "bidding",
        topology=TOPOLOGY_ASYNC_SINGLE_GPU,
        ruleset=RULESET_STANDARD,
        checkpoint_resume=True,
        export=True,
        deployment=True,
        search=False,
    )
    with pytest.raises(ValueError, match="transport disagree"):
        validate_v3_h7_runtime_config(
            resolved, V3H7RuntimeConfig(batch_size=8)
        )
    with pytest.raises(ValueError, match="request protocol"):
        V3H7RuntimeConfig(bidding_runtime_enabled=True)
    with pytest.raises(NotImplementedError, match="combined async"):
        V3H7RuntimeConfig(
            bidding_runtime_enabled=True,
            public_aux_runtime_enabled=True,
            request_protocol=V3_H71E_REQUEST_PROTOCOL,
            replay_protocol=V3_H71E_REPLAY_PROTOCOL,
        )


def test_bidding_slots_round_trip_public_neutral_seat_action_space():
    observation = _bidding_observation()
    slots = SharedBiddingSlots(2, observation.schema.input_width)
    slots.write(1, observation)
    batch = slots.batch([1], observation.feature_schema_hash)
    assert batch.features.shape == (1, observation.schema.input_width)
    assert batch.legal_mask.shape == (1, 4)
    assert torch.equal(
        batch.legal_mask[0],
        torch.from_numpy(observation.bid_action_mask.copy()),
    )
    assert observation.current_seat == "1"
    assert observation.current_seat not in {
        "landlord", "landlord_up", "landlord_down"
    }


def test_bidding_public_encoder_ignores_hidden_allocations():
    ruleset = RuleSet.standard()
    env = Env("adp", ruleset=ruleset)
    raw = env.reset(bidding_order=["0", "1", "2"])
    raw_a = dict(raw)
    raw_b = dict(raw)
    raw_a["opponent_hands"] = {"1": [3] * 17, "2": [4] * 17}
    raw_b["opponent_hands"] = {"1": [5] * 17, "2": [6] * 17}
    raw_a["bottom_cards"] = [20, 30, 17]
    raw_b["bottom_cards"] = [3, 4, 5]
    encoded_a = get_bidding_obs_v2(raw_a, ruleset=ruleset)
    encoded_b = get_bidding_obs_v2(raw_b, ruleset=ruleset)
    assert torch.equal(encoded_a.to_tensor(), encoded_b.to_tensor())
    assert encoded_a.legal_bids == encoded_b.legal_bids


def test_one_coordinator_keeps_bid_and_cardplay_request_kinds_separate():
    observation = _bidding_observation()
    coordinator = AsyncRequestCoordinator(
        build_v2_schema(),
        num_slots=2,
        bidding_feature_width=observation.schema.input_width,
    )
    slot = coordinator.acquire(3)
    coordinator.bidding_inputs.write(slot, observation)
    coordinator.submit_bidding(slot, request_id=7, policy_snapshot=11)
    request = coordinator.claim_ready(1, wait_seconds=0.1)[0]
    assert request.request_kind == int(RequestKind.BIDDING)
    assert request.grouping_key == (11, "bidding")
    assert request.action_count == 4
    assert request.acting_role == -1
    coordinator.complete(slot)
    coordinator.release(slot)
    coordinator.shutdown()


def test_actor_contract_uses_environment_bids_and_separate_replay():
    source = inspect.getsource(async_actor_main)
    assert 'game["env"].step(None, bid_value=bid)' in source
    assert "game[\"bidding_transitions\"]" in source
    assert "bidding_replay_queue.put" in source
    assert "bid not in observation.legal_bids" in source
    assert "episode.action_trace.append" in source
    assert "max_redeals_exceeded" in source


def test_h71e_identity_covers_cadence_and_action_semantics():
    base = _runtime()
    assert base.identity()["bidding_update_interval"] == 1
    assert base.identity()["first_bidder_mode"] == "rotate"
    assert base.stable_hash() != replace(
        base, bidding_update_interval=2
    ).stable_hash()
    assert base.stable_hash() != replace(
        base, first_bidder_mode="seeded_random"
    ).stable_hash()
    assert base.stable_hash() != replace(
        base, bidding_learned_probability=0.5
    ).stable_hash()


def test_h71e_benchmark_requires_positive_bidding_metrics():
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
        bidding_enabled=True,
        measurement_seconds=1.0,
    )
    records = []
    for topology in H7_TOPOLOGIES:
        for repeat, seed in enumerate(protocol.seeds):
            record = {
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
                "strategy_samples_per_second": 0.0,
                "strategy_optimizer_steps_per_second": 0.0,
                "public_aux_parameter_vram_bytes": 0.0,
                "bidding_samples_per_second": 1.0,
                "bidding_optimizer_steps_per_second": 1.0,
                "bidding_parameter_vram_bytes": 1.0,
                "requests_per_microbatch": 1.0,
                "legal_actions_per_batch": 4.0,
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
            }
            records.append(record)
    validate_h7_benchmark_evidence(records, protocol)
    records[0]["bidding_samples_per_second"] = 0.0
    with pytest.raises(ValueError, match="bidding"):
        validate_h7_benchmark_evidence(records, protocol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h71e_cuda_update_checkpoint_resume_and_shutdown(tmp_path):
    from douzero.v3_hybrid import V3HybridModel
    from douzero.v3_hybrid.training.h6_learner import V3H6Learner

    resolved = _resolved()
    runtime = _runtime(
        topology="single_process",
        num_actors=1,
        games_per_actor=1,
    )
    model = V3HybridModel(build_v2_schema(), resolved.model)
    learner = V3H6Learner(
        model,
        ruleset=RuleSet.standard(),
        config=resolved,
    )
    trainer = V3SingleProcessTrainer(learner, resolved, runtime)
    try:
        trainer.collect_episodes(8)
        before = trainer._parameter_update_snapshot()
        metrics = trainer.step()
        assert metrics is not None
        assert trainer._parameters_changed_since(before)
        assert trainer.stats.bidding_optimizer_steps == 1
        checkpoint = tmp_path / "h71e.pt"
        trainer.save_training_checkpoint(
            checkpoint, long_running_state={"cycle": 1}
        )
    finally:
        trainer.shutdown()
    resumed_model = V3HybridModel(build_v2_schema(), resolved.model)
    resumed_learner = V3H6Learner(
        resumed_model,
        ruleset=RuleSet.standard(),
        config=resolved,
    )
    resumed = V3SingleProcessTrainer(resumed_learner, resolved, runtime)
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert resumed.stats.bidding_optimizer_steps == 1
        before = resumed._parameter_update_snapshot()
        resumed.collect_episodes(8)
        metrics = resumed.step()
        assert metrics is not None
        assert resumed._parameters_changed_since(before)
        assert resumed.stats.bidding_optimizer_steps == 2
        tampered = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tampered["runtime_hash"] = "0" * 64
        wrong_identity = tmp_path / "wrong-identity.pt"
        torch.save(tampered, wrong_identity)
        with pytest.raises(ValueError, match="runtime identity"):
            resumed.load_training_checkpoint(wrong_identity)
    finally:
        resumed.shutdown()
