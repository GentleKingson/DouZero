from __future__ import annotations

import json
import copy
import queue
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import benchmarks.freeze_v3_h7_protocol as freeze_h7
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.training.async_single_gpu import AsyncReplayKey
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.integration_replay import assert_public_replay_payload
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.replay import (
    AdaptiveSnapshotProvenance,
    capture_plain_transition,
)
from douzero.v3_hybrid.runtime import (
    V3_H71A_SNAPSHOT_SEMANTICS,
    V3_H71F_LEGACY_REPLAY_PROTOCOL,
    V3_H71F_LEGACY_REQUEST_PROTOCOL,
    V3_H71F_STANDARD_REPLAY_PROTOCOL,
    V3_H71F_STANDARD_REQUEST_PROTOCOL,
    V3AsyncSingleGPUTrainer,
    V3H71ABeliefAlignment,
    V3H71BOracleAlignment,
    V3H71CCooperationAlignment,
    V3H7RuntimeConfig,
    V3H7RuntimeStats,
    resolve_v3_h7_protocols,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.support_matrix import v3_h6_support_matrix_dict
from douzero.v3_hybrid.training.cooperation import (
    build_v3_h5_async_decision_sidecar,
)
from douzero.v3_hybrid.training.h3_learner import (
    build_v3_h3_oracle_sidecar,
)
from douzero.v3_hybrid.training.h4_learner import (
    build_v3_h4_belief_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolved(name: str):
    formal = load_formal_config(ROOT / "configs/v3_formal" / name)
    return build_pilot_resolved_config(formal, allow_standard=True)


def _runtime_config(*, bidding: bool, topology: str = "async_single_gpu"):
    request, replay, snapshot = resolve_v3_h7_protocols(
        belief=True,
        oracle=True,
        cooperation=True,
        public_aux=True,
        bidding=bidding,
    )
    return V3H7RuntimeConfig(
        topology=topology,
        num_actors=1,
        games_per_actor=1,
        batch_size=32,
        replay_capacity=256,
        belief_sidecar_capacity=256,
        oracle_sidecar_capacity=256,
        cooperation_sidecar_capacity=256,
        cooperation_episode_capacity=32,
        bidding_replay_capacity=64,
        bidding_batch_size=4,
        target_microbatch=1,
        request_timeout_seconds=60.0,
        belief_runtime_enabled=True,
        oracle_runtime_enabled=True,
        cooperation_runtime_enabled=True,
        public_aux_runtime_enabled=True,
        bidding_runtime_enabled=bidding,
        request_protocol=request,
        replay_protocol=replay,
        snapshot_semantics=snapshot,
    )


@pytest.mark.parametrize(
    ("config_name", "bidding", "request_protocol", "replay_protocol"),
    (
        (
            "v3_full_hybrid_legacy.yaml",
            False,
            V3_H71F_LEGACY_REQUEST_PROTOCOL,
            V3_H71F_LEGACY_REPLAY_PROTOCOL,
        ),
        (
            "v3_full_hybrid_standard.yaml",
            True,
            V3_H71F_STANDARD_REQUEST_PROTOCOL,
            V3_H71F_STANDARD_REPLAY_PROTOCOL,
        ),
    ),
)
def test_full_hybrid_runtime_validates_before_cuda(
    config_name, bidding, request_protocol, replay_protocol
):
    resolved = _resolved(config_name)
    protocols = resolve_v3_h7_protocols(
        belief=True,
        oracle=True,
        cooperation=True,
        public_aux=True,
        bidding=bidding,
    )
    assert protocols == (
        request_protocol,
        replay_protocol,
        V3_H71A_SNAPSHOT_SEMANTICS,
    )
    runtime = V3H7RuntimeConfig(
        batch_size=32,
        belief_runtime_enabled=True,
        oracle_runtime_enabled=True,
        cooperation_runtime_enabled=True,
        public_aux_runtime_enabled=True,
        bidding_runtime_enabled=bidding,
        request_protocol=request_protocol,
        replay_protocol=replay_protocol,
        snapshot_semantics=V3_H71A_SNAPSHOT_SEMANTICS,
    )
    validate_v3_h7_runtime_config(resolved, runtime)
    assert runtime.identity()["request_protocol"] == request_protocol


def test_partial_combined_transport_fails_closed():
    with pytest.raises(NotImplementedError, match="partial combined"):
        resolve_v3_h7_protocols(
            belief=True,
            oracle=True,
            cooperation=False,
            public_aux=False,
            bidding=False,
        )


def test_support_matrix_declares_exact_full_hybrid_bundles():
    bundles = v3_h6_support_matrix_dict()["async_capability_bundles"]
    assert set(bundles) == {
        "full_hybrid_legacy_v1",
        "full_hybrid_standard_v1",
    }
    assert bundles["full_hybrid_legacy_v1"]["ruleset"] == "legacy"
    assert bundles["full_hybrid_standard_v1"]["ruleset"] == "standard"
    assert "bidding" in bundles["full_hybrid_standard_v1"]["required"]
    assert "bidding" in bundles["full_hybrid_legacy_v1"]["forbidden"]
    with pytest.raises(NotImplementedError, match="partial combined"):
        V3H7RuntimeConfig(
            belief_runtime_enabled=True,
            oracle_runtime_enabled=True,
        )


@pytest.mark.parametrize(
    "config_name",
    ("v3_full_hybrid_legacy.yaml", "v3_full_hybrid_standard.yaml"),
)
def test_protocol_freeze_accepts_full_hybrid_identity(
    config_name, monkeypatch, tmp_path
):
    output = tmp_path / f"{config_name}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_v3_h7_protocol.py",
            "--image-digest",
            f"sha256:{'a' * 64}",
            "--gpu",
            "gpu",
            "--driver",
            "driver",
            "--pytorch",
            "pytorch",
            "--cuda",
            "cuda",
            "--cpu",
            "cpu",
            "--formal-config",
            str(ROOT / "configs/v3_formal" / config_name),
            "--output",
            str(output),
        ],
    )
    freeze_h7.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["oracle_enabled"] is True
    assert payload["cooperation_enabled"] is True
    assert payload["public_aux_enabled"] is True
    assert payload["bidding_enabled"] is config_name.endswith("standard.yaml")


def _decision(role: str, trace_index: int):
    np.random.seed(900 + trace_index)
    env = Env("adp")
    env.reset()
    for _ in range(120):
        if env._acting_player_position == role:
            infoset = copy.deepcopy(env.infoset)
            infoset.legal_actions = infoset.legal_actions[:4]
            observation = get_obs_v2(infoset, ruleset=RuleSet.legacy())
            privileged = PrivilegedObservation(
                all_handcards=dict(infoset.all_handcards),
                acting_role=role,
            )
            break
        action = env.infoset.legal_actions[0]
        _obs, _reward, done, _info = env.step(action)
        if done:
            env.reset()
    else:
        raise AssertionError(f"could not reach {role}")
    row = replace(
        capture_plain_transition(
            observation,
            selected_action_index=0,
            episode_id="actor-1-episode-2",
            deal_id="async-deal-2",
            target_transform="raw",
        ).finalize(1.0),
        adaptive_provenance=AdaptiveSnapshotProvenance(
            q_old=0.0,
            policy_version=7,
            snapshot_slot=0,
            owner_id=1,
            generation=3,
        ),
    )
    key = AsyncReplayKey(actor_id=1, episode_id=2, trace_index=trace_index)
    public_inputs = observation_to_model_inputs(observation)
    belief = build_v3_h4_belief_sidecar(
        observation, privileged, public_inputs=public_inputs
    )
    oracle = build_v3_h3_oracle_sidecar(
        observation,
        privileged,
        action_index=0,
        public_inputs=public_inputs,
    )
    cooperation = (
        None
        if role == "landlord"
        else build_v3_h5_async_decision_sidecar(
            observation,
            selected_action_index=0,
            trace_index=trace_index,
            public_inputs=public_inputs,
            snapshot_policy_version=7,
            policy_id="policy@7",
            teammate_policy_id="policy@7",
        )
    )
    return key, row, belief, oracle, cooperation


def test_full_hybrid_alignment_attaches_every_auxiliary_atomically():
    decisions = [
        _decision("landlord", 0),
        _decision("landlord_up", 1),
        _decision("landlord_down", 2),
    ]
    trainer = object.__new__(V3AsyncSingleGPUTrainer)
    trainer.model = SimpleNamespace(
        schema=SimpleNamespace(stable_hash=lambda: "schema"),
        config=SimpleNamespace(dmc_target_transform="raw"),
    )
    trainer.learner = SimpleNamespace(
        ruleset=SimpleNamespace(identity=lambda: {"ruleset": "legacy"})
    )
    trainer._replay_slots = SimpleNamespace(
        read_ready_v3_aligned=lambda **_kwargs: [
            (row, key, {"min_turns_after": float(key.trace_index)})
            for key, row, _belief, _oracle, _cooperation in decisions
        ]
    )
    trainer._belief_sidecar_queue = queue.Queue()
    trainer._oracle_sidecar_queue = queue.Queue()
    trainer._cooperation_sidecar_queue = queue.Queue()
    for key, _row, belief, oracle, cooperation in decisions:
        trainer._belief_sidecar_queue.put((key, belief))
        trainer._oracle_sidecar_queue.put((key, oracle))
        if cooperation is not None:
            trainer._cooperation_sidecar_queue.put((key, cooperation))
    trainer._belief_alignment = V3H71ABeliefAlignment(32)
    trainer._oracle_alignment = V3H71BOracleAlignment(32)
    trainer._cooperation_alignment = V3H71CCooperationAlignment(32, 8)
    trainer.config = SimpleNamespace(cooperation_episode_capacity=8)
    trainer._cooperation_alignment.mark_episode_complete(
        1,
        2,
        {"landlord_up": 1, "landlord_down": 1},
        total_count=3,
    )
    trainer._full_belief_samples = {}
    trainer._full_oracle_samples = {}
    trainer._full_strategy_targets = {}
    trainer._pending_cooperation_episodes = deque()
    trainer._full_discarded_keys = deque()
    trainer._full_discarded_key_set = set()
    trainer._full_sidecars_received = {
        "belief": 0,
        "oracle": 0,
        "cooperation": 0,
    }
    trainer.buffer = deque(maxlen=32)
    trainer.belief_buffer = deque(maxlen=32)
    trainer.oracle_buffer = deque(maxlen=32)
    trainer.cooperation_buffer = deque(maxlen=8)
    trainer.strategy_target_buffer = deque(maxlen=32)
    trainer.stats = V3H7RuntimeStats()

    assert trainer._drain_full_hybrid_replay() == 3
    episode = trainer.cooperation_buffer[0]
    assert episode.keys == tuple(item[0] for item in decisions)
    assert len(episode.belief_samples) == 3
    assert len(episode.oracle_samples) == 3
    assert len(episode.strategy_targets) == 3
    assert len(trainer.buffer) == len(trainer.belief_buffer) == 3
    assert len(trainer.oracle_buffer) == len(trainer.strategy_target_buffer) == 3
    for row in trainer.buffer:
        assert_public_replay_payload(row.state_dict())
    trainer._belief_alignment.assert_quiescent()
    trainer._oracle_alignment.assert_quiescent()
    trainer._cooperation_alignment.assert_quiescent()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_hybrid_legacy_cuda_update_checkpoint_resume_and_shutdown(tmp_path):
    formal = load_formal_config(
        ROOT / "configs/v3_formal/v3_full_hybrid_legacy.yaml"
    )
    learner, resolved = create_pilot_learner(formal, allow_standard=True)
    runtime_config = _runtime_config(bidding=False)
    runtime = V3AsyncSingleGPUTrainer(learner, resolved, runtime_config)
    checkpoint = tmp_path / "h71f-legacy.pt"
    try:
        runtime.collect_episodes(1)
        before = runtime._parameter_update_snapshot()
        assert runtime.step() is not None
        assert runtime._parameters_changed_since(before)
        boundary = runtime.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["pending_requests"] == 0
        runtime.save_training_checkpoint(
            checkpoint, long_running_state={"cycle": 1}
        )
        saved_steps = runtime.stats.optimizer_steps
    finally:
        runtime.shutdown()

    resumed_learner, resumed_resolved = create_pilot_learner(
        formal, allow_standard=True
    )
    resumed = V3AsyncSingleGPUTrainer(
        resumed_learner, resumed_resolved, runtime_config
    )
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert resumed.stats.optimizer_steps == saved_steps
        resumed.collect_episodes(1)
        assert resumed.step() is not None
        assert resumed.stats.optimizer_steps == saved_steps + 1
        counts = resumed.quiesce_cycle_boundary()
        assert counts["active_slots"] == 0
        assert counts["pending_requests"] == 0
    finally:
        resumed.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_hybrid_standard_cuda_updates_cardplay_and_bidding():
    formal = load_formal_config(
        ROOT / "configs/v3_formal/v3_full_hybrid_standard.yaml"
    )
    learner, resolved = create_pilot_learner(formal, allow_standard=True)
    h3 = learner.base.base.base
    h3.learner_updates = h3.config.schedule.warmup_updates
    runtime = V3AsyncSingleGPUTrainer(
        learner, resolved, _runtime_config(bidding=True)
    )
    try:
        runtime.collect_episodes(1)
        before = runtime._parameter_update_snapshot()
        assert runtime.step() is not None
        assert runtime._parameters_changed_since(before)
        assert runtime.stats.bidding_optimizer_steps == 1
        assert runtime.stats.belief_labels_collected > 0
        assert runtime.stats.oracle_labels_collected > 0
        assert runtime.stats.cooperation_episodes_collected > 0
        assert runtime.stats.strategy_labels_collected > 0
        counts = runtime.quiesce_cycle_boundary()
        assert counts["active_slots"] == 0
        assert counts["pending_requests"] == 0
    finally:
        runtime.shutdown()
