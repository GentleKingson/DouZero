from __future__ import annotations

import sys
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import benchmarks.freeze_v3_h7_protocol as freeze_h7
from douzero.belief.features import build_belief_input
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.observation.schema import build_v2_schema
from douzero.training.async_single_gpu import (
    AsyncReplayKey,
    AsyncRequestCoordinator,
    SharedBeliefInputSlots,
    _formal_action_seed,
)
from douzero.training.seed_stream import FORMAL_SEED_DERIVATION_V1
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.benchmark import h7_trainer_identity_hash
from douzero.v3_hybrid.integration_replay import assert_public_replay_payload
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.replay import capture_plain_transition
from douzero.v3_hybrid.runtime import (
    V3_H71A_REPLAY_PROTOCOL,
    V3_H71A_REQUEST_PROTOCOL,
    V3_H71A_SNAPSHOT_SEMANTICS,
    V3_H7_CHECKPOINT_FORMAT,
    V3_H7_RUNTIME_VERSION,
    V3AsyncSingleGPUTrainer,
    V3H71ABeliefAlignment,
    V3H7RuntimeConfig,
    validate_v3_h7_runtime_config,
    resolve_v3_h7_seed_contract,
    validate_v3_h7_formal_initialization,
)
from douzero.v3_hybrid.support_matrix import (
    RULESET_LEGACY,
    RULESET_STANDARD,
    TOPOLOGY_ASYNC_SINGLE_GPU,
    validate_capability_support,
)
from douzero.v3_hybrid.training.h4_learner import (
    bind_v3_h4_belief_sidecar,
    build_v3_h4_belief_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _belief_formal():
    return load_formal_config(
        ROOT / "configs/v3_formal/v3_belief_legacy.yaml"
    )


def _belief_runtime_config(**changes):
    config = V3H7RuntimeConfig(
        num_actors=1,
        games_per_actor=2,
        batch_size=4,
        replay_capacity=256,
        target_microbatch=2,
        environment_seed=123,
        action_seed=456,
        belief_runtime_enabled=True,
        belief_sidecar_capacity=256,
        request_protocol=V3_H71A_REQUEST_PROTOCOL,
        replay_protocol=V3_H71A_REPLAY_PROTOCOL,
        snapshot_semantics=V3_H71A_SNAPSHOT_SEMANTICS,
    )
    return replace(config, **changes)


def _observation_and_privileged():
    env = Env("adp")
    env.reset()
    observation = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    privileged = PrivilegedObservation(
        all_handcards=dict(env.infoset.all_handcards),
        acting_role=observation.public.acting_role,
    )
    return env, observation, privileged


def _public_row(observation):
    return capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id="episode",
        deal_id="deal",
        target_transform="raw",
    ).finalize(1.0)


def test_h71a_support_and_runtime_transport_fail_closed_before_cuda():
    resolved = build_pilot_resolved_config(_belief_formal())
    with pytest.raises(ValueError, match="belief feature and belief runtime"):
        validate_v3_h7_runtime_config(
            resolved,
            V3H7RuntimeConfig(batch_size=4),
        )
    with pytest.raises(ValueError, match="request protocol"):
        V3H7RuntimeConfig(belief_runtime_enabled=True)

    config = _belief_runtime_config()
    validate_v3_h7_runtime_config(resolved, config)
    validate_capability_support(
        "belief",
        topology=TOPOLOGY_ASYNC_SINGLE_GPU,
        ruleset=RULESET_LEGACY,
        checkpoint_resume=True,
        export=True,
        deployment=True,
        search=False,
    )
    with pytest.raises(ValueError, match="cannot be smaller"):
        validate_v3_h7_runtime_config(
            resolved, replace(config, belief_sidecar_capacity=1)
        )
    belief = resolved.learner.base.base.belief
    shared = replace(
        resolved,
        learner=replace(
            resolved.learner,
            base=replace(
                resolved.learner.base,
                base=replace(
                    resolved.learner.base.base,
                    belief=replace(belief, shared_encoder_updates=True),
                ),
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="shared encoder"):
        validate_v3_h7_runtime_config(shared, config)


def test_public_belief_slots_roundtrip_and_submit_requires_matching_input():
    _env, observation, _privileged = _observation_and_privileged()
    belief_input = build_belief_input(observation.public)
    slots = SharedBeliefInputSlots(1)
    slots.write(0, belief_input)
    restored = slots.read(0)
    assert np.array_equal(restored.feature_vector, belief_input.feature_vector)
    assert np.array_equal(restored.unseen_counts, belief_input.unseen_counts)
    assert restored.acting_role == belief_input.acting_role
    slots.clear(0)
    with pytest.raises(RuntimeError, match="was not written"):
        slots.read(0)

    coordinator = AsyncRequestCoordinator(
        build_v2_schema(),
        num_slots=1,
        output_width=6,
        belief_inputs=True,
    )
    slot = coordinator.acquire(0)
    coordinator.slots.write(slot, observation_to_model_inputs(observation))
    with pytest.raises(RuntimeError, match="was not written"):
        coordinator.submit(slot, request_id=1, policy_snapshot=0)
    coordinator.shutdown()


def test_sidecar_alignment_is_bounded_duplicate_safe_and_not_public_replay():
    _env, observation, privileged = _observation_and_privileged()
    sidecar = build_v3_h4_belief_sidecar(observation, privileged)
    row = _public_row(observation)
    key = AsyncReplayKey(actor_id=1, episode_id=2, trace_index=3)
    alignment = V3H71ABeliefAlignment(2)
    alignment.add_sidecar(key, sidecar)
    alignment.add_public(key, row)
    paired = alignment.pop_ready()
    assert len(paired) == 1
    assert paired[0][0] is row
    assert paired[0][1].label is sidecar.label
    alignment.assert_quiescent()
    with pytest.raises(RuntimeError, match="duplicate"):
        alignment.add_public(key, row)

    public_payload = row.state_dict()
    assert_public_replay_payload(public_payload)
    assert "belief" not in repr(public_payload).lower()


def test_ready_sidecar_pairs_are_aligned_atomically_at_capacity():
    _env, observation, privileged = _observation_and_privileged()
    bundle = observation_to_model_inputs(observation)
    sidecar = build_v3_h4_belief_sidecar(
        observation, privileged, public_inputs=bundle
    )
    row = _public_row(observation)
    alignment = V3H71ABeliefAlignment(1)
    for trace_index in range(2):
        paired = alignment.add_pair(
            AsyncReplayKey(
                actor_id=0, episode_id=0, trace_index=trace_index
            ),
            row,
            sidecar,
        )
        assert paired[0] is row
        assert paired[1].label is sidecar.label
        assert alignment.pop_ready() == []
    alignment.assert_quiescent()


def test_ready_pair_does_not_consume_full_unmatched_backlog_capacity():
    _env, observation, privileged = _observation_and_privileged()
    row = _public_row(observation)
    sidecar = build_v3_h4_belief_sidecar(observation, privileged)
    alignment = V3H71ABeliefAlignment(1)
    alignment.add_public(
        AsyncReplayKey(actor_id=0, episode_id=0, trace_index=0), row
    )
    paired = alignment.add_pair(
        AsyncReplayKey(actor_id=1, episode_id=0, trace_index=0),
        row,
        sidecar,
    )
    assert paired[0] is row
    assert alignment.pending_count == 1


def test_sidecar_alignment_rejects_backlog_and_unmatched_quiesce():
    _env, observation, privileged = _observation_and_privileged()
    sidecar = build_v3_h4_belief_sidecar(observation, privileged)
    alignment = V3H71ABeliefAlignment(1)
    alignment.add_sidecar(
        AsyncReplayKey(actor_id=0, episode_id=0, trace_index=0), sidecar
    )
    with pytest.raises(RuntimeError, match="unmatched belief sidecars"):
        alignment.assert_quiescent()
    with pytest.raises(RuntimeError, match="exceeded capacity"):
        alignment.add_sidecar(
            AsyncReplayKey(actor_id=0, episode_id=0, trace_index=1), sidecar
        )


def test_sidecar_source_binding_rejects_different_public_state():
    env, observation, privileged = _observation_and_privileged()
    sidecar = build_v3_h4_belief_sidecar(observation, privileged)
    env.step(env.infoset.legal_actions[0])
    later = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    with pytest.raises(ValueError, match="acting roles differ|source-state"):
        bind_v3_h4_belief_sidecar(
            observation_to_model_inputs(later), sidecar
        )


def test_sidecar_source_identity_rejects_same_metadata_different_tensor():
    _env, observation, privileged = _observation_and_privileged()
    original = observation_to_model_inputs(observation)
    sidecar = build_v3_h4_belief_sidecar(
        observation, privileged, public_inputs=original
    )
    changed_flat = original.state_context_flat.clone()
    changed_flat[0] += 1.0
    changed = replace(original, state_context_flat=changed_flat)
    assert changed.acting_role == original.acting_role
    assert changed.feature_schema_hash == original.feature_schema_hash
    with pytest.raises(ValueError, match="source-state identity"):
        bind_v3_h4_belief_sidecar(changed, sidecar)


def test_formal_trainer_identity_always_binds_resolved_learner():
    common = {
        "runtime_version": V3_H7_RUNTIME_VERSION,
        "checkpoint_format": V3_H7_CHECKPOINT_FORMAT,
        "request_protocol": V3_H71A_REQUEST_PROTOCOL,
    }
    informal = h7_trainer_identity_hash(
        **common, resolved_learner_hash=None
    )
    formal = h7_trainer_identity_hash(
        **common, resolved_learner_hash="a" * 64
    )
    assert informal != formal
    assert formal == h7_trainer_identity_hash(
        **common, resolved_learner_hash="a" * 64
    )


def test_formal_runtime_seeds_use_frozen_sha256_contract():
    resolved = resolve_v3_h7_seed_contract(
        formal_training_seeds=(101, 202, 303),
        formal_derivation=FORMAL_SEED_DERIVATION_V1,
        requested_environment_seed=None,
        requested_action_seed=None,
    )
    assert resolved[0] == 101
    assert resolved[1] == 101
    assert resolved[2] == FORMAL_SEED_DERIVATION_V1
    with pytest.raises(ValueError, match="not frozen"):
        resolve_v3_h7_seed_contract(
            formal_training_seeds=(101, 202, 303),
            formal_derivation=FORMAL_SEED_DERIVATION_V1,
            requested_environment_seed=404,
            requested_action_seed=None,
        )
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_v3_h7_seed_contract(
            formal_training_seeds=(101, 202, 303),
            formal_derivation=FORMAL_SEED_DERIVATION_V1,
            requested_environment_seed=101,
            requested_action_seed=102,
        )


def test_formal_action_rng_is_stable_per_actor_and_episode():
    seeds = {
        _formal_action_seed(101, actor_id, episode_id)
        for actor_id in range(2)
        for episode_id in range(3)
    }
    assert len(seeds) == 6
    assert _formal_action_seed(101, 1, 2) == _formal_action_seed(101, 1, 2)
    assert _formal_action_seed(202, 1, 2) != _formal_action_seed(101, 1, 2)


def test_formal_checkpoint_initialization_fails_closed():
    validate_v3_h7_formal_initialization("seeded_fresh")
    with pytest.raises(NotImplementedError, match="checkpoint initialization"):
        validate_v3_h7_formal_initialization("checkpoint")


def test_belief_only_update_advances_coupled_served_version():
    runtime = object.__new__(V3AsyncSingleGPUTrainer)
    runtime.learner = SimpleNamespace(policy_version=7)
    runtime._served_version_offset = 0
    metrics = SimpleNamespace(
        base=SimpleNamespace(
            base=SimpleNamespace(belief_updated=True)
        )
    )
    runtime._record_served_update(7, metrics)
    assert runtime.policy_step == 8
    runtime.learner.policy_version = 8
    runtime._record_served_update(7, metrics)
    assert runtime.policy_step == 9


def test_belief_async_standard_combination_is_not_advertised():
    with pytest.raises(ValueError, match="combination"):
        validate_capability_support(
            "belief",
            topology=TOPOLOGY_ASYNC_SINGLE_GPU,
            ruleset=RULESET_STANDARD,
            checkpoint_resume=True,
            export=True,
            deployment=True,
            search=False,
        )


@pytest.mark.parametrize(
    "config_name",
    ("v3_belief_standard.yaml", "v3_full_hybrid_legacy.yaml"),
)
def test_protocol_freeze_rejects_unsupported_formal_config_before_write(
    config_name, monkeypatch, tmp_path
):
    output = tmp_path / "protocol.json"
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
    with pytest.raises((ValueError, NotImplementedError)):
        freeze_h7.main()
    assert not output.exists()


def test_protocol_freeze_rejects_checkpoint_initialization_before_write(
    monkeypatch, tmp_path
):
    formal = _belief_formal()
    formal = replace(
        formal,
        initialization=replace(formal.initialization, kind="checkpoint"),
    )
    output = tmp_path / "protocol.json"
    monkeypatch.setattr(freeze_h7, "load_formal_config", lambda _path: formal)
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
            "ignored.yaml",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(NotImplementedError, match="checkpoint initialization"):
        freeze_h7.main()
    assert not output.exists()


def test_protocol_freeze_binds_complete_formal_identity_and_seeds(
    monkeypatch, tmp_path
):
    formal = _belief_formal()
    output = tmp_path / "protocol.json"
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
            str(ROOT / "configs/v3_formal/v3_belief_legacy.yaml"),
            "--output",
            str(output),
        ],
    )
    freeze_h7.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["formal_config_hash"] == formal.identity_dict()[
        "config_sha256"
    ]
    assert payload["seeds"] == list(formal.seeds.training)


def test_public_belief_request_is_invariant_to_hidden_hand_swap():
    env, before, _privileged = _observation_and_privileged()
    first = build_belief_input(before.public)
    hands = env.infoset.all_handcards
    roles = [
        role for role in hands if role != before.public.acting_role
    ]
    hands[roles[0]], hands[roles[1]] = hands[roles[1]], hands[roles[0]]
    after = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    second = build_belief_input(after.public)
    assert np.array_equal(first.feature_vector, second.feature_vector)
    assert np.array_equal(first.unseen_counts, second.unseen_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_h71a_cuda_coupled_snapshot_update_checkpoint_resume_and_shutdown(
    tmp_path,
):
    formal = _belief_formal()
    learner, resolved = create_pilot_learner(formal)
    runtime = V3AsyncSingleGPUTrainer(
        learner, resolved, _belief_runtime_config()
    )
    checkpoint = tmp_path / "belief-async.pt"
    try:
        runtime.collect_episodes(1)
        public_before = {
            name: value.detach().clone()
            for name, value in learner.model.state_dict().items()
        }
        belief_before = {
            name: value.detach().clone()
            for name, value in runtime.belief_model.state_dict().items()
        }
        metrics = runtime.step()
        assert metrics is not None
        assert metrics.base.base.belief_updated
        assert runtime.stats.belief_optimizer_steps == 1
        assert runtime.stats.belief_labels_collected == len(runtime.buffer)
        assert any(
            not torch.equal(public_before[name], value)
            for name, value in learner.model.state_dict().items()
        )
        assert any(
            not torch.equal(belief_before[name], value)
            for name, value in runtime.belief_model.state_dict().items()
        )
        boundary = runtime.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["pending_requests"] == 0
        assert boundary["belief_replay_occupancy"] == len(runtime.buffer)
        runtime.save_training_checkpoint(
            str(checkpoint), long_running_state={"cycle": 1}
        )
        saved_policy_step = runtime.policy_step
    finally:
        runtime.shutdown()

    resumed_learner, resumed_resolved = create_pilot_learner(formal)
    resumed = V3AsyncSingleGPUTrainer(
        resumed_learner, resumed_resolved, _belief_runtime_config()
    )
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert resumed.policy_step == saved_policy_step
        assert resumed.stats.belief_optimizer_steps == 1
        assert resumed.learner.base.base.phase() == learner.base.base.phase()
        resumed.collect_episodes(1)
        assert resumed.step() is not None
        assert resumed.stats.belief_optimizer_steps == 2
        status = resumed.quiesce_cycle_boundary()
        assert status["active_slots"] == 0
        assert status["in_flight_slots"] == 0
        assert status["pending_requests"] == 0
    finally:
        resumed.shutdown()
    assert not list(tmp_path.glob("*.tmp"))
