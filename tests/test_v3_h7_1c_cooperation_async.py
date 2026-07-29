from __future__ import annotations

import copy
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.training.async_single_gpu import AsyncReplayKey, async_actor_main
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.integration_replay import assert_public_replay_payload
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.replay import capture_plain_transition
from douzero.v3_hybrid.runtime import (
    V3_H71C_REPLAY_PROTOCOL,
    V3_H71C_REQUEST_PROTOCOL,
    V3AsyncSingleGPUTrainer,
    V3H71CCooperationAlignment,
    V3H7RuntimeConfig,
    V3H7RuntimeStats,
    _h71c_needs_collection_retry,
    _remap_h7_cooperation_trajectories,
    _restore_h7_cooperation_alignment_counters,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.training.cooperation import (
    FARMER_ROLES,
    bind_v3_h5_async_decision,
    build_v3_h5_async_decision_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolved():
    return build_pilot_resolved_config(
        load_formal_config(
            ROOT / "configs/v3_formal/v3_farmer_cooperation_legacy.yaml"
        )
    )


def _runtime(**changes):
    return replace(
        V3H7RuntimeConfig(
            batch_size=8,
            cooperation_runtime_enabled=True,
            cooperation_sidecar_capacity=32,
            cooperation_episode_capacity=16,
            request_protocol=V3_H71C_REQUEST_PROTOCOL,
            replay_protocol=V3_H71C_REPLAY_PROTOCOL,
        ),
        **changes,
    )


def _observation(role: str, seed: int):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(120):
        if env._acting_player_position == role:
            infoset = copy.deepcopy(env.infoset)
            infoset.legal_actions = infoset.legal_actions[:4]
            return get_obs_v2(infoset, ruleset=RuleSet.legacy())
        action = next(
            (item for item in env.infoset.legal_actions if item),
            env.infoset.legal_actions[0],
        )
        _obs, _reward, done, _info = env.step(action)
        if done:
            env.reset()
    raise AssertionError(f"could not reach {role}")


def _decision(role: str, trace_index: int, *, actor=1, episode=2):
    observation = _observation(role, 1000 + trace_index + 100 * actor)
    public_inputs = observation_to_model_inputs(observation)
    row = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=f"actor-{actor}-episode-{episode}",
        deal_id=f"async-deal-{episode}",
        target_transform="raw",
    ).finalize(2.0)
    sidecar = build_v3_h5_async_decision_sidecar(
        observation,
        selected_action_index=0,
        trace_index=trace_index,
        public_inputs=public_inputs,
        policy_id="policy@7",
        teammate_policy_id="policy@7",
    )
    return (
        AsyncReplayKey(
            actor_id=actor, episode_id=episode, trace_index=trace_index
        ),
        row,
        sidecar,
    )


def test_cooperation_runtime_support_fails_closed_before_cuda():
    resolved = _resolved()
    validate_v3_h7_runtime_config(resolved, _runtime())
    with pytest.raises(ValueError, match="cooperation feature"):
        validate_v3_h7_runtime_config(
            resolved, V3H7RuntimeConfig(batch_size=8)
        )
    with pytest.raises(ValueError, match="request protocol"):
        V3H7RuntimeConfig(cooperation_runtime_enabled=True)
    with pytest.raises(ValueError, match="cannot be smaller"):
        validate_v3_h7_runtime_config(
            resolved, _runtime(cooperation_sidecar_capacity=4)
        )


def test_public_sidecar_binds_exact_decision_and_never_enters_public_replay():
    _key, row, sidecar = _decision("landlord_up", 3)
    decision = bind_v3_h5_async_decision(row, sidecar)
    assert decision.trace_index == 3
    assert decision.transition is row
    assert decision.public_features.shape == (10,)
    payload = row.state_dict()
    assert_public_replay_payload(payload)
    assert "cooperation" not in repr(payload).lower()
    assert "trajectory" not in repr(payload).lower()
    with pytest.raises(ValueError, match="selected action"):
        bind_v3_h5_async_decision(
            replace(row, selected_action_index=1), sidecar
        )


def test_alignment_builds_unequal_episode_atomic_farmer_pair():
    alignment = V3H71CCooperationAlignment(
        capacity=16, max_episode_transitions=8
    )
    decisions = (
        _decision("landlord_up", 5),
        _decision("landlord_down", 4),
        _decision("landlord_up", 1),
    )
    alignment.add_sidecar(decisions[0][0], decisions[0][2])
    alignment.add_public(decisions[0][0], decisions[0][1])
    assert alignment.pop_ready_pairs() == 1
    for key, row, sidecar in decisions[1:]:
        assert alignment.add_pair(key, row, sidecar) == 1
    alignment.mark_episode_complete(
        1, 2, {"landlord_up": 2, "landlord_down": 1}
    )
    episodes = alignment.pop_ready_episodes()
    assert len(episodes) == 1
    up, down = episodes[0].trajectories
    assert up.decision_indices == (1, 5)
    assert down.decision_indices == (4,)
    assert len(episodes[0].transitions) == 3
    alignment.assert_quiescent()


def test_ordinary_dmc_row_normalization_preserves_trajectory_alignment():
    alignment = V3H71CCooperationAlignment(
        capacity=8, max_episode_transitions=8
    )
    for role, trace in zip(FARMER_ROLES, (0, 1)):
        key, row, sidecar = _decision(role, trace)
        alignment.add_pair(key, row, sidecar)
    alignment.mark_episode_complete(
        1, 2, {"landlord_up": 1, "landlord_down": 1}
    )
    episode = alignment.pop_ready_episodes()[0]
    learner_rows = [
        replace(row, adaptive_provenance=None)
        for row in episode.transitions
    ]
    trajectories = _remap_h7_cooperation_trajectories(
        episode.transitions, learner_rows, episode.trajectories
    )
    assert trajectories is not None
    remapped_rows = [
        row for trajectory in trajectories for row in trajectory.transitions
    ]
    assert all(
        remapped is learner
        for remapped, learner in zip(remapped_rows, learner_rows)
    )
    assert trajectories[0].decision_indices == (0,)
    assert trajectories[1].decision_indices == (1,)


def test_resume_seeds_cumulative_alignment_skip_counters():
    alignment = V3H71CCooperationAlignment(
        capacity=8, max_episode_transitions=8
    )
    stats = V3H7RuntimeStats(
        cooperation_incomplete_episodes=7,
        cooperation_oversized_episodes=11,
    )
    _restore_h7_cooperation_alignment_counters(alignment, stats)
    assert alignment.incomplete_episodes == 7
    assert alignment.oversized_episodes == 11

    key, row, sidecar = _decision("landlord_up", 0)
    alignment.add_pair(key, row, sidecar)
    alignment.mark_episode_complete(
        1, 2, {"landlord_up": 1, "landlord_down": 0}
    )
    assert alignment.incomplete_episodes == 8
    assert alignment.oversized_episodes == 11


def test_collection_retries_only_after_all_rows_arrive_without_eligible_episode():
    assert _h71c_needs_collection_retry(
        completed=4, target=4, received=120, expected=120, replay_size=0
    )
    assert not _h71c_needs_collection_retry(
        completed=4, target=4, received=119, expected=120, replay_size=0
    )
    assert not _h71c_needs_collection_retry(
        completed=4, target=4, received=120, expected=120, replay_size=1
    )
    with pytest.raises(ValueError, match="non-negative ints"):
        _h71c_needs_collection_retry(
            completed=-1, target=4, received=0, expected=0, replay_size=0
        )


def test_alignment_explicitly_skips_incomplete_and_oversized_episodes():
    incomplete = V3H71CCooperationAlignment(
        capacity=8, max_episode_transitions=8
    )
    key, row, sidecar = _decision("landlord_up", 0)
    incomplete.add_pair(key, row, sidecar)
    incomplete.mark_episode_complete(
        1, 2, {"landlord_up": 1, "landlord_down": 0}
    )
    assert incomplete.pop_ready_episodes() == []
    assert incomplete.incomplete_episodes == 1
    incomplete.assert_quiescent()

    oversized = V3H71CCooperationAlignment(
        capacity=8, max_episode_transitions=1
    )
    for role, trace in zip(FARMER_ROLES, (0, 1)):
        key, row, sidecar = _decision(role, trace, actor=3, episode=9)
        oversized.add_pair(key, row, sidecar)
    oversized.mark_episode_complete(
        3, 9, {"landlord_up": 1, "landlord_down": 1}
    )
    assert oversized.pop_ready_episodes() == []
    assert oversized.oversized_episodes == 1
    oversized.assert_quiescent()


def test_alignment_rejects_duplicate_and_episode_mismatch():
    alignment = V3H71CCooperationAlignment(
        capacity=4, max_episode_transitions=4
    )
    key, row, sidecar = _decision("landlord_up", 0)
    alignment.add_pair(key, row, sidecar)
    with pytest.raises(RuntimeError, match="duplicate"):
        alignment.add_pair(key, row, sidecar)
    other = replace(row, episode_id="wrong")
    key2, _row2, sidecar2 = _decision("landlord_down", 1)
    with pytest.raises(ValueError, match="episode identity"):
        alignment.add_pair(key2, other, sidecar2)


def test_actor_cooperation_path_is_public_only_and_does_not_construct_mixer():
    source = inspect.getsource(async_actor_main)
    assert "build_v3_h5_async_decision_sidecar" in source
    assert "FarmerCooperationModule" not in source
    assert "privileged_mixer_state" not in source
    assert ".cooperation(" not in source


def test_primary_h7_cli_selects_cooperation_protocol_before_cuda(
    monkeypatch, tmp_path
):
    import train_v3_h7

    captured = {}

    def capture(_resolved, runtime):
        captured["runtime"] = runtime
        raise RuntimeError("validated-before-cuda")

    monkeypatch.setattr(train_v3_h7, "validate_v3_h7_runtime_config", capture)
    monkeypatch.setattr(sys, "argv", [
        "train_v3_h7.py",
        "--formal-config",
        str(
            ROOT
            / "configs/v3_formal/v3_farmer_cooperation_legacy.yaml"
        ),
        "--checkpoint-path",
        str(tmp_path / "cooperation"),
        "--cooperation-sidecar-capacity",
        "37",
    ])
    with pytest.raises(RuntimeError, match="validated-before-cuda"):
        train_v3_h7.main()
    runtime = captured["runtime"]
    assert runtime.cooperation_runtime_enabled is True
    assert runtime.cooperation_sidecar_capacity == 37
    assert runtime.request_protocol == V3_H71C_REQUEST_PROTOCOL
    assert runtime.replay_protocol == V3_H71C_REPLAY_PROTOCOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h71c_cuda_update_checkpoint_resume_and_shutdown(tmp_path):
    formal = load_formal_config(
        ROOT / "configs/v3_formal/v3_farmer_cooperation_legacy.yaml"
    )
    learner, resolved = create_pilot_learner(formal)
    runtime_config = _runtime(
        batch_size=32,
        num_actors=1,
        games_per_actor=1,
        cooperation_sidecar_capacity=64,
    )
    checkpoint = tmp_path / "cooperation-async.pt"
    trainer = V3AsyncSingleGPUTrainer(
        learner, resolved, runtime_config
    )
    try:
        trainer.collect_episodes(8)
        before = trainer._parameter_update_snapshot()
        metrics = trainer.step()
        assert metrics is not None
        assert metrics.base.cooperation_updated
        assert trainer.stats.cooperation_optimizer_steps == 1
        assert trainer._parameters_changed_since(before)
        boundary = trainer.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["in_flight_slots"] == 0
        assert boundary["pending_requests"] == 0
        assert boundary["cooperation_parameter_vram_bytes"] > 0
        trainer.save_training_checkpoint(
            str(checkpoint), long_running_state={"cycle": 1}
        )
    finally:
        trainer.shutdown()

    resumed_learner, resumed_resolved = create_pilot_learner(formal)
    resumed = V3AsyncSingleGPUTrainer(
        resumed_learner, resumed_resolved, runtime_config
    )
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert resumed.stats.cooperation_optimizer_steps == 1
        resumed.collect_episodes(8)
        assert resumed.step() is not None
        assert resumed.stats.cooperation_optimizer_steps == 2
        boundary = resumed.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["in_flight_slots"] == 0
        assert boundary["pending_requests"] == 0
    finally:
        resumed.shutdown()
