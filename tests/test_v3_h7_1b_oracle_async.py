from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.training.async_single_gpu import AsyncReplayKey, async_actor_main
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.integration_replay import assert_public_replay_payload
from douzero.v3_hybrid.pilot import build_pilot_resolved_config
from douzero.v3_hybrid.replay import capture_plain_transition
from douzero.v3_hybrid.runtime import (
    V3_H71B_REPLAY_PROTOCOL,
    V3_H71B_REQUEST_PROTOCOL,
    V3H71BOracleAlignment,
    V3H7RuntimeConfig,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.training.h3_learner import (
    _h3_oracle_action_keys,
    bind_v3_h3_oracle_sidecar,
    build_v3_h3_oracle_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]


def _oracle_resolved():
    return build_pilot_resolved_config(
        load_formal_config(ROOT / "configs/v3_formal/v3_oracle_legacy.yaml")
    )


def _runtime(**changes):
    return replace(
        V3H7RuntimeConfig(
            batch_size=4,
            oracle_runtime_enabled=True,
            oracle_sidecar_capacity=32,
            request_protocol=V3_H71B_REQUEST_PROTOCOL,
            replay_protocol=V3_H71B_REPLAY_PROTOCOL,
        ),
        **changes,
    )


def _decision():
    env = Env("adp")
    env.reset()
    observation = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    public_inputs = observation_to_model_inputs(observation)
    privileged = PrivilegedObservation(
        all_handcards=dict(env.infoset.all_handcards),
        acting_role=observation.public.acting_role,
    )
    sidecar = build_v3_h3_oracle_sidecar(
        observation,
        privileged,
        action_index=0,
        public_inputs=public_inputs,
    )
    row = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id="episode",
        deal_id="deal",
        target_transform="raw",
    ).finalize(1.0)
    return env, observation, row, sidecar


def test_oracle_runtime_support_is_fail_closed_before_cuda():
    resolved = _oracle_resolved()
    validate_v3_h7_runtime_config(resolved, _runtime())
    with pytest.raises(ValueError, match="Oracle feature"):
        validate_v3_h7_runtime_config(resolved, V3H7RuntimeConfig(batch_size=4))
    with pytest.raises(ValueError, match="request protocol"):
        V3H7RuntimeConfig(oracle_runtime_enabled=True)
    with pytest.raises(ValueError, match="cannot be smaller"):
        validate_v3_h7_runtime_config(
            resolved, _runtime(oracle_sidecar_capacity=1)
        )


def test_oracle_sidecar_binds_terminal_target_and_stays_out_of_public_replay():
    _env, _observation, row, sidecar = _decision()
    sample = bind_v3_h3_oracle_sidecar(row, sidecar)
    assert sample.target_score == 1.0
    assert sample.target_win == 1.0
    assert sample.action_index == row.selected_action_index
    assert sample.privileged_observation is sidecar.privileged_observation
    payload = row.state_dict()
    assert_public_replay_payload(payload)
    assert "all_handcards" not in repr(payload)
    assert "privileged" not in repr(payload).lower()


def test_oracle_action_alignment_stably_disambiguates_duplicate_rule_rows():
    keys = _h3_oracle_action_keys(([3, 3], [3, 3], [4], [3, 3]))
    assert keys == ((-1, 3, 3), (-2, 3, 3), (4,), (-3, 3, 3))
    assert len(set(keys)) == len(keys)
    assert all(key == tuple(sorted(key)) for key in keys)


def test_oracle_sidecar_rejects_source_and_action_mismatch():
    env, _observation, row, sidecar = _decision()
    env.step(env.infoset.legal_actions[0])
    later = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    other = capture_plain_transition(
        later,
        selected_action_index=0,
        episode_id="episode",
        deal_id="deal",
        target_transform="raw",
    ).finalize(1.0)
    with pytest.raises(ValueError, match="source-state"):
        bind_v3_h3_oracle_sidecar(other, sidecar)
    with pytest.raises(ValueError, match="selected action"):
        bind_v3_h3_oracle_sidecar(
            replace(row, selected_action_index=1), sidecar
        )


def test_oracle_alignment_is_bounded_duplicate_safe_and_quiescent():
    _env, _observation, row, sidecar = _decision()
    key = AsyncReplayKey(actor_id=1, episode_id=2, trace_index=3)
    alignment = V3H71BOracleAlignment(1)
    alignment.add_sidecar(key, sidecar)
    alignment.add_public(key, row)
    paired = alignment.pop_ready()
    assert paired[0][0] is row
    assert paired[0][1].target_score == row.mc_return
    alignment.assert_quiescent()
    with pytest.raises(RuntimeError, match="duplicate"):
        alignment.add_public(key, row)

    pending = V3H71BOracleAlignment(1)
    pending.add_sidecar(key, sidecar)
    with pytest.raises(RuntimeError, match="unmatched Oracle"):
        pending.assert_quiescent()


def test_actor_source_never_constructs_or_forwards_privileged_oracle():
    source = inspect.getsource(async_actor_main)
    assert "V3PrivilegedOracle" not in source
    assert ".oracle(" not in source
    assert "build_v3_h3_oracle_sidecar" in source
