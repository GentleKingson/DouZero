from __future__ import annotations

import inspect
import pickle
import queue
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.training.async_single_gpu import AsyncReplayKey, async_actor_main
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.integration_replay import assert_public_replay_payload
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.replay import capture_plain_transition
from douzero.v3_hybrid.runtime import (
    V3_H71B_REPLAY_PROTOCOL,
    V3_H71B_REQUEST_PROTOCOL,
    V3H71BOracleAlignment,
    V3H7RuntimeConfig,
    _drain_sidecar_queue,
    _h7_alignment_capacity,
    _stop_actor_processes,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.training.h3_learner import (
    _h3_oracle_action_keys,
    bind_v3_h3_oracle_sidecar,
    build_v3_h3_oracle_sidecar,
)
from douzero.v3_hybrid.training.oracle_schedule import (
    OracleGuidingScheduleConfig,
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
    assert (
        sample.privileged_observation.all_handcards
        == sidecar.privileged_observation().all_handcards
    )
    payload = row.state_dict()
    assert_public_replay_payload(payload)
    assert "all_handcards" not in repr(payload)
    assert "privileged" not in repr(payload).lower()
    restored = pickle.loads(pickle.dumps(sidecar))
    assert restored == sidecar
    assert bind_v3_h3_oracle_sidecar(row, restored).target_score == 1.0


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


def test_oracle_alignment_capacity_covers_every_ready_replay_slot():
    runtime = _runtime(batch_size=32, oracle_sidecar_capacity=32)
    assert _h7_alignment_capacity(runtime, runtime.oracle_sidecar_capacity) == 64


def test_primary_h7_cli_selects_oracle_sidecar_protocols_before_cuda(
    monkeypatch, tmp_path
):
    import train_v3_h7

    captured = {}
    original_validate = train_v3_h7.validate_v3_h7_runtime_config

    def capture(resolved, runtime_config):
        original_validate(resolved, runtime_config)
        captured["runtime"] = runtime_config

    monkeypatch.setattr(
        train_v3_h7, "validate_v3_h7_runtime_config", capture
    )
    monkeypatch.setattr(train_v3_h7.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v3_h7.py",
            "--formal-config",
            str(ROOT / "configs/v3_formal/v3_oracle_legacy.yaml"),
            "--checkpoint-path",
            str(tmp_path / "oracle"),
            "--oracle-sidecar-capacity",
            "37",
        ],
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_v3_h7.main()

    runtime = captured["runtime"]
    assert runtime.oracle_runtime_enabled is True
    assert runtime.oracle_sidecar_capacity == 37
    assert runtime.request_protocol == V3_H71B_REQUEST_PROTOCOL
    assert runtime.replay_protocol == V3_H71B_REPLAY_PROTOCOL


def test_primary_h7_cli_caps_oracle_run_at_schedule_completion():
    import train_v3_h7

    schedule = OracleGuidingScheduleConfig(
        enabled=True,
        warmup_updates=10_000,
        guided_updates=50_000,
        finetune_updates=20_000,
    )
    learner = SimpleNamespace(
        base=SimpleNamespace(
            base=SimpleNamespace(
                base=SimpleNamespace(
                    config=SimpleNamespace(schedule=schedule)
                )
            )
        )
    )
    assert train_v3_h7._oracle_update_limit(learner, True) == 80_000
    assert train_v3_h7._oracle_update_limit(learner, False) == 0


def test_shutdown_drains_sidecars_and_terminates_stuck_actor():
    class FakeQueue:
        def __init__(self):
            self.items = [object(), object()]

        def get_nowait(self):
            if not self.items:
                raise queue.Empty
            return self.items.pop()

    class FakeProcess:
        name = "stuck"

        def __init__(self):
            self.terminated = False

        def join(self, _timeout):
            return None

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

    sidecars = FakeQueue()
    process = FakeProcess()
    assert _drain_sidecar_queue(sidecars) == 2
    assert _stop_actor_processes([process], [sidecars], 0.0) == []
    assert process.terminated is True


def test_actor_source_never_constructs_or_forwards_privileged_oracle():
    source = inspect.getsource(async_actor_main)
    assert "V3PrivilegedOracle" not in source
    assert ".oracle(" not in source
    assert "build_v3_h3_oracle_sidecar" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h71b_cuda_oracle_update_checkpoint_resume_and_shutdown(tmp_path):
    formal = load_formal_config(
        ROOT / "configs/v3_formal/v3_oracle_legacy.yaml"
    )
    learner, resolved = create_pilot_learner(formal)
    from douzero.v3_hybrid.runtime import V3AsyncSingleGPUTrainer

    runtime = V3AsyncSingleGPUTrainer(learner, resolved, _runtime())
    checkpoint = tmp_path / "oracle-async.pt"
    try:
        runtime.collect_episodes(1)
        public_before = {
            name: value.detach().clone()
            for name, value in learner.model.state_dict().items()
        }
        oracle_before = {
            name: value.detach().clone()
            for name, value in learner.base.base.base.oracle.state_dict().items()
        }
        policy_before = runtime.policy_step
        metrics = runtime.step()
        assert metrics is not None
        assert metrics.base.base.base.oracle_updated
        assert not metrics.base.base.base.public_updated
        assert runtime.policy_step == policy_before
        assert runtime.stats.oracle_optimizer_steps == 1
        assert runtime.stats.oracle_labels_collected == len(runtime.buffer)
        assert all(
            torch.equal(public_before[name], value)
            for name, value in learner.model.state_dict().items()
        )
        assert any(
            not torch.equal(oracle_before[name], value)
            for name, value in learner.base.base.base.oracle.state_dict().items()
        )
        boundary = runtime.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["pending_requests"] == 0
        assert boundary["oracle_parameter_vram_bytes"] > 0
        runtime.save_training_checkpoint(
            str(checkpoint), long_running_state={"cycle": 1}
        )
        schedule_before = learner.base.base.base.schedule_state().as_dict()
    finally:
        runtime.shutdown()

    resumed_learner, resumed_resolved = create_pilot_learner(formal)
    resumed = V3AsyncSingleGPUTrainer(
        resumed_learner, resumed_resolved, _runtime()
    )
    try:
        assert resumed.load_training_checkpoint(checkpoint) == {"cycle": 1}
        assert (
            resumed_learner.base.base.base.schedule_state().as_dict()
            == schedule_before
        )
        assert resumed.stats.oracle_optimizer_steps == 1
        resumed.collect_episodes(1)
        assert resumed.step() is not None
        assert resumed.stats.oracle_optimizer_steps == 2
        boundary = resumed.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        assert boundary["in_flight_slots"] == 0
        assert boundary["pending_requests"] == 0
    finally:
        resumed.shutdown()
    assert not list(tmp_path.glob("*.tmp"))
