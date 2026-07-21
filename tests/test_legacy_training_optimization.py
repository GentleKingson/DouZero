"""Correctness gates for opt-in V1/Legacy training optimizations."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from douzero.dmc.centralized_actor import (
    CentralQueuePressure,
    CentralizedInferenceSlots,
    PendingRequestScheduler,
    PolicyCopyState,
    RequestState,
    _compatible_key,
    should_throttle,
    wait_for_learner_admission,
)
from douzero.dmc.utils import (
    create_buffers,
    get_batch,
    receive_central_action,
    rollout_ready,
)
from douzero.env.env import Env


def _buffer_flags(**overrides):
    values = {
        "unroll_length": 3,
        "num_buffers": 4,
        "batch_size": 2,
        "legacy_contiguous_buffers": True,
        "pin_memory": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factorized_environment_builds_split_observation_at_source():
    env = Env("adp", observation_backend="factorized")
    observation = env.reset()
    assert set(("z_single", "x_state_single", "x_action")) <= set(observation)
    assert "z_batch" not in observation
    assert "x_batch" not in observation
    assert observation["z_single"].shape == (1, 5, 162)
    assert observation["x_state_single"].shape[0] == 1
    assert observation["x_action"].shape[0] == len(observation["legal_actions"])


def test_default_environment_observation_remains_legacy():
    observation = Env("adp").reset()
    assert "z_batch" in observation and "x_batch" in observation
    assert "z_single" not in observation


def test_contiguous_rollout_batch_has_legacy_time_batch_layout():
    flags = _buffer_flags()
    buffers = create_buffers(flags, ["cpu"])["cpu"]["landlord"]
    for key, tensor in buffers.items():
        for index in range(flags.num_buffers):
            tensor[index].fill_(index)
    free_queue = queue.SimpleQueue()
    full_queue = queue.SimpleQueue()
    full_queue.put(1)
    full_queue.put(3)
    batch = get_batch(
        free_queue, full_queue, buffers, flags, threading.Lock()
    )
    assert batch["obs_x_no_action"].shape == (3, 2, 319)
    assert torch.equal(batch["target"][:, 0], torch.ones(3))
    assert torch.equal(batch["target"][:, 1], torch.full((3,), 3.0))
    assert [free_queue.get(), free_queue.get()] == [1, 3]


def test_equal_length_rollout_submission_is_explicitly_opt_in():
    assert not rollout_ready(100, 100, flush_on_equal=False)
    assert rollout_ready(100, 100, flush_on_equal=True)
    assert rollout_ready(101, 100, flush_on_equal=False)


def test_num_buffers_batch_size_fail_fast(tmp_path):
    from douzero.dmc.arguments import parse_args
    from douzero.dmc.dmc import train

    flags = parse_args([
        "--actor_device_cpu",
        "--training_device", "cpu",
        "--num_buffers", "1",
        "--batch_size", "2",
        "--total_frames", "10",
        "--savedir", str(tmp_path),
    ])
    try:
        train(flags)
    except ValueError as exc:
        assert "num_buffers" in str(exc)
    else:
        raise AssertionError("invalid buffer capacity did not fail fast")


def test_packed_factorized_selection_matches_individual_decisions(seed_factory):
    from douzero.dmc.models_factorized import LegacyFactorizedLandlordModel
    from douzero.env.env import get_obs_factorized

    seed_factory(7331)
    envs = [Env("adp"), Env("adp")]
    observations = []
    for env in envs:
        env.reset()
        observations.append(get_obs_factorized(env.infoset))
    model = LegacyFactorizedLandlordModel().eval()
    counts = [observation["x_action"].shape[0] for observation in observations]
    z = torch.cat([
        torch.from_numpy(observation["z_single"])
        for observation in observations
    ])
    state = torch.cat([
        torch.from_numpy(observation["x_state_single"])
        for observation in observations
    ])
    actions = torch.cat([
        torch.from_numpy(observation["x_action"])
        for observation in observations
    ])
    with torch.inference_mode():
        packed = model.select_actions_packed(z, state, actions, counts)
        individual = torch.stack([
            model.forward_factorized(
                torch.from_numpy(observation["z_single"]),
                torch.from_numpy(observation["x_state_single"]),
                torch.from_numpy(observation["x_action"]),
            )["action"]
            for observation in observations
        ])
    assert torch.equal(packed, individual)


@pytest.mark.parametrize("position,state_width", [
    ("landlord", 319), ("landlord_up", 430), ("landlord_down", 430),
])
def test_split_dense1_packed_values_and_actions_match(position, state_width):
    from douzero.dmc.models_factorized import LegacyFactorizedModel

    torch.manual_seed(20260720)
    model = LegacyFactorizedModel(device="cpu").get_model(position).eval()
    counts = [1, 3, 7]
    z = torch.randn(3, 5, 162)
    state = torch.randn(3, state_width)
    actions = torch.randn(sum(counts), 54)
    with torch.inference_mode():
        old = model.select_actions_packed(
            z, state, actions, counts, split_dense1=False
        )
        new = model.select_actions_packed(
            z, state, actions, counts, split_dense1=True
        )
        lstm, _ = model.lstm(z)
        repeats = torch.tensor(counts)
        legacy_input = torch.cat([
            torch.repeat_interleave(lstm[:, -1], repeats, dim=0),
            torch.repeat_interleave(state, repeats, dim=0), actions,
        ], dim=-1)
        legacy_dense = model.dense1(legacy_input)
        split_dense = model._dense1_split(
            lstm[:, -1], state, actions, repeats=repeats
        )
    assert torch.allclose(split_dense, legacy_dense, rtol=1e-6, atol=1e-6)
    assert torch.equal(new, old)


def test_a1_split_dense1_flag_defaults_off_and_is_opt_in():
    from douzero.dmc.arguments import parse_args

    assert parse_args([]).legacy_actor_split_dense1 is False
    assert parse_args([
        "--legacy_actor_split_dense1"
    ]).legacy_actor_split_dense1 is True


def test_factorized_actor_resolves_role_models_once_per_snapshot():
    from unittest.mock import Mock

    from douzero.dmc.utils import _actor_role_models

    snapshot = Mock()
    snapshot.get_model.side_effect = lambda position: f"model:{position}"
    positions = ["landlord", "landlord_up", "landlord_down"]
    resolved = _actor_role_models(snapshot, "factorized", positions)
    assert resolved == {position: f"model:{position}" for position in positions}
    assert snapshot.get_model.call_count == 3


def test_legacy_actor_does_not_resolve_factorized_role_models():
    from unittest.mock import Mock

    from douzero.dmc.utils import _actor_role_models

    snapshot = Mock()
    assert _actor_role_models(snapshot, "legacy", ["landlord"]) is None
    snapshot.get_model.assert_not_called()


@pytest.mark.parametrize("position,state_width", [
    ("landlord", 319), ("landlord_up", 430), ("landlord_down", 430),
])
def test_bucketed_padded_selection_matches_packed(position, state_width):
    from douzero.dmc.models_factorized import LegacyFactorizedModel

    torch.manual_seed(731)
    model = LegacyFactorizedModel(device="cpu").get_model(position).eval()
    counts = [1, 5, 11]
    bucket = 16
    z = torch.randn(3, 5, 162)
    state = torch.randn(3, state_width)
    padded = torch.zeros(3, bucket, 54)
    mask = torch.arange(bucket)[None, :] < torch.tensor(counts)[:, None]
    packed_rows = []
    for index, count in enumerate(counts):
        padded[index, :count] = torch.randn(count, 54)
        packed_rows.append(padded[index, :count])
    packed_actions = torch.cat(packed_rows)
    with torch.inference_mode():
        packed = model.select_actions_packed(
            z, state, packed_actions, counts, split_dense1=True
        )
        bucketed = model.select_actions_padded(
            z, state, padded, mask, split_dense1=True
        )
    assert torch.equal(bucketed, packed)
    assert torch.all(bucketed < torch.tensor(counts))


def test_centralized_slots_are_actor_isolated_and_capacity_checked():
    slots = CentralizedInferenceSlots(num_actors=2, max_actions=64)
    slots.z.zero_()
    slots.x_state.zero_()
    slots.x_action.zero_()
    z = torch.ones(1, 5, 162)
    state = torch.full((1, 319), 2.0)
    actions = torch.full((3, 54), 3.0)
    slots.write(0, "landlord", z, state, actions)
    assert torch.equal(slots.z[0], torch.ones_like(slots.z[0]))
    assert torch.equal(
        slots.x_state[0, :319], torch.full_like(slots.x_state[0, :319], 2)
    )
    assert torch.equal(
        slots.x_action[0, :3], torch.full_like(slots.x_action[0, :3], 3)
    )
    assert not torch.equal(slots.z[1], slots.z[0])
    with pytest.raises(ValueError, match="slot capacity"):
        slots.write(1, "landlord", z, state, torch.empty(65, 54))


def test_centralized_slots_isolate_actor_environment_pairs():
    slots = CentralizedInferenceSlots(2, 64, envs_per_actor=4)
    slots.z.zero_()
    slots.x_state.zero_()
    slots.x_action.zero_()
    for env_slot in range(4):
        value = env_slot + 1
        slots.write(
            1, "landlord", torch.full((1, 5, 162), value),
            torch.full((1, 319), value), torch.full((2, 54), value),
            env_slot=env_slot,
        )
    assert [int(slots.z[slots.index(1, slot), 0, 0]) for slot in range(4)] == [1, 2, 3, 4]
    assert torch.count_nonzero(slots.z[:4]) == 0


def test_pending_scheduler_rejects_stale_cross_slot_and_duplicate_responses():
    scheduler = PendingRequestScheduler(actor_id=3, env_slots=2)
    first = scheduler.prepare(
        0, policy_slot=1, policy_version=7, position="landlord",
        action_count=5,
    )
    scheduler.mark_queued(first)
    stale = ("ok", 3, 1, first.generation, first.request_id, 2)
    with pytest.raises(RuntimeError, match="stale|cross-slot"):
        scheduler.consume(stale)
    response = ("ok", 3, 0, first.generation, first.request_id, 2)
    request, action = scheduler.consume(response)
    assert request == first and action == 2
    assert scheduler.state(0) == RequestState.CONSUMED
    with pytest.raises(RuntimeError, match="duplicate"):
        scheduler.consume(response)
    second = scheduler.prepare(
        0, policy_slot=1, policy_version=8, position="landlord_up",
        action_count=3,
    )
    scheduler.mark_queued(second)
    with pytest.raises(RuntimeError, match="stale"):
        scheduler.consume(response)


def test_pending_scheduler_cancels_every_outstanding_request():
    scheduler = PendingRequestScheduler(actor_id=0, env_slots=3)
    for slot in range(3):
        request = scheduler.prepare(
            slot, policy_slot=0, policy_version=1, position="landlord",
            action_count=2,
        )
        scheduler.mark_queued(request)
    assert scheduler.pending == 3
    scheduler.cancel_all()
    assert scheduler.pending == 0
    assert all(scheduler.state(slot) == RequestState.CANCELLED for slot in range(3))


def test_policy_copy_state_publishes_only_matching_complete_version():
    state = PolicyCopyState()
    token = object()
    state.begin(4, token)
    assert state.loaded_version == -1
    assert state.loading_version == 4
    with pytest.raises(RuntimeError, match="version changed"):
        state.finish(5)
    state.finish(4)
    assert state.loaded_version == 4
    assert state.loading_version is None


def test_legal_action_buckets_allow_different_packed_counts():
    scheduler = PendingRequestScheduler(actor_id=0, env_slots=2)
    left = scheduler.prepare(
        0, policy_slot=1, policy_version=2, position="landlord",
        action_count=17,
    )
    right = scheduler.prepare(
        1, policy_slot=1, policy_version=2, position="landlord",
        action_count=31,
    )
    assert _compatible_key(left, 512) == _compatible_key(right, 512)


def test_central_queue_pressure_tracks_depth_without_polling():
    import multiprocessing

    pressure = CentralQueuePressure(
        multiprocessing.get_context("spawn"), request_slots=3
    )
    now = time.perf_counter_ns()
    pressure.begin_actor_enqueue(0, now)
    pressure.finish_actor_enqueue(0, now + 10)
    pressure.begin_actor_enqueue(1, now + 20)
    pressure.finish_actor_enqueue(1, now + 30)
    snapshot = pressure.snapshot()
    assert snapshot["ingress_queue_depth"] == 2
    assert snapshot["oldest_actor_enqueue_ns"] == now + 10
    pressure.server_received(0, now + 40)
    assert pressure.snapshot()["local_pending_depth"] == 1
    pressure.executing([0])
    assert pressure.snapshot()["executing_requests"] == 1
    assert pressure.snapshot()["total_backlog"] == 2
    pressure.completed([0], now + 50)
    snapshot = pressure.snapshot()
    assert snapshot["total_backlog"] == 1
    assert snapshot["oldest_actor_enqueue_ns"] == now + 30
    pressure.cancel([1])
    assert pressure.snapshot()["total_backlog"] == 0


def test_pressure_accepts_completion_before_actor_put_returns():
    import multiprocessing

    pressure = CentralQueuePressure(
        multiprocessing.get_context("spawn"), request_slots=1
    )
    now = time.perf_counter_ns()
    pressure.begin_actor_enqueue(0, now)
    pressure.server_received(0, now + 1)
    pressure.executing([0])
    pressure.completed([0], now + 2)
    pressure.finish_actor_enqueue(0, now + 3)
    assert pressure.snapshot()["total_backlog"] == 0


def test_pressure_preserves_actor_enqueue_and_server_receive_timestamps():
    import multiprocessing

    pressure = CentralQueuePressure(
        multiprocessing.get_context("spawn"), request_slots=1
    )
    pressure.begin_actor_enqueue(0, 100)
    pressure.finish_actor_enqueue(0, 200)
    pressure.server_received(0, 700)
    assert pressure.timestamps(0) == (200, 700)
    snapshot = pressure.snapshot()
    assert snapshot["oldest_actor_enqueue_ns"] == 200
    assert snapshot["oldest_server_receive_ns"] == 700


@pytest.mark.parametrize("mode", ["fixed_threshold", "predicted_drain_time"])
def test_learner_throttle_modes_exit_when_backlog_drains(mode):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    pressure = CentralQueuePressure(context, request_slots=1)
    pressure.begin_actor_enqueue(0, time.perf_counter_ns())
    pressure.finish_actor_enqueue(0, time.perf_counter_ns())
    result = []
    worker = threading.Thread(target=lambda: result.append(
        wait_for_learner_admission(
            pressure, threading.Event(), mode, high_watermark=1,
            deadline_ms=0.001, drain_target_ms=0.001,
        )
    ))
    worker.start()
    time.sleep(0.02)
    pressure.cancel([0])
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result[0][0] is True


def test_stop_event_bounds_throttle_wait():
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    pressure = CentralQueuePressure(context, request_slots=1)
    pressure.begin_actor_enqueue(0, time.perf_counter_ns())
    pressure.finish_actor_enqueue(0, time.perf_counter_ns())
    stop = threading.Event()
    worker = threading.Thread(target=wait_for_learner_admission, args=(
        pressure, stop, "fixed_threshold",
    ), kwargs={
        "high_watermark": 1, "deadline_ms": 1,
        "drain_target_ms": 1,
    })
    worker.start()
    stop.set()
    worker.join(timeout=0.2)
    assert not worker.is_alive()


def test_predicted_throttle_uses_total_backlog_and_service_rate():
    snapshot = {
        "valid": True, "total_backlog": 10,
        "recent_requests_per_ms": 2.0, "oldest_actor_age_ms": 0.0,
    }
    assert should_throttle(
        snapshot, "predicted_drain_time", high_watermark=100,
        deadline_ms=100, drain_target_ms=4.9,
    )
    assert not should_throttle(
        snapshot, "predicted_drain_time", high_watermark=100,
        deadline_ms=100, drain_target_ms=5.1,
    )


def test_centralized_response_timeout_abort_and_shutdown():
    responses = queue.Queue()
    with pytest.raises(TimeoutError, match="timed out"):
        receive_central_action(responses, 0.001)
    responses.put(("error", "worker crashed"))
    with pytest.raises(RuntimeError, match="worker crashed"):
        receive_central_action(responses, 0.1)
    responses.put(("shutdown", "training is shutting down"))
    assert receive_central_action(responses, 0.1) is None
    responses.put(("ok", 7))
    assert receive_central_action(responses, 0.1) == 7
