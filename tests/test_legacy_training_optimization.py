"""Correctness gates for opt-in V1/Legacy training optimizations."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from douzero.dmc.centralized_actor import CentralizedInferenceSlots
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


def test_a1_split_dense1_flag_defaults_off_and_is_opt_in():
    from douzero.dmc.arguments import parse_args

    assert parse_args([]).legacy_actor_split_dense1 is False
    assert parse_args([
        "--legacy_actor_split_dense1"
    ]).legacy_actor_split_dense1 is True


def test_legacy_matmul_precision_defaults_highest_and_accepts_high():
    from douzero.dmc.arguments import parse_args

    assert parse_args([]).legacy_matmul_precision == "highest"
    assert parse_args([
        "--legacy_matmul_precision", "high"
    ]).legacy_matmul_precision == "high"


def test_configure_legacy_matmul_precision(monkeypatch):
    from types import SimpleNamespace

    from douzero.dmc.dmc import configure_legacy_matmul_precision

    observed = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", observed.append)
    configure_legacy_matmul_precision(
        SimpleNamespace(legacy_matmul_precision="high")
    )
    assert observed == ["high"]


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pinned_stager_reuses_cuda_destinations():
    from types import SimpleNamespace

    from douzero.dmc.utils import PinnedBatchStager

    buffers = {"obs_z": torch.arange(24).view(4, 2, 3)}
    stager = PinnedBatchStager(buffers, SimpleNamespace(batch_size=2))
    first_batch = stager.stage(buffers, [0, 1])
    first = stager.to_device("cuda:0", ("obs_z",))["obs_z"]
    stager.mark_h2d("cuda:0")
    stager.stage(buffers, [2, 3])
    second = stager.to_device("cuda:0", ("obs_z",))["obs_z"]
    torch.cuda.synchronize()

    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(second.cpu(), stager.batch["obs_z"])
    assert first_batch is stager.batch


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
