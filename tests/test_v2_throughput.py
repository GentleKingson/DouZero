"""CPU correctness coverage for V2 batched/compact/async infrastructure."""

from __future__ import annotations

import copy
import inspect
import random

import numpy as np
import pytest
import torch

from douzero.env.env import Env
from douzero.models_v2 import (
    ModelV2,
    ModelV2Config,
    observation_batch_to_model_inputs,
    observation_to_model_inputs,
)
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.training.async_single_gpu import AsyncRequestCoordinator, SlotState
from douzero.training.long_running import LongRunningTrainer
from douzero.training.v2_buffer import (
    CompactTensorTransition,
    CompactTensorReplayBuffer,
    Transition,
    action_count_bucket,
)
from douzero.training.v2_trainer import TrainerConfig, V2Trainer


def _spawn_protocol_probe(coordinator, output_queue):
    output_queue.put((coordinator.context.get_start_method(), int(coordinator.states[0])))


class _LocalDDPContext:
    enabled = True
    rank = 0
    world_size = 1
    local_rank = 0
    backend = "gloo"
    device = torch.device("cpu")
    is_rank_zero = True

    @staticmethod
    def all_true(value):
        return bool(value)


class _LocalDDPWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.config = module.config
        self.schema = module.schema
        self.strategy_feature_config = module.strategy_feature_config

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _observations(count: int = 2):
    np.random.seed(41)
    env = Env("adp")
    env.reset()
    observations = []
    while len(observations) < count:
        if len(env.infoset.legal_actions) > 1:
            observations.append(get_obs_v2(env.infoset))
        _obs, _reward, done, _info = env.step(env.infoset.legal_actions[0])
        if done:
            env.reset()
    return observations


def _scalar(model, obs):
    bundle = observation_to_model_inputs(obs)
    return model(
        bundle.state_card_vectors, bundle.state_context_flat,
        bundle.context_card_vectors, bundle.context_flat,
        bundle.history_tokens, bundle.history_key_padding_mask,
        bundle.action_features, bundle.action_mask, bundle.acting_role,
    )


def _compact_with_action_count(obs, action_count: int, policy_step: int):
    transition = Transition(
        obs=obs,
        action_index=0,
        position=obs.public.acting_role,
        target_win=1.0,
        target_score=1.0,
        target_log_score=0.0,
        policy_step=policy_step,
    )
    record = CompactTensorTransition.from_transition(transition)
    bundle = copy.deepcopy(record.model_inputs)
    row = bundle.action_features[:1]
    bundle.action_features = row.repeat(action_count, 1)
    bundle.action_mask = torch.ones(action_count, dtype=torch.bool)
    record.model_inputs = bundle
    return record


def test_scalar_and_batched_forward_padding_and_gather_parity():
    torch.manual_seed(7)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=32, history_layers=1, history_heads=1),
    ).eval()
    observations = _observations(2)
    chosen = torch.tensor([0, len(observations[1].actions.legal_actions) - 1])
    batch = observation_batch_to_model_inputs(
        observations, chosen, pad_to_actions=128
    )
    with torch.inference_mode():
        scalar = [_scalar(model, obs) for obs in observations]
        output = model.forward_batched(
            batch.state_card_vectors, batch.state_context_flat,
            batch.context_card_vectors, batch.context_flat,
            batch.history_tokens, batch.history_key_padding_mask,
            batch.action_features, batch.action_mask, batch.acting_role,
        )
    gathered = output.gather_chosen(chosen)
    for row, expected in enumerate(scalar):
        count = expected.num_actions
        assert torch.allclose(output.win_logit[row, :count], expected.win_logit, atol=2e-6)
        assert torch.allclose(output.score_mean[row, :count], expected.score_mean, atol=2e-6)
        assert not bool(output.action_mask[row, count:].any())
        assert torch.allclose(
            gathered["win_logit"][row], expected.win_logit[chosen[row]], atol=2e-6
        )
    with pytest.raises(ValueError, match="padding"):
        output.gather_chosen(torch.tensor([127, 127]))


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, 8), (8, 8), (9, 16), (16, 16), (17, 32), (64, 64),
     (65, 128), (128, 128), (129, "overflow")],
)
def test_action_bucket_boundaries(count, expected):
    assert action_count_bucket(count) == expected


def test_compact_replay_round_trip_preserves_labels_and_provenance():
    obs = _observations(1)[0]
    transition = Transition(
        obs=obs, action_index=0, position=obs.public.acting_role,
        target_win=1.0, target_score=4.0, target_log_score=1.5,
        target_min_turns_after=2.0, target_min_turns_exact_mask=1.0,
        target_regain_initiative=0.0, target_teammate_finish=1.0,
        target_teammate_finish_mask=1.0, target_spring_probability=0.0,
        target_structure_cost=3.0, trace_index=11, policy_id="league-a",
        teammate_policy_id="league-b", policy_version="snapshot-7", policy_step=19,
    )
    compact = CompactTensorTransition.from_transition(transition)
    restored = CompactTensorTransition.from_state_dict(compact.state_dict())
    assert restored.targets == compact.targets
    assert restored.trace_index == 11
    assert restored.policy_id == "league-a"
    assert restored.teammate_policy_id == "league-b"
    assert (restored.policy_version, restored.policy_step) == ("snapshot-7", 19)
    assert torch.equal(
        restored.model_inputs.action_features, compact.model_inputs.action_features
    )


def test_compact_replay_bucket_selection_is_weighted_by_occupancy():
    obs = _observations(1)[0]
    small = _compact_with_action_count(obs, 8, 1)
    large = _compact_with_action_count(obs, 9, 2)
    replay = CompactTensorReplayBuffer(capacity_transitions=4160)
    replay.add_many([small] * 64 + [large] * 4096)
    rng = random.Random(20260718)
    small_bucket_samples = 0
    rounds = 2000
    for _ in range(rounds):
        batch = replay.sample_minibatch(64, rng)
        assert batch is not None and batch.model_inputs is not None
        small_bucket_samples += int(
            batch.model_inputs[0].action_features.shape[0] == 8
        )
    # Occupancy weighting predicts about 31 selections. Equal bucket weighting
    # would produce about 1000 and is rejected with a deliberately wide bound.
    assert 1 <= small_bucket_samples < 100


def test_compact_replay_add_many_updates_and_evicts_buckets_incrementally():
    obs = _observations(1)[0]
    small = _compact_with_action_count(obs, 8, 1)
    large = _compact_with_action_count(obs, 9, 2)
    replay = CompactTensorReplayBuffer(capacity_transitions=128)
    replay.add_many([small] * 96 + [large] * 32)
    assert replay.bucket_occupancy()[8] == 96
    assert replay.bucket_occupancy()[16] == 32
    replay.add_many([large] * 64)
    assert len(replay) == 128
    assert replay.bucket_occupancy()[8] == 32
    assert replay.bucket_occupancy()[16] == 96


def test_shared_request_protocol_quiescence_timeout_and_shutdown():
    obs = _observations(1)[0]
    coordinator = AsyncRequestCoordinator(
        build_v2_schema(), num_slots=2, request_timeout_seconds=0.01
    )
    assert coordinator.context.get_start_method() == "spawn"
    assert coordinator.quiesce()["free"] == 2
    slot = coordinator.acquire(actor_id=3)
    coordinator.slots.write(slot, observation_to_model_inputs(obs))
    coordinator.submit(slot, request_id=9, policy_snapshot=2)
    request = coordinator.claim_ready(1, wait_seconds=0.1)[0]
    assert request.slot_id == slot
    with pytest.raises(TimeoutError):
        coordinator.wait_done(slot, 9)
    assert SlotState(int(coordinator.states[slot])) == SlotState.FAILED
    coordinator.shutdown()
    assert all(int(state) == SlotState.SHUTDOWN for state in coordinator.states)


def test_shared_request_protocol_is_spawn_picklable():
    coordinator = AsyncRequestCoordinator(build_v2_schema(), num_slots=1)
    output = coordinator.context.Queue()
    process = coordinator.context.Process(
        target=_spawn_protocol_probe, args=(coordinator, output)
    )
    process.start()
    process.join(5)
    assert process.exitcode == 0
    assert output.get(timeout=1) == ("spawn", int(SlotState.FREE))
    output.close()
    coordinator.shutdown()


def test_long_running_controller_has_no_concrete_buffer_access():
    source = inspect.getsource(LongRunningTrainer)
    assert "trainer.buffer" not in source
    assert "trainer.bidding_buffer" not in source


def test_ddp_scalar_learner_ignores_disabled_strategy_auxiliary_heads():
    core = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=16, history_layers=1, history_heads=1),
    )
    trainer = V2Trainer(
        _LocalDDPWrapper(core),
        config=TrainerConfig(
            max_episodes=1,
            optimizer_steps=1,
            batch_size=1,
            buffer_capacity=256,
            exp_epsilon=0.0,
        ),
        distributed_context=_LocalDDPContext(),
    )
    trainer.collect_episodes()
    assert trainer.step() is not None


def test_async_mode_without_cuda_fails_before_startup(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=16, history_layers=1, history_heads=1),
    )
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        V2Trainer(
            model,
            config=TrainerConfig(
                max_episodes=0, optimizer_steps=0,
                v2_training_mode="async_single_gpu", num_actors=2,
                device="cuda",
            ),
        )


def test_cross_topology_resume_is_rejected_before_restore(tmp_path):
    model_config = ModelV2Config(hidden_size=16, history_layers=1, history_heads=1)
    config = TrainerConfig(max_episodes=0, optimizer_steps=0, batch_size=1)
    trainer = V2Trainer(ModelV2(build_v2_schema(), model_config), config=config)
    path = tmp_path / "single.pt"
    trainer.save_training_checkpoint(str(path))
    payload = torch.load(path, weights_only=True)
    payload.update({
        "checkpoint_version": 4,
        "training_topology": "async_single_gpu",
        "num_actors": 2,
        "replay_schema_version": 1,
        "snapshot_publication_semantics": "cycle_quiescent_atomic_copy_v1",
        "request_ordering_semantics": "policy_bucket_role_fifo_microbatch_v1",
    })
    async_path = tmp_path / "async.pt"
    torch.save(payload, async_path)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    from douzero.checkpoint.io import CheckpointCompatibilityError

    with pytest.raises(CheckpointCompatibilityError, match="training_topology"):
        trainer.load_training_checkpoint(str(async_path))
    assert all(
        torch.equal(before[name], value)
        for name, value in trainer.model.state_dict().items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_async_single_gpu_end_to_end_checkpoint_resume_and_shutdown(tmp_path):
    schema = build_v2_schema()
    model_config = ModelV2Config(hidden_size=16, history_layers=1, history_heads=1)
    trainer_config = TrainerConfig(
        max_episodes=2,
        optimizer_steps=1,
        batch_size=1,
        buffer_capacity=256,
        exp_epsilon=0.1,
        device="cuda",
        v2_training_mode="async_single_gpu",
        num_actors=2,
    )
    trainer = V2Trainer(ModelV2(schema, model_config), config=trainer_config)
    checkpoint = tmp_path / "async.pt"
    workers = []
    try:
        trainer.collect_episodes(2)
        boundary = trainer.quiesce_cycle_boundary()
        assert boundary["in_flight_slots"] == 0
        before = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
        }
        assert trainer.step() is not None
        assert any(
            not torch.equal(before[name], value)
            for name, value in trainer.model.state_dict().items()
        )
        trainer.save_training_checkpoint(str(checkpoint))
        workers = list(trainer._async_workers)
    finally:
        trainer.shutdown()
    assert workers and all(not process.is_alive() for process in workers)

    resumed = V2Trainer(ModelV2(schema, model_config), config=trainer_config)
    resumed_workers = []
    try:
        resumed.load_training_checkpoint(str(checkpoint))
        assert resumed.stats.optimizer_steps == 1
        resumed.collect_episodes(2)
        assert resumed.step() is not None
        boundary = resumed.quiesce_cycle_boundary()
        assert boundary["active_slots"] == 0
        resumed_workers = list(resumed._async_workers)
    finally:
        resumed.shutdown()
    assert resumed_workers and all(
        not process.is_alive() for process in resumed_workers
    )
