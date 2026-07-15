"""P14 actor snapshot, AMP, and DDP runtime regression tests."""

from __future__ import annotations

import os
import socket
import queue
import random
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import multiprocessing as mp
from torch import nn
import torch.distributed as dist

from douzero.runtime import SafeMixedPrecision, VersionedPolicyPool
from douzero.runtime.distributed import DistributedContext, initialize_distributed
from douzero.dmc.utils import get_batch


POSITIONS = ("landlord", "landlord_up", "landlord_down")


class _TinyPolicy:
    def __init__(self) -> None:
        self.models = {position: nn.Linear(1, 1, bias=False) for position in POSITIONS}

    def get_model(self, position):
        return self.models[position]

    def share_memory(self) -> None:
        for model in self.models.values():
            model.share_memory()


def _set_all(policy: _TinyPolicy, value: float) -> None:
    with torch.no_grad():
        for model in policy.models.values():
            model.weight.fill_(value)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    context = initialize_distributed(enabled=True, backend="gloo")
    try:
        assert list(context.shard_indices(6)) == list(range(rank, 6, world_size))
        model = context.wrap(nn.Linear(2, 1))
        loss = model(torch.ones(2, 2) * (rank + 1)).sum()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
    finally:
        context.close()


def _ddp_v2_readiness_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    context = initialize_distributed(enabled=True, backend="gloo")
    try:
        from douzero.models_v2 import ModelV2, ModelV2Config
        from douzero.observation import build_v2_schema
        from douzero.training import TrainerConfig, V2Trainer
        from douzero.training.v2_buffer import Episode, Transition

        core = ModelV2(
            build_v2_schema(),
            ModelV2Config(hidden_size=32, history_layers=1, history_heads=4),
        )
        model = context.wrap(core)
        model.config = core.config
        model.schema = core.schema
        model.strategy_feature_config = core.strategy_feature_config
        trainer = V2Trainer(
            model,
            config=TrainerConfig(
                max_episodes=0,
                optimizer_steps=0,
                batch_size=1,
                buffer_capacity=4,
            ),
            distributed_context=context,
        )
        # Independently collected replay: rank 0 has one real transition while
        # rank 1 remains empty.
        if rank == 0:
            from douzero.env.env import Env
            from douzero.observation.encode_v2 import get_obs_v2

            env = Env("adp")
            env.reset()
            while env._acting_player_position != "landlord":
                env.step(env.infoset.legal_actions[0])
            trainer.buffer.add_episode(
                Episode(
                    transitions=[
                        Transition(
                            obs=get_obs_v2(env.infoset),
                            action_index=0,
                            position="landlord",
                            target_win=1.0,
                            target_score=1.0,
                            target_log_score=1.0,
                        )
                    ],
                    terminal_result={},
                )
            )
        assert trainer.step() is None
        assert trainer.stats.optimizer_steps == 0
    finally:
        context.close()


def _ddp_rank_one_nan_worker(
    rank: int, world_size: int, port: int, failure_stage: str
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    context = initialize_distributed(enabled=True, backend="gloo")
    try:
        model = context.wrap(nn.Linear(2, 1, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        amp = SafeMixedPrecision(
            torch.device("cpu"), enabled=True, dtype="bfloat16"
        )
        calls = 0
        clip_calls = 0

        def closure():
            nonlocal calls
            calls += 1
            loss = model(torch.ones(2, 2)).square().mean()
            if failure_stage == "loss" and rank == 1 and calls == 1:
                loss = loss * torch.tensor(float("nan"))
            return loss

        def clip(parameters, max_norm, error_if_nonfinite=False):
            nonlocal clip_calls
            clip_calls += 1
            if failure_stage == "gradient" and rank == 1 and clip_calls == 1:
                return torch.tensor(float("nan"))
            return nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=error_if_nonfinite
            )

        result = amp.step(
            closure,
            optimizer,
            model.parameters(),
            max_grad_norm=10.0,
            collective_all_true=context.all_true,
            synchronize_abandoned_backward=True,
            clip_grad_norm=clip,
        )
        assert calls == 2
        assert result.fell_back
        assert amp.fallback_count == 1
        parameter = next(model.parameters()).detach()
        gathered = [torch.empty_like(parameter) for _ in range(world_size)]
        dist.all_gather(gathered, parameter)
        assert all(torch.equal(gathered[0], peer) for peer in gathered[1:])
    finally:
        context.close()


def _ddp_v2_self_play_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    context = initialize_distributed(enabled=True, backend="gloo")
    try:
        import numpy as np

        from douzero.models_v2 import ModelV2, ModelV2Config
        from douzero.observation import build_v2_schema
        from douzero.training import TrainerConfig, V2Trainer

        rank_seed = 2400 + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        core = ModelV2(
            build_v2_schema(),
            ModelV2Config(hidden_size=16, history_layers=1, history_heads=4),
        )
        model = context.wrap(core)
        model.config = core.config
        model.schema = core.schema
        model.strategy_feature_config = core.strategy_feature_config
        trainer = V2Trainer(
            model,
            config=TrainerConfig(
                seed=rank_seed,
                rng_seed=rank_seed,
                max_episodes=1,
                optimizer_steps=2,
                batch_size=1,
                buffer_capacity=256,
                exp_epsilon=0.0,
            ),
            distributed_context=context,
        )
        ddp_forward_count = 0
        training_forward = trainer._forward_bundle

        def count_training_forward(*args, **kwargs):
            nonlocal ddp_forward_count
            ddp_forward_count += 1
            return training_forward(*args, **kwargs)

        trainer._forward_bundle = count_training_forward
        trainer.collect_episodes()
        # Rank-local games must never enter the DDP wrapper.
        assert ddp_forward_count == 0
        transition_counts = [
            torch.zeros(1, dtype=torch.int64) for _ in range(world_size)
        ]
        dist.all_gather(
            transition_counts,
            torch.tensor([trainer.stats.transitions_collected], dtype=torch.int64),
        )
        assert transition_counts[0].item() != transition_counts[1].item()

        for _ in range(2):
            assert trainer.step() is not None
        assert trainer.stats.optimizer_steps == 2
        assert ddp_forward_count == 2
    finally:
        context.close()


def test_policy_snapshot_is_stable_for_whole_episode():
    ctx = mp.get_context("spawn")
    slots = [_TinyPolicy(), _TinyPolicy()]
    source = _TinyPolicy()
    _set_all(source, 1.0)
    pool = VersionedPolicyPool(slots, mp_context=ctx)
    pool.initialize(source.models)

    episode_zero = pool.acquire()
    assert episode_zero.version == 0
    assert episode_zero.model.get_model("landlord").weight.item() == 1.0

    _set_all(source, 2.0)
    assert pool.publish(source.models, version=1)
    # The slot leased by the in-flight episode was not modified.
    assert episode_zero.model.get_model("landlord").weight.item() == 1.0

    episode_one = pool.acquire()
    assert episode_one.version == 1
    assert episode_one.model.get_model("landlord").weight.item() == 2.0

    _set_all(source, 3.0)
    # Both slots are leased. Publication skips instead of overwriting either.
    assert pool.publish(source.models, version=2) is False
    pool.release(episode_zero)
    assert pool.publish(source.models, version=2)
    assert episode_one.model.get_model("landlord").weight.item() == 2.0
    pool.release(episode_one)
    assert pool.reader_counts() == (0, 0)


def test_policy_snapshot_rejects_double_release_without_stealing_reader():
    ctx = mp.get_context("spawn")
    pool = VersionedPolicyPool([_TinyPolicy(), _TinyPolicy()], mp_context=ctx)
    first = pool.acquire(owner_id=1)
    second = pool.acquire(owner_id=2)
    pool.release(first)
    with pytest.raises(RuntimeError, match="does not hold"):
        pool.release(first)
    assert sum(pool.reader_counts()) == 1
    pool.release(second)


def test_policy_snapshot_parent_recovers_crashed_actor_lease():
    ctx = mp.get_context("spawn")
    slots = [_TinyPolicy(), _TinyPolicy()]
    source = _TinyPolicy()
    pool = VersionedPolicyPool(slots, mp_context=ctx)
    pool.initialize(source.models)
    pool.acquire(owner_id=7)
    assert pool.recover_owner(7)
    assert not pool.recover_owner(7)
    assert pool.reader_counts() == (0, 0)
    assert pool.publish(source.models, version=1)


def test_amp_cpu_bfloat16_optimizer_step_changes_parameters():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    amp = SafeMixedPrecision(torch.device("cpu"), enabled=True, dtype="bfloat16")
    inputs = torch.ones(8, 4)
    target = torch.zeros(8, 1)
    before = model.weight.detach().clone()

    result = amp.step(
        lambda: ((model(inputs) - target) ** 2).mean(),
        optimizer,
        model.parameters(),
        max_grad_norm=10.0,
    )

    assert result.amp_used
    assert torch.isfinite(result.loss)
    assert not torch.equal(before, model.weight.detach())


def test_v2_cpu_bfloat16_amp_one_training_step():
    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training import LossConfig, TrainerConfig, V2Trainer

    torch.manual_seed(1400)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=64, history_layers=1, history_heads=4),
    )
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=TrainerConfig(
            seed=1400,
            rng_seed=1400,
            max_episodes=4,
            optimizer_steps=1,
            batch_size=1,
            buffer_capacity=128,
            amp_enabled=True,
            amp_dtype="bfloat16",
        ),
    )
    stats = trainer.train()
    assert stats.optimizer_steps == 1
    assert trainer.stats_last_run_changed
    assert stats.amp_fallbacks == 0


def test_amp_nonfinite_disables_amp_and_retries_float32():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    amp = SafeMixedPrecision(torch.device("cpu"), enabled=True, dtype="bfloat16")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls == 1:
            return parameter.sum() * torch.tensor(float("nan"))
        return parameter.square().sum()

    result = amp.step(closure, optimizer, [parameter], max_grad_norm=10.0)
    assert result.fell_back
    assert not result.amp_used
    assert amp.enabled is False
    assert amp.fallback_count == 1
    assert calls == 2
    assert parameter.item() == pytest.approx(0.8)


def test_amp_retry_restores_sampling_and_torch_rng_state():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    amp = SafeMixedPrecision(torch.device("cpu"), enabled=True, dtype="bfloat16")
    rng = random.Random(14)
    torch.manual_seed(14)
    draws = []

    def capture():
        return rng.getstate(), torch.random.get_rng_state()

    def restore(state):
        rng.setstate(state[0])
        torch.random.set_rng_state(state[1])

    def closure():
        draws.append((rng.randrange(1000), float(torch.rand(()))))
        if len(draws) == 1:
            return parameter.sum() * torch.tensor(float("nan"))
        return parameter.square().sum()

    amp.step(
        closure,
        optimizer,
        [parameter],
        max_grad_norm=10.0,
        capture_retry_state=capture,
        restore_retry_state=restore,
    )
    assert draws[0] == draws[1]


def test_cpu_float16_amp_is_rejected():
    with pytest.raises(ValueError, match="CPU autocast"):
        SafeMixedPrecision(torch.device("cpu"), enabled=True, dtype="float16")


def test_pin_memory_flag_is_portable_without_cuda(monkeypatch):
    free = queue.Queue()
    full = queue.Queue()
    full.put(0)
    buffers = {"x": [torch.ones(2)]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = get_batch(
        free, full, buffers,
        SimpleNamespace(batch_size=1, pin_memory=True),
        __import__("threading").Lock(),
    )
    assert batch["x"].shape == (2, 1)
    assert free.get() == 0


def test_get_batch_shutdown_sentinel_wakes_blocked_learner():
    free = queue.Queue()
    full = queue.Queue()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            get_batch(
                free,
                full,
                {"x": [torch.ones(2)]},
                SimpleNamespace(batch_size=1, pin_memory=False),
                threading.Lock(),
            )
        )
    )
    thread.start()
    full.put(None)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result == [None]


def test_legacy_learner_failure_reaches_monitor_and_requests_shutdown():
    from douzero.dmc.dmc import _LearnerThreadSupervisor

    stop_event = threading.Event()
    supervisor = _LearnerThreadSupervisor(stop_event)
    free = queue.Queue()
    full = queue.Queue()
    blocked_result = []

    def blocked_learner():
        blocked_result.append(
            get_batch(
                free,
                full,
                {"x": [torch.ones(2)]},
                SimpleNamespace(batch_size=1, pin_memory=False),
                threading.Lock(),
            )
        )

    def fail_optimizer_step():
        raise FloatingPointError("float32 retry remained non-finite")

    blocked = threading.Thread(
        target=supervisor.run,
        args=(blocked_learner,),
        name="blocked-legacy-learner",
    )
    failing = threading.Thread(
        target=supervisor.run,
        args=(fail_optimizer_step,),
        name="failing-legacy-learner",
    )
    blocked.start()
    failing.start()
    failing.join(timeout=2)
    assert not failing.is_alive()
    assert stop_event.is_set()
    with pytest.raises(FloatingPointError, match="float32 retry remained"):
        supervisor.raise_if_failed()
    full.put(None)
    blocked.join(timeout=2)
    assert not blocked.is_alive()
    assert blocked_result == [None]


@pytest.mark.parametrize(
    ("curriculum_enabled", "lambda_bc", "message"),
    [
        (True, 0.0, "curriculum/coach-label"),
        (False, 0.5, "RL\\+BC"),
    ],
)
def test_ddp_unsupported_file_side_effects_fail_fast(
    curriculum_enabled, lambda_bc, message
):
    from train_v2 import _validate_ddp_features

    config = SimpleNamespace(
        curriculum=SimpleNamespace(enabled=curriculum_enabled)
    )
    with pytest.raises(NotImplementedError, match=message):
        _validate_ddp_features(
            config,
            SimpleNamespace(
                human_prior_enabled=False,
                strategy_aux_enabled=False,
            ),
            SimpleNamespace(lambda_bc=lambda_bc),
            DistributedContext(enabled=True),
        )


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        (
            SimpleNamespace(
                human_prior_enabled=True,
                strategy_aux_enabled=False,
            ),
            "human_prior_enabled",
        ),
        (
            SimpleNamespace(
                human_prior_enabled=False,
                strategy_aux_enabled=True,
            ),
            "strategy_aux_enabled",
        ),
    ],
)
def test_ddp_rejects_enabled_optional_head_without_loss(model_config, message):
    from train_v2 import _validate_ddp_features

    loss_config = SimpleNamespace(
        lambda_bc=0.0,
        lambda_min_turns=0.0,
        lambda_regain_initiative=0.0,
        lambda_teammate_finish=0.0,
        lambda_spring=0.0,
        lambda_structure=0.0,
    )
    with pytest.raises(ValueError, match=message):
        _validate_ddp_features(
            SimpleNamespace(curriculum=SimpleNamespace(enabled=False)),
            model_config,
            loss_config,
            DistributedContext(enabled=True),
        )


def test_disabled_distributed_context_has_nonduplicating_single_rank_shard():
    context = DistributedContext(enabled=False)
    assert context.is_rank_zero
    assert list(context.shard_indices(4)) == [0, 1, 2, 3]


@pytest.mark.timeout(60)
def test_ddp_two_process_cpu_gloo_smoke():
    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("sandbox disallows loopback sockets required by gloo")
    mp.spawn(_ddp_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.timeout(60)
def test_v2_trainer_ddp_skips_asymmetric_replay_readiness_together():
    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("sandbox disallows loopback sockets required by gloo")
    mp.spawn(_ddp_v2_readiness_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.timeout(120)
def test_v2_trainer_ddp_independent_self_play_then_two_optimizer_steps():
    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("sandbox disallows loopback sockets required by gloo")
    mp.spawn(_ddp_v2_self_play_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failure_stage", ["loss", "gradient"])
def test_ddp_amp_rank_one_nan_makes_all_ranks_retry(failure_stage):
    try:
        port = _free_port()
    except PermissionError:
        pytest.skip("sandbox disallows loopback sockets required by gloo")
    mp.spawn(
        _ddp_rank_one_nan_worker,
        args=(2, port, failure_stage),
        nprocs=2,
        join=True,
    )
