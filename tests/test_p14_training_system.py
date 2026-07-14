"""P14 actor snapshot, AMP, and DDP runtime regression tests."""

from __future__ import annotations

import os
import socket
import queue
from types import SimpleNamespace

import pytest
import torch
from torch import multiprocessing as mp
from torch import nn

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
    context = initialize_distributed(enabled=True, backend="gloo")
    try:
        assert list(context.shard_indices(6)) == list(range(rank, 6, world_size))
        model = context.wrap(nn.Linear(2, 1))
        loss = model(torch.ones(2, 2) * (rank + 1)).sum()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
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
