"""Regression coverage for the Legacy V1 benchmark review findings."""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import subprocess
import threading

import pytest
import torch
from torch import nn

from benchmarks import bench_legacy_training
from douzero.dmc.dmc import (
    _SystemSampler,
    _UpdateBudget,
    _physical_gpu_identifier,
    compute_policy_lag,
)
from douzero.dmc.file_writer import FileWriter
from douzero.dmc.legacy_metrics import LegacyMetricStore
from douzero.runtime import VersionedPolicyPool


POSITIONS = ("landlord", "landlord_up", "landlord_down")


class _TinyPolicy:
    def __init__(self):
        self.models = {
            position: nn.Linear(1, 1, bias=False) for position in POSITIONS
        }

    def get_model(self, position):
        return self.models[position]


def test_update_budget_prevents_concurrent_frame_overshoot():
    budget = _UpdateBudget(0, 192_000, 3_200)
    reservations = []
    result_lock = threading.Lock()

    def reserve_all():
        local = 0
        while budget.reserve():
            local += 1
        with result_lock:
            reservations.append(local)

    threads = [threading.Thread(target=reserve_all) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(reservations) == 60


def test_training_rejects_partial_update_frame_budget(tmp_path):
    from douzero.dmc.arguments import parse_args
    from douzero.dmc.dmc import train

    flags = parse_args([
        "--actor_device_cpu",
        "--training_device", "cpu",
        "--num_buffers", "2",
        "--batch_size", "2",
        "--unroll_length", "3",
        "--total_frames", "7",
        "--savedir", str(tmp_path),
    ])
    with pytest.raises(ValueError, match="total_frames must be divisible"):
        train(flags)


def test_policy_lag_separates_batch_mean_from_oldest_transition():
    mean_lag, max_lag = compute_policy_lag(
        10, torch.tensor([9, 9, 0], dtype=torch.int64)
    )
    assert mean_lag == pytest.approx(4.0)
    assert max_lag == 10


def test_metric_store_reports_integer_transition_max_lag():
    store = LegacyMetricStore(mp.get_context("spawn"))
    store.add_learner(
        {"updates": 1, "frames": 100},
        position="landlord",
        mean_policy_lag=1.25,
        max_policy_lag=7,
    )
    lag = store.snapshot()["policy_lag"]["landlord"]
    assert lag == {"mean_updates": 1.25, "max_updates": 7}
    assert isinstance(lag["max_updates"], int)


def test_policy_pool_resume_initializes_version_continuity():
    context = mp.get_context("spawn")
    source = _TinyPolicy()
    pool = VersionedPolicyPool(
        [_TinyPolicy(), _TinyPolicy()], mp_context=context
    )
    pool.initialize(source.models, version=10_000)

    lease = pool.acquire(owner_id=0)
    assert lease.version == 10_000
    pool.release(lease)
    assert pool.publish(source.models, version=10_001)


def test_failed_file_writer_close_marks_metadata_unsuccessful(tmp_path):
    writer = FileWriter(xpid="failed-run", rootdir=str(tmp_path))
    writer.close(successful=False)
    metadata = json.loads(
        (tmp_path / "failed-run" / "meta.json").read_text(encoding="utf-8")
    )
    assert metadata["successful"] is False
    assert metadata["date_end"] is not None


def test_benchmark_timeout_reaps_process_group(monkeypatch):
    class FakeProcess:
        pid = 4242

        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["train"], timeout)
            return -15

    process = FakeProcess()
    popen_kwargs = {}
    signals = []

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        bench_legacy_training.os, "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    with pytest.raises(TimeoutError, match="process group reaped"):
        bench_legacy_training._run_training(
            ["train"], log_file=io.StringIO(), timeout=1
        )

    assert popen_kwargs["start_new_session"] is True
    assert signals == [(4242, bench_legacy_training.signal.SIGTERM)]


def test_p95_uses_nearest_rank_boundary():
    assert bench_legacy_training._p95(list(range(20))) == 18


def test_gpu_sampler_maps_logical_device_through_visible_devices(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,GPU-deadbeef")
    assert _physical_gpu_identifier("0") == "3"
    assert _physical_gpu_identifier("1") == "GPU-deadbeef"

    captured = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_check_output(command, **kwargs):
        captured["command"] = command
        return "75, 1024\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    assert _SystemSampler("GPU-deadbeef")._gpu_stats() == (75.0, 1024.0)
    assert captured["command"][-2:] == ["-i", "GPU-deadbeef"]
