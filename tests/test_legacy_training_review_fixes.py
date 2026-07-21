"""Regression coverage for the Legacy V1 benchmark review findings."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing as mp
import subprocess
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks import bench_legacy_training
from douzero.dmc.dmc import (
    _SystemSampler,
    _TrainingTransactions,
    _UpdateBudget,
    _physical_gpu_identifier,
    _save_legacy_sidecars,
    compute_policy_lag,
)
from douzero.dmc.file_writer import FileWriter
from douzero.dmc.legacy_metrics import ActorMetricRecorder, LegacyMetricStore
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
        while (reservation := budget.reserve()) is not None:
            local += 1
            budget.commit(reservation)
        with result_lock:
            reservations.append(local)

    threads = [threading.Thread(target=reserve_all) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(reservations) == 60


def test_update_budget_cancel_restores_capacity():
    budget = _UpdateBudget(0, 6_400, 3_200)
    cancelled = budget.reserve()
    assert cancelled is not None
    budget.cancel(cancelled)

    first = budget.reserve()
    second = budget.reserve()
    assert first is not None and second is not None
    assert budget.reserve() is None
    budget.commit(first)
    budget.commit(second)


def test_checkpoint_waits_for_optimizer_and_progress_transaction():
    transactions = _TrainingTransactions(POSITIONS)
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    progress = {
        "frames": 0,
        "position_frames": {position: 0 for position in POSITIONS},
        "learner_updates": 0,
    }
    optimizer_stepped = threading.Event()
    checkpoint_started = threading.Event()
    allow_progress_commit = threading.Event()
    checkpoint_done = threading.Event()
    checkpoint_payload = []
    errors = []

    def learner_update():
        try:
            with transactions.update("landlord"):
                optimizer.zero_grad()
                loss = (model(torch.ones(1, 1)) - 1.0).square().mean()
                loss.backward()
                optimizer.step()
                optimizer_stepped.set()
                if not allow_progress_commit.wait(timeout=2):
                    raise TimeoutError("test did not release progress commit")
                with transactions.state_lock:
                    progress["frames"] += 3_200
                    progress["position_frames"]["landlord"] += 3_200
                    progress["learner_updates"] += 1
        except BaseException as exc:
            errors.append(exc)

    def save_checkpoint():
        try:
            if not optimizer_stepped.wait(timeout=2):
                raise TimeoutError("optimizer step was not reached")
            checkpoint_started.set()
            with transactions.snapshot():
                buffer = io.BytesIO()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "frames": progress["frames"],
                        "position_frames": dict(progress["position_frames"]),
                        "learner_updates": progress["learner_updates"],
                    },
                    buffer,
                )
                checkpoint_payload.append(buffer.getvalue())
            checkpoint_done.set()
        except BaseException as exc:
            errors.append(exc)

    learner = threading.Thread(target=learner_update)
    checkpoint = threading.Thread(target=save_checkpoint)
    learner.start()
    assert optimizer_stepped.wait(timeout=2)
    checkpoint.start()
    assert checkpoint_started.wait(timeout=2)

    # The model and optimizer have advanced, but checkpoint cannot observe
    # them until the matching frame/update counters are committed.
    assert progress["frames"] == 0
    assert not checkpoint_done.wait(timeout=0.1)
    allow_progress_commit.set()
    learner.join(timeout=2)
    checkpoint.join(timeout=2)

    assert not learner.is_alive()
    assert not checkpoint.is_alive()
    assert errors == []
    restored = torch.load(
        io.BytesIO(checkpoint_payload[0]), map_location="cpu", weights_only=True
    )
    assert restored["frames"] == 3_200
    assert restored["position_frames"] == {
        "landlord": 3_200,
        "landlord_up": 0,
        "landlord_down": 0,
    }
    assert restored["learner_updates"] == 1
    restored_model = nn.Linear(1, 1, bias=False)
    restored_model.load_state_dict(restored["model"])
    assert torch.equal(restored_model.weight, model.weight)
    restored_optimizer = torch.optim.RMSprop(restored_model.parameters(), lr=0.1)
    restored_optimizer.load_state_dict(restored["optimizer"])
    optimizer_state = next(iter(restored_optimizer.state.values()))
    assert int(optimizer_state["step"].item()) == 1


def test_training_rejects_partial_update_frame_budget(tmp_path):
    from douzero.dmc.arguments import parse_args, parser
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

    help_text = " ".join(parser.format_help().split())
    assert "must be divisible by unroll_length * batch_size" in help_text


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


def test_actor_metric_flush_drops_pre_reset_generation():
    store = LegacyMetricStore(mp.get_context("spawn"))
    recorder = ActorMetricRecorder(store, flush_every=1_000)
    recorder.add(decisions=17, games=2)
    add_actor_entered = threading.Event()
    allow_add_actor = threading.Event()
    original_add_actor = store.add_actor

    def delayed_add_actor(counters, legal_hist, *, generation=None):
        add_actor_entered.set()
        assert allow_add_actor.wait(timeout=2)
        return original_add_actor(
            counters, legal_hist, generation=generation
        )

    store.add_actor = delayed_add_actor
    flush_thread = threading.Thread(target=recorder.flush)
    flush_thread.start()
    assert add_actor_entered.wait(timeout=2)
    store.reset()
    allow_add_actor.set()
    flush_thread.join(timeout=2)

    assert not flush_thread.is_alive()
    actor = store.snapshot()["counts"]["actor"]
    assert actor["decisions"] == 0
    assert actor["games"] == 0


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


def test_file_writer_concurrent_logs_are_complete(tmp_path):
    writer = FileWriter(xpid="concurrent-run", rootdir=str(tmp_path))
    barrier = threading.Barrier(len(POSITIONS))
    errors = []
    writes_per_role = 100

    def write_role(position):
        try:
            barrier.wait(timeout=2)
            for index in range(writes_per_role):
                writer.log({
                    "position": position,
                    "iteration": index,
                    f"loss_{position}": index / 10,
                })
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=write_role, args=(position,))
        for position in POSITIONS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    writer.close()

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    base = tmp_path / "concurrent-run"
    with (base / "fields.csv").open(newline="") as handle:
        fieldnames = next(csv.reader(handle))
    assert len(fieldnames) == len(set(fieldnames))
    with (base / "logs.csv").open(newline="") as handle:
        data_lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(data_lines, fieldnames=fieldnames))
    ticks = [int(row["_tick"]) for row in rows]
    assert len(rows) == len(POSITIONS) * writes_per_role
    assert sorted(ticks) == list(range(len(rows)))


def test_legacy_sidecars_are_atomic_and_retained(tmp_path):
    learner = _TinyPolicy()
    for frames in (3_200, 6_400, 9_600):
        _save_legacy_sidecars(learner, str(tmp_path), frames, retention=2)

    expected_frames = {6_400, 9_600}
    for position in POSITIONS:
        names = sorted(tmp_path.glob(f"{position}_weights_*.ckpt"))
        assert {
            int(path.stem.removeprefix(f"{position}_weights_"))
            for path in names
        } == expected_frames
        assert torch.load(names[-1], map_location="cpu", weights_only=True)
    assert list(tmp_path.glob("*.tmp")) == []


def test_legacy_sidecars_can_be_disabled(tmp_path):
    _save_legacy_sidecars(_TinyPolicy(), str(tmp_path), 3_200, retention=0)
    assert list(tmp_path.iterdir()) == []


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


def test_formal_benchmark_requires_immutable_image_digest():
    digest = "sha256:" + "a" * 64
    git_sha = "b" * 40
    with pytest.raises(ValueError, match="require --docker_image_digest"):
        bench_legacy_training._validate_evidence_mode(SimpleNamespace(
            formal=True, allow_dirty=False, docker_image_digest=None,
            expected_git_sha=git_sha,
        ))
    with pytest.raises(ValueError, match="cannot use --allow_dirty"):
        bench_legacy_training._validate_evidence_mode(SimpleNamespace(
            formal=True, allow_dirty=True, docker_image_digest=digest,
            expected_git_sha=git_sha,
        ))
    for invalid_digest in ("anything", "sha256:test", "sha256:" + "A" * 64):
        with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
            bench_legacy_training._validate_evidence_mode(SimpleNamespace(
                formal=True, allow_dirty=False,
                docker_image_digest=invalid_digest,
                expected_git_sha=git_sha,
            ))
    with pytest.raises(ValueError, match="require --expected_git_sha"):
        bench_legacy_training._validate_evidence_mode(SimpleNamespace(
            formal=True, allow_dirty=False, docker_image_digest=digest,
            expected_git_sha=None,
        ))
    with pytest.raises(ValueError, match="full 40-character lowercase"):
        bench_legacy_training._validate_evidence_mode(SimpleNamespace(
            formal=True, allow_dirty=False, docker_image_digest=digest,
            expected_git_sha="deadbeef",
        ))
    bench_legacy_training._validate_evidence_mode(SimpleNamespace(
        formal=True, allow_dirty=False, docker_image_digest=digest,
        expected_git_sha=git_sha,
    ))


def test_formal_benchmark_provenance_fails_closed():
    git_sha = "b" * 40
    args = SimpleNamespace(
        formal=True, allow_dirty=False, expected_git_sha=git_sha,
    )
    with pytest.raises(RuntimeError, match="verify the Git SHA"):
        bench_legacy_training._validate_provenance(args, {
            "git_sha": None, "git_status_porcelain": [],
        })
    with pytest.raises(RuntimeError, match="worktree status"):
        bench_legacy_training._validate_provenance(args, {
            "git_sha": git_sha, "git_status_porcelain": None,
        })
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        bench_legacy_training._validate_provenance(args, {
            "git_sha": git_sha, "git_status_porcelain": [" M train.py"],
        })
    with pytest.raises(RuntimeError, match="Git SHA mismatch"):
        bench_legacy_training._validate_provenance(args, {
            "git_sha": "c" * 40, "git_status_porcelain": [],
        })
    bench_legacy_training._validate_provenance(args, {
        "git_sha": git_sha, "git_status_porcelain": [],
    })


def test_benchmark_environment_reads_requested_source_root(monkeypatch, tmp_path):
    calls = []

    def fake_check_output(command, **kwargs):
        calls.append((tuple(command), kwargs.get("cwd")))
        if command[0] == "nvidia-smi":
            return "0, uuid, gpu, driver, 1000\n"
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[1:3] == ["status", "--porcelain"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(
        bench_legacy_training.subprocess, "check_output", fake_check_output
    )
    monkeypatch.setattr(
        bench_legacy_training.platform, "platform", lambda: "test-platform"
    )
    environment = bench_legacy_training._environment(tmp_path)

    assert environment["git_sha"] == "a" * 40
    git_calls = [call for call in calls if call[0][0] == "git"]
    assert git_calls
    assert all(cwd == tmp_path for _, cwd in git_calls)


def test_benchmark_checkpoint_mode_is_explicit():
    assert bench_legacy_training._checkpoint_cli_args(False) == [
        "--disable_checkpoint"
    ]
    assert bench_legacy_training._checkpoint_cli_args(True) == [
        "--no-disable_checkpoint", "--save_interval", "1"
    ]


def test_production_a1_config_keeps_checkpointing_safe_defaults():
    from douzero.dmc.arguments import parse_args

    flags = parse_args(["--config", "configs/legacy_single_gpu_a1.yaml"])
    assert flags.legacy_actor_backend == "factorized"
    assert flags.actor_device_cpu is True
    assert flags.training_device == "0"
    assert flags.disable_checkpoint is False
    assert flags.checkpoint_sidecar_retention == 2
    assert flags.legacy_profile is False
    assert flags.legacy_metrics_path == ""
    assert bench_legacy_training.DEFAULT_CONFIGS[0] == (
        "legacy_a0_cpu_actor_thread1.yaml"
    )
    baseline = parse_args([
        "--config", "benchmarks/configs/legacy_a0_cpu_actor_thread1.yaml",
    ])
    assert baseline.legacy_actor_backend == "legacy"
    assert baseline.actor_torch_threads == 1


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
