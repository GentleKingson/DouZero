from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from benchmarks.run_v3_p3_runtime import (
    CheckpointCadence,
    SegmentProfiler,
    _aggregate_runtime_rss_bytes,
    _episodes_per_cycle,
    _learner_updates_per_cycle,
    _learner_digest,
    _module_digest,
    _prime_full_hybrid_phase,
    _run_full,
    _run_full_until,
    _run_runtime_until,
    _strict_runtime_reload,
    _verify_hardware_identity,
)
from douzero.training.seed_stream import (
    FORMAL_SEED_DERIVATION_V1,
    derive_formal_stream_seed,
)
from douzero.v3_hybrid.pilot import (
    _step_forced_action_without_replay,
    derive_pilot_stream_seed,
)
from douzero.v3_hybrid.runtime import V3H7RuntimeConfig
from douzero.v3_hybrid.runtime_decision import (
    P3_RUNTIME_SCHEMA,
    P3_SEGMENTS,
    P3_TOPOLOGIES,
    P3RuntimeProtocol,
    summarize_p3_decision,
    validate_p3_records,
)
from tools.summarize_v3_p3_runtime import write_repository_checksums


def _sha(character: str) -> str:
    return character * 64


def _protocol() -> P3RuntimeProtocol:
    return P3RuntimeProtocol(
        source_git_sha="a" * 40,
        source_tree="b" * 40,
        image_digest=f"sha256:{_sha('c')}",
        base_config_hash=_sha("d"),
        full_config_hash=_sha("e"),
        shared_scale_hash=_sha("f"),
        model_identity_hashes={"base": _sha("1"), "full_hybrid": _sha("2")},
        trainer_identity_hash=_sha("3"),
        replay_protocol_hash=_sha("4"),
        gpu="RTX 5070",
        driver="595",
        pytorch="2.12",
        cuda="13.2",
        cpu="x86_64",
        warmup_seconds=1.0,
        measurement_seconds=10.0,
    )


def _record(
    protocol: P3RuntimeProtocol, topology: str, repeat: int
) -> dict[str, object]:
    full = topology == "full_hybrid_single_process"
    before = {
        "games": 0,
        "decisions": 0,
        "transitions": 0,
        "learner_samples": 0,
        "optimizer_steps": 0,
    }
    samples = 50 if full else 100
    after = {
        "games": 10,
        "decisions": 200,
        "transitions": samples,
        "learner_samples": samples,
        "optimizer_steps": 4,
    }
    elapsed = 10.0
    return {
        "schema": P3_RUNTIME_SCHEMA,
        "protocol_hash": protocol.stable_hash(),
        "topology": topology,
        "repeat": repeat,
        "seed": protocol.seeds[repeat],
        "source_git_sha": protocol.source_git_sha,
        "source_tree": protocol.source_tree,
        "image_digest": protocol.image_digest,
        "config_hash": (
            protocol.full_config_hash if full else protocol.base_config_hash
        ),
        "model_identity_hash": protocol.model_identity_hashes[
            "full_hybrid" if full else "base"
        ],
        "deal_seed_derivation": protocol.deal_seed_derivation,
        "measurement_seconds": elapsed,
        "counters_before": before,
        "counters_after": after,
        "rates": {
            f"{name}_per_second": (after[name] - before[name]) / elapsed
            for name in before
        },
        "segments_seconds": {name: 0.0 for name in P3_SEGMENTS},
        "parameter_update_observed": True,
        "training_phase": {
            "before": protocol.full_hybrid_phase if full else "disabled",
            "after": protocol.full_hybrid_phase if full else "disabled",
            "learner_update_before": (
                protocol.full_hybrid_phase_update if full else 10
            ),
            "learner_update_after": (
                protocol.full_hybrid_phase_update + 4 if full else 14
            ),
        },
        "checkpoint": {
            "path": f"/evidence/{topology}-{repeat}.pt",
            "sha256": _sha("5"),
            "saved": True,
            "strict_reload": True,
            "resumed_update": True,
            "resume_quiesced": True,
        },
        "policy_lag_max": 0,
        "actor_blocked_ratio": 0.0,
        "learner_data_wait_ratio": 0.0,
        "cpu_ram_bytes": 1,
        "shared_memory_bytes": 0,
        "vram_bytes": 1,
        "active_slots": 0,
        "in_flight": 0,
        "pending": 0,
        "shutdown_seconds": 0.1,
        "skipped_long_cooperation_episodes": 0,
        "skipped_incomplete_cooperation_episodes": 0,
    }


def _records(protocol: P3RuntimeProtocol) -> list[dict[str, object]]:
    return [
        _record(protocol, topology, repeat)
        for topology in P3_TOPOLOGIES
        for repeat in range(protocol.repetitions)
    ]


def test_p3_requires_complete_matched_matrix_and_raw_rate_consistency() -> None:
    protocol = _protocol()
    records = _records(protocol)
    validate_p3_records(records, protocol)
    with pytest.raises(ValueError, match="every matched repetition"):
        validate_p3_records(records[:-1], protocol)
    corrupt = copy.deepcopy(records)
    corrupt[0]["rates"]["learner_samples_per_second"] += 0.01
    with pytest.raises(ValueError, match="rate learner_samples is inconsistent"):
        validate_p3_records(corrupt, protocol)


def test_p3_binds_config_model_source_image_and_checkpoint() -> None:
    protocol = _protocol()
    for field, value, message in (
        ("source_tree", "0" * 40, "source_tree drift"),
        ("image_digest", f"sha256:{_sha('9')}", "image_digest drift"),
        ("config_hash", _sha("9"), "config hash drift"),
        ("model_identity_hash", _sha("9"), "model identity drift"),
    ):
        records = _records(protocol)
        records[0][field] = value
        with pytest.raises(ValueError, match=message):
            validate_p3_records(records, protocol)
    records = _records(protocol)
    records[0]["checkpoint"]["strict_reload"] = False
    with pytest.raises(ValueError, match="save/reload/resumed update"):
        validate_p3_records(records, protocol)


def test_p3_rejects_nonfinite_segments_progress_regression_and_live_slots() -> None:
    protocol = _protocol()
    records = _records(protocol)
    records[0]["segments_seconds"]["exact_dp"] = float("nan")
    with pytest.raises(ValueError, match="segment exact_dp"):
        validate_p3_records(records, protocol)
    records = _records(protocol)
    records[0]["counters_after"]["learner_samples"] = -1
    with pytest.raises(ValueError, match="after counter learner_samples"):
        validate_p3_records(records, protocol)
    records = _records(protocol)
    records[0]["pending"] = 1
    with pytest.raises(ValueError, match="quiesce pending"):
        validate_p3_records(records, protocol)
    records = _records(protocol)
    records[0]["skipped_incomplete_cooperation_episodes"] = -1
    with pytest.raises(
        ValueError,
        match="skipped_incomplete_cooperation_episodes counter",
    ):
        validate_p3_records(records, protocol)


def test_p3_requires_guided_phase_and_exercised_resume() -> None:
    protocol = _protocol()
    records = _records(protocol)
    records[-1]["training_phase"]["before"] = "oracle_warmup"
    with pytest.raises(ValueError, match="training phase drift"):
        validate_p3_records(records, protocol)

    records = _records(protocol)
    records[0]["checkpoint"]["resumed_update"] = False
    with pytest.raises(ValueError, match="resumed update"):
        validate_p3_records(records, protocol)

    records = _records(protocol)
    records[0]["deal_seed_derivation"] = "linear"
    with pytest.raises(ValueError, match="seed derivation drift"):
        validate_p3_records(records, protocol)


def test_p3_decision_uses_median_full_to_base_ratio() -> None:
    protocol = _protocol()
    summary = summarize_p3_decision(_records(protocol), protocol)
    assert summary["full_hybrid_to_base_single_ratio"] == pytest.approx(0.5)
    assert summary["implement_h7_1"] is True
    assert summary["release_candidate"] == "NONE"
    assert summary["release_status"] == "NOT READY"
    assert summary["playing_strength"] == "NOT MEASURED"


def test_p3_protocol_freezes_threshold_before_measurement() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol.identity())
    changed["full_hybrid_min_base_ratio"] = 0.6
    changed.pop("schema")
    adjusted = P3RuntimeProtocol(**changed)
    assert adjusted.stable_hash() != protocol.stable_hash()
    with pytest.raises(ValueError, match="threshold"):
        P3RuntimeProtocol(**{
            **changed,
            "full_hybrid_min_base_ratio": 0.0,
        })
    cadence = P3RuntimeProtocol(**{
        **changed,
        "episodes_per_learner_update": 8,
    })
    assert cadence.stable_hash() != protocol.stable_hash()
    phase = P3RuntimeProtocol(**{
        **changed,
        "full_hybrid_phase_update": 20000,
    })
    assert phase.stable_hash() != protocol.stable_hash()


def test_p3_full_learner_digest_tracks_training_only_modules() -> None:
    public = torch.nn.Linear(2, 2)
    oracle = torch.nn.Linear(2, 2)
    belief = torch.nn.Linear(2, 2)
    cooperation = torch.nn.Linear(2, 2)
    h3 = SimpleNamespace(oracle=oracle)
    h4 = SimpleNamespace(base=h3, belief_model=belief)
    h5 = SimpleNamespace(base=h4, cooperation=cooperation)
    learner = SimpleNamespace(base=h5, model=public)
    public_before = _module_digest({"public": public})
    complete_before = _learner_digest(learner)

    with torch.no_grad():
        oracle.weight.add_(1.0)

    assert _module_digest({"public": public}) == public_before
    assert _learner_digest(learner) != complete_before


def test_p3_checkpoint_cadence_saves_every_crossed_boundary() -> None:
    saves = []
    cadence = CheckpointCadence(5, 3, lambda: saves.append(len(saves)))
    cadence.observe(4)
    assert saves == []
    cadence.observe(5)
    assert len(saves) == 1
    cadence.observe(16)
    assert len(saves) == 3
    with pytest.raises(ValueError, match="regressed"):
        cadence.observe(15)


def test_p3_runtime_cycles_keep_one_update_per_four_games() -> None:
    assert _learner_updates_per_cycle(4, 4) == 1
    assert _learner_updates_per_cycle(16, 4) == 4
    assert _learner_updates_per_cycle(32, 4) == 8
    with pytest.raises(ValueError, match="divisible"):
        _learner_updates_per_cycle(15, 4)

    class FakeTrainer:
        policy_step = 0
        _snapshot_step = 0

        def __init__(self) -> None:
            self.stats = SimpleNamespace(optimizer_steps=0)
            self.collections: list[int] = []
            self.optimizations: list[int] = []

        def collect_episodes(self, episodes: int) -> None:
            self.collections.append(episodes)

        def optimize(self, updates: int) -> None:
            self.optimizations.append(updates)
            self.stats.optimizer_steps += updates

    trainer = FakeTrainer()
    with patch(
        "benchmarks.run_v3_p3_runtime.time.monotonic",
        side_effect=(0.0, 2.0),
    ):
        _run_runtime_until(
            trainer,
            deadline=1.0,
            episodes=16,
            episodes_per_learner_update=4,
            checkpoint_cadence=CheckpointCadence(100, 0, lambda: None),
        )
    assert trainer.collections == [16]
    assert trainer.optimizations == [4]


def test_p3_runtime_reload_exercises_collection_update_and_quiescence() -> None:
    checkpoint_state = {
        "p3_protocol_hash": _sha("a"),
        "topology": "base_async_4x4",
        "seed": 101,
    }

    class FakeRestored:
        instances = []

        def __init__(self, learner, _resolved, _runtime_config) -> None:
            self.learner = learner
            self.stats = SimpleNamespace(optimizer_steps=7)
            self.shutdown_called = False
            self.__class__.instances.append(self)

        def load_training_checkpoint(self, _checkpoint):
            return dict(checkpoint_state)

        def collect_episodes(self, episodes: int) -> None:
            assert episodes == 16

        def optimize(self, updates: int) -> None:
            assert updates == 4
            self.stats.optimizer_steps += updates

        def quiesce_cycle_boundary(self):
            return {
                "active_slots": 0,
                "in_flight_slots": 0,
                "pending_requests": 0,
            }

        def shutdown(self) -> None:
            self.shutdown_called = True

    with (
        patch(
            "benchmarks.run_v3_p3_runtime.create_pilot_learner",
            return_value=(SimpleNamespace(), SimpleNamespace()),
        ),
        patch(
            "benchmarks.run_v3_p3_runtime._learner_digest",
            side_effect=("before", "after"),
        ),
    ):
        result = _strict_runtime_reload(
            FakeRestored,
            SimpleNamespace(),
            SimpleNamespace(),
            Path("checkpoint.pt"),
            episodes=16,
            episodes_per_learner_update=4,
            checkpoint_state=checkpoint_state,
        )

    assert result == {
        "strict_reload": True,
        "resumed_update": True,
        "resume_quiesced": True,
    }
    assert FakeRestored.instances[0].shutdown_called is True


def test_p3_async_cycle_queues_every_actor_game_slot() -> None:
    assert _episodes_per_cycle("base_single_process", 1, 1) == 4
    assert _episodes_per_cycle("base_async_4x4", 4, 4) == 16
    assert _episodes_per_cycle("base_async_8x4", 8, 4) == 32
    with pytest.raises(ValueError, match="topology"):
        _episodes_per_cycle("unknown", 1, 1)


def test_p3_matched_deal_seed_uses_one_formal_episode_stream() -> None:
    expected = derive_formal_stream_seed(101, "environment", 0, 7)
    assert derive_pilot_stream_seed(101, "environment", 0, 7) == expected
    config = V3H7RuntimeConfig(
        environment_seed=101,
        environment_seed_derivation=FORMAL_SEED_DERIVATION_V1,
    )
    assert config.environment_seed_derivation == FORMAL_SEED_DERIVATION_V1
    assert config.stable_hash() != V3H7RuntimeConfig(
        environment_seed=101
    ).stable_hash()
    with pytest.raises(ValueError, match="seed derivation"):
        V3H7RuntimeConfig(environment_seed_derivation="unknown")


def test_p3_live_hardware_must_match_every_frozen_axis() -> None:
    protocol = _protocol()
    live = {
        "gpu": protocol.gpu,
        "driver": protocol.driver,
        "pytorch": protocol.pytorch,
        "cuda": protocol.cuda,
        "cpu": protocol.cpu,
    }
    with patch(
        "benchmarks.run_v3_p3_runtime._live_hardware_identity",
        return_value=live,
    ):
        _verify_hardware_identity(protocol)

    for field in live:
        drifted = dict(live)
        drifted[field] = f"{drifted[field]}-different"
        with patch(
            "benchmarks.run_v3_p3_runtime._live_hardware_identity",
            return_value=drifted,
        ), pytest.raises(SystemExit, match=f"live {field}"):
            _verify_hardware_identity(protocol)


def test_p3_async_rss_includes_every_live_actor_process(tmp_path: Path) -> None:
    def write_status(pid: int, rss_kib: int) -> None:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "status").write_text(
            f"Name:\ttest\nVmRSS:\t{rss_kib} kB\n", encoding="utf-8"
        )

    write_status(101, 100)
    write_status(202, 20)
    write_status(303, 30)
    workers = [
        SimpleNamespace(pid=202, is_alive=lambda: True),
        SimpleNamespace(pid=303, is_alive=lambda: True),
    ]
    trainer = SimpleNamespace(_runtime_started=True, _workers=workers)

    assert _aggregate_runtime_rss_bytes(
        trainer, proc_root=tmp_path, parent_pid=101
    ) == 150 * 1024

    workers[1] = SimpleNamespace(pid=303, is_alive=lambda: False)
    with pytest.raises(RuntimeError, match="not live"):
        _aggregate_runtime_rss_bytes(
            trainer, proc_root=tmp_path, parent_pid=101
        )


def test_p3_segment_timers_synchronize_before_and_after_cuda_work() -> None:
    events: list[str] = []
    times = iter((1.0, 3.5, 4.0, 7.0))
    profiler = SegmentProfiler(
        synchronize=lambda: events.append("sync"),
        clock=lambda: next(times),
    )

    with profiler.measure("public_model_forward"):
        events.append("body")
    accumulator = [0.0]
    result = profiler.time_call(
        accumulator, lambda: events.append("call") or "result"
    )

    assert result == "result"
    assert events == ["sync", "body", "sync", "sync", "call", "sync"]
    assert profiler.values["public_model_forward"] == pytest.approx(2.5)
    assert accumulator[0] == pytest.approx(3.0)


def test_p3_full_hybrid_is_primed_to_guided_phase_consistently() -> None:
    class Schedule:
        warmup_updates = 10000

        @staticmethod
        def at(update: int):
            return SimpleNamespace(
                phase="guided" if update >= 10000 else "oracle_warmup",
                learner_update=update,
            )

    h3 = SimpleNamespace(
        learner_updates=0,
        samples_consumed=0,
        statistics=SimpleNamespace(steps=0),
        config=SimpleNamespace(schedule=Schedule()),
    )
    h3.schedule_state = lambda: h3.config.schedule.at(h3.learner_updates)
    h4 = SimpleNamespace(
        base=h3,
        eligible_updates=0,
        samples_consumed=0,
        statistics=SimpleNamespace(steps=0, base_updates=0),
    )
    h5 = SimpleNamespace(
        base=h4,
        eligible_updates=0,
        samples_consumed=0,
        statistics=SimpleNamespace(steps=0),
    )
    learner = SimpleNamespace(
        base=h5,
        eligible_updates=0,
        samples_consumed=0,
        statistics=SimpleNamespace(steps=0),
    )

    state = _prime_full_hybrid_phase(learner, 10000)

    assert state.phase == "guided"
    assert state.learner_update == 10000
    assert (
        learner.eligible_updates,
        h5.eligible_updates,
        h4.eligible_updates,
        h3.learner_updates,
    ) == (10000, 10000, 10000, 10000)
    assert (
        learner.statistics.steps,
        h5.statistics.steps,
        h4.statistics.steps,
        h4.statistics.base_updates,
        h3.statistics.steps,
    ) == (10000, 10000, 10000, 10000, 10000)


def test_p3_full_releases_measured_cuda_graph_before_reload() -> None:
    order: list[str] = []
    measured = {
        "_resume_episode_number": 9,
        "shutdown": 0.0,
    }
    with (
        patch(
            "benchmarks.run_v3_p3_runtime._measure_full",
            side_effect=lambda *_args: order.append("measure") or dict(measured),
        ),
        patch(
            "benchmarks.run_v3_p3_runtime._release_cuda_graph",
            side_effect=lambda: order.append("release"),
        ),
        patch(
            "benchmarks.run_v3_p3_runtime._strict_full_reload",
            side_effect=lambda *_args, **_kwargs: (
                order.append("reload")
                or {
                    "strict_reload": True,
                    "resumed_update": True,
                    "resume_quiesced": True,
                }
            ),
        ),
        patch(
            "benchmarks.run_v3_p3_runtime.time.monotonic",
            side_effect=(1.0, 2.0),
        ),
    ):
        result = _run_full(
            _protocol(),
            SimpleNamespace(),
            101,
            Path("checkpoint.pt"),
        )

    assert order == ["measure", "release", "reload", "release"]
    assert result["checkpoint_reload"] is True
    assert result["resumed_update"] is True
    assert result["resume_quiesced"] is True


def test_p3_full_collection_skips_forced_inference_and_replay() -> None:
    calls: list[list[int]] = []
    env = SimpleNamespace(
        step=lambda action: (
            calls.append(action) or (None, 0.0, False, {"step": 1})
        )
    )
    infoset = SimpleNamespace(
        player_position="landlord_down",
        legal_actions=[[7]],
    )
    trace: list[tuple[str, tuple[int, ...]]] = []

    result = _step_forced_action_without_replay(
        env, infoset, trace, enabled=True
    )

    assert result == (None, 0.0, False, {"step": 1})
    assert calls == [[7]]
    assert trace == [("landlord_down", (7,))]
    assert _step_forced_action_without_replay(
        env,
        SimpleNamespace(
            player_position="landlord_down",
            legal_actions=[[7], [8]],
        ),
        trace,
        enabled=True,
    ) is None
    assert _step_forced_action_without_replay(
        env, infoset, trace, enabled=False
    ) is None


def test_p3_full_runner_enables_matched_forced_action_semantics() -> None:
    learner = SimpleNamespace(
        config=SimpleNamespace(
            learner=SimpleNamespace(
                base=SimpleNamespace(
                    base=SimpleNamespace(
                        base=SimpleNamespace(
                            public=SimpleNamespace(batch_size=32)
                        )
                    )
                )
            )
        ),
        samples_consumed=0,
        eligible_updates=0,
    )
    batch = SimpleNamespace(
        decisions=2,
        transitions=("row",),
        trajectories=None,
        cooperation_skip_reason=None,
    )
    collector = Mock(return_value=batch)

    def train(current, _batch) -> None:
        current.samples_consumed += 1
        current.eligible_updates += 1

    state = {
        "games": 0,
        "decisions": 0,
        "transitions": 0,
        "samples": 0,
        "steps": 0,
        "skipped": 0,
        "skipped_incomplete": 0,
    }
    cadence = CheckpointCadence(10, 0, lambda: None)
    with (
        patch(
            "benchmarks.run_v3_p3_runtime.collect_real_pilot_episode",
            collector,
        ),
        patch(
            "benchmarks.run_v3_p3_runtime.train_pilot_batch",
            side_effect=train,
        ),
        patch(
            "benchmarks.run_v3_p3_runtime.time.monotonic",
            side_effect=(0.0, 2.0),
        ),
    ):
        _run_full_until(
            learner,
            seed=101,
            deadline=1.0,
            state=state,
            profiler=SegmentProfiler(),
            checkpoint_cadence=cadence,
        )

    assert collector.call_args.kwargs["skip_forced_actions"] is True
    assert state == {
        "games": 1,
        "decisions": 2,
        "transitions": 1,
        "samples": 1,
        "steps": 1,
        "skipped": 0,
        "skipped_incomplete": 0,
    }


def test_p3_full_runner_skips_incomplete_nonforced_farmer_pair() -> None:
    learner = SimpleNamespace(
        config=SimpleNamespace(
            learner=SimpleNamespace(
                base=SimpleNamespace(
                    base=SimpleNamespace(
                        base=SimpleNamespace(
                            public=SimpleNamespace(batch_size=32)
                        )
                    )
                )
            )
        ),
        samples_consumed=0,
        eligible_updates=0,
    )
    batch = SimpleNamespace(
        decisions=3,
        transitions=("landlord", "farmer"),
        trajectories=None,
        cooperation_skip_reason="missing_nonforced_farmer_role",
    )
    collector = Mock(return_value=batch)
    trainer = Mock()
    state = {
        "games": 0,
        "decisions": 0,
        "transitions": 0,
        "samples": 0,
        "steps": 0,
        "skipped": 0,
        "skipped_incomplete": 0,
    }
    with (
        patch(
            "benchmarks.run_v3_p3_runtime.collect_real_pilot_episode",
            collector,
        ),
        patch(
            "benchmarks.run_v3_p3_runtime.train_pilot_batch",
            trainer,
        ),
        patch(
            "benchmarks.run_v3_p3_runtime.time.monotonic",
            side_effect=(0.0, 2.0),
        ),
    ):
        _run_full_until(
            learner,
            seed=101,
            deadline=1.0,
            state=state,
            profiler=SegmentProfiler(),
            checkpoint_cadence=CheckpointCadence(10, 0, lambda: None),
        )

    trainer.assert_not_called()
    assert state == {
        "games": 1,
        "decisions": 3,
        "transitions": 2,
        "samples": 0,
        "steps": 0,
        "skipped": 0,
        "skipped_incomplete": 1,
    }


def test_p3_evidence_checksums_resolve_from_repository_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    evidence = root / "artifacts" / "v3-p3"
    evidence.mkdir(parents=True)
    paths = tuple(evidence / name for name in (
        "protocol.json", "records.jsonl", "summary.json"
    ))
    for index, path in enumerate(paths):
        path.write_text(f"payload-{index}\n", encoding="utf-8")
    manifest = evidence / "SHA256SUMS"

    write_repository_checksums(
        manifest, paths, repository_root=root
    )

    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the repository"):
        write_repository_checksums(
            manifest, (outside,), repository_root=root
        )
