"""P2 pilot conversion, evidence, and fail-closed tests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_v3_pilot import (
    _attest_current_container,
    _attest_clean_source,
    _episode_fits_budget,
    _before_deadline,
    _fits_sample_budget,
    _load_run_state,
    _resolve_bounded_limit,
    _save_checkpoint_and_run_state,
    _train_episode_before_deadline,
)
from tools.summarize_v3_pilots import summarize_evidence

from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    P2_PILOT_PROTOCOL,
    P2_PILOT_SCHEMA,
    P2_SEED_DERIVATION,
    P2_VARIANTS,
    _should_collect_strategy_targets,
    build_pilot_resolved_config,
    derive_pilot_stream_seed,
    unique_legal_actions,
    train_pilot_batch,
    validate_pilot_summary,
    write_pilot_summary,
)


def test_pilot_scripts_import_attested_checkout_before_pythonpath(tmp_path):
    shadow = tmp_path / "shadow"
    package = shadow / "douzero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('shadow package imported')\n", encoding="utf-8"
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shadow), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(shadow)]
    )

    for script in ("run_v3_pilot.py", "summarize_v3_pilots.py"):
        result = subprocess.run(
            [sys.executable, str(root / "tools" / script), "--help"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert "shadow package imported" not in result.stderr


def test_episode_training_rejects_update_finishing_at_deadline():
    trained = []
    times = iter((14.0, 15.0))

    class _Metrics:
        def as_dict(self):
            return {"finite": True}

    with pytest.raises(RuntimeError, match="deadline elapsed"):
        _train_episode_before_deadline(
            object(),
            ("first", "second"),
            started=10.0,
            max_seconds=5.0,
            clock=lambda: next(times),
            train_fn=lambda _learner, piece: trained.append(piece) or _Metrics(),
        )

    assert trained == ["first"]


def test_strategy_labels_are_collected_only_during_public_training():
    def learner(public_training):
        return SimpleNamespace(
            config=SimpleNamespace(
                learner=SimpleNamespace(
                    features=SimpleNamespace(strategy=True)
                )
            ),
            base=SimpleNamespace(
                base=SimpleNamespace(
                    base=SimpleNamespace(
                        schedule_state=lambda: SimpleNamespace(
                            public_training=public_training
                        )
                    )
                )
            ),
        )

    assert not _should_collect_strategy_targets(learner(False))
    assert _should_collect_strategy_targets(learner(True))


def _summary():
    return {
        "schema": P2_PILOT_SCHEMA,
        "protocol": P2_PILOT_PROTOCOL,
        "source_git_sha": "1" * 40,
        "formal_config_sha256": "2" * 64,
        "training_semantics_hash": "3" * 64,
        "variant": "v3_role",
        "ruleset": "legacy",
        "seed": 101,
        "limits": {
            "max_seconds": 900.0,
            "max_samples": 1_000_000,
            "max_optimizer_steps": 10_000,
            "checkpoint_every": 100,
        },
        "collection": {
            "root_seed": 101,
            "worker_id": 0,
            "derivation": P2_SEED_DERIVATION,
            "epsilon": 0.01,
        },
        "status": "completed",
        "started_at": 1.0,
        "finished_at": 2.0,
        "wall_clock_seconds": 1.0,
        "samples": 4,
        "optimizer_steps": 1,
        "episodes": 1,
        "decisions": 4,
        "metrics": {
            "loss": 1.0,
            "samples_per_second": 4.0,
            "optimizer_steps_per_second": 1.0,
        },
        "resume": {
            "requested": False,
            "continued_update": False,
            "from_samples": 0,
            "from_optimizer_steps": 0,
            "from_episodes": 0,
            "from_decisions": 0,
            "checkpoint_sha256": None,
            "stop_signal": None,
        },
        "evaluation": {"paired_deals": 0, "status": "not_executed"},
        "checkpoint": {"path": "latest.pt", "sha256": "4" * 64, "saved": True},
        "environment": {
            "image_digest": "sha256:" + "5" * 64,
            "container_id": "container-a",
            "source_tree": "a" * 40,
            "cuda_available": True,
            "gpu": "GPU",
            "driver_version": "1.2.3",
            "torch_version": "2.0",
            "cuda_runtime": "13.2",
            "machine": "x86_64",
        },
        "release_candidate": "NONE",
        "release_status": "NOT READY",
        "playing_strength": "NOT MEASURED",
        "failure": None,
    }


@pytest.mark.parametrize("variant", P2_VARIANTS)
def test_frozen_legacy_variants_convert_to_executable_h6_without_side_effects(variant):
    formal = load_formal_config(f"configs/v3_formal/{variant}_legacy.yaml")
    resolved = build_pilot_resolved_config(formal)
    assert resolved.model.stable_hash() == formal.identity_dict()["model_hash"]
    assert resolved.learner.topology.topology == "single_process"
    assert resolved.learner.topology.ruleset == "legacy"
    for name in ("adaptive_dmc", "oracle", "belief", "cooperation", "strategy", "style"):
        assert getattr(resolved.learner.features, name) is formal.features[name]


def test_pilot_conversion_rejects_standard_and_non_v3_controls():
    with pytest.raises(ValueError, match="legacy card-play"):
        build_pilot_resolved_config(
            load_formal_config("configs/v3_formal/v3_role_standard.yaml")
        )
    with pytest.raises(ValueError, match="six frozen V3"):
        build_pilot_resolved_config(
            load_formal_config("configs/v3_formal/model_v2_legacy.yaml")
        )


def test_pilot_removes_only_exact_duplicate_legal_rows_in_engine_order():
    assert unique_legal_actions(([3], [4, 4], [3], [], [])) == [
        [3], [4, 4], [],
    ]
    with pytest.raises(ValueError, match="no legal action"):
        unique_legal_actions(())


def test_oracle_warmup_does_not_feed_public_strategy_targets():
    captured = {}

    class Learner:
        base = SimpleNamespace(
            base=SimpleNamespace(
                base=SimpleNamespace(
                    schedule_state=lambda: SimpleNamespace(
                        public_training=False,
                        privileged_required=True,
                        oracle_weight=1.0,
                        guidance_weight=0.0,
                    )
                )
            )
        )

        def train_batch(self, transitions, **sidecars):
            captured.update(sidecars)
            return "metrics"

    batch = SimpleNamespace(
        transitions=("row",), trajectories=None, belief_samples=None,
        oracle_samples=("oracle",), strategy_targets=({"label": 1.0},),
    )
    assert train_pilot_batch(Learner(), batch) == "metrics"
    assert captured["oracle_samples"] == ("oracle",)
    assert captured["strategy_targets"] is None


def test_pilot_summary_is_canonical_and_cannot_claim_strength(tmp_path):
    payload = _summary()
    path = tmp_path / "summary.json"
    write_pilot_summary(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    ready = copy.deepcopy(payload)
    ready["release_status"] = "READY"
    with pytest.raises(ValueError, match="cannot declare"):
        validate_pilot_summary(ready)
    measured = copy.deepcopy(payload)
    measured["playing_strength"] = "MEASURED"
    with pytest.raises(ValueError, match="cannot declare"):
        validate_pilot_summary(measured)


def test_pilot_summary_rejects_stale_or_non_commit_source_identity():
    payload = _summary()
    payload["source_git_sha"] = "unknown"
    with pytest.raises(ValueError, match="full Git SHA"):
        validate_pilot_summary(payload)
    payload = _summary()
    payload["checkpoint"]["sha256"] = "short"
    with pytest.raises(ValueError, match="requires SHA-256"):
        validate_pilot_summary(payload)


def test_pilot_summary_recomputes_resume_delta_throughput():
    payload = _summary()
    payload["samples"] = 10
    payload["optimizer_steps"] = 4
    payload["resume"]["from_samples"] = 6
    payload["resume"]["from_optimizer_steps"] = 3
    validate_pilot_summary(payload)
    payload["metrics"]["samples_per_second"] = 10.0
    with pytest.raises(ValueError, match="inconsistent with raw counters"):
        validate_pilot_summary(payload)


def test_run_state_binds_checkpoint_seed_and_episode_continuation(tmp_path):
    class Learner:
        def save_checkpoint(self, path):
            path.write_bytes(b"checkpoint")

    checkpoint = tmp_path / "latest.pt"
    state_path = tmp_path / "pilot-state.json"
    digest = _save_checkpoint_and_run_state(
        Learner(), checkpoint, state_path,
        source_sha="1" * 40,
        formal_config_sha256="2" * 64,
        variant="v3_role",
        root_seed=101,
        episodes_completed=7,
        decisions_completed=99,
    )
    state = _load_run_state(
        state_path,
        checkpoint_sha256=digest,
        source_sha="1" * 40,
        formal_config_sha256="2" * 64,
        variant="v3_role",
        root_seed=101,
    )
    assert state["episodes_completed"] == 7
    assert derive_pilot_stream_seed(101, "environment", 0, 7) == (
        derive_pilot_stream_seed(101, "environment", 0, state["episodes_completed"])
    )
    with pytest.raises(SystemExit, match="root_seed mismatch"):
        _load_run_state(
            state_path,
            checkpoint_sha256=digest,
            source_sha="1" * 40,
            formal_config_sha256="2" * 64,
            variant="v3_role",
            root_seed=999,
        )


def test_frozen_seed_derivation_separates_streams_and_episodes():
    seed = derive_pilot_stream_seed(101, "environment", 0, 7)
    assert seed == derive_pilot_stream_seed(101, "environment", 0, 7)
    assert seed != derive_pilot_stream_seed(101, "exploration", 0, 7)
    assert seed != derive_pilot_stream_seed(101, "environment", 0, 8)


def test_zero_budget_override_is_rejected():
    assert _resolve_bounded_limit(None, 10, "budget") == 10
    with pytest.raises(SystemExit, match="budget must be positive"):
        _resolve_bounded_limit(0, 10, "budget")
    with pytest.raises(SystemExit, match="exceeds the frozen pilot ceiling"):
        _resolve_bounded_limit(11, 10, "budget")


def test_sample_budget_stops_when_next_slice_does_not_fit():
    assert _fits_sample_budget(32, 32, 64)
    assert not _fits_sample_budget(0, 32, 1)
    assert not _fits_sample_budget(63, 2, 64)
    learner = SimpleNamespace(samples_consumed=32, eligible_updates=4)
    pieces = (
        SimpleNamespace(transitions=tuple(range(32))),
        SimpleNamespace(transitions=tuple(range(2))),
    )
    assert _episode_fits_budget(learner, pieces, 66, 6)
    assert not _episode_fits_budget(learner, pieces, 65, 6)
    assert not _episode_fits_budget(learner, pieces, 66, 5)
    assert _before_deadline(10.0, 5.0, 14.999)
    assert not _before_deadline(10.0, 5.0, 15.0)


def test_docker_image_identity_is_bound_to_current_pid_namespace(monkeypatch):
    image_id = "sha256:" + "a" * 64
    current = "a" * 64
    other = "b" * 64

    def fake_get(path, socket_path):
        assert socket_path == "/docker.sock"
        if path == "/containers/json?all=0":
            return [{"Id": current}, {"Id": other}]
        if path == f"/containers/{current}/json":
            return {"State": {"Pid": 123}, "Image": image_id}
        return {"State": {"Pid": 456}, "Image": "sha256:" + "b" * 64}

    monkeypatch.setattr(
        "tools.run_v3_pilot._pid_namespace",
        lambda path: "pid:[123]" if path in {
            "/proc/1/ns/pid", "/host/proc/123/ns/pid"
        } else "pid:[456]",
    )
    monkeypatch.setattr("tools.run_v3_pilot._docker_api_get", fake_get)
    assert _attest_current_container("/docker.sock", "/host/proc") == (
        current, image_id
    )


def test_evidence_summary_rejects_unrelated_resume_pair(tmp_path):
    for index, variant in enumerate(P2_VARIANTS):
        directory = tmp_path / variant
        directory.mkdir()
        before = _summary()
        formal = load_formal_config(f"configs/v3_formal/{variant}_legacy.yaml")
        identity = formal.identity_dict()
        before.update({"variant": variant, "status": "stopped", "samples": 4,
                       "optimizer_steps": 1, "episodes": 2, "decisions": 4,
                       "formal_config_sha256": identity["config_sha256"],
                       "training_semantics_hash": identity["training_semantics_hash"],
                       "seed": formal.seeds.training[0], "ruleset": formal.ruleset["id"]})
        before["resume"]["stop_signal"] = "SIGTERM"
        before["environment"]["container_id"] = f"before-{index}"
        after = copy.deepcopy(before)
        after.update({"status": "completed", "samples": 8, "optimizer_steps": 2,
                      "episodes": 1, "decisions": 2})
        after["resume"].update({
            "requested": True,
            "from_samples": 4,
            "from_optimizer_steps": 1,
            "from_episodes": 2,
            "from_decisions": 4,
            "checkpoint_sha256": before["checkpoint"]["sha256"],
            "continued_update": True,
            "stop_signal": None,
        })
        model_hash = identity["model_hash"]
        resolved_hash = build_pilot_resolved_config(formal).stable_hash()
        after["metrics"].update({"samples_per_second": 4.0,
                                  "optimizer_steps_per_second": 1.0,
                                  "model_hash": model_hash,
                                  "resolved_config_hash": resolved_hash,
                                  "skipped_long_cooperation_episodes": 0})
        before["metrics"].update({"model_hash": model_hash,
                                   "resolved_config_hash": resolved_hash,
                                   "skipped_long_cooperation_episodes": 0})
        after["environment"]["container_id"] = f"after-{index}"
        for name, payload in (("pre-resume-summary.json", before),
                              ("post-resume-summary.json", after)):
            (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    summarize_evidence(tmp_path)
    target = tmp_path / P2_VARIANTS[0] / "post-resume-summary.json"
    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["resume"]["checkpoint_sha256"] = "8" * 64
    target.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        summarize_evidence(tmp_path)
    changed["resume"]["checkpoint_sha256"] = "4" * 64
    changed["samples"] = 3
    changed["metrics"]["samples_per_second"] = -1.0
    target.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="sample counter did not increase"):
        summarize_evidence(tmp_path)


def test_evidence_summary_rejects_failed_resume_and_environment_drift(tmp_path):
    test_evidence_summary_rejects_unrelated_resume_pair(tmp_path)
    target = tmp_path / P2_VARIANTS[0] / "post-resume-summary.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["resume"]["checkpoint_sha256"] = "4" * 64
    payload["status"] = "failed"
    payload["failure"] = {"type": "RuntimeError", "message": "boom"}
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not complete successfully"):
        summarize_evidence(tmp_path)
    payload["status"] = "completed"
    payload["failure"] = None
    payload["environment"]["gpu"] = "different GPU"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accelerator identity"):
        summarize_evidence(tmp_path)
    payload["environment"]["gpu"] = "GPU"
    payload["limits"]["max_samples"] = 1_000_001
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds the frozen pilot ceiling"):
        summarize_evidence(tmp_path)


def test_clean_source_attestation_rejects_dirty_or_wrong_head(monkeypatch):
    results = iter([
        SimpleNamespace(stdout="1" * 40 + "\n"),
        SimpleNamespace(stdout=" M tools/run_v3_pilot.py\n"),
        SimpleNamespace(stdout="a" * 40 + "\n"),
    ])
    monkeypatch.setattr("tools.run_v3_pilot.subprocess.run", lambda *a, **k: next(results))
    with pytest.raises(SystemExit, match="clean runtime source tree"):
        _attest_clean_source("1" * 40)
