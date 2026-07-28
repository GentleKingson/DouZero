from __future__ import annotations

import copy

import pytest

from douzero.v3_hybrid.runtime_decision import (
    P3_RUNTIME_SCHEMA,
    P3_SEGMENTS,
    P3_TOPOLOGIES,
    P3RuntimeProtocol,
    summarize_p3_decision,
    validate_p3_records,
)


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
        "measurement_seconds": elapsed,
        "counters_before": before,
        "counters_after": after,
        "rates": {
            f"{name}_per_second": (after[name] - before[name]) / elapsed
            for name in before
        },
        "segments_seconds": {name: 0.0 for name in P3_SEGMENTS},
        "parameter_update_observed": True,
        "checkpoint": {
            "path": f"/evidence/{topology}-{repeat}.pt",
            "sha256": _sha("5"),
            "saved": True,
            "strict_reload": True,
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
    with pytest.raises(ValueError, match="strict reload"):
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
