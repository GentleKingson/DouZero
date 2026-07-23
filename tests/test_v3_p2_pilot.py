"""P2 pilot conversion, evidence, and fail-closed tests."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    P2_PILOT_PROTOCOL,
    P2_PILOT_SCHEMA,
    P2_VARIANTS,
    build_pilot_resolved_config,
    unique_legal_actions,
    train_pilot_batch,
    validate_pilot_summary,
    write_pilot_summary,
)


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
        },
        "evaluation": {"paired_deals": 0, "status": "not_executed"},
        "checkpoint": {"path": "latest.pt", "sha256": "4" * 64, "saved": True},
        "environment": {"image_digest": "sha256:" + "5" * 64},
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
