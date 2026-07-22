"""H8 formal evidence and promotion-gate contract tests."""

from __future__ import annotations

import copy

import pytest

from douzero.v3_hybrid.formal_evidence import (
    H8_EVIDENCE_SCHEMA_VERSION,
    H8EvidenceError,
    REQUIRED_VARIANTS,
    RULESETS,
    validate_h8_formal_evidence,
)

SHA = "a" * 64
GIT_SHA = "b" * 40
IMAGE = "sha256:" + "c" * 64
SEEDS = [11, 22, 33]


def _delta(value: float = 0.02) -> dict[str, float]:
    return {"estimate": value, "low": value / 2.0, "high": value * 1.5}


def _identity() -> dict[str, object]:
    return {
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "driver": "590.48",
        "cuda": "13.0",
        "pytorch": "2.9.0",
        "gpu": "RTX 5070",
        "cpu": "test cpu",
        "resolved_config_hash": SHA,
        "training_semantics_hash": SHA,
        "workload_hash": SHA,
        "feature_schema_hash": SHA,
        "model_identity_hash": SHA,
        "ruleset_identity_hash": SHA,
        "trainer_topology_hash": SHA,
        "replay_protocol_hash": SHA,
        "initial_checkpoint_sha256": SHA,
        "evaluation_deal_sets": {"legacy": "d" * 64, "standard": "e" * 64},
        "training_seeds": SEEDS,
        "evaluation_seed": 44,
        "wall_clock_budget_seconds": 7200,
        "sample_budget": 1_000_000,
        "search_config_hash": SHA,
        "belief_config_hash": SHA,
        "oracle_config_hash": SHA,
        "cooperation_config_hash": SHA,
        "loss_schedule_hash": SHA,
        "checkpoint_cadence_updates": 100,
        "authorized_bc_data": False,
    }


def _training(variant: str, seed: int) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "config_hash": SHA,
        "hardware_hash": SHA,
        "checkpoint_enabled": True,
        "samples": 1_000_000,
        "wall_clock_seconds": 7200,
        "cumulative_training_seconds": 7200,
        "sigterm_observed": True,
        "fresh_container_resume": True,
        "counter_before_resume": 10,
        "counter_after_resume": 11,
        "model_hash_before_resume": "1" * 64,
        "model_hash_after_resume": "2" * 64,
        "optimizer_state_continuous": True,
        "schedule_continuous": True,
        "policy_version_continuous": True,
        "rng_state_restored": True,
        "loss_finite": True,
        "gradient_finite": True,
        "clean_shutdown": True,
        "policy_lag_max": 1,
        "policy_lag_limit": 4,
        "memory_plateau": True,
        "active": 0,
        "in_flight": 0,
        "pending": 0,
        "temporary_checkpoints": 0,
        "artifact_stale": False,
    }


def _evaluation(variant: str, ruleset: str, seed: int) -> dict[str, object]:
    return {
        "variant": variant,
        "ruleset": ruleset,
        "search_enabled": variant == "v3_full_hybrid_search_on",
        "training_seed": seed,
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "model_checkpoint_sha256": SHA,
        "model_package_sha256": SHA,
        "deal_set_id": "d" * 64 if ruleset == "legacy" else "e" * 64,
        "deals": 100_000,
        "games": 200_000,
        "bootstrap_unit": "deal",
        "confidence_level": 0.95,
        "overall": {"wp_delta": _delta(), "adp_delta": _delta(0.2)},
        "by_role": {
            role: {"wp_delta": _delta(), "adp_delta": _delta(0.2)}
            for role in ("landlord", "landlord_up", "landlord_down")
        },
        "calibration_error": 0.02,
        "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "budget_p99": 5.0},
        "spring_count": 1,
        "anti_spring_count": 1,
        "bomb_count": 1,
        "rocket_count": 1,
        "bidding_games": 200_000 if ruleset == "standard" else 0,
        "search_triggers": 10 if variant == "v3_full_hybrid_search_on" else 0,
        "search_fallbacks": 0,
        "invalid_actions": 0,
        "timeouts": 0,
        "model_load_failures": 0,
        "public_package_validated": True,
        "privileged_leakage": False,
        "artifact_stale": False,
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": H8_EVIDENCE_SCHEMA_VERSION,
        "experiment_identity": _identity(),
        "training_runs": [
            _training(variant, seed)
            for variant in REQUIRED_VARIANTS
            for seed in SEEDS
        ],
        "evaluations": [
            _evaluation(variant, ruleset, seed)
            for variant in REQUIRED_VARIANTS
            for ruleset in RULESETS
            for seed in SEEDS
        ],
    }


def test_complete_evidence_is_recomputed_as_ready() -> None:
    report = validate_h8_formal_evidence(_evidence())
    assert report["release_status"] == "READY"
    assert report["release_candidate"] == "v3_full_hybrid_search_on"
    assert report["issues"] == []
    assert report["training_run_count"] == len(REQUIRED_VARIANTS) * 3
    assert report["evaluation_count"] == len(REQUIRED_VARIANTS) * 2 * 3


def test_missing_and_failed_gates_remain_valid_but_not_ready() -> None:
    payload = _evidence()
    payload["training_runs"].pop()
    payload["evaluations"].pop()
    payload["training_runs"][0]["cumulative_training_seconds"] = 7199
    candidate = next(
        row for row in payload["evaluations"]
        if row["variant"] == "v3_full_hybrid_search_on"
    )
    candidate["overall"]["wp_delta"] = {
        "estimate": 0.0, "low": -0.1, "high": 0.1
    }
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "NOT READY"
    assert report["release_candidate"] == "NONE"
    assert any("missing training runs" in issue for issue in report["issues"])
    assert any("under two hours" in issue for issue in report["issues"])
    assert any("promotion CI failed" in issue for issue in report["issues"])


def test_control_arms_are_reported_without_becoming_promotion_candidates() -> None:
    payload = _evidence()
    control = payload["evaluations"][0]
    control["overall"]["wp_delta"] = {
        "estimate": -0.2, "low": -0.3, "high": -0.1
    }
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "READY"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["training_runs"].append(copy.deepcopy(value["training_runs"][0])), "duplicate training"),
        (lambda value: value["evaluations"][0].update(games=199_999), "games/deals"),
        (lambda value: value["evaluations"][0].update(deal_set_id="f" * 64), "deal-set identity drift"),
        (lambda value: value["evaluations"][0]["latency_ms"].update(p50=4.0), "percentiles are unordered"),
        (lambda value: value["experiment_identity"].update(surprise=True), "fields mismatch"),
    ],
)
def test_contradictory_or_unknown_evidence_fails_closed(mutation, message) -> None:
    payload = _evidence()
    mutation(payload)
    with pytest.raises(H8EvidenceError, match=message):
        validate_h8_formal_evidence(payload)


def test_authorized_bc_requires_the_bc_variant() -> None:
    payload = _evidence()
    payload["experiment_identity"]["authorized_bc_data"] = True
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "NOT READY"
    assert any("v3_full_hybrid_bc" in issue for issue in report["issues"])


def test_stale_artifacts_can_never_be_release_ready() -> None:
    payload = _evidence()
    payload["training_runs"][0]["artifact_stale"] = True
    payload["evaluations"][0]["artifact_stale"] = True
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "NOT READY"
    assert sum("artifact is stale" in issue for issue in report["issues"]) == 2
