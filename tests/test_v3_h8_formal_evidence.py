"""H8a formal evidence, support-matrix, and promotion contract tests."""

from __future__ import annotations

import copy

import pytest

from douzero.v3_hybrid.formal_evidence import (
    DEVELOPMENT,
    H8A_SUPPORT_MATRIX_VERSION,
    H8A_VARIANT_RULESET_SUPPORT,
    H8_EVIDENCE_SCHEMA_VERSION,
    H8EvidenceError,
    PROMOTION,
    REQUIRED_VARIANTS,
    h8a_support_matrix_hash,
    validate_h8_formal_evidence,
)

SHA = "a" * 64
GIT_SHA = "b" * 40
IMAGE = "sha256:" + "c" * 64
SEEDS = [11, 22, 33]


def _ruleset(name: str) -> dict[str, str]:
    return {
        "ruleset_id": name,
        "ruleset_version": f"{name}-v1",
        "ruleset_hash": ("d" if name == "legacy" else "e") * 64,
    }


def _delta(value: float = 0.02) -> dict[str, float]:
    return {"estimate": value, "low": value / 2.0, "high": value * 1.5}


def _identity(*, promotion: bool = False, authorized_bc: bool = False):
    return {
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "driver": "595.71.05",
        "cuda": "13.2",
        "pytorch": "2.12.1+cu132",
        "gpu": "RTX 5070",
        "cpu": "test cpu",
        "feature_schema_hash": SHA,
        "trainer_topology_hash": SHA,
        "replay_protocol_hash": SHA,
        "training_seeds": SEEDS,
        "evaluation_seed": 44,
        "wall_clock_budget_seconds": 7200,
        "sample_budget": 1_000_000,
        "evaluation_deal_sets": {"legacy": "1" * 64, "standard": "2" * 64},
        "checkpoint_cadence_updates": 100,
        "authorized_bc_data": authorized_bc,
        "human_bc": (
            {
                "dataset_identity": "3" * 64,
                "license": "authorized-test-license",
                "dataset_version": "games-v1",
                "pseudonymization_contract": "hmac-sha256-pseudonym-v1",
                "hmac_key_id_hash": "4" * 64,
            }
            if authorized_bc else {
                "dataset_identity": None,
                "license": None,
                "dataset_version": None,
                "pseudonymization_contract": None,
                "hmac_key_id_hash": None,
            }
        ),
        "promotion_requested": promotion,
        "promotion_variant": "v3_full_hybrid" if promotion else None,
        "support_matrix_version": H8A_SUPPORT_MATRIX_VERSION,
        "support_matrix_hash": h8a_support_matrix_hash(),
    }


def _training(variant: str, ruleset: str, seed: int):
    suffix = hashlib_token(variant, ruleset, seed)
    return {
        "variant": variant,
        "seed": seed,
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "ruleset": _ruleset(ruleset),
        "training_config_hash": suffix,
        "model_identity_hash": SHA,
        "initial_checkpoint_sha256": SHA,
        "model_checkpoint_sha256": suffix,
        "hardware_hash": SHA,
        "checkpoint_enabled": True,
        "samples": 1_000_000,
        "wall_clock_seconds": 7200,
        "cumulative_training_seconds": 7200,
        "sigterm_observed": True,
        "fresh_container_resume": True,
        "counter_before_resume": 10,
        "counter_after_resume": 11,
        "model_hash_before_resume": "5" * 64,
        "model_hash_after_resume": "6" * 64,
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


def hashlib_token(variant: str, ruleset: str, seed: int) -> str:
    import hashlib

    return hashlib.sha256(f"{variant}:{ruleset}:{seed}".encode()).hexdigest()


def _evaluation(
    run: dict[str, object],
    *,
    tier: str = DEVELOPMENT,
    search_enabled: bool = False,
):
    deals = 20_000 if tier == DEVELOPMENT else 100_000
    variant = run["variant"]
    ruleset = run["ruleset"]["ruleset_id"]
    return {
        "variant": variant,
        "ruleset": copy.deepcopy(run["ruleset"]),
        "search_enabled": search_enabled,
        "tier": tier,
        "training_seed": run["seed"],
        "git_sha": GIT_SHA,
        "docker_image_digest": IMAGE,
        "training_config_hash": run["training_config_hash"],
        "model_checkpoint_sha256": run["model_checkpoint_sha256"],
        "model_package_sha256": SHA,
        "deal_set_id": "1" * 64 if ruleset == "legacy" else "2" * 64,
        "deals": deals,
        "games": deals * 2,
        "bootstrap_unit": "deal",
        "confidence_level": 0.95,
        "overall": {"wp_delta": _delta(), "adp_delta": _delta(0.2)},
        "by_role": {
            role: {
                "games": deals,
                "wp_delta": _delta(),
                "adp_delta": _delta(0.2),
            }
            for role in ("landlord", "landlord_up", "landlord_down")
        },
        "calibration_error": 0.02,
        "latency_ms": {
            "p50": 1.0, "p95": 2.0, "p99": 3.0, "budget_p99": 5.0
        },
        "spring_count": 1,
        "anti_spring_count": 1,
        "bomb_count": 1,
        "rocket_count": 1,
        "bidding_games": deals * 2 if ruleset == "standard" else 0,
        "search_triggers": 10 if search_enabled else 0,
        "search_fallbacks": 0,
        "invalid_actions": 0,
        "timeouts": 0,
        "model_load_failures": 0,
        "public_package_validated": tier == PROMOTION,
        "privileged_leakage": False,
        "artifact_stale": False,
        "search_effect": (
            {
                "baseline_search_enabled": False,
                "wp_delta": _delta(0.01),
                "adp_delta": _delta(0.1),
                "added_latency_p99_ms": 1.0,
                "latency_budget_ms": 2.0,
            }
            if tier == PROMOTION and search_enabled else None
        ),
    }


def _evidence(*, promotion: bool = False, authorized_bc: bool = False):
    identity = _identity(promotion=promotion, authorized_bc=authorized_bc)
    variants = list(REQUIRED_VARIANTS)
    if authorized_bc:
        variants.append("v3_full_hybrid_bc")
    runs = [
        _training(variant, ruleset, seed)
        for variant in variants
        for ruleset in H8A_VARIANT_RULESET_SUPPORT[variant]
        for seed in SEEDS
    ]
    evaluations = [_evaluation(run) for run in runs]
    if promotion:
        evaluations.extend(
            _evaluation(run, tier=PROMOTION, search_enabled=search_enabled)
            for run in runs
            if run["variant"] == "v3_full_hybrid"
            for search_enabled in (False, True)
        )
    return {
        "schema_version": H8_EVIDENCE_SCHEMA_VERSION,
        "experiment_identity": identity,
        "training_runs": runs,
        "evaluations": evaluations,
    }


def test_support_matrix_matches_current_mainline_contract() -> None:
    assert H8A_VARIANT_RULESET_SUPPORT["legacy_a1"] == ("legacy",)
    assert H8A_VARIANT_RULESET_SUPPORT["model_v2"] == ("legacy", "standard")
    assert H8A_VARIANT_RULESET_SUPPORT["v3_full_hybrid"] == (
        "legacy", "standard"
    )
    assert H8A_VARIANT_RULESET_SUPPORT["v3_full_hybrid_bc"] == ("legacy",)


def test_complete_development_report_is_valid_but_never_release_ready() -> None:
    report = validate_h8_formal_evidence(_evidence())
    assert report["development_status"] == "COMPLETE"
    assert report["release_candidate"] == "NONE"
    assert report["release_status"] == "NOT READY"
    assert report["playing_strength"] == "NOT MEASURED"
    assert not any("100000" in issue for issue in report["issues"])


def test_development_status_includes_training_stability_gates() -> None:
    payload = _evidence()
    payload["training_runs"][0]["fresh_container_resume"] = False
    report = validate_h8_formal_evidence(payload)
    assert report["development_status"] == "INCOMPLETE"
    assert any("fresh_container_resume is false" in issue for issue in report["issues"])


def test_complete_promotion_recomputes_ready() -> None:
    report = validate_h8_formal_evidence(_evidence(promotion=True))
    assert report["release_status"] == "READY"
    assert report["release_candidate"] == "v3_full_hybrid"
    assert report["playing_strength"] == "MEASURED"


def test_development_and_promotion_have_distinct_deal_gates() -> None:
    development = _evidence()
    development["evaluations"][0]["deals"] = 19_999
    development["evaluations"][0]["games"] = 39_998
    for metrics in development["evaluations"][0]["by_role"].values():
        metrics["games"] = 19_999
    report = validate_h8_formal_evidence(development)
    assert any("fewer than 20000" in issue for issue in report["issues"])
    assert not any("fewer than 100000" in issue for issue in report["issues"])

    promotion = _evidence(promotion=True)
    row = next(item for item in promotion["evaluations"] if item["tier"] == PROMOTION)
    row["deals"] = 99_999
    row["games"] = 199_998
    for metrics in row["by_role"].values():
        metrics["games"] = 99_999
    report = validate_h8_formal_evidence(promotion)
    assert any("fewer than 100000" in issue for issue in report["issues"])


def test_bc_is_optional_and_legacy_only() -> None:
    without_bc = validate_h8_formal_evidence(_evidence())
    assert "v3_full_hybrid_bc" not in without_bc["required_variants"]

    with_bc = validate_h8_formal_evidence(_evidence(authorized_bc=True))
    assert "v3_full_hybrid_bc" in with_bc["required_variants"]
    payload = _evidence(authorized_bc=True)
    bc = next(run for run in payload["training_runs"] if run["variant"] == "v3_full_hybrid_bc")
    bc["ruleset"] = _ruleset("standard")
    with pytest.raises(H8EvidenceError, match="unsupported training combination"):
        validate_h8_formal_evidence(payload)


def test_unauthorized_bc_identity_and_rows_fail_closed() -> None:
    payload = _evidence()
    payload["experiment_identity"]["human_bc"]["license"] = "surprise"
    with pytest.raises(H8EvidenceError, match="must be null"):
        validate_h8_formal_evidence(payload)

    payload = _evidence()
    payload["training_runs"].append(_training("v3_full_hybrid_bc", "legacy", 11))
    with pytest.raises(H8EvidenceError, match="requires authorized"):
        validate_h8_formal_evidence(payload)


def test_unsupported_combinations_fail_but_missing_supported_rows_are_not_ready() -> None:
    payload = _evidence()
    payload["training_runs"][0]["ruleset"] = _ruleset("standard")
    with pytest.raises(H8EvidenceError, match="legacy_a1/standard"):
        validate_h8_formal_evidence(payload)

    payload = _evidence()
    removed = payload["evaluations"].pop()
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "NOT READY"
    assert removed["variant"] in " ".join(report["issues"])


def test_search_on_off_share_one_training_checkpoint() -> None:
    payload = _evidence(promotion=True)
    search_on = next(
        row for row in payload["evaluations"]
        if row["tier"] == PROMOTION and row["search_enabled"]
    )
    search_on["model_checkpoint_sha256"] = "f" * 64
    with pytest.raises(H8EvidenceError, match="checkpoint drift"):
        validate_h8_formal_evidence(payload)


def test_each_row_binds_complete_ruleset_and_training_identity() -> None:
    payload = _evidence()
    payload["evaluations"][0]["ruleset"]["ruleset_hash"] = "f" * 64
    with pytest.raises(H8EvidenceError, match="ruleset identity drift"):
        validate_h8_formal_evidence(payload)

    payload = _evidence()
    payload["evaluations"][0]["training_config_hash"] = "f" * 64
    with pytest.raises(H8EvidenceError, match="training config drift"):
        validate_h8_formal_evidence(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["training_runs"].append(
                copy.deepcopy(value["training_runs"][0])
            ),
            "duplicate training",
        ),
        (
            lambda value: value["evaluations"][0].update(games=39_999),
            "games/deals",
        ),
        (
            lambda value: value["evaluations"][0]["latency_ms"].update(p50=4.0),
            "percentiles are unordered",
        ),
        (
            lambda value: value["experiment_identity"].update(surprise=True),
            "fields mismatch",
        ),
    ],
)
def test_malformed_or_contradictory_evidence_fails_closed(mutation, message) -> None:
    payload = _evidence()
    mutation(payload)
    with pytest.raises(H8EvidenceError, match=message):
        validate_h8_formal_evidence(payload)


def test_stale_artifact_is_valid_but_not_ready() -> None:
    payload = _evidence(promotion=True)
    payload["training_runs"][0]["artifact_stale"] = True
    payload["evaluations"][0]["artifact_stale"] = True
    report = validate_h8_formal_evidence(payload)
    assert report["release_status"] == "NOT READY"
    assert sum("artifact is stale" in issue for issue in report["issues"]) == 2
