"""Fail-closed H8 evidence validation for V3 Hybrid promotion.

The validator intentionally separates malformed or contradictory evidence
from evidence that is valid but insufficient for release.  The former raises
``H8EvidenceError``; the latter produces a deterministic ``NOT READY`` report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

H8_EVIDENCE_SCHEMA_VERSION = "v3-hybrid-h8-formal-evidence-v1"
H8_REPORT_SCHEMA_VERSION = "v3-hybrid-h8-release-report-v1"

REQUIRED_VARIANTS = (
    "legacy_a1",
    "model_v2",
    "v3_role",
    "v3_admc",
    "v3_oracle",
    "v3_belief",
    "v3_farmer_cooperation",
    "v3_full_hybrid_search_off",
    "v3_full_hybrid_search_on",
)
ROLES = ("landlord", "landlord_up", "landlord_down")
RULESETS = ("legacy", "standard")

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class H8EvidenceError(ValueError):
    """Raised when formal evidence is malformed or internally inconsistent."""


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise H8EvidenceError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    actual = set(value)
    if actual != fields:
        raise H8EvidenceError(
            f"{label} fields mismatch: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H8EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise H8EvidenceError(f"{label} must be finite and >= {minimum}")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise H8EvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise H8EvidenceError(f"{label} must be bool")
    return value


def _digest(value: object, label: str, pattern: re.Pattern[str] = _HEX64) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise H8EvidenceError(f"{label} has an invalid immutable identity")
    return value


_IDENTITY_FIELDS = {
    "git_sha", "docker_image_digest", "driver", "cuda", "pytorch", "gpu",
    "cpu", "resolved_config_hash", "training_semantics_hash", "workload_hash",
    "feature_schema_hash", "model_identity_hash", "ruleset_identity_hash",
    "trainer_topology_hash", "replay_protocol_hash", "initial_checkpoint_sha256",
    "evaluation_deal_sets", "training_seeds", "evaluation_seed",
    "wall_clock_budget_seconds", "sample_budget", "search_config_hash",
    "belief_config_hash", "oracle_config_hash", "cooperation_config_hash",
    "loss_schedule_hash", "checkpoint_cadence_updates", "authorized_bc_data",
}


def _validate_identity(identity: Mapping[str, Any]) -> None:
    _exact(identity, _IDENTITY_FIELDS, "experiment_identity")
    _digest(identity["git_sha"], "git_sha", _HEX40)
    _digest(identity["docker_image_digest"], "docker_image_digest", _IMAGE_DIGEST)
    for name in (
        "resolved_config_hash", "training_semantics_hash", "workload_hash",
        "feature_schema_hash", "model_identity_hash", "ruleset_identity_hash",
        "trainer_topology_hash", "replay_protocol_hash",
        "initial_checkpoint_sha256", "search_config_hash", "belief_config_hash",
        "oracle_config_hash", "cooperation_config_hash", "loss_schedule_hash",
    ):
        _digest(identity[name], name)
    for name in ("driver", "cuda", "pytorch", "gpu", "cpu"):
        if not isinstance(identity[name], str) or not identity[name].strip():
            raise H8EvidenceError(f"{name} must be a non-empty string")
    seeds = identity["training_seeds"]
    if (
        not isinstance(seeds, list) or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise H8EvidenceError("training_seeds must contain at least three unique ints")
    _integer(identity["evaluation_seed"], "evaluation_seed")
    _integer(identity["wall_clock_budget_seconds"], "wall_clock_budget_seconds", minimum=1)
    _integer(identity["sample_budget"], "sample_budget", minimum=1)
    _integer(identity["checkpoint_cadence_updates"], "checkpoint_cadence_updates", minimum=1)
    _boolean(identity["authorized_bc_data"], "authorized_bc_data")
    deal_sets = _mapping(identity["evaluation_deal_sets"], "evaluation_deal_sets")
    _exact(deal_sets, set(RULESETS), "evaluation_deal_sets")
    for ruleset, digest in deal_sets.items():
        _digest(digest, f"evaluation_deal_sets.{ruleset}")


_TRAINING_FIELDS = {
    "variant", "seed", "git_sha", "docker_image_digest", "config_hash",
    "hardware_hash", "checkpoint_enabled", "samples", "wall_clock_seconds",
    "cumulative_training_seconds", "sigterm_observed", "fresh_container_resume",
    "counter_before_resume", "counter_after_resume", "model_hash_before_resume",
    "model_hash_after_resume", "optimizer_state_continuous", "schedule_continuous",
    "policy_version_continuous", "rng_state_restored", "loss_finite",
    "gradient_finite", "clean_shutdown", "policy_lag_max", "policy_lag_limit",
    "memory_plateau", "active", "in_flight", "pending", "temporary_checkpoints",
    "artifact_stale",
}


def _validate_training_run(run: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    _exact(run, _TRAINING_FIELDS, "training run")
    issues: list[str] = []
    variant = run["variant"]
    if not isinstance(variant, str) or not variant:
        raise H8EvidenceError("training run variant must be non-empty")
    if run["seed"] not in identity["training_seeds"]:
        raise H8EvidenceError(f"training run {variant} uses an undeclared seed")
    if run["git_sha"] != identity["git_sha"]:
        raise H8EvidenceError(f"training run {variant} source SHA drift")
    if run["docker_image_digest"] != identity["docker_image_digest"]:
        raise H8EvidenceError(f"training run {variant} image digest drift")
    for name in ("config_hash", "hardware_hash", "model_hash_before_resume", "model_hash_after_resume"):
        _digest(run[name], f"training run {variant}.{name}")
    samples = _integer(run["samples"], f"training run {variant}.samples", minimum=1)
    wall = _finite(run["wall_clock_seconds"], f"training run {variant}.wall_clock_seconds", minimum=0.0)
    cumulative = _finite(run["cumulative_training_seconds"], f"training run {variant}.cumulative_training_seconds", minimum=0.0)
    before = _integer(run["counter_before_resume"], f"training run {variant}.counter_before_resume")
    after = _integer(run["counter_after_resume"], f"training run {variant}.counter_after_resume")
    lag = _integer(run["policy_lag_max"], f"training run {variant}.policy_lag_max")
    lag_limit = _integer(run["policy_lag_limit"], f"training run {variant}.policy_lag_limit")
    for name in ("active", "in_flight", "pending", "temporary_checkpoints"):
        _integer(run[name], f"training run {variant}.{name}")
    for name in (
        "checkpoint_enabled", "sigterm_observed", "fresh_container_resume",
        "optimizer_state_continuous", "schedule_continuous", "policy_version_continuous",
        "rng_state_restored", "loss_finite", "gradient_finite", "clean_shutdown",
        "memory_plateau", "artifact_stale",
    ):
        _boolean(run[name], f"training run {variant}.{name}")
    required_true = (
        "checkpoint_enabled", "sigterm_observed", "fresh_container_resume",
        "optimizer_state_continuous", "schedule_continuous", "policy_version_continuous",
        "rng_state_restored", "loss_finite", "gradient_finite", "clean_shutdown",
        "memory_plateau",
    )
    for name in required_true:
        if not run[name]:
            issues.append(f"{variant}/seed-{run['seed']}: {name} is false")
    if samples != identity["sample_budget"]:
        issues.append(f"{variant}/seed-{run['seed']}: sample budget mismatch")
    if wall != float(identity["wall_clock_budget_seconds"]):
        issues.append(f"{variant}/seed-{run['seed']}: wall-clock budget mismatch")
    if cumulative < 7200.0:
        issues.append(f"{variant}/seed-{run['seed']}: cumulative training is under two hours")
    if after <= before:
        issues.append(f"{variant}/seed-{run['seed']}: resume counter did not advance")
    if run["model_hash_after_resume"] == run["model_hash_before_resume"]:
        issues.append(f"{variant}/seed-{run['seed']}: model hash did not change after resume")
    if lag > lag_limit:
        issues.append(f"{variant}/seed-{run['seed']}: policy lag exceeded its frozen limit")
    if any(run[name] != 0 for name in ("active", "in_flight", "pending", "temporary_checkpoints")):
        issues.append(f"{variant}/seed-{run['seed']}: shutdown/checkpoint residue is non-zero")
    if run["artifact_stale"]:
        issues.append(f"{variant}/seed-{run['seed']}: artifact is stale")
    return issues


_DELTA_FIELDS = {"estimate", "low", "high"}
_EVALUATION_FIELDS = {
    "variant", "ruleset", "search_enabled", "training_seed", "git_sha",
    "docker_image_digest", "model_checkpoint_sha256", "model_package_sha256",
    "deal_set_id", "deals", "games", "bootstrap_unit", "confidence_level",
    "overall", "by_role", "calibration_error", "latency_ms", "spring_count",
    "anti_spring_count", "bomb_count", "rocket_count", "bidding_games",
    "search_triggers", "search_fallbacks", "invalid_actions", "timeouts",
    "model_load_failures", "public_package_validated", "privileged_leakage",
    "artifact_stale",
}


def _delta(value: object, label: str) -> tuple[float, float, float]:
    payload = _mapping(value, label)
    _exact(payload, _DELTA_FIELDS, label)
    estimate = _finite(payload["estimate"], f"{label}.estimate")
    low = _finite(payload["low"], f"{label}.low")
    high = _finite(payload["high"], f"{label}.high")
    if not low <= estimate <= high:
        raise H8EvidenceError(f"{label} CI must contain its estimate")
    return estimate, low, high


def _validate_evaluation(row: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    _exact(row, _EVALUATION_FIELDS, "evaluation")
    variant = row["variant"]
    ruleset = row["ruleset"]
    seed = row["training_seed"]
    if not isinstance(variant, str) or not variant:
        raise H8EvidenceError("evaluation variant must be non-empty")
    if ruleset not in RULESETS:
        raise H8EvidenceError(f"evaluation {variant} ruleset is unsupported")
    if seed not in identity["training_seeds"]:
        raise H8EvidenceError(f"evaluation {variant} uses an undeclared seed")
    if row["git_sha"] != identity["git_sha"] or row["docker_image_digest"] != identity["docker_image_digest"]:
        raise H8EvidenceError(f"evaluation {variant} execution identity drift")
    for name in ("model_checkpoint_sha256", "model_package_sha256", "deal_set_id"):
        _digest(row[name], f"evaluation {variant}.{name}")
    if row["deal_set_id"] != identity["evaluation_deal_sets"][ruleset]:
        raise H8EvidenceError(f"evaluation {variant} deal-set identity drift")
    deals = _integer(row["deals"], f"evaluation {variant}.deals", minimum=1)
    games = _integer(row["games"], f"evaluation {variant}.games", minimum=1)
    if games < deals * 2 or games % deals:
        raise H8EvidenceError(f"evaluation {variant} games/deals are inconsistent")
    if row["bootstrap_unit"] != "deal" or row["confidence_level"] != 0.95:
        raise H8EvidenceError(f"evaluation {variant} must use deal-clustered 95% CI")
    _boolean(row["search_enabled"], f"evaluation {variant}.search_enabled")
    expected_search = variant == "v3_full_hybrid_search_on"
    if variant in {"v3_full_hybrid_search_on", "v3_full_hybrid_search_off"} and row["search_enabled"] != expected_search:
        raise H8EvidenceError(f"evaluation {variant} search identity mismatch")
    overall = _mapping(row["overall"], f"evaluation {variant}.overall")
    _exact(overall, {"wp_delta", "adp_delta"}, f"evaluation {variant}.overall")
    _, wp_low, _ = _delta(overall["wp_delta"], f"evaluation {variant}.overall.wp_delta")
    _, adp_low, _ = _delta(overall["adp_delta"], f"evaluation {variant}.overall.adp_delta")
    by_role = _mapping(row["by_role"], f"evaluation {variant}.by_role")
    _exact(by_role, set(ROLES), f"evaluation {variant}.by_role")
    role_regressed = False
    for role in ROLES:
        metrics = _mapping(by_role[role], f"evaluation {variant}.{role}")
        _exact(metrics, {"wp_delta", "adp_delta"}, f"evaluation {variant}.{role}")
        _, role_wp_low, _ = _delta(metrics["wp_delta"], f"evaluation {variant}.{role}.wp_delta")
        _, role_adp_low, _ = _delta(metrics["adp_delta"], f"evaluation {variant}.{role}.adp_delta")
        role_regressed |= role_wp_low < 0.0 or role_adp_low < 0.0
    latency = _mapping(row["latency_ms"], f"evaluation {variant}.latency_ms")
    _exact(latency, {"p50", "p95", "p99", "budget_p99"}, f"evaluation {variant}.latency_ms")
    p50 = _finite(latency["p50"], f"evaluation {variant}.latency.p50", minimum=0.0)
    p95 = _finite(latency["p95"], f"evaluation {variant}.latency.p95", minimum=0.0)
    p99 = _finite(latency["p99"], f"evaluation {variant}.latency.p99", minimum=0.0)
    budget = _finite(latency["budget_p99"], f"evaluation {variant}.latency.budget_p99", minimum=0.0)
    if not p50 <= p95 <= p99:
        raise H8EvidenceError(f"evaluation {variant} latency percentiles are unordered")
    _finite(row["calibration_error"], f"evaluation {variant}.calibration_error", minimum=0.0)
    for name in (
        "spring_count", "anti_spring_count", "bomb_count", "rocket_count",
        "bidding_games", "search_triggers", "search_fallbacks", "invalid_actions",
        "timeouts", "model_load_failures",
    ):
        _integer(row[name], f"evaluation {variant}.{name}")
    for name in ("public_package_validated", "privileged_leakage", "artifact_stale"):
        _boolean(row[name], f"evaluation {variant}.{name}")
    issues = []
    promotion_candidate = variant == "v3_full_hybrid_search_on"
    if deals < 100_000:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: fewer than 100000 paired deals")
    if promotion_candidate and (wp_low <= 0.0 or adp_low <= 0.0):
        issues.append(f"{variant}/{ruleset}/seed-{seed}: overall WP/ADP promotion CI failed")
    if promotion_candidate and role_regressed:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: at least one role regressed")
    if promotion_candidate and p99 > budget:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: p99 latency budget failed")
    if promotion_candidate and not row["public_package_validated"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: public package was not validated")
    if row["privileged_leakage"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: privileged leakage detected")
    if row["invalid_actions"] or row["timeouts"] or row["model_load_failures"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: runtime correctness failures observed")
    if row["search_fallbacks"] > row["search_triggers"]:
        raise H8EvidenceError(f"evaluation {variant} search fallback count exceeds triggers")
    if row["artifact_stale"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: artifact is stale")
    return issues


def validate_h8_formal_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate evidence and recompute the release decision from raw fields."""

    root = _mapping(payload, "H8 evidence")
    _exact(root, {"schema_version", "experiment_identity", "training_runs", "evaluations"}, "H8 evidence")
    if root["schema_version"] != H8_EVIDENCE_SCHEMA_VERSION:
        raise H8EvidenceError("unsupported H8 evidence schema_version")
    identity = _mapping(root["experiment_identity"], "experiment_identity")
    _validate_identity(identity)
    training_runs = root["training_runs"]
    evaluations = root["evaluations"]
    if not isinstance(training_runs, Sequence) or isinstance(training_runs, (str, bytes)):
        raise H8EvidenceError("training_runs must be a list")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise H8EvidenceError("evaluations must be a list")

    required = list(REQUIRED_VARIANTS)
    if identity["authorized_bc_data"]:
        required.append("v3_full_hybrid_bc")
    issues: list[str] = []
    seen_runs: Counter[tuple[str, int]] = Counter()
    samples_by_seed: defaultdict[int, set[int]] = defaultdict(set)
    wall_by_seed: defaultdict[int, set[float]] = defaultdict(set)
    for index, raw in enumerate(training_runs):
        run = _mapping(raw, f"training_runs[{index}]")
        issues.extend(_validate_training_run(run, identity))
        key = (run["variant"], run["seed"])
        seen_runs[key] += 1
        samples_by_seed[run["seed"]].add(run["samples"])
        wall_by_seed[run["seed"]].add(float(run["wall_clock_seconds"]))
    duplicates = sorted(key for key, count in seen_runs.items() if count != 1)
    if duplicates:
        raise H8EvidenceError(f"duplicate training variant/seed rows: {duplicates}")
    expected_runs = {(variant, seed) for variant in required for seed in identity["training_seeds"]}
    missing_runs = sorted(expected_runs - set(seen_runs))
    if missing_runs:
        issues.append(f"missing training runs: {missing_runs}")
    unknown_runs = sorted(set(seen_runs) - expected_runs)
    if unknown_runs:
        raise H8EvidenceError(f"undeclared training variants: {unknown_runs}")
    for seed in identity["training_seeds"]:
        if len(samples_by_seed[seed]) > 1 or len(wall_by_seed[seed]) > 1:
            raise H8EvidenceError(f"seed {seed} does not use matched budgets")

    seen_evaluations: Counter[tuple[str, str, int]] = Counter()
    for index, raw in enumerate(evaluations):
        row = _mapping(raw, f"evaluations[{index}]")
        issues.extend(_validate_evaluation(row, identity))
        key = (row["variant"], row["ruleset"], row["training_seed"])
        seen_evaluations[key] += 1
    duplicate_evaluations = sorted(key for key, count in seen_evaluations.items() if count != 1)
    if duplicate_evaluations:
        raise H8EvidenceError(f"duplicate evaluation rows: {duplicate_evaluations}")
    expected_evaluations = {
        (variant, ruleset, seed)
        for variant in required
        for ruleset in RULESETS
        for seed in identity["training_seeds"]
    }
    missing_evaluations = sorted(expected_evaluations - set(seen_evaluations))
    if missing_evaluations:
        issues.append(f"missing formal evaluations: {missing_evaluations}")
    unknown_evaluations = sorted(set(seen_evaluations) - expected_evaluations)
    if unknown_evaluations:
        raise H8EvidenceError(f"undeclared evaluation variants: {unknown_evaluations}")

    release_ready = not issues
    return {
        "schema_version": H8_REPORT_SCHEMA_VERSION,
        "evidence_sha256": canonical_hash(root),
        "release_candidate": "v3_full_hybrid_search_on" if release_ready else "NONE",
        "release_status": "READY" if release_ready else "NOT READY",
        "playing_strength": "measured" if release_ready else "not measured or promotion gates incomplete",
        "required_variants": required,
        "training_run_count": len(training_runs),
        "evaluation_count": len(evaluations),
        "issues": sorted(set(issues)),
    }


__all__ = [
    "H8_EVIDENCE_SCHEMA_VERSION",
    "H8_REPORT_SCHEMA_VERSION",
    "H8EvidenceError",
    "REQUIRED_VARIANTS",
    "ROLES",
    "RULESETS",
    "canonical_hash",
    "validate_h8_formal_evidence",
]
