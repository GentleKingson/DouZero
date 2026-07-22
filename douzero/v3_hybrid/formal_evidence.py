"""Fail-closed H8a evidence contracts for V3 Hybrid evaluation.

H8a validates infrastructure and may merge while release status is NOT READY.
The schema models only combinations supported by the current mainline, keeps
deployment search separate from training, and distinguishes development from
promotion evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

H8_EVIDENCE_SCHEMA_VERSION = "v3-hybrid-h8a-formal-evidence-v2"
H8_REPORT_SCHEMA_VERSION = "v3-hybrid-h8a-release-report-v2"
H8A_SUPPORT_MATRIX_VERSION = "v3-hybrid-h8a-variant-ruleset-v1"

DEVELOPMENT = "development"
PROMOTION = "promotion"
RULESET_LEGACY = "legacy"
RULESET_STANDARD = "standard"
RULESETS = (RULESET_LEGACY, RULESET_STANDARD)
ROLES = ("landlord", "landlord_up", "landlord_down")

_BASE_SUPPORT = {
    "legacy_a1": (RULESET_LEGACY,),
    "model_v2": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_role": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_admc": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_oracle": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_belief": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_farmer_cooperation": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_full_hybrid": (RULESET_LEGACY, RULESET_STANDARD),
    "v3_full_hybrid_bc": (RULESET_LEGACY,),
}
H8A_VARIANT_RULESET_SUPPORT: Mapping[str, tuple[str, ...]] = MappingProxyType(
    _BASE_SUPPORT
)
REQUIRED_VARIANTS = tuple(
    name for name in H8A_VARIANT_RULESET_SUPPORT
    if name != "v3_full_hybrid_bc"
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PUBLIC_METADATA = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+(),-]{0,127}\Z")
_SENSITIVE_METADATA_TOKENS = (
    "secret", "token", "password", "credential", "api_key", "apikey", "bearer",
)


class H8EvidenceError(ValueError):
    """Raised for malformed, contradictory, or unsupported evidence."""


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def h8a_support_matrix_dict() -> dict[str, object]:
    return {
        "version": H8A_SUPPORT_MATRIX_VERSION,
        "search_training_semantics": "deployment_wrapper_shared_checkpoint_v1",
        "variants": {
            name: {
                "rulesets": list(rulesets),
                "requires_authorized_bc_data": name == "v3_full_hybrid_bc",
            }
            for name, rulesets in H8A_VARIANT_RULESET_SUPPORT.items()
        },
    }


def h8a_support_matrix_hash() -> str:
    return canonical_hash(h8a_support_matrix_dict())


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


def _digest(
    value: object, label: str, pattern: re.Pattern[str] = _HEX64
) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise H8EvidenceError(f"{label} has an invalid immutable identity")
    return value


def _public_metadata(value: object, label: str) -> str:
    if not isinstance(value, str) or not _PUBLIC_METADATA.fullmatch(value):
        raise H8EvidenceError(f"{label} must use the public metadata schema")
    lowered = value.lower()
    if any(token in lowered for token in _SENSITIVE_METADATA_TOKENS):
        raise H8EvidenceError(f"{label} contains sensitive metadata")
    return value


def _ruleset_identity(value: object, label: str) -> Mapping[str, Any]:
    identity = _mapping(value, label)
    _exact(
        identity,
        {"ruleset_id", "ruleset_version", "ruleset_hash"},
        label,
    )
    ruleset = identity["ruleset_id"]
    if ruleset not in RULESETS:
        raise H8EvidenceError(f"{label}.ruleset_id is unsupported")
    if not isinstance(identity["ruleset_version"], str) or not identity[
        "ruleset_version"
    ]:
        raise H8EvidenceError(f"{label}.ruleset_version must be non-empty")
    _digest(identity["ruleset_hash"], f"{label}.ruleset_hash")
    return identity


_BC_FIELDS = {
    "dataset_identity", "license", "dataset_version",
    "pseudonymization_contract", "hmac_key_id_hash",
}
_IDENTITY_FIELDS = {
    "git_sha", "docker_image_digest", "driver", "cuda", "pytorch", "gpu",
    "cpu", "feature_schema_hash", "trainer_topology_hash",
    "replay_protocol_hash", "training_seeds", "evaluation_seed",
    "wall_clock_budget_seconds", "sample_budget", "evaluation_deal_sets",
    "checkpoint_cadence_updates", "authorized_bc_data", "human_bc",
    "promotion_requested", "promotion_variant", "support_matrix_version",
    "support_matrix_hash",
    "evaluation_baselines",
}
_BASELINE_FIELDS = {
    "variant", "training_config_hash", "model_checkpoint_sha256",
    "model_package_sha256",
}


def _validate_identity(identity: Mapping[str, Any]) -> None:
    _exact(identity, _IDENTITY_FIELDS, "experiment_identity")
    _digest(identity["git_sha"], "git_sha", _HEX40)
    _digest(identity["docker_image_digest"], "docker_image_digest", _IMAGE_DIGEST)
    for name in (
        "feature_schema_hash", "trainer_topology_hash", "replay_protocol_hash",
        "support_matrix_hash",
    ):
        _digest(identity[name], name)
    if identity["support_matrix_version"] != H8A_SUPPORT_MATRIX_VERSION:
        raise H8EvidenceError("support matrix version mismatch")
    if identity["support_matrix_hash"] != h8a_support_matrix_hash():
        raise H8EvidenceError("support matrix hash mismatch")
    for name in ("driver", "cuda", "pytorch", "gpu", "cpu"):
        _public_metadata(identity[name], name)
    seeds = identity["training_seeds"]
    if (
        not isinstance(seeds, list) or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise H8EvidenceError("training_seeds must contain at least three unique ints")
    _integer(identity["evaluation_seed"], "evaluation_seed")
    _integer(
        identity["wall_clock_budget_seconds"],
        "wall_clock_budget_seconds", minimum=1,
    )
    _integer(identity["sample_budget"], "sample_budget", minimum=1)
    _integer(
        identity["checkpoint_cadence_updates"],
        "checkpoint_cadence_updates", minimum=1,
    )
    authorized = _boolean(identity["authorized_bc_data"], "authorized_bc_data")
    bc = _mapping(identity["human_bc"], "human_bc")
    _exact(bc, _BC_FIELDS, "human_bc")
    if authorized:
        for name in ("dataset_identity", "hmac_key_id_hash"):
            _digest(bc[name], f"human_bc.{name}")
        for name in ("license", "dataset_version", "pseudonymization_contract"):
            _public_metadata(bc[name], f"human_bc.{name}")
    elif any(value is not None for value in bc.values()):
        raise H8EvidenceError(
            "human_bc identity must be null when authorized_bc_data=false"
        )
    promotion_requested = _boolean(
        identity["promotion_requested"], "promotion_requested"
    )
    promotion_variant = identity["promotion_variant"]
    allowed_promotions = {"v3_full_hybrid"}
    if authorized:
        allowed_promotions.add("v3_full_hybrid_bc")
    if promotion_requested:
        if promotion_variant not in allowed_promotions:
            raise H8EvidenceError("promotion_variant is not an eligible candidate")
    elif promotion_variant is not None:
        raise H8EvidenceError(
            "promotion_variant must be null when promotion_requested=false"
        )
    deal_sets = _mapping(identity["evaluation_deal_sets"], "evaluation_deal_sets")
    _exact(deal_sets, set(RULESETS), "evaluation_deal_sets")
    for ruleset, digest in deal_sets.items():
        _digest(digest, f"evaluation_deal_sets.{ruleset}")
    baselines = _mapping(identity["evaluation_baselines"], "evaluation_baselines")
    _exact(baselines, set(RULESETS), "evaluation_baselines")
    for ruleset, raw_baseline in baselines.items():
        baseline = _mapping(raw_baseline, f"evaluation_baselines.{ruleset}")
        _exact(baseline, _BASELINE_FIELDS, f"evaluation_baselines.{ruleset}")
        if baseline["variant"] not in H8A_VARIANT_RULESET_SUPPORT or not _supported(
            baseline["variant"], ruleset
        ):
            raise H8EvidenceError(
                f"evaluation_baselines.{ruleset}.variant is unsupported"
            )
        for name in _BASELINE_FIELDS - {"variant"}:
            _digest(baseline[name], f"evaluation_baselines.{ruleset}.{name}")


def _enabled_variants(identity: Mapping[str, Any]) -> tuple[str, ...]:
    variants = list(REQUIRED_VARIANTS)
    if identity["authorized_bc_data"]:
        variants.append("v3_full_hybrid_bc")
    return tuple(variants)


def _supported(variant: object, ruleset: object) -> bool:
    return (
        isinstance(variant, str)
        and isinstance(ruleset, str)
        and ruleset in H8A_VARIANT_RULESET_SUPPORT.get(variant, ())
    )


_TRAINING_FIELDS = {
    "variant", "seed", "git_sha", "docker_image_digest", "ruleset",
    "training_config_hash", "model_identity_hash", "initial_checkpoint_sha256",
    "model_checkpoint_sha256", "hardware_hash", "checkpoint_enabled", "samples",
    "wall_clock_seconds", "cumulative_training_seconds", "sigterm_observed",
    "fresh_container_resume", "counter_before_resume", "counter_after_resume",
    "model_hash_before_resume", "model_hash_after_resume",
    "optimizer_state_continuous", "schedule_continuous",
    "policy_version_continuous", "rng_state_restored", "loss_finite",
    "gradient_finite", "clean_shutdown", "policy_lag_max", "policy_lag_limit",
    "memory_plateau", "active", "in_flight", "pending", "temporary_checkpoints",
    "artifact_stale",
}


def _validate_training_run(
    run: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[str]:
    _exact(run, _TRAINING_FIELDS, "training run")
    variant = run["variant"]
    ruleset_identity = _ruleset_identity(run["ruleset"], "training run ruleset")
    ruleset = ruleset_identity["ruleset_id"]
    if variant not in H8A_VARIANT_RULESET_SUPPORT:
        raise H8EvidenceError(f"unknown training variant {variant!r}")
    if not _supported(variant, ruleset):
        raise H8EvidenceError(
            f"unsupported training combination {variant}/{ruleset}"
        )
    if variant == "v3_full_hybrid_bc" and not identity["authorized_bc_data"]:
        raise H8EvidenceError("BC evidence requires authorized_bc_data=true")
    seed = _integer(run["seed"], f"training run {variant}.seed")
    if seed not in identity["training_seeds"]:
        raise H8EvidenceError(f"training run {variant} uses an undeclared seed")
    if run["git_sha"] != identity["git_sha"]:
        raise H8EvidenceError(f"training run {variant} source SHA drift")
    if run["docker_image_digest"] != identity["docker_image_digest"]:
        raise H8EvidenceError(f"training run {variant} image digest drift")
    for name in (
        "training_config_hash", "model_identity_hash", "initial_checkpoint_sha256",
        "model_checkpoint_sha256", "hardware_hash", "model_hash_before_resume",
        "model_hash_after_resume",
    ):
        _digest(run[name], f"training run {variant}.{name}")
    samples = _integer(run["samples"], f"training run {variant}.samples", minimum=1)
    wall = _finite(
        run["wall_clock_seconds"], f"training run {variant}.wall_clock_seconds",
        minimum=0.0,
    )
    cumulative = _finite(
        run["cumulative_training_seconds"],
        f"training run {variant}.cumulative_training_seconds", minimum=0.0,
    )
    before = _integer(
        run["counter_before_resume"], f"training run {variant}.counter_before_resume"
    )
    after = _integer(
        run["counter_after_resume"], f"training run {variant}.counter_after_resume"
    )
    lag = _integer(run["policy_lag_max"], f"training run {variant}.policy_lag_max")
    lag_limit = _integer(
        run["policy_lag_limit"], f"training run {variant}.policy_lag_limit"
    )
    for name in ("active", "in_flight", "pending", "temporary_checkpoints"):
        _integer(run[name], f"training run {variant}.{name}")
    boolean_fields = (
        "checkpoint_enabled", "sigterm_observed", "fresh_container_resume",
        "optimizer_state_continuous", "schedule_continuous",
        "policy_version_continuous", "rng_state_restored", "loss_finite",
        "gradient_finite", "clean_shutdown", "memory_plateau", "artifact_stale",
    )
    for name in boolean_fields:
        _boolean(run[name], f"training run {variant}.{name}")
    issues: list[str] = []
    for name in boolean_fields[:-1]:
        if not run[name]:
            issues.append(f"{variant}/{ruleset}/seed-{seed}: {name} is false")
    if samples != identity["sample_budget"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: sample budget mismatch")
    if wall != float(identity["wall_clock_budget_seconds"]):
        issues.append(f"{variant}/{ruleset}/seed-{seed}: wall-clock budget mismatch")
    if cumulative < 7200.0:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}: cumulative training is under two hours"
        )
    if after <= before:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}: resume counter did not advance"
        )
    if run["model_hash_after_resume"] == run["model_hash_before_resume"]:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}: model hash did not change after resume"
        )
    if lag > lag_limit:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}: policy lag exceeded its frozen limit"
        )
    if any(
        run[name] != 0
        for name in ("active", "in_flight", "pending", "temporary_checkpoints")
    ):
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}: shutdown/checkpoint residue is non-zero"
        )
    if run["artifact_stale"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}: artifact is stale")
    return issues


_DELTA_FIELDS = {"estimate", "low", "high"}
_EVALUATION_FIELDS = {
    "variant", "ruleset", "search_enabled", "tier", "training_seed", "git_sha",
    "docker_image_digest", "training_config_hash", "model_checkpoint_sha256",
    "model_package_sha256", "deal_set_id", "deals", "games", "bootstrap_unit",
    "confidence_level", "overall", "by_role", "calibration_error", "latency_ms",
    "spring_count", "anti_spring_count", "bomb_count", "rocket_count",
    "bidding_games", "search_triggers", "search_fallbacks", "invalid_actions",
    "timeouts", "model_load_failures", "public_package_validated",
    "privileged_leakage", "artifact_stale", "search_effect",
    "outcome_histogram", "bootstrap_resamples",
    "reference",
}
_OUTCOME_ATOM_FIELDS = {"count", "overall", "by_role"}
_OUTCOME_METRICS = ("wp_delta", "adp_delta")
_MIN_BOOTSTRAP_RESAMPLES = 1_000
_MAX_BOOTSTRAP_RESAMPLES = 100_000
_MAX_OUTCOME_HISTOGRAM_ATOMS = 4_096
_MAX_BOOTSTRAP_CELLS = 10_000_000


def _bootstrap_shape(
    raw: object, resamples_value: object, *, label: str
) -> tuple[list[object], int]:
    if not isinstance(raw, list) or not raw:
        raise H8EvidenceError(f"{label}.outcome_histogram must be a non-empty list")
    if len(raw) > _MAX_OUTCOME_HISTOGRAM_ATOMS:
        raise H8EvidenceError(f"{label}.outcome_histogram is too large")
    resamples = _integer(
        resamples_value,
        f"{label}.bootstrap_resamples",
        minimum=_MIN_BOOTSTRAP_RESAMPLES,
    )
    if resamples > _MAX_BOOTSTRAP_RESAMPLES:
        raise H8EvidenceError(f"{label}.bootstrap_resamples exceeds the maximum")
    if len(raw) > 1 and len(raw) * resamples > _MAX_BOOTSTRAP_CELLS:
        raise H8EvidenceError(f"{label} bootstrap allocation exceeds the maximum")
    return raw, resamples


def _delta(value: object, label: str) -> tuple[float, float, float]:
    payload = _mapping(value, label)
    _exact(payload, _DELTA_FIELDS, label)
    estimate = _finite(payload["estimate"], f"{label}.estimate")
    low = _finite(payload["low"], f"{label}.low")
    high = _finite(payload["high"], f"{label}.high")
    if not low <= estimate <= high:
        raise H8EvidenceError(f"{label} CI must contain its estimate")
    return estimate, low, high


def _recompute_clustered_deltas(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    deals: int,
    label: str,
) -> dict[str, dict[str, dict[str, float]] | dict[str, float]]:
    raw, resamples = _bootstrap_shape(
        row["outcome_histogram"], row["bootstrap_resamples"], label=label
    )
    counts: list[int] = []
    vectors: list[list[float]] = []
    for index, raw_atom in enumerate(raw):
        atom_label = f"{label}.outcome_histogram[{index}]"
        atom = _mapping(raw_atom, atom_label)
        _exact(atom, _OUTCOME_ATOM_FIELDS, atom_label)
        counts.append(_integer(atom["count"], f"{atom_label}.count", minimum=1))
        overall = _mapping(atom["overall"], f"{atom_label}.overall")
        _exact(overall, set(_OUTCOME_METRICS), f"{atom_label}.overall")
        by_role = _mapping(atom["by_role"], f"{atom_label}.by_role")
        _exact(by_role, set(ROLES), f"{atom_label}.by_role")
        values: list[float] = []
        for metrics, metric_label in ((overall, f"{atom_label}.overall"), *(
            (_mapping(by_role[role], f"{atom_label}.{role}"), f"{atom_label}.{role}")
            for role in ROLES
        )):
            _exact(metrics, set(_OUTCOME_METRICS), metric_label)
            wp = _finite(metrics["wp_delta"], f"{metric_label}.wp_delta")
            adp = _finite(metrics["adp_delta"], f"{metric_label}.adp_delta")
            if not -1.0 <= wp <= 1.0:
                raise H8EvidenceError(f"{metric_label}.wp_delta must be in [-1, 1]")
            values.extend((wp, adp))
        vectors.append(values)
    if sum(counts) != deals:
        raise H8EvidenceError(f"{label} outcome histogram count does not equal deals")

    count_array = np.asarray(counts, dtype=np.int64)
    value_array = np.asarray(vectors, dtype=np.float64)
    estimate = count_array @ value_array / float(deals)
    if len(counts) == 1:
        low = high = estimate
    else:
        seed_material = {
            "evaluation_seed": identity["evaluation_seed"],
            "variant": row["variant"],
            "ruleset": row["ruleset"]["ruleset_id"],
            "training_seed": row["training_seed"],
            "tier": row["tier"],
            "search_enabled": row["search_enabled"],
        }
        seed = int(canonical_hash(seed_material)[:16], 16)
        rng = np.random.default_rng(seed)
        sampled = rng.multinomial(
            deals, count_array.astype(np.float64) / float(deals), size=resamples
        )
        bootstrap = sampled @ value_array / float(deals)
        low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)

    def result(offset: int) -> dict[str, float]:
        return {
            "estimate": float(estimate[offset]),
            "low": float(low[offset]),
            "high": float(high[offset]),
        }

    return {
        "overall": {metric: result(index) for index, metric in enumerate(_OUTCOME_METRICS)},
        "by_role": {
            role: {
                metric: result(2 + role_index * 2 + metric_index)
                for metric_index, metric in enumerate(_OUTCOME_METRICS)
            }
            for role_index, role in enumerate(ROLES)
        },
    }


def _assert_reported_delta(
    reported: object, recomputed: Mapping[str, float], label: str
) -> tuple[float, float, float]:
    values = _delta(reported, label)
    expected = tuple(recomputed[name] for name in ("estimate", "low", "high"))
    if any(
        not math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12)
        for actual, target in zip(values, expected)
    ):
        raise H8EvidenceError(f"{label} does not match recomputed clustered bootstrap")
    return values


def _recompute_search_effect(
    effect: Mapping[str, Any],
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    deals: int,
    label: str,
) -> dict[str, dict[str, float]]:
    raw, resamples = _bootstrap_shape(
        effect["outcome_histogram"], effect["bootstrap_resamples"], label=label
    )
    counts: list[int] = []
    vectors: list[list[float]] = []
    for index, raw_atom in enumerate(raw):
        atom_label = f"{label}.outcome_histogram[{index}]"
        atom = _mapping(raw_atom, atom_label)
        _exact(atom, {"count", "wp_delta", "adp_delta"}, atom_label)
        counts.append(_integer(atom["count"], f"{atom_label}.count", minimum=1))
        wp = _finite(atom["wp_delta"], f"{atom_label}.wp_delta")
        adp = _finite(atom["adp_delta"], f"{atom_label}.adp_delta")
        if not -1.0 <= wp <= 1.0:
            raise H8EvidenceError(f"{atom_label}.wp_delta must be in [-1, 1]")
        vectors.append([wp, adp])
    if sum(counts) != deals:
        raise H8EvidenceError(
            f"{label} outcome histogram count does not equal deals"
        )
    count_array = np.asarray(counts, dtype=np.int64)
    value_array = np.asarray(vectors, dtype=np.float64)
    estimate = count_array @ value_array / float(deals)
    if len(counts) == 1:
        low = high = estimate
    else:
        seed_material = {
            "evaluation_seed": identity["evaluation_seed"],
            "variant": row["variant"],
            "ruleset": row["ruleset"]["ruleset_id"],
            "training_seed": row["training_seed"],
            "tier": row["tier"],
            "search_effect": True,
        }
        rng = np.random.default_rng(int(canonical_hash(seed_material)[:16], 16))
        sampled = rng.multinomial(
            deals, count_array.astype(np.float64) / float(deals), size=resamples
        )
        bootstrap = sampled @ value_array / float(deals)
        low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return {
        metric: {
            "estimate": float(estimate[index]),
            "low": float(low[index]),
            "high": float(high[index]),
        }
        for index, metric in enumerate(_OUTCOME_METRICS)
    }


def _validate_evaluation(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    training: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[str]:
    _exact(row, _EVALUATION_FIELDS, "evaluation")
    variant = row["variant"]
    ruleset_identity = _ruleset_identity(row["ruleset"], "evaluation ruleset")
    ruleset = ruleset_identity["ruleset_id"]
    if variant not in H8A_VARIANT_RULESET_SUPPORT:
        raise H8EvidenceError(f"unknown evaluation variant {variant!r}")
    if not _supported(variant, ruleset):
        raise H8EvidenceError(
            f"unsupported evaluation combination {variant}/{ruleset}"
        )
    if variant == "v3_full_hybrid_bc" and not identity["authorized_bc_data"]:
        raise H8EvidenceError("BC evidence requires authorized_bc_data=true")
    seed = _integer(row["training_seed"], f"evaluation {variant}.training_seed")
    training_key = (variant, ruleset, seed)
    if training_key not in training:
        raise H8EvidenceError(
            f"evaluation {variant}/{ruleset}/seed-{seed} has no training run"
        )
    source_run = training[training_key]
    if row["git_sha"] != identity["git_sha"] or row[
        "docker_image_digest"
    ] != identity["docker_image_digest"]:
        raise H8EvidenceError(f"evaluation {variant} execution identity drift")
    for name in (
        "training_config_hash", "model_checkpoint_sha256",
        "model_package_sha256", "deal_set_id",
    ):
        _digest(row[name], f"evaluation {variant}.{name}")
    if row["training_config_hash"] != source_run["training_config_hash"]:
        raise H8EvidenceError(f"evaluation {variant} training config drift")
    if row["model_checkpoint_sha256"] != source_run["model_checkpoint_sha256"]:
        raise H8EvidenceError(f"evaluation {variant} checkpoint drift")
    if ruleset_identity != source_run["ruleset"]:
        raise H8EvidenceError(f"evaluation {variant} ruleset identity drift")
    if row["deal_set_id"] != identity["evaluation_deal_sets"][ruleset]:
        raise H8EvidenceError(f"evaluation {variant} deal-set identity drift")
    reference = _mapping(row["reference"], f"evaluation {variant}.reference")
    _exact(reference, _BASELINE_FIELDS, f"evaluation {variant}.reference")
    if reference != identity["evaluation_baselines"][ruleset]:
        raise H8EvidenceError(f"evaluation {variant} comparison baseline drift")
    tier = row["tier"]
    if tier not in {DEVELOPMENT, PROMOTION}:
        raise H8EvidenceError(f"evaluation {variant} tier is invalid")
    if tier == PROMOTION and (
        not identity["promotion_requested"]
        or variant != identity["promotion_variant"]
    ):
        raise H8EvidenceError("promotion evidence does not match frozen candidate")
    deals = _integer(row["deals"], f"evaluation {variant}.deals", minimum=1)
    games = _integer(row["games"], f"evaluation {variant}.games", minimum=1)
    if games != deals * 2:
        raise H8EvidenceError(f"evaluation {variant} games/deals are inconsistent")
    if row["bootstrap_unit"] != "deal" or row["confidence_level"] != 0.95:
        raise H8EvidenceError(f"evaluation {variant} must use deal-clustered 95% CI")
    recomputed = _recompute_clustered_deltas(
        row,
        identity,
        deals=deals,
        label=f"evaluation {variant}",
    )
    search_enabled = _boolean(
        row["search_enabled"], f"evaluation {variant}.search_enabled"
    )
    if search_enabled and variant not in {
        "v3_full_hybrid", "v3_full_hybrid_bc"
    }:
        raise H8EvidenceError(
            "search-on evaluation is supported only for a full-hybrid checkpoint"
        )
    overall = _mapping(row["overall"], f"evaluation {variant}.overall")
    _exact(overall, {"wp_delta", "adp_delta"}, f"evaluation {variant}.overall")
    _, wp_low, _ = _assert_reported_delta(
        overall["wp_delta"],
        recomputed["overall"]["wp_delta"],
        f"evaluation {variant}.overall.wp_delta",
    )
    _, adp_low, _ = _assert_reported_delta(
        overall["adp_delta"],
        recomputed["overall"]["adp_delta"],
        f"evaluation {variant}.overall.adp_delta",
    )
    by_role = _mapping(row["by_role"], f"evaluation {variant}.by_role")
    _exact(by_role, set(ROLES), f"evaluation {variant}.by_role")
    role_regressed = False
    for role in ROLES:
        metrics = _mapping(by_role[role], f"evaluation {variant}.{role}")
        _exact(
            metrics, {"games", "wp_delta", "adp_delta"},
            f"evaluation {variant}.{role}",
        )
        if _integer(metrics["games"], f"evaluation {variant}.{role}.games") != deals:
            raise H8EvidenceError(
                f"evaluation {variant} role/deal counts are inconsistent"
            )
        _, role_wp_low, _ = _assert_reported_delta(
            metrics["wp_delta"],
            recomputed["by_role"][role]["wp_delta"],
            f"evaluation {variant}.{role}.wp_delta",
        )
        _, role_adp_low, _ = _assert_reported_delta(
            metrics["adp_delta"],
            recomputed["by_role"][role]["adp_delta"],
            f"evaluation {variant}.{role}.adp_delta",
        )
        role_regressed |= role_wp_low < 0.0 or role_adp_low < 0.0
    latency = _mapping(row["latency_ms"], f"evaluation {variant}.latency_ms")
    _exact(
        latency, {"p50", "p95", "p99", "budget_p99"},
        f"evaluation {variant}.latency_ms",
    )
    p50 = _finite(latency["p50"], f"evaluation {variant}.latency.p50", minimum=0.0)
    p95 = _finite(latency["p95"], f"evaluation {variant}.latency.p95", minimum=0.0)
    p99 = _finite(latency["p99"], f"evaluation {variant}.latency.p99", minimum=0.0)
    budget = _finite(
        latency["budget_p99"], f"evaluation {variant}.latency.budget_p99",
        minimum=0.0,
    )
    if not p50 <= p95 <= p99:
        raise H8EvidenceError(f"evaluation {variant} latency percentiles are unordered")
    _finite(
        row["calibration_error"], f"evaluation {variant}.calibration_error",
        minimum=0.0,
    )
    for name in (
        "spring_count", "anti_spring_count", "bomb_count", "rocket_count",
        "bidding_games", "search_triggers", "search_fallbacks", "invalid_actions",
        "timeouts", "model_load_failures",
    ):
        _integer(row[name], f"evaluation {variant}.{name}")
    for name in ("public_package_validated", "privileged_leakage", "artifact_stale"):
        _boolean(row[name], f"evaluation {variant}.{name}")
    if row["search_fallbacks"] > row["search_triggers"]:
        raise H8EvidenceError(
            f"evaluation {variant} search fallback count exceeds triggers"
        )

    issues: list[str] = []
    minimum_deals = 20_000 if tier == DEVELOPMENT else 100_000
    if deals < minimum_deals:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}/{tier}: fewer than {minimum_deals} paired deals"
        )
    if row["privileged_leakage"]:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}/{tier}: privileged leakage detected"
        )
    if row["invalid_actions"] or row["timeouts"] or row["model_load_failures"]:
        issues.append(
            f"{variant}/{ruleset}/seed-{seed}/{tier}: "
            "runtime correctness failures observed"
        )
    if row["artifact_stale"]:
        issues.append(f"{variant}/{ruleset}/seed-{seed}/{tier}: artifact is stale")

    is_promotion = tier == PROMOTION
    if is_promotion:
        if wp_low <= 0.0 or adp_low <= 0.0:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: overall WP/ADP promotion CI failed"
            )
        if role_regressed:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: at least one role regressed"
            )
        if p99 > budget:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: p99 latency budget failed"
            )
        if not row["public_package_validated"]:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: public package was not validated"
            )
    if is_promotion and search_enabled:
        effect = _mapping(row["search_effect"], f"evaluation {variant}.search_effect")
        _exact(
            effect,
            {
                "baseline_search_enabled", "wp_delta", "adp_delta",
                "added_latency_p99_ms", "latency_budget_ms",
                "outcome_histogram", "bootstrap_resamples",
            },
            f"evaluation {variant}.search_effect",
        )
        if effect["baseline_search_enabled"] is not False:
            raise H8EvidenceError("search effect baseline must be search-off")
        recomputed_effect = _recompute_search_effect(
            effect,
            row,
            identity,
            deals=deals,
            label=f"evaluation {variant}.search_effect",
        )
        _, search_wp_low, _ = _assert_reported_delta(
            effect["wp_delta"],
            recomputed_effect["wp_delta"],
            f"evaluation {variant}.search_effect.wp_delta",
        )
        _, search_adp_low, _ = _assert_reported_delta(
            effect["adp_delta"],
            recomputed_effect["adp_delta"],
            f"evaluation {variant}.search_effect.adp_delta",
        )
        added_latency = _finite(
            effect["added_latency_p99_ms"],
            f"evaluation {variant}.search_effect.added_latency_p99_ms",
            minimum=0.0,
        )
        latency_budget = _finite(
            effect["latency_budget_ms"],
            f"evaluation {variant}.search_effect.latency_budget_ms",
            minimum=0.0,
        )
        if search_wp_low <= 0.0 or search_adp_low <= 0.0:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: search benefit CI failed"
            )
        if added_latency > latency_budget:
            issues.append(
                f"{variant}/{ruleset}/seed-{seed}: search latency budget failed"
            )
    elif row["search_effect"] is not None:
        raise H8EvidenceError(
            "search_effect is allowed only on promotion search-on rows"
        )
    return issues


def _expected_training_keys(
    identity: Mapping[str, Any]
) -> set[tuple[str, str, int]]:
    return {
        (variant, ruleset, seed)
        for variant in _enabled_variants(identity)
        for ruleset in H8A_VARIANT_RULESET_SUPPORT[variant]
        for seed in identity["training_seeds"]
    }


def _expected_development_keys(
    identity: Mapping[str, Any]
) -> set[tuple[str, str, int, str, bool]]:
    return {
        (variant, ruleset, seed, DEVELOPMENT, False)
        for variant, ruleset, seed in _expected_training_keys(identity)
    }


def _expected_promotion_keys(
    identity: Mapping[str, Any]
) -> set[tuple[str, str, int, str, bool]]:
    if not identity["promotion_requested"]:
        return set()
    variant = identity["promotion_variant"]
    return {
        (variant, ruleset, seed, PROMOTION, search_enabled)
        for ruleset in H8A_VARIANT_RULESET_SUPPORT[variant]
        for seed in identity["training_seeds"]
        for search_enabled in (False, True)
    }


def validate_h8_formal_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate evidence and recompute development and release decisions."""

    root = _mapping(payload, "H8 evidence")
    _exact(
        root,
        {"schema_version", "experiment_identity", "training_runs", "evaluations"},
        "H8 evidence",
    )
    if root["schema_version"] != H8_EVIDENCE_SCHEMA_VERSION:
        raise H8EvidenceError("unsupported H8 evidence schema_version")
    identity = _mapping(root["experiment_identity"], "experiment_identity")
    _validate_identity(identity)
    training_rows = root["training_runs"]
    evaluation_rows = root["evaluations"]
    if not isinstance(training_rows, Sequence) or isinstance(
        training_rows, (str, bytes)
    ):
        raise H8EvidenceError("training_runs must be a list")
    if not isinstance(evaluation_rows, Sequence) or isinstance(
        evaluation_rows, (str, bytes)
    ):
        raise H8EvidenceError("evaluations must be a list")

    training_issues: list[str] = []
    development_evaluation_issues: list[str] = []
    promotion_evaluation_issues: list[str] = []
    training: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    samples_by_seed: dict[int, set[int]] = {
        seed: set() for seed in identity["training_seeds"]
    }
    wall_by_seed: dict[int, set[float]] = {
        seed: set() for seed in identity["training_seeds"]
    }
    for index, raw in enumerate(training_rows):
        row = _mapping(raw, f"training_runs[{index}]")
        training_issues.extend(_validate_training_run(row, identity))
        key = (row["variant"], row["ruleset"]["ruleset_id"], row["seed"])
        if key in training:
            raise H8EvidenceError(f"duplicate training row: {key}")
        training[key] = row
        samples_by_seed[row["seed"]].add(row["samples"])
        wall_by_seed[row["seed"]].add(float(row["wall_clock_seconds"]))
    expected_training = _expected_training_keys(identity)
    missing_training = sorted(expected_training - set(training))
    if missing_training:
        training_issues.append(f"missing training runs: {missing_training}")
    unknown_training = sorted(set(training) - expected_training)
    if unknown_training:
        raise H8EvidenceError(f"undeclared training rows: {unknown_training}")
    for variant, ruleset in {
        (variant, ruleset) for variant, ruleset, _seed in expected_training
    }:
        checkpoints = [
            row["model_checkpoint_sha256"]
            for (row_variant, row_ruleset, _seed), row in training.items()
            if row_variant == variant and row_ruleset == ruleset
        ]
        if len(checkpoints) != len(set(checkpoints)):
            raise H8EvidenceError(
                f"{variant}/{ruleset} training seeds must use distinct checkpoints"
            )
        model_identities = {
            row["model_identity_hash"]
            for (row_variant, row_ruleset, _seed), row in training.items()
            if row_variant == variant and row_ruleset == ruleset
        }
        if len(model_identities) > 1:
            raise H8EvidenceError(
                f"{variant}/{ruleset} training seeds must share one model identity"
            )
    for seed in identity["training_seeds"]:
        if len(samples_by_seed[seed]) > 1 or len(wall_by_seed[seed]) > 1:
            raise H8EvidenceError(f"seed {seed} does not use matched budgets")

    seen: Counter[tuple[str, str, int, str, bool]] = Counter()
    for index, raw in enumerate(evaluation_rows):
        row = _mapping(raw, f"evaluations[{index}]")
        row_issues = _validate_evaluation(row, identity, training)
        if row["tier"] == DEVELOPMENT:
            development_evaluation_issues.extend(row_issues)
        else:
            promotion_evaluation_issues.extend(row_issues)
        key = (
            row["variant"], row["ruleset"]["ruleset_id"],
            row["training_seed"], row["tier"], row["search_enabled"],
        )
        seen[key] += 1
    duplicates = sorted(key for key, count in seen.items() if count != 1)
    if duplicates:
        raise H8EvidenceError(f"duplicate evaluation rows: {duplicates}")
    expected_development = _expected_development_keys(identity)
    expected_promotion = _expected_promotion_keys(identity)
    expected_evaluations = expected_development | expected_promotion
    missing_development = sorted(expected_development - set(seen))
    missing_promotion = sorted(expected_promotion - set(seen))
    if missing_development:
        development_evaluation_issues.append(
            f"missing development evaluations: {missing_development}"
        )
    if missing_promotion:
        promotion_evaluation_issues.append(
            f"missing promotion evaluations: {missing_promotion}"
        )
    unknown_evaluations = sorted(set(seen) - expected_evaluations)
    if unknown_evaluations:
        raise H8EvidenceError(f"undeclared evaluation rows: {unknown_evaluations}")

    # Search is a deployment wrapper: both promotion rows must use the same
    # training checkpoint and training semantics.
    if identity["promotion_requested"]:
        for ruleset in H8A_VARIANT_RULESET_SUPPORT[identity["promotion_variant"]]:
            for seed in identity["training_seeds"]:
                grouped = [
                    row for row in evaluation_rows
                    if row["variant"] == identity["promotion_variant"]
                    and row["ruleset"]["ruleset_id"] == ruleset
                    and row["training_seed"] == seed
                    and row["tier"] == PROMOTION
                ]
                if len(grouped) == 2 and (
                    len({row["model_checkpoint_sha256"] for row in grouped}) != 1
                    or len({row["training_config_hash"] for row in grouped}) != 1
                ):
                    raise H8EvidenceError(
                        "search on/off promotion rows must share one training checkpoint"
                    )

    issues = (
        training_issues
        + development_evaluation_issues
        + promotion_evaluation_issues
    )
    development_issues = training_issues + development_evaluation_issues
    release_ready = identity["promotion_requested"] and not issues
    return {
        "schema_version": H8_REPORT_SCHEMA_VERSION,
        "evidence_sha256": canonical_hash(root),
        "support_matrix_hash": h8a_support_matrix_hash(),
        "development_status": (
            "COMPLETE" if not missing_development and not development_issues
            else "INCOMPLETE"
        ),
        "release_candidate": identity["promotion_variant"] if release_ready else "NONE",
        "release_status": "READY" if release_ready else "NOT READY",
        "playing_strength": (
            "MEASURED" if release_ready else "NOT MEASURED"
        ),
        "required_variants": list(_enabled_variants(identity)),
        "training_run_count": len(training_rows),
        "evaluation_count": len(evaluation_rows),
        "issues": sorted(set(issues)),
    }


__all__ = [
    "DEVELOPMENT",
    "H8A_SUPPORT_MATRIX_VERSION",
    "H8A_VARIANT_RULESET_SUPPORT",
    "H8_EVIDENCE_SCHEMA_VERSION",
    "H8_REPORT_SCHEMA_VERSION",
    "H8EvidenceError",
    "PROMOTION",
    "REQUIRED_VARIANTS",
    "ROLES",
    "RULESETS",
    "canonical_hash",
    "h8a_support_matrix_dict",
    "h8a_support_matrix_hash",
    "validate_h8_formal_evidence",
]
