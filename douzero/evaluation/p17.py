"""Fail-closed P17 model inventory and empirical-report collation."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .ablation import ABLATION_NAMES
from .protocol import (
    EVALUATION_PROTOCOL,
    OFFICIAL_PERMUTATION_HASHES,
    P17_MIN_BOOTSTRAP_SAMPLES,
    P17_MIN_PAIRED_DEALS,
    P17_READINESS_PROTOCOL,
    PROMOTION_ESTIMATOR,
    PROMOTION_MODE,
)
from .scenario import BundleSpec, bundle_from_dict, default_seat_permutations
from .statistics import deal_cluster_means, paired_bootstrap_ci


P17_MATRIX_SCHEMA_VERSION = "p17-model-matrix-v1"
P17_REPORT_SCHEMA_VERSION = "p17-evaluation-report-v1"
P17_MODEL_NAMES = (
    "legacy_wp",
    "legacy_adp",
    "legacy_factorized",
    "v2_base",
    "v2_multi_objective",
    "v2_belief_frozen",
    "v2_belief_joint",
    "v2_human_bc",
    "v2_strategy_auxiliary",
    "v2_distillation",
    "v2_population",
    "v2_coach",
    "v2_search",
    "v2_full_stack",
)
P17_PROTOCOLS = ("cardplay_only", "full_game")


class P17MatrixError(ValueError):
    """Raised when a P17 inventory could hide a missing empirical input."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class MatrixEntry:
    status: str
    reason: str
    bundle: BundleSpec | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason or None,
            "bundle": self.bundle.to_dict() if self.bundle is not None else None,
        }


def empty_matrix(reason: str = "No compatible checkpoint supplied") -> dict[str, Any]:
    """Return a complete template that marks every empirical input unavailable."""

    unavailable = {"status": "unavailable", "reason": reason, "bundle": None}
    return {
        "schema_version": P17_MATRIX_SCHEMA_VERSION,
        "models": {
            name: {protocol: dict(unavailable) for protocol in P17_PROTOCOLS}
            for name in P17_MODEL_NAMES
        },
        "ablations": {
            name: {
                "status": "unavailable",
                "reason": "No independently trained compatible checkpoint supplied",
                "protocol": "cardplay_only" if name == "no_bidding" else "full_game",
                "candidate_model": None,
                "baseline_model": None,
            }
            for name in ABLATION_NAMES
        },
    }


def _validate_available_bundle_checkpoints(
    model_name: str,
    protocol: str,
    bundle: BundleSpec,
    *,
    model_config: object | None,
) -> None:
    """Eagerly prove that every declared formal checkpoint is loadable."""
    from douzero.env.rules import RuleSet

    ruleset = RuleSet.legacy() if protocol == "cardplay_only" else RuleSet.standard()

    def fail(label: str, exc: Exception) -> None:
        raise P17MatrixError(
            f"{model_name}.{protocol} checkpoint identity validation failed "
            f"for {label}: {type(exc).__name__}"
        ) from exc

    if bundle.backend in ("v2", "bc"):
        from douzero.checkpoint import load_v2_position_weights
        from douzero.models_v2.model import ModelV2
        from douzero.observation.schema import build_v2_schema

        if model_config is None:
            raise P17MatrixError(
                f"{model_name}.{protocol} requires an explicit V2 model_config"
            )
        schema = build_v2_schema()
        seen: set[str] = set()

        def validate_v2(path: str, label: str) -> None:
            if path in seen:
                return
            seen.add(path)
            try:
                state_dict, _ = load_v2_position_weights(
                    path,
                    expected_schema_hash=schema.stable_hash(),
                    expected_model_config_hash=model_config.stable_hash(),
                    expected_ruleset=ruleset,
                    runtime_model_config=model_config,
                    training_device="cpu",
                )
                ModelV2(schema, model_config).load_state_dict(
                    state_dict, strict=True
                )
            except Exception as exc:
                fail(label, exc)

        for role, path in bundle.checkpoints.items():
            validate_v2(path, role)
        if bundle.bidding_checkpoint:
            validate_v2(bundle.bidding_checkpoint, "bidding")
        if bundle.belief_checkpoint:
            try:
                from douzero.belief.checkpoint import load_belief_checkpoint

                load_belief_checkpoint(
                    bundle.belief_checkpoint,
                    expected_ruleset=ruleset,
                    expected_feature_version="v2",
                )
            except Exception as exc:
                fail("belief", exc)
        return

    if bundle.backend in ("legacy", "legacy_factorized"):
        from douzero.checkpoint import load_position_state_dict_strict

        factories = (
            __import__(
                "douzero.dmc.models_factorized",
                fromlist=["factorized_model_dict"],
            ).factorized_model_dict
            if bundle.backend == "legacy_factorized"
            else __import__("douzero.dmc.models", fromlist=["model_dict"]).model_dict
        )
        for role, path in bundle.checkpoints.items():
            try:
                runtime_model = factories[role]()
                load_position_state_dict_strict(path, runtime_model.state_dict())
            except Exception as exc:
                fail(role, exc)
        return

    raise P17MatrixError(
        f"{model_name}.{protocol} has unsupported formal backend {bundle.backend!r}"
    )


def _parse_entry(
    model_name: str, protocol: str, raw: object
) -> MatrixEntry:
    if not isinstance(raw, Mapping):
        raise P17MatrixError(f"{model_name}.{protocol} must be an object")
    if set(raw) != {"status", "reason", "bundle"}:
        raise P17MatrixError(
            f"{model_name}.{protocol} must contain status, reason, and bundle"
        )
    status = raw["status"]
    reason = raw["reason"]
    if status not in ("available", "unavailable"):
        raise P17MatrixError(f"{model_name}.{protocol} has invalid status {status!r}")
    if not isinstance(reason, str):
        raise P17MatrixError(f"{model_name}.{protocol}.reason must be a string")
    if status == "unavailable":
        if not reason.strip() or raw["bundle"] is not None:
            raise P17MatrixError(
                f"unavailable {model_name}.{protocol} requires a reason and null bundle"
            )
        return MatrixEntry(status, reason, None)
    if reason:
        raise P17MatrixError(f"available {model_name}.{protocol} must have an empty reason")
    if not isinstance(raw["bundle"], Mapping):
        raise P17MatrixError(f"available {model_name}.{protocol} requires a bundle")
    bundle = bundle_from_dict({**dict(raw["bundle"]), "name": model_name})
    if bundle.backend in ("random", "rule"):
        raise P17MatrixError(
            f"{model_name}.{protocol} cannot use a smoke-only {bundle.backend!r} backend"
        )
    missing_files = [
        role for role, path in bundle.checkpoints.items() if not Path(path).is_file()
    ]
    if bundle.belief_checkpoint and not Path(bundle.belief_checkpoint).is_file():
        missing_files.append("belief")
    if bundle.bidding_checkpoint and not Path(bundle.bidding_checkpoint).is_file():
        missing_files.append("bidding")
    if missing_files:
        raise P17MatrixError(
            f"{model_name}.{protocol} checkpoint files are missing for {missing_files}"
        )
    config = None
    if bundle.backend in ("v2", "bc"):
        from douzero.models_v2.config import ModelV2Config

        try:
            config = ModelV2Config(**dict(bundle.model_config))
        except (TypeError, ValueError) as exc:
            raise P17MatrixError(
                f"{model_name}.{protocol} has invalid model_config: {exc}"
            ) from exc
        if config.belief_enabled and not bundle.belief_checkpoint:
            raise P17MatrixError(
                f"{model_name}.{protocol} enables belief fusion but has no "
                "belief_checkpoint"
            )
        if bundle.belief_checkpoint and not config.belief_enabled:
            raise P17MatrixError(
                f"{model_name}.{protocol} supplies a belief_checkpoint while "
                "model_config.belief_enabled=false"
            )
    if protocol == "full_game":
        if bundle.bidding_policy != "learned":
            raise P17MatrixError(
                f"{model_name}.full_game requires manifest-validated learned bidding"
            )
        if config is None or not config.bidding_enabled:
            raise P17MatrixError(
                f"{model_name}.full_game requires model_config.bidding_enabled=true"
            )
    _validate_available_bundle_checkpoints(
        model_name,
        protocol,
        bundle,
        model_config=config,
    )
    return MatrixEntry(status, reason, bundle)


def normalize_matrix(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete model/ablation inventory and redact local paths."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version", "models", "ablations"
    }:
        raise P17MatrixError("matrix must contain schema_version, models, and ablations")
    if raw["schema_version"] != P17_MATRIX_SCHEMA_VERSION:
        raise P17MatrixError("unsupported P17 model-matrix schema_version")
    models = raw["models"]
    if not isinstance(models, Mapping) or set(models) != set(P17_MODEL_NAMES):
        missing = sorted(set(P17_MODEL_NAMES) - set(models or {}))
        extra = sorted(set(models or {}) - set(P17_MODEL_NAMES))
        raise P17MatrixError(f"model matrix mismatch; missing={missing}, extra={extra}")
    parsed: dict[str, dict[str, MatrixEntry]] = {}
    for model_name in P17_MODEL_NAMES:
        rows = models[model_name]
        if not isinstance(rows, Mapping) or set(rows) != set(P17_PROTOCOLS):
            raise P17MatrixError(
                f"{model_name} must declare cardplay_only and full_game"
            )
        parsed[model_name] = {
            protocol: _parse_entry(model_name, protocol, rows[protocol])
            for protocol in P17_PROTOCOLS
        }

    ablations = raw["ablations"]
    if not isinstance(ablations, Mapping) or set(ablations) != set(ABLATION_NAMES):
        missing = sorted(set(ABLATION_NAMES) - set(ablations or {}))
        extra = sorted(set(ablations or {}) - set(ABLATION_NAMES))
        raise P17MatrixError(f"ablation matrix mismatch; missing={missing}, extra={extra}")
    normalized_ablations: dict[str, Any] = {}
    for name in ABLATION_NAMES:
        row = ablations[name]
        required = {
            "status", "reason", "protocol", "candidate_model", "baseline_model"
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise P17MatrixError(f"ablation {name} has an invalid field set")
        status = row["status"]
        protocol = row["protocol"]
        if status not in ("available", "unavailable") or protocol not in P17_PROTOCOLS:
            raise P17MatrixError(f"ablation {name} has an invalid status or protocol")
        if name == "no_bidding" and protocol != "cardplay_only":
            raise P17MatrixError("no_bidding must use the cardplay_only protocol")
        if status == "unavailable":
            if not isinstance(row["reason"], str) or not row["reason"].strip():
                raise P17MatrixError(f"unavailable ablation {name} requires a reason")
            if row["candidate_model"] is not None or row["baseline_model"] is not None:
                raise P17MatrixError(
                    f"unavailable ablation {name} must not claim model inputs"
                )
        else:
            if row["reason"]:
                raise P17MatrixError(f"available ablation {name} must have an empty reason")
            for key in ("candidate_model", "baseline_model"):
                model_name = row[key]
                if model_name not in P17_MODEL_NAMES:
                    raise P17MatrixError(f"ablation {name} references unknown {key}")
                if parsed[model_name][protocol].status != "available":
                    raise P17MatrixError(
                        f"ablation {name} references unavailable {model_name}.{protocol}"
                    )
        normalized_ablations[name] = dict(row)
    return {
        "schema_version": P17_MATRIX_SCHEMA_VERSION,
        "models": {
            name: {
                protocol: parsed[name][protocol].to_dict()
                for protocol in P17_PROTOCOLS
            }
            for name in P17_MODEL_NAMES
        },
        "ablations": normalized_ablations,
    }


def load_result(path: str | Path, expected_mode: str) -> dict[str, Any]:
    """Load one P15 result with enough checks to prevent protocol relabelling."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != EVALUATION_PROTOCOL:
        raise P17MatrixError(f"{path} is not a {EVALUATION_PROTOCOL} result")
    scenario = payload.get("scenario")
    metrics = payload.get("metrics")
    if not isinstance(scenario, dict) or scenario.get("mode") != expected_mode:
        raise P17MatrixError(f"{path} is not a {expected_mode} result")
    if not isinstance(metrics, dict) or not isinstance(payload.get("games"), list):
        raise P17MatrixError(f"{path} is missing metrics or auditable game rows")
    return payload


def _numbers_match(actual: object, expected: float | int) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _recompute_result_evidence(
    result: Mapping[str, Any], *, mode: str
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild release evidence from auditable game rows, never summary claims."""

    scenario = result.get("scenario")
    metrics = result.get("metrics")
    games = result.get("games")
    issues: list[str] = []
    if not isinstance(scenario, Mapping) or not isinstance(metrics, Mapping):
        return {"paired_deals": 0}, ["result scenario or metrics is malformed"]
    if not isinstance(games, list):
        return {"paired_deals": 0}, ["result games must be an auditable list"]

    official_permutations = tuple(default_seat_permutations(mode))
    reported_permutations = scenario.get("seat_permutations")
    try:
        parsed_permutations = tuple(tuple(row) for row in reported_permutations)
    except (TypeError, ValueError):
        parsed_permutations = ()
    if parsed_permutations != official_permutations:
        issues.append("scenario does not use the official seat permutations")
    if scenario.get("seat_permutation_hash") != OFFICIAL_PERMUTATION_HASHES[mode]:
        issues.append("scenario seat-permutation hash is invalid")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    malformed_rows = 0
    for row in games:
        if not isinstance(row, Mapping):
            malformed_rows += 1
            continue
        deal_id = row.get("deal_id")
        assignment = row.get("assignment")
        if (
            not isinstance(deal_id, str)
            or not deal_id
            or row.get("mode") != mode
            or not isinstance(assignment, list)
        ):
            malformed_rows += 1
            continue
        grouped.setdefault(deal_id, []).append(row)
    if malformed_rows:
        issues.append(f"{malformed_rows} game rows are malformed")

    claimed_deals = scenario.get("num_deals")
    if (
        isinstance(claimed_deals, bool)
        or not isinstance(claimed_deals, int)
        or claimed_deals != len(grouped)
    ):
        issues.append("scenario num_deals does not match unique game-row deal IDs")

    expected_assignments = Counter(official_permutations)
    eligible_rows: list[Mapping[str, Any]] = []
    excluded_deals = 0
    malformed_deals = 0
    for rows in grouped.values():
        try:
            assignments = Counter(tuple(row["assignment"]) for row in rows)
        except TypeError:
            malformed_deals += 1
            continue
        if assignments != expected_assignments:
            malformed_deals += 1
            continue
        forced_smoke = any(
            row.get("max_redeals_exceeded") is True
            or row.get("formal_evaluation_eligible", True) is not True
            for row in rows
        )
        if forced_smoke:
            excluded_deals += 1
            continue
        value_name = "candidate_win" if mode == PROMOTION_MODE else "candidate_score"
        if any(
            isinstance(row.get(value_name), bool)
            or not isinstance(row.get(value_name), (int, float))
            or not math.isfinite(float(row[value_name]))
            for row in rows
        ):
            malformed_deals += 1
            continue
        eligible_rows.extend(rows)
    if malformed_deals:
        issues.append(
            f"{malformed_deals} deals have incomplete, duplicate, or invalid game rows"
        )
    if excluded_deals:
        issues.append(
            f"{excluded_deals} deals exhausted the redeal cap and are smoke-only"
        )

    bootstrap_samples = scenario.get("bootstrap_samples")
    confidence_level = scenario.get("confidence_level")
    deterministic_seed = scenario.get("deterministic_seed")
    statistics_config_valid = (
        isinstance(bootstrap_samples, int)
        and not isinstance(bootstrap_samples, bool)
        and 1 <= bootstrap_samples <= 100_000
        and isinstance(confidence_level, (int, float))
        and not isinstance(confidence_level, bool)
        and 0.0 < float(confidence_level) < 1.0
        and isinstance(deterministic_seed, int)
        and not isinstance(deterministic_seed, bool)
    )
    if not statistics_config_valid:
        issues.append("scenario bootstrap configuration is invalid")

    observations = (
        (
            str(row["deal_id"]),
            float(row["candidate_win"]) - 0.5
            if mode == PROMOTION_MODE else float(row["candidate_score"]),
        )
        for row in eligible_rows
    )
    deal_values = deal_cluster_means(observations)
    recomputed_ci = None
    if deal_values and statistics_config_valid:
        recomputed_ci = paired_bootstrap_ci(
            deal_values,
            confidence_level=float(confidence_level),
            samples=bootstrap_samples,
            seed=deterministic_seed + (0 if mode == PROMOTION_MODE else 1),
        )

    expected_estimator = (
        PROMOTION_ESTIMATOR
        if mode == PROMOTION_MODE else "full_game_zero_sum_seat_score"
    )
    if metrics.get("paired_estimator") != expected_estimator:
        issues.append("reported paired estimator does not match the protocol")
    reported_ci = metrics.get("paired_estimate_ci")
    if recomputed_ci is None:
        if not isinstance(reported_ci, Mapping) or reported_ci.get("paired_deals") != 0:
            issues.append("reported paired-deal count does not match game rows")
        recomputed_payload = {
            "estimate": None,
            "low": None,
            "high": None,
            "confidence_level": confidence_level,
            "paired_deals": 0,
            "bootstrap_samples": bootstrap_samples,
        }
    else:
        recomputed_payload = recomputed_ci.to_dict()
        if not isinstance(reported_ci, Mapping) or any(
            not _numbers_match(reported_ci.get(name), expected)
            for name, expected in recomputed_payload.items()
        ):
            issues.append("reported paired_estimate_ci does not match game rows")

    return {
        "paired_deals": len(deal_values),
        "game_rows": len(games),
        "eligible_game_rows": len(eligible_rows),
        "excluded_deals": excluded_deals,
        "malformed_rows": malformed_rows,
        "malformed_deals": malformed_deals,
        "recomputed_paired_estimate_ci": recomputed_payload,
    }, issues


def result_readiness(result: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Apply P17 policy using evidence recomputed from auditable game rows."""

    scenario = result.get("scenario", {})
    evidence, issues = _recompute_result_evidence(result, mode=mode)
    bootstrap_samples = scenario.get("bootstrap_samples")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < P17_MIN_BOOTSTRAP_SAMPLES
    ):
        issues.append(f"requires >= {P17_MIN_BOOTSTRAP_SAMPLES} bootstrap samples")
    if evidence["paired_deals"] < P17_MIN_PAIRED_DEALS:
        issues.append(f"requires >= {P17_MIN_PAIRED_DEALS} paired deals")
    if mode == "full_game" and isinstance(scenario, Mapping):
        for side in ("candidate", "baseline"):
            bundle = scenario.get(side)
            if not isinstance(bundle, Mapping) or bundle.get("bidding_policy") != "learned":
                issues.append(f"{side} bidding is external, not learned")
    return {
        "protocol": P17_READINESS_PROTOCOL,
        "requirements": {
            "bootstrap_samples": P17_MIN_BOOTSTRAP_SAMPLES,
            "paired_deals": P17_MIN_PAIRED_DEALS,
        },
        "status": "eligible" if not issues else "insufficient",
        "issues": issues,
        "evidence": evidence,
    }


def write_p17_artifacts(
    output_dir: str | Path,
    *,
    matrix: Mapping[str, Any],
    cardplay_result: Mapping[str, Any] | None = None,
    full_game_result: Mapping[str, Any] | None = None,
    ablation_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Write the fixed P17 artifact set, retaining explicit NOT RUN rows."""

    normalized = normalize_matrix(matrix)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ablation_results = dict(ablation_results or {})
    unknown = set(ablation_results) - set(ABLATION_NAMES)
    if unknown:
        raise P17MatrixError(f"unknown ablation results: {sorted(unknown)}")

    def validate_result_inventory(result: Mapping[str, Any] | None, mode: str) -> None:
        if result is None:
            return
        scenario = result.get("scenario")
        if not isinstance(scenario, Mapping):
            raise P17MatrixError(f"{mode} result has no valid scenario identity")
        for side in ("candidate", "baseline"):
            actual_bundle = scenario.get(side)
            if not isinstance(actual_bundle, Mapping):
                raise P17MatrixError(f"{mode} result has no valid {side} bundle")
            name = actual_bundle.get("name")
            if name not in P17_MODEL_NAMES:
                raise P17MatrixError(
                    f"{mode} result {side} {name!r} is absent from the P17 matrix"
                )
            matrix_entry = normalized["models"][name][mode]
            if matrix_entry["status"] != "available":
                raise P17MatrixError(
                    f"{mode} result claims unavailable {side} model {name!r}"
                )
            expected_bundle = matrix_entry["bundle"]
            if dict(actual_bundle) != expected_bundle:
                raise P17MatrixError(
                    f"{mode} result {side} checkpoint/manifest identity does not "
                    f"match the validated P17 matrix entry for {name!r}"
                )

    validate_result_inventory(cardplay_result, "cardplay_only")
    validate_result_inventory(full_game_result, "full_game")
    for name, result in ablation_results.items():
        ablation_entry = normalized["ablations"][name]
        if ablation_entry["status"] != "available":
            raise P17MatrixError(
                f"ablation result {name!r} was supplied for an unavailable matrix row"
            )
        mode = ablation_entry["protocol"]
        validate_result_inventory(result, mode)
        scenario = result["scenario"]
        for side, key in (
            ("candidate", "candidate_model"),
            ("baseline", "baseline_model"),
        ):
            if scenario[side]["name"] != ablation_entry[key]:
                raise P17MatrixError(
                    f"ablation {name!r} {side} does not match its declared {key}"
                )

    def result_block(result: Mapping[str, Any] | None, mode: str) -> dict[str, Any]:
        if result is None:
            return {
                "schema_version": P17_REPORT_SCHEMA_VERSION,
                "status": "not_run",
                "reason": "No compatible result supplied",
                "mode": mode,
                "result": None,
            }
        return {
            "schema_version": P17_REPORT_SCHEMA_VERSION,
            "status": "completed",
            "readiness": result_readiness(result, mode=mode),
            "mode": mode,
            "result": result,
        }

    cardplay_block = result_block(cardplay_result, "cardplay_only")
    full_game_block = result_block(full_game_result, "full_game")
    ablation_block = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": {
            name: (
                {
                    "status": "completed",
                    "result": ablation_results[name],
                }
                if name in ablation_results
                else {
                    "status": "not_run",
                    "reason": normalized["ablations"][name]["reason"],
                    "result": None,
                }
            )
            for name in ABLATION_NAMES
        },
    }
    supplied = [result for result in (cardplay_result, full_game_result) if result]
    calibration = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": [
            {
                "mode": result["scenario"]["mode"],
                "candidate": result["scenario"]["candidate"]["name"],
                "calibration": result["metrics"]["calibration"],
            }
            for result in supplied
        ],
        "status": "completed" if supplied else "not_run",
    }
    latency = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": [
            {
                "mode": result["scenario"]["mode"],
                "candidate": result["scenario"]["candidate"]["name"],
                "inference_latency_ms": result["metrics"]["inference_latency_ms"],
                "inference_calls_per_second": result["metrics"][
                    "inference_calls_per_second"
                ],
                "search": result["metrics"].get("search"),
            }
            for result in supplied
        ],
        "status": "completed" if supplied else "not_run",
    }

    payloads = {
        "model_matrix.json": normalized,
        "cardplay_results.json": cardplay_block,
        "full_game_results.json": full_game_block,
        "ablations.json": ablation_block,
        "calibration.json": calibration,
        "latency.json": latency,
    }
    paths = {}
    for name, payload in payloads.items():
        path = root / name
        path.write_text(
            json.dumps(
                _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        paths[name] = str(path.resolve())
    report_path = root / "report.md"
    report_path.write_text(
        render_p17_markdown(
            normalized, cardplay_block, full_game_block, ablation_block
        ),
        encoding="utf-8",
    )
    paths["report.md"] = str(report_path.resolve())
    return paths


def render_p17_markdown(
    matrix: Mapping[str, Any],
    cardplay: Mapping[str, Any],
    full_game: Mapping[str, Any],
    ablations: Mapping[str, Any],
) -> str:
    lines = [
        "# P17 Evaluation Report",
        "",
        "This report distinguishes executed empirical results from unavailable inputs.",
        "Rule/random bidding smoke runs are never treated as learned-bidding strength.",
        "",
        "## Protocol Status",
        "",
        "| Protocol | Execution | Release eligibility |",
        "| --- | --- | --- |",
    ]
    for block in (cardplay, full_game):
        readiness = block.get("readiness", {"status": "not_run"})
        lines.append(
            f"| {block['mode']} | {block['status']} | {readiness['status']} |"
        )
    lines.extend(["", "## Model Matrix", "", "| Model | Card-play | Full game |", "| --- | --- | --- |"]) 
    for name in P17_MODEL_NAMES:
        lines.append(
            f"| {name} | {matrix['models'][name]['cardplay_only']['status']} | "
            f"{matrix['models'][name]['full_game']['status']} |"
        )
    lines.extend(["", "## Ablations", "", "| Ablation | Status |", "| --- | --- |"]) 
    for name in ABLATION_NAMES:
        lines.append(f"| {name} | {ablations['results'][name]['status']} |")
    lines.extend([
        "",
        "P17 empirical readiness requires complete deal-level clustering, at least "
        "2,000 bootstrap samples, at least 1,000 paired deals, and protocol-compatible "
        "checkpoints. "
        "Full-game release evaluation additionally requires learned bidding on both sides.",
        "",
    ])
    return "\n".join(lines)
