"""Fail-closed P17 model inventory and empirical-report collation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from douzero.env.rules import RuleSet

from .ablation import ABLATION_NAMES
from .protocol import (
    EVALUATION_PROTOCOL,
    OFFICIAL_PERMUTATION_HASHES,
    OFFICIAL_CI_METHOD,
    OFFICIAL_CONFIDENCE_LEVEL,
    OFFICIAL_STATISTICAL_UNIT,
    P17_MIN_BOOTSTRAP_SAMPLES,
    P17_MIN_PAIRED_DEALS,
    P17_READINESS_PROTOCOL,
    PROMOTION_ESTIMATOR,
    PROMOTION_MODE,
)
from .replay import REDEAL_CAP_EXCLUSION_REASON, replay_game_record
from .scenario import (
    ROLES,
    BundleSpec,
    bundle_from_dict,
    canonical_deal_hash,
    canonical_deal_id,
    canonical_deal_set_id,
    default_seat_permutations,
)
from .statistics import deal_cluster_means, paired_bootstrap_ci


P17_MATRIX_SCHEMA_VERSION = "p17-model-matrix-v1"
P17_REPORT_SCHEMA_VERSION = "p17-evaluation-report-v2"
P17_ARTIFACT_MANIFEST_SCHEMA_VERSION = "p17-artifact-manifest-v1"
P17_PRIVATE_RESULT_SCHEMA_VERSION = "p17-private-result-projection-v1"
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
    from douzero.evaluation.checkpoint_inputs import load_verified_checkpoint

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
        seen: set[tuple[str, str]] = set()

        def validate_v2(path: str, digest: str, label: str) -> None:
            checkpoint_identity = (path, digest)
            if checkpoint_identity in seen:
                return
            seen.add(checkpoint_identity)
            try:
                def load_and_validate(verified_path: str) -> None:
                    state_dict, _ = load_v2_position_weights(
                        verified_path,
                        expected_schema_hash=schema.stable_hash(),
                        expected_model_config_hash=model_config.stable_hash(),
                        expected_ruleset=ruleset,
                        runtime_model_config=model_config,
                        training_device="cpu",
                    )
                    ModelV2(schema, model_config).load_state_dict(
                        state_dict, strict=True
                    )

                load_verified_checkpoint(
                    path,
                    digest,
                    load_and_validate,
                    label=f"{model_name}.{protocol}.{label}",
                )
            except Exception as exc:
                fail(label, exc)

        for role, path in bundle.checkpoints.items():
            validate_v2(path, bundle.checkpoint_sha256[role], role)
        if bundle.bidding_checkpoint:
            validate_v2(
                bundle.bidding_checkpoint,
                bundle.bidding_checkpoint_sha256,
                "bidding",
            )
        if bundle.belief_checkpoint:
            try:
                from douzero.belief.checkpoint import load_belief_checkpoint

                load_verified_checkpoint(
                    bundle.belief_checkpoint,
                    bundle.belief_checkpoint_sha256,
                    lambda verified_path: load_belief_checkpoint(
                        verified_path,
                        expected_ruleset=ruleset,
                        expected_feature_version="v2",
                        require_full_git_sha=True,
                    ),
                    label=f"{model_name}.{protocol}.belief",
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
                load_verified_checkpoint(
                    path,
                    bundle.checkpoint_sha256[role],
                    lambda verified_path: load_position_state_dict_strict(
                        verified_path, runtime_model.state_dict()
                    ),
                    label=f"{model_name}.{protocol}.{role}",
                )
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


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare redundant JSON evidence without Python bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return bool(
            set(actual) == set(expected)
            and all(
                _strict_json_equal(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return bool(
            len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(actual, expected)
            )
        )
    return actual == expected


_DEAL_ID_PATTERN = re.compile(r"^(?P<index>[0-9]{6,12})-[0-9a-f]{12}$")


def _is_full_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_official_confidence_level(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(
            float(value),
            OFFICIAL_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _append_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def _mapping_numbers_match(
    actual: object, expected: Mapping[str, float]
) -> bool:
    return bool(
        type(actual) is dict
        and set(actual) == set(expected)
        and all(
            type(actual[key]) is type(value)
            and _numbers_match(actual[key], value)
            for key, value in expected.items()
        )
    )


def _recompute_terminal_outcome(
    row: Mapping[str, Any],
    *,
    mode: str,
    assignment: tuple[str, ...],
    replay: Any,
) -> tuple[tuple[str, ...], float, float, str, bool] | None:
    """Derive every terminal result from an engine-replayed approved deal."""

    terminal_fields = {
        "winner_position": replay.winner_position,
        "winner_team": replay.winner_team,
        "bid_value": replay.bid_value,
        "bomb_count": replay.bomb_count,
        "rocket_count": replay.rocket_count,
        "spring": replay.spring,
        "anti_spring": replay.anti_spring,
        "game_length": replay.game_length,
        "redeal_count": replay.redeal_count,
    }
    if any(
        not _strict_json_equal(row.get(name), expected)
        for name, expected in terminal_fields.items()
    ):
        return None
    forced_smoke = replay.max_redeals_exceeded
    formal_eligible = not forced_smoke
    if (
        type(row.get("redeal_count")) is not int
        or row.get("max_redeals_exceeded") is not forced_smoke
        or row.get("formal_evaluation_eligible") is not formal_eligible
        or row.get("exclusion_reason")
        != (REDEAL_CAP_EXCLUSION_REASON if forced_smoke else None)
    ):
        return None

    if mode == "cardplay_only":
        expected_roles = tuple(
            role
            for role, label in zip(ROLES, assignment)
            if label == "candidate"
        )
        if (
            row.get("seat_to_role") is not None
            or row.get("bidding_order") != []
            or row.get("bidding_history") != []
            or row.get("bidding_trace") != []
            or type(row.get("candidate_bid_attempts")) is not int
            or row.get("candidate_bid_attempts") != 0
            or type(row.get("candidate_positive_bids")) is not int
            or row.get("candidate_positive_bids") != 0
            or type(row.get("bidding_inference_calls")) is not int
            or row.get("bidding_inference_calls") != 0
        ):
            return None
    else:
        if (
            not isinstance(replay.seat_to_role, Mapping)
            or not _strict_json_equal(
                row.get("seat_to_role"), dict(replay.seat_to_role)
            )
            or not _strict_json_equal(
                row.get("bidding_order"), list(replay.bidding_order)
            )
            or not _strict_json_equal(
                row.get("bidding_history"),
                [list(entry) for entry in replay.bidding_history],
            )
        ):
            return None
        expected_roles = tuple(
            str(replay.seat_to_role[str(index)])
            for index, label in enumerate(assignment)
            if label == "candidate"
        )
        bidding_trace = row.get("bidding_trace")
        if not isinstance(bidding_trace, list):
            return None
        candidate_bids = [
            bid
            for attempt in bidding_trace
            for seat, bid in attempt
            if assignment[int(seat)] == "candidate"
        ]
        if (
            type(row.get("candidate_bid_attempts")) is not int
            or row.get("candidate_bid_attempts") != len(candidate_bids)
            or type(row.get("candidate_positive_bids")) is not int
            or row.get("candidate_positive_bids")
            != sum(bid > 0 for bid in candidate_bids)
            or type(row.get("bidding_inference_calls")) is not int
            or row.get("bidding_inference_calls") != len(candidate_bids)
        ):
            return None

    candidate_roles = row.get("candidate_roles")
    if (
        not isinstance(candidate_roles, list)
        or tuple(candidate_roles) != expected_roles
    ):
        return None
    candidate_team = "landlord" if expected_roles == ("landlord",) else "farmer"
    candidate_win = float(replay.winner_team == candidate_team)
    candidate_score = float(replay.team_scores[candidate_team])
    candidate_log_score = (
        0.0
        if candidate_score == 0.0
        else math.copysign(math.log1p(abs(candidate_score)), candidate_score)
    )
    expected_role_wins = {role: candidate_win for role in expected_roles}
    expected_role_scores = {role: candidate_score for role in expected_roles}
    if (
        type(row.get("candidate_win")) is not float
        or not _numbers_match(row.get("candidate_win"), candidate_win)
        or type(row.get("candidate_score")) is not float
        or not _numbers_match(row.get("candidate_score"), candidate_score)
        or type(row.get("candidate_log_score")) is not float
        or not _numbers_match(row.get("candidate_log_score"), candidate_log_score)
        or type(row.get("candidate_landlord")) is not int
        or row.get("candidate_landlord") != int(candidate_team == "landlord")
        or not _mapping_numbers_match(row.get("role_wins"), expected_role_wins)
        or not _mapping_numbers_match(row.get("role_scores"), expected_role_scores)
    ):
        return None
    return (
        expected_roles,
        candidate_win,
        candidate_score,
        replay.winner_team,
        forced_smoke,
    )


_DECISION_EVIDENCE_FIELDS = {
    "decision_id",
    "phase",
    "actor_role",
    "actor_seat",
    "latency_ns",
    "prediction_status",
    "prediction",
    "forced_action",
    "search_called",
    "search_timed_out",
    "search_fallback",
}

_GAME_RECORD_FIELDS = {
    "deal_id",
    "leg_id",
    "mode",
    "assignment",
    "candidate_roles",
    "candidate_win",
    "candidate_score",
    "candidate_log_score",
    "role_wins",
    "role_scores",
    "winner_team",
    "winner_position",
    "bid_value",
    "candidate_bid_attempts",
    "candidate_positive_bids",
    "candidate_landlord",
    "bomb_count",
    "rocket_count",
    "spring",
    "anti_spring",
    "game_length",
    "redeal_count",
    "max_redeals_exceeded",
    "candidate_latencies_ms",
    "calibration",
    "search_calls",
    "search_timeouts",
    "search_fallbacks",
    "bidding_inference_calls",
    "formal_evaluation_eligible",
    "exclusion_reason",
    "deal_hash",
    "seat_to_role",
    "bidding_order",
    "bidding_history",
    "candidate_latencies_ns",
    "bidding_trace",
    "cardplay_trace",
    "trace_digest",
    "candidate_decisions",
}


def _expected_candidate_decisions(
    row: Mapping[str, Any],
    *,
    mode: str,
    assignment: tuple[str, ...],
    replay: Any,
) -> list[dict[str, Any]]:
    """Rebuild the candidate decision sequence from verified replay inputs."""

    from .paired import _candidate_decision_id

    deal_id = str(row["deal_id"])
    assignment3 = (assignment[0], assignment[1], assignment[2])
    expected: list[dict[str, Any]] = []
    if mode == "full_game":
        for attempt_index, attempt in enumerate(row["bidding_trace"]):
            for action_index, (seat, _bid) in enumerate(attempt):
                if assignment[int(seat)] != "candidate":
                    continue
                expected.append({
                    "decision_id": _candidate_decision_id(
                        mode,
                        deal_id,
                        assignment3,
                        "bidding",
                        attempt_index,
                        action_index,
                    ),
                    "phase": "bidding",
                    "actor_role": None,
                    "actor_seat": str(seat),
                    "forced_action": False,
                })

    role_to_seat: dict[str, str] = {}
    if mode == "full_game":
        if not isinstance(replay.seat_to_role, Mapping):
            raise ValueError("full-game replay has no seat-to-role mapping")
        role_to_seat = {
            str(role): str(seat) for seat, role in replay.seat_to_role.items()
        }
    legal_action_counts = replay.cardplay_legal_action_counts
    if len(legal_action_counts) != len(row["cardplay_trace"]):
        raise ValueError("replay legal-action evidence is incomplete")
    for action_index, (role, _action) in enumerate(row["cardplay_trace"]):
        if mode == "cardplay_only":
            is_candidate = assignment[ROLES.index(role)] == "candidate"
            actor_seat = None
        else:
            actor_seat = role_to_seat[role]
            is_candidate = assignment[int(actor_seat)] == "candidate"
        if not is_candidate:
            continue
        expected.append({
            "decision_id": _candidate_decision_id(
                mode,
                deal_id,
                assignment3,
                "cardplay",
                action_index,
            ),
            "phase": "cardplay",
            "actor_role": str(role),
            "actor_seat": actor_seat,
            "forced_action": legal_action_counts[action_index] == 1,
        })
    return expected


def _parse_candidate_decisions(
    row: Mapping[str, Any],
    expected: list[dict[str, Any]],
    *,
    winner_team: str,
    candidate_search_enabled: bool,
) -> tuple[
    list[int],
    list[tuple[str, float, float]],
    tuple[int, int, int],
    int,
    int,
    int,
]:
    """Validate one complete candidate cohort and legacy summary projections."""

    raw = row.get("candidate_decisions")
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ValueError("candidate decision count does not match replay")
    ids = [item.get("decision_id") for item in raw if isinstance(item, Mapping)]
    if (
        len(ids) != len(raw)
        or any(not isinstance(decision_id, str) for decision_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("candidate decision IDs are missing or duplicated")

    latencies_ns: list[int] = []
    calibration: list[tuple[str, float, float]] = []
    search_calls = search_timeouts = search_fallbacks = 0
    cardplay_decisions = 0
    calibration_opportunities = 0
    prediction_count = 0
    for item, descriptor in zip(raw, expected):
        if not isinstance(item, Mapping) or set(item) != _DECISION_EVIDENCE_FIELDS:
            raise ValueError("candidate decision evidence has the wrong fields")
        if type(item.get("forced_action")) is not bool:
            raise ValueError("candidate decision forced_action must be a boolean")
        if any(
            not _strict_json_equal(item.get(name), value)
            for name, value in descriptor.items()
        ):
            raise ValueError("candidate decision identity does not match replay")
        latency_ns = item.get("latency_ns")
        if type(latency_ns) is not int or latency_ns < 0:
            raise ValueError("candidate decision latency_ns is invalid")
        prediction_status = item.get("prediction_status")
        prediction = item.get("prediction")
        if prediction_status == "available":
            if (
                type(prediction) is not float
                or not math.isfinite(float(prediction))
                or not 0.0 <= float(prediction) <= 1.0
            ):
                raise ValueError("candidate decision prediction is invalid")
            prediction_value: float | None = float(prediction)
        elif prediction_status == "unavailable" and prediction is None:
            prediction_value = None
        else:
            raise ValueError("candidate decision prediction status is invalid")

        search_called = item.get("search_called")
        search_timed_out = item.get("search_timed_out")
        search_fallback = item.get("search_fallback")
        if (
            type(search_called) is not bool
            or type(search_timed_out) is not bool
            or type(search_fallback) is not bool
            or (not search_called and (search_timed_out or search_fallback))
            or (search_timed_out and not search_fallback)
        ):
            raise ValueError("candidate decision search evidence is invalid")
        expected_search_called = bool(
            descriptor["phase"] == "cardplay"
            and candidate_search_enabled
            and not descriptor["forced_action"]
        )
        if search_called is not expected_search_called:
            raise ValueError("candidate decision search usage contradicts the bundle")
        if descriptor["phase"] == "bidding" and prediction_value is not None:
            raise ValueError("bidding decisions cannot supply p_win calibration")
        if descriptor["forced_action"] and prediction_value is not None:
            raise ValueError("forced card-play decisions cannot supply calibration")

        latencies_ns.append(latency_ns)
        search_calls += int(search_called)
        search_timeouts += int(search_timed_out)
        search_fallbacks += int(search_fallback)
        if descriptor["phase"] == "cardplay":
            cardplay_decisions += 1
            if not descriptor["forced_action"]:
                calibration_opportunities += 1
            if prediction_value is not None:
                prediction_count += 1
                role = str(descriptor["actor_role"])
                label = float(
                    winner_team
                    == ("landlord" if role == "landlord" else "farmer")
                )
                calibration.append((role, prediction_value, label))

    raw_latencies_ns = row.get("candidate_latencies_ns")
    raw_latencies_ms = row.get("candidate_latencies_ms")
    if not _strict_json_equal(
        raw_latencies_ns, latencies_ns
    ) or not isinstance(raw_latencies_ms, list):
        raise ValueError("legacy candidate latency fields do not match decisions")
    if len(raw_latencies_ms) != len(latencies_ns) or any(
        type(milliseconds) is not float
        or not math.isfinite(float(milliseconds))
        or not math.isclose(
            float(milliseconds), nanoseconds / 1_000_000.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for nanoseconds, milliseconds in zip(latencies_ns, raw_latencies_ms)
    ):
        raise ValueError("legacy millisecond latencies do not match latency_ns")
    expected_calibration = [
        [role, prediction] for role, prediction, _label in calibration
    ]
    if not _strict_json_equal(row.get("calibration"), expected_calibration):
        raise ValueError("legacy calibration rows do not match candidate decisions")
    expected_search = (search_calls, search_timeouts, search_fallbacks)
    raw_search = tuple(
        row.get(name)
        for name in ("search_calls", "search_timeouts", "search_fallbacks")
    )
    if any(type(value) is not int for value in raw_search) or raw_search != expected_search:
        raise ValueError("legacy search summaries do not match candidate decisions")
    return (
        latencies_ns,
        calibration,
        expected_search,
        cardplay_decisions,
        calibration_opportunities,
        prediction_count,
    )


def _recompute_result_evidence(
    result: Mapping[str, Any],
    *,
    mode: str,
    approved_deals: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild release evidence by replaying traces against approved deals."""

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

    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    ruleset_identity_valid = scenario.get("ruleset") == ruleset.identity()
    if not ruleset_identity_valid:
        issues.append("scenario does not use the official ruleset identity")

    approved_deals_by_index: dict[int, Mapping[str, Any]] = {}
    approved_deal_hashes: list[str] = []
    if approved_deals is None:
        issues.append("approved deal payloads are required for deterministic replay")
    else:
        try:
            approved_deal_hashes = [
                canonical_deal_hash(deal) for deal in approved_deals
            ]
            if len(set(approved_deal_hashes)) != len(approved_deal_hashes):
                raise ValueError("duplicate approved deals")
            approved_deals_by_index = {
                index: deal for index, deal in enumerate(approved_deals)
            }
            approved_set_id = canonical_deal_set_id(
                mode,
                ruleset,
                approved_deal_hashes,
                seat_permutation_hash=OFFICIAL_PERMUTATION_HASHES[mode],
            )
            if scenario.get("deal_set_id") != approved_set_id:
                issues.append(
                    "scenario deal_set_id does not match approved deal payloads"
                )
        except (KeyError, TypeError, ValueError):
            approved_deals_by_index = {}
            approved_deal_hashes = []
            issues.append("approved deal payloads are malformed")

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    deal_hashes_by_index: dict[int, str] = {}
    malformed_rows = 0
    for row in games:
        if not isinstance(row, Mapping) or set(row) != _GAME_RECORD_FIELDS:
            malformed_rows += 1
            continue
        deal_id = row.get("deal_id")
        deal_hash = row.get("deal_hash")
        assignment = row.get("assignment")
        try:
            assignment_tuple = tuple(assignment)
        except TypeError:
            assignment_tuple = ()
        if (
            not isinstance(deal_id, str)
            or not _is_full_sha256(deal_hash)
            or row.get("mode") != mode
            or not isinstance(assignment, list)
            or assignment_tuple not in official_permutations
        ):
            malformed_rows += 1
            continue
        leg_index = official_permutations.index(assignment_tuple)
        expected_leg_id = (
            f"cardplay-{leg_index}"
            if mode == "cardplay_only"
            else f"full-game-{leg_index}"
        )
        if type(row.get("leg_id")) is not str or row.get("leg_id") != expected_leg_id:
            malformed_rows += 1
            continue
        match = _DEAL_ID_PATTERN.fullmatch(deal_id)
        if match is None:
            malformed_rows += 1
            continue
        deal_index = int(match.group("index"))
        if deal_id != canonical_deal_id(deal_index, str(deal_hash)):
            malformed_rows += 1
            continue
        if approved_deal_hashes and (
            deal_index >= len(approved_deal_hashes)
            or deal_hash != approved_deal_hashes[deal_index]
        ):
            malformed_rows += 1
            continue
        prior_hash = deal_hashes_by_index.setdefault(deal_index, str(deal_hash))
        if prior_hash != deal_hash:
            malformed_rows += 1
            continue
        grouped.setdefault(deal_index, []).append(row)
    if malformed_rows:
        issues.append(f"{malformed_rows} game rows are malformed")

    claimed_deals = scenario.get("num_deals")
    if (
        isinstance(claimed_deals, bool)
        or not isinstance(claimed_deals, int)
        or claimed_deals < 1
        or claimed_deals != len(grouped)
    ):
        issues.append("scenario num_deals does not match unique game-row deal IDs")

    deal_indices_valid = set(grouped) == set(range(len(grouped)))
    if not deal_indices_valid:
        issues.append("game-row deal indices are not contiguous from zero")

    if ruleset_identity_valid and deal_hashes_by_index and deal_indices_valid:
        try:
            recomputed_deal_set_id = canonical_deal_set_id(
                mode,
                ruleset,
                [deal_hashes_by_index[index] for index in range(len(grouped))],
                seat_permutation_hash=OFFICIAL_PERMUTATION_HASHES[mode],
            )
        except (TypeError, ValueError):
            recomputed_deal_set_id = ""
            issues.append("game-row deal hashes are invalid or duplicated")
        if scenario.get("deal_set_id") != recomputed_deal_set_id:
            issues.append("scenario deal_set_id does not match canonical game-row deals")
    else:
        recomputed_deal_set_id = ""
        issues.append("ordered full game-row deal hashes are unavailable")

    expected_assignments = Counter(official_permutations)
    eligible_rows: list[Mapping[str, Any]] = []
    excluded_deals = 0
    malformed_deals = 0
    recomputed_rows: list[
        tuple[
            Mapping[str, Any],
            float,
            float,
            str,
            tuple[str, ...],
            list[dict[str, Any]],
        ]
    ] = []
    for deal_index in sorted(grouped):
        rows = grouped[deal_index]
        try:
            assignments = Counter(tuple(row["assignment"]) for row in rows)
        except TypeError:
            malformed_deals += 1
            continue
        if assignments != expected_assignments:
            malformed_deals += 1
            continue
        approved_deal = approved_deals_by_index.get(deal_index)
        if approved_deal is None:
            malformed_deals += 1
            continue
        deal_rows: list[
            tuple[
                Mapping[str, Any],
                float,
                float,
                str,
                tuple[str, ...],
                list[dict[str, Any]],
            ]
        ] = []
        exclusion_states: list[bool] = []
        row_invalid = False
        for row in rows:
            assignment = tuple(row["assignment"])
            try:
                replay = replay_game_record(
                    row,
                    approved_deal,
                    mode=mode,
                    ruleset=ruleset,
                    deterministic_seed=int(scenario.get("deterministic_seed", -1)),
                )
                outcome = _recompute_terminal_outcome(
                    row,
                    mode=mode,
                    assignment=assignment,
                    replay=replay,
                )
                expected_decisions = _expected_candidate_decisions(
                    row,
                    mode=mode,
                    assignment=assignment,
                    replay=replay,
                )
            except (KeyError, TypeError, ValueError):
                outcome = None
            if outcome is None:
                _append_issue(
                    issues,
                    "game trace does not replay to the reported terminal outcome",
                )
                row_invalid = True
                break
            (
                expected_roles,
                candidate_win,
                candidate_score,
                winner_team,
                forced_smoke,
            ) = outcome
            deal_rows.append(
                (
                    row,
                    candidate_win,
                    candidate_score,
                    winner_team,
                    expected_roles,
                    expected_decisions,
                )
            )
            exclusion_states.append(forced_smoke)
        if row_invalid:
            malformed_deals += 1
            continue
        if len(set(exclusion_states)) != 1:
            _append_issue(
                issues,
                "deal seat rotations disagree on replayed redeal-cap exclusion",
            )
            malformed_deals += 1
            continue
        if exclusion_states[0]:
            excluded_deals += 1
            continue
        eligible_rows.extend(rows)
        recomputed_rows.extend(deal_rows)
    if malformed_deals:
        _append_issue(
            issues,
            f"{malformed_deals} deals have incomplete, duplicate, or invalid game rows"
        )
    if excluded_deals:
        issues.append(
            f"{excluded_deals} deals exhausted the redeal cap and are smoke-only"
        )

    bootstrap_samples = scenario.get("bootstrap_samples")
    confidence_level = scenario.get("confidence_level")
    deterministic_seed = scenario.get("deterministic_seed")
    bootstrap_config_valid = (
        isinstance(bootstrap_samples, int)
        and not isinstance(bootstrap_samples, bool)
        and 1 <= bootstrap_samples <= 100_000
        and isinstance(deterministic_seed, int)
        and not isinstance(deterministic_seed, bool)
    )
    confidence_level_in_range = (
        isinstance(confidence_level, (int, float))
        and not isinstance(confidence_level, bool)
        and math.isfinite(float(confidence_level))
        and 0.0 < float(confidence_level) < 1.0
    )
    if not bootstrap_config_valid or not confidence_level_in_range:
        issues.append("scenario bootstrap configuration is invalid")
    if not _is_official_confidence_level(confidence_level):
        _append_issue(
            issues, f"requires confidence_level={OFFICIAL_CONFIDENCE_LEVEL}"
        )
    if scenario.get("statistical_unit") != OFFICIAL_STATISTICAL_UNIT:
        issues.append(f"requires statistical_unit={OFFICIAL_STATISTICAL_UNIT!r}")
    if scenario.get("ci_method") != OFFICIAL_CI_METHOD:
        issues.append(f"requires ci_method={OFFICIAL_CI_METHOD!r}")
    if scenario.get("release_protocol_id") != EVALUATION_PROTOCOL:
        issues.append(f"requires release_protocol_id={EVALUATION_PROTOCOL!r}")

    observations = (
        (
            str(row["deal_id"]),
            candidate_win - 0.5 if mode == PROMOTION_MODE else candidate_score,
        )
        for (
            row,
            candidate_win,
            candidate_score,
            _winner,
            _roles,
            _decisions,
        ) in recomputed_rows
    )
    deal_values = deal_cluster_means(observations)
    recomputed_ci = None
    if deal_values and bootstrap_config_valid:
        recomputed_ci = paired_bootstrap_ci(
            deal_values,
            confidence_level=OFFICIAL_CONFIDENCE_LEVEL,
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
            "confidence_level": OFFICIAL_CONFIDENCE_LEVEL,
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

    # Every candidate action in the replayed traces must have exactly one
    # structured decision record. Legacy arrays/counters are redundant and are
    # accepted only when they exactly project from this complete cohort.
    latencies_ns: list[int] = []
    calibration: list[tuple[str, float, float]] = []
    search_calls = search_timeouts = search_fallbacks = 0
    cardplay_decisions = calibration_opportunities = prediction_count = 0
    expected_decision_count = 0
    diagnostic_cohort_valid = True
    trusted_games: list[Any] = []
    candidate = scenario.get("candidate")
    candidate_backend = (
        candidate.get("backend") if isinstance(candidate, Mapping) else None
    )
    candidate_search_config = (
        candidate.get("search_config") if isinstance(candidate, Mapping) else None
    )
    candidate_search_enabled = bool(
        candidate_backend in {"v2", "bc"}
        and isinstance(candidate_search_config, Mapping)
        and candidate_search_config.get("enabled") is True
    )
    from .paired import GameRecord

    for (
        row,
        _win,
        _score,
        winner_team,
        _expected_roles,
        expected_decisions,
    ) in recomputed_rows:
        expected_decision_count += len(expected_decisions)
        try:
            (
                row_latencies,
                row_calibration,
                row_search,
                row_cardplay_decisions,
                row_calibration_opportunities,
                row_prediction_count,
            ) = _parse_candidate_decisions(
                row,
                expected_decisions,
                winner_team=winner_team,
                candidate_search_enabled=candidate_search_enabled,
            )
        except (KeyError, TypeError, ValueError) as exc:
            diagnostic_cohort_valid = False
            _append_issue(issues, f"candidate decision evidence is invalid: {exc}")
            continue
        latencies_ns.extend(row_latencies)
        calibration.extend(row_calibration)
        search_calls += row_search[0]
        search_timeouts += row_search[1]
        search_fallbacks += row_search[2]
        cardplay_decisions += row_cardplay_decisions
        calibration_opportunities += row_calibration_opportunities
        prediction_count += row_prediction_count
        trusted_games.append(GameRecord(
            deal_id=str(row["deal_id"]),
            leg_id=str(row["leg_id"]),
            mode=mode,
            assignment=tuple(row["assignment"]),
            candidate_roles=tuple(_expected_roles),
            candidate_win=float(_win),
            candidate_score=float(_score),
            role_wins={role: float(_win) for role in _expected_roles},
            role_scores={role: float(_score) for role in _expected_roles},
            winner_team=winner_team,
            winner_position=str(row["winner_position"]),
            bid_value=int(row["bid_value"]),
            candidate_bid_attempts=int(row["candidate_bid_attempts"]),
            candidate_positive_bids=int(row["candidate_positive_bids"]),
            candidate_landlord=int(row["candidate_landlord"]),
            bomb_count=int(row["bomb_count"]),
            rocket_count=int(row["rocket_count"]),
            spring=bool(row["spring"]),
            anti_spring=bool(row["anti_spring"]),
            game_length=int(row["game_length"]),
            candidate_latencies_ms=tuple(
                value / 1_000_000.0 for value in row_latencies
            ),
            calibration=tuple(
                (role, prediction)
                for role, prediction, _label in row_calibration
            ),
            redeal_count=int(row["redeal_count"]),
            max_redeals_exceeded=False,
            search_calls=row_search[0],
            search_timeouts=row_search[1],
            search_fallbacks=row_search[2],
            bidding_inference_calls=int(row["bidding_inference_calls"]),
            formal_evaluation_eligible=True,
            exclusion_reason="",
            deal_hash=str(row["deal_hash"]),
            candidate_latencies_ns=tuple(row_latencies),
        ))

    if len(latencies_ns) != expected_decision_count:
        diagnostic_cohort_valid = False
    if not diagnostic_cohort_valid:
        _append_issue(
            issues,
            "candidate decision evidence is incomplete, duplicated, reordered, "
            "or inconsistent with replay",
        )

    if 0 < prediction_count < calibration_opportunities:
        diagnostic_cohort_valid = False
        _append_issue(
            issues,
            "candidate card-play prediction evidence is selectively unavailable",
        )
    if (
        candidate_backend in {"v2", "bc"}
        and prediction_count != calibration_opportunities
    ):
        diagnostic_cohort_valid = False
        _append_issue(
            issues,
            "V2/BC candidate requires a prediction for every non-forced card-play decision",
        )

    from .paired import _calibration_metrics
    from .statistics import percentile

    timing = metrics.get("timing_evidence")
    wall_elapsed_ns = (
        timing.get("evaluation_wall_elapsed_ns")
        if isinstance(timing, Mapping) else None
    )
    expected_timing = {
        "clock": "time.perf_counter_ns",
        "scope": "candidate_bidding_and_cardplay_inference_calls_only",
        "inference_calls": len(latencies_ns),
        "inference_elapsed_ns": sum(latencies_ns),
        "evaluation_wall_elapsed_ns": wall_elapsed_ns,
    }
    if diagnostic_cohort_valid and (
        not isinstance(timing, Mapping)
        or type(wall_elapsed_ns) is not int
        or wall_elapsed_ns <= 0
        or wall_elapsed_ns < sum(latencies_ns)
        or not _strict_json_equal(dict(timing), expected_timing)
    ):
        diagnostic_cohort_valid = False
        _append_issue(issues, "timing evidence is unavailable or inconsistent")

    calibration_available = bool(
        diagnostic_cohort_valid
        and calibration_opportunities > 0
        and prediction_count == calibration_opportunities
    )
    if diagnostic_cohort_valid and not calibration_available:
        _append_issue(
            issues,
            "formal calibration is unavailable without complete card-play predictions",
        )

    if diagnostic_cohort_valid:
        latencies = [value / 1_000_000.0 for value in latencies_ns]
        elapsed_seconds = sum(latencies_ns) / 1_000_000_000.0
        instrumented_throughput = (
            len(latencies_ns) / elapsed_seconds if elapsed_seconds > 0 else None
        )
        evaluation_wall_throughput = (
            len(latencies_ns) / (wall_elapsed_ns / 1_000_000_000.0)
            if wall_elapsed_ns > 0 else None
        )
        calibration_payload = _json_safe(_calibration_metrics(calibration))
        latency_payload = _json_safe({
            "count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        })
        search_payload = _json_safe({
            "calls": search_calls,
            "timeouts": search_timeouts,
            "fallbacks": search_fallbacks,
            "timeout_rate": (
                search_timeouts / search_calls
                if search_calls else float("nan")
            ),
            "fallback_rate": (
                search_fallbacks / search_calls
                if search_calls else float("nan")
            ),
        })
        recomputed_diagnostics = {
            "status": "available",
            "calibration_status": (
                "available" if calibration_available else "unavailable"
            ),
            "timing_status": "available",
            "candidate_decision_count": len(latencies_ns),
            "candidate_cardplay_decision_count": cardplay_decisions,
            "calibration_opportunity_count": calibration_opportunities,
            "calibration": calibration_payload if calibration_available else None,
            "inference_latency_ms": latency_payload,
            "instrumented_inference_calls_per_second": instrumented_throughput,
            "evaluation_wall_calls_per_second": evaluation_wall_throughput,
            "search": search_payload,
            "timing_evidence": expected_timing,
        }
        reported_payloads = {
            "calibration": calibration_payload,
            "inference_latency_ms": latency_payload,
            "search": search_payload,
        }
        for name, expected in reported_payloads.items():
            if not _strict_json_equal(_json_safe(metrics.get(name)), expected):
                issues.append(
                    f"reported {name} does not match candidate decision evidence"
                )
        if not _strict_json_equal(
            _json_safe(metrics.get("inference_calls_per_second")),
            _json_safe(instrumented_throughput),
        ):
            issues.append(
                "reported inference_calls_per_second does not match timing evidence"
            )
        if not _strict_json_equal(
            _json_safe(metrics.get("actor_fps")),
            _json_safe(instrumented_throughput),
        ):
            issues.append("reported actor_fps does not match timing evidence")
        if not _strict_json_equal(
            _json_safe(metrics.get("evaluation_wall_calls_per_second")),
            _json_safe(evaluation_wall_throughput),
        ):
            issues.append(
                "reported evaluation_wall_calls_per_second does not match "
                "timing evidence"
            )

        # Close the remainder of the aggregate schema over replay-derived
        # GameRecord values. This covers every redundant rate/count/CI and
        # rejects unknown metric keys instead of publishing unchecked fields.
        aggregate_inputs_valid = bool(
            bootstrap_config_valid
            and confidence_level_in_range
            and malformed_rows == 0
            and malformed_deals == 0
            and excluded_deals == 0
            and len(trusted_games) == len(games)
            and approved_deals is not None
        )
        if aggregate_inputs_valid:
            from types import SimpleNamespace

            from .paired import _aggregate

            candidate_config = scenario.get("candidate")
            baseline_config = scenario.get("baseline")
            aggregate_scenario = SimpleNamespace(
                confidence_level=float(confidence_level),
                bootstrap_samples=bootstrap_samples,
                deterministic_seed=deterministic_seed,
                mode=mode,
                deals=tuple(approved_deals),
                seat_permutations=official_permutations,
                candidate=SimpleNamespace(
                    belief_checkpoint=(
                        bool(candidate_config.get("belief_checkpoint"))
                        if isinstance(candidate_config, Mapping)
                        else False
                    )
                ),
                baseline=SimpleNamespace(
                    belief_checkpoint=(
                        bool(baseline_config.get("belief_checkpoint"))
                        if isinstance(baseline_config, Mapping)
                        else False
                    )
                ),
            )
            expected_metrics = _aggregate(aggregate_scenario, trusted_games)
            expected_metrics["timing_evidence"][
                "evaluation_wall_elapsed_ns"
            ] = wall_elapsed_ns
            expected_metrics["evaluation_wall_calls_per_second"] = (
                len(latencies_ns) / (wall_elapsed_ns / 1_000_000_000.0)
            )
            if not _strict_json_equal(
                _json_safe(metrics), _json_safe(expected_metrics)
            ):
                issues.append(
                    "reported metrics do not exactly match replay-derived aggregates"
                )
    else:
        recomputed_diagnostics = {
            "status": "unavailable",
            "calibration_status": "unavailable",
            "timing_status": "unavailable",
            "reason": "complete strict per-decision diagnostic evidence is unavailable",
            "candidate_decision_count": None,
            "candidate_cardplay_decision_count": None,
            "calibration_opportunity_count": None,
            "calibration": None,
            "inference_latency_ms": None,
            "instrumented_inference_calls_per_second": None,
            "evaluation_wall_calls_per_second": None,
            "search": None,
            "timing_evidence": None,
        }

    candidate_wins = [
        win
        for _row, win, _score, _winner, _roles, _decisions in recomputed_rows
    ]
    candidate_scores = [
        score
        for _row, _win, score, _winner, _roles, _decisions in recomputed_rows
    ]
    recomputed_overall = (
        sum(candidate_wins) / len(candidate_wins) if candidate_wins else None
    )
    recomputed_mean_score = (
        sum(candidate_scores) / len(candidate_scores) if candidate_scores else None
    )
    if not _numbers_match(metrics.get("overall_win_percentage"), recomputed_overall or 0.0):
        issues.append("reported overall_win_percentage does not match raw outcomes")
    if not _numbers_match(metrics.get("mean_score"), recomputed_mean_score or 0.0):
        issues.append("reported mean_score does not match raw outcomes")
    reported_by_role = metrics.get("by_role")
    for role in ROLES:
        role_rows = [
            (win, score)
            for row, win, score, _winner, _roles, _decisions in recomputed_rows
            if role in row["candidate_roles"]
        ]
        reported_role = (
            reported_by_role.get(role, {})
            if isinstance(reported_by_role, Mapping) else {}
        )
        if not role_rows:
            if reported_role.get("games") != 0:
                issues.append(f"reported by_role.{role} games do not match raw outcomes")
            continue
        expected_role = {
            "games": len(role_rows),
            "win_percentage": sum(win for win, _ in role_rows) / len(role_rows),
            "mean_score": sum(score for _, score in role_rows) / len(role_rows),
        }
        if any(
            not _numbers_match(reported_role.get(name), expected)
            for name, expected in expected_role.items()
        ):
            issues.append(f"reported by_role.{role} does not match raw outcomes")

    return {
        "paired_deals": len(deal_values),
        "game_rows": len(games),
        "eligible_game_rows": len(eligible_rows),
        "excluded_deals": excluded_deals,
        "malformed_rows": malformed_rows,
        "malformed_deals": malformed_deals,
        "recomputed_paired_estimate_ci": recomputed_payload,
        "recomputed_deal_set_id": recomputed_deal_set_id,
        "recomputed_diagnostics": recomputed_diagnostics,
    }, issues


def _result_readiness_impl(
    result: Mapping[str, Any],
    *,
    mode: str,
    approved_deals: Sequence[Mapping[str, Any]] | None = None,
    provenance_verified: bool,
) -> dict[str, Any]:
    """Build readiness after the caller has established the trust boundary."""

    scenario = result.get("scenario", {})
    evidence, issues = _recompute_result_evidence(
        result, mode=mode, approved_deals=approved_deals
    )
    evidence["provenance_verified"] = provenance_verified
    if not provenance_verified:
        issues.append(
            "requires a verified detached GitHub OIDC/Sigstore attestation"
        )
    bootstrap_samples = scenario.get("bootstrap_samples")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < P17_MIN_BOOTSTRAP_SAMPLES
    ):
        issues.append(f"requires >= {P17_MIN_BOOTSTRAP_SAMPLES} bootstrap samples")
    if not _is_official_confidence_level(scenario.get("confidence_level")):
        _append_issue(
            issues, f"requires confidence_level={OFFICIAL_CONFIDENCE_LEVEL}"
        )
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
            "confidence_level": OFFICIAL_CONFIDENCE_LEVEL,
        },
        "status": "eligible" if not issues else "insufficient",
        "issues": issues,
        "evidence": evidence,
    }


def result_readiness(
    result: Mapping[str, Any],
    *,
    mode: str,
    approved_deals: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return descriptive readiness for an in-memory, untrusted result.

    Formal provenance is deliberately unavailable through this public helper.
    The release gate is :func:`write_p17_artifacts`, which accepts artifact
    paths plus an attestation policy and performs verification internally.
    """

    return _result_readiness_impl(
        result,
        mode=mode,
        approved_deals=approved_deals,
        provenance_verified=False,
    )


def write_p17_artifacts(
    output_dir: str | Path,
    *,
    matrix: Mapping[str, Any],
    cardplay_result: Any | None = None,
    full_game_result: Any | None = None,
    ablation_results: Mapping[str, Any] | None = None,
    expected_evaluator_git_shas: str | Iterable[str] | None = None,
    expected_cardplay_deal_set_id: str | None = None,
    expected_full_game_deal_set_id: str | None = None,
    approved_cardplay_deals: Sequence[Mapping[str, Any]] | None = None,
    approved_full_game_deals: Sequence[Mapping[str, Any]] | None = None,
    allow_unverified_results: bool = False,
) -> dict[str, str]:
    """Write P17 artifacts from replayed, detached-attested result files.

    Raw mappings are rejected by default. ``allow_unverified_results`` exists
    only for local descriptive tooling; such results are always insufficient.
    """

    from .provenance import (
        AttestedEvaluationInput,
        ProvenanceError,
        VerifiedEvaluationResult,
        verify_github_attested_result,
    )

    formal_result_requested = any(
        isinstance(value, AttestedEvaluationInput)
        for value in (
            cardplay_result,
            full_game_result,
            *((ablation_results or {}).values()),
        )
    )
    if formal_result_requested:
        from .checkpoint_inputs import (
            CheckpointIdentityError,
            require_explicit_matrix_checkpoint_digests,
        )

        try:
            require_explicit_matrix_checkpoint_digests(matrix, kind="p17")
        except CheckpointIdentityError as exc:
            raise P17MatrixError(
                f"formal P17 matrix checkpoint identity is invalid: {exc}"
            ) from exc

    def unwrap_result(value: Any | None) -> tuple[Mapping[str, Any] | None, Any | None]:
        if value is None:
            return None, None
        if isinstance(value, AttestedEvaluationInput):
            try:
                verification = verify_github_attested_result(
                    value.result_path,
                    value.bundle_path,
                    value.policy,
                )
            except ProvenanceError as exc:
                raise P17MatrixError(
                    f"formal result attestation verification failed: {exc}"
                ) from exc
            return verification.result, verification
        if isinstance(value, VerifiedEvaluationResult):
            raise P17MatrixError(
                "formal P17 collation must verify artifact paths internally"
            )
        if isinstance(value, Mapping) and allow_unverified_results:
            return value, None
        if isinstance(value, Mapping):
            raise P17MatrixError(
                "formal P17 collation requires a verified detached attestation"
            )
        raise P17MatrixError("evaluation result has an unsupported trust wrapper")

    cardplay_result, cardplay_verification = unwrap_result(cardplay_result)
    full_game_result, full_game_verification = unwrap_result(full_game_result)
    raw_ablation_results = dict(ablation_results or {})
    ablation_results = {}
    ablation_verifications: dict[str, Any | None] = {}
    for name, value in raw_ablation_results.items():
        payload, verification = unwrap_result(value)
        assert payload is not None
        ablation_results[name] = payload
        ablation_verifications[name] = verification

    normalized = normalize_matrix(matrix)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    unknown = set(ablation_results) - set(ABLATION_NAMES)
    if unknown:
        raise P17MatrixError(f"unknown ablation results: {sorted(unknown)}")
    approved_evaluator_shas = (
        (expected_evaluator_git_shas,)
        if isinstance(expected_evaluator_git_shas, str)
        else tuple(expected_evaluator_git_shas or ())
    )
    has_results = any(
        result is not None
        for result in (cardplay_result, full_game_result, *ablation_results.values())
    )
    if has_results and not approved_evaluator_shas:
        raise P17MatrixError(
            "completed results require an expected evaluator Git SHA allowlist"
        )
    expected_deal_set_ids = {
        "cardplay_only": expected_cardplay_deal_set_id,
        "full_game": expected_full_game_deal_set_id,
    }
    approved_deals_by_mode = {
        "cardplay_only": approved_cardplay_deals,
        "full_game": approved_full_game_deals,
    }
    required_deal_set_modes = {
        mode
        for result, mode in (
            (cardplay_result, "cardplay_only"),
            (full_game_result, "full_game"),
        )
        if result is not None
    }
    required_deal_set_modes.update(
        normalized["ablations"][name]["protocol"]
        for name in ablation_results
    )
    for mode in required_deal_set_modes:
        if not _is_full_sha256(expected_deal_set_ids[mode]):
            raise P17MatrixError(
                f"completed {mode} results require an approved deal_set_id"
            )
        approved_deals = approved_deals_by_mode[mode]
        if not isinstance(approved_deals, Sequence) or isinstance(
            approved_deals, (str, bytes)
        ) or not approved_deals:
            raise P17MatrixError(
                f"completed {mode} results require approved deal payloads for replay"
            )
        try:
            replay_set_id = canonical_deal_set_id(
                mode,
                RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard(),
                [canonical_deal_hash(deal) for deal in approved_deals],
                seat_permutation_hash=OFFICIAL_PERMUTATION_HASHES[mode],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise P17MatrixError(
                f"approved {mode} deal payloads are invalid: {exc}"
            ) from exc
        if replay_set_id != expected_deal_set_ids[mode]:
            raise P17MatrixError(
                f"approved {mode} deal payloads do not match approved deal_set_id"
            )

    def validate_result_inventory(
        result: Mapping[str, Any] | None,
        mode: str,
        verification: Any | None,
    ) -> None:
        if result is None:
            return
        scenario = result.get("scenario")
        if not isinstance(scenario, Mapping):
            raise P17MatrixError(f"{mode} result has no valid scenario identity")
        if scenario.get("deal_set_id") != expected_deal_set_ids[mode]:
            raise P17MatrixError(
                f"{mode} result deal_set_id does not match the approved evaluation set"
            )
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
        from .paired import validate_evaluation_runtime_identity

        try:
            validate_evaluation_runtime_identity(
                result,
                expected_mode=mode,
                expected_source_git_shas=approved_evaluator_shas,
                require_formal_source=verification is not None,
            )
        except (TypeError, ValueError) as exc:
            raise P17MatrixError(
                f"{mode} result runtime identity is invalid: {exc}"
            ) from exc

        if verification is not None and (
            verification.source_git_sha != result["runtime_identity"]["source_git_sha"]
            or verification.result_digest
            != result["result_integrity"]["result_digest"]
        ):
            raise P17MatrixError(
                f"{mode} detached attestation does not match result provenance"
            )
        if verification is not None:
            execution = result["runtime_identity"].get("execution_environment")
            if not isinstance(execution, Mapping) or (
                execution.get("repository") != verification.repository
                or str(execution.get("workflow_ref", "")).split("@", 1)[0]
                != verification.signer_workflow
                or execution.get("workflow_sha") != verification.signer_digest
                or execution.get("source_ref") != verification.source_ref
                or execution.get("run_url") != verification.workflow_run_url
                or execution.get("runner_environment")
                != verification.runner_environment
            ):
                raise P17MatrixError(
                    f"{mode} signed workflow run does not match runtime identity"
                )

    validate_result_inventory(
        cardplay_result, "cardplay_only", cardplay_verification
    )
    validate_result_inventory(full_game_result, "full_game", full_game_verification)
    for name, result in ablation_results.items():
        ablation_entry = normalized["ablations"][name]
        if ablation_entry["status"] != "available":
            raise P17MatrixError(
                f"ablation result {name!r} was supplied for an unavailable matrix row"
            )
        mode = ablation_entry["protocol"]
        validate_result_inventory(result, mode, ablation_verifications[name])
        scenario = result["scenario"]
        for side, key in (
            ("candidate", "candidate_model"),
            ("baseline", "baseline_model"),
        ):
            if scenario[side]["name"] != ablation_entry[key]:
                raise P17MatrixError(
                    f"ablation {name!r} {side} does not match its declared {key}"
                )

    def verification_summary(verification: Any | None) -> dict[str, Any]:
        if verification is None:
            return {"status": "unverified_local"}
        return {
            "status": "verified",
            "repository": verification.repository,
            "source_ref": verification.source_ref,
            "source_git_sha": verification.source_git_sha,
            "signer_workflow": verification.signer_workflow,
            "signer_digest": verification.signer_digest,
            "workflow_run_url": verification.workflow_run_url,
            "runner_environment": verification.runner_environment,
            "result_digest": verification.result_digest,
            "artifact_sha256": verification.artifact_sha256,
            "attestation_count": len(
                verification.attestation_verifications
            ),
        }

    def publishable_result(
        result: Mapping[str, Any], readiness: Mapping[str, Any]
    ) -> dict[str, Any]:
        scenario = result.get("scenario", {})
        if not (
            isinstance(scenario, Mapping)
            and scenario.get("dataset_scope") == "private_holdout"
        ):
            return copy.deepcopy(dict(result))

        # Private traces and per-decision values form a high-bandwidth oracle:
        # a model could encode holdout cards in predictions, latency, or an
        # otherwise unknown nested field. Build a positive allowlist rather
        # than attempting to redact an open-ended signed result object.
        candidate = scenario.get("candidate")
        baseline = scenario.get("baseline")
        evidence = readiness.get("evidence")
        if (
            not isinstance(candidate, Mapping)
            or not isinstance(baseline, Mapping)
            or not isinstance(evidence, Mapping)
        ):
            raise P17MatrixError("private result cannot be projected safely")
        return {
            "schema_version": P17_PRIVATE_RESULT_SCHEMA_VERSION,
            "scenario": {
                "mode": scenario.get("mode"),
                "dataset_scope": "private_holdout",
                "deal_set_id": scenario.get("deal_set_id"),
                "num_deals": scenario.get("num_deals"),
                "candidate": {"name": candidate.get("name")},
                "baseline": {"name": baseline.get("name")},
            },
            "game_evidence": {
                "status": "redacted",
                "published_game_rows": 0,
                "source_game_rows": evidence.get("game_rows"),
            },
        }

    def result_block(
        result: Mapping[str, Any] | None,
        mode: str,
        verification: Any | None,
    ) -> dict[str, Any]:
        if result is None:
            return {
                "schema_version": P17_REPORT_SCHEMA_VERSION,
                "status": "not_run",
                "reason": "No compatible result supplied",
                "mode": mode,
                "result": None,
            }
        readiness = _result_readiness_impl(
            result,
            mode=mode,
            approved_deals=approved_deals_by_mode[mode],
            provenance_verified=verification is not None,
        )
        return {
            "schema_version": P17_REPORT_SCHEMA_VERSION,
            "status": readiness["status"],
            "execution_status": "completed",
            "approved_deal_set_id": expected_deal_set_ids[mode],
            "provenance": verification_summary(verification),
            "readiness": readiness,
            "mode": mode,
            "result": publishable_result(result, readiness),
        }

    cardplay_block = result_block(
        cardplay_result, "cardplay_only", cardplay_verification
    )
    full_game_block = result_block(
        full_game_result, "full_game", full_game_verification
    )
    ablation_result_blocks: dict[str, dict[str, Any]] = {}
    for name in ABLATION_NAMES:
        if name not in ablation_results:
            ablation_result_blocks[name] = {
                "status": "not_run",
                "reason": normalized["ablations"][name]["reason"],
                "result": None,
            }
            continue
        mode = normalized["ablations"][name]["protocol"]
        block = result_block(
            ablation_results[name], mode, ablation_verifications[name]
        )
        block.pop("mode", None)
        ablation_result_blocks[name] = block
    ablation_block = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": ablation_result_blocks,
    }
    supplied_blocks = [
        block for block in (cardplay_block, full_game_block)
        if block["status"] != "not_run"
    ]

    def diagnostic_results(name: str) -> list[dict[str, Any]]:
        rows = []
        for block in supplied_blocks:
            result = block["result"]
            evidence = block["readiness"]["evidence"]["recomputed_diagnostics"]
            rows.append({
                "mode": block["mode"],
                "candidate": result["scenario"]["candidate"]["name"],
                "status": evidence.get(
                    f"{name}_status", evidence.get("status", "unavailable")
                ),
                name: evidence.get(name),
            })
        return rows

    aggregate_status = (
        "not_run"
        if not supplied_blocks
        else (
            "eligible"
            if all(block["status"] == "eligible" for block in supplied_blocks)
            else "insufficient"
        )
    )
    calibration = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": diagnostic_results("calibration"),
        "status": aggregate_status,
    }
    latency = {
        "schema_version": P17_REPORT_SCHEMA_VERSION,
        "results": [],
        "status": aggregate_status,
    }
    for block in supplied_blocks:
        result = block["result"]
        evidence = block["readiness"]["evidence"]["recomputed_diagnostics"]
        latency["results"].append({
            "mode": block["mode"],
            "candidate": result["scenario"]["candidate"]["name"],
            "status": evidence.get("timing_status", "unavailable"),
            "inference_latency_ms": evidence.get("inference_latency_ms"),
            "instrumented_inference_calls_per_second": evidence.get(
                "instrumented_inference_calls_per_second"
            ),
            "evaluation_wall_calls_per_second": evidence.get(
                "evaluation_wall_calls_per_second"
            ),
            "timing_evidence": evidence.get("timing_evidence"),
            "search": evidence.get("search"),
        })

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
    def artifact_identity(path: str) -> dict[str, Any]:
        payload = Path(path).read_bytes()
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    manifest_payload = {
        "schema_version": P17_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "report_schema_version": P17_REPORT_SCHEMA_VERSION,
        "files": {
            name: artifact_identity(path)
            for name, path in sorted(paths.items())
        },
        "release_status": {
            "cardplay_only": cardplay_block["status"],
            "full_game": full_game_block["status"],
            "ablations": {
                name: block["status"]
                for name, block in sorted(ablation_result_blocks.items())
            },
        },
        "source_result_provenance": {
            "cardplay_only": cardplay_block.get("provenance"),
            "full_game": full_game_block.get("provenance"),
            "ablations": {
                name: block.get("provenance")
                for name, block in sorted(ablation_result_blocks.items())
                if block.get("provenance") is not None
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    paths["manifest.json"] = str(manifest_path.resolve())
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
        execution_status = block.get("execution_status", block["status"])
        lines.append(
            f"| {block['mode']} | {execution_status} | {readiness['status']} |"
        )
    lines.extend([
        "", "## Model Matrix", "", "| Model | Card-play | Full game |",
        "| --- | --- | --- |",
    ])
    for name in P17_MODEL_NAMES:
        lines.append(
            f"| {name} | {matrix['models'][name]['cardplay_only']['status']} | "
            f"{matrix['models'][name]['full_game']['status']} |"
        )
    lines.extend([
        "", "## Ablations", "", "| Ablation | Status |", "| --- | --- |",
    ])
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
