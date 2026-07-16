"""Fail-closed P17 model inventory and empirical-report collation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from douzero.env.rules import RuleSet
from douzero.env.scoring import compute_team_score_magnitude

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
from .scenario import (
    ROLES,
    BundleSpec,
    bundle_from_dict,
    canonical_deal_id,
    canonical_deal_set_id,
    default_seat_permutations,
)
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
                    require_full_git_sha=True,
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
        isinstance(actual, Mapping)
        and set(actual) == set(expected)
        and all(_numbers_match(actual[key], value) for key, value in expected.items())
    )


def _recompute_full_game_seat_to_role(
    row: Mapping[str, Any], ruleset: RuleSet
) -> dict[str, str] | None:
    """Derive final roles from the legal final-attempt bidding transcript."""

    order = row.get("bidding_order")
    history = row.get("bidding_history")
    bid_value = row.get("bid_value")
    if (
        not isinstance(order, list)
        or len(order) != 3
        or any(type(seat) is not str for seat in order)
        or set(order) != {"0", "1", "2"}
        or not isinstance(history, list)
        or not 1 <= len(history) <= 3
        or type(bid_value) is not int
    ):
        return None

    current_max = 0
    landlord_seat = ""
    maximum_bid = max(ruleset.bid_values)
    for index, entry in enumerate(history):
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not int
            or entry[0] != order[index]
            or entry[1] not in ruleset.bid_values
            or current_max == maximum_bid
            or (entry[1] != 0 and entry[1] <= current_max)
        ):
            return None
        if entry[1] > current_max:
            current_max = entry[1]
            landlord_seat = entry[0]
    if (
        current_max == 0
        or bid_value != current_max
        or (len(history) < len(order) and current_max != maximum_bid)
    ):
        return None

    landlord_index = order.index(landlord_seat)
    expected_mapping = {
        landlord_seat: "landlord",
        order[(landlord_index + 1) % 3]: "landlord_down",
        order[(landlord_index + 2) % 3]: "landlord_up",
    }
    reported_mapping = row.get("seat_to_role")
    if not isinstance(reported_mapping, Mapping):
        return None
    try:
        return expected_mapping if dict(reported_mapping) == expected_mapping else None
    except (TypeError, ValueError):
        return None


def _recompute_terminal_outcome(
    row: Mapping[str, Any],
    *,
    mode: str,
    assignment: tuple[str, ...],
    ruleset: RuleSet,
) -> tuple[tuple[str, ...], float, float] | None:
    """Derive candidate outcome only from terminal rule facts."""

    winner_position = row.get("winner_position")
    winner_team = row.get("winner_team")
    if winner_position not in ROLES:
        return None
    expected_winner_team = (
        "landlord" if winner_position == "landlord" else "farmer"
    )
    if winner_team != expected_winner_team:
        return None

    if mode == "cardplay_only":
        expected_roles = tuple(
            role
            for role, label in zip(ROLES, assignment)
            if label == "candidate"
        )
    else:
        seat_to_role = _recompute_full_game_seat_to_role(row, ruleset)
        if seat_to_role is None:
            return None
        expected_roles = tuple(
            str(seat_to_role[str(index)])
            for index, label in enumerate(assignment)
            if label == "candidate"
        )

    candidate_roles = row.get("candidate_roles")
    if (
        not isinstance(candidate_roles, list)
        or tuple(candidate_roles) != expected_roles
    ):
        return None
    candidate_team = "landlord" if expected_roles == ("landlord",) else "farmer"

    bid_value = row.get("bid_value")
    bomb_count = row.get("bomb_count")
    rocket_count = row.get("rocket_count")
    spring = row.get("spring")
    anti_spring = row.get("anti_spring")
    if (
        type(bid_value) is not int
        or type(bomb_count) is not int
        or not 0 <= bomb_count <= 13
        or type(rocket_count) is not int
        or rocket_count not in (0, 1)
        or type(spring) is not bool
        or type(anti_spring) is not bool
        or (spring and anti_spring)
    ):
        return None
    if mode == "cardplay_only":
        if bid_value != 0 or spring or anti_spring:
            return None
    elif (
        bid_value not in ruleset.bid_values
        or bid_value == 0
        or (spring and winner_team != "landlord")
        or (anti_spring and winner_team != "farmer")
    ):
        return None

    score_arguments = {
        "bomb_count": bomb_count,
        "rocket_count": rocket_count,
        "bid_value": bid_value,
        "ruleset": ruleset,
        "spring": spring,
        "anti_spring": anti_spring,
    }
    landlord_magnitude = compute_team_score_magnitude(
        team="landlord", **score_arguments
    )
    farmer_magnitude = compute_team_score_magnitude(
        team="farmer", **score_arguments
    )
    team_scores = {
        "landlord": float(
            landlord_magnitude if winner_team == "landlord" else -landlord_magnitude
        ),
        "farmer": float(
            farmer_magnitude if winner_team == "farmer" else -farmer_magnitude
        ),
    }
    candidate_win = float(winner_team == candidate_team)
    candidate_score = team_scores[candidate_team]
    candidate_log_score = (
        0.0
        if candidate_score == 0.0
        else math.copysign(math.log1p(abs(candidate_score)), candidate_score)
    )
    expected_role_wins = {role: candidate_win for role in expected_roles}
    expected_role_scores = {role: candidate_score for role in expected_roles}
    if (
        not _numbers_match(row.get("candidate_win"), candidate_win)
        or not _numbers_match(row.get("candidate_score"), candidate_score)
        or not _numbers_match(row.get("candidate_log_score"), candidate_log_score)
        or type(row.get("candidate_landlord")) is not int
        or row.get("candidate_landlord") != int(candidate_team == "landlord")
        or not _mapping_numbers_match(row.get("role_wins"), expected_role_wins)
        or not _mapping_numbers_match(row.get("role_scores"), expected_role_scores)
    ):
        return None
    return expected_roles, candidate_win, candidate_score


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

    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    ruleset_identity_valid = scenario.get("ruleset") == ruleset.identity()
    if not ruleset_identity_valid:
        issues.append("scenario does not use the official ruleset identity")

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    deal_hashes_by_index: dict[int, str] = {}
    malformed_rows = 0
    for row in games:
        if not isinstance(row, Mapping):
            malformed_rows += 1
            continue
        deal_id = row.get("deal_id")
        deal_hash = row.get("deal_hash")
        assignment = row.get("assignment")
        if (
            not isinstance(deal_id, str)
            or not _is_full_sha256(deal_hash)
            or row.get("mode") != mode
            or not isinstance(assignment, list)
        ):
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
    recomputed_rows: list[tuple[Mapping[str, Any], float, float]] = []
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
        forced_smoke = any(
            row.get("max_redeals_exceeded") is True
            or row.get("formal_evaluation_eligible", True) is not True
            for row in rows
        )
        if forced_smoke:
            excluded_deals += 1
            continue
        deal_rows: list[tuple[Mapping[str, Any], float, float]] = []
        row_invalid = False
        for row in rows:
            assignment = tuple(row["assignment"])
            outcome = _recompute_terminal_outcome(
                row,
                mode=mode,
                assignment=assignment,
                ruleset=ruleset,
            )
            if outcome is None:
                _append_issue(
                    issues,
                    "reported candidate/role outcome does not match terminal rule evidence",
                )
                row_invalid = True
                break
            _expected_roles, candidate_win, candidate_score = outcome
            deal_rows.append((row, candidate_win, candidate_score))
        if row_invalid:
            malformed_deals += 1
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
            candidate_win - 0.5
            if mode == PROMOTION_MODE else candidate_score,
        )
        for row, candidate_win, candidate_score in recomputed_rows
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

    # Recompute every release-facing diagnostic for which raw row evidence is
    # present. Missing raw evidence is never replaced with a caller summary.
    latencies: list[float] = []
    calibration: list[tuple[str, float, float]] = []
    search_calls = search_timeouts = search_fallbacks = 0
    raw_diagnostics_valid = True
    for row, _win, _score in recomputed_rows:
        raw_latencies = row.get("candidate_latencies_ms")
        raw_calibration = row.get("calibration")
        if not isinstance(raw_latencies, list) or not isinstance(raw_calibration, list):
            raw_diagnostics_valid = False
            continue
        try:
            parsed_latencies = [float(value) for value in raw_latencies]
            parsed_calibration = [
                (str(role), float(prediction), float(label))
                for role, prediction, label in raw_calibration
            ]
            counts = [int(row[name]) for name in (
                "search_calls", "search_timeouts", "search_fallbacks"
            )]
        except (TypeError, ValueError):
            raw_diagnostics_valid = False
            continue
        if (
            not all(math.isfinite(value) and value >= 0 for value in parsed_latencies)
            or not all(
                role in ROLES and math.isfinite(prediction)
                and 0 <= prediction <= 1 and label in (0.0, 1.0)
                for role, prediction, label in parsed_calibration
            )
            or any(value < 0 for value in counts)
            or counts[1] > counts[0]
            or counts[2] > counts[0]
        ):
            raw_diagnostics_valid = False
            continue
        latencies.extend(parsed_latencies)
        calibration.extend(parsed_calibration)
        search_calls += counts[0]
        search_timeouts += counts[1]
        search_fallbacks += counts[2]
    if not raw_diagnostics_valid:
        issues.append("raw latency, calibration, or search evidence is unavailable")

    from .paired import _calibration_metrics
    from .statistics import percentile

    recomputed_diagnostics = {
        "calibration": _json_safe(_calibration_metrics(calibration)),
        "inference_latency_ms": _json_safe({
            "count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        }),
        "search": _json_safe({
            "calls": search_calls,
            "timeout_rate": (
                search_timeouts / search_calls if search_calls else float("nan")
            ),
            "fallback_rate": (
                search_fallbacks / search_calls if search_calls else float("nan")
            ),
        }),
    }
    for name, recomputed in recomputed_diagnostics.items():
        if _json_safe(metrics.get(name)) != recomputed:
            issues.append(f"reported {name} does not match raw game-row evidence")
    candidate_wins = [win for _row, win, _score in recomputed_rows]
    candidate_scores = [score for _row, _win, score in recomputed_rows]
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
            for row, win, score in recomputed_rows
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


def write_p17_artifacts(
    output_dir: str | Path,
    *,
    matrix: Mapping[str, Any],
    cardplay_result: Mapping[str, Any] | None = None,
    full_game_result: Mapping[str, Any] | None = None,
    ablation_results: Mapping[str, Mapping[str, Any]] | None = None,
    expected_evaluator_git_shas: str | Iterable[str] | None = None,
    expected_cardplay_deal_set_id: str | None = None,
    expected_full_game_deal_set_id: str | None = None,
) -> dict[str, str]:
    """Write the fixed P17 artifact set, retaining explicit NOT RUN rows."""

    normalized = normalize_matrix(matrix)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ablation_results = dict(ablation_results or {})
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

    def validate_result_inventory(result: Mapping[str, Any] | None, mode: str) -> None:
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
            )
        except (TypeError, ValueError) as exc:
            raise P17MatrixError(
                f"{mode} result runtime identity is invalid: {exc}"
            ) from exc

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
            "approved_deal_set_id": expected_deal_set_ids[mode],
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
                    "approved_deal_set_id": expected_deal_set_ids[
                        normalized["ablations"][name]["protocol"]
                    ],
                    "readiness": result_readiness(
                        ablation_results[name],
                        mode=normalized["ablations"][name]["protocol"],
                    ),
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
