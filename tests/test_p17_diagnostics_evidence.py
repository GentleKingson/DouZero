"""Replay-complete candidate decision evidence for P17 diagnostics."""

from __future__ import annotations

import copy
import pytest

from douzero.env.rules import RuleSet
from douzero.evaluation.p17 import result_readiness
from douzero.evaluation.paired import _calibration_metrics, evaluate_scenario
from douzero.evaluation.scenario import BundleSpec, EvaluationScenario
from douzero.evaluation.statistics import percentile
from evaluate_paired import generate_deals


def _payload(mode: str, *, seed: int = 2401):
    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    scenario = EvaluationScenario(
        mode=mode,
        ruleset=ruleset,
        candidate=BundleSpec(name="candidate-rule", backend="rule"),
        baseline=BundleSpec(name="baseline-rule", backend="rule"),
        deals=generate_deals(mode, 1, seed, ruleset),
        deterministic_seed=seed,
        bootstrap_samples=2000,
    )
    return scenario, evaluate_scenario(scenario).to_dict()


def _sync_diagnostic_summaries(payload):
    calibration = []
    latencies_ns = []
    search_calls = search_timeouts = search_fallbacks = 0
    for row in payload["games"]:
        decisions = row["candidate_decisions"]
        row["candidate_latencies_ns"] = [
            decision["latency_ns"] for decision in decisions
        ]
        row["candidate_latencies_ms"] = [
            value / 1_000_000.0 for value in row["candidate_latencies_ns"]
        ]
        row["calibration"] = [
            [decision["actor_role"], decision["prediction"]]
            for decision in decisions
            if decision["phase"] == "cardplay"
            and decision["prediction_status"] == "available"
        ]
        row["search_calls"] = sum(
            decision["search_called"] for decision in decisions
        )
        row["search_timeouts"] = sum(
            decision["search_timed_out"] for decision in decisions
        )
        row["search_fallbacks"] = sum(
            decision["search_fallback"] for decision in decisions
        )
        latencies_ns.extend(row["candidate_latencies_ns"])
        search_calls += row["search_calls"]
        search_timeouts += row["search_timeouts"]
        search_fallbacks += row["search_fallbacks"]
        for role, prediction in row["calibration"]:
            label = float(
                row["winner_team"]
                == ("landlord" if role == "landlord" else "farmer")
            )
            calibration.append((role, prediction, label))

    metrics = payload["metrics"]
    latencies_ms = [value / 1_000_000.0 for value in latencies_ns]
    inference_seconds = sum(latencies_ns) / 1_000_000_000.0
    instrumented = (
        len(latencies_ns) / inference_seconds
        if inference_seconds > 0 else float("nan")
    )
    wall_elapsed_ns = metrics["timing_evidence"]["evaluation_wall_elapsed_ns"]
    metrics["calibration"] = _calibration_metrics(calibration)
    metrics["inference_latency_ms"] = {
        "count": len(latencies_ms),
        "p50": percentile(latencies_ms, 0.50),
        "p95": percentile(latencies_ms, 0.95),
        "p99": percentile(latencies_ms, 0.99),
    }
    metrics["inference_calls_per_second"] = instrumented
    metrics["actor_fps"] = instrumented
    metrics["evaluation_wall_calls_per_second"] = (
        len(latencies_ns) / (wall_elapsed_ns / 1_000_000_000.0)
    )
    metrics["search"] = {
        "calls": search_calls,
        "timeouts": search_timeouts,
        "fallbacks": search_fallbacks,
        "timeout_rate": (
            search_timeouts / search_calls if search_calls else float("nan")
        ),
        "fallback_rate": (
            search_fallbacks / search_calls if search_calls else float("nan")
        ),
    }
    metrics["timing_evidence"].update({
        "inference_calls": len(latencies_ns),
        "inference_elapsed_ns": sum(latencies_ns),
    })
    metrics["sample_counts"]["inference_calls"] = len(latencies_ns)
    metrics["sample_counts"]["calibration_decisions"] = len(calibration)


def _make_all_predictions_available(payload, prediction: float = 0.5) -> int:
    count = 0
    for row in payload["games"]:
        for decision in row["candidate_decisions"]:
            if decision["phase"] != "cardplay":
                continue
            if decision["forced_action"]:
                continue
            decision["prediction_status"] = "available"
            decision["prediction"] = prediction
            count += 1
    _sync_diagnostic_summaries(payload)
    return count


@pytest.mark.parametrize("mode", ["cardplay_only", "full_game"])
def test_evaluator_emits_one_ordered_record_for_every_candidate_decision(mode):
    scenario, payload = _payload(mode)
    expected_total = 0
    for row in payload["games"]:
        decisions = row["candidate_decisions"]
        assert decisions
        assert len({decision["decision_id"] for decision in decisions}) == len(
            decisions
        )
        assert [decision["latency_ns"] for decision in decisions] == row[
            "candidate_latencies_ns"
        ]
        assert all(decision["prediction_status"] == "unavailable" for decision in decisions)
        assert all(decision["prediction"] is None for decision in decisions)
        assert all(decision["search_called"] is False for decision in decisions)
        expected_total += len(decisions)

    readiness = result_readiness(
        payload, mode=mode, approved_deals=scenario.deals
    )
    diagnostics = readiness["evidence"]["recomputed_diagnostics"]
    assert diagnostics["status"] == "available"
    assert diagnostics["candidate_decision_count"] == expected_total
    assert diagnostics["calibration_status"] == "unavailable"
    assert any("formal calibration is unavailable" in issue for issue in readiness["issues"])


def test_complete_prediction_cohort_recomputes_available_calibration():
    scenario, payload = _payload("cardplay_only", seed=2402)
    prediction_count = _make_all_predictions_available(payload, prediction=0.75)

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )
    diagnostics = readiness["evidence"]["recomputed_diagnostics"]

    assert prediction_count > 0
    assert diagnostics["status"] == "available"
    assert diagnostics["calibration_status"] == "available"
    assert diagnostics["calibration"]["overall"]["count"] == prediction_count
    assert not any("candidate decision evidence" in issue for issue in readiness["issues"])


def test_deleting_all_decisions_is_rejected_even_if_summaries_are_synchronized():
    scenario, payload = _payload("cardplay_only", seed=2403)
    row = next(row for row in payload["games"] if row["candidate_decisions"])
    row["candidate_decisions"] = []
    _sync_diagnostic_summaries(payload)

    diagnostics = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )["evidence"]["recomputed_diagnostics"]

    assert diagnostics["status"] == "unavailable"


def test_deleting_one_decision_is_rejected_even_if_summaries_are_synchronized():
    scenario, payload = _payload("cardplay_only", seed=2407)
    row = next(row for row in payload["games"] if row["candidate_decisions"])
    row["candidate_decisions"].pop()
    _sync_diagnostic_summaries(payload)

    diagnostics = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )["evidence"]["recomputed_diagnostics"]

    assert diagnostics["status"] == "unavailable"


def test_selectively_removed_prediction_is_rejected_after_summary_rewrite():
    scenario, payload = _payload("cardplay_only", seed=2404)
    assert _make_all_predictions_available(payload) > 1
    decision = next(
        decision
        for row in payload["games"]
        for decision in row["candidate_decisions"]
        if decision["phase"] == "cardplay"
        and not decision["forced_action"]
        and decision["prediction_status"] == "available"
    )
    decision["prediction_status"] = "unavailable"
    decision["prediction"] = None
    _sync_diagnostic_summaries(payload)

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["recomputed_diagnostics"]["status"] == "unavailable"
    assert any("selectively unavailable" in issue for issue in readiness["issues"])


@pytest.mark.parametrize("backend", ["v2", "bc"])
def test_v2_and_bc_require_prediction_on_every_cardplay_decision(backend):
    scenario, payload = _payload("cardplay_only", seed=2405)
    payload["scenario"]["candidate"]["backend"] = backend

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["recomputed_diagnostics"]["status"] == "unavailable"
    assert any("V2/BC candidate" in issue for issue in readiness["issues"])


@pytest.mark.parametrize(
    "mutation", ["duplicate", "reorder", "wrong_role", "forced", "search"]
)
def test_structural_or_search_forgery_cannot_survive_synced_summaries(mutation):
    scenario, payload = _payload("cardplay_only", seed=2406)
    row = next(row for row in payload["games"] if len(row["candidate_decisions"]) > 1)
    decisions = row["candidate_decisions"]
    if mutation == "duplicate":
        decisions.append(copy.deepcopy(decisions[0]))
    elif mutation == "reorder":
        decisions[0], decisions[1] = decisions[1], decisions[0]
    elif mutation == "wrong_role":
        decision = next(item for item in decisions if item["phase"] == "cardplay")
        decision["actor_role"] = (
            "landlord_up" if decision["actor_role"] != "landlord_up" else "landlord"
        )
    elif mutation == "forced":
        decision = next(item for item in decisions if item["phase"] == "cardplay")
        decision["forced_action"] = not decision["forced_action"]
    else:
        decision = next(item for item in decisions if item["phase"] == "cardplay")
        decision["search_called"] = True
        decision["search_timed_out"] = True
    _sync_diagnostic_summaries(payload)

    diagnostics = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )["evidence"]["recomputed_diagnostics"]

    assert diagnostics["status"] == "unavailable"


def test_search_enabled_candidate_skips_only_replayed_forced_actions():
    scenario, payload = _payload("cardplay_only", seed=2408)
    candidate = payload["scenario"]["candidate"]
    candidate["backend"] = "v2"
    candidate["search_config"] = {"enabled": True}
    forced = searched = 0
    for row in payload["games"]:
        for decision in row["candidate_decisions"]:
            if decision["phase"] != "cardplay":
                continue
            if decision["forced_action"]:
                forced += 1
                continue
            searched += 1
            decision["prediction_status"] = "available"
            decision["prediction"] = 0.5
            decision["search_called"] = True
    assert forced > 0
    assert searched > 0
    _sync_diagnostic_summaries(payload)

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    diagnostics = readiness["evidence"]["recomputed_diagnostics"]
    assert diagnostics["status"] == "available"
    assert diagnostics["calibration_status"] == "available"
    assert diagnostics["search"]["calls"] == searched


@pytest.mark.parametrize(
    "metric",
    [
        "sample_counts",
        "formal_evaluation",
        "smoke_descriptive",
        "team_win_percentage",
        "paired_win_rate_delta_ci",
        "descriptive_win_rate_delta_ci",
        "mean_raw_score",
        "mean_log_score",
        "zero_sum_seat_score",
        "paired_mean_score_ci",
        "bid_rate",
        "landlord_acquisition_rate",
        "by_bid_value",
        "redeals",
        "bomb_rate",
        "rocket_rate",
        "spring_rate",
        "anti_spring_rate",
        "mean_game_length",
        "belief",
    ],
)
def test_every_redundant_metric_must_match_replay_aggregate(metric):
    scenario, payload = _payload("full_game", seed=2410)
    baseline = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    assert not any(
        "exactly match replay-derived aggregates" in issue
        for issue in baseline["issues"]
    )

    payload["metrics"][metric] = {"forged": True}
    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )

    assert any(
        "exactly match replay-derived aggregates" in issue
        for issue in readiness["issues"]
    )


def test_metric_and_role_ci_extra_fields_are_rejected():
    scenario, payload = _payload("cardplay_only", seed=2411)
    payload["metrics"]["paired_estimate_ci"]["forged"] = True
    payload["metrics"]["by_role"]["landlord"]["win_rate_delta_ci"][
        "forged"
    ] = True

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert any(
        "exactly match replay-derived aggregates" in issue
        for issue in readiness["issues"]
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("sample_counts", "deals"), True),
        (("sample_counts", "excluded_games"), False),
    ],
)
def test_metric_deep_compare_does_not_coerce_booleans_to_integers(
    path, replacement
):
    scenario, payload = _payload("full_game", seed=2416)
    payload["metrics"][path[0]][path[1]] = replacement

    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )

    assert any(
        "exactly match replay-derived aggregates" in issue
        for issue in readiness["issues"]
    )


def test_wall_clock_cannot_be_shorter_than_sequential_inference_samples():
    scenario, payload = _payload("cardplay_only", seed=2412)
    for row in payload["games"]:
        for decision in row["candidate_decisions"]:
            decision["latency_ns"] = 1
    payload["metrics"]["timing_evidence"]["evaluation_wall_elapsed_ns"] = 1
    _sync_diagnostic_summaries(payload)

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["recomputed_diagnostics"]["status"] == (
        "unavailable"
    )
    assert any("timing evidence" in issue for issue in readiness["issues"])


def test_search_timeout_must_record_the_producer_fallback():
    scenario, payload = _payload("cardplay_only", seed=2413)
    payload["scenario"]["candidate"]["backend"] = "v2"
    payload["scenario"]["candidate"]["search_config"] = {"enabled": True}
    target = None
    for row in payload["games"]:
        for decision in row["candidate_decisions"]:
            if decision["phase"] == "cardplay" and not decision["forced_action"]:
                decision["prediction_status"] = "available"
                decision["prediction"] = 0.5
                decision["search_called"] = True
                target = target or decision
    assert target is not None
    target["search_timed_out"] = True
    target["search_fallback"] = False
    _sync_diagnostic_summaries(payload)

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["recomputed_diagnostics"]["status"] == (
        "unavailable"
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("bid_value", False),
        ("spring", 0),
        ("leg_id", "forged-leg"),
        ("unexpected_field", "forged"),
    ],
)
def test_game_row_schema_and_redundant_types_are_exact(mutation, value):
    scenario, payload = _payload("cardplay_only", seed=2414)
    payload["games"][0][mutation] = value

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["malformed_rows"] > 0 or any(
        "trace does not replay" in issue for issue in readiness["issues"]
    )


def test_forced_action_requires_a_json_boolean():
    scenario, payload = _payload("cardplay_only", seed=2415)
    decision = next(
        decision
        for row in payload["games"]
        for decision in row["candidate_decisions"]
        if decision["phase"] == "cardplay"
    )
    decision["forced_action"] = int(decision["forced_action"])
    _sync_diagnostic_summaries(payload)

    diagnostics = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )["evidence"]["recomputed_diagnostics"]

    assert diagnostics["status"] == "unavailable"


@pytest.mark.parametrize(
    "field", ["candidate_win", "candidate_score", "candidate_log_score"]
)
def test_candidate_outcome_redundancy_requires_json_float(field):
    scenario, payload = _payload("cardplay_only", seed=2417)
    row = payload["games"][0]
    row[field] = int(row[field])

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert any("trace does not replay" in issue for issue in readiness["issues"])


def test_redundant_latency_nanoseconds_require_json_integers():
    scenario, payload = _payload("cardplay_only", seed=2419)
    row = next(row for row in payload["games"] if row["candidate_latencies_ns"])
    row["candidate_latencies_ns"][0] = float(row["candidate_latencies_ns"][0])

    diagnostics = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )["evidence"]["recomputed_diagnostics"]

    assert diagnostics["status"] == "unavailable"


def test_role_outcome_redundancy_requires_json_float():
    scenario, payload = _payload("cardplay_only", seed=2418)
    row = payload["games"][0]
    role = next(iter(row["role_wins"]))
    row["role_wins"][role] = int(row["role_wins"][role])

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )

    assert any("trace does not replay" in issue for issue in readiness["issues"])
