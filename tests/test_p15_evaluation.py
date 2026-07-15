"""P15 paired evaluation, clustered confidence intervals, and reports."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from douzero.env.rules import RuleSet
from douzero.evaluation.ablation import ABLATION_NAMES, AblationRunner
from douzero.evaluation.gates import RegressionGateConfig, evaluate_regression_gates
from douzero.evaluation.paired import evaluate_scenario
from douzero.evaluation.reporting import write_report
from douzero.evaluation.scenario import BundleSpec, EvaluationScenario
from douzero.evaluation.statistics import deal_cluster_means, paired_bootstrap_ci
from evaluate_paired import _load_matrix, generate_deals


def _scenario(mode: str, *, seed: int = 17, deals: int = 2):
    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    return EvaluationScenario(
        mode=mode,
        ruleset=ruleset,
        candidate=BundleSpec(name="candidate-rule", backend="rule"),
        baseline=BundleSpec(name="baseline-random", backend="random"),
        deals=generate_deals(mode, deals, seed, ruleset),
        deterministic_seed=seed,
        bootstrap_samples=100,
    )


def test_deal_cluster_bootstrap_does_not_treat_legs_as_independent():
    clustered = deal_cluster_means([
        ("deal-a", 0.5),
        ("deal-a", -0.5),
        ("deal-b", 0.5),
        ("deal-b", 0.5),
    ])
    assert clustered == {"deal-a": 0.0, "deal-b": 0.5}
    first = paired_bootstrap_ci(clustered, samples=200, seed=9)
    second = paired_bootstrap_ci(clustered, samples=200, seed=9)
    assert first == second
    assert first.paired_deals == 2
    assert first.estimate == pytest.approx(0.25)


def test_scenario_default_permutations_are_balanced_and_mode_specific():
    cardplay = _scenario("cardplay_only", deals=1)
    assert cardplay.seat_permutations == (
        ("candidate", "baseline", "baseline"),
        ("baseline", "candidate", "candidate"),
    )
    full = _scenario("full_game", deals=1)
    assert len(full.seat_permutations) == 3
    assert all(permutation.count("candidate") == 1 for permutation in full.seat_permutations)


def test_scenario_rejects_incomplete_or_repeated_official_permutations():
    base = _scenario("full_game", deals=1)
    with pytest.raises(ValueError, match="official.*seat permutations"):
        replace(base, seat_permutations=(base.seat_permutations[0],))
    with pytest.raises(ValueError, match="official.*seat permutations"):
        replace(
            base,
            seat_permutations=(
                base.seat_permutations[0],
                base.seat_permutations[0],
                base.seat_permutations[2],
            ),
        )


def test_cardplay_evaluation_is_reproducible_and_paired_by_deal():
    scenario = _scenario("cardplay_only")
    first = evaluate_scenario(scenario)
    second = evaluate_scenario(scenario)
    # Ignore timing, which is measured rather than fabricated.
    assert [game.candidate_win for game in first.games] == [
        game.candidate_win for game in second.games
    ]
    assert [game.candidate_score for game in first.games] == [
        game.candidate_score for game in second.games
    ]
    assert first.metrics["paired_win_rate_delta_ci"] == second.metrics["paired_win_rate_delta_ci"]
    assert first.metrics["sample_counts"]["deals"] == 2
    assert first.metrics["sample_counts"]["games"] == 4
    assert first.metrics["paired_win_rate_delta_ci"]["paired_deals"] == 2
    assert {game.mode for game in first.games} == {"cardplay_only"}
    with pytest.raises(ValueError, match="at least 1000 bootstrap"):
        first.to_promotion_evaluation()
    official = evaluate_scenario(replace(scenario, bootstrap_samples=1000))
    promotion = official.to_promotion_evaluation()
    assert promotion.evaluator_protocol == "p15_paired_v1"
    assert promotion.paired_games == 2
    assert promotion.mode == "cardplay_only"
    assert promotion.confidence_level == 0.95
    assert promotion.bootstrap_samples == 1000


def test_cardplay_identical_policies_have_zero_promotion_estimate():
    ruleset = RuleSet.legacy()
    scenario = EvaluationScenario(
        mode="cardplay_only",
        ruleset=ruleset,
        candidate=BundleSpec(name="same-candidate", backend="random"),
        baseline=BundleSpec(name="same-baseline", backend="random"),
        deals=generate_deals("cardplay_only", 3, 72, ruleset),
        deterministic_seed=72,
        bootstrap_samples=100,
    )
    result = evaluate_scenario(scenario)
    assert result.metrics["paired_estimate_ci"]["estimate"] == pytest.approx(0.0)


def test_full_game_rotates_candidate_and_reports_rules_metrics():
    scenario = _scenario("full_game", deals=1)
    result = evaluate_scenario(scenario)
    repeated = evaluate_scenario(scenario)
    assert len(result.games) == 3
    assert {game.assignment for game in result.games} == {
        ("candidate", "baseline", "baseline"),
        ("baseline", "candidate", "baseline"),
        ("baseline", "baseline", "candidate"),
    }
    assert result.metrics["sample_counts"]["deals"] == 1
    assert 0.0 <= result.metrics["bid_rate"] <= 1.0
    assert 0.0 <= result.metrics["landlord_acquisition_rate"] <= 1.0
    assert all(game.bid_value >= 1 for game in result.games)
    assert {game.mode for game in result.games} == {"full_game"}
    assert [game.to_dict() | {"candidate_latencies_ms": []} for game in result.games] == [
        game.to_dict() | {"candidate_latencies_ms": []} for game in repeated.games
    ]
    assert result.metrics["paired_win_rate_delta_ci"] is None
    with pytest.raises(ValueError, match="cannot be used for promotion"):
        result.to_promotion_evaluation()


def test_full_game_identical_policies_have_zero_paired_estimate():
    ruleset = RuleSet.standard()
    candidate = BundleSpec(
        name="same-candidate",
        backend="random",
        bidding_policy="random",
    )
    baseline = BundleSpec(
        name="same-baseline",
        backend="random",
        bidding_policy="random",
    )
    scenario = EvaluationScenario(
        mode="full_game",
        ruleset=ruleset,
        candidate=candidate,
        baseline=baseline,
        deals=generate_deals("full_game", 3, 91, ruleset),
        deterministic_seed=91,
        bootstrap_samples=100,
    )
    result = evaluate_scenario(scenario)
    assert result.metrics["paired_estimator"] == "full_game_zero_sum_seat_score"
    assert result.metrics["paired_estimate_ci"]["estimate"] == pytest.approx(0.0)


def test_full_game_all_pass_is_bounded_and_deterministic():
    ruleset = RuleSet.standard()
    scenario = EvaluationScenario(
        mode="full_game",
        ruleset=ruleset,
        candidate=BundleSpec(
            name="candidate-pass", backend="rule", bidding_policy="pass"
        ),
        baseline=BundleSpec(
            name="baseline-pass", backend="rule", bidding_policy="pass"
        ),
        deals=generate_deals("full_game", 1, 21, ruleset),
        deterministic_seed=21,
        bootstrap_samples=20,
    )
    first = evaluate_scenario(scenario)
    second = evaluate_scenario(scenario)
    assert [game.winner_team for game in first.games] == [
        game.winner_team for game in second.games
    ]
    assert all(game.bid_value == 1 for game in first.games)
    assert first.metrics["bid_rate"] == 0.0


def test_report_writes_json_csv_and_markdown_without_nonstandard_nan(tmp_path):
    result = evaluate_scenario(_scenario("cardplay_only", deals=1))
    paths = write_report(result, tmp_path / "report")
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["protocol"] == "p15_paired_v1"
    assert payload["scenario"]["mode"] == "cardplay_only"
    assert payload["metrics"]["bid_rate"] is None
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").startswith("deal_id,")
    assert "Mode: `cardplay_only`" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert set(paths) == {"json", "csv", "markdown"}


def test_private_holdout_name_is_redacted_from_report_identity():
    base = _scenario("cardplay_only", deals=1)
    private = EvaluationScenario(
        mode=base.mode,
        ruleset=base.ruleset,
        candidate=base.candidate,
        baseline=base.baseline,
        deals=base.deals,
        dataset_scope="private_holdout",
        deal_set_name="secret-customer-path",
        bootstrap_samples=10,
    )
    assert private.to_dict()["deal_set_name"] == "private_holdout"
    assert "secret-customer-path" not in json.dumps(private.to_dict())


def test_ablation_runner_requires_explicit_checkpoint_backed_variants():
    scenario = _scenario("cardplay_only", deals=1)
    with pytest.raises(ValueError, match="incomplete ablation matrix"):
        AblationRunner(scenario, {}, require_complete=True)
    with pytest.raises(ValueError, match="unknown ablations"):
        AblationRunner(scenario, {"pretend_toggle": scenario.candidate})
    assert "no_search" in ABLATION_NAMES


def test_model_matrix_parses_bc_and_two_bundle_ablation(tmp_path):
    path = tmp_path / "matrix.json"
    checkpoints = {role: f"/{role}.ckpt" for role in (
        "landlord", "landlord_up", "landlord_down"
    )}
    path.write_text(json.dumps({
        "bundles": {
            "bc-stage": {"backend": "bc", "checkpoints": checkpoints},
            "legacy-base": {"backend": "legacy", "checkpoints": checkpoints},
        },
        "ablations": {
            "no_bidding": {
                "candidate": "legacy-base",
                "baseline": "legacy-base",
            }
        },
    }), encoding="utf-8")
    bundles, ablations = _load_matrix(str(path))
    assert bundles["bc-stage"].backend == "bc"
    assert ablations["no_bidding"]["baseline"] == "legacy-base"


def test_no_bidding_ablation_converts_full_game_to_cardplay_protocol():
    scenario = _scenario("full_game", deals=1)
    results = AblationRunner(
        scenario,
        {"no_bidding": BundleSpec(name="no-bidding", backend="rule")},
    ).run(include_base=False)
    result = results["no_bidding"]
    assert result.scenario["mode"] == "cardplay_only"
    assert result.scenario["ruleset"]["ruleset_id"] == "legacy"
    assert len(result.games) == 2


def test_regression_gates_cover_rules_latency_calibration_and_roles():
    result = evaluate_scenario(_scenario("cardplay_only", deals=1))
    gates = evaluate_regression_gates(
        result.metrics,
        RegressionGateConfig(
            max_p95_latency_ms=10_000,
            min_role_win_percentage={"landlord": 0.0, "landlord_up": 0.0},
            required_checks={"legacy_behavior": True, "environment_rules": False},
        ),
    )
    assert gates["passed"] is False
    assert any(
        check["name"] == "required:environment_rules" and not check["passed"]
        for check in gates["checks"]
    )


@pytest.mark.parametrize("bad_value", ["false", 0, 1, None])
def test_regression_gate_required_checks_reject_non_booleans(bad_value):
    with pytest.raises(TypeError, match="strings to booleans"):
        RegressionGateConfig(required_checks={"legacy_behavior": bad_value})


def test_regression_gate_required_checks_reject_non_string_names():
    with pytest.raises(TypeError, match="strings to booleans"):
        RegressionGateConfig(required_checks={1: False})
