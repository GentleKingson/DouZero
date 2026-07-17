"""P15 paired evaluation, clustered confidence intervals, and reports."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from douzero.env.rules import RuleSet
from douzero.evaluation.ablation import ABLATION_NAMES, AblationRunner
from douzero.evaluation.gates import RegressionGateConfig, evaluate_regression_gates
from douzero.evaluation.paired import (
    GameRecord,
    evaluate_scenario,
    validate_evaluation_runtime_identity,
)
from douzero.evaluation.p17 import result_readiness
from douzero.evaluation.provenance import attach_result_integrity
from douzero.evaluation.replay import replay_game_record
from douzero.evaluation.reporting import write_report
from douzero.evaluation.scenario import (
    BundleSpec,
    EvaluationScenario,
    bundle_from_dict,
)
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


def test_p17_fields_preserve_pre_p17_positional_dataclass_contracts():
    bundle = BundleSpec(
        "positional",
        "rule",
        {},
        "max",
        "pure_win",
        {},
        "",
        {"enabled": False},
        ("pre-p17",),
    )
    assert bundle.search_config == {"enabled": False}
    assert bundle.tags == ("pre-p17",)
    assert bundle.bidding_checkpoint == ""
    matrix_payload = bundle.to_dict(include_paths=True)
    assert "checkpoint_identities" not in matrix_payload
    assert bundle_from_dict(matrix_payload) == bundle
    assert "checkpoint_identities" in bundle.to_dict()

    calibration = (("landlord", 0.75),)
    record = GameRecord(
        "deal",
        "leg",
        "cardplay_only",
        ("candidate", "baseline", "baseline"),
        ("landlord",),
        1.0,
        1.0,
        {"landlord": 1.0},
        {"landlord": 1.0},
        "landlord",
        "landlord",
        0,
        0,
        0,
        1,
        0,
        0,
        False,
        False,
        1,
        (1.25,),
        calibration,
    )
    assert record.candidate_latencies_ms == (1.25,)
    assert record.calibration == calibration
    assert record.redeal_count == 0
    assert record.formal_evaluation_eligible is True


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
    assert first.metrics["actor_fps"] == first.metrics["inference_calls_per_second"]
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

    def without_timing(game):
        payload = game.to_dict()
        payload["candidate_latencies_ms"] = []
        payload["candidate_latencies_ns"] = []
        for decision in payload["candidate_decisions"]:
            decision["latency_ns"] = 0
        return payload

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
    assert [without_timing(game) for game in result.games] == [
        without_timing(game) for game in repeated.games
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
    assert all(not game.formal_evaluation_eligible for game in first.games)
    assert all(
        game.exclusion_reason == "redeal_cap_exhausted_forced_smoke_fallback"
        for game in first.games
    )
    assert all(
        game.redeal_count == ruleset.max_redeals + 1 for game in first.games
    )
    assert all(
        len(game.bidding_trace) == ruleset.max_redeals + 1
        and all(
            len(attempt) == 3 and all(bid == 0 for _seat, bid in attempt)
            for attempt in game.bidding_trace
        )
        for game in first.games
    )
    for game in first.games:
        replay = replay_game_record(
            game.to_dict(),
            scenario.deals[0],
            mode="full_game",
            ruleset=ruleset,
            deterministic_seed=scenario.deterministic_seed,
        )
        assert replay.max_redeals_exceeded is True
        assert replay.redeal_count == ruleset.max_redeals + 1
        assert replay.bid_value == 1
        assert replay.bidding_history == ((replay.bidding_order[0], 1),)
    assert first.metrics["formal_evaluation"]["excluded_deals"] == 1
    assert first.metrics["paired_estimate_ci"]["paired_deals"] == 0
    assert math.isnan(first.metrics["bid_rate"])
    assert first.metrics["smoke_descriptive"]["bid_rate"] == 0.0
    readiness = result_readiness(
        first.to_dict(), mode="full_game", approved_deals=scenario.deals
    )
    assert readiness["status"] == "insufficient"
    assert readiness["evidence"]["excluded_deals"] == 1
    assert readiness["evidence"]["malformed_deals"] == 0
    assert any("redeal cap" in issue for issue in readiness["issues"])


def test_report_writes_json_csv_and_markdown_without_nonstandard_nan(tmp_path):
    result = evaluate_scenario(_scenario("cardplay_only", deals=1))
    paths = write_report(result, tmp_path / "report")
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["protocol"] == "p15_paired_v1"
    assert payload["scenario"]["mode"] == "cardplay_only"
    assert payload["metrics"]["bid_rate"] is None
    csv_text = (tmp_path / "report.csv").read_text(encoding="utf-8")
    markdown_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert csv_text.startswith("deal_id,deal_hash,")
    assert payload["runtime_identity"]["source_git_sha"] in csv_text
    assert "Mode: `cardplay_only`" in markdown_text
    assert payload["runtime_identity"]["source_git_sha"] in markdown_text
    assert payload["runtime_identity"]["evaluation_config_hash"] in markdown_text
    assert set(paths) == {"json", "csv", "markdown"}


def test_evaluation_runtime_identity_binds_protocol_mode_rules_and_schemas():
    payload = evaluate_scenario(_scenario("cardplay_only", deals=1)).to_dict()
    runtime = payload["runtime_identity"]
    assert len(runtime["source_git_sha"]) in (40, 64)
    assert runtime["model_feature_schemas"]["candidate"]["feature_version"] == (
        "builtin-no-model-input"
    )
    assert runtime["model_feature_schemas"]["baseline"]["feature_version"] == (
        "builtin-no-model-input"
    )
    validate_evaluation_runtime_identity(payload, expected_mode="cardplay_only")
    validate_evaluation_runtime_identity(
        payload,
        expected_mode="cardplay_only",
        expected_source_git_shas=runtime["source_git_sha"],
    )
    replacement = "f" if runtime["source_git_sha"][0] != "f" else "e"
    with pytest.raises(ValueError, match="source_git_sha is not approved"):
        validate_evaluation_runtime_identity(
            payload,
            expected_source_git_shas=[replacement * len(runtime["source_git_sha"])],
        )

    wrong_protocol = json.loads(json.dumps(payload))
    wrong_protocol["protocol"] = "pretend-protocol"
    with pytest.raises(ValueError, match="result protocol mismatch"):
        validate_evaluation_runtime_identity(wrong_protocol)

    wrong_mode = json.loads(json.dumps(payload))
    wrong_mode["scenario"]["mode"] = "full_game"
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_evaluation_runtime_identity(
            wrong_mode, expected_mode="cardplay_only"
        )

    wrong_rules = json.loads(json.dumps(payload))
    wrong_rules["runtime_identity"]["ruleset_hash"] = "0" * 64
    with pytest.raises(ValueError, match="result integrity"):
        validate_evaluation_runtime_identity(wrong_rules)

    wrong_schema = json.loads(json.dumps(payload))
    wrong_schema["runtime_identity"]["model_feature_schemas"]["candidate"][
        "feature_version"
    ] = "legacy"
    with pytest.raises(ValueError, match="result integrity"):
        validate_evaluation_runtime_identity(wrong_schema)

    wrong_config = json.loads(json.dumps(payload))
    wrong_config["scenario"]["deterministic_seed"] += 1
    with pytest.raises(ValueError, match="result integrity"):
        validate_evaluation_runtime_identity(wrong_config)


def test_formal_runtime_identity_binds_workflow_container_and_hardware():
    payload = evaluate_scenario(_scenario("cardplay_only", deals=1)).to_dict()
    runtime = payload["runtime_identity"]
    source_sha = runtime["source_git_sha"]
    runtime.update({
        "source_worktree_clean": True,
        "source_identity_stable": True,
        "source_identity_samples": 2,
    })
    runtime["execution_environment"].update({
        "provider": "github_actions",
        "repository": "GentleKingson/DouZero",
        "workflow_ref": (
            "GentleKingson/DouZero/.github/workflows/"
            "formal-evaluation.yml@refs/heads/main"
        ),
        "workflow_sha": source_sha,
        "source_ref": "refs/heads/main",
        "source_sha": source_sha,
        "run_id": "12345",
        "run_attempt": "2",
        "run_url": (
            "https://github.com/GentleKingson/DouZero/"
            "actions/runs/12345/attempts/2"
        ),
        "runner_environment": "github-hosted",
        "container_image_digest": "sha256:" + "b" * 64,
    })
    payload = attach_result_integrity({
        key: value for key, value in payload.items() if key != "result_integrity"
    })

    validate_evaluation_runtime_identity(
        payload,
        expected_source_git_shas=[source_sha],
        require_formal_source=True,
    )

    malformed = json.loads(json.dumps(payload))
    malformed["runtime_identity"]["execution_environment"][
        "container_image_digest"
    ] = None
    malformed = attach_result_integrity({
        key: value
        for key, value in malformed.items()
        if key != "result_integrity"
    })
    with pytest.raises(ValueError, match="workflow/container identity"):
        validate_evaluation_runtime_identity(
            malformed,
            require_formal_source=True,
        )


def test_runtime_source_identity_ignores_environment_sha(monkeypatch):
    monkeypatch.setenv("DOUZERO_GIT_SHA", "0" * 40)
    runtime = evaluate_scenario(
        _scenario("cardplay_only", deals=1)
    ).to_dict()["runtime_identity"]
    assert runtime["source_git_sha"] != "0" * 40
    assert runtime["source_identity_method"] == (
        "git-head-tree-and-tracked-bytes-v1"
    )


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
    payload = evaluate_scenario(private).to_dict()
    assert all(len(row["deal_hash"]) == 64 for row in payload["games"])
    assert all(
        field not in row
        for row in payload["games"]
        for field in ("deal_payload", "deal_digest", "team_scores")
    )


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
