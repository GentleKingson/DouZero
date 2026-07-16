"""Directed regressions for the independent P17 release-trust review."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace

import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel
from douzero.checkpoint import save_v2_position_weights
from douzero.deployment import (
    ModelPackageError,
    create_model_package,
    verify_model_package,
)
from douzero.deployment.manifest import ModelManifest, canonical_hash
from douzero.env.game import GameEnv
from douzero.env.rules import RuleSet
from douzero.env.scoring import compute_team_score_magnitude
from douzero.evaluation.agents import RuleAgent
from douzero.evaluation.legacy_data_adapter import deal_standard_deck
from douzero.evaluation.p17 import result_readiness
from douzero.evaluation.paired import _calibration_metrics, evaluate_scenario
from douzero.evaluation.scenario import (
    BundleSpec,
    EvaluationScenario,
    canonical_deal_hash,
    canonical_deal_id,
)
from douzero.evaluation.statistics import deal_cluster_means, paired_bootstrap_ci
from douzero.models_v2 import ModelV2, ModelV2Config
from douzero.models_v2.output import BiddingModelOutput
from douzero.observation import build_v2_schema
from douzero.observation.bidding import get_bidding_obs_v2
from douzero.training.bidding import (
    BiddingMinibatch,
    BiddingTransition,
    bidding_loss,
)
from douzero.training.v2_trainer import TrainerConfig, V2Trainer
from evaluate_paired import generate_deals


def _model(*, belief_enabled: bool = False) -> ModelV2:
    config = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        role_embedding_dim=8,
        mlp_layers=1,
        belief_enabled=belief_enabled,
        nan_guard=False,
    )
    return ModelV2(build_v2_schema(max_history_len=8), config)


def _refresh_package(package) -> None:
    names = sorted(path.name for path in package.iterdir() if path.name != "SHA256SUMS")
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )


def _evaluation_payload(mode: str, *, deals: int = 2, seed: int = 1901):
    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    bundle = BundleSpec(name="smoke", backend="random")
    scenario = EvaluationScenario(
        mode=mode,
        ruleset=ruleset,
        candidate=bundle,
        baseline=bundle,
        deals=generate_deals(mode, deals, seed, ruleset),
        deterministic_seed=seed,
        bootstrap_samples=2000,
    )
    return scenario, evaluate_scenario(scenario).to_dict()


def test_ordered_full_deal_hashes_drive_deal_set_readiness():
    ruleset = RuleSet.legacy()
    deals = generate_deals("cardplay_only", 2, 1901, ruleset)
    bundle = BundleSpec(name="smoke", backend="random")
    first = EvaluationScenario(
        mode="cardplay_only",
        ruleset=ruleset,
        candidate=bundle,
        baseline=bundle,
        deals=deals,
        bootstrap_samples=2000,
    )
    reordered = replace(first, deals=tuple(reversed(deals)))
    assert first.deal_set_id != reordered.deal_set_id
    mutated_deals = copy.deepcopy(deals)
    left = 0
    right = next(
        index for index, card in enumerate(mutated_deals[0]["landlord_up"])
        if card != mutated_deals[0]["landlord"][left]
    )
    mutated_deals[0]["landlord"][left], mutated_deals[0]["landlord_up"][right] = (
        mutated_deals[0]["landlord_up"][right],
        mutated_deals[0]["landlord"][left],
    )
    mutated = replace(first, deals=tuple(mutated_deals))
    assert first.deal_set_id != mutated.deal_set_id

    payload = evaluate_scenario(first).to_dict()
    assert all(len(row["deal_hash"]) == 64 for row in payload["games"])
    assert all(
        field not in row
        for row in payload["games"]
        for field in ("deal_payload", "deal_digest")
    )
    forged_id = copy.deepcopy(payload)
    forged_id["scenario"]["deal_set_id"] = "0" * 64
    readiness = result_readiness(
        forged_id, mode="cardplay_only", approved_deals=first.deals
    )
    assert any("deal_set_id" in issue for issue in readiness["issues"])

    forged_row_hash = copy.deepcopy(payload)
    row = forged_row_hash["games"][0]
    replacement = "f" if row["deal_hash"][0] != "f" else "e"
    row["deal_hash"] = replacement + row["deal_hash"][1:]
    readiness = result_readiness(
        forged_row_hash, mode="cardplay_only", approved_deals=first.deals
    )
    assert readiness["evidence"]["malformed_rows"] == 1

    forged_whole_deal = copy.deepcopy(payload)
    original_hash = forged_whole_deal["games"][0]["deal_hash"]
    replacement = "f" if original_hash[0] != "f" else "e"
    replacement_hash = replacement * 64
    for row in forged_whole_deal["games"]:
        if row["deal_id"].startswith("000000-"):
            row["deal_hash"] = replacement_hash
            row["deal_id"] = canonical_deal_id(0, replacement_hash)
    readiness = result_readiness(
        forged_whole_deal, mode="cardplay_only", approved_deals=first.deals
    )
    assert readiness["evidence"]["malformed_rows"] == 2

    reordered_rows = copy.deepcopy(payload)
    deal_hashes = {
        int(row["deal_id"].split("-", 1)[0]): row["deal_hash"]
        for row in reordered_rows["games"]
    }
    for row in reordered_rows["games"]:
        index = int(row["deal_id"].split("-", 1)[0])
        swapped_hash = deal_hashes[1 - index]
        row["deal_hash"] = swapped_hash
        row["deal_id"] = canonical_deal_id(index, swapped_hash)
    readiness = result_readiness(
        reordered_rows, mode="cardplay_only", approved_deals=first.deals
    )
    assert readiness["evidence"]["malformed_rows"] == 4

    serialization_only_reorder = copy.deepcopy(payload)
    serialization_only_reorder["games"].reverse()
    readiness = result_readiness(
        serialization_only_reorder,
        mode="cardplay_only",
        approved_deals=first.deals,
    )
    assert not any(
        "deal_set_id" in issue or "deal indices" in issue
        for issue in readiness["issues"]
    )


def test_deal_hash_ignores_metadata_and_card_representation_noise():
    legacy_ruleset = RuleSet.legacy()
    legacy = generate_deals("cardplay_only", 1, 1903, legacy_ruleset)[0]
    reordered_hand = copy.deepcopy(legacy)
    reordered_hand["landlord"].reverse()
    assert canonical_deal_hash(legacy) == canonical_deal_hash(reordered_hand)

    standard_ruleset = RuleSet.standard()
    standard = generate_deals("full_game", 1, 1904, standard_ruleset)[0]
    with_nonce = copy.deepcopy(standard)
    with_nonce["nonce"] = "ignored-metadata"
    assert canonical_deal_hash(standard) == canonical_deal_hash(with_nonce)
    bundle = BundleSpec(name="smoke", backend="random")
    with pytest.raises(ValueError, match="deals must be unique"):
        EvaluationScenario(
            mode="full_game",
            ruleset=standard_ruleset,
            candidate=bundle,
            baseline=bundle,
            deals=(standard, with_nonce),
        )


def test_private_full_game_rows_publish_hashes_not_cards():
    scenario, _payload = _evaluation_payload("full_game", deals=1, seed=1907)
    private = replace(
        scenario,
        dataset_scope="private_holdout",
        deal_set_name="secret-holdout-path",
    )
    payload = evaluate_scenario(private).to_dict()
    assert payload["scenario"]["deal_set_name"] == "private_holdout"
    assert all(len(row["deal_hash"]) == 64 for row in payload["games"])
    assert all(
        field not in row
        for row in payload["games"]
        for field in ("deal_payload", "deck", "three_landlord_cards")
    )


@pytest.mark.parametrize("mode", ["cardplay_only", "full_game"])
@pytest.mark.parametrize(
    "field", ["candidate_win", "candidate_score", "role_wins", "role_scores"]
)
def test_terminal_outcome_redundancies_are_rejected(mode, field):
    _scenario, payload = _evaluation_payload(mode, deals=1)
    row = payload["games"][0]
    if field == "candidate_win":
        row[field] = 1.0 - row[field]
    elif field == "candidate_score":
        row[field] += 1.0
    else:
        role = next(iter(row[field]))
        row[field][role] += 1.0
    readiness = result_readiness(
        payload, mode=mode, approved_deals=_scenario.deals
    )
    assert readiness["evidence"]["malformed_deals"] == 1
    assert any("trace does not replay" in issue for issue in readiness["issues"])


def test_forged_redeal_cap_exclusion_is_malformed_after_replay():
    scenario, payload = _evaluation_payload("full_game", deals=2, seed=1915)
    deal_ids = {row["deal_id"] for row in payload["games"]}
    target_deal = min(
        deal_ids,
        key=lambda deal_id: sum(
            row["candidate_score"]
            for row in payload["games"]
            if row["deal_id"] == deal_id
        ),
    )
    for row in payload["games"]:
        if row["deal_id"] == target_deal:
            row.update({
                "max_redeals_exceeded": True,
                "formal_evaluation_eligible": False,
                "exclusion_reason": (
                    "redeal_cap_exhausted_forced_smoke_fallback"
                ),
            })

    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )

    assert readiness["evidence"]["excluded_deals"] == 0
    assert readiness["evidence"]["malformed_deals"] == 1
    assert any("trace does not replay" in issue for issue in readiness["issues"])


def test_mixed_replayed_redeal_cap_state_across_rotations_is_malformed():
    ruleset = RuleSet.standard()
    deals = generate_deals("full_game", 1, 1916, ruleset)

    def scenario_with_bidding(policy: str) -> EvaluationScenario:
        return EvaluationScenario(
            mode="full_game",
            ruleset=ruleset,
            candidate=BundleSpec(
                name=f"candidate-{policy}",
                backend="rule",
                bidding_policy=policy,
            ),
            baseline=BundleSpec(
                name=f"baseline-{policy}",
                backend="rule",
                bidding_policy=policy,
            ),
            deals=deals,
            deterministic_seed=1916,
            bootstrap_samples=2000,
        )

    pass_scenario = scenario_with_bidding("pass")
    mixed = evaluate_scenario(pass_scenario).to_dict()
    successful = evaluate_scenario(scenario_with_bidding("max")).to_dict()
    replacement = successful["games"][0]
    replacement_assignment = replacement["assignment"]
    index = next(
        index
        for index, row in enumerate(mixed["games"])
        if row["assignment"] == replacement_assignment
    )
    mixed["games"][index] = copy.deepcopy(replacement)

    readiness = result_readiness(
        mixed, mode="full_game", approved_deals=pass_scenario.deals
    )

    assert readiness["evidence"]["excluded_deals"] == 0
    assert readiness["evidence"]["malformed_deals"] == 1
    assert any(
        "seat rotations disagree" in issue for issue in readiness["issues"]
    )


def test_coordinated_terminal_and_metric_forgery_is_rejected_by_original_trace():
    scenario, payload = _evaluation_payload("full_game", deals=1, seed=1911)
    original_trace_digests = [row["trace_digest"] for row in payload["games"]]
    for row in payload["games"]:
        forged_winner_team = (
            "farmer" if row["winner_team"] == "landlord" else "landlord"
        )
        forged_winner_position = (
            "landlord_up" if forged_winner_team == "farmer" else "landlord"
        )
        forged_bombs = row["bomb_count"] + 1
        forged_spring = forged_winner_team == "landlord"
        forged_anti_spring = forged_winner_team == "farmer"
        candidate_role = row["candidate_roles"][0]
        candidate_team = (
            "landlord" if candidate_role == "landlord" else "farmer"
        )
        magnitude = compute_team_score_magnitude(
            team=candidate_team,
            bomb_count=forged_bombs,
            rocket_count=row["rocket_count"],
            bid_value=row["bid_value"],
            ruleset=RuleSet.standard(),
            spring=forged_spring,
            anti_spring=forged_anti_spring,
        )
        candidate_win = float(forged_winner_team == candidate_team)
        candidate_score = float(
            magnitude if candidate_win else -magnitude
        )
        row.update({
            "winner_team": forged_winner_team,
            "winner_position": forged_winner_position,
            "bomb_count": forged_bombs,
            "spring": forged_spring,
            "anti_spring": forged_anti_spring,
            "candidate_win": candidate_win,
            "candidate_score": candidate_score,
            "candidate_log_score": math.copysign(
                math.log1p(abs(candidate_score)), candidate_score
            ),
            "role_wins": {candidate_role: candidate_win},
            "role_scores": {candidate_role: candidate_score},
        })

    forged_games = payload["games"]
    payload["metrics"].update({
        "overall_win_percentage": sum(
            row["candidate_win"] for row in forged_games
        ) / len(forged_games),
        "mean_score": sum(
            row["candidate_score"] for row in forged_games
        ) / len(forged_games),
        "mean_raw_score": sum(
            row["candidate_score"] for row in forged_games
        ) / len(forged_games),
        "mean_log_score": sum(
            row["candidate_log_score"] for row in forged_games
        ) / len(forged_games),
        "bomb_rate": 1.0,
        "spring_rate": sum(row["spring"] for row in forged_games)
        / len(forged_games),
        "anti_spring_rate": sum(
            row["anti_spring"] for row in forged_games
        ) / len(forged_games),
    })
    for role in ("landlord", "landlord_up", "landlord_down"):
        role_rows = [row for row in forged_games if role in row["candidate_roles"]]
        payload["metrics"]["by_role"][role].update({
            "games": len(role_rows),
            "win_percentage": sum(row["candidate_win"] for row in role_rows)
            / len(role_rows),
            "mean_score": sum(row["candidate_score"] for row in role_rows)
            / len(role_rows),
        })
    payload["metrics"]["paired_estimate_ci"] = paired_bootstrap_ci(
        deal_cluster_means(
            (row["deal_id"], row["candidate_score"]) for row in forged_games
        ),
        confidence_level=0.95,
        samples=scenario.bootstrap_samples,
        seed=scenario.deterministic_seed + 1,
    ).to_dict()

    assert [row["trace_digest"] for row in forged_games] == original_trace_digests
    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    assert readiness["evidence"]["paired_deals"] == 0
    assert readiness["evidence"]["malformed_deals"] == 1
    assert any("trace does not replay" in issue for issue in readiness["issues"])


def test_legacy_calibration_triple_with_forged_label_is_rejected():
    scenario, payload = _evaluation_payload("cardplay_only", deals=1, seed=1912)
    row = payload["games"][0]
    role = row["candidate_roles"][0]
    true_label = float(
        row["winner_team"]
        == ("landlord" if role == "landlord" else "farmer")
    )
    forged_label = 1.0 - true_label
    row["calibration"] = [[role, forged_label, forged_label]]
    payload["metrics"]["calibration"] = _calibration_metrics(
        [(role, forged_label, forged_label)]
    )

    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )
    diagnostics = readiness["evidence"]["recomputed_diagnostics"]
    assert readiness["evidence"]["paired_deals"] == 1
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["calibration"] is None
    assert any("calibration" in issue for issue in readiness["issues"])


def test_forged_prediction_summary_cannot_replace_replay_derived_label():
    scenario, payload = _evaluation_payload("full_game", deals=1, seed=1913)
    forged_calibration = []
    prediction_count = 0
    for row in payload["games"]:
        row["calibration"] = []
        for decision in row["candidate_decisions"]:
            if decision["phase"] != "cardplay" or decision["forced_action"]:
                continue
            role = decision["actor_role"]
            true_label = float(
                row["winner_team"]
                == ("landlord" if role == "landlord" else "farmer")
            )
            forged_prediction = 1.0 - true_label
            decision["prediction_status"] = "available"
            decision["prediction"] = forged_prediction
            row["calibration"].append([role, forged_prediction])
            forged_calibration.append(
                (role, forged_prediction, forged_prediction)
            )
            prediction_count += 1
    # This is the perfect aggregate an attacker would obtain by treating the
    # prediction itself as the label. The collator must instead use replay.
    payload["metrics"]["calibration"] = _calibration_metrics(
        forged_calibration
    )
    payload["metrics"]["sample_counts"]["calibration_decisions"] = (
        prediction_count
    )

    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    diagnostics = readiness["evidence"]["recomputed_diagnostics"]
    assert diagnostics["status"] == "available"
    assert diagnostics["calibration"]["overall"]["count"] == prediction_count
    assert diagnostics["calibration"]["overall"]["brier"] == pytest.approx(1.0)
    assert payload["metrics"]["calibration"]["overall"]["brier"] == 0.0
    assert any(
        "reported calibration does not match" in issue
        for issue in readiness["issues"]
    )


def test_calibration_role_must_be_replay_verified_candidate_role():
    scenario, payload = _evaluation_payload("full_game", deals=1, seed=1914)
    row = payload["games"][0]
    baseline_role = next(
        role
        for role in ("landlord", "landlord_up", "landlord_down")
        if role not in row["candidate_roles"]
    )
    row["calibration"] = [[baseline_role, 0.5]]

    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    diagnostics = readiness["evidence"]["recomputed_diagnostics"]
    assert readiness["evidence"]["paired_deals"] == 1
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["calibration"] is None
    assert any("calibration" in issue for issue in readiness["issues"])


def test_synchronized_derived_score_and_ci_tampering_is_rejected():
    scenario, payload = _evaluation_payload("full_game", deals=2, seed=1902)
    for row in payload["games"]:
        row["candidate_score"] *= 2.0
        row["candidate_log_score"] = math.copysign(
            math.log1p(abs(row["candidate_score"])), row["candidate_score"]
        )
        row["role_scores"] = {
            role: score * 2.0 for role, score in row["role_scores"].items()
        }
    deal_values = deal_cluster_means(
        (row["deal_id"], row["candidate_score"]) for row in payload["games"]
    )
    payload["metrics"]["paired_estimate_ci"] = paired_bootstrap_ci(
        deal_values,
        confidence_level=0.95,
        samples=scenario.bootstrap_samples,
        seed=scenario.deterministic_seed + 1,
    ).to_dict()
    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    assert readiness["evidence"]["paired_deals"] == 0
    assert any("trace does not replay" in issue for issue in readiness["issues"])


def test_full_game_roles_are_bound_to_the_bidding_transcript():
    scenario, payload = _evaluation_payload("full_game", deals=1, seed=1905)
    row = payload["games"][0]
    order = row["bidding_order"]
    original_roles = [row["seat_to_role"][seat] for seat in order]
    rotated_roles = (*original_roles[1:], original_roles[0])
    row["seat_to_role"] = dict(zip(order, rotated_roles))
    candidate_seat = str(row["assignment"].index("candidate"))
    candidate_role = row["seat_to_role"][candidate_seat]
    candidate_team = "landlord" if candidate_role == "landlord" else "farmer"
    magnitude = compute_team_score_magnitude(
        team=candidate_team,
        bomb_count=row["bomb_count"],
        rocket_count=row["rocket_count"],
        bid_value=row["bid_value"],
        ruleset=RuleSet.standard(),
        spring=row["spring"],
        anti_spring=row["anti_spring"],
    )
    candidate_score = float(
        magnitude if row["winner_team"] == candidate_team else -magnitude
    )
    candidate_win = float(row["winner_team"] == candidate_team)
    row.update({
        "candidate_roles": [candidate_role],
        "candidate_win": candidate_win,
        "candidate_score": candidate_score,
        "candidate_log_score": math.copysign(
            math.log1p(abs(candidate_score)), candidate_score
        ),
        "candidate_landlord": int(candidate_team == "landlord"),
        "role_wins": {candidate_role: candidate_win},
        "role_scores": {candidate_role: candidate_score},
    })
    readiness = result_readiness(
        payload, mode="full_game", approved_deals=scenario.deals
    )
    assert readiness["evidence"]["malformed_deals"] == 1
    assert any("trace does not replay" in issue for issue in readiness["issues"])

    malformed_json_value = copy.deepcopy(payload)
    malformed_json_value["games"][0]["seat_to_role"]["0"] = []
    readiness = result_readiness(
        malformed_json_value,
        mode="full_game",
        approved_deals=scenario.deals,
    )
    assert readiness["evidence"]["malformed_deals"] == 1


def test_custom_ruleset_p15_result_is_not_formal_p17_evidence():
    custom_ruleset = replace(RuleSet.standard(), base_score=2)
    bundle = BundleSpec(name="descriptive", backend="random")
    scenario = EvaluationScenario(
        mode="full_game",
        ruleset=custom_ruleset,
        candidate=bundle,
        baseline=bundle,
        deals=generate_deals("full_game", 1, 1906, custom_ruleset),
        bootstrap_samples=2000,
    )
    readiness = result_readiness(
        evaluate_scenario(scenario).to_dict(),
        mode="full_game",
        approved_deals=scenario.deals,
    )
    assert any("official ruleset identity" in issue for issue in readiness["issues"])


@pytest.mark.parametrize("confidence_level", [0.01, 0.90, 0.99])
def test_p17_rejects_non_official_confidence_levels(confidence_level):
    scenario, payload = _evaluation_payload("cardplay_only", deals=1)
    payload["scenario"]["confidence_level"] = confidence_level
    readiness = result_readiness(
        payload, mode="cardplay_only", approved_deals=scenario.deals
    )
    assert any("confidence_level=0.95" in issue for issue in readiness["issues"])
    assert (
        readiness["evidence"]["recomputed_paired_estimate_ci"]["confidence_level"]
        == 0.95
    )


def test_verifier_rejects_strict_state_dict_failure_after_all_hashes_refresh(tmp_path):
    model = _model()
    package = tmp_path / "package"
    create_model_package(package, model, RuleSet.legacy())
    weights = torch.load(package / "weights.pt", map_location="cpu", weights_only=True)
    weights["model_state_dict"].pop(next(iter(weights["model_state_dict"])))
    torch.save(weights, package / "weights.pt")
    manifest = ModelManifest.from_dict(
        json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest = replace(
        manifest,
        weights_sha256=hashlib.sha256((package / "weights.pt").read_bytes()).hexdigest(),
    )
    (package / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )
    _refresh_package(package)
    with pytest.raises(ModelPackageError, match="strict-load"):
        verify_model_package(package)


def test_package_training_identity_must_come_from_source_checkpoint(tmp_path):
    model = _model()
    ruleset = RuleSet.legacy()
    source = tmp_path / "source.pt"
    save_v2_position_weights(
        str(source), model, ruleset=ruleset, flags={"seed": 7}
    )
    with pytest.raises(ModelPackageError, match="does not match the source"):
        create_model_package(
            tmp_path / "wrong",
            model,
            ruleset,
            source_checkpoint=source,
            training_config={"seed": 8},
        )
    manifest = create_model_package(
        tmp_path / "matched",
        model,
        ruleset,
        source_checkpoint=source,
        training_config={"seed": 7},
    )
    assert manifest.release_eligible is True
    assert manifest.source_training_config_hash == canonical_hash({"seed": 7})
    assert manifest.source_checkpoint_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_epsilon_random_bid_has_no_policy_imitation_gradient():
    ruleset = RuleSet.standard()
    deck = list(range(3, 15)) * 4 + [17] * 4 + [20, 30]
    env = GameEnv(
        {role: RuleAgent() for role in ("landlord", "landlord_up", "landlord_down")},
        ruleset=ruleset,
    )
    env.card_play_init_standard(deal_standard_deck(deck))
    obs = get_bidding_obs_v2(env.get_bidding_obs(), ruleset=ruleset)
    action = max(obs.legal_bids)
    transition = BiddingTransition(
        obs,
        action,
        "policy",
        "epsilon_random",
        policy_target_valid=False,
    )
    transition.label_from_terminal({
        "team_targets": {"landlord": {"target_win": 1.0, "target_score": 2.0}}
    })
    logits = torch.zeros(4, requires_grad=True)
    output = BiddingModelOutput(
        logits,
        torch.as_tensor(obs.bid_action_mask.copy()),
        torch.tensor(0.0, requires_grad=True),
        torch.tensor(0.0, requires_grad=True),
    )
    loss = bidding_loss(
        [output],
        BiddingMinibatch([transition]),
        lambda_policy=1.0,
        lambda_landlord_win=0.0,
        lambda_landlord_score=0.0,
    )
    loss.total.backward()
    assert loss.policy == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_frozen_belief_resume_rejects_same_config_different_weights(tmp_path):
    torch.manual_seed(1905)
    model = _model(belief_enabled=True)
    belief_config = BeliefConfig(hidden_size=24, num_layers=1)
    belief = BeliefModel(belief_config)
    trainer = V2Trainer(
        model,
        belief_model=belief,
        config=TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=9),
    )
    checkpoint = tmp_path / "trainer.pt"
    trainer.save_training_checkpoint(str(checkpoint))

    torch.manual_seed(1906)
    different = V2Trainer(
        _model(belief_enabled=True),
        belief_model=BeliefModel(belief_config),
        config=TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=9),
    )
    with pytest.raises(Exception, match="frozen belief weights"):
        different.load_training_checkpoint(str(checkpoint))


def test_checkpoint_restore_failure_is_atomic(tmp_path):
    trainer = V2Trainer(
        _model(),
        config=TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=13),
    )
    checkpoint = tmp_path / "valid.pt"
    trainer.save_training_checkpoint(str(checkpoint))
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    bundle["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    corrupt = tmp_path / "corrupt.pt"
    torch.save(bundle, corrupt)

    model_before = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    with pytest.raises(Exception, match="atomically"):
        trainer.load_training_checkpoint(str(corrupt))
    assert all(
        torch.equal(value, trainer.model.state_dict()[name])
        for name, value in model_before.items()
    )
    assert trainer.optimizer.state_dict() == optimizer_before


def test_redeal_cap_marks_whole_episode_excluded():
    config = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        role_embedding_dim=8,
        mlp_layers=1,
        bidding_enabled=True,
        bidding_hidden_size=16,
        nan_guard=False,
    )
    from douzero.training.bidding import BiddingPolicyConfig
    from douzero.training.losses import LossConfig

    trainer = V2Trainer(
        ModelV2(build_v2_schema(), config),
        ruleset=replace(RuleSet.standard(), max_redeals=0),
        loss_config=LossConfig(lambda_bid_policy=1.0),
        bidding_policy_config=BiddingPolicyConfig(policy="pass"),
        config=TrainerConfig(max_episodes=0, optimizer_steps=0, rng_seed=11),
    )
    episode = trainer._run_one_episode()
    assert episode.excluded_from_training is True
    assert episode.exclusion_reason == "redeal_cap_guard"
    assert episode.transitions == []
    assert episode.bidding_transitions == []
