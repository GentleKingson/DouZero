"""P12 coach labels, deterministic curriculum, and evaluation-isolation tests."""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from douzero.coach import (
    BALANCED,
    HARD_FOR_ROLE,
    TRUE_RANDOM,
    CoachLabel,
    CoachLabelStore,
    CoachModel,
    CoachModelConfig,
    CurriculumAuditLogger,
    CurriculumSchedule,
    OpeningRecord,
    OpeningSampler,
    calibration_metrics,
    encode_opening,
    load_coach_checkpoint,
    random_opening,
    save_coach_checkpoint,
    train_coach,
)
from douzero.config import load_config
from douzero.env.env import Env
from douzero.env.rules import RuleSet


class HashCoach:
    def predict(self, opening: OpeningRecord, policy_version: str) -> float:
        salt = sum(policy_version.encode("utf-8"))
        value = (int(opening.opening_id[:8], 16) + salt) % 10001
        return value / 10000.0


def _terminal(landlord_win: bool = True) -> dict:
    score = 2.0 if landlord_win else -2.0
    return {
        "winner_team": "landlord" if landlord_win else "farmer",
        "team_targets": {"landlord": {"target_score": score}},
    }


def test_opening_record_round_trip_and_ruleset_identity():
    opening = random_opening(random.Random(17), RuleSet.legacy())
    restored = OpeningRecord.from_dict(opening.to_dict())
    assert restored == opening
    assert restored.opening_id == opening.opening_id
    assert encode_opening(opening, "policy-v4").shape == (74,)

    tampered = opening.to_dict()
    tampered["deck"][0] = 99
    with pytest.raises(ValueError, match="complete 54-card deck"):
        OpeningRecord.from_dict(tampered)


def test_balanced_sampler_returns_complete_playable_game(tmp_path):
    audit = tmp_path / "openings.jsonl"
    sampler = OpeningSampler(
        policy_version="policy-8",
        coach=HashCoach(),
        mode=BALANCED,
        candidate_pool_size=12,
        seed=2,
        logger=CurriculumAuditLogger(str(audit)),
    )
    opening, record = sampler.sample(progress=0.1)
    assert record.selected_strategy == BALANCED
    assert 0.0 <= record.predicted_landlord_win <= 1.0
    data = opening.to_card_play_data()
    assert Counter(data["landlord"] + data["landlord_up"] + data["landlord_down"]) == Counter(opening.deck)

    env = Env("adp")
    env.reset(opening=opening)
    for _ in range(1000):
        _obs, _reward, done, _info = env.step(env.infoset.legal_actions[0])
        if done:
            break
    else:
        pytest.fail("sampled opening did not produce a terminating legal game")
    logged = json.loads(audit.read_text(encoding="utf-8"))
    assert logged["opening_id"] == opening.opening_id
    assert logged["cumulative_distribution"] == {
        "balanced": 1.0,
        "hard_for_role": 0.0,
        "true_random": 0.0,
    }


def test_standard_opening_is_complete_and_enters_configured_bidding_order():
    ruleset = RuleSet.standard()
    opening = random_opening(random.Random(91), ruleset)
    env = Env("adp", ruleset=ruleset)
    env.reset(opening=opening)
    assert tuple(env._env.bidding_order) == opening.bidding_order
    assert env._env.acting_player_position == opening.bidding_order[0]
    assert Counter(
        opening.to_card_play_data()["landlord"]
        + opening.to_card_play_data()["landlord_up"]
        + opening.to_card_play_data()["landlord_down"]
        + opening.to_card_play_data()["three_landlord_cards"]
    ) == Counter(opening.deck)


def test_fixed_seed_reproduces_mixture_and_openings():
    first = OpeningSampler(
        policy_version="policy-a", coach=HashCoach(), seed=55,
        candidate_pool_size=3,
    )
    second = OpeningSampler(
        policy_version="policy-a", coach=HashCoach(), seed=55,
        candidate_pool_size=3,
    )
    left = [first.sample(progress=index / 9) for index in range(10)]
    right = [second.sample(progress=index / 9) for index in range(10)]
    assert [item[0].opening_id for item in left] == [item[0].opening_id for item in right]
    assert [item[1].selected_strategy for item in left] == [
        item[1].selected_strategy for item in right
    ]


def test_mixture_ratios_and_real_sample_floor():
    proportions = {
        TRUE_RANDOM: 0.25,
        BALANCED: 0.50,
        HARD_FOR_ROLE: 0.25,
    }
    schedule = CurriculumSchedule(
        early=proportions,
        middle=proportions,
        late=proportions,
        min_true_random_ratio=0.25,
    )
    sampler = OpeningSampler(
        policy_version="policy-ratio",
        coach=HashCoach(),
        schedule=schedule,
        candidate_pool_size=1,
        seed=7,
    )
    for _ in range(1200):
        sampler.sample(progress=0.5)
    assert sampler.counts[TRUE_RANDOM] / 1200 == pytest.approx(0.25, abs=0.04)
    assert sampler.counts[BALANCED] / 1200 == pytest.approx(0.50, abs=0.04)
    assert sampler.counts[HARD_FOR_ROLE] / 1200 == pytest.approx(0.25, abs=0.04)

    fixed = OpeningSampler(
        policy_version="policy-fixed",
        coach=HashCoach(),
        mode=BALANCED,
        candidate_pool_size=1,
        seed=11,
    )
    for _ in range(600):
        fixed.sample(progress=0.1)
    assert fixed.counts[TRUE_RANDOM] / 600 == pytest.approx(0.20, abs=0.05)
    assert fixed.counts[HARD_FOR_ROLE] == 0

    with pytest.raises(ValueError, match="below min_true_random_ratio"):
        CurriculumSchedule(
            early=proportions,
            middle=proportions,
            late={TRUE_RANDOM: 0.1, BALANCED: 0.8, HARD_FOR_ROLE: 0.1},
            min_true_random_ratio=0.25,
        )


def test_hard_for_role_selects_opposite_difficulty_tails():
    no_floor = CurriculumSchedule(min_true_random_ratio=0.0)
    landlord = OpeningSampler(
        policy_version="policy-hard",
        coach=HashCoach(),
        mode=HARD_FOR_ROLE,
        hard_role="landlord",
        candidate_pool_size=20,
        schedule=no_floor,
        seed=71,
    )
    farmer = OpeningSampler(
        policy_version="policy-hard",
        coach=HashCoach(),
        mode=HARD_FOR_ROLE,
        hard_role="farmer",
        candidate_pool_size=20,
        schedule=no_floor,
        seed=71,
    )
    _landlord_opening, landlord_record = landlord.sample(progress=0.0)
    _farmer_opening, farmer_record = farmer.sample(progress=0.0)
    assert landlord_record.predicted_landlord_win < farmer_record.predicted_landlord_win


def test_expired_and_other_policy_labels_are_filtered(tmp_path):
    store = CoachLabelStore(str(tmp_path / "labels.jsonl"))
    opening = random_opening(random.Random(3))
    for version, step in (("current", 95), ("current", 70), ("old", 99)):
        store.append(CoachLabel.from_terminal(
            opening,
            _terminal(),
            policy_version=version,
            policy_step=step,
        ))
    fresh = store.load_fresh(
        policy_version="current", current_policy_step=100, max_age_steps=10
    )
    assert [(label.policy_version, label.policy_step) for label in fresh] == [
        ("current", 95)
    ]


def test_coach_training_checkpoint_and_calibration(tmp_path):
    labels = [
        CoachLabel.from_terminal(
            random_opening(random.Random(seed)),
            _terminal(seed % 2 == 0),
            policy_version="policy-train",
            policy_step=20,
        )
        for seed in range(6)
    ]
    model = CoachModel(CoachModelConfig(hidden_size=16))
    losses = train_coach(model, labels, epochs=2, learning_rate=1e-3)
    assert len(losses) == 2 and all(value >= 0.0 for value in losses)
    probabilities = [model.predict(item.opening, item.policy_version) for item in labels]
    metrics = calibration_metrics(
        probabilities, [item.landlord_win for item in labels], bins=3
    )
    assert set(metrics) == {"brier", "ece", "count"}

    path = tmp_path / "coach.pt"
    ruleset_hash = RuleSet.legacy().stable_hash()
    save_coach_checkpoint(
        str(path), model,
        policy_version="policy-train",
        policy_step=20,
        ruleset_hash=ruleset_hash,
        calibration=metrics,
    )
    restored, manifest = load_coach_checkpoint(
        str(path),
        expected_ruleset_hash=ruleset_hash,
        expected_policy_version="policy-train",
        current_policy_step=20,
        max_age_steps=0,
    )
    assert manifest["policy_version"] == "policy-train"
    assert restored.predict(labels[0].opening, "policy-train") == pytest.approx(
        model.predict(labels[0].opening, "policy-train")
    )
    with pytest.raises(ValueError, match="ruleset_hash mismatch"):
        load_coach_checkpoint(
            str(path),
            expected_ruleset_hash="wrong",
            expected_policy_version="policy-train",
            current_policy_step=20,
            max_age_steps=0,
        )


def test_coach_checkpoint_rejects_policy_mismatch_stale_and_future(tmp_path):
    model = CoachModel(CoachModelConfig(hidden_size=8))
    path = tmp_path / "identity-coach.pt"
    ruleset_hash = RuleSet.legacy().stable_hash()
    save_coach_checkpoint(
        str(path),
        model,
        policy_version="policy-a",
        policy_step=100,
        ruleset_hash=ruleset_hash,
    )

    common = {"expected_ruleset_hash": ruleset_hash, "map_location": "cpu"}
    with pytest.raises(ValueError, match="policy_version does not match"):
        load_coach_checkpoint(
            str(path),
            expected_policy_version="policy-b",
            current_policy_step=100,
            max_age_steps=0,
            **common,
        )
    with pytest.raises(ValueError, match="stale"):
        load_coach_checkpoint(
            str(path),
            expected_policy_version="policy-a",
            current_policy_step=111,
            max_age_steps=10,
            **common,
        )
    with pytest.raises(ValueError, match="future-dated"):
        load_coach_checkpoint(
            str(path),
            expected_policy_version="policy-a",
            current_policy_step=99,
            max_age_steps=10,
            **common,
        )


def test_v2_trainer_records_sampled_opening_label(tmp_path):
    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training.v2_trainer import TrainerConfig, V2Trainer

    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            history_heads=1,
        ),
    )
    store = CoachLabelStore(str(tmp_path / "trainer-labels.jsonl"))
    sampler = OpeningSampler(
        policy_version="policy-live", mode=TRUE_RANDOM, seed=19
    )
    trainer = V2Trainer(
        model,
        config=TrainerConfig(
            max_episodes=1,
            optimizer_steps=0,
            batch_size=2,
            exp_epsilon=0.0,
            seed=19,
            rng_seed=19,
        ),
        opening_sampler=sampler,
        coach_label_store=store,
        policy_version="policy-live",
        policy_step=42,
    )
    trainer.collect_episodes(1)
    first_episode = trainer.buffer._episodes[-1]
    assert first_episode.policy_version_at_start == "policy-live"
    assert first_episode.policy_step_at_start == 42
    assert trainer.step() is not None
    assert trainer.stats.optimizer_steps == 1
    trainer.collect_episodes(1)
    second_episode = trainer.buffer._episodes[-1]
    assert second_episode.policy_version_at_start == "policy-live"
    assert second_episode.policy_step_at_start == 43
    labels = store.load_fresh(
        policy_version="policy-live", current_policy_step=43, max_age_steps=1
    )
    assert [label.policy_step for label in labels] == [42, 43]
    assert trainer.stats.opening_strategy_counts == {TRUE_RANDOM: 2}


def test_coach_label_uses_episode_start_identity_not_terminal_state(tmp_path):
    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema
    from douzero.training.v2_buffer import Episode
    from douzero.training.v2_trainer import TrainerConfig, V2Trainer

    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            hidden_size=8,
            history_encoder="lstm",
            history_layers=1,
            history_heads=1,
        ),
    )
    store = CoachLabelStore(str(tmp_path / "start-identity.jsonl"))
    sampler = OpeningSampler(
        policy_version="policy-start", mode=TRUE_RANDOM, seed=5
    )
    trainer = V2Trainer(
        model,
        config=TrainerConfig(optimizer_steps=0),
        opening_sampler=sampler,
        coach_label_store=store,
        policy_version="policy-start",
        policy_step=100,
    )
    opening = random_opening(random.Random(8))
    episode = Episode(
        terminal_result=_terminal(),
        policy_version_at_start="policy-start",
        policy_step_at_start=107,
    )
    trainer.stats.optimizer_steps = 999
    trainer._record_coach_label(opening, episode)
    labels = store.load_fresh(
        policy_version="policy-start",
        current_policy_step=107,
        max_age_steps=0,
    )
    assert len(labels) == 1
    assert labels[0].policy_step == 107


def test_curriculum_defaults_off_and_evaluation_has_no_coach_import():
    enhanced = load_config("configs/enhanced.yaml").curriculum
    assert not enhanced.enabled
    assert enhanced.max_coach_age_steps == 100000
    assert not load_config("configs/legacy.yaml").curriculum.enabled
    root = Path(__file__).resolve().parents[1]
    evaluation_sources = [root / "evaluate.py", *sorted((root / "douzero/evaluation").glob("*.py"))]
    for source in evaluation_sources:
        text = source.read_text(encoding="utf-8")
        assert "douzero.coach" not in text
        assert "OpeningSampler" not in text
