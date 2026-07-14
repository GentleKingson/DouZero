"""P11 policy-league, learner-only replay, and public-style tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel, build_belief_input
from douzero.config import load_config
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.league import (
    LeagueManifest,
    MatchupLogger,
    PolicyEntry,
    PolicyPool,
    PolicyPoolConfig,
    PopulationEpisodeRunner,
    PromotionEvaluation,
    PromotionGate,
    SnapshotManager,
    SnapshotRetention,
    build_frozen_policy_model,
)
from douzero.models_v2 import ModelV2, ModelV2Config, observation_to_model_inputs
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.observation.public import build_public_observation
from douzero.style import (
    STYLE_FEATURE_WIDTH,
    StyleEncoder,
    build_style_features,
)
from douzero.training.v2_buffer import V2ReplayBuffer
from douzero.training.v2_trainer import TrainerConfig, V2Trainer


def _public(action_history=()):
    return build_public_observation(
        acting_role="landlord",
        my_handcards=[3, 3, 4],
        other_handcards=[5, 6],
        played_cards={
            "landlord": [],
            "landlord_down": [17, 17, 5, 5, 5, 5],
            "landlord_up": [6],
        },
        last_move=[],
        last_move_dict={},
        three_landlord_cards=[],
        num_cards_left={
            "landlord": 3,
            "landlord_down": 11,
            "landlord_up": 16,
        },
        legal_actions=[[]],
        action_history=action_history,
        ruleset_hash=RuleSet.legacy().stable_hash(),
    )


def _entry(policy_id: str, *, tags=(), paths=None, model_version="v2"):
    return PolicyEntry(
        policy_id=policy_id,
        checkpoint_paths_by_role=paths or {},
        model_version=model_version,
        ruleset_hash=RuleSet.legacy().stable_hash(),
        objective="adp",
        created_step=10,
        rating=1.0,
        tags=tuple(tags),
    )


def _real_obs(seed: int = 11, steps: int = 4):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(steps):
        env.step(env.infoset.legal_actions[0])
    return get_obs_v2(env.infoset, ruleset=RuleSet.legacy())


def test_style_features_use_public_action_history_and_cold_start():
    cold = build_style_features(_public())
    assert cold.shape == (STYLE_FEATURE_WIDTH,)
    assert np.count_nonzero(cold) == 0

    # landlord -> landlord_down -> landlord_up, repeated. Down passes once,
    # spends two 2s, then later reuses rank 5 (public split tendency).
    history = ([3], [], [6], [4], [17, 17], [], [7], [5], [8], [9], [5], [])
    features = build_style_features(_public(history))
    down, up = features.reshape(2, -1)
    assert down[0] == 1.0 and up[0] == 1.0
    assert down[2] > 0.0
    assert down[3] > 0.0
    assert down[5] > 0.0

    encoder = StyleEncoder(output_dim=12, hidden_dim=8)
    encoded = encoder(torch.from_numpy(cold))
    assert encoded.shape == (12,)
    assert torch.isfinite(encoded).all()


def test_style_has_no_hidden_allocation_or_identity_dependency():
    history = ([3], [], [6], [4], [17], [])
    first = _public(history)
    second = _public(history)
    assert np.array_equal(
        build_style_features(first), build_style_features(second)
    )


def test_style_fuses_into_value_and_belief_models():
    obs = _real_obs()
    cfg = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        style_enabled=True,
        style_embedding_dim=8,
    )
    model = ModelV2(build_v2_schema(), cfg)
    bundle = observation_to_model_inputs(obs, style_enabled=True)
    out = model(
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
        bundle.acting_role,
        style_features=bundle.style_features,
    )
    out.win_logit.mean().backward()
    assert model.style_encoder is not None
    assert any(parameter.grad is not None for parameter in model.style_encoder.parameters())

    binput = build_belief_input(obs.public)
    belief = BeliefModel(BeliefConfig(
        hidden_size=16,
        num_layers=1,
        style_enabled=True,
        style_embedding_dim=8,
    ))
    bout = belief(binput)
    assert bout.logits.shape == (1, 15, 5)
    assert torch.isfinite(bout.logits).all()


def test_style_disabled_preserves_existing_model_identity_and_contract():
    base = ModelV2Config(hidden_size=32, history_layers=1, history_heads=4)
    disabled = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        style_enabled=False,
        style_embedding_dim=7,
    )
    assert base.stable_hash() == disabled.stable_hash()
    model = ModelV2(build_v2_schema(), disabled)
    assert model.style_encoder is None


def test_style_checkpoint_round_trip_and_disabled_runtime_rejection(tmp_path):
    from douzero.checkpoint import (
        CheckpointCompatibilityError,
        load_v2_checkpoint,
        save_v2_checkpoint,
    )

    schema = build_v2_schema()
    cfg = ModelV2Config(
        hidden_size=16,
        history_encoder="lstm",
        history_layers=1,
        history_heads=1,
        style_enabled=True,
        style_embedding_dim=8,
    )
    model = ModelV2(schema, cfg)
    path = str(tmp_path / "style-model.tar")
    save_v2_checkpoint(path, model, ruleset=RuleSet.legacy())
    state, _manifest = load_v2_checkpoint(
        path,
        expected_schema_hash=schema.stable_hash(),
        expected_model_config_hash=cfg.stable_hash(),
        expected_ruleset=RuleSet.legacy(),
        runtime_model_config=cfg,
    )
    restored = ModelV2(schema, cfg)
    restored.load_state_dict(state, strict=True)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in restored.state_dict().items()
    )

    disabled = ModelV2Config(
        hidden_size=16,
        history_encoder="lstm",
        history_layers=1,
        history_heads=1,
    )
    with pytest.raises(CheckpointCompatibilityError, match="model_config_hash"):
        load_v2_checkpoint(
            path,
            expected_schema_hash=schema.stable_hash(),
            expected_model_config_hash=disabled.stable_hash(),
            expected_ruleset=RuleSet.legacy(),
            runtime_model_config=disabled,
        )


def test_league_manifest_round_trip_and_strict_fields(tmp_path):
    current = _entry("current-10", tags=("current",))
    manifest = LeagueManifest().upsert(current, make_primary=True)
    path = tmp_path / "league.json"
    manifest.save(path)
    assert LeagueManifest.load(path) == manifest
    assert not list(tmp_path.glob("*.tmp"))


def test_policy_sampling_is_seeded_and_rotates_learner_seat():
    current = _entry("current-10", tags=("current",))
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_model_version="v2",
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(
            mode="population", seed=123, learner_seats_per_game=1
        ),
    )
    first = [pool.sample_bundle(i) for i in range(6)]
    second = [pool.sample_bundle(i) for i in range(6)]
    assert [bundle.bundle_hash for bundle in first] == [
        bundle.bundle_hash for bundle in second
    ]
    assert len({bundle.learner_controlled_seats[0] for bundle in first}) > 1
    assert all(len(bundle.policy_ids_by_seat) == 3 for bundle in first)
    with pytest.raises(TypeError):
        first[0].policy_ids_by_seat["landlord"] = "changed"


def test_missing_or_incompatible_policies_fail_safely(tmp_path):
    current = _entry("current", tags=("current",))
    missing = _entry(
        "missing", paths={"landlord": str(tmp_path / "missing.pt")}
    )
    incompatible = PolicyEntry(
        policy_id="wrong-rules",
        checkpoint_paths_by_role={},
        model_version="v2",
        ruleset_hash="not-the-runtime-rules",
        objective="adp",
        created_step=1,
    )
    manifest = LeagueManifest((current, missing, incompatible), current.policy_id)
    with pytest.warns(RuntimeWarning):
        pool = PolicyPool(
            manifest,
            current,
            runtime_model_version="v2",
            runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
            config=PolicyPoolConfig(mode="population", include_random_agent=True),
        )
    assert [policy.policy_id for policy in pool.candidates] == ["builtin-random"]


def test_legacy_wp_adp_and_bc_opponents_are_supported(tmp_path):
    current = _entry("current", tags=("current",))
    paths = {}
    for role in ("landlord", "landlord_down", "landlord_up"):
        path = tmp_path / f"{role}.pt"
        path.write_bytes(b"weights")
        paths[role] = str(path)
    legacy = _entry(
        "legacy-wp",
        tags=("legacy-wp",),
        paths=paths,
        model_version="legacy",
    )
    bc = _entry("bc-prior", tags=("bc",), paths=paths, model_version="bc")
    pool = PolicyPool(
        LeagueManifest((current, legacy, bc), current.policy_id),
        current,
        runtime_model_version="v2",
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(
            mode="population", include_random_agent=False
        ),
    )
    assert {policy.policy_id for policy in pool.candidates} == {
        "legacy-wp", "bc-prior"
    }


def test_population_runner_records_only_learner_decisions_and_teammate(tmp_path):
    current = _entry("current", tags=("current",))
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_model_version="v2",
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(
            mode="population", seed=41, learner_seats_per_game=1
        ),
    )
    log_path = tmp_path / "matchups.jsonl"
    runner = PopulationEpisodeRunner(
        pool, lambda obs: 0, logger=MatchupLogger(str(log_path))
    )
    episode, record = runner.run(0)
    learner_seat = record.learner_controlled_seats[0]
    assert episode.transitions
    assert {transition.position for transition in episode.transitions} == {learner_seat}
    assert all(transition.policy_id == current.policy_id for transition in episode.transitions)
    if learner_seat != "landlord":
        assert all(
            transition.teammate_policy_id is not None
            for transition in episode.transitions
        )
    buffer = V2ReplayBuffer()
    buffer.add_episode(episode)
    assert len(buffer) == len(episode.transitions)
    assert record.policy_bundle_hash in log_path.read_text()


def test_single_mode_keeps_legacy_all_seat_self_play():
    current = _entry("current", tags=("current",))
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_model_version="v2",
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(mode="single", include_random_agent=False),
    )
    bundle = pool.sample_bundle(0)
    assert set(bundle.learner_controlled_seats) == {
        "landlord", "landlord_down", "landlord_up"
    }
    assert set(bundle.policy_ids_by_seat.values()) == {current.policy_id}


def test_population_trainer_runs_bounded_learner_update():
    current = _entry("current", tags=("current",))
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_model_version="v2",
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(
            mode="population", seed=7, learner_seats_per_game=1
        ),
    )
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            history_heads=1,
        ),
    )
    trainer = V2Trainer(
        model,
        config=TrainerConfig(
            seed=7,
            rng_seed=7,
            max_episodes=1,
            batch_size=2,
            optimizer_steps=1,
            exp_epsilon=0.0,
        ),
        policy_pool=pool,
    )
    before = copy.deepcopy(model.state_dict())
    trainer.collect_episodes(1)
    loss = trainer.step()
    assert loss is not None
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    assert {
        transition.position
        for episode in trainer.buffer._episodes
        for transition in episode.transitions
    } == set(trainer.buffer._episodes[0].learner_controlled_seats)


def test_historical_weights_load_into_frozen_clone_only():
    learner = torch.nn.Linear(3, 2)
    before = copy.deepcopy(learner.state_dict())
    historical_state = {
        "weight": torch.full_like(learner.weight, 9.0),
        "bias": torch.full_like(learner.bias, -3.0),
    }
    historical = build_frozen_policy_model(learner, historical_state)
    assert all(torch.equal(learner.state_dict()[key], value) for key, value in before.items())
    assert torch.equal(historical.weight, historical_state["weight"])
    assert not any(parameter.requires_grad for parameter in historical.parameters())


def test_snapshot_registers_only_complete_files_and_preserves_pinned(tmp_path):
    manifest_path = tmp_path / "league.json"
    manager = SnapshotManager(
        str(manifest_path),
        retention=SnapshotRetention(keep_recent=1, keep_top_rated=0),
        interval_steps=10,
    )
    assert not manager.should_snapshot(9, 0)
    assert manager.should_snapshot(10, 9)
    checkpoint = tmp_path / "policy.pt"
    entry = _entry(
        "snapshot-10",
        tags=("pinned",),
        paths={"landlord": str(checkpoint)},
    )
    with pytest.raises(RuntimeError, match="complete file"):
        manager.write_and_register(entry, {"landlord": lambda path: None})
    assert not manifest_path.exists()

    manifest = manager.write_and_register(
        entry,
        {"landlord": lambda path: Path(path).write_bytes(b"checkpoint")},
        make_primary=True,
    )
    assert manifest.get(entry.policy_id).policy_id == entry.policy_id
    assert checkpoint.read_bytes() == b"checkpoint"


def test_promotion_gate_requires_p15_ci_and_records_threshold(tmp_path):
    gate = PromotionGate(
        min_pairs=100,
        min_ci_lower_bound=0.01,
        audit_path=str(tmp_path / "promotion.jsonl"),
    )
    rejected = gate.decide(PromotionEvaluation(
        "candidate", "main", 200, 0.02, -0.01, 0.05
    ))
    assert not rejected.promoted
    promoted = gate.decide(PromotionEvaluation(
        "candidate", "main", 200, 0.04, 0.02, 0.06
    ))
    assert promoted.promoted
    assert (tmp_path / "promotion.jsonl").read_text().count("\n") == 2


def test_enhanced_config_carries_disabled_p11_defaults():
    cfg = load_config("configs/enhanced.yaml")
    assert not cfg.league.enabled
    assert cfg.league.mode == "single"
    assert not cfg.model.style_enabled
