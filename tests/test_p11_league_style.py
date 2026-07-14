"""P11 policy-league, learner-only replay, and public-style tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel, build_belief_input
from douzero.config import load_config
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.league import (
    LeagueManifest,
    LoadedPolicySelector,
    MatchupLogger,
    PolicyEntry,
    PolicyLoaderContract,
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
from douzero.observation.seats import ALL_ROLES
from douzero.style import (
    STYLE_FEATURE_WIDTH,
    STYLE_LAYOUT_HASH,
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


def _entry(
    policy_id: str,
    *,
    tags=(),
    paths=None,
    model_version="v2",
    model_config_hash="model-config-v2",
    style_layout_hash="",
    checkpoint_kind="training_checkpoint",
):
    return PolicyEntry(
        policy_id=policy_id,
        checkpoint_paths_by_role=paths or {},
        model_version=model_version,
        ruleset_hash=RuleSet.legacy().stable_hash(),
        feature_schema_hash=build_v2_schema().stable_hash(),
        model_config_hash=model_config_hash,
        model_config_identity_version=3,
        checkpoint_kind=checkpoint_kind,
        objective="adp",
        created_step=10,
        style_layout_hash=style_layout_hash,
        rating=1.0,
        tags=tuple(tags),
    )


def _loader(entry: PolicyEntry, name: str | None = None) -> PolicyLoaderContract:
    return PolicyLoaderContract.from_policy(
        entry, loader_name=name or f"{entry.model_version}-loader"
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
        runtime_loader=_loader(current),
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
    changed = dict(first[0].policy_ids_by_seat)
    changed["landlord"] = "tampered"
    object.__setattr__(first[0], "policy_ids_by_seat", MappingProxyType(changed))
    with pytest.raises(RuntimeError, match="changed during a game"):
        first[0].assert_unchanged(first[0].bundle_hash)


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
        feature_schema_hash=current.feature_schema_hash,
        model_config_hash=current.model_config_hash,
        model_config_identity_version=current.model_config_identity_version,
        checkpoint_kind=current.checkpoint_kind,
        objective="adp",
        created_step=1,
    )
    manifest = LeagueManifest((current, missing, incompatible), current.policy_id)
    with pytest.warns(RuntimeWarning):
        pool = PolicyPool(
            manifest,
            current,
            runtime_loader=_loader(current),
            runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
            config=PolicyPoolConfig(mode="population", include_random_agent=True),
        )
    assert [policy.policy_id for policy in pool.candidates] == ["builtin-random"]


@pytest.mark.parametrize(
    ("changed_field", "value"),
    (
        ("model_config_hash", "same-shape-different-semantics"),
        ("style_layout_hash", STYLE_LAYOUT_HASH),
    ),
)
def test_v2_policy_identity_drift_is_rejected(tmp_path, changed_field, value):
    current = _entry("current", tags=("current",))
    paths = {}
    for role in ALL_ROLES:
        path = tmp_path / f"{role}.ckpt"
        path.write_bytes(b"weights")
        paths[role] = str(path)
    candidate = _entry("incompatible-v2", paths=paths, **{changed_field: value})
    with pytest.warns(RuntimeWarning, match=changed_field):
        pool = PolicyPool(
            LeagueManifest((current, candidate), current.policy_id),
            current,
            runtime_loader=_loader(current),
            runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
            config=PolicyPoolConfig(mode="single", include_random_agent=False),
        )
    assert pool.candidates == ()


def test_loaded_selector_must_use_registered_policy_loader(tmp_path):
    current = _entry("current", tags=("current",))
    paths = {}
    for role in ALL_ROLES:
        path = tmp_path / f"{role}.ckpt"
        path.write_bytes(b"weights")
        paths[role] = str(path)
    historical = _entry("historical", paths=paths)
    loader = _loader(current)
    pool = PolicyPool(
        LeagueManifest((current, historical), current.policy_id),
        current,
        runtime_loader=loader,
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(mode="population", include_random_agent=False),
    )
    wrong = PolicyLoaderContract.from_policy(
        _entry("style-enabled", style_layout_hash=STYLE_LAYOUT_HASH),
        loader_name="wrong-style-loader",
    )
    with pytest.raises(ValueError, match="loaded by"):
        PopulationEpisodeRunner(
            pool,
            lambda obs: 0,
            opponent_selectors={
                historical.policy_id: LoadedPolicySelector(
                    historical.policy_id, wrong, lambda obs: 0
                )
            },
        )


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
        runtime_loader=_loader(current),
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        opponent_loaders={
            "legacy": _loader(legacy, "legacy-position-loader"),
            "bc": _loader(bc, "bc-policy-loader"),
        },
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
        runtime_loader=_loader(current),
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
        runtime_loader=_loader(current),
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(mode="single", include_random_agent=False),
    )
    bundle = pool.sample_bundle(0)
    assert set(bundle.learner_controlled_seats) == {
        "landlord", "landlord_down", "landlord_up"
    }
    assert set(bundle.policy_ids_by_seat.values()) == {current.policy_id}


def test_population_trainer_runs_bounded_learner_update():
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            history_heads=1,
        ),
    )
    runtime_loader = PolicyLoaderContract.for_v2_runtime(
        model.schema,
        model.config,
        checkpoint_kind="training_checkpoint",
    )
    current = _entry(
        "current",
        tags=("current",),
        model_config_hash=model.config.stable_hash(),
    )
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_loader=runtime_loader,
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(
            mode="population", seed=7, learner_seats_per_game=1
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


def test_population_trainer_rejects_manifest_identity_not_matching_live_model():
    current = _entry(
        "current",
        tags=("current",),
        model_config_hash="manifest-can-not-assert-runtime-identity",
    )
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_loader=_loader(current),
        runtime_ruleset_hash=RuleSet.legacy().stable_hash(),
        config=PolicyPoolConfig(mode="single", include_random_agent=False),
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
    with pytest.raises(ValueError, match="V2Trainer model/schema identity"):
        V2Trainer(model, policy_pool=pool)


def test_historical_weights_load_into_frozen_clone_only():
    learner = torch.nn.Linear(3, 2)
    policy = _entry("historical")
    loader = _loader(policy)
    before = copy.deepcopy(learner.state_dict())
    historical_state = {
        "weight": torch.full_like(learner.weight, 9.0),
        "bias": torch.full_like(learner.bias, -3.0),
    }
    historical = build_frozen_policy_model(
        learner,
        historical_state,
        policy=policy,
        loader_contract=loader,
        runtime_contract=loader,
    )
    assert all(torch.equal(learner.state_dict()[key], value) for key, value in before.items())
    assert torch.equal(historical.weight, historical_state["weight"])
    assert not any(parameter.requires_grad for parameter in historical.parameters())

    incompatible_runtime = PolicyLoaderContract.from_policy(
        _entry("other", model_config_hash="same-shape-different-semantics"),
        loader_name="runtime-loader",
    )
    with pytest.raises(ValueError, match="model_config_hash"):
        build_frozen_policy_model(
            learner,
            historical_state,
            policy=policy,
            loader_contract=loader,
            runtime_contract=incompatible_runtime,
        )


def test_snapshot_registers_only_complete_files_and_preserves_pinned(tmp_path):
    manifest_path = tmp_path / "league.json"
    manager = SnapshotManager(
        str(manifest_path),
        snapshot_root=str(tmp_path / "snapshots"),
        retention=SnapshotRetention(keep_recent=1, keep_top_rated=0),
        interval_steps=10,
    )
    assert not manager.should_snapshot(9, 0)
    assert manager.should_snapshot(10, 9)
    entry = _entry(
        "snapshot-10",
        tags=("pinned",),
        paths=manager.checkpoint_paths("snapshot-10"),
    )
    with pytest.raises(RuntimeError, match="complete file"):
        manager.write_and_register(
            entry,
            {role: (lambda path: None) for role in entry.checkpoint_paths_by_role},
        )
    assert not manifest_path.exists()

    manifest = manager.write_and_register(
        entry,
        {
            role: (lambda path: Path(path).write_bytes(b"checkpoint"))
            for role in entry.checkpoint_paths_by_role
        },
        make_primary=True,
    )
    assert manifest.get(entry.policy_id).policy_id == entry.policy_id
    assert all(
        Path(path).read_bytes() == b"checkpoint"
        for path in entry.checkpoint_paths_by_role.values()
    )


def test_retention_refuses_paths_outside_snapshot_root(tmp_path):
    manager = SnapshotManager(
        str(tmp_path / "league.json"),
        snapshot_root=str(tmp_path / "snapshots"),
        retention=SnapshotRetention(keep_recent=0, keep_top_rated=0),
    )
    current = _entry("current", tags=("current",))
    outside_paths = {}
    for role in ALL_ROLES:
        path = tmp_path / "outside" / f"{role}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"important")
        outside_paths[role] = str(path)
    victim = _entry(
        "victim",
        tags=("managed-snapshot",),
        paths=outside_paths,
    )
    manifest = LeagueManifest((current, victim), current.policy_id)
    with pytest.raises(ValueError, match="manager-owned layout"):
        manager.apply_retention(manifest)
    assert all(Path(path).read_bytes() == b"important" for path in outside_paths.values())


def test_retention_refuses_symlink_checkpoint(tmp_path):
    manager = SnapshotManager(
        str(tmp_path / "league.json"),
        snapshot_root=str(tmp_path / "snapshots"),
        retention=SnapshotRetention(keep_recent=0, keep_top_rated=0),
    )
    current = _entry("current", tags=("current",))
    paths = manager.checkpoint_paths("victim")
    target = tmp_path / "outside.ckpt"
    target.write_bytes(b"important")
    for role, raw_path in paths.items():
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == ALL_ROLES[0]:
            path.symlink_to(target)
        else:
            path.write_bytes(b"checkpoint")
    victim = _entry(
        "victim",
        tags=("managed-snapshot",),
        paths=paths,
    )
    with pytest.raises(ValueError, match="outside snapshot_root|symlink"):
        manager.apply_retention(LeagueManifest((current, victim), current.policy_id))
    assert target.read_bytes() == b"important"


def test_retention_failure_keeps_tombstone_and_recovers(tmp_path, monkeypatch):
    manifest_path = tmp_path / "league.json"
    manager = SnapshotManager(
        str(manifest_path),
        snapshot_root=str(tmp_path / "snapshots"),
        retention=SnapshotRetention(keep_recent=0, keep_top_rated=0),
    )
    current = _entry("current", tags=("current",))
    paths = manager.checkpoint_paths("victim")
    for raw_path in paths.values():
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    victim = _entry(
        "victim",
        tags=("managed-snapshot",),
        paths=paths,
    )
    original_delete = manager._delete_managed_checkpoint
    calls = 0

    def flaky_delete(item, role, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected retention failure")
        original_delete(item, role, path)

    monkeypatch.setattr(manager, "_delete_managed_checkpoint", flaky_delete)
    with pytest.raises(OSError, match="injected retention failure"):
        manager.apply_retention(LeagueManifest((current, victim), current.policy_id))

    on_disk = LeagueManifest.load(manifest_path)
    assert {policy.policy_id for policy in on_disk.policies} == {current.policy_id}
    assert [item.policy_id for item in on_disk.pending_deletes] == [victim.policy_id]
    assert any(not Path(path).exists() for path in paths.values())

    monkeypatch.setattr(manager, "_delete_managed_checkpoint", original_delete)
    recovered = manager.load()
    assert recovered.pending_deletes == ()
    assert all(not Path(path).exists() for path in paths.values())


def test_three_role_snapshot_failure_never_publishes_partial_bundle(tmp_path):
    manifest_path = tmp_path / "league.json"
    manager = SnapshotManager(
        str(manifest_path),
        snapshot_root=str(tmp_path / "snapshots"),
    )
    stable = _entry(
        "stable",
        tags=("pinned",),
        paths=manager.checkpoint_paths("stable"),
    )
    writers = {
        role: (lambda path: Path(path).write_bytes(b"old"))
        for role in ALL_ROLES
    }
    manager.write_and_register(stable, writers, make_primary=True)

    partial = _entry("partial", paths=manager.checkpoint_paths("partial"))

    def writer_for(role):
        def write(path):
            if role == ALL_ROLES[1]:
                raise RuntimeError("injected role writer failure")
            Path(path).write_bytes(b"new")

        return write

    with pytest.raises(RuntimeError, match="injected role writer failure"):
        manager.write_and_register(
            partial,
            {role: writer_for(role) for role in ALL_ROLES},
        )
    assert not (manager.snapshot_root / "policies" / partial.policy_id).exists()
    assert all(
        Path(path).read_bytes() == b"old"
        for path in stable.checkpoint_paths_by_role.values()
    )
    assert {policy.policy_id for policy in manager.load().policies} == {stable.policy_id}

    with pytest.raises(FileExistsError, match="immutable snapshot"):
        manager.write_and_register(stable, writers)
    assert all(
        Path(path).read_bytes() == b"old"
        for path in stable.checkpoint_paths_by_role.values()
    )


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
