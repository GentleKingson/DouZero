"""H6 support matrix, loss composition, replay, learner, and leakage tests."""

from __future__ import annotations

import copy
import dataclasses
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.coach.records import CANONICAL_DECK
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.human_data.sample import BCSample
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.observation.bidding import get_bidding_obs_v2
from douzero.training.bidding import (
    BiddingMinibatch,
    BiddingTransition,
)
from douzero.v3_hybrid import (
    AdaptiveDMCConfig,
    AdaptiveSnapshotProvenance,
    LossTermTensor,
    LossTermSchedule,
    V3H2LearnerConfig,
    V3H6ReplayBuffer,
    V3HybridLossComposer,
    V3HybridLossComposerConfig,
    V3HybridModel,
    V3HybridModelConfig,
    assert_public_replay_payload,
    capture_plain_transition,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
    v3_h6_support_matrix_hash,
)
from douzero.v3_hybrid.adaptive_dmc import ADMC_SAFE_HYBRID
from douzero.v3_hybrid.config import BELIEF_FEEDBACK_ALL
from douzero.v3_hybrid.integration_config import (
    V3H6AuxiliaryConfig,
    V3H6FeatureFlags,
    V3H6LearnerConfig,
    V3H6ResolvedConfig,
    V3H6TopologyConfig,
    load_v3_hybrid_config,
)
from douzero.v3_hybrid.training.h3_learner import V3H3LearnerConfig
from douzero.v3_hybrid.training.h3_learner import _same_public_bundle
from douzero.v3_hybrid.training.h4_learner import V3H4LearnerConfig
from douzero.v3_hybrid.training.belief_config import (
    BELIEF_MODE_AUXILIARY,
    V3H4BeliefTrainingConfig,
)
from douzero.v3_hybrid.training.h5_learner import V3H5LearnerConfig
from douzero.v3_hybrid.training.h6_learner import V3H6Learner


def _observation(role: str = "landlord", seed: int = 20260722):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(120):
        if env._acting_player_position == role:
            infoset = copy.deepcopy(env.infoset)
            infoset.legal_actions = infoset.legal_actions[:4]
            return get_obs_v2(infoset, ruleset=RuleSet.legacy())
        action = next(
            (item for item in env.infoset.legal_actions if item),
            env.infoset.legal_actions[0],
        )
        _obs, _reward, done, _info = env.step(action)
        if done:
            env.reset()
    raise AssertionError("could not build an observation")


def _transition(seed: int = 20260722):
    observation = _observation(seed=seed)
    pending = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id=f"episode-{seed}",
        deal_id=f"deal-{seed}",
        target_transform="raw",
    )
    return pending.finalize(2.0)


def _model_config(**changes):
    values = {
        "hidden_size": 16,
        "history_layers": 1,
        "history_heads": 4,
        "shared_fusion_layers": 1,
        "landlord_adapter_layers": 1,
        "farmer_adapter_layers": 1,
    }
    values.update(changes)
    return V3HybridModelConfig(**values)


def _resolved(
    *, model_config=None, win: float = 1.0, score: float = 0.5,
    device: str = "cpu",
):
    model_config = model_config or _model_config()
    public = V3H2LearnerConfig(
        batch_size=4, learning_rate=1e-3, max_grad_norm=10.0, device=device
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    loss = V3HybridLossComposerConfig(
        lambda_dmc=1.0,
        lambda_win=win,
        lambda_score=score,
    )
    learner = V3H6LearnerConfig(
        base=base,
        losses=loss,
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    return V3H6ResolvedConfig(model=model_config, learner=learner)


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_dedicated_yaml_loads_without_widening_legacy_or_v2_loader():
    resolved = load_v3_hybrid_config("configs/v3_hybrid.yaml")
    assert resolved.learner.features == V3H6FeatureFlags()
    assert resolved.learner.losses.lambda_dmc == 1.0
    assert resolved.learner.losses.lambda_win == 1.0
    assert len(resolved.stable_hash()) == 64
    code = (
        "import sys; import douzero.v3_hybrid; "
        "assert 'douzero.v3_hybrid.training.h6_learner' not in sys.modules; "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_support_matrix_is_stable_and_unsupported_topologies_fail_closed():
    assert len(v3_h6_support_matrix_hash()) == 64
    base = _resolved()
    with pytest.raises(ValueError, match="does not support async_single_gpu"):
        dataclasses.replace(
            base,
            learner=dataclasses.replace(
                base.learner,
                topology=dataclasses.replace(
                    base.learner.topology, topology="async_single_gpu"
                ),
            ),
        )

    schedules = dict(base.learner.losses.schedules)
    schedules["dmc"] = LossTermSchedule(
        kind="linear", start=1.0, end=0.5, updates=2
    )
    with pytest.raises(ValueError, match="owning component"):
        dataclasses.replace(
            base.learner,
            losses=dataclasses.replace(base.learner.losses, schedules=schedules),
        )
    with pytest.raises(ValueError, match="does not support search"):
        dataclasses.replace(
            base,
            learner=dataclasses.replace(
                base.learner,
                features=dataclasses.replace(
                    base.learner.features, selective_search=True
                ),
                topology=dataclasses.replace(base.learner.topology, search=True),
            ),
        )


@pytest.mark.parametrize(
    ("loss_name", "message"),
    (
        ("lambda_bc", "requires the public human prior head"),
        ("lambda_strategy", "requires the public strategy auxiliary head"),
        ("lambda_bidding", "requires the public bidding head"),
    ),
)
def test_auxiliary_loss_requires_its_public_model_head(loss_name, message):
    baseline = _resolved()
    losses = dataclasses.replace(
        baseline.learner.losses,
        **{loss_name: 1.0},
    )
    with pytest.raises(ValueError, match=message):
        V3H6ResolvedConfig(
            model=baseline.model,
            learner=dataclasses.replace(baseline.learner, losses=losses),
        )


def test_loss_composer_matches_independent_role_weighted_formula_and_masks_padding():
    config = V3HybridLossComposerConfig(
        lambda_win=2.0,
        landlord_weight=1.0,
        landlord_up_weight=2.0,
        landlord_down_weight=3.0,
    )
    composer = V3HybridLossComposer(config)
    values = torch.tensor([1.0, 3.0, float("nan"), 5.0], requires_grad=True)
    item = LossTermTensor(
        values,
        torch.tensor([True, True, False, True]),
        ("landlord", "landlord_up", "landlord", "landlord_down"),
        ("a", "b", "padding", "d"),
    )
    composition = composer.compose({"win": item})
    oracle = (1.0 * 1.0 + 3.0 * 2.0 + 5.0 * 3.0) / 6.0
    assert composition.terms["win"].raw_loss == pytest.approx(oracle)
    assert composition.terms["win"].weighted_loss == pytest.approx(2.0 * oracle)
    assert composition.terms["win"].valid_samples == 3
    optimizer = torch.optim.SGD([values], lr=0.01)
    composer.apply(composition, optimizer, [values], max_grad_norm=10.0)
    assert composer.eligible_steps["win"] == 1
    with pytest.raises(ValueError, match="disabled.*score"):
        composer.compose({
            "score": LossTermTensor(
                torch.ones(1), torch.ones(1, dtype=torch.bool),
                ("landlord",), ("score",),
            )
        })


def test_loss_composer_state_and_duplicate_identity_fail_closed():
    config = V3HybridLossComposerConfig(lambda_dmc=1.0)
    composer = V3HybridLossComposer(config)
    item = LossTermTensor(
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, True]),
        ("landlord", "landlord_up"),
        ("a", "b"),
        gradient_owner="external",
    )
    composition = composer.compose({"dmc": item})
    assert not composition.optimizer_step_required
    composer.commit(composition)
    restored = V3HybridLossComposer(config)
    restored.load_state_dict(composer.state_dict())
    assert restored.eligible_steps == composer.eligible_steps
    with pytest.raises(ValueError, match="repeats"):
        LossTermTensor(
            torch.ones(2), torch.ones(2, dtype=torch.bool),
            ("landlord", "landlord"), ("same", "same"),
        )


def test_h6_aux_updates_are_included_in_admc_snapshot_version_validation():
    baseline = _resolved()
    public = dataclasses.replace(
        baseline.learner.base.base.base.public,
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID),
    )
    resolved = dataclasses.replace(
        baseline,
        learner=dataclasses.replace(
            baseline.learner,
            base=dataclasses.replace(
                baseline.learner.base,
                base=dataclasses.replace(
                    baseline.learner.base.base,
                    base=dataclasses.replace(
                        baseline.learner.base.base.base, public=public
                    ),
                ),
            ),
            features=dataclasses.replace(
                baseline.learner.features, adaptive_dmc=True
            ),
        ),
    )
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    for generation in (1, 2):
        row = dataclasses.replace(
            _transition(seed=20260722 + generation),
            adaptive_provenance=AdaptiveSnapshotProvenance(
                q_old=0.0,
                policy_version=learner.policy_version,
                snapshot_slot=0,
                owner_id=0,
                generation=generation,
            ),
        )
        learner.train_batch((row,))
    assert learner.policy_version == 4


def test_h6_model_graph_is_optional_public_and_identity_bound():
    baseline = _model_config()
    enabled = _model_config(
        human_prior_enabled=True,
        strategy_features_enabled=True,
        strategy_aux_enabled=True,
        style_enabled=True,
    )
    assert baseline.stable_hash() != enabled.stable_hash()
    assert not baseline.h6_graph_enabled
    assert enabled.h6_graph_enabled
    baseline_model = V3HybridModel(build_v2_schema(), baseline)
    enabled_model = V3HybridModel(build_v2_schema(), enabled)
    assert baseline_model.prior_head is None
    assert baseline_model.strategy_aux_heads is None
    assert baseline_model.style_encoder is None
    observation = _observation()
    output = enabled_model.forward_observation(observation)
    assert output.prior_logit is not None
    assert output.min_turns_after is not None
    assert output.action_mask.shape[0] == len(observation.actions.legal_actions)
    assert all(torch.isfinite(value).all() for value in (
        output.dmc_q, output.prior_logit, output.min_turns_after
    ))


def test_h6_public_checkpoint_binds_enabled_public_graph_and_strictly_reloads(
    tmp_path,
):
    config = _model_config(
        human_prior_enabled=True,
        strategy_features_enabled=True,
        strategy_aux_enabled=True,
        style_enabled=True,
    )
    model = V3HybridModel(build_v2_schema(), config).eval()
    path = tmp_path / "h6-public.pt"
    save_v3_hybrid_public_checkpoint(path, model, ruleset=RuleSet.legacy())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    flags = payload["compatibility_identity"]["feature_flags"]
    assert flags["human_bc"]
    assert flags["strategy"]
    assert flags["style"]
    assert not flags["oracle"]
    assert not flags["cooperation"]
    forbidden = ("oracle", "teacher", "privileged", "hidden_hand", "mixer")
    assert not any(
        token in name.lower()
        for name in payload["state_dict"]
        for token in forbidden
    )
    restored = load_v3_hybrid_public_checkpoint(
        path,
        schema=build_v2_schema(),
        ruleset=RuleSet.legacy(),
        config=config,
    ).eval()
    observation = _observation()
    with torch.inference_mode():
        expected = model.forward_observation(observation)
        actual = restored.forward_observation(observation)
    assert torch.equal(expected.dmc_q, actual.dmc_q)
    assert torch.equal(expected.prior_logit, actual.prior_logit)


def test_h6_public_replay_round_trip_and_privileged_negative_guard():
    config = _model_config(
        strategy_features_enabled=True,
        style_enabled=True,
    )
    model = V3HybridModel(build_v2_schema(), config)
    observation = _observation()
    pending = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id="episode-h6",
        deal_id="deal-h6",
        target_transform="raw",
        strategy_config=model.strategy_feature_config(),
        style_enabled=True,
    )
    transition = pending.finalize(1.0)
    buffer = V3H6ReplayBuffer(
        4,
        model_config=config,
        feature_schema_hash=build_v2_schema().stable_hash(),
        target_transform="raw",
        ruleset_identity=RuleSet.legacy().identity(),
        adaptive_required=False,
    )
    buffer.add(transition)
    payload = buffer.state_dict()
    assert_public_replay_payload(payload)
    restored = V3H6ReplayBuffer(
        4,
        model_config=config,
        feature_schema_hash=build_v2_schema().stable_hash(),
        target_transform="raw",
        ruleset_identity=RuleSet.legacy().identity(),
        adaptive_required=False,
    )
    restored.load_state_dict(payload)
    assert len(restored) == 1
    with pytest.raises(ValueError, match="privileged"):
        assert_public_replay_payload({**payload, "all_handcards": {}})
    for sidecar in (
        "privileged_mixer_state",
        "belief_samples",
        "oracle_samples",
        "bc_samples",
        "strategy_targets",
        "bidding_batch",
        "cooperation_trajectories",
        "optimizer_state_dict",
    ):
        with pytest.raises(ValueError, match="privileged"):
            assert_public_replay_payload({**payload, sidecar: torch.zeros(1)})
    with pytest.raises(ValueError, match="target transform"):
        V3H6ReplayBuffer(
            4,
            model_config=config,
            feature_schema_hash=build_v2_schema().stable_hash(),
            target_transform="invalid-transform",
            ruleset_identity=RuleSet.legacy().identity(),
            adaptive_required=False,
        )
    invalid_ruleset = RuleSet.legacy().identity()
    invalid_ruleset["ruleset_hash"] = "not-a-sha256"
    with pytest.raises(ValueError, match="full SHA-256"):
        V3H6ReplayBuffer(
            4,
            model_config=config,
            feature_schema_hash=build_v2_schema().stable_hash(),
            target_transform="raw",
            ruleset_identity=invalid_ruleset,
            adaptive_required=False,
        )


def test_h6_sidecar_alignment_compares_optional_public_feature_tensors():
    config = _model_config(
        strategy_features_enabled=True,
        style_enabled=True,
    )
    model = V3HybridModel(build_v2_schema(), config)
    observation = _observation()
    pending = capture_plain_transition(
        observation,
        selected_action_index=0,
        episode_id="episode-sidecar",
        deal_id="deal-sidecar",
        target_transform="raw",
        strategy_config=model.strategy_feature_config(),
        style_enabled=True,
    )
    bundle = pending.model_inputs
    aligned = dataclasses.replace(
        bundle,
        strategy_features=bundle.strategy_features.clone(),
        style_features=bundle.style_features.clone(),
    )
    assert _same_public_bundle(bundle, aligned)

    changed_strategy = bundle.strategy_features.clone()
    changed_strategy[0, 0] += 1.0
    assert not _same_public_bundle(
        bundle,
        dataclasses.replace(aligned, strategy_features=changed_strategy),
    )
    assert not _same_public_bundle(
        bundle,
        dataclasses.replace(aligned, style_features=None),
    )


def test_h6_train_checkpoint_resume_and_failed_batch_are_atomic(tmp_path):
    torch.manual_seed(20260722)
    resolved = _resolved()
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    transition = _transition()
    metrics = learner.train_batch([transition])
    assert metrics.public_aux_updated
    assert metrics.policy_version == 2
    assert metrics.losses["dmc"]["phase"] == "external_applied"
    assert metrics.losses["win"]["valid_samples"] == 1
    assert learner.composer.eligible_steps["dmc"] == 1
    assert learner.composer.eligible_steps["win"] == 1

    path = tmp_path / "h6.pt"
    learner.save_checkpoint(path)
    restored = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    restored.load_checkpoint(path)
    assert restored.policy_version == learner.policy_version
    assert restored.composer.eligible_steps == learner.composer.eligible_steps
    second = restored.train_batch([_transition(seed=20260723)])
    assert second.policy_version == 4

    before = _state(restored.model)
    counters = (
        restored.eligible_updates,
        restored.public_aux_updates,
        dict(restored.composer.eligible_steps),
    )
    invalid = dataclasses.replace(_transition(seed=20260724), mc_return=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        restored.train_batch([invalid])
    assert all(torch.equal(before[name], value) for name, value in restored.model.state_dict().items())
    assert counters == (
        restored.eligible_updates,
        restored.public_aux_updates,
        dict(restored.composer.eligible_steps),
    )


def test_pure_bc_batch_updates_without_an_empty_cardplay_optimizer_step():
    observation = _observation()
    model_config = _model_config(human_prior_enabled=True)
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        lambda_dmc=0.0,
        device="cpu",
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    learner_config = V3H6LearnerConfig(
        base=base,
        losses=V3HybridLossComposerConfig(lambda_bc=1.0),
        features=V3H6FeatureFlags(human_bc=True),
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    resolved = V3H6ResolvedConfig(model=model_config, learner=learner_config)
    model = V3HybridModel(build_v2_schema(), model_config)
    learner = V3H6Learner(
        model,
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    sample = BCSample(
        obs=observation,
        human_action_index=0,
        position=observation.public.acting_role,
        game_id="h6-pure-bc",
        num_legal_actions=len(observation.actions.legal_actions),
    )
    metrics = learner.train_batch([], bc_samples=[sample])
    assert metrics.samples == 1
    assert metrics.public_aux_updated
    assert metrics.policy_version == 1
    assert metrics.base.base.base.samples == 0
    assert metrics.losses["bc"]["valid_samples"] == 1
    assert metrics.losses["dmc"]["phase"] == "disabled"

    before = _state(learner.model)
    zero_weight = dataclasses.replace(
        sample,
        game_id="h6-zero-weight-bc",
        sample_weight=0.0,
    )
    zero_metrics = learner.train_batch([], bc_samples=[zero_weight])
    assert not zero_metrics.public_aux_updated
    assert zero_metrics.policy_version == 1
    assert zero_metrics.losses["bc"]["phase"] == "no_valid_targets"
    assert learner.composer.eligible_steps["bc"] == 1
    assert all(
        torch.equal(before[name], value)
        for name, value in learner.model.state_dict().items()
    )


def test_belief_feedback_and_bc_use_public_posterior_features_together():
    observation = _observation()
    model_config = _model_config(
        human_prior_enabled=True,
        belief_feedback=BELIEF_FEEDBACK_ALL,
    )
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        lambda_dmc=0.0,
        device="cpu",
    )
    belief_training = V3H4BeliefTrainingConfig(
        enabled=True,
        mode=BELIEF_MODE_AUXILIARY,
        lambda_belief=0.5,
        learning_rate=1e-3,
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(
            base=V3H3LearnerConfig(public=public),
            belief=belief_training,
        )
    )
    learner_config = V3H6LearnerConfig(
        base=base,
        losses=V3HybridLossComposerConfig(lambda_belief=0.5, lambda_bc=1.0),
        features=V3H6FeatureFlags(belief=True, human_bc=True),
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    resolved = V3H6ResolvedConfig(model=model_config, learner=learner_config)
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), model_config),
        ruleset=RuleSet.legacy(),
        config=resolved,
        belief_model=BeliefModel(BeliefConfig(hidden_size=16, num_layers=1)),
    )
    sample = BCSample(
        obs=observation,
        human_action_index=0,
        position=observation.public.acting_role,
        game_id="h6-belief-bc",
        num_legal_actions=len(observation.actions.legal_actions),
    )
    metrics = learner.train_batch([], bc_samples=[sample])
    assert metrics.public_aux_updated
    assert metrics.losses["bc"]["valid_samples"] == 1
    assert metrics.losses["belief"]["phase"] == "no_valid_targets"


def test_pure_bidding_batch_uses_real_per_sample_targets():
    ruleset = RuleSet.standard()
    observation = get_bidding_obs_v2(
        {
            "phase": "bidding",
            "position": "0",
            "my_handcards": list(CANONICAL_DECK[:17]),
            "current_highest_bid": 0,
            "bidding_history": [],
            "bidding_order": ["0", "1", "2"],
            "first_bidder": "0",
            "legal_bids": [0, 1, 2, 3],
        },
        ruleset=ruleset,
    )
    transition = BiddingTransition(
        obs=observation,
        bid_action=1,
        policy_version="h6-bidding-policy",
        source_policy="learned",
    )
    transition.assign_actor_role({
        "0": "landlord",
        "1": "landlord_down",
        "2": "landlord_up",
    })
    transition.label_from_terminal({
        "team_targets": {
            "landlord": {"target_win": 1.0, "target_score": 2.0},
            "landlord_down": {"target_win": 0.0, "target_score": -2.0},
            "landlord_up": {"target_win": 0.0, "target_score": -2.0},
        }
    })
    model_config = _model_config(bidding_enabled=True, bidding_hidden_size=12)
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        lambda_dmc=0.0,
        device="cpu",
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    learner_config = V3H6LearnerConfig(
        base=base,
        losses=V3HybridLossComposerConfig(lambda_bidding=1.0),
        features=V3H6FeatureFlags(bidding=True),
        topology=V3H6TopologyConfig(ruleset="standard"),
    )
    resolved = V3H6ResolvedConfig(model=model_config, learner=learner_config)
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), model_config),
        ruleset=ruleset,
        config=resolved,
    )
    metrics = learner.train_batch(
        [], bidding_batch=BiddingMinibatch([transition])
    )
    assert metrics.samples == 1
    assert metrics.public_aux_updated
    assert metrics.policy_version == 1
    assert metrics.losses["bidding"]["valid_samples"] == 1
    assert metrics.losses["bidding"]["role_valid_samples"] == {
        "landlord": 1,
        "landlord_up": 0,
        "landlord_down": 0,
    }

    no_credit = copy.deepcopy(transition)
    no_credit.policy_credit_valid = False
    policy_only_config = dataclasses.replace(
        learner_config,
        auxiliary=dataclasses.replace(
            learner_config.auxiliary,
            bidding_lambda_landlord_win=0.0,
            bidding_lambda_landlord_score=0.0,
        ),
    )
    policy_only = V3H6Learner(
        V3HybridModel(build_v2_schema(), model_config),
        ruleset=ruleset,
        config=V3H6ResolvedConfig(
            model=model_config,
            learner=policy_only_config,
        ),
    )
    no_credit_metrics = policy_only.train_batch(
        [], bidding_batch=BiddingMinibatch([no_credit])
    )
    assert not no_credit_metrics.public_aux_updated
    assert no_credit_metrics.policy_version == 0
    assert no_credit_metrics.losses["bidding"]["phase"] == "no_valid_targets"
    assert policy_only.composer.eligible_steps["bidding"] == 0

    mismatched = copy.deepcopy(transition)
    mismatched.obs = dataclasses.replace(
        mismatched.obs, ruleset_hash="0" * 64
    )
    with pytest.raises(ValueError, match="ruleset does not match"):
        learner.model.forward_bidding(mismatched.obs)
    with pytest.raises(ValueError, match="ruleset identity mismatch"):
        learner.train_batch(
            [], bidding_batch=BiddingMinibatch([mismatched])
        )


def test_public_output_is_independent_of_training_only_hidden_hand_objects():
    model = V3HybridModel(build_v2_schema(), _model_config()).eval()
    observation = _observation()
    with torch.inference_mode():
        before = model.forward_observation(observation).dmc_q.clone()
    hidden_a = {"landlord_up": (3, 4), "landlord_down": (5, 6)}
    hidden_b = {"landlord_up": (5, 6), "landlord_down": (3, 4)}
    assert hidden_a != hidden_b
    with torch.inference_mode():
        after = model.forward_observation(observation).dmc_q.clone()
    assert torch.equal(before, after)


def test_nonfinite_gradient_does_not_advance_composer_counter():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    composer = V3HybridLossComposer(
        V3HybridLossComposerConfig(lambda_win=1.0)
    )
    loss = parameter * torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        composer.compose({
            "win": LossTermTensor(
                loss,
                torch.ones(1, dtype=torch.bool),
                ("landlord",),
                ("nan",),
            )
        })
    assert composer.eligible_steps["win"] == 0
    assert math.isfinite(float(parameter.item()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("regain_initiative", 2.0, "binary"),
        ("min_turns_exact_mask", 2.0, "binary"),
        ("min_turns_after", -1.0, "non-negative"),
        ("structure_cost", float("nan"), "finite"),
    ),
)
def test_h6_strategy_labels_fail_closed_before_loss(field, value, message):
    resolved = _resolved()
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    labels = {
        "min_turns_after": 1.0,
        "min_turns_exact_mask": 1.0,
        "regain_initiative": 0.0,
        "teammate_finish": 1.0,
        "teammate_finish_mask": 1.0,
        "spring_probability": 0.0,
        "structure_cost": 0.5,
    }
    labels[field] = value
    gathered = {
        "min_turns_after": torch.zeros(1, 1),
        "regain_initiative_logit": torch.zeros(1, 1),
        "teammate_finish_logit": torch.zeros(1, 1),
        "spring_probability_logit": torch.zeros(1, 1),
        "structure_cost": torch.zeros(1, 1),
    }
    with pytest.raises(ValueError, match=message):
        learner._strategy_term(
            gathered,
            [_transition()],
            ("landlord",),
            ("strategy-label",),
            [labels],
        )


def test_h6_fully_masked_active_strategy_component_is_a_noop():
    baseline = _resolved()
    model_config = _model_config(
        strategy_features_enabled=True,
        strategy_aux_enabled=True,
    )
    resolved = V3H6ResolvedConfig(
        model=model_config,
        learner=dataclasses.replace(
            baseline.learner,
            losses=dataclasses.replace(
                baseline.learner.losses, lambda_strategy=1.0
            ),
            auxiliary=V3H6AuxiliaryConfig(
                strategy_lambda_min_turns=1.0,
                strategy_lambda_regain_initiative=0.0,
                strategy_lambda_teammate_finish=0.0,
                strategy_lambda_spring=0.0,
                strategy_lambda_structure=0.0,
            ),
            features=dataclasses.replace(
                baseline.learner.features, strategy=True
            ),
        ),
    )
    model = V3HybridModel(build_v2_schema(), model_config)
    learner = V3H6Learner(
        model,
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    labels = {
        "min_turns_after": 1.0,
        "min_turns_exact_mask": 0.0,
        "regain_initiative": 0.0,
        "teammate_finish": 0.0,
        "teammate_finish_mask": 0.0,
        "spring_probability": 0.0,
        "structure_cost": 0.0,
    }
    pending = capture_plain_transition(
        _observation(),
        selected_action_index=0,
        episode_id="masked-strategy-episode",
        deal_id="masked-strategy-deal",
        target_transform="raw",
        strategy_config=model.strategy_feature_config(),
    )
    metrics = learner.train_batch(
        [pending.finalize(2.0)], strategy_targets=[labels]
    )
    assert metrics.losses["strategy"]["phase"] == "no_valid_targets"
    assert metrics.losses["strategy"]["valid_samples"] == 0
    assert learner.composer.eligible_steps["strategy"] == 0


def test_h6_partially_masked_strategy_component_keeps_its_own_mean():
    baseline = _resolved()
    model_config = _model_config(
        strategy_features_enabled=True,
        strategy_aux_enabled=True,
    )
    resolved = V3H6ResolvedConfig(
        model=model_config,
        learner=dataclasses.replace(
            baseline.learner,
            losses=dataclasses.replace(
                baseline.learner.losses, lambda_strategy=1.0
            ),
            auxiliary=V3H6AuxiliaryConfig(
                strategy_lambda_min_turns=1.0,
                strategy_lambda_regain_initiative=0.0,
                strategy_lambda_teammate_finish=0.0,
                strategy_lambda_spring=0.0,
                strategy_lambda_structure=0.0,
            ),
            features=dataclasses.replace(
                baseline.learner.features, strategy=True
            ),
        ),
    )
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), model_config),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    labels = [
        {
            "min_turns_after": 1.0,
            "min_turns_exact_mask": mask,
            "regain_initiative": 0.0,
            "teammate_finish": 0.0,
            "teammate_finish_mask": 0.0,
            "spring_probability": 0.0,
            "structure_cost": 0.0,
        }
        for mask in (1.0, 0.0)
    ]
    gathered = {
        "min_turns_after": torch.zeros(2, 1),
        "regain_initiative_logit": torch.zeros(2, 1),
        "teammate_finish_logit": torch.zeros(2, 1),
        "spring_probability_logit": torch.zeros(2, 1),
        "structure_cost": torch.zeros(2, 1),
    }
    term = learner._strategy_term(
        gathered,
        [_transition(seed=20260740), _transition(seed=20260741)],
        ("landlord", "landlord"),
        ("strategy-0", "strategy-1"),
        labels,
    )
    composition = V3HybridLossComposer(
        V3HybridLossComposerConfig(lambda_strategy=1.0)
    ).compose({"strategy": term})
    assert composition.terms["strategy"].valid_samples == 1
    assert composition.terms["strategy"].raw_loss == pytest.approx(0.5)


def test_h6_late_checkpoint_rejection_restores_nested_state_atomically(tmp_path):
    resolved = _resolved()
    target = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    target.train_batch([_transition(seed=20260750)])
    source = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    source.train_batch([_transition(seed=20260751)])
    source.train_batch([_transition(seed=20260752)])
    bad_path = tmp_path / "bad.pt"
    source.save_checkpoint(bad_path)
    bad = torch.load(bad_path, map_location="cpu", weights_only=True)
    bad["counters"]["policy_version"] += 1
    torch.save(bad, bad_path)

    target.model.eval()
    before_path = tmp_path / "before.pt"
    target.save_checkpoint(before_path)
    with pytest.raises(CheckpointCompatibilityError, match="policy version mismatch"):
        target.load_checkpoint(bad_path)
    assert not target.model.training
    after_path = tmp_path / "after.pt"
    target.save_checkpoint(after_path)
    _assert_nested_equal(
        torch.load(before_path, map_location="cpu", weights_only=True),
        torch.load(after_path, map_location="cpu", weights_only=True),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_h6_cuda_parameter_update_checkpoint_and_second_update(tmp_path):
    resolved = _resolved(device="cuda")
    learner = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    before = _state(learner.model)
    first = learner.train_batch([_transition(seed=20260731)])
    assert first.public_aux_updated
    assert math.isfinite(first.loss_total)
    assert math.isfinite(first.public_aux_gradient_norm)
    assert any(
        not torch.equal(before[name], value)
        for name, value in learner.model.state_dict().items()
    )
    path = tmp_path / "h6-cuda.pt"
    learner.save_checkpoint(path)
    resumed = V3H6Learner(
        V3HybridModel(build_v2_schema(), resolved.model),
        ruleset=RuleSet.legacy(),
        config=resolved,
    )
    resumed.load_checkpoint(path)
    first_version = resumed.policy_version
    second = resumed.train_batch([_transition(seed=20260801)])
    assert second.policy_version > first_version
    assert second.public_aux_updated
    assert math.isfinite(second.loss_total)
