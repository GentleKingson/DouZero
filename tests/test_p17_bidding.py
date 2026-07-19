"""P17 standard full-game training and public learned-bidding closure."""

from __future__ import annotations

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from douzero.checkpoint import (
    CheckpointCompatibilityError,
    load_v2_checkpoint,
    load_v2_position_weights,
    save_v2_checkpoint,
    save_v2_position_weights,
)
from douzero.coach import CoachLabelStore, OpeningSampler, TRUE_RANDOM
from douzero.coach.records import CANONICAL_DECK, OpeningRecord
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.league import (
    LeagueManifest,
    PolicyEntry,
    PolicyLoaderContract,
    PolicyPool,
    PolicyPoolConfig,
    PopulationEpisodeRunner,
)
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.models_v2.batch import (
    BatchedBiddingInput,
    bidding_observations_to_model_input,
)
from douzero.models_v2.output import BatchedBiddingOutput, BiddingModelOutput
from douzero.observation.bidding import get_bidding_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    BiddingMinibatch,
    BiddingPolicyConfig,
    BiddingTransition,
    LossConfig,
    TrainerConfig,
    V2Trainer,
    bidding_loss,
    select_bidding_action,
)


def _tiny_config() -> ModelV2Config:
    return ModelV2Config(
        hidden_size=16,
        history_encoder="lstm",
        history_layers=1,
        history_heads=1,
        bidding_enabled=True,
        bidding_hidden_size=12,
    )


def _batch_bidding_outputs(
    outputs: list[BiddingModelOutput],
) -> BatchedBiddingOutput:
    uncertainty = (
        None
        if outputs[0].uncertainty is None
        else torch.stack([output.uncertainty for output in outputs])
    )
    return BatchedBiddingOutput(
        bid_logits=torch.stack([output.bid_logits for output in outputs]),
        bid_action_mask=torch.stack(
            [output.bid_action_mask for output in outputs]
        ),
        landlord_win_logit=torch.stack(
            [output.landlord_win_logit for output in outputs]
        ),
        expected_landlord_score=torch.stack(
            [output.expected_landlord_score for output in outputs]
        ),
        uncertainty=uncertainty,
    )


def _opening(order: tuple[str, str, str]) -> OpeningRecord:
    ruleset = RuleSet.standard()
    return OpeningRecord(
        deck=CANONICAL_DECK,
        bidding_order=order,
        ruleset=ruleset.to_dict(),
        landlord_candidate=order[0],
        public_features={"ruleset_hash": ruleset.stable_hash()},
    )


def _standard_policy_pool(
    model: ModelV2,
    ruleset: RuleSet,
    *,
    seed: int = 31,
    learner_seats_per_game: int = 1,
) -> tuple[PolicyEntry, PolicyPool]:
    contract = PolicyLoaderContract.for_v2_runtime(
        model.schema,
        model.config,
        checkpoint_kind="training_checkpoint",
    )
    current = PolicyEntry(
        policy_id="current-bid-policy",
        checkpoint_paths_by_role={},
        model_version="v2",
        ruleset_hash=ruleset.stable_hash(),
        feature_schema_hash=model.schema.stable_hash(),
        model_config_hash=model.config.stable_hash(),
        model_config_identity_version=model.config.IDENTITY_VERSION,
        checkpoint_kind="training_checkpoint",
        objective="adp",
        created_step=3,
        tags=("current",),
    )
    pool = PolicyPool(
        LeagueManifest((current,), current.policy_id),
        current,
        runtime_loader=contract,
        runtime_ruleset_hash=ruleset.stable_hash(),
        config=PolicyPoolConfig(
            mode="population",
            seed=seed,
            learner_seats_per_game=learner_seats_per_game,
            include_random_agent=True,
        ),
    )
    return current, pool


def _raw_bidding_obs(
    *,
    order: tuple[str, str, str] = ("0", "1", "2"),
    history: tuple[tuple[str, int], ...] = (),
) -> dict:
    ruleset = RuleSet.standard()
    highest = max((bid for _, bid in history), default=0)
    return {
        "phase": "bidding",
        "position": order[len(history)],
        "my_handcards": list(CANONICAL_DECK[:17]),
        "current_highest_bid": highest,
        "bidding_history": list(history),
        "bidding_order": list(order),
        "first_bidder": order[0],
        "legal_bids": [bid for bid in ruleset.bid_values if bid == 0 or bid > highest],
    }


def _labelled_bidding_transition(
    obs,
    action: int,
    *,
    source_policy: str = "learned",
    policy_credit_valid: bool = True,
) -> BiddingTransition:
    transition = BiddingTransition(
        obs=obs,
        bid_action=action,
        policy_version="test-policy",
        source_policy=source_policy,
        policy_credit_valid=policy_credit_valid,
    )
    other_seats = [seat for seat in ("0", "1", "2") if seat != obs.current_seat]
    transition.assign_actor_role({
        obs.current_seat: "landlord",
        other_seats[0]: "landlord_down",
        other_seats[1]: "landlord_up",
    })
    transition.label_from_terminal({
        "team_targets": {
            "landlord": {"target_win": 1.0, "target_score": 2.0},
            "landlord_down": {"target_win": 0.0, "target_score": -1.0},
            "landlord_up": {"target_win": 0.0, "target_score": -1.0},
        }
    })
    return transition


def _bidding_feature_field(obs, name: str) -> np.ndarray:
    offset = 0
    for field in obs.schema.fields:
        if field.name == name:
            return obs.features[offset : offset + field.width]
        offset += field.width
    raise AssertionError(f"missing bidding feature field {name!r}")


@pytest.mark.parametrize(
    "order",
    (("0", "1", "2"), ("1", "2", "0"), ("2", "0", "1")),
)
def test_public_bidding_observation_handles_every_first_bidder(order):
    ruleset = RuleSet.standard()
    env = Env("adp", ruleset=ruleset)
    env.reset(opening=_opening(order))
    seen = []
    for bid in (0, 1, 2):
        obs = get_bidding_obs_v2(env.bidding_obs, ruleset=ruleset)
        seen.append(obs.current_seat)
        assert obs.first_bidder == order[0]
        assert obs.current_seat not in {
            "landlord", "landlord_up", "landlord_down"
        }
        _raw, _reward, done, _info = env.step(None, bid_value=bid)
        assert not done
    assert tuple(seen) == order
    assert env.bidding_obs is None


@pytest.mark.parametrize(
    "order",
    (("0", "1", "2"), ("1", "2", "0"), ("2", "0", "1")),
)
def test_bidding_features_preserve_absolute_neutral_seat_identity(order):
    ruleset = RuleSet.standard()
    first = get_bidding_obs_v2(_raw_bidding_obs(order=order), ruleset=ruleset)
    assert np.array_equal(
        _bidding_feature_field(first, "current_seat"),
        np.eye(3, dtype=np.float32)[int(order[0])],
    )
    assert np.array_equal(
        _bidding_feature_field(first, "first_bidder"),
        np.eye(3, dtype=np.float32)[int(order[0])],
    )

    second = get_bidding_obs_v2(
        _raw_bidding_obs(order=order, history=((order[0], 0),)),
        ruleset=ruleset,
    )
    assert np.array_equal(
        _bidding_feature_field(second, "current_seat"),
        np.eye(3, dtype=np.float32)[int(order[1])],
    )
    history = _bidding_feature_field(second, "bidding_history").reshape(3, -1)
    assert np.array_equal(
        history[0, :3], np.eye(3, dtype=np.float32)[int(order[0])]
    )


def test_first_bidder_rotations_produce_distinct_model_features():
    ruleset = RuleSet.standard()
    observations = [
        get_bidding_obs_v2(_raw_bidding_obs(order=order), ruleset=ruleset)
        for order in (("0", "1", "2"), ("1", "2", "0"), ("2", "0", "1"))
    ]
    assert all(
        not np.array_equal(left.features, right.features)
        for index, left in enumerate(observations)
        for right in observations[index + 1 :]
    )


def test_bidding_encoder_is_allowlisted_hidden_allocation_invariant_and_strict():
    ruleset = RuleSet.standard()
    raw = _raw_bidding_obs(history=(("0", 1),))
    first = copy.deepcopy(raw)
    first.update({"other_hands": {"1": [30]}, "bottom_cards": [3, 4, 5]})
    second = copy.deepcopy(raw)
    second.update({"other_hands": {"1": [3]}, "bottom_cards": [17, 20, 30]})
    obs_a = get_bidding_obs_v2(first, ruleset=ruleset, public_style=np.zeros(8))
    obs_b = get_bidding_obs_v2(second, ruleset=ruleset, public_style=np.zeros(8))
    assert np.array_equal(obs_a.features, obs_b.features)
    assert not obs_a.is_privileged
    assert obs_a.features.flags.writeable is False

    bad = copy.deepcopy(raw)
    bad["my_handcards"] = bad["my_handcards"][:-1]
    with pytest.raises(ValueError, match="exactly 17"):
        get_bidding_obs_v2(bad, ruleset=ruleset)
    bad = copy.deepcopy(raw)
    bad["current_highest_bid"] = 3
    with pytest.raises(ValueError, match="does not match"):
        get_bidding_obs_v2(bad, ruleset=ruleset)
    with pytest.raises(ValueError, match="finite"):
        get_bidding_obs_v2(raw, ruleset=ruleset, public_style=[float("nan")])


def test_bidding_head_is_separate_from_card_action_encoder_and_masks_illegal():
    ruleset = RuleSet.standard()
    obs = get_bidding_obs_v2(
        _raw_bidding_obs(history=(("0", 1),)), ruleset=ruleset
    )
    model = ModelV2(build_v2_schema(), _tiny_config())

    def card_path_must_not_run(*_args, **_kwargs):
        raise AssertionError("card ActionEncoder was called by forward_bidding")

    model.action_encoder.forward = card_path_must_not_run
    output = model.forward_bidding(obs)
    assert output.bid_logits.shape == (4,)
    assert output.landlord_win_logit.ndim == 0
    assert output.expected_landlord_score.ndim == 0
    assert output.masked_bid_logits()[1].item() == float("-inf")
    assert output.argmax_bid() in obs.legal_bids


def test_batched_bidding_output_matches_scalar_wrapper_for_mixed_masks():
    ruleset = RuleSet.standard()
    observations = [
        get_bidding_obs_v2(_raw_bidding_obs(history=history), ruleset=ruleset)
        for history in (
            (),
            (("0", 1),),
            (("0", 1), ("1", 2)),
        )
    ]
    model = ModelV2(build_v2_schema(), _tiny_config()).eval()
    state_keys = tuple(model.state_dict())
    batched = model.forward_bidding_batched(
        bidding_observations_to_model_input(observations)
    )
    scalars = [model.forward_bidding(obs) for obs in observations]
    assert batched.bid_logits.shape == (3, 4)
    assert torch.equal(
        batched.bid_action_mask,
        torch.stack([output.bid_action_mask for output in scalars]),
    )
    for index, scalar in enumerate(scalars):
        assert torch.allclose(batched.bid_logits[index], scalar.bid_logits)
        assert torch.allclose(
            batched.landlord_win_logit[index], scalar.landlord_win_logit
        )
        assert torch.allclose(
            batched.expected_landlord_score[index],
            scalar.expected_landlord_score,
        )
    assert tuple(model.state_dict()) == state_keys


def test_batched_bidding_loss_and_gradients_match_b1_compatibility_path():
    ruleset = RuleSet.standard()
    observations = [
        get_bidding_obs_v2(_raw_bidding_obs(history=history), ruleset=ruleset)
        for history in ((), (("0", 1),), (("0", 1), ("1", 2)))
    ]
    transitions = [
        _labelled_bidding_transition(
            observations[0], 1, source_policy="rule"
        ),
        _labelled_bidding_transition(
            observations[1], 2, source_policy="learned"
        ),
        _labelled_bidding_transition(
            observations[2], 3, source_policy="epsilon_random"
        ),
    ]
    minibatch = BiddingMinibatch(transitions)
    batched_model = ModelV2(build_v2_schema(), _tiny_config())
    scalar_model = copy.deepcopy(batched_model)

    batched_output = batched_model.forward_bidding_batched(
        bidding_observations_to_model_input(observations)
    )
    batched_components = bidding_loss(
        batched_output,
        minibatch.to_targets(batched_output.bid_logits.device),
        lambda_policy=1.0,
        lambda_landlord_win=0.5,
        lambda_landlord_score=0.25,
    )
    batched_components.total.backward()

    scalar_outputs = [
        scalar_model.forward_bidding(observation)
        for observation in observations
    ]
    scalar_output = _batch_bidding_outputs(scalar_outputs)
    scalar_components = bidding_loss(
        scalar_output,
        minibatch.to_targets(scalar_output.bid_logits.device),
        lambda_policy=1.0,
        lambda_landlord_win=0.5,
        lambda_landlord_score=0.25,
    )
    scalar_components.total.backward()

    assert torch.allclose(
        batched_components.total, scalar_components.total, atol=1e-6, rtol=1e-6
    )
    for batched_parameter, scalar_parameter in zip(
        batched_model.bidding_heads.parameters(),
        scalar_model.bidding_heads.parameters(),
    ):
        assert batched_parameter.grad is not None
        assert scalar_parameter.grad is not None
        assert torch.allclose(
            batched_parameter.grad,
            scalar_parameter.grad,
            atol=2e-6,
            rtol=2e-5,
        )


def test_batched_bidding_contract_rejects_empty_and_illegal_rows():
    schema_hash = build_v2_schema().stable_hash()
    with pytest.raises(ValueError, match="legal action"):
        BatchedBiddingInput(
            features=torch.zeros(2, 8),
            legal_mask=torch.tensor(
                [[True, False, False, False], [False, False, False, False]]
            ),
            feature_schema_hash=schema_hash,
        )

    obs = get_bidding_obs_v2(
        _raw_bidding_obs(history=(("0", 1),)), ruleset=RuleSet.standard()
    )
    transition = _labelled_bidding_transition(obs, 3)
    minibatch = BiddingMinibatch([transition])
    output = BatchedBiddingOutput(
        bid_logits=torch.zeros(1, 4),
        bid_action_mask=torch.tensor([[True, False, True, False]]),
        landlord_win_logit=torch.zeros(1),
        expected_landlord_score=torch.zeros(1),
    )
    with pytest.raises(ValueError, match="illegal action"):
        bidding_loss(
            output,
            minibatch.to_targets(output.bid_logits.device),
            lambda_policy=1.0,
            lambda_landlord_win=0.0,
            lambda_landlord_score=0.0,
        )


def test_bidding_batch_controls_inherit_and_validate_independently():
    inherited = TrainerConfig(batch_size=7, optimizer_steps=0)
    assert inherited.bidding_batch_size == 7
    explicit = TrainerConfig(
        batch_size=7,
        bidding_batch_size=3,
        bidding_update_interval=2,
        optimizer_steps=0,
    )
    assert explicit.bidding_batch_size == 3
    assert explicit.bidding_update_interval == 2
    with pytest.raises(ValueError, match="bidding_batch_size"):
        TrainerConfig(bidding_batch_size=0)
    with pytest.raises(ValueError, match="bidding_update_interval"):
        TrainerConfig(bidding_update_interval=0)


def test_masked_bid_loss_uses_actor_return_and_landlord_value_gradients():
    ruleset = RuleSet.standard()
    obs = get_bidding_obs_v2(
        _raw_bidding_obs(history=(("0", 1),)), ruleset=ruleset
    )
    transition = BiddingTransition(obs, 3, "policy-7", "learned")
    transition.assign_actor_role({
        "0": "landlord",
        "1": "landlord_down",
        "2": "landlord_up",
    })
    transition.label_from_terminal(
        {
            "team_targets": {
                "landlord": {"target_win": 1.0, "target_score": 8.0},
                "landlord_down": {"target_win": 0.0, "target_score": -4.0},
                "landlord_up": {"target_win": 0.0, "target_score": -4.0},
            }
        }
    )
    logits = torch.tensor([0.0, 100.0, 1.0, 2.0], requires_grad=True)
    win = torch.tensor(0.2, requires_grad=True)
    score = torch.tensor(1.5, requires_grad=True)
    output = BiddingModelOutput(
        logits,
        torch.tensor([True, False, True, True]),
        win,
        score,
    )
    minibatch = BiddingMinibatch([transition])
    components = bidding_loss(
        _batch_bidding_outputs([output]),
        minibatch.to_targets(logits.device),
        lambda_policy=1.0,
        lambda_landlord_win=0.5,
        lambda_landlord_score=0.25,
    )
    assert torch.isfinite(components.total)
    components.total.backward()
    assert logits.grad is not None
    assert logits.grad[3].item() > 0.0
    assert torch.equal(logits.grad[:3], torch.zeros_like(logits.grad[:3]))
    assert win.grad is not None and score.grad is not None
    assert transition.target_landlord_win == 1.0
    assert transition.target_landlord_score == 8.0
    assert transition.actor_role == "landlord_down"
    assert transition.target_actor_win == 0.0


def test_all_disabled_bid_credit_is_finite_with_illegal_actions_and_zero_gradient():
    ruleset = RuleSet.standard()
    observations = [
        get_bidding_obs_v2(
            _raw_bidding_obs(history=(("0", 1),)), ruleset=ruleset
        ),
        get_bidding_obs_v2(
            _raw_bidding_obs(history=(("0", 1), ("1", 2))), ruleset=ruleset
        ),
    ]
    transitions = [
        _labelled_bidding_transition(
            obs,
            max(obs.legal_bids),
            policy_credit_valid=False,
        )
        for obs in observations
    ]
    logits = [
        torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True),
        torch.tensor([3.0, 2.0, 1.0, 0.0], requires_grad=True),
    ]
    outputs = [
        BiddingModelOutput(
            row,
            torch.as_tensor(obs.bid_action_mask.copy()),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        for row, obs in zip(logits, observations)
    ]
    minibatch = BiddingMinibatch(transitions)
    components = bidding_loss(
        _batch_bidding_outputs(outputs),
        minibatch.to_targets(logits[0].device),
        lambda_policy=1.0,
        lambda_landlord_win=0.0,
        lambda_landlord_score=0.0,
    )
    assert components.policy == 0.0
    assert torch.isfinite(components.total)
    components.total.backward()
    assert all(
        row.grad is not None and torch.equal(row.grad, torch.zeros_like(row))
        for row in logits
    )


def test_bid_regret_fails_closed_without_a_separate_action_value_head():
    ruleset = RuleSet.standard()
    obs = get_bidding_obs_v2(_raw_bidding_obs(), ruleset=ruleset)
    transition = _labelled_bidding_transition(obs, max(obs.legal_bids))
    output = BiddingModelOutput(
        torch.zeros(4),
        torch.as_tensor(obs.bid_action_mask.copy()),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    with pytest.raises(ValueError, match="lambda_bid_regret is unsupported"):
        minibatch = BiddingMinibatch([transition])
        bidding_loss(
            _batch_bidding_outputs([output]),
            minibatch.to_targets(output.bid_logits.device),
            lambda_policy=1.0,
            lambda_landlord_win=0.0,
            lambda_landlord_score=0.0,
            lambda_regret=0.1,
        )

    with pytest.raises(ValueError, match="lambda_bid_regret is unsupported"):
        V2Trainer(
            ModelV2(build_v2_schema(), _tiny_config()),
            ruleset=ruleset,
            loss_config=LossConfig(lambda_bid_regret=0.1),
            bidding_policy_config=BiddingPolicyConfig(policy="max"),
            config=TrainerConfig(max_episodes=0, optimizer_steps=0),
        )


def test_warm_start_bid_is_not_mislabelled_as_learned():
    obs = get_bidding_obs_v2(_raw_bidding_obs(), ruleset=RuleSet.standard())
    bid, source = select_bidding_action(
        obs,
        BiddingPolicyConfig(
            policy="learned", warm_start_policy="rule", learned_probability=0.0
        ),
        random.Random(4),
        lambda _obs: 3,
    )
    assert bid in obs.legal_bids and source == "rule"
    bid, source = select_bidding_action(
        obs,
        BiddingPolicyConfig(
            policy="learned", warm_start_policy="rule", learned_probability=1.0
        ),
        random.Random(4),
        lambda _obs: 3,
    )
    assert bid == 3 and source == "learned"


@pytest.mark.parametrize("field", ("policy", "warm_start_policy"))
def test_epsilon_random_is_provenance_only_not_a_configurable_policy(field):
    with pytest.raises(
        ValueError, match="unsupported bidding policy|warm_start_policy"
    ):
        BiddingPolicyConfig(**{field: "epsilon_random"})


def test_all_pass_redeal_discards_abandoned_bid_transitions():
    ruleset = replace(RuleSet.standard(), max_redeals=1)
    trainer = V2Trainer(
        ModelV2(build_v2_schema(), _tiny_config()),
        ruleset=ruleset,
        loss_config=LossConfig(lambda_bid_policy=1.0),
        bidding_policy_config=BiddingPolicyConfig(policy="pass"),
        config=TrainerConfig(
            max_episodes=0,
            optimizer_steps=0,
            batch_size=1,
            exp_epsilon=1.0,
            max_steps_per_episode=500,
            rng_seed=9,
        ),
    )
    episode = trainer._run_one_episode()
    assert episode.redeal_count == 1
    assert episode.max_redeals_exceeded is True
    assert episode.abandoned_bidding_transitions == 6
    assert episode.bidding_transitions == []

    trainer._run_one_episode = lambda: episode
    trainer.collect_episodes(1)
    assert trainer.stats.redeals == 1
    assert trainer.stats.max_redeals_exceeded == 1
    assert trainer.stats.bidding_transitions_collected == 0
    assert len(trainer.bidding_buffer) == 0


def test_standard_population_runner_records_only_learner_policy_decisions():
    ruleset = RuleSet.standard()
    model = ModelV2(build_v2_schema(), _tiny_config())
    current, pool = _standard_policy_pool(model, ruleset)
    with pytest.raises(ValueError, match="policy_version must match"):
        V2Trainer(
            model,
            ruleset=ruleset,
            loss_config=LossConfig(lambda_bid_policy=1.0),
            bidding_policy_config=BiddingPolicyConfig(policy="max"),
            config=TrainerConfig(max_episodes=0),
            policy_pool=pool,
            policy_version="different-snapshot",
        )
    # Canonical bundle slot "landlord" maps to neutral seat 0 before roles
    # exist. Pick a deterministic game where the learner owns that first bid.
    game_index = next(
        index
        for index in range(100)
        if pool.sample_bundle(index).learner_controlled_seats == ("landlord",)
    )
    runner = PopulationEpisodeRunner(
        pool,
        lambda _obs: 0,
        current_bidding_selector=lambda obs: (max(obs.legal_bids), "learned"),
        ruleset=ruleset,
    )
    episode, record = runner.run(
        game_index,
        policy_version_at_start=current.policy_id,
        policy_step_at_start=3,
    )
    assert episode.bidding_transitions
    assert all(
        transition.policy_version == current.policy_id
        for transition in episode.bidding_transitions
    )
    assert all(
        transition.actor_role == "landlord"
        for transition in episode.bidding_transitions
    )
    assert episode.transitions
    assert {
        transition.position for transition in episode.transitions
    }.issubset(set(record.learner_controlled_seats))
    assert all(
        transition.policy_id == current.policy_id
        for transition in episode.transitions
    )
    assert record.bid_value in (1, 2, 3)
    assert record.redeal_count == episode.redeal_count
    assert record.max_redeals_exceeded is False
    assert record.bidding_transitions == len(episode.bidding_transitions)


def test_population_runner_rejects_bidding_selector_without_source_provenance():
    ruleset = RuleSet.standard()
    model = ModelV2(build_v2_schema(), _tiny_config())
    _current, pool = _standard_policy_pool(model, ruleset)
    game_index = next(
        index
        for index in range(100)
        if pool.sample_bundle(index).learner_controlled_seats == ("landlord",)
    )
    runner = PopulationEpisodeRunner(
        pool,
        lambda _obs: 0,
        current_bidding_selector=lambda obs: max(obs.legal_bids),
        ruleset=ruleset,
    )

    with pytest.raises(TypeError, match="must return.*source_policy"):
        runner.run(game_index)


def test_population_runner_audits_cap_and_discards_all_pass_bids():
    ruleset = replace(RuleSet.standard(), max_redeals=1)
    model = ModelV2(build_v2_schema(), _tiny_config())
    current, pool = _standard_policy_pool(
        model, ruleset, seed=37, learner_seats_per_game=3
    )
    runner = PopulationEpisodeRunner(
        pool,
        lambda _obs: 0,
        current_bidding_selector=lambda _obs: (0, "learned"),
        ruleset=ruleset,
    )

    episode, record = runner.run(
        0,
        policy_version_at_start=current.policy_id,
        policy_step_at_start=3,
    )
    assert episode.redeal_count == 1
    assert episode.max_redeals_exceeded is True
    assert episode.abandoned_bidding_transitions == 6
    assert episode.bidding_transitions == []
    assert record.max_redeals_exceeded is True
    assert record.redeal_count == 1
    assert record.bidding_transitions == 0
    assert record.abandoned_bidding_transitions == 6


def test_standard_coach_runner_labels_kept_opening_but_not_redeal(tmp_path):
    policy_version = "standard-coach-policy"
    ruleset = RuleSet.standard()
    store = CoachLabelStore(str(tmp_path / "kept-opening.jsonl"))
    sampler = OpeningSampler(
        ruleset=ruleset,
        policy_version=policy_version,
        mode=TRUE_RANDOM,
        seed=41,
    )
    trainer = V2Trainer(
        ModelV2(build_v2_schema(), _tiny_config()),
        ruleset=ruleset,
        loss_config=LossConfig(lambda_bid_policy=1.0),
        bidding_policy_config=BiddingPolicyConfig(policy="max"),
        config=TrainerConfig(
            max_episodes=1,
            optimizer_steps=0,
            batch_size=1,
            exp_epsilon=1.0,
            rng_seed=41,
        ),
        opening_sampler=sampler,
        coach_label_store=store,
        policy_version=policy_version,
        policy_step=7,
    )
    trainer.collect_episodes(1)
    labels = store.load_fresh(
        policy_version=policy_version,
        current_policy_step=7,
        max_age_steps=0,
    )
    assert len(labels) == 1
    assert labels[0].opening.ruleset_obj.ruleset_id == "standard"
    assert labels[0].policy_step == 7

    redeal_ruleset = replace(ruleset, max_redeals=1)
    redeal_store = CoachLabelStore(str(tmp_path / "redealt-opening.jsonl"))
    redeal_sampler = OpeningSampler(
        ruleset=redeal_ruleset,
        policy_version=policy_version,
        mode=TRUE_RANDOM,
        seed=43,
    )
    redeal_trainer = V2Trainer(
        ModelV2(build_v2_schema(), _tiny_config()),
        ruleset=redeal_ruleset,
        loss_config=LossConfig(lambda_bid_policy=1.0),
        bidding_policy_config=BiddingPolicyConfig(policy="pass"),
        config=TrainerConfig(
            max_episodes=1,
            optimizer_steps=0,
            batch_size=1,
            exp_epsilon=1.0,
            rng_seed=43,
        ),
        opening_sampler=redeal_sampler,
        coach_label_store=redeal_store,
        policy_version=policy_version,
        policy_step=8,
    )
    redeal_trainer.collect_episodes(1)
    assert redeal_trainer.stats.redeals == 1
    assert redeal_store.load_fresh(
        policy_version=policy_version,
        current_policy_step=8,
        max_age_steps=0,
    ) == []


def test_standard_training_step_and_strict_resume(tmp_path):
    torch.manual_seed(12)
    cfg = _tiny_config()
    loss = LossConfig(
        lambda_win=1.0,
        lambda_score=0.5,
        lambda_bid_policy=1.0,
        lambda_bid_win=0.5,
        lambda_bid_score=0.25,
    )
    trainer_cfg = TrainerConfig(
        max_episodes=2,
        optimizer_steps=1,
        batch_size=1,
        bidding_batch_size=2,
        exp_epsilon=1.0,
        max_steps_per_episode=500,
        rng_seed=17,
    )
    trainer = V2Trainer(
        ModelV2(build_v2_schema(), cfg),
        ruleset=RuleSet.standard(),
        loss_config=loss,
        bidding_policy_config=BiddingPolicyConfig(policy="max"),
        config=trainer_cfg,
    )
    before = trainer.model.bidding_heads.policy.weight.detach().clone()
    stats = trainer.train()
    assert stats.optimizer_steps == 1
    assert stats.learner_cardplay_samples == 1
    assert stats.learner_bidding_samples == 2
    assert stats.bidding_transitions_collected >= 1
    assert {
        transition.obs.current_seat
        for transition in trainer.bidding_buffer._transitions
    } == {"0", "1"}
    assert all(
        transition.actor_role == "landlord"
        for transition in trainer.bidding_buffer._transitions
    )
    assert all(
        transition.policy_id == "current"
        for transition in trainer.buffer._episodes[-1].transitions
    )
    assert not torch.equal(before, trainer.model.bidding_heads.policy.weight)
    assert "loss_bid_policy" in stats.last_loss

    checkpoint = tmp_path / "standard-resume.pt"
    identity = trainer.save_training_checkpoint(str(checkpoint))
    assert identity["checkpoint_version"] == 3
    assert identity["training_topology"] == "single_process"
    assert identity["training_world_size"] == 1
    assert identity["bidding_head_version"]
    assert len(identity["source_git_sha"]) in (40, 64)
    restored = V2Trainer(
        ModelV2(build_v2_schema(), cfg),
        ruleset=RuleSet.standard(),
        loss_config=loss,
        bidding_policy_config=BiddingPolicyConfig(policy="max"),
        config=trainer_cfg,
    )
    restored.load_training_checkpoint(str(checkpoint))
    assert restored.stats.optimizer_steps == 1
    assert all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in trainer.model.state_dict().items()
    )
    restored.collect_episodes(2)
    assert restored.step() is not None
    assert restored.stats.optimizer_steps == 2

    tampered_bundle = torch.load(checkpoint, weights_only=True)
    replacement = "f" if identity["source_git_sha"][0] != "f" else "e"
    tampered_bundle["source_git_sha"] = replacement * len(
        identity["source_git_sha"]
    )
    tampered_checkpoint = tmp_path / "wrong-source-sha.pt"
    torch.save(tampered_bundle, tampered_checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="source_git_sha mismatch"):
        restored.load_training_checkpoint(str(tampered_checkpoint))

    topology_tamper = torch.load(checkpoint, weights_only=True)
    topology_tamper["training_world_size"] = 2
    topology_checkpoint = tmp_path / "wrong-training-topology.pt"
    torch.save(topology_tamper, topology_checkpoint)
    with pytest.raises(
        CheckpointCompatibilityError, match="training_world_size mismatch"
    ):
        restored.load_training_checkpoint(str(topology_checkpoint))


def test_v2_checkpoint_carries_and_validates_explicit_bidding_identity(tmp_path):
    schema = build_v2_schema()
    cfg = _tiny_config()
    model = ModelV2(schema, cfg)
    path = tmp_path / "bidding-model.tar"
    save_v2_checkpoint(str(path), model, ruleset=RuleSet.standard())
    bundle = torch.load(path, weights_only=True)
    assert bundle["bidding_head_version"] == "bid-policy-value-v2"
    assert bundle["bidding_action_schema"] == "score-0-1-2-3-v1"
    assert bundle["bidding_feature_schema_hash"] == model.bidding_schema.stable_hash()
    state, _manifest = load_v2_checkpoint(
        str(path),
        expected_schema_hash=schema.stable_hash(),
        expected_model_config_hash=cfg.stable_hash(),
        expected_ruleset=RuleSet.standard(),
        runtime_model_config=cfg,
    )
    assert set(state) == set(model.state_dict())

    old_semantics = copy.deepcopy(bundle)
    old_semantics["bidding_head_version"] = "bid-policy-value-v1"
    old_path = tmp_path / "old-bidding-semantics.tar"
    torch.save(old_semantics, old_path)
    with pytest.raises(CheckpointCompatibilityError, match="bidding_head_version"):
        load_v2_checkpoint(
            str(old_path),
            expected_schema_hash=schema.stable_hash(),
            expected_model_config_hash=cfg.stable_hash(),
            expected_ruleset=RuleSet.standard(),
            runtime_model_config=cfg,
        )

    bundle["bidding_action_schema"] = "wrong-action-contract"
    tampered = tmp_path / "tampered.tar"
    torch.save(bundle, tampered)
    with pytest.raises(CheckpointCompatibilityError, match="bidding_action_schema"):
        load_v2_checkpoint(
            str(tampered),
            expected_schema_hash=schema.stable_hash(),
            expected_model_config_hash=cfg.stable_hash(),
            expected_ruleset=RuleSet.standard(),
            runtime_model_config=cfg,
        )


def test_pre_p17_bidding_disabled_default_config_hash_is_pinned():
    cfg = ModelV2Config()
    assert cfg.bidding_enabled is False
    assert cfg.IDENTITY_VERSION == 3
    assert (
        cfg.stable_hash()
        == "c4577d155385c79361280e6529ca42b5d5991095ec3dfe526f7ef5f5365962bb"
    )


def test_pre_p17_bidding_disabled_identity_v3_sidecar_strict_loads(tmp_path):
    schema = build_v2_schema()
    cfg = ModelV2Config()
    model = ModelV2(schema, cfg)
    current_path = tmp_path / "current-disabled-sidecar.ckpt"
    save_v2_position_weights(
        str(current_path), model, ruleset=RuleSet.legacy()
    )
    bundle = torch.load(current_path, weights_only=True)
    assert bundle["model_config_identity_version"] == 3

    # P16 identity-v3 sidecars predate the explicit bidding identity fields.
    for key in (
        "bidding_head_version",
        "bidding_action_schema",
        "bidding_feature_schema_hash",
    ):
        assert bundle.pop(key) == ""
    historical_path = tmp_path / "pre-p17-disabled-sidecar.ckpt"
    torch.save(bundle, historical_path)

    state_dict, manifest = load_v2_position_weights(
        str(historical_path),
        expected_schema_hash=schema.stable_hash(),
        expected_model_config_hash=cfg.stable_hash(),
        expected_ruleset=RuleSet.legacy(),
        runtime_model_config=cfg,
        training_device="cpu",
    )
    restored = ModelV2(schema, cfg)
    restored.load_state_dict(state_dict, strict=True)
    assert manifest.checkpoint_kind == "public_policy"
    assert set(state_dict) == set(model.state_dict())
