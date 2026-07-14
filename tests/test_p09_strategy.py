"""P09 structural/cooperation features, auxiliary heads, and gated prior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from douzero.models_v2.batch import observation_to_model_inputs
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.models_v2.output import ModelOutput
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.strategy import (
    STRATEGY_FEATURE_LAYOUT_HASH,
    STRATEGY_FEATURE_NAMES,
    STRATEGY_FEATURE_VERSION,
    StrategyFeatureConfig,
    action_structure_cost,
    build_strategy_feature_matrix,
    hand_decomposition,
    strategy_feature_layout_hash,
)
from douzero.strategy.auxiliary import strategy_auxiliary_loss
from douzero.training.decision_policy import DecisionConfig, select_action
from douzero.training.losses import LossConfig


def _public(
    *,
    role="landlord_down",
    teammate_left=1,
    landlord_left=10,
    landlord_played=(),
    landlord_action_count=0,
):
    hand = (3, 4, 5, 17, 20, 30)
    return SimpleNamespace(
        acting_role=role,
        my_handcards=hand,
        legal_actions=((3,), (17,), (20,), (30,), ()),
        num_cards_left={
            "landlord": landlord_left,
            "landlord_down": 6 if role != "landlord_down" else len(hand),
            "landlord_up": teammate_left,
        },
        played_cards={
            "landlord": tuple(landlord_played),
            "landlord_down": (),
            "landlord_up": (),
        },
        non_pass_action_counts={
            "landlord": landlord_action_count,
            "landlord_down": 0,
            "landlord_up": 0,
        },
        last_move=(4,),
        last_move_dict={"landlord": (), "landlord_down": (), "landlord_up": (4,)},
    )


def _model_output(*, p_win, score, prior):
    n = len(p_win)
    p = torch.tensor(p_win, dtype=torch.float32).reshape(n, 1)
    s = torch.tensor(score, dtype=torch.float32).reshape(n, 1)
    return ModelOutput(
        win_logit=torch.logit(p.clamp(1e-4, 1 - 1e-4)),
        score_if_win=s,
        score_if_loss=s,
        p_win=p,
        score_mean=s,
        action_mask=torch.ones(n, dtype=torch.bool),
        prior_logit=torch.tensor(prior, dtype=torch.float32).reshape(n, 1),
    )


class TestHandDecomposition:
    def test_known_exact_hands(self):
        assert hand_decomposition([3, 4, 5, 6, 7]).min_turns == 1
        assert hand_decomposition([3, 3, 4, 4, 5, 5]).min_turns == 1
        assert hand_decomposition([]).min_turns == 0

    def test_budget_fallback_is_bounded_and_repeatable(self):
        hand = [3, 3, 3, 4, 4, 5, 6, 7, 9, 10, 12, 14, 17, 20, 30]
        first = hand_decomposition(hand, node_budget=1)
        second = hand_decomposition(hand, node_budget=1)
        assert first == second
        assert first.fallback_used and not first.exact
        assert 1 <= first.min_turns <= len(hand)

    def test_rejects_more_than_twenty_cards(self):
        with pytest.raises(ValueError, match="at most 20"):
            hand_decomposition([3] * 21)

    def test_fake_clock_timeout_is_deterministic_and_uses_fixed_fallback(self):
        class TickClock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                self.value += 0.001
                return self.value

        hand = [3, 3, 3, 4, 4, 5, 6, 7, 9, 10, 12, 14, 17, 20, 30]
        first = hand_decomposition(
            hand, node_budget=10_000, time_budget_ms=3, clock=TickClock()
        )
        second = hand_decomposition(
            hand, node_budget=10_000, time_budget_ms=3, clock=TickClock()
        )
        assert first == second
        assert first.fallback_used and not first.exact
        # Fixed fallback: one legal rank group per turn, with the rocket as one.
        assert first.min_turns == len(set(hand)) - 1

    def test_budget_aware_move_generation_preserves_legal_moves(self):
        from douzero.env.move_generator import MovesGener

        hand = [3, 3, 3, 4, 4, 4, 5, 6, 7, 8, 9]
        plain = MovesGener(hand).gen_moves()
        checked = MovesGener(hand, budget_check=lambda: None).gen_moves()
        assert plain == checked

    def test_nonzero_time_budget_is_not_process_cached(self, monkeypatch):
        import douzero.strategy.features as strategy_features

        calls = []
        original = strategy_features.hand_decomposition

        def counted(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(strategy_features, "hand_decomposition", counted)
        config = StrategyFeatureConfig(node_budget=50, time_budget_ms=1)
        strategy_features._decomposition((3, 4, 5), config)
        strategy_features._decomposition((3, 4, 5), config)
        assert len(calls) == 2


class TestStructureCost:
    def test_breaking_bomb_costs_more_than_playing_unrelated_single(self):
        hand = [3, 3, 3, 3, 4, 5, 6]
        assert action_structure_cost(hand, [3]).bomb_break_cost == 1.0
        assert action_structure_cost(hand, [3]).total > action_structure_cost(hand, [4]).total

    def test_splitting_rocket_is_not_free(self):
        cost = action_structure_cost([3, 4, 20, 30], [20])
        assert cost.joker_pair_break == 1.0
        assert cost.total >= 2.0

    def test_four_with_two_is_legal_but_structurally_expensive(self):
        cost = action_structure_cost([3, 3, 3, 3, 4, 5], [3, 3, 3, 3, 4, 5])
        assert cost.bomb_break_cost == 1.0
        assert cost.total > 0.0


class TestFeatureMatrix:
    def test_fixed_layout_and_determinism(self):
        public = _public()
        cfg = StrategyFeatureConfig(node_budget=10)
        first = build_strategy_feature_matrix(public, cfg)
        second = build_strategy_feature_matrix(public, cfg)
        assert first.shape == (len(public.legal_actions), len(STRATEGY_FEATURE_NAMES))
        np.testing.assert_array_equal(first, second)
        assert not first.flags.writeable

    def test_teammate_one_card_small_single_and_landlord_block(self):
        features = build_strategy_feature_matrix(
            _public(teammate_left=1, landlord_left=1),
            StrategyFeatureConfig(node_budget=5),
        )
        index = {name: i for i, name in enumerate(STRATEGY_FEATURE_NAMES)}
        # A low single can be taken by a one-card teammate; a 2 blocks a
        # landlord who is also down to one card.
        assert features[0, index["feeds_teammate"]] == 1.0
        assert features[1, index["blocks_one_card"]] == 1.0

    def test_farmer_roles_are_explicitly_distinct(self):
        cfg = StrategyFeatureConfig(node_budget=5)
        up = build_strategy_feature_matrix(_public(role="landlord_up"), cfg)
        down = build_strategy_feature_matrix(_public(role="landlord_down"), cfg)
        index = {name: i for i, name in enumerate(STRATEGY_FEATURE_NAMES)}
        assert np.all(up[:, index["is_landlord_up"]] == 1.0)
        assert np.all(down[:, index["is_landlord_down"]] == 1.0)

    def test_group_ablation_zeroes_columns(self):
        cfg = StrategyFeatureConfig(
            hand_enabled=False,
            structure_enabled=False,
            control_enabled=False,
            cooperation_enabled=False,
            risk_enabled=False,
        )
        assert np.count_nonzero(build_strategy_feature_matrix(_public(), cfg)) == 0

    @pytest.mark.parametrize(
        "played_cards",
        [
            (3, 3),
            (3, 4, 5, 6, 7),
            (3, 3, 3, 4, 4, 4, 5, 6),
        ],
    )
    def test_anti_spring_risk_counts_actions_not_cards(self, played_cards):
        features = build_strategy_feature_matrix(
            _public(
                landlord_left=4,
                landlord_played=played_cards,
                landlord_action_count=1,
            ),
            StrategyFeatureConfig(hand_enabled=False),
        )
        index = STRATEGY_FEATURE_NAMES.index("spring_risk")
        assert np.all(features[:, index] == 1.0)

    def test_layout_hash_binds_order_version_and_normalization(self):
        assert len(STRATEGY_FEATURE_LAYOUT_HASH) == 64
        assert strategy_feature_layout_hash() == STRATEGY_FEATURE_LAYOUT_HASH
        assert strategy_feature_layout_hash(
            names=tuple(reversed(STRATEGY_FEATURE_NAMES))
        ) != STRATEGY_FEATURE_LAYOUT_HASH
        assert strategy_feature_layout_hash(
            version=STRATEGY_FEATURE_VERSION + "_changed"
        ) != STRATEGY_FEATURE_LAYOUT_HASH

    def test_public_observation_derives_non_pass_action_counts(self):
        from douzero.env.env import Env

        env = Env("adp")
        env.reset()
        pair = next(
            action
            for action in env.infoset.legal_actions
            if len(action) == 2 and action[0] == action[1]
        )
        env.step(pair)

        observation = get_obs_v2(env.infoset).public
        assert len(observation.played_cards["landlord"]) == 2
        assert observation.non_pass_action_counts == {
            "landlord": 1,
            "landlord_down": 0,
            "landlord_up": 0,
        }


class TestModelStrategyWiring:
    def test_default_off_preserves_p08_parameterization(self):
        torch.manual_seed(91)
        first = ModelV2(
            build_v2_schema(),
            ModelV2Config(hidden_size=16, history_encoder="lstm", history_layers=1),
        )
        torch.manual_seed(91)
        explicit = ModelV2(
            build_v2_schema(),
            ModelV2Config(
                hidden_size=16,
                history_encoder="lstm",
                history_layers=1,
                strategy_features_enabled=False,
                strategy_aux_enabled=False,
            ),
        )
        assert first.state_dict().keys() == explicit.state_dict().keys()
        for key in first.state_dict():
            assert torch.equal(first.state_dict()[key], explicit.state_dict()[key])

    def test_disabled_model_rejects_accidental_strategy_tensor(self):
        model = ModelV2(
            build_v2_schema(),
            ModelV2Config(hidden_size=16, history_encoder="lstm", history_layers=1),
        )
        with pytest.raises(ValueError, match="strategy-disabled"):
            model.action_encoder(torch.zeros(2, model._action_width), torch.zeros(2, 28))

    def test_enabled_model_outputs_all_aux_heads_for_all_roles(self):
        from douzero.env.env import Env

        env = Env("adp")
        env.reset()
        obs = get_obs_v2(env.infoset)
        cfg = ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            strategy_features_enabled=True,
            strategy_aux_enabled=True,
            strategy_node_budget=5,
            nan_guard=True,
        )
        model = ModelV2(build_v2_schema(), cfg)
        bundle = observation_to_model_inputs(obs, model.strategy_feature_config())
        for role in ("landlord", "landlord_up", "landlord_down"):
            out = model(
                bundle.state_card_vectors,
                bundle.state_context_flat,
                bundle.context_card_vectors,
                bundle.context_flat,
                bundle.history_tokens,
                bundle.history_key_padding_mask,
                bundle.action_features,
                bundle.action_mask,
                role,
                strategy_features=bundle.strategy_features,
            )
            assert out.min_turns_after.shape == (out.num_actions, 1)
            assert out.regain_initiative_logit.shape == (out.num_actions, 1)
            assert out.teammate_finish_logit.shape == (out.num_actions, 1)
            assert out.spring_probability_logit.shape == (out.num_actions, 1)
            assert out.structure_cost.shape == (out.num_actions, 1)

    def test_auxiliary_losses_backpropagate(self):
        predictions = {
            "min_turns_after": torch.rand(3, 1, requires_grad=True),
            "regain_initiative_logit": torch.randn(3, 1, requires_grad=True),
            "teammate_finish_logit": torch.randn(3, 1, requires_grad=True),
            "spring_probability_logit": torch.randn(3, 1, requires_grad=True),
            "structure_cost": torch.rand(3, 1, requires_grad=True),
        }
        targets = {
            "min_turns_after": torch.tensor([1.0, 2.0, 3.0]),
            "min_turns_exact_mask": torch.tensor([1.0, 1.0, 1.0]),
            "regain_initiative": torch.tensor([0.0, 1.0, 0.0]),
            "teammate_finish": torch.tensor([0.0, 1.0, 0.0]),
            "teammate_finish_mask": torch.tensor([0.0, 1.0, 1.0]),
            "spring_probability": torch.tensor([0.0, 0.0, 1.0]),
            "structure_cost": torch.tensor([0.0, 2.0, 1.0]),
        }
        cfg = LossConfig(
            lambda_win=0.0,
            lambda_score=0.0,
            lambda_min_turns=1.0,
            lambda_regain_initiative=1.0,
            lambda_teammate_finish=1.0,
            lambda_spring=1.0,
            lambda_structure=1.0,
        )
        loss = strategy_auxiliary_loss(predictions, targets, cfg)
        assert torch.isfinite(loss.total)
        loss.total.backward()
        assert all(value.grad is not None for value in predictions.values())

    def test_inexact_min_turn_target_is_masked(self):
        predictions = {
            "min_turns_after": torch.tensor([[100.0]], requires_grad=True),
            "regain_initiative_logit": torch.zeros(1, 1, requires_grad=True),
            "teammate_finish_logit": torch.zeros(1, 1, requires_grad=True),
            "spring_probability_logit": torch.zeros(1, 1, requires_grad=True),
            "structure_cost": torch.zeros(1, 1, requires_grad=True),
        }
        targets = {
            "min_turns_after": torch.tensor([1.0]),
            "min_turns_exact_mask": torch.tensor([0.0]),
            "regain_initiative": torch.tensor([0.0]),
            "teammate_finish": torch.tensor([0.0]),
            "teammate_finish_mask": torch.tensor([0.0]),
            "spring_probability": torch.tensor([0.0]),
            "structure_cost": torch.tensor([0.0]),
        }
        loss = strategy_auxiliary_loss(
            predictions,
            targets,
            LossConfig(lambda_win=0.0, lambda_score=0.0, lambda_min_turns=1.0),
        )
        assert loss.min_turns_after == 0.0
        loss.total.backward()
        assert predictions["min_turns_after"].grad.item() == 0.0

    @pytest.mark.parametrize(
        ("spring", "anti_spring", "expected"),
        [
            (True, False, 1.0),
            (False, True, 1.0),
            (False, False, 0.0),
        ],
    )
    def test_standard_terminal_spring_target_includes_anti_spring(
        self, spring, anti_spring, expected
    ):
        from douzero.env.env import Env
        from douzero.training.v2_buffer import Episode, Transition

        env = Env("adp")
        env.reset()
        observation = get_obs_v2(env.infoset)
        transition = Transition(
            obs=observation,
            action_index=0,
            position="landlord",
        )
        episode = Episode(
            transitions=[transition],
            terminal_result={
                "ruleset_id": "standard",
                "winner_team": "landlord" if spring else "farmer",
                "winner_position": "landlord" if spring else "landlord_down",
                "spring": spring,
                "anti_spring": anti_spring,
            },
        )

        episode.label_strategy_auxiliary(node_budget=5, time_budget_ms=0)

        assert transition.target_spring_probability == expected

    def test_v2_trainer_updates_auxiliary_heads(self, seed_factory):
        from douzero.training.v2_trainer import TrainerConfig, V2Trainer

        seed_factory(909)
        model = ModelV2(
            build_v2_schema(),
            ModelV2Config(
                hidden_size=16,
                history_encoder="lstm",
                history_layers=1,
                strategy_features_enabled=True,
                strategy_aux_enabled=True,
                strategy_node_budget=1,
                nan_guard=True,
            ),
        )
        before = model.strategy_aux_heads.structure_cost.weight.detach().clone()
        trainer = V2Trainer(
            model,
            loss_config=LossConfig(
                lambda_win=1.0,
                lambda_score=0.0,
                lambda_min_turns=0.1,
                lambda_regain_initiative=0.1,
                lambda_teammate_finish=0.1,
                lambda_spring=0.1,
                lambda_structure=0.1,
            ),
            config=TrainerConfig(
                seed=909,
                rng_seed=909,
                max_episodes=1,
                max_steps_per_episode=400,
                exp_epsilon=1.0,
                batch_size=2,
                learning_rate=1e-3,
                optimizer_steps=1,
                buffer_capacity=512,
            ),
        )
        stats = trainer.train()
        after = model.strategy_aux_heads.structure_cost.weight.detach()
        assert stats.optimizer_steps == 1
        assert "aux_loss_total" in stats.last_loss
        assert not torch.equal(before, after)

    def test_p08_checkpoint_migrates_only_when_strategy_disabled(self, tmp_path):
        from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
        from douzero.checkpoint.io import CheckpointCompatibilityError
        from douzero.checkpoint.v2 import _MODEL_CONFIG_IDENTITY_VERSION_KEY
        from douzero.env.rules import RuleSet

        base_cfg = ModelV2Config(
            hidden_size=16, history_encoder="lstm", history_layers=1
        )
        model = ModelV2(build_v2_schema(), base_cfg)
        path = str(tmp_path / "p08.tar")
        ruleset = RuleSet.legacy()
        save_v2_checkpoint(path, model, ruleset=ruleset)
        bundle = torch.load(path, weights_only=False)
        bundle[_MODEL_CONFIG_IDENTITY_VERSION_KEY] = 2
        bundle["model_config_hash"] = base_cfg.stable_hash_v2()
        torch.save(bundle, path)

        state, _ = load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=base_cfg.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=base_cfg,
        )
        assert state

        enabled_cfg = ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            strategy_features_enabled=True,
        )
        with pytest.raises(CheckpointCompatibilityError, match="predates P09"):
            load_v2_checkpoint(
                path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=enabled_cfg.stable_hash(),
                expected_ruleset=ruleset,
                runtime_model_config=enabled_cfg,
            )

    def test_strategy_checkpoint_roundtrip_is_exact(self, tmp_path):
        from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
        from douzero.env.rules import RuleSet

        cfg = ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            strategy_features_enabled=True,
            strategy_aux_enabled=True,
            strategy_node_budget=7,
        )
        model = ModelV2(build_v2_schema(), cfg)
        path = str(tmp_path / "p09.tar")
        ruleset = RuleSet.legacy()
        save_v2_checkpoint(path, model, ruleset=ruleset)
        state, _ = load_v2_checkpoint(
            path,
            expected_schema_hash=model.schema.stable_hash(),
            expected_model_config_hash=cfg.stable_hash(),
            expected_ruleset=ruleset,
            runtime_model_config=cfg,
        )
        clone = ModelV2(build_v2_schema(), cfg)
        clone.load_state_dict(state, strict=True)
        for key, value in model.state_dict().items():
            assert torch.equal(value, clone.state_dict()[key])

    def test_same_width_strategy_version_drift_rejects_checkpoint(
        self, tmp_path, monkeypatch
    ):
        from douzero.checkpoint import load_v2_checkpoint, save_v2_checkpoint
        from douzero.checkpoint.io import CheckpointCompatibilityError
        from douzero.env.rules import RuleSet
        import douzero.strategy.features as strategy_features

        cfg = ModelV2Config(
            hidden_size=16,
            history_encoder="lstm",
            history_layers=1,
            strategy_features_enabled=True,
        )
        model = ModelV2(build_v2_schema(), cfg)
        path = str(tmp_path / "strategy_v1.tar")
        ruleset = RuleSet.legacy()
        save_v2_checkpoint(path, model, ruleset=ruleset)
        old_hash = cfg.stable_hash()

        # Width and parameter shapes stay unchanged; only semantic version
        # changes. The config identity must still reject the old checkpoint.
        monkeypatch.setattr(
            strategy_features, "STRATEGY_FEATURE_VERSION", "strategy_v2"
        )
        new_hash = cfg.stable_hash()
        assert old_hash != new_hash
        with pytest.raises(CheckpointCompatibilityError, match="model_config_hash"):
            load_v2_checkpoint(
                path,
                expected_schema_hash=model.schema.stable_hash(),
                expected_model_config_hash=new_hash,
                expected_ruleset=ruleset,
                runtime_model_config=cfg,
            )


class TestUncertaintyGatedPrior:
    def test_alpha_zero_is_exactly_pure_score(self):
        output = _model_output(p_win=[0.5, 0.5], score=[0.1, 0.2], prior=[10.0, -10.0])
        assert select_action(output, DecisionConfig(mode="uncertainty_gated_prior")) == 1
        assert select_action(output, DecisionConfig(mode="pure_score")) == 1

    def test_prior_only_influences_uncertain_action(self):
        output = _model_output(
            p_win=[0.99, 0.5], score=[0.1, 0.1], prior=[-2.0, 2.0]
        )
        assert select_action(
            output,
            DecisionConfig(mode="uncertainty_gated_prior", prior_alpha=1.0),
        ) == 1
