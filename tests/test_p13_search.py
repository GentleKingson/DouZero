"""P13 belief-search, endgame correctness, budgets, and leakage boundaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.output import ModelOutput
from douzero.observation.encode_v2 import get_obs_v2
from douzero.search import (
    BeliefSearch,
    EndgameSolver,
    SearchBudget,
    SearchConfig,
    SearchGameState,
)
from douzero.search.belief_rollout import CandidateValue
from douzero.search.candidate import Candidate


def _state(**overrides) -> SearchGameState:
    values = {
        "hands": {
            "landlord": (3,),
            "landlord_down": (4,),
            "landlord_up": (5,),
        },
        "acting_role": "landlord",
        "last_move": (),
        "last_non_pass_role": None,
        "consecutive_passes": 0,
        "ruleset": RuleSet.legacy(),
    }
    values.update(overrides)
    return SearchGameState(**values)


def _output(num_actions: int) -> ModelOutput:
    p_win = torch.linspace(0.9, 0.1, num_actions).unsqueeze(-1)
    score = torch.linspace(1.0, -1.0, num_actions).unsqueeze(-1)
    return ModelOutput(
        win_logit=torch.logit(p_win.clamp(0.01, 0.99)),
        score_if_win=torch.ones_like(score),
        score_if_loss=-torch.ones_like(score),
        p_win=p_win,
        score_mean=score,
        action_mask=torch.ones(num_actions, dtype=torch.bool),
    )


def test_exact_small_endgame_matches_forced_result():
    state = _state()
    cfg = SearchConfig(
        enabled=True, max_nodes=1000, max_rollouts=1, max_milliseconds=1000
    )
    value = EndgameSolver(SearchBudget(cfg)).solve(state, "landlord")
    assert value.win_probability == 1.0
    assert value.expected_score == 2.0


def test_solver_cache_is_scoped_by_root_team():
    state = _state()
    cfg = SearchConfig(
        enabled=True, max_nodes=1000, max_rollouts=1, max_milliseconds=1000
    )
    solver = EndgameSolver(SearchBudget(cfg))

    landlord = solver.solve(state, "landlord")
    farmer = solver.solve(state, "farmer")

    assert landlord.win_probability == 1.0
    assert landlord.expected_score == 2.0
    assert farmer.win_probability == 0.0
    assert farmer.expected_score == -1.0


def test_farmer_team_uses_shared_utility_and_passes():
    # landlord cannot beat 4, then landlord_down can shed 5 and win for both
    # farmers. The last move belongs to landlord_up, so the pass context is valid.
    state = _state(
        hands={
            "landlord": (3,),
            "landlord_down": (5,),
            "landlord_up": (4, 6),
        },
        last_move=(4,),
        last_non_pass_role="landlord_up",
    )
    cfg = SearchConfig(
        enabled=True, max_nodes=1000, max_rollouts=1, max_milliseconds=1000
    )
    value = EndgameSolver(SearchBudget(cfg)).solve(state, "farmer")
    assert value.win_probability == 1.0
    assert value.expected_score == 1.0


def test_heuristic_uses_canonical_team_score_units():
    state = _state(hands={
        "landlord": (3,),
        "landlord_down": (4, 5, 6),
        "landlord_up": (7, 8, 9),
    })
    landlord = state.heuristic_value("landlord")
    farmer = state.heuristic_value("farmer")
    assert abs(landlord.expected_score) == 2 * abs(farmer.expected_score)


@pytest.mark.parametrize("bid", [1, 2, 3])
def test_standard_heuristic_applies_bid_and_team_ratio(bid):
    state = _state(
        hands={
            "landlord": (3,),
            "landlord_down": (4, 5, 6),
            "landlord_up": (7, 8, 9),
        },
        ruleset=RuleSet.standard(),
        bid_value=bid,
    )
    landlord = abs(state.heuristic_value("landlord").expected_score)
    farmer = abs(state.heuristic_value("farmer").expected_score)
    assert landlord == 2 * farmer
    base_state = replace(state, bid_value=1)
    assert farmer == bid * abs(base_state.heuristic_value("farmer").expected_score)


def test_heuristic_separates_bomb_rocket_and_caps_multiplier():
    ruleset = replace(
        RuleSet.standard(),
        bomb_multiplier=3,
        rocket_multiplier=5,
        max_multiplier=10,
    )
    state = _state(
        ruleset=ruleset,
        bid_value=2,
        bomb_count=1,
        rocket_count=1,
        hands={
            "landlord": (3,),
            "landlord_down": (4, 5, 6),
            "landlord_up": (7, 8, 9),
        },
    )
    # Uncapped multiplier is 2 * 3 * 5 = 30, capped to 10.
    assert abs(state.heuristic_value("landlord").expected_score) == 20 * (
        2.0 * state.heuristic_value("landlord").win_probability - 1.0
    )


def test_standard_terminal_solver_scores_spring_and_anti_spring():
    spring = _state(
        hands={
            "landlord": (),
            "landlord_down": (4,),
            "landlord_up": (5,),
        },
        ruleset=RuleSet.standard(),
        bid_value=2,
        action_counts={
            "landlord": 1, "landlord_down": 0, "landlord_up": 0,
        },
        winner_position="landlord",
    )
    assert spring.terminal_value("landlord").expected_score == 8.0

    anti = replace(
        spring,
        hands={
            "landlord": (3,),
            "landlord_down": (),
            "landlord_up": (5,),
        },
        action_counts={
            "landlord": 1, "landlord_down": 1, "landlord_up": 1,
        },
        winner_position="landlord_down",
    )
    assert anti.terminal_value("farmer").expected_score == 4.0


def test_search_state_apply_is_independent_and_legal():
    state = _state(hands={
        "landlord": (3, 3),
        "landlord_down": (4,),
        "landlord_up": (5,),
    })
    before = dict(state.hands)
    child = state.apply((3,))
    assert dict(state.hands) == before
    assert state.hands["landlord"] == (3, 3)
    assert child.hands["landlord"] == (3,)
    assert (3,) in state.legal_actions()
    with pytest.raises(ValueError, match="illegal action"):
        state.apply((30,))


def test_budget_zero_returns_base_without_touching_belief_model():
    config = SearchConfig(enabled=True, max_nodes=0)
    actions = ((3,), (4,))
    observation = SimpleNamespace(
        public=SimpleNamespace(kind="public"),
        actions=SimpleNamespace(legal_actions=actions),
    )
    decision = BeliefSearch(config, RuleSet.legacy()).select(
        observation=observation,
        model_output=_output(2),
        base_action_index=1,
        belief_model=object(),
    )
    assert decision.action_index == 1
    assert decision.log.searched_action == (4,)
    assert decision.log.timed_out


def test_timeout_log_retains_generated_sample_count():
    env = Env("adp")
    env.reset()
    obs = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
    config = SearchConfig(
        enabled=True,
        top_k=2,
        belief_samples=1,
        rollout_depth=0,
        endgame_cards_threshold=0,
        max_nodes=100,
        max_rollouts=1,
        max_milliseconds=2000,
    )
    decision = BeliefSearch(config, RuleSet.legacy()).select(
        observation=obs,
        model_output=_output(len(obs.actions.legal_actions)),
        base_action_index=0,
        belief_model=model,
    )
    assert decision.log.timed_out
    assert decision.log.samples == 1
    assert decision.log.rollouts == 1


def test_seeded_search_is_deterministic_and_public_only(monkeypatch):
    env = Env("adp")
    env.reset()
    infoset = env.infoset
    obs = get_obs_v2(infoset, ruleset=RuleSet.legacy())
    # A different privileged hidden allocation cannot alter an already-built
    # public observation. Search has no infoset/privileged argument at all.
    changed = deepcopy(infoset)
    changed.all_handcards["landlord_up"], changed.all_handcards["landlord_down"] = (
        changed.all_handcards["landlord_down"],
        changed.all_handcards["landlord_up"],
    )
    obs_changed = get_obs_v2(changed, ruleset=RuleSet.legacy())
    assert obs.public.other_handcards == obs_changed.public.other_handcards

    model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
    config = SearchConfig(
        enabled=True,
        top_k=2,
        belief_samples=2,
        rollout_depth=1,
        endgame_cards_threshold=0,
        max_nodes=100,
        max_rollouts=4,
        max_milliseconds=2000,
        seed=123,
    )
    output = _output(len(obs.actions.legal_actions))
    search = BeliefSearch(config, RuleSet.legacy())
    original_apply = SearchGameState.apply
    validations = []

    def tracked_apply(self, action, *, validate=True):
        validations.append(validate)
        return original_apply(self, action, validate=validate)

    monkeypatch.setattr(SearchGameState, "apply", tracked_apply)
    first = search.select(
        observation=obs,
        model_output=output,
        base_action_index=0,
        belief_model=model,
    )
    second = search.select(
        observation=obs_changed,
        model_output=output,
        base_action_index=0,
        belief_model=model,
    )
    assert first.action_index == second.action_index
    assert first.log.candidate_values == second.log.candidate_values
    assert first.log.searched_action in obs.actions.legal_actions
    assert validations and not any(validations)


def test_score_selection_uses_corrected_expected_score():
    search = BeliefSearch(
        SearchConfig(enabled=True, selection_mode="score"), RuleSet.legacy()
    )
    candidates = (
        Candidate(0, (3,), 0.9, 1.0),
        Candidate(1, (4,), 0.8, 2.0),
    )
    values = [
        CandidateValue((3,), 0.5, 1.0, 0.5, 2),
        CandidateValue((4,), 0.5, 2.0, 0.5, 2),
    ]
    assert search._choose(candidates, values).index == 1


def test_standard_infoset_and_deep_agent_act_preserve_scoring_state(monkeypatch):
    from douzero.evaluation.deep_agent import DeepAgentV2
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    ruleset = replace(
        RuleSet.standard(), bomb_multiplier=3, rocket_multiplier=5
    )
    env = Env("adp", ruleset=ruleset)
    env.reset()
    env.step(None, bid_value=1)
    env.step(None, bid_value=2)
    env.step(None, bid_value=3)
    env._env.bomb_count = 1
    env._env.rocket_count = 1
    env._env.bomb_num = 2
    env._env.action_counts = {
        "landlord": 2, "landlord_down": 1, "landlord_up": 1,
    }
    env._env.game_infoset = env._env.get_infoset()
    infoset = env._env.game_infoset

    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=32, history_heads=4, history_layers=1),
    )
    agent = DeepAgentV2("landlord", model, ruleset)
    captured = {}

    def select_first(obs):
        captured["public"] = obs.public
        return obs.actions.legal_actions[0]

    monkeypatch.setattr(agent, "_select_from_observation", select_first)
    chosen = agent.act(infoset)
    assert chosen in infoset.legal_actions
    public = captured["public"]
    assert public.bid_value == 3
    assert public.bomb_count == 1
    assert public.rocket_count == 1
    assert dict(public.non_pass_action_counts) == env._env.action_counts
    assert public.total_multiplier == 45  # bid 3 * bomb 3 * rocket 5


def test_legacy_observation_context_remains_pre_p13_compatible():
    env = Env("adp")
    env.reset()
    env._env.bomb_num = 2
    env._env.bomb_count = 1
    env._env.rocket_count = 1
    infoset = env._env.get_infoset()
    obs = get_obs_v2(infoset, ruleset=RuleSet.legacy())
    # Legacy Model V2 historically consumed the conflated bomb count and no
    # rocket/total-multiplier context. Search scoring remains equivalent because
    # legacy compute_game_result combines bomb_count + rocket_count.
    assert obs.public.bomb_count == 2
    assert obs.public.rocket_count == 0
    assert obs.public.total_multiplier == 1


def test_search_only_belief_forward_is_inside_time_budget(monkeypatch):
    from douzero.evaluation.deep_agent import DeepAgentV2
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    env = Env("adp")
    env.reset()
    obs = get_obs_v2(env.infoset, ruleset=RuleSet.legacy())
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(hidden_size=32, history_heads=4, history_layers=1),
    )
    belief = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
    original_forward = belief.forward

    def slow_forward(*args, **kwargs):
        time.sleep(0.02)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(belief, "forward", slow_forward)
    agent = DeepAgentV2(
        "landlord",
        model,
        RuleSet.legacy(),
        belief_model=belief,
        search_config=SearchConfig(
            enabled=True,
            max_nodes=100,
            max_rollouts=2,
            max_milliseconds=1,
            belief_samples=1,
        ),
    )
    chosen = agent.act_v2(obs)
    assert chosen in obs.actions.legal_actions
    assert agent.last_search_log.timed_out
    assert agent.last_search_log.elapsed_milliseconds >= 20.0


def test_single_legal_action_clears_stale_search_log():
    from douzero.evaluation.deep_agent import DeepAgentV2
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    env = Env("adp")
    env.reset()
    infoset = env.infoset
    infoset.legal_actions = infoset.legal_actions[:1]
    agent = DeepAgentV2(
        "landlord",
        ModelV2(
            build_v2_schema(),
            ModelV2Config(hidden_size=32, history_heads=4, history_layers=1),
        ),
        RuleSet.legacy(),
    )
    agent.last_search_log = object()
    agent.act(infoset)
    assert agent.last_search_log is None


def test_search_config_loads_and_defaults_off(tmp_path):
    from douzero.config import load_config

    path = tmp_path / "search.yaml"
    path.write_text(
        "search:\n  enabled: true\n  top_k: 2\n  max_milliseconds: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.search.enabled
    assert cfg.search.top_k == 2
    assert TrainingDefault().search.enabled is False


from douzero.config.schemas import TrainingConfig as TrainingDefault
