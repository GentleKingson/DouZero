"""P13 belief-search, endgame correctness, budgets, and leakage boundaries."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

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


def test_seeded_search_is_deterministic_and_public_only():
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
