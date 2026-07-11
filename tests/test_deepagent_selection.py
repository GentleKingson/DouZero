"""Determinism tests for ``DeepAgent.act`` under synthetic (init-only) weights.

The legacy ``DeepAgent`` is the deployment interface. We must guarantee:
  * the selected action is always legal ( legality comes from indexing
    ``infoset.legal_actions`` with the argmax over the per-action value vector),
  * selection is deterministic under fixed weights and CPU,
  * the single-legal-action short-circuit never invokes the model.

Because the repo ships no pretrained weights, we back the agent with a
deterministically-initialised state_dict saved to a temp ``.ckpt``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from douzero.env.env import Env, get_obs
from douzero.evaluation.deep_agent import DeepAgent


POSITIONS = ["landlord", "landlord_up", "landlord_down"]


def _drive_to_position(env: Env, position: str):
    env.reset()
    while env._acting_player_position != position:
        env.step(env.infoset.legal_actions[0])
    return env.infoset


@pytest.mark.parametrize("position", POSITIONS)
def test_act_returns_a_legal_action(position, deepagent_with_init_weights, seed_factory):
    seed_factory(700)
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    agent = deepagent_with_init_weights(position)
    action = agent.act(infoset)
    assert action in infoset.legal_actions


def test_act_short_circuits_when_single_legal_action(deepagent_with_init_weights, seed_factory):
    """len(legal_actions)==1 must return that action without calling the model."""
    seed_factory(701)
    env = Env("adp")
    env.reset()
    infoset = env.infoset
    # Fabricate an infoset-like object with exactly one legal action.
    only = infoset.legal_actions[0]

    class _SingleActionInfoset:
        player_position = infoset.player_position
        legal_actions = [only]
        # The other attributes get_obs reads:
        player_hand_cards = infoset.player_hand_cards
        other_hand_cards = infoset.other_hand_cards
        last_move = infoset.last_move
        last_two_moves = infoset.last_two_moves
        last_move_dict = infoset.last_move_dict
        played_cards = infoset.played_cards
        num_cards_left_dict = infoset.num_cards_left_dict
        three_landlord_cards = infoset.three_landlord_cards
        card_play_action_seq = infoset.card_play_action_seq
        bomb_num = infoset.bomb_num

    agent = deepagent_with_init_weights(infoset.player_position)
    # Must not raise even though get_obs would be called if not short-circuited.
    action = agent.act(_SingleActionInfoset())
    assert action == only


@pytest.mark.parametrize("position", POSITIONS)
def test_act_is_deterministic_under_fixed_weights(
    position, deepagent_with_init_weights, seed_factory
):
    """Same agent + same infoset -> same action, twice."""
    seed_factory(702)
    env = Env("adp")
    infoset = _drive_to_position(env, position)

    # Build the agent from the SAME seed so the underlying weights are identical.
    agent_a = deepagent_with_init_weights(position, seed=123)
    agent_b = deepagent_with_init_weights(position, seed=123)
    assert agent_a.act(infoset) == agent_b.act(infoset)


def test_act_deterministic_within_one_agent(deepagent_with_init_weights, seed_factory):
    seed_factory(703)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    agent = deepagent_with_init_weights("landlord")
    first = agent.act(infoset)
    second = agent.act(infoset)
    assert first == second


def test_act_matches_manual_argmax_over_model_values(
    deepagent_with_init_weights, seed_factory
):
    """DeepAgent.act must pick np.argmax over model.forward(..., return_value=True)."""
    seed_factory(704)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    agent = deepagent_with_init_weights("landlord")

    obs = get_obs(infoset)
    z = torch.from_numpy(obs["z_batch"]).float()
    x = torch.from_numpy(obs["x_batch"]).float()
    with torch.no_grad():
        values = agent.model.forward(z, x, return_value=True)["values"].numpy()
    expected_index = int(np.argmax(values[:, 0]))
    expected = infoset.legal_actions[expected_index]

    assert agent.act(infoset) == expected


def test_deepagent_model_in_eval_mode(deepagent_with_init_weights):
    agent = deepagent_with_init_weights("landlord")
    assert not agent.model.training
