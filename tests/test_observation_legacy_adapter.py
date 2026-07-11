"""Legacy adapter parity tests (P03).

The V2 → legacy adapter (:func:`legacy_observation_from_v2`) must reconstruct
the legacy ``x_batch`` / ``x_no_action`` / ``z`` / ``z_batch`` tensors from a V2
observation byte-for-byte, so a legacy model can consume a V2 observation
without any model-side change.

These tests build the same infoset with both the legacy encoder (``get_obs``)
and the V2 encoder + adapter, then assert the tensors are identical for each of
the three roles.
"""

from __future__ import annotations

import numpy as np
import pytest

from douzero.env.env import Env, get_obs
from douzero.env.game import GameEnv
from douzero.observation import get_obs_v2, legacy_observation_from_v2


class _NoopAgent:
    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def act(self, infoset):
        if self.action is not None and self.action in infoset.legal_actions:
            a, self.action = self.action, None
            return a
        return infoset.legal_actions[0]


def _drive_to_role(seed: int, target_role: str, data):
    """Build a GameEnv and step until ``target_role`` is the acting player.

    Returns the infoset at that point. Steps use the first legal action so the
    state advances deterministically.
    """
    players = {pos: _NoopAgent() for pos in
               ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    env.card_play_init(data)

    # Step until the target role is acting (or we run out of budget).
    for _ in range(6):
        if env.acting_player_position == target_role:
            break
        action = env.game_infoset.legal_actions[0]
        players[env.acting_player_position].set_action(action)
        env.step()
    return env.game_infoset, env.card_play_action_seq


@pytest.mark.parametrize("role", ["landlord", "landlord_up", "landlord_down"])
def test_x_no_action_parity_for_each_role(role, fixed_card_play_data):
    """The adapter's x_no_action must equal the legacy encoder's x_no_action."""
    infoset, _ = _drive_to_role(0, role, fixed_card_play_data)
    assert infoset.player_position == role

    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs, card_play_action_seq=infoset.card_play_action_seq)

    np.testing.assert_array_equal(
        adapted["x_no_action"], legacy_obs["x_no_action"],
        err_msg=f"x_no_action mismatch for role {role}",
    )


@pytest.mark.parametrize("role", ["landlord", "landlord_up", "landlord_down"])
def test_x_batch_parity_for_each_role(role, fixed_card_play_data):
    """The adapter's x_batch must equal the legacy encoder's x_batch."""
    infoset, _ = _drive_to_role(0, role, fixed_card_play_data)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs, card_play_action_seq=infoset.card_play_action_seq)

    np.testing.assert_array_equal(
        adapted["x_batch"], legacy_obs["x_batch"],
        err_msg=f"x_batch mismatch for role {role}",
    )


@pytest.mark.parametrize("role", ["landlord", "landlord_up", "landlord_down"])
def test_z_parity_for_each_role(role, fixed_card_play_data):
    """The adapter's z history matrix must equal the legacy encoder's z."""
    infoset, _ = _drive_to_role(0, role, fixed_card_play_data)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs, card_play_action_seq=infoset.card_play_action_seq)

    np.testing.assert_array_equal(
        adapted["z"], legacy_obs["z"],
        err_msg=f"z mismatch for role {role}",
    )
    np.testing.assert_array_equal(
        adapted["z_batch"], legacy_obs["z_batch"],
        err_msg=f"z_batch mismatch for role {role}",
    )


def test_legacy_widths_derived_from_constants():
    """The legacy widths (319/373/430/484) are derived from named constants."""
    from douzero.observation.legacy_adapter import (
        FARMER_X_BATCH_WIDTH,
        FARMER_X_NO_ACTION_WIDTH,
        LANDLORD_X_BATCH_WIDTH,
        LANDLORD_X_NO_ACTION_WIDTH,
    )
    assert LANDLORD_X_NO_ACTION_WIDTH == 319
    assert LANDLORD_X_BATCH_WIDTH == 373
    assert FARMER_X_NO_ACTION_WIDTH == 430
    assert FARMER_X_BATCH_WIDTH == 484

    # And they must match the schema helper derivations.
    from douzero.observation.schema import (
        legacy_farmer_state_width,
        legacy_landlord_state_width,
    )
    assert legacy_landlord_state_width() == 319
    assert legacy_farmer_state_width() == 430


def test_adapter_preserves_legal_actions(fixed_card_play_data):
    """The adapted obs legal_actions must match the legacy encoder's."""
    infoset, _ = _drive_to_role(0, "landlord", fixed_card_play_data)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs, card_play_action_seq=infoset.card_play_action_seq)
    assert adapted["legal_actions"] == legacy_obs["legal_actions"]


def test_adapter_position_matches(fixed_card_play_data):
    infoset, _ = _drive_to_role(0, "landlord_up", fixed_card_play_data)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs, card_play_action_seq=infoset.card_play_action_seq)
    assert adapted["position"] == "landlord_up"
