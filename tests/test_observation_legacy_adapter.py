"""Legacy adapter parity tests (P03, hardened round 2 — item 7).

The V2 → legacy adapter (:func:`legacy_observation_from_v2`) must reconstruct
the legacy ``x_batch`` / ``x_no_action`` / ``z`` / ``z_batch`` tensors from an
:class:`ObservationV2` ALONE — no extra infoset/history argument (item 7).

These tests build the same infoset with both the legacy encoder (``get_obs``)
and the V2 encoder + adapter, then assert the tensors are identical across:
all three roles, short history, long history (> 15 steps, exercising the
legacy 15-move windowing), and varying legal-action counts.
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


def _drive_to_role(seed: int, target_role: str, data, max_steps: int = 6):
    """Build a GameEnv and step until ``target_role`` is the acting player."""
    players = {pos: _NoopAgent() for pos in
               ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    env.card_play_init(data)
    for _ in range(max_steps):
        if env.acting_player_position == target_role:
            break
        action = env.game_infoset.legal_actions[0]
        players[env.acting_player_position].set_action(action)
        env.step()
    return env.game_infoset, env.card_play_action_seq


def _drive_n_steps(seed: int, data, n: int):
    """Build a GameEnv and step exactly ``n`` times, returning the infoset."""
    players = {pos: _NoopAgent() for pos in
               ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players)
    env.card_play_init(data)
    for _ in range(n):
        action = env.game_infoset.legal_actions[0]
        players[env.acting_player_position].set_action(action)
        env.step()
    return env.game_infoset


@pytest.mark.parametrize("role", ["landlord", "landlord_up", "landlord_down"])
def test_x_no_action_parity_for_each_role(role, fixed_card_play_data):
    """The adapter's x_no_action must equal the legacy encoder's x_no_action."""
    infoset, _ = _drive_to_role(0, role, fixed_card_play_data)
    assert infoset.player_position == role

    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs)  # item 7: no extra args

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
    adapted = legacy_observation_from_v2(v2_obs)

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
    adapted = legacy_observation_from_v2(v2_obs)

    np.testing.assert_array_equal(
        adapted["z"], legacy_obs["z"],
        err_msg=f"z mismatch for role {role}",
    )
    np.testing.assert_array_equal(
        adapted["z_batch"], legacy_obs["z_batch"],
        err_msg=f"z_batch mismatch for role {role}",
    )


def test_z_parity_long_history_over_15_moves(fixed_card_play_data):
    """z parity must hold when the history exceeds the legacy 15-move window.

    Drives > 15 steps so the legacy ``_process_action_seq`` (length=15) windows
    the history; the adapter must reproduce the SAME windowed matrix.
    """
    infoset = _drive_n_steps(0, fixed_card_play_data, n=18)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs)
    np.testing.assert_array_equal(adapted["z"], legacy_obs["z"])
    np.testing.assert_array_equal(adapted["z_batch"], legacy_obs["z_batch"])
    # Confirm we actually exercised the window (>= 15 actions played).
    assert len(infoset.card_play_action_seq) >= 15


def test_adapter_dtype_matches_legacy(fixed_card_play_data):
    """The adapted dtypes must match the legacy encoder's dtypes exactly."""
    infoset, _ = _drive_to_role(0, "landlord", fixed_card_play_data)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs)
    assert adapted["x_batch"].dtype == legacy_obs["x_batch"].dtype == np.float32
    assert adapted["z_batch"].dtype == legacy_obs["z_batch"].dtype == np.float32
    assert adapted["x_no_action"].dtype == legacy_obs["x_no_action"].dtype == np.int8
    assert adapted["z"].dtype == legacy_obs["z"].dtype == np.int8


def test_adapter_preserves_legal_action_order(fixed_card_play_data):
    """The adapted legal_actions must match the legacy encoder's in order."""
    infoset, _ = _drive_to_role(0, "landlord", fixed_card_play_data)
    legacy_obs = get_obs(infoset)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs)
    assert adapted["legal_actions"] == legacy_obs["legal_actions"]


def test_adapter_takes_no_extra_arguments(fixed_card_play_data):
    """legacy_observation_from_v2 must accept ONLY the observation (item 7)."""
    import inspect
    sig = inspect.signature(legacy_observation_from_v2)
    params = list(sig.parameters)
    assert params == ["obs"], (
        f"legacy_observation_from_v2 must take only 'obs', got {params}"
    )

    infoset, _ = _drive_to_role(0, "landlord", fixed_card_play_data)
    v2_obs = get_obs_v2(infoset)
    # Passing an extra keyword must fail (the contract forbids it).
    with pytest.raises(TypeError):
        legacy_observation_from_v2(
            v2_obs, card_play_action_seq=infoset.card_play_action_seq)


def test_adapter_position_matches(fixed_card_play_data):
    infoset, _ = _drive_to_role(0, "landlord_up", fixed_card_play_data)
    v2_obs = get_obs_v2(infoset)
    adapted = legacy_observation_from_v2(v2_obs)
    assert adapted["position"] == "landlord_up"


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


def test_adapter_parity_across_seeds_and_roles():
    """Adapter parity must hold across many random deals and all three roles.

    Seeds a fresh Env, advances a few steps, and checks x_batch/z parity for
    whichever role is acting. This exercises varying legal-action counts.
    """
    for seed in range(12):
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
        # Advance a few steps to vary the acting role and history length.
        for _ in range(seed % 5):
            env.players[env._acting_player_position].set_action(
                env.infoset.legal_actions[0])
            env.step(env.infoset.legal_actions[0])
        infoset = env.infoset
        legacy_obs = get_obs(infoset)
        v2_obs = get_obs_v2(infoset)
        adapted = legacy_observation_from_v2(v2_obs)
        np.testing.assert_array_equal(adapted["x_batch"], legacy_obs["x_batch"],
                                      err_msg=f"x_batch seed {seed} role {infoset.player_position}")
        np.testing.assert_array_equal(adapted["z_batch"], legacy_obs["z_batch"],
                                      err_msg=f"z_batch seed {seed}")
        np.testing.assert_array_equal(adapted["x_no_action"], legacy_obs["x_no_action"],
                                      err_msg=f"x_no_action seed {seed}")
