"""Portable baseline invariants frozen into the test suite.

``tools/capture_baseline.py`` writes a rich JSON artifact, but that artifact is
git-ignored (it carries host-specific float hashes). The contract that must
survive across machines / commits lives HERE, as integer/string invariants that
do not depend on float reproducibility or torch build:

  * observation shapes and dtypes (per role),
  * role→feature-width mapping (373 landlord / 484 farmer),
  * the legal-action count for the fixed opening deal,
  * the exact set of TYPE_15_WRONG exceptions on the fixed deal,
  * winner/bomb/card-conservation terminal invariants.

Float model outputs are checked for *same-process determinism* in
``test_model_shapes`` (identical output bytes across two runs in one
environment) and for *tolerance* comparison in
``test_baseline_float_tolerance`` below. Raw output hashes from the capture
tool are a same-environment diagnostic ONLY, never a cross-machine hard gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from douzero.dmc.models import LandlordLstmModel, FarmerLstmModel, model_dict
from douzero.env.env import Env, get_obs
from douzero.env.move_detector import get_move_type


POSITIONS = ["landlord", "landlord_up", "landlord_down"]

# ---- Frozen, environment-independent invariants --------------------------- #

#: Role -> total x width (x_no_action + 54-dim action column).
FROZEN_X_WIDTH = {"landlord": 373, "landlord_up": 484, "landlord_down": 484}

#: Role -> x_no_action width (no batch dim, no action column).
FROZEN_X_NO_ACTION_WIDTH = {"landlord": 319, "landlord_up": 430, "landlord_down": 430}

#: History tensor shape (z), identical across roles and runs.
FROZEN_Z_SHAPE = (5, 162)

#: Exact, pre-computed TYPE_15_WRONG exception set for the fixed P00 deal's
#: landlord hand ([3,4,5,6,7] x4). See test_legal_actions_snapshot for the
#: full attribution and gameplay-impact discussion.
FROZEN_OPENING_WRONG_EXCEPTIONS = {
    (3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7),
    (3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7),
}


# --------------------------------------------------------------------------- #
# Shapes and widths (portable integers)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("position", POSITIONS)
def test_frozen_x_widths_match_model_dense1(position):
    """The frozen x widths must match the model's dense1 input (minus LSTM)."""
    model = model_dict[position]()
    expected = FROZEN_X_WIDTH[position] + 128  # lstm hidden concatenated
    assert model.dense1.in_features == expected


def test_frozen_z_shape_matches_lstm_input():
    for cls in (LandlordLstmModel, FarmerLstmModel):
        m = cls()
        assert m.lstm.input_size == FROZEN_Z_SHAPE[1]
        assert m.lstm.hidden_size == 128


@pytest.mark.parametrize("position", POSITIONS)
def test_frozen_obs_shapes_and_dtypes(position, seed_factory):
    """get_obs output shapes/dtypes must match the frozen baseline for each role."""
    seed_factory(4242)
    env = Env("adp")
    env.reset()
    while env._acting_player_position != position:
        env.step(env.infoset.legal_actions[0])
    obs = get_obs(env.infoset)

    assert obs["position"] == position
    assert obs["x_batch"].shape[1] == FROZEN_X_WIDTH[position]
    assert obs["x_no_action"].shape == (FROZEN_X_NO_ACTION_WIDTH[position],)
    assert obs["z_batch"].shape[1:] == FROZEN_Z_SHAPE
    assert obs["z"].shape == FROZEN_Z_SHAPE

    # dtypes are part of the frozen contract.
    assert obs["x_batch"].dtype == np.float32
    assert obs["z_batch"].dtype == np.float32
    assert obs["x_no_action"].dtype == np.int8
    assert obs["z"].dtype == np.int8


# --------------------------------------------------------------------------- #
# Fixed-deal opening invariants (portable integers/sets)
# --------------------------------------------------------------------------- #

def test_frozen_opening_wrong_exception_set(fixed_card_play_data):
    """The fixed deal's TYPE_15_WRONG legal actions must equal the frozen set."""
    from douzero.env.game import GameEnv

    class _Stub:
        def __init__(self):
            self.action = None

        def set_action(self, a):
            self.action = a

        def act(self, infoset):
            return infoset.legal_actions[0]

    players = {p: _Stub() for p in POSITIONS}
    env = GameEnv(players)
    env.card_play_init(fixed_card_play_data)

    observed = {
        tuple(sorted(a))
        for a in env.game_infoset.legal_actions
        if get_move_type(a)["type"] == 15
    }
    assert observed == FROZEN_OPENING_WRONG_EXCEPTIONS


# --------------------------------------------------------------------------- #
# Float determinism vs tolerance contract
# --------------------------------------------------------------------------- #

def test_baseline_float_tolerance_same_seed_two_models(seed_factory):
    """Two models from the SAME seed produce outputs within a tight tolerance.

    This is the cross-instantiation form of the determinism contract: within a
    single environment the outputs are byte-equal (see
    test_model_shapes.test_fixed_init_output_hash_is_stable_across_runs); the
    tolerance here guards against any future non-determinism that stays within
    float noise. It is NOT a cross-machine gate.
    """

    def build_and_run():
        seed_factory(7)
        env = Env("adp")
        env.reset()
        obs = get_obs(env.infoset)
        z = torch.from_numpy(obs["z_batch"]).float()
        x = torch.from_numpy(obs["x_batch"]).float()
        torch.manual_seed(7)
        m = LandlordLstmModel()
        m.eval()
        with torch.no_grad():
            return m(z, x, return_value=True)["values"].numpy()

    a = build_and_run()
    b = build_and_run()
    # Same seed, same env -> numerically identical (atol/rtol float-floor only).
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    # And the stronger byte-equality must hold too.
    assert np.array_equal(a, b)
