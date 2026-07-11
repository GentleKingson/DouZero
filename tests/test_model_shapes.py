"""Shape and numerical-stability tests for the legacy role models.

The legacy architecture is:
  LSTM(162 -> 128) over z (5x162), then concat with x (373 or 484) -> 6 dense
  layers (512) -> scalar value per legal action.

These tests freeze those dimensions and confirm forward passes are finite and
deterministic under fixed initialisation. No pretrained weights are loaded.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from douzero.dmc.models import FarmerLstmModel, LandlordLstmModel, Model, model_dict
from douzero.env.env import Env, get_obs


POSITIONS = ["landlord", "landlord_up", "landlord_down"]
EXPECTED_X_DIM = {"landlord": 373, "landlord_up": 484, "landlord_down": 484}


def _to_tensors(obs):
    z = torch.from_numpy(obs["z_batch"]).float()
    x = torch.from_numpy(obs["x_batch"]).float()
    return z, x


@pytest.mark.parametrize("position", POSITIONS)
def test_model_forward_output_shape(position, seed_factory, tmp_path):
    seed_factory(2024)
    env = Env("adp")
    # Drive the env until it is ``position``'s turn.
    env.reset()
    while env._acting_player_position != position:
        env.step(env.infoset.legal_actions[0])

    obs = get_obs(env.infoset)
    z, x = _to_tensors(obs)

    assert obs["x_batch"].shape[1] == EXPECTED_X_DIM[position]
    assert obs["z_batch"].shape[1:] == (5, 162)

    torch.manual_seed(7)
    model = model_dict[position]()
    model.eval()
    with torch.no_grad():
        out = model(z, x, return_value=True)
    assert out["values"].shape == (z.shape[0], 1)


def test_landlord_dense1_input_width_is_373_plus_128():
    model = LandlordLstmModel()
    # dense1 sees [lstm_hidden(128)] + [x(373)] = 501.
    assert model.dense1.in_features == 373 + 128
    assert model.dense6.out_features == 1


def test_farmer_dense1_input_width_is_484_plus_128():
    model = FarmerLstmModel()
    assert model.dense1.in_features == 484 + 128
    assert model.dense6.out_features == 1


def test_lstm_input_size_matches_z_width():
    for cls in (LandlordLstmModel, FarmerLstmModel):
        m = cls()
        assert m.lstm.input_size == 162
        # The last LSTM timestep is what gets concatenated.
        assert m.lstm.hidden_size == 128


@pytest.mark.parametrize("position", POSITIONS)
def test_forward_output_is_finite(position, seed_factory):
    seed_factory(31)
    env = Env("adp")
    env.reset()
    while env._acting_player_position != position:
        env.step(env.infoset.legal_actions[0])
    from douzero.env.env import get_obs

    obs = get_obs(env.infoset)
    z, x = _to_tensors(obs)
    torch.manual_seed(31)
    model = model_dict[position]()
    model.eval()
    with torch.no_grad():
        out = model(z, x, return_value=True)["values"]
    assert torch.isfinite(out).all()


def test_forward_with_single_legal_action(seed_factory):
    """A state with exactly one legal action must still forward correctly."""
    seed_factory(32)
    env = Env("adp")
    env.reset()
    from douzero.env.env import get_obs

    # Drive near terminal so choices shrink; just confirm shape with N=1 input.
    obs = get_obs(env.infoset)
    z_one = torch.from_numpy(obs["z_batch"][:1]).float()
    x_one = torch.from_numpy(obs["x_batch"][:1]).float()
    model = LandlordLstmModel()
    model.eval()
    with torch.no_grad():
        out = model(z_one, x_one, return_value=True)["values"]
    assert out.shape == (1, 1)


def test_forward_is_deterministic_under_eval_mode(seed_factory):
    """Same model + same input -> identical outputs (no dropout/batchnorm)."""
    seed_factory(33)
    env = Env("adp")
    env.reset()
    from douzero.env.env import get_obs

    obs = get_obs(env.infoset)
    z, x = _to_tensors(obs)
    torch.manual_seed(33)
    model = LandlordLstmModel()
    model.eval()
    with torch.no_grad():
        a = model(z, x, return_value=True)["values"]
        b = model(z, x, return_value=True)["values"]
    assert torch.allclose(a, b)


def test_fixed_init_output_hash_is_stable_across_runs(seed_factory):
    """Two models created with the same seed must produce byte-identical outputs."""
    from douzero.env.env import get_obs

    def run():
        seed_factory(44)
        env = Env("adp")
        env.reset()
        obs = get_obs(env.infoset)
        z, x = _to_tensors(obs)
        torch.manual_seed(44)
        model = LandlordLstmModel()
        model.eval()
        with torch.no_grad():
            vals = model(z, x, return_value=True)["values"]
        return hashlib.sha256(vals.numpy().tobytes()).hexdigest()

    assert run() == run()


def test_training_model_wrapper_holds_three_roles():
    model = Model(device="cpu")
    assert set(model.get_models().keys()) == set(POSITIONS)
    assert isinstance(model.get_model("landlord"), LandlordLstmModel)
    assert isinstance(model.get_model("landlord_up"), FarmerLstmModel)
    assert isinstance(model.get_model("landlord_down"), FarmerLstmModel)


def test_training_wrapper_forward_delegates_to_role_model(seed_factory):
    """Model.forward(position, ...) must delegate to the held role submodule.

    We compare the wrapper's own landlord submodule output against the wrapper's
    forward() output (same underlying object -> must be identical). A separate
    freshly-built model would differ because Model() initialises all three roles
    and consumes RNG state between them.
    """
    from douzero.env.env import get_obs

    seed_factory(55)
    env = Env("adp")
    env.reset()
    obs = get_obs(env.infoset)
    z, x = _to_tensors(obs)

    wrapper = Model(device="cpu")
    wrapper.eval()

    with torch.no_grad():
        via_wrapper = wrapper.forward("landlord", z, x, training=True)["values"]
        via_direct = wrapper.get_model("landlord")(z, x, return_value=True)["values"]
    assert torch.allclose(via_wrapper, via_direct)


def test_backward_pass_runs_without_error(seed_factory):
    """One forward + backward step must complete (gradient flow sanity)."""
    from douzero.env.env import get_obs

    seed_factory(66)
    env = Env("adp")
    env.reset()
    obs = get_obs(env.infoset)
    z, x = _to_tensors(obs)
    torch.manual_seed(66)
    model = LandlordLstmModel()
    out = model(z, x, return_value=True)["values"]
    loss = out.mean()
    loss.backward()
    # At least one parameter must have a non-None gradient.
    assert any(p.grad is not None for p in model.parameters())
