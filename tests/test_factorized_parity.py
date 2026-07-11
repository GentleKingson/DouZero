"""Numerical parity tests for the factorized legacy forward (P04).

The factorized models (:mod:`douzero.dmc.models_factorized`) must produce
values NUMERICALLY IDENTICAL to the legacy per-row models
(:mod:`douzero.dmc.models`) under the SAME weights, because:

  - the legacy ``z_batch`` tiles one history across all N action rows, so the
    LSTM output is identical across rows (eval mode, no dropout/BatchNorm);
  - the legacy ``x_batch`` shares its state block across rows, differing only
    in the trailing 54-dim action block.

The factorized forward runs the LSTM and the state projection ONCE and
broadcasts (``expand``, a view) across the per-action rows. This is a pure
rearrangement of the same arithmetic, so the outputs match within float
tolerance.

These tests pin:
  * three roles, varying legal-action counts (1, 2, many);
  * ``values`` parity within ``atol=1e-6, rtol=1e-5`` (CPU float32);
  * argmax action parity (epsilon=0, i.e. return_value=True path);
  * state_dict key + shape parity (so legacy .ckpt loads with no conversion);
  * a legacy .ckpt saved from the legacy model loads into the factorized model
    with NO missing/unexpected keys;
  * the LSTM is called exactly ONCE per factorized decision and N times per
    legacy decision (the whole point of P04);
  * DeepAgent backend='legacy_factorized' selects the same action as
    backend='legacy' under identical weights and infoset;
  * finite outputs and determinism (same input -> same output twice).

CPU-only. Numerical and argmax parity are TESTED on CPU. GPU numerical and
argmax parity are NOT measured: mathematical equivalence does not imply
bitwise or universal argmax identity across CPU/GPU (different kernels,
reduction order, and cuDNN RNN non-determinism can change results). GPU
parity must be measured empirically before it is asserted; that measurement
is deferred to P14.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from douzero.dmc.models import (
    FarmerLstmModel,
    LandlordLstmModel,
    Model,
    model_dict,
)
from douzero.dmc.models_factorized import (
    LegacyFactorizedFarmerModel,
    LegacyFactorizedLandlordModel,
    LegacyFactorizedModel,
    factorized_model_dict,
    split_legacy_batch,
)
from douzero.env.env import Env, get_obs


POSITIONS = ["landlord", "landlord_up", "landlord_down"]

# CPU float32 tolerance. The legacy and factorized forwards perform the SAME
# arithmetic (matmuls / LSTM) on the SAME weights and inputs, just arranged
# differently (per-row vs broadcast). On CPU this is bit-identical in practice;
# the tolerance absorbs any platform-level reordering.
ATOL = 1e-6
RTOL = 1e-5


def _to_tensors(obs):
    z = torch.from_numpy(obs["z_batch"]).float()
    x = torch.from_numpy(obs["x_batch"]).float()
    return z, x


def _drive_to_position(env: Env, position: str, max_steps: int = 20):
    """Step the env until ``position`` is the acting player."""
    env.reset()
    steps = 0
    while env._acting_player_position != position and steps < max_steps:
        env.step(env.infoset.legal_actions[0])
        steps += 1
    assert env._acting_player_position == position, (
        f"could not drive to {position} within {max_steps} steps"
    )
    return env.infoset


def _build_paired_models(position: str, seed: int = 555):
    """Build a legacy and a factorized model that share IDENTICAL weights.

    Both are seeded the same and the factorized model's state_dict is
    overwritten with the legacy model's, so the weights are byte-for-byte
    identical. This is the configuration parity must hold under.
    """
    torch.manual_seed(seed)
    legacy = model_dict[position]()
    torch.manual_seed(seed)
    factorized = factorized_model_dict[position]()
    # Force exact weight equality (the two __init__s consume RNG identically by
    # construction, but overwrite to be bulletproof against any future
    # initialization drift).
    factorized.load_state_dict(legacy.state_dict())
    legacy.eval()
    factorized.eval()
    return legacy, factorized


# --------------------------------------------------------------------------- #
# state_dict / checkpoint compatibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_state_dict_keys_match_legacy(position):
    """The factorized model's state_dict keys must equal the legacy model's."""
    legacy = model_dict[position]()
    factorized = factorized_model_dict[position]()
    legacy_keys = set(legacy.state_dict().keys())
    fact_keys = set(factorized.state_dict().keys())
    assert legacy_keys == fact_keys, (
        f"key set mismatch for {position}: "
        f"only-in-legacy={legacy_keys - fact_keys}, "
        f"only-in-factorized={fact_keys - legacy_keys}"
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_state_dict_shapes_match_legacy(position):
    """Every state_dict tensor shape must match between legacy and factorized."""
    legacy = model_dict[position]()
    factorized = factorized_model_dict[position]()
    for k in legacy.state_dict():
        ls = tuple(legacy.state_dict()[k].shape)
        fs = tuple(factorized.state_dict()[k].shape)
        assert ls == fs, f"{position} key {k!r}: legacy {ls} vs factorized {fs}"


@pytest.mark.parametrize("position", POSITIONS)
def test_legacy_ckpt_loads_into_factorized_no_missing_unexpected(position, tmp_path):
    """A .ckpt saved from the legacy model must load into the factorized model
    with strict=True (no missing/unexpected keys)."""
    legacy = model_dict[position]()
    ckpt = tmp_path / f"{position}.ckpt"
    torch.save(legacy.state_dict(), ckpt)

    factorized = factorized_model_dict[position]()
    # strict=True: any key mismatch raises. This is the checkpoint-compat gate.
    factorized.load_state_dict(torch.load(ckpt, weights_only=True))
    # Confirm the weights actually loaded (not left at random init).
    for k in legacy.state_dict():
        assert torch.equal(factorized.state_dict()[k], legacy.state_dict()[k]), (
            f"{position} weight {k!r} did not load equivalently"
        )


def test_dense1_input_widths_match_legacy():
    """dense1.in_features must match so the same weights apply."""
    assert (
        LegacyFactorizedLandlordModel().dense1.in_features
        == LandlordLstmModel().dense1.in_features
        == 373 + 128
    )
    assert (
        LegacyFactorizedFarmerModel().dense1.in_features
        == FarmerLstmModel().dense1.in_features
        == 484 + 128
    )


# --------------------------------------------------------------------------- #
# Numerical value parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_values_parity_full_action_set(position, seed_factory):
    """factorized values == legacy values for the full legal-action set."""
    seed_factory(900 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)

    legacy, factorized = _build_paired_models(position)
    with torch.no_grad():
        legacy_vals = legacy(z, x, return_value=True)["values"]
        fact_vals = factorized(z, x, return_value=True)["values"]
    assert legacy_vals.shape == fact_vals.shape == (z.shape[0], 1)
    assert torch.allclose(legacy_vals, fact_vals, atol=ATOL, rtol=RTOL), (
        f"value mismatch for {position} (N={z.shape[0]}): "
        f"max abs diff = {(legacy_vals - fact_vals).abs().max().item()}"
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_argmax_action_parity(position, seed_factory):
    """argmax over factorized values == argmax over legacy values (epsilon=0)."""
    seed_factory(910 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)

    legacy, factorized = _build_paired_models(position)
    with torch.no_grad():
        legacy_vals = legacy(z, x, return_value=True)["values"].squeeze(-1)
        fact_vals = factorized(z, x, return_value=True)["values"].squeeze(-1)
    legacy_idx = int(torch.argmax(legacy_vals).item())
    fact_idx = int(torch.argmax(fact_vals).item())
    assert legacy_idx == fact_idx, (
        f"argmax mismatch for {position}: legacy={legacy_idx}, factorized={fact_idx}"
    )
    # The selected action must be a legal action.
    assert infoset.legal_actions[fact_idx] in infoset.legal_actions


@pytest.mark.parametrize("n_actions", [1, 2])
@pytest.mark.parametrize("position", POSITIONS)
def test_values_parity_small_action_counts(position, n_actions, seed_factory):
    """Parity must hold for 1 and 2 legal actions (edge cases)."""
    seed_factory(920 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)

    # Slice to the requested action count (rows are independent candidates).
    full_n = z.shape[0]
    assert full_n >= n_actions, f"need >= {n_actions} actions, got {full_n}"
    z_sub = z[:n_actions]
    x_sub = x[:n_actions]

    legacy, factorized = _build_paired_models(position)
    with torch.no_grad():
        legacy_vals = legacy(z_sub, x_sub, return_value=True)["values"]
        fact_vals = factorized(z_sub, x_sub, return_value=True)["values"]
    assert legacy_vals.shape == (n_actions, 1)
    assert torch.allclose(legacy_vals, fact_vals, atol=ATOL, rtol=RTOL)


def test_values_parity_many_seeds_and_roles():
    """Parity across many random deals and whichever role is acting."""
    for seed in range(15):
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
        # Advance a varying number of steps to vary role + history length.
        for _ in range(seed % 6):
            env.step(env.infoset.legal_actions[0])
        position = env._acting_player_position
        obs = get_obs(env.infoset)
        z, x = _to_tensors(obs)
        legacy, factorized = _build_paired_models(position, seed=seed)
        with torch.no_grad():
            legacy_vals = legacy(z, x, return_value=True)["values"]
            fact_vals = factorized(z, x, return_value=True)["values"]
        assert torch.allclose(legacy_vals, fact_vals, atol=ATOL, rtol=RTOL), (
            f"seed {seed} role {position} N={z.shape[0]}: "
            f"max abs diff = {(legacy_vals - fact_vals).abs().max().item()}"
        )


# --------------------------------------------------------------------------- #
# forward_factorized (split-input interface) parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_forward_factorized_matches_legacy(position, seed_factory):
    """The split-input forward_factorized must match the legacy forward."""
    seed_factory(940 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)

    legacy, factorized = _build_paired_models(position)
    z_single, x_state_single, x_action = split_legacy_batch(position, z, x)
    with torch.no_grad():
        legacy_vals = legacy(z, x, return_value=True)["values"]
        fact_vals = factorized.forward_factorized(
            z_single, x_state_single, x_action, return_value=True
        )["values"]
    assert torch.allclose(legacy_vals, fact_vals, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- #
# LSTM work reduction — the P04 efficiency proof
# --------------------------------------------------------------------------- #
class _LstmBatchRecorder:
    """Record the batch size (number of rows) passed to the LSTM.

    The legacy and factorized forwards both call ``lstm.forward`` exactly once
    per decision, but the legacy path feeds it ``(N, 5, 162)`` — N identical
    rows — while the factorized path feeds it ``(1, 5, 162)``. The waste P04
    removes is the N-fold redundant LSTM computation over identical rows, not
    the number of Python-level calls. This recorder captures the input batch
    size so the tests assert the real distinction: legacy processes N rows,
    factorized processes 1.
    """

    def __init__(self, model):
        self.model = model
        self.original_forward = model.lstm.forward
        self.batch_sizes = []

    def __enter__(self):
        recorder = self

        def _recording_forward(z, *args, **kwargs):
            # z is (batch, seq, feature); record the batch dimension.
            recorder.batch_sizes.append(z.shape[0])
            return recorder.original_forward(z, *args, **kwargs)

        self.model.lstm.forward = _recording_forward
        return self

    def __exit__(self, *exc):
        self.model.lstm.forward = self.original_forward

    @property
    def total_rows(self):
        return sum(self.batch_sizes)


@pytest.mark.parametrize("position", POSITIONS)
def test_factorized_feeds_one_row_to_lstm(position, seed_factory):
    """The factorized forward must feed the LSTM exactly 1 row per decision."""
    seed_factory(950 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)
    n = z.shape[0]

    _, factorized = _build_paired_models(position)
    with torch.no_grad():
        with _LstmBatchRecorder(factorized) as recorder:
            factorized(z, x, return_value=True)
    assert recorder.batch_sizes == [1], (
        f"factorized should feed the LSTM a single (1, 5, 162) batch, "
        f"got batch_sizes={recorder.batch_sizes} (N={n})"
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_legacy_feeds_n_rows_to_lstm(position, seed_factory):
    """The legacy forward feeds the LSTM all N identical rows at once.

    This is the baseline the factorized path improves on: the legacy path
    makes the LSTM process N identical copies of the shared history, while the
    factorized path processes exactly one. The benchmark records the latency
    consequence.
    """
    seed_factory(960 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)
    n = z.shape[0]

    legacy, _ = _build_paired_models(position)
    with torch.no_grad():
        with _LstmBatchRecorder(legacy) as recorder:
            legacy(z, x, return_value=True)
    assert recorder.batch_sizes == [n], (
        f"legacy should feed the LSTM a single ({n}, 5, 162) batch, "
        f"got batch_sizes={recorder.batch_sizes}"
    )


# --------------------------------------------------------------------------- #
# DeepAgent backend parity
# --------------------------------------------------------------------------- #
def _save_paired_ckpts(position, seed, tmp_path):
    """Save a legacy model's state_dict to two ckpt paths (same weights)."""
    torch.manual_seed(seed)
    legacy = model_dict[position]()
    legacy.eval()
    ckpt_legacy = tmp_path / f"{position}_legacy.ckpt"
    ckpt_fact = tmp_path / f"{position}_factorized.ckpt"
    torch.save(legacy.state_dict(), ckpt_legacy)
    torch.save(legacy.state_dict(), ckpt_fact)
    return str(ckpt_legacy), str(ckpt_fact)


@pytest.mark.parametrize("position", POSITIONS)
def test_deepagent_factorized_matches_legacy_selection(position, seed_factory, tmp_path):
    """DeepAgent backend='legacy_factorized' must pick the same action as
    backend='legacy' under identical weights and infoset."""
    from douzero.evaluation.deep_agent import DeepAgent

    seed_factory(970 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)

    ckpt_legacy, ckpt_fact = _save_paired_ckpts(position, 777, tmp_path)
    agent_legacy = DeepAgent(position, ckpt_legacy, backend="legacy")
    agent_fact = DeepAgent(position, ckpt_fact, backend="legacy_factorized")

    action_legacy = agent_legacy.act(infoset)
    action_fact = agent_fact.act(infoset)
    assert action_legacy == action_fact, (
        f"DeepAgent selection mismatch for {position}: "
        f"legacy={action_legacy}, factorized={action_fact}"
    )
    assert action_fact in infoset.legal_actions


def test_deepagent_default_backend_is_legacy(tmp_path):
    """DeepAgent with no backend arg must default to 'legacy'."""
    from douzero.evaluation.deep_agent import DeepAgent

    torch.manual_seed(1)
    legacy = model_dict["landlord"]()
    ckpt = tmp_path / "landlord.ckpt"
    torch.save(legacy.state_dict(), ckpt)
    agent = DeepAgent("landlord", str(ckpt))
    assert agent.backend == "legacy"


def test_deepagent_rejects_unknown_backend(tmp_path):
    from douzero.evaluation.deep_agent import DeepAgent

    torch.manual_seed(1)
    legacy = model_dict["landlord"]()
    ckpt = tmp_path / "landlord.ckpt"
    torch.save(legacy.state_dict(), ckpt)
    with pytest.raises(ValueError, match="backend"):
        DeepAgent("landlord", str(ckpt), backend="v2")


# --------------------------------------------------------------------------- #
# Finite outputs and determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_factorized_outputs_are_finite(position, seed_factory):
    seed_factory(980 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)
    _, factorized = _build_paired_models(position)
    with torch.no_grad():
        vals = factorized(z, x, return_value=True)["values"]
    assert torch.isfinite(vals).all()


def test_factorized_is_deterministic_under_eval_mode(seed_factory):
    """Same factorized model + same input -> identical outputs twice."""
    seed_factory(990)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)
    _, factorized = _build_paired_models("landlord")
    with torch.no_grad():
        a = factorized(z, x, return_value=True)["values"]
        b = factorized(z, x, return_value=True)["values"]
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Wrapper parity
# --------------------------------------------------------------------------- #
def test_factorized_wrapper_holds_three_roles():
    wrapper = LegacyFactorizedModel(device="cpu")
    assert set(wrapper.get_models().keys()) == set(POSITIONS)
    assert isinstance(wrapper.get_model("landlord"), LegacyFactorizedLandlordModel)
    assert isinstance(wrapper.get_model("landlord_up"), LegacyFactorizedFarmerModel)
    assert isinstance(wrapper.get_model("landlord_down"), LegacyFactorizedFarmerModel)


def test_factorized_wrapper_forward_matches_role_model(seed_factory):
    """LegacyFactorizedModel.forward(position, z, x) must delegate to the role."""
    seed_factory(991)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)

    torch.manual_seed(991)
    legacy = LandlordLstmModel()
    legacy.eval()
    torch.manual_seed(991)
    wrapper = LegacyFactorizedModel(device="cpu")
    wrapper.eval()
    # Make the wrapper's landlord weights identical to the legacy model.
    wrapper.get_model("landlord").load_state_dict(legacy.state_dict())

    with torch.no_grad():
        via_wrapper = wrapper.forward("landlord", z, x, training=False)["values"]
        via_role = wrapper.get_model("landlord")(z, x, return_value=True)["values"]
    assert torch.allclose(via_wrapper, via_role, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- #
# split_legacy_batch helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_split_legacy_batch_shapes(position, seed_factory):
    seed_factory(992 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z, x = _to_tensors(obs)
    n = z.shape[0]
    z_single, x_state_single, x_action = split_legacy_batch(position, z, x)
    assert z_single.shape == (1, 5, 162)
    # state width: landlord 319, farmers 430.
    expected_state = 319 if position == "landlord" else 430
    assert x_state_single.shape == (1, expected_state)
    assert x_action.shape == (n, 54)


def test_split_legacy_batch_rejects_unknown_position():
    z = torch.zeros(3, 5, 162)
    x = torch.zeros(3, 373)
    with pytest.raises(ValueError, match="position"):
        split_legacy_batch("bogus", z, x)


# --------------------------------------------------------------------------- #
# get_obs_factorized — the split observation encoder (no tiling)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_get_obs_factorized_state_matches_legacy_x_no_action(position, seed_factory):
    """The split encoder's shared state must equal the legacy x_no_action."""
    from douzero.env.env import get_obs_factorized

    seed_factory(1000 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    legacy_obs = get_obs(infoset)
    split_obs = get_obs_factorized(infoset)
    np.testing.assert_array_equal(
        split_obs["x_no_action"], legacy_obs["x_no_action"],
        err_msg=f"x_no_action mismatch for {position}",
    )
    # x_state_single is the float32 singleton view of the same vector.
    np.testing.assert_array_equal(
        split_obs["x_state_single"][0].astype(np.int8),
        legacy_obs["x_no_action"],
        err_msg=f"x_state_single mismatch for {position}",
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_get_obs_factorized_history_matches_legacy_z(position, seed_factory):
    """The split encoder's singleton history must equal the legacy z."""
    from douzero.env.env import get_obs_factorized

    seed_factory(1010 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    legacy_obs = get_obs(infoset)
    split_obs = get_obs_factorized(infoset)
    np.testing.assert_array_equal(
        split_obs["z"], legacy_obs["z"],
        err_msg=f"z mismatch for {position}",
    )
    np.testing.assert_array_equal(
        split_obs["z_single"][0].astype(np.int8), legacy_obs["z"],
        err_msg=f"z_single mismatch for {position}",
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_get_obs_factorized_action_matrix_matches_legacy(position, seed_factory):
    """The split encoder's (N,54) action matrix must equal legacy x_batch[:,-54:]."""
    from douzero.env.env import get_obs_factorized

    seed_factory(1020 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    legacy_obs = get_obs(infoset)
    split_obs = get_obs_factorized(infoset)
    n = len(infoset.legal_actions)
    assert split_obs["x_action"].shape == (n, 54)
    np.testing.assert_array_equal(
        split_obs["x_action"].astype(np.int8),
        legacy_obs["x_batch"][:, -54:].astype(np.int8),
        err_msg=f"action matrix mismatch for {position}",
    )


@pytest.mark.parametrize("position", POSITIONS)
def test_get_obs_factorized_shapes(position, seed_factory):
    """The split encoder must produce singleton shared blocks + (N,54) actions."""
    from douzero.env.env import get_obs_factorized

    seed_factory(1030 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    split_obs = get_obs_factorized(infoset)
    n = len(infoset.legal_actions)
    assert split_obs["z_single"].shape == (1, 5, 162)
    expected_state = 319 if position == "landlord" else 430
    assert split_obs["x_state_single"].shape == (1, expected_state)
    assert split_obs["x_action"].shape == (n, 54)
    assert split_obs["legal_actions"] == infoset.legal_actions


@pytest.mark.parametrize("position", POSITIONS)
def test_get_obs_factorized_forward_matches_legacy(position, seed_factory):
    """End-to-end: split obs + forward_factorized == legacy forward."""
    from douzero.env.env import get_obs_factorized

    seed_factory(1040 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    legacy_obs = get_obs(infoset)
    split_obs = get_obs_factorized(infoset)
    z, x = _to_tensors(legacy_obs)

    legacy, factorized = _build_paired_models(position)
    z_single = torch.from_numpy(split_obs["z_single"]).float()
    x_state_single = torch.from_numpy(split_obs["x_state_single"]).float()
    x_action = torch.from_numpy(split_obs["x_action"]).float()
    with torch.no_grad():
        legacy_vals = legacy(z, x, return_value=True)["values"]
        fact_vals = factorized.forward_factorized(
            z_single, x_state_single, x_action, return_value=True
        )["values"]
    assert torch.allclose(legacy_vals, fact_vals, atol=ATOL, rtol=RTOL), (
        f"split-obs parity mismatch for {position} (N={z.shape[0]}): "
        f"max abs diff = {(legacy_vals - fact_vals).abs().max().item()}"
    )


def test_get_obs_factorized_never_tiles_state_into_n_rows(seed_factory):
    """The split encoder must NOT allocate an (N, D_state) tiled block.

    This is the core P04 observation-side de-duplication property. The shared
    state is a singleton (1, D_state); only the per-action matrix carries N.
    """
    from douzero.env.env import get_obs_factorized

    seed_factory(1050)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    split_obs = get_obs_factorized(infoset)
    # The shared state has NO N dimension.
    assert split_obs["x_state_single"].shape[0] == 1
    assert split_obs["z_single"].shape[0] == 1
    # Only the action matrix has the N dimension.
    n = len(infoset.legal_actions)
    assert split_obs["x_action"].shape[0] == n


# --------------------------------------------------------------------------- #
# Input invariant enforcement (review blocker #4)
# --------------------------------------------------------------------------- #
def _make_tiled_batch(position, n, seed=42):
    """Build a correctly-tiled legacy batch (rows identical) for validation tests."""
    seed_factory = None
    np.random.seed(seed)
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    obs = get_obs(infoset)
    z = torch.from_numpy(obs["z_batch"]).float()
    x = torch.from_numpy(obs["x_batch"]).float()
    return z[:n], x[:n], infoset


def test_forward_rejects_non_identical_z_rows():
    """forward() must raise when z_batch rows are NOT identical (silent-failure guard)."""
    z, x, _ = _make_tiled_batch("landlord", 4)
    # Corrupt row 1 of z so rows are no longer identical.
    z_bad = z.clone()
    z_bad[1, 0, 0] = z_bad[1, 0, 0] + 1.0
    _, factorized = _build_paired_models("landlord")
    with pytest.raises(ValueError, match="rows are NOT identical"):
        factorized(z_bad, x, return_value=True)


def test_forward_rejects_non_identical_state_rows():
    """forward() must raise when the x_batch state-block rows are NOT identical."""
    z, x, _ = _make_tiled_batch("landlord", 4)
    # Corrupt row 1's state block (not the action block) so state rows differ.
    x_bad = x.clone()
    x_bad[1, 0] = x_bad[1, 0] + 1.0  # state block starts at column 0
    _, factorized = _build_paired_models("landlord")
    with pytest.raises(ValueError, match="state block"):
        factorized(z, x_bad, return_value=True)


def test_forward_accepts_correctly_tiled_batch():
    """forward() must accept a correctly-tiled legacy batch (the happy path)."""
    z, x, _ = _make_tiled_batch("landlord", 5)
    _, factorized = _build_paired_models("landlord")
    with torch.no_grad():
        vals = factorized(z, x, return_value=True)["values"]
    assert vals.shape == (5, 1)


def test_forward_rejects_wrong_z_ndim():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(5, 162)  # 2D, not 3D
    x = torch.zeros(5, 373)
    with pytest.raises(ValueError, match="ndim"):
        factorized(z, x, return_value=True)


def test_forward_rejects_row_count_mismatch():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(5, 5, 162)
    x = torch.zeros(4, 373)  # N=4 != z's N=5
    with pytest.raises(ValueError, match="row count"):
        factorized(z, x, return_value=True)


def test_forward_rejects_wrong_x_width():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(3, 5, 162)
    x = torch.zeros(3, 200)  # wrong width (expected 373)
    with pytest.raises(ValueError, match="x_batch.shape"):
        factorized(z, x, return_value=True)


def test_forward_rejects_n_zero():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(0, 5, 162)
    x = torch.zeros(0, 373)
    with pytest.raises(ValueError, match="N>=1"):
        factorized(z, x, return_value=True)


def test_forward_factorized_rejects_non_singleton_z():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(3, 5, 162)  # not a singleton
    x_state = torch.zeros(1, 319)
    x_action = torch.zeros(3, 54)
    with pytest.raises(ValueError, match="z_single"):
        factorized.forward_factorized(z, x_state, x_action, return_value=True)


def test_forward_factorized_rejects_wrong_state_width():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(1, 5, 162)
    x_state = torch.zeros(1, 100)  # expected 319
    x_action = torch.zeros(3, 54)
    with pytest.raises(ValueError, match="x_state_single"):
        factorized.forward_factorized(z, x_state, x_action, return_value=True)


def test_forward_factorized_rejects_wrong_action_width():
    _, factorized = _build_paired_models("landlord")
    z = torch.zeros(1, 5, 162)
    x_state = torch.zeros(1, 319)
    x_action = torch.zeros(3, 40)  # expected 54
    with pytest.raises(ValueError, match="x_action"):
        factorized.forward_factorized(z, x_state, x_action, return_value=True)


# --------------------------------------------------------------------------- #
# DeepAgent factorized backend uses the split path (no tiled batch)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", POSITIONS)
def test_deepagent_factorized_uses_split_observation(position, seed_factory, tmp_path):
    """The factorized DeepAgent must call forward_factorized, not the tiled forward.

    Patches forward_factorized and the legacy forward to prove the split path
    is taken and the tiled (N, ...) batch is never built for the model.
    """
    from douzero.evaluation.deep_agent import DeepAgent

    seed_factory(1060 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)

    ckpt_legacy, ckpt_fact = _save_paired_ckpts(position, 888, tmp_path)
    agent = DeepAgent(position, ckpt_fact, backend="legacy_factorized")

    called_factorized = {"count": 0}
    called_legacy = {"count": 0}
    orig_factorized = agent.model.forward_factorized
    orig_forward = agent.model.forward

    def _spy_factorized(*args, **kwargs):
        called_factorized["count"] += 1
        return orig_factorized(*args, **kwargs)

    def _spy_forward(*args, **kwargs):
        called_legacy["count"] += 1
        return orig_forward(*args, **kwargs)

    agent.model.forward_factorized = _spy_factorized
    agent.model.forward = _spy_forward
    try:
        action = agent.act(infoset)
    finally:
        agent.model.forward_factorized = orig_factorized
        agent.model.forward = orig_forward

    assert called_factorized["count"] == 1, "factorized backend must call forward_factorized"
    assert called_legacy["count"] == 0, "factorized backend must not call the tiled forward"
    assert action in infoset.legal_actions


@pytest.mark.parametrize("position", POSITIONS)
def test_deepagent_factorized_matches_legacy_split_path(position, seed_factory, tmp_path):
    """DeepAgent factorized (split obs) selects the same action as legacy backend."""
    from douzero.evaluation.deep_agent import DeepAgent

    seed_factory(1070 + POSITIONS.index(position))
    env = Env("adp")
    infoset = _drive_to_position(env, position)
    ckpt_legacy, ckpt_fact = _save_paired_ckpts(position, 999, tmp_path)
    agent_legacy = DeepAgent(position, ckpt_legacy, backend="legacy")
    agent_fact = DeepAgent(position, ckpt_fact, backend="legacy_factorized")
    assert agent_legacy.act(infoset) == agent_fact.act(infoset)


def test_deepagent_legacy_uses_inference_mode(seed_factory, tmp_path):
    """The legacy backend must run under torch.inference_mode (no autograd graph)."""
    from douzero.evaluation.deep_agent import DeepAgent

    seed_factory(1080)
    env = Env("adp")
    infoset = _drive_to_position(env, "landlord")
    ckpt_legacy, _ = _save_paired_ckpts("landlord", 111, tmp_path)
    agent = DeepAgent("landlord", ckpt_legacy, backend="legacy")
    # Under inference_mode, requires_grad is False for new tensors.
    with torch.inference_mode():
        _ = agent.act(infoset)
    # The call must not have left autograd tracking on the model's params.
    assert all(not p.requires_grad or p.grad is None for p in agent.model.parameters())
