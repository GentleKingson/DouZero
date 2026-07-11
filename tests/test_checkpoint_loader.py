"""Tests for checkpoint save/load round-trips used by training and evaluation.

Two on-disk formats exist in the legacy code:

  1. ``{position}_weights_{frames}.ckpt`` -- a bare ``state_dict`` loaded by
     ``DeepAgent`` via ``_load_model`` (evaluation).
  2. ``model.tar`` -- a bundle with model/optimizer/stats/flags/frames/
     position_frames, used to resume training (``dmc.train``).

These tests build synthetic state_dicts from freshly-initialised models (no
downloaded weights) and verify both formats round-trip without silent key loss.
"""

from __future__ import annotations

import pytest
import torch

from douzero.dmc.models import LandlordLstmModel, Model, model_dict
from douzero.evaluation.deep_agent import _load_model


POSITIONS = ["landlord", "landlord_up", "landlord_down"]


@pytest.mark.parametrize("position", POSITIONS)
def test_eval_ckpt_roundtrip_is_elementwise_equal(position, seed_factory, tmp_path):
    """Saving a model's state_dict to .ckpt and reloading must reproduce params.

    Legacy ``_load_model`` filters pretrained keys to those present in a fresh
    model's state_dict. With a complete state_dict, no keys should be dropped.
    """
    seed_factory(900)
    torch.manual_seed(900)
    model = model_dict[position]()
    model.eval()
    ckpt = tmp_path / f"{position}.ckpt"
    torch.save(model.state_dict(), ckpt)

    loaded = _load_model(position, str(ckpt))
    loaded.eval()

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param {n1} differs after round-trip"


@pytest.mark.parametrize("position", POSITIONS)
def test_eval_ckpt_keys_are_not_silently_dropped(position, seed_factory, tmp_path):
    """No partial-load masquerade: every key in the saved dict must be consumed.

    ``_load_model`` keeps ``k in model_state_dict`` only. A correct role ckpt
    has exactly the same key set as a fresh role model, so the intersection
    must equal the full set.
    """
    seed_factory(901)
    torch.manual_seed(901)
    model = model_dict[position]()
    sd = model.state_dict()
    ckpt = tmp_path / f"{position}.ckpt"
    torch.save(sd, ckpt)

    fresh = model_dict[position]().state_dict()
    pretrained = torch.load(ckpt, map_location="cpu")
    consumed = {k for k in pretrained.keys() if k in fresh}
    assert consumed == set(pretrained.keys()), "some ckpt keys would be dropped"
    assert consumed == set(fresh.keys()), "key set mismatch between ckpt and model"


def test_model_tar_bundle_roundtrip(seed_factory, tmp_path):
    """The training bundle must save and restore model+optimizer+frames together.

    This mirrors ``dmc.train.checkpoint`` (dmc.py:191-209) without running the
    full async loop. We verify the bundle contains every expected top-level key
    and that the model state_dicts round-trip exactly.
    """
    seed_factory(902)
    torch.manual_seed(902)
    learner_model = Model(device="cpu")
    flags = {"xpid": "test", "frames": 0}
    optimizers = {p: torch.optim.RMSprop(learner_model.parameters(p)) for p in POSITIONS}
    stats = {"loss_landlord": 0.0}
    position_frames = {p: 0 for p in POSITIONS}
    frames = 1234

    models = learner_model.get_models()
    bundle_path = tmp_path / "model.tar"
    torch.save(
        {
            "model_state_dict": {k: models[k].state_dict() for k in models},
            "optimizer_state_dict": {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            "flags": flags,
            "frames": frames,
            "position_frames": position_frames,
        },
        bundle_path,
    )

    bundle = torch.load(bundle_path, map_location="cpu")
    assert set(bundle.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "stats",
        "flags",
        "frames",
        "position_frames",
    }
    assert bundle["frames"] == 1234
    assert set(bundle["model_state_dict"].keys()) == set(POSITIONS)

    # Reload into a fresh model and compare parameters.
    reloaded = Model(device="cpu")
    for position in POSITIONS:
        reloaded.get_model(position).load_state_dict(
            bundle["model_state_dict"][position]
        )
        for p1, p2 in zip(
            learner_model.get_model(position).parameters(),
            reloaded.get_model(position).parameters(),
        ):
            assert torch.equal(p1, p2), f"{position} params differ after bundle reload"


def test_landlord_and_farmer_dense1_shapes_differ():
    """Landlord and farmer models share key *names* but differ in dense1 shape.

    A naive cross-role ``load_state_dict`` must fail (shape mismatch). This
    documents that cross-role loading is unsafe today; P16 will promote this
    into a strict manifest check.
    """
    landlord = model_dict["landlord"]().state_dict()
    farmer = model_dict["landlord_up"]().state_dict()
    assert landlord["dense1.weight"].shape != farmer["dense1.weight"].shape

    fresh_farmer = model_dict["landlord_up"]()
    with pytest.raises(RuntimeError):
        fresh_farmer.load_state_dict(landlord)


def test_load_model_sets_eval_mode(seed_factory, tmp_path):
    """``_load_model`` must leave the model in eval mode (no stochastic fwd)."""
    seed_factory(903)
    torch.manual_seed(903)
    model = model_dict["landlord"]()
    ckpt = tmp_path / "landlord.ckpt"
    torch.save(model.state_dict(), ckpt)
    loaded = _load_model("landlord", str(ckpt))
    assert loaded.training is False
