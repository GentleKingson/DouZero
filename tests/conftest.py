"""Shared pytest fixtures and seeding helpers for the DouZero baseline (P00).

The legacy codebase performs NO internal seeding: the only source of randomness
during a normal episode is ``np.random.shuffle`` in ``Env.reset``. These helpers
make tests deterministic without modifying any production module.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

# Force CPU for every test. The legacy models have no ``device`` argument and
# probe ``torch.cuda.is_available()`` directly; setting the env var before torch
# is imported would be ideal, but in practice CUDA is not present in the test
# image, so this is belt-and-braces.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DEFAULT_SEED = 1234


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG the legacy code path can reach.

    The legacy ``DeepAgent.act`` always calls the model with ``return_value=True``
    so it never touches ``torch.randint`` / ``np.random.rand`` exploration.
    Still, we seed all three for completeness and so fresh model initialisation
    is reproducible across runs.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # PyTorch CPU determinism: legacy models contain no conv/dropout/batchnorm,
    # so this is cheap and harmless.
    torch.use_deterministic_algorithms(False)


@pytest.fixture(autouse=True)
def _seed_each_test():
    """Reset RNG state before every test for full independence."""
    set_seed(DEFAULT_SEED)
    yield


@pytest.fixture
def seed_factory():
    """Return a callable that reseeds all RNGs with a chosen seed."""
    return set_seed


def _build_card_play_data(deck_order):
    """Slice a full 54-card deck in the exact layout Env.reset uses.

    Mirrors ``douzero/env/env.py`` reset: landlord gets 20, up gets 17, down
    gets 17, bottom cards are deck[17:20]. Each list is sorted ascending.
    """
    deck_list = list(deck_order)
    assert len(deck_list) == 54
    data = {
        "landlord": sorted(deck_list[:20]),
        "landlord_up": sorted(deck_list[20:37]),
        "landlord_down": sorted(deck_list[37:54]),
        "three_landlord_cards": sorted(deck_list[17:20]),
    }
    return data


@pytest.fixture
def fixed_card_play_data():
    """A fixed, hand-verifiable deal used across snapshot/rollout tests.

    This deal is independent of any RNG: it is constructed directly from a
    deterministic deck so the legal-action snapshot is stable forever and does
    not depend on numpy's shuffle implementation or version.
    """
    # Canonical deck: 3..14 (x4), 17 (x4), 20, 30.
    deck_order = []
    for rank in range(3, 15):
        deck_order.extend([rank] * 4)
    deck_order.extend([17] * 4)
    deck_order.extend([20, 30])
    return _build_card_play_data(deck_order)


@pytest.fixture
def seeded_env(seed_factory):
    """An ``Env`` whose reset produced a known, seeded deal."""
    from douzero.env.env import Env

    seed_factory(20240501)
    env = Env("adp")
    env.reset()
    return env


def _instantiate_model_with_fixed_init(position: str, seed: int = DEFAULT_SEED):
    """Build a role model with deterministic (seeded) random initialisation.

    Returns the underlying ``nn.Module`` (``LandlordLstmModel`` or
    ``FarmerLstmModel``) in eval mode on CPU. No checkpoint is loaded.
    """
    from douzero.dmc.models import model_dict

    torch.manual_seed(seed)
    model = model_dict[position]()
    model.eval()
    return model


@pytest.fixture
def deepagent_with_init_weights(tmp_path):
    """Return a factory building a ``DeepAgent`` backed by a *synthetic* ckpt.

    The legacy ``DeepAgent`` only loads weights from a ``.ckpt`` path. To test it
    offline without downloading pretrained weights, we save a freshly-seeded
    ``state_dict`` to a temp ``.ckpt`` and return a builder for each position.
    """
    from douzero.dmc.models import model_dict
    from douzero.evaluation.deep_agent import DeepAgent

    def build(position: str, seed: int = DEFAULT_SEED):
        torch.manual_seed(seed)
        model = model_dict[position]()
        ckpt = tmp_path / f"{position}.ckpt"
        torch.save(model.state_dict(), ckpt)
        return DeepAgent(position, str(ckpt))

    return build


@pytest.fixture
def fresh_model():
    """Factory for deterministic fresh role models (no checkpoint)."""
    return _instantiate_model_with_fixed_init
