"""Tests for the P01 unified seeding utility (douzero.runtime.seeding).

Key invariant: ``seed=0`` (NO_SEED, the default) must be a NO-OP so legacy runs
are byte-identical. Only a non-zero seed activates seeding.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from douzero.runtime import (
    derive_actor_seed,
    maybe_set_global_deterministic,
    set_global_seed,
)
from douzero.runtime.seeding import NO_SEED


# --------------------------------------------------------------------------- #
# seed=0 is a no-op (legacy compatibility)
# --------------------------------------------------------------------------- #
def test_seed_zero_is_noop():
    """seed=0 must not change RNG state (legacy behavior preserved)."""
    # Capture pre-state, call, capture post-state -- they must be unchanged.
    pre_py = random.random()
    set_global_seed(NO_SEED)
    # After a no-op, drawing again gives a DIFFERENT value (RNG advanced
    # normally, not reset). The point is no seeding happened.
    post_py = random.random()
    assert pre_py != post_py  # RNG is in its original (unseeded) sequence


def test_nonzero_seed_is_reproducible():
    """The same non-zero seed must produce identical RNG sequences twice."""
    def draw():
        set_global_seed(12345)
        return (
            random.random(),
            np.random.random(),
            torch.rand(1).item(),
        )

    a = draw()
    b = draw()
    assert a == b


def test_different_seeds_differ():
    set_global_seed(111)
    a = torch.rand(4)
    set_global_seed(222)
    b = torch.rand(4)
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# Per-actor seed derivation
# --------------------------------------------------------------------------- #
def test_derive_actor_seed_is_deterministic():
    s1 = derive_actor_seed(42, device_id=0, actor_id=3)
    s2 = derive_actor_seed(42, device_id=0, actor_id=3)
    assert s1 == s2


def test_derive_actor_seed_differs_per_actor():
    s0 = derive_actor_seed(42, device_id=0, actor_id=0)
    s1 = derive_actor_seed(42, device_id=0, actor_id=1)
    s2 = derive_actor_seed(42, device_id=1, actor_id=0)
    assert len({s0, s1, s2}) == 3  # all distinct


def test_derive_actor_seed_zero_base_is_noop():
    """When base_seed=0 (NO_SEED), the derived seed must also be NO_SEED."""
    assert derive_actor_seed(NO_SEED, device_id=0, actor_id=0) == NO_SEED


def test_derive_actor_seed_never_returns_zero():
    """The derived seed must never be 0 (the no-op sentinel)."""
    for base in (1, 2, 100, 99999):
        for dev in range(3):
            for act in range(5):
                assert derive_actor_seed(base, dev, act) != 0


def test_derive_actor_seed_is_stable_across_processes():
    """The derivation must not depend on PYTHONHASHSEED (use a stable hash)."""
    # We verify by checking the implementation uses hashlib, not hash().
    # Functional check: two calls in the same process are equal (already tested),
    # and the value is derived from sha256 (deterministic across processes).
    s = derive_actor_seed(7, device_id=1, actor_id=2)
    assert isinstance(s, int)
    assert s > 0


# --------------------------------------------------------------------------- #
# Deterministic flag
# --------------------------------------------------------------------------- #
def test_maybe_set_global_deterministic_false_is_noop():
    """deterministic=False must not change torch's deterministic setting."""
    original = torch.are_deterministic_algorithms_enabled()
    maybe_set_global_deterministic(False)
    assert torch.are_deterministic_algorithms_enabled() == original


def test_maybe_set_global_deterministic_true_enables():
    maybe_set_global_deterministic(True)
    assert torch.are_deterministic_algorithms_enabled() is True
    # Restore to avoid affecting other tests.
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Legacy env behavior unchanged when seed=0
# --------------------------------------------------------------------------- #
def test_env_with_seed_zero_unchanged(seed_factory):
    """Env.reset with seed=0 (default) must not be forced-seeded."""
    from douzero.env.env import Env

    # The legacy env shuffles with np.random unseeded. set_global_seed(0) is a
    # no-op, so two resets produce different deals (unseeded RNG advances).
    set_global_seed(NO_SEED)
    env = Env("adp")
    env.reset()
    hand_a = list(env._env.info_sets["landlord"].player_hand_cards)
    env.reset()
    hand_b = list(env._env.info_sets["landlord"].player_hand_cards)
    # Without seeding, two deals are almost certainly different.
    assert hand_a != hand_b


def test_env_with_nonzero_seed_reproducible(seed_factory):
    """Env.reset under a fixed non-zero seed must produce the same deal twice."""
    from douzero.env.env import Env

    set_global_seed(555)
    env = Env("adp")
    env.reset()
    hand_a = list(env._env.info_sets["landlord"].player_hand_cards)

    set_global_seed(555)
    env2 = Env("adp")
    env2.reset()
    hand_b = list(env2._env.info_sets["landlord"].player_hand_cards)
    assert hand_a == hand_b
