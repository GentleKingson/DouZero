"""Unified seeding for reproducible training and inference (P01).

The legacy codebase performs NO seeding anywhere (the only RNG is
``np.random.shuffle`` in ``Env.reset``). P01 adds an opt-in, centralized
seeding utility:

  - ``set_global_seed(seed)``: seeds Python ``random``, ``numpy``, and
    ``torch`` (CPU) in one call.
  - ``derive_actor_seed(base_seed, device_id, actor_id)``: deterministic
    per-actor seed so each actor process is independently reproducible.
  - ``maybe_set_global_deterministic(flag)``: enables
    ``torch.use_deterministic_algorithms`` only when explicitly requested.

Legacy behavior is preserved by default: when ``--seed 0`` (the default), no
seeding is applied at all (the function is a no-op), so existing runs are
byte-identical to pre-P01. Only when the user passes a non-zero ``--seed`` does
seeding activate. ``--deterministic`` only takes effect when explicitly set.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "derive_actor_seed",
    "maybe_set_global_deterministic",
    "set_global_seed",
]

#: The sentinel "no seeding" value. When --seed equals this (the default 0),
#: set_global_seed is a no-op so legacy behavior is byte-identical.
NO_SEED = 0


def set_global_seed(seed: int) -> None:
    """Seed Python random, numpy, and torch (CPU).

    If ``seed == NO_SEED`` (0, the legacy default), this is a **no-op**: nothing
    is seeded and the run is byte-identical to pre-P01 behavior. This is the key
    backward-compatibility invariant -- seeding is strictly opt-in.
    """
    if seed == NO_SEED:
        return  # legacy: no seeding
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA seeding is applied unconditionally when the seed is non-zero and CUDA
    # is available; it is a no-op on CPU-only builds.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_actor_seed(base_seed: int, device_id: int, actor_id: int) -> int:
    """Derive a deterministic per-actor seed from (base_seed, device, actor).

    Each actor process gets a unique but reproducible seed, so a fixed base_seed
    yields the same set of deals and rollouts across runs. The derivation uses
    a stable hash (not Python's randomized hash) so it is reproducible across
    process restarts (important for spawn-based multiprocessing).

    Returns ``NO_SEED`` (0) when ``base_seed == NO_SEED``, so unseeded actors
    keep legacy behavior.
    """
    if base_seed == NO_SEED:
        return NO_SEED
    # Use a stable SHA-based derivation (Python's built-in hash is randomized
    # per process by PYTHONHASHSEED, so it cannot be used for reproducibility).
    token = f"douzero-actor|{base_seed}|{device_id}|{actor_id}".encode()
    digest = hashlib.sha256(token).digest()
    # Take the first 8 bytes as a positive 63-bit int (avoid sign issues).
    derived = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return derived if derived != 0 else 1  # never return 0 (the no-seed sentinel)


def maybe_set_global_deterministic(deterministic: bool) -> None:
    """Enable torch deterministic algorithms only when explicitly requested.

    Legacy default is ``deterministic=False`` (off), which leaves
    ``torch.use_deterministic_algorithms`` at its default (False). This is a
    no-op unless the user passes ``--deterministic``.
    """
    if not deterministic:
        return
    import torch

    torch.use_deterministic_algorithms(True)
    # Cudnn deterministic mode (no-op on CPU).
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
