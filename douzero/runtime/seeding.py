"""Unified seeding for reproducible training and inference (P01).

The legacy codebase performs NO seeding anywhere (the only RNG is
``np.random.shuffle`` in ``Env.reset``). P01 adds an opt-in, centralized
seeding utility:

  - ``set_global_seed(seed)``: seeds Python ``random``, ``numpy``, and
    ``torch`` (CPU) in one call.
  - ``derive_actor_seed(base_seed, device_token, actor_id)``: deterministic
    per-actor seed so each actor process is independently reproducible.
  - ``maybe_set_global_deterministic(flag)``: enables
    ``torch.use_deterministic_algorithms`` only when explicitly requested.

Legacy behavior is preserved by default: when ``--seed 0`` (the default), no
seeding is applied at all (the function is a no-op), so existing runs are
byte-identical to pre-P01. Only when the user passes a non-zero ``--seed`` does
seeding activate. ``--deterministic`` only takes effect when explicitly set.

NumPy constraint: ``np.random.seed`` only accepts seeds in ``[0, 2**32 - 1]``.
A derived actor seed must therefore be clamped to that range (it must also be
non-zero so it never collides with ``NO_SEED``). We mask to 32 bits; the
entropy of SHA-256 is far larger than 32 bits so cross-actor collisions among
the few dozen actors in a run are astronomically unlikely.
"""

from __future__ import annotations

import hashlib
from typing import Union

__all__ = [
    "derive_actor_seed",
    "maybe_set_global_deterministic",
    "set_global_seed",
]

#: The sentinel "no seeding" value. When --seed equals this (the default 0),
#: set_global_seed is a no-op so legacy behavior is byte-identical.
NO_SEED = 0

# np.random.seed only accepts [0, 2**32 - 1]. Derived actor seeds are masked to
# this range so they can be passed directly to numpy without rejection.
_MAX_NUMPY_SEED = (1 << 32) - 1


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


def derive_actor_seed(
    base_seed: int, device_token: Union[int, str], actor_id: int
) -> int:
    """Derive a deterministic per-actor seed from (base_seed, device, actor).

    Each actor process gets a unique but reproducible seed, so a fixed base_seed
    yields the same set of deals and rollouts across runs. The derivation uses
    a stable hash (not Python's randomized hash) so it is reproducible across
    process restarts (important for spawn-based multiprocessing).

    ``device_token`` may be an integer GPU id (e.g. ``0``) or the string
    ``"cpu"``. The actor's ``device`` argument in ``douzero.dmc.utils.act`` is
    the device token passed to ``act()``; it is ``"cpu"`` for CPU actors and a
    GPU index for CUDA actors. Both are accepted so the caller never needs to
    coerce it to an int (which would crash for ``"cpu"``).

    Returns ``NO_SEED`` (0) when ``base_seed == NO_SEED``, so unseeded actors
    keep legacy behavior.

    The result is always in ``[1, 2**32 - 1]`` (non-zero, numpy-safe), so it can
    be passed directly to ``set_global_seed`` / ``np.random.seed``.
    """
    if base_seed == NO_SEED:
        return NO_SEED
    # Use a stable SHA-based derivation (Python's built-in hash is randomized
    # per process by PYTHONHASHSEED, so it cannot be used for reproducibility).
    # device_token may be "cpu" or a GPU index; str() makes the token canonical
    # and stable regardless of type.
    token = f"douzero-actor|{base_seed}|{device_token}|{actor_id}".encode()
    digest = hashlib.sha256(token).digest()
    # Mask to 32 bits so the result fits in np.random.seed's accepted range,
    # and avoid 0 (the no-seed sentinel). Collisions among the small number of
    # actors in a run are astronomically unlikely with SHA-256 entropy.
    derived = int.from_bytes(digest[:4], "big") & _MAX_NUMPY_SEED
    return derived if derived != 0 else 1


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
