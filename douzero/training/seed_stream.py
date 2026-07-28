"""Canonical deterministic seed streams shared by formal training runners."""

from __future__ import annotations

import hashlib
import json

FORMAL_SEED_DERIVATION_V1 = (
    "sha256(root_seed,stream_name,worker_id,episode_id)-v1"
)
TOPOLOGY_LOCAL_SEED_DERIVATION_V1 = "topology-local-environment-stream-v1"


def derive_formal_stream_seed(
    root_seed: int,
    stream_name: str,
    worker_id: int,
    episode_id: int,
) -> int:
    """Derive one stable 32-bit seed from the frozen formal contract."""

    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (root_seed, worker_id, episode_id)
    ):
        raise TypeError("formal seed coordinates must be integers")
    if worker_id < 0 or episode_id < 0 or not stream_name:
        raise ValueError("formal seed coordinates must be non-negative and named")
    envelope = json.dumps(
        {
            "contract": FORMAL_SEED_DERIVATION_V1,
            "episode_id": episode_id,
            "root_seed": root_seed,
            "stream_name": stream_name,
            "worker_id": worker_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(envelope).digest()[:4], "big")


__all__ = [
    "FORMAL_SEED_DERIVATION_V1",
    "TOPOLOGY_LOCAL_SEED_DERIVATION_V1",
    "derive_formal_stream_seed",
]
