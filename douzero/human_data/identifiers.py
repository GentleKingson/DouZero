"""Opaque identifiers for canonical human-game records.

External platform identifiers are personal data.  They must be mapped with a
project-held secret before a record is constructed; the raw value must never
be written to canonical JSONL.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Union


GAME_ID_PREFIX = "dzg_"
_GAME_ID_PATTERN = re.compile(r"^dzg_[0-9a-f]{64}$")
_INTERNAL_ID_DOMAIN = b"douzero.synthetic-game-id.v1\0"


def is_canonical_game_id(value: object) -> bool:
    """Return whether ``value`` is an opaque canonical game identifier."""
    return (
        isinstance(value, str)
        and _GAME_ID_PATTERN.fullmatch(value) is not None
    )


def make_internal_game_id(stable_key: str) -> str:
    """Create a deterministic opaque ID for synthetic/internal records only.

    ``stable_key`` must not be an identifier imported from an external data
    source.  External IDs require :func:`pseudonymize_external_game_id`.
    """
    if not isinstance(stable_key, str) or not stable_key:
        raise ValueError("stable_key must be a non-empty string")
    digest = hashlib.sha256(
        _INTERNAL_ID_DOMAIN + stable_key.encode("utf-8")
    ).hexdigest()
    return GAME_ID_PREFIX + digest


def pseudonymize_external_game_id(
    external_id: str,
    *,
    project_key: Union[bytes, bytearray],
) -> str:
    """Map an external ID to a deterministic, irreversible canonical ID.

    A distinct project key prevents cross-dataset correlation and dictionary
    recovery of low-entropy platform identifiers.  The key is never stored in
    the canonical record.
    """
    if not isinstance(external_id, str) or not external_id:
        raise ValueError("external_id must be a non-empty string")
    if not isinstance(project_key, (bytes, bytearray)) or len(project_key) < 32:
        raise ValueError("project_key must contain at least 32 bytes")
    digest = hmac.new(
        bytes(project_key),
        b"douzero.external-game-id.v1\0" + external_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return GAME_ID_PREFIX + digest
