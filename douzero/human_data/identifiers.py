"""Opaque identifiers for canonical human-game records.

External platform identifiers are personal data.  They must be mapped with a
project-held secret before a record is constructed; the raw value must never
be written to canonical JSONL.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


GAME_ID_PREFIX = "dzg_"
_GAME_ID_PATTERN = re.compile(r"^dzg_[0-9a-f]{64}$")
_INTERNAL_ID_DOMAIN = b"douzero.synthetic-game-id.v1\0"
_EXTERNAL_ATTESTATION_DOMAIN = b"douzero.external-game-id-attestation.v1\0"
HMAC_KEY_FILE_ENV = "DOUZERO_HUMAN_DATA_HMAC_KEY_FILE"
EXTERNAL_ID_ATTESTATION_VERSION = "douzero-external-id-attestation-v1"


@dataclass(frozen=True)
class ExternalGameIdentity:
    """Opaque keyed identity returned to an external adapter.

    The attestation proves at the ingest boundary that ``game_id`` came from
    the project-key pseudonymizer used for this run. It contains neither the
    raw platform identifier nor the project key.
    """

    game_id: str
    attestation: str = field(repr=False)
    version: str = EXTERNAL_ID_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        if not is_canonical_game_id(self.game_id):
            raise ValueError("external identity game_id is not canonical")
        if self.version != EXTERNAL_ID_ATTESTATION_VERSION:
            raise ValueError("unsupported external identity attestation version")
        if not re.fullmatch(r"[0-9a-f]{64}", self.attestation):
            raise ValueError("external identity attestation is malformed")


class ExternalGameIdPseudonymizer:
    """Project-key holder supplied to strict external adapters.

    Adapter code receives this object rather than raw key bytes. Its repr is
    deliberately constant so exceptions and logs cannot reveal key material.
    """

    __slots__ = ("__project_key",)

    def __init__(self, project_key: Union[bytes, bytearray]) -> None:
        if not isinstance(project_key, (bytes, bytearray)) or len(project_key) < 32:
            raise ValueError("project_key must contain at least 32 bytes")
        self.__project_key = bytes(project_key)

    def __repr__(self) -> str:
        return "ExternalGameIdPseudonymizer(<redacted>)"

    def pseudonymize(self, external_id: str) -> ExternalGameIdentity:
        """Return a keyed canonical ID plus a run-verifiable attestation."""

        game_id = pseudonymize_external_game_id(
            external_id, project_key=self.__project_key
        )
        attestation = hmac.new(
            self.__project_key,
            _EXTERNAL_ATTESTATION_DOMAIN + game_id.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return ExternalGameIdentity(game_id=game_id, attestation=attestation)

    def verify(self, identity: object, *, record_game_id: str) -> bool:
        """Verify an adapter result without accepting shape-only identifiers."""

        if not isinstance(identity, ExternalGameIdentity):
            return False
        if not is_canonical_game_id(record_game_id):
            return False
        if not hmac.compare_digest(identity.game_id, record_game_id):
            return False
        expected = hmac.new(
            self.__project_key,
            _EXTERNAL_ATTESTATION_DOMAIN + record_game_id.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(identity.attestation, expected)


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


def load_hmac_project_key(path: str | os.PathLike[str] | None = None) -> bytes:
    """Load the pseudonymization key without logging its path or contents.

    ``path`` defaults to :envvar:`DOUZERO_HUMAN_DATA_HMAC_KEY_FILE`. The key
    remains external to canonical data, checkpoints, reports, and packages.
    """

    resolved = (
        os.fspath(path)
        if path is not None
        else os.environ.get(HMAC_KEY_FILE_ENV, "")
    )
    if not resolved:
        raise ValueError(f"set {HMAC_KEY_FILE_ENV} to an authorized HMAC key file")
    key_path = Path(resolved)
    if not key_path.is_file():
        raise ValueError(f"{HMAC_KEY_FILE_ENV} does not name a regular file")
    try:
        key = key_path.read_bytes()
    except OSError:
        raise ValueError("unable to read configured human-data HMAC key file") from None
    if len(key) < 32:
        raise ValueError("human-data HMAC key file must contain at least 32 bytes")
    return key
