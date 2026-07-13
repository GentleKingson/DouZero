"""Privacy detection for the human-game pipeline (P08).

Centralizes the forbidden-key and sensitive-value detection so it can be applied
at BOTH the ingest boundary (:mod:`douzero.human_data.adapters`) AND the
canonical record boundary (:mod:`douzero.human_data.schema`). Moving the scan
to the record boundary (Blocker 2, round 4) means a record constructed directly
via ``record_from_dict`` — bypassing ingest — is still audited before it can
reach validation, BC sampling, or training.

AGENTS.md: "Do not store personal identifiers or credentials." This module is
the single source of truth for what counts as a personal identifier or
credential. It scans recursively (mappings, lists, tuples, sets) and matches
both forbidden KEY substrings and sensitive VALUE patterns (credentials + PII).
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Mapping


# Forbidden KEY substrings — a key is rejected if its lowercased form CONTAINS
# any of these (so ``api_token``, ``client_secret``, ``user_email``,
# ``auth_token``, ``device_id`` are all caught, not just exact matches).
FORBIDDEN_KEY_SUBSTRINGS: tuple[str, ...] = (
    "password", "passwd", "pwd",
    "token", "secret", "apikey", "api_key",
    "cookie", "session",
    "email", "phone",
    "ip_address", "ipaddress",
    "user_id", "userid", "username", "user_name",
    "account", "device_id", "deviceid",
    "credential", "private_key", "privatekey",
    "ssn", "national_id",
)

# Credential-like VALUE patterns.
CREDENTIAL_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^bearer\s+",
        r"^(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S",
        r"^-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",
        r"^gh[ps]_[A-Za-z0-9]{20,}",
        r"^AKIA[0-9A-Z]{16}",
    )
)

# PII VALUE patterns — personal identifiers under benign keys.
PII_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",      # email
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",       # IPv4
        r"\+?\d[\d\s\-().]{7,}\d",                        # phone
    )
)

# Container types that are recursively scanned (tuple/set included so they
# cannot bypass the scan).
_RECURSABLE_CONTAINERS = (Mapping, list, tuple, set, frozenset)


def key_is_forbidden(key: str) -> bool:
    """Return True if ``key`` (any case) contains a forbidden substring."""
    if not isinstance(key, str):
        return False
    kl = key.lower()
    return any(sub in kl for sub in FORBIDDEN_KEY_SUBSTRINGS)


def value_is_sensitive(value: Any) -> bool:
    """Return True if ``value`` looks like a leaked credential OR PII."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    return (
        any(p.search(v) for p in CREDENTIAL_VALUE_PATTERNS)
        or any(p.search(v) for p in PII_VALUE_PATTERNS)
    )


def scan_for_forbidden(
    obj: Any, path: str = ""
) -> Iterator[tuple[str, str]]:
    """Recursively yield ``(path, reason)`` for forbidden keys/values.

    Traverses nested mappings, lists, tuples, and sets so a credential or PII
    hidden inside any container is found.
    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key_is_forbidden(key):
                yield (child_path, f"forbidden key {key!r}")
            if isinstance(value, _RECURSABLE_CONTAINERS):
                yield from scan_for_forbidden(value, child_path)
            elif value_is_sensitive(value):
                yield (child_path, "sensitive value")
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]"
            yield from scan_for_forbidden(item, child_path)
    elif value_is_sensitive(obj):
        yield (path or "<root>", "sensitive value")


def assert_no_forbidden(obj: Any, label: str = "object") -> None:
    """Raise ``ValueError`` if ``obj`` (recursively) carries a forbidden field.

    ``label`` prefixes the error message so the caller can identify which
    record field failed (e.g. ``"source_metadata"``).
    """
    findings = list(scan_for_forbidden(obj))
    if findings:
        detail = ", ".join(f"{p} ({r})" for p, r in findings[:5])
        raise ValueError(
            f"{label} carries forbidden personal-identifier/credential "
            f"field(s): {detail}. Drop/sanitize them before constructing the "
            f"record."
        )


def sanitize_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively sanitized copy of ``metadata``.

    Drops forbidden keys and sensitive values, recurses into all container
    types (tuple/set -> list). The output is JSON-safe.
    """
    return _sanitize_mapping(metadata)


class _DropSentinel:
    __slots__ = ()


_DROP = _DropSentinel()


def _sanitize_mapping(obj: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(key, str) and key_is_forbidden(key):
            continue
        sanitized = _sanitize_value(value)
        if sanitized is _DROP:
            continue
        out[key] = sanitized
    return out


def _sanitize_sequence(obj: "list | tuple | set | frozenset") -> list:
    out: list = []
    for item in obj:
        sanitized = _sanitize_value(item)
        if sanitized is _DROP:
            continue
        out.append(sanitized)
    return out


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return _sanitize_sequence(value)
    if value_is_sensitive(value):
        return _DROP
    return value
