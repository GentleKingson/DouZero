"""Privacy detection for the human-game pipeline (P08).

Canonical ``source_metadata`` uses a strict flat allowlist at both the ingest
and record boundaries. Pattern detection remains defense in depth for other
fixed-schema string fields such as timestamps and scoring results.

AGENTS.md: "Do not store personal identifiers or credentials." This module is
the single source of truth for what counts as a personal identifier or
credential. It scans recursively (mappings, lists, tuples, sets) and matches
both forbidden KEY substrings and sensitive VALUE patterns (credentials + PII).
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Iterator, Mapping


SOURCE_METADATA_ALLOWED_KEYS: frozenset[str] = frozenset({
    "source",
    "license",
    "dataset_version",
    "batch_id",
    "collection_method",
})

_SOURCE_METADATA_PATTERNS: dict[str, re.Pattern[str]] = {
    "source": re.compile(r"^[a-z0-9][a-z0-9._/-]{0,63}$"),
    "license": re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+_()/-]{0,127}$"),
    "dataset_version": re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
    "batch_id": re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
    "collection_method": re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$"),
}


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
# The phone patterns are deliberately tight: a bare run of 8+ digits (which a
# SHA-256 hash contains) is NOT a phone number. We require EITHER a '+' prefix
# (international format) OR explicit group separators (US-style ddd-ddd-dddd).
PII_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",      # email
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",       # IPv4
        r"\+\d[\d\s\-().]{7,}\d",                         # phone: +1-555-...
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",                  # phone: 555-123-4567
        r"https?://[^\s?]+\?[^\s]*(?:id|user|player|account)=",  # URL query ID
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
    if (
        any(p.search(v) for p in CREDENTIAL_VALUE_PATTERNS)
        or any(p.search(v) for p in PII_VALUE_PATTERNS)
    ):
        return True
    try:
        ipaddress.ip_address(v.strip("[]"))
    except ValueError:
        return False
    return True


def assert_valid_source_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the canonical, flat provenance allowlist.

    Metadata is intentionally not an extension bag.  Unknown or nested fields
    are rejected because their semantics cannot prove that they are free of
    player identifiers.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    for key in metadata:
        if not isinstance(key, str):
            raise ValueError(
                f"source_metadata key {key!r} must be a string, got "
                f"{type(key).__name__}"
            )
    unknown = set(metadata) - SOURCE_METADATA_ALLOWED_KEYS
    if unknown:
        unknown_display = sorted(repr(key) for key in unknown)
        raise ValueError(
            f"source_metadata has unknown key(s) {unknown_display!r}; allowed "
            f"keys are {sorted(SOURCE_METADATA_ALLOWED_KEYS)!r}"
        )
    for key, value in metadata.items():
        if not isinstance(value, str):
            raise ValueError(
                f"source_metadata[{key!r}] must be a string, got "
                f"{type(value).__name__}"
            )
        if _SOURCE_METADATA_PATTERNS[key].fullmatch(value) is None:
            raise ValueError(
                f"source_metadata[{key!r}] has an invalid length or character set"
            )
        if value_is_sensitive(value):
            raise ValueError(
                f"source_metadata[{key!r}] carries a sensitive value"
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
    """Return only valid fields from the flat provenance allowlist."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in SOURCE_METADATA_ALLOWED_KEYS:
            continue
        try:
            assert_valid_source_metadata({key: value})
        except ValueError:
            continue
        out[key] = value
    return out
