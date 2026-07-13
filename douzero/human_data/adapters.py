"""External-format adapters for the human-game pipeline (P08).

The training code never hard-codes a specific platform's record format.
Instead, an :class:`Adapter` is a thin converter from a raw external payload
(a platform export, a research dataset, etc.) into a canonical
:class:`~douzero.human_data.schema.HumanGameRecord`.

AGENTS.md: "Use only lawfully obtained and authorized data." Adapters MUST NOT
perform scraping, account automation, anti-detection, or any platform-ToS
bypass. They operate on already-acquired, authorized files only. They MUST
drop personal identifiers and credentials (see
:func:`audit_source_metadata`).
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

from .schema import HumanGameRecord, RecordValidationError


# Forbidden KEY substrings — a key is rejected if its lowercased form CONTAINS
# any of these (so ``api_token``, ``client_secret``, ``user_email``,
# ``auth_token``, ``device_id`` are all caught, not just exact matches).
_FORBIDDEN_KEY_SUBSTRINGS: tuple[str, ...] = (
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

# Credential-like VALUE patterns. A string value matching any of these is
# treated as a leaked credential even when the key looks benign
# (e.g. ``{"note": "Bearer abc123"}``).
_CREDENTIAL_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^bearer\s+",                 # Authorization: Bearer ...
        r"^(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S",  # key=value form
        r"^-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",  # PEM private key
        r"^gh[ps]_[A-Za-z0-9]{20,}",   # GitHub token shape
        r"^AKIA[0-9A-Z]{16}",          # AWS access key id shape
    )
)

# PII VALUE patterns — personal identifiers that must not be stored even under
# a benign key (Blocker 2: the value detector previously caught only credential
# shapes, letting plain emails/phones/IPs through). These are searched anywhere
# in the string value (not just at the start).
_PII_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",      # email
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",       # IPv4 address
        r"\+?\d[\d\s\-().]{7,}\d",                        # phone (8+ digits)
    )
)

# Container types that are recursively scanned/sanitized. tuple and set are
# included (Blocker 2: a tuple of dicts bypassed the earlier Mapping/list-only
# recursion). frozenset is NOT JSON-serializable and is rejected at the boundary.
_RECURSABLE_CONTAINERS = (Mapping, list, tuple, set, frozenset)


def _key_is_forbidden(key: str) -> bool:
    """Return True if ``key`` (any case) contains a forbidden substring."""
    if not isinstance(key, str):
        return False
    kl = key.lower()
    return any(sub in kl for sub in _FORBIDDEN_KEY_SUBSTRINGS)


def _value_is_sensitive(value: Any) -> bool:
    """Return True if ``value`` looks like a leaked credential OR PII.

    Blocker 2: the earlier ``_value_is_credential`` only matched credential
    shapes (Bearer, PEM, token shapes). Plain personal identifiers — emails,
    phone numbers, IPv4 addresses — under benign keys slipped through. This
    also matches PII patterns so ``{"contact": "alice@example.com"}`` or
    ``{"origin": "203.0.113.10"}`` are caught and dropped.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    return (
        any(p.search(v) for p in _CREDENTIAL_VALUE_PATTERNS)
        or any(p.search(v) for p in _PII_VALUE_PATTERNS)
    )


# Backward-compatible alias.
_value_is_credential = _value_is_sensitive


def _scan_for_forbidden(
    obj: Any, path: str = ""
) -> Iterator[tuple[str, str]]:
    """Recursively yield ``(path, reason)`` for forbidden keys/values.

    Traverses nested mappings, lists, tuples, AND sets so a credential or PII
    hidden inside ``{"profile": {"email": ...}}``, ``[{"token": ...}]``, or
    ``({"user_email": ...},)`` is still found (Blocker 2: tuples previously
    bypassed the recursion).
    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and _key_is_forbidden(key):
                yield (child_path, f"forbidden key {key!r}")
            if isinstance(value, _RECURSABLE_CONTAINERS):
                yield from _scan_for_forbidden(value, child_path)
            elif _value_is_sensitive(value):
                yield (child_path, "sensitive value")
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, item in enumerate(obj):
            child_path = f"{path}[{i}]"
            yield from _scan_for_forbidden(item, child_path)
    elif _value_is_sensitive(obj):
        yield (path or "<root>", "sensitive value")


@runtime_checkable
class Adapter(Protocol):
    """Convert one external-format payload into a canonical record.

    Implementations may be a function or a class with a ``__call__`` method.
    The protocol is structural (``runtime_checkable``) so an adapter can be a
    plain function: ``def my_adapter(raw: Mapping) -> HumanGameRecord: ...``.

    Contract
    --------
    - The adapter MUST NOT reach the network. It consumes an already-acquired
      in-memory mapping.
    - The adapter MUST NOT retain personal identifiers; call
      :func:`audit_source_metadata` on any metadata it attaches.
    - The adapter is responsible for mapping the external card encoding onto
      the legacy integer code points (3..14, 17, 20, 30) and for producing the
      :class:`~douzero.env.rules.RuleSet` identity triple the game was played
      under. Rule normalization (resolving the ruleset hash) typically belongs
      here, not in the trainer.
    - The adapter does NOT validate game legality — that is the job of the
      replay validator (:mod:`douzero.human_data.validate`). It only normalizes
      the shape into a :class:`HumanGameRecord`.
    """

    def __call__(self, raw: Mapping[str, Any]) -> HumanGameRecord:
        ...

def audit_source_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively sanitized copy of ``source_metadata``.

    Drops any key whose lowercased form contains a forbidden substring
    (``api_token``, ``client_secret``, ``user_email``, …), recurses into nested
    mappings/lists, and drops credential-like string values
    (``Bearer ...``, ``api_key=...``, PEM private keys, known token shapes).
    Adapters should pass their metadata through this before attaching it to a
    record. The returned mapping never shares mutable structure with the input.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    cleaned = _sanitize_mapping(metadata)
    return cleaned


def _sanitize_mapping(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively rebuild a mapping without forbidden keys/sensitive values."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(key, str) and _key_is_forbidden(key):
            continue
        sanitized = _sanitize_value(value)
        if sanitized is _DROP:
            continue
        out[key] = sanitized
    return out


def _sanitize_sequence(obj: "list | tuple | set | frozenset") -> list:
    """Recursively rebuild a sequence (list/tuple/set -> list) without
    sensitive scalar items. tuple/set are converted to list so the output is
    JSON-serializable (Blocker 2: tuples previously bypassed sanitization)."""
    out: list = []
    for item in obj:
        sanitized = _sanitize_value(item)
        if sanitized is _DROP:
            continue
        out.append(sanitized)
    return out


class _DropSentinel:
    """Sentinel returned by :func:`_sanitize_value` to mean 'drop this entry'."""

    __slots__ = ()


_DROP = _DropSentinel()


def _sanitize_value(value: Any) -> Any:
    """Sanitize one value; returns :data:`_DROP` if it should be removed."""
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return _sanitize_sequence(value)
    # Scalar: drop credential-like / PII strings.
    if _value_is_sensitive(value):
        return _DROP
    return value


def assert_no_forbidden_metadata(metadata: Mapping[str, Any]) -> None:
    """Raise if ``metadata`` (recursively) carries a forbidden field.

    Used by the ingest boundary so an adapter that forgets to call
    :func:`audit_source_metadata` still cannot leak a personal identifier or
    credential into a canonical record. Scans nested mappings/lists and
    credential-like values, not just top-level exact-match keys.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    findings = list(_scan_for_forbidden(metadata))
    if findings:
        detail = ", ".join(f"{p} ({r})" for p, r in findings[:5])
        raise RecordValidationError(
            f"source_metadata carries forbidden personal-identifier/credential "
            f"field(s): {detail}. Drop/sanitize them in the adapter "
            f"(audit_source_metadata)."
        )
