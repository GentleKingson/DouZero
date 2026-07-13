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

from typing import Any, Mapping, Protocol, runtime_checkable

from .schema import HumanGameRecord, RecordValidationError


# Keys that must NEVER appear in source_metadata (personal identifiers /
# credentials). An adapter carrying any of these is rejected at ingest.
_FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "user_id",
        "username",
        "user_name",
        "account",
        "account_id",
        "email",
        "phone",
        "ip",
        "ip_address",
        "password",
        "token",
        "cookie",
        "session",
        "device_id",
    }
)


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
    """Return a sanitized, audited copy of ``source_metadata``.

    Drops any key in :data:`_FORBIDDEN_METADATA_KEYS` and any key whose value
    looks like a credential (a string containing ``"password"`` or
    ``"token"``). Adapters should pass their metadata through this before
    attaching it to a record.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if key.lower() in _FORBIDDEN_METADATA_KEYS:
            continue
        if isinstance(key, str) and any(
            bad in key.lower() for bad in ("password", "token", "secret", "cookie")
        ):
            continue
        cleaned[key] = value
    return cleaned


def assert_no_forbidden_metadata(metadata: Mapping[str, Any]) -> None:
    """Raise if ``metadata`` carries a forbidden identifier/credential key.

    Used by the ingest boundary so an adapter that forgets to call
    :func:`audit_source_metadata` still cannot leak a forbidden field into a
    canonical record.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    for key in metadata:
        if isinstance(key, str) and key.lower() in _FORBIDDEN_METADATA_KEYS:
            raise RecordValidationError(
                f"source_metadata key {key!r} is a forbidden personal "
                f"identifier/credential; drop it in the adapter."
            )
