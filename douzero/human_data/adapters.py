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

from .privacy import (
    assert_valid_source_metadata,
    sanitize_mapping as _sanitize_metadata,
)
from .schema import HumanGameRecord, RecordValidationError


@runtime_checkable
class Adapter(Protocol):
    """Convert one external-format payload into a canonical record.

    Implementations may be a function or a zero-argument class with a
    ``__call__`` method. The CLI instantiates adapter classes once. Configured
    adapters should be exposed as callable instances or adapter functions that
    close over configuration. The protocol is structural
    (``runtime_checkable``), so an adapter can also be a plain function:
    ``def my_adapter(raw: Mapping) -> HumanGameRecord: ...``.

    Contract
    --------
    - The adapter MUST NOT reach the network. It consumes an already-acquired
      in-memory mapping.
    - The adapter MUST NOT retain personal identifiers; call
      :func:`audit_source_metadata` on any metadata it attaches. The canonical
      record boundary (:class:`~douzero.human_data.schema.HumanGameRecord`)
      also runs a fail-closed privacy scan, so a forbidden field will be caught
      there even if the adapter forgets to sanitize.
    - Raw external game/player identifiers MUST be mapped with
      :func:`douzero.human_data.pseudonymize_external_game_id` and a project
      secret before constructing the record.
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
    """Return a sanitized allowlisted copy of ``source_metadata``.

    Thin wrapper around :func:`douzero.human_data.privacy.sanitize_mapping`.
    Keeps only the strict, flat provenance allowlist and drops invalid values.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    return _sanitize_metadata(metadata)


def assert_no_forbidden_metadata(metadata: Mapping[str, Any]) -> None:
    """Raise unless ``metadata`` matches the strict provenance schema.

    Used by the ingest boundary; the canonical record boundary
    (:class:`~douzero.human_data.schema.HumanGameRecord`) also runs this scan
    fail-closed at construction so a direct ``record_from_dict`` load cannot
    bypass it.
    """
    if not isinstance(metadata, Mapping):
        raise RecordValidationError(
            f"source_metadata must be a mapping, got {type(metadata).__name__}"
        )
    try:
        assert_valid_source_metadata(metadata)
    except ValueError as exc:
        raise RecordValidationError(str(exc)) from exc
