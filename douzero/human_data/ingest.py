"""Ingest external human-game records into canonical JSONL (P08).

The ingest stage:

1. Reads raw external-format records (already acquired and authorized — no
   scraping/automation here).
2. Converts each to an attested canonical record via an
   :class:`~douzero.human_data.adapters.Adapter` supplied with the run's
   project-key pseudonymizer.
3. Sanitizes ``source_metadata`` (drops forbidden personal-identifier /
   credential keys) at the boundary.
4. De-duplicates by ``game_id`` (first occurrence wins, in iteration order).
5. Writes canonical JSONL (sorted by ``game_id`` for reproducibility).

Ingest does NOT validate game legality — that is the separate
:mod:`douzero.human_data.validate` stage. The two stages are intentionally
decoupled so a quarantined record can be inspected without re-running ingest.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Mapping

from .adapters import (
    Adapter,
    AttestedAdapterRecord,
    assert_no_forbidden_metadata,
)
from .identifiers import ExternalGameIdPseudonymizer
from .schema import HumanGameRecord, RecordValidationError, write_jsonl

logger = logging.getLogger(__name__)


class IngestError(ValueError):
    """Raised when ingest cannot convert a raw payload canonically."""


def ingest_record(
    raw: Mapping[str, Any],
    adapter: Adapter,
    *,
    project_key: bytes | bytearray | None = None,
) -> HumanGameRecord:
    """Convert one raw payload to a canonical record via ``adapter``.

    ``project_key`` is mandatory for this external-data boundary. The adapter
    must return an :class:`AttestedAdapterRecord` generated through the supplied
    pseudonymizer; shape-only canonical IDs fail closed.
    """
    try:
        pseudonymizer = ExternalGameIdPseudonymizer(project_key)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IngestError(
            "external ingest requires an authorized project HMAC key"
        ) from exc
    return _ingest_record(raw, adapter, pseudonymizer=pseudonymizer)


def _ingest_record(
    raw: Mapping[str, Any],
    adapter: Adapter,
    *,
    pseudonymizer: ExternalGameIdPseudonymizer,
) -> HumanGameRecord:
    """Apply and verify one adapter without exposing key or raw-ID details."""

    if not callable(adapter):
        raise IngestError(
            f"adapter must be callable, got {type(adapter).__name__}"
        )
    try:
        adapted = adapter(raw, pseudonymizer=pseudonymizer)
    except KeyboardInterrupt:
        raise
    except BaseException:
        raise IngestError("adapter rejected payload (details redacted)") from None
    if not isinstance(adapted, AttestedAdapterRecord):
        raise IngestError(
            "adapter must return AttestedAdapterRecord under the strict keyed "
            f"contract, got {type(adapted).__name__}"
        )
    record = adapted.record
    if not pseudonymizer.verify(
        adapted.identity, record_game_id=record.game_id
    ):
        raise IngestError(
            "adapter game_id is not bound to this run's project-key "
            "pseudonymization attestation"
        )
    # Defence in depth: even if the adapter forgot to sanitize metadata, reject
    # any forbidden identifier/credential before the record leaves ingest.
    try:
        assert_no_forbidden_metadata(record.source_metadata)
    except RecordValidationError as exc:
        raise IngestError(
            f"adapter leaked a forbidden metadata key: {exc}"
        ) from exc
    return record


def dedupe_by_game_id(
    records: Iterable[HumanGameRecord],
) -> Iterator[HumanGameRecord]:
    """Yield records with duplicate ``game_id`` removed (first wins).

    Duplicate games are a common ingestion artefact (re-exports, overlapping
    batches). They are dropped, not merged, so the canonical set is stable.
    """
    seen: set[str] = set()
    for record in records:
        if record.game_id in seen:
            logger.warning(
                "dropping duplicate canonical game record (identifier redacted; "
                "first occurrence kept)"
            )
            continue
        seen.add(record.game_id)
        yield record


def ingest_batch(
    raw_records: Iterable[Mapping[str, Any]],
    adapter: Adapter,
    *,
    sort_by_game_id: bool = True,
    project_key: bytes | bytearray | None = None,
) -> list[HumanGameRecord]:
    """Ingest a batch of raw payloads into canonical, de-duplicated records.

    Returns a list (the canonical set is bounded by the input). When
    ``sort_by_game_id`` is True the output is sorted by ``game_id`` so the
    canonical file is byte-stable for a fixed input.
    """
    try:
        pseudonymizer = ExternalGameIdPseudonymizer(project_key)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IngestError(
            "external ingest requires an authorized project HMAC key"
        ) from exc
    converted: list[HumanGameRecord] = []
    for raw in raw_records:
        converted.append(
            _ingest_record(raw, adapter, pseudonymizer=pseudonymizer)
        )
    deduped = list(dedupe_by_game_id(converted))
    if sort_by_game_id:
        deduped.sort(key=lambda r: r.game_id)
    return deduped


def ingest_to_jsonl(
    raw_records: Iterable[Mapping[str, Any]],
    adapter: Adapter,
    output_path: str,
    *,
    sort_by_game_id: bool = True,
    project_key: bytes | bytearray | None = None,
) -> int:
    """Ingest raw payloads and write canonical JSONL. Returns records written."""
    records = ingest_batch(
        raw_records,
        adapter,
        sort_by_game_id=sort_by_game_id,
        project_key=project_key,
    )
    return write_jsonl(records, output_path)
