"""Ingest external human-game records into canonical JSONL (P08).

The ingest stage:

1. Reads raw external-format records (already acquired and authorized — no
   scraping/automation here).
2. Converts each to a canonical :class:`~douzero.human_data.schema.HumanGameRecord`
   via an :class:`~douzero.human_data.adapters.Adapter`.
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

from .adapters import Adapter, assert_no_forbidden_metadata
from .schema import HumanGameRecord, RecordValidationError, write_jsonl

logger = logging.getLogger(__name__)


class IngestError(ValueError):
    """Raised when ingest cannot convert a raw payload canonically."""


def ingest_record(
    raw: Mapping[str, Any],
    adapter: Adapter,
) -> HumanGameRecord:
    """Convert one raw payload to a canonical record via ``adapter``.

    Sanitizes ``source_metadata`` and rejects any payload whose metadata still
    carries a forbidden key after the adapter ran (defence in depth).
    """
    if not callable(adapter):
        raise IngestError(
            f"adapter must be callable, got {type(adapter).__name__}"
        )
    try:
        record = adapter(raw)
    except RecordValidationError as exc:
        raise IngestError(f"adapter rejected payload: {exc}") from exc
    if not isinstance(record, HumanGameRecord):
        raise IngestError(
            f"adapter must return HumanGameRecord, got {type(record).__name__}"
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
                "dropping duplicate game_id %r (first occurrence kept)",
                record.game_id,
            )
            continue
        seen.add(record.game_id)
        yield record


def ingest_batch(
    raw_records: Iterable[Mapping[str, Any]],
    adapter: Adapter,
    *,
    sort_by_game_id: bool = True,
) -> list[HumanGameRecord]:
    """Ingest a batch of raw payloads into canonical, de-duplicated records.

    Returns a list (the canonical set is bounded by the input). When
    ``sort_by_game_id`` is True the output is sorted by ``game_id`` so the
    canonical file is byte-stable for a fixed input.
    """
    converted: list[HumanGameRecord] = []
    for raw in raw_records:
        converted.append(ingest_record(raw, adapter))
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
) -> int:
    """Ingest raw payloads and write canonical JSONL. Returns records written."""
    records = ingest_batch(
        raw_records, adapter, sort_by_game_id=sort_by_game_id
    )
    return write_jsonl(records, output_path)
