"""Deterministic canonical-dataset rebuild with game-level exclusion."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .identifiers import is_canonical_game_id
from .schema import (
    dataset_manifest_path,
    read_jsonl,
    verify_jsonl_manifest,
    write_jsonl,
)


@dataclass(frozen=True)
class RebuildReport:
    input_records: int
    output_records: int
    excluded_records: int
    requested_ids: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "excluded_records": self.excluded_records,
            "requested_ids": self.requested_ids,
        }


def rebuild_without_game_ids(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    excluded_game_ids: Iterable[str],
) -> RebuildReport:
    """Atomically rebuild canonical JSONL while excluding complete games.

    The report contains counts only. It never returns or logs game identifiers
    or record contents.
    """

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ for a rebuild")
    excluded = set(excluded_game_ids)
    if not excluded:
        raise ValueError("at least one excluded game_id is required")
    if any(not is_canonical_game_id(game_id) for game_id in excluded):
        raise ValueError("every excluded game_id must be a canonical opaque ID")

    source_manifest = verify_jsonl_manifest(source)
    records = list(read_jsonl(str(source)))
    retained = [record for record in records if record.game_id not in excluded]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        write_jsonl(
            retained,
            str(temporary),
            config_identity={
                "operation": "rebuild_without_game_ids",
                "requested_ids": len(excluded),
                "source_dataset_sha256": source_manifest["dataset_sha256"],
            },
        )
        os.replace(temporary, destination)
        os.replace(
            dataset_manifest_path(temporary),
            dataset_manifest_path(destination),
        )
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            dataset_manifest_path(temporary).unlink(missing_ok=True)
    return RebuildReport(
        input_records=len(records),
        output_records=len(retained),
        excluded_records=len(records) - len(retained),
        requested_ids=len(excluded),
    )
