#!/usr/bin/env python3
"""Atomically publish human data without the requested opaque game IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from douzero.human_data.rebuild import (
    RebuildCommitUncertainError,
    RebuildPostCommitError,
    rebuild_without_game_ids,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="validated canonical JSONL")
    parser.add_argument(
        "--output",
        required=True,
        help="active dataset pointer backed by an immutable JSONL/manifest version",
    )
    parser.add_argument(
        "--exclude-game-id",
        action="append",
        default=[],
        help="opaque dzg_... identifier to remove; repeat as needed",
    )
    parser.add_argument(
        "--exclude-ids-file",
        help="newline-delimited opaque IDs (never printed)",
    )
    args = parser.parse_args(argv)
    excluded = list(args.exclude_game_id)
    if args.exclude_ids_file:
        excluded.extend(
            line.strip()
            for line in Path(args.exclude_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    try:
        report = rebuild_without_game_ids(args.input, args.output, excluded)
    except RebuildPostCommitError as exc:
        print(
            json.dumps(
                {
                    "status": "post_commit_error",
                    "committed": exc.committed,
                    "durable": exc.durable,
                    "current": exc.current,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    except RebuildCommitUncertainError as exc:
        print(
            json.dumps(
                {
                    "status": "commit_uncertain",
                    "committed": exc.committed,
                    "durable": exc.durable,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 4
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
