#!/usr/bin/env python3
"""Rebuild canonical human-game JSONL excluding complete opaque game IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.human_data.rebuild import rebuild_without_game_ids


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="validated canonical JSONL")
    parser.add_argument("--output", required=True, help="new canonical JSONL")
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
    report = rebuild_without_game_ids(args.input, args.output, excluded)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
