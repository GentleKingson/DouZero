"""Validate human-game records by replaying them through the rule engine (P08).

Reads canonical JSONL, replays every record through ``GameEnv`` to enforce
legality / turn order / hand conservation / terminal consistency, and writes:

- ``<output>.jsonl``  — the valid records (canonical, ready for BC sampling).
- ``<output>.quarantine.jsonl`` — the invalid records, each prefixed with a
  diagnostic line carrying the failure reason.

Invalid records are NEVER silently dropped or repaired (AGENTS.md).

No real human data is required: pass ``--synthetic`` to generate deterministic
synthetic records and validate them end-to-end (smoke / CI path).

Example (CPU smoke)::

    python validate_human_games.py --input /tmp/games.jsonl --output /tmp/valid
    python validate_human_games.py --synthetic --num_synthetic 8 \
        --output /tmp/valid
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from douzero.human_data.schema import (
    HumanGameRecord,
    iter_jsonl_resilient,
    write_jsonl,
)
from douzero.human_data.synthetic import generate_synthetic_records
from douzero.human_data.validate import (
    ReplayValidationError,
    validate_record,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_human_games",
        description="Replay-validate human-game records through the rule "
                    "engine; quarantine invalid games.",
    )
    p.add_argument(
        "--input", default="",
        help="canonical JSONL input path (omitted when --synthetic is used)",
    )
    p.add_argument(
        "--output", required=True,
        help="output basename; writes <output>.jsonl and "
             "<output>.quarantine.jsonl",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="generate deterministic synthetic records instead of reading input "
             "(smoke / CI path; no real data)",
    )
    p.add_argument("--num_synthetic", type=int, default=8)
    p.add_argument("--synthetic_seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    valid: list[HumanGameRecord] = []
    quarantined: list[str] = []  # JSONL lines: {"reason":..., "record"/"line":...}
    total = 0
    n_parse_errors = 0

    if args.synthetic:
        for record in generate_synthetic_records(
            num_games=args.num_synthetic, base_seed=args.synthetic_seed
        ):
            total += 1
            result = validate_record(record)
            if result.ok:
                valid.append(record)
            else:
                quarantined.append(
                    json.dumps(
                        {
                            "game_id": record.game_id,
                            "reason": result.reason,
                            "error": result.error,
                            "record": record.to_dict(),
                        },
                        sort_keys=True, ensure_ascii=False,
                    )
                )
    else:
        if not args.input:
            raise SystemExit("--input is required when --synthetic is not set")
        # Resilient read (Blocker 3): a malformed line (invalid JSON / wrong
        # schema version / missing field / bad type) is QUARANTINED, not fatal.
        # This is the only way JSON/schema errors reach the quarantine file
        # alongside replay failures.
        for line_result in iter_jsonl_resilient(args.input):
            total += 1
            if line_result.error:
                n_parse_errors += 1
                quarantined.append(
                    json.dumps(
                        {
                            "lineno": line_result.lineno,
                            "reason": "parse_error",
                            "error": line_result.error,
                        },
                        sort_keys=True, ensure_ascii=False,
                    )
                )
                continue
            record = line_result.record
            assert record is not None  # guaranteed when error is empty
            result = validate_record(record)
            if result.ok:
                valid.append(record)
            else:
                quarantined.append(
                    json.dumps(
                        {
                            "game_id": record.game_id,
                            "reason": result.reason,
                            "error": result.error,
                            "record": record.to_dict(),
                        },
                        sort_keys=True, ensure_ascii=False,
                    )
                )

    valid_path = args.output + ".jsonl"
    quarantine_path = args.output + ".quarantine.jsonl"
    n_valid = write_jsonl(valid, valid_path)
    with open(quarantine_path, "w", encoding="utf-8") as fh:
        for line in quarantined:
            fh.write(line)
            fh.write("\n")

    print(
        f"[validate_human_games] total={total} valid={n_valid} "
        f"quarantined={len(quarantined)} parse_errors={n_parse_errors}",
        file=sys.stderr,
    )
    print(f"[validate_human_games] valid -> {valid_path}", file=sys.stderr)
    print(f"[validate_human_games] quarantine -> {quarantine_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
