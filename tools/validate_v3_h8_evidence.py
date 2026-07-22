#!/usr/bin/env python3
"""Validate V3 Hybrid H8 evidence and print a recomputed release decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.v3_hybrid.formal_evidence import validate_h8_formal_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", help="H8 canonical JSON evidence bundle")
    parser.add_argument("--output", help="optional path for the validated report")
    args = parser.parse_args()
    payload = json.loads(
        Path(args.evidence).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value!r}")
        ),
    )
    report = validate_h8_formal_evidence(payload)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["release_status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
