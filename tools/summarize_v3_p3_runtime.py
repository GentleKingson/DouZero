#!/usr/bin/env python3
"""Validate P3 matched records and recompute the H7.1 runtime decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if not sys.path or sys.path[0] != str(_ROOT):
    sys.path.insert(0, str(_ROOT))

from douzero.v3_hybrid.runtime_decision import (
    P3_RUNTIME_SCHEMA,
    P3RuntimeProtocol,
    summarize_p3_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    if payload.pop("schema", None) != P3_RUNTIME_SCHEMA:
        raise ValueError("P3 runtime protocol schema mismatch")
    payload["seeds"] = tuple(payload["seeds"])
    protocol = P3RuntimeProtocol(**payload)
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_p3_decision(records, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
