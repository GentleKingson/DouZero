"""Validate frozen H7 topology evidence without trusting derived claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.v3_hybrid.benchmark import (
    V3H7BenchmarkProtocol,
    validate_h7_benchmark_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    args = parser.parse_args()
    protocol_payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol_payload.pop("schema", None) != "v3-hybrid-h7-benchmark-v1":
        raise ValueError("H7 benchmark protocol schema mismatch")
    protocol_payload["seeds"] = tuple(protocol_payload["seeds"])
    protocol = V3H7BenchmarkProtocol(**protocol_payload)
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_h7_benchmark_evidence(records, protocol)
    print(json.dumps({
        "protocol_hash": protocol.stable_hash(),
        "records": len(records),
        "status": "valid",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
