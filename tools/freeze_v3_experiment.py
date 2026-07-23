#!/usr/bin/env python3
"""Freeze a P1 config into canonical resolved-config and identity JSON."""

from __future__ import annotations

import argparse
import json

from douzero.v3_hybrid.formal_config import freeze_formal_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a validated DouZero V3 formal experiment identity."
    )
    parser.add_argument("config", help="Path to a runnable YAML config")
    parser.add_argument("output_dir", help="Empty or existing output directory")
    args = parser.parse_args()
    identity = freeze_formal_config(args.config, args.output_dir)
    print(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
