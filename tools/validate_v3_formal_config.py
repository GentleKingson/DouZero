#!/usr/bin/env python3
"""Validate and print the immutable identity of one P1 formal config."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from douzero.v3_hybrid.formal_config import (
    load_formal_config,
    validate_initial_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a fail-closed DouZero V3 formal experiment config."
    )
    parser.add_argument("config", help="Path to a runnable YAML config")
    args = parser.parse_args()
    config = load_formal_config(args.config)
    validate_initial_checkpoint(config, config_path=Path(args.config).resolve())
    identity = config.identity_dict()
    print(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
