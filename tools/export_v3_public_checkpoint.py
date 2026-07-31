#!/usr/bin/env python3
"""Export a strict public V3 sidecar from a strict H7 training checkpoint."""

from __future__ import annotations

import argparse

from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.public_export import export_h7_public_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_h7_public_checkpoint(
        args.training_checkpoint,
        args.output,
        formal_config=load_formal_config(args.formal_config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
