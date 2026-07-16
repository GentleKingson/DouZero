#!/usr/bin/env python3
"""Create a private, digest-verified checkpoint snapshot model matrix."""

from __future__ import annotations

import argparse
import json

from douzero.evaluation.checkpoint_inputs import (
    CheckpointIdentityError,
    snapshot_model_matrix_file,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="approved input matrix JSON")
    parser.add_argument(
        "--kind",
        choices=("auto", "evaluator", "p17"),
        default="auto",
        help="matrix schema to validate",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True, help="new rewritten matrix JSON")
    args = parser.parse_args(argv)
    try:
        output = snapshot_model_matrix_file(
            args.matrix,
            args.output,
            args.checkpoint_dir,
            kind=args.kind,
        )
    except CheckpointIdentityError as exc:
        parser.error(str(exc))
    print(json.dumps({"snapshot_matrix": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
