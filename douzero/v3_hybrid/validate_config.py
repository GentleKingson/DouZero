"""Validate and print the resolved H6 configuration identity."""

from __future__ import annotations

import argparse
import json

from .integration_config import load_v3_hybrid_config
from .support_matrix import v3_h6_support_matrix_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a fail-closed DouZero V3 Hybrid H6 YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to the H6 YAML file.")
    parser.add_argument(
        "--show-support-matrix",
        action="store_true",
        help="Include the machine-readable support matrix in the JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = load_v3_hybrid_config(args.config)
    output = {
        "format": "v3-hybrid-h6-config-validation-v1",
        "config_hash": resolved.stable_hash(),
        "model_hash": resolved.model.stable_hash(),
        "learner_hash": resolved.learner.stable_hash(),
        "playing_strength": "not measured",
    }
    if args.show_support_matrix:
        output["support_matrix"] = v3_h6_support_matrix_dict()
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
