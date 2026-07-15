#!/usr/bin/env python3
"""Build a strict P16 release package from a public Model V2 sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.deployment import create_model_package
from douzero.env.rules import RuleSet
from douzero.evaluation.deep_agent import load_v2_model
from douzero.models_v2 import ModelV2Config
from douzero.observation import build_v2_schema


def _json(path: str | None) -> dict:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ruleset", choices=("legacy", "standard"), default="legacy")
    parser.add_argument("--model-config", help="JSON ModelV2Config object")
    parser.add_argument("--training-config", help="JSON training config for audit hash")
    parser.add_argument("--model-card", help="reviewed Markdown model card")
    parser.add_argument("--max-history-len", type=int, default=100)
    parser.add_argument("--search-compatible", action="store_true")
    args = parser.parse_args()

    ruleset = RuleSet.standard() if args.ruleset == "standard" else RuleSet.legacy()
    config = ModelV2Config(**_json(args.model_config))
    schema = build_v2_schema(max_history_len=args.max_history_len)
    model = load_v2_model(args.checkpoint, schema, ruleset, config, device="cpu")
    card = (
        Path(args.model_card).read_text(encoding="utf-8")
        if args.model_card
        else None
    )
    manifest = create_model_package(
        args.output,
        model,
        ruleset,
        training_config=_json(args.training_config),
        search_compatible=args.search_compatible,
        model_card=card,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
