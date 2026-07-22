#!/usr/bin/env python3
"""Create a strict public-only DouZero V3 Hybrid model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.belief.model import BeliefConfig
from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema
from douzero.v3_hybrid import (
    BELIEF_FEEDBACK_NONE,
    V3HybridModelConfig,
    create_v3_public_model_package,
)


def _json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--decision-config", required=True)
    parser.add_argument("--belief-config")
    parser.add_argument("--formal-evidence")
    parser.add_argument("--ruleset", choices=("legacy", "standard"), default="legacy")
    parser.add_argument("--max-history-len", type=int, default=100)
    parser.add_argument("--search-compatible", action="store_true")
    args = parser.parse_args()

    model_config = V3HybridModelConfig.from_dict(_json(args.model_config))
    belief_config = BeliefConfig(**_json(args.belief_config)) if args.belief_config else None
    if model_config.belief_feedback != BELIEF_FEEDBACK_NONE and belief_config is None:
        parser.error("--belief-config is required by this model configuration")
    ruleset = RuleSet.standard() if args.ruleset == "standard" else RuleSet.legacy()
    manifest = create_v3_public_model_package(
        args.output,
        args.checkpoint,
        schema=build_v2_schema(max_history_len=args.max_history_len),
        ruleset=ruleset,
        model_config=model_config,
        belief_config=belief_config,
        source_git_sha=args.source_git_sha,
        decision_config=_json(args.decision_config),
        search_compatible=args.search_compatible,
        formal_evidence=_json(args.formal_evidence) if args.formal_evidence else None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
