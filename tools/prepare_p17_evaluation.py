#!/usr/bin/env python3
"""Validate and collate P17 evaluation inputs into the fixed artifact layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.evaluation.p17 import (
    ABLATION_NAMES,
    empty_matrix,
    load_result,
    normalize_matrix,
    write_p17_artifacts,
)


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ABLATION=/path/to/result.json")
    name, path = value.split("=", 1)
    if name not in ABLATION_NAMES or not path:
        raise argparse.ArgumentTypeError("unknown ablation name or empty path")
    return name, path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", help="P17 model matrix JSON")
    parser.add_argument("--write-matrix-template", help="write an unavailable template and exit")
    parser.add_argument("--cardplay-result", help="P15 cardplay_only result JSON")
    parser.add_argument("--full-game-result", help="P15 full_game result JSON")
    parser.add_argument(
        "--ablation-result", action="append", type=_assignment, default=[]
    )
    parser.add_argument(
        "--expected-evaluator-git-sha",
        action="append",
        default=[],
        help=(
            "approved full evaluator Git SHA; repeat for an explicit cross-version "
            "allowlist"
        ),
    )
    parser.add_argument(
        "--expected-cardplay-deal-set-id",
        help="pre-approved cardplay_only deal-set SHA-256",
    )
    parser.add_argument(
        "--expected-full-game-deal-set-id",
        help="pre-approved full_game deal-set SHA-256",
    )
    parser.add_argument("--output", default="artifacts/evaluation/p17")
    args = parser.parse_args(argv)

    if args.write_matrix_template:
        path = Path(args.write_matrix_template)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(empty_matrix(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if not args.matrix:
        parser.error("--matrix is required unless --write-matrix-template is used")
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    normalized_matrix = normalize_matrix(matrix)
    ablation_protocols = {
        name: normalized_matrix["ablations"][name]["protocol"]
        for name, _path in args.ablation_result
    }
    if (
        args.cardplay_result
        or args.full_game_result
        or args.ablation_result
    ) and not args.expected_evaluator_git_sha:
        parser.error(
            "--expected-evaluator-git-sha is required when collating results"
        )
    needs_cardplay_set = bool(args.cardplay_result) or any(
        ablation_protocols[name] == "cardplay_only"
        for name, _path in args.ablation_result
    )
    needs_full_game_set = bool(args.full_game_result) or any(
        ablation_protocols[name] == "full_game"
        for name, _path in args.ablation_result
    )
    if needs_cardplay_set and not args.expected_cardplay_deal_set_id:
        parser.error(
            "--expected-cardplay-deal-set-id is required for cardplay results"
        )
    if needs_full_game_set and not args.expected_full_game_deal_set_id:
        parser.error(
            "--expected-full-game-deal-set-id is required for full-game results"
        )
    cardplay = (
        load_result(args.cardplay_result, "cardplay_only")
        if args.cardplay_result else None
    )
    full_game = (
        load_result(args.full_game_result, "full_game")
        if args.full_game_result else None
    )
    ablations = {
        name: load_result(path, ablation_protocols[name])
        for name, path in args.ablation_result
    }
    paths = write_p17_artifacts(
        args.output,
        matrix=matrix,
        cardplay_result=cardplay,
        full_game_result=full_game,
        ablation_results=ablations,
        expected_evaluator_git_shas=args.expected_evaluator_git_sha,
        expected_cardplay_deal_set_id=args.expected_cardplay_deal_set_id,
        expected_full_game_deal_set_id=args.expected_full_game_deal_set_id,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
