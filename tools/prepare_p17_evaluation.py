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
    cardplay = (
        load_result(args.cardplay_result, "cardplay_only")
        if args.cardplay_result else None
    )
    full_game = (
        load_result(args.full_game_result, "full_game")
        if args.full_game_result else None
    )
    ablations = {
        name: load_result(path, "cardplay_only" if name == "no_bidding" else "full_game")
        for name, path in args.ablation_result
    }
    paths = write_p17_artifacts(
        args.output,
        matrix=matrix,
        cardplay_result=cardplay,
        full_game_result=full_game,
        ablation_results=ablations,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
