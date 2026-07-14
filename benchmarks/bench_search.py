"""CPU latency benchmark for the P13 exact endgame solver."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from douzero.env.rules import RuleSet
from douzero.search import EndgameSolver, SearchBudget, SearchConfig, SearchGameState


def _state() -> SearchGameState:
    return SearchGameState(
        hands={
            "landlord": (3, 3, 4),
            "landlord_down": (5, 6),
            "landlord_up": (7, 8),
        },
        acting_role="landlord",
        last_move=(),
        last_non_pass_role=None,
        consecutive_passes=0,
        ruleset=RuleSet.legacy(),
    )


def benchmark(iterations: int) -> dict[str, float | int]:
    """Return measured p50/p95 milliseconds and mean expanded nodes."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    latencies: list[float] = []
    nodes: list[int] = []
    for _ in range(iterations):
        config = SearchConfig(
            enabled=True,
            max_nodes=100_000,
            max_rollouts=1,
            max_milliseconds=10_000,
        )
        budget = SearchBudget(config)
        started = time.perf_counter()
        EndgameSolver(budget).solve(_state(), "landlord")
        latencies.append((time.perf_counter() - started) * 1000.0)
        nodes.append(budget.nodes)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "iterations": iterations,
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "mean_nodes": statistics.mean(nodes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args(argv)
    result = benchmark(args.iterations)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
