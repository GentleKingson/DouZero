"""Standalone belief-model evaluation entry point (P07).

Loads a trained belief checkpoint and reports the AGENTS.md belief metrics
(rank accuracy, exact-match, count-MAE) plus the conservation sanity check on
a freshly collected synthetic dataset. CPU-only; honest about the fact that
random-play data is not a real-strength measurement.

Example::

    python evaluate_belief.py --checkpoint /tmp/belief_smoke/belief.pt \\
        --num_episodes 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from douzero._version import git_sha
from douzero.belief.checkpoint import load_belief_checkpoint
from douzero.belief.data import collect_random_dataset
from douzero.env.rules import RuleSet


BELIEF_EVALUATION_SCHEMA_VERSION = "belief-evaluation-result-v1"


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _full_git_sha() -> str:
    value = git_sha()
    if (
        len(value) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(
            "belief evaluation requires a full source Git SHA; set "
            "DOUZERO_GIT_SHA in source-less runtimes"
        )
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluate_belief",
        description="Evaluate a P07 belief checkpoint on synthetic self-play "
                    "data (CPU).",
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to a belief checkpoint .pt")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="number of random self-play games for evaluation data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ruleset",
        choices=("legacy", "standard"),
        default="legacy",
        help="checkpoint ruleset identity and evaluation collection state machine",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ruleset = (
        RuleSet.standard() if args.ruleset == "standard" else RuleSet.legacy()
    )
    source_sha = _full_git_sha()
    model = load_belief_checkpoint(
        args.checkpoint,
        expected_ruleset=ruleset,
        map_location="cpu",
        require_full_git_sha=True,
    )
    config = model.config
    checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    print(
        f"[evaluate_belief] loaded checkpoint sha256={checkpoint_sha256[:12]}",
        file=sys.stderr,
    )
    print(f"[evaluate_belief] belief_config_hash={config.stable_hash()}",
          file=sys.stderr)

    dataset = collect_random_dataset(
        args.num_episodes,
        seed=args.seed,
        ruleset=ruleset if args.ruleset == "standard" else None,
    )
    n = len(dataset)
    print(f"[evaluate_belief] {n} evaluation samples", file=sys.stderr)
    if n == 0:
        print("[evaluate_belief] ERROR: no samples", file=sys.stderr)
        return 1

    targets = np.stack([s.label.allocation for s in dataset.samples], axis=0)
    totals = np.array(
        [s.binput.opponent_a_total for s in dataset.samples], dtype=np.int64
    )
    n_samples = len(dataset)

    # Forward in chunks to keep memory bounded on CPU. We compute BOTH:
    #   (a) the independent per-rank "factor" argmax (informational; does NOT
    #       respect the total-count constraint), and
    #   (b) the constrained DP MAP decode (the actual deployment decoder; MUST
    #       be 100% total-conservative by construction).
    # Reporting both (Medium #4) makes the conservation guarantee visible and
    # avoids presenting the unconstrained factor argmax as the model's output.

    model.eval()
    chunk = 256
    factor_argmax_preds = []
    map_preds = []
    map_conservation_ok = 0
    with torch.no_grad():
        for start in range(0, n_samples, chunk):
            sl = slice(start, start + chunk)
            inputs = [s.binput for s in dataset.samples[start:start + chunk]]
            out = model(inputs)
            lg = out.legal.cpu().numpy()
            factor_probs = out.factor_probs.cpu().numpy()
            # (a) independent per-rank argmax (restricted to legal slots).
            factor_argmax_preds.append(
                np.where(lg, factor_probs, -1.0).argmax(axis=-1)
            )
            # (b) constrained DP MAP decode (exact total constraint).
            map_alloc = model.decode_map(out)
            map_preds.append(map_alloc)
            for i in range(map_alloc.shape[0]):
                if int(map_alloc[i].sum()) == int(totals[start + i]):
                    map_conservation_ok += 1
    factor_argmax_all = np.concatenate(factor_argmax_preds, axis=0)
    map_all = np.concatenate(map_preds, axis=0)
    map_conservation_total = int(map_all.shape[0])

    # Factor-argmax metrics (independent per-rank; NOT total-conservative).
    factor_metrics = _allocation_metrics(factor_argmax_all, targets)
    # Constrained DP MAP metrics (the deployment decoder).
    map_metrics = _allocation_metrics(map_all, targets)
    map_conservation = map_conservation_ok / map_conservation_total

    print("[evaluate_belief] factor-argmax metrics (independent per-rank):",
          file=sys.stderr)
    for k, v in factor_metrics.items():
        print(f"  factor_argmax_{k}: {v:.4f}", file=sys.stderr)
    print("[evaluate_belief] constrained MAP metrics (DP decoder, deployed):",
          file=sys.stderr)
    for k, v in map_metrics.items():
        print(f"  constrained_map_{k}: {v:.4f}", file=sys.stderr)
    print(
        f"[evaluate_belief] constrained_map_conservation: "
        f"{map_conservation_ok}/{map_conservation_total} "
        f"({map_conservation:.4f}) [must be 1.0]",
        file=sys.stderr,
    )

    ruleset_identity = ruleset.identity()
    evaluation_config = {
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "ruleset": ruleset_identity,
        "feature_version": "v2",
        "belief_input_schema": "belief_input_public_v1",
        "belief_config_hash": config.stable_hash(),
        "checkpoint_sha256": checkpoint_sha256,
    }
    out = {
        "schema_version": BELIEF_EVALUATION_SCHEMA_VERSION,
        "runtime_identity": {
            "source_git_sha": source_sha,
            "evaluation_config_hash": _canonical_hash(evaluation_config),
            "ruleset": ruleset_identity,
            "feature_version": "v2",
            "belief_input_schema": "belief_input_public_v1",
            "belief_config_hash": config.stable_hash(),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "num_samples": n,
        "factor_argmax": factor_metrics,
        "constrained_map": map_metrics,
        "constrained_map_conservation": map_conservation,
    }
    print(json.dumps(out, indent=2))
    return 0


def _allocation_metrics(pred, target):
    """Rank accuracy / exact match / count MAE for a (B,15) int allocation."""
    rank_match = (pred == target)
    return {
        "rank_accuracy": float(rank_match.mean()),
        "exact_match": float(rank_match.all(axis=-1).mean()),
        "count_mae": float(np.abs(pred - target).mean()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
