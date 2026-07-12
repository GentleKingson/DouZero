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
import sys
from typing import Sequence

import numpy as np
import torch

from douzero.belief import belief_metrics
from douzero.belief.checkpoint import load_belief_checkpoint
from douzero.belief.constraints import legal_mask
from douzero.belief.data import collect_random_dataset
from douzero.env.rules import RuleSet


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
    p.add_argument("--ruleset", default="legacy",
                   choices=["legacy", "standard"])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ruleset = RuleSet.legacy() if args.ruleset == "legacy" else RuleSet.standard()
    model = load_belief_checkpoint(
        args.checkpoint, expected_ruleset=ruleset, map_location="cpu"
    )
    config = model.config
    print(f"[evaluate_belief] loaded {args.checkpoint}", file=sys.stderr)
    print(f"[evaluate_belief] belief_config_hash={config.stable_hash()}",
          file=sys.stderr)

    dataset = collect_random_dataset(args.num_episodes, seed=args.seed)
    n = len(dataset)
    print(f"[evaluate_belief] {n} evaluation samples", file=sys.stderr)
    if n == 0:
        print("[evaluate_belief] ERROR: no samples", file=sys.stderr)
        return 1

    feats = torch.from_numpy(dataset.feature_matrix().astype(np.float32))
    legal = dataset.legal_mask_tensor().numpy()
    targets = np.stack([s.label.allocation for s in dataset.samples], axis=0)

    # Forward in chunks to keep memory bounded on CPU.
    model.eval()
    chunk = 512
    all_probs = []
    conservation_ok = 0
    conservation_total = 0
    with torch.no_grad():
        for start in range(0, feats.shape[0], chunk):
            sl = slice(start, start + chunk)
            logits = model._forward_logits(feats[sl]).numpy()
            lg = legal[sl]
            masked = np.where(lg, logits, -1e30)
            probs = torch.softmax(
                torch.from_numpy(masked), dim=-1
            ).numpy()
            all_probs.append(probs)
            # Conservation check: argmax-restricted MAP decodes must sum to the
            # per-sample opponent-A total. We check the per-rank cap here; the
            # exact-total DP is covered by the test suite.
            pred = np.where(lg, probs, -1.0).argmax(axis=-1)
            for i in range(pred.shape[0]):
                if int(pred[i].sum()) == int(
                    dataset.samples[start + i].binput.opponent_a_total
                ):
                    conservation_ok += 1
                conservation_total += 1
    probs_all = np.concatenate(all_probs, axis=0)

    metrics = belief_metrics(probs_all, targets, legal)
    print("[evaluate_belief] metrics:", file=sys.stderr)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}", file=sys.stderr)
    print(
        f"[evaluate_belief] argmax-total conservation: "
        f"{conservation_ok}/{conservation_total}",
        file=sys.stderr,
    )

    # Machine-readable JSON to stdout for logging.
    import json
    out = {
        "checkpoint": args.checkpoint,
        "num_samples": n,
        "belief_config_hash": config.stable_hash(),
        **metrics,
        "argmax_total_conservation": conservation_ok / conservation_total,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
