"""Standalone belief-model training entry point (P07).

Trains a :class:`~douzero.belief.model.BeliefModel` on synthetic self-play
data with the masked cross-entropy loss, then saves a manifest-bearing
checkpoint. CPU-only by design; this is a smoke / pretraining path, not the
high-throughput trainer (which is P14).

Example (CPU smoke)::

    python train_belief.py --save_dir /tmp/belief_smoke --num_episodes 20 \\
        --epochs 3 --batch_size 32 --learning_rate 1e-3

The script intentionally has NO dependency on the value model, the privileged
type ever crossing into a deployment ``act()`` path, or downloaded weights.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import Sequence

import numpy as np
import torch

from douzero.belief import (
    BeliefConfig,
    BeliefModel,
    belief_loss,
)
from douzero.belief.checkpoint import save_belief_checkpoint
from douzero.belief.data import (
    BeliefDataset,
    collect_random_dataset,
    iterate_minibatches,
)
from douzero.env.rules import RuleSet


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_belief",
        description="Train the P07 joint hidden-hand belief model on "
                    "synthetic self-play data (CPU smoke).",
    )
    p.add_argument("--save_dir", default="/tmp/belief_smoke",
                   help="output directory for the checkpoint")
    p.add_argument("--save_name", default="belief.pt",
                   help="checkpoint filename")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="number of random self-play games to collect")
    p.add_argument("--epochs", type=int, default=3,
                   help="training epochs over the collected dataset")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--style_enabled", action="store_true",
                   help="condition the belief model on public P11 action style")
    p.add_argument("--style_embedding_dim", type=int, default=32)
    p.add_argument("--lambda_count_reg", type=float, default=0.0)
    p.add_argument("--lambda_entropy_reg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ruleset",
        choices=("legacy", "standard"),
        default="legacy",
        help="ruleset identity and synthetic collection state machine",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = BeliefConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        style_enabled=args.style_enabled,
        style_embedding_dim=args.style_embedding_dim,
    )
    model = BeliefModel(config)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-5
    )

    ruleset = RuleSet.standard() if args.ruleset == "standard" else RuleSet.legacy()
    print(
        f"[train_belief] collecting {args.num_episodes} random {args.ruleset} episodes...",
        file=sys.stderr,
    )
    dataset = collect_random_dataset(
        args.num_episodes,
        seed=args.seed,
        ruleset=ruleset if args.ruleset == "standard" else None,
    )
    n = len(dataset)
    print(f"[train_belief] collected {n} labelled samples", file=sys.stderr)
    if n == 0:
        print("[train_belief] ERROR: no samples collected", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    frames_seen = 0
    for epoch in range(args.epochs):
        batches = iterate_minibatches(
            dataset,
            args.batch_size,
            shuffle=True,
            rng=rng,
            include_style=args.style_enabled,
        )
        epoch_loss = 0.0
        epoch_ce = 0.0
        nb = 0
        for batch in batches:
            if args.style_enabled:
                feats, targets, legal, style_features = batch
            else:
                feats, targets, legal = batch
                style_features = None
            optimizer.zero_grad()
            logits = model._forward_logits(feats, style_features)
            comps = belief_loss(
                logits, targets, legal,
                lambda_count_reg=args.lambda_count_reg,
                lambda_entropy_reg=args.lambda_entropy_reg,
            )
            if not torch.isfinite(comps.total):
                raise FloatingPointError(
                    "non-finite belief loss; aborting (check inputs/weights)."
                )
            comps.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=40.0, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_loss += float(comps.total.detach().float().item())
            epoch_ce += comps.cross_entropy
            nb += 1
            frames_seen += int(feats.shape[0])
        avg = epoch_loss / max(1, nb)
        avg_ce = epoch_ce / max(1, nb)
        print(
            f"[train_belief] epoch {epoch + 1}/{args.epochs} "
            f"loss={avg:.4f} ce={avg_ce:.4f} batches={nb}",
            file=sys.stderr,
        )

    import os
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.save_name)
    save_belief_checkpoint(
        out_path, model, ruleset=ruleset,
        feature_version="v2", frames=frames_seen,
        extra_config={
            "num_episodes": args.num_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lambda_count_reg": args.lambda_count_reg,
            "lambda_entropy_reg": args.lambda_entropy_reg,
            "seed": args.seed,
            "style_enabled": args.style_enabled,
            "style_embedding_dim": args.style_embedding_dim,
            "ruleset": args.ruleset,
        },
    )
    print(f"[train_belief] saved checkpoint to {out_path}", file=sys.stderr)
    print(f"[train_belief] belief_config_hash={config.stable_hash()}",
          file=sys.stderr)
    print(f"[train_belief] frames_seen={frames_seen}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
