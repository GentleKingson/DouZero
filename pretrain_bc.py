"""Pretrain the listwise policy-prior head on validated human games (P08).

Trains a :class:`~douzero.models_v2.model.ModelV2` (built with
``human_prior_enabled=True``) on validated human-game BC samples using the
listwise cross-entropy over the legal-action list, then saves a
manifest-bearing V2 checkpoint. CPU-only by design (this is the BC pretrain /
smoke path; high-throughput training is P14).

Two data modes:

1. ``--data <validated.jsonl>``: read canonical records (run
   ``validate_human_games.py`` first to produce this file).
2. ``--synthetic``: generate deterministic random-self-play records (smoke /
   CI path when no ``<HUMAN_DATA_PATH>`` exists).

Example (CPU smoke)::

    python pretrain_bc.py --synthetic --num_synthetic 8 \\
        --save_dir /tmp/bc_smoke --epochs 3 --batch_size 8
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import torch

from douzero.env.rules import RuleSet
from douzero.human_data.sample import build_bc_samples
from douzero.human_data.schema import read_jsonl
from douzero.human_data.synthetic import generate_synthetic_records
from douzero.human_data.validate import validate_record
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training.bc_trainer import BCTrainer, BCTrainerConfig, BCTrainerError


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pretrain_bc",
        description="Pretrain the V2 listwise policy-prior head on validated "
                    "human games (CPU smoke).",
    )
    p.add_argument("--data", default="",
                   help="validated canonical JSONL of human games (omitted with "
                        "--synthetic)")
    p.add_argument("--save_dir", default="/tmp/bc_smoke",
                   help="output directory for the checkpoint")
    p.add_argument("--save_name", default="bc_prior.pt",
                   help="checkpoint filename")
    p.add_argument("--synthetic", action="store_true",
                   help="generate deterministic synthetic records (no real data)")
    p.add_argument("--num_synthetic", type=int, default=8)
    p.add_argument("--synthetic_seed", type=int, default=0)
    # Trainer hyperparameters.
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=40.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--skill_weight_clip", type=float, default=10.0,
                   help="cap on the composite sample weight before normalization")
    p.add_argument("--seed", type=int, default=0)
    # Model architecture (small defaults so CPU smoke runs quickly).
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--history_layers", type=int, default=1)
    p.add_argument("--history_heads", type=int, default=4)
    p.add_argument("--history_encoder", default="lstm",
                   choices=["lstm", "transformer"])
    p.add_argument("--skip_validation", action="store_true",
                   help="skip replay-validation of --data (use only when the "
                        "input is already known-valid; otherwise invalid games "
                        "would silently drop during BC sampling)")
    return p.parse_args(argv)


def _load_records(args: argparse.Namespace):
    if args.synthetic:
        yield from generate_synthetic_records(
            num_games=args.num_synthetic, base_seed=args.synthetic_seed
        )
        return
    if not args.data:
        raise SystemExit("either --synthetic or --data is required")
    yield from read_jsonl(args.data)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Seed (seed=0 -> no-op per the project convention).
    if args.seed != 0:
        torch.manual_seed(args.seed)

    # 1. Load + validate records, then build BC samples.
    # Blocker 3: records that fail replay validation are QUARANTINED to a
    # structured file (never silently dropped). The BC sample builder is
    # fail-closed per-record: a record that does not replay raises rather than
    # silently yielding no samples, so we catch and quarantine here.
    import json as _json

    print(f"[pretrain_bc] loading records...", file=sys.stderr)
    samples = []
    n_records = 0
    n_valid = 0
    quarantined: list[str] = []
    for record in _load_records(args):
        n_records += 1
        if not args.skip_validation and not args.synthetic:
            result = validate_record(record)
            if not result.ok:
                quarantined.append(
                    _json.dumps(
                        {
                            "game_id": record.game_id,
                            "stage": "replay_validation",
                            "reason": result.reason,
                            "error": result.error,
                        },
                        sort_keys=True, ensure_ascii=False,
                    )
                )
                continue
        n_valid += 1
        try:
            samples.extend(build_bc_samples(record))
        except Exception as exc:  # BCSampleError or ReplayValidationError
            quarantined.append(
                _json.dumps(
                    {
                        "game_id": record.game_id,
                        "stage": "bc_sampling",
                        "reason": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True, ensure_ascii=False,
                )
            )
    # Write the quarantine file alongside the checkpoint so invalid records
    # are auditable (Blocker 3: no silent drops in the production path).
    if quarantined:
        q_path = os.path.join(args.save_dir, "quarantine.jsonl")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(q_path, "w", encoding="utf-8") as fh:
            for line in quarantined:
                fh.write(line)
                fh.write("\n")
        print(
            f"[pretrain_bc] quarantined {len(quarantined)} records -> {q_path}",
            file=sys.stderr,
        )
    print(
        f"[pretrain_bc] records={n_records} valid={n_valid} bc_samples="
        f"{len(samples)} quarantined={len(quarantined)}",
        file=sys.stderr,
    )
    if not samples:
        print("[pretrain_bc] ERROR: no BC samples (provide valid data).",
              file=sys.stderr)
        return 1

    # 1b. Compute composite sample weights (Blocker 4): clip + mean-normalize
    # the raw skill weights so a single high-skill outlier cannot dominate a
    # minibatch, then stamp them onto each sample's `sample_weight`.
    from douzero.human_data.weights import (
        WeightConfig,
        apply_sample_weights,
        stratified_stats,
    )

    samples = apply_sample_weights(
        samples, config=WeightConfig(skill_weight_clip=args.skill_weight_clip)
    )
    stats_summary = stratified_stats(samples)
    print(
        f"[pretrain_bc] sample-weight stats: total={stats_summary['total']} "
        f"by_position={stats_summary['by_position']} "
        f"by_winner_team={stats_summary['by_winner_team']}",
        file=sys.stderr,
    )

    # 2. Build the model with the prior head enabled.
    model_cfg = ModelV2Config(
        hidden_size=args.hidden_size,
        history_layers=args.history_layers,
        history_heads=args.history_heads,
        history_encoder=args.history_encoder,
        human_prior_enabled=True,
        nan_guard=False,
    )
    schema = build_v2_schema()
    model = ModelV2(schema, model_cfg)

    # 3. Train. Wrap config construction + training so a bad config (e.g.
    # epochs=0, label_smoothing out of range) returns rc=1 without saving a
    # checkpoint (Blocker: --epochs 0 must not produce a checkpoint).
    try:
        trainer_cfg = BCTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            early_stopping_patience=args.early_stopping_patience,
            max_grad_norm=args.max_grad_norm,
            temperature=args.temperature,
            seed=args.seed,
        )
        trainer = BCTrainer(model, samples, trainer_cfg)
        stats = trainer.train()
    except (BCTrainerError, ValueError) as exc:
        print(f"[pretrain_bc] ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 1
    for e in stats.epoch_stats:
        print(
            f"[pretrain_bc] epoch {e.epoch}: "
            f"train_loss={e.train_loss:.4f} train_top1={e.train_top1:.3f} "
            f"val_loss={e.val_loss:.4f} val_top1={e.val_top1:.3f}",
            file=sys.stderr,
        )
    print(
        f"[pretrain_bc] done: epochs_run={stats.epochs_run} "
        f"best_val_loss={stats.best_val_loss:.4f} "
        f"final_val_top1={stats.final_val_top1:.3f} "
        f"stopped_early={stats.stopped_early}",
        file=sys.stderr,
    )

    # 4. Save a manifest-bearing V2 checkpoint. The prior-head weights are
    # part of the ModelV2 state_dict; the manifest records human_prior_enabled
    # via model_config.compatibility_dict().
    from douzero.checkpoint.v2 import save_v2_checkpoint

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.save_name)
    # The synthetic smoke path uses the legacy ruleset (the only collection
    # mode in P08); never mislabel it as standard.
    ruleset = RuleSet.legacy()
    manifest = save_v2_checkpoint(
        out_path,
        model,
        ruleset=ruleset,
        model_config=model_cfg,
        frames=sum(e.train_num_decisions for e in stats.epoch_stats),
        config_dict={
            "pretrain": "bc",
            "bc_samples": len(samples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "seed": args.seed,
            "synthetic": bool(args.synthetic),
        },
    )
    print(f"[pretrain_bc] saved checkpoint to {out_path}", file=sys.stderr)
    print(
        f"[pretrain_bc] model_config_hash={model_cfg.stable_hash()}",
        file=sys.stderr,
    )
    print(f"[pretrain_bc] manifest_schema={manifest.schema_version}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
