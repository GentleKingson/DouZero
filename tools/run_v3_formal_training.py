#!/usr/bin/env python3
"""Resolve one frozen P4 training run to its strict family-owned CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from douzero.v3_hybrid.formal_config import load_formal_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument(
        "--tier",
        choices=("development", "promotion"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--elapsed-wall-seconds",
        type=float,
        default=0.0,
        help="Legacy-only wall time already consumed before fresh-container resume.",
    )
    parser.add_argument("--print-command", action="store_true")
    return parser


def _v3_command(formal_path, formal, tier, seed, output, resume):
    command = [
        sys.executable,
        str(ROOT / "train_v3_h7.py"),
        "--formal-config",
        str(formal_path),
        "--formal-budget-tier",
        tier,
        "--seed",
        str(seed),
        "--checkpoint-path",
        str(output / "training"),
        "--checkpoint-every-cycles",
        "0",
    ]
    if resume:
        command.extend([
            "--resume",
            str(output / "training-latest.json"),
        ])
    return command


def _v2_command(formal, tier, seed, output, resume):
    budget = formal.budgets[tier]
    config = (
        ROOT / "configs/standard_v2.yaml"
        if formal.ruleset["id"] == "standard"
        else ROOT / "configs/enhanced.yaml"
    )
    command = [
        sys.executable,
        str(ROOT / "train_v2.py"),
        "--config",
        str(config),
        "--long_running",
        "--device",
        formal.runtime.device,
        "--seed",
        str(seed),
        "--batch_size",
        str(formal.runtime.batch_size),
        "--buffer_capacity",
        str(formal.runtime.replay_capacity),
        "--v2_training_mode",
        formal.runtime.topology,
        "--num_actors",
        str(formal.runtime.num_actors),
        "--games_per_actor",
        str(formal.runtime.games_per_actor),
        "--episodes_per_cycle",
        str(formal.runtime.episodes_per_cycle),
        "--optimizer_steps_per_cycle",
        str(formal.runtime.optimizer_steps_per_cycle),
        "--max_total_optimizer_steps",
        str(budget.optimizer_step_budget),
        "--max_wall_time_minutes",
        str(budget.wall_clock_seconds / 60.0),
        "--checkpoint_every_cycles",
        "0",
        "--checkpoint_every_steps",
        str(formal.runtime.checkpoint_cadence_updates),
        "--checkpoint_path",
        str(output / "training.pt"),
        "--metrics_path",
        str(output / "metrics.json"),
    ]
    if resume:
        command.extend([
            "--resume_checkpoint",
            str(output / "training-latest.json"),
        ])
    return command


def _legacy_command(formal, tier, seed, output, resume, elapsed_wall_seconds):
    budget = formal.budgets[tier]
    if elapsed_wall_seconds < 0.0 or elapsed_wall_seconds >= budget.wall_clock_seconds:
        raise ValueError(
            "legacy elapsed wall time must be within the frozen wall ceiling"
        )
    remaining_minutes = (
        budget.wall_clock_seconds - elapsed_wall_seconds
    ) / 60.0
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(ROOT / "configs/legacy_single_gpu_a1.yaml"),
        "--xpid",
        "run",
        "--savedir",
        str(output),
        "--seed",
        str(seed),
        "--total_frames",
        str(budget.sample_budget),
        "--batch_size",
        str(formal.runtime.batch_size),
        "--unroll_length",
        str(formal.runtime.legacy_unroll_length),
        "--num_actors",
        str(formal.runtime.num_actors),
        "--checkpoint_every_updates",
        str(formal.runtime.checkpoint_cadence_updates),
        "--max_wall_time_minutes",
        str(remaining_minutes),
        "--legacy_metrics_path",
        str(output / "metrics.json"),
    ]
    if resume:
        command.append("--load_model")
    return command


def build_formal_training_command(
    formal_path: Path,
    *,
    tier: str,
    seed: int,
    output: Path,
    resume: bool,
    elapsed_wall_seconds: float = 0.0,
) -> tuple[list[str], dict[str, object]]:
    formal_path = formal_path.resolve()
    formal = load_formal_config(formal_path)
    budget = formal.budgets[tier]
    allowed_seeds = formal.seeds.training[:budget.training_seed_count]
    if seed not in allowed_seeds:
        raise ValueError(
            f"seed {seed} is not frozen for the {tier} budget"
        )
    if formal.variant.startswith("v3_"):
        if elapsed_wall_seconds != 0.0:
            raise ValueError(
                "V3 cumulative wall time is restored from its checkpoint"
            )
        command = _v3_command(
            formal_path, formal, tier, seed, output, resume
        )
    elif formal.variant == "model_v2":
        if elapsed_wall_seconds != 0.0:
            raise ValueError(
                "V2 cumulative wall time is restored from its checkpoint"
            )
        command = _v2_command(formal, tier, seed, output, resume)
    elif formal.variant == "legacy_a1":
        command = _legacy_command(
            formal,
            tier,
            seed,
            output,
            resume,
            elapsed_wall_seconds,
        )
    else:
        raise ValueError("formal training variant is unsupported")
    record = {
        "schema_version": "v3-p4-training-command-v1",
        "formal_config": str(formal_path.relative_to(ROOT)),
        "formal_config_sha256": formal.identity_dict()["config_sha256"],
        "training_semantics_hash": (
            formal.identity_dict()["training_semantics_hash"]
        ),
        "workload_hash": formal.identity_dict()["workload_hash"],
        "variant": formal.variant,
        "ruleset": formal.ruleset["id"],
        "runtime_profile": formal.runtime.profile,
        "tier": tier,
        "seed": seed,
        "resume": resume,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "command": command,
        "release_candidate": "NONE",
        "release_status": "NOT READY",
        "playing_strength": "NOT MEASURED",
    }
    return command, record


def main() -> int:
    args = _parser().parse_args()
    command, record = build_formal_training_command(
        args.formal_config,
        tier=args.tier,
        seed=args.seed,
        output=args.output_dir.resolve(),
        resume=args.resume,
        elapsed_wall_seconds=args.elapsed_wall_seconds,
    )
    if args.print_command:
        print(json.dumps(record, sort_keys=True, indent=2, allow_nan=False))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = "resume" if args.resume else "initial"
    record_path = args.output_dir / f"command-{phase}.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite {record_path}")
    record_path.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
