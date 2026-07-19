#!/usr/bin/env python
"""Repeatable end-to-end V1 single-GPU training benchmark orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "legacy_a0_cpu_actor.yaml",
    "legacy_b0_gpu_actor.yaml",
    "legacy_a1_cpu_factorized.yaml",
    "legacy_b1_gpu_factorized.yaml",
)


def _p95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _environment():
    import torch

    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        git = os.environ.get("DOUZERO_GIT_SHA")
    return {
        "git_sha": git,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "platform": platform.platform(),
    }


def _flatten(payload):
    rates = payload["rates"]
    system = payload.get("system", {})
    legal = payload.get("legal_actions", {})
    return {
        **rates,
        "gpu_utilization_median": system.get("gpu_percent", {}).get("median"),
        "gpu_utilization_p95": system.get("gpu_percent", {}).get("p95"),
        "cpu_utilization_median": system.get("cpu_percent", {}).get("median"),
        "rss_mib_max": system.get("rss_mib", {}).get("max"),
        "vram_mib_max": system.get("vram_mib", {}).get("max"),
        "policy_lag_max": max(
            (item.get("max_updates", 0) or 0)
            for item in payload.get("policy_lag", {}).values()
        ),
        "amp_fallbacks": sum(
            payload.get("stats", {}).get(f"amp_fallbacks_{role}", 0)
            for role in ("landlord", "landlord_up", "landlord_down")
        ),
        "legal_actions_p50": legal.get("p50"),
        "legal_actions_p95": legal.get("p95"),
        "legal_actions_max": legal.get("max"),
        "single_legal_ratio": legal.get("single_ratio"),
    }


def _aggregate(rows):
    result = {}
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row[key] is not None]
        if not values:
            continue
        median = statistics.median(values)
        result[key] = {
            "median": median,
            "p95": _p95(values),
            "min": min(values),
            "max": max(values),
            "relative_range": (
                (max(values) - min(values)) / median if median else None
            ),
        }
    return result


def _markdown(bundle):
    lines = [
        "# Legacy V1 Single-GPU Training Benchmark",
        "",
        "Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.",
        "",
        "| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in bundle["aggregate"].items():
        def metric(key, statistic="median"):
            item = value.get(key)
            return "n/a" if item is None else f"{item[statistic]:.3f}"
        lines.append(
            f"| {name} | {value['repeats']} | "
            f"{metric('frames_per_second')} | {metric('frames_per_second', 'p95')} | "
            f"{metric('decisions_per_second')} | "
            f"{metric('learner_updates_per_second')} | "
            f"{metric('gpu_utilization_median')} | "
            f"{metric('cpu_utilization_median')} | "
            f"{metric('vram_mib_max', 'max')} | {metric('rss_mib_max', 'max')} | "
            f"{metric('policy_lag_max', 'max')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", action="append", default=[],
        help="YAML path; repeat for multiple candidates",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=19200)
    parser.add_argument("--measure_frames", type=int, default=57600)
    parser.add_argument("--num_actors", type=int)
    parser.add_argument("--profile_sample_interval", type=int, default=1)
    parser.add_argument("--monitor_interval_seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_buffers", type=int)
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--sync_interval_updates", type=int)
    parser.add_argument("--output_dir", default="artifacts/legacy-training")
    parser.add_argument("--timeout_seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        raise ValueError("at least three repetitions are required")
    if args.warmup_frames < 0 or args.measure_frames <= 0:
        raise ValueError("warmup_frames must be non-negative and measure_frames positive")

    configs = [Path(item) for item in args.config]
    if not configs:
        configs = [ROOT / "benchmarks" / "configs" / name for name in DEFAULT_CONFIGS]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    aggregate = {}

    for config in configs:
        candidate_rows = []
        for repeat in range(1, args.repeats + 1):
            run_name = f"{config.stem}-r{repeat}"
            metrics_path = output_dir / f"{run_name}.json"
            log_path = output_dir / f"{run_name}.log"
            command = [
                sys.executable, str(ROOT / "train.py"),
                "--config", str(config),
                "--total_frames", str(args.warmup_frames + args.measure_frames),
                "--benchmark_warmup_frames", str(args.warmup_frames),
                "--legacy_metrics_path", str(metrics_path),
                "--legacy_profile_sample_interval",
                str(args.profile_sample_interval),
                "--legacy_monitor_interval_seconds",
                str(args.monitor_interval_seconds),
                "--seed", str(args.seed),
                "--disable_checkpoint",
                "--savedir", str(output_dir / "run-logs"),
                "--xpid", run_name,
            ]
            if args.num_actors is not None:
                command.extend(["--num_actors", str(args.num_actors)])
            for option in (
                "batch_size", "num_buffers", "num_threads",
                "sync_interval_updates",
            ):
                value = getattr(args, option)
                if value is not None:
                    command.extend([f"--{option}", str(value)])
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                    check=False,
                )
            wall_seconds = time.monotonic() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{run_name} failed with exit code {completed.returncode}; "
                    f"inspect {log_path}"
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "configuration": config.stem,
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                **_flatten(payload),
            }
            raw_rows.append(row)
            candidate_rows.append(row)
        aggregate[config.stem] = {
            "repeats": len(candidate_rows),
            **_aggregate(candidate_rows),
        }

    bundle = {
        "schema_version": "legacy-training-benchmark-v1",
        "environment": _environment(),
        "protocol": {
            "warmup_frames": args.warmup_frames,
            "measure_frames": args.measure_frames,
            "repeats": args.repeats,
            "checkpoint_disabled": True,
            "seed": args.seed,
        },
        "raw_runs": raw_rows,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(bundle), encoding="utf-8")
    if raw_rows:
        with (output_dir / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(raw_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(raw_rows)
    print(_markdown(bundle))


if __name__ == "__main__":
    main()
