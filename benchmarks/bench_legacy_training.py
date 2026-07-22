#!/usr/bin/env python
"""Repeatable end-to-end V1 single-GPU training benchmark orchestrator.

Timeout cleanup reaps the complete training process group on POSIX. Native
Windows only terminates the direct child; use WSL2/Linux for formal runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "legacy_a0_cpu_actor_thread1.yaml",
    "legacy_a1_cpu_factorized.yaml",
)
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _p95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_evidence_mode(args):
    if not args.formal:
        return
    if args.allow_dirty:
        raise ValueError("formal benchmarks cannot use --allow_dirty")
    if not args.docker_image_digest:
        raise ValueError(
            "formal benchmarks require --docker_image_digest or "
            "DOUZERO_IMAGE_DIGEST"
        )
    if not DIGEST_PATTERN.fullmatch(args.docker_image_digest):
        raise ValueError(
            "formal benchmarks require a sha256 digest with 64 lowercase "
            "hexadecimal characters"
        )
    if not args.expected_git_sha:
        raise ValueError(
            "formal benchmarks require --expected_git_sha or "
            "DOUZERO_EXPECTED_GIT_SHA"
        )
    if not GIT_SHA_PATTERN.fullmatch(args.expected_git_sha):
        raise ValueError(
            "formal benchmarks require a full 40-character lowercase Git SHA"
        )


def _validate_provenance(args, environment):
    git_sha = environment["git_sha"]
    git_status = environment["git_status_porcelain"]
    if args.formal:
        if environment.get("git_toplevel") != environment.get("source_root"):
            raise RuntimeError(
                "formal benchmark source_root must be the Git worktree top level"
            )
        if (
            not isinstance(git_sha, str)
            or not GIT_SHA_PATTERN.fullmatch(git_sha)
        ):
            raise RuntimeError("formal benchmark could not verify the Git SHA")
        if git_status is None:
            raise RuntimeError(
                "formal benchmark could not verify the Git worktree status"
            )
        if git_status:
            raise RuntimeError(
                "formal benchmark requires a clean Git worktree"
            )
        if git_sha != args.expected_git_sha:
            raise RuntimeError(
                "formal benchmark Git SHA mismatch: expected "
                f"{args.expected_git_sha}, found {git_sha}"
            )
        return
    if git_status is None and not args.allow_dirty:
        raise RuntimeError(
            "could not verify Git worktree status; use --allow_dirty only "
            "for exploratory runs"
        )
    if git_status and not args.allow_dirty:
        raise RuntimeError(
            "refusing to produce benchmark evidence from a dirty checkout; "
            "use --allow_dirty only for exploratory runs"
        )


def _run_training(command, *, log_file, timeout, cwd=ROOT):
    """Run one benchmark in its own process group and reap it on timeout."""
    use_process_group = os.name == "posix"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=use_process_group,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if use_process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if use_process_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
        raise TimeoutError(
            f"benchmark timed out after {timeout} seconds; process group reaped"
        ) from exc


def _checkpoint_cli_args(enabled):
    if enabled:
        return ["--no-disable_checkpoint", "--save_interval", "1"]
    return ["--disable_checkpoint"]


def _environment(source_root=ROOT):
    import torch

    source_root = source_root.resolve()

    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git = os.environ.get("DOUZERO_GIT_SHA")
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        git_status = None
    try:
        git_toplevel = str(
            Path(
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=source_root,
                    text=True,
                ).strip()
            ).resolve()
        )
    except (OSError, subprocess.SubprocessError):
        git_toplevel = None
    return {
        "git_sha": git,
        "git_status_porcelain": git_status,
        "source_root": str(source_root),
        "git_toplevel": git_toplevel,
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


def _validate_policy_lag(row, max_updates):
    if max_updates is None:
        return
    observed = row["policy_lag_max"]
    if observed > max_updates:
        raise RuntimeError(
            "policy lag exceeded the configured upper bound: "
            f"observed {observed} updates, allowed {max_updates}"
        )


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
    parser.add_argument(
        "--source_root", type=Path, default=ROOT,
        help="Clean source checkout to benchmark; defaults to this repository",
    )
    parser.add_argument(
        "--checkpoint_enabled", action="store_true",
        help="Keep checkpoints enabled and require a final model.tar per repeat",
    )
    parser.add_argument(
        "--max_policy_lag_updates", type=int,
        help="Fail when any repeat observes a larger maximum policy lag",
    )
    parser.add_argument(
        "--docker_image_digest", default=os.environ.get("DOUZERO_IMAGE_DIGEST"),
        help=(
            "Caller-declared immutable image ID/digest recorded in evidence; "
            "the runner cannot introspect the outer container runtime"
        ),
    )
    parser.add_argument(
        "--expected_git_sha", default=os.environ.get("DOUZERO_EXPECTED_GIT_SHA"),
        help="Exact full Git SHA required for a formal evidence run",
    )
    parser.add_argument(
        "--allow_dirty", action="store_true",
        help="Allow exploratory runs from a dirty checkout",
    )
    parser.add_argument(
        "--formal", action="store_true",
        help="Require an immutable image digest and a clean checkout",
    )
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    if not (source_root / "train.py").is_file():
        raise ValueError(f"source root has no train.py: {source_root}")
    _validate_evidence_mode(args)
    if args.repeats < 3:
        raise ValueError("at least three repetitions are required")
    if args.warmup_frames < 0 or args.measure_frames <= 0:
        raise ValueError("warmup_frames must be non-negative and measure_frames positive")
    if args.max_policy_lag_updates is not None and args.max_policy_lag_updates < 0:
        raise ValueError("max_policy_lag_updates must be non-negative")

    configs = [Path(item).resolve() for item in args.config]
    if not configs:
        configs = [
            source_root / "benchmarks" / "configs" / name
            for name in DEFAULT_CONFIGS
        ]
    environment = _environment(source_root)
    _validate_provenance(args, environment)
    output_dir = Path(args.output_dir).resolve()
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
                sys.executable, str(source_root / "train.py"),
                "--config", str(config),
                "--total_frames", str(args.warmup_frames + args.measure_frames),
                "--benchmark_warmup_frames", str(args.warmup_frames),
                "--legacy_metrics_path", str(metrics_path),
                "--legacy_profile_sample_interval",
                str(args.profile_sample_interval),
                "--legacy_monitor_interval_seconds",
                str(args.monitor_interval_seconds),
                "--seed", str(args.seed),
                "--savedir", str(output_dir / "run-logs"),
                "--xpid", run_name,
            ]
            command.extend(_checkpoint_cli_args(args.checkpoint_enabled))
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
                returncode = _run_training(
                    command, log_file=log_file,
                    timeout=args.timeout_seconds,
                    cwd=source_root,
                )
            wall_seconds = time.monotonic() - started
            if returncode != 0:
                raise RuntimeError(
                    f"{run_name} failed with exit code {returncode}; "
                    f"inspect {log_path}"
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint_path = output_dir / "run-logs" / run_name / "model.tar"
            if args.checkpoint_enabled and not checkpoint_path.is_file():
                raise RuntimeError(
                    f"{run_name} completed without {checkpoint_path}"
                )
            measured_frames = payload["counts"]["learner"]["frames"]
            expected_total = args.warmup_frames + args.measure_frames
            if payload["frames_total"] != expected_total:
                raise RuntimeError(
                    f"{run_name} completed {payload['frames_total']} frames; "
                    f"expected exactly {expected_total}"
                )
            if measured_frames != args.measure_frames:
                raise RuntimeError(
                    f"{run_name} measured {measured_frames} learner frames; "
                    f"expected exactly {args.measure_frames}"
                )
            row = {
                "configuration": config.stem,
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                "frames_total": payload["frames_total"],
                "measurement_frames": measured_frames,
                "metrics_sha256": _sha256(metrics_path),
                "checkpoint_sha256": (
                    _sha256(checkpoint_path) if args.checkpoint_enabled else None
                ),
                **_flatten(payload),
            }
            _validate_policy_lag(row, args.max_policy_lag_updates)
            raw_rows.append(row)
            candidate_rows.append(row)
        aggregate[config.stem] = {
            "repeats": len(candidate_rows),
            **_aggregate(candidate_rows),
        }

    bundle = {
        "schema_version": "legacy-training-benchmark-v2",
        "environment": environment,
        "protocol": {
            "warmup_frames": args.warmup_frames,
            "measure_frames": args.measure_frames,
            "repeats": args.repeats,
            "checkpoint_disabled": not args.checkpoint_enabled,
            "max_policy_lag_updates": args.max_policy_lag_updates,
            "source_root": str(source_root),
            "seed": args.seed,
            "docker_image_digest": args.docker_image_digest,
            "docker_image_digest_source": "caller_declared",
            "docker_image_identity_verified": False,
            "expected_git_sha": args.expected_git_sha,
            "formal": args.formal,
            "config_sha256": {
                str(config): _sha256(config) for config in configs
            },
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
    artifact_paths = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "sha256sums.json"
    ]
    (output_dir / "sha256sums.json").write_text(
        json.dumps(
            {path.name: _sha256(path) for path in sorted(artifact_paths)},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(_markdown(bundle))


if __name__ == "__main__":
    main()
