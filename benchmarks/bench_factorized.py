#!/usr/bin/env python
"""Micro-benchmark: legacy vs factorized forward (P04).

Measures CPU latency/throughput for the legacy per-action forward versus the
P04 factorized forward (which encodes the shared history/state once per
decision), at several legal-action counts. Also records the LSTM input batch
size per decision — the direct evidence that the factorized path feeds the
LSTM 1 row instead of N identical rows.

This is a MEASUREMENT tool, not an optimisation claim. It reports honest
medians and p95s on the current host and makes no preset assumption about the
speedup. GPU timing is only collected when CUDA is available.

Usage:
    python benchmarks/bench_factorized.py
    python benchmarks/bench_factorized.py --rounds 50 --output artifacts/benchmark/bench_factorized.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DEFAULT_ROUNDS = 30
DEFAULT_WARMUP = 3
DEFAULT_OUTPUT = "artifacts/benchmark/bench_factorized.json"


def _percentiles(samples_ms, p):
    if not samples_ms:
        return None
    s = sorted(samples_ms)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _bench(fn, rounds, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "rounds": len(samples),
        "median_ms": round(statistics.median(samples), 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "p95_ms": round(_percentiles(samples, 95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


class _LstmBatchRecorder:
    """Record the batch size (number of rows) passed to the LSTM.

    The legacy and factorized forwards both call ``lstm.forward`` once per
    decision; the distinction is that the legacy path feeds it ``N`` identical
    rows while the factorized path feeds it ``1`` row. This records the input
    batch size so the benchmark reports the real work reduction.
    """

    def __init__(self, model):
        self.model = model
        self.original_forward = model.lstm.forward
        self.batch_sizes = []

    def __enter__(self):
        recorder = self

        def _recording_forward(z, *args, **kwargs):
            recorder.batch_sizes.append(z.shape[0])
            return recorder.original_forward(z, *args, **kwargs)

        self.model.lstm.forward = _recording_forward
        return self

    def __exit__(self, *exc):
        self.model.lstm.forward = self.original_forward


def _make_env_and_obs(seed):
    """Build a landlord infoset + legacy obs at the opening (full action set)."""
    import numpy as np
    import torch
    from douzero.env.env import Env, get_obs

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    # Drive to landlord's turn if not already.
    for _ in range(5):
        if env._acting_player_position == "landlord":
            break
        env.step(env.infoset.legal_actions[0])
    if env._acting_player_position != "landlord":
        # Force landlord by resetting with a deterministic seed.
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
    infoset = env.infoset
    obs = get_obs(infoset)
    z_full = torch.from_numpy(obs["z_batch"]).float()
    x_full = torch.from_numpy(obs["x_batch"]).float()
    return z_full, x_full, z_full.shape[0]


def bench_forward_action_buckets(rounds, warmup, seed):
    """Compare legacy vs factorized forward latency at action-count buckets."""
    import torch
    from douzero.dmc.models import model_dict
    from douzero.dmc.models_factorized import factorized_model_dict

    z_full, x_full, full_n = _make_env_and_obs(seed)

    torch.manual_seed(seed)
    legacy = model_dict["landlord"]()
    legacy.eval()
    torch.manual_seed(seed)
    factorized = factorized_model_dict["landlord"]()
    factorized.load_state_dict(legacy.state_dict())
    factorized.eval()

    results = {}
    # Buckets: 1, ~10, ~50, full opening set.
    for n in sorted({1, 10, 50, full_n}):
        if n > full_n:
            continue
        z = z_full[:n]
        x = x_full[:n]

        def run_legacy(z=z, x=x):
            with torch.no_grad():
                legacy(z, x, return_value=True)

        def run_factorized(z=z, x=x):
            with torch.no_grad():
                factorized(z, x, return_value=True)

        legacy_stats = _bench(run_legacy, rounds, warmup)
        fact_stats = _bench(run_factorized, rounds, warmup)
        results[f"n={n}"] = {
            "legacy": legacy_stats,
            "factorized": fact_stats,
            "speedup_median": (
                round(legacy_stats["median_ms"] / fact_stats["median_ms"], 3)
                if fact_stats["median_ms"] else None
            ),
        }
    return {"landlord_forward_buckets": results, "full_action_count": full_n}


def bench_lstm_call_counts(seed):
    """Record the LSTM input batch size per decision for legacy vs factorized.

    This is the P04 efficiency proof: the legacy forward feeds the LSTM ``N``
    identical rows (the tiled history), while the factorized forward feeds it
    exactly ``1`` row (the shared history, encoded once). Both call the LSTM
    once; the work reduction is the N-fold fewer rows processed.
    """
    import torch
    from douzero.dmc.models import model_dict
    from douzero.dmc.models_factorized import factorized_model_dict

    z_full, x_full, full_n = _make_env_and_obs(seed)

    torch.manual_seed(seed)
    legacy = model_dict["landlord"]()
    legacy.eval()
    torch.manual_seed(seed)
    factorized = factorized_model_dict["landlord"]()
    factorized.load_state_dict(legacy.state_dict())
    factorized.eval()

    counts = {}
    for n in sorted({1, 10, full_n}):
        if n > full_n:
            continue
        z = z_full[:n]
        x = x_full[:n]
        with torch.no_grad():
            with _LstmBatchRecorder(legacy) as rec:
                legacy(z, x, return_value=True)
            legacy_rows = rec.batch_sizes
            with _LstmBatchRecorder(factorized) as rec:
                factorized(z, x, return_value=True)
            factorized_rows = rec.batch_sizes
        counts[f"n={n}"] = {
            "legal_actions": n,
            "legacy_lstm_rows": legacy_rows,
            "factorized_lstm_rows": factorized_rows,
        }
    return {"lstm_rows_per_decision": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=20240611)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    from douzero._version import environment_info

    seed = args.seed

    results = {}
    results.update(bench_forward_action_buckets(args.rounds, args.warmup, seed))
    results.update(bench_lstm_call_counts(seed))

    env_info = environment_info()
    bundle = {
        "schema_version": "p04-bench-v1",
        "description": (
            "P04 factorized vs legacy forward CPU micro-benchmark. The "
            "factorized path encodes the shared history/state once per "
            "decision; the legacy path feeds the LSTM N identical rows (the "
            "tiled history). Numbers are host-specific and measure "
            "DETERMINISTIC paths; they are not playing-strength claims."
        ),
        "environment": env_info,
        "config": {"rounds": args.rounds, "warmup": args.warmup, "seed": seed},
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=True)

    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(_to_markdown(bundle))
    print(f"benchmark -> {out_path} (+ {md_path.name})")
    print(_to_markdown(bundle))
    return out_path


def _to_markdown(bundle):
    lines = [
        "# DouZero Factorized Forward Benchmark (P04)",
        "",
        f"- host: `{bundle['environment'].get('platform')}`",
        f"- python: `{bundle['environment'].get('python_version')}`",
        f"- torch: `{bundle['environment'].get('torch_version')}` "
        f"(cuda: `{bundle['environment'].get('cuda_available')}`)",
        f"- git_sha: `{bundle['environment'].get('git_sha')}`",
        f"- rounds: {bundle['config']['rounds']}, warmup: {bundle['config']['warmup']}, "
        f"seed: {bundle['config']['seed']}",
        "",
        "## Forward latency by legal-action count (landlord)",
        "",
        "| actions | legacy median (ms) | factorized median (ms) | speedup (median) |",
        "|---:|---:|---:|---:|",
    ]
    buckets = bundle["results"].get("landlord_forward_buckets", {})
    full_n = bundle["results"].get("full_action_count", "?")
    for name, b in buckets.items():
        lines.append(
            f"| {name} (full={full_n}) | {b['legacy']['median_ms']} "
            f"| {b['factorized']['median_ms']} | {b['speedup_median']} |"
        )
    lines += [
        "",
        "## LSTM rows processed per decision",
        "",
        "| actions | legacy LSTM rows | factorized LSTM rows |",
        "|---:|---:|---:|",
    ]
    calls = bundle["results"].get("lstm_rows_per_decision", {})
    for name, c in calls.items():
        lines.append(
            f"| {name} | {c['legacy_lstm_rows']} | {c['factorized_lstm_rows']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
