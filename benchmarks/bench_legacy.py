#!/usr/bin/env python
"""Micro-benchmark for the legacy DouZero inference and env paths (P00).

Measures CPU latency/throughput for the operations the baseline depends on:
  * ``get_obs`` (observation encoding),
  * ``model.forward`` at several legal-action counts,
  * ``DeepAgent.act`` (encode + forward + argmax),
  * ``Env.step`` (one card-play step).

This is a *measurement* tool, not an optimisation claim. It reports honest
medians and p95s on the current host. GPU timing is only collected when CUDA
is available (in the P00 test image it is not).

Usage:
    python benchmarks/bench_legacy.py
    python benchmarks/bench_legacy.py --rounds 50 --output artifacts/benchmark/bench.json
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
DEFAULT_OUTPUT = "artifacts/benchmark/bench.json"


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


def _make_seed_and_env(seed):
    import numpy as np
    from douzero.env.env import Env

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    return env


def bench_get_obs(rounds, warmup, seed):
    import numpy as np
    from douzero.env.env import Env, get_obs

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    # Drive a few steps so the history is non-trivial.
    for _ in range(3):
        env.step(env.infoset.legal_actions[0])
    infoset = env.infoset

    return {"get_obs": _bench(lambda: get_obs(infoset), rounds, warmup)}


def bench_model_forward(rounds, warmup, seed):
    """Forward at action-count buckets: 1, ~10, ~50, and full opening set."""
    import torch
    from douzero.dmc.models import model_dict
    from douzero.env.env import Env, get_obs

    import numpy as np

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    obs = get_obs(env.infoset)
    full_n = obs["z_batch"].shape[0]
    z_full = torch.from_numpy(obs["z_batch"]).float()
    x_full = torch.from_numpy(obs["x_batch"]).float()
    torch.manual_seed(seed)
    model = model_dict["landlord"]()
    model.eval()

    results = {}
    for n in sorted({1, 10, 50, full_n}):
        if n > full_n:
            continue
        z = z_full[:n]
        x = x_full[:n]

        def run(z=z, x=x):
            with torch.no_grad():
                model(z, x, return_value=True)

        results[f"forward_n={n}"] = _bench(run, rounds, warmup)
    return {"model_forward_landlord_opening": results}


def bench_deepagent_act(rounds, warmup, seed):
    import tempfile

    import torch
    from douzero.dmc.models import model_dict
    from douzero.dmc.models import model_dict as _md
    from douzero.env.env import Env
    from douzero.evaluation.deep_agent import DeepAgent

    torch.manual_seed(seed)
    model = model_dict["landlord"]()
    ckpt = os.path.join(tempfile.mkdtemp(prefix="bench_"), "landlord.ckpt")
    torch.save(model.state_dict(), ckpt)
    agent = DeepAgent("landlord", ckpt)

    import numpy as np

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    infoset = env.infoset

    return {"deepagent_act_opening": _bench(lambda: agent.act(infoset), rounds, warmup)}


def bench_env_step(rounds, warmup, seed):
    """Steps-per-second over many fresh games, one step each (opening move)."""
    import numpy as np
    from douzero.env.env import Env

    # We measure reset+step because a single step is cheap; report per (reset+step).
    def run():
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
        env.step(env.infoset.legal_actions[0])

    stats = _bench(run, rounds, warmup)
    sps = round(1000.0 / stats["median_ms"], 2) if stats["median_ms"] else None
    # Keep throughput as a SEPARATE entry so every value in a group is itself a
    # stats dict (the markdown table iterates group values uniformly).
    return {
        "env_reset_plus_opening_step": stats,
        "opening_steps_per_second_approx": {
            "median_ms": sps,
            "mean_ms": sps,
            "p95_ms": sps,
            "min_ms": sps,
            "max_ms": sps,
            "note": "1000/median_ms of reset+opening_step; throughput proxy",
        },
    }


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
    results.update(bench_get_obs(args.rounds, args.warmup, seed))
    results.update(bench_model_forward(args.rounds, args.warmup, seed))
    results.update(bench_deepagent_act(args.rounds, args.warmup, seed))
    results.update(bench_env_step(args.rounds, args.warmup, seed))

    # Flatten into a single-level {name: stats} map so the markdown table and
    # downstream tooling can treat every entry uniformly. A group value that is
    # itself a dict of stats (e.g. model_forward at several action counts) is
    # expanded with prefixed names; a bare stats dict is kept as-is.
    flat = {}

    def _is_stats(v):
        return isinstance(v, dict) and "median_ms" in v

    for gname, gval in results.items():
        if _is_stats(gval):
            flat[gname] = gval
        elif isinstance(gval, dict):
            for sub, subval in gval.items():
                if _is_stats(subval):
                    flat[f"{gname}/{sub}"] = subval
    results = flat

    env_info = environment_info()
    bundle = {
        "schema_version": "p00-bench-v1",
        "description": (
            "Legacy DouZero CPU inference/env micro-benchmark. Numbers are "
            "host-specific and measure DETERMINISTIC paths; they are not "
            "playing-strength or optimisation claims."
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
        "# DouZero Legacy Benchmark (P00)",
        "",
        f"- host: `{bundle['environment'].get('platform')}`",
        f"- python: `{bundle['environment'].get('python_version')}`",
        f"- torch: `{bundle['environment'].get('torch_version')}` "
        f"(cuda: `{bundle['environment'].get('cuda_available')}`)",
        f"- git_sha: `{bundle['environment'].get('git_sha')}`",
        f"- rounds: {bundle['config']['rounds']}, warmup: {bundle['config']['warmup']}, "
        f"seed: {bundle['config']['seed']}",
        "",
        "| benchmark | median (ms) | mean (ms) | p95 (ms) | min (ms) | max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in bundle["results"].items():
        lines.append(
            f"| {name} | {s['median_ms']} | {s['mean_ms']} | {s['p95_ms']} "
            f"| {s['min_ms']} | {s['max_ms']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
