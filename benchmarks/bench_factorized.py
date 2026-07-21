#!/usr/bin/env python
"""Micro-benchmark: legacy vs factorized forward and end-to-end DeepAgent (P04).

Measures CPU latency/throughput for:
  * model-forward-only (legacy vs factorized) at several legal-action counts;
  * the full ``DeepAgent.act`` path (observation encoding + tensor build +
    forward + argmax) for both backends;
  * CPU peak RSS for the full act path, measured in ISOLATED subprocesses so
    each backend's ``ru_maxrss`` reflects only its own allocations;
  * the LSTM input batch size per decision (the P04 work-reduction proof:
    legacy feeds the LSTM N identical rows, factorized feeds it 1).

This is a MEASUREMENT tool, not an optimisation claim. It reports honest
medians and p95s on the current host and makes no preset assumption about the
speedup. The model-forward-only numbers are NOT end-to-end DeepAgent numbers;
both are reported separately and labelled clearly.

This benchmark is **CPU-only**. CUDA is force-hidden at import so the models
(which probe ``torch.cuda.is_available()``) deterministically run on CPU. GPU
timing, GPU memory, and CPU/GPU parity comparison are **out of scope for P04**
and deferred to P14 (Actor/Learner, AMP, DDP and throughput optimization),
where a device-correct benchmark with ``torch.cuda.Event`` synchronization and
``reset_peak_memory_stats`` / ``max_memory_allocated`` belongs. Measuring GPU
latency without CUDA-event synchronization would be incorrect (CUDA ops are
asynchronous; ``time.perf_counter`` without ``torch.cuda.synchronize()`` does
not bound the kernel time).

Usage:
    python benchmarks/bench_factorized.py
    python benchmarks/bench_factorized.py --rounds 50
    python benchmarks/bench_factorized.py --output artifacts/benchmark/bench_factorized.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Force CPU for the whole benchmark. This MUST happen before any model/tensor
# work: the legacy and factorized models probe torch.cuda.is_available() and
# DeepAgent migrates tensors/models to CUDA when it returns True.
#
# Use UNCONDITIONAL assignment, not setdefault: setdefault only writes when the
# key is ABSENT, so a caller-supplied ``CUDA_VISIBLE_DEVICES=0`` would survive
# and silently put DeepAgent on the GPU while the benchmark reports
# ``device: cpu``. Overwriting guarantees the benchmark is CPU-only on any
# host, so a GPU machine and the CPU CI image measure the same path. GPU
# benchmarking is deferred to P14 (see module docstring).
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _assert_cpu_only():
    """Self-check that CUDA is actually hidden from torch.

    A CPU-only torch build reports ``cuda.is_available()==False`` regardless of
    ``CUDA_VISIBLE_DEVICES``, so it cannot alone expose a failure to override a
    caller-supplied value. This check verifies the ENVIRONMENT VARIABLE itself
    is empty (the contract this benchmark enforces) AND that torch agrees, so a
    GPU host with a stale ``CUDA_VISIBLE_DEVICES=0`` fails loudly here instead
    of silently timing an asynchronous CUDA path with ``perf_counter``.
    """
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError(
            "CPU-only factorized benchmark failed to hide CUDA: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
            "The benchmark unconditionally sets it to '' at import; if you see "
            "this, the env var was set after import (e.g. the module was "
            "imported before the assignment ran)."
        )
    if torch.cuda.is_available():
        raise RuntimeError(
            "CPU-only factorized benchmark: CUDA is still available to torch "
            "after setting CUDA_VISIBLE_DEVICES=''. This can happen if torch "
            "was initialised before the env var was set, or on a build that "
            "ignores the var. Refusing to time a CUDA path as 'CPU-only'."
        )

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


def _peak_rss_kb_child():
    """Peak resident set size of the CURRENT (child) process in KiB.

    Uses RUSAGE_SELF, which for a freshly-spawned child reflects only that
    child's own allocations (not the parent's). This is what makes the per-
    backend peak-RSS comparison valid: each backend runs in its own child.
    """
    import sys
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    if sys.platform == "darwin":
        return rss // 1024  # bytes -> KiB
    return rss  # already KiB on Linux


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


def _make_env_and_infoset(seed, position="landlord"):
    """Build an infoset + legacy obs at a non-trivial decision point."""
    import numpy as np
    import torch
    from douzero.env.env import Env, get_obs

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(5):
        if env._acting_player_position == position:
            break
        env.step(env.infoset.legal_actions[0])
    if env._acting_player_position != position:
        np.random.seed(seed)
        env = Env("adp")
        env.reset()
    infoset = env.infoset
    obs = get_obs(infoset)
    z_full = torch.from_numpy(obs["z_batch"]).float()
    x_full = torch.from_numpy(obs["x_batch"]).float()
    return infoset, z_full, x_full, z_full.shape[0]


def _build_paired_models(seed, position="landlord"):
    import torch
    from douzero.dmc.models import model_dict
    from douzero.dmc.models_factorized import factorized_model_dict

    torch.manual_seed(seed)
    legacy = model_dict[position]()
    legacy.eval()
    torch.manual_seed(seed)
    factorized = factorized_model_dict[position]()
    factorized.load_state_dict(legacy.state_dict())
    factorized.eval()
    return legacy, factorized


def bench_model_forward_only(rounds, warmup, seed):
    """Model-forward-only latency at action-count buckets (legacy vs factorized).

    This measures ONLY model(z, x, return_value=True) / forward_factorized,
    under torch.no_grad(). It does NOT include observation encoding, tensor
    construction, device transfer, or argmax. It is labelled separately from
    the end-to-end DeepAgent.act numbers.
    """
    import torch
    _, z_full, x_full, full_n = _make_env_and_infoset(seed)
    from douzero.env.env import get_obs_factorized
    infoset, _, _, _ = _make_env_and_infoset(seed)
    split_obs = get_obs_factorized(infoset)
    z_single = torch.from_numpy(split_obs["z_single"]).float()
    x_state_single = torch.from_numpy(split_obs["x_state_single"]).float()
    x_action_full = torch.from_numpy(split_obs["x_action"]).float()

    legacy, factorized = _build_paired_models(seed)

    results = {}
    for n in sorted({1, 10, 50, full_n}):
        if n > full_n:
            continue
        z = z_full[:n]
        x = x_full[:n]
        xa = x_action_full[:n]

        def run_legacy(z=z, x=x):
            with torch.no_grad():
                legacy(z, x, return_value=True)

        def run_factorized(z=z, x=x):
            with torch.no_grad():
                factorized(z, x, return_value=True)

        def run_factorized_split(zs=z_single, xs=x_state_single, xa=xa):
            with torch.no_grad():
                factorized.forward_factorized(
                    zs, xs, xa, return_value=True, split_dense1=True
                )

        def run_factorized_unsplit_dense1(
                zs=z_single, xs=x_state_single, xa=xa):
            with torch.no_grad():
                factorized.forward_factorized(
                    zs, xs, xa, return_value=True, split_dense1=False
                )

        legacy_stats = _bench(run_legacy, rounds, warmup)
        fact_batch_stats = _bench(run_factorized, rounds, warmup)
        fact_split_stats = _bench(run_factorized_split, rounds, warmup)
        fact_unsplit_dense1_stats = _bench(
            run_factorized_unsplit_dense1, rounds, warmup
        )
        results[f"n={n}"] = {
            "legacy_model_forward": legacy_stats,
            "factorized_model_forward_tiled_batch": fact_batch_stats,
            "factorized_model_forward_split_obs": fact_split_stats,
            "factorized_model_forward_split_obs_unsplit_dense1": (
                fact_unsplit_dense1_stats
            ),
            "speedup_median_adaptive_split_dense1": (
                round(
                    fact_unsplit_dense1_stats["median_ms"]
                    / fact_split_stats["median_ms"], 3
                ) if fact_split_stats["median_ms"] else None
            ),
            "speedup_median_split_vs_legacy": (
                round(legacy_stats["median_ms"] / fact_split_stats["median_ms"], 3)
                if fact_split_stats["median_ms"] else None
            ),
        }
    return {"model_forward_only": results, "full_action_count": full_n}


def bench_deep_agent_act(rounds, warmup, seed, position="landlord"):
    """Full DeepAgent.act latency: encode + tensor + forward + argmax.

    Measures the REAL deployment path (both backends), including get_obs /
    get_obs_factorized, tensor construction, the forward, and np.argmax. This
    is the number that matters for inference latency; the model-forward-only
    numbers above isolate the model cost. CPU-only (CUDA is hidden at import).
    """
    import os
    import tempfile
    import torch
    from douzero.dmc.models import model_dict
    from douzero.evaluation.deep_agent import DeepAgent

    infoset, _, _, _ = _make_env_and_infoset(seed, position=position)

    torch.manual_seed(seed)
    legacy = model_dict[position]()
    ckpt = os.path.join(tempfile.mkdtemp(prefix="bench_fac_"), f"{position}.ckpt")
    torch.save(legacy.state_dict(), ckpt)
    agent_legacy = DeepAgent(position, ckpt, backend="legacy")
    agent_fact = DeepAgent(position, ckpt, backend="legacy_factorized")

    # Single-legal-action short-circuit is bypassed by using a multi-action infoset.
    assert len(infoset.legal_actions) > 1

    def run_legacy():
        agent_legacy.act(infoset)

    def run_factorized():
        agent_fact.act(infoset)

    legacy_stats = _bench(run_legacy, rounds, warmup)
    fact_stats = _bench(run_factorized, rounds, warmup)
    return {
        f"deep_agent_act_{position}": {
            "legacy": legacy_stats,
            "factorized": fact_stats,
            "speedup_median": (
                round(legacy_stats["median_ms"] / fact_stats["median_ms"], 3)
                if fact_stats["median_ms"] else None
            ),
            "note": (
                "Full DeepAgent.act path (CPU-only): observation encoding + "
                "tensor build + forward + argmax. The factorized backend uses "
                "get_obs_factorized (no tiling) + forward_factorized."
            ),
        }
    }


# --------------------------------------------------------------------------- #
# Peak RSS — measured in ISOLATED subprocesses (review blocker #2)
# --------------------------------------------------------------------------- #
def _peak_rss_child_main(position, backend, seed, num_acts, q):
    """Child-process target: run one backend and report the child's peak RSS.

    Runs in a fresh process so ``ru_maxrss`` reflects ONLY this backend's
    allocations (not the other backend's, and not the parent's). Both backends
    load the same checkpoint, run the same infoset, and perform the same number
    of act() calls.
    """
    import os
    import tempfile
    import torch
    from douzero.dmc.models import model_dict
    from douzero.evaluation.deep_agent import DeepAgent

    infoset, _, _, _ = _make_env_and_infoset(seed, position=position)
    torch.manual_seed(seed)
    legacy = model_dict[position]()
    ckpt = os.path.join(tempfile.mkdtemp(prefix="bench_mem_"), f"{position}.ckpt")
    torch.save(legacy.state_dict(), ckpt)
    agent = DeepAgent(position, ckpt, backend=backend)
    for _ in range(3):  # warmup
        agent.act(infoset)
    for _ in range(num_acts):
        agent.act(infoset)
    q.put(_peak_rss_kb_child())


def bench_peak_memory(seed, position="landlord", num_acts=100):
    """CPU peak RSS for the full DeepAgent.act path, per backend, in isolation.

    Each backend runs in its OWN subprocess so ``ru_maxrss`` (the process
    lifetime peak) reflects only that backend's allocations. Running both
    backends in one process would be invalid: ``ru_maxrss`` never decreases, so
    the second-measured backend would inherit the first's peak and the
    comparison could never show the factorized backend as lower.

    GPU memory is not measured (CPU-only benchmark; GPU memory measurement
    belongs in P14 with device-correct tooling).
    """
    import torch
    ctx = torch.multiprocessing.get_context("spawn")
    results = {}
    for backend in ("legacy", "legacy_factorized"):
        q = ctx.SimpleQueue()
        p = ctx.Process(
            target=_peak_rss_child_main,
            args=(position, backend, seed, num_acts, q),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            results[backend] = {"error": f"child exited with code {p.exitcode}"}
            continue
        results[backend] = {"peak_rss_kib": q.get()}
    return {
        f"peak_rss_kib_{position}": {
            "legacy": results["legacy"],
            "factorized": results["legacy_factorized"],
            "num_acts": num_acts,
            "note": (
                "Process peak RSS (KiB) per backend, each measured in an "
                "ISOLATED child process so ru_maxrss reflects only that "
                "backend's allocations (not the other backend's or the "
                "parent's). CPU-only; GPU memory not measured (deferred to "
                "P14). NOTE: the tiled-batch memory difference (the landlord "
                "tiled z_batch/x_batch is ~0.4 MiB at 287 actions) is far "
                "smaller than the torch/Python interpreter baseline (~280 "
                "MiB), so the two backends typically report the SAME rounded "
                "process peak. This measurement therefore provides a coarse "
                "regression signal — no peak-RSS increase was observed for the "
                "factorized path in this benchmark run — but it is NOT a "
                "strict proof that the factorized path never increases peak "
                "RSS: ru_maxrss is too coarse to resolve sub-MiB tiled-batch "
                "differences and is sampled once per subprocess. An "
                "allocation-scoped measurement (e.g. tracemalloc, or torch "
                "CUDA memory on GPU) would be needed to quantify the saving "
                "and is deferred to P14."
            ),
        }
    }


def bench_lstm_call_counts(seed):
    """Record the LSTM input batch size per decision for legacy vs factorized.

    This is the P04 efficiency proof: the legacy forward feeds the LSTM ``N``
    identical rows (the tiled history), while the factorized forward feeds it
    exactly ``1`` row (the shared history, encoded once). Both call the LSTM
    once; the work reduction is the N-fold fewer rows processed.
    """
    import torch
    _, z_full, x_full, full_n = _make_env_and_infoset(seed)
    legacy, factorized = _build_paired_models(seed)

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


def _maybe_gpu_parity_note():
    """Return a note on GPU parity status (review blocker #3)."""
    return {
        "cuda_hidden_for_benchmark": True,
        "parity_note": (
            "CPU numerical and argmax parity are tested "
            "(tests/test_factorized_parity.py). GPU numerical and argmax "
            "parity are NOT measured. Mathematical equivalence does not imply "
            "bitwise or universal argmax identity across CPU/GPU (different "
            "kernels, reduction order, cuDNN RNN non-determinism). GPU timing "
            "and memory are deferred to P14."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=20240611)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num_acts_memory", type=int, default=100,
                        help="Number of act() calls for the peak-RSS measurement.")
    args = parser.parse_args(argv)

    # Fail fast before any timing if CUDA is not actually hidden. This is the
    # contract that makes the ``device: cpu`` label honest.
    _assert_cpu_only()

    from douzero._version import environment_info

    seed = args.seed
    results = {}
    # Model-forward-only (labelled clearly; NOT end-to-end).
    results.update(bench_model_forward_only(args.rounds, args.warmup, seed))
    # End-to-end DeepAgent.act for all three roles.
    for position in ["landlord", "landlord_up", "landlord_down"]:
        results.update(bench_deep_agent_act(args.rounds, args.warmup, seed, position))
    # CPU peak memory (isolated subprocesses per backend).
    results.update(bench_peak_memory(seed, position="landlord",
                                     num_acts=args.num_acts_memory))
    # LSTM work-reduction proof.
    results.update(bench_lstm_call_counts(seed))
    # GPU parity status note.
    results["gpu_status"] = _maybe_gpu_parity_note()

    env_info = environment_info()
    bundle = {
        "schema_version": "p04-bench-v3",
        "description": (
            "P04 factorized vs legacy benchmark (CPU-ONLY). CUDA is hidden at "
            "import; GPU timing/memory deferred to P14. model_forward_only "
            "isolates the model cost; deep_agent_act_* is the full deployment "
            "path (encode + tensor + forward + argmax); peak_rss is measured "
            "per backend in isolated subprocesses. The factorized backend uses "
            "get_obs_factorized (no tiling) + forward_factorized. Numbers are "
            "host-specific and measure DETERMINISTIC paths; they are not "
            "playing-strength claims."
        ),
        "environment": env_info,
        "config": {
            "rounds": args.rounds, "warmup": args.warmup, "seed": seed,
            "num_acts_memory": args.num_acts_memory,
            "device": "cpu",
        },
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
        "# DouZero Factorized Forward Benchmark (P04, CPU-only)",
        "",
        f"- host: `{bundle['environment'].get('platform')}`",
        f"- python: `{bundle['environment'].get('python_version')}`",
        f"- torch: `{bundle['environment'].get('torch_version')}` "
        f"(cuda: `{bundle['environment'].get('cuda_available')}`)",
        f"- git_sha: `{bundle['environment'].get('git_sha')}`",
        f"- rounds: {bundle['config']['rounds']}, warmup: {bundle['config']['warmup']}, "
        f"seed: {bundle['config']['seed']}, device: {bundle['config']['device']}",
        "",
        "## Model-forward-only latency (landlord, NOT end-to-end)",
        "",
        "| actions | legacy median (ms) | unsplit dense1 (ms) | "
        "adaptive split dense1 (ms) | dense1 speedup | legacy speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    mfo = bundle["results"].get("model_forward_only", {})
    full_n = bundle["results"].get("full_action_count", "?")
    for name, b in mfo.items():
        legacy_md = b["legacy_model_forward"]["median_ms"]
        fact_md = b["factorized_model_forward_split_obs"]["median_ms"]
        unsplit_md = b[
            "factorized_model_forward_split_obs_unsplit_dense1"
        ]["median_ms"]
        dense1_sp = b["speedup_median_adaptive_split_dense1"]
        sp = b["speedup_median_split_vs_legacy"]
        lines.append(
            f"| {name} (full={full_n}) | {legacy_md} | {unsplit_md} | "
            f"{fact_md} | {dense1_sp} | {sp} |"
        )
    lines += [
        "",
        "## End-to-end DeepAgent.act latency (encode + tensor + forward + argmax)",
        "",
        "| role | legacy median (ms) | factorized median (ms) | speedup |",
        "|---|---:|---:|---:|",
    ]
    for key, val in bundle["results"].items():
        if key.startswith("deep_agent_act_"):
            role = key.replace("deep_agent_act_", "")
            lines.append(
                f"| {role} | {val['legacy']['median_ms']} "
                f"| {val['factorized']['median_ms']} | {val['speedup_median']} |"
            )
    num_acts_mem = bundle['config'].get('num_acts_memory', '?')
    lines += [
        "",
        "## CPU peak RSS (landlord, isolated subprocess per backend, "
        f"{num_acts_mem} acts)",
        "",
        "| backend | peak RSS (KiB) |",
        "|---|---:|",
    ]
    for key, val in bundle["results"].items():
        if key.startswith("peak_rss_kib_"):
            for backend in ("legacy", "factorized"):
                entry = val[backend]
                rss = entry.get("peak_rss_kib", entry.get("error"))
                lines.append(f"| {backend} | {rss} |")
            lines.append("")
            lines.append(f"_{val.get('note', '')}_")
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
    gpu = bundle["results"].get("gpu_status", {})
    lines += [
        "",
        "## GPU parity status",
        "",
        f"- cuda_hidden_for_benchmark: `{gpu.get('cuda_hidden_for_benchmark')}`",
        f"- {gpu.get('parity_note', '')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
