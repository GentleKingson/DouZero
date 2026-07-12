#!/usr/bin/env python
"""Micro-benchmark: Model V2 forward latency and parameter report (P05).

Measures CPU latency for the V2 shared state-action model:

  * parameter count (per submodule + total) for the default config;
  * model-forward-only latency at several legal-action counts (1, 10, 50,
    full action set) — the per-decision cost as the candidate set grows;
  * full ``DeepAgentV2.act`` latency (observation encoding + tensor build +
    forward + selection) for the default config;
  * a comparison row against the legacy factorized forward at the same action
    counts, so the V2 cost is contextualised (NOT a parity claim — the
    architectures differ).

This is a MEASUREMENT tool, not a strength or speedup claim. It reports honest
medians and p95s on the current host and makes no preset assumption about the
relative cost. The model-forward-only numbers are NOT end-to-end act numbers;
both are reported separately and labelled clearly.

This benchmark is **CPU-only**, mirroring ``bench_factorized.py``. CUDA is
force-hidden at import. GPU timing, GPU memory, and AMP/DDP comparison are out
of scope for P05 and deferred to P14.

Usage:
    python benchmarks/bench_model_v2.py
    python benchmarks/bench_model_v2.py --rounds 50
    python benchmarks/bench_model_v2.py --output artifacts/benchmark/bench_model_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Force CPU for the whole benchmark (see bench_factorized.py for the rationale:
# unconditional assignment, not setdefault, so a caller-supplied GPU id does
# not survive and silently time an asynchronous CUDA path as "CPU-only").
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _assert_cpu_only():
    """Self-check that CUDA is actually hidden from torch."""
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError(
            "CPU-only V2 benchmark failed to hide CUDA: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
        )
    if torch.cuda.is_available():
        raise RuntimeError(
            "CPU-only V2 benchmark: CUDA is still available to torch after "
            "setting CUDA_VISIBLE_DEVICES=''."
        )


DEFAULT_ROUNDS = 30
DEFAULT_WARMUP = 3
DEFAULT_OUTPUT = "artifacts/benchmark/bench_model_v2.json"


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


def _make_v2_obs(seed: int = 42, steps_into_game: int = 0):
    """Build a real ObservationV2 at the landlord's turn for benchmarking."""
    import numpy as np
    from douzero.env.env import Env
    from douzero.observation import get_obs_v2

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(steps_into_game):
        try:
            env.step(env.infoset.legal_actions[0])
        except Exception:
            break
    while env._acting_player_position != "landlord":
        env.step(env.infoset.legal_actions[0])
    return get_obs_v2(env.infoset)


def _slice_actions(bundle, n):
    """Return a bundle's action block sliced to the first ``n`` actions."""
    return bundle.action_features[:n], bundle.action_mask[:n]


def bench_parameter_count(model):
    """Return the per-submodule + total parameter counts."""
    counts = model.parameter_count()
    return {
        "submodules": {k: int(v) for k, v in counts.items() if k != "total"},
        "total": int(counts["total"]),
        "config": {
            "hidden_size": model.config.hidden_size,
            "history_encoder": model.config.history_encoder,
            "history_layers": model.config.history_layers,
            "history_heads": model.config.history_heads,
            "role_embedding_dim": model.config.role_embedding_dim,
            "mlp_layers": model.config.mlp_layers,
        },
    }


def bench_forward_only(model, obs, rounds, warmup):
    """Model-forward-only latency at several legal-action counts."""
    from douzero.models_v2 import observation_to_model_inputs
    import torch

    bundle = observation_to_model_inputs(obs)
    full_n = bundle.action_features.shape[0]
    action_counts = sorted({1, 10, 50, full_n})
    action_counts = [n for n in action_counts if n <= full_n]
    if full_n not in action_counts:
        action_counts.append(full_n)

    results = {}
    for n in action_counts:
        act_feat, act_mask = _slice_actions(bundle, n)

        def _fn():
            with torch.inference_mode():
                model(
                    bundle.state_card_vectors,
                    bundle.state_context_flat,
                    bundle.context_card_vectors,
                    bundle.context_flat,
                    bundle.history_tokens,
                    bundle.history_key_padding_mask,
                    act_feat,
                    act_mask,
                    bundle.acting_role,
                )

        results[f"n_actions={n}"] = _bench(_fn, rounds, warmup)
    results["full_action_set_count"] = full_n
    return results


def bench_deep_agent_act(agent, env_seed, rounds, warmup):
    """Full DeepAgentV2.act(infoset) latency (encode + forward + select)."""
    import numpy as np
    from douzero.env.env import Env

    def _fn():
        np.random.seed(env_seed)
        env = Env("adp")
        env.reset()
        agent.act(env.infoset)

    return _bench(_fn, rounds, warmup)


def main():
    _assert_cpu_only()
    parser = argparse.ArgumentParser(description="Model V2 forward benchmark (P05)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps_into_game", type=int, default=4,
                        help="Pre-roll the env this many steps so the history is non-empty.")
    args = parser.parse_args()

    import torch
    from douzero.models_v2 import ModelV2, ModelV2Config
    from douzero.observation import build_v2_schema

    schema = build_v2_schema()
    torch.manual_seed(1234)
    model = ModelV2(schema, ModelV2Config())
    model.eval()

    # Build a non-trivial decision point (history present, many legal actions).
    obs = _make_v2_obs(seed=42, steps_into_game=args.steps_into_game)

    report = {
        "schema_version": "p05-bench-v1",
        "device": "cpu",
        "torch_version": torch.__version__,
        "parameter_count": bench_parameter_count(model),
        "forward_only_ms": bench_forward_only(model, obs, args.rounds, args.warmup),
    }

    # DeepAgentV2 end-to-end act path. Constructed with an explicit RuleSet
    # (REQUIRED by DeepAgentV2 since the ruleset-binding fix); legacy matches
    # the default ModelV2 / Env("adp") training context. This block is NOT
    # wrapped in a broad try/except: a failure on the core measurement path
    # must fail the benchmark LOUDLY (non-zero exit, no report written), not be
    # swallowed into an {"error": ...} field that the JSON/Markdown summary
    # would otherwise render as if it were a latency measurement.
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2
    agent = DeepAgentV2("landlord", model, RuleSet.legacy())
    report["deep_agent_act_ms"] = bench_deep_agent_act(
        agent, env_seed=42, rounds=max(args.rounds // 3, 5),
        warmup=max(args.warmup, 1),
    )
    report["deep_agent_act_ms"]["note"] = (
        "Full act(infoset) path: get_obs_v2 + tensor build + forward + "
        "argmax. Fewer rounds because each iteration rebuilds the env."
    )

    # Write JSON + Markdown summary.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Model V2 Benchmark (P05)\n\n")
        fh.write(f"- device: cpu\n")
        fh.write(f"- torch: {report['torch_version']}\n")
        pc = report["parameter_count"]
        fh.write(f"- total parameters: {pc['total']:,}\n")
        fh.write(f"- config: hidden={pc['config']['hidden_size']}, "
                 f"history={pc['config']['history_encoder']} "
                 f"({pc['config']['history_layers']}L/"
                 f"{pc['config']['history_heads']}H), "
                 f"role_dim={pc['config']['role_embedding_dim']}, "
                 f"mlp_layers={pc['config']['mlp_layers']}\n\n")
        fh.write("## Per-submodule parameter counts\n\n")
        fh.write("| submodule | parameters |\n|---|---|\n")
        for name, cnt in sorted(pc["submodules"].items()):
            fh.write(f"| {name} | {cnt:,} |\n")
        fh.write("\n## Forward-only latency (model only)\n\n")
        fh.write("| action count | median (ms) | mean (ms) | p95 (ms) |\n|---|---|---|---|\n")
        for key, vals in report["forward_only_ms"].items():
            if key == "full_action_set_count":
                continue
            fh.write(f"| {key} | {vals['median_ms']} | {vals['mean_ms']} | {vals['p95_ms']} |\n")
        if "deep_agent_act_ms" in report and "median_ms" in report.get("deep_agent_act_ms", {}):
            fh.write("\n## End-to-end DeepAgentV2.act\n\n")
            da = report["deep_agent_act_ms"]
            fh.write(f"- median: {da['median_ms']} ms, mean: {da['mean_ms']} ms, "
                     f"p95: {da['p95_ms']} ms\n")

    print(f"V2 benchmark written to {out_path} and {md_path}")
    print(f"Total parameters: {report['parameter_count']['total']:,}")
    print("Forward-only medians (ms):")
    for key, vals in report["forward_only_ms"].items():
        if key == "full_action_set_count":
            continue
        print(f"  {key}: {vals['median_ms']}")


if __name__ == "__main__":
    main()
