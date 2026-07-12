#!/usr/bin/env python
"""Micro-benchmark: Model V2 forward latency and parameter report (P05).

Measures CPU latency for the V2 shared state-action model:

  * parameter count (per submodule + total) for the default config;
  * model-forward-only latency at several legal-action counts (1, 10, 50,
    full action set) — the per-decision cost as the candidate set grows;
  * full ``DeepAgentV2.act`` latency (observation encoding + tensor build +
    forward + selection) for the default config, measured with the
    environment prepared ONCE outside the timed loop.

This is a MEASUREMENT tool, not a strength or speedup claim. It reports honest
medians and p95s on the current host and makes no preset assumption about the
relative cost. The model-forward-only numbers are NOT end-to-end act numbers;
both are reported separately and labelled clearly. A legacy-factorized
comparison is NOT included here; it lives in ``bench_factorized.py``.

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


def _landlord_env(seed: int, steps_into_game: int):
    """Reset and pre-roll an ``Env("adp")`` to a landlord decision point.

    Shared by the forward-only path (via :func:`_make_v2_obs`) and the
    end-to-end DeepAgent path (:func:`bench_deep_agent_act`) so the two latency
    numbers are measured at the SAME (non-degenerate) decision point with a REAL
    history — not an opening move whose empty history would understate the cost.

    Fail-closed: ``steps_into_game`` must stay within one game's length. If a
    pre-roll step ends the game (the ``done`` flag returned by ``env.step``), the
    requested depth exceeds the game and a ``ValueError`` is raised with a clear
    message — the benchmark exits loudly rather than falling through to step a
    terminal game and crash cryptically. (Same fail-closed contract blocker #3
    established for the deep-agent path.) The default ``--steps_into_game=4`` is
    well within a full game.
    """
    import numpy as np
    from douzero.env.env import Env

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(steps_into_game):
        _obs, _reward, done, _info = env.step(env.infoset.legal_actions[0])
        if done:
            raise ValueError(
                f"--steps_into_game={steps_into_game} reaches/passes the end of "
                f"the game; reduce it to stay within one game's length."
            )
    while env._acting_player_position != "landlord":
        _obs, _reward, done, _info = env.step(env.infoset.legal_actions[0])
        if done:
            raise ValueError(
                "no landlord turn is reachable before the game ends at the "
                "requested depth; reduce --steps_into_game."
            )
    return env


def _make_v2_obs(seed: int = 42, steps_into_game: int = 0):
    """Build a real ObservationV2 at the landlord's turn for benchmarking."""
    from douzero.observation import get_obs_v2

    env = _landlord_env(seed, steps_into_game)
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


def bench_deep_agent_act(agent, env_seed, rounds, warmup, steps_into_game=0):
    """Full DeepAgentV2.act(infoset) latency (encode + forward + select).

    The environment is prepared ONCE, outside the timed loop, so the reported
    per-decision number covers ONLY what ``DeepAgentV2.act`` does on each call:
    ``get_obs_v2`` (observation encode, including ``ObservationV2.__post_init__``
    schema/alignment validation) + tensor build + forward + argmax. Env
    reset/deal/pre-roll cost is reported separately as ``environment_setup_ms``
    so a reader cannot mistake an env-construction cost for an agent latency.

    Fail-closed: if the pre-rolled decision point has fewer than 2 legal
    actions, ``act`` would short-circuit on a single action and skip inference,
    so the measurement would NOT reflect a model decision — raise loudly
    rather than report a degenerate (single-action) timing.
    """
    import time as _time

    t_setup0 = _time.perf_counter()
    env = _landlord_env(env_seed, steps_into_game)
    infoset = env.infoset
    environment_setup_ms = (_time.perf_counter() - t_setup0) * 1000.0

    if len(infoset.legal_actions) <= 1:
        raise ValueError(
            f"benchmark decision point has {len(infoset.legal_actions)} legal "
            f"action(s); need >= 2 so act() exercises model inference rather "
            f"than the single-action short-circuit. Try a different env_seed "
            f"or steps_into_game."
        )

    def _fn():
        agent.act(infoset)

    result = _bench(_fn, rounds, warmup)
    result["environment_setup_ms"] = round(environment_setup_ms, 4)
    return result


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
        agent, env_seed=42, rounds=args.rounds,
        warmup=max(args.warmup, 1), steps_into_game=args.steps_into_game,
    )
    report["deep_agent_act_ms"]["note"] = (
        "Per-decision act(infoset) path ONLY: get_obs_v2 + tensor build + "
        "forward + argmax. The env is prepared ONCE outside the timed loop "
        "(see environment_setup_ms); each iteration re-invokes act() on the "
        "same infoset."
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
            fh.write(f"- environment setup (one-time, outside timed loop): "
                     f"{da.get('environment_setup_ms', 'n/a')} ms\n")

    print(f"V2 benchmark written to {out_path} and {md_path}")
    print(f"Total parameters: {report['parameter_count']['total']:,}")
    print("Forward-only medians (ms):")
    for key, vals in report["forward_only_ms"].items():
        if key == "full_action_set_count":
            continue
        print(f"  {key}: {vals['median_ms']}")


if __name__ == "__main__":
    main()
