#!/usr/bin/env python
"""CUDA-event capability benchmark for isolated gpu_v3 architectures."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from douzero.gpu_v3 import GPUV3Config, IndependentRoleDualTower, SharedTrunkRoleHeads
from douzero.gpu_v3.config import SHARED_TRUNK_ROLE_HEADS


def _measure(model, inputs, rounds, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered)) - 1)],
    }


def _inputs(batch, actions, state_width, action_width, device):
    return (
        torch.randn(batch, state_width, device=device),
        torch.randn(batch, actions, action_width, device=device),
        torch.ones(batch, actions, dtype=torch.bool, device=device),
        torch.arange(batch, device=device) % 3,
    )


def run(rounds=100, warmup=20):
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_v3 benchmark requires CUDA")
    device = torch.device("cuda:0")
    state_width, action_width = 1240, 54
    independent_config = GPUV3Config()
    shared_config = GPUV3Config(architecture=SHARED_TRUNK_ROLE_HEADS)
    models = {
        "independent_role_dual_tower": IndependentRoleDualTower(
            state_width, action_width, independent_config
        ).to(device).eval(),
        "shared_trunk_role_heads": SharedTrunkRoleHeads(
            state_width, action_width, shared_config
        ).to(device).eval(),
    }
    results = {}
    for name, model in models.items():
        model_results = {"parameters": model.parameter_count(), "cases": {}}
        for batch, actions in ((1, 64), (16, 256), (64, 512)):
            timing = _measure(
                model,
                _inputs(batch, actions, state_width, action_width, device),
                rounds,
                warmup,
            )
            timing["decisions_per_second"] = batch / (timing["median_ms"] / 1000)
            model_results["cases"][f"batch={batch},actions={actions}"] = timing
        results[name] = model_results
    return {
        "schema_version": "gpu-v3-capability-v1",
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "rounds": rounds,
        "warmup": warmup,
        "results": results,
    }


def _markdown(report):
    lines = [
        "# gpu_v3 CUDA capability benchmark",
        "",
        f"GPU: `{report['environment']['gpu']}`; torch: "
        f"`{report['environment']['torch']}`; CUDA: `{report['environment']['cuda']}`.",
        "",
        "| architecture | parameters | case | median ms | p95 ms | decisions/s |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, model in report["results"].items():
        for case, timing in model["cases"].items():
            lines.append(
                f"| {name} | {model['parameters']} | {case} | "
                f"{timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | "
                f"{timing['decisions_per_second']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", default="artifacts/gpu-v3/benchmark.json")
    args = parser.parse_args()
    report = run(args.rounds, args.warmup)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
