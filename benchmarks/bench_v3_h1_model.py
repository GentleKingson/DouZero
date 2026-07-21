#!/usr/bin/env python3
"""Model-only H1 ablation benchmark; not an end-to-end training claim."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time

import numpy as np
import torch

from douzero.env.env import Env
from douzero.models_v2 import ModelV2, ModelV2Config, observation_to_model_inputs
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.v3_hybrid import CHANNEL_GATE_SE, V3HybridModel, V3HybridModelConfig


def _bundle(actions: int, device: torch.device):
    np.random.seed(7)
    env = Env("adp")
    env.reset()
    bundle = observation_to_model_inputs(get_obs_v2(env.infoset)).to(device)
    source = bundle.action_features
    indices = torch.arange(actions, device=device) % source.shape[0]
    bundle.action_features = source.index_select(0, indices).clone()
    bundle.action_mask = torch.ones(actions, dtype=torch.bool, device=device)
    return bundle


def _call(model, bundle):
    return model(
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
        bundle.acting_role,
    )


def _measure(model, bundle, *, warmup: int, steps: int, repeats: int):
    device = next(model.parameters()).device
    records = []
    for repeat in range(repeats):
        model.train()
        for _ in range(warmup):
            output = _call(model, bundle)
            loss = getattr(output, "dmc_q", output.score_mean).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            output = _call(model, bundle)
            loss = getattr(output, "dmc_q", output.score_mean).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        records.append({
            "repeat": repeat,
            "elapsed_s": elapsed,
            "steps_per_s": steps / elapsed,
            "actions_per_s": steps * bundle.action_features.shape[0] / elapsed,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--action-buckets", default="8,32,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.hidden_size, args.warmup, args.steps, args.repeats) <= 0:
        parser.error("hidden-size, warmup, steps, and repeats must be positive")
    buckets = [int(value) for value in args.action_buckets.split(",")]
    if not buckets or any(value <= 0 for value in buckets):
        parser.error("action-buckets must contain positive comma-separated ints")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    schema = build_v2_schema()
    common = V3HybridModelConfig(
        hidden_size=args.hidden_size,
        history_layers=2,
        history_heads=8,
        nan_guard=False,
    )
    variants = {
        "model_v2": ModelV2(
            schema,
            ModelV2Config(
                hidden_size=args.hidden_size,
                history_encoder="lstm",
                history_layers=2,
                history_heads=8,
                nan_guard=False,
            ),
        ),
        "v3_shared_only": V3HybridModel(
            schema,
            dataclasses.replace(
                common,
                landlord_adapter_layers=0,
                farmer_adapter_layers=0,
            ),
        ),
        "v3_role_adapters": V3HybridModel(schema, common),
        "v3_farmer_channel_gate": V3HybridModel(
            schema,
            dataclasses.replace(common, farmer_channel_gate=CHANNEL_GATE_SE),
        ),
    }
    report = {
        "benchmark": "v3_h1_model_only",
        "claim_scope": "forward_backward_only_not_end_to_end_training",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "variants": {},
    }
    for name, model in variants.items():
        model.to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        rows = {}
        for actions in buckets:
            records = _measure(
                model,
                _bundle(actions, device),
                warmup=args.warmup,
                steps=args.steps,
                repeats=args.repeats,
            )
            rows[str(actions)] = {
                "records": records,
                "median_steps_per_s": statistics.median(
                    record["steps_per_s"] for record in records
                ),
                "median_actions_per_s": statistics.median(
                    record["actions_per_s"] for record in records
                ),
            }
        report["variants"][name] = {
            "parameters": parameters,
            "action_buckets": rows,
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
