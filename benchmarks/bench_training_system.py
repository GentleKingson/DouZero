#!/usr/bin/env python
"""P14 CPU/GPU training-system profiler and comparison benchmark.

The report separates environment, observation, model, queue, learner, and
snapshot-publication timings. Unsupported hardware paths are reported as
``not_run`` rather than filled with estimates.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from douzero.dmc.models import Model, model_dict
from douzero.dmc.models_factorized import factorized_model_dict
from douzero.env.env import Env, get_obs, get_obs_factorized
from douzero.models_v2 import ModelV2, ModelV2Config, observation_to_model_inputs
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.runtime import SafeMixedPrecision, VersionedPolicyPool
from douzero.training.v2_buffer import (
    CompactTensorReplayBuffer,
    CompactTensorTransition,
    Transition,
)


def _measure(fn, rounds: int, warmup: int = 1) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "rounds": rounds,
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(p95, 4),
        "mean_ms": round(statistics.fmean(values), 4),
    }


def _decision(seed: int = 1400):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    while len(env.infoset.legal_actions) < 2:
        env.step(env.infoset.legal_actions[0])
    return env


def build_report(rounds: int) -> dict:
    torch.manual_seed(1400)
    env = _decision()
    obs = get_obs(env.infoset)
    z = torch.from_numpy(obs["z_batch"]).float()
    x = torch.from_numpy(obs["x_batch"]).float()
    position = env._acting_player_position

    legacy = model_dict[position]().eval()
    factorized = factorized_model_dict[position]().eval()
    factorized.load_state_dict(legacy.state_dict())
    split = get_obs_factorized(env.infoset)
    z_single = torch.from_numpy(split["z_single"]).float()
    x_state = torch.from_numpy(split["x_state_single"]).float()
    x_action = torch.from_numpy(split["x_action"]).float()

    schema = build_v2_schema()
    v2 = ModelV2(schema, ModelV2Config()).eval()
    v2_obs = get_obs_v2(env.infoset)
    v2_bundle = observation_to_model_inputs(v2_obs)

    def legacy_forward():
        with torch.inference_mode():
            legacy(z, x, return_value=True)

    def factorized_forward():
        with torch.inference_mode():
            factorized.forward_factorized(z_single, x_state, x_action, return_value=True)

    def v2_forward():
        with torch.inference_mode():
            v2(
                v2_bundle.state_card_vectors, v2_bundle.state_context_flat,
                v2_bundle.context_card_vectors, v2_bundle.context_flat,
                v2_bundle.history_tokens, v2_bundle.history_key_padding_mask,
                v2_bundle.action_features, v2_bundle.action_mask,
                v2_bundle.acting_role,
            )

    amp = SafeMixedPrecision(torch.device("cpu"), enabled=True, dtype="bfloat16")

    def v2_amp_forward():
        with torch.inference_mode(), amp.autocast():
            v2(
                v2_bundle.state_card_vectors, v2_bundle.state_context_flat,
                v2_bundle.context_card_vectors, v2_bundle.context_flat,
                v2_bundle.history_tokens, v2_bundle.history_key_padding_mask,
                v2_bundle.action_features, v2_bundle.action_mask,
                v2_bundle.acting_role,
            )

    train_model = model_dict[position]()
    optimizer = torch.optim.RMSprop(train_model.parameters(), lr=1e-4)
    target = torch.zeros(z.shape[0])
    fp32 = SafeMixedPrecision(torch.device("cpu"), enabled=False)

    def learner_step():
        fp32.step(
            lambda: ((train_model(z, x, return_value=True)["values"].squeeze(-1)
                      - target) ** 2).mean(),
            optimizer, train_model.parameters(), max_grad_norm=40.0,
        )

    actor_slots = [Model(device="cpu"), Model(device="cpu")]
    source = Model(device="cpu")
    import multiprocessing as multiprocessing
    pool = VersionedPolicyPool(actor_slots,
                               mp_context=multiprocessing.get_context("spawn"))
    pool.initialize(source.get_models())
    sync_version = 0

    def weight_sync():
        nonlocal sync_version
        sync_version += 1
        if not pool.publish(source.get_models(), version=sync_version):
            raise RuntimeError("benchmark snapshot slot unexpectedly busy")

    q = queue.Queue()
    q.put(1)

    def queue_wait():
        item = q.get()
        q.put(item)

    def observation_encoding():
        get_obs(env.infoset)

    step_env = _decision(1401)

    def environment_step():
        nonlocal step_env
        _obs, _reward, done, _info = step_env.step(step_env.infoset.legal_actions[0])
        if done:
            step_env = _decision(1401)

    replay_record = CompactTensorTransition.from_transition(Transition(
        obs=v2_obs,
        action_index=0,
        position=v2_obs.public.acting_role,
        target_win=1.0,
        target_score=1.0,
        target_log_score=0.0,
    ))
    compact_replay = CompactTensorReplayBuffer(capacity_transitions=4096)
    compact_replay.add_many([replay_record] * 4032)

    def compact_replay_ingest_near_capacity():
        compact_replay.add_many([replay_record] * 64)

    ddp_status = (
        "available_via_torchrun; not_run_in_single_process_benchmark"
        if torch.distributed.is_available()
        else "unavailable_in_torch_build"
    )
    return {
        "schema_version": "p14-training-system-v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "profiler_ms": {
            "actor_env_step": _measure(environment_step, rounds),
            "observation_encoding": _measure(observation_encoding, rounds),
            "queue_wait": _measure(queue_wait, rounds),
            "learner_forward_backward_step": _measure(learner_step, rounds),
            "weight_sync": _measure(weight_sync, rounds),
            "compact_replay_ingest_64_near_capacity": _measure(
                compact_replay_ingest_near_capacity, rounds
            ),
        },
        "forward_comparison_ms": {
            "legacy_fp32": _measure(legacy_forward, rounds),
            "factorized_fp32": _measure(factorized_forward, rounds),
            "v2_fp32": _measure(v2_forward, rounds),
            "v2_cpu_bfloat16_amp": _measure(v2_amp_forward, rounds),
        },
        "ddp": {"status": ddp_status, "measured": False},
        "cuda_amp": {"status": "not_run_cpu_benchmark", "measured": False},
    }


def _markdown(report: dict) -> str:
    lines = ["# P14 Training System Benchmark", "", "All timings are measured; unavailable paths are marked not run.", ""]
    for group in ("profiler_ms", "forward_comparison_ms"):
        lines.extend([f"## {group}", "", "| Path | Median ms | p95 ms |", "|---|---:|---:|"])
        for name, value in report[group].items():
            lines.append(f"| {name} | {value['median_ms']} | {value['p95_ms']} |")
        lines.append("")
    lines.append(f"DDP: {report['ddp']['status']}")
    lines.append(f"CUDA AMP: {report['cuda_amp']['status']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile P14 training-system paths")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="artifacts/benchmark/p14_training_system.json")
    args = parser.parse_args()
    if args.rounds < 1:
        raise ValueError("rounds must be >= 1")
    report = build_report(args.rounds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
