"""Loss-only H2 Adaptive DMC microbenchmark; not a training-speed claim."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from douzero.v3_hybrid.adaptive_dmc import (
    ADMC_DISABLED,
    ADMC_PAPER_RATIO,
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    adaptive_dmc_loss,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    mode: str,
    batch_size: int,
    *,
    device: torch.device,
    warmup: int,
    steps: int,
    repeats: int,
    target_transform: str,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(20260721)
    q_old = torch.randn(batch_size, generator=generator, device=device)
    q_old[::8] *= 1e-5
    q_seed = q_old + 0.25 * torch.randn(
        batch_size, generator=generator, device=device
    )
    returns = 8.0 * torch.randn(batch_size, generator=generator, device=device)
    config = AdaptiveDMCConfig(mode=mode)

    def step() -> None:
        q_new = q_seed.detach().clone().requires_grad_(True)
        result = adaptive_dmc_loss(
            q_new,
            returns,
            config=config,
            target_transform=target_transform,
            target_clamp=32.0,
            learner_update=5_000,
            q_old=None if mode == ADMC_DISABLED else q_old,
        )
        result.loss_per_sample.mean().backward()

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    elapsed = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        for _ in range(steps):
            step()
        _sync(device)
        elapsed.append(time.perf_counter() - started)
    median_seconds = statistics.median(elapsed)
    return {
        "mode": mode,
        "batch_size": batch_size,
        "repeat_seconds": elapsed,
        "median_steps_per_second": steps / median_seconds,
        "median_samples_per_second": steps * batch_size / median_seconds,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Loss-only H2 ADMC forward/backward microbenchmark; not an "
            "end-to-end learner, throughput, or playing-strength claim."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="32,256,2048")
    parser.add_argument(
        "--modes",
        default=f"{ADMC_DISABLED},{ADMC_PAPER_RATIO},{ADMC_SAFE_HYBRID}",
    )
    parser.add_argument("--target-transform", choices=("raw", "signed_log"), default="raw")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    batches = [int(value) for value in args.batch_sizes.split(",")]
    modes = args.modes.split(",")
    if any(value < 1 for value in batches):
        raise ValueError("batch sizes must be positive")
    if args.warmup < 0 or args.steps < 1 or args.repeats < 1:
        raise ValueError("warmup must be non-negative; steps/repeats must be positive")

    rows = [
        _run(
            mode,
            batch_size,
            device=device,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            target_transform=args.target_transform,
        )
        for batch_size in batches
        for mode in modes
    ]
    ordinary = {
        row["batch_size"]: row["median_samples_per_second"]
        for row in rows
        if row["mode"] == ADMC_DISABLED
    }
    for row in rows:
        baseline = ordinary.get(row["batch_size"])
        row["samples_per_second_vs_disabled"] = (
            None
            if baseline is None
            else row["median_samples_per_second"] / baseline
        )
    payload = {
        "scope": "loss_only_forward_backward_microbenchmark",
        "end_to_end_training_claim": False,
        "playing_strength_measured": False,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "target_transform": args.target_transform,
        "warmup": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "results": rows,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="ascii") as handle:
            handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
