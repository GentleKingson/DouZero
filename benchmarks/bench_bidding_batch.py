"""Microbenchmark the vectorized Standard V2 bidding forward/backward path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from douzero.models_v2 import BatchedBiddingInput, ModelV2, ModelV2Config
from douzero.observation.schema import build_v2_schema


def _elapsed_ms(device: torch.device, operation) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    started = time.perf_counter()
    operation()
    return (time.perf_counter() - started) * 1000.0


def run_benchmark(
    *,
    device: str,
    batch_sizes: tuple[int, ...] = (1, 32, 64, 128),
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA bidding benchmark requested but CUDA is unavailable")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")

    torch.manual_seed(0)
    model = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            hidden_size=256,
            history_layers=4,
            history_heads=8,
            bidding_enabled=True,
            bidding_hidden_size=128,
            nan_guard=True,
        ),
    ).to(target)
    model.train()
    width = model.bidding_schema.input_width
    schema_hash = model.bidding_schema.stable_hash()
    results = []
    for batch_size in batch_sizes:
        generator = torch.Generator(device=target).manual_seed(batch_size)
        inputs = BatchedBiddingInput(
            features=torch.randn(
                batch_size, width, generator=generator, device=target
            ),
            legal_mask=torch.ones(batch_size, 4, dtype=torch.bool, device=target),
            feature_schema_hash=schema_hash,
        )

        def forward_backward() -> None:
            model.zero_grad(set_to_none=True)
            output = model.forward_bidding_batched(inputs)
            loss = (
                output.bid_logits.float().square().mean()
                + output.landlord_win_logit.float().square().mean()
                + output.expected_landlord_score.float().square().mean()
            )
            loss.backward()

        for _ in range(warmup):
            forward_backward()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        samples = [
            _elapsed_ms(target, forward_backward) for _ in range(iterations)
        ]
        ordered = sorted(samples)
        results.append({
            "batch_size": batch_size,
            "mean_ms": statistics.fmean(samples),
            "p50_ms": statistics.median(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "iterations": iterations,
        })

    return {
        "schema_version": "standard-v2-bidding-microbenchmark-v1",
        "device": str(target),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_sizes": list(batch_sizes),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = run_benchmark(
        device=args.device,
        batch_sizes=tuple(args.batch_sizes),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
