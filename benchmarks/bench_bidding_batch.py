"""Benchmark the bidding head, scalar inference, and full learner hot path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from douzero._version import git_sha
from douzero.coach.records import CANONICAL_DECK
from douzero.env.rules import RuleSet
from douzero.models_v2 import BatchedBiddingInput, ModelV2, ModelV2Config
from douzero.models_v2.batch import bidding_observations_to_model_input
from douzero.observation.bidding import get_bidding_obs_v2
from douzero.observation.schema import build_v2_schema
from douzero.training.bidding import BiddingMinibatch, BiddingTransition, bidding_loss


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


def _wall_elapsed_ms(device: torch.device, operation) -> float:
    """Measure host-visible latency, including tensorization, copies, and syncs."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _summary(samples: list[float], iterations: int) -> dict:
    ordered = sorted(samples)
    # Nearest-rank percentile: p95 is the 95th ordered sample for n=100.
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "iterations": iterations,
    }


def _bidding_observations_and_minibatch(batch_size: int):
    ruleset = RuleSet.standard()
    histories = ((), (("0", 1),), (("0", 1), ("1", 2)))
    observations = []
    transitions = []
    sources = ("rule", "learned", "epsilon_random")
    for index in range(batch_size):
        history = histories[index % len(histories)]
        order = ("0", "1", "2")
        highest = max((bid for _, bid in history), default=0)
        raw = {
            "phase": "bidding",
            "position": order[len(history)],
            "my_handcards": list(CANONICAL_DECK[:17]),
            "current_highest_bid": highest,
            "bidding_history": list(history),
            "bidding_order": list(order),
            "first_bidder": order[0],
            "legal_bids": [
                bid for bid in ruleset.bid_values if bid == 0 or bid > highest
            ],
        }
        observation = get_bidding_obs_v2(raw, ruleset=ruleset)
        action = min(bid for bid in observation.legal_bids if bid > highest)
        transition = BiddingTransition(
            obs=observation,
            bid_action=action,
            policy_version="benchmark",
            source_policy=sources[index % len(sources)],
        )
        others = [seat for seat in order if seat != observation.current_seat]
        transition.assign_actor_role(
            {
                observation.current_seat: "landlord",
                others[0]: "landlord_down",
                others[1]: "landlord_up",
            }
        )
        transition.label_from_terminal(
            {
                "team_targets": {
                    "landlord": {"target_win": 1.0, "target_score": 2.0},
                    "landlord_down": {"target_win": 0.0, "target_score": -1.0},
                    "landlord_up": {"target_win": 0.0, "target_score": -1.0},
                }
            }
        )
        observations.append(observation)
        transitions.append(transition)
    return observations, BiddingMinibatch(transitions)


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
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    width = model.bidding_schema.input_width
    schema_hash = model.bidding_schema.stable_hash()
    results = []
    for batch_size in batch_sizes:
        model.load_state_dict(initial_state, strict=True)
        model.train()
        generator = torch.Generator(device=target).manual_seed(batch_size)
        inputs = BatchedBiddingInput(
            features=torch.randn(
                batch_size, width, generator=generator, device=target
            ),
            legal_mask=torch.ones(batch_size, 4, dtype=torch.bool, device=target),
            feature_schema_hash=schema_hash,
        )

        def head_forward_backward() -> None:
            model.zero_grad(set_to_none=True)
            output = model.forward_bidding_batched(inputs)
            loss = (
                output.bid_logits.float().square().mean()
                + output.landlord_win_logit.float().square().mean()
                + output.expected_landlord_score.float().square().mean()
            )
            loss.backward()

        for _ in range(warmup):
            head_forward_backward()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        head_samples = [
            _elapsed_ms(target, head_forward_backward) for _ in range(iterations)
        ]

        observations, minibatch = _bidding_observations_and_minibatch(batch_size)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

        def learner_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            learner_inputs = bidding_observations_to_model_input(observations)
            learner_output = model.forward_bidding_batched(learner_inputs)
            learner_targets = minibatch.to_targets(
                learner_output.bid_logits.device,
                dtype=learner_output.bid_logits.dtype,
            )
            components = bidding_loss(
                learner_output,
                learner_targets,
                lambda_policy=1.0,
                lambda_landlord_win=0.5,
                lambda_landlord_score=0.25,
            )
            components.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 40.0, error_if_nonfinite=True
            )
            optimizer.step()
            components.as_log_dict()

        for _ in range(warmup):
            learner_step()
        learner_samples = [
            _wall_elapsed_ms(target, learner_step) for _ in range(iterations)
        ]
        results.append({
            "batch_size": batch_size,
            "head_forward_backward": _summary(head_samples, iterations),
            "learner_step_wall": _summary(learner_samples, iterations),
        })

    model.load_state_dict(initial_state, strict=True)
    model.eval()
    scalar_observation = _bidding_observations_and_minibatch(1)[0][0]

    def scalar_fast_forward() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type=target.type, enabled=False
        ):
            model.forward_bidding(scalar_observation)

    def scalar_fast_decision() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type=target.type, enabled=False
        ):
            model.forward_bidding(scalar_observation).argmax_bid()

    def scalar_batched_forward() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type=target.type, enabled=False
        ):
            model.forward_bidding_batched(
                bidding_observations_to_model_input((scalar_observation,))
            ).select(0)

    def scalar_batched_decision() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type=target.type, enabled=False
        ):
            model.forward_bidding_batched(
                bidding_observations_to_model_input((scalar_observation,))
            ).select(0).argmax_bid()

    for _ in range(warmup):
        scalar_fast_forward()
        scalar_fast_decision()
        scalar_batched_forward()
        scalar_batched_decision()
    scalar_inference = {
        "fast_forward_wall": _summary(
            [
                _wall_elapsed_ms(target, scalar_fast_forward)
                for _ in range(iterations)
            ],
            iterations,
        ),
        "fast_decision_wall": _summary(
            [
                _wall_elapsed_ms(target, scalar_fast_decision)
                for _ in range(iterations)
            ],
            iterations,
        ),
        "batched_wrapper_forward_wall": _summary(
            [
                _wall_elapsed_ms(target, scalar_batched_forward)
                for _ in range(iterations)
            ],
            iterations,
        ),
        "batched_wrapper_decision_wall": _summary(
            [
                _wall_elapsed_ms(target, scalar_batched_decision)
                for _ in range(iterations)
            ],
            iterations,
        ),
    }

    device_metadata = None
    if target.type == "cuda":
        properties = torch.cuda.get_device_properties(target)
        device_metadata = {
            "name": properties.name,
            "total_memory_mib": properties.total_memory / (1024 * 1024),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "schema_version": "standard-v2-bidding-microbenchmark-v3",
        "source_git_sha": git_sha(),
        "device": str(target),
        "device_metadata": device_metadata,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inference_mode": "eval_inference_mode_fp32",
        "batch_sizes": list(batch_sizes),
        "results": results,
        "scalar_inference": scalar_inference,
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
