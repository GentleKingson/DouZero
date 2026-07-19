#!/usr/bin/env python
"""Build deterministic and measured Standard V2 R1 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.standard_v2_reference import build_standard_v2_reference
from douzero.training.standard_v2_contract import (
    STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
    STANDARD_V2_R1_CONFIG_HASH,
    STANDARD_V2_R1_CONTRACT_VERSION,
)


def _load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark input {path!r} must contain a JSON object")
    return payload


def _load_last_cycle(path: str) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cycles = [record for record in records if record.get("event") == "cycle"]
    if not cycles:
        raise ValueError("cycle metrics input contains no cycle event")
    return cycles[-1]


def build_unified_benchmark(
    *,
    training_metrics: dict[str, Any] | None = None,
    cycle_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the golden digest with a stable, hardware-neutral metric shape."""

    reference = build_standard_v2_reference()
    standard = dict((training_metrics or {}).get("standard_v2", {}))
    counts = dict(standard.get("counts", {}))
    rates = dict(standard.get("rates", {}))
    queue = dict(standard.get("queue_latency_ms", {}))
    gpu_seconds = dict(standard.get("gpu_seconds", {}))
    staging_seconds = standard.get("staging_seconds")
    peak_vram_mib = standard.get("peak_vram_mib")

    if cycle_metrics is not None:
        counts.update({
            "games": cycle_metrics.get("cycle_games"),
            "cardplay_decisions": cycle_metrics.get("cycle_cardplay_decisions"),
            "bidding_decisions": cycle_metrics.get("cycle_bidding_decisions"),
            "play_transitions": cycle_metrics.get("cycle_play_transitions"),
            "bid_transitions": cycle_metrics.get("cycle_bid_transitions"),
            "abandoned_bidding_transitions": cycle_metrics.get(
                "cycle_abandoned_bidding_transitions"
            ),
            "learner_cardplay_samples": cycle_metrics.get(
                "cycle_learner_cardplay_samples"
            ),
            "learner_bidding_samples": cycle_metrics.get(
                "cycle_learner_bidding_samples"
            ),
            "learner_samples": cycle_metrics.get("cycle_learner_samples"),
        })
        rates.update({
            "games_per_second": cycle_metrics.get("games_per_second"),
            "cardplay_decisions_per_second": cycle_metrics.get(
                "cardplay_decisions_per_second"
            ),
            "bidding_decisions_per_second": cycle_metrics.get(
                "bidding_decisions_per_second"
            ),
            "play_transitions_per_second": cycle_metrics.get(
                "transitions_per_second"
            ),
            "bid_transitions_per_second": cycle_metrics.get(
                "bid_transitions_per_second"
            ),
            "learner_steps_per_second": cycle_metrics.get(
                "learner_steps_per_second"
            ),
            "learner_samples_per_second": cycle_metrics.get(
                "learner_samples_per_second"
            ),
        })
        queue = {
            "p50": cycle_metrics.get("inference_queue_p50_ms"),
            "p95": cycle_metrics.get("inference_queue_p95_ms"),
        }
        gpu_seconds = {
            "inference": cycle_metrics.get("inference_gpu_seconds"),
            "learner": cycle_metrics.get("learner_gpu_seconds"),
        }
        staging_seconds = cycle_metrics.get("staging_seconds")
        peak_vram_bytes = cycle_metrics.get("peak_vram_bytes")
        peak_vram_mib = (
            round(float(peak_vram_bytes) / (1024.0 * 1024.0), 3)
            if peak_vram_bytes is not None else None
        )

    required_counts = (
        "games",
        "cardplay_decisions",
        "bidding_decisions",
        "play_transitions",
        "bid_transitions",
        "abandoned_bidding_transitions",
        "learner_cardplay_samples",
        "learner_bidding_samples",
        "learner_samples",
        "learner_steps",
    )
    required_rates = (
        "games_per_second",
        "cardplay_decisions_per_second",
        "bidding_decisions_per_second",
        "play_transitions_per_second",
        "bid_transitions_per_second",
        "learner_samples_per_second",
        "learner_steps_per_second",
    )
    measured = training_metrics is not None or cycle_metrics is not None
    measurement = {
        "source_schema_version": (
            (training_metrics or {}).get("schema_version")
            or (cycle_metrics or {}).get("schema_version")
        ),
        "device_type": (training_metrics or {}).get("device_type"),
        "training_wall_seconds": (training_metrics or {}).get(
            "training_wall_seconds"
        ),
        "phase_wall_seconds": standard.get("wall_seconds"),
        "amp": (training_metrics or {}).get("amp"),
        "compile": (training_metrics or {}).get("compile"),
        "distributed": (training_metrics or {}).get("distributed"),
        "parameter_update_observed": (training_metrics or {}).get(
            "parameter_update_observed"
        ),
    }
    return {
        "schema_version": STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": STANDARD_V2_R1_CONFIG_HASH,
        "reference_digest": reference["reference_digest"],
        "coverage": reference["coverage"],
        "performance": {
            "status": "measured" if measured else "not_run",
            "measurement": measurement,
            "counts": {name: counts.get(name) for name in required_counts},
            "rates": {name: rates.get(name) for name in required_rates},
            "queue_latency_ms": {
                "p50": queue.get("p50"),
                "p95": queue.get("p95"),
            },
            "gpu_seconds": {
                "inference": gpu_seconds.get("inference"),
                "learner": gpu_seconds.get("learner"),
            },
            "staging_seconds": staging_seconds,
            "peak_vram_mib": peak_vram_mib,
        },
        "privacy": "sanitized_no_host_or_device_identifiers",
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metrics", default="")
    parser.add_argument("--cycle-metrics", default="")
    parser.add_argument("--reference-output", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    training = _load_json(args.training_metrics) if args.training_metrics else None
    cycle = _load_last_cycle(args.cycle_metrics) if args.cycle_metrics else None
    if args.reference_output:
        _write_json(args.reference_output, build_standard_v2_reference())
    _write_json(
        args.output,
        build_unified_benchmark(training_metrics=training, cycle_metrics=cycle),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
