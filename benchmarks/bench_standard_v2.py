#!/usr/bin/env python
"""Build deterministic and measured Standard V2 R1 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from benchmarks.standard_v2_reference import build_standard_v2_reference
from douzero.training.standard_v2_contract import (
    STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
    STANDARD_V2_R1_BENCHMARK_WORKLOAD_HASH,
    STANDARD_V2_R1_CONFIG_HASH,
    STANDARD_V2_R1_CONTRACT_VERSION,
    STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD,
    STANDARD_V2_R1_REFERENCE_DIGEST,
    STANDARD_V2_R1_TRAINING_SEMANTICS_HASH,
)


_TRAINING_METRICS_SCHEMA_VERSION = "p17-gpu-run-v1"
_CYCLE_METRICS_SCHEMA_VERSION = "v2-long-running-cycle-v2"
_IDENTITY_FIELDS = frozenset({
    "schema_version",
    "contract_version",
    "config_hash",
    "training_semantics_hash",
    "benchmark_workload_hash",
    "reference_digest",
    "qualification",
})
_COUNT_FIELDS = (
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
_RATE_FIELDS = (
    "games_per_second",
    "cardplay_decisions_per_second",
    "bidding_decisions_per_second",
    "play_transitions_per_second",
    "bid_transitions_per_second",
    "learner_samples_per_second",
    "learner_steps_per_second",
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


def _require_mapping(
    value: object,
    name: str,
    *,
    fields: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    payload = dict(value)
    if fields is not None and set(payload) != set(fields):
        missing = sorted(set(fields) - set(payload))
        extra = sorted(set(payload) - set(fields))
        raise ValueError(
            f"{name} has an invalid field set: missing={missing}, extra={extra}"
        )
    return payload


def _require_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
    ):
        qualifier = "positive finite" if positive else "finite and non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _require_optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _require_number(value, name)


def _require_count(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (positive and value <= 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _validate_identity(value: object, name: str) -> dict[str, Any]:
    identity = _require_mapping(value, name, fields=_IDENTITY_FIELDS)
    expected = {
        "schema_version": STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": STANDARD_V2_R1_CONFIG_HASH,
        "training_semantics_hash": STANDARD_V2_R1_TRAINING_SEMANTICS_HASH,
        "benchmark_workload_hash": STANDARD_V2_R1_BENCHMARK_WORKLOAD_HASH,
        "reference_digest": STANDARD_V2_R1_REFERENCE_DIGEST,
        "qualification": "r1",
    }
    if identity != expected:
        raise ValueError(f"{name} does not identify the frozen Standard V2 R1 run")
    return identity


def _validate_metrics_history(value: object, name: str) -> dict[str, Any]:
    history = _require_mapping(value, name, fields={"complete", "source"})
    if history["complete"] is not True:
        raise ValueError(f"{name} must declare complete metric history")
    if not isinstance(history["source"], str) or not history["source"]:
        raise ValueError(f"{name}.source must be non-empty text")
    return history


def _validate_counts(value: object, name: str) -> dict[str, int]:
    raw = _require_mapping(value, name, fields=set(_COUNT_FIELDS))
    counts = {
        field: _require_count(
            raw[field],
            f"{name}.{field}",
            positive=field in {"games", "learner_samples", "learner_steps"},
        )
        for field in _COUNT_FIELDS
    }
    if counts["learner_samples"] != (
        counts["learner_cardplay_samples"]
        + counts["learner_bidding_samples"]
    ):
        raise ValueError("learner sample counters do not reconcile")
    if counts["bidding_decisions"] != (
        counts["bid_transitions"]
        + counts["abandoned_bidding_transitions"]
    ):
        raise ValueError("bidding decision counters do not reconcile")
    if counts["cardplay_decisions"] < counts["play_transitions"]:
        raise ValueError("cardplay decisions cannot be below play transitions")
    return counts


def _validate_rates(
    value: object,
    name: str,
    *,
    expected: Mapping[str, tuple[int, float]],
) -> dict[str, float]:
    raw = _require_mapping(value, name, fields=set(_RATE_FIELDS))
    rates: dict[str, float] = {}
    for field in _RATE_FIELDS:
        reported = _require_number(raw[field], f"{name}.{field}")
        count, seconds = expected[field]
        recomputed = float(count) / seconds
        canonical = round(recomputed, 6)
        if reported != canonical:
            raise ValueError(
                f"{name}.{field} disagrees with its count and wall time"
            )
        rates[field] = canonical
    return rates


def _validate_phase_wall(
    wall: Mapping[str, float],
    name: str,
    *,
    bound_overhead: bool,
) -> None:
    total = wall["total"]
    phase_sum = wall["collection"] + wall["optimization"]
    tolerance = max(2.0e-6, total * 1.0e-6)
    if phase_sum > total + tolerance:
        raise ValueError(f"{name} phases exceed total wall time")
    if bound_overhead and total > phase_sum * 1.05 + 0.1:
        raise ValueError(f"{name} total wall time has implausible overhead")


def _validate_amp(
    value: object,
    name: str,
    *,
    include_observation: bool,
) -> dict[str, Any]:
    fields = {"enabled", "dtype", "fallback_on_nonfinite"}
    if include_observation:
        fields.update({"fallback_count", "fallback_exercised"})
    amp = _require_mapping(value, name, fields=fields)
    if amp["enabled"] is not False or amp["dtype"] != "float16":
        raise ValueError(f"{name} does not match the frozen R1 AMP config")
    if amp["fallback_on_nonfinite"] is not True:
        raise ValueError(f"{name}.fallback_on_nonfinite must be true")
    if include_observation:
        fallback_count = _require_count(
            amp["fallback_count"], f"{name}.fallback_count"
        )
        if amp["fallback_exercised"] is not bool(fallback_count):
            raise ValueError(f"{name} fallback observation is inconsistent")
    return amp


def _validate_compile(value: object, name: str) -> dict[str, bool]:
    compile_info = _require_mapping(value, name, fields={"enabled"})
    if compile_info["enabled"] is not False:
        raise ValueError(f"{name} does not match the frozen R1 compile config")
    return compile_info


def _validate_distributed(value: object, name: str) -> dict[str, Any]:
    distributed = _require_mapping(
        value, name, fields={"enabled", "world_size"}
    )
    if distributed["enabled"] is not False or distributed["world_size"] != 1:
        raise ValueError(f"{name} must describe a non-DDP single-GPU run")
    return distributed


def _validate_optional_sections(
    *,
    queue: object,
    gpu_seconds: object,
    staging_seconds: object,
    prefix: str,
) -> tuple[dict[str, float | None], dict[str, float | None], float | None]:
    queue_payload = _require_mapping(
        queue, f"{prefix}.queue_latency_ms", fields={"p50", "p95"}
    )
    queue_result = {
        name: _require_optional_number(
            queue_payload[name], f"{prefix}.queue_latency_ms.{name}"
        )
        for name in ("p50", "p95")
    }
    if (
        queue_result["p50"] is not None
        and queue_result["p95"] is not None
        and queue_result["p95"] < queue_result["p50"]
    ):
        raise ValueError("queue p95 cannot be below p50")
    gpu_payload = _require_mapping(
        gpu_seconds, f"{prefix}.gpu_seconds", fields={"inference", "learner"}
    )
    gpu_result = {
        name: _require_optional_number(
            gpu_payload[name], f"{prefix}.gpu_seconds.{name}"
        )
        for name in ("inference", "learner")
    }
    staging = _require_optional_number(
        staging_seconds, f"{prefix}.staging_seconds"
    )
    return queue_result, gpu_result, staging


def _validate_training_metrics(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "device_type",
        "training_wall_seconds",
        "counts",
        "metrics",
        "benchmark_identity",
        "metrics_history",
        "standard_v2",
        "amp",
        "compile",
        "distributed",
        "parameter_update_observed",
        "privacy",
    }
    training = _require_mapping(payload, "training_metrics", fields=fields)
    if training["schema_version"] != _TRAINING_METRICS_SCHEMA_VERSION:
        raise ValueError("training_metrics has an unsupported schema_version")
    if training["status"] != "passed":
        raise ValueError("training_metrics status must be passed")
    if training["device_type"] != "cuda":
        raise ValueError("R1 measured baseline requires device_type='cuda'")
    if training["parameter_update_observed"] is not True:
        raise ValueError("R1 measured baseline requires an observed parameter update")
    if training["privacy"] != "sanitized_no_host_or_device_identifiers":
        raise ValueError("training_metrics privacy contract is invalid")
    training_wall = _require_number(
        training["training_wall_seconds"],
        "training_metrics.training_wall_seconds",
        positive=True,
    )
    identity = _validate_identity(
        training["benchmark_identity"], "training_metrics.benchmark_identity"
    )
    history = _validate_metrics_history(
        training["metrics_history"], "training_metrics.metrics_history"
    )
    amp = _validate_amp(
        training["amp"], "training_metrics.amp", include_observation=True
    )
    compile_info = _validate_compile(
        training["compile"], "training_metrics.compile"
    )
    distributed = _validate_distributed(
        training["distributed"], "training_metrics.distributed"
    )

    standard_fields = set(_IDENTITY_FIELDS) | {
        "wall_seconds",
        "counts",
        "rates",
        "queue_latency_ms",
        "gpu_seconds",
        "staging_seconds",
        "peak_vram_mib",
    }
    standard = _require_mapping(
        training["standard_v2"],
        "training_metrics.standard_v2",
        fields=standard_fields,
    )
    nested_identity = _validate_identity(
        {name: standard[name] for name in _IDENTITY_FIELDS},
        "training_metrics.standard_v2 identity",
    )
    if nested_identity != identity:
        raise ValueError("training metric identities disagree")
    counts = _validate_counts(
        standard["counts"], "training_metrics.standard_v2.counts"
    )
    frozen_trainer = STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD[
        "trainer_config"
    ]
    if counts["games"] != frozen_trainer["max_episodes"]:
        raise ValueError("game count disagrees with the frozen R1 workload")
    if counts["learner_steps"] != frozen_trainer["optimizer_steps"]:
        raise ValueError("learner steps disagree with the frozen R1 workload")
    wall = _require_mapping(
        standard["wall_seconds"],
        "training_metrics.standard_v2.wall_seconds",
        fields={"total", "collection", "optimization"},
    )
    wall = {
        name: _require_number(
            wall[name],
            f"training_metrics.standard_v2.wall_seconds.{name}",
            positive=True,
        )
        for name in ("total", "collection", "optimization")
    }
    if not math.isclose(wall["total"], training_wall, abs_tol=1.0e-6):
        raise ValueError("training wall time disagrees with Standard V2 metrics")
    _validate_phase_wall(
        wall,
        "training_metrics.standard_v2.wall_seconds",
        bound_overhead=True,
    )
    rates = _validate_rates(
        standard["rates"],
        "training_metrics.standard_v2.rates",
        expected={
            "games_per_second": (counts["games"], wall["collection"]),
            "cardplay_decisions_per_second": (
                counts["cardplay_decisions"],
                wall["collection"],
            ),
            "bidding_decisions_per_second": (
                counts["bidding_decisions"],
                wall["collection"],
            ),
            "play_transitions_per_second": (
                counts["play_transitions"],
                wall["collection"],
            ),
            "bid_transitions_per_second": (
                counts["bid_transitions"],
                wall["collection"],
            ),
            "learner_samples_per_second": (
                counts["learner_samples"],
                wall["optimization"],
            ),
            "learner_steps_per_second": (
                counts["learner_steps"],
                wall["optimization"],
            ),
        },
    )
    queue, gpu_seconds, staging = _validate_optional_sections(
        queue=standard["queue_latency_ms"],
        gpu_seconds=standard["gpu_seconds"],
        staging_seconds=standard["staging_seconds"],
        prefix="training_metrics.standard_v2",
    )
    peak_vram = _require_number(
        standard["peak_vram_mib"],
        "training_metrics.standard_v2.peak_vram_mib",
        positive=True,
    )
    top_metrics = _require_mapping(
        training["metrics"],
        "training_metrics.metrics",
        fields={
            "peak_memory_mib",
            "peak_reserved_memory_mib",
            "cardplay_transitions_per_second",
            "bidding_decisions_per_second",
            "samples_per_second",
            "decisions_per_second",
            "learner_steps_per_second",
        },
    )
    top_counts = _require_mapping(
        training["counts"],
        "training_metrics.counts",
        fields={
            "episodes",
            "games",
            "cardplay_decisions",
            "cardplay_transitions",
            "bidding_decisions",
            "bidding_transitions",
            "abandoned_bidding_transitions",
            "total_decisions",
            "learner_cardplay_samples",
            "learner_bidding_samples",
            "learner_samples",
            "learner_steps",
            "redeals",
            "max_redeals_exceeded",
            "belief_supervised_steps",
        },
    )
    for name, value in top_counts.items():
        _require_count(value, f"training_metrics.counts.{name}")
    if top_counts["episodes"] != top_counts["games"]:
        raise ValueError("completed episodes disagree with collected games")
    repeated_counts = {
        "games": "games",
        "cardplay_decisions": "cardplay_decisions",
        "bidding_decisions": "bidding_decisions",
        "cardplay_transitions": "play_transitions",
        "bidding_transitions": "bid_transitions",
        "abandoned_bidding_transitions": "abandoned_bidding_transitions",
        "learner_cardplay_samples": "learner_cardplay_samples",
        "learner_bidding_samples": "learner_bidding_samples",
        "learner_samples": "learner_samples",
        "learner_steps": "learner_steps",
    }
    for top_name, standard_name in repeated_counts.items():
        if top_counts[top_name] != counts[standard_name]:
            raise ValueError(f"{top_name} counters disagree")
    if top_counts["total_decisions"] != (
        top_counts["cardplay_decisions"]
        + top_counts["bidding_decisions"]
    ):
        raise ValueError("total decision counter does not reconcile")

    peak_memory = _require_number(
        top_metrics["peak_memory_mib"],
        "training_metrics.metrics.peak_memory_mib",
        positive=True,
    )
    peak_reserved = _require_number(
        top_metrics["peak_reserved_memory_mib"],
        "training_metrics.metrics.peak_reserved_memory_mib",
        positive=True,
    )
    if peak_memory != peak_vram:
        raise ValueError("peak VRAM disagrees with top-level training metrics")
    if peak_reserved < peak_memory:
        raise ValueError("reserved VRAM cannot be below allocated VRAM")
    top_rate_expectations = {
        "cardplay_transitions_per_second": (
            top_counts["cardplay_transitions"] / training_wall
        ),
        "bidding_decisions_per_second": (
            top_counts["bidding_decisions"] / training_wall
        ),
        "samples_per_second": (
            (
                top_counts["cardplay_transitions"]
                + top_counts["bidding_transitions"]
            )
            / training_wall
        ),
        "decisions_per_second": (
            top_counts["total_decisions"] / training_wall
        ),
        "learner_steps_per_second": (
            top_counts["learner_steps"] / training_wall
        ),
    }
    for name, expected_rate in top_rate_expectations.items():
        reported_rate = _require_number(
            top_metrics[name], f"training_metrics.metrics.{name}"
        )
        if reported_rate != round(expected_rate, 6):
            raise ValueError(
                f"training_metrics.metrics.{name} disagrees with counts and wall time"
            )

    return {
        "identity": identity,
        "counts": counts,
        "rates": rates,
        "queue_latency_ms": queue,
        "gpu_seconds": gpu_seconds,
        "staging_seconds": staging,
        "peak_vram_mib": peak_vram,
        "measurement": {
            "source_schema_version": training["schema_version"],
            "device_type": training["device_type"],
            "training_wall_seconds": training_wall,
            "phase_wall_seconds": wall,
            "amp": amp,
            "compile": compile_info,
            "distributed": distributed,
            "parameter_update_observed": True,
            "metrics_history": history,
        },
    }


def _validate_cycle_metrics(payload: object) -> dict[str, Any]:
    cycle = _require_mapping(payload, "cycle_metrics")
    required_fields = {
        "schema_version",
        "event",
        "benchmark_identity",
        "metrics_history",
        "device_type",
        "amp",
        "compile",
        "distributed",
        "parameter_update_observed",
        "total_episodes",
        "total_transitions",
        "total_optimizer_steps",
        "cycle_games",
        "cycle_cardplay_decisions",
        "cycle_bidding_decisions",
        "cycle_play_transitions",
        "cycle_bid_transitions",
        "cycle_abandoned_bidding_transitions",
        "cycle_learner_cardplay_samples",
        "cycle_learner_bidding_samples",
        "cycle_learner_samples",
        "cycle_learner_steps",
        "cycle_wall_seconds",
        "collection_seconds",
        "optimization_seconds",
        "games_per_second",
        "cardplay_decisions_per_second",
        "bidding_decisions_per_second",
        "transitions_per_second",
        "bid_transitions_per_second",
        "learner_samples_per_second",
        "learner_steps_per_second",
        "inference_queue_p50_ms",
        "inference_queue_p95_ms",
        "inference_gpu_seconds",
        "learner_gpu_seconds",
        "staging_seconds",
        "peak_vram_bytes",
        "amp_fallback",
    }
    missing = sorted(required_fields - set(cycle))
    if missing:
        raise ValueError(f"cycle_metrics is missing required fields: {missing}")
    if cycle["schema_version"] != _CYCLE_METRICS_SCHEMA_VERSION:
        raise ValueError("cycle_metrics has an unsupported schema_version")
    if cycle["event"] != "cycle":
        raise ValueError("cycle_metrics must contain a cycle event")
    if cycle["device_type"] != "cuda":
        raise ValueError("R1 measured baseline requires device_type='cuda'")
    if cycle["parameter_update_observed"] is not True:
        raise ValueError("R1 cycle requires an observed parameter update")
    identity = _validate_identity(
        cycle["benchmark_identity"], "cycle_metrics.benchmark_identity"
    )
    history = _validate_metrics_history(
        cycle["metrics_history"], "cycle_metrics.metrics_history"
    )
    amp = _validate_amp(cycle["amp"], "cycle_metrics.amp", include_observation=False)
    fallback_count = _require_count(
        cycle["amp_fallback"], "cycle_metrics.amp_fallback"
    )
    amp = {
        **amp,
        "fallback_count": fallback_count,
        "fallback_exercised": bool(fallback_count),
    }
    compile_info = _validate_compile(cycle["compile"], "cycle_metrics.compile")
    distributed = _validate_distributed(
        cycle["distributed"], "cycle_metrics.distributed"
    )
    count_source = {
        "games": cycle["cycle_games"],
        "cardplay_decisions": cycle["cycle_cardplay_decisions"],
        "bidding_decisions": cycle["cycle_bidding_decisions"],
        "play_transitions": cycle["cycle_play_transitions"],
        "bid_transitions": cycle["cycle_bid_transitions"],
        "abandoned_bidding_transitions": cycle[
            "cycle_abandoned_bidding_transitions"
        ],
        "learner_cardplay_samples": cycle["cycle_learner_cardplay_samples"],
        "learner_bidding_samples": cycle["cycle_learner_bidding_samples"],
        "learner_samples": cycle["cycle_learner_samples"],
        "learner_steps": cycle["cycle_learner_steps"],
    }
    counts = _validate_counts(count_source, "cycle_metrics.counts")
    frozen_trainer = STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD[
        "trainer_config"
    ]
    if counts["games"] != frozen_trainer["max_episodes"]:
        raise ValueError("cycle games disagree with the frozen R1 workload")
    if counts["learner_steps"] != frozen_trainer["optimizer_steps"]:
        raise ValueError("cycle learner steps disagree with the frozen R1 workload")
    totals = {
        name: _require_count(cycle[name], f"cycle_metrics.{name}")
        for name in (
            "total_episodes",
            "total_transitions",
            "total_optimizer_steps",
        )
    }
    if totals["total_episodes"] < counts["games"]:
        raise ValueError("cycle games exceed cumulative episodes")
    if totals["total_transitions"] < counts["play_transitions"]:
        raise ValueError("cycle transitions exceed cumulative transitions")
    if totals["total_optimizer_steps"] < counts["learner_steps"]:
        raise ValueError("cycle learner steps exceed cumulative learner steps")
    wall = {
        "total": _require_number(
            cycle["cycle_wall_seconds"],
            "cycle_metrics.cycle_wall_seconds",
            positive=True,
        ),
        "collection": _require_number(
            cycle["collection_seconds"],
            "cycle_metrics.collection_seconds",
            positive=True,
        ),
        "optimization": _require_number(
            cycle["optimization_seconds"],
            "cycle_metrics.optimization_seconds",
            positive=True,
        ),
    }
    _validate_phase_wall(wall, "cycle_metrics wall time", bound_overhead=False)
    rate_source = {
        "games_per_second": cycle["games_per_second"],
        "cardplay_decisions_per_second": cycle[
            "cardplay_decisions_per_second"
        ],
        "bidding_decisions_per_second": cycle[
            "bidding_decisions_per_second"
        ],
        "play_transitions_per_second": cycle["transitions_per_second"],
        "bid_transitions_per_second": cycle["bid_transitions_per_second"],
        "learner_samples_per_second": cycle["learner_samples_per_second"],
        "learner_steps_per_second": cycle["learner_steps_per_second"],
    }
    rates = _validate_rates(
        rate_source,
        "cycle_metrics.rates",
        expected={
            "games_per_second": (counts["games"], wall["collection"]),
            "cardplay_decisions_per_second": (
                counts["cardplay_decisions"],
                wall["collection"],
            ),
            "bidding_decisions_per_second": (
                counts["bidding_decisions"],
                wall["collection"],
            ),
            "play_transitions_per_second": (
                counts["play_transitions"],
                wall["collection"],
            ),
            "bid_transitions_per_second": (
                counts["bid_transitions"],
                wall["collection"],
            ),
            "learner_samples_per_second": (
                counts["learner_samples"],
                wall["optimization"],
            ),
            "learner_steps_per_second": (
                counts["learner_steps"],
                wall["optimization"],
            ),
        },
    )
    queue, gpu_seconds, staging = _validate_optional_sections(
        queue={
            "p50": cycle["inference_queue_p50_ms"],
            "p95": cycle["inference_queue_p95_ms"],
        },
        gpu_seconds={
            "inference": cycle["inference_gpu_seconds"],
            "learner": cycle["learner_gpu_seconds"],
        },
        staging_seconds=cycle["staging_seconds"],
        prefix="cycle_metrics",
    )
    peak_vram_bytes = _require_number(
        cycle["peak_vram_bytes"], "cycle_metrics.peak_vram_bytes", positive=True
    )
    return {
        "identity": identity,
        "counts": counts,
        "rates": rates,
        "queue_latency_ms": queue,
        "gpu_seconds": gpu_seconds,
        "staging_seconds": staging,
        "peak_vram_mib": round(peak_vram_bytes / (1024.0 * 1024.0), 3),
        "measurement": {
            "source_schema_version": cycle["schema_version"],
            "device_type": cycle["device_type"],
            "training_wall_seconds": wall["total"],
            "phase_wall_seconds": wall,
            "amp": amp,
            "compile": compile_info,
            "distributed": distributed,
            "parameter_update_observed": True,
            "metrics_history": history,
        },
    }


def validate_unified_benchmark_output(payload: object) -> dict[str, Any]:
    """Fail closed on a final checked-in Standard V2 R1 artifact."""

    root_fields = {
        "schema_version", "contract_version", "config_hash",
        "training_semantics_hash", "benchmark_workload_hash",
        "reference_digest", "coverage", "performance", "privacy",
    }
    artifact = _require_mapping(payload, "unified_benchmark", fields=root_fields)
    expected_identity = {
        "schema_version": STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": STANDARD_V2_R1_CONFIG_HASH,
        "training_semantics_hash": STANDARD_V2_R1_TRAINING_SEMANTICS_HASH,
        "benchmark_workload_hash": STANDARD_V2_R1_BENCHMARK_WORKLOAD_HASH,
        "reference_digest": STANDARD_V2_R1_REFERENCE_DIGEST,
    }
    for name, expected in expected_identity.items():
        if artifact[name] != expected:
            raise ValueError(f"unified_benchmark.{name} is not frozen R1")
    reference = build_standard_v2_reference()
    if artifact["coverage"] != reference["coverage"]:
        raise ValueError("unified_benchmark.coverage disagrees with the reference")
    if artifact["privacy"] != "sanitized_no_host_or_device_identifiers":
        raise ValueError("unified_benchmark privacy contract is invalid")

    performance = _require_mapping(
        artifact["performance"],
        "unified_benchmark.performance",
        fields={
            "status", "measurement", "counts", "rates", "queue_latency_ms",
            "gpu_seconds", "staging_seconds", "peak_vram_mib",
        },
    )
    if performance["status"] == "not_run":
        null_counts = _require_mapping(
            performance["counts"], "unified_benchmark.performance.counts",
            fields=set(_COUNT_FIELDS),
        )
        null_rates = _require_mapping(
            performance["rates"], "unified_benchmark.performance.rates",
            fields=set(_RATE_FIELDS),
        )
        queue = _require_mapping(
            performance["queue_latency_ms"],
            "unified_benchmark.performance.queue_latency_ms",
            fields={"p50", "p95"},
        )
        gpu = _require_mapping(
            performance["gpu_seconds"],
            "unified_benchmark.performance.gpu_seconds",
            fields={"inference", "learner"},
        )
        if (
            performance["measurement"] is not None
            or performance["staging_seconds"] is not None
            or performance["peak_vram_mib"] is not None
            or any(value is not None for value in null_counts.values())
            or any(value is not None for value in null_rates.values())
            or any(value is not None for value in queue.values())
            or any(value is not None for value in gpu.values())
        ):
            raise ValueError("not_run benchmark contains measured values")
        return artifact
    if performance["status"] != "measured":
        raise ValueError("unified_benchmark.performance.status is invalid")

    counts = _validate_counts(
        performance["counts"], "unified_benchmark.performance.counts"
    )
    frozen_trainer = STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD[
        "trainer_config"
    ]
    if counts["games"] != frozen_trainer["max_episodes"]:
        raise ValueError("unified benchmark games disagree with frozen workload")
    if counts["learner_steps"] != frozen_trainer["optimizer_steps"]:
        raise ValueError("unified benchmark steps disagree with frozen workload")

    measurement = _require_mapping(
        performance["measurement"],
        "unified_benchmark.performance.measurement",
        fields={
            "source_schema_version", "device_type", "training_wall_seconds",
            "phase_wall_seconds", "amp", "compile", "distributed",
            "parameter_update_observed", "metrics_history",
        },
    )
    if measurement["source_schema_version"] not in {
        _TRAINING_METRICS_SCHEMA_VERSION, _CYCLE_METRICS_SCHEMA_VERSION,
    }:
        raise ValueError("unified benchmark source schema is unsupported")
    if measurement["device_type"] != "cuda":
        raise ValueError("unified measured benchmark must come from CUDA")
    if measurement["parameter_update_observed"] is not True:
        raise ValueError("unified measured benchmark requires an update")
    training_wall = _require_number(
        measurement["training_wall_seconds"],
        "unified_benchmark.performance.measurement.training_wall_seconds",
        positive=True,
    )
    raw_wall = _require_mapping(
        measurement["phase_wall_seconds"],
        "unified_benchmark.performance.measurement.phase_wall_seconds",
        fields={"total", "collection", "optimization"},
    )
    wall = {
        name: _require_number(
            raw_wall[name],
            f"unified_benchmark.performance.measurement.phase_wall_seconds.{name}",
            positive=True,
        )
        for name in ("total", "collection", "optimization")
    }
    if not math.isclose(wall["total"], training_wall, abs_tol=1.0e-6):
        raise ValueError("unified benchmark total wall time is inconsistent")
    _validate_phase_wall(
        wall, "unified_benchmark.performance.measurement.phase_wall_seconds",
        bound_overhead=False,
    )
    _validate_amp(
        measurement["amp"],
        "unified_benchmark.performance.measurement.amp",
        include_observation=True,
    )
    _validate_compile(
        measurement["compile"],
        "unified_benchmark.performance.measurement.compile",
    )
    _validate_distributed(
        measurement["distributed"],
        "unified_benchmark.performance.measurement.distributed",
    )
    _validate_metrics_history(
        measurement["metrics_history"],
        "unified_benchmark.performance.measurement.metrics_history",
    )
    _validate_rates(
        performance["rates"], "unified_benchmark.performance.rates",
        expected={
            "games_per_second": (counts["games"], wall["collection"]),
            "cardplay_decisions_per_second": (
                counts["cardplay_decisions"], wall["collection"]
            ),
            "bidding_decisions_per_second": (
                counts["bidding_decisions"], wall["collection"]
            ),
            "play_transitions_per_second": (
                counts["play_transitions"], wall["collection"]
            ),
            "bid_transitions_per_second": (
                counts["bid_transitions"], wall["collection"]
            ),
            "learner_samples_per_second": (
                counts["learner_samples"], wall["optimization"]
            ),
            "learner_steps_per_second": (
                counts["learner_steps"], wall["optimization"]
            ),
        },
    )
    _validate_optional_sections(
        queue=performance["queue_latency_ms"],
        gpu_seconds=performance["gpu_seconds"],
        staging_seconds=performance["staging_seconds"],
        prefix="unified_benchmark.performance",
    )
    _require_number(
        performance["peak_vram_mib"],
        "unified_benchmark.performance.peak_vram_mib",
        positive=True,
    )
    return artifact


def build_unified_benchmark(
    *,
    training_metrics: dict[str, Any] | None = None,
    cycle_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an R1 artifact only from complete, identity-bound evidence."""

    if training_metrics is not None and cycle_metrics is not None:
        raise ValueError("provide exactly one measured benchmark source")
    reference = build_standard_v2_reference()
    if reference["reference_digest"] != STANDARD_V2_R1_REFERENCE_DIGEST:
        raise RuntimeError("loaded Standard V2 reference has an unexpected digest")
    measured = training_metrics is not None or cycle_metrics is not None
    source = None
    if training_metrics is not None:
        source = _validate_training_metrics(training_metrics)
    elif cycle_metrics is not None:
        source = _validate_cycle_metrics(cycle_metrics)

    counts = (
        source["counts"] if source is not None
        else {name: None for name in _COUNT_FIELDS}
    )
    rates = (
        source["rates"] if source is not None
        else {name: None for name in _RATE_FIELDS}
    )
    queue = (
        source["queue_latency_ms"] if source is not None
        else {"p50": None, "p95": None}
    )
    gpu_seconds = (
        source["gpu_seconds"] if source is not None
        else {"inference": None, "learner": None}
    )
    artifact = {
        "schema_version": STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": STANDARD_V2_R1_CONFIG_HASH,
        "training_semantics_hash": STANDARD_V2_R1_TRAINING_SEMANTICS_HASH,
        "benchmark_workload_hash": STANDARD_V2_R1_BENCHMARK_WORKLOAD_HASH,
        "reference_digest": reference["reference_digest"],
        "coverage": reference["coverage"],
        "performance": {
            "status": "measured" if measured else "not_run",
            "measurement": source["measurement"] if source is not None else None,
            "counts": counts,
            "rates": rates,
            "queue_latency_ms": queue,
            "gpu_seconds": gpu_seconds,
            "staging_seconds": (
                source["staging_seconds"] if source is not None else None
            ),
            "peak_vram_mib": (
                source["peak_vram_mib"] if source is not None else None
            ),
        },
        "privacy": "sanitized_no_host_or_device_identifiers",
    }
    validate_unified_benchmark_output(artifact)
    return artifact


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
