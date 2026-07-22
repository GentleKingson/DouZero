"""Frozen H7 end-to-end benchmark protocol and fail-closed evidence checks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

H7_BENCHMARK_SCHEMA = "v3-hybrid-h7-benchmark-v1"
H7_TOPOLOGIES = ("single_process", "async_4x4", "async_8x4")


def _hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H7BenchmarkProtocol:
    source_git_sha: str
    image_digest: str
    config_hash: str
    model_identity_hash: str
    trainer_identity_hash: str
    replay_protocol_hash: str
    gpu: str
    driver: str
    pytorch: str
    cuda: str
    cpu: str
    warmup_seconds: float = 30.0
    measurement_seconds: float = 300.0
    checkpoint_enabled: bool = True
    seeds: tuple[int, ...] = (101, 202, 303)
    repetitions: int = 3

    def __post_init__(self) -> None:
        for name in (
            "source_git_sha", "image_digest", "config_hash",
            "model_identity_hash", "trainer_identity_hash",
            "replay_protocol_hash", "gpu", "driver", "pytorch", "cuda", "cpu",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"H7 benchmark {name} must be non-empty")
        if len(self.source_git_sha) != 40:
            raise ValueError("H7 benchmark source SHA must be full length")
        for name in ("config_hash", "model_identity_hash", "trainer_identity_hash", "replay_protocol_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"H7 benchmark {name} must be SHA-256")
        if not self.image_digest.startswith("sha256:") or len(self.image_digest) != 71:
            raise ValueError("H7 benchmark image digest must be sha256:<64 hex>")
        for name in ("warmup_seconds", "measurement_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"H7 benchmark {name} must be positive and finite")
        if self.repetitions < 3 or len(self.seeds) < 3:
            raise ValueError("H7 benchmark requires at least three repeats and seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("H7 benchmark seeds must be unique")
        if self.checkpoint_enabled is not True:
            raise ValueError("H7 benchmark evidence must be checkpoint-enabled")

    def identity(self) -> dict[str, object]:
        return {"schema": H7_BENCHMARK_SCHEMA, **asdict(self)}

    def stable_hash(self) -> str:
        return _hash(self.identity())


_METRICS = (
    "games_per_second", "decisions_per_second", "transitions_per_second",
    "learner_samples_per_second", "optimizer_steps_per_second",
    "requests_per_microbatch", "legal_actions_per_batch",
    "queue_wait_seconds", "slot_read_seconds", "collate_seconds",
    "h2d_seconds", "forward_seconds", "d2h_seconds", "publish_seconds",
    "replay_drain_seconds", "learner_throttle_seconds", "actor_blocked_ratio",
    "learner_data_wait_ratio", "policy_lag_max", "cpu_ram_bytes",
    "shared_memory_bytes", "vram_bytes", "shutdown_seconds",
)


def validate_h7_benchmark_evidence(
    records: Sequence[Mapping[str, object]],
    protocol: V3H7BenchmarkProtocol,
) -> None:
    """Reject incomplete, mismatched, or non-finite topology evidence."""
    expected_keys = {
        "schema", "protocol_hash", "topology", "repeat", "seed",
        "measurement_seconds", "checkpoint_path", "parameter_update_observed",
        "active_slots", "in_flight", "pending", *_METRICS,
    }
    seen: set[tuple[str, int]] = set()
    counts = {topology: 0 for topology in H7_TOPOLOGIES}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError("H7 benchmark record fields mismatch")
        if record["schema"] != H7_BENCHMARK_SCHEMA:
            raise ValueError("H7 benchmark record schema mismatch")
        if record["protocol_hash"] != protocol.stable_hash():
            raise ValueError("H7 benchmark protocol hash mismatch")
        topology = record["topology"]
        if topology not in H7_TOPOLOGIES:
            raise ValueError("H7 benchmark topology is unknown")
        repeat = record["repeat"]
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("H7 benchmark repeat must be non-negative")
        if repeat >= protocol.repetitions:
            raise ValueError("H7 benchmark repeat exceeds the frozen protocol")
        key = (topology, repeat)
        if key in seen:
            raise ValueError("duplicate H7 benchmark topology repeat")
        seen.add(key)
        counts[topology] += 1
        if record["seed"] != protocol.seeds[repeat]:
            raise ValueError("H7 benchmark seed does not match its frozen repeat")
        if record["parameter_update_observed"] is not True:
            raise ValueError("H7 benchmark did not observe a parameter update")
        if not isinstance(record["checkpoint_path"], str) or not record["checkpoint_path"]:
            raise ValueError("H7 benchmark checkpoint path is missing")
        for name in _METRICS + ("measurement_seconds",):
            value = record[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"H7 benchmark metric {name} is invalid")
        if record["measurement_seconds"] < protocol.measurement_seconds:
            raise ValueError("H7 benchmark measurement window is too short")
        for name in ("active_slots", "in_flight", "pending"):
            if record[name] != 0:
                raise ValueError(f"H7 benchmark did not quiesce {name}")
        for name in ("actor_blocked_ratio", "learner_data_wait_ratio"):
            if record[name] > 1.0:
                raise ValueError(f"H7 benchmark ratio {name} exceeds one")
        if record["policy_lag_max"] > 128:
            raise ValueError("H7 benchmark policy lag exceeds the frozen gate")
    if any(count != protocol.repetitions for count in counts.values()):
        raise ValueError("H7 benchmark requires all topology repetitions")


__all__ = [
    "H7_BENCHMARK_SCHEMA",
    "H7_TOPOLOGIES",
    "V3H7BenchmarkProtocol",
    "validate_h7_benchmark_evidence",
]
