"""P3 matched-runtime benchmark identity, validation, and decision logic."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

P3_RUNTIME_SCHEMA = "v3-p3-runtime-decision-v5"
P3_MEASUREMENT_SEED_WINDOW = "fresh-runtime-episode-zero-v1"
P3_FULL_CHECKPOINT_FORMAT = "douzero-v3-p3-full-runtime-checkpoint-v1"
P3_TOPOLOGIES = (
    "base_single_process",
    "base_async_4x4",
    "base_async_8x4",
    "full_hybrid_single_process",
)
P3_SEGMENTS = (
    "public_model_forward",
    "oracle_forward_backward",
    "belief_logits",
    "exact_dp",
    "cooperation_trajectory_assembly",
    "mixer",
    "strategy_features",
    "style",
    "bidding",
    "collate",
    "h2d",
    "d2h",
    "queue",
    "replay",
    "checkpoint",
)
_COUNTERS = (
    "games",
    "decisions",
    "transitions",
    "learner_samples",
    "optimizer_steps",
)
_RATES = tuple(f"{name}_per_second" for name in _COUNTERS)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _require_sha256(name: str, value: object, *, prefix: bool = False) -> None:
    expected_length = 71 if prefix else 64
    text = value[7:] if prefix and isinstance(value, str) else value
    if (
        not isinstance(value, str)
        or len(value) != expected_length
        or (prefix and not value.startswith("sha256:"))
        or not isinstance(text, str)
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"P3 {name} must be a full SHA-256")


@dataclass(frozen=True)
class P3RuntimeProtocol:
    source_git_sha: str
    source_tree: str
    image_digest: str
    base_config_hash: str
    full_config_hash: str
    shared_scale_hash: str
    model_identity_hashes: Mapping[str, str]
    trainer_identity_hash: str
    replay_protocol_hash: str
    gpu: str
    driver: str
    pytorch: str
    cuda: str
    cpu: str
    warmup_seconds: float = 30.0
    measurement_seconds: float = 300.0
    checkpoint_cadence_updates: int = 1000
    batch_size: int = 32
    seeds: tuple[int, ...] = (101, 202, 303)
    repetitions: int = 3
    full_hybrid_min_base_ratio: float = 0.70
    max_policy_lag: int = 128
    checkpoint_enabled: bool = True
    deal_seed_derivation: str = (
        "sha256(root_seed,stream_name,worker_id,episode_id)-v1"
    )
    episodes_per_learner_update: int = 4
    full_hybrid_phase: str = "guided"
    full_hybrid_phase_update: int = 10000
    measurement_seed_window: str = P3_MEASUREMENT_SEED_WINDOW
    full_checkpoint_format: str = P3_FULL_CHECKPOINT_FORMAT

    def __post_init__(self) -> None:
        for name in (
            "source_git_sha",
            "source_tree",
            "gpu",
            "driver",
            "pytorch",
            "cuda",
            "cpu",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"P3 {name} must be non-empty")
        for name in ("source_git_sha", "source_tree"):
            value = getattr(self, name)
            if len(value) != 40 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"P3 {name} must be a full Git identity")
        _require_sha256("image_digest", self.image_digest, prefix=True)
        for name in (
            "base_config_hash",
            "full_config_hash",
            "shared_scale_hash",
            "trainer_identity_hash",
            "replay_protocol_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if set(self.model_identity_hashes) != {"base", "full_hybrid"}:
            raise ValueError("P3 model identity fields mismatch")
        for name, value in self.model_identity_hashes.items():
            _require_sha256(f"model_identity_hashes.{name}", value)
        for name in ("warmup_seconds", "measurement_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"P3 {name} must be positive and finite")
        for name in (
            "checkpoint_cadence_updates",
            "batch_size",
            "repetitions",
            "max_policy_lag",
            "episodes_per_learner_update",
            "full_hybrid_phase_update",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"P3 {name} must be a positive integer")
        if self.repetitions < 3 or len(self.seeds) < self.repetitions:
            raise ValueError("P3 requires at least three matched repetitions")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("P3 seeds must be unique")
        if (
            not math.isfinite(self.full_hybrid_min_base_ratio)
            or not 0.0 < self.full_hybrid_min_base_ratio <= 1.0
        ):
            raise ValueError("P3 full-hybrid threshold must be in (0, 1]")
        if self.checkpoint_enabled is not True:
            raise ValueError("P3 benchmark must be checkpoint-enabled")
        if self.deal_seed_derivation != (
            "sha256(root_seed,stream_name,worker_id,episode_id)-v1"
        ):
            raise ValueError("P3 deal seed derivation is unsupported")
        if self.full_hybrid_phase != "guided":
            raise ValueError("P3 full-hybrid benchmark phase must be guided")
        if self.measurement_seed_window != P3_MEASUREMENT_SEED_WINDOW:
            raise ValueError("P3 measurement seed window is unsupported")
        if self.full_checkpoint_format != P3_FULL_CHECKPOINT_FORMAT:
            raise ValueError("P3 full checkpoint format is unsupported")

    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_identity_hashes"] = dict(self.model_identity_hashes)
        return {"schema": P3_RUNTIME_SCHEMA, **payload}

    def stable_hash(self) -> str:
        return _canonical_hash(self.identity())


def _number(name: str, value: object, *, nonnegative: bool = True) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        raise ValueError(f"P3 {name} is invalid")
    return float(value)


def validate_p3_records(
    records: Sequence[Mapping[str, object]],
    protocol: P3RuntimeProtocol,
) -> None:
    """Reject incomplete, unmatched, or arithmetically inconsistent evidence."""

    expected = {
        "schema",
        "protocol_hash",
        "topology",
        "repeat",
        "seed",
        "source_git_sha",
        "source_tree",
        "image_digest",
        "config_hash",
        "model_identity_hash",
        "deal_seed_derivation",
        "measurement_seed_window",
        "measurement_seconds",
        "counters_before",
        "counters_after",
        "rates",
        "segments_seconds",
        "parameter_update_observed",
        "training_phase",
        "checkpoint",
        "policy_lag_max",
        "actor_blocked_ratio",
        "learner_data_wait_ratio",
        "cpu_ram_bytes",
        "shared_memory_bytes",
        "vram_bytes",
        "active_slots",
        "in_flight",
        "pending",
        "shutdown_seconds",
        "skipped_long_cooperation_episodes",
        "skipped_incomplete_cooperation_episodes",
    }
    seen: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("P3 benchmark record fields mismatch")
        if record["schema"] != P3_RUNTIME_SCHEMA:
            raise ValueError("P3 benchmark record schema mismatch")
        if record["protocol_hash"] != protocol.stable_hash():
            raise ValueError("P3 benchmark protocol hash mismatch")
        topology = record["topology"]
        if topology not in P3_TOPOLOGIES:
            raise ValueError("P3 benchmark topology is unknown")
        repeat = record["repeat"]
        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or not 0 <= repeat < protocol.repetitions
        ):
            raise ValueError("P3 benchmark repeat is invalid")
        key = (str(topology), repeat)
        if key in seen:
            raise ValueError("duplicate P3 benchmark topology/repeat")
        seen.add(key)
        if record["seed"] != protocol.seeds[repeat]:
            raise ValueError("P3 benchmark seed does not match the protocol")
        for name in ("source_git_sha", "source_tree", "image_digest"):
            if record[name] != getattr(protocol, name):
                raise ValueError(f"P3 benchmark {name} drift")
        expected_kind = (
            "full_hybrid"
            if topology == "full_hybrid_single_process"
            else "base"
        )
        expected_config = (
            protocol.full_config_hash
            if expected_kind == "full_hybrid"
            else protocol.base_config_hash
        )
        if record["config_hash"] != expected_config:
            raise ValueError("P3 benchmark config hash drift")
        if record["model_identity_hash"] != protocol.model_identity_hashes[
            expected_kind
        ]:
            raise ValueError("P3 benchmark model identity drift")
        if record["deal_seed_derivation"] != protocol.deal_seed_derivation:
            raise ValueError("P3 benchmark deal seed derivation drift")
        if record["measurement_seed_window"] != protocol.measurement_seed_window:
            raise ValueError("P3 benchmark measurement seed window drift")
        elapsed = _number("measurement_seconds", record["measurement_seconds"])
        if elapsed < protocol.measurement_seconds:
            raise ValueError("P3 benchmark measurement window is too short")
        before = record["counters_before"]
        after = record["counters_after"]
        rates = record["rates"]
        if not isinstance(before, Mapping) or set(before) != set(_COUNTERS):
            raise ValueError("P3 counters_before fields mismatch")
        if not isinstance(after, Mapping) or set(after) != set(_COUNTERS):
            raise ValueError("P3 counters_after fields mismatch")
        if any(before[counter] != 0 for counter in _COUNTERS):
            raise ValueError(
                "P3 measurement counters must start from a fresh runtime"
            )
        if not isinstance(rates, Mapping) or set(rates) != set(_RATES):
            raise ValueError("P3 rate fields mismatch")
        for counter in _COUNTERS:
            for label, payload in (("before", before), ("after", after)):
                value = payload[counter]
                if type(value) is not int or value < 0:
                    raise ValueError(f"P3 {label} counter {counter} is invalid")
            if after[counter] < before[counter]:
                raise ValueError(f"P3 counter {counter} regressed")
            expected_rate = (after[counter] - before[counter]) / elapsed
            observed_rate = _number(f"{counter}_per_second", rates[
                f"{counter}_per_second"
            ])
            if not math.isclose(observed_rate, expected_rate, rel_tol=1e-9):
                raise ValueError(f"P3 rate {counter} is inconsistent")
        if not isinstance(record["parameter_update_observed"], bool):
            raise ValueError("P3 parameter update observation must be bool")
        training_phase = record["training_phase"]
        if not isinstance(training_phase, Mapping) or set(training_phase) != {
            "before",
            "after",
            "learner_update_before",
            "learner_update_after",
        }:
            raise ValueError("P3 training phase fields mismatch")
        expected_phase = (
            protocol.full_hybrid_phase
            if expected_kind == "full_hybrid"
            else "disabled"
        )
        if (
            training_phase["before"] != expected_phase
            or training_phase["after"] != expected_phase
        ):
            raise ValueError("P3 benchmark training phase drift")
        for name in ("learner_update_before", "learner_update_after"):
            value = training_phase[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"P3 training phase {name} is invalid")
        if (
            training_phase["learner_update_after"]
            <= training_phase["learner_update_before"]
        ):
            raise ValueError("P3 training phase did not advance")
        if (
            training_phase["learner_update_after"]
            - training_phase["learner_update_before"]
            != after["optimizer_steps"] - before["optimizer_steps"]
        ):
            raise ValueError("P3 training phase/update counter drift")
        if (
            expected_kind == "full_hybrid"
            and training_phase["learner_update_before"]
            < protocol.full_hybrid_phase_update
        ):
            raise ValueError("P3 full-hybrid phase was not pre-advanced")
        checkpoint = record["checkpoint"]
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "path",
            "sha256",
            "saved",
            "strict_reload",
            "resumed_update",
            "resume_quiesced",
        }:
            raise ValueError("P3 checkpoint fields mismatch")
        if any(
            checkpoint[name] is not True
            for name in (
                "saved",
                "strict_reload",
                "resumed_update",
                "resume_quiesced",
            )
        ):
            raise ValueError(
                "P3 checkpoint save/reload/resumed update was not demonstrated"
            )
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
            raise ValueError("P3 checkpoint path is missing")
        _require_sha256("checkpoint.sha256", checkpoint["sha256"])
        segments = record["segments_seconds"]
        if not isinstance(segments, Mapping) or set(segments) != set(P3_SEGMENTS):
            raise ValueError("P3 segmented timing fields mismatch")
        for name in P3_SEGMENTS:
            _number(f"segment {name}", segments[name])
        for name in (
            "policy_lag_max",
            "actor_blocked_ratio",
            "learner_data_wait_ratio",
            "cpu_ram_bytes",
            "shared_memory_bytes",
            "vram_bytes",
            "shutdown_seconds",
        ):
            _number(name, record[name])
        if record["policy_lag_max"] > protocol.max_policy_lag:
            raise ValueError("P3 policy lag exceeds the frozen gate")
        for name in ("actor_blocked_ratio", "learner_data_wait_ratio"):
            if record[name] > 1.0:
                raise ValueError(f"P3 {name} exceeds one")
        for name in ("active_slots", "in_flight", "pending"):
            if record[name] != 0:
                raise ValueError(f"P3 benchmark did not quiesce {name}")
        for name in (
            "skipped_long_cooperation_episodes",
            "skipped_incomplete_cooperation_episodes",
        ):
            skipped = record[name]
            if type(skipped) is not int or skipped < 0:
                raise ValueError(f"P3 {name} counter is invalid")
    expected_pairs = {
        (topology, repeat)
        for topology in P3_TOPOLOGIES
        for repeat in range(protocol.repetitions)
    }
    if seen != expected_pairs:
        raise ValueError("P3 benchmark requires every matched repetition")


def summarize_p3_decision(
    records: Sequence[Mapping[str, object]],
    protocol: P3RuntimeProtocol,
) -> dict[str, object]:
    """Recompute the runtime gate from validated raw records."""

    validate_p3_records(records, protocol)
    grouped: dict[str, list[float]] = {name: [] for name in P3_TOPOLOGIES}
    for record in records:
        grouped[str(record["topology"])].append(
            float(record["rates"]["learner_samples_per_second"])
        )
    medians = {
        topology: statistics.median(values)
        for topology, values in grouped.items()
    }
    base = medians["base_single_process"]
    full = medians["full_hybrid_single_process"]
    ratio = 0.0 if base == 0.0 else full / base
    runtime_stable = all(
        record["parameter_update_observed"] is True
        and record["checkpoint"]["strict_reload"] is True
        and record["checkpoint"]["resumed_update"] is True
        and record["checkpoint"]["resume_quiesced"] is True
        and record["policy_lag_max"] <= protocol.max_policy_lag
        for record in records
    )
    implement_h7_1 = (
        not runtime_stable
        or base == 0.0
        or ratio < protocol.full_hybrid_min_base_ratio
    )
    return {
        "schema": P3_RUNTIME_SCHEMA,
        "protocol_hash": protocol.stable_hash(),
        "median_learner_samples_per_second": medians,
        "full_hybrid_to_base_single_ratio": ratio,
        "threshold": protocol.full_hybrid_min_base_ratio,
        "runtime_stable": runtime_stable,
        "implement_h7_1": implement_h7_1,
        "release_candidate": "NONE",
        "release_status": "NOT READY",
        "playing_strength": "NOT MEASURED",
    }


__all__ = [
    "P3_FULL_CHECKPOINT_FORMAT",
    "P3_MEASUREMENT_SEED_WINDOW",
    "P3_RUNTIME_SCHEMA",
    "P3_SEGMENTS",
    "P3_TOPOLOGIES",
    "P3RuntimeProtocol",
    "summarize_p3_decision",
    "validate_p3_records",
]
