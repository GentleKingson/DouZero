"""Run one checkpoint-enabled H7 topology benchmark repetition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch

from douzero._version import git_sha
from douzero.env.rules import RuleSet
from douzero.observation.schema import build_v2_schema
from douzero.v3_hybrid import V3HybridModel
from douzero.v3_hybrid.benchmark import (
    H7_BENCHMARK_SCHEMA,
    V3H7BenchmarkProtocol,
    h7_trainer_identity_hash,
)
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.h7_smoke import build_v3_h7_smoke_config
from douzero.v3_hybrid.integration_config import load_v3_hybrid_config
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.runtime import (
    V3_H7_CHECKPOINT_FORMAT,
    V3_H7_REPLAY_PROTOCOL,
    V3_H7_REQUEST_PROTOCOL,
    V3_H7_RUNTIME_VERSION,
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
    V3SingleProcessTrainer,
    resolve_v3_h7_protocols,
    resolve_v3_h7_seed_contract,
    validate_v3_h7_formal_initialization,
)
from douzero.v3_hybrid.support_matrix import (
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
)
from douzero.v3_hybrid.training.h6_learner import V3H6Learner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--topology", choices=("single_process", "async_4x4", "async_8x4"),
        required=True,
    )
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    config = parser.add_mutually_exclusive_group()
    config.add_argument(
        "--formal-config",
        type=Path,
        help="run an H7 sidecar topology for a committed formal config",
    )
    config.add_argument(
        "--config",
        type=Path,
        help="run a committed H7 resolved H6 configuration",
    )
    return parser


def _load_protocol(path: Path) -> V3H7BenchmarkProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema", None) != H7_BENCHMARK_SCHEMA:
        raise ValueError("H7 benchmark protocol file schema mismatch")
    seeds = payload.get("seeds")
    if isinstance(seeds, list):
        payload["seeds"] = tuple(seeds)
    protocol = V3H7BenchmarkProtocol(**payload)
    if git_sha() != protocol.source_git_sha:
        raise ValueError("H7 benchmark source SHA does not match the frozen protocol")
    return protocol


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _validate_live_identity(
    protocol: V3H7BenchmarkProtocol, resolved, *, bound_config: bool
) -> None:
    image_digest = os.environ.get("DOUZERO_IMAGE_DIGEST")
    if image_digest != protocol.image_digest:
        raise ValueError("H7 benchmark live image digest mismatch")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "-i",
            str(torch.cuda.current_device()),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    observed = {
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "driver": query[0].strip() if len(query) == 1 else "",
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cpu": platform.machine(),
    }
    for name, value in observed.items():
        if value != getattr(protocol, name):
            raise ValueError(f"H7 benchmark live {name} mismatch")
    belief_enabled = resolved.learner.features.belief
    oracle_enabled = resolved.learner.features.oracle
    cooperation_enabled = resolved.learner.features.cooperation
    public_aux_enabled = (
        resolved.learner.features.strategy
        or resolved.learner.features.style
    )
    bidding_enabled = resolved.learner.features.bidding
    if protocol.oracle_enabled != oracle_enabled:
        raise ValueError("H7 benchmark Oracle capability identity mismatch")
    if protocol.cooperation_enabled != cooperation_enabled:
        raise ValueError("H7 benchmark cooperation capability identity mismatch")
    if protocol.public_aux_enabled != public_aux_enabled:
        raise ValueError("H7 benchmark public auxiliary identity mismatch")
    if protocol.bidding_enabled != bidding_enabled:
        raise ValueError("H7 benchmark bidding capability identity mismatch")
    request, replay, _snapshot = resolve_v3_h7_protocols(
        belief=belief_enabled,
        oracle=oracle_enabled,
        cooperation=cooperation_enabled,
        public_aux=public_aux_enabled,
        bidding=bidding_enabled,
    )
    trainer_identity_hash = h7_trainer_identity_hash(
        runtime_version=V3_H7_RUNTIME_VERSION,
        checkpoint_format=V3_H7_CHECKPOINT_FORMAT,
        request_protocol=request,
        resolved_learner_hash=(
            resolved.learner.stable_hash() if bound_config else None
        ),
    )
    if protocol.trainer_identity_hash != trainer_identity_hash:
        raise ValueError("H7 benchmark trainer identity mismatch")
    if protocol.replay_protocol_hash != _hash({"replay": replay}):
        raise ValueError("H7 benchmark replay identity mismatch")


def _shared_memory_bytes(trainer) -> int:
    if not getattr(trainer, "_runtime_started", False):
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    owners = [
        trainer._coordinator,
        trainer._coordinator.slots,
        trainer._replay_slots,
        trainer._replay_slots.observations,
    ]
    if trainer._coordinator.belief_inputs is not None:
        owners.append(trainer._coordinator.belief_inputs)
    if trainer._coordinator.bidding_inputs is not None:
        owners.append(trainer._coordinator.bidding_inputs)
    for owner in owners:
        for value in vars(owner).values():
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def _run_until(trainer, deadline: float, *, episodes: int) -> tuple[int, int]:
    steps = 0
    max_lag = 0
    while time.monotonic() < deadline:
        trainer.collect_episodes(episodes)
        if trainer.step() is None:
            continue
        steps += 1
        max_lag = max(max_lag, trainer.policy_step - trainer._snapshot_step)
    return steps, max_lag


def main() -> None:
    args = _parser().parse_args()
    protocol = _load_protocol(args.protocol)
    seed = protocol.seeds[args.repeat]
    formal = (
        None
        if args.formal_config is None
        else load_formal_config(args.formal_config)
    )
    if formal is not None:
        validate_v3_h7_formal_initialization(formal.initialization.kind)
    resolved = (
        load_v3_hybrid_config(args.config)
        if args.config is not None
        else (
            build_v3_h7_smoke_config()
            if formal is None
            else build_pilot_resolved_config(formal, allow_standard=True)
        )
    )
    formal_config_hash = (
        None
        if formal is None
        else str(formal.identity_dict()["config_sha256"])
    )
    if protocol.formal_config_hash != formal_config_hash:
        raise ValueError("H7 benchmark formal config identity mismatch")
    if resolved.stable_hash() != protocol.config_hash:
        raise ValueError("H7 benchmark config hash mismatch")
    if resolved.model.stable_hash() != protocol.model_identity_hash:
        raise ValueError("H7 benchmark model hash mismatch")
    _validate_live_identity(
        protocol,
        resolved,
        bound_config=formal is not None or args.config is not None,
    )

    if args.topology == "single_process":
        topology = TOPOLOGY_SINGLE_PROCESS
        actors, games, episodes = 1, 1, 4
        trainer_type = V3SingleProcessTrainer
    else:
        topology = TOPOLOGY_ASYNC_SINGLE_GPU
        actors = 4 if args.topology == "async_4x4" else 8
        games, episodes = 4, 4
        trainer_type = V3AsyncSingleGPUTrainer
    belief_enabled = resolved.learner.features.belief
    oracle_enabled = resolved.learner.features.oracle
    cooperation_enabled = resolved.learner.features.cooperation
    public_aux_enabled = (
        resolved.learner.features.strategy
        or resolved.learner.features.style
    )
    bidding_enabled = resolved.learner.features.bidding
    environment_seed, action_seed, seed_derivation = (
        resolve_v3_h7_seed_contract(
            formal_training_seeds=(
                None if formal is None else formal.seeds.training
            ),
            formal_derivation=(
                None if formal is None else formal.seeds.derivation
            ),
            requested_environment_seed=seed,
            requested_action_seed=None,
        )
    )
    request_protocol, replay_protocol, snapshot_semantics = (
        resolve_v3_h7_protocols(
            belief=belief_enabled,
            oracle=oracle_enabled,
            cooperation=cooperation_enabled,
            public_aux=public_aux_enabled,
            bidding=bidding_enabled,
        )
    )
    runtime_config = V3H7RuntimeConfig(
        topology=topology,
        num_actors=actors,
        games_per_actor=games,
        batch_size=32,
        replay_capacity=4096,
        target_microbatch=4,
        environment_seed=environment_seed,
        environment_seed_derivation=seed_derivation,
        action_seed=action_seed,
        belief_runtime_enabled=belief_enabled,
        oracle_runtime_enabled=oracle_enabled,
        cooperation_runtime_enabled=cooperation_enabled,
        public_aux_runtime_enabled=public_aux_enabled,
        bidding_runtime_enabled=bidding_enabled,
        request_protocol=request_protocol,
        replay_protocol=replay_protocol,
        snapshot_semantics=snapshot_semantics,
    )
    if formal is None:
        model = V3HybridModel(build_v2_schema(), resolved.model)
        learner = V3H6Learner(
            model,
            ruleset=(
                RuleSet.standard() if bidding_enabled else RuleSet.legacy()
            ),
            config=resolved,
        )
    else:
        learner, learner_resolved = create_pilot_learner(
            formal, allow_standard=True
        )
        if learner_resolved != resolved:
            raise RuntimeError("H7 benchmark formal resolution is not stable")
    if protocol.learner_phase == "oracle_guided":
        h3 = learner.base.base.base
        h3.learner_updates = h3.config.schedule.warmup_updates
    trainer = trainer_type(learner, resolved, runtime_config)
    try:
        _run_until(
            trainer, time.monotonic() + protocol.warmup_seconds,
            episodes=episodes,
        )
        trainer.quiesce_cycle_boundary()
        torch.cuda.reset_peak_memory_stats(trainer.device)
        before = {
            "games": trainer.stats.games_collected,
            "decisions": trainer.stats.decisions_collected,
            "transitions": trainer.stats.transitions_collected,
            "samples": trainer.stats.learner_cardplay_samples,
            "steps": trainer.stats.optimizer_steps,
            "oracle_labels": trainer.stats.oracle_labels_collected,
            "oracle_steps": trainer.stats.oracle_optimizer_steps,
            "cooperation_labels": (
                trainer.stats.cooperation_labels_collected
            ),
            "cooperation_episodes": (
                trainer.stats.cooperation_episodes_collected
            ),
            "cooperation_steps": (
                trainer.stats.cooperation_optimizer_steps
            ),
            "strategy_labels": trainer.stats.strategy_labels_collected,
            "strategy_steps": trainer.stats.strategy_optimizer_steps,
            "bidding_samples": trainer.stats.learner_bidding_samples,
            "bidding_steps": trainer.stats.bidding_optimizer_steps,
        }
        parameter_snapshot = trainer._parameter_update_snapshot()
        started = time.monotonic()
        steps, max_lag = _run_until(
            trainer, started + protocol.measurement_seconds,
            episodes=episodes,
        )
        elapsed = time.monotonic() - started
        boundary = trainer.quiesce_cycle_boundary()
        after = {
            "games": trainer.stats.games_collected,
            "decisions": trainer.stats.decisions_collected,
            "transitions": trainer.stats.transitions_collected,
            "samples": trainer.stats.learner_cardplay_samples,
            "steps": trainer.stats.optimizer_steps,
            "oracle_labels": trainer.stats.oracle_labels_collected,
            "oracle_steps": trainer.stats.oracle_optimizer_steps,
            "cooperation_labels": (
                trainer.stats.cooperation_labels_collected
            ),
            "cooperation_episodes": (
                trainer.stats.cooperation_episodes_collected
            ),
            "cooperation_steps": (
                trainer.stats.cooperation_optimizer_steps
            ),
            "strategy_labels": trainer.stats.strategy_labels_collected,
            "strategy_steps": trainer.stats.strategy_optimizer_steps,
            "bidding_samples": trainer.stats.learner_bidding_samples,
            "bidding_steps": trainer.stats.bidding_optimizer_steps,
        }
        shared_memory = _shared_memory_bytes(trainer)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_training_checkpoint(
            str(args.checkpoint),
            long_running_state={
                "benchmark_protocol_hash": protocol.stable_hash(),
                "topology": args.topology,
                "repeat": args.repeat,
            },
        )
        changed = trainer._parameters_changed_since(parameter_snapshot)
        cpu_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        shutdown_started = time.monotonic()
        trainer.shutdown()
        shutdown_seconds = time.monotonic() - shutdown_started
        trainer = None
        record = {
            "schema": H7_BENCHMARK_SCHEMA,
            "protocol_hash": protocol.stable_hash(),
            "topology": args.topology,
            "repeat": args.repeat,
            "seed": seed,
            "measurement_seconds": elapsed,
            "checkpoint_path": str(args.checkpoint),
            "parameter_update_observed": changed and steps > 0,
            "active_slots": int(boundary["active_slots"]),
            "in_flight": int(boundary["in_flight_slots"]),
            "pending": int(boundary["pending_requests"]),
            "games_per_second": (after["games"] - before["games"]) / elapsed,
            "decisions_per_second": (
                after["decisions"] - before["decisions"]
            ) / elapsed,
            "transitions_per_second": (
                after["transitions"] - before["transitions"]
            ) / elapsed,
            "learner_samples_per_second": (
                after["samples"] - before["samples"]
            ) / elapsed,
            "optimizer_steps_per_second": (
                after["steps"] - before["steps"]
            ) / elapsed,
            "oracle_samples_per_second": (
                after["oracle_labels"] - before["oracle_labels"]
            ) / elapsed,
            "oracle_optimizer_steps_per_second": (
                after["oracle_steps"] - before["oracle_steps"]
            ) / elapsed,
            "oracle_parameter_vram_bytes": int(
                boundary["oracle_parameter_vram_bytes"]
            ),
            "cooperation_samples_per_second": (
                after["cooperation_labels"] - before["cooperation_labels"]
            ) / elapsed,
            "cooperation_episodes_per_second": (
                after["cooperation_episodes"]
                - before["cooperation_episodes"]
            ) / elapsed,
            "cooperation_optimizer_steps_per_second": (
                after["cooperation_steps"] - before["cooperation_steps"]
            ) / elapsed,
            "cooperation_parameter_vram_bytes": int(
                boundary["cooperation_parameter_vram_bytes"]
            ),
            "strategy_samples_per_second": (
                after["strategy_labels"] - before["strategy_labels"]
            ) / elapsed,
            "strategy_optimizer_steps_per_second": (
                after["strategy_steps"] - before["strategy_steps"]
            ) / elapsed,
            "public_aux_parameter_vram_bytes": int(
                boundary["public_aux_parameter_vram_bytes"]
            ),
            "bidding_samples_per_second": (
                after["bidding_samples"] - before["bidding_samples"]
            ) / elapsed,
            "bidding_optimizer_steps_per_second": (
                after["bidding_steps"] - before["bidding_steps"]
            ) / elapsed,
            "bidding_parameter_vram_bytes": int(
                boundary["bidding_parameter_vram_bytes"]
            ),
            "requests_per_microbatch": float(boundary["requests_per_microbatch"]),
            "legal_actions_per_batch": float(boundary["actions_per_microbatch"]),
            "queue_wait_seconds": float(boundary["claim_wait_seconds"]),
            "slot_read_seconds": float(boundary["slot_read_seconds"]),
            "collate_seconds": float(boundary["collate_seconds"]),
            "h2d_seconds": float(boundary["h2d_seconds"]),
            "forward_seconds": float(
                boundary["forward_seconds"] + boundary["belief_forward_seconds"]
            ),
            "d2h_seconds": float(boundary["d2h_seconds"]),
            "publish_seconds": float(boundary["publish_seconds"]),
            "replay_drain_seconds": float(boundary["replay_drain_seconds"]),
            "learner_throttle_seconds": float(boundary["learner_throttle_seconds"]),
            "actor_blocked_ratio": float(boundary["actor_blocked_ratio"]),
            "learner_data_wait_ratio": float(boundary["learner_data_wait_ratio"]),
            "policy_lag_max": float(max_lag),
            "cpu_ram_bytes": float(cpu_ram),
            "shared_memory_bytes": float(shared_memory),
            "vram_bytes": float(torch.cuda.max_memory_allocated()),
            "shutdown_seconds": shutdown_seconds,
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in record.values()
        ):
            raise ValueError("H7 benchmark produced a non-finite metric")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    finally:
        if trainer is not None:
            trainer.shutdown()


if __name__ == "__main__":
    main()
