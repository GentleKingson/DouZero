"""Frozen Standard V2 R1 semantics and version names.

This module is deliberately dependency-light.  It gives benchmarks,
checkpoint code, and future async protocol implementations one place to
import the names that define the R1 mathematical boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


STANDARD_V2_R1_CONTRACT_VERSION = "standard-v2-r1-contract-v2"
STANDARD_V2_REFERENCE_SCHEMA_VERSION = "standard-v2-r1-reference-v1"
STANDARD_V2_BENCHMARK_SCHEMA_VERSION = "standard-v2-r1-benchmark-v1"
STANDARD_V2_R1_REFERENCE_DIGEST = (
    "bfcefc9ab6f26f3f259c144bdf7adb7d5f1677b5e7c1a20b5024ab56dd05af18"
)

# The currently implemented base async path.  These values are written into
# async checkpoints so protocol changes cannot be mistaken for exact resume.
BASE_ASYNC_PROTOCOL_VERSION = 1
BASE_ASYNC_PROTOCOL_SEMANTICS = "base_play_only_async_v1"
BASE_EPISODE_TASK_SEMANTICS = "actor_local_task_queue_v1"
BASE_EPISODE_COMMIT_SEMANTICS = "cardplay_count_reconciled_v1"
BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION = 0

# Reserved names for the M2-M4 Standard async implementation.  Defining them
# now prevents individual components from inventing incompatible identities.
STANDARD_ASYNC_PROTOCOL_VERSION = 2
STANDARD_ASYNC_PROTOCOL_SEMANTICS = "standard_bid_play_async_v2"
STANDARD_EPISODE_TASK_SEMANTICS = "global_episode_domain_rng_v2"
STANDARD_EPISODE_COMMIT_SEMANTICS = "atomic_bid_play_terminal_v2"
STANDARD_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION = 1
STANDARD_SNAPSHOT_PUBLICATION_SEMANTICS = (
    "cycle_quiescent_atomic_standard_bundle_v2"
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_identity_hash(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 used by R1 contract artifacts."""

    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def standard_v2_r1_config_identity(config: Any) -> dict[str, Any]:
    """Extract only production-relevant R1 semantics from ``TrainingConfig``."""

    return {
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "runtime": {
            "ruleset": config.ruleset,
            "feature_version": config.feature_version,
            "model_version": config.model_version,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "batch_size": config.batch_size,
            "bidding_batch_size": (
                config.batch_size
                if config.bidding_batch_size is None
                else config.bidding_batch_size
            ),
            "bidding_update_interval": config.bidding_update_interval,
            "exp_epsilon": config.exp_epsilon,
            "max_grad_norm": config.max_grad_norm,
            "amp_enabled": config.amp_enabled,
            "amp_dtype": config.amp_dtype,
            "amp_fallback_on_nonfinite": config.amp_fallback_on_nonfinite,
            "ddp_enabled": config.ddp_enabled,
            "ddp_backend": config.ddp_backend,
            "compile_model": config.compile_model,
            "first_bidder_mode": config.first_bidder_mode,
        },
        "optimizer": {
            "learning_rate": config.optimizer.learning_rate,
            "alpha": config.optimizer.alpha,
            "momentum": config.optimizer.momentum,
            "epsilon": config.optimizer.epsilon,
        },
        "model": {
            "version": config.model.version,
            "hidden_size": config.model.hidden_size,
            "history_encoder": config.model.history_encoder,
            "history_layers": config.model.history_layers,
            "history_heads": config.model.history_heads,
            "role_embedding_dim": config.model.role_embedding_dim,
            "belief_enabled": config.model.belief_enabled,
            "human_prior_enabled": config.model.human_prior_enabled,
            "style_enabled": config.model.style_enabled,
            "strategy_features_enabled": config.model.strategy_features_enabled,
            "strategy_aux_enabled": config.model.strategy_aux_enabled,
            "bidding_enabled": config.model.bidding_enabled,
            "bidding_hidden_size": config.model.bidding_hidden_size,
            "bidding_uncertainty_enabled": (
                config.model.bidding_uncertainty_enabled
            ),
        },
        "loss": {
            name: getattr(config.loss, name)
            for name in (
                "lambda_win",
                "lambda_score",
                "lambda_uncertainty",
                "lambda_bc",
                "lambda_min_turns",
                "lambda_regain_initiative",
                "lambda_teammate_finish",
                "lambda_spring",
                "lambda_structure",
                "lambda_bid_policy",
                "lambda_bid_win",
                "lambda_bid_score",
                "lambda_bid_regret",
                "score_delta",
                "score_target_transform",
                "score_clamp",
            )
        },
        "decision_policy": {
            name: getattr(config.decision_policy, name)
            for name in (
                "mode",
                "abs_tol",
                "rel_tol",
                "risk_penalty",
                "prior_alpha",
            )
        },
        "bidding": {
            "enabled": config.bidding.enabled,
            "policy": config.bidding.policy,
            "warm_start_policy": config.bidding.warm_start_policy,
            "learned_probability": config.bidding.learned_probability,
        },
    }


def resolved_standard_v2_config_identity(
    *,
    trainer_config: Any,
    model_config: Any,
    loss_config: Any,
    decision_config: Any,
    bidding_config: Any | None,
    ruleset: str,
    feature_version: str,
    model_version: str,
    deterministic: bool,
    ddp_enabled: bool,
    ddp_backend: str,
    world_size: int,
    compile_model: bool,
    bidding_enabled: bool,
) -> dict[str, Any]:
    """Bind mathematical semantics and the complete benchmark workload."""

    bidding_policy = bidding_config
    training_semantics = {
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "runtime": {
            "ruleset": str(ruleset),
            "feature_version": str(feature_version),
            "model_version": str(model_version),
            "seed": int(trainer_config.seed),
            "deterministic": bool(deterministic),
            "batch_size": int(trainer_config.batch_size),
            "bidding_batch_size": int(trainer_config.bidding_batch_size),
            "bidding_update_interval": int(
                trainer_config.bidding_update_interval
            ),
            "exp_epsilon": float(trainer_config.exp_epsilon),
            "max_grad_norm": float(trainer_config.max_grad_norm),
            "amp_enabled": bool(trainer_config.amp_enabled),
            "amp_dtype": str(trainer_config.amp_dtype),
            "amp_fallback_on_nonfinite": bool(
                trainer_config.amp_fallback_on_nonfinite
            ),
            "ddp_enabled": bool(ddp_enabled),
            "ddp_backend": str(ddp_backend),
            "compile_model": bool(compile_model),
            "first_bidder_mode": str(trainer_config.first_bidder_mode),
        },
        "optimizer": {
            "learning_rate": float(trainer_config.learning_rate),
            "alpha": float(trainer_config.rmsprop_alpha),
            "momentum": float(trainer_config.rmsprop_momentum),
            "epsilon": float(trainer_config.rmsprop_epsilon),
        },
        "model": {
            "version": str(model_version),
            "hidden_size": int(model_config.hidden_size),
            "history_encoder": str(model_config.history_encoder),
            "history_layers": int(model_config.history_layers),
            "history_heads": int(model_config.history_heads),
            "role_embedding_dim": int(model_config.role_embedding_dim),
            "belief_enabled": bool(model_config.belief_enabled),
            "human_prior_enabled": bool(model_config.human_prior_enabled),
            "style_enabled": bool(model_config.style_enabled),
            "strategy_features_enabled": bool(
                model_config.strategy_features_enabled
            ),
            "strategy_aux_enabled": bool(model_config.strategy_aux_enabled),
            "bidding_enabled": bool(model_config.bidding_enabled),
            "bidding_hidden_size": int(model_config.bidding_hidden_size),
            "bidding_uncertainty_enabled": bool(
                model_config.bidding_uncertainty_enabled
            ),
        },
        "loss": {
            name: getattr(loss_config, name)
            for name in (
                "lambda_win",
                "lambda_score",
                "lambda_uncertainty",
                "lambda_bc",
                "lambda_min_turns",
                "lambda_regain_initiative",
                "lambda_teammate_finish",
                "lambda_spring",
                "lambda_structure",
                "lambda_bid_policy",
                "lambda_bid_win",
                "lambda_bid_score",
                "lambda_bid_regret",
                "score_delta",
                "score_target_transform",
                "score_clamp",
            )
        },
        "decision_policy": {
            name: getattr(decision_config, name)
            for name in (
                "mode",
                "abs_tol",
                "rel_tol",
                "risk_penalty",
                "prior_alpha",
            )
        },
        "bidding": {
            "enabled": bool(bidding_enabled),
            "policy": (
                str(bidding_policy.policy) if bidding_policy is not None else "rule"
            ),
            "warm_start_policy": (
                str(bidding_policy.warm_start_policy)
                if bidding_policy is not None
                else "rule"
            ),
            "learned_probability": (
                float(bidding_policy.learned_probability)
                if bidding_policy is not None
                else 0.0
            ),
        },
    }
    if not is_dataclass(trainer_config):
        raise TypeError("trainer_config must be a dataclass instance")
    benchmark_workload = {
        # TrainerConfig deliberately contains no paths or logging destinations,
        # so every field is workload- or trajectory-relevant and is bound.
        "trainer_config": asdict(trainer_config),
        "execution": {
            "ddp_enabled": bool(ddp_enabled),
            "ddp_backend": str(ddp_backend),
            "world_size": int(world_size),
            "compile_model": bool(compile_model),
        },
    }
    return {
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "training_semantics": training_semantics,
        "benchmark_workload": benchmark_workload,
    }


STANDARD_V2_R1_EXPECTED_TRAINING_SEMANTICS: dict[str, Any] = {
    "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
    "runtime": {
        "ruleset": "standard",
        "feature_version": "v2",
        "model_version": "v2",
        "seed": 0,
        "deterministic": False,
        "batch_size": 32,
        "bidding_batch_size": 32,
        "bidding_update_interval": 1,
        "exp_epsilon": 0.05,
        "max_grad_norm": 40.0,
        "amp_enabled": False,
        "amp_dtype": "float16",
        "amp_fallback_on_nonfinite": True,
        "ddp_enabled": False,
        "ddp_backend": "auto",
        "compile_model": False,
        "first_bidder_mode": "rotate",
    },
    "optimizer": {
        "learning_rate": 0.0001,
        "alpha": 0.99,
        "momentum": 0.0,
        "epsilon": 0.00001,
    },
    "model": {
        "version": "v2",
        "hidden_size": 256,
        "history_encoder": "transformer",
        "history_layers": 4,
        "history_heads": 8,
        "role_embedding_dim": 32,
        "belief_enabled": False,
        "human_prior_enabled": False,
        "style_enabled": False,
        "strategy_features_enabled": False,
        "strategy_aux_enabled": False,
        "bidding_enabled": True,
        "bidding_hidden_size": 128,
        "bidding_uncertainty_enabled": False,
    },
    "loss": {
        "lambda_win": 1.0,
        "lambda_score": 0.5,
        "lambda_uncertainty": 0.0,
        "lambda_bc": 0.0,
        "lambda_min_turns": 0.0,
        "lambda_regain_initiative": 0.0,
        "lambda_teammate_finish": 0.0,
        "lambda_spring": 0.0,
        "lambda_structure": 0.0,
        "lambda_bid_policy": 1.0,
        "lambda_bid_win": 0.5,
        "lambda_bid_score": 0.25,
        "lambda_bid_regret": 0.0,
        "score_delta": 1.0,
        "score_target_transform": "raw",
        "score_clamp": 32.0,
    },
    "decision_policy": {
        "mode": "pure_win",
        "abs_tol": 0.0,
        "rel_tol": 0.0,
        "risk_penalty": 0.0,
        "prior_alpha": 0.0,
    },
    "bidding": {
        "enabled": True,
        "policy": "learned",
        "warm_start_policy": "rule",
        "learned_probability": 0.1,
    },
}

STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD: dict[str, Any] = {
    "trainer_config": {
        "seed": 0,
        "max_episodes": 16,
        "max_steps_per_episode": 600,
        "exp_epsilon": 0.05,
        "batch_size": 32,
        "bidding_batch_size": 32,
        "bidding_update_interval": 1,
        "learning_rate": 0.0001,
        "rmsprop_alpha": 0.99,
        "rmsprop_momentum": 0.0,
        "rmsprop_epsilon": 0.00001,
        "max_grad_norm": 40.0,
        "optimizer_steps": 1,
        "buffer_capacity": 4096,
        "rng_seed": 0,
        "device": "cuda",
        "amp_enabled": False,
        "amp_dtype": "float16",
        "amp_fallback_on_nonfinite": True,
        "belief_training_mode": "frozen",
        "belief_supervised_weight": 0.0,
        "belief_alternating_interval": 1,
        "belief_supervised_batch_size": 16,
        "first_bidder_mode": "rotate",
        "v2_training_mode": "single_process",
        "num_actors": 1,
        "games_per_actor": 4,
        "replay_schema_version": 1,
        "snapshot_publication_semantics": (
            "cycle_quiescent_atomic_copy_v1"
        ),
        "request_ordering_semantics": (
            "policy_inference_bucket_interleaved_games_v3"
        ),
        "async_protocol_version": BASE_ASYNC_PROTOCOL_VERSION,
        "compact_bidding_replay_schema_version": (
            BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION
        ),
        "episode_task_semantics": BASE_EPISODE_TASK_SEMANTICS,
        "episode_commit_semantics": BASE_EPISODE_COMMIT_SEMANTICS,
    },
    "execution": {
        "ddp_enabled": False,
        "ddp_backend": "auto",
        "world_size": 1,
        "compile_model": False,
    },
}

STANDARD_V2_R1_EXPECTED_CONFIG: dict[str, Any] = {
    "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
    "training_semantics": STANDARD_V2_R1_EXPECTED_TRAINING_SEMANTICS,
    "benchmark_workload": STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD,
}
STANDARD_V2_R1_TRAINING_SEMANTICS_HASH = stable_identity_hash(
    STANDARD_V2_R1_EXPECTED_TRAINING_SEMANTICS
)
STANDARD_V2_R1_BENCHMARK_WORKLOAD_HASH = stable_identity_hash(
    STANDARD_V2_R1_EXPECTED_BENCHMARK_WORKLOAD
)
STANDARD_V2_R1_CONFIG_HASH = stable_identity_hash(
    STANDARD_V2_R1_EXPECTED_CONFIG
)


def validate_standard_v2_r1_config(config: Any) -> dict[str, Any]:
    """Fail closed when the production YAML drifts from the frozen matrix."""

    identity = standard_v2_r1_config_identity(config)
    if identity != STANDARD_V2_R1_EXPECTED_TRAINING_SEMANTICS:
        raise ValueError(
            "Standard V2 R1 training semantics drifted from the frozen contract: "
            f"actual={stable_identity_hash(identity)}, "
            f"expected={STANDARD_V2_R1_TRAINING_SEMANTICS_HASH}"
        )
    return identity


def standard_v2_benchmark_identity(
    config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify an actual resolved config without relabelling non-R1 runs."""

    actual_hash = stable_identity_hash(config_identity)
    is_r1 = dict(config_identity) == STANDARD_V2_R1_EXPECTED_CONFIG
    semantics = config_identity.get("training_semantics")
    workload = config_identity.get("benchmark_workload")
    semantics_hash = (
        stable_identity_hash(semantics)
        if isinstance(semantics, Mapping)
        else None
    )
    workload_hash = (
        stable_identity_hash(workload)
        if isinstance(workload, Mapping)
        else None
    )
    return {
        "schema_version": STANDARD_V2_BENCHMARK_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": actual_hash,
        "training_semantics_hash": semantics_hash,
        "benchmark_workload_hash": workload_hash,
        "reference_digest": (
            STANDARD_V2_R1_REFERENCE_DIGEST if is_r1 else None
        ),
        "qualification": "r1" if is_r1 else "non_r1",
    }


def standard_v2_version_contract() -> dict[str, Any]:
    """Return the complete current/reserved async version registry."""

    return {
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "current_base_async": {
            "protocol_version": BASE_ASYNC_PROTOCOL_VERSION,
            "protocol_semantics": BASE_ASYNC_PROTOCOL_SEMANTICS,
            "compact_bidding_replay_schema_version": (
                BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION
            ),
            "episode_task_semantics": BASE_EPISODE_TASK_SEMANTICS,
            "episode_commit_semantics": BASE_EPISODE_COMMIT_SEMANTICS,
        },
        "reserved_standard_async": {
            "protocol_version": STANDARD_ASYNC_PROTOCOL_VERSION,
            "protocol_semantics": STANDARD_ASYNC_PROTOCOL_SEMANTICS,
            "compact_bidding_replay_schema_version": (
                STANDARD_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION
            ),
            "episode_task_semantics": STANDARD_EPISODE_TASK_SEMANTICS,
            "episode_commit_semantics": STANDARD_EPISODE_COMMIT_SEMANTICS,
            "snapshot_publication_semantics": (
                STANDARD_SNAPSHOT_PUBLICATION_SEMANTICS
            ),
        },
    }
