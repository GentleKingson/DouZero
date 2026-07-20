"""Configuration loading and conversion utilities (P01).

The legacy training entry point (``douzero.dmc.train``) consumes an
``argparse.Namespace`` whose attributes are the 23 legacy flags plus the
optimizer flags. To preserve that contract exactly, this module converts
between ``TrainingConfig`` and ``argparse.Namespace``.

Precedence (highest wins): CLI flags > YAML config file > dataclass defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from douzero.config.schemas import (
    BCConfig,
    BiddingConfig,
    CurriculumConfig,
    DistillationConfig,
    DecisionPolicyConfig,
    LossConfig,
    LeagueConfig,
    ModelConfig,
    OptimizerConfig,
    SearchConfig,
    TrainingConfig,
)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> dict:
    """Load a YAML file into a dict without requiring PyYAML at import time.

    PyYAML (``pyyaml``) is a declared runtime dependency (see ``pyproject.toml``
    ``[project] dependencies``). It is imported lazily here so that plain
    ``--help`` and module imports never require it; only ``--config <yaml>`` does.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "PyYAML (pyyaml) is a declared dependency of douzero and is required "
            "to load YAML configs. It should be installed automatically; if it is "
            "missing, run `pip install pyyaml`. The import is lazy so that plain "
            "`--help` and module imports never require it."
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data)}")
    return dict(data)


def load_config(yaml_path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    Unknown keys raise; missing keys fall back to the dataclass defaults.
    Nested ``optimizer`` mapping is supported.
    """
    raw = _load_yaml(yaml_path)
    return _build_training_config(raw)


def load_legacy_config() -> TrainingConfig:
    """Load the bundled legacy config (shipped inside the wheel).

    Uses importlib.resources so it works from an installed wheel (no repo
    checkout needed). The bundled file is douzero/config/data/legacy.yaml and
    is identical to the repo's configs/legacy.yaml.
    """
    from importlib.resources import files

    legacy_path = files("douzero.config.data").joinpath("legacy.yaml")
    with legacy_path.open("r", encoding="utf-8") as fh:
        import yaml

        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise RuntimeError("bundled legacy.yaml is not a mapping")
    return _build_training_config(raw)


# --------------------------------------------------------------------------- #
# dict <-> dataclass
# --------------------------------------------------------------------------- #
def _build_training_config(raw: Mapping[str, Any]) -> TrainingConfig:
    """Construct a TrainingConfig from a raw mapping, validating keys."""
    nested_blocks = (
        "optimizer", "loss", "decision_policy", "model", "bc", "bidding",
        "distillation", "league", "curriculum", "search",
    )
    valid_top = {
        f.name for f in fields(TrainingConfig) if f.name not in nested_blocks
    }
    valid_opt_names = {f.name for f in fields(OptimizerConfig)}
    valid_loss_names = {f.name for f in fields(LossConfig)}
    valid_decision_names = {f.name for f in fields(DecisionPolicyConfig)}
    valid_model_names = {f.name for f in fields(ModelConfig)}
    valid_bc_names = {f.name for f in fields(BCConfig)}
    valid_bidding_names = {f.name for f in fields(BiddingConfig)}
    valid_distillation_names = {f.name for f in fields(DistillationConfig)}
    valid_league_names = {f.name for f in fields(LeagueConfig)}
    valid_curriculum_names = {f.name for f in fields(CurriculumConfig)}
    valid_search_names = {f.name for f in fields(SearchConfig)}

    # 'optimizer', 'loss', 'decision_policy', 'model', 'bc', and 'rules' are
    # valid top-level keys (handled separately).
    unknown_top = set(raw.keys()) - valid_top - set(nested_blocks) - {"rules"}
    if unknown_top:
        raise ValueError(f"Unknown config keys: {sorted(unknown_top)}")

    optimizer_raw = raw.get("optimizer", {})
    if not isinstance(optimizer_raw, Mapping):
        raise TypeError("'optimizer' must be a mapping")
    unknown_opt = set(optimizer_raw.keys()) - valid_opt_names
    if unknown_opt:
        raise ValueError(f"Unknown optimizer config keys: {sorted(unknown_opt)}")

    loss_raw = raw.get("loss", {})
    if not isinstance(loss_raw, Mapping):
        raise TypeError("'loss' must be a mapping")
    unknown_loss = set(loss_raw.keys()) - valid_loss_names
    if unknown_loss:
        raise ValueError(f"Unknown loss config keys: {sorted(unknown_loss)}")

    decision_raw = raw.get("decision_policy", {})
    if not isinstance(decision_raw, Mapping):
        raise TypeError("'decision_policy' must be a mapping")
    unknown_decision = set(decision_raw.keys()) - valid_decision_names
    if unknown_decision:
        raise ValueError(
            f"Unknown decision_policy config keys: {sorted(unknown_decision)}"
        )

    model_raw = raw.get("model", {})
    if not isinstance(model_raw, Mapping):
        raise TypeError("'model' must be a mapping")
    unknown_model = set(model_raw.keys()) - valid_model_names
    if unknown_model:
        raise ValueError(f"Unknown model config keys: {sorted(unknown_model)}")

    # P08: 'bc' block (behaviour-cloning prior).
    bc_raw = raw.get("bc", {})
    if not isinstance(bc_raw, Mapping):
        raise TypeError("'bc' must be a mapping")
    unknown_bc = set(bc_raw.keys()) - valid_bc_names
    if unknown_bc:
        raise ValueError(f"Unknown bc config keys: {sorted(unknown_bc)}")

    bidding_raw = raw.get("bidding", {})
    if not isinstance(bidding_raw, Mapping):
        raise TypeError("'bidding' must be a mapping")
    unknown_bidding = set(bidding_raw.keys()) - valid_bidding_names
    if unknown_bidding:
        raise ValueError(
            f"Unknown bidding config keys: {sorted(unknown_bidding)}"
        )

    distillation_raw = raw.get("distillation", {})
    if not isinstance(distillation_raw, Mapping):
        raise TypeError("'distillation' must be a mapping")
    unknown_distillation = set(distillation_raw.keys()) - valid_distillation_names
    if unknown_distillation:
        raise ValueError(
            f"Unknown distillation config keys: {sorted(unknown_distillation)}"
        )

    league_raw = raw.get("league", {})
    if not isinstance(league_raw, Mapping):
        raise TypeError("'league' must be a mapping")
    unknown_league = set(league_raw.keys()) - valid_league_names
    if unknown_league:
        raise ValueError(f"Unknown league config keys: {sorted(unknown_league)}")

    curriculum_raw = raw.get("curriculum", {})
    if not isinstance(curriculum_raw, Mapping):
        raise TypeError("'curriculum' must be a mapping")
    unknown_curriculum = set(curriculum_raw.keys()) - valid_curriculum_names
    if unknown_curriculum:
        raise ValueError(
            f"Unknown curriculum config keys: {sorted(unknown_curriculum)}"
        )

    search_raw = raw.get("search", {})
    if not isinstance(search_raw, Mapping):
        raise TypeError("'search' must be a mapping")
    unknown_search = set(search_raw.keys()) - valid_search_names
    if unknown_search:
        raise ValueError(f"Unknown search config keys: {sorted(unknown_search)}")

    # P06 r6: unify top-level ``model_version`` and nested ``model.version``
    # into a single source of truth. Without this, a YAML like
    # ``model_version: v2`` + ``model: {version: legacy}`` would be accepted,
    # with the identity gate reading ``model_version`` while the model
    # constructor reads ``model.*``.
    top_version = raw.get("model_version", TrainingConfig.__dataclass_fields__["model_version"].default)
    has_explicit_nested = "version" in model_raw
    nested_version = model_raw.get("version", top_version)
    if nested_version != top_version:
        raise ValueError(
            f"model.version ({nested_version!r}) must match model_version "
            f"({top_version!r}). The top-level and nested model identity "
            f"fields must agree; set only one or ensure both are the same."
        )
    # Validate an EXPLICITLY provided nested version against the supported
    # set. When no model: block is present, the top-level version validation
    # is handled by _validate_legacy_only_versions downstream.
    if has_explicit_nested and nested_version not in _LEGACY_ONLY_VERSIONS["model_version"]:
        raise ValueError(
            f"model.version has unsupported value {nested_version!r}. "
            f"Supported values are {sorted(_LEGACY_ONLY_VERSIONS['model_version'])}."
        )
    # Ensure model_raw carries the resolved version so ModelConfig picks it up.
    model_raw = dict(model_raw)
    model_raw["version"] = nested_version

    # P02: a 'rules' block is accepted but not stored on TrainingConfig (which
    # only carries the version string). We validate it here so a malformed
    # rules block fails loudly, and so the ruleset version string is
    # cross-checked against it.
    rules_raw = raw.get("rules")
    if rules_raw is not None:
        from douzero.env.rules import RuleSet
        RuleSet.from_dict(rules_raw)  # validates types/ranges; result discarded

    kwargs: dict[str, Any] = {}
    for name in valid_top:
        if name in raw:
            kwargs[name] = raw[name]
    if optimizer_raw:
        kwargs["optimizer"] = OptimizerConfig(**dict(optimizer_raw))
    if loss_raw:
        kwargs["loss"] = LossConfig(**dict(loss_raw))
    if decision_raw:
        kwargs["decision_policy"] = DecisionPolicyConfig(**dict(decision_raw))
    if model_raw:
        kwargs["model"] = ModelConfig(**dict(model_raw))
    if bc_raw:
        kwargs["bc"] = BCConfig(**dict(bc_raw))
    if bidding_raw:
        kwargs["bidding"] = BiddingConfig(**dict(bidding_raw))
    if distillation_raw:
        kwargs["distillation"] = DistillationConfig(**dict(distillation_raw))
    if league_raw:
        kwargs["league"] = LeagueConfig(**dict(league_raw))
    if curriculum_raw:
        kwargs["curriculum"] = CurriculumConfig(**dict(curriculum_raw))
    if search_raw:
        kwargs["search"] = SearchConfig(**dict(search_raw))
    cfg = TrainingConfig(**kwargs)
    _validate_types(cfg)
    _validate_training_system(cfg)
    _validate_legacy_only_versions(cfg)
    return cfg


# Expected runtime types per field, for validating YAML/dict input. Booleans
# must be real bools (YAML bool, not the string "true"); ints/floats must be
# numbers; strings must be str. This catches wrong-type YAML values that a
# frozen dataclass would otherwise silently accept.
_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "xpid": str, "save_interval": int, "checkpoint_sidecar_retention": int,
    "objective": str,
    "actor_device_cpu": bool, "gpu_devices": str, "num_actor_devices": int,
    "num_actors": int, "games_per_actor": int,
    "training_device": str, "load_model": bool,
    "disable_checkpoint": bool, "savedir": str, "total_frames": int,
    "exp_epsilon": float, "batch_size": int,
    "bidding_batch_size": (int, type(None)), "bidding_update_interval": int,
    "unroll_length": int,
    "num_buffers": int, "num_threads": int, "max_grad_norm": float,
    "v2_training_mode": str,
    "sync_interval_updates": int, "policy_snapshot_slots": int,
    "amp_enabled": bool, "amp_dtype": str,
    "amp_fallback_on_nonfinite": bool, "pin_memory": bool,
    "ddp_enabled": bool, "ddp_backend": str, "compile_model": bool,
    "legacy_actor_backend": str, "actor_torch_threads": int,
    "legacy_contiguous_buffers": bool, "legacy_bulk_rollout": bool,
    "legacy_flush_ge": bool, "legacy_reusable_pinned_staging": bool,
    "legacy_log_interval_seconds": float,
    "legacy_monitor_interval_seconds": float,
    "legacy_profile": bool, "legacy_profile_sample_interval": int,
    "legacy_metrics_path": str, "benchmark_warmup_frames": int,
    "compile_actor": bool, "compile_learner": bool,
    "rmsprop_foreach": bool, "grad_clip_foreach": bool,
    "central_actor_max_actions": int, "central_actor_microbatch": int,
    "central_actor_envs_per_actor": int,
    "central_actor_min_microbatch": int,
    "central_actor_target_microbatch": int,
    "central_actor_max_microbatch": int,
    "central_actor_max_delay_ms": float,
    "central_actor_max_pending_requests": int,
    "central_actor_queue_high_watermark": int,
    "central_actor_inference_deadline_ms": float,
    "central_actor_learner_throttle": bool,
    "central_actor_learner_throttle_mode": str,
    "central_actor_predicted_drain_target_ms": float,
    "central_actor_use_stream_priority": bool,
    "central_actor_async_policy_copy": bool,
    "central_actor_runtime": str,
    "central_actor_split_dense1": bool,
    "central_actor_staging_dtype": str,
    "central_actor_inference_layout": str,
    "central_actor_timeout_seconds": float,
    "belief_training_mode": str, "belief_supervised_weight": float,
    "belief_alternating_interval": int, "belief_supervised_batch_size": int,
    "belief_supervised_episodes": int,
    "first_bidder_mode": str,
    "seed": int, "deterministic": bool, "config": str,
    "feature_version": str, "ruleset": str, "model_version": str,
    "learning_rate": float, "alpha": float, "momentum": float, "epsilon": float,
    # P06 multi-objective loss + decision-policy nested fields.
    "lambda_win": float, "lambda_score": float,
    "lambda_uncertainty": float, "score_delta": float,
    "lambda_bc": float, "lambda_bid_policy": float, "lambda_bid_win": float,
    "lambda_bid_score": float, "lambda_bid_regret": float,
    "lambda_min_turns": float,
    "lambda_regain_initiative": float, "lambda_teammate_finish": float,
    "lambda_spring": float, "lambda_structure": float,
    "score_target_transform": str, "score_clamp": float,
    "mode": str, "abs_tol": float, "rel_tol": float, "risk_penalty": float,
    "prior_alpha": float,
    # P06 r5: V2 model architecture nested fields.
    "version": str, "hidden_size": int, "history_encoder": str,
    "history_layers": int, "history_heads": int, "role_embedding_dim": int,
    "belief_enabled": bool, "human_prior_enabled": bool,
    "bidding_enabled": bool, "bidding_hidden_size": int,
    "bidding_uncertainty_enabled": bool,
    "style_enabled": bool, "style_embedding_dim": int,
    "strategy_features_enabled": bool, "strategy_hand_enabled": bool,
    "strategy_structure_enabled": bool, "strategy_control_enabled": bool,
    "strategy_cooperation_enabled": bool, "strategy_risk_enabled": bool,
    "strategy_aux_enabled": bool, "strategy_node_budget": int,
    "strategy_time_budget_ms": int,
    # P08: behaviour-cloning nested fields. (Blocker 3: ``enabled`` and
    # ``lambda_bc`` were removed from BCConfig — ``loss.lambda_bc`` is the sole
    # enable condition / weight, so they are no longer valid bc: keys.)
    "data_path": str,
    "policy": str, "warm_start_policy": str, "learned_probability": float,
    "temperature": float, "label_smoothing": float,
    "skill_weight_clip": float, "schedule": str,
    "schedule_steps": int, "schedule_floor": float,
    # P10: privileged-teacher distillation block.
    "enabled": bool, "teacher_checkpoint": str, "dataset_path": str,
    "cache_path": str, "batch_size": int, "top_k": int, "lambda_kl": float,
    "distillation_temperature": float,
    "lambda_rank": float, "lambda_teacher_win": float,
    "lambda_teacher_score": float, "lambda_supervised_win": float,
    "lambda_supervised_score": float,
}


def _check_field(name: str, value: Any, source: str) -> None:
    expected = _FIELD_TYPES.get(name)
    if expected is None:
        return
    # bool is a subclass of int; for int fields we must reject bools, and for
    # float fields we accept int (numpy/JSON style) but reject bool.
    if expected == (int, type(None)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(
                f"Config field {name!r} ({source}) must be int or null, got "
                f"{type(value).__name__}: {value!r}"
            )
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Config field {name!r} ({source}) must be int, got "
                f"{type(value).__name__}: {value!r}"
            )
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Config field {name!r} ({source}) must be float, got "
                f"{type(value).__name__}: {value!r}"
            )
    elif expected is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"Config field {name!r} ({source}) must be bool, got "
                f"{type(value).__name__}: {value!r}"
            )
    else:
        if not isinstance(value, expected):
            raise TypeError(
                f"Config field {name!r} ({source}) must be {expected.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )


def _validate_types(cfg: TrainingConfig) -> None:
    for name in _FIELD_TYPES:
        if name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            _check_field(name, getattr(cfg.optimizer, name), "optimizer")
        elif name in {
            "lambda_win", "lambda_score", "lambda_uncertainty", "lambda_bc",
            "lambda_bid_policy", "lambda_bid_win", "lambda_bid_score",
            "lambda_bid_regret",
            "lambda_min_turns", "lambda_regain_initiative",
            "lambda_teammate_finish", "lambda_spring", "lambda_structure",
            "score_delta", "score_clamp",
        }:
            _check_field(name, getattr(cfg.loss, name), "loss")
        elif name == "score_target_transform":
            _check_field(name, getattr(cfg.loss, name), "loss")
        elif name in {"mode", "abs_tol", "rel_tol", "risk_penalty", "prior_alpha"}:
            _check_field(name, getattr(cfg.decision_policy, name), "decision_policy")
        elif name in {
            "version", "hidden_size", "history_encoder", "history_layers",
            "history_heads", "role_embedding_dim", "belief_enabled",
            "human_prior_enabled", "bidding_enabled", "bidding_hidden_size",
            "bidding_uncertainty_enabled",
            "style_enabled", "style_embedding_dim",
            "strategy_features_enabled", "strategy_hand_enabled",
            "strategy_structure_enabled", "strategy_control_enabled",
            "strategy_cooperation_enabled", "strategy_risk_enabled",
            "strategy_aux_enabled", "strategy_node_budget",
            "strategy_time_budget_ms",
        }:
            _check_field(name, getattr(cfg.model, name), "model")
        elif name in {
            "data_path", "temperature", "label_smoothing",
            "skill_weight_clip", "schedule", "schedule_steps", "schedule_floor",
        }:
            _check_field(name, getattr(cfg.bc, name), "bc")
        elif name in {
            "enabled", "teacher_checkpoint", "dataset_path", "cache_path",
            "distillation_temperature", "top_k", "lambda_kl", "lambda_rank", "lambda_teacher_win",
            "lambda_teacher_score", "lambda_supervised_win",
            "lambda_supervised_score",
        }:
            _check_field(name, getattr(cfg.distillation, name), "distillation")
        elif hasattr(cfg, name):
            _check_field(name, getattr(cfg, name), "training")
    # ``batch_size`` exists at both the top-level learner and P10 distillation
    # scopes, so validate the nested value separately instead of shadowing the
    # established top-level field in the name-based dispatch above.
    _check_field("batch_size", cfg.distillation.batch_size, "distillation")


def _validate_training_system(cfg: TrainingConfig) -> None:
    """Validate P14 concurrency and precision controls."""
    import math

    if cfg.v2_training_mode not in {"single_process", "async_single_gpu"}:
        raise ValueError(
            "v2_training_mode must be 'single_process' or 'async_single_gpu'"
        )
    if cfg.num_actors < 1 or cfg.games_per_actor < 1:
        raise ValueError("num_actors and games_per_actor must be >= 1")
    if cfg.checkpoint_sidecar_retention < -1:
        raise ValueError("checkpoint_sidecar_retention must be -1 or greater")
    if cfg.sync_interval_updates < 1:
        raise ValueError("sync_interval_updates must be >= 1")
    if cfg.policy_snapshot_slots < 2:
        raise ValueError("policy_snapshot_slots must be >= 2")
    if cfg.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
    if cfg.ddp_backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("ddp_backend must be 'auto', 'nccl', or 'gloo'")
    if cfg.legacy_actor_backend not in {
        "legacy", "factorized", "centralized_factorized"
    }:
        raise ValueError("legacy_actor_backend is unsupported")
    if cfg.actor_torch_threads < 0:
        raise ValueError("actor_torch_threads must be >= 0")
    if cfg.legacy_log_interval_seconds < 0:
        raise ValueError("legacy_log_interval_seconds must be non-negative")
    if cfg.legacy_monitor_interval_seconds <= 0:
        raise ValueError("legacy_monitor_interval_seconds must be positive")
    if cfg.legacy_profile_sample_interval < 1:
        raise ValueError("legacy_profile_sample_interval must be >= 1")
    if cfg.benchmark_warmup_frames < 0:
        raise ValueError("benchmark_warmup_frames must be non-negative")
    if cfg.central_actor_max_actions < 64:
        raise ValueError("central_actor_max_actions must be >= 64")
    if cfg.central_actor_microbatch < 1:
        raise ValueError("central_actor_microbatch must be >= 1")
    if cfg.central_actor_envs_per_actor < 1:
        raise ValueError("central_actor_envs_per_actor must be >= 1")
    if not (1 <= cfg.central_actor_min_microbatch
            <= cfg.central_actor_target_microbatch
            <= cfg.central_actor_max_microbatch):
        raise ValueError(
            "central actor microbatches must satisfy 1 <= min <= target <= max"
        )
    if cfg.central_actor_max_delay_ms < 0:
        raise ValueError("central_actor_max_delay_ms must be non-negative")
    if cfg.central_actor_max_pending_requests < cfg.central_actor_max_microbatch:
        raise ValueError(
            "central_actor_max_pending_requests must be >= max microbatch"
        )
    if not (1 <= cfg.central_actor_queue_high_watermark
            <= cfg.central_actor_max_pending_requests):
        raise ValueError(
            "central_actor_queue_high_watermark must be within queue capacity"
        )
    if cfg.central_actor_inference_deadline_ms <= 0:
        raise ValueError("central_actor_inference_deadline_ms must be positive")
    if cfg.central_actor_learner_throttle_mode not in {
        "off", "fixed_threshold", "predicted_drain_time"
    }:
        raise ValueError("invalid central_actor_learner_throttle_mode")
    if cfg.central_actor_predicted_drain_target_ms <= 0:
        raise ValueError(
            "central_actor_predicted_drain_target_ms must be positive"
        )
    if cfg.central_actor_runtime not in {"process", "thread"}:
        raise ValueError("central_actor_runtime must be process or thread")
    if cfg.central_actor_staging_dtype not in {"float32", "int8"}:
        raise ValueError("central_actor_staging_dtype must be float32 or int8")
    if cfg.central_actor_inference_layout not in {"packed", "padded"}:
        raise ValueError("central_actor_inference_layout must be packed or padded")
    if cfg.central_actor_timeout_seconds <= 0:
        raise ValueError("central_actor_timeout_seconds must be positive")
    if cfg.belief_training_mode not in {"frozen", "joint", "alternating"}:
        raise ValueError(
            "belief_training_mode must be 'frozen', 'joint', or 'alternating'"
        )
    if (
        not math.isfinite(cfg.belief_supervised_weight)
        or cfg.belief_supervised_weight < 0
    ):
        raise ValueError("belief_supervised_weight must be non-negative")
    if cfg.belief_alternating_interval < 1:
        raise ValueError("belief_alternating_interval must be >= 1")
    if cfg.belief_supervised_batch_size < 1:
        raise ValueError("belief_supervised_batch_size must be >= 1")
    if cfg.belief_supervised_episodes < 0:
        raise ValueError("belief_supervised_episodes must be non-negative")
    if cfg.first_bidder_mode not in {"rotate", "seeded_random"}:
        raise ValueError("first_bidder_mode must be 'rotate' or 'seeded_random'")
    if cfg.bidding_batch_size is not None and cfg.bidding_batch_size < 1:
        raise ValueError("bidding_batch_size must be >= 1 when set")
    if cfg.bidding_update_interval < 1:
        raise ValueError("bidding_update_interval must be >= 1")


# P01 only supports the "legacy" feature/rule/model versions. Later phases
# (P02 rules, P03 observations, P05 model) widen these sets. A YAML or dict
# config that sets a non-"legacy" value is rejected here so it fails loudly
# rather than silently producing a run the codebase does not support.
_LEGACY_ONLY_VERSIONS: dict[str, frozenset[str]] = {
    # P03 widens the feature_version allowed set to include "v2". The V2
    # observation schema is opt-in; the legacy encoder remains the default and
    # is byte-for-byte unchanged. Training still uses the legacy observation
    # path until P05/P06 wire the V2 model and multi-objective losses.
    "feature_version": frozenset({"legacy", "v2"}),
    # P02 widens the ruleset allowed set to include "standard".
    "ruleset": frozenset({"legacy", "standard"}),
    # P04 widened model_version to "factorized" (deployment-only,
    # checkpoint-compatible forward). P05 widens it further to "v2", the
    # shared state/action model with multi-head outputs. "v2" is accepted by
    # the config layer so DeepAgentV2 and model construction can be selected;
    # the training gate in dmc.py still rejects v2 training until P06 wires
    # multi-objective losses and the actor/learner loop to it.
    "model_version": frozenset({"legacy", "factorized", "v2"}),
}


def _validate_legacy_only_versions(cfg: TrainingConfig) -> None:
    """Reject version identifiers this codebase does not support.

    The argparse flags enforce ``choices`` for CLI input; this validator covers
    the YAML/dict path so a config file cannot smuggle in an unsupported
    version either. The allowed sets are widened per phase: P02 added
    ``ruleset="standard"``, P03 added ``feature_version="v2"``.
    """
    for name, allowed in _LEGACY_ONLY_VERSIONS.items():
        val = getattr(cfg, name)
        if val not in allowed:
            raise ValueError(
                f"Config field {name!r} has unsupported value {val!r}. "
                f"Supported values are {sorted(allowed)}. Later phases widen "
                f"the allowed set."
            )


def serialize(cfg: TrainingConfig) -> dict:
    """Convert a TrainingConfig to a JSON/YAML-serializable dict."""
    return asdict(cfg)


# --------------------------------------------------------------------------- #
# argparse Namespace <-> TrainingConfig
# --------------------------------------------------------------------------- #
# The set of attribute names train(flags) reads off the Namespace. These are
# the EXACT argparse dests from douzero/dmc/arguments.py.
_TRAINING_NAMESPACE_FIELDS: tuple[str, ...] = (
    "xpid", "save_interval", "checkpoint_sidecar_retention", "objective",
    "actor_device_cpu", "gpu_devices", "num_actor_devices", "num_actors",
    "training_device", "load_model", "disable_checkpoint", "savedir",
    "total_frames", "exp_epsilon", "batch_size", "bidding_batch_size",
    "bidding_update_interval", "unroll_length",
    "num_buffers", "num_threads", "max_grad_norm",
    "sync_interval_updates", "policy_snapshot_slots", "amp_enabled",
    "amp_dtype", "amp_fallback_on_nonfinite", "pin_memory",
    "ddp_enabled", "ddp_backend", "compile_model",
    "legacy_actor_backend", "actor_torch_threads",
    "legacy_contiguous_buffers", "legacy_bulk_rollout", "legacy_flush_ge",
    "legacy_reusable_pinned_staging", "legacy_log_interval_seconds",
    "legacy_monitor_interval_seconds", "legacy_profile",
    "legacy_profile_sample_interval", "legacy_metrics_path",
    "benchmark_warmup_frames", "compile_actor", "compile_learner",
    "rmsprop_foreach", "grad_clip_foreach",
    "central_actor_max_actions", "central_actor_microbatch",
    "central_actor_envs_per_actor", "central_actor_min_microbatch",
    "central_actor_target_microbatch", "central_actor_max_microbatch",
    "central_actor_max_delay_ms", "central_actor_max_pending_requests",
    "central_actor_queue_high_watermark", "central_actor_inference_deadline_ms",
    "central_actor_learner_throttle", "central_actor_use_stream_priority",
    "central_actor_learner_throttle_mode",
    "central_actor_predicted_drain_target_ms",
    "central_actor_async_policy_copy", "central_actor_timeout_seconds",
    "central_actor_runtime",
    "central_actor_split_dense1",
    "central_actor_staging_dtype",
    "central_actor_inference_layout",
    "learning_rate", "alpha", "momentum", "epsilon",
    # P01-added argparse dests (optional; default to legacy values if absent).
    "seed", "deterministic", "config",
    # Version identifiers carried through config <-> Namespace (item 4).
    "feature_version", "ruleset", "model_version",
)


def from_argparse(ns: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from a legacy argparse Namespace.

    Optimizer fields (learning_rate/alpha/momentum/epsilon) live at the
    Namespace top level (that is how arguments.py declares them), so they are
    pulled into the nested OptimizerConfig.
    """
    opt_keys = {"learning_rate", "alpha", "momentum", "epsilon"}
    opt_kwargs = {k: getattr(ns, k) for k in opt_keys if hasattr(ns, k)}
    optimizer = OptimizerConfig(**opt_kwargs) if opt_kwargs else OptimizerConfig()

    training_kwargs: dict[str, Any] = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if name in opt_keys:
            continue
        if hasattr(ns, name):
            training_kwargs[name] = getattr(ns, name)
    training_kwargs["optimizer"] = optimizer
    return TrainingConfig(**training_kwargs)


def to_argparse_namespace(cfg: TrainingConfig) -> argparse.Namespace:
    """Convert a TrainingConfig back to a legacy argparse Namespace.

    The returned Namespace has the SAME attributes that ``train(flags)``
    reads (the optimizer fields are flattened to the top level, matching how
    arguments.py declares them). This keeps ``train.py`` unchanged: it can
    call ``train(flags)`` with this Namespace exactly as before.
    """
    d: dict[str, Any] = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if name == "config":
            d[name] = getattr(cfg, name, "")
        elif name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            d[name] = getattr(cfg.optimizer, name)
        elif hasattr(cfg, name):
            d[name] = getattr(cfg, name)
    return argparse.Namespace(**d)


# --------------------------------------------------------------------------- #
# Merge (CLI overrides YAML)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Merge (CLI overrides YAML)
# --------------------------------------------------------------------------- #
def merge(base: TrainingConfig, override_ns: argparse.Namespace) -> TrainingConfig:
    """Overlay explicitly-set CLI overrides from a Namespace onto a base config.

    Precedence: dataclass defaults < YAML (base) < explicit CLI flags.

    ``override_ns`` is expected to contain ONLY the flags the user explicitly
    typed (produced by re-parsing with default=SUPPRESS in
    ``arguments.parse_args``). Because absent flags are simply missing from the
    Namespace, every attribute present is a genuine override -- including
    ``store_true`` flags, which appear only when the user actually typed them.
    This avoids the classic "argparse default clobbers YAML" bug for booleans.
    """
    base_d = asdict(base)
    opt_overrides = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if not hasattr(override_ns, name):
            continue
        val = getattr(override_ns, name)
        if name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            opt_overrides[name] = val
        else:
            base_d[name] = val
    if opt_overrides:
        base_d["optimizer"] = {**base_d["optimizer"], **opt_overrides}
    return _build_training_config(base_d)
