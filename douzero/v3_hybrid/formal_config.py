"""Immutable P1 experiment configuration and identity contracts.

The formal configs are deliberately separate from the day-to-day trainer
defaults.  They freeze comparisons before any pilot result is inspected and
can be validated without initializing CUDA, workers, replay, or checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from douzero.env.rules import RuleSet

from .config import V3HybridModelConfig
from .formal_evidence import (
    H8A_SUPPORT_MATRIX_VERSION,
    H8A_VARIANT_RULESET_SUPPORT,
    h8a_support_matrix_hash,
)

V3_FORMAL_CONFIG_SCHEMA = "v3-formal-experiment-config-v1"
V3_FORMAL_IDENTITY_SCHEMA = "v3-formal-experiment-identity-v1"
TRAINING_SEMANTICS_VERSION = "v3-formal-training-semantics-v1"
WORKLOAD_IDENTITY_VERSION = "v3-formal-workload-v1"
V3_FORMAL_INITIAL_CHECKPOINT_SCHEMA = "v3-formal-initial-checkpoint-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

_FEATURE_NAMES = (
    "role_model", "adaptive_dmc", "oracle", "belief", "cooperation",
    "human_bc", "strategy", "style", "bidding", "search",
)
_LOSS_NAMES = (
    "dmc", "win", "score", "oracle", "belief", "cooperation", "bc",
    "strategy", "bidding",
)
_ROLE_NAMES = ("landlord", "landlord_up", "landlord_down")
_BUDGET_NAMES = ("pilot", "development", "promotion")

_EXPECTED_FEATURES = {
    "legacy_a1": set(),
    "model_v2": set(),
    "v3_role": {"role_model"},
    "v3_admc": {"role_model", "adaptive_dmc"},
    "v3_oracle": {"role_model", "oracle"},
    "v3_belief": {"role_model", "belief"},
    "v3_farmer_cooperation": {"role_model", "cooperation"},
    "v3_full_hybrid": {
        "role_model", "adaptive_dmc", "oracle", "belief", "cooperation",
        "strategy", "style",
    },
}


class FormalConfigError(ValueError):
    """Raised before runtime side effects for an invalid formal config."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False):
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise FormalConfigError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalConfigError("formal config must be canonical finite JSON") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FormalConfigError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalConfigError(f"{label} must be an object")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FormalConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise FormalConfigError(f"{label} must be finite and >= {minimum}")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise FormalConfigError(f"{label} must be bool")
    return value


@dataclass(frozen=True)
class FormalBudget:
    training_seed_count: int
    wall_clock_seconds: int
    sample_budget: int
    optimizer_step_budget: int
    paired_deals: int

    @classmethod
    def from_dict(cls, value: object, label: str) -> "FormalBudget":
        data = _mapping(value, label)
        _exact(data, set(cls.__dataclass_fields__), label)
        return cls(**{
            name: _integer(data[name], f"{label}.{name}", 1)
            for name in cls.__dataclass_fields__
        })


@dataclass(frozen=True)
class FormalSeeds:
    training: tuple[int, ...]
    evaluation: int
    deal_set: int
    derivation: str

    @classmethod
    def from_dict(cls, value: object) -> "FormalSeeds":
        data = _mapping(value, "seeds")
        _exact(data, set(cls.__dataclass_fields__), "seeds")
        training = data["training"]
        if (
            not isinstance(training, (list, tuple)) or len(training) < 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in training)
            or len(set(training)) != len(training)
        ):
            raise FormalConfigError("seeds.training must contain at least three unique ints")
        derivation = data["derivation"]
        if derivation != "sha256(root_seed,stream_name,worker_id,episode_id)-v1":
            raise FormalConfigError("unsupported RNG seed derivation contract")
        return cls(
            tuple(training),
            _integer(data["evaluation"], "seeds.evaluation"),
            _integer(data["deal_set"], "seeds.deal_set"),
            derivation,
        )


@dataclass(frozen=True)
class FormalRuntime:
    topology: str
    device: str
    batch_size: int
    replay_capacity: int
    checkpoint_cadence_updates: int
    policy_lag_limit: int
    checkpoint_enabled: bool

    @classmethod
    def from_dict(cls, value: object) -> "FormalRuntime":
        data = _mapping(value, "runtime")
        _exact(data, set(cls.__dataclass_fields__), "runtime")
        if data["topology"] != "single_process":
            raise FormalConfigError("P1 formal configs support single_process only")
        if data["device"] != "cuda":
            raise FormalConfigError("formal training device must be cuda")
        return cls(
            topology=data["topology"], device=data["device"],
            batch_size=_integer(data["batch_size"], "runtime.batch_size", 1),
            replay_capacity=_integer(data["replay_capacity"], "runtime.replay_capacity", 1),
            checkpoint_cadence_updates=_integer(
                data["checkpoint_cadence_updates"],
                "runtime.checkpoint_cadence_updates", 1,
            ),
            policy_lag_limit=_integer(data["policy_lag_limit"], "runtime.policy_lag_limit"),
            checkpoint_enabled=_boolean(data["checkpoint_enabled"], "runtime.checkpoint_enabled"),
        )


@dataclass(frozen=True)
class FormalInitialization:
    kind: str
    source: str
    seed: int
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    checkpoint_kind: str | None

    @classmethod
    def from_dict(cls, value: object) -> "FormalInitialization":
        data = _mapping(value, "initialization")
        _exact(data, set(cls.__dataclass_fields__), "initialization")
        kind = data["kind"]
        if kind not in {"seeded_fresh", "checkpoint"}:
            raise FormalConfigError("initialization.kind is unsupported")
        seed = _integer(data["seed"], "initialization.seed")
        optional = (data["checkpoint_path"], data["checkpoint_sha256"], data["checkpoint_kind"])
        if kind == "seeded_fresh" and any(item is not None for item in optional):
            raise FormalConfigError("seeded_fresh initialization cannot name a checkpoint")
        if kind == "checkpoint":
            if not all(isinstance(item, str) and item for item in optional):
                raise FormalConfigError("checkpoint initialization requires path, sha256, and kind")
            if not _HEX64.fullmatch(data["checkpoint_sha256"]):
                raise FormalConfigError("initial checkpoint sha256 is invalid")
        if not isinstance(data["source"], str) or not data["source"]:
            raise FormalConfigError("initialization.source must be non-empty")
        return cls(**dict(data))


@dataclass(frozen=True)
class FormalExperimentConfig:
    metadata: Mapping[str, str]
    variant: str
    ruleset: Mapping[str, str]
    model: Mapping[str, Any]
    features: Mapping[str, bool]
    losses: Mapping[str, Any]
    feature_configs: Mapping[str, Any]
    runtime: FormalRuntime
    seeds: FormalSeeds
    budgets: Mapping[str, FormalBudget]
    initialization: FormalInitialization
    evaluation: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "FormalExperimentConfig":
        data = _mapping(value, "config")
        expected = {
            "schema_version", "metadata", "variant", "ruleset", "model",
            "features", "losses", "feature_configs", "runtime", "seeds",
            "budgets", "initialization", "evaluation",
        }
        _exact(data, expected, "config")
        if data["schema_version"] != V3_FORMAL_CONFIG_SCHEMA:
            raise FormalConfigError("formal config schema version mismatch")
        metadata = _mapping(data["metadata"], "metadata")
        _exact(metadata, {"name", "description"}, "metadata")
        if any(not isinstance(item, str) or not item for item in metadata.values()):
            raise FormalConfigError("metadata values must be non-empty strings")
        variant = data["variant"]
        if variant not in _EXPECTED_FEATURES:
            raise FormalConfigError(f"unsupported formal variant {variant!r}")

        ruleset_data = _mapping(data["ruleset"], "ruleset")
        _exact(ruleset_data, {"id", "version", "hash"}, "ruleset")
        ruleset_id = ruleset_data["id"]
        if ruleset_id not in H8A_VARIANT_RULESET_SUPPORT[variant]:
            raise FormalConfigError(f"unsupported variant/ruleset combination: {variant}/{ruleset_id}")
        canonical_ruleset = RuleSet.legacy() if ruleset_id == "legacy" else RuleSet.standard()
        if ruleset_data != {
            "id": canonical_ruleset.ruleset_id,
            "version": canonical_ruleset.ruleset_version,
            "hash": canonical_ruleset.stable_hash(),
        }:
            raise FormalConfigError("ruleset identity does not match the canonical rule engine")

        model = _mapping(data["model"], "model")
        _exact(model, {"family", "version", "config"}, "model")
        expected_family = {
            "legacy_a1": "legacy_a1", "model_v2": "model_v2",
        }.get(variant, "v3_hybrid")
        if model["family"] != expected_family or not isinstance(model["config"], Mapping):
            raise FormalConfigError("variant/model family mismatch")
        if expected_family == "v3_hybrid":
            try:
                V3HybridModelConfig.from_dict(dict(model["config"]))
            except (TypeError, ValueError) as exc:
                raise FormalConfigError(f"invalid V3 model config: {exc}") from exc

        features = _mapping(data["features"], "features")
        _exact(features, set(_FEATURE_NAMES), "features")
        enabled = {name for name, flag in features.items() if _boolean(flag, f"features.{name}")}
        expected_features = set(_EXPECTED_FEATURES[variant])
        if variant == "v3_full_hybrid" and ruleset_id == "standard":
            expected_features.add("bidding")
        if enabled != expected_features:
            raise FormalConfigError(
                f"features do not exactly describe {variant}/{ruleset_id}: "
                f"expected={sorted(expected_features)}, got={sorted(enabled)}"
            )

        losses = _mapping(data["losses"], "losses")
        _exact(losses, {"weights", "schedules", "role_weights"}, "losses")
        weights = _mapping(losses["weights"], "losses.weights")
        schedules = _mapping(losses["schedules"], "losses.schedules")
        role_weights = _mapping(losses["role_weights"], "losses.role_weights")
        _exact(weights, set(_LOSS_NAMES), "losses.weights")
        _exact(schedules, set(_LOSS_NAMES), "losses.schedules")
        _exact(role_weights, set(_ROLE_NAMES), "losses.role_weights")
        for name in _LOSS_NAMES:
            weight = _number(weights[name], f"losses.weights.{name}")
            schedule = _mapping(schedules[name], f"losses.schedules.{name}")
            _exact(schedule, {"kind", "start", "end", "updates"}, f"losses.schedules.{name}")
            if schedule["kind"] not in {"constant", "linear"}:
                raise FormalConfigError(f"unsupported {name} loss schedule")
            _number(schedule["start"], f"losses.schedules.{name}.start")
            _number(schedule["end"], f"losses.schedules.{name}.end")
            _integer(schedule["updates"], f"losses.schedules.{name}.updates")
            gated_feature = {"oracle": "oracle", "belief": "belief", "cooperation": "cooperation", "bc": "human_bc", "strategy": "strategy", "bidding": "bidding"}.get(name)
            if gated_feature and (weight > 0.0) != features[gated_feature]:
                raise FormalConfigError(f"loss {name} must exactly match feature gate {gated_feature}")
        for role, weight in role_weights.items():
            if _number(weight, f"losses.role_weights.{role}") <= 0.0:
                raise FormalConfigError("role weights must be positive")

        feature_configs = _mapping(data["feature_configs"], "feature_configs")
        _exact(feature_configs, {"adaptive_dmc", "oracle", "belief", "cooperation", "human_bc", "strategy", "style", "bidding", "search"}, "feature_configs")
        for name, config in feature_configs.items():
            config_map = _mapping(config, f"feature_configs.{name}")
            if config_map.get("enabled") is not features[name if name != "adaptive_dmc" else "adaptive_dmc"]:
                raise FormalConfigError(f"feature_configs.{name}.enabled disagrees with feature gate")
        if feature_configs["human_bc"].get("dataset_identity") is not None:
            raise FormalConfigError("P1 has no authorized BC dataset identity")
        if feature_configs["search"].get("training_semantics") != "deployment_wrapper_shared_checkpoint_v1":
            raise FormalConfigError("search training semantics mismatch")

        budgets_data = _mapping(data["budgets"], "budgets")
        _exact(budgets_data, set(_BUDGET_NAMES), "budgets")
        budgets = {name: FormalBudget.from_dict(budgets_data[name], f"budgets.{name}") for name in _BUDGET_NAMES}
        if budgets["pilot"].training_seed_count != 1 or budgets["development"].training_seed_count < 3 or budgets["promotion"].training_seed_count < 3:
            raise FormalConfigError("pilot/development/promotion seed counts are invalid")
        if budgets["development"].paired_deals < 20_000 or budgets["promotion"].paired_deals < 100_000 or budgets["promotion"].wall_clock_seconds < 7_200:
            raise FormalConfigError("formal evaluation budgets are below frozen minimums")

        evaluation = _mapping(data["evaluation"], "evaluation")
        _exact(evaluation, {"bootstrap_unit", "confidence", "deal_set_strategy"}, "evaluation")
        if evaluation["bootstrap_unit"] != "deal" or _number(evaluation["confidence"], "evaluation.confidence") != 0.95:
            raise FormalConfigError("formal evaluation requires deal bootstrap at 95% confidence")

        return cls(
            metadata=dict(metadata), variant=variant, ruleset=dict(ruleset_data),
            model=dict(model), features=dict(features), losses=dict(losses),
            feature_configs=dict(feature_configs),
            runtime=FormalRuntime.from_dict(data["runtime"]),
            seeds=FormalSeeds.from_dict(data["seeds"]), budgets=budgets,
            initialization=FormalInitialization.from_dict(data["initialization"]),
            evaluation=dict(evaluation),
        )

    def resolved_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = V3_FORMAL_CONFIG_SCHEMA
        return value

    def identity_dict(self) -> dict[str, Any]:
        resolved = self.resolved_dict()
        semantic = {key: value for key, value in resolved.items() if key not in {"metadata", "runtime", "seeds", "budgets", "evaluation"}}
        workload = {key: resolved[key] for key in ("runtime", "seeds", "budgets", "evaluation")}
        deal_sets = {
            phase: canonical_hash({
                "strategy": self.evaluation["deal_set_strategy"],
                "ruleset_hash": self.ruleset["hash"],
                "deal_seed": self.seeds.deal_set,
                "paired_deals": budget.paired_deals,
            })
            for phase, budget in self.budgets.items()
        }
        return {
            "schema_version": V3_FORMAL_IDENTITY_SCHEMA,
            "config_sha256": canonical_hash(resolved),
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "training_semantics_hash": canonical_hash(semantic),
            "workload_identity_version": WORKLOAD_IDENTITY_VERSION,
            "workload_hash": canonical_hash(workload),
            "ruleset_hash": self.ruleset["hash"],
            "model_hash": (
                V3HybridModelConfig.from_dict(dict(self.model["config"])).stable_hash()
                if self.model["family"] == "v3_hybrid"
                else canonical_hash(self.model)
            ),
            "support_matrix_version": H8A_SUPPORT_MATRIX_VERSION,
            "support_matrix_hash": h8a_support_matrix_hash(),
            "initial_checkpoint_hash": self.initialization.checkpoint_sha256,
            "seeds": asdict(self.seeds),
            "budgets": {name: asdict(budget) for name, budget in self.budgets.items()},
            "deal_set_hashes": deal_sets,
            "release_candidate": "NONE",
            "release_status": "NOT READY",
            "playing_strength": "NOT MEASURED",
        }


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and set(value) == {"__replace__"}:
            result[key] = value["__replace__"]
        elif (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise FormalConfigError(f"cannot read formal config {path}: {exc}") from exc
    return _mapping(value, str(path))


def load_formal_config(path: str | Path) -> FormalExperimentConfig:
    config_path = Path(path).resolve()
    overlay = _load_yaml(config_path)
    parent = overlay.get("extends")
    if parent is not None:
        if not isinstance(parent, str) or Path(parent).name != parent:
            raise FormalConfigError("extends must name a sibling config fragment")
        parent_path = config_path.parent / parent
        if parent_path == config_path or not parent_path.is_file():
            raise FormalConfigError("extends target is missing or recursive")
        base = _load_yaml(parent_path)
        if "extends" in base:
            raise FormalConfigError("nested formal config inheritance is forbidden")
        payload = _deep_merge(base, overlay)
    else:
        payload = dict(overlay)
    return FormalExperimentConfig.from_dict(payload)


def freeze_formal_config(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = load_formal_config(path)
    validate_initial_checkpoint(config, config_path=Path(path).resolve())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = _canonical_bytes(config.resolved_dict()) + b"\n"
    identity = _canonical_bytes(config.identity_dict()) + b"\n"
    (output / "resolved_config.json").write_bytes(resolved)
    (output / "identity.json").write_bytes(identity)
    return config.identity_dict()


def validate_initial_checkpoint(
    config: FormalExperimentConfig, *, config_path: Path | None = None
) -> None:
    """Validate a configured initial checkpoint without loading any tensors.

    A sidecar is mandatory and exact. Tensor loading remains the owning model
    loader's job; this preflight prevents cross-family/ruleset selection and
    runs before any checkpoint or CUDA initialization in the training command.
    """

    initialization = config.initialization
    if initialization.kind == "seeded_fresh":
        return
    checkpoint = Path(initialization.checkpoint_path or "")
    if not checkpoint.is_absolute() and config_path is not None:
        checkpoint = config_path.parent / checkpoint
    try:
        content = checkpoint.read_bytes()
    except OSError as exc:
        raise FormalConfigError(f"initial checkpoint cannot be read: {checkpoint}") from exc
    if hashlib.sha256(content).hexdigest() != initialization.checkpoint_sha256:
        raise FormalConfigError("initial checkpoint content hash mismatch")
    manifest_path = checkpoint.with_name(checkpoint.name + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalConfigError("initial checkpoint manifest is missing or invalid") from exc
    manifest = _mapping(manifest, "initial checkpoint manifest")
    _exact(
        manifest,
        {"schema_version", "checkpoint_kind", "model_family", "model_hash", "ruleset_hash"},
        "initial checkpoint manifest",
    )
    expected = {
        "schema_version": V3_FORMAL_INITIAL_CHECKPOINT_SCHEMA,
        "checkpoint_kind": initialization.checkpoint_kind,
        "model_family": config.model["family"],
        "model_hash": config.identity_dict()["model_hash"],
        "ruleset_hash": config.ruleset["hash"],
    }
    if manifest != expected:
        raise FormalConfigError("initial checkpoint identity is incompatible")


__all__ = [
    "FormalBudget", "FormalConfigError", "FormalExperimentConfig",
    "FormalInitialization", "FormalRuntime", "FormalSeeds",
    "V3_FORMAL_CONFIG_SCHEMA", "V3_FORMAL_IDENTITY_SCHEMA", "canonical_hash",
    "V3_FORMAL_INITIAL_CHECKPOINT_SCHEMA", "freeze_formal_config",
    "load_formal_config", "validate_initial_checkpoint",
]
