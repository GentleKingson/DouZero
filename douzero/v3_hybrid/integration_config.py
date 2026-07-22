"""Typed H6 integration config and pre-side-effect validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from .adaptive_dmc import ADMC_DISABLED
from .config import V3HybridModelConfig
from .loss_composer import V3HybridLossComposerConfig
from .support_matrix import (
    RULESET_STANDARD,
    TOPOLOGY_SINGLE_PROCESS,
    v3_h6_support_matrix_hash,
    validate_capability_support,
)
from .training.h5_learner import V3H5LearnerConfig

V3_H6_CONFIG_FORMAT = "v3-hybrid-h6-config-v1"


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _default_loss_config() -> V3HybridLossComposerConfig:
    return V3HybridLossComposerConfig(lambda_dmc=1.0)


@dataclass(frozen=True)
class V3H6FeatureFlags:
    role_model: bool = True
    adaptive_dmc: bool = False
    oracle: bool = False
    belief: bool = False
    cooperation: bool = False
    human_bc: bool = False
    strategy: bool = False
    style: bool = False
    league: bool = False
    curriculum: bool = False
    bidding: bool = False
    selective_search: bool = False
    public_export: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise TypeError(f"V3 H6 feature flag {name} must be bool")
        if not self.role_model:
            raise ValueError("V3 H6 requires the canonical role model")

    def enabled_capabilities(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value)


@dataclass(frozen=True)
class V3H6TopologyConfig:
    topology: str = TOPOLOGY_SINGLE_PROCESS
    ruleset: str = RULESET_STANDARD
    checkpoint_resume: bool = True
    export: bool = True
    deployment: bool = True
    search: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint_resume", "export", "deployment", "search"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"V3 H6 topology field {name} must be bool")


@dataclass(frozen=True)
class V3H6AuxiliaryConfig:
    score_delta: float = 1.0
    score_target_transform: str = "raw"
    bc_temperature: float = 1.0
    bc_label_smoothing: float = 0.0
    strategy_lambda_min_turns: float = 1.0
    strategy_lambda_regain_initiative: float = 1.0
    strategy_lambda_teammate_finish: float = 1.0
    strategy_lambda_spring: float = 1.0
    strategy_lambda_structure: float = 1.0
    bidding_lambda_policy: float = 1.0
    bidding_lambda_landlord_win: float = 1.0
    bidding_lambda_landlord_score: float = 0.5
    bidding_lambda_regret: float = 0.0

    def __post_init__(self) -> None:
        if self.score_target_transform not in {"raw", "signed_log"}:
            raise ValueError("H6 score_target_transform must be raw or signed_log")
        for name, value in asdict(self).items():
            if name == "score_target_transform":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"H6 auxiliary field {name} must be finite and non-negative")
        if self.score_delta <= 0.0 or self.bc_temperature <= 0.0:
            raise ValueError("H6 score delta and BC temperature must be positive")
        if not 0.0 <= self.bc_label_smoothing < 1.0:
            raise ValueError("H6 BC label smoothing must be in [0, 1)")

    @property
    def strategy_component_weight(self) -> float:
        return math.fsum((
            self.strategy_lambda_min_turns,
            self.strategy_lambda_regain_initiative,
            self.strategy_lambda_teammate_finish,
            self.strategy_lambda_spring,
            self.strategy_lambda_structure,
        ))

    @property
    def bidding_component_weight(self) -> float:
        return math.fsum((
            self.bidding_lambda_policy,
            self.bidding_lambda_landlord_win,
            self.bidding_lambda_landlord_score,
            self.bidding_lambda_regret,
        ))


@dataclass(frozen=True)
class V3H6LearnerConfig:
    base: V3H5LearnerConfig = field(default_factory=V3H5LearnerConfig)
    losses: V3HybridLossComposerConfig = field(
        default_factory=_default_loss_config
    )
    auxiliary: V3H6AuxiliaryConfig = field(default_factory=V3H6AuxiliaryConfig)
    features: V3H6FeatureFlags = field(default_factory=V3H6FeatureFlags)
    topology: V3H6TopologyConfig = field(default_factory=V3H6TopologyConfig)

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.base, V3H5LearnerConfig):
            raise TypeError("H6 base must be V3H5LearnerConfig")
        if not isinstance(self.losses, V3HybridLossComposerConfig):
            raise TypeError("H6 losses must be V3HybridLossComposerConfig")
        if not isinstance(self.auxiliary, V3H6AuxiliaryConfig):
            raise TypeError("H6 auxiliary config has an invalid type")
        if not isinstance(self.features, V3H6FeatureFlags):
            raise TypeError("H6 feature flags have an invalid type")
        if not isinstance(self.topology, V3H6TopologyConfig):
            raise TypeError("H6 topology config has an invalid type")
        public = self.base.base.base.public
        expected = {
            "adaptive_dmc": public.adaptive_dmc.mode != ADMC_DISABLED,
            "oracle": self.base.base.base.schedule.enabled,
            "belief": self.base.base.belief.enabled,
            "cooperation": self.base.cooperation.enabled,
        }
        for name, value in expected.items():
            if getattr(self.features, name) != value:
                raise ValueError(
                    f"H6 feature flag {name} disagrees with its owning config"
                )
        exact_weights = {
            "dmc": public.lambda_dmc,
            "belief": self.base.base.belief.lambda_belief,
            "cooperation": self.base.cooperation.lambda_coop,
        }
        for name, expected_weight in exact_weights.items():
            if self.losses.weight(name) != expected_weight:
                raise ValueError(
                    f"H6 lambda_{name} disagrees with its owning H{ {'dmc': 2, 'belief': 4, 'cooperation': 5}[name] } config"
                )
        oracle_weight = self.losses.weight("oracle")
        if self.features.oracle != (oracle_weight > 0.0):
            raise ValueError("H6 lambda_oracle must gate the Oracle feature")
        if self.features.oracle and oracle_weight != 1.0:
            raise ValueError("enabled H6 Oracle uses lambda_oracle=1; H3 owns annealing")
        if self.losses.role_weights != public.role_weights:
            raise ValueError("H6 role weights must exactly match the public learner")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "base": self.base.compatibility_dict(),
            "losses": self.losses.compatibility_dict(),
            "auxiliary": asdict(self.auxiliary),
            "features": asdict(self.features),
            "topology": asdict(self.topology),
            "support_matrix_hash": v3_h6_support_matrix_hash(),
            "optimizer_phase": (
                "nested_h3_h4_h5_then_h6_public_aux_atomic_rollback_v1"
            ),
            "replay_schema": "h2_public_plus_separate_privileged_sidecars_v1",
            "checkpoint_format": "v3-hybrid-h6-trainer-v1",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H6LearnerConfig":
        if not isinstance(payload, Mapping) or set(payload) != {
            "base", "losses", "auxiliary", "features", "topology"
        }:
            raise ValueError("H6 learner config fields mismatch")
        auxiliary = payload["auxiliary"]
        features = payload["features"]
        topology = payload["topology"]
        if not isinstance(auxiliary, Mapping) or set(auxiliary) != set(
            V3H6AuxiliaryConfig.__dataclass_fields__
        ):
            raise ValueError("H6 auxiliary config fields mismatch")
        if not isinstance(features, Mapping) or set(features) != set(
            V3H6FeatureFlags.__dataclass_fields__
        ):
            raise ValueError("H6 feature config fields mismatch")
        if not isinstance(topology, Mapping) or set(topology) != set(
            V3H6TopologyConfig.__dataclass_fields__
        ):
            raise ValueError("H6 topology config fields mismatch")
        return cls(
            base=V3H5LearnerConfig.from_dict(payload["base"]),
            losses=V3HybridLossComposerConfig.from_dict(payload["losses"]),
            auxiliary=V3H6AuxiliaryConfig(**dict(auxiliary)),
            features=V3H6FeatureFlags(**dict(features)),
            topology=V3H6TopologyConfig(**dict(topology)),
        )


@dataclass(frozen=True)
class V3H6ResolvedConfig:
    model: V3HybridModelConfig
    learner: V3H6LearnerConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model, V3HybridModelConfig):
            raise TypeError("H6 resolved model config has an invalid type")
        if not isinstance(self.learner, V3H6LearnerConfig):
            raise TypeError("H6 resolved learner config has an invalid type")
        features = self.learner.features
        losses = self.learner.losses
        graph_checks = {
            "human_bc": self.model.human_prior_enabled and losses.lambda_bc > 0.0,
            "strategy": self.model.strategy_aux_enabled and losses.lambda_strategy > 0.0,
            "style": self.model.style_enabled,
            "bidding": self.model.bidding_enabled and losses.lambda_bidding > 0.0,
        }
        for name, enabled in graph_checks.items():
            if getattr(features, name) != enabled:
                raise ValueError(
                    f"H6 feature flag {name} disagrees with model/loss graph"
                )
        if self.model.strategy_aux_enabled and not self.model.strategy_features_enabled:
            raise ValueError("H6 strategy auxiliary requires public strategy features")
        if losses.lambda_strategy > 0.0 and self.learner.auxiliary.strategy_component_weight <= 0.0:
            raise ValueError("H6 strategy loss has no active component")
        if losses.lambda_bidding > 0.0 and self.learner.auxiliary.bidding_component_weight <= 0.0:
            raise ValueError("H6 bidding loss has no active component")
        if self.model.score_clamp <= 0.0:
            raise ValueError("H6 score clamp must be positive")
        self.validate_startup()

    def validate_startup(self) -> None:
        """Run before model/CUDA/checkpoint/replay/worker construction."""

        topology = self.learner.topology
        if topology.search != self.learner.features.selective_search:
            raise ValueError("H6 search operation and feature flag disagree")
        if topology.export and not self.learner.features.public_export:
            raise ValueError("H6 export operation requires public_export capability")
        for capability in self.learner.features.enabled_capabilities():
            validate_capability_support(
                capability,
                topology=topology.topology,
                ruleset=topology.ruleset,
                checkpoint_resume=topology.checkpoint_resume,
                export=topology.export,
                deployment=topology.deployment,
                search=topology.search,
            )

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "format": V3_H6_CONFIG_FORMAT,
            "model": self.model.compatibility_dict(),
            "learner": self.learner.compatibility_dict(),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())


def load_v3_hybrid_config(path: str | Path) -> V3H6ResolvedConfig:
    """Load only the dedicated H6 YAML; Legacy/V2 loaders remain untouched."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"format", "model", "learner"}:
        raise ValueError("V3 H6 YAML root fields mismatch")
    if payload["format"] != V3_H6_CONFIG_FORMAT:
        raise ValueError("V3 H6 YAML format mismatch")
    return V3H6ResolvedConfig(
        model=V3HybridModelConfig.from_dict(dict(payload["model"])),
        learner=V3H6LearnerConfig.from_dict(payload["learner"]),
    )


__all__ = [
    "V3_H6_CONFIG_FORMAT",
    "V3H6AuxiliaryConfig",
    "V3H6FeatureFlags",
    "V3H6LearnerConfig",
    "V3H6ResolvedConfig",
    "V3H6TopologyConfig",
    "load_v3_hybrid_config",
]
