"""Explicit tiny H7 CUDA test identity; never a playing-strength config."""

from __future__ import annotations

from .adaptive_dmc import ADMC_SAFE_HYBRID, AdaptiveDMCConfig
from .config import V3HybridModelConfig
from .h2_learner import V3H2LearnerConfig
from .integration_config import (
    V3H6FeatureFlags,
    V3H6LearnerConfig,
    V3H6ResolvedConfig,
    V3H6TopologyConfig,
)
from .loss_composer import V3HybridLossComposerConfig
from .training.h3_learner import V3H3LearnerConfig
from .training.h4_learner import V3H4LearnerConfig
from .training.h5_learner import V3H5LearnerConfig


def build_v3_h7_smoke_config() -> V3H6ResolvedConfig:
    model = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
    )
    public = V3H2LearnerConfig(
        batch_size=32,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        device="cuda",
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID),
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    learner = V3H6LearnerConfig(
        base=base,
        losses=V3HybridLossComposerConfig(lambda_dmc=1.0),
        features=V3H6FeatureFlags(adaptive_dmc=True),
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    return V3H6ResolvedConfig(model=model, learner=learner)


__all__ = ["build_v3_h7_smoke_config"]
