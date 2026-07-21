"""Opt-in DouZero V3 Hybrid contracts and H1 public card-play model."""

from .checkpoint import (
    V3_HYBRID_H1_CHECKPOINT_FORMAT,
    h1_compatibility_identity,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
)
from .config import (
    CHANNEL_GATE_NONE,
    CHANNEL_GATE_SE,
    DMC_TARGET_RAW,
    DMC_TARGET_SIGNED_LOG,
    HISTORY_ENCODER_LSTM,
    HISTORY_ENCODER_TRANSFORMER,
    V3HybridModelConfig,
)

from .contract import (
    V3_HYBRID_CHECKPOINT_KIND,
    V3_HYBRID_CONTRACT_VERSION,
    V3_HYBRID_FEATURE_VERSION,
    V3_HYBRID_LOSS_TERMS,
    V3_HYBRID_MODEL_VERSION,
    V3_HYBRID_OBSERVATION_SCHEMA_HASH,
    V3_HYBRID_OBSERVATION_SCHEMA_VERSION,
    V3_HYBRID_PHASES,
    V3HybridCompatibilityIdentity,
    assert_v3_hybrid_compatible,
    v3_hybrid_semantic_contract,
)
from .export import ExportableV3HybridModel, export_v3_hybrid_padded
from .model import V3_HYBRID_ROLES, V3HybridModel
from .output import BatchedV3HybridModelOutput, V3HybridModelOutput

__all__ = [
    "V3_HYBRID_CHECKPOINT_KIND",
    "V3_HYBRID_CONTRACT_VERSION",
    "V3_HYBRID_FEATURE_VERSION",
    "V3_HYBRID_LOSS_TERMS",
    "V3_HYBRID_MODEL_VERSION",
    "V3_HYBRID_OBSERVATION_SCHEMA_HASH",
    "V3_HYBRID_OBSERVATION_SCHEMA_VERSION",
    "V3_HYBRID_PHASES",
    "V3HybridCompatibilityIdentity",
    "assert_v3_hybrid_compatible",
    "v3_hybrid_semantic_contract",
    "CHANNEL_GATE_NONE",
    "CHANNEL_GATE_SE",
    "DMC_TARGET_RAW",
    "DMC_TARGET_SIGNED_LOG",
    "HISTORY_ENCODER_LSTM",
    "HISTORY_ENCODER_TRANSFORMER",
    "V3HybridModelConfig",
    "V3_HYBRID_ROLES",
    "V3HybridModel",
    "V3HybridModelOutput",
    "BatchedV3HybridModelOutput",
    "V3_HYBRID_H1_CHECKPOINT_FORMAT",
    "h1_compatibility_identity",
    "save_v3_hybrid_public_checkpoint",
    "load_v3_hybrid_public_checkpoint",
    "ExportableV3HybridModel",
    "export_v3_hybrid_padded",
]
