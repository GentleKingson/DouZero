"""Versioned policy league and leakage-safe population self-play (P11)."""

from .manifest import LeagueManifest, PendingDelete, PolicyEntry
from .policy_pool import (
    LoadedPolicySelector,
    PolicyBundle,
    PolicyLoaderContract,
    PolicyPool,
    PolicyPoolConfig,
    build_frozen_policy_model,
)
from .promotion import PromotionDecision, PromotionEvaluation, PromotionGate
from .self_play import MatchupLogger, MatchupRecord, PopulationEpisodeRunner
from .snapshot import SnapshotManager, SnapshotRetention

__all__ = [
    "LeagueManifest",
    "LoadedPolicySelector",
    "MatchupLogger",
    "MatchupRecord",
    "PendingDelete",
    "PolicyBundle",
    "PolicyEntry",
    "PolicyLoaderContract",
    "PolicyPool",
    "PolicyPoolConfig",
    "PopulationEpisodeRunner",
    "PromotionDecision",
    "PromotionEvaluation",
    "PromotionGate",
    "SnapshotManager",
    "SnapshotRetention",
    "build_frozen_policy_model",
]
