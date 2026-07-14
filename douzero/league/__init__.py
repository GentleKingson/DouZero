"""Versioned policy league and leakage-safe population self-play (P11)."""

from .manifest import LeagueManifest, PolicyEntry
from .policy_pool import (
    PolicyBundle,
    PolicyPool,
    PolicyPoolConfig,
    build_frozen_policy_model,
)
from .promotion import PromotionDecision, PromotionEvaluation, PromotionGate
from .self_play import MatchupLogger, MatchupRecord, PopulationEpisodeRunner
from .snapshot import SnapshotManager, SnapshotRetention

__all__ = [
    "LeagueManifest",
    "MatchupLogger",
    "MatchupRecord",
    "PolicyBundle",
    "PolicyEntry",
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
