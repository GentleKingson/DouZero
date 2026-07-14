"""Coach-guided opening curriculum for training only (P12)."""

from .curriculum import (
    BALANCED,
    GUIDED_MODES,
    HARD_FOR_ROLE,
    MIXTURE,
    SAMPLING_MODES,
    TRUE_RANDOM,
    CurriculumAuditLogger,
    CurriculumSchedule,
    OpeningSampler,
    SamplingRecord,
)
from .labels import CoachLabel, CoachLabelStore, calibration_metrics
from .model import (
    COACH_FEATURE_VERSION,
    COACH_INPUT_SIZE,
    COACH_MODEL_VERSION,
    CoachModel,
    CoachModelConfig,
    encode_opening,
    load_coach_checkpoint,
    save_coach_checkpoint,
    train_coach,
)
from .records import CANONICAL_DECK, CARD_RANKS, OpeningRecord, random_opening

__all__ = [
    "BALANCED",
    "CANONICAL_DECK",
    "CARD_RANKS",
    "COACH_FEATURE_VERSION",
    "COACH_INPUT_SIZE",
    "COACH_MODEL_VERSION",
    "CoachLabel",
    "CoachLabelStore",
    "CoachModel",
    "CoachModelConfig",
    "CurriculumAuditLogger",
    "CurriculumSchedule",
    "GUIDED_MODES",
    "HARD_FOR_ROLE",
    "MIXTURE",
    "OpeningRecord",
    "OpeningSampler",
    "SAMPLING_MODES",
    "SamplingRecord",
    "TRUE_RANDOM",
    "calibration_metrics",
    "encode_opening",
    "load_coach_checkpoint",
    "random_opening",
    "save_coach_checkpoint",
    "train_coach",
]
