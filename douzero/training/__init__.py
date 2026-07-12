"""Multi-objective training, decision policy, and calibration (P06).

This package implements the P06 layer:

- :mod:`douzero.training.labels` — team-perspective terminal labels derived
  from :class:`~douzero.env.scoring.GameResult`. These replace the legacy
  single-scalar ``target`` and centralize the sign convention so farmer-team
  wins are positive for both farmer roles without scattered negations.
- :mod:`douzero.training.losses` — multi-objective loss module: a stable
  BCE-with-logits win-probability loss, masked Huber conditional-score
  losses (supervised only on the applicable terminal outcome), an optional
  log-score auxiliary loss, and an optional uncertainty-NLL (default off).
- :mod:`douzero.training.decision_policy` — configurable action selection
  over a :class:`~douzero.models_v2.output.ModelOutput`: pure win, pure
  expected score, lexicographic win-then-score / score-then-win, and a
  risk-aware mode (default off).
- :mod:`douzero.training.calibration` — win-probability calibration metrics
  (Brier score, NLL, expected calibration error, reliability bins).
- :mod:`douzero.training.v2_buffer` — a per-episode V2 transition store
  friendly to variable legal-action counts.
- :mod:`douzero.training.v2_trainer` — a minimal single-process V2 self-play
  trainer that runs short CPU rollouts, computes the multi-objective loss,
  and applies at least one optimizer step.

The legacy training path (:mod:`douzero.dmc`) is unchanged; P06 ships a new
:mod:`douzero.training.v2_trainer` and a ``train_v2.py`` CLI entry point so
the legacy multiprocessing loop is not destabilised.
"""

from douzero.training.labels import (
    FARMER_POSITIONS,
    LANDLORD_POSITIONS,
    LogScoreTransform,
    team_target_log_score,
    team_target_score,
    team_target_win,
    team_targets,
)
from douzero.training.losses import (
    LossConfig,
    LossComponents,
    MultiObjectiveLoss,
    bce_win_loss,
    conditional_score_huber_loss,
    log_score_aux_loss,
)
from douzero.training.decision_policy import (
    DecisionConfig,
    select_action,
    SUPPORTED_DECISION_MODES,
)
from douzero.training.calibration import (
    brier_score,
    expected_calibration_error,
    nll,
    reliability_bins,
)
from douzero.training.v2_buffer import (
    Episode,
    Minibatch,
    Transition,
    V2ReplayBuffer,
)
from douzero.training.v2_trainer import (
    TrainerConfig,
    TrainerStats,
    V2Trainer,
)

__all__ = [
    "FARMER_POSITIONS",
    "LANDLORD_POSITIONS",
    "LogScoreTransform",
    "team_target_log_score",
    "team_target_score",
    "team_target_win",
    "team_targets",
    "LossConfig",
    "LossComponents",
    "MultiObjectiveLoss",
    "bce_win_loss",
    "conditional_score_huber_loss",
    "log_score_aux_loss",
    "DecisionConfig",
    "select_action",
    "SUPPORTED_DECISION_MODES",
    "brier_score",
    "expected_calibration_error",
    "nll",
    "reliability_bins",
    "Episode",
    "Minibatch",
    "Transition",
    "V2ReplayBuffer",
    "TrainerConfig",
    "TrainerStats",
    "V2Trainer",
]
