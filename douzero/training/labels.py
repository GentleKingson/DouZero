"""Team-perspective multi-objective terminal labels (P06).

This module centralizes the sign convention mandated by AGENTS.md
"Rewards, targets, and action selection":

    ``target_win`` is from the current acting player's team perspective.
    ``target_score`` is the final signed score from that same perspective.
    A farmer win is positive for both farmer roles.

The legacy actor/learner negated ``episode_return`` for farmer positions in a
scattered way (``douzero/dmc/utils.py`` negated the env's landlord-perspective
reward for farmers). This module replaces that convention with one pure
function family so the loss module, the buffer, and evaluation all derive
labels from the same source of truth: the structured
:class:`~douzero.env.scoring.GameResult`.

Sign convention
---------------
- ``landlord``: ``team_score = game_result.landlord_score`` (positive when
  the landlord team wins, negative when it loses; wins count double).
- ``landlord_up`` / ``landlord_down``: ``team_score = game_result.farmer_score``
  (positive when the farmer team wins, negative when it loses).
- Both farmers share team utility: the same ``GameResult`` yields the same
  ``team_score`` for ``landlord_up`` and ``landlord_down``. The score
  conservation invariant ``landlord_score + 2 * farmer_score == 0`` is
  respected (see ``douzero/env/scoring.py``).

These helpers NEVER read hidden hands. They consume only the terminal
:class:`GameResult`, which is fully public once the game is over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from douzero.env.scoring import FARMER_POSITIONS, LANDLORD_POSITIONS

#: All three positions, in canonical order.
ALL_POSITIONS: tuple[str, ...] = ("landlord", "landlord_up", "landlord_down")

#: Sentinel returned by ``_coerce_int`` for missing values (defensive; should
#: not occur for a well-formed GameResult dict).
_MISSING: int = 0


def _coerce_int(value: Any, default: int = _MISSING) -> int:
    """Coerce a JSON/dict value to int without silent float rounding."""
    if value is None:
        return default
    if isinstance(value, bool):
        # Booleans are ints in Python; reject so a ``spring: true`` field does
        # not silently become 1.
        raise TypeError(f"expected int, got bool: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"expected int, got non-integer float: {value!r}")
        return int(value)
    raise TypeError(f"expected int, got {type(value).__name__}: {value!r}")


def _winner_team(result: Any) -> str:
    """Read ``winner_team`` from a GameResult-like object or dict."""
    if isinstance(result, Mapping):
        team = result.get("winner_team")
    else:
        team = getattr(result, "winner_team", None)
    if team not in ("landlord", "farmer"):
        raise ValueError(
            f"GameResult.winner_team must be 'landlord' or 'farmer', got {team!r}"
        )
    return team


def _team_score_for(result: Any, position: str) -> int:
    """Read the signed team score for ``position`` from a GameResult-like object."""
    if position in LANDLORD_POSITIONS:
        key = "landlord_score"
    elif position in FARMER_POSITIONS:
        key = "farmer_score"
    else:
        raise ValueError(
            f"unknown position {position!r}; expected one of {ALL_POSITIONS}"
        )
    if isinstance(result, Mapping):
        return _coerce_int(result.get(key, 0))
    return _coerce_int(getattr(result, key, 0))


def team_target_win(result: Any, position: str) -> int:
    """Return ``target_win`` in {0, 1} from ``position``'s team perspective.

    The landlord team wins iff ``winner_team == "landlord"``; otherwise the
    farmer team wins (and both farmer roles share the win). This is the
    canonical team-perspective win label used by the BCE win head.
    """
    if position not in ALL_POSITIONS:
        raise ValueError(
            f"unknown position {position!r}; expected one of {ALL_POSITIONS}"
        )
    team = _winner_team(result)
    if position in LANDLORD_POSITIONS:
        return 1 if team == "landlord" else 0
    return 1 if team == "farmer" else 0


def team_target_score(result: Any, position: str) -> float:
    """Return the signed final team score from ``position``'s team perspective.

    Positive = the acting team won; negative = the acting team lost. For the
    landlord this is ``GameResult.landlord_score`` (which counts wins/losses
    double); for either farmer this is ``GameResult.farmer_score`` (per
    farmer; both farmers share the sign and magnitude by construction).
    """
    return float(_team_score_for(result, position))


def team_target_log_score(
    result: Any,
    position: str,
    *,
    floor: float = 1.0,
) -> float:
    """Return a numerically stable log-score auxiliary target.

    ``sign(score) * log1p(abs(score))`` compresses the long bomb/rocket tail
    while preserving sign and zero. The optional ``floor`` (default 1.0) is
    applied to the magnitude before the log, so a 0-magnitude game (e.g. a
    legacy non-multiplier game recorded with score 0) maps to 0 rather than
    ``log1p(0) == 0`` (a no-op) and a small-magnitude game still produces a
    non-zero magnitude. With the default ``floor=1.0``:

        score =  0   -> log_score =  0.0  (sign(0) = 0)
        score = +1   -> log_score = +log1p(1)  = +0.6931...
        score = -1   -> log_score = -0.6931...
        score = +8   -> log_score = +log1p(8)  = +2.1972...
        score = -32  -> log_score = -log1p(32) = -3.4965...

    This is a stable auxiliary target (P06 task item 1); it is NOT a
    replacement for the conditional score heads.
    """
    score = team_target_score(result, position)
    if score == 0:
        return 0.0
    magnitude = max(abs(score), float(floor))
    return math.copysign(math.log1p(magnitude), score)


def team_targets(result: Any, position: str, *, log_floor: float = 1.0) -> dict[str, float]:
    """Return the full multi-objective label dict for ``position``.

    Keys: ``target_win`` (int 0/1), ``target_score`` (float),
    ``target_log_score`` (float). This is the canonical per-transition label
    stored by the V2 transition buffer; the loss module reads these keys by
    name so the storage layout and the loss layout stay synchronized.
    """
    return {
        "target_win": float(team_target_win(result, position)),
        "target_score": team_target_score(result, position),
        "target_log_score": team_target_log_score(result, position, floor=log_floor),
    }


@dataclass(frozen=True)
class LogScoreTransform:
    """Stateful adapter exposing the log-score transform as a callable.

    Provided so a trainer or test can pass a configured transform object
    (e.g. with a custom ``floor``) without juggling positional arguments.
    """

    floor: float = 1.0

    def __call__(self, score: float) -> float:
        if score == 0:
            return 0.0
        magnitude = max(abs(score), float(self.floor))
        return math.copysign(math.log1p(magnitude), score)
