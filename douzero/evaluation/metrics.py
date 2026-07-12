"""Per-role evaluation metrics and calibration aggregation (P06).

The legacy :func:`douzero.evaluation.simulation.evaluate` aggregates team
win-count and total score across all games. P06 adds:

- :class:`RoleMetrics` — per-role (landlord / landlord_up /
  landlord_down) win percentage, mean score, and game count. The legacy
  team-level numbers are derivable from these by aggregation.
- :class:`CalibrationAggregator` — running Brier / NLL / ECE over the
  ``p_win`` predictions a V2 model makes at decision time, against the
  eventual terminal outcome. This is the bridge from the per-decision
  :mod:`douzero.training.calibration` metrics to an evaluation cohort.
- :func:`summarize` — format the metrics as a JSON-serializable dict and
  a human-readable multi-line string for the evaluate.py CLI.

These helpers do NOT change the existing multiprocessing evaluation path.
They are consumed by the V2-aware evaluation entry point
:func:`douzero.evaluation.simulation.evaluate_v2` (P06) and by P15's
unified evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from douzero.training.calibration import (
    brier_score,
    expected_calibration_error,
    nll,
)


ROLES: tuple[str, ...] = ("landlord", "landlord_up", "landlord_down")


@dataclass
class RoleMetrics:
    """Per-role win-percentage and mean-score aggregates."""

    wins: int = 0
    losses: int = 0
    score_sum: float = 0.0

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def win_percentage(self) -> float:
        total = self.games
        return float(self.wins) / total if total else float("nan")

    @property
    def mean_score(self) -> float:
        total = self.games
        return float(self.score_sum) / total if total else float("nan")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "games": self.games,
            "win_percentage": self.win_percentage,
            "mean_score": self.mean_score,
        }


@dataclass
class CalibrationAggregator:
    """Running win-probability calibration over a stream of decisions.

    Accumulates ``(p_win, target_win)`` pairs per role as flat Python lists
    and computes :func:`brier_score`, :func:`nll`, and
    :func:`expected_calibration_error` on demand. Cheap to update inside an
    evaluation worker; the metrics are computed once at the end.

    Samples are tagged by role so per-role calibration can be inspected
    (a landlord-trained model may be better calibrated on landlord
    decisions than on farmer decisions).
    """

    p_win_by_role: dict[str, list[float]] = field(default_factory=dict)
    target_by_role: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for role in ROLES:
            self.p_win_by_role.setdefault(role, [])
            self.target_by_role.setdefault(role, [])

    def add(self, role: str, p_win: float, target_win: float) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        self.p_win_by_role[role].append(float(p_win))
        self.target_by_role[role].append(float(target_win))

    def metrics(self, role: str, n_bins: int = 15) -> dict[str, float]:
        """Return calibration metrics for ``role`` (empty -> NaN)."""
        ps = self.p_win_by_role.get(role, [])
        ts = self.target_by_role.get(role, [])
        if not ps:
            return {"count": 0.0, "brier": float("nan"), "nll": float("nan"), "ece": float("nan")}
        import torch

        p_tensor = torch.tensor(ps, dtype=torch.float32)
        t_tensor = torch.tensor(ts, dtype=torch.float32)
        return {
            "count": float(len(ps)),
            "brier": brier_score(p_tensor, t_tensor),
            "nll": nll(p_tensor, t_tensor),
            "ece": expected_calibration_error(p_tensor, t_tensor, n_bins=n_bins),
        }


@dataclass
class EvaluationReport:
    """Top-level evaluation report (per-role metrics + calibration)."""

    role_metrics: dict[str, RoleMetrics] = field(default_factory=dict)
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict]:
        return {
            "roles": {r: m.to_dict() for r, m in self.role_metrics.items()},
            "calibration": dict(self.calibration),
        }


def summarize(report: EvaluationReport, *, title: str = "P06 evaluation") -> str:
    """Return a human-readable multi-line summary of an EvaluationReport."""
    lines = [f"{title}"]
    lines.append("Per-role results:")
    for role in ROLES:
        m = report.role_metrics.get(role)
        if m is None or m.games == 0:
            lines.append(f"  {role:14s}: (no games)")
            continue
        lines.append(
            f"  {role:14s}: games={m.games:4d}  WP={m.win_percentage:.4f}  "
            f"mean_score={m.mean_score:+.4f}"
        )
    if any(report.calibration.get(r, {}).get("count", 0) > 0 for r in ROLES):
        lines.append("Calibration (V2 win head):")
        for role in ROLES:
            c = report.calibration.get(role, {})
            if c.get("count", 0) == 0:
                continue
            lines.append(
                f"  {role:14s}: n={int(c['count']):4d}  brier={c['brier']:.4f}  "
                f"nll={c['nll']:.4f}  ece={c['ece']:.4f}"
            )
    return "\n".join(lines)
