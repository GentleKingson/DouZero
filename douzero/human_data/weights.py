"""Sample-weight computation for listwise BC (P08).

Combines per-sample signals into a single non-negative weight used to scale the
listwise cross-entropy. AGENTS.md:

    "Sample weights: player_skill_weight, data integrity, rule match, optional
    action advantage estimate, weight clipping and normalization."

The weight is multiplicative but **clipped** to ``[0, skill_weight_clip]`` so a
single high-skill outlier cannot dominate a batch, and then optionally
**normalized** toward mean one with capped-simplex water filling.  Zero-weight
samples remain zero; if their removal makes mean one infeasible under the cap,
the largest feasible total is used.

All weights are non-negative; a zero weight drops the sample (e.g. a
ruleset-mismatched record). The computation is pure and deterministic so a
fixed dataset yields a fixed weight vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class WeightError(ValueError):
    """Raised when weight configuration is invalid."""


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for :func:`compute_sample_weights`.

    ``skill_weight_clip`` is an absolute cap before and after normalization,
    preventing a single high-skill / high-integrity sample from dominating a
    minibatch. ``rule_mismatch_action`` controls what happens when
    a sample's record ruleset does not match the target:

    - ``"zero"`` (default): weight set to 0 (the sample is dropped).
    - ``"keep"``: ruleset mismatch ignored (weight unaffected).
    """

    skill_weight_clip: float = 10.0
    rule_mismatch_action: str = "zero"
    integrity_default: float = 1.0
    rule_match_default: float = 1.0
    normalize_to_mean: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skill_weight_clip, (int, float)) or isinstance(
            self.skill_weight_clip, bool
        ):
            raise WeightError(
                "skill_weight_clip must be a number, got "
                f"{type(self.skill_weight_clip).__name__}"
            )
        if not math.isfinite(float(self.skill_weight_clip)):
            raise WeightError(
                f"skill_weight_clip must be finite, got {self.skill_weight_clip}"
            )
        if self.skill_weight_clip <= 0:
            raise WeightError(
                f"skill_weight_clip must be positive, got {self.skill_weight_clip}"
            )
        if self.rule_mismatch_action not in ("zero", "keep"):
            raise WeightError(
                f"rule_mismatch_action must be 'zero' or 'keep', got "
                f"{self.rule_mismatch_action!r}"
            )
        for name, val in (("integrity_default", self.integrity_default),
                          ("rule_match_default", self.rule_match_default)):
            val_v = getattr(self, name)
            if not isinstance(val_v, (int, float)) or isinstance(val_v, bool):
                raise WeightError(
                    f"{name} must be a number, got {type(val_v).__name__}"
                )
            if not math.isfinite(float(val_v)):
                raise WeightError(f"{name} must be finite, got {val_v}")
            if not 0.0 <= val_v:
                raise WeightError(
                    f"{name} must be non-negative, got {val_v}"
                )


def compute_sample_weight(
    *,
    skill_weight: float = 1.0,
    integrity_weight: float = 1.0,
    rule_match: bool = True,
    action_advantage: float = 0.0,
    config: WeightConfig | None = None,
) -> float:
    """Compute a single sample weight (clipped, non-negative).

    The raw weight is::

        skill_weight * integrity_weight * rule_match_factor * (1 + advantage)

    where ``rule_match_factor`` is 0 on a ruleset mismatch (default config) or 1
    if ``rule_match`` is True. The result is clipped to
    ``[0, skill_weight_clip]``.
    """
    cfg = config or WeightConfig()
    skill = _finite_number("skill_weight", skill_weight)
    integrity = _finite_number("integrity_weight", integrity_weight)
    advantage = _finite_number("action_advantage", action_advantage)
    if skill < 0.0:
        raise WeightError(f"skill_weight must be non-negative, got {skill_weight}")
    if integrity < 0.0:
        raise WeightError(f"integrity_weight must be non-negative, got {integrity_weight}")
    if not isinstance(rule_match, bool):
        raise WeightError(
            f"rule_match must be a bool, got {type(rule_match).__name__}"
        )
    rule_factor = 1.0
    if not rule_match:
        if cfg.rule_mismatch_action == "zero":
            return 0.0
        rule_factor = cfg.rule_match_default
    advantage_factor = 1.0 + max(0.0, advantage)
    raw = (
        skill
        * integrity
        * float(rule_factor)
        * float(advantage_factor)
    )
    if raw < 0.0:
        raw = 0.0
    return min(raw, float(cfg.skill_weight_clip))


def compute_sample_weights(
    *,
    skill_weights,
    integrity_weights=None,
    rule_matches=None,
    action_advantages=None,
    config: WeightConfig | None = None,
):
    """Vectorised :func:`compute_sample_weight` returning a list of floats.

    All optional sequences default to "all ones / all True". They must be the
    same length as ``skill_weights``. When ``config.normalize_to_mean`` is True
    the clipped weights are water-filled toward a sum of ``len(weights)`` while
    preserving the absolute cap. Zero weights are not revived; consequently an
    extremely sparse vector may have a smaller, maximum-feasible sum.
    """
    cfg = config or WeightConfig()
    n = len(skill_weights)
    if integrity_weights is None:
        integrity_weights = [cfg.integrity_default] * n
    if rule_matches is None:
        rule_matches = [True] * n
    if action_advantages is None:
        action_advantages = [0.0] * n
    if not (
        len(integrity_weights) == n
        and len(rule_matches) == n
        and len(action_advantages) == n
    ):
        raise WeightError(
            "all optional weight sequences must match skill_weights length"
        )

    clipped = [
        compute_sample_weight(
            skill_weight=skill_weights[i],
            integrity_weight=integrity_weights[i],
            rule_match=rule_matches[i],
            action_advantage=action_advantages[i],
            config=cfg,
        )
        for i in range(n)
    ]

    if not cfg.normalize_to_mean or n == 0:
        return clipped

    if not any(w > 0.0 for w in clipped):
        # All-zero weights: return as-is (the caller will see a zero-loss
        # batch; do not divide by zero).
        return clipped
    return _normalize_with_cap(clipped, float(cfg.skill_weight_clip))


def _finite_number(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WeightError(f"{name} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise WeightError(f"{name} must be finite, got {value}")
    return result


def _normalize_with_cap(weights: list[float], cap: float) -> list[float]:
    """Water-fill positive weights without exceeding ``cap``.

    Zero weights remain zero so a rejected ruleset sample is never revived by
    normalization.  When too few positive samples exist to reach mean one
    under the cap, the largest feasible total is used instead.
    """
    result = [0.0] * len(weights)
    active = [i for i, weight in enumerate(weights) if weight > 0.0]
    target = min(float(len(weights)), float(len(active)) * cap)
    remaining = target

    while active:
        active_total = sum(weights[i] for i in active)
        scale = remaining / active_total
        newly_capped = [i for i in active if weights[i] * scale >= cap]
        if not newly_capped:
            for i in active:
                result[i] = weights[i] * scale
            break
        for i in newly_capped:
            result[i] = cap
        remaining -= cap * len(newly_capped)
        capped_set = set(newly_capped)
        active = [i for i in active if i not in capped_set]
        if remaining <= 0.0:
            break

    return result


# --------------------------------------------------------------------------- #
# Stratified statistics (survivorship-bias audit)
# --------------------------------------------------------------------------- #
def stratified_stats(samples) -> dict[str, dict[str, int | float]]:
    """Return per-position, per-winner-team, and per-action-count counts.

    AGENTS.md: "Do not train only on won games. Provide result-stratified
    statistics, to avoid survivorship bias." This helper reports the team,
    position, and legal-action-count distribution so a training run can be
    audited for imbalance. ``by_winner_team`` is the survivorship-bias audit
    key — a dataset where ``by_winner_team`` is dominated by one team signals
    that only winners' decisions were kept.
    """
    by_position: dict[str, int] = {}
    by_winner_team: dict[str, int] = {}
    by_num_actions: dict[str, int] = {}
    for s in samples:
        by_position[s.position] = by_position.get(s.position, 0) + 1
        team = getattr(s, "winner_team", "") or "unknown"
        by_winner_team[team] = by_winner_team.get(team, 0) + 1
        bucket = _bucket(s.num_legal_actions)
        by_num_actions[bucket] = by_num_actions.get(bucket, 0) + 1
    return {
        "total": len(samples),
        "by_position": by_position,
        "by_winner_team": by_winner_team,
        "by_num_legal_actions": by_num_actions,
    }


def _bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 10:
        return "5-10"
    if n <= 30:
        return "11-30"
    return "31+"


def apply_sample_weights(
    samples,
    *,
    config: WeightConfig | None = None,
    integrity_weights=None,
    rule_matches=None,
    action_advantages=None,
):
    """Compute composite weights across a dataset and stamp them onto samples.

    Blocker 4: this is the production wiring. It reads each sample's raw
    ``skill_weight``, computes the clipped+normalized composite weight via
    :func:`compute_sample_weights`, and returns NEW samples (BCSample is frozen)
    with ``sample_weight`` set. The original ``skill_weight`` is preserved.

    ``integrity_weights`` / ``rule_matches`` / ``action_advantages`` default to
    "all neutral" (integrity 1.0, rule_match True, advantage 0.0). Future
    callers can supply per-sample signals (e.g. a ruleset-mismatch mask from
    the record identity).
    """
    import dataclasses

    n = len(samples)
    skill = [float(s.skill_weight) for s in samples]
    computed = compute_sample_weights(
        skill_weights=skill,
        integrity_weights=integrity_weights,
        rule_matches=rule_matches,
        action_advantages=action_advantages,
        config=config,
    )
    out = []
    for s, w in zip(samples, computed):
        out.append(dataclasses.replace(s, sample_weight=float(w)))
    return out
