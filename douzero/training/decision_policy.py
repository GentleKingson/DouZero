"""Multi-objective action selection over a :class:`ModelOutput` (P06).

The legacy deployment path selected the action with the highest scalar
``values`` estimate. The V2 model produces three heads (``win_logit``,
``score_if_win``, ``score_if_loss``) plus a derived ``score_mean`` and
``p_win``. P06 lets a deployment choose among several conversion strategies
without re-running the model:

- ``pure_win``       — argmax ``p_win`` (the legacy V2 default, aliased ``win``).
- ``pure_score``     — argmax ``expected_score = score_mean`` (aliased ``score``).
- ``win_then_score`` — keep actions whose ``p_win`` is within tolerance of the
  best, then break ties by ``score_mean``.
- ``score_then_win`` — keep actions whose ``score_mean`` is within tolerance
  of the best, then break ties by ``p_win``.
- ``risk_aware``     — ``score_mean - λ · uncertainty_penalty`` (default off;
  the penalty uses a ``p_win``-derived proxy since Model V2 has no dedicated
  uncertainty head — see :mod:`douzero.training.losses` for the same
  decision).

Tolerance semantics
-------------------
Each lexicographic mode keeps all actions within a tolerance band of the
best. The tolerance is additive (NOT multiplicative), so it behaves
identically for negative and positive values — a multiplicative threshold
``|x - best| <= rel · |best|`` would silently widen the band for large
magnitudes and collapse it near zero, which is the exact bug AGENTS.md warns
about ("negative scores must not use a wrong multiplicative threshold").

The band is: ``value >= best - abs_tol - rel_tol · max(1, |best|)``. The
``max(1, |best|)`` factor scales the relative tolerance smoothly through
zero without becoming a multiplicative threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from douzero.models_v2.output import ModelOutput


#: Canonical decision mode names. ``win`` and ``score`` are kept as aliases
#: for the P05 ``decision_mode`` values so the DeepAgentV2 validator stays
#: backward-compatible. ``pure_prior`` is the P08 ablation mode that selects the
#: highest human-play prior logit (requires a model built with
#: ``human_prior_enabled=True``).
SUPPORTED_DECISION_MODES: tuple[str, ...] = (
    "pure_win",
    "pure_score",
    "win_then_score",
    "score_then_win",
    "risk_aware",
    "pure_prior",
    "uncertainty_gated_prior",
    # P05 aliases (kept for backward compatibility with the existing
    # DeepAgentV2 ``decision_mode`` argument).
    "win",
    "score",
)

#: Map from alias to canonical mode. ``win`` -> ``pure_win``;
#: ``score`` -> ``pure_score``.
_ALIAS_MAP: dict[str, str] = {
    "win": "pure_win",
    "score": "pure_score",
    "pure_win": "pure_win",
    "pure_score": "pure_score",
    "win_then_score": "win_then_score",
    "score_then_win": "score_then_win",
    "risk_aware": "risk_aware",
    "pure_prior": "pure_prior",
    "uncertainty_gated_prior": "uncertainty_gated_prior",
}


def canonical_mode(mode: str) -> str:
    """Return the canonical mode name, resolving aliases."""
    if mode not in _ALIAS_MAP:
        raise ValueError(
            f"unknown decision mode {mode!r}; supported: {SUPPORTED_DECISION_MODES}"
        )
    return _ALIAS_MAP[mode]


@dataclass(frozen=True)
class DecisionConfig:
    """Configuration for :func:`select_action`.

    Tolerances are additive-only (see module docstring). The
    ``risk_penalty`` controls the (default-off) ``risk_aware`` mode.
    """

    mode: str = "pure_win"
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    risk_penalty: float = 0.0
    prior_alpha: float = 0.0

    def __post_init__(self) -> None:
        # Resolve aliases eagerly so a stored config always carries the
        # canonical name (consistent logging / checkpointing).
        object.__setattr__(self, "mode", canonical_mode(self.mode))
        if self.abs_tol < 0.0 or not math.isfinite(self.abs_tol):
            raise ValueError(f"abs_tol must be non-negative finite, got {self.abs_tol!r}")
        if self.rel_tol < 0.0 or not math.isfinite(self.rel_tol):
            raise ValueError(f"rel_tol must be non-negative finite, got {self.rel_tol!r}")
        if self.risk_penalty < 0.0 or not math.isfinite(self.risk_penalty):
            raise ValueError(
                f"risk_penalty must be non-negative finite, got {self.risk_penalty!r}"
            )
        if self.prior_alpha < 0.0 or not math.isfinite(self.prior_alpha):
            raise ValueError(
                f"prior_alpha must be non-negative finite, got {self.prior_alpha!r}"
            )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "abs_tol": float(self.abs_tol),
            "rel_tol": float(self.rel_tol),
            "risk_penalty": float(self.risk_penalty),
            "prior_alpha": float(self.prior_alpha),
        }


def _within_tolerance_band(
    values: torch.Tensor,
    mask: torch.Tensor,
    abs_tol: float,
    rel_tol: float,
) -> torch.Tensor:
    """Boolean mask of actions within ``tol`` of the masked best.

    The band is additive: ``value >= best - abs_tol - rel_tol * max(1, |best|)``
    where ``best`` is the maximum value among masked-in actions. This is
    sign-safe: a band of 0.05 around ``best = -0.5`` keeps actions down to
    ``-0.55`` (NOT ``-0.5 * 1.05 = -0.525``), so the same call works
    identically for ``p_win ∈ [0, 1]`` and ``score_mean`` which can be
    negative.
    """
    if not bool(mask.any()):
        raise ValueError("cannot form a tolerance band over zero valid actions")
    masked_vals = values.clone()
    masked_vals[~mask] = float("-inf")
    best = masked_vals.max()
    scale = max(1.0, float(best.abs().item()))
    threshold = best - float(abs_tol) - float(rel_tol) * scale
    return (values >= threshold) & mask


def _argmax_masked(values: torch.Tensor, mask: torch.Tensor) -> int:
    """Argmax over masked-in entries (entries with ``mask == False`` are -inf)."""
    if not bool(mask.any()):
        raise ValueError("cannot argmax over zero valid actions")
    v = values.clone()
    v[~mask] = float("-inf")
    return int(torch.argmax(v).item())


def _tiebreak_argmax(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    mask: torch.Tensor,
    abs_tol: float,
    rel_tol: float,
) -> int:
    """Keep actions in the primary tolerance band, then argmax secondary."""
    band = _within_tolerance_band(primary, mask, abs_tol, rel_tol)
    return _argmax_masked(secondary, band)


def select_action(
    output: ModelOutput,
    config: DecisionConfig | None = None,
    *,
    mode: str | None = None,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    risk_penalty: float | None = None,
    prior_alpha: float | None = None,
) -> int:
    """Select an action index from a :class:`ModelOutput` per the configured mode.

    Can be called either with a :class:`DecisionConfig` or with keyword
    overrides (keyword wins when both are supplied). Padded actions
    (``action_mask == False``) are NEVER selected. When two valid actions
    are exactly tied, the lowest index wins (PyTorch ``argmax`` semantics).
    """
    if config is None:
        config = DecisionConfig(
            mode=mode or "pure_win",
            abs_tol=abs_tol or 0.0,
            rel_tol=rel_tol or 0.0,
            risk_penalty=risk_penalty or 0.0,
            prior_alpha=prior_alpha or 0.0,
        )
    else:
        # Overlay explicit keyword overrides.
        m = canonical_mode(mode) if mode is not None else config.mode
        at = config.abs_tol if abs_tol is None else float(abs_tol)
        rt = config.rel_tol if rel_tol is None else float(rel_tol)
        rp = config.risk_penalty if risk_penalty is None else float(risk_penalty)
        pa = config.prior_alpha if prior_alpha is None else float(prior_alpha)
        config = DecisionConfig(
            mode=m, abs_tol=at, rel_tol=rt, risk_penalty=rp, prior_alpha=pa
        )

    mask = output.action_mask
    if not bool(mask.any()):
        raise ValueError("cannot select from zero valid actions")

    p_win = output.p_win.squeeze(-1)
    score_mean = output.score_mean.squeeze(-1)

    if config.mode == "pure_win":
        return _argmax_masked(p_win, mask)
    if config.mode == "pure_score":
        return _argmax_masked(score_mean, mask)
    if config.mode == "win_then_score":
        return _tiebreak_argmax(p_win, score_mean, mask, config.abs_tol, config.rel_tol)
    if config.mode == "score_then_win":
        return _tiebreak_argmax(score_mean, p_win, mask, config.abs_tol, config.rel_tol)
    if config.mode == "risk_aware":
        # Uncertainty proxy: high p_win variance proxy + score spread.
        # p_win*(1-p_win) peaks at 0.25 when p_win=0.5 (most uncertain); the
        # score-spread |score_if_win - score_if_loss| captures magnitude
        # uncertainty. Default risk_penalty=0 reduces to pure_score.
        win_unc = p_win * (1.0 - p_win)
        score_spread = (output.score_if_win - output.score_if_loss).squeeze(-1).abs()
        # Normalize the score spread by a running max over valid actions so
        # the two uncertainty terms are on comparable scales.
        spread_max = score_spread[mask].max().clamp(min=1e-6)
        norm_spread = score_spread / spread_max
        penalty = win_unc + 0.5 * norm_spread
        risk_adjusted = score_mean - config.risk_penalty * penalty
        return _argmax_masked(risk_adjusted, mask)
    if config.mode == "pure_prior":
        # P08 ablation: select the highest human-play prior logit. Requires a
        # model built with human_prior_enabled=True (a prior head); the helper
        # raises ValueError when prior_logit is None.
        return output.argmax_prior()
    if config.mode == "uncertainty_gated_prior":
        prior = output.selected_prior_logit()
        valid_prior = prior[mask]
        mean = valid_prior.mean()
        std = valid_prior.std(unbiased=False).clamp(min=1e-6)
        normalized_prior = ((prior - mean) / std).clamp(-3.0, 3.0)
        # Bernoulli uncertainty is zero at confident endpoints and one at
        # p=0.5. alpha=0 is exactly pure_score, the default-off guarantee.
        uncertainty_gate = 4.0 * p_win * (1.0 - p_win)
        final_score = (
            score_mean
            + config.prior_alpha * uncertainty_gate * normalized_prior
        )
        return _argmax_masked(final_score, mask)
    raise ValueError(f"unhandled decision mode {config.mode!r}")  # pragma: no cover
