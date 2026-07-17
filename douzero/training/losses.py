"""Multi-objective losses for the V2 multi-head value model (P06).

The legacy training loop used a single MSE on one scalar head. The V2 model
(:mod:`douzero.models_v2`) produces three trainable heads per legal action:
``win_logit``, ``score_if_win``, ``score_if_loss`` (the derived ``score_mean``
and ``p_win`` are NOT loss targets — see :mod:`douzero.models_v2.heads`).
P06 trains them with two complementary losses:

- ``win_loss`` — :class:`torch.nn.BCEWithLogitsLoss` on ``win_logit`` against
  the team-perspective ``target_win``. Stable by construction (the loss
  itself is the canonical stable BCE).
- ``score_loss`` — Huber loss on the *conditional* score heads using
  per-sample selection. Each sample's prediction is taken from the head
  that matches its actual terminal outcome (``score_if_win`` for won
  samples, ``score_if_loss`` for lost samples), then a single mean Huber
  is computed over the whole minibatch. This keeps the per-sample loss
  weight identical regardless of the batch's win/loss composition
  (P06 r1 fix: the previous ``0.5 * (win_term + loss_term)`` halved the
  only active term on a pure-win or pure-loss batch).

Plus an optional:

- ``uncertainty_nll`` — heteroscedastic Gaussian NLL treating the spread
  between the conditional heads as a learned log-variance proxy. Disabled
  by default (``lambda_uncertainty = 0.0``). P06 r1 fix: this path no
  longer references an undefined variable and accepts ``(B, 1)`` head
  tensors directly.

The total loss is ``L = λ_win·L_win + λ_score·L_score +
λ_uncertainty·L_uncertainty``. All λ live on :class:`LossConfig` so they
are configurable, logged, and audited through the checkpoint manifest's
``effective_config`` block.

Log-score semantics (P06 r1 fix)
--------------------------------
A scalar head cannot simultaneously be supervised toward ``raw=32`` and
``signed_log≈3.5`` — those are incompatible targets. The additive
``lambda_log`` term that did this is removed. Instead
:class:`LossConfig]` carries ``score_target_transform: "raw" | "signed_log"]``:
a mutually exclusive choice of which target the conditional heads are
trained against. When ``"signed_log"``, the heads learn
``sign(s)·log1p(|s|)`` (which compresses the bomb/rocket tail to a few
units and stays well inside the head's clamp), and the decision policy
reads ``score_mean`` on that scale. The two modes cannot be combined at
once; a single set of heads has one consistent supervision target.

Sign-convention guarantee
-------------------------
Every target is in the *acting team's* perspective before reaching this
module (see :mod:`douzero.training.labels`). The loss module performs NO
sign flipping. A unit test enforces landlord/farmer symmetry by feeding
labels with the same magnitude and checking the loss is equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, huber_loss

from douzero.models_v2.output import ModelOutput

#: Valid choices for :attr:`LossConfig.score_target_transform`.
SCORE_TARGET_RAW = "raw"
SCORE_TARGET_SIGNED_LOG = "signed_log"
_VALID_SCORE_TRANSFORMS = frozenset({SCORE_TARGET_RAW, SCORE_TARGET_SIGNED_LOG})


def _validate_nonneg_weight(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-negative number, got {type(value).__name__}")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative number, got bool")
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")


@dataclass(frozen=True)
class LossConfig:
    """Weights and knobs for :class:`MultiObjectiveLoss`.

    All weights default to a sensible multi-objective setting (win = 1.0,
    score = 0.5, uncertainty = 0.0). Setting any λ to 0 disables that term
    cleanly (the corresponding component is exactly zero and receives no
    gradient).

    ``score_delta`` is the Huber delta for the conditional-score loss. The
    bomb/rocket tail can produce raw scores of ±32 or beyond, so a plain
    MSE would let a few huge-tail samples dominate the gradient. Huber
    clamps the gradient contribution of large residuals while remaining
    MSE-like near zero.

    ``score_target_transform`` selects which target the conditional heads
    are supervised against (P06 r1 fix; mutually exclusive — a head cannot
    fit two scales at once):

    - ``"raw"`` (default) — fit the raw team score directly. The target is
      clamped to ``[-score_clamp, score_clamp]`` so it matches what the
      heads can represent (the heads are clamped inside
      :class:`~douzero.models_v2.heads.ValueHeads`).
    - ``"signed_log"`` — fit ``sign(s)·log1p(|s|)``. Compresses the tail
      (``log1p(32) ≈ 3.5``) so the heads operate well inside their clamp.
      Recommended for tail-heavy rulesets. The decision policy then reads
      ``score_mean`` on the signed-log scale.
    """

    lambda_win: float = 1.0
    lambda_score: float = 0.5
    lambda_uncertainty: float = 0.0
    #: P08: listwise BC auxiliary weight. Default 0 disables the BC term (the
    #: pre-P08 path is unchanged). When > 0 the V2 trainer adds
    #: ``lambda_bc * L_BC`` to the RL loss, where ``L_BC`` is the listwise
    #: cross-entropy over the legal-action list on human BC samples. Requires a
    #: model built with ``human_prior_enabled=True`` and a BC sample source.
    lambda_bc: float = 0.0
    lambda_bid_policy: float = 0.0
    lambda_bid_win: float = 0.0
    lambda_bid_score: float = 0.0
    lambda_bid_regret: float = 0.0
    lambda_min_turns: float = 0.0
    lambda_regain_initiative: float = 0.0
    lambda_teammate_finish: float = 0.0
    lambda_spring: float = 0.0
    lambda_structure: float = 0.0
    score_delta: float = 1.0
    score_target_transform: str = SCORE_TARGET_RAW
    #: The head clamp magnitude. The raw target is clamped to
    #: ``[-score_clamp, score_clamp]`` when ``score_target_transform == "raw"``
    #: so it matches what the heads can represent. Must match the value the
    #: model was constructed with (see :class:`ModelV2Config.score_clamp`).
    score_clamp: float = 32.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_win", self.lambda_win),
            ("lambda_score", self.lambda_score),
            ("lambda_uncertainty", self.lambda_uncertainty),
            ("lambda_bc", self.lambda_bc),
            ("lambda_bid_policy", self.lambda_bid_policy),
            ("lambda_bid_win", self.lambda_bid_win),
            ("lambda_bid_score", self.lambda_bid_score),
            ("lambda_bid_regret", self.lambda_bid_regret),
            ("lambda_min_turns", self.lambda_min_turns),
            ("lambda_regain_initiative", self.lambda_regain_initiative),
            ("lambda_teammate_finish", self.lambda_teammate_finish),
            ("lambda_spring", self.lambda_spring),
            ("lambda_structure", self.lambda_structure),
        ):
            _validate_nonneg_weight(name, value)
        if self.score_delta <= 0.0 or not math.isfinite(self.score_delta):
            raise ValueError(f"score_delta must be positive and finite, got {self.score_delta!r}")
        if self.score_target_transform not in _VALID_SCORE_TRANSFORMS:
            raise ValueError(
                f"score_target_transform must be one of {sorted(_VALID_SCORE_TRANSFORMS)}, "
                f"got {self.score_target_transform!r}"
            )
        if self.score_clamp <= 0.0 or not math.isfinite(self.score_clamp):
            raise ValueError(f"score_clamp must be positive and finite, got {self.score_clamp!r}")

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serializable dict (for audit in the checkpoint manifest)."""
        return {
            "lambda_win": float(self.lambda_win),
            "lambda_score": float(self.lambda_score),
            "lambda_uncertainty": float(self.lambda_uncertainty),
            "lambda_bc": float(self.lambda_bc),
            "lambda_bid_policy": float(self.lambda_bid_policy),
            "lambda_bid_win": float(self.lambda_bid_win),
            "lambda_bid_score": float(self.lambda_bid_score),
            "lambda_bid_regret": float(self.lambda_bid_regret),
            "lambda_min_turns": float(self.lambda_min_turns),
            "lambda_regain_initiative": float(self.lambda_regain_initiative),
            "lambda_teammate_finish": float(self.lambda_teammate_finish),
            "lambda_spring": float(self.lambda_spring),
            "lambda_structure": float(self.lambda_structure),
            "score_delta": float(self.score_delta),
            "score_target_transform": str(self.score_target_transform),
            "score_clamp": float(self.score_clamp),
        }


@dataclass
class LossComponents:
    """Per-term loss values returned by :class:`MultiObjectiveLoss`.

    All scalar fields are python floats (detached from the graph) so they
    can be logged directly. ``total`` is the gradient-bearing tensor (kept
    as a :class:`torch.Tensor` so the caller can ``.backward()`` it).
    """

    total: torch.Tensor
    win: float
    score: float
    uncertainty: float
    #: Number of win-team samples in the batch (for diagnostics).
    num_win: int
    #: Number of loss-team samples in the batch (for diagnostics).
    num_loss: int

    def as_log_dict(self) -> dict[str, float]:
        """Return a flat dict suitable for JSONL/TensorBoard logging."""
        return {
            "loss_total": float(self.total.detach().float().item()),
            "loss_win": float(self.win),
            "loss_score": float(self.score),
            "loss_uncertainty": float(self.uncertainty),
            "num_win_samples": float(self.num_win),
            "num_loss_samples": float(self.num_loss),
        }


# --------------------------------------------------------------------------- #
# Individual loss terms (callable; tested in isolation)
# --------------------------------------------------------------------------- #
def bce_win_loss(win_logit: torch.Tensor, target_win: torch.Tensor) -> torch.Tensor:
    """Stable BCE-with-logits win-probability loss.

    Parameters
    ----------
    win_logit:
        Shape ``(B, 1)`` raw logits from the win head.
    target_win:
        Shape ``(B,)`` or ``(B, 1)`` float labels in {0.0, 1.0}.

    Returns
    -------
    torch.Tensor
        Scalar mean BCE loss. Finite by construction (BCE-with-logits is the
        canonical stable form; it never takes ``log(sigmoid(x))`` directly).
    """
    if win_logit.dim() == 2 and win_logit.shape[-1] == 1:
        logits = win_logit.squeeze(-1)
    else:
        logits = win_logit
    target = target_win.float().reshape_as(logits)
    return binary_cross_entropy_with_logits(logits, target, reduction="mean")


def _signed_log(score: torch.Tensor) -> torch.Tensor:
    """Element-wise ``sign(s)·log1p(|s|)`` (the stable log-score transform)."""
    s = score.float()
    # sign(0) == 0 so the transform maps 0 -> 0 naturally.
    return torch.sign(s) * torch.log1p(s.abs())


def resolve_score_target(
    target_score: torch.Tensor,
    *,
    score_target_transform: str,
    score_clamp: float,
    target_log_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform and clamp terminal scores to the Model V2 head scale.

    This is the shared target boundary for ordinary V2 training and P10
    teacher/student training. Callers must not supervise ``score_mean``;
    the returned value targets the applicable conditional score head.
    """

    if score_target_transform not in _VALID_SCORE_TRANSFORMS:
        raise ValueError(
            f"score_target_transform must be one of {sorted(_VALID_SCORE_TRANSFORMS)}, "
            f"got {score_target_transform!r}"
        )
    if (
        isinstance(score_clamp, bool)
        or not isinstance(score_clamp, (int, float))
        or not math.isfinite(score_clamp)
        or score_clamp <= 0.0
    ):
        raise ValueError(f"score_clamp must be positive and finite, got {score_clamp!r}")
    if score_target_transform == SCORE_TARGET_SIGNED_LOG:
        resolved = (
            target_log_score.float()
            if target_log_score is not None
            else _signed_log(target_score)
        )
    else:
        resolved = target_score.float()
    return resolved.clamp(-float(score_clamp), float(score_clamp))


def _select_per_sample(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_win: torch.Tensor,
) -> torch.Tensor:
    """Pick the head matching each sample's terminal outcome.

    Returns a 1-D tensor of length ``B``: ``score_if_win[i]`` where
    ``target_win[i] == 1`` and ``score_if_loss[i]`` otherwise. This is the
    per-sample selection that makes the conditional-score loss independent
    of the batch's win/loss composition (P06 r1 fix).
    """
    if score_if_win.shape != score_if_loss.shape:
        raise ValueError(
            f"score_if_win and score_if_loss must have identical shapes, got "
            f"{tuple(score_if_win.shape)} vs {tuple(score_if_loss.shape)}"
        )
    if score_if_win.shape[-1] != 1:
        raise ValueError(
            "conditional score heads must have trailing dim 1, got "
            f"{tuple(score_if_win.shape)}"
        )
    win_mask = target_win.float().reshape(-1) >= 0.5
    pred = torch.where(
        win_mask.reshape(-1, 1),
        score_if_win,
        score_if_loss,
    ).squeeze(-1)
    return pred


def conditional_score_huber_loss(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_score: torch.Tensor,
    target_win: torch.Tensor,
    *,
    delta: float = 1.0,
    score_clamp: float | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Per-sample Huber loss for the conditional score heads.

    Each sample's prediction is taken from the head matching its actual
    terminal outcome (``score_if_win`` for won samples, ``score_if_loss``
    for lost samples), and a single mean Huber is computed over the whole
    batch. This keeps each sample's contribution to the loss identical
    regardless of how many win/loss samples the batch contains (P06 r1 fix
    for the ``0.5 * (win_term + loss_term)`` scaling bug).

    ``score_clamp`` (when provided) clamps the RAW target to
    ``[-score_clamp, score_clamp]`` so it matches what the heads can
    represent (the heads themselves are clamped inside
    :class:`~douzero.models_v2.heads.ValueHeads`). This prevents a target
    of, e.g., 64 (a 5-bomb legacy ADP landlord game) from saturating the
    head and zeroing its gradient.

    Returns ``(loss, num_win, num_loss)``. The loss is a mean over the
    full batch, so it is always finite (no division by zero even when the
    batch is all-win or all-loss).
    """
    win_labels = target_win.reshape(-1)
    win_mask = win_labels >= 0.5
    num_win = int(win_mask.sum().item())
    num_loss = int(win_labels.shape[0]) - num_win

    pred = _select_per_sample(score_if_win, score_if_loss, target_win)
    target_flat = target_score.float().reshape(-1)
    if score_clamp is not None and score_clamp > 0.0:
        target_flat = target_flat.clamp(-float(score_clamp), float(score_clamp))
    loss = huber_loss(pred, target_flat, reduction="mean", delta=delta)
    return loss, num_win, num_loss


def uncertainty_nll(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_score: torch.Tensor,
    target_win: torch.Tensor,
    *,
    score_clamp: float | None = None,
) -> torch.Tensor:
    """Optional heteroscedastic Gaussian NLL using the head spread as variance.

    The model has no dedicated uncertainty head (P06 explicitly avoids
    adding one — see ``docs/multi_objective_training.md``). As a cheap,
    default-off proxy we treat the spread between the conditional heads as
    a learned log-variance proxy:

        log_var = log1p(|score_if_win - score_if_loss| + eps)
        variance = exp(log_var)

    and compute the standard heteroscedastic Gaussian NLL on the
    applicable head per sample (win samples use ``score_if_win`` as the
    mean; loss samples use ``score_if_loss``). This is an experimental
    auxiliary regularizer (it penalizes over-confident head spreads on the
    tail); it is OFF by default (``lambda_uncertainty = 0.0``) and intended
    for ablation.

    P06 r1 fix: this function no longer references an undefined variable
    and accepts ``(B, 1)`` head tensors directly. The win/loss mask is
    derived from ``target_win`` exactly once.
    """
    if score_if_win.shape != score_if_loss.shape:
        raise ValueError(
            f"score_if_win and score_if_loss must have identical shapes, got "
            f"{tuple(score_if_win.shape)} vs {tuple(score_if_loss.shape)}"
        )
    if score_if_win.shape[-1] != 1:
        raise ValueError(
            "conditional score heads must have trailing dim 1, got "
            f"{tuple(score_if_win.shape)}"
        )
    b = score_if_win.shape[0]
    if b == 0:
        return score_if_win.new_zeros(())

    pred = _select_per_sample(score_if_win, score_if_loss, target_win)
    spread = (score_if_win - score_if_loss).squeeze(-1).abs()
    log_var = torch.log1p(spread + 1e-6)
    target_flat = target_score.float().reshape(-1)
    if score_clamp is not None and score_clamp > 0.0:
        target_flat = target_flat.clamp(-float(score_clamp), float(score_clamp))
    sq = (pred - target_flat) ** 2
    # Heteroscedastic Gaussian NLL: 0.5 * (log_var + sq / exp(log_var)).
    # log_var is bounded below by log1p(eps) so the variance is always
    # positive and finite.
    nll_per_sample = 0.5 * (log_var + sq / torch.exp(log_var))
    return nll_per_sample.mean()


# --------------------------------------------------------------------------- #
# Combiner
# --------------------------------------------------------------------------- #
class MultiObjectiveLoss(nn.Module):
    """Combine the per-head losses into one gradient-bearing scalar.

    The module is an :class:`nn.Module` so it is part of the model's
    state-dict audit (its config is recoverable from ``effective_config``,
    not from learned parameters — the module has NO parameters). Use it as:

        loss_fn = MultiObjectiveLoss(LossConfig(lambda_uncertainty=0.5))
        components = loss_fn.forward_gathered(
            win_logit, score_if_win, score_if_loss, batch_labels
        )
        components.total.backward()

    ``batch_labels`` must carry ``target_win`` and ``target_score`` (and,
    when ``score_target_transform == "signed_log"``, the dict may also
    carry a pre-computed ``target_log_score``; otherwise it is derived
    from ``target_score`` here).
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: ModelOutput,
        batch_labels: dict[str, torch.Tensor],
    ) -> LossComponents:
        """Compute the combined loss for a ModelOutput over one decision.

        The loss is applied to one action per decision; the caller selects
        that action via ``batch_labels['action_indices']`` (absolute row
        indices into the head tensors' leading dim). If ``action_indices``
        is absent, the FIRST valid action of the decision is used.
        """
        action_indices = batch_labels.get("action_indices")
        win_logit, score_if_win, score_if_loss = _gather_action(output, action_indices)
        return self._from_gathered_heads(
            win_logit, score_if_win, score_if_loss, batch_labels
        )

    def forward_gathered(
        self,
        win_logit: torch.Tensor,
        score_if_win: torch.Tensor,
        score_if_loss: torch.Tensor,
        batch_labels: dict[str, torch.Tensor],
    ) -> LossComponents:
        """Compute the combined loss from pre-gathered per-decision heads.

        Each input tensor has shape ``(B, 1)`` — one value per decision for
        the action the actor chose. This is the trainer's primary entry
        point: it forwards each decision separately, gathers the chosen
        action's head values, concatenates them, and calls this method.
        """
        _assert_b1("win_logit", win_logit)
        _assert_b1("score_if_win", score_if_win)
        _assert_b1("score_if_loss", score_if_loss)
        return self._from_gathered_heads(
            win_logit, score_if_win, score_if_loss, batch_labels
        )

    def _resolve_score_target(self, batch_labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the score target the heads will be supervised against.

        When ``score_target_transform == "signed_log"`` the target is
        ``sign(s)·log1p(|s|)`` (computed from ``target_score`` if a
        pre-transformed ``target_log_score`` is not supplied). Otherwise
        the raw ``target_score`` is returned.

        P06 r5: in BOTH modes the resolved target is clamped to
        ``[-score_clamp, score_clamp]`` so it stays inside what the heads
        can represent. For ``raw`` mode this bounds large bomb/rocket
        scores; for ``signed_log`` mode it bounds the edge case where a
        very small ``score_clamp`` (e.g. 0.5) is still smaller than
        ``log1p(1.0)≈0.693`` — the smallest common non-zero score.
        """
        return resolve_score_target(
            batch_labels["target_score"],
            score_target_transform=self.config.score_target_transform,
            score_clamp=self.config.score_clamp,
            target_log_score=batch_labels.get("target_log_score"),
        )

    def _validate_loss_labels(
        self,
        win_logit: torch.Tensor,
        batch_labels: dict[str, torch.Tensor],
    ) -> None:
        """Validate batch labels at the public loss boundary (P06 r5).

        Checks:
        - ``target_win`` has the same leading dim as ``win_logit``, is
          finite, and strictly binary ``{0.0, 1.0}`` (a NaN outcome label
          would silently route to ``score_if_loss`` via
          ``NaN >= 0.5 → False`` in per-sample selection).
        - ``target_score`` has the right length and is finite when any
          score-related loss is active (``lambda_score > 0`` or
          ``lambda_uncertainty > 0``).
        - ``target_log_score`` (when provided and signed_log mode is
          active) has the right length and is finite.
        """
        b = win_logit.shape[0]
        target_win = batch_labels["target_win"]
        tw = target_win.float().reshape(-1)
        if tw.shape[0] != b:
            raise ValueError(
                f"target_win length {tw.shape[0]} != win_logit batch {b}"
            )
        if not bool(torch.isfinite(tw).all()):
            raise ValueError(
                "target_win contains non-finite values (NaN/Inf); the loss "
                "module rejects invalid outcome labels at its boundary."
            )
        # Binary check with tiny tolerance for float serialization.
        dist = (tw - tw.round().clamp(0.0, 1.0)).abs()
        if bool((dist > 1e-6).any()):
            raise ValueError(
                f"target_win must be binary {{0, 1}}, got non-binary values "
                f"(max distance to nearest of {{0,1}} = "
                f"{float(dist.max().item()):.6g})."
            )

        active_score = (
            self.config.lambda_score > 0.0 or self.config.lambda_uncertainty > 0.0
        )
        target_score = batch_labels["target_score"].float().reshape(-1)
        if target_score.shape[0] != b:
            raise ValueError(
                f"target_score length {target_score.shape[0]} != batch {b}"
            )
        if active_score and not bool(torch.isfinite(target_score).all()):
            raise ValueError(
                "target_score contains non-finite values while a score-"
                "related loss is active; the loss module rejects invalid "
                "score labels at its boundary."
            )

        if self.config.score_target_transform == SCORE_TARGET_SIGNED_LOG:
            target_log = batch_labels.get("target_log_score")
            if target_log is not None:
                tl = target_log.float().reshape(-1)
                if tl.shape[0] != b:
                    raise ValueError(
                        f"target_log_score length {tl.shape[0]} != batch {b}"
                    )
                if not bool(torch.isfinite(tl).all()):
                    raise ValueError(
                        "target_log_score contains non-finite values; the "
                        "loss module rejects invalid log-score labels."
                    )

    def _from_gathered_heads(
        self,
        win_logit: torch.Tensor,
        score_if_win: torch.Tensor,
        score_if_loss: torch.Tensor,
        batch_labels: dict[str, torch.Tensor],
    ) -> LossComponents:
        # P06 r5: validate labels at the public loss boundary. The replay
        # buffer's Transition.validate() protects the trainer path, but
        # forward_gathered() is a public API that P14 or external callers
        # can invoke directly. A NaN target_win would silently route to
        # score_if_loss via ``NaN >= 0.5 → False`` in _select_per_sample,
        # turning an invalid label into a "loss sample" instead of raising.
        self._validate_loss_labels(win_logit, batch_labels)
        target_win = batch_labels["target_win"]
        target_score = self._resolve_score_target(batch_labels)

        # P06 r4: skip the computation entirely when the corresponding λ is
        # 0. This avoids (a) wasted compute on disabled terms, and (b)
        # ``0 * NaN = NaN`` propagation when a disabled term would have
        # produced NaN. Each disabled term is a graph-bearing zero so the
        # total stays differentiable when at least one term is active.
        zero = win_logit.new_zeros(())
        clamp = self.config.score_clamp if self.config.score_target_transform == SCORE_TARGET_RAW else None

        if self.config.lambda_win > 0.0:
            win_term = bce_win_loss(win_logit, target_win)
        else:
            win_term = zero

        if self.config.lambda_score > 0.0:
            score_term, num_win, num_loss = conditional_score_huber_loss(
                score_if_win,
                score_if_loss,
                target_score,
                target_win,
                delta=self.config.score_delta,
                score_clamp=clamp,
            )
        else:
            score_term = zero
            # Still compute diagnostics for the log even when disabled.
            win_labels = target_win.reshape(-1)
            num_win = int((win_labels >= 0.5).sum().item())
            num_loss = int(win_labels.shape[0]) - num_win

        total = self.config.lambda_win * win_term + self.config.lambda_score * score_term

        # Optional uncertainty-NLL term (default off).
        if self.config.lambda_uncertainty > 0.0:
            unc_term = uncertainty_nll(
                score_if_win,
                score_if_loss,
                target_score,
                target_win,
                score_clamp=clamp,
            )
            total = total + self.config.lambda_uncertainty * unc_term
        else:
            unc_term = zero

        return LossComponents(
            total=total,
            win=float(win_term.detach().float().item()),
            score=float(score_term.detach().float().item()),
            uncertainty=float(unc_term.detach().float().item()),
            num_win=num_win,
            num_loss=num_loss,
        )


def _assert_b1(name: str, t: torch.Tensor) -> None:
    if t.dim() != 2 or t.shape[-1] != 1:
        raise ValueError(
            f"{name} must have shape (B, 1), got {tuple(t.shape)} (P06 r1: the "
            f"trainer must concatenate per-decision heads with torch.cat, not "
            f"torch.stack which produces (B, 1, 1))."
        )


def _gather_action(
    output: ModelOutput,
    action_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one action per decision from the (B*N, 1) head tensors.

    ``action_indices`` are absolute row indices into the head tensors'
    leading dim. If ``None``, the first valid action of the (single)
    decision is selected — a deterministic fallback for synthetic tests.

    P06 r7: validates the indices before gathering:
    - dtype must be integer (bool is rejected because ``True``/``False``
      are silently cast to 1/0);
    - negative indices are rejected (PyTorch wraps them to the tail,
      silently selecting the wrong row);
    - out-of-bounds indices are rejected;
    - indices pointing at padded rows (``action_mask=False``) are rejected.
    """
    head = output.win_logit
    if head.dim() != 2 or head.shape[-1] != 1:
        raise ValueError(
            "expected head tensor of shape (B*N, 1), got " f"{tuple(head.shape)}"
        )
    num_rows = head.shape[0]
    if action_indices is None:
        # Single-decision fallback: take the first valid action.
        mask = output.action_mask
        idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
        return (
            output.win_logit[idx : idx + 1],
            output.score_if_win[idx : idx + 1],
            output.score_if_loss[idx : idx + 1],
        )

    # P06 r7: reject bool and non-integer dtypes before the implicit
    # .to(torch.long) cast silently converts them.
    if isinstance(action_indices, torch.Tensor):
        if action_indices.dtype == torch.bool:
            raise TypeError(
                "action_indices must have an integer dtype, got torch.bool. "
                "True/False would be silently cast to 1/0."
            )
        if action_indices.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"action_indices must have an integer dtype, got "
                f"{action_indices.dtype}. Float indices would be silently "
                f"truncated."
            )

    rows = action_indices.reshape(-1).to(torch.long)

    if rows.numel() == 0:
        raise ValueError(
            "action_indices is empty; at least one action must be selected."
        )
    if bool((rows < 0).any()):
        raise ValueError(
            f"action_indices contains negative values "
            f"({rows[rows < 0].tolist()}); negative indices would wrap to "
            f"the tail and silently select the wrong row."
        )
    if bool((rows >= num_rows).any()):
        raise ValueError(
            f"action_indices contains out-of-bounds values "
            f"(max {int(rows.max().item())} for {num_rows} action rows)."
        )
    # Reject indices pointing at padded (action_mask=False) rows. The
    # ModelOutput contract requires consumers to honour the mask.
    mask = output.action_mask
    if mask is not None and mask.shape[0] == num_rows:
        selected_mask = mask[rows]
        if not bool(selected_mask.all()):
            bad = rows[~selected_mask].tolist()
            raise ValueError(
                f"action_indices select padded/invalid action rows: {bad}. "
                f"The action_mask is False for these rows (padding)."
            )

    return (
        output.win_logit[rows],
        output.score_if_win[rows],
        output.score_if_loss[rows],
    )
