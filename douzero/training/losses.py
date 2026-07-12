"""Multi-objective losses for the V2 multi-head value model (P06).

The legacy training loop used a single MSE on one scalar head. The V2 model
(:mod:`douzero.models_v2`) produces three trainable heads per legal action:
``win_logit``, ``score_if_win``, ``score_if_loss`` (the derived ``score_mean``
and ``p_win`` are NOT loss targets — see
:mod:`douzero.models_v2.heads`). P06 trains them with three complementary
losses:

- ``win_loss`` — :class:`torch.nn.BCEWithLogitsLoss` on ``win_logit`` against
  the team-perspective ``target_win``. Stable by construction (the loss
  itself is the canonical stable BCE).
- ``score_loss`` — masked Huber loss on the *conditional* score heads.
  ``score_if_win`` is supervised only on samples where the acting team won
  (``target_win == 1``); ``score_if_loss`` only where the acting team lost
  (``target_win == 0``). When the relevant mask is empty the loss is exactly
  zero and produces no NaN/Inf, so a pure-win or pure-loss minibatch trains
  cleanly.
- ``log_score_loss`` — optional Huber loss on a log-score auxiliary target
  against the win/loss-conditional score head. Disabled by default (weight
  0.0). Enabled by setting ``lambda_log > 0``.

- ``uncertainty_nll`` — optional heteroscedastic NLL treating ``score_if_win``
  and ``score_if_loss`` as Gaussian with a learned log-variance derived from
  the head spread. Disabled by default (``lambda_uncertainty = 0.0``).

The total loss is ``L = λ_win·L_win + λ_score·L_score + λ_log·L_log +
λ_uncertainty·L_uncertainty``. All λ live on :class:`LossConfig` so they are
configurable, logged, and audited through the checkpoint manifest's
``effective_config`` block.

Sign-convention guarantee
-------------------------
Every target is in the *acting team's* perspective before reaching this
module (see :mod:`douzero.training.labels`). The loss module performs NO
sign flipping — that is the whole point of centralizing the convention in
``labels.py``. A unit test enforces landlord/farmer symmetry by feeding
labels with the same magnitude and opposite sign and checking the loss is
equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, huber_loss

from douzero.models_v2.output import ModelOutput


def _validate_nonneg_weight(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-negative number, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")


@dataclass(frozen=True)
class LossConfig:
    """Weights and knobs for :class:`MultiObjectiveLoss`.

    All weights default to a sensible multi-objective setting (win = 1.0,
    score = 0.5, log = 0.0, uncertainty = 0.0). Setting any λ to 0 disables
    that term cleanly (the corresponding component is exactly zero and
    receives no gradient).

    ``score_delta`` is the Huber delta for the conditional-score losses. The
    bomb/rocket tail can produce scores of ±32 or beyond, so a plain MSE
    would let a few huge-tail samples dominate the gradient. Huber clamps
    the gradient contribution of large residuals while remaining MSE-like
    near zero.
    """

    lambda_win: float = 1.0
    lambda_score: float = 0.5
    lambda_log: float = 0.0
    lambda_uncertainty: float = 0.0
    score_delta: float = 1.0
    log_score_delta: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_win", self.lambda_win),
            ("lambda_score", self.lambda_score),
            ("lambda_log", self.lambda_log),
            ("lambda_uncertainty", self.lambda_uncertainty),
        ):
            _validate_nonneg_weight(name, value)
        if self.score_delta <= 0.0 or not math.isfinite(self.score_delta):
            raise ValueError(f"score_delta must be positive and finite, got {self.score_delta!r}")
        if self.log_score_delta <= 0.0 or not math.isfinite(self.log_score_delta):
            raise ValueError(
                f"log_score_delta must be positive and finite, got {self.log_score_delta!r}"
            )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dict (for audit in the checkpoint manifest)."""
        return {
            "lambda_win": float(self.lambda_win),
            "lambda_score": float(self.lambda_score),
            "lambda_log": float(self.lambda_log),
            "lambda_uncertainty": float(self.lambda_uncertainty),
            "score_delta": float(self.score_delta),
            "log_score_delta": float(self.log_score_delta),
        }


@dataclass
class LossComponents:
    """Per-term loss values returned by :class:`MultiObjectiveLoss`.

    All fields are python floats (detached from the graph) so they can be
    logged directly. ``total`` is the gradient-bearing tensor (kept as a
    :class:`torch.Tensor` so the caller can ``.backward()`` it).
    """

    total: torch.Tensor
    win: float
    score: float
    log: float
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
            "loss_log": float(self.log),
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


def _masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Mean Huber loss over the masked subset; zero when the mask is empty.

    ``mask`` is a bool tensor. When ``mask.any()`` is False the result is a
    zero scalar (with ``requires_grad`` preserved so the total loss stays a
    graph-bearing tensor even if every other term is also disabled).
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")
    if not bool(mask.any()):
        # Zero scalar that still participates in autograd (so a disabled loss
        # term does not break the graph when other terms are active).
        return prediction.new_zeros(())
    pred = prediction.squeeze(-1)[mask]
    tgt = target.float().reshape_as(prediction.squeeze(-1))[mask]
    return huber_loss(pred, tgt, reduction="mean", delta=delta)


def conditional_score_huber_loss(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_score: torch.Tensor,
    target_win: torch.Tensor,
    *,
    delta: float = 1.0,
) -> tuple[torch.Tensor, int, int]:
    """Masked Huber loss for the two conditional score heads.

    ``score_if_win`` is supervised on samples where ``target_win == 1``;
    ``score_if_loss`` is supervised on samples where ``target_win == 0``.

    Returns ``(loss, num_win, num_loss)``. When either subset is empty the
    corresponding contribution is exactly zero (and produces no NaN). The
    two terms are averaged together with equal weight, which matches the
    AGENTS.md guidance of training each conditional head only on the
    applicable subset.

    Notes
    -----
    On a minibatch that is all-win or all-loss, exactly one of the two terms
    is non-zero. This is the correct behaviour: the un-supervised head
    receives no gradient signal, and the loss does not blow up trying to fit
    a meaningless target.
    """
    if score_if_win.shape[-1] != 1 or score_if_loss.shape[-1] != 1:
        raise ValueError(
            "conditional score heads must have trailing dim 1, got "
            f"score_if_win {tuple(score_if_win.shape)}, "
            f"score_if_loss {tuple(score_if_loss.shape)}"
        )
    win_labels = target_win.reshape(-1)
    win_mask = win_labels >= 0.5
    loss_mask = ~win_mask
    num_win = int(win_mask.sum().item())
    num_loss = int(loss_mask.sum().item())

    target_score_flat = target_score.reshape_as(score_if_win.squeeze(-1))
    win_term = _masked_huber(score_if_win, target_score_flat, win_mask, delta)
    loss_term = _masked_huber(score_if_loss, target_score_flat, loss_mask, delta)
    # Equal weight between the two conditional terms (each is already a mean
    # over its own masked subset, so this is NOT a sum). When one subset is
    # empty its term is zero, and the average reduces to the other term.
    combined = 0.5 * (win_term + loss_term)
    return combined, num_win, num_loss


def log_score_aux_loss(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_log_score: torch.Tensor,
    target_win: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Optional Huber loss on a log-score auxiliary target.

    Same masking structure as :func:`conditional_score_huber_loss`, but the
    target is the numerically stable ``sign(s)·log1p(|s|)`` transform. This
    compresses the long tail so a 32x-bomb game and a 1x game contribute
    comparable gradient magnitudes. Disabled when ``lambda_log == 0``.
    """
    win_labels = target_win.reshape(-1)
    win_mask = win_labels >= 0.5
    loss_mask = ~win_mask
    target_flat = target_log_score.reshape_as(score_if_win.squeeze(-1))
    win_term = _masked_huber(score_if_win, target_flat, win_mask, delta)
    loss_term = _masked_huber(score_if_loss, target_flat, loss_mask, delta)
    return 0.5 * (win_term + loss_term)


def _uncertainty_nll(
    score_if_win: torch.Tensor,
    score_if_loss: torch.Tensor,
    target_score: torch.Tensor,
    target_win: torch.Tensor,
) -> torch.Tensor:
    """Optional heteroscedastic Gaussian NLL using the head spread as variance.

    The model has no dedicated uncertainty head (P06 explicitly avoids adding
    one — see the docs/multi_objective_training.md decision log). As a
    cheap, default-off proxy we treat the spread between the conditional
    heads as a learned log-variance proxy:

        log_var = log1p(|score_if_win - score_if_loss| + eps)
        variance = exp(log_var)

    and compute the standard heteroscedastic Gaussian NLL on the applicable
    subset (win samples use ``score_if_win`` as the mean; loss samples use
    ``score_if_loss``). This is an experimental auxiliary regularizer (it
    penalizes over-confident head spreads on the tail); it is OFF by default
    (``lambda_uncertainty = 0.0``) and intended for ablation.
    """
    del target_win  # mask derived implicitly via the chosen head below
    win_labels = target_win.reshape(-1)
    win_mask = win_labels >= 0.5
    loss_mask = ~win_mask
    if not (bool(win_mask.any()) or bool(loss_mask.any())):
        return score_if_win.new_zeros(())
    # Mean prediction per sample (use the head matching the actual outcome).
    pred = torch.where(
        win_mask.reshape(-1, 1),
        score_if_win,
        score_if_loss,
    ).squeeze(-1)
    spread = (score_if_win - score_if_loss).squeeze(-1).abs()
    log_var = torch.log1p(spread + 1e-6)
    target_flat = target_score.reshape_as(pred)
    sq = (pred - target_flat) ** 2
    # NLL = 0.5 * (log_var + sq / exp(log_var))
    nll_per_sample = 0.5 * (log_var + sq / torch.exp(log_var))
    # Average over the active subset (samples where the corresponding head
    # is being supervised — i.e. always, since every sample is either a win
    # or a loss).
    mask = win_mask | loss_mask
    return nll_per_sample[mask].mean()


# --------------------------------------------------------------------------- #
# Combiner
# --------------------------------------------------------------------------- #
class MultiObjectiveLoss(nn.Module):
    """Combine the per-head losses into one gradient-bearing scalar.

    The module is an :class:`nn.Module` so it is part of the model's
    state-dict audit (its config is recoverable from ``effective_config``,
    not from learned parameters — the module has NO parameters). Use it as:

        loss_fn = MultiObjectiveLoss(LossConfig(lambda_log=0.25))
        components = loss_fn(model_output, batch_labels)
        components.total.backward()

    The ``batch_labels`` dict must carry ``target_win``, ``target_score``,
    and (when the log term is enabled) ``target_log_score`` — exactly the
    keys produced by :func:`douzero.training.labels.team_targets`.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        output: ModelOutput,
        batch_labels: dict[str, torch.Tensor],
    ) -> LossComponents:
        """Compute the combined loss for one minibatch.

        Parameters
        ----------
        output:
            The :class:`ModelOutput` from a V2 forward pass. The loss is
            applied to one action per decision; the caller selects that
            action via ``batch_labels['action_indices']`` (absolute row
            indices into the head tensors' leading dim). If
            ``action_indices`` is absent, the FIRST valid action of each
            decision is used (a deterministic fallback for synthetic tests
            where the choice is arbitrary). For a single-decision
            ModelOutput this reduces to picking the first valid action.
        batch_labels:
            Dict with keys ``target_win`` ``(B,)`` or ``(B,1)``, and
            ``target_score`` ``(B,)`` or ``(B,1)``. When
            ``config.lambda_log > 0`` the dict must also carry
            ``target_log_score``.
        """
        action_indices = batch_labels.get("action_indices")
        win_logit, score_if_win, score_if_loss = _gather_action(
            output, action_indices
        )
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
        action's head values, stacks them, and calls this method.
        """
        return self._from_gathered_heads(
            win_logit, score_if_win, score_if_loss, batch_labels
        )

    def _from_gathered_heads(
        self,
        win_logit: torch.Tensor,
        score_if_win: torch.Tensor,
        score_if_loss: torch.Tensor,
        batch_labels: dict[str, torch.Tensor],
    ) -> LossComponents:
        target_win = batch_labels["target_win"]
        target_score = batch_labels["target_score"]

        # Win loss (always on; λ=0 disables it).
        win_term = bce_win_loss(win_logit, target_win)
        # Conditional score loss.
        score_term, num_win, num_loss = conditional_score_huber_loss(
            score_if_win,
            score_if_loss,
            target_score,
            target_win,
            delta=self.config.score_delta,
        )

        total = self.config.lambda_win * win_term + self.config.lambda_score * score_term

        # Optional log-score auxiliary term.
        if self.config.lambda_log > 0.0:
            target_log = batch_labels.get("target_log_score")
            if target_log is None:
                raise KeyError(
                    "LossConfig.lambda_log > 0 but batch_labels is missing "
                    "'target_log_score'. Provide it via team_targets()."
                )
            log_term = log_score_aux_loss(
                score_if_win,
                score_if_loss,
                target_log,
                target_win,
                delta=self.config.log_score_delta,
            )
            total = total + self.config.lambda_log * log_term
        else:
            log_term = win_logit.new_zeros(())

        # Optional uncertainty-NLL term.
        if self.config.lambda_uncertainty > 0.0:
            unc_term = _uncertainty_nll(
                score_if_win,
                score_if_loss,
                target_score,
                target_win,
            )
            total = total + self.config.lambda_uncertainty * unc_term
        else:
            unc_term = win_logit.new_zeros(())

        return LossComponents(
            total=total,
            win=float(win_term.detach().float().item()),
            score=float(score_term.detach().float().item()),
            log=float(log_term.detach().float().item()),
            uncertainty=float(unc_term.detach().float().item()),
            num_win=num_win,
            num_loss=num_loss,
        )


def _gather_action(
    output: ModelOutput,
    action_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one action per decision from the (B, N, 1) head tensors.

    The ModelOutput carries per-action head tensors of shape ``(B*N, 1)`` or
    ``(N, 1)`` (for a single decision). For training we need one value per
    decision. ``action_indices`` is an integer tensor of shape ``(B,)`` (or
    ``(B, 1)``) giving the chosen action's row index within each decision's
    block of ``N`` rows.

    To support both layouts (batched flat ``(B*N, 1)`` and single ``(N, 1)``)
    we infer the batch size from the label tensor inside the caller. Here we
    accept the indices and gather; if ``action_indices`` is ``None`` we take
    the FIRST valid action per decision (a deterministic fallback used only
    in synthetic tests).
    """
    n = output.num_actions
    # The output heads always have shape (N, 1) for one decision. For B
    # decisions batched together, the caller supplies a stacked ModelOutput
    # whose head tensors are (B*N, 1) (action blocks concatenated). The
    # ``action_indices`` select one row per decision.
    if action_indices is None:
        # Single-decision fallback: take the first valid action.
        mask = output.action_mask
        if output.win_logit.dim() == 2 and output.win_logit.shape[0] == n:
            # (N, 1) — single decision
            idx = torch.nonzero(mask, as_tuple=False)[0].item()
            return (
                output.win_logit[idx : idx + 1],
                output.score_if_win[idx : idx + 1],
                output.score_if_loss[idx : idx + 1],
            )
        # Fall through: assume first row of each block.
        action_indices = torch.zeros(
            output.win_logit.shape[0] // max(n, 1),
            dtype=torch.long,
            device=output.win_logit.device,
        )

    idx = action_indices.reshape(-1).to(torch.long)
    # If the head tensor is a flat (B*N, 1), build absolute offsets.
    head = output.win_logit
    if head.dim() != 2 or head.shape[-1] != 1:
        raise ValueError(
            "expected head tensor of shape (B*N, 1), got " f"{tuple(head.shape)}"
        )
    total_rows = head.shape[0]
    b = idx.shape[0]
    if total_rows % b != 0:
        raise ValueError(
            f"head tensor has {total_rows} rows but {b} decisions were supplied; "
            f"the per-decision action count {total_rows // b} is inconsistent."
        )
    n_per = total_rows // b
    offsets = torch.arange(b, device=head.device) * n_per
    rows = (idx + offsets).clamp(min=0, max=total_rows - 1)
    return (
        output.win_logit[rows],
        output.score_if_win[rows],
        output.score_if_loss[rows],
    )
