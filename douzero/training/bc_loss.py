"""Listwise behaviour-cloning loss over the legal-action list (P08).

The BC loss trains the optional :class:`~douzero.models_v2.heads.PriorHead` to
predict the recorded human action's position in the *current* legal-action
list. It is a listwise cross-entropy over the N legal actions — NOT a
softmax over a fixed global action-class catalogue. AGENTS.md:

    "Human behaviour cloning must score the current legal-action list; do not
    replace the variable action representation with a brittle global class
    list."

Loss
----
For one decision with N legal actions, prior logits ``z ∈ R^N`` (the prior
head output, padded rows masked to -inf), and the recorded human action at
index ``y``:

    L_BC = cross_entropy(softmax(z), y) = -log( softmax(z)_y )

The masked CE is numerically stable (``F.cross_entropy`` with ``-inf`` padding
rows). Because N varies per decision, the loss is computed **per decision**
(not batched across decisions of different N) and then averaged. A per-sample
``weight`` scales each decision's contribution (see
:mod:`douzero.human_data.weights`).

Imperfect-information boundary
------------------------------
The BC loss reads only the public prior-logit head output + the legal-action
mask + the recorded human action index. The human action index is privileged
training-only data, carried in :class:`~douzero.human_data.sample.BCSample`.
It never reaches the deployment ``DeepAgentV2.act``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class BCLossError(ValueError):
    """Raised when the BC loss inputs are malformed."""


@dataclass(frozen=True)
class BCLossConfig:
    """Configuration for the listwise BC loss.

    ``lambda_bc`` is the weight applied when BC is combined with the
    multi-objective RL loss (P08 task 11). ``temperature`` sharpens or flattens
    the prior distribution before the CE; ``label_smoothing`` is forwarded to
    :func:`torch.nn.functional.cross_entropy`.
    """

    lambda_bc: float = 0.0
    temperature: float = 1.0
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.lambda_bc, bool) or not isinstance(
            self.lambda_bc, (int, float)
        ):
            raise BCLossError(
                f"lambda_bc must be a number, got {type(self.lambda_bc).__name__}"
            )
        if self.lambda_bc < 0.0 or not math.isfinite(self.lambda_bc):
            raise BCLossError(
                f"lambda_bc must be a non-negative finite number, got {self.lambda_bc}"
            )
        if not isinstance(self.temperature, (int, float)) or isinstance(
            self.temperature, bool
        ):
            raise BCLossError(
                f"temperature must be a number, got {type(self.temperature).__name__}"
            )
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise BCLossError(
                f"temperature must be positive finite, got {self.temperature}"
            )
        if (
            not isinstance(self.label_smoothing, (int, float))
            or isinstance(self.label_smoothing, bool)
        ):
            raise BCLossError(
                "label_smoothing must be a number, got "
                f"{type(self.label_smoothing).__name__}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise BCLossError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_bc": float(self.lambda_bc),
            "temperature": float(self.temperature),
            "label_smoothing": float(self.label_smoothing),
        }


@dataclass
class BCLossComponents:
    """Result of a BC loss computation.

    ``total`` carries the gradient; the scalar fields are detached floats for
    logging. ``num_decisions`` is the number of per-decision CE terms averaged.
    """

    total: torch.Tensor
    cross_entropy: float
    num_decisions: int
    top1_correct: int

    def as_log_dict(self) -> dict[str, float]:
        return {
            "bc_cross_entropy": float(self.cross_entropy),
            "bc_num_decisions": float(self.num_decisions),
            "bc_top1_accuracy": (
                float(self.top1_correct) / self.num_decisions
                if self.num_decisions > 0
                else 0.0
            ),
        }


def listwise_bc_loss(
    prior_logit: torch.Tensor,
    action_mask: torch.Tensor,
    target_index: int,
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, bool]:
    """Listwise cross-entropy over N legal actions for ONE decision.

    Parameters
    ----------
    prior_logit:
        Shape ``(N, 1)`` — the prior-head output for one decision.
    action_mask:
        Shape ``(N,)`` bool — ``True`` for a valid action. Padded rows are
        masked to ``-inf`` before the softmax so they never receive probability.
    target_index:
        The recorded human action's row index in the legal-action list. Must
        satisfy ``0 <= target_index < N`` AND ``action_mask[target_index]``
        (the human action must be a valid legal action).
    weight:
        Non-negative scalar weight for this decision's contribution.
    temperature / label_smoothing:
        Forwarded to the temperature scaling / label smoothing of the CE.

    Returns
    -------
    (loss, top1_correct)
        ``loss`` is a scalar tensor carrying the gradient (already scaled by
        ``weight``). ``top1_correct`` is True iff argmax of the masked logits
        equals ``target_index`` (a top-1 accuracy signal for logging).

    Raises
    ------
    BCLossError
        If shapes are wrong, the mask is all-False, the target index is out of
        range or points at a padded action, or the weight is negative.
    """
    if not isinstance(prior_logit, torch.Tensor):
        raise BCLossError(
            f"prior_logit must be a Tensor, got {type(prior_logit).__name__}"
        )
    if prior_logit.ndim != 2 or prior_logit.shape[-1] != 1:
        raise BCLossError(
            f"prior_logit must have shape (N, 1), got {tuple(prior_logit.shape)}"
        )
    n = int(prior_logit.shape[0])
    if n == 0:
        raise BCLossError("prior_logit has zero actions (N=0); undefined loss")
    if not isinstance(action_mask, torch.Tensor):
        raise BCLossError(
            f"action_mask must be a Tensor, got {type(action_mask).__name__}"
        )
    if action_mask.shape != (n,):
        raise BCLossError(
            f"action_mask must have shape ({n},), got {tuple(action_mask.shape)}"
        )
    if action_mask.dtype != torch.bool:
        raise BCLossError(
            f"action_mask must be bool, got {action_mask.dtype}"
        )
    if not bool(action_mask.any()):
        raise BCLossError("cannot compute BC loss over zero valid actions")
    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise BCLossError(
            f"target_index must be an int, got {type(target_index).__name__}"
        )
    if not (0 <= target_index < n):
        raise BCLossError(
            f"target_index {target_index} out of range [0, {n})"
        )
    if not bool(action_mask[target_index].item()):
        raise BCLossError(
            f"target_index {target_index} points at a padded (invalid) action"
        )
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise BCLossError(
            f"weight must be a number, got {type(weight).__name__}"
        )
    if weight < 0.0:
        raise BCLossError(f"weight must be non-negative, got {weight}")
    if temperature <= 0.0:
        raise BCLossError(f"temperature must be positive, got {temperature}")

    logits = prior_logit.squeeze(-1) / float(temperature)
    masked = logits.clone()
    masked[~action_mask] = float("-inf")
    # F.cross_entropy expects (batch, classes, [optional dims]); one decision.
    loss = F.cross_entropy(
        masked.unsqueeze(0),
        torch.tensor([target_index], dtype=torch.long, device=masked.device),
        label_smoothing=float(label_smoothing),
    )
    weighted = loss * float(weight)
    pred = int(torch.argmax(masked).item())
    return weighted, pred == target_index


def average_bc_losses(
    per_decision: list[tuple[torch.Tensor, bool]],
) -> BCLossComponents:
    """Average a list of per-decision (weighted_loss, top1_correct) terms.

    The returned ``total`` is the mean of the weighted per-decision losses
    (a scalar tensor carrying the gradient). When the list is empty the total
    is a zero-valued graph-bearing tensor (so a no-BC minibatch does not break
    the combined loss) and the metrics are zero.
    """
    if not per_decision:
        return BCLossComponents(
            total=torch.zeros((), requires_grad=True, dtype=torch.float32),
            cross_entropy=0.0,
            num_decisions=0,
            top1_correct=0,
        )
    losses = [t for t, _ in per_decision]
    stacked = torch.stack(losses)
    total = stacked.mean()
    ce = float(total.detach().item())
    n = len(per_decision)
    correct = sum(1 for _, hit in per_decision if hit)
    return BCLossComponents(
        total=total, cross_entropy=ce, num_decisions=n, top1_correct=correct
    )
