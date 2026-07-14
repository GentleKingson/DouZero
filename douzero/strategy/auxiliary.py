"""Auxiliary target provenance and differentiable P09 loss combination."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, huber_loss


@dataclass
class StrategyAuxLossComponents:
    total: torch.Tensor
    min_turns_after: float
    regain_initiative: float
    teammate_finish: float
    spring_probability: float
    structure_cost: float

    def as_log_dict(self) -> dict[str, float]:
        return {
            "aux_loss_total": float(self.total.detach().item()),
            "aux_min_turns_after": self.min_turns_after,
            "aux_regain_initiative": self.regain_initiative,
            "aux_teammate_finish": self.teammate_finish,
            "aux_spring_probability": self.spring_probability,
            "aux_structure_cost": self.structure_cost,
        }


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.bool().reshape(-1)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    return binary_cross_entropy_with_logits(
        logits.reshape(-1)[selected], targets.float().reshape(-1)[selected]
    )


def _masked_huber(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask.bool().reshape(-1)
    if not bool(selected.any()):
        return predictions.sum() * 0.0
    return huber_loss(
        predictions.reshape(-1)[selected],
        targets.float().reshape(-1)[selected],
        delta=1.0,
    )


def strategy_auxiliary_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config,
) -> StrategyAuxLossComponents:
    """Combine five independently ablatable auxiliary objectives.

    ``min_turns_after`` and ``structure_cost`` are directly computed labels;
    the three binary targets are generated from the future/terminal trajectory.
    ``teammate_finish_mask`` excludes landlord decisions because landlords have
    no teammate.  Empty masks return graph-connected zero losses.
    """

    required_predictions = {
        "min_turns_after", "regain_initiative_logit",
        "teammate_finish_logit", "spring_probability_logit", "structure_cost",
    }
    missing = required_predictions - predictions.keys()
    if missing:
        raise ValueError(f"missing strategy auxiliary predictions: {sorted(missing)}")
    all_mask = torch.ones_like(targets["min_turns_after"], dtype=torch.bool)
    min_turns = _masked_huber(
        predictions["min_turns_after"],
        targets["min_turns_after"],
        targets["min_turns_exact_mask"],
    )
    regain = _masked_bce(
        predictions["regain_initiative_logit"], targets["regain_initiative"], all_mask
    )
    teammate = _masked_bce(
        predictions["teammate_finish_logit"],
        targets["teammate_finish"],
        targets["teammate_finish_mask"],
    )
    spring = _masked_bce(
        predictions["spring_probability_logit"],
        targets["spring_probability"],
        all_mask,
    )
    structure = huber_loss(
        predictions["structure_cost"].reshape(-1),
        targets["structure_cost"].float().reshape(-1),
        delta=1.0,
    )
    total = (
        float(config.lambda_min_turns) * min_turns
        + float(config.lambda_regain_initiative) * regain
        + float(config.lambda_teammate_finish) * teammate
        + float(config.lambda_spring) * spring
        + float(config.lambda_structure) * structure
    )
    return StrategyAuxLossComponents(
        total=total,
        min_turns_after=float(min_turns.detach().item()),
        regain_initiative=float(regain.detach().item()),
        teammate_finish=float(teammate.detach().item()),
        spring_probability=float(spring.detach().item()),
        structure_cost=float(structure.detach().item()),
    )
