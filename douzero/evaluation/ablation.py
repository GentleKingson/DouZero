"""Explicit checkpoint-backed ablation matrix runner for P15."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .paired import PairedEvaluationResult, evaluate_scenario
from .scenario import (
    BundleSpec,
    EvaluationScenario,
    default_seat_permutations,
)


ABLATION_NAMES = (
    "no_bidding",
    "single_head",
    "no_belief",
    "no_human_bc",
    "no_auxiliary",
    "no_distillation",
    "no_population",
    "no_search",
)


@dataclass(frozen=True)
class AblationVariant:
    """Candidate and optional baseline bundle for one named ablation."""

    candidate: BundleSpec
    baseline: BundleSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, BundleSpec):
            raise TypeError("ablation candidate must be a BundleSpec")
        if self.baseline is not None and not isinstance(self.baseline, BundleSpec):
            raise TypeError("ablation baseline must be a BundleSpec or null")


class AblationRunner:
    """Run named, independently trained candidate bundles on one deal set.

    P15 deliberately does not mutate a checkpoint's architecture flags at
    evaluation time: doing so either violates manifest identity or does not
    remove what training learned. Each ablation therefore names an explicit
    compatible bundle/checkpoint set. This makes the matrix reproducible and
    prevents a cosmetic toggle from being reported as an experiment.
    """

    def __init__(
        self,
        scenario: EvaluationScenario,
        variants: Mapping[str, BundleSpec | AblationVariant],
        *,
        require_complete: bool = False,
    ) -> None:
        unknown = set(variants) - set(ABLATION_NAMES)
        if unknown:
            raise ValueError(f"unknown ablations: {sorted(unknown)}")
        if require_complete:
            missing = set(ABLATION_NAMES) - set(variants)
            if missing:
                raise ValueError(f"incomplete ablation matrix; missing {sorted(missing)}")
        if any(
            not isinstance(variant, (BundleSpec, AblationVariant))
            for variant in variants.values()
        ):
            raise TypeError(
                "each ablation variant must be a BundleSpec or AblationVariant"
            )
        self.scenario = scenario
        self.variants = {
            name: (
                variant
                if isinstance(variant, AblationVariant)
                else AblationVariant(candidate=variant)
            )
            for name, variant in variants.items()
        }

    def run(self, *, include_base: bool = True) -> dict[str, PairedEvaluationResult]:
        results: dict[str, PairedEvaluationResult] = {}
        if include_base:
            results["base"] = evaluate_scenario(self.scenario, ablation="base")
        for name in ABLATION_NAMES:
            variant = self.variants.get(name)
            if variant is None:
                continue
            scenario = replace(
                self.scenario,
                candidate=variant.candidate,
                baseline=variant.baseline or self.scenario.baseline,
            )
            if name == "no_bidding" and scenario.mode == "full_game":
                scenario = _without_bidding(scenario)
            results[name] = evaluate_scenario(scenario, ablation=name)
        return results


def _without_bidding(scenario: EvaluationScenario) -> EvaluationScenario:
    """Convert standard decks to fixed-landlord card-play deals.

    The recorded first bidder becomes landlord, preserving the deck and bottom
    cards while removing only the bidding phase. Weighted bundles used here
    must carry legacy-ruleset-compatible checkpoints; manifest validation will
    reject a standard checkpoint rather than silently crossing rule identity.
    """
    from douzero.env.rules import RuleSet
    from douzero.evaluation.legacy_data_adapter import deal_standard_deck

    converted = []
    for deal in scenario.deals:
        dealt = deal_standard_deck(deal["deck"])
        hands = {
            "0": dealt["landlord"],
            "1": dealt["landlord_up"],
            "2": dealt["landlord_down"],
        }
        order = deal["bidding_order"]
        landlord_seat, down_seat, up_seat = order
        bottom = dealt["three_landlord_cards"]
        converted.append({
            "landlord": sorted(hands[landlord_seat] + bottom),
            "landlord_up": sorted(hands[up_seat]),
            "landlord_down": sorted(hands[down_seat]),
            "three_landlord_cards": list(bottom),
        })
    return replace(
        scenario,
        mode="cardplay_only",
        ruleset=RuleSet.legacy(),
        deals=tuple(converted),
        seat_permutations=default_seat_permutations("cardplay_only"),
        parent_deal_set_id=scenario.deal_set_id,
    )
