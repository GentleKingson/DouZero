"""Typed, reproducible inputs for the P15 paired evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from douzero.env.rules import RuleSet


EVALUATION_PROTOCOL = "p15_paired_v1"
SCENARIO_MODES = ("cardplay_only", "full_game")
DATASET_SCOPES = ("public", "private_holdout")
BACKENDS = ("random", "rule", "legacy", "legacy_factorized", "v2", "bc")
BIDDING_POLICIES = ("rule", "random", "pass", "max")
SEATS = ("0", "1", "2")
ROLES = ("landlord", "landlord_up", "landlord_down")


@dataclass(frozen=True)
class BundleSpec:
    """One auditable three-role policy bundle.

    Arbitrary ``name`` values let the same loader represent legacy WP/ADP,
    BC, V2 stages, and historical snapshots without teaching the evaluator
    product-specific names. Checkpoint paths are role keyed when required.
    """

    name: str
    backend: str
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    bidding_policy: str = "rule"
    decision_mode: str = "pure_win"
    model_config: Mapping[str, Any] = field(default_factory=dict)
    belief_checkpoint: str = ""
    search_config: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.backend, str):
            raise TypeError("bundle name and backend must be strings")
        if not self.name.strip():
            raise ValueError("bundle name must not be empty")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.bidding_policy not in BIDDING_POLICIES:
            raise ValueError(
                f"bidding_policy must be one of {BIDDING_POLICIES}, "
                f"got {self.bidding_policy!r}"
            )
        for field_name in ("checkpoints", "model_config", "search_config"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"bundle {field_name} must be a mapping")
        if self.backend in ("legacy", "legacy_factorized", "v2", "bc"):
            missing = [role for role in ROLES if not self.checkpoints.get(role)]
            if missing:
                raise ValueError(
                    f"bundle {self.name!r} is missing checkpoints for {missing}"
                )

    def to_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        checkpoints = (
            dict(self.checkpoints)
            if include_paths
            else {role: bool(path) for role, path in self.checkpoints.items()}
        )
        return {
            "name": self.name,
            "backend": self.backend,
            "checkpoints": checkpoints,
            "bidding_policy": self.bidding_policy,
            "decision_mode": self.decision_mode,
            "model_config": dict(self.model_config),
            "belief_checkpoint": bool(self.belief_checkpoint),
            "search_config": dict(self.search_config),
            "tags": list(self.tags),
        }


def default_seat_permutations(mode: str) -> tuple[tuple[str, str, str], ...]:
    """Return the minimum balanced candidate/baseline assignments."""
    if mode == "cardplay_only":
        # Role order is landlord, landlord_up, landlord_down.
        return (
            ("candidate", "baseline", "baseline"),
            ("baseline", "candidate", "candidate"),
        )
    if mode == "full_game":
        # Neutral seat order is 0, 1, 2. Rotate the candidate through all seats.
        return (
            ("candidate", "baseline", "baseline"),
            ("baseline", "candidate", "baseline"),
            ("baseline", "baseline", "candidate"),
        )
    raise ValueError(f"unknown scenario mode {mode!r}")


@dataclass(frozen=True)
class EvaluationScenario:
    """Complete input identity for a paired evaluation run."""

    mode: str
    ruleset: RuleSet
    candidate: BundleSpec
    baseline: BundleSpec
    deals: tuple[dict[str, Any], ...]
    deterministic_seed: int = 0
    seat_permutations: tuple[tuple[str, str, str], ...] = ()
    dataset_scope: str = "public"
    deal_set_name: str = "generated"
    parent_deal_set_id: str = ""
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.ruleset, RuleSet):
            raise TypeError("ruleset must be a RuleSet")
        if not isinstance(self.candidate, BundleSpec) or not isinstance(
            self.baseline, BundleSpec
        ):
            raise TypeError("candidate and baseline must be BundleSpec instances")
        if isinstance(self.deterministic_seed, bool) or not isinstance(
            self.deterministic_seed, int
        ):
            raise TypeError("deterministic_seed must be an integer")
        if self.mode not in SCENARIO_MODES:
            raise ValueError(f"mode must be one of {SCENARIO_MODES}, got {self.mode!r}")
        if self.dataset_scope not in DATASET_SCOPES:
            raise ValueError(
                f"dataset_scope must be one of {DATASET_SCOPES}, "
                f"got {self.dataset_scope!r}"
            )
        object.__setattr__(self, "deals", tuple(self.deals))
        if not self.deals:
            raise ValueError("evaluation requires at least one deal")
        if any(not isinstance(deal, dict) for deal in self.deals):
            raise TypeError("every evaluation deal must be a dict")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        permutations = self.seat_permutations or default_seat_permutations(self.mode)
        for permutation in permutations:
            if len(permutation) != 3:
                raise ValueError("each seat permutation must contain exactly 3 entries")
            if any(label not in ("candidate", "baseline") for label in permutation):
                raise ValueError(
                    "seat permutations may contain only 'candidate' and 'baseline'"
                )
            if "candidate" not in permutation or "baseline" not in permutation:
                raise ValueError("each permutation must compare both bundles")
            candidate_indices = {
                index for index, label in enumerate(permutation)
                if label == "candidate"
            }
            if self.mode == "cardplay_only" and candidate_indices not in (
                {0}, {1, 2}
            ):
                raise ValueError(
                    "cardplay_only candidates must control either the landlord "
                    "or both farmer roles"
                )
            if self.mode == "full_game" and len(candidate_indices) != 1:
                raise ValueError(
                    "full_game permutations must rotate exactly one candidate seat"
                )
        object.__setattr__(self, "seat_permutations", tuple(permutations))
        expected_ruleset = "legacy" if self.mode == "cardplay_only" else "standard"
        if self.ruleset.ruleset_id != expected_ruleset:
            raise ValueError(
                f"{self.mode} requires a {expected_ruleset!r} ruleset, got "
                f"{self.ruleset.ruleset_id!r}"
            )
        if self.mode == "full_game":
            from .legacy_data_adapter import _validate_standard_record

            for index, deal in enumerate(self.deals):
                _validate_standard_record(deal, index, self.ruleset)
        else:
            expected_counts = Counter(list(range(3, 15)) * 4 + [17] * 4 + [20, 30])
            for index, deal in enumerate(self.deals):
                required = {
                    "landlord", "landlord_up", "landlord_down",
                    "three_landlord_cards",
                }
                if set(deal) != required:
                    raise ValueError(
                        f"legacy deal {index} must contain exactly {sorted(required)}"
                    )
                cards = (
                    list(deal["landlord"])
                    + list(deal["landlord_up"])
                    + list(deal["landlord_down"])
                )
                if Counter(cards) != expected_counts:
                    raise ValueError(f"legacy deal {index} is not a valid 54-card deal")
                if not (
                    len(deal["landlord"]) == 20
                    and len(deal["landlord_up"]) == 17
                    and len(deal["landlord_down"]) == 17
                    and len(deal["three_landlord_cards"]) == 3
                ):
                    raise ValueError(f"legacy deal {index} has invalid hand sizes")
                if Counter(deal["three_landlord_cards"]) - Counter(deal["landlord"]):
                    raise ValueError(
                        f"legacy deal {index} bottom cards are not in landlord hand"
                    )

    @property
    def deal_set_id(self) -> str:
        """Content identity; never includes the private holdout path/name."""
        payload = json.dumps(self.deals, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EVALUATION_PROTOCOL,
            "mode": self.mode,
            "ruleset": self.ruleset.identity(),
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "deal_set_id": self.deal_set_id,
            "parent_deal_set_id": self.parent_deal_set_id or None,
            "deal_set_name": (
                self.deal_set_name
                if self.dataset_scope == "public"
                else "private_holdout"
            ),
            "dataset_scope": self.dataset_scope,
            "num_deals": len(self.deals),
            "deterministic_seed": self.deterministic_seed,
            "seat_permutations": [list(p) for p in self.seat_permutations],
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
        }


def bundle_from_dict(data: Mapping[str, Any]) -> BundleSpec:
    """Parse a model-matrix bundle entry with strict unknown-key checking."""
    allowed = {field.name for field in BundleSpec.__dataclass_fields__.values()}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown bundle fields: {sorted(unknown)}")
    converted = dict(data)
    if "tags" in converted:
        converted["tags"] = tuple(converted["tags"])
    return BundleSpec(**converted)
