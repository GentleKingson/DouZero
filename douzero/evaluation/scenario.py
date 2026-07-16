"""Typed, reproducible inputs for the P15 paired evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from douzero.env.rules import RuleSet

from .checkpoint_inputs import checkpoint_sha256, validate_sha256
from .protocol import (
    EVALUATION_PROTOCOL,
    OFFICIAL_CI_METHOD,
    OFFICIAL_STATISTICAL_UNIT,
    OFFICIAL_PERMUTATIONS,
    OFFICIAL_PERMUTATION_HASHES,
)


SCENARIO_MODES = ("cardplay_only", "full_game")
DATASET_SCOPES = ("public", "private_holdout")
BACKENDS = ("random", "rule", "legacy", "legacy_factorized", "v2", "bc")
BIDDING_POLICIES = ("rule", "random", "pass", "max", "learned")
SEATS = ("0", "1", "2")
ROLES = ("landlord", "landlord_up", "landlord_down")
DEAL_SET_IDENTITY_SCHEMA = "canonical-evaluation-deal-set-v1"


def _canonical_mapping_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_deal_hash(deal: Mapping[str, Any]) -> str:
    """Hash only canonical fields that affect the evaluated game."""

    if deal.get("format_version") == 2:
        payload = {
            "format_version": 2,
            "schema_version": deal["schema_version"],
            "ruleset_id": deal["ruleset_id"],
            "ruleset_version": deal["ruleset_version"],
            "ruleset_hash": deal["ruleset_hash"],
            "deck": list(deal["deck"]),
            "first_bidder": deal["first_bidder"],
            "bidding_order": list(deal["bidding_order"]),
            "bidding_script": None,
        }
    else:
        payload = {
            "landlord": sorted(deal["landlord"]),
            "landlord_up": sorted(deal["landlord_up"]),
            "landlord_down": sorted(deal["landlord_down"]),
            "three_landlord_cards": sorted(deal["three_landlord_cards"]),
        }
    return _canonical_mapping_hash(payload)


def canonical_deal_id(index: int, deal_hash: str) -> str:
    """Return the stable human-readable ID for one ordered deal."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("deal index must be a non-negative integer")
    if (
        not isinstance(deal_hash, str)
        or len(deal_hash) != 64
        or any(char not in "0123456789abcdef" for char in deal_hash)
    ):
        raise ValueError("deal_hash must be a full lowercase SHA-256")
    return f"{index:06d}-{deal_hash[:12]}"


def canonical_deal_set_id(
    mode: str,
    ruleset: RuleSet | Mapping[str, Any],
    deal_hashes: Sequence[str],
    *,
    seat_permutation_hash: str,
) -> str:
    """Hash the ordered real deals and required experiment arrangement."""

    if any(
        not isinstance(deal_hash, str)
        or len(deal_hash) != 64
        or any(char not in "0123456789abcdef" for char in deal_hash)
        for deal_hash in deal_hashes
    ):
        raise ValueError("deal hashes must be full lowercase SHA-256 values")
    if len(set(deal_hashes)) != len(deal_hashes):
        raise ValueError("formal evaluation deal sets must not contain duplicates")
    ruleset_identity = (
        ruleset.identity() if isinstance(ruleset, RuleSet) else dict(ruleset)
    )
    payload = {
        "schema_version": DEAL_SET_IDENTITY_SCHEMA,
        "mode": mode,
        "ruleset": ruleset_identity,
        "deal_hashes": list(deal_hashes),
        "seat_permutation_hash": seat_permutation_hash,
        "permutations_per_deal": len(default_seat_permutations(mode)),
    }
    return _canonical_mapping_hash(payload)


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
    # P17 field appended after every pre-P17 positional field.
    bidding_checkpoint: str = ""
    # File identities are separate from paths so a formal matrix can declare
    # the approved bytes before anything is loaded. They remain appended for
    # compatibility with the historical positional constructor contract.
    checkpoint_sha256: Mapping[str, str] = field(default_factory=dict)
    belief_checkpoint_sha256: str = ""
    bidding_checkpoint_sha256: str = ""
    checkpoint_digests_explicit: bool = field(
        init=False, repr=False, compare=False, default=False
    )

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
        for field_name in (
            "checkpoints",
            "checkpoint_sha256",
            "model_config",
            "search_config",
        ):
            if not isinstance(getattr(self, field_name), Mapping):
                raise TypeError(f"bundle {field_name} must be a mapping")
        if any(
            not isinstance(role, str) or not isinstance(path, str)
            for role, path in self.checkpoints.items()
        ):
            raise TypeError("bundle checkpoints must map role strings to path strings")
        if not isinstance(self.belief_checkpoint, str) or not isinstance(
            self.bidding_checkpoint, str
        ):
            raise TypeError("belief_checkpoint and bidding_checkpoint must be strings")
        if self.backend in ("legacy", "legacy_factorized", "v2", "bc"):
            missing = [role for role in ROLES if not self.checkpoints.get(role)]
            if missing:
                raise ValueError(
                    f"bundle {self.name!r} is missing checkpoints for {missing}"
                )
        if self.bidding_policy == "learned":
            if self.backend not in ("v2", "bc"):
                raise ValueError("learned bidding requires a V2 bundle backend")
            if not self.bidding_checkpoint:
                raise ValueError("learned bidding requires bidding_checkpoint")
        elif self.bidding_checkpoint:
            raise ValueError(
                "bidding_checkpoint is only valid with bidding_policy='learned'"
            )
        if not isinstance(self.belief_checkpoint_sha256, str) or not isinstance(
            self.bidding_checkpoint_sha256, str
        ):
            raise TypeError(
                "belief and bidding checkpoint SHA-256 values must be strings"
            )

        paths = {role: path for role, path in self.checkpoints.items() if path}
        declared_roles = dict(self.checkpoint_sha256)
        supplied_values = [
            *declared_roles.values(),
            self.belief_checkpoint_sha256,
            self.bidding_checkpoint_sha256,
        ]
        any_declared = any(bool(value) for value in supplied_values)
        all_declared = (
            set(declared_roles) == set(paths)
            and all(bool(declared_roles[role]) for role in paths)
            and bool(self.belief_checkpoint)
            == bool(self.belief_checkpoint_sha256)
            and bool(self.bidding_checkpoint)
            == bool(self.bidding_checkpoint_sha256)
        )
        if any_declared and not all_declared:
            raise ValueError(
                "checkpoint SHA-256 declarations must cover every role, belief, "
                "and bidding checkpoint exactly"
            )
        if any_declared:
            normalized_roles = {
                role: validate_sha256(
                    declared_roles[role],
                    label=f"bundle {self.name!r} {role} checkpoint_sha256",
                )
                for role in sorted(paths)
            }
            belief_digest = (
                validate_sha256(
                    self.belief_checkpoint_sha256,
                    label=f"bundle {self.name!r} belief_checkpoint_sha256",
                )
                if self.belief_checkpoint
                else ""
            )
            bidding_digest = (
                validate_sha256(
                    self.bidding_checkpoint_sha256,
                    label=f"bundle {self.name!r} bidding_checkpoint_sha256",
                )
                if self.bidding_checkpoint
                else ""
            )
            explicit = True
        else:
            # Backward-compatible local construction snapshots identities once.
            # Missing files stay unbound and will be rejected if loading is tried.
            def local_digest(path: str) -> str:
                try:
                    return checkpoint_sha256(path)
                except ValueError:
                    return ""

            normalized_roles = {
                role: local_digest(path) for role, path in sorted(paths.items())
            }
            belief_digest = (
                local_digest(self.belief_checkpoint) if self.belief_checkpoint else ""
            )
            bidding_digest = (
                local_digest(self.bidding_checkpoint) if self.bidding_checkpoint else ""
            )
            explicit = (
                not paths
                and not self.belief_checkpoint
                and not self.bidding_checkpoint
            )
        object.__setattr__(
            self, "checkpoints", MappingProxyType(dict(self.checkpoints))
        )
        object.__setattr__(
            self, "checkpoint_sha256", MappingProxyType(normalized_roles)
        )
        object.__setattr__(self, "belief_checkpoint_sha256", belief_digest)
        object.__setattr__(self, "bidding_checkpoint_sha256", bidding_digest)
        object.__setattr__(self, "checkpoint_digests_explicit", explicit)

    def checkpoint_identities(self) -> dict[str, Any]:
        """Return path-free identities for every checkpoint in this bundle."""

        return {
            "scheme": "predeclared-sha256-file-including-manifest-v1",
            "explicitly_predeclared": self.checkpoint_digests_explicit,
            "roles": {
                role: self.checkpoint_sha256.get(role) or None
                for role in sorted(self.checkpoints)
            },
            "belief": self.belief_checkpoint_sha256 or None,
            "bidding": self.bidding_checkpoint_sha256 or None,
        }

    def to_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        checkpoints = (
            dict(self.checkpoints)
            if include_paths
            else {role: bool(path) for role, path in self.checkpoints.items()}
        )
        payload = {
            "name": self.name,
            "backend": self.backend,
            "checkpoints": checkpoints,
            "bidding_policy": self.bidding_policy,
            "decision_mode": self.decision_mode,
            "model_config": dict(self.model_config),
            "belief_checkpoint": (
                self.belief_checkpoint
                if include_paths else bool(self.belief_checkpoint)
            ),
            "bidding_checkpoint": (
                self.bidding_checkpoint
                if include_paths else bool(self.bidding_checkpoint)
            ),
            "search_config": dict(self.search_config),
            "tags": list(self.tags),
        }
        if include_paths:
            payload.update({
                "checkpoint_sha256": dict(self.checkpoint_sha256),
                "belief_checkpoint_sha256": self.belief_checkpoint_sha256,
                "bidding_checkpoint_sha256": self.bidding_checkpoint_sha256,
            })
        if not include_paths:
            payload["checkpoint_identities"] = self.checkpoint_identities()
        return payload


def default_seat_permutations(mode: str) -> tuple[tuple[str, str, str], ...]:
    """Return the minimum balanced candidate/baseline assignments."""
    try:
        return OFFICIAL_PERMUTATIONS[mode]
    except KeyError as exc:
        raise ValueError(f"unknown scenario mode {mode!r}") from exc


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
        permutations = tuple(
            tuple(permutation)
            for permutation in (
                self.seat_permutations or default_seat_permutations(self.mode)
            )
        )
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
        if tuple(permutations) != default_seat_permutations(self.mode):
            raise ValueError(
                f"{EVALUATION_PROTOCOL} requires exactly the official "
                f"{self.mode} seat permutations in canonical order"
            )
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
                if deal.get("bidding_script") is not None:
                    raise ValueError(
                        "formal paired evaluation does not support bidding_script"
                    )
        else:
            from .legacy_data_adapter import _validate_legacy_record

            for index, deal in enumerate(self.deals):
                _validate_legacy_record(deal, index)
        deal_hashes = [canonical_deal_hash(deal) for deal in self.deals]
        if len(set(deal_hashes)) != len(deal_hashes):
            raise ValueError("formal evaluation deals must be unique")

    @property
    def deal_set_id(self) -> str:
        """Order-sensitive identity of the real deals and row arrangement."""
        deal_hashes = [canonical_deal_hash(deal) for deal in self.deals]
        return canonical_deal_set_id(
            self.mode,
            self.ruleset,
            deal_hashes,
            seat_permutation_hash=OFFICIAL_PERMUTATION_HASHES[self.mode],
        )

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
            "seat_permutation_hash": OFFICIAL_PERMUTATION_HASHES[self.mode],
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "statistical_unit": OFFICIAL_STATISTICAL_UNIT,
            "ci_method": OFFICIAL_CI_METHOD,
            "release_protocol_id": EVALUATION_PROTOCOL,
        }


def bundle_from_dict(
    data: Mapping[str, Any], *, require_checkpoint_digests: bool = False
) -> BundleSpec:
    """Parse a model-matrix bundle entry with strict unknown-key checking."""
    allowed = {
        field.name
        for field in BundleSpec.__dataclass_fields__.values()
        if field.init
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown bundle fields: {sorted(unknown)}")
    converted = dict(data)
    if "tags" in converted:
        converted["tags"] = tuple(converted["tags"])
    bundle = BundleSpec(**converted)
    if require_checkpoint_digests and not bundle.checkpoint_digests_explicit:
        raise ValueError(
            f"bundle {bundle.name!r} requires explicit predeclared checkpoint "
            "SHA-256 values"
        )
    return bundle
