"""Recoverable, atomically-written policy-league manifest."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

LEAGUE_SCHEMA_VERSION = 2
_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLES = frozenset({"landlord", "landlord_up", "landlord_down"})


@dataclass(frozen=True)
class PolicyEntry:
    """One immutable policy revision available to the league."""

    policy_id: str
    checkpoint_paths_by_role: Mapping[str, str]
    model_version: str
    ruleset_hash: str
    feature_schema_hash: str
    model_config_hash: str
    model_config_identity_version: int
    checkpoint_kind: str
    objective: str
    created_step: int
    style_layout_hash: str = ""
    strategy_layout_hash: str = ""
    belief_config_hash: str = ""
    rating: float = 0.0
    tags: tuple[str, ...] = ()
    eligible_for_training: bool = True

    def __post_init__(self) -> None:
        if not _POLICY_ID_RE.fullmatch(self.policy_id):
            raise ValueError(f"invalid policy_id {self.policy_id!r}")
        paths = {str(role): str(path) for role, path in self.checkpoint_paths_by_role.items()}
        unknown = set(paths) - _ROLES
        if unknown:
            raise ValueError(
                f"policy {self.policy_id!r} has unknown checkpoint roles {sorted(unknown)}"
            )
        if (
            isinstance(self.created_step, bool)
            or not isinstance(self.created_step, int)
            or self.created_step < 0
        ):
            raise ValueError("created_step must be a non-negative int")
        for name in (
            "model_version",
            "ruleset_hash",
            "feature_schema_hash",
            "model_config_hash",
            "checkpoint_kind",
            "objective",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.model_config_identity_version, bool)
            or not isinstance(self.model_config_identity_version, int)
            or self.model_config_identity_version <= 0
        ):
            raise ValueError("model_config_identity_version must be a positive int")
        for name in (
            "style_layout_hash",
            "strategy_layout_hash",
            "belief_config_hash",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be str")
        if isinstance(self.rating, bool) or not isinstance(self.rating, (int, float)):
            raise TypeError("rating must be a number")
        if not math.isfinite(self.rating):
            raise ValueError("rating must be finite")
        if not isinstance(self.eligible_for_training, bool):
            raise TypeError("eligible_for_training must be bool")
        if not all(isinstance(tag, str) and tag for tag in self.tags):
            raise ValueError("tags must contain non-empty strings")
        object.__setattr__(self, "checkpoint_paths_by_role", MappingProxyType(paths))
        object.__setattr__(self, "tags", tuple(dict.fromkeys(self.tags)))

    @property
    def is_builtin(self) -> bool:
        return (
            self.policy_id == "builtin-random"
            and self.model_version == "random"
            and "random" in self.tags
        ) or (
            self.policy_id == "builtin-rule"
            and self.model_version == "rule"
            and "rule" in self.tags
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "checkpoint_paths_by_role": dict(self.checkpoint_paths_by_role),
            "model_version": self.model_version,
            "ruleset_hash": self.ruleset_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "model_config_hash": self.model_config_hash,
            "model_config_identity_version": self.model_config_identity_version,
            "checkpoint_kind": self.checkpoint_kind,
            "objective": self.objective,
            "created_step": self.created_step,
            "style_layout_hash": self.style_layout_hash,
            "strategy_layout_hash": self.strategy_layout_hash,
            "belief_config_hash": self.belief_config_hash,
            "rating": self.rating,
            "tags": list(self.tags),
            "eligible_for_training": self.eligible_for_training,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyEntry":
        required = {
            "policy_id", "checkpoint_paths_by_role", "model_version",
            "ruleset_hash", "feature_schema_hash", "model_config_hash",
            "model_config_identity_version", "checkpoint_kind", "objective",
            "created_step", "style_layout_hash", "strategy_layout_hash",
            "belief_config_hash", "rating", "tags", "eligible_for_training",
        }
        missing = required - set(raw)
        unknown = set(raw) - required
        if missing or unknown:
            raise ValueError(
                f"policy entry fields invalid; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if not isinstance(raw["checkpoint_paths_by_role"], Mapping):
            raise TypeError("checkpoint_paths_by_role must be a mapping")
        if not isinstance(raw["tags"], list):
            raise TypeError("tags must be a JSON list")
        if not isinstance(raw["eligible_for_training"], bool):
            raise TypeError("eligible_for_training must be bool")
        if isinstance(raw["created_step"], bool) or not isinstance(
            raw["created_step"], int
        ):
            raise TypeError("created_step must be int")
        if isinstance(raw["model_config_identity_version"], bool) or not isinstance(
            raw["model_config_identity_version"], int
        ):
            raise TypeError("model_config_identity_version must be int")
        for name in (
            "policy_id", "model_version", "ruleset_hash", "feature_schema_hash",
            "model_config_hash", "checkpoint_kind", "objective",
            "style_layout_hash", "strategy_layout_hash", "belief_config_hash",
        ):
            if not isinstance(raw[name], str):
                raise TypeError(f"{name} must be str")
        if (
            isinstance(raw["rating"], bool)
            or not isinstance(raw["rating"], (int, float))
        ):
            raise TypeError("rating must be a number")
        if not all(isinstance(tag, str) for tag in raw["tags"]):
            raise TypeError("every policy tag must be str")
        return cls(
            policy_id=raw["policy_id"],
            checkpoint_paths_by_role=dict(raw["checkpoint_paths_by_role"]),
            model_version=raw["model_version"],
            ruleset_hash=raw["ruleset_hash"],
            feature_schema_hash=raw["feature_schema_hash"],
            model_config_hash=raw["model_config_hash"],
            model_config_identity_version=raw["model_config_identity_version"],
            checkpoint_kind=raw["checkpoint_kind"],
            objective=raw["objective"],
            created_step=raw["created_step"],
            style_layout_hash=raw["style_layout_hash"],
            strategy_layout_hash=raw["strategy_layout_hash"],
            belief_config_hash=raw["belief_config_hash"],
            rating=raw["rating"],
            tags=tuple(raw["tags"]),
            eligible_for_training=raw["eligible_for_training"],
        )


@dataclass(frozen=True)
class PendingDelete:
    """A removed managed snapshot whose files still require cleanup."""

    policy_id: str
    checkpoint_paths_by_role: Mapping[str, str]

    def __post_init__(self) -> None:
        if not _POLICY_ID_RE.fullmatch(self.policy_id):
            raise ValueError(f"invalid pending-delete policy_id {self.policy_id!r}")
        paths = {
            str(role): str(path)
            for role, path in self.checkpoint_paths_by_role.items()
        }
        if set(paths) != _ROLES:
            raise ValueError("pending delete must contain all three role checkpoints")
        object.__setattr__(self, "checkpoint_paths_by_role", MappingProxyType(paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "checkpoint_paths_by_role": dict(self.checkpoint_paths_by_role),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PendingDelete":
        required = {"policy_id", "checkpoint_paths_by_role"}
        if set(raw) != required:
            raise ValueError(
                f"pending delete fields must be {sorted(required)}, got {sorted(raw)}"
            )
        if not isinstance(raw["policy_id"], str):
            raise TypeError("pending delete policy_id must be str")
        if not isinstance(raw["checkpoint_paths_by_role"], Mapping):
            raise TypeError("pending delete paths must be a mapping")
        return cls(raw["policy_id"], dict(raw["checkpoint_paths_by_role"]))


@dataclass(frozen=True)
class LeagueManifest:
    """Full league state, recoverable without scanning checkpoint folders."""

    policies: tuple[PolicyEntry, ...] = ()
    primary_policy_id: str = ""
    generation: int = 0
    pending_deletes: tuple[PendingDelete, ...] = ()
    schema_version: int = LEAGUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEAGUE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported league schema_version {self.schema_version}; "
                f"expected {LEAGUE_SCHEMA_VERSION}"
            )
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        ids = [policy.policy_id for policy in self.policies]
        if len(ids) != len(set(ids)):
            raise ValueError("league manifest contains duplicate policy_id values")
        if self.primary_policy_id and self.primary_policy_id not in set(ids):
            raise ValueError(
                f"primary_policy_id {self.primary_policy_id!r} is not registered"
            )
        pending_ids = [item.policy_id for item in self.pending_deletes]
        if len(pending_ids) != len(set(pending_ids)):
            raise ValueError("league manifest contains duplicate pending deletes")
        overlap = set(ids) & set(pending_ids)
        if overlap:
            raise ValueError(
                f"active policies cannot also be pending deletion: {sorted(overlap)}"
            )

    def get(self, policy_id: str) -> PolicyEntry:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        raise KeyError(policy_id)

    def upsert(self, policy: PolicyEntry, *, make_primary: bool = False) -> "LeagueManifest":
        policies = [p for p in self.policies if p.policy_id != policy.policy_id]
        policies.append(policy)
        primary = policy.policy_id if make_primary else self.primary_policy_id
        if not primary:
            primary = policy.policy_id
        return LeagueManifest(
            policies=tuple(policies),
            primary_policy_id=primary,
            generation=self.generation + 1,
            pending_deletes=self.pending_deletes,
        )

    def without(self, policy_ids: set[str]) -> "LeagueManifest":
        if self.primary_policy_id in policy_ids:
            raise ValueError("cannot remove the primary policy")
        return LeagueManifest(
            policies=tuple(p for p in self.policies if p.policy_id not in policy_ids),
            primary_policy_id=self.primary_policy_id,
            generation=self.generation + 1,
            pending_deletes=self.pending_deletes,
        )

    def mark_pending_delete(self, policy_ids: set[str]) -> "LeagueManifest":
        if self.primary_policy_id in policy_ids:
            raise ValueError("cannot remove the primary policy")
        removed = [p for p in self.policies if p.policy_id in policy_ids]
        return LeagueManifest(
            policies=tuple(p for p in self.policies if p.policy_id not in policy_ids),
            primary_policy_id=self.primary_policy_id,
            generation=self.generation + 1,
            pending_deletes=self.pending_deletes + tuple(
                PendingDelete(p.policy_id, p.checkpoint_paths_by_role)
                for p in removed
            ),
        )

    def clear_pending_deletes(self) -> "LeagueManifest":
        if not self.pending_deletes:
            return self
        return LeagueManifest(
            policies=self.policies,
            primary_policy_id=self.primary_policy_id,
            generation=self.generation + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "primary_policy_id": self.primary_policy_id,
            "policies": [policy.to_dict() for policy in self.policies],
            "pending_deletes": [item.to_dict() for item in self.pending_deletes],
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically persist the manifest after fsyncing the complete JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LeagueManifest":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        required = {
            "schema_version", "generation", "primary_policy_id", "policies",
            "pending_deletes",
        }
        if set(raw) != required:
            raise ValueError(
                f"league manifest fields must be {sorted(required)}, got {sorted(raw)}"
            )
        if not isinstance(raw["schema_version"], int) or isinstance(
            raw["schema_version"], bool
        ):
            raise TypeError("league schema_version must be int")
        if not isinstance(raw["generation"], int) or isinstance(
            raw["generation"], bool
        ):
            raise TypeError("league generation must be int")
        if not isinstance(raw["primary_policy_id"], str):
            raise TypeError("league primary_policy_id must be str")
        if not isinstance(raw["policies"], list):
            raise TypeError("league policies must be a JSON list")
        if not isinstance(raw["pending_deletes"], list):
            raise TypeError("league pending_deletes must be a JSON list")
        return cls(
            schema_version=raw["schema_version"],
            generation=raw["generation"],
            primary_policy_id=raw["primary_policy_id"],
            policies=tuple(PolicyEntry.from_dict(item) for item in raw["policies"]),
            pending_deletes=tuple(
                PendingDelete.from_dict(item) for item in raw["pending_deletes"]
            ),
        )
