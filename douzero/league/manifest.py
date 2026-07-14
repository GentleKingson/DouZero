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

LEAGUE_SCHEMA_VERSION = 1
_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLES = frozenset({"landlord", "landlord_up", "landlord_down"})


@dataclass(frozen=True)
class PolicyEntry:
    """One immutable policy revision available to the league."""

    policy_id: str
    checkpoint_paths_by_role: Mapping[str, str]
    model_version: str
    ruleset_hash: str
    objective: str
    created_step: int
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
        for name in ("model_version", "ruleset_hash", "objective"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
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
        return any(tag in self.tags for tag in ("current", "random", "rule"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "checkpoint_paths_by_role": dict(self.checkpoint_paths_by_role),
            "model_version": self.model_version,
            "ruleset_hash": self.ruleset_hash,
            "objective": self.objective,
            "created_step": self.created_step,
            "rating": self.rating,
            "tags": list(self.tags),
            "eligible_for_training": self.eligible_for_training,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PolicyEntry":
        required = {
            "policy_id", "checkpoint_paths_by_role", "model_version",
            "ruleset_hash", "objective", "created_step", "rating", "tags",
            "eligible_for_training",
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
        for name in ("policy_id", "model_version", "ruleset_hash", "objective"):
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
            objective=raw["objective"],
            created_step=raw["created_step"],
            rating=raw["rating"],
            tags=tuple(raw["tags"]),
            eligible_for_training=raw["eligible_for_training"],
        )


@dataclass(frozen=True)
class LeagueManifest:
    """Full league state, recoverable without scanning checkpoint folders."""

    policies: tuple[PolicyEntry, ...] = ()
    primary_policy_id: str = ""
    generation: int = 0
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
        )

    def without(self, policy_ids: set[str]) -> "LeagueManifest":
        if self.primary_policy_id in policy_ids:
            raise ValueError("cannot remove the primary policy")
        return LeagueManifest(
            policies=tuple(p for p in self.policies if p.policy_id not in policy_ids),
            primary_policy_id=self.primary_policy_id,
            generation=self.generation + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "primary_policy_id": self.primary_policy_id,
            "policies": [policy.to_dict() for policy in self.policies],
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
        required = {"schema_version", "generation", "primary_policy_id", "policies"}
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
        return cls(
            schema_version=raw["schema_version"],
            generation=raw["generation"],
            primary_policy_id=raw["primary_policy_id"],
            policies=tuple(PolicyEntry.from_dict(item) for item in raw["policies"]),
        )
