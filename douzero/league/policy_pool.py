"""Deterministic policy sampling and game-boundary policy bundles."""

from __future__ import annotations

import copy
import hashlib
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from douzero.observation.seats import ALL_ROLES

from .manifest import LeagueManifest, PolicyEntry

_SUPPORTED_OPPONENT_MODEL_VERSIONS = frozenset(
    {"legacy", "factorized", "v2", "bc", "random", "rule"}
)


def build_frozen_policy_model(learner_model, state_dict):
    """Load historical weights into a detached clone, never into the learner."""

    historical = copy.deepcopy(learner_model)
    historical.load_state_dict(state_dict, strict=True)
    historical.eval()
    for parameter in historical.parameters():
        parameter.requires_grad_(False)
    return historical


@dataclass(frozen=True)
class PolicyPoolConfig:
    mode: str = "single"
    seed: int = 0
    learner_seats_per_game: int = 1
    include_random_agent: bool = True
    include_rule_agent: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be int")
        for name in ("include_random_agent", "include_rule_agent"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.mode not in ("single", "population"):
            raise ValueError("mode must be 'single' or 'population'")
        if self.learner_seats_per_game not in (1, 2, 3):
            raise ValueError("learner_seats_per_game must be 1, 2, or 3")


@dataclass(frozen=True)
class PolicyBundle:
    """A complete seat assignment selected once at the start of a game."""

    game_index: int
    policy_ids_by_seat: Mapping[str, str]
    learner_controlled_seats: tuple[str, ...]
    policy_version: str
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.game_index, bool)
            or not isinstance(self.game_index, int)
            or self.game_index < 0
        ):
            raise ValueError("game_index must be a non-negative int")
        assignments = dict(self.policy_ids_by_seat)
        if set(assignments) != set(ALL_ROLES):
            raise ValueError("policy bundle must assign all three seats")
        if not self.learner_controlled_seats:
            raise ValueError("policy bundle must have a learner-controlled seat")
        if len(set(self.learner_controlled_seats)) != len(
            self.learner_controlled_seats
        ):
            raise ValueError("learner-controlled seats must be unique")
        if not set(self.learner_controlled_seats).issubset(assignments):
            raise ValueError("learner-controlled seats must be assigned")
        if not self.policy_version or any(not value for value in assignments.values()):
            raise ValueError("policy IDs and policy_version must be non-empty")
        if any(
            assignments[seat] != self.policy_version
            for seat in self.learner_controlled_seats
        ):
            raise ValueError("every learner-controlled seat must use policy_version")
        payload = "|".join(
            [str(self.game_index), self.policy_version]
            + [f"{role}:{assignments[role]}" for role in ALL_ROLES]
            + list(self.learner_controlled_seats)
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.bundle_hash and self.bundle_hash != digest:
            raise ValueError("policy bundle hash does not match its assignments")
        object.__setattr__(self, "policy_ids_by_seat", MappingProxyType(assignments))
        object.__setattr__(self, "bundle_hash", digest)

    def assert_unchanged(self, expected_hash: str) -> None:
        if self.bundle_hash != expected_hash:
            raise RuntimeError("policy bundle changed during a game")

    def teammate_policy_id(self, learner_seat: str) -> str | None:
        if learner_seat == "landlord":
            return None
        teammate = (
            "landlord_down" if learner_seat == "landlord_up" else "landlord_up"
        )
        return self.policy_ids_by_seat[teammate]


class PolicyPool:
    """Available historical/baseline policies with deterministic sampling."""

    def __init__(
        self,
        manifest: LeagueManifest,
        current: PolicyEntry,
        *,
        runtime_model_version: str,
        runtime_ruleset_hash: str,
        config: PolicyPoolConfig | None = None,
    ) -> None:
        self.config = config or PolicyPoolConfig()
        self.current = current
        self.runtime_model_version = runtime_model_version
        self.runtime_ruleset_hash = runtime_ruleset_hash
        if current.model_version != runtime_model_version:
            raise ValueError(
                f"current policy model_version {current.model_version!r} does not "
                f"match runtime {runtime_model_version!r}"
            )
        if current.ruleset_hash != runtime_ruleset_hash:
            raise ValueError("current policy ruleset_hash does not match runtime")
        candidates: list[PolicyEntry] = []
        for policy in manifest.policies:
            if policy.policy_id == current.policy_id or not policy.eligible_for_training:
                continue
            if policy.ruleset_hash != runtime_ruleset_hash:
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: ruleset_hash mismatch",
                    RuntimeWarning,
                )
                continue
            if policy.model_version not in _SUPPORTED_OPPONENT_MODEL_VERSIONS:
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: unsupported "
                    f"model_version {policy.model_version!r}",
                    RuntimeWarning,
                )
                continue
            if (
                not policy.is_builtin
                and set(policy.checkpoint_paths_by_role) != set(ALL_ROLES)
            ):
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: checkpoint paths "
                    "must cover all three roles",
                    RuntimeWarning,
                )
                continue
            missing = [
                path for path in policy.checkpoint_paths_by_role.values()
                if not path or not Path(path).is_file()
            ]
            if missing and not policy.is_builtin:
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: missing checkpoints "
                    f"{missing}",
                    RuntimeWarning,
                )
                continue
            candidates.append(policy)
        if self.config.include_random_agent:
            candidates.append(PolicyEntry(
                policy_id="builtin-random",
                checkpoint_paths_by_role={},
                model_version="random",
                ruleset_hash=runtime_ruleset_hash,
                objective=current.objective,
                created_step=0,
                tags=("random",),
            ))
        if self.config.include_rule_agent:
            candidates.append(PolicyEntry(
                policy_id="builtin-rule",
                checkpoint_paths_by_role={},
                model_version="rule",
                ruleset_hash=runtime_ruleset_hash,
                objective=current.objective,
                created_step=0,
                tags=("rule",),
            ))
        self.candidates = tuple(
            {policy.policy_id: policy for policy in candidates}.values()
        )
        if self.config.mode == "population" and not self.candidates:
            raise ValueError("population mode has no compatible opponent policies")

    def sample_bundle(self, game_index: int) -> PolicyBundle:
        """Sample a reproducible assignment without mutable global RNG state."""

        if game_index < 0:
            raise ValueError("game_index must be non-negative")
        rng = random.Random(self.config.seed + game_index * 1_000_003)
        if self.config.mode == "single":
            assignments = {role: self.current.policy_id for role in ALL_ROLES}
            learner_seats = tuple(ALL_ROLES)
        else:
            seats = list(ALL_ROLES)
            rng.shuffle(seats)
            learner_seats = tuple(seats[: self.config.learner_seats_per_game])
            assignments = {}
            for role in ALL_ROLES:
                if role in learner_seats:
                    assignments[role] = self.current.policy_id
                else:
                    assignments[role] = rng.choice(self.candidates).policy_id
        return PolicyBundle(
            game_index=game_index,
            policy_ids_by_seat=assignments,
            learner_controlled_seats=learner_seats,
            policy_version=self.current.policy_id,
        )
