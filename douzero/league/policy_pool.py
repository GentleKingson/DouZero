"""Deterministic policy sampling and game-boundary policy bundles."""

from __future__ import annotations

import copy
import hashlib
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from douzero.observation.seats import ALL_ROLES

from .manifest import LeagueManifest, PolicyEntry

_IDENTITY_FIELDS = (
    "model_version",
    "feature_schema_hash",
    "model_config_hash",
    "model_config_identity_version",
    "checkpoint_kind",
    "style_layout_hash",
    "strategy_layout_hash",
    "belief_config_hash",
)


@dataclass(frozen=True)
class PolicyLoaderContract:
    """Exact checkpoint identity accepted by one concrete policy loader."""

    loader_name: str
    model_version: str
    feature_schema_hash: str
    model_config_hash: str
    model_config_identity_version: int
    checkpoint_kind: str
    style_layout_hash: str = ""
    strategy_layout_hash: str = ""
    belief_config_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "loader_name",
            "model_version",
            "feature_schema_hash",
            "model_config_hash",
            "checkpoint_kind",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.model_config_identity_version, bool)
            or not isinstance(self.model_config_identity_version, int)
            or self.model_config_identity_version <= 0
        ):
            raise ValueError("model_config_identity_version must be a positive int")

    @classmethod
    def from_policy(
        cls, policy: PolicyEntry, *, loader_name: str
    ) -> "PolicyLoaderContract":
        """Rebuild a contract from trusted checkpoint metadata.

        Runtime code should prefer :meth:`for_v2_runtime`; this constructor is
        for non-V2 loaders after they independently validate their checkpoint.
        """

        return cls(
            loader_name=loader_name,
            **{name: getattr(policy, name) for name in _IDENTITY_FIELDS},
        )

    @classmethod
    def for_v2_runtime(
        cls,
        schema,
        model_config,
        *,
        checkpoint_kind: str,
        loader_name: str = "v2-checkpoint-loader",
        belief_config_hash: str = "",
    ) -> "PolicyLoaderContract":
        """Derive the expected identity from the live V2 schema and config."""

        style_layout_hash = ""
        if bool(getattr(model_config, "style_enabled", False)):
            from douzero.style.features import STYLE_LAYOUT_HASH

            style_layout_hash = STYLE_LAYOUT_HASH
        strategy_layout_hash = ""
        if bool(getattr(model_config, "strategy_features_enabled", False)):
            from douzero.strategy.features import STRATEGY_FEATURE_LAYOUT_HASH

            strategy_layout_hash = STRATEGY_FEATURE_LAYOUT_HASH
        if bool(getattr(model_config, "belief_enabled", False)) and not belief_config_hash:
            raise ValueError(
                "belief_config_hash is required for a belief-enabled V2 runtime"
            )
        return cls(
            loader_name=loader_name,
            model_version="v2",
            feature_schema_hash=schema.stable_hash(),
            model_config_hash=model_config.stable_hash(),
            model_config_identity_version=model_config.IDENTITY_VERSION,
            checkpoint_kind=checkpoint_kind,
            style_layout_hash=style_layout_hash,
            strategy_layout_hash=strategy_layout_hash,
            belief_config_hash=belief_config_hash,
        )

    def mismatches(self, policy: PolicyEntry) -> tuple[str, ...]:
        return tuple(
            f"{name}: policy={getattr(policy, name)!r}, "
            f"loader={getattr(self, name)!r}"
            for name in _IDENTITY_FIELDS
            if getattr(policy, name) != getattr(self, name)
        )

    def identity_mismatches(
        self, other: "PolicyLoaderContract"
    ) -> tuple[str, ...]:
        return tuple(
            f"{name}: checkpoint={getattr(self, name)!r}, "
            f"runtime={getattr(other, name)!r}"
            for name in _IDENTITY_FIELDS
            if getattr(self, name) != getattr(other, name)
        )

    def assert_compatible(self, policy: PolicyEntry) -> None:
        mismatches = self.mismatches(policy)
        if mismatches:
            raise ValueError(
                f"policy {policy.policy_id!r} is incompatible with loader "
                f"{self.loader_name!r}: {'; '.join(mismatches)}"
            )


@dataclass(frozen=True)
class LoadedPolicySelector:
    """An action selector bound to the exact identity it loaded."""

    policy_id: str
    contract: PolicyLoaderContract
    select: Callable[[object], int]
    select_bidding: Callable[[object], int] | None = None

    def __call__(self, observation: object) -> int:
        return int(self.select(observation))

    def bid(self, observation: object) -> int:
        if self.select_bidding is None:
            raise RuntimeError(
                f"policy {self.policy_id!r} has no learned bidding selector"
            )
        return int(self.select_bidding(observation))


def build_frozen_policy_model(
    learner_model,
    state_dict,
    *,
    policy: PolicyEntry,
    loader_contract: PolicyLoaderContract,
    runtime_contract: PolicyLoaderContract,
):
    """Validate identity, then load weights into a detached learner clone."""

    loader_contract.assert_compatible(policy)
    mismatches = loader_contract.identity_mismatches(runtime_contract)
    if mismatches:
        raise ValueError(
            "historical checkpoint identity does not match the learner clone "
            f"runtime: {'; '.join(mismatches)}"
        )
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
        digest = self._current_digest(assignments)
        if self.bundle_hash and self.bundle_hash != digest:
            raise ValueError("policy bundle hash does not match its assignments")
        object.__setattr__(self, "policy_ids_by_seat", MappingProxyType(assignments))
        object.__setattr__(self, "bundle_hash", digest)

    def _current_digest(self, assignments: Mapping[str, str] | None = None) -> str:
        assignments = assignments or self.policy_ids_by_seat
        payload = "|".join(
            [str(self.game_index), self.policy_version]
            + [f"{role}:{assignments[role]}" for role in ALL_ROLES]
            + list(self.learner_controlled_seats)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_unchanged(self, expected_hash: str) -> None:
        if self._current_digest() != expected_hash:
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
        runtime_loader: PolicyLoaderContract,
        runtime_ruleset_hash: str,
        opponent_loaders: Mapping[str, PolicyLoaderContract] | None = None,
        config: PolicyPoolConfig | None = None,
    ) -> None:
        self.config = config or PolicyPoolConfig()
        self.current = current
        self.runtime_loader = runtime_loader
        self.runtime_ruleset_hash = runtime_ruleset_hash
        runtime_loader.assert_compatible(current)
        if current.ruleset_hash != runtime_ruleset_hash:
            raise ValueError("current policy ruleset_hash does not match runtime")
        try:
            registered_current = manifest.get(current.policy_id)
        except KeyError as exc:
            raise ValueError("current policy is not registered in the manifest") from exc
        if registered_current != current:
            raise ValueError("current policy differs from its manifest entry")
        loaders = {runtime_loader.model_version: runtime_loader}
        for model_version, loader in (opponent_loaders or {}).items():
            if model_version != loader.model_version:
                raise ValueError(
                    f"opponent loader key {model_version!r} does not match "
                    f"contract model_version {loader.model_version!r}"
                )
            if model_version == runtime_loader.model_version and loader != runtime_loader:
                raise ValueError("opponent loader cannot replace the runtime loader")
            loaders[model_version] = loader
        self.loaders = MappingProxyType(loaders)
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
            loader = self.loaders.get(policy.model_version)
            if loader is None:
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: no explicit loader "
                    f"registered for model_version {policy.model_version!r}",
                    RuntimeWarning,
                )
                continue
            mismatches = loader.mismatches(policy)
            if mismatches:
                warnings.warn(
                    f"Skipping policy {policy.policy_id!r}: incompatible with "
                    f"loader {loader.loader_name!r}: {'; '.join(mismatches)}",
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
                feature_schema_hash="builtin",
                model_config_hash="builtin-random",
                model_config_identity_version=1,
                checkpoint_kind="builtin",
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
                feature_schema_hash="builtin",
                model_config_hash="builtin-rule",
                model_config_identity_version=1,
                checkpoint_kind="builtin",
                objective=current.objective,
                created_step=0,
                tags=("rule",),
            ))
        self.candidates = tuple(
            {policy.policy_id: policy for policy in candidates}.values()
        )
        if self.config.mode == "population" and not self.candidates:
            raise ValueError("population mode has no compatible opponent policies")

    def policy(self, policy_id: str) -> PolicyEntry:
        if policy_id == self.current.policy_id:
            return self.current
        for policy in self.candidates:
            if policy.policy_id == policy_id:
                return policy
        raise KeyError(policy_id)

    def validate_loaded_selector(self, selector: LoadedPolicySelector) -> None:
        policy = self.policy(selector.policy_id)
        expected = self.loaders.get(policy.model_version)
        if expected is None:
            raise ValueError(
                f"policy {policy.policy_id!r} has no registered checkpoint loader"
            )
        if selector.contract != expected:
            raise ValueError(
                f"selector for policy {policy.policy_id!r} was loaded by "
                f"{selector.contract.loader_name!r}, expected {expected.loader_name!r}"
            )
        expected.assert_compatible(policy)

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
