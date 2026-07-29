"""H7 bounded single-GPU runtime over the existing async V2 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping

import torch
import numpy as np

from douzero.belief.model import BeliefModel, belief_features_from_probs
from douzero.training.seed_stream import (
    FORMAL_SEED_DERIVATION_V1,
    TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
    derive_formal_stream_seed,
)
from douzero.training.async_single_gpu import (
    AsyncReplayKey,
    AsyncRequestCoordinator,
    PendingRequestScheduler,
    PinnedObservationBatchStager,
    SharedReplaySlots,
    async_actor_main,
)

from .adaptive_dmc import ADMC_DISABLED
from .config import BELIEF_FEEDBACK_NONE
from .integration_config import V3H6ResolvedConfig
from .belief_policy import V3BeliefPolicy
from .replay import V3ReplayTransition
from .support_matrix import (
    RULESET_LEGACY,
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
    validate_capability_support,
)
from .training.h6_learner import V3H6Learner
from .training.belief_config import BELIEF_MODE_AUXILIARY
from .training.h4_learner import (
    V3H4BeliefSample,
    V3H4BeliefSidecar,
    bind_v3_h4_belief_sidecar,
    build_v3_h4_belief_sidecar,
)
from .training.h3_learner import (
    V3H3OracleSidecar,
    bind_v3_h3_oracle_sidecar,
    build_v3_h3_oracle_sidecar,
)
from .training.cooperation import (
    FARMER_ROLES,
    MIXER_PRIVILEGED,
    V3H5AsyncDecisionSidecar,
    V3H5FarmerTrajectory,
    bind_v3_h5_async_decision,
    build_v3_h5_async_decision_sidecar,
)

V3_H7_RUNTIME_VERSION = "v3-hybrid-h7-1c-runtime-v9"
V3_H7_CHECKPOINT_FORMAT = "v3-hybrid-h7-runtime-checkpoint-v6"
V3_H7_REQUEST_PROTOCOL = "v2-shared-slots-v3-dmc-q-v1"
V3_H7_REPLAY_PROTOCOL = "v3-public-selected-action-q-old-v1"
V3_H71A_REQUEST_PROTOCOL = (
    "v2-shared-slots-v3-dmc-q-public-belief-coupled-snapshot-v1"
)
V3_H71A_REPLAY_PROTOCOL = (
    "v3-public-replay-plus-privileged-belief-sidecar-source-fingerprint-v2"
)
V3_H71A_SNAPSHOT_SEMANTICS = (
    "game-boundary-quiescent-public-policy-plus-belief-copy-v1"
)
V3_H71B_REQUEST_PROTOCOL = "v2-shared-slots-v3-dmc-q-oracle-sidecar-v1"
V3_H71B_REPLAY_PROTOCOL = (
    "v3-public-replay-plus-privileged-oracle-sidecar-source-fingerprint-v2"
)
V3_H71C_REQUEST_PROTOCOL = (
    "v2-shared-slots-v3-dmc-q-public-farmer-cooperation-sidecar-v1"
)
V3_H71C_REPLAY_PROTOCOL = (
    "v3-public-replay-plus-episode-atomic-farmer-trajectories-v1"
)
V3_H71AB_REQUEST_PROTOCOL = (
    "v2-shared-slots-v3-dmc-q-belief-oracle-sidecars-v1"
)
V3_H71AB_REPLAY_PROTOCOL = (
    "v3-public-replay-plus-separate-belief-oracle-sidecars-v1"
)


def resolve_v3_h7_seed_contract(
    *,
    formal_training_seeds: tuple[int, ...] | None,
    formal_derivation: str | None,
    requested_environment_seed: int | None,
    requested_action_seed: int | None,
) -> tuple[int, int, str]:
    """Resolve topology seeds without bypassing a frozen formal contract."""

    if formal_training_seeds is None:
        return (
            1 if requested_environment_seed is None else requested_environment_seed,
            2 if requested_action_seed is None else requested_action_seed,
            TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
        )
    if formal_derivation != FORMAL_SEED_DERIVATION_V1:
        raise ValueError("H7 formal config seed derivation is unsupported")
    if not formal_training_seeds:
        raise ValueError("H7 formal config has no training seeds")
    root_seed = (
        formal_training_seeds[0]
        if requested_environment_seed is None
        else requested_environment_seed
    )
    if root_seed not in formal_training_seeds:
        raise ValueError("H7 requested seed is not frozen by the formal config")
    if requested_action_seed is not None:
        raise ValueError("H7 formal action seed is derived and cannot be overridden")
    return root_seed, root_seed, FORMAL_SEED_DERIVATION_V1


def validate_v3_h7_formal_initialization(initialization_kind: str) -> None:
    """Reject formal initialization modes that H7.1a cannot faithfully apply."""

    if initialization_kind != "seeded_fresh":
        raise NotImplementedError(
            "H7.1a formal checkpoint initialization is not implemented"
        )


def _stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H7RuntimeConfig:
    topology: str = TOPOLOGY_ASYNC_SINGLE_GPU
    num_actors: int = 4
    games_per_actor: int = 4
    batch_size: int = 32
    replay_capacity: int = 4096
    max_actions: int = 4096
    target_microbatch: int = 4
    microbatch_delay_ms: float = 2.0
    request_timeout_seconds: float = 30.0
    max_policy_lag: int = 128
    environment_seed: int = 1
    environment_seed_derivation: str = TOPOLOGY_LOCAL_SEED_DERIVATION_V1
    action_seed: int = 2
    epsilon: float = 0.01
    max_steps_per_episode: int = 1000
    snapshot_semantics: str = "game-boundary-quiescent-copy-v1"
    request_protocol: str = V3_H7_REQUEST_PROTOCOL
    replay_protocol: str = V3_H7_REPLAY_PROTOCOL
    belief_runtime_enabled: bool = False
    belief_sidecar_capacity: int = 4096
    oracle_runtime_enabled: bool = False
    oracle_sidecar_capacity: int = 4096
    cooperation_runtime_enabled: bool = False
    cooperation_sidecar_capacity: int = 4096
    cooperation_episode_capacity: int = 1024

    def __post_init__(self) -> None:
        if self.topology not in {
            TOPOLOGY_SINGLE_PROCESS, TOPOLOGY_ASYNC_SINGLE_GPU,
        }:
            raise ValueError("unknown H7 runtime topology")
        for name in (
            "belief_runtime_enabled",
            "oracle_runtime_enabled",
            "cooperation_runtime_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"H7 {name} must be bool")
        positive = (
            "num_actors", "games_per_actor", "batch_size", "replay_capacity",
            "max_actions", "target_microbatch", "max_policy_lag",
            "max_steps_per_episode", "belief_sidecar_capacity",
            "oracle_sidecar_capacity", "cooperation_sidecar_capacity",
            "cooperation_episode_capacity",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"H7 runtime {name} must be a positive int")
        for name in ("environment_seed", "action_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"H7 runtime {name} must be a non-negative int")
        if self.environment_seed_derivation not in {
            TOPOLOGY_LOCAL_SEED_DERIVATION_V1,
            FORMAL_SEED_DERIVATION_V1,
        }:
            raise ValueError("unknown H7 environment seed derivation")
        for name in ("microbatch_delay_ms", "request_timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"H7 runtime {name} must be positive and finite")
        if not math.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("H7 runtime epsilon must be in [0, 1]")
        enabled_sidecars = sum((
            self.belief_runtime_enabled,
            self.oracle_runtime_enabled,
            self.cooperation_runtime_enabled,
        ))
        if enabled_sidecars > 1:
            raise NotImplementedError(
                "combined async H7.1 capability transports are not supported"
            )
        if self.belief_runtime_enabled:
            expected_request = V3_H71A_REQUEST_PROTOCOL
            expected_replay = V3_H71A_REPLAY_PROTOCOL
        elif self.oracle_runtime_enabled:
            expected_request = V3_H71B_REQUEST_PROTOCOL
            expected_replay = V3_H71B_REPLAY_PROTOCOL
        elif self.cooperation_runtime_enabled:
            expected_request = V3_H71C_REQUEST_PROTOCOL
            expected_replay = V3_H71C_REPLAY_PROTOCOL
        else:
            expected_request = V3_H7_REQUEST_PROTOCOL
            expected_replay = V3_H7_REPLAY_PROTOCOL
        expected_snapshot = (
            V3_H71A_SNAPSHOT_SEMANTICS
            if self.belief_runtime_enabled
            else "game-boundary-quiescent-copy-v1"
        )
        if self.request_protocol != expected_request:
            raise ValueError("unknown H7 request protocol for selected capabilities")
        if self.replay_protocol != expected_replay:
            raise ValueError("unknown H7 replay protocol for selected capabilities")
        if self.snapshot_semantics != expected_snapshot:
            raise ValueError("unknown H7 snapshot semantics for selected capabilities")

    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        if not self.belief_runtime_enabled:
            payload["belief_sidecar_capacity"] = None
        if not self.oracle_runtime_enabled:
            payload["oracle_sidecar_capacity"] = None
        if not self.cooperation_runtime_enabled:
            payload["cooperation_sidecar_capacity"] = None
            payload["cooperation_episode_capacity"] = None
        return {"version": V3_H7_RUNTIME_VERSION, **payload}

    def stable_hash(self) -> str:
        return _stable_hash(self.identity())


def _h7_alignment_capacity(config: V3H7RuntimeConfig, sidecar_capacity: int) -> int:
    """Cover both the bounded sidecar queue and every ready replay slot."""

    inference_slots = max(2, config.num_actors * config.games_per_actor)
    replay_slots = max(
        inference_slots * 2,
        min(config.batch_size * 2, 64),
    )
    return max(sidecar_capacity, replay_slots)


def _drain_sidecar_queue(sidecar_queue) -> int:
    if sidecar_queue is None:
        return 0
    drained = 0
    while True:
        try:
            sidecar_queue.get_nowait()
        except queue.Empty:
            return drained
        drained += 1


def _stop_actor_processes(processes, sidecar_queues, timeout_seconds: float) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    alive = list(processes)
    while alive and time.monotonic() < deadline:
        for sidecar_queue in sidecar_queues:
            _drain_sidecar_queue(sidecar_queue)
        for process in alive:
            process.join(min(0.05, max(0.0, deadline - time.monotonic())))
        alive = [process for process in alive if process.is_alive()]
    for process in alive:
        process.terminate()
    for process in alive:
        process.join(1.0)
    return [process.name for process in alive if process.is_alive()]


@dataclass
class V3H7RuntimeStats:
    games_collected: int = 0
    episodes_completed: int = 0
    transitions_collected: int = 0
    decisions_collected: int = 0
    optimizer_steps: int = 0
    learner_cardplay_samples: int = 0
    belief_labels_collected: int = 0
    belief_optimizer_steps: int = 0
    oracle_labels_collected: int = 0
    oracle_optimizer_steps: int = 0
    cooperation_labels_collected: int = 0
    cooperation_episodes_collected: int = 0
    cooperation_optimizer_steps: int = 0
    cooperation_incomplete_episodes: int = 0
    cooperation_oversized_episodes: int = 0
    amp_fallbacks: int = 0
    episodes_per_team: dict[str, int] = field(
        default_factory=lambda: {"landlord": 0, "farmer": 0}
    )


class V3H71ABeliefAlignment:
    """Bound public replay and privileged labels without co-serializing them."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("H7.1a alignment capacity must be positive")
        self.capacity = capacity
        self.public_rows: dict[AsyncReplayKey, V3ReplayTransition] = {}
        self.sidecars: dict[AsyncReplayKey, V3H4BeliefSidecar] = {}
        self._completed = deque(maxlen=capacity)
        self._completed_set: set[AsyncReplayKey] = set()

    @property
    def pending_count(self) -> int:
        return len(self.public_rows) + len(self.sidecars)

    def _check_new(self, key: AsyncReplayKey, store: Mapping) -> None:
        if not isinstance(key, AsyncReplayKey):
            raise TypeError("H7.1a alignment key has an invalid type")
        if key in store or key in self._completed_set:
            raise RuntimeError("duplicate H7.1a alignment key")

    def add_public(
        self, key: AsyncReplayKey, row: V3ReplayTransition
    ) -> None:
        self._check_new(key, self.public_rows)
        if not isinstance(row, V3ReplayTransition):
            raise TypeError("H7.1a public alignment row has an invalid type")
        if len(self.public_rows) >= self.capacity:
            raise RuntimeError("H7.1a belief alignment backlog exceeded capacity")
        self.public_rows[key] = row

    def add_sidecar(
        self, key: AsyncReplayKey, sidecar: V3H4BeliefSidecar
    ) -> None:
        self._check_new(key, self.sidecars)
        if not isinstance(sidecar, V3H4BeliefSidecar):
            raise TypeError("H7.1a sidecar alignment row has an invalid type")
        if len(self.sidecars) >= self.capacity:
            raise RuntimeError("H7.1a belief alignment backlog exceeded capacity")
        self.sidecars[key] = sidecar

    def add_pair(
        self,
        key: AsyncReplayKey,
        row: V3ReplayTransition,
        sidecar: V3H4BeliefSidecar,
    ) -> tuple[V3ReplayTransition, V3H4BeliefSample]:
        self._check_new(key, self.public_rows)
        self._check_new(key, self.sidecars)
        if not isinstance(row, V3ReplayTransition):
            raise TypeError("H7.1a public alignment row has an invalid type")
        if not isinstance(sidecar, V3H4BeliefSidecar):
            raise TypeError("H7.1a sidecar alignment row has an invalid type")
        sample = bind_v3_h4_belief_sidecar(row.model_inputs, sidecar)
        self._remember_completed(key)
        return row, sample

    def _remember_completed(self, key: AsyncReplayKey) -> None:
        if len(self._completed) == self.capacity:
            expired = self._completed.popleft()
            self._completed_set.remove(expired)
        self._completed.append(key)
        self._completed_set.add(key)

    def pop_ready(
        self,
    ) -> list[tuple[V3ReplayTransition, V3H4BeliefSample]]:
        result = []
        for key in tuple(self.public_rows):
            sidecar = self.sidecars.get(key)
            if sidecar is None:
                continue
            row = self.public_rows.pop(key)
            self.sidecars.pop(key)
            sample = bind_v3_h4_belief_sidecar(row.model_inputs, sidecar)
            self._remember_completed(key)
            result.append((row, sample))
        return result

    def assert_quiescent(self) -> None:
        if self.pending_count:
            raise RuntimeError(
                "H7.1a cannot quiesce with unmatched belief sidecars"
            )


class V3H71BOracleAlignment:
    """Bind finalized public replay to learner-only privileged Oracle data."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("H7.1b alignment capacity must be positive")
        self.capacity = capacity
        self.public_rows: dict[AsyncReplayKey, V3ReplayTransition] = {}
        self.sidecars: dict[AsyncReplayKey, V3H3OracleSidecar] = {}
        self._completed = deque(maxlen=capacity)
        self._completed_set: set[AsyncReplayKey] = set()

    @property
    def pending_count(self) -> int:
        return len(self.public_rows) + len(self.sidecars)

    def _check_new(self, key: AsyncReplayKey, store: Mapping) -> None:
        if not isinstance(key, AsyncReplayKey):
            raise TypeError("H7.1b alignment key has an invalid type")
        if key in store or key in self._completed_set:
            raise RuntimeError("duplicate H7.1b alignment key")

    def add_public(self, key: AsyncReplayKey, row: V3ReplayTransition) -> None:
        self._check_new(key, self.public_rows)
        if not isinstance(row, V3ReplayTransition):
            raise TypeError("H7.1b public alignment row has an invalid type")
        if len(self.public_rows) >= self.capacity:
            raise RuntimeError("H7.1b Oracle alignment backlog exceeded capacity")
        self.public_rows[key] = row

    def add_sidecar(self, key: AsyncReplayKey, sidecar: V3H3OracleSidecar) -> None:
        self._check_new(key, self.sidecars)
        if not isinstance(sidecar, V3H3OracleSidecar):
            raise TypeError("H7.1b sidecar alignment row has an invalid type")
        if len(self.sidecars) >= self.capacity:
            raise RuntimeError("H7.1b Oracle alignment backlog exceeded capacity")
        self.sidecars[key] = sidecar

    def add_pair(self, key, row, sidecar):
        self._check_new(key, self.public_rows)
        self._check_new(key, self.sidecars)
        sample = bind_v3_h3_oracle_sidecar(row, sidecar)
        self._remember_completed(key)
        return row, sample

    def _remember_completed(self, key: AsyncReplayKey) -> None:
        if len(self._completed) == self.capacity:
            expired = self._completed.popleft()
            self._completed_set.remove(expired)
        self._completed.append(key)
        self._completed_set.add(key)

    def pop_ready(self):
        result = []
        for key in tuple(self.public_rows):
            sidecar = self.sidecars.get(key)
            if sidecar is None:
                continue
            row = self.public_rows.pop(key)
            self.sidecars.pop(key)
            result.append((row, bind_v3_h3_oracle_sidecar(row, sidecar)))
            self._remember_completed(key)
        return result

    def assert_quiescent(self) -> None:
        if self.pending_count:
            raise RuntimeError("H7.1b cannot quiesce with unmatched Oracle sidecars")


@dataclass(frozen=True)
class V3H71CCooperationEpisode:
    """One complete, episode-atomic farmer training unit."""

    transitions: tuple[V3ReplayTransition, ...]
    trajectories: tuple[V3H5FarmerTrajectory, V3H5FarmerTrajectory]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("H7.1c cooperation episode cannot be empty")
        if len(self.trajectories) != 2:
            raise ValueError("H7.1c cooperation episode requires two farmers")
        trajectory_rows = tuple(
            row for trajectory in self.trajectories for row in trajectory.transitions
        )
        if self.transitions != trajectory_rows:
            raise ValueError("H7.1c episode rows and trajectories disagree")


class V3H71CCooperationAlignment:
    """Aggregate unequal farmer decisions into complete learner-side pairs."""

    def __init__(self, capacity: int, max_episode_transitions: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("H7.1c alignment capacity must be positive")
        if (
            isinstance(max_episode_transitions, bool)
            or not isinstance(max_episode_transitions, int)
            or max_episode_transitions < 1
        ):
            raise ValueError("H7.1c episode limit must be positive")
        self.capacity = capacity
        self.max_episode_transitions = max_episode_transitions
        self.public_rows: dict[AsyncReplayKey, V3ReplayTransition] = {}
        self.sidecars: dict[AsyncReplayKey, V3H5AsyncDecisionSidecar] = {}
        self._episode_decisions: dict[
            tuple[int, int],
            dict[str, list[tuple[object, V3H5AsyncDecisionSidecar]]],
        ] = {}
        self._expected: dict[tuple[int, int], dict[str, int]] = {}
        self._ready: deque[V3H71CCooperationEpisode] = deque()
        self._completed = deque(maxlen=capacity)
        self._completed_set: set[AsyncReplayKey] = set()
        self.incomplete_episodes = 0
        self.oversized_episodes = 0

    @property
    def pending_count(self) -> int:
        staged = sum(
            len(items)
            for roles in self._episode_decisions.values()
            for items in roles.values()
        )
        return (
            len(self.public_rows)
            + len(self.sidecars)
            + staged
            + len(self._expected)
        )

    def _check_new(self, key: AsyncReplayKey, store: Mapping) -> None:
        if not isinstance(key, AsyncReplayKey):
            raise TypeError("H7.1c alignment key has an invalid type")
        if key in store or key in self._completed_set:
            raise RuntimeError("duplicate H7.1c alignment key")

    def add_public(self, key: AsyncReplayKey, row: V3ReplayTransition) -> None:
        self._check_new(key, self.public_rows)
        if not isinstance(row, V3ReplayTransition) or row.role not in FARMER_ROLES:
            raise TypeError("H7.1c public alignment row must be a farmer replay row")
        if len(self.public_rows) >= self.capacity:
            raise RuntimeError("H7.1c public alignment backlog exceeded capacity")
        self.public_rows[key] = row

    def add_sidecar(
        self, key: AsyncReplayKey, sidecar: V3H5AsyncDecisionSidecar
    ) -> None:
        self._check_new(key, self.sidecars)
        if not isinstance(sidecar, V3H5AsyncDecisionSidecar):
            raise TypeError("H7.1c sidecar alignment row has an invalid type")
        if len(self.sidecars) >= self.capacity:
            raise RuntimeError("H7.1c sidecar alignment backlog exceeded capacity")
        self.sidecars[key] = sidecar

    def _remember_completed(self, key: AsyncReplayKey) -> None:
        if len(self._completed) == self.capacity:
            expired = self._completed.popleft()
            self._completed_set.remove(expired)
        self._completed.append(key)
        self._completed_set.add(key)

    def _stage(
        self,
        key: AsyncReplayKey,
        row: V3ReplayTransition,
        sidecar: V3H5AsyncDecisionSidecar,
    ) -> None:
        if key.trace_index != sidecar.trace_index:
            raise ValueError("H7.1c key and sidecar trace index differ")
        expected_episode = f"actor-{key.actor_id}-episode-{key.episode_id}"
        expected_deal = f"async-deal-{key.episode_id}"
        if row.episode_id != expected_episode or row.deal_id != expected_deal:
            raise ValueError("H7.1c replay episode identity mismatch")
        decision = bind_v3_h5_async_decision(row, sidecar)
        identity = (key.actor_id, key.episode_id)
        staged = sum(
            len(items)
            for episode_roles in self._episode_decisions.values()
            for items in episode_roles.values()
        )
        if staged >= self.capacity:
            raise RuntimeError(
                "H7.1c episode assembly backlog exceeded capacity"
            )
        roles = self._episode_decisions.setdefault(
            identity, {role: [] for role in FARMER_ROLES}
        )
        roles[row.role].append((decision, sidecar))
        self._remember_completed(key)
        self._finalize_ready(identity)

    def add_pair(self, key, row, sidecar) -> int:
        self._check_new(key, self.public_rows)
        self._check_new(key, self.sidecars)
        self._stage(key, row, sidecar)
        return 1

    def pop_ready_pairs(self) -> int:
        completed = 0
        for key in tuple(self.public_rows):
            sidecar = self.sidecars.get(key)
            if sidecar is None:
                continue
            row = self.public_rows.pop(key)
            self.sidecars.pop(key)
            self._stage(key, row, sidecar)
            completed += 1
        return completed

    def mark_episode_complete(
        self,
        actor_id: int,
        episode_id: int,
        farmer_counts: Mapping[str, object],
    ) -> None:
        identity = (actor_id, episode_id)
        if identity in self._expected:
            raise RuntimeError("duplicate H7.1c episode completion")
        if not isinstance(farmer_counts, Mapping) or set(farmer_counts) != set(
            FARMER_ROLES
        ):
            raise ValueError("H7.1c farmer count fields mismatch")
        normalized = {}
        for role in FARMER_ROLES:
            value = farmer_counts[role]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("H7.1c farmer decision count is invalid")
            normalized[role] = value
        self._expected[identity] = normalized
        self._episode_decisions.setdefault(
            identity, {role: [] for role in FARMER_ROLES}
        )
        self._finalize_ready(identity)

    def _finalize_ready(self, identity: tuple[int, int]) -> None:
        expected = self._expected.get(identity)
        roles = self._episode_decisions.get(identity)
        if expected is None or roles is None:
            return
        actual = {role: len(roles[role]) for role in FARMER_ROLES}
        if any(actual[role] > expected[role] for role in FARMER_ROLES):
            raise RuntimeError("H7.1c received excess farmer decisions")
        if actual != expected:
            return
        self._expected.pop(identity)
        self._episode_decisions.pop(identity)
        if any(actual[role] == 0 for role in FARMER_ROLES):
            self.incomplete_episodes += 1
            return
        trajectories = []
        for role in FARMER_ROLES:
            entries = sorted(
                roles[role], key=lambda item: item[0].trace_index
            )
            sidecars = [item[1] for item in entries]
            policy_ids = {item.policy_id for item in sidecars}
            teammate_ids = {item.teammate_policy_id for item in sidecars}
            if len(policy_ids) != 1 or len(teammate_ids) != 1:
                raise ValueError("H7.1c policy provenance changed within one game")
            decisions = tuple(item[0] for item in entries)
            first = decisions[0].transition
            trajectories.append(V3H5FarmerTrajectory(
                episode_id=first.episode_id,
                deal_id=first.deal_id,
                role=role,
                policy_id=next(iter(policy_ids)),
                teammate_policy_id=next(iter(teammate_ids)),
                decisions=decisions,
                team_return=first.mc_return,
            ))
        up, down = trajectories
        if up.teammate_policy_id != down.policy_id:
            raise ValueError("H7.1c landlord-up teammate provenance mismatch")
        if down.teammate_policy_id != up.policy_id:
            raise ValueError("H7.1c landlord-down teammate provenance mismatch")
        transitions = tuple(
            row for trajectory in trajectories for row in trajectory.transitions
        )
        if len(transitions) > self.max_episode_transitions:
            self.oversized_episodes += 1
            return
        self._ready.append(V3H71CCooperationEpisode(
            transitions=transitions,
            trajectories=(up, down),
        ))

    def pop_ready_episodes(self) -> list[V3H71CCooperationEpisode]:
        result = list(self._ready)
        self._ready.clear()
        return result

    def discard_pending(self) -> int:
        identities = {
            *self._episode_decisions,
            *self._expected,
            *((key.actor_id, key.episode_id) for key in self.public_rows),
            *((key.actor_id, key.episode_id) for key in self.sidecars),
        }
        self.public_rows.clear()
        self.sidecars.clear()
        self._episode_decisions.clear()
        self._expected.clear()
        self._ready.clear()
        return len(identities)

    def assert_quiescent(self) -> None:
        if self.pending_count:
            raise RuntimeError(
                "H7.1c cannot quiesce with unmatched farmer trajectories"
            )


def validate_v3_h7_runtime_config(
    resolved_config: V3H6ResolvedConfig,
    runtime_config: V3H7RuntimeConfig,
) -> None:
    """Validate H7 support before model construction, CUDA, or worker startup."""
    topology = resolved_config.learner.topology
    features = resolved_config.learner.features
    if topology.topology != TOPOLOGY_SINGLE_PROCESS:
        raise ValueError(
            "H7 runtime requires a validated single_process learner; "
            "the outer runtime owns async topology identity"
        )
    if topology.ruleset != RULESET_LEGACY:
        raise NotImplementedError(
            "H7 async runtime currently supports legacy card-play rules only"
        )
    enabled = set(features.enabled_capabilities())
    unsupported = enabled - {
        "role_model", "adaptive_dmc", "belief", "oracle", "cooperation",
        "public_export",
    }
    if unsupported:
        raise NotImplementedError(
            "H7 async runtime rejects unsupported capabilities before worker "
            f"startup: {sorted(unsupported)}"
        )
    belief_enabled = "belief" in enabled
    oracle_enabled = "oracle" in enabled
    cooperation_enabled = "cooperation" in enabled
    if sum((belief_enabled, oracle_enabled, cooperation_enabled)) > 1:
        raise NotImplementedError(
            "H7.1 capability transports remain isolated until a later integration"
        )
    if belief_enabled != runtime_config.belief_runtime_enabled:
        raise ValueError(
            "H7 runtime belief feature and belief runtime transport disagree"
        )
    if belief_enabled:
        belief = resolved_config.learner.base.base.belief
        if belief.mode != BELIEF_MODE_AUXILIARY:
            raise NotImplementedError(
                "H7.1a belief runtime currently supports auxiliary phase only"
            )
        if resolved_config.model.belief_feedback == BELIEF_FEEDBACK_NONE:
            raise ValueError("H7.1a belief runtime requires public belief feedback")
        if belief.shared_encoder_updates:
            raise NotImplementedError(
                "H7.1a async belief rejects shared encoder updates"
            )
        if runtime_config.belief_sidecar_capacity < runtime_config.batch_size:
            raise ValueError(
                "H7.1a belief sidecar capacity cannot be smaller than batch_size"
            )
    if oracle_enabled != runtime_config.oracle_runtime_enabled:
        raise ValueError(
            "H7 runtime Oracle feature and Oracle runtime transport disagree"
        )
    if oracle_enabled:
        if runtime_config.oracle_sidecar_capacity < runtime_config.batch_size:
            raise ValueError(
                "H7.1b Oracle sidecar capacity cannot be smaller than batch_size"
            )
        if not resolved_config.learner.base.base.base.schedule.enabled:
            raise ValueError("H7.1b Oracle runtime requires an enabled H3 schedule")
    if cooperation_enabled != runtime_config.cooperation_runtime_enabled:
        raise ValueError(
            "H7 runtime cooperation feature and transport disagree"
        )
    if cooperation_enabled:
        cooperation = resolved_config.learner.base.cooperation
        if cooperation.mixer_mode == MIXER_PRIVILEGED:
            raise NotImplementedError(
                "H7.1c privileged mixer state transport is not implemented"
            )
        if (
            runtime_config.cooperation_sidecar_capacity
            < runtime_config.batch_size
        ):
            raise ValueError(
                "H7.1c cooperation sidecar capacity cannot be smaller than batch_size"
            )
    for capability in enabled:
        validate_capability_support(
            capability,
            topology=runtime_config.topology,
            ruleset=topology.ruleset,
            checkpoint_resume=topology.checkpoint_resume,
            export=topology.export,
            deployment=topology.deployment,
            search=False,
        )
    if (
        not features.adaptive_dmc
        and not belief_enabled
        and not oracle_enabled
        and not cooperation_enabled
    ):
        raise ValueError("H7 async replay requires Adaptive DMC q_old provenance")
    if (
        resolved_config.learner.base.base.base.public.adaptive_dmc.mode
        == ADMC_DISABLED
        and not belief_enabled
        and not oracle_enabled
        and not cooperation_enabled
    ):
        raise ValueError("H7 async runtime cannot use disabled Adaptive DMC")
    learner_batch_size = resolved_config.learner.base.base.base.public.batch_size
    if runtime_config.batch_size > learner_batch_size:
        raise ValueError(
            "H7 runtime batch_size cannot exceed the learner batch_size"
        )


class V3AsyncSingleGPUTrainer:
    """V3 async trainer for base ADMC and isolated H7.1 capability transports."""

    def __init__(
        self,
        learner: V3H6Learner,
        resolved_config: V3H6ResolvedConfig,
        runtime_config: V3H7RuntimeConfig,
    ) -> None:
        if not isinstance(learner, V3H6Learner):
            raise TypeError("H7 runtime requires V3H6Learner")
        if learner.config != resolved_config:
            raise ValueError("H7 learner and resolved config disagree")
        validate_v3_h7_runtime_config(resolved_config, runtime_config)
        if (
            type(self) is V3AsyncSingleGPUTrainer
            and runtime_config.topology != TOPOLOGY_ASYNC_SINGLE_GPU
        ):
            raise ValueError("H7 async trainer requires async_single_gpu topology")
        if learner.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("H7 async runtime requires CUDA and never falls back")
        self.learner = learner
        self.model = learner.model
        self.resolved_config = resolved_config
        self.config = runtime_config
        self.device = learner.device
        self.stats = V3H7RuntimeStats()
        self.buffer: deque[V3ReplayTransition] = deque(
            maxlen=runtime_config.replay_capacity
        )
        self.belief_buffer: deque[V3H4BeliefSample] | None = (
            deque(maxlen=runtime_config.replay_capacity)
            if runtime_config.belief_runtime_enabled
            else None
        )
        self.oracle_buffer: deque | None = (
            deque(maxlen=runtime_config.replay_capacity)
            if runtime_config.oracle_runtime_enabled
            else None
        )
        self.cooperation_buffer: deque[V3H71CCooperationEpisode] | None = (
            deque(maxlen=runtime_config.cooperation_episode_capacity)
            if runtime_config.cooperation_runtime_enabled
            else None
        )
        self._belief_alignment = (
            V3H71ABeliefAlignment(
                _h7_alignment_capacity(
                    runtime_config, runtime_config.belief_sidecar_capacity
                )
            )
            if runtime_config.belief_runtime_enabled
            else None
        )
        self._oracle_alignment = (
            V3H71BOracleAlignment(
                _h7_alignment_capacity(
                    runtime_config, runtime_config.oracle_sidecar_capacity
                )
            )
            if runtime_config.oracle_runtime_enabled
            else None
        )
        self._cooperation_alignment = (
            V3H71CCooperationAlignment(
                _h7_alignment_capacity(
                    runtime_config, runtime_config.cooperation_sidecar_capacity
                ),
                runtime_config.batch_size,
            )
            if runtime_config.cooperation_runtime_enabled
            else None
        )
        self._rng = random.Random(runtime_config.action_seed)
        self._runtime_started = False
        self._served_version_offset = 0
        self._snapshot_step = self.policy_step
        self._reset_metrics()
        self.inference_model = copy.deepcopy(self.model).to(self.device).eval()
        self.belief_model: BeliefModel | None = self.learner.base.base.belief_model
        if runtime_config.belief_runtime_enabled and self.belief_model is None:
            raise TypeError("H7.1a runtime requires the H4 BeliefModel")
        if not runtime_config.belief_runtime_enabled and self.belief_model is not None:
            raise ValueError("H7 base runtime rejects an unserved belief model")
        self.inference_belief_model = (
            None
            if self.belief_model is None
            else copy.deepcopy(self.belief_model).to(self.device).eval()
        )
        self.inference_policy = (
            None
            if self.inference_belief_model is None
            else V3BeliefPolicy(
                self.inference_model,
                self.inference_belief_model,
                ruleset=self.learner.ruleset,
            ).eval()
        )
        self.runtime_identity = {
            "runtime": runtime_config.identity(),
            "runtime_hash": runtime_config.stable_hash(),
            "training_hash": learner.compatibility_hash,
            "model_hash": self.model.config.stable_hash(),
            "belief_model_hash": (
                None
                if self.belief_model is None
                else self.belief_model.config.stable_hash()
            ),
            "ruleset": learner.ruleset.identity(),
            "belief_sidecar": (
                "disabled"
                if self.belief_model is None
                else "separate-privileged-keyed-sidecar-never-public-replay-v1"
            ),
            "oracle_sidecar": (
                "disabled"
                if self.oracle_buffer is None
                else "separate-privileged-keyed-sidecar-learner-only-v1"
            ),
            "cooperation_sidecar": (
                "disabled"
                if self.cooperation_buffer is None
                else (
                    "public-decision-sidecar-episode-atomic-learner-mixer-"
                    "never-public-replay-v1"
                )
            ),
        }
        self.runtime_hash = _stable_hash(self.runtime_identity)

    @property
    def policy_step(self) -> int:
        return int(self.learner.policy_version) + self._served_version_offset

    @property
    def policy_version(self) -> str:
        return f"v3_hybrid:{self.model.config.stable_hash()[:16]}"

    def _reset_metrics(self) -> None:
        self._segments = {
            name: 0.0 for name in (
                "claim_wait", "slot_read", "collate", "h2d", "forward",
                "belief_forward", "d2h", "publish", "replay_drain",
                "learner", "data_wait",
            )
        }
        self._requests = 0
        self._actions = 0
        self._microbatches = 0
        self._batch_histogram: dict[str, int] = {}
        self._bucket_histogram: dict[str, int] = {}
        self._queue_latencies_ms: list[float] = []
        self._actor_blocked_seconds = 0.0
        self._actor_wall_seconds = 0.0

    @staticmethod
    def _increment(histogram: dict[str, int], value: object) -> None:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1

    def _start_runtime(self) -> None:
        import multiprocessing as mp
        from douzero.training.decision_policy import DecisionConfig

        context = mp.get_context("spawn")
        cfg = self.config
        self._tasks = context.Queue()
        self._events = context.Queue()
        self._policy_step = context.Value("q", self.policy_step, lock=True)
        slots = max(2, cfg.num_actors * cfg.games_per_actor)
        self._coordinator = AsyncRequestCoordinator(
            self.model.schema,
            num_slots=slots,
            max_actions=cfg.max_actions,
            output_width=6,
            request_timeout_seconds=cfg.request_timeout_seconds,
            belief_inputs=cfg.belief_runtime_enabled,
        )
        self._replay_slots = SharedReplaySlots(
            self.model.schema,
            num_slots=max(slots * 2, min(cfg.batch_size * 2, 64)),
            max_actions=cfg.max_actions,
            v3_provenance=True,
        )
        self._scheduler = PendingRequestScheduler(
            max_batch_size=slots,
            target_batch_size=cfg.target_microbatch,
            max_delay_seconds=cfg.microbatch_delay_ms / 1000.0,
        )
        self._belief_sidecar_queue = (
            context.Queue(maxsize=cfg.belief_sidecar_capacity)
            if cfg.belief_runtime_enabled
            else None
        )
        self._oracle_sidecar_queue = (
            context.Queue(maxsize=cfg.oracle_sidecar_capacity)
            if cfg.oracle_runtime_enabled
            else None
        )
        self._cooperation_sidecar_queue = (
            context.Queue(maxsize=cfg.cooperation_sidecar_capacity)
            if cfg.cooperation_runtime_enabled
            else None
        )
        self._stagers: dict[int, PinnedObservationBatchStager] = {}
        self._workers = []
        for actor_id in range(cfg.num_actors):
            process = context.Process(
                target=async_actor_main,
                args=(
                    actor_id, self._tasks, self._events,
                    self._coordinator, self._replay_slots,
                ),
                kwargs={
                    "environment_seed": cfg.environment_seed,
                    "environment_seed_derivation": (
                        cfg.environment_seed_derivation
                    ),
                    "action_rng_seed": cfg.action_seed,
                    "epsilon": cfg.epsilon,
                    "max_steps": cfg.max_steps_per_episode,
                    "decision_config": DecisionConfig(),
                    "ruleset": None,
                    "feature_schema_hash": self.model.schema.stable_hash(),
                    "policy_version": self.policy_version,
                    "policy_step": self._policy_step,
                    "games_per_actor": cfg.games_per_actor,
                    "runtime_kind": "v3_hybrid",
                    "belief_sidecar_queue": self._belief_sidecar_queue,
                    "oracle_sidecar_queue": self._oracle_sidecar_queue,
                    "cooperation_sidecar_queue": self._cooperation_sidecar_queue,
                },
                name=f"douzero-v3-actor-{actor_id}",
            )
            process.start()
            self._workers.append(process)
        self._runtime_started = True

    def _service_requests(self, wait_seconds: float = 0.001) -> int:
        started = time.perf_counter()
        requests = self._coordinator.claim_ready(
            max_items=self.config.num_actors * self.config.games_per_actor,
            wait_seconds=wait_seconds,
        )
        self._segments["claim_wait"] += time.perf_counter() - started
        for request in requests:
            if request.policy_snapshot != self._snapshot_step:
                raise RuntimeError("H7 request references an unpublished snapshot")
        self._scheduler.add(requests)
        scheduled = self._scheduler.pop_ready()
        if scheduled is None:
            return 0
        (_snapshot, bucket), group = scheduled
        capacity = (
            int(bucket)
            if isinstance(bucket, int)
            else min(
                self.config.max_actions,
                1 << (max(row.action_count for row in group) - 1).bit_length(),
            )
        )
        stager = self._stagers.get(capacity)
        if stager is None:
            stager = PinnedObservationBatchStager(
                self._coordinator.slots,
                max_batch_size=self.config.num_actors * self.config.games_per_actor,
                action_capacity=capacity,
            )
            self._stagers[capacity] = stager
        started = time.perf_counter()
        size = stager.gather_slots([row.slot_id for row in group])
        self._segments["slot_read"] += time.perf_counter() - started
        started = time.perf_counter()
        batch = stager.batch_view(size, self.model.schema.stable_hash())
        self._segments["collate"] += time.perf_counter() - started
        started = time.perf_counter()
        batch.to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)
        self._segments["h2d"] += time.perf_counter() - started
        belief_features = None
        if self.inference_belief_model is not None:
            public_belief_inputs = [
                self._coordinator.belief_inputs.read(request.slot_id)
                for request in group
            ]
            started = time.perf_counter()
            with torch.inference_mode():
                shared_context = None
                if self.inference_belief_model.config.shared_context_dim:
                    shared_context = (
                        self.inference_model.encode_input_batch_context(batch)
                    )
                belief_output = self.inference_belief_model(
                    public_belief_inputs,
                    shared_context=shared_context,
                )
                belief_numpy = belief_features_from_probs(
                    belief_output.constrained_probs,
                    belief_output.opponent_a_total,
                    np.stack([
                        item.unseen_counts for item in public_belief_inputs
                    ]),
                )
                belief_features = torch.from_numpy(belief_numpy).to(
                    device=self.device,
                    dtype=next(self.inference_model.parameters()).dtype,
                    non_blocking=True,
                )
            torch.cuda.synchronize(self.device)
            self._segments["belief_forward"] += (
                time.perf_counter() - started
            )
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.inference_model.forward_batched(
                batch.state_card_vectors,
                batch.state_context_flat,
                batch.context_card_vectors,
                batch.context_flat,
                batch.history_tokens,
                batch.history_key_padding_mask,
                batch.action_features,
                batch.action_mask,
                batch.acting_role,
                belief_features=belief_features,
            )
            packed = torch.stack((
                output.win_logit.squeeze(-1),
                output.score_if_win.squeeze(-1),
                output.score_if_loss.squeeze(-1),
                output.p_win.squeeze(-1),
                output.score_mean.squeeze(-1),
                output.dmc_q.squeeze(-1),
            ), dim=-1).float().contiguous()
        torch.cuda.synchronize(self.device)
        self._segments["forward"] += time.perf_counter() - started
        started = time.perf_counter()
        packed_cpu = stager.stage_outputs(packed)
        torch.cuda.synchronize(self.device)
        self._segments["d2h"] += time.perf_counter() - started
        started = time.perf_counter()
        for row_index, request in enumerate(group):
            count = request.action_count
            self._coordinator.slots.output_values[
                request.slot_id, :count
            ].copy_(packed_cpu[row_index, :count])
            self._coordinator.complete(request.slot_id)
            self._queue_latencies_ms.append(
                (time.monotonic_ns() - request.submitted_ns) / 1_000_000.0
            )
        self._segments["publish"] += time.perf_counter() - started
        self._requests += len(group)
        self._actions += sum(row.action_count for row in group)
        self._microbatches += 1
        self._increment(self._batch_histogram, len(group))
        self._increment(self._bucket_histogram, bucket)
        return len(group)

    def _drain_replay(self) -> int:
        started = time.perf_counter()
        if (
            self.belief_buffer is None
            and self.oracle_buffer is None
            and self.cooperation_buffer is None
        ):
            rows = self._replay_slots.read_ready_v3(
                feature_schema_hash=self.model.schema.stable_hash(),
                target_transform=self.model.config.dmc_target_transform,
                ruleset_identity=self.learner.ruleset.identity(),
            )
            self.buffer.extend(rows)
            completed = len(rows)
        elif self.cooperation_buffer is not None:
            aligned = self._replay_slots.read_ready_v3_aligned(
                feature_schema_hash=self.model.schema.stable_hash(),
                target_transform=self.model.config.dmc_target_transform,
                ruleset_identity=self.learner.ruleset.identity(),
            )
            queued_sidecars = {}
            while True:
                try:
                    message = self._cooperation_sidecar_queue.get_nowait()
                except queue.Empty:
                    break
                if (
                    not isinstance(message, tuple)
                    or len(message) != 2
                    or not isinstance(message[0], AsyncReplayKey)
                    or not isinstance(message[1], V3H5AsyncDecisionSidecar)
                ):
                    raise TypeError("H7.1c cooperation sidecar envelope mismatch")
                key, sidecar = message
                if key in queued_sidecars:
                    raise RuntimeError(
                        "duplicate H7.1c cooperation sidecar queue key"
                    )
                queued_sidecars[key] = sidecar
            completed = 0
            for row, key in aligned:
                if row.role not in FARMER_ROLES:
                    self.buffer.append(row)
                    completed += 1
                    continue
                sidecar = queued_sidecars.pop(key, None)
                if sidecar is None:
                    self._cooperation_alignment.add_public(key, row)
                else:
                    completed += self._cooperation_alignment.add_pair(
                        key, row, sidecar
                    )
                    self.stats.cooperation_labels_collected += 1
            for key, sidecar in queued_sidecars.items():
                self._cooperation_alignment.add_sidecar(key, sidecar)
            paired = self._cooperation_alignment.pop_ready_pairs()
            completed += paired
            self.stats.cooperation_labels_collected += paired
            episodes = self._cooperation_alignment.pop_ready_episodes()
            for episode in episodes:
                self.cooperation_buffer.append(episode)
                self.buffer.extend(episode.transitions)
                self.stats.cooperation_episodes_collected += 1
            self.stats.cooperation_incomplete_episodes = (
                self._cooperation_alignment.incomplete_episodes
            )
            self.stats.cooperation_oversized_episodes = (
                self._cooperation_alignment.oversized_episodes
            )
        else:
            aligned = self._replay_slots.read_ready_v3_aligned(
                feature_schema_hash=self.model.schema.stable_hash(),
                target_transform=self.model.config.dmc_target_transform,
                ruleset_identity=self.learner.ruleset.identity(),
            )
            sidecar_queue = (
                self._belief_sidecar_queue
                if self.belief_buffer is not None
                else self._oracle_sidecar_queue
            )
            sidecar_type = (
                V3H4BeliefSidecar
                if self.belief_buffer is not None
                else V3H3OracleSidecar
            )
            alignment = (
                self._belief_alignment
                if self.belief_buffer is not None
                else self._oracle_alignment
            )
            queued_sidecars = {}
            while True:
                try:
                    message = sidecar_queue.get_nowait()
                except queue.Empty:
                    break
                if (
                    not isinstance(message, tuple)
                    or len(message) != 2
                    or not isinstance(message[0], AsyncReplayKey)
                    or not isinstance(message[1], sidecar_type)
                ):
                    raise TypeError("H7 training sidecar envelope mismatch")
                key, sidecar = message
                if key in queued_sidecars:
                    raise RuntimeError("duplicate H7 training sidecar queue key")
                queued_sidecars[key] = sidecar
            paired = []
            for row, key in aligned:
                sidecar = queued_sidecars.pop(key, None)
                if sidecar is None:
                    alignment.add_public(key, row)
                else:
                    paired.append(
                        alignment.add_pair(key, row, sidecar)
                    )
            for key, sidecar in queued_sidecars.items():
                alignment.add_sidecar(key, sidecar)
            paired.extend(alignment.pop_ready())
            completed = len(paired)
            for row, sample in paired:
                self.buffer.append(row)
                if self.belief_buffer is not None:
                    self.belief_buffer.append(sample)
                    self.stats.belief_labels_collected += 1
                else:
                    self.oracle_buffer.append(sample)
                    self.stats.oracle_labels_collected += 1
        self._segments["replay_drain"] += time.perf_counter() - started
        return completed

    def _publish_snapshot(self) -> None:
        if self._runtime_started:
            self._coordinator.quiesce()
        self.inference_model.load_state_dict(self.model.state_dict(), strict=True)
        self.inference_model.eval()
        if self.inference_belief_model is not None:
            self.inference_belief_model.load_state_dict(
                self.belief_model.state_dict(), strict=True
            )
            self.inference_belief_model.eval()
        torch.cuda.synchronize(self.device)
        if self._runtime_started:
            with self._policy_step.get_lock():
                self._policy_step.value = self.policy_step
        self._snapshot_step = self.policy_step

    def collect_episodes(self, num_episodes: int | None = None) -> None:
        target = int(num_episodes or 0)
        if target < 0:
            raise ValueError("H7 episode target must be non-negative")
        if target == 0:
            return
        if not self._runtime_started:
            self._start_runtime()
        self._publish_snapshot()
        for episode_id in range(
            self.stats.games_collected,
            self.stats.games_collected + target,
        ):
            self._tasks.put(episode_id)
        completed = expected = received = 0
        deadline = time.monotonic() + self.config.request_timeout_seconds * max(1, target)
        while completed < target or received < expected:
            self._coordinator._raise_if_failed()
            self._service_requests()
            received += self._drain_replay()
            for process in self._workers:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"H7 actor {process.name} exited with {process.exitcode}"
                    )
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("H7 async collection timed out")
                continue
            if event[0] == "failed":
                raise RuntimeError(event[2])
            if event[0] == "started":
                self._coordinator.active_games += 1
            elif event[0] == "completed":
                completed += 1
                self._coordinator.active_games -= 1
                count = int(event[3])
                expected += count
                self.stats.games_collected += 1
                self.stats.episodes_completed += 1
                self.stats.transitions_collected += count
                self.stats.decisions_collected += int(event[6])
                team = "landlord" if int(event[4]) == 0 else "farmer"
                self.stats.episodes_per_team[team] += 1
                if len(event) >= 9:
                    self._actor_blocked_seconds += float(event[7])
                    self._actor_wall_seconds += float(event[8])
                if self._cooperation_alignment is not None:
                    if len(event) != 10:
                        raise ValueError(
                            "H7.1c completion event lacks farmer decision counts"
                        )
                    self._cooperation_alignment.mark_episode_complete(
                        int(event[1]), int(event[2]), event[9]
                    )
                    episodes = self._cooperation_alignment.pop_ready_episodes()
                    for episode in episodes:
                        self.cooperation_buffer.append(episode)
                        self.buffer.extend(episode.transitions)
                        self.stats.cooperation_episodes_collected += 1
                    self.stats.cooperation_incomplete_episodes = (
                        self._cooperation_alignment.incomplete_episodes
                    )
                    self.stats.cooperation_oversized_episodes = (
                        self._cooperation_alignment.oversized_episodes
                    )

    def optimize(self, num_steps: int) -> None:
        if num_steps < 0:
            raise ValueError("H7 optimizer steps must be non-negative")
        for _ in range(num_steps):
            if self.cooperation_buffer is not None:
                if not self.cooperation_buffer:
                    raise ValueError(
                        "H7.1c replay has no complete farmer episode"
                    )
                episode = self._rng.choice(tuple(self.cooperation_buffer))
                rows = list(episode.transitions)
                trajectories = episode.trajectories
                indices = []
            elif len(self.buffer) < self.config.batch_size:
                raise ValueError("H7 replay has fewer rows than batch_size")
            else:
                indices = self._rng.sample(
                    range(len(self.buffer)), self.config.batch_size
                )
                rows = [self.buffer[index] for index in indices]
                trajectories = None
            started = time.perf_counter()
            learner_rows = self._learner_rows(rows)
            belief_samples = (
                None
                if self.belief_buffer is None
                else [self.belief_buffer[index] for index in indices]
            )
            oracle_samples = self._oracle_samples_for_indices(indices)
            learner_policy_before = int(self.learner.policy_version)
            metrics = self.learner.train_batch(
                learner_rows,
                trajectories=trajectories,
                belief_samples=belief_samples,
                oracle_samples=oracle_samples,
            )
            self._record_served_update(learner_policy_before, metrics)
            if metrics.base.base.belief_updated:
                self.stats.belief_optimizer_steps += 1
            h3_metrics = metrics.base.base.base
            if h3_metrics is not None and h3_metrics.oracle_updated:
                self.stats.oracle_optimizer_steps += 1
            if metrics.base.cooperation_updated:
                self.stats.cooperation_optimizer_steps += 1
            self.stats.optimizer_steps += 1
            self.stats.learner_cardplay_samples += len(rows)
            self._segments["learner"] += time.perf_counter() - started

    def step(self):
        """Run one learner update for the shared long-running controller."""
        if self.cooperation_buffer is not None:
            if not self.cooperation_buffer:
                return None
            episode = self._rng.choice(tuple(self.cooperation_buffer))
            rows = list(episode.transitions)
            trajectories = episode.trajectories
            indices = []
        else:
            if len(self.buffer) < self.config.batch_size:
                return None
            indices = self._rng.sample(
                range(len(self.buffer)), self.config.batch_size
            )
            rows = [self.buffer[index] for index in indices]
            trajectories = None
        learner_rows = self._learner_rows(rows)
        belief_samples = (
            None
            if self.belief_buffer is None
            else [self.belief_buffer[index] for index in indices]
        )
        oracle_samples = self._oracle_samples_for_indices(indices)
        started = time.perf_counter()
        learner_policy_before = int(self.learner.policy_version)
        metrics = self.learner.train_batch(
            learner_rows,
            trajectories=trajectories,
            belief_samples=belief_samples,
            oracle_samples=oracle_samples,
        )
        self._record_served_update(learner_policy_before, metrics)
        if metrics.base.base.belief_updated:
            self.stats.belief_optimizer_steps += 1
        h3_metrics = metrics.base.base.base
        if h3_metrics is not None and h3_metrics.oracle_updated:
            self.stats.oracle_optimizer_steps += 1
        if metrics.base.cooperation_updated:
            self.stats.cooperation_optimizer_steps += 1
        self.stats.optimizer_steps += 1
        self.stats.learner_cardplay_samples += len(rows)
        self._segments["learner"] += time.perf_counter() - started
        return metrics

    def _record_served_update(self, learner_policy_before: int, metrics) -> None:
        """Version belief-only changes to the coupled served snapshot."""

        learner_policy_after = int(self.learner.policy_version)
        if learner_policy_after < learner_policy_before:
            raise RuntimeError("H7 learner policy version moved backwards")
        if (
            metrics.base.base.belief_updated
            and learner_policy_after == learner_policy_before
        ):
            self._served_version_offset += 1

    def _learner_rows(
        self, rows: list[V3ReplayTransition]
    ) -> list[V3ReplayTransition]:
        if self.resolved_config.learner.features.adaptive_dmc:
            return rows
        if (
            self.belief_buffer is None
            and self.oracle_buffer is None
            and self.cooperation_buffer is None
        ):
            raise RuntimeError(
                "H7 ordinary DMC replay requires a sidecar-enabled runtime"
            )
        return [replace(row, adaptive_provenance=None) for row in rows]

    def _oracle_samples_for_indices(self, indices: list[int]):
        if self.oracle_buffer is None:
            return None
        h3 = self.learner.base.base.base
        return (
            [self.oracle_buffer[index] for index in indices]
            if h3._privileged_needed(h3.schedule_state())
            else None
        )

    def _parameter_update_snapshot(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.detach().clone()
            for parameter in self._tracked_training_parameters()
        )

    def _tracked_training_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        parameters = tuple(self.model.parameters())
        if self.belief_model is not None:
            parameters += tuple(self.belief_model.parameters())
        oracle = self.learner.base.base.base.oracle
        if oracle is not None:
            parameters += tuple(oracle.parameters())
        cooperation = self.learner.base.cooperation
        if cooperation is not None:
            parameters += tuple(cooperation.parameters())
        return parameters

    def _parameters_changed_since(
        self, snapshots: tuple[torch.Tensor, ...]
    ) -> bool:
        parameters = self._tracked_training_parameters()
        if len(snapshots) != len(parameters):
            raise ValueError("H7 parameter update snapshot shape changed")
        return any(
            not torch.equal(snapshot, parameter.detach())
            for snapshot, parameter in zip(snapshots, parameters)
        )

    def quiesce_cycle_boundary(self) -> dict[str, object]:
        if self._runtime_started:
            self._drain_replay()
            counts = self._coordinator.quiesce()
            # Actors are between games here. Publishing before checkpointing
            # makes learner-to-served lag observable and exactly zero without
            # treating idle optimizer updates as actor policy lag.
            self._publish_snapshot()
        else:
            counts = {"writing": 0, "ready": 0, "running": 0}
        lag = self.policy_step - self._snapshot_step
        if lag > self.config.max_policy_lag:
            raise RuntimeError("H7 policy lag exceeded its configured bound")
        if self._belief_alignment is not None:
            self._belief_alignment.assert_quiescent()
        if self._oracle_alignment is not None:
            self._oracle_alignment.assert_quiescent()
        if self._cooperation_alignment is not None:
            self._cooperation_alignment.assert_quiescent()
        latencies = sorted(self._queue_latencies_ms)

        def percentile(fraction: float) -> float:
            if not latencies:
                return 0.0
            return latencies[int((len(latencies) - 1) * fraction)]

        result = {
            "active_slots": counts["writing"] + counts["ready"] + counts["running"],
            "in_flight_slots": counts["ready"] + counts["running"],
            "pending_requests": self._scheduler.pending_count if self._runtime_started else 0,
            "replay_occupancy": len(self.buffer),
            "belief_replay_occupancy": (
                0 if self.belief_buffer is None else len(self.belief_buffer)
            ),
            "belief_labels_collected": self.stats.belief_labels_collected,
            "belief_optimizer_steps": self.stats.belief_optimizer_steps,
            "oracle_replay_occupancy": (
                0 if self.oracle_buffer is None else len(self.oracle_buffer)
            ),
            "oracle_labels_collected": self.stats.oracle_labels_collected,
            "oracle_optimizer_steps": self.stats.oracle_optimizer_steps,
            "oracle_parameter_vram_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in (
                    ()
                    if self.learner.base.base.base.oracle is None
                    else self.learner.base.base.base.oracle.parameters()
                )
            ),
            "cooperation_replay_occupancy": (
                0
                if self.cooperation_buffer is None
                else len(self.cooperation_buffer)
            ),
            "cooperation_labels_collected": (
                self.stats.cooperation_labels_collected
            ),
            "cooperation_episodes_collected": (
                self.stats.cooperation_episodes_collected
            ),
            "cooperation_optimizer_steps": (
                self.stats.cooperation_optimizer_steps
            ),
            "cooperation_incomplete_episodes": (
                self.stats.cooperation_incomplete_episodes
            ),
            "cooperation_oversized_episodes": (
                self.stats.cooperation_oversized_episodes
            ),
            "cooperation_parameter_vram_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in (
                    ()
                    if self.learner.base.cooperation is None
                    else self.learner.base.cooperation.parameters()
                )
            ),
            "requests_per_microbatch": self._requests / max(1, self._microbatches),
            "actions_per_microbatch": self._actions / max(1, self._microbatches),
            "inference_queue_p50_ms": percentile(0.50),
            "inference_queue_p95_ms": percentile(0.95),
            "inference_queue_p99_ms": percentile(0.99),
            "policy_lag": lag,
            "actor_blocked_ratio": (
                self._actor_blocked_seconds / max(self._actor_wall_seconds, 1.0e-12)
            ),
            "learner_data_wait_ratio": 0.0,
            "learner_throttle_seconds": 0.0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "microbatch_size_histogram": dict(self._batch_histogram),
            "inference_bucket_histogram": dict(self._bucket_histogram),
            **{f"{name}_seconds": value for name, value in self._segments.items()},
        }
        self._reset_metrics()
        return result

    def clear_replay(self) -> None:
        self.buffer.clear()
        if self.belief_buffer is not None:
            self.belief_buffer.clear()
        if self.oracle_buffer is not None:
            self.oracle_buffer.clear()
        if self.cooperation_buffer is not None:
            self.cooperation_buffer.clear()

    def save_training_checkpoint(self, path: str, *, long_running_state) -> None:
        if self._belief_alignment is not None:
            self._belief_alignment.assert_quiescent()
        if self._oracle_alignment is not None:
            self._oracle_alignment.assert_quiescent()
        if self._cooperation_alignment is not None:
            self._cooperation_alignment.assert_quiescent()
        with tempfile.TemporaryDirectory(prefix="douzero-h7-save-") as temporary:
            inner_path = Path(temporary) / "h6.pt"
            self.learner.save_checkpoint(inner_path)
            bundle = {
                "format": V3_H7_CHECKPOINT_FORMAT,
                "artifact_access": "privileged_training_only",
                "runtime_identity": self.runtime_identity,
                "runtime_hash": self.runtime_hash,
                "h6_checkpoint": torch.load(
                    inner_path, map_location="cpu", weights_only=True
                ),
                "stats": asdict(self.stats),
                "rng_state": self._rng.getstate(),
                "served_version_offset": self._served_version_offset,
                "snapshot_step": self._snapshot_step,
                "long_running_state": dict(long_running_state),
            }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(bundle, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_training_checkpoint(self, path: str | Path):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format", "artifact_access", "runtime_identity", "runtime_hash",
            "h6_checkpoint", "stats", "rng_state", "served_version_offset",
            "snapshot_step",
            "long_running_state",
        }
        if not isinstance(bundle, dict) or set(bundle) != expected:
            raise ValueError("H7 checkpoint fields mismatch")
        if bundle["format"] != V3_H7_CHECKPOINT_FORMAT:
            raise ValueError("H7 checkpoint format mismatch")
        if bundle["artifact_access"] != "privileged_training_only":
            raise ValueError("H7 checkpoint access class mismatch")
        if bundle["runtime_hash"] != self.runtime_hash:
            raise ValueError("H7 runtime identity mismatch")
        if bundle["runtime_identity"] != self.runtime_identity:
            raise ValueError("H7 runtime identity payload mismatch")
        stats_payload = bundle["stats"]
        if not isinstance(stats_payload, dict) or set(stats_payload) != {
            "games_collected", "episodes_completed", "transitions_collected",
            "decisions_collected", "optimizer_steps", "episodes_per_team",
            "amp_fallbacks", "learner_cardplay_samples",
            "belief_labels_collected", "belief_optimizer_steps",
            "oracle_labels_collected", "oracle_optimizer_steps",
            "cooperation_labels_collected",
            "cooperation_episodes_collected",
            "cooperation_optimizer_steps",
            "cooperation_incomplete_episodes",
            "cooperation_oversized_episodes",
        }:
            raise ValueError("H7 checkpoint statistics fields mismatch")
        candidate_stats = V3H7RuntimeStats(**stats_payload)
        for name in (
            "games_collected", "episodes_completed", "transitions_collected",
            "decisions_collected", "optimizer_steps",
            "amp_fallbacks", "learner_cardplay_samples",
            "belief_labels_collected", "belief_optimizer_steps",
            "oracle_labels_collected", "oracle_optimizer_steps",
            "cooperation_labels_collected",
            "cooperation_episodes_collected",
            "cooperation_optimizer_steps",
            "cooperation_incomplete_episodes",
            "cooperation_oversized_episodes",
        ):
            value = getattr(candidate_stats, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"H7 checkpoint statistic {name} is invalid")
        if (
            not isinstance(candidate_stats.episodes_per_team, dict)
            or set(candidate_stats.episodes_per_team) != {"landlord", "farmer"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in candidate_stats.episodes_per_team.values()
            )
        ):
            raise ValueError("H7 checkpoint team statistics are invalid")
        if not self.config.belief_runtime_enabled and (
            candidate_stats.belief_labels_collected
            or candidate_stats.belief_optimizer_steps
        ):
            raise ValueError("H7 base checkpoint contains belief progress")
        if not self.config.oracle_runtime_enabled and (
            candidate_stats.oracle_labels_collected
            or candidate_stats.oracle_optimizer_steps
        ):
            raise ValueError("H7 checkpoint contains unexpected Oracle progress")
        if not self.config.cooperation_runtime_enabled and any((
            candidate_stats.cooperation_labels_collected,
            candidate_stats.cooperation_episodes_collected,
            candidate_stats.cooperation_optimizer_steps,
            candidate_stats.cooperation_incomplete_episodes,
            candidate_stats.cooperation_oversized_episodes,
        )):
            raise ValueError(
                "H7 checkpoint contains unexpected cooperation progress"
            )
        if (
            candidate_stats.belief_optimizer_steps
            > candidate_stats.optimizer_steps
            or candidate_stats.belief_labels_collected
            > candidate_stats.transitions_collected
        ):
            raise ValueError("H7 checkpoint belief statistics are invalid")
        if (
            candidate_stats.oracle_optimizer_steps
            > candidate_stats.optimizer_steps
            or candidate_stats.oracle_labels_collected
            > candidate_stats.transitions_collected
        ):
            raise ValueError("H7 checkpoint Oracle statistics are invalid")
        if (
            candidate_stats.cooperation_optimizer_steps
            > candidate_stats.optimizer_steps
            or candidate_stats.cooperation_labels_collected
            > candidate_stats.transitions_collected
            or candidate_stats.cooperation_episodes_collected
            > candidate_stats.episodes_completed
            or (
                candidate_stats.cooperation_episodes_collected
                + candidate_stats.cooperation_incomplete_episodes
                + candidate_stats.cooperation_oversized_episodes
                > candidate_stats.episodes_completed
            )
        ):
            raise ValueError("H7 checkpoint cooperation statistics are invalid")
        candidate_rng = random.Random()
        candidate_rng.setstate(bundle["rng_state"])
        snapshot_step = bundle["snapshot_step"]
        if isinstance(snapshot_step, bool) or not isinstance(snapshot_step, int):
            raise ValueError("H7 checkpoint snapshot step is invalid")
        served_version_offset = bundle["served_version_offset"]
        if (
            isinstance(served_version_offset, bool)
            or not isinstance(served_version_offset, int)
            or served_version_offset < 0
        ):
            raise ValueError("H7 checkpoint served version offset is invalid")
        inner_counters = bundle["h6_checkpoint"].get("counters", {})
        inner_policy_step = inner_counters.get("policy_version")
        if (
            isinstance(inner_policy_step, bool)
            or not isinstance(inner_policy_step, int)
            or inner_policy_step < 0
        ):
            raise ValueError("H7 nested learner policy version is invalid")
        coupled_policy_step = inner_policy_step + served_version_offset
        if snapshot_step < 0 or snapshot_step > coupled_policy_step:
            raise ValueError("H7 checkpoint snapshot is newer than learner")
        if coupled_policy_step - snapshot_step > self.config.max_policy_lag:
            raise ValueError("H7 checkpoint policy lag exceeds its bound")
        previous_stats = copy.deepcopy(self.stats)
        previous_rng = self._rng.getstate()
        previous_offset = self._served_version_offset
        previous_snapshot = self._snapshot_step
        with tempfile.TemporaryDirectory(prefix="douzero-h7-load-") as temporary:
            inner_path = Path(temporary) / "h6.pt"
            torch.save(bundle["h6_checkpoint"], inner_path)
            try:
                self.learner.load_checkpoint(inner_path)
                self.stats = candidate_stats
                self._rng.setstate(bundle["rng_state"])
                self._served_version_offset = served_version_offset
                self._snapshot_step = snapshot_step
                if int(self.learner.policy_version) != inner_policy_step:
                    raise ValueError(
                        "H7 nested learner policy version failed to restore"
                    )
            except Exception:
                self.stats = previous_stats
                self._rng.setstate(previous_rng)
                self._served_version_offset = previous_offset
                self._snapshot_step = previous_snapshot
                raise
        return bundle["long_running_state"]

    def shutdown(self) -> None:
        if not self._runtime_started:
            return
        error = None
        try:
            self._coordinator.request_shutdown()
            for _ in self._workers:
                self._tasks.put(None)
            alive = _stop_actor_processes(
                self._workers,
                (
                    self._belief_sidecar_queue,
                    self._oracle_sidecar_queue,
                    self._cooperation_sidecar_queue,
                ),
                5.0,
            )
            if alive:
                error = RuntimeError(f"H7 actor shutdown timed out: {alive}")
            from douzero.training.async_single_gpu import SlotState

            active_slots = sum(
                int((self._coordinator.states == int(state)).sum().item())
                for state in (SlotState.WRITING, SlotState.READY, SlotState.RUNNING)
            )
            if active_slots:
                error = RuntimeError("H7 shutdown left active inference slots")
            if self._scheduler.pending_count:
                error = RuntimeError("H7 shutdown left pending requests")
            self._coordinator.active_games = 0
            self._coordinator.completed_episodes_pending = 0
        finally:
            self._coordinator.shutdown()
            self._replay_slots.close()
            if self._belief_sidecar_queue is not None:
                self._belief_sidecar_queue.close()
                self._belief_sidecar_queue.join_thread()
            if self._oracle_sidecar_queue is not None:
                self._oracle_sidecar_queue.close()
                self._oracle_sidecar_queue.join_thread()
            if self._cooperation_sidecar_queue is not None:
                self._cooperation_sidecar_queue.close()
                self._cooperation_sidecar_queue.join_thread()
            if self._cooperation_alignment is not None:
                self._cooperation_alignment.discard_pending()
            self._tasks.close()
            self._events.close()
            self._runtime_started = False
        if error is not None:
            raise error


class V3SingleProcessTrainer(V3AsyncSingleGPUTrainer):
    """Reference end-to-end V3 self-play topology without actor workers."""

    def __init__(
        self,
        learner: V3H6Learner,
        resolved_config: V3H6ResolvedConfig,
        runtime_config: V3H7RuntimeConfig,
    ) -> None:
        if runtime_config.topology != TOPOLOGY_SINGLE_PROCESS:
            raise ValueError("H7 single-process trainer requires single_process topology")
        if runtime_config.num_actors != 1 or runtime_config.games_per_actor != 1:
            raise ValueError("H7 single-process topology requires 1 actor x 1 game")
        super().__init__(learner, resolved_config, runtime_config)

    def collect_episodes(self, num_episodes: int | None = None) -> None:
        import numpy as np

        from douzero.env.env import Env
        from douzero.observation.encode_v2 import get_obs_v2
        from douzero.observation.privileged import PrivilegedObservation

        from .replay import (
            AdaptiveSnapshotProvenance,
            capture_plain_transition,
        )

        target = int(num_episodes or 0)
        if target < 0:
            raise ValueError("H7 episode target must be non-negative")
        self._publish_snapshot()
        for episode_number in range(
            self.stats.games_collected,
            self.stats.games_collected + target,
        ):
            environment_seed = (
                derive_formal_stream_seed(
                    self.config.environment_seed,
                    "environment",
                    0,
                    episode_number,
                )
                if self.config.environment_seed_derivation
                == FORMAL_SEED_DERIVATION_V1
                else (self.config.environment_seed + episode_number) % (1 << 32)
            )
            np.random.seed(environment_seed)
            action_rng = (
                random.Random(derive_formal_stream_seed(
                    self.config.action_seed,
                    "action",
                    0,
                    episode_number,
                ))
                if self.config.environment_seed_derivation
                == FORMAL_SEED_DERIVATION_V1
                else self._rng
            )
            env = Env("adp")
            env.reset()
            pending = []
            pending_belief = []
            pending_oracle = []
            pending_cooperation = []
            decisions = 0
            steps = 0
            while True:
                position = env._acting_player_position
                legal_actions = env.infoset.legal_actions
                decisions += 1
                if len(legal_actions) == 1:
                    action_index = 0
                else:
                    observation = get_obs_v2(env.infoset, ruleset=self.learner.ruleset)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = (
                            self.inference_model.forward_observation(observation)
                            if self.inference_policy is None
                            else self.inference_policy.forward_observation(
                                observation
                            )
                        )
                    torch.cuda.synchronize(self.device)
                    self._segments["forward"] += time.perf_counter() - started
                    mask = output.action_mask.bool()
                    q_values = output.dmc_q[:, 0].masked_fill(
                        ~mask, float("-inf")
                    )
                    valid = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                    action_index = (
                        int(action_rng.choice(valid))
                        if self.config.epsilon > 0.0
                        and action_rng.random() < self.config.epsilon
                        else int(torch.argmax(q_values).item())
                    )
                    captured = capture_plain_transition(
                        observation,
                        selected_action_index=action_index,
                        episode_id=f"single-episode-{episode_number}",
                        deal_id=f"single-deal-{episode_number}",
                        target_transform=self.model.config.dmc_target_transform,
                    )
                    if self.belief_buffer is not None:
                        sidecar = build_v3_h4_belief_sidecar(
                            observation,
                            PrivilegedObservation(
                                all_handcards=dict(env.infoset.all_handcards),
                                acting_role=position,
                            ),
                            public_inputs=captured.model_inputs,
                        )
                        pending_belief.append(
                            bind_v3_h4_belief_sidecar(
                                captured.model_inputs, sidecar
                            )
                        )
                    if self.oracle_buffer is not None:
                        pending_oracle.append(build_v3_h3_oracle_sidecar(
                            observation,
                            PrivilegedObservation(
                                all_handcards=dict(env.infoset.all_handcards),
                                acting_role=position,
                            ),
                            action_index=action_index,
                            public_inputs=captured.model_inputs,
                        ))
                    if self.cooperation_buffer is not None and position in FARMER_ROLES:
                        provenance = f"{self.policy_version}@{self._snapshot_step}"
                        pending_cooperation.append((
                            len(pending),
                            build_v3_h5_async_decision_sidecar(
                                observation,
                                selected_action_index=action_index,
                                trace_index=steps,
                                public_inputs=captured.model_inputs,
                                policy_id=provenance,
                                teammate_policy_id=provenance,
                            ),
                        ))
                    pending.append(replace(
                        captured,
                        adaptive_provenance=AdaptiveSnapshotProvenance(
                            q_old=float(q_values[action_index].item()),
                            policy_version=self._snapshot_step,
                            snapshot_slot=0,
                            owner_id=0,
                            generation=episode_number + 1,
                        ),
                    ))
                _obs, _reward, done, info = env.step(legal_actions[action_index])
                steps += 1
                if done:
                    break
                if steps >= self.config.max_steps_per_episode:
                    raise RuntimeError(
                        "H7 single-process episode exceeded max_steps_per_episode"
                    )
            terminal = info or {}
            team_targets = terminal.get("team_targets")
            if not isinstance(team_targets, dict):
                raise ValueError("H7 terminal result is missing team_targets")
            rows = [
                row.finalize(float(team_targets[row.role]["target_score"]))
                for row in pending
            ]
            self.buffer.extend(rows)
            if self.belief_buffer is not None:
                if len(pending_belief) != len(rows):
                    raise RuntimeError(
                        "H7.1a single-process belief alignment mismatch"
                    )
                self.belief_buffer.extend(pending_belief)
                self.stats.belief_labels_collected += len(pending_belief)
            if self.oracle_buffer is not None:
                if len(pending_oracle) != len(rows):
                    raise RuntimeError(
                        "H7.1b single-process Oracle alignment mismatch"
                    )
                self.oracle_buffer.extend(
                    bind_v3_h3_oracle_sidecar(row, sidecar)
                    for row, sidecar in zip(rows, pending_oracle)
                )
                self.stats.oracle_labels_collected += len(pending_oracle)
            if self.cooperation_buffer is not None:
                by_role = {role: [] for role in FARMER_ROLES}
                for row_index, sidecar in pending_cooperation:
                    decision = bind_v3_h5_async_decision(
                        rows[row_index], sidecar
                    )
                    by_role[sidecar.role].append((decision, sidecar))
                if any(not by_role[role] for role in FARMER_ROLES):
                    self.stats.cooperation_incomplete_episodes += 1
                else:
                    trajectories = []
                    for role in FARMER_ROLES:
                        entries = sorted(
                            by_role[role], key=lambda item: item[0].trace_index
                        )
                        first = entries[0][0].transition
                        trajectories.append(V3H5FarmerTrajectory(
                            episode_id=first.episode_id,
                            deal_id=first.deal_id,
                            role=role,
                            policy_id=entries[0][1].policy_id,
                            teammate_policy_id=(
                                entries[0][1].teammate_policy_id
                            ),
                            decisions=tuple(item[0] for item in entries),
                            team_return=first.mc_return,
                        ))
                    episode_rows = tuple(
                        row
                        for trajectory in trajectories
                        for row in trajectory.transitions
                    )
                    if len(episode_rows) > self.config.batch_size:
                        self.stats.cooperation_oversized_episodes += 1
                    else:
                        episode = V3H71CCooperationEpisode(
                            transitions=episode_rows,
                            trajectories=(
                                trajectories[0], trajectories[1]
                            ),
                        )
                        self.cooperation_buffer.append(episode)
                        self.stats.cooperation_episodes_collected += 1
                self.stats.cooperation_labels_collected += len(
                    pending_cooperation
                )
            self.stats.games_collected += 1
            self.stats.episodes_completed += 1
            self.stats.transitions_collected += len(rows)
            self.stats.decisions_collected += decisions
            team = terminal.get("winner_team")
            if team not in {"landlord", "farmer"}:
                raise ValueError("H7 terminal winner_team is invalid")
            self.stats.episodes_per_team[team] += 1

    def quiesce_cycle_boundary(self) -> dict[str, object]:
        self._publish_snapshot()
        return super().quiesce_cycle_boundary()

    def shutdown(self) -> None:
        return None


__all__ = [
    "V3_H71A_REPLAY_PROTOCOL",
    "V3_H71A_REQUEST_PROTOCOL",
    "V3_H71A_SNAPSHOT_SEMANTICS",
    "V3_H71B_REPLAY_PROTOCOL",
    "V3_H71B_REQUEST_PROTOCOL",
    "V3_H71C_REPLAY_PROTOCOL",
    "V3_H71C_REQUEST_PROTOCOL",
    "V3_H7_CHECKPOINT_FORMAT",
    "V3_H7_REPLAY_PROTOCOL",
    "V3_H7_REQUEST_PROTOCOL",
    "V3_H7_RUNTIME_VERSION",
    "V3AsyncSingleGPUTrainer",
    "V3H71ABeliefAlignment",
    "V3H71BOracleAlignment",
    "V3H71CCooperationAlignment",
    "V3H71CCooperationEpisode",
    "V3SingleProcessTrainer",
    "V3H7RuntimeConfig",
    "V3H7RuntimeStats",
    "validate_v3_h7_runtime_config",
    "validate_v3_h7_formal_initialization",
]
