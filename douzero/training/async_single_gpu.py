"""Spawn-safe shared-memory protocol for V2 centralized GPU inference.

Only slot identifiers and compact integer metadata cross queues.  Observation
tensors live in a fixed CPU shared-memory slab owned by the main process.  The
state machine is deliberately independent of CUDA so timeout, crash and
quiescence behavior can be tested on CPU hosts.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import ctypes
import math
from dataclasses import dataclass
from enum import IntEnum

import torch
import numpy as np

from douzero.models_v2.batch import ModelInputBundle
from douzero.models_v2.config import SUPPORTED_ROLES
from douzero.observation.schema import (
    action_width,
    context_width,
    history_token_width,
    state_width,
)
from douzero.training.v2_buffer import action_count_bucket, compact_model_input_shapes


def _shared_tensor(shape, dtype=torch.float32):
    """Create a tensor view over stdlib shared memory (no torch_shm_manager)."""
    numel = math.prod(shape)
    sizes = {
        torch.float32: 4,
        torch.bool: 1,
        torch.int8: 1,
        torch.int32: 4,
        torch.int64: 8,
    }
    if dtype not in sizes:
        raise TypeError(f"unsupported shared tensor dtype {dtype}")
    owner = mp.get_context("spawn").RawArray(ctypes.c_ubyte, numel * sizes[dtype])
    tensor = torch.frombuffer(owner, dtype=dtype, count=numel).reshape(shape)
    tensor.zero_()
    return tensor, owner


def _restore_shared_tensor(owner, shape, dtype):
    return torch.frombuffer(
        owner, dtype=dtype, count=math.prod(shape)
    ).reshape(shape)


class SlotState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    RUNNING = 3
    DONE = 4
    FAILED = 5
    SHUTDOWN = 6


@dataclass(frozen=True)
class RequestMetadata:
    slot_id: int
    actor_id: int
    request_id: int
    policy_snapshot: int
    action_count: int
    acting_role: int
    submitted_ns: int

    @property
    def grouping_key(self) -> tuple[int, int | str, int]:
        return (
            self.policy_snapshot,
            action_count_bucket(self.action_count),
            self.acting_role,
        )


class SharedObservationSlots:
    """Preallocated CPU shared tensors for inference requests and responses."""

    def __init__(self, schema, num_slots: int, max_actions: int = 256) -> None:
        if num_slots < 1 or max_actions < 1:
            raise ValueError("shared slot dimensions must be positive")
        card_dim = schema.card_vector_dim
        state_flat = state_width(schema) - 6 * card_dim
        context_flat = context_width(schema) - 2 * card_dim
        history_width = history_token_width(schema)
        action_feature_width = action_width(schema)
        self.num_slots = int(num_slots)
        self.max_actions = int(max_actions)

        self._shared_owners = []
        self._shared_specs = []

        def shared(shape, dtype=torch.float32):
            tensor, owner = _shared_tensor(shape, dtype)
            self._shared_owners.append(owner)
            self._shared_specs.append((tuple(shape), dtype))
            return tensor

        self.state_cards = shared((num_slots, 6, card_dim))
        self.state_flat = shared((num_slots, state_flat))
        self.context_cards = shared((num_slots, 2, card_dim))
        self.context_flat = shared((num_slots, context_flat))
        self.history = shared(
            (num_slots, schema.max_history_len, history_width)
        )
        self.history_padding = shared(
            (num_slots, schema.max_history_len), torch.bool
        )
        self.actions = shared(
            (num_slots, max_actions, action_feature_width)
        )
        self.action_mask = shared((num_slots, max_actions), torch.bool)
        self.output_win = shared((num_slots, max_actions))
        self.output_score = shared((num_slots, max_actions))
        self.output_score_win = shared((num_slots, max_actions))
        self.output_score_loss = shared((num_slots, max_actions))
        self.output_p_win = shared((num_slots, max_actions))
        self.action_counts = shared((num_slots,), torch.int32)
        self.roles = shared((num_slots,), torch.int64)

    _TENSOR_FIELDS = (
        "state_cards", "state_flat", "context_cards", "context_flat",
        "history", "history_padding", "actions", "action_mask",
        "output_win", "output_score", "output_score_win", "output_score_loss",
        "output_p_win", "action_counts", "roles",
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def write(self, slot_id: int, bundle: ModelInputBundle) -> None:
        count = int(bundle.action_features.shape[0])
        if count < 1:
            raise ValueError("inference request has zero legal actions")
        if count > self.max_actions:
            raise ValueError(
                f"inference request has {count} actions; shared max is {self.max_actions}"
            )
        self.state_cards[slot_id].copy_(torch.stack(bundle.state_card_vectors))
        self.state_flat[slot_id].copy_(bundle.state_context_flat)
        self.context_cards[slot_id].copy_(torch.stack(bundle.context_card_vectors))
        self.context_flat[slot_id].copy_(bundle.context_flat)
        self.history[slot_id].copy_(bundle.history_tokens)
        self.history_padding[slot_id].copy_(bundle.history_key_padding_mask)
        self.actions[slot_id].zero_()
        self.action_mask[slot_id].zero_()
        self.actions[slot_id, :count].copy_(bundle.action_features)
        self.action_mask[slot_id, :count].copy_(bundle.action_mask)
        self.action_counts[slot_id] = count
        try:
            role = SUPPORTED_ROLES.index(bundle.acting_role)
        except ValueError as exc:
            raise ValueError("unsupported acting role") from exc
        self.roles[slot_id] = role

    def read_bundle(self, slot_id: int, feature_schema_hash: str) -> ModelInputBundle:
        count = int(self.action_counts[slot_id])
        role = SUPPORTED_ROLES[int(self.roles[slot_id])]
        return ModelInputBundle(
            state_card_vectors=tuple(self.state_cards[slot_id, i] for i in range(6)),
            state_context_flat=self.state_flat[slot_id],
            context_card_vectors=tuple(self.context_cards[slot_id, i] for i in range(2)),
            context_flat=self.context_flat[slot_id],
            history_tokens=self.history[slot_id],
            history_key_padding_mask=self.history_padding[slot_id],
            action_features=self.actions[slot_id, :count],
            action_mask=self.action_mask[slot_id, :count],
            acting_role=role,
            feature_schema_hash=feature_schema_hash,
        )


class SharedReplaySlots:
    """Shared tensor handoff for completed, terminal-labelled transitions."""

    TARGET_NAMES = (
        "target_win", "target_score", "target_log_score",
        "target_min_turns_after", "target_min_turns_exact_mask",
        "target_regain_initiative", "target_teammate_finish",
        "target_teammate_finish_mask", "target_spring_probability",
        "target_structure_cost",
    )

    def __init__(self, schema, num_slots: int, max_actions: int = 256) -> None:
        self.context = mp.get_context("spawn")
        self.observations = SharedObservationSlots(schema, num_slots, max_actions)
        self._validation_shapes = compact_model_input_shapes(schema)
        self._shared_owners = []
        self._shared_specs = []
        self.labels, owner = _shared_tensor((num_slots, len(self.TARGET_NAMES)))
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots, len(self.TARGET_NAMES)), torch.float32))
        self.labels.fill_(float("nan"))
        self.action_indices, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.action_indices.fill_(-1)
        self.trace_indices, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.trace_indices.fill_(-1)
        self.policy_steps, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.policy_steps.fill_(-1)
        self.free_queue = self.context.Queue()
        self.ready_queue = self.context.Queue()
        for slot_id in range(num_slots):
            self.free_queue.put(slot_id)

    _TENSOR_FIELDS = ("labels", "action_indices", "trace_indices", "policy_steps")

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def write_transition(
        self, transition, bundle: ModelInputBundle, policy_step: int,
        timeout_seconds: float, abort_event=None, shutdown_event=None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("async runtime aborted while waiting for replay slot")
            if shutdown_event is not None and shutdown_event.is_set():
                raise RuntimeError("async runtime shut down while waiting for replay slot")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a shared replay slot")
            try:
                slot_id = int(self.free_queue.get(timeout=min(0.05, remaining)))
                break
            except queue.Empty:
                continue
        self.observations.write(slot_id, bundle)
        self.action_indices[slot_id] = transition.action_index
        self.trace_indices[slot_id] = transition.trace_index
        self.policy_steps[slot_id] = policy_step
        for index, name in enumerate(self.TARGET_NAMES):
            self.labels[slot_id, index] = float(getattr(transition, name))
        self.ready_queue.put(slot_id)

    def read_ready(self, feature_schema_hash: str, policy_version: str):
        from douzero.training.v2_buffer import (
            COMPACT_REPLAY_SCHEMA_VERSION,
            CompactTensorTransition,
        )

        records = []
        while True:
            try:
                slot_id = int(self.ready_queue.get_nowait())
            except queue.Empty:
                break
            source = self.observations.read_bundle(slot_id, feature_schema_hash)
            bundle = ModelInputBundle(
                state_card_vectors=tuple(value.to(torch.int8) for value in source.state_card_vectors),
                state_context_flat=source.state_context_flat.to(torch.int8),
                context_card_vectors=tuple(value.to(torch.int8) for value in source.context_card_vectors),
                context_flat=source.context_flat.to(torch.int32),
                history_tokens=source.history_tokens.to(torch.int8),
                history_key_padding_mask=source.history_key_padding_mask.clone(),
                action_features=source.action_features.to(torch.int8),
                action_mask=source.action_mask.clone(),
                acting_role=source.acting_role,
                feature_schema_hash=feature_schema_hash,
            )
            targets = {
                name: float(self.labels[slot_id, index].item())
                for index, name in enumerate(self.TARGET_NAMES)
            }
            records.append(CompactTensorTransition(
                model_inputs=bundle,
                action_index=int(self.action_indices[slot_id]),
                position=source.acting_role,
                targets=targets,
                trace_index=int(self.trace_indices[slot_id]),
                policy_id=policy_version,
                teammate_policy_id=None,
                policy_version=policy_version,
                policy_step=int(self.policy_steps[slot_id]),
                schema_version=COMPACT_REPLAY_SCHEMA_VERSION,
            ))
            records[-1].validate(
                feature_schema_hash,
                expected_tensor_shapes=self._validation_shapes,
            )
            self.free_queue.put(slot_id)
        return records

    def close(self) -> None:
        self.free_queue.close()
        self.ready_queue.close()


class AsyncRequestCoordinator:
    """Fail-fast shared request state machine using a ``spawn`` context."""

    def __init__(
        self,
        schema,
        *,
        num_slots: int,
        max_actions: int = 256,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.context = mp.get_context("spawn")
        if self.context.get_start_method() != "spawn":
            raise RuntimeError("async V2 requires multiprocessing spawn")
        self.slots = SharedObservationSlots(schema, num_slots, max_actions)
        self._shared_owners = []
        self._shared_specs = []
        self.states, owner = _shared_tensor((num_slots,), torch.int8)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int8))
        self.request_ids, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.request_ids.fill_(-1)
        self.policy_snapshots, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.policy_snapshots.fill_(-1)
        self.actor_ids, owner = _shared_tensor((num_slots,), torch.int32)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int32))
        self.actor_ids.fill_(-1)
        self.submitted_ns, owner = _shared_tensor((num_slots,), torch.int64)
        self._shared_owners.append(owner)
        self._shared_specs.append(((num_slots,), torch.int64))
        self.ready_queue = self.context.Queue()
        self.free_queue = self.context.Queue()
        for slot_id in range(num_slots):
            self.free_queue.put(slot_id)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._submitted_at: dict[int, float] = {}
        self.abort_event = self.context.Event()
        self.shutdown_event = self.context.Event()
        self.failure_message = self.context.Array(ctypes.c_char, 1024, lock=True)
        self.active_games = 0
        self.completed_episodes_pending = 0

    _TENSOR_FIELDS = (
        "states", "request_ids", "policy_snapshots", "actor_ids", "submitted_ns"
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))

    def _raise_if_failed(self) -> None:
        if self.abort_event.is_set():
            with self.failure_message.get_lock():
                reason = bytes(self.failure_message.value).decode("utf-8", "replace")
            raise RuntimeError(
                f"async actor runtime failed: {reason or 'unknown worker failure'}"
            )
        if self.shutdown_event.is_set():
            raise RuntimeError("async actor runtime is shut down")

    def acquire(self, actor_id: int, timeout: float | None = None) -> int:
        self._raise_if_failed()
        timeout = self.request_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            self._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a free inference slot")
            try:
                slot_id = int(self.free_queue.get(timeout=min(0.05, remaining)))
                break
            except queue.Empty:
                continue
        if int(self.states[slot_id]) != SlotState.FREE:
            raise RuntimeError("free queue returned a non-FREE slot")
        self.states[slot_id] = int(SlotState.WRITING)
        self.actor_ids[slot_id] = actor_id
        return slot_id

    def submit(
        self, slot_id: int, *, request_id: int, policy_snapshot: int
    ) -> None:
        self._raise_if_failed()
        if int(self.states[slot_id]) != SlotState.WRITING:
            raise RuntimeError("only a WRITING slot may be submitted")
        count = int(self.slots.action_counts[slot_id])
        if count < 1 or not bool(self.slots.action_mask[slot_id, :count].any()):
            self.states[slot_id] = int(SlotState.FAILED)
            raise ValueError("request has zero legal actions")
        self.request_ids[slot_id] = request_id
        self.policy_snapshots[slot_id] = policy_snapshot
        self.states[slot_id] = int(SlotState.READY)
        self._submitted_at[slot_id] = time.monotonic()
        self.submitted_ns[slot_id] = time.monotonic_ns()
        self.ready_queue.put(slot_id)

    def claim_ready(self, max_items: int, wait_seconds: float = 0.0) -> list[RequestMetadata]:
        self._raise_if_failed()
        if max_items < 1:
            raise ValueError("max_items must be positive")
        slot_ids: list[int] = []
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while len(slot_ids) < max_items:
            timeout = max(0.0, deadline - time.monotonic()) if wait_seconds else 0.0
            try:
                slot_ids.append(int(self.ready_queue.get(timeout=timeout)))
            except queue.Empty:
                break
        metadata = []
        for slot_id in slot_ids:
            if int(self.states[slot_id]) != SlotState.READY:
                raise RuntimeError("ready queue returned a non-READY slot")
            self.states[slot_id] = int(SlotState.RUNNING)
            metadata.append(RequestMetadata(
                slot_id=slot_id,
                actor_id=int(self.actor_ids[slot_id]),
                request_id=int(self.request_ids[slot_id]),
                policy_snapshot=int(self.policy_snapshots[slot_id]),
                action_count=int(self.slots.action_counts[slot_id]),
                acting_role=int(self.slots.roles[slot_id]),
                submitted_ns=int(self.submitted_ns[slot_id]),
            ))
        return metadata

    def complete(self, slot_id: int) -> None:
        if int(self.states[slot_id]) != SlotState.RUNNING:
            raise RuntimeError("only a RUNNING slot may complete")
        self.states[slot_id] = int(SlotState.DONE)

    def wait_done(self, slot_id: int, request_id: int) -> None:
        deadline = time.monotonic() + self.request_timeout_seconds
        while time.monotonic() < deadline:
            self._raise_if_failed()
            state = SlotState(int(self.states[slot_id]))
            if state == SlotState.DONE:
                if int(self.request_ids[slot_id]) != request_id:
                    raise RuntimeError("inference response request_id mismatch")
                return
            if state in {SlotState.FAILED, SlotState.SHUTDOWN}:
                raise RuntimeError(f"inference request ended in state {state.name}")
            time.sleep(0.001)
        self.fail(f"request {request_id} timed out")
        raise TimeoutError(f"inference request {request_id} timed out")

    def release(self, slot_id: int) -> None:
        if int(self.states[slot_id]) != SlotState.DONE:
            raise RuntimeError("only a DONE slot may be released")
        self.states[slot_id] = int(SlotState.FREE)
        self._submitted_at.pop(slot_id, None)
        self.free_queue.put(slot_id)

    def fail(self, reason: str) -> None:
        message = (str(reason) or "unknown worker failure").encode("utf-8")[:1023]
        with self.failure_message.get_lock():
            if not self.failure_message.value:
                self.failure_message.value = message
        self.abort_event.set()
        for slot_id in range(len(self.states)):
            if int(self.states[slot_id]) not in {
                SlotState.FREE, SlotState.DONE, SlotState.SHUTDOWN
            }:
                self.states[slot_id] = int(SlotState.FAILED)

    def quiesce(self) -> dict[str, int]:
        self._raise_if_failed()
        counts = {
            state.name.lower(): int((self.states == int(state)).sum().item())
            for state in SlotState
        }
        if self.active_games or self.completed_episodes_pending:
            raise RuntimeError("cannot quiesce with active or uncommitted episodes")
        if counts["ready"] or counts["running"] or counts["writing"]:
            raise RuntimeError("cannot quiesce with in-flight inference requests")
        return counts

    def request_shutdown(self) -> None:
        """Publish shutdown before joins without closing shared queues."""
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        for slot_id in range(len(self.states)):
            self.states[slot_id] = int(SlotState.SHUTDOWN)

    def shutdown(self) -> None:
        self.request_shutdown()
        self.ready_queue.close()
        self.free_queue.close()


def async_actor_main(
    actor_id: int,
    task_queue,
    event_queue,
    coordinator: AsyncRequestCoordinator,
    replay_slots: SharedReplaySlots,
    *,
    seed: int,
    epsilon: float,
    max_steps: int,
    decision_config,
    ruleset,
    feature_schema_hash: str,
    policy_version: str,
    policy_step,
) -> None:
    """CPU-only actor entry point.  CUDA is never imported or initialized."""
    import random

    from douzero.env.env import Env
    from douzero.models_v2.batch import observation_to_model_inputs
    from douzero.models_v2.output import ModelOutput
    from douzero.observation.encode_v2 import get_obs_v2
    from douzero.training.decision_policy import select_action
    from douzero.training.v2_buffer import Episode, Transition

    rng = random.Random(None if seed == 0 else seed + actor_id)
    if seed:
        np.random.seed(seed + actor_id)
    request_id = actor_id << 48
    try:
        while True:
            if coordinator.shutdown_event.is_set():
                return
            coordinator._raise_if_failed()
            try:
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                return
            episode_id = int(task)
            snapshot = int(policy_step.value)
            event_queue.put(("started", actor_id, episode_id, snapshot))
            env = Env("adp", ruleset=ruleset)
            env.reset()
            episode = Episode(
                policy_version_at_start=policy_version,
                policy_step_at_start=snapshot,
            )
            for _ in range(max_steps):
                position = env._acting_player_position
                infoset = env.infoset
                legal_actions = infoset.legal_actions
                if len(legal_actions) == 1:
                    action_index = 0
                else:
                    obs = get_obs_v2(infoset, ruleset=ruleset)
                    if epsilon > 0 and rng.random() < epsilon:
                        action_index = rng.randrange(len(legal_actions))
                    else:
                        bundle = observation_to_model_inputs(obs)
                        slot_id = coordinator.acquire(actor_id)
                        coordinator.slots.write(slot_id, bundle)
                        request_id += 1
                        coordinator.submit(
                            slot_id, request_id=request_id,
                            policy_snapshot=snapshot,
                        )
                        coordinator.wait_done(slot_id, request_id)
                        count = int(coordinator.slots.action_counts[slot_id])
                        mask = coordinator.slots.action_mask[slot_id, :count].clone()
                        win = coordinator.slots.output_win[slot_id, :count].clone().unsqueeze(-1)
                        score_win = coordinator.slots.output_score_win[
                            slot_id, :count
                        ].clone().unsqueeze(-1)
                        score_loss = coordinator.slots.output_score_loss[
                            slot_id, :count
                        ].clone().unsqueeze(-1)
                        p_win = coordinator.slots.output_p_win[
                            slot_id, :count
                        ].clone().unsqueeze(-1)
                        score_mean = coordinator.slots.output_score[
                            slot_id, :count
                        ].clone().unsqueeze(-1)
                        output = ModelOutput(
                            win_logit=win,
                            score_if_win=score_win,
                            score_if_loss=score_loss,
                            p_win=p_win,
                            score_mean=score_mean,
                            action_mask=mask,
                        )
                        action_index = select_action(output, decision_config)
                        coordinator.release(slot_id)
                    episode.transitions.append(Transition(
                        obs=obs,
                        action_index=action_index,
                        position=position,
                        trace_index=len(episode.action_trace),
                        policy_id=policy_version,
                        policy_version=policy_version,
                        policy_step=snapshot,
                    ))
                action = legal_actions[action_index]
                episode.action_trace.append((position, tuple(sorted(action))))
                _obs, _reward, done, info = env.step(action)
                if done:
                    episode.terminal_result = info or {}
                    break
            else:
                raise RuntimeError(f"actor episode exceeded max_steps={max_steps}")
            episode.label_from_terminal()
            for transition in episode.transitions:
                replay_slots.write_transition(
                    transition,
                    observation_to_model_inputs(transition.obs),
                    snapshot,
                    coordinator.request_timeout_seconds,
                    coordinator.abort_event,
                    coordinator.shutdown_event,
                )
            team = episode.terminal_result.get("winner_team", "landlord")
            event_queue.put((
                "completed", actor_id, episode_id, len(episode.transitions),
                0 if team == "landlord" else 1, snapshot,
                len(episode.action_trace),
            ))
    except BaseException as exc:
        if coordinator.shutdown_event.is_set() and not coordinator.abort_event.is_set():
            event_queue.put(("stopped", actor_id))
            return
        message = f"actor {actor_id}: {type(exc).__name__}: {exc}"
        coordinator.fail(message)
        event_queue.put(("failed", actor_id, message))
        raise
