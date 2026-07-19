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
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum

import torch
import numpy as np

from douzero.models_v2.batch import BatchedModelInputBundle, ModelInputBundle
from douzero.models_v2.config import SUPPORTED_ROLES
from douzero.observation.schema import (
    action_width,
    context_width,
    history_token_width,
    state_width,
)
from douzero.training.v2_buffer import compact_model_input_shapes


# Inference has a different padding/launch tradeoff from learner replay.  The
# learner keeps its fine-grained action buckets, while centralized inference
# deliberately uses two broad buckets so a small actor pool does not fragment
# into singleton GPU launches.
INFERENCE_ACTION_BUCKET_LIMITS: tuple[int, ...] = (64, 512)


def inference_action_count_bucket(action_count: int) -> int | str:
    """Return the coarse centralized-inference padding bucket."""
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise TypeError("action_count must be an int")
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    for limit in INFERENCE_ACTION_BUCKET_LIMITS:
        if action_count <= limit:
            return limit
    return "overflow"


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
    def grouping_key(self) -> tuple[int, int | str]:
        return (
            self.policy_snapshot,
            inference_action_count_bucket(self.action_count),
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
        # Keep all response heads adjacent per action.  The inference service
        # publishes one contiguous row copy; compatibility views retain the
        # named fields used by the actor and protocol tests.
        self.output_values = shared((num_slots, max_actions, 5))
        self._bind_output_views()
        self.action_counts = shared((num_slots,), torch.int32)
        self.roles = shared((num_slots,), torch.int64)

    _TENSOR_FIELDS = (
        "state_cards", "state_flat", "context_cards", "context_flat",
        "history", "history_padding", "actions", "action_mask",
        "output_values", "action_counts", "roles",
    )

    _OUTPUT_VIEW_FIELDS = (
        "output_win", "output_score_win", "output_score_loss",
        "output_p_win", "output_score",
    )

    def _bind_output_views(self) -> None:
        self.output_win = self.output_values[..., 0]
        self.output_score_win = self.output_values[..., 1]
        self.output_score_loss = self.output_values[..., 2]
        self.output_p_win = self.output_values[..., 3]
        self.output_score = self.output_values[..., 4]

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._TENSOR_FIELDS + self._OUTPUT_VIEW_FIELDS:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, owner, (shape, dtype) in zip(
            self._TENSOR_FIELDS, self._shared_owners, self._shared_specs
        ):
            setattr(self, name, _restore_shared_tensor(owner, shape, dtype))
        self._bind_output_views()

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
        # Only mask state can make stale padded rows observable.  Clearing the
        # previous live range avoids touching the full 4096-row slot on every
        # decision; action rows in the new live range are overwritten below.
        previous_count = int(self.action_counts[slot_id])
        if previous_count:
            self.action_mask[slot_id, :previous_count].zero_()
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


class PinnedObservationBatchStager:
    """Reusable shared-SoA to pinned-batch staging for one action capacity."""

    def __init__(
        self,
        slots: SharedObservationSlots,
        *,
        max_batch_size: int,
        action_capacity: int,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if action_capacity < 1 or action_capacity > slots.max_actions:
            raise ValueError("action_capacity is outside the shared slot range")
        self.slots = slots
        self.max_batch_size = int(max_batch_size)
        self.action_capacity = int(action_capacity)

        def pinned(shape, dtype):
            return torch.empty(shape, dtype=dtype, pin_memory=True)

        batch = self.max_batch_size
        self.state_cards = pinned(
            (batch, *slots.state_cards.shape[1:]), slots.state_cards.dtype
        )
        self.state_flat = pinned(
            (batch, *slots.state_flat.shape[1:]), slots.state_flat.dtype
        )
        self.context_cards = pinned(
            (batch, *slots.context_cards.shape[1:]), slots.context_cards.dtype
        )
        self.context_flat = pinned(
            (batch, *slots.context_flat.shape[1:]), slots.context_flat.dtype
        )
        self.history = pinned(
            (batch, *slots.history.shape[1:]), slots.history.dtype
        )
        self.history_padding = pinned(
            (batch, *slots.history_padding.shape[1:]), slots.history_padding.dtype
        )
        self.actions = pinned(
            (batch, action_capacity, slots.actions.shape[-1]), slots.actions.dtype
        )
        self.action_mask = pinned(
            (batch, action_capacity), slots.action_mask.dtype
        )
        self.roles = pinned((batch,), slots.roles.dtype)
        self.output_values = pinned((batch, action_capacity, 5), torch.float32)

    @staticmethod
    def _gather(source: torch.Tensor, indices: torch.Tensor, destination) -> None:
        torch.index_select(source, 0, indices, out=destination)

    def gather_slots(self, slot_ids: list[int]) -> int:
        """Copy shared slot rows directly into the reusable pinned buffers."""
        batch_size = len(slot_ids)
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("staged request count is outside batch capacity")
        indices = torch.tensor(slot_ids, dtype=torch.long)
        counts = self.slots.action_counts.index_select(0, indices)
        if bool((counts < 1).any()) or bool((counts > self.action_capacity).any()):
            raise ValueError("request action count is outside staging capacity")

        self._gather(self.slots.state_cards, indices, self.state_cards[:batch_size])
        self._gather(self.slots.state_flat, indices, self.state_flat[:batch_size])
        self._gather(
            self.slots.context_cards, indices, self.context_cards[:batch_size]
        )
        self._gather(self.slots.context_flat, indices, self.context_flat[:batch_size])
        self._gather(self.slots.history, indices, self.history[:batch_size])
        self._gather(
            self.slots.history_padding,
            indices,
            self.history_padding[:batch_size],
        )
        self._gather(
            self.slots.actions[:, :self.action_capacity],
            indices,
            self.actions[:batch_size],
        )
        self._gather(
            self.slots.action_mask[:, :self.action_capacity],
            indices,
            self.action_mask[:batch_size],
        )
        self._gather(self.slots.roles, indices, self.roles[:batch_size])
        return batch_size

    def batch_view(
        self,
        batch_size: int,
        feature_schema_hash: str,
    ) -> BatchedModelInputBundle:
        """Build the model-facing views over an already gathered batch."""
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("staged request count is outside batch capacity")
        state_cards = self.state_cards[:batch_size]
        context_cards = self.context_cards[:batch_size]
        return BatchedModelInputBundle(
            state_card_vectors=tuple(
                state_cards[:, index] for index in range(state_cards.shape[1])
            ),
            state_context_flat=self.state_flat[:batch_size],
            context_card_vectors=tuple(
                context_cards[:, index] for index in range(context_cards.shape[1])
            ),
            context_flat=self.context_flat[:batch_size],
            history_tokens=self.history[:batch_size],
            history_key_padding_mask=self.history_padding[:batch_size],
            action_features=self.actions[:batch_size],
            action_mask=self.action_mask[:batch_size],
            acting_role=self.roles[:batch_size],
            chosen_action_index=None,
            feature_schema_hashes=(feature_schema_hash,) * batch_size,
        )

    def stage_inputs(
        self,
        slot_ids: list[int],
        feature_schema_hash: str,
    ) -> BatchedModelInputBundle:
        """Gather shared rows and return their model-facing pinned views."""
        batch_size = self.gather_slots(slot_ids)
        return self.batch_view(batch_size, feature_schema_hash)

    def stage_outputs(self, values: torch.Tensor) -> torch.Tensor:
        expected = (
            values.shape[0], self.action_capacity, self.output_values.shape[-1]
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"packed inference output must have shape {expected}, "
                f"got {tuple(values.shape)}"
            )
        destination = self.output_values[:values.shape[0]]
        destination.copy_(values, non_blocking=True)
        return destination


class PendingRequestScheduler:
    """FIFO request groups retained across service iterations."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        target_batch_size: int = 4,
        max_delay_seconds: float = 0.002,
    ) -> None:
        if max_batch_size < 1 or target_batch_size < 1:
            raise ValueError("scheduler batch sizes must be positive")
        if max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be non-negative")
        self.max_batch_size = int(max_batch_size)
        self.target_batch_size = min(int(target_batch_size), self.max_batch_size)
        self.max_delay_seconds = float(max_delay_seconds)
        self._groups: dict[tuple[int, int | str], deque[RequestMetadata]] = (
            defaultdict(deque)
        )

    def add(self, requests: list[RequestMetadata]) -> None:
        for request in requests:
            self._groups[request.grouping_key].append(request)

    @property
    def pending_count(self) -> int:
        return sum(len(group) for group in self._groups.values())

    def pop_ready(
        self, *, now_ns: int | None = None
    ) -> tuple[tuple[int, int | str], list[RequestMetadata]] | None:
        if not self._groups:
            return None
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        eligible = []
        for key, group in self._groups.items():
            oldest_wait = max(0.0, (now_ns - group[0].submitted_ns) / 1e9)
            if len(group) >= self.target_batch_size or (
                oldest_wait >= self.max_delay_seconds
            ):
                eligible.append((key, group, oldest_wait))
        if not eligible:
            return None
        # Prefer a launch that fills the GPU, then the oldest request.  FIFO is
        # preserved within every (snapshot, inference bucket) group.
        key, group, _ = max(
            eligible,
            key=lambda item: (
                min(len(item[1]), self.max_batch_size), item[2]
            ),
        )
        requests = [
            group.popleft()
            for _ in range(min(len(group), self.max_batch_size))
        ]
        if not group:
            del self._groups[key]
        return key, requests


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
        # Shared result tensors use RawArray storage, so publishing DONE in a
        # separate RawArray is not a synchronization boundary.  A per-slot
        # Event provides the release/acquire hand-off: the coordinator writes
        # every result tensor before set(), and the actor waits before reading.
        self.response_events = [
            self.context.Event() for _ in range(num_slots)
        ]
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
        self.response_events[slot_id].clear()
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
        wait_seconds = max(0.0, wait_seconds)
        # Wait for the first request, then give its peers a complete
        # coalescing window.  Starting the deadline before the first request
        # arrives leaves almost no batching opportunity when the queue was
        # initially empty.
        try:
            if wait_seconds:
                slot_ids.append(int(self.ready_queue.get(timeout=wait_seconds)))
            else:
                slot_ids.append(int(self.ready_queue.get_nowait()))
        except queue.Empty:
            return []
        deadline = time.monotonic() + wait_seconds
        while len(slot_ids) < max_items:
            timeout = max(0.0, deadline - time.monotonic()) if wait_seconds else 0.0
            try:
                if wait_seconds:
                    slot_ids.append(int(self.ready_queue.get(timeout=timeout)))
                else:
                    slot_ids.append(int(self.ready_queue.get_nowait()))
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
        self.response_events[slot_id].set()

    def wait_done(self, slot_id: int, request_id: int) -> None:
        deadline = time.monotonic() + self.request_timeout_seconds
        response_event = self.response_events[slot_id]
        while True:
            self._raise_if_failed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not response_event.wait(timeout=min(0.05, remaining)):
                continue
            # Event.wait() is the acquire side of the result publication
            # barrier.  Only inspect state and RawArray-backed outputs after it
            # has observed complete(), fail(), or shutdown().
            self._raise_if_failed()
            state = SlotState(int(self.states[slot_id]))
            if state == SlotState.DONE:
                if int(self.request_ids[slot_id]) != request_id:
                    raise RuntimeError("inference response request_id mismatch")
                return
            if state in {SlotState.FAILED, SlotState.SHUTDOWN}:
                raise RuntimeError(f"inference request ended in state {state.name}")
            raise RuntimeError(
                f"inference response event published unexpected state {state.name}"
            )
        self.fail(f"request {request_id} timed out")
        raise TimeoutError(f"inference request {request_id} timed out")

    def release(self, slot_id: int) -> None:
        if int(self.states[slot_id]) != SlotState.DONE:
            raise RuntimeError("only a DONE slot may be released")
        self.states[slot_id] = int(SlotState.FREE)
        self.response_events[slot_id].clear()
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
            self.response_events[slot_id].set()

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
            self.response_events[slot_id].set()

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
    environment_seed: int,
    action_rng_seed: int,
    epsilon: float,
    max_steps: int,
    decision_config,
    ruleset,
    feature_schema_hash: str,
    policy_version: str,
    policy_step,
    games_per_actor: int,
) -> None:
    """CPU-only interleaved-game actor. CUDA is never initialized here."""
    import random

    from douzero.env.env import Env
    from douzero.models_v2.batch import observation_to_model_inputs
    from douzero.models_v2.output import ModelOutput
    from douzero.observation.encode_v2 import get_obs_v2
    from douzero.training.decision_policy import select_action
    from douzero.training.v2_buffer import Episode, Transition

    rng = (
        random.Random()
        if action_rng_seed == 0
        else random.Random(action_rng_seed + actor_id)
    )
    if environment_seed:
        np.random.seed((environment_seed + actor_id) % (1 << 32))
    if games_per_actor < 1:
        raise ValueError("games_per_actor must be positive")
    request_id = actor_id << 48

    def start_game(task):
        episode_id = int(task)
        snapshot = int(policy_step.value)
        event_queue.put(("started", actor_id, episode_id, snapshot))
        env = Env("adp", ruleset=ruleset)
        env.reset()
        return {
            "episode_id": episode_id,
            "snapshot": snapshot,
            "env": env,
            "episode": Episode(
                policy_version_at_start=policy_version,
                policy_step_at_start=snapshot,
            ),
            "steps": 0,
            "pending": None,
        }

    def finish_game(game) -> None:
        episode = game["episode"]
        episode.label_from_terminal()
        for transition in episode.transitions:
            replay_slots.write_transition(
                transition,
                observation_to_model_inputs(transition.obs),
                game["snapshot"],
                coordinator.request_timeout_seconds,
                coordinator.abort_event,
                coordinator.shutdown_event,
            )
        team = episode.terminal_result.get("winner_team", "landlord")
        event_queue.put((
            "completed", actor_id, game["episode_id"], len(episode.transitions),
            0 if team == "landlord" else 1, game["snapshot"],
            len(episode.action_trace),
        ))

    def apply_action(game, action_index, obs, position, legal_actions) -> bool:
        episode = game["episode"]
        if obs is not None:
            episode.transitions.append(Transition(
                obs=obs,
                action_index=action_index,
                position=position,
                trace_index=len(episode.action_trace),
                policy_id=policy_version,
                policy_version=policy_version,
                policy_step=game["snapshot"],
            ))
        action = legal_actions[action_index]
        episode.action_trace.append((position, tuple(sorted(action))))
        _obs, _reward, done, info = game["env"].step(action)
        game["steps"] += 1
        if done:
            episode.terminal_result = info or {}
            finish_game(game)
            return True
        if game["steps"] >= max_steps:
            raise RuntimeError(f"actor episode exceeded max_steps={max_steps}")
        return False

    def advance_until_request_or_done(game) -> bool:
        nonlocal request_id
        while True:
            position = game["env"]._acting_player_position
            infoset = game["env"].infoset
            legal_actions = infoset.legal_actions
            if len(legal_actions) == 1:
                if apply_action(game, 0, None, position, legal_actions):
                    return True
                continue

            obs = get_obs_v2(infoset, ruleset=ruleset)
            if epsilon > 0 and rng.random() < epsilon:
                action_index = rng.randrange(len(legal_actions))
                if apply_action(
                    game, action_index, obs, position, legal_actions
                ):
                    return True
                continue

            bundle = observation_to_model_inputs(obs)
            slot_id = coordinator.acquire(actor_id)
            coordinator.slots.write(slot_id, bundle)
            request_id += 1
            coordinator.submit(
                slot_id,
                request_id=request_id,
                policy_snapshot=game["snapshot"],
            )
            game["pending"] = (
                slot_id, request_id, obs, position, legal_actions
            )
            return False

    def resolve_request(game):
        slot_id, pending_id, obs, position, legal_actions = game["pending"]
        coordinator.wait_done(slot_id, pending_id)
        count = int(coordinator.slots.action_counts[slot_id])
        mask = coordinator.slots.action_mask[slot_id, :count].clone()
        packed = coordinator.slots.output_values[slot_id, :count].clone()
        coordinator.release(slot_id)
        game["pending"] = None
        output = ModelOutput(
            win_logit=packed[:, 0:1],
            score_if_win=packed[:, 1:2],
            score_if_loss=packed[:, 2:3],
            p_win=packed[:, 3:4],
            score_mean=packed[:, 4:5],
            action_mask=mask,
        )
        return (
            select_action(output, decision_config), obs, position, legal_actions
        )

    try:
        active = []
        while True:
            if coordinator.shutdown_event.is_set():
                return
            coordinator._raise_if_failed()
            while len(active) < games_per_actor:
                try:
                    task = (
                        task_queue.get(timeout=0.1)
                        if not active else task_queue.get_nowait()
                    )
                except queue.Empty:
                    break
                if task is None:
                    return
                active.append(start_game(task))
            if not active:
                continue

            completed = []
            for game in tuple(active):
                if advance_until_request_or_done(game):
                    completed.append(game)
            for game in completed:
                active.remove(game)

            pending_games = [
                game for game in active if game["pending"] is not None
            ]
            # All games submit before this actor waits.  Responses are cloned
            # and slots released for the whole wave before terminal replay
            # publication can block on replay capacity.
            resolved = [
                (game, *resolve_request(game)) for game in pending_games
            ]
            for game, action_index, obs, position, legal_actions in resolved:
                if apply_action(
                    game, action_index, obs, position, legal_actions
                ):
                    active.remove(game)
    except BaseException as exc:
        if coordinator.shutdown_event.is_set() and not coordinator.abort_event.is_set():
            event_queue.put(("stopped", actor_id))
            return
        message = f"actor {actor_id}: {type(exc).__name__}: {exc}"
        coordinator.fail(message)
        event_queue.put(("failed", actor_id, message))
        raise
