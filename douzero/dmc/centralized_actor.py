"""Centralized CUDA inference primitives for interleaved Legacy V1 actors."""

from __future__ import annotations

import enum
import math
import queue
import time
import traceback
from dataclasses import dataclass, replace

import torch

from .models_factorized import LegacyFactorizedModel


POSITIONS = ("landlord", "landlord_up", "landlord_down")


class RequestState(str, enum.Enum):
    FREE = "free"
    PREPARED = "prepared"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InferenceRequest:
    actor_id: int
    env_slot: int
    generation: int
    request_id: int
    policy_slot: int
    policy_version: int
    position: str
    action_count: int
    queued_ns: int

    @property
    def storage_slot(self) -> int:
        raise AttributeError("storage_slot depends on envs_per_actor")


class PendingRequestScheduler:
    """Actor-local request state machine with strict response correlation."""

    def __init__(self, actor_id: int, env_slots: int) -> None:
        if env_slots < 1:
            raise ValueError("env_slots must be positive")
        self.actor_id = actor_id
        self._next_id = 1
        self._generation = [0] * env_slots
        self._states = [RequestState.FREE] * env_slots
        self._requests: list[InferenceRequest | None] = [None] * env_slots

    def prepare(self, env_slot: int, *, policy_slot: int, policy_version: int,
                position: str, action_count: int) -> InferenceRequest:
        if self._states[env_slot] not in {
            RequestState.FREE, RequestState.CONSUMED, RequestState.CANCELLED
        }:
            raise RuntimeError(f"environment slot {env_slot} already has a request")
        self._generation[env_slot] += 1
        request = InferenceRequest(
            actor_id=self.actor_id,
            env_slot=env_slot,
            generation=self._generation[env_slot],
            request_id=self._next_id,
            policy_slot=policy_slot,
            policy_version=policy_version,
            position=position,
            action_count=action_count,
            queued_ns=time.perf_counter_ns(),
        )
        self._next_id += 1
        self._requests[env_slot] = request
        self._states[env_slot] = RequestState.PREPARED
        return request

    def mark_queued(self, request: InferenceRequest) -> None:
        self._expect(request, RequestState.PREPARED)
        self._states[request.env_slot] = RequestState.QUEUED

    def consume(self, response) -> tuple[InferenceRequest, int] | None:
        status, actor_id, env_slot, generation, request_id, payload = response
        if status == "shutdown":
            self.cancel_all()
            return None
        if status == "error":
            self.cancel_all()
            raise RuntimeError(f"centralized actor inference failed: {payload}")
        if status != "ok" or actor_id != self.actor_id:
            raise RuntimeError("invalid centralized inference response identity")
        if env_slot < 0 or env_slot >= len(self._requests):
            raise RuntimeError("invalid centralized inference response slot")
        request = self._requests[env_slot]
        if request is None or (request.generation, request.request_id) != (
            generation, request_id
        ):
            raise RuntimeError("stale or cross-slot centralized inference response")
        if self._states[env_slot] != RequestState.QUEUED:
            raise RuntimeError("duplicate centralized inference response")
        self._states[env_slot] = RequestState.CONSUMED
        return request, int(payload)

    def cancel_all(self) -> None:
        for index, state in enumerate(self._states):
            if state in {RequestState.PREPARED, RequestState.QUEUED}:
                self._states[index] = RequestState.CANCELLED

    @property
    def pending(self) -> int:
        return sum(state in {RequestState.PREPARED, RequestState.QUEUED}
                   for state in self._states)

    def state(self, env_slot: int) -> RequestState:
        return self._states[env_slot]

    def _expect(self, request: InferenceRequest, state: RequestState) -> None:
        if self._requests[request.env_slot] != request or self._states[request.env_slot] != state:
            raise RuntimeError("invalid centralized inference request transition")


class CentralQueuePressure:
    """Small shared queue-pressure summary used by actors and learners."""

    def __init__(self, mp_context) -> None:
        self._lock = mp_context.Lock()
        self._depth = mp_context.Value("i", 0, lock=False)
        self._oldest_ns = mp_context.Value("q", 0, lock=False)

    def enqueued(self, queued_ns: int) -> None:
        with self._lock:
            self._depth.value += 1
            if not self._oldest_ns.value:
                self._oldest_ns.value = queued_ns

    def dequeued(self) -> None:
        with self._lock:
            self._depth.value = max(0, self._depth.value - 1)
            if self._depth.value == 0:
                self._oldest_ns.value = 0

    def snapshot(self) -> tuple[int, float]:
        with self._lock:
            depth = int(self._depth.value)
            oldest = int(self._oldest_ns.value)
        age_ms = ((time.perf_counter_ns() - oldest) / 1e6
                  if depth and oldest else 0.0)
        return depth, age_ms


class CentralizedInferenceSlots:
    """One preallocated shared request slot per actor environment."""

    def __init__(self, num_actors: int, max_actions: int,
                 envs_per_actor: int = 1) -> None:
        if num_actors < 1 or envs_per_actor < 1:
            raise ValueError("actor and environment counts must be positive")
        self.max_actions = max_actions
        self.envs_per_actor = envs_per_actor
        count = num_actors * envs_per_actor
        self.z = torch.empty(count, 5, 162, dtype=torch.int8).share_memory_()
        self.x_state = torch.empty(count, 430, dtype=torch.int8).share_memory_()
        self.x_action = torch.empty(
            count, max_actions, 54, dtype=torch.int8
        ).share_memory_()

    def index(self, actor_id: int, env_slot: int = 0) -> int:
        if env_slot < 0 or env_slot >= self.envs_per_actor:
            raise ValueError("invalid centralized environment slot")
        index = actor_id * self.envs_per_actor + env_slot
        if index < 0 or index >= self.z.shape[0]:
            raise ValueError("invalid centralized actor id")
        return index

    def write(self, actor_id, position, z_single, x_state_single, x_action,
              env_slot=0):
        count = x_action.shape[0]
        if count > self.max_actions:
            raise ValueError("central inference request exceeds slot capacity")
        index = self.index(actor_id, env_slot)
        state_width = 319 if position == "landlord" else 430
        self.z[index].copy_(z_single[0].to(dtype=torch.int8))
        self.x_state[index, :state_width].copy_(
            x_state_single[0].to(dtype=torch.int8)
        )
        self.x_action[index, :count].copy_(x_action.to(dtype=torch.int8))


class PolicyCopyState:
    """Per-policy-slot async copy state, also usable in CPU-only tests."""

    def __init__(self) -> None:
        self.loaded_version = -1
        self.loading_version: int | None = None
        self.ready_event = None

    def begin(self, version: int, event=None) -> None:
        if self.loading_version is not None:
            raise RuntimeError("policy copy already in progress")
        self.loading_version = version
        self.ready_event = event

    def finish(self, version: int) -> None:
        if self.loading_version != version:
            raise RuntimeError("policy copy version changed while loading")
        self.loaded_version = version
        self.loading_version = None

    def fail(self) -> None:
        self.loading_version = None
        self.ready_event = None


def _copy_policy_slot(gpu_policy, policy_pool, policy_slot):
    source = policy_pool.models[policy_slot]
    with torch.no_grad():
        for position in POSITIONS:
            target_state = gpu_policy.get_model(position).state_dict(keep_vars=True)
            source_state = source.get_model(position).state_dict(keep_vars=True)
            if target_state.keys() != source_state.keys():
                raise RuntimeError(f"policy state keys changed for {position}")
            for key, target in target_state.items():
                if target.shape != source_state[key].shape:
                    raise RuntimeError(f"policy state shape changed for {position}.{key}")
                target.copy_(source_state[key], non_blocking=True)


def _priority_stream(device, enabled):
    if not enabled:
        return torch.cuda.Stream(device=device), False
    try:
        _least, greatest = torch.cuda.get_stream_priority_range()
        return torch.cuda.Stream(device=device, priority=greatest), True
    except (AttributeError, RuntimeError):
        return torch.cuda.Stream(device=device), False


def _put_response(response_queues, request, status, payload):
    response_queues[request.actor_id].put((
        status, request.actor_id, request.env_slot, request.generation,
        request.request_id, payload,
    ))


def _compatible_key(request, max_actions):
    # Broad power-of-two buckets bound packed staging variance without requiring
    # equal legal-action counts.
    bucket = min(max_actions, max(64, 2 ** math.ceil(math.log2(request.action_count))))
    return (request.policy_slot, request.policy_version, request.position, bucket)


def centralized_inference_loop(
    device, policy_pool, slots, request_queue, response_queues, stop_event,
    min_microbatch, target_microbatch, max_microbatch, max_delay_ms,
    max_queued_requests=128, use_stream_priority=True,
    async_policy_copy=True, metric_store=None,
    queue_pressure=None,
):
    """Adaptively batch compatible requests and return correlated responses."""
    try:
        torch.set_num_threads(1)
        cuda_device = torch.device("cuda:" + str(device))
        torch.cuda.set_device(cuda_device)
        policies = [LegacyFactorizedModel(device=device) for _ in policy_pool.models]
        states = [PolicyCopyState() for _ in policies]
        for policy in policies:
            policy.eval()
        inference_stream, priority_active = _priority_stream(
            cuda_device, use_stream_priority
        )
        copy_stream = torch.cuda.Stream(device=cuda_device)

        pinned_z = torch.empty(
            max_microbatch, 5, 162, dtype=torch.float32, pin_memory=True
        )
        pinned_state = torch.empty(
            max_microbatch, 430, dtype=torch.float32, pin_memory=True
        )
        pinned_action = torch.empty(
            max_microbatch * slots.max_actions, 54,
            dtype=torch.float32, pin_memory=True,
        )
        pending = []
        accepting = True
        while accepting and not stop_event.is_set():
            if not pending:
                try:
                    first = request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if first is None:
                    break
                pending.append(replace(first, queued_ns=time.perf_counter_ns()))
                if queue_pressure is not None:
                    queue_pressure.dequeued()

            oldest_ns = min(request.queued_ns for request in pending)
            deadline_ns = oldest_ns + int(max_delay_ms * 1e6)
            while len(pending) < max_queued_requests:
                group_sizes = {}
                for request in pending:
                    key = _compatible_key(request, slots.max_actions)
                    group_sizes[key] = group_sizes.get(key, 0) + 1
                now_ns = time.perf_counter_ns()
                if (max(group_sizes.values()) >= target_microbatch
                        or (len(pending) >= max_microbatch
                            and max(group_sizes.values()) >= min_microbatch)
                        or now_ns >= deadline_ns):
                    break
                timeout = max(0.0, (deadline_ns - now_ns) / 1e9)
                try:
                    item = request_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    accepting = False
                    break
                pending.append(replace(item, queued_ns=time.perf_counter_ns()))
                if queue_pressure is not None:
                    queue_pressure.dequeued()

            groups = {}
            for request in pending:
                groups.setdefault(
                    _compatible_key(request, slots.max_actions), []
                ).append(request)
            ready = [
                (key, group) for key, group in groups.items()
                if len(group) >= target_microbatch
            ]
            if ready:
                selected_key, selected_group = max(
                    ready, key=lambda item: len(item[1])
                )
            else:
                oldest_request = min(pending, key=lambda item: item.queued_ns)
                selected_key = _compatible_key(oldest_request, slots.max_actions)
                selected_group = groups[selected_key]
            requests = selected_group[:max_microbatch]
            selected_ids = {id(request) for request in requests}
            pending = [request for request in pending if id(request) not in selected_ids]
            groups = {selected_key: requests}

            for (policy_slot, policy_version, position, _bucket), group in groups.items():
                state = states[policy_slot]
                copying = False
                if state.loaded_version != policy_version:
                    try:
                        event = torch.cuda.Event()
                        state.begin(policy_version, event)
                        stream = copy_stream if async_policy_copy else inference_stream
                        with torch.cuda.stream(stream):
                            _copy_policy_slot(policies[policy_slot], policy_pool, policy_slot)
                            event.record(stream)
                        inference_stream.wait_event(event)
                        copying = True
                    except BaseException:
                        state.fail()
                        raise

                state_width = 319 if position == "landlord" else 430
                counts = [request.action_count for request in group]
                action_total = sum(counts)
                batch_started_ns = time.perf_counter_ns()
                for batch_index, request in enumerate(group):
                    index = slots.index(request.actor_id, request.env_slot)
                    pinned_z[batch_index].copy_(slots.z[index])
                    pinned_state[batch_index, :state_width].copy_(
                        slots.x_state[index, :state_width]
                    )
                offset = 0
                for request in group:
                    index = slots.index(request.actor_id, request.env_slot)
                    count = request.action_count
                    pinned_action[offset:offset + count].copy_(
                        slots.x_action[index, :count]
                    )
                    offset += count
                with torch.cuda.stream(inference_stream):
                    timing_events = None
                    if metric_store is not None:
                        timing_events = [
                            torch.cuda.Event(enable_timing=True) for _ in range(3)
                        ]
                        timing_events[0].record(inference_stream)
                    z = pinned_z[:len(group)].to(cuda_device, non_blocking=True)
                    actor_state = pinned_state[:len(group), :state_width].to(
                        cuda_device, non_blocking=True
                    )
                    actions = pinned_action[:action_total].to(
                        cuda_device, non_blocking=True
                    )
                    if timing_events is not None:
                        timing_events[1].record(inference_stream)
                    with torch.inference_mode():
                        indices = policies[policy_slot].get_model(
                            position
                        ).select_actions_packed(z, actor_state, actions, counts)
                    if timing_events is not None:
                        timing_events[2].record(inference_stream)
                    indices_cpu = indices.to("cpu", non_blocking=False).tolist()
                if copying:
                    # The blocking D2H above proves the inference stream has
                    # observed the complete copy before version publication.
                    state.finish(policy_version)
                completed_ns = time.perf_counter_ns()
                for request, action_index in zip(group, indices_cpu):
                    _put_response(response_queues, request, "ok", int(action_index))
                if metric_store is not None:
                    h2d_ns = int(
                        timing_events[0].elapsed_time(timing_events[1]) * 1e6
                    )
                    forward_ns = int(
                        timing_events[1].elapsed_time(timing_events[2]) * 1e6
                    )
                    metric_store.add_central(
                        microbatch_size=len(group), legal_actions=action_total,
                        queue_wait_ns=[batch_started_ns - r.queued_ns for r in group],
                        batching_wait_ns=batch_started_ns - oldest_ns,
                        h2d_ns=h2d_ns, forward_ns=forward_ns,
                        response_ns=completed_ns - batch_started_ns,
                        stream_priority_active=priority_active,
                    )
        for request in pending:
            _put_response(response_queues, request, "shutdown", "inference stopping")
    except BaseException as exc:
        stop_event.set()
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        for actor_id, response_queue in enumerate(response_queues):
            response_queue.put(("error", actor_id, -1, -1, -1, detail))
        raise
