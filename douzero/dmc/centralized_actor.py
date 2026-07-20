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
    actor_prepared_ns: int
    actor_enqueue_started_ns: int = 0
    actor_enqueued_ns: int = 0
    server_received_ns: int = 0

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
        self.last_response_sent_ns = 0

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
            actor_prepared_ns=time.perf_counter_ns(),
        )
        self._next_id += 1
        self._requests[env_slot] = request
        self._states[env_slot] = RequestState.PREPARED
        return request

    def begin_enqueue(self, request: InferenceRequest) -> InferenceRequest:
        self._expect(request, RequestState.PREPARED)
        updated = replace(
            request, actor_enqueue_started_ns=time.perf_counter_ns()
        )
        self._requests[request.env_slot] = updated
        return updated

    def mark_queued(self, request: InferenceRequest,
                    enqueued_ns: int | None = None) -> InferenceRequest:
        self._expect(request, RequestState.PREPARED)
        updated = (
            request if enqueued_ns is None
            else replace(request, actor_enqueued_ns=enqueued_ns)
        )
        self._requests[request.env_slot] = updated
        self._states[request.env_slot] = RequestState.QUEUED
        return updated

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
        if isinstance(payload, tuple):
            action, response_sent_ns = payload
            self.last_response_sent_ns = int(response_sent_ns)
        else:
            action = payload
            self.last_response_sent_ns = 0
        return request, int(action)

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
    """Bounded authoritative request lifecycle and service-rate snapshot."""

    FREE, INGRESS, LOCAL_PENDING, EXECUTING = range(4)

    def __init__(self, mp_context, request_slots: int = 1024) -> None:
        if request_slots < 1:
            raise ValueError("request_slots must be positive")
        self._condition = mp_context.Condition(mp_context.RLock())
        self._states = mp_context.Array("b", request_slots, lock=False)
        self._actor_enqueue_ns = mp_context.Array(
            "q", request_slots, lock=False
        )
        self._server_receive_ns = mp_context.Array(
            "q", request_slots, lock=False
        )
        self._ingress = mp_context.Value("i", 0, lock=False)
        self._local = mp_context.Value("i", 0, lock=False)
        self._executing = mp_context.Value("i", 0, lock=False)
        self._oldest_actor_ns = mp_context.Value("q", 0, lock=False)
        self._oldest_server_ns = mp_context.Value("q", 0, lock=False)
        self._last_completed_ns = mp_context.Value("q", 0, lock=False)
        self._service_rate = mp_context.Value("d", 0.0, lock=False)
        self._valid = mp_context.Value("b", 1, lock=False)

    def _refresh_locked(self) -> None:
        ingress = local = executing = 0
        oldest_actor = oldest_server = 0
        for index, state in enumerate(self._states):
            if state == self.FREE:
                continue
            if state == self.INGRESS:
                ingress += 1
            elif state == self.LOCAL_PENDING:
                local += 1
            elif state == self.EXECUTING:
                executing += 1
            actor_ns = int(self._actor_enqueue_ns[index])
            server_ns = int(self._server_receive_ns[index])
            if actor_ns and (not oldest_actor or actor_ns < oldest_actor):
                oldest_actor = actor_ns
            if server_ns and (not oldest_server or server_ns < oldest_server):
                oldest_server = server_ns
        self._ingress.value = ingress
        self._local.value = local
        self._executing.value = executing
        self._oldest_actor_ns.value = oldest_actor
        self._oldest_server_ns.value = oldest_server

    def begin_actor_enqueue(self, storage_slot: int, started_ns: int) -> None:
        with self._condition:
            if self._states[storage_slot] != self.FREE:
                raise RuntimeError("central request slot is already active")
            self._actor_enqueue_ns[storage_slot] = started_ns
            self._server_receive_ns[storage_slot] = 0
            self._states[storage_slot] = self.INGRESS
            self._refresh_locked()
            self._condition.notify_all()

    def finish_actor_enqueue(self, storage_slot: int, enqueued_ns: int) -> None:
        with self._condition:
            if self._states[storage_slot] == self.FREE:
                raise RuntimeError("central request completed before enqueue")
            self._actor_enqueue_ns[storage_slot] = enqueued_ns
            self._refresh_locked()
            self._condition.notify_all()

    def server_received(self, storage_slot: int, received_ns: int) -> None:
        with self._condition:
            if self._states[storage_slot] != self.INGRESS:
                raise RuntimeError("server received a non-ingress request")
            self._server_receive_ns[storage_slot] = received_ns
            self._states[storage_slot] = self.LOCAL_PENDING
            self._refresh_locked()
            self._condition.notify_all()

    def executing(self, storage_slots) -> None:
        with self._condition:
            for storage_slot in storage_slots:
                if self._states[storage_slot] != self.LOCAL_PENDING:
                    raise RuntimeError("executing request was not locally pending")
                self._states[storage_slot] = self.EXECUTING
            self._refresh_locked()
            self._condition.notify_all()

    def completed(self, storage_slots, completed_ns: int) -> None:
        with self._condition:
            elapsed_ns = completed_ns - int(self._last_completed_ns.value)
            if elapsed_ns > 0 and self._last_completed_ns.value:
                observed = len(storage_slots) / (elapsed_ns / 1e6)
                previous = float(self._service_rate.value)
                self._service_rate.value = (
                    observed if previous <= 0 else previous * 0.8 + observed * 0.2
                )
            self._last_completed_ns.value = completed_ns
            for storage_slot in storage_slots:
                self._states[storage_slot] = self.FREE
                self._actor_enqueue_ns[storage_slot] = 0
                self._server_receive_ns[storage_slot] = 0
            self._refresh_locked()
            self._condition.notify_all()

    def cancel(self, storage_slots) -> None:
        with self._condition:
            for storage_slot in storage_slots:
                self._states[storage_slot] = self.FREE
                self._actor_enqueue_ns[storage_slot] = 0
                self._server_receive_ns[storage_slot] = 0
            self._refresh_locked()
            self._condition.notify_all()

    def invalidate(self) -> None:
        with self._condition:
            for index in range(len(self._states)):
                self._states[index] = self.FREE
                self._actor_enqueue_ns[index] = 0
                self._server_receive_ns[index] = 0
            self._valid.value = 0
            self._refresh_locked()
            self._condition.notify_all()

    def timestamps(self, storage_slot: int) -> tuple[int, int]:
        with self._condition:
            return (
                int(self._actor_enqueue_ns[storage_slot]),
                int(self._server_receive_ns[storage_slot]),
            )

    def snapshot(self) -> dict:
        with self._condition:
            now_ns = time.perf_counter_ns()
            ingress = int(self._ingress.value)
            local = int(self._local.value)
            executing = int(self._executing.value)
            oldest_actor = int(self._oldest_actor_ns.value)
            oldest_server = int(self._oldest_server_ns.value)
            return {
                "valid": bool(self._valid.value),
                "ingress_queue_depth": ingress,
                "local_pending_depth": local,
                "executing_requests": executing,
                "total_backlog": ingress + local + executing,
                "oldest_actor_enqueue_ns": oldest_actor,
                "oldest_server_receive_ns": oldest_server,
                "oldest_actor_age_ms": (
                    (now_ns - oldest_actor) / 1e6 if oldest_actor else 0.0
                ),
                "oldest_server_age_ms": (
                    (now_ns - oldest_server) / 1e6 if oldest_server else 0.0
                ),
                "last_completed_ns": int(self._last_completed_ns.value),
                "recent_requests_per_ms": float(self._service_rate.value),
            }

    def wait_for_change(self, timeout: float = 0.1) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)


def should_throttle(snapshot, mode, *, high_watermark, deadline_ms,
                    drain_target_ms, epsilon=1e-6):
    if mode == "off" or not snapshot["valid"]:
        return False
    if mode == "fixed_threshold":
        return (
            snapshot["total_backlog"] >= high_watermark
            or snapshot["oldest_actor_age_ms"] >= deadline_ms
        )
    if mode == "predicted_drain_time":
        drain_ms = snapshot["total_backlog"] / max(
            snapshot["recent_requests_per_ms"], epsilon
        )
        return drain_ms > drain_target_ms
    raise ValueError(f"unknown learner throttle mode {mode!r}")


def wait_for_learner_admission(backlog, stop_event, mode, *,
                               high_watermark, deadline_ms,
                               drain_target_ms):
    started_ns = time.perf_counter_ns()
    waited = False
    while not stop_event.is_set() and should_throttle(
        backlog.snapshot(), mode, high_watermark=high_watermark,
        deadline_ms=deadline_ms, drain_target_ms=drain_target_ms,
    ):
        waited = True
        backlog.wait_for_change(timeout=0.1)
    return waited, time.perf_counter_ns() - started_ns


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
                received_ns = time.perf_counter_ns()
                if queue_pressure is not None:
                    storage_slot = slots.index(first.actor_id, first.env_slot)
                    queue_pressure.server_received(storage_slot, received_ns)
                    actor_enqueued_ns, _ = queue_pressure.timestamps(storage_slot)
                else:
                    actor_enqueued_ns = first.actor_enqueued_ns
                pending.append(replace(
                    first, actor_enqueued_ns=actor_enqueued_ns,
                    server_received_ns=received_ns,
                ))

            oldest_ns = min(request.server_received_ns for request in pending)
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
                received_ns = time.perf_counter_ns()
                if queue_pressure is not None:
                    storage_slot = slots.index(item.actor_id, item.env_slot)
                    queue_pressure.server_received(storage_slot, received_ns)
                    actor_enqueued_ns, _ = queue_pressure.timestamps(storage_slot)
                else:
                    actor_enqueued_ns = item.actor_enqueued_ns
                pending.append(replace(
                    item, actor_enqueued_ns=actor_enqueued_ns,
                    server_received_ns=received_ns,
                ))

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
                oldest_request = min(
                    pending, key=lambda item: item.server_received_ns
                )
                selected_key = _compatible_key(oldest_request, slots.max_actions)
                selected_group = groups[selected_key]
            requests = selected_group[:max_microbatch]
            selected_ids = {id(request) for request in requests}
            pending = [request for request in pending if id(request) not in selected_ids]
            groups = {selected_key: requests}
            executing_slots = [
                slots.index(request.actor_id, request.env_slot)
                for request in requests
            ]
            if queue_pressure is not None:
                queue_pressure.executing(executing_slots)

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
                packing_done_ns = time.perf_counter_ns()
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
                    _put_response(response_queues, request, "ok", (
                        int(action_index), completed_ns
                    ))
                if queue_pressure is not None:
                    queue_pressure.completed(executing_slots, completed_ns)
                if metric_store is not None:
                    h2d_ns = int(
                        timing_events[0].elapsed_time(timing_events[1]) * 1e6
                    )
                    forward_ns = int(
                        timing_events[1].elapsed_time(timing_events[2]) * 1e6
                    )
                    metric_store.add_central(
                        microbatch_size=len(group), legal_actions=action_total,
                        ipc_wait_ns=[
                            r.server_received_ns - r.actor_enqueued_ns
                            for r in group
                        ],
                        batch_wait_ns=[
                            batch_started_ns - r.server_received_ns
                            for r in group
                        ],
                        batching_wait_ns=batch_started_ns - oldest_ns,
                        cpu_packing_ns=packing_done_ns - batch_started_ns,
                        h2d_ns=h2d_ns, forward_ns=forward_ns,
                        d2h_response_ns=max(
                            0, completed_ns - packing_done_ns
                            - h2d_ns - forward_ns
                        ),
                        stream_priority_active=priority_active,
                    )
        for request in pending:
            _put_response(response_queues, request, "shutdown", "inference stopping")
        if queue_pressure is not None:
            queue_pressure.cancel([
                slots.index(request.actor_id, request.env_slot)
                for request in pending
            ])
    except BaseException as exc:
        stop_event.set()
        if queue_pressure is not None:
            queue_pressure.invalidate()
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        for actor_id, response_queue in enumerate(response_queues):
            response_queue.put(("error", actor_id, -1, -1, -1, detail))
        raise
