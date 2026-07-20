"""Optional centralized CUDA inference service for factorized V1 CPU actors."""

from __future__ import annotations

import queue
import time
import traceback

import torch

from .models_factorized import LegacyFactorizedModel


POSITIONS = ("landlord", "landlord_up", "landlord_down")


class CentralizedInferenceSlots:
    """One preallocated shared request slot per CPU actor."""

    def __init__(self, num_actors: int, max_actions: int) -> None:
        self.max_actions = max_actions
        self.z = torch.empty(
            num_actors, 5, 162, dtype=torch.int8
        ).share_memory_()
        self.x_state = torch.empty(
            num_actors, 430, dtype=torch.int8
        ).share_memory_()
        self.x_action = torch.empty(
            num_actors, max_actions, 54, dtype=torch.int8
        ).share_memory_()

    def write(self, actor_id, position, z_single, x_state_single, x_action):
        count = x_action.shape[0]
        if count > self.max_actions:
            raise ValueError("central inference request exceeds slot capacity")
        state_width = 319 if position == "landlord" else 430
        self.z[actor_id].copy_(z_single[0].to(dtype=torch.int8))
        self.x_state[actor_id, :state_width].copy_(
            x_state_single[0].to(dtype=torch.int8)
        )
        self.x_action[actor_id, :count].copy_(x_action.to(dtype=torch.int8))


def _copy_policy_slot(gpu_policy, policy_pool, policy_slot):
    source = policy_pool.models[policy_slot]
    with torch.no_grad():
        for position in POSITIONS:
            target_state = gpu_policy.get_model(position).state_dict(keep_vars=True)
            source_state = source.get_model(position).state_dict(keep_vars=True)
            for key, target in target_state.items():
                target.copy_(source_state[key], non_blocking=True)


def centralized_inference_loop(
    device,
    policy_pool,
    slots,
    request_queue,
    response_queues,
    stop_event,
    target_microbatch,
    max_delay_ms,
):
    """Batch compatible requests and return only selected action indices."""
    try:
        torch.set_num_threads(1)
        cuda_device = torch.device("cuda:" + str(device))
        torch.cuda.set_device(cuda_device)
        policies = [LegacyFactorizedModel(device=device)
                    for _ in policy_pool.models]
        loaded_versions = [-1] * len(policies)
        for policy in policies:
            policy.eval()

        pinned_z = torch.empty(
            target_microbatch, 5, 162, dtype=torch.float32, pin_memory=True
        )
        pinned_state = torch.empty(
            target_microbatch, 430, dtype=torch.float32, pin_memory=True
        )
        pinned_action = torch.empty(
            target_microbatch * slots.max_actions, 54,
            dtype=torch.float32, pin_memory=True,
        )

        while not stop_event.is_set():
            try:
                first = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                return
            requests = [first]
            deadline = time.perf_counter() + max_delay_ms / 1000.0
            while len(requests) < target_microbatch and time.perf_counter() < deadline:
                try:
                    item = request_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0)
                    continue
                if item is None:
                    stop_event.set()
                    break
                requests.append(item)

            groups = {}
            for request in requests:
                actor_id, policy_slot, policy_version, position, count = request
                bucket = 64 if count <= 64 else slots.max_actions
                groups.setdefault(
                    (policy_slot, policy_version, position, bucket), []
                ).append(request)

            for (policy_slot, policy_version, position, _bucket), group in groups.items():
                if loaded_versions[policy_slot] != policy_version:
                    _copy_policy_slot(policies[policy_slot], policy_pool, policy_slot)
                    torch.cuda.synchronize(cuda_device)
                    loaded_versions[policy_slot] = policy_version
                state_width = 319 if position == "landlord" else 430
                counts = [request[4] for request in group]
                action_total = sum(counts)
                for batch_index, request in enumerate(group):
                    actor_id = request[0]
                    pinned_z[batch_index].copy_(slots.z[actor_id])
                    pinned_state[batch_index, :state_width].copy_(
                        slots.x_state[actor_id, :state_width]
                    )
                offset = 0
                for request in group:
                    actor_id, _, _, _, count = request
                    pinned_action[offset:offset + count].copy_(
                        slots.x_action[actor_id, :count]
                    )
                    offset += count
                z = pinned_z[:len(group)].to(cuda_device, non_blocking=True)
                state = pinned_state[:len(group), :state_width].to(
                    cuda_device, non_blocking=True
                )
                actions = pinned_action[:action_total].to(
                    cuda_device, non_blocking=True
                )
                with torch.inference_mode():
                    indices = policies[policy_slot].get_model(
                        position
                    ).select_actions_packed(z, state, actions, counts)
                indices_cpu = indices.cpu().tolist()
                for request, action_index in zip(group, indices_cpu):
                    response_queues[request[0]].put(("ok", int(action_index)))
    except BaseException as exc:
        stop_event.set()
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        for response_queue in response_queues:
            response_queue.put(("error", detail))
        raise
