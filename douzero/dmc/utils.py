import os 
import typing
import logging
import traceback
import queue
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array
from .legacy_metrics import ActorMetricRecorder
from .profiling import legacy_profile_range

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def rollout_ready(size, unroll_length, flush_on_equal=False):
    """Return whether an actor may submit a complete rollout."""
    return size >= unroll_length if flush_on_equal else size > unroll_length

def create_env(flags):
    backend = getattr(flags, "legacy_actor_backend", "legacy")
    if backend == "centralized_factorized":
        backend = "factorized"
    return Env(
        flags.objective,
        observation_backend=backend,
        profile_timing=getattr(flags, "legacy_profile", False),
    )


def receive_central_action(response_queue, timeout_seconds):
    """Decode one bounded centralized-inference response."""
    try:
        status, response = response_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError('centralized actor inference timed out') from exc
    if status == 'shutdown':
        return None
    if status != 'ok':
        raise RuntimeError(
            f'centralized actor inference failed: {response}'
        )
    return int(response)


class PinnedBatchStager:
    """Reusable pinned destination for contiguous CPU rollout buffers."""

    def __init__(self, buffers, flags):
        if not all(isinstance(value, torch.Tensor) for value in buffers.values()):
            raise ValueError("reusable pinned staging requires contiguous buffers")
        self.batch_size = flags.batch_size
        self.storage = {
            key: torch.empty(
                (flags.batch_size, *value.shape[1:]),
                dtype=value.dtype,
                pin_memory=True,
            )
            for key, value in buffers.items()
        }
        self._h2d_done = None

    def stage(self, buffers, indices):
        if self._h2d_done is not None:
            self._h2d_done.synchronize()
            self._h2d_done = None
        index = torch.tensor(indices, dtype=torch.long)
        for key, source in buffers.items():
            torch.index_select(source, 0, index, out=self.storage[key])
        return {key: value.transpose(0, 1) for key, value in self.storage.items()}

    def mark_h2d(self, device):
        if torch.device(device).type != "cuda":
            return
        event = torch.cuda.Event()
        event.record()
        self._h2d_done = event

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock,
              stager=None,
              return_timings=False):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending them to ``free_queue``. A ``None`` item is a
    shutdown sentinel and returns ``None`` to the learner thread.
    """
    indices = []
    wait_started_ns = time.perf_counter_ns()
    profile_ranges = getattr(flags, "legacy_profile", False)
    with legacy_profile_range(profile_ranges, "learner.batch_wait"):
        with lock:
            for _ in range(flags.batch_size):
                index = full_queue.get()
                if index is None:
                    for acquired in indices:
                        free_queue.put(acquired)
                    return None
                indices.append(index)
    batch_wait_ns = time.perf_counter_ns() - wait_started_ns
    assembly_started_ns = time.perf_counter_ns()
    with legacy_profile_range(profile_ranges, "learner.batch_assembly"):
        if stager is not None:
            batch = stager.stage(buffers, indices)
        elif all(isinstance(value, torch.Tensor) for value in buffers.values()):
            first = next(iter(buffers.values()))
            index_device = first.device if first.is_cuda else torch.device("cpu")
            index = torch.tensor(indices, dtype=torch.long, device=index_device)
            batch = {
                key: value.index_select(0, index).transpose(0, 1)
                for key, value in buffers.items()
            }
        else:
            batch = {
                key: torch.stack([buffers[key][m] for m in indices], dim=1)
                for key in buffers
            }
    batch_assembly_ns = time.perf_counter_ns() - assembly_started_ns
    pin_memory_ns = 0
    if (stager is None and getattr(flags, "pin_memory", False)
            and torch.cuda.is_available()
            and not any(tensor.is_cuda for tensor in batch.values())):
        pin_started_ns = time.perf_counter_ns()
        with legacy_profile_range(profile_ranges, "learner.pin_memory"):
            batch = {key: tensor.pin_memory() for key, tensor in batch.items()}
        pin_memory_ns = time.perf_counter_ns() - pin_started_ns
    if any(tensor.is_cuda for tensor in batch.values()):
        torch.cuda.synchronize(next(iter(batch.values())).device)
    for m in indices:
        free_queue.put(m)
    if return_timings:
        return batch, {
            "batch_wait_ns": batch_wait_ns,
            "batch_assembly_ns": batch_assembly_ns,
            "pin_memory_ns": pin_memory_ns,
        }
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
            foreach=getattr(flags, "rmsprop_foreach", False))
        optimizers[position] = optimizer
    return optimizers

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
                policy_version=dict(size=(T,), dtype=torch.int64),
            )
            if getattr(flags, "legacy_contiguous_buffers", False):
                target_device = (
                    torch.device("cpu") if device == "cpu"
                    else torch.device("cuda:" + str(device))
                )
                _buffers = {
                    key: torch.empty(
                        (flags.num_buffers, *spec["size"]),
                        dtype=spec["dtype"],
                        device=target_device,
                    ).share_memory_()
                    for key, spec in specs.items()
                }
            else:
                _buffers: Buffers = {key: [] for key in specs}
                for _ in range(flags.num_buffers):
                    for key in _buffers:
                        if not device == "cpu":
                            _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                        else:
                            _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                        _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, policy_pool, buffers, flags,
        stop_event=None, metric_store=None, central_slots=None,
        central_request_queue=None, central_response_queue=None):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    lease = None
    env = None
    recorder = ActorMetricRecorder(metric_store)
    try:
        T = flags.unroll_length
        actor_threads = getattr(flags, "actor_torch_threads", 0)
        if actor_threads < 0:
            raise ValueError("actor_torch_threads must be >= 0")
        if actor_threads == 0 and getattr(
            flags, "legacy_actor_backend", "legacy"
        ) in {"factorized", "centralized_factorized"}:
            actor_threads = 1
        if actor_threads > 0:
            torch.set_num_threads(actor_threads)
        # P01: per-actor deterministic seed + optional determinism (opt-in;
        # both are no-ops when seed=0 / deterministic=False). ``device`` is the
        # device token passed to act(); it is "cpu" for CPU actors or a GPU
        # index for CUDA actors. derive_actor_seed accepts both (it never
        # coerces to int, which would crash for "cpu").
        from douzero.runtime import (
            derive_actor_seed,
            maybe_set_global_deterministic,
            set_global_seed,
        )

        base_seed = getattr(flags, "seed", 0)
        actor_seed = derive_actor_seed(base_seed, device_token=device, actor_id=i)
        set_global_seed(actor_seed)
        maybe_set_global_deterministic(getattr(flags, "deterministic", False))
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        policy_version_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()
        lease = policy_pool.acquire(owner_id=i)
        model = lease.model
        episode_policy_version = lease.version

        while stop_event is None or not stop_event.is_set():
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                policy_version_buf[position].append(episode_policy_version)
                legal_count = len(obs['legal_actions'])
                recorder.legal_actions(legal_count)
                inference_started_ns = time.perf_counter_ns()
                backend = getattr(flags, "legacy_actor_backend", "legacy")
                if (legal_count == 1
                        and backend in {"factorized", "centralized_factorized"}):
                    # Preserve legacy epsilon RNG consumption even though the
                    # only possible action makes model inference unnecessary.
                    if (flags.exp_epsilon > 0
                            and np.random.rand() < flags.exp_epsilon):
                        torch.randint(1, (1,))[0]
                    _action_idx = 0
                else:
                    if (backend == "centralized_factorized"
                            and legal_count <= central_slots.max_actions):
                        central_slots.write(
                            i, position, obs['z_single'],
                            obs['x_state_single'], obs['x_action'],
                        )
                        explore = (
                            flags.exp_epsilon > 0
                            and np.random.rand() < flags.exp_epsilon
                        )
                        central_request_queue.put((
                            i, lease.slot, episode_policy_version,
                            position, legal_count,
                        ))
                        response = receive_central_action(
                            central_response_queue,
                            getattr(
                                flags, 'central_actor_timeout_seconds', 30.0
                            ),
                        )
                        if response is None:
                            return
                        _action_idx = (
                            int(torch.randint(legal_count, (1,))[0].item())
                            if explore else int(response)
                        )
                    else:
                        inference_context = (
                            torch.inference_mode()
                            if backend in {"factorized", "centralized_factorized"}
                            else torch.no_grad()
                        )
                        with inference_context, legacy_profile_range(
                            getattr(flags, "legacy_profile", False),
                            "actor.inference",
                        ):
                            if backend in {"factorized", "centralized_factorized"}:
                                role_model = model.get_model(position)
                                agent_output = role_model.forward_factorized(
                                    obs['z_single'], obs['x_state_single'],
                                    obs['x_action'], flags=flags,
                                )
                            else:
                                agent_output = model.forward(
                                    position, obs['z_batch'], obs['x_batch'], flags=flags
                                )
                        _action_idx = int(agent_output['action'].item())
                inference_ns = time.perf_counter_ns() - inference_started_ns
                action = obs['legal_actions'][_action_idx]
                if getattr(flags, "legacy_actor_backend", "legacy") in {
                    "factorized", "centralized_factorized"
                }:
                    encoded_action = obs['x_action'][_action_idx]
                else:
                    encoded_action = _cards2tensor(action)
                obs_action_buf[position].append(encoded_action.to(dtype=torch.int8))
                size[position] += 1
                position, obs, env_output = env.step(action)
                timing = env_output.get('timing', {})
                recorder.add(
                    decisions=1,
                    single_legal_actions=int(legal_count == 1),
                    inference_ns=inference_ns,
                    env_step_ns=timing.get('env_step_ns', 0),
                    legal_actions_ns=timing.get('legal_actions_ns', 0),
                    observation_ns=timing.get('observation_ns', 0),
                )
                if env_output['done']:
                    recorder.add(games=1)
                    terminal_return = float(env_output['episode_return'].item())
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            episode_return = terminal_return if p == 'landlord' else -terminal_return
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            # The environment has already reset, but no inference for the next
            # game has happened. This is the only safe actor policy switch point.
            policy_pool.release(lease)
            lease = None
            lease = policy_pool.acquire(owner_id=i)
            model = lease.model
            episode_policy_version = lease.version

            for p in positions:
                def has_rollout():
                    return rollout_ready(
                        size[p], T, getattr(flags, "legacy_flush_ge", False)
                    )

                while has_rollout():
                    queue_started_ns = time.perf_counter_ns()
                    index = free_queue[p].get()
                    free_queue_wait_ns = time.perf_counter_ns() - queue_started_ns
                    if index is None:
                        return
                    write_started_ns = time.perf_counter_ns()
                    source_buffers = {
                        'done': done_buf[p],
                        'episode_return': episode_return_buf[p],
                        'target': target_buf[p],
                        'obs_x_no_action': obs_x_no_action_buf[p],
                        'obs_action': obs_action_buf[p],
                        'obs_z': obs_z_buf[p],
                        'policy_version': policy_version_buf[p],
                    }
                    if getattr(flags, "legacy_bulk_rollout", False):
                        for key, values in source_buffers.items():
                            target = buffers[p][key][index]
                            selected = values[:T]
                            if selected and isinstance(selected[0], torch.Tensor):
                                source = torch.stack(selected).reshape(target.shape)
                                source = source.to(
                                    device=target.device, dtype=target.dtype
                                )
                            else:
                                source = torch.as_tensor(
                                    selected, device=target.device, dtype=target.dtype
                                ).reshape(target.shape)
                            target.copy_(source)
                    else:
                        for t in range(T):
                            for key, values in source_buffers.items():
                                buffers[p][key][index][t, ...] = values[t]
                    first_target = buffers[p]['done'][index]
                    if first_target.is_cuda:
                        # Queue publication is the cross-process visibility
                        # boundary; finish actor-stream writes before exposing
                        # this CUDA slot to the learner process.
                        torch.cuda.synchronize(first_target.device)
                    rollout_write_ns = time.perf_counter_ns() - write_started_ns
                    put_started_ns = time.perf_counter_ns()
                    full_queue[p].put(index)
                    full_queue_put_ns = time.perf_counter_ns() - put_started_ns
                    recorder.add(
                        transitions=T,
                        rollout_write_ns=rollout_write_ns,
                        free_queue_wait_ns=free_queue_wait_ns,
                        full_queue_put_ns=full_queue_put_ns,
                    )
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    policy_version_buf[p] = policy_version_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
    finally:
        recorder.flush()
        if lease is not None:
            policy_pool.release(lease)
        if env is not None:
            env.close()

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix
