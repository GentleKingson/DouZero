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


def _flush_actor_rollouts(completed, sizes, free_queue, full_queue, buffers,
                          flags, recorder, stop_event):
    """Flush complete role rollouts assembled by one or more actor envs."""
    T = flags.unroll_length
    for position in ('landlord', 'landlord_up', 'landlord_down'):
        while rollout_ready(
            sizes[position], T, getattr(flags, "legacy_flush_ge", False)
        ):
            queue_started_ns = time.perf_counter_ns()
            index = free_queue[position].get()
            free_queue_wait_ns = time.perf_counter_ns() - queue_started_ns
            if index is None or (stop_event is not None and stop_event.is_set()):
                return False
            write_started_ns = time.perf_counter_ns()
            for key, values in completed[position].items():
                target = buffers[position][key][index]
                selected = values[:T]
                if getattr(flags, "legacy_bulk_rollout", False):
                    if selected and isinstance(selected[0], torch.Tensor):
                        source = torch.stack(selected).reshape(target.shape)
                        source = source.to(device=target.device, dtype=target.dtype)
                    else:
                        source = torch.as_tensor(
                            selected, device=target.device, dtype=target.dtype
                        ).reshape(target.shape)
                    target.copy_(source)
                else:
                    for t, value in enumerate(selected):
                        target[t, ...] = value
            if buffers[position]['done'][index].is_cuda:
                torch.cuda.synchronize(buffers[position]['done'][index].device)
            rollout_write_ns = time.perf_counter_ns() - write_started_ns
            put_started_ns = time.perf_counter_ns()
            full_queue[position].put(index)
            recorder.add(
                transitions=T,
                rollout_write_ns=rollout_write_ns,
                free_queue_wait_ns=free_queue_wait_ns,
                full_queue_put_ns=time.perf_counter_ns() - put_started_ns,
            )
            for values in completed[position].values():
                del values[:T]
            sizes[position] -= T
    return True


def act_centralized_multi(i, device, free_queue, full_queue, policy_pool,
                          buffers, flags, stop_event, metric_store,
                          central_slots, central_request_queue,
                          central_response_queue, central_queue_pressure):
    """Run several independent games in one actor with async GPU requests."""
    from douzero.runtime import (
        derive_actor_seed, maybe_set_global_deterministic, set_global_seed,
    )
    from .centralized_actor import PendingRequestScheduler

    env_count = getattr(flags, 'central_actor_envs_per_actor', 4)
    actor_threads = getattr(flags, "actor_torch_threads", 0) or 1
    torch.set_num_threads(actor_threads)
    set_global_seed(derive_actor_seed(
        getattr(flags, "seed", 0), device_token=device, actor_id=i
    ))
    maybe_set_global_deterministic(getattr(flags, "deterministic", False))
    recorder = ActorMetricRecorder(metric_store)
    scheduler = PendingRequestScheduler(i, env_count)
    positions = ('landlord', 'landlord_up', 'landlord_down')
    completed = {
        position: {key: [] for key in (
            'done', 'episode_return', 'target', 'obs_x_no_action',
            'obs_action', 'obs_z', 'policy_version',
        )} for position in positions
    }
    completed_sizes = {position: 0 for position in positions}
    slots = []
    try:
        for env_slot in range(env_count):
            env = Environment(create_env(flags), device)
            position, obs, env_output = env.initial()
            lease = policy_pool.acquire(owner_id=i * env_count + env_slot)
            slots.append({
                'env': env, 'position': position, 'obs': obs,
                'env_output': env_output, 'lease': lease,
                'episode_version': lease.version, 'waiting': False,
                'explore': False,
                'episode': {
                    position: {key: [] for key in (
                        'obs_x_no_action', 'obs_action', 'obs_z',
                        'policy_version',
                    )} for position in positions
                },
            })

        def advance(env_slot, action_index, inference_ns):
            state = slots[env_slot]
            position, obs, env_output = (
                state['position'], state['obs'], state['env_output']
            )
            episode = state['episode'][position]
            legal_count = len(obs['legal_actions'])
            action = obs['legal_actions'][action_index]
            episode['obs_x_no_action'].append(env_output['obs_x_no_action'])
            episode['obs_z'].append(env_output['obs_z'])
            episode['policy_version'].append(state['episode_version'])
            episode['obs_action'].append(
                obs['x_action'][action_index].to(dtype=torch.int8)
            )
            next_position, next_obs, next_output = state['env'].step(action)
            timing = next_output.get('timing', {})
            recorder.add(
                decisions=1,
                single_legal_actions=int(legal_count == 1),
                inference_ns=inference_ns,
                env_step_ns=timing.get('env_step_ns', 0),
                legal_actions_ns=timing.get('legal_actions_ns', 0),
                observation_ns=timing.get('observation_ns', 0),
            )
            state.update(position=next_position, obs=next_obs,
                         env_output=next_output, waiting=False)
            if not next_output['done']:
                return
            recorder.add(games=1)
            terminal_return = float(next_output['episode_return'].item())
            for role in positions:
                role_episode = state['episode'][role]
                count = len(role_episode['obs_action'])
                if not count:
                    continue
                role_completed = completed[role]
                for key in ('obs_x_no_action', 'obs_action', 'obs_z',
                            'policy_version'):
                    role_completed[key].extend(role_episode[key])
                role_completed['done'].extend([False] * (count - 1) + [True])
                reward = terminal_return if role == 'landlord' else -terminal_return
                role_completed['episode_return'].extend([0.0] * (count - 1) + [reward])
                role_completed['target'].extend([reward] * count)
                completed_sizes[role] += count
                for values in role_episode.values():
                    values.clear()
            policy_pool.release(state['lease'])
            state['lease'] = policy_pool.acquire(owner_id=i * env_count + env_slot)
            state['episode_version'] = state['lease'].version

        while not stop_event.is_set():
            made_local_progress = False
            for env_slot, state in enumerate(slots):
                if state['waiting']:
                    continue
                obs = state['obs']
                position = state['position']
                legal_count = len(obs['legal_actions'])
                recorder.legal_actions(legal_count)
                started_ns = time.perf_counter_ns()
                if legal_count == 1:
                    if flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                        torch.randint(1, (1,))[0]
                    advance(env_slot, 0, time.perf_counter_ns() - started_ns)
                    made_local_progress = True
                    continue
                if legal_count > central_slots.max_actions:
                    raise RuntimeError(
                        "legal-action count exceeds centralized staging capacity"
                    )
                central_slots.write(
                    i, position, obs['z_single'], obs['x_state_single'],
                    obs['x_action'], env_slot=env_slot,
                )
                request = scheduler.prepare(
                    env_slot, policy_slot=state['lease'].slot,
                    policy_version=state['episode_version'], position=position,
                    action_count=legal_count,
                )
                state['explore'] = (
                    flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon
                )
                while not stop_event.is_set():
                    try:
                        central_request_queue.put(request, timeout=0.1)
                        if central_queue_pressure is not None:
                            central_queue_pressure.enqueued(request.queued_ns)
                        break
                    except queue.Full:
                        # Bounded blocking provides backpressure while still
                        # observing shutdown promptly.
                        continue
                if stop_event.is_set():
                    scheduler.cancel_all()
                    return
                scheduler.mark_queued(request)
                state['waiting'] = True

            if not _flush_actor_rollouts(
                completed, completed_sizes, free_queue, full_queue, buffers,
                flags, recorder, stop_event,
            ):
                return
            if scheduler.pending:
                try:
                    response = central_response_queue.get(timeout=getattr(
                        flags, 'central_actor_timeout_seconds', 30.0
                    ))
                except queue.Empty:
                    raise TimeoutError('centralized actor inference timed out')
                consumed = scheduler.consume(response)
                if consumed is None:
                    return
                request, selected = consumed
                state = slots[request.env_slot]
                action_index = (
                    int(torch.randint(request.action_count, (1,))[0].item())
                    if state['explore'] else selected
                )
                advance(
                    request.env_slot, action_index,
                    time.perf_counter_ns() - request.queued_ns,
                )
            elif not made_local_progress:
                stop_event.wait(0.01)
    except BaseException:
        stop_event.set()
        raise
    finally:
        scheduler.cancel_all()
        recorder.flush()
        for env_slot, state in enumerate(slots):
            if state.get('lease') is not None:
                policy_pool.release(state['lease'])
            state['env'].close()


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
        central_request_queue=None, central_response_queue=None,
        central_queue_pressure=None):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    if getattr(flags, "legacy_actor_backend", "legacy") == "centralized_factorized":
        return act_centralized_multi(
            i, device, free_queue, full_queue, policy_pool, buffers, flags,
            stop_event, metric_store, central_slots, central_request_queue,
            central_response_queue, central_queue_pressure,
        )
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
                        with inference_context:
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
        if stop_event is not None:
            stop_event.set()
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
