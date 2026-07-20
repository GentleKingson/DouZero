import os
import queue
import math
import threading
import time
import timeit
import pprint
import statistics
import subprocess
from contextlib import ExitStack, contextmanager
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp

from .file_writer import FileWriter
from .models import Model
from .models_factorized import LegacyFactorizedModel
from .utils import (
    PinnedBatchStager,
    get_batch,
    log,
    create_env,
    create_buffers,
    create_optimizers,
    act,
)
from .legacy_metrics import LegacyMetricStore, write_metrics
from .centralized_actor import (
    CentralQueuePressure,
    CentralizedInferenceSlots,
    centralized_inference_loop,
    wait_for_learner_admission,
)
from douzero.runtime import SafeMixedPrecision, VersionedPolicyPool

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}


def _save_legacy_sidecars(learner_model, directory, frames, retention):
    """Atomically save evaluation weights and prune older role snapshots."""
    if retention == 0:
        return

    from douzero.checkpoint import save_legacy_position_weights

    os.makedirs(directory, exist_ok=True)
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        filename = f'{position}_weights_{frames}.ckpt'
        save_legacy_position_weights(
            os.path.join(directory, filename),
            learner_model.get_model(position).state_dict(),
        )
        if retention < 0:
            continue

        prefix = f'{position}_weights_'
        candidates = []
        for name in os.listdir(directory):
            if not name.startswith(prefix) or not name.endswith('.ckpt'):
                continue
            frame_text = name[len(prefix):-len('.ckpt')]
            if frame_text.isdigit():
                candidates.append((int(frame_text), name))
        for _, name in sorted(candidates)[:-retention]:
            os.remove(os.path.join(directory, name))


def _physical_gpu_identifier(logical_device):
    """Map a logical CUDA index through CUDA_VISIBLE_DEVICES for nvidia-smi."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(logical_device)
    devices = [item.strip() for item in visible.split(',') if item.strip()]
    try:
        return devices[int(logical_device)]
    except (IndexError, TypeError, ValueError):
        return str(logical_device)


class _SystemSampler:
    """Portable best-effort process/GPU sampling for benchmark reports."""

    def __init__(self, gpu_identifier='0'):
        self.samples = []
        self._last_time = None
        self._last_ticks = None
        self._gpu_identifier = str(gpu_identifier)

    @staticmethod
    def _process_stats(pids):
        ticks = 0
        rss_bytes = 0
        for pid in pids:
            try:
                stat = open(f'/proc/{pid}/stat', encoding='ascii').read().split()
                ticks += int(stat[13]) + int(stat[14])
                for line in open(f'/proc/{pid}/status', encoding='ascii'):
                    if line.startswith('VmRSS:'):
                        rss_bytes += int(line.split()[1]) * 1024
                        break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return ticks, rss_bytes

    def _gpu_stats(self):
        if not torch.cuda.is_available():
            return None, None
        try:
            output = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits',
                    '-i', self._gpu_identifier,
                ],
                text=True,
                timeout=2,
            ).strip().splitlines()[0]
            utilization, memory_mib = output.split(',')
            return float(utilization), float(memory_mib.strip())
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None

    def sample(self, pids):
        now = time.monotonic()
        ticks, rss_bytes = self._process_stats(pids)
        cpu_percent = None
        if self._last_time is not None and self._last_ticks is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                cpu_percent = (
                    (ticks - self._last_ticks)
                    / os.sysconf('SC_CLK_TCK') / elapsed * 100.0
                )
        gpu_percent, vram_mib = self._gpu_stats()
        self.samples.append({
            'elapsed_seconds': now,
            'cpu_percent': cpu_percent,
            'rss_mib': rss_bytes / (1024 * 1024),
            'gpu_percent': gpu_percent,
            'vram_mib': vram_mib,
        })
        self._last_time, self._last_ticks = now, ticks

    @staticmethod
    def _summary(values):
        values = [value for value in values if value is not None]
        if not values:
            return {'median': None, 'p95': None, 'max': None}
        ordered = sorted(values)
        p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        return {
            'median': statistics.median(ordered),
            'p95': p95,
            'max': max(ordered),
        }

    def report(self):
        report = {
            name: self._summary([sample[name] for sample in self.samples])
            for name in ('cpu_percent', 'rss_mib', 'gpu_percent', 'vram_mib')
        }
        if self.samples:
            report['rss_growth_mib'] = (
                self.samples[-1]['rss_mib'] - self.samples[0]['rss_mib']
            )
            first_vram = self.samples[0]['vram_mib']
            last_vram = self.samples[-1]['vram_mib']
            report['vram_growth_mib'] = (
                last_vram - first_vram
                if first_vram is not None and last_vram is not None else None
            )
        report['samples'] = self.samples
        return report


class _LearnerThreadSupervisor:
    """Propagate learner failures to the monitoring thread."""

    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.errors = queue.Queue()

    def run(self, target, *args):
        try:
            target(*args)
        except BaseException as exc:
            self.errors.put((exc, exc.__traceback__))
            self.stop_event.set()

    def raise_if_failed(self):
        try:
            exc, traceback = self.errors.get_nowait()
        except queue.Empty:
            return
        raise exc.with_traceback(traceback)


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss


def compute_policy_lag(learner_updates, versions):
    """Return transition-weighted mean lag and the oldest transition lag."""
    if versions is None or versions.numel() == 0:
        return 0.0, 0
    reference = int(learner_updates)
    mean_lag = max(
        0.0,
        float(reference) - float(versions.to(torch.float64).mean().item()),
    )
    max_lag = max(0, reference - int(versions.min().item()))
    return mean_lag, max_lag


class _UpdateBudget:
    """Atomically reserve complete learner updates against a frame limit."""

    def __init__(self, completed_frames, total_frames, frames_per_update):
        self._lock = threading.Lock()
        remaining_frames = int(total_frames) - int(completed_frames)
        step = int(frames_per_update)
        if remaining_frames < 0 or remaining_frames % step:
            raise ValueError("frame budget must contain complete learner updates")
        self._available = remaining_frames // step
        self._next_token = 0
        self._active = set()

    def reserve(self):
        with self._lock:
            if self._available == 0:
                return None
            token = self._next_token
            self._next_token += 1
            self._available -= 1
            self._active.add(token)
            return token

    def commit(self, token):
        with self._lock:
            if token not in self._active:
                raise ValueError("unknown or completed update reservation")
            self._active.remove(token)

    def cancel(self, token):
        with self._lock:
            if token not in self._active:
                raise ValueError("unknown or completed update reservation")
            self._active.remove(token)
            self._available += 1


class _TrainingTransactions:
    """Coordinate atomic learner updates and checkpoint snapshots."""

    def __init__(self, positions):
        self.positions = tuple(positions)
        # learn() already takes this lock around optimizer.step(). RLock lets
        # the caller extend the same critical section through progress commit.
        self.position_locks = {
            position: threading.RLock() for position in self.positions
        }
        self.state_lock = threading.Lock()

    def update(self, position):
        return self.position_locks[position]

    @contextmanager
    def freeze_updates(self):
        with ExitStack() as stack:
            for position in self.positions:
                stack.enter_context(self.position_locks[position])
            yield

    @contextmanager
    def snapshot(self):
        # Every path acquires role locks before state_lock. Keeping that order
        # fixed prevents checkpoint/publication deadlocks.
        with self.freeze_updates():
            with self.state_lock:
                yield


def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock,
          amp_controller=None,
          learner_updates=0,
          profile=False,
          stager=None):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    non_blocking = bool(
        getattr(flags, 'pin_memory', False)
        or getattr(flags, 'legacy_reusable_pinned_staging', False)
    )
    h2d_started_ns = time.perf_counter_ns()
    cuda_h2d_events = None
    if profile and device.type == 'cuda':
        cuda_h2d_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        cuda_h2d_events[0].record()
    obs_x_no_action = batch['obs_x_no_action'].to(device, non_blocking=non_blocking)
    obs_action = batch['obs_action'].to(device, non_blocking=non_blocking)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device, non_blocking=non_blocking), 0, 1).float()
    target = torch.flatten(batch['target'].to(device, non_blocking=non_blocking), 0, 1)
    if stager is not None:
        stager.mark_h2d(device)
    if cuda_h2d_events is not None:
        cuda_h2d_events[1].record()
        cuda_h2d_events[1].synchronize()
        h2d_ns = int(cuda_h2d_events[0].elapsed_time(cuda_h2d_events[1]) * 1e6)
    else:
        h2d_ns = time.perf_counter_ns() - h2d_started_ns if profile else 0
    episode_returns = batch['episode_return'][batch['done']]
    if episode_returns.numel() > 0:
        mean_episode_return_buf[position].append(
            torch.mean(episode_returns).to(device)
        )
        
    with lock:
        if amp_controller is None:
            amp_controller = SafeMixedPrecision(device, enabled=False)

        def loss_closure():
            learner_outputs = model(obs_z, obs_x, return_value=True)
            return compute_loss(learner_outputs['values'], target)

        step_result = amp_controller.step(
            loss_closure,
            optimizer,
            model.parameters(),
            max_grad_norm=flags.max_grad_norm,
            clip_grad_norm=(
                (lambda parameters, max_norm, error_if_nonfinite=False:
                 torch.nn.utils.clip_grad_norm_(
                     parameters, max_norm,
                     error_if_nonfinite=error_if_nonfinite, foreach=True,
                 ))
                if getattr(flags, 'grad_clip_foreach', False) else None
            ),
            profile=profile,
        )
        loss = step_result.loss
        policy_lag_mean, policy_lag_max = compute_policy_lag(
            learner_updates, batch.get('policy_version')
        )
        stats = {
            'mean_episode_return_'+position: (
                torch.mean(torch.stack([
                    _r for _r in mean_episode_return_buf[position]
                ])).item()
                if mean_episode_return_buf[position] else 0.0
            ),
            'loss_'+position: loss.item(),
            'policy_lag_mean_'+position: policy_lag_mean,
            'policy_lag_max_'+position: policy_lag_max,
            'amp_fallbacks_'+position: float(amp_controller.fallback_count),
        }
        timings = {'h2d_ns': h2d_ns}
        if step_result.timings_ns:
            timings.update({f'{key}_ns': value
                            for key, value in step_result.timings_ns.items()})
        if profile:
            for parameter in model.parameters():
                if not bool(torch.isfinite(parameter.detach()).all().item()):
                    raise FloatingPointError('non-finite learner parameter')
        return stats, timings

def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then it will call
    learning function with multiple threads.
    """
    # P03: training only supports the legacy observation feature version. The
    # V2 observation schema (douzero/observation/) is accepted by configuration
    # but is NOT yet wired into the actor/learner — the buffers, models, and
    # loss all assume the legacy x_batch/z_batch tensors. Rejecting a V2 run
    # here, BEFORE any CUDA/FileWriter/checkpoint/model/buffer/actor
    # initialisation, prevents an identity-mismatched training run from
    # silently producing checkpoints stamped with the wrong feature_version or
    # from spawning actor subprocesses that cannot be cleanly reaped. Training
    # integration of V2 arrives in P05/P06.
    feature_version = getattr(flags, 'feature_version', 'legacy')
    if feature_version != 'legacy':
        raise ValueError(
            f"Training does not yet support feature_version="
            f"{feature_version!r}. Only 'legacy' is supported for training. "
            f"The observation V2 schema (douzero/observation/) is accepted by "
            f"configuration but is not yet wired into the actor/learner. "
            f"Training integration arrives in P05/P06."
        )

    # P02: training only supports the legacy ruleset. Standard mode adds
    # bidding/scoring which requires model/buffer changes (P05/P06).
    ruleset = getattr(flags, 'ruleset', 'legacy')
    if ruleset != 'legacy':
        raise ValueError(
            f"Training does not yet support ruleset={ruleset!r}. "
            f"Only 'legacy' is supported for training in P02. "
            f"Standard ruleset is available for evaluation and environment "
            f"testing via `evaluate.py --ruleset standard`. Training "
            f"integration arrives in P05/P06."
        )
    # P04: training only supports the legacy model_version. The "factorized"
    # model is a DEPLOYMENT-only, checkpoint-compatible forward that is
    # numerically equivalent to legacy under the same weights; the actor/learner
    # loop is not yet wired to it. Reject it here so a config that sets
    # model_version=factorized cannot start a training run that would silently
    # use the legacy forward while stamping the checkpoint with the wrong
    # model_version. Training integration of factorized/v2 arrives in P05/P06.
    model_version = getattr(flags, 'model_version', 'legacy')
    if model_version != 'legacy':
        raise ValueError(
            f"Training does not yet support model_version="
            f"{model_version!r}. Only 'legacy' is supported for training. "
            f"The 'factorized' model (P04) is a deployment-only forward "
            f"(DeepAgent backend='legacy_factorized') that is numerically "
            f"equivalent to legacy under the same weights; the actor/learner "
            f"loop is not yet wired to it. Training integration arrives in "
            f"P05/P06."
        )
    if flags.num_buffers < flags.batch_size:
        raise ValueError(
            f"num_buffers ({flags.num_buffers}) must be >= batch_size "
            f"({flags.batch_size})"
        )
    if getattr(flags, 'actor_torch_threads', 0) < 0:
        raise ValueError("actor_torch_threads must be >= 0")
    if getattr(flags, 'legacy_profile_sample_interval', 10) < 1:
        raise ValueError("legacy_profile_sample_interval must be >= 1")
    if getattr(flags, 'legacy_monitor_interval_seconds', 5.0) <= 0:
        raise ValueError("legacy_monitor_interval_seconds must be > 0")
    if getattr(flags, 'legacy_log_interval_seconds', 0.0) < 0:
        raise ValueError("legacy_log_interval_seconds must be >= 0")
    if getattr(flags, 'benchmark_warmup_frames', 0) < 0:
        raise ValueError("benchmark_warmup_frames must be >= 0")
    if getattr(flags, 'benchmark_warmup_frames', 0) >= flags.total_frames:
        raise ValueError("benchmark_warmup_frames must be less than total_frames")
    sidecar_retention = getattr(flags, 'checkpoint_sidecar_retention', 2)
    if sidecar_retention < -1:
        raise ValueError("checkpoint_sidecar_retention must be -1 or greater")
    if getattr(flags, 'legacy_reusable_pinned_staging', False):
        if not flags.actor_device_cpu:
            raise ValueError("reusable pinned staging is for CPU actors only")
        if not getattr(flags, 'legacy_contiguous_buffers', False):
            raise ValueError("reusable pinned staging requires contiguous buffers")
        if not getattr(flags, 'pin_memory', False):
            raise ValueError("reusable pinned staging requires --pin_memory")
    if getattr(flags, 'compile_actor', False):
        raise NotImplementedError(
            "V1 actor action counts are dynamic; compile_actor is unsupported"
        )
    if getattr(flags, 'legacy_actor_backend', 'legacy') == 'centralized_factorized':
        if not flags.actor_device_cpu:
            raise ValueError("centralized_factorized requires CPU actors")
        if flags.training_device == 'cpu':
            raise ValueError("centralized_factorized requires a CUDA training device")
        if getattr(flags, 'central_actor_max_actions', 512) < 64:
            raise ValueError("central_actor_max_actions must be >= 64")
        if getattr(flags, 'central_actor_microbatch', 4) < 1:
            raise ValueError("central_actor_microbatch must be >= 1")
        envs_per_actor = getattr(flags, 'central_actor_envs_per_actor', 4)
        minimum = getattr(flags, 'central_actor_min_microbatch', 2)
        target = getattr(flags, 'central_actor_target_microbatch', 8)
        maximum = getattr(flags, 'central_actor_max_microbatch', 16)
        capacity = getattr(flags, 'central_actor_max_pending_requests', 128)
        high_watermark = getattr(flags, 'central_actor_queue_high_watermark', 32)
        if envs_per_actor < 1:
            raise ValueError("central_actor_envs_per_actor must be >= 1")
        if not (1 <= minimum <= target <= maximum):
            raise ValueError(
                "central actor microbatches must satisfy 1 <= min <= target <= max"
            )
        if capacity < maximum:
            raise ValueError(
                "central_actor_max_pending_requests must be >= max microbatch"
            )
        if not 1 <= high_watermark <= capacity:
            raise ValueError("central actor high watermark exceeds queue capacity")
        if getattr(flags, 'central_actor_inference_deadline_ms', 10.0) <= 0:
            raise ValueError("central_actor_inference_deadline_ms must be positive")
        if getattr(flags, 'central_actor_learner_throttle_mode',
                   'fixed_threshold') not in {
                       'off', 'fixed_threshold', 'predicted_drain_time'
                   }:
            raise ValueError("invalid central_actor_learner_throttle_mode")
        if getattr(flags, 'central_actor_predicted_drain_target_ms', 10.0) <= 0:
            raise ValueError(
                "central_actor_predicted_drain_target_ms must be positive"
            )
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    T = flags.unroll_length
    B = flags.batch_size
    frames_per_update = T * B
    if flags.total_frames % frames_per_update:
        raise ValueError(
            "total_frames must be divisible by unroll_length * batch_size "
            f"({flags.unroll_length} * {flags.batch_size} = "
            f"{frames_per_update}); got {flags.total_frames}"
        )
    benchmark_warmup = getattr(flags, 'benchmark_warmup_frames', 0)
    if getattr(flags, 'legacy_metrics_path', ''):
        if benchmark_warmup % frames_per_update:
            raise ValueError(
                "benchmark_warmup_frames must be divisible by "
                "unroll_length * batch_size"
            )
        if (flags.total_frames - benchmark_warmup) % frames_per_update:
            raise ValueError(
                "benchmark measurement frames must be divisible by "
                "unroll_length * batch_size"
            )
    log_interval = getattr(flags, 'legacy_log_interval_seconds', 0.0)
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
        flush_interval_seconds=log_interval,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    ctx = mp.get_context('spawn')
    metric_store = (
        LegacyMetricStore(ctx)
        if (getattr(flags, 'legacy_profile', False)
            or getattr(flags, 'legacy_metrics_path', ''))
        else None
    )
    warmup_frames = getattr(flags, 'benchmark_warmup_frames', 0)
    measurement_started = warmup_frames == 0
    system_sampler = (
        _SystemSampler(_physical_gpu_identifier(flags.training_device))
        if metric_store is not None else None
    )
    if metric_store is not None and measurement_started:
        metric_store.reset()

    if getattr(flags, 'sync_interval_updates', 1) < 1:
        raise ValueError("sync_interval_updates must be >= 1")
    if getattr(flags, 'policy_snapshot_slots', 2) < 2:
        raise ValueError("policy_snapshot_slots must be >= 2")
    if getattr(flags, 'ddp_enabled', False):
        raise NotImplementedError(
            "The legacy three-role learner is not DDP-compatible; use the P14 "
            "V2 torchrun path. DDP helpers live in douzero.runtime.distributed."
        )
    if getattr(flags, 'compile_model', False):
        raise NotImplementedError(
            "compile_model is ambiguous for V1; use --compile_learner instead"
        )

    # Initialize immutable shared actor policy slots.
    models = {}
    actor_model_class = (
        LegacyFactorizedModel
        if getattr(flags, 'legacy_actor_backend', 'legacy') in {
            'factorized', 'centralized_factorized'
        }
        else Model
    )
    for device in device_iterator:
        slots = []
        for _ in range(getattr(flags, 'policy_snapshot_slots', 2)):
            model = actor_model_class(device=device)
            model.share_memory()
            model.eval()
            slots.append(model)
        owner_count = flags.num_actors
        if getattr(flags, 'legacy_actor_backend', 'legacy') == 'centralized_factorized':
            owner_count *= getattr(flags, 'central_actor_envs_per_actor', 4)
        models[device] = VersionedPolicyPool(
            slots, mp_context=ctx, max_owners=owner_count
        )

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    threads = []
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    amp_controllers = {}
    learner_device = (
        torch.device('cpu') if flags.training_device == 'cpu'
        else torch.device('cuda:' + str(flags.training_device))
    )
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        amp_controllers[position] = SafeMixedPrecision(
            learner_device,
            enabled=getattr(flags, 'amp_enabled', False),
            dtype=getattr(flags, 'amp_dtype', 'float16'),
            fallback_on_nonfinite=getattr(flags, 'amp_fallback_on_nonfinite', True),
        )

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)
    learner_forward_models = learner_model.get_models()
    if getattr(flags, 'compile_learner', False):
        if not hasattr(torch, 'compile'):
            raise RuntimeError("compile_learner requires torch.compile")
        learner_forward_models = {
            position: torch.compile(model)
            for position, model in learner_model.get_models().items()
        }

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
        'policy_lag_mean_landlord',
        'policy_lag_mean_landlord_up',
        'policy_lag_mean_landlord_down',
        'policy_lag_max_landlord',
        'policy_lag_max_landlord_up',
        'policy_lag_max_landlord_down',
        'amp_fallbacks_landlord',
        'amp_fallbacks_landlord_up',
        'amp_fallbacks_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}
    resumed_learner_updates = 0

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        from douzero.checkpoint import load_checkpoint

        expected_feature = getattr(flags, "feature_version", "legacy")
        expected_ruleset = getattr(flags, "ruleset", "legacy")
        checkpoint_states, ckpt_manifest = load_checkpoint(
            checkpointpath,
            expected_feature_version=expected_feature,
            expected_ruleset_id=expected_ruleset,
            training_device=flags.training_device,
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
        runtime_state = checkpoint_states.get('runtime_state', {})
        resumed_learner_updates = int(runtime_state.get('learner_updates', 0))
        amp_state = runtime_state.get('amp_controllers', {})
        for position, controller_state in amp_state.items():
            if position in amp_controllers:
                amp_controllers[position].load_state_dict(controller_state)
        # Old checkpoints do not contain P14 lag/AMP counters. Merge rather
        # than replace so they resume with zero-valued new diagnostics.
        stats.update(checkpoint_states["stats"])
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    for device in device_iterator:
        models[device].initialize(
            learner_model.get_models(), version=resumed_learner_updates
        )

    positions = ('landlord', 'landlord_up', 'landlord_down')
    transactions = _TrainingTransactions(positions)
    position_locks = transactions.position_locks
    training_state_lock = transactions.state_lock
    publish_lock = threading.Lock()
    learner_updates = resumed_learner_updates
    last_published_version = resumed_learner_updates

    def current_learner_updates():
        with training_state_lock:
            return learner_updates

    def publish_snapshot_if_due(committed_version):
        nonlocal last_published_version
        interval = getattr(flags, 'sync_interval_updates', 1)
        if committed_version % interval:
            return
        with publish_lock:
            # Freeze all role learners while copying a coherent snapshot.
            with transactions.freeze_updates():
                with training_state_lock:
                    snapshot_version = learner_updates
                if snapshot_version <= last_published_version:
                    return
                source = learner_model.get_models()
                for pool in models.values():
                    publish_started_ns = time.perf_counter_ns()
                    published = pool.publish(source, version=snapshot_version)
                    if metric_store is not None:
                        metric_store.add_learner({
                            'snapshot_publish_ns': (
                                time.perf_counter_ns() - publish_started_ns
                            ),
                            'snapshot_publishes': int(published),
                            'snapshot_skips': int(not published),
                        })
                last_published_version = snapshot_version

    # Starting actor processes
    stop_event = ctx.Event()
    learner_supervisor = _LearnerThreadSupervisor(stop_event)
    central_process = None
    central_slots = None
    central_request_queue = None
    central_response_queues = None
    central_queue_pressure = None
    if getattr(flags, 'legacy_actor_backend', 'legacy') == 'centralized_factorized':
        central_slots = CentralizedInferenceSlots(
            flags.num_actors, getattr(flags, 'central_actor_max_actions', 512),
            getattr(flags, 'central_actor_envs_per_actor', 4),
        )
        central_request_queue = ctx.Queue(maxsize=getattr(
            flags, 'central_actor_max_pending_requests', 128
        ))
        central_response_queues = [ctx.Queue() for _ in range(flags.num_actors)]
        central_queue_pressure = CentralQueuePressure(
            ctx,
            flags.num_actors * getattr(flags, 'central_actor_envs_per_actor', 4),
        )
        central_process = ctx.Process(
            target=centralized_inference_loop,
            name='legacy-centralized-inference',
            args=(
                flags.training_device,
                models['cpu'],
                central_slots,
                central_request_queue,
                central_response_queues,
                stop_event,
                getattr(flags, 'central_actor_min_microbatch', 2),
                getattr(flags, 'central_actor_target_microbatch', 8),
                getattr(flags, 'central_actor_max_microbatch', 16),
                getattr(flags, 'central_actor_max_delay_ms', 2.0),
                getattr(flags, 'central_actor_max_pending_requests', 128),
                getattr(flags, 'central_actor_use_stream_priority', True),
                getattr(flags, 'central_actor_async_policy_copy', True),
                metric_store,
                central_queue_pressure,
            ),
        )
        central_process.start()
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device],
                      models[device], buffers[device], flags, stop_event,
                      metric_store, central_slots, central_request_queue,
                      (central_response_queues[i]
                       if central_response_queues is not None else None),
                      central_queue_pressure))
            actor.start()
            actor_processes.append((actor, device, i))

    def stop_workers():
        """Wake blocked workers, reap actors, and join learner threads."""
        stop_event.set()
        if central_queue_pressure is not None:
            central_queue_pressure.invalidate()
        if central_response_queues is not None:
            for response_queue in central_response_queues:
                response_queue.put((
                    'shutdown', -1, -1, -1, -1, 'training is shutting down'
                ))
            try:
                central_request_queue.put(None, timeout=1)
            except queue.Full:
                pass
        for device in device_iterator:
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                for _ in range(flags.num_actors):
                    free_queue[device][position].put(None)
                for _ in range(flags.num_threads):
                    full_queue[device][position].put(None)
        for actor, device, actor_id in actor_processes:
            actor.join(timeout=5)
            if actor.is_alive():
                actor.terminate()
                actor.join(timeout=5)
            if actor.is_alive():
                log.error(
                    'Actor %i on device %s could not be reaped.', actor_id, device
                )
            else:
                env_count = (
                    getattr(flags, 'central_actor_envs_per_actor', 4)
                    if getattr(flags, 'legacy_actor_backend', 'legacy')
                    == 'centralized_factorized' else 1
                )
                for env_slot in range(env_count):
                    models[device].recover_owner(actor_id * env_count + env_slot)
        if central_process is not None:
            central_process.join(timeout=5)
            if central_process.is_alive():
                central_process.terminate()
                central_process.join(timeout=5)
            if central_process.is_alive():
                log.error('Centralized inference process could not be reaped.')
        learner_join_deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(
                timeout=max(0.0, learner_join_deadline - time.monotonic())
            )
            if thread.is_alive():
                log.error(
                    'Learner thread %s did not stop within 5 seconds.', thread.name
                )

    update_budget = _UpdateBudget(frames, flags.total_frames, frames_per_update)

    def batch_and_learn(i, device, position, local_lock, position_lock):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats, measurement_started
        nonlocal learner_updates
        profile_sequence = 0
        stager = None
        if getattr(flags, 'legacy_reusable_pinned_staging', False):
            stager = PinnedBatchStager(buffers[device][position], flags)
        while not stop_event.is_set():
            reservation = update_budget.reserve()
            if reservation is None:
                return
            try:
                batch_result = get_batch(
                    free_queue[device][position], full_queue[device][position],
                    buffers[device][position], flags, local_lock,
                    stager=stager,
                    return_timings=metric_store is not None,
                )
                if batch_result is None:
                    return
                if metric_store is not None:
                    batch, batch_timings = batch_result
                else:
                    batch, batch_timings = batch_result, {}
                if (central_queue_pressure is not None
                        and getattr(flags, 'central_actor_learner_throttle', False)):
                    waited, throttle_ns = wait_for_learner_admission(
                        central_queue_pressure, stop_event,
                        getattr(
                            flags, 'central_actor_learner_throttle_mode',
                            'fixed_threshold',
                        ),
                        high_watermark=getattr(
                            flags, 'central_actor_queue_high_watermark', 32
                        ),
                        deadline_ms=getattr(
                            flags, 'central_actor_inference_deadline_ms', 10.0
                        ),
                        drain_target_ms=getattr(
                            flags, 'central_actor_predicted_drain_target_ms', 10.0
                        ),
                    )
                    if metric_store is not None and waited:
                        metric_store.add_throttle(throttle_ns)
                profile_sequence += 1
                profile_step = (
                    getattr(flags, 'legacy_profile', False)
                    and profile_sequence
                    % getattr(flags, 'legacy_profile_sample_interval', 10) == 0
                )
                with transactions.update(position):
                    _stats, learner_timings = learn(
                        position, models, learner_forward_models[position], batch,
                        optimizers[position], flags, position_lock,
                        amp_controller=amp_controllers[position],
                        learner_updates=current_learner_updates(),
                        profile=profile_step,
                        stager=stager,
                    )

                    # The model/optimizer step and all persisted progress fields
                    # are one role transaction relative to checkpoint().
                    with training_state_lock:
                        frames += frames_per_update
                        position_frames[position] += frames_per_update
                        learner_updates += 1
                        committed_version = learner_updates
                        for k in _stats:
                            stats[k] = _stats[k]
                        update_budget.commit(reservation)
                        reservation = None
                        started_now = False
                        if (metric_store is not None and not measurement_started
                                and frames >= warmup_frames):
                            metric_store.reset()
                            measurement_started = True
                            started_now = True
                        to_log = None
                        if log_interval == 0.0:
                            to_log = dict(frames=frames)
                            to_log.update({k: stats[k] for k in stat_keys})

                log_write_ns = 0
                if to_log is not None:
                    log_started_ns = time.perf_counter_ns()
                    plogger.log(to_log)
                    log_write_ns = time.perf_counter_ns() - log_started_ns

                publish_snapshot_if_due(committed_version)
                if metric_store is not None and not started_now:
                    metric_store.add_learner(
                        {
                            'updates': 1,
                            'profile_samples': int(profile_step),
                            'frames': frames_per_update,
                            'log_write_ns': log_write_ns,
                            'log_writes': int(log_interval == 0.0),
                            **batch_timings,
                            **learner_timings,
                        },
                        position=position,
                        queue_wait_ns=batch_timings.get('batch_wait_ns', 0),
                        mean_policy_lag=(
                            _stats['policy_lag_mean_' + position]
                        ),
                        max_policy_lag=(
                            _stats['policy_lag_max_' + position]
                        ),
                    )
            finally:
                if reservation is not None:
                    update_budget.cancel(reservation)

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    locks = {}
    for device in device_iterator:
        locks[device] = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=learner_supervisor.run,
                    name='batch-and-learn-%d-%s-%s' % (i, device, position),
                    args=(
                        batch_and_learn,
                        i,
                        device,
                        position,
                        locks[device][position],
                        position_locks[position],
                    ),
                )
                thread.start()
                threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        from douzero.checkpoint import save_checkpoint

        log.info('Saving checkpoint to %s', checkpointpath)
        with transactions.snapshot():
            # If a learner failed after mutating an optimizer but before its
            # progress commit, preserve the last good checkpoint instead of
            # serializing that failed in-flight generation.
            learner_supervisor.raise_if_failed()
            checkpoint_frames = frames
            checkpoint_position_frames = dict(position_frames)
            checkpoint_stats = dict(stats)
            checkpoint_learner_updates = learner_updates
            _models = learner_model.get_models()
            save_checkpoint(
                checkpointpath,
                learner_models=_models,
                optimizers=optimizers,
                stats=checkpoint_stats,
                flags=flags,
                frames=checkpoint_frames,
                position_frames=checkpoint_position_frames,
                runtime_state={
                    'amp_controllers': {
                        position: controller.state_dict()
                        for position, controller in amp_controllers.items()
                    },
                    'learner_updates': checkpoint_learner_updates,
                },
            )

            # Evaluation sidecars use the same atomic disk protocol as model.tar.
            _save_legacy_sidecars(
                learner_model,
                os.path.dirname(checkpointpath),
                checkpoint_frames,
                sidecar_retention,
            )

    fps_log = []
    timer = timeit.default_timer

    def final_metrics(status):
        if metric_store is None or not getattr(flags, 'legacy_metrics_path', ''):
            return
        payload = metric_store.snapshot()
        payload.update({
            'status': status,
            'frames_total': frames,
            'position_frames': dict(position_frames),
            'learner_updates_total': learner_updates,
            'config': dict(vars(flags)),
            'stats': dict(stats),
            'system': system_sampler.report() if system_sampler is not None else {},
            'cuda': {
                'available': torch.cuda.is_available(),
                'peak_allocated_mib': (
                    torch.cuda.max_memory_allocated(learner_device) / 1024 ** 2
                    if learner_device.type == 'cuda' else None
                ),
                'peak_reserved_mib': (
                    torch.cuda.max_memory_reserved(learner_device) / 1024 ** 2
                    if learner_device.type == 'cuda' else None
                ),
            },
            'workers': {
                'actors': [
                    {'device': str(device), 'actor_id': actor_id,
                     'exitcode': actor.exitcode, 'alive': actor.is_alive()}
                    for actor, device, actor_id in actor_processes
                ],
                'learner_threads_alive': sum(thread.is_alive() for thread in threads),
                'centralized_inference': (
                    None if central_process is None else {
                        'exitcode': central_process.exitcode,
                        'alive': central_process.is_alive(),
                    }
                ),
            },
        })
        write_metrics(flags.legacy_metrics_path, payload)

    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        last_periodic_log = timer()
        while frames < flags.total_frames:
            learner_supervisor.raise_if_failed()
            for actor, device, actor_id in actor_processes:
                if actor.exitcode not in (None, 0):
                    raise RuntimeError(
                        f'Actor {actor_id} on device {device} exited with '
                        f'code {actor.exitcode}'
                    )
            if (central_process is not None
                    and central_process.exitcode not in (None, 0)):
                raise RuntimeError(
                    'Centralized inference process exited with code '
                    f'{central_process.exitcode}'
                )
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            stop_event.wait(
                timeout=getattr(flags, 'legacy_monitor_interval_seconds', 5.0)
            )
            learner_supervisor.raise_if_failed()

            if system_sampler is not None and measurement_started:
                sampled_pids = (
                    [os.getpid()]
                    + [actor.pid for actor, _, _ in actor_processes]
                    + ([central_process.pid] if central_process is not None else [])
                )
                system_sampler.sample(sampled_pids)

            if (log_interval > 0.0
                    and timer() - last_periodic_log >= log_interval):
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                log_started_ns = time.perf_counter_ns()
                plogger.log(to_log)
                if metric_store is not None:
                    metric_store.add_learner({
                        'log_write_ns': time.perf_counter_ns() - log_started_ns,
                        'log_writes': 1,
                    })
                last_periodic_log = timer()

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint()
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))
        learner_supervisor.raise_if_failed()

    except KeyboardInterrupt:
        stop_workers()
        final_metrics('interrupted')
        plogger.close(successful=False)
        return 
    except BaseException:
        stop_workers()
        final_metrics('failed')
        plogger.close(successful=False)
        raise
    else:
        stop_workers()
        log.info('Learning finished after %d frames.', frames)

    checkpoint()
    final_metrics('completed')
    plogger.close(successful=True)
