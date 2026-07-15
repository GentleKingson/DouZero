import os
import queue
import threading
import time
import timeit
import pprint
from contextlib import ExitStack
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act
from douzero.runtime import SafeMixedPrecision, VersionedPolicyPool

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}


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

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock,
          amp_controller=None,
          published_version=0):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    non_blocking = bool(getattr(flags, 'pin_memory', False))
    obs_x_no_action = batch['obs_x_no_action'].to(device, non_blocking=non_blocking)
    obs_action = batch['obs_action'].to(device, non_blocking=non_blocking)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device, non_blocking=non_blocking), 0, 1).float()
    target = torch.flatten(batch['target'].to(device, non_blocking=non_blocking), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
        
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
        )
        loss = step_result.loss
        versions = batch.get('policy_version')
        policy_lag = 0.0
        if versions is not None:
            policy_lag = max(
                0.0,
                float(published_version) - float(versions.float().mean().item()),
            )
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position: loss.item(),
            'policy_lag_'+position: policy_lag,
            'amp_fallbacks_'+position: float(amp_controller.fallback_count),
        }
        return stats

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
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    ctx = mp.get_context('spawn')

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
            "compile_model is supported by train_v2.py only; the legacy actor "
            "model has variable per-action forwards and remains eager."
        )

    # Initialize immutable shared actor policy slots.
    models = {}
    for device in device_iterator:
        slots = []
        for _ in range(getattr(flags, 'policy_snapshot_slots', 2)):
            model = Model(device=device)
            model.share_memory()
            model.eval()
            slots.append(model)
        models[device] = VersionedPolicyPool(
            slots, mp_context=ctx, max_owners=flags.num_actors
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

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
        'policy_lag_landlord',
        'policy_lag_landlord_up',
        'policy_lag_landlord_down',
        'amp_fallbacks_landlord',
        'amp_fallbacks_landlord_up',
        'amp_fallbacks_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

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
        # Old checkpoints do not contain P14 lag/AMP counters. Merge rather
        # than replace so they resume with zero-valued new diagnostics.
        stats.update(checkpoint_states["stats"])
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    for device in device_iterator:
        models[device].initialize(learner_model.get_models())

    position_locks = {
        'landlord': threading.Lock(),
        'landlord_up': threading.Lock(),
        'landlord_down': threading.Lock(),
    }
    publish_lock = threading.Lock()
    learner_updates = 0

    def publish_snapshot_if_due():
        nonlocal learner_updates
        with publish_lock:
            learner_updates += 1
            interval = getattr(flags, 'sync_interval_updates', 1)
            if learner_updates % interval:
                return
            # Freeze all role learners while copying a coherent snapshot.
            with ExitStack() as stack:
                for position in ['landlord', 'landlord_up', 'landlord_down']:
                    stack.enter_context(position_locks[position])
                source = learner_model.get_models()
                for pool in models.values():
                    pool.publish(source, version=learner_updates)

    # Starting actor processes
    stop_event = ctx.Event()
    learner_supervisor = _LearnerThreadSupervisor(stop_event)
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device],
                      models[device], buffers[device], flags, stop_event))
            actor.start()
            actor_processes.append((actor, device, i))

    def stop_workers():
        """Wake blocked workers, reap actors, and join learner threads."""
        stop_event.set()
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
                models[device].recover_owner(actor_id)
        learner_join_deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(
                timeout=max(0.0, learner_join_deadline - time.monotonic())
            )
            if thread.is_alive():
                log.error(
                    'Learner thread %s did not stop within 5 seconds.', thread.name
                )

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while not stop_event.is_set() and frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            if batch is None:
                return
            _stats = learn(position, models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock,
                amp_controller=amp_controllers[position],
                published_version=models[device].version)
            publish_snapshot_if_due()

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

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
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        from douzero.checkpoint import save_checkpoint

        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        save_checkpoint(
            checkpointpath,
            learner_models=_models,
            optimizers=optimizers,
            stats=stats,
            flags=flags,
            frames=frames,
            position_frames=position_frames,
        )

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            learner_supervisor.raise_if_failed()
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            stop_event.wait(timeout=5)
            learner_supervisor.raise_if_failed()

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
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
        plogger.close()
        return 
    except BaseException:
        stop_workers()
        plogger.close()
        raise
    else:
        stop_workers()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
