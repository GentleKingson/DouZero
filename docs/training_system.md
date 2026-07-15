# P14 Training System

P14 removes the actor/learner weight-update race and adds opt-in mixed
precision, DDP runtime support, data-transfer controls, and a measured training
profiler. Legacy rules, observations, rewards, model outputs, and checkpoint
layouts are unchanged.

## Actor policy publication

The legacy learner no longer calls `load_state_dict()` on a model that an actor
may be forwarding. Each actor device owns a `VersionedPolicyPool` with at least
two shared-memory slots:

1. An actor leases the active slot and version before a game.
2. The learner copies all three role models into an unused, unleased slot.
3. Only after the complete copy succeeds does it atomically flip the active
   slot and version.
4. The actor releases its lease and acquires the latest snapshot after terminal,
   before the first inference of the next game.

If every inactive slot is still leased, publication is skipped rather than
overwriting an in-flight policy. The next configured publication point retries.
Each transition records `policy_version`; learner logs report per-role policy
lag. `sync_interval_updates` controls publication frequency.

Leases carry an actor owner and generation, so duplicate releases cannot steal
another actor's reader count. After an actor has exited or been terminated, the
parent reclaims any lease still registered to that actor. Shutdown also sends
sentinels to both actor free queues and learner full queues before joining every
learner thread with a bounded timeout. Learner-thread exceptions are returned
to the monitoring thread, which requests shutdown and re-raises the original
exception instead of leaving a partially trained role running silently.

## Mixed precision

`amp_enabled` defaults to false. CUDA uses `torch.autocast` and the current
`torch.amp.GradScaler` API. CPU AMP is accepted only when the user explicitly
selects `amp_dtype: bfloat16`; CPU float16 is rejected. Loss and gradient norm
are checked before every optimizer mutation. An AMP anomaly disables AMP and
retries once in float32; another anomaly raises `FloatingPointError`.
Under DDP, loss and gradient finite flags are reduced across ranks before the
next collective-sensitive action. Any anomaly makes every rank retry together;
the trainer restores Python, NumPy, Torch, CUDA, and trainer RNG state so the
retry uses the same BC selections and dropout stream.

Example CUDA legacy learner settings:

```yaml
amp_enabled: true
amp_dtype: float16
amp_fallback_on_nonfinite: true
pin_memory: true
```

## DDP

The V2 entry point accepts one-process-per-device DDP through `torchrun`.
Explicit nonzero seeds are derived per rank, DDP averages gradients, and public
replay samples are collected independently per rank. The runtime exposes
non-overlapping rank shards for future dataset-backed training. Replay
readiness is reduced across ranks before each optimizer step; all ranks either
enter backward or skip the step together. Console summaries are rank-zero only.
Self-play forwards call the rank-local underlying module, never the DDP wrapper;
only synchronized optimizer closures enter DDP forward/backward. The wrapper
uses a static reducer, avoiding the recurring graph traversal cost of
`find_unused_parameters=True`, for ordinary FP32 training. AMP-enabled runs use
the dynamic reducer because a globally non-finite loss can abandon a forward
before backward and then retry it coherently in FP32.

Curriculum/coach-label output and RL+BC validation/quarantine are currently
rejected under DDP because those modes do not yet have coordinated single-writer
side effects. Disable those features for DDP, or run them in one process; the
entry point fails before opening their output files.

DDP also rejects enabled optional trainable heads when their corresponding loss
is disabled. In particular, `human_prior_enabled` requires BC loss and
`strategy_aux_enabled` requires at least one strategy auxiliary weight; because
RL+BC is currently unsupported under DDP, the prior head must remain disabled.

```bash
torchrun --standalone --nproc-per-node=2 train_v2.py \
  --config configs/enhanced.yaml --ddp_enabled --ddp_backend nccl \
  --device cuda --episodes 8 --optimizer_steps 2
```

CPU/gloo is supported for smoke validation. The legacy three-role DMC entry
point rejects `ddp_enabled` explicitly rather than silently pretending to use
DDP. Checkpoint loading should use the rank-local
`douzero.runtime.distributed.checkpoint_map_location` helper; the existing
legacy checkpoint format is unchanged.

## Compile and transfer controls

`pin_memory` enables pinned CPU batches and non-blocking learner transfers.
It is disabled by default for portable CPU execution. `compile_model` is an
opt-in V2 forward feature using dynamic shapes; enable it only after the local
benchmark shows that steady-state savings exceed compilation cost. Variable
legal-action counts remain supported.

## Profiling and benchmark

Run the P14 comparison with:

```bash
python benchmarks/bench_training_system.py --rounds 10
```

It writes JSON and Markdown under `artifacts/benchmark/` and separately times
actor environment steps, observation encoding, queue wait, learner
forward/backward, weight publication, and legacy/factorized/V2 forward paths.
Unavailable CUDA AMP and DDP paths are recorded as `not_run`, never estimated.
The first compile is intentionally not mixed into steady-state forward timing.
