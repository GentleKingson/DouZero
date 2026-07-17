# P14 Training System

P14 removes the actor/learner weight-update race and adds opt-in mixed
precision, base V2 card-play DDP support, data-transfer controls, and a
measured training profiler. Legacy rules, observations, rewards, model outputs,
and checkpoint layouts are unchanged. Later default-off graphs have narrower
support boundaries described below.

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

The V2 entry point accepts one-process-per-device DDP through `torchrun` for
compatible legacy-ruleset card-play configurations.
Explicit nonzero seeds are derived per rank, DDP averages gradients, and public
replay samples are collected independently per rank. The runtime exposes
non-overlapping rank shards for future dataset-backed training. Replay
readiness is reduced across ranks before each optimizer step; all ranks either
enter backward or skip the step together. Console summaries are rank-zero only.
Self-play forwards call the rank-local underlying module, never the DDP wrapper;
only synchronized optimizer closures enter DDP forward/backward. The wrapper
uses a static reducer, avoiding the recurring graph traversal cost of
`find_unused_parameters=True`. When a globally non-finite loss invalidates an
AMP forward, every rank completes a sanitized synchronization-only backward to
drain that reducer iteration, discards those gradients, and then retries the
same RNG state coherently in FP32.

Curriculum/coach-label output and RL+BC validation/quarantine are currently
rejected under DDP because those modes do not yet have coordinated single-writer
side effects. Disable those features for DDP, or run them in one process; the
entry point fails before opening their output files.

DDP also rejects enabled optional trainable heads when their corresponding loss
is disabled. In particular, `human_prior_enabled` requires BC loss and
`strategy_aux_enabled` requires at least one strategy auxiliary weight; because
RL+BC is currently unsupported under DDP, the prior head must remain disabled.

P17 standard learned bidding is **not** DDP-enabled: `V2Trainer` rejects a
standard `RuleSet` under DDP before collection because the mixed auction and
card-play graph has not been validated across ranks. P17 `joint` and
`alternating` belief modes also fail closed because BeliefModel gradients are
not synchronized; only `belief_training_mode=frozen` can participate in a
compatible DDP run. These are implementation limitations, not merely missing
GPU measurements, and a successful base V2 DDP smoke does not clear them.

In single-process standard training, bidding policy credit is source-aware.
Explicit rule demonstrations use masked CE. Learned and exploratory behavior
actions are resolved from their neutral physical seat to the final role and
fit only the selected bid's actor-win value logit with a bounded binary loss. A
failed selected bid is therefore pushed down while a successful bid is pushed
up; neither is treated as a self-imitation class label. All-pass redeals and
the redeal-cap guard remain excluded from this objective.
This fitted bidding value is a terminal win-probability target. It does not
claim to optimize an action-conditioned ADP score; the scalar landlord score
head remains an auxiliary state prediction.
The reserved `lambda_bid_regret` setting must remain zero because regret needs
a separate action-value head; nonzero values fail before collection.

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

Single-process P17 trainer checkpoints bind the model, ruleset, bidding,
belief, loss, policy snapshot, and full source Git SHA. Resume rejects an
unknown or different source SHA before restoring model/optimizer state; a
wheel or source archive without Git metadata must set `DOUZERO_GIT_SHA` to the
exact build commit. Standard and joint trainer checkpoint save/resume remains
single-process only and fails closed when DDP is enabled. Trainer-checkpoint
format 3 records `training_topology=single_process` and
`training_world_size=1`; formats 1/2 and any topology mismatch are rejected
rather than inferred to be compatible.

## Compile and transfer controls

`pin_memory` enables pinned CPU batches and non-blocking learner transfers.
It is disabled by default for portable CPU execution. `compile_model` is an
opt-in V2 forward feature using dynamic shapes; enable it only after the local
benchmark shows that steady-state savings exceed compilation cost. Variable
legal-action counts remain supported. Compilation fails closed for
bidding-enabled models because auctions use the separate `forward_bidding`
contract, and for joint/alternating belief because the differentiable coupled
graph has not been validated. Frozen, bidding-disabled base V2 is the supported
compile scope.

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
