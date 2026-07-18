# P14 Training System

P14 removes the actor/learner weight-update race and adds opt-in mixed
precision, base V2 card-play DDP support, data-transfer controls, and a
measured training profiler. Legacy rules, observations, rewards, model outputs,
and checkpoint layouts are unchanged. Later default-off graphs have narrower
support boundaries described below.

For native Windows, WSL2, and Linux device/topology boundaries, including the
difference between legacy CUDA actors and a CUDA learner, see
[`docs/windows_training.md`](windows_training.md).

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
format 3 remains the single-process format and records
`training_topology=single_process` plus `training_world_size=1`. Async runs use
format 4 and additionally bind actor count, compact replay schema, snapshot
publication semantics, and request ordering/batching semantics. Format 3 has
an explicit single-process compatibility path; formats 1/2, unknown versions,
unknown topology, and cross-topology resume are rejected.

## V2 single-GPU throughput topology

`--v2_training_mode` defaults to `single_process`. The opt-in
`async_single_gpu` topology uses `spawn` CPU actors, preallocated CPU shared
observation/replay slots, and queues containing only slot IDs plus small
metadata. Actors make epsilon decisions with their local RNG and never own a
CUDA tensor. The main process owns an independent inference model and learner
model in one CUDA context, groups requests by immutable policy snapshot,
action-count bucket and role, and publishes a complete inference state only at
a quiescent boundary. Both replay implementations sample uniformly across all
resident transitions first, including action buckets smaller than the learner
batch size. The learner then partitions the sampled records by action-count
bucket and forwards each homogeneous sub-batch separately, reducing padding
without starving rare decisions. Async replay drain uses incremental batched
insertion and O(1) bucket eviction rather than rebuilding every bucket for each
transition.

Every compact transition is validated twice at the async trust boundary:
before its shared replay slot is returned to an actor, and again before
batched insertion. Validation binds the feature-schema hash and compact schema
version, checks every tensor's CPU device, dtype and structural shape, verifies
the acting role and selected legal action, and rejects non-finite, partial, or
out-of-domain labels and malformed policy provenance.

Each inference group packs win, conditional-score, probability, and expected
score outputs into one contiguous `[B, A, 5]` tensor and performs one blocking
group-level device-to-host copy. CUDA events measure inference work without a
hot-path global device synchronization; shared-slot writes then slice the
single CPU staging tensor.

Tensor value validation occurs while model-input bundles are still on CPU.
CUDA model guards use asynchronous device assertions rather than Python
`bool(tensor)` scalar reads, avoiding mask, role, and chosen-index host
synchronization in the inference and learner hot paths.

| Combination | `single_process` | `async_single_gpu` |
|---|---:|---:|
| Legacy-ruleset base V2 | yes | yes |
| Standard bidding | yes | fail closed |
| League / curriculum | yes | fail closed |
| RL+BC / human prior | yes | fail closed |
| Style / strategy | yes | fail closed |
| Frozen/joint/alternating belief | yes | fail closed |
| DDP | existing scoped support | rejected |

Async startup requires CUDA and never silently falls back. Request timeout,
worker exit, invalid state transition, and bounded shutdown all fail the run
instead of waiting indefinitely. Abort and shutdown are spawn-shared events,
not process-local flags; slot acquisition, response waits, replay-slot waits,
actor task loops, and the coordinator service all observe the same state. The
first failure reason is stored in shared memory so a failure in one actor wakes
other actors blocked on slots without waiting for their request timeout. Cycle
quiescence requires no active games,
no WRITING/READY/RUNNING slots, and no completed episode waiting to enter
compact replay. Replay is then cleared through the trainer lifecycle API,
preserving the existing P0 cycle-boundary rule.

## Long-running V2 state machine

`train_v2.py` keeps its original one-shot behavior unless `--long_running` is
explicitly present. Long-running single-process V2 uses this state machine:

```text
check limits -> collect K episodes -> optimize U steps -> advance policy step
-> cycle boundary -> atomic checkpoint -> optional evaluation -> metrics -> repeat
```

Any configured maximum is a stop condition: cycles, cumulative episodes,
cumulative optimizer steps, or wall-clock minutes. Episode and optimizer-step
targets are clipped in the final cycle. Wall-clock expiry, `SIGINT`, and
`SIGTERM` request a stop; collection and optimization finish the current cycle,
then the process checkpoints at the safe boundary and exits with an explicit
`stop_reason`. `--no-save_on_interrupt` disables the extra signal/stop-event
save, but not a checkpoint already due by another schedule.
Wall time is cumulative across resumes: every clean boundary stores
`total_wall_seconds`, and a resumed process receives only the remaining part
of `--max_wall_time_minutes`. The budget includes checkpoint publication,
rotation, periodic evaluation, and metrics emission. Post-publication time is
persisted atomically in the latest run-state manifest without rewriting the
immutable model checkpoint. A collect-only configuration with
`optimizer_steps_per_cycle=0` rejects an otherwise unreachable optimizer-step
limit unless another cycle, episode, or wall-time limit can stop the run.

Checkpoints are immutable sequence files written to a temporary file and
published with `os.replace`. Only after that succeeds is `*-latest.json`
atomically replaced. Rotation retains the newest `--keep_last_checkpoints`
files. A failed save never removes or repoints the previous valid checkpoint.
Before fresh-run checks or resume reconciliation, the controller atomically
creates a per-series ownership lock and holds it until training exits. A second
fresh or resumed process fails closed. Locks are never stolen automatically;
after a crash, an operator must verify the recorded PID is gone before
explicitly removing the stale lock.
Each series has a persistent run ID in its state and filenames. Starting a
fresh run against an existing manifest or matching files fails closed; resume
through the manifest, or choose a new `--checkpoint_path`, rather than
silently overwriting another run. Manifest resume also binds the output series:
omitting `--checkpoint_path` continues the manifest's series, while an explicit
mismatching path is rejected. Manifest identity and counters are checked
against the checkpoint, and resuming an older sequence than `latest` fails
closed. Direct checkpoint resume performs the same whole-series scan as
manifest resume. Duplicate sequence numbers, skipped sequences, and ignored
orphans fail closed. A copied standalone cycle file cannot implicitly create
or fork a series; either its matching latest manifest or the initial
publication-intent sidecar is required.
Each checkpoint includes cumulative trainer statistics, cycle state, policy
version and step, optimizer/mixed-precision state, all RNG state, and the
existing strict source/model/feature/rules/loss/bidding/belief identity.

Replay is deliberately not checkpointed. Every completed cycle is an
empty-replay boundary, including cycles where no file is due. Resume therefore
continues only from a clean cycle boundary and never fabricates in-progress
replay. With the same seed, identity, `episodes_per_cycle`, and
`optimizer_steps_per_cycle`, N+M uninterrupted cycles match N cycles followed
by resume for M cycles in cumulative counts, policy step, model weights, and
optimizer state. Changing either cycle-shape field fails closed; operational
stop limits and retention may change on resume.

If a checkpoint file was atomically published but the process stopped before
the latest manifest could be replaced, the next manifest resume validates the
single contiguous orphan checkpoint and promotes it. Multiple or invalid
orphans fail closed. Rotation cleanup failures are recorded separately and do
not invalidate a checkpoint whose file and manifest were already published.
An atomic pending-intent sidecar makes the same recovery possible when the
very first checkpoint succeeds but creation of the first latest manifest
fails; it is removed after successful manifest publication.

```bash
python train_v2.py --long_running --config configs/enhanced.yaml --seed 17 \
  --episodes_per_cycle 64 --optimizer_steps_per_cycle 16 \
  --max_wall_time_minutes 720 --max_total_optimizer_steps 100000 \
  --checkpoint_path runs/v2/train.pt --checkpoint_every_cycles 1 \
  --checkpoint_every_steps 100 --checkpoint_every_minutes 30 \
  --keep_last_checkpoints 5 --metrics_path runs/v2/metrics.json
```

Resume through the stable manifest:

```bash
python train_v2.py --long_running --config configs/enhanced.yaml --seed 17 \
  --episodes_per_cycle 64 --optimizer_steps_per_cycle 16 \
  --max_total_optimizer_steps 200000 --checkpoint_path runs/v2/train.pt \
  --resume_checkpoint runs/v2/train-latest.json
```

Periodic evaluation delegates to an existing evaluation command. It does not
implement rules or another evaluator. The command is tokenized without a shell;
`{checkpoint}` and `{cycle}` are replaced at runtime. Evaluation failures are
recorded and fail fast by default; `--no-eval_fail_fast` explicitly records and
continues. If the evaluation subprocess exits because the controller received
`SIGINT` or `SIGTERM`, it is recorded as interrupted and the run follows normal
safe-boundary signal shutdown instead of treating it as a fail-fast error. The
evaluator runs in a separate process group; the controller polls its stop event,
requests group termination, and escalates to a kill after a bounded grace
period rather than waiting indefinitely for evaluation to return.

```bash
python train_v2.py --long_running --episodes_per_cycle 32 \
  --optimizer_steps_per_cycle 8 --max_cycles 20 --eval_every_cycles 5 \
  --eval_command ".venv/bin/python evaluate_paired.py --candidate {checkpoint} --baseline /path/to/baseline --num-deals 1000 --output runs/v2/eval-{cycle}.json"
```

Cycle metrics contain cumulative counts, cycle/collection/optimization time,
AMP fallback delta, checkpoint path/status/error, resume source, evaluation
status/error, and peak CUDA memory when available. CPU reports peak memory as
unavailable. They also include total game decisions, trainable decisions,
transitions, and learner steps per second,
requests/actions per microbatch, inference queue p50/p95, inference GPU time,
replay occupancy, active/in-flight slots, policy lag, and quiesce time.
Per-cycle records append to `<metrics-stem>-cycles.jsonl`; the
requested `--metrics_path` is a small atomically replaced run summary. Resume
appends to the existing JSONL history, and failure paths publish a `failed`
summary with an error type. Metrics retain only checkpoint basenames and error
classes, never absolute checkpoint/resume paths or exception messages. Cycle
records are streamed rather than retained by the production controller, so
metrics memory remains bounded. The summary retains its most recent cycle
record after finalization, including failure finalization. Metrics summary and
JSONL paths are resolved and rejected before any series write if they overlap
the checkpoint base, latest/pending manifests, lock, or cycle namespace.
Resume validates the complete metrics summary schema, run ID, and JSONL name;
an orphan JSONL without its summary fails closed. These semantics are
CPU-tested; CUDA soak and
long-duration GPU stability are not validated here.

Checkpoint sequence ordering is numeric rather than lexical, including after
sequence 999999. Evaluation commands use platform-aware tokenization. A JSON
string array is the unambiguous cross-platform form, especially for Windows
paths containing backslashes or spaces:

```powershell
python train_v2.py --long_running ... `
  --eval_every_cycles 5 `
  --eval_command '["python", "evaluate_paired.py", "--candidate", "{checkpoint}"]'
```

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
forward/backward, weight publication, compact replay ingestion near capacity,
and legacy/factorized/V2 forward paths.
Unavailable CUDA AMP and DDP paths are recorded as `not_run`, never estimated.
The first compile is intentionally not mixed into steady-state forward timing.

On a CUDA host, run the bounded async topology smoke before benchmarking:

```bash
python -m pytest -q \
  tests/test_v2_throughput.py::test_async_single_gpu_end_to_end_checkpoint_resume_and_shutdown
```

The smoke starts two spawned actors, collects games through centralized CUDA
inference, performs optimizer steps, checks parameter updates and quiescence,
saves and resumes a topology-bound checkpoint, and verifies worker shutdown.
Passing CPU CI does not substitute for this CUDA execution or for the matched
`single_process` versus `async_single_gpu` A/B benchmark.
