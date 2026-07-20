# Legacy V1 single-GPU training and benchmark

The recommended optimized topology is **A1**: factorized CPU Actors with one
CUDA learner. Use `configs/legacy_single_gpu_a1.yaml` for long-running
training. Files under `benchmarks/configs/` are evidence workloads: they
intentionally disable checkpointing and enable profiling, so they are not safe
as unattended training configurations.

```bash
python train.py \
  --config configs/legacy_single_gpu_a1.yaml \
  --xpid douzero-a1 \
  --total_frames 100000000000
```

This production configuration saves to `douzero_checkpoints/douzero-a1` every
30 minutes. Add `--load_model` to resume that experiment. Its checkpoint state
is transactionally consistent and the main `model.tar` is written through a
same-directory temporary file followed by atomic replacement. Per-role
evaluation sidecars use the same atomic write protocol and retain the newest
two snapshots by default. Set `--checkpoint_sidecar_retention 0` to disable
them or `-1` to preserve every snapshot.

Legacy progress advances by one complete learner update at a time. Therefore
`total_frames` must be divisible by `unroll_length * batch_size`; the supplied
A1 values require a multiple of `100 * 32 = 3,200` frames.

The benchmark runner defaults to the thread-limited A0 baseline and A1.
The unrestricted historical A0, direct-GPU Actor configurations B0/B1, and
centralized inference C0 must be selected explicitly with `--config`. B1 may
emit CUDA IPC lifecycle warnings and is not a production recommendation.

## Docker runtime requirements

The tested A1 layout needs more shared memory than Docker's 64 MiB default:

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ulimit nofile=65536:65536 \
  --mount type=bind,src="$(pwd)",dst=/workspace/DouZero \
  --workdir /workspace/DouZero \
  douzero-test:latest \
  python train.py \
    --config configs/legacy_single_gpu_a1.yaml \
    --xpid douzero-a1 \
    --total_frames 100000000000
```

Eight GiB is the validated value for the supplied 12-actor/64-buffer A1
configuration, not a universal minimum. Required `/dev/shm` capacity changes
with `num_actors`, `num_buffers`, `batch_size`, and `unroll_length`.

## Formal evidence

Formal evidence must come from a clean checkout and record an immutable Docker
image ID or digest plus the exact expected Git head:

```bash
IMAGE_ID="$(docker image inspect --format '{{.Id}}' douzero-test:latest)"
HEAD_SHA="$(git rev-parse HEAD)"

docker run --rm --gpus all \
  --shm-size=8g \
  --ulimit nofile=65536:65536 \
  --mount type=bind,src="$(pwd)",dst=/workspace/DouZero \
  --mount type=bind,src="$(pwd)/.git",dst=/workspace/DouZero/.git,readonly \
  --workdir /workspace/DouZero \
  douzero-test:latest \
  python benchmarks/bench_legacy_training.py \
    --formal \
    --docker_image_digest "$IMAGE_ID" \
    --expected_git_sha "$HEAD_SHA" \
    --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
    --warmup_frames 64000 \
    --measure_frames 128000 \
    --repeats 3
```

Formal mode fails closed if the Git SHA or worktree status cannot be read, the
actual SHA differs from `--expected_git_sha`, the checkout is dirty, or the
image digest is not `sha256:` followed by 64 lowercase hexadecimal characters.
The output records the Git SHA, clean-worktree status, configuration hashes,
per-run metric hashes, image digest, and an artifact checksum manifest. Keep
complete raw runs as CI artifacts or release attachments rather than adding
them to the main repository.

The runner records the caller-supplied image identity but cannot introspect the
outer container runtime to prove that it launched that image. The documented
`docker image inspect` step is therefore part of the evidence procedure, not a
cryptographic runtime attestation.

Timeout cleanup reaps the complete training process tree on POSIX by using a
dedicated process group. Native Windows only terminates the direct child, so
formal CUDA benchmarks must run on Linux or WSL2.

## C0 interleaved centralized inference

`legacy_c0_centralized_gpu_actor.yaml` is a production-candidate shape, but C0
remains experimental until formal evidence beats A1. Each CPU actor owns
`central_actor_envs_per_actor` independent games (four by default), with
separate environments, trajectories, request generations, and episode policy
leases. It submits every runnable game before consuming correlated responses.
A completed game resets immediately and takes a new policy lease; a policy
version never changes during a game.

The bounded queue applies backpressure at
`central_actor_max_pending_requests`. Inference groups compatible requests by
policy slot/version, role, and a power-of-two legal-action bucket. Different
action counts stay packed. It waits for `central_actor_min_microbatch`, aims
for `central_actor_target_microbatch`, and never exceeds
`central_actor_max_microbatch` or `central_actor_max_delay_ms`. Pinned staging
is allocated once at maximum capacity, so requests are batched, not truncated.

Inference uses a separate CUDA stream with high priority when supported.
Policy copies use a copy stream and CUDA event, and become visible atomically
only after inference observes the complete copy. With
`central_actor_learner_throttle`, learner threads postpone new updates while
queue depth exceeds `central_actor_queue_high_watermark` or the oldest request
exceeds `central_actor_inference_deadline_ms`. Unsupported priority safely
falls back to a normal stream and is reported in metrics.

```yaml
central_actor_envs_per_actor: 4
central_actor_max_actions: 4096
central_actor_min_microbatch: 2
central_actor_target_microbatch: 8
central_actor_max_microbatch: 16
central_actor_max_delay_ms: 2.0
central_actor_max_pending_requests: 128
central_actor_queue_high_watermark: 32
central_actor_inference_deadline_ms: 10.0
central_actor_learner_throttle: true
central_actor_use_stream_priority: true
central_actor_async_policy_copy: true
```

These controls are inert for A1 and the original Legacy actor. Old YAML files
receive schema defaults. Checkpoints retain the Legacy three-role state-dict,
optimizer, AMP, policy-publication, and progress contract. Resume deliberately
discards half-finished games and pending inference requests and starts new
games. C0 needs one CUDA device and shared memory for rollout buffers plus
`actors * envs_per_actor` request slots; use at least `--shm-size=8g` for the
supplied benchmark configuration.

Compare A1, PR #26's synchronous C0 shape, and interleaved C0 on the same clean
commit, immutable image, hardware, warmup, and measured frames:

```bash
python benchmarks/bench_legacy_training.py --formal --repeats 3 \
  --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
  --config benchmarks/configs/legacy_c0_sync_baseline.yaml \
  --config benchmarks/configs/legacy_c0_centralized_gpu_actor.yaml \
  --docker_image_digest "$DOUZERO_IMAGE_DIGEST" \
  --expected_git_sha "$DOUZERO_EXPECTED_GIT_SHA"
```

The report includes exact frames, microbatch and queue-wait distributions,
actor blocking, learner waiting/throttling, utilization, VRAM, policy lag, and
worker exit status. Do not promote C0 unless three-repeat median frames/s shows
a stable gain over both old C0 and A1. GPU utilization alone is not evidence.

### Optimized C0 runtime

The current experimental configuration uses `central_actor_runtime: thread`.
Actors remain CPU-only processes, while inference, learner work, policy copies,
and checkpoint coordination share the training process's single CUDA context.
Set the runtime to `process` only to reproduce the older dual-CUDA-process C0
baseline.

The inference backlog snapshot accounts separately for ingress, server-local,
and executing requests. Fixed-threshold and predicted-drain learner admission
are available through `central_actor_learner_throttle_mode`; waits use the
shared condition and record count, duration, and distribution. Latency metrics
separate queue-put blocking, IPC, server batching, CPU packing, H2D, GPU cast,
forward, response, response consumption, and end-to-end actor latency.

Threaded C0 publishes learner parameters directly into immutable same-device
GPU inference slots. A CUDA event gates visibility before the corresponding
metadata policy slot becomes active. Temporary GPU snapshots are reconstructed
from the learner after resume and are not part of the checkpoint contract.

The following inference experiments retain checkpoint-compatible parameter
names and shapes:

```yaml
central_actor_split_dense1: true
central_actor_staging_dtype: int8
central_actor_inference_layout: packed
```

Split dense1 computes the history/state projection once per decision and the
action projection per legal action using views of the existing `dense1`
weight. Int8 staging transfers observations in their source dtype, casts into
reused FP32 GPU buffers on the inference stream, and reports cast time.
Bucketed padded inference is an opt-in attribution path that replaces segmented
Python argmax with a tensor mask and reports padding, effective-FLOPs, and
compatible-group fragmentation. The supplied candidate stays packed because
the padded smoke run fragmented compatible requests and did not clear the
end-to-end retention gate. Float32 staging and unsplit paths also remain
selectable for attribution.

Padded bucket specialization can reduce compatible microbatch size. Treat a
lower forward latency as insufficient when fragmentation or padding reduces
end-to-end frames/s. Compile and CUDA Graph experiments should only be enabled
after a stable eager padded configuration covers most traffic; neither is a
production dependency.
