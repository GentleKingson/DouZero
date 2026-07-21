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

## A1 profiler baseline

With `legacy_profile: true`, A1 emits stable PyTorch profiler/NVTX ranges for
`actor.inference`, `learner.batch_wait`, `learner.batch_assembly`,
`learner.pin_memory`, `learner.h2d`, and `learner.optimization_step`. The ranges
are absent when profiling is disabled, so the production configuration does
not pay profiler instrumentation overhead. These ranges complement aggregate
JSON timings and expose CPU packing, transfer, and learner work separately in
PyTorch Profiler or Nsight Systems.

`legacy_matmul_precision` isolates float32 matmul precision from AMP. The
production default is `highest`; `legacy_a5_cpu_factorized_tf32.yaml` measures
`high`, while `legacy_a6_cpu_factorized_tf32_compile.yaml` combines it with the
existing fixed-shape learner compile path. Neither changes actor precision or
the Legacy checkpoint parameter contract.

## Legacy A1 model-optimization decision record

This split branch starts from `origin/main` and contains neither the C0
centralized-inference experiment nor gpu_v3. A1 remains the production
topology; its new optimization controls remain explicit and opt-in.

On the RTX 5070 test host, the 200-round CPU attribution benchmark measured
adaptive split-dense1 speedups over the original Legacy forward of `1.103x`
at 10 actions, `1.412x` at 50 actions, and `1.695x` at 287 actions. One-action
latency regressed to `0.968x`, which is why the experiment retains the
single-action bypass and only applies the split above its action-count
threshold.

Two complete formal 8-actor/48-buffer A1 runs measured `12,632.96` and
`12,729.28` frames/s, with `13,023.60` and `13,026.43` decisions/s. Both
reported CUDA available, zero live learner threads at exit, and completed
checkpoint output. The third repeat was killed externally at startup on this
host, so this is diagnostic evidence rather than a promotable three-repeat
result.

Accordingly, adaptive A1 split-dense1, TF32, and learner compile remain
opt-in. Reusable pinned/device staging and per-snapshot role lookup caching are
retained because their correctness tests pass and they do not alter model or
checkpoint identity. Production A1 keeps `legacy_matmul_precision: highest`,
AMP disabled, and compile disabled until a complete three-repeat run clears
both throughput and numerical gates.
