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
