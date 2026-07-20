# Legacy V1 single-GPU benchmark

The recommended optimized topology is **A1**: factorized CPU Actors with one
CUDA learner. The benchmark runner defaults to A0 and A1 only. Direct-GPU
Actor configurations B0/B1 and centralized inference C0 are experimental and
must be selected explicitly with `--config`. B1 may emit CUDA IPC lifecycle
warnings and is not a production recommendation.

Formal evidence must come from a clean checkout and include an immutable
Docker image ID or digest:

```bash
python benchmarks/bench_legacy_training.py \
  --formal \
  --docker_image_digest sha256:<image-id> \
  --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
  --warmup_frames 64000 \
  --measure_frames 128000 \
  --repeats 3
```

The output records the Git SHA, clean-worktree status, configuration hashes,
per-run metric hashes, image digest, and an artifact checksum manifest. Keep
complete raw runs as CI artifacts or release attachments rather than adding
them to the main repository.

Timeout cleanup reaps the complete training process tree on POSIX by using a
dedicated process group. Native Windows only terminates the direct child, so
formal CUDA benchmarks must run on Linux or WSL2.
