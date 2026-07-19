# P17 GPU, AMP, and NCCL Validation

## Result

**Standard V2 M1 manual GPU validation: PASSED**

Source `408c97a25088451d884d9836ce6f6fbf32d810c6` was built and tested on
`LocalServer:/opt/DouZero` in `douzero-test:latest`, immutable image ID
`sha256:417ae0638a9ffa1cc1ba41c2a23e9a20074866b79a7b7a08e6efe1b4221ac2cb`.
The sanitized environment records Python 3.12.3, PyTorch 2.12.1+cu132, CUDA
13.2, driver 595.71.05, and one RTX 5070. No GitHub GPU workflow was used.

The auditable bundle is
`benchmarks/evidence/standard_v2_m1_408c97a/`. It includes the Docker build
log and command inventory, full CPU pytest log, explicit CUDA pytest log,
environment JSON, B=1/32/64/128 bidding benchmark, raw R1 training metrics,
strictly generated unified benchmark, and `SHA256SUMS`. Hostname, username,
GPU UUIDs, serial numbers, process lists, and environment variables are
deliberately excluded.

B=32 bidding head forward/backward passed the 1.5 ms gate at 1.055507 ms mean
and 1.198880 ms p95. The additional 16-game R1 repeat observed a parameter
update at 8.009856 games/s and 266.845 MiB peak allocated VRAM. The checked-in
canonical baseline remains 7.896824 games/s; both artifact digests are recorded
in the evidence manifest rather than presenting timing variance as a rewrite.

There are also independent implementation blockers: standard learned-bidding
DDP, joint/alternating belief DDP synchronization, and distributed trainer
checkpoint resume intentionally fail closed. Therefore NCCL DDP was
not run and must not be described as hardware-ready even on a two-GPU host.

## Reproducible Entry Point

Run the probe without training:

```bash
scripts/validate_gpu_training.sh --probe-only
```

Run target-hardware smokes on a CUDA host:

```bash
scripts/validate_gpu_training.sh
```

The script writes ignored local artifacts under `artifacts/gpu-validation/`:

```text
environment.json
single_gpu_fp32.json
single_gpu_fp16.json
single_gpu_bf16.json
amp_nonfinite_fallback.json
ddp_2gpu.json
belief_frozen.json
belief_joint.json
checkpoint_resume.json
summary.md
```

It uses `configs/standard_v2.yaml`, exercises learned-bidding standard training,
CUDA FP32 and AMP, saves and strictly resumes a trainer checkpoint, and records
measured peak allocated/reserved memory, card-play transitions/s, bidding
decisions/s, total decisions/s, and learner steps/s. The default smoke batch is
4 (not the YAML's release-training batch of 32), so eight episodes can populate
both replay buffers. The guarded AMP case deliberately injects one non-finite
loss and requires the float32 retry to finish with a finite parameter update.
Training metrics separately audit `redeals` and `max_redeals_exceeded`; a
redeal-cap forced fallback cannot be hidden inside ordinary bidding counts.

Standard DDP is recorded as `blocked_implementation`; the script does not
launch `torchrun` for a configuration the trainer rejects by design. Unsupported
BF16 is recorded as `unsupported_hardware`. A short passing command proves
execution compatibility only, not playing strength or sustained throughput.

Frozen and joint belief validation require a public belief-enabled standard V2
config and a compatible standard-ruleset belief checkpoint:

```bash
python train_belief.py --ruleset standard \
    --save_dir /tmp/belief_standard --num_episodes 20 \
    --epochs 3 --batch_size 32 --seed 0

DOUZERO_GPU_BELIEF_CONFIG=/path/to/belief-standard-v2.yaml \
DOUZERO_GPU_BELIEF_CHECKPOINT=/tmp/belief_standard/belief.pt \
scripts/validate_gpu_training.sh
```

Without both inputs, `belief_frozen.json` and `belief_joint.json` explicitly say
`not_run`; no synthetic or incompatible checkpoint is substituted.

## Remaining Gate

Single-GPU CUDA standard checkpoint resume, the CUDA input-error contract, and
format 7 async checkpoint compatibility passed in the explicit Docker test
set. Broader P17 validation still requires FP16/BF16, controlled non-finite AMP
fallback, and compatible frozen/joint belief artifacts. The blocked
standard/joint DDP graph and distributed checkpoint path must be implemented
before rank-local seeds and replay, rank-zero-only side effects, or clean NCCL
shutdown can be empirically accepted. The generic manual script continues to
return nonzero while those independent blockers remain.

`torch.compile` remains disabled by default. It must not be recommended until
first-compile cost, graph breaks, dynamic legal-action behavior, feature-path
coverage, and eager/compiled throughput are measured on the target hardware.
