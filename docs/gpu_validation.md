# P17 GPU, AMP, and NCCL Validation

## Result

**Empirical GPU validation: NOT RUN**

Reason: the audit host has no NVIDIA device or driver interface. PyTorch
reported `cuda_available=false`, `gpu_count=0`, `torch.version.cuda=null`, and
`nccl_available=false`. `nvidia-smi` is absent. Docker is available, but its
configured runtimes are `runc` and `io.containerd.runc.v2`; no NVIDIA runtime
is configured.

The sanitized probe was run on 2026-07-15 with Python 3.14.6, PyTorch 2.13.0,
Darwin arm64, and Docker Desktop. Hostname, username, paths, GPU UUIDs, serial
numbers, process lists, and environment variables are deliberately excluded.

No peak-memory, samples/s, decisions/s, learner-steps/s, compile-time, graph
break, or eager-versus-compile number is available. CPU tests and short CPU
rollouts are not substitutes for these measurements.

There are also independent implementation blockers: standard learned-bidding
training, joint/alternating belief synchronization, and distributed trainer
checkpoint resume intentionally fail closed under DDP. Therefore NCCL DDP was
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

Checkpoint resume is wired through `--checkpoint_path` and
`--resume_checkpoint`, but remains **NOT RUN** on this host. Target validation
must run standard full games, FP32/FP16/BF16 where supported, frozen and joint
belief modes, the controlled non-finite AMP fallback, and strict checkpoint
resume. The blocked standard/joint DDP graph and distributed checkpoint path
must be implemented before rank-local seeds and replay, rank-zero-only side
effects, or clean NCCL shutdown can be empirically accepted. The manual script
returns nonzero while those blockers remain.

`torch.compile` remains disabled by default. It must not be recommended until
first-compile cost, graph breaks, dynamic legal-action behavior, feature-path
coverage, and eager/compiled throughput are measured on the target hardware.
