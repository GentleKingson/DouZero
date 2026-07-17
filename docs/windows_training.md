# Windows Training

This page documents the training paths implemented by the current repository.
It distinguishes a CUDA **learner** from CUDA multiprocessing **actors**; they
have different Windows support boundaries. The continuous test matrix runs on
Ubuntu with CUDA hidden. Native Windows CUDA execution is not continuously
validated by this repository.

## 1. Support scope

| Entry point and topology | CPU | One CUDA GPU | Multiple GPUs | Native Windows boundary |
|---|---|---|---|---|
| Legacy `train.py`, CPU actors + CPU learner | Yes | N/A | N/A | Code-supported; not covered by a native Windows CI job |
| Legacy `train.py`, CPU actors + CUDA learner | N/A | Yes | Learner is one device | Code-supported; native Windows CUDA is not continuously validated |
| Legacy `train.py`, CUDA actors | N/A | Implemented with shared CUDA models/tensors | Implemented on Linux-oriented multiprocessing path | Not a supported or project-validated native Windows path |
| V2 `train_v2.py`, single process | Yes, default | Yes via `--device cuda` or `auto` | N/A | Code-supported; native Windows CUDA is not continuously validated |
| V2 base card-play DDP | Gloo | NCCL | Yes, one process per device | Use WSL2/Linux for CUDA DDP; native Windows CUDA DDP is not supported by this project path |
| P17 standard/full-game V2 | Yes | Yes, single process | No | DDP fails closed; native Windows CUDA is not continuously validated |

"Code-supported" means the current device and CLI paths permit the mode. It
does not mean that the mode has passed native Windows CI or long-duration
training. The mandatory CI matrix is Ubuntu CPU-only. The repository also has a
manual Linux/NVIDIA validation workflow, but a workflow definition is not proof
that a particular commit or GPU completed it.

## 2. Choosing native Windows, WSL2, or Linux

Native Windows is a reasonable starting point for the all-CPU legacy path, the
legacy CPU-actor/CUDA-learner split, and single-process V2 CPU or CUDA smokes.
Run a bounded smoke before a long job because there is no native Windows CUDA
CI lane.

Prefer WSL2 or Linux for legacy CUDA actors, CUDA DDP/NCCL, multi-GPU work,
long-running formal training, and an environment closer to project CI and the
manual GPU validation script. WSL2 still requires a compatible Windows driver,
a CUDA-enabled PyTorch build, adequate memory/shared-memory resources, and a
successful local smoke; it does not automatically remove those constraints.

## 3. Python and PyTorch installation

The package requires Python 3.11 or newer. In PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For CUDA, select the command for the installed driver and desired CUDA build
from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Install that PyTorch build first, then install DouZero from the checkout:

```powershell
python -m pip install -e .
```

Do not infer CUDA compatibility from a GPU model name alone. Newer GPU
architectures require a PyTorch build containing kernels for the applicable
compute capability.

## 4. CUDA availability verification

`torch.cuda.is_available()` is only the first check. Also allocate tensors and
execute a real CUDA operation:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('available', torch.cuda.is_available()); x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), float(y[0,0]))"
```

If that fails, fix the driver/PyTorch installation before debugging DouZero.

## 5. Legacy CPU actors + CPU learner

`--actor_device_cpu` makes the legacy actor device iterator use CPU.
`--training_device cpu` constructs the learner models and training batches on
CPU. The legacy multiprocessing context is explicitly `spawn`.

```powershell
python train.py `
  --actor_device_cpu `
  --training_device cpu
```

This is the portable legacy path. It still starts actor subprocesses and
learner threads; it is not the single-process V2 trainer.

## 6. Legacy CPU actors + GPU learner

The actor and learner choices are independent. With CPU actors, actor policy
models and replay buffers use shared CPU tensors, while `--training_device 0`
moves the learner and learner batches to logical CUDA device 0.

```powershell
python train.py `
  --actor_device_cpu `
  --gpu_devices 0 `
  --training_device 0
```

`train.py` assigns `--gpu_devices` to `CUDA_VISIBLE_DEVICES`; device indices
used by `--training_device` are indices in that visible set. The example makes
physical GPU 0 visible and selects logical GPU 0. This split path is allowed by
the code, but it has no continuous native Windows CUDA validation. Start with a
short, disposable experiment before committing to a long run.

## 7. Why native Windows legacy GPU actors are not recommended

Without `--actor_device_cpu`, legacy training constructs actor policy models on
CUDA and calls `share_memory()` on them. It also creates replay buffers as CUDA
tensors and calls `share_memory_()`. Those objects are then used by actor
processes created with the `spawn` multiprocessing context. This is materially
different from keeping actors on CPU and using CUDA only in the learner.

The project does not continuously validate that shared-CUDA actor topology on
native Windows, so it is outside the supported Windows boundary even though the
code has no platform-name check. Use CPU actors with a CPU or CUDA learner on
native Windows, or move legacy GPU actors to a locally validated WSL2/Linux
environment.

## 8. V2 CPU smoke training

`train_v2.py` is a single-process entry point and defaults to CPU. Its device is
selected by `--device`, not by legacy YAML fields such as `gpu_devices`,
`num_actors`, or `training_device`.

```powershell
python train_v2.py `
  --device cpu `
  --episodes 4 `
  --optimizer_steps 1 `
  --batch_size 1 `
  --seed 1
```

## 9. V2 single-GPU training

`--device cuda` requires CUDA and fails clearly if it is unavailable.
`--device auto` selects CUDA when `torch.cuda.is_available()` is true and CPU
otherwise. V2 does not use legacy CUDA actor subprocesses and does not clear
`CUDA_VISIBLE_DEVICES`.

```powershell
python train_v2.py `
  --config configs\enhanced.yaml `
  --device cuda `
  --episodes 8 `
  --optimizer_steps 2 `
  --checkpoint_path artifacts\windows\v2-checkpoint.pt `
  --metrics_path artifacts\windows\v2-metrics.json
```

## 10. V2 AMP

V2 accepts `--amp_enabled`, `--amp_dtype float16`, and
`--amp_dtype bfloat16`. CUDA AMP uses autocast and `GradScaler`; a non-finite
AMP step can retry once in float32 when fallback is enabled (the default).

```powershell
python train_v2.py `
  --config configs\enhanced.yaml `
  --device cuda `
  --episodes 8 `
  --optimizer_steps 2 `
  --amp_enabled `
  --amp_dtype float16
```

CPU AMP is opt-in and accepts only `bfloat16`; CPU `float16` is rejected. CUDA
FP16/BF16 behavior remains dependent on the installed GPU and PyTorch build.

## 11. P17 standard/full-game single-process training

`configs/standard_v2.yaml` selects the standard ruleset, enables the bidding
state machine and bidding head, and uses the learned bidding policy with a rule
warm start. Standard mode fails closed unless `bidding.enabled` and
`model.bidding_enabled` are both true. It is currently an eager,
single-process CPU or single-GPU path: DDP and `compile_model` are rejected.

```powershell
python train_v2.py `
  --config configs\standard_v2.yaml `
  --device cuda `
  --episodes 2 `
  --optimizer_steps 1 `
  --batch_size 1 `
  --checkpoint_path artifacts\windows\standard-v2.pt `
  --metrics_path artifacts\windows\standard-v2-metrics.json
```

For a CPU smoke, change `--device cuda` to `--device cpu`. Belief-enabled runs
also require a strictly compatible `--belief_checkpoint`; joint/alternating
belief modes are single-process and eager. Nonzero reserved bid-regret loss,
standard DDP, standard `torch.compile`, missing bidding identity, and invalid
belief combinations fail closed rather than falling back silently.

P17 provides training and readiness infrastructure, not a released model:

```text
Release candidate: NONE
Release status: NOT READY
```

## 12. Checkpoint save and resume

Single-process V2 saves a resumable checkpoint with `--checkpoint_path`,
restores one with `--resume_checkpoint`, and writes sanitized metrics with
`--metrics_path`.

```powershell
python train_v2.py `
  --config configs\enhanced.yaml `
  --device cuda `
  --resume_checkpoint artifacts\windows\v2-checkpoint.pt `
  --checkpoint_path artifacts\windows\v2-resumed.pt `
  --episodes 8 `
  --optimizer_steps 2
```

Resume is intentionally strict. The checkpoint binds the full source Git SHA,
single-process topology/world size, trainer configuration, feature schema,
ruleset, model configuration, loss configuration, bidding identity/policy, and
belief identity when applicable. Use the same source and training identity; do
not bypass a mismatch. A source archive or wheel without Git metadata must set
the exact build commit, not an arbitrary replacement:

```powershell
$env:DOUZERO_GIT_SHA = "<full-git-sha>"
```

Checkpoint save/resume is explicitly rejected under DDP.

## 13. DDP and multi-GPU limits

Base, legacy-ruleset V2 DDP must be launched with `torchrun`; calling
`train_v2.py --ddp_enabled` directly leaves `WORLD_SIZE=1` and is rejected.
The runtime selects NCCL when CUDA is available and Gloo otherwise. NCCL gives
one CUDA device per process; Gloo in this project path uses CPU.

```powershell
torchrun --standalone --nproc-per-node=2 train_v2.py `
  --config configs\enhanced.yaml `
  --ddp_enabled `
  --ddp_backend gloo `
  --device cpu `
  --episodes 8 `
  --optimizer_steps 2
```

This PowerShell command is a CPU/Gloo topology example, not a native Windows
CUDA recommendation. PyTorch documents Windows distributed support as
prototype and states that Windows does not provide the NCCL backend; it
recommends NCCL for CUDA and Gloo for CPU. See the official
[`torch.distributed` backend documentation](https://docs.pytorch.org/docs/stable/distributed.html)
and [`torchrun` documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).
For this project's CUDA DDP path, use WSL2 or Linux and validate the exact host.

The following DDP combinations fail closed: P17 standard/learned bidding,
joint or alternating belief training, checkpoint save/resume,
curriculum/coach-label output, and RL+BC auxiliary training. Legacy `train.py`
also explicitly rejects `--ddp_enabled`.

## 14. Common errors and troubleshooting

- `--device cuda requested but CUDA is unavailable`: install a CUDA-enabled
  PyTorch build and rerun the real CUDA operation in section 4.
- `DDP requires WORLD_SIZE >= 2; launch with torchrun`: use `torchrun` and at
  least two processes, or disable DDP.
- NCCL/backend initialization failure on native Windows: do not treat CPU/Gloo
  availability as CUDA DDP support; use a validated WSL2/Linux environment for
  the project CUDA DDP path.
- Legacy actor `operation not supported` or shared-memory failures: confirm
  `--actor_device_cpu` is present. CUDA actors and a CUDA learner are separate
  choices.
- V2 warns that `gpu_devices`, `num_actors`, or `training_device` is ignored:
  select V2 placement with `--device`; those are legacy multiprocessing fields.
- Checkpoint identity mismatch: restore with the exact source SHA, topology,
  ruleset, model/loss, bidding/belief identity, and trainer configuration that
  created it. Do not edit the checkpoint or spoof the SHA.
- A recent GPU fails during a real CUDA operation: install a PyTorch CUDA build
  that contains kernels for that GPU's compute capability; availability alone
  is insufficient.

## 15. Verified and unverified scope

- **Continuously verified:** Ubuntu CPU CI across Python 3.11, 3.12, and 3.13,
  including CLI help and the project test suite.
- **Covered by focused CPU tests:** V2 optimizer steps, CPU BF16 AMP, CPU/Gloo
  DDP behavior, standard learned-bidding training, and strict checkpoint
  identity/resume contracts.
- **Available as a manual gate, not continuous proof:** Linux/NVIDIA CUDA AMP,
  checkpoint, and NCCL validation workflow.
- **Not verified here:** native Windows CPU/GPU execution, native Windows CUDA
  learner stability, WSL2 CUDA, and long-duration target-hardware training.
- **Outside the documented native Windows support boundary:** legacy CUDA
  actors. The code has no Windows platform rejection, but the shared-CUDA
  topology is not project-validated there.
- **Explicitly unsupported by the current runtime:** native Windows CUDA
  DDP/NCCL, P17 standard DDP, joint/alternating belief DDP, and DDP checkpoint
  save/resume.
