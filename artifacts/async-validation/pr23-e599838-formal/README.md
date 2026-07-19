# PR #23 commit-bound recovery-soak evidence

This artifact records the formal CUDA validation of the committed PR source.

## Identity

- PR head: `e59983869ad88370af661d5238f6e50d4841b39e`
- campaign base reference: `fb9caf19c9903aed83eabed586e533197ab8d391`
- Docker image ID/digest: `sha256:8c994b0b0ab8998f433d71ec92c25640d5beb16f5e45ba2fc496272e9c76b282`
- test host project: `/opt/DouZero`
- GPU: NVIDIA GeForce RTX 5070, driver 595.71.05, CUDA 13.2

The image was built from a clean `git archive` of the full PR head. The
evidence-specific Docker context includes all tracked files so that the image's
source checkout has an empty `git status --porcelain`. The training containers
did not mount source code. They mounted only `/opt/DouZero/.git` read-only at
`/workspace/DouZero/.git` and their topology-specific evidence directory at
`/evidence`.

## Image build

```text
docker build --progress=plain \
  --file /tmp/douzero-pr23-e599838-image-context/Dockerfile.evidence \
  --build-arg DOUZERO_GIT_SHA=e59983869ad88370af661d5238f6e50d4841b39e \
  -t douzero-test:latest \
  /tmp/douzero-pr23-e599838-image-context
```

The build context was prepared from `git archive` rather than the mutable host
working tree. `Dockerfile.evidence.dockerignore` differs from the repository
default only by retaining the tracked baseline placeholder and logo so the
container source identity remains clean.

## Test commands

Full CPU suite:

```text
docker run --rm \
  --mount type=bind,src=/opt/DouZero/.git,dst=/workspace/DouZero/.git,readonly \
  douzero-test:latest
```

Result: PASS, with the CUDA-only end-to-end test skipped as expected and two
unrelated warnings. See `cpu-tests.log`.

CUDA checkpoint/resume and shutdown smoke:

```text
docker run --rm --gpus all \
  --mount type=bind,src=/opt/DouZero/.git,dst=/workspace/DouZero/.git,readonly \
  douzero-test:latest \
  python -m pytest -q \
  tests/test_v2_throughput.py::test_async_single_gpu_end_to_end_checkpoint_resume_and_shutdown
```

Result: PASS. See `cuda-smoke.log`.

## Formal campaigns

Both topologies ran the required sequence:

```text
30 minutes -> real SIGTERM -> fresh container -> latest-manifest resume -> 30 minutes
```

The exact create/start/signal/wait commands are in each topology's
`phase1-command.txt` and `phase2-command.txt`. Every phase exited with code 0.
The combined analysis is in `summary.md` and `analysis.json`; raw cycle JSONL,
five-second resource CSV, logs, manifests, timestamps, container inspections,
source identities, and cleanup records are retained alongside it.

## Checkpoint retention

Checkpoint binaries were deleted after trusted in-container analysis to keep
the published artifact small. `checkpoint-binaries.sha256` retains hashes for
all ten removed files. The signal and first-resumed checkpoint file hashes,
model-state hashes, counters, losses, and gradient norms are also recorded in
`analysis.json`; manifests and the dedicated `.sha256` files remain present.

## Cleanup

Only task-created containers and temporary checkpoint binaries were removed.
The standard image `douzero-test:latest` and Docker build cache were retained.
Both topology directories record an empty task-container listing and clean host
checkout before/after status. No source or repository file was changed by the
container runs.
