# Deployment and Release Audit

P17 package format 2 extends the P16 fail-closed directory rather than
publishing a loose pickle. A package contains `weights.pt`, `manifest.json`,
`ruleset.json`, `feature_schema.json`, `model_config.json`, a hash-only
`training_config.json`, an explicit `model_card.md`, evaluation and GPU summaries,
`rollback.md`, `THIRD_PARTY_NOTICES`, and `SHA256SUMS`. Belief-enabled packages
also require a manifest-bearing `belief_weights.pt` plus an identity-bound
`belief_config.json`; both are covered by `SHA256SUMS`. Learned-bidding packages
require `bidding_schema.json` bound to the head version, action schema, feature
schema hash, and exact `0/1/2/3` action list.

The hash-only `training_config.json` also carries a machine-checked runtime
support matrix. It marks standard learned-bidding DDP, joint/alternating belief
DDP, and distributed trainer resume as unsupported; packaging cannot turn an
unvalidated training topology into a capability claim. Earlier format-2
packages with schema-1 hash-only training identity remain readable, but the
missing support matrix is interpreted as declaring no optional distributed
capability, never as evidence that one was validated.

Format-1 P16 directories are intentionally rejected by the format-2 verifier.
Continue using one only with its matching P16 runtime, or rebuild it from the
original manifest-bearing public checkpoint and reviewed metadata. Do not add
files by hand or edit checksums to imitate migration.

## Package a Public Model

```bash
python tools/package_model.py \
  --checkpoint /path/to/public_policy.ckpt \
  --output artifacts/release/model-v2 \
  --ruleset standard \
  --model-config /path/to/model_config.json \
  --training-config /path/to/training_config.json \
  --belief-checkpoint /path/to/belief.pt \
  --model-card /path/to/reviewed_model_card.md \
  --evaluation-summary /path/to/evaluation_summary.md \
  --gpu-validation-summary /path/to/gpu_validation_summary.md \
  --rollback /path/to/rollback.md
```

Omit `--belief-checkpoint` only when `belief_enabled=false`. The packaging
command loads the belief checkpoint through the strict weights-only loader and
verifies its ruleset, feature version, and actual `BeliefConfig` compatibility
identity before copying it. An arbitrary JSON mapping cannot stand in for a
belief checkpoint or architecture identity.

The output directory must be absent or empty. Packaging derives schema and
model-config hashes from the constructed model, saves through the strict V2
sidecar writer, and checksums every payload. Raw training configuration is not
copied because it commonly contains private filesystem or dataset paths. Its
canonical hash is inherited from the source checkpoint manifest;
`--training-config` is only a cross-check and cannot replace Git, ruleset,
schema, model, or training identity. Direct in-memory exports without a trusted
source checkpoint are marked `source_checkpoint_kind=migration_artifact` and
`release_eligible=false`. A privileged model cannot be labelled as
public. Package verification rejects unexpected files, so canonical or raw
human data cannot be smuggled into a release directory outside `SHA256SUMS`.

Load with runtime-owned expectations, never values copied from the package:

```python
from douzero.deployment import load_model_package

model = load_model_package(
    "artifacts/release/model-v2",
    schema=runtime_schema,
    ruleset=runtime_ruleset,
    config=runtime_model_config,
    device="cpu",
)
```

For a belief-enabled package, the returned `ModelV2` has the verified,
eval-mode public `BeliefModel` attached as `model.belief_model`.
`DeepAgentV2(..., model, runtime_ruleset)` consumes that attachment by default;
callers no longer need an untracked external belief checkpoint. The return type
and construction path for belief-disabled packages are unchanged.

The verifier strict-constructs `ModelV2` and strict-loads every finite tensor,
so successful verification establishes CPU loadability under the same runtime
identity. The loader rejects checksum changes, unknown manifest fields, wrong feature or
ruleset identity, wrong model or belief config, a training-config identity
mismatch, missing or empty release documents, unexpected files, missing
package versions, unsafe access class, bare legacy weights, and incomplete state
dictionaries. It also requires
a known source Git SHA and exact `model_abi_version` plus
`implementation_hash` match against the installed deployment implementation.
The inner `weights.pt` and `belief_weights.pt` sidecar manifests are
independently checked against the outer package identity. Checkpoint
deserialization uses PyTorch's `weights_only=True` path.

## Training Compatibility Boundary

Packaging and single-process inference support standard learned-bidding and
belief-enabled models when every required identity-bound artifact is present.
That must not be confused with distributed training support. Standard
learned-bidding training currently rejects DDP and `compile_model`; joint and
alternating belief training also reject DDP and compilation. Compatible
legacy-ruleset V2 card-play training retains P14 DDP, and frozen belief remains
the only belief mode eligible for that base distributed path. These requests
fail closed rather than silently dropping an auction head or belief gradients.

## Agent Runtime

`DeepAgentV2` accepts an explicit device and runs inference under
`torch.inference_mode()`. Evaluation mode makes the base policy deterministic;
search uses its configured seed. `timeout_ms` and runtime exceptions use a
configured base agent when possible, then a deterministic conservative legal
move. Identity and privileged-input failures happen before fallback and remain
hard failures. Search-enabled agents also reject a package unless its manifest
explicitly declares `search_compatible=true`.

Decision explanations are disabled by default. When requested they contain
only action indices and model scores, not hand/card plaintext or personal
identifiers. Search audit logs remain separately controlled by the caller.

## Export

`export_padded_model` uses `torch.export` with a fixed maximum action dimension
and a boolean valid-action mask. It reloads the artifact, compares all value
outputs with PyTorch, and always writes an adjacent `.report.json`. Belief,
strategy, and style feature paths currently report unsupported rather than
silently exporting a reduced model. Export failure does not invalidate the
state-dict deployment path. Export is written to a same-directory temporary
file and atomically published only after reload and numerical alignment; failed
attempts remove temporary and stale target files.

## Release Gate

Run `.docker/run_release_gate.sh`. It covers compile/CLI checks, the full test
suite, a deterministic legacy snapshot smoke, and `git diff --check`. Docker
can run the same gate in a clean CPU environment:

```bash
docker build -f .docker/Dockerfile.test -t douzero-p17-test .
docker run --rm douzero-p17-test bash .docker/run_release_gate.sh
```

GPU latency, production traffic, and unpublished model quality are not measured
by this gate and must stay marked as unmeasured in the model card.

GPU validation is manual and opt-in through `.github/workflows/gpu-validation.yml`
on an explicitly labelled self-hosted NVIDIA runner. Default CI remains
CPU-only and offline after dependency installation. The manual job failing or
remaining unrun is a release blocker, not evidence that the GPU path passed.
