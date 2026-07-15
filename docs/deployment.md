# Deployment and Release Audit

P16 publishes Model V2 as a fail-closed directory rather than a loose pickle.
A package contains `weights.pt`, `manifest.json`, `ruleset.json`,
`feature_schema.json`, a model card, `THIRD_PARTY_NOTICES`, and `SHA256SUMS`.

## Package a Public Model

```bash
python tools/package_model.py \
  --checkpoint /path/to/public_policy.ckpt \
  --output artifacts/release/model-v2 \
  --ruleset standard \
  --model-config /path/to/model_config.json \
  --training-config /path/to/training_config.json \
  --model-card /path/to/reviewed_model_card.md
```

The output directory must be absent or empty. Packaging derives schema and
model-config hashes from the constructed model, saves through the strict V2
sidecar writer, and checksums every payload. A privileged model cannot be
labelled as public.

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

The loader rejects checksum changes, unknown manifest fields, wrong feature or
ruleset identity, wrong model config, missing package versions, unsafe access
class, bare legacy weights, and incomplete state dictionaries. Checkpoint
deserialization uses PyTorch's `weights_only=True` path.

## Agent Runtime

`DeepAgentV2` accepts an explicit device and runs inference under
`torch.inference_mode()`. Evaluation mode makes the base policy deterministic;
search uses its configured seed. `timeout_ms` and runtime exceptions use a
configured base agent when possible, then a deterministic conservative legal
move. Identity and privileged-input failures happen before fallback and remain
hard failures.

Decision explanations are disabled by default. When requested they contain
only action indices and model scores, not hand/card plaintext or personal
identifiers. Search audit logs remain separately controlled by the caller.

## Export

`export_padded_model` uses `torch.export` with a fixed maximum action dimension
and a boolean valid-action mask. It reloads the artifact, compares all value
outputs with PyTorch, and always writes an adjacent `.report.json`. Belief,
strategy, and style feature paths currently report unsupported rather than
silently exporting a reduced model. Export failure does not invalidate the
state-dict deployment path.

## Release Gate

Run `.docker/run_release_gate.sh`. It covers compile/CLI checks, the full test
suite, a deterministic legacy snapshot smoke, and `git diff --check`. Docker
can run the same gate in a clean CPU environment:

```bash
docker build -f .docker/Dockerfile.test -t douzero-p16 .
docker run --rm douzero-p16 bash .docker/run_release_gate.sh
```

GPU latency, production traffic, and unpublished model quality are not measured
by this gate and must stay marked as unmeasured in the model card.
