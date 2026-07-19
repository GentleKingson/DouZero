# Checkpoint compatibility

> P01 adds a versioned `CheckpointManifest` to training checkpoints
> (`model.tar`) so that incompatible loads fail loudly instead of silently
> partial-loading.

## Formats

Two checkpoint formats coexist:

1. **Training bundle `model.tar`** — written by `douzero/dmc/dmc.py`'s
   `checkpoint(frames)`. As of P01 it contains the six legacy keys **plus** a
   new `manifest` key:

   | Key | Contents |
   |---|---|
   | `model_state_dict` | `{position: state_dict}` for the 3 roles |
   | `optimizer_state_dict` | `{position: optimizer_state_dict}` |
   | `stats` | training stats dict |
   | `flags` | `vars(flags)` (the effective run config) |
   | `frames` | int |
   | `position_frames` | `{position: int}` |
   | `manifest` | (P01) serialized `CheckpointManifest` |

2. **Per-position eval sidecars `{position}_weights_{frames}.ckpt`** — bare
   `state_dict` files consumed by `DeepAgent`. **Unchanged in P01.** Their
   permissive key filtering (`{k:v for k,v in pretrained.items() if k in
   model_state_dict}`) is the legacy behavior; it is **not** behind an explicit
   opt-in at the `DeepAgent` call site (the filter runs unconditionally in the
   existing code). P00 tests pin this behavior; P16 will replace it with a
   strict, manifest-backed, explicitly-opt-in load. Until then, the honest
   status is: the permissive filter is the unchanged legacy default, not a new
   opt-in gate.

## The manifest

`douzero/checkpoint/manifest.py` → `CheckpointManifest` (frozen dataclass):

| Field | Meaning |
|---|---|
| `schema_version` | manifest schema version (currently 1); bumped only on a breaking manifest-schema change |
| `model_version` | model version (`legacy` in P01; `factorized` in P04 as a deployment-only forward sharing the legacy state_dict; `v2` arrives in P05) |
| `feature_version` | observation feature version (`legacy` in P01; `v2` in P03) |
| `ruleset_id` | rule set (`legacy` in P01; `standard` in P02) |
| `checkpoint_kind` | one of `training_checkpoint` / `position_weights` / `privileged_teacher` / `public_policy` |
| `model_access` | `public` for ordinary/deployment models or `privileged` for a P10 training-only teacher; absent legacy fields default to `public` |
| `git_sha` | commit SHA, or `"unknown"` (always a string, never None) |
| `python_version` | interpreter version at save time |
| `torch_version` | torch version at save time (native `str`, not `TorchVersion`) |
| `effective_config` | the full flag dict for auditability |
| `frames`, `position_frames` | training progress counters |
| `created_at` | ISO-8601 UTC timestamp |

## Loading behavior

`douzero/checkpoint/io.py` → `load_checkpoint(path, ...)`:

- **manifest present + compatible** → returns `(bundle, manifest)`.
- **manifest present + incompatible** → raises `CheckpointCompatibilityError`
  with the offending field and expected/actual values. The core fields
  checked**: `schema_version`, `model_version`, `feature_version`,
  `ruleset_id`, `checkpoint_kind`, and `model_access`. There is **no** permissive fallback — a
  mismatch on any one always raises.
- **manifest absent (legacy pre-P01 checkpoint)** → delegates to
  `load_legacy_model_tar`, returns `(bundle, manifest=None)`. No version
  validation is possible because there is no manifest; callers assume the
  legacy feature/rule identity (acceptable in P01 because only `legacy`
  exists).

## P04 factorized model — shared state_dict

The P04 factorized models (`douzero/dmc/models_factorized.py`) declare the
**same submodule names and shapes** as the legacy models
(`lstm`, `dense1` … `dense6`), so a legacy per-position `.ckpt` loads into a
factorized model with **no conversion** via the existing
`load_legacy_position_ckpt` + key-filter path. `state_dict()` keys and shapes
are byte-for-byte identical; only `forward` differs (the factorized forward
encodes the shared history/state once per decision).

Consequences:

- A checkpoint produced by legacy training can be served with
  `DeepAgent(..., backend='legacy_factorized')` and produces the **same**
  selected action.
- No migration tool or weight conversion is required.
- The manifest `model_version` distinguishes a legacy-trained checkpoint
  (`legacy`) from a factorized-trained one (`factorized`). In P04, training
  is still legacy-only (the `dmc.py` gate rejects `model_version='factorized'`
  for training), so no `factorized`-stamped training checkpoint is produced
  yet. P06 widens this.

## P05 Model V2 — separate state_dict, strict load

The P05 Model V2 (`douzero/models_v2/`) is a **different architecture** from
the legacy / factorized models: it has a shared trunk, role embeddings, a
Transformer/LSTM history encoder, and multi-head outputs. Its `state_dict`
keys and shapes do **not** match the legacy models, so a legacy `.ckpt`
**cannot** be loaded into a V2 model and vice versa.

P05 adds V2-aware checkpoint helpers in `douzero/checkpoint/v2.py`. Both the
full bundle and the deployment sidecar are **manifest-bearing** and bound to
the same identity axes (model_version, feature_schema_hash,
model_config_hash, ruleset id/version/hash, checkpoint_kind, model_access):

- `save_v2_checkpoint(path, model, *, ruleset, ...)` — writes the full
  `model_v2.tar` bundle. The model identity (feature schema hash +
  `ModelV2Config` hash) is **derived from the model itself**, not self-reported
  by the caller — a caller-supplied `schema_hash` / `model_config` that
  disagrees with the model's own is rejected, so a bundle can never be labelled
  with an identity that does not match the actual weights. The full `RuleSet`
  is **required** and its complete identity (id + version + hash) is stamped
  onto the manifest, supporting custom rule families. The default call (no
  overrides) always produces a loadable file.
- `load_v2_checkpoint(path, *, expected_schema_hash, expected_model_config_hash,
  expected_ruleset, ...)` — reads a V2 bundle and validates all identity
  axes against RUNTIME-supplied expectations (never the manifest's self-reported
  values). Every expected value is a required argument. Raises
  `CheckpointCompatibilityError` on any mismatch, including an attempt to load a
  legacy/factorized `model.tar` here.
- `save_v2_position_weights(path, model, *, ruleset, schema_hash=None,
  model_config=None)` — writes the **manifest-bearing** deployment sidecar
  (`.ckpt`) for `DeepAgentV2`. It is NOT a bare state_dict: it carries the
  schema hash, model-config hash, and full ruleset identity so the strict
  identity check applies at deployment too. Like `save_v2_checkpoint`, the
  schema/config identity is derived from the model (overrides verified).

The strict V2 loader (`load_v2_model` in `deep_agent.py`) performs a full
key-set + shape match and raises `ValueError` on any mismatch — there is no
permissive partial load. The existing `load_checkpoint` (legacy/factorized
`model.tar`) already rejects a `model_version` mismatch via the manifest
validator, so a V2 bundle cannot be silently loaded as legacy.

Training is still legacy-only (the `dmc.py` gate rejects `model_version='v2'`
for training until P06); see `docs/model_v2.md` for the deployment path.

## P10 privileged teacher checkpoints

P10 uses a dedicated `privileged_teacher` bundle with
`model_access=privileged`. Only `load_teacher_checkpoint` accepts this pair;
the legacy loader and V2 public-policy loader reject it. Public student export
uses `public_policy` plus `model_access=public`, and the export function accepts
only `ModelV2`. Older manifests have no access field and are interpreted as
public, preserving their existing load behavior.

## Security: weights_only by default

Both `load_checkpoint` and `load_legacy_model_tar` default to
`weights_only=True` (PyTorch's safe unpickling mode, which restricts
deserialization to tensors, primitives, and standard containers). A P01
training bundle (RMSProp optimizer state dicts + plain-dict stats/flags/
manifest) loads cleanly under this safe mode.

A checkpoint that embeds objects safe mode cannot reconstruct (e.g. a pickled
`argparse.Namespace`, a `datetime`, or a custom stats object) requires the
caller to explicitly pass `allow_unsafe_pickle=True`, which switches to
`weights_only=False` and logs a warning. This keeps untrusted checkpoints safe
by default — arbitrary code execution via pickle is never the default path.

The removed `TRAINING_CHECKPOINT_TRUSTED` constant (always `True`) was a
pseudo-gate that provided no real protection; it has been deleted.

## Resumable V2 trainer topology

New resumable trainer checkpoints use format 6 for both `single_process` and
`async_single_gpu`. Format 6 records identity version 2 and the complete
`TrainerConfig`, including independent bidding batch size and update cadence.
Formats 3, 4, and 5 remain same-source-shape compatibility paths only: the
runtime still requires the checkpoint's full `source_git_sha` to equal the
running build. They are not a cross-commit migration promise. Missing M1
bidding controls are accepted only when `bidding_batch_size == batch_size` and
`bidding_update_interval == 1`; otherwise resume fails before model or optimizer
state is mutated.

Format 4 introduced the async actor/replay identity and format 5 added protocol,
task, and commit semantics. A v3 checkpoint is accepted only by
`single_process`; cross-topology and unknown-version resume fail closed. Async
resume is a safe empty cycle boundary, not a claim of bitwise N+M determinism.
Resume is rejected after Actor startup so process-local Actor RNG semantics
cannot diverge from already-running workers. Cross-commit reuse must export and
load a weights-only model artifact or use a future explicit offline migration
tool, never the resumable trainer loader.

## Round-trip and tests

`tests/test_checkpoint_manifest.py` pins:
- new-format `model.tar` round-trips under the safe default (`weights_only=True`);
- legacy `model.tar` (no manifest) still loads;
- incompatible `schema_version` / `model_version` / `feature_version` /
  `ruleset_id` / `checkpoint_kind` each raise a precise
  `CheckpointCompatibilityError`;
- `git_sha` is `"unknown"` (string) when git is unavailable;
- a checkpoint with a non-safe global is **refused** under the default and
  accepted only via `allow_unsafe_pickle=True` (or hard-refused under
  `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`);
- `manifest.torch_version` is a native `str` (not `TorchVersion`), so the
  manifest is `weights_only=True`-loadable.

The legacy `tests/test_checkpoint_loader.py` (P00) still passes, confirming
the eval-sidecar path is unchanged.

## Migration and rollback

- **Old `model.tar` → new code:** loads via the compat path (manifest=None).
  No migration needed.
- **New `model.tar` → old code:** legacy loaders that ignore unknown keys
  (e.g. a pre-P01 `torch.load`) simply see the extra `manifest` key and ignore
  it — backward compatible.
- **Rollback P01:** `git revert` the P01 commits; the manifest code is
  additive and the legacy tensor keys/meanings never changed.
