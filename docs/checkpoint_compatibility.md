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
  with the offending field and expected/actual values. **Five fields are
  checked**: `schema_version`, `model_version`, `feature_version`,
  `ruleset_id`, and `checkpoint_kind`. There is **no** permissive fallback — a
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
  yet. P05/P06 widen this.

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
