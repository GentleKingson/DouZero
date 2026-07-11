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
   model_state_dict}`) is pinned by P00 tests; P16 replaces it with a strict,
   manifest-backed load.

## The manifest

`douzero/checkpoint/manifest.py` → `CheckpointManifest` (frozen dataclass):

| Field | Meaning |
|---|---|
| `schema_version` | manifest schema version (currently 1); bumped only on a breaking manifest-schema change |
| `model_version` | model version (`legacy` in P01; `v2` arrives in P05) |
| `feature_version` | observation feature version (`legacy` in P01; `v2` in P03) |
| `ruleset_id` | rule set (`legacy` in P01; `standard` in P02) |
| `git_sha` | commit SHA, or `"unknown"` (always a string, never None) |
| `python_version` | interpreter version at save time |
| `torch_version` | torch version at save time |
| `effective_config` | the full flag dict for auditability |
| `frames`, `position_frames` | training progress counters |
| `created_at` | ISO-8601 UTC timestamp |

## Loading behavior

`douzero/checkpoint/io.py` → `load_checkpoint(path, ...)`:

- **manifest present + compatible** → returns `(bundle, manifest)`.
- **manifest present + incompatible** → raises `CheckpointCompatibilityError`
  with the offending field and expected/actual values. The three checked
  fields are `schema_version`, `feature_version`, `ruleset_id`. There is **no**
  permissive fallback — a mismatch always raises.
- **manifest absent (legacy pre-P01 checkpoint)** → delegates to
  `load_legacy_model_tar`, returns `(bundle, manifest=None)`. No version
  validation is possible because there is no manifest; callers assume the
  legacy feature/rule identity (acceptable in P01 because only `legacy`
  exists).

## Round-trip and tests

`tests/test_checkpoint_manifest.py` pins:
- new-format `model.tar` round-trips (tensors + manifest intact);
- legacy `model.tar` (no manifest) still loads;
- incompatible `feature_version` / `ruleset_id` / `schema_version` each raise
  a precise `CheckpointCompatibilityError`;
- `git_sha` is `"unknown"` (string) when git is unavailable.

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
