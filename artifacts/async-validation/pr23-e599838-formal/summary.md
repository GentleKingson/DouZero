# PR #23 commit-bound async recovery soak

Source head: `e59983869ad88370af661d5238f6e50d4841b39e`
Image: `sha256:8c994b0b0ab8998f433d71ec92c25640d5beb16f5e45ba2fc496272e9c76b282`

The image contains the committed source; only read-only `.git` metadata and the evidence output directory were mounted.

| Topology | Phase | Cycles | Median decisions/s | Median transitions/s | Min requests/microbatch | Queue p50 median ms | Queue p95 median ms | Boundary violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| async4x4 | phase1 | 863 | 1324.55 | 708.83 | 5.29 | 14.84 | 24.62 | 0 |
| async4x4 | phase2 | 896 | 1209.36 | 721.98 | 5.83 | 14.78 | 24.35 | 0 |
| async8x4 | phase1 | 1150 | 1662.40 | 954.40 | 9.28 | 19.29 | 33.72 | 0 |
| async8x4 | phase2 | 1181 | 1564.65 | 979.89 | 9.80 | 19.04 | 32.27 | 0 |

| Topology | Phase | Peak RAM plateau cycles | Peak VRAM plateau cycles | Container RAM tail slope MiB/hour | GPU memory tail slope MiB/hour | PID tail slope/hour |
|---|---|---:|---:|---:|---:|---:|
| async4x4 | phase1 | 381 | 792 | 9.200 | 0.000 | 0.000 |
| async4x4 | phase2 | 216 | 314 | 8.707 | 0.000 | 0.000 |
| async8x4 | phase1 | 69 | 82 | 37.389 | 0.000 | 0.000 |
| async8x4 | phase2 | 292 | 118 | 49.803 | 18.953 | 0.000 |

## Checkpoints

- `async4x4`: optimizer step 863 -> 880; model hash changed: True; finite loss/grad: True.
  Signal checkpoint SHA-256: `96ca71e9386de97208aed38bda4e5ce98fadd5af65e3fedbd86b5c98e62d652b`
  First resumed checkpoint SHA-256: `8dc91fabe582613a8281ca96f80b95c98af1aca4132786c31f5d89890c248e06`
- `async8x4`: optimizer step 1150 -> 1160; model hash changed: True; finite loss/grad: True.
  Signal checkpoint SHA-256: `75c19d49bac340bad874eafb8747776f26930b2245223a4da6f12e4078672a27`
  First resumed checkpoint SHA-256: `6bd18ae64986c44b1be9511ebbc2a8e47f56e87ca44cf8427378d1a4a1f97022`

## Gates

- `async4x4`: PASS
- `async8x4`: PASS

Overall commit-bound soak: **PASS**

Independent approving review remains a separate merge gate.
