# gpu_v3 capability and performance decision record

`gpu_v3` is an isolated GPU-native experiment. It uses `feature_version=v2`,
`model_version=gpu_v3`, and `checkpoint_kind=public_policy`; its strict loader
rejects Legacy, factorized, and V2 checkpoints. No gpu_v3 module is selected by
the Legacy A1 or C0 training entry point.

Two architectures are retained for capability study:

- `independent_role_dual_tower` preserves completely independent state/action
  representation and value heads for all three roles.
- `shared_trunk_role_heads` computes the state/action representation once and
  applies small role-specific value heads, reducing parameters and increasing
  compatible batch density.

Legacy teacher distillation accepts factorized public inputs, pads them into a
fixed GPU bucket, and trains only the gpu_v3 student. Teacher checkpoints remain
read-only Legacy `position_weights`; student output always uses the separate
gpu_v3 manifest.

The CUDA capability benchmark uses CUDA events and reports parameter count,
median/p95 latency, and decisions/s for `(batch, action bucket)` cases of
`(1,64)`, `(16,256)`, and `(64,512)`. These are synthetic model-only capacity
measurements, not gameplay strength or end-to-end training claims.

On the RTX 5070 with PyTorch `2.12.1+cu132` and CUDA `13.2`, the 100-round
benchmark measured:

| architecture | parameters | batch/actions | median ms | decisions/s |
|---|---:|---:|---:|---:|
| independent dual tower | 6,899,715 | 1/64 | 0.325 | 3,081.6 |
| independent dual tower | 6,899,715 | 16/256 | 1.171 | 13,667.4 |
| independent dual tower | 6,899,715 | 64/512 | 5.302 | 12,071.2 |
| shared trunk/role heads | 3,617,795 | 1/64 | 0.323 | 3,099.0 |
| shared trunk/role heads | 3,617,795 | 16/256 | 0.909 | 17,595.1 |
| shared trunk/role heads | 3,617,795 | 64/512 | 5.740 | 11,149.5 |

The shared model is the preferred capability candidate: it uses 47.6% fewer
parameters and is 22.3% faster at the representative 16/256 batch. The 64/512
case is slower than the independent model, so larger buckets are not assumed
to be automatically better.

gpu_v3 substantially increases arithmetic density, so centralized inference
may become worth re-evaluating. That does not reverse the Legacy C0 decision:
C0 remains experimental, and no production topology changes until a gpu_v3
actor/learner integration demonstrates three-repeat end-to-end gains over A1,
checkpoint/resume stability, bounded policy lag, and acceptable playing
strength. Current production recommendation remains Legacy-compatible A1.
