# Privileged Teacher and Public Student Distillation

P10 adds a training-only perfect-information teacher. The deployed policy
remains `douzero.models_v2.ModelV2` and receives the same public Observation V2
tensors as before.

## Information boundary

- `TeacherModel.forward(public_input, privileged_observation)` requires a
  `PrivilegedObservation` containing the exact remaining hands. It lives only
  under `douzero.distillation`.
- `ModelV2.forward` has no hidden-hand argument. `DeepAgentV2` imports no
  privileged module and accepts only `ObservationV2`.
- Teacher checkpoints carry `checkpoint_kind=privileged_teacher` and
  `model_access=privileged`. Public policy sidecars carry
  `checkpoint_kind=public_policy` and `model_access=public`.
- `export_public_student` accepts only a public `ModelV2`; it rejects a teacher
  before writing anything.

Perfect information is used only to create dense training targets. It is not
stored in a public checkpoint, public example input, `DeepAgentV2`, or the
student forward signature.

## Offline data and cache

`DistillationSample.tensorize()` separates the public `ModelInputBundle` from
the explicitly named `PrivilegedObservation`. `save_offline_dataset` writes a
weights-only-loadable bundle and `load_offline_dataset` validates dataset,
feature-schema, and ruleset identities before returning samples.

Teacher output is aligned to student rows by sorted canonical action keys.
The cache key includes both public tensors and the privileged allocation,
because different true hands can share the same public observation. Cache
metadata binds the feature-schema hash, ruleset hash, exact teacher state-dict
SHA-256, and cache version; any mismatch is rejected.

## Losses

The configurable student objective combines:

- temperature KL over the current legal-action list;
- top-k pairwise ranking loss;
- teacher `p_win` and expected-score regression;
- retained Monte-Carlo win and score supervision on the action actually
  played.

`distillation.enabled` defaults to `false`. Disabled mode refuses a teacher or
teacher cache and trains only from public inputs plus terminal labels, making
the ablation explicit.

## Commands

Create an authorized offline self-play dataset with the programmatic dataset
API, then enable the `distillation` block in a copy of
`configs/enhanced.yaml`.

```bash
python -m douzero.distillation.train_teacher \
  --config configs/p10.yaml --dataset artifacts/p10/selfplay.pt \
  --output artifacts/p10/teacher.pt --epochs 10

python -m douzero.distillation.distill_student \
  --config configs/p10.yaml --dataset artifacts/p10/selfplay.pt \
  --teacher artifacts/p10/teacher.pt \
  --output artifacts/p10/public_student.ckpt --epochs 10
```

The second command exports a manifest-bearing public policy sidecar loadable by
`DeepAgentV2`. A privileged teacher checkpoint is deliberately incompatible
with the ordinary and V2 public-policy loaders. Legacy rules, observations,
checkpoints, and default CLIs are unchanged; no migration is required.
