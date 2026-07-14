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
SHA-256, the teacher configuration hash (including the complete public Model
V2 configuration), and cache version; any mismatch is rejected.

## Losses

The configurable student objective combines:

- temperature KL over the current legal-action list;
- top-k pairwise ranking loss;
- teacher `p_win` and both conditional-score-head regressions;
- retained Monte-Carlo win supervision and outcome-selected conditional-score
  supervision on the action actually played.

`score_mean` remains a decision-only derived value and is never a loss target.
Teacher and student CLIs build Model V2 through the same repository-config
bridge as `train_v2.py`, including `loss.score_target_transform` and
`loss.score_clamp`. Terminal score labels use the ordinary V2 signed-log/raw
transform, representable-range clamp, and configurable Huber delta. Teacher
dense score targets are the two conditional heads on that same scale and are
clamped to the student's representable range.

Student epochs are split into independent minibatches controlled by
`distillation.batch_size` (default `32`) or the `--batch_size` override. Each
minibatch has its own forward, backward, and optimizer step, so graph memory is
bounded by the minibatch rather than the complete offline dataset.

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
  --output artifacts/p10/public_student.ckpt --epochs 10 --batch_size 32
```

The second command exports a manifest-bearing public policy sidecar loadable by
`DeepAgentV2`. A privileged teacher checkpoint is deliberately incompatible
with the ordinary and V2 public-policy loaders. Legacy rules, observations,
checkpoints, and default CLIs are unchanged; no migration is required.

P10 cache version 2 and teacher checkpoint version 2 are intentionally
incompatible with the earlier Draft formats. Delete and rebuild any P10 cache
or teacher checkpoint produced before this version; public and legacy
checkpoints are unaffected.
