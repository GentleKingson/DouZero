# Coach-Guided Opening Curriculum (P12)

P12 adds an opt-in, training-only opening curriculum. It does not alter the
legacy environment default, deployment agents, or evaluation deal selection.
The implementation lives in `douzero.coach`; no evaluation module imports it.

## Data and model boundaries

`OpeningRecord` stores a complete 54-card deck, bidding order, RuleSet,
bidding result/candidate landlord, and initial public metadata. The full deck
is privileged training data. `Env.reset(opening=...)` converts it into the
normal validated deal and never attaches it to an infoset or observation.

Each completed sampled game may append a `CoachLabel` to a separate JSONL
store. Labels contain the exact `policy_version` and numeric `policy_step`.
`CoachLabelStore.load_fresh` accepts only matching policy versions and rejects
future or older-than-configured labels. A live curriculum therefore produces
fresh labels continuously; callers should periodically refit the coach from
that filtered window and publish a new independent coach checkpoint.

The repository provides that refit path directly:

```bash
python train_coach.py \
  --labels artifacts/coach/labels.jsonl \
  --output artifacts/coach/coach.pt \
  --policy_version policy-120000 \
  --current_policy_step 120000 \
  --max_label_age_steps 20000
```

The command rejects mixed RuleSet identities, uses a stable content-addressed
holdout when enough labels exist, reports measured calibration, and writes the
new checkpoint atomically.

`CoachModel` predicts landlord win probability from an opening plus a stable
encoding of the policy version. Its checkpoint is separate from value, belief,
teacher, and league checkpoints. The manifest pins:

- coach model and feature versions;
- architecture hash;
- policy version and step used for labels;
- RuleSet hash;
- measured calibration metrics, when supplied.

`calibration_metrics` reports Brier score and expected calibration error. No
unmeasured calibration or win-rate result is shipped with this change.

## Sampling modes

- `true_random`: one uniformly shuffled complete deck.
- `balanced`: among a configured candidate pool, choose the coach prediction
  nearest landlord win probability 0.5.
- `hard_for_role`: choose the lowest landlord probability for landlord
  training, or the highest for farmer training.
- `mixture`: sample one of the three strategies from the active curriculum
  phase.

The early, middle, and late phase proportions are configuration fields. Every
phase must satisfy `min_true_random_ratio`; the default late phase returns to
90 percent true-random deals. This fixed real-deal floor prevents permanent
distribution collapse without claiming an approximate importance weight.
Fixed `balanced` and `hard_for_role` modes also mix in the configured minimum
true-random ratio; guided-only sampling is deliberately not a production mode.

Every sampled opening can be written to the audit JSONL stream. Each row
contains the configured proportions, selected strategy and its probability,
predicted landlord win probability, opening ID, phase/progress, and cumulative
actual counts/distribution. The training distribution can therefore be
reconstructed without retaining model tensors in the log.

## Configuration

The `curriculum:` block in `configs/enhanced.yaml` is disabled by default.
For guided modes, set `enabled: true`, provide `coach_checkpoint`, and choose
optional `labels_path` and `audit_log_path` outputs. `train_v2.py` loads and
validates the coach before starting collection. `true_random` mode needs no
coach checkpoint and can be used to exercise deterministic opening replay.

Final evaluation must continue to use `evaluate.py` and its ordinary random
or fixed paired deal data. The evaluation package contains no coach sampler
import and cannot enable this curriculum.
