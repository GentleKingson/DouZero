# H8 Formal Evaluation And Release Contract

## Status

- Base SHA: `92da343ce71a2ec77ca03260a9c09abc80d7b4ba`
- Release candidate: **NONE**
- Release status: **NOT READY**
- Playing strength: **NOT MEASURED**

No formal V3 weights, authorized human-game corpus, three-seed ablation
matrix, or 100,000-deal paired evaluation was available when this contract was
implemented. Functional tests and synthetic evidence must not be cited as
playing-strength evidence.

## Scope

H8 does not change the model graph, losses, replay, trainer protocol, or search
algorithm. It adds a fail-closed evidence validator and a strict public-only V3
package. Algorithmic defects discovered by formal runs must return to H1-H7.

## Evidence

`v3-hybrid-h8-formal-evidence-v1` binds source, image, hardware, resolved
configuration, training semantics, workload, feature/model/ruleset identities,
trainer topology, replay protocol, initial checkpoint, deal sets, seeds,
budgets, auxiliary configurations, schedules, and checkpoint cadence.

The validator recomputes the release decision from raw rows. It requires:

- every frozen ablation at every declared training seed;
- matched sample and wall-clock budgets;
- at least two cumulative hours, real SIGTERM, fresh-container resume, state
  continuity, a subsequent update, bounded policy lag, clean shutdown, and no
  stale artifact for every run;
- legacy and standard paired evaluation for every variant and seed;
- at least 100,000 deals, deal-clustered 95% confidence intervals, overall WP
  and ADP lower bounds above zero, and no role regression;
- ordered latency percentiles within the frozen budget, strict package
  validation, zero privileged leakage, invalid actions, timeouts, and model
  load failures.

Malformed identities, duplicate rows, contradictory counts, non-finite values,
unfrozen variants, and mismatched deal sets raise an error. Complete but failed
or missing gates produce `NOT READY`; they are not coerced into malformed data.

Run:

```bash
python tools/validate_v3_h8_evidence.py evidence.json --output report.json
```

Exit status is zero only for `READY`; valid `NOT READY` evidence exits with
status 2.

## Public Package

`v3-hybrid-h8-public-package-v1` accepts only a strict H1 public-policy or H4
coupled public-belief checkpoint. It reloads the checkpoint before copying it,
uses an exact file allow-list, checksums every asset, binds model, feature,
belief, ruleset, decision, source, search, and formal-evidence identities, and
reloads the copied checkpoint again before returning.

The package excludes training-only Oracle, mixer, labels, replay, optimizer,
raw trajectories, human identifiers, caches, local paths, and secrets. A
package without passing formal evidence is deliberately produced as a
non-release research artifact with playing strength not measured.

Run:

```bash
python tools/package_v3_hybrid.py --help
```

## Formal Work Remaining

The following items require approved weights/data and commit-bound Docker GPU
execution and therefore remain open:

1. Three seeds for the frozen ablation matrix under matched sample and time
   budgets.
2. Two-hour stability plus SIGTERM/fresh-container resume for every candidate.
3. Legacy and standard 100,000-deal paired evaluation with role confidence
   intervals.
4. Search-off/search-on overall-benefit comparison.
5. Authorized BC strength evaluation, if licensed data is supplied.
6. Final empty-environment validation of the promoted, checksummed package.

Until all items pass against one immutable identity, release remains `NONE / NOT READY`.
