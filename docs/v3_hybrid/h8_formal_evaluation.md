# H8a Formal Evaluation And Public-Package Infrastructure

## Status

- Base SHA: `92da343ce71a2ec77ca03260a9c09abc80d7b4ba`
- Release candidate: **NONE**
- Release status: **NOT READY**
- Playing strength: **NOT MEASURED**

H8a delivers contracts, validation, packaging, CLI coverage, and tests. It does
not run formal training, create candidate weights, or claim playing strength.
Those empirical gates belong to H8b, so H8a may merge in the status above.

## Scope

H8a does not change model, loss, replay, trainer, or search algorithms. It adds
a fail-closed evidence validator and a strict public-only V3 package. An
algorithmic defect discovered by later experiments must return to H1-H7.

## Support Matrix

The versioned machine-readable matrix is exposed by
`h8a_support_matrix_dict()`. It permits only current mainline combinations:

| Variant | Rulesets |
| --- | --- |
| `legacy_a1` | legacy |
| `model_v2` | legacy, standard |
| `v3_role` | legacy, standard |
| `v3_admc` | legacy, standard |
| `v3_oracle` | legacy, standard |
| `v3_belief` | legacy, standard |
| `v3_farmer_cooperation` | legacy, standard |
| `v3_full_hybrid` | legacy, standard |
| `v3_full_hybrid_bc` | legacy, and only with authorized data |

Unsupported rows fail closed. Missing supported rows produce a valid but
incomplete `NOT READY` report. Authorized BC evidence binds dataset identity,
license, version, pseudonymization contract, and HMAC key identity hash.

## Evidence Contract

`v3-hybrid-h8a-formal-evidence-v2` binds source, image, hardware, budgets,
seeds, feature/trainer/replay identities, deal sets, and one immutable reference
checkpoint/package identity per ruleset. Every training and evaluation row
additionally binds its complete ruleset identity, training configuration hash,
checkpoint hash, and the frozen ruleset reference used for comparison.
Resulting checkpoint digests must be distinct across the three training seeds
for each variant/ruleset experiment.

Evaluation has two distinct tiers:

- `development`: at least 20,000 paired deals for ablation screening; never
  creates a release candidate.
- `promotion`: at least 100,000 paired deals with deal-clustered 95% confidence
  intervals and all promotion gates satisfied.

Search is an evaluation/deployment wrapper. Promotion search-off and search-on
rows share one full-hybrid training checkpoint; H8a does not define a separate
search-trained variant.

The validator recomputes readiness from raw rows. Malformed identities,
duplicates, unsupported combinations, non-finite values, contradictory counts,
stale identities, and ruleset or checkpoint drift are rejected.

Each evaluation row carries a deal-cluster outcome histogram: exact counts of
identical overall and per-role WP/ADP delta vectors. Counts must sum to the
declared paired deals. The validator recomputes estimates and deterministic
deal-clustered bootstrap intervals from those counts and rejects caller-reported
statistics that differ. This keeps large evidence compact without trusting
derived confidence intervals.

Promotion search-on rows carry a separate paired search-effect histogram. Its
counts must cover the same deals, and the validator independently recomputes
the search-on versus search-off WP/ADP estimates and confidence intervals.
Histogram cardinality, bootstrap resamples, and their allocation product are
bounded before NumPy allocation so untrusted package evidence cannot request
unbounded verification work.

```bash
python tools/validate_v3_h8_evidence.py evidence.json --output report.json
```

Exit status is zero only for `READY`; valid `NOT READY` evidence exits with
status 2.

## Public Package

`v3-hybrid-h8-public-package-v1` strictly reloads H1 public policy checkpoints
and H4 coupled public-belief checkpoints. H6 public strategy, style, human
prior, and bidding modules are part of the V3 model identity and are preserved
by the same strict checkpoint path. The package also binds decision config,
search compatibility, ruleset identity, source SHA, and evidence identity.

The package carries the raw formal-evidence bundle and recomputes its report
during verification; a caller cannot establish `READY` by rewriting only the
manifest and report. The packaged checkpoint must match a passing promotion
row for the package ruleset and declared search mode. Decision configuration
is restricted to the public `argmax_dmc_q` schema and zero temperature, so
arbitrary path- or credential-bearing metadata is rejected.

The exact allow-list excludes Oracle, teacher, cooperation mixer, optimizer,
replay, privileged labels, trajectories, human identifiers, caches, local
paths, secrets, symlinks, and non-regular assets. A package without promotion evidence is deliberately a
non-release research artifact with playing strength not measured.

```bash
python tools/package_v3_hybrid.py --help
```

## H8b Work

H8b will freeze and execute real commit-bound experiments: three matched
training seeds, stability and fresh-container resume, development ablations,
100,000-deal promotion evaluation, search comparison, and empty-environment
validation of a promoted package. Authorized BC strength evaluation remains
conditional on licensed data. Until those gates pass, the release state stays
`NONE / NOT READY / NOT MEASURED`.
