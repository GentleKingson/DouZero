# Third-Party Review

> License audit for the DouZero enhancement project. This file records every
> external source consulted during the enhancement, its license, and whether
> any code was copied. The baseline project (`kwai/DouZero`) is Apache-2.0;
> this distribution remains Apache-2.0, so **GPL-licensed or
> license-unverified code must not be copied** into this repository.
> Algorithmic ideas may be implemented independently from papers and public
> behaviour.

## Principle

Per `AGENTS.md` ("Dependencies, licenses, and third-party code"):

> Before copying or adapting third-party code:
> 1. Identify its exact license.
> 2. Determine compatibility with this repository's intended distribution.
> 3. Record the source and decision in third-party documentation.
> 4. Prefer an independent implementation from papers and public behavior when
>    the source is GPL, has no clear license, or is otherwise incompatible.

For every external source referenced by an enhancement phase, the table below
records the source, its license, the idea consulted, and confirms whether code
was copied.

## Consulted sources

| Source | License | Phase | Idea consulted | Code copied? |
|--------|---------|-------|----------------|--------------|
| `kwai/DouZero` | Apache-2.0 | all | The upstream project this repo derives from. Modified under the Apache-2.0 terms; copyright notices preserved in `LICENSE`. | n/a (this is the base) |
| `RuBP17/AlphaDou` | GPLv3 | P02, P06, P08 | Full bidding + win-probability/expected-score dual objectives; listwise action selection over legal actions. Implemented independently from the published description. | **No** (GPLv3 incompatible with Apache-2.0 distribution) |
| `DouZero+` (paper / public behaviour) | research paper | P07, P08, P12 | Opponent hidden-card prediction; coach-guided curriculum. Implemented independently; no reference implementation copied. | **No** |
| `DouRN` (paper / public behaviour) | research paper | P05 | Residual backbone + per-role evaluation. Implemented independently. | **No** |
| PyTorch (`torch.autocast`, `GradScaler`, DDP, `torch.compile`) | BSD-style | P14 (future) | Official PyTorch training APIs; used per the upstream docs. | n/a (library usage) |

## P09 — strategy features and cooperation

No third-party strategy implementation was consulted or copied. The bounded
hand-decomposition search, structure costs, cooperation features, trajectory
labels, auxiliary heads, and uncertainty gate were implemented independently
using this repository's Apache-2.0 move generator/detector and standard
PyTorch loss primitives. In particular, no GPL-licensed or license-unverified
DouDizhu derivative supplied code or file structure for this phase.

## P08 — human-game data and listwise BC

No third-party code was copied for the human-game data pipeline or the
listwise behaviour-cloning prior. Specifically:

- The JSONL canonical record format and the replay validator are original to
  this project; they use only the project's own rule engine
  (`douzero.env.game.GameEnv`).
- The listwise cross-entropy over the legal-action list is a direct
  application of `torch.nn.functional.cross_entropy` to a per-decision action
  set (the standard supervised-learning formulation); it is not adapted from
  any DouZero derivative's implementation.
- No scraping, account-automation, anti-detection, or platform-ToS-bypass code
  was written or referenced. The pipeline operates exclusively on
  already-acquired, authorized offline data.

## New runtime dependencies introduced by the enhancement

None beyond the upstream `kwai/DouZero` dependencies (`torch`, `rlcard`,
`GitPython`, `pyyaml`). P08 deliberately uses JSONL (stdlib `json`) rather
than Parquet to avoid adding `pyarrow`/`pandas` as runtime dependencies.
P09 likewise adds no dependency.

## P10 — privileged teacher distillation

No third-party implementation was consulted or copied. The privileged branch,
canonical legal-action alignment, temperature KL, pairwise ranking loss,
offline tensor bundle, strict cache identity, and checkpoint access guard were
implemented independently with this repository's existing Observation V2 and
PyTorch primitives. P10 adds no runtime dependency.

## Conclusion

No GPL-licensed, license-unverified, or otherwise-incompatible code has been
copied into this Apache-2.0 distribution. All consulted ideas are implemented
independently from papers and public behaviour.
