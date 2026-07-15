# DouZero Model Card Template

Complete this document for every published model. Packaging will generate a
conservative placeholder when no reviewed card is supplied; placeholders do
not constitute evaluation results.

## Model Details

- Model/checkpoint version:
- Feature schema version and hash:
- Ruleset ID and hash:
- Supported roles:
- Dtype and target hardware:
- Belief model enabled:
- Search enabled during reported evaluation:

## Training Data

List data categories, authorization/provenance, date range, filtering, and the
handling of personal identifiers. Do not embed raw personal identifiers in the
model package.

## Evaluation

Report metrics by role, opponent, and seat using paired or seat-rotated games
with confidence intervals. Distinguish base-policy and search-enabled results.
Do not insert unmeasured numbers.

## Latency

Report device, batch/action padding limit, p50/p95 latency, export backend, and
whether the fallback path was exercised. Mark absent measurements as unmeasured.

## Known Limitations

Document unsupported rulesets, out-of-distribution opponents, calibration
limits, maximum action padding, belief/search failure modes, and rollback
conditions.

## Intended and Prohibited Uses

This project is for offline research and authorized competition/deployment. It
is not intended for platform account automation, scraping, anti-detection,
access to private information, or violation of service terms.

## License

Apache-2.0. Include `THIRD_PARTY_NOTICES` in every distributed package.
