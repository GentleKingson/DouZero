# Model Migration and Rollback

## Legacy to Factorized

Legacy per-role weights can run under `legacy_factorized` without conversion.
This is the only intentionally permissive compatibility path and exists to
preserve original `DeepAgent` behavior. New release packages must not contain
bare legacy state dictionaries.

## Factorized to Model V2

Model V2 changes the observation schema, architecture, role handling, and value
heads. Weights cannot be converted by renaming keys. Construct a V2 model,
train or distill it, save a manifest-bearing `public_policy` sidecar, then build
a P16 model package. The runtime must independently provide the expected
`RuleSet`, `FeatureSchemaManifest`, and `ModelV2Config`.

## Older V2 Checkpoints

The strict V2 loader supports documented P05 and P06-P08 config-hash migration
only when semantics remain provably compatible. Unknown identity versions,
missing manifests, partial state dictionaries, feature drift, rule drift, and
privileged teachers are rejected. Re-save a successfully loaded older model as
a current public sidecar before packaging; never edit manifest hashes by hand.
P16 release packages additionally bind to the deployment ABI version and exact
implementation hash. A semantic implementation change requires a new package;
do not weaken the hash check to reuse an older release artifact.

## Evaluation Data

Legacy card-play JSON is adapted through
`douzero.evaluation.legacy_data_adapter`. Preserve the original input and write
converted output to a new path. Stamp the ruleset identity used for conversion
and compare a fixed subset before launching a full evaluation.

## Rollback

1. Stop routing new games to the P16 package and retain its logs/checksums.
2. Restore the previous immutable package directory and its runtime-owned
   schema/ruleset/config values.
3. Verify `SHA256SUMS`, run one fixed-state inference, then resume traffic.
4. Do not downgrade by loading V2 weights through the legacy partial loader.

Packages are immutable. Build a new versioned directory for every release or
rollback candidate; the packaging command refuses to overwrite a non-empty
directory.
