"""Create, verify, and load self-contained Model V2 release directories."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch

from douzero.deployment.abi import MODEL_ABI_VERSION, model_implementation_hash
from douzero.deployment.manifest import (
    PUBLIC_MODEL,
    ModelManifest,
    ModelManifestError,
    build_model_manifest,
    canonical_hash,
)

if TYPE_CHECKING:
    from douzero.belief.model import BeliefConfig, BeliefModel
    from douzero.env.rules import RuleSet
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import FeatureSchemaManifest

_CORE_REQUIRED_FILES = frozenset({
    "weights.pt",
    "manifest.json",
    "ruleset.json",
    "feature_schema.json",
    "model_config.json",
    "training_config.json",
    "README.md",
    "model_card.md",
    "evaluation_summary.md",
    "gpu_validation_summary.md",
    "rollback.md",
    "THIRD_PARTY_NOTICES",
    "SHA256SUMS",
})


def _required_files(
    *, belief_enabled: bool, bidding_enabled: bool
) -> frozenset[str]:
    extra = (
        {"belief_config.json", "belief_weights.pt"}
        if belief_enabled
        else set()
    )
    if bidding_enabled:
        extra.add("bidding_schema.json")
    return frozenset(set(_CORE_REQUIRED_FILES) | extra)


class ModelPackageError(RuntimeError):
    """Raised when a model package is incomplete, corrupt, or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _default_model_card(manifest: ModelManifest) -> str:
    return f"""# DouZero Unreleased Model Card

## Release Status

- Release candidate: **NONE**
- Release status: **NOT READY**

## Model Details

- Model version: `{manifest.model_version}`
- Model ABI: `{manifest.model_abi_version}`
- Implementation hash: `{manifest.implementation_hash}`
- Source Git SHA: `{manifest.git_sha}`
- Feature version: `{manifest.feature_version}`
- Ruleset: `{manifest.ruleset_id}`
- Roles: `{', '.join(manifest.role_support)}`
- Belief model enabled: `{str(manifest.belief_enabled).lower()}`
- Learned bidding enabled: `{str(manifest.bidding_enabled).lower()}`
- Search compatible: `{str(manifest.search_compatible).lower()}`
- Numeric dtype: `{manifest.dtype}`
- Training configuration hash: `{manifest.training_config_hash}`
- Belief configuration hash: `{manifest.belief_config_hash}`

## Training Data

- Data categories: `NOT AVAILABLE`
- Authorization and provenance: `NOT AVAILABLE`
- Authorized human-data status: `NOT AVAILABLE`
- Training hardware: `NOT MEASURED`

Raw personal identifiers and canonical or raw game records are not included in
this package.

## Evaluation

- Paired card-play metrics: `NOT MEASURED`
- Standard full-game metrics: `NOT MEASURED`
- Role metrics and confidence intervals: `NOT MEASURED`
- Calibration and ablations: `NOT MEASURED`

## Latency

- Target hardware p50/p95/p99: `NOT MEASURED`
- Peak memory and throughput: `NOT MEASURED`
- AMP and NCCL DDP: `NOT MEASURED`

## Known Limitations

This package was created without reviewed empirical summaries and is not a
release candidate. It may fail outside the declared ruleset and feature
schema. Roll back for checksum/identity failure, non-finite inference, illegal
actions, privileged-information leakage, or an unexplained paired regression.

## Intended And Prohibited Uses

Offline research and explicitly authorized evaluation only. Prohibited uses
include platform automation, account operation, scraping, anti-detection,
service-control bypass, or decisions involving undisclosed hidden information.

## License

Apache-2.0. See `THIRD_PARTY_NOTICES` for dependency and reference-source attribution.
"""


_DEFAULT_EVALUATION_SUMMARY = """# Evaluation Summary

Status: **NOT MEASURED**

No paired card-play, standard full-game, calibration, latency, or ablation
result was supplied to the packaging command. This package must not be treated
as a release candidate until the declared evaluation gates are completed.
"""


_DEFAULT_GPU_VALIDATION_SUMMARY = """# GPU Validation Summary

Status: **NOT MEASURED**

No target-hardware FP32, AMP, NCCL DDP, checkpoint-resume, memory, or throughput
report was supplied to the packaging command.
"""


_DEFAULT_ROLLBACK = """# Rollback Instructions

1. Stop routing new games to this model package.
2. Restore the previously approved, checksummed package without modifying it.
3. Verify that package with the runtime-owned ruleset, feature schema, and
   model configuration identities before serving traffic.
4. Preserve this package and its evaluation logs for incident analysis.

Rollback is required for checksum or identity failures, non-finite inference,
illegal-action output, material release-gate regression, or undeclared use of
privileged information.
"""


def _validated_markdown(value: str | None, default: str, label: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ModelPackageError(f"{label} must be non-empty Markdown")
    return value if value.endswith("\n") else value + "\n"


def _belief_config_payload(config: "BeliefConfig") -> dict[str, Any]:
    """Serialize only a real BeliefConfig and its compatibility identity."""

    from douzero.belief.model import BeliefConfig

    if not isinstance(config, BeliefConfig):
        raise TypeError(
            f"belief config must be BeliefConfig, got {type(config).__name__}"
        )
    return {
        "schema_version": 2,
        "belief_config_hash": config.stable_hash(),
        "config": asdict(config),
        "compatibility": config.compatibility_dict(),
    }


def _parse_belief_config_payload(value: Any) -> "BeliefConfig":
    """Reconstruct and identity-check a packaged BeliefConfig payload."""

    from dataclasses import fields

    from douzero.belief.model import BeliefConfig

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "belief_config_hash",
        "config",
        "compatibility",
    }:
        raise ModelPackageError("belief_config.json has an invalid schema")
    if value["schema_version"] != 2:
        raise ModelPackageError(
            "belief_config.json has an unsupported schema_version"
        )
    config_payload = value["config"]
    expected_fields = {field.name for field in fields(BeliefConfig)}
    if not isinstance(config_payload, dict) or set(config_payload) != expected_fields:
        raise ModelPackageError(
            "belief_config.json must contain the exact BeliefConfig fields"
        )
    try:
        config = BeliefConfig(**config_payload)
    except (TypeError, ValueError) as exc:
        raise ModelPackageError(f"invalid BeliefConfig in belief_config.json: {exc}") from exc
    if (
        value["compatibility"] != config.compatibility_dict()
        or value["belief_config_hash"] != config.stable_hash()
    ):
        raise ModelPackageError(
            "belief_config.json identity does not match the runtime BeliefConfig"
        )
    return config


def create_model_package(
    output_dir: str | Path,
    model: "ModelV2",
    ruleset: "RuleSet",
    *,
    training_config: Mapping[str, Any] | None = None,
    search_compatible: bool = False,
    model_card: str | None = None,
    evaluation_summary: str | None = None,
    gpu_validation_summary: str | None = None,
    rollback_instructions: str | None = None,
    belief_checkpoint: str | Path | None = None,
) -> ModelManifest:
    """Write a checksummed public Model V2 package.

    The destination must be absent or empty; existing release contents are
    never overwritten silently.
    """

    from douzero.checkpoint import save_v2_position_weights

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ModelPackageError(f"output directory is not empty: {root}")
    belief_enabled = bool(model.config.belief_enabled)
    belief_model: "BeliefModel | None" = None
    belief_source: Path | None = None
    if belief_enabled:
        if belief_checkpoint is None:
            raise ModelPackageError(
                "belief-enabled packages require a manifest-bearing "
                "belief_checkpoint"
            )
        belief_source = Path(belief_checkpoint)
        if not belief_source.is_file():
            raise ModelPackageError(
                f"belief checkpoint is not a file: {belief_source}"
            )
        from douzero.belief.checkpoint import load_belief_checkpoint

        try:
            belief_model = load_belief_checkpoint(
                str(belief_source),
                expected_ruleset=ruleset,
                expected_feature_version=model.schema.feature_version,
                map_location="cpu",
                require_full_git_sha=True,
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ModelPackageError(
                f"belief checkpoint identity does not match the package: {exc}"
            ) from exc
    elif belief_checkpoint is not None:
        raise ModelPackageError(
            "belief_checkpoint was supplied for a belief-disabled model"
        )
    root.mkdir(parents=True, exist_ok=True)

    manifest = build_model_manifest(
        model,
        ruleset,
        training_config=training_config,
        belief_config=belief_model.config if belief_model is not None else None,
        search_compatible=search_compatible,
        public_or_privileged=PUBLIC_MODEL,
    )
    weights_path = root / "weights.pt"
    save_v2_position_weights(
        str(weights_path),
        model,
        ruleset=ruleset,
        flags={
            "training_config_hash": manifest.training_config_hash,
            "training_config_payload_policy": "hash_only",
        },
    )
    manifest = replace(manifest, weights_sha256=_sha256(weights_path))
    if belief_source is not None:
        shutil.copyfile(belief_source, root / "belief_weights.pt")
    _write_json(root / "manifest.json", manifest.to_dict())
    _write_json(root / "ruleset.json", ruleset.to_dict())
    _write_json(root / "feature_schema.json", model.schema.to_dict())
    _write_json(root / "model_config.json", {
        "schema_version": 1,
        "model_config_hash": manifest.model_config_hash,
        "config": asdict(model.config),
    })
    # Training configs commonly contain private paths.  A public package keeps
    # their canonical identity but never copies the raw payload by default.
    _write_json(root / "training_config.json", {
        "schema_version": 1,
        "training_config_hash": manifest.training_config_hash,
        "payload_policy": "hash_only",
        "payload_included": False,
    })
    if manifest.belief_enabled:
        assert belief_model is not None
        belief_payload = _belief_config_payload(belief_model.config)
        if belief_payload["belief_config_hash"] != manifest.belief_config_hash:
            raise ModelPackageError(
                "belief checkpoint config identity does not match manifest.json"
            )
        _write_json(root / "belief_config.json", belief_payload)
    if manifest.bidding_enabled:
        from douzero.observation.bidding import BIDDING_ACTIONS

        _write_json(root / "bidding_schema.json", {
            "schema_version": 1,
            "bidding_head_version": manifest.bidding_head_version,
            "bidding_action_schema": manifest.bidding_action_schema,
            "bidding_actions": list(BIDDING_ACTIONS),
            "bidding_feature_schema_hash": manifest.bidding_feature_schema_hash,
            "feature_schema": model.bidding_schema.compatibility_dict(),
        })
    card = _validated_markdown(
        model_card, _default_model_card(manifest), "model_card"
    )
    (root / "model_card.md").write_text(card, encoding="utf-8")
    (root / "README.md").write_text(
        "# DouZero Model Package\n\n"
        "Verify this directory with `verify_model_package` before loading it. "
        "See `model_card.md`, `evaluation_summary.md`, "
        "`gpu_validation_summary.md`, and `rollback.md`.\n",
        encoding="utf-8",
    )
    (root / "evaluation_summary.md").write_text(
        _validated_markdown(
            evaluation_summary, _DEFAULT_EVALUATION_SUMMARY, "evaluation_summary"
        ),
        encoding="utf-8",
    )
    (root / "gpu_validation_summary.md").write_text(
        _validated_markdown(
            gpu_validation_summary,
            _DEFAULT_GPU_VALIDATION_SUMMARY,
            "gpu_validation_summary",
        ),
        encoding="utf-8",
    )
    (root / "rollback.md").write_text(
        _validated_markdown(
            rollback_instructions, _DEFAULT_ROLLBACK, "rollback_instructions"
        ),
        encoding="utf-8",
    )

    notices = Path(__file__).with_name("THIRD_PARTY_NOTICES")
    if not notices.is_file():
        raise ModelPackageError(f"release audit file is missing: {notices}")
    shutil.copyfile(notices, root / "THIRD_PARTY_NOTICES")

    required_files = _required_files(
        belief_enabled=manifest.belief_enabled,
        bidding_enabled=manifest.bidding_enabled,
    )
    checksummed = sorted(required_files - {"SHA256SUMS"})
    sums = "".join(f"{_sha256(root / name)}  {name}\n" for name in checksummed)
    (root / "SHA256SUMS").write_text(sums, encoding="ascii")
    return manifest


def _installed_version(package: str) -> str | None:
    if package == "python":
        return ".".join(str(v) for v in sys.version_info[:3])
    if package == "torch":
        return str(torch.__version__).split("+", 1)[0]
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def _satisfies(installed: str | None, constraint: str) -> bool:
    if installed is None:
        return False
    if constraint.startswith("=="):
        return installed == constraint[2:]
    if constraint.startswith(">="):
        have = _version_tuple(installed)
        need = _version_tuple(constraint[2:])
        width = max(len(have), len(need))
        return have + (0,) * (width - len(have)) >= need + (0,) * (width - len(need))
    return False


def verify_model_package(
    package_dir: str | Path,
    *,
    expected_ruleset: "RuleSet | None" = None,
    expected_schema_hash: str | None = None,
    expected_feature_version: str | None = None,
    expected_model_config_hash: str | None = None,
    expected_belief_enabled: bool | None = None,
    expected_bidding_enabled: bool | None = None,
    allow_privileged: bool = False,
    check_package_versions: bool = True,
) -> ModelManifest:
    """Verify files, checksums, access class, versions, and runtime identity."""

    root = Path(package_dir)
    if not root.is_dir():
        raise ModelPackageError(f"model package is not a directory: {root}")
    # Read the manifest before applying the format-2 file allowlist. A valid
    # P16 directory otherwise looks merely "incomplete" because it predates
    # the P17 documents. It still must fail closed, but with actionable
    # compatibility guidance rather than an invitation to add files by hand.
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ModelPackageError("model package manifest.json must not be a symlink")
    if not manifest_path.is_file():
        raise ModelPackageError("model package is missing files: ['manifest.json']")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"invalid manifest.json: {exc}") from exc
    manifest_format = (
        manifest_payload.get("format_version")
        if isinstance(manifest_payload, Mapping)
        else None
    )
    if (
        isinstance(manifest_format, int)
        and not isinstance(manifest_format, bool)
        and manifest_format == 1
    ):
        raise ModelPackageError(
            "format-1 P16 package requires its matching P16 runtime and cannot "
            "be loaded by the format-2 P17 verifier; rebuild a new package "
            "from the original manifest-bearing public checkpoint and reviewed "
            "metadata rather than editing the old directory"
        )
    try:
        manifest = ModelManifest.from_dict(manifest_payload)
    except ModelManifestError as exc:
        raise ModelPackageError(f"invalid manifest.json: {exc}") from exc

    # The manifest determines the conditional belief/bidding payload. Check the
    # complete format-2 set only after its version and capabilities are known.
    required_files = _required_files(
        belief_enabled=manifest.belief_enabled,
        bidding_enabled=manifest.bidding_enabled,
    )
    missing = sorted(name for name in required_files if not (root / name).is_file())
    if missing:
        raise ModelPackageError(f"model package is missing files: {missing}")
    actual_entries = {path.name for path in root.iterdir()}
    unexpected = sorted(actual_entries - required_files)
    if unexpected:
        raise ModelPackageError(
            f"model package contains unexpected files: {unexpected}"
        )
    if any((root / name).is_symlink() for name in required_files):
        raise ModelPackageError("model package required files must not be symlinks")
    for name in (
        "README.md", "model_card.md", "evaluation_summary.md", "gpu_validation_summary.md", "rollback.md"
    ):
        if not (root / name).read_text(encoding="utf-8").strip():
            raise ModelPackageError(f"model package document is empty: {name}")
    if manifest.public_or_privileged != PUBLIC_MODEL and not allow_privileged:
        raise ModelPackageError(
            "privileged models are training-only and are rejected by the production loader"
        )
    if manifest.model_version != "v2":
        raise ModelPackageError(
            f"unsupported packaged model_version {manifest.model_version!r}; expected 'v2'"
        )
    if manifest.model_abi_version != MODEL_ABI_VERSION:
        raise ModelPackageError(
            f"model ABI mismatch: package has {manifest.model_abi_version!r}, "
            f"runtime expects {MODEL_ABI_VERSION!r}"
        )
    runtime_implementation_hash = model_implementation_hash()
    if manifest.implementation_hash != runtime_implementation_hash:
        raise ModelPackageError(
            "model implementation hash mismatch: the runtime deployment code "
            "differs from the code used to build this package"
        )

    expected_lines = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] not in required_files - {"SHA256SUMS"}:
            raise ModelPackageError(f"malformed SHA256SUMS entry: {line!r}")
        expected_lines[parts[1]] = parts[0]
    expected_names = required_files - {"SHA256SUMS"}
    if set(expected_lines) != expected_names:
        raise ModelPackageError("SHA256SUMS does not cover every required payload")
    for name, expected in expected_lines.items():
        actual = _sha256(root / name)
        if actual != expected:
            raise ModelPackageError(f"checksum mismatch for {name}: {actual} != {expected}")
    if manifest.weights_sha256 != _sha256(root / "weights.pt"):
        raise ModelPackageError("manifest weights_sha256 does not match weights.pt")
    try:
        ruleset_payload = json.loads((root / "ruleset.json").read_text(encoding="utf-8"))
        schema_payload = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
        model_config_payload = json.loads(
            (root / "model_config.json").read_text(encoding="utf-8")
        )
        training_config_payload = json.loads(
            (root / "training_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"invalid package JSON: {exc}") from exc
    from douzero.env.rules import RuleSet

    packaged_ruleset = RuleSet.from_dict(ruleset_payload)
    if (
        packaged_ruleset.ruleset_id != manifest.ruleset_id
        or packaged_ruleset.stable_hash() != manifest.ruleset_hash
    ):
        raise ModelPackageError("ruleset.json identity does not match manifest.json")
    if expected_ruleset is not None and (
        expected_ruleset.ruleset_id != manifest.ruleset_id
        or expected_ruleset.stable_hash() != manifest.ruleset_hash
    ):
        raise ModelPackageError("runtime ruleset does not match packaged model")
    from douzero.observation.schema import FeatureSchemaManifest, FieldSpec

    # Reconstruct only to recompute the canonical compatibility hash. This also
    # makes field-order or dtype tampering visible even if the manifest is intact.
    schema_raw = dict(schema_payload)
    for group in (
        "state_fields", "action_fields", "history_token_fields",
        "context_fields", "bidding_token_fields",
    ):
        schema_raw[group] = tuple(
            FieldSpec(
                name=item["name"],
                shape=tuple(item["shape"]),
                dtype=item["dtype"],
                description=item["description"],
            )
            for item in schema_raw[group]
        )
    packaged_schema = FeatureSchemaManifest(**schema_raw)
    if packaged_schema.feature_version != manifest.feature_version:
        raise ModelPackageError(
            "feature_schema.json feature_version does not match manifest.json"
        )
    if packaged_schema.stable_hash() != manifest.feature_schema_hash:
        raise ModelPackageError("feature_schema.json identity does not match manifest.json")

    if not isinstance(model_config_payload, dict) or set(model_config_payload) != {
        "schema_version", "model_config_hash", "config"
    }:
        raise ModelPackageError("model_config.json has an invalid schema")
    if model_config_payload["schema_version"] != 1:
        raise ModelPackageError("model_config.json has an unsupported schema_version")
    from douzero.models_v2.config import ModelV2Config

    try:
        packaged_model_config = ModelV2Config(**model_config_payload["config"])
    except (TypeError, ValueError) as exc:
        raise ModelPackageError(f"invalid model_config.json: {exc}") from exc
    if (
        model_config_payload["model_config_hash"] != manifest.model_config_hash
        or packaged_model_config.stable_hash() != manifest.model_config_hash
    ):
        raise ModelPackageError("model_config.json identity does not match manifest.json")

    expected_training_payload = {
        "schema_version": 1,
        "training_config_hash": manifest.training_config_hash,
        "payload_policy": "hash_only",
        "payload_included": False,
    }
    if training_config_payload != expected_training_payload:
        raise ModelPackageError(
            "training_config.json identity or hash-only payload policy does not "
            "match manifest.json"
        )
    packaged_belief_config = None
    if manifest.belief_enabled:
        try:
            belief_payload = json.loads(
                (root / "belief_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelPackageError(f"invalid belief_config.json: {exc}") from exc
        packaged_belief_config = _parse_belief_config_payload(belief_payload)
        if packaged_belief_config.stable_hash() != manifest.belief_config_hash:
            raise ModelPackageError(
                "belief_config.json identity does not match manifest.json"
            )
        from douzero.belief.checkpoint import load_belief_checkpoint

        try:
            checked_belief_model = load_belief_checkpoint(
                str(root / "belief_weights.pt"),
                expected_ruleset=packaged_ruleset,
                expected_feature_version=packaged_schema.feature_version,
                expected_belief_config=packaged_belief_config,
                map_location="cpu",
                require_full_git_sha=True,
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ModelPackageError(
                "belief_weights.pt checkpoint identity does not match the "
                f"package: {exc}"
            ) from exc
        if checked_belief_model.config.stable_hash() != manifest.belief_config_hash:
            raise ModelPackageError(
                "belief_weights.pt config identity does not match manifest.json"
            )
    elif manifest.belief_config_hash != canonical_hash(None):
        raise ModelPackageError(
            "belief-disabled package has a non-empty belief configuration identity"
        )
    if manifest.bidding_enabled:
        try:
            bidding_payload = json.loads(
                (root / "bidding_schema.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelPackageError(f"invalid bidding_schema.json: {exc}") from exc
        from douzero.observation.bidding import (
            BIDDING_ACTIONS,
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
            build_bidding_schema,
        )

        runtime_bidding_schema = build_bidding_schema()
        expected_bidding_payload = {
            "schema_version": 1,
            "bidding_head_version": BIDDING_HEAD_VERSION,
            "bidding_action_schema": BIDDING_ACTION_SCHEMA_VERSION,
            "bidding_actions": list(BIDDING_ACTIONS),
            "bidding_feature_schema_hash": runtime_bidding_schema.stable_hash(),
            "feature_schema": runtime_bidding_schema.compatibility_dict(),
        }
        if bidding_payload != expected_bidding_payload or (
            manifest.bidding_head_version != BIDDING_HEAD_VERSION
            or manifest.bidding_action_schema != BIDDING_ACTION_SCHEMA_VERSION
            or manifest.bidding_feature_schema_hash
            != runtime_bidding_schema.stable_hash()
        ):
            raise ModelPackageError(
                "bidding_schema.json identity does not match manifest or runtime"
            )

    # Validate the manifest-bearing checkpoint sidecar itself, not only the
    # outer package metadata. A release pipeline must not approve a package
    # whose checksums are internally consistent but whose weights were replaced
    # by another valid public-policy sidecar.
    from douzero.checkpoint import (
        CheckpointCompatibilityError,
        load_v2_position_weights,
    )

    try:
        state_dict, checkpoint_manifest = load_v2_position_weights(
            str(root / "weights.pt"),
            expected_schema_hash=manifest.feature_schema_hash,
            expected_model_config_hash=manifest.model_config_hash,
            expected_ruleset=packaged_ruleset,
            runtime_model_config=packaged_model_config,
            training_device="cpu",
        )
    except (CheckpointCompatibilityError, OSError, KeyError, TypeError, RuntimeError) as exc:
        raise ModelPackageError(
            f"weights.pt checkpoint identity does not match the package: {exc}"
        ) from exc
    if checkpoint_manifest.git_sha != manifest.git_sha:
        raise ModelPackageError(
            "weights.pt git_sha does not match the outer deployment manifest"
        )
    expected_inner_config = {
        "training_config_hash": manifest.training_config_hash,
        "training_config_payload_policy": "hash_only",
    }
    if checkpoint_manifest.effective_config != expected_inner_config:
        raise ModelPackageError(
            "weights.pt training configuration identity does not match the "
            "outer package hash-only policy"
        )
    floating_dtypes = {
        str(tensor.dtype).removeprefix("torch.")
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    }
    if floating_dtypes != {manifest.dtype}:
        raise ModelPackageError(
            f"weights dtype {sorted(floating_dtypes)} does not match manifest "
            f"dtype {manifest.dtype!r}"
        )
    if expected_schema_hash and expected_schema_hash != manifest.feature_schema_hash:
        raise ModelPackageError("runtime feature schema does not match packaged model")
    if expected_feature_version and expected_feature_version != manifest.feature_version:
        raise ModelPackageError("runtime feature_version does not match packaged model")
    if expected_model_config_hash and expected_model_config_hash != manifest.model_config_hash:
        raise ModelPackageError("runtime model config does not match packaged model")
    if (
        expected_belief_enabled is not None
        and expected_belief_enabled != manifest.belief_enabled
    ):
        raise ModelPackageError("runtime belief setting does not match packaged model")
    if (
        expected_bidding_enabled is not None
        and expected_bidding_enabled != manifest.bidding_enabled
    ):
        raise ModelPackageError("runtime bidding setting does not match packaged model")

    if check_package_versions:
        incompatible = {
            name: {"required": constraint, "installed": _installed_version(name)}
            for name, constraint in manifest.required_package_versions.items()
            if not _satisfies(_installed_version(name), constraint)
        }
        if incompatible:
            raise ModelPackageError(f"required package versions are not satisfied: {incompatible}")
    return manifest


def load_model_package(
    package_dir: str | Path,
    *,
    schema: "FeatureSchemaManifest",
    ruleset: "RuleSet",
    config: "ModelV2Config",
    device: str | torch.device = "cpu",
) -> "ModelV2":
    """Strictly load an eval-mode Model V2 and its packaged public belief model.

    Belief-disabled packages preserve the long-standing return contract: the
    result is a :class:`ModelV2`. A belief-enabled result is the same type and
    has its verified, eval-mode :class:`BeliefModel` attached as
    ``model.belief_model`` for direct ``DeepAgentV2`` construction.
    """

    from douzero.evaluation.deep_agent import load_v2_model

    manifest = verify_model_package(
        package_dir,
        expected_ruleset=ruleset,
        expected_schema_hash=schema.stable_hash(),
        expected_feature_version=schema.feature_version,
        expected_model_config_hash=config.stable_hash(),
        expected_belief_enabled=config.belief_enabled,
        expected_bidding_enabled=config.bidding_enabled,
    )
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ModelPackageError("CUDA device requested but CUDA is unavailable")
    root = Path(package_dir)
    model = load_v2_model(
        str(root / "weights.pt"), schema, ruleset, config
    )
    target_dtype = getattr(torch, manifest.dtype)
    model.to(device=target, dtype=target_dtype)
    model.eval()
    model.deployment_manifest = manifest
    if manifest.belief_enabled:
        try:
            belief_payload = json.loads(
                (root / "belief_config.json").read_text(encoding="utf-8")
            )
            belief_config = _parse_belief_config_payload(belief_payload)
            from douzero.belief.checkpoint import load_belief_checkpoint

            belief_model = load_belief_checkpoint(
                str(root / "belief_weights.pt"),
                expected_ruleset=ruleset,
                expected_feature_version=schema.feature_version,
                expected_belief_config=belief_config,
                map_location="cpu",
                require_full_git_sha=True,
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise ModelPackageError(
                f"failed to load packaged belief model: {exc}"
            ) from exc
        belief_model.to(device=target)
        belief_model.eval()
        belief_model.deployment_manifest = manifest
        model.belief_model = belief_model
    return model
