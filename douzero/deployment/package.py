"""Create, verify, and load self-contained Model V2 release directories."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
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
)

if TYPE_CHECKING:
    from douzero.env.rules import RuleSet
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import FeatureSchemaManifest

_REQUIRED_FILES = frozenset({
    "weights.pt",
    "manifest.json",
    "ruleset.json",
    "feature_schema.json",
    "README.md",
    "THIRD_PARTY_NOTICES",
    "SHA256SUMS",
})


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
    return f"""# DouZero Model Card

## Model Details

- Model version: `{manifest.model_version}`
- Model ABI: `{manifest.model_abi_version}`
- Implementation hash: `{manifest.implementation_hash}`
- Source Git SHA: `{manifest.git_sha}`
- Feature version: `{manifest.feature_version}`
- Ruleset: `{manifest.ruleset_id}`
- Roles: `{', '.join(manifest.role_support)}`
- Belief model enabled: `{str(manifest.belief_enabled).lower()}`
- Search compatible: `{str(manifest.search_compatible).lower()}`
- Numeric dtype: `{manifest.dtype}`

## Training Data

Record the authorized training data categories and provenance before release.
Raw personal identifiers must not be included in this package.

## Evaluation

Record paired, seat-rotated metrics and confidence intervals for every
supported role and opponent. Metrics are **not measured** by the packaging
command.

## Latency

Not measured. Benchmark the packaged model on the target hardware before deployment.

## Known Limitations

This research model may fail outside the declared ruleset and feature schema.
It is not suitable for platform automation, account operation, scraping,
anti-detection, or decisions involving undisclosed hidden information.

## License

Apache-2.0. See `THIRD_PARTY_NOTICES` for dependency and reference-source attribution.
"""


def create_model_package(
    output_dir: str | Path,
    model: "ModelV2",
    ruleset: "RuleSet",
    *,
    training_config: Mapping[str, Any] | None = None,
    search_compatible: bool = False,
    model_card: str | None = None,
) -> ModelManifest:
    """Write a checksummed public Model V2 package.

    The destination must be absent or empty; existing release contents are
    never overwritten silently.
    """

    from douzero.checkpoint import save_v2_position_weights

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ModelPackageError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    manifest = build_model_manifest(
        model,
        ruleset,
        training_config=training_config,
        search_compatible=search_compatible,
        public_or_privileged=PUBLIC_MODEL,
    )
    weights_path = root / "weights.pt"
    save_v2_position_weights(
        str(weights_path), model, ruleset=ruleset, flags=training_config or {}
    )
    manifest = replace(manifest, weights_sha256=_sha256(weights_path))
    _write_json(root / "manifest.json", manifest.to_dict())
    _write_json(root / "ruleset.json", ruleset.to_dict())
    _write_json(root / "feature_schema.json", model.schema.to_dict())
    (root / "README.md").write_text(
        model_card or _default_model_card(manifest), encoding="utf-8"
    )

    notices = Path(__file__).with_name("THIRD_PARTY_NOTICES")
    if not notices.is_file():
        raise ModelPackageError(f"release audit file is missing: {notices}")
    shutil.copyfile(notices, root / "THIRD_PARTY_NOTICES")

    checksummed = sorted(_REQUIRED_FILES - {"SHA256SUMS"})
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
    allow_privileged: bool = False,
    check_package_versions: bool = True,
) -> ModelManifest:
    """Verify files, checksums, access class, versions, and runtime identity."""

    root = Path(package_dir)
    if not root.is_dir():
        raise ModelPackageError(f"model package is not a directory: {root}")
    missing = sorted(name for name in _REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise ModelPackageError(f"model package is missing files: {missing}")
    if any((root / name).is_symlink() for name in _REQUIRED_FILES):
        raise ModelPackageError("model package required files must not be symlinks")

    try:
        manifest = ModelManifest.from_dict(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ModelManifestError) as exc:
        raise ModelPackageError(f"invalid manifest.json: {exc}") from exc
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
        if len(parts) != 2 or parts[1] not in _REQUIRED_FILES - {"SHA256SUMS"}:
            raise ModelPackageError(f"malformed SHA256SUMS entry: {line!r}")
        expected_lines[parts[1]] = parts[0]
    expected_names = _REQUIRED_FILES - {"SHA256SUMS"}
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
    """Strictly verify a public package and return its eval-mode Model V2."""

    from douzero.evaluation.deep_agent import load_v2_model

    manifest = verify_model_package(
        package_dir,
        expected_ruleset=ruleset,
        expected_schema_hash=schema.stable_hash(),
        expected_feature_version=schema.feature_version,
        expected_model_config_hash=config.stable_hash(),
        expected_belief_enabled=config.belief_enabled,
    )
    model = load_v2_model(
        str(Path(package_dir) / "weights.pt"), schema, ruleset, config
    )
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ModelPackageError("CUDA device requested but CUDA is unavailable")
    target_dtype = getattr(torch, manifest.dtype)
    model.to(device=target, dtype=target_dtype)
    model.eval()
    model.deployment_manifest = manifest
    return model
