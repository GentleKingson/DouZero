"""Fail-closed deployment manifest for published DouZero models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping

from douzero._version import __version__, git_sha
from douzero.deployment.abi import MODEL_ABI_VERSION, model_implementation_hash
from douzero.models_v2.config import SUPPORTED_ROLES

if TYPE_CHECKING:
    from douzero.belief.model import BeliefConfig
    from douzero.env.rules import RuleSet
    from douzero.models_v2.model import ModelV2

# P17 adds audited model/training configuration identities, empirical-summary
# placeholders, and rollback instructions to every release directory.  Older
# P16 packages remain readable as files, but are intentionally not accepted by
# the stricter production verifier without being rebuilt.
CURRENT_MODEL_FORMAT_VERSION = 2
PUBLIC_MODEL = "public"
PRIVILEGED_MODEL = "privileged"
_ACCESS_CLASSES = frozenset({PUBLIC_MODEL, PRIVILEGED_MODEL})
_DTYPES = frozenset({"float32", "float16", "bfloat16"})


class ModelManifestError(ValueError):
    """Raised when deployment metadata is malformed or incompatible."""


def canonical_hash(value: Mapping[str, Any] | None) -> str:
    """Return a stable SHA-256 for a JSON-compatible mapping."""

    payload = json.dumps(value or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    """Complete identity and capability contract for one deployment model."""

    format_version: int
    model_abi_version: str
    implementation_hash: str
    model_version: str
    feature_version: str
    feature_schema_hash: str
    model_config_hash: str
    ruleset_id: str
    ruleset_hash: str
    git_sha: str
    training_config_hash: str
    belief_config_hash: str
    role_support: tuple[str, ...]
    belief_enabled: bool
    bidding_enabled: bool
    bidding_head_version: str
    bidding_action_schema: str
    bidding_feature_schema_hash: str
    search_compatible: bool
    public_or_privileged: str
    dtype: str
    required_package_versions: dict[str, str]
    source_checkpoint_sha256: str
    source_checkpoint_manifest_sha256: str
    source_git_sha: str
    source_training_config_hash: str
    source_ruleset_hash: str
    source_feature_schema_hash: str
    source_model_config_hash: str
    source_checkpoint_kind: str
    release_eligible: bool
    weights_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format_version != CURRENT_MODEL_FORMAT_VERSION:
            raise ModelManifestError(
                f"unsupported format_version {self.format_version!r}; expected "
                f"{CURRENT_MODEL_FORMAT_VERSION}"
            )
        for name in (
            "model_version",
            "model_abi_version",
            "implementation_hash",
            "feature_version",
            "feature_schema_hash",
            "model_config_hash",
            "ruleset_id",
            "ruleset_hash",
            "git_sha",
            "training_config_hash",
            "belief_config_hash",
            "source_checkpoint_sha256",
            "source_checkpoint_manifest_sha256",
            "source_git_sha",
            "source_training_config_hash",
            "source_ruleset_hash",
            "source_feature_schema_hash",
            "source_model_config_hash",
            "source_checkpoint_kind",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ModelManifestError(f"{name} must be a non-empty string")
        for name in (
            "implementation_hash",
            "feature_schema_hash",
            "model_config_hash",
            "ruleset_hash",
            "training_config_hash",
            "belief_config_hash",
            "source_checkpoint_sha256",
            "source_checkpoint_manifest_sha256",
            "source_training_config_hash",
            "source_ruleset_hash",
            "source_feature_schema_hash",
            "source_model_config_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ModelManifestError(f"{name} must be a lowercase SHA-256")
        if len(self.git_sha) not in (40, 64) or any(
            c not in "0123456789abcdef" for c in self.git_sha
        ):
            raise ModelManifestError("git_sha must be a full lowercase Git object ID")
        if len(self.source_git_sha) not in (40, 64) or any(
            c not in "0123456789abcdef" for c in self.source_git_sha
        ):
            raise ModelManifestError("source_git_sha must be a full lowercase Git object ID")
        if not isinstance(self.release_eligible, bool):
            raise ModelManifestError("release_eligible must be bool")
        if self.release_eligible and self.source_checkpoint_kind != "public_policy":
            raise ModelManifestError(
                "release-eligible packages require a public_policy source checkpoint"
            )
        if not self.role_support or len(set(self.role_support)) != len(self.role_support):
            raise ModelManifestError("role_support must contain unique supported roles")
        unknown_roles = set(self.role_support) - set(SUPPORTED_ROLES)
        if unknown_roles:
            raise ModelManifestError(f"unknown role_support entries: {sorted(unknown_roles)}")
        if self.public_or_privileged not in _ACCESS_CLASSES:
            raise ModelManifestError(
                "public_or_privileged must be 'public' or 'privileged'"
            )
        if self.dtype not in _DTYPES:
            raise ModelManifestError(f"unsupported dtype {self.dtype!r}")
        if (
            not isinstance(self.belief_enabled, bool)
            or not isinstance(self.bidding_enabled, bool)
            or not isinstance(self.search_compatible, bool)
        ):
            raise ModelManifestError(
                "belief_enabled, bidding_enabled, and search_compatible must be booleans"
            )
        bidding_identity = (
            self.bidding_head_version,
            self.bidding_action_schema,
            self.bidding_feature_schema_hash,
        )
        if self.bidding_enabled:
            if not all(isinstance(value, str) and value for value in bidding_identity):
                raise ModelManifestError(
                    "bidding-enabled manifests require complete bidding identity"
                )
            if len(self.bidding_feature_schema_hash) != 64 or any(
                char not in "0123456789abcdef"
                for char in self.bidding_feature_schema_hash
            ):
                raise ModelManifestError(
                    "bidding_feature_schema_hash must be a lowercase SHA-256"
                )
        elif any(bidding_identity):
            raise ModelManifestError(
                "bidding-disabled manifests must not carry bidding identity"
            )
        if not isinstance(self.required_package_versions, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) and v
            for k, v in self.required_package_versions.items()
        ):
            raise ModelManifestError(
                "required_package_versions must map package names to constraints"
            )
        if self.weights_sha256 and (
            len(self.weights_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.weights_sha256)
        ):
            raise ModelManifestError("weights_sha256 must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["role_support"] = list(self.role_support)
        result["required_package_versions"] = dict(self.required_package_versions)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        """Parse a manifest, rejecting missing and unknown fields."""

        if not isinstance(value, Mapping):
            raise ModelManifestError("model manifest must be a mapping")
        expected = {field.name for field in fields(cls)}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ModelManifestError(
                f"model manifest field mismatch; missing={missing}, unknown={unknown}"
            )
        raw = dict(value)
        roles = raw["role_support"]
        if not isinstance(roles, (list, tuple)):
            raise ModelManifestError("role_support must be a list")
        raw["role_support"] = tuple(roles)
        if not isinstance(raw["required_package_versions"], Mapping):
            raise ModelManifestError("required_package_versions must be a mapping")
        raw["required_package_versions"] = dict(raw["required_package_versions"])
        return cls(**raw)


def build_model_manifest(
    model: "ModelV2",
    ruleset: "RuleSet",
    *,
    training_config: Mapping[str, Any] | None = None,
    belief_config: "BeliefConfig | None" = None,
    role_support: tuple[str, ...] = SUPPORTED_ROLES,
    search_compatible: bool = False,
    public_or_privileged: str = PUBLIC_MODEL,
    dtype: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> ModelManifest:
    """Build deployment metadata from the actual model and ruleset identity."""

    from douzero.env.rules import RuleSet
    from douzero.models_v2.model import ModelV2

    if not isinstance(model, ModelV2):
        raise TypeError(f"model must be ModelV2, got {type(model).__name__}")
    if not isinstance(ruleset, RuleSet):
        raise TypeError(f"ruleset must be RuleSet, got {type(ruleset).__name__}")
    source_sha = git_sha()
    if source_sha == "unknown":
        raise ModelManifestError(
            "release packages require a known git_sha; build from a Git checkout "
            "or set DOUZERO_GIT_SHA to the exact source commit"
        )
    if (
        public_or_privileged == PUBLIC_MODEL
        and getattr(model, "model_access", PUBLIC_MODEL) != PUBLIC_MODEL
    ):
        raise ModelManifestError("a privileged model cannot be labelled as public")
    if dtype is None:
        try:
            dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        except StopIteration:
            dtype = "float32"
    belief_enabled = bool(model.config.belief_enabled)
    if belief_enabled:
        from douzero.belief.model import BeliefConfig

        if not isinstance(belief_config, BeliefConfig):
            raise TypeError(
                "belief-enabled manifests require an actual BeliefConfig "
                "identity"
            )
        belief_config_hash = belief_config.stable_hash()
    else:
        if belief_config is not None:
            raise TypeError(
                "belief_config was supplied for a belief-disabled model"
            )
        belief_config_hash = canonical_hash(None)

    bidding_enabled = bool(getattr(model.config, "bidding_enabled", False))
    if bidding_enabled:
        if ruleset.ruleset_id != "standard":
            raise ModelManifestError(
                "learned-bidding packages require the standard ruleset"
            )
        from douzero.observation.bidding import (
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
            build_bidding_schema,
        )

        bidding_head_version = BIDDING_HEAD_VERSION
        bidding_action_schema = BIDDING_ACTION_SCHEMA_VERSION
        bidding_feature_schema_hash = build_bidding_schema().stable_hash()
    else:
        bidding_head_version = ""
        bidding_action_schema = ""
        bidding_feature_schema_hash = ""
    if source_provenance is None:
        # Direct in-memory export remains useful for smoke tests and migration,
        # but it cannot manufacture training provenance.
        source_provenance = {
            "source_checkpoint_sha256": canonical_hash(None),
            "source_checkpoint_manifest_sha256": canonical_hash(None),
            "source_git_sha": source_sha,
            "source_training_config_hash": canonical_hash(None),
            "source_ruleset_hash": ruleset.stable_hash(),
            "source_feature_schema_hash": model.schema.stable_hash(),
            "source_model_config_hash": model.config.stable_hash(),
            "source_checkpoint_kind": "migration_artifact",
            "release_eligible": False,
        }
    required_source = {
        "source_checkpoint_sha256", "source_checkpoint_manifest_sha256",
        "source_git_sha", "source_training_config_hash", "source_ruleset_hash",
        "source_feature_schema_hash", "source_model_config_hash",
        "source_checkpoint_kind", "release_eligible",
    }
    if set(source_provenance) != required_source:
        raise ModelManifestError("source checkpoint provenance has an invalid field set")
    if (
        source_provenance["source_ruleset_hash"] != ruleset.stable_hash()
        or source_provenance["source_feature_schema_hash"] != model.schema.stable_hash()
        or source_provenance["source_model_config_hash"] != model.config.stable_hash()
    ):
        raise ModelManifestError("source checkpoint identity does not match the model")
    return ModelManifest(
        format_version=CURRENT_MODEL_FORMAT_VERSION,
        model_abi_version=MODEL_ABI_VERSION,
        implementation_hash=model_implementation_hash(),
        model_version="v2",
        feature_version=model.schema.feature_version,
        feature_schema_hash=model.schema.stable_hash(),
        model_config_hash=model.config.stable_hash(),
        ruleset_id=ruleset.ruleset_id,
        ruleset_hash=ruleset.stable_hash(),
        git_sha=source_sha,
        training_config_hash=str(source_provenance["source_training_config_hash"]),
        belief_config_hash=belief_config_hash,
        role_support=tuple(role_support),
        belief_enabled=belief_enabled,
        bidding_enabled=bidding_enabled,
        bidding_head_version=bidding_head_version,
        bidding_action_schema=bidding_action_schema,
        bidding_feature_schema_hash=bidding_feature_schema_hash,
        search_compatible=bool(search_compatible),
        public_or_privileged=public_or_privileged,
        dtype=dtype,
        required_package_versions={
            "python": ">=3.11",
            "numpy": ">=1.24",
            "torch": ">=2.0",
            "douzero": f"=={__version__}",
        },
        **dict(source_provenance),
    )
