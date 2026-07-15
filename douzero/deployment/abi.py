"""Versioned hash of deployment code that defines Model V2 semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_ABI_VERSION = "model-v2-deployment-1"

# These modules define public observation construction, Model V2 forward
# semantics, optional public feature paths, action selection, search, and strict
# checkpoint loading. Documentation and training-only data pipelines are
# intentionally excluded so non-semantic edits do not invalidate a package.
_ABI_PATHS = (
    "belief",
    "checkpoint",
    "evaluation/deep_agent.py",
    "models_v2",
    "observation",
    "search",
    "strategy",
    "style",
    "training/decision_policy.py",
)


def model_implementation_hash() -> str:
    """Hash the installed Python sources that define deployment semantics."""

    package_root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    for relative in _ABI_PATHS:
        path = package_root / relative
        if path.is_dir():
            sources.extend(sorted(path.rglob("*.py")))
        elif path.is_file():
            sources.append(path)
        else:
            raise RuntimeError(f"model ABI source is missing: {path}")
    digest = hashlib.sha256()
    ordered_sources = sorted(
        set(sources),
        key=lambda item: item.relative_to(package_root).as_posix(),
    )
    for path in ordered_sources:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
