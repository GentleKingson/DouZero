"""Runtime version metadata for the DouZero project.

This module is intentionally dependency-free so it can be imported by tooling
(checkpoint manifests, baseline capture, documentation) without pulling in
torch/numpy. It records the project version and the best-effort git revision
at import time.

P00 scope note: this file only *adds* metadata; it does not change any
algorithm, observation, model, or CLI behavior.
"""

from __future__ import annotations

import os
import subprocess

__all__ = ["__version__", "git_sha", "environment_info"]

# Mirrors setup.py. Kept here as the single source of truth so that future
# phases can import it without importing setuptools at runtime.
__version__ = "1.1.0"

# Sentinel used to distinguish "git_sha() has not been called yet" from a
# cached (possibly 'unknown') result. None cannot serve that role because the
# function now always returns a string.
_SENTINEL = object()


def _run_git(args: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if git is unavailable.

    The value is cached after the first call (success or failure). Reading from
    the DOUZERO_GIT_SHA environment variable takes precedence (useful in frozen
    containers / CI where the .git directory is absent). On any failure (no git
    binary, not a repo, git not installed) the function returns the explicit
    string 'unknown' rather than None, so manifest metadata is always a string
    and never a silent null.
    """
    cached = getattr(git_sha, "_cached", _SENTINEL)
    if cached is not _SENTINEL:
        return cached  # type: ignore[return-value]

    env_override = os.environ.get("DOUZERO_GIT_SHA")
    if env_override:
        git_sha._cached = env_override  # type: ignore[attr-defined]
        return env_override

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    result = sha if sha is not None else "unknown"
    git_sha._cached = result  # type: ignore[attr-defined]
    return result


def environment_info() -> dict:
    """Collect a lightweight, JSON-serializable environment summary.

    Imports of optional dependencies (torch, numpy, rlcard) happen lazily so
    that importing this module never fails in a stripped-down environment.
    """
    import platform

    info: dict = {
        "project": "douzero",
        "version": __version__,
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "torch", "rlcard"):
        try:
            mod = __import__(name)
            # str() coerces non-native str subclasses (e.g. torch's TorchVersion)
            # so the value pickles cleanly under weights_only=True. Without this,
            # a TorchVersion object in the manifest triggers an "Unsupported
            # global" error on safe checkpoint loads.
            info[name + "_version"] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - best-effort metadata
            info[name + "_version"] = None
    # CUDA availability only when torch is present.
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - best-effort metadata
        info["cuda_available"] = None
    return info
