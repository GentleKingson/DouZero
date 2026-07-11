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


def git_sha() -> str | None:
    """Return the current git commit SHA, or None if git is unavailable.

    The value is cached after the first successful call. Reading from the
    DOUZERO_GIT_SHA environment variable takes precedence (useful in frozen
    containers where the .git directory is absent).
    """
    cached = getattr(git_sha, "_cached", None)
    if cached is not None:
        return cached

    env_override = os.environ.get("DOUZERO_GIT_SHA")
    if env_override:
        git_sha._cached = env_override  # type: ignore[attr-defined]
        return env_override

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    git_sha._cached = sha  # type: ignore[attr-defined]
    return sha


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
            info[name + "_version"] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - best-effort metadata
            info[name + "_version"] = None
    # CUDA availability only when torch is present.
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - best-effort metadata
        info["cuda_available"] = None
    return info
