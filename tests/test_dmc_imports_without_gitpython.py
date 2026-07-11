"""Verify the lazy-import contract: douzero.dmc must import without GitPython.

Before P01, ``douzero/dmc/file_writer.py`` did ``import git`` at module load,
which transitively made ``import douzero.dmc`` (and ``train.py --help``, and the
P00 tests that do ``from douzero.dmc.models import ...``) require GitPython even
though GitPython is only needed for run-metadata stamping during training.

These tests pin the fix: even if the ``git`` Python module is unavailable,
importing ``douzero.dmc`` / ``douzero.dmc.models`` must not raise, and
``gather_metadata`` must degrade to ``git=None`` instead of crashing.
"""

from __future__ import annotations

import importlib
import sys


def _reload_dmc_without_git():
    """Force re-import of douzero.dmc / file_writer with `git` unavailable.

    Setting sys.modules['git'] = None tells Python "this module is not
    importable"; a subsequent `import git` raises ImportError. We remove the
    already-loaded douzero.dmc submodules so they re-execute their top-level
    code under the no-git condition.
    """
    # Pretend GitPython is absent. None as a sys.modules value makes
    # `import git` raise ImportError.
    saved_git = sys.modules.pop("git", None)
    sys.modules["git"] = None

    # Evict the douzero.dmc submodules that import (or transitively load)
    # file_writer, so they re-run their module bodies.
    for mod in list(sys.modules):
        if mod == "douzero.dmc" or mod.startswith("douzero.dmc."):
            sys.modules.pop(mod, None)

    try:
        importlib.import_module("douzero.dmc")
        importlib.import_module("douzero.dmc.models")
        importlib.import_module("douzero.dmc.file_writer")
        return sys.modules["douzero.dmc.file_writer"]
    finally:
        # Restore the real git module so subsequent tests are unaffected.
        sys.modules.pop("git", None)
        if saved_git is not None:
            sys.modules["git"] = saved_git
        # Re-import douzero.dmc normally for the rest of the suite.
        for mod in list(sys.modules):
            if mod == "douzero.dmc" or mod.startswith("douzero.dmc."):
                sys.modules.pop(mod, None)
        importlib.import_module("douzero.dmc")


def test_dmc_imports_without_gitpython():
    """import douzero.dmc / .models must succeed when GitPython is absent."""
    # If this raises, the lazy-import contract is broken.
    fw = _reload_dmc_without_git()
    assert fw is not None
    # And the real import still works afterward.
    import douzero.dmc  # noqa: F401
    import douzero.dmc.models  # noqa: F401


def test_gather_metadata_records_error_when_git_unavailable(monkeypatch):
    """gather_metadata must not raise when `import git` fails.

    When GitPython is unavailable, the git metadata degrades to a dict carrying
    an error_type (rather than None or a crash), so the failure is recorded
    instead of silently swallowed.
    """
    import douzero.dmc.file_writer as fw

    # Force `import git` inside gather_metadata to raise ImportError.
    monkeypatch.setitem(sys.modules, "git", None)
    meta = fw.gather_metadata()
    assert isinstance(meta["git"], dict)
    assert meta["git"].get("commit") is None
    assert "error_type" in meta["git"]
    assert meta["git"]["error_type"] == "GitPythonNotInstalled"
    # The other metadata fields are still populated.
    assert "date_start" in meta
    assert meta["successful"] is False


def test_gather_metadata_succeeds_when_git_available():
    """Sanity: with GitPython present (the test env has it), git is populated.

    This guards against the lazy import breaking the normal path. If GitPython
    is not installed in the running environment, this test is skipped.
    """
    import importlib.util

    if importlib.util.find_spec("git") is None:
        import pytest

        pytest.skip("GitPython not installed in this environment")
    import douzero.dmc.file_writer as fw

    meta = fw.gather_metadata()
    # git may still be None if not run inside a git repo, but it must not raise.
    assert "git" in meta
