"""Regression tests for the CPU-only contract of the P04 factorized benchmark.

The benchmark (``benchmarks/bench_factorized.py``) claims to be CPU-only and
labels its output ``device: cpu``. That claim is only honest if the benchmark
UNCONDITIONALLY hides CUDA, even when the caller has preset
``CUDA_VISIBLE_DEVICES`` to a real GPU id. ``os.environ.setdefault`` would NOT
overwrite a preset value, so the benchmark previously could have silently
timed an asynchronous CUDA path (via ``DeepAgent``'s ``torch.cuda.is_available``
migration) while reporting ``device: cpu``.

These tests run the benchmark module's import and its ``_assert_cpu_only`` in a
FRESH child process with ``CUDA_VISIBLE_DEVICES=0`` preset, then assert the
child sees an empty value. A CPU-only torch build reports
``cuda.is_available()==False`` regardless of the env var, so the env-var
assertion is what actually pins the contract here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_child(snippet: str, env_extra: dict[str, str] | None = None):
    """Run a Python snippet in a fresh subprocess; return (returncode, stdout+stderr)."""
    import os
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    # Force unbuffered output so the child's prints/tracebacks are captured.
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return result.returncode, (result.stdout + result.stderr)


def test_benchmark_overrides_preset_cuda_visible_devices():
    """The benchmark must force CUDA_VISIBLE_DEVICES to '' even when preset to '0'.

    This is the exact failure mode setdefault could not catch: a caller running
    ``CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_factorized.py`` on a GPU
    host would leave DeepAgent on CUDA while the output says device: cpu.
    """
    snippet = textwrap.dedent(
        """
        import os
        # Importing the benchmark module runs its top-level env assignment.
        import benchmarks.bench_factorized  # noqa: F401
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "", (
            "benchmark did not override CUDA_VISIBLE_DEVICES; got "
            + repr(os.environ.get("CUDA_VISIBLE_DEVICES"))
        )
        print("CUDA_FORCE_HIDE_OK")
        """
    )
    rc, out = _run_child(snippet, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    assert rc == 0, f"child failed (rc={rc}):\n{out}"
    assert "CUDA_FORCE_HIDE_OK" in out, f"unexpected output:\n{out}"


def test_benchmark_assert_cpu_only_passes_when_cuda_hidden():
    """_assert_cpu_only must pass when CUDA is genuinely hidden.

    Imports the module (which sets the env var to '') and then calls the
    self-check, proving the self-check agrees with a hidden CUDA state.
    """
    snippet = textwrap.dedent(
        """
        import benchmarks.bench_factorized as bf  # top-level sets env to ''
        bf._assert_cpu_only()
        print("ASSERT_CPU_ONLY_OK")
        """
    )
    # Pass a preset the import must overwrite; after import the self-check sees ''.
    rc, out = _run_child(snippet, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    assert rc == 0, f"_assert_cpu_only raised (rc={rc}):\n{out}"
    assert "ASSERT_CPU_ONLY_OK" in out, f"unexpected output:\n{out}"


def test_benchmark_assert_cpu_only_rejects_post_import_override():
    """_assert_cpu_only must raise if CUDA_VISIBLE_DEVICES is set non-empty.

    The module's top-level assignment hides CUDA at import, but a caller that
    re-sets the env var AFTER import (or any path that leaves a non-empty value
    at check time) must be caught. We set the value non-empty AFTER importing
    so the import-time assignment is not re-run, then confirm the self-check
    raises. (On a CPU-only build the torch.cuda branch won't fire, so this
    specifically pins the env-var half of the contract.)
    """
    snippet = textwrap.dedent(
        """
        import os
        import benchmarks.bench_factorized as bf  # sets env to '' at import
        # Now clobber the value AFTER import to simulate a stale/external set.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            bf._assert_cpu_only()
        except RuntimeError as e:
            assert "CUDA_VISIBLE_DEVICES" in str(e), str(e)
            print("ASSERT_REJECTED_NONEMPTY_OK")
            raise SystemExit(0)
        raise SystemExit("expected RuntimeError from _assert_cpu_only")
        """
    )
    rc, out = _run_child(snippet, env_extra={"CUDA_VISIBLE_DEVICES": ""})
    assert "ASSERT_REJECTED_NONEMPTY_OK" in out, (
        f"_assert_cpu_only did not reject a non-empty CUDA_VISIBLE_DEVICES:\n{out}"
    )
