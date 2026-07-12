"""Regression tests for the Model V2 benchmark's DeepAgent path (P05).

Blocker #3 (review round): ``bench_model_v2.py`` constructed ``DeepAgentV2``
WITHOUT the required ``RuleSet`` argument and then swallowed the resulting
``TypeError`` in a broad ``except Exception``, writing ``{"error": ...}`` into
the report. The benchmark therefore exited 0 and rendered its JSON/Markdown
summary as if the DeepAgent latency had been measured — when in fact the
end-to-end ``act()`` path never ran. The reported ``12–55 ms/decision`` was a
model-forward number mislabelled as end-to-end.

These tests pin both halves of the fix:

- The benchmark is run end-to-end in a fresh subprocess (mirroring
  ``test_benchmark_cuda_contract.py``) so the REAL ``main()`` path is
  exercised, including the fail-closed contract: a failure on the core
  measurement path must exit non-zero and write no misleading latency, not be
  hidden in an ``error`` field.
- The report's ``deep_agent_act_ms`` must carry a real ``median_ms`` and no
  ``error`` key, and the Markdown must render the end-to-end section.
- The production source must construct ``DeepAgentV2`` with an explicit
  ``RuleSet`` (legacy, matching ``Env("adp")``), so the latency is a true
  end-to-end ``act()`` number.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "benchmarks" / "bench_model_v2.py"


def _run_bench(*args: str, timeout: int = 300) -> tuple[int, str]:
    """Run the V2 benchmark script; return (returncode, stdout+stderr)."""
    result = subprocess.run(
        [sys.executable, str(BENCH), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def test_bench_model_v2_deep_agent_path_runs(tmp_path):
    """Blocker #3: the DeepAgentV2 path actually runs (median_ms present, no
    error) and main() fails closed — exit 0 only when the path succeeds."""
    out = tmp_path / "bench_model_v2.json"
    rc, output = _run_bench(
        "--rounds", "2", "--warmup", "1", "--output", str(out),
    )
    assert rc == 0, f"benchmark exited {rc} (should fail closed only on error):\n{output}"
    assert out.exists(), f"benchmark wrote no JSON; output:\n{output}"

    report = json.loads(out.read_text())
    assert "deep_agent_act_ms" in report, (
        f"deep_agent_act_ms missing from report:\n{json.dumps(report, indent=2)}"
    )
    da = report["deep_agent_act_ms"]
    # The end-to-end path must produce a real measurement, not an error field.
    assert "error" not in da, (
        f"deep-agent path errored and was swallowed into a field:\n{da}"
    )
    for key in ("median_ms", "mean_ms", "p95_ms"):
        assert key in da, f"deep-agent result missing {key}:\n{da}"

    # The Markdown summary must have rendered the end-to-end section — it only
    # renders when deep_agent_act_ms has a median_ms, so this confirms the path
    # produced real latency data (not an error stub).
    md = out.with_suffix(".md")
    assert md.exists(), f"benchmark wrote no Markdown; output:\n{output}"
    assert "End-to-end DeepAgentV2.act" in md.read_text(), (
        "Markdown missing the end-to-end DeepAgentV2 section"
    )


def test_bench_model_v2_constructs_agent_with_explicit_ruleset():
    """Blocker #3 (contract pin): the benchmark must construct DeepAgentV2 with
    an explicit RuleSet (the 2-arg bare form is a regression). The functional
    test above catches the runtime failure; this pins the source contract."""
    src = BENCH.read_text()
    assert 'DeepAgentV2("landlord", model, RuleSet.legacy())' in src, (
        "benchmark must construct DeepAgentV2 with an explicit RuleSet.legacy() "
        "(the required ruleset argument), not the bare 2-arg form."
    )
