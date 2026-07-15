"""Contract tests for the P14 measured profiler report."""

from benchmarks.bench_training_system import build_report


def test_training_system_benchmark_reports_all_required_components():
    report = build_report(rounds=1)
    assert report["schema_version"] == "p14-training-system-v1"
    assert set(report["profiler_ms"]) == {
        "actor_env_step",
        "observation_encoding",
        "queue_wait",
        "learner_forward_backward_step",
        "weight_sync",
    }
    assert set(report["forward_comparison_ms"]) == {
        "legacy_fp32",
        "factorized_fp32",
        "v2_fp32",
        "v2_cpu_bfloat16_amp",
    }
    for group in ("profiler_ms", "forward_comparison_ms"):
        for measurement in report[group].values():
            assert measurement["rounds"] == 1
            assert measurement["median_ms"] >= 0
    assert report["ddp"]["measured"] is False
    assert report["cuda_amp"]["measured"] is False
