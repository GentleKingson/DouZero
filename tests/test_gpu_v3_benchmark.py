import pytest
import torch

from benchmarks.bench_gpu_v3 import run


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_v3_benchmark_smoke():
    report = run(rounds=2, warmup=1)
    assert report["schema_version"] == "gpu-v3-capability-v1"
    assert set(report["results"]) == {
        "independent_role_dual_tower",
        "shared_trunk_role_heads",
    }
    for model in report["results"].values():
        assert model["parameters"] > 0
        assert all(
            case["median_ms"] > 0 and case["decisions_per_second"] > 0
            for case in model["cases"].values()
        )
