# P14 CPU Reference Benchmark

Measured on 2026-07-15 with Python 3.14.6, PyTorch 2.13.0, and macOS arm64.
Command:

```bash
python benchmarks/bench_training_system.py --rounds 5
```

| Path | Median ms | p95 ms |
|---|---:|---:|
| actor environment step | 0.1661 | 0.1880 |
| observation encoding | 0.2227 | 0.3969 |
| queue wait | 0.0028 | 0.0076 |
| learner forward/backward step | 11.6355 | 12.6849 |
| versioned weight publication | 5.3629 | 10.6779 |
| legacy fp32 forward | 2.5192 | 2.8077 |
| factorized fp32 forward | 0.5312 | 1.2628 |
| V2 fp32 forward | 5.9499 | 14.6665 |
| V2 CPU bfloat16 forward | 42.1036 | 68.9163 |

This run does not support enabling CPU bfloat16 by default: it was materially
slower than fp32 on this host. CUDA AMP was not measured because no CUDA device
was available. Two-process CPU/gloo DDP was validated separately as a smoke
test, not presented as a throughput result. The machine-readable local report
is written to `artifacts/benchmark/p14_training_system.json`; `artifacts/` is
intentionally gitignored.
