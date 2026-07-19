# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_a0_cpu_actor_thread1 | 3 | 9963.974 | 9982.509 | 10082.187 | 3.114 | 0.000 | 1205.458 | 1298.000 | 9586.969 | 2.718 |
| legacy_a0_log | 3 | 9941.275 | 9966.849 | 10134.743 | 3.107 | 0.000 | 1206.090 | 1454.000 | 9587.801 | 2.685 |
| legacy_a0_snapshot | 3 | 10524.305 | 10537.827 | 11049.973 | 3.289 | 0.000 | 1205.095 | 1380.000 | 9548.293 | 15.560 |
| legacy_a0_factorized | 3 | 16341.827 | 16371.554 | 16487.006 | 5.107 | 3.000 | 1208.840 | 1278.000 | 9110.645 | 15.445 |
| legacy_a1_cpu_factorized | 3 | 16668.971 | 16677.885 | 16853.521 | 5.209 | 3.000 | 1208.371 | 1300.000 | 9103.582 | 15.365 |
