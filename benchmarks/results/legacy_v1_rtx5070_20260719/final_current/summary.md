# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_a1_cpu_factorized | 3 | 16642.130 | 16683.441 | 16804.843 | 5.201 | 3.000 | 1208.798 | 1280.000 | 9104.805 | 15.255 |
| legacy_a2_cpu_factorized_amp | 3 | 16717.693 | 16751.548 | 16764.338 | 5.224 | 3.000 | 1207.636 | 1276.000 | 9400.668 | 15.295 |
| legacy_a2_cpu_factorized_bf16 | 3 | 16693.436 | 16706.571 | 16878.357 | 5.217 | 0.000 | 1207.619 | 1106.000 | 9443.004 | 15.105 |
| legacy_a3_cpu_factorized_compile | 3 | 16640.819 | 16677.509 | 16773.370 | 5.200 | 6.000 | 1207.421 | 1308.000 | 9188.414 | 15.235 |
| legacy_a4_cpu_factorized_compile_bf16 | 3 | 16602.923 | 16660.625 | 16788.307 | 5.188 | 1.000 | 1207.376 | 1088.000 | 9529.137 | 15.490 |
