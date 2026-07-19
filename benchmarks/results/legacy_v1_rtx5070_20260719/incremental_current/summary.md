# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_a0_cpu_actor_thread1 | 3 | 9379.558 | 9400.397 | 8836.883 | 2.931 | 0.000 | 1204.472 | 1296.000 | 9198.102 | 3.984 |
| legacy_a0_log | 3 | 9409.440 | 9437.804 | 8913.614 | 2.940 | 0.000 | 1205.424 | 1298.000 | 9190.668 | 3.886 |
| legacy_a0_snapshot | 3 | 9476.297 | 9494.411 | 9312.154 | 2.961 | 0.000 | 1203.139 | 1278.000 | 8974.465 | 0.000 |
| legacy_a0_factorized | 3 | 14297.091 | 14481.517 | 11424.236 | 4.468 | 3.500 | 1207.073 | 1298.000 | 8747.516 | 0.000 |
| legacy_a1_cpu_factorized | 3 | 10486.122 | 13499.486 | 12904.893 | 3.277 | 0.000 | 1204.031 | 1278.000 | 8744.328 | 0.000 |
