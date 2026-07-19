# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_a1_cpu_factorized | 3 | 15718.974 | 16476.984 | 15839.143 | 4.912 | 3.000 | 1207.961 | 1300.000 | 8780.098 | 0.000 |
| legacy_a2_cpu_factorized_amp | 3 | 14546.986 | 14884.662 | 15368.253 | 4.546 | 0.000 | 1206.548 | 1300.000 | 9080.199 | 0.000 |
| legacy_a3_cpu_factorized_compile | 3 | 16206.745 | 16553.798 | 16701.599 | 5.065 | 5.500 | 1207.175 | 1368.000 | 8878.285 | 0.000 |
