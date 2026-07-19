# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_b1_gpu_factorized | 3 | 3140.416 | 3142.838 | 3138.478 | 0.981 | 98.000 | 401.064 | 2683.000 | 5337.812 | 12.925 |
| legacy_c0_centralized_gpu_actor | 3 | 6627.006 | 6734.069 | 6578.092 | 2.071 | 26.000 | 292.013 | 1677.000 | 9865.805 | 15.205 |
