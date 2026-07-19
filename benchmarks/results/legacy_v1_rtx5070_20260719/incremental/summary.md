# Legacy V1 Single-GPU Training Benchmark

Measured values are produced by complete actor/learner runs. No row is a theoretical estimate.

| configuration | repeats | frames/s median | frames/s p95 | decisions/s median | updates/s median | GPU median | CPU median | VRAM max MiB | RSS max MiB | policy lag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_a0_log | 3 | 10232.362 | 10356.953 | 10750.375 | 3.198 | 0.000 | 1206.687 | 1330.000 | 8904.738 | 3.737 |
| legacy_a0_snapshot | 3 | 10165.058 | 14854.969 | 11033.853 | 3.177 | 0.000 | 1208.106 | 1338.000 | 8711.812 | 0.000 |
| legacy_a0_factorized | 3 | 14539.205 | 14596.457 | 11958.669 | 4.544 | 0.500 | 1210.019 | 1298.000 | 8453.676 | 0.000 |
| legacy_a1_cpu_factorized | 3 | 15487.256 | 16513.735 | 11571.410 | 4.840 | 8.500 | 1209.720 | 1280.000 | 8423.234 | 0.000 |
| legacy_a2_cpu_factorized_amp | 3 | 16543.568 | 16883.397 | 11370.667 | 5.170 | 4.000 | n/a | 1404.000 | 8728.785 | 0.000 |
| legacy_a3_cpu_factorized_compile | 3 | 18216.061 | 18279.819 | 11401.120 | 5.693 | 12.000 | n/a | 1308.000 | 8519.793 | 0.000 |
