# Standard V2 M1 manual GPU evidence

This bundle was produced on `LocalServer:/opt/DouZero` from source commit
`408c97a25088451d884d9836ce6f6fbf32d810c6`. Every dependency install, test,
CUDA check, training run, and benchmark ran inside `douzero-test:latest`; no
GitHub GPU workflow was used.

The Docker image ID is
`sha256:417ae0638a9ffa1cc1ba41c2a23e9a20074866b79a7b7a08e6efe1b4221ac2cb`.
The environment report records PyTorch 2.12.1+cu132, CUDA 13.2, driver
595.71.05, and one NVIDIA GeForce RTX 5070.

## Results

- The complete CPU suite passed in Docker.
- Five explicit GPU tests passed, including recoverable CUDA validation errors,
  AMP non-fallback, standard checkpoint resume, eligible cadence resume, and
  async checkpoint compatibility.
- B=32 bidding head forward/backward measured 1.055507 ms mean and 1.198880 ms
  p95, below the 1.5 ms gate.
- The repeated frozen 16-game R1 run observed a parameter update and measured
  8.009856 games/s, 384.473094 play transitions/s, 22.027104 bid
  transitions/s, 261.953683 learner samples/s, and 266.845 MiB peak allocated
  VRAM.

The checked-in canonical baseline remains
`benchmarks/baselines/standard_v2_r1_single_gpu.json` with SHA-256
`c1513aa2a828c2832e1a980a00fdec6673c8d3e3568a792cd602224f1c30d23d`.
This bundle's additional repeat has SHA-256
`58ea23ac02395835910d2ee2e119a23461a8f296782834e39c73784ebf766a87`.
Timing variance is retained rather than replacing the canonical run.

`SHA256SUMS` binds every evidence file other than itself. The eventual
evidence-only commit changes documentation and artifacts, not the runtime code
represented by `source-git-sha.txt` and the Docker image.
