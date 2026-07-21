# Legacy A1 formal validation, 2026-07-21

## Frozen environment

- main: `388870f9a4c40a093fd86fe0d8de2da821e903f6`
- A1 candidate: `34b36cf0fbf27adfde527d1045e18ba34d200795`
- merge-base: `388870f9a4c40a093fd86fe0d8de2da821e903f6`
- image: `sha256:6f57b50161e8a4c4147fda854b76b255bdf643a2703c7da464b89277da01f953`
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB, driver 595.71.05
- CPU: Intel Core Ultra 5 245KF
- Python/PyTorch/CUDA: 3.12.3 / 2.12.1+cu132 / 13.2
- config SHA-256: `4e7623449d669527ef6b545e64a935023a53bf3301d002d8aa112d7357e82fb3`

Both source checkouts were clean. Each run used the same host, immutable image,
seed, A1 benchmark configuration, 64,000 warmup frames, 128,000 measured
frames, and enabled checkpoint output. The benchmark failed closed above a
maximum policy lag of 128 learner updates.

## Same-host comparison

| source | repeat | frames/s | decisions/s | updates/s | max policy lag |
|---|---:|---:|---:|---:|---:|
| main | 1 | 15,455.568 | 16,958.380 | 4.830 | 20 |
| main | 2 | 15,475.522 | 17,049.794 | 4.836 | 20 |
| main | 3 | 15,488.405 | 16,543.068 | 4.840 | 20 |
| A1 candidate | 1 | 14,273.276 | 15,740.302 | 4.460 | 20 |
| A1 candidate | 2 | 14,242.937 | 15,691.043 | 4.451 | 20 |
| A1 candidate | 3 | 14,241.909 | 15,515.448 | 4.451 | 20 |

The main median was 15,475.522 frames/s and the candidate median was
14,242.937 frames/s. The candidate is 7.97% slower. Relative ranges were
0.21% and 0.22%, respectively, so the result is not explained by run-to-run
dispersion. All six final checkpoints existed and were hashed independently.
The summary artifact SHA-256 values were
`c880c89bb54f645040ccc392ccc129c3eb76249dc14a714c1913753d101acb27`
for main and
`268b7910571e2a51c9d1fcd97d61ce519e0c4f1965f8c793d0d6b025f23ddecd`
for the candidate.

## Checkpoint-enabled soak and resume

The production A1 topology completed four consecutive checkpoint-enabled
segments from the same experiment identity:

| segment | final frames | measured seconds | frames/s | max policy lag | status |
|---|---:|---:|---:|---:|---|
| initial | 43,200,000 | 2,558.207 | 16,886.829 | 20 | completed |
| resume 1 | 46,400,000 | 197.100 | 16,235.404 | 20 | completed |
| resume 2 | 56,640,000 | 624.199 | 16,405.027 | 20 | completed |
| final resume | 56,960,000 | 26.263 | 12,184.634 | 20 | completed |

The checkpoint advanced monotonically from 43,200,000 frames and 13,500
learner updates to 56,960,000 frames and 17,800 updates. Every Actor exited
with code zero, no learner thread remained alive, and policy lag remained
below the predeclared 128-update bound. A1 accumulated 3,405.769 measured
seconds of soak time. The final checkpoint SHA-256 was
`7fba07db497827483fc151ca5d82fb7919bbfcb7cd00d3932409822cd05b0f93`.

## Decision

The A1 topology from main remains the production recommendation. Checkpoint
resume and policy-lag behavior passed, but this candidate branch does not clear
the throughput promotion gate. Its model/runtime additions remain opt-in or
experimental until the 7.97% regression is isolated and removed.
