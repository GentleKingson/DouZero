# Legacy C0 closeout validation, 2026-07-21

## Frozen environment

- C0 closeout: `aeacef168edd7cc60b894578dcc793e6b22147ca`
- main and merge-base: `388870f9a4c40a093fd86fe0d8de2da821e903f6`
- image: `sha256:6f57b50161e8a4c4147fda854b76b255bdf643a2703c7da464b89277da01f953`
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB, driver 595.71.05
- CPU: Intel Core Ultra 5 245KF
- Python/PyTorch/CUDA: 3.12.3 / 2.12.1+cu132 / 13.2
- config SHA-256: `8a1c5045b4a223df84a724e5971b8b92780c732552add0747bf034d7ed0c6791`

The source checkout was clean. Docker used `--gpus all`, `--shm-size=8g`,
and `--ulimit nofile=65536:65536` for every command.

## Concurrent failure injection

The 49-test C0 optimization/review-fix subset passed. A second explicit run
reported four passes for these failure paths:

- a full central request queue observes shutdown and releases the blocked put;
- an injected centralized/learner worker-thread exception sets the global stop
  event and rethrows the original exception in the monitor;
- abnormal multi-environment Actor cleanup recovers every policy-pool owner and
  rejects a stale lease release;
- metrics survive a completed service batch when the Actor never consumes the
  response, without masking the original failure through division by zero.

The explicit fault-injection log SHA-256 was
`e6da2210ef7c603a077a8888005c33add6d191452f42228afd2bf7f1ab1930f3`.

## Checkpoint-enabled soak and resume

| segment | final frames | measured seconds | frames/s | max policy lag | status |
|---|---:|---:|---:|---:|---|
| initial | 27,200,000 | 3,379.506 | 8,048.513 | 20 | completed |
| resume | 30,400,000 | 427.812 | 7,479.929 | 20 | completed |

The checkpoint advanced monotonically from 27,200,000 frames and 8,500
learner updates to 30,400,000 frames and 9,500 updates. All eight Actors exited
with code zero, the centralized inference thread exited normally, and no
learner thread remained alive. Policy lag stayed below the same predeclared
128-update bound. C0 accumulated 3,807.318 measured seconds; combined with the
A1 soak, the validation covered 7,213.086 seconds (2.004 hours). The final C0
checkpoint SHA-256 was
`a1e7af99eaddcbec3c8bbca2cf5602a332a94ee058acb92408c4d8fa93548448`.

## Decision

C0's supervised shutdown, owner recovery, checkpoint resume, and policy-lag
checks pass. Its sustained throughput remains about half of A1 on this Legacy
V1 model, so C0 remains explicit and experimental. This validation does not
change the production recommendation from A1.
