# V1/Legacy DouZero 单 GPU 性能报告

日期：2026-07-19
结论状态：实测候选已经完成三次重复；30 分钟稳定性与固定 frames paired evaluation 见对应章节。

## 结论

**唯一首选：A1，12 个单线程 CPU factorized Actor + CUDA:0 eager FP32 Learner。**

配置为 `benchmarks/configs/legacy_a1_cpu_factorized.yaml`。最终固定 seed 长协议三次
实测 median 为 **16,642.1 frames/s**，p95 为 16,683.4，范围/median 为 0.68%。
它比同机可测的 Legacy CPU Actor 基线高 67.0%，比最佳 B1 高 430%，并且只使用
1.28 GiB 峰值显存。FP16、BF16、compile 和 compile+BF16 相对 A1 均不足 0.5%，
因此按预先规定的 5% 规则保留简单 eager FP32 路径。

**唯一备用：A2 BF16，A1 加 learner-only BF16 AMP。** 配置为
`benchmarks/configs/legacy_a2_cpu_factorized_bf16.yaml`。其 median 为 16,693.4
frames/s（仅 +0.31%），没有 fallback，峰值显存约低 214 MiB；但进程树 RAM 约高
343 MiB，且混合精度增加训练语义风险，所以只作为显存受限时的备用而不是首选。

## 测试环境

以下均为**实际测量**，没有用理论 FLOPS 代替端到端训练：

| 项目 | 值 |
|---|---|
| 测试机 / 路径 | `LocalServer` / `/opt/DouZero` |
| Git commit | `ce6c4237fd18207b9dd8c3c15b94e9cdaa8fdac5` 加本报告中的未提交优化补丁 |
| Docker image | `douzero-test:latest` |
| Image ID | `sha256:6f57b50161e8a4c4147fda854b76b255bdf643a2703c7da464b89277da01f953` |
| Python / PyTorch | 3.12.3 / 2.12.1+cu132 |
| CUDA / cuDNN | 13.2 / 92000 |
| GPU / Driver | NVIDIA GeForce RTX 5070 12,227 MiB / 595.71.05 |
| CPU | Intel Core Ultra 5 245KF，14 physical cores / 14 logical CPUs |
| OS | Linux 7.0.0-28-generic x86_64, glibc 2.39 |

最终代码同步后使用以下命令重建并保留了上述标准镜像：

```bash
cd /opt/DouZero
git status
docker build --progress=plain \
  --build-arg DOUZERO_GIT_SHA="$(git rev-parse HEAD)" \
  -t douzero-test:latest .
```

主长测协议为 64,000 warmup frames + 128,000 measurement frames，每项 3 次，
checkpoint 全部关闭。`batch_size=32`、`unroll_length=100`、objective、epsilon、
learning rate、seed 与 deterministic 设置保持一致。p95 是三次样本的 nearest-rank
p95，因此等于最大值；同时报告范围/median 作为离散度。

## 拓扑对比

下表均为最终固定 seed 长协议的三次**实测 median**。RAM 是整个训练进程树 RSS，
CPU 100% 表示一个逻辑核；C0 的 RSS/CPU 已包含集中推理进程。三种拓扑均使用修复
后的 policy-lag 聚合器。

| 拓扑 | 组合 | frames/s | decisions/s | transitions/s | games/s | updates/s | GPU | CPU | VRAM MiB | RAM MiB | max lag | 离散度 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 12 CPU factorized Actor + GPU eager Learner | 16,642.1 | 16,804.8 | 16,869.6 | 333.79 | 5.201 | 3% | 1209% | 1,280 | 9,082 | 14.79 | 0.68% | 首选 |
| B1 | 4 GPU factorized Actor + GPU eager Learner | 3,140.4 | 3,138.5 | 3,135.7 | 62.02 | 0.981 | 98% | 401% | 2,683 | 5,324 | 12.51 | 2.10% | GPU 争用，淘汰 |
| C0 | 12 CPU env + 集中 GPU 推理 + GPU Learner | 6,627.0 | 6,578.1 | 6,558.0 | 127.12 | 2.071 | 26% | 292% | 1,657 | 9,863 | 14.98 | 5.18% | RPC/微批等待，淘汰 |

**基于 profiling 的推断：** B1 的三个 learner role 平均每批等待 3.24--3.29 s，
而 A1 为 0.52--0.54 s；B1 的 98% GPU utilization 主要代表多个 Actor CUDA context
与 Learner 竞争，并不代表更高有效吞吐。C0 每次集中推理往返平均 1.67 ms，远高于
A1 本地 CPU inference 的 0.43 ms；当前单局/Actor 请求方式没有形成足够微批。

### Actor 数量扫描

这些是 6,400 warmup + 19,200 measurement frames 的三次**实测 median**：

| 路径 | Actor 数 | frames/s | 观察 |
|---|---:|---:|---|
| A0 Legacy CPU，Actor threads=1 | 4 | 5,570.7 | 核心未用满，离散度 63.3% |
| A0 Legacy CPU，Actor threads=1 | 8 | 9,901.3 | 接近线性扩展 |
| A0 Legacy CPU，Actor threads=1 | 12 | 10,277.2 | 本机最佳 A0 |
| B0 Legacy GPU | 1 | 2,499.4 | 本机最佳 B0 |
| B0 Legacy GPU | 2 | 2,274.4 | 多进程 CUDA 开始倒退 |
| B0 Legacy GPU | 4 | 2,127.3 | GPU 86--98% 但供数更慢 |
| B0 Legacy GPU | 6 | 2,220.6 | 离散度 13.7% |
| B1 factorized GPU | 1 | 2,886.3 | factorized 有收益 |
| B1 factorized GPU | 2 | 2,917.5 | 接近 |
| B1 factorized GPU | 4 | 3,285.2 | 短测最佳 |
| B1 factorized GPU | 6 | 2,981.5 | 资源竞争再次恶化 |

真正不修改 Actor 线程设置的 A0（`actor_torch_threads=0`）在 4 Actors 下运行超过
8 分钟仍为 **0 frames**，容器 CPU 约 1,391%、RSS 1.75 GiB，因每个 Actor
继承多线程导致严重 oversubscription；该任务所属容器已停止。为得到可比较吞吐，
A0 扫描单独记录为 `legacy_a0_cpu_actor_thread1.yaml`，不能把它冒充原始默认结果。
list-of-tensors 基线还需要 `--ulimit nofile=65536:65536`，否则 Docker 默认 1024
FD 会在 1,050 个共享 tensor 附近报 `Too many open files`。

## 增量收益

下表采用最终 64,000 warmup + 128,000 measurement 长协议、固定 seed、12 CPU
Actors、三次 median；相对值以可测 A0 thread1 的 9,964.0 frames/s 为基准。
这些是**实际测量**。

| 阶段 | 主要变化 | frames/s | 相对 A0 |
|---|---|---:|---:|
| A0 | Legacy Actor，threads=1 | 9,964.0 | 基准 |
| log | 5 s 主线程写日志、复用 CSV handle | 9,941.3 | -0.2% |
| snapshot | 3 slots、sync=16、锁外配对 copy | 10,524.3 | +5.6% |
| factorized | 从 observation 源头拆分输入、单动作跳过 | 16,341.8 | +64.0% |
| A1 data path | contiguous buffers、bulk copy、`>=T`、reusable pinned | 16,669.0 | +67.3% |

最终固定 seed 长协议进一步比较了所有 learner 路径：FP32 A1 16,642.1；FP16
16,717.7，但每次仍累计 3 次 fallback，实际很快退回 FP32；BF16 16,693.4；
compile FP32 16,640.8；compile+BF16 16,602.9。全部相对 A1 不足 0.5%，所以
FP16 因 fallback 淘汰，compile 因没有收益淘汰，BF16 只保留为低显存备用。

其他**实测消融**：batch 64 median 15,274.6 frames/s，低于 batch 32，峰值显存
约 1.93 GiB；`num_threads=2` median 15,875.4，仅约 1% 且显存更高；sync 8 与
32 分别为 14,785.6 和 14,851.0，均低于 sync 16。batch 64 未进行充分学习行为
验证，因此不会被推荐。

## Profile 明细

以下为 A1 一个代表性长测 run 的采样均值，CPU 使用 `perf_counter_ns`，CUDA stage
使用预复用 CUDA Events；没有在每个 decision 中 `cuda.synchronize()` 或打印：

| Actor stage | ms/decision | Learner stage | ms/profiled update |
|---|---:|---|---:|
| env step | 0.177 | batch assembly | 4.604 |
| legal action generation | 0.021 | H2D | 0.259 |
| observation encode | 0.085 | forward | 3.222 |
| inference | 0.425 | backward | 2.910 |
| rollout write | 0.004 | gradient clip | 0.036 |
| free queue wait | 0.00018 | optimizer | 0.788 |
| full queue put | 0.00032 | snapshot publish | 2.719 |

合法动作数量 p50=2、p95=22、max=669，单合法动作比例 44.16%。A3 的 optimizer
由 0.788 ms 降至约 0.422 ms，但 forward 仍为 3.27 ms；说明本机 A3 收益主要
来自 foreach，而不是 compile 的 learner forward。

## 正确性门禁

| 门禁 | 实测状态 |
|---|---|
| V1 / factorized CPU 数值与动作 parity | 通过现有与新增测试 |
| CUDA value 与 epsilon=0 action-index parity，三角色 | 3/3 通过 |
| 非法动作 | 所有 smoke、benchmark 与稳定性 run 未发生 |
| finite loss / gradient / parameters | 训练期抽样检查通过；checkpoint 参数全部 finite |
| 旧 checkpoint / 新 checkpoint | 原始 key/shape 不变，兼容 loader 测试通过 |
| resume | 240 -> 440 frames；22 learner updates；三个 optimizer state 各 16 项 |
| crash propagation / bounded shutdown | A1、B1、C0 smoke 与 benchmark worker exitcode=0；线程存活数 0 |
| 默认行为 | 新开关默认关闭；`legacy_actor_backend=legacy`；两份 legacy 默认 YAML 一致 |

最终标准 Docker 镜像中的完整 CPU suite 100% 通过（7 个需要 CUDA 的测试按预期
skip，2 个既有 warning）；三角色 CUDA parity 3/3 通过。最终复测命令见
“复现命令”。

## 稳定性

34,006,400 frames 的 A1 **实际运行 1,978.6 秒（32.98 分钟 measurement window）**：
17,155.0 frames/s、5.361 updates/s、447.59 games/s。394 个 5 秒系统样本中，
RSS 在启动后的第 5 分钟为 8,917 MiB，之后最高 8,929 MiB，结束为 8,925 MiB；
VRAM 在第 5 分钟稳定为 1,478 MiB 并保持到结束。首尾增长 178/200 MiB 来自前几
分钟的 CUDA/allocator warmup，不能解释为持续泄漏。12 个 Actor exitcode 全为 0，
learner 存活线程为 0，三个 role loss/参数 finite，max policy lag 均为 16 updates。
原始 394 点时序在 `benchmarks/results/legacy_v1_rtx5070_20260719/stability/metrics.json`。

## 固定 Frames Paired Evaluation

A1 FP32 与 A2 BF16 使用同一 seed `20260719`、相同 1,024,000 目标 frames 训练；
两者因三个并行 role learner 的边界在完全相同的 1,030,400 frames 停止，并生成
原始 V1 三角色 sidecar。随后对 1,000 固定 public deals 做两种 seat rotations，
共 2,000 局，使用 5,000 次 deal-clustered bootstrap：

| 指标（BF16 - FP32） | estimate | 95% CI |
|---|---:|---:|
| paired win-rate delta | -0.25 pp | [-1.90, +1.40] pp |
| paired mean score / ADP delta | +0.013 | [-0.084, +0.106] |

总体 CI 跨 0，且排除了超过 1.9 percentage points 或 0.084 score 的总体负向差异，
没有显示有意义的总体能力退化。分角色 farmer 点估计为 -2.7 pp，CI 仍跨 0 且下界
约 -5.8 pp，因此 BF16 仍只作为备用；此短训练 checkpoint 不能代表收敛后绝对牌力。
JSON/CSV/Markdown、checkpoint SHA-256 身份和训练 metrics 均固化在
`benchmarks/results/legacy_v1_rtx5070_20260719/paired/`。这是 local、非 formal-release
评估；provenance 标记 source stable=true、clean=false，因为优化补丁尚未提交。

## 实现范围

主要改动包括：

- factorized observation 在环境源头生成，Learner 仍为原始 Legacy `Model`；
- `torch.inference_mode()`、单动作跳过、`.item()`、编码动作复用；
- contiguous rollout buffer、批量 `copy_`、`>=T`、GPU publish 前同步；
- reusable pinned batch staging、`index_select(out=...)`、nonblocking H2D 和 Event
  所有权保护；
- policy snapshot 预配对 tensor、锁外 copy、三角色原子 flip、整局固定 slot；
- learner stage profiler、系统 sampler、policy lag/queue/合法动作统计与 JSON 输出；
- learner-only compile、foreach optimizer/clip、带 finite fallback 的 AMP；
- C0 共享 slot + 小 metadata queue + role/policy/bucket 聚合 + timeout/abort/shutdown；
- checkpoint runtime state、optimizer/frames 恢复，以及 FileWriter resume 修复。

三角色 checkpoint 仍保存原始 V1 state_dict；没有使用 `model_version=factorized`
绕过门禁。epsilon 抽样顺序在单合法动作优化中保留，RMSprop 默认仍走原路径。

## 复现命令

进入远端后必须先检查状态：

```bash
ssh LocalServer
cd /opt/DouZero
git status
docker version
nvidia-smi
```

长协议 A1、FP16、BF16、compile 与 compile+BF16：

```bash
docker run --rm --gpus all --shm-size=4g --ulimit nofile=65536:65536 \
  --mount type=bind,src=/opt/DouZero/.git,dst=/workspace/DouZero/.git,readonly \
  --mount type=bind,src=/tmp/douzero-legacy-results,dst=/output \
  douzero-test:latest python benchmarks/bench_legacy_training.py \
  --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
  --config benchmarks/configs/legacy_a2_cpu_factorized_amp.yaml \
  --config benchmarks/configs/legacy_a2_cpu_factorized_bf16.yaml \
  --config benchmarks/configs/legacy_a3_cpu_factorized_compile.yaml \
  --config benchmarks/configs/legacy_a4_cpu_factorized_compile_bf16.yaml \
  --repeats 3 --warmup_frames 64000 --measure_frames 128000 \
  --profile_sample_interval 10 --monitor_interval_seconds 1 \
  --seed 20260719 \
  --output_dir /output/a-final
```

B1 与 C0 把 `--config` 替换为各自 YAML。Actor sweep 另加 `--num_actors N`。

稳定性 run 的容器内命令：

```bash
python train.py \
  --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
  --num_actors 12 --total_frames 34000000 \
  --benchmark_warmup_frames 64000 \
  --legacy_profile_sample_interval 100 \
  --legacy_monitor_interval_seconds 5 \
  --legacy_metrics_path /output/metrics.json \
  --savedir /output --xpid a1-stability --disable_checkpoint
```

paired checkpoint 训练使用同一 Docker 参数和 `/output` mount，分别执行：

```bash
python train.py --config benchmarks/configs/legacy_a1_cpu_factorized.yaml \
  --seed 20260719 --total_frames 1024000 --no-disable_checkpoint \
  --savedir /output --xpid paired-a1

python train.py --config benchmarks/configs/legacy_a2_cpu_factorized_bf16.yaml \
  --seed 20260719 --total_frames 1024000 --no-disable_checkpoint \
  --savedir /output --xpid paired-bf16

python evaluate_paired.py --mode cardplay_only \
  --candidate a2_bf16 --baseline a1_fp32 \
  --model-matrix /output/model_matrix.json \
  --num-deals 1000 --seed 20260719 --bootstrap-samples 5000 \
  --output /output/evaluation
```

`model_matrix.json` 的可复用副本位于 raw-results 的 `paired/` 目录。

CPU 与 CUDA 测试：

```bash
docker run --rm \
  --mount type=bind,src=/opt/DouZero/.git,dst=/workspace/DouZero/.git,readonly \
  douzero-test:latest python -m pytest -q -p no:cacheprovider

docker run --rm --gpus all \
  --mount type=bind,src=/opt/DouZero/.git,dst=/workspace/DouZero/.git,readonly \
  douzero-test:latest python -m pytest -q -p no:cacheprovider \
  tests/test_factorized_parity.py -m cuda
```

每个 benchmark 目录含逐 run JSON、`raw.csv`、`summary.json` 和自动生成的
`summary.md`。这些文件是报告数字的原始来源。

## 已知限制与回滚

**实际确认的限制：** B 拓扑仅支持 Linux/WSL2 NVIDIA；本次只测 RTX 5070。
GPU Actor 的三个最终 run 在有界正常退出时仍出现 PyTorch producer-exit CUDA IPC
warning；publish 前同步没有消除此告警，B 因性能和清理风险共同淘汰。C0 已实现共享 slot、bucket 和微批，但尚未实现建议的
每 Actor 四局交错，因此不能推断更完整的 C 设计一定仍慢。30 分钟运行和 paired
evaluation 之外，没有执行完整数十亿 frames 收敛实验。

**尚未验证的假设：** 更多物理核、不同 GPU/驱动或真正多局交错 C 可能改变排序；
必须在目标机器重新跑相同协议，不能从本报告外推。

回滚不需要 checkpoint 迁移：保持 `legacy_actor_backend: legacy`，并关闭
`legacy_contiguous_buffers`、`legacy_bulk_rollout`、`legacy_flush_ge`、
`legacy_reusable_pinned_staging`、`compile_learner`、`rmsprop_foreach`、
`grad_clip_foreach` 与 AMP，即回到原始 Legacy 数值/数据路径。代码仍能加载旧 V1
checkpoint；若完全移除补丁，优化期新 checkpoint 也是原始 key/shape。

测试结束后已显式删除本任务创建的 21 个 `/tmp/douzero-legacy-*` 目录、训练日志、
paired 临时 checkpoint 和意外同步的远端 `.venv`/`.claude`；task-owned container 为
0。保留 `douzero-test:latest` 和 Docker build cache。最终 `/opt/DouZero` 的
`git status` 只包含本交付的代码/配置/测试/文档修改，没有测试生成文件。
