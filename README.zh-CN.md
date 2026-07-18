# [ICML 2021] DouZero: 从零开始通过自我博弈强化学习来学打斗地主
<img width="500" src="https://gitee.com/daochenzha/DouZero/raw/main/imgs/douzero_logo.jpg" alt="Logo" />

[![Building](https://github.com/kwai/DouZero/actions/workflows/python-package.yml/badge.svg)](https://github.com/kwai/DouZero/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/douzero.svg)](https://badge.fury.io/py/douzero)
[![Downloads](https://pepy.tech/badge/douzero)](https://pepy.tech/project/douzero)
[![Downloads](https://pepy.tech/badge/douzero/month)](https://pepy.tech/project/douzero)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)

[English README](README.md)

DouZero是一个为斗地主设计的强化学习框架。斗地主十分具有挑战性。它包含合作、竞争、非完全信息、庞大的状态空间。斗地主也有非常大的动作空间，并且每一步合法的牌型会非常不一样。DouZero由快手AI平台部开发。

*   在线演示: [https://www.douzero.org/](https://www.douzero.org/)
      * :loudspeaker: 抢先体验叫牌版本（调试中）: [https://www.douzero.org/bid](https://www.douzero.org/bid)
*   离线运行演示: [https://github.com/datamllab/rlcard-showdown](https://github.com/datamllab/rlcard-showdown)
*   论文: [https://arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135) 
*   视频: [YouTube](https://youtu.be/inHIi8sej7Y)
*   论文: [https://arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135) 
*   相关仓库: [RLCard Project](https://github.com/datamllab/rlcard)
*   相关资源: [Awesome-Game-AI](https://github.com/datamllab/awesome-game-ai)
*   由社区贡献者开发的非官方改进版: [[DouZero ResNet]](https://github.com/Vincentzyx/Douzero_Resnet) [[DouZero FullAuto]](https://github.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto)
*   知乎：[https://zhuanlan.zhihu.com/p/526723604](https://zhuanlan.zhihu.com/p/526723604)
*   杂项资源：您听说过以数据为中心的人工智能吗？请查看我们的 [data-centric AI survey](https://arxiv.org/abs/2303.10158) 和 [awesome data-centric AI resources](https://github.com/daochenzha/data-centric-AI)!

**社区:**
*  **Slack**: 加入 [DouZero](https://join.slack.com/t/douzero/shared_invite/zt-rg3rygcw-ouxxDk5o4O0bPZ23vpdwxA) 频道.
*  **QQ群**: 加入我们的QQ群讨论。密码: douzeroqqgroup
	*  一群：819204202
	*  二群：954183174
	*  三群：834954839
	*  四群：211434658
	*  五群：189203636

**最新动态:**
*   感谢 [@Vincentzyx](https://github.com/Vincentzyx) 实现了可移植的 CPU Actor/训练路径，该路径也适用于 Windows。

<img width="500" src="https://douzero.org/public/demo.gif" alt="Demo" />

## 引用
如果您用到我们的项目，请添加以下引用：

Zha, Daochen et al. “DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning.” ICML (2021).

```bibtex
@InProceedings{pmlr-v139-zha21a,
  title = 	 {DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning},
  author =       {Zha, Daochen and Xie, Jingru and Ma, Wenye and Zhang, Sheng and Lian, Xiangru and Hu, Xia and Liu, Ji},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12333--12344},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zha21a/zha21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/zha21a.html},
  abstract = 	 {Games are abstractions of the real world, where artificial agents learn to compete and cooperate with other agents. While significant achievements have been made in various perfect- and imperfect-information games, DouDizhu (a.k.a. Fighting the Landlord), a three-player card game, is still unsolved. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. Unfortunately, modern reinforcement learning algorithms mainly focus on simple and small action spaces, and not surprisingly, are shown not to make satisfactory progress in DouDizhu. In this work, we propose a conceptually simple yet effective DouDizhu AI system, namely DouZero, which enhances traditional Monte-Carlo methods with deep neural networks, action encoding, and parallel actors. Starting from scratch in a single server with four GPUs, DouZero outperformed all the existing DouDizhu AI programs in days of training and was ranked the first in the Botzone leaderboard among 344 AI agents. Through building DouZero, we show that classic Monte-Carlo methods can be made to deliver strong results in a hard domain with a complex action space. The code and an online demo are released at https://github.com/kwai/DouZero with the hope that this insight could motivate future work.}
}
```

## 为什么斗地主具有挑战性
除了非完全信息带来的挑战外，斗地主本身也包含巨大的状态和动作空间。具体来说，斗地主的动作空间大小高达10^4（详见[该表格](https://github.com/datamllab/rlcard#available-environments)）。不幸的是，大部分强化学习算法都只能处理很小的动作空间。并且，斗地主的玩家需要在部分可观测的环境中，与其他玩家对抗或合作，例如：两个农民玩家需要作为一个团队对抗地主玩家。对对抗和合作同时进行建模一直以来是学术界的一个开放性问题。

在本研究工作中，我们提出了将深度蒙特卡洛（Deep Monte Carlo, DMC）与动作编码和并行演员（Parallel Actors）相结合的方法，为斗地主提供了一个简单而有效的解决方案，详见[我们的论文](https://arxiv.org/abs/2106.06135)。

## 安装
训练部分的代码是基于GPU设计的，因此如果想要训练模型，您需要先安装CUDA。安装步骤可以参考[官网教程](https://docs.nvidia.com/cuda/index.html#installation-guides)。对于评估部分，CUDA是可选项，您可以使用CPU进行评估。

首先，克隆本仓库（如果您访问Github较慢，国内用户可以使用[Gitee镜像](https://gitee.com/daochenzha/DouZero)）：
```
git clone https://github.com/kwai/DouZero.git
```

确保您已经安装好Python 3.11及以上版本，然后安装依赖：
```
cd douzero
pip3 install -r requirements.txt
```
我们推荐通过以下命令安装稳定版本的Douzero：
```
pip3 install douzero
```
如果您访问较慢，国内用户可以通过清华镜像源安装：
```
pip3 install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或是安装最新版本（可能不稳定）：
```
pip3 install -e .
```
Windows 并非只能笼统地使用 CPU：legacy 训练可以把 CPU Actor 与 CUDA
Learner 分开，单进程 V2 训练也可以选择 CPU 或 CUDA。不过仓库没有持续运行
的原生 Windows CUDA CI；legacy CUDA Actor 与 CUDA DDP 仍不在原生 Windows
文档支持边界内。选择训练拓扑前请阅读 [Windows 训练说明](docs/windows_training.md)。

## 项目文档

- [Windows 训练说明](docs/windows_training.md)：原生 Windows/WSL2 选择、CPU Actor 与 CUDA Learner、V2、checkpoint 和 DDP 限制。
- [部署与发布审计](docs/deployment.md)：严格模型包、导出与发布门禁。
- [迁移与回滚](docs/migration.md)：legacy、factorized 与 V2 兼容及回滚步骤。
- [模型卡模板](docs/model_card.md)：每次模型发布需要补全的审计信息。

## 训练
当前代码有两个不同的训练入口。legacy `train.py` 使用多进程 Actor 和 Learner
线程；V2 `train_v2.py` 默认单进程，通过 `--device` 选择设备，并支持 V2
checkpoint 与 metrics 输出。

| 路径 | CPU | 单块 CUDA GPU | 多 GPU | 原生 Windows 状态 |
|---|---|---|---|---|
| Legacy CPU Actor + CPU Learner | 支持 | 不适用 | 不适用 | 代码路径支持；无原生 Windows CI |
| Legacy CPU Actor + CUDA Learner | 不适用 | 支持 | Learner 仅一个设备 | 代码路径支持；原生 Windows CUDA 未持续验证 |
| Legacy CUDA Actor | 不适用 | 使用共享 CUDA 多进程 | 面向 Linux 的路径 | 原生 Windows 明确不作为受支持/已验证路径 |
| V2 单进程 | 默认 CPU | `--device cuda` 或 `auto` | 不适用 | 代码路径支持；原生 Windows CUDA 未持续验证 |
| 基础 V2 DDP | CPU/Gloo | CUDA/NCCL | 通过 `torchrun` | CUDA DDP 建议 WSL2/Linux |
| P17 standard/full-game V2 | 支持 | 支持 | 不支持，会 fail closed | 仅单进程 |

持续 CI 是 Ubuntu CPU-only。这里的“代码路径支持”不代表已经通过原生
Windows CUDA 验证。长期循环、checkpoint 与恢复已通过 CPU 测试，但没有
宣称 CUDA soak 已完成。完整安装、边界与排错见
[docs/windows_training.md](docs/windows_training.md)。

原有面向 Linux 的 legacy GPU 拓扑可运行：
```
python train.py
```
这会使用一块GPU训练DouZero。如果需要用多个GPU训练Douzero，使用以下参数：
*   `--gpu_devices`: 用作训练的GPU设备名
*   `--num_actor_devices`: 被用来进行模拟（如自我对弈）的GPU数量
*   `--num_actors`: 每个设备的演员进程数
*   `--training_device`: 用来进行模型训练的设备

例如，如果我们拥有4块GPU，我们想用前3个GPU进行模拟，每个GPU拥有15个演员，而使用第四个GPU进行训练，我们可以运行以下命令：
```
python train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
Legacy 的 Actor 与 Learner 设备可以独立选择：
*   `--training_device cpu`: 用CPU来训练
*   `--actor_device_cpu`: 用CPU来模拟

Legacy 全 CPU（PowerShell）：

```powershell
python train.py `
  --actor_device_cpu `
  --training_device cpu
```

Legacy CPU Actor + 逻辑 CUDA Learner 0（PowerShell）：

```powershell
python train.py `
  --actor_device_cpu `
  --gpu_devices 0 `
  --training_device 0
```

V2 CPU 冒烟（PowerShell）：

```powershell
python train_v2.py `
  --device cpu `
  --episodes 4 `
  --optimizer_steps 1 `
  --batch_size 1 `
  --seed 1
```

V2 长期训练必须显式传入 `--long_running`；上面的原有一次性命令行为不变。
下面的例子训练 100 个 cycle，并只保留最近 5 个原子 checkpoint：

```bash
python train_v2.py --long_running --device cpu --seed 17 \
  --episodes_per_cycle 32 --optimizer_steps_per_cycle 8 --max_cycles 100 \
  --checkpoint_path artifacts/v2/train.pt --checkpoint_every_cycles 1 \
  --keep_last_checkpoints 5 --metrics_path artifacts/v2/metrics.json
```

可用稳定的 latest 清单恢复；累计计数、policy step、优化器、模型和 RNG
都会从保存边界继续：

```bash
python train_v2.py --long_running --device cpu --seed 17 \
  --episodes_per_cycle 32 --optimizer_steps_per_cycle 8 --max_cycles 200 \
  --checkpoint_path artifacts/v2/train.pt \
  --resume_checkpoint artifacts/v2/train-latest.json
```

每个 cycle 边界都会清空 replay。恢复只发生在该边界，不会序列化或伪造
中途 replay。全新训练不会复用已有 checkpoint 系列：必须恢复该系列或使用
新路径。逐 cycle 指标追加到 `metrics-cycles.jsonl`，`metrics.json` 保持为
原子更新的运行摘要。细节见 [V2 长期训练状态机](docs/training_system.md#long-running-v2-state-machine)。

V2 单 GPU，并输出 checkpoint 与 metrics（PowerShell）：

```powershell
python train_v2.py `
  --config configs\enhanced.yaml `
  --device cuda `
  --episodes 8 `
  --optimizer_steps 2 `
  --checkpoint_path artifacts\windows\v2-checkpoint.pt `
  --metrics_path artifacts\windows\v2-metrics.json
```

P17 standard/full-game 单进程冒烟（PowerShell）：

```powershell
python train_v2.py `
  --config configs\standard_v2.yaml `
  --device cuda `
  --episodes 2 `
  --optimizer_steps 1 `
  --batch_size 1 `
  --checkpoint_path artifacts\windows\standard-v2.pt `
  --metrics_path artifacts\windows\standard-v2-metrics.json
```

P17 CPU 冒烟只需改为 `--device cpu`。standard learned bidding 当前仅支持
单进程 eager；DDP 与 `compile_model` 会 fail closed。P17 是基础设施而非正式
模型发布：**Release candidate: NONE。Release status: NOT READY。** V2 AMP、
严格 checkpoint 恢复和 DDP 边界见 [Windows 训练说明](docs/windows_training.md)
与 [P14 训练系统](docs/training_system.md)。

其他定制化的训练配置可以参考以下可选项：
```
--xpid XPID           实验id（默认值：douzero）
--save_interval SAVE_INTERVAL
                      保存模型的时间间隔（以分钟为单位）
--objective {adp,wp,logadp}
                      使用 ADP、WP 或 log-ADP 作为奖励（默认值：ADP）
--actor_device_cpu    用CPU进行模拟
--gpu_devices GPU_DEVICES
                      用作训练的GPU设备名
--num_actor_devices NUM_ACTOR_DEVICES
                      被用来进行模拟（如自我对弈）的GPU数量
--num_actors NUM_ACTORS
                      每个设备的演员进程数
--training_device TRAINING_DEVICE
                      用来进行模型训练的设备。`cpu`表示用CPU训练
--load_model          读取已有的模型
--disable_checkpoint  禁用保存检查点
--savedir SAVEDIR     实验数据存储跟路径
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      探索概率
--batch_size BATCH_SIZE
                      训练批尺寸
--unroll_length UNROLL_LENGTH
                      展开长度（时间维度）
--num_buffers NUM_BUFFERS
                      共享内存缓冲区的数量
--num_threads NUM_THREADS
                      学习者线程数
--max_grad_norm MAX_GRAD_NORM
                      最大梯度范数
--learning_rate LEARNING_RATE
                      学习率
--alpha ALPHA         RMSProp平滑常数
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## 评估
评估可以使用GPU或CPU进行（GPU效率会高得多）。预训练模型可以通过[Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing)或[百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ), 提取码: 4624 下载。将预训练权重放到`baselines/`目录下。模型性能通过自我对弈进行评估。我们提供了一些其他预训练模型和一些启发式方法作为基准：
*   [random](douzero/evaluation/random_agent.py): 智能体随机出牌（均匀选择）
*   [rlcard](douzero/evaluation/rlcard_agent.py): [RLCard](https://github.com/datamllab/rlcard)项目中的规则模型
*   SL (`baselines/sl/`): 基于人类数据进行深度学习的预训练模型
*   DouZero-ADP (`baselines/douzero_ADP/`): 以平均分数差异（Average Difference Points, ADP）为目标训练的Douzero智能体
*   DouZero-WP (`baselines/douzero_WP/`): 以胜率（Winning Percentage, WP）为目标训练的Douzero智能体

### 第1步：生成评估数据
```
python3 generate_eval_data.py
```
以下为一些重要的超参数。
*   `--output`: pickle数据存储路径
*   `--num_games`: 生成数据的游戏局数，默认值 10000

## 第2步：自我对弈
```
python3 evaluate.py
```
以下为一些重要的超参数。
*   `--landlord`: 扮演地主的智能体，可选值：random, rlcard或预训练模型的路径
*   `--landlord_up`: 扮演地主上家的智能体，可选值：random, rlcard或预训练模型的路径
*   `--landlord_down`: 扮演地主下家的智能体，可选值：random, rlcard或预训练模型的路径
*   `--eval_data`: 包含评估数据的pickle文件
*   `--num_workers`: 用多少个进程进行模拟
*   `--gpu_device`: 用哪个GPU设备进行模拟。默认用CPU

例如，可以通过以下命令评估DouZero-ADP智能体作为地主对阵随机智能体
```
python3 evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```
以下命令可以评估DouZero-ADP智能体作为农民对阵RLCard智能体
```
python3 evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```

### 配对评测（P15）

需要同牌配对置信区间、地主/农民镜像对局、完整竞叫与座位轮换、校准、
延迟、模型矩阵和消融实验时，请使用 `evaluate_paired.py`。无需权重的 CPU
冒烟命令如下：

```
python3 evaluate_paired.py --candidate rule --baseline random --num-deals 8
```

统计单位、输出格式、私有留出集接口和模型矩阵格式详见
[P15 评测协议](docs/evaluation_protocol.md)。

默认情况下，我们的模型会每半小时保存在`douzero_checkpoints/douzero`路径下。我们提供了一个脚本帮助您定位最近一次保存检查点。运行
```
sh get_most_recent.sh douzero_checkpoints/douzero/
```
之后您可以在`most_recent_model`路径下找到最近一次保存的模型。

## Windows下的问题
不要把“GPU Actor”和“GPU Learner”混为同一种模式。使用
`--actor_device_cpu` 时，legacy Actor 的共享模型和 replay buffer 保持在 CPU，
而 `--training_device 0` 仍可把 Learner 放到 CUDA。V2 是单进程入口，通过
`--device` 独立选择 CPU/CUDA。

Legacy CUDA Actor 会在 spawn 子进程之间共享 CUDA 模型和 buffer；该拓扑不是
原生 Windows 上受支持或持续验证的项目路径。仓库也没有原生 Windows CUDA
CI。对于代码允许的单 GPU 路径，请先做有界冒烟；对于 legacy CUDA Actor、
CUDA DDP/NCCL、多 GPU 和正式长时间训练，优先使用并实际验证 WSL2/Linux。
安装、真实 CUDA 运算检查、checkpoint 恢复和具体错误排查见
[Windows 训练说明](docs/windows_training.md)。

## 核心团队
*   算法：[Daochen Zha](https://github.com/daochenzha), [Jingru Xie](https://github.com/karoka), Wenye Ma, Sheng Zhang, [Xiangru Lian](https://xrlian.com/), Xia Hu, [Ji Liu](http://jiliu-ml.org/)
*   GUI演示：[Songyi Huang](https://github.com/hsywhu)
*   社区贡献者: [@Vincentzyx](https://github.com/Vincentzyx)

## 鸣谢
*   本演示基于[RLCard-Showdown](https://github.com/datamllab/rlcard-showdown)项目
*   代码实现受到[TorchBeast](https://github.com/facebookresearch/torchbeast)项目的启发
