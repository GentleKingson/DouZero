# [ICML 2021] DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning
<img width="500" src="https://raw.githubusercontent.com/kwai/DouZero/main/imgs/douzero_logo.jpg" alt="Logo" />

[![Building](https://github.com/kwai/DouZero/actions/workflows/python-package.yml/badge.svg)](https://github.com/kwai/DouZero/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/douzero.svg)](https://badge.fury.io/py/douzero)
[![Downloads](https://pepy.tech/badge/douzero)](https://pepy.tech/project/douzero)
[![Downloads](https://pepy.tech/badge/douzero/month)](https://pepy.tech/project/douzero)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)

[中文文档](README.zh-CN.md)

DouZero is a reinforcement learning framework for [DouDizhu](https://en.wikipedia.org/wiki/Dou_dizhu) ([斗地主](https://baike.baidu.com/item/%E6%96%97%E5%9C%B0%E4%B8%BB/177997)), the most popular card game in China. It is a shedding-type game where the player’s objective is to empty one’s hand of all cards before other players. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. DouZero is developed by AI Platform, Kwai Inc. (快手).

*   Online Demo: [https://www.douzero.org/](https://www.douzero.org/)
       * :loudspeaker: New Version with Bid（叫牌版本）: [https://www.douzero.org/bid](https://www.douzero.org/bid)
*   Run the Demo Locally: [https://github.com/datamllab/rlcard-showdown](https://github.com/datamllab/rlcard-showdown)
*   Video: [YouTube](https://youtu.be/inHIi8sej7Y)
*   Paper: [https://arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135) 
*   Related Project: [RLCard Project](https://github.com/datamllab/rlcard)
*   Related Resources: [Awesome-Game-AI](https://github.com/datamllab/awesome-game-ai)
*   Google Colab: [jupyter notebook](https://github.com/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)
*   Unofficial improved versions of DouZero by the community: [[DouZero ResNet]](https://github.com/Vincentzyx/Douzero_Resnet) [[DouZero FullAuto]](https://github.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto)
*   Zhihu: [https://zhuanlan.zhihu.com/p/526723604](https://zhuanlan.zhihu.com/p/526723604)
*   Miscellaneous Resources:
	*   Check out our open-sourced [Large Time Series Model (LTSM)](https://github.com/daochenzha/ltsm)!
	*   Have you heard of data-centric AI? Please check out our [data-centric AI survey](https://arxiv.org/abs/2303.10158) and [awesome data-centric AI resources](https://github.com/daochenzha/data-centric-AI)!

**Community:**
*  **Slack**: Discuss in [DouZero](https://join.slack.com/t/douzero/shared_invite/zt-rg3rygcw-ouxxDk5o4O0bPZ23vpdwxA) channel.
*  **QQ Group**: Join our QQ group to discuss. Password: douzeroqqgroup

	*  Group 1: 819204202
	*  Group 2: 954183174
	*  Group 3: 834954839
	*  Group 4: 211434658
	*  Group 5: 189203636

**News:**
*   Thanks to [@Vincentzyx](https://github.com/Vincentzyx) for the portable CPU actor/training path, which is also useful on Windows.

<img width="500" src="https://douzero.org/public/demo.gif" alt="Demo" />

## Cite this Work
If you find this project helpful in your research, please cite our paper:

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

## What Makes DouDizhu Challenging?
In addition to the challenge of imperfect information, DouDizhu has huge state and action spaces. In particular, the action space of DouDizhu is 10^4 (see [this table](https://github.com/datamllab/rlcard#available-environments)). Unfortunately, most reinforcement learning algorithms can only handle very small action spaces. Moreover, the players in DouDizhu need to both compete and cooperate with others in a partially-observable environment with limited communication, i.e., two Peasants players will play as a team to fight against the Landlord player. Modeling both competing and cooperation is an open research challenge.

In this work, we propose Deep Monte Carlo (DMC) algorithm with action encoding and parallel actors. This leads to a very simple yet surprisingly effective solution for DouDizhu. Please read [our paper](https://arxiv.org/abs/2106.06135) for more details.

## Installation
The training code is designed for GPUs. Thus, you need to first install CUDA if you want to train models. You may refer to [this guide](https://docs.nvidia.com/cuda/index.html#installation-guides). For evaluation, CUDA is optional and you can use CPU for evaluation.

First, clone the repo with (if you are in China and Github is slow, you can use the mirror in [Gitee](https://gitee.com/daochenzha/DouZero)):
```
git clone https://github.com/kwai/DouZero.git
```
Make sure you have python 3.11+ installed. Install dependencies:
```
cd douzero
pip3 install -r requirements.txt
```
We recommend installing the stable version of DouZero with
```
pip3 install douzero
```
If you are in China and the above command is too slow, you can use the mirror provided by Tsinghua University:
```
pip3 install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```
or install the up-to-date version (it could be not stable) with
```
pip3 install -e .
```
Windows is not limited to a blanket "CPU-only" mode: the legacy trainer can
separate CPU actors from a CUDA learner, and the single-process V2 trainer can
select CPU or CUDA. Native Windows CUDA is not continuously tested, and legacy
CUDA actors and CUDA DDP remain outside the documented native Windows boundary.
See [Windows training](docs/windows_training.md) before choosing a topology.

## Project documentation

Engineering docs live under [`docs/`](docs/):

- [Architecture (current)](docs/architecture/current.md) — the legacy baseline reference.
- [Reproducibility](docs/reproducibility.md) — seeding, the determinism contract, and the CI Python matrix.
- [Windows training](docs/windows_training.md) — native Windows/WSL2 choices, CPU actors versus CUDA learners, V2, checkpoints, and DDP limits.
- [Configuration](docs/configuration.md) — the typed config system and `--config` YAML support.
- [Packaging](docs/packaging.md) — Python support policy and dependencies.
- [Checkpoint compatibility](docs/checkpoint_compatibility.md) — the versioned checkpoint manifest.
- [Deployment and release audit](docs/deployment.md) — strict model packages, export, and release gates.
- [Migration and rollback](docs/migration.md) — legacy/factorized/V2 compatibility and rollback steps.
- [Model card](docs/model_card.md) — required release documentation template.

## Training
The current code has two distinct training entry points. Legacy `train.py` uses
spawned actors and learner threads. V2 `train_v2.py` is single-process by
default, selects its device with `--device`, and supports V2 checkpoints and
metrics.

| Path | CPU | Single CUDA GPU | Multiple GPUs | Native Windows status |
|---|---|---|---|---|
| Legacy CPU actors + CPU learner | Yes | N/A | N/A | Code-supported; no native Windows CI |
| Legacy CPU actors + CUDA learner | N/A | Yes | One learner device | Code-supported; native Windows CUDA not continuously validated |
| Legacy CUDA actors | N/A | Shared CUDA multiprocessing | Linux-oriented path | Not supported/validated on native Windows |
| V2 single process | Default | `--device cuda` or `auto` | N/A | Code-supported; native Windows CUDA not continuously validated |
| Base V2 DDP | CPU/Gloo | CUDA/NCCL | `torchrun` | Use WSL2/Linux for CUDA DDP |
| P17 standard/full-game V2 | Yes | Yes | No; fails closed | Single process only |

The continuous CI matrix is Ubuntu CPU-only. "Code-supported" does not mean a
path has passed native Windows CUDA validation. Long-running control,
checkpoint, and resume are CPU-tested, but no CUDA soak claim is made. Full
Windows setup and troubleshooting live in [docs/windows_training.md](docs/windows_training.md).

For the original Linux-oriented legacy GPU topology, run
```
python train.py
```
This will train DouZero on one GPU. To train DouZero on multiple GPUs. Use the following arguments.
*   `--gpu_devices`: what gpu devices are visible
*   `--num_actor_devices`: how many of the GPU deveices will be used for simulation, i.e., self-play
*   `--num_actors`: how many actor processes will be used for each device
*   `--training_device`: which device will be used for training DouZero

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
The legacy actor and learner devices are independent:
*   `--training_device cpu`: Use CPU to train the model
*   `--actor_device_cpu`: Use CPU as actors

Legacy all-CPU (PowerShell):

```powershell
python train.py `
  --actor_device_cpu `
  --training_device cpu
```

Legacy CPU actors with logical CUDA learner 0 (PowerShell):

```powershell
python train.py `
  --actor_device_cpu `
  --gpu_devices 0 `
  --training_device 0
```

V2 CPU smoke (PowerShell):

```powershell
python train_v2.py `
  --device cpu `
  --episodes 4 `
  --optimizer_steps 1 `
  --batch_size 1 `
  --seed 1
```

V2 long-running mode is opt-in; the existing one-shot command above is
unchanged. This example stops after 100 cycles and retains the newest five
atomic cycle-boundary checkpoints:

```bash
python train_v2.py --long_running --device cpu --seed 17 \
  --episodes_per_cycle 32 --optimizer_steps_per_cycle 8 --max_cycles 100 \
  --checkpoint_path artifacts/v2/train.pt --checkpoint_every_cycles 1 \
  --keep_last_checkpoints 5 --metrics_path artifacts/v2/metrics.json
```

Resume from the stable latest manifest; cumulative counters, policy step,
optimizer, model, and RNG state continue from the saved boundary:

```bash
python train_v2.py --long_running --device cpu --seed 17 \
  --episodes_per_cycle 32 --optimizer_steps_per_cycle 8 --max_cycles 200 \
  --checkpoint_path artifacts/v2/train.pt \
  --resume_checkpoint artifacts/v2/train-latest.json
```

Replay is deliberately empty at every cycle boundary. Resume occurs only at
that boundary; replay is not serialized or reconstructed. A fresh run refuses
to reuse an existing checkpoint series, so resume it or select a new path.
Cycle metrics append to `metrics-cycles.jsonl` while `metrics.json` remains an
atomic, path-sanitized run summary. Manifest resume automatically continues
the same checkpoint series and rejects a conflicting `--checkpoint_path`.
Direct cycle-file resume is reconciled against the same manifest, so it cannot
ignore a newer committed orphan or duplicate a sequence. Wall-time limits are
cumulative across resumes and include checkpoint, evaluation, and metrics
boundary work. A per-series process lock prevents concurrent fresh or resumed
writers, and metrics paths are rejected if they overlap checkpoint artifacts.
See
[the V2 long-running state machine](docs/training_system.md#long-running-v2-state-machine).

V2 also has an explicit multi-CPU-actor, centralized single-GPU mode. It never
falls back when CUDA is unavailable and it rejects DDP:

```bash
python train_v2.py --v2_training_mode async_single_gpu --device cuda \
  --num_actors 4 --episodes 64 --optimizer_steps 8 --batch_size 64
```

The current async support scope is deliberately narrow: base legacy-ruleset
V2 card play only. Standard bidding, league, curriculum, RL+BC, style,
strategy features/auxiliaries, and belief fusion fail before workers start.
`single_process` remains the default and retains all existing combinations.
Compact replay records are schema-, tensor-, action-, label-, and provenance-
validated before shared slots are released and before replay insertion. Actor
failure and shutdown use spawn-shared signals so blocked peers exit promptly.

V2 single GPU with checkpoint and metrics output (PowerShell):

```powershell
python train_v2.py `
  --config configs\enhanced.yaml `
  --device cuda `
  --episodes 8 `
  --optimizer_steps 2 `
  --checkpoint_path artifacts\windows\v2-checkpoint.pt `
  --metrics_path artifacts\windows\v2-metrics.json
```

P17 standard/full-game single-process smoke (PowerShell):

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

Change the P17 command to `--device cpu` for a CPU smoke. Standard learned
bidding is single-process and eager; DDP and `compile_model` fail closed. P17 is
infrastructure, not a released model: **Release candidate: NONE. Release
status: NOT READY.** V2 AMP, strict checkpoint resume, and DDP boundaries are
documented in [Windows training](docs/windows_training.md) and
[the P14 training system](docs/training_system.md).
For more customized configuration of training, see the following optional arguments:
```
--xpid XPID           Experiment id (default: douzero)
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--objective {adp,wp,logadp}
                      Use ADP, WP, or log-ADP as reward (default: ADP)
--actor_device_cpu    Use CPU as actor device
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models. `cpu`
                	  means using cpu
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## Evaluation
The evaluation can be performed with GPU or CPU (GPU will be much faster). Pretrained model is available at [Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing) or [百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ), 提取码: 4624. Put pre-trained weights in `baselines/`. The performance is evaluated through self-play. We have provided pre-trained models and some heuristics as baselines:
*   [random](douzero/evaluation/random_agent.py): agents that play randomly (uniformly)
*   [rlcard](douzero/evaluation/rlcard_agent.py): the rule-based agent in [RLCard](https://github.com/datamllab/rlcard)
*   SL (`baselines/sl/`): the pre-trained deep agents on human data
*   DouZero-ADP (`baselines/douzero_ADP/`): the pretrained DouZero agents with Average Difference Points (ADP) as objective
*   DouZero-WP (`baselines/douzero_WP/`): the pretrained DouZero agents with Winning Percentage (WP) as objective

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
Some important hyperparameters are as follows.
*   `--output`: where the pickled data will be saved
*   `--num_games`: how many random games will be generated, default 10000

### Step 2: Self-Play
```
python3 evaluate.py
```
Some important hyperparameters are as follows.
*   `--landlord`: which agent will play as Landlord, which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_up`: which agent will play as LandlordUp (the one plays before the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_down`: which agent will play as LandlordDown (the one plays after the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--eval_data`: the pickle file that contains evaluation data
*   `--num_workers`: how many subprocesses will be used
*   `--gpu_device`: which GPU to use. It will use CPU by default

For example, the following command evaluates DouZero-ADP in Landlord position against random agents
```
python3 evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```
The following command evaluates DouZero-ADP in Peasants position against RLCard agents
```
python3 evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```

### Paired Evaluation (P15)

For deal-paired confidence intervals, mirrored landlord/farmer matchups,
full-game seat rotation, calibration, latency, model matrices, and ablations,
use `evaluate_paired.py`. A weight-free CPU smoke run is:

```
python3 evaluate_paired.py --candidate rule --baseline random --num-deals 8
```

See [the P15 evaluation protocol](docs/evaluation_protocol.md) for the
statistical unit, output formats, private-holdout interface, and model-matrix
schema.

By default, our model will be saved in `douzero_checkpoints/douzero` every half an hour. We provide a script to help you identify the most recent checkpoint. Run
```
sh get_most_recent.sh douzero_checkpoints/douzero/
```
The most recent model will be in `most_recent_model`.

## Issues in Windows
Do not treat "GPU actors" and a "GPU learner" as the same mode. With
`--actor_device_cpu`, legacy shared actor models and replay buffers stay on CPU,
while `--training_device 0` can place the learner on CUDA. The V2 entry point is
single-process and independently selects CPU/CUDA with `--device`.

Legacy CUDA actors share CUDA models and buffers across spawned processes; that
topology is not a supported or continuously validated native Windows path. The
repository also has no native Windows CUDA CI. Use bounded smokes for the
code-supported single-GPU paths, and prefer WSL2/Linux for legacy CUDA actors,
CUDA DDP/NCCL, multi-GPU, and formal long-running training. See
[Windows training](docs/windows_training.md) for installation, verified CUDA
operations, checkpoint/resume commands, and error-specific guidance.

## Core Team
*   Algorithm: [Daochen Zha](https://github.com/daochenzha), [Jingru Xie](https://github.com/karoka), Wenye Ma, Sheng Zhang, [Xiangru Lian](https://xrlian.com/), Xia Hu, [Ji Liu](http://jiliu-ml.org/)
*   GUI Demo: [Songyi Huang](https://github.com/hsywhu)
*   Community contributors: [@Vincentzyx](https://github.com/Vincentzyx)

## Acknowlegements
*   The demo is largely based on [RLCard-Showdown](https://github.com/datamllab/rlcard-showdown)
*   Code implementation is inspired by [TorchBeast](https://github.com/facebookresearch/torchbeast)

