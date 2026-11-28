# DexMani RL

## 项目简介

DexMani RL 在 dexmani_sim 的 xArm7 + XHand 仿真中研究灵巧手 FlowMatch 策略的后训练。四种方法从预训练的行为克隆（BC）checkpoint 出发，在相同任务和测试 seed 上比较原始 BC 策略与后训练策略的成功率。

## 后训练策略

- [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://diffusion-steering.github.io/)
- [RFS: Reinforcement Learning with Residual Flow Steering for Dexterous Manipulation](https://weirdlabuw.github.io/rfs/)
- [RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process](https://student-of-holmes.github.io/RoboFFT/)
- [BORA: Bridging Offline Reinforcement Learning and Online Residual Adaptation for Real-World Dexterous VLA Models](https://chenzhongxi-sjtu.github.io/BORA/bora_results.html)

## 项目架构

| 位置 | 职责 |
| --- | --- |
| [configs/](configs/) | DSRL-NA、RFS-PPO、RoboFFT-F 和 BORA–FlowMatch 的训练配方 |
| [dexmani_rl/](dexmani_rl/) | **train_*.py** 编排训练，对应的算法模块定义网络与更新规则。**replay.py** 存储 DSRL 转移，**bora_data.py** 读取示教并采集 BORA 在线轨迹 |
| [dexmani_rl/policy/](dexmani_rl/policy/) | FlowMatch 模型、观测编码、动作解码与归一化 |
| [dexmani_rl/common/](dexmani_rl/common/) | BC 加载、仿真环境、GAE、选模与 checkpoint 保存和恢复 |
| [dexmani_rl/core/](dexmani_rl/core/) | 模型无关的 ActionSpec、PolicyAdapter 和 EnvAdapter 契约 |
| [dexmani_rl/policies/](dexmani_rl/policies/) | 具体策略后端的运行时适配器；模型实现仍留在对应后端目录 |
| [dexmani_rl/envs/](dexmani_rl/envs/) | 环境接口适配和物理动作转换 |
| [dexmani_rl/evaluate.py](dexmani_rl/evaluate.py) | 按 seed 配对评估并输出成功率比较 |

架构边界、扩展规则和兼容策略见 [docs/architecture.md](docs/architecture.md)。

## 常用命令

从仓库根目录运行，使用已安装 dexmani_sim、任务资产及评估 seed 池的 policy 环境。将 BC 换成实际权重路径，ALG 可选 dsrl、rfs、robofft 或 bora。每次运行使用新的输出路径。

~~~bash
conda activate policy
BC=/path/to/bc.pt
ALG=rfs

# 使用对应的默认配方训练
python -m dexmani_rl.train_${ALG} --checkpoint "$BC" --output "runs/$ALG"

# 从训练快照恢复
python -m dexmani_rl.train_${ALG} --resume "runs/$ALG/resume.pt" --output "runs/${ALG}_resumed"

# 跨方法比较统一使用 last.pt，并跑完同一组 100 个测试 seed
python -m dexmani_rl.evaluate --rl-checkpoint "runs/$ALG/last.pt" --episodes 100 --output "runs/${ALG}_eval.json"
~~~

单独评估 BC 时，将评估命令中的 rl-checkpoint 参数换为 checkpoint 参数。BORA 默认读取任务的成功示教，可用 demonstrations 参数指定含 episode_*.h5 的目录。
