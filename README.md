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
| [configs/methods/](configs/methods/) | 四种方法的默认实验配方 |
| [dexmani_rl/methods/](dexmani_rl/methods/) | 各方法的算法、训练、评估与专属数据。DSRL 管理转移回放，BORA 管理示教与在线采集 |
| [dexmani_rl/policies/flowmatch/](dexmani_rl/policies/flowmatch/) | FlowMatch 模型、BC 加载、RL 运行时与推理适配器 |
| [dexmani_rl/core/](dexmani_rl/core/) | 模型无关的 ActionSpec、PolicyAdapter 和 EnvAdapter 契约 |
| [dexmani_rl/envs/](dexmani_rl/envs/) | 环境接口适配和物理动作转换 |
| [dexmani_rl/evaluation/](dexmani_rl/evaluation/) | 配对评估统计 |

架构边界、扩展规则和验证方式见 [docs/architecture.md](docs/architecture.md)。

## 安装

在 Python 3.10 的 policy 环境中，从仓库根目录执行：

~~~bash
python -m pip install -e .
~~~

FlowMatch 的点云编码使用 PyTorch3D 的 FPS 和 kNN 算子。仿真训练和评估还需要 dexmani_sim，以及与 BC checkpoint 匹配的任务资产和 seed 池。

## 常用命令

从仓库根目录运行，使用已安装 dexmani_sim、任务资产及评估 seed 池的 policy 环境。将 BC 换成实际权重路径，ALG 可选 dsrl、rfs、robofft 或 bora。每次运行使用新的输出路径。

~~~bash
conda activate policy
BC=/path/to/bc.pt
ALG=rfs

# 使用对应的默认配方训练
python -m dexmani_rl.methods.${ALG}.train --checkpoint "$BC" --output "runs/$ALG"

# 从训练快照恢复
python -m dexmani_rl.methods.${ALG}.train --resume "runs/$ALG/resume.pt" --output "runs/${ALG}_resumed"

# 跨方法比较统一使用 last.pt，并跑完同一组 100 个测试 seed
python -m dexmani_rl.methods.${ALG}.evaluate --rl-checkpoint "runs/$ALG/last.pt" --episodes 100 --output "runs/${ALG}_eval.json"

# 单独评估原始 BC 权重
python -m dexmani_rl.methods.${ALG}.evaluate --checkpoint "$BC" --episodes 100 --output "runs/bc_eval.json"
~~~

BORA 默认读取任务的成功示教，可用 `--demonstrations` 参数指定含 episode_*.h5 的目录。
