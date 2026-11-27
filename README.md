# DexMani RL

冻结 FlowMatch，在 `dexmani_sim` 中比较两个在线后训练 baseline：DSRL-NA 学习初始噪声，RFS-PPO 联合学习初始噪声与动作残差。基础策略、视觉编码器和 normalizer 均冻结，推理统一使用 FP32。当前完成算法与训练链路实现，尚缺真实预训练 BC checkpoint，未验证成功率提升。

## 来源与结构

- [DSRL 论文](https://arxiv.org/html/2506.15799v2)及[官方实现](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d311b5c42fe571ff3eeb72f622ca616a149d/stable_baselines3/dsrl/dsrl.py)：独立实现更新方程，不依赖 SB3 或 DPPO。
- [RFS 论文](https://arxiv.org/html/2602.01789v3)及[项目主页](https://weirdlabuw.github.io/rfs/)：按论文实现在线联合调制与 PPO；调研时官方代码尚未公开。本仓库适配 xArm7+XHand 和冻结视觉特征，不包含论文的特权状态教师、蒸馏、离线 TD3+BC、六任务环境或真机流程。
- 冻结推理代码来自 `dexmani_policy@8031c85e99a11e7b93451dedbfa291fc2eaf07ca`，仿真接口基于 `dexmani_sim@64abf055af5ceca1e86294a63776c1eb77ea7e92`；相邻两个仓库未修改。

| 模块 | 职责 |
| --- | --- |
| `train_dsrl.py`、`train_rfs.py` | 各自独立的采集、更新、选模、保存与恢复循环 |
| `dsrl.py`、`rfs.py` | 各算法的网络与更新方程 |
| `evaluate.py` | rollout、同 seed 配对评估及统计 |
| `env.py` | 仿真生命周期、观测历史和动作块执行 |
| `base_policy.py` | BC 加载、冻结特征、动作映射与策略维度 |
| `replay.py` | DSRL 的 CPU float32 replay |
| `checkpoint.py` | 仅 `save`、`capture_rng`、`restore_rng` 三个函数 |

`policy/` 保留 8 个实际推理模块：`flowmatch`、`dit`、`encoder`、`pointcloud`、`vision`、`fusion`、`normalizer`、`preprocess`。标准/factorized DiTX、DINO/DINOv3/SigLIP 和 LoRA 的 checkpoint 权重名称与配置保持支持。DSRL 的优化器按 `actor`、`critic`、`noise_critic`、`alpha` 命名保存；RFS 只接收状态、噪声、动作维度与配置/device，不持有基础策略。没有共用 trainer 或兼容层。

观测历史由 `ChunkEnv` 补齐和堆叠；冻结编码器输出完整 flow context，以及每帧场景 token 均值和关节历史组成的 RL state。DSRL 保存冻结 context/state 到 replay；RFS 保存当轮状态、原始联合采样、log-prob、价值与转移，不使用 replay。两者更新时均不重新编码图像。

## 环境与 BC 权重

使用现有 conda `policy` 环境：Python 3.10、PyTorch 2.4.1+cu124、torchvision 0.19.1、NumPy 1.26.4、SciPy 1.15.3、SAPIEN 3.0.0b1、mplib 0.2.1、PyTorch3D 0.7.8、transformers 4.48.0、timm 1.0.22；GPU 为 RTX 4090 24 GB。

```bash
conda activate policy
# 可选；在仓库根目录也可直接运行模块
python -m pip install -e . --no-deps
```

`dexmani_sim` 需已安装并具备任务资产和评估种子池；PyTorch3D 需与 PyTorch/CUDA 匹配。运行不依赖安装 `dexmani_policy`；LoRA 权重额外需要 `peft`。

BC checkpoint 格式不变：包含 `effective_config`、`model_state` 和可选 `ema_model_state`，动作键为 `action`，输出19维绝对关节目标（弧度）。EMA/raw 默认遵循 checkpoint 配置，指定权重缺失时显式失败。加载使用原始参数名和 normalizer，不下载视觉权重、不读取示教集或重新拟合统计量。

带标定的 RGB 保持原始分辨率；`dataset.rgb_preprocess_size`、`rgb_random_crop_size`、`rgb_color_aug` 必须未设置或为 `null`。缩放通过 `agent.rgb_backbone_config.image_size` 配置，投影同步使用半像素变换。

## 算法与默认配方

| 项目 | DSRL-NA | RFS-PPO |
| --- | --- | --- |
| 调制 | `u=tanh(z)`，flow 噪声 `w=1.5*u` | 联合对角 Gaussian `[w,u_r]`，无 tanh |
| 网络 | 3×2048，Tanh + LayerNorm | actor/value 独立 `[256,128,64]`，ReLU |
| 学习率 | Adam `3e-4` | actor `3e-4`，value `1e-3` |
| 每轮更新 | 动作 Q/actor 20 次，噪声 Q 10 次 | rollout 2048块，10 epochs，batch 1024 |
| 其他配置 | batch 256，tau 0.005，replay 20,000块 | GAE 0.95，PPO clip 0.2，value 系数0.5，entropy 系数0，梯度范数1 |
| 初始化 | 标准高斯 warmup 24,000控制步 | 均值输出层零初始化，逐维可学习 `log_std=0` |
| 选模间隔 | `eval_interval_chunks=1000`，回合边界触发 | `eval_interval_updates=5`，每5次 PPO 更新 |
| 训练/总墙钟预算 | 7 / 8小时 | 23 / 24小时 |

两者默认 seed=42、gamma=0.99、`total_steps=200000`、`decode_batch=64`；完整配置见 `configs/`。RFS 的网络、学习率、batch、GAE、clip、价值系数及梯度裁剪沿用论文报告值；rollout 长度、epochs、零熵系数、残差尺度、初始化与固定学习率是本地选择。

DSRL 的 log-prob 在 tanh 后的 `u` 空间计算，含变换修正，target entropy=0。噪声 Q 接收 `u`；BC、warmup 和蒸馏使用标准高斯 `w`，蒸馏输入 `w/1.5` 不裁剪。动作 Q 的 TD target 包含噪声策略熵项，两个噪声 Q 分别蒸馏对应的当前动作 Q；actor 只通过噪声 Q 反传，不通过 flow。动作 Q/replay 统一使用物理限位裁剪后再归一化的动作。

RFS 在归一化动作空间合成 `a_norm = FM_norm(context,w) + 0.1*u_r`，再反归一化并按机器人关节限位裁剪；`0.1` 不是弧度，示教区间 `[-1,1]` 也不是硬限位。PPO 的 log-prob、ratio 始终在原始联合采样 `[w,u_r]` 上沿全部维度求和，不对物理动作计算概率。使用整轮 advantage 归一化、clipped surrogate 和价值 MSE；flow 只参与采集，无梯度，value 不裁剪。

完整 flow 噪声为 `[B,H,19]`，通常 H=16；从 `n_obs_steps-1` 开始执行 K=8 个动作，具体窗口和控制周期沿用 BC 配置，Euler 时间方向为0→1。两算法共享如下时间尺度：

- `total_steps`、`warmup_steps` 和日志 `steps` 是实际执行的控制步；`rollout_steps`、日志 `chunks`、gamma 和 GAE 的一步是一个动作块。
- 每控制步奖励为 `float(info['success'])`，采用成功保持判定；块内奖励求和，每块折扣一次，不用 `gamma**K`。
- 成功/失败不 bootstrap；纯超时用重置前观测 bootstrap。RFS 在成功、失败和超时处均切断跨回合 GAE。块内终止立即停止执行。
- 达到控制步上限时，RFS 可更新不足长度的 rollout；训练时间耗尽则丢弃未更新样本并记入 `discarded_chunks`。预算在动作块、更新等操作边界检查，单次操作可能略超时。

## 训练、恢复与评估

以下 `PATH_TO_BC.pt` 需要替换为真实预训练权重；正式性能实验尚未执行。

```bash
python -m dexmani_rl.train_dsrl --checkpoint PATH_TO_BC.pt --config configs/dsrl.yaml --output runs/dsrl_task_seed42
python -m dexmani_rl.train_rfs --checkpoint PATH_TO_BC.pt --config configs/rfs.yaml --output runs/rfs_task_seed42
python -m dexmani_rl.train_rfs --resume runs/rfs_task_seed42/resume.pt --output runs/rfs_task_seed42_resume
python -m dexmani_rl.evaluate --rl-checkpoint runs/rfs_task_seed42_resume/best.pt --episodes 100 --output runs/rfs_comparison.json
python -m dexmani_rl.evaluate --checkpoint PATH_TO_BC.pt --output runs/baseline.json
```

DSRL 恢复同样使用 `train_dsrl --resume`；两种 RL 评估统一使用 `--rl-checkpoint`。所有入口支持 `--device`；新训练和独立 BC 评估可用 `--weights raw/ema`。恢复使用保存的配置、基础权重路径和权重选择，需保留该 BC 文件；输出目录/文件必须尚不存在。

评估池排序后以1024打乱，前100个作为测试，后20个选模，训练排除整个评估池。三组种子写入 RL checkpoint，恢复和独立 RL 评估沿用保存结果。BC 独立评估从当前池划分；`--episodes` 默认完整测试池，显式值须为正且不超过池长度。

配对评估对每个 seed 重设环境与策略 RNG，BC 和调制策略采用相同窗口、积分步数与采样预算，无 best-of-N。每环境最多25次回合尝试，双策略每批12个 seed；只提交完整配对，截止时舍弃未完成的整对。`comparison` 报告成功数、成功率差和 `paired_conservative_95_ci`：两个不一致结果的97.5% Clopper–Pearson 区间经 Bonferroni 合成至少95%覆盖区间；它不代表跨训练 seed 的稳定性。

`best.pt` 按完整选模集成功率保存 actor 与重建元数据；`last.pt` 保存最新 actor。`resume.pt` 保存最新模型、优化器、RNG、计数、选模进度、历史最佳 actor 和累计墙钟，DSRL 另保存 replay。DSRL 每25次回合尝试保存恢复点，RFS 每次 PPO 更新后保存；恢复从新回合开始，不恢复物理场景或旧 RFS rollout。最终评估使用独立的最佳 actor，保留最新训练状态；已有更新但无最佳模型时评估 `last.pt`，首次更新前预算耗尽则仅评估 BC。

`config.json` 记录 RL/BC 配置与权重来源；RL checkpoint 用 `algorithm='dsrl'|'rfs'` 标识方法，保存 RL 配置、基础权重路径与选择；BC 配置从基础 checkpoint 读取。`metrics.jsonl` 记录训练及 reset 拒绝，`evaluation.json` 统一输出 `algorithm`、`rl_checkpoint`、`baseline`、`steered`、`comparison`，成功计数字段为 `steered_successes`。训练输出的 `rl_checkpoint` 相对当前输出目录，独立评估为绝对路径；独立 BC 的 `algorithm='bc'`、`rl_checkpoint=null`、`steered=[]`。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -m mypy --no-incremental --cache-dir=/dev/null --ignore-missing-imports --follow-imports=silent dexmani_rl
PYTHONPATH=. python tests/validate_integration.py --algorithm dsrl --output runs/chain_dsrl_integration_02 --reference-root ../dexmani_policy
PYTHONPATH=. python tests/validate_integration.py --algorithm rfs --output runs/chain_rfs_integration_02 --reference-root ../dexmani_policy
```

当前验收：72 项 CPU 测试通过，mypy 检查19个源码文件通过；wheel 构建及包内三个 CLI 模块导入通过。结构清理前后，固定输入与 RNG 下的两算法采样、loss、更新后模型、优化器和 RNG 逐项一致。

GPU 集成脚本生成随机权重 `fixture.pt`、短程配置 `smoke.yaml` 和 `validation.json`；DSRL 完成4次更新，RFS 完成2次更新，两者均在中途保存模型、优化器及 RNG 后重新加载并继续更新，冻结权重不变；非恒等 normalizer、非零时间与观测调制下的推理与原仓库对照通过。正式训练入口输出位于 `runs/chain_dsrl_train_01/`、`runs/chain_rfs_train_01/`；统一评估入口各完成2个 seed 的配对评估，输出为 `runs/chain_dsrl_eval_01.json`、`runs/chain_rfs_eval_01.json`。这些随机权重产物仅验证链路，不用于成功率或复现性能结论。
