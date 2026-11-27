# DexMani RL

本仓库在 `dexmani_sim` 中研究 FlowMatch 后训练。DSRL-NA 学习初始噪声，RFS-PPO 联合学习初始噪声与动作残差，RoboFFT-F 微调 action decoder。这三种方法均冻结视觉编码器。BORA 从离线示教联合学习策略和逐动作价值，再冻结离线策略进行两轮在线残差学习。四种方法均使用 checkpoint 的 normalizer 和 FP32。当前的集成验证使用测试权重，任务级预训练 BC 的成功率实验待验证。

## 来源与结构

- DSRL-NA 的更新方程与官方实现对照。[论文](https://arxiv.org/html/2506.15799v2) · [官方实现](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d311b5c42fe571ff3eeb72f622ca616a149d/stable_baselines3/dsrl/dsrl.py)
- RFS-PPO 按论文实现在 xArm7+XHand 仿真中的在线联合调制和 PPO，使用冻结视觉特征。[论文](https://arxiv.org/html/2602.01789v3) · [项目主页](https://weirdlabuw.github.io/rfs/)
- 冻结推理代码来自 `dexmani_policy@8031c85e99a11e7b93451dedbfa291fc2eaf07ca`，仿真接口基于 `dexmani_sim@64abf055af5ceca1e86294a63776c1eb77ea7e92`。

| 模块 | 职责 |
| --- | --- |
| `train_dsrl.py`、`train_rfs.py`、`train_robofft.py` | 各算法的采集、更新、选模与恢复循环 |
| `training.py` | 通用训练入口与选模规则 |
| `dsrl.py`、`rfs.py`、`robofft.py` | 各算法的网络、采样和更新方程 |
| `evaluate.py` | rollout、同 seed 配对评估及统计 |
| `env.py` | 仿真生命周期、观测历史和动作块执行 |
| `base_policy.py` | BC 加载、冻结特征、动作映射与策略维度 |
| `replay.py` | DSRL 的 CPU float32 replay |
| `bora.py`、`bora_data.py`、`train_bora.py` | BORA 的更新方程、逐帧 HDF5 数据和两阶段训练循环 |
| `checkpoint.py` | 原子保存和训练 RNG 状态 |

`policy/` 包含 FlowMatch、DiTX、观测编码器、点云、视觉、融合、归一化及预处理模块。标准/factorized DiTX、DINO/DINOv3/SigLIP 和 LoRA 的 checkpoint 权重名称与配置保持支持。DSRL 的优化器按 `actor`、`critic`、`noise_critic`、`alpha` 命名保存。RFS 只接收状态、噪声、动作维度与配置/device，不持有基础策略。

观测历史由 `ChunkEnv` 补齐和堆叠。冻结编码器输出完整 flow context，以及每帧场景 token 均值和关节历史组成的 RL state。DSRL 保存冻结 context/state 到 replay。RFS 保存当轮状态、原始联合采样、log-prob、价值与转移，不使用 replay。RoboFFT 缓存完整 flow context、匹配噪声、时间步和旧 surrogate。更新复用冻结的视觉特征。

## 环境与 BC 权重

使用现有 conda `policy` 环境：Python 3.10、PyTorch 2.4.1+cu124、torchvision 0.19.1、NumPy 1.26.4、SciPy 1.15.3、SAPIEN 3.0.0b1、mplib 0.2.1、PyTorch3D 0.7.8、transformers 4.48.0、timm 1.0.22。GPU 为 RTX 4090 24 GB。

```bash
conda activate policy
# 可选。在仓库根目录也可直接运行模块
python -m pip install -e . --no-deps
```

源码运行和 editable 安装可从任意工作目录读取仓库默认配方。独立安装 wheel 时，通过 `--config` 指定 YAML。

`dexmani_sim` 需已安装并具备任务资产和评估种子池。PyTorch3D 需与 PyTorch/CUDA 匹配。运行不依赖安装 `dexmani_policy`。LoRA 权重额外需要 `peft`。

BC checkpoint 包含 `effective_config`、`model_state` 和可选 `ema_model_state`，动作键为 `action`，输出19维绝对关节目标（弧度）。EMA/raw 默认遵循 checkpoint 配置，指定权重缺失时显式失败。加载使用原始参数名和 normalizer，不下载视觉权重、不读取示教集或重新拟合统计量。

带标定的 RGB 保持原始分辨率。`dataset.rgb_preprocess_size`、`rgb_random_crop_size`、`rgb_color_aug` 必须未设置或为 `null`。缩放通过 `agent.rgb_backbone_config.image_size` 配置，投影同步使用半像素变换。

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

两者默认 seed=42、gamma=0.99、`total_steps=200000`。DSRL 从 replay 更新时，每批最多解码 64 个样本。RFS 的网络、学习率、batch、GAE、clip、价值系数及梯度裁剪沿用论文报告值。rollout 长度、epochs、零熵系数、残差尺度、初始化与固定学习率是本地选择。

DSRL 的 log-prob 在 tanh 后的 `u` 空间计算，含变换修正，target entropy=0。噪声 Q 接收 `u`。BC、warmup 和蒸馏使用标准高斯 `w`，蒸馏输入 `w/1.5` 不裁剪。动作 Q 的 TD target 包含噪声策略熵项，两个噪声 Q 分别蒸馏对应的当前动作 Q。actor 只通过噪声 Q 反传，不通过 flow。动作 Q/replay 统一使用物理限位裁剪后再归一化的动作。

RFS 在归一化动作空间合成 `a_norm = FM_norm(context,w) + 0.1*u_r`，再反归一化并按机器人关节限位裁剪。`0.1` 不是弧度，示教区间 `[-1,1]` 也不是硬限位。PPO 的 log-prob、ratio 始终在原始联合采样 `[w,u_r]` 上沿全部维度求和，不对物理动作计算概率。使用整轮 advantage 归一化、clipped surrogate 和价值 MSE。flow 只参与采集，无梯度，value 不裁剪。

完整 flow 噪声为 `[B,H,19]`，通常 H=16。从 `n_obs_steps-1` 开始执行 K=8 个动作，具体窗口和控制周期沿用 BC 配置，Euler 时间方向为0→1。DSRL、RFS 和 RoboFFT 共享下述动作块时间尺度。

- `total_steps`、`warmup_steps` 和日志 `steps` 是实际执行的控制步。`rollout_steps`、日志 `chunks`、gamma 和 GAE 的一步是一个动作块。
- 每控制步奖励为 `float(info['success'])`，采用成功保持判定。块内奖励求和，每块折扣一次，不用 `gamma**K`。
- 成功/失败不 bootstrap。纯超时用重置前观测 bootstrap。RFS 和 RoboFFT 在成功、失败和超时处均切断跨回合 GAE。块内终止立即停止执行。
- 达到控制步上限时，RFS 可更新不足长度的 rollout。训练时间耗尽则丢弃未更新样本并记入 `discarded_chunks`。DSRL 在动作块结束时保留已采集的转移，按训练截止时间决定是否启动更新。

## RoboFFT-F

RoboFFT-F 采用官方提交 `a9ca8c8fc7684fc4718b2b7758bcc992eb710e46` 的 Can-state 配方，微调本地 FlowMatch decoder。[论文](https://arxiv.org/html/2609.32236v1) · [官方实现](https://github.com/shshsunny/RoboFFT/tree/a9ca8c8fc7684fc4718b2b7758bcc992eb710e46)

`robofft.py` 包含采样、matching surrogate、PPO、value network 和 reward scaling。`train_robofft.py` 负责采集、更新与保存。加载沿用本仓库 FlowMatch，不依赖安装或修改 `dexmani_policy`。冻结 observation encoder，只复制并解冻 DiTX decoder，保持 `eval()` 禁用 dropout。critic 输入复用冻结特征与关节历史。完整 `[H,19]` 归一化生成样本用于 matching，从 `n_obs_steps-1` 开始执行 K 步。物理关节限位不会反向覆盖 matching target。

以 release 的公式构造 matching surrogate。`x_t=t*a+(1-t)*eps`，目标速度 `a-eps`。`logp=-mean_H,D((v-target)^2)`。旧、新 logp 分别 clamp 到 `[-1,1]`，每个 MC 独立计算 `exp(8*(new-old))` 并做 PPO，最后平均。它与论文先平均 MC loss 再 exp 的写法不同。H16/D19 仍使用 mean(H,D)、alpha=8。KL、ratio 和 clip fraction 的原始诊断采用未乘 alpha 的 ratio，不能解释为真实 likelihood/KL。

采样在同一动作块的 Euler 子步间复用 DiTX context K/V。训练从 Gaussian 初始噪声出发，每个 Euler 步先裁剪归一化动作到 `[-1,1]`，再加 `0.25*sqrt(dt)` Gaussian。末步再裁剪。评估关闭逐步噪声，保留初始 Gaussian。积分步数沿用 BC checkpoint（通常10），forward t 在对应离散网格均匀抽样。每次采集缓存 eps、t 和旧 logp。首个更新 minibatch 在执行优化器前校验逐样本的原始 logp。

| 项目 | 默认值及约定 |
| --- | --- |
| Actor / critic | 原 DiTX / 256宽 Mish residual MLP |
| AdamW LR | actor最高 `2e-6`、最低 `1e-6`，critic `1e-3`。weight decay=0 |
| 调度 | 151个外层 iteration，10步 warmup 后 cosine。critic恒定LR |
| 更新 | PPO clip `.02`，target surrogate KL `.01`，MC1，10 epochs，每条transition重复10次 |
| 优势与价值 | 每个逻辑minibatch标准化优势。value loss=`.5*MSE`，系数再乘`.5`。不裁剪梯度或value |
| 时间尺度 | gamma `.999`、GAE `.95`，按动作块。成功/失败不bootstrap，纯timeout用末观测bootstrap |
| 采集规模 | 单环境，512块/rollout，逻辑batch256、microbatch16。冻结context存CPU |
| 墙钟限制 | 训练22小时、含最终评估24小时。在采集块/完整更新/评估边界检查 |

外层 iteration 0、10、20…用于选模。iteration 1 只更新 critic，iteration 2 起更新 actor 并推进其学习率调度。critic 使用固定学习率。KL 超阈值的 minibatch 仍完成更新，随后停止。每条转移的 10 次重复复用采集时的 eps 和 t。奖励缩放使用官方的跨 rollout 折扣 return、回合切断和 RMS 合并分母 `total_count-1`。奖励除以标准差后裁剪到 `[-10,10]`，评估报告原始奖励。

本地适配包括冻结多模态编码器、DiTX actor、critic输入、单环境batch规模、按BC窗口执行，以及正确处理timeout与环境重置。没有引入anchor、BC regularization、熵或noise network。归一化采样裁剪与机器人关节限位分别记录，logp饱和也单独记录。饱和造成零梯度时不会自动改变算法。

## BORA–FlowMatch

BORA 按结果页对应的五任务论文 v2 移植到 xArm7 + XHand 仿真。作者仓库公开了论文、图表和视频；训练代码、五任务轨迹和 BORA 专用权重未在该快照发布。网页的 86%（标准物体）和 70%（未见物体）来自作者的真机评估。这里保留 FlowMatch、原始 BC 基线和本地任务的成功判据。[结果页](https://chenzhongxi-sjtu.github.io/BORA/bora_results.html) · [论文 v2](https://arxiv.org/html/2605.30226v2) · [作者仓库快照](https://github.com/ChenZhongxi-sjtu/BORA/tree/b8cdf0c915f6885c8ef537dc9d0088468f9ad1f4)

**离线。** 直接从 HDF5 读取同一控制帧的观测和将要执行的绝对关节目标。示教与采集数据共用滑动窗口。每个相对位置通过可学习的查询汇聚观测 tokens，Q 额外读取该位置的动作。Q 的 TD target 在窗内移到下一位置，完整窗口的末项使用相邻控制帧的第一个 V。成功或失败的终止动作参与训练，后续动作屏蔽。纯超时使用真实末观测 bootstrap。Q 的 Bellman 残差、V 的 expectile 和 actor 损失均按每个窗口的有效步数取均值，再平均整个 batch。

先做动作 BC 和 critic 更新，再加入逐动作 clipped PPO。使用停止梯度的 Q−V 向窗内前序位置递推优势。原始优势过门限后，归一化并裁剪到 ±2。现有 DiTX 在完整 horizon 上执行可微 Euler 积分，积分步数来自 checkpoint。每个控制步以解码结果为均值、固定标准差的对角 Gaussian 对示教动作打分，沿 19 个关节求和。每个 batch 固定 flow 初始噪声及更新前的 log-prob，此条件 Gaussian 是 FlowMatch 的代理分布。离线训练更新 decoder 和观测编码器，RGB 主干遵循 checkpoint 的 `freeze/full/lora` 设置，保持 `eval()` 关闭 dropout。原论文使用少步 Consistency Policy 与 VLM tokens。[论文方法与附录 A、B](https://arxiv.org/html/2605.30226v2#S3) · [Consistency Policy](https://consistency-policy.github.io/)

PPO 根据优势符号在 log 空间选择相应的裁剪边界，再计算带优势权重的目标值。日志的 `mean_log_ratio` 记录有效位置上的平均 log-ratio。每次 AdamW 写入前检查损失与梯度的有限性。

**在线。** 冻结离线 actor，继承 Q/V。残差 MLP 输入 token 均值、当前关节状态和基底动作块。第一轮修正离线策略，第二轮以第一轮冻结的组合策略及其实际混合系数为基底。每条轨迹采集结束后，从示教和累计在线轨迹各采半个 batch 更新 critic 与当前残差。

保守改进项 `relu(Q_base + margin − Q_final)` 的两个候选动作都经过与仿真执行相同的弧度转换和关节限位，再归一化给 Q 打分。默认 margin 为零，两个候选动作产生相同命令时，该项为零。BC 用 replay 的实际执行动作监督限位前的预测，使越界时仍有指向可行目标的梯度。残差自身受 L2 约束。跨轮更新采用 v2 Algorithm 1，三项损失采用 v3 式 (6)。v2 表 3 另列出 critic regularization 和 actor anchor 的系数，缺少对应定义。[v2 Algorithm 1](https://arxiv.org/html/2605.30226v2#A2.SS4) · [v3 在线目标](https://arxiv.org/html/2605.30226v3#S3.SS3)

| 配方 | 数值及单位 |
| --- | --- |
| 离线更新 | 70,000 个 batch，前 40,000 个只对 actor 做 BC。单卡 batch 8 |
| 在线更新 | 两轮，每轮 10 条轨迹，每条完成后 750 个混合 batch，batch 64 |
| 本地选择 | expectile 0.7，GAE λ 0.95，Gaussian 标准差 1（归一化关节空间），PPO clip 0.2，每批 2 次 actor 更新 |
| 在线调度 | 每轮从更新 0 起计。残差混合系数 0.2→0.75 / 4,000 步，BC 系数 2→1 / 10,000 步 |
| 仿真奖励 | 每控制帧 −0.01，成功终止 +1，首次接管 −0.25，操作员确认恢复 +0.25 |

默认自主采集。接入人工控制时，从 Python 调用 `dexmani_rl.train_bora.main(correction=choose_target, confirm_recovery=confirm)`，命令行参数与模块入口相同。

- 执行前调用 `choose_target(raw_obs, proposed_joint_target)`。返回人工关节目标（弧度）表示接管，返回 `None` 表示自主执行。
- 人工指令执行后调用 `confirm(next_raw_obs, terminated, truncated, info)`。返回 `True` 时，将恢复奖励记在刚执行的动作上。每段连续接管最多记一次，成功、失败与超时的最后一步均可确认。省略该回调时只记录接管。

在 `policy` 环境运行。训练默认读取 `dexmani_sim.DATA_DIR/episodes/<checkpoint任务>/succeeded_episode`；也可传 `--demonstrations` 指向含 `episode_*.h5` 的目录。训练排除整个评估 seed 池。以下示例分两次运行同一配方，默认 `--through round2` 可一次跑完。

```bash
python -m dexmani_rl.train_bora --checkpoint PATH_TO_BC.pt --config configs/bora.yaml --through round1 --output runs/bora_round1
python -m dexmani_rl.train_bora --resume runs/bora_round1/resume.pt --output runs/bora_round2
python -m dexmani_rl.evaluate --rl-checkpoint runs/bora_round2/last.pt --episodes 100 --output runs/bora_comparison.json
```

`--through offline` 可单独训练离线策略。`offline.pt` 和 `last.pt` 用于独立评估，评估器从 actor 读取已训练的轮次。评估离线 actor 时，`baseline` 和 `steered` 分别记录原生 BC 和离线策略。评估在线 actor 时，`steered` 记录当前组合策略，`offline` 记录同一批 seeds 上关闭残差的离线策略。

`resume.pt` 保存模型、优化器、训练进度、RNG 和 HDF5 路径。恢复时需要保留源示教、BC 权重及原运行的 `online/` 目录。每次调用使用新的输出目录，`config.json` 和 `metrics.jsonl` 记录来源、配置及实际更新。

## DSRL、RFS 和 RoboFFT 的训练、恢复与评估

以下命令从仓库根目录运行。用真实预训练权重替换 `PATH_TO_BC.pt`，并为每次运行指定新输出路径。

```bash
python -m dexmani_rl.train_dsrl --checkpoint PATH_TO_BC.pt --config configs/dsrl.yaml --output runs/dsrl_task_seed42
python -m dexmani_rl.train_rfs --checkpoint PATH_TO_BC.pt --config configs/rfs.yaml --output runs/rfs_task_seed42
python -m dexmani_rl.train_robofft --checkpoint PATH_TO_BC.pt --config configs/robofft.yaml --output runs/robofft_task_seed42
python -m dexmani_rl.train_robofft --resume runs/robofft_task_seed42/resume.pt --output runs/robofft_task_seed42_resume
python -m dexmani_rl.evaluate --rl-checkpoint runs/robofft_task_seed42/best.pt --episodes 100 --output runs/robofft_comparison.json
python -m dexmani_rl.evaluate --checkpoint PATH_TO_BC.pt --output runs/baseline.json
```

恢复时使用对应算法的训练入口和 `resume.pt`。独立 RL 评估使用 `best.pt` 或 `last.pt`。所有入口支持 `--device`，新训练和独立 BC 评估支持 `--weights raw/ema`。恢复和独立 RL 评估沿用 checkpoint 所记录的配置及 BC 权重路径。

评估池排序后用种子 1024 打乱，前 100 个用于测试，后 20 个用于选模。训练排除整个评估池。每种策略在每个 seed 上单独建场，首次 reset 后开始回合，并用同一个 seed 初始化策略 RNG。每组完整跑完才计入结果。`comparison` 为原生 BC 与所选 actor 的配对比较。RoboFFT 额外用 `sampler_baseline` 记录采用同一采样器的初始 decoder，并报告 `comparison_vs_sampler`。配对成功率差的置信区间由两个不一致成功结果的 97.5% Clopper–Pearson 区间合成。

各算法按配置执行定期选模，结束时在剩余预算内补评尚未经完整选模的最新 actor。完整选模集的成功率严格提高才替换 `best.pt`，同分保留已有最佳。恢复时沿用快照中的选模结果和进度。无完整选模结果时使用 `last.pt`。DSRL 和 RFS 在首次 actor 更新前仅评估 BC，RoboFFT 可用初始 decoder 参加选模。

`last.pt` 保存最新 actor。`resume.pt` 保存训练状态、RNG、进度和累计墙钟。DSRL 另保存 replay，RoboFFT 另保存 actor 调度器和 reward scaler。使用含当前训练字段的快照恢复，从新回合开始。`config.json` 记录配置和权重来源，`metrics.jsonl` 记录更新及选模阶段与完成状态，`evaluation.json` 记录最终评估及实际使用的 checkpoint。

## 验证

在 `policy` 环境从仓库根目录运行，复跑时为每个命令选用新的输出位置。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -m mypy --no-incremental --cache-dir=/dev/null --ignore-missing-imports --follow-imports=silent dexmani_rl
for algorithm in dsrl rfs robofft; do
    PYTHONPATH=. python tests/validate_integration.py --algorithm "$algorithm" --output "runs/${algorithm}_integration_new" --reference-root ../dexmani_policy
done
PYTHONPATH=. python tests/validate_bora_integration.py --output runs/bora_integration_new
```

BORA 的 CUDA 验证用小型合成 FlowMatch 权重读取真实 `multi_grasp` 示教，完成 BC、有效 PPO、两轮自主采集和跨轮恢复，再将原生 BC、离线和在线策略放入相同的两个测试场景。测试配置将每回合限为 16 个控制步。150 条示教记录了 30,183 个动作，其中 28,735 个位于首次终止及以前；示教 seed 与 200 个评估 seed 无交集。

GPU 集成测试使用合成权重，检查仿真采集、冻结权重、恢复后的更新及与原生 FlowMatch 的推理对照。RoboFFT 的合成收缩速度场用于在官方 logp 下限 `-1` 下检验非零梯度。配对评估回归用同一策略在 seed `42039` 上逐项比较初始观测、首个动作块和回合结果。

官方数值对照使用 `test` extra 中的 Python 依赖，直接加载 `--reference-root` 下的官方源码。以下命令固定参考提交。验证覆盖 MC1/2 的 logp、梯度、一次 AdamW 更新、训练与评估采样、奖励缩放、critic 和 151 轮学习率调度。

```bash
git clone https://github.com/shshsunny/RoboFFT.git ../RoboFFT
git -C ../RoboFFT checkout --detach a9ca8c8fc7684fc4718b2b7758bcc992eb710e46
PYTHONDONTWRITEBYTECODE=1 python tests/validate_robofft_reference.py --reference-root ../RoboFFT/robofft-f --output runs/robofft_reference_new.json
```
