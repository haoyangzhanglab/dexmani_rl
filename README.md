# DexMani RL

冻结 FlowMatch，通过 DSRL-NA 学习初始噪声分布。只实现在线后训练：双动作 Q、双噪声 Q 蒸馏、squashed Gaussian actor 和自动熵温度。基础策略、观测编码器及 normalizer 全部冻结。

## 来源与结构

- [DSRL 论文](https://arxiv.org/html/2506.15799v2)、[官方实现](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d311b5c42fe571ff3eeb72f622ca616a149d/stable_baselines3/dsrl/dsrl.py)。按其更新方程独立实现 PyTorch 训练循环，不依赖 SB3 或 DPPO。
- `policy/` 的 FlowMatch 推理依赖来自 `dexmani_policy` 提交 `8031c85e99a11e7b93451dedbfa291fc2eaf07ca`。保留模块权重名称，仅保留当前 FlowMatch 的推理路径；点云采样固定为确定性 FPS，视觉编码只返回场景融合使用的 patch tokens。
- 仿真对接检查基于 `dexmani_sim` 提交 `64abf055af5ceca1e86294a63776c1eb77ea7e92`；未修改两个相邻仓库。

核心模块直接按研究流程组织：

| 模块 | 职责 |
| --- | --- |
| `train.py` | 采集、更新、选模、恢复与累计时间预算 |
| `evaluate.py` | 单回合 rollout、按 seed 配对评估与统计 |
| `dsrl.py` | actor、双动作 Q、双噪声 Q 与更新方程 |
| `replay.py` | CPU replay，保存冻结特征和动作块转移 |
| `env.py` | `dexmani_sim` 生命周期、观测历史与动作块执行 |
| `base_policy.py` | checkpoint 加载、冻结特征和动作空间映射 |

`policy/` 只保留 8 个实现模块，不再沿用预训练仓库的多层目录和基类：

```text
policy/
├── flowmatch.py   # 观测归一化、完整噪声窗口、Euler 积分与动作切片
├── dit.py         # DiTX、时间嵌入、factorized arm/hand action tokens
├── encoder.py     # 场景 token 与关节历史编码
├── pointcloud.py  # FPS、局部几何和空间注意力
├── vision.py      # DINO / DINOv3 / SigLIP 与图像处理
├── fusion.py      # 相机投影与 RGB/点云融合
├── normalizer.py  # checkpoint 中的仿射归一化
└── preprocess.py  # 原始仿真观测的形状、类型与张量布局
```

`ChunkEnv` 独自维护、补齐并堆叠最近的观测；预处理直接接收 `[T,...]`。冻结编码器产生完整 context 和汇总后的 RL state：actor 从 state 采样噪声，FlowMatch 用 context 和噪声生成动作，`FrozenPolicy` 裁剪物理限位并转换到动作 Q 的归一化空间。执行结果和冻结特征进入 replay，只有 DSRL 网络参与更新。

带相机标定的 RGB 保持原始分辨率，缩放只在 `vision.py` 的 `ImageProcessor` 中进行，并同步生成半像素投影变换。与上游 FlowMatch 数据集约束一致，checkpoint 的 `dataset.rgb_preprocess_size`、`rgb_random_crop_size`、`rgb_color_aug` 必须未设置或为 `null`；图像尺寸通过 `agent.rgb_backbone_config.image_size` 配置。

保留 checkpoint 的模块属性和 `state_dict` 参数名，标准/factorized 解码与三种视觉骨干（含 LoRA）沿用当前配置。

## 环境

已验证：Python 3.10、PyTorch 2.4.1+cu124、torchvision 0.19.1、NumPy 1.26.4、SciPy 1.15.3、SAPIEN 3.0.0b1、mplib 0.2.1、PyTorch3D 0.7.8、transformers 4.48.0、timm 1.0.22，RTX 4090 24 GB。现有 conda `policy` 环境可直接运行。

```bash
conda activate policy
python -m pip install -e . --no-deps
```

上面的 editable 安装是可选步骤；在本仓库根目录可直接运行模块。`dexmani_sim` 必须已安装并具备任务资产与评估种子池，PyTorch3D 必须与 PyTorch/CUDA 匹配；这些环境组件不由本项目自动重装。运行时无需安装 `dexmani_policy`。LoRA 视觉 checkpoint 额外需要 `peft`。

## 训练与评估

需要当前 FlowMatch 格式的真实 BC checkpoint，内含 `effective_config`、`model_state` 和可选 `ema_model_state`。EMA/raw 默认遵循 checkpoint 配置；缺失请求的权重时显式报错。推理恢复不下载外部视觉权重、不读取示教集，也不重新拟合 normalizer。

以下 `PATH_TO_BC.pt` 为待提供的权重路径，命令用于正式实验，尚未执行性能训练：

```bash
python -m dexmani_rl.train --checkpoint PATH_TO_BC.pt --config configs/dsrl.yaml --output runs/task_seed42
python -m dexmani_rl.train --resume runs/task_seed42/resume.pt --output runs/task_seed42_resume
python -m dexmani_rl.evaluate --dsrl-checkpoint runs/task_seed42_resume/best.pt --output runs/comparison.json
python -m dexmani_rl.evaluate --checkpoint PATH_TO_BC.pt --output runs/baseline.json
```

所有入口支持 `--device`。新训练可指定 `--config` 和 `--weights raw/ema`；恢复直接读取 `resume.pt` 中的基础权重路径、权重选择和训练配置。DSRL 评估同样从导出的文件读取基础策略信息，单独评估 BC 才使用 `--checkpoint` 和可选的 `--weights`。保留记录的基础权重文件；输出目录必须不存在。

训练输出的 `evaluation.json` 用 `dsrl_checkpoint` 记录实际评估的文件名，相对于当前输出目录；已有 RL 更新但未产生选模 checkpoint 时使用 `last.pt`。若预算在首次 RL 更新前耗尽，只评估基础策略，该字段为 `null`。恢复从新回合开始，不恢复物理场景中途状态。只加载自己信任的训练文件。

默认单环境，3×2048 MLP，Tanh＋LayerNorm，batch 256，Adam 3e-4，gamma 0.99，tau 0.005，噪声幅度 1.5。每动作块更新动作 Q/actor 20 次、噪声 Q 10 次，前向分批 64。先以标准高斯噪声采集 24,000 控制步，总上限 200,000 步。总预算 8 GPU 小时（单 GPU 进程墙钟），训练及选模最多 7 小时，剩余用于测试；恢复累计已用时间。每个动作块开始前检查预算，reset 若已耗尽训练时间则停止采集和更新；单次动作块、更新、环境创建和保存可能小幅超过截止点。

每 25 次回合尝试重建环境并保存 `resume.pt`，其中包含完整 DSRL 网络、优化器、replay、Torch 与环境种子 RNG、选模进度及累计预算。`last.pt` 和 `best.pt` 只保存用于推理的 actor 与配置；历史最佳 actor 在恢复后的最终评估前也会导出。最终评估使用独立的最佳 actor，恢复文件始终保存最近的训练状态。

选模、训练结束评估和独立评估共用同一 rollout。评估按 seed 依次运行所需策略，每个环境最多执行 25 个回合；BC/DSRL 配对时每批 12 个 seed。每个回合重设环境与噪声 RNG，只有该 seed 的所有策略都完成才记录结果，预算截止时舍弃未完成的整对。统计要求两组 seed 按相同顺序完整对齐，缺失或错配会显式报错。

`config.json` 记录训练配置、基础权重路径与 EMA/raw 选择、策略配置、PyTorch 版本和评估种子。`metrics.jsonl` 记录每块损失和被环境拒绝的 reset；`evaluation.json` 保存所用 checkpoint、配对回合和统计。源码来源版本见上文。

## 研究语义

- 完整噪声窗口为 `[B,H,19]`，默认 H=16；从索引 `n_obs_steps-1` 开始执行 K=8 个动作。时间积分为均匀 Euler，0→1。BC 配置中的训练期时间采样和 FPS 随机化参数不进入冻结推理。
- 19 维绝对关节目标，弧度制，环境关节限位裁剪后再按 checkpoint normalizer 进入动作 Q。基线、replay、TD target 和蒸馏使用同一映射。
- actor 产生 `u=tanh(z)`，向 flow 输入 `w=b*u`；噪声 Q 接收 u。标准高斯 w 用于基线、warmup 和蒸馏，蒸馏输入为 w/b，不裁剪。log-prob 在 u 空间计算，target entropy=0。
- 动作 Q 的 target 含下一状态的噪声策略熵项；逐个 critic 蒸馏当前动作 Q；actor 只通过噪声 Q 反传，不通过 flow。
- 每个控制步奖励为 `float(info['success'])`，使用成功保持判定，不使用环境进度奖励、不加 reward offset。块内奖励求和，每块折扣一次。真实成功/失败不 bootstrap；纯超时保留 bootstrap，并使用超时前观测。块内终止立即停止，仍保存原计划完整动作块。
- 每帧场景 token 均值与按 checkpoint 配置处理的关节历史作为 RL 状态；`joint_state: identity` 直接保留关节输入。完整冻结 token 供 flow 解码。replay 默认 CPU float32，20,000 个块，默认模型约需 8 GB RAM。该视觉特征设计、单环境、窗口长度和环境终止处理是本任务适配，与官方 Robomimic 配方存在差异。
- 训练 seed=42；评估池以 1024 打乱，前100个作测试，后20个作选模；训练排除全部池内种子。选模间隔1000动作块，在回合边界触发，首次 RL 更新前不选模或评估 actor。初始化阶段使用原始高斯噪声策略作为基线，与[社区 DSRL-NA 实现](https://github.com/arayabrain/dsrl_pi0_na/blob/afcb58f2bd95a2e3771db359b8f98eb8f4f9ebee/examples/train_utils_aloha_sim_na.py#L287)一致。最终使用同种子、同窗口、同采样步数的随机策略配对评估，无 best-of-N；预算不足时仅报告实际完成的配对数。
- 统计包含成功次数、成功率差与 `paired_conservative_95_ci`。配对差值为“仅 DSRL 成功”的概率减去“仅 BC 成功”的概率：对两者分别调用 [SciPy Clopper–Pearson 精确区间](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)，各取97.5%置信水平，再用 Bonferroni 合成至少95%覆盖率的保守区间。单对样本或所有差值相同也保留非零区间；这些区间只反映评估回合的不确定性，单训练 seed 不能说明跨训练种子的稳定性。稀疏奖励下若基础策略没有成功探索，训练可能无法改善。

## 已执行验证

```bash
python -m pytest -q
python -m mypy --ignore-missing-imports --follow-imports=silent dexmani_rl
PYTHONPATH=. python tests/validate_integration.py --output runs/alignment_integration_01 --reference-root ../dexmani_policy
python -m dexmani_rl.train --checkpoint runs/alignment_integration_01/fixture.pt --config runs/alignment_integration_01/smoke.yaml --output runs/alignment_train_smoke_01
python -m dexmani_rl.train --resume runs/architecture_train_smoke_01/resume.pt --output runs/alignment_resume_smoke_01
```

现有输出目录不能重复使用。GPU 检查在工具审批允许的沙箱外执行；沙箱内 CUDA 不可用。

35 项 CPU 测试、16 个源码文件的 mypy 检查及 wheel 构建均通过。开发检查依赖由 `pyproject.toml` 的 `test` extra 声明。

CPU 测试覆盖 SAC log-prob 及梯度、非有限梯度中止、终止/超时 TD mask、网络更新、replay 恢复、初始历史补齐与块内终止观测、RGB/点云张量化、标定投影与 checkpoint 图像配置约束、关节 identity/limits 归一化和配对区间。评估测试检查环境复用、逐 seed 的噪声、截止时整对舍弃、配对错位报错及资源关闭。入口回归执行真实 DSRL 更新和保存加载，并验证 reset 耗尽预算、warmup、选模间隔恢复、最佳 actor 导出、独立评估及最近训练状态的保留。

GPU 集成检查采用随机初始化的测试权重（小 DiTX、默认 DINO），与原仓库逐项对比观测编码与固定噪声输出，检查非零速度场下的积分、真实仿真交互、全部 RL 更新、基础参数冻结与保存加载。标准和 factorized DiTX 的权重加载与有/无 KV 缓存输出另经 CPU 对照。三种视觉骨干的 freeze/full/LoRA 共9种配置，其权重加载、图像变换与 patch tokens 均与清理前逐元素一致。恢复检查使用已有的当前格式产物，核对模型、四个优化器、replay 和 RNG。交付前已检查源码差异、语法和空白。

**这些测试权重不是 BC checkpoint。** 短程入口仅执行8个控制步的训练；随机策略评估结果不构成算法性能复现。真实预训练 checkpoint 尚未提供，单任务成功率提升尚未验证。
