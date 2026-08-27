# 流形约束 CNOP：Phase 1 研究范围

## 状态

Phase 1（研究范围界定）于 2026-08-26 完成。本文件记录研究问题简报、初步方法蓝图、范围边界、关键假设以及调研前的反方检查。它是研究规划文件，不代表其中任何方法已经得到验证。

## 主题

将 CNOP 从原始物理场空间中的幅度约束优化，扩展为在数据支持的地球系统状态流形上进行可微、有限幅度的优化。

这里有意使用“数据支持的气候状态流形”（*data-supported climate-state manifold*），而不是“真实地球流形”（*the true Earth manifold*）：diffusion 或其他生成模型学习的是由训练数据和模型结构所表示的分布，而不是所有物理上可能出现的地球状态的完整集合。

## 主要研究问题

**对于一个已经训练好的全球海气预测模型，基于可微生成流形参数化的 CNOP 能否在保持扰动地球系统状态自然性的同时，稳定地优化有限幅度预测响应？**

该问题检验三个相互关联的性质：（i）优化得到的扰动是否仍接近自然支持的气候状态；（ii）梯度是否沿可接受的流形方向传播；（iii）这种方法是否比“先沿原始梯度移动、再进行 diffusion 去噪”的后处理流程更加稳定。

## 候选问题

| 候选问题 | FINER 平均分 | 作用 |
|---|---:|---|
| 可微生成流形 CNOP 能否在保持状态自然性的同时稳定优化有限幅度响应？ | 4.6/5 | 选定的主要研究问题 |
| 与普通 CNOP 相比，diffusion 投影能否产生更自然的 ENSO 前期征兆？ | 3.7/5 | 子问题；需要先定义“自然” |
| latent-space、score-based 和 tangent-space 三种约束中，哪一种最适合 CNOP？ | 4.0/5 | 方法比较子问题 |
| 流形约束能否识别更可信的 ENSO OPR 和敏感区？ | 3.8/5 | 下游科学应用问题 |

## 初步 FINER 评估

| 维度 | 分数 | 理由 |
|---|---:|---|
| Feasible（可行性） | 4/5 | WalkerNet 可微，已有 CNOP 代码可以接入生成先验和约束优化器。 |
| Interesting（有趣性） | 5/5 | 直接处理“数学上有效的扰动”与“自然支持的气候状态”之间的根本差异。 |
| Novel（新颖性） | 4/5 | 将 CNOP、生成式气候状态先验以及流形/切空间优化结合起来，具有较强交叉创新空间；具体新颖性需要 Phase 2 核验。 |
| Ethical（伦理性） | 5/5 | 不需要人体受试者；主要研究诚信风险是把学习到的可行性过度解释为真实物理因果。 |
| Relevant（相关性） | 5/5 | 可能服务于 ENSO 前兆分析、反事实预测和目标观测设计。 |
| **平均分** | **4.6/5** | 足以进入文献与可行性调研。 |

## 研究范围边界

### 纳入范围

- WalkerNet 类已经训练好的全球海气预测模型。
- 月尺度的 TOS、ZOS、TAUU 和 TAUV 场。
- 连续历史窗口（初始采用 WalkerNet 的 12 个月输入窗口），而不是孤立单帧。
- Historical CMIP6 多模式数据；条件允许时采用 source/year holdout。
- 12–18 个月自回归预测中的 ENSO 和 Niño3.4 响应。
- 三类候选约束：latent-space 参数化、基于 score 的自然性正则化、可微切空间投影。
- 与以下方法进行比较：无约束或仅幅度约束的 CNOP、梯度更新后再进行 diffusion 去噪的后处理投影、zero-state local gradient，以及 matched-radius random perturbation。

### 排除范围

- 声称学习到的生成器恢复了完整真实地球系统动力学流形。
- 声称高概率生成状态就是唯一因果机制，或保证一定会在地球上发生。
- 一开始就扩展到所有 SSP 情景或所有气候变量。
- 用单个 ENSO 案例证明普遍优越性。

### 关键假设

1. Historical CMIP6 数据能够提供有用但不完整的状态分布先验。
2. 生成模型能够学习主要的空间、季节和跨变量耦合关系。
3. 自然性必须通过多个独立诊断评估，不能只依赖 diffusion likelihood。
4. 不能因为真实 ENSO 前兆在总体分布中的概率较低，就将其自动排除。
5. 生成器与 WalkerNet 之间共享的数据偏差，需要通过 source/year holdout，以及条件允许时的独立再分析资料进行检验。

## 子问题

1. **优化稳定性：** 与普通 CNOP 和 diffusion 后处理投影相比，可微流形约束 CNOP 能否降低目标函数振荡、回退频率以及不同随机种子之间的方差？
2. **约束形式：** 在计算预算和幅度预算匹配的条件下，latent-space、score-based 和 tangent-space 约束中，哪一种能在自然性与预测响应之间取得最稳定的折中？
3. **科学解释：** 流形约束是否会改变 CNOP 识别出的 ENSO 最优前期征兆和目标观测敏感区？这些模式能否在不同随机种子、数据源和流形模型之间保持稳定？

所有子问题默认继承相同的 WalkerNet、CMIP6 historical、ENSO、12–18 个月、四变量范围。它们的差异仅在于分别强调稳定性、方法比较和科学解释。

## 初步方法蓝图

### 状态表示

首选研究对象是一个轨迹/时间窗口：

\[
X_t = [x_{t-11},\ldots,x_t],
\]

其中每个 \(x_t\) 包含四个全球物理场。与单帧流形相比，这种表示保留了时间连续性、季节位相以及 SST–SSH–风应力之间的耦合结构。

### 条件生成先验

在数据支持的情况下，生成先验应当条件化于日历月份、数据源/模式身份、变量身份、时间顺序以及局地 ENSO 背景状态。无条件生成器可以作为消融实验，但不应成为唯一的科学先验。

### 候选方法形式

#### Latent-space CNOP

令

\[
X=G_\theta(z,c),
\]

优化：

\[
z^* = \arg\max_z J\big(\mathcal{F}(G_\theta(z,c))\big) - \lambda_z R_z(z),
\]

其梯度为：

\[
\nabla_z J = \left(\frac{\partial G_\theta}{\partial z}\right)^\top \nabla_X J.
\]

这是实现“梯度限制在生成器支持方向内”最直接的方式。

#### Score-based 约束 CNOP

利用 score：

\[
s_\theta(X,\sigma)\approx\nabla_X\log p_\theta(X),
\]

在预测目标中加入显式自然性惩罚。这种方式表达的是对高密度状态的偏好，而不是严格的硬流形约束。

#### Tangent-space CNOP

利用生成器 Jacobian 或局部 score 几何结构，近似当前状态处的可接受切空间 \(T_X\mathcal{M}\)，然后先将预测梯度投影到该切空间，再更新扰动。这一形式在概念上最接近真正的内禀流形优化，但计算难度也可能最高。

## 必须评估的维度

### 预测响应

- 远期三个月 Niño3.4 响应和 lead-12 响应。
- 相对于未扰动预测的响应增益。
- 相对于 local-gradient 基线的配对改进。

### 状态自然性

- diffusion 重构误差和/或 score energy。
- latent distance 以及与留出 historical 状态的最近邻距离。
- 单步状态转移的合理性。
- TOS/ZOS/TAUU/TAUV 的边缘分布和联合分布偏移。
- 空间谱、时间谱和跨变量相关性。
- 对低概率但真实观测到的 ENSO 前兆状态的保留程度。

### 优化稳定性

- 目标函数轨迹和目标单调性。
- 接受/拒绝提案比例以及回退频率。
- 不同随机种子之间的方差和最终目标方差。
- 优化过程中的梯度范数、投影范数和自然性漂移。

### 科学解释

- 不同随机种子和流形模型下 OPR 的空间稳定性。
- 不同 CMIP6 模式来源之间的一致性。
- 与历史 ENSO 前兆复合场的一致性。
- 目标观测敏感区对自然性定义的敏感度。

## Phase 1 反方检查

**暂定结论：在大规模检索前需要修订。** 目前没有发现致命错误，但以下重大风险必须在大规模实验前处理。

1. **数据分布不等于真实流形。** Diffusion model 学到的是数据支持的分布，而不是完整地球动力学。应使用“数据支持的自然性”和“模型分布可行性”等校准后的表述。
2. **自然性没有唯一标量定义。** likelihood、重构误差、历史样本距离以及物理统计一致性可能给出不同排序。必须预注册多指标评估，而不能在看到结果后选择有利指标。
3. **稀有事件可能被抑制。** 全局先验可能删除低概率但真实存在的 ENSO 前兆。需要加入低概率历史观测状态以及条件化/位相感知的先验。
4. **生成器覆盖范围可能不足。** 弱生成器可能只是因为无法表达真实变异，才显得其 latent manifold 很“干净”。在把它作为硬约束前，必须测量重构保真度和状态覆盖度。
5. **模型和数据的共同偏差。** 如果生成器和 WalkerNet 使用同一批 CMIP6 记录训练，二者可能共享数据伪影。应采用 source/year holdout，并在条件允许时加入独立观测检查。

### 最强反方论证

该方法可能找到的是某一个 diffusion model 偏好的状态，而不是更自然的 CNOP。最终扰动、ENSO 响应和 OPR 都可能反映生成器偏差，而不是地球系统结构。因此 Phase 2 必须把生成器选择视为不确定性来源，并加入跨先验和留出验证。

## Phase 2 检索计划

围绕以下六类证据进行检索和核验：

1. CNOP、conditional nonlinear optimal perturbation、optimal precursor 和 targeted observation。
2. 用于天气/气候状态生成以及气候流形学习的 diffusion models。
3. Score-based constraints、score distillation 和可微投影。
4. 带生成先验的 latent-space optimization。
5. Riemannian optimization、tangent-space gradients 和 manifold-constrained learning。
6. Physics-informed generative models、conservation-aware diffusion 和 climate plausibility。

调研将主动寻找关于 mode collapse/过度平滑、likelihood 与物理真实性不一致，以及 latent-space 遗漏稀有事件的反方证据。

