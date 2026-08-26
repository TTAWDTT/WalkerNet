# WalkerNet CNOP–ENSO 论文大纲 v3

> 本版本按照作者给出的五部分结构重写。  
> `Introduction` 已经完成，本文档只规划后续结构和正文衔接。  
> 本文件是写作蓝图，不是正文，不修改 `main.tex`。

## 总体结构

```text
1. Introduction                         （已完成）
2. Related Works
3. Method
   3.1 WalkerNet
   3.2 CNOP
4. Results
   4.1 WalkerNet forecast performance
   4.2 CNOP and Niño3.4 evolution
5. Conclusion
```

暂时不设置 `Future Works` 章节。局限性如果需要，可以在 Conclusion 前后用一小段简短说明，不单独扩展成新的主体章节。

建议论文主线保持为：

> WalkerNet provides the nonlinear autoregressive forecast operator, while CNOP searches for a bounded initial perturbation that can induce a targeted ENSO-like Niño3.4 evolution. The results first establish the forecast skill of WalkerNet and then show that CNOP can be solved and can excite an ENSO-like response from selected initial states.

---

# 1. Introduction（已完成）

## 1.1 当前状态

Introduction 已经由现有 `main.tex` 写成，主要内容包括：

- AI 天气和气候预报的发展；
- XAI 对模型内部动力学解释的需求；
- gradient-based attribution 的局地一阶性质；
- CNOP 的有限幅度非线性优化思想；
- ENSO 作为事件型 XAI 测试对象；
- WalkerNet 作为自回归全球海气场预测模型；
- CNOP 与 optimal precursor / counterfactual ENSO 的联系。

## 1.2 后续只需做的检查

Introduction 暂时不重写，只检查：

- 引用是否完整；
- CNOP、OPR、OGIE 等概念是否与 Related Works 的表述一致；
- 是否提前写出了尚未在 Results 中证实的 Global、Indian 或 gradient/random 结论；
- `Introduction` 最后一段是否自然引出本文的 Method 和 Results；
- 论文主问题是否集中到“WalkerNet 预报性能 + CNOP 能否激发 ENSO-like response”。

## 1.3 Introduction 结尾建议落点

Introduction 的最后应该明确提出两个核心问题：

1. WalkerNet 是否具备足够的 Niño3.4 预报能力，可以作为 CNOP 的非线性预报算子？
2. 在给定初始扰动约束下，CNOP 是否能够找到有效扰动，并通过 WalkerNet rollout 激发 ENSO-like Niño3.4 演进？

**过渡到 Related Works：** 先回顾 AI 气候预报、XAI、CNOP 和 ENSO 前兆研究，再说明本文具体填补的空缺。

---

# 2. Related Works（约 800–1,000 words）

## 2.1 AI weather and climate forecasting

**目的：** 介绍 AI 模型在天气和气候预测中的发展，并说明多变量、自回归、全球场预测的研究背景。

**主要内容：**

- AI 在天气和气候预测中的应用；
- data-driven model 与传统数值模式的区别；
- 多变量海气场预测对于 ENSO 研究的意义；
- WalkerNet 所处的研究背景。

**可使用的现有引用：**

- `dai2026ai`；
- `luo2026ai`；
- `qin2024enso`；
- `wang2025ocean`。

**本节不写：** WalkerNet 的具体网络结构，把具体结构放到 Method 3.1。

## 2.2 Explainable AI for weather and climate models

**目的：** 回顾 gradient、saliency 和 attribution 方法在天气/气候模型解释中的常见用法。

**主要内容：**

- 梯度敏感性、saliency map 和 feature attribution；
- 这些方法如何识别输入变量和空间区域；
- 梯度方法的优势：计算直接、解释简单、适合局地敏感性分析；
- 梯度方法的限制：局地线性近似、微小扰动假设、难以描述有限幅度的长时间演化。

**关键表述：** 不把 gradient 方法描述为错误，而是说明它和 CNOP 回答的是不同问题。

**可使用的现有引用：**

- `guo2026cnop`；
- `mu2025predictability`；
- `qin2026physics`。

## 2.3 CNOP and nonlinear sensitivity analysis

**目的：** 回顾 CNOP 的理论来源以及它在天气和气候问题中的应用。

**主要内容：**

- CNOP 的 constrained nonlinear optimization 定义；
- CNOP 与传统线性敏感性分析的区别；
- OGIE、目标观测敏感区和 nonlinear error growth；
- 自动微分和可微分 AI 模型使 CNOP 可以应用于 data-driven forecast operator。

**可使用的现有引用：**

- `mu2003cnop`；
- `guo2026cnop`；
- `mu2025predictability`。

## 2.4 ENSO predictability and precursor studies

**目的：** 将相关工作从一般 CNOP 引向 ENSO 预测和前兆识别。

**主要内容：**

- ENSO 的 predictability 问题；
- Niño3.4 指数作为事件强度指标；
- ENSO 预测中的初始状态、海气耦合和前期异常；
- 传统 OPR/precursor 研究与 AI forecast model 的区别。

**可使用的现有引用：**

- `qin2024enso`；
- `ji2025optimal`；
- `wang2025ocean`。

## 2.5 Research gap and position of this study

**目的：** 用一小节明确说明本文与已有工作的区别。

**拟表达的 gap：**

现有研究已经分别讨论了：

- AI 气候预测；
- gradient-based XAI；
- CNOP 非线性敏感性；
- ENSO predictability 和 precursor。

但还缺少一个将以下要素统一起来的框架：

```text
learned autoregressive forecast model
        +
finite-amplitude constrained perturbation
        +
multi-month Niño3.4 target
        +
counterfactual ENSO-like evolution
```

本文的定位是：

> We use WalkerNet as a differentiable nonlinear forecast operator and apply CNOP to search for bounded initial perturbations whose effects unfold through the autoregressive rollout toward a targeted Niño3.4 response.

**过渡到 Method：** 相关工作说明了为什么需要这一框架，下一节给出 WalkerNet 和 CNOP 的具体定义。

---

# 3. Method（约 1,400–1,700 words）

Method 只分成两部分：

```text
3.1 WalkerNet
3.2 CNOP
```

不再额外拆出 Background、Problem Formulation、Experimental Design 等并列大章节。

## 3.1 WalkerNet

### 3.1.1 Architecture and autoregressive forecast

**目的：** 用一个连续小节说明 WalkerNet 的输入、主要模块和多步预报方式，不把每个网络组件拆成独立标题。

**输入与输出：**

$$
\mathbf{x}_{t-11:t}
\in
\mathbb{R}^{12\times4\times180\times360},
$$

其中四个变量为 TOS、ZOS、TAUU 和 TAUV。模型单步输出为：

$$
\widehat{\mathbf{x}}_{t+1}
=
F_\theta(\mathbf{x}_{t-11:t},m_{t+1},s).
$$

正文依次概括：变量感知的 4×4 patch embedding、每个 patch 内的时间–变量 local fusion、45×90 patch grid 展平为 4050 个 spatial tokens、6 个 Spatial Attention Blocks、target-month gated TMoE、两级 PixelShuffle×2 decoder，以及 1×1 Conv 输出四变量场。模型采用残差形式：

$$
\widehat{\mathbf{x}}_{t+1}
=
\mathbf{x}_{t}+\Delta\mathbf{y}_{t+1}.
$$

多步 rollout 使用滑动窗口：

$$
\mathbf{W}_{\ell+1}
=
\operatorname{append}
\left(
\operatorname{drop\ oldest}(\mathbf{W}_{\ell}),
\widehat{\mathbf{x}}_{t+\ell}
\right),
$$

因此 CNOP 优化的是完整的多月非线性轨迹，而不是单步静态输出。

**建议图：** WalkerNet 总架构图；Spatial Attention Block、TMoE 和 Decoder 详图作为补充图，不在 Method 中继续拆标题。

### 3.1.2 Niño3.4 calculation and anomaly convention

**目的：** 统一说明如何从 WalkerNet 的 TOS 场得到评测指标，避免 truth 与 baseline 使用不同口径。

定义 lead-$\ell$ 的 Niño3.4 指数：

$$
N_{\ell}
=
\operatorname{AreaMean}_{R_{\mathrm{Nino3.4}}}
\left(
\mathrm{TOS}_{t+\ell}
-
\mathrm{climatology}_{t+\ell}
\right).
$$

需要说明：

- Niño3.4 是从预测 TOS 场计算的，不是模型直接输出；
- truth 使用 observed climatology；
- baseline 和 perturbed 使用 forecast-model climatology；
- 三者的 anomaly、lead 和区域定义保持一致；
- 如果图中展示 raw TOS，必须和 anomaly 图明确区分。

## 3.2 CNOP

### 3.2.1 Perturbation, constraint and objective

**目的：** 在一个小节中完成 CNOP 的数学定义，避免把扰动、约束和目标拆成多个过细标题。

扰动后的输入为：

$$
\mathbf{x}^{\delta}
=
\mathbf{x}
+
\mathbf{M}_{D}\odot\boldsymbol{\delta},
$$

其中扰动只作用于最后一个输入月的 TOS 和 ZOS，TAUU 和 TAUV 不直接扰动，$D$ 表示允许扰动的区域。

relative initial $L_2$ 约束为：

$$
\mathcal{C}_{D}(r_D)
=
\left\{
\boldsymbol{\delta}:
\frac{\|\boldsymbol{\delta}\|_2}
{\|\mathbf{x}\|_2}
\le r_D
\right\},
\qquad r_D=0.03
$$

正文说明 normalization、TOS/ZOS balancing、projected Adam 和 `constraint_ratio`，但不再单独拆标题。

令：

$$
N_{\ell}^{\mathrm{base}}=N_{\ell}(\mathbf{0}),
\qquad
N_{\ell}^{\mathrm{pert}}=N_{\ell}(\boldsymbol{\delta}),
$$

$$
\Delta N_{\ell}
=
N_{\ell}^{\mathrm{pert}}
-
N_{\ell}^{\mathrm{base}}.
$$

normal objective 根据正式实验协议选择其一：

$$
J_{\mathrm{normal}}=\Delta N_{12},
$$

或：

$$
J_{\mathrm{normal}}
=
\frac{1}{3}\sum_{\ell=10}^{12}\Delta N_{\ell}.
$$

delayed-onset objective 可以写成：

$$
J_{\mathrm{delay}}
=
\Delta N_{12}
-
\lambda_{\mathrm{early}}
E_{\mathrm{early}},
$$

$$
E_{\mathrm{early}}
=
\sum_{\ell=1}^{\ell_e}
w_{\ell}|\Delta N_{\ell}|.
$$

必须明确 delayed term 是优化阶段的目标项，而不是优化完成后的 post-hoc 筛选。

### 3.2.2 Multi-start optimization and candidate comparison

**目的：** 说明多个起点如何产生候选解，以及 rank-1 / top-3 如何定义。

设置 $K$ 个初始扰动：

$$
\left\{
\boldsymbol{\delta}^{(0)}_1,\ldots,
\boldsymbol{\delta}^{(0)}_K
\right\}.
$$

每个起点独立优化：

$$
\boldsymbol{\delta}^{*}_k
=
\arg\max_{\boldsymbol{\delta}\in\mathcal{C}_{D}}
J(\boldsymbol{\delta};\boldsymbol{\delta}^{(0)}_k).
$$

最终 rank-1 为：

$$
k^*
=
\arg\max_kJ(\boldsymbol{\delta}^{*}_k),
\qquad
\boldsymbol{\delta}_{\mathrm{CNOP}}^*
=
\boldsymbol{\delta}_{k^*}^*.
$$

如果保留 top-3：

$$
\mathcal{T}_3
=
\operatorname{Top3}
\left\{
J(\boldsymbol{\delta}^{*}_k)
\right\}_{k=1}^{K}.
$$

本节末尾用一段文字定义三条评测轨迹：truth 是观测目标演进，baseline 是未加 CNOP 的 WalkerNet rollout，perturbed 是加入优化扰动后的 rollout。三者的 anomaly、climatology 和 lead 口径沿用 3.1.2。

**过渡到 Results：** Method 先验证 WalkerNet 能否提供可信的 ENSO forecast operator，再评估 CNOP 是否能在该 operator 上求解出有效扰动并改变 Niño3.4 演进。

---

# 4. Results（约 1,500–1,900 words）

Results 严格分成两部分，不再拆成多个并列大章节。

## 4.1 WalkerNet forecast performance

**目的：** 先证明 WalkerNet 具备足够的 Niño3.4 预测能力，能够作为后续 CNOP 实验的 nonlinear forecast operator。

### 4.1.1 Overall lead-dependent forecast skill

报告：

- lead-1 至 lead-36 的 Niño3.4 ACC；
- monthly ACC；
- monthly RMSE；
- three-month ACC / RMSE（如果对应指标已经生成）；
- 与 persistence 的比较（如果保留）。

建议图：

- `Nino3.4 ACC` 随 lead month 的曲线；
- 模型和 persistence 使用不同线型，不要让 persistence 抢主视觉；
- y 轴固定为 0–1；
- x 轴为 lead month 1–36。

### 4.1.2 Start-month-dependent forecast skill

报告不同起始月份下的 lead-1 至 lead-18 或 lead-36 ACC：

$$
\mathrm{ACC}(m_{start},\ell).
$$

建议图：

- 横轴：lead month；
- 纵轴：start month；
- 颜色：ACC；
- 若需要标出下降最快的四个 lead month，使用透明阴影或 hatch，不改变底层数值。

建议表：

```text
start month × lead month ACC
```

### 4.1.3 Interpretation for the CNOP experiment

这里不要重复介绍网络结构，而是总结：

- WalkerNet 在短期 lead 上具备较高 skill；
- skill 随 lead 增大而下降；
- 不同起始月份存在差异；
- 因此 CNOP 的 lead-12 response 应放在模型 skill 已经下降但仍具有预测意义的时间范围内解释。

这一小节的结论只能支持“WalkerNet 可作为研究工具”，不能直接证明 CNOP 的物理正确性。

## 4.2 CNOP and Niño3.4 evolution

**目的：** 展示 CNOP 是否成功求解，以及 CNOP 扰动是否能够通过 WalkerNet rollout 激发 ENSO-like response。

### 4.2.1 CNOP optimization and constraint verification

先报告：

- 优化是否正常收敛；
- 每个案例是否生成 summary / NPZ / candidate；
- relative initial $L_2$ constraint ratio；
- rank-1 和 top-3 的 objective；
- 多起点之间的 objective 分布。

建议表：

| Case | Start count | Constraint ratio | Lead-12 baseline | Lead-12 perturbed | Gain | Status |
|---|---:|---:|---:|---:|---:|---|

### 4.2.2 Spatial structure of the optimized perturbation

展示：

- 十个初始场的 rank-1 CNOP TOS perturbation；
- 如果需要，附带 ZOS perturbation；
- 同一套经纬度范围；
- 同一套颜色语义和 colorbar；
- Nino3.4 区域框；
- 不把平滑/插值后的图当成新的数值结果。

重点描述：

- 扰动是否集中于 Pacific；
- 不同案例之间是否具有共同结构；
- top-3 是否呈现相近或互补模式；
- rank-1 是否只是多起点中的最大候选，而不是唯一解。

### 4.2.3 Truth, baseline and perturbed lead-12 fields

展示四列 overview：

```text
Initial perturbation | Truth | Baseline | Perturbed
```

每个案例报告：

- truth Niño3.4；
- baseline Niño3.4；
- perturbed Niño3.4；
- baseline–perturbed gain；
- baseline 与 truth 的接近程度；
- perturbation 是否把响应推向 ENSO-like 状态。

### 4.2.4 Niño3.4 response evolution

这是 Results 第二部分的核心图和核心讨论。

展示：

- truth / baseline / perturbed 的 lead-1 至 lead-12 Niño3.4；
- normal CNOP 和 delayed-onset CNOP 的对比；
- early response、late response 和 lead-12 gain；
- onset timing；
- TOS / ZOS response evolution（如果保留）。

普通 CNOP 重点说明：

> The optimized perturbation changes the subsequent Niño3.4 trajectory through the WalkerNet autoregressive rollout.

Delayed-onset 重点说明：

> The delayed-onset objective suppresses premature response and promotes a later increase in Niño3.4, when this behavior is observed in the qualified cases.

### 4.2.5 Evidence for an ENSO-like response

这里要定义“激发 ENSO 一类现象”的证据标准，例如：

- perturbed Niño3.4 明显高于 baseline；
- perturbed response 在 lead-12 或 late-season 达到预设阈值；
- baseline 没有同样程度的响应；
- truth / baseline / perturbed 的比较口径一致；
- 结果不是由色标、平滑或图像裁剪造成的视觉差异。

建议不要写成：

> CNOP proves the real physical cause of ENSO.

而写成：

> CNOP can generate an ENSO-like counterfactual response within the WalkerNet model under the prescribed initial-state and perturbation constraints.

### 4.2.6 Multi-start and case-to-case variation

最后讨论：

- 不同起点的结果是否收敛；
- top-3 是否稳定；
- 哪些案例成功产生较强 ENSO-like response；
- 哪些案例没有明显 crossing；
- 为什么不能只展示最成功的单个案例来代表所有情况。

如果 delayed-onset 不是所有案例都成功，应如实报告 qualified-case fraction，而不是概括为所有案例都实现延迟爆发。

**过渡到 Conclusion：** Results 先建立 WalkerNet 的预报能力，再展示 CNOP 的可求解性及其对 Niño3.4 trajectory 的影响，由此进入结论。

---

# 5. Conclusion（约 250–350 words）

Conclusion 只回答论文已经验证的两个问题，不再加入 Future Works。

## 5.1 WalkerNet 的作用

总结：

- WalkerNet 能够在多个 lead 上提供可量化的 Niño3.4 预测 skill；
- 其自回归结构提供了研究多月非线性响应的 forecast operator；
- skill 随 lead 增大而下降，因此 CNOP 结果需要结合对应 lead 的 baseline skill 解读。

## 5.2 CNOP 的主要发现

在实验数据支持的范围内总结：

1. 在给定变量、区域和 relative initial $L_2$ 约束下，CNOP 可以被稳定求解；
2. 多起点优化能够产生并排序一组候选最优扰动；
3. 选定的 CNOP 扰动可以通过 WalkerNet rollout 改变 Niño3.4 的后续演进；
4. 在满足判据的案例中，CNOP 可以激发 ENSO-like response；
5. delayed-onset 目标在成功案例中可以把响应从早期重新分配到后期。

## 5.3 解释边界

最后保留一段简短限定：

- 结果是 WalkerNet 内部的 model-based counterfactual；
- CNOP 不是现实气候系统唯一的因果前兆；
- 图像平滑和插值只用于显示；
- 结论只适用于实际报告的案例、checkpoint、约束和 objective。

不单独设置 `Future Works` 章节。

---

# 图表安排

## 主文建议图

1. **WalkerNet architecture**：模型总体结构；
2. **CNOP schematic**：多起点、约束、rollout、目标函数和 rank selection；
3. **WalkerNet forecast performance**：lead-1–36 Niño3.4 ACC/RMSE；
4. **Start-month × lead-month ACC**：不同起始月的预测性能；
5. **Pacific CNOP overview**：initial perturbation / truth / baseline / perturbed；
6. **Niño3.4 response evolution**：truth / baseline / perturbed 及 normal/delayed 对比。

## 补充图

- Spatial Attention Block；
- Patch Embedding + Local Fusion；
- TMoE；
- Decoder + residual rollout；
- top-3 candidate patterns；
- 额外海域或未纳入主文的 exploratory results。

## 主文建议表

1. WalkerNet forecast skill summary；
2. CNOP case-level optimization and constraint audit；
3. CNOP lead-12 response summary。

---

# 写作顺序

1. 保留并检查 Introduction；
2. 完成 Related Works 引用和 gap 段；
3. 写 Method 3.1 WalkerNet；
4. 写 Method 3.2 CNOP；
5. 整理 WalkerNet forecast performance 表和图；
6. 整理 CNOP constraint / summary / candidate 审计表；
7. 写 CNOP spatial structure 和 Niño3.4 evolution；
8. 写 Conclusion；
9. 最后统一 Abstract、Title、图注、术语和引用。

---

# 写作前必须冻结的术语

- `truth`：观测目标演进；
- `baseline`：未加 CNOP 扰动的 WalkerNet rollout；
- `perturbed`：加入 CNOP 扰动后的 WalkerNet rollout；
- `response`：perturbed − baseline；
- `initial perturbation`：最后输入月上的 TOS/ZOS 扰动；
- `CNOP*`：候选集合中目标函数最大的合格扰动；
- `ENSO-like response`：满足预先定义的 Niño3.4 响应和 baseline 对照条件的模型内部响应；
- `delayed-onset`：优化目标中显式抑制 early response、保留 late response 的配置。

# 当前大纲的写作边界

- 不单独设置 Future Works；
- 不把架构细节拆成过多主体章节；
- 不提前写未完成的 Global / Indian 结论；
- 不提前声称 CNOP 优于 gradient，除非正式对照结果已经完成；
- 不把单个 rank-1 案例写成所有案例的普遍规律；
- 不把可视化插值和平滑当成新的科学数据；
- 不在 Results 之前先写强结论。
