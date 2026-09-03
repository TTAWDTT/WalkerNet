# WalkerNet CNOP–ENSO 论文大纲 v2

> 状态：结构规划稿，供作者审阅和修改。  
> 本文件不等同于正文，不替代 `main.tex`，也不代表所有实验结果已经完成审计。

## 0. 当前范围与版本说明

论文工作目录：

```text
D:/Github/WalkerNet/paper/WalkerNet_CNOP_workshop_overleaf_2026-08-21/
```

主稿入口：

```text
main.tex
```

当前 `main.tex` 的摘要仍包含 Pacific、Atlantic–Indian、Global 以及 matched gradient/random comparison；而目前最完整、最适合作为主线的结果是 Pacific CNOP 与 Pacific delayed-onset。因此，本大纲采用以下证据边界：

- Pacific normal CNOP：主结果候选；
- Pacific delayed-onset CNOP：核心结果候选；
- multi-start / top-3 稳定性：主要稳健性分析；
- Global、Indian、Atlantic–Indian：在完成统一审计前作为扩展结果或补充材料；
- gradient/random 对照：只有正式结果完成后，才能写成已验证的比较结论。

正式写作前需要冻结四个决定：

1. 主文是否只关注 Pacific，还是保留三海域比较；
2. Global / Indian 结果放主文还是 Supplement；
3. gradient/random 是否已经达到正式证据标准；
4. CNOP 目标使用单独 lead-12，还是 leads 10–12 三个月平均。

---

## 1. 论文主线

本文围绕以下逻辑展开：

```text
ENSO 前兆识别问题
        ↓
梯度 XAI 的局地一阶局限
        ↓
CNOP 的有限幅度非线性优化
        ↓
WalkerNet 作为可微分自回归气候预报算子
        ↓
Pacific CNOP 与 delayed-onset CNOP
        ↓
空间结构、时间演化和多起点稳定性
        ↓
模型内部的 ENSO-like counterfactual precursor
```

### 中心论点（建议版本）

在给定 WalkerNet、初始状态、扰动变量和相对扰动约束的条件下，CNOP 可以识别面向目标 ENSO 响应的有限幅度扰动；进一步加入 delayed-onset 目标后，可以抑制早期 Niño3.4 响应，并使目标响应在后期增强。

如果 matched gradient/random 实验尚未完成，暂时不要写成“CNOP 已经优于 gradient”。可以使用较稳妥的表述：

> CNOP provides a model-internal finite-amplitude search for event-oriented ENSO precursors, whereas gradient attribution characterizes only local first-order sensitivity.

---

## 2. 暂定标题

### 首选标题

> Event-oriented nonlinear XAI for ENSO precursors with conditional nonlinear optimal perturbations

### 突出 delayed-onset 的标题

> Delayed-onset nonlinear optimal precursors for ENSO in an autoregressive climate model

### 突出 WalkerNet 的标题

> Discovering nonlinear ENSO precursors in WalkerNet with conditional nonlinear optimal perturbations

### 中文工作标题

> 基于条件非线性最优扰动和 WalkerNet 的 ENSO 延迟爆发前兆识别

标题暂时不要直接强调“不同海域比较”，除非三海域实验和统一审计都已经完成。

---

## 3. 摘要结构

建议英文摘要长度：约 180–250 words。

摘要只写已经有证据支持的内容，建议按以下顺序组织：

1. AI 气候预报模型需要从预测性能走向动力学和事件型解释；
2. 梯度方法提供局地一阶敏感性，但不一定描述有限幅度非线性增长；
3. CNOP 在给定约束下直接搜索最大化目标响应的初始扰动；
4. 将 CNOP 嵌入 WalkerNet 的完整自回归 rollout；
5. 只扰动最后输入月的 TOS 和 ZOS；
6. 以 lead-12 或 late-season Niño3.4 response 为目标；
7. Pacific normal 与 delayed-onset 实验的主要发现；
8. delayed-onset 是否实现“早期弱、后期强”的响应重排；
9. 将结果解释为 model-based optimal precursors，而非现实气候系统唯一的因果解释。

### 摘要中的证据边界

- 如果 Global / Indian 只作为 exploratory result，摘要中不要写成全面验证；
- 如果 gradient/random 尚未完成，使用 `we formulate` 或 `we design`，不要使用 `we demonstrate superiority`；
- 如果 delayed-onset 只在部分案例成功，使用 `in qualified cases` 或 `for a subset of cases`；
- 不要把可视化平滑、插值后的图形当作新的科学证据。

---

# 1. Introduction（约 900–1,000 words）

## 1.1 AI 气候预报与可解释性需求

**目的：** 说明为什么高预测技巧不足以回答气候事件形成机制问题。

**内容：**

- AI 天气和气候预报的发展；
- 从 forecast skill 到 model interpretation 的研究需求；
- 预测模型需要回答哪些初始区域、变量和异常最容易影响 ENSO；
- WalkerNet 作为多变量全球海气场自回归模型，为事件型 XAI 提供可微分预报算子。

**可使用的现有文献：** `dai2026ai`、`luo2026ai`。

**本节要避免：** 不把“AI 学到了真实动力学”当作已验证事实。

## 1.2 Gradient-based XAI 的局限

**目的：** 说明局地梯度和有限幅度非线性搜索之间的区别。

**内容：**

梯度方法刻画的是：

\[
\frac{\partial J}{\partial \mathbf{x}},
\]

即当前状态附近极小扰动的局地一阶响应。ENSO 演化还涉及：

- 有限幅度扰动；
- 海气耦合；
- 多变量协同；
- 多个月份的状态依赖反馈；
- 自回归预测中的误差传播。

**关键表述：** 梯度方法不是错误，而是回答了不同的问题：它描述局地敏感性，而不是直接寻找最有效的有限幅度事件前兆。

**可使用的现有文献：** `guo2026cnop`、`mu2025predictability`、`qin2026physics`。

## 1.3 CNOP、OGIE 与 OPR

**目的：** 将 CNOP 从误差增长分析引向事件前兆识别。

**内容：**

介绍：

- OGIE：optimal growing initial error；
- targeted observation-sensitive area；
- OPR：optimal precursor。

CNOP 的一般形式为：

\[
\boldsymbol{\delta}^{*}
=
\arg\max_{\boldsymbol{\delta}\in\mathcal{C}}
J(\mathbf{x}+\boldsymbol{\delta}).
\]

本文将目标函数改写为面向 ENSO 形成的事件目标。

**可使用的现有文献：** `mu2003cnop`、`guo2026cnop`、`qin2024enso`、`ji2025optimal`。

## 1.4 ENSO 作为事件型 XAI 测试对象

**目的：** 说明 ENSO 为什么适合作为 CNOP 的案例。

**内容：**

- ENSO 有明确的 Niño3.4 指数；
- ENSO response 具有明显的时间演化；
- 可以从近中性状态开始构造 counterfactual rollout；
- 可以区分 early response、onset timing 和 late-season response；
- Pacific 区域具有清晰的海气耦合背景。

## 1.5 科学问题与贡献

### 研究问题

**RQ1：** 在相同初始时刻、变量、空间支持和扰动预算下，CNOP 能否产生目标 Niño3.4 response？

**RQ2：** Pacific CNOP 的有限幅度扰动是否会通过自回归 rollout 逐步产生 ENSO-like response？

**RQ3：** delayed-onset objective 是否可以抑制早期响应，并增强后期目标响应？

**RQ4（可选）：** 不同海域约束是否对应不同的 ENSO precursor pathway？

### 主要贡献

1. 将 basin-constrained CNOP 引入 WalkerNet 的全球多变量自回归预报框架；
2. 用多起点 projected Adam 搜索面向事件目标的有限幅度扰动；
3. 用 delayed-onset objective 研究 ENSO 前兆的时间结构，而不只是最终幅度；
4. 在明确的模型内部解释边界下，构造 neutral-state ENSO-like counterfactual。

**过渡：** 下一节给出 WalkerNet、CNOP 约束和 Niño3.4 objective 的数学定义。

---

# 2. Background and Problem Formulation（约 900–1,100 words）

## 2.1 WalkerNet autoregressive field forecasting

定义输入：

\[
\mathbf{x}_{t-11:t}
\in
\mathbb{R}^{12\times4\times180\times360}.
\]

四个变量为：

```text
TOS, ZOS, TAUU, TAUV
```

单步预测：

\[
\widehat{\mathbf{x}}_{t+1}
=
F_\theta(\mathbf{x}_{t-11:t},m_{t+1},s).
\]

残差形式：

\[
\widehat{\mathbf{x}}_{t+1}
=
\mathbf{x}_{t}+\Delta\mathbf{y}_{t+1}.
\]

说明：

- Niño3.4 由 TOS 预测场计算，而不是模型直接输出；
- truth 使用 observed climatology；
- baseline / perturbed 使用 forecast-model climatology；
- rollout 会将预测结果加入下一步输入窗口。

## 2.2 Basin-constrained perturbation

定义：

\[
\mathbf{x}^{\delta}
=
\mathbf{x}+\mathbf{M}_{D}\odot\boldsymbol{\delta}.
\]

其中：

- (D)：指定海域；
- (mathbf{M}_{D})：区域和变量掩膜；
- 扰动施加在最后一个输入月；
- 直接扰动 TOS 与 ZOS；
- TAUU 与 TAUV 不直接扰动。

## 2.3 Relative initial (L_2) constraint

\[
\mathcal{C}_{D}(r_D)
=
\left\{
\boldsymbol{\delta}:
\frac{\|\boldsymbol{\delta}\|_2}
{\|\mathbf{x}\|_2}
\le r_D
\right\}.
\]

对于 3% 方案：

\[
r_D=0.03.
\]

必须说明：

- 约束是在归一化空间还是物理空间计算；
- TOS 与 ZOS 是否先做 RMS balancing；
- 每个 Adam step 是否投影回约束集合；
- `constraint_ratio` 如何定义；
- 数值误差如何处理。

## 2.4 Niño3.4 objective

\[
N_{\ell}(\boldsymbol{\delta})
=
\frac{
\sum_{i,j}
\mathbf{1}_{R_{\mathrm{Nino3.4}}}(i,j)
\cos\phi_i\,
\widehat{\mathrm{TOS}}_{t+\ell}^{\delta}(i,j)
}{
\sum_{i,j}
\mathbf{1}_{R_{\mathrm{Nino3.4}}}(i,j)
\cos\phi_i
}.
\]

其中：

\[
R_{\mathrm{Nino3.4}}
=
5^\circ\mathrm{S}\text{--}5^\circ\mathrm{N},
\quad
170^\circ\mathrm{W}\text{--}120^\circ\mathrm{W}.
\]

定义：

\[
N_{\ell}^{\mathrm{base}}=N_{\ell}(\mathbf{0}),
\]

\[
\Delta N_{\ell}
=
N_{\ell}^{\mathrm{pert}}
-N_{\ell}^{\mathrm{base}}.
\]

## 2.5 Normal CNOP objective

单月 lead-12 版本：

\[
J_{\mathrm{normal}}(\boldsymbol{\delta})
=
\Delta N_{12}.
\]

late-season 三个月版本：

\[
J_{\mathrm{normal}}(\boldsymbol{\delta})
=
\frac{1}{3}
\sum_{\ell=10}^{12}\Delta N_{\ell}.
\]

正文、摘要、代码和图注必须只选择其中一个正式版本。

## 2.6 Delayed-onset objective

建议写成：

\[
J_{\mathrm{delay}}(\boldsymbol{\delta})
=
\Delta N_{12}
-\lambda_{\mathrm{early}}
E_{\mathrm{early}}(\boldsymbol{\delta}),
\]

其中：

\[
E_{\mathrm{early}}(\boldsymbol{\delta})
=
\sum_{\ell=1}^{\ell_e}
w_{\ell}|\Delta N_{\ell}|.
\]

如果代码采用最大早期响应惩罚，也可以写成：

\[
E_{\mathrm{early}}
=
\max_{\ell\le\ell_e}|\Delta N_{\ell}|.
\]

必须明确 delayed penalty 是：

- 优化过程中的正式目标项；还是
- 优化完成后才进行的 post-hoc selection。

两者不能混写。

## 2.7 Multi-start CNOP

设置 (K) 个起点：

\[
\left\{
\boldsymbol{\delta}^{(0)}_1,ldots,
\boldsymbol{\delta}^{(0)}_K
\right\}.
\]

每个起点独立优化：

\[
\boldsymbol{\delta}^{*}_k
=
\arg\max_{\boldsymbol{\delta}\in\mathcal{C}_{D}}
J(\boldsymbol{\delta};
\boldsymbol{\delta}^{(0)}_k).
\]

最终 rank-1：

\[
k^*
=
\arg\max_k J(\boldsymbol{\delta}^{*}_k),
\qquad
\boldsymbol{\delta}_{\mathrm{CNOP}}^*
=
\boldsymbol{\delta}_{k^*}^*.
\]

---

# 3. Experimental Design（约 800–1,000 words）

## 3.1 Data and checkpoint

写清楚：

- 五个 climate-model sources；
- 数据时间范围；
- train / validation / test split；
- 180×360 网格；
- 四个变量；
- checkpoint 版本；
- normalization statistics；
- forecast climatology 的计算和缓存方式。

重点区分：

```text
truth:
observed climatology correction

baseline / perturbed:
forecast-model climatology correction
```

## 3.2 Case selection

中性案例应满足：

- 12 个月输入窗口完整；
- 目标年份不属于强 ENSO 事件；
- baseline rollout 不提前达到目标 ENSO 阈值；
- truth / baseline / perturbed 场完整；
- source-wise climatology 可用；
- baseline max-3-month 指标满足预注册阈值。

## 3.3 Perturbation domains

如果论文只关注 Pacific：

- Pacific 作为主实验；
- Global / Indian 放入 Supplement 或 future work；
- 摘要不宣称完整三海域结论。

如果保留三海域，则统一：

- longitude convention；
- latitude range；
- ocean mask；
- constraint radius；
- TOS/ZOS balancing；
- candidate count；
- normal/delayed objective。

## 3.4 Optimization protocol

建议用一张主协议表：

| Component | Primary setting |
|---|---|
| Input window | 12 months |
| Directly perturbed variables | TOS and ZOS |
| Perturbation time | final input month |
| Parameter grid | 45×90 |
| Forecast grid | 180×360 |
| Rollout horizon | 12 months |
| Constraint | relative initial (L_2) |
| Constraint ratio | 3% |
| Optimizer | projected Adam |
| Starts | 8 / 24, depending on experiment |
| Optimization steps | 40 / 100, depending on experiment |
| Selection | rank-1 and top-3 |
| Objective | normal or delayed-onset |
| Main target | lead-12 Niño3.4 |

不同实验配置必须分开标记，不能把 3%/40 steps、30%/1000 steps 和 24 starts/100 steps 混成一个协议。

## 3.5 Matched controls

如果完成正式对照，所有方法必须匹配：

```text
initial time
variables
spatial mask
constraint radius
perturbation norm
```

比较对象：

- CNOP；
- zero-state local gradient；
- matched random perturbations。

如果结果尚未完成，这一节只写设计，不写结果。

## 3.6 Evaluation metrics

### Constraint diagnostics

- relative initial (L_2)；
- constraint ratio；
- physical TOS/ZOS amplitude；
- candidate completeness；
- missing-file audit。

### ENSO response

- lead-1 至 lead-12 Niño3.4；
- lead-12 gain；
- late-season gain；
- baseline–perturbed difference；
- ENSO threshold crossing。

### Delayed-onset diagnostics

- early response；
- late response；
- early penalty；
- onset timing；
- normal/delayed difference。

### Robustness diagnostics

- multi-start objective distribution；
- top-3 consistency；
- source-wise variation；
- initial-condition sensitivity；
- basin sensitivity。

---

# 4. Results（约 1,500–1,800 words）

## 4.1 File and numerical audit

先报告结果完整性：

- summary 是否齐全；
- NPZ 是否齐全；
- candidate 文件是否齐全；
- constraint ratio 是否合格；
- map-derived values 是否与 summary 一致；
- 是否存在失败起点；
- 是否存在重复案例。

建议表格：

| Case | Source | Year | Domain | Constraint ratio | Baseline lead-12 | Perturbed lead-12 | Gain | Status |
|---|---|---:|---|---:|---:|---:|---:|---|

## 4.2 Pacific normal CNOP

展示：

- 十个案例的 initial TOS/ZOS perturbation；
- truth / baseline / perturbed lead-12 场；
- lead-1 至 lead-12 Niño3.4；
- response evolution；
- lead-12 gain；
- top-3 结构。

需要回答：

1. 初始扰动是否具有共同空间结构；
2. 扰动是否集中在热带 Pacific；
3. baseline 和 truth 是否足够接近；
4. perturbed 是否显著改变 Niño3.4；
5. 不同 source 是否表现出一致性。

## 4.3 Pacific response evolution

展示：

- truth / baseline / perturbed Niño3.4 曲线；
- TOS 和 ZOS response-evolution；
- 初始 perturbation 与 lead-12 response 的对照；
- baseline-to-perturbed response。

重点说明 CNOP 影响的是完整 rollout，而不是单一 lead 的静态场。

## 4.4 Pacific delayed-onset

这是主结果候选章节。

比较：

```text
normal objective
vs.
delayed-onset objective
```

必须报告：

- early lead response；
- lead-12 response；
- onset timing；
- early penalty；
- final gain；
- constraint ratio；
- top-3 stability。

核心表述建议：

> The delayed-onset objective changes the temporal allocation of the response by suppressing premature Niño3.4 growth while preserving or enhancing the late-season target response.

如果只有部分案例成功，应明确写出：

> The intended temporal reorganization is obtained in a subset of qualified Pacific cases, with substantial case-to-case variation.

## 4.5 Multi-start and top-3 stability

分析：

- rank-1 是否依赖偶然起点；
- 8/24 个起点是否收敛到相似结构；
- top-3 objective 是否接近；
- top-3 空间相关性；
- top-3 是否形成一组高性能候选。

推荐表述：

> The optimization landscape contains a set of high-performing candidate perturbations with partially shared spatial structure.

不要将 rank-1 称为现实系统唯一的 precursor。

## 4.6 CNOP versus gradient/random

只有正式对照完成后才作为主要 Results 使用：

\[
\Delta N_{12}^{\mathrm{CNOP}},
\quad
\Delta N_{12}^{\mathrm{gradient}},
\quad
\Delta N_{12}^{\mathrm{random}}.
\]

报告：

- mean / median；
- distribution；
- case-wise success rate；
- threshold crossing rate；
- response trajectory。

如果 CNOP 与 gradient 接近，也应如实报告，而不是预设 CNOP 必然更优。

## 4.7 Global / Indian / Atlantic–Indian extensions

如果保留，放在 Pacific 结果之后；如果主文只关注 Pacific，则移到 Supplement。

Global 可分析：

- 全海域扰动是否更容易形成目标 response；
- 是否存在非 Pacific compensating structures；
- delayed-onset 是否仍有效；
- response 是否更大但更不稳定。

Indian / Atlantic–Indian 只能在当前模式和约束下解释，不应写成对现实气候作用的否定。

---

# 5. Discussion（约 900–1,100 words）

## 5.1 CNOP as event-oriented nonlinear XAI

讨论从：

```text
local sensitivity
        ↓
finite-amplitude nonlinear optimization
        ↓
event-oriented precursor discovery
```

到事件型 XAI 的方法转变。

## 5.2 Delayed-onset 的科学意义

说明：

- normal objective 可能倾向于尽快放大 Niño3.4；
- delayed objective 强调后期目标响应；
- 这更接近 precursor–development–maturation 的描述；
- delayed objective 是条件性设计，不等于唯一自然演化路径。

## 5.3 Pacific 空间结构与响应路径

讨论：

- 初始 TOS/ZOS 结构；
- 赤道 Pacific 的正负异常排列；
- ZOS 与热含量/海平面异常的关系；
- TOS/ZOS 以及 TAUU/TAUV 的后续响应；
- response 如何通过 WalkerNet 传播。

使用 `model-consistent pathway`，不要写成 `verified real-world mechanism`。

## 5.4 与 gradient attribution 的关系

如果两者差异明显：

- 有限幅度 nonlinear pathway 改变了解释。

如果两者相似：

- 当前状态可能处于局地线性近似较好的 regime。

## 5.5 Counterfactual interpretation and model dependence

必须明确：

- CNOP 结果依赖 WalkerNet；
- 依赖训练数据、normalization 和 forecast climatology；
- 依赖 objective、constraint radius 和起点数量；
- 不是现实气候系统唯一的 causal precursor；
- 是模型内部的受约束 counterfactual experiment。

## 5.6 Limitations

至少包括：

1. 案例数量有限；
2. 主要使用一个 checkpoint；
3. 优化可能受局部最优影响；
4. 3% 约束是方法选择，不等于真实物理概率；
5. delayed penalty 权重影响 onset timing；
6. 缺少独立数值模式或观测验证；
7. 不同海域结果可能受 source distribution 影响；
8. 图像平滑和插值只用于显示，不增加科学信息。

---

# 6. Conclusion（约 250–350 words）

结论只保留审计后有证据支持的内容：

1. CNOP 为 WalkerNet 提供了面向事件目标的有限幅度 nonlinear XAI 框架；
2. Pacific CNOP 可以在给定约束下构造模型内部的 ENSO-like counterfactual response；
3. delayed-onset objective 可以改变 response timing，使部分案例表现出 early weak / late strong 的演化；
4. multi-start 与 top-3 分析可以检验候选结构稳定性；
5. 这些结果应解释为 model-based optimal precursors，而不是现实气候系统唯一的因果解释。

如果 gradient/random 对照已经完成，再补充其相对于局地一阶方向的比较结论。

---

# 7. Figures and Tables Plan

## Figures

### Figure 1 — WalkerNet overall architecture

内容：输入、patch embedding、local fusion、6 个 Spatial Attention Blocks、TMoE、decoder 和 residual rollout。

### Figure 2 — CNOP schematic

内容：

```text
x0
 ↓
K initial perturbation starts
 ↓
constraint projection
 ↓
Adam optimization
 ↓
WalkerNet rollout
 ↓
Niño3.4 objective
 ↓
rank candidates
 ↓
CNOP*
```

### Figure 3 — Spatial Attention Block

内容：

```text
LN → MHA → residual → LN → FFN → residual
```

### Figure 4 — Pacific normal CNOP overview

四列：

```text
Initial perturbation | Truth | Baseline | Perturbed
```

### Figure 5 — Pacific delayed-onset overview

展示 delayed rank-1 的 truth / baseline / perturbed 和 lead-12 Niño3.4。

### Figure 6 — Response evolution

展示 normal 与 delayed 的 Niño3.4、TOS response、ZOS response。

### Figure 7 — Multi-start and top-3 stability

展示 rank-1/2/3 空间结构、objective distribution 和 onset timing。

### Figure 8 — Optional basin comparison

仅在三海域结果审计通过后使用；否则放 Supplement。

## Tables

### Table 1 — WalkerNet and CNOP protocol

输入、变量、grid、rollout、constraint、starts、steps、objective 和 selection。

### Table 2 — Case-level CNOP results

source、year、baseline、perturbed、gain、max-3-month response、constraint ratio 和 qualification status。

### Table 3 — Normal versus delayed

early response、lead-12 response、late-season gain、onset timing 和 qualified-case count。

### Table 4 — Optional CNOP / gradient / random comparison

只有正式对照完成后加入。

---

# 8. Evidence Map

| Section | Evidence/source | Role |
|---|---|---|
| Introduction: AI forecasting | `dai2026ai`, `luo2026ai` | AI weather/climate forecasting context |
| Introduction: CNOP | `mu2003cnop`, `guo2026cnop` | CNOP and nonlinear XAI formulation |
| Introduction: ENSO predictability | `qin2024enso`, `mu2025predictability` | ENSO and AI predictability background |
| Methods: WalkerNet | `src/model.py`, `configs/default.yaml` | Architecture and tensor shapes |
| Methods: CNOP | `scripts/cnop/compute_tos_zos_cnop.py` | Objective, constraint and optimization |
| Methods: case selection | frozen manifest and case-screen records | Case eligibility |
| Results: Pacific | Pacific summary / NPZ / overview | Main empirical evidence |
| Results: delayed-onset | delayed summaries and response-evolution files | Temporal-structure evidence |
| Results: controls | matched gradient/random outputs, if complete | Comparative evidence |
| Discussion | results plus CNOP literature | Interpretation and limitations |

---

# 9. 建议写作顺序

1. 冻结 anomaly、ENSO threshold 和 lead 定义；
2. 确认 normal / delayed objective 的准确公式；
3. 完成 Pacific 结果和文件审计；
4. 确认 Global、Indian 和 gradient/random 是否进主文；
5. 建立“结果—图—表—结论”证据矩阵；
6. 先写 Methods；
7. 再写 Results；
8. 根据 Results 写 Discussion；
9. 再写 Introduction；
10. 最后写 Abstract、Title 和 Conclusion；
11. 进行 citation、figure、table、anomaly 和数值口径一致性检查。

---

# 10. 写作前检查清单

- [ ] 论文主线是否明确为 Pacific delayed-onset；
- [ ] normal 和 delayed objective 是否与代码完全一致；
- [ ] truth / baseline / perturbed 的 climatology 口径是否明确；
- [ ] 所有案例的 summary / NPZ / candidate 文件是否完整；
- [ ] map-derived 数值与 summary 是否一致；
- [ ] 所有图的经纬度范围和色标是否统一记录；
- [ ] 所有图注是否说明平滑和插值仅用于可视化；
- [ ] gradient/random 是否有正式完成的统计证据；
- [ ] Global / Indian 是否放主文或 Supplement；
- [ ] 所有引用是否已在参考文献中出现；
- [ ] 所有现实机制表述是否与模型内部证据边界一致；
- [ ] Limitations 是否明确写出模型依赖和案例数量限制。
