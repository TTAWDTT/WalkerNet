# WalkerNet TOS/ZOS CNOP 方法说明

## 目标

在已经训练好的 WalkerNet 上计算 CNOP-like 初始扰动。这里把 WalkerNet 的 12 个月自回归 rollout 看作非线性预报算子：

```text
F: x_{Jan-Dec}^{Y-1} -> \hat{x}_{Jan-Dec}^{Y}
```

目标是在一个本来不发生 ENSO 的目标年 `Y` 上，寻找施加于输入窗口最后一个月，即 `Y-1` 年 12 月的 `tos/zos` 扰动，使模型预报出的目标年 Niño3.4 anomaly 最大。

## Neutral Case 选择

脚本从所有 source 的历史年份中寻找目标年 `Y`，要求：

```text
输入窗口:   Y-1 年 1 月 - 12 月
目标窗口:   Y   年 1 月 - 12 月
```

对真实 `tos` 计算目标年 Niño3.4 anomaly，并取三个月滑动平均。默认选择 observed `max(abs(3-month Niño3.4 anomaly))` 最小的 10 个 Jan-Dec 年段。

这样做是为了满足“目标年本来不发生 ENSO”的设定。

## 扰动变量

当前只优化：

```text
tos: 海表温度
zos: 海面高度 / 上层海洋状态代理
```

扰动只作用在输入窗口的第 12 个月：

```text
x'[:, 11, tos] = x[:, 11, tos] + delta_tos
x'[:, 11, zos] = x[:, 11, zos] + delta_zos
```

其余月份和 `tauu/tauv` 不变。

默认不直接优化每个 1° 格点的独立扰动，而是在 `45 x 90` patch 网格上优化，再双线性上采样到 `180 x 360`。这样可以减少棋盘格和逐格点噪声，也更接近模型自己的 patch 表示尺度。

## 默认扰动区域

默认只允许热带太平洋区域有扰动：

```text
lat: 20S - 20N
lon: 120E - 290E
```

这是为了减少全局任意格点扰动带来的非物理自由度，也更接近 ENSO 前兆问题。

## 约束

优化在归一化空间进行。默认每个变量分别满足：

```text
RMS(delta_tos) <= 0.5
RMS(delta_zos) <= 0.5
abs(delta)     <= 2.0
```

初版 smoke test 发现 `0.5` 半径会把 WalkerNet 内部响应推到过强，容易变成模型敏感性利用而不是可解释前兆。因此当前推荐主实验使用更保守的：

```text
RMS(delta_tos) <= 0.1
RMS(delta_zos) <= 0.1
smoothness_weight = 0.001
```

这里的 0.1 表示训练归一化后的 0.1 个标准差量级。脚本每一步优化后都会把扰动投影回约束集合。

也可以使用更接近经典 CNOP 表述的相对二范数约束：

```bash
--constraint-mode relative_initial_l2 \
--relative-l2-fraction 0.1
```

此时在物理量空间、扰动允许区域内分别满足：

```text
||delta_tos||_2 <= 0.1 * ||tos_initial||_2
||delta_zos||_2 <= 0.1 * ||zos_initial||_2
```

其中 `tos_initial/zos_initial` 指输入窗口第 12 个月的初始场。

## 目标函数

对扰动后的输入做 12 个月 rollout：

```text
\hat{x}'_1, ..., \hat{x}'_12 = F(x + delta)
```

从每个月预测 `tos` 场计算 Niño3.4 anomaly，然后计算目标年内所有三个月滑动平均：

```text
N_3, N_4, ..., N_12
```

优化目标是这些三个月平均值的 soft maximum：

```text
J(delta) = tau * logsumexp([N_3, ..., N_12] / tau)
```

默认 `tau = 0.25`。这样比直接 `max` 更平滑，适合梯度优化。

如果希望按“第 12 个月扰动前后 Niño3.4 指数差”定义 CNOP，可以使用：

```bash
--objective-mode lead_delta \
--objective-lead 12
```

此时目标函数变为：

```text
J(delta) = Nino3.4_12(F(x + delta)) - Nino3.4_12(F(x))
```

其中 `F(x)` 是未加扰动的 baseline rollout，`F(x + delta)` 是加扰动后的 rollout。

## 优化方法

使用 PyTorch autograd 直接穿过 WalkerNet rollout 链。当前脚本不再只从单个零扰动初值出发，而是默认做多初值搜索：

```text
for each neutral case:
    for start in 1..num_starts:
        delta_start <- zero or random perturbation
        delta <- Adam ascent on J(delta)
        delta <- project(delta)
    keep top-k local CNOP candidates
    optionally refine top-k with projected L-BFGS
```

默认参数：

```text
num_starts = 16
top_k = 5
random_init_scale = 0.02
lbfgs_steps = 0
```

也就是说，默认保存每个 case 中目标函数最强的 5 个局部最优扰动。`lbfgs_steps` 默认为 0，是因为 L-BFGS 精修更贵，建议先用多初值 Adam 找候选，再对 top-k 做小步数精修实验。

这样做的原因是 CNOP 目标函数是非线性、多峰的。单个扰动只能称为一个局部 CNOP-like 解；多初值 top-k 可以检查同一个 case 是否存在多种不同触发路径。

为节省显存：

- 冻结模型参数，只对 `delta` 求梯度；
- 使用 AMP；
- 对 rollout forward 使用 gradient checkpointing。

## 输出

脚本输出：

```text
cnop_summary.csv
method.json
case_{source}_{year}.npz
case_{source}_{year}_history.json
case_{source}_{year}_candidates.json
cnop_gain_summary.png
best_case_cnop_maps_and_nino.png
cnop_candidate_summary.csv
```

其中 `best_case_cnop_maps_and_nino.png` 展示：

1. 最强 case 的 TOS CNOP 空间图；
2. 最强 case 的 ZOS CNOP 空间图；
3. baseline 与 CNOP 后 monthly Niño3.4 anomaly；
4. baseline 与 CNOP 后三个月平均 Niño3.4 anomaly。

`case_{source}_{year}.npz` 仍保留旧字段 `delta_norm/delta_phys/cnop_nino/cnop_3m`，它们对应 top-1 最优扰动；同时新增：

```text
top_delta_norm
top_delta_phys
top_cnop_nino
top_cnop_3m
top_objective
top_lead_nino
top_lead_delta
top_cnop_max_3m
top_gain_max_3m
top_start_idx
```

这些字段用于比较同一个 case 的多个局部 CNOP 候选。
`rank` 是候选扰动在同一个 case 内按 `objective` 从高到低排序后的序号；
它不是 EOF/主成分意义上的物理模态编号。如果多个 rank 的形态和目标函数值都很接近，
通常应解释为多次优化收敛到同一个主导敏感方向，而不是发现了多个可分辨的局地最优模态。

完成 CNOP 后，可以用下面的诊断脚本生成更适合汇报和论文草图的合成图：

```bash
python scripts/cnop/plot_cnop_diagnostics.py \
    --input-dir outputs/cnop_tos_zos_patch_0703
```

该脚本额外输出：

```text
figures/cnop_composite_diagnostics.png
figures/cnop_composite_diagnostics.pdf
figures/cnop_precursor_diagnostics.png
figures/cnop_precursor_diagnostics.pdf
figures/cnop_tos_case_atlas.png
figures/cnop_tos_case_atlas.pdf
figures/cnop_zos_case_atlas.png
figures/cnop_zos_case_atlas.pdf
figures/cnop_factor_comparison.png
figures/cnop_factor_comparison.pdf
figures/cnop_precursor_indices.csv
figures/cnop_precursor_analysis.md
```

## 注意事项

这是“模型内 CNOP”，即对 WalkerNet 这个非线性预报算子的最优扰动，不等价于真实气候系统的严格 CNOP。若扰动图出现尖峰或不物理结构，需要继续加入更严格的物理约束、平滑约束或更小扰动半径敏感性实验。
