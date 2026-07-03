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

## 优化方法

使用 PyTorch autograd 直接穿过 WalkerNet rollout 链：

```text
delta <- Adam ascent on J(delta)
delta <- project(delta)
```

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
cnop_gain_summary.png
best_case_cnop_maps_and_nino.png
```

其中 `best_case_cnop_maps_and_nino.png` 展示：

1. 最强 case 的 TOS CNOP 空间图；
2. 最强 case 的 ZOS CNOP 空间图；
3. baseline 与 CNOP 后 monthly Niño3.4 anomaly；
4. baseline 与 CNOP 后三个月平均 Niño3.4 anomaly。

## 注意事项

这是“模型内 CNOP”，即对 WalkerNet 这个非线性预报算子的最优扰动，不等价于真实气候系统的严格 CNOP。若扰动图出现尖峰或不物理结构，需要继续加入更严格的物理约束、平滑约束或更小扰动半径敏感性实验。
