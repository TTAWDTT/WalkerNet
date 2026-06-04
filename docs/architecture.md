# WalkerNet Architecture Notes

本文档记录当前 `src/model.py` 已实现的神经网络结构。
模型对外接口保持为：

```python
y_pred = model(x, target_month, rollout_step=None)
```


## Task Definition

- 输入：连续 `L` 个月的全球物理场。
- 输出：下 1 个月的全球物理场。
- 空间网格：`180 x 360` 的 1 度规则经纬度网格。
- CNOP 研究在预报模型训练完成后进行，本模型本身只负责预报。


## Variables

张量中的变量顺序固定为：

```text
0 -> tos   Sea surface temperature
1 -> zos   Sea surface height above geoid
2 -> tauu  Zonal wind stress
3 -> tauv  Meridional wind stress
```


## Shapes

```text
x:      (B, L, 4, 180, 360)
y_pred: (B, 1, 4, 180, 360)
```

默认配置中：

```text
L = 3
patch_size = 4
N = (180 / 4) * (360 / 4) = 4050
d = d_model
```


## Pipeline

整体流程：

```text
(B, L, 4, 180, 360)
-> Explicit Time-Variable Patch Embedding
-> (B, 4050, d)
-> Rollout Step Embedding
-> Spatial Attention Blocks
-> TMoE
-> Coupled Variable Decoder
-> (B, 1, 4, 180, 360)
```


## Explicit Time-Variable Patch Embedding

模型前端不再把 `L x 4` 直接压成 channel。
当前实现分为四步。

### 1. 变量专属 Patch Projection

每个变量使用自己的浅层 patch projection，同一变量跨时间共享权重：

```text
(B, L, 4, 180, 360)
-> (B, L, 4, 4050, d)
```

这一步只负责把单变量、单时间步的局地 `4 x 4` 网格块转成 token，不负责变量交互。

### 2. 时间、月份、变量编码

每个 patch token 加入三类 embedding：

```text
relative_time_embed   历史窗口内的位置
calendar_month_embed  该历史步实际对应的月份
variable_embed        tos / zos / tauu / tauv 的变量身份
```

其中输入历史月份由 `target_month` 和 `L` 在模型内部推出，不要求 Dataset 额外返回。

### 3. Patch 内 Time-Variable Fusion

对每个空间 patch 内的 `L * 4` 个 token 加入一个 learnable fusion token，并做轻量 self-attention：

```text
(B, 4050, L*4, d)
-> 加 fusion token
-> (B, 4050, 1 + L*4, d)
-> FusionAttentionBlock
-> 取 fusion token
-> (B, 4050, d)
```

这个模块只建模同一空间 patch 内部的时间-变量关系。
不同空间 patch 之间的全球交互交给后续 Spatial Attention。

### 4. 二维空间位置编码

融合后加入二维 patch 位置编码：

```text
pos = lat_patch_embed + lon_patch_embed
```

输出仍为：

```text
(B, 4050, d)
```


## Rollout Step Embedding

`rollout_step` 用于自回归预测时注入 lead-time 条件。
单步训练时可以传 `None`，等价于 `rollout_step = 0`。


## Spatial Attention Blocks

主干使用多层 pre-norm Transformer block：

```text
LayerNorm
MultiheadAttention
Residual
LayerNorm
FFN
Residual
```

形状保持：

```text
(B, 4050, d) -> (B, 4050, d)
```

该部分用于建模全球空间遥相关。


## TMoE

TMoE 使用 `target_month` 做月份条件路由：

```text
target_month
-> month embedding
-> gate
-> top-k expert weights
```

每个 expert 是一个 FFN，TMoE 输出形状仍为：

```text
(B, 4050, d)
```


## Coupled Variable Decoder

Decoder 将 token map 还原为四变量全球场：

```text
(B, 4050, d)
-> (B, d, 45, 90)
-> Conv refine
-> PixelShuffle x2
-> Conv refine
-> PixelShuffle x2
-> Conv2d to 4 variables
-> (B, 1, 4, 180, 360)
```

四个变量在 decoder 的 channel 维中联合解码。


## Conditioning Signals

当前模型使用三类条件信息：

1. `target_month`：预测目标月份，用于输入历史月份推导和 TMoE 路由。
2. `rollout_step`：自回归步数，用于 lead-time embedding。
3. 历史窗口相对位置：用于区分最早历史月和最近历史月。


## CNOP Integration

训练完成后，可以利用模型自动微分计算 CNOP：

- 预报敏感性分析。
- CNOP 初始化的集合预报。
- 初始扰动对 ENSO 相关指标的影响分析。
