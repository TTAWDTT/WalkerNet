# WalkerNet 模型升级记录

## 1. 改动目标

当前模型已经满足基本输入输出接口：

```text
model(x, target_month, rollout_step=None) -> y_pred
```

其中：

```text
x:      (B, L, 4, 180, 360)
y_pred: (B, 1, 4, 180, 360)
```

## 2. 当前主要问题

当前 `PatchEmbedding` 的核心做法是：

```text
(B, L, 4, H, W)
-> reshape
(B, L*4, H, W)
-> Conv2d patch projection
(B, N, d)
```

这个设计能运行，但存在明显不足：

1. 时间维被直接压进 channel，模型没有显式的历史位置编码。
2. 变量维被直接压进 channel，模型没有显式的变量身份编码。
3. patch 内的时间-变量交互主要依赖一次线性卷积投影，表达力不足。
4. 对 ENSO 相关任务来说，`tos/zos/tauu/tauv` 之间的局地耦合和时间滞后关系很关键，不应在模型最前端被过早压扁。


## 3. 新的 Patch Embedding 设计

新的前端应保留时间维和变量维，先得到每个时间步、每个变量、每个空间 patch 的 token：

```text
输入:
(B, L, 4, 180, 360)

per-time/per-variable patchify:
(B, L, 4, 45, 90, d)

展平空间 patch:
(B, L, 4, 4050, d)
```

其中：

```text
45 = 180 / 4
90 = 360 / 4
4050 = 45 * 90
```

然后在每个空间 patch 内，把 `L * 4` 个时间-变量 token 放在一起：

```text
(B, L, 4, 4050, d)
-> (B, 4050, L*4, d)
```

默认 `L=3` 时，每个空间 patch 内有：

```text
3 * 4 = 12 个 token
```


## 4. 时间编码

时间编码应包含两类信息。

### 4.1 相对历史位置编码

相对历史位置表示输入窗口内部的远近顺序：

```text
0 -> 最早的历史月
1 -> 中间历史月
...
L-1 -> 最近的历史月
```

建议使用可学习 embedding：

```text
relative_time_embed: (L, d)
```

加到 patch token 上时，通过广播变为：

```text
(B, L, 4, N, d)
```

对应形式：

```python
z = z + relative_time_embed[None, :, None, None, :]
```

### 4.2 日历月份编码

日历月份表示该历史时间步实际对应几月。  
它不同于相对历史位置。

例如：

```text
target_month = 7
L = 3

输入历史月份:
4 月, 5 月, 6 月
```

建议使用可学习月份 embedding：

```text
calendar_month_embed: (12, d)
```

输入月份可以由 `target_month` 和 `L` 推出：

```text
input_month[k] = target_month - L + k
```

再做 12 个月循环取模，得到范围 `1-12` 的月份。

加到 patch token 上时：

```python
z = z + calendar_month_embed[input_month][:, :, None, None, :]
```


## 5. 变量编码

变量编码用于显式告诉模型当前 token 属于哪个物理变量：

```text
0 -> tos
1 -> zos
2 -> tauu
3 -> tauv
```

建议使用可学习 embedding：

```text
variable_embed: (4, d)
```

加到 patch token 上：

```python
z = z + variable_embed[None, None, :, None, :]
```

加完之后，每个 token 同时携带：

1. 空间 patch 信息。
2. 相对历史位置。
3. 日历月份。
4. 物理变量身份。


## 6. Patch 内 Time-Variable Fusion

加入时间编码和变量编码后，需要在每个空间 patch 内融合 `L * 4` 个 token：

```text
(B, N, L*4, d)
-> fusion
(B, N, d)
```

使用轻量 attention pooling 或小型 Transformer block。

推荐优先方案：

```text
learnable query
+ MultiheadAttention over L*4 tokens
+ FFN
+ residual
```

这样每个空间 patch 内部可以学习：

1. 不同历史月份之间的关系。
2. 不同变量之间的关系。
3. 某个变量在某个滞后月份对预测的贡献。

该 fusion 只在每个 patch 内部处理 `L*4` 个 token，计算量远小于全局空间 attention。


## 7. 后续主干结构

Patch 内融合之后，输出仍保持：

```text
(B, N, d)
```

因此后续主干接口可以保持不变：

```text
(B, N, d)
-> Spatial Attention
-> TMoE
-> Decoder
-> (B, 1, 4, 180, 360)
```

新的完整张量流为：

```text
(B, L, 4, 180, 360)
-> (B, L, 4, 45, 90, d)
-> (B, L, 4, 4050, d)
-> 加 relative time embedding
-> 加 calendar month embedding
-> 加 variable embedding
-> (B, 4050, L*4, d)
-> patch 内 time-variable fusion
-> (B, 4050, d)
-> Spatial Attention
-> TMoE
-> Decoder
-> (B, 1, 4, 180, 360)
```


## 8. 对外接口保持不变

模型对外接口不应改变：

```python
y_pred = model(x, target_month, rollout_step=None)
```

也就是说，数据侧和训练侧仍只需要提供：

```text
x:            (B, L, 4, 180, 360)
target_month: (B,)
```

新增的输入历史月份应由模型内部根据 `target_month` 和 `L` 推出，不要求 Dataset 额外返回。


## 9. 需要同步修改的内容

确认该设计后，需要修改：

1. `src/model.py`
   - 替换当前 `PatchEmbedding`。
   - 新增时间编码、月份编码、变量编码。
   - 新增 patch 内 time-variable fusion 模块。

2. `tests/test_model_shapes.py`
   - 更新 PatchEmbedding 相关测试。
   - 增加 time embedding / variable embedding 梯度测试。
   - 增加 `target_month` 影响输入月份编码的测试。

3. `docs/architecture.md`
   - 将模型前端描述同步为新的设计。
   - 修正变量说明中仍写 `HC` 的问题，当前变量为 `zos`。

4. `configs/default.yaml`
   - 如需要，加入 patch fusion 的超参数，例如：

```yaml
model:
  patch_fusion_heads: 4
  patch_fusion_layers: 1
```