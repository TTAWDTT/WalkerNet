# Field Forecast Rollout / Loss Design 调研记录

日期：2026-06-04

## 结论摘要

这轮调研支持 WalkerNet 下一步改成：

1. 预测相对最近输入场的 residual / delta，而不是直接从零生成完整场。
2. 训练时加入 autoregressive rollout，让模型在训练中见到自己的预测输入。
3. MSE 可以保留为主 loss，但不应单独使用；应加入 residual loss、gradient/spectral 类结构约束，以及从预测场计算出来的 Niño3.4 辅助 loss。

## 关键依据

### GraphCast

- 论文：GraphCast: Learning skillful medium-range global weather forecasting
- URL：https://arxiv.org/abs/2212.12794
- 相关点：
  - GraphCast 是 autoregressive forecast model，会把自己的预测继续喂回输入。
  - decoder 预测的是相对最近输入状态的 residual update。
  - 训练 objective 是在多个 autoregressive step 上计算 MSE，并且训练过程中逐步把 step 数从 1 增加到 12。
  - 论文也提到，多步 MSE 有利于长 lead，但会鼓励更平滑/模糊的预测。

对 WalkerNet 的启发：

- 我们当前直接输出完整场，不如改成 `y_pred = x_last + delta`。
- 如果评测要 rollout，训练也应 rollout-aware，否则 train/eval mismatch 会很明显。

### FourCastNet

- 论文：FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators
- URL：https://arxiv.org/abs/2202.11214
- 相关点：
  - 先训练单步 `X(t) -> X(t+1)`。
  - fine-tuning 阶段让模型用自己的第一步输出继续预测第二步。
  - loss 同时比较第一步和第二步预测与对应 truth。
  - 推理时 free-running autoregressive inference。

对 WalkerNet 的启发：

- 可以采用课程式训练：先短 rollout，再增加 rollout length。
- 多步训练不一定一开始就上 18 步，可以从 2/3/6 步逐步增加。

### FastNet / ML Weather Prediction Loss

- 论文：FastNet: Improving the physical consistency of machine-learning weather prediction models through loss function design
- URL：https://arxiv.org/abs/2509.17601
- 相关点：
  - 标准 MSE 容易带来模糊、小尺度结构损失。
  - 论文测试了 spherical harmonic / spectral amplitude loss、horizontal gradient loss 等结构约束。
  - 单独使用这些 loss 可能略伤 RMSE，但和 MSE 组合可以保持接近 MSE 的分数，同时提升谱结构和物理一致性。

对 WalkerNet 的启发：

- 纯 MSE 不够，尤其会让 `tos/zos` 的异常与空间结构偏平滑。
- 第一版可以先用 gradient loss，后续再考虑 spectral loss。

### ENSO-PhyNet

- 论文：Incorporating heat budget dynamics in a Transformer-based deep learning model for skillful ENSO prediction
- URL：https://www.nature.com/articles/s41612-024-00741-y
- 相关点：
  - 模型预测 SST 场，再评估 Niño3.4 skill。
  - 加入热收支相关物理过程后，尤其在 6-13 lead month 改善明显。
  - 强调物理一致性与可解释性，不只是追求 index 分数。

对 WalkerNet 的启发：

- 我们必须保留 field-first，再从 `tos` 场计算 Niño3.4。
- 如果当前变量只有 `tos/zos/tauu/tauv`，至少 loss 上要强调 Niño3.4 区域和异常演变。

## 建议的 WalkerNet 下一版 loss

建议第一版不要过度复杂，先做：

```text
loss_lead =
  w_field * MSE(y_pred, y)
+ w_delta * MSE(y_pred - x_last, y - x_last)
+ w_grad  * MSE(grad(y_pred), grad(y))
+ w_nino  * MSE(Nino34(y_pred_tos), Nino34(y_tos))
```

然后对 rollout lead 求加权和：

```text
loss = sum_k lead_weight[k] * loss_lead_k
```

建议初始权重：

```text
w_field = 1.0
w_delta = 0.5
w_grad  = 0.05 ~ 0.1
w_nino  = 0.1 ~ 0.2
```

建议 rollout 权重：

```text
lead_weight = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
```

对应 lead 可先训练 6 步，再扩到 12/18。

## 需要避免

- 不建议只换成 MAE/Huber 来解决平滑问题；它们改变误差惩罚形态，但不能保证空间结构和 ENSO 区域演变。
- 不建议直接预测 Niño3.4 作为主任务；这不符合“先预报场，再算指数”的要求。
- 不建议只做单步训练再指望长 rollout 自然变好；这会有明显 exposure bias。
