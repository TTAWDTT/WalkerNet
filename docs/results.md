# WalkerNet 实验结果记录

本文档记录当前可复现实验结论，重点关注自由滚动 rollout 后从预测 `tos` 场计算的 Niño3.4 anomaly skill。

## 固定评测协议

- 输入：过去 12 个月、4 个变量、`180 x 360` 全球 1° 网格。
- 输出：下一月 4 变量场。
- 长 lead：逐月 autoregressive rollout。
- 测试集：`2008-2014`。
- 评测 lead：`1, 3, 6, 9, 12, 18` 月。
- 对照：persistence，即输入窗口最后一个月场保持不变。
- 指标：Niño3.4 anomaly RMSE / ACC，指数从预测 `tos` 场计算，不直接预测指数。

## 6-step rollout 训练

Checkpoint:

```text
/mnt/sda/WalkerNet/checkpoints_rollout_0606/best.pt
```

主要结论：

- 单步和 1/3/6 月明显赢 persistence。
- 12/18 月 RMSE 仍然输 persistence。
- 修复 `evaluate_rollout.py` 中未传 `rollout_step` 的问题后，长 lead ACC 有提升，但不足以单独解决 12/18 月问题。

## 12-step rollout fine-tune

Checkpoint:

```text
/mnt/sda/WalkerNet/checkpoints_rollout12_0607/best.pt
```

训练方式：

- 从 6-step best 初始化。
- `rollout_steps=12`。
- 低学习率 fine-tune。
- 训练停止在 epoch 19 附近；最佳 checkpoint 为 epoch 13。

### Monthly Niño3.4 anomaly

| Lead | Model RMSE | Persistence RMSE | RMSE 改善 | Model ACC | Persistence ACC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3403 | 0.4845 | 29.8% | 0.9743 | 0.9520 |
| 3 | 0.8241 | 1.2948 | 36.4% | 0.8449 | 0.7016 |
| 6 | 1.2970 | 2.0019 | 35.2% | 0.5820 | 0.3109 |
| 9 | 1.5038 | 2.2648 | 33.6% | 0.3809 | 0.0031 |
| 12 | 1.6869 | 2.4001 | 29.7% | 0.2214 | -0.2451 |
| 18 | 1.9639 | 2.8585 | 31.3% | -0.0293 | -0.4230 |

### 3-month mean Niño3.4 anomaly

| Lead | Model RMSE | Persistence RMSE | RMSE 改善 | Model ACC | Persistence ACC |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.5620 | 0.8682 | 35.3% | 0.9268 | 0.8524 |
| 6 | 1.1316 | 1.7729 | 36.2% | 0.6835 | 0.4480 |
| 9 | 1.4135 | 2.1641 | 34.7% | 0.4487 | 0.1044 |
| 12 | 1.5975 | 2.3108 | 30.9% | 0.2792 | -0.1707 |
| 18 | 1.9106 | 2.8147 | 32.1% | 0.0041 | -0.4250 |

## 当前判断

12-step rollout 训练已经解决了“长 lead RMSE 完全比不过 persistence”的问题。当前瓶颈转为：

- 9/12/18 月 ACC 仍偏弱；
- 长 lead anomaly 相位与振幅需要进一步约束；
- `val_loss` 与真正的 rollout skill 不完全一致。

因此下一轮重点不是继续训更久，而是：

1. 使用 rollout-aware checkpoint selection，保存 `best_rollout.pt`。
2. 训练 loss 加入 area-weighted field loss。
3. 加入 Niño3.4 区域异常形态相关和振幅约束。
4. 从 `checkpoints_rollout12_0607/best.pt` 低学习率 fine-tune，而不是重新训练。
