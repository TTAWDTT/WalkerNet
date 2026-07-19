# SSP126 首轮训练失败复盘

## 结论

首轮 SSP126 实验不是因为 CDO remap 或时间对齐损坏而失败。核心问题是：将历史模型后期使用的长 rollout 微调配方，直接用于随机初始化模型，跳过了历史实验中约 10000 次更新的 12-step 基础训练阶段。

## 已排除的数据问题

- 五个 source、四个变量均为连续的 2015-01 至 2100-12，共 1032 个月。
- 网格均为 `lat=-89.5..89.5`、`lon=0.5..359.5`、`180x360`。
- 四变量时间轴严格一致，没有时变缺测月。
- historical 与 SSP126 的海陆 mask 逐格一致。
- 单位一致：`tos=degC`、`zos=m`、`tauu/tauv=Pa`。
- SSP126 的归一化均值和标准差与 historical 同量级。
- SSP126 模型在 test 上的 lead-1 Niño3.4 ACC 为 0.932，排除了整体月份错位。

CESM2 historical 使用 `r1i1p1f1`，SSP126 使用 `r10i1p1f1`。这不影响 SSP126 独立训练，但两段数据不能跨 2014/2015 构造连续样本。

## 训练方案为何不等价

| 项目 | Historical 基础训练 | SSP126 首轮训练 |
|---|---:|---:|
| 训练年份/source | 151 | 71 |
| optimizer steps/epoch | 约 560 | 约 258 |
| 最佳/停止时累计 steps | 约 10080 | 约 2580 |
| 初始学习率 | `1e-4` | `3e-5` |
| 基础 rollout | 固定 12-step | 12→15→18-step |
| 早停 | 基础训练充分后 | epoch 4 起、patience 3 |

SSP126 在 epoch 8 切换到 18-step 后，`val_loss` 从 epoch 7 的 0.195 上升到约 0.247，`ACC@18` 从 0.101 下降到 0.033，之后接近 0。

## 交叉评测证据

将 historical checkpoint 不经 SSP126 训练，直接在 SSP126 test 上评测：

| Lead | SSP126 首轮 best | Historical zero-shot |
|---:|---:|---:|
| 3 | 0.806 | **0.919** |
| 6 | 0.440 | **0.734** |
| 9 | 0.299 | **0.594** |
| 12 | 0.271 | **0.486** |
| 18 | 0.082 | **0.156** |

SSP126 数据能够被 historical 模型直接预测，说明数据本身保留了可学习的海气动力学。首轮 SSP126 训练没有获得足够的基础优化预算。

## 修正方案

### Stage 1：12-step 基础训练

- 随机初始化 WalkerNet。
- 固定 12-step rollout。
- 使用 historical 基础训练的 loss 和 `lr=1e-4`。
- 至少训练到 epoch 40，约等于 10000 次 optimizer update 后才允许早停。
- checkpoint 单独写入 `checkpoints_ssp126_scratch_stage1_ddp8/`。

### Stage 2：长 rollout 微调

- 从 Stage 1 的 `best_skill.pt` 初始化，不恢复 Stage 1 optimizer。
- 学习率降到 `1e-5`。
- epoch 1-7 使用 12-step，8-15 使用 15-step，之后使用 18-step。
- checkpoint 单独写入 `checkpoints_ssp126_scratch_stage2_ddp8/`。

服务器流水线：

```bash
bash scripts/train/run_ssp126_scratch_pipeline.sh
```

流水线在中断后会从当前 stage 的 `latest.pt` 恢复；Stage 1 正常结束并产生 `best_skill.pt` 后，才会启动 Stage 2。

## 后续评测改进

当前 checkpoint 选择把五个 source 拼接后计算 pooled ACC。后续应同时计算每个 source 的 ACC，并以 macro-average 作为补充选择指标，避免强 source 掩盖弱 source。
