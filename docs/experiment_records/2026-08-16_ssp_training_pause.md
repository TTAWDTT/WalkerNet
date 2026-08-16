# SSP 训练暂停记录（2026-08-16）

## 暂停位置

- 流水线：`walker_sourcewise_ddp8_full_0814`
- 当前模型：SSP245 Stage 3（15 -> 18 month rollout）
- 已完成 epoch：7
- 当前 rollout：18 months
- latest：`/mnt/sda/WalkerNet/checkpoints_ssp245_s15_s18_stage3/latest.pt`
- best skill：`/mnt/sda/WalkerNet/checkpoints_ssp245_s15_s18_stage3/best_skill.pt`
- best skill：0.661289（epoch 4，15-month rollout）
- 暂停时间：2026-08-16T23:18:49+08:00

暂停发生在 epoch 7 的验证和 `latest.pt` 保存完成后。训练进程已经退出，SSP370、SSP585 的 Stage 3 尚未开始。

## 恢复方式

从项目根目录执行：

```bash
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29553 \
bash scripts/train/run_sourcewise_s15_s18_sequence.sh
```

流水线会检测 SSP245 Stage 3 的 `latest.pt`，从 epoch 7 继续，并在 SSP245 完成后依次训练 SSP370、SSP585。

## 暂停原因

为 NeurIPS workshop 的 Historical CNOP 海盆控制实验释放 GPU。暂停不代表当前 SSP245 Stage 3 已经完成，后续模型比较应继续使用完整训练并正式评测过的 checkpoint。
