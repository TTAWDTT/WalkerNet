# WalkerNet rollout accuracy record

## Current status

WalkerNet 的 Niño3.4 推理 skill 已有部分汇总结果，但当前本地工作区没有完整的 lead-1--18 逐月评测表，也没有按预测起始月分组的 lead-12 评测表。

当前结果均应明确区分 checkpoint、split、异常定义和统计口径，不能把不同 checkpoint 的表格混合成一套曲线。

## Existing accuracy records

### `best_skill.pt` test evaluation

来源：`README.md`。测试样本数为 305，最大 rollout lead 为 24 个月；当前公开表格列出 lead 3/6/9/12/18/24。

| Lead | Monthly ACC | Monthly RMSE | 3-month ACC | 3-month RMSE |
|---:|---:|---:|---:|---:|
| 3 | 0.915 | 0.520 | 0.958 | 0.363 |
| 6 | 0.834 | 0.769 | 0.873 | 0.658 |
| 9 | 0.755 | 1.007 | 0.789 | 0.904 |
| 12 | 0.701 | 1.177 | 0.729 | 1.098 |
| 18 | 0.636 | 1.346 | 0.663 | 1.294 |
| 24 | 0.465 | 1.493 | 0.504 | 1.461 |

README 同时记录了 persistence ACC/RMSE 对照。lead-24 属于训练课程最长 18 个月之外的外推评测。

### `checkpoints_rollout12_0607/best.pt` record

来源：`docs/results.md`。该表包含 lead 1/3/6/9/12/18，但不是连续的 lead-1--18 表。

| Lead | Model ACC | Model RMSE |
|---:|---:|---:|
| 1 | 0.9743 | 0.3403 |
| 3 | 0.8449 | 0.8241 |
| 6 | 0.5820 | 1.2970 |
| 9 | 0.3809 | 1.5038 |
| 12 | 0.2214 | 1.6869 |
| 18 | -0.0293 | 1.9639 |

这轮 checkpoint 与 `best_skill.pt` 的表格不能直接拼接或当作同一模型曲线。

## What was not yet available before the formal run

1. 固定起始月协议下 lead 1、2、3、...、18 的完整 monthly ACC/RMSE 和 3-month mean ACC/RMSE。
2. 按预测起始月份分组的 lead-12 ACC/RMSE：January start、February start、...、December start。
3. 对应的逐月样本数、置信区间/Bootstrap 区间和 persistence 分组对照。

## Formal evaluation completed: `historical_mixed5_best_skill.pt`

日期：2026-08-25。使用 GPU007 上的正式评测，checkpoint、test split、source-wise training climatology 和 rollout policy 全程固定。评测请求 `max_lead=18`、`leads=1,2,...,18`；由于必须完整 rollout 到 lead-18，最终有效样本为 **335**（原 test 样本 420）。这是一个新的、内部一致的 checkpoint 结果，不应与上面的 README 或 `checkpoints_rollout12_0607/best.pt` 数字拼接。

远端输出目录：

```text
/data/WalkerNet/outputs/eval_rollout_best_skill_test_20260825/
```

正式输出包括：

```text
eval_rollout_best_skill_monthly_lead1_18.csv
eval_rollout_best_skill_monthly_lead1_18.json
eval_rollout_best_skill_lead12_by_start_month.csv
eval_rollout_best_skill_lead12_by_start_month.json
test_rollout_metrics.json
test_rollout_field_metrics.csv
test_rollout_nino34_lead_metrics.csv
test_rollout_nino34_lead_acc.png
```

### Monthly Niño3.4 anomaly skill（selected leads）

下表是新 checkpoint 的 monthly anomaly 指标；完整 lead-1--18 表在上述 CSV/JSON 中。

| Lead | Model ACC | Model RMSE | Persistence ACC | Persistence RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.971 | 0.289 | 0.904 | 0.533 |
| 3 | 0.909 | 0.530 | 0.585 | 1.247 |
| 6 | 0.839 | 0.767 | 0.266 | 1.707 |
| 9 | 0.768 | 0.992 | -0.006 | 1.912 |
| 12 | 0.716 | 1.144 | -0.290 | 1.952 |
| 18 | 0.633 | 1.317 | -0.405 | 2.390 |

### Lead-12 grouped by input-window end / first forecast target month

这里的 `start_month` 是输入窗口结束、首个预测目标月份，即模型调用时的第一个 `target_month`；不是 lead-12 对应月份。每个月的 model/persistence 样本数为 30（January--July）或 25（August--December），总计 335。

| Start month | n | Model ACC | Model RMSE | Persistence ACC | Persistence RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 0.713 | 1.366 | -0.293 | 2.384 |
| 2 | 30 | 0.802 | 1.265 | -0.290 | 2.326 |
| 3 | 30 | 0.814 | 1.094 | -0.278 | 2.099 |
| 4 | 30 | 0.762 | 1.004 | -0.163 | 1.783 |
| 5 | 30 | 0.725 | 1.075 | -0.153 | 1.710 |
| 6 | 30 | 0.558 | 1.059 | -0.054 | 1.473 |
| 7 | 30 | 0.526 | 1.019 | -0.049 | 1.463 |
| 8 | 25 | 0.681 | 0.995 | -0.350 | 1.604 |
| 9 | 25 | 0.774 | 1.052 | -0.433 | 1.730 |
| 10 | 25 | 0.657 | 1.257 | -0.458 | 1.960 |
| 11 | 25 | 0.765 | 1.196 | -0.501 | 2.188 |
| 12 | 25 | 0.756 | 1.287 | -0.484 | 2.403 |

本轮新增评测器已经把这两组结果写成独立 CSV/JSON；尚未计算 bootstrap 置信区间，因此论文中如果需要误差棒，应在这组固定结果之上再做统计后处理。

## Reproducible evaluation protocol

评测脚本：`src/evaluate_rollout.py`。

固定起始月的完整 lead 表应使用同一个 checkpoint、同一个 split、同一个数据缓存和同一个 climatology 定义，并请求：

```text
max_lead = 18
leads = 1,2,3,...,18
```

每个 lead 同时输出：

- monthly Niño3.4 anomaly ACC；
- monthly Niño3.4 anomaly RMSE；
- 3-month mean ACC；
- 3-month mean RMSE；
- persistence ACC/RMSE；
- 模型相对 persistence 的改善量。

异常定义必须保持当前约定：从预测 TOS 场计算 Niño3.4，再减去对应 source/month 的气候态；模型不直接预测 Niño3.4 指数。

## Start-month evaluation

“不同起始月的 lead-12 准确性”中的起始月，定义为输入窗口结束、开始预测的 target month，而不是 lead-12 对应月份。

评测器需要在有效 rollout 样本筛选后，按每个样本的 target month 分组，分别统计：

```text
January start   -> lead-12 ACC/RMSE
February start  -> lead-12 ACC/RMSE
...
December start  -> lead-12 ACC/RMSE
```

建议输出：

```text
eval_rollout_best_skill_monthly_lead1_18.csv
eval_rollout_best_skill_monthly_lead1_18.json
eval_rollout_best_skill_lead12_by_start_month.csv
eval_rollout_best_skill_lead12_by_start_month.json
```

每个按月分组的 row 应至少包含：`start_month`、`n_samples`、`lead`、`monthly_acc`、`monthly_rmse`、`three_month_acc`、`three_month_rmse`、`persistence_acc`、`persistence_rmse`。

## Resource note

正式评测使用的是 GPU007 的 GPU0；启动时 GPU0--7 均为真正空闲状态（显存约 0--4 MiB / 143771 MiB）。评测完成后已释放 GPU0，其他卡未被占用。
