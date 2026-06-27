<p align="center">
  <img src="./logo.png" alt="WalkerNet logo" width="180">
</p>

<h1 align="center">WalkerNet</h1>

<p align="center">
  面向全球海气物理场的自回归预测模型，用预测场评估 ENSO 技巧。
</p>

<p align="center">
  <img alt="Task" src="https://img.shields.io/badge/task-global%20field%20forecasting-006D77">
  <img alt="Eval" src="https://img.shields.io/badge/eval-Ni%C3%B1o3.4%20ACC-14213D">
  <img alt="Grid" src="https://img.shields.io/badge/grid-1%C2%B0%20%7C%20180%C3%97360-2EC4B6">
  <img alt="Data" src="https://img.shields.io/badge/data-CESM2%20%2B%20CMIP6%20mixed-FFB703">
  <img alt="Framework" src="https://img.shields.io/badge/framework-PyTorch-EE4C2C">
</p>

---

## 项目目标

WalkerNet 的目标是：

```text
用过去 12 个月的全球海气物理场，预测未来 1 个月的全球场；
再把预测场自回归接回输入窗口，滚动得到 1-18 个月预报；
最后从预测 tos 场计算 Niño3.4 anomaly ACC / RMSE。
```

模型训练坚持 **field-first**：模型不直接输出 ENSO 指数，而是先预报完整物理场，再从场里计算指数。

## 张量约定

输入输出固定为：

```text
x:      (B, 12, 4, 180, 360)
y_pred: (B,  1, 4, 180, 360)
```

四个变量顺序固定：

| Channel | Variable | Meaning |
|---:|---|---|
| 0 | `tos` | 海表温度 |
| 1 | `zos` | 海面高度 |
| 2 | `tauu` | 纬向风应力 |
| 3 | `tauv` | 经向风应力 |

网格统一为 1° 全球规则网格：

```text
H = 180
W = 360
lat = -89.5 ... 89.5
lon =   0.5 ... 359.5
```

## 模型接口

模型实现需要遵守：

```python
y_pred = model(x, target_month, rollout_step=None)
```

| Name | Shape | Note |
|---|---|---|
| `x` | `(B, 12, 4, 180, 360)` | 归一化后的历史窗口 |
| `target_month` | `(B,)` | 目标月份，取值 `1-12` |
| `rollout_step` | `(B,)` 或 `None` | 第几次自回归滚动 |
| `y_pred` | `(B, 1, 4, 180, 360)` | 下一月预测场 |

## 当前训练策略

当前主线实验使用 5 个 source 混合训练：

```text
CESM2
EC-Earth3
GFDL-ESM4
IPSL-CM6A-LR
MPI-ESM1-2-HR
```

一个样本内部始终来自同一个 source，不同样本会混合 shuffle。

rollout curriculum：

```text
epoch <= 22: 12-step rollout
epoch <= 26: 15-step rollout
epoch >  26: 18-step rollout
```

每个 lead 同等重要：

```text
L_total = mean(L_1, L_2, ..., L_K)
```

单个 lead 的 loss：

```text
L_k =
1.0 * L_field
+ 0.1 * L_tropical_pacific
+ 0.3 * L_nino34
+ 0.1 * L_nino34_structure
```

含义：

| Loss | 作用 |
|---|---|
| `L_field` | 四变量全球场误差，主目标 |
| `L_tropical_pacific` | 热带太平洋四变量场误差 |
| `L_nino34` | 从预测 `tos` 计算 Niño3.4 区域平均并约束 |
| `L_nino34_structure` | 约束 Niño3.4 区域内部冷暖结构 |

## Rollout Skill 口径

训练中保存 `best_skill.pt` 使用的是验证集上的 `rollout_skill`，它不是 loss，而是 Niño3.4 anomaly ACC 的简化选择指标。

当前配置：

```yaml
rollout_selection:
  leads: [6, 9, 12, 18]
  mode: "three_month_mean"
  score: "mean_acc"
```

计算流程：

```text
1. 从验证样本滚动预测到 18 个月，得到完整预测场。
2. 从每个月预测 tos 场中计算 Niño3.4 区域平均。
3. 减去对应 source、对应月份的 Niño3.4 气候态，得到 anomaly。
4. 对指定 lead 使用三个月滑动平均：
   lead 6  = mean(lead 4, 5, 6)
   lead 9  = mean(lead 7, 8, 9)
   lead 12 = mean(lead 10, 11, 12)
   lead 18 = mean(lead 16, 17, 18)
5. 分别计算 acc@6 / acc@9 / acc@12 / acc@18。
6. rollout_skill = mean(acc@6, acc@9, acc@12, acc@18)。
```

因此，`rollout_skill` 主要用于训练过程中快速选择 checkpoint；正式报告仍应同时查看每个 lead 的 ACC、RMSE 和空间场评估。

## 常用命令

单卡训练：

```bash
python -m src.train --config configs/server_3090_mixed5.yaml
```

DDP 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --nproc_per_node=8 \
  -m src.train \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --num-workers 2
```

只加载模型权重作为新实验起点：

```bash
python -m src.train \
  --config configs/server_3090_mixed5.yaml \
  --init-checkpoint /path/to/best_skill.pt
```

完整恢复训练状态：

```bash
python -m src.train \
  --config configs/server_3090_mixed5.yaml \
  --resume /path/to/latest.pt
```

## 数据处理

原始数据需要先重网格到 1°：

```bash
bash scripts/remap_to_1x1.sh
```

检查重网格结果：

```bash
python scripts/check_remapped_data.py --data-dir data_1x1
```

大文件目录不应提交到 GitHub，例如：

```text
data_1x1/
cmip6_1x1/
cache/
outputs/
checkpoints*/
```

## 代码结构

```text
configs/          训练配置
docs/             架构与实验记录
scripts/          数据处理、检查与服务器辅助脚本
src/dataset.py    数据集与多 source 读取
src/model.py      WalkerNet 模型
src/trainer.py    rollout 训练、loss、checkpoint
src/evaluate*.py  评测脚本
```

## 当前状态

- 已支持 5-source 混合训练。
- 已支持 12 -> 15 -> 18 的 rollout curriculum。
- 已支持 field-first 的 Niño3.4 anomaly ACC/RMSE 评测。
- 当前推荐实验从历史 `best_skill.pt` 使用 `--init-checkpoint` 开始，而不是恢复旧 optimizer。
