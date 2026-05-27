<p align="center">
  <img src="./logo.png" alt="WalkerNet logo" width="180">
</p>

<h1 align="center">WalkerNet</h1>

<p align="center">
  全球海气物理场预测 · ENSO 技巧评估 · PyTorch 研究代码库
</p>

<p align="center">
  <img alt="Task" src="https://img.shields.io/badge/task-global%20field%20forecasting-006D77">
  <img alt="Target" src="https://img.shields.io/badge/eval-Ni%C3%B1o3.4%20%2F%20ENSO-14213D">
  <img alt="Grid" src="https://img.shields.io/badge/grid-1%C2%B0%20%7C%20180%C3%97360-2EC4B6">
  <img alt="Data" src="https://img.shields.io/badge/data-CESM2%20remapped-FFB703">
  <img alt="Model" src="https://img.shields.io/badge/model-interface%20ready-lightgrey">
</p>

---

## 项目目标

WalkerNet 用历史全球物理场预测下一月全球场，并从预测的 `tos` 中计算 Niño3.4 指数，用于 ENSO 相关评估。

当前任务是单步预测：

```text
输入 x: B x L x 4 x 180 x 360
输出 y: B x 1 x 4 x 180 x 360
```

更长 lead time 通过 autoregressive rollout 实现。

## 变量约定

所有张量的变量维度固定为：

| Channel | Variable | Meaning |
|---:|---|---|
| 0 | `tos` | Sea surface temperature |
| 1 | `zos` | Sea surface height above geoid |
| 2 | `tauu` | Zonal wind stress |
| 3 | `tauv` | Meridional wind stress |

对应代码约定见 `src/interfaces.py`：

```python
VARIABLES = ("tos", "zos", "tauu", "tauv")
```

## 网格约定

原始数据已经通过 CDO 重网格到 1° 全球规则网格：

```text
H = 180
W = 360
lat = -89.5 ... 89.5
lon =   0.5 ... 359.5
```

## 模型接口

模型侧只需要遵守这个最小接口：

```python
y_pred = model(x, target_month, rollout_step=None)
```

| Name | Shape | Type | Note |
|---|---|---|---|
| `x` | `(B, L, 4, 180, 360)` | `float32` | 归一化后的历史窗口 |
| `target_month` | `(B,)` | `int64` | 预测目标月份，取值 `1-12` |
| `rollout_step` | `(B,)` or `None` | `int64` | 自回归步数，单步训练可不传 |
| `y_pred` | `(B, 1, 4, 180, 360)` | `float32` | 下一月预测结果 |

## 架构方向

模型侧参考 `docs/architecture.md`，当前计划包括：

1. Joint time-variable patch embedding
2. Spatial attention，用于捕捉全球空间遥相关
3. TMoE，使用 target month 做月份条件路由
4. Coupled variable decoder，联合解码 4 个变量
5. Rollout step embedding，用于自回归多步预测时注入 lead-time 信息

## 数据准备

原始 CESM2 示例数据不直接进入训练，需要先用 CDO 重网格。

WSL/Linux：

```bash
bash scripts/remap_to_1x1.sh
```

Windows PowerShell：

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/d/Github/WalkerNet/scripts/remap_to_1x1.sh
```

输出目录：

```text
data_1x1/
  tos_1x1.nc
  zos_1x1.nc
  tauu_1x1.nc
  tauv_1x1.nc
```

检查数据：

```bash
python scripts/check_remapped_data.py --data-dir data_1x1
```

> `data_example/` 和 `data_1x1/` 都是大文件目录，不应提交到 GitHub。

## 代码结构

```text
configs/
  default.yaml
  grid_1x1_180x360.txt

scripts/
  remap_to_1x1.sh
  check_remapped_data.py

src/
  interfaces.py
  dataset.py
  metrics.py
  trainer.py
  train.py
  utils.py
  model.py
```

## 当前进度

- [x] 统一变量接口：`tos / zos / tauu / tauv`
- [x] 使用 CDO 将原始数据重网格到 `180 x 360`
- [x] 实现 `WalkerDataset`
- [x] 实现基础 metrics
- [x] 实现 masked MSE 与 Trainer 骨架
- [x] 实现训练入口 `src/train.py`
- [ ] 实现 `src/model.py`
- [ ] 跑通 synthetic tensor smoke test
- [ ] 跑通真实数据最小训练
- [ ] 加入完整 ENSO 评估实验

## 运行入口

模型实现完成后：

```bash
python -m src.train --config configs/default.yaml
```

当前 `src/model.py` 仍是占位模板，因此完整训练需要等模型侧实现完成后再运行。
