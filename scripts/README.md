# 脚本索引

脚本按职责分组。所有命令默认从项目根目录执行。

| 目录 | 用途 | 常用入口 |
|---|---|---|
| `data/` | CDO 重网格、输出校验 | `remap_to_1x1.sh`、`check_remapped_data.py` |
| `train/` | 单卡/DDP 训练、显存冒烟、等待 GPU | `train_ddp.sh`、`smoke_test_model.py` |
| `eval/` | 常规预报结果可视化 | `plot_rollout_spatial_diagnostics.py` |
| `cnop/` | CNOP 约束、优化、聚类和绘图 | `compute_tos_zos_cnop.py` |
| `cnop/runs/` | 固定参数的历史实验流水线 | `run_cnop_*.sh` |

常用命令：

```bash
# 8 卡训练
bash scripts/train/train_ddp.sh

# 数据重网格与校验
bash scripts/data/remap_to_1x1.sh
python scripts/data/check_remapped_data.py --data-dir data_1x1

# CNOP 优化
python scripts/cnop/compute_tos_zos_cnop.py --help
```

`cnop/runs/` 中的脚本记录具体服务器路径和实验参数，用于复现实验；新的通用能力应放在上一级 Python 工具中。
