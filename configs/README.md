# 配置索引

配置文件仍保持在同一层，避免训练入口和历史命令失效；按文件名可分为以下几类：

| 命名 | 用途 |
|---|---|
| `default.yaml` | 本地开发默认值和完整字段参考 |
| `server_3090.yaml` | 单 source 服务器训练 |
| `server_3090_mixed5*.yaml` | 五 source 混合训练，含单卡、DDP4、DDP8 和 smoke |
| `server_3090_ssp126_ddp8.yaml` | SSP126 五模式混合训练，8 卡 DDP |
| `server_3090_rollout*.yaml` | 历史 rollout 与微调实验 |
| `server_smoke.yaml`、`server_ddp_smoke.yaml` | 单卡和 DDP 冒烟测试 |
| `grid_1x1_180x360.txt` | CDO 目标网格定义 |

约定：

1. 可复现实验配置提交到这里，机器上的临时 runtime 配置不提交。
2. 生产训练必须显式传入 `--config`，不要依赖脚本默认值。
3. checkpoint、cache、数据路径使用服务器绝对路径，代码入口保持项目内相对路径。
4. 新实验名称应描述数据、目标与并行规模，不再只使用日期区分。
