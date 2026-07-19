# walker-3090 目录结构

服务器项目根目录为 `/mnt/sda/WalkerNet`。代码与大体量实验资产使用不同的管理规则。

```text
/mnt/sda/WalkerNet/
  src/                    核心 Python 包
  configs/                已复现实验配置
  scripts/                按 data/train/eval/cnop 分类的工具
  docs/                   项目文档
  tests/                  自动化测试

  raw/                    historical 原始资料
  cmip6-ssp/              SSP 原始资料
  data_1x1/               单 source 1 度资料
  cmip6_1x1/              historical 多模式 1 度资料
  cmip6_ssp_1x1/          SSP 多模式 1 度资料
  cache/                  Dataset mmap 与归一化缓存

  checkpoints_*/          各实验 checkpoint
  outputs/                评测、CNOP 结果和日志
  runtime/                服务器临时配置与运行清单
  backups/code/           同步代码前的小体量备份
```

管理约定：

1. 数据、cache、checkpoint 和已发布实验输出不纳入 Git。
2. 正在训练时不移动配置中引用的数据、cache、checkpoint 和日志路径。
3. `configs/` 只保留可复现实验；临时 `_runtime.yaml` 放入 `runtime/configs/`。
4. 训练日志统一写入 `outputs/logs/`，旧的根级日志放入 `outputs/logs/archive/`。
5. 同步服务器代码前，将被覆盖的小文件放入 `backups/code/<date>_<reason>/`。
6. `scripts/cnop/runs/` 保存固定参数实验流水线；通用逻辑应放在上一级 Python 工具中。
