# WalkerNet 服务器连接、资源概况与数据位置

本文档记录 WalkerNet 项目的远程登录路径、节点用途、数据/模型/缓存位置，以及本地结果和绘图脚本位置。

如果需要按 agent 状态机复现登录，请先阅读：[WalkerNet 集群连接教程（面向 Agent）](agent_cluster_connection_tutorial.md)。

> 更新时间：2026-08-28  
> 安全说明：本文档不保存堡垒机密码、SSH 密钥、GitHub token 或其他凭据。认证信息应通过交互式输入、SSH agent 或个人安全凭据管理器提供。

## 1. 连接总览

常用链路为：

```text
Windows 本机
  ↓ SSH
yundun.insightst.com:60022
  ↓ GateShell 交互菜单
k8s-sh-azb-gpu-005 / gpu-006 / gpu-007 等计算节点
  ↓ 容器选择
WalkerNet 容器 shell
```

项目历史上还使用过另一条 GateShell 入口：

```text
Windows 本机
  ↓ SSH
osm.insightst.com:18000
  ↓ GateShell 菜单：选择 node 2，再选择 container 1
k8s-node3-gpu 容器
```

两条入口都需要以当时 GateShell 菜单实际显示的节点和容器为准；不要根据旧记录盲选容器。

## 2. 推荐登录方式

堡垒机登录必须使用**交互式、带 TTY 的 SSH**。本机没有可用的 `ssh-agent`，也没有统一的 `~/.ssh/config`；因此不要把“检查 SSH key”或非交互式 `ssh host command` 当作登录成功的判据。私钥和密码都不写入本项目文档。

### 2.1 osm 入口：已验证的 node 2 → container 1 流程

这是目前最明确、最容易复现的 GateShell 流程。在 Windows PowerShell、Windows Terminal 或 WSL 中执行：

```bash
ssh -tt zhen.luo@osm.insightst.com -p 18000
```

看到密码提示后，在终端中交互输入堡垒机密码。登录成功后不要直接输入 Linux 命令；此时仍处于 GateShell 菜单，需要按下面顺序发送：

```text
2<Enter>       # 选择 node 2（历史名称：k8s-node3-gpu）
1<Enter>       # 选择该 node 下的 container 1
stty -echo<Enter>
```

也就是完整链路：

```text
Windows 本机
  → ssh zhen.luo@osm.insightst.com -p 18000
  → 交互输入密码
  → GateShell 发送 2 + Enter
  → GateShell 发送 1 + Enter
  → 容器 shell
  → stty -echo
```

进入容器后先确认位置和 GPU：

```bash
hostname
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

离开前恢复回显：

```bash
stty echo
exit
```

`2` 和 `1` 是这条 osm node 2 路径的已验证菜单选择，不应直接套用到其他入口；如果 GateShell 菜单发生变化，必须先读取当前菜单。

### 2.2 yundun 入口：GPU005/006/007 路径

连接 yundun 时同样必须分两阶段：先 SSH 到 GateShell，再在菜单中选计算节点和容器。

```bash
ssh -tt zhen.luo@yundun.insightst.com -p 60022
```

看到密码提示后交互输入密码。进入 GateShell 后：

1. 等待菜单完整显示；
2. 选择目标节点，例如 `k8s-sh-azb-gpu-005`、`k8s-sh-azb-gpu-006` 或 `k8s-sh-azb-gpu-007`；
3. 进入该节点后选择目标容器；
4. 进入容器 shell 后执行 `stty -echo`；
5. 任务结束执行 `stty echo`，再依次退出容器和 GateShell。

历史脚本曾用 `j` 键移动菜单光标，也曾使用 GateShell 搜索节点名；这些是交互手段，不是固定的节点编号。不要假定固定移动次数，也不要把 yundun 的菜单编号与 osm 的 `2 → 1` 流程混用。

### 2.3 Paramiko/自动化脚本的要求

需要脚本化时，必须使用 Paramiko 的 `invoke_shell()` 或等价的伪终端连接，而不是普通的非交互 `exec_command()`：

```python
client.connect(
    "osm.insightst.com", port=18000,
    username="zhen.luo",
    look_for_keys=False,
    allow_agent=False,
)
channel = client.invoke_shell(width=240, height=40)
```

脚本应等待 GateShell 提示符后，发送 `2\r`、等待节点菜单，再发送 `1\r`；只有收到容器 shell 提示后才发送 Linux 命令。密码应由交互输入、环境外部的安全凭据注入或 SSH agent 提供，不能硬编码进脚本、日志或 Markdown。

### 2.4 常见失败原因

截图中“SSH agent 不可用”“`~/.ssh/config` 不存在”并不表示堡垒机不可用，而是说明：

- 本机没有运行 SSH agent；
- 当前用户目录没有 SSH config；
- 不能依赖公钥自动登录；
- 需要带 `-tt` 建立交互终端，并在密码提示处手动完成认证；
- 认证成功后还必须完成 GateShell 的节点和容器选择，SSH 连接本身不等于已经进入 GPU 容器。

如果出现 `Unable to open channel`，优先检查是否用了非交互方式、是否已有旧的堡垒机 shell 占用会话、以及是否尚未完成 GateShell/二次认证；不要因此修改远端作业或反复尝试杀掉其他会话。

## 3. 远程节点与 GPU 概况

下表是项目记录中的“最近已知用途”，不是实时占用表。开始任何作业前都必须在目标容器中非破坏性检查 `nvidia-smi`、进程和目标输出目录。

| 节点 | 设备/历史用途 | 资源使用注意 |
|---|---|---|
| `k8s-sh-azb-gpu-005` | 8 张 L20X；正式三海域 CNOP（30 jobs）、GPU005 迁移和部分正式生产 | 这是正式 CNOP 生产的主要节点；不得假设现在空闲 |
| `k8s-sh-azb-gpu-006` | 早期 Global pilot、case-screen、约束校准和依赖准备 | 历史记录明确有其他实验；除非现场确认空闲，不应占用 |
| `k8s-sh-azb-gpu-007` | 8 张 L20X；Pacific delayed-onset、Global delayed-onset 及 rollout 评测 | 近期评测和 delayed 实验主要使用该节点；必须先检查进程 |
| `k8s-node3-gpu`（GateShell node 2） | 通过 `osm.insightst.com:18000` 进入的容器节点 | 容器和 GPU 映射随菜单变化，需现场查询 |

建议的只读检查：

```bash
hostname
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
pgrep -af 'python|torchrun|mpirun|cnop|evaluate_rollout' || true
```

资源分配原则：

- 只使用真正空闲的 GPU；“显存较低”不等于没有作业，必须同时检查进程；
- 不停止、杀掉或修改其他用户的进程；
- 多作业启动时保留 PID、日志和独立输出目录；
- 远端实验正在运行时，不要从同一目录覆盖脚本、cache 或结果文件；
- GPU 的实时可用性不写死在本文档中，以现场 `nvidia-smi` 为准。

## 4. 远程 WalkerNet 目录

远程项目根目录：

```text
/data/WalkerNet/
```

主要子目录：

| 内容 | 远程路径 | 说明 |
|---|---|---|
| 代码仓库 | `/data/WalkerNet/repo/` | 远程运行代码；运行前记录 `git rev-parse HEAD` 和 `git status --short` |
| Python 环境 | `/data/WalkerNet/venv313/` | WalkerNet 运行环境；历史迁移曾修复过 Python 解释器的绝对路径问题 |
| 基础 Python | `/data/WalkerNet/anaconda3/` | GPU005 迁移后使用的可移植基础运行时 |
| 模型输入/权重 | `/data/WalkerNet/input/` | checkpoint、normalization stats 等只读输入 |
| 实验输出 | `/data/WalkerNet/outputs/` | 所有 CNOP、评测和绘图输出的远端根目录 |

远端输入文件的关键路径：

```text
/data/WalkerNet/input/artifacts/historical_mixed5_best_skill.pt
/data/WalkerNet/input/artifacts/mixed5_norm_stats_train.pt
/data/WalkerNet/repo/configs/server_gpu006_historical_mixed5.yaml
```

不同实验应使用其自己的 output root；不要把旧 pilot 的结果复制到正式证据目录。

## 5. 远程实验结果目录

### 5.1 正式三海域 CNOP

```text
/data/WalkerNet/outputs/cnop_basin_relative3pct_lead12_steps100_gpu005_v1/
```

预期结构：

```text
manifest/
logs/
pacific/
indian/
global/
summary/
figures/
```

该实验使用 3% `relative_initial_l2`、100 Adam steps、每个案例 8 个起点，并覆盖 Pacific、Indian 和 Global 三个 domain。其正式审计和图件应以该目录下的 manifest、summary 和 figures 为准。

### 5.2 Pacific delayed-onset CNOP

```text
/data/WalkerNet/outputs/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/
```

常见子目录：

```text
metadata/
normal/
delayed/
figures/
```

这是 Pacific delayed-onset 的主要结果目录，包含 24 starts 配置下保留的 top-3 候选和后续图件。v1 旧结果保留用于 provenance，不应与 v2 正式结果混用。

### 5.3 Global delayed-onset CNOP

```text
/data/WalkerNet/outputs/cnop_global_delayed_onset_24starts_steps100_v1/
```

只有在 summary、NPZ、candidate 和数值一致性审计通过后，才将图件复制到本地 delayed 目录。

### 5.4 约束尺度校准

```text
/data/WalkerNet/outputs/cnop_global_constraint_calibration_v1/
```

该目录对应冻结案例在多个 constraint scale 下的校准实验；历史协议为 scale 0.20、0.30、0.40，使用 accepted Adam、1000 steps、8 starts。旧的 Global 0.20 pilot 仅作历史参考，不能作为正式证据。

### 5.5 WalkerNet rollout 评测

lead-1--18 正式评测：

```text
/data/WalkerNet/outputs/eval_rollout_best_skill_test_20260825/
```

lead-1--36 扩展评测及起始月分组结果：

```text
/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/
```

后者包含 monthly lead-1--36、按 start month 分组的 lead-1--36，以及 12×12 start/end-month 矩阵和 persistence 对照。

## 6. 本地仓库与数据位置

本地项目根目录：

```text
D:\Github\WalkerNet
```

### 6.1 本地历史数据

```text
D:\Github\WalkerNet\.local\data_1x1\historical\
```

当前本地缓存的 source 包括：

```text
CESM2/
EC-Earth3/
GFDL-ESM4/
IPSL-CM6A-LR/
MPI-ESM1-2-HR/
```

每个 source 目录应包含：

```text
tos_1x1.nc
zos_1x1.nc
tauu_1x1.nc
tauv_1x1.nc
```

目标网格为 180×360，坐标中心约为：

```text
lat = -89.5, -88.5, ..., 89.5
lon =   0.5,   1.5, ..., 359.5
```

缺测/陆地由 `valid_mask` 管理；不要把缺测当作物理零值写回原始 NetCDF。

### 6.2 本地模型、统计量和绘图 cache

用于本地复绘的可移植 bundle：

```text
D:\Github\WalkerNet\tmp\cnop_basin_local_replot\portable\historical_mixed5_best_skill.pt
D:\Github\WalkerNet\tmp\cnop_basin_local_replot\portable\mixed5_norm_stats_train.pt
D:\Github\WalkerNet\tmp\cnop_basin_local_replot\portable\forecast_tos_climatology_train_h12.npz
D:\Github\WalkerNet\tmp\cnop_basin_local_replot\portable\observed_tos_climatology_train.npz
```

source-specific 数据 cache 位于：

```text
D:\Github\WalkerNet\tmp\local_mixed5_data_cache_CESM2\
D:\Github\WalkerNet\tmp\local_mixed5_data_cache_EC-Earth3\
D:\Github\WalkerNet\tmp\local_mixed5_data_cache_GFDL-ESM4\
D:\Github\WalkerNet\tmp\local_mixed5_data_cache_IPSL-CM6A-LR\
D:\Github\WalkerNet\tmp\local_mixed5_data_cache_MPI-ESM1-2-HR\
```

Pacific delayed CNOP 的本地 NPZ 暂存目录：

```text
D:\Github\WalkerNet\tmp\pacific_delayed_remote\
```

其中每个案例目录通常包含 `case_<source>_<year>.npz`，字段包括 `delta_phys`、`delta_norm`、`top_delta_phys`、`top_delta_norm` 以及候选的 Niño3.4/constraint 元数据。

## 7. 本地图件和结果目录

### 7.1 三海域正式 CNOP 图件

```text
D:\Github\WalkerNet\docs\assets\cnop_basin_relative3pct_lead12_steps100_v1\
```

主要图件包括：

```text
figures\overview\cnop_overview_10cases_pacific.png
figures\overview\cnop_overview_10cases_indian.png
figures\overview\cnop_overview_10cases_global.png
figures\response_evolution\response_evolution_10cases_pacific.png
figures\response_evolution\response_evolution_10cases_indian.png
figures\response_evolution\response_evolution_10cases_global.png
figures\basin_response_summary.png
summary\constraint_audit.csv
summary\cnop_case_summary.csv
```

### 7.2 Pacific delayed-onset 图件

```text
D:\Github\WalkerNet\docs\assets\cnop_pacific_delayed_onset_24starts_steps100_v2_top3\
```

legacy response-evolution 整理目录：

```text
legacy_response_evolution\overview\
legacy_response_evolution\Pacific\
legacy_response_evolution\pacific delay\
legacy_response_evolution\indian\
legacy_response_evolution\global\
legacy_response_evolution\global delay\
```

虽然 Indian delayed 结果曾经存在于历史 bundle 中，但当前论文范围聚焦 Pacific；使用时应查看对应 provenance 和实验记录，避免把 Indian 结果当作当前主结论。

### 7.3 rollout skill 图件

```text
D:\Github\WalkerNet\docs\assets\walkernet_rollout_skill\
```

重要文件：

```text
walkernet_acc_vs_lead.png
walkernet_acc_by_start_month_lead1_24.png
walkernet_acc_by_start_month_lead1_36.png
walkernet_acc_by_start_month_model_persistence_lead1_24.png
walkernet_start_end_month_acc_model_persistence.png
matrix_data\eval_rollout_best_skill_start_end_month_acc_12x12_model.csv
matrix_data\eval_rollout_best_skill_start_end_month_acc_12x12_persistence.csv
eval_rollout_best_skill_lead1_36_by_start_month.csv
```

## 8. 关键脚本位置

### CNOP 与 response-evolution

```text
D:\Github\WalkerNet\scripts\cnop\plot_cnop_ten_case_lead12.py
D:\Github\WalkerNet\scripts\cnop\plot_cnop_monthly_response.py
D:\Github\WalkerNet\scripts\cnop\compute_tos_zos_cnop.py
```

`plot_cnop_ten_case_lead12.py` 是十案例 lead-12 overview 的 canonical panel 脚本；应在该脚本上调整数据读取、共享色标和显示层参数，不要用 PIL 把小图覆盖回旧 PNG。

### rollout skill 图

```text
D:\Github\WalkerNet\scripts\plotting\plot_rollout_acc_vs_lead.py
D:\Github\WalkerNet\scripts\plotting\plot_acc_by_start_month.py
D:\Github\WalkerNet\scripts\plotting\plot_start_end_month_acc.py
```

起始月×lead 图的正式输入是：

```text
D:\Github\WalkerNet\docs\assets\walkernet_rollout_skill\eval_rollout_best_skill_lead1_36_by_start_month.csv
```

该 CSV 同时包含 `system=model` 和 `system=persistence`，因此可以在同一脚本中生成左右对照 panel。

## 9. 运行和归档规范

每次远端实验或评测至少归档以下信息：

```text
git commit / branch
完整 YAML 配置
checkpoint 路径和 SHA-256
数据 source、年份和 remapping 信息
split、lead 范围和 anomaly/climatology 定义
seed、optimizer、starts、constraint 和 objective
launcher PID、日志路径和 GPU 分配
summary CSV、NPZ、audit JSON/CSV
绘图脚本、输出路径和 provenance
```

图件交付前应检查：

- 原始数据和显示层变换没有被覆盖；
- 共享色标、坐标范围和单位在比较 panel 中一致；
- NaN/陆地、物理零值和低于显示阈值的数据没有被混淆；
- PNG/PDF 文件确实来自一次完整脚本运行，而不是后处理拼接；
- 输出目录包含对应的 provenance、alt text 或 README；
- 远端结果拉回本地后保留原目录结构和文件名，并记录 checksum。
