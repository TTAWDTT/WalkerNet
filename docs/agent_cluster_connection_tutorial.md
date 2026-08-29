# WalkerNet 集群连接教程（面向 Agent）

本文档说明一个自动化 agent 如何从 Windows 本机通过堡垒机进入 WalkerNet 的 GPU 容器，并在进入后安全地执行检查、读文件和运行任务。

> 适用范围：WalkerNet 远程节点、GateShell 菜单和容器 shell。  
> 更新时间：2026-08-29。  
> 安全要求：本文档不包含任何真实密码、私钥、token 或二次认证码。密码只能由用户交互输入、SSH agent 或外部秘密管理器提供。

## 1. 先理解连接链路

Agent 必须把连接过程看成多个状态，而不是“一条 SSH 命令直接到 GPU”：

```text
本机 PowerShell / WSL / Python
        │
        │  1. 带 TTY 的 SSH
        ▼
堡垒机 GateShell 登录界面
        │
        │  2. 交互输入 SSH/堡垒机密码
        │  3. 如出现 MFA，等待用户完成二次认证
        ▼
GateShell 节点菜单
        │
        │  4. 选择目标 node
        ▼
目标计算节点菜单
        │
        │  5. 选择 container
        ▼
GPU 容器 shell
        │
        │  6. stty -echo + hostname + nvidia-smi
        ▼
WalkerNet 工作目录和实验环境
```

在 GateShell 菜单状态下直接发送 `hostname`、`ls` 或 `nvidia-smi` 通常不会执行 Linux 命令；agent 必须先完成 node/container 选择。

## 2. 凭据和 `CLUSTER PWD` 的含义

项目中没有统一的 `CLUSTER_PWD` 文件或固定环境变量。界面出现 `CLUSTER PWD` 时，通常表示当前 GateShell/集群平台要求的目标集群密码字段；它不一定等同于：

- GitHub/GitLab 密码或 token；
- Linux `sudo` 密码；
- 其他节点的密码；
- 已经通过 SSH 登录堡垒机时使用的密码。

Agent 的处理规则：

1. 不从仓库、命令历史或旧日志猜测密码；
2. 不把密码拼接进命令行、Python 源码、Markdown、日志或提交记录；
3. 如果提示需要 `CLUSTER PWD`，暂停在认证状态，向用户请求通过安全渠道提供或手动输入；
4. 如果只是 SSH 密码提示，使用用户在终端中的交互输入；
5. 二次认证由用户完成时，agent 等待认证完成，不重复创建多个 SSH 会话。

## 3. 入口选择

### 3.1 `osm` 入口：node 2 → container 1（已验证流程）

这是进入历史 `k8s-node3-gpu` 容器的明确流程。

在 Windows Terminal、PowerShell 或 WSL 中启动一个带伪终端的 SSH：

```bash
ssh -tt -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password \
    -p 18000 zhen.luo@osm.insightst.com
```

看到密码提示后，用户在交互终端输入密码。登录成功后，GateShell 仍处于菜单状态，按顺序发送：

```text
2<Enter>       # 选择 node 2（历史名称：k8s-node3-gpu）
1<Enter>       # 选择该 node 下的 container 1
stty -echo<Enter>
```

进入容器后验证：

```bash
hostname
pwd
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

只有看到目标容器的 hostname、Linux 路径和 `nvidia-smi` 输出，才算连接成功。

### 3.2 `yundun` 入口：GPU005/006/007 等节点

启动带 TTY 的 SSH：

```bash
ssh -tt -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password \
    -p 60022 zhen.luo@yundun.insightst.com
```

然后按 GateShell 当前菜单选择：

```text
目标 node（例如 k8s-sh-azb-gpu-005/006/007）
        ↓
目标 node 下的 container
        ↓
stty -echo
```

不要把 `osm` 的 `2 → 1` 直接套到 `yundun`。yundun 菜单顺序可能改变，agent 应先读取完整菜单，按节点名称匹配；历史脚本使用过 `j`/`k` 移动光标，但移动次数不是稳定 API。

## 4. Agent 的连接状态机

### 状态 A：建立单个交互会话

每次任务只建立一个主 SSH/Tty 会话，并保存 session/channel 标识。不要并发创建多个登录尝试，以免触发连接限制或导致 agent 读错菜单输出。

必须满足：

- 使用 `ssh -tt` 或 Paramiko `invoke_shell()`；
- 关闭 SSH key/agent 的自动尝试（如果平台要求密码）；
- 设置连接和读取超时；
- 保留原始输出用于诊断，但清理密码、token 和 MFA 内容。

### 状态 B：等待认证

agent 需要区分以下提示：

```text
password:
Password:
CLUSTER PWD:
MFA / verification code:
GateShell menu:
```

处理原则：

- `password:` / `Password:`：等待用户交互输入，或调用外部秘密注入接口；
- `CLUSTER PWD:`：不能自行猜测，确认凭据范围后再输入；
- `MFA`：等待用户完成二次认证；
- GateShell 菜单：不要把菜单文本当作 Linux shell。

### 状态 C：选择 node 和 container

`osm` 已验证的最小序列：

```text
send("2\\r") → 等待 node/container 菜单变化
send("1\\r") → 等待容器 shell
send("stty -echo\\n")
```

`yundun` 的序列应由当前菜单动态决定：

```text
读取菜单 → 匹配目标 node 名称 → 发送方向键/确认键
读取目标 node 的 container 菜单 → 匹配容器 → 发送确认键
读取容器提示符 → 发送 stty -echo
```

不要只依据“SSH channel 已打开”判断已经进入 GPU 节点。

### 状态 D：进入后的安全验证

进入容器后依次执行：

```bash
printf '__BEGIN__\n'
hostname
id -un
pwd
test -d /data/WalkerNet && echo WalkerNet_data=OK || echo WalkerNet_data=MISSING
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
printf '__END__\n'
```

如果没有收到 `__END__`，说明命令可能仍在 GateShell 菜单、输出被截断或 shell 没有正常返回；agent 应读取剩余输出，而不是立即再开新连接。

## 5. Paramiko 参考实现

下面的示例只展示连接和状态控制。密码通过 `getpass` 交互输入，不写入源码；实际生产 agent 应优先使用外部秘密管理器，并对接平台的 MFA 流程。

```python
from __future__ import annotations

import getpass
import re
import time

import paramiko


def read_until(channel, patterns, timeout=60.0, chunk_size=100_000):
    """Read an interactive channel until one pattern appears."""
    patterns = [re.compile(p, re.I) for p in patterns]
    data = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if channel.recv_ready():
            data.extend(channel.recv(chunk_size))
            text = data.decode("utf-8", "ignore")
            if any(p.search(text) for p in patterns):
                return text
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {patterns}; bytes={len(data)}")


def connect_osm_node2_container1():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass("SSH/bastion password (input is not logged): ")
    client.connect(
        "osm.insightst.com",
        port=18000,
        username="zhen.luo",
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        auth_timeout=60,
        banner_timeout=60,
    )
    channel = client.invoke_shell(width=240, height=40)

    # Wait for the GateShell menu or an MFA/password-related prompt.
    menu = read_until(
        channel,
        [r"GateShell", r"\d+:", r"node", r"password", r"MFA", r"CLUSTER PWD"],
        timeout=90,
    )
    if re.search(r"MFA|CLUSTER PWD", menu, re.I):
        raise RuntimeError("Interactive MFA/CLUSTER PWD input is required before menu selection")

    # This sequence is specific to the verified osm node-2 route.
    channel.send("2\r")
    read_until(channel, [r"container", r"1:", r"shell", r"password"], timeout=60)
    channel.send("1\r")
    read_until(channel, [r"[$#>]\s*$", r"/data", r"container"], timeout=90)
    channel.send("stty -echo\n")
    read_until(channel, [r"[$#>]\s*$"], timeout=30)
    return client, channel


def run_readonly_probe(channel):
    marker = "__WALKERNET_PROBE_END__"
    channel.send(
        "hostname; pwd; "
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv; "
        f"printf '{marker}\\n'\n"
    )
    return read_until(channel, [re.escape(marker)], timeout=60)
```

注意：某些 GateShell 实现不会把密码提示作为普通 channel 输出，或者会在用户完成 MFA 前暂停输出；此时不能让脚本自动猜密码，应将会话保持在认证状态并等待用户。

## 6. yundun 节点选择的 agent 策略

由于 yundun 节点菜单顺序可能变化，推荐采用“菜单文本匹配”而不是固定按键次数：

1. 读取菜单直到出现目标节点名或节点列表结束；
2. 记录每个节点的显示名称和当前菜单索引；
3. 使用方向键移动到完全匹配的目标节点；
4. 发送 Enter，等待 container 菜单；
5. 同样按容器名称匹配并进入；
6. 进入容器后才运行 `hostname` 和 `nvidia-smi`。

历史目标名称：

```text
k8s-sh-azb-gpu-005
k8s-sh-azb-gpu-006
k8s-sh-azb-gpu-007
```

节点当前是否空闲不能从这份教程推断。agent 必须在目标容器中检查：

```bash
nvidia-smi
pgrep -af 'python|torchrun|mpirun|cnop|evaluate_rollout' || true
```

## 7. 连接成功后的 WalkerNet 路径

远端固定根目录：

```text
/data/WalkerNet/
```

常用位置：

```text
/data/WalkerNet/repo/
/data/WalkerNet/venv313/
/data/WalkerNet/anaconda3/
/data/WalkerNet/input/artifacts/historical_mixed5_best_skill.pt
/data/WalkerNet/input/artifacts/mixed5_norm_stats_train.pt
/data/WalkerNet/outputs/
```

进入目标容器后，推荐先执行：

```bash
cd /data/WalkerNet/repo
git rev-parse --short HEAD
git status --short
/data/WalkerNet/venv313/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

除非任务明确要求切换分支或修改代码，否则 agent 不应在远端仓库执行 `git reset --hard`、覆盖脚本或删除输出。

## 8. 只读文件检查与安全传输

连接成功后，优先用只读命令确认文件：

```bash
find /data/WalkerNet/outputs/<experiment> -maxdepth 2 -type f | sort | head -100
stat /data/WalkerNet/input/artifacts/historical_mixed5_best_skill.pt
sha256sum /data/WalkerNet/input/artifacts/historical_mixed5_best_skill.pt
```

如果堡垒机环境不允许 `scp`/`sftp`，可使用已有的 Paramiko channel 做 base64 分块传输，但应满足：

- 远端只读时只执行 `base64`/`sha256sum`/`printf`；
- 大文件分块并在本地合并，不一次性读入无限长度缓冲区；
- 传输后比较远端和本地 SHA-256；
- 输出目录和原文件保留，不覆盖正式结果；
- 不在传输命令中出现密码或 token。

## 9. GPU 作业启动前的检查

只有在连接成功、代码/环境/输入文件均通过检查后，才允许考虑启动作业。启动前记录：

```bash
hostname
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
pgrep -af 'python|torchrun|mpirun|cnop|evaluate_rollout' || true
git rev-parse HEAD
git status --short
```

作业必须：

- 只使用真正空闲的 GPU；
- 使用新的、明确命名的 output root；
- 单独保存 launcher PID、每个作业 PID 和日志；
- 保留 checkpoint、manifest、summary、NPZ 和 audit 输出；
- 不中断其他用户作业；
- 不把旧 pilot 结果混入正式证据。

## 10. 常见失败与恢复方法

### `ssh-agent` 不可用 / 没有 `~/.ssh/config`

这只表示本机没有自动密钥代理或 SSH 配置，不代表堡垒机不可达。改用带 `-tt` 的交互 SSH，并在密码提示处输入凭据。

### `Unable to open channel`

按以下顺序检查：

1. 是否使用了 `invoke_shell()` 或 `ssh -tt`；
2. 是否仍有旧会话占用堡垒机 channel；
3. 是否正在等待 MFA/`CLUSTER PWD`；
4. GateShell 菜单是否已经完全出现；
5. 是否把 node/container 菜单误当成 Linux shell。

不要因为这个错误去杀掉远端 Python、MPI 或 CNOP 进程。

### 登录后一直显示菜单，命令没有执行

说明尚未完成 node/container 选择。重新读取菜单状态；`osm` 路径使用 `2 → 1`，yundun 路径按节点名和容器名动态选择。

### `hostname` 成功但 `nvidia-smi` 失败

可能进入的是 CPU 容器、GPU 没有映射或容器初始化未完成。记录 `hostname`、容器名和完整错误，先不要启动任务。

### 看到 GPU 但显存为 0

不能仅凭显存判断空闲。必须同时检查 `nvidia-smi` 的利用率、计算进程列表和目标用户作业 PID；若仍不确定，视为不可用并报告阻塞。

## 11. 退出流程

任务结束时：

```bash
stty echo
exit              # 退出容器
exit              # 退出 GateShell/节点 shell（按实际层数执行）
```

agent 应确认 SSH channel 已关闭，并清理本地内存中的密码变量。不要把交互输出原样写入持久化日志；至少过滤 `password`、`CLUSTER PWD`、MFA code 和 token。

## 12. 相关项目文档

- [服务器、GPU 与数据位置总览](server_and_data_locations.md)
- [WalkerNet 数据布局](data_layout.md)
- [可复现性检查清单](reproducibility.md)
- [rollout 评测记录](walkernet_rollout_accuracy_record.md)
- [Pacific delayed-onset CNOP 计划](cnop_pacific_delayed_onset_plan.md)
- [Global delayed-onset CNOP 计划](cnop_global_delayed_onset_plan.md)

