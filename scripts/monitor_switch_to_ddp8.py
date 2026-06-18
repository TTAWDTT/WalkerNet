"""监控 GPU 状态，并在 8 卡可用时把 mixed5 训练从 4 卡切到 8 卡。

这个脚本设计成跑在服务器 tmux 里：

    python scripts/monitor_switch_to_ddp8.py

安全约定：
    1. 只观察 GPU 0-3 是否空出来，不会杀其它用户进程。
    2. 只停止本项目指定的 4 卡 tmux 会话。
    3. 必须等 4 卡训练写出 latest.pt 后，才会启动 8 卡 resume。
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch WalkerNet mixed5 DDP4 training to DDP8 when GPUs are free.")
    parser.add_argument("--project-dir", type=Path, default=Path("/mnt/sda/WalkerNet"))
    parser.add_argument("--python-bin", default="/home/cpji/wwb/torch/bin/python")
    parser.add_argument("--ddp4-session", default="walker_mixed5_ddp4_0618")
    parser.add_argument("--ddp8-session", default="walker_mixed5_ddp8_0618")
    parser.add_argument("--ddp8-config", default="configs/server_3090_mixed5_ddp8.yaml")
    parser.add_argument("--resume", default="/mnt/sda/WalkerNet/checkpoints_mixed5_ddp4/latest.pt")
    parser.add_argument("--ddp8-log", default="outputs/logs/mixed5_ddp8_train_0618.log")
    parser.add_argument("--master-port", type=int, default=29519)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--free-gpus", default="0,1,2,3", help="这些卡空出来后才允许切 8 卡。")
    parser.add_argument("--all-gpus", default="0,1,2,3,4,5,6,7", help="8 卡训练使用的 CUDA_VISIBLE_DEVICES。")
    parser.add_argument("--free-mem-threshold-mib", type=int, default=500)
    parser.add_argument("--stop-timeout-seconds", type=int, default=90)
    return parser.parse_args()


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def run(command: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=check)


def tmux_has_session(name: str) -> bool:
    return run(["tmux", "has-session", "-t", name]).returncode == 0


def gpu_memory_mib() -> dict[int, int]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
    )
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        idx_text, mem_text = [part.strip() for part in line.split(",", maxsplit=1)]
        usage[int(idx_text)] = int(mem_text)
    return usage


def selected_gpus_are_free(gpus: list[int], threshold_mib: int) -> tuple[bool, dict[int, int]]:
    usage = gpu_memory_mib()
    busy = {idx: usage.get(idx, 0) for idx in gpus if usage.get(idx, 0) > threshold_mib}
    return not busy, busy


def checkpoint_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stop_ddp4_session(session: str, timeout_seconds: int) -> None:
    if not tmux_has_session(session):
        log(f"4-card session {session!r} is already absent")
        return

    log(f"send Ctrl-C to 4-card session {session!r}")
    run(["tmux", "send-keys", "-t", session, "C-c"])
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not tmux_has_session(session):
            log("4-card session exited")
            return
        time.sleep(5)

    log(f"4-card session did not exit in {timeout_seconds}s; kill our tmux session")
    run(["tmux", "kill-session", "-t", session])


def wait_all_gpus_free(gpus: list[int], threshold_mib: int) -> None:
    while True:
        free, busy = selected_gpus_are_free(gpus, threshold_mib)
        if free:
            return
        log(f"waiting GPUs to release after stopping 4-card session: {busy}")
        time.sleep(10)


def start_ddp8(args: argparse.Namespace) -> None:
    if tmux_has_session(args.ddp8_session):
        log(f"8-card session {args.ddp8_session!r} already exists; nothing to start")
        return

    command = (
        f"CUDA_VISIBLE_DEVICES={args.all_gpus} {args.python_bin} -u "
        f"-m torch.distributed.run --nproc_per_node=8 --master_port={args.master_port} "
        f"-m src.train --config {args.ddp8_config} --num-workers 2 --resume {args.resume} "
        f"> {args.ddp8_log} 2>&1"
    )
    log(f"start 8-card session {args.ddp8_session!r}")
    run(["tmux", "new-session", "-d", "-s", args.ddp8_session, command], cwd=args.project_dir, check=True)


def main() -> None:
    args = parse_args()
    free_gpus = [int(item) for item in args.free_gpus.split(",") if item.strip()]
    all_gpus = [int(item) for item in args.all_gpus.split(",") if item.strip()]
    resume_path = Path(args.resume)

    log("monitor started")
    log(f"watch free GPUs={free_gpus}, resume={resume_path}")

    while True:
        if tmux_has_session(args.ddp8_session):
            log(f"8-card session {args.ddp8_session!r} already running; monitor exits")
            return

        ckpt_ok = checkpoint_ready(resume_path)
        free, busy = selected_gpus_are_free(free_gpus, args.free_mem_threshold_mib)
        if ckpt_ok and free:
            log("checkpoint is ready and GPUs 0-3 are free; switching to 8-card training")
            stop_ddp4_session(args.ddp4_session, args.stop_timeout_seconds)
            wait_all_gpus_free(all_gpus, args.free_mem_threshold_mib)
            start_ddp8(args)
            log("switch submitted; monitor exits")
            return

        reason_parts = []
        if not ckpt_ok:
            reason_parts.append("waiting latest.pt")
        if not free:
            reason_parts.append(f"waiting GPUs: {busy}")
        log("; ".join(reason_parts))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
