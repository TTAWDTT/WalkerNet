"""等待 GPU 空闲后自动启动 mixed5 训练。

这个脚本放在服务器 tmux 中长期运行：

    python scripts/train/wait_start_mixed5_training.py

安全约定：
- 只观察 GPU 显存占用，不终止任何其它用户进程。
- 优先使用 8 卡；如果不足 8 卡但至少 4 卡空闲，则启动 4 卡训练。
- 如果目标训练 tmux 已存在，则不重复启动。
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for free GPUs and start WalkerNet mixed5 training.")
    parser.add_argument("--project-dir", type=Path, default=Path("/mnt/sda/WalkerNet"))
    parser.add_argument("--python-bin", default="/home/cpji/wwb/torch/bin/python")
    parser.add_argument("--session-prefix", default="walker_mixed5_skill")
    parser.add_argument("--ddp4-config", default="configs/server_3090_mixed5_ddp4.yaml")
    parser.add_argument("--ddp8-config", default="configs/server_3090_mixed5_ddp8.yaml")
    parser.add_argument("--resume", default="/mnt/sda/WalkerNet/checkpoints_mixed5_ddp8/latest.pt")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--master-port", type=int, default=29529)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--free-mem-threshold-mib", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
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


def free_gpu_indices(threshold_mib: int) -> tuple[list[int], dict[int, int]]:
    usage = gpu_memory_mib()
    free = [idx for idx, mem in sorted(usage.items()) if mem <= threshold_mib]
    return free, usage


def choose_world_size(free_gpus: list[int]) -> int:
    """优先 8 卡，不够时接受 4 卡。"""
    if len(free_gpus) >= 8:
        return 8
    if len(free_gpus) >= 4:
        return 4
    return 0


def start_training(args: argparse.Namespace, world_size: int, gpus: list[int]) -> None:
    session = f"{args.session_prefix}_ddp{world_size}"
    if tmux_has_session(session):
        log(f"training session {session!r} already exists; monitor exits")
        return

    config = args.ddp8_config if world_size == 8 else args.ddp4_config
    log_dir = args.project_dir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.session_prefix}_ddp{world_size}.log"
    visible = ",".join(str(idx) for idx in gpus[:world_size])

    resume_part = f"--resume {args.resume}" if Path(args.resume).exists() else ""
    if resume_part:
        log(f"resume from {args.resume}")
    else:
        log(f"resume checkpoint not found, start from scratch: {args.resume}")

    command = (
        f"cd {args.project_dir} && "
        f"CUDA_VISIBLE_DEVICES={visible} {args.python_bin} -u "
        f"-m torch.distributed.run --nproc_per_node={world_size} --master_port={args.master_port} "
        f"-m src.train --config {config} --num-workers {args.num_workers} {resume_part} "
        f"> {log_path} 2>&1"
    )
    log(f"start {world_size}-GPU training session={session} gpus={visible} log={log_path}")
    run(["tmux", "new-session", "-d", "-s", session, command], cwd=args.project_dir, check=True)


def main() -> None:
    args = parse_args()
    log("wait-start monitor started")
    log(
        f"threshold={args.free_mem_threshold_mib} MiB poll={args.poll_seconds}s "
        f"resume={args.resume}"
    )

    while True:
        for session in (f"{args.session_prefix}_ddp8", f"{args.session_prefix}_ddp4"):
            if tmux_has_session(session):
                log(f"training session {session!r} already running; monitor exits")
                return

        free_gpus, usage = free_gpu_indices(args.free_mem_threshold_mib)
        world_size = choose_world_size(free_gpus)
        if world_size:
            start_training(args, world_size, free_gpus)
            log("training submitted; monitor exits")
            return

        log(f"waiting for >=4 free GPUs; free={free_gpus}; usage_mib={usage}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
