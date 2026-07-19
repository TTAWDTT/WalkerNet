"""等待 8 张 GPU 全部空闲后启动 DDP8 训练。"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait all GPUs free and start WalkerNet DDP8.")
    parser.add_argument("--project-dir", type=Path, default=Path("/mnt/sda/WalkerNet"))
    parser.add_argument("--python-bin", default="/home/cpji/wwb/torch/bin/python")
    parser.add_argument("--session", default="walker_mixed5_skill_recover8_0621_ddp8")
    parser.add_argument("--config", default="configs/server_3090_mixed5_ddp8.yaml")
    parser.add_argument("--resume", default="/mnt/sda/WalkerNet/checkpoints_mixed5_ddp8/best.pt")
    parser.add_argument("--log", default="outputs/logs/walker_mixed5_skill_recover8_0621_ddp8.log")
    parser.add_argument("--master-port", type=int, default=29549)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--threshold-mib", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def tmux_has_session(name: str) -> bool:
    return run(["tmux", "has-session", "-t", name]).returncode == 0


def gpu_memory() -> dict[int, int]:
    result = run([
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ])
    result.check_returncode()
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        idx, mem = [item.strip() for item in line.split(",", maxsplit=1)]
        usage[int(idx)] = int(mem)
    return usage


def start(args: argparse.Namespace) -> None:
    if tmux_has_session(args.session):
        log(f"training session exists: {args.session}")
        return
    Path(args.project_dir / Path(args.log).parent).mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {args.project_dir} && "
        f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 {args.python_bin} -u "
        f"-m torch.distributed.run --nproc_per_node=8 --master_port={args.master_port} "
        f"-m src.train --config {args.config} --num-workers {args.num_workers} --resume {args.resume} "
        f"> {args.log} 2>&1"
    )
    run(["tmux", "new-session", "-d", "-s", args.session, command], cwd=args.project_dir).check_returncode()
    log(f"started {args.session} from {args.resume}")


def main() -> None:
    args = parse_args()
    log(f"wait all 8 GPUs free; resume={args.resume}")
    while True:
        if tmux_has_session(args.session):
            log(f"training session already running: {args.session}")
            return
        usage = gpu_memory()
        busy = {idx: mem for idx, mem in usage.items() if mem > args.threshold_mib}
        if not busy:
            start(args)
            return
        log(f"waiting; busy={busy}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
