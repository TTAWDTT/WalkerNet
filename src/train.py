"""WalkerNet 训练入口。

用法：
    python -m src.train --config configs/default.yaml

注意：
    当前 model.py 仍由 Ziyi Zhuang 实现。如果 WalkerNet 还没有实现，
    本入口会在创建模型时明确报错。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import WalkerDataset
from .model import WalkerNet
from .trainer import Trainer
from .utils import count_parameters, get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def is_distributed() -> bool:
    """判断当前进程是否由 torchrun 拉起。"""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[bool, int, int, int]:
    """初始化 DDP，返回 distributed/rank/local_rank/world_size。"""
    distributed = is_distributed()
    if not distributed:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    """训练结束后关闭 DDP 进程组。"""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def rank_log(rank: int, message: str) -> None:
    """打印带 rank 前缀的启动阶段日志，方便定位 DDP 卡点。"""
    print(f"[rank {rank}] {message}", flush=True)


def prepare_norm_stats_cache(config: dict[str, Any], distributed: bool, rank: int) -> None:
    """DDP 训练前准备归一化统计缓存。

    WalkerDataset 默认会在 train split 上计算 mean/std。多卡时如果每个 rank
    都重复计算，会在启动阶段浪费很多时间；这里让 rank 0 先算并落盘，其它 rank
    barrier 等待后直接读取缓存。
    """
    data_config = config.get("data", {})
    stats_path = data_config.get("norm_stats_path")
    norm = str(data_config.get("norm", "zscore")).lower()
    if not distributed or not stats_path or norm == "none":
        return

    if rank == 0:
        rank_log(rank, f"prepare norm stats cache: {stats_path}")
        WalkerDataset(data_config["path"], config, split="train")
        rank_log(rank, "norm stats cache ready")
    dist.barrier()


def build_dataloaders(
    config: dict[str, Any],
    num_workers: int = 0,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader。"""
    data_config = config["data"]
    training_config = config["training"]
    batch_size = int(training_config.get("batch_size", 1))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_set = WalkerDataset(data_config["path"], config, split="train")
    val_set = WalkerDataset(data_config["path"], config, split="val", norm_stats=train_set.norm_stats)
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    set_seed(int(config.get("training", {}).get("seed", 42)))
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device() if args.device is None else torch.device(args.device)

    rank_log(rank, "building dataloaders...")
    prepare_norm_stats_cache(config, distributed=distributed, rank=rank)
    train_loader, val_loader = build_dataloaders(
        config,
        num_workers=args.num_workers,
        distributed=distributed,
    )
    rank_log(rank, "dataloaders ready")

    rank_log(rank, "building model...")
    model = WalkerNet(config)
    rank_log(rank, "model built")
    total_params, trainable_params = count_parameters(model)
    if distributed:
        rank_log(rank, f"moving model to {device}")
        model.to(device)
        rank_log(rank, "wrapping model with DistributedDataParallel")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        rank_log(rank, "ddp model ready")
    if is_main:
        print(f"device={device}")
        print(f"distributed={distributed} world_size={world_size}")
        print(f"params total={total_params:,} trainable={trainable_params:,}")
        print(f"train batches={len(train_loader)} val batches={len(val_loader)}")

    try:
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            config,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if args.resume is not None:
            epoch, metrics = trainer.load_checkpoint(args.resume)
            if is_main:
                print(f"resumed from {args.resume} (epoch={epoch}, metrics={metrics})")
        trainer.train()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
