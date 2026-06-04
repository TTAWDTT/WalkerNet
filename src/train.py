"""WalkerNet 训练入口。

用法：
    python -m src.train --config configs/default.yaml

注意：
    当前 model.py 仍由 Ziyi Zhuang 实现。如果 WalkerNet 还没有实现，
    本入口会在创建模型时明确报错。
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

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


def build_dataloaders(config: dict[str, Any], num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """构建 train/val DataLoader。"""
    data_config = config["data"]
    training_config = config["training"]
    batch_size = int(training_config.get("batch_size", 1))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_set = WalkerDataset(data_config["path"], config, split="train")
    val_set = WalkerDataset(data_config["path"], config, split="val", norm_stats=train_set.norm_stats)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = get_device() if args.device is None else args.device

    train_loader, val_loader = build_dataloaders(config, num_workers=args.num_workers)

    model = WalkerNet(config)
    total_params, trainable_params = count_parameters(model)
    print(f"device={device}")
    print(f"params total={total_params:,} trainable={trainable_params:,}")
    print(f"train batches={len(train_loader)} val batches={len(val_loader)}")

    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    if args.resume is not None:
        epoch, metrics = trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} (epoch={epoch}, metrics={metrics})")
    trainer.train()


if __name__ == "__main__":
    main()
