"""双进程验证 rollout 指标与早停信号的 DDP 同步。

运行方式：
    python -m torch.distributed.run --standalone --nproc-per-node=2 tests/test_trainer_ddp_sync.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer import Trainer  # noqa: E402


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        return {
            "x": torch.zeros(12, 4, 4, 4),
            "y": torch.zeros(1, 4, 4, 4),
            "target_month": 1,
            "valid_mask": torch.ones(4, 4, 4, dtype=torch.bool),
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, target_month, rollout_step=None):
        batch, _, _, height, width = x.shape
        return self.bias.reshape(1, 1, 1, 1, 1).expand(batch, 1, 4, height, width)


def run_rank(rank: int, world_size: int, init_method: str | None = None) -> None:
    backend = os.environ.get("TEST_BACKEND", "gloo")
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if init_method is None:
        dist.init_process_group(backend)
    else:
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    loader = DataLoader(TinyDataset(), batch_size=1)
    trainer = Trainer(
        TinyModel(),
        loader,
        loader,
        {"training": {}, "logging": {"log_interval": 0}},
        device=device,
        rank=rank,
        world_size=world_size,
    )

    expected = {"score": 0.42, "leads": [6, 12]}
    if rank == 0:
        trainer._evaluate_rollout_selection_on_main = lambda epoch: expected  # type: ignore[method-assign]

    assert trainer.evaluate_rollout_selection(epoch=3) == expected
    trainer._distributed_barrier()
    assert trainer._broadcast_bool_from_main(rank == 0) is True

    dist.destroy_process_group()
    print(f"rank={rank} PASS", flush=True)


def main() -> None:
    if "RANK" in os.environ:
        run_rank(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
        return

    # Windows 的部分 PyTorch 构建没有 libuv，直接用 FileStore 启动本地双进程测试。
    with tempfile.TemporaryDirectory() as tmpdir:
        rendezvous = Path(tmpdir) / "gloo_rendezvous"
        init_method = rendezvous.as_uri()
        torch.multiprocessing.spawn(run_rank, args=(2, init_method), nprocs=2, join=True)


if __name__ == "__main__":
    main()
