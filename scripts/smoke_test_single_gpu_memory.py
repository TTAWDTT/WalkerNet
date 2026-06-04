"""Single-GPU memory probe for WalkerNet.

This script does not read real NetCDF data. It creates synthetic full-resolution
batches and runs forward/backward to estimate whether a single GPU can train the
current model.

Example on the server:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_single_gpu_memory.py \
        --config configs/server_3090.yaml --batch-sizes 1 2 4
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces import NUM_VARIABLES
from src.model import WalkerNet
from src.trainer import masked_mse_loss
from src.utils import count_parameters, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet single-GPU memory probe")
    parser.add_argument("--config", type=str, default="configs/server_3090.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if config enables it")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _gb(num_bytes: int) -> float:
    return num_bytes / 1024**3


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_probe(model: WalkerNet, config: dict, batch_size: int, device: torch.device, amp: bool) -> bool:
    data_cfg = config["data"]
    L = int(data_cfg["L"])
    H = int(data_cfg["H"])
    W = int(data_cfg["W"])
    V = NUM_VARIABLES

    _clear_cuda()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)

    try:
        x = torch.randn(batch_size, L, V, H, W, device=device)
        y = torch.randn(batch_size, 1, V, H, W, device=device)
        valid_mask = torch.ones(batch_size, V, H, W, dtype=torch.bool, device=device)
        target_month = torch.randint(1, 13, (batch_size,), dtype=torch.long, device=device)

        scaler = GradScaler(enabled=amp)
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        with autocast(enabled=amp):
            pred = model(x, target_month)
            loss = masked_mse_loss(pred, y, valid_mask)

        scaler.scale(loss).backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        print(
            f"batch_size={batch_size} OK "
            f"loss={loss.detach().item():.4f} "
            f"time={elapsed:.2f}s "
            f"base_alloc={_gb(base_alloc):.2f}GB "
            f"base_reserved={_gb(base_reserved):.2f}GB "
            f"peak_alloc={_gb(peak_alloc):.2f}GB "
            f"peak_reserved={_gb(peak_reserved):.2f}GB"
        )
        return True
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"batch_size={batch_size} OOM: {exc}")
        return False
    finally:
        model.zero_grad(set_to_none=True)
        _clear_cuda()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this memory probe")

    set_seed(args.seed)
    config = load_config(args.config)
    device = torch.device(args.device)
    amp = bool(config.get("training", {}).get("amp", False)) and not args.no_amp

    torch.cuda.set_device(device)
    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"amp={amp}")

    model = WalkerNet(config).to(device)
    total, trainable = count_parameters(model)
    print(f"params total={total:,} trainable={trainable:,}")

    for batch_size in args.batch_sizes:
        run_probe(model, config, batch_size, device, amp)


if __name__ == "__main__":
    main()
