"""Synthetic smoke test for WalkerNet.

Verifies forward shape, backward pass, and gradient flow on the production
config (180x360, 6 ViT layers). Targets GPU; CPU is intentionally not supported
since the production model exceeds typical local CPU memory budgets.

Run on a GPU machine from project root:
    python scripts/smoke_test_model.py
"""

from __future__ import annotations

import time

import torch

from src.model import WalkerNet
from src.trainer import masked_mse_loss


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. WalkerNet smoke test is GPU-only. "
            "Run this on a GPU server (the production model needs ~12-20 GB activations on full config)."
        )

    device = torch.device("cuda")
    print(f"device={device} ({torch.cuda.get_device_name(0)})")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    config = {
        "data": {"L": 3, "H": 180, "W": 360},
        "model": {
            "patch_size": 4,
            "d_model": 256,
            "nhead": 8,
            "dim_ff": 1024,
            "num_layers": 6,
            "num_experts": 12,
            "dropout": 0.1,
            "max_rollout_steps": 24,
        },
    }

    torch.manual_seed(0)
    model = WalkerNet(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"params={total:,}")

    B, L, V, H, W = 2, 3, 4, 180, 360
    x = torch.randn(B, L, V, H, W, device=device)
    target_month = torch.randint(1, 13, (B,), dtype=torch.long, device=device)
    y = torch.randn(B, 1, V, H, W, device=device)
    valid_mask = torch.ones(B, V, H, W, dtype=torch.bool, device=device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    y_pred = model(x, target_month)
    print(f"y_pred (no rollout) shape={tuple(y_pred.shape)} dtype={y_pred.dtype}")
    assert y_pred.shape == (B, 1, V, H, W), f"shape mismatch: {y_pred.shape}"
    assert y_pred.dtype == torch.float32

    rollout = torch.tensor([0, 5], dtype=torch.long, device=device)
    y_pred2 = model(x, target_month, rollout_step=rollout)
    print(f"y_pred (rollout=[0,5]) shape={tuple(y_pred2.shape)}")
    assert y_pred2.shape == (B, 1, V, H, W)

    loss = masked_mse_loss(y_pred, y, valid_mask)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    grad_ok = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    grad_nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"loss={loss.item():.4f} grad_finite={grad_ok}/{n_total} grad_nonzero={grad_nonzero}/{n_total}")
    print(f"peak_gpu_mem={peak_mem_gb:.2f} GB  elapsed={elapsed:.2f}s")

    assert grad_ok == n_total, "some params have non-finite grads"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
