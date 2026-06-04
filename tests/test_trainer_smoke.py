"""Trainer smoke tests with synthetic data.

These tests verify:
- gradient accumulation triggers the expected number of optimizer steps
- checkpoint save/load keeps the model state intact

Run standalone:
    python tests/test_trainer_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer import Trainer  # noqa: E402


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        x = torch.zeros(1, 12, 4, 4, 4, dtype=torch.float32)
        y = torch.ones(1, 1, 4, 4, 4, dtype=torch.float32)
        mask = torch.ones(1, 4, 4, 4, dtype=torch.bool)
        return {
            "x": x.squeeze(0),
            "y": y.squeeze(0),
            "target_month": int((idx % 12) + 1),
            "valid_mask": mask.squeeze(0),
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, target_month, rollout_step=None):
        B, _, _, H, W = x.shape
        return self.bias.view(1, 1, 1, 1, 1).expand(B, 1, 4, H, W)


def test_trainer_gradient_accumulation_and_checkpoint():
    train_loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)
    val_loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)
    config = {
        "training": {
            "epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "grad_accum_steps": 2,
            "grad_clip": 1.0,
            "amp": False,
        },
        "logging": {"log_interval": 1, "save_dir": "checkpoints"},
    }

    model = TinyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        config["logging"]["save_dir"] = tmpdir
        trainer = Trainer(model, train_loader, val_loader, config, device="cpu")

        step_calls: list[int] = []
        original_step = trainer.optimizer.step

        def counted_step(*args, **kwargs):
            step_calls.append(1)
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counted_step  # type: ignore[method-assign]

        before = model.bias.detach().item()
        metrics = trainer.train_epoch(1)
        after = model.bias.detach().item()

        assert metrics["optimizer_steps"] == 2.0
        assert len(step_calls) == 2
        assert after != before

        ckpt_path = Path(tmpdir) / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1, metrics=metrics)
        assert ckpt_path.exists()

        restored = TinyModel()
        restored_trainer = Trainer(restored, train_loader, val_loader, config, device="cpu")
        epoch, loaded_metrics = restored_trainer.load_checkpoint(ckpt_path)
        assert epoch == 1
        assert loaded_metrics["optimizer_steps"] == 2.0
        assert torch.allclose(restored.bias, model.bias)


def _run_all():
    import traceback

    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failed: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
        else:
            print(f"PASS  {name}")
    print()
    if failed:
        print(f"{len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
