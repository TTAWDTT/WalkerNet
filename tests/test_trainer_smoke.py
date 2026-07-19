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
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer import Trainer, nino34_delta_mse_loss  # noqa: E402


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


class TinyRolloutDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        x = torch.zeros(12, 4, 4, 4, dtype=torch.float32)
        y_rollout = torch.stack([
            torch.ones(4, 4, 4, dtype=torch.float32),
            torch.full((4, 4, 4), 2.0, dtype=torch.float32),
        ])
        return {
            "x": x,
            "y": y_rollout[:1],
            "y_rollout": y_rollout,
            "target_month": int((idx % 12) + 1),
            "target_months": torch.tensor([1, 2], dtype=torch.long),
            "valid_mask": torch.ones(4, 4, 4, dtype=torch.bool),
        }


class RecordingRolloutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.seen_steps: list[int] = []

    def forward(self, x, target_month, rollout_step=None):
        if rollout_step is not None:
            self.seen_steps.extend(int(v) for v in rollout_step.detach().cpu().tolist())
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


def test_trainer_uses_rollout_targets_and_steps():
    train_loader = DataLoader(TinyRolloutDataset(), batch_size=1, shuffle=False)
    config = {
        "training": {
            "epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "grad_accum_steps": 1,
            "grad_clip": 1.0,
            "amp": False,
            "rollout_steps": 2,
            "lead_weights": [1.0, 0.8],
            "detach_rollout": True,
            "loss_weights": {
                "field": 1.0,
                "residual": 0.5,
                "gradient": 0.05,
                "nino34": 0.2,
            },
        },
        "logging": {"log_interval": 0, "save_dir": "checkpoints"},
    }

    model = RecordingRolloutModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        config["logging"]["save_dir"] = tmpdir
        trainer = Trainer(model, train_loader, None, config, device="cpu")
        metrics = trainer.train_epoch(1)

    assert metrics["optimizer_steps"] == 3.0
    assert model.seen_steps == [0, 1, 0, 1, 0, 1]


def test_nino34_delta_loss_tracks_index_correction():
    H, W = 180, 360
    x_last = torch.zeros(1, 1, 4, H, W, dtype=torch.float32)
    pred = x_last.clone()
    target = x_last.clone()
    valid_mask = torch.ones(1, 4, H, W, dtype=torch.bool)

    lat = torch.linspace(-89.5, 89.5, H)
    lat_mask = (lat >= -5.0) & (lat <= 5.0)

    pred[:, :, 0, lat_mask, :] = 2.0
    target[:, :, 0, lat_mask, :] = 3.0

    loss = nino34_delta_mse_loss(pred, target, x_last, valid_mask)

    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_rollout_selection_is_broadcast_to_worker_rank():
    """非主 rank 应等待并接收 rank 0 的 rollout 指标。"""
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)
    trainer = Trainer(TinyModel(), loader, loader, {"training": {}}, device="cpu", rank=1)
    trainer.is_distributed = True

    expected = {"score": 0.42, "leads": [6, 12]}

    def receive_from_main(payload, src):
        assert src == 0
        payload[0] = (expected, None)

    with patch("src.trainer.dist.broadcast_object_list", side_effect=receive_from_main) as broadcast:
        result = trainer.evaluate_rollout_selection(epoch=3)

    assert result == expected
    broadcast.assert_called_once()


def test_rollout_selection_error_is_broadcast_to_worker_rank():
    """rank 0 的评测异常应让全部 rank 一起失败，避免其它 rank 继续训练。"""
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)
    trainer = Trainer(TinyModel(), loader, loader, {"training": {}}, device="cpu", rank=1)
    trainer.is_distributed = True

    def receive_error(payload, src):
        payload[0] = ({}, "ValueError: bad rollout data")

    with patch("src.trainer.dist.broadcast_object_list", side_effect=receive_error):
        try:
            trainer.evaluate_rollout_selection(epoch=3)
        except RuntimeError as exc:
            assert "bad rollout data" in str(exc)
        else:
            raise AssertionError("worker rank did not receive rank 0 rollout error")


def test_early_stop_flag_uses_rank_zero_decision():
    """早停控制信号必须以 rank 0 为准。"""
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)
    trainer = Trainer(TinyModel(), loader, loader, {"training": {}}, device="cpu", rank=1)
    trainer.is_distributed = True

    def receive_stop(flag, src):
        flag.fill_(1)

    with patch("src.trainer.dist.broadcast", side_effect=receive_stop):
        assert trainer._broadcast_bool_from_main(False) is True


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
