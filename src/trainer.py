"""WalkerNet 训练循环。

依赖 interfaces.py 中约定的调用形式：

    y_pred = model(batch["x"], batch["target_month"])

loss 使用 valid_mask 忽略无效区域。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    variable_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """按变量计算 masked MSE，再对变量取平均。

    Args:
        pred: ``(B, 1, 4, H, W)`` 1 是输出的长度（不是输入的长度）
        target: ``(B, 1, 4, H, W)``
        valid_mask: ``(B, 4, H, W)``，True 表示有效
        variable_weights: 可选 ``(4,)``，用于变量加权

    Returns:
        标量 loss
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim != 5:
        raise ValueError(f"pred/target must be (B, 1, 4, H, W), got {pred.shape}")
    if valid_mask.shape != (pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]):
        raise ValueError(f"valid_mask shape mismatch: got {valid_mask.shape}, pred={pred.shape}")

    mask = valid_mask[:, None].to(device=pred.device, dtype=pred.dtype)
    squared = (pred - target) ** 2

    per_var_losses = []
    for var_idx in range(pred.shape[2]):
        var_squared = squared[:, :, var_idx]
        var_mask = mask[:, :, var_idx]
        denom = var_mask.sum().clamp_min(1.0)
        per_var_losses.append((var_squared * var_mask).sum() / denom)

    losses = torch.stack(per_var_losses)
    if variable_weights is not None:
        weights = variable_weights.to(device=pred.device, dtype=pred.dtype)
        weights = weights / weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
        return (losses * weights).sum()
    return losses.mean()


class Trainer:
    """负责训练、验证、保存 checkpoint 的轻量 Trainer。"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.training_config = config.get("training", config)
        self.logging_config = config.get("logging", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model.to(self.device)

        self.epochs = int(self.training_config.get("epochs", 1))
        self.grad_accum_steps = max(1, int(self.training_config.get("grad_accum_steps", 1)))
        self.max_train_steps_per_epoch = int(self.training_config.get("max_train_steps_per_epoch", 0))
        self.max_val_steps = int(self.training_config.get("max_val_steps", 0))
        self.grad_clip = self.training_config.get("grad_clip")
        self.amp_enabled = bool(self.training_config.get("amp", False)) and self.device.type == "cuda"
        self.log_interval = int(self.logging_config.get("log_interval", 50))
        self.save_dir = Path(self.logging_config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_config.get("lr", 1e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
            except TypeError:  # pragma: no cover - older torch signature variance
                self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        else:  # pragma: no cover - older torch fallback
            from torch.cuda.amp import GradScaler as CudaGradScaler

            self.scaler = CudaGradScaler(enabled=self.amp_enabled)

        self.best_val_loss = float("inf")
        self.global_step = 0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader, start=1):
            if self.max_train_steps_per_epoch > 0 and step > self.max_train_steps_per_epoch:
                break

            batch = self._move_batch(batch)

            with autocast() if self.amp_enabled else nullcontext():
                pred = self.model(batch["x"], batch["target_month"])
                loss = masked_mse_loss(pred, batch["y"], batch["valid_mask"])
                scaled_loss = loss / self.grad_accum_steps

            if self.amp_enabled:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_last_limited_step = self.max_train_steps_per_epoch > 0 and step == self.max_train_steps_per_epoch
            should_step = step % self.grad_accum_steps == 0 or step == len(self.train_loader) or is_last_limited_step
            if should_step:
                if self.grad_clip is not None:
                    if self.amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))

                if self.amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                self.global_step += 1

            total_loss += float(loss.detach().cpu())
            num_batches += 1

            if self.log_interval > 0 and step % self.log_interval == 0:
                avg = total_loss / max(num_batches, 1)
                print(
                    f"epoch={epoch} step={step}/{len(self.train_loader)} "
                    f"optimizer_steps={optimizer_steps} train_loss={avg:.6f}"
                )

        return {
            "loss": total_loss / max(num_batches, 1),
            "optimizer_steps": float(optimizer_steps),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """在验证集上计算 loss。"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            if self.max_val_steps > 0 and num_batches >= self.max_val_steps:
                break

            batch = self._move_batch(batch)
            with autocast() if self.amp_enabled else nullcontext():
                pred = self.model(batch["x"], batch["target_month"])
                loss = masked_mse_loss(pred, batch["y"], batch["valid_mask"])
            total_loss += float(loss.detach().cpu())
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        """保存模型、优化器和指标。"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> tuple[int, dict[str, float]]:
        """读取 checkpoint，返回 epoch 和 metrics。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))
        return int(checkpoint.get("epoch", 0)), dict(checkpoint.get("metrics", {}))

    def train(self) -> None:
        """完整训练循环。"""
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            message = f"epoch={epoch} train_loss={train_metrics['loss']:.6f}"
            if val_metrics:
                message += f" val_loss={val_metrics['loss']:.6f}"
            print(message)

            latest_path = self.save_dir / "latest.pt"
            self.save_checkpoint(latest_path, epoch, {"train": train_metrics, "val": val_metrics})

            if val_metrics and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = self.save_dir / "best.pt"
                self.save_checkpoint(best_path, epoch, {"train": train_metrics, "val": val_metrics})

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """把 batch 中的 tensor 移到训练设备。"""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved
