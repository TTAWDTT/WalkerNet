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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

try:
    from .interfaces import FILL_VALUE
except ImportError:  # pragma: no cover
    from interfaces import FILL_VALUE


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


def masked_gradient_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """约束空间梯度，缓解只用 MSE 时的过度平滑。

    这里分别计算纬向/经向一阶差分；只有相邻两个格点都有效时，该差分才参与 loss。
    """
    mask = valid_mask[:, None].to(device=pred.device, dtype=torch.bool)

    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = mask[..., 1:, :] & mask[..., :-1, :]

    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = mask[..., :, 1:] & mask[..., :, :-1]

    return 0.5 * (
        _masked_mse_5d(pred_dy, target_dy, mask_dy)
        + _masked_mse_5d(pred_dx, target_dx, mask_dx)
    )


def nino34_region_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """从预测 tos 场计算 Niño3.4 区域平均，再和真实场比较。

    该项仍然是 field-first：loss 输入是预测场，不是模型直接输出的指数。
    """
    if pred.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"pred/target spatial mismatch: {pred.shape} vs {target.shape}")

    H, W = pred.shape[-2:]
    lat = torch.linspace(-89.5, 89.5, H, device=pred.device, dtype=pred.dtype)
    lon = torch.linspace(0.5, 359.5, W, device=pred.device, dtype=pred.dtype)
    lat_mask = (lat >= -5.0) & (lat <= 5.0)
    lon_mask = (lon >= 190.0) & (lon <= 240.0)
    if not bool(lat_mask.any()) or not bool(lon_mask.any()):
        return pred.new_zeros(())

    pred_region = pred[:, :, 0][..., lat_mask, :][..., lon_mask]
    target_region = target[:, :, 0][..., lat_mask, :][..., lon_mask]
    mask_region = valid_mask[:, None, 0][..., lat_mask, :][..., lon_mask].to(device=pred.device)

    weights = torch.cos(torch.deg2rad(lat[lat_mask]))
    weights = weights / weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
    view_shape = [1] * pred_region.ndim
    view_shape[-2] = weights.numel()
    weights = weights.view(*view_shape)

    pred_index = _weighted_region_mean(pred_region, mask_region, weights)
    target_index = _weighted_region_mean(target_region, mask_region, weights)
    return torch.mean((pred_index - target_index) ** 2)


def forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_last: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: dict[str, float],
) -> torch.Tensor:
    """组合单个 lead 的 field/residual/gradient/Niño 区域 loss。"""
    loss = pred.new_zeros(())

    field_weight = float(weights.get("field", 1.0))
    residual_weight = float(weights.get("residual", 0.0))
    gradient_weight = float(weights.get("gradient", 0.0))
    nino_weight = float(weights.get("nino34", 0.0))

    if field_weight:
        loss = loss + field_weight * masked_mse_loss(pred, target, valid_mask)
    if residual_weight:
        loss = loss + residual_weight * masked_mse_loss(pred - x_last, target - x_last, valid_mask)
    if gradient_weight:
        loss = loss + gradient_weight * masked_gradient_mse_loss(pred, target, valid_mask)
    if nino_weight:
        loss = loss + nino_weight * nino34_region_mse_loss(pred, target, valid_mask)
    return loss


def _masked_mse_5d(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 ``(B, 1, 4, H, W)`` 类张量按变量平均 masked MSE。"""
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    per_var_losses = []
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    squared = (pred - target) ** 2
    for var_idx in range(pred.shape[2]):
        var_squared = squared[:, :, var_idx]
        var_mask = mask_f[:, :, var_idx]
        denom = var_mask.sum().clamp_min(1.0)
        per_var_losses.append((var_squared * var_mask).sum() / denom)
    return torch.stack(per_var_losses).mean()


def _weighted_region_mean(values: torch.Tensor, mask: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    """计算带纬向权重和缺测 mask 的区域平均。"""
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    weighted = values * mask_f * lat_weights
    denom = (mask_f * lat_weights).sum(dim=(-2, -1)).clamp_min(torch.finfo(values.dtype).eps)
    return weighted.sum(dim=(-2, -1)) / denom


class Trainer:
    """负责训练、验证、保存 checkpoint 的轻量 Trainer。"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        device: torch.device | str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.training_config = config.get("training", config)
        self.logging_config = config.get("logging", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_distributed = self.world_size > 1 and dist.is_available() and dist.is_initialized()
        self.is_main = self.rank == 0

        self.model.to(self.device)

        self.epochs = int(self.training_config.get("epochs", 1))
        self.grad_accum_steps = max(1, int(self.training_config.get("grad_accum_steps", 1)))
        self.max_train_steps_per_epoch = int(self.training_config.get("max_train_steps_per_epoch", 0))
        self.max_val_steps = int(self.training_config.get("max_val_steps", 0))
        self.grad_clip = self.training_config.get("grad_clip")
        self.amp_enabled = bool(self.training_config.get("amp", False)) and self.device.type == "cuda"
        self.rollout_steps = max(1, int(self.training_config.get("rollout_steps", 1)))
        self.detach_rollout = bool(self.training_config.get("detach_rollout", True))
        self.lead_weights = self._build_lead_weights()
        self.loss_weights = dict(self.training_config.get("loss_weights", {"field": 1.0}))
        self.log_interval = int(self.logging_config.get("log_interval", 50))
        self.save_dir = Path(self.logging_config.get("save_dir", "checkpoints"))
        if self.is_main:
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
            is_last_limited_step = self.max_train_steps_per_epoch > 0 and step == self.max_train_steps_per_epoch
            should_step = step % self.grad_accum_steps == 0 or step == len(self.train_loader) or is_last_limited_step

            # DDP + 梯度累积：非 optimizer step 的小步先不做跨卡梯度同步。
            with self._sync_context(should_step):
                with autocast() if self.amp_enabled else nullcontext():
                    loss = self._compute_batch_loss(batch)
                    scaled_loss = loss / self.grad_accum_steps

                if self.amp_enabled:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if should_step:
                if self.grad_clip is not None:
                    if self.amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model_for_state().parameters(), float(self.grad_clip))

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

            if self.is_main and self.log_interval > 0 and step % self.log_interval == 0:
                avg = total_loss / max(num_batches, 1)
                print(
                    f"epoch={epoch} step={step}/{len(self.train_loader)} "
                    f"optimizer_steps={optimizer_steps} train_loss={avg:.6f}"
                )

        total_loss, num_batches = self._reduce_loss_stats(total_loss, num_batches)
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
                loss = self._compute_batch_loss(batch)
            total_loss += float(loss.detach().cpu())
            num_batches += 1

        total_loss, num_batches = self._reduce_loss_stats(total_loss, num_batches)
        return {"loss": total_loss / max(num_batches, 1)}

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        """保存模型、优化器和指标。"""
        if not self.is_main:
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self._model_for_state().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> tuple[int, dict[str, float]]:
        """读取 checkpoint，返回 epoch 和 metrics。"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model"]
        self._model_for_state().load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))
        return int(checkpoint.get("epoch", 0)), dict(checkpoint.get("metrics", {}))

    def train(self) -> None:
        """完整训练循环。"""
        for epoch in range(1, self.epochs + 1):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.is_main:
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

    def _model_for_state(self) -> torch.nn.Module:
        """返回真正持有参数的模型；DDP 包装时需要取 .module。"""
        return self.model.module if hasattr(self.model, "module") else self.model

    def _sync_context(self, should_step: bool):
        """DDP 梯度累积时，非 optimizer step 的 micro-batch 关闭梯度同步。"""
        if self.is_distributed and not should_step and hasattr(self.model, "no_sync"):
            return self.model.no_sync()
        return nullcontext()

    def _reduce_loss_stats(self, total_loss: float, num_batches: int) -> tuple[float, int]:
        """把各 rank 的 loss sum / batch count 汇总成全局平均所需统计量。"""
        if not self.is_distributed:
            return total_loss, num_batches

        stats = torch.tensor([total_loss, float(num_batches)], dtype=torch.float64, device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return float(stats[0].item()), int(stats[1].item())

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """把 batch 中的 tensor 移到训练设备。"""
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _compute_batch_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """计算单步或多步滚动训练 loss。"""
        window = batch["x"]
        valid_mask = batch["valid_mask"]
        targets = batch.get("y_rollout", batch["y"])
        if targets.ndim != 5:
            raise ValueError(f"targets must be (B, K, 4, H, W), got {targets.shape}")

        available_steps = targets.shape[1]
        steps = min(self.rollout_steps, available_steps)
        target_months = batch.get("target_months")
        total = window.new_zeros(())
        lead_weights = self.lead_weights[:steps].to(device=window.device, dtype=window.dtype)

        for step_idx in range(steps):
            x_last = window[:, -1:].contiguous()
            target = targets[:, step_idx : step_idx + 1]
            if target_months is None:
                target_month = self._advance_month(batch["target_month"], step_idx)
            else:
                target_month = target_months[:, step_idx]
            rollout_step = torch.full(
                (window.shape[0],),
                step_idx,
                dtype=torch.long,
                device=window.device,
            )

            pred = self.model(window, target_month, rollout_step=rollout_step)
            total = total + lead_weights[step_idx] * forecast_loss(
                pred,
                target,
                x_last,
                valid_mask,
                self.loss_weights,
            )

            if step_idx + 1 < steps:
                next_frame = pred.detach() if self.detach_rollout else pred
                next_frame = self._mask_next_frame(next_frame, valid_mask)
                window = torch.cat([window[:, 1:], next_frame], dim=1)

        return total

    def _build_lead_weights(self) -> torch.Tensor:
        """读取并归一化 rollout lead 权重。"""
        raw = self.training_config.get("lead_weights")
        if raw is None:
            raw = [1.0]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            raise ValueError("training.lead_weights must be a non-empty list")

        values = torch.tensor([float(x) for x in raw], dtype=torch.float32)
        if values.numel() < self.rollout_steps:
            pad = values[-1].repeat(self.rollout_steps - values.numel())
            values = torch.cat([values, pad])
        values = values[: self.rollout_steps]
        return values / values.sum().clamp_min(torch.finfo(values.dtype).eps)

    @staticmethod
    def _advance_month(target_month: torch.Tensor, step_idx: int) -> torch.Tensor:
        """当 batch 没有 target_months 时，根据首个目标月推算后续月份。"""
        return ((target_month.to(dtype=torch.long) - 1 + step_idx) % 12) + 1

    @staticmethod
    def _mask_next_frame(frame: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """把预测场接回输入窗口前，确保无效格点仍是填充值。"""
        mask = valid_mask[:, None].to(device=frame.device, dtype=torch.bool)
        fill = torch.as_tensor(FILL_VALUE, device=frame.device, dtype=frame.dtype)
        return torch.where(mask, frame, fill)
