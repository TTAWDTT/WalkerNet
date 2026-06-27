"""WalkerNet 训练循环。

依赖 interfaces.py 中约定的调用形式：

    y_pred = model(batch["x"], batch["target_month"])

loss 使用 valid_mask 忽略无效区域。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

try:
    from .interfaces import FILL_VALUE
    from .metrics import compute_nino34
except ImportError:  # pragma: no cover
    from interfaces import FILL_VALUE
    from metrics import compute_nino34


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    variable_weights: torch.Tensor | None = None,
    area_weighted: bool = False,
) -> torch.Tensor:
    """按变量计算 masked MSE，再对变量取平均。

    Args:
        pred: ``(B, 1, 4, H, W)`` 1 是输出的长度（不是输入的长度）
        target: ``(B, 1, 4, H, W)``
        valid_mask: ``(B, 4, H, W)``，True 表示有效
        variable_weights: 可选 ``(4,)``，用于变量加权
        area_weighted: 是否按 ``cos(lat)`` 做面积权重

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
        area = _latitude_area_weights(var_squared, enabled=area_weighted)
        denom = (var_mask * area).sum().clamp_min(torch.finfo(pred.dtype).eps)
        per_var_losses.append((var_squared * var_mask * area).sum() / denom)

    losses = torch.stack(per_var_losses)
    if variable_weights is not None:
        weights = variable_weights.to(device=pred.device, dtype=pred.dtype)
        weights = weights / weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
        return (losses * weights).sum()
    return losses.mean()


def masked_gradient_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    area_weighted: bool = False,
) -> torch.Tensor:
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
        _masked_mse_5d(pred_dy, target_dy, mask_dy, area_weighted=area_weighted)
        + _masked_mse_5d(pred_dx, target_dx, mask_dx, area_weighted=area_weighted)
    )


def nino34_region_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """从预测 tos 场计算 Niño3.4 区域平均，再和真实场比较。

    该项仍然是 field-first：loss 输入是预测场，不是模型直接输出的指数。
    """
    pred_index = _nino34_index(pred, valid_mask)
    target_index = _nino34_index(target, valid_mask)
    if pred_index.numel() == 0:
        return pred.new_zeros(())
    return torch.mean((pred_index - target_index) ** 2)


def nino34_delta_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_last: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """约束 Niño3.4 指数相对上一月的修正量。

    评测时会先预测完整 ``tos`` 场，再计算 Niño3.4 指数；这里也保持相同路径。
    与直接约束指数值不同，该项关注模型相对 persistence 做出的冷暖修正，
    用来减少自由滚动时的相位漂移。
    """
    pred_index = _nino34_index(pred, valid_mask)
    target_index = _nino34_index(target, valid_mask)
    last_index = _nino34_index(x_last, valid_mask)
    if pred_index.numel() == 0:
        return pred.new_zeros(())
    return torch.mean(((pred_index - last_index) - (target_index - last_index)) ** 2)


def nino34_pattern_corr_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """约束 Niño3.4 区域内 SST 异常形态相关。

    这里仍然是 field-first：先取预测/目标 tos 场的 Niño3.4 区域，再对每个样本
    减去区域平均，计算空间形态相关。它不是正式时间序列 ACC，但在 batch_size=1
    时也能提供“相位/形态不要塌掉”的训练信号。
    """
    pred_region, target_region, mask_region, weights = _nino34_region_tensors(pred, target, valid_mask)
    if pred_region.numel() == 0:
        return pred.new_zeros(())
    pred_anom = pred_region - _weighted_region_mean(pred_region, mask_region, weights)[..., None, None]
    target_anom = target_region - _weighted_region_mean(target_region, mask_region, weights)[..., None, None]
    mask_f = mask_region.to(device=pred.device, dtype=pred.dtype)
    weighted_mask = mask_f * weights
    numerator = (pred_anom * target_anom * weighted_mask).sum(dim=(-2, -1))
    pred_energy = (pred_anom.square() * weighted_mask).sum(dim=(-2, -1))
    target_energy = (target_anom.square() * weighted_mask).sum(dim=(-2, -1))
    denom = torch.sqrt(pred_energy * target_energy).clamp_min(torch.finfo(pred.dtype).eps)
    corr = numerator / denom
    return (1.0 - corr).mean()


def nino34_pattern_variance_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """约束 Niño3.4 区域内异常振幅，避免长 lead 过度平滑。"""
    pred_region, target_region, mask_region, weights = _nino34_region_tensors(pred, target, valid_mask)
    if pred_region.numel() == 0:
        return pred.new_zeros(())
    pred_anom = pred_region - _weighted_region_mean(pred_region, mask_region, weights)[..., None, None]
    target_anom = target_region - _weighted_region_mean(target_region, mask_region, weights)[..., None, None]
    mask_f = mask_region.to(device=pred.device, dtype=pred.dtype)
    weighted_mask = mask_f * weights
    pred_var = (pred_anom.square() * weighted_mask).sum(dim=(-2, -1))
    target_var = (target_anom.square() * weighted_mask).sum(dim=(-2, -1))
    target_var = target_var.clamp_min(torch.finfo(pred.dtype).eps)
    return ((pred_var - target_var) / target_var).square().mean()


def nino34_structure_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """约束 Niño3.4 区域内部冷暖结构，而不是只约束区域平均值。

    做法：先分别减去预测/目标在 Niño3.4 区域的加权平均，再对去均值后的
    tos 空间分布计算 masked MSE。这样保留 ``nino34`` 指数约束的同时，
    也要求区域内部的冷暖形态不要塌成一片。
    """
    pred_region, target_region, mask_region, weights = _nino34_region_tensors(pred, target, valid_mask)
    if pred_region.numel() == 0:
        return pred.new_zeros(())

    pred_anom = pred_region - _weighted_region_mean(pred_region, mask_region, weights)[..., None, None]
    target_anom = target_region - _weighted_region_mean(target_region, mask_region, weights)[..., None, None]
    mask_f = mask_region.to(device=pred.device, dtype=pred.dtype)
    weighted_mask = mask_f * weights
    denom = weighted_mask.sum().clamp_min(torch.finfo(pred.dtype).eps)
    return ((pred_anom - target_anom).square() * weighted_mask).sum() / denom


def nino34_batch_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """约束 batch 内 Niño3.4 时间序列相关，作为 ACC 的可训练近似。

    正式评测使用 Niño3.4 anomaly ACC；训练时每卡 batch 可能只有 1，
    所以 DDP 下先跨 rank 聚合 Niño3.4 标量，再计算相关损失。聚合时
    其它 rank 的值作为常量，本 rank 的值保留梯度，DDP 会再汇总梯度。
    """
    pred_index = _nino34_index(pred, valid_mask)
    target_index = _nino34_index(target, valid_mask)
    if pred_index.numel() == 0:
        return pred.new_zeros(())

    pred_values = _gather_1d_with_local_grad(pred_index, distributed, rank, world_size)
    target_values = _gather_1d_with_local_grad(target_index, distributed, rank, world_size)
    if pred_values.numel() < 2:
        return pred.new_zeros(())

    pred_anom = pred_values - pred_values.mean()
    target_anom = target_values - target_values.mean()
    numerator = (pred_anom * target_anom).sum()
    denom = torch.sqrt(pred_anom.square().sum() * target_anom.square().sum()).clamp_min(
        torch.finfo(pred.dtype).eps
    )
    corr = numerator / denom
    return 1.0 - corr


def tropical_pacific_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    area_weighted: bool = False,
) -> torch.Tensor:
    """热带太平洋四变量区域 loss。

    ENSO 不是只看一个指数；它依赖热带太平洋的海温、海面高度和风应力耦合。
    这里仍然计算四变量场误差，只是把热带太平洋这块“重点黑板”单独加权。
    """
    pred_region, target_region, mask_region = _region_tensors(
        pred,
        target,
        valid_mask,
        lat_bounds=(-20.0, 20.0),
        lon_bounds=(120.0, 290.0),
        variable_idx=None,
    )
    if pred_region.numel() == 0:
        return pred.new_zeros(())
    return _masked_mse_5d(pred_region, target_region, mask_region, area_weighted=area_weighted)


def forecast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_last: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: dict[str, float],
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """组合单个 lead 的 field/residual/gradient/Niño 区域 loss。"""
    loss = pred.new_zeros(())

    field_weight = float(weights.get("field", 1.0))
    residual_weight = float(weights.get("residual", 0.0))
    gradient_weight = float(weights.get("gradient", 0.0))
    tropical_pacific_weight = float(weights.get("tropical_pacific", 0.0))
    nino_weight = float(weights.get("nino34", 0.0))
    nino_delta_weight = float(weights.get("nino34_delta", 0.0))
    nino_corr_weight = float(weights.get("nino34_pattern_corr", 0.0))
    nino_var_weight = float(weights.get("nino34_pattern_variance", 0.0))
    nino_structure_weight = float(weights.get("nino34_structure", 0.0))
    nino_batch_corr_weight = float(weights.get("nino34_batch_corr", 0.0))
    area_weighted = bool(weights.get("area_weighted", False))

    if field_weight:
        loss = loss + field_weight * masked_mse_loss(pred, target, valid_mask, area_weighted=area_weighted)
    if residual_weight:
        pred_delta = _scale_residual_delta(pred - x_last, weights)
        target_delta = _scale_residual_delta(target - x_last, weights)
        loss = loss + residual_weight * masked_mse_loss(
            pred_delta,
            target_delta,
            valid_mask,
            area_weighted=area_weighted,
        )
    if gradient_weight:
        loss = loss + gradient_weight * masked_gradient_mse_loss(pred, target, valid_mask, area_weighted=area_weighted)
    if tropical_pacific_weight:
        loss = loss + tropical_pacific_weight * tropical_pacific_mse_loss(
            pred,
            target,
            valid_mask,
            area_weighted=area_weighted,
        )
    if nino_weight:
        loss = loss + nino_weight * nino34_region_mse_loss(pred, target, valid_mask)
    if nino_delta_weight:
        loss = loss + nino_delta_weight * nino34_delta_mse_loss(pred, target, x_last, valid_mask)
    if nino_corr_weight:
        loss = loss + nino_corr_weight * nino34_pattern_corr_loss(pred, target, valid_mask)
    if nino_var_weight:
        loss = loss + nino_var_weight * nino34_pattern_variance_loss(pred, target, valid_mask)
    if nino_structure_weight:
        loss = loss + nino_structure_weight * nino34_structure_loss(pred, target, valid_mask)
    if nino_batch_corr_weight:
        loss = loss + nino_batch_corr_weight * nino34_batch_corr_loss(
            pred,
            target,
            valid_mask,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
    return loss


def _tensor_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算一维张量相关系数。"""
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float(((x * y).sum() / denom).item())


def _tensor_rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算一维张量 RMSE。"""
    diff = x.double().reshape(-1) - y.double().reshape(-1)
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def _compute_nino34_numpy(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """从 ``(T, H, W)`` 的 tos 数据计算 Niño3.4 区域平均。"""
    lat_mask = (lat >= -5.0) & (lat <= 5.0)
    lon_mask = (lon >= 190.0) & (lon <= 240.0)
    region = data[:, lat_mask, :][:, :, lon_mask]
    lon_mean = np.nanmean(region, axis=2)
    weights = np.cos(np.deg2rad(lat[lat_mask])).astype(np.float64)
    weights = weights / weights.sum()
    return np.nansum(lon_mean * weights[None, :], axis=1).astype(np.float32)


def _masked_mse_5d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    area_weighted: bool = False,
) -> torch.Tensor:
    """对 ``(B, 1, 4, H, W)`` 类张量按变量平均 masked MSE。"""
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    per_var_losses = []
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    squared = (pred - target) ** 2
    for var_idx in range(pred.shape[2]):
        var_squared = squared[:, :, var_idx]
        var_mask = mask_f[:, :, var_idx]
        area = _latitude_area_weights(var_squared, enabled=area_weighted)
        denom = (var_mask * area).sum().clamp_min(torch.finfo(pred.dtype).eps)
        per_var_losses.append((var_squared * var_mask * area).sum() / denom)
    return torch.stack(per_var_losses).mean()


def _weighted_region_mean(values: torch.Tensor, mask: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    """计算带纬向权重和缺测 mask 的区域平均。"""
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    weighted = values * mask_f * lat_weights
    denom = (mask_f * lat_weights).sum(dim=(-2, -1)).clamp_min(torch.finfo(values.dtype).eps)
    return weighted.sum(dim=(-2, -1)) / denom


def _gather_1d_with_local_grad(
    values: torch.Tensor,
    distributed: bool,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """跨 rank 聚合一维张量，同时保留本 rank 片段的梯度。"""
    values = values.reshape(-1)
    if not distributed or world_size <= 1:
        return values

    gathered = [torch.zeros_like(values) for _ in range(world_size)]
    dist.all_gather(gathered, values.detach())
    gathered[rank] = values
    return torch.cat(gathered, dim=0)


def _nino34_index(field: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """从完整场中计算 Niño3.4 区域平均 ``tos`` 指数。"""
    dummy = field
    target = field
    region, _, mask_region, weights = _nino34_region_tensors(dummy, target, valid_mask)
    if region.numel() == 0:
        return field.new_empty((0,))
    return _weighted_region_mean(region, mask_region, weights)


def _latitude_area_weights(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    """返回可广播到 ``(..., H, W)`` 的纬向面积权重。"""
    if not enabled:
        return tensor.new_ones((1,) * tensor.ndim)
    h = tensor.shape[-2]
    lat = torch.linspace(-89.5, 89.5, h, device=tensor.device, dtype=tensor.dtype)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(torch.finfo(tensor.dtype).eps)
    weights = weights / weights.mean().clamp_min(torch.finfo(tensor.dtype).eps)
    view_shape = [1] * tensor.ndim
    view_shape[-2] = h
    return weights.view(*view_shape)


def _nino34_region_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """取出 Niño3.4 区域 tos 张量和对应 mask/纬向权重。"""
    H, W = pred.shape[-2:]
    lat = torch.linspace(-89.5, 89.5, H, device=pred.device, dtype=pred.dtype)
    lon = torch.linspace(0.5, 359.5, W, device=pred.device, dtype=pred.dtype)
    lat_mask = (lat >= -5.0) & (lat <= 5.0)
    lon_mask = (lon >= 190.0) & (lon <= 240.0)
    if not bool(lat_mask.any()) or not bool(lon_mask.any()):
        empty = pred.new_empty((0,))
        return empty, empty, empty.to(dtype=torch.bool), empty

    pred_region = pred[:, :, 0][..., lat_mask, :][..., lon_mask]
    target_region = target[:, :, 0][..., lat_mask, :][..., lon_mask]
    mask_region = valid_mask[:, None, 0][..., lat_mask, :][..., lon_mask].to(device=pred.device)
    weights = torch.cos(torch.deg2rad(lat[lat_mask]))
    weights = weights / weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
    view_shape = [1] * pred_region.ndim
    view_shape[-2] = weights.numel()
    return pred_region, target_region, mask_region, weights.view(*view_shape)


def _region_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    variable_idx: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按经纬度裁剪区域；variable_idx=None 表示保留全部变量。"""
    H, W = pred.shape[-2:]
    lat = torch.linspace(-89.5, 89.5, H, device=pred.device, dtype=pred.dtype)
    lon = torch.linspace(0.5, 359.5, W, device=pred.device, dtype=pred.dtype)
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    if not bool(lat_mask.any()) or not bool(lon_mask.any()):
        empty = pred.new_empty((0,))
        return empty, empty, empty.to(dtype=torch.bool)

    if variable_idx is None:
        pred_region = pred[..., lat_mask, :][..., lon_mask]
        target_region = target[..., lat_mask, :][..., lon_mask]
        mask_region = valid_mask[:, None][..., lat_mask, :][..., lon_mask].to(device=pred.device)
        return pred_region, target_region, mask_region

    pred_region = pred[:, :, variable_idx : variable_idx + 1][..., lat_mask, :][..., lon_mask]
    target_region = target[:, :, variable_idx : variable_idx + 1][..., lat_mask, :][..., lon_mask]
    mask_region = valid_mask[:, None, variable_idx : variable_idx + 1][..., lat_mask, :][..., lon_mask].to(
        device=pred.device
    )
    return pred_region, target_region, mask_region


def _scale_residual_delta(delta: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
    """按变量月变化尺度缩放 residual loss；未配置时保持原样。"""
    raw = weights.get("residual_delta_std")
    if raw is None:
        return delta
    scale = torch.as_tensor(raw, device=delta.device, dtype=delta.dtype)
    if scale.numel() != delta.shape[2]:
        raise ValueError(f"residual_delta_std must have {delta.shape[2]} values, got {scale.numel()}")
    view_shape = [1] * delta.ndim
    view_shape[2] = delta.shape[2]
    scale = scale.view(*view_shape).clamp_min(torch.finfo(delta.dtype).eps)
    return delta / scale


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
        self.active_rollout_steps = self.rollout_steps
        self.rollout_curriculum = list(self.training_config.get("rollout_curriculum", []))
        self.detach_rollout = bool(self.training_config.get("detach_rollout", True))
        self.lead_weights = self._build_lead_weights()
        self.loss_weights = dict(self.training_config.get("loss_weights", {"field": 1.0}))
        early_stopping_config = dict(self.training_config.get("early_stopping", {}))
        self.early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
        self.early_stopping_monitor = str(early_stopping_config.get("monitor", "val_loss"))
        self.early_stopping_patience = max(1, int(early_stopping_config.get("patience", 10)))
        self.early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.0))
        self.early_stopping_start_epoch = max(1, int(early_stopping_config.get("start_epoch", 1)))
        self.early_stopping_bad_epochs = 0
        self.rollout_selection_config = dict(self.training_config.get("rollout_selection", {}))
        self.rollout_selection_enabled = bool(self.rollout_selection_config.get("enabled", False))
        self.rollout_selection_loader: DataLoader | None = None
        self.best_rollout_skill = float("-inf")
        self.log_interval = int(self.logging_config.get("log_interval", 50))
        self.save_dir = Path(self.logging_config.get("save_dir", "checkpoints"))
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.rollout_selection_loader = self._build_rollout_selection_loader()

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
        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        self.active_rollout_steps = self._rollout_steps_for_epoch(epoch)
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
                loss = self._backward_batch_loss(batch)

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
                    f"optimizer_steps={optimizer_steps} rollout_steps={self.active_rollout_steps} "
                    f"train_loss={avg:.6f}"
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
            "best_val_loss": self.best_val_loss,
            "best_rollout_skill": self.best_rollout_skill,
            "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
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
        epoch = int(checkpoint.get("epoch", 0))
        metrics = dict(checkpoint.get("metrics", {}))
        self.best_val_loss = self._load_best_val_loss(checkpoint, metrics)
        self.best_rollout_skill = self._load_best_rollout_skill(checkpoint, metrics)
        self.early_stopping_bad_epochs = int(checkpoint.get("early_stopping_bad_epochs", 0))
        self.start_epoch = epoch + 1
        return epoch, metrics

    def _load_best_val_loss(self, checkpoint: dict[str, Any], metrics: dict[str, Any]) -> float:
        """从 checkpoint 或同目录 best.pt 恢复历史最优验证 loss。"""
        if "best_val_loss" in checkpoint:
            return float(checkpoint["best_val_loss"])

        best_path = self.save_dir / "best.pt"
        if best_path.exists():
            try:
                best_checkpoint = torch.load(best_path, map_location="cpu")
                best_metrics = dict(best_checkpoint.get("metrics", {}))
                best_val = best_metrics.get("val")
                if isinstance(best_val, dict) and "loss" in best_val:
                    return float(best_val["loss"])
            except Exception as exc:  # pragma: no cover - 只影响恢复 best 计数，不影响训练主体。
                if self.is_main:
                    print(f"warn: failed to read best checkpoint {best_path}: {exc}")

        val_metrics = metrics.get("val")
        if isinstance(val_metrics, dict) and "loss" in val_metrics:
            return float(val_metrics["loss"])
        return float("inf")

    def _load_best_rollout_skill(self, checkpoint: dict[str, Any], metrics: dict[str, Any]) -> float:
        """从 checkpoint 或历史指标中恢复最佳 rollout skill。"""
        if "best_rollout_skill" in checkpoint:
            return float(checkpoint["best_rollout_skill"])

        rollout_metrics = metrics.get("rollout_selection")
        if isinstance(rollout_metrics, dict) and "score" in rollout_metrics:
            return float(rollout_metrics["score"])
        return float("-inf")

    def _update_early_stopping(
        self,
        val_metrics: dict[str, float],
        rollout_metrics: dict[str, Any],
        epoch: int,
    ) -> tuple[bool, bool, bool]:
        """更新 best/early stopping 状态，返回 loss best、skill best、是否停止。"""
        loss_improved = False
        skill_improved = False
        monitored_improved = False

        if val_metrics and "loss" in val_metrics:
            current_loss = float(val_metrics["loss"])
            loss_improved = current_loss < self.best_val_loss
            loss_improved_for_stop = current_loss < self.best_val_loss - self.early_stopping_min_delta
            if loss_improved:
                self.best_val_loss = current_loss
        else:
            loss_improved_for_stop = False

        if rollout_metrics and "score" in rollout_metrics:
            current_skill = float(rollout_metrics["score"])
            skill_improved = current_skill > self.best_rollout_skill
            skill_improved_for_stop = current_skill > self.best_rollout_skill + self.early_stopping_min_delta
            if skill_improved:
                self.best_rollout_skill = current_skill
        else:
            skill_improved_for_stop = False

        monitor = self.early_stopping_monitor
        if monitor == "val_loss":
            monitored_improved = loss_improved_for_stop
            has_metric = bool(val_metrics and "loss" in val_metrics)
        elif monitor == "rollout_skill":
            monitored_improved = skill_improved_for_stop
            has_metric = bool(rollout_metrics and "score" in rollout_metrics)
        else:
            raise ValueError("training.early_stopping.monitor must be 'val_loss' or 'rollout_skill'")

        if not has_metric:
            return loss_improved, skill_improved, False

        if epoch < self.early_stopping_start_epoch:
            return loss_improved, skill_improved, False

        if monitored_improved:
            self.early_stopping_bad_epochs = 0
        else:
            self.early_stopping_bad_epochs += 1

        should_stop = self.early_stopping_enabled and self.early_stopping_bad_epochs >= self.early_stopping_patience
        return loss_improved, skill_improved, should_stop

    def train(self) -> None:
        """完整训练循环。"""
        for epoch in range(self.start_epoch, self.epochs + 1):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            rollout_metrics = self.evaluate_rollout_selection(epoch)

            if self.is_main:
                message = f"epoch={epoch} train_loss={train_metrics['loss']:.6f}"
                if val_metrics:
                    message += f" val_loss={val_metrics['loss']:.6f}"
                if rollout_metrics:
                    acc_text = " ".join(
                        f"acc@{lead}={rollout_metrics[f'acc@{lead}']:.4f}"
                        for lead in rollout_metrics["leads"]
                    )
                    message += f" rollout_skill={rollout_metrics['score']:.4f} {acc_text}"
                print(message)

            is_best_loss, is_best_skill, should_stop = self._update_early_stopping(val_metrics, rollout_metrics, epoch)
            latest_path = self.save_dir / "latest.pt"
            all_metrics = {"train": train_metrics, "val": val_metrics, "rollout_selection": rollout_metrics}
            self.save_checkpoint(latest_path, epoch, all_metrics)

            if is_best_loss:
                self.save_checkpoint(self.save_dir / "best_loss.pt", epoch, all_metrics)
                # 兼容旧脚本默认读取 best.pt 的习惯；新实验请优先使用 best_loss/best_skill。
                self.save_checkpoint(self.save_dir / "best.pt", epoch, all_metrics)
            if is_best_skill:
                self.save_checkpoint(self.save_dir / "best_skill.pt", epoch, all_metrics)
            if self.is_main and self.early_stopping_enabled:
                best_text = (
                    f"best_val_loss={self.best_val_loss:.6f}"
                    if self.early_stopping_monitor == "val_loss"
                    else f"best_rollout_skill={self.best_rollout_skill:.6f}"
                )
                print(
                    "early_stopping "
                    f"bad_epochs={self.early_stopping_bad_epochs}/{self.early_stopping_patience} "
                    f"monitor={self.early_stopping_monitor} "
                    f"{best_text} "
                    f"min_delta={self.early_stopping_min_delta:.6g} "
                    f"start_epoch={self.early_stopping_start_epoch}"
                )

            if should_stop:
                if self.is_main:
                    print(f"early_stopping stop at epoch={epoch}")
                break

    def _build_rollout_selection_loader(self) -> DataLoader | None:
        """为 rank 0 构建 source-balanced 的 rollout skill 验证子集。"""
        if not self.rollout_selection_enabled or self.val_loader is None:
            return None

        dataset = self.val_loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        required_attrs = ("sample_indices", "source_names", "source_payloads", "data_config")
        if not all(hasattr(dataset, name) for name in required_attrs):
            raise TypeError("rollout_selection requires a WalkerDataset-like validation dataset")

        max_lead = int(self.rollout_selection_config.get("max_lead", 18))
        max_per_source = int(self.rollout_selection_config.get("max_samples_per_source", 24))
        positions = self._rollout_selection_positions(dataset, max_lead, max_per_source)
        if not positions:
            raise ValueError("No validation samples can satisfy rollout_selection.max_lead")

        batch_size = int(self.rollout_selection_config.get("batch_size", 2))
        num_workers = int(self.rollout_selection_config.get("num_workers", 0))
        loader = DataLoader(
            Subset(dataset, positions),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        print(
            f"rollout_selection samples={len(positions)} "
            f"max_lead={max_lead} batch_size={batch_size}",
            flush=True,
        )
        return loader

    def _rollout_selection_positions(self, dataset: Any, max_lead: int, max_per_source: int) -> list[int]:
        """按 source 均匀抽取可完整验证到 max_lead 的样本位置。"""
        split_years = dataset.data_config.get(f"{dataset.split}_years")
        end_year = int(split_years[1]) if split_years is not None else None
        by_source: dict[int, list[int]] = {idx: [] for idx in range(len(dataset.source_names))}

        for pos, sample_index in enumerate(dataset.sample_indices):
            source_idx = int(sample_index[0])
            target_t = int(sample_index[1])
            payload = dataset.source_payloads[source_idx]
            final_t = target_t + max_lead - 1
            if final_t >= len(payload["years"]):
                continue
            if end_year is not None and int(payload["years"][final_t]) > end_year:
                continue
            by_source[source_idx].append(pos)

        positions: list[int] = []
        for source_idx in sorted(by_source):
            candidates = by_source[source_idx]
            if max_per_source > 0 and len(candidates) > max_per_source:
                pick = np.linspace(0, len(candidates) - 1, max_per_source, dtype=np.int64)
                candidates = [candidates[int(i)] for i in pick]
            positions.extend(candidates)
        return positions

    @torch.no_grad()
    def evaluate_rollout_selection(self, epoch: int) -> dict[str, Any]:
        """用验证集自由滚动 skill 选择 checkpoint。"""
        if not self.is_main or self.rollout_selection_loader is None:
            return {}

        interval = max(1, int(self.rollout_selection_config.get("interval_epochs", 1)))
        if epoch % interval != 0:
            return {}

        dataset = self.rollout_selection_loader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

        max_lead = int(self.rollout_selection_config.get("max_lead", 18))
        leads = [int(x) for x in self.rollout_selection_config.get("leads", [6, 9, 12, 18])]
        mode = str(self.rollout_selection_config.get("mode", "three_month_mean"))
        score_name = str(self.rollout_selection_config.get("score", "mean_acc"))
        trained_rollout_steps = max(1, int(getattr(self, "active_rollout_steps", self.rollout_steps)))

        model_series, persistence_series, target_series = self._collect_rollout_nino_series(
            dataset,
            max_lead=max_lead,
            trained_rollout_steps=trained_rollout_steps,
        )
        metrics = self._rollout_skill_metrics(model_series, persistence_series, target_series, leads, mode)
        acc_values = [metrics[lead]["corr"] for lead in leads if lead in metrics]
        if not acc_values:
            raise ValueError(f"No rollout skill metrics were computed for leads={leads}, mode={mode}")
        if score_name != "mean_acc":
            raise ValueError("training.rollout_selection.score currently supports only 'mean_acc'")

        result: dict[str, Any] = {
            "score": float(sum(acc_values) / len(acc_values)),
            "mode": mode,
            "score_name": score_name,
            "leads": leads,
            "max_lead": max_lead,
            "num_samples": int(next(iter(target_series.values())).numel()),
        }
        for lead in leads:
            row = metrics[lead]
            result[f"acc@{lead}"] = float(row["corr"])
            result[f"rmse@{lead}"] = float(row["rmse"])
            result[f"persistence_acc@{lead}"] = float(row["persistence_corr"])
            result[f"persistence_rmse@{lead}"] = float(row["persistence_rmse"])
        return result

    def _collect_rollout_nino_series(
        self,
        dataset: Any,
        max_lead: int,
        trained_rollout_steps: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """收集 1..max_lead 的 Niño3.4 anomaly 序列。"""
        self.model.eval()
        model_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}
        persistence_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}
        target_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}
        climatology = self._compute_source_nino34_climatology(dataset).to(device=self.device, dtype=torch.float32)
        lat = torch.as_tensor(dataset.lat, dtype=torch.float32, device=self.device)
        lon = torch.as_tensor(dataset.lon, dtype=torch.float32, device=self.device)

        for batch in self.rollout_selection_loader:
            window = batch["x"].to(self.device)
            persistence_phys = dataset.denormalize(window[:, -1:].contiguous())
            source_index = batch.get("source_index")
            if source_index is None:
                source_index = torch.zeros(window.shape[0], dtype=torch.long)
            source_index = source_index.to(device=self.device, dtype=torch.long)
            base_target_t = batch["time_index"].to(device=self.device, dtype=torch.long)

            for step in range(1, max_lead + 1):
                target_t = base_target_t + step - 1
                target_month = self._target_months(dataset, source_index, target_t)
                rollout_step = torch.full(
                    (window.shape[0],),
                    min(step - 1, trained_rollout_steps - 1),
                    dtype=torch.long,
                    device=self.device,
                )
                pred_norm = self.model(window, target_month, rollout_step=rollout_step)
                pred_phys = dataset.denormalize(pred_norm)
                target_phys = self._target_phys(dataset, source_index, target_t)

                clim = climatology[source_index, target_month].detach().cpu()
                model_raw = compute_nino34(pred_phys[:, 0, 0], lat, lon).detach().cpu()
                persistence_raw = compute_nino34(persistence_phys[:, 0, 0], lat, lon).detach().cpu()
                target_raw = compute_nino34(target_phys[:, 0, 0], lat, lon).detach().cpu()
                model_nino[step].append(model_raw - clim)
                persistence_nino[step].append(persistence_raw - clim)
                target_nino[step].append(target_raw - clim)

                next_frame = self._mask_next_frame(pred_norm, batch["valid_mask"].to(self.device))
                window = torch.cat([window[:, 1:], next_frame], dim=1)

        return (
            {lead: torch.cat(values) for lead, values in model_nino.items()},
            {lead: torch.cat(values) for lead, values in persistence_nino.items()},
            {lead: torch.cat(values) for lead, values in target_nino.items()},
        )

    def _rollout_skill_metrics(
        self,
        model: dict[int, torch.Tensor],
        persistence: dict[int, torch.Tensor],
        target: dict[int, torch.Tensor],
        leads: list[int],
        mode: str,
    ) -> dict[int, dict[str, float]]:
        """计算 monthly 或 3-month mean 的 Niño3.4 anomaly skill。"""
        rows: dict[int, dict[str, float]] = {}
        for lead in leads:
            if mode == "monthly":
                model_values = model[lead]
                persistence_values = persistence[lead]
                target_values = target[lead]
            elif mode == "three_month_mean":
                if lead < 3:
                    continue
                model_values = (model[lead - 2] + model[lead - 1] + model[lead]) / 3.0
                persistence_values = (persistence[lead - 2] + persistence[lead - 1] + persistence[lead]) / 3.0
                target_values = (target[lead - 2] + target[lead - 1] + target[lead]) / 3.0
            else:
                raise ValueError("training.rollout_selection.mode must be 'monthly' or 'three_month_mean'")

            rows[lead] = {
                "corr": _tensor_corr(model_values, target_values),
                "rmse": _tensor_rmse(model_values, target_values),
                "persistence_corr": _tensor_corr(persistence_values, target_values),
                "persistence_rmse": _tensor_rmse(persistence_values, target_values),
            }
        return rows

    @staticmethod
    def _compute_source_nino34_climatology(dataset: Any) -> torch.Tensor:
        """用训练年份为每个 source 计算 Niño3.4 月气候态。"""
        train_start, train_end = dataset.data_config["train_years"]
        climatology = np.zeros((len(dataset.source_payloads), 13), dtype=np.float32)
        for source_idx, payload in enumerate(dataset.source_payloads):
            years = payload["years"]
            months = payload["months"]
            train_mask = (years >= int(train_start)) & (years <= int(train_end))
            tos = np.asarray(payload["data"][:, 0])
            nino = _compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
            for month in range(1, 13):
                month_mask = train_mask & (months == month)
                climatology[source_idx, month] = float(np.nanmean(nino[month_mask]))
        return torch.from_numpy(climatology)

    @staticmethod
    def _target_months(dataset: Any, source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """按 source/target_t 读取目标月份。"""
        source_np = source_indices.detach().cpu().numpy()
        target_np = target_indices.detach().cpu().numpy()
        months = [
            int(dataset.source_payloads[int(source_idx)]["months"][int(target_t)])
            for source_idx, target_t in zip(source_np, target_np)
        ]
        return torch.as_tensor(months, dtype=torch.long, device=target_indices.device)

    def _target_phys(self, dataset: Any, source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """读取未来目标场的物理量版本。"""
        source_np = source_indices.detach().cpu().numpy()
        target_np = target_indices.detach().cpu().numpy()
        raw = np.stack(
            [
                np.asarray(dataset.source_payloads[int(source_idx)]["data"][int(target_t)], dtype=np.float32)
                for source_idx, target_t in zip(source_np, target_np)
            ],
            axis=0,
        )
        return torch.from_numpy(raw).float().to(self.device)[:, None]

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
        steps = min(self.active_rollout_steps, available_steps)
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
                distributed=self.is_distributed,
                rank=self.rank,
                world_size=self.world_size,
            )

            if step_idx + 1 < steps:
                next_frame = pred.detach() if self.detach_rollout else pred
                next_frame = self._mask_next_frame(next_frame, valid_mask)
                window = torch.cat([window[:, 1:], next_frame], dim=1)

        return total

    def _backward_batch_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """逐 lead 反传一个 batch，降低多步 rollout 的显存峰值。

        旧写法会先把所有 lead 的 loss 累加成一个总 loss，再一次性 backward；
        这样会同时保留多个 lead 的计算图。当前默认 ``detach_rollout=True``，
        每个 lead 之间不需要跨步反传，因此可以每算完一个 lead 就立即 backward。
        """
        if not self.detach_rollout:
            with autocast() if self.amp_enabled else nullcontext():
                loss = self._compute_batch_loss(batch)
                scaled_loss = loss / self.grad_accum_steps
            self._backward_scaled_loss(scaled_loss)
            return loss.detach()

        window = batch["x"]
        valid_mask = batch["valid_mask"]
        targets = batch.get("y_rollout", batch["y"])
        if targets.ndim != 5:
            raise ValueError(f"targets must be (B, K, 4, H, W), got {targets.shape}")

        available_steps = targets.shape[1]
        steps = min(self.active_rollout_steps, available_steps)
        target_months = batch.get("target_months")
        lead_weights = self.lead_weights[:steps].to(device=window.device, dtype=window.dtype)
        total_for_logging = window.new_zeros(())

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

            with autocast() if self.amp_enabled else nullcontext():
                pred = self.model(window, target_month, rollout_step=rollout_step)
                lead_loss = lead_weights[step_idx] * forecast_loss(
                    pred,
                    target,
                    x_last,
                    valid_mask,
                    self.loss_weights,
                    distributed=self.is_distributed,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                scaled_loss = lead_loss / self.grad_accum_steps

            self._backward_scaled_loss(scaled_loss)
            total_for_logging = total_for_logging + lead_loss.detach()

            if step_idx + 1 < steps:
                next_frame = self._mask_next_frame(pred.detach(), valid_mask)
                window = torch.cat([window[:, 1:], next_frame], dim=1)

        return total_for_logging

    def _backward_scaled_loss(self, scaled_loss: torch.Tensor) -> None:
        """根据 AMP 设置反传已经按 grad_accum_steps 缩放过的 loss。"""
        if self.amp_enabled:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

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

    def _rollout_steps_for_epoch(self, epoch: int) -> int:
        """按课程表决定当前 epoch 实际训练几步 rollout。

        配置示例：
            rollout_curriculum:
              - until_epoch: 21
                steps: 12
              - until_epoch: 25
                steps: 15
              - steps: 18
        """
        for item in self.rollout_curriculum:
            until_epoch = item.get("until_epoch")
            if until_epoch is None or epoch <= int(until_epoch):
                return max(1, min(self.rollout_steps, int(item["steps"])))
        return self.rollout_steps

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
