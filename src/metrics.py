"""WalkerNet 评价指标。

指标：
- 通用场预测指标：RMSE、ACC
- ENSO 指标：Niño3.4 区域平均、Niño3.4 相关系数/RMSE

约定：
- 指标函数接收 torch.Tensor。
- 需要物理量单位的指标，应先用 dataset.denormalize() 还原。
- 经度使用 0-360，因此 Niño3.4 的 170W-120W 等价于 190E-240E。
"""

from __future__ import annotations

import torch


def _as_tensor(values: torch.Tensor | list[float] | tuple[float, ...], device: torch.device) -> torch.Tensor:
    """把 lat/lon 这类输入转换成 torch.Tensor。"""
    if isinstance(values, torch.Tensor):
        return values.to(device=device)
    return torch.tensor(values, device=device, dtype=torch.float32)


def compute_nino34(sst: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """从 SST 场计算 Niño3.4 指数。

    Niño3.4 区域：
        5S-5N, 170W-120W

    当前经度为 0-360，因此经度范围写作：
        190E-240E

    Args:
        sst: SST 张量，支持形状 ``(..., H, W)``。
        lat: 纬度坐标，shape = ``(H,)``。
        lon: 经度坐标，shape = ``(W,)``。

    Returns:
        去掉 H/W 后的区域平均结果，shape = ``sst.shape[:-2]``。
    """
    if sst.ndim < 2:
        raise ValueError(f"sst must have at least 2 dims (..., H, W), got shape={tuple(sst.shape)}")

    device = sst.device
    dtype = sst.dtype
    lat_t = _as_tensor(lat, device=device).to(dtype=dtype)
    lon_t = _as_tensor(lon, device=device).to(dtype=dtype)

    if sst.shape[-2] != lat_t.numel() or sst.shape[-1] != lon_t.numel():
        raise ValueError(
            f"sst spatial shape {tuple(sst.shape[-2:])} does not match "
            f"lat/lon lengths {(lat_t.numel(), lon_t.numel())}"
        )

    lat_mask = (lat_t >= -5.0) & (lat_t <= 5.0)
    lon_mask = (lon_t >= 190.0) & (lon_t <= 240.0)
    if not bool(lat_mask.any()) or not bool(lon_mask.any()):
        raise ValueError("Niño3.4 region is empty for the provided lat/lon coordinates")

    region = sst[..., lat_mask, :][..., lon_mask]

    # 面积权重近似用 cos(lat)。在 5S-5N 差异很小，但保留这个写法更规范。
    weights = torch.cos(torch.deg2rad(lat_t[lat_mask])).to(dtype=dtype)
    weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)
    view_shape = [1] * region.ndim
    view_shape[-2] = weights.numel()
    weighted = region * weights.view(*view_shape)
    return weighted.nanmean(dim=-1).sum(dim=-1)


def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """计算 RMSE。

    Args:
        pred: 预测张量。
        target: 目标张量，shape 应与 pred 一致。
        mask: 可选 bool mask，可广播到 pred/target。

    Returns:
        标量 tensor。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    squared = (pred - target) ** 2
    if mask is not None:
        mask = mask.to(device=pred.device, dtype=torch.bool)
        squared = squared.masked_select(mask)
        if squared.numel() == 0:
            return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
        return torch.sqrt(squared.mean())
    return torch.sqrt(torch.nanmean(squared))


def anomaly_correlation_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 anomaly correlation coefficient (ACC)。

    这里默认输入已经是 anomaly；如果传入原始场，本函数会先减去整体均值，
    得到的是整体相关系数，不是按气候态 anomaly 的正式 ENSO ACC。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    if mask is not None:
        mask = mask.to(device=pred.device, dtype=torch.bool)
        pred = pred.masked_select(mask)
        target = target.masked_select(mask)
    else:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        finite = torch.isfinite(pred) & torch.isfinite(target)
        pred = pred[finite]
        target = target[finite]

    if pred.numel() == 0:
        return torch.tensor(float("nan"), device=target.device, dtype=target.dtype)

    pred_anom = pred - pred.mean()
    target_anom = target - target.mean()
    numerator = (pred_anom * target_anom).sum()
    denominator = torch.sqrt((pred_anom**2).sum() * (target_anom**2).sum()).clamp_min(eps)
    return numerator / denominator


def nino34_correlation(pred_sst: torch.Tensor, target_sst: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """计算预测和观测 Niño3.4 指数的相关系数。"""
    pred_index = compute_nino34(pred_sst, lat, lon).reshape(-1)
    target_index = compute_nino34(target_sst, lat, lon).reshape(-1)
    return anomaly_correlation_coefficient(pred_index, target_index)


def nino34_rmse(pred_sst: torch.Tensor, target_sst: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """计算预测和观测 Niño3.4 指数的 RMSE。"""
    pred_index = compute_nino34(pred_sst, lat, lon)
    target_index = compute_nino34(target_sst, lat, lon)
    return rmse(pred_index, target_index)
