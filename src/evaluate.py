"""WalkerNet 单步预测评测入口。

默认评测 test split 上的 best checkpoint，输出：
- 每个变量的 normalized / physical RMSE、MAE、相关系数
- tos 的 Niño3.4 指数 RMSE 和相关系数
- JSON、CSV 和可选 PNG 图
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import WalkerDataset
from .interfaces import VARIABLES
from .metrics import compute_nino34
from .model import WalkerNet
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet Evaluation")
    parser.add_argument("--config", type=str, default="configs/server_3090.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    return parser.parse_args()


def _empty_stats() -> dict[str, torch.Tensor]:
    """创建逐变量累计统计量。"""
    n = len(VARIABLES)
    return {
        "sse": torch.zeros(n, dtype=torch.float64),
        "sae": torch.zeros(n, dtype=torch.float64),
        "sum_pred": torch.zeros(n, dtype=torch.float64),
        "sum_target": torch.zeros(n, dtype=torch.float64),
        "sum_pred2": torch.zeros(n, dtype=torch.float64),
        "sum_target2": torch.zeros(n, dtype=torch.float64),
        "sum_cross": torch.zeros(n, dtype=torch.float64),
        "count": torch.zeros(n, dtype=torch.float64),
    }


def _update_stats(
    stats: dict[str, torch.Tensor],
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """按变量累计 masked RMSE/MAE/correlation 需要的统计量。"""
    pred_work = pred.detach().double()
    target_work = target.detach().double()
    mask_work = valid_mask.detach().to(device=pred_work.device, dtype=torch.bool)

    for idx in range(len(VARIABLES)):
        mask = mask_work[:, :, idx]
        p = pred_work[:, :, idx][mask]
        y = target_work[:, :, idx][mask]
        if p.numel() == 0:
            continue

        diff = p - y
        stats["sse"][idx] += (diff * diff).sum().cpu()
        stats["sae"][idx] += diff.abs().sum().cpu()
        stats["sum_pred"][idx] += p.sum().cpu()
        stats["sum_target"][idx] += y.sum().cpu()
        stats["sum_pred2"][idx] += (p * p).sum().cpu()
        stats["sum_target2"][idx] += (y * y).sum().cpu()
        stats["sum_cross"][idx] += (p * y).sum().cpu()
        stats["count"][idx] += p.numel()


def _finalize_stats(stats: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """把累计统计量转成每个变量的标量指标。"""
    result: dict[str, dict[str, float]] = {}
    eps = 1e-12

    for idx, name in enumerate(VARIABLES):
        count = stats["count"][idx].clamp_min(1.0)
        rmse = torch.sqrt(stats["sse"][idx] / count)
        mae = stats["sae"][idx] / count

        sum_p = stats["sum_pred"][idx]
        sum_y = stats["sum_target"][idx]
        numerator = stats["sum_cross"][idx] - sum_p * sum_y / count
        pred_var = stats["sum_pred2"][idx] - sum_p * sum_p / count
        target_var = stats["sum_target2"][idx] - sum_y * sum_y / count
        corr = numerator / torch.sqrt((pred_var * target_var).clamp_min(eps))

        result[name] = {
            "rmse": float(rmse.item()),
            "mae": float(mae.item()),
            "corr": float(corr.item()),
            "count": int(stats["count"][idx].item()),
        }
    return result


def _tensor_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算一维张量相关系数。"""
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return float(((x * y).sum() / denom).item())


def _write_variable_csv(path: Path, systems: dict[str, dict[str, Any]]) -> None:
    """写出逐变量指标表。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "variable", "space", "rmse", "mae", "corr", "count"])
        for system_name, system_metrics in systems.items():
            for variable in VARIABLES:
                for space in ("normalized", "physical"):
                    row = system_metrics[space][variable]
                    writer.writerow([
                        system_name,
                        variable,
                        space,
                        row["rmse"],
                        row["mae"],
                        row["corr"],
                        row["count"],
                    ])


def _write_nino_csv(
    path: Path,
    target_month: torch.Tensor,
    model_raw: torch.Tensor,
    persistence_raw: torch.Tensor,
    target_raw: torch.Tensor,
    model_anom: torch.Tensor,
    persistence_anom: torch.Tensor,
    target_anom: torch.Tensor,
) -> None:
    """写出 Niño3.4 时间序列。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "target_month",
            "model_raw",
            "persistence_raw",
            "target_raw",
            "model_anomaly",
            "persistence_anomaly",
            "target_anomaly",
        ])
        values = zip(
            target_month.tolist(),
            model_raw.tolist(),
            persistence_raw.tolist(),
            target_raw.tolist(),
            model_anom.tolist(),
            persistence_anom.tolist(),
            target_anom.tolist(),
        )
        for idx, row in enumerate(values):
            writer.writerow([idx, *row])


def _maybe_plot_nino(path: Path, model: torch.Tensor, persistence: torch.Tensor, target: torch.Tensor) -> bool:
    """如果 matplotlib 可用，则画 Niño3.4 时间序列。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = list(range(len(model)))
    plt.figure(figsize=(12, 4))
    plt.plot(x, target.numpy(), label="target", linewidth=1.8)
    plt.plot(x, model.numpy(), label="WalkerNet", linewidth=1.8)
    plt.plot(x, persistence.numpy(), label="persistence", linewidth=1.4, alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("test sample")
    plt.ylabel("tos Niño3.4 anomaly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _compute_nino34_numpy(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """从 ``(T, H, W)`` 的 tos 数据计算 Niño3.4 区域平均。"""
    lat_mask = (lat >= -5.0) & (lat <= 5.0)
    lon_mask = (lon >= 190.0) & (lon <= 240.0)
    region = data[:, lat_mask, :][:, :, lon_mask]
    lon_mean = np.nanmean(region, axis=2)
    weights = np.cos(np.deg2rad(lat[lat_mask])).astype(np.float64)
    weights = weights / weights.sum()
    return np.nansum(lon_mean * weights[None, :], axis=1).astype(np.float32)


def _compute_nino34_climatology(dataset: WalkerDataset) -> torch.Tensor:
    """用训练年份计算 Niño3.4 月气候态，返回 shape=(13,)；索引 1-12 有效。"""
    data_config = dataset.data_config
    train_start, train_end = data_config["train_years"]
    train_mask = (dataset.years >= int(train_start)) & (dataset.years <= int(train_end))

    tos = np.asarray(dataset.data[:, 0])
    nino = _compute_nino34_numpy(tos, np.asarray(dataset.lat), np.asarray(dataset.lon))
    climatology = np.zeros(13, dtype=np.float32)
    for month in range(1, 13):
        month_mask = train_mask & (dataset.months == month)
        climatology[month] = float(np.nanmean(nino[month_mask]))
    return torch.from_numpy(climatology)


def _nino_summary(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Niño3.4 一维序列指标。"""
    diff = pred - target
    return {
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "mae": float(torch.mean(diff.abs()).item()),
        "corr": _tensor_corr(pred, target),
        "num_samples": int(pred.numel()),
    }


def _build_comparison(model: dict[str, Any], persistence: dict[str, Any]) -> dict[str, Any]:
    """计算 WalkerNet 相对 persistence 的 RMSE 改善幅度。"""
    variable_improvement: dict[str, dict[str, float]] = {}
    for variable in VARIABLES:
        model_rmse = float(model["physical"][variable]["rmse"])
        base_rmse = float(persistence["physical"][variable]["rmse"])
        variable_improvement[variable] = {
            "model_physical_rmse": model_rmse,
            "persistence_physical_rmse": base_rmse,
            "rmse_improvement_ratio": (base_rmse - model_rmse) / base_rmse if base_rmse > 0 else float("nan"),
        }

    nino_model = float(model["nino34_anomaly"]["rmse"])
    nino_base = float(persistence["nino34_anomaly"]["rmse"])
    return {
        "physical_rmse_by_variable": variable_improvement,
        "nino34_anomaly": {
            "model_rmse": nino_model,
            "persistence_rmse": nino_base,
            "rmse_improvement_ratio": (nino_base - nino_model) / nino_base if nino_base > 0 else float("nan"),
            "model_corr": float(model["nino34_anomaly"]["corr"]),
            "persistence_corr": float(persistence["nino34_anomaly"]["corr"]),
        },
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: WalkerDataset,
    device: torch.device,
) -> dict[str, Any]:
    """执行完整评测。"""
    model.eval()

    model_norm_stats = _empty_stats()
    model_phys_stats = _empty_stats()
    persistence_norm_stats = _empty_stats()
    persistence_phys_stats = _empty_stats()

    target_months: list[torch.Tensor] = []
    model_nino_raw: list[torch.Tensor] = []
    persistence_nino_raw: list[torch.Tensor] = []
    target_nino_raw: list[torch.Tensor] = []

    lat = torch.as_tensor(dataset.lat, dtype=torch.float32, device=device)
    lon = torch.as_tensor(dataset.lon, dtype=torch.float32, device=device)
    climatology = _compute_nino34_climatology(dataset).to(device=device, dtype=torch.float32)

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        target_month = batch["target_month"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred = model(x, target_month)
        persistence = x[:, -1:].contiguous()

        _update_stats(model_norm_stats, pred, y, valid_mask[:, None])
        _update_stats(persistence_norm_stats, persistence, y, valid_mask[:, None])

        pred_phys = dataset.denormalize(pred)
        persistence_phys = dataset.denormalize(persistence)
        y_phys = dataset.denormalize(y)
        _update_stats(model_phys_stats, pred_phys, y_phys, valid_mask[:, None])
        _update_stats(persistence_phys_stats, persistence_phys, y_phys, valid_mask[:, None])

        # tos 是变量 0；Niño3.4 指标在物理量空间中计算。
        model_nino = compute_nino34(pred_phys[:, 0, 0], lat, lon).detach().cpu()
        persistence_nino = compute_nino34(persistence_phys[:, 0, 0], lat, lon).detach().cpu()
        target_nino = compute_nino34(y_phys[:, 0, 0], lat, lon).detach().cpu()

        target_months.append(target_month.detach().cpu())
        model_nino_raw.append(model_nino)
        persistence_nino_raw.append(persistence_nino)
        target_nino_raw.append(target_nino)

        if batch_idx % 20 == 0:
            print(f"evaluated batches={batch_idx}/{len(loader)}", flush=True)

    target_month_all = torch.cat(target_months).long()
    model_nino_raw_all = torch.cat(model_nino_raw)
    persistence_nino_raw_all = torch.cat(persistence_nino_raw)
    target_nino_raw_all = torch.cat(target_nino_raw)

    clim = climatology.detach().cpu()[target_month_all]
    model_nino_anom = model_nino_raw_all - clim
    persistence_nino_anom = persistence_nino_raw_all - clim
    target_nino_anom = target_nino_raw_all - clim

    return {
        "model": {
            "normalized": _finalize_stats(model_norm_stats),
            "physical": _finalize_stats(model_phys_stats),
            "nino34_raw": _nino_summary(model_nino_raw_all, target_nino_raw_all),
            "nino34_anomaly": _nino_summary(model_nino_anom, target_nino_anom),
        },
        "persistence": {
            "normalized": _finalize_stats(persistence_norm_stats),
            "physical": _finalize_stats(persistence_phys_stats),
            "nino34_raw": _nino_summary(persistence_nino_raw_all, target_nino_raw_all),
            "nino34_anomaly": _nino_summary(persistence_nino_anom, target_nino_anom),
        },
        "nino34_series": {
            "target_month": target_month_all,
            "model_raw": model_nino_raw_all,
            "persistence_raw": persistence_nino_raw_all,
            "target_raw": target_nino_raw_all,
            "model_anomaly": model_nino_anom,
            "persistence_anomaly": persistence_nino_anom,
            "target_anomaly": target_nino_anom,
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = WalkerNet(config).to(device)
    model.load_state_dict(checkpoint["model"])

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"split={args.split} samples={len(dataset)} batches={len(loader)} device={device}")

    result = evaluate(model, loader, dataset, device)
    series = result.pop("nino34_series")
    result["comparison"] = _build_comparison(result["model"], result["persistence"])

    json_payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "num_samples": len(dataset),
        "batch_size": args.batch_size,
        **result,
    }

    metrics_json = output_dir / f"{args.split}_metrics.json"
    metrics_csv = output_dir / f"{args.split}_variable_metrics.csv"
    nino_csv = output_dir / f"{args.split}_nino34_timeseries.csv"
    nino_png = output_dir / f"{args.split}_nino34_anomaly_timeseries.png"

    metrics_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_variable_csv(
        metrics_csv,
        {
            "WalkerNet": result["model"],
            "Persistence": result["persistence"],
        },
    )
    _write_nino_csv(
        nino_csv,
        series["target_month"],
        series["model_raw"],
        series["persistence_raw"],
        series["target_raw"],
        series["model_anomaly"],
        series["persistence_anomaly"],
        series["target_anomaly"],
    )
    plotted = _maybe_plot_nino(
        nino_png,
        series["model_anomaly"],
        series["persistence_anomaly"],
        series["target_anomaly"],
    )

    print(json.dumps(json_payload, indent=2, ensure_ascii=False))
    print(f"wrote {metrics_json}")
    print(f"wrote {metrics_csv}")
    print(f"wrote {nino_csv}")
    if plotted:
        print(f"wrote {nino_png}")
    else:
        print("matplotlib unavailable; skipped nino34 png")


if __name__ == "__main__":
    main()
