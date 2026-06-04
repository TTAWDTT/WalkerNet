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
    pred_cpu = pred.detach().cpu().double()
    target_cpu = target.detach().cpu().double()
    mask_cpu = valid_mask.detach().cpu().bool()

    for idx in range(len(VARIABLES)):
        mask = mask_cpu[:, :, idx]
        p = pred_cpu[:, :, idx][mask]
        y = target_cpu[:, :, idx][mask]
        if p.numel() == 0:
            continue

        diff = p - y
        stats["sse"][idx] += (diff * diff).sum()
        stats["sae"][idx] += diff.abs().sum()
        stats["sum_pred"][idx] += p.sum()
        stats["sum_target"][idx] += y.sum()
        stats["sum_pred2"][idx] += (p * p).sum()
        stats["sum_target2"][idx] += (y * y).sum()
        stats["sum_cross"][idx] += (p * y).sum()
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


def _write_variable_csv(path: Path, normalized: dict[str, Any], physical: dict[str, Any]) -> None:
    """写出逐变量指标表。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "space", "rmse", "mae", "corr", "count"])
        for variable in VARIABLES:
            for space, metrics in (("normalized", normalized), ("physical", physical)):
                row = metrics[variable]
                writer.writerow([variable, space, row["rmse"], row["mae"], row["corr"], row["count"]])


def _write_nino_csv(path: Path, pred: torch.Tensor, target: torch.Tensor) -> None:
    """写出 Niño3.4 时间序列。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "pred_nino34", "target_nino34", "error"])
        for idx, (p, y) in enumerate(zip(pred.tolist(), target.tolist())):
            writer.writerow([idx, p, y, p - y])


def _maybe_plot_nino(path: Path, pred: torch.Tensor, target: torch.Tensor) -> bool:
    """如果 matplotlib 可用，则画 Niño3.4 时间序列。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = list(range(len(pred)))
    plt.figure(figsize=(12, 4))
    plt.plot(x, target.numpy(), label="target", linewidth=1.8)
    plt.plot(x, pred.numpy(), label="pred", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("test sample")
    plt.ylabel("tos Niño3.4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: WalkerDataset,
    device: torch.device,
) -> dict[str, Any]:
    """执行完整评测。"""
    model.eval()

    norm_stats = _empty_stats()
    phys_stats = _empty_stats()
    nino_pred: list[torch.Tensor] = []
    nino_target: list[torch.Tensor] = []

    lat = torch.as_tensor(dataset.lat, dtype=torch.float32, device=device)
    lon = torch.as_tensor(dataset.lon, dtype=torch.float32, device=device)

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        target_month = batch["target_month"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred = model(x, target_month)
        _update_stats(norm_stats, pred, y, valid_mask[:, None])

        pred_phys = dataset.denormalize(pred)
        y_phys = dataset.denormalize(y)
        _update_stats(phys_stats, pred_phys, y_phys, valid_mask[:, None])

        # tos 是变量 0；Niño3.4 指标在物理量空间中计算。
        pred_nino = compute_nino34(pred_phys[:, 0, 0], lat, lon).detach().cpu()
        target_nino = compute_nino34(y_phys[:, 0, 0], lat, lon).detach().cpu()
        nino_pred.append(pred_nino)
        nino_target.append(target_nino)

        if batch_idx % 20 == 0:
            print(f"evaluated batches={batch_idx}/{len(loader)}", flush=True)

    pred_nino_all = torch.cat(nino_pred)
    target_nino_all = torch.cat(nino_target)
    nino_rmse = torch.sqrt(torch.mean((pred_nino_all - target_nino_all) ** 2))
    nino_mae = torch.mean((pred_nino_all - target_nino_all).abs())

    return {
        "normalized": _finalize_stats(norm_stats),
        "physical": _finalize_stats(phys_stats),
        "nino34": {
            "rmse": float(nino_rmse.item()),
            "mae": float(nino_mae.item()),
            "corr": _tensor_corr(pred_nino_all, target_nino_all),
            "num_samples": int(pred_nino_all.numel()),
        },
        "nino34_series": {
            "pred": pred_nino_all,
            "target": target_nino_all,
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
    nino_png = output_dir / f"{args.split}_nino34_timeseries.png"

    metrics_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_variable_csv(metrics_csv, result["normalized"], result["physical"])
    _write_nino_csv(nino_csv, series["pred"], series["target"])
    plotted = _maybe_plot_nino(nino_png, series["pred"], series["target"])

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
