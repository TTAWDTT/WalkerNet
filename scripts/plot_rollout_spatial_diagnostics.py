"""绘制 rollout 空间诊断图。

输出两类图：
1. 单个样本的 anomaly 预报场 / 标签场 / 误差场对比；
2. test set 上逐格点 anomaly ACC。

示例：
    python scripts/plot_rollout_spatial_diagnostics.py \
        --config configs/server_3090_mixed5_ddp8.yaml \
        --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_ddp8/latest.pt \
        --output-dir outputs/fig_mixed5_latest_spatial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WalkerDataset
from src.evaluate_rollout import _parse_leads, _valid_subset_positions
from src.interfaces import VARIABLES
from src.model import WalkerNet
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WalkerNet rollout spatial diagnostics.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/fig_rollout_spatial")
    parser.add_argument("--variable", type=str, default="tos", choices=VARIABLES)
    parser.add_argument("--max-lead", type=int, default=18)
    parser.add_argument("--leads", type=str, default="1,3,6,9,12,18")
    parser.add_argument("--map-leads", type=str, default="1,6,12,18")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-position", type=int, default=0, help="在可用 subset 中选第几个样本画空间对比图。")
    return parser.parse_args()


def compute_grid_climatology(dataset: WalkerDataset, variable_idx: int) -> np.ndarray:
    """按 source/month 计算训练期逐格点气候态，shape=(S, 13, H, W)。"""
    train_start, train_end = dataset.data_config["train_years"]
    h, w = len(dataset.lat), len(dataset.lon)
    clim = np.full((len(dataset.source_payloads), 13, h, w), np.nan, dtype=np.float32)

    for source_idx, payload in enumerate(dataset.source_payloads):
        print(f"compute climatology source={dataset.source_names[source_idx]}", flush=True)
        years = payload["years"]
        months = payload["months"]
        train_mask = (years >= int(train_start)) & (years <= int(train_end))
        for month in range(1, 13):
            indices = np.where(train_mask & (months == month))[0]
            total = np.zeros((h, w), dtype=np.float64)
            count = np.zeros((h, w), dtype=np.float64)
            for start in range(0, len(indices), 8):
                chunk_indices = indices[start : start + 8]
                chunk = np.asarray(payload["data"][chunk_indices, variable_idx], dtype=np.float64)
                finite = np.isfinite(chunk)
                total += np.where(finite, chunk, 0.0).sum(axis=0)
                count += finite.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                clim[source_idx, month] = (total / count).astype(np.float32)
    return clim


def target_months(dataset: WalkerDataset, source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """按 source/target_t 读取目标月份。"""
    source_np = source_indices.detach().cpu().numpy()
    target_np = target_indices.detach().cpu().numpy()
    months = [
        int(dataset.source_payloads[int(source_idx)]["months"][int(target_t)])
        for source_idx, target_t in zip(source_np, target_np)
    ]
    return torch.as_tensor(months, dtype=torch.long, device=target_indices.device)


def target_physical(
    dataset: WalkerDataset,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """读取目标物理场，返回 (B, 1, 4, H, W)。"""
    source_np = source_indices.detach().cpu().numpy()
    target_np = target_indices.detach().cpu().numpy()
    raw = np.stack(
        [
            np.asarray(dataset.source_payloads[int(source_idx)]["data"][int(target_t)], dtype=np.float32)
            for source_idx, target_t in zip(source_np, target_np)
        ],
        axis=0,
    )
    return torch.from_numpy(raw).to(device=device, dtype=torch.float32)[:, None]


def anomaly_for_variable(
    field_phys: torch.Tensor,
    source_indices: torch.Tensor,
    months: torch.Tensor,
    clim: np.ndarray,
    variable_idx: int,
) -> torch.Tensor:
    """从物理场提取指定变量 anomaly，返回 (B, H, W)。"""
    source_np = source_indices.detach().cpu().numpy()
    month_np = months.detach().cpu().numpy()
    clim_np = np.stack([clim[int(source_idx), int(month)] for source_idx, month in zip(source_np, month_np)], axis=0)
    clim_tensor = torch.from_numpy(clim_np).to(device=field_phys.device, dtype=field_phys.dtype)
    return field_phys[:, 0, variable_idx] - clim_tensor


@torch.no_grad()
def collect_rollout(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    dataset: WalkerDataset,
    device: torch.device,
    max_lead: int,
    trained_rollout_steps: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    """对一个 batch rollout，返回每个 lead 的 pred/target 物理场和 target_month。"""
    window = batch["x"].to(device)
    source_index = batch.get("source_index")
    if source_index is None:
        source_index = torch.zeros(window.shape[0], dtype=torch.long)
    source_index = source_index.to(device=device, dtype=torch.long)
    base_target_t = batch["time_index"].to(device=device, dtype=torch.long)

    preds: dict[int, torch.Tensor] = {}
    targets: dict[int, torch.Tensor] = {}
    months: dict[int, torch.Tensor] = {}
    for step in range(1, max_lead + 1):
        target_t = base_target_t + step - 1
        target_month = target_months(dataset, source_index, target_t)
        rollout_step = torch.full(
            (window.shape[0],),
            min(step - 1, trained_rollout_steps - 1),
            dtype=torch.long,
            device=device,
        )
        pred_norm = model(window, target_month, rollout_step=rollout_step)
        pred_phys = dataset.denormalize(pred_norm)
        target_phys = target_physical(dataset, source_index, target_t, device)

        preds[step] = pred_phys.detach()
        targets[step] = target_phys.detach()
        months[step] = target_month.detach()
        window = torch.cat([window[:, 1:], pred_norm], dim=1)
    return preds, targets, months, source_index


def update_acc_stats(
    stats: dict[int, dict[str, torch.Tensor]],
    lead: int,
    pred_anom: torch.Tensor,
    target_anom: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """累计逐格点 ACC 需要的 sum/sum2/cross/count。"""
    mask = valid_mask[:, None].to(device=pred_anom.device, dtype=torch.bool)
    finite = torch.isfinite(pred_anom) & torch.isfinite(target_anom) & mask[:, 0]
    p = torch.where(finite, pred_anom.double(), torch.zeros((), device=pred_anom.device, dtype=torch.float64))
    y = torch.where(finite, target_anom.double(), torch.zeros((), device=target_anom.device, dtype=torch.float64))
    count = finite.double()

    stats[lead]["sum_pred"] += p.sum(dim=0).cpu()
    stats[lead]["sum_target"] += y.sum(dim=0).cpu()
    stats[lead]["sum_pred2"] += (p * p).sum(dim=0).cpu()
    stats[lead]["sum_target2"] += (y * y).sum(dim=0).cpu()
    stats[lead]["sum_cross"] += (p * y).sum(dim=0).cpu()
    stats[lead]["count"] += count.sum(dim=0).cpu()


def finalize_acc(stat: dict[str, torch.Tensor]) -> np.ndarray:
    """把累计量转成逐格点相关系数。"""
    count = stat["count"].clamp_min(1.0)
    numerator = stat["sum_cross"] - stat["sum_pred"] * stat["sum_target"] / count
    pred_var = stat["sum_pred2"] - stat["sum_pred"] * stat["sum_pred"] / count
    target_var = stat["sum_target2"] - stat["sum_target"] * stat["sum_target"] / count
    acc = numerator / torch.sqrt((pred_var * target_var).clamp_min(1e-12))
    acc = torch.where(stat["count"] >= 3, acc, torch.full_like(acc, float("nan")))
    return acc.numpy()


def plot_map_diagnostics(
    output_dir: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    leads: list[int],
    pred_maps: dict[int, np.ndarray],
    target_maps: dict[int, np.ndarray],
    variable: str,
    sample_title: str,
) -> None:
    """画预报/标签/误差空间对比图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs

        projection = ccrs.PlateCarree()
        subplot_kw = {"projection": projection}
        transform = projection
    except Exception:
        ccrs = None
        subplot_kw = {}
        transform = None

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(len(leads), 3, figsize=(9.2, 2.1 * len(leads)), subplot_kw=subplot_kw)
    if len(leads) == 1:
        axes = np.asarray([axes])

    lon2, lat2 = np.meshgrid(lon, lat)
    all_values = np.concatenate([pred_maps[lead].ravel() for lead in leads] + [target_maps[lead].ravel() for lead in leads])
    vmax = float(np.nanpercentile(np.abs(all_values), 98))
    err_values = np.concatenate([(pred_maps[lead] - target_maps[lead]).ravel() for lead in leads])
    evmax = float(np.nanpercentile(np.abs(err_values), 98))
    cmap = "RdBu_r"

    for row, lead in enumerate(leads):
        panels = [
            (pred_maps[lead], f"Lead {lead} forecast", vmax),
            (target_maps[lead], f"Lead {lead} target", vmax),
            (pred_maps[lead] - target_maps[lead], f"Lead {lead} error", evmax),
        ]
        for col, (data, title, limit) in enumerate(panels):
            ax = axes[row, col]
            kwargs = {"cmap": cmap, "vmin": -limit, "vmax": limit, "shading": "auto"}
            if transform is not None:
                kwargs["transform"] = transform
            mesh = ax.pcolormesh(lon2, lat2, data, **kwargs)
            if ccrs is not None:
                ax.coastlines(linewidth=0.35)
                ax.set_global()
            else:
                ax.set_xlim(float(lon.min()), float(lon.max()))
                ax.set_ylim(float(lat.min()), float(lat.max()))
            ax.set_title(title)
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.055, pad=0.04)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle(f"{variable} anomaly spatial comparison | {sample_title}", y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{variable}_forecast_target_error_maps.png")
    fig.savefig(output_dir / f"{variable}_forecast_target_error_maps.pdf")
    plt.close(fig)


def plot_grid_acc(
    output_dir: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    leads: list[int],
    acc_maps: dict[int, np.ndarray],
    variable: str,
) -> None:
    """画逐格点 ACC。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs

        projection = ccrs.PlateCarree()
        subplot_kw = {"projection": projection}
        transform = projection
    except Exception:
        ccrs = None
        subplot_kw = {}
        transform = None

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "figure.dpi": 240,
        "savefig.dpi": 240,
        "axes.titleweight": "bold",
    })

    ncols = 3
    nrows = int(np.ceil(len(leads) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 2.7 * nrows), subplot_kw=subplot_kw)
    axes = np.asarray(axes).reshape(nrows, ncols)
    lon2, lat2 = np.meshgrid(lon, lat)

    for idx, lead in enumerate(leads):
        ax = axes[idx // ncols, idx % ncols]
        kwargs = {"cmap": "RdBu_r", "vmin": -1.0, "vmax": 1.0, "shading": "auto"}
        if transform is not None:
            kwargs["transform"] = transform
        mesh = ax.pcolormesh(lon2, lat2, acc_maps[lead], **kwargs)
        if ccrs is not None:
            ax.coastlines(linewidth=0.35)
            ax.set_global()
        else:
            ax.set_xlim(float(lon.min()), float(lon.max()))
            ax.set_ylim(float(lat.min()), float(lat.max()))
        ax.set_title(f"Lead {lead}")

    for idx in range(len(leads), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.055, pad=0.055)
    cb.set_label("Grid-point anomaly correlation")
    fig.suptitle(f"{variable} grid-point ACC over test rollout", y=0.995, fontsize=10, fontweight="bold")
    fig.savefig(output_dir / f"{variable}_gridpoint_acc.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{variable}_gridpoint_acc.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leads = _parse_leads(args.leads, args.max_lead)
    map_leads = _parse_leads(args.map_leads, args.max_lead)
    variable_idx = VARIABLES.index(args.variable)

    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    positions = _valid_subset_positions(dataset, args.max_lead)
    subset = Subset(dataset, positions)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = WalkerNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", 1))

    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"checkpoint_epoch={checkpoint.get('epoch')}", flush=True)
    print(f"split={args.split} usable_samples={len(subset)} leads={leads} map_leads={map_leads}", flush=True)

    clim = compute_grid_climatology(dataset, variable_idx)
    h, w = len(dataset.lat), len(dataset.lon)
    stats = {
        lead: {
            "sum_pred": torch.zeros((h, w), dtype=torch.float64),
            "sum_target": torch.zeros((h, w), dtype=torch.float64),
            "sum_pred2": torch.zeros((h, w), dtype=torch.float64),
            "sum_target2": torch.zeros((h, w), dtype=torch.float64),
            "sum_cross": torch.zeros((h, w), dtype=torch.float64),
            "count": torch.zeros((h, w), dtype=torch.float64),
        }
        for lead in leads
    }

    sample_maps_pred: dict[int, np.ndarray] = {}
    sample_maps_target: dict[int, np.ndarray] = {}
    sample_title = ""

    for batch_idx, batch in enumerate(loader, start=1):
        preds, targets, months, source_index = collect_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            device=device,
            max_lead=args.max_lead,
            trained_rollout_steps=trained_rollout_steps,
        )
        valid_mask = batch["valid_mask"].to(device)[:, variable_idx]

        for lead in leads:
            pred_anom = anomaly_for_variable(preds[lead], source_index, months[lead], clim, variable_idx)
            target_anom = anomaly_for_variable(targets[lead], source_index, months[lead], clim, variable_idx)
            update_acc_stats(stats, lead, pred_anom, target_anom, valid_mask)

        if batch_idx - 1 == args.sample_position:
            source_name = dataset.source_names[int(source_index[0].detach().cpu().item())]
            target_t = int(batch["time_index"][0].item())
            sample_title = f"{source_name}, target_t={target_t}, subset_position={args.sample_position}"
            for lead in map_leads:
                pred_anom = anomaly_for_variable(preds[lead], source_index, months[lead], clim, variable_idx)
                target_anom = anomaly_for_variable(targets[lead], source_index, months[lead], clim, variable_idx)
                sample_maps_pred[lead] = pred_anom[0].detach().cpu().numpy()
                sample_maps_target[lead] = target_anom[0].detach().cpu().numpy()

        if batch_idx % 10 == 0:
            print(f"processed batches={batch_idx}/{len(loader)}", flush=True)

    acc_maps = {lead: finalize_acc(stats[lead]) for lead in leads}
    np.savez_compressed(
        output_dir / f"{args.variable}_gridpoint_acc_maps.npz",
        **{f"lead_{lead}": acc_maps[lead] for lead in leads},
        lat=np.asarray(dataset.lat),
        lon=np.asarray(dataset.lon),
    )

    plot_map_diagnostics(
        output_dir=output_dir,
        lon=np.asarray(dataset.lon),
        lat=np.asarray(dataset.lat),
        leads=map_leads,
        pred_maps=sample_maps_pred,
        target_maps=sample_maps_target,
        variable=args.variable,
        sample_title=sample_title,
    )
    plot_grid_acc(
        output_dir=output_dir,
        lon=np.asarray(dataset.lon),
        lat=np.asarray(dataset.lat),
        leads=leads,
        acc_maps=acc_maps,
        variable=args.variable,
    )
    print(f"wrote figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
