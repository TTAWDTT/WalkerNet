"""Plot CNOP cluster composites.

输出两类合成图：
1. 四类 CNOP 初始扰动的平均合成：delta_tos / delta_zos。
2. 四类扰动导致的平均误差演化：perturbed rollout - baseline rollout。

这里的 error evolution 采用 CNOP 文献中常用的扰动增长视角，即同一个初始场
叠加 CNOP 前后两次积分的差值，而不是 forecast - truth。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import NeutralCase, apply_delta, make_case_input  # noqa: E402
from scripts.cnop.plot_cnop_monthly_response import load_model, rollout_fields, smooth_field  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


VARIABLES = ("tos", "zos", "tauu", "tauv")
MAP_BOX = (100.0, 300.0, -35.0, 35.0)
NINO34_BOX = (190.0, 240.0, -5.0, 5.0)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CNOP cluster composite perturbations and error evolution.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--cluster-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--composite-cache", type=Path, default=None, help="已有合成 npz；存在时直接重画图，不重新 rollout。")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--leads", type=str, default="3,6,9,12")
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--fill-plot-nans", action="store_true", default=True, help="仅绘图时用最近邻填补 NaN 边界后再平滑。")
    parser.add_argument("--no-fill-plot-nans", dest="fill_plot_nans", action="store_false")
    parser.add_argument("--arrow-stride", type=int, default=8)
    parser.add_argument("--arrow-scale", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
        }
    )


def make_case(dataset: WalkerDataset, row: Any) -> NeutralCase:
    return NeutralCase(
        source_idx=dataset.source_names.index(str(row.source)),
        source_name=str(row.source),
        target_t=int(row.target_t),
        target_year=int(row.target_year),
        neutral_score=float(row.observed_max_3m_abs),
        observed_max_3m_abs=float(row.observed_max_3m_abs),
    )


def setup_map_axis(ax: plt.Axes, *, show_x: bool, show_y: bool) -> None:
    ax.set_xlim(MAP_BOX[0], MAP_BOX[1])
    ax.set_ylim(MAP_BOX[2], MAP_BOX[3])
    ax.set_xticks([120, 150, 180, 210, 240, 270, 300])
    ax.set_yticks([-30, -10, 10, 30])
    ax.set_xticklabels(["120E", "150E", "180", "150W", "120W", "90W", "60W"] if show_x else [])
    ax.set_yticklabels(["30S", "10S", "10N", "30N"] if show_y else [])
    ax.grid(color="#9AA3AF", alpha=0.16, linewidth=0.32)
    lon_min, lon_max, lat_min, lat_max = NINO34_BOX
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="#111827",
        lw=0.65,
    )
    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")
        spine.set_linewidth(0.55)


def fill_nan_for_plot(field: np.ndarray) -> np.ndarray:
    """仅用于绘图美化：最近邻补小空洞/海陆边界 NaN，不回写科学数据。"""

    arr = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return arr
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return arr
    _, indices = distance_transform_edt(~finite, return_indices=True)
    filled = arr[tuple(indices)]
    return filled.astype(np.float32)


def pretty_field(field: np.ndarray, sigma: float, fill_nans: bool) -> np.ndarray:
    arr = fill_nan_for_plot(field) if fill_nans else np.asarray(field, dtype=np.float32)
    return smooth_field(arr, sigma)


def add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )


def load_cluster_rows(cnop_dir: Path, cluster_summary: Path | None) -> pd.DataFrame:
    path = cluster_summary or cnop_dir / "cluster" / "cnop_cluster_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    if "cluster" not in rows.columns:
        raise ValueError(f"{path} must contain a 'cluster' column")
    return rows.sort_values(["cluster", "source", "target_year"]).reset_index(drop=True)


def accumulate_composites(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    rows: pd.DataFrame,
    cnop_dir: Path,
    horizon: int,
    trained_rollout_steps: int,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """返回 cluster -> mean_delta, mean_response, response_count, case_count。"""

    h, w = 180, 360
    clusters = sorted(int(item) for item in rows["cluster"].unique())
    delta_sum = {cluster: np.zeros((2, h, w), dtype=np.float64) for cluster in clusters}
    delta_count = {cluster: np.zeros((2, h, w), dtype=np.float64) for cluster in clusters}
    response_sum = {cluster: np.zeros((horizon, 4, h, w), dtype=np.float64) for cluster in clusters}
    response_count = {cluster: np.zeros((horizon, 4, h, w), dtype=np.float64) for cluster in clusters}
    case_count = {cluster: 0 for cluster in clusters}

    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        cluster = int(row.cluster)
        source = str(row.source)
        year = int(row.target_year)
        npz_path = cnop_dir / f"case_{source}_{year}.npz"
        with np.load(npz_path) as payload:
            delta_norm = np.asarray(payload["delta_norm"], dtype=np.float32)

        case = make_case(dataset, row)
        source_payload = dataset.source_payloads[case.source_idx]
        valid = np.asarray(source_payload["valid_mask"], dtype=bool)
        delta_valid = valid[:2] & np.isfinite(delta_norm)
        delta_sum[cluster] += np.where(delta_valid, delta_norm, 0.0)
        delta_count[cluster] += delta_valid.astype(np.float64)

        x0 = make_case_input(dataset, case, device)
        delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
        x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
        baseline = rollout_fields(model, dataset, case, x0, horizon, trained_rollout_steps).numpy()
        perturbed = rollout_fields(model, dataset, case, x_pert, horizon, trained_rollout_steps).numpy()
        response = perturbed - baseline
        response_valid = np.broadcast_to(valid[None], response.shape) & np.isfinite(response)
        response_sum[cluster] += np.where(response_valid, response, 0.0)
        response_count[cluster] += response_valid.astype(np.float64)
        case_count[cluster] += 1
        print(f"[composite] {idx}/{len(rows)} cluster={cluster} {source} {year}", flush=True)

    mean_delta = {
        cluster: np.divide(delta_sum[cluster], delta_count[cluster], out=np.full_like(delta_sum[cluster], np.nan), where=delta_count[cluster] > 0)
        for cluster in clusters
    }
    mean_response = {
        cluster: np.divide(
            response_sum[cluster],
            response_count[cluster],
            out=np.full_like(response_sum[cluster], np.nan),
            where=response_count[cluster] > 0,
        )
        for cluster in clusters
    }
    return mean_delta, mean_response, response_count, case_count


def plot_initial_perturbation(
    mean_delta: dict[int, np.ndarray],
    case_count: dict[int, int],
    lat: np.ndarray,
    lon: np.ndarray,
    output_dir: Path,
    smooth_sigma: float,
    fill_plot_nans: bool,
    dpi: int,
) -> Path:
    clusters = sorted(mean_delta)
    all_tos = np.concatenate([mean_delta[c][0].ravel() for c in clusters])
    all_zos = np.concatenate([mean_delta[c][1].ravel() for c in clusters])
    tos_v = max(float(np.nanpercentile(np.abs(all_tos), 98)), 1.0e-6)
    zos_v = max(float(np.nanpercentile(np.abs(all_zos), 98)), 1.0e-6)
    fig, axes = plt.subplots(len(clusters), 2, figsize=(8.6, 9.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.09, wspace=0.045, hspace=0.10)
    meshes = [None, None]
    for row_idx, cluster in enumerate(clusters):
        for col_idx, (var_idx, title, cmap, vmax) in enumerate(
            [(0, "TOS initial perturbation", "RdBu_r", tos_v), (1, "ZOS initial perturbation", "PRGn", zos_v)]
        ):
            ax = axes[row_idx, col_idx]
            field = pretty_field(mean_delta[cluster][var_idx], smooth_sigma, fill_plot_nans)
            meshes[col_idx] = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
            setup_map_axis(ax, show_x=row_idx == len(clusters) - 1, show_y=col_idx == 0)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            add_panel_label(ax, f"Cluster {cluster}  n={case_count[cluster]}")
    cax0 = fig.add_axes([0.18, 0.035, 0.25, 0.018])
    cax1 = fig.add_axes([0.57, 0.035, 0.25, 0.018])
    fig.colorbar(meshes[0], cax=cax0, orientation="horizontal").set_label("mean delta_norm TOS")
    fig.colorbar(meshes[1], cax=cax1, orientation="horizontal").set_label("mean delta_norm ZOS")
    fig.suptitle("Cluster-mean CNOP initial perturbations", fontsize=12.5, fontweight="bold", y=0.975)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cnop64_cluster_mean_initial_perturbations.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def plot_error_evolution(
    mean_response: dict[int, np.ndarray],
    case_count: dict[int, int],
    lat: np.ndarray,
    lon: np.ndarray,
    leads: list[int],
    output_dir: Path,
    smooth_sigma: float,
    fill_plot_nans: bool,
    arrow_stride: int,
    arrow_scale: float,
    dpi: int,
) -> tuple[Path, Path]:
    clusters = sorted(mean_response)
    view_mask = (lat[:, None] >= MAP_BOX[2]) & (lat[:, None] <= MAP_BOX[3]) & (lon[None, :] >= MAP_BOX[0]) & (lon[None, :] <= MAP_BOX[1])
    tos_values = []
    zos_values = []
    tau_values = []
    for cluster in clusters:
        resp = mean_response[cluster]
        for lead in leads:
            idx = lead - 1
            tos_values.append(resp[idx, 0][view_mask])
            zos_values.append(resp[idx, 1][view_mask])
            tau_values.append(np.sqrt(resp[idx, 2][view_mask] ** 2 + resp[idx, 3][view_mask] ** 2))
    tos_v = max(float(np.nanpercentile(np.abs(np.concatenate(tos_values)), 98)), 1.0e-6)
    zos_v = max(float(np.nanpercentile(np.abs(np.concatenate(zos_values)), 98)), 1.0e-6)
    tau_ref = max(float(np.nanpercentile(np.concatenate(tau_values), 95)), 1.0e-6)

    def draw(kind: str) -> Path:
        is_tos = kind == "tos"
        cmap = "RdYlBu_r" if is_tos else "BrBG"
        vmax = tos_v if is_tos else zos_v
        label = "TOS response + wind response" if is_tos else "ZOS response"
        fig, axes = plt.subplots(len(clusters), len(leads), figsize=(13.0, 8.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.055, right=0.925, top=0.91, bottom=0.085, wspace=0.035, hspace=0.085)
        lon2, lat2 = np.meshgrid(lon, lat)
        mesh = None
        for row_idx, cluster in enumerate(clusters):
            resp = mean_response[cluster]
            for col_idx, lead in enumerate(leads):
                ax = axes[row_idx, col_idx]
                idx = lead - 1
                field_idx = 0 if is_tos else 1
                field = pretty_field(resp[idx, field_idx], smooth_sigma, fill_plot_nans)
                mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
                setup_map_axis(ax, show_x=row_idx == len(clusters) - 1, show_y=col_idx == 0)
                if row_idx == 0:
                    ax.set_title(f"Lead {lead}", fontweight="bold")
                if col_idx == 0:
                    add_panel_label(ax, f"Cluster {cluster}  n={case_count[cluster]}")
                if is_tos:
                    step = max(1, int(arrow_stride))
                    sl_lat = (lat >= MAP_BOX[2]) & (lat <= MAP_BOX[3])
                    sl_lon = (lon >= MAP_BOX[0]) & (lon <= MAP_BOX[1])
                    lat_idx = np.where(sl_lat)[0][::step]
                    lon_idx = np.where(sl_lon)[0][::step]
                    # 合成风场只画稀疏箭头，避免把海温色块盖住。
                    ax.quiver(
                        lon2[np.ix_(lat_idx, lon_idx)],
                        lat2[np.ix_(lat_idx, lon_idx)],
                        pretty_field(resp[idx, 2], smooth_sigma, fill_plot_nans)[np.ix_(lat_idx, lon_idx)],
                        pretty_field(resp[idx, 3], smooth_sigma, fill_plot_nans)[np.ix_(lat_idx, lon_idx)],
                        color="#1F2937",
                        width=0.00135,
                        headwidth=3.0,
                        headlength=3.5,
                        headaxislength=3.2,
                        scale=max(tau_ref / max(float(arrow_scale), 1.0e-6), 1.0e-6),
                        scale_units="xy",
                        angles="xy",
                        alpha=0.68,
                    )
        assert mesh is not None
        cax = fig.add_axes([0.28, 0.035, 0.42, 0.018])
        fig.colorbar(mesh, cax=cax, orientation="horizontal").set_label("cluster-mean perturbed rollout - baseline rollout")
        fig.suptitle(f"Cluster-mean CNOP error evolution: {label}", fontsize=12.5, fontweight="bold", y=0.972)
        path = output_dir / f"cnop64_cluster_mean_error_evolution_{kind}.png"
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
        plt.close(fig)
        return path

    return draw("tos"), draw("zos")


def main() -> None:
    set_style()
    args = parse_args()
    output_dir = args.output_dir or args.cnop_dir / "cluster_composites"
    leads = [int(item.strip()) for item in args.leads.split(",") if item.strip()]
    if any(lead < 1 or lead > args.horizon for lead in leads):
        raise ValueError(f"leads must be within 1..{args.horizon}, got {leads}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    rows = load_cluster_rows(args.cnop_dir, args.cluster_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.composite_cache or output_dir / "cnop64_cluster_composites.npz"
    if cache_path.exists():
        with np.load(cache_path) as data:
            clusters = [int(item) for item in data["clusters"]]
            counts = [int(item) for item in data["case_count"]]
            delta_stack = np.asarray(data["mean_delta"], dtype=np.float64)
            response_stack = np.asarray(data["mean_response"], dtype=np.float64)
            lat = np.asarray(data["lat"], dtype=np.float64)
            lon = np.asarray(data["lon"], dtype=np.float64)
        mean_delta = {cluster: delta_stack[idx] for idx, cluster in enumerate(clusters)}
        mean_response = {cluster: response_stack[idx] for idx, cluster in enumerate(clusters)}
        case_count = {cluster: counts[idx] for idx, cluster in enumerate(clusters)}
        print(f"[composite] using cache {cache_path}", flush=True)
    else:
        mean_delta, mean_response, _response_count, case_count = accumulate_composites(
            model,
            dataset,
            rows,
            args.cnop_dir,
            args.horizon,
            trained_rollout_steps,
            device,
        )
        payload = dataset.source_payloads[0]
        lat = np.asarray(payload["lat"], dtype=np.float64)
        lon = np.asarray(payload["lon"], dtype=np.float64)
        np.savez_compressed(
            cache_path,
            clusters=np.asarray(sorted(mean_delta), dtype=np.int64),
            case_count=np.asarray([case_count[c] for c in sorted(mean_delta)], dtype=np.int64),
            mean_delta=np.stack([mean_delta[c] for c in sorted(mean_delta)]),
            mean_response=np.stack([mean_response[c] for c in sorted(mean_response)]),
            lat=lat,
            lon=lon,
            leads=np.asarray(leads, dtype=np.int64),
            checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1), dtype=np.int64),
        )
    p0 = plot_initial_perturbation(mean_delta, case_count, lat, lon, output_dir, args.smooth_sigma, args.fill_plot_nans, args.dpi)
    p1, p2 = plot_error_evolution(
        mean_response,
        case_count,
        lat,
        lon,
        leads,
        output_dir,
        args.smooth_sigma,
        args.fill_plot_nans,
        args.arrow_stride,
        args.arrow_scale,
        args.dpi,
    )
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"wrote {p0}")
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
