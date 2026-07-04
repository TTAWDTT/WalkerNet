"""Create a paper-style CNOP monthly response figure.

This figure is intentionally more polished than the diagnostic atlas:

- left: a large summary map for one selected month;
- right: six selected months, each with TOS + wind response above ZOS response;
- all scalar fields are lightly low-pass filtered for display;
- coastlines and land outlines are drawn with Cartopy when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_cnop_monthly_response import (  # noqa: E402
    NINO34_BOX,
    apply_delta,
    load_case_npz,
    load_model,
    make_case,
    make_case_input,
    month_labels,
    read_case_from_summary,
    rollout_fields,
    smooth_field,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402

try:  # noqa: SIM105
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:  # pragma: no cover - optional plotting dependency
    HAS_CARTOPY = False


MAP_BOX = (120.0, 290.0, -35.0, 35.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style CNOP response figure.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--case-source", type=str, default="")
    parser.add_argument("--case-year", type=int, default=0)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--months", type=str, default="2,4,6,8,10,12", help="1-based forecast months shown on right.")
    parser.add_argument("--summary-month", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--vector-sigma", type=float, default=4.0)
    parser.add_argument("--arrow-stride", type=int, default=9)
    parser.add_argument("--arrow-scale", type=float, default=4.5)
    parser.add_argument("--tos-vmax", type=float, default=2.6)
    parser.add_argument("--zos-vmax", type=float, default=0.08)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.65,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 420,
        }
    )


def parse_months(value: str) -> list[int]:
    months = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not months:
        raise ValueError("months must contain at least one month")
    if any(month < 1 or month > 12 for month in months):
        raise ValueError(f"months must be in [1, 12], got {months}")
    return months


def projection():
    if HAS_CARTOPY:
        return ccrs.PlateCarree(central_longitude=180)
    return None


def add_projected_box(ax: plt.Axes, bounds: tuple[float, float, float, float], color: str) -> None:
    """画 Niño3.4 框；Cartopy 坐标轴需要显式声明数据经纬度坐标系。"""

    lon_min, lon_max, lat_min, lat_max = bounds
    kwargs = {"color": color, "lw": 0.9, "zorder": 5}
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        **kwargs,
    )


def add_map_features(ax: plt.Axes, show_xticks: bool, show_yticks: bool) -> None:
    if HAS_CARTOPY:
        data_crs = ccrs.PlateCarree()
        ax.set_extent(MAP_BOX, crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F4F1E8", edgecolor="none", zorder=2)
        ax.coastlines(resolution="110m", linewidth=0.45, color="#2F3A3F", zorder=3)
        ax.set_xticks([150, 180, 210, 240, 270], crs=data_crs)
        ax.set_yticks([-30, -10, 10, 30], crs=data_crs)
    else:
        ax.set_xlim(MAP_BOX[0], MAP_BOX[1])
        ax.set_ylim(MAP_BOX[2], MAP_BOX[3])
        ax.set_xticks([150, 180, 210, 240, 270])
        ax.set_yticks([-30, -10, 10, 30])
    ax.set_xticklabels(["150E", "180", "150W", "120W", "90W"] if show_xticks else [])
    ax.set_yticklabels(["30S", "10S", "10N", "30N"] if show_yticks else [])
    ax.grid(color="#6B7280", linewidth=0.35, alpha=0.35)
    add_projected_box(ax, NINO34_BOX, "#0F766E")


def add_layer_label(ax: plt.Axes, label: str) -> None:
    """在小图左上角放置层标签，避免和月份标题、经纬度刻度抢位置。"""

    ax.text(
        0.018,
        0.91,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        fontweight="bold",
        color="#1F2933",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        zorder=8,
    )


def contour_map(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, levels: np.ndarray, cmap: str):
    kwargs = {"levels": levels, "cmap": cmap, "extend": "both"}
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    return ax.contourf(lon, lat, field, **kwargs)


def quiver_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    tau_ref: float,
    stride: int,
    arrow_scale: float,
) -> None:
    lon2, lat2 = np.meshgrid(lon, lat)
    sl_lat = (lat >= MAP_BOX[2]) & (lat <= MAP_BOX[3])
    sl_lon = (lon >= MAP_BOX[0]) & (lon <= MAP_BOX[1])
    lat_idx = np.where(sl_lat)[0][:: max(1, stride)]
    lon_idx = np.where(sl_lon)[0][:: max(1, stride)]
    kwargs = {}
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    ax.quiver(
        lon2[np.ix_(lat_idx, lon_idx)],
        lat2[np.ix_(lat_idx, lon_idx)],
        u[np.ix_(lat_idx, lon_idx)],
        v[np.ix_(lat_idx, lon_idx)],
        color="#263238",
        width=0.00125,
        headwidth=2.6,
        headlength=3.0,
        headaxislength=2.8,
        scale=max(tau_ref / max(float(arrow_scale), 1.0e-6), 1.0e-6),
        scale_units="xy",
        angles="xy",
        alpha=0.62,
        zorder=4,
        **kwargs,
    )


def build_response(
    config_path: Path,
    checkpoint_path: Path,
    cnop_dir: Path,
    case_source: str,
    case_year: int,
    rank: int,
    split: str,
    device_name: str,
    horizon: int,
    trained_rollout_steps_arg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, int]:
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    config = load_config(config_path)
    dataset = WalkerDataset(config["data"]["path"], config, split=split)
    model, _checkpoint = load_model(config, checkpoint_path, device)
    trained_rollout_steps = int(trained_rollout_steps_arg or config.get("training", {}).get("rollout_steps", horizon))

    source, year, target_t, observed = read_case_from_summary(cnop_dir, case_source, case_year)
    case = make_case(dataset, source, year, target_t, observed)
    delta_norm, _npz_path = load_case_npz(cnop_dir, source, year, rank)

    x0 = make_case_input(dataset, case, device)
    delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
    x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
    baseline = rollout_fields(model, dataset, case, x0, horizon, trained_rollout_steps)
    perturbed = rollout_fields(model, dataset, case, x_pert, horizon, trained_rollout_steps)
    response = (perturbed - baseline).numpy()

    payload = dataset.source_payloads[case.source_idx]
    lat = np.asarray(payload["lat"], dtype=np.float64)
    lon = np.asarray(payload["lon"], dtype=np.float64)
    labels = month_labels(case, dataset, horizon)
    return response, lat, lon, labels, source, year


def lowpass_response(response: np.ndarray, scalar_sigma: float, vector_sigma: float) -> np.ndarray:
    plot_response = np.empty_like(response)
    for month_idx in range(response.shape[0]):
        for var_idx in range(response.shape[1]):
            sigma = vector_sigma if var_idx in (2, 3) else scalar_sigma
            plot_response[month_idx, var_idx] = smooth_field(response[month_idx, var_idx], sigma)
    return plot_response


def plot_paper_figure(
    response: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    labels: list[str],
    source: str,
    year: int,
    rank: int,
    months: list[int],
    summary_month: int,
    output_dir: Path,
    dpi: int,
    smooth_sigma: float,
    vector_sigma: float,
    arrow_stride: int,
    arrow_scale: float,
    tos_vmax: float,
    zos_vmax: float,
) -> Path:
    plot_response = lowpass_response(response, smooth_sigma, vector_sigma)
    tos_levels = np.linspace(-tos_vmax, tos_vmax, 35)
    zos_levels = np.linspace(-zos_vmax, zos_vmax, 31)
    tau = np.sqrt(plot_response[:, 2] ** 2 + plot_response[:, 3] ** 2)
    view_mask = (lat[:, None] >= MAP_BOX[2]) & (lat[:, None] <= MAP_BOX[3]) & (lon[None, :] >= MAP_BOX[0]) & (lon[None, :] <= MAP_BOX[1])
    tau_ref = max(float(np.nanpercentile(tau[..., view_mask], 94)), 1.0e-6)

    proj = projection()
    fig = plt.figure(figsize=(15.5, 6.05))
    outer = fig.add_gridspec(2, 4, width_ratios=(1.45, 1.0, 1.0, 1.0), wspace=0.13, hspace=0.28)

    ax_main = fig.add_subplot(outer[:, 0], projection=proj) if HAS_CARTOPY else fig.add_subplot(outer[:, 0])
    summary_idx = min(max(summary_month, 1), response.shape[0]) - 1
    main = contour_map(ax_main, lon, lat, plot_response[summary_idx, 0], tos_levels, "RdYlBu_r")
    quiver_map(
        ax_main,
        lon,
        lat,
        plot_response[summary_idx, 2],
        plot_response[summary_idx, 3],
        tau_ref,
        max(arrow_stride, 10),
        arrow_scale,
    )
    add_map_features(ax_main, show_xticks=True, show_yticks=True)
    ax_main.set_title(f"(a) Lead {summary_month}: {labels[summary_idx]} TOS + wind response", y=1.025, fontweight="bold")

    right_axes: list[plt.Axes] = []
    tos_mappable = main
    zos_mappable = None
    panel_ord = 1
    for pos, month in enumerate(months[:6]):
        row, col = divmod(pos, 3)
        cell = outer[row, col + 1].subgridspec(2, 1, height_ratios=(1.0, 0.68), hspace=0.04)
        ax_tos = fig.add_subplot(cell[0, 0], projection=proj) if HAS_CARTOPY else fig.add_subplot(cell[0, 0])
        ax_zos = fig.add_subplot(cell[1, 0], projection=proj) if HAS_CARTOPY else fig.add_subplot(cell[1, 0], sharex=ax_tos)
        right_axes.extend([ax_tos, ax_zos])
        idx = month - 1

        tos_mappable = contour_map(ax_tos, lon, lat, plot_response[idx, 0], tos_levels, "RdYlBu_r")
        quiver_map(ax_tos, lon, lat, plot_response[idx, 2], plot_response[idx, 3], tau_ref, arrow_stride, arrow_scale)
        add_map_features(ax_tos, show_xticks=False, show_yticks=col == 0)
        add_layer_label(ax_tos, "TOS + wind")
        ax_tos.set_title(f"({chr(97 + panel_ord)}) Lead {month}: {labels[idx]}", y=1.02, fontweight="bold")
        panel_ord += 1

        zos_mappable = contour_map(ax_zos, lon, lat, plot_response[idx, 1], zos_levels, "BrBG")
        add_map_features(ax_zos, show_xticks=row == 1, show_yticks=col == 0)
        add_layer_label(ax_zos, "ZOS")

    cax_tos = fig.add_axes([0.11, 0.08, 0.26, 0.022])
    cax_zos = fig.add_axes([0.50, 0.08, 0.34, 0.022])
    cb1 = fig.colorbar(tos_mappable, cax=cax_tos, orientation="horizontal")
    cb1.set_label("TOS response")
    cb2 = fig.colorbar(zos_mappable, cax=cax_zos, orientation="horizontal")
    cb2.set_label("ZOS response")
    fig.text(0.86, 0.087, "vectors: wind stress response", fontsize=7.5, color="#263238")

    fig.suptitle(
        f"CNOP response evolution: {source} {year}, candidate rank {rank}",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.17)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_paper_response_{source}_{year}_rank{rank}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    set_style()
    months = parse_months(args.months)
    response, lat, lon, labels, source, year = build_response(
        args.config,
        args.checkpoint,
        args.cnop_dir,
        args.case_source,
        args.case_year,
        args.candidate_rank,
        args.split,
        args.device,
        args.horizon,
        args.trained_rollout_steps,
    )
    output_dir = args.output_dir or args.cnop_dir / "figures"
    path = plot_paper_figure(
        response,
        lat,
        lon,
        labels,
        source,
        year,
        args.candidate_rank,
        months,
        args.summary_month,
        output_dir,
        args.dpi,
        args.smooth_sigma,
        args.vector_sigma,
        args.arrow_stride,
        args.arrow_scale,
        args.tos_vmax,
        args.zos_vmax,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
