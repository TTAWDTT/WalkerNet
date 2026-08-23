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
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.plot_cnop_monthly_response import (  # noqa: E402
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
from scripts.cnop.forecast_field_climatology import (  # noqa: E402
    load_or_compute_forecast_field_climatology,
    monthly_observed_field_climatology,
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
# Match the original paper-style monthly response palette exactly.
TOS_CMAP = "RdYlBu_r"
ZOS_CMAP = "BrBG"


@dataclass
class CaseProducts:
    """一个 CNOP 候选扰动对应的全部可视化材料。"""

    perturbation: np.ndarray
    truth: np.ndarray
    baseline: np.ndarray
    perturbed: np.ndarray
    response: np.ndarray
    observed_field_climatology: np.ndarray
    source_idx: int
    target_months: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    labels: list[str]
    source: str
    year: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style CNOP response figure.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--case-source", type=str, default="")
    parser.add_argument("--case-year", type=int, default=0)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--candidate-ranks", type=str, default="", help="Comma-separated candidate ranks; overrides --candidate-rank.")
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
    parser.add_argument("--tos-vmax", type=float, default=0.0, help="0 means the old 98th-percentile response scaling.")
    parser.add_argument("--zos-vmax", type=float, default=0.0, help="0 means the old 98th-percentile response scaling.")
    parser.add_argument("--perturb-tos-vmax", type=float, default=0.0, help="0 means auto percentile.")
    parser.add_argument("--perturb-zos-vmax", type=float, default=0.0, help="0 means auto percentile.")
    parser.add_argument("--constraint-label", type=str, default="", help="Optional perturbation constraint label shown on perturbation figures.")
    parser.add_argument("--contour-levels", type=int, default=23)
    parser.add_argument("--zero-contour", action="store_true", default=True)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--skip-response", action="store_true")
    parser.add_argument("--skip-perturbation", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-multi-perturbation", action="store_true")
    parser.add_argument(
        "--comparison-anomaly",
        action="store_true",
        help="Plot truth and forecasts as anomalies in their respective observed/model climatological worlds.",
    )
    parser.add_argument(
        "--forecast-climatology",
        choices=("train", "split", "all"),
        default="train",
        help="Reference period for the lead/month-specific model forecast climatology.",
    )
    parser.add_argument("--forecast-climatology-cache", type=Path, default=None)
    parser.add_argument("--climatology-batch-size", type=int, default=2)
    parser.add_argument("--no-cartopy", action="store_true", help="Disable Cartopy features for offline rendering.")
    parser.add_argument("--title-suffix", default="", help="Optional text appended to the response-evolution title.")
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


def parse_ranks(value: str, fallback: int) -> list[int]:
    if not value.strip():
        return [fallback]
    ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ranks:
        raise ValueError("candidate ranks must contain at least one rank")
    if any(rank < 1 for rank in ranks):
        raise ValueError(f"candidate ranks must be positive, got {ranks}")
    return ranks


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


def contour_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    levels: np.ndarray,
    cmap: str | LinearSegmentedColormap,
    draw_zero: bool,
):
    kwargs = {"levels": levels, "cmap": cmap, "extend": "both"}
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    mappable = ax.contourf(lon, lat, field, **kwargs)
    if draw_zero:
        zero_kwargs = {"levels": [0.0], "colors": "#293241", "linewidths": 0.28, "alpha": 0.55}
        if HAS_CARTOPY:
            zero_kwargs["transform"] = ccrs.PlateCarree()
        ax.contour(lon, lat, field, **zero_kwargs)
    return mappable


def plain_contour_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    levels: np.ndarray,
    cmap: str | LinearSegmentedColormap,
    extend: str = "both",
):
    kwargs = {"levels": levels, "cmap": cmap, "extend": extend}
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


def build_case_products(
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
) -> CaseProducts:
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
    perturbation = (dataset.denormalize(x_pert)[:, -1, :2] - dataset.denormalize(x0)[:, -1, :2])[0].detach().cpu().numpy()
    baseline = rollout_fields(model, dataset, case, x0, horizon, trained_rollout_steps)
    perturbed = rollout_fields(model, dataset, case, x_pert, horizon, trained_rollout_steps)
    response = (perturbed - baseline).numpy()

    payload = dataset.source_payloads[case.source_idx]
    truth = np.asarray(payload["data"][case.target_t : case.target_t + horizon], dtype=np.float32)
    lat = np.asarray(payload["lat"], dtype=np.float64)
    lon = np.asarray(payload["lon"], dtype=np.float64)
    labels = month_labels(case, dataset, horizon)
    target_months = np.asarray(payload["months"][case.target_t : case.target_t + horizon], dtype=np.int64)
    observed_field_climatology = monthly_observed_field_climatology(dataset, case.source_idx, target_months)
    return CaseProducts(
        perturbation=perturbation,
        truth=truth,
        baseline=baseline.numpy(),
        perturbed=perturbed.numpy(),
        response=response,
        observed_field_climatology=observed_field_climatology,
        source_idx=case.source_idx,
        target_months=target_months,
        lat=lat,
        lon=lon,
        labels=labels,
        source=source,
        year=year,
    )


def lowpass_response(response: np.ndarray, scalar_sigma: float, vector_sigma: float) -> np.ndarray:
    plot_response = np.empty_like(response)
    for month_idx in range(response.shape[0]):
        for var_idx in range(response.shape[1]):
            sigma = vector_sigma if var_idx in (2, 3) else scalar_sigma
            plot_response[month_idx, var_idx] = smooth_field(response[month_idx, var_idx], sigma)
    return plot_response


def symmetric_vmax(field: np.ndarray, fallback: float, percentile: float = 98.0) -> float:
    if fallback > 0:
        return float(fallback)
    return max(float(np.nanpercentile(np.abs(field), percentile)), 1.0e-8)


def plot_perturbation_figure(
    perturbation: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    source: str,
    year: int,
    rank: int,
    output_dir: Path,
    dpi: int,
    smooth_sigma: float,
    tos_vmax: float,
    zos_vmax: float,
    contour_levels: int,
    zero_contour: bool,
    constraint_label: str,
) -> Path:
    """画 CNOP 本体：输入第 12 个月上实际加入的 δTOS 与 δZOS。"""

    plot_delta = np.stack([smooth_field(perturbation[0], smooth_sigma), smooth_field(perturbation[1], smooth_sigma)])
    tos_lim = symmetric_vmax(plot_delta[0], tos_vmax)
    zos_lim = symmetric_vmax(plot_delta[1], zos_vmax)
    tos_levels = np.linspace(-tos_lim, tos_lim, contour_levels)
    zos_levels = np.linspace(-zos_lim, zos_lim, contour_levels)

    proj = projection()
    fig = plt.figure(figsize=(10.2, 3.65))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    axes = [
        fig.add_subplot(gs[0, 0], projection=proj) if HAS_CARTOPY else fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1], projection=proj) if HAS_CARTOPY else fig.add_subplot(gs[0, 1]),
    ]
    m0 = contour_map(axes[0], lon, lat, plot_delta[0], tos_levels, TOS_CMAP, zero_contour)
    add_map_features(axes[0], show_xticks=True, show_yticks=True)
    axes[0].set_title("a  TOS perturbation", loc="left", fontweight="bold")

    m1 = contour_map(axes[1], lon, lat, plot_delta[1], zos_levels, TOS_CMAP, zero_contour)
    add_map_features(axes[1], show_xticks=True, show_yticks=False)
    axes[1].set_title("b  ZOS perturbation", loc="left", fontweight="bold")

    cax0 = fig.add_axes([0.145, 0.145, 0.31, 0.026])
    cax1 = fig.add_axes([0.565, 0.145, 0.31, 0.026])
    cb0 = fig.colorbar(m0, cax=cax0, orientation="horizontal")
    cb0.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb0.set_label("delta TOS")
    cb1 = fig.colorbar(m1, cax=cax1, orientation="horizontal")
    cb1.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb1.set_label("delta ZOS")
    fig.suptitle(f"CNOP candidate {rank}: initial perturbation for {source} {year}", fontsize=10.5, fontweight="bold", y=0.965)
    top = 0.82
    if constraint_label:
        fig.text(0.5, 0.905, constraint_label, ha="center", fontsize=7.4, color="#4B5563")
    else:
        top = 0.86
    fig.subplots_adjust(left=0.045, right=0.99, top=top, bottom=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_initial_perturbation_{source}_{year}_rank{rank}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def plot_multi_perturbation_figure(
    products_by_rank: list[tuple[int, CaseProducts]],
    output_dir: Path,
    dpi: int,
    smooth_sigma: float,
    tos_vmax: float,
    zos_vmax: float,
    contour_levels: int,
    zero_contour: bool,
    constraint_label: str,
) -> Path:
    """把同一个初始场的多个候选扰动放在一张图里横向比较。"""

    if not products_by_rank:
        raise ValueError("products_by_rank must not be empty")
    source = products_by_rank[0][1].source
    year = products_by_rank[0][1].year
    lat = products_by_rank[0][1].lat
    lon = products_by_rank[0][1].lon
    deltas = [
        (
            rank,
            np.stack(
                [
                    smooth_field(products.perturbation[0], smooth_sigma),
                    smooth_field(products.perturbation[1], smooth_sigma),
                ]
            ),
        )
        for rank, products in products_by_rank
    ]
    tos_lim = symmetric_vmax(np.stack([delta[0] for _rank, delta in deltas]), tos_vmax)
    zos_lim = symmetric_vmax(np.stack([delta[1] for _rank, delta in deltas]), zos_vmax)
    tos_levels = np.linspace(-tos_lim, tos_lim, contour_levels)
    zos_levels = np.linspace(-zos_lim, zos_lim, contour_levels)

    proj = projection()
    nrows = len(deltas)
    fig = plt.figure(figsize=(10.8, 2.05 * nrows + 1.2))
    gs = fig.add_gridspec(nrows, 2, wspace=0.08, hspace=0.18)
    m_tos = None
    m_zos = None
    for row, (rank, delta) in enumerate(deltas):
        ax_tos = fig.add_subplot(gs[row, 0], projection=proj) if HAS_CARTOPY else fig.add_subplot(gs[row, 0])
        ax_zos = fig.add_subplot(gs[row, 1], projection=proj) if HAS_CARTOPY else fig.add_subplot(gs[row, 1])
        m_tos = contour_map(ax_tos, lon, lat, delta[0], tos_levels, TOS_CMAP, zero_contour)
        m_zos = contour_map(ax_zos, lon, lat, delta[1], zos_levels, TOS_CMAP, zero_contour)
        add_map_features(ax_tos, show_xticks=row == nrows - 1, show_yticks=True)
        add_map_features(ax_zos, show_xticks=row == nrows - 1, show_yticks=False)
        ax_tos.set_ylabel(f"candidate {rank}", fontsize=8, fontweight="bold")
        if row == 0:
            ax_tos.set_title("a  TOS perturbation", loc="left", fontweight="bold")
            ax_zos.set_title("b  ZOS perturbation", loc="left", fontweight="bold")

    cax0 = fig.add_axes([0.17, 0.075, 0.28, 0.020])
    cax1 = fig.add_axes([0.57, 0.075, 0.28, 0.020])
    cb0 = fig.colorbar(m_tos, cax=cax0, orientation="horizontal")
    cb0.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb0.set_label("delta TOS")
    cb1 = fig.colorbar(m_zos, cax=cax1, orientation="horizontal")
    cb1.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb1.set_label("delta ZOS")
    ranks_text = ",".join(str(rank) for rank, _products in products_by_rank)
    fig.suptitle(
        f"CNOP candidates from repeated starts: {source} {year}, ordered {ranks_text}",
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )
    if constraint_label:
        fig.text(0.5, 0.945, constraint_label, ha="center", fontsize=7.4, color="#4B5563")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.15)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_multi_initial_perturbations_{source}_{year}_ranks{ranks_text}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def plot_comparison_figure(
    truth: np.ndarray,
    baseline: np.ndarray,
    perturbed: np.ndarray,
    response: np.ndarray,
    observed_field_climatology: np.ndarray,
    forecast_field_climatology: np.ndarray | None,
    target_months: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    labels: list[str],
    source: str,
    year: int,
    rank: int,
    lead: int,
    output_dir: Path,
    dpi: int,
    smooth_sigma: float,
    tos_vmax: float,
    zos_vmax: float,
    contour_levels: int,
    zero_contour: bool,
    comparison_anomaly: bool,
) -> Path:
    """画真值、baseline、叠加扰动后预测、二者差值，明确展示对比链条。"""

    idx = min(max(lead, 1), baseline.shape[0]) - 1
    obs_fields = truth[idx, :2].copy()
    base_fields = baseline[idx, :2].copy()
    pert_fields = perturbed[idx, :2].copy()
    if comparison_anomaly:
        if forecast_field_climatology is None:
            raise ValueError("forecast_field_climatology is required when --comparison-anomaly is set")
        month = int(target_months[idx])
        observed_climatology = observed_field_climatology[idx, :2]
        model_climatology = forecast_field_climatology[idx, month, :2]
        if not np.isfinite(model_climatology).all():
            raise ValueError(f"No model forecast climatology for lead={lead}, month={month}")
        obs_fields -= observed_climatology
        base_fields -= model_climatology
        pert_fields -= model_climatology
    obs = np.stack([smooth_field(obs_fields[0], smooth_sigma), smooth_field(obs_fields[1], smooth_sigma)])
    base = np.stack([smooth_field(base_fields[0], smooth_sigma), smooth_field(base_fields[1], smooth_sigma)])
    pert = np.stack([smooth_field(pert_fields[0], smooth_sigma), smooth_field(pert_fields[1], smooth_sigma)])
    diff = np.stack([smooth_field(response[idx, 0], smooth_sigma), smooth_field(response[idx, 1], smooth_sigma)])

    if comparison_anomaly:
        tos_abs_max = symmetric_vmax(np.stack([obs[0], base[0], pert[0]]), 0.0)
        zos_abs_max = symmetric_vmax(np.stack([obs[1], base[1], pert[1]]), 0.0)
        tos_abs_min = -tos_abs_max
        zos_abs_min = -zos_abs_max
    else:
        tos_abs_min = float(np.nanpercentile(np.stack([obs[0], base[0], pert[0]]), 2))
        tos_abs_max = float(np.nanpercentile(np.stack([obs[0], base[0], pert[0]]), 98))
        zos_abs_min = float(np.nanpercentile(np.stack([obs[1], base[1], pert[1]]), 2))
        zos_abs_max = float(np.nanpercentile(np.stack([obs[1], base[1], pert[1]]), 98))
    tos_abs_levels = np.linspace(tos_abs_min, tos_abs_max, contour_levels)
    zos_abs_levels = np.linspace(zos_abs_min, zos_abs_max, contour_levels)
    tos_diff_levels = np.linspace(-tos_vmax, tos_vmax, contour_levels)
    zos_diff_levels = np.linspace(-zos_vmax, zos_vmax, contour_levels)

    proj = projection()
    fig = plt.figure(figsize=(15.6, 5.6))
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.16)
    prefix = "anomaly" if comparison_anomaly else ""
    col_titles = (
        f"Observed truth {prefix}".strip(),
        f"Baseline forecast {prefix}".strip(),
        f"CNOP-perturbed forecast {prefix}".strip(),
        "Difference",
    )
    row_labels = ("TOS", "ZOS")
    mappables: list[object] = []
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(gs[row, col], projection=proj) if HAS_CARTOPY else fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                m = plain_contour_map(ax, lon, lat, obs[0], tos_abs_levels, "RdBu_r" if comparison_anomaly else "Spectral_r")
            elif row == 0 and col == 1:
                m = plain_contour_map(ax, lon, lat, base[0], tos_abs_levels, "RdBu_r" if comparison_anomaly else "Spectral_r")
            elif row == 0 and col == 2:
                m = plain_contour_map(ax, lon, lat, pert[0], tos_abs_levels, "RdBu_r" if comparison_anomaly else "Spectral_r")
            elif row == 0 and col == 3:
                m = contour_map(ax, lon, lat, diff[0], tos_diff_levels, TOS_CMAP, zero_contour)
            elif row == 1 and col == 0:
                m = plain_contour_map(ax, lon, lat, obs[1], zos_abs_levels, "BrBG" if comparison_anomaly else "viridis")
            elif row == 1 and col == 1:
                m = plain_contour_map(ax, lon, lat, base[1], zos_abs_levels, "BrBG" if comparison_anomaly else "viridis")
            elif row == 1 and col == 2:
                m = plain_contour_map(ax, lon, lat, pert[1], zos_abs_levels, "BrBG" if comparison_anomaly else "viridis")
            else:
                m = contour_map(ax, lon, lat, diff[1], zos_diff_levels, ZOS_CMAP, zero_contour)
            mappables.append(m)
            add_map_features(ax, show_xticks=row == 1, show_yticks=col == 0)
            if row == 0:
                ax.set_title(f"({chr(97 + col)}) {col_titles[col]}", fontweight="bold")
            add_layer_label(ax, row_labels[row])

    caxes = [
        fig.add_axes([0.06, 0.08, 0.20, 0.020]),
        fig.add_axes([0.30, 0.08, 0.20, 0.020]),
        fig.add_axes([0.54, 0.08, 0.20, 0.020]),
        fig.add_axes([0.78, 0.08, 0.18, 0.020]),
    ]
    cb_abs_tos = fig.colorbar(mappables[0], cax=caxes[0], orientation="horizontal")
    cb_abs_tos.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb_abs_tos.set_label("TOS anomaly" if comparison_anomaly else "absolute TOS")
    cb_abs_zos = fig.colorbar(mappables[4], cax=caxes[1], orientation="horizontal")
    cb_abs_zos.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb_abs_zos.set_label("ZOS anomaly" if comparison_anomaly else "absolute ZOS")
    cb_diff_tos = fig.colorbar(mappables[3], cax=caxes[2], orientation="horizontal")
    cb_diff_tos.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb_diff_tos.set_label("TOS difference")
    cb_diff_zos = fig.colorbar(mappables[7], cax=caxes[3], orientation="horizontal")
    cb_diff_zos.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb_diff_zos.set_label("ZOS difference")
    fig.suptitle(
        f"Forecast comparison at lead {lead} ({labels[idx]}): {source} {year}, candidate rank {rank}",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.16)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_forecast_comparison_lead{lead}_{source}_{year}_rank{rank}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


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
    contour_levels: int,
    zero_contour: bool,
    title_suffix: str,
) -> Path:
    plot_response = lowpass_response(response, smooth_sigma, vector_sigma)
    view_mask = (lat[:, None] >= MAP_BOX[2]) & (lat[:, None] <= MAP_BOX[3]) & (lon[None, :] >= MAP_BOX[0]) & (lon[None, :] <= MAP_BOX[1])
    if tos_vmax <= 0:
        tos_vmax = max(float(np.nanpercentile(np.abs(plot_response[:, 0][..., view_mask]), 98)), 1.0e-6)
    if zos_vmax <= 0:
        zos_vmax = max(float(np.nanpercentile(np.abs(plot_response[:, 1][..., view_mask]), 98)), 1.0e-6)
    tos_levels = np.linspace(-tos_vmax, tos_vmax, contour_levels)
    zos_levels = np.linspace(-zos_vmax, zos_vmax, contour_levels)
    tau = np.sqrt(plot_response[:, 2] ** 2 + plot_response[:, 3] ** 2)
    view_mask = (lat[:, None] >= MAP_BOX[2]) & (lat[:, None] <= MAP_BOX[3]) & (lon[None, :] >= MAP_BOX[0]) & (lon[None, :] <= MAP_BOX[1])
    tau_ref = max(float(np.nanpercentile(tau[..., view_mask], 94)), 1.0e-6)

    proj = projection()
    # Match the original paper layout: one centered summary map at left and
    # two rows of three compact TOS/ZOS month pairs at right.  Explicit axes
    # rectangles keep the relative panel sizes stable across output DPI.
    fig = plt.figure(figsize=(12.8, 5.05))

    def add_axis(rect: tuple[float, float, float, float]) -> plt.Axes:
        if HAS_CARTOPY:
            return fig.add_axes(rect, projection=proj)
        return fig.add_axes(rect)

    ax_main = add_axis((0.025, 0.365, 0.30, 0.30))
    summary_idx = min(max(summary_month, 1), response.shape[0]) - 1
    main = contour_map(ax_main, lon, lat, plot_response[summary_idx, 0], tos_levels, TOS_CMAP, zero_contour)
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

    tos_mappable = main
    zos_mappable = None
    panel_ord = 1
    x_positions = (0.355, 0.575, 0.795)
    for pos, month in enumerate(months[:6]):
        row, col = divmod(pos, 3)
        y_tos = 0.705 if row == 0 else 0.305
        y_zos = 0.535 if row == 0 else 0.135
        ax_tos = add_axis((x_positions[col], y_tos, 0.18, 0.17))
        ax_zos = add_axis((x_positions[col], y_zos, 0.18, 0.13))
        idx = month - 1

        tos_mappable = contour_map(ax_tos, lon, lat, plot_response[idx, 0], tos_levels, TOS_CMAP, zero_contour)
        quiver_map(ax_tos, lon, lat, plot_response[idx, 2], plot_response[idx, 3], tau_ref, arrow_stride, arrow_scale)
        add_map_features(ax_tos, show_xticks=False, show_yticks=col == 0)
        add_layer_label(ax_tos, "TOS + wind")
        ax_tos.set_title(f"({chr(97 + panel_ord)}) Lead {month}: {labels[idx]}", y=1.02, fontsize=7.2, fontweight="bold")
        panel_ord += 1

        zos_mappable = contour_map(ax_zos, lon, lat, plot_response[idx, 1], zos_levels, ZOS_CMAP, zero_contour)
        add_map_features(ax_zos, show_xticks=row == 1, show_yticks=col == 0)
        add_layer_label(ax_zos, "ZOS")

    cax_tos = fig.add_axes([0.105, 0.055, 0.27, 0.020])
    cax_zos = fig.add_axes([0.50, 0.055, 0.34, 0.020])
    cb1 = fig.colorbar(tos_mappable, cax=cax_tos, orientation="horizontal")
    cb1.set_label("TOS response")
    cb2 = fig.colorbar(zos_mappable, cax=cax_zos, orientation="horizontal")
    cb2.set_label("ZOS response")
    fig.text(0.86, 0.063, "vectors: wind stress response", fontsize=7.2, color="#263238")

    suffix = f", {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"CNOP response evolution: {source} {year}, candidate rank {rank}{suffix}",
        fontsize=9.0,
        fontweight="bold",
        y=0.985,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_paper_response_{source}_{year}_rank{rank}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    global HAS_CARTOPY
    args = parse_args()
    if args.no_cartopy:
        HAS_CARTOPY = False
    set_style()
    months = parse_months(args.months)
    ranks = parse_ranks(args.candidate_ranks, args.candidate_rank)
    output_dir = args.output_dir or args.cnop_dir / "figures"
    forecast_climatology_by_source: dict[int, np.ndarray] = {}
    if args.comparison_anomaly and not args.skip_comparison:
        if not args.case_source:
            raise ValueError("--case-source is required with --comparison-anomaly")
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        config = load_config(args.config)
        dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
        model, _checkpoint = load_model(config, args.checkpoint, device)
        source_idx = dataset.source_names.index(args.case_source)
        cache_path = args.forecast_climatology_cache
        if cache_path is None:
            cache_path = args.cnop_dir / f"forecast_field_climatology_{args.forecast_climatology}_h{args.horizon}.npz"
        forecast_climatology_by_source = load_or_compute_forecast_field_climatology(
            model,
            dataset,
            [source_idx],
            args.horizon,
            int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon)),
            device,
            args.forecast_climatology,
            args.split,
            args.climatology_batch_size,
            cache_path,
        )
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    products_by_rank: list[tuple[int, CaseProducts]] = []
    for rank in ranks:
        products = build_case_products(
            args.config,
            args.checkpoint,
            args.cnop_dir,
            args.case_source,
            args.case_year,
            rank,
            args.split,
            args.device,
            args.horizon,
            args.trained_rollout_steps,
        )
        products_by_rank.append((rank, products))
        forecast_field_climatology = forecast_climatology_by_source.get(products.source_idx)
        if not args.skip_perturbation:
            path = plot_perturbation_figure(
                products.perturbation,
                products.lat,
                products.lon,
                products.source,
                products.year,
                rank,
                output_dir,
                args.dpi,
                args.smooth_sigma,
                args.perturb_tos_vmax,
                args.perturb_zos_vmax,
                args.contour_levels,
                args.zero_contour,
                args.constraint_label,
            )
            print(f"wrote {path}")
        if not args.skip_comparison:
            path = plot_comparison_figure(
                products.truth,
                products.baseline,
                products.perturbed,
                products.response,
                products.observed_field_climatology,
                forecast_field_climatology,
                products.target_months,
                products.lat,
                products.lon,
                products.labels,
                products.source,
                products.year,
                rank,
                args.summary_month,
                output_dir,
                args.dpi,
                args.smooth_sigma,
                args.tos_vmax,
                args.zos_vmax,
                args.contour_levels,
                args.zero_contour,
                args.comparison_anomaly,
            )
            print(f"wrote {path}")
        if not args.skip_response:
            path = plot_paper_figure(
                products.response,
                products.lat,
                products.lon,
                products.labels,
                products.source,
                products.year,
                rank,
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
                args.contour_levels,
                args.zero_contour,
                args.title_suffix,
            )
            print(f"wrote {path}")
    if len(products_by_rank) > 1 and not args.skip_perturbation and not args.skip_multi_perturbation:
        path = plot_multi_perturbation_figure(
            products_by_rank,
            output_dir,
            args.dpi,
            args.smooth_sigma,
            args.perturb_tos_vmax,
            args.perturb_zos_vmax,
            args.contour_levels,
            args.zero_contour,
            args.constraint_label,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
