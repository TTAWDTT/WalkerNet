"""绘制 Historical 三海盆 CNOP 对照实验的论文图版。

输出包括初始扰动、三组逐月响应、三组 lead-12 对比，以及
Niño3.4/随机扰动统计总结。所有空间图采用同一全球网格和跨海盆统一色标。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import compute_nino34_numpy  # noqa: E402
from scripts.cnop.plot_cnop_monthly_response import (  # noqa: E402
    apply_delta,
    load_case_npz,
    load_model,
    make_case,
    make_case_input,
    read_case_from_summary,
    rollout_fields,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from cartopy.util import add_cyclic_point
except ImportError as exc:  # pragma: no cover - server plotting dependency
    raise RuntimeError("plot_basin_cnop_experiment.py requires cartopy") from exc

try:
    from scipy.ndimage import gaussian_filter
except ImportError as exc:  # pragma: no cover - server plotting dependency
    raise RuntimeError("plot_basin_cnop_experiment.py requires scipy") from exc


DOMAINS = ("pacific", "atlantic_indian", "global")
DOMAIN_LABELS = {
    "pacific": "Pacific only",
    "atlantic_indian": "Atlantic + Indian",
    "global": "Global ocean",
}
DOMAIN_COLORS = {
    "pacific": "#D55E00",
    "atlantic_indian": "#009E73",
    "global": "#0072B2",
}
NINO34_BOX = (190.0, 240.0, -5.0, 5.0)
MAP_EXTENT = (0.0, 359.999, -60.0, 60.0)
DATA_CRS = ccrs.PlateCarree()
MAP_CRS = ccrs.PlateCarree(central_longitude=180.0)


@dataclass
class BasinProducts:
    domain: str
    delta_phys: np.ndarray
    perturbed: np.ndarray
    response: np.ndarray
    nino_perturbed: np.ndarray
    objective: float
    constraint_radius: float
    constraint_ratio: float
    random_objectives: np.ndarray


@dataclass
class ExperimentProducts:
    source: str
    year: int
    lat: np.ndarray
    lon: np.ndarray
    valid_mask: np.ndarray
    months: np.ndarray
    truth: np.ndarray
    baseline: np.ndarray
    nino_truth: np.ndarray
    nino_baseline: np.ndarray
    basins: dict[str, BasinProducts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the three-basin Historical CNOP experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--forecast-climatology-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lead-month", type=int, default=12)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--vector-sigma", type=float, default=3.0)
    parser.add_argument("--monthly-arrow-stride", type=int, default=24)
    parser.add_argument("--comparison-arrow-stride", type=int, default=14)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def set_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_forecast_climatology(path: Path, source_idx: int) -> np.ndarray:
    with np.load(path) as data:
        source_indices = [int(item) for item in np.asarray(data["source_indices"]).tolist()]
        if source_idx not in source_indices:
            raise ValueError(f"source_idx={source_idx} is absent from {path}")
        return np.asarray(data["climatology"][source_indices.index(source_idx)], dtype=np.float32)


def monthly_observed_tos_climatology(dataset: WalkerDataset, source_idx: int) -> np.ndarray:
    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    start_year, end_year = dataset.data_config["train_years"]
    result = np.full((13, *payload["data"].shape[-2:]), np.nan, dtype=np.float32)
    for month in range(1, 13):
        selected = (years >= int(start_year)) & (years <= int(end_year)) & (months == month)
        result[month] = np.nanmean(np.asarray(payload["data"][selected, 0], dtype=np.float32), axis=0)
    return result


def nino_truth_series(
    truth_tos: np.ndarray,
    observed_climatology: np.ndarray,
    months: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    values = [
        float(compute_nino34_numpy((truth_tos[idx] - observed_climatology[int(month)])[None], lat, lon)[0])
        for idx, month in enumerate(months)
    ]
    return np.asarray(values, dtype=np.float32)


def nino_forecast_series(
    forecast_tos: np.ndarray,
    forecast_climatology: np.ndarray,
    months: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    values = [
        float(
            compute_nino34_numpy(
                (forecast_tos[idx] - forecast_climatology[idx, int(month)])[None], lat, lon
            )[0]
        )
        for idx, month in enumerate(months)
    ]
    return np.asarray(values, dtype=np.float32)


def ocean_smooth(field: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    """经度循环的掩膜归一化高斯平滑，避免海陆边界和日期变更线断裂。"""

    field = np.asarray(field, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(field)
    if sigma <= 0:
        return np.where(valid, field, np.nan)
    numerator = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma, mode=("nearest", "wrap"))
    denominator = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode=("nearest", "wrap"))
    smoothed = numerator / np.maximum(denominator, 1.0e-6)
    return np.where(valid_mask, smoothed, np.nan)


def masked_percentile(values: list[np.ndarray], percentile: float, minimum: float) -> float:
    finite = np.concatenate([np.abs(item[np.isfinite(item)]).ravel() for item in values if np.isfinite(item).any()])
    return max(float(np.percentile(finite, percentile)), minimum)


def field_range(values: list[np.ndarray], low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = np.concatenate([item[np.isfinite(item)].ravel() for item in values if np.isfinite(item).any()])
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def add_cyclic(field: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cyclic_field, cyclic_lon = add_cyclic_point(field, coord=lon, axis=-1)
    return np.asarray(cyclic_field), np.asarray(cyclic_lon)


def setup_map(ax: plt.Axes, show_x: bool, show_y: bool, nino_box: bool = True) -> None:
    ax.set_extent(MAP_EXTENT, crs=DATA_CRS)
    ax.set_facecolor("#F7FAFC")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#E8E5DE", edgecolor="none", zorder=8)
    ax.coastlines(resolution="110m", color="#4B5563", linewidth=0.32, zorder=9)
    x_ticks = [0, 60, 120, 180, 240, 300]
    y_ticks = [-60, -30, 0, 30, 60]
    ax.set_xticks(x_ticks, crs=DATA_CRS)
    ax.set_yticks(y_ticks, crs=DATA_CRS)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True, dateline_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    ax.gridlines(
        crs=DATA_CRS,
        xlocs=x_ticks,
        ylocs=y_ticks,
        linewidth=0.28,
        color="#64748B",
        alpha=0.28,
        linestyle=":",
    )
    if nino_box:
        lon0, lon1, lat0, lat1 = NINO34_BOX
        ax.plot(
            [lon0, lon1, lon1, lon0, lon0],
            [lat0, lat0, lat1, lat1, lat0],
            transform=DATA_CRS,
            color="#111827",
            linewidth=0.55,
            zorder=10,
        )


def add_domain_boundary(ax: plt.Axes, domain: str) -> None:
    if domain == "global":
        return
    for longitude in (120.0, 290.0):
        ax.plot(
            [longitude, longitude],
            [-60.0, 60.0],
            transform=DATA_CRS,
            color="#374151",
            linewidth=0.55,
            linestyle="--",
            alpha=0.8,
            zorder=10,
        )


def scalar_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    cmap: str,
    levels: np.ndarray,
    show_x: bool,
    show_y: bool,
    nino_box: bool = True,
):
    cyclic_field, cyclic_lon = add_cyclic(field, lon)
    mappable = ax.contourf(
        cyclic_lon,
        lat,
        cyclic_field,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=DATA_CRS,
        antialiased=True,
    )
    setup_map(ax, show_x, show_y, nino_box=nino_box)
    return mappable


def quiver_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    stride: int,
    reference: float,
    color: str = "#263238",
):
    lat_idx = np.arange(0, len(lat), max(1, stride))
    lon_idx = np.arange(0, len(lon), max(1, stride))
    lon2, lat2 = np.meshgrid(lon, lat)
    u_sub = u[np.ix_(lat_idx, lon_idx)]
    v_sub = v[np.ix_(lat_idx, lon_idx)]
    return ax.quiver(
        lon2[np.ix_(lat_idx, lon_idx)],
        lat2[np.ix_(lat_idx, lon_idx)],
        np.ma.masked_invalid(u_sub),
        np.ma.masked_invalid(v_sub),
        transform=DATA_CRS,
        color=color,
        width=0.0020,
        headwidth=3.1,
        headlength=3.8,
        headaxislength=3.3,
        scale=max(reference / 14.0, 1.0e-8),
        scale_units="xy",
        angles="xy",
        alpha=0.88,
        zorder=7,
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    print(f"[figure] {png_path}", flush=True)
    return png_path


def build_products(args: argparse.Namespace) -> ExperimentProducts:
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", args.horizon))

    first_dir = args.experiment_dir / "combined" / DOMAINS[0]
    source, year, target_t, observed = read_case_from_summary(first_dir, "", 0)
    case = make_case(dataset, source, year, target_t, observed)
    payload = dataset.source_payloads[case.source_idx]
    lat = np.asarray(payload["lat"], dtype=np.float64)
    lon = np.mod(np.asarray(payload["lon"], dtype=np.float64), 360.0)
    sort_idx = np.argsort(lon)
    lon = lon[sort_idx]
    valid_mask = payload["valid_mask"].detach().cpu().numpy().astype(bool)[..., sort_idx]
    months = np.asarray(payload["months"][target_t : target_t + args.horizon], dtype=np.int64)
    truth = np.asarray(payload["data"][target_t : target_t + args.horizon], dtype=np.float32)[..., sort_idx]

    x0 = make_case_input(dataset, case, device)
    baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()[..., sort_idx]
    forecast_clim = load_forecast_climatology(args.forecast_climatology_cache, case.source_idx)[..., sort_idx]
    observed_clim = monthly_observed_tos_climatology(dataset, case.source_idx)[..., sort_idx]
    nino_truth = nino_truth_series(truth[:, 0], observed_clim, months, lat, lon)
    nino_baseline = nino_forecast_series(baseline[:, 0], forecast_clim, months, lat, lon)

    basins: dict[str, BasinProducts] = {}
    for domain in DOMAINS:
        cnop_dir = args.experiment_dir / "combined" / domain
        delta_norm, _ = load_case_npz(cnop_dir, source, year, args.candidate_rank)
        delta_tensor = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
        x_perturbed = apply_delta(x0, delta_tensor, torch.ones_like(delta_tensor, dtype=torch.bool))
        perturbed = rollout_fields(
            model, dataset, case, x_perturbed, args.horizon, trained_rollout_steps
        ).numpy()[..., sort_idx]
        delta_phys = (
            dataset.denormalize(x_perturbed)[:, -1, :2] - dataset.denormalize(x0)[:, -1, :2]
        )[0].detach().cpu().numpy()[..., sort_idx]
        summary = read_csv_rows(cnop_dir / "cnop_summary.csv")[0]
        random_rows = read_csv_rows(args.experiment_dir / "random_controls" / f"{domain}.csv")
        basins[domain] = BasinProducts(
            domain=domain,
            delta_phys=delta_phys,
            perturbed=perturbed,
            response=perturbed - baseline,
            nino_perturbed=nino_forecast_series(perturbed[:, 0], forecast_clim, months, lat, lon),
            objective=float(summary["best_objective"]),
            constraint_radius=float(summary["constraint_radius"]),
            constraint_ratio=float(summary["constraint_ratio"]),
            random_objectives=np.asarray([float(row["objective"]) for row in random_rows], dtype=np.float32),
        )

    print(
        f"[data] checkpoint_epoch={checkpoint.get('epoch')} source={source} year={year} "
        f"domains={','.join(DOMAINS)}",
        flush=True,
    )
    return ExperimentProducts(
        source=source,
        year=year,
        lat=lat,
        lon=lon,
        valid_mask=valid_mask,
        months=months,
        truth=truth,
        baseline=baseline,
        nino_truth=nino_truth,
        nino_baseline=nino_baseline,
        basins=basins,
    )


def smoothed_delta(products: ExperimentProducts, basin: BasinProducts, sigma: float) -> np.ndarray:
    fields = []
    for var_idx in range(2):
        smoothed = ocean_smooth(basin.delta_phys[var_idx], products.valid_mask[var_idx], sigma)
        support = np.abs(basin.delta_phys[var_idx]) > 1.0e-12
        fields.append(np.where(products.valid_mask[var_idx], np.where(support, smoothed, 0.0), np.nan))
    return np.stack(fields)


def smoothed_response(
    products: ExperimentProducts,
    basin: BasinProducts,
    scalar_sigma: float,
    vector_sigma: float,
) -> np.ndarray:
    result = np.empty_like(basin.response)
    for lead_idx in range(basin.response.shape[0]):
        for var_idx in range(basin.response.shape[1]):
            sigma = vector_sigma if var_idx in (2, 3) else scalar_sigma
            result[lead_idx, var_idx] = ocean_smooth(
                basin.response[lead_idx, var_idx], products.valid_mask[var_idx], sigma
            )
    return result


def plot_initial_overview(
    products: ExperimentProducts,
    deltas: dict[str, np.ndarray],
    output_dir: Path,
    dpi: int,
) -> Path:
    tos_limit = masked_percentile([deltas[item][0] for item in DOMAINS], 99.0, 1.0e-4)
    zos_limit = masked_percentile([deltas[item][1] for item in DOMAINS], 99.0, 1.0e-5)
    tos_levels = np.linspace(-tos_limit, tos_limit, 35)
    zos_levels = np.linspace(-zos_limit, zos_limit, 35)
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 7.1), subplot_kw={"projection": MAP_CRS})
    fig.subplots_adjust(left=0.055, right=0.94, top=0.91, bottom=0.12, wspace=0.035, hspace=0.12)
    m_tos = None
    m_zos = None
    for row, domain in enumerate(DOMAINS):
        m_tos = scalar_map(
            axes[row, 0], products.lon, products.lat, deltas[domain][0], "RdBu_r", tos_levels,
            show_x=row == 2, show_y=True, nino_box=True,
        )
        m_zos = scalar_map(
            axes[row, 1], products.lon, products.lat, deltas[domain][1], "BrBG", zos_levels,
            show_x=row == 2, show_y=False, nino_box=True,
        )
        add_domain_boundary(axes[row, 0], domain)
        add_domain_boundary(axes[row, 1], domain)
        axes[row, 0].text(
            -0.12, 0.5, DOMAIN_LABELS[domain], transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=9, fontweight="bold",
        )
        if row == 0:
            axes[row, 0].set_title("Initial CNOP: TOS perturbation")
            axes[row, 1].set_title("Initial CNOP: ZOS perturbation")
    assert m_tos is not None and m_zos is not None
    cax0 = fig.add_axes([0.13, 0.065, 0.31, 0.018])
    cax1 = fig.add_axes([0.56, 0.065, 0.31, 0.018])
    fig.colorbar(m_tos, cax=cax0, orientation="horizontal").set_label("TOS perturbation")
    fig.colorbar(m_zos, cax=cax1, orientation="horizontal").set_label("ZOS perturbation")
    fig.suptitle(
        f"Initial CNOP structures under basin-specific 0.1 C_D constraints | {products.source} {products.year}",
        fontsize=12, fontweight="bold", y=0.97,
    )
    return save_figure(fig, output_dir, "fig01_initial_cnop_domains", dpi)


def plot_monthly_response(
    products: ExperimentProducts,
    domain: str,
    response: np.ndarray,
    tos_limit: float,
    zos_limit: float,
    tau_reference: float,
    arrow_stride: int,
    output_dir: Path,
    figure_number: int,
    dpi: int,
) -> Path:
    tos_levels = np.linspace(-tos_limit, tos_limit, 33)
    zos_levels = np.linspace(-zos_limit, zos_limit, 33)
    fig, axes = plt.subplots(2, 12, figsize=(24.0, 3.15), subplot_kw={"projection": MAP_CRS})
    fig.subplots_adjust(left=0.035, right=0.985, top=0.78, bottom=0.27, wspace=0.035, hspace=0.08)
    month_names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    m_tos = None
    m_zos = None
    for idx in range(12):
        m_tos = scalar_map(
            axes[0, idx], products.lon, products.lat, response[idx, 0], "RdBu_r", tos_levels,
            show_x=False, show_y=idx == 0,
        )
        quiver_map(
            axes[0, idx], products.lon, products.lat, response[idx, 2], response[idx, 3],
            arrow_stride, tau_reference,
        )
        m_zos = scalar_map(
            axes[1, idx], products.lon, products.lat, response[idx, 1], "BrBG", zos_levels,
            show_x=True, show_y=idx == 0,
        )
        axes[0, idx].set_title(f"L{idx + 1}  {month_names[int(products.months[idx]) - 1]}", pad=2)
    axes[0, 0].text(-0.30, 0.5, "TOS + wind", transform=axes[0, 0].transAxes, rotation=90, va="center", fontweight="bold")
    axes[1, 0].text(-0.30, 0.5, "ZOS", transform=axes[1, 0].transAxes, rotation=90, va="center", fontweight="bold")
    assert m_tos is not None and m_zos is not None
    cax0 = fig.add_axes([0.20, 0.105, 0.25, 0.023])
    cax1 = fig.add_axes([0.57, 0.105, 0.25, 0.023])
    fig.colorbar(
        m_tos, cax=cax0, orientation="horizontal", ticks=np.linspace(-tos_limit, tos_limit, 5)
    ).set_label("TOS response (perturbed - baseline)")
    fig.colorbar(
        m_zos, cax=cax1, orientation="horizontal", ticks=np.linspace(-zos_limit, zos_limit, 5)
    ).set_label("ZOS response (perturbed - baseline)")
    fig.text(0.84, 0.11, "Arrows: wind-stress response", fontsize=7.5, color="#374151")
    fig.suptitle(
        f"Monthly CNOP-induced response | {DOMAIN_LABELS[domain]} | {products.source} {products.year}",
        fontsize=12, fontweight="bold", y=0.96,
    )
    return save_figure(fig, output_dir, f"fig{figure_number:02d}_monthly_response_{domain}", dpi)


def annotation(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.018, 0.94, text, transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5}, zorder=12,
    )


def plot_lead_comparison(
    products: ExperimentProducts,
    domain: str,
    response: np.ndarray,
    lead_idx: int,
    absolute_tos_range: tuple[float, float],
    absolute_zos_range: tuple[float, float],
    tos_diff_limit: float,
    zos_diff_limit: float,
    absolute_tau_reference: float,
    response_tau_reference: float,
    arrow_stride: int,
    output_dir: Path,
    figure_number: int,
    dpi: int,
) -> Path:
    basin = products.basins[domain]
    fields = (products.truth, products.baseline, basin.perturbed, basin.response)
    titles = ("Truth", "Baseline forecast", "CNOP-perturbed forecast", "Perturbed - baseline")
    tos_abs_levels = np.linspace(*absolute_tos_range, 35)
    zos_abs_levels = np.linspace(*absolute_zos_range, 35)
    tos_diff_levels = np.linspace(-tos_diff_limit, tos_diff_limit, 33)
    zos_diff_levels = np.linspace(-zos_diff_limit, zos_diff_limit, 33)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 4.45), subplot_kw={"projection": MAP_CRS})
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.24, wspace=0.035, hspace=0.08)
    m_tos_abs = None
    m_zos_abs = None
    m_tos_diff = None
    m_zos_diff = None
    nino_values = (
        products.nino_truth[lead_idx],
        products.nino_baseline[lead_idx],
        basin.nino_perturbed[lead_idx],
        basin.nino_perturbed[lead_idx] - products.nino_baseline[lead_idx],
    )
    for col, data in enumerate(fields):
        if col < 3:
            tos_field = ocean_smooth(data[lead_idx, 0], products.valid_mask[0], 2.0)
            zos_field = ocean_smooth(data[lead_idx, 1], products.valid_mask[1], 2.0)
            u_field = ocean_smooth(data[lead_idx, 2], products.valid_mask[2], 3.0)
            v_field = ocean_smooth(data[lead_idx, 3], products.valid_mask[3], 3.0)
            m_tos_abs = scalar_map(
                axes[0, col], products.lon, products.lat, tos_field, "Spectral_r", tos_abs_levels,
                show_x=False, show_y=col == 0,
            )
            m_zos_abs = scalar_map(
                axes[1, col], products.lon, products.lat, zos_field, "viridis", zos_abs_levels,
                show_x=True, show_y=col == 0,
            )
            quiver_map(
                axes[0, col], products.lon, products.lat, u_field, v_field,
                arrow_stride, absolute_tau_reference,
            )
        else:
            m_tos_diff = scalar_map(
                axes[0, col], products.lon, products.lat, response[lead_idx, 0], "RdBu_r", tos_diff_levels,
                show_x=False, show_y=False,
            )
            m_zos_diff = scalar_map(
                axes[1, col], products.lon, products.lat, response[lead_idx, 1], "BrBG", zos_diff_levels,
                show_x=True, show_y=False,
            )
            quiver_map(
                axes[0, col], products.lon, products.lat, response[lead_idx, 2], response[lead_idx, 3],
                arrow_stride, response_tau_reference,
            )
        axes[0, col].set_title(titles[col], pad=3)
        label = f"Niño3.4 = {nino_values[col]:+.3f}" if col < 3 else f"ΔNiño3.4 = {nino_values[col]:+.3f}"
        annotation(axes[0, col], label)
    axes[0, 0].text(-0.16, 0.5, "TOS + wind", transform=axes[0, 0].transAxes, rotation=90, va="center", fontweight="bold")
    axes[1, 0].text(-0.16, 0.5, "ZOS", transform=axes[1, 0].transAxes, rotation=90, va="center", fontweight="bold")
    assert all(item is not None for item in (m_tos_abs, m_zos_abs, m_tos_diff, m_zos_diff))
    caxes = [
        fig.add_axes([0.08, 0.085, 0.18, 0.020]),
        fig.add_axes([0.31, 0.085, 0.18, 0.020]),
        fig.add_axes([0.54, 0.085, 0.18, 0.020]),
        fig.add_axes([0.77, 0.085, 0.18, 0.020]),
    ]
    fig.colorbar(
        m_tos_abs, cax=caxes[0], orientation="horizontal", ticks=np.linspace(*absolute_tos_range, 5)
    ).set_label("TOS")
    fig.colorbar(
        m_zos_abs, cax=caxes[1], orientation="horizontal", ticks=np.linspace(*absolute_zos_range, 5)
    ).set_label("ZOS")
    fig.colorbar(
        m_tos_diff, cax=caxes[2], orientation="horizontal",
        ticks=np.linspace(-tos_diff_limit, tos_diff_limit, 5),
    ).set_label("TOS difference")
    fig.colorbar(
        m_zos_diff, cax=caxes[3], orientation="horizontal",
        ticks=np.linspace(-zos_diff_limit, zos_diff_limit, 5),
    ).set_label("ZOS difference")
    fig.suptitle(
        f"Lead {lead_idx + 1} forecast comparison | {DOMAIN_LABELS[domain]} | {products.source} {products.year}",
        fontsize=12, fontweight="bold", y=0.96,
    )
    return save_figure(fig, output_dir, f"fig{figure_number:02d}_lead12_comparison_{domain}", dpi)


def plot_statistical_summary(products: ExperimentProducts, output_dir: Path, dpi: int) -> Path:
    fig = plt.figure(figsize=(12.8, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.75, 1.0), hspace=0.50, wspace=0.28)
    ax_curve = fig.add_subplot(gs[:, 0])
    leads = np.arange(1, 13)
    ax_curve.plot(leads, products.nino_truth, color="#111827", linestyle="--", marker="o", label="Truth")
    ax_curve.plot(leads, products.nino_baseline, color="#8C8C8C", marker="s", label="Baseline")
    for domain in DOMAINS:
        ax_curve.plot(
            leads, products.basins[domain].nino_perturbed, color=DOMAIN_COLORS[domain],
            marker="o", markevery=2, label=DOMAIN_LABELS[domain],
        )
    ax_curve.axhline(0.5, color="#D55E00", linewidth=0.8, linestyle=":")
    ax_curve.axhline(-0.5, color="#0072B2", linewidth=0.8, linestyle=":")
    ax_curve.axhspan(-0.5, 0.5, color="#64748B", alpha=0.055)
    ax_curve.set_xlim(1, 12)
    ax_curve.set_xticks(leads)
    ax_curve.set_xlabel("Forecast lead (month)")
    ax_curve.set_ylabel("Niño3.4 anomaly")
    ax_curve.set_title("Niño3.4 evolution")
    ax_curve.grid(True, color="#94A3B8", alpha=0.22, linewidth=0.6)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(ncol=2, loc="best")

    rng = np.random.default_rng(42)
    for row, domain in enumerate(DOMAINS):
        ax = fig.add_subplot(gs[row, 1])
        basin = products.basins[domain]
        random_values = basin.random_objectives
        jitter = rng.uniform(-0.08, 0.08, size=len(random_values))
        parts = ax.violinplot(random_values, positions=[0.0], widths=0.50, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(DOMAIN_COLORS[domain])
            body.set_edgecolor("none")
            body.set_alpha(0.20)
        ax.scatter(jitter, random_values, s=7, color=DOMAIN_COLORS[domain], alpha=0.28, linewidths=0)
        ax.scatter([0.0], [basin.objective], marker="*", s=95, color="#111827", zorder=5, label="CNOP")
        p95 = float(np.percentile(random_values, 95))
        ax.axhline(p95, color=DOMAIN_COLORS[domain], linewidth=1.0, linestyle="--")
        empirical_p = (1 + int(np.sum(random_values >= basin.objective))) / (len(random_values) + 1)
        ax.set_xlim(-0.42, 0.42)
        ax.set_xticks([])
        ax.set_ylabel("Objective")
        ax.set_title(
            f"{DOMAIN_LABELS[domain]}  |  R={basin.constraint_radius:.3f}  |  "
            f"p ≤ {empirical_p:.4f}  |  ||δ||/R={basin.constraint_ratio:.6f}",
            fontsize=8.2,
        )
        ax.grid(axis="y", color="#94A3B8", alpha=0.20, linewidth=0.5)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.text(0.39, p95, "random P95", ha="right", va="bottom", fontsize=6.5, color=DOMAIN_COLORS[domain])
    fig.suptitle(
        f"CNOP forecast impact and equal-radius random controls | {products.source} {products.year}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12)
    return save_figure(fig, output_dir, "fig08_nino_random_summary", dpi)


def main() -> None:
    args = parse_args()
    if args.horizon != 12:
        raise ValueError("This paper layout requires --horizon 12")
    set_style(args.dpi)
    products = build_products(args)
    deltas = {
        domain: smoothed_delta(products, products.basins[domain], args.smooth_sigma)
        for domain in DOMAINS
    }
    responses = {
        domain: smoothed_response(
            products, products.basins[domain], args.smooth_sigma, args.vector_sigma
        )
        for domain in DOMAINS
    }

    tos_response_limit = masked_percentile(
        [responses[domain][:, 0] for domain in DOMAINS], 99.0, 1.0e-4
    )
    zos_response_limit = masked_percentile(
        [responses[domain][:, 1] for domain in DOMAINS], 99.0, 1.0e-5
    )
    tau_response_reference = masked_percentile(
        [np.hypot(responses[domain][:, 2], responses[domain][:, 3]) for domain in DOMAINS],
        95.0,
        1.0e-6,
    )
    lead_idx = min(max(args.lead_month, 1), args.horizon) - 1
    physical_fields = [products.truth[lead_idx], products.baseline[lead_idx]] + [
        products.basins[domain].perturbed[lead_idx] for domain in DOMAINS
    ]
    tos_absolute_range = field_range(
        [ocean_smooth(field[0], products.valid_mask[0], args.smooth_sigma) for field in physical_fields],
        1.0,
        99.0,
    )
    zos_absolute_range = field_range(
        [ocean_smooth(field[1], products.valid_mask[1], args.smooth_sigma) for field in physical_fields],
        1.0,
        99.0,
    )
    tau_absolute_reference = masked_percentile(
        [np.hypot(field[2], field[3]) for field in physical_fields], 95.0, 1.0e-6
    )

    generated: list[Path] = []
    generated.append(plot_initial_overview(products, deltas, args.output_dir, args.dpi))
    for figure_number, domain in enumerate(DOMAINS, start=2):
        generated.append(
            plot_monthly_response(
                products,
                domain,
                responses[domain],
                tos_response_limit,
                zos_response_limit,
                tau_response_reference,
                args.monthly_arrow_stride,
                args.output_dir,
                figure_number,
                args.dpi,
            )
        )
    for figure_number, domain in enumerate(DOMAINS, start=5):
        generated.append(
            plot_lead_comparison(
                products,
                domain,
                responses[domain],
                lead_idx,
                tos_absolute_range,
                zos_absolute_range,
                tos_response_limit,
                zos_response_limit,
                tau_absolute_reference,
                tau_response_reference,
                args.comparison_arrow_stride,
                args.output_dir,
                figure_number,
                args.dpi,
            )
        )
    generated.append(plot_statistical_summary(products, args.output_dir, args.dpi))

    manifest: dict[str, Any] = {
        "source": products.source,
        "target_year": products.year,
        "checkpoint": str(args.checkpoint),
        "experiment_dir": str(args.experiment_dir),
        "smooth_sigma": args.smooth_sigma,
        "vector_sigma": args.vector_sigma,
        "shared_limits": {
            "tos_response": tos_response_limit,
            "zos_response": zos_response_limit,
            "tos_absolute": tos_absolute_range,
            "zos_absolute": zos_absolute_range,
        },
        "domains": {
            domain: {
                "objective": products.basins[domain].objective,
                "constraint_radius": products.basins[domain].constraint_radius,
                "constraint_ratio": products.basins[domain].constraint_ratio,
                "random_max": float(products.basins[domain].random_objectives.max()),
                "lead12_nino": float(products.basins[domain].nino_perturbed[lead_idx]),
            }
            for domain in DOMAINS
        },
        "figures": [path.name for path in generated],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] generated {len(generated)} figures in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
