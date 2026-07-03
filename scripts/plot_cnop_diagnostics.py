"""绘制 TOS/ZOS CNOP 的合成诊断图与前兆分析图。

输入目录需要包含 ``cnop_summary.csv`` 和 ``case_*.npz``。脚本不会重新计算
CNOP，只负责把已有结果整理成更适合论文/汇报展示的图。
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


VARIABLE_NAMES = ("TOS", "ZOS")
NINO34_BOX = (190.0, 240.0, -5.0, 5.0)
PERTURB_BOX = (120.0, 290.0, -20.0, 20.0)
MAP_VIEW_BOX = (90.0, 320.0, -35.0, 35.0)


@dataclass(frozen=True)
class CaseData:
    """单个 CNOP case 的图形诊断所需数据。"""

    source: str
    year: int
    baseline_max_3m: float
    cnop_max_3m: float
    gain_max_3m: float
    delta_phys: np.ndarray
    baseline_nino: np.ndarray
    cnop_nino: np.ndarray
    baseline_3m: np.ndarray
    cnop_3m: np.ndarray
    top_delta_phys: np.ndarray | None = None
    top_cnop_3m: np.ndarray | None = None
    top_gain_max_3m: np.ndarray | None = None
    top_start_idx: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WalkerNet TOS/ZOS CNOP diagnostics.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/cnop_tos_zos_patch_0703"),
        help="目录内应包含 cnop_summary.csv 与 case_*.npz。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写回 input-dir/figures。",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def set_plot_style() -> None:
    """设置稳重、紧凑的科研绘图风格。"""

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "grid.color": "#D5D9E2",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
        }
    )


def read_summary(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """读取 cnop_summary.csv，用 source/year 连接 npz 文件。"""

    rows: dict[tuple[str, int], dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], int(row["target_year"]))
            rows[key] = {
                "baseline_max_3m": float(row["baseline_max_3m"]),
                "cnop_max_3m": float(row["cnop_max_3m"]),
                "gain_max_3m": float(row["gain_max_3m"]),
            }
    return rows


def parse_case_name(path: Path) -> tuple[str, int]:
    """从 case_EC-Earth3_1959.npz 这类文件名中解析 source/year。"""

    match = re.fullmatch(r"case_(.+)_(\d{4})\.npz", path.name)
    if match is None:
        raise ValueError(f"无法解析 case 文件名：{path.name}")
    return match.group(1), int(match.group(2))


def load_cases(input_dir: Path) -> tuple[list[CaseData], np.ndarray, np.ndarray]:
    summary = read_summary(input_dir / "cnop_summary.csv")
    cases: list[CaseData] = []
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None

    for path in sorted(input_dir.glob("case_*.npz")):
        source, year = parse_case_name(path)
        metrics = summary[(source, year)]
        data = np.load(path)
        lat = np.asarray(data["lat"], dtype=np.float64)
        lon = np.asarray(data["lon"], dtype=np.float64)
        top_delta_phys = np.asarray(data["top_delta_phys"], dtype=np.float64) if "top_delta_phys" in data.files else None
        top_cnop_3m = np.asarray(data["top_cnop_3m"], dtype=np.float64) if "top_cnop_3m" in data.files else None
        top_gain_max_3m = np.asarray(data["top_gain_max_3m"], dtype=np.float64) if "top_gain_max_3m" in data.files else None
        top_start_idx = np.asarray(data["top_start_idx"], dtype=np.int32) if "top_start_idx" in data.files else None
        cases.append(
            CaseData(
                source=source,
                year=year,
                baseline_max_3m=metrics["baseline_max_3m"],
                cnop_max_3m=metrics["cnop_max_3m"],
                gain_max_3m=metrics["gain_max_3m"],
                delta_phys=np.asarray(data["delta_phys"], dtype=np.float64),
                baseline_nino=np.asarray(data["baseline_nino"], dtype=np.float64),
                cnop_nino=np.asarray(data["cnop_nino"], dtype=np.float64),
                baseline_3m=np.asarray(data["baseline_3m"], dtype=np.float64),
                cnop_3m=np.asarray(data["cnop_3m"], dtype=np.float64),
                top_delta_phys=top_delta_phys,
                top_cnop_3m=top_cnop_3m,
                top_gain_max_3m=top_gain_max_3m,
                top_start_idx=top_start_idx,
            )
        )

    if not cases or lat is None or lon is None:
        raise FileNotFoundError(f"{input_dir} 中没有可用 case_*.npz")
    return cases, lat, lon


def region_mask(lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bounds
    return ((lat[:, None] >= lat_min) & (lat[:, None] <= lat_max)) & (
        (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)
    )


def area_mean(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, bounds: tuple[float, float, float, float]) -> float:
    """对指定经纬度框做 cos(lat) 加权平均。"""

    mask = region_mask(lat, lon, bounds)
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size), dtype=np.float64)
    valid = mask & np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(field[valid] * weights[valid]) / np.nansum(weights[valid]))


def add_region_box(ax: plt.Axes, bounds: tuple[float, float, float, float], color: str, label: str | None = None) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    rect = Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        fill=False,
        edgecolor=color,
        linewidth=1.15,
        linestyle="-",
        zorder=5,
        label=label,
    )
    ax.add_patch(rect)


def setup_map_axis(ax: plt.Axes, *, wide: bool = False) -> None:
    if wide:
        lon_min, lon_max, lat_min, lat_max = MAP_VIEW_BOX
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xticks([100, 140, 180, 220, 260, 300])
        ax.set_yticks([-30, -20, -10, 0, 10, 20, 30])
    else:
        ax.set_xlim(110, 300)
        ax.set_ylim(-25, 25)
        ax.set_xticks([120, 160, 200, 240, 280])
        ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    add_region_box(ax, PERTURB_BOX, "#60646C", "perturbation domain")
    add_region_box(ax, NINO34_BOX, "#111827", "Niño3.4")
    ax.axhline(0, color="#111827", linewidth=0.65, alpha=0.55)


def mask_outside_perturb_domain(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """扰动域外不参与解释，画图时遮掉以减少视觉噪声。"""

    masked = np.array(field, dtype=np.float64, copy=True)
    masked[~region_mask(lat, lon, PERTURB_BOX)] = np.nan
    return masked


def mask_outside_view(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """保留更宽视野内的场，避免画出与太平洋诊断无关的区域。"""

    masked = np.array(field, dtype=np.float64, copy=True)
    masked[~region_mask(lat, lon, MAP_VIEW_BOX)] = np.nan
    return masked


def plot_diverging_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    title: str,
    unit: str,
    cmap: str,
    *,
    mask_domain: bool = True,
    wide: bool = False,
    vmax: float | None = None,
) -> mpl.cm.ScalarMappable:
    field = mask_outside_perturb_domain(field, lat, lon) if mask_domain else mask_outside_view(field, lat, lon)
    color_region = PERTURB_BOX if mask_domain else MAP_VIEW_BOX
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(field[region_mask(lat, lon, color_region)]), 98))
    vmax = max(vmax, 1.0e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, norm=norm)
    setup_map_axis(ax, wide=wide)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.82, pad=0.018)
    cbar.set_label(unit)
    return mesh


def plot_sequential_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    title: str,
    unit: str,
    cmap: str,
    vmax: float | None = None,
) -> mpl.cm.ScalarMappable:
    field = mask_outside_perturb_domain(field, lat, lon)
    if vmax is None:
        vmax = float(np.nanpercentile(field[region_mask(lat, lon, PERTURB_BOX)], 98))
    vmax = max(vmax, 1.0e-6)
    mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    setup_map_axis(ax)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.82, pad=0.018)
    cbar.set_label(unit)
    return mesh


def plot_main_figure(cases: list[CaseData], lat: np.ndarray, lon: np.ndarray, output_dir: Path, dpi: int) -> None:
    deltas = np.stack([case.delta_phys for case in cases], axis=0)
    composite = np.nanmean(deltas, axis=0)

    order = np.argsort([case.gain_max_3m for case in cases])
    sorted_cases = [cases[i] for i in order]
    labels = [f"{case.source}\n{case.year}" for case in sorted_cases]
    gains = np.asarray([case.gain_max_3m for case in sorted_cases])

    months = np.arange(1, 13)
    fig = plt.figure(figsize=(12.2, 8.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 0.95), height_ratios=(1.0, 0.95))
    ax_tos = fig.add_subplot(gs[0, 0])
    ax_zos = fig.add_subplot(gs[1, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_gain = fig.add_subplot(gs[1, 1])

    plot_diverging_map(ax_tos, lon, lat, composite[0], "A  Composite CNOP perturbation: TOS", "TOS perturbation", "RdBu_r")
    plot_diverging_map(ax_zos, lon, lat, composite[1], "B  Composite CNOP perturbation: ZOS", "ZOS perturbation", "BrBG")

    for case in cases:
        ax_curve.plot(months, case.baseline_nino, color="#8792A2", linewidth=0.8, alpha=0.45)
        ax_curve.plot(months, case.cnop_nino, color="#D55E00", linewidth=0.9, alpha=0.55)
    ax_curve.plot(
        months,
        np.mean([case.baseline_nino for case in cases], axis=0),
        color="#4B5563",
        linewidth=2.2,
        label="Baseline mean",
    )
    ax_curve.plot(
        months,
        np.mean([case.cnop_nino for case in cases], axis=0),
        color="#C2410C",
        linewidth=2.4,
        label="CNOP mean",
    )
    ax_curve.axhline(0.5, color="#7F1D1D", linestyle="--", linewidth=1.0, label="0.5 threshold")
    ax_curve.axhline(0, color="#111827", linewidth=0.7, alpha=0.5)
    ax_curve.set_title("C  Niño3.4 response after perturbing previous December", loc="left", pad=7, fontweight="bold")
    ax_curve.set_xlabel("Forecast month")
    ax_curve.set_ylabel("Niño3.4 anomaly")
    ax_curve.set_xticks(months)
    ax_curve.grid(True)
    ax_curve.legend(frameon=False, ncols=2, loc="upper left")

    colors = ["#D55E00" if case.baseline_max_3m >= 0 else "#0072B2" for case in sorted_cases]
    ax_gain.barh(np.arange(len(sorted_cases)), gains, color=colors, alpha=0.88)
    ax_gain.set_yticks(np.arange(len(sorted_cases)))
    ax_gain.set_yticklabels(labels)
    ax_gain.set_xlabel("CNOP gain in max 3-month Niño3.4")
    ax_gain.set_title("D  Case-wise amplification", loc="left", pad=7, fontweight="bold")
    ax_gain.grid(True, axis="x")
    ax_gain.set_axisbelow(True)
    for idx, case in enumerate(sorted_cases):
        ax_gain.text(gains[idx] + 0.035, idx, f"{case.baseline_max_3m:+.2f}→{case.cnop_max_3m:.2f}", va="center", fontsize=7.6)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cnop_composite_diagnostics.{suffix}", dpi=dpi)
    plt.close(fig)


def plot_precursor_figure(cases: list[CaseData], lat: np.ndarray, lon: np.ndarray, output_dir: Path, dpi: int) -> list[dict[str, float | str | int]]:
    region_defs = {
        "Nino34": NINO34_BOX,
        "WestEqPac": (120.0, 160.0, -5.0, 5.0),
        "CentralEastEqPac": (180.0, 260.0, -5.0, 5.0),
        "NorthOffEq": (160.0, 250.0, 5.0, 15.0),
        "SouthOffEq": (160.0, 250.0, -15.0, -5.0),
    }
    rows: list[dict[str, float | str | int]] = []
    for case in cases:
        tos = case.delta_phys[0]
        zos = case.delta_phys[1]
        row: dict[str, float | str | int] = {
            "source": case.source,
            "year": case.year,
            "baseline_max_3m": case.baseline_max_3m,
            "cnop_max_3m": case.cnop_max_3m,
            "gain_max_3m": case.gain_max_3m,
        }
        for name, bounds in region_defs.items():
            row[f"tos_{name}"] = area_mean(tos, lat, lon, bounds)
            row[f"zos_{name}"] = area_mean(zos, lat, lon, bounds)
        row["zos_east_west_tilt"] = float(row["zos_CentralEastEqPac"] - row["zos_WestEqPac"])
        row["tos_eq_gradient"] = float(row["tos_CentralEastEqPac"] - row["tos_WestEqPac"])
        rows.append(row)

    deltas = np.stack([case.delta_phys for case in cases], axis=0)
    mean = np.nanmean(deltas, axis=0)
    std = np.nanstd(deltas, axis=0)
    signal_to_spread = np.divide(np.abs(mean), std + 1.0e-6)
    positive_fraction = np.mean(deltas > 0, axis=0)

    fig = plt.figure(figsize=(12.2, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 0.95))
    ax_consistency = fig.add_subplot(gs[0, 0])
    ax_fraction = fig.add_subplot(gs[1, 0])
    ax_region = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, 1])

    plot_sequential_map(
        ax_consistency,
        lon,
        lat,
        signal_to_spread[0],
        "A  TOS perturbation robustness |mean| / spread",
        "dimensionless",
        "viridis",
        vmax=min(3.0, float(np.nanpercentile(signal_to_spread[0][region_mask(lat, lon, PERTURB_BOX)], 99))),
    )

    plot_diverging_map(
        ax_fraction,
        lon,
        lat,
        positive_fraction[0] - 0.5,
        "B  TOS sign agreement across neutral cases",
        "positive fraction - 0.5",
        "PuOr_r",
    )

    metric_names = ["tos_Nino34", "tos_WestEqPac", "tos_CentralEastEqPac", "zos_east_west_tilt", "tos_eq_gradient"]
    display_names = ["TOS Niño3.4", "TOS west eq.", "TOS central/east eq.", "ZOS east-west tilt", "TOS east-west contrast"]
    values = np.asarray([[float(row[name]) for row in rows] for name in metric_names])
    means = np.nanmean(values, axis=1)
    errs = np.nanstd(values, axis=1)
    bar_colors = ["#C2410C", "#0072B2", "#D55E00", "#009E73", "#6B7280"]
    ax_region.barh(np.arange(len(metric_names)), means, xerr=errs, color=bar_colors, alpha=0.86)
    ax_region.axvline(0, color="#111827", linewidth=0.75)
    ax_region.set_yticks(np.arange(len(metric_names)))
    ax_region.set_yticklabels(display_names)
    ax_region.set_xlabel("Area-mean perturbation")
    ax_region.set_title("C  Regional precursor indices", loc="left", pad=7, fontweight="bold")
    ax_region.grid(True, axis="x")
    ax_region.set_axisbelow(True)

    baseline = np.asarray([case.baseline_max_3m for case in cases])
    gains = np.asarray([case.gain_max_3m for case in cases])
    cnop = np.asarray([case.cnop_max_3m for case in cases])
    sc = ax_scatter.scatter(baseline, gains, c=cnop, cmap="magma", s=64, edgecolor="white", linewidth=0.8)
    for case in cases:
        if case.gain_max_3m > np.nanpercentile(gains, 80) or case.cnop_max_3m > np.nanpercentile(cnop, 80):
            ax_scatter.annotate(f"{case.source} {case.year}", (case.baseline_max_3m, case.gain_max_3m), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax_scatter.axvline(0, color="#111827", linewidth=0.7, alpha=0.5)
    ax_scatter.set_xlabel("Baseline max 3-month Niño3.4")
    ax_scatter.set_ylabel("CNOP gain")
    ax_scatter.set_title("D  Where CNOP amplification is largest", loc="left", pad=7, fontweight="bold")
    ax_scatter.grid(True)
    cbar = plt.colorbar(sc, ax=ax_scatter, pad=0.02)
    cbar.set_label("CNOP max 3-month Niño3.4")

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cnop_precursor_diagnostics.{suffix}", dpi=dpi)
    plt.close(fig)

    return rows


def plot_case_atlas(
    cases: list[CaseData],
    lat: np.ndarray,
    lon: np.ndarray,
    output_dir: Path,
    dpi: int,
    *,
    variable_index: int,
    variable_name: str,
    cmap: str,
) -> None:
    """把每一个 case 的扰动都画出来，并使用统一色标方便横向比较。"""

    sorted_cases = sorted(cases, key=lambda item: item.gain_max_3m, reverse=True)
    fields = np.stack([case.delta_phys[variable_index] for case in sorted_cases], axis=0)
    perturb = region_mask(lat, lon, PERTURB_BOX)
    vmax = float(np.nanpercentile(np.abs(fields[:, perturb]), 98))
    vmax = max(vmax, 1.0e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(5, 2, figsize=(13.0, 14.5), layout="constrained", sharex=True, sharey=True)
    for ax, case, field in zip(axes.ravel(), sorted_cases, fields, strict=True):
        field = mask_outside_view(field, lat, lon)
        mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, norm=norm)
        setup_map_axis(ax, wide=True)
        ax.set_title(
            f"{case.source} {case.year}   gain {case.gain_max_3m:.2f}   {case.baseline_max_3m:+.2f}->{case.cnop_max_3m:.2f}",
            loc="left",
            pad=5,
            fontsize=8.8,
            fontweight="bold",
        )

    fig.suptitle(
        f"{variable_name} CNOP perturbation atlas: all neutral cases",
        fontsize=14,
        fontweight="bold",
    )
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.72, pad=0.012)
    cbar.set_label(f"{variable_name} perturbation, shared scale")

    stem = f"cnop_{variable_name.lower()}_case_atlas"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi)
    plt.close(fig)


def plot_factor_comparison(rows: list[dict[str, float | str | int]], output_dir: Path, dpi: int) -> None:
    """用矩阵图比较每个 case 的扰动因子。"""

    sorted_rows = sorted(rows, key=lambda item: float(item["gain_max_3m"]), reverse=True)
    factor_keys = ["tos_Nino34", "tos_WestEqPac", "tos_CentralEastEqPac", "tos_eq_gradient", "zos_east_west_tilt"]
    factor_labels = ["TOS\nNiño3.4", "TOS\nwest eq.", "TOS\ncentral/east", "TOS\neast-west", "ZOS\neast-west tilt"]
    raw = np.asarray([[float(row[key]) for key in factor_keys] for row in sorted_rows], dtype=np.float64)
    scale = np.nanmax(np.abs(raw), axis=0)
    scale[scale == 0] = 1.0
    normalized = raw / scale[None, :]
    gains = np.asarray([float(row["gain_max_3m"]) for row in sorted_rows])
    cnop = np.asarray([float(row["cnop_max_3m"]) for row in sorted_rows])
    labels = [f"{row['source']} {row['year']}" for row in sorted_rows]

    fig = plt.figure(figsize=(12.8, 7.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.45, 0.38, 0.38))
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1], sharey=ax_heat)
    ax_cnop = fig.add_subplot(gs[0, 2], sharey=ax_heat)

    image = ax_heat.imshow(normalized, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xticks(np.arange(len(factor_labels)))
    ax_heat.set_xticklabels(factor_labels)
    ax_heat.set_title("A  Perturbation-factor matrix", loc="left", pad=8, fontweight="bold")
    ax_heat.tick_params(axis="x", length=0)
    ax_heat.tick_params(axis="y", length=0)
    for row_idx in range(raw.shape[0]):
        for col_idx in range(raw.shape[1]):
            color = "white" if abs(normalized[row_idx, col_idx]) > 0.62 else "#111827"
            ax_heat.text(col_idx, row_idx, f"{raw[row_idx, col_idx]:+.2f}", ha="center", va="center", fontsize=7.2, color=color)
    ax_heat.set_xticks(np.arange(-0.5, raw.shape[1], 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, raw.shape[0], 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.1)
    cbar = fig.colorbar(image, ax=ax_heat, shrink=0.82, pad=0.012)
    cbar.set_label("Column-normalized sign and strength")

    y = np.arange(len(labels))
    ax_gain.barh(y, gains, color="#0072B2", alpha=0.88)
    ax_gain.set_title("B  Gain", loc="left", pad=8, fontweight="bold")
    ax_gain.set_xlabel("Gain")
    ax_gain.grid(True, axis="x")
    ax_gain.tick_params(axis="y", labelleft=False, length=0)

    ax_cnop.barh(y, cnop, color="#D55E00", alpha=0.88)
    ax_cnop.axvline(0.5, color="#7F1D1D", linestyle="--", linewidth=1.0)
    ax_cnop.set_title("C  CNOP", loc="left", pad=8, fontweight="bold")
    ax_cnop.set_xlabel("Max 3m Niño3.4")
    ax_cnop.grid(True, axis="x")
    ax_cnop.tick_params(axis="y", labelleft=False, length=0)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cnop_factor_comparison.{suffix}", dpi=dpi)
    plt.close(fig)


def plot_best_case_topk(cases: list[CaseData], lat: np.ndarray, lon: np.ndarray, output_dir: Path, dpi: int) -> None:
    """如果结果包含 top-k CNOP，画出最强 case 的多个局部最优扰动。"""

    available = [
        case
        for case in cases
        if case.top_delta_phys is not None and case.top_delta_phys.size > 0 and case.top_gain_max_3m is not None
    ]
    if not available:
        return

    case = max(available, key=lambda item: item.gain_max_3m)
    top_delta = case.top_delta_phys
    assert top_delta is not None
    k = min(top_delta.shape[0], 5)
    fields = top_delta[:k]
    vmax_tos = max(float(np.nanpercentile(np.abs(fields[:, 0][..., region_mask(lat, lon, PERTURB_BOX)]), 98)), 1.0e-6)
    vmax_zos = max(float(np.nanpercentile(np.abs(fields[:, 1][..., region_mask(lat, lon, PERTURB_BOX)]), 98)), 1.0e-6)

    fig, axes = plt.subplots(2, k, figsize=(3.2 * k, 6.2), layout="constrained", sharex=True, sharey=True)
    if k == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col in range(k):
        gain = float(case.top_gain_max_3m[col]) if case.top_gain_max_3m is not None else float("nan")
        start = int(case.top_start_idx[col]) if case.top_start_idx is not None else col
        tos = mask_outside_view(fields[col, 0], lat, lon)
        zos = mask_outside_view(fields[col, 1], lat, lon)
        axes[0, col].pcolormesh(lon, lat, tos, shading="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax_tos, vmax_tos))
        axes[1, col].pcolormesh(lon, lat, zos, shading="auto", cmap="BrBG", norm=TwoSlopeNorm(0, -vmax_zos, vmax_zos))
        axes[0, col].set_title(f"rank {col + 1}  start {start}  gain {gain:.2f}", loc="left", fontsize=8.5, fontweight="bold")
        for row in range(2):
            setup_map_axis(axes[row, col], wide=True)
            if col > 0:
                axes[row, col].set_ylabel("")
        axes[0, col].set_xlabel("")
    axes[0, 0].text(0.0, 1.06, "TOS", transform=axes[0, 0].transAxes, fontsize=10, fontweight="bold")
    axes[1, 0].text(0.0, 1.06, "ZOS", transform=axes[1, 0].transAxes, fontsize=10, fontweight="bold")
    fig.suptitle(f"Top-{k} local CNOP candidates: {case.source} {case.year}", fontsize=13, fontweight="bold")

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"cnop_best_case_topk_candidates.{suffix}", dpi=dpi)
    plt.close(fig)


def write_precursor_tables(rows: list[dict[str, float | str | int]], output_dir: Path) -> None:
    csv_path = output_dir / "cnop_precursor_indices.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in fieldnames
        if key not in {"source", "year"}
    }
    md = [
        "# CNOP 前兆诊断摘要",
        "",
        "这些指标由 `case_*.npz` 中的物理扰动场计算，扰动只作用在输入窗口第 12 个月的 TOS/ZOS。",
        "",
        "## 合成信号",
        "",
        f"- 平均 Niño3.4 TOS 扰动：`{means['tos_Nino34']:+.4f}`。",
        f"- 平均西太平洋赤道 TOS 扰动：`{means['tos_WestEqPac']:+.4f}`。",
        f"- 平均中东太平洋赤道 TOS 扰动：`{means['tos_CentralEastEqPac']:+.4f}`。",
        f"- 平均 TOS 东西向对比：`{means['tos_eq_gradient']:+.4f}`。",
        f"- 平均 ZOS 东西向倾斜指标：`{means['zos_east_west_tilt']:+.4f}`。",
        "",
        "## 读图要点",
        "",
        "- `cnop_tos_case_atlas` 和 `cnop_zos_case_atlas` 把每个 case 的扰动单独画出，且使用统一色标，适合检查个例差异。",
        "- `cnop_factor_comparison` 把每个 case 的区域扰动因子放在同一张矩阵里，适合比较 TOS 与 ZOS 谁更稳定。",
        "- 如果重新运行 CNOP 时启用多初值，`cnop_best_case_topk_candidates` 会展示最强 case 的多个局部最优扰动。",
        "- 如果 Niño3.4 与中东太平洋 TOS 为正，说明最优扰动直接预热 ENSO 关键区。",
        "- 如果西太平洋与中东太平洋存在反号或明显梯度，说明扰动更像在调整东西向海温梯度。",
        "- ZOS 东西向倾斜可作为上层海洋状态/热跃层变化的替代线索，但它不是真实热含量。",
        "- `|mean| / spread` 越高，说明该位置不是单个 case 的偶然纹理，而是跨 neutral case 的稳定前兆。",
        "",
    ]
    (output_dir / "cnop_precursor_analysis.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_plot_style()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, lat, lon = load_cases(input_dir)
    plot_main_figure(cases, lat, lon, output_dir, args.dpi)
    rows = plot_precursor_figure(cases, lat, lon, output_dir, args.dpi)
    plot_case_atlas(cases, lat, lon, output_dir, args.dpi, variable_index=0, variable_name="TOS", cmap="RdBu_r")
    plot_case_atlas(cases, lat, lon, output_dir, args.dpi, variable_index=1, variable_name="ZOS", cmap="BrBG")
    plot_factor_comparison(rows, output_dir, args.dpi)
    plot_best_case_topk(cases, lat, lon, output_dir, args.dpi)
    write_precursor_tables(rows, output_dir)
    print(f"Wrote CNOP diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
