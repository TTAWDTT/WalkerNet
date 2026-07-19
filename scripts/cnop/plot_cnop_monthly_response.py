"""Plot monthly CNOP-induced rollout response fields.

The figure shows ``F(x + delta) - F(x)`` month by month:

- upper panel of each month: TOS response with TAUU/TAUV response vectors;
- lower panel of each month: ZOS response.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import (  # noqa: E402
    NeutralCase,
    apply_delta,
    make_case_input,
    target_month_tensor,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.model import WalkerNet  # noqa: E402
from src.utils import load_config  # noqa: E402


VARIABLES = ("tos", "zos", "tauu", "tauv")
NINO34_BOX = (190.0, 240.0, -5.0, 5.0)
MAP_BOX = (100.0, 300.0, -35.0, 35.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot monthly CNOP response maps.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--case-source", type=str, default="")
    parser.add_argument("--case-year", type=int, default=0)
    parser.add_argument("--candidate-rank", type=int, default=1, help="Use top-k candidate rank from case npz, 1-based.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--arrow-stride", type=int, default=6)
    parser.add_argument("--arrow-scale", type=float, default=4.0, help="Approximate degree length of the reference vector.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma for plotted response fields.")
    parser.add_argument("--contour-levels", type=int, default=33)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    return parser.parse_args()


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
        }
    )


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[WalkerNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WalkerNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def read_case_from_summary(cnop_dir: Path, case_source: str, case_year: int) -> tuple[str, int, int, float]:
    summary_path = cnop_dir / "cnop_summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {summary_path}")
    row = None
    if case_source and case_year:
        for item in rows:
            if item["source"] == case_source and int(item["target_year"]) == case_year:
                row = item
                break
    else:
        row = rows[0]
    if row is None:
        raise ValueError(f"Case {case_source} {case_year} not found in {summary_path}")
    return row["source"], int(row["target_year"]), int(row["target_t"]), float(row["observed_max_3m_abs"])


def load_case_npz(cnop_dir: Path, source: str, year: int, rank: int) -> tuple[np.ndarray, Path]:
    path = cnop_dir / f"case_{source}_{year}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "top_delta_norm" in data.files and data["top_delta_norm"].shape[0] >= rank:
            delta = np.asarray(data["top_delta_norm"][rank - 1], dtype=np.float32)
        else:
            delta = np.asarray(data["delta_norm"], dtype=np.float32)
    return delta, path


def make_case(dataset: WalkerDataset, source: str, year: int, target_t: int, observed: float) -> NeutralCase:
    source_idx = dataset.source_names.index(source)
    return NeutralCase(
        source_idx=source_idx,
        source_name=source,
        target_t=target_t,
        target_year=year,
        neutral_score=observed,
        observed_max_3m_abs=observed,
    )


def rollout_fields(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x_norm: torch.Tensor,
    horizon: int,
    trained_rollout_steps: int,
) -> torch.Tensor:
    """Return denormalized predictions with shape ``(T, 4, H, W)``."""

    preds: list[torch.Tensor] = []
    window = x_norm
    for step in range(horizon):
        target_month = target_month_tensor(dataset, case, step, x_norm.device)
        rollout_step = torch.tensor([min(step, trained_rollout_steps - 1)], dtype=torch.long, device=x_norm.device)
        with torch.no_grad():
            pred_norm = model(window, target_month, rollout_step=rollout_step)
            pred_phys = dataset.denormalize(pred_norm)[:, 0]
        preds.append(pred_phys[0].detach().cpu())
        window = torch.cat([window[:, 1:], pred_norm], dim=1)
    return torch.stack(preds, dim=0)


def add_box(ax: plt.Axes, bounds: tuple[float, float, float, float], color: str) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], color=color, lw=0.8)


def setup_axis(ax: plt.Axes, show_xticks: bool, show_yticks: bool) -> None:
    ax.set_xlim(MAP_BOX[0], MAP_BOX[1])
    ax.set_ylim(MAP_BOX[2], MAP_BOX[3])
    ax.set_xticks([120, 150, 180, 210, 240, 270, 300])
    ax.set_yticks([-30, -10, 10, 30])
    if not show_xticks:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(["120E", "150E", "180", "150W", "120W", "90W", "60W"])
    if not show_yticks:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(["30S", "10S", "10N", "30N"])
    ax.grid(color="#9AA3AF", alpha=0.32, linewidth=0.45)
    add_box(ax, NINO34_BOX, "#111827")


def month_labels(case: NeutralCase, dataset: WalkerDataset, horizon: int) -> list[str]:
    payload = dataset.source_payloads[case.source_idx]
    months = payload["months"][case.target_t : case.target_t + horizon]
    names = ("Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.")
    return [names[int(month) - 1] for month in months]


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Lightly smooth a plotted field while preserving NaN masks."""

    if sigma <= 0:
        return field
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return field

    finite = np.isfinite(field)
    filled = np.where(finite, field, 0.0)
    weights = gaussian_filter(finite.astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = gaussian_filter(filled, sigma=sigma, mode="nearest") / np.maximum(weights, 1.0e-6)
    smoothed[~finite & (weights < 0.25)] = np.nan
    return smoothed


def plot_monthly_response(
    response: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    labels: list[str],
    source: str,
    year: int,
    rank: int,
    output_dir: Path,
    dpi: int,
    arrow_stride: int,
    arrow_scale: float,
    smooth_sigma: float,
    contour_levels: int,
) -> Path:
    view_mask = (lat[:, None] >= MAP_BOX[2]) & (lat[:, None] <= MAP_BOX[3]) & (lon[None, :] >= MAP_BOX[0]) & (lon[None, :] <= MAP_BOX[1])
    tos_vmax = max(float(np.nanpercentile(np.abs(response[:, 0][..., view_mask]), 98)), 1.0e-6)
    zos_vmax = max(float(np.nanpercentile(np.abs(response[:, 1][..., view_mask]), 98)), 1.0e-6)
    tau = np.sqrt(response[:, 2] ** 2 + response[:, 3] ** 2)
    tau_ref = max(float(np.nanpercentile(tau[..., view_mask], 95)), 1.0e-6)
    plot_response = np.empty_like(response)
    for month_idx in range(response.shape[0]):
        for var_idx in range(response.shape[1]):
            plot_response[month_idx, var_idx] = smooth_field(response[month_idx, var_idx], smooth_sigma)
    tos_levels = np.linspace(-tos_vmax, tos_vmax, max(9, int(contour_levels)))
    zos_levels = np.linspace(-zos_vmax, zos_vmax, max(9, int(contour_levels)))

    fig, axes = plt.subplots(4, 6, figsize=(18.0, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.15, wspace=0.055, hspace=0.12)
    lon2, lat2 = np.meshgrid(lon, lat)
    tos_mesh = None
    zos_mesh = None

    for idx in range(response.shape[0]):
        block, col = divmod(idx, 6)
        tos_row = block * 2
        zos_row = tos_row + 1
        ax_tos = axes[tos_row, col]
        ax_zos = axes[zos_row, col]
        show_y = col == 0
        show_x = zos_row == 3

        tos_mesh = ax_tos.contourf(lon, lat, plot_response[idx, 0], levels=tos_levels, cmap="RdYlBu_r", extend="both")
        setup_axis(ax_tos, show_xticks=False, show_yticks=show_y)
        ax_tos.set_title(f"({chr(97 + idx)}) {labels[idx]}", fontweight="bold", pad=2)
        if show_y:
            ax_tos.set_ylabel("TOS + tau", fontsize=7)

        step = max(1, int(arrow_stride))
        sl_lat = (lat >= MAP_BOX[2]) & (lat <= MAP_BOX[3])
        sl_lon = (lon >= MAP_BOX[0]) & (lon <= MAP_BOX[1])
        lat_idx = np.where(sl_lat)[0][::step]
        lon_idx = np.where(sl_lon)[0][::step]
        ax_tos.quiver(
            lon2[np.ix_(lat_idx, lon_idx)],
            lat2[np.ix_(lat_idx, lon_idx)],
            plot_response[idx, 2][np.ix_(lat_idx, lon_idx)],
            plot_response[idx, 3][np.ix_(lat_idx, lon_idx)],
            color="#1F2937",
            width=0.0018,
            headwidth=3.0,
            headlength=3.5,
            headaxislength=3.2,
            scale=max(tau_ref / max(float(arrow_scale), 1.0e-6), 1.0e-6),
            scale_units="xy",
            angles="xy",
            alpha=0.82,
        )

        zos_mesh = ax_zos.contourf(lon, lat, plot_response[idx, 1], levels=zos_levels, cmap="BrBG", extend="both")
        setup_axis(ax_zos, show_xticks=show_x, show_yticks=show_y)
        if show_y:
            ax_zos.set_ylabel("ZOS", fontsize=7)
        if not show_x:
            ax_zos.set_xticklabels([])

    assert tos_mesh is not None and zos_mesh is not None
    cbar_tos_ax = fig.add_axes([0.18, 0.075, 0.30, 0.018])
    cbar_zos_ax = fig.add_axes([0.55, 0.075, 0.30, 0.018])
    cbar_tos = fig.colorbar(tos_mesh, cax=cbar_tos_ax, orientation="horizontal")
    cbar_tos.set_label("TOS response: perturbed rollout - baseline rollout")
    cbar_zos = fig.colorbar(zos_mesh, cax=cbar_zos_ax, orientation="horizontal")
    cbar_zos.set_label("ZOS response: perturbed rollout - baseline rollout")
    fig.suptitle(
        f"CNOP monthly response fields: {source} {year}, candidate rank {rank}",
        fontsize=14,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnop_monthly_response_{source}_{year}_rank{rank}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    set_plot_style()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))

    source, year, target_t, observed = read_case_from_summary(args.cnop_dir, args.case_source, args.case_year)
    case = make_case(dataset, source, year, target_t, observed)
    delta_norm, npz_path = load_case_npz(args.cnop_dir, source, year, args.candidate_rank)

    x0 = make_case_input(dataset, case, device)
    delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
    x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))

    baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps)
    perturbed = rollout_fields(model, dataset, case, x_pert, args.horizon, trained_rollout_steps)
    response = (perturbed - baseline).numpy()

    payload = dataset.source_payloads[case.source_idx]
    lat = np.asarray(payload["lat"], dtype=np.float64)
    lon = np.asarray(payload["lon"], dtype=np.float64)
    labels = month_labels(case, dataset, args.horizon)
    output_dir = args.output_dir or args.cnop_dir / "figures"
    path = plot_monthly_response(
        response,
        lat,
        lon,
        labels,
        source,
        year,
        args.candidate_rank,
        output_dir,
        args.dpi,
        args.arrow_stride,
        args.arrow_scale,
        args.smooth_sigma,
        args.contour_levels,
    )
    print(f"checkpoint_epoch={checkpoint.get('epoch')} case_npz={npz_path}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
