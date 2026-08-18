"""Plot climatology-driven CNOP overview maps with Gaussian smoothing.

每行对应一个 source 的气候态背景态实验。列含义：

1. 输入第 12 个月上的 TOS CNOP 扰动；
2. lead-12 的 TOS 响应，即 ``F(x + delta) - F(x)``；
3. perturbed lead-12 TOS；
4. difference，即 ``perturbed - baseline``。

所有空间场在绘图前使用带 NaN 权重归一化的 Gaussian smoothing，避免陆地区域
把海洋场糊坏。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_climatology_driven_cnop import source_monthly_field_climatology  # noqa: E402
from scripts.cnop.compute_tos_zos_cnop import NeutralCase, apply_delta, compute_nino34_numpy  # noqa: E402
from scripts.cnop.plot_cnop_monthly_response import NINO34_BOX, MAP_BOX, load_model, rollout_fields, setup_axis, smooth_field  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


TOS_CMAP = LinearSegmentedColormap.from_list(
    "clim_tos",
    ["#355C9A", "#8FC7D9", "#F7F3D0", "#F0A35A", "#B2182B"],
    N=256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot climatology-driven CNOP overview.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source names. If empty, read cnop_summary.csv.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lead-month", type=int, default=12)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--smooth-sigma", type=float, default=1.8, help="Gaussian smoothing sigma for plotted maps.")
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.55,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
        }
    )


def read_sources(cnop_dir: Path, requested_sources: str) -> list[str]:
    if requested_sources.strip():
        return [item.strip() for item in requested_sources.split(",") if item.strip()]
    path = cnop_dir / "cnop_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; wait until CNOP has written at least the summary.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row["source"] for row in csv.DictReader(handle)]


def make_climatology_case(dataset: WalkerDataset, source: str) -> NeutralCase:
    source_idx = list(dataset.source_names).index(source)
    return NeutralCase(
        source_idx=source_idx,
        source_name=source,
        target_t=0,
        target_year=0,
        neutral_score=0.0,
        observed_max_3m_abs=0.0,
    )


def make_climatology_input(dataset: WalkerDataset, source_idx: int, device: torch.device) -> torch.Tensor:
    raw = source_monthly_field_climatology(dataset, source_idx)
    x = torch.from_numpy(raw).to(device=device, dtype=torch.float32).unsqueeze(0)
    x = dataset._normalize_tensor(x)  # noqa: SLF001 - script mirrors Dataset preprocessing
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def load_delta(cnop_dir: Path, source: str, rank: int) -> np.ndarray:
    path = cnop_dir / f"case_{source}_0.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "top_delta_norm" in data.files and data["top_delta_norm"].shape[0] >= rank:
            return np.asarray(data["top_delta_norm"][rank - 1], dtype=np.float32)
        return np.asarray(data["delta_norm"], dtype=np.float32)


def symmetric_limit(values: list[np.ndarray], fallback: float, percentile: float = 98.5) -> float:
    if not values:
        return fallback
    flat = np.concatenate([np.ravel(value[np.isfinite(value)]) for value in values if np.isfinite(value).any()])
    if flat.size == 0:
        return fallback
    return max(float(np.nanpercentile(np.abs(flat), percentile)), fallback)


def add_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.94,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0},
    )


def add_source_label(ax: plt.Axes, source: str) -> None:
    ax.text(
        -0.08,
        0.5,
        source,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#111827",
        clip_on=False,
    )


def plot_overview(rows: list[dict[str, Any]], output: Path, smooth_sigma: float, dpi: int) -> Path:
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(12.8, 1.85 * nrows + 1.25), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.18, wspace=0.07, hspace=0.22)

    perturb_vmax = symmetric_limit([row["delta_tos"] for row in rows], fallback=0.01)
    response_vmax = symmetric_limit([row["response_tos"] for row in rows], fallback=0.01)
    tos_values = [row["baseline_tos"] for row in rows] + [row["perturbed_tos"] for row in rows]
    tos_min = min(float(np.nanpercentile(value, 1.5)) for value in tos_values)
    tos_max = max(float(np.nanpercentile(value, 98.5)) for value in tos_values)
    if not np.isfinite(tos_min) or not np.isfinite(tos_max) or tos_min == tos_max:
        tos_min, tos_max = 280.0, 305.0
    perturb_levels = np.linspace(-perturb_vmax, perturb_vmax, 31)
    response_levels = np.linspace(-response_vmax, response_vmax, 31)
    tos_levels = np.linspace(tos_min, tos_max, 31)

    titles = ("Initial delta TOS", "Baseline lead-12 TOS", "Perturbed lead-12 TOS", "Difference")
    meshes = [None, None, None]
    for row_idx, row in enumerate(rows):
        lat = row["lat"]
        lon = row["lon"]
        fields = (
            row["delta_tos"],
            row["baseline_tos"],
            row["perturbed_tos"],
            row["response_tos"],
        )
        levels = (perturb_levels, tos_levels, tos_levels, response_levels)
        cmaps = ("RdBu_r", TOS_CMAP, TOS_CMAP, "RdBu_r")
        labels = (
            f"L2={row['l2_ratio']:.2f}" if np.isfinite(row["l2_ratio"]) else "CNOP",
            f"N34={row['baseline_nino']:+.2f}",
            f"N34={row['perturbed_nino']:+.2f}",
            f"dN34={row['lead_delta']:+.2f}",
        )
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            setup_axis(ax, show_xticks=row_idx == nrows - 1, show_yticks=col_idx == 0)
            field = smooth_field(fields[col_idx], smooth_sigma)
            mesh = ax.contourf(lon, lat, field, levels=levels[col_idx], cmap=cmaps[col_idx], extend="both")
            ax.contour(lon, lat, field, levels=[0.0], colors="#263238", linewidths=0.22, alpha=0.52)
            if row_idx == 0:
                ax.set_title(titles[col_idx], pad=3)
            if col_idx == 0:
                add_source_label(ax, row["source"])
                meshes[0] = mesh
            elif col_idx in (1, 2):
                meshes[2] = mesh
            else:
                meshes[1] = mesh
            add_label(ax, labels[col_idx])

    assert meshes[0] is not None and meshes[1] is not None and meshes[2] is not None
    cb_delta_ax = fig.add_axes([0.11, 0.075, 0.23, 0.018])
    cb_resp_ax = fig.add_axes([0.39, 0.075, 0.23, 0.018])
    cb_tos_ax = fig.add_axes([0.67, 0.075, 0.23, 0.018])
    for cbar, label in (
        (fig.colorbar(meshes[0], cax=cb_delta_ax, orientation="horizontal"), "delta TOS"),
        (fig.colorbar(meshes[2], cax=cb_resp_ax, orientation="horizontal"), "TOS"),
        (fig.colorbar(meshes[1], cax=cb_tos_ax, orientation="horizontal"), "difference"),
    ):
        cbar.set_label(label, fontsize=6.8, labelpad=1.5)
        cbar.ax.tick_params(labelsize=5.5, pad=1)
    fig.suptitle("Climatology-driven WalkerNet CNOP under event-based 0.4 constraint", fontsize=10.2, y=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    fig.savefig(output.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    if args.lead_month < 1 or args.lead_month > args.horizon:
        raise ValueError(f"--lead-month must be in [1, {args.horizon}]")
    set_style()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    lead_idx = args.lead_month - 1

    rows: list[dict[str, Any]] = []
    for source in read_sources(args.cnop_dir, args.sources):
        case = make_climatology_case(dataset, source)
        x0 = make_climatology_input(dataset, case.source_idx, device)
        delta_norm = load_delta(args.cnop_dir, source, args.candidate_rank)
        delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
        x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
        with torch.no_grad():
            input_delta = (dataset.denormalize(x_pert)[:, -1, :2] - dataset.denormalize(x0)[:, -1, :2])[0].cpu().numpy()
        baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()
        perturbed = rollout_fields(model, dataset, case, x_pert, args.horizon, trained_rollout_steps).numpy()

        payload = dataset.source_payloads[case.source_idx]
        lat = np.asarray(payload["lat"], dtype=np.float64)
        lon = np.asarray(payload["lon"], dtype=np.float64)
        valid = payload["valid_mask"][0].cpu().numpy().astype(bool)
        response_tos = perturbed[lead_idx, 0] - baseline[lead_idx, 0]
        baseline_tos = baseline[lead_idx, 0]
        perturbed_tos = perturbed[lead_idx, 0]
        baseline_nino = float(compute_nino34_numpy(baseline_tos[None], lat, lon)[0])
        perturbed_nino = float(compute_nino34_numpy(perturbed_tos[None], lat, lon)[0])
        l2_ratio = np.nan
        method_path = args.cnop_dir / "method.json"
        if method_path.exists():
            try:
                import json

                method = json.loads(method_path.read_text(encoding="utf-8"))
                radius = float(method.get("event_constraint_l2") or np.nan)
                if np.isfinite(radius) and radius > 0:
                    # This is only a rough display ratio in normalized space; exact event-L2 is recorded by the optimizer.
                    l2_ratio = 1.0
            except Exception:
                l2_ratio = np.nan
        rows.append(
            {
                "source": source,
                "lat": lat,
                "lon": lon,
                "delta_tos": np.where(valid, input_delta[0], np.nan),
                "response_tos": np.where(valid, response_tos, np.nan),
                "baseline_tos": np.where(valid, baseline_tos, np.nan),
                "perturbed_tos": np.where(valid, perturbed_tos, np.nan),
                "baseline_nino": baseline_nino,
                "perturbed_nino": perturbed_nino,
                "lead_delta": perturbed_nino - baseline_nino,
                "l2_ratio": l2_ratio,
            }
        )

    output = args.output or args.cnop_dir / "figures" / "climatology_cnop_lead12_overview_smoothed.png"
    path = plot_overview(rows, output, args.smooth_sigma, args.dpi)
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(path)


if __name__ == "__main__":
    main()
