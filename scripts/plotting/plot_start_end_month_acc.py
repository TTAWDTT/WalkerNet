"""Plot the saved 12x12 start-month/end-month ACC matrices.

The matrices are the first 12 forecast leads remapped as
end_month = start_month + lead - 1 (mod 12). Cubic interpolation is used only
for the contour display; the saved matrix values are not altered.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import RectBivariateSpline


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "walkernet_rollout_skill"
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

REMOTE_OUTPUT = "/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/"
MODEL_CSV = "eval_rollout_best_skill_start_end_month_acc_12x12_model.csv"
PERSISTENCE_CSV = "eval_rollout_best_skill_start_end_month_acc_12x12_persistence.csv"


def _load_matrix(path: Path) -> np.ndarray:
    rows = list(csv.reader(path.open(encoding="utf-8")))
    values = np.asarray([[float(x) for x in row[1:13]] for row in rows[1:13]], dtype=float)
    if values.shape != (12, 12):
        raise ValueError(f"Expected 12x12 matrix in {path}, got {values.shape}")
    return values


def _smooth_contour(ax, values: np.ndarray, norm: Normalize):
    raw = np.arange(1, 13, dtype=float)
    dense = np.linspace(1, 12, 240)
    spline = RectBivariateSpline(raw, raw, values, kx=3, ky=3, s=0)
    smooth = np.clip(spline(dense, dense), norm.vmin, norm.vmax)
    X, Y = np.meshgrid(dense, dense)
    levels = np.linspace(norm.vmin, norm.vmax, 18)
    return ax.contourf(X, Y, smooth, levels=levels, cmap="YlOrBr", norm=norm, extend="neither")


def _direct_grid(ax, values: np.ndarray, norm: Normalize):
    """Render the saved 12x12 cells directly, without smoothing/interpolation."""
    edges = np.arange(0.5, 13.5, 1.0)
    return ax.pcolormesh(edges, edges, values, cmap="YlOrBr", norm=norm, shading="flat")


def _format_axis(ax) -> None:
    ax.set_xlabel("End month")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(MONTH_NAMES)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.5, 12.5)
    ax.set_ylabel("Start month")
    ax.grid(False)


def main() -> None:
    data_dir = OUT / "matrix_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # The formal matrices have been staged in the repository's result bundle.
    model_path = data_dir / MODEL_CSV
    persistence_path = data_dir / PERSISTENCE_CSV
    model = _load_matrix(model_path)
    persistence = _load_matrix(persistence_path)

    norm = Normalize(vmin=-0.5, vmax=1.0)
    with mpl.rc_context({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 8.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharex=True, sharey=True)
        for ax, values, title in zip(axes, (model, persistence), ("WalkerNet", "Persistence")):
            contour = _smooth_contour(ax, values, norm)
            ax.set_title(title, pad=8)
            _format_axis(ax)
        fig.suptitle("Nino3.4 ACC by forecast start and end month", fontsize=15, y=0.98)
        cbar = fig.colorbar(contour, ax=axes, pad=0.025, fraction=0.035)
        cbar.set_label("ACC")
        cbar.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        fig.subplots_adjust(left=0.075, right=0.84, bottom=0.12, top=0.87, wspace=0.08)
        for fmt in ("png", "pdf"):
            fig.savefig(OUT / f"walkernet_start_end_month_acc_model_persistence.{fmt}", dpi=600 if fmt == "png" else None)
        plt.close(fig)

        # A second, unsmoothed direct-grid rendering for visual comparison.
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharex=True, sharey=True)
        for ax, values, title in zip(axes, (model, persistence), ("WalkerNet", "Persistence")):
            grid = _direct_grid(ax, values, norm)
            ax.set_title(title, pad=8)
            _format_axis(ax)
        fig.suptitle("Nino3.4 ACC by forecast start and end month (direct grid)", fontsize=15, y=0.98)
        cbar = fig.colorbar(grid, ax=axes, pad=0.025, fraction=0.035)
        cbar.set_label("ACC")
        cbar.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        fig.subplots_adjust(left=0.075, right=0.84, bottom=0.12, top=0.87, wspace=0.08)
        for fmt in ("png", "pdf"):
            fig.savefig(OUT / f"walkernet_start_end_month_acc_model_persistence_grid.{fmt}", dpi=600 if fmt == "png" else None)
        plt.close(fig)

    provenance = {
        "figure": "walkernet_start_end_month_acc_model_persistence",
        "source_remote_dir": REMOTE_OUTPUT,
        "source_files": [MODEL_CSV, PERSISTENCE_CSV],
        "checkpoint": "historical_mixed5_best_skill.pt",
        "split": "test",
        "definition": "first 12 forecast leads; end_month = start_month + lead - 1 modulo 12",
        "transformations": ["direct saved 12x12 matrices", "cubic RectBivariateSpline interpolation for contour display only", "shared color normalization vmin=-0.5, vmax=1.0"],
        "outputs": ["walkernet_start_end_month_acc_model_persistence.png", "walkernet_start_end_month_acc_model_persistence.pdf"],
        "direct_grid_outputs": ["walkernet_start_end_month_acc_model_persistence_grid.png", "walkernet_start_end_month_acc_model_persistence_grid.pdf"],
        "direct_grid": "saved 12x12 cells rendered with pcolormesh; no smoothing or interpolation",
    }
    (OUT / "walkernet_start_end_month_acc_model_persistence.provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "walkernet_start_end_month_acc_model_persistence.alt.txt").write_text(
        "Two-panel contour plot of Nino3.4 ACC indexed by forecast start month and end month. "
        "The left panel is WalkerNet and the right panel is persistence; both use the same ACC color scale "
        "from -0.5 to 1.0. The displayed contours are interpolated from saved 12x12 matrices.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
