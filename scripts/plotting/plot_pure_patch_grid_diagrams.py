"""Draw pure 2:1 rectangular WalkerNet patch-grid diagrams, no data field."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "walkernet_patch_grids"


def draw_grid(rows: int, cols: int, name: str, label: str, line_width: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal")
    ax.set_facecolor("#FFF8D8")
    # Subtle alternating yellow cells keep the pure diagram legible without
    # introducing any scientific data or spatial field.
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#FFF0A8", edgecolor="none", alpha=0.48, zorder=0))
    for x in range(cols + 1):
        ax.axvline(x, color="#225EA8", linewidth=line_width, alpha=0.72, zorder=2)
    for y in range(rows + 1):
        ax.axhline(y, color="#225EA8", linewidth=line_width, alpha=0.72, zorder=2)
    ax.add_patch(Rectangle((0, 0), cols, rows, fill=False, edgecolor="#123B6D", linewidth=1.6, zorder=3))
    ax.axis("off")
    ax.set_title(f"WalkerNet patch grid: {label}", fontsize=17, pad=12)
    ax.text(0.012, 0.965, f"{rows}×{cols} = {rows * cols:,} rectangular cells",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#225EA8", "alpha": 0.9, "pad": 5})
    output = OUT / f"{name}.png"
    fig.savefig(output, dpi=600, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_grid(45, 90, "walker_net_pure_grid_45x90", "dense 4°×4° layout", 0.16)
    draw_grid(23, 44, "walker_net_pure_grid_23x44", "coarse approximately quarter-count layout", 0.65)
    provenance = {
        "type": "pure geometric grid diagram",
        "data_field": "none",
        "aspect_ratio": "2:1 rectangle",
        "dense": "45x90 = 4050 cells",
        "coarse": "23x44 = 1012 cells, approximately one quarter of 4050",
        "palette": {"grid": "#225EA8", "alternate_cell": "#FFF0A8", "background": "#FFF8D8"},
        "outputs": ["walker_net_pure_grid_45x90.png", "walker_net_pure_grid_23x44.png"],
    }
    (OUT / "walker_net_pure_patch_grids.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
