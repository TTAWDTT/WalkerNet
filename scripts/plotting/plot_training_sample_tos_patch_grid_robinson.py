"""Robinson-style global TOS map with exact 4x4-degree WalkerNet patches."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tmp" / "walkernet_training_sample_cesm2_t0.npz"
OUT = ROOT / "docs" / "assets" / "walkernet_training_sample"

# Robinson projection tabulation (Snyder/USGS standard, 5-degree nodes).
ROBIN_LAT = np.arange(0, 91, 5.0)
ROBIN_X = np.array([1.0000, .9986, .9954, .9900, .9822, .9730, .9600, .9427, .9216, .8962, .8679, .8350, .7986, .7597, .7186, .6732, .6213, .5722, .5322])
ROBIN_Y = np.array([0.0000, .0620, .1240, .1860, .2480, .3100, .3720, .4340, .4958, .5571, .6176, .6769, .7346, .7903, .8435, .8936, .9394, .9761, 1.0000])


def robinson(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.deg2rad(lon_deg)
    sign = np.sign(lat_deg)
    a = np.abs(lat_deg)
    x_factor = np.interp(a, ROBIN_LAT, ROBIN_X)
    y_factor = np.interp(a, ROBIN_LAT, ROBIN_Y)
    x = 0.8487 * x_factor * lon
    y = 1.3523 * y_factor * sign
    return x, y


def main() -> None:
    data = np.load(INPUT, allow_pickle=True)
    tos = np.asarray(data["fields"][0], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    source = str(np.asarray(data["source"]).item())
    time_index = int(np.asarray(data["time_index"]).item())

    # Reorder 0..360 data into -180..180 for a continuous Robinson map.
    order = np.r_[180:360, 0:180]
    lon_centers = np.r_[lon[180:] - 360.0, lon[:180]]
    field = np.ma.masked_invalid(tos[:, order])
    lon_edges = np.linspace(-180.0, 180.0, 361)
    lat_edges = np.linspace(-90.0, 90.0, 181)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)
    X, Y = robinson(lon_grid, lat_grid)
    finite = field.compressed()
    norm = Normalize(float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98)))
    cmap = mpl.colormaps["RdYlBu_r"].with_extremes(bad="#F4F1E8")
    OUT.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context({"font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": 9}):
        fig, ax = plt.subplots(figsize=(12.8, 6.6), layout="constrained")
        mesh = ax.pcolormesh(X, Y, field, cmap=cmap, norm=norm, shading="auto", rasterized=True)

        # Exact 4-degree patch boundaries. Longitude/latitude curves are
        # projected, so cells remain geographic 4x4 patches rather than a
        # decorative Cartesian grid.
        grid_color = "#263238"
        for lat_line in np.arange(-88.0, 89.0, 4.0):
            lo = np.linspace(-180, 180, 721)
            xx, yy = robinson(lo, np.full_like(lo, lat_line))
            ax.plot(xx, yy, color=grid_color, linewidth=0.22, alpha=0.34, zorder=3)
        for lon_line in np.arange(-180.0, 181.0, 4.0):
            la = np.linspace(-90, 90, 361)
            xx, yy = robinson(np.full_like(la, lon_line), la)
            ax.plot(xx, yy, color=grid_color, linewidth=0.22, alpha=0.34, zorder=3)

        # Robinson outline.
        lo = np.linspace(-180, 180, 721)
        for la in (-90, 90):
            xx, yy = robinson(lo, np.full_like(lo, la)); ax.plot(xx, yy, color="#20252A", lw=0.9, zorder=4)
        la = np.linspace(-90, 90, 361)
        for lo0 in (-180, 180):
            xx, yy = robinson(np.full_like(la, lo0), la); ax.plot(xx, yy, color="#20252A", lw=0.9, zorder=4)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"WalkerNet TOS training sample with 4°×4° Robinson patch grid ({source}, time index {time_index})", fontsize=15, pad=10)
        ax.text(0.015, 0.965, "45×90 = 4050 patches", transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#263238", "alpha": 0.88, "pad": 4})
        cb = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.035); cb.set_label("TOS")
        for fmt in ("png", "pdf"):
            fig.savefig(OUT / f"walker_net_training_sample_tos_patch_grid_robinson.{fmt}", dpi=600 if fmt == "png" else None)
        plt.close(fig)

    provenance = {
        "input": str(INPUT), "source": source, "time_index": time_index, "variable": "TOS",
        "projection": "Robinson-style tabulated projection with curved side boundaries",
        "patch_grid": "45x90 geographic cells, each 4x4 degrees",
        "transformations": ["reorder longitude to -180..180", "Robinson projection", "no field interpolation", "exact projected 4-degree grid overlay"],
        "outputs": ["walker_net_training_sample_tos_patch_grid_robinson.png", "walker_net_training_sample_tos_patch_grid_robinson.pdf"],
    }
    (OUT / "walker_net_training_sample_tos_patch_grid_robinson.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT / "walker_net_training_sample_tos_patch_grid_robinson.alt.txt").write_text(
        "Global Robinson-style TOS map with curved side boundaries and a projected geographic grid of 45 by 90 cells, each four degrees by four degrees.", encoding="utf-8"
    )
    print(OUT / "walker_net_training_sample_tos_patch_grid_robinson.png")


if __name__ == "__main__":
    main()
