"""Stack the four global training-sample pseudo-3D surfaces vertically."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plotting.plot_training_sample_global_pseudo3d import (
    INPUT,
    OUT,
    VARIABLES,
    map_xy,
    plot_surface,
    smooth_interpolate,
)


def main() -> None:
    data = np.load(INPUT, allow_pickle=True)
    fields = np.asarray(data["fields"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    source = str(np.asarray(data["source"]).item())
    time_index = int(np.asarray(data["time_index"]).item())
    fields_hi = [smooth_interpolate(field, factor=4, sigma=1.4) for field in fields]
    lon_hi = np.linspace(lon[0], lon[-1], fields_hi[0].shape[1])
    lat_hi = np.linspace(lat[0], lat[-1], fields_hi[0].shape[0])
    cmap = mpl.colormaps["RdYlBu_r"]
    norms = []
    for field in fields_hi:
        finite = field[np.isfinite(field)]
        norms.append(Normalize(float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))))

    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": 9}):
        fig = plt.figure(figsize=(13.2, 10.0), facecolor="white")
        # Compact vertical stack: each surface gets a little more than 20% of
        # the canvas and neighboring layers are separated by a narrow gap.
        y_positions = [0.735, 0.525, 0.315, 0.105]
        axes = []
        meshes = []
        corners = ((0.08, 0.17), (0.94, 0.17), (0.19, 0.82), (0.86, 0.82))
        for y, name, field, norm in zip(y_positions, VARIABLES, fields_hi, norms):
            ax = fig.add_axes((0.045, y, 0.82, 0.20))
            mesh = plot_surface(ax, lon_hi, lat_hi, field, corners, cmap, norm, name, title_fontsize=17)
            axes.append(ax); meshes.append(mesh)
        fig.suptitle(f"WalkerNet training sample: global {source}, time index {time_index}", fontsize=17, fontweight="bold", y=0.975)
        for y, mesh, name in zip(y_positions, meshes, VARIABLES):
            cax = fig.add_axes((0.885, y + 0.025, 0.016, 0.15))
            cb = fig.colorbar(mesh, cax=cax)
            cb.set_label(name, fontsize=10, fontweight="bold", labelpad=7)
            cb.ax.tick_params(labelsize=7)
        png = out_dir / "walker_net_training_sample_global_pseudo3d_stack.png"
        pdf = out_dir / "walker_net_training_sample_global_pseudo3d_stack.pdf"
        fig.savefig(png, dpi=320, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    provenance = {
        "input": str(INPUT), "source": source, "time_index": time_index,
        "variables": VARIABLES, "layout": "four vertically stacked pseudo-3D surfaces with compact gaps",
        "transformations": ["rectangular lat-lon fields", "NaN-aware Gaussian smoothing sigma=1.4", "4x cubic interpolation", "pseudo-3D bilinear trapezoid mapping", "35 contour levels per variable"],
        "outputs": [png.name, pdf.name],
    }
    (out_dir / "walker_net_training_sample_global_pseudo3d_stack.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(png)


if __name__ == "__main__":
    main()
