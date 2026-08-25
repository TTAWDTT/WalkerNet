"""Plot one WalkerNet training sample as four global pseudo-3D maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tmp" / "walkernet_training_sample_cesm2_t0.npz"
OUT = ROOT / "docs" / "assets" / "walkernet_training_sample"
VARIABLES = ["TOS", "ZOS", "TAUX", "TAUY"]


def smooth_interpolate(field: np.ndarray, factor: int = 4, sigma: float = 1.4) -> np.ndarray:
    valid = np.isfinite(field)
    numerator = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.full_like(field, np.nan, dtype=float), where=denominator > 1e-8)
    values = zoom(np.where(np.isfinite(smooth), smooth, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    weights = zoom(np.isfinite(smooth).astype(float), (factor, factor), order=1, mode="nearest")
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0.35)


def map_xy(lon: np.ndarray, lat: np.ndarray, corners: tuple[tuple[float, float], ...]):
    u = (lon - 0.0) / 360.0
    v = (lat + 90.0) / 180.0
    sw, se, nw, ne = corners
    x = (1-u)*(1-v)*sw[0] + u*(1-v)*se[0] + (1-u)*v*nw[0] + u*v*ne[0]
    y = (1-u)*(1-v)*sw[1] + u*(1-v)*se[1] + (1-u)*v*nw[1] + u*v*ne[1]
    return x, y


def plot_surface(
    ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, corners, cmap, norm,
    title: str, title_fontsize: float = 12,
):
    ax.add_patch(Polygon([corners[2], corners[0], corners[1], corners[3]], closed=True,
                         facecolor="#A7A7A7", edgecolor="#656565", lw=0.45, alpha=0.16, zorder=0))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    xg, yg = map_xy(lon_grid, lat_grid, corners)
    levels = np.linspace(norm.vmin, norm.vmax, 35)
    mesh = ax.contourf(xg, yg, field, levels=levels, cmap=cmap, norm=norm, extend="both", antialiased=True, zorder=1)
    ax.contour(xg, yg, field, levels=levels[::2], colors="#4F5962", linewidths=0.22, alpha=0.38, zorder=3)
    outline = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
    ax.plot(outline[:, 0], outline[:, 1], color="#30343A", lw=0.85, zorder=4)
    for lo in (0, 90, 180, 270, 360):
        x, y = map_xy(np.array([lo, lo]), np.array([-90, 90]), corners)
        ax.plot(x, y, color="#4F5962", lw=0.28, alpha=0.25, ls=":", zorder=3)
    for la in (-60, -30, 0, 30, 60):
        x, y = map_xy(np.array([0, 360]), np.array([la, la]), corners)
        ax.plot(x, y, color="#4F5962", lw=0.32 if la == 0 else 0.22, alpha=0.35, zorder=3)
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=title_fontsize, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return mesh


def main() -> None:
    data = np.load(INPUT, allow_pickle=True)
    fields = np.asarray(data["fields"], dtype=float)
    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    lon_hi = np.linspace(lon[0], lon[-1], fields.shape[-1] * 4)
    lat_hi = np.linspace(lat[0], lat[-1], fields.shape[-2] * 4)
    fields_hi = [smooth_interpolate(field, factor=4, sigma=1.4) for field in fields]
    cmap = mpl.colormaps["RdYlBu_r"]

    OUT.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": 9}):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), subplot_kw={"aspect": "auto"})
        norms = []
        for field in fields_hi:
            finite = field[np.isfinite(field)]
            vmin, vmax = np.nanpercentile(finite, [2, 98])
            norms.append(Normalize(float(vmin), float(vmax)))
        corners = ((0.08, 0.18), (0.94, 0.18), (0.19, 0.82), (0.86, 0.82))
        meshes = []
        for ax, name, field, norm in zip(axes.flat, VARIABLES, fields_hi, norms):
            meshes.append(plot_surface(ax, lon_hi, lat_hi, field, corners, cmap, norm, name))
        source = str(np.asarray(data["source"]).item())
        time_index = int(np.asarray(data["time_index"]).item())
        fig.suptitle(f"WalkerNet training sample: global {source}, time index {time_index}", fontsize=17, fontweight="bold", y=0.98)
        for ax, mesh, name in zip(axes.flat, meshes, VARIABLES):
            cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(name)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.91, wspace=0.04, hspace=0.04)
        png = OUT / "walker_net_training_sample_global_pseudo3d.png"
        pdf = OUT / "walker_net_training_sample_global_pseudo3d.pdf"
        fig.savefig(png, dpi=320, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    provenance = {
        "input": str(INPUT),
        "source": source,
        "time_index": time_index,
        "variables": VARIABLES,
        "native_grid": [int(fields.shape[-2]), int(fields.shape[-1])],
        "display_grid": [int(fields_hi[0].shape[-2]), int(fields_hi[0].shape[-1])],
        "transformations": ["rectangular lat-lon fields", "NaN-aware Gaussian smoothing sigma=1.4", "4x cubic interpolation", "pseudo-3D bilinear trapezoid mapping", "35 contour levels per variable"],
        "outputs": [png.name, pdf.name],
    }
    (OUT / "walker_net_training_sample_global_pseudo3d.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (OUT / "walker_net_training_sample_global_pseudo3d.alt.txt").write_text(
        "Four global pseudo-3D surfaces from one CESM2 WalkerNet training sample: TOS, ZOS, TAUX, and TAUY. "
        "Each field was first represented on a rectangular latitude-longitude grid, then smoothed, densely interpolated, and mapped to a skewed pseudo-3D surface.", encoding="utf-8"
    )
    print(png)


if __name__ == "__main__":
    main()
