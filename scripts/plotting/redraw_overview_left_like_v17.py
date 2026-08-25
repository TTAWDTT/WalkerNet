"""Replace the left overview column with per-case TOS/ZOS perturbation maps.

The right three columns are copied byte-for-byte from the existing overview.
Only the left map interiors are regenerated from saved delayed rank-1 NPZ
fields; no model/CNOP calculation is performed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
DESKTOP_BASE = Path(r"C:\Users\zhen.luo\Desktop\article\overview.png")
BASE = DESKTOP_BASE if DESKTOP_BASE.exists() else ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_pseudo3d_v18.png"
DESKTOP_OUT = Path(r"C:\Users\zhen.luo\Desktop\article\overview_pseudo3d_v18.png")
NPZ_ROOT = ROOT / "tmp" / "pacific_delayed_remote"
CASES = [
    "IPSL-CM6A-LR_1880", "EC-Earth3_1905", "GFDL-ESM4_1960", "MPI-ESM1-2-HR_1863",
    "GFDL-ESM4_1995", "GFDL-ESM4_1938", "EC-Earth3_1970", "EC-Earth3_1889",
    "EC-Earth3_1942", "GFDL-ESM4_1930",
]

X0, X1 = 635, 1431
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 150.0, 330.0, -35.0, 35.0


def smooth_grid(field: np.ndarray, passes: int = 1) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
    kernel /= kernel.sum()
    out = np.asarray(field, dtype=float).copy()
    valid = np.isfinite(out)
    out = np.where(valid, out, 0.0)
    weights = valid.astype(float)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        padded_w = np.pad(weights, ((1, 1), (1, 1)), mode="edge")
        new_out = np.zeros_like(out); new_w = np.zeros_like(weights)
        for i in range(3):
            for j in range(3):
                new_out += kernel[i, j] * padded[i:i + out.shape[0], j:j + out.shape[1]]
                new_w += kernel[i, j] * padded_w[i:i + out.shape[0], j:j + out.shape[1]]
        out = np.divide(new_out, new_w, out=np.full_like(new_out, np.nan), where=new_w > 0)
        weights = (new_w > 0).astype(float)
    return np.where(weights > 0, out, np.nan)


def interpolate_grid(field: np.ndarray, factor: int = 4) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    up_values = zoom(np.where(valid, values, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    up_weights = zoom(valid.astype(float), (factor, factor), order=1, mode="nearest")
    return np.divide(up_values, up_weights, out=np.full_like(up_values, np.nan), where=up_weights > 0.35)


def make_panel(tos: np.ndarray, zos: np.ndarray, lon: np.ndarray, lat: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    slon = (lon >= LON_MIN) & (lon <= LON_MAX)
    slat = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon0, lat0 = lon[slon], lat[slat]
    tos0 = tos[np.ix_(slat, slon)]
    zos0 = zos[np.ix_(slat, slon)]
    factor = 4
    lon_hi = np.linspace(lon0[0], lon0[-1], tos0.shape[1] * factor)
    lat_hi = np.linspace(lat0[0], lat0[-1], tos0.shape[0] * factor)
    tos_hi = interpolate_grid(smooth_grid(tos0, passes=1), factor=factor)
    zos_hi = interpolate_grid(smooth_grid(zos0, passes=1), factor=factor)
    tos_hi = np.clip(tos_hi, -0.8, 0.8)
    zos_hi = np.clip(zos_hi, -0.03, 0.03)

    # Match the supplied mean-perturbation figure exactly: RdYlBu_r for TOS
    # at +/-0.8 C and BrBG for ZOS at +/-0.03.
    tos_cmap = mpl.colormaps["RdYlBu_r"].with_extremes(bad="#F4F1E8")
    zos_cmap = mpl.colormaps["BrBG"].with_extremes(bad="#F4F1E8")

    fig = plt.figure(figsize=((X1 - X0) / 100.0, MAP_HEIGHT / 100.0), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top = ((0.08, 0.515), (0.96, 0.515), (0.18, 0.82), (0.88, 0.82))
    bottom = ((0.08, 0.20), (0.96, 0.20), (0.18, 0.49), (0.88, 0.49))

    def map_xy(lo, la, corners):
        u = (lo - LON_MIN) / (LON_MAX - LON_MIN)
        v = (la - LAT_MIN) / (LAT_MAX - LAT_MIN)
        sw, se, nw, ne = corners
        x = (1-u)*(1-v)*sw[0] + u*(1-v)*se[0] + (1-u)*v*nw[0] + u*v*ne[0]
        y = (1-u)*(1-v)*sw[1] + u*(1-v)*se[1] + (1-u)*v*nw[1] + u*v*ne[1]
        return x, y

    ax.add_patch(Polygon([bottom[2], top[2], top[0], bottom[0]], facecolor="#B8B8B8", edgecolor="#777777", lw=0.25, alpha=0.15))
    ax.add_patch(Polygon([top[1], top[3], bottom[3], bottom[1]], facecolor="#B8B8B8", edgecolor="#777777", lw=0.25, alpha=0.15))
    lon_grid, lat_grid = np.meshgrid(lon_hi, lat_hi)
    # Use the original map's land pixels as a stable mask for both layers.
    from PIL import Image as _Image
    mask_hi = np.asarray(_Image.fromarray(land_mask.astype(np.uint8) * 255).resize((tos_hi.shape[1], tos_hi.shape[0]), _Image.Resampling.NEAREST)) > 127
    for field, corners, cmap, vmax, label in ((tos_hi, top, tos_cmap, 0.8, "TOS"), (zos_hi, bottom, zos_cmap, 0.03, "ZOS")):
        field = np.where(mask_hi, np.nan, field)
        xg, yg = map_xy(lon_grid, lat_grid, corners)
        levels = np.linspace(-vmax, vmax, 28)
        ax.contourf(xg, yg, field, levels=levels, cmap=cmap, extend="both", antialiased=True, zorder=1)
        ax.contour(xg, yg, field, levels=levels[::2], colors="#4B5563", linewidths=0.22, alpha=0.28, zorder=3)
        outline = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
        ax.plot(outline[:, 0], outline[:, 1], color="#343A40", lw=0.45, zorder=4)
        if label == "TOS":
            ax.text(corners[3][0] - 0.01, corners[3][1] + 0.012, "mean TOS perturbation (°C)", ha="right", va="bottom", fontsize=8.5, fontweight="bold", color="#222222", zorder=5)
        else:
            ax.text(0.50, 0.503, "mean ZOS perturbation", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#222222", zorder=5)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    # Rendered panel is exactly the map-interior size at dpi=1; preserve land
    # from the original overview's map mask.
    rgba[..., :3][land_mask] = 255
    rgba[..., 3][land_mask] = 255
    return rgba


def main() -> None:
    base = np.asarray(Image.open(BASE).convert("RGBA"))
    out = base.copy()
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 22)
    for row, case in enumerate(CASES):
        path = NPZ_ROOT / case / f"case_{case}.npz"
        with np.load(path) as data:
            delta = np.asarray(data["delta_phys"], dtype=np.float32)
            lon = np.asarray(data["lon"], dtype=float)
            lat = np.asarray(data["lat"], dtype=float)
        top = ROW_TOPS[row]
        original_panel = base[top : top + MAP_HEIGHT, X0:X1, :3]
        land_mask = original_panel.mean(axis=2) > 245
        panel = make_panel(delta[0], delta[1], lon, lat, land_mask)
        # Remove the old latitude/longitude frame and ticks from the entire
        # left-column cell so the pseudo-3D surfaces are not nested inside it.
        out[top - 3 : top + MAP_HEIGHT + 3, 450 : X1 + 5] = 255
        out[top : top + MAP_HEIGHT, X0:X1] = panel

    # Correct the first-column title to reflect the two perturbation fields.
    image = Image.fromarray(out, mode="RGBA")
    draw = ImageDraw.Draw(image)
    draw.rectangle((X0, 190, X1, 228), fill="white")
    draw.text(((X0 + X1) // 2, 192), "Initial delta TOS + ZOS", fill="#222222", font=font, anchor="ma")
    image.save(OUT, dpi=(320, 320))
    if DESKTOP_OUT.parent.exists():
        image.save(DESKTOP_OUT, dpi=(320, 320))
    print(OUT)


if __name__ == "__main__":
    main()
