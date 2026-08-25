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
BASE = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_tos_zos.png"
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


def smooth_zoom(field: np.ndarray, factor: int = 4, sigma: float = 1.8) -> np.ndarray:
    f = np.asarray(field, dtype=np.float32)
    f = gaussian_filter(f, sigma=sigma, mode="nearest")
    return zoom(f, (factor, factor), order=3, mode="nearest", prefilter=True)


def make_panel(tos: np.ndarray, zos: np.ndarray, lon: np.ndarray, lat: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    slon = (lon >= LON_MIN) & (lon <= LON_MAX)
    slat = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon0, lat0 = lon[slon], lat[slat]
    tos0 = tos[np.ix_(slat, slon)]
    zos0 = zos[np.ix_(slat, slon)]
    factor = 4
    lon_hi = np.linspace(lon0[0], lon0[-1], tos0.shape[1] * factor)
    lat_hi = np.linspace(lat0[0], lat0[-1], tos0.shape[0] * factor)
    tos_hi = smooth_zoom(tos0, factor=factor)
    zos_hi = smooth_zoom(zos0, factor=factor)
    tos_hi = np.clip(tos_hi, -0.8, 0.8)
    zos_hi = np.clip(zos_hi, -0.03, 0.03)

    # Soft palettes retain sign while reducing saturation.
    tos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "soft_tos", ["#5B8DB8", "#A8C7D9", "#F3EAD5", "#E9A89A", "#C65B5B"], N=256
    )
    zos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "soft_zos", ["#4C8A76", "#9BC5A5", "#F0E7C7", "#E2C56A", "#B8943E"], N=256
    )

    fig = plt.figure(figsize=((X1 - X0) / 100.0, MAP_HEIGHT / 100.0), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top = ((0.08, 0.55), (0.96, 0.55), (0.18, 0.84), (0.88, 0.84))
    bottom = ((0.08, 0.19), (0.96, 0.19), (0.18, 0.48), (0.88, 0.48))

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
        ax.contour(xg, yg, field, levels=levels[::2], colors="#59636A", linewidths=0.18, alpha=0.36, zorder=3)
        outline = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
        ax.plot(outline[:, 0], outline[:, 1], color="#343A40", lw=0.45, zorder=4)
        ax.text(corners[0][0] + 0.015, corners[2][1] - 0.02, label, fontsize=7.5, fontweight="bold", color="#263238",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5}, zorder=5)
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
    print(OUT)


if __name__ == "__main__":
    main()
