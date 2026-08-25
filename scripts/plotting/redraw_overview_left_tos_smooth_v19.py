"""Replace the overview's perturbation column with smooth planar TOS maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
DESKTOP_BASE = Path(r"C:\Users\zhen.luo\Desktop\article\overview.png")
BASE = DESKTOP_BASE if DESKTOP_BASE.exists() else ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_tos_smooth_v19.png"
DESKTOP_OUT = Path(r"C:\Users\zhen.luo\Desktop\article\overview_tos_smooth_v19.png")
NPZ_ROOT = ROOT / "tmp" / "pacific_delayed_remote"
CASES = [
    "IPSL-CM6A-LR_1880", "EC-Earth3_1905", "GFDL-ESM4_1960", "MPI-ESM1-2-HR_1863",
    "GFDL-ESM4_1995", "GFDL-ESM4_1938", "EC-Earth3_1970", "EC-Earth3_1889",
    "EC-Earth3_1942", "GFDL-ESM4_1930",
]

X0, X1 = 635, 1431
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 100.0, 300.0, -35.0, 35.0


def smooth_interpolate(field: np.ndarray, factor: int = 6, sigma: float = 1.35) -> np.ndarray:
    valid = np.isfinite(field)
    numerator = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.full_like(field, np.nan, dtype=float), where=denominator > 1e-8)
    values = zoom(np.where(np.isfinite(smooth), smooth, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    weights = zoom(np.isfinite(smooth).astype(float), (factor, factor), order=1, mode="nearest")
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0.35)


def make_panel(tos: np.ndarray, lon: np.ndarray, lat: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    slon = (lon >= LON_MIN) & (lon <= LON_MAX)
    slat = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon0, lat0 = lon[slon], lat[slat]
    tos0 = tos[np.ix_(slat, slon)]
    factor = 6
    tos_hi = np.clip(smooth_interpolate(tos0, factor=factor), -0.8, 0.8)
    lon_hi = np.linspace(lon0[0], lon0[-1], tos_hi.shape[1])
    lat_hi = np.linspace(lat0[0], lat0[-1], tos_hi.shape[0])
    mask_hi = np.asarray(
        Image.fromarray((land_mask.astype(np.uint8) * 255)).resize(
            (tos_hi.shape[1], tos_hi.shape[0]), Image.Resampling.NEAREST
        )
    ) > 127
    tos_hi[mask_hi] = np.nan

    cmap = mpl.colormaps["RdYlBu_r"].with_extremes(bad="#FFFFFF")
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    fig = plt.figure(figsize=((X1 - X0) / 100.0, MAP_HEIGHT / 100.0), dpi=100, facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX); ax.axis("off")
    lon_grid, lat_grid = np.meshgrid(lon_hi, lat_hi)
    levels = np.linspace(-0.8, 0.8, 45)
    ax.contourf(lon_grid, lat_grid, tos_hi, levels=levels, cmap=cmap, norm=norm, extend="both", antialiased=True)
    ax.contour(lon_grid, lat_grid, tos_hi, levels=levels[::2], colors="#65717B", linewidths=0.18, alpha=0.25)
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    rgba[..., :3][land_mask] = 255
    rgba[..., 3][land_mask] = 255
    return rgba


def main() -> None:
    base = np.asarray(Image.open(BASE).convert("RGBA"))
    out = base.copy()
    for row, case in enumerate(CASES):
        path = NPZ_ROOT / case / f"case_{case}.npz"
        with np.load(path) as data:
            delta = np.asarray(data["delta_phys"], dtype=np.float32)
            lon = np.asarray(data["lon"], dtype=float)
            lat = np.asarray(data["lat"], dtype=float)
        top = ROW_TOPS[row]
        original_panel = base[top : top + MAP_HEIGHT, X0:X1, :3]
        land_mask = original_panel.mean(axis=2) > 245
        panel = make_panel(delta[0], lon, lat, land_mask)
        out[top - 3 : top + MAP_HEIGHT + 3, 450 : X1 + 5] = 255
        out[top : top + MAP_HEIGHT, X0:X1] = panel

    image = Image.fromarray(out, mode="RGBA")
    draw = ImageDraw.Draw(image)
    draw.rectangle((X0, 185, X1, 228), fill="white")
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 24)
    draw.text(((X0 + X1) // 2, 191), "Initial ΔTOS", fill="#222222", font=font, anchor="ma")
    image.save(OUT, dpi=(320, 320))
    if DESKTOP_OUT.parent.exists():
        image.save(DESKTOP_OUT, dpi=(320, 320))
    print(OUT)


if __name__ == "__main__":
    main()
