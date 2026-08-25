"""Render the overview's left TOS perturbations directly from saved fields.

Unlike raster recoloring, this version uses the stored ``delta_phys`` values,
so the displayed contours are not constrained by square pixels from the old
screenshot.  The original overview layout and all non-perturbation pixels are
preserved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
DESKTOP_BASE = Path(r"C:\Users\zhen.luo\Desktop\article\overview.png")
BASE = DESKTOP_BASE if DESKTOP_BASE.exists() else ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_tos_data_smooth_v22.png"
DESKTOP_OUT = Path(r"C:\Users\zhen.luo\Desktop\article\overview_tos_data_smooth_v22.png")
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


def nan_smooth_interpolate(field: np.ndarray, factor: int = 8, sigma: float = 1.15) -> np.ndarray:
    valid = np.isfinite(field)
    numerator = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.full_like(field, np.nan, dtype=float), where=denominator > 1e-8)
    values = zoom(np.where(np.isfinite(smooth), smooth, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    weights = zoom(np.isfinite(smooth).astype(float), (factor, factor), order=1, mode="nearest")
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0.35)


def render_panel(tos: np.ndarray, lon: np.ndarray, lat: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    slon = (lon >= LON_MIN) & (lon <= LON_MAX)
    slat = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    tos0 = np.asarray(tos[np.ix_(slat, slon)], dtype=float)
    lon0, lat0 = lon[slon], lat[slat]
    factor = 8
    tos_hi = np.clip(nan_smooth_interpolate(tos0, factor=factor), -0.8, 0.8)
    lon_hi = np.linspace(lon0[0], lon0[-1], tos_hi.shape[1])
    lat_hi = np.linspace(lat0[0], lat0[-1], tos_hi.shape[0])
    mask_hi = np.asarray(
        Image.fromarray((land_mask.astype(np.uint8) * 255)).resize(
            (tos_hi.shape[1], tos_hi.shape[0]), Image.Resampling.NEAREST
        )
    ) > 127
    tos_hi[mask_hi] = np.nan

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "reference_soft_red_blue",
        ["#2F6DA3", "#86B9D8", "#D9D8D5", "#E89A86", "#B63E3E"], N=256,
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    h, w = land_mask.shape
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX); ax.axis("off")
    lon_grid, lat_grid = np.meshgrid(lon_hi, lat_hi)
    levels = np.linspace(-0.8, 0.8, 81)
    ax.contourf(lon_grid, lat_grid, tos_hi, levels=levels, cmap=cmap, norm=norm, extend="both", antialiased=True)
    ax.contour(lon_grid, lat_grid, tos_hi, levels=levels[::4], colors="#65717B", linewidths=0.15, alpha=0.22)
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    rendered[..., :3][land_mask] = 255
    rendered[..., 3][land_mask] = 255
    return rendered


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
        original = base[top : top + MAP_HEIGHT, X0:X1]
        land_mask = original[..., :3].mean(axis=2) > 245
        panel = render_panel(delta[0], lon, lat, land_mask)
        # Restore a narrow frame/tick strip from the original overview so the
        # operation changes only the map interior, not the layout.
        panel[:8] = original[:8]; panel[-10:] = original[-10:]
        panel[:, :10] = original[:, :10]; panel[:, -8:] = original[:, -8:]
        out[top : top + MAP_HEIGHT, X0:X1] = panel
    Image.fromarray(out, mode="RGBA").save(OUT, dpi=(320, 320))
    if DESKTOP_OUT.parent.exists():
        Image.fromarray(out, mode="RGBA").save(DESKTOP_OUT, dpi=(320, 320))
    print(OUT)


if __name__ == "__main__":
    main()
