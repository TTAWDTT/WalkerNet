"""Rebuild the complete Pacific overview on one shared pixel/coordinate grid.

The right-hand forecast panels are retained as the supplied reference layer;
the entire left perturbation column is freshly rendered from saved delta_phys
fields, including its frame, grid lines, and tick calibration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\Users\zhen.luo\Desktop\article\overview.png")
if not SOURCE.exists():
    SOURCE = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_final_v28.png"
DESKTOP_OUT = Path(r"C:\Users\zhen.luo\Desktop\article\overview_final_v28.png")
NPZ_ROOT = ROOT / "tmp" / "pacific_delayed_remote"
CASES = [
    "IPSL-CM6A-LR_1880", "EC-Earth3_1905", "GFDL-ESM4_1960", "MPI-ESM1-2-HR_1863",
    "GFDL-ESM4_1995", "GFDL-ESM4_1938", "EC-Earth3_1970", "EC-Earth3_1889",
    "EC-Earth3_1942", "GFDL-ESM4_1930",
]

WIDTH, HEIGHT = 4207, 5126
LEFT_X0, LEFT_X1 = 635, 1431
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 120.0, 300.0, -35.0, 35.0


def smooth_interpolate(field: np.ndarray, factor: int = 8, sigma: float = 1.15) -> np.ndarray:
    valid = np.isfinite(field)
    num = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smooth = np.divide(num, den, out=np.full_like(field, np.nan, dtype=float), where=den > 1e-8)
    values = zoom(np.where(np.isfinite(smooth), smooth, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    weights = zoom(np.isfinite(smooth).astype(float), (factor, factor), order=1, mode="nearest")
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0.35)


def render_left_panel(tos: np.ndarray, lon: np.ndarray, lat: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    slon = (lon >= LON_MIN) & (lon <= LON_MAX)
    slat = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    data = tos[np.ix_(slat, slon)].astype(float)
    lon0, lat0 = lon[slon], lat[slat]
    factor = 8
    hi = np.clip(smooth_interpolate(data, factor=factor), -0.8, 0.8)
    lon_hi = np.linspace(lon0[0], lon0[-1], hi.shape[1])
    lat_hi = np.linspace(lat0[0], lat0[-1], hi.shape[0])
    mask_hi = np.asarray(
        Image.fromarray((land_mask.astype(np.uint8) * 255)).resize((hi.shape[1], hi.shape[0]), Image.Resampling.NEAREST)
    ) > 127
    hi[mask_hi] = np.nan
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "reference_soft_red_blue", ["#2F6DA3", "#86B9D8", "#D9D8D5", "#E89A86", "#B63E3E"], N=256
    ).with_extremes(bad="#FFFFFF")
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    h, w = land_mask.shape
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX); ax.axis("off")
    lg, ag = np.meshgrid(lon_hi, lat_hi)
    levels = np.linspace(-0.8, 0.8, 81)
    ax.contourf(lg, ag, hi, levels=levels, cmap=cmap, norm=norm, extend="both", antialiased=True)
    # Grid is drawn in the same coordinate system as the right-hand panels.
    for lo in (120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0):
        ax.plot([lo, lo], [LAT_MIN, LAT_MAX], color="#9AA3AF", linewidth=0.35, alpha=0.28)
    for la in (-18.0, 0.0, 18.0):
        ax.plot([LON_MIN, LON_MAX], [la, la], color="#9AA3AF", linewidth=0.48 if la == 0 else 0.35, alpha=0.40)
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    rendered[..., :3][land_mask] = 255
    rendered[..., 3][land_mask] = 255
    return rendered


def main() -> None:
    base = np.asarray(Image.open(SOURCE).convert("RGBA"))
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    # Copy only the non-left-column content from the supplied complete figure:
    # title/header and row labels, plus the right three forecast columns.
    # Copy only the case-name strip. The old left-axis tick labels extend well
    # beyond this strip, so no old tick glyphs can leak into the new panel.
    canvas.paste(Image.fromarray(base[:, :360]), (0, 0))
    canvas.paste(Image.fromarray(base[:230, 360:]), (360, 0))
    canvas.paste(Image.fromarray(base[230:, 1450:]), (1450, 230))

    for row, case in enumerate(CASES):
        path = NPZ_ROOT / case / f"case_{case}.npz"
        with np.load(path) as data:
            delta = np.asarray(data["delta_phys"], dtype=np.float32)
            lon = np.asarray(data["lon"], dtype=float)
            lat = np.asarray(data["lat"], dtype=float)
        top = ROW_TOPS[row]
        # Land mask comes from the corresponding original map cell only; no
        # old color pixels are reused in the newly rendered field.
        old = base[top : top + MAP_HEIGHT, LEFT_X0:LEFT_X1]
        old_mask = old[..., :3].mean(axis=2) > 245
        start = int(round((120.0 - 100.0) / (300.0 - 100.0) * old_mask.shape[1]))
        land_mask = np.asarray(Image.fromarray((old_mask[:, start:] * 255).astype(np.uint8)).resize((old_mask.shape[1], old_mask.shape[0]), Image.Resampling.NEAREST)) > 127
        panel = render_left_panel(delta[0], lon, lat, land_mask)
        canvas.paste(Image.fromarray(panel, mode="RGBA"), (LEFT_X0, top))

        draw = ImageDraw.Draw(canvas)
        # Crisp common frame and latitude ticks for every left panel.
        draw.rectangle((LEFT_X0, top, LEFT_X1 - 1, top + MAP_HEIGHT - 1), outline="#30343A", width=2)
        for la in (-35.0, -18.0, 0.0, 18.0, 35.0):
            y = int(round(top + (LAT_MAX - la) / (LAT_MAX - LAT_MIN) * (MAP_HEIGHT - 1)))
            draw.line((LEFT_X0 - 8, y, LEFT_X0, y), fill="#30343A", width=2)
            if row == 0 or True:
                label = "35N" if la == 35 else "18N" if la == 18 else "0" if la == 0 else "18S" if la == -18 else "35S"
                draw.text((LEFT_X0 - 12, y), label, fill="#30343A", font=ImageFont.truetype("C:/Windows/Fonts/times.ttf", 16), anchor="rm")
        if row == len(CASES) - 1:
            labels = ("120E", "150E", "180", "150W", "120W", "90W", "60W")
            y = top + MAP_HEIGHT + 8
            for i, label in enumerate(labels):
                x = int(round(LEFT_X0 + i * (LEFT_X1 - LEFT_X0 - 1) / 6.0))
                draw.text((x, y), label, fill="#30343A", font=ImageFont.truetype("C:/Windows/Fonts/times.ttf", 16), anchor="ma")

    # Restore the left-column title only, while keeping the global title from
    # the supplied reference header.
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((360, 55, 635, 245), fill="white")
    draw.rectangle((635, 55, 1450, 225), fill="white")
    # The cleanup above intentionally reaches below the header to remove the
    # old clipped 35N glyph; restore the new first-row tick afterward.
    first_y = ROW_TOPS[0]
    first_tick_y = int(round(first_y + (LAT_MAX - 35.0) / (LAT_MAX - LAT_MIN) * (MAP_HEIGHT - 1)))
    draw.line((LEFT_X0 - 8, first_tick_y, LEFT_X0, first_tick_y), fill="#30343A", width=2)
    draw.text((LEFT_X0 - 12, first_tick_y), "35N", fill="#30343A", font=ImageFont.truetype("C:/Windows/Fonts/times.ttf", 16), anchor="rm")
    draw.text(((LEFT_X0 + LEFT_X1) // 2, 190), "Initial delta TOS", fill="#222222", font=ImageFont.truetype("C:/Windows/Fonts/times.ttf", 22), anchor="ma")
    canvas.save(OUT, dpi=(320, 320))
    if DESKTOP_OUT.parent.exists():
        canvas.save(DESKTOP_OUT, dpi=(320, 320))
    print(OUT)


if __name__ == "__main__":
    main()
