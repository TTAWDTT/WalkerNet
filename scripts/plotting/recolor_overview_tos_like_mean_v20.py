"""Recolor only the original overview TOS perturbation interiors.

The layout, axes, labels, right-hand forecast columns, and colorbars are
preserved byte-for-byte from the supplied overview.  Only the ten left-column
map interiors are smoothed and remapped to the mean-perturbation figure's
blue--pale-yellow--red palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
DESKTOP_SOURCE = Path(r"C:\Users\zhen.luo\Desktop\article\overview.png")
SOURCE = DESKTOP_SOURCE if DESKTOP_SOURCE.exists() else ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUTPUT = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_tos_recolored_smooth_v20.png"
DESKTOP_OUTPUT = Path(r"C:\Users\zhen.luo\Desktop\article\overview_tos_recolored_smooth_v20.png")

X0, X1 = 635, 1431
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434


def refine(panel: np.ndarray) -> np.ndarray:
    rgb = panel[..., :3].astype(np.float32) / 255.0
    # White land and transparent pixels are retained exactly as white.
    valid = ~((rgb.mean(axis=2) > 0.975) | (panel[..., 3] < 128))
    # Recover a signed display-only scalar from the existing red/blue raster,
    # then smooth and remap it. No model or CNOP values are recomputed.
    scalar = (rgb[..., 0] - rgb[..., 2]) * valid
    scalar_hi = zoom(scalar, (4, 4), order=3, mode="nearest", prefilter=True)
    valid_hi = zoom(valid.astype(float), (4, 4), order=1, mode="nearest") > 0.40
    numerator = gaussian_filter(np.where(valid_hi, scalar_hi, 0.0), sigma=3.2, mode="nearest")
    denominator = gaussian_filter(valid_hi.astype(float), sigma=3.2, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-5)
    smooth = zoom(smooth, (1 / 4, 1 / 4), order=3, mode="nearest", prefilter=True)
    smooth = np.clip(smooth, -0.95, 0.95)

    # Exact soft TOS palette used by the supplied reference figure's upper
    # panel: muted blue negative anomalies, gray-white near zero, and brick-red
    # positive anomalies.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "reference_soft_red_blue",
        ["#2F6DA3", "#86B9D8", "#D9D8D5", "#E89A86", "#B63E3E"],
        N=256,
    )
    mapped = cmap((smooth + 0.95) / 1.90)[..., :3]
    out = np.rint(mapped * 255.0).astype(np.uint8)
    out[~valid] = 255
    return np.dstack([out, panel[..., 3]])


def main() -> None:
    base = np.asarray(Image.open(SOURCE).convert("RGBA"))
    out = base.copy()
    for top in ROW_TOPS:
        panel = base[top : top + MAP_HEIGHT, X0:X1]
        out[top : top + MAP_HEIGHT, X0:X1] = refine(panel)
    Image.fromarray(out, mode="RGBA").save(OUTPUT, dpi=(320, 320))
    if DESKTOP_OUTPUT.parent.exists():
        Image.fromarray(out, mode="RGBA").save(DESKTOP_OUTPUT, dpi=(320, 320))
    print(OUTPUT)


if __name__ == "__main__":
    main()
