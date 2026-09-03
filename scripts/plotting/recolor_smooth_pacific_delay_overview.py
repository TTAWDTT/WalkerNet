"""Recolor and smoothly refine the existing single-column Pacific overview.

Only the ten initial-delta-TOS map interiors are changed. The signed raster
proxy is smoothed and remapped to the exact overview TOS palette; all other
columns, labels, colorbars, and layout pixels are preserved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUTPUTS = [
    ROOT / "docs" / "assets" / "article" / "pacific_delay_overview_recolored_smoothed.png",
    ROOT / "docs" / "assets" / "walkernet_rollout_skill" / "pacific_delay_overview_recolored_smoothed.png",
]
X0, X1 = 635, 1431
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434

TARGET_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "overview_tos_exact",
    ["#4B56A6", "#8FC7D9", "#F7F3D0", "#F0A35A", "#B61732"],
    N=256,
)


def refine(panel: np.ndarray) -> np.ndarray:
    rgb = panel[..., :3].astype(np.float32) / 255.0
    valid = ~((rgb.mean(axis=2) > 0.975) | (panel[..., 3] < 128))
    # The original left-column map uses a red-blue signed diverging palette.
    # Red-blue separation is used only to recover a display scalar from the
    # already-rendered raster; it is not used for any scientific metric.
    scalar = (rgb[..., 0] - rgb[..., 2]) * valid
    scalar_hi = zoom(scalar, (3, 3), order=3, mode="nearest", prefilter=True)
    valid_hi = zoom(valid.astype(float), (3, 3), order=1, mode="nearest") > 0.45
    numerator = gaussian_filter(np.where(valid_hi, scalar_hi, 0.0), sigma=2.1, mode="nearest")
    denominator = gaussian_filter(valid_hi.astype(float), sigma=2.1, mode="nearest")
    smooth = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-5)
    smooth = zoom(smooth, (1 / 3, 1 / 3), order=3, mode="nearest", prefilter=True)
    smooth = np.clip(smooth, -0.95, 0.95)
    mapped = TARGET_CMAP((smooth + 0.95) / 1.9)[..., :3]
    out = np.rint(mapped * 255.0).astype(np.uint8)
    out[~valid] = 255
    return np.dstack([out, panel[..., 3]])


def main() -> None:
    base = np.asarray(Image.open(SOURCE).convert("RGBA"))
    out = base.copy()
    for top in ROW_TOPS:
        panel = base[top : top + MAP_HEIGHT, X0:X1]
        out[top : top + MAP_HEIGHT, X0:X1] = refine(panel)
    for destination in OUTPUTS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, mode="RGBA").save(destination, dpi=(320, 320))
        print(destination)


if __name__ == "__main__":
    main()
