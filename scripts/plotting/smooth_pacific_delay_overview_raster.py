"""Raster-only cosmetic smoothing for the existing Pacific delayed overview.

This does not recompute CNOP or alter any map values. It operates only on the
interior pixels of the ten initial-delta-TOS panels, preserving the existing
layout, labels, colorbars, and all three response columns. Land/invalid white
pixels are kept masked while the ocean raster is upsampled, Gaussian-smoothed,
and returned to the original panel dimensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "assets" / "article" / "pacific_delay_overview.png"
OUTPUTS = [
    SOURCE.with_name("pacific_delay_overview_smoothed.png"),
    ROOT / "docs" / "assets" / "walkernet_rollout_skill" / "pacific_delay_overview_smoothed.png",
]

# Measured from the existing 4207x5126 overview raster. Borders, labels, and
# colorbars are deliberately excluded; only map interiors are replaced.
X0, X1 = 635, 1431
ROW_TOPS = [230, 716, 1201, 2173, 2659, 3144, 3630, 4116, 4602]
# There are ten rows; the regular 486-pixel pitch includes the 47-pixel gap.
ROW_TOPS = [230 + 486 * i for i in range(10)]
MAP_HEIGHT = 434


def smooth_masked_ocean(rgba: np.ndarray, sigma: float = 2.2, upsample: int = 2) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    white = (rgb.mean(axis=2) > 0.975) | (alpha[..., 0] < 0.5)
    valid = (~white).astype(np.float32)

    # Upsample before smoothing so contour stair-steps are not merely blurred
    # at the original coarse raster resolution.
    rgb_hi = zoom(rgb, (upsample, upsample, 1), order=3)
    valid_hi = zoom(valid, (upsample, upsample), order=3).clip(0.0, 1.0)
    sig = sigma * upsample
    smooth = np.empty_like(rgb_hi)
    weight = gaussian_filter(valid_hi, sig, mode="nearest")
    for channel in range(3):
        smooth[..., channel] = gaussian_filter(rgb_hi[..., channel] * valid_hi, sig, mode="nearest") / np.maximum(weight, 1e-4)
    smooth = zoom(smooth, (1 / upsample, 1 / upsample, 1), order=3).clip(0.0, 1.0)
    valid_back = zoom(valid_hi, (1 / upsample, 1 / upsample), order=3).clip(0.0, 1.0)
    out = np.rint(smooth * 255.0).astype(np.uint8)
    # Preserve the original land/invalid mask, with a narrow anti-aliased edge.
    out[valid_back < 0.45] = 255
    out = np.concatenate([out, rgba[..., 3:4]], axis=2)
    return out


def main() -> None:
    image = Image.open(SOURCE).convert("RGBA")
    rgba = np.asarray(image)
    out = rgba.copy()
    for top in ROW_TOPS:
        bottom = top + MAP_HEIGHT
        panel = rgba[top:bottom, X0:X1]
        out[top:bottom, X0:X1] = smooth_masked_ocean(panel, sigma=2.2, upsample=2)

    for destination in OUTPUTS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, mode="RGBA").save(destination, dpi=(320, 320))
        print(destination)


if __name__ == "__main__":
    main()
