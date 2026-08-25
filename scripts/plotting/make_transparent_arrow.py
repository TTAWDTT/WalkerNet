"""Remove the white raster background from the standalone residual arrow."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "figures" / "walkernet_residual_arrow.png"
dst = ROOT / "figures" / "walkernet_residual_arrow_transparent.png"

image = Image.open(src).convert("RGBA")
pixels = []
for r, g, b, _ in image.getdata():
    # The raster was rendered over white. Use the darkest channel as a
    # conservative opacity estimate, preserving anti-aliased edge pixels.
    alpha = 255 - min(r, g, b)
    pixels.append((r, g, b, alpha))
image.putdata(pixels)
image.save(dst, format="PNG", optimize=True)
print(dst)
