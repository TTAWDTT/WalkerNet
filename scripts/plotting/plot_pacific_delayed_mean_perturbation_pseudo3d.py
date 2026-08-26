"""Plot the ten-case mean Pacific delayed CNOP perturbation.

The visual layer deliberately follows the supplied EKE/heat-content figure:
one upper TOS map and one lower ZOS map are drawn on skewed quadrilaterals,
with one shared zero-centred color scale per variable and one NaN-aware
Gaussian-like smoothing pass for display only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import zoom


LON_MIN, LON_MAX = 120.0, 280.0
LAT_MIN, LAT_MAX = -20.0, 20.0


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 12,
            "axes.titlesize": 17,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def smooth_grid(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
    kernel /= kernel.sum()
    out = np.asarray(grid, dtype=float).copy()
    valid = np.isfinite(out)
    out = np.where(valid, out, 0.0)
    weights = valid.astype(float)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        padded_w = np.pad(weights, ((1, 1), (1, 1)), mode="edge")
        new_out = np.zeros_like(out)
        new_w = np.zeros_like(weights)
        for i in range(3):
            for j in range(3):
                new_out += kernel[i, j] * padded[i : i + out.shape[0], j : j + out.shape[1]]
                new_w += kernel[i, j] * padded_w[i : i + out.shape[0], j : j + out.shape[1]]
        out = np.divide(new_out, new_w, out=np.full_like(new_out, np.nan), where=new_w > 0)
        weights = (new_w > 0).astype(float)
    return np.where(weights > 0, out, np.nan)


def interpolate_grid(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    """Upsample a display field with bicubic interpolation; never used for metrics."""
    if factor <= 1:
        return np.asarray(grid, dtype=float)
    values = np.asarray(grid, dtype=float)
    valid = np.isfinite(values)
    up_values = zoom(np.where(valid, values, 0.0), zoom=(factor, factor), order=3, mode="nearest", prefilter=True)
    up_weights = zoom(valid.astype(float), zoom=(factor, factor), order=3, mode="nearest", prefilter=True)
    return np.divide(up_values, up_weights, out=np.full_like(up_values, np.nan), where=up_weights > 0.35)


def land_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Rasterize Natural Earth land before smoothing so land zeros do not bleed into ocean."""
    try:
        from cartopy.io import shapereader
        shp = shapereader.natural_earth(resolution="50m", category="physical", name="land")
        geometries = shapereader.Reader(shp).geometries()
    except Exception:
        return np.zeros((lat.size, lon.size), dtype=bool)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.c_[lon_grid.ravel(), lat_grid.ravel()]
    mask = np.zeros(points.shape[0], dtype=bool)
    for geom in geometries:
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            if coords.ndim != 2 or coords.shape[0] < 3:
                continue
            poly_lon = np.where(coords[:, 0] < 0.0, coords[:, 0] + 360.0, coords[:, 0])
            poly_lat = coords[:, 1]
            keep = (poly_lon >= LON_MIN - 8.0) & (poly_lon <= LON_MAX + 8.0) & (poly_lat >= LAT_MIN - 8.0) & (poly_lat <= LAT_MAX + 8.0)
            if np.count_nonzero(keep) >= 3:
                mask |= MplPath(np.c_[poly_lon[keep], poly_lat[keep]]).contains_points(points)
    return mask.reshape(lat.size, lon.size)


def bilinear_map(u: np.ndarray, v: np.ndarray, corners):
    sw, se, nw, ne = corners
    x = (1 - u) * (1 - v) * sw[0] + u * (1 - v) * se[0] + (1 - u) * v * nw[0] + u * v * ne[0]
    y = (1 - u) * (1 - v) * sw[1] + u * (1 - v) * se[1] + (1 - u) * v * nw[1] + u * v * ne[1]
    return x, y


def lonlat_to_surface(lon: np.ndarray, lat: np.ndarray, corners):
    u = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    v = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    return bilinear_map(u, v, corners)


def make_clip_patch(corners) -> PathPatch:
    return PathPatch(MplPath(np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])))


def draw_land(ax: plt.Axes, corners) -> None:
    """Reuse the reference figure's Natural Earth land treatment when available."""
    try:
        from cartopy.io import shapereader
    except ImportError:
        return
    try:
        shp = shapereader.natural_earth(resolution="50m", category="physical", name="land")
        geometries = shapereader.Reader(shp).geometries()
    except Exception:
        return
    clip_patch = make_clip_patch(corners)
    clip_patch.set_transform(ax.transData)
    for geom in geometries:
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            if coords.ndim != 2 or coords.shape[0] < 3:
                continue
            lon = np.where(coords[:, 0] < 0.0, coords[:, 0] + 360.0, coords[:, 0])
            lat = coords[:, 1]
            keep = (lon >= LON_MIN - 8.0) & (lon <= LON_MAX + 8.0) & (lat >= LAT_MIN - 8.0) & (lat <= LAT_MAX + 8.0)
            if not np.any(keep):
                continue
            x, y = lonlat_to_surface(lon[keep], lat[keep], corners)
            if x.size < 3:
                continue
            patch = Polygon(np.c_[x, y], closed=True, facecolor="#F4F1E8", edgecolor="#2F3A3F", linewidth=0.28, zorder=4)
            patch.set_clip_path(clip_patch)
            ax.add_patch(patch)


def grid_edges(values: np.ndarray) -> np.ndarray:
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def draw_layer(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, corners, norm, cmap, levels: np.ndarray, label: str, draw_contours: bool = True) -> None:
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    xg, yg = lonlat_to_surface(lon_grid, lat_grid, corners)
    mesh = ax.contourf(xg, yg, field, levels=levels, cmap=cmap, norm=norm, extend="both", antialiased=draw_contours, zorder=1)
    if not draw_contours:
        # Dense smooth fills are display raster content rather than editable
        # vector geometry; rasterizing just these collections keeps the PDF
        # compact while preserving the 300-dpi PNG output.
        mesh.set_rasterized(True)
    if draw_contours:
        ax.contour(xg, yg, field, levels=levels[::2], colors="#4B5563", linewidths=0.22, alpha=0.28, zorder=5)
    draw_land(ax, corners)
    outline = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.05, zorder=6)
    for lon_line in (150.0, 180.0, 210.0, 240.0, 270.0):
        xs, ys = lonlat_to_surface(np.array([lon_line, lon_line]), np.array([LAT_MIN, LAT_MAX]), corners)
        ax.plot(xs, ys, color="black", linewidth=0.40, alpha=0.18, linestyle="--", zorder=5)
    for lat_line in (-10.0, 0.0, 10.0):
        xs, ys = lonlat_to_surface(np.array([LON_MIN, LON_MAX]), np.array([lat_line, lat_line]), corners)
        ax.plot(xs, ys, color="black", linewidth=0.75 if lat_line == 0 else 0.50, alpha=0.52, zorder=5)
    for tick, text in ((20.0, r"20$^\circ$N"), (0.0, r"0$^\circ$"), (-20.0, r"20$^\circ$S")):
        xt, yt = lonlat_to_surface(np.array([LON_MIN]), np.array([tick]), corners)
        ax.text(xt[0] - 0.026, yt[0], text, ha="right", va="center", fontsize=11.2, fontweight="bold")
    if corners[3][1] > 0.60:
        ax.text(corners[3][0] - 0.01, corners[3][1] + 0.012, label, ha="right", va="bottom", fontsize=12.2, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.04})
    else:
        ax.text(0.50, 0.503, label, ha="center", va="center", fontsize=10.0, fontweight="bold", bbox={"facecolor": "#E4E4E4", "edgecolor": "none", "alpha": 0.82, "pad": 0.05})
    return mesh


def plot(input_npz: Path, output: Path, palette_name: str = "soft", draw_contours: bool = True, tos_vmax: float = 0.8, zos_vmax: float = 0.03) -> None:
    set_style()
    with np.load(input_npz, allow_pickle=True) as data:
        mean = np.asarray(data["mean_delta_phys"], dtype=float)
        lat = np.asarray(data["lat"], dtype=float)
        lon = np.asarray(data["lon"], dtype=float)
        case_count = int(np.asarray(data["case_count"]).item())
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    sub_lon, sub_lat = lon[lon_mask], lat[lat_mask]
    land = land_mask(sub_lon, sub_lat)
    tos_raw = np.where(land, np.nan, mean[0][np.ix_(lat_mask, lon_mask)])
    zos_raw = np.where(land, np.nan, mean[1][np.ix_(lat_mask, lon_mask)])
    tos = interpolate_grid(smooth_grid(tos_raw, passes=1), factor=12)
    zos = interpolate_grid(smooth_grid(zos_raw, passes=1), factor=12)
    sub_lon = np.linspace(sub_lon[0], sub_lon[-1], tos.shape[1])
    sub_lat = np.linspace(sub_lat[0], sub_lat[-1], tos.shape[0])
    # The Natural Earth rasterization is done once on the native grid; linearly
    # upsample the mask for a smooth display boundary instead of re-testing
    # millions of high-resolution points against every polygon.
    land_hi = zoom(land.astype(float), zoom=(12, 12), order=1, mode="nearest") >= 0.5
    tos[land_hi] = np.nan
    zos[land_hi] = np.nan
    # Match the legacy response-figure convention: fixed TOS display range
    # ±0.8 °C; ZOS remains in its own physical units at ±0.03.
    if tos_vmax <= 0 or zos_vmax <= 0:
        raise ValueError("tos_vmax and zos_vmax must be positive")
    tos_norm = TwoSlopeNorm(vmin=-tos_vmax, vcenter=0.0, vmax=tos_vmax)
    zos_norm = TwoSlopeNorm(vmin=-zos_vmax, vcenter=0.0, vmax=zos_vmax)
    if palette_name == "original_light":
        def lighten(base):
            colors = base(np.linspace(0.0, 1.0, 256))
            colors[:, :3] = 0.85 * colors[:, :3] + 0.15
            return mpl.colors.ListedColormap(colors)
        tos_cmap = lighten(mpl.colormaps["RdYlBu_r"])
        zos_cmap = lighten(mpl.colormaps["BrBG"])
    elif palette_name == "original":
        tos_cmap = mpl.colormaps["RdYlBu_r"]
        zos_cmap = mpl.colormaps["BrBG"]
    elif palette_name == "contrast":
        tos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "contrast_red_blue", ["#2166AC", "#67A9CF", "#E6E6E6", "#EF8A62", "#B2182B"], N=256
        )
        zos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "contrast_yellow_green", ["#1B7837", "#7FBF7B", "#E8E8D5", "#F6E8A8", "#C89018"], N=256
        )
    else:
        tos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "soft_red_blue", ["#2F6DA3", "#86B9D8", "#D9D8D5", "#E89A86", "#B63E3E"], N=256
        )
        zos_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "soft_yellow_green", ["#28734E", "#72B18B", "#D8D8C8", "#E2C56A", "#B2861A"], N=256
        )
    tos_cmap = tos_cmap.with_extremes(bad="#F4F1E8")
    zos_cmap = zos_cmap.with_extremes(bad="#F4F1E8")

    fig = plt.figure(figsize=(11.8, 8.4))
    ax = fig.add_axes((0.035, 0.035, 0.86, 0.92))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top = ((0.08, 0.515), (0.96, 0.515), (0.18, 0.82), (0.88, 0.82))
    bottom = ((0.08, 0.20), (0.96, 0.20), (0.18, 0.49), (0.88, 0.49))
    # Subtle side walls visually bind the two mapped layers into one pseudo-3D
    # object instead of leaving a disconnected white gap between them.
    ax.add_patch(Polygon(np.array([bottom[2], top[2], top[0], bottom[0]]), closed=True, facecolor="#B8B8B8", edgecolor="#777777", linewidth=0.45, alpha=0.20, zorder=0))
    ax.add_patch(Polygon(np.array([top[1], top[3], bottom[3], bottom[1]]), closed=True, facecolor="#B8B8B8", edgecolor="#777777", linewidth=0.45, alpha=0.20, zorder=0))
    # Dense contourf polygons approximate a continuous color mapping while
    # keeping the warped pseudo-3D surface and NaN handling robust.
    level_count = 257 if not draw_contours else 31
    tos_levels = np.linspace(-tos_vmax, tos_vmax, level_count)
    zos_levels = np.linspace(-zos_vmax, zos_vmax, level_count)
    draw_layer(ax, sub_lon, sub_lat, tos, top, tos_norm, tos_cmap, tos_levels, "mean TOS perturbation (°C)", draw_contours=draw_contours)
    draw_layer(ax, sub_lon, sub_lat, zos, bottom, zos_norm, zos_cmap, zos_levels, "mean ZOS perturbation", draw_contours=draw_contours)
    for tick, text in ((120.0, r"120$^\circ$E"), (180.0, r"180$^\circ$"), (240.0, r"120$^\circ$W")):
        xx, yy = lonlat_to_surface(np.array([tick]), np.array([LAT_MIN]), bottom)
        ax.text(xx[0], yy[0] - 0.032, text, ha="center", va="top", fontsize=11.8, fontweight="bold")
        xt, yt = lonlat_to_surface(np.array([tick]), np.array([LAT_MIN]), top)
        ax.plot([xx[0], xt[0]], [yy[0], yt[0]], color="black", linewidth=0.55, linestyle=":", alpha=0.55, zorder=7)
    ax.text(0.5, 0.985, f"(a) Ten-case mean Pacific delayed-onset CNOP perturbation (n={case_count}, rank-1)", ha="center", va="top", fontsize=16.2, fontweight="bold", clip_on=False)
    cax1 = fig.add_axes((0.900, 0.55, 0.018, 0.30))
    cax2 = fig.add_axes((0.900, 0.15, 0.018, 0.30))
    cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=tos_norm, cmap=tos_cmap), cax=cax1, orientation="vertical")
    cbar1.ax.tick_params(labelsize=9)
    cbar1.set_label(f"TOS perturbation (°C), ±{tos_vmax:.2f}", fontsize=10, labelpad=10)
    cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=zos_norm, cmap=zos_cmap), cax=cax2, orientation="vertical")
    cbar2.ax.tick_params(labelsize=9)
    cbar2.set_label(f"ZOS perturbation, ±{zos_vmax:.3f}", fontsize=10, labelpad=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    provenance = {
        "raw_input": str(input_npz),
        "case_count": case_count,
        "aggregation": "arithmetic mean of rank-1 delta_phys across the ten delayed Pacific cases",
        "region": [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        "smoothing": "one NaN-aware pass of the 3x3 [1,2,1;2,4,2;1,2,1]/16 kernel",
        "interpolation": "12x bicubic display interpolation after land-masked smoothing",
        "palette": palette_name,
        "color_norm": f"separate zero-centred TwoSlopeNorm per variable, TOS ±{tos_vmax:.3f}, ZOS ±{zos_vmax:.3f}",
        "contours": bool(draw_contours),
        "fill_levels": int(level_count),
        "outputs": [output.name, output.with_suffix(".pdf").name, "pacific_delayed_mean_perturbation_rank1_mean.npz"],
    }
    output.with_name(output.stem + "_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output)
    print(output.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palette", choices=("soft", "contrast", "original", "original_light"), default="soft")
    parser.add_argument("--no-contours", action="store_true", help="omit contour-line overlays and use dense smooth fill levels")
    parser.add_argument("--tos-vmax", type=float, default=0.8)
    parser.add_argument("--zos-vmax", type=float, default=0.03)
    args = parser.parse_args()
    plot(args.input, args.output, palette_name=args.palette, draw_contours=not args.no_contours, tos_vmax=args.tos_vmax, zos_vmax=args.zos_vmax)


if __name__ == "__main__":
    main()
