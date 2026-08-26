"""Pseudo-3D rendering of the legacy Pacific delayed response-evolution figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter, zoom

LON_MIN, LON_MAX = 150.0, 300.0
LAT_MIN, LAT_MAX = -35.0, 35.0
NINO_BOX = (190.0, 240.0, -5.0, 5.0)
TOS_VMAX, ZOS_VMAX = 0.8, 0.03
# Convert wind-stress anomalies to an equivalent 10-m wind-speed vector for
# the requested m/s legend using tau = rho_air * C_D * |U| U.  The conversion
# is documented and applied consistently to every response panel.
AIR_DENSITY = 1.225  # kg m-3
DRAG_COEFF = 1.3e-3


def soften_cmap(name: str, amount: float = 0.16) -> ListedColormap:
    """Blend a reference colormap toward white for the paper-style palette."""
    base = mpl.colormaps[name](np.linspace(0.0, 1.0, 256))
    base[:, :3] = base[:, :3] * (1.0 - amount) + amount
    return ListedColormap(base, name=f"{name}_soft")


def smooth_grid(grid: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """NaN-aware Gaussian smoothing for display-only response fields."""
    values = np.asarray(grid, dtype=float)
    valid = np.isfinite(values)
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-8)


def upsample(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    valid = np.isfinite(grid)
    values = zoom(np.where(valid, grid, 0.0), (factor, factor), order=3, mode="nearest", prefilter=True)
    weights = zoom(valid.astype(float), (factor, factor), order=3, mode="nearest", prefilter=True)
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 0.35)


def stress_to_equivalent_wind(tau_u: np.ndarray, tau_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert surface-stress components (N m-2) to equivalent wind (m s-1)."""
    magnitude = np.hypot(tau_u, tau_v)
    denominator = np.sqrt(AIR_DENSITY * DRAG_COEFF * magnitude)
    u = np.divide(tau_u, denominator, out=np.zeros_like(tau_u), where=denominator > 1e-8)
    v = np.divide(tau_v, denominator, out=np.zeros_like(tau_v), where=denominator > 1e-8)
    invalid = ~np.isfinite(tau_u) | ~np.isfinite(tau_v)
    u[invalid] = np.nan; v[invalid] = np.nan
    return u, v


def map_xy(lon: np.ndarray, lat: np.ndarray, corners):
    sw, se, nw, ne = corners
    u = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    v = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    x = (1-u)*(1-v)*sw[0] + u*(1-v)*se[0] + (1-u)*v*nw[0] + u*v*ne[0]
    y = (1-u)*(1-v)*sw[1] + u*(1-v)*se[1] + (1-u)*v*nw[1] + u*v*ne[1]
    return x, y


def edges(values):
    out = np.empty(len(values) + 1)
    out[1:-1] = 0.5 * (values[:-1] + values[1:])
    out[0] = values[0] - 0.5 * (values[1] - values[0])
    out[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return out


def land_polygons():
    try:
        from cartopy.io import shapereader
        shp = shapereader.natural_earth(resolution="50m", category="physical", name="land")
        return list(shapereader.Reader(shp).geometries())
    except Exception:
        return []


LAND = None


def draw_land(ax, corners):
    global LAND
    if LAND is None:
        LAND = land_polygons()
    clip = PathPatch(MplPath(np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])), transform=ax.transData)
    for geom in LAND:
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            lon = np.where(coords[:, 0] < 0, coords[:, 0] + 360, coords[:, 0]); lat = coords[:, 1]
            keep = (lon >= LON_MIN-8) & (lon <= LON_MAX+8) & (lat >= LAT_MIN-8) & (lat <= LAT_MAX+8)
            if np.count_nonzero(keep) < 3: continue
            x, y = map_xy(lon[keep], lat[keep], corners)
            patch = Polygon(np.c_[x, y], closed=True, facecolor="#F4F1E8", edgecolor="#2F3A3F", linewidth=0.28, zorder=4)
            patch.set_clip_path(clip); ax.add_patch(patch)


def raster_land_mask(lon, lat):
    global LAND
    if LAND is None: LAND = land_polygons()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pts = np.c_[lon_grid.ravel(), lat_grid.ravel()]
    mask = np.zeros(len(pts), dtype=bool)
    for geom in LAND:
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            lo = np.where(coords[:, 0] < 0, coords[:, 0] + 360, coords[:, 0]); la = coords[:, 1]
            keep = (lo >= LON_MIN-8) & (lo <= LON_MAX+8) & (la >= LAT_MIN-8) & (la <= LAT_MAX+8)
            if np.count_nonzero(keep) >= 3:
                mask |= MplPath(np.c_[lo[keep], la[keep]]).contains_points(pts)
    return mask.reshape(len(lat), len(lon))


def draw_layer(ax, lon, lat, field, corners, norm, cmap, levels, label, show_y, wind_u=None, wind_v=None):
    quiver_handle = None
    xx, yy = np.meshgrid(edges(lon), edges(lat))
    xg, yg = map_xy(xx, yy, corners)
    fv = np.pad(field, ((0, 1), (0, 1)), mode="edge")
    ax.pcolormesh(xg, yg, fv, cmap=cmap, norm=norm, shading="auto", edgecolors="none", antialiased=False, rasterized=True, zorder=1)
    xc, yc = map_xy(*np.meshgrid(lon, lat), corners)
    draw_land(ax, corners)
    if wind_u is not None and wind_v is not None:
        # Sparse quiver overlay for the two wind-stress response components.
        # It is drawn only on the TOS layer, where vector direction remains
        # legible without obscuring the ZOS surface below.
        stride_y = max(1, len(lat) // 14); stride_x = max(1, len(lon) // 28)
        qlon, qlat = np.meshgrid(lon[::stride_x], lat[::stride_y])
        qu, qv = wind_u[::stride_y, ::stride_x], wind_v[::stride_y, ::stride_x]
        qx, qy = map_xy(qlon, qlat, corners)
        valid_q = np.isfinite(qu) & np.isfinite(qv)
        if np.any(valid_q):
            quiver_handle = ax.quiver(qx[valid_q], qy[valid_q], qu[valid_q], qv[valid_q],
                                      color="#34495e", alpha=0.72, width=0.0018,
                                      headwidth=3.2, headlength=4.2, headaxislength=3.5,
                                      angles="xy", scale_units="xy", scale=65.0,
                                      pivot="mid", zorder=7, rasterized=True)
    outline = np.array([corners[0], corners[1], corners[3], corners[2], corners[0]])
    ax.plot(outline[:, 0], outline[:, 1], color="black", lw=0.8, zorder=6)
    for lo in (180, 210, 240, 270, 300):
        x, y = map_xy(np.array([lo, lo]), np.array([LAT_MIN, LAT_MAX]), corners)
        ax.plot(x, y, color="black", lw=0.3, alpha=0.18, ls="--", zorder=5)
    for la in (-20, 0, 20):
        x, y = map_xy(np.array([LON_MIN, LON_MAX]), np.array([la, la]), corners)
        ax.plot(x, y, color="black", lw=0.55 if la == 0 else 0.35, alpha=0.48, zorder=5)
    x0, x1, y0, y1 = NINO_BOX
    xb, yb = map_xy(np.array([x0,x1,x1,x0,x0]), np.array([y0,y0,y1,y1,y0]), corners)
    ax.plot(xb, yb, color="#007C78", lw=0.8, zorder=7)
    if show_y:
        for la, text in ((30, "30N"), (10, "10N"), (-10, "10S"), (-30, "30S")):
            x, y = map_xy(np.array([LON_MIN]), np.array([la]), corners)
            ax.text(x[0]-0.018, y[0], text, ha="right", va="center", fontsize=6.8, fontweight="bold")
    ax.text(corners[0][0]+0.015, corners[0][1]+0.012, label, fontsize=8.0, fontweight="bold", va="bottom", zorder=8)
    return quiver_handle


def draw_pseudo_panel(ax, lon, lat, tos, zos, title, tos_norm, zos_norm, tos_cmap, zos_cmap, tos_levels, zos_levels, main=False, show_y=False, wind_u=None, wind_v=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if main:
        top = ((0.08,0.55),(0.96,0.55),(0.14,0.80),(0.90,0.80)); bottom=((0.08,0.19),(0.96,0.19),(0.14,0.44),(0.90,0.44))
    else:
        top = ((0.10,0.55),(0.94,0.55),(0.16,0.79),(0.88,0.79)); bottom=((0.10,0.18),(0.94,0.18),(0.16,0.43),(0.88,0.43))
    # shallow side walls bind the two layers
    ax.add_patch(Polygon(np.array([bottom[2],top[2],top[0],bottom[0]]), closed=True, facecolor="#B8B8B8", edgecolor="#777777", lw=0.3, alpha=0.16, zorder=0))
    ax.add_patch(Polygon(np.array([top[1],top[3],bottom[3],bottom[1]]), closed=True, facecolor="#B8B8B8", edgecolor="#777777", lw=0.3, alpha=0.16, zorder=0))
    quiver_handle = draw_layer(ax, lon, lat, tos, top, tos_norm, tos_cmap, tos_levels, "TOS", show_y, wind_u=wind_u, wind_v=wind_v)
    draw_layer(ax, lon, lat, zos, bottom, zos_norm, zos_cmap, zos_levels, "ZOS", show_y)
    for lo, text in ((150,"150E"),(180,"180"),(210,"150W"),(240,"120W"),(270,"90W"),(300,"60W")):
        x, y = map_xy(np.array([lo]), np.array([LAT_MIN]), bottom)
        ax.text(x[0], y[0]-0.026, text, ha="center", va="top", fontsize=6.6 if not main else 8.0, fontweight="bold")
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=8.7 if not main else 11.0, fontweight="bold", clip_on=False)
    return quiver_handle


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--wind-input",type=Path,default=None,help="optional same-case full response NPZ with TAUU/TAUV in channels 2/3"); p.add_argument("--lead-delta",default="+0.76"); p.add_argument("--case",default="GFDL-ESM4_1930"); args=p.parse_args()
    mpl.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white","pdf.fonttype":42})
    z=np.load(args.input); response=np.asarray(z["response"],float); perturbation=np.asarray(z["perturbation"],float); lon=np.asarray(z["lon"],float); lat=np.asarray(z["lat"],float)
    wind_response=None
    if args.wind_input is not None:
        wind_z=np.load(args.wind_input); wind_response=np.asarray(wind_z["response"],float)
        if wind_response.ndim != 4 or wind_response.shape[1] < 4:
            raise ValueError("--wind-input must contain response with channels [TOS, ZOS, TAUU, TAUV]")
    slon=(lon>=LON_MIN)&(lon<=LON_MAX); slat=(lat>=LAT_MIN)&(lat<=LAT_MAX); lon=lon[slon]; lat=lat[slat]
    mask=raster_land_mask(lon,lat); fields=[]
    for lead in (1,3,5,7,9,11):
        tos=upsample(smooth_grid(np.where(mask,np.nan,response[lead,0][np.ix_(slat,slon)])),4); zos=upsample(smooth_grid(np.where(mask,np.nan,response[lead,1][np.ix_(slat,slon)])),4)
        wind_u=wind_v=None
        if wind_response is not None:
            wind_tau_u=upsample(smooth_grid(np.where(mask,np.nan,wind_response[lead,2][np.ix_(slat,slon)])),4); wind_tau_v=upsample(smooth_grid(np.where(mask,np.nan,wind_response[lead,3][np.ix_(slat,slon)])),4)
            wind_u, wind_v = stress_to_equivalent_wind(wind_tau_u, wind_tau_v)
        lon_hi=np.linspace(lon[0],lon[-1],tos.shape[1]); lat_hi=np.linspace(lat[0],lat[-1],tos.shape[0]); land_hi=zoom(mask.astype(float),(4,4),order=1,mode="nearest")>=0.5; tos[land_hi]=np.nan; zos[land_hi]=np.nan
        if wind_u is not None: wind_u[land_hi]=np.nan; wind_v[land_hi]=np.nan
        fields.append((lon_hi,lat_hi,tos,zos,wind_u,wind_v))
    # The left panel is the corresponding initial CNOP perturbation.  Apply
    # exactly the same mask, upsampling, Gaussian smoothing, and color scales
    # used by the lead-response panels so the comparison remains honest.
    init_raw_tos=np.where(mask,np.nan,perturbation[0][np.ix_(slat,slon)]); init_raw_zos=np.where(mask,np.nan,perturbation[1][np.ix_(slat,slon)])
    init_tos=upsample(smooth_grid(init_raw_tos),4); init_zos=upsample(smooth_grid(init_raw_zos),4)
    # The constrained field is exactly zero outside its source region.  Treat
    # interpolation values below display precision as zero; this removes the
    # faint edge ringing without introducing artificial white holes.
    init_tos[np.abs(init_tos) < 1e-4] = 0.0; init_zos[np.abs(init_zos) < 1e-5] = 0.0
    lon_hi=np.linspace(lon[0],lon[-1],init_tos.shape[1]); lat_hi=np.linspace(lat[0],lat[-1],init_tos.shape[0]); land_hi=zoom(mask.astype(float),(4,4),order=1,mode="nearest")>=0.5; init_tos[land_hi]=np.nan; init_zos[land_hi]=np.nan
    # Dense levels are retained only for colorbar ticks; the plotted field uses
    # a continuous TwoSlopeNorm so adjacent cells blend rather than forming
    # discrete contour-like bands.
    tos_levels=np.linspace(-TOS_VMAX,TOS_VMAX,81); zos_levels=np.linspace(-ZOS_VMAX,ZOS_VMAX,61)
    # Exact palettes from the legacy renderer: do not substitute custom
    # approximations when comparing this 3D version with figure4.png.
    tos_cmap=soften_cmap("RdYlBu_r",0.03).with_extremes(bad="#F4F1E8")
    zos_cmap=soften_cmap("BrBG",0.02).with_extremes(bad="#F4F1E8")
    tos_norm=TwoSlopeNorm(vmin=-TOS_VMAX,vcenter=0.0,vmax=TOS_VMAX); zos_norm=TwoSlopeNorm(vmin=-ZOS_VMAX,vcenter=0.0,vmax=ZOS_VMAX)
    fig=plt.figure(figsize=(16.5,9.2)); ax=fig.add_axes((0.015,0.17,0.32,0.75)); draw_pseudo_panel(ax,lon_hi,lat_hi,init_tos,init_zos,"(a) Initial CNOP perturbation (rank 1)",tos_norm,zos_norm,tos_cmap,zos_cmap,tos_levels,zos_levels,main=True,show_y=True)
    leads=(2,4,6,8,10,12); positions=[(0.36+0.205*(i%3),0.51-0.38*(i//3)) for i in range(6)]
    quiver_ref=None; quiver_ax=None
    for i,(lead,pos) in enumerate(zip(leads,positions)):
        lon_i,lat_i,tos_i,zos_i,wind_u_i,wind_v_i=fields[i]; a=fig.add_axes((pos[0],pos[1],0.19,0.34)); q=draw_pseudo_panel(a,lon_i,lat_i,tos_i,zos_i,f"({chr(98+i)}) Lead {lead}",tos_norm,zos_norm,tos_cmap,zos_cmap,tos_levels,zos_levels,main=False,show_y=(i%3==0),wind_u=wind_u_i,wind_v=wind_v_i)
        if quiver_ref is None and q is not None:
            quiver_ref=q; quiver_ax=a
    if quiver_ref is not None and quiver_ax is not None:
        quiver_ax.quiverkey(quiver_ref, X=0.64, Y=1.18, U=3.0, label="3 m s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 7})
    cax1=fig.add_axes((0.10,0.07,0.28,0.018)); cax2=fig.add_axes((0.55,0.07,0.28,0.018)); cb1=fig.colorbar(mpl.cm.ScalarMappable(norm=tos_norm,cmap=tos_cmap),cax=cax1,orientation="horizontal",extend="both"); cb2=fig.colorbar(mpl.cm.ScalarMappable(norm=zos_norm,cmap=zos_cmap),cax=cax2,orientation="horizontal",extend="both"); cb1.set_label("TOS response (°C)"); cb2.set_label("ZOS response"); cb1.set_ticks(np.linspace(-TOS_VMAX,TOS_VMAX,7)); cb2.set_ticks(np.linspace(-ZOS_VMAX,ZOS_VMAX,7))
    source, year = args.case.rsplit("_", 1)
    fig.suptitle(f"CNOP response evolution: {source} {year}, delayed candidate rank 1 (lead12 ΔNiño={args.lead_delta})",fontsize=16,fontweight="bold",y=0.985)
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=300,facecolor="white"); fig.savefig(args.output.with_suffix(".pdf"),facecolor="white"); plt.close(fig); print(args.output)


if __name__ == "__main__": main()
