"""Shared geographic masks for the three-basin CNOP experiments."""

from __future__ import annotations

import numpy as np
import torch


BASIN_DOMAINS = ("pacific", "atlantic_indian", "global")
PACIFIC_LON_BOUNDS = (120.0, 290.0)


def numpy_basin_region(
    lat: np.ndarray,
    lon: np.ndarray,
    domain: str,
    lat_bounds: tuple[float, float],
) -> np.ndarray:
    """Return a ``(H, W)`` mask using the experiment's longitude sectors."""

    if domain not in BASIN_DOMAINS:
        raise ValueError(f"Unsupported basin domain: {domain}")
    lat_array = np.asarray(lat, dtype=np.float64)
    lon_360 = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    latitude = (lat_array >= lat_bounds[0]) & (lat_array <= lat_bounds[1])
    pacific = (lon_360 >= PACIFIC_LON_BOUNDS[0]) & (lon_360 <= PACIFIC_LON_BOUNDS[1])
    if domain == "global":
        longitude = np.ones_like(pacific, dtype=bool)
    else:
        longitude = pacific if domain == "pacific" else ~pacific
    return latitude[:, None] & longitude[None, :]


def torch_basin_region(
    lat: torch.Tensor,
    lon: torch.Tensor,
    domain: str,
    lat_bounds: tuple[float, float],
) -> torch.Tensor:
    """Torch equivalent of :func:`numpy_basin_region`."""

    if domain not in BASIN_DOMAINS:
        raise ValueError(f"Unsupported basin domain: {domain}")
    lon_360 = torch.remainder(lon, 360.0)
    latitude = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    pacific = (lon_360 >= PACIFIC_LON_BOUNDS[0]) & (lon_360 <= PACIFIC_LON_BOUNDS[1])
    if domain == "global":
        longitude = torch.ones_like(pacific, dtype=torch.bool)
    else:
        longitude = pacific if domain == "pacific" else ~pacific
    return latitude[:, None] & longitude[None, :]
