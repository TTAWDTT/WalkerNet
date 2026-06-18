"""Check WalkerNet remapped NetCDF files.

This script verifies that data_1x1 contains the four shared variables in the
expected (time, lat, lon) layout.

Run:
    python scripts/check_remapped_data.py
    python scripts/check_remapped_data.py --data-dir data_1x1
    python scripts/check_remapped_data.py --data-dir /mnt/sda/WalkerNet/cmip6_1x1 --multi-source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WalkerDataset


VARIABLES = ("tos", "zos", "tauu", "tauv")
EXPECTED_SHAPE = (1980, 180, 360)
EXPECTED_DIMS = ("time", "lat", "lon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check remapped WalkerNet data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_1x1"),
        help="Directory containing *_1x1.nc files.",
    )
    parser.add_argument(
        "--multi-source",
        action="store_true",
        help="Treat data-dir as a root containing one subdirectory per source.",
    )
    return parser.parse_args()


def check_file(data_dir: Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = data_dir / f"{variable}_1x1.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with xr.open_dataset(path, decode_times=False) as ds:
        if variable not in ds.data_vars:
            raise ValueError(f"{path}: variable '{variable}' not found. Found: {list(ds.data_vars)}")

        arr = ds[variable]
        if arr.dims != EXPECTED_DIMS:
            raise ValueError(f"{path}: expected dims {EXPECTED_DIMS}, got {arr.dims}")

        if arr.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected shape {EXPECTED_SHAPE}, got {arr.shape}")

        lat = ds["lat"].values
        lon = ds["lon"].values
        finite_ratio = float(arr.notnull().mean().values)
        print(f"OK {path}: {variable}{arr.shape}, finite={finite_ratio:.4f}")
    years, months = WalkerDataset._read_year_month(path)
    return lat, lon, years, months


def check_source(data_dir: Path) -> None:
    """检查一个 source 目录里的四个变量是否互相对齐。"""
    reference_lat = None
    reference_lon = None
    reference_years = None
    reference_months = None
    for variable in VARIABLES:
        lat, lon, years, months = check_file(data_dir, variable)
        if reference_lat is None:
            reference_lat = lat
            reference_lon = lon
            reference_years = years
            reference_months = months
            continue
        if not np.allclose(lat, reference_lat):
            raise ValueError(f"{data_dir}: latitude mismatch for {variable}")
        if not np.allclose(lon, reference_lon):
            raise ValueError(f"{data_dir}: longitude mismatch for {variable}")
        if not np.array_equal(years, reference_years) or not np.array_equal(months, reference_months):
            raise ValueError(f"{data_dir}: year/month mismatch for {variable}")
    print(f"All variables align in {data_dir}")


def main() -> None:
    args = parse_args()
    if args.multi_source:
        source_dirs = sorted(path for path in args.data_dir.iterdir() if path.is_dir())
        if not source_dirs:
            raise FileNotFoundError(f"No source directories found under {args.data_dir}")
        for source_dir in source_dirs:
            print(f"== {source_dir.name} ==")
            check_source(source_dir)
    else:
        check_source(args.data_dir)
    print("All remapped files look good.")


if __name__ == "__main__":
    main()
