"""Check WalkerNet remapped NetCDF files.

This script verifies that data_1x1 contains the four shared variables in the
expected (time, lat, lon) layout.

Run:
    python scripts/check_remapped_data.py
    python scripts/check_remapped_data.py --data-dir data_1x1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


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
    return parser.parse_args()


def check_file(data_dir: Path, variable: str) -> None:
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

        print(f"OK {path}: {variable}{arr.shape}, dims={arr.dims}")


def main() -> None:
    args = parse_args()
    for variable in VARIABLES:
        check_file(args.data_dir, variable)
    print("All remapped files look good.")


if __name__ == "__main__":
    main()
