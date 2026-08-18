#!/usr/bin/env bash
set -euo pipefail

# Remap raw CESM2 NetCDF files to the shared WalkerNet 1-degree grid.
#
# Run from WSL/Linux:
#   bash scripts/data/remap_to_1x1.sh
#
# Or from Windows PowerShell:
#   wsl -d Ubuntu-24.04 -- bash /path/to/WalkerNet/scripts/data/remap_to_1x1.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRID_FILE="$ROOT_DIR/configs/grid_1x1_180x360.txt"
INPUT_DIR="${1:-$ROOT_DIR/data_example}"
OUTPUT_DIR="${2:-$ROOT_DIR/data_1x1}"

if ! command -v cdo >/dev/null 2>&1; then
  echo "Error: cdo is not installed or not on PATH." >&2
  echo "Install it in WSL/Linux first, for example: sudo apt-get install cdo netcdf-bin" >&2
  exit 1
fi

if [[ ! -f "$GRID_FILE" ]]; then
  echo "Error: target grid file not found: $GRID_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

remap_one() {
  local variable="$1"
  local input_file="$2"
  local output_file="$OUTPUT_DIR/${variable}_1x1.nc"

  if [[ ! -f "$input_file" ]]; then
    echo "Error: input file not found: $input_file" >&2
    exit 1
  fi

  echo "Remapping $variable"
  echo "  input : $input_file"
  echo "  output: $output_file"

  cdo -O -f nc4 -z zip_4 remapbil,"$GRID_FILE" "$input_file" "$output_file"
}

remap_one "tos"  "$INPUT_DIR/tos_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
remap_one "zos"  "$INPUT_DIR/zos_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
remap_one "tauu" "$INPUT_DIR/tauu_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
remap_one "tauv" "$INPUT_DIR/tauv_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"

echo
echo "Done. Remapped files are in: $OUTPUT_DIR"
echo "Run this to verify:"
echo "  python scripts/data/check_remapped_data.py --data-dir \"$OUTPUT_DIR\""
