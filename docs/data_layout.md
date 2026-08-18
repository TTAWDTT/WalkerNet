# Data Layout

## Remapped source

WalkerNet reads one directory per source. Each directory contains exactly:

```text
tos_1x1.nc
zos_1x1.nc
tauu_1x1.nc
tauv_1x1.nc
```

Every file must have dimensions `(time, lat, lon)). Latitude and longitude
coordinates must be identical across variables after remapping:

```text
lat = -89.5, -88.5, ..., 89.5
lon =   0.5,   1.5, ..., 359.5
```

Missing locations are carried by `valid_mask`. Invalid values are replaced
by zero after normalization and excluded from masked losses and metrics.

## Multi-source configuration

Use `data.sources` to mix sources. A sample never combines sources
internally; the source is selected at sample level:

```yaml
data:
  sources:
    - {name: CESM2, path: data/historical/CESM2}
    - {name: GFDL-ESM4, path: data/historical/GFDL-ESM4}
```

Keep cache and normalization-stat paths separate for different datasets. The
loader derives source-specific cache names when a multi-source config provides
`data_cache_path`.

## CDO remapping

The target grid is stored in `configs/grid_1x1_180x360.txt`. Historical and
SSP remappers call CDO directly and fail on CDO errors; there is no scipy
fallback. Run the validator after remapping:

```bash
python scripts/data/check_remapped_data.py --data-dir /path/to/source
```

For SSP data, use the layout
`raw/<scenario>/<source>/<variable-resolution>/*.nc` and run
`scripts/data/remap_cmip6_ssp_to_1x1.py` with explicit paths.

## Data licensing

The repository does not redistribute CMIP6 NetCDF files. Obtain the data from
the official provider and comply with its terms. Record the exact source,
version, file list, and checksum in each experiment record.
