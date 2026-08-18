# Scripts

The scripts are grouped by responsibility:

- `data/`: CDO remapping and remapped-file validation.
- `cnop/`: CNOP constraint calculation, TOS/ZOS optimization, clustering,
  and research figures.
- `diagnostics/`: forecast-field and rollout diagnostics.
- `train/`: generic launchers and short smoke checks.

Scripts that use data, checkpoints, or output directories take paths as
arguments. Machine-specific launchers and historical one-off runs belong in a
local ignored directory, not in the public repository.
