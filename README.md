# WalkerNet

WalkerNet is a PyTorch model for autoregressive global air-sea field forecasting.
It predicts the next monthly field from the previous 12 months and evaluates
ENSO skill from the predicted `tos` field rather than predicting an index
directly.

The repository contains the model, data interface, training/evaluation tools,
CDO remapping utilities, and the TOS/ZOS CNOP research scripts. Raw NetCDF
data, caches, checkpoints, server logs, and private experiment paths are not
part of the public repository.

## Interface

```text
x       (B, 12, 4, 180, 360)
y_pred  (B,  1, 4, 180, 360)
```

The fixed channel order is `tos`, `zos`, `tauu`, `tauv`. The target is a regular
1-degree grid with 180 latitude points and 360 longitude points. Latitude
centers are `-89.5 ... 89.5`; longitude centers are `0.5 ... 359.5`.

## Installation

```bash
conda create -n walkernet python=3.11
conda activate walkernet
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install the CUDA-compatible PyTorch build for your machine before GPU runs.

## Data

Each source directory must contain:

```text
<source>/
  tos_1x1.nc
  zos_1x1.nc
  tauu_1x1.nc
  tauv_1x1.nc
```

Each file must expose `(time, lat, lon)` with shape `(T, 180, 360)` and share
the same coordinates and monthly timestamps. The repository does not include
the full research data. See [data layout](docs/data_layout.md).

Remap raw CMIP6 files with CDO and explicit paths:

```bash
python scripts/data/remap_cmip6_to_1x1.py \
  --input-root /path/to/CMIP6-historical-data \
  --output-root /path/to/cmip6_1x1 \
  --cdo-bin cdo

python scripts/data/check_remapped_data.py \
  --data-dir /path/to/cmip6_1x1 \
  --multi-source
```

## Configuration and training

Public relative-path templates are in `configs/`:

```text
configs/default.yaml
configs/examples/smoke.yaml
configs/examples/mixed5.yaml
```

Run the short verification first:

```bash
python -m pytest -q
python -m src.train --config configs/examples/smoke.yaml --device cpu
```

Single-device training:

```bash
python -m src.train \
  --config configs/examples/mixed5.yaml \
  --device cuda \
  --num-workers 2
```

Single-node DDP:

```bash
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONFIG_PATH=configs/examples/mixed5.yaml \
bash scripts/train/train_ddp.sh --num-workers 2
```

Evaluate a checkpoint:

```bash
python -m src.evaluate \
  --config configs/examples/mixed5.yaml \
  --checkpoint /path/to/checkpoints/best.pt \
  --split test \
  --device cuda \
  --output-dir outputs/eval
```

The evaluator reports masked field metrics and Niño3.4 anomaly metrics. For
autoregressive field evaluation use `src.evaluate_rollout` with an explicit
`--max-lead`.

## CNOP research tools

The CNOP workflow is separate from the core training path:

```text
scripts/cnop/compute_cnop_constraint.py
scripts/cnop/compute_tos_zos_cnop.py
scripts/cnop/plot_cnop_diagnostics.py
scripts/cnop/cluster_cnop_cases.py
```

It optimizes TOS/ZOS perturbations applied to the twelfth input month, rolls
the trained model forward, and maximizes a Niño3.4 response. This is a
model-based sensitivity experiment, not a claim that the result is a
physically realizable precursor. Read [the CNOP method](docs/cnop_method.md)
before running it and specify all data, checkpoint, constraint, seed, and
output paths explicitly.

## Repository layout

```text
src/                  model, dataset, training, evaluation, metrics
scripts/data/         CDO remapping and validation
scripts/cnop/         constraint, optimization, clustering, CNOP plots
scripts/diagnostics/  field and rollout diagnostics
scripts/train/        generic launchers and smoke checks
configs/              public relative-path templates
docs/                 architecture, data, method, and reproducibility notes
tests/                unit and smoke tests
docs/assets/          selected lightweight research figures and tables
```

## Reproducibility

Record the Git tag, full YAML config, data source and preprocessing command,
software/CUDA/CDO versions, random seed, checkpoint checksum, exact evaluation
command, climatology definition, and output tables. See
[docs/reproducibility.md](docs/reproducibility.md). Large data and model files
belong in a data/model repository, not in Git.

## License and citation

See `LICENSE` and `CITATION.cff`. CMIP6 data remain subject to their original
provider terms and are not relicensed by this code repository.
