# Reproducibility Checklist

Before sharing an experiment, record:

1. Git commit or release tag.
2. Full YAML configuration used for the run.
3. Dataset source, scenario, years, remapping command, and file checksums.
4. Python, PyTorch, CUDA, CDO, and operating-system versions.
5. Random seed, device count, batch size, gradient accumulation, and AMP mode.
6. Checkpoint identifier and SHA-256 checksum.
7. Exact evaluation command, split, lead range, and climatology definition.
8. CSV/JSON metrics and the script that generated every published figure.

## Minimal public verification

From a clean checkout with a small remapped source:

```bash
python -m pytest -q
python -m src.train --config configs/examples/smoke.yaml --device cpu
```

This checks imports, data loading, normalization, model construction, one
training pass, and checkpoint writing. It is not a scientifically meaningful
forecast by itself.

## Full forecast verification

Use the same dataset split and normalization statistics for the model and
evaluator. Report masked field RMSE/MAE/correlation for all four variables and
Niño3.4 anomaly ACC/RMSE at every requested lead, together with persistence
and the climatology definition.

For CNOP, additionally report the constraint definition and radius, optimized
variables, perturbed month, objective, number of starts, optimizer, and seed.
