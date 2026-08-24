# CNOP response-evolution figure organization

- `pacific/`: Pacific normal-objective response-evolution figures.
- `pacific_delayed/`: Pacific delayed-objective candidate panels and response-evolution figures.
- `overview/`: delayed rank-1 and lead-12 overview products.
- `global/`: existing formal Global normal-objective overview and response-evolution figures.
- `global_delayed/`: audited Global delayed-objective overview, top-3 panels, and rank-1 response-evolution figures. The formal response-evolution subfolder uses fixed `TOS ±0.8 °C` and `ZOS ±0.03` display scales; the adjacent `response_evolution_old_layout_autoscale/` folder is archival only.
- `Indian/`: existing formal Indian overview and response-evolution figures.

The Global delayed run uses the same ten neutral cases, 3% relative initial-L2 constraint, 12 starts, 100 Adam steps, and delayed-lead objective as the Pacific delayed experiment. Its audited figures are included in this local bundle.

Display-scale note: the Pacific legacy response-evolution renderer still used a per-case percentile floor (for example, `TOS ±0.806 °C`), while the Global delayed formal renderer is fixed at `±0.8 °C`. This is a plotting-only discrepancy and is tracked before the final cross-basin comparison.
