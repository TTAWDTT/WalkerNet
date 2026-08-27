# WalkerNet rollout-skill figures

This folder contains the first requested figure: Niño3.4 anomaly ACC versus lead month.

Source data are the formal GPU007 evaluation outputs:

```text
/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/
```

The plot uses the saved WalkerNet monthly and three-month-mean ACC values for `historical_mixed5_best_skill.pt` on the complete lead-36 test subset (`n=245`). The y-axis is fixed to `[0, 1]`; persistence is intentionally omitted from this presentation. A shape-preserving PCHIP interpolation is used only to render a smooth display curve; the original lead values remain visible as markers and are not filtered or replaced. The plot script records the provenance and the exact plotting transformation:

- [plot_rollout_acc_vs_lead.py](../../../scripts/plotting/plot_rollout_acc_vs_lead.py)
- [walkernet_acc_vs_lead.provenance.json](walkernet_acc_vs_lead.provenance.json)
- [walkernet_acc_vs_lead.alt.txt](walkernet_acc_vs_lead.alt.txt)


The corrected 24-lead presentation is:

- [walkernet_acc_by_start_month_lead1_24.png](walkernet_acc_by_start_month_lead1_24.png)
- [walkernet_acc_by_start_month_lead1_24.pdf](walkernet_acc_by_start_month_lead1_24.pdf)

It uses start month on the y-axis, lead month 1--24 on the x-axis, and ACC as the color field. Values below ACC=0.5 are rendered white for display only. For each start-month row, hatching marks the six endpoint lead cells following the largest one-step ACC drops within leads 1--24. The previous 1--36 presentation is retained as `walkernet_acc_by_start_month_lead1_36.*` for traceability.

The matched WalkerNet/persistence comparison using the same start-month × lead layout is:

- [walkernet_acc_by_start_month_model_persistence_lead1_24.png](walkernet_acc_by_start_month_model_persistence_lead1_24.png)
- [walkernet_acc_by_start_month_model_persistence_lead1_24.pdf](walkernet_acc_by_start_month_model_persistence_lead1_24.pdf)

The left panel is WalkerNet and the right panel is persistence. Both panels use the same ACC display range (0.5--1.0), cubic display interpolation, and per-system hatching of the six fastest one-step ACC declines. The underlying formal 12×36 grouped table is preserved as `eval_rollout_best_skill_lead1_36_by_start_month.csv`.

The earlier 36-lead figure is:

- [walkernet_acc_by_start_month_lead1_36.png](walkernet_acc_by_start_month_lead1_36.png)
- [walkernet_acc_by_start_month_lead1_36.pdf](walkernet_acc_by_start_month_lead1_36.pdf)

It uses start month on the y-axis, lead month 1--36 on the x-axis, and ACC as the color field. Values below ACC=0.5 are rendered white for display only. For each start-month row, hatching marks the six endpoint lead cells following the largest one-step ACC drops `ACC(lead) - ACC(lead+1)`. The previous 12-small-multiple orientation is retained only under `archive_wrong_orientation/` for traceability.

The saved 12×12 start/end-month matrices are also plotted as a two-panel contour comparison:

- [walkernet_start_end_month_acc_model_persistence.png](walkernet_start_end_month_acc_model_persistence.png)
- [walkernet_start_end_month_acc_model_persistence.pdf](walkernet_start_end_month_acc_model_persistence.pdf)

The left panel is WalkerNet and the right panel is persistence. Both use the same ACC color scale (`-0.5` to `1.0`) and the same cubic contour-display interpolation.

For direct comparison without any smoothing, the raw-cell version is:

- [walkernet_start_end_month_acc_model_persistence_grid.png](walkernet_start_end_month_acc_model_persistence_grid.png)
- [walkernet_start_end_month_acc_model_persistence_grid.pdf](walkernet_start_end_month_acc_model_persistence_grid.pdf)

This version renders each saved 12×12 cell directly with `pcolormesh`; no interpolation is applied.
