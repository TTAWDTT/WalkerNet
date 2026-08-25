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

The second start-month × target-month figure is intentionally not generated yet; its hatching rule needs to be fixed before plotting.
