# WalkerNet rollout-skill figures

This folder contains the first requested figure: Niño3.4 anomaly ACC versus lead month.

Source data are the formal GPU007 evaluation outputs:

```text
/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/
```

The plot uses the saved monthly and three-month-mean ACC values for `historical_mixed5_best_skill.pt` on the complete lead-36 test subset (`n=245`). No smoothing, interpolation, filtering, or axis clipping was applied. The plot script records the provenance and the exact plotting transformation:

- [plot_rollout_acc_vs_lead.py](../../../scripts/plotting/plot_rollout_acc_vs_lead.py)
- [walkernet_acc_vs_lead.provenance.json](walkernet_acc_vs_lead.provenance.json)
- [walkernet_acc_vs_lead.alt.txt](walkernet_acc_vs_lead.alt.txt)

The second start-month × target-month figure is intentionally not generated yet; its hatching rule needs to be fixed before plotting.
