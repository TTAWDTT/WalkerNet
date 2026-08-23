# Pacific delayed-onset CNOP experiment

## Frozen design

- Cases: the ten records in `cnop_basin_relative3pct_lead12_steps100_v1/formal_manifest_v1.csv`.
- Domain: Pacific mask (valid ocean cells, 120°E–290°E, −60°–60° latitude).
- Paired branches per case: `normal` (`lead_delta`) and `delayed` (`delayed_lead_delta`).
- Starts: 12 paired seeds per branch; both branches use identical start seeds.
- Optimization: Adam, 100 steps, lead-12 objective, `relative_initial_l2=0.03`.
- Delayed objective: lead-12 response minus the early-lead penalty (first three leads, threshold 0.2 °C, weight 2.0).
- Retention: top-3 candidates per branch, with the existing diversity filter; no existing formal outputs are overwritten.

## Execution order and resources

1. Freeze the manifest, method parameters, commit, and launcher metadata under the remote experiment directory.
2. Run the six targeted CNOP tests and a CUDA/checkpoint/data smoke test on GPU007.
3. Launch at most eight concurrent jobs on genuinely idle GPU007 L20X devices; schedule 20 jobs total (10 normal + 10 delayed) and preserve per-job logs/PIDs.
4. Audit all 20 summaries and candidate files before plotting.
5. Replay the retained top-3 perturbations and create one six-panel candidate figure per case: normal rank 1–3 on the first row, delayed rank 1–3 on the second; each candidate contains TOS/ZOS at leads 2, 4, 6, 8, 10, 12 with shared scales and labels.
6. Copy the final figures and audit metadata to `docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v1/` and update the TODO.

## Storage

- Remote results: `/data/WalkerNet/outputs/cnop_pacific_delayed_onset_24starts_steps100_v1/`.
- Remote code: `/data/WalkerNet/repo` at the delayed-objective branch/commit.
- Local figures and audit bundle: `docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v1/`.

## Stop/continue criteria

Do not select by visual appearance alone. Continue only when every job has a complete summary/NPZ/candidate record, the retained constraint ratio is approximately one, delayed candidates reduce early response relative to their paired normal candidates, and lead-12 amplification remains interpretable. Any missing or inconsistent record is a blocker for plotting.
