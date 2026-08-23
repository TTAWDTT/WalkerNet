# CNOP Basin Follow-up Plan

Status: approved for execution on 2026-08-23.

## Fixed configuration

- Eight starts per case; rank-1 candidate retained.
- `relative_initial_l2` constraint at 3% for TOS and ZOS separately.
- Adam optimizer, 100 steps for the formal run; the existing 40-step results remain the pilot reference.
- `lead_delta` objective at forecast lead 12.
- Same checkpoint, training climatology, patch parameterization, normalization, and random seeds across basin runs.
- Formal domains: Pacific (`120E-290E`), Indian (`20E-120E`), and Global (all valid ocean cells within the configured mask).
- Historical `atlantic_indian` remains diagnostic only and is not used as the Indian result.

## Ordered tasks and GPU schedule

1. Freeze the manifest, configuration, seed, code commit, and output root.
2. Run Indian-only mask tests and one-case CLI smoke test on GPU005 GPU0.
3. Run the 3 representative cases x 3 domains convergence check on GPU005 GPUs 0-2. Compare 40 versus 100 steps before formal production.
4. Freeze the formal ten-case manifest and run 10 cases x 3 domains (30 CNOP jobs) on GPU005 GPUs 0-7, with at most eight concurrent jobs and queued waves.
5. Audit every summary and NPZ: constraint ratio, objective, lead-12 values, three-month values, baseline/truth agreement, and start-to-start stability.
6. Generate overview maps, response-evolution plots, basin comparison plots, and machine-readable audit tables.
7. Pull the completed result bundle to the local asset directory and visually inspect all figures.
8. Only after the primary comparison is accepted, run the optional 1%/2%/3% sensitivity on three representative cases.

GPU006 is excluded from this plan because it hosts unrelated experiments. GPU005 is the only production node, and the scheduler must preserve the eight-GPU concurrency limit.

## Storage

Remote root:

```text
/data/WalkerNet/outputs/cnop_basin_relative3pct_lead12_steps100_gpu005_v1/
```

Expected remote subdirectories:

```text
manifest/  logs/  pacific/  indian/  global/  summary/  figures/
```

Local delivery root:

```text
D:/Github/WalkerNet/docs/assets/cnop_basin_relative3pct_lead12_steps100_v1/
```

Expected local figures:

```text
cnop_overview_10cases_pacific.png
cnop_overview_10cases_indian.png
cnop_overview_10cases_global.png
response_evolution_10cases_pacific.png
response_evolution_10cases_indian.png
response_evolution_10cases_global.png
basin_response_summary.png
constraint_audit.csv
cnop_case_summary.csv
```

## Acceptance criteria

- Every completed case has `constraint_ratio` in `[0.99, 1.01]`.
- The 100-step objective is compared with the 40-step pilot; any change above 5% is reported rather than hidden.
- Baseline, truth, and perturbed fields use one explicitly recorded climatology convention.
- Lead-12 and three-month rolling metrics are reported separately.
- ENSO classification uses `Nino3.4 >= 0.5 C` and is never inferred from color alone.
- Basin comparisons include the actual physical TOS/ZOS L2 norms; raw objectives are not compared as if the budgets were identical.
- No old pilot output is mixed into formal evidence.
