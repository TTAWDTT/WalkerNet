# CNOP Basin Follow-up TODO

- [x] Commit Indian-only mask implementation and tests.
- [x] Run Indian mask and CLI smoke test on GPU005.
- [x] Run representative 3 x 3 convergence check (40 vs 100 steps); max relative metric change 2.01%, freeze 100 steps.
- [x] Freeze formal ten-case manifest and configuration (`formal_manifest_v1`).
- [x] Run Pacific, Indian, and Global CNOP production (30 jobs, GPU005 only).
- [x] Audit summary/NPZ values and baseline/truth consistency; 30/30 rows, no missing outputs, no failures, constraint ratio 0.9999999-1.0000001.
- [x] Generate overview, response-evolution, and basin comparison figures on GPU005.
- [x] Pull and visually inspect the local figure bundle; SHA-256 hashes match the GPU005 artifact manifest.
- [ ] Decide whether the optional 1%/2%/3% sensitivity is needed.
- [x] Launch Pacific delayed-onset paired experiment (10 cases × normal/delayed; 12 starts, 100 Adam steps, 3% relative initial L2; top-3 per branch).
- [x] Audit paired outputs: 20/20 summaries, no failures, constraint ratios within 5e-5, delayed rank-1 early response lower in all 10 cases, and complete top-3 records.
- [x] Generate one 2×3 candidate-panel figure per case (normal rank 1–3 / delayed rank 1–3; leads 2,4,6,8,10,12; shared scales) and copy to `docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/figures/`.
- [x] Re-render all 60 retained candidates with the established wide response-evolution layer (TOS + ZOS, smoothing, shared fixed ranges) under `docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/legacy_response_evolution/`.
- [x] Create a delayed rank-1 lead-12 overview with the three requested fields (truth, baseline, perturbed) under `docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/`.
- [ ] Compare delayed vs normal candidates and decide whether to extend to other basins.
