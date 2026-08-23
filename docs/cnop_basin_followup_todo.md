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
- [ ] Launch Pacific delayed-onset paired experiment (10 cases × normal/delayed; 12 starts, 100 Adam steps, 3% relative initial L2; top-3 per branch).
- [ ] Audit paired outputs: constraint ratios, early-lead response, lead-12 gain, and complete top-3 records.
- [ ] Generate one 2×3 candidate-panel figure per case (normal rank 1–3 / delayed rank 1–3; leads 2,4,6,8,10,12; shared scales).
- [ ] Compare delayed vs normal candidates and decide whether to extend to other basins.
