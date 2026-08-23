# CNOP Basin Follow-up TODO

- [x] Commit Indian-only mask implementation and tests.
- [x] Run Indian mask and CLI smoke test on GPU005.
- [x] Run representative 3 x 3 convergence check (40 vs 100 steps); max relative metric change 2.01%, freeze 100 steps.
- [x] Freeze formal ten-case manifest and configuration (`formal_manifest_v1`).
- [ ] Run Pacific, Indian, and Global CNOP production (30 jobs, GPU005 only); queued until all GPU005 cards are genuinely idle.
- [ ] Audit summary/NPZ values and baseline/truth consistency.
- [ ] Generate overview, response-evolution, and basin comparison figures.
- [ ] Pull and visually inspect the local figure bundle.
- [ ] Decide whether the optional 1%/2%/3% sensitivity is needed.
