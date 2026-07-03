# WalkerNet CNOP Research Log

## 2026-07-03

- Initialized autoresearch state for CNOP experiments.
- Literature bootstrap:
  - Mu, Duan, and Wang (2003) define CNOP for nonlinear predictability and apply it to ENSO.
  - Recent ENSO CNOP literature commonly optimizes SST plus thermocline-depth-like variables under an energy constraint.
  - AI-enabled CNOP work supports iterative optimization of perturbations in trained DL forecast systems.
- Working formulation for WalkerNet:
  - Use `tos` and `zos` as the perturbable variables.
  - Perturb only the 12th month of the input window.
  - Use a 12-month autoregressive rollout as the nonlinear forecast operator.
  - Objective: maximize the 12-month target year's Niño3.4 three-month mean anomaly.
  - Constraint: projected gradient optimization under variable-wise normalized RMS bounds.
- Implemented `scripts/compute_tos_zos_cnop.py`.
- Smoke test with full-grid perturbations succeeded numerically but produced high-frequency/checkerboard perturbation structures.
- Revised the main method to optimize perturbations on a patch grid and upsample to the full grid.
- Main patch-grid experiment completed on 10 neutral cases:
  - observed neutral score range: 0.126-0.252;
  - baseline max 3-month Niño3.4 range: -0.606 to 0.430;
  - CNOP max 3-month Niño3.4 range: 0.863 to 1.879;
  - all 10 cases were pushed into positive ENSO-like response.
