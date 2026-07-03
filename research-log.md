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
