# Global delayed-onset CNOP experiment

## Frozen protocol

- Cases: the existing ten-case `formal_manifest_v1.csv` used by the formal basin run.
- Perturbation domain: all valid ocean cells (`domain=global`).
- Objective: `delayed_lead_delta`, maximizing the lead-12 Niño3.4 increment while penalizing excessive response over the first three leads.
- Optimizer: Adam, 100 steps, learning rate 0.08.
- Starts: 12 independent starts per case; retain top-3 candidates.
- Constraint: `relative_initial_l2=0.03`.
- Delay penalty: early leads 1–3, threshold 0.2 °C, weight 2.0.
- Seed: 42; model rollout uses the frozen historical mixed5 checkpoint.

The existing formal Global normal results remain unchanged and are treated as the paired non-delayed reference. Indian results are not rerun with the delay objective in this phase.

## Remote outputs

`/data/WalkerNet/outputs/cnop_global_delayed_onset_24starts_steps100_v1/`

## Local figure destination

`docs/assets/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/global_delayed/`

Only complete, audited outputs will be copied into the local destination.
