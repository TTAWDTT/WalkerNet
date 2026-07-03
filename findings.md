# WalkerNet CNOP Findings

## Current Understanding

CNOP asks for the bounded initial perturbation that produces the largest nonlinear forecast response. For this project, the trained WalkerNet rollout model can act as the nonlinear forecast operator, and PyTorch autograd replaces an adjoint model.

The closest WalkerNet analogue to classical ENSO CNOP variables is:

- `tos`: sea surface temperature.
- `zos`: sea-surface-height / upper-ocean state proxy.

The first experiment should avoid changing all 12 input months. We perturb only the final input month because the user requested the perturbation act on the 12th month of the previous-year window.

## Results as of 2026-07-03

The first complete patch-grid TOS/ZOS CNOP experiment succeeded on 10 neutral Jan-Dec cases. The selected cases have observed neutral scores between 0.126 and 0.252. With a conservative normalized RMS radius of 0.1 for both `tos` and `zos`, all 10 cases were pushed to positive El Niño-like forecast responses.

The CNOP max 3-month Niño3.4 anomaly ranged from 0.863 to 1.879. The largest gain was 1.721 for GFDL-ESM4 target year 1930. The strongest final response was 1.879 for MPI-ESM1-2-HR target year 2003.

An exploratory full-grid perturbation produced high-frequency/checkerboard patterns, so the main method now optimizes a low-resolution patch grid and upsamples it. This keeps the perturbation more interpretable while preserving the induced ENSO response.

Composite diagnostics show that the dominant precursor is a robust TOS pattern, not an isolated single-case texture. Across the 10 neutral cases, the optimized perturbation consistently warms the Nino3.4 / central-eastern equatorial Pacific region while the western equatorial Pacific is weaker or negative. The mean regional indices are:

- Nino3.4 TOS perturbation: +1.2826.
- Western equatorial Pacific TOS perturbation: -0.1701.
- Central/eastern equatorial Pacific TOS perturbation: +1.0858.
- TOS east-west contrast: +1.2560.
- ZOS east-west tilt proxy: +0.0727.

This suggests WalkerNet's CNOP direction mainly acts by preconditioning the equatorial Pacific SST gradient toward an El Nino-like response. ZOS contributes a weaker upper-ocean-state proxy signal, but it should not be interpreted as true heat content or thermocline depth.

## Initial Method

For each neutral target year `Y`:

1. Use previous January-December as model input.
2. Roll out WalkerNet for January-December of year `Y`.
3. Keep the original rollout as the neutral baseline.
4. Optimize a bounded perturbation on previous December `tos/zos`.
5. Maximize a differentiable soft maximum of the target-year Niño3.4 three-month mean anomaly.

## Open Questions

- What perturbation norm is physically reasonable for `zos` in this model?
- Should perturbations be restricted to the tropical Pacific or allowed globally?
- How sensitive are the precursor patterns to the perturbation radius (`0.05`, `0.1`, `0.2`)?
- Does the same TOS-dominant precursor appear for negative CNOP / La Nina targets?
