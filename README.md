<p align="center">
  <img src="./logo.png" alt="WalkerNet logo" width="240">
</p>

# WalkerNet

A lightweight research codebase for global physical field forecasting and ENSO skill evaluation.

## Goal

Use historical global physical fields to predict future global fields, and evaluate the prediction skill through Niño3.4 / ENSO metrics.

## Input Variables

- Sea surface temperature, SST
- Ocean heat content, HC
- Zonal wind stress, taux
- Meridional wind stress, tauy

Input shape:

B × L × 4 × H × W

## Output

Future global physical fields:

B × K × 4 × H × W

Niño3.4 index is derived from predicted SST for ENSO evaluation.

## Current Architecture Idea

1. Joint time-variable patch embedding
2. Regional / spatial attention
3. Target-month and rollout-step conditioned TMoE
4. Variable-specific decoders

## Current Status

- [ ] Build model skeleton
- [ ] Run synthetic tensor smoke test
- [ ] Implement Niño3.4 metrics
- [ ] Add real dataset loader
- [ ] Run baseline experiments
