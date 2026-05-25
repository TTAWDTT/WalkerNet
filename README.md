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

B × L × 4 × H × W（L = 3 或 12 个月，可配置）

## Output

Future global physical fields:

B × 1 × 4 × H × W (1 month output, autoregressive rollout for longer lead times)

Niño3.4 index is derived from predicted SST for ENSO evaluation.

## Current Architecture Idea

1. Joint time-variable patch embedding
2. Spatial attention (global spatial teleconnections)
3. TMoE (target-month conditioned routing)
4. Coupled variable decoder (4 variables jointly decoded)
5. Rollout step embedding (independent conditioning signal for multi-step prediction)

## Current Status

- [ ] Build model skeleton
- [ ] Run synthetic tensor smoke test
- [ ] Implement Niño3.4 metrics
- [ ] Add real dataset loader
- [ ] Run baseline experiments
