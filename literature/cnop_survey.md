# CNOP Literature Survey for WalkerNet

## Mu, Duan, and Wang (2003)

- Source: https://npg.copernicus.org/articles/10/493/2003/
- DOI: https://doi.org/10.5194/npg-10-493-2003
- Key point: CNOP was proposed for nonlinear predictability in weather and climate models, with an ENSO coupled ocean-atmosphere model as an example.
- Relevance: Provides the core definition: maximize nonlinear forecast response under an initial perturbation constraint.

## Shi and Ma (2024), Sampling Method for ENSO Optimal Precursors

- Source: https://npg.copernicus.org/articles/31/165/2024/
- Key point: ENSO CNOP/optimal precursor studies often consider SST anomalies and thermocline-depth anomalies, nondimensionalized before applying a norm constraint.
- Relevance: Supports using WalkerNet `tos` and `zos` as the two perturbable fields and using a normalized energy-style constraint.

## Zhou et al. (2025), AI-Enabled O-CNOP for Extreme El Niño

- Source: https://www.nature.com/articles/s41612-025-01303-6
- Key point: AI-enabled O-CNOP uses iterative selection and optimization under uniform energy constraints to improve DL ENSO ensemble prediction.
- Relevance: Supports computing CNOP-like perturbations directly in trained AI forecast systems.

## Advances in Atmospheric Sciences (2024), Iterative O-CNOP

- Source: https://link.springer.com/article/10.1007/s00376-024-4069-y
- Key point: Gradient-based iterative ideas can compute O-CNOPs more efficiently than sequential optimization.
- Relevance: Supports projected gradient ascent as a practical first implementation.
