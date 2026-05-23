# Architecture Notes

## Main Idea

Cross-variable and cross-time interactions are handled early in the embedding stage. The later backbone focuses mainly on spatial / regional interactions.

## Pipeline

Input:
B × L × 4 × H × W

↓ Joint Time-Variable Patch Embedding

Z:
B × 4 × N × d

↓ Regional Spatial Attention

↓ Target-conditioned TMoE

↓ Variable-specific Decoder

Output:
B × K × 4 × H × W

## Why This Design

- Avoid full attention over L × 4 × N tokens
- Preserve variable-centered representations
- Focus model capacity on global spatial teleconnections
- Use target month / rollout step to model seasonal and lead-time differences