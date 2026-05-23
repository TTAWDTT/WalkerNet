# Architecture Notes

## Main Idea

Cross-variable and cross-time interactions are handled early in the embedding stage. The later backbone focuses mainly on spatial / regional interactions.

## Pipeline

Input:
B × L × 4 × H × W
(SST, HC, taux, tauy)

↓ Joint Time-Variable Patch Embedding

Z:
B × 4 × N × d

↓ Regional Spatial Attention
(借鉴 ENSO-PhyNet，在 attention 中嵌入热收支方程结构)

↓ Target-conditioned TMoE
(同时用目标月份和 rollout step 调制)

↓ Coupled Variable Decoder
(4 变量联合解码，利用 SST-HC-taux-tauy 的物理耦合关系)

Output:
B × K × 4 × H × W

## Why This Design

- Avoid full attention over L × 4 × N tokens
- Preserve variable-centered representations
- Focus model capacity on global spatial teleconnections
- Use target month / rollout step to model seasonal and lead-time differences

## Key Design Decisions

### Coupled Decoder (非 Variable-specific Decoder)

WalkerNet 的 4 个输入变量（SST, HC, taux, tauy）是同一海气耦合系统的不同状态变量，物理上强耦合：

- 风应力（taux/tauy）驱动洋流 → 影响热含量（HC）→ 影响海表温度（SST）
- 纬向平流反馈：u'·∂SST/∂x 直接关联风应力和 SST 梯度
- 温跃层反馈：HC 变化 → 次表层热输送 → SST 变化

因此采用耦合解码，让解码过程中变量间信息互通，而非各自独立解码。

### Physics-informed Attention (借鉴 ENSO-PhyNet)

SST 演化遵循热收支方程：

    ∂SST'/∂t ≈ -u'·∂SST/∂x - ū·∂SST'/∂x + thermocline_feedback + Q'

将热收支方程的结构融入 attention 计算，使 attention map 具有物理可解释性：
- 模型关注哪些区域（如赤道太平洋）
- 关注哪些过程（温跃层反馈 vs 纬向平流反馈）

### CNOP Integration (后续阶段)

训练完成后，利用模型自动微分计算 CNOP（条件非线性最优扰动）：
- 预报敏感性分析：哪些初始误差场对 ENSO 预报影响最大
- CNOP 初始化的集合预报
- 参考 Tao et al. 2026, Zhou et al. 2025

## References

- STCast: Zhang et al. 2025, arXiv:2504.05574
- ENSO-PhyNet: Mu et al. 2024, npj Climate and Atmospheric Science, DOI:10.1038/s41612-024-00741-y
- AI-Enabled CNOP: Zhou et al. 2025, npj Climate and Atmospheric Science, DOI:10.1038/s41612-025-01303-6
- 3D CNOP: Tao et al. 2026, Climate Dynamics, DOI:10.1007/s00382-025-07986-0
- AI for Atmosphere-Ocean: Luo et al. 2026, National Science Review, DOI:10.1093/nsr/nwag063