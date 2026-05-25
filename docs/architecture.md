# Architecture Notes

## Task Definition

- 输入：连续 3 个月的全球物理场
- 输出：下 1 个月的全球物理场
- 支持滚动预测（autoregressive rollout）
- CNOP 研究在预报模型训练好之后进行，本项目本身只做预报

## Variables

4 个变量，耦合的海气系统状态变量：

- SST（海表温度）
- HC（海洋热含量）
- taux（纬向风应力）
- tauy（经向风应力）

## Shapes

输入：B × L × 4 × H × W（L 为历史窗口长度，如 3 或 12 个月）
输出：B × 1 × 4 × H × W（预测下 1 个月）

## Pipeline

输入: B × L × 4 × H × W（L = 3 或 12，可配置）

↓ Joint Time-Variable Patch Embedding
将 L 个时间步 × 4 个变量一起 patch 化，捕获跨时间和跨变量的交互

Z: B × (L×4) × N × d

↓ Spatial Attention
空间注意力，捕捉全球尺度的遥相关

↓ TMoE (Temporal Mixture-of-Experts)
用 target month 条件化路由，不同月份分配到不同 expert

↓ Coupled Variable Decoder
4 变量联合解码，输出 B × 1 × 4 × H × W

## Rollout 机制

单步预测只预报 1 个月。多步预测通过自回归 rollout：

- 将上一步输出拼接到输入窗口末尾，滑动一步，再预测下一个月
- 每步的 rollout step 编码为一个 embedding，注入模型作为条件信息
- rollout embedding 记录当前是第几步预测，让模型感知 lead time

注意：rollout step embedding 是独立的条件编码，不一定要用于 TMoE 的路由。

## Conditioning Signals

模型有两个条件信号：

1. **Target month**：预测目标的月份（1-12），用于 TMoE 路由，捕捉季节性差异
2. **Rollout step**：当前滚动步数，编码为 embedding 注入，让模型区分直接预测 vs 多步累积预测

## Why This Design

- Joint embedding 避免 L × 4 × N 的全注意力开销
- 变量耦合解码利用 SST-HC-taux-tauy 的物理关联
- TMoE 捕捉不同月份的预报差异
- Rollout embedding 记录 lead time 信息，区分短期和累积误差

## CNOP Integration (后续)

训练完成后，利用模型自动微分计算 CNOP（条件非线性最优扰动）：
- 预报敏感性分析：哪些初始误差场对预报影响最大
- CNOP 初始化的集合预报
- 参考 Tao et al. 2026, Zhou et al. 2025

## References

- STCast: Zhang et al. 2025, arXiv:2504.05574
- AI-Enabled CNOP: Zhou et al. 2025, npj Climate and Atmospheric Science, DOI:10.1038/s41612-025-01303-6
- 3D CNOP: Tao et al. 2026, Climate Dynamics, DOI:10.1007/s00382-025-07986-0
