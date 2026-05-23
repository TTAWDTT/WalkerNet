# Literature Review: Key Papers for WalkerNet

## 1. STCast

- **Title:** STCast: A Spatiotemporal Model for Global Weather and Climate Forecasting
- **Year:** 2025
- **arXiv:** 2504.05574v1
- **URL:** https://arxiv.org/abs/2504.05574
- **Source:** https://arxiv.org/html/2504.05574v1
- **License:** CC BY 4.0

### Authors
Yongshan Zhang, Xin Zhao, Shenghao Xiao, Chenyue Li, Yifan Chu, Shuangshuang He, Xinyan Liu, Haihong Yang, Hao Wu, Yuying Liao, Yijun He, Zhiyuan Zhao, Jiaqi Tang, Zhenwei Yu, Xinyi Ye, Ke Wang, Tian Xie

### Abstract
Accurate weather and climate forecasting is crucial for disaster preparedness, agriculture, and sustainable development. In this study, we propose STCast, a spatiotemporal model for global weather and climate forecasting. STCast jointly captures spatial and temporal dependencies by adapting a pretrained spatial Vision Transformer (ViT) encoder to construct a spatiotemporal ViT, enhanced with a novel Spatial-Aligned Attention (SAA) mechanism and a Temporal Mixture-of-Experts (TMoE) module. Furthermore, we integrate four variable-specific decoders with a climatology-aware multi-task loss to model the unique physical characteristics and climatological patterns of each variable. Extensive experimental results show that STCast achieves overall superior performance over state-of-the-art data-driven models and operational forecasting systems for both weather forecasting and climate prediction. The ensemble version further enhances forecast reliability via uncertainty quantification, achieving an ACC of 0.92 for 14-day lead time and outperforming ECMWF at lead times of 1-2 days. Climate predictions up to 7 months show superior Niño 3.4 skill scores and more accurate tropical rainfall patterns relative to SEAS4.33.

### Key Architecture Components
1. **Joint Time-Variable Patch Embedding** - Captures spatial, temporal, and inter-variable correlations
2. **Spatiotemporal ViT Encoder** - Adapts ViT with Swin Transformer backbone
3. **Spatial-Aligned Attention (SAA)** - Complex-to-real projection + L2 normalization for robustness
4. **Temporal Mixture-of-Experts (TMoE)** - Expert subnetworks with gating networks (initialized via SVD) for temporal dynamics
5. **Variable-specific Decoders** - Tailored to different physical fields (geopotential, wind, temperature, humidity, etc.)
6. **Climatology-aware Multi-task Loss** - Precomputed climatological means and variances

---

## 2. CNOP Papers (Conditional Nonlinear Optimal Perturbation)

### Paper 2a: AI-Enabled CNOP for El Niño Prediction

- **Title:** AI-Enabled conditional nonlinear optimal perturbation enhances ensemble prediction of extreme El Niño events
- **Authors:** Lumin Zhou, Rong-Hua Zhang, Lingjiang Tao
- **Year:** 2025
- **Journal:** npj Climate and Atmospheric Science
- **DOI:** 10.1038/s41612-025-01303-6
- **Source:** https://www.nature.com/articles/s41612-025-01303-6

### Abstract
Introduces orthogonal conditional nonlinear optimal perturbation (O-CNOP) integrated into a deep learning-based ensemble prediction system. Shows significant improvements in DL model predictions when initialized in spring for extreme El Niño events.

### Paper 2b: CNOP + Deep Learning for La Niña Prediction

- **Title:** Integrating an CNOP analysis into a deep learning model to identify optimal initial errors for 2020–2022 La Niña prediction
- **Authors:** Lingjiang Tao, Rong-Hua Zhang, Wansuo Duan, Lumin Zhou, Tiaoye Li
- **Year:** 2026
- **Journal:** Climate Dynamics
- **DOI:** 10.1007/s00382-025-07986-0
- **Source:** https://link.springer.com/article/10.1007/s00382-025-07986-0

### Abstract
Introduces an innovative 3D CNOP analysis framework within a data-driven model context. Combines CNOP analysis with deep learning to identify three-dimensional optimal initial errors (OIEs) and assess their influence on La Niña prediction.

### CNOP Background Paper (from arXiv)

- **Title:** The Sampling Method for Optimal Precursors of ENSO Events
- **Authors:** Bin Shi, Junjie Ma
- **Year:** 2023
- **arXiv:** 2308.13830v1
- **URL:** https://arxiv.org/abs/2308.13830

### Abstract
Addresses ENSO forecasting using the Zebiak-Cane (ZC) intermediate coupled ocean-atmosphere model. Applies a sampling algorithm based on statistical machine learning to obtain optimal precursors via the CNOP approach. Reduces gradient (first-order information) to objective function value (zeroth-order information), eliminating the need for an adjoint model. Supports parallel computation.

---

## 3. AI for Atmosphere-Ocean Sciences

- **Title:** AI for atmosphere–ocean sciences: advancements, challenges and ways forward
- **Authors:** Jing-Jia Luo (lead), with 40+ co-authors (including Christopher Bretherton, Pierre Gentine, Niklas Boers, Lina Yao)
- **Year:** 2026
- **Journal:** National Science Review
- **DOI:** 10.1093/nsr/nwag063
- **PMID:** 41822045
- **PMCID:** PMC12976684

### Abstract
A comprehensive review covering:
- Deep-learning methods for weather/climate forecasting that outperform dynamical models in accuracy and computational efficiency
- AI applications in detecting complex phenomena, data assimilation, bias correction, and downscaling
- Advocacy for hybrid physics-AI modeling to ensure generalizability
- AI-based model intercomparison (AI-MIP) framework
- Explainable AI to address the "black-box" nature
- Future "AI agents for Earth science — autonomous, goal-oriented systems"

### Keywords
AI agent; AI application and challenge; AI-MIP; atmosphere–ocean sciences; explainable AI

---

## 4. ENSO-PhyNet (Heat Budget + Transformer)

- **Title:** Incorporating heat budget dynamics in a Transformer-based deep learning model for skillful ENSO prediction
- **Authors:** Bin Mu, Yuehan Cui, Shijin Yuan, Bo Qin
- **Year:** 2024
- **Journal:** npj Climate and Atmospheric Science
- **DOI:** 10.1038/s41612-024-00741-y
- **Source:** https://www.nature.com/articles/s41612-024-00741-y

### Abstract
Addresses how deep learning models for ENSO prediction often suffer from being "black-box" systems lacking physical consistency. Introduces ENSO-PhyNet, a Transformer-based approach that predicts SST in the equatorial Pacific. Incorporates heat budget dynamical processes through self-attention computations and achieves skillful Niño 3.4 index predictions with up to 22 months lead time. Self-attention maps reveal which processes and regions drive predictions. Case studies of recent El Niño/La Niña events highlight the roles of thermocline feedback and zonal advection feedback in the 2015 warming and anomalous easterlies in the emergence of the second-year La Niña in 2021, demonstrating physical interpretability.

### Key Methodology
- **Transformer-based architecture** (ENSO-PhyNet)
- Embeds **ocean heat budget dynamics** directly into the self-attention mechanism
- Physical equations governing SST evolution (heat budget terms like thermocline and zonal advection feedbacks) are structurally incorporated
- Attention maps offer interpretable, physically consistent insight
- Predicts Niño 3.4 index with 22-month lead time
