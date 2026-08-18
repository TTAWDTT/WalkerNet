# WalkerNet TOS/ZOS CNOP 初步结果

日期：2026-07-03

## 实验设置

- 模型：`checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt`
- checkpoint epoch：6
- 预报算子：WalkerNet 12 个月 autoregressive rollout
- 目标：最大化目标年内 Niño3.4 anomaly 的三个月滑动平均 soft maximum
- case 选择：从全历史 source 中选择 observed `max(abs(3-month Niño3.4 anomaly))` 最小的 10 个 Jan-Dec 年段
- 扰动变量：输入窗口第 12 个月的 `tos` 和 `zos`
- 扰动区域：热带太平洋 `20S-20N, 120E-290E`
- 扰动参数化：`45 x 90` patch 网格，双线性上采样到 `180 x 360`
- 约束：

```text
RMS(delta_tos) <= 0.1
RMS(delta_zos) <= 0.1
abs(delta)     <= 2.0
smoothness_weight = 0.001
```

## 结果表

| Source | Target year | Observed neutral score | Baseline max 3m Niño3.4 | CNOP max 3m Niño3.4 | Gain |
|---|---:|---:|---:|---:|---:|
| IPSL-CM6A-LR | 1880 | 0.126 | -0.031 | 1.355 | 1.386 |
| EC-Earth3 | 1959 | 0.158 | -0.225 | 1.075 | 1.301 |
| EC-Earth3 | 1942 | 0.183 | 0.286 | 1.760 | 1.474 |
| EC-Earth3 | 1975 | 0.195 | -0.086 | 1.247 | 1.334 |
| MPI-ESM1-2-HR | 1876 | 0.221 | -0.432 | 1.092 | 1.525 |
| GFDL-ESM4 | 1995 | 0.229 | 0.100 | 1.688 | 1.588 |
| MPI-ESM1-2-HR | 2003 | 0.232 | 0.430 | 1.879 | 1.449 |
| CESM2 | 1856 | 0.234 | 0.108 | 0.863 | 0.755 |
| EC-Earth3 | 1878 | 0.241 | 0.326 | 1.715 | 1.389 |
| GFDL-ESM4 | 1930 | 0.252 | -0.606 | 1.114 | 1.721 |

## 主要发现

1. 选出的 10 个目标年在真实场上都非常 neutral，observed neutral score 为 `0.126-0.252`。
2. baseline 预报也基本不表现为强 El Niño，其中 6 个 case 的 baseline max 3m Niño3.4 低于 0.1 或为负。
3. 加入 CNOP 后，10 个 case 全部超过 `0.86`，其中 9 个超过 `1.0`，说明 WalkerNet 内部确实存在能触发 ENSO-like 响应的 TOS/ZOS 最优扰动方向。
4. 最大 gain 出现在 `GFDL-ESM4 1930`，从 `-0.606` 推到 `1.114`，gain 为 `1.721`。
5. `GFDL-ESM4 1995` 的最终强度最高，达到 `1.688`。

## 扰动因子与前兆诊断

重新绘制合成图后，CNOP 的主要结构比单个 case 图更清楚：

1. **TOS 是主导因子。** 10 个 neutral case 的合成扰动在 Niño3.4 与中东赤道太平洋呈稳定正值，Niño3.4 区域平均 TOS 扰动为 `+1.2826`，中东赤道太平洋为 `+1.0858`。
2. **关键前兆像“东暖西弱”的赤道海温梯度调整。** 西太平洋赤道区域平均 TOS 扰动为 `-0.1701`，而中东太平洋明显为正，东西向对比为 `+1.2560`。这说明最优扰动不是简单全海盆升温，而是在 ENSO 敏感区加强东/中太平洋暖异常。
3. **ZOS 提供上层海洋状态线索，但不如 TOS 稳。** ZOS 东西向倾斜指标平均为 `+0.0727`，可以理解为 WalkerNet 内部利用了海面高度/上层海洋状态代理来配合海温响应；但该信号跨 case 的一致性弱于 TOS，不能把它等同于真实热含量或严格热跃层深度。
4. **扰动空间形态具有跨 case 一致性。** `|mean| / spread` 与 sign-agreement 图显示，Niño3.4 盒及其附近的正 TOS 扰动并不是单个年份偶然纹理，而是多个 neutral case 共同指向的敏感方向。
5. **gain 大小还受 baseline 初态影响。** baseline 越偏负的 case 往往有更大的可增长空间，例如 `GFDL-ESM4 1930` 从 `-0.606` 被推到 `1.114`，gain 最大；而最终强度最高的是 baseline 已经偏正的 `MPI-ESM1-2-HR 2003`。

## 图像输出

服务器输出目录：

```text
outputs/cnop_tos_zos
```

本地已拉取：

```text
outputs/cnop_tos_zos_patch_0703/cnop_gain_summary.png
outputs/cnop_tos_zos_patch_0703/best_case_cnop_maps_and_nino.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_composite_diagnostics.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_precursor_diagnostics.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_tos_case_atlas.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_zos_case_atlas.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_factor_comparison.png
outputs/cnop_tos_zos_patch_0703/figures/cnop_precursor_indices.csv
outputs/cnop_tos_zos_patch_0703/figures/cnop_precursor_analysis.md
```

新图说明：

- `cnop_composite_diagnostics.png/pdf`：展示 TOS/ZOS 合成 CNOP 扰动、baseline/CNOP Niño3.4 月序列，以及各 case 的 gain。
- `cnop_precursor_diagnostics.png/pdf`：展示 TOS 扰动稳健性、符号一致性、区域前兆指数，以及 baseline 与 gain 的关系。
- `cnop_tos_case_atlas.png/pdf`：逐个展示 10 个 neutral case 的 TOS 扰动，使用统一色标和更宽经纬度视野。
- `cnop_zos_case_atlas.png/pdf`：逐个展示 10 个 neutral case 的 ZOS 扰动，便于检查 ZOS 带状/倾斜结构是否稳定。
- `cnop_factor_comparison.png/pdf`：把每个 case 的 TOS Niño3.4、TOS 西太平洋、TOS 中东太平洋、TOS 东西向对比、ZOS 东西向倾斜放在同一张矩阵中比较。

## 方法修正记录

初始 full-grid 扰动会产生明显逐格点高频/棋盘格结构，不适合作为主结果展示。随后改为 patch-grid 参数化，扰动图更平滑，且仍能稳定诱发 Niño3.4 正响应。

另一次 `0.5σ` 半径 smoke test 能把 Niño3.4 推到极大值，说明模型对该方向非常敏感，但该半径容易变成“模型敏感性利用”，因此主结果采用 `0.1σ`。

后续方法升级为多初值 top-k CNOP 搜索：每个 case 默认从 16 个初值出发做 projected Adam，保存目标函数最强的 5 个局部 CNOP 候选，并支持对 top-k 进行 projected L-BFGS 精修。当前 `outputs/cnop_tos_zos_patch_0703` 的图来自升级前的 top-1 结果；如需比较“同一个 case 的多个最优扰动”，需要用新脚本重新运行 CNOP，输出中会包含 `top_delta_phys` 与 `cnop_candidate_summary.csv`。

按“扰动二范数限制为初始 TOS/ZOS 场二范数 10%，并最大化第 12 个月 Niño3.4 扰动前后差值”的定义，已对 `IPSL-CM6A-LR 1880` 做真模型 smoke 实验：

```text
constraint_mode = relative_initial_l2
relative_l2_fraction = 0.1
objective_mode = lead_delta
objective_lead = 12
num_starts = 8
steps = 40
```

结果显示最强候选的第 12 个月 Niño3.4 从 `-0.3489` 推到 `3.1160`，`lead_delta = 3.4649`；TOS 与 ZOS 的物理二范数比例均校验为 `0.1`。该约束明显宽于此前 `0.1σ RMS` 约束，因此响应强度也更大。

## 当前结论

在 WalkerNet 这个训练好的非线性预报算子上，已经可以稳定求出 TOS/ZOS CNOP-like 扰动。该扰动施加在前一年 12 月输入场上，可以把本来 neutral 的目标年预报推向 El Niño-like 状态。

这仍是“模型内 CNOP”，不是严格物理气候系统 CNOP。下一步需要做：

1. 对比正 CNOP 与负 CNOP，即 El Niño / La Niña 双向目标。
2. 做扰动半径敏感性实验，例如 `0.05, 0.1, 0.2`。
3. 增加更物理的空间平滑或能量约束。
4. 检查 CNOP 模式是否跨 source 一致。
