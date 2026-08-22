# CNOP 中性年到 El Ni\~no 反事实实验规范

本文档定义 workshop 主结果的案例资格，避免将探索性响应图误写成
“CNOP 从中性态诱发 ENSO”的证据。

## 冻结的案例筛选

每个候选都必须是完整的 Jan--Dec 目标年，并同时通过下列门槛：

| Gate | Criterion | Purpose |
|---|---|---|
| Truth neutral | observed maximum 3-month $|\mathrm{Ni\~no3.4}| \leq 0.5^\circ$C | 真值目标年不是 ENSO |
| Forecast fidelity | 12-month baseline--truth Ni\~no3.4 RMSE $\leq 0.40^\circ$C | 基准预报足以代表该中性态 |
| Endpoint fidelity | lead-12 absolute error $\leq 0.50^\circ$C | 排除末期明显漂移的预报 |
| Baseline neutral | baseline maximum 3-month Ni\~no3.4 $< 0.5^\circ$C | 排除模型本身已经预报 El Ni\~no 的个例 |

`scripts/cnop/sample_cnop_cases_by_baseline.py` 将所有中性候选写入
`neutral_baseline_candidates.csv`，并只把通过所有 gate 的样本写入
`selected_cases.csv`。如果样本不足，程序会显式失败；不得以较差的
case 静默补足。

### Anomaly reference

The truth, baseline forecast, and CNOP-perturbed forecast used for the
counterfactual gates are all expressed relative to the **same source-wise
monthly Niño3.4 climatology estimated from the training years**.  Thus their
thresholds and baseline--truth RMSE share one zero point.  A separate,
lead-dependent forecast climatology may be recorded for model-bias diagnosis,
but it is never mixed with source-referenced truth in a fidelity RMSE or an
event-threshold decision.

## CNOP 后资格

对 **同一份** `selected_cases.csv`，分别在 `pacific`、`atlantic_indian`
和 `global` 三个掩膜下求解；每个域都必须使用相同变量、目标函数、
rollout 长度和 constraint scale。一个域内的结果仅在以下条件下才是
合格的 El Ni\~no 反事实：

\[
\mathrm{truth\ neutral}\ \land\ \mathrm{baseline\ neutral}\ \land\
\max\overline{\mathrm{Ni\~no3.4}}_{\mathrm{CNOP}} \geq 0.5^\circ\mathrm{C}.
\]

使用 `scripts/cnop/qualify_cnop_counterfactuals.py` 生成每个域的
`counterfactual_qualified_cases.csv`。该文件而非原始 summary 是 workshop
主图的唯一 case 来源。

## 执行顺序

1. 在 Historical WalkerNet checkpoint 与固定 forecast climatology 上运行
   baseline 筛选，冻结 `selected_cases.csv` 并提交版本控制。
2. 先以 3 个冻结 case 做 `0.05`、`0.10` 的三域 pilot，检查优化稳定性和
   资格通过率；`0.20` 只作为敏感性分析。
3. 对同一冻结 case set 运行正式三域 CNOP。
4. 对每个 domain 运行资格脚本；报告合格率和所有未通过的原因，不能只
   展示成功个例。
5. 代表图按 baseline RMSE 最低、同时 CNOP 跨阈值且扰动幅度未触及 clip
   上限的 case 选择；composite 使用全部合格个例。

这项试验检验的是 WalkerNet 内部的受约束反事实敏感性。CNOP 扰动是
model-based optimal precursor，不应被表述为对真实气候系统唯一因果解释。
