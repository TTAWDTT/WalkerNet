# Manifold-Constrained CNOP: Phase 1 Research Scope

## Status

Phase 1 (scoping) completed on 2026-08-26. This document records the research-question brief, preliminary methodology blueprint, scope boundaries, assumptions, and the pre-investigation adversarial check. It is a planning artifact, not evidence that any proposed method has already been validated.

## Topic

Extend CNOP from amplitude-constrained optimization in raw physical-field space to differentiable, finite-amplitude optimization on a data-supported Earth-system state manifold.

The phrase *data-supported climate-state manifold* is intentionally used instead of *the true Earth manifold*: a diffusion or other generative model learns the distribution represented by its training data and architecture, not the complete set of physically possible Earth states.

## Primary Research Question

**In a trained global ocean–atmosphere forecasting model, can a differentiable generative-manifold parameterization enable stable finite-amplitude CNOP optimization while preserving the naturalness of the perturbed Earth-system state?**

The question tests three linked properties: (i) whether optimized perturbations remain close to naturally supported climate states, (ii) whether gradients propagate through admissible manifold directions, and (iii) whether this is more stable than taking a raw gradient step followed by post-hoc diffusion denoising.

## Candidate Questions Considered

| Candidate | FINER average | Role |
|---|---:|---|
| Can differentiable generative-manifold CNOP stably optimize finite-amplitude responses while preserving state naturalness? | 4.6/5 | Selected primary question |
| Does diffusion projection produce more natural ENSO precursors than ordinary CNOP? | 3.7/5 | Sub-question; “natural” needs an operational definition |
| Which of latent-space, score-based, and tangent-space constraints is most suitable for CNOP? | 4.0/5 | Method-comparison sub-question |
| Does a manifold constraint identify more credible ENSO OPR and sensitive areas? | 3.8/5 | Downstream scientific application |

## Preliminary FINER Assessment

| Criterion | Score | Rationale |
|---|---:|---|
| Feasible | 4/5 | WalkerNet is differentiable and existing CNOP code can be extended with a generative prior and constrained optimizer. |
| Interesting | 5/5 | Addresses the fundamental gap between a mathematically effective perturbation and a naturally supported climate state. |
| Novel | 4/5 | The combination of CNOP, generative climate-state priors, and manifold/tangent-space optimization is a strong intersectional opportunity; exact novelty requires Phase 2 verification. |
| Ethical | 5/5 | No human subjects are required; the main integrity risk is overinterpreting learned plausibility as physical causality. |
| Relevant | 5/5 | Potentially informs ENSO precursor analysis, counterfactual forecasting, and targeted observations. |
| **Average** | **4.6/5** | Sufficiently precise for literature and feasibility investigation. |

## Scope Boundaries

### In scope

- WalkerNet-like trained global ocean–atmosphere forecasting models.
- Monthly TOS, ZOS, TAUU, and TAUV fields.
- A continuous historical window (initially the 12-month WalkerNet input window), rather than an isolated frame.
- Historical CMIP6 multi-model data, with source/year holdouts where feasible.
- ENSO and Niño3.4 forecast responses over 12–18 autoregressive months.
- Three candidate constraint families: latent-space parameterization, score-based naturalness regularization, and differentiable tangent-space projection.
- Comparisons against unconstrained or amplitude-constrained CNOP, post-hoc gradient-plus-denoising projection, zero-state local gradients, and matched-radius random perturbations.

### Out of scope

- Claiming that a learned generator recovers the complete true Earth-system dynamical manifold.
- Claiming that a high-probability generated state is a unique causal mechanism or is guaranteed to occur on Earth.
- Extending immediately to all SSP scenarios or all climate variables.
- Using a single ENSO case to establish universal superiority.

### Key assumptions

1. Historical CMIP6 data provide a useful, though incomplete, state-distribution prior.
2. A generative model can learn major spatial, seasonal, and cross-variable couplings.
3. Naturalness must be assessed by multiple independent diagnostics, not diffusion likelihood alone.
4. Rare but observed ENSO precursor states must not be rejected merely because they have low global probability.
5. Shared data biases between the generator and WalkerNet will be tested with source/year holdouts and, where possible, independent reanalysis checks.

## Sub-questions

1. **Optimization stability:** Compared with ordinary CNOP and post-hoc diffusion projection, does differentiable manifold-constrained CNOP reduce objective oscillation, rollback frequency, and seed-to-seed variance?
2. **Constraint family:** Which of latent-space, score-based, and tangent-space constraints gives the most stable trade-off between naturalness and forecast response under matched compute and amplitude budgets?
3. **Scientific interpretation:** Does the constraint change the ENSO optimal precursor and targeted-observation-sensitive area identified by CNOP, and are those patterns stable across seeds, sources, and manifold models?

All sub-questions inherit the same WalkerNet/CMIP6 historical/ENSO/12–18-month/four-variable scope. Their deviations are only the stated emphasis on stability, method comparison, or scientific interpretation.

## Preliminary Methodology Blueprint

### State representation

The preferred primary object is a trajectory/window:

\[
X_t = [x_{t-11},\ldots,x_t],
\]

where each \(x_t\) contains the four global fields. This preserves temporal continuity, seasonal phase, and coupled SST–SSH–wind structure that a single-frame manifold may miss.

### Conditional generative prior

The prior should condition, where supported by data, on calendar month, source/model identity, variable identity, temporal order, and the local ENSO background state. An unconditional generator is a useful ablation but should not be the sole scientific prior.

### Candidate formulations

#### Latent-space CNOP

With \(X=G_\theta(z,c)\), optimize:

\[
z^* = \arg\max_z J\big(\mathcal{F}(G_\theta(z,c))\big) - \lambda_z R_z(z),
\]

with gradient

\[
\nabla_z J = \left(\frac{\partial G_\theta}{\partial z}\right)^\top \nabla_X J.
\]

This is the most direct realization of gradients restricted to generator-supported directions.

#### Score-based constrained CNOP

Using a score \(s_\theta(X,\sigma)\approx\nabla_X\log p_\theta(X)\), optimize a forecast objective with an explicit naturalness penalty. This defines a high-density state preference rather than a strict hard manifold.

#### Tangent-space CNOP

Approximate the admissible tangent space \(T_X\mathcal{M}\) using the generator Jacobian or local score geometry and project forecast gradients into that space before updating. This is conceptually closest to intrinsic manifold optimization but likely the most computationally difficult.

## Required Evaluation Axes

### Forecast response

- Late-lead three-month Niño3.4 response and lead-12 response.
- Response gain relative to the unperturbed forecast.
- Paired improvement relative to the local-gradient baseline.

### State naturalness

- Diffusion reconstruction error and/or score energy.
- Latent distance and nearest-neighbour distance to held-out historical states.
- One-step transition plausibility.
- Marginal and joint TOS/ZOS/TAUU/TAUV distribution shift.
- Spatial and temporal spectra and cross-variable correlations.
- Retention of rare but observed ENSO precursor states.

### Optimization stability

- Objective trajectory and objective monotonicity.
- Accepted/rejected proposal ratio and rollback frequency.
- Seed-to-seed variance and final-objective variance.
- Gradient norm, projection norm, and naturalness drift during optimization.

### Scientific interpretation

- Spatial stability of OPR across seeds and manifold models.
- Cross-source consistency across CMIP6 models.
- Agreement with historical ENSO precursor composites.
- Sensitivity of targeted-observation areas to the chosen naturalness definition.

## Phase-1 Adversarial Check

**Provisional verdict: REVISE BEFORE LARGE-SCALE SEARCH.** No fatal flaw was found, but the following major risks must be addressed before a large experiment.

1. **Distribution is not the true manifold.** A diffusion model learns the data-supported distribution, not the complete Earth dynamics. Use calibrated language such as “data-supported naturalness” and “model-distribution plausibility.”
2. **Naturalness has no unique scalar definition.** Likelihood, reconstruction error, historical proximity, and physical-statistical consistency may disagree. Pre-register a multi-metric evaluation instead of selecting a favourable metric after seeing results.
3. **Rare-event suppression.** A global prior may remove low-probability but genuinely observed ENSO precursors. Include rare-but-observed states and conditional/phase-aware priors.
4. **Generator coverage.** A weak generator can make its latent manifold look clean simply because it cannot represent real variability. Measure reconstruction fidelity and coverage before using it as a hard constraint.
5. **Shared model/data bias.** A generator and WalkerNet trained on the same CMIP6 records may share artifacts. Use source/year holdouts and independent observations where available.

### Strongest counter-argument

The method may discover states preferred by one diffusion model rather than more natural CNOPs. The perturbation, ENSO response, and OPR could all reflect generator bias rather than Earth-system structure. Phase 2 must therefore treat generator choice as a source of uncertainty and include cross-prior and held-out validation.

## Phase-2 Search Plan

Search and verify six evidence families:

1. CNOP, conditional nonlinear optimal perturbation, optimal precursor, and targeted observations.
2. Diffusion models for weather/climate state generation and climate manifolds.
3. Score-based constraints, score distillation, and differentiable projection.
4. Latent-space optimization with generative priors.
5. Riemannian optimization, tangent-space gradients, and manifold-constrained learning.
6. Physics-informed generative models, conservation-aware diffusion, and climate plausibility.

The investigation will actively seek counter-evidence on mode collapse/over-smoothing, likelihood–physics mismatch, and latent-space omission of rare events.

