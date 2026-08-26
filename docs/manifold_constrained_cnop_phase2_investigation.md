# Manifold-Constrained CNOP: Phase 2 Investigation

## Status and evidence boundary

- Phase: Phase 2 (Investigation), first targeted search pass.
- Search date: 2026-08-27 (Asia/Shanghai).
- Scope: CNOP/optimal precursors; climate and weather diffusion; score/manifold geometry; differentiable guidance and inverse problems; physics-informed generative models; on-manifold perturbations.
- Retrieval surfaces: Google Web search for discovery; Crossref DOI metadata and abstracts; arXiv API records and abstracts; official Nature/AMS/AGU/Springer/NeurIPS landing pages when surfaced.
- Evidence status: DOI metadata and abstracts were verified for the sources explicitly summarized below. Full-text verification was not completed for every source; abstract-level claims are labeled accordingly.
- This is a targeted Phase-2 investigation, not yet a PRISMA-complete systematic review. Search-engine hit counts were not treated as reproducible database counts and are therefore not invented.

## Search strategy

### Query families

1. `conditional nonlinear optimal perturbation CNOP optimal precursor climate`
2. `diffusion models weather climate forecasting`
3. `generative diffusion model Earth system state manifold climate`
4. `score-based generative models detect manifolds`
5. `Riemannian diffusion models constrained domains`
6. `diffusion posterior sampling manifold constrained gradient`
7. `physics-informed diffusion climate weather`
8. `on-manifold adversarial examples latent-space perturbation`

### Inclusion criteria

- Directly addresses CNOP/optimal precursor/targeted observation, or provides a method directly usable for a generative-manifold-constrained CNOP.
- Peer-reviewed paper, major conference paper, official preprint, or authoritative project paper with a stable DOI/arXiv/official landing page.
- Contains a method, theoretical result, or evaluation relevant to state-distribution support, manifold geometry, differentiable guidance, physical constraints, or nonlinear optimal perturbation.

### Exclusion criteria

- Generic diffusion or adversarial papers with no transferable manifold/guidance/constraint mechanism.
- Climate papers that use “manifold” only as informal prose without a state-generation or geometry method.
- Search-result summaries without a DOI, arXiv record, or authoritative landing page.

## Evidence matrix

| Cluster | Representative sources | What is established | Relevance to the proposed study | Main limitation |
|---|---|---|---|---|
| CNOP foundations and ENSO OPR | Mu et al. (2003); Duan et al. (2004); Duan et al. (2009); Mu et al. (2010); Zhou et al. (2025/2026) | CNOP is a constrained nonlinear optimal perturbation framework; ENSO studies report nonlinear optimal precursors distinct from linear singular vectors; recent O-CNOP work uses historical candidate samples and orthogonal perturbations in a DL ensemble | Establishes the objective, OPR target, and a strong recent baseline for learned-model CNOP | Existing O-CNOP candidate generation is historical/ensemble based rather than a differentiable learned-manifold optimization |
| CNOP algorithms and observations | Sun et al. (2010); Mu et al. (2009); Zhou et al. (2021); Zu et al. (2025) | Optimization algorithms, targeted-observation applications, ensemble perturbations, and time-dimension extensions are active CNOP topics | Provides baselines, algorithmic concerns, and evaluation ideas | Does not impose a learned probabilistic state support during CNOP optimization |
| Score/diffusion geometry | Song et al. (2021); Pidstrigach (2022); Huang et al. (2022); Jo & Hwang (2023); Fishman et al. (2023) | Score fields can support generation near low-dimensional data structure; diffusion can be generalized to Riemannian manifolds and constrained domains | Supplies the mathematical language for score/tangent/latent CNOP variants | General theory and benchmarks are not Earth-system-specific; “data manifold” still depends on data and score accuracy |
| Diffusion guidance and inverse problems | Graikos et al. (2022); Chung et al. (2022); Bansal et al. (2023) | A differentiable auxiliary objective can guide a fixed diffusion prior; diffusion posterior sampling blends denoising with a manifold-constrained gradient | Directly addresses the user’s concern about objective guidance through denoising | These methods are approximate sampling/inference procedures, not monotone maximization of a CNOP forecast objective |
| Climate/ESM generative models | Bassetti et al. (2023, 2024); Christensen et al. (2024); Brenowitz et al. (2025); Bouabid et al. (2026) | Diffusion/score models can emulate climate/ESM distributions, joint variables, extremes, or large-scale modes; diagnostics must include marginals, correlations, tails, regimes, and forced responses | Supports training a climate-state prior and gives a concrete naturalness audit suite | Several works report failure modes under regime shifts, extrapolation, bias, or unresolved tails; generation is not equivalent to physical validity |
| Physics-informed generative modeling | Bastek et al. (2024) and related physics-informed diffusion work | Physical residuals and equality/inequality constraints can be included in diffusion training or generation | Suggests a route beyond purely statistical manifold support | Generic PDE demonstrations do not automatically encode coupled ocean–atmosphere balances or climate conservation laws |
| On-manifold perturbations | Rahman et al. (2022); related latent-space on-manifold attack work | Perturbations constrained to a learned data manifold can remain effective; off-manifold distance is not a sufficient robustness criterion | Supports testing whether a natural perturbation can still strongly affect a downstream model | Adversarial image settings are not climate dynamics; “on-manifold” quality depends strongly on generator reconstruction and coverage |

## Annotated bibliography

### A. CNOP and optimal precursor literature

**Mu, M., Duan, W., & Wang, B. (2003). Conditional nonlinear optimal perturbation and its applications. *Nonlinear Processes in Geophysics, 10*, 493–501.** [DOI](https://doi.org/10.5194/npg-10-493-2003)

- Relevance: Foundational CNOP definition and early ENSO application.
- Abstract-level finding: CNOP is formulated for nonlinear predictability; in a coupled ocean–atmosphere ENSO model, same-norm CNOPs were reported to evolve toward El Niño or La Niña more readily than linear singular vectors.
- Contribution: Establishes the nonlinear constrained-optimization target that the proposed manifold extension must preserve.
- Limitation: Idealized coupled model; no learned generative prior.

**Duan, W., & Mu, M. (2004). Conditional nonlinear optimal perturbations as the optimal precursors for El Niño–Southern Oscillation events. *Journal of Geophysical Research: Atmospheres*.** [DOI](https://doi.org/10.1029/2004JD004756)

- Relevance: Direct precedent for OPR and ENSO.
- Abstract-level finding: CNOP/local CNOP patterns were reported to have higher likelihood of developing into El Niño/La Niña than LSVs in a theoretical coupled model, with qualitative agreement to observations.
- Contribution: Gives the scientific definition of OPR used by the proposed work.
- Limitation: The claim is model-based and theoretical; it does not establish natural-state support in a data-driven field space.

**Duan, W., Xue, F., & Mu, M. (2009). Investigating a nonlinear characteristic of El Niño events by conditional nonlinear optimal perturbation. *Atmospheric Research*.** [DOI](https://doi.org/10.1016/j.atmosres.2008.09.003)

- Relevance: Shows CNOP can interrogate nonlinear ENSO characteristics under different background states.
- Contribution: Motivates conditioning the learned prior on background/season rather than using a single unconditional manifold.

**Mu, M., Zhang, K., & Wang, Q. (2022). Recent progress in applications of the conditional nonlinear optimal perturbation approach to atmosphere–ocean sciences. *Chinese Annals of Mathematics, Series B*.** [DOI](https://doi.org/10.1007/s11401-022-0376-8)

- Relevance: Review source for CNOP applications, stability, sensitivity, and atmosphere–ocean use.
- Contribution: Useful source for positioning the proposed method relative to classical CNOP applications.

**Sun, G., Mu, M., & Zhang, Y. (2010). Algorithm studies on how to obtain a conditional nonlinear optimal perturbation (CNOP). *Advances in Atmospheric Sciences*.** [DOI](https://doi.org/10.1007/s00376-010-9088-1)

- Relevance: Directly relevant to optimizer stability and algorithm choice.
- Contribution: Supports treating optimization behavior—not only the final objective—as a first-class methodological outcome.

**Mu, M., Zhou, F., & Wang, H. (2009). A method for identifying the sensitive areas in targeted observations for tropical cyclone prediction: Conditional nonlinear optimal perturbation. *Monthly Weather Review*.** [DOI](https://doi.org/10.1175/2008MWR2640.1)

- Relevance: CNOP-based targeted observation sensitive areas.
- Abstract-level finding: CNOP errors could differ from first singular vectors and had larger forecast impact in the tested tropical-cyclone cases.
- Contribution: Provides a downstream test for whether manifold constraints change sensitive-region maps.

**Zhou, Q., Chen, L., Duan, W., et al. (2021). Using conditional nonlinear optimal perturbation to generate initial perturbations in ENSO ensemble forecasts. *Weather and Forecasting*.** [DOI](https://doi.org/10.1175/WAF-D-21-0063.1)

- Relevance: CNOP used to generate ensemble initial perturbations for operational ENSO forecasting.
- Abstract-level finding: Replacing a leading CSV perturbation with CNOP structures increased ensemble spread and improved several reliability-related diagnostics in the reported experiments.
- Contribution: Suggests natural-manifold CNOP could be evaluated as an ensemble-perturbation generator, not only as an explanatory map.

**Zu, Z., Mu, M., Xia, J., & Wang, Q. (2025). An extension of conditional nonlinear optimal perturbation in the time dimension and its applications in targeted observations. *Advances in Atmospheric Sciences*.** [DOI](https://doi.org/10.1007/s00376-025-4297-9)

- Relevance: Recent extension of CNOP beyond a purely initial-state formulation.
- Contribution: Important adjacent work for deciding whether the naturalness constraint should be imposed on one frame or on a temporal window/trajectory.

**Zhou, L., Zhang, R.-H., & Tao, L. (2025; published in volume 9, 2026). AI-enabled conditional nonlinear optimal perturbation enhances ensemble prediction of extreme El Niño events. *npj Climate and Atmospheric Science*.** [DOI](https://doi.org/10.1038/s41612-025-01303-6)

- Relevance: Closest recent DL/CNOP precedent identified in the overlap search.
- Full-text/official-page finding: The study uses O-CNOP perturbations for a DL ENSO ensemble, generates candidate samples from historical simulations of 23 CMIP6 models plus SODA reanalysis under energy constraints, selects/optimizes perturbations iteratively, and uses an independent 2015/16 El Niño validation after constructing perturbations from other events.
- Contribution: Demonstrates that a learned-model CNOP workflow can benefit from historical candidate support and orthogonal ensemble perturbations without claiming a diffusion manifold.
- Limitation: Candidate generation is sample/ensemble based rather than a differentiable generator parameterization; the reported evaluation is for four extreme El Niño events and a DL ensemble, not a general natural-state manifold.

### B. Score models and manifold geometry

**Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations*.** [arXiv](https://arxiv.org/abs/2011.13456)

- Relevance: Score field and reverse-time SDE foundation.
- Abstract-level finding: A forward SDE perturbs data toward a known prior; a reverse-time SDE uses the time-dependent score to generate samples, with predictor–corrector and neural-ODE formulations.
- Contribution: Supplies the differentiable score formalism needed for score-constrained CNOP.

**Pidstrigach, J. (2022). Score-based generative models detect manifolds. *Advances in Neural Information Processing Systems*.** [arXiv](https://arxiv.org/abs/2206.01018)

- Relevance: Directly addresses whether score-based models can recover low-dimensional data structure.
- Abstract-level finding: Gives conditions under which score-based generative models produce samples from an underlying data manifold, while warning that sample frequencies need not match the true generating distribution.
- Contribution: Strong theoretical support for separating manifold support from probability calibration—central to the proposed study.

**Huang, C.-W., Aghajohari, M., Bose, A. J., Panangaden, P., & Courville, A. (2022). Riemannian diffusion models. *Advances in Neural Information Processing Systems*.** [arXiv](https://arxiv.org/abs/2208.07949)

- Relevance: Diffusion defined directly on Riemannian manifolds.
- Abstract-level finding: Generalizes continuous-time diffusion to arbitrary Riemannian manifolds and relates variational likelihood estimation to Riemannian score matching.
- Contribution: Provides a principled route if a climate state manifold can be represented geometrically, though the 4D ocean–atmosphere state manifold is not known a priori.

**Jo, J., & Hwang, S. J. (2023). Generative modeling on manifolds through mixture of Riemannian diffusion processes. *International Conference on Learning Representations*.** [arXiv](https://arxiv.org/abs/2310.07216)

- Relevance: Scalable generative modeling on general manifolds.
- Abstract-level finding: Uses mixtures of Riemannian bridge processes and tangent directions without requiring heat-kernel estimates.
- Contribution: Potentially relevant to multimodal climate regimes and disconnected/locally different state neighborhoods.

**Fishman, N., Klarner, L., De Bortoli, V., Mathieu, E., & Hutchinson, M. (2023). Diffusion models for constrained domains. *International Conference on Learning Representations*.** [arXiv](https://arxiv.org/abs/2304.05364)

- Relevance: Diffusion under inequality constraints rather than a smooth known manifold.
- Abstract-level finding: Develops logarithmic-barrier and reflected-Brownian noising processes for constrained domains.
- Contribution: Useful if climate naturalness is operationalized through inequality constraints, such as bounds, balance residuals, or regime-specific admissibility.

### C. Diffusion guidance and constrained inference

**Graikos, A., Malkin, N., Jojic, N., & Samaras, D. (2022). Diffusion models as plug-and-play priors. *Advances in Neural Information Processing Systems*.** [arXiv](https://arxiv.org/abs/2206.09012)

- Relevance: Fixed diffusion prior combined with a differentiable auxiliary constraint.
- Abstract-level finding: Iteratively differentiates through denoising networks with multiple noise levels to solve conditional inference/optimization problems.
- Contribution: Closest conceptual precedent to “forecast objective + fixed diffusion prior,” but it does not provide a CNOP-specific monotonicity guarantee.

**Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). Diffusion posterior sampling for general noisy inverse problems. *International Conference on Learning Representations*.** [arXiv](https://arxiv.org/abs/2209.14687)

- Relevance: Explicitly discusses blending diffusion sampling with a manifold-constrained gradient.
- Abstract-level finding: Extends diffusion inverse-problem solvers to noisy nonlinear settings and avoids a strict measurement-consistency projection in favor of an approximate posterior path.
- Contribution: Directly supports your observation that post-hoc projection changes the optimization problem; the resulting path must be treated as a composite stochastic objective.

**Bansal, A., Borgnia, E., Chu, H.-M., et al. (2023). Universal guidance for diffusion models. *International Conference on Learning Representations*.** [arXiv](https://arxiv.org/abs/2302.07121)

- Relevance: General-purpose gradient guidance without retraining the diffusion model.
- Contribution: A possible implementation reference for coupling WalkerNet’s differentiable forecast objective to a frozen diffusion prior.

### D. Climate and Earth-system generative models

**Bassetti, S., Hutchinson, B., Tebaldi, C., & Kravitz, B. (2023). DiffESM: Conditional emulation of temperature and precipitation in Earth system models with diffusion models.** [arXiv](https://arxiv.org/abs/2304.11699)

- Relevance: Early climate-specific diffusion emulator.
- Abstract-level finding: Conditional diffusion maps monthly ESM averages to daily temperature/precipitation while matching selected temporal/spatial statistics and extremes.
- Contribution: Demonstrates that a diffusion prior can represent climate-like temporal statistics, but the variables and temporal resolution differ from WalkerNet’s coupled monthly fields.

**Christensen, K., Otto, L., Bassetti, S., Tebaldi, C., & Hutchinson, B. (2024). Diffusion-based joint temperature and precipitation emulation of Earth system models.** [arXiv](https://arxiv.org/abs/2404.08797)

- Relevance: Joint multi-variable climate generation.
- Abstract-level finding: Joint diffusion generation can preserve interactions between temperature and precipitation metrics better than independently generating variables.
- Contribution: Supports modeling TOS/ZOS/TAUU/TAUV jointly instead of applying separate variable-wise naturalness filters.

**Brenowitz, N. D., Ge, T., Subramaniam, A., et al. (2025). Climate in a bottle: Towards a generative foundation model for the kilometer-scale global atmosphere.** [arXiv](https://arxiv.org/abs/2505.06474)

- Relevance: Large-scale climate generative model with guided sampling.
- Abstract-level finding: cBottle samples global atmospheric states with a diffusion framework and uses guided diffusion to produce physically credible targeted samples, including tropical-cyclone guidance.
- Contribution: Shows that a climate generator can be used interactively, but “guided” does not automatically mean that the generated state is a valid perturbation relative to a specific initial state.

**Bouabid, S., Souza, A. N., & Ferrari, R. (2026). Score-based generative emulation of impact-relevant Earth system model outputs. *Journal of Advances in Modeling Earth Systems*.** [DOI](https://doi.org/10.1029/2025MS005558)

- Relevance: Recent score-based ESM emulation with a broad diagnostic suite.
- Abstract-level finding: The model evaluates probability densities, cross-variable correlations, time of emergence, and tail behavior across multiple ESMs and regimes; it reports failure cases under strong seasonal regime shifts.
- Contribution: Strong template for the proposed naturalness audit and a warning against using one scalar score.

**Brenowitz, N. D., et al. (2025). Climate in a bottle: Towards a generative foundation model for the kilometer-scale global atmosphere.** [arXiv](https://arxiv.org/abs/2505.06474) **and Price, I., et al. (2024). Probabilistic weather forecasting with machine learning. *Nature*.** [GenCast DOI](https://doi.org/10.1038/s41586-024-08252-9)

- Relevance: Demonstrates the maturity of probabilistic diffusion weather forecasting and ensemble state generation.
- Contribution: Motivates using generative models to represent a distribution of plausible futures, while also emphasizing calibration and ensemble reliability rather than one “canonical” state.

### E. Physics-informed generation and perturbation plausibility

**Bastek, J.-H., Sun, W., & Kochmann, D. M. (2024). Physics-informed diffusion models.** [arXiv](https://arxiv.org/abs/2403.14404)

- Relevance: Adds first-principles residuals and equality/inequality constraints to diffusion training/generation.
- Abstract-level finding: Physics-based loss terms reduce residual errors in a fluid-flow case study and act as a regularizer.
- Contribution: Suggests augmenting statistical naturalness with conservation/balance diagnostics, but direct ocean–atmosphere implementation remains open.

**Rahman, M. M., et al. (2022). Understanding adversarial robustness against on-manifold adversarial examples. *Pattern Recognition*.** [arXiv](https://arxiv.org/abs/2210.00430)

- Relevance: Direct evidence that on-manifold perturbations can remain effective and should not be treated as harmless by default.
- Abstract-level finding: On-manifold adversarial examples can achieve substantial attack rates; the true manifold is unknown and approximated using real/synthetic data.
- Contribution: Supports evaluating “natural” perturbations for impact, not assuming manifold restriction makes the perturbation scientifically irrelevant.

## Preliminary synthesis

### What the literature supports

1. CNOP has a well-established nonlinear optimal-precursor role in atmosphere–ocean science, especially ENSO, and is explicitly defined under a physical constraint.
2. Score/diffusion models provide differentiable fields or sampling procedures that can encode data-supported state structure; manifold detection theory makes the distinction between support and probability important.
3. Diffusion guidance and posterior-sampling work already combine a learned prior with a differentiable objective, but those methods generally target inference or conditional sampling, not a CNOP forecast-response maximum with a stability guarantee.
4. Climate/ESM diffusion work demonstrates that naturalness must be evaluated with joint distributions, temporal structure, regimes, extremes, and forced responses; it also reports failure cases.
5. Physics-informed diffusion indicates that purely statistical support can be augmented with explicit physical residuals.

### What is not yet established by this search

- No directly matching paper was found in the targeted search that formulates CNOP for a learned climate emulator with a differentiable learned-manifold or latent-space constraint.
- No source found here proves that post-hoc denoising preserves or increases a CNOP objective.
- No source found here validates “diffusion likelihood = physically possible Earth state.”
- No source found here establishes a universal climate-state manifold across CMIP6 models, historical/reanalysis data, seasons, and ENSO regimes.

These are search-bounded gap statements, not proof of absolute novelty. Phase 2 must continue with citation chaining, full-text screening, and a more exhaustive search before the manuscript uses “first” or “novel.”

## Implications for the proposed method

The strongest near-term route is not a hard post-hoc projection. The literature instead points toward a composite optimization/inference formulation:

\[
\max_{z}
J\!\left(\mathcal{F}(G_\theta(z,c))\right)
\;-
\lambda_{\mathrm{nat}}R_{\mathrm{nat}}(G_\theta(z,c))
\;-
\lambda_{\mathrm{amp}}R_{\mathrm{amp}}(z),
\]

with explicit held-out naturalness evaluation and, if possible, a conditional generator around the observed history. The first implementation should compare:

1. raw amplitude-constrained CNOP;
2. post-hoc gradient step + diffusion denoising;
3. differentiable latent-space CNOP through a frozen generator;
4. latent-space CNOP plus score/transition naturalness regularization;
5. an optional physics-residual term.

The key claim should initially be **optimization on a data-supported climate-state prior**, not “optimization on the true Earth manifold.”

## Phase-2 limitations and next step

- Several sources were verified only from abstracts or official metadata; full-text methods and supplementary details remain to be screened.
- Search coverage is strong for the six planned clusters but not yet exhaustive across all climate ML venues, Chinese literature, and unpublished CNOP implementations.
- The next required step is source-by-source full-text verification and a contradiction/feasibility matrix, followed by Phase 3 synthesis and a devil’s-advocate checkpoint.
