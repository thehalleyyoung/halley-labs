# Finite-Width Phase Diagrams for Neural Network Training Dynamics

## Problem Statement

No existing tool answers the most basic predictive question about neural network training: given a concrete architecture and dataset, in which *dynamical regime* will training operate? Will the network remain in the lazy/kernel regime, or will it transition to feature learning? At what learning rate does this transition occur, and how does it depend on width, depth, and data structure? Three mature theoretical frameworks — Neural Tangent Kernel theory (Jacot et al., 2018), mean-field limits (Mei, Montanari & Nguyen, 2018), and maximal update parameterization (Yang & Hu, 2021) — each illuminate one infinite-width limit of one regime. But practitioners train finite-width networks, and no system stitches these fragments together at finite width to produce falsifiable, experimentally validated phase diagrams. We build that system: **the first integrated system that computes finite-width phase diagrams predicting the lazy-to-rich transition boundary, validated experimentally against ground-truth training runs.**

The primary theoretical contribution is the derivation and numerical computation of 1/N corrections to the Neural Tangent Kernel for convolutional and residual architectures — extending the perturbative kernel evolution results of Dyer & Gur-Ari (2020) and Huang & Yau (2020) from vanilla MLPs to weight-shared and skip-connected architectures. This requires new derivations of the Hessian-Jacobian contraction tensor H_{ijk} for architectures where weight sharing introduces spatial index structure (ConvNets) and skip connections introduce additive Jacobian composition (ResNets). The resulting coupled matrix-valued ODE system's linearized eigenvalue analysis locates phase boundaries via bifurcation detection. The lazy-to-rich transition is controlled by a dimensionless coupling parameter γ = η · N^{-(1−a−b)}, where a, b are µP scaling exponents. The phase boundary in (γ, depth, data complexity) space is computed by finding where eigenvalues of the linearized dynamics cross stability thresholds — a concrete, computable bifurcation problem.

The system is organized as a compiler pipeline: an architecture specification is parsed into a computation-graph intermediate representation, then lowered through three parallel backends — a symbolic kernel engine (NTK + 1/N corrections), a mean-field baseline (precomputed 2–3 layer solutions with interpolation), and a tensor program analyzer (µP scaling signatures). A dataset statistics module computes kernel-task spectral alignment via Nyström-approximated Gram matrices. These backend outputs feed a phase diagram mapper that sweeps hyperparameter axes and classifies each point into dynamical regimes (lazy, rich, chaotic, divergent) using bifurcation detection and critical-exponent estimation. A lightweight uncertainty quantification module tracks perturbative validity, moment-closure quality, and Nyström error propagation, flagging predictions where the system operates outside its reliable regime.

We are honest about scope. The system operates on subsampled datasets (n ≈ 2K–5K via Nyström approximation). Mean-field baselines use precomputed solutions for 2–3 layer networks; deeper networks use empirical depth corrections without theoretical justification for normalized architectures. All finite-width corrections are approximate: moment-closure truncation at 4th-order cumulants introduces systematic bias that is tracked via uncertainty quantification — predictions where |Θ⁽¹⁾/Θ⁽⁰⁾| > 0.5 or κ₄ magnitude exceeds calibrated thresholds are flagged as low-confidence. One-loop vs. two-loop convergence is checked at each target width; claims are restricted to widths where the expansion converges. The system does not handle transformer architectures (softmax attention breaks NTK and mean-field analyses). The prediction target is the phase diagram — regime classification and boundary location — not exact loss values.

## Value Proposition

**Theoretical ML researchers studying training dynamics** are the primary audience. The NTK, mean-field, and µP communities each work with pen-and-paper calculations or bespoke one-off scripts. Deriving finite-width corrections for a new architecture takes weeks to months of manual calculation. A unified computational framework that evaluates finite-width kernel evolution ODEs, provides mean-field baselines, and extracts µP scaling exponents from architecture graphs gives theorists a laboratory for testing conjectures, computing phase diagrams for new architectures, and generating the quantitative predictions that connect theory to experiment. The 1/N corrections for convolutional and residual architectures are themselves publishable mathematical contributions that emerge from building the system.

**ML practitioners designing hyperparameter sweeps** for non-transformer architectures (ConvNets, ResNets, MLPs for tabular/scientific computing) can use phase diagrams as pre-training triage. Before committing GPU resources to a sweep over (learning rate, width, initialization scale), a phase diagram computed in minutes on CPU identifies which regions of hyperparameter space are categorically wasteful (deep in the lazy regime, where depth is wasted) and which are near phase boundaries (where training outcomes are sensitive to initialization and small perturbations cause qualitative failure). This enables two concrete decision workflows:

1. **Sweep triage:** Eliminate hyperparameter regions classified as deep-lazy with high confidence, reducing the sweep volume by an estimated 30–50% for architectures with wide lazy regimes.
2. **Transfer validation:** When µTransfer recommends transferring a learning rate from width 256 to width 4096, the phase diagram predicts whether that transfer crosses a phase boundary — which would invalidate the transfer and cause silent performance degradation.

**ML safety researchers** can use phase boundary proximity as a leading indicator of training instability. Qualitative regime changes during training (when a learning rate schedule crosses a phase boundary) manifest as sudden loss spikes and behavioral shifts. The system provides advance warning of such crossings.

## Technical Difficulty

The core system comprises ~65K lines of code across 7 essential subsystems. The complexity is multiplicative: the kernel engine and finite-width corrections must handle 4 architecture types (dense, convolutional, residual, combined) with distinct mathematical structure, and the phase mapper must handle multi-dimensional hyperparameter spaces with non-trivial bifurcation topology.

**Computation Graph IR + Lowering Passes (15K LoC).** The integration linchpin. The NTK lowering pass computes per-layer kernel recursion; the µP lowering pass propagates scaling exponents; the mean-field pass indexes precomputed solutions. A ResidualBlock must simultaneously express skip-connection topology (for NTK kernel composition), depth indexing (for mean-field baseline lookup), and parameter scaling class (for µP). No existing IR handles this multi-theory annotation requirement.

**Symbolic Kernel Engine (20K LoC).** Per-layer NTK and NNGP kernel composition for dense, convolutional, and residual layers across multiple nonlinearities. Convolutional NTK requires numerical integration over patch structure with no closed-form solution — Neural Tangents does not handle this at finite width. Lazy evaluation and caching for O(n²) kernel matrices under memory constraints.

**Finite-Width Correction Engine (18K LoC).** The most mathematically novel subsystem. Symbolic higher-order cumulant computation for structured architectures, moment-closure truncation with systematic bias tracking, and uncertainty quantification. Extending Dyer & Gur-Ari's 1/N expansion beyond MLPs to convolutional and residual architectures requires re-deriving the cumulant structure for each architecture type. Includes perturbative validity checks (|Θ⁽¹⁾/Θ⁽⁰⁾| monitoring), moment-closure quality indicators (κ₄ magnitude tracking), and optional two-loop convergence verification. No existing library handles finite-width kernel evolution.

**Phase Diagram Mapper (10K LoC).** Hyperparameter sweeps with adaptive boundary refinement, eigenvalue tracking for bifurcation detection, and numerical continuation (pseudo-arclength methods) along phase boundaries. Confidence-stratified output: predictions are labeled high-confidence, moderate, or near-boundary based on eigenvalue gap magnitude and UQ flags.

**Remaining subsystems:** Architecture Parser (4K), Evaluation Harness (5K), Core Testing (8K). The mean-field backend uses precomputed solutions for benchmark architectures rather than a full ODE solver; µP scaling exponents are provided for benchmark architectures with automated inference as a future extension. Approximately 60% of the core codebase is built from scratch because the central contributions have no existing implementations.

## New Mathematics Required

Approximately 70% of the mathematics is careful implementation of known results stitched into a unified computational framework. The remaining 30% — where genuine novelty and risk concentrate — comprises:

**Kernel ODE System with 1/N Corrections.** The finite-width NTK evolves as Θ(t) = Θ^(0) + (1/N)Θ^(1)(t) + O(1/N²), where the first-order correction satisfies dΘ^(1)_{ij}/dt = −η Σ_k r_k(t) H_{ijk}(t). The Hessian-Jacobian contraction tensor H_{ijk} decomposes into per-layer contributions scaling as 1/N_ℓ. For MLPs, this is known (Dyer & Gur-Ari, 2020). **For convolutional architectures** (where weight sharing introduces spatial index structure in H_{ijk}) **and residual architectures** (where skip connections introduce additive composition in the Jacobian product), the derivation is genuinely new. The perturbation hierarchy couples each 1/N order to the next cumulant: order 1/N^0 is Gaussian (frozen NTK), order 1/N requires 4th cumulants κ_4. We truncate at one-loop (1/N) with mandatory convergence verification against two-loop results at each target width.

**Moment Closure with Uncertainty Quantification.** The 4th cumulant κ_4 closing the 1/N hierarchy is approximated via Gaussian closure (κ_4 = 0) as the default. This is an approximation ansatz, not a theorem. The systematic bias is quantified via: (a) sensitivity analysis — perturbing κ_4 by ±50% and measuring phase boundary displacement, (b) perturbative validity monitoring — flagging predictions where the correction term exceeds 50% of the leading order, (c) moment-closure quality indicators — tracking κ_4 magnitude as a proxy for non-Gaussianity. Predictions where these indicators exceed calibrated thresholds are reported as low-confidence. The system aims to correctly identify >90% of its own prediction failures as low-confidence.

**Tensor Program Algebra.** Each weight matrix carries scaling exponents (a, b, c) governing initialization variance, learning rate, and output contribution. These propagate through the architecture DAG via composition rules for MatMul, nonlinearity, aggregation, and skip connections. The effective coupling γ = η · N^{-(1−a−b)} determines whether the 1/N expansion is well-ordered: under µP (a=1, b=1), γ = η is O(1), making the expansion convergent; under standard parameterization (a=1, b=0), the expansion requires η → 0 with N, forcing the lazy regime. This is implementing Yang's framework with µP exponents provided for benchmark architectures.

**Phase Boundary Computation.** The lazy-to-rich transition is formulated as an eigenvalue-crossing problem: linearize the kernel ODE around the NTK fixed point, and find the critical γ_c where max Re(λ_i(J(γ_c))) = 0. This is generically a transcritical bifurcation. Computing phase boundaries via numerical continuation (pseudo-arclength methods) in (γ, depth, data-complexity) space is the novel methodological contribution. Hopf bifurcations (oscillatory instability) may arise at large learning rates with discrete updates, connecting to edge-of-stability phenomena.

## Best Paper Argument

This work merits best-paper consideration for four reasons.

**Novel mathematical contribution with immediate computational realization.** The primary contribution — 1/N corrections to the convolutional and residual NTK, yielding the Hessian-Jacobian contraction tensor for weight-shared and skip-connected architectures — is genuinely new mathematics. No prior work computes these objects. The system makes these derivations immediately usable: input an architecture, get finite-width phase predictions. Best papers in ML rarely achieve the combination of new theory and a working computational tool.

**Sharp falsifiable claims with honest uncertainty.** The core claims are: (a) phase boundary localization within 15% relative error for >80% of configurations, with AUC > 0.90 on lazy-vs-rich classification, (b) outperformance of both infinite-width NTK baseline and non-theoretical heuristic baseline by ≥15 percentage points, (c) self-calibrated uncertainty with >90% of prediction failures flagged as low-confidence. These are concrete, quantitative, and experimentally testable. The claims are bold enough to matter and honest enough to be credible — including explicit reporting of where the system fails and why.

**Validated by retrodiction and prediction.** The system is validated in two complementary modes: (a) retrodiction — recovering analytically known phase boundaries for 2-layer ReLU networks (Chizat & Bach, 2019) and linear networks (Saxe et al., 2014), demonstrating the system agrees with exact results; (b) prediction — computing phase diagrams for architectures where no analytic solution exists (ConvNets, ResNets) and validating against ground-truth training runs. This dual validation breaks the circular dependency between theory and experiment.

**Unification revealing new structure.** By computing phase diagrams across architecture families with the same mathematical framework, the system enables direct comparison of how architecture determines phase structure. Architecture-dependent critical exponents (how phase boundaries scale with width) and phase boundary sharpness (width of the crossover region as a function of N) are novel theoretical predictions that emerge from the computation without additional derivation.

## Evaluation Plan

All evaluation is fully automated — code running against code, zero human involvement.

**Primary metric: Phase boundary prediction accuracy.** For each architecture family (MLP, ConvNet, ResNet), sweep a grid over (learning rate, width, initialization scale). At each point, the system predicts lazy or rich regime with a confidence level; ground-truth training runs (20 seeds for boundary-adjacent configurations, 5 seeds elsewhere) classify the actual regime using a multi-criteria conjunction: kernel alignment drift, representation similarity, weight displacement norm, and linear probing accuracy gap. A configuration is classified as "rich" only if ≥3/4 indicators agree. Report AUC (target: >0.90), boundary localization error (target: <15% relative error for >80% of configurations), and confidence-stratified accuracy (separately for high-confidence, moderate, and near-boundary predictions).

**Ablation studies (mandatory):**
1. **1/N corrections ON vs. OFF.** Phase classification accuracy with and without finite-width corrections, isolating the core contribution's value.
2. **Single-backend vs. multi-backend.** Accuracy by regime for NTK-only, mean-field-only, µP-only, and fused prediction.
3. **Nyström rank sensitivity.** Prediction accuracy vs. compute time across ranks {100, 500, 1000, 2000}.
4. **Moment-closure sensitivity.** Phase boundary displacement under ±50% κ₄ perturbation.

**Retrodiction validation:** Recover analytically known phase boundaries for (a) 2-layer ReLU MLP (Chizat & Bach, 2019), (b) linear networks (Saxe et al., 2014), (c) µP-parameterized networks (Yang & Hu, 2021 critical exponents). Demonstrate convergence to known results as a function of N.

**Convergence verification:** Report one-loop vs. two-loop phase boundary agreement at all target widths {64, 128, 256, 512, 1024}. If disagreement exceeds grid spacing at N ≤ 256, restrict all claims to widths where convergence is demonstrated.

**Baselines:** (1) Infinite-width NTK via Neural Tangents (predicts everything is lazy), (2) µTransfer (predicts scaling and regime classification), (3) Non-theoretical heuristic (predict rich if η·N^{1-a-b} > C with fitted C), (4) Bordelon & Pehlevan (2024) spectral scaling predictions, (5) Width-interpolation heuristic (train at N₁, N₂, linearly interpolate boundary).

**Secondary metric: Scaling exponent prediction.** For architectures under µP, predict the width-scaling exponent of optimal learning rate. Compare predicted exponents against measured scaling laws from training runs across widths {64, 128, 256, 512, 1024}. Report relative error.

**Tertiary metric: Hyperparameter transfer error.** Transfer learning rate from width 256 to width 1024 using (a) phase-diagram-informed prediction, (b) µTransfer, (c) heuristic baseline. Measure final loss gap relative to grid-search-optimal.

**Novel predictions:** Report (a) phase boundary sharpness (crossover region width as a function of N) and (b) architecture-dependent critical exponents across MLP, ConvNet, and ResNet families. If these differ across architectures, this constitutes a novel theoretical prediction connecting neural network training dynamics to statistical-mechanical universality classes.

**Benchmarks:** MNIST (n=2K subsample), Fashion-MNIST (n=2K), CIFAR-10 (n=2K, with PCA to 100 dimensions for CPU feasibility), and two tabular regression datasets. All datasets are subsampled and preprocessed offline. Architectures: 2–4 layer MLPs (widths 64–1024), 3-layer ConvNets, shallow ResNets.

**UQ self-evaluation:** Report the fraction of prediction failures that the UQ module flags as low-confidence. Target: >90% of failures are flagged, demonstrating calibrated self-knowledge.

## Laptop CPU Feasibility

The system runs entirely on CPU because it never trains a neural network — it solves equations *about* neural network training.

**All computation is ODE/PDE solving + linear algebra.** The kernel evolution ODE operates on n×n matrices where n is the dataset subsample size (2K–5K), not the parameter count. A single ODE step on a 2K×2K kernel matrix involves O(n³) ≈ 8 GFLOP — under one second on a modern laptop CPU. The full phase diagram requires ~100–1000 such solves across the hyperparameter grid, totaling minutes to tens of minutes.

**Key trick: kernel-space, not parameter-space.** The 1/N expansion reformulates parameter-space dynamics (millions of parameters) as kernel-space dynamics (thousands of data points). This is not an approximation — it is the exact mathematical structure of the perturbative expansion. The Nyström approximation further reduces large datasets to m ≈ 500–2000 landmarks with controlled O(1/m) error.

**Dataset subsampling for evaluation.** Ground-truth training runs use small networks (width ≤ 1024, depth ≤ 20) on subsampled data, completing in minutes on CPU via PyTorch's CPU backend. The evaluation harness runs 20-seed training for boundary-adjacent grid points and 5-seed training elsewhere, parallelizable across cores. Estimated total wall time: 12–24 hours on an 8-core laptop for full evaluation of one architecture family.

**No human annotation, no external services.** The entire pipeline — from architecture specification through phase diagram computation through evaluation against ground truth — is a single automated script. Input: architecture + dataset. Output: phase diagram + accuracy metrics + UQ indicators + ablation results.

## Slug

`finite-width-phase-diagrams`
