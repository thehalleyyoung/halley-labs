# Final Approach: Calibrated Kernel Observatory with Bifurcation-Theoretic Phase Boundaries

## Overview

We build the first integrated system that computes finite-width phase diagrams predicting the lazy-to-rich transition boundary for neural network training, validated experimentally against ground-truth training runs. The system is anchored in Approach B's empirical calibration infrastructure — the safest engineering foundation (feasibility 8/10, math success ~70%) — and enhanced with Approach A's spectral bifurcation theory for phase boundary detection and the H_{ijk} derivation for structured architectures. Approach C's RG framing is dropped entirely as a computational method; it adds cost without adding capability (debate consensus: ~70% decorative math, ~25% success probability, dual-paradigm architecture guarantees scope creep).

The core technical pipeline is: compute exact NTKs at 5+ calibration widths via autodiff, extract finite-width corrections Θ^(1) by regression, feed these into a linearized kernel ODE whose eigenvalue crossings locate phase boundaries via bifurcation detection, and track those boundaries through hyperparameter space via pseudo-arclength continuation. The system explicitly addresses the crossover-vs-transition problem — at finite width, the lazy-to-rich "transition" is a smooth crossover, and we define the "phase boundary" operationally as the locus of steepest gradient of a continuous order parameter (kernel alignment drift rate), not as a sharp threshold. This definition is non-circular, threshold-free, and falsifiable.

The project delivers three tiers of results with decreasing probability: (1) a working phase diagram tool with empirically calibrated corrections (~70% probability), (2) analytically derived H_{ijk} for MLP and 1D ConvNet architectures with bifurcation analysis (~35% probability), and (3) evidence for architecture-dependent critical exponents — the killer prediction that survived debate scrutiny as genuine novel math (~10% probability, but best-paper-caliber if achieved).

## Extreme Value Delivered

**Theoretical ML researchers studying training dynamics** (NTK, mean-field, µP communities) get a computational laboratory. Currently, deriving finite-width corrections for a new architecture takes weeks of manual calculation with no way to verify convergence. This system evaluates finite-width kernel evolution ODEs, provides mean-field baselines, and extracts µP scaling exponents from architecture graphs — letting theorists test conjectures computationally rather than algebraically. The 1/N corrections for convolutional and residual architectures are themselves publishable mathematical contributions.

**ML practitioners training non-transformer architectures** (ConvNets for medical imaging, ResNets for autonomous driving, MLPs for scientific computing/tabular data) get pre-training regime diagnostics. Before committing GPU resources to a hyperparameter sweep, a phase diagram computed in minutes on CPU identifies which regions are categorically wasteful (deep in lazy regime, where depth is wasted) and which are near phase boundaries (where training is sensitive to initialization). Concrete workflow: sweep triage eliminates 30–50% of hyperparameter grid volume; µTransfer validation detects when transferring a learning rate across widths crosses a phase boundary.

**ML safety researchers** get a leading indicator of training instability. Phase boundary proximity predicts when a learning rate schedule will trigger qualitative regime changes (sudden loss spikes, behavioral shifts) before they occur.

**Honest scoping of audience:** We acknowledge the transformer exclusion limits practical audience. The honest framing is that this is a theoretical contribution validated on classical architectures with practical applicability to the substantial (but shrinking) non-transformer deployment base. We do not claim "$10K–$100K compute savings" without production-scale validation, which is outside project scope.

## Technical Architecture

The system is organized as a compiler pipeline with four subsystems:

### Subsystem 1: Architecture Parser + Computation Graph IR (~12K LoC)
Architecture specification → computation-graph intermediate representation. The IR annotates each node with: (a) kernel recursion type (for NTK computation), (b) µP scaling exponents (a, b, c), (c) weight-sharing structure (for convolutional layers), (d) skip-connection topology (for residual blocks). This IR is shared by all downstream computation.

### Subsystem 2: Kernel Engine with Finite-Width Corrections (~20K LoC)
Two parallel computation paths, both feeding the same phase mapper:

**Path A — Empirical calibration (primary, always available):** Compute exact NTK via autodiff at calibration widths N_cal = {32, 64, 128, 256, 512} (5 points, giving 2 degrees of freedom for the 3-parameter 1/N² model). Extract Θ^(1) by weighted least squares regression. Bootstrap confidence intervals from resampling over initialization seeds. Uncertainty in Θ^(1) propagates analytically to uncertainty in phase boundary location.

**Path B — Analytic H_{ijk} (enhancement, architecture-dependent):** For architectures where the Hessian-Jacobian contraction tensor can be derived (MLPs: known; 1D ConvNets: novel derivation; 2D ConvNets: stretch goal), compute Θ^(1) directly from the tensor structure. Validated against Path A at overlapping widths. When Path B is available and validated, it replaces Path A for that architecture class, providing faster computation and theoretical insight.

Both paths use Nyström approximation (rank m = 500–2000) for datasets exceeding n = 2K, with adaptive rank selection that increases m near detected phase boundaries where eigenvalue gaps shrink. Nyström error bound: |λ_i - λ̂_i| ≤ ||K - K_m||₂ / δ_i.

### Subsystem 3: Phase Diagram Mapper (~10K LoC)
Takes corrected kernel evolution ODE and maps phase boundaries in (γ, depth, data-complexity) space. Core algorithm:
1. Coarse grid sweep to identify approximate boundary locations
2. Bifurcation detection: linearize kernel ODE around NTK fixed point, find γ_c where max Re(λ_i) crosses zero
3. Pseudo-arclength continuation to track boundary curves with error-controlled step size
4. Confidence stratification: high-confidence (perturbative validity V < 0.2), moderate (0.2 ≤ V < 0.5), low-confidence (V ≥ 0.5)

The phase boundary is defined as the **steepest gradient locus** of kernel alignment drift rate (continuous order parameter), not a threshold crossing. This addresses the crossover-vs-transition problem head-on.

### Subsystem 4: Evaluation Harness (~8K LoC)
Ground-truth regime classification uses a continuous order parameter (kernel alignment drift rate) rather than binary classification. The "ground truth phase boundary" is the steepest-gradient locus of the same order parameter measured from actual training runs, making predicted and ground-truth boundaries directly comparable without circularity.

**Data flow:** Architecture spec → IR → Kernel Engine (Path A and/or B) → Corrected kernel ODE → Phase Mapper → Phase diagram + UQ + confidence stratification → Evaluation against ground-truth training runs.

### Key Design Decisions
- **No SymPy dependency.** H_{ijk} for structured architectures is computed via numerical differentiation (finite differences against brute-force exact kernels at small N), not symbolic algebra. SymPy is a critical single point of failure (notoriously slow for large expressions, no drop-in replacement). Numerical differentiation is validated against analytic results for MLPs where the answer is known.
- **No dual-paradigm architecture.** One computational backend (kernel ODE with empirical/analytic corrections). RG interpretation is relegated to paper discussion section, not implemented as a computational path.
- **5+ calibration widths, not 3.** The 3-point calibration flaw (0 degrees of freedom) was a fatal flaw identified in debate. Using N_cal = {32, 64, 128, 256, 512} gives 2 DOF for the 3-parameter model, enabling genuine error estimation and residual analysis.

## Core Mathematical Contributions

Only load-bearing math that survived adversarial debate scrutiny:

### 1. Empirically Calibrated Finite-Width Corrections (from Approach B)
The finite-width NTK evolves as Θ(N) = Θ^(0) + Θ^(1)/N + Θ^(2)/N². The regression estimator for Θ^(1) from calibration widths N_1, ..., N_J:

$$\hat{\Theta}^{(1)} = \arg\min_{\Theta^{(0)}, \Theta^{(1)}, \Theta^{(2)}} \sum_{j=1}^{J} w_j \left\| \Theta(N_j) - \Theta^{(0)} - \frac{\Theta^{(1)}}{N_j} - \frac{\Theta^{(2)}}{N_j^2} \right\|_F^2$$

with weights w_j and J ≥ 5 calibration widths. Bootstrap over initialization seeds provides confidence intervals. **Error propagation from Θ^(1) estimation to phase boundary uncertainty** is derived analytically via implicit function theorem applied to the bifurcation condition, filling the key mathematical gap identified in debate.

### 2. Spectral Bifurcation Theory for Kernel ODEs (from Approach A)
The linearized kernel evolution around the NTK fixed point defines a linear operator L(γ) on symmetric n×n matrices. The lazy-to-rich transition occurs at γ_c where the spectral abscissa s(L(γ_c)) = max Re(λ_i(L(γ_c))) crosses zero. This is a concrete, computable bifurcation problem. The bifurcation is generically transcritical; the eigenvalue crossing direction determines the nature of the transition.

This is the genuine mathematical contribution that Approach A provides over Approach B's pure engineering. It transforms phase boundary detection from a threshold-dependent heuristic into a threshold-free eigenvalue problem.

### 3. H_{ijk} for Convolutional Architectures (from Approach A, scope-reduced)
For convolutional layers with weight sharing, the Hessian-Jacobian contraction tensor H_{ijk} has spatial index structure. The key factorization hypothesis:

$$H^{(\text{conv})}_{ijk} = H^{(\text{dense})}_{ijk} \otimes K^{(\text{patch})}_{pp'}$$

where K^(patch) is the patch-overlap Gram matrix. This is validated numerically (not symbolically) by:
1. Derive for 1D convolutions first
2. Compare against brute-force finite-difference computation at N = {8, 16, 32}
3. Accept if agreement < 5% relative error; abandon gracefully if factorization fails

If the factorization holds, per-step ODE cost is O(n³ · rank(K^patch)) instead of O(n³S²). If it fails, we fall back to empirical calibration (Path A), which always works. This is a genuine research contribution with a well-defined fallback — not an all-or-nothing bet on the overall system.

### 4. Pseudo-Arclength Continuation with Error Control (from Approach B)
Phase boundaries tracked as curves in hyperparameter space using predictor-corrector continuation with adaptive step size based on eigenvalue gap preservation. This is known numerical methods applied cleanly; the contribution is the integration with bifurcation detection and the error-controlled step size that prevents jumping across nearby bifurcation curves.

### What is NOT a mathematical contribution
- The "Bayesian fusion" of four indicators is logistic regression with 4 features. We call it logistic regression.
- The perturbative validity functional V[Θ] is a useful diagnostic, not a mathematical contribution.
- Universality classes and critical exponents are observational predictions that may emerge from the computation, not core deliverables. They are reported if observed but not promised.
- The RG framework is an interpretive lens for the discussion section, not load-bearing math.

## Implementation Strategy

### Phase 1: Foundation (Weeks 1–3)
**Build:** Architecture parser, computation graph IR, exact NTK computation via JAX autodiff, basic kernel evaluation for dense layers.
**Validate:** Reproduce known infinite-width NTK for 2-layer ReLU MLP against Neural Tangents library.
**Dependencies:** JAX, NumPy, SciPy. No SymPy.

### Phase 2: Calibration Pipeline (Weeks 3–5)
**Build:** 5-width calibration pipeline, weighted least squares regression for Θ^(1), bootstrap confidence intervals, Nyström approximation with adaptive rank selection.
**Validate:** Verify that extracted Θ^(1) for MLPs matches known analytic results (Dyer & Gur-Ari, 2020) within 5% relative error.
**Key decision point:** If calibration pipeline cannot extract Θ^(1) reliably for MLPs, the entire project needs re-scoping.

### Phase 3: Phase Boundary Detection (Weeks 5–7)
**Build:** Linearized kernel ODE, eigenvalue tracking, bifurcation detection, pseudo-arclength continuation, phase diagram mapper.
**Validate:** Recover analytically known phase boundaries for 2-layer ReLU networks (Chizat & Bach, 2019) and linear networks (Saxe et al., 2014). This is the retrodiction test.

### Phase 4: Convolutional Extension (Weeks 7–9)
**Build:** Kernel computation for convolutional layers, H_{ijk} derivation for 1D convolutions (numerical, not symbolic), convolutional NTK evaluation.
**Validate:** H_{ijk} factorization test at small N = {8, 16, 32}. If factorization fails, mark convolutional architectures as "empirical calibration only" and proceed.

### Phase 5: Evaluation and Hardening (Weeks 9–11)
**Build:** Ground-truth training harness, continuous order parameter measurement (kernel alignment drift rate), evaluation metrics, ablation framework.
**Validate:** Full evaluation against ground-truth training runs for MLPs and (if Phase 4 succeeded) ConvNets.

### Phase 6: Stretch Goals (Weeks 11–12)
**If capacity allows:** Residual architecture support, 2D convolutional H_{ijk}, architecture-dependent critical exponent measurement. These are bonus contributions, not core deliverables.

### Parallelization Opportunities
- Phases 1–2 are sequential (foundation must exist before calibration)
- Phase 3 (bifurcation detection) and Phase 4 (ConvNet extension) can be partially parallelized
- Phase 5 ground-truth training runs can start as soon as Phase 3 produces MLP phase diagrams

## Hardest Challenges and Mitigations

### Challenge 1: H_{ijk} Factorization for Convolutions May Not Hold
**Risk:** The factorization H^(conv)_{ijk} = H^(dense)_{ijk} ⊗ K^(patch)_{pp'} is a hypothesis, not a theorem. Weight sharing introduces non-trivial correlations in higher cumulants that may not factor cleanly.
**Mitigation:** (a) Test numerically at small scale (N=8, 16, 32) before committing to full implementation. (b) Start with 1D convolutions where spatial structure is simpler. (c) If factorization fails, fall back to empirical calibration (Path A), which always works — the system degrades gracefully from "analytically derived corrections" to "numerically calibrated corrections" with no change to the rest of the pipeline. This is a research outcome, not a system failure.
**Probability of challenge materializing:** ~50%.

### Challenge 2: Perturbation Theory Diverges at Phase Boundaries
**Risk:** The 1/N expansion has effective parameter γ/N^{something}, and near phase boundaries γ is O(1). The convergence radius shrinks to zero at transitions — a theorem in statistical mechanics. The system is least reliable precisely where predictions are most valuable.
**Mitigation:** (a) The bifurcation detection finds where the linearized dynamics change stability — it identifies the boundary without requiring accurate perturbation theory *at* the boundary. (b) The perturbative validity functional V[Θ] = ||Θ^(1)||_op / ||Θ^(0)||_op quantifies divergence; predictions where V ≥ 0.5 are flagged as low-confidence. (c) The "steepest gradient" boundary definition is robust to perturbative inaccuracy because it finds the maximum of a continuous function, which is stable under small perturbations. (d) Empirical calibration at 5 widths provides a non-perturbative anchor.
**Probability of challenge materializing:** ~80% (near-certain at the boundary itself, but mitigations keep the system useful).

### Challenge 3: Ground-Truth CPU Budget
**Risk:** Full 20-seed ground-truth evaluation was estimated at ~250 hours, far exceeding the 12–24 hour laptop budget.
**Mitigation:** (a) Reduce to 5 seeds for most grid points, 10 seeds for boundary-adjacent points only. (b) Reduce architecture coverage: full evaluation for MLPs and 1D ConvNets only; 2D ConvNets and ResNets get spot-check validation. (c) Use continuous order parameter (kernel alignment drift rate) which requires fewer seeds than binary classification to achieve stable estimates. (d) Accept that ground-truth evaluation is a multi-day campaign, not a single-session computation. Estimated revised budget: ~60–80 hours (3–4 days on 8-core laptop with overnight runs).
**Probability of challenge materializing:** 100% (this is a known constraint, not a risk).

### Challenge 4: Crossover Width May Be Too Broad for "Phase Boundary" to Be Meaningful
**Risk:** At target widths (64–1024), the lazy-to-rich transition may be so broad that the "phase boundary" spans a factor of 4× in learning rate, making the concept vacuous.
**Mitigation:** (a) The steepest-gradient definition provides a unique, well-defined point even for broad crossovers. (b) Report crossover width as a function of N — this is itself a novel theoretical prediction (how sharp is the transition?). If the crossover is broad, that is a scientifically interesting finding, not a failure. (c) For practical value, report confidence-stratified predictions: "definitely lazy," "crossover region," "definitely rich." The crossover region is where practitioners should invest evaluation effort.
**Probability of challenge materializing:** ~60%.

### Challenge 5: Regression Conditioning at Calibration Widths
**Risk:** Even with 5 calibration widths, the 1/N² model may be poorly conditioned if higher-order terms (1/N³, log(N)/N²) contribute significantly at N=32.
**Mitigation:** (a) Compute condition number of the regression design matrix and report it. (b) If conditioning is poor, constrain Θ^(0) from infinite-width theory (reducing to 2-parameter fit from 5 points = 3 DOF). (c) Alternatively, use leave-one-out cross-validation across calibration widths to detect extrapolation failure. (d) Report residuals at each calibration width; systematic patterns in residuals indicate model misspecification.
**Probability of challenge materializing:** ~30%.

## Evaluation Plan

### Primary Metric: Phase Boundary Localization Error
For each architecture family (MLP, 1D ConvNet, optionally 2D ConvNet):
- Sweep a grid over (learning rate × width × initialization scale)
- At each point, compute the continuous order parameter (kernel alignment drift rate) from ground-truth training runs (5 seeds, 10 near detected boundary)
- Locate the ground-truth boundary as the steepest gradient of the order parameter surface
- Compare predicted boundary location against ground-truth boundary
- **Target:** < 20% relative error for > 75% of boundary segments (revised from original 15%/80% after debate realism check)

### Secondary Metric: Regime Classification AUC
- Binary classification: lazy vs. rich at each grid point
- Ground truth: order parameter above/below median
- **Target:** AUC > 0.85 (revised from 0.90 after acknowledging crossover effects)

### Ablation Studies (Mandatory)
1. **1/N corrections ON vs. OFF:** Phase boundary accuracy with and without finite-width corrections, isolating the core contribution
2. **Calibration widths {32, 64, 128} vs. {32, 64, 128, 256, 512}:** Quantify value of additional calibration points
3. **Nyström rank sensitivity:** Accuracy vs. compute across ranks {100, 500, 1000, 2000}
4. **Analytic H_{ijk} vs. empirical calibration:** For architectures where both are available, compare accuracy and compute cost

### Retrodiction Validation
Recover analytically known results:
- 2-layer ReLU MLP phase boundary (Chizat & Bach, 2019)
- Linear network dynamics (Saxe et al., 2014)
- µP scaling exponents (Yang & Hu, 2021)

### Baselines
1. Infinite-width NTK (predicts everything is lazy) — lower bound
2. µTransfer predictions
3. One-parameter heuristic: predict rich if η·N^{1-a-b} > C with fitted C
4. Width-interpolation: train at N₁, N₂, linearly interpolate boundary

### UQ Self-Evaluation
- Report fraction of prediction failures flagged as low-confidence
- **Target:** >85% of failures flagged (revised from 90% for honesty)
- Also report **coverage rate:** fraction of predictions that are high-confidence
- **Target:** >60% of predictions are high-confidence (a system that says "I don't know" most of the time is useless)

### Benchmarks
- MNIST (n=2K subsample)
- Fashion-MNIST (n=2K)
- CIFAR-10 (n=2K, PCA to 100 dims for CPU feasibility)
- Two tabular regression datasets
- Architectures: 2–4 layer MLPs (widths 64–1024), 1D ConvNets (3-layer), optionally shallow ResNets

### Novel Predictions (Reported if Observed, Not Promised)
- Crossover width as a function of N (how sharp is the transition?)
- Architecture-dependent scaling of phase boundary with depth
- If critical exponents differ across MLP/ConvNet/ResNet, this is reported as a novel finding

## Best Paper Argument

The paper makes three concrete, falsifiable claims that together constitute a best-paper case:

**Claim 1: Finite-width corrections are necessary and sufficient for accurate phase boundary prediction.** The ablation (1/N corrections ON vs. OFF) will show that infinite-width NTK predictions are qualitatively wrong at widths 64–512, while the corrected predictions match ground truth within 20% relative error. This is a sharp, quantitative demonstration that the theoretical corrections matter practically.

**Claim 2: Spectral bifurcation theory provides a threshold-free, non-circular phase boundary definition.** Unlike heuristic regime classification (which depends on arbitrary thresholds), the eigenvalue-crossing formulation gives a unique, principled boundary. The steepest-gradient operational definition of the boundary in the ground truth matches the bifurcation prediction, demonstrating that the theory captures the correct mathematical structure of the transition.

**Claim 3 (stretch): Architecture-dependent critical exponents exist and are computable.** If the H_{ijk} derivation succeeds for ConvNets, and if the extracted critical exponents (how the phase boundary scales with width) differ between MLPs and ConvNets, this is the first demonstration that architecture imposes symmetry constraints on training dynamics with measurable consequences. This connects neural network theory to the deep structure of phase transitions — a genuinely new insight, not a restatement of known results.

**Why this wins over Approach A pure or Approach B pure:** Approach A alone has ~45% math success probability and ~3/10 feasibility — too risky. Approach B alone is "a well-engineered software tool, not a research contribution" (debate verdict). This synthesis gets the reliability of B (70% floor) with the theoretical ambition of A (bifurcation theory, H_{ijk}) scoped to what can actually be delivered.

## Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Fills a real gap for theorists; practical value bounded by transformer exclusion; honest about audience |
| **Difficulty** | 8 | Novel math (H_{ijk} for ConvNets, bifurcation theory) + substantial systems engineering; harder than pure B, easier than pure A |
| **Potential** | 8 | Architecture-dependent critical exponents are genuinely exciting if they survive; floor is still a solid tool contribution |
| **Feasibility** | 7 | B's infrastructure is well-scoped (8/10); A's math additions add risk (offset by graceful degradation); 5-width calibration and descoped evaluation keep CPU budget realistic |

## What We Explicitly Do NOT Do

1. **No transformer support.** Softmax attention breaks NTK and mean-field analyses. This is a scope boundary, not a future extension within this project.

2. **No RG computational framework.** The RG interpretation is used in the paper discussion only. No dual-paradigm architecture, no spectral truncation of infinite-dimensional RG operators, no fixed-point computation. Debate consensus: RG framing is decorative (70% of Approach C's math) and adds ~50K LoC for zero computational advantage.

3. **No two-loop (1/N²) verification.** Computing 6th-order cumulants for structured architectures is intractable within project scope. One-loop truncation is validated empirically against ground-truth training runs, not against two-loop corrections.

4. **No universality class claims as deliverables.** Architecture-dependent critical exponents are reported if observed, not promised. The universality language (from Approaches A and C) is not part of the core claims.

5. **No codimension-2 bifurcation analysis.** Cusps and Hopf bifurcations are detected and flagged ("complex topology here") but not classified. This is decorative math identified by the debate.

6. **No symbolic algebra engine.** No SymPy. All H_{ijk} computations use numerical differentiation validated against known analytic results. Purpose-built index contraction for specific tensor structures if needed.

7. **No "$10K–$100K compute savings" claims.** Not validated on production-scale benchmarks. Practical value is stated as "sweep triage" and "µTransfer validation," not dollar amounts.

8. **No 20-seed ground-truth evaluation.** Descoped to 5 seeds (10 near boundaries) to fit CPU budget. Statistical power is reduced but sufficient for the revised accuracy targets.

9. **No full-resolution CIFAR-10.** PCA to 100 dimensions for CPU feasibility. This strips spatial structure that makes ConvNets interesting — acknowledged as a limitation, not hidden.

10. **No claims about phase boundary accuracy where V[Θ] ≥ 0.5.** The system explicitly flags where perturbation theory breaks down and makes no predictions there.

## Critical Path and Timeline

### Critical Path (Sequential Dependencies)
```
Week 1-2: Architecture IR + Exact NTK computation (JAX)
    ↓
Week 3-4: 5-width calibration pipeline + Θ^(1) regression
    ↓ [GATE: Does extracted Θ^(1) match Dyer & Gur-Ari for MLPs?]
Week 5-6: Linearized kernel ODE + eigenvalue bifurcation detection
    ↓ [GATE: Does system recover Chizat & Bach phase boundary?]
Week 7-8: Pseudo-arclength continuation + phase diagram mapper
    ↓
Week 9-10: Ground-truth evaluation harness + full MLP evaluation
    ↓
Week 11: Paper writing + ablation analysis
```

### Parallel Tracks (Can Run Alongside Critical Path)
```
Week 5-9: H_{ijk} numerical derivation for 1D ConvNets
           (parallel with Weeks 5-8 of critical path)
           [GATE at Week 7: Does factorization hold at N=8,16,32?]
           If YES → integrate into main pipeline for ConvNet evaluation
           If NO → report negative result, ConvNets use empirical calibration only

Week 7-10: Ground-truth training runs (overnight/weekend batches)
            (parallel with Weeks 7-10 of critical path)
            MLP runs first, ConvNet runs if Phase 4 gate passes

Week 10-12: Stretch goals if capacity allows:
             - Residual architecture support
             - 2D ConvNet H_{ijk} attempt
             - Critical exponent measurement
```

### Decision Gates
- **Gate 1 (Week 4):** Calibration pipeline produces Θ^(1) for MLPs matching known results within 5%. If NO: debug regression pipeline (1 week buffer).
- **Gate 2 (Week 6):** Bifurcation detection recovers known 2-layer ReLU phase boundary. If NO: fall back to pure empirical calibration (Approach B floor).
- **Gate 3 (Week 7):** H_{ijk} factorization for 1D ConvNets holds at small N. If NO: ConvNets use empirical calibration only (graceful degradation, not project failure).

### Total Estimated LoC: ~50K
- Architecture IR + Parser: 10K
- Kernel Engine (NTK computation + calibration): 15K
- Finite-Width Corrections (H_{ijk} numerical + regression): 8K
- Phase Diagram Mapper (bifurcation + continuation): 7K
- Evaluation Harness: 5K
- Testing: 5K

This is significantly descoped from the ~65K estimates in all three original approaches, reflecting the debate consensus that code volume signals engineering complexity, not idea depth. The reduction comes from: no symbolic engine (~15K saved), no dual-paradigm architecture (~20K saved), no RG computational framework (~15K saved), offset by additions for 5-width calibration, error propagation, and continuous order parameter computation.
