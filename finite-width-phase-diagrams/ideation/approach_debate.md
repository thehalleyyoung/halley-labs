# Approach Debate

This document synthesizes the adversarial skeptic review, math depth assessment, and difficulty assessment into a structured debate for each of the three approaches to finite-width phase diagrams.

---

## Approach A: Theory-First — Perturbative Field Theory of Kernel Evolution

### Skeptic's Attack

1. **The H_{ijk} factorization is a hope, not a theorem.** The entire approach bets on `H^(conv)_{ijk} = H^(dense)_{ijk} ⊗ K^(patch)_{pp'}`. Weight sharing introduces non-trivial correlations between spatial positions in higher cumulants — the 4th cumulant of weight-shared variables does not generically factor into a dense ⊗ spatial tensor product. If it fails, per-step cost becomes O(n³S²), making the approach computationally infeasible on a laptop at S=32×32=1024.

2. **Perturbation theory diverges at phase boundaries.** The 1/N expansion has an effective parameter γ/N^{something}, and near phase boundaries γ is O(1). The convergence radius shrinks to zero at transitions — a theorem in statistical mechanics. The approach is *least reliable precisely where it claims to be most interesting*. The perturbative validity functional V[Θ] can detect failure but cannot fix it.

3. **Gaussian moment closure is uncontrolled.** No theorem bounds the error of Gaussian closure (κ₄ = 0) for neural network cumulants. The ±50% perturbation sensitivity analysis is arbitrary; actual κ₄ could be orders of magnitude larger near phase transitions where fluctuations diverge.

4. **Benchmarks are suspiciously friendly.** MNIST/Fashion-MNIST are nearly linearly separable (lazy vs. rich barely matters). CIFAR-10 subsampled to n=2K with PCA to 100 dimensions strips the spatial structure that makes ConvNets interesting.

5. **"Why not just..." attack:** Compute exact NTKs at 3–5 widths via autodiff, fit a power law to the phase boundary. Zero new math, uses existing JAX/PyTorch autodiff, gives *exact* finite-width kernels.

### Math Depth Assessment

**Estimated math success probability: ~45%.**

- **Load-bearing math (~60% of total):** The H_{ijk} factorization for convolutions is the single load-bearing wall. If it fails, the approach has no fallback that preserves its theoretical identity. The spectral bifurcation theory for kernel ODEs and the residual skip-connection Jacobian composition are genuinely novel and necessary.

- **Decorative math (~40% of total):** The universality class language (critical exponents, codimension-2 analysis) is borrowed from statistical mechanics and applied to systems with 10⁴–10⁶ parameters rather than 10²³. The symbolic term-rewriting system and codimension-2 bifurcation analysis are architecturally ambitious but not load-bearing for the core result. The perturbative validity functional V[Θ] is a band-aid metric, not a substantive mathematical contribution.

- **Key gap:** No convergence theorem for the 1/N expansion at the target widths (64–512). The two-loop (1/N²) verification involving 6th-order cumulants may be intractable, leaving the one-loop truncation unvalidated.

### Difficulty Assessment

- **LoC underestimated by 30–50%.** The approaches doc estimates ~65K lines; realistic estimate is **80–95K** actual. The symbolic kernel engine (SymPy-based term rewriting, index contraction verification, symbolic-to-numerical lowering) alone could consume 25–30K lines. Bifurcation continuation with codimension-2 handling adds another 10–15K beyond the naive estimate.

- **SymPy is a critical single point of failure.** The entire symbolic pipeline depends on SymPy's ability to handle architecture-dependent index contractions at acceptable speed. SymPy is notoriously slow for large symbolic expressions. If SymPy chokes on ConvNet index structures, there is no drop-in replacement — the team would need to build a custom symbolic engine mid-project.

- **The H_{ijk} factorization is all-or-nothing.** Unlike Approach B where calibration accuracy degrades gracefully, the factorization either holds (enabling O(n³·rank(K)) computation) or doesn't (forcing O(n³S²), a ~10⁶× blowup). There is no middle ground.

- **CPU budget:** The 20-seed ground-truth evaluation runs (~250 hours) far exceed the stated 12–24 hour budget. Even with the perturbative computations themselves being fast, validation is the bottleneck.

### Cross-Examiner Challenges

**Math Assessor vs. Skeptic:**
- The math assessor rates success at ~45%, while the skeptic's "Why Not Just..." attack suggests the approach may be unnecessary even if the math succeeds. *Tension:* Even if H_{ijk} factorization works perfectly, the skeptic argues that exact NTK computation via autodiff at 3–5 widths could produce equivalent results with zero new math. The math assessor implicitly assumes the theoretical contribution has value; the skeptic questions whether the *prediction target* (phase boundaries for ConvNets at width ≤1024) justifies the mathematical investment.

**Difficulty Assessor vs. Visionary:**
- The visionary scores feasibility at 5/10, which the difficulty assessor considers *generous*. The LoC underestimate (30–50%), SymPy dependency risk, and all-or-nothing H_{ijk} bet collectively push real feasibility closer to 3/10. The visionary's fallback ("degrade to empirical calibration → Approach B") is an admission that the theoretical core is high-risk, not a mitigation strategy.

**Skeptic vs. Math Assessor on Universality Classes:**
- The skeptic argues that architecture-dependent universality classes have been claimed before (Roberts, Yaida & Hanin 2022; Halverson et al. 2021) and the conceptual contribution is not new. The math assessor flags that ~40% of the math is decorative, including the universality class language. *Agreement:* Both reviewers converge on the assessment that the universality narrative is oversold. The *computation* of architecture-dependent critical exponents matching experiment would be novel, but neither reviewer believes the approach as specified can deliver this within the feasibility envelope.

### Verdict

**What survives scrutiny:**
- The core idea of computing finite-width corrections to the NTK via perturbative expansion is mathematically sound and well-motivated.
- The spectral bifurcation framework for phase boundary detection is a genuinely good idea that could work if fed accurate corrections.
- The 1D convolution → validate → extend strategy for H_{ijk} is a reasonable research plan.

**What must be changed:**
- Drop the universality class and critical exponent narrative from the core deliverable. Make it a "future work" observation, not a central claim.
- Replace SymPy dependency with a purpose-built index contraction engine for the specific tensor structures needed (not general symbolic algebra).
- Add calibration widths {256, 512} to the H_{ijk} validation pipeline to catch failures early.

**What must be abandoned:**
- Two-loop (1/N²) verification within this project scope — it is intractable.
- Codimension-2 bifurcation analysis — decorative, not load-bearing.
- Claims about the approach working at phase boundaries where perturbation theory provably diverges.

---

## Approach B: Systems-First — Adaptive Kernel Observatory with Empirical Calibration

### Skeptic's Attack

1. **3-point calibration is statistically vacuous.** Fitting Θ(N) = Θ^(0) + Θ^(1)/N + Θ^(2)/N² to widths {32, 64, 128} gives exactly zero degrees of freedom — a 3-parameter model fit to 3 points. The fit is perfect but meaningless. Higher-order terms (1/N³, log(N)/N²) are absorbed into fitted coefficients, making extrapolation to N=1024 systematically biased.

2. **Ground-truth classification is circular.** The four indicators (kernel alignment drift, CKA, weight displacement, linear probing gap) are heuristics with thresholds that must be set. Different threshold choices give different "ground truths." Validating predictions against a threshold-dependent ground truth measures agreement between definitions, not accuracy against reality.

3. **Simple baselines are threatening.** The one-parameter heuristic "predict rich if η·N^{1-a-b} > C" captures essential µP scaling. If the ~65K-line system only beats this by 15 percentage points, the engineering effort is disproportionate.

4. **The tool solves yesterday's problem.** Practitioners training ConvNets/ResNets in 2024–2025 are switching to transformers. The target audience is shrinking. Benchmarks (MNIST, CIFAR-10 at n=2K) are toy problems no practitioner would use.

5. **"Why not just..." attack:** Run a coarse grid search at N=64 (3 hours), identify the interesting learning rate range, then fine search at target width (10 hours). Total: 13 hours of actual training producing actual models. The phase diagram validation alone costs more than this.

### Math Depth Assessment

**Estimated math success probability: ~70%.**

- **Load-bearing math (~90% of total):** Because there's so little math, nearly all of it is load-bearing. The regression estimator for Θ^(1), the Nyström-corrected spectral decomposition, the pseudo-arclength continuation — these are the working parts, and they use well-understood techniques (weighted least squares, low-rank approximation, predictor-corrector continuation).

- **Decorative math (~10% of total):** The "Bayesian fusion" of four indicators is a logistic regression with 4 features. Calling it Bayesian inflates a trivial model. The multi-indicator regime classification framework is reasonable engineering but not a mathematical contribution.

- **Key gap: Error propagation is missing.** The regression estimator for Θ^(1) is stated but the uncertainty propagation from Θ^(1) estimation error to phase boundary location error is not derived. Without this, the approach cannot provide calibrated confidence intervals on its predictions. This is the biggest mathematical gap.

- **Regression conditioning problem.** 3 parameters from 3 points gives a perfectly conditioned system with no residual to estimate error. Need either: (a) more calibration widths (4–5 points), or (b) a constrained model (e.g., fix Θ^(0) from infinite-width theory, reducing to 2 parameters from 3 points).

### Difficulty Assessment

- **LoC estimate is realistic.** The 65–75K estimate is the most honest of the three approaches. The components are well-understood: kernel computation (Neural Tangents exists as reference), Nyström approximation (textbook), continuation methods (textbook). The integration challenge is real but bounded.

- **CPU budget is feasible with descoping.** The kernel ODE computation for phase diagram sweeps fits within the 12–24 hour budget for MLP and shallow ConvNet architectures. ResNet-20+ will require the Nyström approximation at rank m=500–2000, which is feasible but tight.

- **Dependency risk is moderate.** JAX/Neural Tangents for kernel computation, SciPy for ODE solving and eigenvalue computation. These are mature, well-maintained libraries. No single point of failure.

- **Ground-truth evaluation is the shared bottleneck.** The 20-seed evaluation runs (~250 hours) exceed the budget. This is shared across all approaches but is most problematic for Approach B, whose value proposition depends on empirical validation against ground truth.

### Cross-Examiner Challenges

**Math Assessor vs. Skeptic:**
- The math assessor rates success at ~70%, significantly higher than A or C. But the skeptic argues that "success" for a system with ~90% load-bearing but trivial math means "correctly implementing known techniques." The disagreement is about *what counts as success* — the math assessor evaluates mathematical correctness; the skeptic evaluates whether mathematical correctness translates to a meaningful contribution. A well-engineered tool using known math may be useful but may not be publishable at a top venue.

**Difficulty Assessor vs. Visionary:**
- The visionary scores feasibility at 8/10; the difficulty assessor largely agrees. *However*, the difficulty assessor challenges the "immediate practical impact" narrative: the benchmarks are toy-scale, and the validation cost (ground-truth training) exceeds the cost of just doing the hyperparameter sweep directly at small scale. The tool only saves money at scales the benchmarks don't test.

**Skeptic vs. Difficulty Assessor on Value Proposition:**
- The skeptic's "$10K–$100K savings" critique and the difficulty assessor's observation that the benchmarks are MNIST-scale converge on the same point: *the value proposition is stated at production scale but validated at toy scale.* Neither reviewer believes the MNIST/CIFAR benchmarks demonstrate the claimed practical value. This is a framing problem, not a technical problem — the system might work at scale, but the paper cannot prove it does.

### Verdict

**What survives scrutiny:**
- The empirical calibration strategy is mathematically the safest path. Using known techniques (NTK computation, regression fitting, Nyström approximation) minimizes mathematical risk.
- The adaptive phase boundary refinement (pseudo-arclength continuation with error control) is a solid computational contribution.
- The graceful degradation strategy (multiple fallback levels) is good engineering.
- Production-quality tooling with robust error handling, UQ, and visualization is genuinely valuable.

**What must be changed:**
- Fix the 3-point regression: either add calibration widths {256} to get 4 points and 1 degree of freedom, or fix Θ^(0) analytically and fit 2 parameters from 3 points.
- Derive and implement error propagation from Θ^(1) estimation to phase boundary uncertainty. Without this, the UQ claims are hollow.
- Add at least one non-toy benchmark (CIFAR-10 at full resolution and full dataset, or a medical imaging task) to test the practical value proposition.
- Stop calling the 4-feature logistic regression "Bayesian fusion." It's logistic regression. Call it that.

**What must be abandoned:**
- Claims of "$10K–$100K compute savings" unless validated on production-scale benchmarks (which are outside project scope).
- The framing as a research contribution at a top venue (it is a systems/tools contribution — target JMLR MLOSS or a workshop).
- The pretense that ground-truth validation at 20 seeds is computationally cheap.

---

## Approach C: Hybrid — Renormalization Group Flow on Architecture Graphs

### Skeptic's Attack

1. **The finite-width kernel RG is conceptually inconsistent.** R_N is a random operator at finite N, not a deterministic one. A random operator doesn't have fixed points in the usual sense — it has distributional fixed points. Working with E[R_N[K]] discards the fluctuations, but the fluctuations *are* the finite-width corrections the approach claims to capture. You cannot simultaneously use the RG framework (requiring deterministic flow) and capture finite-width effects (which are stochastic).

2. **Spectral truncation is uncontrolled.** The relevant operators (eigenvalues > 1 of DR_N) might not align with the top eigenvectors of K*. In statistical mechanics, relevant operators near critical points are often *not* aligned with the equilibrium correlation structure. Truncating to m=50–100 could miss the most important perturbation entirely.

3. **"Depth as RG scale" is a metaphor, not a theorem.** In statistical physics, the RG transformation has a rigorous scale factor. In neural networks, each layer is a different nonlinear map with different weights. The "flow" is not iterating the same map — it's composing different maps with shared statistics. Calling this "RG flow" imports connotations of universality and fixed points without the mathematical foundations.

4. **Approach A does everything C does with less risk.** Both compute architecture-dependent critical exponents. A does it via bifurcation analysis of a well-defined ODE (mature numerics); C wraps the same computation in RG language that adds overhead without clear computational advantage. The claimed speedup is theoretical, not demonstrated.

5. **Dual-paradigm design enables cherry-picking.** When RG agrees with ODE, report RG and claim validation. When they disagree, report ODE and call the disagreement "an opportunity." This makes the RG framework unfalsifiable.

### Math Depth Assessment

**Estimated math success probability: ~25%.**

- **Load-bearing math (~30% of total):** The finite-width kernel RG transformation R_N and its linearization DR_N are genuinely novel constructions. The eigenvalue-based phase classification (|λ_i| > 1 → rich, |λ_i| < 1 → lazy) is a clean framework *if* the RG operator is well-defined.

- **Decorative math (~70% of total):** The RG framing repackages Approach A's computations in physics terminology without adding computational capability. The universality class claims (architecture-dependent critical exponent ν) are unlikely to survive scrutiny because: (a) ν is scheme-dependent in standard RG theory, meaning it depends on the truncation scheme, not just the architecture; (b) at the target widths (64–1024), finite-size effects dominate any putative universality. The "anomalous dimensions" language, "spatial coarse-graining" for ConvNets, and "scale-invariant architectures" rhetoric borrow prestige from condensed matter physics without importing the theorems that make those concepts rigorous there.

- **Key gap: Three unproven conjectures.** (1) That the spectral truncation to m=50–100 eigenvectors preserves relevant operators; (2) That convolutional block-diagonalization via spatial Fourier transform works for the linearized RG operator (not just for the kernel itself); (3) That the RG fixed points are well-defined at finite N (not just as distributional objects). All three must hold for the approach to produce meaningful results.

### Difficulty Assessment

- **LoC underestimated by ~2×.** The approaches doc estimates ~65K lines; realistic estimate is **90–120K** actual. The dual-paradigm architecture (RG + direct ODE, with shared IR but different computational backends) roughly doubles the implementation surface. The spectral truncation engine, RG operator implementation, and cross-validation harness between paradigms add substantial code that isn't accounted for.

- **Dual-paradigm architecture guarantees scope creep.** Maintaining two computational backends (RG and direct ODE) that must agree means debugging two systems, not one. When they disagree (which is the interesting case), the team must determine whether the disagreement is a bug in either implementation or a genuine limitation of the RG truncation — a research question masquerading as a debugging task.

- **Three unproven mathematical conjectures create cascading risk.** If conjecture 1 (spectral truncation) fails, the RG eigenvalue computation is wrong. If conjecture 2 (convolutional block-diagonalization) fails, the approach cannot handle ConvNets. If conjecture 3 (fixed-point well-definedness) fails, the entire framework is meaningless. These are not independent risks — failure of any one invalidates the approach, and their joint success probability is ~25%³ ≈ 1.5% if treated as independent (though they're correlated, so perhaps ~15–25%).

- **CPU budget is unrealistic.** The spectral truncation and eigenvalue computation for DR_N require repeated applications of the RG operator, each costing O(n²) per layer. For deep networks (20+ layers), this multiplies the computational cost. Combined with the shared ground-truth evaluation bottleneck (~250 hours), the approach cannot complete within any reasonable single-machine budget.

### Cross-Examiner Challenges

**Math Assessor vs. Skeptic:**
- The math assessor and skeptic are in *strong agreement* on Approach C. Both identify the decorative-to-load-bearing ratio as the core problem: ~70% of the mathematical apparatus is physics terminology draped over Approach A's computations. The math assessor's 25% success probability and the skeptic's characterization as "a physics paper cosplaying as an ML paper" are convergent diagnoses. *No significant disagreement between reviewers.*

**Difficulty Assessor vs. Visionary:**
- The visionary scores feasibility at 4/10, which the difficulty assessor considers *still too generous*. The 2× LoC underestimate, cascading conjecture risk, and dual-paradigm scope creep collectively push real feasibility to 2/10. The visionary's own score is the most honest of the three proposals, but even it underestimates the engineering challenge.

**Skeptic's Dominance Argument:**
- The skeptic's claim that "Approach A does everything C does with less risk" is largely validated by both other reviewers. The math assessor confirms that the RG framing doesn't add computational capability. The difficulty assessor confirms that the dual-paradigm architecture adds massive engineering cost. The only potential advantage of C over A — faster computation via fixed-point analysis instead of grid sweeps — is theoretical and undemonstrated.

### Verdict

**What survives scrutiny:**
- The eigenvalue-based phase classification (|λ_i| > 1 → rich phase) is an elegant and potentially useful *interpretation* of the same bifurcation analysis that Approach A performs. It could serve as a conceptual framework for presenting results, even if not used as the primary computational method.
- The idea of cross-validating RG predictions against direct ODE computations is good scientific methodology.

**What must be changed:**
- Demote the RG framework from "primary computational method" to "interpretive lens applied to Approach A's computations." Use Approach A's ODE as the computational engine; use the RG language to organize and interpret the results.
- Replace the three unproven conjectures with empirical tests: check spectral truncation accuracy, check block-diagonalization, check fixed-point convergence — all at small scale before committing to the full pipeline.

**What must be abandoned:**
- The dual-paradigm architecture. Choose one computational backend (the direct ODE), and use the RG framework for interpretation only.
- The universality class claims as a deliverable. They are scheme-dependent and unmeasurable at the target widths.
- The spatial coarse-graining for ConvNets as a novel contribution — it is an uncontrolled approximation layered on top of an already-uncontrolled spectral truncation.
- The 90–120K line implementation scope. This is not feasible.

---

## Cross-Cutting Issues

### 1. The Crossover Problem

All three approaches assume the lazy-to-rich transition is a *phase transition* with a well-defined boundary. At finite width, it is almost certainly a **smooth crossover**, not a sharp transition. The "phase boundary" is then an arbitrary isoline of a continuous order parameter, and its location depends on the threshold chosen. This doesn't invalidate the work, but it means:
- The claimed 15% relative error on boundary localization measures agreement between two arbitrary definitions, not a physical quantity.
- Different reasonable threshold choices could move the boundary by 50% or more.
- The entire concept of a "phase diagram" is somewhat misleading at finite width — "regime map with soft boundaries" would be more honest.

### 2. The Ground-Truth CPU Bottleneck

All approaches require 20-seed ground-truth evaluation runs for validation, estimated at ~250 hours of compute. This far exceeds the stated 12–24 hour budget. None of the approaches acknowledge this bottleneck adequately. Options: (a) reduce to 5 seeds (weaker statistics), (b) reduce architecture/dataset coverage, (c) use a cluster (changes the feasibility model), or (d) accept that ground-truth validation is a separate computational campaign run over days/weeks.

### 3. The Transformer Elephant

The problem statement explicitly excludes transformers, which constitute >90% of architectures practitioners train in 2024–2025. A phase diagram system for ConvNets/ResNets/MLPs has a narrow and shrinking audience. This is a problem statement issue, not an approach issue, but it caps the practical impact of all three approaches. The honest framing is: this is a *theoretical* contribution validated on classical architectures, not a *practical* tool for current practitioners.

### 4. The 65K Lines of Code Problem

All three approaches claim ~65K lines. For a research contribution, this signals that complexity is in engineering, not ideas. The best theory papers prove results in 20 pages of math; the best systems papers justify complexity with 10× performance gains. This project risks falling between: too much engineering for a theory paper, not enough performance gain for a systems paper, targeting architectures the community is moving away from.

### 5. Coverage vs. Calibration

The problem statement targets >90% identification of prediction failures as low-confidence. But if 40% of predictions are flagged as low-confidence, the system is useless even with perfect calibration. None of the approaches specify a target *coverage rate* (fraction of predictions that are high-confidence). A system that says "I don't know" 60% of the time and is correct 90% of the remaining time is less useful than a simple heuristic that's 75% accurate everywhere.

---

## Comparative Verdict

### Risk-Adjusted Ranking

| Dimension | Approach A | Approach B | Approach C |
|-----------|-----------|-----------|-----------|
| **Math success probability** | ~45% | ~70% | ~25% |
| **Realistic LoC** | 80–95K | 65–75K | 90–120K |
| **Feasibility (reassessed)** | 3–4/10 | 7/10 | 2–3/10 |
| **Novelty if successful** | High | Moderate | Very High |
| **Fallback quality** | Degrades to B | Always works at some level | Degrades to A |
| **Audience size** | Small (theorists) | Medium (practitioners + theorists) | Very small (physics-of-DL) |

### Best Risk-Adjusted Potential: Approach B

**Approach B has the best risk-adjusted potential.** It is the only approach whose difficulty matches its scope, whose math is likely to succeed, and whose engineering is realistically scoped. Its weakness is novelty — it is a systems/tools contribution, not a theoretical breakthrough. But a tool that works beats a theory that doesn't.

### Elements to Preserve from Each Approach

**From Approach A, preserve:**
- The spectral bifurcation framework for phase boundary detection (eigenvalue crossing analysis of the linearized kernel ODE). This is the strongest mathematical idea across all three approaches and can be implemented within Approach B's engineering framework.
- The perturbative validity functional V[Θ] as a self-diagnostic. Even as a heuristic, it provides useful uncertainty quantification.
- The 1D-convolution-first validation strategy for any new mathematical derivations.

**From Approach B, preserve (as the foundation):**
- The empirical calibration pipeline (compute exact NTK at calibration widths, fit 1/N model).
- The adaptive Nyström rank selection for scalability.
- The production-quality engineering ethos: robust error handling, graceful degradation, informative diagnostics.
- The pseudo-arclength continuation for phase boundary tracking.

**From Approach C, preserve:**
- The eigenvalue-based phase classification (|λ_i| > 1 → rich) as an interpretive framework — not as a computational method, but as a way to present and understand the results.
- The cross-validation methodology (comparing independent computational paths to the same prediction).

### Recommended Strategy

**Build Approach B's infrastructure first** (empirical calibration, kernel computation, adaptive refinement, ground-truth harness). Then **layer Approach A's bifurcation analysis on top** as a theoretical enhancement for architectures where H_{ijk} factorization can be derived and validated (start with MLPs, attempt 1D ConvNets). **Use Approach C's RG framework as interpretive language** in the paper's discussion section, not as a computational method.

This hybrid strategy yields:
- **Floor:** A working phase diagram tool with empirical calibration (~70% probability of success).
- **Ceiling:** A working tool *plus* theoretically derived corrections with bifurcation analysis for selected architectures (~35% probability, i.e., B succeeds AND A's math works for at least one architecture class).
- **Moonshot:** All of the above *plus* evidence for architecture-dependent critical exponents (~10% probability, but if achieved, a best-paper-caliber result).

### Final Assessment

The debate reveals a clear hierarchy: Approach B is the pragmatic foundation, Approach A is the high-risk theoretical enhancement, and Approach C is a repackaging of A's computations in physics language that adds cost without adding capability. The optimal path is B-first with A-flavored theoretical enrichment and C-flavored interpretive framing. All approaches must confront the cross-cutting issues (crossover vs. transition, ground-truth CPU cost, transformer exclusion, coverage vs. calibration) that no amount of clever mathematics can circumvent.
