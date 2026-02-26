# Certified Phase Atlases for Biological ODE Models: Machine-Checkable Regime Classification via Validated Numerics, SMT Certificates, and Adaptive Parameter-Space Refinement

## Problem Description

Every systems-biology publication that fits an ODE model to data confronts a question it almost never answers rigorously: do the model's qualitative predictions survive parameter variation? A toggle-switch model may exhibit bistability at the fitted parameter point, but does bistability persist across the entire region compatible with experimental measurement uncertainty? The answer matters profoundly — cell-fate decisions, oscillatory gene regulation, and extinction thresholds all depend on qualitative regime identity, not on precise trajectory values. Yet parameter robustness of qualitative predictions is rarely verified in published systems-biology models. Individual case studies have demonstrated fragility — for example, Chickarmane et al. (2006) showed that bistability in a stem-cell circuit model is confined to a narrow parameter subregion, and Otero-Muras et al. (2017) demonstrated that multi-stability claims in several published models are sensitive to parameter uncertainty — but no systematic census of this fragility exists. The problem is real and documented, but its prevalence across the literature remains an open empirical question.

The tools that exist address fragments of this problem but leave its core unsolved. DSGRN provides combinatorial decompositions of parameter space for switching-network abstractions, but requires the modeler to commit to a Hill-function switching approximation and cannot handle general rational or polynomial right-hand sides. AUTO and MATCONT perform numerical continuation of bifurcation curves with remarkable efficiency, but provide no correctness guarantees — their output is floating-point and subject to silent failure near degenerate bifurcations, cusps, or high-codimension points. No existing tool produces a machine-checkable certificate that a claimed regime classification is correct.

We propose the first system that produces machine-checkable certified phase diagrams for parameterized nonlinear biological ODE models. The system rests on three pillars, each indispensable. **Pillar I: Validated numerics** provides the rigorous verification backbone. Taylor-model interval enclosures propagate guaranteed error bounds through ODE integration. The Krawczyk operator certifies existence and local uniqueness of equilibria within parameter-dependent boxes. The radii-polynomial approach certifies existence of periodic orbits by verifying a contraction condition on a Newton-like operator in a Banach space of Fourier–Chebyshev coefficients. Conley index computation via cubical homology classifies isolated invariant sets without requiring explicit knowledge of unstable manifolds. **Pillar II: Tiered certification** provides the logical verification layer. Tier 1 (always available): independent interval-arithmetic recomputation verifies every regime claim via a separate validated-numerics implementation. Tier 2 (when feasible): regime claims are encoded as first-order formulas over the reals and checked by the δ-complete SMT solver dReal, producing proof witnesses in SMT-LIB2 format. Tier 3 (for polynomial-only models without transcendentals): exact verification via Z3 on the QF_NRA fragment. This tiered architecture ensures that every claim has at least one independent verification path, while providing stronger SMT-backed certificates where solver scalability permits. **Pillar III: Adaptive parameter-space refinement** provides scalability. Adaptive mesh refinement of the parameter space is guided by eigenvalue-sensitivity-based relevance scoring of parameter directions, concentrating computational effort near phase boundaries where regime identity changes. This reduces the cost of systematic parameter-space coverage from exponential in parameter dimension to exponential only in the effective dimension of the phase-boundary manifold.

For a biological ODE model with $n$ state variables and $p$ parameters, within a user-specified parameter box $\mathcal{P} \subset \mathbb{R}^p$, the system produces: **(a)** a hierarchical partition of $\mathcal{P}$ into certified regime regions, each labeled with the number and stability type of equilibria (for $n \leq 6$), periodic orbit existence and approximate period (for $n \leq 3$), and Conley index classification of isolated invariant sets (for $n \leq 4$); **(b)** tiered certificates for every regime claim — interval-arithmetic verification always, SMT certificates in SMT-LIB2 format when dReal scalability permits; **(c)** a coverage report quantifying the fraction of $\mathcal{P}$ that has been certified, with explicit identification of uncertified gaps; **(d)** a visual phase atlas rendering the partition as a navigable diagram. These dimensional targets reflect honest engineering constraints based on known algorithmic scaling: equilibrium certification via the Krawczyk operator scales as $O(n^3)$ per box and remains practical to $n = 6$ (stretch goal: $n = 10$ for purely polynomial systems); periodic orbit certification via radii polynomials has been demonstrated for $n = 2$–$3$ in the literature and extending to $n \leq 3$ is the realistic target (with $n = 4$–$5$ as a stretch goal pending Milestone 0 benchmarking); Conley index computation requires cubical homology of index pairs whose combinatorial complexity grows exponentially in $n$, restricting it to $n \leq 4$. The primary parameter-dimension target is $p \leq 4$ for systematic coverage; $p = 5$–$6$ is a stretch goal; for $p > 6$, the system supports certified probing of user-specified slices and submanifolds.

This is not a claim to solve the parameter robustness problem in full generality. Chaotic regimes are detected via positive Lyapunov exponent enclosures but not classified beyond "chaotic dynamics present." Delay differential equations, partial differential equations, and hybrid discrete-continuous models are out of scope. The system maps what it can certify with mathematical rigor and reports what it cannot, with explicit characterization of uncertified regions.

An optional GP-based adaptive exploration layer accelerates the pipeline without compromising its guarantees. Gaussian-process surrogate models, trained on the outcomes of validated integrations, predict regime identity across unvisited parameter boxes and estimate uncertainty. A phase-boundary-aware acquisition function directs sampling toward regions of high predicted regime-transition probability, reducing the number of expensive validated-integration calls compared to uniform grid search. Crucially, the GP layer is engineering infrastructure: it decides *where* to look, never *what* to certify. Every regime claim that enters the final atlas is backed by a validated-numerics computation and (where feasible) an SMT certificate. The novelty of this project lies in the certified verification pipeline, not in the surrogate-model machinery.

## Value Proposition

**For computational dynamicists and systems biologists**, this system transforms qualitative claims from empirical observations into certified guarantees. "We found bistability at our fitted parameters" becomes "bistability is guaranteed across this parameter region, with a machine-checkable proof." This is a qualitative shift in the epistemic status of computational results, comparable to the difference between a numerical simulation and a formal proof. The primary audience is the ~20–80 research groups worldwide working at the intersection of bifurcation analysis, parameter robustness, and formal methods for dynamical systems.

**For synthetic biologists**, the system provides rigorous robustness guarantees for genetic circuit design. A toggle switch intended to exhibit bistability can be certified as bistable across a manufacturing-tolerance region of its parameters — or the system will identify the subregion where bistability fails, enabling redesign before fabrication.

**For tool builders and the computational biology community**, certified phase atlases define a new interoperability layer. Two groups using different models of the same biological system can compare their certified phase diagrams directly, identifying regions of agreement and disagreement with mathematical precision.

**What becomes possible**: Reproducibility via independent certificate verification — any reviewer can check a regime claim without re-running the original code. Systematic model comparison via overlaid certified phase diagrams. A public repository of certified phase atlases that accumulates community knowledge about biological dynamical systems in a machine-verifiable format.

## Technical Difficulty

The project decomposes into seven core subproblems plus two optional extension modules. The core delivers the complete certified-atlas pipeline for equilibria; extensions add periodic orbit certification and stochastic-model support.

### Core Subproblems

1. **Validated ODE integration engine** — Specialization of existing Taylor-model libraries (CAPD or TM-lib) for biological rational right-hand sides (Michaelis–Menten, Hill functions), with wrapping-effect mitigation and adaptive step-size selection.

2. **Equilibrium certification via Krawczyk operator** — Parameter-dependent interval Newton methods for certifying existence and uniqueness of equilibria within boxes, including eigenvalue enclosure for stability classification and handling of near-singular Jacobians at bifurcation points.

3. **SMT certificate generation** — Translation of validated-numerics results into first-order formulas, encoding of regime claims in SMT-LIB2 format compatible with dReal's δ-complete decision procedures, soundness-preserving formula simplification, and certificate composition for hierarchical claims.

4. **Tiered verification architecture** — Tier 1: independent interval-arithmetic recomputation. Tier 2: dReal δ-complete verification. Tier 3: Z3 exact verification for polynomial-only models. Automatic tier selection based on formula complexity and solver capabilities.

5. **Adaptive parameter-space refinement** — Adaptive octree-style refinement of parameter space, eigenvalue-sensitivity analysis for identifying relevant parameter directions, anisotropic splitting strategies that align with phase-boundary geometry.

6. **Biological model ingestion** — Adapter layer over libSBML for SBML format support; automatic extraction of ODE right-hand sides, parameter ranges, and conservation laws; symbolic preprocessing (steady-state reduction) to reduce effective dimension.

7. **Visualization and infrastructure** — Phase-atlas rendering with hierarchical drill-down, certificate provenance tracking, parallel work-distribution across CPU cores, checkpoint/restart for long computations.

### Optional Extension Modules

**E1. Periodic orbit certification via radii polynomials** — Fourier–Chebyshev discretization of the periodic orbit boundary-value problem, construction of the radii-polynomial inequalities, rigorous computation of the required operator norms. Target: n≤3 systems.

**E2. Moment-closure error bounds** — Rigorous truncation error analysis for moment-closure approximations of stochastic kinetic models. This is a self-contained research contribution and is explicitly scoped as future work unless Milestone 0 benchmarking demonstrates surplus capacity.

### Lines-of-Code Breakdown (New Code)

| # | Subsystem | New LoC | Library Dependencies | Justification |
|---|-----------|---------|---------------------|---------------|
| 1 | Validated ODE integration (bio specialization) | 6,000 | CAPD or TM-lib (~50K existing) | Rational-RHS enclosures (2K), positivity exploitation (1.5K), adapter layer (1K), tests (1.5K). Core Taylor-model arithmetic provided by library. |
| 2 | Equilibrium certification (Krawczyk) | 10,000 | — | Parameter-dependent interval Newton (4K), eigenvalue enclosure (3K), stability classification (1.5K), tests (1.5K). Largely from scratch; no suitable library exists for parameter-dependent certified equilibrium analysis. |
| 3 | SMT certificate generation | 10,000 | dReal, Z3 | Formula encoding (4K), δ-completeness interface (2.5K), certificate composition (2K), SMT-LIB2 serialization (1.5K). |
| 4 | Tiered verification | 4,000 | — | Tier dispatch logic (1.5K), independent IA recomputation harness (1.5K), tests (1K). |
| 5 | Adaptive refinement | 8,000 | — | Adaptive octree (3K), eigenvalue-sensitivity analysis (2.5K), convergence monitoring (1.5K), tests (1K). |
| 6 | Model ingestion | 4,000 | libSBML (~200K existing) | SBML adapter (2K), symbolic preprocessing (1K), tests (1K). Heavy lifting done by libSBML. |
| 7 | Visualization & infrastructure | 6,000 | Plotly or Matplotlib | Renderer (2.5K), parallel dispatch (2K), checkpoint/restart (1.5K). |
| E1 | Periodic orbit certification (optional) | 12,000 | — | Fourier–Chebyshev operations (4K), radii-polynomial construction (4K), operator norms (2.5K), tests (1.5K). |
| E2 | Moment-closure (optional, future work) | 8,000 | — | Moment-closure ODE generation (3K), Grönwall error propagation (3K), tests (2K). |
| — | GP-based exploration (optional) | 4,000 | GPyTorch or scikit-learn | Acquisition functions (2K), batch sampling (1K), tests (1K). Core GP provided by library. |
| | **Core Total** | **48,000** | | |
| | **With all optional modules** | **72,000** | | |

## New Mathematics Required

The following mathematical contributions are required for the core system. Each is stated with sufficient precision to evaluate its difficulty and novelty.

**A1: Taylor-model validated integration for rational biological RHS.** Extend Taylor-model enclosure theory to rational RHS by representing denominators via validated reciprocal enclosures, with tight remainder bounds that exploit the positivity of biological state variables to avoid division-by-zero singularities. This is a careful specialization of known techniques, not a fundamentally new theorem.

**A2: Parameter-dependent Krawczyk theorem for uniform equilibrium certificates.** Given a parameter box $\mathcal{B} \subset \mathbb{R}^p$ and a state box $\mathcal{X} \subset \mathbb{R}^n$, certify that for *every* $\mu \in \mathcal{B}$, the system $f(x, \mu) = 0$ has a unique solution in $\mathcal{X}$, and that solution has consistent stability type throughout $\mathcal{B}$. This requires a uniform contraction argument over the parameter fiber. Variants exist in the literature (e.g., Rump, Tucker); the contribution is the biological-ODE specialization and integration with the certificate pipeline.

**B1: δ-complete SMT encoding of regime claims (soundness theorem).** Formalize the encoding of regime claims as first-order formulas in the theory of the reals with transcendental functions. Prove that the encoding is sound: if dReal returns UNSAT, the regime claim holds for all parameters in the specified box, up to the δ-perturbation inherent in δ-complete decision procedures. Characterize the gap between δ-soundness and exact soundness, providing explicit bounds on δ sufficient for biological applications. This is a genuine research contribution — the soundness argument requires careful treatment of the interaction between interval-arithmetic verification and δ-complete semantics.

**B2: Certificate composition calculus.** Define a formal calculus for composing certificates from subregions into certificates for larger regions. Prove that hierarchical composition preserves soundness. Handle the boundary between adjacent certified regions by requiring overlap or explicit boundary certificates. This is straightforward but requires careful formalization.

**C1: Adaptive refinement convergence.** Prove that eigenvalue-sensitivity-guided adaptive refinement converges: as refinement depth increases, the fraction of uncertified parameter-space volume converges to zero, provided the regime boundaries are piecewise-smooth manifolds of codimension ≥ 1. This is a standard AMR convergence argument specialized to the eigenvalue-sensitivity heuristic.

**(Optional) A3: Radii-polynomial periodic orbit certification for dimension ≤ 3.** Extension of existing techniques (Lessard, van den Berg) to biological ODE right-hand sides with rational nonlinearities. The n≤3 target is realistic; n=4–5 is a stretch goal requiring novel bounds on operator norms in product Banach spaces.

**(Optional, future work) D1: Moment-closure error bounds via Grönwall inequalities.** Rigorous bounds on the error between stochastic system means and moment-closure ODE solutions. Self-contained contribution, not required for the core atlas pipeline.

## Best Paper Argument

**First demonstration of machine-checkable regime certification.** The central intellectual contribution is the concept of a *certified phase atlas* as a first-class scientific artifact — analogous to proof-carrying code in programming languages. No prior system produces machine-checkable certificates that a claimed regime classification is correct for a biological ODE model. Even a working demonstration on the Gardner toggle switch alone would be a first.

**Tiered certification architecture.** The design of a verification architecture with graceful degradation — always providing interval-arithmetic verification, adding SMT certificates where solver scalability permits, and offering exact verification for the polynomial fragment — is a practical architectural contribution to the formal-methods-for-science community.

**Multi-paper research program.** This project is best positioned as a 2–3 paper program targeting specific venues:
- **Paper 1:** "Certified Equilibrium Atlases for Biological ODEs via Validated Numerics and δ-Complete SMT" → CAV, TACAS, or HSCC. Focus: soundness theorem (B1), tiered verification architecture, toggle-switch + repressilator demonstrations.
- **Paper 2:** "Radii-Polynomial Periodic Orbit Certification for Biological Oscillators" → Journal of Computational Dynamics or Nonlinearity. Focus: A3, Goodwin oscillator demonstration.
- **Paper 3:** "Certified Phase Diagrams as Reproducibility Infrastructure for Systems Biology" → PLOS Computational Biology or Bioinformatics. Focus: end-to-end pipeline, 5-model benchmark suite, usability evaluation.

**Reproducibility contribution.** If a reviewer can independently verify the certificate, the qualitative claims of a paper are reproducible by construction. This is a concrete, technically grounded response to the reproducibility concerns in computational biology.

## Evaluation Plan

### Milestone 0: dReal Feasibility Benchmarking (First Priority)

Before any pipeline implementation, benchmark dReal on formulas representative of the target models:

| Test | Variables | Transcendentals | Expected Outcome |
|------|-----------|-----------------|------------------|
| Toggle switch equilibrium claim | 6 (n=2, p=4) | Hill functions | Feasible (baseline) |
| Repressilator equilibrium claim | 7 (n=3, p=4) | Hill functions | Likely feasible |
| Brusselator equilibrium claim | 4 (n=2, p=2) | None (polynomial) | Feasible; also test Z3 |
| Goodwin equilibrium claim | 7 (n=3, p=4) | Hill functions | Likely feasible |
| EMT equilibrium claim (if attempted) | 12 (n=6, p=6) | Hill functions | Likely infeasible; determines stretch-goal viability |

**Decision gate:** If dReal cannot handle the toggle-switch formula in <10 minutes, the SMT tier must be redesigned or scoped to polynomial-only models. If toggle-switch and repressilator succeed but EMT fails, cap SMT tier at n+p ≤ 8 and rely on interval-arithmetic tier for larger models.

### Benchmark Models

Five biological ODE models spanning the feasible scope:

| # | Model | $n$ | $p$ | Expected Regimes | Certification Target |
|---|-------|-----|-----|------------------|---------------------|
| 1 | Gardner toggle switch | 2 | 4 | Bistability, monostability | Full atlas, all tiers |
| 2 | Repressilator | 3 | 4 | Oscillation, stable equilibrium | Equilibrium atlas, periodic orbit stretch |
| 3 | Goodwin oscillator | 3 | 4 | Hopf bifurcation, damped oscillation | Equilibrium atlas, periodic orbit stretch |
| 4 | Brusselator | 2 | 2 | Limit cycle, stable focus | Full atlas, all tiers (polynomial — Z3 exact) |
| 5 | Sel'kov glycolysis | 2 | 3 | Oscillation, steady state | Full atlas, all tiers |

**Stretch models** (attempted only if Milestone 0 indicates feasibility):

| # | Model | $n$ | $p$ | Notes |
|---|-------|-----|-----|-------|
| S1 | Lac operon | 5 | 6 | Equilibria only, interval-arithmetic tier |
| S2 | EMT network | 6 | 6 | Equilibria only, dReal feasibility TBD |

### Automated Metrics

1. **Tiered certificate verification rate.** Fraction of regime claims verified by at least one independent method. Tier 1 (interval-arithmetic recomputation): target 100%. Tier 2 (dReal): report success rate by model and formula size. Tier 3 (Z3, polynomial models only): target 100%. Any Tier 1 failure indicates a soundness bug.

2. **Coverage completeness.** Fraction of $\mathcal{P}$ covered by certified regime regions. Target: ≥ 95% for $p \leq 4$. Report actual coverage for each benchmark model.

3. **Classification accuracy.** Agreement between certified regime labels and ground-truth labels obtained by dense numerical continuation (AUTO). Target: 100% agreement wherever both methods produce a classification.

4. **DSGRN comparison.** For toggle switch and repressilator, compare certified phase atlas against DSGRN's combinatorial decomposition.

5. **Scalability curves.** Wall-clock time and peak memory as functions of $n$ and $p$, reported separately for equilibrium certification and (where applicable) periodic orbit certification. Compare empirical scaling against theoretical predictions.

6. **Adaptive refinement speedup.** Ratio of certified volume per unit time for adaptive refinement versus uniform grid refinement. Target: ≥ 10× speedup for $p = 4$.

7. **Certificate size and verification time.** Total size of certificate files (interval-arithmetic logs + SMT-LIB2 files) and time to independently verify them.

## Laptop CPU Feasibility

This system is designed to run on a single multi-core laptop CPU. Every computational kernel is CPU-native; no GPU, cluster, or cloud infrastructure is required.

**Variable-precision interval arithmetic is CPU-native.** Interval arithmetic operations map directly to x86/ARM floating-point instructions with rounding-mode control. Taylor-model coefficient arithmetic is dense linear algebra on small matrices ($n \leq 6$), well within L1/L2 cache. The MPFR library provides arbitrary-precision interval arithmetic when double precision is insufficient.

**SMT solving is CPU-optimized but scalability-limited.** dReal is a single-threaded CDCL(T) solver optimized for CPU execution. Its interval constraint propagation core is CPU-native. However, **dReal's scalability on formulas with transcendental functions and 10+ real variables is a known limitation.** The tiered certification architecture mitigates this: models where dReal is infeasible still receive Tier 1 (interval-arithmetic) certification. Milestone 0 benchmarking will establish empirically which models can receive Tier 2 certification.

**Taylor-model integration parallelizes across parameter boxes.** The outer loop — iterating over parameter boxes — is embarrassingly parallel. A thread pool across 6–8 CPU cores provides linear speedup with minimal synchronization overhead.

**Cubical homology is integer linear algebra.** Conley index computation reduces to Smith normal form of integer boundary matrices. For $n \leq 4$, these are small enough that computation completes in seconds on a single core (via CHomP library).

**dReal scalability contingency.** If Milestone 0 reveals that dReal cannot handle target formulas within reasonable time, the project falls back to Tier 1 certification only (interval-arithmetic recomputation). This reduces the "certified" claim from "SMT-backed proof witness" to "independently recomputed validated-numerics result" — still a significant improvement over uncertified numerical continuation, but a weaker guarantee. The tiered architecture ensures this fallback does not invalidate the pipeline.

**Projected compute budget (pending Milestone 0 validation).** For the 5-model core benchmark suite: equilibrium certification ≈ 2–8 hours per model (based on CAPD benchmarks for comparable systems); Conley index ≈ 1–4 hours for n≤4 models. SMT certification time is unknown pending Milestone 0 and is explicitly not estimated here. Total core computation: approximately 2–5 days on a 6-core laptop, excluding SMT certification time.

---

**Slug:** `certified-bio-phase-cartographer`
