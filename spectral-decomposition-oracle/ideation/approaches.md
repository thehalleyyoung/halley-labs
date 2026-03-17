# Competing Approaches — Spectral Decomposition Oracle

> **Project**: spectral-decomposition-oracle
> **Phase**: Ideation
> **Binding constraints**: Amendment E (census-first, T2 motivational only, honest scope ≤30K LoC,
> external baselines via GCG/SCIP-native, target JoC, futility *predictor* not certificate,
> feature ablation as core experiment)

---

## Approach A: Census-Heavy (Empirical Infrastructure)

### A.1 Extreme Value

**Who needs this desperately**: Operations-research practitioners who encounter a new MIP and ask
*"Should I even attempt decomposition, and if so, which kind?"* Today there is no systematic
answer. GCG detects Dantzig-Wolfe structure but ignores Benders and Lagrangian amenability. The
MIPLIB benchmark library ships zero decomposition metadata. Every researcher who studies
decomposition methods re-derives structure labels ad hoc.

The census artifact answers a question the community has punted on for 20 years: *for every
standard benchmark instance, what does each decomposition method actually achieve?* This is the
MIP equivalent of the SAT Competition runtime matrices — a public good that enables all
subsequent algorithm-selection and portfolio research without requiring anyone to re-run expensive
decomposition experiments.

**Primary users**:
- Algorithm-selection researchers (Kruber et al. 2017 successors) who need ground-truth labels
- Solver developers evaluating decomposition heuristics on standardized benchmarks
- Instructors needing canonical examples of Benders/DW/Lagrangian structure

### A.2 Genuine Software Difficulty

| Subproblem | Why it's hard | Approach |
|---|---|---|
| **Reproducible decomposition harness** | Must wrap GCG (DW), SCIP-native Benders, and a Lagrangian relaxation driver with identical timeout/gap semantics, logging schema, and deterministic seeds across 500+ instances | Thin PySCIPOpt adapter layer with a shared `DecompositionResult` schema; GCG called via subprocess with MPS roundtrip; Lagrangian via ConicBundle C API |
| **Structure detection at scale** | GCG's detection is DW-only; Benders structure requires identifying complicating variables (not complicating constraints); Lagrangian requires identifying dualizeable constraints | Implement three independent detectors: (1) GCG's `detect` for DW, (2) variable-participation heuristic for Benders (rows touched by ≤k variables → linking), (3) constraint-clustering for Lagrangian (rows with high inter-block coupling weight) |
| **Fair cross-method comparison** | Different methods produce incomparable dual bounds (DW master LP vs. Lagrangian dual vs. Benders master) at different computational costs | Normalize to *dual bound achieved at wall-clock parity* (e.g., 300s, 900s) with SCIP's default LP relaxation as the baseline denominator |
| **Census annotation pipeline** | 500 instances × 3 methods × 2 time limits = 3,000 runs; must handle SCIP/GCG crashes, memory limits, infeasibility, and numerical issues gracefully | SQLite-backed job queue with idempotent retry; Docker container per run for isolation; Parquet output with provenance columns |

**Architecture**:
```
census/
  harness/
    scip_benders_wrapper.py     # PySCIPOpt Benders adapter
    gcg_dw_wrapper.py           # GCG subprocess driver
    lagrangian_wrapper.py       # ConicBundle / subgradient driver
    result_schema.py            # Shared DecompositionResult dataclass
  detectors/
    dw_detector.py              # Delegates to GCG detect
    benders_detector.py         # Variable-participation heuristic
    lagrangian_detector.py      # Constraint-clustering heuristic
  pipeline/
    job_queue.py                # SQLite job management
    runner.py                   # Instance → detect → decompose → log
    aggregator.py               # Collect results, compute summary stats
  features/
    syntactic_features.py       # Constraint-matrix statistics (density, block structure metrics)
    spectral_features.py        # Hypergraph Laplacian eigenvalues (one feature family among many)
    graph_features.py           # Variable-interaction graph statistics
  analysis/
    ablation.py                 # Feature-family ablation experiment
    census_stats.py             # Summary tables, coverage metrics
    visualization.py            # Heatmaps, performance profiles
```

**LoC Breakdown**:
| Subsystem | New LoC | Notes |
|---|---|---|
| Solver wrappers (3 methods) | 3,500 | PySCIPOpt + subprocess + ConicBundle |
| Structure detectors | 2,500 | DW via GCG, Benders/Lagrangian heuristic |
| Pipeline infrastructure | 3,000 | Job queue, runner, aggregator |
| Feature extraction (all families) | 3,500 | Spectral, syntactic, graph |
| Analysis & ablation | 2,500 | Stats, plots, ablation harness |
| Tests & CI | 2,000 | Unit + integration on small instances |
| **Total** | **17,000** | Conservative; no solver reimplementation |

### A.3 New Math Required

**Minimal new math — the contribution is empirical.** Load-bearing formal content:

1. **Lemma L3 (Partition-to-Bound Bridge)** — restated and given a self-contained proof.
   > Let $P = \{B_1, \ldots, B_k\}$ be a partition of constraint rows into blocks, and let
   > $E_{\text{cross}}$ be the set of hyperedges (variables) spanning multiple blocks. Then
   > $$z_{LP} - z_D(P) \leq \sum_{e \in E_{\text{cross}}} |y^*_e| \cdot w(e)$$
   > where $y^*$ is an optimal dual solution to the monolithic LP and $w(e)$ is the
   > multiplicity of $e$ across block boundaries.

   **Role**: Justifies why partition quality (measured by crossing weight) is a meaningful
   predictor target. This is a known-type result (e.g., Vanderbeck & Savelsbergh 2006) but
   we give the first explicit statement for the hypergraph case with shadow-price weighting.

2. **Spectral feature definitions** — formal definitions of the 8 spectral features extracted
   from the constraint hypergraph Laplacian $L_H$, ensuring reproducibility:
   - Spectral gap $\gamma_2 = \lambda_2(L_H)$
   - Fiedler vector localization entropy
   - Coupling energy $\delta^2 = \sum_{e \in E_{\text{cross}}} w(e)^2$
   - Eigenvalue decay rate (exponential fit to $\lambda_2, \ldots, \lambda_{k+1}$)
   - Algebraic connectivity ratio $\gamma_2 / \gamma_k$
   - Spectral gap ratio $\delta^2 / \gamma^2$ (T2's predictor, used empirically)
   - Block separability index (based on eigenvector clustering quality)
   - Effective spectral dimension

   **Role**: These are the feature definitions for the ablation experiment. No theorems —
   just precise, reproducible definitions.

3. **T2 (Spectral Scaling Law)** — stated as a *motivational proposition* with an explicit
   caveat that $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$ is vacuous on ill-conditioned
   instances. Occupies ≤2 pages. Cited as *structural motivation* for why $\delta^2/\gamma^2$
   is a reasonable predictor feature.

### A.4 Best-Paper Potential

**Honest assessment: low ceiling, high floor.** The census is a public good; it will be cited
by anyone doing decomposition research for the next decade. But "we ran 3,000 experiments and
built a table" is not best-paper material at any venue. The ablation showing spectral features
help is incremental over Kruber et al. (2017).

**Best-paper path** (unlikely but possible): If the census reveals a *surprising structural
finding* — e.g., that Lagrangian relaxation dominates DW on a previously-unrecognized class of
MIPLIB instances, or that >40% of instances are futile for all decomposition methods — the paper
becomes a "Surprising Empirical Finding" contribution, which JoC occasionally rewards.

**What makes it publishable regardless**: First cross-method decomposition census of a standard
benchmark library; reproducible infrastructure; feature ablation with external baselines. This is
solid JoC material even without surprises.

### A.5 Hardest Technical Challenge

**Fair Lagrangian relaxation at scale.** GCG handles DW; SCIP handles Benders. But there is no
standardized, well-maintained Lagrangian relaxation solver for general MIPs. ConicBundle exists
but requires manual problem decomposition. Subgradient methods are fragile (step-size tuning).

**How to address it**:
- Limit Lagrangian to instances where the constraint-clustering detector identifies
  ≥2 blocks with ≥70% of constraints block-diagonal
- Use ConicBundle with a fixed parameter grid (5 configurations) and take best dual bound
- If ConicBundle is intractable: fall back to volume algorithm (Barahona & Anbil 2000)
  with 1,000-iteration cap
- **Kill condition**: If Lagrangian bounds are strictly dominated by DW bounds on >95% of
  instances where both apply, drop Lagrangian from the census and note this finding (which
  is itself a publishable observation)

### A.6 Timeline with Kill Gates

| Week | Milestone | Kill gate |
|---|---|---|
| 1 | GCG + SCIP Benders wrappers operational on 10 pilot instances | G2: ≥8/10 produce valid bounds |
| 2 | Spectral feature extraction + 50-instance pilot | G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4 |
| 3-4 | Lagrangian wrapper; structure detectors for all 3 methods | Lagrangian produces bounds on ≥50% of structured instances |
| 5-6 | 200-instance dev set: all 3 methods + feature extraction | G3: Spectral features improve selector over syntactic-only |
| 7-9 | 500-instance stratified evaluation | G4: Combined metrics pass |
| 10-11 | Paper draft; full 1,065-instance spectral annotation artifact | G5: Internal review pass |
| 12 | Submission-ready | — |

### A.7 What Could Go Wrong

| Risk | Probability | Recovery |
|---|---|---|
| GCG API changes break wrapper | Medium | Pin GCG version; distribute Docker image |
| Lagrangian relaxation intractable for most instances | High | Drop to 2-method census (Benders + DW); note Lagrangian absence as finding |
| Spectral features don't beat syntactic | Medium | Paper becomes pure census paper (still publishable at JoC, lower impact) |
| 500-instance eval exceeds compute budget | Low | Reduce to 300 instances with tighter stratification |
| Reviewer says "just use GCG" | High | Emphasize cross-method scope; GCG is DW-only |

### A.8 Scores

| Dimension | Score | Rationale |
|---|---|---|
| **Value** | 6/10 | Census is genuinely useful public good; not transformative |
| **Difficulty** | 4/10 | Mostly engineering; Lagrangian wrapper is the only hard part |
| **Best-Paper** | 2/10 | Solid JoC publication; not competitive for best paper |
| **Feasibility** | 8/10 | Low technical risk; main risk is compute time |

---

## Approach B: Spectral-Feature-First (Feature Engineering)

### B.1 Extreme Value

**Who needs this desperately**: The algorithm-selection-for-MIP community, which currently relies
on *syntactic* instance features (constraint matrix density, variable-type ratios, coefficient
statistics) that are blind to geometric structure. Kruber et al. (2017) showed ML can select DW
decompositions, but their features are GCG's binary structure flags — discrete, method-specific,
and unavailable before detection. The ISA (Instance Space Analysis) community (Smith-Miles et al.)
has no MIP-specific structural features.

Spectral features from the constraint hypergraph Laplacian are **continuous, method-agnostic,
and geometry-aware**. They capture the "shape" of the constraint interaction structure before
any decomposition is attempted. If the ablation demonstrates that spectral features carry
information beyond syntactic features, this validates an entirely new feature family for MIP
instance characterization — useful not just for decomposition selection but for any
algorithm-selection or portfolio task on MIPs.

**Primary users**:
- Algorithm selection researchers (AutoML for combinatorial optimization)
- MIP solver developers tuning heuristics based on instance structure
- Portfolio solver designers (SATzilla successors for MIP)

### B.2 Genuine Software Difficulty

| Subproblem | Why it's hard | Approach |
|---|---|---|
| **Hypergraph Laplacian construction** | The constraint matrix $A$ defines a hypergraph where rows are vertices and columns (variables) are hyperedges. The clique-expansion Laplacian $L_H = D - W$ (where $W$ is the weighted adjacency from clique expansion of hyperedges) is dense for instances with high-arity constraints (e.g., set-covering rows touching 500+ variables). Direct construction is $O(m \cdot d_{\max}^2)$ where $d_{\max}$ is the max hyperedge degree. | Sparse clique expansion with degree-weighted normalization; threshold hyperedges with degree >200 using random sampling of $O(\sqrt{d})$ pairs; validate that spectral features are robust to sampling via consistency check on 50 instances |
| **Eigendecomposition at scale** | Bottom-$k$ eigenvectors of a sparse matrix with $m$ up to 10^6 rows (MIPLIB's largest instances have ~1M constraints). ARPACK/scipy.sparse.linalg.eigsh may not converge for highly ill-conditioned Laplacians. | Use shift-invert mode with σ=0 for bottom eigenvectors; fall back to LOBPCG with random initialization for instances where ARPACK fails; cache eigendecompositions; set $k = \min(20, m/10)$ |
| **Feature robustness validation** | Spectral features are sensitive to matrix scaling, constraint reordering, and numerical precision. A feature that changes under permutation is useless. | Validate permutation-invariance: for 50 instances, randomly permute rows/columns 10 times, check feature coefficient of variation <5%. Report robustness metrics in paper. |
| **Ablation experimental design** | The ablation must be rigorous: 4 feature families (spectral, syntactic, graph-based, combined) × 3 classifiers (RF, XGBoost, logistic regression) × 5-fold stratified CV, with proper multiple-comparison correction | Use nested CV (outer 5-fold for evaluation, inner 3-fold for hyperparameter tuning); McNemar's test for pairwise feature-family comparison; report with confidence intervals, not just point estimates |
| **Ground-truth label acquisition** | Need {Benders, DW, Lagrangian, none} labels for each instance, but "best method" depends on time limit and metric (dual bound quality vs. gap closure vs. wall-clock to first feasible). | Define label as argmax of *dual bound improvement over LP relaxation at 300s wall-clock*; use GCG for DW dual bound, SCIP Benders for Benders dual bound; if neither improves LP bound by >1%, label "none" |

**Architecture**:
```
spectral/
  hypergraph/
    laplacian.py               # Hypergraph → sparse Laplacian (clique expansion)
    sampling.py                # High-degree hyperedge sampling
    normalization.py           # Degree-weighted, symmetric, random-walk normalizations
  eigensolve/
    solver.py                  # ARPACK + LOBPCG with fallback
    cache.py                   # Eigendecomposition cache (HDF5)
  features/
    spectral_features.py       # 8 spectral features from eigenvalues/eigenvectors
    syntactic_features.py      # 25 syntactic features (baseline)
    graph_features.py          # 10 variable-interaction graph features
    feature_pipeline.py        # Unified extraction: instance → feature vector
  census/
    wrappers.py                # GCG + SCIP Benders (label generation)
    labeling.py                # Dual-bound comparison → ground-truth label
  experiments/
    ablation.py                # Feature-family ablation with nested CV
    classifiers.py             # RF, XGBoost, LogReg with hyperparameter grids
    statistical_tests.py       # McNemar, Wilcoxon, confidence intervals
    performance_profiles.py    # Dolan-Moré profiles for solver comparison
  theory/
    l3_verification.py         # Numerical verification of L3 on census instances
    t2_correlation.py          # Empirical correlation of δ²/γ² with bound degradation
```

**LoC Breakdown**:
| Subsystem | New LoC | Notes |
|---|---|---|
| Hypergraph Laplacian + eigensolve | 4,000 | Core novel code; sparse + sampling |
| Spectral feature extraction | 2,000 | 8 features with robustness checks |
| Syntactic + graph features | 2,000 | Baseline feature families |
| Census wrappers + labeling | 2,500 | GCG + SCIP Benders only (drop Lagrangian from core) |
| Ablation experiment harness | 3,000 | Nested CV, statistical tests, plots |
| Theory verification scripts | 1,500 | L3 numerics, T2 correlation |
| Tests | 2,000 | Unit + property-based (permutation invariance) |
| **Total** | **17,000** | Could reach 20K with visualization |

### B.3 New Math Required

**Moderate new math — features need formal grounding.** Load-bearing content:

1. **Lemma L3 (Partition-to-Bound Bridge)** — same as Approach A, but here it serves a deeper
   role: it formally connects the spectral feature $\delta^2$ (coupling energy) to the
   decomposition quality metric. The proof uses the dual interpretation: crossing weight bounds
   the Lagrangian dual gap, which bounds the DW master LP gap via LP duality.

   **Tighter statement** (novel):
   > For a partition $P$ with $k$ blocks and crossing hyperedge set $E_{\text{cross}}$:
   > $$z_{LP} - z_D(P) \leq \sum_{e \in E_{\text{cross}}} |y^*_e| \cdot (n_e - 1)$$
   > where $n_e$ is the number of blocks spanned by hyperedge $e$, and $y^*$ is the
   > optimal LP dual vector. Moreover, for the spectral partition $P^*$ obtained from
   > the Fiedler vector of $L_H$:
   > $$\sum_{e \in E_{\text{cross}}(P^*)} w(e) \leq \frac{\delta^2}{\gamma_2}$$
   > where $\gamma_2 = \lambda_2(L_H)$ and $\delta^2 = \|L_H - L_{\text{block-diag}}\|_F^2$.

   This second inequality is the *spectral* partition bound — it says spectral partitioning
   controls crossing weight, which L3 connects to dual bound gap. The chain
   *spectral features → partition quality → dual bound quality* is the theoretical backbone.

2. **Proposition T2 (Motivational)** — stated with explicit vacuousness caveat:
   > $z_{LP} - z_D(\hat\pi) \leq C \cdot \delta^2/\gamma^2$ where $C = O(k\kappa^4\|c\|_\infty)$.
   > **Remark**: For MIPLIB instances with $\kappa > 10^3$ (≈60-70% of the library), $C$ exceeds
   > $10^{12}$, rendering this bound uninformative. We present T2 as *structural motivation* for
   > the spectral ratio predictor, not as a quantitative tool.

   **Role**: Explains *why* $\delta^2/\gamma^2$ is a natural feature. The ablation then tests
   whether this theoretical motivation translates to empirical predictive power.

3. **Feature-theoretic propositions** (small, new):
   > **Proposition F1 (Permutation Invariance)**: All 8 spectral features are invariant under
   > simultaneous row and column permutations of $A$.
   > *Proof*: The hypergraph Laplacian $L_H$ depends only on the hypergraph structure, which is
   > permutation-invariant. Eigenvalues are similarity-invariant. Eigenvector-derived features
   > (localization entropy, clustering quality) depend only on eigenvector component magnitudes.

   > **Proposition F2 (Scaling Sensitivity)**: Under row scaling $A \to DA$ where
   > $D = \text{diag}(d_1,\ldots,d_m)$, the spectral gap satisfies
   > $\gamma_2(L_{DA}) \in [\gamma_2(L_A) / \|D\|_2^2, \; \gamma_2(L_A) \cdot \|D\|_2^2]$.
   > *Consequence*: Pre-scaling (e.g., equilibration) is essential before spectral extraction.

   **Role**: F1 ensures features are well-defined; F2 motivates the preprocessing pipeline.

### B.4 Best-Paper Potential

**Medium ceiling.** If the ablation demonstrates a *statistically significant and practically
meaningful* improvement from spectral features (e.g., decomposition selection accuracy jumps from
62% → 74% when adding spectral features to syntactic), this validates an entirely new feature
family for MIP instance characterization. The contribution generalizes beyond decomposition
selection to any MIP algorithm-selection task.

**Best-paper path**: The ablation shows spectral features contain complementary information not
captured by syntactic features, AND the spectral futility predictor achieves >85% precision on
"don't decompose" predictions. This combination — *new features that work* + *practical futility
detection* — is a strong JoC contribution. Could potentially interest Mathematical Programming
Computation if the empirical story is clean enough.

**What makes it publishable regardless**: Feature engineering with rigorous ablation methodology;
L3 as a standalone theoretical contribution; census as an artifact. Even if spectral features are
only marginally better, the *methodology* (hypergraph Laplacian features for MIP characterization)
is novel and worth documenting.

### B.5 Hardest Technical Challenge

**Spectral feature robustness across MIPLIB's extreme heterogeneity.** MIPLIB instances range
from 50-row knapsack covers to 500K-row supply-chain models. Coefficient magnitudes span $10^{-6}$
to $10^{12}$. The spectral gap $\gamma_2$ might be $10^{-15}$ for a nearly-disconnected instance
or $10^{3}$ for a dense one. Features that work on one scale might be meaningless on another.

**How to address it**:
1. **Equilibration preprocessing**: Apply Ruiz scaling (alternating row/column normalization) to
   $A$ before Laplacian construction. This is standard in SCIP but must be replicated exactly.
2. **Relative features**: Use ratios ($\gamma_2/\gamma_k$, $\delta^2/\gamma^2$) rather than
   absolute values to achieve scale-invariance.
3. **Feature normalization validation**: For each spectral feature, plot distribution across MIPLIB
   and verify it has reasonable spread (not bimodal with 90% at 0). Drop features with <0.1 bits
   of entropy.
4. **Robustness test (built into pipeline)**: For the 200-instance dev set, add Gaussian noise
   ($\epsilon = 10^{-8}$) to constraint coefficients and verify feature stability ($r^2 > 0.99$
   between original and perturbed features).
5. **Fallback**: If spectral features are unstable on >30% of instances, restrict to the "clean"
   subset (well-conditioned, $\kappa < 10^6$) and report coverage metrics honestly. This reduces
   the claim but preserves validity.

### B.6 Timeline with Kill Gates

| Week | Milestone | Kill gate |
|---|---|---|
| 1 | Hypergraph Laplacian construction + eigensolve on 50 pilot instances | Eigendecomposition converges on ≥90% of pilot |
| 2 | Spectral feature extraction + correlation with bound degradation | **G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4** |
| 3 | GCG + SCIP Benders wrappers; ground-truth labels for pilot | G2: Wrappers produce valid bounds on ≥80% of pilot |
| 4 | Syntactic + graph features; feature robustness validation | Permutation-invariance CV <5% on all spectral features |
| 5-6 | 200-instance dev set: full feature extraction + ablation | **G3: Spectral features improve selector accuracy by ≥3pp over syntactic-only** |
| 7-8 | 500-instance stratified evaluation: ablation with nested CV | **G4: Combined metrics pass; p < 0.05 on McNemar's test** |
| 9-10 | L3 numerical verification; T2 correlation analysis | L3 bound holds within 2× on ≥80% of instances |
| 11-12 | Paper draft; artifact preparation | G5: Internal review pass |

### B.7 What Could Go Wrong

| Risk | Probability | Recovery |
|---|---|---|
| Eigendecomposition fails on large instances (>100K rows) | Medium | Restrict to instances with $m < 50\text{K}$; report coverage. Use randomized SVD for larger instances. |
| Spectral features highly correlated with syntactic | Medium-High | Feature is still novel *definition*; report correlation analysis honestly. Pivot to "spectral features are computable proxies for syntactic features with better theoretical grounding." |
| $\delta^2/\gamma^2$ has ρ < 0.4 with bound degradation (G1 fails) | Medium | If ρ ≥ 0.25, investigate nonlinear relationship (Kendall τ, mutual information). If ρ < 0.25, spectral premise is invalid — pivot to census-only (Approach A). |
| Too few instances with exploitable structure | Medium | Futility prediction becomes the main finding: "most MIPs don't benefit from decomposition" is itself publishable. |
| Reviewer says "just kernel features on $A$" | Medium | Pre-empt by including RBF kernel features on $A$ as a baseline in ablation; show spectral features capture different information. |

### B.8 Scores

| Dimension | Score | Rationale |
|---|---|---|
| **Value** | 7/10 | New feature family for MIP instance characterization; generalizes beyond this paper |
| **Difficulty** | 5/10 | Hypergraph Laplacian at scale is genuinely novel; ablation methodology is standard |
| **Best-Paper** | 4/10 | Strong JoC contribution if ablation is clean; not top-venue best-paper |
| **Feasibility** | 7/10 | G1 kill gate is the main risk; eigendecomposition scalability is secondary |

---

## Approach C: Oracle-System (Unified Reformulation Selector)

### C.1 Extreme Value

**Who needs this desperately**: Industrial MIP users (airline scheduling, supply-chain
optimization, energy grid planning) who solve the *same model class* weekly but don't know
whether Benders, DW, or Lagrangian would accelerate convergence — and cannot afford to try
all three. Today, the choice is made by a human expert based on intuition, or not made at all
(monolithic solve). The expert's time costs $200/hour; a wrong decomposition choice wastes
8-48 hours of compute. A working oracle that says *"use Benders with this partition"* or
*"don't decompose"* with 80%+ accuracy saves thousands of dollars per model class per year.

Unlike Approaches A and B, which produce *research artifacts*, Approach C produces a
*usable tool*: a command-line utility that takes an MPS file and outputs a decomposition
recommendation with a confidence score. This is the difference between a dataset paper and a
systems paper.

**Primary users**:
- Industrial optimization teams with recurring MIP classes
- Solver developers wanting an automatic decomposition preprocessor
- Research groups benchmarking new decomposition methods against an automated baseline

### C.2 Genuine Software Difficulty

| Subproblem | Why it's hard | Approach |
|---|---|---|
| **End-to-end pipeline latency** | The oracle must run in <60s (preprocessing overhead) to be practical. Hypergraph Laplacian construction + eigendecomposition + feature extraction + ML prediction + partition generation must all fit within this budget for instances with up to 100K constraints. | Lazy evaluation: construct Laplacian incrementally; use randomized eigensolve (Halko-Martinsson-Tropp) for $k$-SVD in $O(mk)$ time; precompile feature extraction in Cython/Numba; cache model for ML prediction |
| **Partition injection into solvers** | GCG accepts decomposition via `.dec` files; SCIP Benders accepts via custom constraint handler; Lagrangian requires explicit subproblem construction. Each solver has a different API and a different contract for what constitutes a valid partition. | Unified `Partition` class that serializes to: (a) GCG `.dec` format, (b) SCIP Benders variable partition, (c) Lagrangian constraint partition. Validation layer ensures partition is solver-feasible before dispatch. |
| **Adaptive partition refinement** | A spectral partition (from Fiedler vector thresholding) may not align with solver-preferred structure. E.g., GCG prefers partitions where the master has few linking constraints; Benders prefers partitions where the subproblem is an LP. | Method-aware refinement: after initial spectral partition, run a 10-iteration local search that moves rows/variables between blocks to optimize a method-specific objective (coupling constraint count for DW, continuous-subproblem fraction for Benders). |
| **Futility prediction with calibration** | The oracle must not only predict the best method but also predict *whether decomposition helps at all*. A miscalibrated futility predictor that says "don't decompose" on instances where DW gives 30% gap improvement is worse than no oracle. | Train a separate binary classifier (decompose vs. don't) with asymmetric loss: false-negative (missing a good decomposition) costs 3× false-positive. Calibrate via Platt scaling. Report calibration curves in paper. |
| **Confidence estimation** | Users need to know *how confident* the oracle is. A raw softmax from a tree ensemble is poorly calibrated. | Use conformal prediction to produce valid prediction sets: "with 90% probability, the best method is in {DW, Lagrangian}." This gives the user a menu of safe choices rather than a single point prediction. |

**Architecture**:
```
oracle/
  core/
    pipeline.py                # End-to-end: MPS → recommendation
    config.py                  # Runtime configuration
  spectral/
    laplacian.py               # Hypergraph Laplacian (sparse, incremental)
    eigensolve.py              # Randomized SVD + ARPACK fallback
    features.py                # 8 spectral features
  features/
    syntactic.py               # 25 syntactic features
    graph.py                   # 10 graph features
    combined.py                # Feature vector assembly
  partition/
    spectral_partition.py      # Fiedler bisection + recursive
    refinement.py              # Method-aware local search (10 iters)
    validation.py              # Solver-feasibility checks
    serializers.py             # → .dec (GCG), → SCIP Benders, → Lagrangian blocks
  prediction/
    selector.py                # 4-class RF/XGBoost: {Benders, DW, Lagrangian, none}
    futility.py                # Binary classifier with asymmetric loss
    conformal.py               # Conformal prediction sets
    calibration.py             # Platt scaling, reliability diagrams
  dispatch/
    gcg_driver.py              # GCG subprocess with .dec injection
    scip_benders_driver.py     # PySCIPOpt Benders
    lagrangian_driver.py       # ConicBundle / subgradient
    result_collector.py        # Unified result schema
  census/
    pipeline.py                # Census generation (serves as training data + evaluation)
    labeling.py                # Ground-truth from solver runs
  experiments/
    ablation.py                # Feature ablation
    system_eval.py             # End-to-end oracle vs. baselines
    latency_bench.py           # Oracle overhead benchmarking
  cli/
    main.py                    # $ oracle recommend instance.mps
    report.py                  # Human-readable recommendation report
```

**LoC Breakdown**:
| Subsystem | New LoC | Notes |
|---|---|---|
| Spectral engine (Laplacian + eigensolve + features) | 4,500 | Core spectral code |
| Feature extraction (syntactic + graph + combined) | 2,500 | Baseline features + assembly |
| Partition generation + refinement | 3,000 | Spectral partition + method-aware refinement |
| Prediction (selector + futility + conformal) | 3,500 | ML pipeline with calibration |
| Solver dispatch (3 drivers + result collection) | 3,000 | GCG + SCIP Benders + Lagrangian |
| Census pipeline + labeling | 2,500 | Training data generation |
| CLI + report generation | 1,000 | User-facing tool |
| Experiments (ablation + system eval + latency) | 2,500 | Evaluation harness |
| Tests | 2,500 | Unit + integration + property-based |
| **Total** | **25,500** | At upper bound of 25-30K constraint |

### C.3 New Math Required

**Most math of the three approaches — but all load-bearing.**

1. **Lemma L3 (Partition-to-Bound Bridge)** — as in Approach B, but extended:
   > **Corollary L3-C (Method-Specific Bounds)**: For a Benders partition $(y, x)$ where $y$ are
   > complicating variables:
   > $$z_{LP} - z_{\text{Benders}}^{\text{master}} \leq \sum_{j: y_j \text{ couples blocks}} |r_j^*| \cdot |\text{blocks}(j)|$$
   > where $r_j^*$ is the reduced cost of $y_j$ in the monolithic LP.
   >
   > For a DW partition with $k$ blocks:
   > $$z_{LP} - z_{\text{DW}}^{\text{master}} \leq \sum_{i \in \text{linking}} |\mu_i^*| \cdot (k-1)$$
   > where $\mu_i^*$ is the dual value of linking constraint $i$.

   **Role**: These method-specific bounds guide the partition refinement heuristic. The
   refinement minimizes the relevant bound by moving rows/variables between blocks.

2. **Proposition T2 (Motivational)** — identical caveat as Approach B. Here it additionally
   motivates the futility predictor: if $\delta^2/\gamma^2$ is large, the spectral ratio
   predicts poor decomposition quality, triggering the "none" recommendation.

3. **Proposition C1 (Conformal Coverage Guarantee)** — standard conformal prediction result
   adapted to the decomposition-selection setting:
   > Let $\{(x_i, y_i)\}_{i=1}^n$ be exchangeable feature-label pairs and $\hat{C}(x)$ be the
   > conformal prediction set at level $\alpha$. Then $P(y_{n+1} \in \hat{C}(x_{n+1})) \geq 1-\alpha$.
   > **Applied to oracle**: With $\alpha = 0.1$, the oracle's prediction set contains the true
   > best method with ≥90% probability.

   **Role**: This is not new math — it's a known result — but it must be formally stated in
   context to justify the confidence estimation. The novelty is in the *application* to
   reformulation selection.

4. **Proposition F3 (Partition Refinement Convergence)**:
   > The method-aware local search (Algorithm 1) reduces the L3 bound monotonically and
   > terminates in at most $m$ iterations (where $m$ is the number of constraints), with
   > each iteration costing $O(m \cdot d_{\max})$ time.
   > *Proof sketch*: Each iteration moves one row to a block that reduces its coupling
   > contribution. The L3 bound is a non-negative sum; each move strictly reduces one term.
   > Termination follows from finiteness. Practical convergence is ≤10 iterations on all
   > tested instances.

   **Role**: Justifies why the refinement adds negligible overhead (<1s) to the partition step.

### C.4 Best-Paper Potential

**Highest ceiling of the three approaches.** If the oracle achieves ≥75% accuracy on 4-class
selection AND the futility predictor has ≥85% precision AND the end-to-end system demonstrates
measurable solver speedups on a held-out test set, this is a *working system* paper that solves
a real problem. JoC loves system papers with reproducible artifacts and strong empirical results.

**Best-paper path**: The oracle is packaged as a pip-installable tool that any MIP user can run.
The paper shows: (1) spectral features carry unique information, (2) the oracle outperforms
GCG's method selection, (3) the futility predictor saves wasted compute on ~40% of instances,
(4) conformal prediction sets give valid coverage. A reviewer can install the tool, run it on
their own instance, and see the recommendation. This kind of *immediately usable artifact* is
rare in optimization and is what JoC rewards.

**What makes it publishable regardless**: Even if the oracle is only marginally better than
"always use GCG," the system contribution (cross-method selection with futility detection and
confidence estimation) is novel and the census + features are still valuable artifacts.

### C.5 Hardest Technical Challenge

**Training data scarcity for the 4-class selector.** The label distribution will be heavily
imbalanced: most MIPLIB instances either (a) have no exploitable structure (label "none") or
(b) are DW-amenable (because DW is the most general decomposition). The Benders and Lagrangian
classes may have <30 instances each in a 500-instance sample. Training a 4-class classifier
with <30 minority-class examples is unreliable.

**How to address it**:
1. **Hierarchical classification**: First predict {decompose, don't} (binary, well-balanced).
   Then, conditional on "decompose," predict {Benders, DW, Lagrangian} (3-class, smaller but
   more balanced within structured instances).
2. **SMOTE on spectral features**: The continuous nature of spectral features makes synthetic
   oversampling more reasonable than on discrete syntactic features.
3. **Cost-sensitive learning**: Weight minority classes inversely to frequency in the loss function.
4. **Reduced label set**: If Lagrangian is dominated by DW on >90% of instances, merge them into
   a single "column-generation-amenable" class. This 3-class problem ({Benders, CG-amenable,
   none}) may be much more tractable.
5. **Instance augmentation**: For each MIPLIB instance, generate 2-3 perturbations (coefficient
   scaling, constraint sampling) to increase effective training set size. Validate that
   perturbations preserve structure labels.
6. **Fallback**: If 4-class is hopeless, pivot to *ranking*: predict a permutation over
   {Benders, DW, Lagrangian, none} and evaluate using NDCG@1. This is strictly easier than
   classification and still useful.

### C.6 Timeline with Kill Gates

| Week | Milestone | Kill gate |
|---|---|---|
| 1 | Spectral engine: Laplacian + eigensolve on 50 pilot instances | Eigensolve converges on ≥90%; latency <30s per instance |
| 2 | Feature extraction + correlation with bound degradation | **G1: Spearman ρ(δ²/γ², bound degradation) ≥ 0.4** |
| 3 | GCG + SCIP Benders wrappers; ground-truth labels for pilot | G2: Wrappers produce valid bounds on ≥80% |
| 4 | Partition generation + method-aware refinement | Refinement reduces L3 bound on ≥70% of structured instances |
| 5 | Futility predictor on 200-instance dev set | Futility precision ≥80% |
| 6-7 | 4-class selector + conformal prediction | **G3: Selector accuracy >55% (vs. 25% random)** on dev set |
| 8-9 | 500-instance stratified evaluation: full oracle pipeline | **G4: All combined metrics pass** |
| 10 | End-to-end system eval: oracle vs. GCG-only vs. monolithic | Oracle ≥ GCG on ≥60% of decomposable instances |
| 11-12 | Paper draft; CLI tool packaging; artifact preparation | G5: Internal review pass |

### C.7 What Could Go Wrong

| Risk | Probability | Recovery |
|---|---|---|
| 4-class label imbalance makes classifier unreliable | High | Pivot to hierarchical (decompose/don't → which method) or ranking |
| End-to-end latency exceeds 60s on large instances | Medium | Use randomized SVD; restrict to $m < 50\text{K}$; report latency-accuracy tradeoff |
| Partition refinement doesn't improve over raw spectral partition | Medium | Drop refinement module; use spectral partition directly; saves ~3K LoC |
| Oracle doesn't beat "always use GCG" on DW instances | Medium | Oracle's value is cross-method + futility; GCG handles only DW. Frame as "oracle extends GCG to Benders and adds futility." |
| Conformal prediction sets are too wide (avg size >2.5/4 classes) | Medium | Drop conformal; report standard confidence calibration instead. Prediction sets >2.5 are uninformative. |
| 25K LoC is tight; integration bugs consume schedule | Medium-High | Deprioritize Lagrangian dispatch (keep as "future work"); saves ~2K LoC and removes hardest wrapper |
| Reviewer questions system-paper framing | Medium | Census + features are fallback contributions (degrades to Approach B) |

### C.8 Scores

| Dimension | Score | Rationale |
|---|---|---|
| **Value** | 8/10 | Working tool that solves a real problem; pip-installable oracle |
| **Difficulty** | 7/10 | End-to-end system with 6 interacting subsystems; label scarcity; latency constraints |
| **Best-Paper** | 5/10 | Highest ceiling; system papers with artifacts do well at JoC |
| **Feasibility** | 5/10 | Most things must work for the paper to hold together; many failure modes |

---

## Comparative Summary

| Dimension | A: Census-Heavy | B: Spectral-Feature-First | C: Oracle-System |
|---|---|---|---|
| **Lead contribution** | MIPLIB decomposition census | Spectral features for MIP characterization | Working reformulation selection tool |
| **New math** | L3 (standalone) + T2 (motivational) | L3 + F1 + F2 + T2 (motivational) | L3 + L3-C + F3 + C1 + T2 (motivational) |
| **Core experiment** | Cross-method census statistics | Feature-family ablation | End-to-end oracle evaluation |
| **New LoC** | ~17K | ~17-20K | ~25K |
| **Value** | 6 | 7 | 8 |
| **Difficulty** | 4 | 5 | 7 |
| **Best-Paper** | 2 | 4 | 5 |
| **Feasibility** | 8 | 7 | 5 |
| **Risk profile** | Low risk, low ceiling | Medium risk, medium ceiling | Higher risk, higher ceiling |
| **Degradation path** | Already minimal | Degrades to A (census-only) | Degrades to B (features) or A (census) |
| **JoC fit** | Dataset / benchmark paper | Computational study | Systems paper |

### Decision Framework

- **Choose A** if: team is risk-averse, compute-limited, or wants a guaranteed publication
  with community service contribution. Fastest path to submission.
- **Choose B** if: team believes spectral features carry novel information and wants to make
  a methodological contribution to MIP instance characterization. Best risk-reward tradeoff.
- **Choose C** if: team has strong engineering capacity and wants the highest-impact paper.
  Requires all subsystems to work; but if they do, this is the most citable paper and the
  most useful artifact.

### Recommended Strategy

**Start with B's infrastructure (Weeks 1-6), evaluate at G3, then decide:**
- If G3 fails (spectral features ≤ syntactic) → pivot to A (census paper)
- If G3 passes marginally (+3-5pp) → stay with B (feature paper)
- If G3 passes strongly (+8pp or more) → extend to C (oracle system)

This strategy respects the kill gates while preserving optionality. The spectral engine and
census infrastructure built in Weeks 1-4 are shared across all three approaches, so no work
is wasted regardless of the pivot decision.
