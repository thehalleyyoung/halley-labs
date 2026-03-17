# Final Approach — Spectral Decomposition Oracle

> **Project**: spectral-decomposition-oracle
> **Phase**: Ideation → Implementation (binding specification)
> **Date**: 2025-07-22
> **Status**: APPROVED — synthesized from Approaches A/B/C, three critiques, depth check
> **Binding constraints**: All 7 depth-check amendments; all fatal/serious flaws from critiques

---

## 0. Title and Abstract

**Title**: *Spectral Features for MIP Decomposition Selection: A Computational Study with the First MIPLIB 2017 Decomposition Census*

**Abstract**: We introduce spectral features extracted from the constraint hypergraph Laplacian — spectral gaps, eigenvector localization, algebraic connectivity, and coupling energy — as a new continuous, geometry-aware feature family for characterizing mixed-integer program (MIP) instances. We evaluate these features via a rigorous ablation study on a stratified 500-instance subset of MIPLIB 2017, demonstrating that spectral features improve decomposition method selection accuracy by ≥5 percentage points over syntactic features alone (or honestly reporting the margin if smaller). The core experiment compares spectral, syntactic, and combined feature sets at matched feature budgets using nested cross-validation with external decomposition backends (GCG for Dantzig–Wolfe, SCIP-native Benders). We provide a spectral futility predictor that identifies instances where no block decomposition will improve the dual bound. As a community artifact, we release the first systematic decomposition census of MIPLIB 2017: spectral structural annotations for all 1,065 instances, and decomposition evaluation results for 500 stratified instances. Theoretical grounding is provided by Lemma L3 (a partition-to-bound bridge bounding dual gap by crossing weight), its method-specific specializations L3-C for Benders and Dantzig–Wolfe, and a motivational spectral scaling law (Proposition T2) explaining why the spectral ratio δ²/γ² predicts decomposition quality. If spectral features pass the G3 gate strongly, we extend to a lightweight 2-method oracle that injects spectral partitions into GCG and SCIP Benders and recommends a decomposition strategy via independent binary classifiers.

---

## 1. Contributions (Ranked)

### Primary Contributions

**P1. Spectral feature family for MIP instance characterization.**
Eight spectral features from the constraint hypergraph Laplacian, formally defined with permutation-invariance guarantees (F1) and scaling-sensitivity analysis (F2, corrected). Validated via feature-count-controlled ablation against syntactic, graph-based, and (discussed) GNN-learned features. This is the scientific thesis of the paper.

**P2. MIPLIB 2017 decomposition census.**
Spectral structural annotations for all 1,065 instances (~9 hours compute). Decomposition evaluation (Benders via SCIP-native, Dantzig–Wolfe via GCG) on 500 stratified instances at multiple time cutoffs (60s, 300s, 900s). Released as open Parquet dataset with provenance. First systematic cross-method decomposition evaluation on the standard MIP benchmark.

**P3. Lemma L3 and L3-C (partition-to-bound bridge).**
A formal bound on the LP-vs-decomposed-dual gap in terms of crossing hyperedge weight, with method-specific specializations for Benders (reduced-cost weighting) and Dantzig–Wolfe (linking-constraint dual weighting). Standalone value for evaluating any partition — GCG's, manual, or spectral.

### Secondary Contributions

**S1. Spectral futility predictor.**
An empirically calibrated binary classifier that predicts when no k-block decomposition will improve the dual bound beyond tolerance ε. Trained on census data, evaluated by precision/recall at multiple thresholds.

**S2. Cross-method reformulation selection framing.**
Formal distinction between algorithm selection ("which solver?") and reformulation selection ("which mathematical object should the solver see?"). Novel problem formulation extending Kruber et al. (2017) from DW-only to cross-method.

### Conditional Contribution (if G3 passes strongly)

**C1. Lightweight 2-method oracle (C-lite extension).**
Binary classifiers for Benders-amenability and DW-amenability, with spectral partition injection into GCG (.dec format) and SCIP Benders (variable partition). End-to-end recommendation: MPS → spectral features → classify → partition → dispatch. Docker research prototype, not pip-installable.

### Motivational (not a contribution)

**M1. Proposition T2 (Spectral Scaling Law).**
$z_{LP} - z_D(\hat{\pi}) \leq C \cdot \delta^2/\gamma^2$ where $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$. Vacuous on ~60–70% of MIPLIB. Presented in ≤2 pages as structural motivation for why δ²/γ² is a principled predictor feature.

---

## 2. Architecture

```
spectral-decomposition-oracle/
│
├── spectral/                          # CORE SPECTRAL ENGINE (~6,500 LoC)
│   ├── hypergraph/
│   │   ├── laplacian.py               # Constraint matrix → sparse Laplacian
│   │   ├── incidence.py               # Bolla (1993) incidence-matrix Laplacian (no clique expansion)
│   │   ├── clique_expansion.py        # Clique expansion for d_max ≤ 200 (exact)
│   │   ├── normalization.py           # Degree-weighted, symmetric, random-walk variants
│   │   └── preprocessing.py           # Ruiz / SCIP-native / geometric-mean scaling
│   ├── eigensolve/
│   │   ├── solver.py                  # ARPACK shift-invert + LOBPCG fallback chain
│   │   ├── cache.py                   # HDF5 eigendecomposition cache
│   │   └── diagnostics.py             # Convergence checks, condition monitoring
│   └── features/
│       ├── spectral_features.py       # 8 spectral features (formal definitions below)
│       ├── robustness.py              # Permutation-invariance + scaling-robustness validation
│       └── feature_pipeline.py        # Instance → spectral feature vector
│
├── features/                          # BASELINE FEATURES (~2,500 LoC)
│   ├── syntactic_features.py          # 25 syntactic features (density, degree stats, coeff range)
│   ├── graph_features.py              # 10 variable-interaction graph features
│   ├── kruber_features.py             # Kruber et al. (2017) feature set reimplementation
│   └── combined.py                    # Feature vector assembly + normalization
│
├── census/                            # CENSUS INFRASTRUCTURE (~4,500 LoC)
│   ├── harness/
│   │   ├── scip_benders_wrapper.py    # PySCIPOpt Benders adapter (variable partition input)
│   │   ├── gcg_dw_wrapper.py          # GCG subprocess + .dec format driver
│   │   └── result_schema.py           # Shared DecompositionResult dataclass
│   ├── pipeline/
│   │   ├── job_queue.py               # SQLite-backed idempotent job management
│   │   ├── runner.py                  # Instance → detect → decompose → log
│   │   └── aggregator.py              # Result collection, coverage metrics
│   ├── labeling/
│   │   ├── ground_truth.py            # Dual-bound comparison → label at {60s, 300s, 900s}
│   │   ├── label_stability.py         # Cross-cutoff label flip analysis
│   │   └── consensus_labels.py        # Majority vote across cutoffs if flip rate > 20%
│   └── detectors/
│       ├── dw_detector.py             # Delegates to GCG detect
│       └── benders_detector.py        # Variable-participation heuristic
│
├── prediction/                        # ML PIPELINE (~2,500 LoC)
│   ├── classifiers.py                 # RF, XGBoost, LogReg with hyperparameter grids
│   ├── futility.py                    # Binary futility predictor (asymmetric loss)
│   ├── benders_classifier.py          # Binary: Benders-amenable? (for C-lite)
│   ├── dw_classifier.py               # Binary: DW-amenable? (for C-lite)
│   └── calibration.py                 # Platt scaling + reliability diagrams
│
├── partition/                         # PARTITION INJECTION — C-LITE (~2,000 LoC)
│   ├── spectral_partition.py          # Fiedler bisection + recursive multiway
│   ├── refinement.py                  # Method-aware greedy local search (≤10 iters)
│   ├── serializers.py                 # → .dec (GCG), → SCIP Benders variable partition
│   └── validation.py                  # Solver-feasibility checks before dispatch
│
├── experiments/                       # EVALUATION HARNESS (~3,000 LoC)
│   ├── ablation.py                    # Feature-family ablation with nested CV
│   ├── statistical_tests.py           # McNemar, Wilcoxon, confidence intervals
│   ├── feature_budget.py              # Top-k spectral vs. top-k syntactic (k=3,5,8)
│   ├── correlation_analysis.py        # Feature intercorrelation, PCA effective rank
│   ├── scaling_robustness.py          # Features under 3 scaling methods (Spearman ρ)
│   ├── presolve_comparison.py         # Pre-presolve vs. post-presolve features
│   ├── performance_profiles.py        # Dolan-Moré profiles for solver comparison
│   └── visualization.py              # Heatmaps, confusion matrices, calibration curves
│
├── theory/                            # THEORETICAL VERIFICATION (~1,500 LoC)
│   ├── l3_verification.py             # Numerical verification of L3 on census instances
│   ├── l3c_verification.py            # L3-C (Benders/DW specializations) numerics
│   └── t2_correlation.py              # Empirical correlation of δ²/γ² with bound degradation
│
├── tests/                             # TESTS (~3,500 LoC)
│   ├── test_laplacian.py              # Known-spectrum instances, clique expansion correctness
│   ├── test_eigensolve.py             # Convergence on synthetic + real instances
│   ├── test_features.py               # Permutation invariance (property-based via hypothesis)
│   ├── test_scaling.py                # Feature stability under 3 scaling methods
│   ├── test_wrappers.py               # GCG + SCIP Benders on 5 small instances
│   ├── test_partition.py              # Serializer roundtrip + solver acceptance
│   └── test_integration.py            # End-to-end on 3 curated instances
│
└── Dockerfile                         # Reproducible environment (SCIP + GCG + PySCIPOpt)
```

### LoC Breakdown (Honest, Critique-Adjusted)

| Subsystem | New LoC | Source | Notes |
|---|---|---|---|
| Spectral engine (Laplacian + eigensolve + features) | 6,500 | B core, expanded per Difficulty Assessor | Incidence-matrix Laplacian adds ~1,500 over Approach B's estimate |
| Baseline features (syntactic + graph + Kruber) | 2,500 | B + Skeptic's B-S4 fix | Kruber et al. reimplementation adds ~500 |
| Census infrastructure (wrappers + pipeline + labeling) | 4,500 | A core, trimmed | No Lagrangian; multi-cutoff labeling adds ~500 |
| ML pipeline (classifiers + futility + calibration) | 2,500 | B prediction + C binary classifiers | No conformal; binary classifiers for C-lite |
| Partition injection (C-lite, conditional) | 2,000 | C partition, trimmed per Difficulty Assessor | Refinement budget: ~400 LoC (not 3,000). No Lagrangian serializer. |
| Evaluation harness | 3,000 | B experiments, expanded per Skeptic | Feature-budget comparison, scaling robustness, presolve analysis |
| Theory verification | 1,500 | B theory + C's L3-C | L3 + L3-C numerics + T2 correlation |
| Tests | 3,500 | Difficulty Assessor estimate (higher than all proposals) | Property-based + integration; 3 solver backends |
| **Total** | **26,500** | Within 25–30K target | 3,500 LoC buffer to 30K cap |

---

## 3. Spectral Feature Definitions

The following 8 features are extracted from the constraint hypergraph Laplacian $L_H$ after equilibration preprocessing. All definitions assume $L_H$ has been constructed from the constraint matrix $A$ (see §4 for Laplacian construction details).

| # | Feature | Definition | Rationale |
|---|---------|-----------|-----------|
| 1 | **Spectral gap** $\gamma_2$ | $\lambda_2(L_H)$, second-smallest eigenvalue of the normalized Laplacian | Algebraic connectivity; large gap → well-separated blocks |
| 2 | **Spectral gap ratio** $\delta^2/\gamma^2$ | Coupling energy / spectral gap squared (T2's predictor) | Directly from perturbation theory; predicts partition quality |
| 3 | **Eigenvalue decay rate** | Slope of exponential fit to $\lambda_2, \ldots, \lambda_{k+1}$ | Fast decay → few dominant blocks; slow decay → distributed coupling |
| 4 | **Fiedler vector localization entropy** | $H = -\sum_i |v_i|^2 \log |v_i|^2$ where $v$ is the Fiedler vector | Localized Fiedler vector → clear block boundary |
| 5 | **Algebraic connectivity ratio** $\gamma_2/\gamma_k$ | Ratio of 2nd to $k$-th smallest eigenvalue | Near 1 → clean $k$-way partition; near 0 → hierarchical structure |
| 6 | **Coupling energy** $\delta^2$ | $\|L_H - L_{\text{block-diag}}\|_F^2$ (from spectral clustering) | Direct measure of inter-block interaction strength |
| 7 | **Block separability index** | Silhouette score of $k$-means on bottom-$k$ eigenvectors (using $VV^T$ for rotation invariance) | Quality of spectral partition; handles repeated-eigenvalue ambiguity via Gram matrix |
| 8 | **Effective spectral dimension** | Number of eigenvalues < $\lambda_{\text{median}}/10$ | Indicates number of exploitable block components |

**Proposition F1 (Permutation Invariance):** All 8 features are invariant under simultaneous row/column permutations of $A$. Features 1–6 and 8 depend only on eigenvalues (similarity-invariant). Feature 7 uses the Gram matrix $VV^T$ (invariant to eigenvector sign and rotation within degenerate eigenspaces). *Proof effort: 0.5 days. Subtlety: state the rotation-invariance fix for repeated eigenvalues explicitly.*

**Proposition F2 (Scaling Sensitivity) — CORRECTED:** Under row scaling $A \to DA$ where $D = \text{diag}(d_1, \ldots, d_m)$ with $d_i > 0$:
$$\gamma_2(L_A) / \kappa(D) \leq \gamma_2(L_{DA}) \leq \gamma_2(L_A) \cdot \kappa(D)$$
where $\kappa(D) = \|D\|_2 \cdot \|D^{-1}\|_2$ is the condition number of $D$.

**Consequence:** Spectral features are sensitive to constraint scaling. Mitigation: (1) Apply equilibration preprocessing before Laplacian construction; (2) Report features under 3 scaling methods (Ruiz, SCIP-native, geometric mean) and verify cross-scaling Spearman $\rho > 0.9$; (3) If features are not robust, restrict to ratio features (γ₂/γ_k, δ²/γ²) which have reduced scaling sensitivity. *Proof effort: 2–3 days; requires pinning Laplacian definition (incidence-based vs. clique-expansion) and specifying weighted vs. unweighted adjacency.*

**Feature intercorrelation:** Report PCA effective rank and full correlation matrix of spectral features across the 500-instance sample. If effective dimensionality ≤ 3, state honestly: "8 spectral features, of which ~3 are approximately independent." This is still a valid contribution — the definitions are reproducible and the theoretical grounding remains.

---

## 4. The Clique-Expansion Problem: Resolution

**The problem:** The clique-expansion Laplacian creates $\binom{d}{2}$ edges per hyperedge of degree $d$. For MIPLIB instances with $d_{\max} = 500$, this produces ~125K edges per constraint — intractable for instances with thousands of such constraints. All three critics identify this as the #1 technical risk.

**Resolution: Two-tier Laplacian construction.**

| Instance regime | Laplacian method | Theoretical backing | Approximation guarantee |
|---|---|---|---|
| $d_{\max} \leq 200$ (~85% of MIPLIB) | Exact clique expansion | Standard; all theoretical results (L3 spectral bound, T2) apply directly | Exact |
| $d_{\max} > 200$ (~15% of MIPLIB) | Incidence-matrix Laplacian (Bolla 1993; Zhou et al. 2006) | $L_I = H W H^T - D_v$ where $H$ is the $m \times n$ incidence matrix and $W$ is the diagonal hyperedge weight matrix | No clique expansion; avoids quadratic blowup. Spectral properties differ from clique expansion but are well-studied. |

**Implementation:**
- For the incidence-matrix Laplacian: $L_I = H W H^T - D_v$ where $H_{ie} = 1$ if vertex $i \in e$, $W_{ee}$ is the hyperedge weight, and $D_v = \text{diag}(\sum_e W_{ee} H_{ie})$. This is an $m \times m$ sparse matrix with $O(\text{nnz}(A))$ entries — no quadratic blowup.
- The eigensolve uses the same ARPACK/LOBPCG pipeline on $L_I$.
- Feature definitions (§3) transfer directly: $\gamma_2(L_I)$ replaces $\gamma_2(L_H)$ for high-degree instances.

**Impact on theoretical results:**
- L3's spectral partition bound must be restated for $L_I$ instead of the clique-expansion Laplacian. The Cheeger-type inequality takes a different form (Chan et al. 2018). This adds ~2 days of proof effort.
- T2's qualitative scaling argument is unaffected (the perturbation model is independent of the specific Laplacian).
- Report which Laplacian was used per-instance in the census artifact.

**Validation:** On the 50-instance pilot (Week 2), compute features from both Laplacian variants on instances where both are tractable ($d_{\max} \leq 200$). Verify Spearman $\rho > 0.85$ between feature vectors. If disagreement is large, investigate and document.

**LoC cost:** The incidence-matrix Laplacian adds ~1,500 LoC (incidence.py + adapted normalization). This is already budgeted in the 6,500 LoC spectral engine estimate.

---

## 5. Complete Math Inventory

| # | Item | Statement | Load-bearing? | Correctness status | Proof effort | Risk |
|---|------|-----------|---------------|-------------------|-------------|------|
| **L3** | Partition-to-Bound Bridge | For partition $P$ with crossing set $E_{\text{cross}}$: $z_{LP} - z_D(P) \leq \sum_{e \in E_{\text{cross}}} \|y^*_e\| \cdot (n_e - 1)$ where $y^*$ is the monolithic LP dual and $n_e$ is blocks spanned by hyperedge $e$ | **Yes** — quality metric for any partition | Sound in principle; must specify which dual bound (Lagrangian dual at LP relaxation); must note LP gap not IP gap | 3–5 days | Low |
| **L3-sp** | Spectral Partition Bound | For spectral partition $P^*$ from bottom-$k$ eigenvectors of $L_H$: $\sum_{e \in E_{\text{cross}}(P^*)} w(e) \leq O(\delta^2 \cdot d_{\max} / \gamma_2^2)$ | **Yes** — connects spectral features to partition quality, completing the chain features → partition → bound | **$d_{\max}$ factor added** per Math Assessor; must specify normalized Laplacian; must handle clique-expansion vs. incidence Laplacian separately | 7–10 days | **Medium-High** — if clique expansion distortion requires a separate lemma, could reach 12 days |
| **L3-C (Benders)** | Method-specific Benders bound | $z_{LP} - z_{\text{Benders}}^{(t)} \leq \sum_{j: y_j \text{ couples}} \|r_j^{(t)}\| \cdot \|\text{blocks}(j)\|$ where $r_j^{(t)}$ is the reduced cost in the *current* Benders master (not monolithic LP) at iteration $t$ | **Yes** — guides Benders partition refinement | **Corrected**: uses current master duals, not monolithic LP duals; bound is for partial Benders (before convergence); stated as motivational for refinement, not as a tight bound | 2–3 days | Low-Medium |
| **L3-C (DW)** | Method-specific DW bound | $z_{LP} - z_{\text{DW}}^{(t)} \leq \sum_{i \in \text{linking}} \|\mu_i^{(t)}\| \cdot (\|\text{blocks}(i)\| - 1)$ where $\mu_i^{(t)}$ is the DW master dual at iteration $t$ | **Yes** — most novel math item; guides DW partition refinement | Sound; uses DW master duals (correct object); $(|\text{blocks}(i)| - 1)$ tightens the original $(k-1)$ factor | 2–3 days | Low |
| **F1** | Permutation Invariance | All 8 spectral features are invariant under simultaneous row/column permutation of $A$ | **Yes** — correctness requirement | Correct; uses Gram matrix $VV^T$ for rotation invariance of eigenvector-derived features | 0.5 days | Low |
| **F2** | Scaling Sensitivity (CORRECTED) | $\gamma_2(L_A)/\kappa(D) \leq \gamma_2(L_{DA}) \leq \gamma_2(L_A) \cdot \kappa(D)$ | **Yes** — motivates equilibration preprocessing | **Corrected**: uses $\kappa(D)$ not $\|D\|_2^2$; must specify Laplacian definition (incidence-based vs. clique expansion); must state for weighted adjacency | 2–3 days | Low (after correction) |
| **T2** | Spectral Scaling Law (MOTIVATIONAL) | $z_{LP} - z_D(\hat\pi) \leq C \cdot \delta^2/\gamma^2$ where $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$ | **Motivational only** — explains why δ²/γ² is a natural predictor; never presented as quantitative tool | Qualitatively correct; $C$ is vacuous for $\kappa > 10^3$; Davis–Kahan chain assumes bisection ($k=2$); for $k > 2$, relevant gap is $\lambda_{k+1} - \lambda_k$, not $\gamma_2$ | 5–7 days | Medium — hidden $k$-way assumption must be stated |
| **F3** | Refinement Convergence (C-LITE ONLY) | Method-aware local search converges to a local minimum of the L3 bound in ≤ $m$ iterations, each costing $O(m \cdot d_{\max})$ | **Yes (if C-lite activated)** — justifies 10-iteration budget | Convergence to local minimum (not strict monotonicity); uses approximate duals (from a few Simplex iterations), not $y^*$; stated honestly as "approximate monotonicity" | 1–2 days | Low |

**Total proof effort: 18–26 person-days.** Critical path: L3-sp (7–10 days) + T2 (5–7 days) = 12–17 days on the longest chain. Schedule proof sprints in Weeks 1–4, parallel with engineering.

**What is NOT claimed:**
- We do NOT claim T2 is a tight or useful quantitative bound.
- We do NOT claim a formal futility certificate — it is a learned predictor.
- We do NOT claim exchangeability of MIPLIB instances (conformal prediction dropped).
- We do NOT claim the spectral partition bound holds for approximate Laplacians without the incidence-matrix variant or explicit error bound.

---

## 6. Evaluation Plan

### 6.1 Core Experiment: Feature Ablation

The paper's central experiment. Six feature configurations, three classifiers, nested CV.

| Configuration | Features | Feature count | Purpose |
|---|---|---|---|
| **SPEC-8** | Spectral features only | 8 | Spectral value in isolation |
| **SYNT-25** | Syntactic features only (density, degree, coeff stats) | 25 | Standard baseline |
| **GRAPH-10** | Graph features only (bipartite constraint-variable graph) | 10 | Graph-theoretic baseline |
| **KRUBER-21** | Kruber et al. (2017) feature set | 21 | Prior-art baseline |
| **COMB-ALL** | All features combined | 43+ | Upper bound on predictive power |
| **RANDOM** | Random baseline | 0 | Lower bound |

**Feature-count-controlled comparison** (per Skeptic's B-S2 fix):
- Top-$k$ spectral vs. top-$k$ syntactic vs. top-$k$ graph for $k \in \{3, 5, 8\}$
- Feature ranking by mutual information with decomposition labels
- Report permutation feature importance from the combined model

**Classifiers:** Random Forest, XGBoost, Logistic Regression. Nested CV: outer 5-fold (evaluation), inner 3-fold (hyperparameter tuning). Stratified by structure type × size bin.

**Statistical tests:** McNemar's test for pairwise feature-family comparison. Holm–Bonferroni correction for multiple comparisons. Report with 95% confidence intervals, not just point estimates.

**Label definition:** argmax of dual bound improvement over LP relaxation at wall-clock parity. Labels: {Benders-amenable, DW-amenable, neither}. Ground truth from GCG (DW) and SCIP-native Benders. **No Lagrangian in ground truth** — acknowledged as limitation.

**Label stability analysis:** Report labels at 4 cutoffs (60s, 300s, 900s, 3600s). If > 20% of labels flip between adjacent cutoffs, use consensus labels (majority vote) and report instability as a finding.

### 6.2 Specific Metrics and Targets

| Metric | Target | Measurement | Kill condition |
|---|---|---|---|
| **Spectral feature value** | ≥ +5pp accuracy or Δρ ≥ 0.1 | SPEC-8 vs. SYNT-25 at matched $k$; McNemar's p < 0.05 | G3: ≤ 0pp gain kills spectral thesis |
| **Spectral scaling validation** | Spearman $\rho \geq 0.4$ | Rank correlation: δ²/γ² vs. observed bound degradation | G1: ρ < 0.4 kills spectral premise (binding); ρ ∈ [0.3, 0.4) triggers nonlinear investigation |
| **Feature redundancy check** | R²(γ₂ ~ density + degree stats) < 0.70 | Linear regression on 50 pilot instances | G0: R² ≥ 0.70 kills spectral novelty |
| **Method selection accuracy** | ≥ 65% | 3-class accuracy (Benders / DW / none) on held-out fold | G4 component |
| **Futility predictor precision** | ≥ 80% | Among instances where predictor fires, verify no method improves bound by > ε | G4 component |
| **L3 empirical correlation** | Spearman $\rho \geq 0.4$ | L3 bound vs. actual LP-decomposition gap | G3 component |
| **Spectral overhead** | < 30s per instance | Wall-clock for Laplacian + eigensolve + features on single CPU core | G2 component |
| **Census coverage** | ≥ 80% of 500 stratified instances | Decomposition runs complete without crash within time cap | G4 component |
| **Cross-scaling robustness** | Spearman $\rho > 0.9$ per feature | Features under Ruiz / SCIP-native / geometric-mean scaling | Report as finding; if ρ < 0.9, restrict to ratio features |
| **Presolve agreement** | Spearman $\rho > 0.7$ | Pre-presolve vs. post-presolve features on 50 instances | If ρ < 0.7, mandate post-presolve; report disagreement |

**Per-structure-type breakdowns** required for all accuracy metrics. Report confusion matrices for each feature configuration. Show that accuracy improvement comes from interesting classes (Benders, DW), not just majority class (none).

### 6.3 Baselines

| Baseline | Implementation | Purpose |
|---|---|---|
| **Monolithic SCIP** | SCIP default (no decomposition) | Is decomposition worth the overhead? |
| **GCG (DW-only)** | GCG with automatic structure detection | Independent DW baseline; also provides reference structure labels |
| **SCIP-native Benders** | `SCIPcreateBendersDefault` | Independent Benders baseline |
| **Random selector** | Uniform random among {Benders, DW, none} | Lower bound |
| **Trivial (always-none)** | Always predict "don't decompose" | Majority-class baseline |
| **Syntactic-only selector** | Best classifier on SYNT-25 features | Ablation comparator |
| **Kruber et al. feature set** | 21 features from their Table 2 | Prior-art comparator |

**GNN baseline:** At minimum, include a discussion section (§7.3 in paper) arguing why spectral features (interpretable, O(m) eigensolve, theoretically grounded via L3/T2) are preferable to GNN-learned features (black-box, O(m·d) message passing, no formal connection to decomposition bounds). If feasible within timeline, include a pretrained GCNN encoder (Gasse et al. 2019) repurposed as feature extractor, adding ~1,000 LoC. This is a should-do, not a must-do; prioritize if schedule permits after G3.

---

## 7. Pre-Solve Timing Specification

**Decision: Compute spectral features AFTER SCIP presolve.**

Rationale: The presolved model is what the solver actually decomposes. Features of the original formulation may describe artifacts (redundant constraints, fixed variables) that SCIP will eliminate.

**Implementation:**
1. Load MPS instance into SCIP.
2. Run `SCIPpresolve()` to completion.
3. Extract the presolved constraint matrix via PySCIPOpt's `getTransformedCons()` + `getValsLinear()`.
4. Construct hypergraph Laplacian from the presolved matrix.
5. Extract features.

**Validation (Week 2):** On 50 pilot instances, compute features on both original and presolved models. Report Spearman $\rho$ per feature. If any feature has $\rho < 0.7$, investigate and document. If majority of features have $\rho < 0.7$, this is a finding worth reporting.

**Cost:** ~200 LoC in the feature pipeline. ~1 day of engineering + 1 day of analysis.

---

## 8. Timeline with Kill Gates

| Week | Milestone | Gate | Kill condition | Recovery |
|---|---|---|---|---|
| **0** | **G0: Spectral redundancy check.** Compute γ₂ on 50 pilot instances. Regress on constraint density + max degree + CV(degree). | G0 | R²(γ₂ ~ syntactic stats) ≥ 0.70 | ABANDON spectral premise. Pivot to census-only (Approach A fallback). |
| **1** | Laplacian construction (both variants) + eigensolve on 50 pilot instances. Eigendecomposition converges on ≥ 90% of pilot. | — | — | — |
| **2** | **G1: Spectral–decomposition correlation.** Feature extraction + GCG/SCIP Benders on pilot. Spearman ρ(δ²/γ², bound degradation). | G1 | ρ < 0.4 on 50 instances (depth-check binding threshold) | Hard kill at ρ < 0.3 → ABANDON spectral, pivot to A. Investigation zone ρ ∈ [0.3, 0.4): investigate nonlinear relationships (Kendall τ, mutual information); if nonlinear signal found, CONTINUE with caution; otherwise ABANDON. Depth check specifies ρ < 0.4 as the binding gate; we add a graduated response below it. |
| **3** | Laplacian stress test: 10 largest MIPLIB instances. Eigensolve < 30s. Pre-solve feature pipeline operational. | G2-partial | Eigensolve fails or > 60s on > 50% of 10 largest | Restrict to m < 50K; report coverage. |
| **4** | **G2: Solver wrappers operational.** GCG + SCIP Benders produce valid dual bounds on ≥ 80% of 50-instance pilot. | G2 | < 80% valid bounds | Debug wrappers for 2 more weeks. If still failing at Week 6, ABANDON. |
| **5–6** | 200-instance dev set: all features (spectral + syntactic + graph + Kruber) + ablation framework. Scaling robustness test (3 methods). Label stability analysis (4 cutoffs). L3 proof sprint (parallel track). | — | — | — |
| **7–8** | **G3: Spectral feature value.** Full ablation on 200-instance dev set with nested CV. | G3 | Spectral accuracy ≤ syntactic accuracy (≤ 0pp gain) AND L3 bound ρ < 0.4 | Pivot to A (census paper with negative spectral result). |
| | **G3 branching decision:** | | • G3 marginal (+3–5pp): Stay with B (feature paper) | |
| | | | • G3 strong (+8pp or more): Extend to C-lite (add partition injection + binary classifiers) | |
| **9–10** | 500-instance stratified evaluation. Census annotation artifact for all 1,065 instances. If C-lite: partition injection + binary classifiers. | — | — | — |
| **11–12** | **G4: Combined evaluation.** Results support at least one of: (a) ρ ≥ 0.5, (b) selector accuracy ≥ 65%, (c) futility precision ≥ 80%. | G4 | All three fail | Salvage census-only paper (lower impact). |
| **13–14** | Paper draft. Artifact preparation. | G5 | Internal review: "fundamental problems" | Revision cycle; delay submission by 4 weeks. |
| **15–16** | Submission-ready. Full 1,065-instance spectral annotation artifact finalized. | — | — | — |

**Total: 16 weeks** (extended 4 weeks beyond Approach B's original 12 to account for Difficulty Assessor's timeline realism critique and the C-lite conditional extension).

---

## 9. Degradation Ladder

If the approach fails at successive gates, what survives?

| Gate failure | Fallback publication | Venue | Estimated P(pub) |
|---|---|---|---|
| **G0 fails** (γ₂ redundant with syntactic) | *Negative result*: "Spectral features of the constraint hypergraph are predictable from syntactic statistics." Short paper + census data release. | CPAIOR workshop / OR Letters | 0.25 |
| **G1 fails** (no spectral–decomposition correlation) | *Census-only paper*: "A Computational Study of Benders vs. Dantzig–Wolfe Decomposition on MIPLIB 2017." Census artifact + L3 verification. Must discover a finding. | C&OR / JoC (data paper) | 0.40 |
| **G2 fails** (solver wrappers unreliable) | *Spectral analysis paper*: "Spectral Characterization of MIPLIB 2017." Features + annotations only, no decomposition evaluation. | CPAIOR / Optimization Letters | 0.30 |
| **G3 fails** (spectral ≤ syntactic) | *Census + negative feature result*: "The first MIPLIB decomposition census reveals that spectral features do not improve decomposition selection over syntactic features." Honest negative result with census artifact. | JoC (if census has a finding) / C&OR | 0.45 |
| **G3 marginal** (3–5pp gain) | *Feature paper (Approach B)*: Spectral features as a new MIP characterization family. Modest but honest improvement. Census artifact. | JoC | 0.50 |
| **G3 strong** (≥ 8pp), C-lite works | *Full paper with oracle*: Spectral features + census + 2-method oracle. Strongest paper. | JoC / MPC stretch | 0.55–0.65 |
| **G3 strong**, C-lite fails | *Feature paper with partition analysis*: Features + census + L3/L3-C partition evaluation (no oracle). Still strong. | JoC | 0.50 |
| **G4 fails** (evaluation doesn't support claims) | *Reduced claims paper*: Census + whichever metrics pass. Honest about what works and what doesn't. | C&OR | 0.40 |

---

## 10. Risk Register

| # | Risk | Probability | Impact | Detection | Recovery |
|---|------|-------------|--------|-----------|----------|
| **R1** | Spectral features redundant with syntactic (γ₂ ≈ f(density, degree)) | 0.30 | Fatal to spectral thesis | G0 (Week 0) | Pivot to census-only (Approach A). Negative result is still publishable. |
| **R2** | Clique-expansion Laplacian intractable for d_max > 200 | 0.40 | Limits coverage to ~85% of MIPLIB | Week 3 stress test | Incidence-matrix Laplacian (already budgeted). Report coverage honestly. |
| **R3** | Eigendecomposition fails on ill-conditioned instances (γ₂ ≈ 0) | 0.35 | Features undefined for ~30% of instances | Week 1 pilot | LOBPCG fallback; for persistent failures, assign NaN and use missing-feature-aware classifiers (XGBoost handles NaN natively). Report coverage. |
| **R4** | GCG build/API breaks wrapper | 0.30 | Blocks label generation | Week 1 | Docker pin to GCG 3.5 + SCIP 8.0. If Docker fails: use precomputed GCG results from literature as reference. |
| **R5** | Spectral features don't beat syntactic at G3 | 0.40 | Kills spectral thesis (degrades to census) | G3 (Week 8) | Census paper + negative feature result. L3/L3-C still contribute. |
| **R6** | Label instability (> 20% flip across time cutoffs) | 0.25 | Undermines downstream classifier | Week 6 stability analysis | Consensus labels (majority vote). Report instability as finding. |
| **R7** | Pre-solve features disagree with original (ρ < 0.7) | 0.20 | Feature interpretation ambiguous | Week 2 | Mandate post-presolve. Report disagreement as finding. |
| **R8** | L3-sp proof exceeds 12 days | 0.25 | Delays paper; weakens theoretical story | Weeks 2–4 | Weaken to empirical correlation claim. Drop "spectral partition bound" from title. |
| **R9** | Partition injection (C-lite) crashes GCG on > 20% of instances | 0.35 | C-lite contribution fails | Week 9–10 | Drop C-lite. Publish as feature paper (B) without oracle. Still strong. |
| **R10** | Too few Benders-amenable instances (< 30) | 0.30 | Binary Benders classifier unreliable | Week 6 labeling | Merge into 2-class (decomposable/not). Report Benders scarcity as finding. |
| **R11** | GNN baseline outperforms spectral features | 0.15 | Weakens novelty claim | Week 9 (if included) | Argue interpretability + theoretical grounding. Spectral features have L3 connection; GNNs don't. Both valid contributions. |
| **R12** | Compute budget exceeds laptop capacity for 500-instance eval | 0.15 | Delays timeline | Week 9 | Reduce to 300 instances with tighter stratification. Or: borrow 4-core server for batch run. |

---

## 11. What is DROPPED (and Why)

### From Approach A (Census-Heavy)

| Dropped element | Reason |
|---|---|
| **Lagrangian relaxation wrapper (ConicBundle)** | ConicBundle last released 2014; C++ with no Python bindings; HIGH dependency risk. All three critics recommend deferral. Lagrangian labels too scarce for classification. Framed as future work. |
| **Lagrangian structure detector** | No canonical signature for Lagrangian-amenable structure. Circular quality criterion (Math Assessor). Unnecessary without Lagrangian wrapper. |
| **Full 1,065-instance decomposition census** | 60% timeout rate makes "complete census" misleading (Skeptic A-S3). Replaced with 500-instance stratified eval + 1,065-instance spectral-only annotation. |
| **T2 as census motivation** | T2 is ornamental in a census context (Math Assessor: "T2 motivates nothing" in Approach A). Keep T2 as B's feature motivation. |

### From Approach B (Spectral-Feature-First)

| Dropped element | Reason |
|---|---|
| **F2 as originally stated** | INCORRECT bound (uses ‖D‖²₂ not κ(D)). Replaced with corrected version. |
| **L3 spectral bound without d_max** | Missing factor; would be caught by reviewers. Corrected to O(δ² · d_max / γ²₂). |
| **Unprincipled sampling heuristic** | Replaced with incidence-matrix Laplacian for high-degree instances. Sampling is unprincipled with no approximation guarantee (all three critics). |
| **8 features claimed as independent** | Likely effective dimensionality 2–3. Report correlation matrix honestly. Still valuable as reproducible definitions. |

### From Approach C (Oracle-System)

| Dropped element | Reason |
|---|---|
| **4-class classifier** | Statistically invalid with < 30 minority-class examples (Skeptic C-F1; mitigations demolished one by one). Replaced with independent binary classifiers. |
| **Conformal prediction (C1)** | Exchangeability assumption invalid for MIPLIB (Math Assessor); prediction sets uninformatively wide (Skeptic C-S1); adds complexity without proportionate value. Dropped entirely. |
| **pip-installable tool claim** | GCG/SCIP require multi-day builds; claim is "reckless" (Skeptic C-F3). Replaced with Docker research prototype. |
| **Lagrangian dispatch** | Same as Approach A — ConicBundle dependency risk + label scarcity. |
| **3,000 LoC partition refinement** | Overclaimed (Difficulty Assessor: "simple greedy heuristic" at 3/10 difficulty). Budget reduced to ~400 LoC. Validated on pilot first; dropped if < 10% L3 improvement on < 50% of instances. |
| **End-to-end latency < 60s claim** | Depends on instance size and Laplacian construction. Report latency distribution honestly; don't promise a hard budget. |

### What We DO NOT Attempt

| Item | Reason |
|---|---|
| **Tightening T2's constant C** | Would require κ⁴ → κ² via smoothed analysis; 1–2 months of focused theory (Math Assessor). Out of scope for JoC. |
| **Formal optimality guarantee for spectral partitioning** | Approximation ratio relative to optimal partition requires 2–3 weeks of additional theory. Nice-to-have; not essential. |
| **NP-hardness of optimal decomposition selection** | Standard reduction from graph partitioning; 1 week. Useful but not load-bearing. Defer. |
| **Lower bound (spectral features are necessary)** | Constructing instance families that are syntactically identical but spectrally distinct. 1–2 weeks. The killer argument for spectral features, but too risky for the timeline. Recommend as future work. |

---

## 12. Venue Targeting

| Venue | Probability | Conditions | Fit |
|---|---|---|---|
| **INFORMS Journal on Computing (JoC)** | **0.50** | All amendments; spectral features ≥ +5pp; census reveals non-trivial findings | **PRIMARY.** Computational study with open artifact. JoC values reproducibility and community infrastructure. |
| **Computers & Operations Research (C&OR)** | **0.65** | All amendments; even modest spectral advantage; census artifact | **SECONDARY.** More tolerant of incremental empirical work. Census alone may suffice. |
| **CPAIOR** | **0.55** | Feature ablation + census data; algorithm-selection papers welcome | **SECONDARY.** Conference venue; faster turnaround. |
| **Mathematical Programming Computation (MPC)** | **0.10** | Only if T2's constant tightened to κ² or better + strong empirical story | **STRETCH.** Requires tight theorems; our T2 is motivational only. |
| **IPCO** | **0.03** | Only if L3/L3-C yield surprising theoretical results | **DO NOT TARGET.** Theory venue; our contribution is empirical. |
| **Best paper at any venue** | **0.05** | Census reveals surprising structural finding + spectral features dominate syntactic by wide margin + C-lite oracle works | Unlikely but possible at JoC if census yields a "40% of MIPLIB has unexploited structure" finding. |

---

## 13. Scores

**Note on calibration:** The depth-check panel consensus (binding) scored this project at V5/D4/BP3/L6. We adopt those scores as our baseline. The synthesis adds L3-C method-specific bounds and the incidence-matrix Laplacian resolution, but these are incremental improvements that do not justify systematic re-scoring. We preserve the panel's calibration.

| Dimension | Score | Depth-Check Panel | Rationale |
|---|---|---|---|
| **Value** | **5/10** | 5 | Census is genuinely useful public good; spectral features are a new characterization family. Not transformative — audience is decomposition researchers, not all OR. Cross-method selection extends Kruber et al. (2017) incrementally. |
| **Difficulty** | **4/10** | 4 | One genuinely hard problem (hypergraph Laplacian at scale, now addressed via incidence-matrix variant). ML pipeline is standard. Solver integration is operational, not inventive. Proofs are real work (18–26 person-days) but standard techniques. |
| **Best-Paper Potential** | **3/10** | 3 | Publishable at JoC; not competitive for best paper at any target venue. Would require census to reveal a genuinely surprising structural finding AND spectral features to dominate syntactic by a wide margin. |
| **Feasibility** | **6/10** | 6 | G0/G1 gates catch fatal failures early (Weeks 0–2). Degradation ladder ensures some publication regardless. Main risk is spectral features being redundant (P ≈ 0.30). Depth-check's 25–30K LoC constraint is met with buffer. Full census iteration is slow but stratified design mitigates. |

**Composite: 4.5/10** (matching depth check's panel assessment). This is a solid B-tier contribution honestly packaged as such.

**Team size assumption:** This timeline assumes a 2-person team (one theory-focused, one engineering-focused) working in parallel during Weeks 1–4. For a solo researcher, add 3–4 weeks to the timeline (total ~19–20 weeks), with proof sprints deferred to Weeks 5–8 after core engineering stabilizes.

---

## 14. Summary: What Makes This Approach Win

This synthesized approach takes the best risk-adjusted path:

1. **From Approach A**: The census infrastructure, honest LoC scope, and multi-cutoff labeling provide a guaranteed publication floor. The census is the most comprehensive decomposition evaluation on MIPLIB to date regardless of spectral feature outcomes.

2. **From Approach B**: The feature ablation is the core experiment — a clean scientific question with a definitive answer either way. The spectral feature definitions, L3 spectral partition bound (corrected), and matched-budget ablation design provide methodological rigor.

3. **From Approach C**: The L3-C method-specific bounds (the most novel math item) strengthen the theoretical contribution. The binary-classifier architecture (not 4-class) is statistically defensible. The partition injection and 2-method oracle (C-lite) add a systems contribution if G3 passes strongly.

4. **From the critics**: The G0 gate (Week 0, 1 day) catches the most likely failure mode before any investment. The incidence-matrix Laplacian resolves the #1 technical risk. The corrected F2 and L3-sp avoid reviewer attacks. The scaling robustness protocol and pre-solve specification address cross-cutting gaps that all approaches ignored.

The approach succeeds if spectral features carry signal (P ≈ 0.60 of at least marginal success); it produces a useful artifact regardless (census + L3); and it fails cheap (G0 at 1 day, G1 at 2 weeks) if the premise is wrong.

---

*This document is the binding specification for the implementation phase. The project proceeds under this plan, subject to the kill gates defined in §8. Any material deviation requires a new synthesis review.*

---

## Appendix V: Independent Verification Report

**Verifier**: Independent Reviewer
**Date**: 2026-03-08
**Verdict**: APPROVED WITH CONDITIONS

### Checklist Results

#### Completeness

- [x] **Concrete title and abstract?** PASS. §0 provides both. Title matches Amendment 1's census-first framing. Abstract is detailed, honest about conditional contributions, and appropriately hedged ("or honestly reporting the margin if smaller").
- [x] **Contributions ranked?** PASS. §1 has a clear hierarchy: P1–P3 (primary), S1–S2 (secondary), C1 (conditional on G3), M1 (motivational). This is exactly the structure the depth check demands.
- [x] **Architecture specified?** PASS. §2 has a file-level directory tree with per-module descriptions and a clear mapping from subsystems to LoC estimates. Sufficient detail for implementation.
- [x] **LoC estimates realistic?** PASS. 26,500 LoC total with 3,500 LoC buffer to the 30K cap. The per-subsystem breakdown is consistent with the Difficulty Assessor's itemization (depth check §1.4: 18.5K core + 50% buffer ≈ 28K). Tests at 3,500 LoC reflects the Difficulty Assessor's upward correction.
- [x] **Math inventory complete?** PASS. §5 lists 8 items with proof effort (18–26 person-days), risk levels, correctness status, and explicit statements of what is NOT claimed. The critical path analysis (L3-sp + T2 = 12–17 days) is useful for scheduling.
- [x] **Evaluation plan specific?** PASS. §6 has 10 metrics with numeric targets, measurement methods, and kill conditions. Per-structure-type breakdowns are mandated.
- [x] **Kill gates defined?** PASS. §8 has G0–G5 with specific thresholds, deadlines, and recovery actions. The addition of G0 (spectral redundancy, Week 0) relative to the depth check is a valuable early-failure detector.
- [x] **Degradation ladder?** PASS. §9 maps 8 failure scenarios to specific fallback publications with venue and probability estimates. Each level produces a viable (if reduced) publication.
- [x] **Venue targets?** PASS. §12 lists 6 venues with probability estimates and fit assessments.
- [x] **Scores with rationale?** PASS WITH NOTES. §13 provides 4 scores with rationale. However, all scores are inflated by +1 relative to the depth check's binding panel consensus (see Condition 1 below).

#### Consistency with Binding Constraints

- [x] **Amendment 1 (census-first restructuring)?** PASS. Title is "Spectral Features for MIP Decomposition Selection: A Computational Study with the First MIPLIB 2017 Decomposition Census." Census is P2. T2 is demoted to M1 (motivational, ≤2 pages). Paper is explicitly framed as a computational study.
- [x] **Amendment 2 (external baselines)?** PASS. §6.1 uses GCG for DW and SCIP-native Benders as independent baselines. §6.3 lists all 7 baselines. Evaluation is reframed as selector ablation with fixed external backends. Lagrangian honestly acknowledged as absent from ground truth.
- [x] **Amendment 3 (honest scope 25–30K LoC)?** PASS. 26,500 LoC. No claims of "six major subsystems." System described as "preprocessing plugin and evaluation framework" in spirit.
- [x] **Amendment 4 (honest terminology)?** PASS. "Futility predictor" used throughout. T2 labeled "motivational." No "bridging theorem" or "certificate" language. §5 "What is NOT claimed" section is excellent.
- [x] **Amendment 5 (stratified evaluation)?** PASS. §6.1 uses 500-instance stratified evaluation. §8 Week 9–10 has full 1,065-instance spectral annotation as artifact. Multi-cutoff labeling at 4 thresholds (60s, 300s, 900s, 3600s).
- [x] **Amendment 6 (feature ablation as core experiment)?** PASS. §6.1 is explicitly titled "Core Experiment: Feature Ablation." Six configurations at matched feature budgets (top-k for k ∈ {3,5,8}). McNemar's test with Holm–Bonferroni correction.
- [x] **Amendment 7 (venue targeting)?** PASS. JoC primary (0.50), C&OR/CPAIOR secondary. IPCO at 0.03 with implicit "do not target." MPC at 0.10 — slightly higher than depth check's 0.05, but explicitly labeled "STRETCH" contingent on tightening T2's constant.

#### Critic Flaw Resolution

- [x] **F2 corrected (κ(D) not ‖D‖²₂)?** PASS. §3 Proposition F2 uses κ(D) = ‖D‖₂·‖D⁻¹‖₂. §11 explicitly lists original F2 as "INCORRECT" and replaced.
- [x] **L3 spectral bound corrected (d_max factor)?** PASS. §5 L3-sp: O(δ²·d_max/γ²₂). §11 confirms original was "Missing factor."
- [x] **Clique-expansion problem addressed?** PASS. §4 provides a detailed two-tier resolution: exact clique expansion for d_max ≤ 200 (~85% of MIPLIB), incidence-matrix Laplacian (Bolla 1993) for d_max > 200. Validation protocol included. Impact on theoretical results assessed. LoC budgeted. This is one of the document's strongest sections.
- [x] **Pre-solve timing specified?** PASS. §7 specifies post-SCIP-presolve with a 5-step implementation plan and validation protocol on 50 pilot instances.
- [x] **Scaling robustness addressed?** PASS. §3 F2 consequence mandates 3 scaling methods; §6.2 includes cross-scaling Spearman ρ > 0.9 metric; fallback to ratio features if robustness fails.
- [x] **GNN baseline at least discussed?** PASS. §6.3 includes a discussion paragraph plan with the interpretability/computation argument. Optional implementation (~1,000 LoC) if schedule permits. Correctly labeled "should-do, not must-do."
- [x] **Ablation design feature-count-controlled?** PASS. §6.1 explicitly includes top-k comparison for k ∈ {3,5,8} with mutual information ranking. Addresses the Skeptic's B-S2 stacking critique.
- [x] **Label stability analyzed at multiple cutoffs?** PASS. §6.1 mandates 4 cutoffs (60s, 300s, 900s, 3600s) with 20% flip threshold triggering consensus labels.
- [x] **Conformal prediction dropped?** PASS. §11 explicitly drops it (exchangeability invalid, sets uninformative). §5 confirms in "What is NOT claimed."
- [x] **4-class classifier replaced with binary classifiers?** PASS WITH NOTES. The 4-class classifier (with Lagrangian) is eliminated. §6.1 uses a 3-class formulation {Benders, DW, none} for the core evaluation, while C-lite (§1 C1) uses independent binary classifiers as the depth check recommended. The 3-class still has a potential Benders minority-class risk, acknowledged in R10 with a merge-to-2-class fallback. Acceptable resolution.
- [x] **pip-install claim removed?** PASS. §1 C1: "Docker research prototype, not pip-installable." §11 lists it as dropped with "reckless" citation.
- [x] **Lagrangian dropped with honest explanation?** PASS. §11 documents the drop for all three approaches with specific reasons (ConicBundle dependency risk, label scarcity, no canonical structure signature).

#### Technical Soundness

- [x] **Math statements correct (as corrected)?** PASS WITH NOTES. All corrected statements are sound in principle. Minor notation issue in L3: y*_e uses hyperedge indexing for a quantity (LP dual) that is constraint-indexed. The L3-C specializations clarify the correct objects (reduced costs for Benders, linking duals for DW), so this is a notation cleanup needed during proof writing, not a conceptual error.
- [x] **Proof effort timeline feasible alongside engineering?** PASS WITH NOTES. The 18–26 person-day proof effort runs parallel with heavy engineering (Laplacian + eigensolve + wrappers) in Weeks 1–4. For a single-person project, this is optimistic by 2–3 weeks. For a 2-person team (one theory, one engineering), it is feasible. The document does not specify team size. See Condition 3.
- [x] **Evaluation plan avoids circularity?** PASS. The selector-ablation framing with external backends (GCG for DW, SCIP-native for Benders) cleanly breaks the circularity identified in depth-check F2. Labels reflect external solver quality, not internal implementation quality.
- [x] **Kill gates well-calibrated?** PASS WITH NOTES. G0 (R² ≥ 0.70, new) is well-designed — cheap and catches the most likely failure mode. However, G1's kill threshold is ρ < 0.3, while the depth check's binding gate specifies ρ < 0.4. The final approach softens this to a graduated response (ρ < 0.25 → abandon, 0.25–0.3 → investigate nonlinear). This is a deviation from the binding depth check. See Condition 2.
- [x] **Degradation ladder produces viable publications?** PASS. Each level maps to a specific venue with a reasonable probability estimate. The G0-fail scenario (negative result short paper, P = 0.25) is honest. The G3-fail scenario (census + negative feature result at JoC, P = 0.45) is arguably the most important insurance policy.
- [x] **Venue probability estimates reasonable?** PASS WITH NOTES. JoC at 0.50 and C&OR at 0.65 are consistent with (and slightly more conservative than) the depth check. MPC at 0.10 is double the depth check's 0.05; given T2's vacuousness and the depth check's explicit warning against MPC, this is mildly optimistic but not unreasonable as a stretch-target probability.

#### Missing Items

- [x] **Anything critics raised that the synthesis failed to address?** PASS. All fatal flaws (F1–F3) and serious flaws (S1–S5) from the depth check are addressed. All cross-cutting issues from the debate (§3.1–3.5) are addressed. The debate's "lower bound" argument (instances syntactically identical but spectrally distinct) is properly deferred to future work in §11. The Skeptic's challenge S→M2 about making this mandatory is noted but the synthesis's judgment to defer is defensible given timeline constraints.
- [x] **Obvious gaps in the evaluation plan?** PASS WITH NOTES. One minor gap: the evaluation does not explicitly address how to handle instances where both Benders and DW improve the dual bound comparably (ties or near-ties in label assignment). The multi-cutoff stability analysis should surface this, but an explicit tie-breaking protocol would strengthen the design. Additionally, the 50-instance pilot selection criteria (how are these 50 chosen?) are not specified.
- [x] **Dependency risks not in risk register?** PASS. R4 covers GCG build issues with Docker pinning. PySCIPOpt API compatibility is implicit but not explicitly listed. This is minor — Docker pinning mitigates it.

### Conditions

The following conditions must be met for this approval to hold:

**Condition 1 (MUST FIX): Align scores with depth-check binding consensus or justify deviations explicitly.**

§13 scores are uniformly +1 above the depth check's panel-consensus scores:

| Dimension | Final Approach §13 | Depth Check §2 (binding) | Delta |
|-----------|-------------------|--------------------------|-------|
| Value | 6 | 5 | +1 |
| Difficulty | 5 | 4 | +1 |
| Best-Paper | 4 | 3 | +1 |
| Feasibility | 7 | 6 (Laptop CPU) | +1 |

The depth check states these are median scores from a 3-expert panel with Best-Paper at a unanimous 3. Inflating all four by exactly 1 point without acknowledging or justifying the deviations undermines the depth check's authority and risks miscalibrated expectations (e.g., a Best-Paper score of 4 may encourage MPC targeting that the depth check explicitly warns against).

**Action required:** Either (a) revert to the depth check's binding scores (5/4/3/6), or (b) add an explicit paragraph in §13 acknowledging the depth check's scores and arguing why the synthesis justifies each +1 adjustment with specific evidence (e.g., "Value increases from 5 to 6 because the synthesis adds L3-C from Approach C, which the depth check did not consider").

**Condition 2 (SHOULD FIX): Reconcile G1 kill threshold with depth check.**

The depth check §7 specifies G1's kill condition as ρ < 0.4. The final approach §8 uses ρ < 0.3 (with investigation range 0.25–0.3). The graduated response is a reasonable engineering judgment, but the deviation from a binding threshold should be explicitly acknowledged with reasoning. Suggested fix: state "Depth check specifies ρ < 0.4; we adopt ρ < 0.3 as the hard kill with a ρ ∈ [0.3, 0.4) investigation zone because [reason]."

**Condition 3 (INFORMATIONAL): Clarify team size assumption for timeline feasibility.**

The 16-week timeline with parallel proof sprints (12–17 person-days) and engineering (Laplacian + eigensolve + wrappers) in Weeks 1–4 is feasible for a 2-person team but optimistic by 2–3 weeks for a solo researcher. The document should state the assumed team size, or add a contingency note for the single-person case.

### Overall Assessment

The final approach document is a thorough, well-structured synthesis that successfully integrates the best elements of three competing approaches while respecting all seven binding amendments from the depth check. The critical technical risks — clique-expansion intractability (§4), evaluation circularity (§6.1), and scaling sensitivity (§3 F2) — are all addressed with concrete resolutions rather than hand-waves. The addition of a G0 gate (spectral redundancy check at Week 0, 1-day cost) is a genuine improvement over the depth check's kill-gate schedule and demonstrates good risk-management judgment.

The document's greatest strength is its intellectual honesty: the "What is NOT claimed" section in §5, the explicit dropping of conformal prediction and Lagrangian with reasons in §11, and the degradation ladder in §9 all reflect a team that has internalized the depth check's critique. The conditional C-lite extension (§1 C1, gated on G3) is a well-designed optionality mechanism that preserves the ceiling without committing prematurely.

The primary concern is a systematic +1 score inflation across all four dimensions relative to the depth check's binding panel consensus. While each individual deviation might be defensible, the pattern suggests optimism bias rather than justified reappraisal. This must be corrected (Condition 1) to ensure the document's credibility as a binding specification. The G1 threshold softening (Condition 2) is a lesser concern but should be acknowledged. With these conditions addressed, the document is ready to serve as the implementation specification.
