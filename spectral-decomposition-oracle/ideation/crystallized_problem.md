# Spectral Features for MIP Decomposition Selection: A Computational Study with the First Complete MIPLIB 2017 Decomposition Census

## Problem and Approach

Selecting the right decomposition strategy for a large-scale mixed-integer program (MIP) is a *reformulation selection* problem — fundamentally distinct from algorithm selection (SATzilla, AutoFolio), which asks "which solver configuration?", reformulation selection asks "which mathematical object should the solver see?" This is a strictly harder question: changing the feasible region's representation, not just solver parameters. Today, no automated system addresses cross-method reformulation selection. GCG (Generic Column Generation, Gamrath et al. 2010–2023) automates Dantzig–Wolfe reformulation via hypergraph partitioning heuristics but is restricted to a single decomposition method, provides no cross-method prediction, and offers no guidance on when decomposition is futile.

We propose a **spectral decomposition oracle**: a lightweight preprocessing layer that extracts spectral features from the constraint hypergraph Laplacian — spectral gaps, eigenvector localization, algebraic connectivity, partition quality metrics — and uses them to predict which decomposition method (Benders, Dantzig–Wolfe, Lagrangian relaxation, or none) will yield the strongest dual bounds. Unlike GCG's combinatorial partitioning (which either finds blocks or doesn't), spectral features are *continuous and geometry-aware*, degrading gracefully with coupling strength. The oracle dispatches to existing solver implementations (SCIP's native Benders, GCG for Dantzig–Wolfe, standard bundle methods for Lagrangian relaxation), contributing the spectral analysis and prediction layer rather than reimplementing textbook decomposition algorithms.

The **primary contribution** is the first complete *decomposition census* of MIPLIB 2017: all 1,065 instances annotated with detected structure type, predicted decomposition method, resulting dual bound, and futility prediction. This census — released as an open artifact — provides the missing empirical foundation for decomposition research: no prior work has systematically answered "which MIPLIB instances are amenable to Benders, DW, or Lagrangian — and by how much?"

**Theoretical grounding.** We provide a structural scaling analysis (Proposition T2) explaining *why* spectral features predict decomposition quality. Under a perturbation model $A = A_{\text{block}} + E$ where $A_{\text{block}}$ is an ideal block-diagonal form and $E$ captures coupling, the spectral ratio $\delta^2/\gamma^2$ (coupling energy squared over spectral gap squared) controls both partition misclassification rate and dual bound degradation. The Davis–Kahan sin-theta theorem bounds eigenspace error as $O(\delta/\gamma)$; a rounding analysis converts this to misclassification rate $O(\delta^2/\gamma^2)$; and a partition-to-bound bridge (Lemma L3) translates misclassified constraints into LP relaxation gap inflation. The multiplicative constant $C$ in the resulting bound scales as $O(k \cdot \kappa^4 \cdot \|c\|_\infty)$, which is vacuous for ill-conditioned instances (big-M formulations with $\kappa > 10^3$). We therefore present T2 as a *structural scaling law* — establishing that spectral quality degrades decomposition performance *gracefully* (quadratically in coupling, not catastrophically) — rather than as a tight numerical prediction. The empirical validation (Spearman $\rho \geq 0.4$ between $\delta^2/\gamma^2$ and observed bound degradation) confirms whether the spectral ratio is predictive in practice.

A key standalone result is **Lemma L3 (Partition-to-Bound Bridge)**: for any partition of the constraint set into $k$ blocks (spectral, heuristic, or manual), the gap $z_{LP} - z_D$ between the monolithic LP bound and the decomposed dual bound is bounded above by the total weight of hyperedges crossing block boundaries, weighted by shadow price magnitude. This result is useful independently of spectral methods — it provides a quality metric for *any* partition, including GCG's.

The system also includes a **futility predictor**: when the spectral gap is below an empirically calibrated threshold, the oracle predicts that no $k$-block decomposition will improve the dual bound by more than a user-specified tolerance $\epsilon$. This threshold is calibrated via cross-validation on held-out MIPLIB instances, not from T2's theoretical constant.

## Value Proposition

This work delivers four things no existing system provides:

**(1) The first complete MIPLIB 2017 decomposition census.** Every instance annotated with detected structure, best decomposition method, dual bound quality, and futility prediction — a public baseline for all future decomposition research.

**(2) Spectral hypergraph features as a new instance characterization family.** Continuous, geometry-aware descriptors (spectral gaps, eigenvector localization, algebraic connectivity) that capture decomposition amenability more expressively than syntactic features or combinatorial partitioning. Validated via ablation: algorithm selection with and without spectral features, reporting per-structure-type accuracy.

**(3) A unified reformulation selection oracle** that chooses among Benders, Dantzig–Wolfe, and Lagrangian relaxation — not just one method — with a learned futility predictor for instances where decomposition is predicted unhelpful.

**(4) Lemma L3 (partition-to-bound bridge)** providing a formal quality metric for any block partition, applicable to evaluating GCG's partitions, manual decompositions, or any future structure detection method.

The primary competition is GCG (DW-only, combinatorial partitioning) and algorithm selection systems (SATzilla, AutoFolio — solver-level, not reformulation-level). This project's spectral features, cross-method prediction, futility predictor, and comprehensive census distinguish it from both.

## Technical Difficulty

The system comprises four major subsystems plus shared infrastructure, totaling approximately 55,000–70,000 lines of code:

| Subsystem | Estimated LoC | Core Complexity |
|---|---|---|
| **Spectral analysis engine** | ~8,000 | Hypergraph Laplacian construction from sparse constraint matrices, ARPACK/Spectra integration for eigensolves, spectral clustering, feature extraction (gaps, localization entropy, connectivity), spectral ratio computation |
| **Solver integration layer** | ~8,000 | SCIP Benders API wrapper, GCG DW interface, Lagrangian bundle method (ConicBundle or custom ~3K LoC), unified subproblem/master interface, partition injection into each solver's decomposition framework |
| **Strategy oracle + futility predictor** | ~8,000 | Spectral feature vector → method classifier (random forest/gradient boosting), empirical threshold calibration for futility predictor, partition refinement, structure type classification (block-angular, bordered-block-diagonal, staircase, network) |
| **MIPLIB census infrastructure** | ~15,000 | Instance parsing (MPS/LP), tiered execution framework (100/500/1065), result database, statistical analysis, AutoFolio integration for feature ablation, reproducibility harness |
| **Shared infrastructure** | ~16,000 | Sparse matrix algebra, hypergraph data structures, solver abstraction layer (SCIP/HiGHS/GCG), logging, configuration, I/O |

The dominant engineering challenges are: (a) making spectral analysis numerically robust across the full diversity of MIPLIB instances without per-instance tuning, (b) interfacing cleanly with three different solver decomposition APIs that have different assumptions and data formats, and (c) building census infrastructure that runs reliably across 1,065 instances with diverse numerical characteristics.

## New Mathematics Required

The paper introduces 8 load-bearing mathematical items (reduced from 19 by scoping T2 as motivational), with an estimated 4 person-weeks of proof effort.

**Proposition T2 (Spectral Scaling Law).** Let $\mathcal{H}$ be the constraint hypergraph of a MIP with constraint matrix $A = A_{\text{block}} + E$, where $A_{\text{block}}$ is block-diagonal with $k$ blocks and spectral gap $\gamma > 0$, and $\|E\|_F = \delta$. Let $\hat{\pi}$ be the partition recovered by spectral clustering on the bottom $k$ eigenvectors of the normalized hypergraph Laplacian $\mathcal{L}(\mathcal{H})$. Then the dual bound $z_D(\hat{\pi})$ satisfies $z_{LP} - z_D(\hat{\pi}) \leq C \cdot \delta^2 / \gamma^2$, where $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$.

**Status:** The constant $C$ renders this bound vacuous on ill-conditioned instances (most of MIPLIB). T2 is presented as a *structural scaling law* explaining the qualitative relationship between spectral quality and decomposition performance, not as a quantitative predictor. The empirical program validates the $\delta^2/\gamma^2$ ratio as a predictor independent of $C$.

*Proof technique:* Davis–Kahan $\sin\Theta$ theorem on $\mathcal{L}(\mathcal{H})$ → eigenspace angle bound $O(\delta/\gamma)$ → rounding to partition misclassification rate $O(\delta^2/\gamma^2)$ → Lemma L3 to bound LP gap inflation.

**Lemma L3 (Partition-to-Bound Bridge).** For any partition of the constraint set into $k$ blocks, the gap $z_{LP} - z_D$ is bounded above by the total weight of hyperedges crossing block boundaries, where weight is defined by the dual sensitivity (shadow price magnitude) of the corresponding constraints. *This result has standalone value independent of spectral methods.*

**Futility Predictor.** When $\gamma < \gamma_{\text{thresh}}$ — an empirically calibrated threshold learned via cross-validation — the oracle predicts that decomposition is unlikely to improve the dual bound by more than $\epsilon$. This is a *learned predictor*, not a formal certificate; the theoretical threshold from T2 is too loose to be practical.

## Best Paper Argument

This work is best positioned as a **computational study at INFORMS JoC** rather than a theory paper. The strongest claims are: (1) the MIPLIB census is the first systematic, reproducible evaluation of decomposition potential across the entire standard benchmark — a community infrastructure contribution analogous to the MIPLIB papers themselves; (2) spectral hypergraph features are a new, principled feature family for MIP instance characterization, validated by demonstrating they improve decomposition selection accuracy by ≥5 percentage points over syntactic features alone; (3) Lemma L3 provides a theoretically grounded quality metric for any block partition; and (4) the reformulation-selection framing is a genuinely novel problem formulation distinct from algorithm selection. The combination of new features, new benchmark, and theoretical grounding is the shape of a JoC contribution; MPC is a stretch target achievable only if the spectral features show strong, consistent advantage.

## Evaluation Plan

All evaluation is fully automated with no human judgment in the loop:

| Metric | Target | Method |
|---|---|---|
| Structure detection F1 | ≥ 0.75 | Spectral partition vs. GCG detections (as reference) + synthetic planted-structure instances |
| Method prediction accuracy | ≥ 65% | Oracle prediction vs. best-performing method (determined by independent baselines: GCG for DW, SCIP-native for Benders, bundle method for Lagrangian) |
| Spectral feature value | ≥ +5pp accuracy or Δρ ≥ 0.1 | AutoFolio with vs. without spectral features; paired permutation test, p < 0.05 |
| Dual bound quality vs. GCG | Competitive or better | Pairwise comparison on ALL instances (not just GCG-compatible) |
| Futility predictor precision | ≥ 80% | Among instances where predictor fires, verify no method improves bound by > $\epsilon$ (cross-validated) |
| Gap closure on structured instances | ≥ 15% | $(z_D - z_{LP}^{\text{root}}) / (z^* - z_{LP}^{\text{root}})$ on instances with detected structure |
| Spectral overhead | < 30s per instance | Wall-clock time for eigendecomposition on single CPU core |
| MIPLIB coverage | 100% (1,065 instances) | Census completes without crash or timeout (1-hour per-instance cap) |
| Spectral scaling validation | Spearman $\rho \geq 0.4$ | Rank correlation between $\delta^2/\gamma^2$ and observed bound degradation |

Baselines: monolithic SCIP (no decomposition), GCG (DW-only, independent implementation), SCIP-native Benders (independent), random decomposition selection, trivial predictor (always most common method), AutoFolio with syntactic features only.

**Per-structure-type breakdowns** are required for all accuracy metrics (network, block-angular, bordered-block-diagonal, staircase, unstructured).

## Laptop CPU Feasibility

The computational bottleneck is the sparse eigendecomposition: computing the bottom $k$ eigenvectors (typically $k \leq 20$) of a sparse matrix with up to $10^7$ nonzeros via implicitly restarted Lanczos (ARPACK) completes in under 30 seconds on a modern laptop CPU. The decomposition methods delegate to existing solver APIs (SCIP, GCG, HiGHS), so subproblem solving scales with the solver's capability.

**Tiered census for development iteration:**
- **Tier 1 (daily):** 100 curated instances spanning all structure types, <10 min each → ~17 hours on 1 core
- **Tier 2 (weekly):** 500 stratified instances → ~4 days on 4 cores
- **Tier 3 (release):** Full 1,065 instances → ~12 days on 4 cores, run once for the paper

Memory is dominated by the sparse constraint matrix and eigenvector storage: an instance with $10^6$ nonzeros requires approximately 50 MB; the largest instances (~30M nonzeros) may approach 2–4 GB, within modern laptop capability. Everything is fully automated — no human annotation or studies.

---

**Slug:** `spectral-decomposition-oracle`
