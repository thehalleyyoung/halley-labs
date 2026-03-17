# Final Approach: Diagnosis-Guided Repair of Floating-Point Error via Error Amplification Graphs

## 1. One-Paragraph Summary

Penumbra is an end-to-end tool that instruments unmodified scientific Python code to construct an *Error Amplification Graph* (EAG) — a weighted DAG where nodes are operations and edges carry first-order error-flow magnitudes — then uses this graph to diagnose root causes of precision loss (catastrophic cancellation, absorption, smearing, amplified rounding, ill-conditioned subproblem), select targeted repairs from a pattern library, and certify error reduction via interval arithmetic. The primary contribution is the *diagnosis-first paradigm*: rather than searching over transformations blindly (Herbie, Precimonious, FPTuner), Penumbra first explains *why* error accumulated through the EAG's causal structure, then prescribes the repair addressing the diagnosed cause. The paper's value rests on demonstrating this pipeline on ≥5 real pipeline-level bugs in SciPy/scikit-learn/Astropy that no existing tool handles end-to-end, backed by T4 (submodularity-based optimality of diagnosis-guided repair on monotone DAGs) and novel treewidth measurements of error-flow graphs across real scientific codebases.

## 2. Framing Strategy

### Primary Submission Framing

**B's framing: Diagnosis-Repair Pipeline as a tool paper with proportionate math.**

Title: *Penumbra: Diagnosis-Guided Repair of Floating-Point Error in Scientific Pipelines*

**Parallel A submission (PLDI, if τ works):** *Error Amplification Graphs: A Program Representation for Causal Analysis of Floating-Point Error*

**Title decision settled:** The primary submission leads with the tool name and diagnosis-repair framing; the EAG appears as the enabling mechanism. The depth check's recommendation ("Error Amplification Graphs: Causal Diagnosis of...") is reserved for the parallel A submission where the EAG *is* the headline.

### Target Venue

**SC (primary) or FSE (secondary).** SC values practical impact on real scientific codes above theoretical elegance — Satire won distinguished paper at ASPLOS with exactly this profile (shadow-value analysis at scale, empirical evaluation on real codes, no deep theorems). FSE values automated diagnosis-repair pipelines as a software engineering methodology contribution.

### Elevator Pitch to the PC

"Every existing FP tool either detects error (Verificarlo, Satire), repairs expressions (Herbie), or tunes precisions (Precimonious) — none diagnoses *why* error accumulated and prescribes targeted repairs. Penumbra introduces a diagnosis-first paradigm: it constructs the first reified causal graph of error flow (the EAG), uses it to classify root causes, and generates certified patches. On 5+ real SciPy/scikit-learn bugs, it achieves 10× error reduction where `satire | herbie` fails. We prove diagnosis-guided repair is optimal on monotone error-flow DAGs via submodularity (T4), and present the first measurements of treewidth in real scientific code's error-flow structure."

### Preempting the "Just Herbie + Verificarlo" Attack

This is the most predictable reviewer objection and must be addressed head-on in Section 1 of the paper. The defense rests on **three concrete demonstrations, not claims**:

1. **≥3 bugs where both Herbie and Verificarlo fail individually.** Herbie operates on single expressions and cannot repair cross-function error propagation. Verificarlo detects but cannot diagnose or repair. Penumbra must find bugs where error originates in function A, propagates through function B, and manifests in function C — requiring the EAG's causal structure to localize and the cross-function repair synthesizer to fix.
2. **Quantitative causal attribution that no heatmap provides.** Verificarlo gives per-operation error bars. The EAG gives "73% of output error flows through path A→B→C with amplification factor 10⁴; root cause is cancellation at node A." This is a qualitative capability gap: *causal path decomposition*, not just magnitude annotation.
3. **Repair selection that requires diagnosis.** Show cases where the *wrong* repair (e.g., Kahan summation for a cancellation bug that actually needs log-space reformulation) makes error worse. Diagnosis-guided selection avoids this; blind application doesn't.

### Preempting the "Just Satire's Shadow Values in a Graph" Attack

The Skeptic's killer question: *"How is this different from running Satire, collecting per-operation error, and storing results in a graph database?"*

Defense: Satire's shadow values give per-operation error magnitudes — the EAG adds *directional sensitivity edges* (∂ε̂ⱼ/∂ε̂ᵢ) that encode how much error at node i contributes to error at node j. This enables:
- **Path decomposition**: attribute output error to specific propagation paths, not just individual operations.
- **Counterfactual reasoning**: "If we repair node X, output error decreases by Y" — computed from graph structure without re-execution.
- **Optimality guarantees**: T4 proves diagnosis-guided repair is step-optimal on monotone DAGs — impossible without the edge structure.

The demonstration must be *algorithmic, not representational*: show a concrete analysis (e.g., optimal repair ordering) that the EAG enables and raw shadow values do not.

## 3. Architecture Overview

### System Components

```
User Python Code (NumPy/SciPy/sklearn)
         │
    ┌────▼────────────────┐
    │  Shadow Instrument.  │  Tier 1: __array_ufunc__/__array_function__
    │  (Rust via PyO3)     │  Tier 2: LAPACK/BLAS monkey-patch wrappers
    └────┬────────────────┘
         │ per-op trace events (streaming)
    ┌────▼────────────────┐
    │  MPFR Replay Engine  │  128-bit shadow values, sensitivity perturbation
    │  (Rust + MPFR/rug)   │
    └────┬────────────────┘
         │ (shadow values, sensitivities)
    ┌────▼────────────────┐
    │  EAG Builder         │  Streaming DAG construction, edge weight computation
    │  (Rust)              │  Sparsification, aggregation (worst-case/mean/percentile)
    └────┬────────────────┘
         │ EAG (on-disk, compressed)
    ┌────▼────────────────┐
    │  Diagnosis Engine    │  5 classifiers on EAG subgraphs
    │  (Python + Rust)     │  Per-node: category + confidence + repair recommendation
    └────┬────────────────┘
         │ diagnoses
    ┌────▼────────────────┐
    │  Repair Synthesizer  │  Pattern library (30 rewrites) + mixed-precision fallback
    │  (Python + Rust)     │  Greedy repair in T4-optimal order
    └────┬────────────────┘
         │ candidate patches
    ┌────▼────────────────┐
    │  Certification       │  Interval arithmetic via MPFR
    │  (Rust)              │  Coverage-weighted: Tier 1 formal, Tier 2 empirical
    └────┬────────────────┘
         │ certified patches
    ┌────▼────────────────┐
    │  Source Rewriter     │  LibCST AST patches, human-reviewable
    │  (Python)            │
    └──────────────────────┘
```

### LoC Estimates (Honest)

| Component | LoC Range | Novel LoC | Notes |
|-----------|-----------|-----------|-------|
| Shadow Instrumentation | 8–14K | ~5K | Rust hot path, PyO3 bridge |
| MPFR Replay | 5–8K | ~3K | 100+ ufunc wrappers, tedious not novel |
| EAG Builder | 4–7K | 4–7K | **Core novelty**: streaming DAG + sensitivity edges |
| Diagnosis Engine | 3–6K | 3–6K | **Core novelty**: formal classifiers on EAG |
| Repair Synthesizer | 8–12K | ~5K | Pattern library + constraint solver |
| Certification | 3–5K | ~2K | Interval arithmetic wrapper |
| Trace Storage | 3–5K | ~1K | LZ4/Zstd, mmap, append-only |
| Source Rewriter | 2–4K | ~1K | LibCST-based |
| Benchmarks + CLI + Tests | 15–26K | 0 | Infrastructure |
| **TOTAL** | **51–87K** | **~25–31K** | Lower end sufficient for publication |

### Language Choices

- **Rust (via PyO3):** Hot path — per-element MPFR arithmetic, EAG construction, certification. Pure Python with gmpy2 would be 100×+ slower. PyO3 provides zero-copy NumPy array access.
- **Python:** Dispatch layer, diagnosis classifiers (readability over speed — classifiers run once per EAG node, not per element), source rewriting (LibCST is Python-native), CLI, benchmarks.
- **C (minimal):** Direct MPFR/GMP calls where rug's abstraction imposes overhead.

### Shared Infrastructure vs. Novel Contribution

**Shared infrastructure (~60% of LoC):** Shadow instrumentation, MPFR replay, trace storage, LibCST rewriting, benchmark adapters, tests, CLI. These are engineering work following established patterns (Satire's shadow values, Verificarlo's MPFR integration, standard AST rewriting).

**Novel contribution (~40% of LoC):** EAG builder (streaming construction with first-order sensitivity edges — no prior tool builds this structure), diagnosis engine (formal classifiers on the EAG that produce structured root-cause attributions — no prior tool does automated FP diagnosis), repair synthesizer's T4-optimal ordering (provably optimal repair selection on monotone DAGs — no prior tool connects diagnosis to optimal repair ordering).

## 4. Mathematical Contributions

### T1: EAG Soundness (Routine — 95% achievable)

**Statement:** For an EAG G = (V, E, w) where w(oᵢ→oⱼ) = |∂ε̂ⱼ/∂ε̂ᵢ| computed by central finite differencing with step h ∈ [ε_mach, √ε_mach], the total output error satisfies: |ε_out| ≤ Σ_{paths p: source→sink} (Π_{edges (i,j)∈p} w(i,j)) · |ε_source(p)|, assuming (a) ε·n·max(Lᵢ) ≪ 1 (first-order regime), (b) acyclic trace graph.

**Why load-bearing:** Without T1, the EAG is a visualization, not a formal object. T1 gives meaning to edge weights and path-weight products, enabling the quantitative causal attribution that differentiates Penumbra from Satire's magnitude-only localization.

**Proof strategy:** Standard first-order forward error propagation (Higham Ch. 3) applied to the DAG structure. Finite-difference step selection follows Moré-Wild (2011). The bound is conservative (independent path summation ignores cross-path cancellation) — document tightness loss explicitly.

**Fallback:** T1 itself will not fail (it's a standard bound). The risk is that it's *too loose* on reconvergent DAGs. Fallback: report both T1 bound and direct shadow-value measurement; flag when T1 bound exceeds shadow measurement by >100×, indicating the first-order regime is violated. The tool's functionality is not contingent on T1 tightness.

**Scope limitation (stated explicitly):** First-order assumption breaks for condition numbers 10⁸–10¹⁶. This is *exactly* where users care most. The paper must honestly acknowledge: "T1 provides formal backing for well-conditioned pipelines; for ill-conditioned pipelines, Penumbra falls back to direct shadow-value comparison without the path-decomposition guarantee."

### T4: Diagnosis-Guided Repair Dominance (Load-Bearing — 85% achievable)

**Statement:** Given a monotone error-flow DAG G (all edge weights positive) and a repair budget of k actions, the greedy strategy of repairing nodes in descending order of EAG-attributed error contribution is step-optimal: no alternative k-repair sequence reduces total output error by more.

**Why load-bearing:** This is the paper's central theorem. It answers the most dangerous reviewer question: "Why not just try all rewrites?" Without T4, diagnosis-guided repair is a heuristic. With T4, it's provably optimal (on monotone DAGs) — the diagnosis-first paradigm has teeth.

**Proof strategy:** Show that the error-reduction function f(S) = (original error) − (error after repairing set S) is monotone submodular on monotone DAGs. Monotonicity: repairing more nodes never increases error (from monotone edge weights). Submodularity: the marginal error reduction of repairing node v given set S is non-increasing in S (because each node's error contribution is bounded by its EAG-attributed share, and shares are disjoint on monotone DAGs). Greedy maximization of monotone submodular functions is (1−1/e)-optimal (Nemhauser-Wolsey-Fisher 1978); step-optimality follows from the matroid structure of single-node repair actions.

**Achievability:** 85%. The submodularity argument is clean for monotone DAGs. The risk is that real DAGs have negative edge weights (error cancellation across paths), violating monotonicity. For non-monotone DAGs, provide empirical dominance ratios (greedy vs. random/exhaustive).

**Fallback:** If submodularity fails for the general case, restrict the theorem to *locally monotone* DAGs (monotone within each connected component) — still covers the vast majority of real cases. The greedy heuristic works regardless; T4 provides the *justification*, not the *mechanism*.

### C1: Certification Correctness (Routine — 95% achievable)

**Statement:** If interval-arithmetic certification reports error reduction from interval [a,b] to [a',b'] with b' < b, then for all inputs in the certified domain, the repaired pipeline's actual error is strictly less than the original.

**Why load-bearing:** Certification is the differentiator between "we think we fixed it" and "we prove we fixed it." Without C1, repair claims are empirical.

**Proof strategy:** Follows directly from the inclusion property of interval arithmetic (Moore 1966). The novel component is *coverage-weighted certification*: for pipelines with X% Tier 1 coverage, certification covers X% of the error path with formal guarantees and (100−X)% with empirical bounds from black-box LAPACK comparison. This honest framing avoids overclaiming certification on LAPACK-heavy pipelines.

**Fallback:** C1 itself is routine. The risk is that certification rates are low on LAPACK-heavy pipelines (because black-box nodes have empirical, not formal, bounds). Mitigation: report certification coverage alongside error reduction; restrict headline certification claims to Tier 1-dominated pipelines.

### τ Tightness Ratio for Series-Parallel EAGs (Should-Attempt — 70% achievable)

**Statement:** For series-parallel EAGs, define τ(G) = |ε_actual| / (T1 bound). Prove τ(G) ≥ 1/|paths(G)| and characterize τ as a function of the graph's reconvergence structure. For linear chains, τ = 1 (tight). For k-way fan-out/fan-in, τ ≥ 1/k.

**Why load-bearing (conditionally):** τ transforms T1 from a vacuously loose bound into a diagnostic signal. If τ > 0.1 on real programs, the EAG's path decomposition *accurately explains* error flow, not just bounds it. This is the strongest differentiation from Satire (which provides magnitude without causal decomposition). If τ ≈ 0, the EAG's quantitative claims are hollow.

**Proof strategy:** For series-parallel graphs (treewidth ≤ 2), the junction-tree decomposition is explicit. Error cancellation across paths is bounded by the number of merge points. The key lemma: at each parallel-to-series merge, error from the two branches can cancel by at most min(|ε_branch1|, |ε_branch2|), giving a recursive tightness bound.

**Fallback:** If the proof fails, present τ as an *empirical metric* measured across all benchmarks. Report τ distributions. If τ > 0.1 on most real programs (empirical validation), the practical conclusion holds even without the theorem. The paper loses a theorem but retains a useful metric.

**Kill gate:** If τ < 0.01 on all 5 real-program EAGs during the first 4 weeks, abandon the tightness contribution entirely. Do not pursue τ for bounded-treewidth (30% achievable — too risky for the time investment).

### Treewidth Measurements (Empirical — 90% achievable)

**What it is:** Measure treewidth of every EAG across all target codebases. Report distributions, not point estimates.

**Why it matters:** No prior work has characterized the graph-theoretic structure of FP error flow in real code. If treewidth ≤ 5 (as hypothesized for feed-forward scientific pipelines), this motivates future compositional analysis and validates the intuition that error-flow structure is simpler than arbitrary program structure.

**Method:** Exact treewidth via PACE competition solvers (Tamaki 2017) for EAGs ≤ 10⁴ nodes; heuristic upper bounds (min-fill elimination ordering) for larger EAGs. Report both exact values and heuristic upper bounds with the gap.

**This is not a theorem.** It is novel empirical data. Present it as a "Structural Analysis of Error Flow in Scientific Pipelines" subsection in the evaluation. The Skeptic's concern — "sequential programs have low treewidth, you discovered that sequences are sequential" — is partially valid. Mitigate by including non-trivial targets: scikit-learn's PCA (matrix operations with data-dependent branching), GPy's Cholesky path (conditional control flow), OpenMDAO's coupled-system solver (iterative with reconvergence).

### Explicitly Excluded

- **τ for bounded-treewidth EAGs (30% achievable):** Moonshot. The junction-tree decomposition for general bounded-treewidth graphs does not cleanly transfer to multiplicative error propagation.
- **Propagation decay law as a theorem (25% achievable):** Mention as an empirical observation. Do not invest proof effort.
- **T2 in general (40% achievable for restricted; open problem for general):** Demoted to open problem with empirical treewidth data. The system works without it.
- **T3 as a "theorem":** The Mathematician correctly identifies this as formalized bookkeeping. Present the taxonomy with completeness argument over first-order IEEE 754 patterns, but don't oversell it as a deep result.

## 5. Core Technical Approach

### 5.1 Shadow Instrumentation

**What it does:** Intercepts every FP operation in a Python scientific pipeline and maintains a parallel multi-precision shadow execution.

**How it works:**
- *Tier 1 (element-wise):* A `ShadowArray` subclass of `numpy.ndarray` overrides `__array_ufunc__` and `__array_function__`. Every ufunc dispatch creates the corresponding MPFR operation at 128-bit precision. Per-element provenance metadata tracks which source operation produced each value.
- *Tier 2 (LAPACK/BLAS black-box):* Monkey-patches ~20–30 target functions (`scipy.linalg.expm`, `numpy.linalg.solve`, `scipy.linalg.cholesky`, etc.). Each wrapper: (a) captures inputs, (b) calls the original function, (c) replays at multi-precision via `mpmath` matrix operations, (d) computes input→output error amplification element-wise.

**What's hard:** Faithful reproduction of NumPy's exact dispatch contract — broadcasting rules, type promotion cascades, reduction order for `sum`/`prod`/`dot`, special-value handling (inf/NaN/denormal). A single divergence produces a phantom EAG edge. The 100+ ufunc surface is vast.

**What's novel:** The two-tier architecture combining element-level shadow tracking with black-box library wrapping. Satire does shadow values at compile time (C/Fortran); Verificarlo does stochastic rounding at LLVM IR level. Neither operates at Python dispatch level or handles the LAPACK gap explicitly.

**Instrumentation coverage metric:** Every run reports the fraction of pipeline FLOPs in Tier 1 (full internal visibility) vs. Tier 2 (aggregate input→output only). Pipelines with <50% Tier 1 coverage are flagged, and diagnosis claims are qualified.

### 5.2 EAG Construction

**What it does:** Builds the Error Amplification Graph from the shadow execution trace.

**How it works:** Streaming construction. Each trace event (operation + shadow values + sensitivities) becomes a node. Edges are weighted by first-order sensitivity: w(oᵢ→oⱼ) = |∂ε̂ⱼ/∂ε̂ᵢ|, computed by central finite differencing with step h = √ε_mach · max(1, |x̂ᵢ|). For array operations, edges aggregate over elements (worst-case, mean, or p95 — configurable). Sparsification: edges with w < threshold (default: ε_mach) are pruned to keep the graph tractable.

**What's hard:** Memory management for dense matrix operations. A 1000×1000 matrix multiply naively produces ~10⁹ element-level edges. Solution: per-operation aggregation — the EAG operates at *operation* granularity (one node per `np.dot` call), not element granularity, with per-element detail available on demand for diagnosed high-error nodes. Streaming construction with LZ4-compressed on-disk storage keeps memory bounded.

**What's novel:** The EAG itself — the reification of error flow as a weighted DAG with sensitivity edges. Prior tools compute shadow values (per-node error magnitudes) but not *directional error flow* (how much of node j's error came from node i). The sensitivity edges are the qualitative difference enabling path decomposition.

### 5.3 Taxonomic Diagnosis Engine

**What it does:** Classifies every high-error EAG node into one of five root-cause categories, producing structured diagnoses with confidence scores.

**How it works:** Five classifiers, each operating on local EAG neighborhoods:

1. **Catastrophic cancellation:** Detects subtraction of near-equal operands. Trigger: |a−b| / max(|a|,|b|) < threshold, with relative error blowup > 10×. Examines EAG parent edges to confirm both operands carry similar-magnitude values.
2. **Absorption:** Detects small addend lost in large accumulator. Trigger: |small|/|large| < ε_mach. Common in long reductions (`np.sum` on mixed-magnitude arrays).
3. **Smearing:** Detects alternating-sign additions with gradual precision loss. Trigger: alternating-sign operand sequence in a reduction with cumulative error growth sublinear in n.
4. **Amplified rounding:** High condition number amplifies input error. Trigger: EAG edge weight (sensitivity) > 100 for a single operation.
5. **Ill-conditioned subproblem:** A LAPACK black-box node with measured error amplification > 10⁴. Applicable only to Tier 2 nodes; internal diagnosis is unavailable.

Each node receives per-category confidence scores (0–1) and a primary classification. Multi-label nodes (e.g., both cancellation and absorption) receive ranked diagnoses.

**What's hard:** Thresholds. Every classifier has at least one numerical threshold that affects sensitivity/specificity tradeoffs. The paper must report results across a range of threshold settings (sensitivity analysis), not just at the "best" setting.

**What's novel:** No prior tool performs automated root-cause classification of FP error. Fluctuat decomposes error contributions but doesn't classify *why* each contribution is large. Satire localizes *where* error is large but not *what kind* of error pattern caused it. The diagnosis engine is the bridge from detection to repair.

### 5.4 Repair Synthesizer

**What it does:** Selects and applies repairs guided by diagnosis, in T4-optimal order.

**How it works:** Each diagnosis category maps to a repair family:

| Diagnosis | Repair Family | Example |
|-----------|---------------|---------|
| Cancellation | Log-space reformulation, algebraic rewrite | `log(exp(a) + exp(b))` → `a + log1p(exp(b-a))` |
| Absorption | Compensated summation (Kahan, pairwise) | `np.sum(x)` → `kahan_sum(x)` |
| Smearing | Reordering, partial sums | Sort-then-sum, blocked summation |
| Amplified rounding | Mixed-precision promotion | Promote critical operations to float128/MPFR |
| Ill-conditioned | Algorithm substitution, regularization | Pivoted QR instead of direct solve |

The repair synthesizer:
1. Ranks nodes by EAG-attributed error contribution (descending).
2. For each node, selects the repair matching its primary diagnosis.
3. Applies the repair and re-certifies (see §5.5).
4. Iterates until error budget is met or all high-error nodes are repaired.

**The 30-pattern library is honestly a lookup table, not a synthesizer.** The Skeptic is right: calling 30 hand-coded rewrite rules "synthesis" overpromises. The paper should use "repair selection" or "repair prescription" rather than "synthesis." The mixed-precision promotion fallback (promote critical-path operations to higher precision) is the universal backstop when no algebraic rewrite matches.

**What's hard:** LibCST source rewriting. Translating abstract repairs (e.g., "replace this summation with Kahan summation") into concrete, human-reviewable patches that preserve formatting, comments, and variable names is harder than it sounds. Each pattern must handle broadcasting, in-place operations, and syntactic variants.

**What's novel:** The *diagnosis-guided ordering* is the novel element, not the individual rewrites. T4 proves this ordering is optimal on monotone DAGs. No prior tool connects root-cause classification to provably optimal repair ordering.

### 5.5 Certification Engine

**What it does:** Validates that repairs actually reduce error, with formal bounds where possible.

**How it works:** Interval arithmetic via MPFR. For each repair:
1. Compute original output error interval [a, b] by running the original pipeline at multi-precision across the certified input domain.
2. Compute repaired output error interval [a', b'] by running the repaired pipeline identically.
3. If b' < b (worst-case repaired error < worst-case original error), certify.
4. Report the error reduction ratio b/b'.

**Coverage-weighted certification:** For pipelines with X% Tier 1 (fully traced) operations:
- Tier 1 paths: formal interval-arithmetic certification (C1 applies).
- Tier 2 paths (LAPACK black boxes): empirical certification — re-run at multi-precision and compare, but without the formal inclusion guarantee of interval arithmetic.
- Report both Tier 1 and Tier 2 certification separately. Do not claim formal certification on LAPACK-heavy pipelines.

**What's hard:** Input domain specification. Interval arithmetic requires a bounded input domain. For benchmarks, this is specified per-test-case. For general pipelines, the certification covers the *observed input distribution* — narrower than Fluctuat's all-inputs guarantee, but achievable without user-provided specifications.

**What's novel:** Coverage-weighted certification that honestly delineates formal vs. empirical bounds based on instrumentation coverage. No prior tool provides this granularity of certification reporting.

## 6. Evaluation Plan

### Target Codebases and Specific Targets

| Codebase | Specific Targets | Expected Bug Types |
|----------|------------------|--------------------|
| **SciPy** | `expm` for ill-conditioned matrices (issue #18534-family), `logsumexp` underflow/overflow, `betainc` accuracy loss, `hyp2f1` precision near branch cuts | Cancellation, ill-conditioning, absorption |
| **scikit-learn** | PCA on near-rank-deficient data, kernel matrix conditioning in SVM/GP, `log_loss` numerical stability | Amplified rounding, cancellation |
| **Astropy** | Redshift calculations at extreme z, unit conversions across extreme dynamic ranges | Absorption, smearing |
| **FPBench** | Standard single-expression benchmarks (for direct comparison with Herbie) | All categories |

**Dropped from evaluation (per depth check):** FEniCS, Firedrake (C/Fortran core via PETSc — unreachable by Python instrumentation). OpenMDAO retained only if bugs are found during scouting; otherwise dropped. GPy/PyMC retained as stretch targets.

### BC4 Strategy: Finding ≥5 Pipeline-Level Bugs

**Definition of "pipeline-level":** Error must propagate across ≥2 function boundaries, and diagnosis must require ≥2 EAG nodes. Expression-level bugs (solvable by Herbie on a single expression) do not count.

**Scouting methodology (first 2 weeks):**
1. Mine SciPy, scikit-learn, and Astropy issue trackers for precision-related issues. Filter for cross-function error reports (keywords: "precision," "accuracy," "numerical," "overflow," "underflow," "ill-conditioned," "NaN").
2. For each candidate, determine if the bug is expression-level or pipeline-level by tracing the error chain in the issue discussion.
3. Reproduce the top 10 candidates under instrumentation (even with a prototype tracer).
4. Run Herbie on the innermost expression of each candidate to determine if it's Herbie-solvable.

**Target: ≥8 candidates, expecting ≥5 to survive as genuinely pipeline-level.**

**Critical requirement:** ≥3 of these bugs must demonstrate that *both* Satire and Herbie fail individually — Satire localizes wrong or localizes to a non-actionable region, AND Herbie cannot repair across the function boundary. These 3 are the paper's showstopper results.

**The "merged SciPy PR" strategy:** For the most compelling bug found, develop and submit a PR to the upstream library. A merged PR is the single most powerful result in the paper — it transforms Penumbra from "we claim this works" to "SciPy maintainers agree this works." Budget 2 weeks for this effort.

### Baselines and Comparison Methodology

| Baseline | What it Tests | How |
|----------|---------------|-----|
| **Verificarlo** | Does EAG provide more actionable info than per-operation error bars? | Same pipelines, compare localization precision (EAG attributes to root cause; Verificarlo gives magnitude per op) |
| **Herbie** | Does pipeline-level diagnosis outperform per-expression repair? | Apply Herbie to each expression in the pipeline independently; compare total error reduction |
| **Precimonious** | Does diagnosis-guided precision assignment find smaller promotions? | Compare number of operations promoted and resulting error |
| **Satire** | Does EAG provide better root-cause attribution than magnitude localization? | Compare diagnosis accuracy (which operation is the *root cause* vs. which has the *largest error*) |
| **Manual expert** | Does Penumbra match human expert repair? | For bugs with commit-history fixes, compare Penumbra's repair to the developer's fix |
| **Random/exhaustive repair** | Does T4-guided ordering outperform blind search? | Ablation: same repair library, random ordering vs. T4 ordering. Report error reduction per repair action. |

### Metrics with Statistical Rigor

| Metric | Target | Reporting |
|--------|--------|-----------|
| Error reduction | ≥10× median (ulps) | Per-bug and aggregate, with 95% bootstrap CI |
| Diagnosis accuracy | ≥85% vs. ground truth | With sample size, exact binomial CI |
| Runtime overhead | ≤50× tracing, ≤5× repaired | Median + IQR across pipelines |
| Repair minimality | Fewer promoted ops than Precimonious | Per-bug comparison |
| Certification rate | ≥90% of Tier 1 paths | Separate Tier 1 and Tier 2 rates |
| Instrumentation coverage | Reported per target | Tier 1 % vs. Tier 2 % |
| Treewidth | Distribution across all EAGs | Exact + heuristic upper bound |
| τ (if pursued) | Report distribution | With CI; flag if <0.1 |

**Ground-truth methodology:** Root causes are established from developer discussions in issue trackers and commit messages. For bugs without clear developer attribution, use independent multi-precision analysis (run at 256-bit, identify the operation where high-precision and low-precision results first diverge). Report inter-method agreement rate.

**Held-out discipline:** Diagnosis patterns and thresholds are developed on fault-injection benchmarks and FPBench. The ≥5 real pipeline-level bugs are held out — not used for threshold tuning. Report held-out accuracy separately from development accuracy.

## 7. Risk Mitigation

### R1: Too Few Pipeline-Level Bugs (BC4 Failure)
- **Severity:** FATAL for the primary framing.
- **Mitigation:** Scout aggressively in weeks 1–2. If only 3 bugs found, supplement with fault-injection benchmarks and honestly reframe: "3 real + 5 semi-synthetic." If <3 real bugs found, pivot to A's framing (EAG-as-representation) or C's empirical study.
- **Kill gate:** Week 4. If <3 real pipeline-level bugs are confirmed, abandon B's framing.

### R2: τ ≈ 0 on Real Programs (Tightness Vacuous)
- **Severity:** SERIOUS for A's parallel submission; manageable for B's primary submission.
- **Mitigation:** Measure τ early (week 3–4) on the first instrumented pipelines. If τ < 0.01 everywhere, abandon the tightness contribution entirely. B's paper does not depend on τ.
- **Kill gate:** Week 4. If τ < 0.01 on all tested programs, drop τ from the paper.

### R3: MPFR Replay Infidelity
- **Severity:** SERIOUS. Corrupts all downstream analysis.
- **Mitigation:** Differential testing with Hypothesis: for each ufunc, generate 10K random inputs, compare MPFR-at-53-bits with NumPy, require bitwise equality. Maintain a compatibility matrix. Ufuncs failing differential testing are flagged in the EAG as "uncertain shadow value."
- **Kill gate:** None — this is a progressive engineering effort. If >20% of ufuncs fail differential testing, the tool's coverage claims must be scaled back.

### R4: "Just Satire + Herbie" Convinces Reviewers
- **Severity:** SERIOUS (submission-killing).
- **Mitigation:** The ≥3 bugs where both fail individually (see §2) must be found and demonstrated convincingly. If these bugs don't exist, the paper's novelty claim is genuinely weakened and the framing should pivot to the empirical treewidth contribution.
- **Kill gate:** Week 6. If no bug is found where Satire + Herbie jointly fail, reconsider the primary framing.

### R5: Low Instrumentation Coverage on Target Pipelines
- **Severity:** MANAGEABLE.
- **Mitigation:** Select target pipelines during scouting to ensure ≥50% Tier 1 coverage. LAPACK-dominated pipelines (e.g., pure linear algebra) are poor targets — prefer pipelines mixing element-wise and library calls. Report coverage honestly; do not oversell diagnosis on low-coverage pipelines.
- **Kill gate:** None — this is managed by target selection.

### R6: First-Order Assumption Violation
- **Severity:** MANAGEABLE but philosophically embarrassing (breaks on the exact cases users care about).
- **Mitigation:** Automatic detection (compare T1 bound vs. shadow measurement). Flag affected subgraphs. Fall back to direct shadow comparison. The paper must acknowledge this honestly in the limitations section, not bury it.
- **Kill gate:** None — this is a scope limitation, not a bug.

### R7: Repair Patches Are Unreviewable
- **Severity:** MANAGEABLE.
- **Mitigation:** LibCST preserves formatting. Patches are minimally invasive (replace one expression/call, not rewrite the function). Include before/after snippets in the paper. The "merged SciPy PR" strategy (§6) is the ultimate test of reviewability.
- **Kill gate:** None — quality improves iteratively.

### R8: Fluctuat Comparison Challenged by Reviewers
- **Severity:** MANAGEABLE.
- **Mitigation:** Correct the comparison table. Honest differentiation: Fluctuat provides *static, formally sound* error decomposition for C programs with *stronger soundness guarantees* on any single program. Penumbra provides *dynamic, Python-native* pipeline-level analysis with *repair synthesis*. They are complementary, not competing. The EAG is a different *kind* of representation (graph-algorithmic vs. abstract-interpretation), not a better one.
- **Kill gate:** None — this is a framing correction.

## 8. Timeline and Milestones

### Phase 0: Gating Questions (Weeks 1–2)

| Task | Gate | Success Criterion |
|------|------|-------------------|
| BC4 scout: mine SciPy/sklearn/Astropy trackers | Week 2 | ≥8 candidate bugs identified, ≥3 confirmed pipeline-level |
| Prototype shadow tracer on 3 candidate bugs | Week 2 | Basic MPFR shadow values computed end-to-end |
| Satire+Herbie baseline on candidates | Week 2 | Determine which bugs they can/can't handle |

**Week 2 decision point:** If <3 pipeline-level bugs confirmed → pivot to A's framing or C's empirical study. Otherwise, proceed.

### Phase 1: Core Infrastructure (Weeks 3–6)

| Task | Duration |
|------|----------|
| Shadow instrumentation (Tier 1 + Tier 2) | Weeks 3–5 |
| MPFR replay engine + differential testing | Weeks 3–5 |
| EAG builder (streaming) | Weeks 4–6 |
| τ empirical check on first 5 EAGs | Week 5 |

**Week 4 decision point:** τ empirical check. If τ > 0.1 on ≥2 programs → pursue tightness theorem (parallel track). If τ < 0.01 everywhere → drop τ.

**Week 6 decision point:** End-to-end trace → EAG working on ≥3 target pipelines. If not → re-scope.

### Phase 2: Analysis & Repair (Weeks 7–10)

| Task | Duration |
|------|----------|
| Diagnosis engine (5 classifiers) | Weeks 7–8 |
| Repair pattern library (30 patterns) | Weeks 7–9 |
| Certification engine | Weeks 9–10 |
| T4 proof (submodularity) | Weeks 7–9 |
| Treewidth measurements | Week 10 |

### Phase 3: Evaluation & Paper (Weeks 11–14)

| Task | Duration |
|------|----------|
| Full evaluation on ≥5 real bugs + fault injection | Weeks 11–12 |
| Baseline comparisons | Week 12 |
| SciPy PR submission | Week 12 |
| Paper writing | Weeks 13–14 |
| τ theorem (if pursuing parallel A submission) | Weeks 11–14 |

## 9. Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **6/10** | Real pain for a real (if niche) audience. SciPy tracker confirms ~40+ precision issues. The diagnosis-first paradigm is a genuine conceptual advance over blind search. Narrowed honestly to library developers and precision-sensitive researchers — not "all scientific Python users." Docked because the LLM-competitive environment means qualitative diagnosis is now cheap; Penumbra's value is specifically *quantitative, automated, pipeline-level tracing*. The depth check scored this 5/10; amendments (narrowed scope, honest framing, LAPACK strategy) justify +1 but not +2 — the audience hasn't changed and bugs haven't been found yet. |
| **Difficulty** | **7/10** | Engineering-hard at real breadth (6 codebases × 100+ ufuncs × 30 patterns), not algorithm-hard. The only genuinely research-hard subproblem is A's tightness characterization (pursued in parallel). T4's submodularity proof is non-trivial but tractable. MPFR replay fidelity is a vast correctness surface. Honest: most individual components follow established patterns; the difficulty is in making them work together at scale. |
| **Best-Paper Potential** | **6/10** | Strong tool paper at SC/FSE. The "merged SciPy PR" result, if achieved, elevates significantly. T4 (submodularity) gives proportionate math. Treewidth data adds novelty. But without A's tightness theorem, there's no "one surprising result" that best papers typically need. If τ works and the parallel PLDI submission happens, best-paper potential rises to 8. |
| **Feasibility** | **8/10** | Engineering risk, not research risk. The existential risk is BC4 (do the bugs exist?), which is empirically testable in weeks 1–2. MPFR replay is tedious but tractable. Laptop-feasible: CPU-bound, 32GB sufficient with streaming, 8–24 hour full evaluation. |

**Composite: 6.75/10.** This is a CONTINUE — above the 6.25 conditional threshold from the depth check, reflecting the incorporated amendments (T2 demoted, LAPACK addressed, Fluctuat corrected, scope narrowed). The independent verifier assessed the original 7.0 composite as slightly inflated; the corrected Value score (6→from 7) brings this to a more defensible 6.75.

## 10. The Skeptic's Remaining Concerns (Unresolved)

### 1. "The most important bugs live inside LAPACK, where you're blind."
**Status: Partially mitigated, not resolved.** Black-box wrapping measures aggregate error amplification but provides no internal diagnosis. The paper will contain an inherent tension: the most error-prone operations (matrix factorizations, solves, eigendecompositions) are exactly the ones where Penumbra provides the least diagnostic granularity. Tier 2 coverage reporting is honest, but the fundamental limitation remains. A reviewer who asks "why not use Verificarlo for the LAPACK internals and Penumbra for the Python glue?" has a point.

### 2. "First-order analysis fails on ill-conditioned problems — the exact cases users care about."
**Status: Acknowledged, not resolved.** The fallback (direct shadow-value comparison when T1 bounds diverge from measurements) works in practice but abandons the formal path-decomposition guarantee. The paper must frame this as a scope limitation, not sweep it under the rug. A reviewer who says "your formal backing disappears exactly when I need it" is correct.

### 3. "30 patterns is a lookup table, not synthesis."
**Status: Acknowledged.** The paper should use "repair selection" or "repair prescription." The mixed-precision fallback makes the tool functional for any bug, but the pattern library is extensible-by-code-change, not extensible-by-users. A future version could use e-graph equality saturation (Herbie-style) for genuine synthesis; v1 is a curated library. This is an honest v1 limitation.

### 4. "Diagnosis accuracy on 5–20 bugs has wide confidence intervals."
**Status: Mitigated but fundamentally limited by sample size.** Bootstrap CIs will be wide (±15–20% at n=10). The paper must report these honestly. The "merged SciPy PR" result carries more weight than any accuracy percentage on a small sample. A reviewer who says "n=10 is not statistically meaningful" is correct; our response is "n=10 with real bugs and a merged PR is more meaningful than n=100 on synthetic benchmarks."

### 5. "You haven't demonstrated that τ > 0 on real programs."
**Status: Unresolved until week 4 measurement.** If τ ≈ 0 everywhere, the EAG's quantitative path decomposition is vacuously loose, and the tool reduces to a fancier version of Satire's shadow-value localization with a taxonomy bolted on. This doesn't kill the primary (B) submission — which leads with bugs-found-and-fixed, not with τ — but it weakens the "EAG is a new representation" narrative.

### 6. "Fluctuat already does causal error decomposition with stronger guarantees."
**Status: Mitigated by honest reframing.** Penumbra targets Python (Fluctuat targets C), operates dynamically (Fluctuat operates statically), and performs repair (Fluctuat does not). These are real differentiators. But a reviewer who says "Penumbra is Fluctuat-for-Python-with-weaker-guarantees-plus-a-repair-lookup-table" is not entirely wrong. The EAG's graph-algorithmic structure (path decomposition, submodular optimization) must demonstrate capabilities Fluctuat's zonotopes do not support.

### 7. Portfolio overlap with fp-condition-flow-engine (area-079).
**Status: Partially resolved.** Penumbra's EAG is a *causal error-flow DAG* with sensitivity-weighted edges derived from MPFR shadow values, targeting *diagnosis and repair* of floating-point errors. If fp-condition-flow-engine tracks condition numbers through execution, the differentiation is: (a) Penumbra's edges encode *error amplification* (∂ε_out/∂ε_in), not condition numbers; (b) Penumbra's output is *repair patches*, not condition reports; (c) Penumbra's evaluation is *bugs found and fixed*, not condition characterization. Submission venues must not overlap. If the projects share infrastructure, this should be framed as a shared platform with distinct analysis layers.

---

*This document is the synthesis of Approaches A, B, and C from the ideation phase, incorporating all binding conditions (BC1–BC6) from the depth check, all consensus recommendations from the adversarial debate, and the Skeptic's unresolved concerns. It prioritizes B's framing (practical impact via real bugs) with A's intellectual core (EAG as representation), C's empirical contribution (treewidth data), and honest acknowledgment of limitations.*
