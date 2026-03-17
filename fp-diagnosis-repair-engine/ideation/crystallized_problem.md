# Error Amplification Graphs: Causal Diagnosis of Floating-Point Error in Scientific Pipelines

**Slug:** `fp-diagnosis-repair-engine`

---

## Problem Statement

Floating-point arithmetic is the bedrock of computational science, yet IEEE 754 double-precision silently corrupts results in ways invisible to the scientists who depend on them. A climate model integrating PDEs over millions of timesteps, a Bayesian engine inverting ill-conditioned covariance matrices, an astrophysics pipeline integrating over extreme dynamic ranges — all accumulate rounding errors that compound through long pipelines, producing answers that look plausible but may be wrong by orders of magnitude in their least-significant digits. The insidious property of floating-point error is not that it exists but that it *hides*. No exception, no warning. The bits simply lie.

Existing tools address fragments. Verificarlo and CADNA *detect* error via stochastic arithmetic but produce heatmaps without explaining *why* error accumulated. Herbie *repairs* isolated expressions but cannot reason across function boundaries or loop iterations. Precimonious and FPTuner *tune* precisions but search blindly without diagnosis. Satire localizes error at scale but stops at detection. Fluctuat provides static, zonotope-based error decomposition with formally sound bounds for C programs, but does not target Python, does not operate at the dynamic pipeline level, and performs no repair synthesis. The gap is a *reified causal structure*: no tool constructs an explicit, queryable representation of how error flows through a pipeline — one that supports graph algorithms for diagnosis, attribution, and repair guidance.

Penumbra introduces the *Error Amplification Graph* (EAG): a novel program representation that makes causal error flow explicit and algorithmically tractable. Analogous to how Program Dependence Graphs enabled slicing, SSA form enabled optimization, and e-graphs enabled equality saturation, the EAG is the first reified causal graph of floating-point error flow that supports graph algorithms for diagnosis and repair. Penumbra instruments unmodified scientific Python code (NumPy, SciPy, scikit-learn) by intercepting array operations via `__array_ufunc__`/`__array_function__` and treating compiled-library calls (LAPACK/BLAS via SciPy) as black boxes with input/output error comparison at multi-precision (MPFR). The resulting EAG is a DAG whose nodes are operations, whose edges carry error-flow magnitudes derived from first-order sensitivity analysis, and whose structure makes causal error accumulation explicit and queryable.

A taxonomic diagnosis engine — the first application demonstrating the EAG's utility — classifies every high-error node into root-cause categories: catastrophic cancellation, absorption, smearing, amplified rounding, ill-conditioned subproblem. Each diagnosis maps to a specific repair family. The repair synthesizer — the second application — uses these diagnoses to select and compose algebraic rewrites (compensated summation, log-sum-exp stabilization, Kahan reduction) and mixed-precision promotions, producing a patched pipeline with error-reduction bounds established through interval arithmetic.

The central contributions are: (1) the EAG as a novel program representation for floating-point error flow, with a soundness theorem (T1) bounding total output error via path-weight products; (2) a formally complete first-order taxonomy of error patterns (T3), proved by exhaustive case analysis over IEEE 754 rounding; and (3) empirical evidence that real scientific pipelines exhibit low treewidth, suggesting compositional analysis is tractable — formalized as the EAG Decomposition Conjecture (T2), an open problem with measured treewidth data across all target codebases.

---

## Related Work and Differentiation

| Tool | Detect | Localize | Diagnose | Repair | Certify | Python-native | Pipeline-level |
|------|--------|----------|----------|--------|---------|---------------|----------------|
| Herbie (Panchekha+ PLDI'15) | — | — | — | ✓ | — | — | — |
| Verificarlo (Denis+ '16) | ✓ | ✓ | — | — | — | — | — |
| CADNA (Jézéquel+ '08) | ✓ | ✓ | — | — | — | — | — |
| Satire (Benz+ ASPLOS'23) | ✓ | ✓ | — | — | — | — | ✓ |
| Fluctuat (Goubault+ '11) | ✓ | ✓ | ✓ | — | — | — | — |
| Rosa/Daisy (Darulova+ '14/'18) | ✓ | — | — | — | ✓ | — | — |
| Precimonious (Rubio-González+ SC'13) | ✓ | — | — | ✓ | — | — | — |
| FPTuner (Chiang+ POPL'17) | ✓ | — | — | ✓ | — | — | — |
| **Penumbra** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Herbie rewrites single expressions via e-graph equality saturation — powerful for isolated formulas but with no notion of a computational pipeline or cross-function error flow. Verificarlo/CADNA randomize rounding modes for statistical detection, producing per-operation error estimates but offering no root-cause diagnosis and no repair. Satire achieves shadow-value localization at unprecedented scale using compile-time instrumentation, but provides no root-cause classification or fix synthesis — it tells you *where* error is large, not *why*.

**Fluctuat** performs error decomposition via zonotopes and affine arithmetic with *formally sound* static bounds, attributing error contributions to individual operations in C programs. Its soundness guarantees are superior to Penumbra's dynamic analysis on any single trace: Fluctuat covers all inputs within a specified range, whereas Penumbra's EAG captures error flow for observed executions. Penumbra differentiates on three axes: (1) *Python target* — Fluctuat targets C; no equivalent exists for the Python scientific ecosystem; (2) *dynamic pipeline-level analysis* — Penumbra traces error through multi-library pipelines (NumPy→SciPy→scikit-learn) at dynamic execution granularity, capturing data-dependent error patterns invisible to static analysis; (3) *repair synthesis* — Fluctuat diagnoses but does not generate repairs; Penumbra closes the loop from diagnosis to certified patches. The EAG-centered framing positions Penumbra as a complementary *graph-algorithmic* framework (dynamic, reified causal structure) alongside Fluctuat's *abstract-interpretation* framework (static, zonotope-based).

Rosa/Daisy provide compositional real-valued specifications with automated error bounds for Scala/C but don't support Python, don't operate at the pipeline level, and perform no diagnosis or repair. Precimonious delta-debugs precision assignments; FPTuner optimizes precision via rigorous optimization; neither diagnoses *why* particular variables need promotion.

**The "just Herbie + Verificarlo" defense.** A natural objection is that Penumbra merely combines detection (Verificarlo) with repair (Herbie). This misses the central contribution: the *EAG as a program representation* connecting detection to repair through a reified causal structure. Without the EAG, combining these tools requires a human expert to interpret Verificarlo's error heatmap, identify the root cause, and manually select the appropriate rewrite. The EAG makes error flow a first-class, algorithmically queryable object — the novel intellectual core.

---

## Value Proposition

**Who needs this.** Library developers maintaining precision-sensitive code (SciPy special functions, scikit-learn numerical routines) who spend days diagnosing precision bugs reported in issue trackers. Researchers in extreme-precision domains — probabilistic ML (PyMC, GPy) encountering silent corruption in log-likelihood gradients and covariance inversions, astrophysicists (Astropy) integrating over extreme dynamic ranges where absorption destroys small contributions, aerospace engineers (OpenMDAO) solving coupled systems with mixed-scale variables.

**Why now.** These communities debug FP issues by ad-hoc intuition: a senior analyst stares at intermediates, guesses, and manually rewrites. Slow (days per bug), unreliable, non-transferable. The bugs in SciPy's tracker — decreasing accuracy of `expm` for ill-conditioned matrices, precision loss in special functions like `logsumexp` and `betainc` — represent ~40+ confirmed precision issues. In 2026, an LLM can provide a reasonable *qualitative* diagnosis of a suspected precision problem in seconds. Penumbra's unique value is what LLMs cannot provide: *quantified, automated, pipeline-level tracing* — precise attribution of how many ulps of output error originate from which operation, through which propagation path, with certified repair bounds.

**What becomes possible.** A library developer runs their pipeline under instrumentation, learns that 73% of output error originates from cancellation in one matrix subtraction on line 847, obtains a compensated Kahan-style rewrite with a certified ≥10× error reduction bound, applies the patch, and the corruption vanishes. The EAG provides the quantitative causal evidence that no heatmap, no stochastic sample, and no LLM conversation can produce.

---

## Non-Goals and Scope Limitations

Penumbra v1 targets single-process, CPU-executed scientific Python using NumPy/SciPy. Explicit non-goals for v1 (scope boundaries, not fundamental limitations):

- **GPU-accelerated code** (CuPy, JAX on GPU): GPU runtimes bypass Python dispatch hooks.
- **JIT-compiled code** (Numba, JAX/XLA): JIT eliminates `__array_ufunc__`/`__array_function__` interception points.
- **Distributed computations** (Dask, Ray, MPI): cross-process shadow-value coordination is orthogonal to diagnosis.
- **Non-Python scientific code** (Fortran, C++, Julia): tools like Verificarlo and Fluctuat serve these ecosystems.
- **Per-operation tracing inside compiled libraries** (LAPACK/BLAS): these are treated as black boxes with input/output error comparison (see Technical Difficulty). The EAG models their aggregate error contribution but not internal error flow.

---

## Technical Difficulty

A systems-heavy project spanning Python runtime internals, multi-precision arithmetic, graph algorithms, and numerical mathematics. Rust↔Python interface uses PyO3 throughout.

**Shadow Instrumentation (8–14K LoC, Rust + Python via PyO3).** Two-tier interception strategy:

*Tier 1 — Element-wise operations:* Intercepts NumPy/SciPy operations via `__array_ufunc__`/`__array_function__`. Handles broadcasting, type coercion, stride patterns, in-place ops, and lazy evaluation. Shadow arrays track per-element provenance metadata at <8× memory overhead (4× MPFR shadow value, 2× metadata, pooled remainder). Rust inner loop keeps tracing overhead at 10–50× vs. uninstrumented execution.

*Tier 2 — LAPACK/BLAS black-box wrapping:* `__array_ufunc__`/`__array_function__` cannot intercept compiled LAPACK/BLAS routines dispatched by `scipy.linalg.*`, `numpy.linalg.*`, and related F2PY/Cython wrappers. These are the linear algebra core where the most consequential FP errors often occur (`expm`, `solve`, `eigh`, `cholesky`, `svd`). Penumbra addresses this via Python-level monkey-patching: each target function (e.g., `scipy.linalg.expm`) is wrapped to capture inputs and outputs, replay both at multi-precision (MPFR), and compute aggregate input→output error amplification. The wrapped function appears as a single node in the EAG with edges weighted by measured error amplification, rather than an expanded subgraph of per-FLOP operations. This provides pipeline-level error attribution through opaque library calls at the cost of internal visibility — the EAG knows *how much* error a `scipy.linalg.solve` call amplified but not *which internal LAPACK operation* caused it. Coverage is tracked via an **instrumentation coverage metric**: the fraction of pipeline FLOPs passing through traced (Tier 1) vs. black-box (Tier 2) nodes.

**Multi-Precision Replay (5–8K LoC, Rust + C).** Wraps MPFR for high-precision replay of 100+ ufuncs, faithfully reproducing NumPy's exact reduction order, broadcasting expansion order, and special-case handling for inf/NaN/denormals at higher precision. For black-box LAPACK calls, replay invokes the same SciPy function with `mpmath`-backed arbitrary-precision inputs via matrix conversion, comparing outputs element-wise.

**EAG Builder (4–7K LoC, Rust).** Streaming construction of the error-amplification graph from trace logs. Each node is an array operation (element-wise) or a black-box library call; edges carry error differentials from first-order sensitivity analysis. Configurable aggregation (worst-case, mean, percentile) over array elements. Streaming-capable for traces too large for memory.

**Taxonomic Diagnosis Engine (3–6K LoC, Python + Rust) — CORE APPLICATION OF THE EAG.** Five formally defined classifiers: (1) catastrophic cancellation (near-equal subtraction with relative error blowup), (2) absorption (small addend lost in large accumulator), (3) smearing (alternating-sign additions with gradual error growth), (4) amplified rounding (high condition number amplifies input error), (5) ill-conditioned subproblem (linear solve / eigendecomposition unreliable at working precision). Each operates on EAG subgraphs, producing structured diagnoses with confidence scores and repair recommendations. For black-box nodes, diagnosis is limited to category (5) — ill-conditioned subproblem — based on measured input/output error amplification.

**Repair Synthesizer (8–12K LoC, Python + Rust).** Two strategies selected by diagnosis. *Algebraic rewrites* for known patterns: compensated summation for absorption, log-sum-exp for cancellation in softmax-like patterns, Kahan summation for long reductions, Welford's algorithm for variance, reformulated quadratic formula (~30 patterns total). *Mixed-precision promotion* as universal fallback: a constraint solver determines the minimal set of operations to promote to higher precision, guided by EAG sensitivity edges to promote only operations on the critical error path.

**Certification Engine (3–5K LoC, Rust).** Interval arithmetic via `rug`/MPFR validates that repaired output intervals are strictly tighter than original error.

**Trace Storage & Serialization (3–5K LoC, Rust).** Streaming append-only writes with LZ4/Zstd compression, memory-mapped replay for random-access EAG construction. Handles multi-GB traces.

**Source Rewriting & Patch Application (2–4K LoC, Python).** AST-level repair via LibCST — translates abstract repair prescriptions into concrete, human-reviewable source patches preserving formatting and comments.

**Benchmark Suite & Evaluation (8–15K LoC, Python).** Automated harness with per-codebase adapters, ground-truth infrastructure, statistical analysis with confidence intervals.

**CLI, Reporting, API (4–7K LoC, Python).** `penumbra trace|diagnose|repair|certify`. JSON + human-readable reports. Instrumentation coverage summary per run.

**Test Infrastructure (7–11K LoC).** Property-based (Hypothesis), regression, integration, and differential tests against Mathematica references.

**Aggregate: ~51K–87K LoC (realistic range); ~25–31K novel.** Lower end: focused prototype sufficient for publication. Upper end: full production system. Novelty concentrates in the EAG builder, diagnosis engine, and repair synthesizer; instrumentation, storage, benchmarks, and test infrastructure are engineering work built on established patterns.

---

## New Mathematics Required

Three theorems (T1, T3, T4) and one implementation correctness property (C1) provide the formal backbone. T2 is formalized as a conjecture and open problem with empirical treewidth data. T6 is conditional on T2.

### T1: EAG Soundness

The EAG edge weight w(oᵢ → oⱼ) = |∂ε̂ⱼ/∂ε̂ᵢ| (finite-differenced from shadow values) yields a sound bound: total output error ≤ sum over all source-to-sink paths of (product of edge weights × source rounding error). **Assumptions:** (a) first-order validity — higher-order terms O(ε²) negligible when ε·n·max(Lᵢ) ≪ 1; (b) DAG acyclicity (always true for single traces); (c) finite-differencing step h ∈ [ε_mach, √ε_mach] for derivative soundness. **Tightness:** tight for linear pipelines; conservatively loose (up to exponential) for reconvergent DAGs due to independent path summation ignoring cancellation. Soundness (no missed errors) is prioritized over tightness.

**Scope boundary (v1):** The first-order assumption ε·n·max(Lᵢ) ≪ 1 breaks down for ill-conditioned targets where condition numbers reach 10⁸–10¹⁶. For such pipelines, T1's bound may underestimate actual error. Penumbra detects this condition (by comparing the first-order bound against measured shadow-value error) and flags affected subgraphs as "outside first-order regime" in the EAG, falling back to direct shadow-value comparison without the path-decomposition guarantee.

### T2: EAG Decomposition — Conjecture and Open Problem

**We conjecture** that an EAG with treewidth ≤ k and locally Lipschitz operations (|∂ε̂_out/∂ε̂_in| ≤ Lᵢ) decomposes into subgraphs {G₁,…,Gₘ} with boundary overlap ≤ k, such that each Gⱼ is independently diagnosable and repairable, with composed error ≤ Σⱼ(subgraph bound) + O(k·max(Lᵢ)·ε_boundary).

**Status: open problem, not a central contribution.** The additive decomposition structure of graphical model inference (Lauritzen & Spiegelhalter, 1988) does not directly transfer to the multiplicative error propagation in EAGs. Proving T2 — even for restricted graph classes such as series-parallel EAGs — remains an open problem and would constitute a significant theoretical advance if achieved.

**Empirical contribution:** We *measure* (not assert) treewidth across all target codebases. Hypothesis: most scientific pipelines have treewidth ≤ 5 due to sequential feed-forward structure. This treewidth data is novel — no prior work has characterized the graph-theoretic structure of FP error flow in real scientific code. If confirmed, it motivates future theoretical work on compositional FP analysis. *Fallback:* high-treewidth pipelines use heuristic cut-point decomposition with empirical (not certified) composition bounds, reported separately.

### T3: Taxonomic Completeness

Every **first-order error pattern** — an error effect attributable to a single rounding operation or a pair of adjacent operations in the EAG — falls into **at least one** of: cancellation, absorption, smearing, amplified rounding, ill-conditioned subproblem. Coverage without uniqueness: patterns can span multiple categories; the classifier assigns per-category confidence scores. Proof by exhaustive case analysis over IEEE 754 rounding, partitioned by operation type, operand magnitudes, and condition number.

**Scope boundary (v1):** T3 covers first-order patterns only. Multi-hop error amplification — where error emerges from the interaction of three or more operations with no single pair exhibiting anomalous behavior — is outside the scope of v1's taxonomy. Such emergent pipeline patterns are detected by the EAG's path-weight analysis (T1) but are not classified into named root-cause categories. Extending the taxonomy to multi-hop patterns is future work.

### T4: Diagnosis-Guided Repair Dominance

**Objective:** minimize total pipeline output error (ulps) subject to budget k repair actions. For **monotone error-flow DAGs**, diagnosis-guided repair (descending error contribution order) is step-optimal. For **general DAGs**, we prove empirically — not theoretically — that diagnosis-guided repair outperforms blind search (random, delta-debugging, enumeration) on all benchmarks, reporting the dominance ratio.

### Implementation Correctness Properties

**C1: Certification Correctness.** If interval certification reports error reduction from [a,b] to [a',b'] with b' < b, then the repaired pipeline's actual error is strictly less for all inputs in the certified interval. Follows from the inclusion property of interval arithmetic; verified by differential testing.

### T6: Repair Composition Safety (Conditional on T2)

Composed repair R = R₁∘⋯∘Rₘ satisfies: global error ≤ Σⱼ δⱼ·(original Gⱼ error) + O(k·max(Lᵢ)·ε_boundary). **If T2 holds:** the tree-decomposition guarantees adjacent subgraphs interact only through boundary nodes; each subgraph repair's interval certificate bounds boundary error, and propagation through ≤ k separator nodes yields the coupling term. **If T2 fails** (high treewidth or conjecture disproved): T6 degrades to empirical validation — compose repairs and verify directly via full-pipeline interval arithmetic. This fallback is always available and sufficient for practical use; T2 would provide the theoretical justification for *why* compositional repair works, not the mechanism.

---

## Best Paper Argument

**New program representation.** The EAG is the first reified causal graph of floating-point error flow that supports graph algorithms. Just as PDGs enabled program slicing by making data and control dependence explicit, and SSA enabled optimization by making value flow explicit, the EAG enables causal error diagnosis by making error flow explicit. This is a *foundations* contribution — a new way of representing and reasoning about a fundamental program property — not merely a tool contribution. Diagnosis and repair are applications demonstrating the representation's utility, analogous to how slicing demonstrated the PDG's utility.

**Diagnosis-first paradigm.** The idea that floating-point repair should be *prescribed by formal root-cause analysis* rather than discovered by search is a conceptual shift. Every prior tool (Herbie, Precimonious, FPTuner) treats repair as optimization: search over transformations and pick the best. Penumbra inverts this: *first* understand why error occurs via the EAG, *then* select the repair addressing the diagnosed cause. This mirrors the evolution from brute-force testing to specification-driven verification.

**Depth.** Spans four distinct areas at research depth: runtime instrumentation (systems), multi-precision arithmetic (numerical methods), graph-theoretic analysis (algorithms), and program repair (PL/SE). Taxonomic completeness (T3) requires formalizing error patterns historically described only informally (Higham, 2002; Muller et al., 2018). The empirical treewidth characterization of scientific code's error-flow structure is novel data with implications beyond this project.

**The "just Herbie + Verificarlo" defense.** Without the EAG, combining detection and repair requires human expert interpretation. The EAG is a causal graph (not a heatmap), the classifier is a formal pattern matcher (not a threshold check), and the representation is where the new science lives.

**Impact.** Real codebases, real bugs, real repairs. Python focus maximizes reach within the target audience of library developers and precision-sensitive researchers.

---

## Evaluation Plan

Fully automated — no human subjects or manual labeling. Confidence intervals reported on all metrics. Training/validation separated from held-out evaluation.

**Targets:** SciPy (`expm` conditioning for ill-conditioned matrices, special-function precision loss in `logsumexp`/`betainc`/`hyp2f1`), scikit-learn (PCA on near-rank-deficient data, kernel matrix conditioning), Astropy (extreme dynamic-range redshift calculations, unit conversions), OpenMDAO (coupled-system solvers with mixed-scale variables), GPy/PyMC (Cholesky decomposition of near-singular covariance matrices, log-probability gradient computation), FPBench suite (standard FP research benchmarks for direct comparison).

**Real pipeline-level bugs (≥5 required).** At least five previously-documented or newly-discovered pipeline-level bugs where error propagates across multiple operations/functions, drawn from SciPy/scikit-learn/Astropy issue trackers and commit history. These serve as the held-out evaluation set — diagnosis and repair patterns are not tuned to these specific bugs.

**Fault-injection benchmarks.** Semi-synthetic benchmarks created by injecting known precision-degrading perturbations (reduced precision at specific operations, artificially ill-conditioned inputs, removed compensations) into working pipelines. Honestly framed as *fault injection* — standard methodology in reliability engineering — not as natural bugs. These are used for development and validation, not for headline accuracy claims.

**Baselines:** Verificarlo (stochastic detection — does Penumbra's EAG provide more actionable information than per-operation error bars?), Herbie (expression repair — does pipeline-level diagnosis outperform applying Herbie per-expression?), Precimonious (precision tuning — does diagnosis-guided assignment find smaller promotions faster?), Satire (localization — does the EAG provide more precise root-cause attribution?), manual expert repair (from commit history where available).

**Metrics:** error reduction (≥10× median in ulps, with 95% confidence intervals), diagnosis accuracy (≥85% vs. ground-truth from issue discussions, with sample sizes and confidence intervals reported), runtime overhead (≤50× tracing, ≤5× repaired), repair minimality (fewer ops than baselines), certification rate (≥90%), treewidth distribution across all EAGs, **instrumentation coverage** (fraction of pipeline FLOPs in Tier 1 traced vs. Tier 2 black-box nodes, reported per target).

**Infrastructure:** per-codebase adapter scripts, top-level runner, structured JSON output. Full suite: 8–24 hours on 8-core/32GB laptop.

---

## Laptop CPU Feasibility

Entirely CPU-bound: MPFR is sequential per-operation (arbitrary-precision multiplication cannot be efficiently mapped to GPU SIMD), EAGs are sparse irregular graphs poorly suited to GPU parallelism, diagnosis is rule-based, repair is constraint solving. No GPU benefit. Shadow memory ~24N bytes per N-element array (16 bytes MPFR at 128-bit + 8 bytes metadata); 32GB handles ~1B elements; streaming trace storage (LZ4-compressed, memory-mapped) bounds memory for larger pipelines. EAG node metadata for dense matrix operations may require aggressive summarization when matrices exceed ~5000×5000 (streaming mode avoids 32GB memory ceiling). Ground-truth labels mined automatically from GitHub issue trackers and commit messages. Entire evaluation: `make eval`.

---

## Risk Assessment

**R1: Overhead exceeds 50×.** Rust (PyO3) handles the hot path; Python is dispatch-only. Profiling shows ~2μs per `__array_ufunc__` dispatch, acceptable for arrays ≥1000 elements where MPFR dominates. Sampling mode for small-array workloads.

**R2: MPFR replay mismatches NumPy semantics.** NumPy's reduction order and type-promotion rules are complex. Differential testing on randomized inputs per ufunc; discrepancies documented and flagged in EAG as uncertain shadow values.

**R3: Unbounded treewidth.** Empirically measured on all targets; scientific codes are predominantly sequential. Fallback: heuristic cut-point decomposition with empirical composition bounds, reported separately. T2 is an open problem; system functionality does not depend on its resolution.

**R4: Ambiguous classification.** Per-category confidence scores; multi-class repair application. T3 guarantees first-order coverage, not uniqueness. Multi-hop patterns outside v1 taxonomy are flagged but not classified.

**R5: Insufficient rewrite patterns.** Mixed-precision promotion is universal fallback. Rewrite library extensible; coverage tracked.

**R6: Too few real bugs.** Fault-injection benchmarks supplement real bugs for development and validation. FPBench adds standardized single-expression cases. Held-out evaluation requires ≥5 real pipeline-level bugs; failure to find these is a kill-gate condition.

**R7: Low instrumentation coverage on LAPACK-heavy pipelines.** Pipelines dominated by `scipy.linalg.*` calls will have most FLOPs in black-box Tier 2 nodes, limiting the EAG's internal diagnostic granularity. Instrumentation coverage is reported per target; pipelines with <50% Tier 1 coverage are flagged, and diagnosis claims are qualified accordingly.

**R8: First-order regime violated.** For ill-conditioned targets (condition numbers 10⁸–10¹⁶), T1's first-order bound may underestimate error. Detected automatically by shadow-value comparison; affected subgraphs flagged as outside first-order regime with fallback to direct error measurement.
