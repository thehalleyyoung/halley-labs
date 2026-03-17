# Penumbra: Three Competing Approaches

**Project:** fp-diagnosis-repair-engine  
**Phase:** Ideation — Approach Generation  
**Constraint context:** Composite 6.25/10 CONDITIONAL CONTINUE; BC1–BC6 binding conditions active; T2 demoted; LAPACK black-boxed; Fluctuat comparison corrected; ≥5 real pipeline bugs required.

---

## Approach A: "EAG-First Foundations"

**Pitch:** The Error Amplification Graph is a new program representation — the PDG of floating-point error. Diagnosis and repair are applications that demonstrate it, not the other way around.

### Extreme Value Delivered

**Who desperately needs this:** PL researchers building the next generation of FP analysis tools. Today, every FP tool reinvents error propagation tracking ad hoc — Herbie's egraph rewrites, Verificarlo's stochastic samples, Satire's shadow values — because no shared program representation for error flow exists. The EAG fills the role that PDGs filled for slicing and SSA filled for optimization: a canonical, reified, algorithmically queryable structure that becomes the substrate for an entire family of analyses. The secondary user is the SciPy/scikit-learn library maintainer who currently spends 2–5 days per precision bug manually tracing error through intermediates. Penumbra hands them a causal graph where `penumbra diagnose` returns "73% of output error originates from cancellation in `_solve_triangular` at line 847, propagated through 3 operations with amplification factor 10⁴." This is must-have because the alternative — ad hoc expert intuition — doesn't scale, doesn't transfer, and doesn't produce evidence reviewable in a PR.

### Why Genuinely Difficult as a Software Artifact

The EAG looks simple in the abstract (a weighted DAG with sensitivity edges), but its construction requires solving three coupled hard subproblems simultaneously:

1. **Faithful shadow execution.** Every NumPy ufunc (100+) must be replicated at multi-precision (128-bit MPFR minimum) with identical reduction order, broadcasting semantics, type promotion, and special-value handling. A single divergence between fp64 and shadow execution produces a spurious EAG edge. This is not wrapping `mpfr_add` — it's reimplementing NumPy's entire dispatch contract at higher precision, in Rust via PyO3 for performance.

2. **Sensitivity estimation at scale.** EAG edge weights w(oᵢ→oⱼ) = |∂ε̂ⱼ/∂ε̂ᵢ| require finite-difference perturbation of every operation's shadow error, with step-size selection in [ε_mach, √ε_mach] that avoids both truncation error and cancellation error in the derivative estimate itself. For array operations, this is per-element, producing O(n²) potential edges per n-element operation — demanding aggressive sparsification (threshold by magnitude) and streaming construction to stay within 32GB.

3. **Soundness under composition.** T1 (path-weight product bounds total output error) is straightforward for linear chains but conservative to the point of uselessness for reconvergent DAGs where independent-path summation ignores error cancellation across paths. The soundness proof must characterize the tightness gap and provide computable tightness certificates for specific EAG topologies — this requires novel analysis combining classical forward error analysis (Higham) with graph-theoretic path enumeration.

Architecturally, the system demands a streaming pipeline: Python dispatch → Rust shadow engine → streaming EAG builder → on-disk compressed trace → diagnosis passes. Each stage must handle multi-GB traces without exceeding 32GB memory.

### New Math Required

- **T1 (EAG Soundness with tightness characterization).** The first-order bound is standard; the novel component is a computable *tightness ratio* τ(G) = (actual error) / (T1 bound) characterized by the EAG's reconvergence structure. For series-parallel EAGs, we prove τ ≥ 1/|paths|; for bounded-treewidth EAGs, τ ≥ 1/2^tw. This transforms T1 from a loose bound into a useful diagnostic: τ near 1 means the EAG accurately explains error flow; τ near 0 means significant error cancellation across paths.
- **Treewidth characterization theorem (restricted T2).** For series-parallel EAGs (the most common topology in sequential scientific pipelines), prove that diagnosis on each series/parallel component composes with error bounded by the number of merge nodes. This is achievable because series-parallel graphs have treewidth ≤ 2, and the junction-tree decomposition is explicit.
- **T3 (Taxonomic Completeness).** Exhaustive case analysis over IEEE 754 rounding, partitioned by {add, sub, mul, div, sqrt} × {operand magnitude regimes} × {condition number ranges}. Proves every first-order error pattern maps to ≥1 taxonomy category.

### Best-Paper Potential

**Target: OOPSLA or PLDI.** The pitch is "we introduce a new program representation." This is the framing that wins distinguished papers: PDGs (Ferrante+ TOPLAS'87), SSA (Cytron+ TOPLAS'91), e-graphs (Willsey+ POPL'21). If the EAG can be presented with a clean formal definition, a soundness theorem with tightness characterization, and two compelling applications (diagnosis, repair) that are infeasible without it, this hits the PL community's sweet spot. The treewidth characterization of real scientific code's error-flow structure is genuinely novel empirical data with implications for any future compositional FP analysis. Weakness: PLDI reviewers may view the first-order restriction as limiting, and the engineering breadth may read as "too systems-y." OOPSLA is the safer target — it values the representation + applications narrative.

### Hardest Technical Challenge & Mitigation

**Challenge:** MPFR replay fidelity. If shadow execution diverges from fp64 execution in reduction order or special-value handling, the EAG contains phantom edges, and every downstream analysis is corrupted. NumPy's internal dispatch is complex, underdocumented, and version-dependent. **Mitigation:** Differential testing with Hypothesis (property-based): for each ufunc, generate random inputs, compare fp64 result with MPFR-at-53-bits result, require bitwise equality. Maintain a compatibility matrix. For ufuncs where exact replay is infeasible (vendor-specific BLAS), fall back to black-box wrapping with tolerance bounds.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Value | 6/10 | Strong for PL community; narrower immediate practical audience |
| Difficulty | 8/10 | Tightness characterization + faithful MPFR replay are genuinely hard |
| Best-Paper Potential | 8/10 | "New program representation" is the proven path to distinguished papers |
| Feasibility | 6/10 | Tightness theorem is a research risk; MPFR replay is a large engineering surface |

---

## Approach B: "Diagnosis-Repair Pipeline"

**Pitch:** An end-to-end tool that finds real FP bugs in real libraries, diagnoses their root cause, and generates certified patches. The EAG is an implementation detail; the bugs found are the headline.

### Extreme Value Delivered

**Who desperately needs this:** The SciPy maintainer staring at issue #18534 ("expm returns inaccurate results for ill-conditioned matrices") who has spent 3 days inserting `print` statements, manually computing intermediates in `mpmath`, and still can't localize the error source. The scikit-learn contributor investigating why PCA produces nonsensical components on near-rank-deficient data. The Astropy developer trying to understand why unit conversions at extreme redshifts lose 8 digits of precision. These people exist — the SciPy tracker has ~40+ confirmed precision issues, many open for years. Penumbra replaces their multi-day manual workflow with `penumbra trace mytest.py && penumbra diagnose && penumbra repair`. This is must-have because the alternative is human-hours that don't scale, don't transfer, and produce fixes without certified error bounds.

The differentiation from "just use an LLM" is concrete and quantitative: an LLM can suggest "maybe try Kahan summation" but cannot tell you that 73% of output error flows through operation X with amplification 10⁴, that the root cause is catastrophic cancellation (not absorption), and that compensated summation reduces error by 10× with interval-arithmetic certification. Penumbra provides *evidence*, not suggestions.

### Why Genuinely Difficult as a Software Artifact

This approach front-loads the engineering breadth the depth check identified as the true difficulty:

1. **Six-codebase evaluation with adapter infrastructure.** Each target (SciPy, scikit-learn, Astropy, OpenMDAO, GPy/PyMC, FPBench) has different conventions, test harnesses, and precision-sensitive code paths. Building per-codebase adapters that set up inputs, run the pipeline under instrumentation, and validate outputs is 8–15K LoC of careful integration work.

2. **Ground-truth methodology.** Mining ≥5 real pipeline-level bugs from GitHub issue trackers and commit histories, establishing ground-truth root causes from developer discussions, and computing diagnosis accuracy with confidence intervals at small sample sizes. This is empirical science, not engineering — the methodology must survive reviewer scrutiny.

3. **Repair synthesis for real code.** The 30-pattern rewrite library must handle real code's complexity: mixed NumPy/SciPy calls, implicit broadcasting, in-place operations, variable reuse across loop iterations. LibCST-based AST rewriting that produces human-reviewable patches preserving formatting and comments is harder than it sounds — each rewrite pattern needs to handle dozens of syntactic variants.

4. **LAPACK black-box wrapping at production quality.** Monkey-patching `scipy.linalg.expm`, `numpy.linalg.solve`, etc., capturing inputs/outputs at multi-precision, and correctly handling the full API surface (keyword arguments, output arrays, info flags) for ~20–30 target functions. Each wrapper must faithfully reproduce the function's error amplification without access to internal LAPACK state.

### New Math Required

- **T4 (Diagnosis-Guided Repair Dominance) with formal guarantees for monotone DAGs.** For monotone error-flow DAGs (all edge weights positive — the common case), prove that greedy repair in descending error-contribution order is step-optimal: no alternative k-repair sequence reduces more error. This uses submodularity of error reduction on monotone DAGs — a non-trivial but tractable proof. For general DAGs, provide empirical dominance ratios.
- **C1 (Certification Correctness) with coverage analysis.** Interval-arithmetic certification inherits MPFR's inclusion property, but must account for black-box LAPACK nodes where interval evaluation is approximate. Formalize the coverage-weighted certification: for pipelines with X% Tier 1 coverage, certification covers X% of the error path with formal guarantees and (100-X)% with empirical bounds.
- **Error-reduction lower bounds per pattern.** For each of the 5 taxonomy categories, prove a minimum error-reduction factor when the prescribed repair is applied. E.g., compensated summation for absorption reduces relative error by ≥ n·ε_mach → ε_mach² (classic result, but must be formalized in the EAG context).

### Best-Paper Potential

**Target: SC or FSE.** SC values "real impact on real scientific codes" above all. If Penumbra finds a previously unknown bug in SciPy, gets a PR merged, and demonstrates 10× error reduction across 5+ real pipeline-level bugs, this is SC's ideal paper. The narrative is "we built a tool, it found real bugs, here are the patches." FSE values the diagnosis-repair pipeline as a software engineering contribution — automated root-cause analysis for a class of bugs that currently require deep domain expertise. Weakness: without the EAG-as-representation framing, this reads as a strong tool paper rather than a foundations contribution. Tool papers can win best paper at SC (Satire did at ASPLOS) but face higher "just engineering" skepticism at PL venues.

### Hardest Technical Challenge & Mitigation

**Challenge:** Finding ≥5 real pipeline-level bugs (BC4). If the bugs don't exist, or existing tools already find them trivially, the paper collapses. **Mitigation:** (1) Pre-survey SciPy/scikit-learn issue trackers to identify candidate bugs before committing to this approach — there are ~40+ confirmed precision issues, but many may be expression-level (Herbie-solvable) rather than pipeline-level. (2) Distinguish "pipeline-level" operationally: error must propagate across ≥2 function boundaries, and diagnosis must require ≥2 EAG nodes. (3) Fault-injection benchmarks as honest supplement, clearly separated from real-bug evaluation.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Value | 8/10 | Directly solves an acute pain for a real (if niche) audience |
| Difficulty | 7/10 | Breadth-hard; no single algorithmically deep subproblem |
| Best-Paper Potential | 6/10 | Strong tool paper; lacks the "new idea" spark for distinguished paper |
| Feasibility | 8/10 | Engineering risk, not research risk; bugs are the main uncertainty |

---

## Approach C: "Empirical Science of FP Error Flow"

**Pitch:** The first large-scale empirical study of how floating-point error actually propagates through real scientific code — treewidth distributions, error-pattern frequencies, propagation-path characteristics. The EAG is the measurement instrument; the findings are the contribution.

### Extreme Value Delivered

**Who desperately needs this:** Every researcher building FP analysis tools. Today, tool designers make assumptions about error flow structure — Herbie assumes expressions are isolated, Satire assumes shadow-value magnitude is sufficient for localization, Precimonious assumes precision assignment is independent across operations — but nobody has measured whether these assumptions hold in real code. Approach C provides the first empirical ground truth: What is the treewidth distribution of error flow in scientific Python? Which error patterns dominate (cancellation? absorption? ill-conditioning?)? How far does error typically propagate before being absorbed? How often do multiple error sources interact? These measurements directly inform the design of every future FP tool. The secondary audience is the numerical methods educator who teaches Higham's error taxonomy but has no data on which patterns actually occur in practice and at what frequency. This is must-have for the research community because building tools without understanding the phenomenon is flying blind — and every current tool is flying blind about pipeline-level error structure.

### Why Genuinely Difficult as a Software Artifact

Empirical studies at this scale require infrastructure that is itself a significant engineering artifact:

1. **Instrumentation across 6+ codebases with statistical rigor.** Each codebase must be instrumented, traced, and analyzed under multiple input distributions to avoid sampling bias. The instrumentation must be reliable enough that measurement artifacts don't contaminate the findings — a single MPFR replay divergence or dropped edge corrupts the dataset. Cross-codebase normalization (different pipeline lengths, array sizes, operation mixes) requires careful experimental design.

2. **Treewidth computation at scale.** Exact treewidth is NP-hard. For EAGs with 10³–10⁶ nodes, we need either (a) exact computation for small EAGs via state-of-the-art solvers (PACE competition winners), (b) upper/lower bounds for large EAGs via heuristic elimination orderings (min-fill, min-degree) plus Robertson-Seymour minor testing for lower bounds, or (c) approximate treewidth via sampling. The challenge is producing publishable treewidth data with rigorous error bars, not just point estimates.

3. **Error-pattern frequency estimation with confidence intervals.** Classifying every high-error EAG node into taxonomy categories across all codebases, computing frequency distributions, and reporting with bootstrap confidence intervals. Must handle the multi-label case (a node can be both cancellation and absorption) and the black-box case (LAPACK nodes classified only as "ill-conditioned subproblem").

4. **Propagation-path analysis.** Characterizing the distribution of error propagation path lengths, the frequency of reconvergent paths (where error from a single source reaches a sink via multiple routes), and the amplification-factor distribution along paths. This requires path-enumeration algorithms efficient enough for EAGs with 10⁵+ nodes.

### New Math Required

- **Treewidth characterization of scientific computation DAGs.** Formalize the hypothesis that feed-forward scientific pipelines have bounded treewidth. Prove that common computational patterns (map-reduce, scan, stencil, matrix chain) produce EAGs with treewidth bounded by a function of their "pipeline width" (maximum number of live intermediate arrays). This connects the empirical observation to a structural property of scientific code.
- **Error-pattern frequency model.** A probabilistic model of error-pattern occurrence as a function of pipeline characteristics (operation mix, condition-number distribution, array sizes). Fit to observed data with cross-validation across codebases. If the model predicts pattern frequency from static pipeline features, it enables tool designers to prioritize which patterns their tools must handle without running Penumbra.
- **Propagation decay law.** Empirical observation formalized as a testable hypothesis: in scientific pipelines with bounded condition numbers, the contribution of a rounding error at operation oᵢ to output error decays exponentially with the graph distance from oᵢ to the output. If confirmed, this justifies local (rather than global) diagnosis and provides theoretical backing for windowed analysis — a direct speedup for all future FP tools.

### Best-Paper Potential

**Target: ICSE, ESEC-FSE, or OOPSLA (empirical track).** The best empirical SE papers change how the community thinks about a phenomenon. "An Empirical Study of API Misuses" (ICSE'16), "How Developers Fix Performance Bugs" (FSE'12) — these papers succeed because they provide data that overturns assumptions. If Approach C shows that (a) treewidth is universally low (≤5) in scientific code, validating compositional analysis; (b) cancellation dominates absorption 4:1, directing tool investment; (c) error propagation decays exponentially, justifying local analysis — these are findings that reshape the FP tools research agenda. OOPSLA's empirical track would value the treewidth measurement as connecting graph theory to PL practice. Weakness: empirical papers rarely win *best* paper at top venues — they win influential-paper awards 10 years later. The immediate best-paper ceiling is lower than Approach A, but the long-term citation impact may be higher.

### Hardest Technical Challenge & Mitigation

**Challenge:** Ensuring measurement validity — that observed patterns reflect genuine FP error behavior rather than instrumentation artifacts or sampling bias. A single systematic error in MPFR replay (e.g., different reduction order for a common ufunc) could bias pattern frequencies across all codebases. **Mitigation:** (1) Triple-check MPFR replay fidelity via differential testing at fp64 precision (MPFR-at-53-bits must produce bitwise-identical results to NumPy for all tested ufuncs). (2) Report instrumentation coverage per codebase and exclude findings from low-coverage (>50% black-box) pipelines. (3) Sensitivity analysis: repeat all measurements at 128-bit and 256-bit precision to confirm findings are precision-independent. (4) Inter-rater reliability for taxonomy classification using independent re-implementation of classifiers.

### Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Value | 7/10 | Reshapes the research agenda for all FP tool builders; less direct practitioner impact |
| Difficulty | 6/10 | Infrastructure-hard; treewidth computation is the only algorithmically deep piece |
| Best-Paper Potential | 5/10 | Empirical papers rarely win best paper; high long-term citation potential |
| Feasibility | 9/10 | No research risk — measurements always produce data; quality is the variable |

---

## Comparison Matrix

| Criterion | A: EAG-First | B: Diagnosis-Repair | C: Empirical Science |
|-----------|:---:|:---:|:---:|
| Value | 6 | **8** | 7 |
| Difficulty | **8** | 7 | 6 |
| Best-Paper Potential | **8** | 6 | 5 |
| Feasibility | 6 | **8** | **9** |
| **Mean** | **7.0** | **7.25** | **6.75** |

### Strategic Assessment

**Approach A** maximizes best-paper potential but carries the highest research risk (tightness theorem, restricted T2 proof). If the theory lands, this is a distinguished-paper candidate at OOPSLA. If it doesn't, it degrades to a weaker version of B.

**Approach B** maximizes practical value and feasibility. The risk is concentrated in BC4 (finding ≥5 real pipeline bugs). If the bugs exist and the tool finds them, this is a strong SC paper. If bugs are scarce or shallow, the paper reads as "we built a tool and it works on synthetic benchmarks."

**Approach C** maximizes feasibility and minimizes research risk — measurements always produce data. But the best-paper ceiling is lower, and the lack of a repair component means no "showstopper result" (a merged SciPy PR). Best as a companion paper or fallback.

**Recommended hybrid:** Lead with A's framing (EAG as novel representation) but deliver B's evaluation (real bugs, real repairs). Use C's measurements (treewidth, pattern frequencies) as supporting evidence for the EAG's design choices. This captures A's best-paper potential, B's practical impact, and C's empirical novelty — the three facets of a complete paper.
