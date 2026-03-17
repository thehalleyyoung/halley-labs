# TaintFlow: Quantitative Information-Flow Auditing for ML Pipeline Leakage

**Stage:** Crystallized Problem Statement
**Community:** area-041-machine-learning-and-ai-systems
**Venue targets:** ICML or NeurIPS (ML systems track); OOPSLA or FSE as backup
**Slug:** `ml-pipeline-leakage-auditor`

---

## Problem and Approach

Train–test data leakage, first systematically characterized by Kaufman et al. (TKDD 2012), is the most pervasive silent failure mode in machine learning.
Unlike a crash or a type error, leakage produces no observable symptom—it produces
*inflated metrics*. A model that achieves 95% accuracy with leakage and 82% without
appears to work perfectly throughout development; the discrepancy surfaces only when
production metrics diverge from offline evaluations, or never surfaces at all. Empirical
studies estimate that 15–25% of public Kaggle kernels contain some form of leakage
(Yang et al., ASE 2022), and the problem extends into published research and production
ML systems in finance, healthcare, and advertising. The root cause is structural:
standard ML workflows encourage a sequential script where preprocessing, feature
engineering, and model fitting are interleaved in a single namespace, making it trivially
easy to call `StandardScaler.fit_transform()` on the full dataset before
`train_test_split()`, or to compute target-encoded features using the entire target
column. The resulting contamination is invisible—no exception is raised, no warning is
printed, and the model appears to generalize far better than it actually does. Yet the
field possesses no tool that can answer the fundamental diagnostic question: *how many
bits of test-set information have contaminated each training feature, through which
pipeline stages, and with what severity?*

Existing approaches to leakage detection fall into two categories, both fundamentally
limited. **Pattern-matching tools** (LeakageDetector, Yang et al., ASE 2022 / SANER
2025) perform syntactic analysis of Python AST to flag three predefined anti-patterns:
overlap leakage, multi-test leakage, and preprocessing leakage. They achieve high
precision on these specific patterns but miss indirect leakage through variable aliasing,
function calls, custom transformers, or statistical operations whose leakage properties
depend on data flow rather than code shape. They provide no quantification—a
`StandardScaler` fitted on 100,000 training rows plus 1 test row is flagged identically
to one fitted on 50% test data, though the information leakage differs by orders of
magnitude. **Empirical comparison tools** (LeakGuard) detect leakage by executing the
pipeline with and without suspected contamination and measuring accuracy deltas. This
requires full re-execution, produces model-dependent proxy scores (not
information-theoretic quantities), and can miss leakage that does not happen to affect
accuracy on a particular dataset/model combination. Neither category provides sound
guarantees, quantitative measurement in bits, or per-feature per-stage attribution. We
argue that the field needs a fundamentally new class of diagnostic: a *differential
information auditor* that precisely quantifies test-set contamination across the entire
pipeline.

**TaintFlow** performs differential information auditing for ML pipelines. Given a
scikit-learn/pandas pipeline, TaintFlow constructs an abstract model of the pipeline's
information flow and computes a *leakage spectrum*: a per-feature, per-pipeline-stage
decomposition of exactly where and how much test-set information enters the training
process, measured in bits of mutual information. The technical architecture is a hybrid
of dynamic and static analysis, resolving a key tension identified in prior work between
Python's extreme dynamism and the need for formal guarantees. First, TaintFlow executes
the pipeline once under lightweight instrumentation (Python `sys.settrace` and AST-level
hooks) to extract the concrete dataflow DAG—resolving Python's dynamic dispatch, runtime
column names, and polymorphic API calls without attempting brittle static type inference
on arbitrary Python code. This dynamic phase observes actual types, shapes, and
operation sequences, producing a precise DAG that would be impossible to extract
statically from Python code using `getattr`, `**kwargs`, monkey-patching, and dynamic
column names. Second, TaintFlow applies a novel abstract interpretation engine to the
extracted DAG, propagating quantitative taint labels through every operation to compute
sound upper bounds on leakage. The abstract interpretation runs entirely in Rust for
performance, operating over a purpose-built partition-taint lattice that tracks both data
origins (train, test, external) and quantitative bit-bounds. This hybrid design
eliminates the critical weakness of pure static analysis for Python (dynamic features
defeat type inference in ~15–30% of real pipelines) while preserving the formal
soundness guarantees of abstract interpretation on the observed execution paths. The
result is a tool that works on real-world ML code—not just textbook pipelines—with
mathematically grounded output.

The key technical insight is that statistical operations in ML pipelines act as *lossy
information-theoretic channels* whose capacity can be bounded using each operation's
algebraic structure. Consider a `StandardScaler.fit()` called on a DataFrame containing
$n_{\text{tr}}$ training rows and $n_{\text{te}}$ test rows. The fitted `mean_`
parameter for feature $j$ is a sufficient statistic of the combined data: the sample
mean $\bar{X}_j = \frac{1}{n}\sum_{i=1}^{n} X_{ij}$. Viewed as a channel from the
$n_{\text{te}}$ test rows to this scalar output, the channel capacity is bounded by
$C_{\text{mean}} \leq \frac{1}{2}\log_2(1 + n_{\text{te}}/(n - n_{\text{te}}))$ bits
under the Gaussian channel model. For a typical scenario with 100 test rows among 10,000
total, this is approximately 0.007 bits per feature—negligible. But for a pipeline that
calls `fit_transform` on 50% test data (a common mistake in scripts that concatenate
before splitting), the bound is 0.5 bits per feature, and with 100 features, the total
contamination is 50 bits of test-set signal in the training representation. Similarly, a
`GroupBy.transform('mean')` on mixed data leaks at most $H(\text{group\_key})$ bits per
group; a `PCA.fit()` leaks at most $d^2 \cdot C_{\text{cov}}(n_{\text{te}}, n)$ bits
through the covariance matrix, where $C_{\text{cov}}$ is the channel capacity of a
single covariance entry; and a `TargetEncoder.fit()` on combined data leaks up to
$H(Y | \text{group})$ bits per categorical level. By composing these per-operation
channel capacity bounds through the pipeline DAG—using the data-processing inequality
for sequential stages and the chain rule for parallel branches—TaintFlow produces a
*leakage certificate*: a per-feature upper bound on the mutual information
$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}})$ between the test set and each output
training feature, expressed in bits. When the pipeline's data is available, an optional
empirical refinement phase tightens these bounds using non-parametric mutual information
estimation (KSG estimator), with the abstract bounds serving as sound upper clamps via a
formally defined reduced product of the two domains.

TaintFlow is the *first sound, quantitative* leakage analysis for real ML pipelines. No
prior work combines quantitative information flow theory (Alvim, Smith — measuring
leakage in bits via channel models), abstract interpretation (Cousot — sound
over-approximation with lattice-based fixpoints), and ML pipeline semantics (transfer
functions for pandas/sklearn operations) in a single framework. Existing taint analysis
tools in the security domain (FlowDroid for Android, TaintDroid, Joana for Java, Snyk
Code) track binary taint; they cannot quantify *how much* information flows through a
taint path, and they do not model statistical operations on DataFrames. Python-specific
security taint tools (Pysa, CodeQL for Python) track binary taint in Python programs but
lack quantitative information-flow analysis and statistical operation semantics. Quantitative
information flow (QIF) theory has been applied to cryptographic protocols, password
checkers, side-channel attacks, and database query privacy (geo-indistinguishability),
but never to ML preprocessing pipelines—despite the structural similarity between a
`SELECT AVG(salary) FROM employees` query channel and a
`StandardScaler.fit_transform()` channel. Neural network verification via abstract
interpretation (DeepPoly, AI², PRIMA) verifies properties of *trained models*
(robustness, fairness), not *training pipelines* (leakage, contamination). Work on
differential privacy (DP-SGD, privacy profiles, Rényi divergence accounting) provides
sound quantitative bounds for a *different* threat model—individual record privacy rather
than evaluation integrity—and targets noise-injection mechanisms rather than
deterministic preprocessing. Differential privacy asks whether an adversary can learn about any single individual from a pipeline's output; TaintFlow asks how much aggregate test-set signal contaminates training representations, inflating evaluation metrics—a fundamentally different threat model targeting evaluation integrity rather than individual privacy. TaintFlow bridges these adjacent fields at their unexplored
intersection, defining novel abstract domains for DataFrame-level information flow,
deriving channel capacity bounds for the 80 most common pandas/sklearn operations, and
proving that the resulting analysis is sound. The contribution opens a new research
direction: *quantitative information-flow analysis for data science code*.

---

## Value Proposition

**Who needs this, and why desperately:**

- **ML platform teams** at companies running thousands of feature pipelines (Airbnb, Spotify, Netflix, any organization with a feature store). A single leakage bug in a production feature pipeline silently inflates offline model metrics, leading to overconfident deployment decisions and real-world performance degradation. Today, these teams rely on manual code review—error-prone and unscalable. TaintFlow provides a CI/CD gate that blocks leaky pipelines before deployment, with per-feature leakage reports that tell engineers exactly *which* feature, *which* pipeline stage, and *how many bits* of contamination.

- **ML practitioners debugging metric discrepancies.** The most common question on ML forums is "why do my offline and online metrics disagree?" In a large fraction of cases, the answer is leakage, but practitioners have no diagnostic tool. TaintFlow's leakage spectrum directly answers: "Feature X has 3.7 bits of test-set contamination introduced by `StandardScaler.fit_transform()` applied before `train_test_split()`, which inflates accuracy by an estimated 2.1 percentage points."

- **Regulatory bodies and auditors** in high-stakes domains (credit scoring under ECOA/FCRA, clinical trial ML under FDA guidance, recidivism prediction under COMPAS scrutiny). Current audit practices cannot systematically verify that a model's reported accuracy is not inflated by leakage. A TaintFlow leakage certificate—a machine-checkable, per-feature bound on test-set contamination—provides the kind of auditable evidence that regulatory frameworks increasingly demand.

- **ML competition platforms** (Kaggle, DrivenData) that currently rely on private leaderboards as a blunt instrument against leakage. TaintFlow could provide automated leakage scoring for submitted kernels, improving competition integrity without requiring re-execution of every submission.

- **ML researchers and reviewers** who need to verify that published results are leakage-free. A TaintFlow report attached to a paper submission would serve as a reproducibility artifact, analogous to a proof-of-correctness certificate.

**What becomes possible that wasn't before:** For the first time, an ML practitioner can run a single command on their pipeline and receive a quantitative, per-feature, per-stage leakage decomposition with formal soundness guarantees—without modifying their code, without re-executing the pipeline multiple times, and without any machine learning expertise in information theory. The output transforms leakage debugging from "guess and re-run" into "read the report and fix the flagged stage."

---

## Technical Difficulty

This project requires genuine breakthroughs at the intersection of information theory, program analysis, and ML systems engineering. The core difficulty is not in any single component but in making three independently hard problems compose correctly.

### Hard Subproblem 1: Defining Sound Abstract Transfer Functions for Statistical Operations

Unlike standard security taint analysis (where taint is binary—tainted or not),
information flow through statistical aggregations is inherently *quantitative* and
*lossy*. Bounding the channel capacity of
`GroupBy.transform(lambda x: x.rolling(7).mean())` requires combining
information-theoretic reasoning (how much information about test rows survives averaging
within a group?) with abstract interpretation soundness guarantees (the bound must hold
for *all* possible input data, not just typical cases). For the 80 most common pandas
operations and 50 most common sklearn estimators targeted in this work, each operation
has genuinely different mathematical semantics:

- **Per-column statistics** (`StandardScaler`, `MinMaxScaler`, `RobustScaler`): Compute
  independent statistics per feature. Channel capacity scales as
  $O(\log(1 + n_{\text{te}}/n_{\text{tr}}))$ per feature—well-understood.
- **Cross-column operations** (`PCA`, `TruncatedSVD`, `NMF`): Mix information across
  *all* columns through covariance estimation or matrix factorization. Leakage scales
  with $d^2$ (number of covariance entries), making these the highest-capacity channels
  in typical pipelines.
- **Group-based operations** (`GroupBy.transform`, `TargetEncoder`): Capacity depends on
  group cardinality and the distribution of test rows across groups—inherently
  data-dependent. Static bounds must over-approximate group structure using worst-case
  cardinality estimates.
- **Iterative operations** (`IterativeImputer`, `KNNImputer`): The channel from test
  rows to imputed values depends on convergence behavior (iterative) or the
  nearest-neighbor graph (KNN), which are entirely data-dependent. No static bound
  exists that isn't trivially $H(\mathcal{D}_{\text{te}})$.
- **User-defined operations** (lambdas in `.apply()`, custom transformers): No static
  bound possible for arbitrary code.

For the first two categories (~30–40 operations), closed-form channel capacity bounds
are derivable and expected to be tight. For group-based operations, bounds are derivable
but may be loose (over-approximate by 10–50×). For the last two categories, the tool
must gracefully degrade to conservative $\infty$ bounds while still providing useful
analysis for the remaining pipeline stages. This graduated precision is a fundamental
design constraint, not a limitation to be apologized for: the tool is honest about what
it can and cannot bound.

### Hard Subproblem 2: Compositionality Across the Pipeline DAG

Pipelines involve mutable state (fitted estimators whose parameters carry latent information about training data), the fit/predict paradigm (where `fit()` aggregates data into parameters and `transform()` propagates those parameters into outputs), and dynamic shapes (feature selection changes dimensionality mid-pipeline). The `fit_transform` pattern is particularly challenging: it is not a pure sequential channel (the output depends on statistics computed *from* the input, which is then applied *to* the same input), making standard data-processing inequality arguments require careful reformulation. Cross-validation wrappers (`GridSearchCV`, `cross_val_score`) create implicit loops over train/test splits with shared state, requiring specialized abstract semantics. The fixpoint computation over the pipeline DAG must handle all of this while maintaining sound bounds and terminating in reasonable time on a laptop CPU.

### Hard Subproblem 3: Bound Tightness for Practical Utility

Sound over-approximation is necessary but not sufficient. If the abstract bounds are 100× the true leakage for common operations, the tool reports "≤ 320 bits" when the truth is 3.2 bits, and the quantitative claim is vacuous. The paper must empirically demonstrate that bounds are within 10× of true leakage for the 30–40 operations where closed-form channel capacity bounds exist (simple aggregates: mean, std, var, sum, count, min, max, quantile). For operations where bounds are looser, the narrative pivots to *leakage severity ordering*: the tool correctly ranks features by leakage magnitude, even if absolute numbers carry a multiplicative slack. This dual-mode interpretation (tight absolute bounds where achievable, correct relative ordering elsewhere) is essential for the paper to withstand reviewer scrutiny on the "are the numbers meaningful?" question.

### Hard Subproblem 4: Hybrid Dynamic-Static Architecture at Scale

The dynamic DAG extraction phase must instrument arbitrary Python ML code without perturbing its behavior, capturing every DataFrame operation, sklearn API call, and data-flow edge with minimal overhead. The instrumentation must handle pandas method chains (`df.groupby('a').transform('mean').fillna(0)`), sklearn compositional patterns (`Pipeline`, `ColumnTransformer`, `FeatureUnion` with nested sub-pipelines), and implicit operations (index alignment during merge, broadcasting during arithmetic). The extracted DAG must be serialized into a form consumable by the Rust analysis engine. The static analysis phase then applies the abstract interpretation over this DAG, computing fixpoints with context-sensitive analysis (the same transformer may appear in multiple pipeline stages with different column subsets), widening operators for convergence guarantees, and parallel branch analysis via Rayon for laptop-CPU performance.

### Estimated Subsystem Breakdown (~155K LoC)

| Subsystem | Estimated LoC | Notes |
|-----------|--------------|-------|
| **Dynamic DAG extraction & instrumentation** | ~15K | Python `sys.settrace`/AST hooks, tree-sitter for lightweight parsing, pandas/sklearn API interception, DAG serialization to Rust-consumable format |
| **Abstract domain engine** (partition-taint lattice + channel capacity) | ~22K | Core lattice infrastructure, powerset taint domain (roaring bitmaps), entropy-approximation domain (~30 entropy-bounding lemmas), cardinality domain, reduced product construction, property-based test harnesses for all lattice laws |
| **Transfer functions for pandas operations** (80 ops) | ~18K | `merge`/`join`/`concat`, `groupby`+aggregation, reshaping, selection/filtering, fill/imputation, arithmetic/statistical, window operations, index operations — each with abstract taint + entropy + cardinality semantics. Reduced from 25K by targeting 80 most common operations and using a specification-driven code generator for structurally similar operations |
| **Transfer functions for sklearn operations** (50 estimators) | ~15K | Preprocessing transformers, encoding transformers, imputation, feature selection, decomposition/manifold, pipeline/composition, cross-validation wrappers — each with abstract `fit`/`transform`/`predict` semantics. Reduced from 20K by targeting 50 most common estimators |
| **Information-flow propagation engine** (fixpoint) | ~16K | Worklist algorithm with RPO ordering, context-sensitivity engine (k-CFA style), function summary computation for user-defined transformers, widening/narrowing strategies with delayed widening, incremental re-analysis support, Rayon-based parallel branch analysis |
| **Quantitative analysis & attribution** | ~10K | Bits-of-leakage calculator, leakage path tracer (critical-edge identification), leakage classifier (target/feature/preprocessing/CV leakage), severity scoring, remediation suggestions, min-cut-based per-stage attribution |
| **Report generator** (SARIF, terminal, JSON) | ~8K | SARIF v2.1 compliance for GitHub/VS Code integration, colored terminal output with ASCII flow diagrams, JSON for CI/CD, diff-based reporting across code versions |
| **Python bindings & CLI** | ~8K | PyO3 bindings exposing analysis API, Click-based CLI, configuration file support, output format selection, Jupyter integration helpers |
| **Test infrastructure** | ~20K | Property-based tests for lattice laws (proptest), unit tests for all transfer functions, integration tests (end-to-end on synthetic pipelines), evaluation harness for automated precision/recall computation |
| **Benchmark suite** (synthetic + Kaggle corpus) | ~15K | Synthetic pipeline generator (~30 leakage patterns × parameterized instantiation), real-world corpus curation (200+ Kaggle kernels), empirical ground-truth oracle, benchmark runner and reporting |
| **Infrastructure** (build, CI, PyO3) | ~3K | Cargo workspace, CI configuration, PyO3 build glue, release packaging |
| **Total** | **~150K** | |

The total lands at ~150K LoC. The reduction from the original 181K estimate comes from three changes: (1) replacing the custom Python static analysis frontend (35K) with dynamic DAG extraction via tree-sitter + instrumentation (15K), saving ~20K; (2) reducing pandas operations from 120 to 80 and sklearn estimators from 80 to 50, saving ~12K; (3) using specification-driven code generation for structurally similar transfer functions, saving ~5K. These reductions are partially offset by test and benchmark infrastructure for the hybrid architecture.

---

## New Mathematics Required

Six load-bearing mathematical contributions (M1–M6) form the theoretical core of the paper.

### M1: Partition-Taint Lattice for DataFrames

**Informal statement:** Define a complete lattice $\mathcal{T} = (\mathcal{P}(\mathcal{O}) \times [0, B_{\max}], \sqsubseteq)$ where each element is a pair $(O, b)$ of data origins $O \subseteq \{\text{tr}, \text{te}, \text{ext}\}$ and an upper bound $b$ on bits of test-set mutual information. Extend this to a DataFrame abstract domain $\mathcal{A}_{\text{df}} = (\text{ColNames} \to \mathcal{T}) \times \mathcal{R}$ with row-provenance tracking $\mathcal{R} \in \{\texttt{train-only}, \texttt{test-only}, \texttt{mixed}(\rho)\}$. Prove this forms a Galois connection with the concrete domain of (DataFrame, partition) pairs.

**Why load-bearing:** This is the foundational mathematical object. Without it, there is no abstract domain, no transfer functions, no fixpoint computation, no soundness theorem. Every other contribution depends on M1. The lattice must simultaneously track *qualitative* origin information (which partitions contributed to this value) and *quantitative* information content (how many bits). This dual-tracking in a single lattice element is what enables both the soundness guarantee (via abstract interpretation theory) and the quantitative output (via information-theoretic bounds).

**Novelty level:** Genuinely novel (★★★). Abstract interpretation has been applied to numerical programs (Astrée), pointer analysis, and neural networks (DeepPoly), but never to DataFrame-level information flow with quantitative bit-bounds. The combination of set-valued origin tracking with quantitative channel capacity bounds in a single lattice is new. Estimated effort: 1.5 person-months.

### M2: Channel Capacity Bounds for Statistical Aggregates

**Informal statement:** For each of the ~30 most common statistical operations $\phi \in \{\text{mean}, \text{std}, \text{var}, \text{median}, \text{quantile}_p, \text{sum}, \text{count}, \text{min}, \text{max}, \text{cov}, \text{corr}, \ldots\}$, derive a closed-form upper bound on the channel capacity $C_\phi(n_{\text{te}}, n, d)$ when $\phi$ is computed over a mixture of $n_{\text{te}}$ test rows and $n - n_{\text{te}}$ train rows. For example: $C_{\text{mean}}(n_{\text{te}}, n) \leq \frac{1}{2}\log_2(1 + n_{\text{te}}/(n - n_{\text{te}}))$ bits under the Gaussian channel model. Prove the bounds are tight for Gaussian data and within $O(\log n)$ for sub-Gaussian data.

**Why load-bearing:** These bounds are the "fuel" for the abstract transformers in M1. Without them, the bit-bound $b$ in the lattice would always be $\infty$—trivially sound but useless. The precision of the entire analysis depends on the tightness of these per-operation bounds. This is where the paper's quantitative claims live or die: if bounds are within 10× of true leakage for common operations, the tool provides meaningful diagnostics; if they are 100× loose, the tool degenerates to qualitative taint analysis.

**Novelty level:** Moderate-to-high (★★☆). The individual bounds for simple aggregates follow from textbook information theory (Gaussian channel capacity). The novelty is in the *systematic catalog* for all sklearn-relevant operations, the *sub-Gaussian tightness results*, and the bounds for non-trivial operations like `groupby().transform()` and `PCA.fit()` where the channel structure involves data-dependent group cardinalities or matrix eigenstructure. Approximately 60% novel, 40% textbook adaptation. Estimated effort: 2 person-months.

### M3: Soundness of Abstract Pipeline Analysis

**Informal statement:** For any pipeline $\pi$, dataset $\mathcal{D}$ with partition $(\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}})$, and the DAG $G$ extracted from executing $\pi$ on $\mathcal{D}$, the abstract fixpoint $\sigma^\sharp$ computed by the worklist algorithm satisfies $I(\mathcal{D}_{\text{te}}; \pi(\mathcal{D})_j) \leq \sigma^\sharp_j.b$ for all output features $j$. That is, the reported bit-bound is a sound over-approximation of the true mutual information between the test set and each output training feature. Note: soundness holds relative to the observed execution path, which is standard for dynamic-analysis-assisted static tools (cf. concolic testing, hybrid type inference).

**Why load-bearing:** This is the central theorem. It transforms the tool from "a heuristic that guesses leakage" into "a verified analysis that certifies leakage bounds." It is the qualitative leap beyond LeakageDetector (which can miss leakage it doesn't pattern-match) and LeakGuard (which gives a model-dependent empirical estimate). Reviewers will judge the paper primarily on whether this theorem is correct and non-trivial. The proof combines induction on the pipeline DAG with the data-processing inequality applied through abstract transformers, with the main difficulty being the `fit_transform` pattern (not a pure sequential channel, since the output depends on statistics of the input applied back to the same input). The proof requires a dedicated fit_transform channel lemma establishing that for an estimator $e$, the mutual information through fit_transform decomposes into the channel capacity of the fitting operation plus the direct data flow through the transform, providing the non-trivial inductive step.

**Novelty level:** Genuinely novel (★★★). The proof technique—DPI-based induction through abstract transformers over a DataFrame information-flow lattice—has no precedent. The closest work is soundness proofs for quantitative information-flow type systems (Smith 2009, Alvim et al. 2012), but those operate on imperative programs with scalar variables, not DataFrame pipelines with statistical operations and the fit/predict paradigm. Estimated effort: 2.5 person-months.

### M4: Sensitivity Types for Pipeline Operation Composition

**Informal statement:** Define a sensitivity type system where each pipeline operation $s_k$ is annotated with a function $\delta_k: [0,\infty]^{d_{\text{in}}} \to [0,\infty]^{d_{\text{out}}}$ mapping input bit-bounds to output bit-bounds, satisfying three properties: (1) monotonicity — more input leakage implies at least as much output leakage; (2) sub-additivity — leakage from independent sources adds at most linearly; (3) DPI consistency — no stage amplifies total leakage beyond its inputs. Prove that sensitivity types compose: $\delta_\pi = \delta_K \circ \cdots \circ \delta_1$ is a valid sensitivity type for the composed pipeline $\pi$.

**Why load-bearing:** Sensitivity types are the compositional backbone. Without them, the analysis must reason about the entire pipeline monolithically, which does not scale. Sensitivity types enable *modular verification*: prove each operation's transfer function correct once, then compose freely. They are the information-theoretic analogs of Lipschitz conditions in differential privacy (ε-DP composition) and condition numbers in numerical analysis, instantiated for the novel domain of statistical operations on DataFrames.

**Novelty level:** Genuinely novel (★★★). This is the contribution most likely to inspire follow-up work. The formulation as a type system for information-flow in statistical operations, with the specific DPI-consistency requirement, is new. The individual composition properties (monotonicity, sub-additivity) exist in differential privacy and Lipschitz analysis, but their instantiation for DataFrame operations with the fit/predict paradigm is original. Estimated effort: 2 person-months.

### M5: Reduced Product of Abstract and Empirical Domains

**Informal statement:** Define the reduced product $\mathcal{A}_{\text{hybrid}} = \mathcal{A}_{\text{taint}} \otimes \mathcal{A}_{\text{MI}}$ with reduction operator $\rho(\tau, \hat{I}) = (\tau[b \mapsto \min(b, \hat{I})], \min(\hat{I}, \tau.b))$. Prove that the reduced product is strictly more precise than either component alone: $\gamma_{\text{hybrid}}(\rho(\tau, \hat{I})) \subsetneq \gamma_{\text{taint}}(\tau)$ for any pipeline with at least one leakage site where the abstract bound is not tight.

**Why load-bearing:** This theorem justifies the hybrid static-dynamic design. Without it, one could argue that the abstract interpretation (Phase 1) and the empirical MI estimation (optional Phase 2) are independent analyses whose results are simply presented side by side. The strict precision improvement demonstrates that the combination is synergistic: the abstract bound clamps the empirical estimate from above (ensuring soundness even when the KSG estimator overestimates due to finite samples), while the empirical estimate tightens the abstract bound from below (improving precision for operations where the channel capacity bound is loose). The reduction operator uses a one-sided confidence upper bound from the KSG estimator (adding a confidence margin $\varepsilon_\alpha$) to ensure soundness at confidence level $1-\alpha$, rather than the raw KSG estimate. This dual-mode clamping is itself a novel technical contribution.

**Novelty level:** Moderate-to-high (★★☆). Reduced products are a standard technique in abstract interpretation (Cousot & Cousot 1979). The novelty is in the specific combination of a partition-taint lattice with an MI-estimate domain, and in the proof that the reduction operator yields strict improvement for the information-flow setting. Approximately 70% novel. Estimated effort: 1.5 person-months.

### M6: Information-Theoretic Min-Cut for Per-Stage Attribution

**Informal statement:** The leakage from $\mathcal{D}_{\text{te}}$ to output feature $j$ is bounded by the minimum information-theoretic cut in the pipeline DAG: $I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq \min_{\text{cut}} \sum_{e \in \text{cut}} I_e$. This bound is tight for deterministic pipelines (which ML preprocessing pipelines are). The min-cut identifies the pipeline stage(s) that are the bottleneck for leakage, enabling per-stage attribution without requiring the exponential-cost Shapley decomposition.

**Why load-bearing:** This theorem provides the structural decomposition that makes per-stage leakage attribution tractable. Without it, the tool can only report *total* leakage per feature but cannot answer "which pipeline stage is responsible?" The min-cut computation identifies the critical stage(s) in polynomial time, providing actionable diagnostics: "the leakage bottleneck is the `StandardScaler.fit_transform()` at line 47, which contributes 2.1 of the total 3.2 bits." When multiple minimum cuts exist, TaintFlow reports the cut closest to the data source (earliest pipeline stage) as the primary attribution target, with alternative cuts available in the detailed report. This is the contribution that transforms a theoretical framework into a practical debugging tool.

**Novelty level:** Moderate (★★☆). The information-theoretic max-flow/min-cut theorem is well known (Cover & Thomas, Chapter 15). The novelty is in applying it to ML pipeline DAGs, proving tightness for deterministic operations, and using it as a tractable replacement for exponential-cost Shapley attribution. Approximately 30% novel, but the application is impactful. Estimated effort: 1 person-month.

### Math Tier Summary

| Tier | Contributions | Timeline | Total Effort |
|------|--------------|----------|--------------|
| **Tier 1** (critical path, months 1–4) | M1, M2, M3 | Must complete before transfer function implementation | ~6 person-months |
| **Tier 2** (high value, months 4–8) | M4, M5, M6 | Enables compositionality and attribution | ~4.5 person-months |
| **Tier 3** (nice-to-have, if time permits) | M7 (linear-Gaussian closed form), M8 (complexity), M9 (finite-sample concentration), M10 (Shapley attribution) | Validation and extensions | ~6 person-months |
| **Paper includes** | M1–M6 | 6 load-bearing contributions | ~12 person-months |

---

## Best Paper Argument

### The "One New Idea"

The partition-taint lattice equipped with channel capacity bounds, composed via sensitivity types. This is a single, clean intellectual contribution that bridges three mature fields (quantitative information flow, abstract interpretation, ML pipeline semantics) at an intersection that nobody has explored. A program committee member can evaluate the idea in one sentence: *"We define the first quantitative information-flow abstract domain for statistical operations on tabular data, prove soundness of the resulting pipeline analysis, and demonstrate that the bounds are tight enough to catch real leakage in thousands of Kaggle pipelines."* The idea is falsifiable (the soundness theorem either holds or it doesn't), quantitative (precision and bound tightness are measurable), and novel (no prior work has formalized information-flow abstract domains for pandas/sklearn operations).

### Theoretical Depth

The soundness theorem (M3) requires a non-trivial proof combining induction on the pipeline DAG with the data-processing inequality applied through abstract transformers—with the specific complication that `fit_transform` creates a non-standard channel structure (feedback from computed statistics to the same data). The sensitivity type system (M4) provides a principled compositional framework whose monotonicity, sub-additivity, and DPI-consistency properties are the information-theoretic analogs of Lipschitz conditions, but instantiated for a novel domain. The reduced product theorem (M5) proves strict precision improvement from combining abstract and empirical analysis—a result that is easy to state but requires careful construction of the reduction operator to achieve. These six theorems (M1–M6) collectively represent genuine mathematical novelty, not just engineering dressed in formalism.

### Practical Impact

TaintFlow runs on real Kaggle pipelines. The evaluation on 200+ real-world kernels with known leakage patterns, plus 500+ synthetic pipelines with calibrated leakage injection, demonstrates that the framework catches real bugs that no existing tool detects—including indirect leakage through statistical operations, function calls, and non-standard pipeline compositions that LeakageDetector's pattern matching misses. The tool produces actionable output: per-feature leakage bounds in bits, per-stage attribution identifying the critical operation, and severity classification (negligible / warning / critical). For linear-Gaussian pipelines, the bounds are provably tight to within $O(\log d)$; for the 30 most common aggregation operations, empirical validation demonstrates bounds within 10× of true leakage—tight enough for meaningful severity ordering and practical debugging.

### Opens a New Subfield

TaintFlow establishes "quantitative information-flow analysis for data science code" as
a research direction. The partition-taint lattice, channel capacity catalog, and
sensitivity type system are all extensible: future work can add transfer functions for
new libraries (PyTorch data loaders, Spark ML pipelines, R's tidymodels, Julia's
MLJ), define channel capacities for new operation types (deep learning preprocessing,
graph neural network feature engineering, time-series windowing), and explore tighter
bounds using richer abstract domains (relational domains that track inter-column
correlations, widening strategies that exploit distributional assumptions). The framework
also invites natural theoretical follow-ups:

- Can we define *principal sensitivity types* with inference, eliminating the need for
  manual annotation of transfer functions?
- Can we extend the soundness theorem to pipelines with *stochastic* operations (data
  augmentation, dropout, random feature sampling)?
- Can we connect the leakage certificate to *downstream model performance bounds*—i.e.,
  "if your pipeline leaks ≤ X bits, your accuracy is inflated by at most Y percentage
  points"?
- Can we generalize from train-test leakage to *temporal leakage* in production
  pipelines (where features computed from future data contaminate past predictions)?
- Can we adapt the framework to *federated learning* pipelines, where leakage occurs
  across organizational boundaries rather than across data partitions?

These are substantial, publishable research questions that the paper opens without
answering—the hallmark of a subfield-creating contribution. The interdisciplinary nature
of the work (PL theory × information theory × ML engineering) ensures that follow-up
can come from multiple research communities, amplifying impact.

### Why This Wins Over Simpler Alternatives

A reviewer might ask: "Why not just run the pipeline twice with different splits and compare?" (The LeakGuard approach.) The answer is threefold. First, empirical comparison is model-dependent—it measures how much leakage affects *this* model on *this* dataset, not an intrinsic property of the pipeline. Second, it requires full re-execution per check, making it impractical for CI/CD gates on thousands of pipelines. Third, it provides no formal guarantee: a lucky test split might show no metric difference even when leakage exists. TaintFlow provides *sound, model-independent, per-feature bounds* from a single analysis pass—a qualitatively different capability. Another reviewer might ask: "Why not just extend LeakageDetector with quantitative annotations?" Adding channel capacity formulas to pattern-matched leakage sites would not be *sound*: LeakageDetector misses leakage sites that its three patterns don't cover, so the quantitative annotations would apply only to found sites. TaintFlow's abstract interpretation guarantees that *all* information-flow paths are analyzed—no leakage site is missed, even if the bound at some sites is conservative.

---

## Evaluation Plan

All evaluation is fully automated, requires no human annotation, and runs on a laptop CPU.

### Synthetic Benchmark Suite (500+ pipelines)

A parameterized pipeline generator produces synthetic pipelines with *calibrated leakage
injection*. Each synthetic pipeline specifies:

- **Pipeline topology:** Linear chains (5–50 stages), branching/merging DAGs (via
  `ColumnTransformer` with 2–10 branches), and nested compositions (pipelines within
  pipelines).
- **Feature dimensionality:** 10–500 columns, with a mix of numerical, categorical, and
  text features.
- **Leakage injection points:** 0–10 per pipeline, at known locations, with controlled
  magnitude. Leakage is injected by deliberately calling `fit_transform` on combined
  train+test data, computing target-encoded features using the full target column,
  applying `fillna(df.mean())` before splitting, or other canonical anti-patterns.
- **Leakage magnitude:** Controlled by adjusting the test fraction $\rho$ at each
  injection point: $\rho \in \{0.0, 0.001, 0.01, 0.1, 0.5\}$. At $\rho = 0$, there is
  no leakage; at $\rho = 0.5$, half the data is test data, maximizing contamination.
- **Operation types:** Drawn from ~30 canonical leakage patterns (fit-before-split,
  target encoding, time-series lookahead, grouped aggregation on mixed data, feature
  selection on combined data, rolling statistics across time boundaries,
  imputation using global statistics, cross-validation with external feature
  computation, etc.)

Ground truth is known by construction: each pipeline's true leakage is determined by
its generation parameters. For the linear-Gaussian subset (~100 pipelines where all
operations are linear and data is sampled from a multivariate Gaussian), exact mutual
information is computable in closed form via M7, providing a gold-standard tightness
comparison with zero approximation error.

### Real-World Corpus (200+ Kaggle kernels)

Scripts automatically download, clean, and standardize 200+ real ML pipelines from Kaggle
kernels identified in the Yang et al. (ASE 2022) study as containing leakage, plus a
control set of 100+ leakage-free pipelines. The corpus covers diverse domains (tabular
classification, regression, time series, NLP feature engineering) and diverse pipeline
complexities (10–500 lines of pandas/sklearn code). For each pipeline:

- **Empirical oracle:** Execute with and without leakage (by programmatically rewriting
  the pipeline to move preprocessing inside a proper `Pipeline`), measure accuracy delta,
  and convert to an information-theoretic proxy via
  $\Delta I \approx H(\text{accuracy}_{\text{leaky}}) - H(\text{accuracy}_{\text{clean}})$
  under a logistic model of the accuracy-information relationship. This proxy is
  acknowledged as imperfect (model-dependent, finite-sample) but provides the best
  available reference for real-world quantitative ground truth.
- **Binary ground truth:** Whether the pipeline contains leakage, from Yang et al.'s
  manually verified labels.
- **Leakage pattern labels:** Which category of leakage (preprocessing, target, temporal,
  overlap) from Yang et al.'s taxonomy, enabling per-pattern precision/recall analysis.

### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Precision** | Fraction of reported leakage sites that are true leakage | ≥ 0.85 |
| **Recall** | Fraction of true leakage sites detected | ≥ 0.95 (soundness implies 1.0 on analyzed paths) |
| **F1** | Harmonic mean of precision and recall | ≥ 0.90 |
| **Bound tightness ratio** | Median of (abstract bound) / (empirical leakage) across operations | ≤ 10× for top-30 operations |
| **Severity ordering accuracy** | Spearman rank correlation between reported and true leakage across features | ≥ 0.8 |
| **Analysis time** | Wall-clock time per pipeline on laptop CPU (8-core, 16GB RAM) | ≤ 30 seconds median |
| **Pipeline coverage** | Fraction of real-world pipelines that produce meaningful (non-$\infty$) results | ≥ 0.90 (via hybrid architecture) |

### Baselines

1. **LeakageDetector (Yang et al., ASE 2022 / SANER 2025):** Binary pattern-matching
   static analysis. Detects three predefined anti-patterns: overlap leakage, multi-test
   leakage, and preprocessing leakage. Expected to have high precision on its three
   patterns but low recall on indirect leakage (through function calls, variable
   aliasing, custom transformers, non-standard pipeline compositions) and zero
   quantitative output. The comparison demonstrates the value of semantic analysis over
   syntactic pattern matching.

2. **Binary taint analysis:** An ablation of TaintFlow that replaces the quantitative
   partition-taint lattice with a standard binary taint domain (tainted/untainted, no
   bit-bounds). This baseline uses the same dynamic DAG extraction and the same
   propagation engine, isolating the contribution of the channel capacity bounds (M2)
   and the quantitative lattice (M1). Expected to achieve similar recall (taint
   propagation is sound) but with no severity ordering capability—every leaking feature
   is equally "tainted" regardless of whether it carries 0.01 bits or 50 bits of
   contamination.

3. **Empirical comparison baseline ("run twice"):** Execute each pipeline with and
   without suspected leakage, measure accuracy delta. This is the LeakGuard approach
   applied uniformly across the benchmark. Expected to have reasonable detection power
   for large leakage (>1 bit) but miss small leakage that doesn't measurably affect
   accuracy on a particular dataset. Also expected to fail on pipelines where the
   leakage-free version cannot be automatically constructed (non-standard pipeline
   structures). This comparison demonstrates the value of static quantitative analysis
   over dynamic empirical comparison.

4. **Random baseline:** Reports each feature as leaking with probability equal to the
   base rate in the corpus. Establishes the difficulty of the detection task and ensures
   reported metrics are above chance.

### Ground Truth Strategy

- **Synthetic pipelines:** Ground truth is exact by construction. True leakage magnitude
  is a generation parameter. True leakage location is the injection point. This corpus
  enables evaluation of both detection (is leakage found?) and quantification (is the
  reported bound within 10× of the true value?).

- **Linear-Gaussian pipelines:** A subset of synthetic pipelines where all operations are
  linear transformations and data is drawn from a multivariate Gaussian. Exact mutual
  information is computable in closed form via M7 (the linear-Gaussian closed-form
  result), providing a zero-error gold standard for tightness evaluation. This is the
  strongest possible validation: if TaintFlow's bounds are not tight on linear-Gaussian
  pipelines, the framework is fundamentally miscalibrated.

- **Real-world pipelines:** Binary labels from Yang et al.'s annotations. Quantitative
  ground truth approximated via the empirical oracle (run-twice accuracy delta converted
  to bits). The empirical oracle is acknowledged as imperfect (model-dependent,
  finite-sample, and unable to isolate per-feature leakage) but provides the best
  available reference for aggregate leakage magnitude. We report both the binary
  detection metrics (precision, recall, F1 against Yang et al.'s labels) and the
  quantitative correlation (Spearman rank correlation between TaintFlow's severity
  ordering and the empirical oracle's accuracy delta) to give a complete picture of
  practical utility.

---

## Laptop CPU Feasibility

TaintFlow is designed from the ground up for laptop-CPU execution. Every architectural
decision is shaped by the constraint that analysis must complete in under 30 seconds for
a typical pipeline on an 8-core laptop with 16GB RAM.

### Why the Analysis Is Inherently Cheap

The abstract interpretation operates over a finite lattice of small height. Each element
of the partition-taint lattice is a pair $(O, b)$ where $O$ is drawn from
$\mathcal{P}(\{\text{tr}, \text{te}, \text{ext}\}) = 8$ elements and $b$ is drawn from
$\{0, 1, 2, \ldots, B_{\max}, \infty\}$ with $B_{\max} = 64$ in the default
configuration. The total lattice height (longest ascending chain) is therefore $3 + 65 = 68$, guaranteeing
termination without widening for acyclic pipelines. For a typical pipeline with $K = 50$
stages and $d = 200$ features, the fixpoint computation requires at most
$K \times d \times 68 \approx 6.8 \times 10^5$ lattice element updates—each of which
is a constant-time operation (set union + max). In Rust, this completes in well under
1 second.

### Architectural Decisions for Performance

1. **Rust core engine.** All performance-critical analysis (abstract interpretation,
   fixpoint computation, channel capacity calculations, DAG operations) runs in Rust,
   providing C-level performance with memory safety. A Python-only implementation would
   be 10–50× slower for the fixpoint computation over lattice states with 100+ columns.
   PyO3 bindings expose the analysis API to Python for the CLI and benchmark layers.

2. **Lightweight dynamic instrumentation.** The DAG extraction phase executes the user's
   pipeline *once* under Python's `sys.settrace` and AST-level hooks. For typical Kaggle
   pipelines (seconds to minutes of execution), this adds <20% overhead. The
   instrumentation captures operation types, column names, shapes, and data-flow
   edges—nothing more. No data values are copied or stored; only metadata flows to the
   Rust engine.

3. **Efficient data structures.** Roaring bitmaps for taint set operations (Subsystem C),
   hash-consing for abstract states (to share identical lattice elements across columns),
   arena allocation for DAG nodes (to avoid per-node heap allocation). These reduce
   memory footprint by 3–5× compared to naive representations, keeping the analysis
   within 1–2 GB for pipelines with 500+ columns.

4. **Rayon-based parallelism.** Independent pipeline branches (e.g., the 5–10 parallel
   transformers inside a `ColumnTransformer`) are analyzed in parallel on 8–16 cores,
   yielding 4–6× speedup for pipelines with parallel feature engineering paths. Join
   points merge abstract states from parallel branches.

5. **No neural components.** All analysis is symbolic/algebraic. No ML-based heuristics
   requiring GPU for training or inference. Results are deterministic and reproducible
   across runs—essential for CI/CD integration and regression testing.

### Evaluation Runtime Budget

The full benchmark suite (500+ synthetic + 200+ real pipelines) runs as a single
orchestrated invocation. At ~30 seconds median per pipeline for the analysis phase
(excluding pipeline execution time for the dynamic DAG extraction, which varies by
pipeline), the full suite completes in ~6 hours on a laptop—comfortably within an
overnight run. The synthetic pipeline generator, empirical oracle, metric computation,
and report generation are all scripted with no human-in-the-loop steps. No human
annotation, no manual labeling, no interactive decisions. The entire evaluation is
reproducible from a single `make bench` command.

### Why Execution of Analyzed Pipelines Is Feasible

A potential concern with the hybrid architecture is that executing the analyzed pipeline
requires the pipeline's data. For the evaluation corpus, this is not an obstacle: Kaggle
kernels include their datasets, and synthetic pipelines generate their own data. For
production use, TaintFlow requires a single execution of the pipeline (which the user
presumably runs already during development). The execution itself is the user's existing
workflow; TaintFlow merely instruments it. No additional data access beyond what the
pipeline already requires is needed.

---

## Slug

```
ml-pipeline-leakage-auditor
```
