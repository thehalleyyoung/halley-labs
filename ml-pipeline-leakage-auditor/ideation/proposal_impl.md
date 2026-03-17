# Implementation Scope: ML Pipeline Leakage Auditor

## Quantitative Information-Flow Analysis for Train–Test Leakage Detection

**Community:** area-041-machine-learning-and-ai-systems
**Phase:** Crystallization — Implementation Scope
**Language:** Rust (core engine) + Python (bindings, benchmarks, CLI glue)

---

## 1. SYSTEM ARCHITECTURE

### 1.1 High-Level Data Flow

```
                         ┌──────────────────────────────────────────┐
                         │         User's Python ML Pipeline        │
                         │   (sklearn, pandas, numpy source code)   │
                         └──────────────┬───────────────────────────┘
                                        │ .py files
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  A. Python AST / Bytecode Analyzer    │
                    │  - Parse Python source → typed IR     │
                    │  - Resolve imports, aliases, closures │
                    │  - Handle dynamic dispatch patterns   │
                    └──────────────┬────────────────────────┘
                                   │ Typed IR (serialized)
                                   ▼
                    ┌───────────────────────────────────────┐
                    │  B. Pipeline DAG Extractor            │
                    │  - Recognize sklearn Pipeline,        │
                    │    ColumnTransformer, FeatureUnion     │
                    │  - Build operation dependency graph   │
                    │  - Resolve fit/transform call chains  │
                    └──────────────┬────────────────────────┘
                                   │ Pipeline DAG
                                   ▼
              ┌────────────────────────────────────────────────────┐
              │  C. Abstract Domain Engine                         │
              │  - Lattice of taint labels (⊥ → taint(bits) → ⊤) │
              │  - Widening/narrowing operators                    │
              │  - Reduced product of domains                      │
              │  - Quantitative channel-capacity domain            │
              └─────┬──────────────────────────────┬───────────────┘
                    │                              │
        ┌───────────┘                              └──────────┐
        ▼                                                     ▼
┌──────────────────────────┐              ┌──────────────────────────────┐
│ D. Transfer Functions:   │              │ E. Transfer Functions:       │
│    Pandas Operations     │              │    Sklearn Operations        │
│ - merge, join, concat    │              │ - fit, transform,            │
│ - groupby, agg, apply    │              │   fit_transform              │
│ - fillna, dropna, clip   │              │ - StandardScaler, PCA,       │
│ - pivot, melt, stack     │              │   OneHotEncoder, etc.        │
│ - rolling, expanding     │              │ - Pipeline, ColumnTransformer│
│ - indexing, slicing      │              │ - Cross-validation wrappers  │
└───────────┬──────────────┘              └────────────┬─────────────────┘
            │ Abstract transformers                    │
            └──────────┬───────────────────────────────┘
                       ▼
              ┌───────────────────────────────────────┐
              │  F. Information-Flow Propagation       │
              │     Engine (Fixpoint Computation)      │
              │  - Worklist algorithm over DAG         │
              │  - Context-sensitive analysis          │
              │  - Iteration with widening/narrowing   │
              │  - Incremental re-analysis support     │
              └──────────────┬────────────────────────┘
                             │ Fixed-point taint state
                             ▼
              ┌───────────────────────────────────────┐
              │  G. Quantitative Analysis Module       │
              │  - Bits-of-leakage computation         │
              │  - Channel capacity estimation         │
              │  - Severity classification             │
              │  - Leakage path reconstruction         │
              └──────────────┬────────────────────────┘
                             │ Leakage report data
                             ▼
              ┌───────────────────────────────────────┐
              │  H. Report Generator                   │
              │  - Human-readable diagnostics          │
              │  - JSON/SARIF structured output        │
              │  - IDE integration (LSP diagnostics)   │
              │  - Visualization (leakage flow graphs) │
              └──────────────┬────────────────────────┘
                             │
                             ▼
              ┌───────────────────────────────────────┐
              │  I/J. Test Suite + Benchmark Suite     │
              │  - Unit/integration/property tests     │
              │  - Synthetic leakage pipelines         │
              │  - Real-world pipeline corpus          │
              │  - Automated accuracy evaluation       │
              └───────────────────────────────────────┘
```

### 1.2 Architectural Principles

1. **Separation of concerns:** The abstract domain engine (C) is completely independent of both the Python frontend (A/B) and the domain-specific transfer functions (D/E). This enables reuse of the fixpoint engine for other languages and testing the lattice properties in isolation.

2. **Rust core, Python shell:** All performance-critical analysis runs in Rust. Python is used only for (a) PyO3 bindings exposing the analysis API, (b) benchmark driver scripts, and (c) the CLI entry point. This maximizes throughput on laptop CPUs.

3. **Extensible transfer function registry:** New pandas/sklearn operations are added by implementing a `TransferFunction` trait, registering it in a dispatch table keyed by `(module, function_name, arity)`. This is where the bulk of domain modeling LoC lives, and each function requires careful abstract semantics.

4. **Quantitative not just qualitative:** Unlike binary taint analysis, we track information-theoretic channel capacity (bits of leakage) through the abstract domain, using a novel reduced product of a taint lattice and an entropy-approximation lattice.

---

## 2. SUBSYSTEM BREAKDOWN

### Subsystem A: Python AST/Bytecode Analyzer

**Purpose:** Parse Python source files into a typed intermediate representation (IR) that the Rust analysis engine can consume. Must handle the full complexity of real-world ML code: dynamic imports, star imports, monkey-patching, decorators, context managers, comprehensions, and f-strings referencing DataFrame columns.

**Key Technical Challenges:**
- Python's extreme dynamism: `getattr`, `**kwargs`, `eval()`, dynamic module loading
- Type inference without runtime information: must resolve `df.merge(...)` to `pandas.DataFrame.merge` through alias chains (`import pandas as pd; df = pd.read_csv(...)`)
- Handling Jupyter notebook cell ordering (non-linear execution)
- Closure capture analysis for lambdas passed to `.apply()`, `.pipe()`, `.agg()`
- Python version compatibility (3.8–3.12 AST differences)
- Bytecode analysis as fallback for cases where source is unavailable (installed packages)

**Estimated LoC: 18,000 Rust + 2,000 Python**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Python AST parser (Rust, via `rustpython-parser` fork) | 4,000 | Full Python 3.8–3.12 grammar handling, including f-strings, walrus operator, match statements |
| Type inference engine | 5,000 | Flow-sensitive type inference for DataFrame column types, sklearn estimator types; must handle conditional assignments, loops, try/except |
| Import resolver | 2,500 | Virtual filesystem for resolving `from sklearn.preprocessing import StandardScaler`, handling `__init__.py`, relative imports, `importlib` |
| IR definition and construction | 3,000 | Typed IR nodes for ~80 expression/statement types, builder pattern, validation |
| Closure/lambda analyzer | 1,500 | Capture analysis, free-variable resolution for lambdas in `.apply()`, `.groupby().agg()` |
| Notebook (.ipynb) support | 1,000 | JSON parsing, cell ordering heuristics, magic command stripping |
| Bytecode analyzer (fallback) | 1,000 | CPython bytecode disassembly for installed-package introspection |

*Why 20K is necessary:* Python is one of the most complex languages to statically analyze. Real ML code uses extensive dynamic features. A naive parser handles <30% of real pipelines. Each percentage point of additional coverage requires handling another corner case (e.g., `pd.DataFrame({col: func(x) for col, func in mapping.items()})` requires dict comprehension + dynamic dispatch resolution).

**Dependencies:** None (entry point of the pipeline).

---

### Subsystem B: Pipeline DAG Extractor

**Purpose:** Transform the typed IR into a directed acyclic graph of ML pipeline operations, recognizing sklearn's compositional patterns (`Pipeline`, `ColumnTransformer`, `FeatureUnion`, `make_pipeline`, `make_column_transformer`) and pandas method chains.

**Key Technical Challenges:**
- Recognizing sklearn's `Pipeline` constructor patterns (list of `(name, estimator)` tuples, `make_pipeline` variadic args)
- Handling `ColumnTransformer` with heterogeneous column selections (string names, integer indices, boolean masks, callable selectors, regex patterns)
- Resolving custom transformers (user-defined classes inheriting `BaseEstimator`, `TransformerMixin`)
- Tracking the fit/predict split: which operations see training data vs. test data
- Handling `cross_val_score`, `GridSearchCV` which create implicit train/test splits
- Method chain disambiguation: `df.groupby('a').transform('mean')` vs `df.groupby('a').agg('mean')` have different leakage semantics

**Estimated LoC: 14,000 Rust + 1,000 Python**

| Component | LoC | Justification |
|-----------|-----|---------------|
| sklearn Pipeline pattern recognizer | 3,000 | Handle Pipeline, make_pipeline, FeatureUnion, make_column_transformer, plus nested compositions |
| ColumnTransformer column-selection resolver | 2,000 | 6 types of column selectors, each with edge cases; must handle dynamic selection via `make_column_selector` |
| Pandas method-chain DAG builder | 3,000 | Convert chained `.merge().groupby().transform()` into DAG nodes with proper data-flow edges |
| Custom transformer resolver | 2,000 | Inspect user classes for `fit`/`transform` methods, map to abstract transformer signatures |
| Train/test split detector | 2,000 | Recognize `train_test_split`, `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `cross_val_score`, `GridSearchCV` |
| DAG data structures and algorithms | 2,000 | DAG node types, edge types (data-flow, control-flow, fit-dependency), topological sort, cycle detection, subgraph extraction |

*Why 15K is necessary:* sklearn's Pipeline API has >20 composition patterns, each with 3-5 constructor signatures. ColumnTransformer alone has 6 column-selection modes. Pandas has >200 DataFrame methods that form chains. Each pattern requires a dedicated recognizer with its own edge cases.

**Dependencies:** Subsystem A (consumes typed IR).

---

### Subsystem C: Abstract Domain Engine

**Purpose:** Provide the mathematical infrastructure for abstract interpretation: lattices, Galois connections, widening/narrowing operators, and the novel quantitative information-flow domain that tracks bits of leakage.

**Key Technical Challenges:**
- Designing a lattice that can represent *quantitative* information flow (not just binary taint). This requires a novel reduced product of:
  - A powerset taint lattice (which columns are tainted by which test-set sources)
  - An entropy-approximation lattice (how many bits of information flow through each edge)
  - A cardinality domain (tracking dataset sizes for normalizing leakage)
- Widening operators that converge quickly but don't over-approximate excessively
- Efficient lattice operations on large sets of column labels (real DataFrames have 100s of columns)
- Reduced product correctness: ensuring the combination of domains is sound
- Parametric abstraction: user-configurable precision/speed tradeoffs

**Estimated LoC: 22,000 Rust**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Core lattice traits and generic infrastructure | 2,000 | `Lattice`, `JoinSemiLattice`, `MeetSemiLattice`, `BoundedLattice`, `Widening`, `Narrowing` traits with default impls, property-test harnesses |
| Powerset taint domain | 3,000 | Bit-parallel set operations over column labels, efficient representation using roaring bitmaps, hash-consing for label sets |
| Entropy-approximation domain | 5,000 | Novel domain: tracks upper bounds on mutual information I(X;Y) through abstract operations. Requires careful over-approximation of entropy for each abstract transformer (e.g., mean reduces entropy by log(n) bits). 30+ entropy-bounding lemmas implemented as Rust functions. |
| Cardinality domain | 2,000 | Interval abstraction of dataset row counts, handling splits, joins, filters, groupby aggregations |
| Reduced product construction | 3,000 | Generic `ReducedProduct<A, B>` combinator, reduction operators, correctness invariants, specialized reductions for taint×entropy |
| Relational domain (optional, for column correlations) | 3,000 | Octagon-like domain tracking linear relationships between column taint levels, required for precise analysis of PCA/SVD |
| Domain serialization and visualization | 1,500 | Serde-based serialization for checkpointing, DOT-format lattice visualization |
| Property-based test infrastructure for lattice laws | 2,500 | Proptest strategies for generating arbitrary lattice elements, checking associativity, commutativity, absorption, idempotency, widening termination |

*Why 22K is necessary:* This is the core intellectual contribution. A binary taint domain would be ~3K LoC but cannot answer "how many bits of leakage." The entropy-approximation domain is novel research requiring ~30 entropy-bounding lemmas, each derived from information theory and each requiring its own implementation and proof of soundness. The reduced product adds another layer of complexity. Property-based testing of lattice laws is essential for correctness.

**Dependencies:** None (pure mathematical infrastructure).

---

### Subsystem D: Transfer Functions for Pandas Operations

**Purpose:** Define the abstract semantics of every pandas operation that might transmit information between DataFrames. Each transfer function maps an input abstract state to an output abstract state, over-approximating the concrete information flow.

**Key Technical Challenges:**
- Pandas has >300 DataFrame/Series methods; ~120 are information-flow-relevant
- Many operations have complex semantics depending on arguments (e.g., `merge` with `how='inner'` vs `how='outer'` vs `how='cross'` have different information-flow profiles)
- Aggregation operations (groupby + agg) require reasoning about group structure
- Window operations (rolling, expanding, ewm) create temporal information flow
- Index operations (`set_index`, `reset_index`, `reindex`) can leak information through index alignment
- String operations (`str.contains`, `str.extract`) can leak information through pattern matching
- `apply` with arbitrary lambdas requires interprocedural analysis

**Estimated LoC: 25,000 Rust**

| Component | LoC | Justification |
|-----------|-----|---------------|
| DataFrame creation/IO | 2,000 | `read_csv`, `read_parquet`, `read_sql`, `DataFrame()`, `from_dict`, `from_records` — initial taint assignment |
| Merge/join/concat | 3,500 | `merge`, `join`, `concat`, `append`, `combine_first` — key-based information mixing with 4 join types × 3 key modes |
| Groupby + aggregation | 4,000 | `groupby`, `agg`, `transform`, `filter`, `apply`, `resample` — must model group structure, partial aggregation, and the critical `transform` (which is the #1 leakage vector) |
| Reshaping | 2,500 | `pivot`, `pivot_table`, `melt`, `stack`, `unstack`, `crosstab`, `get_dummies` — structural transformations that rearrange taint |
| Selection/filtering | 2,500 | `loc`, `iloc`, `query`, `where`, `mask`, boolean indexing — information flow through predicates |
| Fill/imputation | 2,000 | `fillna`, `interpolate`, `bfill`, `ffill`, `dropna`, `replace` — imputation methods that may use statistics computed from full dataset (leakage!) |
| Arithmetic/statistical | 2,000 | `mean`, `std`, `corr`, `cov`, `describe`, `value_counts`, `nunique` — aggregation that collapses dimensions |
| Window operations | 2,500 | `rolling`, `expanding`, `ewm` with `mean`, `std`, `apply` — temporal information flow with configurable window sizes |
| String/categorical operations | 2,000 | `str.*` accessor methods, `Categorical`, `get_dummies` — text-based information flow |
| Index operations | 2,000 | `set_index`, `reset_index`, `reindex`, `align`, `sort_index`, `sort_values` — index-mediated leakage |

*Why 25K is necessary:* Each pandas operation requires: (1) argument parsing and dispatch, (2) abstract semantics for the taint domain, (3) abstract semantics for the entropy domain, (4) abstract semantics for the cardinality domain, (5) reduction with the combined domain, and (6) unit tests. With ~120 operations × ~200 LoC average (including tests) = ~24K. This is not padding: each operation genuinely has different information-flow semantics. For example, `fillna(df.mean())` computes a statistic from the *full* DataFrame and propagates it to every missing cell — this is one of the most common leakage patterns and requires precise modeling.

**Dependencies:** Subsystem C (abstract domain engine).

---

### Subsystem E: Transfer Functions for Sklearn Operations

**Purpose:** Define abstract semantics for scikit-learn's estimator API, modeling how `fit()` aggregates training data into model parameters and how `transform()`/`predict()` propagates those parameters into outputs.

**Key Technical Challenges:**
- The `fit` → `transform` pattern: `fit()` computes statistics (mean, variance, principal components, etc.) that are stored in the estimator and applied during `transform()`. If `fit()` sees test data, those statistics are tainted.
- ~80 sklearn estimators, each with different `fit` semantics
- `fit_transform` as a special case (may differ from `fit` + `transform`)
- Pipeline composition: a `Pipeline` calls `fit_transform` on all steps except the last, then `fit` on the last
- Cross-validation: `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV` create multiple train/test splits
- Feature selection: `SelectKBest`, `RFE`, `RFECV` use target variable statistics (potential leakage)
- Preprocessing that computes global statistics: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`

**Estimated LoC: 20,000 Rust**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Estimator trait hierarchy | 2,000 | `Estimator`, `Transformer`, `Predictor`, `Classifier`, `Regressor`, `ClusterMixin` — abstract base types with fit/transform/predict signatures |
| Preprocessing transformers | 4,000 | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `KBinsDiscretizer`, `QuantileTransformer`, `PowerTransformer`, `FunctionTransformer`, `PolynomialFeatures`, `SplineTransformer` — each with abstract fit/transform |
| Encoding transformers | 3,000 | `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `LabelBinarizer`, `TargetEncoder` — categorical encoding with target leakage analysis |
| Imputation transformers | 2,000 | `SimpleImputer`, `IterativeImputer`, `KNNImputer`, `MissingIndicator` — imputation with potential for leakage through computed fill values |
| Feature selection | 2,500 | `SelectKBest`, `SelectPercentile`, `GenericUnivariateSelect`, `RFE`, `RFECV`, `SelectFromModel`, `VarianceThreshold` — target-dependent selection is a major leakage vector |
| Decomposition/manifold | 2,500 | `PCA`, `TruncatedSVD`, `NMF`, `FastICA`, `KernelPCA`, `TSNE`, `UMAP` (via umap-learn) — dimensionality reduction that mixes column information |
| Pipeline/composition | 2,500 | `Pipeline`, `FeatureUnion`, `ColumnTransformer`, `make_pipeline`, `make_union`, `make_column_transformer`, `TransformedTargetRegressor` — compositional semantics |
| Model selection/CV | 1,500 | `cross_val_score`, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`, `learning_curve` — implicit train/test split handling |

*Why 20K is necessary:* ~80 sklearn estimators × ~250 LoC average per estimator (abstract fit + abstract transform + abstract predict + entropy computation + tests). Each estimator has genuinely different mathematical semantics — `PCA.fit()` computes eigenvectors (mixing all column information), while `StandardScaler.fit()` computes per-column means (column-independent). These differences matter for quantitative leakage analysis.

**Dependencies:** Subsystem C (abstract domain engine).

---

### Subsystem F: Information-Flow Propagation Engine

**Purpose:** Drive the abstract interpretation to a fixpoint over the pipeline DAG. Implements the core worklist algorithm with context-sensitivity, widening/narrowing, and incremental re-analysis.

**Key Technical Challenges:**
- **Context sensitivity:** The same sklearn transformer may be used in multiple pipeline stages or with different column subsets. Each usage context requires separate abstract state.
- **Interprocedural analysis:** User-defined functions passed to `.apply()`, `.pipe()`, or custom transformers require function summaries.
- **Widening strategy:** Must balance convergence speed (for laptop CPU) with precision. Too aggressive widening → false positives. Too conservative → non-termination on loops.
- **Incremental re-analysis:** When a user fixes one leakage issue, only the affected subgraph should be re-analyzed. Requires dependency tracking and invalidation.
- **Memory management:** Abstract states for DataFrames with 1000+ columns can be large. Must support efficient state representation and garbage collection of unreachable states.
- **Parallel analysis:** Independent pipeline branches can be analyzed in parallel using Rayon.

**Estimated LoC: 16,000 Rust**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Worklist algorithm | 3,000 | Priority-based worklist with RPO ordering, iteration counting, convergence detection, timeout handling |
| Context-sensitivity engine | 3,000 | Call-site sensitivity (k-CFA style), context creation/merging, context tree management |
| Function summary computation | 2,500 | Interprocedural analysis via summaries: compute abstract effect of user functions, cache and reuse |
| Widening/narrowing strategy | 2,000 | Delayed widening, widening with thresholds, narrowing iterations, strategy selection heuristics |
| Incremental analysis | 2,500 | Change detection, dependency graph, invalidation propagation, partial re-analysis |
| Parallelism (Rayon-based) | 1,500 | DAG partitioning, parallel branch analysis, state merging at join points |
| Memory management | 1,500 | Abstract state pooling, reference-counted sharing, GC for unreachable states |

*Why 16K is necessary:* Fixpoint computation is deceptively complex. A textbook worklist algorithm is ~200 lines, but production-quality fixpoint computation with context sensitivity, widening strategies, incremental analysis, and parallelism requires 100x that. Each of these features is necessary: context sensitivity for precision, widening for termination, incremental analysis for usability, parallelism for laptop-CPU performance.

**Dependencies:** Subsystems B (pipeline DAG), C (abstract domain), D (pandas transfer functions), E (sklearn transfer functions).

---

### Subsystem G: Quantitative Analysis Module

**Purpose:** Post-process the fixpoint results to compute human-interpretable leakage metrics: bits of leakage per feature, leakage paths, severity scores, and remediation suggestions.

**Key Technical Challenges:**
- **Bits-of-leakage computation:** Convert abstract entropy bounds into concrete "X bits of test-set information leaked into feature Y." Requires careful calibration.
- **Leakage path reconstruction:** Trace the information flow from test-set source to training feature through the DAG, identifying the critical operation(s) that caused leakage.
- **False positive mitigation:** Not all information flow is harmful leakage. Leakage of the *marginal distribution* of features (e.g., global mean for imputation) is usually benign. Must classify leakage types.
- **Severity scoring:** Combine bits-of-leakage, leakage-type classification, and downstream model impact estimates into a single severity score.
- **Counterfactual analysis:** Estimate how much model performance would drop if the leakage were removed (without actually retraining).

**Estimated LoC: 10,000 Rust**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Bits-of-leakage calculator | 2,500 | Convert abstract states to concrete bit counts, handle different leakage types (direct, statistical, structural), confidence intervals |
| Leakage path tracer | 2,500 | DAG traversal to reconstruct information-flow paths, minimal path computation, critical-edge identification |
| Leakage classifier | 2,000 | Classify leakage as target leakage, feature leakage, preprocessing leakage, cross-validation leakage; severity levels |
| Remediation suggester | 1,500 | Pattern-matched suggestions: "Move StandardScaler inside Pipeline", "Use cross_val_score instead of manual split", etc. |
| Counterfactual estimator | 1,500 | Estimate model performance impact using information-theoretic bounds without retraining |

*Why 10K is necessary:* The quantitative analysis is the key differentiator from existing tools. Binary taint analysis can say "there is leakage." We say "there are 3.7 bits of leakage in feature X, caused by applying StandardScaler before train_test_split, which inflates accuracy by an estimated 2.1 percentage points." Each of those numbers requires careful computation and calibration.

**Dependencies:** Subsystem F (fixpoint results), Subsystem C (abstract domain queries).

---

### Subsystem H: Report Generator

**Purpose:** Transform raw analysis results into multiple output formats: human-readable terminal reports, JSON for programmatic consumption, SARIF for IDE integration, HTML for web-based viewing, and DOT/SVG for flow graph visualization.

**Key Technical Challenges:**
- Multiple output formats with shared rendering logic but format-specific presentation
- SARIF (Static Analysis Results Interchange Format) compliance for VS Code / GitHub integration
- Flow graph visualization that is readable even for large pipelines (layout algorithms)
- Incremental reporting: show intermediate results during long analyses
- Diff-based reporting: compare leakage reports across code versions

**Estimated LoC: 8,000 Rust + 2,000 Python**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Core report data model | 1,500 | Shared data structures for leakage findings, severity, paths, suggestions |
| Terminal/text renderer | 1,500 | Colored terminal output with ASCII flow diagrams, severity indicators, code snippets |
| JSON/SARIF renderer | 2,000 | SARIF v2.1 compliance (complex schema), JSON output for CI/CD integration |
| HTML renderer | 1,500 | Interactive HTML report with collapsible sections, syntax-highlighted code, flow graphs |
| DOT/SVG graph renderer | 1,500 | Pipeline flow graph with taint coloring, leakage annotations, automatic layout |
| Diff reporter | 1,000 | Compare two reports, show new/fixed/changed leakage findings |
| Python CLI wrapper | 1,000 | Click-based CLI, configuration file support, output format selection |
| Python IDE integration | 1,000 | LSP-compatible diagnostic output, VS Code extension helpers |

*Why 10K is necessary:* Five output formats × ~1.5K each for format-specific rendering + shared infrastructure. SARIF alone has a 50-page specification. Graph layout for readable flow visualizations is non-trivial.

**Dependencies:** Subsystem G (analysis results).

---

### Subsystem I: Test Suite and Evaluation Harness

**Purpose:** Comprehensive testing infrastructure including unit tests, integration tests, property-based tests, mutation tests, and an automated evaluation harness that measures precision/recall of leakage detection against ground-truth benchmarks.

**Key Technical Challenges:**
- Property-based testing of lattice laws (associativity, commutativity, etc.)
- Mutation testing for transfer functions: mutate a pipeline, verify the analysis detects the introduced leakage
- Ground-truth generation: automatically constructing pipelines with known leakage quantities
- Evaluation metrics: precision, recall, F1 for leakage detection; mean absolute error for bits-of-leakage quantification
- Regression testing: ensure analysis results don't change unexpectedly across engine changes

**Estimated LoC: 18,000 Rust + 5,000 Python**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Unit tests for lattice operations (proptest) | 3,000 | Property-based tests for every lattice law, every domain, every combinator |
| Unit tests for transfer functions | 5,000 | ~200 transfer functions × ~25 LoC per test = 5,000; each needs input abstract state, expected output state, edge cases |
| Integration tests (end-to-end pipelines) | 3,000 | ~50 synthetic pipelines exercising different leakage patterns, verified end-to-end |
| Evaluation harness | 3,000 | Automated precision/recall computation, statistical significance testing, result aggregation |
| Mutation testing framework | 2,000 | Pipeline mutation operators (insert leakage, remove leakage, change operation), mutation-score computation |
| Regression test infrastructure | 2,000 | Golden-file comparison, snapshot testing, bisection support |
| Python test wrappers | 3,000 | pytest infrastructure, fixture generation, integration with Python test runner |
| Fuzz testing infrastructure | 2,000 | Rust libfuzzer integration for parser, domain operations, fixpoint engine |

*Why 23K is necessary:* For a best-paper artifact, correctness is paramount. A static analysis tool that produces wrong results is worse than useless. Every transfer function needs tests, every lattice operation needs property tests, and the overall system needs integration tests. The evaluation harness is critical for the paper's empirical claims.

**Dependencies:** All other subsystems (tests exercise everything).

---

### Subsystem J: Benchmark Suite

**Purpose:** A curated corpus of ML pipelines — both synthetic (with known ground-truth leakage) and real-world (from Kaggle, GitHub, research papers) — used for evaluation in the paper.

**Key Technical Challenges:**
- Synthetic pipeline generation with precisely controlled leakage amounts
- Real-world pipeline collection: scraping, cleaning, making reproducible
- Ground-truth labeling: automated leakage quantification via empirical measurement (run pipeline with/without leakage, measure accuracy difference)
- Reproducibility: pinned dependencies, deterministic random seeds
- Scale: need 500+ pipelines for statistical significance in evaluation

**Estimated LoC: 5,000 Rust + 12,000 Python**

| Component | LoC | Justification |
|-----------|-----|---------------|
| Synthetic pipeline generator | 3,000 Py | Parameterized generator producing pipelines with controlled leakage: N operations, M columns, K leakage points, B bits per leak |
| Leakage pattern library | 2,000 Py | ~30 canonical leakage patterns (fit-before-split, target encoding, time-series lookahead, etc.) with parametric instantiation |
| Real-world pipeline corpus | 3,000 Py | Scripts to download, clean, and standardize 200+ real pipelines from Kaggle kernels and GitHub repos |
| Empirical ground-truth oracle | 2,000 Py | Run each pipeline with/without leakage, measure accuracy delta, convert to bits via information-theoretic calibration |
| Benchmark runner | 2,000 Py | Orchestrate analysis of all benchmarks, collect timing/memory/accuracy metrics |
| Benchmark data structures (Rust) | 2,500 Rs | Benchmark metadata, result storage, comparison operators |
| Benchmark analysis/reporting | 2,500 Rs | Aggregate metrics, generate paper-ready tables and plots, statistical tests |

*Why 17K is necessary:* A best-paper artifact needs a convincing evaluation. 500+ benchmarks × pipeline setup + ground truth + analysis + reporting = substantial infrastructure. The synthetic generator alone must produce diverse pipelines covering ~30 leakage patterns × multiple parameterizations. Real-world corpus curation requires significant data engineering.

**Dependencies:** Subsystem H (report generator for benchmark comparison).

---

## 3. LOC BUDGET

### Summary Table

| Category | Rust LoC | Python LoC | Total LoC | % of Total |
|----------|----------|------------|-----------|------------|
| **A. Python AST/Bytecode Analyzer** | 18,000 | 2,000 | 20,000 | 12.7% |
| **B. Pipeline DAG Extractor** | 14,000 | 1,000 | 15,000 | 9.5% |
| **C. Abstract Domain Engine** | 22,000 | 0 | 22,000 | 14.0% |
| **D. Pandas Transfer Functions** | 25,000 | 0 | 25,000 | 15.9% |
| **E. Sklearn Transfer Functions** | 20,000 | 0 | 20,000 | 12.7% |
| **F. Propagation Engine** | 16,000 | 0 | 16,000 | 10.2% |
| **G. Quantitative Analysis** | 10,000 | 0 | 10,000 | 6.4% |
| **H. Report Generator** | 8,000 | 2,000 | 10,000 | 6.4% |
| **I. Test Suite + Eval Harness** | 18,000 | 5,000 | 23,000 | 14.6% |
| **J. Benchmark Suite** | 5,000 | 12,000 | 17,000 | 10.8% |
| **Infrastructure (build, CI, PyO3 bindings)** | 2,000 | 1,000 | 3,000 | 1.9% |
| | | | | |
| **TOTAL** | **158,000** | **23,000** | **181,000** | **100%** |

### Justification That 150K+ Is Necessary (Not Padding)

**Core analysis engine (C + F + G): 48K LoC**
This is the intellectual heart: a novel quantitative information-flow analysis. The abstract domain engine alone requires ~30 entropy-bounding lemmas, each a non-trivial implementation. The fixpoint engine with context-sensitivity, widening, and incremental analysis is production-grade static analysis infrastructure. For comparison, the IKOS abstract interpretation framework (NASA) is ~80K LoC for a simpler (non-quantitative) domain.

**Domain modeling (D + E): 45K LoC**
This is the largest single category and the most critical for practical utility. Covering ~200 pandas/sklearn operations, each with unique abstract semantics, is inherently expensive. Each transfer function requires ~200 LoC (argument handling + taint transfer + entropy computation + tests). This is analogous to how a compiler's instruction selection phase scales linearly with the number of supported instructions. Cutting this means supporting fewer operations, which means missing leakage in real pipelines.

**Frontend (A + B): 35K LoC**
Parsing Python statically is hard. For comparison, `rust-analyzer` (Rust's LSP) is ~200K LoC, and even focused Python analyzers like `pyright` are ~100K LoC. Our 35K covers only the subset needed for ML pipeline analysis, which is already aggressive scoping.

**Test infrastructure (I): 23K LoC**
Test code is typically 1-1.5x the size of the code under test for well-tested systems. Our ratio of 23K tests to 158K implementation (0.15x) is actually *low*. The evaluation harness is specifically required for the paper's empirical claims.

**Benchmarks (J): 17K LoC**
The benchmark suite is the empirical backbone of the paper. 500+ pipelines with ground truth require substantial generation, curation, and orchestration infrastructure.

**What could NOT be cut without losing the core contribution:**
- Domain modeling (D+E): Cutting to <20K means supporting <50 operations → misses most real leakage patterns → tool is not useful.
- Abstract domain (C): Cutting the entropy domain reduces to binary taint analysis → loses the "quantitative" differentiator → not novel enough for best paper.
- Test infrastructure (I): Cutting tests below 15K risks correctness bugs that undermine all empirical claims.

---

## 4. FEASIBILITY ANALYSIS

### 4.1 Hardest Engineering Challenges

**Challenge 1: Soundness of quantitative entropy bounds (Risk: HIGH)**
The entropy-approximation domain is novel research. For each abstract transformer, we must prove (or at least argue convincingly) that our entropy bound is sound — i.e., we never *underestimate* information leakage. This requires deriving information-theoretic inequalities for ~120 pandas operations. Some operations (e.g., `groupby().transform()` with a user lambda) may have no tight bound.

*Mitigation:* Use conservative (over-approximate) bounds where exact bounds are unknown. Validate empirically against the ground-truth oracle. Acknowledge imprecision in the paper.

**Challenge 2: Python static analysis precision (Risk: HIGH)**
Real Python ML code uses dynamic features extensively: `getattr`, `**kwargs`, runtime type dispatch, monkey-patching. Our static analysis will inevitably encounter code it cannot precisely analyze.

*Mitigation:* Design a graceful degradation strategy. When static analysis fails, fall back to (a) dynamic tracing via instrumented execution, or (b) conservative over-approximation (assume worst-case leakage). Report analysis coverage metrics alongside leakage findings.

**Challenge 3: Scaling to real-world pipeline size (Risk: MEDIUM)**
Real Kaggle kernels have 500+ lines of pandas code with 50+ DataFrame operations. The abstract state space can explode combinatorially.

*Mitigation:* Widening operators, abstract garbage collection, and parallel analysis on independent branches. The laptop-CPU constraint makes this critical. Profile memory/time on largest benchmarks during development.

**Challenge 4: Cross-validation leakage modeling (Risk: MEDIUM)**
`GridSearchCV` and `cross_val_score` create implicit loops over train/test splits. Modeling the information flow through these higher-order sklearn patterns is complex.

*Mitigation:* Treat CV as a well-known pattern with a hand-crafted abstract semantics, rather than trying to analyze the sklearn source code. This is sound because the CV API contract is well-documented.

### 4.2 What Might Blow Up Scope

1. **Python type inference rabbit hole.** Every new real-world pipeline will exercise another Python corner case. Must set a firm "85% coverage" target and accept graceful degradation for the rest.

2. **Entropy bound derivations.** Some operations may require genuine mathematical research to derive sound bounds. Budget 2-3 months for this. Have a plan B: use empirically calibrated bounds instead of proven bounds for the hardest cases.

3. **Sklearn version compatibility.** sklearn's API has changed across versions (0.24 → 1.0 → 1.3 → 1.4). Supporting multiple versions multiplies transfer function code. *Decision:* Target sklearn 1.3+ only.

4. **Benchmark corpus quality.** Real-world pipelines may not have easily extractable ground truth. The synthetic generator must be the primary evaluation vehicle, with real-world pipelines as supplementary evidence.

### 4.3 What Can Be Descoped

**Descope Level 1 (lose depth, keep breadth):**
- Drop the relational domain from Subsystem C (–3K LoC). PCA/SVD analysis becomes less precise but the system still works.
- Drop incremental analysis from Subsystem F (–2.5K LoC). Users must re-analyze from scratch on each change.
- Drop HTML report format from Subsystem H (–1.5K LoC). Keep terminal + JSON + SARIF.
- **Impact:** –7K LoC, minimal impact on paper contributions.

**Descope Level 2 (lose some breadth):**
- Reduce pandas operations from 120 to 60 most common (–12K LoC from Subsystem D).
- Reduce sklearn estimators from 80 to 40 most common (–10K LoC from Subsystem E).
- **Impact:** –22K LoC, tool misses leakage in ~30% of real pipelines. Acknowledge as limitation.

**Descope Level 3 (lose quantitative, keep qualitative):**
- Replace entropy domain with binary taint (–5K from C, –30% from D/E ≈ –13K).
- Drop counterfactual estimator from G (–1.5K).
- **Impact:** –19.5K LoC, but the paper loses its core differentiator ("quantitative"). Only do this if entropy bounds prove mathematically intractable.

**Descope Level 4 (minimum viable paper):**
- All of above + reduce benchmarks to 100 pipelines (–8K from J).
- **Impact:** –56.5K LoC, total ~125K. Still a solid tool paper but unlikely best-paper.

### 4.4 Laptop-CPU Constraint Impact on Design

The laptop-CPU constraint (no GPU, likely 8-16 cores, 16-64 GB RAM) fundamentally shapes several design decisions:

1. **Rust as implementation language:** A Python-only implementation would be 10-50x slower for fixpoint computation. Rust gives us C-level performance with memory safety.

2. **Efficient data structures:** Roaring bitmaps for taint sets (subsystem C), hash-consing for abstract states, arena allocation for DAG nodes.

3. **Parallelism via Rayon:** Exploit 8-16 cores for independent pipeline branch analysis. Expected 4-6x speedup over single-threaded.

4. **Widening aggressiveness:** Must converge in <100 iterations per DAG node to keep analysis under 5 minutes for typical pipelines. This favors aggressive widening (at the cost of some precision).

5. **Benchmark analysis time budget:** Full benchmark suite (500 pipelines) must complete in <4 hours. At ~30 seconds per pipeline, this is feasible.

6. **No neural components:** All analysis is symbolic/algebraic. No ML-based heuristics (which would need GPU for training). This is actually a strength: the tool's results are deterministic and explainable.

---

## 5. COMPETING ARCHITECTURE PROPOSALS

### Architecture A: Monolithic Rust Analyzer (Chosen)

**Description:** As described above — a fully static analysis approach implemented primarily in Rust, with Python bindings for the CLI/benchmark layer. Analysis is purely static (no execution of the analyzed pipeline).

**Pros:**
- Deterministic, reproducible results
- No need to execute (potentially expensive/dangerous) user code
- Scales to large pipelines without running out of time/memory on data
- Elegant information-theoretic framework with soundness guarantees

**Cons:**
- Static analysis of Python is imprecise
- Cannot handle truly dynamic patterns (e.g., column names computed at runtime from data)
- Significant engineering effort for the Python frontend

### Architecture B: Hybrid Static-Dynamic Analysis

**Description:** Use dynamic tracing (via Python's `sys.settrace` or AST instrumentation) to observe actual dataflow during a pipeline execution, then use static analysis only to *quantify* the information flow along observed paths.

**Data Flow:**
```
User pipeline → Instrumented execution → Execution trace → Static quantification → Report
```

**Pros:**
- Avoids the Python static analysis problem entirely — dynamic tracing sees actual types, column names, shapes
- Much simpler frontend (5K LoC instrumentation vs 35K static analysis)
- 100% precision on observed paths (no false positives from dynamic analysis)

**Cons:**
- Requires actually executing the pipeline → needs the data, needs compute time
- Only sees one execution path → misses leakage in unexecuted branches
- Non-deterministic (depends on input data)
- "Laptop CPU" constraint is harder: must execute the pipeline itself, not just analyze code
- Cannot analyze pipelines for which you don't have data access

**Estimated LoC:** ~100K (simpler frontend, same domain modeling, different propagation engine)

### Architecture C: Modular Plugin Architecture with IR

**Description:** Define a language-agnostic intermediate representation (IR) for ML pipelines, then build separate frontends for different source languages (Python, R, Julia) and a single analysis backend over the IR. The IR would be a first-class serializable format, enabling third-party tool integration.

**Data Flow:**
```
Python frontend ─┐
R frontend ──────┼──→ Universal ML-Pipeline IR ──→ Analysis Engine ──→ Reports
Julia frontend ──┘
```

**Pros:**
- Maximum reusability: one analysis engine serves multiple languages
- Clean separation of concerns: frontend teams and backend teams work independently
- Third-party extensibility: other tools can produce/consume the IR
- Future-proofed for non-Python ML ecosystems

**Cons:**
- IR design is itself a major research contribution (and engineering effort: +10-15K LoC)
- Abstraction penalty: the IR must be expressive enough to preserve information-flow-relevant semantics from all source languages
- Scope explosion: supporting even two languages doubles frontend effort
- Premature generalization: the paper contribution is about Python/sklearn leakage detection

**Estimated LoC:** ~200K (IR definition + multiple frontends + analysis engine)

### Decision: Architecture A

**Architecture A is the recommended choice.** Rationale:

1. **Scope discipline.** Architecture C is the "right" long-term engineering choice but is a 200K LoC project — too risky for a single paper. Architecture A at 181K LoC is already ambitious.

2. **Static > dynamic for a paper contribution.** Architecture B's dynamic approach is simpler to build but harder to publish. The static approach with quantitative information flow is a genuine research contribution (novel abstract domain). The dynamic approach is "just" engineering.

3. **No data requirement.** Architecture A analyzes *code*, not *data*. Users don't need to share their datasets. This is critical for adoption and for analyzing pipelines found on GitHub/Kaggle.

4. **Determinism.** Static analysis produces the same results every time. This is essential for reproducible research and CI/CD integration.

5. **Soundness guarantees.** A static analysis can be *sound* (never misses true leakage). A dynamic analysis cannot. Soundness is a strong selling point for a paper.

**Hybrid fallback:** If Python static analysis proves too imprecise in practice (>40% of pipelines fail to parse), incorporate a lightweight dynamic tracing mode as an optional fallback. Budget 5K LoC for this contingency (already included in the 181K estimate under graceful-degradation paths in Subsystem A).

---

## 6. RISK-ADJUSTED TIMELINE ESTIMATE

| Phase | Duration | Subsystems | Milestone |
|-------|----------|------------|-----------|
| Foundation | Months 1–3 | C (domain), A (parser) | Lattice laws pass property tests; parser handles 50 synthetic pipelines |
| Core Engine | Months 3–5 | F (propagation), B (DAG extraction) | Fixpoint converges on 10 hand-crafted pipelines |
| Domain Modeling | Months 5–8 | D (pandas), E (sklearn) | 80+ transfer functions; handles real Kaggle kernels |
| Quantification | Months 8–9 | G (quantitative) | Bits-of-leakage numbers validated against empirical oracle |
| Tooling | Months 9–10 | H (reports), CLI | End-to-end pipeline: code in → leakage report out |
| Evaluation | Months 10–12 | I (tests), J (benchmarks) | 500+ benchmarks, precision/recall tables for paper |
| Paper Writing | Months 10–14 | (concurrent) | Submission-ready paper with all experiments |

**Critical path:** C → F → D → G → J (domain engine → propagation → pandas ops → quantification → evaluation). Python frontend (A, B) is on a parallel track.

---

## 7. DISTINCTION FROM ml-pipeline-selfheal

The `ml-pipeline-selfheal` project (if it exists in this community) would focus on *automatic repair* of ML pipelines — detecting failures and applying patches. This project (`ml-pipeline-leakage-auditor`) is fundamentally different:

| Dimension | ml-pipeline-selfheal | ml-pipeline-leakage-auditor |
|-----------|---------------------|-----------------------------|
| **Goal** | Fix broken pipelines | Detect silent statistical errors |
| **Problem type** | Crash/exception recovery | Information leakage (no crash) |
| **Technique** | Program repair / synthesis | Abstract interpretation / information theory |
| **Output** | Patched pipeline code | Quantitative leakage report |
| **Novelty** | Repair strategies for ML | Quantitative information-flow domain for ML ops |

There is no overlap in core techniques. The two projects are complementary: one fixes pipelines that crash, the other finds pipelines that silently produce inflated metrics.
