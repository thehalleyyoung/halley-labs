# ML Pipeline Leakage Auditor — Problem Framing Proposals

## Context & Differentiation

**Seed idea:** A quantitative information-flow analysis engine that detects train–test data leakage in scikit-learn/pandas ML pipelines by propagating abstract taint labels through statistical operations, measuring how many bits of test-set signal contaminate each training feature.

**Key constraint vs. portfolio:** `ml-pipeline-selfheal` focuses on *repairing* broken pipelines. This project is fundamentally about *measurement*—building an abstract interpretation framework that quantifies information flow in bits through statistical operations. The artifact is a static/symbolic analysis engine, not a fixer.

**Hard constraints:** Laptop CPU only, no human annotation, fully automated evaluation, 150K+ LoC necessary complexity.

---

## Framing A: "LeakageIR — An Abstract Interpretation Framework for Quantitative Information-Flow Certification of ML Pipelines"

### Angle: Verification / Static Analysis Paradigm

### Problem & Approach

Data leakage—where test-set information contaminates model training—is the single most common source of unreproducible ML results. Published estimates suggest 25–50% of Kaggle competition kernels and a non-trivial fraction of published ML papers contain some form of leakage. Yet the field has no principled tool for *detecting* leakage, let alone *measuring* it. Practitioners rely on manual code review, which is error-prone and does not scale.

We introduce **LeakageIR**, an abstract interpretation framework that models ML pipelines as information-flow programs over a novel *taint lattice*. Each data element carries a label drawn from a lattice of abstract taint domains—encoding not just *whether* test information has leaked, but *how many bits* of test-set signal have propagated into each training feature. The framework defines abstract transfer functions for every pandas/scikit-learn operation (joins, groupbys, fit-transform, feature engineering) that soundly over-approximate information flow. A `fit` call on a concatenated dataframe, for example, is modeled as an entropy-bounded channel from test rows to learned parameters.

The key technical insight is that statistical operations (e.g., computing a mean for imputation, fitting a scaler) act as *lossy channels* whose capacity can be bounded using the operation's algebraic structure. A `StandardScaler.fit()` on *n* rows leaks at most *O(log n)* bits per feature from any single row; a `GroupBy.transform('mean')` leaks at most *H(group_key)* bits. By composing these per-operation bounds through the pipeline DAG, LeakageIR produces a *leakage certificate*: a per-feature upper bound on the mutual information between the training representation and the held-out test set, expressed in bits.

### Who Desperately Needs This

- **ML platform teams** at companies like Airbnb, Spotify, and Netflix who run thousands of feature pipelines and cannot manually audit each one. A single leakage bug in a production feature pipeline can silently inflate model metrics, leading to overconfident deployment decisions and real-world performance degradation. LeakageIR gives them a CI/CD gate that blocks leaky pipelines before deployment.
- **Regulatory bodies and auditors** tasked with evaluating ML model validity in high-stakes domains (credit scoring, clinical trials, recidivism prediction). Current audit practices cannot systematically verify that a model's reported accuracy is not inflated by leakage. A quantitative leakage certificate provides auditable, machine-checkable evidence.
- **ML researchers and reviewers** who need to verify that published results are leakage-free. A LeakageIR report attached to a paper submission would be the equivalent of a reproducibility artifact badge.

### Why This Is Genuinely Hard

1. **Defining sound abstract transfer functions for statistical operations** is an open problem. Unlike standard taint analysis for security (where taint is binary), information-flow through statistical aggregations is inherently *quantitative* and *lossy*. Bounding the channel capacity of `GroupBy.transform(lambda x: x.rolling(7).mean())` requires combining information-theoretic reasoning with abstract interpretation soundness guarantees.
2. **Handling the pandas/scikit-learn API surface** is enormous. Pandas alone has ~300 DataFrame methods, many with complex polymorphic semantics (e.g., `merge` with different join types, index alignment, MultiIndex operations). Scikit-learn has ~150 estimator classes with diverse fit/transform semantics. Modeling each requires careful domain-specific abstract semantics.
3. **Compositionality across the pipeline DAG** is non-trivial. Pipelines involve mutable state (fitted estimators), control flow (cross-validation loops, hyperparameter search), and dynamic shapes (feature selection changes dimensionality). The abstract interpreter must handle all of this while maintaining sound bounds.
4. **The lattice design must balance precision and tractability.** Too coarse (binary taint) loses the quantitative signal. Too fine (exact mutual information) is undecidable. The sweet spot—bounded channel capacities composed over DAGs—requires novel lattice constructions.

### Best-Paper Argument

This paper introduces the *first formal framework for quantifying data leakage in ML pipelines*, bridging abstract interpretation (a mature PL technique) with information-theoretic channel capacity (a mature information theory concept) in a novel domain. The contribution is not incremental: no prior work provides *sound, quantitative* leakage bounds for real-world ML code. The framework is both theoretically principled (soundness theorem, compositionality proof) and practically impactful (evaluated on thousands of real Kaggle pipelines, catching known and unknown leakage bugs). It opens a new subfield—*quantitative information-flow analysis for data science code*—that neither the PL community nor the ML community has explored.

### Fatal Flaws & Weaknesses

- **Soundness vs. precision tradeoff:** Sound over-approximation may produce leakage warnings on perfectly safe pipelines (false positives). If the false positive rate is too high, practitioners will ignore the tool.
- **Dynamic Python semantics:** Python's dynamic typing, monkey-patching, and eval() make sound static analysis fundamentally incomplete. Pipelines using custom transformers or arbitrary lambdas may be unanalyzable.
- **Scope creep risk:** The 150K+ LoC requirement for modeling the full pandas/sklearn API is enormous engineering effort that may not translate into *intellectual* novelty—reviewers may see it as "just engineering."
- **Channel capacity bounds may be too loose:** Bounding information flow through a `GroupBy.transform('mean')` by *H(group_key)* is sound but may be orders of magnitude looser than the true leakage, making the quantitative "bits" measurement meaningless in practice.

---

## Framing B: "TaintFlow — Differential Information Auditing for Machine Learning Reproducibility"

### Angle: Reproducibility / Debugging Diagnostics

### Problem & Approach

The ML reproducibility crisis has a root cause that is poorly understood: the *silent, quantitative* nature of data leakage. Unlike a crash or a type error, leakage does not produce an observable failure—it produces *inflated metrics*. A model that achieves 95% accuracy with leakage and 82% without leakage will appear to work perfectly in development. The leakage is only discovered months later when production metrics diverge from offline evaluations, or never discovered at all. We argue that the field needs a fundamentally new class of diagnostic tool: not a linter that flags suspicious patterns, but a *differential information auditor* that precisely quantifies how much test-set knowledge has contaminated each model decision.

**TaintFlow** performs *differential information auditing*: given an ML pipeline, it constructs two abstract executions—one where train and test data are informationally isolated (the *clean* execution) and one reflecting the actual code (the *observed* execution). By comparing the information content of intermediate representations across these two executions, TaintFlow computes a *leakage spectrum*: a per-feature, per-pipeline-stage decomposition of exactly where and how much test-set information enters the training process. This goes far beyond binary "leaked/not-leaked" classification—it provides a *heat map* of information contamination across the entire pipeline.

The technical core is a *taint propagation algebra* over abstract information states. Each pipeline operation is modeled as a morphism in a category of information-flow transformers, where composition corresponds to sequential pipeline stages and products correspond to feature concatenation. The algebra supports both *forward analysis* (tracking where test information flows) and *backward analysis* (attributing a model's leakage to specific upstream operations). This bidirectional analysis is critical for debugging: forward analysis tells you *what* is contaminated; backward analysis tells you *why*.

### Who Desperately Needs This

- **ML practitioners debugging metric discrepancies.** The #1 question on ML forums is "why do my offline and online metrics disagree?" In a large fraction of cases, the answer is leakage, but practitioners have no tool to diagnose this. TaintFlow's leakage spectrum directly answers "which features, in which pipeline stages, are responsible for your inflated offline metrics?"
- **ML competition platforms** (Kaggle, DrivenData) that currently rely on private leaderboards as a blunt instrument against leakage. TaintFlow could provide automated leakage scoring for submitted kernels, improving competition integrity.
- **Pharma/biotech teams** running clinical ML pipelines where leakage can lead to inflated diagnostic accuracy, failed clinical trials, and wasted resources. A per-feature leakage decomposition lets biostatisticians identify exactly which features carry problematic signal.

### Why This Is Genuinely Hard

1. **The differential execution model** requires precisely characterizing the "clean" counterfactual execution for arbitrary pipeline topologies. This is non-trivial when pipelines involve shared state, caching, or lazy evaluation.
2. **Backward attribution through statistical operations** is fundamentally harder than forward taint propagation. Attributing a model's leakage to a specific upstream `merge` operation requires inverting the information-flow semantics of all intermediate operations—a problem related to program slicing but complicated by lossy statistical transformations.
3. **Scaling to real pipelines** with hundreds of features, nested cross-validation, and dynamic feature selection creates an exponential blowup in the abstract state space. Novel widening operators and abstract domain design are needed.
4. **Validating the quantitative accuracy** of leakage estimates requires ground truth, but ground truth for "bits of leakage" in real pipelines does not exist. We must design novel empirical validation methodologies (synthetic benchmarks with known leakage injected at calibrated levels).

### Best-Paper Argument

This paper reframes data leakage from a *binary classification problem* (leaked or not) to a *quantitative measurement problem* (how many bits, through which channels, at which pipeline stages). The differential auditing methodology is entirely novel—no prior work computes a per-feature, per-stage leakage decomposition. The bidirectional analysis (forward taint + backward attribution) provides actionable diagnostics that existing tools cannot match. The evaluation on thousands of real-world pipelines with synthetic leakage injection at calibrated levels establishes the first benchmark for quantitative leakage detection. This is the kind of "new lens on an old problem" contribution that top venues reward.

### Fatal Flaws & Weaknesses

- **The "differential" framing may be seen as just a presentation wrapper** around standard abstract interpretation. Reviewers may argue the two-execution comparison adds complexity without fundamental novelty.
- **Backward attribution accuracy** may degrade rapidly as pipeline depth increases, producing attribution maps that are too diffuse to be actionable.
- **The reproducibility angle may feel like motivation-washing**—packaging a static analysis tool in reproducibility language to seem more impactful than it is. Reviewers at ML venues may be skeptical of PL-flavored contributions.
- **Competition with simpler baselines:** If a simple "run the pipeline twice with different splits and compare" heuristic catches 90% of leakage, the elaborate information-theoretic machinery may seem unjustified.

---

## Framing C: "BitLeak — Information-Theoretic Type Checking for Data Science Programs"

### Angle: New Type System / Programming Languages for ML

### Problem & Approach

We observe a deep analogy between *data leakage in ML pipelines* and *information-flow violations in security-typed languages*. In the security setting, a type system tracks how secret data flows through a program and rejects programs where secrets leak to public outputs. We propose an analogous framework for data science: a *leakage type system* where data partitions (train, validation, test, production) are tracked through pipeline operations via type-level annotations, and the type checker enforces that no more than a specified number of bits flow across partition boundaries.

**BitLeak** introduces *information-flow types* for data science operations. Every DataFrame, Series, and ndarray carries a *partition type* and a *leakage bound*—a type-level natural number bounding the bits of cross-partition information the value may contain. Pipeline operations are typed with *information-flow signatures*: `StandardScaler.fit : DataFrame[train, 0] → Scaler[train, 0]`, but `StandardScaler.fit : DataFrame[train ∪ test, 0] → Scaler[train, ⌈log₂(n_test)⌉]`. The type checker composes these signatures across the pipeline DAG and rejects pipelines where any feature's leakage bound exceeds a user-specified threshold.

The theoretical contribution is a *quantitative information-flow type theory* instantiated for the domain of statistical computations. Unlike prior quantitative information-flow work (which targets imperative programs with simple datatypes), BitLeak must handle DataFrames with schema-level types, statistical aggregations with data-dependent information flow, and the fit/predict paradigm where learned parameters carry latent information about training data. We prove *soundness* (well-typed pipelines have leakage bounded by their type annotation) and *principal types* (the type checker infers the tightest bounds without user annotations in common cases).

### Who Desperately Needs This

- **Data science platform builders** (Databricks, Weights & Biases, MLflow) who want to provide *built-in leakage guarantees* as a platform feature. BitLeak's type-checking approach integrates naturally into pipeline definition APIs—leakage bounds become part of the pipeline's type signature, caught at definition time rather than runtime.
- **Regulated industries** (finance, healthcare) where model validation requires *formal guarantees* about data separation. Current validation is manual and incomplete. BitLeak's type-level certificates provide machine-checkable proofs of compliant data handling that satisfy regulatory auditors.
- **The PL+ML research community** which is actively searching for the right abstractions to bring programming language rigor to ML workflows. BitLeak provides a concrete, useful instance of "PL for ML" that demonstrates the value of type-theoretic thinking in data science.

### Why This Is Genuinely Hard

1. **Designing a type system that is both sound and usable** for data scientists (who are not PL researchers) is a fundamental tension. The type annotations must be inferrable in common cases and the error messages must be interpretable by someone who has never heard of information flow.
2. **Quantitative information-flow types for statistical operations** are unstudied. Prior QIF type systems handle simple imperative programs. Extending this to operations over high-dimensional DataFrames with schema polymorphism (column names as types), groupby semantics, and index alignment is a genuine theoretical contribution.
3. **The fit/predict paradigm creates a unique challenge:** a `fit()` call produces an estimator that *encapsulates* information about its training data. Typing the information content of fitted model parameters requires bounding the capacity of arbitrary sklearn estimators—from linear models (bounded by parameter count) to tree ensembles (bounded by tree depth × n_trees) to kernel methods (bounded by support vector count).
4. **Type inference over pipeline DAGs** with branching, merging, and feedback (cross-validation) requires solving constraint systems over the leakage-bound lattice, which may have exponential solution spaces.

### Best-Paper Argument

BitLeak is a *category-creating contribution*: it establishes "information-flow type systems for data science" as a research direction at the intersection of PL and ML. The paper makes three distinct contributions that independently merit publication: (1) the first quantitative information-flow type theory for statistical computations, with soundness and principal type theorems; (2) a practical type checker for scikit-learn/pandas pipelines that infers leakage bounds without user annotations; (3) a large-scale empirical evaluation demonstrating that the type system catches real leakage bugs in thousands of public pipelines while maintaining a low false-positive rate. The combination of deep theory with practical impact is the hallmark of best-paper winners at venues like ICML and NeurIPS.

### Fatal Flaws & Weaknesses

- **The type system framing may alienate ML reviewers** who are unfamiliar with or hostile to PL concepts. "Type checking" is not in the ML vocabulary, and the paper may be desk-rejected at pure ML venues as "out of scope."
- **The analogy to security type systems may break down** in important ways. Security information flow is about *confidentiality* (preventing any leakage); ML leakage is about *validity* (preventing leakage that inflates metrics). These have different threat models, and forcing them into the same framework may produce an awkward fit.
- **Principal type inference may not hold** for realistic pipelines with complex control flow, custom transformers, or dynamic feature selection—degrading to a system that requires extensive manual annotation.
- **The 150K LoC target** may be hard to hit with a type-checker architecture (which is typically more compact) without padding with API models, which reviewers may view as uninteresting bulk.

---

## Comparative Assessment

| Dimension | **A: LeakageIR** | **B: TaintFlow** | **C: BitLeak** |
|---|---|---|---|
| **Primary angle** | Verification / certification | Debugging / diagnostics | Type system / PL theory |
| **Target venue** | MLSys, KDD | NeurIPS, ICML | ICML, POPL (stretch) |
| **Theory depth** | Medium (abstract interpretation) | Medium (differential analysis) | High (type theory + soundness) |
| **Practitioner appeal** | High (CI/CD integration) | Highest (debugging pain point) | Medium (conceptual leap needed) |
| **Novelty** | High (first quantitative framework) | Medium-High (novel decomposition) | Highest (new type theory) |
| **Risk of "just engineering"** | Medium | Low-Medium | Low |
| **Risk of "out of scope"** | Low | Low | Medium-High |
| **Feasibility at 150K LoC** | High (API surface drives LoC) | High (dual analysis + API surface) | Medium (type checker is compact) |
| **Best-paper potential** | Strong | Strong | Strongest (if audience is right) |

### Recommendation

**Framing A (LeakageIR)** is the safest choice: it is clearly scoped, has obvious practical value, and maps naturally to the 150K LoC requirement through the extensive API modeling needed. It will be well-received at systems-oriented venues (MLSys, KDD).

**Framing B (TaintFlow)** has the strongest practitioner story and the most compelling "new lens on old problem" narrative. The differential auditing angle and bidirectional analysis are distinctive. Best fit for NeurIPS or ICML where "useful new perspective" papers win awards.

**Framing C (BitLeak)** has the highest ceiling but also the highest risk. If the type theory is clean and the empirical results are strong, it could be a landmark paper. But it requires the reviewers to buy into the PL framing, which is not guaranteed at ML venues.

**For maximum best-paper probability at an ML venue, we recommend Framing B (TaintFlow) as the primary framing, with the technical core of Framing A (abstract interpretation with channel capacity bounds) as the engine, and the theoretical ambition of Framing C (soundness theorems, principal types) as stretch contributions.** This combines the strongest motivation (debugging reproducibility failures), the most distinctive methodology (differential auditing with bidirectional attribution), and sufficient theoretical depth to stand out from engineering contributions.
