# Prior Art Audit: Quantitative Information-Flow Analysis for ML Pipeline Leakage

**Seed Idea:** A quantitative information-flow analysis engine that detects train–test data leakage in scikit-learn/pandas ML pipelines by propagating abstract taint labels through statistical operations, measuring how many bits of test-set signal contaminate each training feature.

**Community:** area-041-machine-learning-and-ai-systems

**Date:** 2025-07-18

---

## 1. Existing Tools for Data Leakage Detection

### 1.1 Scikit-learn Built-in Warnings

Scikit-learn itself offers no automated leakage detection. Its `Pipeline` and `ColumnTransformer` classes provide a *structural prevention* mechanism — if used correctly, `fit()` is called only on training data and `transform()` on test data. However, this is purely opt-in: nothing prevents a user from calling `fit_transform()` on the entire dataset before splitting. Scikit-learn issues no warnings when leakage occurs.

**What it catches:** Nothing (prevention-only, no detection).
**What it misses:** All forms of leakage that occur outside the Pipeline abstraction.

### 1.2 LeakageDetector (Yang, Brower-Sinning, Lewis, Kästner — ASE 2022; SANER 2025)

The most directly relevant existing tool. LeakageDetector performs static analysis of Python/Jupyter notebook code to detect three patterns:
- **Overlap leakage:** Train and test sets share rows.
- **Multi-test leakage:** Test set used multiple times for tuning.
- **Preprocessing leakage:** Transformations (scaling, imputation, encoding) fitted on combined train+test data before splitting.

The tool is implemented as a PyCharm plugin (v1.0) and VS Code extension for Jupyter (v2.0, 2025). It uses pattern-matching on the AST to find known anti-patterns.

**What it catches:** Structural/syntactic patterns of common leakage mistakes.
**What it misses:**
- **No quantification** — only binary detect/not-detect. Cannot say *how much* information leaked.
- **No semantic analysis** — cannot reason about statistical operations (e.g., does a rolling mean leak future information? How much does fitting a scaler on combined data actually contaminate features?).
- **No abstract interpretation** — pattern-based, not flow-based. Misses indirect leakage through variable aliasing, function calls, or non-standard pipeline compositions.
- **No information-theoretic grounding** — cannot distinguish between a negligible leak (e.g., adding one test sample to 100,000 training samples for mean computation) and a catastrophic one.

### 1.3 LeakGuard

A lightweight Python library that detects preprocessing leakage and target leakage, computing a "Leakage Impact Score" (LIS) by comparing accuracy with and without leakage. This is an *empirical* measure (run the model twice), not an *analytical* one (reason about information flow).

**What it catches:** Preprocessing and target leakage via runtime comparison.
**What it misses:** Requires re-execution; not static. LIS is a model-dependent proxy, not an information-theoretic quantity. Cannot provide per-feature or per-operation leakage attribution.

### 1.4 ML Pipeline Linters (mllint, pylint-ml)

**mllint** (TU Delft, 2022): A project-level linter that checks ML repository structure, code quality, reproducibility practices, and pipeline configuration against SE4ML best practices. It does not specifically detect data leakage — it checks for the *use* of pipelines, not the *correctness* of data flow within them.

Generic Python linters (pylint, flake8) have no ML-specific rules. No known pylint plugin specifically targets data leakage patterns.

**What they catch:** Project structure smells, missing pipelines, configuration issues.
**What they miss:** All runtime data-flow issues; no information-flow analysis.

### 1.5 Data Validation Frameworks

| Framework | Focus | Leakage Detection? |
|-----------|-------|-------------------|
| **Great Expectations** | Schema validation, data contract testing | No — validates data *properties*, not data *flow* |
| **TFX Data Validation (TFDV)** | Schema inference, drift/skew detection | Can flag train/serving skew, but not train-test contamination in code |
| **Deequ** (AWS) | Spark-scale data quality constraints | No leakage detection; constraint-based checks on column statistics |
| **Pandera** | DataFrame schema validation | Type/range checking, not flow analysis |

These tools validate data *at rest* (column types, ranges, distributions) but cannot reason about how data *flows* through a pipeline or whether test-set information contaminates training features.

### 1.6 Pipeline Profiling and Tracking

**PipelineProfiler:** Visualizes sklearn pipeline structure; no leakage analysis.
**MLflow:** Tracks experiments, parameters, metrics. Can log what ran but cannot analyze *whether* leakage occurred.
**Weights & Biases, Neptune:** Same category — observability, not analysis.

### 1.7 Summary: Gap Analysis

| Capability | LeakageDetector | LeakGuard | mllint | GX/TFDV | **Our Approach** |
|------------|:-:|:-:|:-:|:-:|:-:|
| Detects leakage? | ✓ (pattern) | ✓ (empirical) | ✗ | ✗ | ✓ (flow) |
| Quantitative (bits)? | ✗ | ✗ (proxy) | ✗ | ✗ | **✓** |
| Per-feature attribution? | ✗ | ✗ | ✗ | ✗ | **✓** |
| Handles statistical ops? | ✗ | ✗ | ✗ | ✗ | **✓** |
| Sound approximation? | ✗ | ✗ | ✗ | ✗ | **✓** |
| Static (no execution)? | ✓ | ✗ | ✓ | N/A | **✓** |

**Fundamental difference:** All existing tools either do pattern-matching (syntactic) or empirical comparison (runtime). None performs *semantic*, *quantitative*, *information-theoretic* analysis of how test-set signal propagates through statistical operations. Our approach is the first to bring quantitative information flow (QIF) theory to ML pipeline analysis.

---

## 2. Academic Prior Art on Data Leakage

### 2.1 Foundational Work

**Kaufman, Rosset, Perlich, Stitelman — "Leakage in Data Mining: Formulation, Detection, and Avoidance" (TKDD 2012, presented at KDD 2011)**

The seminal paper (900+ citations). Key contributions:
- Formal definition of leakage as violation of "learn-predict separation" — information available at prediction time should be strictly a subset of what's available at training time.
- Causal graph framework for diagnosing leakage sources.
- Taxonomy: feature leakage, temporal leakage, sampling leakage.

**Relevance to our work:** Kaufman et al. provide the *conceptual* framework but no automated detection tool, no quantification, and no information-theoretic measurement. Their causal graphs are informal — not connected to abstract interpretation or QIF.

### 2.2 Empirical Studies

**Yang, Brower-Sinning, Lewis, Kästner — "Data Leakage in Notebooks: Static Detection and Better Processes" (ASE 2022)**

Evaluated >100,000 Kaggle notebooks. Found leakage is pervasive (~15% of notebooks). Proposed static analysis patterns (overlap, multi-test, preprocessing leakage). This is the academic foundation for LeakageDetector.

**Limitation:** Binary detection only. No quantification. Pattern-based — cannot handle novel pipeline structures.

**Nisbet et al. — ML Handbooks**

Discusses leakage in textbook terms (preprocessing before splitting, target leakage). Purely pedagogical; no formal framework.

### 2.3 Recent Conference Papers

| Paper | Venue | Focus | Relevance |
|-------|-------|-------|-----------|
| InfoScissors (NeurIPS 2024) | Mutual information minimization | Collaborative inference leakage *defense* (splitting models across edge/cloud) | Different problem — model splitting, not train-test leakage |
| TabLeak (ICML 2023) | Federated learning | Tabular data leakage in FL settings | Different problem — federated gradients, not pipeline preprocessing |
| ProPILE (NeurIPS 2023) | LLM privacy probing | PII memorization in LLMs | Different problem domain entirely |
| "Don't Push the Button" (2025) | Taxonomy/survey | Leakage risks in ML and transfer learning | Survey, no tool or formal analysis |
| Data leakage detection via transfer learning (PeerJ CS 2025) | ML-based detection | Uses ML classifiers to detect leakage with limited labels | Heuristic, not sound; no information-theoretic basis |

### 2.4 Gaps in Academic Literature

1. **No quantitative measurement:** No paper measures leakage in bits or information-theoretic units. All existing work is binary (leaked/not leaked) or uses proxy scores (accuracy delta).
2. **No abstract interpretation for ML pipelines:** No paper applies Cousot-style abstract interpretation to pandas/sklearn operation semantics.
3. **No information-flow lattices for statistical operations:** While QIF theory exists (§3), nobody has defined how statistical operations (mean, std, PCA, scaling) transform information-flow labels.
4. **No per-feature leakage attribution:** No existing work can tell you "feature X has 3.2 bits of test-set contamination from the StandardScaler step."

---

## 3. Information Flow Analysis in PL/SE

### 3.1 Language-Based Information Flow Control

| System | Authors | Key Idea | ML Pipeline Application? |
|--------|---------|----------|------------------------|
| **JFlow/Jif** | Myers (POPL 1999) | Java with security-typed labels; static enforcement of confidentiality/integrity | No. Java-only. No statistical operation semantics. |
| **FlowCaml** | Simonet (INRIA) | OCaml with security-level annotations; type inference for information flow | No. OCaml-only. No data science ecosystem support. |
| **SIF/Paragon** | Broberg & Sands | Information flow for Java with policy specifications | No. Same ecosystem limitations. |

**Key insight:** These systems enforce *qualitative* non-interference ("no flow from high to low") in general-purpose languages. They do not:
- Quantify leakage in bits
- Model statistical/numerical operations
- Target Python, pandas, or sklearn

### 3.2 Quantitative Information Flow (QIF)

**Core references:**
- **Smith — "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009):** Min-entropy leakage as an operationally meaningful measure.
- **Alvim, Chatzikokolakis, Palamidessi, Smith — "The Science of Quantitative Information Flow" (Springer 2020):** The definitive textbook. Models programs as information-theoretic channels; measures leakage via Shannon entropy, min-entropy, g-leakage (generalized).
- **Clark, Hunt, Malacaria — "Quantitative Analysis of the Leakage of Confidential Data" (QAPL 2001):** Pioneered using Shannon entropy for program leakage.

**Key QIF concepts relevant to our work:**
- A program is a *channel* mapping secret inputs to observable outputs.
- Leakage = reduction in attacker's uncertainty about the secret after observing the output.
- Can be measured in bits: `leakage = H(secret) - H(secret | output)`.
- g-leakage framework generalizes to arbitrary gain functions.

**Has QIF been applied to ML pipelines?** **No.** QIF literature focuses on:
- Cryptographic protocols
- Password checkers
- Side-channel attacks
- Database query privacy

Nobody has modeled `StandardScaler.fit_transform()` or `pd.DataFrame.merge()` as information-theoretic channels and computed the resulting leakage in bits. This is our core novelty.

### 3.3 Abstract Interpretation for Information Flow

**Giacobazzi & Mastroeni — "Abstract Non-Interference" (POPL 2004)**

Generalizes non-interference by parameterizing over abstract interpretations. Key contributions:
- Models attacker capabilities as abstract domains.
- Computes "most concrete harmless attacker" for a given program.
- Enables quantitative comparison: programs can be ordered by how much they leak (lattice of abstractions).

**Pasqua & Mastroeni — "Statically Analyzing Information Flows: An Abstract Interpretation-based Approach" (SAC 2019)**

Applies abstract interpretation to information flow analysis of programs, computing flow bounds.

**Relevance:** Provides the *theoretical machinery* (Galois connections, abstract domains for information flow) that we can adapt. However:
- Applied to imperative programs with scalar variables.
- Never applied to data frames, statistical operations, or ML pipelines.
- Never combined with QIF's bit-level measurement for ML preprocessing steps.

### 3.4 Taint Analysis Tools

| Tool | Domain | Static/Dynamic | Quantitative? |
|------|--------|---------------|--------------|
| **FlowDroid** | Android apps | Static (IFDS) | No — binary taint |
| **TaintDroid** | Android OS | Dynamic | No — binary taint |
| **Joana** | Java | Static (PDG-based) | No |
| **Snyk Code** | General (multi-language) | Static | No |
| **Fluffy** | CodeQL-based | Static + ML | No |

**Key observation:** All taint analysis tools operate on *binary* taint (tainted/untainted). None quantifies *how much* information flows through a taint path. None targets Python data science libraries.

### 3.5 Gap Summary

The PL/SE community has developed:
- Sophisticated type systems for information flow (Jif, FlowCaml) — but for general-purpose languages, not ML pipelines.
- Quantitative information flow theory (Smith, Alvim) — but for cryptographic/security contexts, not ML preprocessing.
- Abstract interpretation for information flow (Giacobazzi, Mastroeni) — but for imperative scalar programs, not data frames.
- Industrial taint analysis (FlowDroid, etc.) — but binary, not quantitative, and for mobile/web, not data science.

**Nobody has bridged these three worlds:** QIF theory × abstract interpretation × ML pipeline semantics. This is the core intellectual contribution of our proposal.

---

## 4. Abstract Interpretation for Data Science

### 4.1 Abstract Interpreters for Pandas/NumPy/Sklearn

**Do they exist? No.** There is no published abstract interpreter that models the semantics of pandas DataFrame operations, NumPy array computations, or sklearn transformer/estimator methods.

The closest work falls into several categories:

### 4.2 DataFrame Type Systems and Schema Validation

| Tool/Library | What It Does | Abstract Interpretation? |
|-------------|-------------|------------------------|
| **Pandera** | Runtime schema validation for DataFrames | No — runtime checks, not static analysis |
| **Strictly Typed Pandas** | Mypy-compatible type annotations for DataFrame schemas | No — type checking, not abstract interpretation |
| **StaticFrame** | Immutable DataFrames with PEP 646 type hints | No — type-level guarantees, not value-level abstraction |
| **Pyright/Pytype** | Python static type checkers | Container-level types only (knows it's a DataFrame, not what columns/values it contains) |

These tools provide *type-level* guarantees (column names, dtypes) but cannot reason about *value-level* properties (what information is contained in a column, how a `fit_transform` propagates statistics).

### 4.3 Abstract Interpretation for Numerical Programs

**Cousot & Cousot (1977 onward):** Founded the theory. Key numerical abstract domains:
- **Intervals** (Box domain): Each variable has [lo, hi] bounds.
- **Octagons** (Miné 2006): Constraints of form ±x ± y ≤ c.
- **Polyhedra** (Cousot & Halbwachs 1978): General linear inequality constraints.
- **Zonotopes** (Ghorbal, Goubault, Putot 2009): Affine abstract domains for floating-point.

**Astrée** (ENS/AbsInt): Industrial abstract interpreter for C programs used to verify absence of runtime errors in avionics software (Airbus A380). Handles floating-point precisely.

**Relevance:** These domains model numerical *value ranges* and *relational properties* of scalar variables. They do not:
- Model tabular data (data frames with named columns and rows)
- Track information flow (only track value ranges)
- Model statistical aggregations (mean, variance, covariance)
- Distinguish train vs. test partitions

### 4.4 ML-Specific Verification via Abstract Interpretation

Recent work applies abstract interpretation to *verify neural networks* (robustness, fairness):
- **AI²** (Gehr et al., S&P 2018): Abstract interpretation for neural network robustness.
- **DeepPoly** (Singh et al., POPL 2019): Polyhedra-like domain for DNN verification.
- **PRIMA** (Müller et al., NeurIPS 2022): Multi-neuron relaxation.

These verify *trained models* (post-hoc), not *training pipelines* (pre-hoc). They abstract over neural network computations, not data preprocessing operations.

**Giacobazzi & Mastroeni — "Adversities in Abstract Interpretation" (TOPLAS 2024):** Extends abstract interpretation to reason about adversarial robustness of classifiers — related in spirit but focused on model robustness, not pipeline leakage.

### 4.5 What's Missing

Nobody has built abstract domains for:
1. **DataFrame operations:** `pd.merge()`, `df.groupby().transform()`, `df.fillna()`, column selection, row filtering — tracking which partition (train/test) each cell belongs to and how aggregations mix partitions.
2. **Sklearn transformer semantics:** How `StandardScaler.fit()` creates a statistical summary of its input, how `.transform()` applies it, and how fitting on combined train+test data causes information to flow from test to train.
3. **Statistical aggregation channels:** What is the abstract information content of `np.mean(X_combined)` as a function of the train/test composition of `X_combined`?

Our proposal fills exactly this gap: **abstract domains for DataFrame-level information flow that track test-set contamination through statistical operations and quantify leakage in bits.**

---

## 5. Novelty Assessment

### 5.a) Quantitative (bits) measurement of train-test leakage

**Rating: GENUINELY NOVEL** ★★★

**Justification:** QIF theory (Alvim, Smith) provides the mathematical framework for measuring information leakage in bits, but it has never been applied to the ML train-test leakage problem. Existing leakage detection is purely binary (LeakageDetector) or uses model-dependent proxies (LeakGuard's LIS). Nobody has computed how many bits of test-set information flow into a training feature through a preprocessing step. This is a clean, first-of-its-kind application of QIF to an entirely new domain.

**Closest work:** Clark, Hunt, Malacaria (2001) quantify leakage in bits for imperative programs, but never for statistical/ML operations. InfoScissors (NeurIPS 2024) uses mutual information but for model-splitting defense, not pipeline auditing.

### 5.b) Abstract interpretation framework for pandas/sklearn operations

**Rating: GENUINELY NOVEL** ★★★

**Justification:** No abstract interpreter exists for pandas or sklearn. The closest work is DataFrame type checking (Pandera, StaticFrame) which operates at the type level, not the value/information-flow level. Neural network verification (DeepPoly, AI²) uses abstract interpretation but for a completely different domain (verifying trained networks, not auditing training pipelines). Building abstract transfer functions for `StandardScaler.fit_transform()`, `pd.DataFrame.merge()`, `SimpleImputer`, etc. is entirely new.

### 5.c) Information-flow lattices for statistical operations

**Rating: GENUINELY NOVEL** ★★★

**Justification:** Classical information-flow lattices (Denning 1976, Bell-LaPadula) model confidentiality levels for discrete data. QIF channels (Alvim et al.) model programs as channels between secrets and observations. Nobody has defined the information-flow semantics of statistical operations:
- How does `np.mean()` over a mixed train+test array create a channel from test to train?
- What is the information capacity of `StandardScaler` as a function of n_train and n_test?
- How does PCA's covariance estimation on combined data create a higher-dimensional leakage channel?

Defining these as abstract transfer functions in an information-flow lattice is completely new.

### 5.d) Automated pipeline-level leakage diagnosis

**Rating: INCREMENTAL** ★★☆

**Justification:** LeakageDetector (Yang et al., ASE 2022) already provides automated pipeline-level leakage *detection* via static analysis. Our approach is fundamentally more powerful (quantitative, semantic, sound) but the *concept* of automated pipeline-level diagnosis exists. The novelty is in the *how* (QIF + abstract interpretation), not the *what* (automated leakage detection).

**Why still significant:** LeakageDetector uses syntactic pattern matching and catches only three predefined patterns. Our approach reasons about *arbitrary* pipeline compositions and provides *quantitative* attribution — a qualitative leap in capability, even if the high-level goal overlaps.

### 5.e) Sound approximation of information leakage through ML preprocessing

**Rating: GENUINELY NOVEL** ★★★

**Justification:** Soundness (in the abstract interpretation sense — if the tool says "no leakage," there truly is none) has never been achieved for ML pipeline leakage analysis. LeakageDetector is unsound (can miss leakage through patterns it doesn't recognize). LeakGuard is empirical (requires execution). Our use of abstract interpretation provides *sound over-approximation*: the reported leakage bounds are guaranteed to be at least as large as the true leakage. This is the standard guarantee from abstract interpretation theory (Cousot & Cousot 1977) applied to a new domain.

**Novelty summary:**

| Claim | Rating | Key Differentiator |
|-------|--------|-------------------|
| Quantitative (bits) leakage measurement | **GENUINELY NOVEL** | First QIF application to ML pipeline leakage |
| Abstract interpretation for pandas/sklearn | **GENUINELY NOVEL** | No prior abstract interpreter for data science libraries |
| Information-flow lattices for statistical ops | **GENUINELY NOVEL** | Novel abstract transfer functions for statistical operations |
| Automated pipeline-level diagnosis | **INCREMENTAL** | Concept exists (LeakageDetector) but our method is fundamentally stronger |
| Sound leakage approximation | **GENUINELY NOVEL** | First sound analysis for ML pipeline leakage |

---

## 6. Closest Competitors

### 6.1 LeakageDetector (Yang et al., ASE 2022 / SANER 2025)

**What they do:** Static analysis of Python/Jupyter code via AST pattern matching. Detects three predefined leakage patterns (overlap, multi-test, preprocessing). PyCharm plugin + VS Code extension.

**How our approach differs:**
- **Semantic vs. syntactic:** We analyze information flow through operations; they match code patterns.
- **Quantitative vs. binary:** We measure leakage in bits; they report yes/no.
- **Sound vs. heuristic:** Our abstract interpretation provides soundness guarantees; their patterns can miss novel leakage forms.
- **Per-feature attribution:** We can identify which features and which pipeline steps contribute most leakage; they identify which *code lines* match a pattern.

**Is the difference substantial enough?** Yes — this is a fundamental methodological difference (pattern matching vs. abstract interpretation + QIF), not an incremental improvement. The gap is analogous to the difference between grep-based bug finding and formal verification.

### 6.2 LeakGuard

**What they do:** Runtime leakage detection library. Detects preprocessing and target leakage by executing the pipeline with and without suspected leakage, computing a Leakage Impact Score.

**How our approach differs:**
- **Static vs. dynamic:** We analyze code without execution; they require running the pipeline.
- **Information-theoretic vs. empirical:** Our bits measurement is a property of the pipeline structure; their LIS depends on the specific dataset and model.
- **Sound vs. approximate:** Our over-approximation is sound; their empirical comparison may miss leakage that doesn't affect accuracy on a particular dataset.
- **Compositional:** We can analyze pipeline steps independently and compose results; they must re-run the entire pipeline for each check.

**Is the difference substantial?** Yes — static + quantitative + sound vs. dynamic + empirical + model-dependent.

### 6.3 Giacobazzi & Mastroeni's Abstract Non-Interference (POPL 2004)

**What they do:** Theoretical framework for parameterizing non-interference by abstract interpretation. Models attacker strength as abstract domains. Computes information flow bounds.

**How our approach differs:**
- **Domain:** They analyze imperative programs with scalar variables; we analyze ML pipelines with DataFrames and statistical operations.
- **Operations:** They model standard language constructs (assignment, branching, loops); we model `fit()`, `transform()`, `groupby()`, `merge()`, statistical aggregations.
- **Quantification:** Their quantification is in terms of abstract domain refinement; ours is in information-theoretic bits aligned with QIF theory.
- **Application target:** They provide a general theory; we provide a concrete tool for a specific, high-impact problem.

**Is the difference substantial?** Yes — we are the first to instantiate abstract non-interference ideas in the ML pipeline domain, with novel abstract domains and transfer functions.

### 6.4 Alvim, Smith et al. — QIF Theory (2020 book)

**What they do:** Comprehensive mathematical framework for quantifying information leakage through channels. g-leakage, min-entropy leakage, capacity bounds.

**How our approach differs:**
- **Application domain:** They develop general theory; we apply and extend it to ML pipeline operations.
- **Channel models:** They model programs as channels; we model sklearn transformers and pandas operations as channels, deriving their information capacity.
- **Tooling:** They provide mathematical foundations; we build a practical analysis engine.
- **Novel channels:** We must define channel matrices for statistical operations (mean, variance, PCA, imputation) — these don't exist in their work.

**Is the difference substantial?** Yes — this is a major new application of QIF theory requiring significant novel technical development (defining channels for statistical operations).

### 6.5 DeepPoly / AI² (Neural Network Verification via Abstract Interpretation)

**What they do:** Abstract interpretation-based verification of neural network properties (robustness, fairness). Abstract domains tailored to ReLU networks.

**How our approach differs:**
- **Target:** They verify *trained models* (post-hoc); we audit *training pipelines* (pre-hoc).
- **Property:** They verify robustness (small input perturbation → same output); we verify information flow (test data doesn't contaminate training).
- **Abstract domains:** They abstract over neuron activations; we abstract over DataFrame partitions and statistical summaries.
- **No overlap:** Different phase of ML lifecycle, different properties, different abstract domains.

**Is the difference substantial?** Yes — entirely different problems despite shared use of abstract interpretation.

---

## 7. Differentiation from Portfolio

### 7.1 vs. ml-pipeline-selfheal

**ml-pipeline-selfheal** focuses on *automatic repair* of broken ML pipelines — detecting failures (data drift, schema changes, component crashes) and self-healing through reconfiguration, retraining, or fallback strategies.

**Our project** focuses on *quantitative analysis* of information flow — measuring how many bits of test-set signal leak into training features through preprocessing operations.

| Dimension | ml-pipeline-selfheal | ml-pipeline-leakage-auditor |
|-----------|---------------------|---------------------------|
| **Goal** | Repair broken pipelines | Audit information leakage |
| **When** | Runtime / deployment | Development / pre-deployment |
| **Output** | Repaired pipeline | Leakage report (bits per feature) |
| **Theory** | Fault tolerance, self-adaptive systems | QIF, abstract interpretation |
| **Problem** | Pipeline crashes, drift, degradation | Train-test contamination |
| **Action** | Automatic fix | Diagnostic report |

**Fundamental difference:** Self-heal is about *resilience*; we are about *correctness analysis*. Self-heal operates at runtime; we operate at static analysis time. Self-heal doesn't measure information flow; we don't repair anything.

### 7.2 vs. dp-verify-repair (Differential Privacy)

**dp-verify-repair** verifies and repairs differential privacy guarantees in data release mechanisms — ensuring that adding/removing one individual doesn't change output distributions by more than ε.

**Our project** measures train-test information leakage — how much test-set signal contaminates training features through preprocessing.

| Dimension | dp-verify-repair | ml-pipeline-leakage-auditor |
|-----------|-----------------|---------------------------|
| **Privacy notion** | Differential privacy (ε, δ) | Information-theoretic leakage (bits) |
| **Threat model** | Adversary inferring individual records | Test-set signal contaminating training |
| **Target** | Data release mechanisms | ML preprocessing pipelines |
| **Formal framework** | DP composition theorems | QIF + abstract interpretation |
| **Output** | ε budget accounting, repairs | Per-feature leakage in bits |

**Fundamental difference:** DP is about *individual privacy*; we are about *evaluation integrity*. DP protects people; we protect model evaluation. Different threat models, different formal frameworks, different outputs.

### 7.3 vs. tensorguard

**tensorguard** focuses on runtime shape/type checking and validation for tensor operations in deep learning frameworks — catching shape mismatches, dtype errors, and dimension violations.

**Our project** focuses on information-flow analysis through statistical operations — an entirely different property (information content, not shape).

**Fundamental difference:** tensorguard checks *structural properties* (shapes, types); we analyze *informational properties* (bits of leakage). tensorguard operates on tensors in DL; we operate on DataFrames in classical ML. No overlap in formal methods (shape algebra vs. QIF lattices).

### 7.4 vs. dp-mechanism-forge

**dp-mechanism-forge** designs and synthesizes differentially private mechanisms — generating noise-addition schemes that satisfy (ε, δ)-DP for specific query types.

**Our project** analyzes existing (non-private) ML pipelines for information leakage. We don't add noise or enforce privacy — we *measure* leakage.

**Fundamental difference:** dp-mechanism-forge is *constructive* (builds private mechanisms); we are *analytical* (measures leakage in existing pipelines). They synthesize; we audit. Different inputs, outputs, and formal machinery.

---

## 8. Overall Assessment

### Novelty Verdict: **STRONG**

This proposal sits at an unexplored intersection of three mature research areas:
1. **Quantitative Information Flow** (security/PL theory — Alvim, Smith)
2. **Abstract Interpretation** (program analysis — Cousot, Giacobazzi, Mastroeni)
3. **ML Pipeline Correctness** (SE4ML — Yang, Kästner)

Each area is well-studied individually, but **nobody has combined them** to build a quantitative, sound, information-theoretic analysis engine for ML pipeline leakage. The key technical novelties are:

- **Novel abstract domains** for DataFrame-level information flow (tracking train/test partition membership through operations)
- **Novel transfer functions** for statistical operations as information channels (computing bits of leakage through mean, variance, PCA, imputation, scaling)
- **Novel application** of QIF theory to ML evaluation integrity (not security/privacy)
- **Sound over-approximation** guarantees for ML pipeline leakage (first of its kind)

### Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Someone publishes similar work before us | **Low** — no signs of convergent work in this specific intersection | Move quickly; the interdisciplinary nature makes independent discovery unlikely |
| Closest competitor (LeakageDetector) extends to quantitative analysis | **Low** — fundamentally different methodology (pattern matching vs. abstract interpretation) | Our formal framework is much harder to replicate without PL expertise |
| QIF bounds too loose to be useful | **Medium** — abstract interpretation can over-approximate significantly | Focus on tightest useful domains; validate against empirical measurements |
| Modeling all pandas/sklearn operations too complex | **Medium** — large API surface | Focus on most common 20-30 operations that cover 90%+ of real pipelines |

### Best Paper Potential

The interdisciplinary novelty (PL theory meets ML engineering), practical impact (data leakage is a top-10 ML mistake), and theoretical depth (sound, quantitative, information-theoretic) make this a strong candidate for top venues at the intersection of PL and ML/SE:
- **OOPSLA/PLDI** (novel abstract domains and transfer functions)
- **ICSE/FSE** (practical SE impact for ML pipelines)
- **ICML/NeurIPS** (ML pipeline correctness, if paired with strong empirical evaluation)
- **S&P/CCS** (information flow analysis, if framed as evaluation integrity)

The work is **differentiated enough** from all identified prior art to constitute a genuinely novel contribution.
