# Mathematical Specification: Quantitative Information-Flow Analysis for ML Pipeline Leakage Auditing

**Stage:** Crystallization — Math Enumeration  
**Project:** ml-pipeline-leakage-auditor  
**Community:** area-041 (ICML, NeurIPS, AAAI, ICLR, MLSys, KDD)

---

## 0. Notation and Preliminaries

| Symbol | Meaning |
|--------|---------|
| $\mathcal{D}$ | A tabular dataset (matrix of rows × columns) |
| $\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}}$ | Train and test partitions of $\mathcal{D}$ |
| $X_j$ | The $j$-th feature column (random variable) |
| $\pi = (s_1, s_2, \dots, s_K)$ | An ML pipeline of $K$ stages |
| $s_k: \mathcal{D} \to \mathcal{D}'$ | A pipeline stage (e.g., `StandardScaler.fit_transform`) |
| $H(X)$ | Shannon entropy of $X$ |
| $I(X; Y)$ | Mutual information between $X$ and $Y$ |
| $C(\mathcal{C})$ | Channel capacity of abstract channel $\mathcal{C}$ |
| $(L, \sqsubseteq, \sqcup, \sqcap, \bot, \top)$ | A complete lattice |
| $\alpha: \mathcal{P}(\text{Concrete}) \to L$ | Abstraction function |
| $\gamma: L \to \mathcal{P}(\text{Concrete})$ | Concretization function |
| $f^\sharp: L \to L$ | Abstract transformer for concrete operation $f$ |
| $\lambda_k^{(j)}$ | Leakage (in bits) at stage $k$ for feature $j$ |
| $\Lambda(\pi)$ | Total pipeline leakage: $\sum_j \max_k \lambda_k^{(j)}$ |

Throughout, "leakage" means information flow from the test partition $\mathcal{D}_{\text{te}}$ into quantities that influence the training process on $\mathcal{D}_{\text{tr}}$. A pipeline with zero leakage computes training features using only $\mathcal{D}_{\text{tr}}$.

---

## 1. Three Competing Mathematical Framings

### 1.A — Framing A: Abstract Interpretation over Information-Flow Lattices

#### 1.A.1 Core Idea

Model every pipeline operation as an *abstract transformer* over a lattice that tracks how much test-set information each intermediate value carries. The lattice elements are not merely "tainted/untainted" (binary) but encode *quantitative upper bounds* on information content, measured in bits. A fixpoint computation over the pipeline DAG produces sound per-feature leakage certificates.

#### 1.A.2 Mathematical Objects

**Object A1: The Partition-Taint Lattice $\mathcal{T}$.**

Define the set of *data origins* $\mathcal{O} = \{\text{tr}, \text{te}, \text{ext}\}$ (train, test, external/constant). A *taint label* is a pair $\tau = (O, b)$ where $O \subseteq \mathcal{O}$ is the set of origins contributing to this value, and $b \in [0, \infty]$ is an upper bound on the bits of test-set information.

$$\mathcal{T} = \bigl(\mathcal{P}(\mathcal{O}) \times [0,\infty], \sqsubseteq\bigr)$$

with partial order:

$$(O_1, b_1) \sqsubseteq (O_2, b_2) \iff O_1 \subseteq O_2 \;\land\; b_1 \leq b_2$$

- $\bot = (\emptyset, 0)$: no data dependency, zero leakage.
- $\top = (\mathcal{O}, \infty)$: depends on everything, unbounded leakage.
- Join: $(O_1, b_1) \sqcup (O_2, b_2) = (O_1 \cup O_2, \max(b_1, b_2))$.
- Meet: $(O_1, b_1) \sqcap (O_2, b_2) = (O_1 \cap O_2, \min(b_1, b_2))$.

This forms a complete lattice of finite height when $b$ is discretized to $\{0, 1, 2, \dots, B_{\max}, \infty\}$ for some precision parameter $B_{\max}$.

**Object A2: The DataFrame Abstract Domain $\mathcal{A}_{\text{df}}$.**

A DataFrame abstract state maps each column name to a taint label, plus *row-provenance* tracking:

$$\mathcal{A}_{\text{df}} = \bigl(\text{ColNames} \to \mathcal{T}\bigr) \times \mathcal{R}$$

where $\mathcal{R}$ is a row-provenance domain:
$$\mathcal{R} \in \{\texttt{train-only}, \texttt{test-only}, \texttt{mixed}(\rho)\}$$

with $\rho \in [0,1]$ denoting the fraction of rows originating from the test set. This is essential because operations like `pd.concat([df_train, df_test])` create mixed-provenance frames, and subsequent `fit()` on such frames leaks $\rho \cdot H(X_{\text{te}})$ bits per column.

**Object A3: The Estimator State Domain $\mathcal{A}_{\text{est}}$.**

Scikit-learn estimators have *fitted parameters* (e.g., `mean_`, `scale_`, `components_`). The abstract domain for an estimator is:

$$\mathcal{A}_{\text{est}} = \bigl(\text{ParamNames} \to \mathcal{T}\bigr)$$

When `est.fit(X)` is called, each fitted parameter inherits a taint label derived from the rows and columns of `X` that influence it. For example, `StandardScaler.fit(X)` sets `mean_[j]` with taint label:

$$\tau(\texttt{mean\_}[j]) = \bigl(\text{origins}(X_j), \; \text{bits}_{\text{te}}(X_j, \mathcal{R})\bigr)$$

where $\text{bits}_{\text{te}}$ is defined below.

**Object A4: Quantitative Bit-Bound Function.**

For a statistical aggregate $\phi$ (mean, std, max, quantile, etc.) computed over a column $X_j$ with row-provenance $\mathcal{R}$:

$$\text{bits}_{\text{te}}(X_j, \mathcal{R}) = \begin{cases}
0 & \text{if } \mathcal{R} = \texttt{train-only} \\
H(X_j^{\text{te}}) & \text{if } \mathcal{R} = \texttt{test-only} \\
C_\phi(n_{\text{te}}, n_{\text{tr}}, d_j) & \text{if } \mathcal{R} = \texttt{mixed}(\rho)
\end{cases}$$

Here $C_\phi$ is the *channel capacity* of the statistical operation $\phi$ viewed as a channel from test rows to the aggregate output. The key insight is that `mean` over $n$ test rows among $N$ total rows transmits at most:

$$C_{\text{mean}}(n_{\text{te}}, N) \leq \log_2\!\Bigl(1 + \frac{n_{\text{te}}}{N} \cdot \text{SNR}(X_j)\Bigr)$$

where $\text{SNR}$ is the signal-to-noise ratio of the feature. For bounded features ($X_j \in [a, b]$), this simplifies to:

$$C_{\text{mean}}(n_{\text{te}}, N) \leq \log_2\!\Bigl(1 + \frac{n_{\text{te}}}{N}\Bigr) + \frac{1}{2}\log_2(2\pi e \cdot \text{Var}(X_j))$$

but the first term dominates and gives a clean, data-independent bound.

**Object A5: Abstract Transformers.**

For each pipeline operation $s_k$, we define an abstract transformer $s_k^\sharp: \mathcal{A}_{\text{df}} \times \mathcal{A}_{\text{est}} \to \mathcal{A}_{\text{df}} \times \mathcal{A}_{\text{est}}$.

*Example: `StandardScaler.fit_transform(X)`*

```
fit phase:
  for each column j in X:
    τ(mean_[j]) = (origins(X_j), bits_te(X_j, R))
    τ(scale_[j]) = (origins(X_j), bits_te(X_j, R))

transform phase:
  for each column j in X:
    τ(X'_j) = τ(X_j) ⊔ τ(mean_[j]) ⊔ τ(scale_[j])
```

The join $\sqcup$ in the transform phase captures that the output depends on both the input and the fitted parameters. If `fit` was called on mixed data, the fitted parameters carry test taint, which propagates to all transformed outputs.

*Example: `pd.merge(left, right, on='key')`*

```
τ(output_col) = τ(left_col) ⊔ τ(right_col) ⊔ τ(key_left) ⊔ τ(key_right)
R_output = mix(R_left, R_right, join_type)
```

The merge key columns contribute taint because the *set of matched rows* depends on key values from both sides.

*Example: `GroupBy.transform(func)`*

```
for each output column j:
    τ(X'_j) = ⊔_{j' ∈ group_cols ∪ {j}} τ(X_{j'})
    bits updated: bits_te(X'_j) = C_func(n_te_per_group, n_per_group)
```

GroupBy is particularly dangerous because group-level aggregates computed over mixed data leak test information into every row of the group.

**Object A6: Pipeline DAG and Fixpoint.**

The pipeline $\pi$ induces a directed acyclic graph $G = (V, E)$ where vertices are intermediate DataFrames/estimators and edges are operations. The analysis computes:

$$\text{lfp}(F^\sharp) \quad \text{where} \quad F^\sharp(\sigma) = \bigsqcup_{k} s_k^\sharp(\sigma|_{\text{inputs}(k)})$$

over the product domain $\prod_{v \in V} \mathcal{A}_v$ using a worklist algorithm. Since the lattice has finite height (discretized bit-bounds), termination is guaranteed without widening for acyclic pipelines. For pipelines with loops (e.g., iterative imputation), we apply widening:

$$\sigma_{i+1} = \sigma_i \nabla F^\sharp(\sigma_i) \quad \text{where} \quad (O, b_1) \nabla (O', b_2) = \bigl(O \cup O', \; b_2 \leq b_1 \,?\, b_1 : \infty\bigr)$$

#### 1.A.3 Key Theorems

**Theorem A1 (Soundness of Partition-Taint Abstraction).** *Let $\pi$ be a pipeline, $\mathcal{D}$ a dataset with partition into $\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}}$. Let $\sigma^\sharp = \text{lfp}(F^\sharp)$ be the abstract fixpoint. Then for every feature $j$ in the output of $\pi$:*

$$I(\mathcal{D}_{\text{te}}; \pi(\mathcal{D})_j) \leq \sigma^\sharp_j.b$$

*That is, the abstract bit-bound is a sound over-approximation of the true mutual information between the test set and each output feature.*

- **Status:** Novel theorem; proof by induction on pipeline depth, using the data-processing inequality at each stage.
- **Difficulty:** 2–3 person-months. The main challenge is handling operations that both read and write state (fit_transform) and ensuring the channel capacity bounds compose correctly.
- **What breaks without it:** The entire quantitative claim. Without soundness, the reported "bits of leakage" are meaningless.

**Theorem A2 (Compositionality of Channel Capacity Bounds).** *For a sequential pipeline $\pi = s_K \circ \cdots \circ s_1$, the end-to-end leakage satisfies:*

$$\lambda^{(j)}_{\text{end-to-end}} \leq \min_{1 \leq k \leq K} C(s_k^\sharp)$$

*where $C(s_k^\sharp)$ is the channel capacity of stage $k$ viewed as a channel from test inputs to outputs. Moreover, for parallel branches that join:*

$$\lambda^{(j)}_{\text{join}} \leq \sum_{b \in \text{branches}} \lambda^{(j)}_b$$

- **Status:** Follows from the data-processing inequality (DPI) for sequential composition. The parallel bound is a direct application of the chain rule for mutual information. The novelty is in applying these to the specific abstract domain.
- **Difficulty:** 1 person-month (proof is straightforward given DPI; the work is in formalizing the pipeline DAG structure).

**Theorem A3 (Termination and Complexity).** *For an acyclic pipeline DAG with $K$ stages, $d$ features, and bit-bound precision $B_{\max}$, the abstract fixpoint computation terminates in $O(K \cdot d)$ abstract transformer applications, with each application costing $O(d)$, yielding total complexity $O(K \cdot d^2)$.*

- **Status:** Standard result from abstract interpretation theory, adapted to our domain. The finite lattice height is $|\mathcal{P}(\mathcal{O})| \times B_{\max} = 8 \cdot B_{\max}$.
- **Difficulty:** 0.5 person-months.

**Theorem A4 (Tightness for Linear Pipelines).** *For pipelines consisting only of linear operations (scaling, centering, linear projection), the abstract bit-bound is tight up to a factor of $\log(d)$:*

$$\frac{1}{\log d} \cdot \sigma^\sharp_j.b \leq I(\mathcal{D}_{\text{te}}; \pi(\mathcal{D})_j) \leq \sigma^\sharp_j.b$$

- **Status:** Novel; requires showing that the Gaussian channel capacity bound is approximately tight for linear statistics.
- **Difficulty:** 2 person-months.

#### 1.A.4 Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Soundness** | Strong: formal over-approximation guarantee |
| **Precision** | Moderate: may over-approximate significantly for nonlinear operations |
| **Novelty** | High: first abstract domain for DataFrame-level information flow |
| **Reviewer accessibility** | Moderate: abstract interpretation is well-known but less familiar to ML audience |
| **Implementation** | Natural: abstract transformers map directly to code |

---

### 1.B — Framing B: Direct Information-Theoretic Analysis

#### 1.B.1 Core Idea

Directly estimate the mutual information $I(\mathcal{D}_{\text{te}}; X_j^{\text{out}})$ for each output feature $X_j^{\text{out}}$ by modeling the pipeline as a Markov chain and decomposing the information flow along the DAG using the chain rule. Use non-parametric MI estimators with finite-sample correction to produce confidence intervals on leakage.

#### 1.B.2 Mathematical Objects

**Object B1: Pipeline Information DAG.**

Model the pipeline as a Bayesian network $\mathcal{G} = (V, E)$ where:
- Source nodes: $\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}}$ (observed random variables)
- Intermediate nodes: fitted parameters $\theta_k$ and intermediate DataFrames $Z_k$
- Sink nodes: output features $X_j^{\text{out}}$

Each edge represents a functional dependency. The joint distribution factorizes as:

$$P(\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}}, \theta_1, Z_1, \dots, X^{\text{out}}) = P(\mathcal{D}_{\text{tr}}) P(\mathcal{D}_{\text{te}}) \prod_{k=1}^K P(\theta_k, Z_k \mid \text{pa}(k))$$

where $\text{pa}(k)$ denotes the parents of node $k$ in $\mathcal{G}$.

**Object B2: Leakage Decomposition via Chain Rule.**

The total leakage into output feature $j$ decomposes along any path from $\mathcal{D}_{\text{te}}$ to $X_j^{\text{out}}$:

$$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) = \sum_{k \in \text{path}} I(\mathcal{D}_{\text{te}}; Z_k \mid Z_{k-1})$$

by the chain rule for mutual information along a Markov chain. More precisely, applying the data-processing inequality along each path and then taking the maximum over paths:

$$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq \min_{\text{cut } S} \sum_{e \in S} I_e$$

where the minimum is over all cuts $S$ separating $\mathcal{D}_{\text{te}}$ from $X_j^{\text{out}}$ in $\mathcal{G}$, and $I_e$ is the MI along edge $e$. This is the *information-flow analog of max-flow/min-cut*.

**Object B3: Per-Operation MI Estimators.**

For each operation type, we define a specialized MI estimator:

*Aggregation operations (mean, std, sum, count):*

Let $\phi: \mathbb{R}^n \to \mathbb{R}$ be a sufficient statistic. Under regularity conditions, the MI between a subset $S \subset \{1, \dots, n\}$ of inputs and $\phi(x_1, \dots, x_n)$ satisfies:

$$I(X_S; \phi(X)) = h(\phi(X)) - h(\phi(X) \mid X_S)$$

For the sample mean with $|S| = n_{\text{te}}$ test rows among $n$ total:

$$I(X_S; \bar{X}) = \frac{1}{2}\log\!\Bigl(1 + \frac{n_{\text{te}}}{n - n_{\text{te}}} \cdot \frac{\text{Var}(X_S)}{\text{Var}(X \setminus X_S) / (n - n_{\text{te}})}\Bigr)$$

Under the assumption $X_i \sim_{\text{iid}} \mathcal{N}(\mu, \sigma^2)$:

$$I(X_S; \bar{X}) = \frac{1}{2}\log\!\Bigl(1 + \frac{n_{\text{te}}}{n - n_{\text{te}}}\Bigr) \quad \text{(nats)}$$

Converting to bits: $\lambda = \frac{1}{2\ln 2}\ln\!\bigl(1 + \frac{n_{\text{te}}}{n - n_{\text{te}}}\bigr)$.

*Merge/Join operations:*

A join on key $K$ transmits information about row membership. The leakage through a join is bounded by the entropy of the key intersection:

$$I(\mathcal{D}_{\text{te}}; \text{join result}) \leq H(\mathbb{1}[K_i \in K_{\text{te}}]) \leq n \cdot h\bigl(\frac{n_{\text{te}}}{n}\bigr)$$

where $h(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy function.

*Dimensionality reduction (PCA, feature selection):*

PCA computes the top eigenvectors of the covariance matrix. The leakage through each principal component is:

$$I(\mathcal{D}_{\text{te}}; v_i) \leq \sum_{j=1}^d I(\mathcal{D}_{\text{te}}; \Sigma_{jk}) \leq d^2 \cdot C_{\text{cov-entry}}(n_{\text{te}}, n)$$

where $C_{\text{cov-entry}}$ is the channel capacity for a single covariance matrix entry.

**Object B4: Non-Parametric MI Estimation with Error Bounds.**

For operations where closed-form MI is unavailable, we use the Kraskov-Stögbauer-Grassberger (KSG) estimator with finite-sample bias correction:

$$\hat{I}_{\text{KSG}}(X; Y) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(N)$$

where $\psi$ is the digamma function and $n_x, n_y$ are neighbor counts. The estimation error satisfies:

$$|\hat{I}_{\text{KSG}} - I(X; Y)| \leq O\!\Bigl(\frac{1}{\sqrt{N}} + \frac{k}{N}\Bigr) \quad \text{w.h.p.}$$

For high-dimensional intermediates, we apply the *local non-uniformity* (LNN) correction of Gao et al. (2015).

**Object B5: Combinatorial Explosion Management.**

The number of paths in the pipeline DAG can be exponential in $K$. To manage this:

1. **d-separation pruning:** Eliminate all variables $Z$ such that $\mathcal{D}_{\text{te}} \perp\!\!\!\perp Z \mid \text{observed}$ in $\mathcal{G}$. This is computed in $O(|V| + |E|)$ time.

2. **Modular decomposition:** Factor the DAG into *modules* — maximal subgraphs where information enters through a single interface. Compute MI once per module.

3. **Budget-bounded estimation:** Allocate a computational budget $B$ (in seconds) and use an anytime algorithm that refines MI estimates for the highest-leakage paths first, guided by the abstract interpretation upper bounds from Framing A.

#### 1.B.3 Key Theorems

**Theorem B1 (Information-Theoretic Min-Cut for Pipeline Leakage).** *For a pipeline DAG $\mathcal{G}$ with source $\mathcal{D}_{\text{te}}$ and sink $X_j^{\text{out}}$, the leakage satisfies:*

$$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq \min_{\text{cut } S} \sum_{(u,v) \in S} I(Z_u; Z_v)$$

*Moreover, this bound is tight when all operations are deterministic (which they are in ML pipelines).*

- **Status:** Adaptation of known information-theoretic results (Cover & Thomas, Ch. 15) to the pipeline DAG setting. The tightness result for deterministic functions is novel for this context.
- **Difficulty:** 1 person-month.
- **What breaks without it:** Cannot decompose end-to-end leakage into per-stage contributions.

**Theorem B2 (Closed-Form Leakage for Linear-Gaussian Pipelines).** *For a pipeline where all operations are linear and data is jointly Gaussian, the leakage into feature $j$ is:*

$$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) = \frac{1}{2}\log\det\!\Bigl(I + \frac{n_{\text{te}}}{n_{\text{tr}}} \cdot \Sigma_{\text{te}|j} \Sigma_{\text{tr}|j}^{-1}\Bigr)$$

*where $\Sigma_{\text{te}|j}, \Sigma_{\text{tr}|j}$ are the conditional covariance matrices of test and train contributions to feature $j$.*

- **Status:** Novel closed-form result. Proof uses the Gaussian MI formula and traces the linear algebra through pipeline stages.
- **Difficulty:** 1.5 person-months.

**Theorem B3 (Finite-Sample Concentration for Leakage Estimates).** *For the KSG-based leakage estimator $\hat{\lambda}$ with $N$ samples and $k$ neighbors:*

$$P\bigl(|\hat{\lambda} - \lambda| > \epsilon\bigr) \leq 2\exp\!\Bigl(-\frac{N\epsilon^2}{C_d \cdot k^{2/d}}\Bigr)$$

*where $d$ is the ambient dimension and $C_d$ is a constant depending only on $d$.*

- **Status:** Extension of existing KSG concentration results (Gao et al., 2015; Berrett et al., 2019) to the pipeline leakage setting where samples are not independent (due to shared fitted parameters).
- **Difficulty:** 2 person-months. The non-independence introduces technical complications.

**Theorem B4 (Anytime Refinement Convergence).** *The budget-bounded estimation algorithm, using abstract interpretation bounds as initial estimates, converges to the true MI vector at rate $O(1/\sqrt{t})$ where $t$ is wall-clock time, with the highest-leakage features converging first.*

- **Status:** Novel algorithm and convergence result.
- **Difficulty:** 1.5 person-months.

#### 1.B.4 Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Soundness** | Weaker than A: provides confidence intervals, not hard bounds |
| **Precision** | Strong: directly estimates MI rather than bounding it |
| **Novelty** | Moderate: applies known MI estimation to new domain |
| **Reviewer accessibility** | High: information theory is universal in ML |
| **Implementation** | Harder: requires sampling / execution to estimate MI |

**Critical weakness:** This framing requires *executing* the pipeline (or at least simulating it) to estimate MI, which conflicts with the static analysis design goal. It also requires actual data, whereas Framing A works from code alone.

---

### 1.C — Framing C: Hybrid Abstract-Interpretation / Information-Theoretic Approach

#### 1.C.1 Core Idea

Use abstract interpretation (Framing A) for *structural* analysis — determining which operations create information flows from test to train — and information theory (Framing B) for *quantification* — measuring how many bits flow through each identified channel. The abstract interpretation provides soundness and works statically; the information-theoretic layer provides precision and works with concrete data.

**The interface** is a set of *leakage sites*: locations in the pipeline DAG where test information enters a computation that influences training. The abstract interpretation identifies and bounds these sites; the information-theoretic layer tightens the bounds.

#### 1.C.2 Mathematical Objects

**Object C1: Leakage Site Descriptor.**

A leakage site is a tuple:

$$\ell = (k, j, \tau^\sharp, \mathcal{C})$$

where:
- $k$ is the pipeline stage index
- $j$ is the feature index (or set of feature indices)
- $\tau^\sharp = (O, b) \in \mathcal{T}$ is the abstract taint label (from Framing A)
- $\mathcal{C}: \mathbb{R}^{n_{\text{in}}} \to \mathbb{R}^{n_{\text{out}}}$ is the concrete function implementing this stage (for MI estimation)

**Object C2: The Two-Phase Analysis.**

*Phase 1 (Static, Abstract):*
Run the abstract interpretation from Framing A to compute $\sigma^\sharp$. Extract the set of leakage sites:

$$\mathcal{L} = \{(k, j) : \sigma^\sharp_{k,j}.\text{origins} \ni \text{te} \;\land\; \sigma^\sharp_{k,j}.b > 0\}$$

This runs in $O(K \cdot d^2)$ time and requires no data.

*Phase 2 (Dynamic, Information-Theoretic):*
For each leakage site $\ell \in \mathcal{L}$, estimate the true MI using Framing B techniques:

$$\hat{\lambda}_\ell = \hat{I}_{\text{KSG}}(\mathcal{D}_{\text{te}}; Z_\ell)$$

constrained by $\hat{\lambda}_\ell \leq \sigma^\sharp_\ell.b$ (the abstract bound is an upper bound, so we clamp).

*Phase 3 (Attribution):*
Decompose the per-feature leakage across stages using the DAG structure:

$$\hat{\lambda}^{(j)} = \sum_{\ell \in \mathcal{L}: \ell.j = j} \hat{\lambda}_\ell - \text{double-counting correction}$$

The double-counting correction uses the inclusion-exclusion principle over shared parents in the DAG.

**Object C3: The Reduced Product Domain.**

The hybrid analysis can be formalized as a *reduced product* of two abstract domains:

$$\mathcal{A}_{\text{hybrid}} = \mathcal{A}_{\text{taint}} \otimes \mathcal{A}_{\text{MI}}$$

where:
- $\mathcal{A}_{\text{taint}}$ is the partition-taint lattice from Framing A (always available)
- $\mathcal{A}_{\text{MI}}$ is an optional MI-estimate domain (available when data is present)

The *reduction operator* $\rho: \mathcal{A}_{\text{taint}} \times \mathcal{A}_{\text{MI}} \to \mathcal{A}_{\text{taint}} \times \mathcal{A}_{\text{MI}}$ tightens both components using the other:

$$\rho(\tau, \hat{I}) = \bigl(\tau[b \mapsto \min(b, \hat{I})], \; \min(\hat{I}, \tau.b)\bigr)$$

**Object C4: Confidence-Aware Leakage Report.**

The output of the hybrid analysis is a *leakage spectrum*:

$$\Lambda = \bigl\{(j, [\hat{\lambda}_j^{\text{lo}}, \hat{\lambda}_j^{\text{hi}}], \text{path}_j)\bigr\}_{j=1}^d$$

where:
- $\hat{\lambda}_j^{\text{hi}} = \sigma^\sharp_j.b$ (abstract upper bound, always available)
- $\hat{\lambda}_j^{\text{lo}} = \hat{I}_{\text{KSG}, j} - \epsilon_j$ (MI estimate minus confidence interval, available with data)
- $\text{path}_j$ is the highest-leakage path from $\mathcal{D}_{\text{te}}$ to feature $j$

**Object C5: Sensitivity Typing for Transfer Functions.**

To bridge the two phases, we introduce *sensitivity types* for pipeline operations. Each operation $s_k$ is annotated with a sensitivity signature:

$$s_k : \mathcal{D}[\tau_1, \dots, \tau_d] \xrightarrow{\delta_k} \mathcal{D}[\tau_1', \dots, \tau_{d'}']$$

where $\delta_k: [0, \infty]^d \to [0, \infty]^{d'}$ maps input bit-bounds to output bit-bounds. The sensitivity function $\delta_k$ satisfies:

1. **Monotonicity:** $b \leq b' \implies \delta_k(b) \leq \delta_k(b')$ (more input leakage → more output leakage)
2. **Sub-additivity:** $\delta_k(b + b') \leq \delta_k(b) + \delta_k(b')$ (leakage from independent sources adds at most linearly)
3. **DPI consistency:** $\|\delta_k(b)\|_\infty \leq \|b\|_\infty$ (no stage amplifies total leakage)

These are the *information-theoretic analogs* of Lipschitz conditions in differential privacy.

#### 1.C.3 Key Theorems

**Theorem C1 (Soundness of Hybrid Analysis).** *The hybrid analysis is sound: for all features $j$,*

$$I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq \hat{\lambda}_j^{\text{hi}}$$

*When Phase 2 data is available, the interval $[\hat{\lambda}_j^{\text{lo}}, \hat{\lambda}_j^{\text{hi}}]$ contains the true leakage with probability $\geq 1 - \alpha$.*

- **Status:** Follows from Theorem A1 (soundness of abstract interpretation) and Theorem B3 (concentration of MI estimator). The combination is novel.
- **Difficulty:** 1 person-month.

**Theorem C2 (Precision Improvement via Reduction).** *The reduced product domain $\mathcal{A}_{\text{hybrid}}$ is strictly more precise than either component alone:*

$$\gamma_{\text{hybrid}}(\rho(\tau, \hat{I})) \subsetneq \gamma_{\text{taint}}(\tau) \cap \gamma_{\text{MI}}(\hat{I})$$

*for non-trivial pipelines (those with at least one leakage site where the abstract bound is not tight).*

- **Status:** Novel; proof by construction of a pipeline where neither domain alone achieves the reduced product's precision.
- **Difficulty:** 1 person-month.

**Theorem C3 (Optimal Budget Allocation for Phase 2).** *Given a computational budget $B$ (in samples) for Phase 2 MI estimation, the allocation that minimizes the maximum interval width $\max_j (\hat{\lambda}_j^{\text{hi}} - \hat{\lambda}_j^{\text{lo}})$ is:*

$$n_\ell^\star \propto \sqrt{\sigma^\sharp_\ell.b \cdot d_\ell}$$

*where $d_\ell$ is the dimensionality of leakage site $\ell$. This allocates more samples to high-leakage, high-dimensional sites.*

- **Status:** Novel optimization result; proof via Lagrange multipliers on the KSG convergence rate.
- **Difficulty:** 1 person-month.

**Theorem C4 (Sensitivity Type Soundness).** *If every transfer function $s_k^\sharp$ satisfies its declared sensitivity signature $\delta_k$, then the composed pipeline satisfies:*

$$\forall j: \; I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq (\delta_K \circ \cdots \circ \delta_1)(\lambda^{\text{input}})_j$$

*where $\lambda^{\text{input}}_j = H(X_j^{\text{te}})$ for mixed-provenance columns and $0$ otherwise.*

- **Status:** Novel formulation; proof follows from DPI applied inductively through the sensitivity types.
- **Difficulty:** 1.5 person-months. The main challenge is defining $\delta_k$ for all ~200 operations.

**Theorem C5 (Graceful Degradation).** *When Phase 2 estimation fails for some leakage sites (e.g., due to high dimensionality or insufficient samples), the analysis degrades gracefully to Phase 1 bounds for those sites. The overall soundness guarantee (Theorem C1) is preserved.*

- **Status:** Follows directly from the lattice structure.
- **Difficulty:** 0.5 person-months.

#### 1.C.4 Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Soundness** | Strong: inherits from Framing A |
| **Precision** | Strong: inherits from Framing B |
| **Novelty** | Highest: the interface between AI and IT is the core contribution |
| **Reviewer accessibility** | High: each component is individually familiar |
| **Implementation** | Most complex but most modular |

---

## 2. Comparative Analysis of Framings

### 2.1 Core Mathematical Objects Summary

| Object | Framing A | Framing B | Framing C |
|--------|-----------|-----------|-----------|
| Primary structure | Lattice $\mathcal{T}$ | Bayesian network $\mathcal{G}$ | Reduced product $\mathcal{A}_{\text{hybrid}}$ |
| Quantification | Channel capacity bounds | MI estimation | Both, with reduction |
| Composition | Abstract transformer composition | Chain rule / min-cut | Sensitivity types |
| Data requirement | None (static) | Full data required | None required; data improves precision |
| Output | Hard upper bound per feature | Point estimate ± CI | Interval $[\text{lo}, \text{hi}]$ per feature |
| Computational cost | $O(K \cdot d^2)$ | $O(K \cdot d \cdot N)$ per sample | $O(K \cdot d^2) + O(|\mathcal{L}| \cdot N)$ |

### 2.2 Theorem Inventory

| Theorem | Framing | Novel? | Difficulty | Load-bearing? |
|---------|---------|--------|------------|---------------|
| A1: Soundness of taint abstraction | A | Yes | 2–3 mo | Critical |
| A2: Compositionality of channel capacity | A | Partial | 1 mo | Critical |
| A3: Termination and complexity | A | No | 0.5 mo | Necessary |
| A4: Tightness for linear pipelines | A | Yes | 2 mo | Important |
| B1: Information-theoretic min-cut | B | Partial | 1 mo | Critical |
| B2: Closed-form Gaussian leakage | B | Yes | 1.5 mo | Important |
| B3: Finite-sample concentration | B | Partial | 2 mo | Critical |
| B4: Anytime convergence | B | Yes | 1.5 mo | Nice-to-have |
| C1: Hybrid soundness | C | Yes | 1 mo | Critical |
| C2: Precision improvement | C | Yes | 1 mo | Important |
| C3: Optimal budget allocation | C | Yes | 1 mo | Nice-to-have |
| C4: Sensitivity type soundness | C | Yes | 1.5 mo | Critical |
| C5: Graceful degradation | C | No | 0.5 mo | Necessary |

### 2.3 Risk Assessment

| Risk | Framing A | Framing B | Framing C |
|------|-----------|-----------|-----------|
| Bounds too loose to be useful | **High** (channel capacity is worst-case) | Low | **Medium** (Phase 1 bounds may dominate) |
| Requires execution / data | None | **High** (requires full execution) | Low (data is optional) |
| Scope of transfer function engineering | **High** (~200 operations) | Medium (~50 closed forms + KSG fallback) | **Highest** (both abstract + sensitivity) |
| Theoretical depth insufficient for best paper | Medium | **High** (mostly known techniques) | Low |
| Reviewers find it too PL-flavored | **Medium** | Low | Medium |

---

## 3. Recommendation

### 3.1 Verdict: Framing C (Hybrid) — with Framing A as the foundation

**Framing C is the strongest choice for a best-paper-caliber contribution**, for the following reasons:

1. **Novel intellectual contribution at the interface.** The reduced product of abstract interpretation and information theory, connected by sensitivity types, is genuinely new. Neither the PL nor the ML community has formalized this interface for data science pipelines. This is the kind of "bridge" contribution that best-paper committees reward.

2. **Graceful degradation preserves practical impact.** The system works *without data* (Phase 1 only, pure static analysis) and *with data* (Phase 1 + Phase 2, tightened intervals). This means the tool is useful in CI/CD (no data available) and in notebooks (data available). This dual-mode design is itself a contribution.

3. **The math is deep but accessible.** Each individual component (abstract interpretation, mutual information, channel capacity, sensitivity analysis) is well-known. The novelty is in their *composition* and the *specific instantiation* for ML pipeline operations. This makes the paper readable by reviewers from either the PL or ML tradition.

4. **The theorems tell a coherent story.** Soundness (C1) → Precision improvement (C2) → Sensitivity types as the compositional backbone (C4) → Graceful degradation (C5). This is a clean narrative arc.

5. **It subsumes the others.** If a reviewer prefers pure static analysis, the paper delivers Framing A. If a reviewer prefers empirical MI estimation, the paper delivers Framing B. Framing C is the unification.

### 3.2 Specific Best-Paper Argument

The **load-bearing mathematical novelty** is:

> *We define the first quantitative information-flow abstract domain for statistical operations on tabular data, equip it with sensitivity types that enable compositional reasoning, and prove that the resulting analysis is sound (every reported bound is a valid upper bound on mutual information) and that the reduced product with empirical MI estimation is strictly more precise than either component alone.*

This is a single, clean sentence that a program committee can evaluate. It is:
- **Falsifiable:** The soundness theorem either holds or it doesn't.
- **Quantitative:** The precision improvement theorem makes a measurable claim.
- **Novel:** No prior work has formalized information-flow abstract domains for `pandas`/`sklearn` operations.
- **Significant:** Train-test leakage is a recognized problem affecting real ML systems.

### 3.3 Suggested Theorem Budget (14 months, 1.5 FTE)

| Priority | Theorems | Person-months |
|----------|----------|---------------|
| **Must-have** | A1 (soundness), C1 (hybrid soundness), C4 (sensitivity types) | 5 |
| **Should-have** | A2 (compositionality), B1 (min-cut), C2 (precision improvement) | 3 |
| **Nice-to-have** | A4 (tightness), B2 (Gaussian closed form), C3 (budget allocation) | 4.5 |
| **Infrastructure** | A3 (termination), C5 (degradation) | 1 |
| **Total** | | **13.5 person-months** |

---

## 4. Enumeration of Mathematical Contributions

### M1: Partition-Taint Lattice for DataFrames

- **Statement:** Define a complete lattice $\mathcal{T} = \mathcal{P}(\mathcal{O}) \times [0, B_{\max}]$ with a concretization that maps each abstract element to the set of DataFrames whose test-set mutual information is bounded by $b$. Prove this forms a Galois connection with the concrete domain of (DataFrame, partition) pairs.
- **Why load-bearing:** This is the foundational mathematical object. Without it, there is no abstract domain, no transfer functions, no fixpoint computation, no soundness theorem. The entire analysis framework collapses.
- **Novelty:** Truly new. Abstract interpretation has been applied to numerical programs (Astrée), pointer analysis (Steensgaard/Andersen), and neural networks (DeepPoly), but never to DataFrame-level information flow with quantitative bit-bounds. The combination of *set-valued origin tracking* with *quantitative bit-bounds* in a single lattice element is novel.

### M2: Channel Capacity Bounds for Statistical Aggregates

- **Statement:** For each common statistical aggregate $\phi \in \{\text{mean}, \text{std}, \text{var}, \text{median}, \text{quantile}_p, \text{sum}, \text{count}, \text{min}, \text{max}\}$, derive a closed-form upper bound on the channel capacity $C_\phi(n_{\text{te}}, n, d)$ when $\phi$ is computed over a mixture of $n_{\text{te}}$ test rows and $n - n_{\text{te}}$ train rows. Prove the bounds are tight for Gaussian data and within a factor of $O(\log n)$ for sub-Gaussian data.
- **Why load-bearing:** These bounds are the "fuel" for the abstract transformers. Without them, the bit-bound $b$ in the lattice would always be $\infty$ (trivially sound but useless). The precision of the entire analysis depends on the tightness of these per-operation bounds.
- **Novelty:** Adaptation of known information-theoretic techniques (channel capacity, sufficient statistics) to the specific setting of ML pipeline operations. The individual bounds follow from textbook information theory, but the *systematic derivation for all sklearn-relevant operations* and the *sub-Gaussian tightness result* are new. Estimated 60% novel, 40% textbook.

### M3: Soundness of Abstract Pipeline Analysis

- **Statement (Theorem A1 restated):** For any pipeline $\pi$ and dataset $\mathcal{D}$ with partition $(\mathcal{D}_{\text{tr}}, \mathcal{D}_{\text{te}})$, the abstract fixpoint $\sigma^\sharp$ computed by the worklist algorithm satisfies $I(\mathcal{D}_{\text{te}}; \pi(\mathcal{D})_j) \leq \sigma^\sharp_j.b$ for all output features $j$.
- **Why load-bearing:** This is the central theorem. It transforms the tool from "a heuristic that guesses leakage" into "a verified analysis that certifies leakage bounds." Reviewers will judge the paper primarily on whether this theorem is correct and non-trivial.
- **Novelty:** The *proof technique* is novel — it combines induction on the pipeline DAG with the data-processing inequality applied through abstract transformers. The closest precedent is the soundness proof for quantitative information-flow type systems (Smith 2009, Alvim et al. 2012), but those operate on imperative programs, not DataFrame pipelines with statistical operations.

### M4: Sensitivity Types for Pipeline Operations

- **Statement:** Define a sensitivity type system where each operation $s_k$ is annotated with a function $\delta_k: [0,\infty]^{d_{\text{in}}} \to [0,\infty]^{d_{\text{out}}}$ satisfying monotonicity, sub-additivity, and DPI consistency. Prove that sensitivity types compose: $\delta_\pi = \delta_K \circ \cdots \circ \delta_1$ is a valid sensitivity type for the composed pipeline $\pi$.
- **Why load-bearing:** Sensitivity types are the compositional backbone. Without them, the analysis would need to reason about the entire pipeline monolithically, which doesn't scale. Sensitivity types enable modular verification: prove each operation correct once, compose freely.
- **Novelty:** Truly new. Sensitivity analysis exists in differential privacy ($\epsilon$-DP composition) and in numerical analysis (condition numbers), but the formulation as a *type system for information-flow in statistical operations* with the specific DPI-consistency requirement is novel. This is the contribution most likely to inspire follow-up work.

### M5: Reduced Product of Abstract and Empirical Domains

- **Statement (Theorem C2 restated):** Define the reduced product $\mathcal{A}_{\text{hybrid}} = \mathcal{A}_{\text{taint}} \otimes \mathcal{A}_{\text{MI}}$ with reduction operator $\rho$. Prove that $\gamma_{\text{hybrid}}(\rho(\tau, \hat{I})) \subsetneq \gamma_{\text{taint}}(\tau)$ for any pipeline with at least one leakage site where the abstract bound is not tight.
- **Why load-bearing:** This theorem justifies the hybrid design. Without it, one could argue that the two phases are independent and the combination adds no value. The strict precision improvement demonstrates that the combination is synergistic, not just a union.
- **Novelty:** Reduced products are a standard technique in abstract interpretation (Cousot & Cousot 1979), but the specific combination of a *taint lattice* with an *MI-estimate domain* is new. The novelty is in the reduction operator and the proof that it yields strict improvement. Estimated 70% novel.

### M6: Information-Theoretic Min-Cut for Pipeline DAGs

- **Statement (Theorem B1 restated):** The leakage from $\mathcal{D}_{\text{te}}$ to output feature $j$ is bounded by the minimum information-theoretic cut in the pipeline DAG: $I(\mathcal{D}_{\text{te}}; X_j^{\text{out}}) \leq \min_{\text{cut}} \sum_{e \in \text{cut}} I_e$. This bound is tight for deterministic pipelines.
- **Why load-bearing:** This theorem provides the structural decomposition that makes per-stage attribution possible. Without it, we can only report total leakage, not *where* in the pipeline it occurs.
- **Novelty:** The information-theoretic max-flow/min-cut theorem is known (Cover & Thomas). Applying it to ML pipeline DAGs and proving tightness for deterministic operations is a modest but useful contribution. Estimated 30% novel.

### M7: Closed-Form Leakage for the Linear-Gaussian Special Case

- **Statement (Theorem B2 restated):** When all pipeline operations are linear and data is jointly Gaussian, derive an exact closed-form expression for $I(\mathcal{D}_{\text{te}}; X_j^{\text{out}})$ in terms of the pipeline's Jacobian and the data covariance structure.
- **Why load-bearing:** Serves as a validation oracle — the abstract analysis can be checked against the exact answer for linear pipelines. Also provides intuition: in the linear case, leakage scales as $\Theta(n_{\text{te}} / n_{\text{tr}})$ in bits, which is a clean and memorable result.
- **Novelty:** Moderate. The calculation follows from known Gaussian MI formulas, but the expression in terms of pipeline Jacobians and the resulting scaling law are new insights. Estimated 40% novel.

### M8: Convergence and Complexity of the Fixpoint Algorithm

- **Statement (Theorem A3 restated):** The worklist algorithm for computing $\sigma^\sharp$ over the product lattice $\prod_{v \in V} \mathcal{A}_v$ terminates in $O(K \cdot d \cdot |\mathcal{P}(\mathcal{O})| \cdot B_{\max})$ iterations, with each iteration costing $O(d)$ abstract transformer evaluations.
- **Why load-bearing:** Ensures the tool runs in polynomial time on laptop CPUs, as required by the constraints.
- **Novelty:** Low. Standard abstract interpretation complexity analysis applied to a new domain. Estimated 10% novel.

### M9: Finite-Sample Validity of Empirical Leakage Estimates

- **Statement (Theorem B3 restated):** Concentration inequality for the KSG MI estimator in the pipeline setting, accounting for non-independence of samples due to shared fitted parameters.
- **Why load-bearing:** Without this, the empirical estimates in Phase 2 have no statistical validity. The confidence intervals could be arbitrarily wrong.
- **Novelty:** The non-independence correction is new. Standard KSG theory assumes i.i.d. samples; in a pipeline, the fitted parameters create dependencies. Estimated 50% novel.

### M10: Leakage Decomposition and Attribution

- **Statement:** For a pipeline with multiple leakage sites $\ell_1, \dots, \ell_m$ contributing to the same output feature $j$, decompose the total leakage using Shapley values over the leakage sites:

$$\phi_\ell = \sum_{S \subseteq \mathcal{L} \setminus \{\ell\}} \frac{|S|!(m-|S|-1)!}{m!}\bigl[\lambda(S \cup \{\ell\}) - \lambda(S)\bigr]$$

Prove that the Shapley attribution is the unique attribution satisfying efficiency ($\sum_\ell \phi_\ell = \lambda^{(j)}$), symmetry, and null-player properties.
- **Why load-bearing:** Enables actionable diagnostics. Without attribution, the tool reports "feature $j$ leaks 3.2 bits" but cannot say "2.1 bits come from the scaler fitted on mixed data and 1.1 bits come from the target encoder." This is the difference between a research prototype and a useful tool.
- **Novelty:** Shapley values for attribution are well-known (SHAP); applying them to information-flow decomposition across pipeline stages is new. Estimated 40% novel.

---

## 5. Dependency Graph of Mathematical Contributions

```
M1 (Lattice)
 ├── M2 (Channel capacity bounds)
 │    └── M3 (Soundness) ← CENTRAL THEOREM
 │         ├── M4 (Sensitivity types) ← MOST NOVEL
 │         │    └── M8 (Complexity)
 │         └── M5 (Reduced product)
 │              ├── M6 (Min-cut)
 │              │    └── M10 (Attribution)
 │              └── M9 (Finite-sample)
 └── M7 (Linear-Gaussian closed form) [validation oracle]
```

**Critical path:** M1 → M2 → M3 → M4 (must be done sequentially; ~7 person-months).

**Parallelizable:** M6, M7, M9, M10 can proceed concurrently once M1 is established.

---

## 6. Summary Table

| ID | Contribution | Type | Novel? | Difficulty | Load-bearing? |
|----|-------------|------|--------|------------|---------------|
| M1 | Partition-taint lattice for DataFrames | Definition | ★★★ New | 1.5 mo | **Critical** |
| M2 | Channel capacity for statistical aggregates | Theorem collection | ★★☆ Adaptation | 2 mo | **Critical** |
| M3 | Soundness of abstract pipeline analysis | Theorem | ★★★ New proof | 2.5 mo | **Critical** |
| M4 | Sensitivity types for pipeline operations | Type system + theorem | ★★★ New | 2 mo | **Critical** |
| M5 | Reduced product of abstract + empirical | Theorem | ★★☆ New combination | 1 mo | Important |
| M6 | Information-theoretic min-cut for DAGs | Theorem | ★☆☆ Adaptation | 1 mo | Important |
| M7 | Linear-Gaussian closed form | Theorem | ★★☆ New calculation | 1.5 mo | Validation |
| M8 | Fixpoint convergence and complexity | Theorem | ★☆☆ Standard | 0.5 mo | Necessary |
| M9 | Finite-sample concentration for MI | Theorem | ★★☆ Extension | 2 mo | Important |
| M10 | Shapley-based leakage attribution | Definition + theorem | ★★☆ New application | 1.5 mo | Important |
| | **Total** | | | **15.5 mo** | |

---

## 7. What a Best Paper Needs (and How This Math Delivers)

### 7.1 Checklist

| Best-paper criterion | How this math delivers |
|---------------------|----------------------|
| **Single clean novelty** | M4 (sensitivity types for information flow in statistical operations) — this is the "one new idea" |
| **Theoretical depth** | M3 (soundness) + M4 (compositionality) — non-trivial proofs with real content |
| **Practical significance** | M2 (concrete bounds) + M10 (attribution) — the math directly produces useful numbers |
| **Elegance** | The lattice $\mathcal{T}$ is simple; the sensitivity types are a natural formulation; the reduced product is a clean design pattern |
| **Validation path** | M7 (exact solution for linear case) provides ground truth; M9 (concentration) provides confidence intervals |
| **Breadth of impact** | The sensitivity type framework (M4) generalizes beyond leakage to any information-flow question about data pipelines |

### 7.2 The One-Paragraph Pitch

This work defines the first formal framework for *quantifying* train-test leakage in ML pipelines, measured in bits of mutual information. We model pipeline operations as channels with bounded capacity (M2), compose them via a novel sensitivity type system (M4) over a purpose-built abstract domain for tabular data flows (M1), and prove the resulting analysis is sound (M3): every reported leakage bound is a valid upper bound on the true mutual information between test data and training features. When concrete data is available, we tighten these bounds via a reduced product with empirical MI estimation (M5), yielding confidence intervals rather than just upper bounds. The framework identifies not just *how much* leakage exists, but *where* it originates and *how* it propagates, enabling actionable repair guidance.
