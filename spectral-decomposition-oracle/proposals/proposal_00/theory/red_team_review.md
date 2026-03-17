# Red-Team Review: Spectral Decomposition Oracle

**Reviewer Role:** Adversarial Red Team — find every weakness before hostile reviewers do
**Target:** "Spectral Features for MIP Decomposition Selection" (INFORMS JoC)
**Date:** 2025-07-22
**Severity Scale:** FATAL / SERIOUS / MODERATE / COSMETIC

---

## 1. Proof Attacks

### 1.1 Lemma L3 (Partition-to-Bound Bridge)

**ATTACK L3-A: "Which dual bound?" ambiguity [SERIOUS]**

L3 claims: $z_{LP} - z_D \leq \sum_{e \in E_{\text{cross}}} |y^*_e| \cdot (n_e - 1)$

The statement conflates three different objects that are called "the decomposed dual bound":
- The Lagrangian dual $z_{LR}(\lambda)$ at a specific multiplier $\lambda$
- The DW master LP bound $z_{DW}$ (which equals the Lagrangian dual at optimality but differs during column generation)
- The Benders master bound (which bounds from the opposite direction — it approaches the LP optimum from below via cuts)

For DW: $z_{LP} = z_{LR}^* = z_{DW}^*$ at convergence of the LP relaxation. So $z_{LP} - z_D = 0$ at convergence, and L3 becomes trivially true but vacuous. The lemma is only interesting *during* the solution process (partial convergence) or when comparing the LP of the monolithic formulation against the LP of the decomposed formulation — but these are equal for DW by LP duality.

**The actual nontrivial statement** must be about the *restricted* DW master (with finitely many columns) or the Lagrangian dual at a *suboptimal* multiplier. But then $y^*$ (the monolithic LP dual) is not available — you'd need it to evaluate the bound, which requires solving the problem you're trying to avoid solving.

**Fix required:** L3 must explicitly specify: (a) "decomposed dual bound" means the LP relaxation of the decomposed formulation with *approximate* duals/columns, not the converged value; or (b) the bound holds for the *first iteration* gap, with $y^*$ replaced by an accessible quantity. Otherwise, the bound is either trivially zero or requires an oracle for $y^*$.

**ATTACK L3-B: Hyperedge weight definition is circular [MODERATE]**

The bound weights crossing hyperedges by $|y^*_e|$ — the monolithic LP dual values. To compute $|y^*_e|$, you must solve the monolithic LP. But the whole point of decomposition is to *avoid* solving the monolithic LP (or at least to solve something easier). If you have $y^*$, you already have $z_{LP}$ and $z_D$ is computable — you don't need L3.

This makes L3 a *retrospective* quality metric ("given the LP solution, how good was this partition?") rather than a *prospective* guide ("before decomposing, which partition should I use?"). The paper's framing implies the latter. The distinction must be stated clearly.

**Counter-argument the authors could make:** L3 can be evaluated using approximate duals from a few simplex iterations. **Red-team rebuttal:** Then the bound holds for $\tilde{y}$ not $y^*$, and you need an additional term bounding $|y^* - \tilde{y}|$, which brings back the conditioning issues.

**ATTACK L3-C: The "well-known" objection [MODERATE]**

The core of L3 — that coupling constraints inflate the dual gap by their dual multiplier magnitude — is a reformulation of the weak duality gap in Lagrangian relaxation. Specifically, for Lagrangian relaxation with coupling constraints dualized:

$$z_{LP} - z_{LR}(\lambda) = \sum_i \lambda_i \cdot (\text{violation of constraint } i \text{ by LR solution})$$

This is textbook material (Fisher 1981, Geoffrion 1974). L3 repackages it in hypergraph language with the $(n_e - 1)$ factor accounting for multi-block spanning. A hostile reviewer at MPC would write: "Lemma 3 is a straightforward consequence of weak duality applied to the Lagrangian relaxation. The $(n_e - 1)$ factor is a counting argument. The 3–5 day proof effort estimate suggests the authors consider this routine, which is appropriate for the level of the result."

**Severity depends on framing.** If L3 is claimed as a *novel theorem*, this is SERIOUS. If framed as a "useful formalization of a well-known principle, specialized to the hypergraph setting," it is COSMETIC. The current framing ("standalone value") leans toward overclaiming.

### 1.2 Lemma L3-sp (Spectral Partition Bound)

**ATTACK L3-sp-A: The $d_{\max}$ factor makes this vacuous on dense constraints [SERIOUS]**

L3-sp claims: crossing weight $\leq O(\delta^2 \cdot d_{\max} / \gamma_2^2)$

For MIPLIB instances with constraints involving all variables (e.g., budget constraints, knapsack covers), $d_{\max} = n$ where $n$ is the number of variables. For instances with $n = 10{,}000$ and a modest $\delta^2/\gamma_2^2 \approx 1$, the bound says crossing weight $\leq O(10{,}000)$ — typically larger than the total weight of all edges in the hypergraph. The bound becomes vacuous precisely on the instances where a universal constraint (budget, capacity) links all blocks — exactly the "bordered block-diagonal" structure that Benders decomposition is designed for.

**The irony:** L3-sp is tightest on instances with small $d_{\max}$ (pure block-diagonal with narrow coupling), which are exactly the instances where decomposition is obviously beneficial and no spectral analysis is needed to discover this.

**ATTACK L3-sp-B: Clique-expansion vs. incidence Laplacian gap [SERIOUS]**

L3-sp is proven for one Laplacian variant. The proposal uses two variants (clique-expansion for $d_{\max} \leq 200$, incidence-matrix for $d_{\max} > 200$). The Cheeger-type inequality takes a *different form* for each variant (Chan et al. 2018 vs. standard graph Cheeger). The proposal acknowledges this ("adds ~2 days of proof effort") but does not state whether the same bound form holds.

If the incidence-matrix variant gives a bound of $O(\delta^2 \cdot d_{\max}^2 / \gamma_2^2)$ (plausible — clique expansion absorbs one $d_{\max}$ into edge weights), then the two Laplacian variants give *qualitatively different* partition quality guarantees. A reviewer can ask: "Which theoretical bound applies to the 15% of instances using the incidence Laplacian? Is $d_{\max}$ in the clique-expansion bound an artifact of that specific construction?"

**ATTACK L3-sp-C: Normalized vs. unnormalized Laplacian [MODERATE]**

The proposal says "normalized hypergraph Laplacian" but doesn't pin the normalization. For graphs, the normalized Laplacian $\mathcal{L} = D^{-1/2} L D^{-1/2}$ has Cheeger constant $h \geq \lambda_2/2$ (Cheeger) and $\lambda_2 \geq h^2/2$ (Alon-Milman). For hypergraphs, the analogues depend on which normalization is used (degree-weighted, symmetric random-walk, or Bolla-style). The spectral gap $\gamma_2$ and the Cheeger-type bound both change with normalization.

If the proof uses one normalization but the implementation uses another, the theorem doesn't apply to the computed features.

### 1.3 L3-C (Method-Specific Bounds)

**ATTACK L3-C-Benders-A: Current-master duals are not accessible pre-solve [MODERATE]**

L3-C (Benders) uses $r_j^{(t)}$ — the reduced cost in the *current* Benders master at iteration $t$. This is only available after starting the Benders decomposition. But L3-C is presented as guiding "partition refinement" — suggesting it's used to *improve* a partition before or during decomposition. If you need to run Benders to evaluate the partition quality metric, and the metric is supposed to tell you which partition to use for Benders, you have a chicken-and-egg problem.

**ATTACK L3-C-DW-A: Linking constraint dual magnitudes are dominated by scaling [MODERATE]**

L3-C (DW) weights linking constraints by $|\mu_i^{(t)}|$, the DW master dual. But master duals are notoriously sensitive to: (a) the set of columns currently in the master, (b) dual degeneracy of the master LP, and (c) constraint scaling. A constraint scaled by $10^6$ has a dual scaled by $10^{-6}$. This makes the L3-C bound a function of the formulation's scaling convention, not its mathematical structure. A reviewer testing L3-C on a rescaled instance would get wildly different bound values.

### 1.4 Proposition T2 (Spectral Scaling Law)

**ATTACK T2-A: The perturbation model $A = A_{\text{block}} + E$ doesn't exist for real instances [SERIOUS]**

T2 assumes a clean decomposition: the "true" block-diagonal form $A_{\text{block}}$ exists and $E$ is a perturbation. For real MIPLIB instances, there is no canonical $A_{\text{block}}$. The decomposition of $A$ into block-diagonal + coupling is itself the output of the spectral analysis (or GCG, or manual inspection). This means:
- $\delta = \|E\|_F$ depends on which partition you choose
- $\gamma$ is the spectral gap of $A_{\text{block}}$'s Laplacian, which also depends on the partition
- The bound $z_{LP} - z_D \leq C \cdot \delta^2/\gamma^2$ is a function of the partition, not a property of the instance

So T2 is really: "for any partition, if you call the coupling part $E$, then..." But the partition is what you're trying to choose! This circularity means T2 doesn't *predict* which partition is good — it *describes* a quantity that is computable only after you've already chosen a partition.

**Counter-argument the authors could make:** T2 motivates $\delta^2/\gamma^2$ as a *feature* (computable from the spectral decomposition of the original $A$). **Red-team rebuttal:** Then $\gamma$ is the spectral gap of the *full* Laplacian, not of $A_{\text{block}}$, and the perturbation model breaks down. You'd need to define $\gamma$ as $\lambda_{k+1} - \lambda_k$ of the full Laplacian and argue this approximates the "ideal" gap. This is possible but requires a separate argument not currently in the proposal.

**ATTACK T2-B: Davis-Kahan for $k > 2$ is not $\gamma_2$ [SERIOUS]**

The proposal uses $\gamma$ (the spectral gap $\lambda_2$) throughout, but for $k$-way partitioning ($k > 2$), the relevant quantity is the gap $\lambda_{k+1} - \lambda_k$, not $\lambda_2 - \lambda_1$. Davis-Kahan's $\sin\Theta$ theorem for the $k$-dimensional eigenspace uses $\min_{i \notin S} |\lambda_i - \lambda_j|$ for $\lambda_j$ in the target eigenspace, which is $\lambda_{k+1} - \lambda_k$ at best.

The proposal notes this in passing ("for $k > 2$, relevant gap is $\lambda_{k+1} - \lambda_k$, not $\gamma_2$") but then defines Feature #2 as $\delta^2/\gamma_2^2$, using $\gamma_2$, not $\lambda_{k+1} - \lambda_k$. So the feature being used empirically ($\delta^2/\gamma_2^2$) is *not* the quantity that T2's proof actually bounds against (which would be $\delta^2/(\lambda_{k+1} - \lambda_k)^2$).

This disconnect means: either the feature definition should use $\lambda_{k+1} - \lambda_k$ (but $k$ is unknown a priori and must be estimated), or T2 should be restricted to $k = 2$ (bisection), losing applicability to multi-block decompositions. The proposal does neither.

**ATTACK T2-C: The rounding step (eigenspace → partition) loses more than $O(\delta^2/\gamma^2)$ [MODERATE]**

The proof chain is: Davis-Kahan → eigenspace angle $O(\delta/\gamma)$ → rounding → misclassification rate $O(\delta^2/\gamma^2)$. The rounding step (converting continuous eigenvectors to a discrete partition via $k$-means) uses the analysis of Ng, Jordan, Weiss (2001) or similar. But that analysis assumes the ideal eigenvectors are piecewise constant (one value per block), which holds only when $A_{\text{block}}$ has identical blocks. For heterogeneous blocks (different sizes, different internal structure), the ideal eigenvectors are piecewise constant with *different* values per block, and the rounding analysis requires the blocks to be well-separated in eigenvector space — which is an additional condition not stated in T2.

**ATTACK T2-D: The κ⁴ factor — is it the best possible? [MODERATE]**

$C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$ contains $\kappa^4$. This arises from:
- Hoffman's bound relating dual perturbation to constraint perturbation: $O(\kappa^2)$
- A second $\kappa^2$ from... where exactly?

The proposal doesn't decompose where each $\kappa^2$ factor enters. If one factor comes from the eigenspace-to-partition step (which is combinatorial, not depending on $\kappa$) and one from the partition-to-bound step (L3, which does depend on constraint magnitudes), then it may be possible to reduce to $\kappa^2$ by a more careful analysis. A reviewer at MPC would ask: "Is $\kappa^4$ tight, or an artifact of composing two independent $\kappa^2$ bounds? Can you construct an instance achieving $\Theta(\kappa^4)$?"

If $\kappa^4$ is an artifact and can be reduced to $\kappa^2$, the paper has missed an opportunity. If it's tight, constructing a matching lower bound would significantly strengthen the theorem.

### 1.5 Propositions F1 and F2

**ATTACK F1-A: Permutation invariance of Feature 7 (silhouette score) [MODERATE]**

Feature 7 is the silhouette score of $k$-means on the Gram matrix $VV^T$. The claim is that using $VV^T$ instead of $V$ handles sign ambiguity and rotation within degenerate eigenspaces. However:
- $k$-means on $VV^T$ (an $m \times m$ PSD matrix) is not standard. Do you cluster the *rows* of $VV^T$? The rows of $VV^T$ are the projections of each constraint onto the eigenspace, which are indeed rotation-invariant. But the silhouette score depends on cluster *assignments*, which depend on $k$-means initialization.
- $k$-means is initialization-dependent. Two runs of $k$-means on the same $VV^T$ can give different silhouette scores. This makes Feature 7 *not* a deterministic function of $A$, violating the spirit of F1 (though the proposition only claims invariance under permutation, not determinism).
- **Fix:** Use $k$-means++ with a fixed seed, or use spectral clustering with a deterministic rounding scheme (e.g., the pivoted QR algorithm of Yu & Shi 2003). State that determinism requires fixing the random seed.

**ATTACK F2-A: F2 only bounds $\gamma_2$, not all 8 features [MODERATE]**

F2 gives a scaling sensitivity bound for $\gamma_2$ specifically. What about the other 7 features? Feature 4 (Fiedler vector localization entropy) depends on $|v_i|^2$ where $v$ is the Fiedler vector. Under row scaling $A \to DA$, the Fiedler vector changes — and the entropy can change dramatically if scaling concentrates the vector's mass. No bound is provided for features 3–8 individually.

The paper claims "scaling-sensitivity analysis" but only analyzes one of eight features. A reviewer will ask: "Is eigenvalue decay rate (Feature 3) robust to scaling? Is effective spectral dimension (Feature 8) robust?"

### 1.6 Notation Inconsistencies

**ATTACK N1: $y^*_e$ — constraint-indexed or hyperedge-indexed? [COSMETIC]**

L3 uses $y^*_e$ with subscript $e \in E_{\text{cross}}$, suggesting hyperedge indexing. But LP duals are constraint-indexed ($y^*_i$ for constraint $i$). In the constraint hypergraph, constraints *are* hyperedges — but this identification must be stated. If constraints and hyperedges have different indexing (e.g., equality constraints split into two inequalities), the notation is ambiguous.

**ATTACK N2: $\gamma$ vs. $\gamma_2$ vs. "spectral gap" [COSMETIC]**

T2 uses $\gamma$ (unsubscripted), L3-sp uses $\gamma_2$, Feature 5 uses "$\gamma_2/\gamma_k$". These are presumably: $\gamma = \gamma_2 = \lambda_2$ (algebraic connectivity). But T2's proof for $k > 2$ should use $\lambda_{k+1} - \lambda_k$, not $\lambda_2$. The notation suggests they're all the same, masking the $k$-dependence issue.

---

## 2. Algorithm Attacks

### 2.1 Eigensolve Numerical Issues

**ATTACK ALG-1: Near-zero spectral gap crashes feature computation [SERIOUS]**

Feature 2 is $\delta^2/\gamma_2^2$. When $\gamma_2 \approx 0$ (disconnected or nearly disconnected hypergraph), this feature is $\infty$ or numerically huge. The proposal acknowledges this for eigensolve convergence (R3: "assign NaN") but not for feature computation. A numerically tiny $\gamma_2$ (say $10^{-15}$) that isn't exactly zero will produce $\delta^2/\gamma_2^2 \approx 10^{30}$ — a finite but meaningless number that will dominate any classifier. XGBoost may handle this via tree splits, but logistic regression (listed as a classifier) will not.

**Fix:** Define a minimum $\gamma_2$ threshold (e.g., $10^{-10}$) below which the feature is set to a sentinel value or the instance is classified as "disconnected — decompose trivially by connected components."

**ATTACK ALG-2: Repeated eigenvalues break Fiedler vector extraction [MODERATE]**

If $\lambda_2 = \lambda_3$ (multiplicity > 1), "the Fiedler vector" is not unique — any vector in the eigenspace is valid. Feature 4 (Fiedler vector localization entropy) depends on which vector is returned, making it implementation-dependent (ARPACK vs. LOBPCG may return different vectors in the degenerate eigenspace). The proposal handles this for Feature 7 (Gram matrix) but not for Feature 4.

This occurs on instances with exact symmetry (e.g., symmetric transportation problems). MIPLIB has several such instances.

**ATTACK ALG-3: LOBPCG fallback has different convergence properties [MODERATE]**

The eigensolve uses ARPACK (shift-invert Lanczos) with LOBPCG as fallback. These methods have different convergence behaviors:
- ARPACK with shift-invert converges fast for smallest eigenvalues but requires factoring $(L - \sigma I)$
- LOBPCG is matrix-free but converges slowly for small gaps

If ARPACK fails on an instance (e.g., factorization fails due to near-singularity) and LOBPCG takes over, the computed eigenvalues may have different accuracy. Feature values from ARPACK-solved instances and LOBPCG-solved instances are not directly comparable. At minimum, the eigensolve method should be recorded per-instance and checked for systematic bias in the classification results.

### 2.2 Two Laplacian Variants

**ATTACK ALG-4: Feature discontinuity at $d_{\max} = 200$ threshold [SERIOUS]**

Instances with $d_{\max} = 199$ use clique expansion; instances with $d_{\max} = 201$ use the incidence-matrix Laplacian. The two Laplacians have different spectra for the same hypergraph. Feature values may have a systematic discontinuity at $d_{\max} = 200$. If the classifier learns to exploit this discontinuity (e.g., "if Feature 1 > threshold AND $d_{\max}$ near 200, then..."), it's learning an artifact of the feature computation pipeline, not a property of the instance.

The validation protocol ("Spearman $\rho > 0.85$ between variants on instances where both are tractable") only tests the *overlap* region. It doesn't test whether the classifier's behavior is consistent across the boundary.

**Fix:** Include $d_{\max}$ as a feature. Run the ablation with and without the Laplacian variant indicator. Better: use the incidence-matrix Laplacian for all instances (avoiding the discontinuity) and verify that the clique expansion provides negligible additional information on the $d_{\max} \leq 200$ subset.

### 2.3 Spectral Clustering Failure Modes

**ATTACK ALG-5: $k$ selection is unspecified [SERIOUS]**

The proposal uses "bottom-$k$ eigenvectors" throughout but never specifies how $k$ is chosen. For the spectral features, $k$ directly affects:
- Feature 5 (algebraic connectivity ratio $\gamma_2/\gamma_k$): value depends on $k$
- Feature 6 (coupling energy $\delta^2$): requires knowing the block-diagonal structure, which requires $k$
- Feature 7 (silhouette score): clustering into $k$ groups
- Feature 8 (effective spectral dimension): independent of $k$ by definition

If $k$ is chosen by eigengap heuristic (the most common method), then $k$ itself is a function of the spectrum, creating a circular dependency: features depend on $k$, which depends on the spectrum, which the features summarize. More practically, the eigengap heuristic is unreliable on instances without clear block structure — it may return $k = 1$ or $k = n/2$.

**Reviewer question:** "How is $k$ selected, and how sensitive are the results to this choice?"

**ATTACK ALG-6: k-means initialization sensitivity [MODERATE]**

Feature 7 (silhouette score) uses $k$-means clustering. $k$-means is initialization-sensitive, especially for small $k$ with elongated clusters (common in spectral embeddings). Running $k$-means 10 times with different seeds and taking the best silhouette is standard practice, but adds computational cost and doesn't fully resolve the issue.

The proposal specifies "fixed seed" implicitly via F1's determinism requirement, but a single fixed seed may consistently produce bad clusterings for certain eigenspace geometries.

---

## 3. Evaluation Attacks

### 3.1 Label Quality

**ATTACK EVAL-1: "Best method" label is fragile to time cutoff [SERIOUS]**

The label is argmax of dual bound improvement at wall-clock parity. But:
- GCG (DW) has expensive setup (structure detection, initial column generation) but fast convergence once started
- SCIP Benders has cheap setup but may converge slowly on dual bounds
- At 60s: Benders may win (fast setup). At 900s: DW may win (better convergence). At 3600s: results may flip again.

The proposal includes a 4-cutoff stability analysis, which is good. But the *labels used for training* come from one specific cutoff choice. If 30% of labels flip between cutoffs, the classifier is learning a noisy target. The consensus-label protocol (majority vote) partially addresses this, but majority vote over 4 cutoffs can still produce labels that aren't meaningful. Consider: an instance where Benders wins at 60s and 300s, but DW wins at 900s and 3600s. Majority vote says "Benders" — but a practitioner with 3600s would choose DW.

**ATTACK EVAL-2: Dual bound improvement ≠ practical usefulness [MODERATE]**

The label is based on dual bound improvement, not solve time or gap closure. An instance where DW improves the dual bound by 0.1% in 900s while monolithic SCIP closes the gap in 30s would be labeled "DW-amenable" even though DW is practically useless. The proposal doesn't include a "decomposition overhead" filter: if monolithic SCIP solves the instance within the time cutoff, decomposition is unnecessary regardless of dual bound improvement.

**ATTACK EVAL-3: GCG and SCIP Benders are not equally mature [MODERATE]**

GCG has 15+ years of development specifically for DW decomposition. SCIP's Benders is a more recent addition with less tuning. If the census shows DW wins on 60% of structured instances, this may reflect GCG's engineering maturity, not DW's structural superiority. The proposal acknowledges this generally ("labels reflect implementation quality") but the fix (external baselines) doesn't fully resolve it — the external baselines *are* GCG and SCIP Benders, which have this asymmetry.

### 3.2 Feature Ablation Design

**ATTACK EVAL-4: "Spectral features are just proxies for density" — the killer critique [SERIOUS]**

The single most dangerous reviewer argument:

> "The spectral gap $\gamma_2$ of the constraint hypergraph Laplacian is strongly correlated with constraint density and variable degree distribution. Your 8 'spectral features' are expensive-to-compute proxies for cheap syntactic statistics. Table X shows R² = 0.65 between $\gamma_2$ and density — the remaining 35% of variance is noise, not signal."

The G0 gate (R² < 0.70) is designed to catch this, but:
- R² = 0.65 would pass G0 but still leave the paper vulnerable to this critique
- The linear model used for G0 may miss nonlinear relationships (e.g., $\gamma_2 \approx f(\text{density})^2$)
- Even if R² < 0.70, a reviewer can argue the residual is noise: "R² = 0.55 means 55% of the spectral gap is predicted by density. Your 8 features add 45% of noisy variance to the classifier."

**Fix:** The G0 test should include nonlinear regressors (random forest regressing $\gamma_2$ on syntactic features) alongside the linear model. If RF R² > 0.80, spectral features are likely redundant regardless of linear R².

**ATTACK EVAL-5: Top-$k$ feature comparison doesn't control for information content [MODERATE]**

The ablation compares "top-3 spectral vs. top-3 syntactic" features. But mutual information ranking (used for feature selection) may pick correlated spectral features (e.g., $\gamma_2$, $\delta^2/\gamma_2^2$, and $\gamma_2/\gamma_k$ — all functions of eigenvalues). Three correlated features carry less information than three independent features. The syntactic set (density, degree max, coefficient range) may have higher effective dimensionality.

**Fix:** Report effective dimensionality (PCA) per feature subset. Or use mRMR (minimum redundancy, maximum relevance) for feature selection instead of simple MI ranking.

**ATTACK EVAL-6: Nested CV with 500 instances and 3 classes yields thin strata [MODERATE]**

500 instances, 5-fold outer CV = 400 train / 100 test per fold. With 3 classes (Benders/DW/none), if Benders is 15% of instances, you have ~15 Benders test instances per fold. Per-structure-type breakdowns (5 types) yield ~3 Benders instances per structure type per fold. Confidence intervals on accuracy will be enormous.

The R10 risk (< 30 Benders instances total) is acknowledged, but the per-fold-per-stratum issue is not. The paper may report "accuracy on Benders-amenable network instances" based on 2–3 examples — statistically meaningless.

### 3.3 Sample and Scope

**ATTACK EVAL-7: 500 of 1,065 is not a "census" [MODERATE]**

The title says "First Complete MIPLIB 2017 Decomposition Census" but the decomposition evaluation covers only 500 stratified instances. The remaining 565 get spectral annotations only (no decomposition results). "First 47%-Complete MIPLIB 2017 Decomposition Census" is more accurate.

The spectral annotations for all 1,065 are genuinely complete, but the *decomposition* census — which is the claimed novel contribution — is not. A reviewer will note this discrepancy.

**ATTACK EVAL-8: Only 10–25% of MIPLIB has block structure — is the census useful? [MODERATE]**

If 75–90% of instances are classified as "none" (no useful decomposition), the census is 75–90% "here's an instance where nothing works." The interesting data (which decomposition helps and by how much) covers 100–250 instances. The census's value depends entirely on the *quality* of findings on this subset, not the *coverage* of the full 1,065.

---

## 4. Novelty Attacks

### 4.1 L3: Is This Just LP Duality Restated?

**ATTACK NOV-1: L3's intellectual contribution [SERIOUS]**

The core insight of L3 — that relaxing coupling constraints creates a gap bounded by dual multiplier × violation — is Fisher (1981), Geoffrion (1974), and literally Chapter 6 of any integer programming textbook. The "hypergraph language" and $(n_e - 1)$ factor are notational contributions, not conceptual ones.

**What would make L3 genuinely novel:** A *tight* bound (matching lower bound), or a bound that connects to computationally accessible quantities (not $y^*$), or a bound that works for IP duality gaps (not just LP). None of these are claimed.

### 4.2 Spectral Features: How Different from ML4CO?

**ATTACK NOV-2: GNN features subsume spectral features [MODERATE]**

The ML for Combinatorial Optimization (ML4CO) community routinely uses graph neural networks on the bipartite constraint-variable graph (Gasse et al. 2019, Nair et al. 2020). GNN message passing on this graph *implicitly computes* spectral features — GNNs with $L$ layers compute features in the span of the first $L$ eigenvectors of the graph Laplacian (Xu et al. 2019 GIN paper; spectral graph theory ↔ GNN equivalence). The "new feature family" may be a strictly less expressive subset of what GNNs already compute.

**The authors' defense** (interpretability, theoretical grounding via L3) is valid but must be stated forcefully. If the GNN baseline (§6.3, labeled "should-do, not must-do") outperforms spectral features, the paper's thesis collapses.

### 4.3 The Census: "Just Run GCG and SCIP and Record Results"

**ATTACK NOV-3: Census novelty is engineering, not science [MODERATE]**

A hostile reviewer: "This census could be produced by a competent MS student with a bash script: `for instance in miplib/*.mps; do run_gcg $instance; run_scip_benders $instance; record_results; done`. The 'spectral annotations' add eigenvalue computation, which is an ARPACK call. What is the intellectual contribution?"

**Defense:** The contribution is in (a) the stratified evaluation design, (b) the multi-cutoff label stability analysis, (c) the spectral structural annotations as a *new characterization*, and (d) the open artifact. But the defense is strongest when paired with *findings*. If the census reveals only "some instances have block structure, others don't" — something the community already knows from Bergner et al. (2015) — the census is indeed incremental.

### 4.4 T2: Is Davis-Kahan + Rounding + LP Gap a Known Chain?

**ATTACK NOV-4: T2's proof technique is textbook composition [MODERATE]**

- Davis-Kahan $\sin\Theta$ theorem: textbook (Stewart & Sun 1990)
- Eigenspace → partition rounding: textbook spectral clustering (Ng, Jordan, Weiss 2001)
- Partition quality → LP gap: L3, which is itself close to textbook (see Attack NOV-1)

Each step is standard. The novelty claim is in the *composition* applied to the *MIP decomposition setting*. This is legitimate but thin — "I applied three known theorems in sequence to a new context." At IPCO, this would be insufficient. At JoC (computational study), it's fine as motivation.

---

## 5. Scope Attacks

### 5.1 Reformulation Selection vs. Algorithm Selection

**ATTACK SCOPE-1: Is the distinction practically meaningful? [MODERATE]**

The paper claims "reformulation selection" is "strictly harder" than algorithm selection. But from a machine learning perspective, both are classification problems: instance features → label. The "strictly harder" claim conflates the *mathematical* difficulty of reformulation (changing the feasible region) with the *predictive* difficulty (classifying instances). Predicting which reformulation is best may be easier or harder than predicting which solver configuration is best — it depends on the signal-to-noise ratio in the features, not on the mathematical sophistication of the transformation.

A reviewer could write: "The authors conflate the difficulty of performing a reformulation with the difficulty of predicting which reformulation to perform. The latter is a standard classification problem regardless of what the labels represent."

### 5.2 Why Not Just Use AutoFolio?

**ATTACK SCOPE-2: AutoFolio + graph features as a baseline is too weak [MODERATE]**

The paper compares against "AutoFolio with syntactic features." But AutoFolio (Lindauer et al. 2015) is an algorithm selection framework that can use *any* features, including graph features from the constraint-variable bipartite graph. The relevant baseline is: "AutoFolio with the best known feature set for MIP (Hurley et al. 2014, Kruber et al. 2017), with decomposition method as the 'algorithm'." If this baseline already achieves 60% accuracy, the marginal contribution of spectral features (claimed ≥5pp) is modest.

The proposal includes the Kruber et al. feature set as a baseline (good), but doesn't discuss whether AutoFolio's meta-learning (algorithm selector selection) might find a better feature combination automatically.

### 5.3 Is This a Theory Paper or Not?

**ATTACK SCOPE-3: Identity crisis — paper tries to be too many things [MODERATE]**

The paper contains:
1. A theorem (T2) — but vacuous, demoted to motivation
2. A lemma (L3) — but "well-known" content in hypergraph notation
3. A feature engineering study — the claimed core
4. A census / benchmark — the strongest artifact
5. A machine learning pipeline — standard RF/XGBoost
6. An oracle / tool — conditional on G3

JoC reviewers expect coherent computational studies, not grab-bags. A reviewer might write: "The authors should decide whether this is a theory paper (in which case T2 must be tightened), a feature engineering paper (in which case the census is supplementary), or a benchmark paper (in which case the ML pipeline is supplementary). Currently it is all three, and none convincingly."

---

## 6. Contradiction Detection

### 6.1 Internal Contradictions

**CONTRADICTION C1: $\gamma_2$ in T2 vs. $\lambda_{k+1} - \lambda_k$ in the proof [SERIOUS]**

T2 states the bound in terms of $\gamma^2$ (= $\gamma_2^2 = \lambda_2^2$). The proof requires Davis-Kahan for the $k$-dimensional eigenspace, which uses $\lambda_{k+1} - \lambda_k$. Feature 2 uses $\delta^2/\gamma_2^2$. These are three different quantities for $k > 2$. The proposal notes this ("for $k > 2$, relevant gap is $\lambda_{k+1} - \lambda_k$, not $\gamma_2$") but then defines the feature using $\gamma_2$ anyway. This is either: (a) a deliberate simplification that should be stated as "we use $\gamma_2$ as a proxy for $\lambda_{k+1} - \lambda_k$; see Appendix," or (b) an error.

**CONTRADICTION C2: "First MIPLIB 2017 Decomposition Census" vs. 500-instance evaluation [MODERATE]**

The title says "First Complete MIPLIB 2017 Decomposition Census." The evaluation is on 500 instances. The spectral annotation covers 1,065 but is not a *decomposition* census — it's a spectral feature census. The depth check amendment says "500-instance stratified sample for paper evaluation" with the full 1,065 as "supplementary material." If the full 1,065 results include 60% timeouts, calling it "complete" is misleading.

**CONTRADICTION C3: T2 is "motivational" but $\delta^2/\gamma^2$ is the core feature [MODERATE]**

If T2 is demoted to "motivational" and acknowledged as vacuous, then the feature $\delta^2/\gamma^2$ loses its theoretical justification. The paper then argues: "T2 explains *why* $\delta^2/\gamma^2$ is a good feature" while simultaneously saying "T2 is vacuous and not a contribution." A reviewer can ask: "If the theorem that justifies your feature is vacuous, why should I believe the feature is principled rather than ad hoc?"

The paper needs a crisper narrative: either T2 provides genuine (qualitative) justification for the feature choice, or the features are justified purely empirically and T2 is irrelevant decoration.

### 6.2 Depth-Check Violations

**VIOLATION D1: Score inflation [MODERATE]**

The Verification Report (Appendix V of proposal) caught systematic +1 score inflation above depth-check binding scores. If the final submitted theory document still uses inflated scores, this violates the binding depth check.

**VIOLATION D2: G1 threshold softened [MODERATE]**

Depth check specifies G1 kill at $\rho < 0.4$. Proposal uses $\rho < 0.3$ with investigation zone $[0.3, 0.4)$. The Verification Report flagged this. If the spectral correlation is $\rho = 0.35$, the depth check says ABANDON, but the proposal says INVESTIGATE. This is a material deviation from a binding constraint.

### 6.3 Evaluation-Theory Conflict

**CONFLICT E1: L3 requires $y^*$ but evaluation doesn't compute it [MODERATE]**

L3's empirical verification (§theory/l3_verification.py) presumably compares the L3 bound against the actual LP-decomposition gap. To compute the L3 bound, you need $y^*$ (monolithic LP dual). To compute the actual gap, you need both $z_{LP}$ and $z_D$. But if you have $z_{LP}$ (from solving the monolithic LP) and $z_D$ (from running decomposition), you can directly compute the gap — L3 is unnecessary for verification. L3 verification becomes: "we computed $y^*$, computed the L3 bound, and verified it's larger than $z_{LP} - z_D$." This always holds (L3 is an upper bound) — so verification is trivially satisfied. The *informative* test is "how tight is L3?" — but this isn't a verification, it's a measurement.

---

## 7. Killer Questions

These are the five questions that, asked by a reviewer, would be hardest to answer convincingly:

### KQ1: "Can you demonstrate an instance where spectral features change the decomposition decision relative to syntactic features alone?"

*Why it's deadly:* This asks for a concrete, verifiable example — not a statistical aggregate. If spectral features add +5pp accuracy on average, there must exist instances where the syntactic classifier says "DW" but the spectral classifier correctly says "Benders" (or vice versa). Showing this concretely is essential; if you can't produce 5–10 such examples with explanations, the statistical improvement may be noise.

*Why it's hard to answer:* The +5pp gain may come from boundary cases where no method is clearly better, or from label noise. Finding clean, interpretable examples of spectral features making the correct decision when syntactic features don't may be difficult.

### KQ2: "Your Lemma L3 requires the LP dual solution $y^*$, which means solving the monolithic LP. If I have $y^*$, I already know $z_{LP}$. What does L3 buy me that I don't already know?"

*Why it's deadly:* It attacks the practical utility of the paper's primary theoretical contribution. The "retrospective vs. prospective" issue (Attack L3-B).

*Why it's hard to answer:* Any answer involving "approximate duals" introduces approximation error that weakens L3. Any answer involving "LP solves are cheap" undermines the decomposition motivation. The honest answer ("L3 is a theoretical quality metric, not a practical tool") reduces its importance.

### KQ3: "Your Feature 2 ($\delta^2/\gamma_2^2$) uses $\gamma_2$, but your Proposition T2's proof requires $\lambda_{k+1} - \lambda_k$. For what value of $k$ are you computing this feature? How sensitive are results to $k$?"

*Why it's deadly:* It exposes the $k$-selection gap (Attack ALG-5) and the $\gamma_2$ vs. $\lambda_{k+1} - \lambda_k$ inconsistency (Contradiction C1) simultaneously.

*Why it's hard to answer:* Admitting $k$ is heuristically chosen undermines the theoretical grounding. Using $\lambda_{k+1} - \lambda_k$ instead of $\gamma_2$ introduces a new hyperparameter ($k$) into the feature definition. There is no clean answer.

### KQ4: "What specific finding does your census reveal that was not previously known from Bergner et al. (2015) or from running GCG on MIPLIB?"

*Why it's deadly:* If the answer is "we confirm that ~15% of MIPLIB has block structure amenable to DW, which Bergner already showed," the census is confirmatory, not novel. The census's value depends on *new findings* — previously unknown structure, surprising method comparisons, unexpected futility patterns.

*Why it's hard to answer:* The authors don't know the answer yet (the census hasn't been run). If the answer is disappointing, the paper's primary contribution evaporates. This is the central empirical risk, and no amount of theory can mitigate it.

### KQ5: "Why should I use your 8 spectral features when a 3-layer GNN on the constraint-variable bipartite graph provably subsumes the information in the first 3 eigenvectors, and GNN-based approaches already exist in ML4CO?"

*Why it's deadly:* It challenges the novelty of the entire feature family. GNNs operating on the constraint-variable graph can approximate spectral features (Xu et al. 2019 GIN ≈ WL test ≈ spectral decomposition). If a GNN baseline outperforms, the "new feature family" claim collapses.

*Why it's hard to answer:* The interpretability/efficiency defense ("spectral features are faster and interpretable") is valid but may not satisfy a reviewer who views ML4CO as the state of the art. The L3 connection ("spectral features have formal connection to decomposition bounds; GNN features don't") is the strongest defense but depends on L3's novelty, which is itself attacked.

---

## 8. Severity Classification Summary

### FATAL (0 items)
None — the depth check and critiques have already caught and resolved the truly fatal issues (T2 as headline, 155K LoC, evaluation circularity). The remaining issues are serious but not individually fatal for JoC.

### SERIOUS (8 items)
| # | Finding | Section |
|---|---------|---------|
| S1 | L3 "which dual bound?" ambiguity — lemma may be trivially true or undefined | §1.1 L3-A |
| S2 | L3-sp vacuous for large $d_{\max}$ (bordered block-diagonal instances) | §1.2 L3-sp-A |
| S3 | L3-sp requires separate proof for incidence Laplacian variant | §1.2 L3-sp-B |
| S4 | T2 perturbation model doesn't exist for real instances | §1.4 T2-A |
| S5 | Feature 2 uses $\gamma_2$ but T2 proof needs $\lambda_{k+1} - \lambda_k$ | §1.4 T2-B |
| S6 | Near-zero spectral gap crashes Feature 2 computation | §2.1 ALG-1 |
| S7 | Feature discontinuity at $d_{\max} = 200$ Laplacian switch | §2.2 ALG-4 |
| S8 | $k$ selection unspecified — affects Features 5, 6, 7 | §2.3 ALG-5 |

### MODERATE (18 items)
| # | Finding | Section |
|---|---------|---------|
| M1 | L3 requires $y^*$ (circular — need LP solution to evaluate) | §1.1 L3-B |
| M2 | L3 is restatement of weak duality in hypergraph notation | §1.1 L3-C |
| M3 | L3-sp normalized vs. unnormalized Laplacian unspecified | §1.2 L3-sp-C |
| M4 | L3-C Benders: chicken-and-egg with current-master duals | §1.3 |
| M5 | L3-C DW: linking duals dominated by scaling | §1.3 |
| M6 | T2 rounding step needs homogeneous-block assumption | §1.4 T2-C |
| M7 | T2 $\kappa^4$ — tight or artifact? | §1.4 T2-D |
| M8 | F1: Feature 7 silhouette depends on $k$-means initialization | §1.5 F1-A |
| M9 | F2 only covers $\gamma_2$, not features 3–8 | §1.5 F2-A |
| M10 | Repeated eigenvalues break Feature 4 | §2.1 ALG-2 |
| M11 | ARPACK vs. LOBPCG accuracy differences | §2.1 ALG-3 |
| M12 | $k$-means initialization sensitivity for Feature 7 | §2.3 ALG-6 |
| M13 | Label fragility across time cutoffs | §3.1 EVAL-1 |
| M14 | Dual bound improvement ≠ practical usefulness | §3.1 EVAL-2 |
| M15 | GCG vs. SCIP Benders maturity asymmetry | §3.1 EVAL-3 |
| M16 | Spectral features as density proxies (surviving G0) | §3.2 EVAL-4 |
| M17 | Top-$k$ comparison doesn't control for correlation | §3.2 EVAL-5 |
| M18 | Census "complete" claim vs. 500-instance reality | §6.1 C2 |

### COSMETIC (4 items)
| # | Finding | Section |
|---|---------|---------|
| C1 | $y^*_e$ notation — constraint vs. hyperedge index | §1.6 N1 |
| C2 | $\gamma$ vs. $\gamma_2$ notation inconsistency | §1.6 N2 |
| C3 | Score inflation above depth-check binding | §6.2 D1 |
| C4 | G1 threshold deviation from binding constraint | §6.2 D2 |

---

## 9. Priority Fixes

Ordered by impact-per-effort:

1. **Pin L3's dual bound definition** (S1) — 1 day of writing. Clarify that $z_D$ is the Lagrangian dual at a specific multiplier, not the converged value. State explicitly that L3 is a retrospective quality metric, and note that $y^*$ can be approximated.

2. **Define $k$-selection protocol** (S8) — 1 day of design. Use eigengap heuristic with a fallback ($k \in [2, 20]$, maximize eigengap). Report sensitivity analysis ($k \pm 2$). Include $k$ as a meta-feature.

3. **Handle $\gamma_2 \approx 0$** (S6) — 0.5 day. Floor $\gamma_2$ at $10^{-10}$. For disconnected instances ($\lambda_2 = 0$ exactly), report as "trivially decomposable" and assign to connected-component decomposition.

4. **Reconcile $\gamma_2$ vs. $\lambda_{k+1} - \lambda_k$** (S5) — 1 day. Define Feature 2 as $\delta^2/(\lambda_{k+1} - \lambda_k)^2$ and add $\delta^2/\gamma_2^2$ as a secondary feature. Or: restrict T2 to $k = 2$ and note the $k$-way generalization.

5. **Address Laplacian discontinuity at $d_{\max} = 200$** (S7) — 2 days. Run both Laplacians on the overlap set. Report systematic differences. Consider using incidence-matrix Laplacian uniformly and verifying minimal information loss.

6. **State L3-sp for both Laplacian variants** (S3) — 3–5 days of proof work. This is on the critical path; budget it explicitly.

7. **Acknowledge T2 perturbation model limits** (S4) — 1 day of writing. Define $\delta$ and $\gamma$ in terms of the *full* Laplacian spectrum (not the unknown $A_{\text{block}}$). Restate T2 with: "$\gamma$ is the $k$-th eigengap of the constraint hypergraph Laplacian; $\delta^2$ is the coupling energy measured by..."

8. **L3-sp $d_{\max}$ vacuousness** (S2) — 1 day of writing. State the regime where L3-sp is informative. Acknowledge it degrades on wide constraints. Show that the empirical L3 correlation may be stronger than the L3-sp theoretical bound.

---

## 10. Meta-Assessment

This project has been thoroughly pre-critiqued by three expert panels and a depth check. The surviving design is defensible for JoC. The remaining vulnerabilities fall into two categories:

**Category A: Precision issues in the mathematical claims (S1–S5).** These are all fixable with careful writing. None require new mathematical ideas — they require stating assumptions, defining terms unambiguously, and acknowledging limitations. If these are not fixed, Reviewer 2 at JoC will spend their review pointing them out, and the paper will require major revision.

**Category B: The empirical bet (KQ4, EVAL-1/4/7/8).** The paper's ultimate success depends on two unknowable outcomes: (1) do spectral features beat syntactic features? and (2) does the census reveal interesting findings? No amount of theoretical polish can substitute for these empirical results. The kill-gate design (G0–G4) is well-calibrated to detect failure early, but the project team should mentally prepare for the G3-marginal scenario (+3–5pp gain) and have a concrete plan for making that into a publishable story. A marginal gain with a thorough census and honest negative-result reporting is still publishable at C&OR or CPAIOR.

**The single most important thing the team can do right now:** Run the G0 gate (regress $\gamma_2$ on syntactic features for 50 instances). If R² > 0.80, save months of work.
