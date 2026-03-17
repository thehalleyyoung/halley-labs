# Red-Team Report: Spectral Decomposition Oracle

**Role:** Adversarial Reviewer — Destroy Every Weak Point  
**Target Venue:** INFORMS Journal on Computing  
**Date:** 2025-07-22  
**Mandate:** Attack every claim until only bulletproof ones survive.
**Scope:** All load-bearing claims: L3, L3-sp, L3-C, T2, F1, F2, 8 spectral features, two Laplacian variants, feature ablation, MIPLIB census.

---

## 1. Proof Attacks

### 1.1 Lemma L3 (Partition-to-Bound Bridge)

**Claim:** $z_{LP} - z_D \leq \sum_{e \in E_{\text{cross}}} |y^*_e| \cdot (n_e - 1)$ for crossing edges $e$ spanning $n_e$ blocks.

**ATTACK L3-1: Hidden dependence on LP dual non-uniqueness [SERIOUS]**

L3 requires $y^*$, the optimal dual of the monolithic LP relaxation. Optimal duals are generically non-unique for degenerate LPs — the standard case for structured MIPs with many constraints. The bound depends on *which* optimal dual you pick. Does L3 hold for the worst-case dual? The minimum-norm dual? Any dual? If it holds for any dual, you can tighten by solving $\min_{y^* \in Y^*_{\text{opt}}} \sum |y^*_e|(n_e - 1)$, which is itself an LP — never stated. If it holds for a specific dual, SCIP's LP solver may return a different one. Dual degeneracy is ubiquitous in MIPLIB, invalidating the bound for practical computation unless the proof works for *all* optimal duals.

**ATTACK L3-2: L3 is Geoffrion (1974) in a hypergraph costume [FATAL if confirmed]**

The core of L3 — that relaxing coupling constraints creates a gap bounded by dual multiplier magnitude — is the weak duality gap in Lagrangian relaxation. For Lagrangian relaxation dualizing constraint subset $S$: $z_{LP} - z_{LR}(\lambda) = \sum_{i \in S} \lambda_i (b_i - a_i^T x^*)$. L3 repackages this in hypergraph language with the $(n_e - 1)$ factor accounting for multi-block spanning. A reviewer versed in Lagrangian relaxation (Geoffrion 1974, Fisher 1985) will recognize this as an over-decorated version of a textbook bound: "the gap equals the dual weight on relaxed constraints." What intellectual content does L3 add beyond the scalar case? Is $(n_e-1)$ a genuine structural insight or a counting argument that follows in three lines from decomposing the dual feasible region?

**ATTACK L3-3: The $(n_e - 1)$ factor is unexplained and possibly loose [MODERATE]**

Why $(n_e - 1)$ and not $n_e$, $(n_e-1)/n_e$, or $\binom{n_e}{2}$? For a hyperedge crossing $n_e$ blocks, is the $(n_e-1)$ multiplier tight? Construct: a constraint with support across all $k$ blocks but coefficient pattern where the actual contribution to the gap is $|y^*_e| \cdot 1$, not $|y^*_e| \cdot (k-1)$. If such examples exist, the factor is loose by $k-1$, which matters for $k \geq 5$.

**ATTACK L3-4: Direction confusion for non-converged decompositions [SERIOUS]**

The bound states $z_{LP} - z_D(P) \leq \text{cw}$, implying $z_D \leq z_{LP}$ (weak duality). This holds only when the decomposed dual is a *restriction* of the monolithic dual. For DW with converged column generation: $z_{LP} = z_{DW}^*$, so $z_{LP} - z_D = 0$ and L3 is trivially true but vacuous. For Benders with a finite number of cuts: $z_D^{BD}$ approaches $z_{LP}$ from below, and intermediate values may not satisfy the inequality in the claimed direction. The evaluation strategy (§6.2) computes $\Delta = |z_{LP}^* - z_{\text{decomp}}^*|$ with absolute value — suspicious, since it suggests the direction is not guaranteed. If the inequality direction is wrong for some decomposition variants, L3 is *false* as stated.

**ATTACK L3-5: L3 requires LP duals but is pitched as preprocessing [SERIOUS]**

L3's crossing weight requires $y^*$ from the monolithic LP (Algorithm 3.4, line 4). The oracle is pitched as a "lightweight preprocessing layer." But computing $y^*$ requires solving the monolithic LP — the very problem decomposition is meant to avoid. If you have $y^*$, you already have $z_{LP}$, plus reduced costs, basis structure, and constraint activity — all of which directly reveal decomposition structure, making the 8 spectral features (computed from coefficients only) *redundant*. L3 is a *retrospective* quality metric, not a *prospective* guide. The simulated Reviewer 3 in the verification framework (§7.3) explicitly identifies this: "If L3 requires solving the LP first, its value as a preprocessing tool is limited." The paper has no response.

### 1.2 L3-sp (Spectral Partition Crossing Weight)

**Claim:** Spectral partition crossing weight $\leq O(\delta^2 \cdot d_{\max} / \gamma_2^2)$.

**ATTACK L3-sp-1: $d_{\max}$ dependence is devastating [SERIOUS]**

MIPLIB instances frequently have $d_{\max} > 10{,}000$ (set-covering/packing formulations). The bound scales linearly with $d_{\max}$, making it vacuous for exactly the instances where spectral analysis should help most — large, structured problems with wide constraints. The incidence-matrix Laplacian is triggered for $d_{\max} > 200$, but L3-sp's constant may differ between the two Laplacian variants (they have "different non-trivial spectrum" per line 208–209 of the algorithms document). No analysis addresses what happens to L3-sp's quality under the incidence-matrix Laplacian.

**ATTACK L3-sp-2: Tautological in the interesting regime [MODERATE]**

When the constraint hypergraph has weak block structure (the boundary case where the oracle's decision matters), $\gamma_2$ is small and $\delta^2/\gamma_2^2$ is large. The bound says: "when spectral structure is weak, the crossing weight is large." This is tautological. The bound is non-vacuous only when $\gamma_2$ is large (strong structure), i.e., exactly when decomposition's benefit is already obvious without any bound.

**ATTACK L3-sp-3: Clique-expansion vs. incidence Laplacian gap [SERIOUS]**

L3-sp is proved for one Laplacian variant. The two variants have different Cheeger-type inequalities (Chan et al. 2018 vs. standard graph Cheeger). The proposal acknowledges this requires "~2 days of proof effort" but does not state whether the same bound form holds for the incidence-matrix Laplacian. If the incidence variant gives $O(\delta^2 \cdot d_{\max}^2 / \gamma_2^2)$, the two Laplacians give qualitatively different partition guarantees — and 15% of instances use the incidence variant.

### 1.3 L3-C (Method-Specific Bounds)

**ATTACK L3-C-1: Benders specialization mixes variables and constraints [MODERATE]**

L3 partitions *variables*; Benders partitions variables into complicating (master) and subproblem sets. L3-C Benders (Algorithm 4.2) uses "reduced costs of coupling variables" as weights. Reduced costs are primal-dual quantities depending on the LP basis, not just the dual vector — they coincide with shadow prices only at an optimal basis. If the LP solver returns a non-basic optimal (interior-point methods), reduced costs are undefined. This gap between L3 and L3-C Benders is a potential inconsistency, fixable only by restricting to simplex-based LP solves — an unstated assumption.

**ATTACK L3-C-2: DW specialization assumes converged column generation [SERIOUS]**

L3-C DW uses optimal DW master duals $\mu_i$ as weights, available only after column generation converges. Before convergence, master duals are suboptimal and L3-C DW doesn't apply. The census pipeline (Algorithm 6.1, Phase 6) runs GCG with time limits; if GCG times out before CG convergence, the "DW dual bound" is the master LP bound at termination — not the converged value. L3-C DW systematically does not apply to census data where GCG times out, undermining the empirical validation.

**ATTACK L3-C-3: DW linking-constraint duals are scale-dependent [MODERATE]**

L3-C DW weights linking constraints by $|\mu_i|$. Master duals are notoriously sensitive to: (a) the column set in the master, (b) dual degeneracy, (c) constraint scaling. A constraint scaled by $10^6$ has dual scaled by $10^{-6}$. This makes L3-C DW a function of formulation scaling convention, not mathematical structure. Rescaling an instance changes the L3-C bound arbitrarily.

### 1.4 Proposition T2 (Spectral Scaling Law)

**Claim:** $z_{LP} - z_D(\hat{\pi}) \leq C \cdot \delta^2/\gamma^2$ with $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$.

**ATTACK T2-1: Davis-Kahan requires gap in the *perturbed* matrix [SERIOUS]**

The standard Davis-Kahan $\sin\Theta$ theorem bounds $\|\sin\Theta(\hat{V}, V)\| \leq \|E\|_F / \delta_{\text{gap}}$ where $\delta_{\text{gap}}$ is the gap in the *full* matrix $A_{\text{block}} + E$, not the ideal $A_{\text{block}}$. By Weyl's inequality, this gap is at least $\gamma - 2\|E\|_2$. If $\|E\|_2 > \gamma/2$, the gap collapses and Davis-Kahan gives no bound. This means T2 has an *unstated assumption*: $\delta < \gamma/2$. Without this, the proof is incomplete. The missing assumption restricts T2 to instances where the perturbation is already small relative to the gap — the easy cases.

**ATTACK T2-2: The perturbation model $A = A_{\text{block}} + E$ doesn't exist for real instances [SERIOUS]**

For real MIPLIB instances, there is no canonical $A_{\text{block}}$. The decomposition into block-diagonal + coupling is *itself the output* of spectral analysis. Thus $\delta = \|E\|_F$ and $\gamma$ both depend on the chosen partition. T2 becomes: "for any partition, if you call the coupling part $E$, then..." But the partition is what you're trying to choose. T2 doesn't predict which partition is good — it describes a quantity computable only after partition selection. The circularity means $\delta$ and $\gamma$ in T2's statement refer to the full Laplacian's spectrum, but the proof requires them to refer to the ideal $A_{\text{block}}$'s spectrum. These are different objects.

**ATTACK T2-3: Rounding analysis requires distributional assumptions [MODERATE]**

The proof chain: Davis-Kahan → eigenspace error $O(\delta/\gamma)$ → rounding → misclassification rate $O(\delta^2/\gamma^2)$. The squaring comes from the spectral clustering rounding step (e.g., Lei & Rinaldo 2015), which requires: balanced cluster sizes, bounded degree ratios, planted partition model. Constraint matrices have wildly varying row norms, block sizes, and degree distributions. The rounding analysis must state explicit assumptions on partition geometry, or the $O(\delta^2/\gamma^2)$ rate is not guaranteed.

**ATTACK T2-4: $\gamma_2$ in T2 vs. $\lambda_{k+1} - \lambda_k$ for $k > 2$ [SERIOUS]**

For $k$-way partitioning ($k > 2$), Davis-Kahan for the $k$-dimensional eigenspace uses the gap $\lambda_{k+1} - \lambda_k$, not $\lambda_2 - \lambda_1 = \gamma_2$. The proposal acknowledges this in passing but defines Feature 2 as $\delta^2/\gamma_2^2$, using $\gamma_2$. The feature actually validated empirically is *not* the quantity T2's proof bounds. Either: (a) the feature should use $\lambda_{k+1} - \lambda_k$ (but $k$ is unknown a priori), or (b) T2 is restricted to $k=2$ (losing multi-block applicability).

**ATTACK T2-5: $\kappa^4$ makes T2 useless — why include it? [MODERATE]**

A bound evaluating to $10^{30}$ on typical instances does not motivate; it discourages. If the motivation is "δ²/γ² is the right predictor," state it as a conjecture and test empirically. The full T2 with $\kappa^4$ adds nothing that the conjecture doesn't capture, invites attacks T2-1 through T2-4, and wastes 2–3 pages of proof that could be spent on census analysis. This is a strategic error that consumes reviewer goodwill.

### 1.5 Propositions F1 and F2

**ATTACK F1-1: Eigenvector-based features are NOT automatically permutation-invariant [MODERATE]**

F1 claims all 8 features are permutation-invariant. Eigenvalues are permutation-invariant, but half the features (4: Fiedler localization, 6: coupling energy, 7: separability, and implicitly 2: δ²/γ²) depend on eigenvectors. Eigenvectors are permutation-equivariant, not invariant. For Feature 4, the entropy $H = -\sum p_j \log p_j$ with $p_j = v_{2,j}^2 / \|v_2\|^2$ IS invariant (symmetric function of entries), but this must be verified feature-by-feature. Feature 7 (Silhouette on eigenvector rows) is invariant because it uses pairwise distances, which are symmetric. Feature 6 ($\delta^2$) is invariant because the Frobenius norm is permutation-invariant. The claim is almost certainly correct, but the proof must be explicit for each feature, not hand-waved as "eigenvalues are invariant."

**ATTACK F2-1: Equilibration does NOT eliminate scaling sensitivity [MODERATE]**

F2 analyzes sensitivity to column scaling $A \to AD$. After Ruiz equilibration, $\kappa(\tilde{A})$ is bounded but features of $\tilde{A}$ still depend on the original scaling through $D_r, D_c$. Two instances identical up to diagonal scaling produce the same equilibrated $\tilde{A}$ only if Ruiz converges to the same fixed point — which it doesn't guarantee (converges to a neighborhood). The bound $\kappa(D) < 10$ (algorithms line 79) is empirical, not theoretical. H5 (ICC across equilibrations) tests this empirically, but the paper should not claim theoretical scaling invariance.

**ATTACK F2-2: F2 only analyzes $\gamma_2$, not all 8 features [MODERATE]**

F2 bounds scaling sensitivity for $\gamma_2$ specifically. What about the other 7? Feature 4 (Fiedler localization entropy) depends on $|v_i|^2$; under row scaling $A \to DA$, the Fiedler vector changes and entropy can shift dramatically if scaling concentrates mass. No bound is provided for features 3–8. "Scaling-sensitivity analysis" that covers 1 of 8 features is incomplete.

---

## 2. Algorithm Attacks

### 2.1 Numerical Instability

**ATTACK ALG-1: Shift-invert Lanczos on near-singular Laplacians — silent errors [SERIOUS]**

The Laplacian $L_H$ has a guaranteed zero eigenvalue. Shift-invert with $\sigma = -10^{-6}$ (Algorithm 2.1, line 11) factors $L_H + 10^{-6}I$. For Laplacians with $\lambda_2 \approx 10^{-10}$ (near-disconnected hypergraphs), this matrix has condition number $\approx \lambda_n / 10^{-6} \geq 10^{10}$. The LU factorization will be inaccurate. ARPACK may return wrong eigenvalues without convergence failure — a *silent* error. The residual check (lines 23–26) uses $r_i / \max(|\lambda_i|, 1)$, denomininating by 1 when $|\lambda_i| < 1$. For eigenvalues of order $10^{-10}$, this is insensitive: a residual of $10^{-6}$ passes the check even though the relative error in $\lambda_i$ is $10^4$. All 8 spectral features and the partition will be wrong without any warning.

**ATTACK ALG-2: LOBPCG fallback unreliable for clustered eigenvalues [MODERATE]**

When bottom $k$ eigenvalues are near-degenerate (nearly-disconnected hypergraphs with $k$ components), LOBPCG converges slowly or to the wrong eigenspace. The Jacobi preconditioner ($M^{-1} = 1/\max(\text{diag}(L_H), 10^{-10})$) is a poor preconditioner for graph Laplacians with heterogeneous degree distributions. The fallback chain (ARPACK → LOBPCG → shift) may systematically fail on instances where spectral features matter most (borderline block structure with small gaps).

**ATTACK ALG-3: Feature 2 (δ²/γ²) produces ∞ or numerically huge values [MODERATE]**

Algorithm 2.2 (line 7–8) returns $+\infty$ when $\gamma_2 < \epsilon_{\text{gap}}$. A numerically tiny $\gamma_2$ (say $10^{-15}$) that isn't exactly zero produces $\delta^2/\gamma_2^2 \approx 10^{30}$ — a finite but meaningless number that dominates tree splits. The paper claims "XGBoost handles $+\infty$ natively" (line 425), but this is fragile: XGBoost treats $+\infty$ as a very large finite number, which can dominate splits. Logistic regression (listed as a classifier) certainly cannot handle this. If 5–10% of instances produce $+\infty$, the classifier learns a spurious rule: "if δ²/γ² = ∞, predict 'neither'" — which is just the $\gamma_2$ threshold in disguise.

### 2.2 Contradictions Between Laplacian Variants

**ATTACK ALG-4: Two Laplacians, one feature set, no reconciliation [SERIOUS]**

Clique-expansion ($d_{\max} \leq 200$) and incidence-matrix ($d_{\max} > 200$) Laplacians have "different non-trivial spectrum" (line 208). Features 1–8 from the clique-expansion Laplacian are on a *different scale* than from the incidence-matrix Laplacian. The ML classifier sees feature vectors from two different distributions conflated into one dataset. The proposed validation ("Spearman $\rho > 0.85$ on overlap") tests rank correlation, not scale — insufficient because classifier decision boundaries depend on scale.

The hard switch at $d_{\max} = 200$ creates a spurious discontinuity: instances with $d_{\max} = 199$ vs. $d_{\max} = 201$ get features from different Laplacians, potentially with discontinuous values. A classifier trained on mixed-Laplacian features may learn to exploit this artifact.

**ATTACK ALG-5: Clique expansion memory blow-up [MODERATE]**

For $d_{\max} = 200$: $\binom{200}{2} = 19{,}900$ edges per constraint. For 10,000 such constraints, the Laplacian has up to $2 \times 10^8$ nonzeros (~3 GB per the algorithms document). Add ARPACK's LU factorization with fill-in → 6–10 GB total. Combined with SCIP's LP memory → exceeds 16 GB on many laptops. The "laptop feasible" claim is questionable near the $d_{\max}$ threshold.

### 2.3 Failure Modes

**ATTACK ALG-6: Feature 6 (δ²) is circular with the partition [MODERATE]**

$\delta^2 = \|L_H - L_{\text{block}}\|_F^2$ where $L_{\text{block}}$ is the block-diagonal restriction according to the spectral partition — computed from $L_H$'s own eigenvectors. Spectral clustering minimizes a related quantity (normalized cut), so $\delta^2$ is approximately the clustering objective residual. Using this as a classifier feature is like using k-means objective as a feature — it correlates with cluster quality by construction, not because it captures decomposition structure independently.

**ATTACK ALG-7: k-means non-determinism affects all downstream quantities [MODERATE]**

Algorithm 3.1 uses k-means with 10 random restarts. Different seeds → different partitions → different crossing weights → different L3 bounds → different labels. The paper never analyzes partition stability: how much does the crossing weight vary across restarts? If variance is high, the L3 bound is practically meaningless. The evaluation must fix random seeds AND report sensitivity.

**ATTACK ALG-8: $k$ selection is unspecified [SERIOUS]**

The pipeline uses "bottom-$k$ eigenvectors" but never specifies how $k$ is chosen. Features 5, 6, 7 depend on $k$. If $k$ is chosen by eigengap heuristic, then $k$ itself depends on the spectrum, creating circularity: features depend on $k$, $k$ depends on the spectrum, features summarize the spectrum. The eigengap heuristic is unreliable without clear block structure — it may return $k=1$ or $k=n/2$. The census pipeline (Algorithm 6.1) uses $k \in \{2, 5, 10, 20\}$ — so features are computed 4 times with different $k$. Which $k$'s features enter the classifier?

---

## 3. Evaluation Attacks

### 3.1 Misleading Results

**ATTACK EVAL-1: "≥5pp improvement" is hedged to unfalsifiability [MODERATE]**

The abstract claims improvement "≥5pp over syntactic features alone (or honestly reporting the margin if smaller)." The parenthetical escape hatch means the claim can never be falsified. 2pp on a 3-class problem with 500 instances and 60% class imbalance is within noise. The paper must commit to a firm threshold or accept a null result is possible.

**ATTACK EVAL-2: Class imbalance destroys minority-class statistical power [SERIOUS]**

With 60–75% "neither," 500 instances yield ~325 "neither" vs. ~100 "DW" vs. ~75 "Benders." In 5-fold CV, each test fold has ~15–20 minority-class instances. McNemar's test on 15–20 discordant pairs has very low power. The mitigation (SMOTE, class-weighted loss) doesn't fix the *test set* sample-size problem. Power analysis should target the minority class, not the overall sample.

**ATTACK EVAL-3: 500 instances for 54 experimental cells is underpowered [MODERATE]**

6 feature configs × 3 top-k × 3 classifiers = 54 cells. After Holm-Bonferroni over 15 pairwise comparisons, per-comparison α ≈ 0.003. McNemar's at α = 0.003 on 100 test instances with ~15% discordant pairs requires |discordant| ≥ 15 with OR ≥ 2.5. The paper's own power analysis assumes 40% discordance rate — plausible only if one feature set is dramatically better. If improvement is modest, discordance ≈ 10–15%, giving 50–75 discordant pairs — insufficient after correction.

### 3.2 Label Quality

**ATTACK EVAL-4: Ground-truth labels reflect software quality, not structure [SERIOUS]**

Labels are determined by which of {SCIP, GCG, SCIP-Benders} produces the best dual bound. But the "best" method depends on implementation maturity, parameter tuning, and software version. GCG has 15+ years of DW-specific development; SCIP's Benders is more recent. If GCG wins on 60% of structured instances, this reflects engineering asymmetry, not DW's structural superiority. The oracle learns to predict software behavior. Version-pinning doesn't help because the software-quality-to-structure relationship changes with releases.

**ATTACK EVAL-5: 30% label instability across cutoffs [MODERATE]**

Labels computed at T ∈ {60, 300, 900, 3600}s with majority vote. Decomposition methods need more warmup time. An instance labeled "neither" at T=300 may flip to "DW" at T=3600 — and the 3600s label may flip at T=36000. The target of 70% stability means 30% labels are known unreliable. Training on 30% noisy labels (beyond the ~20% tolerance of random forests for binary classification, worse for 3-class) substantially degrades accuracy. Results should be framed as "oracle for T=900s decisions," not "oracle for decomposition amenability."

**ATTACK EVAL-6: Dual bound improvement ≠ practical usefulness [MODERATE]**

Labels use dual bound improvement, not solve time or gap closure. An instance where DW improves dual by 0.1% in 900s while monolithic SCIP closes the gap in 30s is labeled "DW-amenable" even though DW is useless. No "decomposition overhead" filter exists: if monolithic SCIP solves within the cutoff, decomposition is unnecessary.

### 3.3 Sample Size

**ATTACK EVAL-7: Per-structure-type strata are paper-thin [MODERATE]**

500 instances with 5 structure types: ~80 block-angular, ~60 BBD, ~40 staircase, ~30 dual-block-angular, ~290 none. In 5-fold CV, the staircase test fold has ~8 instances, dual-block-angular has ~6. Reporting "accuracy on Benders-amenable staircase instances" based on 3–4 examples is statistically meaningless. Confidence intervals will be enormous.

### 3.4 Spectral = Proxy for Density?

**ATTACK EVAL-8: Spectral gap correlates with density — you may be predicting density [SERIOUS]**

$\gamma_2$ of a graph Laplacian is bounded by $\gamma_2 \leq n \cdot \bar{d}_{\min}/(n-1)$ and related to edge connectivity. For constraint hypergraphs, $\gamma_2$ correlates with $\text{nnz}/(m \cdot n)$. If $\gamma_2$ is a fancy proxy for density, spectral features are a nonlinear transformation of a syntactic feature. The non-redundancy test H6 (max R² < 0.70) uses linear OLS; R² = 0.69 passes but still indicates substantial redundancy. The real test: *after partialing out density, degree statistics, and size*, do spectral features still predict decomposition benefit? H1 includes partial correlation for spectral ratio but H2 (the main ablation) does not. A random forest regressing $\gamma_2$ on syntactic features with R² > 0.80 would indicate spectral features are redundant regardless of linear R². This experiment is not in the evaluation plan.

This is the existential threat to the paper's thesis. If spectral features proxy density, the contribution reduces to "eigenvalues correlate with constraint matrix density, which everyone already knew."

---

## 4. Novelty Attacks

### 4.1 L3 vs. Standard LP Duality Gap

**ATTACK NOV-1: L3 is Lagrangian relaxation theory in hypergraph notation [SERIOUS]**

The Lagrangian dual gap for relaxing subset $S$ is bounded by $\sum_{i \in S} |\lambda_i^*| \cdot |b_i - a_i^T x^*|$. L3 bounds the gap for partition-based decomposition; "relaxed constraints" are those crossing blocks. The weight $|y_e^*| \cdot (n_e - 1)$ generalizes the scalar case via hyperedge multiplicity. This is a valid generalization, but the intellectual delta over Geoffrion (1974) is *just the hyperedge structure* — essentially saying "each crossing constraint can be relaxed $(n_e-1)$ times, once per block boundary." Is this a lemma or an observation? If L3 requires non-trivial machinery (explicit decomposed dual feasibility construction), it has content. If it follows in 5 lines from LP duality, it's an observation that doesn't warrant "standalone value" framing.

### 4.2 Spectral Features vs. ML4CO

**ATTACK NOV-2: GNN-based approaches learn features automatically [MODERATE]**

ML4CO uses GNNs on the bipartite constraint-variable graph (Gasse et al. 2019, Nair et al. 2020) to learn instance representations end-to-end. GNN message passing implicitly computes spectral features — GNNs with $L$ layers compute features in the span of the first $L$ eigenvectors (Xu et al. 2019 GIN spectral equivalence). Eight hand-crafted spectral features feel like a 2005 approach in 2025. The GRAPH-10 baseline partially addresses this, but a simple 2-layer GCN on the bipartite graph would be a more direct comparison. Without it, a reviewer asks: "why not just use a GNN?"

The interpretability defense is valid for JoC, and the L3 connection ("spectral features have formal link to decomposition bounds; GNN features don't") is the strongest counter. But this must be stated forcefully.

### 4.3 Census Novelty

**ATTACK NOV-3: GCG already detects structure for all of MIPLIB [MODERATE]**

GCG's detection loop classifies structure types on every input. The "first decomposition census" is misleading if GCG's team has applied detection to MIPLIB internally. The novelty is in: (a) cross-method comparison (DW + Benders, not just DW), (b) spectral annotations, (c) a published machine-readable dataset. If (a) reduces to "we also ran SCIP-Benders on everything," census novelty is primarily the published format — a community service, not a research contribution.

**ATTACK NOV-4: T2's proof technique is textbook composition [MODERATE]**

Davis-Kahan (textbook), eigenspace→partition rounding (textbook spectral clustering), partition quality→LP gap (L3, itself close to textbook). Each step is standard. Novelty is in composition applied to MIP decomposition — legitimate but thin for a theory venue. Fine as JoC motivation.

---

## 5. Scope Attacks

### 5.1 Reformulation vs. Algorithm Selection

**ATTACK SCOPE-1: The distinction is philosophically interesting but practically irrelevant [MODERATE]**

Reformulation selection is framed as "strictly harder" because it changes the feasible region. But practitioners want the fastest path to a solution. AutoFolio already selects among solver *configurations* including decomposition-enabling modes (GCG as a SCIP mode). The "reformulation selection" problem exists in AutoFolio's configuration space; calling it by a new name doesn't make it new. The paper should cite AutoFolio's ability to select decomposition configurations and explain why a specialized spectral oracle outperforms the generic approach.

### 5.2 Why Not AutoFolio Directly?

**ATTACK SCOPE-2: AutoFolio + SPEC-8 is the natural baseline [SERIOUS]**

Instead of building a custom oracle, add 8 spectral features to AutoFolio's ~150 instance features and let AutoFolio select among {SCIP, GCG, SCIP-Benders}. If AutoFolio + SPEC-8 ≈ custom oracle accuracy, the "oracle" contribution is zero — spectral features are the only contribution, publishable as a feature engineering paper with AutoFolio as consumer. This experiment MUST be run. Its absence is a glaring gap.

### 5.3 Only 10–25% Have Block Structure

**ATTACK SCOPE-3: The oracle helps on ~150 of 1,065 instances [MODERATE]**

Bergner et al. (2015): ~10–25% of MIPLIB has exploitable block structure. The oracle's recommendation differs from "do nothing" on ~250 instances at most, of which the DW-vs-Benders decision is relevant for ~100. Building 25K LoC for 100 instances is a questionable investment. The paper should quantify: on how many instances does the oracle's recommendation *change the outcome* (recommended method produces better bound than monolithic SCIP)?

### 5.4 Lagrangian Relaxation is Missing

**ATTACK SCOPE-4: Three methods claimed, two evaluated [MODERATE]**

The problem statement lists Benders, DW, and Lagrangian relaxation. The evaluation uses GCG (DW) and SCIP-Benders. There is no external Lagrangian solver; Amendment 2 mandates external baselines. The architecture doesn't include a Lagrangian implementation. The three-method framing collapses to binary {Benders, DW, neither}. The paper must either implement Lagrangian relaxation or honestly reduce scope.

---

## 6. Contradictions

### 6.1 L3 Requires LP Duals But Is Pitched as Preprocessing

L3's crossing weight requires $y^*$ from the LP relaxation. The oracle is a "preprocessing layer" running before solving. If you solve the LP to get $y^*$, you already have dual information (reduced costs, basis structure, constraint activity) that directly reveals decomposition structure — making spectral features redundant. The verification framework's simulated Reviewer 3 calls this out explicitly. There is no response in the paper.

### 6.2 Equilibration Fixes Scaling But Laplacian Depends on Equilibration

F2 analyzes scaling sensitivity; three equilibration methods are presented. H5 tests stability (ICC ≥ 0.85). But the pipeline makes a hard choice of one equilibration. If ICC < 0.85 for some features (conceded possible: "any feature with ICC < 0.60 is flagged"), then different equilibration settings → different recommendations. Two users get different oracle outputs. This contradicts reproducibility claims.

### 6.3 Census Claims Completeness But 60% Will Time Out

Title: "Complete MIPLIB 2017 Decomposition Census." Depth check S5: "Census with 60% timeouts is not a census." Spectral annotations ARE complete (1,065 instances, ~9 hours). Decomposition evaluations are NOT (500 instances with timeouts). "Complete" is false for decomposition results. Change to "systematic" or "comprehensive."

### 6.4 T2 Is "Motivational" But L3-sp Depends on It

T2 is demoted to "motivational" (Amendments 1, 4). But L3-sp (spectral partition crossing weight bound) appears derived from T2's proof chain (Davis-Kahan → rounding → L3). If L3-sp depends on T2, then T2 is load-bearing and cannot be "motivational." If L3-sp is independent, it needs a self-contained proof. The proof dependency graph must clarify this — currently ambiguous.

### 6.5 $\gamma_2$ in T2 vs. $\lambda_{k+1} - \lambda_k$ in the Proof vs. $\gamma_2$ in the Feature

T2 states the bound using $\gamma^2$ (= $\gamma_2^2 = \lambda_2^2$). The proof requires Davis-Kahan for the $k$-dimensional eigenspace, using $\lambda_{k+1} - \lambda_k$. Feature 2 uses $\delta^2/\gamma_2^2$. These are three different quantities for $k > 2$. Either: (a) state $\gamma_2$ as proxy for $\lambda_{k+1} - \lambda_k$ with explicit discussion, or (b) there is an error.

### 6.6 T2 Is "Motivational" But δ²/γ² Is the Core Feature

If T2 is demoted and acknowledged as vacuous, then Feature 2 ($\delta^2/\gamma^2$) loses theoretical justification. The paper argues "T2 explains *why* δ²/γ² is good" while saying "T2 is vacuous." A reviewer asks: "If the theorem justifying your feature is vacuous, why believe the feature is principled rather than ad hoc?" Needs a crisper narrative.

---

## 7. Five Killer Questions

### KQ1: "How does L3 differ from the standard result that the Lagrangian dual gap is bounded by dual weight on relaxed constraints (Geoffrion 1974)? State precisely the intellectual contribution beyond Geoffrion."

*Why this kills:* If the authors cannot articulate a crisp distinction — not just "we use hypergraphs" but a genuine structural insight — L3 is a known result in costume. The paper loses its only theorem. The verification framework (RF-F2) flags this as FATAL.

*Expected weak answer:* "L3 generalizes to hyperedges with the $(n_e-1)$ factor." *Killer follow-up:* "So the generalization is a counting argument. Is there a matching lower bound? Can you construct an instance where $(n_e-1)$ is tight?"

### KQ2: "You compute spectral features from the Laplacian (~30s). You also solve the LP to get shadow prices for L3 (~60–300s). After solving the LP, you have dual information, reduced costs, basis structure, and constraint activity. Why should I trust 8 spectral features over the rich LP-derived information you're computing anyway?"

*Why this kills:* The LP solve provides strictly more information than the eigendecomposition. If L3 requires LP duals, spectral features become redundant with LP-derived features. Defense must be: (a) spectral features are faster than LP solving (true for hard LPs) or (b) complementary to LP information. Neither is currently argued.

### KQ3: "Your two Laplacian variants (clique-expansion vs. incidence-matrix) have different spectra. Your classifier trains on mixed features. Have you verified distributions are comparable? Show the performance of a classifier trained on clique-expansion-only instances vs. the mixed dataset."

*Why this kills:* Exposes the Laplacian switch as a confound. The classifier may learn a mixture model where "which Laplacian was used" is a hidden variable correlated with $d_{\max}$ and instance structure. This experiment is not in the evaluation plan.

### KQ4: "After controlling for density, average degree, and size — all syntactic features — what residual predictive power do your 8 spectral features retain? Show the partial correlation."

*Why this kills:* The density-proxy attack made specific. If partial correlations drop below 0.2 after controlling for syntactic features, spectral features add marginal information. H6 tests linear redundancy (R²); the real threat is nonlinear redundancy ($\gamma_2 \approx f(\text{density}, \text{size})$). A random forest regression with R² > 0.80 would indicate redundancy not captured by H6.

### KQ5: "Your labels have 30% noise across cutoffs. What is the theoretical accuracy ceiling for a classifier trained on labels with 30% noise? Have you computed the Bayes error rate? If the noise ceiling is 75% and your classifier achieves 70%, are you measuring signal or memorizing noise?"

*Why this kills:* Questions whether the ML pipeline measures anything real. With 30% label noise and class imbalance, a random forest can achieve 65–70% by memorizing noise correlations. The evaluation reports label stability (§4.3) but never connects it to performance bounds. Needs a theoretical or simulation-based analysis of how label noise affects achievable accuracy.

---

## 8. Severity Classification

### FATAL (paper cannot be published without resolution)

| ID | Finding | Attack | Defense Path |
|----|---------|--------|-------------|
| **F-1** | L3 may be a trivial restatement of Lagrangian duality | L3-2, KQ1 | Prove L3 requires non-trivial machinery (explicit dual feasibility construction); articulate delta over Geoffrion 1974 |
| **F-2** | L3 direction ambiguity for non-converged methods | L3-4 | Restrict L3 to converged bounds; document which census entries satisfy this |
| **F-3** | L3-sp dependency on T2 contradicts T2's "motivational" demotion | §6.4 | Provide independent proof of L3-sp via Cheeger inequality, not through T2 |

### SERIOUS (requires significant revision; any two unresolved → reject)

| ID | Finding | Attack | Defense Path |
|----|---------|--------|-------------|
| **S-1** | L3 dual degeneracy: which optimal dual? | L3-1 | Prove L3 for all optimal duals; or specify minimum-norm dual |
| **S-2** | L3 requires LP duals, contradicting preprocessing pitch | L3-5, KQ2 | Reframe L3 as retrospective metric; show spectral features are useful *without* L3 |
| **S-3** | L3-C DW assumes converged CG; census doesn't guarantee this | L3-C-2 | Restrict L3-C DW validation to converged runs; report convergence rate |
| **S-4** | T2 missing assumption δ = O(γ) | T2-1 | Add assumption explicitly; discuss percentage of MIPLIB satisfying it |
| **S-5** | T2 uses $\gamma_2$ but proof needs $\lambda_{k+1} - \lambda_k$ | T2-4, §6.5 | Use $\lambda_{k+1} - \lambda_k$ in feature; or restrict T2 to $k=2$ |
| **S-6** | Two Laplacian variants produce incompatible features | ALG-4, KQ3 | Normalize features; or use incidence-matrix Laplacian everywhere |
| **S-7** | Spectral features may proxy density | EVAL-8, KQ4 | Run partial-correlation analysis after controlling for syntactic features |
| **S-8** | AutoFolio + SPEC-8 baseline not included | SCOPE-2 | Run experiment; position oracle relative to AutoFolio |
| **S-9** | Class imbalance destroys minority-class power | EVAL-2 | Power analysis on minority class; consider binary formulation |
| **S-10** | Ground-truth labels reflect software, not structure | EVAL-4 | Discuss limitation; test label stability across solver versions |
| **S-11** | Silent eigenvalue errors from shift-invert | ALG-1 | Use absolute tolerance for small eigenvalues; validate on synthetic instances |
| **S-12** | $k$ selection unspecified | ALG-8 | Document protocol; report sensitivity analysis |

### MODERATE (should be addressed; individually non-fatal)

| ID | Finding | Attack |
|----|---------|--------|
| M-1 | $(n_e-1)$ factor possibly loose by $k-1$ | L3-3 |
| M-2 | L3-sp vacuous when $d_{\max}$ large | L3-sp-1 |
| M-3 | L3-sp tautological when $\gamma_2$ small | L3-sp-2 |
| M-4 | L3-C Benders requires simplex LP solve | L3-C-1 |
| M-5 | L3-C DW duals are scale-dependent | L3-C-3 |
| M-6 | T2 perturbation model doesn't exist for real instances | T2-2 |
| M-7 | T2 rounding requires distributional assumptions | T2-3 |
| M-8 | T2 inclusion is a strategic error | T2-5 |
| M-9 | F1 proof incomplete for eigenvector-based features | F1-1 |
| M-10 | F2 equilibration convergence is approximate | F2-1 |
| M-11 | F2 only analyzes 1 of 8 features | F2-2 |
| M-12 | LOBPCG fallback unreliable for clustered eigenvalues | ALG-2 |
| M-13 | Clique expansion memory blow-up near threshold | ALG-5 |
| M-14 | Feature 6 (δ²) circular with partition | ALG-6 |
| M-15 | k-means non-determinism affects L3 | ALG-7 |
| M-16 | "≥5pp" claim hedged to unfalsifiability | EVAL-1 |
| M-17 | 500 instances underpowered for 54 cells | EVAL-3 |
| M-18 | 30% label noise limits accuracy ceiling | EVAL-5, KQ5 |
| M-19 | Dual bound ≠ practical usefulness | EVAL-6 |
| M-20 | Per-structure strata paper-thin | EVAL-7 |
| M-21 | No GNN feature comparison | NOV-2 |
| M-22 | Census completeness overstated | §6.3 |
| M-23 | Oracle effective coverage ~150/1065 | SCOPE-3 |
| M-24 | Lagrangian relaxation scope gap | SCOPE-4 |
| M-25 | Identity crisis: paper tries to be 3 things | NOV-4 |

### COSMETIC (nice to fix; won't affect acceptance)

| ID | Finding | Attack |
|----|---------|--------|
| C-1 | Feature 2 = ∞ handling fragile | ALG-3 |
| C-2 | $y^*_e$ notation: constraint vs. hyperedge index | — |
| C-3 | $\gamma$ vs. $\gamma_2$ notation inconsistency | — |
| C-4 | "Reformulation selection" framing debatable | SCOPE-1 |
| C-5 | T2 theoretical futility threshold should be a remark | §6.6 |

---

## Summary

**Total findings:** 3 FATAL, 12 SERIOUS, 25 MODERATE, 5 COSMETIC.

**Three existential threats:**

1. **L3 triviality** (F-1): If L3 is Geoffrion 1974 in hypergraph notation, the paper has no theorem. Defense: prove L3 requires non-trivial construction (explicit decomposed dual feasibility argument, not just weak duality).

2. **L3 direction/convergence** (F-2): If the inequality fails for non-converged methods, L3 doesn't apply to census data. Defense: restrict L3 to converged bounds; document which census entries satisfy this.

3. **L3-sp ↔ T2 dependency** (F-3): If L3-sp needs T2's proof chain, T2 is load-bearing and cannot be "motivational." Defense: provide independent proof of L3-sp via Cheeger inequality, not through T2.

**The paper is publishable if and only if all three FATALs are resolved and at least 8 of 12 SERIOUS findings are addressed.** Resolution of the FATALs likely requires 1–2 weeks of focused mathematical work. The SERIOUS findings are addressable within the evaluation framework already designed.

*This report is intentionally adversarial. Its purpose is to make the final paper bulletproof, not to discourage the authors. Every attack has a defense; the paper's job is to preempt each one.*
