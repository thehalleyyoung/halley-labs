# Verification Framework — Spectral Decomposition Oracle

> **Role**: Verification Chair
> **Phase**: Theory stage quality gates
> **Date**: 2025-07-22
> **Status**: ACTIVE — governs CONTINUE/ABANDON verdict for theory artifacts
> **Artifacts under review**: `approach.json`, `paper.tex`

---

## 0. Preamble

This document defines the complete verification protocol for the theory stage of the Spectral Decomposition Oracle project. The theory stage must produce two artifacts — `approach.json` (structured theory document) and `paper.tex` (publication-quality LaTeX, >50KB) — both containing: L3 (main contribution), L3-sp (specializations), L3-C Benders/DW, T2 (motivational), F1, F2, algorithms, complexity analysis, and evaluation plan.

**Baseline scores from depth check**: V5 / D4 / BP3 / L6. Composite 4.5/10.
**Probability estimates**: P(JoC) ≈ 0.55, P(best-paper) ≈ 0.03, P(abandon) ≈ 0.25.

The verification framework must be strict enough to prevent weak artifacts from consuming implementation effort, yet calibrated to the Amendment E framing (computational study, not theorem paper) so that we do not apply theory-paper standards to a computational-study paper.

---

## 1. Theory Quality Rubric

Each component is scored 1–10. Descriptors for key thresholds:

- **1–3**: Fundamentally flawed, incomplete, or incorrect. Would cause desk rejection.
- **4–5**: Serious gaps but salvageable with major revision. Borderline for continuation.
- **6–7**: Adequate for submission. Minor gaps that reviewers might note but would not reject over.
- **8–9**: Strong. Exceeds expectations for the venue. Reviewer-proof on this dimension.
- **10**: Exceptional. Could anchor a paper at a stronger venue on this dimension alone.

---

### 1.1 Definitions (score 1–10)

**What we check:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| D1. Precision | 0.30 | Every mathematical object (hypergraph Laplacian, spectral features, block partition, crossing weight) has an unambiguous formal definition. No object is used before it is defined. |
| D2. Ambiguity audit | 0.25 | A hostile reviewer cannot find two valid but contradictory interpretations of any definition. Particular danger zones: (a) "normalized" Laplacian (which normalization?), (b) "hyperedge weight" (binary vs. coefficient-magnitude), (c) "dual bound" (LP relaxation dual vs. Lagrangian dual vs. decomposition-specific bound). |
| D3. Notation consistency | 0.20 | The same symbol is never used for two objects. A notation table is provided. Greek letters, subscripts, and overloaded operators are tracked. Key check: $\gamma$ (spectral gap) vs. $\gamma$ (any other use); $\delta$ (coupling norm) vs. $\delta$ (tolerance). |
| D4. Citation vs. reinvention | 0.15 | Standard definitions (LP relaxation, Benders decomposition, DW reformulation, spectral clustering, Davis-Kahan theorem) cite canonical sources. The paper does not re-derive textbook results without attribution. Particular check: Bolla (1993) incidence-based hypergraph Laplacian must be cited, not presented as novel. |
| D5. Completeness | 0.10 | Every object that appears in a theorem statement or algorithm is defined somewhere earlier in the document. No "let X be as defined in [Author, Year]" without making the definition self-contained for a reader without access to that reference. |

**Scoring protocol:**
- Start at 10.
- Deduct 1 for each ambiguous definition a reviewer could exploit.
- Deduct 2 for each object used before definition.
- Deduct 3 for notation collision (same symbol, two meanings).
- Deduct 1 for each reinvented standard definition without citation.
- Floor at 1.

---

### 1.2 Proofs (score 1–10)

**What we check:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| P1. Completeness | 0.30 | No proof step uses "clearly," "it is easy to see," "straightforward," or "by a standard argument" to skip a non-trivial step. Every inequality has a named justification (Cauchy-Schwarz, triangle inequality, Davis-Kahan, etc.). |
| P2. Assumption tracking | 0.25 | Every theorem/lemma/proposition has an explicit list of assumptions. For L3: what properties must the partition satisfy? For T2: what must hold about the perturbation $E$? Are assumptions testable on actual MIPLIB instances? |
| P3. Gap identification | 0.20 | Where full proofs are tedious but straightforward (e.g., verifying that the Bolla Laplacian is positive semidefinite), the gap is explicitly flagged with a proof sketch and a reference. This is better than a hidden gap. |
| P4. Logical structure | 0.15 | The proof dependency graph is acyclic: Lemma L3 → L3-C (Benders) → L3-C (DW) → T2 (uses L3 as a step). No circular dependencies. No theorem is used before it is proved. |
| P5. Error resilience | 0.10 | If an error were found in T2 (the motivational result), does the paper survive? The answer must be "yes" because L3 is independent and the empirical program is independent. This must be structurally verifiable: L3's proof must not invoke T2. |

**Scoring protocol:**
- Start at 10.
- Deduct 3 for any incorrect proof step (wrong inequality direction, missing case, etc.).
- Deduct 2 for each hidden gap (step claimed trivial but actually requires argument).
- Deduct 1 for each "clearly" or "it is easy to see" that skips a substantive step.
- Deduct 2 for circular dependency in proof structure.
- Deduct 1 for each unstated assumption.
- Floor at 1.

**Critical proofs requiring line-by-line verification (§4.1):**
1. Lemma L3 (partition-to-bound bridge) — THE main theorem. Every step verified.
2. L3-C Benders specialization — reduced-cost weighting derivation.
3. L3-C DW specialization — linking-constraint dual weighting derivation.
4. T2 proof chain: Davis-Kahan application → rounding analysis → L3 invocation.

---

### 1.3 Algorithms (score 1–10)

**What we check:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| A1. Pseudocode clarity | 0.25 | Each algorithm has pseudocode that a graduate student in optimization could implement from scratch without consulting additional sources. Inputs, outputs, and invariants are specified. |
| A2. Complexity bounds | 0.25 | Each algorithm has a stated time and space complexity. The complexity is proved (not just claimed). Key algorithms: (a) hypergraph Laplacian construction, (b) spectral feature extraction, (c) partition construction from eigenvectors, (d) crossing-weight computation for L3. |
| A3. Numerical stability | 0.25 | For the spectral engine: (a) what happens when the spectral gap is near zero? (b) what happens with degenerate eigenvalues? (c) is the Laplacian construction numerically stable for ill-conditioned constraint matrices? (d) is shift-invert Lanczos stable for the problem sizes and condition numbers encountered in MIPLIB? Stability issues must be discussed, not ignored. |
| A4. Theory-algorithm connection | 0.15 | Each algorithm is explicitly connected to the theorem that justifies it. The crossing-weight computation implements L3. The spectral partition implements the rounding step in T2. The futility predictor implements the empirical threshold from the scaling law. |
| A5. Edge cases | 0.10 | Empty partitions, single-block degeneracy, instances with no block structure, instances with zero spectral gap, instances where ARPACK fails to converge — all handled with documented fallback behavior. |

**Scoring protocol:**
- Start at 10.
- Deduct 2 for each algorithm without pseudocode.
- Deduct 2 for each missing or incorrect complexity bound.
- Deduct 2 for unaddressed numerical stability issue that would affect MIPLIB execution.
- Deduct 1 for each missing theory-algorithm connection.
- Deduct 1 for each unhandled edge case.
- Floor at 1.

---

### 1.4 Evaluation Plan (score 1–10)

**What we check:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| E1. Falsifiability | 0.25 | Each hypothesis has a quantitative criterion that could disprove it. "Spectral features help" must have a measurable threshold. The G1-G5 kill gates from the depth check must be honored. |
| E2. Baseline fairness | 0.25 | All baselines use independently maintained implementations (GCG for DW, SCIP-native for Benders — Amendment 2). The syntactic feature baseline uses standard features (density, degree stats, coefficient range) not strawman subsets. AutoFolio comparison uses published feature sets, not degraded versions. |
| E3. Statistical rigor | 0.20 | Appropriate tests specified: paired permutation tests for accuracy differences, Spearman rank correlation for spectral-ratio validation, bootstrap confidence intervals for coverage metrics. Multiple comparisons correction (Bonferroni or Holm) if testing multiple hypothesis simultaneously. Power analysis: ≥20 instances per stratum (Amendment 5). |
| E4. Reproducibility | 0.15 | The evaluation plan is self-contained: instance selection criteria, stratification scheme, time cutoffs, hardware specification, random seeds, and code/data availability statement. A reader could replicate the study from the paper alone. |
| E5. Depth-check compliance | 0.15 | All seven amendments are reflected in the evaluation plan. Specifically: (a) census-first framing (A1), (b) external baselines (A2), (c) honest LoC scope (A3), (d) terminology (A4), (e) stratified evaluation (A5), (f) ablation as core experiment (A6), (g) JoC targeting (A7). |

**Scoring protocol:**
- Start at 10.
- Deduct 3 for any unfalsifiable claim that appears as a contribution.
- Deduct 3 for circular evaluation (testing own implementation against itself).
- Deduct 2 for missing or inappropriate statistical tests.
- Deduct 2 for any depth-check amendment not honored.
- Deduct 1 for each reproducibility gap (missing seed, unspecified hardware, etc.).
- Floor at 1.

---

### 1.5 Paper Structure (score 1–10)

**What we check:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| ST1. Narrative coherence | 0.25 | The paper tells one story: "spectral features from the constraint hypergraph Laplacian provide a principled, continuous characterization of decomposition amenability, validated on the first MIPLIB decomposition census." Every section serves this narrative. |
| ST2. Census-first framing | 0.25 | The census is presented as a primary contribution, not an afterthought. It appears in the abstract, introduction, and contribution list before the spectral features. This is Amendment 1 compliance. |
| ST3. T2 demotion | 0.20 | T2 occupies ≤3 pages. It is labeled "Theoretical Motivation" or "Structural Scaling Analysis." It is never called "the main result" or "bridging theorem." The vacuousness of $C$ is honestly stated. This is Amendment 4 compliance. |
| ST4. Contribution clarity | 0.15 | Each contribution is clearly labeled as: (a) novel to this paper, (b) known but applied in a new context, or (c) known and included for completeness. L3 is novel. The Davis-Kahan application in T2 is known technique, new context. Spectral clustering is known. |
| ST5. Size and density | 0.15 | `paper.tex` exceeds 50KB. The paper has appropriate density for JoC — not padded, not compressed. Figures, tables, and algorithms are substantial, not placeholder. Appendices contain proof details, not main contributions. |

**Scoring protocol:**
- Start at 10.
- Deduct 3 if T2 is positioned as main contribution (fatal misframing).
- Deduct 2 if census is relegated to "supplementary material" instead of primary contribution.
- Deduct 2 if the paper tries to be three things (theory/system/benchmark) without clear prioritization.
- Deduct 1 for each section that does not serve the main narrative.
- Deduct 1 if paper.tex < 50KB.
- Floor at 1.

---

## 2. Minimum Quality Thresholds

### 2.1 CONTINUE Thresholds (all must be met simultaneously)

| Component | Hard Floor | Rationale |
|-----------|-----------|-----------|
| Definitions | ≥ 6 | Notation and definitions are fixable in revision, but scores below 6 indicate systemic problems (objects undefined, notation collisions) that propagate into proofs and algorithms. |
| Proofs | ≥ 7 | L3 is the paper's main theoretical contribution. A proof error in L3 is fatal. The higher bar reflects that proof quality is harder to fix post-hoc than definitions. |
| Algorithms | ≥ 5 | Algorithms can be refined during implementation. Pseudocode imprecision is fixable. But scores below 5 indicate missing algorithms or fundamentally wrong complexity claims. |
| Evaluation Plan | ≥ 7 | The paper is a computational study. The evaluation plan IS the paper. A weak plan means the entire project is misguided. All depth-check amendments must be verifiable in the plan. |
| Paper Structure | ≥ 6 | Structure can be rearranged, but scores below 6 indicate the paper is still framed as the original (rejected) proposal rather than Amendment E. |
| **Composite** | **≥ 6.0** | Weighted average: 0.15·Def + 0.25·Proofs + 0.15·Algo + 0.25·Eval + 0.20·Struct ≥ 6.0 |

### 2.2 Conditional-Continue Zone

If the composite is 5.0–5.9, continuation is permitted ONLY if:
- No single component is below its hard floor by more than 1 point.
- The Proofs score is ≥ 6 (L3 must be correct even if rough).
- A specific remediation plan exists for each component below threshold.
- The remediation can be completed in ≤3 working days.

### 2.3 ABANDON Zone

If any of the following hold, the verdict is ABANDON:
- Composite < 5.0.
- Proofs score < 5 (fundamental correctness issues in L3).
- Evaluation Plan score < 5 (evaluation design is circular or unfalsifiable).
- Two or more components below their hard floors by ≥ 2 points.

---

## 3. Red Flags Checklist

Each red flag is classified as FATAL (triggers immediate ABANDON) or SERIOUS (triggers conditional continue with mandatory remediation).

### FATAL Red Flags

| ID | Red Flag | Detection Method | Why Fatal |
|----|----------|-----------------|-----------|
| RF-F1 | **Proof error in L3 (partition-to-bound bridge)** | Line-by-line verification (§4.1). Construct a 2-block counterexample: if the bound $z_{LP} - z_D \leq w(\text{crossing})$ fails on the counterexample, L3 is wrong. | L3 is the main theoretical contribution. If it is wrong, the paper has no theorem. Unlike T2 (which is motivational), L3 cannot be demoted further. |
| RF-F2 | **L3 is a trivial restatement of LP duality** | Expert review: does L3 follow immediately from strong duality + complementary slackness with zero intellectual content? If L3 reduces to "the LP relaxation gap equals the sum of violated dual constraints weighted by primals," it is a textbook observation, not a lemma. | If L3 is trivial, the paper's sole theoretical contribution is vacuous. The census remains, but a census-only paper is a data paper, not a JoC paper. |
| RF-F3 | **Evaluation design with fundamental circularity** | Check: does the ground-truth labeling (which decomposition method is "best") depend on the same implementation being evaluated? If the oracle learns from SCIP Benders results and is tested on SCIP Benders results without proper holdout, the evaluation is circular. | Amendment 2 compliance is non-negotiable. Circularity was a fatal flaw in the original proposal. |
| RF-F4 | **T2 is still positioned as main contribution** | Check paper structure: is T2 in the abstract's first sentence? Is T2 listed as Contribution 1? Does the introduction spend >1 page on T2 before mentioning the census? | This violates Amendment 1 (census-first restructuring) and Amendment 4 (honest terminology). The depth check panel explicitly rejected the T2-centered framing. |
| RF-F5 | **L3 and L3-C are inconsistent** | Verify: does L3-C (Benders) follow from L3 by specialization to Benders partitions? Does L3-C (DW) follow from L3 by specialization to DW block structure? If L3-C contradicts L3, the theory is internally inconsistent. | Internal consistency is non-negotiable. Contradictory results in the same paper guarantee rejection. |

### SERIOUS Red Flags

| ID | Red Flag | Detection Method | Remediation |
|----|----------|-----------------|-------------|
| RF-S1 | **Spectral feature definitions (F1) lack permutation-invariance proof** | Check: is there a formal statement and proof that the 8 features are invariant to constraint/variable permutation? If only claimed, not proved, this is a gap. | Add proof or proof sketch. This is straightforward (eigenvalues are permutation-invariant) but must be stated. |
| RF-S2 | **F2 (scaling sensitivity) analysis is missing or dishonest** | Check: does F2 analyze what happens when columns of $A$ are scaled? Features based on $\|A\|_F$ are scaling-sensitive. If the paper claims scaling invariance when features are actually scaling-sensitive, this is a misrepresentation. | Correct to honest analysis: which features are scaling-sensitive, which are not, and what preprocessing mitigates sensitivity. |
| RF-S3 | **Algorithm pseudocode is missing for crossing-weight computation** | The crossing-weight computation (implementing L3) must be algorithmic, not just mathematical. If only the formula is given but not the computational procedure, a reviewer will note this. | Add pseudocode with complexity analysis. |
| RF-S4 | **Evaluation plan does not specify all seven amendments** | Cross-check each amendment (A1–A7) against the evaluation plan section of the paper. Any missing amendment is a serious gap. | Add missing amendment compliance. |
| RF-S5 | **Complexity claims are incorrect** | Trace each complexity bound: (a) Laplacian construction is $O(\text{nnz}(A) \cdot d_{max})$ for clique expansion or $O(\text{nnz}(A))$ for Bolla — verify. (b) Eigensolve is $O(k \cdot \text{nnz}(L) \cdot \text{iters})$ for Lanczos — verify iteration count claim. | Correct the bounds and re-derive. |
| RF-S6 | **No treatment of numerical stability in eigensolve** | ARPACK shift-invert on near-singular systems can fail silently (returning wrong eigenvalues). If the paper does not discuss convergence checks, residual monitoring, or fallback strategies, MIPLIB execution will produce silent errors. | Add stability analysis section or at minimum a paragraph on convergence monitoring. |
| RF-S7 | **Spectral gap near zero is not handled** | If $\gamma \approx 0$, the ratio $\delta^2/\gamma^2$ diverges, features are undefined, and the partition is meaningless. The paper must specify behavior for $\gamma < \epsilon_{machine}$. | Define fallback: if $\gamma < \tau_{min}$, report "no block structure detected" and skip spectral features for this instance. |
| RF-S8 | **The paper claims >30K LoC of novel code** | Amendment 3 mandates 25–30K LoC. If the paper inflates beyond this, credibility is compromised. | Correct LoC count to honest estimate. |

---

## 4. Verification Protocol

### 4.1 Proof Checking

**Priority 1 (line-by-line, every step):**

1. **Lemma L3 (partition-to-bound bridge).**
   - Verify the setup: LP $\min\{c^T x : Ax \geq b, x \geq 0\}$, partition $\Pi = \{B_1, \ldots, B_k\}$ of constraint indices.
   - Verify the decomposed dual construction: how are dual variables constrained in the decomposed problem?
   - Verify the crossing-weight definition: which hyperedges cross? How is "weight" defined (dual variable magnitude × coefficient magnitude, or something else)?
   - Verify the main inequality: $z_{LP} - z_D(\Pi) \leq \sum_{e \in \text{crossing}(\Pi)} w(e)$.
   - Check direction: is it $z_{LP} - z_D$ or $z_D - z_{LP}$? For a minimization LP, $z_D \leq z_{LP}$, so the gap is $z_{LP} - z_D \geq 0$.
   - **Counterexample test**: construct a 3-constraint, 2-variable LP with known optimal and decomposed dual. Verify the bound holds. Then construct a near-tight example to check that the bound is not vacuously loose.

2. **L3-C Benders.**
   - Verify: starting from L3, specialize to Benders partition (complicating variables vs. subproblem variables).
   - Check: does the "crossing weight" reduce to something involving reduced costs or Benders cut coefficients?
   - Verify: the specialization is a strict consequence of L3, not a separate result with independent assumptions.

3. **L3-C DW (Dantzig-Wolfe).**
   - Verify: starting from L3, specialize to DW block structure (linking constraints vs. block constraints).
   - Check: does the "crossing weight" reduce to dual variables on linking constraints?
   - Verify: the specialization is a strict consequence of L3.

4. **Proposition T2 (spectral scaling law).**
   - Verify the Davis-Kahan application: is $\sin\Theta(\hat{V}, V) \leq \|E\|_F / \gamma$ correctly invoked? Check the exact form of Davis-Kahan used (there are multiple versions with different norms and assumptions).
   - Verify the rounding analysis: eigenspace perturbation → partition misclassification. Is the $O(\delta^2/\gamma^2)$ rate correct, or should it be $O(\delta/\gamma)$? (Squared comes from rounding, which squares the angle error — verify this.)
   - Verify the L3 invocation: does the misclassification rate correctly translate to crossing weight via L3?
   - Verify the constant: $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$ — where does each factor come from?
   - **Vacuousness check**: plug in $\kappa = 10^3$ (common for big-M), $k = 5$, $\|c\|_\infty = 10^3$. Confirm the bound is indeed vacuous (>$10^{15}$). This must be honestly stated.

**Priority 2 (proof sketch verification, major steps only):**

5. **F1 (permutation invariance).**
   - Verify: eigenvalues of a symmetric matrix are invariant to simultaneous row/column permutation. This is standard but must be stated.
   - Check: are all 8 features functions of eigenvalues only, or do some depend on eigenvectors (which ARE permutation-sensitive up to sign/rotation)? Eigenvector localization entropy depends on eigenvector entries — is this actually permutation-invariant?

6. **F2 (scaling sensitivity).**
   - Verify: which features change under column scaling $A \to A D$ where $D$ is diagonal? Spectral gap changes. Algebraic connectivity changes. Under what preprocessing is the analysis valid?

### 4.2 Algorithm Checking

**Algorithms requiring trace-on-example:**

1. **Hypergraph Laplacian construction.**
   - Trace on a 3-constraint, 4-variable example with known structure.
   - Verify: Bolla incidence Laplacian and clique-expansion Laplacian are both computed correctly.
   - Verify: the Laplacian is positive semidefinite (all eigenvalues ≥ 0).
   - Verify: the zero eigenvalue has the correct multiplicity (equals number of connected components).

2. **Spectral feature extraction.**
   - Trace on the same example.
   - Compute all 8 features by hand. Verify they match the algorithm's output.
   - Check: what happens when the spectral gap is zero? Features should degrade gracefully (return 0 or NaN with a documented convention).

3. **Crossing-weight computation (L3 implementation).**
   - Trace on the 3-constraint, 2-variable example used for L3 counterexample checking.
   - Verify: the computed crossing weight matches the theoretical bound.

4. **Spectral partition algorithm (k-way spectral clustering).**
   - Trace on a 6-constraint example with known 2-block structure.
   - Verify: the partition recovers the planted blocks.
   - Verify: the algorithm handles degenerate cases (zero spectral gap, repeated eigenvalues).

### 4.3 Consistency Checking

| Check | What to verify |
|-------|---------------|
| **approach.json ↔ paper.tex** | Every theorem/lemma/definition in approach.json appears in paper.tex with identical statement. No theorem is "upgraded" (weakened assumptions, stronger conclusions) between the two documents without explicit justification. |
| **Notation consistency** | Build a notation table from paper.tex. Verify no symbol collision. Cross-reference with approach.json. |
| **Proof dependency graph** | Extract the dependency graph from both documents. Verify they match. Verify acyclicity. Expected graph: F1, F2 → features; L3 → L3-C Benders, L3-C DW; L3, Davis-Kahan → T2. |
| **Amendment compliance** | Verify all 7 amendments from depth_check.md §4 are honored in both documents. Use the checklist in §4.4 below. |
| **Kill gate compliance** | Verify all 5 kill gates (G1–G5) from depth_check.md §7 are reflected in the evaluation plan. |
| **Evaluation metrics match** | Verify that every metric in the evaluation plan (§3 of the problem statement, §5 of the depth check) appears in paper.tex's experimental section. |

### 4.4 Completeness Checking — Amendment Compliance

| Amendment | Verification Criterion | Pass/Fail |
|-----------|----------------------|-----------|
| A1: Census-first restructuring | Census appears as contribution P1 or P2 in paper structure. Census is in the abstract. Paper title includes "census" or equivalent. | |
| A2: External baselines | GCG is used for DW. SCIP-native Benders is used. Custom implementation only for Lagrangian, with disclosure. Evaluation is selector-ablation, not method-comparison. | |
| A3: Honest scope | LoC estimate is 25–30K. No claims of >50K. System described as "preprocessing plugin and evaluation framework," not "standalone decomposition suite." | |
| A4: Honest terminology | No "certificate" for the futility predictor. No "bridging theorem" for T2. No "main result" applied to T2. "Spectral futility predictor" used throughout. | |
| A5: Stratified evaluation | 500-instance stratified subset for paper evaluation. Spectral annotations for all 1,065 as artifact. Per-stratum power analysis (≥20/stratum). | |
| A6: Ablation as core experiment | Feature ablation (spectral vs. syntactic vs. combined vs. random vs. trivial) is the central experiment. Clearly described with matched feature budgets. | |
| A7: Venue targeting | JoC is the primary target. Paper structure matches JoC conventions. No claims suited only for MPC/IPCO (tight bounds, etc.). | |

### 4.5 Completeness Checking — Required Components

| Component | Required in approach.json | Required in paper.tex |
|-----------|--------------------------|----------------------|
| L3 statement | Full formal statement | Full formal statement with proof |
| L3 proof | Proof sketch or full proof | Full proof |
| L3-sp (specializations) | Formal statement | Formal statement with proof |
| L3-C Benders | Full statement | Full statement with proof |
| L3-C DW | Full statement | Full statement with proof |
| T2 statement | Full formal statement | Full formal statement (2–3 pages max) |
| T2 proof | Proof sketch | Proof or reference chain |
| F1 (permutation invariance) | Statement | Statement with proof/sketch |
| F2 (scaling sensitivity) | Analysis | Honest analysis |
| Spectral feature definitions | All 8 features formally | All 8 features formally |
| Algorithms | At least pseudocode for: Laplacian construction, feature extraction, crossing-weight computation, spectral partition | Publication-quality pseudocode |
| Complexity analysis | All algorithms | All algorithms |
| Evaluation plan | Full plan with metrics, baselines, statistical tests | Full plan integrated into paper narrative |

---

## 5. CONTINUE/ABANDON Decision Framework

### 5.1 Decision Tree

```
START
│
├── Check FATAL Red Flags (§3, RF-F1 through RF-F5)
│   ├── ANY fatal red flag triggered?
│   │   └── YES → ABANDON (no remediation possible for fatal flags)
│   └── NO → continue
│
├── Score all 5 components (§1.1–§1.5)
│   │
│   ├── Composite ≥ 6.0 AND all components ≥ hard floors?
│   │   └── YES → CONTINUE (unconditional)
│   │
│   ├── Composite ∈ [5.0, 5.9] AND no component below floor by >1?
│   │   ├── Proofs ≥ 6?
│   │   │   ├── YES → Check serious red flags (RF-S1 through RF-S8)
│   │   │   │   ├── ≤ 3 serious flags AND all remediable in ≤3 days?
│   │   │   │   │   └── CONDITIONAL CONTINUE with remediation plan
│   │   │   │   └── > 3 serious flags OR any not remediable?
│   │   │   │       └── ABANDON
│   │   │   └── NO → ABANDON
│   │   └── (any component below floor by >1) → ABANDON
│   │
│   └── Composite < 5.0?
│       └── ABANDON
│
END
```

### 5.2 CONTINUE Conditions (Unconditional)

All of the following must hold:
1. No fatal red flags (RF-F1 through RF-F5).
2. Definitions ≥ 6, Proofs ≥ 7, Algorithms ≥ 5, Evaluation ≥ 7, Structure ≥ 6.
3. Composite ≥ 6.0.
4. All 7 depth-check amendments verifiably honored.
5. Both approach.json and paper.tex are present and consistent.
6. paper.tex > 50KB.

### 5.3 CONDITIONAL CONTINUE Conditions

All of the following must hold:
1. No fatal red flags.
2. Composite ∈ [5.0, 5.9].
3. No component below its hard floor by more than 1 point.
4. Proofs ≥ 6 (L3 must be fundamentally correct even if rough).
5. ≤ 3 serious red flags, all with identified remediation completable in ≤ 3 days.
6. A written remediation plan specifying: what is wrong, what the fix is, and how long it takes.

Under conditional continue, the theory stage is NOT marked "done" — it is marked "needs remediation" and the remediation must be completed before implementation begins.

### 5.4 ABANDON Conditions

Any ONE of the following triggers ABANDON:
1. Any fatal red flag (RF-F1 through RF-F5) is present.
2. Composite < 5.0.
3. Proofs < 5.
4. Evaluation Plan < 5.
5. Two or more components below hard floor by ≥ 2 points.
6. More than 3 serious red flags.
7. A serious red flag that cannot be remediated in ≤ 3 days.
8. approach.json or paper.tex is missing.

### 5.5 Weighting: Theoretical Quality vs. Practical Value

This is a **computational study**, not a theory paper. The weighting reflects this:

| Dimension | Weight in Decision | Rationale |
|-----------|-------------------|-----------|
| Evaluation plan quality | 0.25 | This IS the paper. A weak evaluation plan means the empirical program is misguided. |
| Proof correctness (L3) | 0.25 | L3 is the main theorem. It must be correct. But it need not be deep — correct and useful > deep and fragile. |
| Paper structure | 0.20 | JoC reviewers will evaluate the paper as a computational study. Structure must match expectations. |
| Definition quality | 0.15 | Precision matters but can be fixed in revision. |
| Algorithm quality | 0.15 | Will be refined during implementation. Pseudocode errors are fixable. |

Note: if this were a theory paper (MPC/IPCO), proofs would be 0.40 and evaluation 0.10. The weights here are calibrated to JoC.

---

## 6. Score Calibration

### 6.1 Baseline Scores

From depth check: V5 / D4 / BP3 / L6. Composite 4.5/10.

These scores reflect the *proposal* (post-Amendment E framing), not the *executed theory*. The theory stage can move scores up or down based on evidence.

### 6.2 Evidence for Score Adjustment

| Score | Evidence for Increase | Evidence for Decrease |
|-------|----------------------|----------------------|
| V (Value, baseline 5) | L3 is genuinely useful and non-trivial (not just LP duality). L3-C specializations provide actionable metrics for practitioners. Census reveals surprising structural findings in MIPLIB. | L3 is trivial (RF-F2). Census annotations are uninformative. Spectral features, upon analysis, are highly correlated with simple syntactic features. |
| D (Difficulty, baseline 4) | Proof of L3 requires novel techniques beyond textbook LP duality. Spectral engine has non-trivial numerical stability challenges. | L3 follows in 3 lines from complementary slackness. All algorithms are straightforward applications of existing libraries. |
| BP (Best-Paper, baseline 3) | L3 has standalone value beyond this paper. Spectral features reveal a clean theoretical story. Paper achieves the "aha moment" of connecting spectral graph theory to decomposition quality. | Theory is routine. Features are incrementally better than syntactic features. Census is the only contribution, and it is descriptive rather than insightful. |
| L (Laptop CPU, baseline 6) | Theory stage is irrelevant to L (laptop feasibility is an implementation concern). | Theory reveals that the spectral computation is harder than anticipated (e.g., full eigendecomposition needed, not just bottom-k). |

### 6.3 Plausible Post-Theory Scores

| Scenario | V | D | BP | L | Composite | P(scenario) |
|----------|---|---|----|---|-----------|-------------|
| **Best case**: L3 is non-trivial and useful, features are well-defined, evaluation plan is rigorous | 6 | 5 | 4 | 6 | 5.25 | 0.20 |
| **Expected case**: L3 is correct and moderately interesting, features are clean, plan is solid | 5 | 4 | 3 | 6 | 4.50 | 0.50 |
| **Weak case**: L3 is correct but borderline trivial, some gaps in features, plan is adequate | 4 | 4 | 2 | 6 | 4.00 | 0.20 |
| **Failure case**: L3 has errors or is trivially LP duality, major gaps | 3 | 3 | 1 | 6 | 3.25 | 0.10 |

**Expected post-theory composite**: 0.20(5.25) + 0.50(4.50) + 0.20(4.00) + 0.10(3.25) = **4.53**.

The theory stage is unlikely to significantly move the composite because the paper's value is primarily empirical. The theory stage's main job is to ensure the theoretical components are *correct* and *honestly framed*, not to produce breakthrough mathematics.

### 6.4 Score Update Rules

Post-theory, update each score as follows:
- V: adjust ±1 based on L3 novelty assessment and feature analysis quality.
- D: adjust ±1 based on proof complexity and algorithm non-triviality.
- BP: adjust ±1 based on whether the theory adds an "aha moment" to the paper.
- L: no change (theory stage does not affect computational feasibility).
- **Never adjust any score by more than ±1 in the theory stage.** Larger adjustments require new empirical evidence (implementation stage).

---

## 7. Reviewer Simulation

### 7.1 Reviewer 1: Theory-Oriented

**Profile**: Associate professor in optimization theory. Has published in MPC and Math Programming. Evaluates proof correctness, novelty, and mathematical depth. Will read every proof line by line.

**Likely assessment of this paper:**

> *"The paper's main theoretical contribution, Lemma L3 (partition-to-bound bridge), provides a bound on the gap between the monolithic LP relaxation and the decomposed dual bound in terms of crossing hyperedge weight. This is a reasonable result but I have concerns about its novelty. The connection between crossing constraints and dual bound degradation is implicit in the Lagrangian relaxation literature — specifically, the gap between the LP relaxation and the Lagrangian dual for a given set of relaxed constraints is bounded by the dual values on those constraints. L3 appears to formalize this observation for partition-based decomposition, which has some value, but the intellectual content may be limited.*
>
> *Proposition T2 is honestly presented as motivational, which I appreciate. The constant $C = O(k \cdot \kappa^4 \cdot \|c\|_\infty)$ is indeed vacuous for practical instances, and the paper is right not to position this as a tight bound. The Davis-Kahan application is correct but routine.*
>
> *I would like to see: (1) a more careful discussion of how L3 relates to existing bounds in Lagrangian relaxation theory (e.g., Geoffrion 1974, Fisher 1985); (2) a concrete example demonstrating L3's tightness; (3) a discussion of whether L3 can be tightened by exploiting integrality constraints rather than just the LP relaxation."*

**Predicted score**: 6/10 (minor revision). Would not reject on theory alone but would note limited novelty. Would accept if empirical results are strong.

**Implications for verification**: We must ensure L3 is clearly distinguished from known Lagrangian relaxation bounds. The paper needs a "Relationship to Existing Theory" subsection. L3's proof must be airtight since Reviewer 1 will check every step.

### 7.2 Reviewer 2: Computation-Oriented

**Profile**: Senior researcher at an optimization software company or applied OR group. Has published in JoC and C&OR. Evaluates experimental methodology, reproducibility, and practical impact. Will skim proofs but carefully evaluate the experimental section.

**Likely assessment:**

> *"The experimental design is the strength of this paper. The stratified evaluation on 500 MIPLIB instances with external baselines (GCG for DW, SCIP-native Benders) avoids the circularity trap that plagues many decomposition papers. The feature ablation (spectral vs. syntactic vs. combined) is well-designed and will provide clear evidence for or against the spectral feature hypothesis.*
>
> *Concerns: (1) The time cutoffs (60s, 300s, 900s) may not be long enough for the larger MIPLIB instances — some need hours to see decomposition benefits. The paper should discuss this limitation. (2) The labeling scheme (which method is 'best' at each cutoff) creates a moving target. Label stability analysis is mentioned but must be thorough. (3) The 500-instance subset introduces selection bias — the paper needs to argue convincingly that the stratification covers the relevant diversity. (4) Spectral feature extraction adds overhead. The paper claims <30s per instance, but this should be benchmarked and compared against the syntactic feature extraction time. If spectral features take 30s and syntactic features take 0.1s, the speedup from better decomposition selection must exceed the feature extraction overhead.*
>
> *The MIPLIB census artifact is genuinely valuable and could become a standard reference for decomposition research. I recommend release in an easily parseable format (Parquet or CSV with clear schema), with a DOI.*
>
> *Minor: The futility predictor is interesting but the 80% precision target feels arbitrary. What is the cost of false positives (missing a beneficial decomposition) vs. false negatives (wasting time on futile decomposition)? An asymmetric loss analysis would strengthen this component."*

**Predicted score**: 7/10 (minor revision). Would accept based on census value and experimental design, even if spectral features show modest improvement.

**Implications for verification**: The evaluation plan must address Reviewer 2's concerns: time cutoff justification, label stability, stratification defense, and feature extraction overhead. These must be explicitly present in both approach.json and paper.tex.

### 7.3 Reviewer 3: Domain Expert (GCG/Benders/DW)

**Profile**: Professor or senior researcher who has contributed to GCG development, or published on Benders decomposition in the context of MIP. Knows the decomposition landscape intimately. Has opinions about what spectral methods can and cannot do.

**Likely assessment:**

> *"I have worked with GCG for 10 years and have seen many proposals to improve decomposition detection. My primary concerns with this paper are:*
>
> *(1) The authors use 'decomposition selection' to mean choosing between Benders and DW. But in practice, the choice is not Benders-vs-DW — it is 'which specific partition/decomposition?' For DW, GCG already explores a large space of decompositions via its detection loop. The oracle here selects a method, then presumably uses a single spectral partition. This is a much coarser decision than what GCG does internally. The paper needs to position its oracle relative to GCG's detection loop, not as a replacement.*
>
> *(2) Spectral clustering of the constraint hypergraph is related to but distinct from the hypergraph partitioning that GCG uses (which is based on hMETIS). The paper should explicitly compare spectral partitions to hMETIS partitions on the same instances — not just spectral features to syntactic features. This is a different and arguably more important comparison.*
>
> *(3) The Benders decomposition selection aspect is underdeveloped. Benders decomposition is typically triggered by specific problem structure (stochastic programs, network design) rather than generic block structure. Using spectral features to identify Benders-amenable instances is interesting but needs much more justification. What does a Benders-amenable spectral signature look like?*
>
> *(4) Lemma L3 gives a bound in terms of 'shadow price magnitude.' But shadow prices are only known after solving the LP relaxation. If L3 requires solving the LP first, its value as a preprocessing tool is limited — you may as well solve the LP and use reduced costs directly. The paper needs to discuss this practical limitation honestly.*
>
> *(5) The census is welcome. However, 'complete MIPLIB decomposition census' is a strong claim. At 1-hour cutoffs with inevitable timeouts, the decomposition results for harder instances are really 'partial evaluation' results, not definitive assessments of decomposition potential. The spectral annotations are complete; the decomposition evaluations are not. The paper should distinguish these clearly."*

**Predicted score**: 5/10 (major revision). Would not reject outright but would demand substantial revisions: positioning relative to GCG's detection, spectral-vs-hMETIS comparison, honest discussion of L3's practical limitations, and more careful census claims.

**Implications for verification**: The paper must contain:
- Explicit discussion of spectral partitions vs. hMETIS partitions.
- Honest acknowledgment that L3 requires LP dual information (post-hoc, not a priori).
- Clear distinction between "spectral annotations" (complete, all 1,065) and "decomposition evaluations" (partial, 500 with timeouts).
- Positioning relative to GCG's internal detection loop.
- Justification for Benders-amenability detection via spectral features.

---

## 8. Consolidated Verification Checklist

This checklist is the operational document for the final verification pass. Each item is checked once, marked PASS/FAIL, with notes.

### 8.1 Fatal Checks (any FAIL → ABANDON)

| # | Check | PASS/FAIL | Notes |
|---|-------|-----------|-------|
| FC-1 | L3 proof is correct (line-by-line verified) | | |
| FC-2 | L3 is non-trivial (not a direct restatement of LP duality) | | |
| FC-3 | L3-C Benders is consistent with L3 | | |
| FC-4 | L3-C DW is consistent with L3 | | |
| FC-5 | Evaluation plan uses external baselines (GCG, SCIP Benders) | | |
| FC-6 | T2 is NOT positioned as main contribution | | |
| FC-7 | No fatal logical errors in any proof | | |

### 8.2 Quality Checks (scored 1–10)

| # | Component | Score | Notes |
|---|-----------|-------|-------|
| QC-1 | Definitions | /10 | |
| QC-2 | Proofs | /10 | |
| QC-3 | Algorithms | /10 | |
| QC-4 | Evaluation Plan | /10 | |
| QC-5 | Paper Structure | /10 | |
| QC-W | **Weighted Composite** | /10 | 0.15·QC1 + 0.25·QC2 + 0.15·QC3 + 0.25·QC4 + 0.20·QC5 |

### 8.3 Serious Red Flag Checks

| # | Red Flag | Present? | Remediable? | Notes |
|---|----------|----------|------------|-------|
| RF-S1 | F1 permutation-invariance proof missing | | | |
| RF-S2 | F2 scaling analysis missing or dishonest | | | |
| RF-S3 | Crossing-weight pseudocode missing | | | |
| RF-S4 | Amendment compliance incomplete | | | |
| RF-S5 | Complexity claims incorrect | | | |
| RF-S6 | Numerical stability not addressed | | | |
| RF-S7 | Zero spectral gap not handled | | | |
| RF-S8 | LoC claim exceeds 30K | | | |

### 8.4 Amendment Compliance

| Amendment | Honored? | Evidence |
|-----------|----------|----------|
| A1: Census-first | | |
| A2: External baselines | | |
| A3: Honest scope | | |
| A4: Honest terminology | | |
| A5: Stratified evaluation | | |
| A6: Ablation core | | |
| A7: JoC targeting | | |

### 8.5 Reviewer Readiness

| Reviewer Concern | Addressed? | Location in paper |
|-----------------|-----------|-------------------|
| L3 vs. existing Lagrangian bounds (R1) | | |
| L3 tightness example (R1) | | |
| Time cutoff justification (R2) | | |
| Label stability analysis (R2) | | |
| Stratification defense (R2) | | |
| Feature extraction overhead (R2) | | |
| Census artifact format and release (R2) | | |
| Spectral vs. hMETIS partition comparison (R3) | | |
| Positioning relative to GCG detection (R3) | | |
| L3 practical limitation (requires LP duals) (R3) | | |
| Benders-amenability justification (R3) | | |
| Census completeness claims (R3) | | |

---

## 9. Timeline and Process

### 9.1 Verification Sequence

1. **Pre-check (1 hour)**: Verify both artifacts exist, paper.tex > 50KB, approach.json is valid JSON with all required fields.
2. **Fatal flag scan (2 hours)**: Execute FC-1 through FC-7. If any FAIL, stop immediately with ABANDON.
3. **Proof verification (4 hours)**: Line-by-line check of L3, L3-C, T2. Score Proofs component.
4. **Component scoring (3 hours)**: Score Definitions, Algorithms, Evaluation, Structure. Compute composite.
5. **Red flag audit (1 hour)**: Check RF-S1 through RF-S8.
6. **Consistency check (1 hour)**: approach.json ↔ paper.tex, notation, amendment compliance.
7. **Reviewer simulation validation (1 hour)**: Verify all reviewer concerns from §7 are addressed.
8. **Verdict (30 min)**: Apply decision tree from §5.1. Write verdict document.

**Total estimated time**: 13.5 hours.

### 9.2 Deliverable

The verification produces a **verdict document** containing:
- CONTINUE / CONDITIONAL CONTINUE / ABANDON decision.
- Completed checklist (§8).
- Component scores with justification.
- Updated probability estimates (P(JoC), P(best-paper), P(abandon)).
- Updated depth-check scores (V/D/BP/L) with evidence.
- If CONDITIONAL CONTINUE: specific remediation plan with timeline.
- If ABANDON: specific reason(s) and any salvageable components.

---

*This verification framework is binding for the theory stage. The Verification Chair applies it to the final theory artifacts without modification. Any deviation from this framework requires re-approval.*
