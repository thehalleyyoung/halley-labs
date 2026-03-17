# Verification Report: Spectral Decomposition Oracle — Theory Stage

**Verification Chair Assessment**
**Date:** 2025-07-22
**Deliverables reviewed:** `theory/approach.json`, `theory/paper.tex`
**Reference documents:** `depth_check.md`, `theory/red_team_report.md`
**Binding scores:** V5 / D4 / BP3 / L6 (composite 4.5/10)

---

## Checklist

### 1. Depth-Check Amendments Honored (7 mandatory) — PARTIAL PASS

| Amendment | Status | Finding |
|-----------|--------|---------|
| **A1: Census-First Restructuring** | ⚠ PASS with caveat | Title uses "First Complete MIPLIB 2017 Decomposition Census" but only spectral annotations cover all 1,065 instances; decomposition evaluation covers 500. The word "Complete" contradicts depth-check S5 ("Census with 60% timeouts is not a census") and red-team §6.3. The intro paragraph (line 98) correctly says census is the "primary contribution," but the numbered contribution list puts Spectral Features as P1 and Census as P2, creating an internal ordering contradiction. |
| **A2: External Baselines** | ✅ PASS | GCG for DW and SCIP-native Benders used as external baselines. Selector ablation framing adopted (§6.3). No custom solver reimplementations claimed. Labels constructed from external backends at wall-clock parity (§6.2). |
| **A3: Honest Scope** | ✅ PASS | approach.json targets 26.5K LoC with 30K cap. No 155K claims. System described as "lightweight preprocessing layer" (line 87), consistent with "preprocessing plugin and evaluation framework." |
| **A4: Honest Terminology** | ✅ PASS | "No-go certificate" → "spectral futility predictor" throughout. T2 labeled "motivational" with explicit vacuousness acknowledgment. No "bridging theorem" or "main result" applied to T2. Zero occurrences of "certificate" in paper.tex. |
| **A5: Stratified Evaluation** | ✅ PASS | Four tiers implemented: Pilot (50), Development (200), Paper (500 stratified), Artifact (1,065 spectral-only). Stratification by structure type × size bin × conditioning documented (§6.1). |
| **A6: Feature Ablation** | ✅ PASS | Six configurations (SPEC-8, SYNT-25, GRAPH-10, KRUBER-21, COMB-ALL, RANDOM) at matched feature budgets via mRMR. Feature-count-controlled comparison at k ∈ {3, 5, 8}. Random and trivial baselines included. Nested 5×3 CV. |
| **A7: Venue Targeting** | ✅ PASS | Primary: INFORMS JoC. Secondary: C&OR, CPAIOR. No MPC/IPCO claims. |

**Action required:** Change "Complete" to "Systematic" or "Comprehensive" in the title. Resolve the P1/P2 ordering to align with the census-first directive.

---

### 2. Proofs Complete Where Claimed — FAIL

**Lemma L3 (Partition-to-Bound Bridge): Two gaps identified.**

**Gap 1 — Step 3 feasibility (lines 503–508).** The proof constructs $\bar{y}$ by setting crossing duals to zero ($\bar{y}_i = 0$ for $i \in \mathcal{C}$, $\bar{y}_i = y_i^*$ otherwise) and asserts $\bar{y}$ is feasible for the decomposed dual $z_D(P) = \max\{b^\top y : A^\top y \leq c, y \geq 0, y_i = 0 \; \forall i \in \mathcal{C}\}$. Feasibility requires $A_{\mathcal{I}}^\top y^*_{\mathcal{I}} \leq c$. From $A^\top y^* \leq c$ we get $A_{\mathcal{I}}^\top y^*_{\mathcal{I}} \leq c - A_{\mathcal{C}}^\top y^*_{\mathcal{C}}$. This gives $A_{\mathcal{I}}^\top y^*_{\mathcal{I}} \leq c$ only if $A_{\mathcal{C}}^\top y^*_{\mathcal{C}} \geq 0$ componentwise. For general $A$ with mixed-sign entries, this is NOT guaranteed. The paper says "for general $A$, we use the fact that $z_D(P) \geq b^\top \bar{y}$ by feasibility" — this asserts the conclusion rather than proving it. The correct approach would use Lagrangian relaxation (dualizing crossing constraints into the objective) rather than restricting the dual.

**Gap 2 — Step 5, the $(n_e - 1)$ factor (lines 518–528).** Step 4 yields $z_{\text{LP}} - z_D(P) \leq \sum_{i \in \mathcal{C}} b_i y_i^*$. The transition to $\sum_{i \in \mathcal{C}} |y_i^*| \cdot (n_{e_i} - 1)$ is justified only by the hand-wave "each crossing constraint creates $n_{e_i} - 1$ independent coupling relaxations." The $(n_e - 1)$ factor — identified by the paper as one of L3's novel contributions — arises from the variable-duplication model of Lagrangian decomposition, not from the constraint-dropping model used in the proof. This transition is not formally derived; the two models are conflated. The gap between $\sum b_i y_i^*$ and $\sum |y_i^*|(n_e-1)$ is non-trivial and can be arbitrarily large (consider $b_i \gg n_{e_i}$ or $b_i \ll 0$).

**L3-C Benders (lines 559–574): Proof sketch, not complete proof.** The bound is asserted via "By LP duality of the Benders master" without formal derivation. The connection between reduced costs and the coupling gap is claimed, not shown. Acceptable as a sketch if labeled accordingly — but the paper presents it under `\begin{proof}`, not `\begin{proof}[Proof sketch]`.

**L3-C DW (lines 587–602): Proof sketch, not complete proof.** Same issue. The claim that "violation under the restricted master is bounded by $|\mathrm{blocks}(i)| - 1$" is asserted with informal justification. Again presented as a full proof, not a sketch.

**L3-sp (lines 628–654): Labeled "Proof sketch."** Acceptable. Step 2 ($\|E_\mathcal{L}\|_F \leq \delta\sqrt{d_{\max}}$) holds only for the clique-expansion Laplacian; no bound for the incidence variant (acknowledged in the subsequent remark). Step 3 cites Lei & Rinaldo 2015 but does not verify the balanced-cluster assumptions on constraint matrices. These are noted limitations, not proof errors.

**T2 (lines 678–686): Labeled "Proof sketch."** Acceptable given motivational status. The missing $\delta < \gamma/2$ assumption (red-team T2-1) is still absent.

**Verdict on proofs:** L3 is the paper's **main theorem** (approach.json: `"status": "main_contribution"`). Two non-trivial gaps in the proof make this a **FAIL**. The gaps are fixable — the underlying bound is a known consequence of Lagrangian duality — but the current proof does not establish what it claims. L3-C proofs should either be completed or relabeled as proof sketches.

---

### 3. Red-Team Findings Addressed — PARTIAL PASS

**FATAL findings (3 total):**

| ID | Finding | Status | Assessment |
|----|---------|--------|------------|
| F-1 | L3 is Geoffrion 1974 in hypergraph costume | ⚠ Partial | Remark (line 538–543) honestly acknowledges the relationship and identifies specific new elements. However, the red-team asked for non-trivial proof machinery beyond a counting argument. The paper does not provide a matching lower bound or demonstrate that L3 requires anything beyond 5 lines from LP duality. The honesty is good; the defense is thin. |
| F-2 | L3 direction ambiguity for non-converged methods | ✅ Addressed | L3 is stated for the Lagrangian dual (well-defined direction). L3-C handles iteration-$t$ bounds explicitly. The decomposition overhead filter (line 959) avoids trivially converged cases. |
| F-3 | L3-sp depends on T2, contradicting demotion | ✅ Addressed | The proof dependency DAG (lines 706–712) correctly shows L3 → L3-sp → T2. T2 depends on L3-sp, not the reverse. L3-sp is self-contained (Davis-Kahan + L3, no T2 needed). |

**SERIOUS findings (12 total):**

| ID | Finding | Status | Assessment |
|----|---------|--------|------------|
| S-1 | Dual degeneracy: which optimal dual? | ❌ Not addressed | Assumption 1 picks "a" dual $(x^*, y^*)$ without specifying which. Paper does not discuss minimum-norm dual or worst-case dual. |
| S-2 | L3 retrospective, contradicts preprocessing | ✅ Addressed | Remark at line 531; approximate duals at 10/100/1000 simplex iterations tested; limitations section acknowledges. |
| S-3 | L3-C DW assumes converged CG | ⚠ Partial | L3-C DW stated at "iteration $t$" which is correct, but no restriction of empirical validation to converged runs. GCG timeouts produce non-converged duals; paper doesn't report convergence rate. |
| S-4 | T2 missing assumption $\delta = O(\gamma)$ | ❌ Not addressed | Davis-Kahan requires $\gamma_k > 2\|E\|_2$ for the gap not to collapse (Weyl). This implicit assumption is not stated in the perturbation model (Definition 7) or Proposition T2. |
| S-5 | $\gamma_2$ vs $\lambda_{k+1} - \lambda_k$ | ✅ Addressed | Explicitly acknowledged as conservative proxy in SF2 note, approach.json, and limitations section. Honest treatment. |
| S-6 | Two Laplacian variants incompatible | ⚠ Partial | $d_{\max}$ included as control variable. Spearman validation proposed for overlap. But no feature normalization, no unified Laplacian option, no classifier-split experiment (red-team KQ3). |
| S-7 | Spectral = proxy for density | ⚠ Partial | G0 gate with RF R² test. H6 tests linear redundancy. But the nonlinear RF regression the red-team specifically requested ($\gamma_2 \sim f(\text{density, size})$ with R² > 0.80) and partial correlation analysis are absent. |
| S-8 | AutoFolio + SPEC-8 baseline missing | ❌ Not addressed | AutoFolio is cited in related work but NOT included as a baseline. Red-team (SCOPE-2) called this "a glaring gap." The 7 baselines do not include AutoFolio configuration selection. |
| S-9 | Class imbalance power on minority | ⚠ Partial | Balanced accuracy and per-structure-type reporting included. But no minority-class power analysis (with ~15–20 minority instances per fold, statistical power is questionable). |
| S-10 | Labels reflect software, not structure | ✅ Addressed | Threats to validity acknowledges solver maturity asymmetry. Docker pinning. Limitations section discusses explicitly. |
| S-11 | Silent eigenvalue errors | ⚠ Partial | ARPACK→LOBPCG fallback chain. NaN returns on failure. But the specific shift-invert issue (condition number $\sim 10^{10}$ for $\lambda_2 \approx 10^{-10}$) with relative vs. absolute tolerance is not addressed. No synthetic validation. |
| S-12 | $k$ selection unspecified | ✅ Addressed | Eigengap heuristic documented. Sensitivity at $k \in \{2,5,10,20\}$. Fallbacks in approach.json. |

**Summary:** 2/3 FATALs resolved; 4/12 SERIOUS fully addressed, 5/12 partially, 3/12 not addressed. Red-team threshold was "all FATALs resolved and ≥8/12 SERIOUS addressed." The paper falls short with only 4–6.5 of 12 SERIOUS addressed (counting partials at 0.5).

---

### 4. approach.json Consistent with paper.tex — PASS with minor issues

| Item | Consistent? | Note |
|------|-------------|------|
| Title | ✅ | Exact match |
| Contributions (P1/P2/P3) | ⚠ | Ordering matches but paper intro paragraph says census is "primary contribution" while numbered list puts features first |
| Scores (V5/D4/BP3/L6) | ✅ | Exact match with depth check |
| Kill gates (G0–G5) | ✅ | Match |
| Hypotheses (H1–H7) | ✅ | Match |
| Features (SF1–SF8) | ✅ | Match in count, naming, and definitions |
| Theorems (L3, L3-sp, L3-C, T2, F1, F2) | ✅ | All present in both |
| Baselines (7 items) | ✅ | Match |
| LoC budget | ✅ | 26.5K / 30K cap; paper doesn't contradict |
| Proof dependency DAG | ✅ | approach.json and paper §4.6 match |
| T2 as motivational | ✅ | Both label it motivational only |

---

### 5. Scores Not Inflated Beyond Depth-Check Binding — PASS

| Score | Depth Check | approach.json | Delta |
|-------|-------------|---------------|-------|
| Value | 5 | 5 | 0 |
| Difficulty | 4 | 4 | 0 |
| Best Paper | 3 | 3 | 0 |
| Laptop/CPU | 6 | 6 | 0 |
| P(JoC) | 0.55 | 0.55 | 0 |
| P(best paper) | 0.03 | 0.03 | 0 |
| P(abandon) | 0.25 | 0.25 | 0 |

Scores are exactly the depth-check binding values. No inflation.

---

### 6. paper.tex Exceeds 50KB — PASS

File size: **63,808 bytes** (63.8 KB). Exceeds 50KB threshold by 27.6%.

Content is substantive: 8 definitions, 4 lemmas, 3 propositions, 1 cited theorem, 10 remarks, 6 algorithms with pseudocode, 7 sections, 30 bibliography entries. Not padded.

---

### 7. Every Definition/Theorem Is Load-Bearing — PASS

All 27 formal mathematical environments audited:

- **8 definitions:** Constraint hypergraph, Ruiz equilibration, clique-expansion Laplacian, incidence-matrix Laplacian, normalized Laplacian, block-diagonal perturbation, spectral features (SF1–SF8), crossing hyperedges/weight. All directly used in features, algorithms, or proofs.
- **4 lemmas:** L3, L3-C Benders, L3-C DW, L3-sp. All load-bearing per approach.json.
- **3 propositions:** F1 (permutation invariance), F2 (scaling sensitivity), T2 (motivational scaling law). F1/F2 are correctness guarantees for the feature family. T2 is motivational but serves the paper's narrative.
- **1 theorem:** Davis-Kahan (cited, used in L3-sp and T2 proofs).
- **10 remarks:** Each addresses a specific concern (non-canonicality, retrospective nature, novelty vs. Geoffrion, vacuousness, etc.). These are defensive annotations, not ornamental.

No ornamental math detected.

---

## Residual Issues

### Critical (must fix before implementation stage)

1. **L3 proof gaps.** The main theorem's proof has two non-trivial holes: feasibility of $\bar{y}$ for general $A$ (Step 3), and the $(n_e-1)$ factor derivation (Step 5). The underlying bound is correct (it follows from Lagrangian duality), but the proof as written does not establish it. Fix: rewrite using the variable-duplication Lagrangian relaxation model, which naturally yields the $(n_e-1)$ factor.

2. **"Complete" in title.** Contradicts depth-check S5 and red-team §6.3. Spectral annotations are complete; decomposition evaluations are not. Change to "Systematic" or "First Comprehensive."

3. **AutoFolio baseline missing (S-8).** The most natural baseline — adding 8 spectral features to AutoFolio's existing feature set — is absent despite the red-team calling it "a glaring gap." This is the experiment most likely to be demanded by a JoC reviewer.

### Serious (should fix before implementation stage)

4. **T2 missing $\delta < \gamma/2$ assumption (S-4).** Davis-Kahan's bound degrades to vacuity when the perturbation exceeds half the gap. This unstated assumption restricts T2 to easy cases. Add explicitly.

5. **L3-C proofs labeled as full proofs.** Both L3-C Benders and L3-C DW use `\begin{proof}` but contain proof-sketch-level arguments. Either complete them or relabel as `\begin{proof}[Proof sketch]` for consistency with L3-sp.

6. **Dual degeneracy (S-1).** L3's Assumption 1 should specify "for any optimal dual $y^*$" (making the bound hold universally) or "for the minimum-norm optimal dual" (making it computable). Currently ambiguous.

7. **No minority-class power analysis.** With ~75 Benders and ~100 DW instances in 500, each 5-fold test split has ~15–20 minority samples. McNemar's test at these sample sizes after Holm-Bonferroni correction has very low power. This should be computed explicitly.

### Minor

8. Contribution ordering: intro paragraph says census is primary, but numbered list puts features first.
9. L3-sp incidence-matrix variant deferred to future work — creates a gap for 15% of instances.
10. No synthetic validation for eigensolve correctness near $\lambda_2 \approx 0$.

---

## Final Verdict

### **CONTINUE** — with mandatory revisions before implementation

**Justification:**

The theory deliverables demonstrate a well-structured, honestly framed computational study that largely honors the depth-check binding conditions. The scores are not inflated. The paper is substantive (63.8KB, no ornamental math). Six of seven amendments are cleanly honored; the seventh (census-first) has a minor title issue. The proof dependency structure is sound, T2's motivational status is properly maintained, and the feature ablation design is rigorous.

However, the L3 proof — the paper's **main theoretical contribution** — has two genuine gaps that must be fixed before proceeding. These are fixable (estimated 2–3 days of focused work) because the underlying bound is a known consequence of Lagrangian duality theory. The gaps are in the *presentation*, not the *correctness* of the result. Three SERIOUS red-team findings (S-1, S-4, S-8) remain unaddressed and should be resolved.

The project does NOT warrant ABANDON because: (a) the empirical program (census + feature ablation) is independent of the proof issues; (b) the proof gaps are presentation errors, not fundamental flaws; (c) the overall framing is honest and appropriately scoped.

**Mandatory before implementation:**
1. Fix L3 proof (Steps 3 and 5)
2. Change "Complete" → "Systematic" in title
3. Add AutoFolio + SPEC-8 baseline to evaluation design

**Strongly recommended:**
4. Add $\delta < \gamma/2$ assumption to T2 and L3-sp
5. Relabel L3-C proofs as proof sketches or complete them
6. Address dual degeneracy in L3 assumption

---

## Recommended Scores

| Pillar | Recommended | Binding Max | Note |
|--------|-------------|-------------|------|
| Value | **5** | 5 | Census is valuable; niche audience limits broader impact |
| Difficulty | **4** | 4 | Post-descoping, novel work is moderate |
| Best Paper | **3** | 3 | Solid JoC paper, not competitive for best paper |
| Laptop/CPU | **6** | 6 | Spectral analysis is fast; full census is a batch job |
| P(JoC) | **0.50** | 0.55 | Slight discount for unresolved proof gaps and missing AutoFolio baseline. If mandatory revisions are completed, reverts to 0.55. |
| P(abandon) | **0.25** | — | Unchanged; spectral correlation hypothesis still untested |

*Scores do not exceed depth-check binding.*

---

*Verification Chair — 2025-07-22*
*This report is based on line-by-line review of both deliverables against the depth check and red-team report. No items were rubber-stamped.*
