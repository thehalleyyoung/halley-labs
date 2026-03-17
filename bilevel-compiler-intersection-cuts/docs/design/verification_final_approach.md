# Verification Signoff: BiCut Final Approach

**Verifier:** Independent Verifier  
**Date:** 2025-07-18  
**Document reviewed:** `final_approach.md` (Modified Approach C — Hybrid with cuts-primary emphasis)  
**Cross-referenced against:** `approaches.md`, `skeptic_critique.md`, `math_depth_review.md`, `difficulty_review.md`, `math_skeptic_response.md`, `problem_statement.md`

---

## A. Critique Incorporation

### A1. Skeptic's fatal flaw for Approach C — bidirectional mapping correctness
**✅ PASS**

The final approach resolves this in §7 (Risk 6) and §10 (Amendment 11). It incorporates the Math Assessor's rebuttal: the backward mapping is a projection (x,y,λ,z) → (x,y), which is always well-defined. Cuts are derived and expressed in the original (x,y) bilevel variable space and extended to the MILP space with zero coefficients on auxiliary variables — trivially valid. The implementation enforces this invariant, and integration testing against MibS verifies correctness. The Skeptic's concern about non-injectivity of the forward mapping is correctly identified as irrelevant to backward-mapping correctness (the bilevel feasibility check depends only on (x,y), not auxiliary variables). The final approach also honestly acknowledges the residual *performance* concern: zero-coefficient cuts may be weak in the full MILP space because they do not constrain auxiliary variables.

### A2. "Two papers stapled together" critique
**✅ PASS**

Addressed in §2 (Skeptic point 2), §7 (Risk 4), and §8. The paper targets JOC as primary venue, where compiler + cuts + evaluation combinations are natural (cites CVXPY JOC paper as structural precedent). The narrative is unified: "the compiler makes bilevel cutting-plane research possible for the first time; the cuts are the first demonstration of what this infrastructure enables." The 25-page main text structure (compiler 5–6pp, cut theory 8–10pp, evaluation 8–10pp, full proofs in supplement) is credible for JOC. The approach explicitly does NOT attempt Math Programming unless the full facet-defining proof succeeds — correctly avoiding the venue mismatch the Skeptic warned about.

### A3. Math Assessor's difficulty grade inflation concerns
**✅ PASS**

The final approach systematically adjusts difficulty grades to match the Math Assessor's revised assessments:
- Polyhedrality theorem: graded B (not C), described as "new formalization of known parametric LP structure"
- Compiler soundness (base): graded A, explicitly called "formalization of known results"
- Compiler soundness (extended): graded B−, acknowledging the cut-validity dependency as the source of genuine difficulty
- Compilability decision: graded A, presented as a system property, not a theorem (Amendment 8)
- Certificate composition: Necula comparison dropped (Amendment 6), presented as engineering contribution
- Separation complexity: graded B− (parameterized complexity argument)
- O(h²) error bound: corrected — LP lower levels get exact piecewise-linear evaluation, MILP lower levels get L¹ bounds only (Amendment 4)

The overall math grade is stated as "B (with upside to A− if M3 succeeds)," consistent with the Math Assessor's revised grade of B for Approach C.

### A4. Difficulty Assessor's honest LoC estimates
**⚠️ CONDITIONAL**

The final approach states "All LoC estimates in this document use the Difficulty Assessor's honest figures" (§2, Difficulty Assessor point 1). The §4 table shows total novel LoC of ~12,200–15,500. The Difficulty Assessor's honest estimate for Approach C was ~11,000–13,800. The upper bound of the final approach (15,500) exceeds the Difficulty Assessor's upper bound (13,800) by ~12%. This creep appears to come from novel LoC attributed to S1 (500–700), S4 (800–1,000), S7 (200–300), S10 (300–400), and S11 (500–700) — subsystems where the Difficulty Assessor either did not break out novel LoC or gave lower estimates. The inflation is mild and the individual line items are defensible, but the claim of strict adherence to the Difficulty Assessor's figures is not precisely accurate.

**Fix required:** Either align the upper bound to ~13,800 or note explicitly that the range extends slightly beyond the Difficulty Assessor's estimate due to inclusion of novel logic in additional subsystems (S1, S7, S10, S11).

### A5. BilevelJuMP comparison critique
**✅ PASS**

Addressed comprehensively in §2 (Skeptic point 3), §7 (Risk 5), and §10 (Amendment 13). Three clear differentiators are stated: (a) bilevel intersection cuts — novel, unavailable anywhere; (b) correctness certificates with sound CQ verification — BilevelJuMP has none; (c) value-function reformulation for MILP lower levels — BilevelJuMP doesn't support this. The paper is explicitly NOT positioned as "a better BilevelJuMP" — it is positioned as "the infrastructure that makes bilevel cutting-plane research possible." Amendment 13 mandates a head-to-head comparison on shared instances with the same solver, reporting honestly where BilevelJuMP is comparable.

### A6. Value-function lifting vacuity
**✅ PASS**

Addressed in §2 (Skeptic point 1), §5 (M5), and §10 (Amendment 3). Value-function lifting is demoted from co-primary mathematical contribution to conditional stretch goal. It is pursued only if the prototype shows lifted cuts provide ≥2% additional gap closure over unlifted cuts on LP-lower-level instances. The paper does not promise Gomory-Johnson results. The risk assessment (§7, Risk 3) assigns 50% probability of vacuity and states the intersection cuts stand on their own without lifting. This is exactly the posture both the Math Assessor and Skeptic recommended.

### A7. Binding amendments (prototype gate, go/no-go thresholds, facet-defining conditions)
**✅ PASS**

All binding amendments from the depth check panel are incorporated and extended:
- **Prototype gate (Phase 0):** Mandatory 2-week gate with explicit numerical thresholds — gap closure <5% kills cuts, 5–10% descopes to computational contribution, ≥10% proceeds. Oracle overhead and cache hit rate thresholds specified.
- **Go/no-go thresholds:** Tabulated in §6 with kill/pivot/go columns for three metrics.
- **Facet-defining conditions:** Clearly labeled as stretch goal (M3), not core deliverable. Paper is structured to be strong at JOC with computational cut results only. Restricted-class proof (non-degenerate LP lower levels) is the realistic target.
- **Decision tree:** §6, Phase 0 provides a four-way decision based on gap closure level, routing to different emphasis and venue targets.

---

## B. Technical Soundness

### B1. Mathematical content honestly graded
**✅ PASS**

The §5 math portfolio table assigns grades consistent with the Math Assessor's revised assessment:

| Result | Final Approach Grade | Math Assessor Revised | Consistent? |
|--------|---------------------|-----------------------|-------------|
| M1 Polyhedrality | B | B | ✓ |
| M2 Separation | B− | B− | ✓ |
| M3 Facets | C | C | ✓ |
| M4 Convergence | B− | B− | ✓ |
| M5 Lifting | C/B+ | C/B+ | ✓ |
| M6 Soundness (ext.) | B− | B− | ✓ |
| M7 Selection | A | A | ✓ |
| M8 Compilability | A | A | ✓ |

Overall grade of "B (with upside to A− if M3 succeeds)" matches the Math Assessor's revised B for Approach C. No inflation detected.

### B2. Difficulty claims consistent with Difficulty Assessor
**✅ PASS**

The final approach scores difficulty at 7/10 (§9), matching the Difficulty Assessor's honest score of 7/10 for Approach C. The original approaches document claimed 8/10; the reduction is acknowledged in §2 (Difficulty Assessor point 2).

### B3. LoC estimates realistic
**⚠️ CONDITIONAL**

As noted in A4, the novel LoC range (12,200–15,500) slightly exceeds the Difficulty Assessor's honest range (11,000–13,800). The individual subsystem estimates are reasonable, but the aggregate upper bound is ~12% above the Difficulty Assessor's. This is a mild concern — the discrepancy is within the noise of such estimates — but the document's claim of strict adherence to the Difficulty Assessor's figures should be corrected.

**Note:** The total system LoC (~92,500 core, ~120,000 with overhead) is not itself contested — only the novel LoC fraction is at issue.

### B4. Implementation plan feasible
**⚠️ CONDITIONAL**

The phased plan (Phase 0: 2 weeks; Phase 1: months 1–3; Phase 2: months 4–6; Phase 3: months 7–9; Phase 4: months 9–12) targets core delivery in 9 months. The Difficulty Assessor estimated 10–16 months for a 4–5 person team. The final approach does not specify team size or explain why the timeline is achievable at the lower end of the Difficulty Assessor's range. Key concerns:

1. **Phase 1 parallelism assumption:** The compiler backbone and cut engine are developed concurrently (two parallel tracks). This requires at least two experienced developers working independently — the plan should state this explicitly.
2. **Phase 3 evaluation compute time:** The reduced matrix of ~20,800 configurations at ~5 min average = ~72 days serial / ~9 days with 8-core parallelism. Feasible but tight, especially given that some large instances may take hours. This should be acknowledged.
3. **Month 6 kill gate timing:** If the facet-defining proof is "stuck" at month 6, the plan descopes to computational cuts. But month 6 may be too late to produce a polished paper by month 9. The plan should acknowledge that descoping at month 6 likely pushes the paper timeline to month 12.

**Fix required:** State the assumed team size and acknowledge the compute-time budget for the evaluation phase.

### B5. Kill gates well-defined and actionable
**✅ PASS**

The kill gates are exemplary:
- **Phase 0 (2 weeks):** Three metrics with explicit numerical thresholds (gap closure, oracle overhead, cache hit rate), each with kill/pivot/go criteria. Four-way decision tree based on gap closure bands.
- **Phase 1 checkpoint (month 3):** End-to-end test on 100 BOBILib instances. If integration is broken, 2-week debugging sprint with fallback to cuts-as-standalone.
- **Phase 2 checkpoint (month 6):** Full BOBILib coverage on two backends, certificates operational, gap closure measured. Cross-solver consistency check (>5 percentage point difference between SCIP and Gurobi triggers investigation).
- **Amendment 10:** Cross-solver validation is mandatory — both Gurobi and SCIP must show ≥10% gap closure for the cut contribution to stand.

These are specific, falsifiable, and actionable. Best-in-class for a research project plan.

---

## C. Narrative Quality

### C1. Unified story (not "two papers stapled together")
**✅ PASS**

The narrative thread is clear: "the compiler is infrastructure that enables a new class of computational experiments (cut deployment across reformulations), and the cuts are the first such experiment." The paper structure maps naturally to JOC: compiler (infrastructure) → cut theory (the math enabled by the infrastructure) → evaluation (what the infrastructure + math delivers). This is materially different from the Skeptic's "two papers stapled together" scenario because each component motivates the other — the cuts cannot be systematically evaluated without the compiler, and the compiler's extensibility claim is validated by the cuts.

The residual risk is that a reviewer finds the cut theory insufficiently deep for the page count allocated. This is acknowledged (§7, Risk 4) and mitigated by JOC targeting.

### C2. Honest about user base size
**✅ PASS**

The final approach is admirably honest:
- Primary users: 200–500 (MIBLP researchers + applied practitioners)
- Secondary users: 100–200 (educators + reviewers)
- Ceiling: ~500–700 direct users
- Explicit statement: "three orders of magnitude smaller than CVXPY's user base"
- CVXPY analogy dropped (Amendment 5)
- Value justification pivots to capability gap (no substitute for bilevel cuts), trust infrastructure (certificates for small community), and platform value (enables future research)

### C3. Best-paper argument credible at target venue
**✅ PASS**

The JOC best-paper argument rests on four pillars: (1) novel cutting-plane family demonstrated on 2600+ instances, (2) first correctness-certified bilevel compiler, (3) clean falsifiable evaluation with standard metrics, (4) enables new research direction. Each is genuine and defensible at JOC, which explicitly values software + computation combinations. The argument does not overclaim — it acknowledges that without the facet-defining proof, the contribution is computational, not theoretical. The reach venue (Math Programming) is correctly conditioned on full facet proof success.

### C4. Fallback strategies viable
**✅ PASS**

Clear fallback hierarchy with venue targets for each scenario:

| Scenario | Primary Venue | Fallback Venue |
|----------|---------------|----------------|
| Full success (facets + ≥15% gap closure) | Math Programming | Operations Research |
| Standard success (computational cuts + ≥10%) | JOC | CPAIOR |
| Modest success (cuts 5–10%) | JOC | Optimization Methods & Software |
| Cuts fail entirely | JOC (compiler-only) | Computers & OR |

Each scenario is well-defined with clear entry conditions. The compiler-only fallback (Approach B) is viable because the compiler, certificates, and solver-agnostic emission have independent value — the Skeptic's own assessment gives this path a 70% acceptance rate at JOC.

---

## D. Risk Assessment

### D1. All fatal flaws identified by Skeptic mitigated
**✅ PASS**

Every fatal flaw and major concern from the Skeptic is explicitly addressed:

| Skeptic Concern | Mitigation | Adequate? |
|-----------------|------------|-----------|
| Bidirectional mapping correctness | Projection + zero-coefficient extension | Yes — Math Assessor's proof is convincing |
| "Two papers stapled together" | JOC targeting + unified narrative | Yes |
| BilevelJuMP covers 80% | Three clear differentiators + head-to-head comparison | Yes |
| 80K+ configurations infeasible | Reduced to ~20,800 | Yes |
| Value-function lifting vacuous | Conditional on prototype evidence | Yes |
| Certificate composition trivial | Necula comparison dropped; presented as engineering | Yes |
| Single-solver evaluation | Two solvers mandatory + HiGHS cross-validation | Yes |
| Cut-induced degeneracy breaks convergence | Perturbation argument acknowledged; non-degenerate statement with remark | Adequate |

### D2. Kill probabilities realistic
**⚠️ CONDITIONAL**

The final approach estimates 30–35% total kill probability. The Skeptic estimated 40%. The difference is attributed to risk reduction from: (a) value-function lifting descoped to conditional, (b) CPLEX deferred, (c) prototype gate catches existential risks early, (d) well-defined fallback to compiler-only.

The risk reduction from (a)–(d) is real but the final approach's decomposition is slightly optimistic:
- "Cuts fail at prototype gate (25%) → compiler-only rejected as incremental (40% conditional) → 10% total kill from this path." The 40% conditional rejection of compiler-only is the Skeptic's figure, but the Math Assessor argued this should be 40–50%. Using 50% conditional: 25% × 50% = 12.5% — a 2.5 percentage point difference.
- "Integration fails (10%) → fall back to cuts-standalone → 3% total kill." If integration fails at month 3+, the remaining runway for a polished cuts-standalone paper is tight. The conditional failure rate may be higher than 30%.

The aggregate effect is small — perhaps 33–38% rather than 30–35%. Not a blocking issue, but the document should acknowledge the Skeptic's 40% figure and explain the delta.

**Fix required:** Acknowledge the Skeptic's 40% kill probability estimate and provide a clearer reconciliation of the delta.

### D3. Pivot strategy clearly defined
**✅ PASS**

The Phase 0 decision tree provides a clear, four-way pivot based on gap closure bands:
- ≥15%: Cuts primary, compiler as infrastructure, target Math Programming
- 10–15%: Balanced hybrid, target JOC
- 5–10%: Compiler primary, cuts as modest pass, no facet theory, target JOC
- <5%: Kill cuts, pivot to compiler-only (Approach B), target JOC

The pivot to compiler-only is described with 10+ months of remaining runway. The minimum viable paper (compiler + SCIP + Gurobi + cuts on LP lower levels + certificates + BOBILib evaluation) is explicitly defined at the month 6 gate.

---

## Additional Observations

### Strengths Not Required by Checklist

1. **Amendment 10 (cross-solver validation for cuts)** is an excellent addition that neither the Skeptic nor the Math Assessor explicitly required. Requiring both Gurobi and SCIP to show ≥10% gap closure prevents solver-specific artifact claims.

2. **Amendment 12 (degeneracy reporting)** commits to honest reporting of separation oracle performance on degenerate vs. non-degenerate instances. This proactively addresses the Skeptic's concern about degenerate lower levels.

3. **The honest assessment section (§9)** with value 5/10, difficulty 7/10, potential 6/10, feasibility 6/10 is refreshingly candid. The scores are internally consistent and match the reviewers' assessments.

4. **The failure scenario descriptions** (§9, "What Failure Looks Like") are unusually honest for a project plan and demonstrate awareness of realistic negative outcomes.

### Minor Issues Not Blocking Approval

1. **Reformulation-aware cut selection** is included in INT subsystem (800–1,200 novel LoC) but has no formal statement or guarantees. The Skeptic flagged this as a heuristic, not an algorithm. The final approach treats it as an engineering design decision, which is the correct posture, but could be more explicit that no formal optimality guarantee is claimed for the selection strategy.

2. **The value score of 5/10** is raised "from depth check's 4 by incorporating the reproducibility-layer framing and the QP extension." This justification is thin — the QP extension adds implementation scope, not user base. The score is defensible but the justification should rest on the capability gap argument (no substitute for bilevel cuts) rather than the QP extension.

3. **The finite convergence result (M4)** acknowledges cut-induced degeneracy as a gap requiring a perturbation argument but rates risk as "Very low (5%)." Given that the Math Assessor conceded this is a "real subtlety" and the Skeptic identified it as a genuine gap, 5% failure risk seems too low. A 10–15% risk with the perturbation argument as the primary mitigation would be more honest.

---

## Overall Verdict

### **⚠️ CONDITIONAL APPROVE**

The final approach is comprehensive, honest, and well-structured. It incorporates critiques from all three reviewers with precision and intellectual honesty that is rare in project planning documents. The mathematical grading is accurate. The kill gates are exemplary. The narrative is unified and credible for JOC. The fallback hierarchy provides genuine protection against the dominant failure modes.

### Fixes Required Before Implementation

1. **LoC estimate alignment (A4/B3):** Either reduce the upper bound of the novel LoC range from 15,500 to ~14,000 (to stay within ~10% of the Difficulty Assessor's 13,800 upper bound), or add an explicit note explaining that the range extends beyond the Difficulty Assessor's estimate due to novel logic in S1, S7, S10, and S11, with per-subsystem justification.

2. **Implementation plan team size (B4):** State the assumed team size (the Difficulty Assessor's estimate of 4–5 people for 10–16 months should be the reference). Acknowledge that the 9-month Phase 0–3 target requires at least 3 concurrent developers and note the compute-time budget (~9 days with 8-core parallelism) for the 20,800-run evaluation matrix.

3. **Kill probability reconciliation (D2):** Add a sentence acknowledging the Skeptic's 40% kill probability estimate and briefly explain why the final approach estimates 30–35% (prototype gate catches ~5% of risk early; descoping value-function lifting removes ~2–3% of risk; reduced evaluation matrix removes ~1–2% of schedule risk). The reconciliation need not be longer than a paragraph.

### Assessment of Readiness

These three fixes are minor editorial corrections, not structural changes. None requires returning to the debate phase. The final approach is substantively sound and ready for implementation once the fixes are applied. The 2-week prototype gate (Phase 0) will provide the first empirical test of the approach's viability, and the kill gate structure ensures that failure is caught early.

**Confidence in verdict: High.** The final approach represents a thorough synthesis of three rounds of review (Math Assessor, Difficulty Assessor, Skeptic) with a Math Assessor rebuttal. The residual issues are quantitative precision matters, not conceptual gaps.
