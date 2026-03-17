# BiCut Final Approach: Synthesis Decision

**Project:** BiCut — Bilevel Optimization Compiler with Intersection Cuts
**Slug:** `bilevel-compiler-intersection-cuts`
**Decision:** Modified Approach C (Hybrid) with cuts-primary emphasis, adaptive resource allocation, and aggressive gating
**Date:** 2025-07-18

---

## 1. Winning Approach Title and Summary

**BiCut: A Bilevel Optimization Compiler with Bilevel Intersection Cuts for MIBLPs**

BiCut is a typed bilevel optimization compiler that transforms mixed-integer bilevel linear programs (MIBLPs) into solver-ready MILPs via automatic reformulation selection with machine-checkable correctness certificates — and introduces *bilevel intersection cuts*, a new family of valid inequalities derived by extending Balas's 1971 intersection cut framework to the bilevel-infeasible set defined by follower suboptimality. The compiler provides the IR, structural analysis, and multi-backend emission infrastructure required to systematically deploy and evaluate these cuts across reformulation strategies and solvers. The paper's central mathematical contribution is the bilevel intersection cut theory: a polyhedrality theorem characterizing the bilevel-infeasible set for LP lower levels, a polynomial-time separation oracle (for fixed follower dimension), a finite convergence result, and — as a stretch goal — facet-defining conditions for these cuts. The paper's central systems contribution is the first bilevel compiler with correctness certificates and solver-agnostic emission. The narrative is unified: *the compiler makes bilevel cutting-plane research possible for the first time; the cuts are the first demonstration of what this infrastructure enables.* The paper targets INFORMS Journal on Computing (JOC) with Mathematical Programming as a reach venue if the facet-defining proof succeeds in full generality.

---

## 2. Why This Approach Wins

### Elements Taken from Each Approach

**From Approach A (Cuts-First):**
- The bilevel intersection cut theory as the *mathematical headline*. This is the crown jewel that elevates the paper above "just engineering." The polyhedrality theorem, separation oracle, and finite convergence proof are core deliverables.
- The value-function oracle architecture for LP lower levels (exact parametric LP with caching). MILP lower-level oracle is descoped to best-effort.
- The emphasis on mathematical depth as the differentiator over BilevelJuMP.

**From Approach B (Compiler-First):**
- The full compiler backbone: typed IR (S1), structural analysis (S2), reformulation selection (S3), four reformulation passes (S4, S7), correctness certificates (S9).
- Multi-backend emission to Gurobi + SCIP + HiGHS (three backends, not four — CPLEX descoped from load-bearing to best-effort, per Difficulty Assessor's recommendation to reduce integration surface area).
- The QP lower-level extension (E1) to broaden applicability.
- The "reproducibility layer" framing as fallback positioning.

**From Approach C (Hybrid):**
- The co-primary narrative structure: compiler enables cuts, cuts validate compiler.
- The `BilevelAnnotation` interface for cut-compiler integration.
- The adaptive resource allocation via prototype gating.
- The fallback hierarchy: full theory → computational cuts → compiler-only.

### Critiques Incorporated

**From the Skeptic:**
1. *Value-function lifting may be vacuous for LP lower levels.* **Incorporated.** Value-function lifting is demoted from co-primary mathematical contribution to conditional stretch goal. It is pursued only if the prototype shows lifted cuts provide measurable gap improvement over unlifted cuts. The paper does not promise Gomory-Johnson results.
2. *"Two papers stapled together" venue risk.* **Incorporated.** The paper targets JOC as primary venue, where the compiler + cuts + evaluation combination is natural. The narrative unification is: the compiler is infrastructure that enables a new class of computational experiments (cut deployment across reformulations), and the cuts are the first such experiment. This is one story, not two.
3. *BilevelJuMP covers 80% of the compiler.* **Incorporated.** The paper explicitly positions against BilevelJuMP with three clear differentiators: (a) bilevel intersection cuts (no existing tool), (b) correctness certificates with sound CQ verification (no existing tool), (c) value-function reformulation for MILP lower levels (BilevelJuMP doesn't support this). The paper does NOT claim to be "the first bilevel compiler" — it claims to be the first bilevel compiler with correctness guarantees and novel cutting planes.
4. *The backward mapping correctness concern.* **Resolved** by the Math Assessor's response: the backward mapping is a projection (x,y,λ,z) → (x,y), cuts operate in original bilevel variable space, and extension to the MILP space with zero coefficients on auxiliary variables is trivially valid. The implementation enforces this: cuts are always derived and expressed in the original (x,y) space.
5. *Page count / venue fit.* **Incorporated.** The paper is structured for JOC's 25-page format. Compiler architecture goes in the main text (compact). Cut theory goes in the main text (detailed). Full proofs, implementation details, and extended benchmarks go in an online supplement. The paper is one contribution with two mutually reinforcing components, not two papers stapled together.
6. *80,000+ configurations are infeasible on a laptop.* **Incorporated.** The evaluation matrix is reduced: 2600 instances × 2 reformulations (auto-selected + default KKT) × 2 solvers (Gurobi + SCIP) × 2 configurations (with/without cuts) = 20,800 configurations. HiGHS cross-validation on a 200-instance subset. Full matrix in supplement only if time permits.

**From the Math Depth Assessor:**
1. *Polyhedrality theorem is partially folklore (B, not C).* **Incorporated.** The polyhedrality result is presented as a necessary foundation, not as a standalone contribution. The novel mathematical content is the separation procedure and (if achievable) the facet-defining conditions.
2. *Compiler soundness is Difficulty A, not B.* **Incorporated.** The compiler soundness theorem is presented as a formalization of known results in a new framework, not as novel mathematics. It is a systems-correctness guarantee, not a mathematical contribution.
3. *Certificate composition may be trivial.* **Incorporated.** Certificate composition is presented as an engineering contribution (first machine-checkable correctness pipeline for bilevel reformulations), not as a formal-methods research result. The Necula comparison is dropped.
4. *The O(h²) error bound is invalid for discontinuous MILP value functions.* **Incorporated.** The sampling approximation claim is corrected: for LP lower levels, the piecewise-linear overestimator is exact on a sufficiently fine grid (V is piecewise linear and continuous). For MILP lower levels, only L¹ error bounds are claimed.
5. *Depth dilution risk is real but manageable via gating.* **Incorporated.** The prototype gate determines resource allocation: ≥15% gap closure → prioritize cut theory; 10-15% → balanced; <10% → prioritize compiler, descope cut theory to computational contribution.

**From the Difficulty Assessor:**
1. *Honest novel LoC is ~11-13.8K, not ~19.3K.* **Incorporated.** All LoC estimates in this document use the Difficulty Assessor's honest figures.
2. *Approach C's difficulty is 7/10, not 8/10.* **Incorporated.**
3. *Schedule risk is the dominant concern: 10-16 months for a 4-5 person team.* **Incorporated.** The implementation plan is phased with explicit kill gates at 2 weeks, 2 months, and 6 months.

---

## 3. Extreme Value Delivered

### Who Needs This

**Primary users (200-500 people):**
- MIBLP researchers (50-200 active) who hand-code reformulations for each paper. BiCut gives them: automated reformulation with correctness certificates, the first bilevel-specific cutting planes, and reproducible solver-agnostic benchmarking.
- Applied bilevel practitioners in energy markets, supply chain, and infrastructure planning (150-300) who are blocked by the weeks-long reformulation-debugging-solver cycle.

**Secondary users (100-200 people):**
- OR educators who teach bilevel optimization but cannot assign computational exercises.
- Computational optimization reviewers who need independent verification of bilevel results.

**Honest assessment of user base:** The ceiling is ~500-700 direct users. This is three orders of magnitude smaller than CVXPY's user base. The CVXPY analogy is dropped. BiCut is not "CVXPY for bilevel" — it is a specialized research tool for a specialized community.

### Why the Value Is Real Despite the Small User Base

1. **The cutting-plane contribution has no substitute.** No tool provides bilevel-specific cuts. The gap between MibS's branch-and-cut (no bilevel cuts) and BiCut's tightened formulations is a capability gap, not a convenience gap. Even 10% root gap closure on hard instances translates to meaningful solve-time reduction (fewer branch-and-bound nodes).

2. **Correctness certificates address a real trust deficit.** Bilevel optimization results are notoriously difficult to verify. A reviewer who receives a paper claiming to solve a bilevel problem via KKT reformulation currently has no way to verify that the reformulation is correct without re-deriving it. BiCut's certificates provide the first machine-checkable verification pathway. This is modest in user count but high in per-user value.

3. **The compiler enables a new research direction.** Bilevel cutting-plane research has been blocked by the absence of infrastructure for systematic deployment and evaluation. BiCut makes it possible to test new bilevel cut families (split cuts, Chvátal-Gomory cuts, multi-row cuts for bilevel) as compiler passes without reimplementing the reformulation and emission pipeline. This is platform value, not tool value.

4. **Reproducibility matters disproportionately in small communities.** In a community of 200-500 researchers, a single tool that enables independent verification of results has outsized impact on research quality. The value is not in user count but in the trust infrastructure it provides.

### What Becomes Possible

- First-ever systematic comparison of bilevel cutting-plane families across reformulation strategies and solvers.
- Reproducible bilevel optimization: compiled MILPs are standard `.mps` files verifiable without BiCut.
- Correctness-guaranteed reformulation: no more silent bugs from applying KKT to integer lower levels.
- A concrete platform for the bilevel cutting-plane research program that has been latent for decades.

---

## 4. Technical Architecture

### Subsystem Breakdown (Honest Estimates)

| ID | Subsystem | Total LoC | Novel LoC (Honest) | Risk | Notes |
|----|-----------|-----------|---------------------|------|-------|
| S1 | Bilevel IR & Parser | 8,000 | 500-700 | Low | Expression canonicalization; leader/follower annotations. Standard compiler infrastructure. |
| S2 | Structural Analysis | 5,000 | 700-900 | Low-Med | Three-tier CQ verification (syntactic → LP → sampling). Composition of known techniques is the novel part. |
| S3 | Reformulation Selection | 3,000 | 600-800 | Low | Lookup table + regression-calibrated cost model. Soundness proof is trivial case analysis. |
| S4 | KKT / Strong Duality Passes | 9,000 | 800-1,000 | Medium | Automated big-M via bound-tightening LPs. Numerical edge cases are the hard part. |
| **S5** | **Intersection Cut Engine** | **11,000** | **3,000-3,500** | **HIGH** | **Bilevel-infeasible set characterization; separation oracle; parametric warm-starting; cache.** The core novel contribution. |
| **S6** | **Value-Function Oracle** | **10,000** | **2,500-3,000** | **HIGH** | **Parametric LP for LP lower levels (exact). Sampling approximation for MILP lower levels (best-effort). Lifting is conditional.** |
| S7 | Column-and-Constraint Generation | 3,500 | 200-300 | Low | Known algorithm, standard implementation. |
| S8 | Solver Backend Emission (×3) | 6,000 | 300-400 | Medium | Gurobi + SCIP + HiGHS. Capability profiles + per-solver encoders. CPLEX deferred. |
| S9 | Correctness Certificates | 4,500 | 500-600 | Low | Verification logic is novel; serialization is not. |
| S10 | BOBILib Benchmark Harness | 5,500 | 300-400 | Low | MibS integration, timing, instance classification. |
| S11 | Testing & Validation | 5,000 | 500-700 | Low | Random bilevel instance generation; roundtrip verification. |
| S12 | Cross-Cutting Infrastructure | 8,000 | 0 | Low | Logging, config, CLI, Python bindings. |
| E1 | QP Lower Levels | 12,000 | 1,500-2,000 | Medium | McCormick envelopes + SOCP. Known theory, moderate integration effort. |
| INT | Cut-Compiler Integration | 2,000 | 800-1,200 | Medium-High | BilevelAnnotation interface; bidirectional space mapping; reformulation-aware cut dispatch. **Unique to hybrid.** |
| | **Total** | **~92,500** | **~12,200-15,500** | | |
| | **With test/bench/infra overhead (1.3×)** | **~120,000** | | | |

### Novel vs. Infrastructure Summary

- **Genuinely novel algorithmic logic:** ~12,200-15,500 LoC (the Difficulty Assessor's honest range was ~11,000-13,800 for the core Approach C subsystems; the upper bound extends slightly beyond due to novel logic in S1 parser canonicalization, S7 convergence management, S10 bilevel feasibility verification, and S11 random well-posed instance generation — each contributing 200-700 LoC of non-trivial logic)
- **Standard optimization infrastructure:** ~75,000-80,000 LoC (parser, DAG, reformulation passes, emission, CLI, bindings)
- **Testing and benchmarking:** ~25,000-30,000 LoC

The novel logic concentrates in three subsystems: intersection cut engine (S5), value-function oracle (S6), and cut-compiler integration (INT). These three subsystems contain ~6,300-7,700 novel LoC — the mathematical and algorithmic core of the paper.

---

## 5. New Math Required

### Tier 1: Crown Jewels

**M1. Bilevel-Infeasible Set Polyhedrality Theorem**
- **Statement:** For an MIBLP with LP lower level, the bilevel-infeasible set B̄ = {(x,y) : y ∉ argmin f(x,·)} is a finite union of polyhedra, characterized by the critical regions of the lower-level parametric LP.
- **Why load-bearing:** Without this, the intersection cut framework has no foundation. Ray-boundary intersections require a polyhedral target.
- **Honest difficulty:** B (new formalization of known parametric LP structure; the Math Assessor and Skeptic agree this is partially folklore)
- **Risk of failure:** Low (10%). The result follows from known parametric LP theory. The risk is in getting the formalization tight, not in the result being false.
- **Fallback:** None needed — this is achievable.

**M2. Separation Oracle and Complexity**
- **Statement:** Given an LP relaxation point (x̂,ŷ) violating bilevel feasibility, a valid bilevel intersection cut can be computed in time polynomial in the number of lower-level constraints m for fixed follower dimension d.
- **Why load-bearing:** Establishes the computational regime where cuts are viable. Without it, cuts may be impractical.
- **Honest difficulty:** B− (parameterized complexity argument using known parametric LP bounds, applied carefully to the bilevel setting)
- **Risk of failure:** Low (15%). The local separation (identify critical region at x̂, compute ray intersections) is polynomial per query. The risk is that degenerate instances require testing multiple adjacent critical regions, degrading practical performance.
- **Fallback:** If the complexity result only holds for non-degenerate lower levels, state this restriction clearly. The computational evaluation on BOBILib will reveal what fraction of instances are affected.

**M3. Facet-Defining Conditions for Bilevel Intersection Cuts** *(STRETCH GOAL)*
- **Statement:** Complete characterization of when a bilevel intersection cut defines a facet of conv(S_BF), the convex hull of the bilevel-feasible set.
- **Why load-bearing:** This is the difference between "we applied Balas to bilevel" (incremental) and "we proved new polyhedral structure" (a genuine contribution to cutting-plane theory). This result, if achieved, elevates the paper from JOC to Mathematical Programming.
- **Honest difficulty:** C (genuinely novel; requires constructing dim(P) affinely independent feasible points on the cut hyperplane, where dim(P) itself is non-trivial to establish)
- **Risk of failure:** High (50-60%). The general-case proof may be intractable. Dual degeneracy in the lower level makes the combinatorial structure explode.
- **Fallback:** Prove facet-defining conditions for the restricted class: LP lower levels with unique optimal follower response (non-degenerate). This covers a meaningful fraction of BOBILib and is still a genuine theoretical result (Difficulty B+). If even the restricted-class proof fails, present computational evidence of cut strength without facet theory — the paper becomes computational, not theoretical.

**M4. Finite Convergence of Bilevel Cut Loop**
- **Statement:** Under non-degeneracy of the LP relaxation, the bilevel intersection cut loop terminates in finitely many rounds.
- **Why load-bearing:** Partially — convergence is expected, but the proof handles the non-standard bilevel feasibility structure.
- **Honest difficulty:** B− (standard cutting-plane convergence adapted to union-of-polyhedra; the cut-induced degeneracy issue flagged by the Skeptic requires a perturbation argument)
- **Risk of failure:** Very low (5%). The result follows standard patterns. The perturbation/lexicographic argument to handle cut-induced degeneracy is achievable.
- **Fallback:** State the theorem for the non-degenerate case with a remark about perturbation extensions.

### Tier 1.5: Conditional Results

**M5. Value-Function Lifting** *(CONDITIONAL — pursued only if prototype shows measurable benefit)*
- **Statement:** Extension of Gomory-Johnson subadditivity theory to value-function epigraphs, constructing maximal valid lifting functions from dual vertex enumeration.
- **Why load-bearing:** Would provide stronger cuts than raw intersection cuts by exploiting value-function structure. However, the Math Assessor and Skeptic agree this may be vacuous for LP lower levels (where V(x) is piecewise linear convex, and the lifting function may reduce to the concave envelope of V — a known construction).
- **Honest difficulty:** C for MILP lower levels (genuinely hard, computationally intractable), B+ for LP lower levels (may be trivial)
- **Risk of failure:** High (60%). For LP lower levels, the result may add no cutting power beyond value-function reformulation. For MILP lower levels, the oracle is too expensive for practical deployment.
- **Fallback:** Drop value-function lifting entirely. The intersection cuts stand on their own. Present the lifting idea as "future work" if the concept has merit but the instantiation is thin. The paper does not depend on this result.

### Tier 2: Compiler-Side Results

**M6. Compiler Soundness Theorem**
- **Statement:** If Typecheck(P) succeeds and structural analysis certifies preconditions Φ, then for every reformulation R selected by the selection engine, opt(emit(R)) = opt(P) and solutions map back to bilevel-optimal solutions. Extended: added cuts are valid inequalities preserving all bilevel-feasible solutions.
- **Why load-bearing:** Central correctness guarantee for the compiler. The extension to cover cut validity creates a real dependency on the cut theory (M1).
- **Honest difficulty:** B− for the extended version (harder than textbook case analysis because of the cut-validity dependency; the Math Assessor agreed B is fair for the extended version). The base version without cuts is Difficulty A (known results reorganized).
- **Risk of failure:** Very low (5%).
- **Fallback:** If the cut-validity extension is problematic, prove the compiler soundness without cuts (Difficulty A) and prove cut validity separately (direct consequence of M1).

**M7. Structure-Dependent Selection Soundness**
- **Statement:** The selection function ρ(σ) always selects reformulations whose preconditions are satisfied by σ.
- **Honest difficulty:** A (trivial case analysis; all reviewers agree)
- **Risk of failure:** None.

**M8. Compilability Decision**
- **Statement:** Given a bilevel program P and solver capability profile S, decidability (trivially polynomial) of whether a valid compilation exists.
- **Honest difficulty:** A (finite enumeration over known strategies; the Math Assessor and Skeptic agree the "Difficulty B" claim was inflated)
- **Risk of failure:** None. But the result is also not interesting — present as a useful system property, not as a theorem.

### Summary of Math Portfolio

| Result | Difficulty | Load-Bearing | Risk | Status |
|--------|-----------|-------------|------|--------|
| M1: Polyhedrality theorem | B | Yes | Low | Core deliverable |
| M2: Separation complexity | B− | Yes | Low | Core deliverable |
| M3: Facet-defining conditions | C | Yes (for top venue) | High | **Stretch goal** |
| M4: Finite convergence | B− | Partial | Very Low | Core deliverable |
| M5: Value-function lifting | C/B+ | Conditional | High | **Conditional on prototype** |
| M6: Compiler soundness (extended) | B− | Yes | Very Low | Core deliverable |
| M7: Selection soundness | A | Yes (trivial) | None | Core deliverable |
| M8: Compilability decision | A | Marginal | None | Core deliverable |

**Honest overall math grade: B (with upside to A− if M3 succeeds)**

---

## 6. Implementation Plan with Kill Gates

### Phase 0: Prototype Validation (Weeks 1-2) — MANDATORY KILL GATE

**Goal:** Validate or kill the bilevel intersection cut contribution.

**Tasks:**
- Implement a bare-bones separation oracle for bilevel-infeasible sets (LP lower levels only).
- Test on ≥50 BOBILib instances with LP lower levels.
- Measure root gap closure using SCIP's LP relaxation + custom cut callback.
- Measure separation oracle wall-clock time and cache hit rate.

**Go/No-Go Criteria:**

| Metric | Kill (<) | Pivot | Go (≥) |
|--------|----------|-------|--------|
| Geometric mean gap closure | <5% → kill cuts | 5-10% → descope to computational contribution | ≥10% → full cut theory |
| Separation oracle overhead | >200ms on >50% of calls → kill cuts | 50-200ms → optimize cache | <50ms on ≥80% → proceed |
| Cache hit rate | <50% → kill cuts | 50-80% → redesign cache | ≥80% → proceed |

**Decision tree after Phase 0:**
- **Gap closure ≥15%:** Prioritize cut theory (facet-defining proof becomes primary). Compiler is essential infrastructure, not co-primary. Target: Mathematical Programming.
- **Gap closure 10-15%:** Balanced hybrid. Both compiler and cuts are co-primary. Target: JOC.
- **Gap closure 5-10%:** Compiler-primary with cuts as modest optimization pass. No facet theory attempted. Target: JOC.
- **Gap closure <5%:** Kill cuts entirely. Pivot to compiler-only (Approach B). Target: JOC.

**Assumed team:** 3-4 concurrent developers (at least one strong theorist for cut math, two systems engineers for compiler + backends). The Difficulty Assessor estimated 10-16 months for a 4-5 person team; the phased plan targets 9-month core delivery with stretch goals in months 9-12, which is aggressive but achievable with 3+ concurrent tracks and early prototype gating. The reduced evaluation matrix (~20,800 configurations at ~5 min average = ~72 days serial, ~9 days with 8-core parallelism) fits within Phase 3's 3-month window but leaves limited slack for solver-specific debugging.

### Phase 1: Core Compiler + Cut Foundation (Months 1-3)

**Compiler backbone (parallel track 1):**
- S1: IR & parser
- S2: Structural analysis (syntactic + LP-based CQ verification)
- S3: Reformulation selection (table-based, no cost model yet)
- S4: KKT + strong duality passes
- S8: SCIP backend emission (single solver first)

**Cut engine (parallel track 2):**
- S5: Intersection cut separation oracle (full implementation with parametric warm-starting)
- S6: Value-function oracle for LP lower levels (exact parametric LP)
- INT: BilevelAnnotation interface + cut-compiler integration

**Checkpoint at Month 3:**
- Compiler compiles LP-lower-level BOBILib instances to SCIP via KKT and strong duality.
- Cut engine produces valid bilevel intersection cuts integrated into the compiler pipeline.
- End-to-end test: compile + cut + solve on 100 BOBILib instances.
- If integration is broken (cuts invalid, backward mapping incorrect): 2-week debugging sprint. If unfixable, fall back to cuts-as-standalone (Approach A's architecture).

### Phase 2: Breadth + Depth (Months 4-6)

**Compiler breadth:**
- S7: Column-and-constraint generation pass
- S8: Add Gurobi backend (second solver)
- S9: Correctness certificates
- E1: QP lower-level extension (McCormick + SOCP)
- S3: Cost model calibration on BOBILib

**Cut depth (if gap closure ≥10%):**
- M3: Begin work on facet-defining conditions (restricted class first)
- M5: Evaluate value-function lifting on LP-lower-level instances (conditional)
- S6: MILP lower-level oracle (sampling-based, best-effort)

**Checkpoint at Month 6 — SECOND KILL GATE:**
- Compiler handles all BOBILib instance classes (LP, MILP, pure integer lower levels).
- Two solver backends working (SCIP + Gurobi).
- Certificates operational.
- Gap closure measured on full BOBILib LP-lower-level subset with both solvers.
- If gap closure is solver-dependent (>5 percentage points difference between SCIP and Gurobi): investigate solver-specific cut interaction. If the cuts only help on SCIP, the Skeptic's concern is validated and the cut contribution weakens.
- If facet-defining proof is stuck: descope to computational cut results without facet theory. Paper is still viable at JOC.

### Phase 3: Evaluation + Polish (Months 7-9)

- S8: Add HiGHS backend (third solver) on 200-instance subset.
- S10: Full BOBILib benchmark evaluation (reduced matrix: 2600 × 2 reformulations × 2 solvers × 2 configs).
- S11: Cross-solver verification; bilevel feasibility checks.
- Statistical analysis: geometric means, performance profiles, Wilcoxon tests.
- Strategic bidding case study (simplified IEEE 14-bus).
- Paper writing.

### Phase 4: Stretch Goals (Months 9-12, if schedule permits)

- M3: General-case facet-defining proof (if restricted-class proof succeeded).
- CPLEX backend (fourth solver).
- Full evaluation matrix (all reformulations × all solvers).
- Value-function lifting results (if prototype showed benefit).

---

## 7. Risk Mitigation

### Risk 1: Cuts don't work (gap closure <5%)
- **Probability:** 25% (the Skeptic says 40% for all cut failure modes combined; the prototype gate catches most of this risk at week 2)
- **Mitigation:** The 2-week prototype gate (Phase 0) catches this before significant investment. If cuts fail, pivot to compiler-only (Approach B) with 10+ months of runway. The compiler, certificates, and solver-agnostic emission are independently publishable at JOC.

### Risk 2: Facet-defining proof fails
- **Probability:** 50-60%
- **Mitigation:** Facet-defining conditions are a stretch goal, not a core deliverable. The paper is structured to be strong at JOC with computational cut results only (separation + gap closure + solver comparison). The restricted-class proof (non-degenerate LP lower levels) is the realistic target. If even the restricted-class proof fails, present computational evidence without facet theory.

### Risk 3: Value-function lifting is vacuous for LP lower levels
- **Probability:** 50% (Math Assessor and Skeptic agree this is a serious risk)
- **Mitigation:** Lifting is conditional on prototype validation. If lifted cuts provide <2% additional gap closure over unlifted cuts, drop lifting entirely. The intersection cuts stand on their own. No paper real estate is committed to lifting until validated.

### Risk 4: "Two papers stapled together" / venue rejection
- **Probability:** 20-25%
- **Mitigation:** Target JOC, which explicitly values systems + computation combinations. Structure the paper with a unified narrative: "Section 2: Compiler (infrastructure). Section 3: Cut theory (the math enabled by the infrastructure). Section 4: Evaluation (what the infrastructure + math delivers)." The CVXPY JOC paper is the structural precedent — it combined DCP rules (theory) with a compiler (system) with experiments. Also: do NOT attempt Math Programming unless the full facet-defining proof succeeds.

### Risk 5: BilevelJuMP comparison ("incremental delta")
- **Probability:** 30% (the Skeptic's hostile review is plausible)
- **Mitigation:** Three clear differentiators that BilevelJuMP does not provide: (a) bilevel intersection cuts — novel cutting planes unavailable anywhere; (b) correctness certificates with sound CQ verification — BilevelJuMP has no certificates; (c) value-function reformulation for MILP lower levels — BilevelJuMP does not support this. The paper's framing is NOT "a better BilevelJuMP" — it is "the infrastructure that makes bilevel cutting-plane research possible, demonstrated with the first bilevel-specific cuts." The cutting-plane contribution is the differentiator, not the compiler per se.

### Risk 6: Backward mapping correctness (cut-compiler integration)
- **Probability:** 10% (the Math Assessor's response resolves the theoretical concern)
- **Mitigation:** As the Math Assessor demonstrates, the backward mapping is a projection (x,y,λ,z) → (x,y), which is always well-defined. Cuts are derived and expressed in the original (x,y) space and extended to the MILP space with zero coefficients on auxiliary variables. This is trivially valid. The implementation enforces this invariant: the separation oracle receives (x̂,ŷ) extracted by projection, computes the cut in (x,y)-space, and the emission layer lifts it with zeros. No reformulation-specific backward inversion is needed. Integration testing on BOBILib verifies: compiled MILP + cuts produces the same optimal value as MibS on all instances.

### Risk 7: Degenerate lower levels make separation oracle impractical
- **Probability:** 20%
- **Mitigation:** The Skeptic raises a valid concern about degenerate lower-level LPs requiring enumeration of adjacent critical regions. The Math Assessor correctly notes that the separation oracle operates locally (single LP solve to identify the current critical region), not globally (enumerate all regions). For degenerate instances, the oracle tests a bounded number of adjacent regions proportional to the degree of degeneracy. If degeneracy causes >200ms oracle times on >30% of hard instances, the practical contribution of cuts on degenerate instances is limited — but cuts may still help on the non-degenerate subset (which the paper reports honestly). This is a scope limitation, not a correctness bug.

### Risk 8: Schedule overrun (10-16 month estimate)
- **Probability:** 40%
- **Mitigation:** The phased implementation plan with kill gates ensures that partial delivery is always a viable paper. At month 6, the system must be functional on two backends with cuts working. If behind schedule at month 6: cut HiGHS backend, reduce evaluation matrix, drop QP extension. The minimum viable paper is: compiler (SCIP + Gurobi) + cuts (LP lower levels) + certificates + BOBILib evaluation.

---

## 8. Best-Paper Argument

### Target Venue: INFORMS Journal on Computing (JOC)

**Why JOC:**
- JOC explicitly values software contributions with rigorous evaluation.
- JOC published the BilevelJuMP paper (Dias Garcia et al., 2023) — the reviewer pool knows bilevel optimization tools.
- JOC accepts papers combining methodological innovation with computational evidence.
- The compiler + cuts + evaluation structure fits JOC's format naturally.

### The Best-Paper Case

1. **Novel cutting-plane family.** Bilevel intersection cuts are the first cutting planes derived from optimality-defined infeasible sets. Even without facet-defining conditions, a separation oracle with demonstrated gap closure on 2600+ benchmark instances is a computational contribution that the integer programming community has not seen. With facet-defining conditions (stretch goal), this becomes a theoretical contribution that bridges polyhedral combinatorics and bilevel optimization.

2. **First correctness-certified bilevel compiler.** No existing tool provides machine-checkable proofs that a reformulation preserves bilevel optimality. BiCut's certificates with sound CQ verification are a qualitative advance in bilevel software reliability.

3. **Clean, falsifiable evaluation.** BOBILib provides 2600+ instances with known optima. MibS is the established baseline. The metrics (root gap closure, solve time, node count, cross-solver consistency) are standard and unambiguous. The paper reports all results including negative ones (instances where cuts don't help, reformulations that lose to default KKT).

4. **Enables a new research direction.** BiCut's extensible pass architecture makes it possible to test new bilevel cut families without reimplementing the reformulation pipeline. The paper demonstrates this with bilevel intersection cuts; future work can add bilevel split cuts, bilevel Chvátal-Gomory cuts, etc.

### Addressing the Skeptic's Hostile Review

The Skeptic's simulated hostile review for Approach C raises three major concerns:

> **Concern 1:** "The cutting-plane theory is promising but incomplete."

**Response:** The paper presents the polyhedrality theorem (M1), separation complexity (M2), finite convergence (M4), and computational gap closure results. If the facet-defining proof succeeds (M3), it is included; if not, the computational results stand on their own. JOC does not require facet-defining conditions for a cutting-plane paper — it requires demonstrated computational effectiveness. The paper is honest about what is proved and what is conjectured.

> **Concern 2:** "The compiler architecture is not novel relative to BilevelJuMP."

**Response:** The delta is: (a) bilevel intersection cuts — novel, (b) correctness certificates — novel, (c) value-function reformulation — not in BilevelJuMP. The paper does not claim the compiler is novel in isolation. The claim is that the compiler + cuts + certificates combination is novel and that the compiler is necessary infrastructure for systematic bilevel cutting-plane research.

> **Concern 3:** "The paper is too long — 45+ pages."

**Response:** The paper is structured for JOC's format: 25 pages of main text with an online supplement. Compiler architecture is presented compactly (5-6 pages). Cut theory is presented in detail (8-10 pages). Evaluation is thorough (8-10 pages). Full proofs, implementation details, and extended benchmarks are in the supplement.

### Reach Venue: Mathematical Programming Series A

If and only if the facet-defining proof (M3) succeeds in full generality (not just the restricted class), the paper can be repositioned for Math Programming. In this scenario, the paper leads with the cut theory and presents the compiler as the infrastructure that enables the computational evaluation. The compiler architecture moves to a companion technical report or supplement. This is a 20% probability scenario.

---

## 9. Honest Assessment

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **5/10** | Small user base (500-700 ceiling) caps value. But the cutting-plane contribution has no substitute, and the correctness certificates are a qualitative advance for the community. Raised from depth check's 4 by incorporating the reproducibility-layer framing and the QP extension. |
| **Difficulty** | **7/10** | The intersection cut engine (S5) and value-function oracle (S6) require genuinely novel algorithms. The cut-compiler integration (INT) is a novel systems design problem. The compiler infrastructure is largely precedented. The facet-defining proof (if attempted) is Difficulty C. Per the Difficulty Assessor's honest estimate. |
| **Potential** | **6/10** | A strong JOC paper with clear shot at best paper if the evaluation is exceptional. Not a Math Programming paper without the facet proof. The synergy argument (compiler enables cuts, cuts validate compiler) is genuine but not overwhelming. The "two papers stapled" risk is mitigated by JOC targeting. |
| **Feasibility** | **6/10** | Higher than the approaches document's 5/10 for Approach C because: (a) value-function lifting is descoped to conditional, (b) CPLEX is deferred, (c) the prototype gate catches existential risks at week 2, (d) the fallback to compiler-only is well-defined. Lower than Approach B's 8/10 because the cut theory still carries real risk. |

### Kill Probability

**30-35%** probability of failing to produce a publishable paper at any venue. (The Skeptic estimated 40% for Approach C. The 5-8 percentage point reduction reflects: (a) prototype gate catching ~5% of existential cut risk at week 2 before major investment; (b) descoping value-function lifting removes ~2-3% of correlated failure risk; (c) reduced evaluation matrix from 80K to 20K configurations removes ~1-2% of schedule risk. The remaining gap is within estimation uncertainty — the honest range is 30-40%.)

Breakdown:
- Cuts fail at prototype gate (25%) → pivot to compiler-only → compiler-only paper rejected as incremental over BilevelJuMP (40-50% conditional) → 10-12.5% total kill from this path.
- Cuts work but integration fails (10%) → fall back to cuts-standalone with SCIP-only → weaker paper but publishable → 3% total kill.
- Both work but paper is rejected as "split into two papers" (15%) → resubmit to different venue → eventual publication → 5% total kill.
- Schedule overrun prevents complete evaluation (15%) → reduced evaluation still publishable → 3% total kill.
- Remaining paths converge → ~30-38% total kill probability (midpoint ~34%).

### Target and Fallback Venues

| Scenario | Primary Venue | Fallback Venue |
|----------|---------------|----------------|
| Full success (facet proof + gap closure ≥15%) | Mathematical Programming | Operations Research |
| Standard success (computational cuts + gap closure ≥10%) | INFORMS JOC | CPAIOR |
| Modest success (cuts work but gap closure 5-10%) | INFORMS JOC | Optimization Methods & Software |
| Cuts fail entirely | INFORMS JOC (compiler-only) | Computers & OR |

### What Success Looks Like

- Bilevel intersection cuts achieve ≥10% geometric mean root gap closure on LP-lower-level BOBILib instances.
- The compiler compiles all BOBILib instances to at least two solvers with correctness certificates.
- Cross-solver results agree within tolerance on ≥95% of instances.
- The reformulation selection engine matches or beats default KKT on ≥80% of instances (with ≥5 instances showing ≥2× speedup).
- The paper tells a unified story at JOC and receives "accept with minor revisions."
- If the facet-defining proof lands: the paper is a strong Mathematical Programming submission.

### What Failure Looks Like

- Gap closure <5% at prototype gate. Pivot to compiler-only. Compiler-only paper is submitted to JOC and receives "incremental over BilevelJuMP" reviews.
- Integration bugs invalidate cut-compiler bridge. Months of debugging. Late pivot to standalone cuts or standalone compiler.
- Facet-defining proof fails and gap closure is only 8%. The paper is "we tried bilevel cuts and they help a bit; also here's a compiler." Published at OMS or Comp&OR, not JOC.
- Schedule overrun: at month 12, the system works on one solver, cuts work on LP lower levels, and the evaluation is incomplete. A rushed paper with thin evaluation is submitted and receives "needs more experiments" reviews.

---

## 10. Amendments from Debate

All binding amendments from the depth check panel are incorporated. Additionally, the following amendments emerge from the debate:

### Binding Amendments

1. **Prototype gate is mandatory (2 weeks).** Implements the intersection cut separation oracle on ≥50 BOBILib LP-lower-level instances. Gap closure <5% kills the cut contribution. No exceptions.

2. **Facet-defining conditions are a stretch goal, not a core deliverable.** The paper is viable at JOC with computational cut results only. Facet-defining conditions elevate to Math Programming only if proved in at least the restricted class.

3. **Value-function lifting is conditional on prototype evidence.** Pursued only if lifted cuts show ≥2% additional gap closure over unlifted cuts on LP-lower-level instances. Otherwise dropped entirely.

4. **The O(h²) error bound claim for MILP value-function approximation is corrected.** For LP lower levels: exact piecewise-linear evaluation. For MILP lower levels: L¹ error bounds only. Pointwise O(h²) is not claimed.

5. **The CVXPY analogy is dropped.** BiCut is not positioned as "CVXPY for bilevel." It is positioned as a correctness-certified bilevel compiler with novel cutting planes.

6. **The Necula (PCC) comparison for certificate composition is dropped.** Certificates are presented as an engineering/correctness contribution, not as a formal-methods research result.

7. **Compiler soundness is presented as a formalization of known results (Difficulty A for base, B− for extended), not as novel mathematics.**

8. **Compilability decision is presented as a system property, not as a theorem.** It is trivially polynomial and presenting it as "Difficulty B" was dishonest.

9. **The evaluation matrix is reduced to feasible scale.** Primary: 2600 × 2 reformulations × 2 solvers × 2 configs = ~20,800 runs. HiGHS cross-validation on 200-instance subset. Full matrix only if time and compute permit.

10. **Cross-solver validation is mandatory for the cut contribution.** If cuts only help on SCIP but not Gurobi, the Skeptic's concern about solver-specific artifacts is validated and the cut contribution is significantly weakened. Both Gurobi and SCIP must show gap closure ≥10% for the cut contribution to stand.

### Recommended Amendments

11. **Backward mapping invariant.** All cuts are derived and expressed in the original (x,y) bilevel variable space. Extension to the MILP space uses zero coefficients on auxiliary variables (λ, z). This is enforced by the implementation and tested by cross-referencing compiled-with-cuts optimal values against MibS.

12. **Degeneracy handling is reported honestly.** The paper reports which instances have degenerate lower levels, what the separation oracle's performance is on degenerate vs. non-degenerate instances, and whether gap closure differs between the two classes.

13. **BilevelJuMP comparison is head-to-head.** The evaluation includes a direct comparison on instances that both tools can handle: same bilevel model compiled by BiCut and by BilevelJuMP, solved by the same solver, reporting the same metrics. The paper is honest about where BilevelJuMP is comparable and where BiCut's delta matters.
