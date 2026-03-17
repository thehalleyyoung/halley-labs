# Mathematician Evaluation: Penumbra (fp-diagnosis-repair-engine) — Post-Theory Stage

**Evaluator:** Deep Mathematician (quantity/quality of NEW, LOAD-BEARING math)
**Method:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) → independent proposals → adversarial cross-critique → synthesis → independent verifier signoff
**Date:** 2026-03-08
**Proposal:** proposal_00 — "Penumbra: Diagnosis-Guided Repair of Floating-Point Error in Scientific Pipelines"

---

## Executive Summary

**Composite: 5.1/10 — CONDITIONAL CONTINUE (Weak)**

| Axis | Score | One-Line |
|------|-------|----------|
| 1. Extreme Value | **4.5/10** | Real but niche; LAPACK blind spot narrows the effective audience further |
| 2. Genuine Difficulty | **5.5/10** | Engineering-hard at breadth; not algorithm-hard; zero research difficulty demonstrated |
| 3. Best-Paper Potential | **3.5/10** | Zero artifacts; plausible SC tool paper conditional on everything working; no crystalline theorem |
| 4. Laptop-CPU & No-Humans | **7.5/10** | CPU-bound by nature; 32GB sufficient with streaming; fully automated |
| 5. Feasibility | **4.5/10** | Penumbra-Lite (15-24K LoC) is realistic; full proposal (51-87K) is not from zero |

**Verdict: CONDITIONAL CONTINUE with hard kill gates. Week 2: ≥3 pipeline-level bugs or ABANDON immediately.**

---

## The Central Finding: The Math Is Supportive, Not Load-Bearing

The most important finding of this evaluation: **Penumbra's theorems (T1–T4, C1, τ) do not drive the system.** The tool delivers ~80% of its value without any theorem. Every component operates on numerical heuristics, thresholds, and direct computation:

- Shadow instrumentation → established technique (Satire 2023)
- Sensitivity computation → numerical methods textbook (Moré-Wild 2011, finite differences)
- Diagnosis classifiers → threshold-based pattern matching on Higham's (2002) informal taxonomy
- Repair selection → 30-pattern lookup table (acknowledged by proposal as "not a synthesizer")
- Certification → interval arithmetic inclusion property (Moore 1966)

The proposal itself admits this at the most critical juncture: **"T4 provides the *justification*, not the *mechanism*"** (final_approach.md §4, T4 section). When the authors say the math doesn't drive the system, believe them.

**theory_bytes = 0.** The theory stage produced zero bytes of formal output. The State.json records `theory_bytes: 0` with the theory stage completing in approximately 3 seconds — indicating a process crash rather than failed mathematical work. This is mitigated as an execution failure rather than a substance failure, but the outcome is the same: **no definitions formalized, no lemmas stated, no proof obligations discharged.** All claimed theorems exist only as proof sketches in planning documents.

---

## Theorem-by-Theorem Assessment

### T1: EAG Soundness — Routine, Load-Bearing Floor (95% achievable)

**Statement:** |ε_out| ≤ Σ_{paths} (Π_{edges} w(i,j)) · |ε_source|, under first-order assumption.

**Assessment:** Standard forward error propagation (Higham Ch. 3) applied to a DAG. This is not novel mathematics — it is a known bound applied to a new data structure. However, it is **load-bearing as the floor**: without T1, the EAG's edge weights have no formal semantics. T1 gives meaning to path-weight products and enables the quantitative causal attribution that differentiates Penumbra from Satire's magnitude-only localization.

**Critical weakness:** The first-order assumption ε·n·max(Lᵢ) ≪ 1 breaks down for condition numbers 10⁸–10¹⁶ — precisely the ill-conditioned problems where users need diagnosis most. The formal backing disappears exactly when users need it. The proposal acknowledges this honestly but does not resolve it.

**Load-bearing?** Yes, as infrastructure. No, as a contribution.

### T2: EAG Decomposition — Abandoned Conjecture (40% achievable, demoted correctly)

**Assessment:** Correctly demoted to open problem with empirical treewidth data. The additive-multiplicative mismatch (graphical model decomposition assumes additive potentials; FP error propagation is multiplicative) is a genuine mathematical obstacle. The Skeptic's prior assessment that T2 as a central contribution was "submission-killing" was correct and acted upon.

**The treewidth measurements remain valuable as cheap empirical data** — first-ever characterization of error-flow graph complexity in real scientific code. Achievable at 90% once the EAG builder works.

### T3: Taxonomic Completeness — Formalized Bookkeeping (99% achievable)

**Assessment:** Exhaustive case analysis over IEEE 754 rounding. The depth check correctly identified this as "formalized bookkeeping." Five classifiers operating on graph subgraphs with threshold-based detection. The formal definition adds rigor but the underlying logic is Higham's textbook.

**One underappreciated aspect:** This is the first formal taxonomy of FP error patterns. Higham describes these patterns informally; T3 exhaustively partitions the IEEE 754 operation space and proves coverage. This is an intellectual contribution — making implicit practitioner knowledge auditable — even if the math is shallow.

**Load-bearing?** No. The classifiers work identically whether or not T3 is stated as a theorem.

### T4: Diagnosis-Guided Repair Dominance — Central Claim, Unproven, Partially Suspect (50-60% achievable for useful version)

**Statement:** On monotone error-flow DAGs, greedy repair in descending error-contribution order is step-optimal via submodularity (Nemhauser-Wolsey-Fisher 1978).

**The cross-critique revealed a critical underspecification:** T4's truth depends on the **repair model**, which the proposal never defines precisely:

- **Repair = eliminate all error at a node:** f is modular (trivially submodular). T4 is provably correct but says nothing beyond "fix stuff in any order." This version is a truism.
- **Repair = reduce error by a fixed fraction:** Marginal gains are constant on monotone DAGs with additive error composition. Submodularity holds. This is the achievable middle ground — meaningful enough to publish, provable with focused effort.
- **Repair = apply a specific algebraic rewrite:** Error reduction depends on specific values, which change when other nodes are repaired. Submodularity is almost certainly false in general. The "disjoint shares" claim fails on reconvergent DAGs with heterogeneous repairs.

**The Skeptic's strongest attack:** Even if T4 holds for the fraction-reduction model, "fix worst first" is what every practitioner already does. The (1−1/e) guarantee adds nothing practically for k ≤ 10 repairs where exhaustive search is feasible.

**The Synthesizer's rescue attempt:** Extending T4 to non-monotone DAGs via Buchbinder et al. (FOCS 2012, 1/2-approximation for unconstrained non-monotone submodular maximization) was ruled a red herring by the cross-critique — it requires submodularity as a precondition, which is the unresolved question.

**Revised achievability:** 50-60% for the fraction-reduction model on monotone DAGs. 20-30% for a version covering realistic algebraic rewrites.

**Load-bearing?** Conditionally. If proved in a meaningful form, T4 answers the reviewer question "why not try all rewrites?" If proved only in the trivial form, it's paper-dressing.

### τ: Tightness Ratio — The Swing Factor (70% achievable for series-parallel, NEVER MEASURED)

**Assessment:** τ(G) = |ε_actual| / (T1 bound) is the most mathematically interesting object in the proposal. If τ > 0.1 on real programs, the EAG is an accurate causal model — path decomposition *explains* error flow, not just bounds it. If τ ≈ 0, the path decomposition is vacuously loose and the EAG degrades to a fancier shadow-value viewer.

**τ has never been measured on a single real program.** Kill-gated at τ < 0.01 by week 4.

**Load-bearing?** τ is the one element that could make the math genuinely load-bearing. Without it, the EAG's quantitative claims are hollow.

### C1: Certification Correctness — Routine (95% achievable)

Follows from the inclusion property of interval arithmetic. Not a contribution.

---

## Hidden Mathematical Value (Synthesizer's Findings)

The Scavenging Synthesizer identified three mathematical contributions buried in the "engineering." The cross-critique adjudicated each:

### Error Automatic Differentiation (EAD): Novel Combination, Not Novel Primitive

Computing ∂ε̂ⱼ/∂ε̂ᵢ (sensitivity of error at j to error at i) is genuinely distinct from standard AD (∂f/∂x) and standard sensitivity analysis (∂output/∂parameter). It requires: (a) defining ε̂ᵢ via MPFR shadow, (b) perturbing ε̂ᵢ specifically, (c) measuring response at ε̂ⱼ. However, Verificarlo's stochastic arithmetic perturbation analysis does something closely related. **Verdict:** Legitimate new computational procedure worth a 2-page formalization; not a deep mathematical primitive.

### Tropical Geometry Connection: Elegant Observation, Not a Contribution

T1's path-product bound in the log domain IS the tropical shortest-path computation. The isomorphism is real. But tropical shortest-path = standard longest-path in the original domain, which the EAG already computes. **Verdict:** A one-line observation. No new algorithm enabled.

### Structural Causal Model Framing: Overreaching

The EAG's first-order finite-difference approximations are not the stable structural equations Pearl's SCM framework requires. Repairing a node changes error dynamics non-locally on reconvergent DAGs, violating modularity. **Verdict:** Suggestive analogy, not formally sound.

---

## The EAG's Novelty: Genuine But Modest (5/10)

The Skeptic demanded: "Name one algorithm the EAG enables that you can't do with Satire's shadow values."

**Answer: Causal path decomposition.** Given output error, the EAG decomposes it into per-path contributions ("73% of output error flows through path A→B→C with amplification 10⁴"). Satire gives per-operation magnitudes but cannot attribute output error to propagation paths because it lacks inter-operation sensitivity edges. This is a genuine capability gap.

**However:** Fluctuat's zonotope decomposition provides similar (and formally sounder) per-contribution attribution for C programs. The EAG's novelty is relative to the Python ecosystem and dynamic analysis, not absolute. The PDG/SSA/e-graph comparison is indefensible and must be dropped.

---

## Fatal Flaws

| ID | Flaw | Severity | Resolution |
|----|------|----------|------------|
| **F1** | theory_bytes = 0 after theory stage | NEAR-FATAL | Process crash (3s runtime), not substance failure. T4 must be attempted in first 4 weeks or formally abandoned. |
| **F2** | impl_loc = 0 after two stages | NEAR-FATAL | Mitigated only by Penumbra-Lite scope cut (15-24K LoC). Full proposal (51-87K) unrealistic from zero. |
| **F3** | BC4 unvalidated: ≥5 pipeline-level bugs are an existence claim | POTENTIALLY FATAL | Resolved by week-2 scouting gate. Threshold reduced to ≥3. |
| **F4** | T4 repair model undefined | SERIOUS | Must specify before proof attempt. "Fraction-reduction on monotone DAGs" is achievable target. |
| **F5** | EAG novelty overstated vs. Fluctuat | SERIOUS | Drop PDG/SSA comparison. Acknowledge Fluctuat's stronger soundness for C. |
| **F6** | "10× error reduction" is fabricated | SERIOUS | Remove from all materials. Replace with measured data once it exists. |
| **F7** | First-order assumption fails on ill-conditioned problems | SURVIVABLE | Automatic fallback to direct shadow comparison. Honest scope limitation. |
| **F8** | 30-pattern "synthesis" is a lookup table | SURVIVABLE | Reframe as "repair selection/prescription." |

---

## Expert Votes

| Expert | Vote | Composite | Key Position |
|--------|------|-----------|-------------|
| Independent Auditor | CONDITIONAL CONTINUE | 5.6/10 | Math is supportive; T4 at 65%; continue if BC4 validates |
| Fail-Fast Skeptic | **ABANDON** | 3.8/10 | Nothing exists; T4 is wrong/trivial; EAG is not novel; Fluctuat already does this |
| Scavenging Synthesizer | CONTINUE | 7.0/10 | Hidden gems (EAD, tropical, SCM); Causal Error Flow Theorem would be best-paper |
| **Chair (consensus)** | **CONDITIONAL CONTINUE (Weak)** | **5.1/10** | Asymmetric risk of 2-week gate justifies continuation |

**Verifier ruling:** APPROVED. All scores within ±0.5 of independently defensible values. No reasoning errors found. Two moderate missing issues noted (NumPy dispatch evolution, ground-truth circularity) — neither changes the verdict.

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| P(top venue: SC/FSE/OOPSLA) | **18–25%** |
| P(best-paper at any venue) | **2–4%** |
| P(any publication incl. workshop) | **38–48%** |
| P(project abandoned) | **35–45%** |
| P(T4 proved, useful version) | **45–55%** |
| P(≥3 real pipeline-level bugs found) | **50–65%** |
| P(merged upstream PR) | **12–20%** |

---

## Binding Conditions and Kill Gates

| ID | Condition | Gate | Kill Action |
|----|-----------|------|-------------|
| **G1** | ≥3 pipeline-level bug candidates with reproduction scripts | **Week 2** | ABANDON — no negotiation |
| **G2** | Working EAG on ≥1 real pipeline (end-to-end trace → graph) | **Week 6** | ABANDON — core contribution unachievable |
| **G3** | T4 proof for fraction-reduction model, OR formal downgrade to empirical-only | **Week 8** | Commit to pure tool paper, no formal claims |
| **G4** | EAG diagnosis outperforms magnitude-sorting on ≥1 real bug | **Week 8** | Pivot to empirical study or ABANDON |
| **G5** | ≥1 repair demonstrably outperforms Herbie + Satire on a real bug | **Week 10** | ABANDON — no demonstrated advantage |

### Decision Tree

```
Week 2: G1 (≥3 bugs?)
  ├─ NO → ABANDON (cost: 2 weeks)
  └─ YES → Week 6: G2 (EAG works?)
       ├─ NO → ABANDON (cost: 6 weeks)
       └─ YES → Week 8: G3 + G4
            ├─ T4 proved + EAG > magnitude → Full Penumbra path (SC + optional PLDI)
            ├─ No T4 + EAG > magnitude → Penumbra-Lite at SC (pure tool paper)
            ├─ T4 proved + EAG ≈ magnitude → Reframe as empirical study
            └─ No T4 + EAG ≈ magnitude → ABANDON (cost: 8 weeks)
```

---

## What Would Change Everything

1. **Finding a real, previously-unknown SciPy bug and getting a PR merged.** This single result transforms the paper from "we claim this works" to "SciPy maintainers agree this works." Raises Value to 7, Best-Paper to 6-7. This is the correct north star.

2. **Proving T4 for a realistic repair model.** Not the trivial version (eliminate all error) and not the impossible version (arbitrary rewrites), but the middle ground: fraction-reduction on monotone DAGs with empirical evidence that real repairs approximate this model. Raises Difficulty to 7, Best-Paper to 5.

3. **τ > 0.1 on ≥3 real programs.** Transforms the EAG from "visualization with no formal tightness" to "accurate causal model." Would justify the "new program representation" framing.

---

## The Mathematician's Bottom Line

As a deep mathematician evaluating load-bearing math: **the math here is thin.** T1 is Higham Chapter 3 applied to a DAG. T3 is exhaustive case analysis. T4 is either a truism or unproven. T2 was abandoned. τ was never measured. C1 is Moore 1966. The Synthesizer's hidden gems (EAD, tropical, SCM) add texture but not depth — the cross-critique correctly classified them as "novel combination," "elegant observation," and "overreach" respectively.

This is fundamentally a **tool paper** — a well-conceived engineering project targeting a genuine gap in the Python scientific computing ecosystem. Its value will come from bugs found and fixed, not from theorems proved. The math is proportionate for SC (where Satire won distinguished paper with no deep theorems), but insufficient for PLDI/POPL.

The project deserves 2 more weeks of life to test its existential hypothesis (do enough real pipeline-level bugs exist?). After that, the kill gates determine whether it lives or dies.

**CONDITIONAL CONTINUE (Weak). Composite 5.1/10. P(top venue) ≈ 18-25%. P(best-paper) ≈ 2-4%.**
