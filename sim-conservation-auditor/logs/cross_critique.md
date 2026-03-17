# Cross-Critique: ConservationLint (sim-conservation-auditor)

**Team Lead synthesis of adversarial cross-critique**
**Evaluators:** Auditor · Skeptic · Synthesizer
**Date:** 2025-07-18

---

## 1. SCORE COMPARISON TABLE

The Auditor and Synthesizer use the same 4-axis / 1–10 scale. The Skeptic uses a different 8-dimension / 1–5 scale. Below, Skeptic scores are shown on their native scale with a normalized /10 equivalent in parentheses for the closest-matching axis.

### Primary Axes (shared by Auditor & Synthesizer)

| Axis | Auditor | Synthesizer | Skeptic (closest match, normalized) |
|------|---------|-------------|--------------------------------------|
| **Extreme & Obvious Value** | **5/10** | **6/10** | Demand validation 1/5 (2/10); Novelty 4/5 (8/10) — split verdict |
| **Genuine Difficulty** | **6/10** | **7/10** | Engineering feasibility 2/5 (4/10); Math depth 2.5/5 (5/10) |
| **Best-Paper Potential** | **5/10** | **5/10** | Overall best-paper 2.5/5 (5/10) |
| **Laptop CPU + No Humans** | **7/10** | **8/10** | Not scored directly; analysis says "plausible for happy path" ≈ 6–7/10 |
| **Composite** | **5.75** | **6.5** | **~5.0** (estimated) |

### Skeptic's Full Scorecard (native 1–5 scale)

| Dimension | Score | /10 equiv. |
|-----------|-------|------------|
| Novelty of core idea | 4 | 8 |
| Mathematical depth | 2.5 | 5 |
| Engineering feasibility | 2 | 4 |
| Evaluation credibility | 1.5 | 3 |
| Demand validation | 1 | 2 |
| LLM resilience | 2.5 | 5 |
| Scope realism | 1 | 2 |
| Overall best-paper potential | 2.5 | 5 |

### LoC Estimates (all three evaluators)

| Evaluator | Estimated Real LoC | % Inflation from 162K |
|-----------|-------------------|-----------------------|
| Auditor | ~94K | ~42% |
| Synthesizer | ~92K | ~43% |
| Skeptic | ~15–25K (paper-scope) | N/A (advocates radical cut) |

---

## 2. AREAS OF AGREEMENT (High-Confidence Findings)

These findings are supported by all three evaluators and should be treated as established:

### A1. The core idea is genuinely novel
All three confirm that bridging Noether's theorem with program analysis of simulation code is unexplored territory. No counterexample of a prior tool was found. The Auditor calls it "a genuinely novel and elegant idea," the Skeptic says "genuinely new combination; the 'bridge' is real" (4/5 novelty), and the Synthesizer identifies it as "the most durable idea in the proposal."

### A2. Code→math extraction is the existential risk
All three identify Subsystem 1 (code→math lifting) as the project's load-bearing assumption and its weakest link. Specific shared concerns: NumPy broadcasting semantics, opaque library calls (SciPy, JAX), pattern-lifting brittleness, and the massive effort to model library function semantics.

### A3. The 162K LoC estimate is inflated ~40–45%
Auditor estimates ~94K real LoC; Synthesizer estimates ~92K. Both arrive at nearly identical subsystem-level reductions. The Skeptic calls 162K "more code than the entire Rust compiler at v1.0" and recommends cutting to 15–25K for a paper.

### A4. The 40–60% coverage claim is fabricated
All three note: no measurement exists, no code exists, the liftable fragment is not yet defined, and the cited codebases (Dedalus, JAX-MD) use opaque library calls that defeat the proposed extraction. The Skeptic's language is strongest: "This number is fabricated."

### A5. T2 (Computable Obstruction Criterion) is the strongest theorem
Auditor: "The best of the three theorems" (6/10 novelty). Skeptic: "Potentially the most interesting result." Synthesizer: identifies it as one of three "genuinely hard subproblems." All three agree its value depends on the efficiency of the criterion and honest framing of the truncation limitation.

### A6. T3 (Liftable Fragment) is a definition, not a theorem
Auditor: "A definition dressed as a theorem" (3/10 novelty). Skeptic: "Exactly 'programs we can handle' dressed up as a formal result." Synthesizer: "the pure math community would find [it] uninteresting (it's a PL/SE contribution wearing math clothing)."

### A7. Self-constructed benchmarks are insufficient
All three flag the circularity of evaluating the tool on benchmarks designed by the tool's builders. All recommend external or adversarial benchmarks sourced from real bug trackers (LAMMPS, GROMACS, CESM).

### A8. LLMs are a real competitive threat
All three acknowledge that LLMs can already handle simple-to-medium conservation analysis. ConservationLint's moat is formal guarantees — but only if the liftable fragment is large enough to matter.

### A9. Scope must be radically reduced
All three recommend cutting runtime monitoring, repair synthesis, and IDE/LSP integration. All agree the core story is: extraction → symmetry analysis → backward error → localization.

### A10. Realistic venue is OOPSLA acceptance; best-paper is aspirational
Auditor: "A solid OOPSLA or CAV paper, not a best-paper winner." Skeptic: "Overall best-paper potential: 2.5/5." Synthesizer: "Strong venue paper; best-paper requires real-bug demo."

### A11. The benchmark suite has standalone durable value
All three identify the conservation-annotated benchmark suite as independently publishable and useful regardless of whether the analysis tool succeeds.

### A12. The repair engine should be cut
Auditor: "Template matching, not synthesis" (low novelty). Skeptic: lists repair among items to drop. Synthesizer: "Weakest component — template-based repair with 30 patterns is engineering, not research."

---

## 3. AREAS OF DISAGREEMENT (Contested)

### D1. Value of T1 (Provenance-Tagged BCH Expansion)

**This is the sharpest disagreement across the three evaluations.**

| Evaluator | Position | Score |
|-----------|----------|-------|
| **Auditor** | "Bookkeeping on known expansions." Useful engineering, not deep math. | 4/10 novelty |
| **Skeptic** | "Bookkeeping on known mathematics, not a new theorem." Not publishable standalone. | Agrees with Auditor |
| **Synthesizer** | "Crown Jewel #2" and "the strongest mathematical contribution." High survivability as standalone theory paper. | Disagrees sharply |

**Auditor/Skeptic argument:** The BCH expansion itself is textbook (Blanes, Casas & Murua 2008). "Tagging" which terms come from which operators is implicit in the expansion — anyone writing `exp(A)exp(B)exp(C)` can read off attributions. This is a notation, not a theorem.

**Synthesizer counter-argument:** The novelty is in the rigorous treatment of *heterogeneous* (mixed-order) compositions, which existing BCH results don't cover. Classical results assume all sub-integrators are the same order. Real codes compose methods of different orders. The provenance structure in the mixed-order case is genuinely non-trivial.

**Resolution:** The Auditor/Skeptic position is stronger for *homogeneous* compositions, where tagging is indeed implicit. The Synthesizer is correct that the *mixed-order* case is under-explored in existing literature. **T1's value depends entirely on whether the mixed-order generalization yields non-trivial structural insights** (e.g., "the leading symmetry-breaking term always comes from the lowest-order sub-integrator"). If the proof is just mechanical BCH bookkeeping with labels, Auditor/Skeptic win. If the mixed-order analysis reveals unexpected structure, the Synthesizer wins. **Verdict: T1 is conditionally valuable — the mixed-order case must be proved first before claiming it as a crown jewel.**

### D2. Overall Value Score (TAM Size)

| Evaluator | Score | Position |
|-----------|-------|----------|
| Auditor | 5/10 | ~5K developers, not their #1 pain, LLM competition |
| Synthesizer | 6/10 | Real but narrow; reframe as research verification tool |
| Skeptic | Demand 1/5 (2/10) | Zero evidence of demand; culture problem, not tool gap |

**Skeptic's strongest argument:** "A simpler intervention (a conservation-test template library for pytest) might capture 80% of the value at 1% of the cost." No user interviews, no demand validation.

**Synthesizer's strongest counter:** The demand narrative should target numerical methods researchers verifying their own implementations — a natural, immediate audience that matches the Python scope.

**Resolution:** The Skeptic's demand criticism is legitimate — zero validated demand is a real gap. But the Auditor's and Synthesizer's position that the pain is real (Wan et al. is compelling) also holds. The issue is not whether the pain exists but whether the proposed solution matches the audience. **Verdict: Value is ~5/10. The demand narrative must be reframed (per Synthesizer's Amendment 1) and validated (per Skeptic's Condition 6).**

### D3. Genuine Difficulty

| Evaluator | Score | Position |
|-----------|-------|----------|
| Auditor | 6/10 | "Hard senior engineer project, not engineering breakthroughs" |
| Synthesizer | 7/10 | 3 of 7 subproblems genuinely hard |
| Skeptic | Eng. feasibility 2/5 | Code→math extraction underestimated by 5–10× |

**Key tension:** The Auditor and Skeptic agree that the *claimed* difficulty is inflated, but disagree on the *actual* difficulty. The Auditor thinks ~30K LoC is genuinely novel (moderate difficulty). The Skeptic thinks the extraction problem is so hard it may be infeasible, which paradoxically makes it *more* difficult than claimed — but in a way that threatens the project rather than validating it.

**Resolution:** The Synthesizer's framing is most accurate: the project has 3 genuinely hard subproblems (extraction, T2, localization) and 4 well-understood ones. The difficulty is real but concentrated. **Verdict: 6/10 — genuinely challenging in the core pipeline, but the inflation around peripheral subsystems dilutes the honest difficulty signal.**

### D4. Laptop CPU Feasibility

| Evaluator | Score | Position |
|-----------|-------|----------|
| Auditor | 7/10 | Core tractable; BCH order 6 with k=10 is borderline (~17 min) |
| Synthesizer | 8/10 | "Genuine strength"; symbolic throughout |
| Skeptic | ~6–7/10 (est.) | "Plausible for happy path; worst-case unknown" |

**Resolution:** Minor disagreement. All agree the symbolic computation is laptop-tractable. The Auditor's specific calculation (order 6, k=10 ≈ 17 minutes exceeding 10-min budget) is the most rigorous. The Synthesizer may be slightly generous. **Verdict: 7/10. The 10-minute budget holds for typical cases (k≤5, p≤4) but fails for extreme parameters. Noether's Razor baseline is likely invalid CPU-only (all three agree).**

### D5. Scope Reduction Target

| Evaluator | Target LoC | What to Keep |
|-----------|-----------|--------------|
| Auditor | Not specified (minimal) | Core pipeline on single domain (Hamiltonian splitting in Python) |
| Skeptic | 15–25K | Extraction + symmetry + backward error + localization on <500 LoC programs, 5–10 benchmarks |
| Synthesizer | ~72K | Full static pipeline (7 subsystems), 15 benchmarks, no monitor/repair/LSP |

**Key tension:** The Skeptic's 15–25K target produces a proof-of-concept; the Synthesizer's 72K target produces a publishable system. The Skeptic prioritizes validation before building; the Synthesizer prioritizes a complete story.

**Resolution:** The Skeptic's scope is appropriate for the *validation phase* (prove extraction works, prove T2, measure coverage). The Synthesizer's scope is appropriate for the *paper-submission phase*. These are sequential, not contradictory. **Verdict: Phase 1 should target ~20K (Skeptic's scope) to validate feasibility. Phase 2, conditional on Phase 1 success, expands to ~70K (Synthesizer's scope).**

---

## 4. SKEPTIC CHALLENGES TO AUDITOR

### 4.1 The Auditor was too generous on Value (5/10)

**Challenge:** The Auditor grants 5/10 without addressing the total absence of demand evidence. The Wan et al. anecdote is compelling but singular. The Auditor acknowledges the TAM is ~5K developers but still scores this above the midpoint. With zero user interviews, zero letters of support, and an existing ecosystem of domain-specific conservation monitors (GROMACS `gmx energy`, LAMMPS `thermo_style`), a score of 3–4/10 is more honest.

**Strongest argument:** "Adding another tool doesn't fix a culture where teams don't write conservation regression tests. A simpler intervention (a conservation-test template library for pytest) might capture 80% of the value at 1% of the cost."

### 4.2 The Auditor missed the self-constructed benchmark flaw's severity

**Challenge:** The Auditor flags Flaw 2 (unverified coverage) and Flaw 4 (repair engine oversold) but treats the self-constructed benchmark problem as a secondary concern. The Skeptic argues this is a *primary* flaw: every quantitative claim in the paper (≥90% detection, ≤10% FP, 60% repair) is circular if the benchmark is author-constructed. The Auditor should have flagged this as a CRITICAL flaw, not merely SERIOUS.

### 4.3 The Auditor's evidence for Difficulty (6/10) relies on subsystem decomposition, not end-to-end feasibility

**Challenge:** The Auditor meticulously audits each subsystem's LoC but doesn't assess the *integration risk*. The hardest part of this project isn't any individual subsystem — it's making them work together end-to-end. The IR produced by Subsystem 1 must be consumable by Subsystems 2–4, which requires perfect semantic fidelity. A single abstraction mismatch (e.g., the IR can't represent a force splitting the BCH engine needs to analyze) kills the pipeline. The Auditor's subsystem-by-subsystem approach misses this integration risk.

### 4.4 The Auditor was too generous on Best-Paper (5/10)

**Challenge:** A project with no code, no proofs, no prototype, and no demand validation does not deserve 5/10 for best-paper potential. The Auditor correctly identifies T1 as bookkeeping and T3 as a definition, then still scores best-paper at the midpoint. With only T2 as a genuine theoretical contribution, and T2 itself possibly trivial (Tarski-Seidenberg makes finite-order obstruction decidable by default), the honest score is 3–4/10.

---

## 5. AUDITOR CHALLENGES TO SKEPTIC

### 5.1 The Skeptic was unfairly harsh on demand (1/5)

**Challenge:** A 1/5 demand score implies "zero evidence of need." But the Wan et al. citation is direct evidence: a conservation bug persisted for 3 years in a major climate model, affecting published science. The Oberkampf & Roy V&V framework explicitly includes conservation checking as standard protocol — which itself validates that conservation matters enough to formalize. The Skeptic's own acknowledgment that "the problem is real" contradicts a 1/5 score. A more honest score would be 2/5: real problem, unvalidated demand for *this specific solution*.

### 5.2 The Skeptic dismissed T1 too aggressively

**Challenge:** The Skeptic says T1 is "simply labeling which terms in the BCH expansion come from which operators — this is implicit in the expansion itself." This is true for homogeneous compositions (all sub-integrators same order) but not for heterogeneous compositions with mixed orders and different step sizes. The Skeptic's own statement — "The novelty may be in the rigorous treatment of mixed-order compositions... If the proof handles this carefully, there may be a genuine technical contribution" — concedes the point but buries it under the dismissal. The Skeptic should have scored T1 conditionally rather than dismissing it outright.

### 5.3 The Skeptic's "80% of value from pytest templates" claim is unsupported

**Challenge:** The Skeptic asserts "a conservation-test template library for pytest might capture 80% of the value at 1% of the cost." This 80% figure is fabricated by the same standard the Skeptic applies to the proposal's 40–60% coverage claim. Pytest templates can check energy drift; they cannot do causal localization, obstruction detection, or provenance-tagged backward error analysis. These are qualitatively different capabilities, not "the remaining 20%."

### 5.4 The Skeptic's engineering feasibility score (2/5) conflates difficulty with infeasibility

**Challenge:** A 2/5 engineering feasibility score implies "probably can't be built." But the Skeptic's own analysis acknowledges that the restricted-ansatz symmetry analysis is tractable, BCH at order 4 is feasible, and Tree-sitter parsing is fast. The feasibility concerns are about *extraction completeness* on real codes, not about *whether the core algorithm works*. A tool that works on a restricted fragment (pure NumPy integrators) is feasible — it just has limited coverage. The Skeptic should distinguish between "can't be built" and "won't cover enough code to be useful at scale."

### 5.5 The Skeptic's counter-arguments about existing conservation monitors are overstated

**Challenge:** The Skeptic cites GROMACS `gmx energy`, LAMMPS `thermo_style`, and ESMValTool as evidence that the problem is already solved. But these are runtime monitors that detect *that* conservation is violated, not *why* or *where*. They are the equivalent of "your test failed" without a stack trace. ConservationLint's value proposition is causal localization and obstruction detection — capabilities none of these tools provide. The Skeptic correctly acknowledges this ("None of these approaches do causal localization or obstruction detection") but then doesn't adequately weight this novelty in the scoring.

---

## 6. SYNTHESIZER CHALLENGES TO BOTH

### 6.1 Both missed the conservation-aware IR as a standalone contribution

**Challenge:** The Auditor and Skeptic focus on the end-to-end pipeline viability and treat the IR as an implementation detail. The Synthesizer identifies the conservation-aware IR as "Crown Jewel #1" with standalone value. Even if the Noether/symmetry machinery never works, an IR that captures the mathematical structure a simulation discretizes is novel infrastructure. It could support method identification, refactoring equivalence checking, and backward error analysis *independent of conservation*. Neither the Auditor nor the Skeptic assess this standalone value.

### 6.2 Both undervalue the benchmark suite as a contribution

**Challenge:** The Auditor rates the benchmark suite as "mostly test fixtures" (no novelty) and the Skeptic treats it primarily as an evaluation concern. The Synthesizer correctly identifies it as potentially the most durable contribution: "the kind of artifact that gets cited for a decade." No standard benchmark for conservation correctness in simulation code exists. Even if ConservationLint fails, a carefully curated benchmark with ground-truth conservation annotations is independently valuable.

### 6.3 Scope reduction proposals are complementary, not contradictory

**Challenge:** The Skeptic's 15–25K scope and the Synthesizer's ~72K scope appear to disagree, but they address different project phases:

| Phase | Scope | Source | Purpose |
|-------|-------|--------|---------|
| Phase 1: Validate | ~20K LoC | Skeptic | Prove extraction works, prove T2 on paper, measure actual coverage on 5 real codes |
| Phase 2: Build | ~72K LoC | Synthesizer | Full static pipeline, 15+ benchmarks, complete evaluation |

The Auditor's recommendation (core pipeline on single domain) maps to Phase 1. **None of the three evaluators disagree that Phase 1 must succeed before Phase 2 begins.** The apparent scope disagreement is really an agreement on phasing.

### 6.4 The Synthesizer's amendments address the Auditor/Skeptic's core concerns

| Concern (Auditor/Skeptic) | Synthesizer Amendment |
|---|---|
| 40–60% coverage is unverified | Amendment 5: Dedicated "Coverage Honesty" section measuring liftable fragment on 5 real codebases |
| Self-constructed benchmarks | Amendment 4: Commit to 25 specific kernels; Amendment 6: Add stronger baselines |
| T2 is underspecified | Amendment 2: State T2 precisely or downgrade to validated conjecture |
| Demand is unvalidated | Amendment 1: Reframe around research tooling, not production CI/CD |
| Repair engine is weak | Amendment 3: Drop repair, invest in deeper localization evaluation |
| LLM baseline missing | Amendment 6: Add SymPy-assisted manual analysis and abstract interpretation baselines |

The Synthesizer's amendments are responsive and specific. The question is whether the Auditor and Skeptic would accept them as *sufficient* binding conditions.

### 6.5 Both Auditor and Skeptic underweight the "portfolio sibling" synergy

**Challenge:** The Synthesizer identifies that ConservationLint shares infrastructure with fp-error-audit-engine (Penumbra): Tree-sitter parsing, NumPy/SciPy semantics database, source-line attribution. A shared `sci-python-analysis-core` library could save ~10–15K LoC across both projects. Neither the Auditor nor the Skeptic consider this efficiency gain, which strengthens the feasibility case for both projects.

---

## 7. CONSENSUS SCORES (Post-Cross-Critique)

After weighing all challenges, counter-arguments, and resolutions:

### Axis 1: EXTREME AND OBVIOUS VALUE — **5/10**

The pain is real (Wan et al.) but affects a small community (~5K developers). Demand is assumed, not validated. The LLM threat is genuine for the diagnostic use case. ConservationLint's defensible value is in *formal guarantees* (obstruction proofs, provenance tracking) that LLMs cannot provide — but only on the analyzable fragment. The Synthesizer's reframing around research tooling improves the story but doesn't enlarge the market. The Skeptic's demand criticism (1/5) is overly harsh given the real anecdotal evidence, but the Auditor's 5/10 is already generous. **Consensus: 5/10.**

### Axis 2: GENUINE DIFFICULTY — **6/10**

The Auditor and Synthesizer agree on ~90–95K real LoC with ~30–35K genuinely novel. Three subproblems are genuinely hard (extraction, T2, localization). The Skeptic's concern about integration risk is valid and not captured in either the Auditor's or Synthesizer's subsystem decomposition. The "restricted Lie-symmetry analysis" is correctly identified by all three as sophisticated linear algebra, not a research breakthrough. The project is a hard senior engineer effort with 1–2 genuinely open research questions (extraction completeness, T2 proof). **Consensus: 6/10.**

### Axis 3: BEST-PAPER POTENTIAL — **5/10**

All three score this at 5/10 independently. T2 is the only genuine theorem; T1 is conditionally valuable (mixed-order case); T3 is a definition. The "killer result" (rediscovering the Wan et al. bug on real code) would be dramatic but requires end-to-end pipeline success on non-trivial code — a high bar. Realistic landing: solid OOPSLA or CAV acceptance. Best-paper requires either the real-bug demo or a surprisingly elegant T2 proof. **Consensus: 5/10.**

### Axis 4: LAPTOP CPU + NO HUMANS — **7/10**

All symbolic/algebraic; no GPU needed. The Auditor's calculation that BCH order 6 with k=10 exceeds the 10-minute budget is the binding constraint, but most practical cases (k≤5, p≤4) are well within budget. The Noether's Razor baseline is likely invalid CPU-only (all agree). IR extraction on complex codes (10K LoC, deep call graphs) may exceed 10 minutes. The 10-minute claim should be qualified as "for typical kernels under 5K LoC." **Consensus: 7/10.**

### Axis 5: FATAL FLAWS

1. **Code→math extraction on real code is unvalidated** (all three: CRITICAL). The entire pipeline depends on this, and no prototype exists.
2. **40–60% coverage is fabricated** (all three: CRITICAL). No measurement, no code, no formal definition of the liftable fragment.
3. **Self-constructed benchmarks produce circular evaluation** (all three: SERIOUS → CRITICAL when combined with #2).
4. **Cutoff-based force truncation — the most common MD conservation-bug source — is excluded by the liftable fragment** (Auditor: CRITICAL; Skeptic and Synthesizer: implicitly agree via coverage concerns).
5. **162K LoC scope is a research program, not a paper** (all three: SERIOUS).
6. **Zero validated user demand** (Skeptic: CRITICAL; Auditor/Synthesizer: MODERATE).

---

## 8. CONSENSUS VERDICT

### CONDITIONAL CONTINUE

All three evaluators independently reach CONDITIONAL CONTINUE. The core idea (Noether → program analysis bridge) is genuinely novel and worth pursuing. The execution plan requires radical correction.

### Composite Score

| Axis | Score |
|------|-------|
| Extreme and Obvious Value | 5 |
| Genuine Difficulty | 6 |
| Best-Paper Potential | 5 |
| Laptop CPU + No Humans | 7 |
| **Composite (average)** | **5.75/10** |

### Merged Binding Conditions

These conditions must ALL be met before the project advances beyond Phase 1. Conditions are tagged with their source evaluator(s).

| # | Condition | Source(s) | Priority |
|---|-----------|-----------|----------|
| **BC1** | **Radical scope reduction to two phases.** Phase 1 (~20K LoC): extraction + symmetry + BCH + localization on pure-NumPy Verlet/leapfrog integrators (<500 LoC input). Phase 2 (~70K LoC): full static pipeline, conditional on Phase 1 success. | All three | NON-NEGOTIABLE |
| **BC2** | **Honest coverage measurement.** Implement the liftable fragment checker on 5 real codebases (JAX-MD, Dedalus, a SciPy ODE suite, a gray radiation kernel, an ASE-based MD simulation). Report actual coverage with failure taxonomy. If coverage <15%, reconsider static-analysis viability. | Skeptic (C2), Synthesizer (A5) | NON-NEGOTIABLE |
| **BC3** | **External benchmarks.** Include ≥5 conservation bugs from real simulation repositories (LAMMPS issues, GROMACS changelogs, CESM bug tracker). No self-constructed-only evaluation. | Skeptic (C3), Auditor (F2) | NON-NEGOTIABLE |
| **BC4** | **Validate T2 on paper before building.** Prove the obstruction criterion for 2–3 concrete examples. If T2 is trivial (just Tarski-Seidenberg) or the criterion is exponential, revise the theoretical contribution. | Skeptic (C5), Auditor (§3 T2) | NON-NEGOTIABLE |
| **BC5** | **LLM baseline.** Include GPT-4/Claude as a baseline: paste each benchmark kernel + conservation question. Report detection rate, localization accuracy, analysis time. | Skeptic (C4), Auditor (§1 LLM) | STRONGLY REC. |
| **BC6** | **Reframe demand narrative.** Lead with numerical methods research verification, not production CI/CD. National-lab pipeline is future direction. | Synthesizer (A1) | STRONGLY REC. |
| **BC7** | **User demand validation.** Talk to 3–5 simulation developers. Show mock-up output. Document whether they would use the tool. | Skeptic (C6) | RECOMMENDED |
| **BC8** | **Drop repair synthesis; deepen localization evaluation.** Replace the repair engine with a thorough evaluation of localization accuracy at function/loop/line/expression granularity. | Synthesizer (A3), Auditor (Flaw 4) | RECOMMENDED |

---

## 9. AMENDMENTS NEEDED

### Unanimous (all three agree)

| # | Amendment | Rationale |
|---|-----------|-----------|
| **U1** | Cut 162K scope by ≥50%. Phase 1 target: ~20K LoC, single domain. | All three independently conclude the scope is a research program, not a project |
| **U2** | Drop runtime monitor from core contribution | All three: standard engineering, not novel |
| **U3** | Drop repair synthesis from core contribution | All three: template matching, not synthesis; weakest component |
| **U4** | Drop LSP/IDE integration from core contribution | All three: engineering polish, not research |
| **U5** | Measure liftable fragment coverage empirically; report honestly | All three flag 40–60% as unverified |
| **U6** | Include external benchmarks from real bug trackers | All three flag self-constructed evaluation as circular |
| **U7** | State T3 as a formal definition / scope characterization, not a "theorem" | All three: it's a definition |
| **U8** | Strengthen or precisely state T2 with all quantifiers, complexity bounds, and truncation limitations | All three: T2 is the crown jewel but underspecified |
| **U9** | Address LLM competition explicitly in the paper | All three identify this as an unaddressed threat |
| **U10** | Qualify the 10-minute budget to "typical kernels <5K LoC, k≤5, order≤4" | Auditor: order 6/k=10 exceeds budget; Skeptic: worst-case unknown; Synthesizer: acknowledges |

### Majority (two of three agree)

| # | Amendment | Supporters | Dissent |
|---|-----------|------------|---------|
| **M1** | Include LLM (GPT-4/Claude) as an evaluation baseline | Skeptic, Auditor | Synthesizer doesn't oppose but recommends SymPy-manual baseline instead; both can coexist |
| **M2** | Reframe demand around research tooling, not production CI/CD | Synthesizer, Auditor | Skeptic would go further: validate demand exists at all before proceeding |
| **M3** | Commit to a specific list of benchmark kernels (25 kernels, named) | Synthesizer, Skeptic | Auditor doesn't propose specific list but agrees benchmarks need tightening |
| **M4** | Validate T2 on paper (2–3 examples) before any tool building | Skeptic, Auditor | Synthesizer offers alternative: downgrade to "Obstruction Conjecture" with computational evidence |
| **M5** | Factor out shared infrastructure with fp-error-audit-engine (Penumbra) | Synthesizer | Auditor/Skeptic don't discuss; not opposed, just not in scope of their review |
| **M6** | Conduct user demand validation (3–5 developer interviews) | Skeptic, Synthesizer (implicitly via reframing) | Auditor recommends focusing on delivery, not market research |

### Contested (one evaluator proposes, others don't address or disagree)

| # | Amendment | Proposer | Status |
|---|-----------|----------|--------|
| **C1** | T1 (provenance-tagged BCH) is a crown jewel and should lead the paper | Synthesizer | Contested — Auditor/Skeptic consider it bookkeeping; resolution depends on mixed-order proof |
| **C2** | Add abstract interpretation baseline (Fluctuat-style) | Synthesizer | Not opposed by others; low cost to include |
| **C3** | Publish the benchmark suite as a standalone JOSS paper regardless of tool outcome | Synthesizer | Not addressed by others; hedge strategy |

---

*End of cross-critique. The above represents the stress-tested consensus of three independent evaluations. The binding conditions in §8 define the gate for continued investment.*
