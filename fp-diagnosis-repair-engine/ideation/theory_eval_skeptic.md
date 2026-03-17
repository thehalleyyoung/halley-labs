# Verification Gate: Fail-Fast Skeptic Evaluation — Penumbra (fp-diagnosis-repair-engine)

**Evaluator:** Verification Panel (3-expert team: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Independent proposals → adversarial critiques → synthesis by chair  
**Stage:** Post-theory verification — theory_bytes=0, impl_loc=0  
**Date:** 2026-03-08  
**Prior:** Depth check composite 6.25/10 (CONDITIONAL CONTINUE, 2-1, Skeptic dissenting at 4.3/10)

---

## Executive Summary

The theory stage produced **zero bytes of formal artifacts**. No proofs, no proof sketches, no lemma statements, no definitions. After two full pipeline stages (ideation + theory), the project's asset base is: comprehensive planning documents and nothing else. The Skeptic's prior dissent at the depth check (4.3/10, argued CONDITIONAL REJECT) has been substantially validated.

**Verdict: CONDITIONAL CONTINUE at 4.4/10 — FINAL ROUND (2-1, Skeptic dissents ABANDON at 3.1/10).**

The project receives a compressed 4-week kill-gate schedule. Failure at any gate triggers immediate ABANDON with no further review rounds. The full 51–87K LoC proposal is dead; the only viable path is a radically scoped minimal version (12–18K LoC).

---

## Scores

| Axis | Auditor | Skeptic | Synthesizer | **Chair** | Rationale |
|------|---------|---------|-------------|-----------|-----------|
| 1. Extreme & Obvious Value | 4 | 3 | 6 | **4/10** | Real niche gap (~100–200 users), LLM-competitive for qualitative diagnosis, "desperate need" is narrative not evidence |
| 2. Genuine Software Difficulty | 6 | 4 | 6 | **5/10** | Engineering-hard at breadth, but theory_bytes=0 eliminates claimed research difficulty. Classifiers are pattern matchers; repairs are a lookup table |
| 3. Best-Paper Potential | 4 | 2 | 5.5 | **3/10** | Zero formal results, zero code, zero bugs found. No crystalline surprise. Even perfectly executed: solid tool paper, not best-paper |
| 4. Laptop CPU + No Humans | 8 | 6 | 8.5 | **7/10** | CPU-bound by nature, 32GB sufficient with streaming, no GPUs or human annotation. Strongest axis |
| 5. Overall Feasibility | 4 | 2 | 5 | **3/10** | Zero artifacts after two stages. 51–87K LoC in 14 weeks from zero is not credible. BC4 unvalidated. T4 achievability downgraded from 85% to 35–45% |
| **Composite** | **~5.2** | **~3.1** | **~6.5** | **4.4/10** | Weighted toward Skeptic given theory stage validated Skeptic's prior concerns |

**Decision: CONDITIONAL CONTINUE — FINAL ROUND (2-1, Skeptic dissents ABANDON)**

---

## Three Pillars Assessment

### Pillar 1: Does This Deliver Extreme and Obvious Value? — NO

The problem is **real but niche**. No Python-native tool provides pipeline-level FP error diagnosis — the gap between detection (Verificarlo/Satire) and repair (Herbie) is genuine. SciPy's tracker confirms ~40 precision issues.

But "extreme and obvious value" requires *desperate need*, and the evidence is thin:
- ~40 SciPy issues represent <0.3% of the tracker. The "tip of the iceberg" claim is asserted without evidence.
- If hidden FP errors "go unnoticed," by definition they aren't causing measured pain. The proposal never demonstrates that hidden errors change scientific conclusions.
- Most scientists use float64 and never encounter problems severe enough to justify adopting a new tool with 10–50× overhead.
- In 2026, an LLM provides reasonable qualitative FP diagnosis in seconds. Penumbra's unique value — *quantitative, automated, pipeline-level tracing* — is real but serves a vanishingly small population.
- The honest audience: library developers maintaining precision-sensitive routines (~50–100 people) and extreme-precision researchers. Not "all scientific Python users."

### Pillar 2: Is This Genuinely Difficult as Software? — ENGINEERING-HARD, NOT RESEARCH-HARD

The project spans real breadth: Python runtime instrumentation, Rust↔Python interop (PyO3), multi-precision arithmetic (MPFR replay of 100+ ufuncs), streaming graph construction, pattern classifiers, algebraic rewrites, interval-arithmetic certification, LibCST source rewriting.

But the difficulty is predominantly *engineering breadth*, not *algorithmic depth*:
- The EAG builder is a streaming DAG with weighted edges from first-order finite differences — standard numerical differentiation applied to a dependency graph.
- The diagnosis engine is five threshold-based pattern matchers on graph neighborhoods — textbook logic (Higham 2002) encoded in code.
- The "30-pattern repair library" is, by the proposal's own admission, a lookup table, not synthesis.
- T4's submodularity proof was the only genuinely non-trivial mathematical contribution — and the theory stage produced zero bytes toward it.
- The mixed-precision "universal fallback" reduces to "promote to higher precision."

### Pillar 3: Does This Have Real Best-Paper Potential? — NO

Best papers need a crystalline, surprising result:
- **Herbie (PLDI'15 Distinguished):** Genuinely novel synthesis technique (equality saturation over FP).
- **Satire (ASPLOS'23):** Shadow-value analysis at unprecedented scale.
- **Penumbra:** A design document with zero formal results and zero working code.

Even perfectly executed, Penumbra is a solid tool paper — "we built X, it finds Y" — not a revelatory result. Without T4 proven, the "diagnosis-first paradigm" is a heuristic, not a provably optimal strategy. Without τ, the EAG's path decomposition may be vacuously loose. Without a showstopper bug found, the practical impact is hypothetical.

---

## Fatal & Serious Flaws

### theory_bytes = 0 — SERIOUS (near-FATAL)

The theory stage completed and produced **nothing**. Zero definitions formalized, zero proof sketches, zero lemmas attempted. The proposal claims T1 is "95% achievable" and T4 is "85% achievable" — but after an entire theory stage, not a single line of formal content exists.

**Impact:** Every theorem (T1, T3, T4, C1, τ) is currently vapor. The achievability estimates are no longer credible. T4's 85% is downgraded to 35–45%. The project must proceed as a **pure tool paper** unless T4 materializes within 2 weeks.

**The Skeptic's argument:** "A project that cannot produce a single definition during its dedicated theory stage will not produce publication-quality proofs during the implementation crunch while also writing 51K+ LoC." **The chair agrees this is compelling.**

### T4 Submodularity Fails on Non-Monotone DAGs — SERIOUS

T4 claims greedy repair ordering is step-optimal on "monotone DAGs" (all edge weights positive). The monotonicity assumption fails whenever error cancellation occurs — which is a routine phenomenon in floating-point arithmetic:
- Compensated summation works precisely because of error cancellation
- Symmetric matrix operations exhibit structured cancellation
- Backward error analysis exists because forward bounds (which assume monotone amplification) are too pessimistic

T4 applies to programs where every operation makes error worse and error never accidentally improves. These are programs where the diagnosis is trivial ("everything is bad"). On the hard case — where cancellation creates non-monotone error flow, where fixing one hotspot might worsen another — T4 says nothing.

**Impact:** The "central theorem" has a central hole. Without T4, diagnosis-guided repair is a heuristic, not a provably optimal strategy.

### The EAG ≠ PDG/SSA/e-graphs — SERIOUS

PDGs and SSA are *static, universal* program representations used by every compiler; they spawned hundreds-to-thousands of papers. The EAG is *dynamic* (trace-specific), *approximate* (first-order), and *scope-limited* (fails on ill-conditioned cases). Claiming equivalence inflates the contribution.

**Chair's ruling:** The defensible claim is: "the first Python-native tool constructing a queryable causal graph of floating-point error flow." This is weaker than "novel program representation analogous to PDGs" but still publishable. The critical question — does the EAG add value over magnitude sorting? — is **untested**.

### LAPACK Blindness on Marquee Targets — SERIOUS

The proposal's most compelling targets (`expm`, `cholesky`, `svd`, `solve`, `eigh`) dispatch to LAPACK where Penumbra is blind to internal error flow. Black-box wrapping provides aggregate error amplification but no root-cause diagnosis inside the most error-prone routines. The tool provides the *least* diagnostic granularity exactly where it would be *most* useful.

### First-Order Analysis Fails Where It Matters Most — SERIOUS

T1's soundness requires ε·n·max(Lᵢ) ≪ 1. For ill-conditioned problems (condition numbers 10⁸–10¹⁶) — the exact cases users have — this assumption fails. When it fails, the entire pipeline degrades to Satire + a pattern library. The formal analysis framework collapses on the target problems.

### 51–87K LoC in 14 Weeks Is Not Credible — SERIOUS

At the lower bound (51K LoC in 14 weeks), that's ~3,600 LoC/week or ~730 LoC/day of tested, integrated code spanning Rust, Python, and C. For reference: Herbie is ~15K LoC developed over years by a team. Satire's core is ~5K LoC. The project has produced zero artifacts in two pipeline stages.

### ≥5 Pipeline-Level Bugs May Not Exist — SERIOUS

Cross-function FP error propagation bugs that are (a) real, (b) documented, (c) not expression-level, (d) reproducible, and (e) reachable by Python-level instrumentation form a very narrow target set. Nobody has demonstrated these exist in the requisite quantity. BC4 remains entirely unvalidated.

### Fluctuat Subsumption Risk — SURVIVABLE

Fluctuat already performs error decomposition via zonotopes with *formally sound* static bounds. The honest differentiation: "Penumbra provides Fluctuat-style diagnosis for the Python ecosystem, with dynamic precision that captures execution-specific patterns at the cost of static soundness." This is a *porting + adaptation* contribution, not a paradigm shift. Framing it as a paradigm shift will get the paper rejected by any Fluctuat-aware reviewer.

### Evaluation Is Statistically Weak — SURVIVABLE

- 85% diagnosis accuracy on n=5–20 bugs → exact binomial 95% CI of [55%, 98%]. Statistically meaningless.
- Ground truth from GitHub issue discussions is informal, often wrong, and incomplete.
- "10× error reduction" target is aspirational with no evidence basis.
- Semi-synthetic benchmarks create circularity risk.

---

## Expert Votes

| Expert | Vote | Score | Key Rationale |
|--------|------|-------|---------------|
| Independent Auditor | CONDITIONAL CONTINUE (leaning ABANDON) | 5.2/10 | theory_bytes=0 is severe. LAPACK gap and T4 hole are serious. Must-pass gates in 1–2 weeks or abandon. P(top venue) = 25–35%. |
| Fail-Fast Skeptic | **ABANDON** | 3.1/10 | Three FATAL flaws (0 theory, EAG not novel, T4 fails on real programs). Four SERIOUS flaws. The project has consumed two pipeline stages and produced markdown. "This is a castle built on promises." |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 6.5/10 | Crown jewel (EAG) survives theory collapse. Minimal viable paper (EAG + diagnosis + real bugs) at 15–24K LoC is achievable. Existential gate: BC-S4 (≥3 bugs where Satire+Herbie jointly fail). |
| **Chair (consensus)** | **CONDITIONAL CONTINUE — FINAL** | **4.4/10** | Skeptic's prior dissent validated. But no claim falsified. 4-week kill-gate probe is low-cost, high-information. This is the last round. |

---

## Where the Experts Agreed

1. **theory_bytes=0 is a severe negative signal** — eliminates all claims of mathematical contribution from the project's current asset base.
2. **The problem is real but the audience is small** — genuine niche value, not desperate need.
3. **The EAG is the strongest surviving asset** — but its status as "foundational representation" is unsubstantiated without formal results.
4. **The difficulty is engineering-breadth, not research-depth** — legitimate but not algorithmically novel.
5. **BC4 (≥5 real pipeline-level bugs) is the existential unretired risk** — unvalidated after two stages.
6. **The 51–87K LoC timeline is not credible** — radical scope reduction is mandatory.

## Where They Disagreed

| Question | Skeptic | Auditor | Synthesizer | **Chair** |
|----------|---------|---------|-------------|-----------|
| Is theory_bytes=0 FATAL? | Yes — dispositive | Severe but not fatal | Significant negative update | **SERIOUS, not FATAL** — evidence of non-execution, not impossibility |
| Is the EAG novel? | No — "shadow values in a graph" | Marketing for weighted trace DAG | Yes — first reified causal error-flow graph | **Partially novel** — legitimate engineering novelty, not foundational |
| Does Fluctuat subsume Penumbra? | Largely yes | Partially | No — Python gap is real | **Differentiation is real but narrow** — porting + adaptation, not paradigm shift |
| CONTINUE or ABANDON? | ABANDON (p=0.70) | Conditional continue (barely) | Conditional continue (6.5/10) | **Final conditional continue (4.4/10)** |

**The Skeptic's dissent is formally noted and acknowledged as substantially correct in its prior predictions.** The Skeptic predicted that the theory stage would fail; it produced nothing. If KG1 fails, the Skeptic's ABANDON recommendation should have been adopted at the prior round.

---

## Scope Mandate

The full 51–87K LoC proposal is **DEAD**. The only viable path:

### Minimal Viable Paper: "Error Amplification Graphs: Tracing Causal FP Error Flow in Scientific Python"

**Keep:**
- Shadow tracer (Tier 1 only: `__array_ufunc__`)
- MPFR replay engine
- EAG builder (streaming DAG with sensitivity edges)
- 3 classifiers (cancellation, absorption, amplified rounding)
- Treewidth measurements
- T1 only (routine soundness bound)

**Cut:**
- Repair synthesizer (all 30 patterns)
- Certification engine (interval arithmetic)
- Source rewriter (LibCST)
- Tier 2 LAPACK black-box wrapping (defer to future work)
- T2 (decomposition conjecture), T4 (submodularity), T6 (composition safety)
- τ claims (report empirically if measured, do not claim bounds)
- FPBench evaluation
- Mixed-precision fallback
- Smearing and ill-conditioned classifiers

**Target LoC:** 12–18K (achievable in 10–12 weeks)  
**Target venue:** SC (software/tool track) or FSE (tool track). Not PLDI. Not OOPSLA.

---

## Kill Gates (Non-Negotiable)

Failure at any gate triggers immediate ABANDON with no further review.

| ID | Gate | Deadline | Criterion | Failure → |
|----|------|----------|-----------|-----------|
| **KG1** | Execution proof | Week 2 | Working shadow tracer for `numpy.add`, `subtract`, `multiply`, `exp`, `log` on ≥10 test inputs each, verified against `mpmath`. Minimum viable code. | **ABANDON** |
| **KG2** | Bug scouting | Week 2 | ≥3 candidate pipeline-level bugs identified from SciPy/sklearn/Astropy trackers with reproduction scripts | **PIVOT** to empirical study (≥2) or **ABANDON** (<2) |
| **KG3** | EAG viability | Week 3 | EAG constructed for ≥1 real bug with correct sensitivity edges verified against manual computation | **ABANDON** |
| **KG4** | EAG > baseline | Week 4 | EAG path-decomposition diagnosis produces different AND better repair ordering than magnitude-sorting on ≥1 reconvergent example | **PIVOT** to empirical study |

### Calendar Kill Schedule
```
Week 2:  KG1 fails → ABANDON immediately
         KG2 <2 bugs → ABANDON; 2 bugs → PIVOT to empirical study
Week 3:  KG3 fails → ABANDON
Week 4:  KG4 fails → PIVOT to empirical study (EAG as visualization only)
Week 6:  Diagnosis accuracy <60% on real bugs → PIVOT to empirical study
Week 8:  No end-to-end demo on ≥3 bugs → ABANDON
```

---

## Binding Conditions (Carried Forward + New)

| ID | Condition | Gate | Status |
|----|-----------|------|--------|
| BC1 | T2 demoted from central contribution; EAG + Diagnosis centered | Structural | ✅ Applied in final_approach.md |
| BC2 | LAPACK black-box strategy designed; FEniCS/Firedrake dropped | Structural | ✅ Applied in final_approach.md |
| BC3 | Fluctuat comparison corrected to "complementary" | Structural | ✅ Applied in final_approach.md |
| BC4 | ≥5 real pipeline-level bugs identified | Week 4 kill gate | ❌ Unvalidated — downgraded to ≥3 (KG2) |
| BC5 | Instrumentation coverage metric defined and measured | Before evaluation | ❌ Not started |
| BC6 | Ground-truth methodology with sample sizes and CIs | Before evaluation | ❌ Not started |
| **BC7** | **Scope restricted to minimal paper (no repair, no certification)** | **Before implementation** | **NEW — mandatory** |
| **BC8** | **T4 attempted but not blocking (proceed as tool paper if fails)** | **Week 4** | **NEW — T4 is optional** |
| **BC9** | **Do NOT compare EAG to PDG/SSA/e-graphs in paper** | **Before writeup** | **NEW — the analogy is unsupported** |
| **BC10** | **Report τ as empirical data only (no claimed bounds)** | **Before writeup** | **NEW — if measured at all** |

---

## Probability Estimates

| Outcome | Prior (Depth Check) | **Chair (This Round)** | Change |
|---------|--------------------|-----------------------|--------|
| P(top venue: SC/FSE/OOPSLA/ASPLOS) | 45–55% | **15–25%** | ↓↓ theory_bytes=0 |
| P(best-paper at any venue) | 5–10% | **2–4%** | ↓ no surprising result |
| P(any publication incl. workshops) | 65–75% | **35–45%** | ↓ execution undemonstrated |
| P(project abandoned at kill gates) | 15–25% | **35–50%** | ↑↑ theory_bytes=0, BC4 unvalidated |
| P(≥5 real pipeline-level bugs found) | ~60% | **25–40%** | ↓ never scouted |
| P(T4 proof completed) | ~85% | **35–45%** | ↓↓ theory stage produced nothing |
| P(merged SciPy PR) | ~30% | **8–15%** | ↓ naive timeline for PR review |

---

## What Would Change Everything

1. **KG1 passes (working shadow tracer in 2 weeks).** Demonstrates execution capability. P(any pub) jumps from ~35% to ~55%. The theory_bytes=0 signal is reinterpreted as a process failure, not a substance failure.

2. **Find a real, previously-unknown SciPy bug.** A single concrete result — "Penumbra discovered bug X in `scipy.Y`, silent since 20XX" — raises Value to 6, Best-Paper to 5. This is the single highest-leverage activity.

3. **T4 proof materializes in 2 weeks.** Changes the paper from "tool" to "tool + proportionate math." Raises Difficulty to 6, Best-Paper to 5. But probability is only 35–45%.

4. **τ > 0.1 on ≥2 real programs.** Validates EAG's quantitative path decomposition. Raises the EAG from "convenient data structure" to "representation with predictive power." But probability is unknown.

---

## Salvage Options (If ABANDON)

| Option | Venue | LoC | Risk | Value |
|--------|-------|-----|------|-------|
| Minimal prototype + 2–3 case studies | CORRECTNESS workshop @ SC | 3–5K | Low | Establishes EAG concept, gets community feedback |
| Empirical study of FP error flow | ICSE empirical track / MSR | 8–12K | Low | Novel data on treewidth, error patterns |
| Bug corpus documentation | Experience report | 0–2K | Very low | Community contribution if bugs exist |
| τ characterization (if provable) | PLDI/POPL standalone | Theory only | High | Genuine theoretical contribution about first-order tightness |

---

## The Honest Assessment

This is a project with a **6/10 idea and 0/10 execution**. The idea quality alone is not sufficient to justify continued investment — ideas are cheap, execution is expensive. The 4-week probe is a limited-downside bet that the execution problem is a process failure (fixable) rather than a substance failure (terminal).

The Skeptic's closing argument deserves repetition: *"This proposal has consumed two pipeline stages and produced excellent markdown. The theory doesn't exist, the engineering hasn't started, the novelty is thin once you strip the marketing, and the audience is small. I recommend ABANDON."*

The chair's response: *"The Skeptic is probably right. But 'probably' is not 'certainly.' Four weeks to find out costs little. If KG1 fails, we should have listened to the Skeptic at the depth check."*

**This is the last CONDITIONAL CONTINUE this project will receive.**

---

*Signed: Verification Panel Chair (3-expert consensus with 1 dissent). theory_bytes=0, impl_loc=0. The project lives or dies on the kill gates.*
