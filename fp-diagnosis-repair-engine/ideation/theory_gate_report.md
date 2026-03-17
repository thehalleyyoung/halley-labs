# Verification Gate Report: Penumbra (fp-diagnosis-repair-engine)

**Evaluator:** Verification Panel Chair (5-expert consensus: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer + 2 prior evaluations)  
**Method:** Independent proposals → adversarial cross-critiques → chair synthesis  
**Stage:** Post-theory verification — theory_bytes=0, impl_loc=0  
**Date:** 2026-03-08  
**Prior:** Depth check composite 6.25/10 (CONDITIONAL CONTINUE, 2-1, Skeptic dissent at 4.3)

---

## Executive Summary

The theory stage produced **zero bytes of formal artifacts** (State.json records theory_bytes: 0 with ~3-second runtime, indicating a process crash). No implementation code exists. After two full pipeline stages (ideation + theory), the project's entire asset base is ~200KB of planning documents and evaluations — zero bytes of code, proofs, or data.

Five independent evaluations converge on the same picture: a 6/10 idea with 0/10 execution.

**Verdict: CONDITIONAL CONTINUE — FINAL ROUND (4.4/10). 2-1 panel vote, Skeptic dissents ABANDON at 3.3/10.**

The full 51–87K LoC proposal is **DEAD**. The only viable path is a radically scoped MVP (12–20K LoC) reframed as "the first causal debugger for floating-point error in Python." Compressed 4-week kill-gate schedule: failure at any gate triggers immediate ABANDON with no further review rounds.

---

## Scores

| Axis | Auditor | Skeptic | Synthesizer | Prior Skeptic Eval | Prior Math Eval | **Chair** |
|------|---------|---------|-------------|-------------------|-----------------|-----------|
| 1. Extreme & Obvious Value | 4 | 3 | 5 | 4 | 4.5 | **4/10** |
| 2. Genuine Software Difficulty | 5 | 5 | 6 | 5 | 5.5 | **5/10** |
| 3. Best-Paper Potential | 3 | 2 | 4 | 3 | 3.5 | **3/10** |
| 4. Laptop CPU + No Humans | — | — | — | 7 | 7.5 | **7/10** |
| 5. Overall Feasibility | 3 | 2 | 5 | 3 | 4.5 | **3/10** |
| **Composite** | **4.0** | **3.3** | **5.0** | **4.4** | **5.1** | **4.4/10** |

**Decision: CONDITIONAL CONTINUE — FINAL ROUND (2-1, Skeptic dissents ABANDON)**

---

## Three Pillars Assessment

### Pillar 1: Does This Deliver Extreme and Obvious Value? — NO (4/10)

The problem is **real but niche**. No Python-native tool provides pipeline-level FP error diagnosis — the gap between detection (Verificarlo/Satire) and repair (Herbie) is genuine. SciPy's tracker confirms ~40 precision issues.

But "extreme and obvious value" requires *desperate need*, and the evidence is thin:

- ~40 SciPy issues represent <0.3% of the tracker. The "tip of the iceberg" claim is asserted without evidence.
- If hidden FP errors "go unnoticed," they aren't causing measured pain. The proposal never demonstrates that hidden errors change scientific conclusions.
- **The LLM ceiling:** In 2026, an LLM provides reasonable qualitative FP diagnosis (~70-80% of what a practitioner needs) in seconds, for free. Penumbra's unique value is *quantitative, automated, pipeline-level tracing* — genuine but narrow, serving the ~30-40% of cases where qualitative diagnosis fails.
- **Audience reality check:** Realistic user base is 80–200 people — primarily SciPy/scikit-learn library maintainers and extreme-precision researchers. Not "all scientific Python users."

### Pillar 2: Is This Genuinely Difficult as Software? — ENGINEERING-HARD, NOT RESEARCH-HARD (5/10)

The project spans real breadth: Python runtime instrumentation, Rust↔Python interop (PyO3), multi-precision arithmetic (MPFR replay), streaming graph construction, pattern classifiers, and interval-arithmetic certification. This is legitimately hard engineering.

But the difficulty is predominantly *engineering breadth*, not *algorithmic depth*:

- The EAG builder is a streaming DAG with weighted edges from first-order finite differences — standard numerical methods applied to a dependency graph.
- The diagnosis engine is threshold-based pattern matchers on graph neighborhoods — textbook logic (Higham 2002) encoded in code.
- The "30-pattern repair library" is a lookup table, not synthesis.
- T4's submodularity proof was the only non-trivial mathematical claim — and theory_bytes=0 means it was never attempted.
- The mixed-precision "universal fallback" reduces to "promote to higher precision."

### Pillar 3: Does This Have Real Best-Paper Potential? — NO (3/10)

Best papers need a crystalline, surprising result:

- **Herbie (PLDI'15 Distinguished):** Novel synthesis technique (equality saturation over FP). Surprising effectiveness.
- **Satire (ASPLOS'23 Distinguished):** Shadow-value analysis at unprecedented scale. Surprising scalability.
- **Penumbra (current state):** 200KB of planning documents. Zero executable evidence of anything.

Even perfectly executed, Penumbra is a solid tool paper — "we built X, it finds Y" — not a revelatory result. Without T4 proven, the "diagnosis-first paradigm" is a heuristic. Without τ measured, the EAG's quantitative claims are hollow. Without a showstopper bug found, the practical impact is hypothetical.

---

## Critical Findings

### Finding 1: theory_bytes = 0 — SERIOUS (near-FATAL)

The theory stage completed in ~3 seconds and produced nothing. This is consistent with a process crash, not a deliberate failure. But the outcome is identical: **zero definitions formalized, zero proof sketches, zero lemmas stated.**

**Chair's ruling:** SERIOUS, not FATAL. The 3-second runtime contains no information about whether theorems are provable. But the meta-signal — nobody noticed or retried — suggests theory was deprioritized. All theorems (T1, T3, T4, C1, τ) must be treated as unproven. The project proceeds as a **pure tool paper** unless T4 materializes within 4 weeks.

### Finding 2: Zero Artifacts After Two Stages — SERIOUS

The project has consumed two full pipeline stages (ideation + theory) and produced ~200KB of markdown planning documents. Zero bytes of code, proofs, or empirical data. The claim audit found **2 of 10 major claims supported by any evidence** (Fluctuat comparison framing; T2 demotion).

**Risk-adjusted P(success) ≈ 13%** (Auditor's calculation assuming independent risks). Even accounting for portfolio salvage paths, **P(any meaningful publication) ≈ 28-38%**.

### Finding 3: BC4 (Pipeline-Level Bugs) Remains Unvalidated — POTENTIALLY FATAL

Cross-function FP error propagation bugs that are simultaneously (a) real, (b) documented, (c) not expression-level, (d) reproducible, (e) reachable by Python instrumentation, and (f) not inside LAPACK form a **near-empty target set** until demonstrated otherwise. Nobody — including the proposal authors — has shown these exist in sufficient quantity. This is the existential gate.

### Finding 4: EAG Novelty Is Genuine But Modest

The Skeptic challenged: "Name one algorithm the EAG enables that you can't do with Satire's shadow values." **Answer: causal path decomposition** — attributing output error to specific propagation paths with quantitative weights. Satire gives per-operation magnitudes but cannot trace causal chains.

**However:** Fluctuat's zonotope decomposition provides similar (and formally sounder) per-contribution attribution for C. The EAG's novelty is relative to the Python ecosystem and dynamic analysis, not absolute. The PDG/SSA/e-graph comparison is **indefensible and must be dropped** (BC9).

### Finding 5: T4 Submodularity Is Either Trivial or False

The cross-critique confirmed:
- **Repair = eliminate all error at a node:** T4 is a truism (fix stuff in any order).
- **Repair = reduce by a fixed fraction:** T4 is provable but says "fix worst first" — what practitioners already do.
- **Repair = apply specific algebraic rewrite:** Submodularity is almost certainly false in general.

T4 adds formal teeth to the diagnosis-first paradigm only in a narrow middle ground. It is now **optional** (BC8).

### Finding 6: Scope Must Be Radically Cut

The full 51–87K LoC proposal is dead. At the lower bound (51K LoC in 14 weeks), that's ~3,600 LoC/week from zero, spanning Rust, Python, and C. For reference: Herbie is ~15K LoC developed over years.

**Mandatory scope: Penumbra-Lite MVP (12–20K LoC)**

| Keep | Cut |
|------|-----|
| Shadow tracer (Tier 1 only) | Repair synthesizer (all 30 patterns) |
| MPFR replay (20 most common ufuncs) | Certification engine |
| EAG builder (streaming DAG + sensitivity edges) | Source rewriter (LibCST) |
| 3 classifiers (cancellation, absorption, amplified rounding) | Tier 2 LAPACK wrapping |
| Treewidth measurement | T2, T4, T6, τ theorems |
| Minimal CLI + tests | FPBench evaluation, mixed-precision fallback |

---

## Expert Votes

| Expert | Vote | Score | Key Rationale |
|--------|------|-------|---------------|
| Independent Auditor | CONDITIONAL CONTINUE (barely) | 4.0/10 | 2/10 claims supported. P(success)≈13%. Kill gates cheaply testable in 2 weeks. |
| Fail-Fast Skeptic | **ABANDON** | 3.3/10 | Two stages, zero artifacts. LLM handles 70-80% of qualitative diagnosis. 50-200 users. "Nothing exists." |
| Scavenging Synthesizer | CONDITIONAL CONTINUE | 5.0/10 | MVP at 12-20K LoC is achievable. Hidden gems (EAD, treewidth, coverage metric). Three salvage paths. |
| Prior Skeptic Eval | CONDITIONAL CONTINUE — FINAL | 4.4/10 | theory_bytes=0 is severe. Kill gates mandatory. "This is the last CONDITIONAL CONTINUE." |
| Prior Mathematician Eval | CONDITIONAL CONTINUE (Weak) | 5.1/10 | Math is supportive, not load-bearing. Tool paper at best without T4. |
| **Chair (consensus)** | **CONDITIONAL CONTINUE — FINAL** | **4.4/10** | Skeptic's prior validated. But no claim falsified. 2-week probe is low-cost. |

---

## Where Experts Agreed

1. **theory_bytes=0 is a severe negative signal** — eliminates all claims of mathematical contribution from current asset base.
2. **The problem is real but the audience is small** — genuine niche value, not desperate need.
3. **The EAG is the strongest surviving asset** — but its "foundational representation" status is unsubstantiated without working code.
4. **The difficulty is engineering-breadth, not research-depth** — legitimate but not algorithmically novel.
5. **BC4 (≥3 pipeline-level bugs) is the existential unretired risk** — unvalidated after two stages.
6. **The 51–87K LoC timeline is not credible** — radical scope reduction to 12–20K LoC is mandatory.
7. **The PDG/SSA/e-graph comparison must be dropped** — the analogy is unsupported.

## Where Experts Disagreed

| Question | Skeptic | Auditor | Synthesizer | **Chair Ruling** |
|----------|---------|---------|-------------|------------------|
| Is theory_bytes=0 FATAL? | Yes — dispositive | Severe but not fatal | Significant negative | **SERIOUS, not FATAL** — process crash, not substance failure |
| Is the EAG novel? | No — "shadow values in a graph" | Genuine but modest | First causal error-flow graph | **Partially novel** — causal path decomposition is genuinely new for Python |
| CONTINUE or ABANDON? | ABANDON (P=0.70) | Conditional continue (barely) | Conditional continue | **CONDITIONAL CONTINUE — FINAL** |
| Audience size? | 50-200 | 100-200 | 200-500 | **80-200** — Skeptic is closest |
| Salvage path viability? | P(pub)≈15% | P(pub)≈30-40% | P(pub)≈65-75% | **P(pub)≈28-38%** — Skeptic closest |
| LLM ceiling threat? | Severe (70-80% replacement) | Real but quantified | Irrelevant to quantitative tracing | **Auditor's balanced view** — real threat that narrows the moat |
| Hidden gems? | Dressed-up engineering | Not assessed | Real sub-contributions | **Two of three are real but minor** (EAD, treewidth) |

**The Skeptic's dissent is formally noted. The Skeptic's prior predictions were validated (theory stage produced nothing). If KG1 fails, the Skeptic's ABANDON recommendation should have been adopted at the depth check.**

---

## Scope Mandate

### Minimal Viable Paper: "Penumbra: Causal Debugging of Floating-Point Error via Error Amplification Graphs"

**Reframing (from Synthesizer, endorsed by Chair):** From "diagnosis-repair pipeline" to "causal debugger." Debuggers don't need repair synthesis (GDB doesn't fix bugs), don't need coverage guarantees, and "first causal FP debugger for Python" is crisp and literally true.

**Target LoC:** 12–20K (achievable in 10–12 weeks)  
**Target venue:** SC software/tool track (primary), FSE tool track (secondary), CORRECTNESS workshop @ SC (safety net).  
**Not:** PLDI, OOPSLA, ASPLOS — insufficient formal results.

---

## Kill Gates (Non-Negotiable)

Failure at any gate triggers the specified action with no appeal.

| Week | Gate | Criterion | Failure → |
|------|------|-----------|-----------|
| **2** | **KG1: Execution Proof** | Working shadow tracer for `numpy.add`, `subtract`, `multiply`, `exp`, `log` on ≥10 test inputs each, verified against `mpmath` | **ABANDON** |
| **2** | **KG2: Bug Scouting** | ≥3 candidate pipeline-level bugs from SciPy/sklearn/Astropy with reproduction scripts | **<2 → ABANDON; =2 → PIVOT to empirical study** |
| **3** | **KG3: EAG Viability** | EAG constructed for ≥1 real bug with correct sensitivity edges verified against manual computation | **ABANDON** |
| **4** | **KG4: EAG > Baseline** | EAG path-decomposition produces different AND better ordering than magnitude-sorting on ≥1 example | **PIVOT to empirical study** |
| **4** | **KG5: τ Measurement** | τ > 0.01 on ≥1 real program | **Downgrade EAG to visualization-only** |
| **6** | **KG6: Diagnosis** | ≥60% diagnosis accuracy on identified real bugs | **PIVOT to empirical study** |
| **8** | **KG7: End-to-End** | Complete trace→EAG→diagnosis demo on ≥3 real bugs | **ABANDON** |
| **8** | **KG8: T4 Decision** | T4 proved OR formally abandoned → pure tool paper | **Commit to tool paper** |

### Calendar Kill Schedule

```
Week 2:  KG1 fails → ABANDON immediately
         KG2 <2 bugs → ABANDON; =2 → PIVOT to empirical study
Week 3:  KG3 fails → ABANDON
Week 4:  KG4 fails → PIVOT to empirical study (EAG as visualization only)
         KG5 τ<0.01 → downgrade quantitative claims
Week 6:  KG6 <60% → PIVOT to empirical study
Week 8:  KG7 fails → ABANDON
         KG8 T4 fails → commit to pure tool paper
```

---

## Binding Conditions

| ID | Condition | Status | Gate |
|----|-----------|--------|------|
| BC1 | T2 demoted from central contribution | ✅ Applied | Structural |
| BC2 | LAPACK black-box strategy designed | ✅ Applied | Structural |
| BC3 | Fluctuat comparison corrected to "complementary" | ✅ Applied | Structural |
| BC4 | ≥3 real pipeline-level bugs (downgraded from 5) | ❌ Unvalidated | KG2 Week 2 |
| BC5 | Instrumentation coverage metric defined and measured | ❌ Not started | Before evaluation |
| BC6 | Ground-truth methodology with sample sizes and CIs | ❌ Not started | Before evaluation |
| **BC7** | **Scope restricted to MVP (no repair, no certification)** | **NEW — mandatory** | Before implementation |
| **BC8** | **T4 optional — proceed as tool paper if fails** | **NEW** | KG8 Week 8 |
| **BC9** | **Do NOT compare EAG to PDG/SSA/e-graphs** | **NEW — mandatory** | Before writeup |
| **BC10** | **Report τ as empirical data only (no bounds claims)** | **NEW** | Before writeup |
| **BC11** | **Repair "synthesizer" reframed as "repair selector/prescriber"** | **NEW** | Before writeup |
| **BC12** | **LLM baseline comparison required in evaluation** | **NEW** | Before submission |

---

## Probability Estimates

| Outcome | Depth Check | Prior Skeptic Eval | Prior Math Eval | **Chair (This Round)** |
|---------|-------------|-------------------|-----------------|------------------------|
| P(top venue: SC/FSE) | 45-55% | 15-25% | 18-25% | **15-20%** |
| P(best-paper) | 5-10% | 2-4% | 2-4% | **2-4%** |
| P(any publication) | 65-75% | 35-45% | 38-48% | **28-38%** |
| P(abandoned at kill gates) | 15-25% | 35-50% | 35-45% | **45-55%** |
| P(≥3 pipeline-level bugs) | ~60% | 25-40% | 50-65% | **35-50%** |
| P(T4 proved, useful version) | ~85% | 35-45% | 45-55% | **30-45%** |
| P(merged upstream PR) | ~30% | 8-15% | 12-20% | **8-15%** |

---

## What Would Change Everything

1. **Find a real, previously-unknown SciPy bug and get a PR merged.** This single result transforms Value to 6-7, Best-Paper to 5-6. The strongest single leverage point. P ≈ 8-15%.

2. **KG1 passes (working shadow tracer in 2 weeks).** Demonstrates execution capability. P(any pub) jumps from ~33% to ~55%. Reinterprets theory_bytes=0 as process failure, not substance failure.

3. **τ > 0.1 on ≥3 real programs.** Validates EAG's quantitative causal attribution. Transforms EAG from "convenient data structure" to "representation with predictive power."

4. **T4 proof materializes (fraction-reduction model).** Elevates paper from "tool" to "tool + proportionate math." But lowest-leverage improvement — practitioners don't need (1-1/e) guarantees for k ≤ 10 repairs.

---

## Salvage Options (If ABANDON)

| Option | Venue | LoC | Risk | P(accept) |
|--------|-------|-----|------|-----------|
| Empirical study of FP error flow structure | MSR / ICSE-SEIP | 5-8K | Low | 50-65% |
| Instrumentation coverage survey | PyHPC @ SC / SciPy conf | 1-3K | Very low | 60-75% |
| Error sensitivity analysis tool | ISSTA / ASE tool track | 4-7K | Medium | 40-55% |
| Bug corpus documentation | Experience report | 0-2K | Very low | 30-50% |

---

## The Honest Assessment

This is a project with a **sound idea and zero execution**. The idea quality alone is not sufficient to justify continued investment — ideas are cheap, execution is expensive. The 2-week probe is a limited-downside bet that the execution problem is a process failure (fixable) rather than a substance failure (terminal).

The Skeptic's closing argument: *"This proposal has consumed two pipeline stages and produced excellent markdown. The theory doesn't exist, the engineering hasn't started, the novelty is thin once you strip the marketing, and the audience is small. ABANDON."*

The chair's response: *"The Skeptic is probably right. But 'probably' is not 'certainly.' Two weeks to find out costs little. If KG1 fails, we should have listened to the Skeptic at the depth check."*

**This is the last CONDITIONAL CONTINUE this project will receive.**

---

*Signed: Verification Panel Chair (5-expert consensus with 1 dissent). theory_bytes=0, impl_loc=0. Composite 4.4/10. CONDITIONAL CONTINUE — FINAL. The project lives or dies on the kill gates.*
