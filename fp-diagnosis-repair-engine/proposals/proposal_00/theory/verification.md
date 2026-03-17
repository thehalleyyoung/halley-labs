# Post-Theory Verification: Scavenging Synthesizer Evaluation

**Proposal:** Penumbra — Diagnosis-Guided Repair of Floating-Point Error in Scientific Pipelines via Error Amplification Graphs  
**Evaluator:** Scavenging Synthesizer (Expert 3/3)  
**Stage:** Post-theory verification  
**Signal:** `theory_bytes=0`, `impl_loc=0`, `monograph_bytes=0`  
**Prior round:** Composite 6.25/10, CONDITIONAL CONTINUE (2-1), 8 amendments applied  
**Date:** 2026-03-08

---

## 0. Interpreting theory_bytes=0

Before scoring, I must interpret the signal honestly. `theory_bytes=0` means the theory directory is empty and the theory log is blank — not a single line of proof attempt was recorded. Three interpretations:

1. **Process failure:** The theory stage ran but produced no output (timeout, crash, empty generation). The *ideas* may be fine; the *pipeline* failed.
2. **Substance failure:** Proof attempts were made mentally/informally and abandoned because the claims don't hold.
3. **Deprioritization:** The final_approach.md already acknowledged T2 as demoted, τ as risky, and T1/C1 as "routine." The theory stage may have been starved of effort because the proposal is fundamentally an engineering/empirical contribution.

**My read:** Interpretation (3) with elements of (1). The final_approach.md is remarkably self-aware — it already demoted T2, declared T3 "formalized bookkeeping," placed T4 at 85% achievability, and said "the difficulty is primarily engineering-breadth, not algorithmic-depth." The proposal *told us* it's not primarily a theory contribution. The theory_bytes=0 signal confirms that the theory wasn't attempted, but the proposal was already pivoting toward a tool paper with proportionate math. This is less damning than it appears — **but it still means zero formal artifacts exist, and any theory claims are currently vapor.**

---

## 1. Scoring

### As-Is Scores (theory_bytes=0, impl_loc=0)

| Axis | As-Is | Best-Case w/ Reframing | Notes |
|------|-------|------------------------|-------|
| **1. Extreme & Obvious Value** | **5.5/10** | **7/10** | Real problem, real gap. Minimal viable version has value. |
| **2. Genuine Software Difficulty** | **6.5/10** | **7.5/10** | Engineering-hard survives; algorithm-hard claims are vapor. |
| **3. Best-Paper Potential** | **3.5/10** | **6.5/10** | Without proofs or implementation, potential is theoretical. With showstopper result, jumps. |
| **4. Laptop-CPU Feasibility** | **7.5/10** | **8/10** | Fundamentally laptop-feasible. No GPU needed. |
| **5. Feasibility** | **5/10** | **7/10** | 14-week timeline with zero artifacts is tight. Radical scope cut makes it achievable. |
| **Composite** | **5.6/10** | **7.2/10** | — |

### Axis-by-Axis Analysis

#### Axis 1: Extreme & Obvious Value — 5.5 (as-is) / 7 (best-case)

**What survives under pessimistic assumptions:**

Even a *broken* Penumbra that only does shadow-value tracing + EAG visualization is more than what Python scientists have today. The minimal viable product is:

> `pip install penumbra && penumbra trace my_pipeline.py` → colored DAG showing where error is large and which operations amplify it.

No existing tool provides this. Verificarlo requires LLVM recompilation. Herbie is expression-level. Satire targets C/Fortran. The "Verificarlo for Python" gap is real and unfilled.

**The practitioner floor:** A library developer debugging `scipy.special.expm` precision can today: (a) read Higham, (b) sprinkle `mpmath` calls manually, (c) ask an LLM. Option (c) is free but non-quantitative. Penumbra at its *worst* (no repair, no proofs, just tracing + visualization) gives quantitative per-operation error and causal flow — something no LLM provides. This floor has genuine, if niche, value.

**What would raise to 7:** Finding 1 real previously-unknown SciPy bug. Just one. That transforms the tool from "we claim this is useful" to "this found something humans missed."

#### Axis 2: Genuine Software Difficulty — 6.5 (as-is) / 7.5 (best-case)

**What's genuinely hard, independent of theory:**

1. **MPFR replay fidelity for 100+ ufuncs** — Each ufunc has edge cases (denormals, inf, NaN, reduction order). Faithful reproduction is a vast correctness surface. This is *harder* than "Verificarlo for Python" because Verificarlo instruments at LLVM IR level (one interception point); Penumbra must handle Python's dispatch zoo (`__array_ufunc__`, `__array_function__`, monkey-patches).
2. **Streaming EAG construction under memory constraints** — Dense matrix ops produce O(n³) potential edges. The aggregation/sparsification strategy is a real systems design problem.
3. **Two-tier architecture** — Combining element-level tracing with black-box LAPACK wrapping in a coherent framework is a novel engineering contribution. No one has done this.

**What's less hard than claimed:** The diagnosis classifiers are threshold-based pattern matchers on well-understood phenomena (Higham 2002). The repair patterns are a lookup table. T4 is a clean application of known results (Nemhauser-Wolsey-Fisher).

**Docked 0.5 from prior round** because theory_bytes=0 removes the "proportionate math" that distinguished this from pure engineering.

#### Axis 3: Best-Paper Potential — 3.5 (as-is) / 6.5 (best-case)

**As-is:** With zero proofs and zero implementation, best-paper potential is theoretical. You cannot win best paper with a proposal document. This is the axis most damaged by theory_bytes=0.

**Best-case with reframing:** The gap between 3.5 and 6.5 is entirely determined by execution. The ingredients for a 6.5 exist:
- T4 proof (Nemhauser-Wolsey-Fisher application — 85% achievable per the proposal's own estimate)
- ≥1 merged SciPy PR (transforms the narrative)
- EAG reframing as novel program representation
- Treewidth empirical data (genuinely novel measurements)

**The showstopper result** that creates best-paper potential: *"Penumbra discovered a silent 10⁸-ulp error in scipy.linalg.expm for matrices with eigenvalue spread >10⁶, affecting ≥3 downstream packages. The fix (a Schur-decomposition pre-conditioner suggested by diagnosis) was merged as scipy/scipy#XXXXX."* This single result would raise Axis 3 to 7+.

#### Axis 4: Laptop-CPU Feasibility — 7.5 (as-is) / 8 (best-case)

Unchanged from prior round. MPFR is inherently sequential. EAGs are sparse graphs. No GPU benefit. 32GB handles streaming construction. 8-24 hour evaluation on curated targets. The minimal publishable evaluation (FPBench + 3 real bugs + 2 fault-injection) runs in under 12 hours.

#### Axis 5: Feasibility — 5 (as-is) / 7 (best-case)

**As-is at 5:** Zero artifacts exist. The 14-week timeline in the final_approach.md is aggressive even *with* artifacts. Starting from absolute zero with 14 weeks and needing both a working tool AND an evaluation is tight.

**Best-case at 7:** Radical scope reduction makes this feasible:
- Cut repair from v1 (EAG + Diagnosis only)
- Target 3 bugs instead of 5
- Use FPBench as primary benchmark, real bugs as case studies
- Write T4 proof only (drop T1 formal proof, keep as informal argument)
- 10 weeks engineering, 2 weeks evaluation, 2 weeks writing

---

## 2. Salvage Analysis Per Component

### If T4 proof fails

**What remains:** The greedy diagnosis-guided repair ordering *still works as a heuristic*. Every prior FP repair tool (Herbie, Precimonious, FPTuner) operates without optimality guarantees. Penumbra without T4 is still the first tool that connects diagnosis to repair ordering — the heuristic just lacks a proof.

**Publishability:** YES, as a tool paper at SC/FSE. Empirical demonstration that greedy-by-diagnosis outperforms random ordering on ≥5 benchmarks is sufficient. Report the ratio (greedy error reduction) / (exhaustive-search error reduction) as an empirical optimality gap. If the gap is <10% across all benchmarks, the practical conclusion is the same as the theorem.

**Reframing:** "We conjecture diagnosis-guided repair is submodular-optimal on monotone DAGs (proof in progress); empirically, it achieves within X% of exhaustive search on all N benchmarks."

### If BC4 fails (can't find ≥5 real bugs)

**What evaluation story survives:**

| Real bugs found | Evaluation strategy | Publishable? |
|----------------|---------------------|-------------|
| 0 | FPBench + fault injection only | Weak — workshop paper at best |
| 1-2 | Real bugs as case studies + FPBench + fault injection | YES at SC/FSE as tool paper |
| 3-4 | Real bugs + FPBench, reframe "≥5" threshold | YES, strong tool paper |

**The critical floor is 2 real bugs.** Even with just 2 real bugs where Penumbra outperforms Herbie+Verificarlo, combined with FPBench and fault injection, the evaluation tells a coherent story. Below 2, the "pipeline-level" claim becomes aspirational.

**Fallback strategy:** If scouting (weeks 1-2) finds <3 candidates, immediately pivot to:
1. FPBench as the primary evaluation (30+ expressions, head-to-head with Herbie)
2. Fault injection as the pipeline-level substitute (inject known errors at function boundaries, measure diagnosis accuracy)
3. Real bugs as bonus case studies, not the headline

### If theory_bytes=0 persists (no proofs achieved)

**What kind of paper is this?** A **pure tool/systems paper** — and that's a legitimate publication category at SC, FSE, and ISSTA. Precedents:

- **Satire (ASPLOS'23):** Shadow-value analysis. No deep theorems. Distinguished paper.
- **Verificarlo:** Stochastic arithmetic instrumentation. Primarily systems contribution.
- **AddressSanitizer (USENIX ATC'12):** Memory error detection. Pure engineering. Highly cited.

**The paper becomes:** *"We built the first Python-native pipeline-level FP diagnosis tool. It constructs EAGs, classifies root causes, and suggests repairs. On N real bugs and M benchmarks, it achieves X× error reduction. Here is a merged SciPy PR."*

**What's lost:** T4 optimality guarantee (becomes a heuristic claim). Formal soundness of T1 (becomes an empirical observation). The "proportionate math" that the final_approach.md carefully positioned. The paper is weaker but still publishable.

**What's NOT lost:** The EAG as a novel representation, the diagnosis taxonomy, the empirical results, the treewidth data. These are all implementation/empirical contributions that don't require proofs.

### If the EAG adds no value over Satire + graph database

**This is the most dangerous scenario.** If the sensitivity edges (∂ε̂ⱼ/∂ε̂ᵢ) provide no better diagnosis than simply sorting operations by shadow-value error magnitude, the EAG is an over-engineered visualization.

**Test:** On each benchmark, compare:
- (A) EAG path-decomposition diagnosis → repair ordering
- (B) Sort-by-magnitude diagnosis → same repair ordering

If (A) and (B) produce identical repair sequences on >80% of benchmarks, the EAG edges add no value.

**Fallback if EAG ≈ magnitude sorting:**
1. **Reframe the EAG as an engineering convenience, not a conceptual advance.** "The graph structure enables efficient incremental re-analysis after repair, avoiding full re-execution." This is a genuine systems benefit even if the diagnosis doesn't improve.
2. **Pivot to the diagnosis taxonomy as the primary contribution.** "The first automated root-cause classification of FP errors" is novel even without the EAG — but it's weaker.
3. **Honest reporting:** Include the ablation. If magnitude-sorting matches EAG in 80% of cases, report it. The 20% where EAG wins (reconvergent error paths, multi-hop amplification) may be the most interesting cases.

### If repair library is dismissed as "not synthesis"

**Reframing options:**

1. **"Diagnosis-guided repair selection"** — Honest language. The contribution is not the rewrites (which are textbook) but the *selection mechanism* (which is novel). Analogy: a doctor's value is diagnosis + prescription, not inventing the drugs.

2. **"Repair recommendation engine"** — Position Penumbra as a developer assistant that *recommends* repairs with explanations, not an automatic synthesizer. This is a weaker but more honest framing that reviewers cannot attack.

3. **Drop repair from the headline entirely.** Make the paper "EAG + Diagnosis" and present repair as a demonstration application. The paper becomes: "We introduce the EAG for causal diagnosis of FP error. As a demonstration, we show it enables targeted repair selection that outperforms blind approaches."

**My recommendation:** Option 3. The repair library is the weakest part of the proposal. Every minute spent defending "synthesis" is a minute not spent on the EAG and diagnosis, which are genuinely novel.

---

## 3. Creative Reframings

### Reframing A: Pure Empirical Study — "The Error Landscape of Scientific Python"

**What it is:** Drop all tool claims. Instead, instrument 10-20 real SciPy/sklearn pipelines at multi-precision and *characterize* the error landscape:
- Where does error accumulate? (Distribution of error across pipeline stages)
- What are the dominant error patterns? (Taxonomy frequency analysis)
- What is the graph structure of error flow? (Treewidth, path length, reconvergence frequency)
- How often do library calls dominate error? (Tier 1 vs. Tier 2 analysis)

**Why it works:**
- Requires only the shadow instrumentation + EAG builder (no repair, no certification)
- Novel empirical data that doesn't exist in the literature
- No theorem requirements — this is a measurement paper
- Treewidth measurements alone are a genuine contribution (no one has measured this)
- Directly useful to the FP tools community (Herbie, Verificarlo, Fluctuat developers)

**Venue:** ISSTA, ICSE (empirical software engineering track), SC.

**Probability of strong publication:** 55-65%. The data is novel, the methodology is straightforward, and the audience is well-defined. Risk: "measurement papers" are harder to make exciting. Needs a surprising finding (e.g., "90% of error in SciPy pipelines flows through <3% of operations" or "treewidth ≤ 3 for all 20 pipelines studied").

**What's preserved from original:** EAG representation, shadow instrumentation, treewidth measurements, diagnosis taxonomy (as classification framework, not automated tool).

### Reframing B: "EAG as a Program Representation" — Foundations Paper

**What it is:** The EAG is the sole contribution. No repair. Minimal diagnosis. Instead:
- Define the EAG formally (weighted DAG with sensitivity edges)
- Prove T1 (soundness — routine, 95% achievable)
- Prove T4 restricted to the EAG context (submodularity of error attribution)
- Measure treewidth across real programs
- Demonstrate 3 graph algorithms enabled by the EAG:
  1. Critical-path identification (max-weight path = dominant error propagation route)
  2. Cut-vertex analysis (single operations whose repair maximally reduces output error)
  3. Counterfactual estimation (predict error reduction from repairing node X without re-execution)

**Why it works:**
- Analogous framing to foundational representation papers: PDGs, SSA, e-graphs
- "What can you *compute* on this representation that you can't compute on raw shadow values?" has a clean answer
- T1 + T4 give proportionate math without requiring the risky T2 or τ
- Clean, focused contribution — one idea, not five

**Venue:** OOPSLA (program representation contributions), PLDI (with stronger proofs).

**Probability of strong publication:** 40-50%. Higher ceiling (PLDI!), lower floor (must demonstrate the EAG enables non-trivial algorithms). Risk: reviewers ask "why not just use automatic differentiation?" and the answer must be compelling.

**What's preserved from original:** EAG (everything), T1, T4, treewidth. Diagnosis and repair become "example applications" in Section 6, not headline contributions.

### Reframing C (Bonus): Radically Scoped Tool Paper — "Penumbra-Lite"

**What it is:** Cut scope to the absolute minimum publishable tool:
- Shadow instrumentation (Tier 1 only — drop LAPACK wrapping entirely)
- EAG construction
- Diagnosis (5 classifiers)
- No repair. No certification. No T4.
- Evaluation: FPBench + 3 real bugs + comparison to Herbie on diagnosis accuracy

**Why it works:**
- Implementation is ~15-20K LoC instead of 50-80K
- Achievable in 8 weeks
- Clean "diagnosis tool" paper without the repair complexity
- Head-to-head with Herbie on "which operations cause error" is a clean experiment

**Venue:** ISSTA tool track, FSE tool demonstrations, ICSE-SEIP.

**Probability of acceptance:** 50-60% at tool tracks. Lower prestige but higher probability.

---

## 4. Fatal and Serious Flaws (Honest)

### FATAL FLAW: Zero Artifacts After Theory Stage

**Severity: PROJECT-THREATENING**

The state is: `theory_bytes=0`, `impl_loc=0`, `code_loc=0`, `monograph_bytes=0`. After two pipeline stages (ideation + theory), the project has produced *excellent planning documents* and *nothing else*. The final_approach.md is one of the best-structured proposals I've seen — but a proposal is not a paper, and planning is not execution.

**The honest assessment:** This project has demonstrated strong analytical thinking and zero execution capability. The theory stage's empty output could be a pipeline failure — but even so, the implementation stage has not started. The 14-week timeline starts from literally zero lines of code.

**What makes this survivable:** The final_approach.md already acknowledged that this is primarily an engineering project. The theory was always supplementary. If the implementation stage begins immediately with a radically scoped version (Reframing C), the project can still produce a publishable artifact.

### SERIOUS FLAW: BC4 Remains Unvalidated

Finding ≥5 real pipeline-level bugs is an *existence claim* that cannot be planned around — the bugs either exist in a findable form or they don't. No amount of planning changes this. The final_approach.md's scouting strategy is sensible, but until someone actually mines those issue trackers and reproduces candidates, BC4 is hope, not evidence.

**Mitigation:** The 2-week scouting phase is correctly positioned as the first activity. If it fails, the pivot strategies (Reframings A/C) are available.

### SERIOUS FLAW: No Proof of EAG > Magnitude Sorting

The entire EAG contribution rests on the claim that sensitivity edges provide better diagnosis than simply sorting operations by error magnitude. This has never been demonstrated, even on a toy example. If the sensitivity edges are correlated with magnitude (which is plausible for many pipeline structures), the EAG reduces to expensive magnitude sorting.

**Mitigation:** This can be tested in week 1 with a prototype. Construct a 5-operation pipeline where error reconverges (fan-out/fan-in). Compare magnitude-sorting diagnosis to EAG-path diagnosis. If EAG wins on this toy example, the concept is validated. If not, the concept may be flawed.

### SURVIVABLE FLAW: 30-Pattern Repair Library

Already acknowledged by the proposal. Reframing as "repair selection" or dropping repair from the headline (Reframing B) resolves this cleanly.

### SURVIVABLE FLAW: First-Order Limitation

Acknowledged. The tool is most useful where users need it least (well-conditioned problems) and least useful where users need it most (ill-conditioned problems). This is an inherent scope limitation of first-order analysis, not a bug — but it must be stated prominently.

---

## 5. VERDICT: CONDITIONAL CONTINUE

**Vote: CONDITIONAL CONTINUE at 5.6/10 composite (down from 7.3 in prior round)**

**Rationale:** The `theory_bytes=0` signal is a significant negative update. My prior-round 7.3 was based on the assumption that theory development would produce at least T4 (which I assessed at 85% achievable). Zero output is below even pessimistic expectations, and I must update accordingly. However:

1. **The idea space remains strong.** The EAG is still a genuine contribution that no one has built. The problem is still real. The architecture is still sound. Nothing in the theory stage's failure invalidates the core idea.

2. **The proposal was never primarily theoretical.** The final_approach.md already positioned this as a tool paper with proportionate math. The theory was always supplementary. `theory_bytes=0` hurts the "proportionate math" angle but doesn't kill the tool contribution.

3. **Execution risk is now dominant.** The question is no longer "is this a good idea?" (yes) or "are the theorems true?" (probably, but unwritten). The question is "can this team ship a working tool and find real bugs in 14 weeks starting from zero?" This is a pure execution question.

**Conditions for CONTINUE:**

| ID | Condition | Gate | Consequence of Failure |
|----|-----------|------|----------------------|
| PC1 | Working shadow tracer on 1 SciPy function within 2 weeks | Week 2 | ABANDON — execution capability not demonstrated |
| PC2 | ≥3 pipeline-level bug candidates identified | Week 2 | Pivot to Reframing A or C |
| PC3 | EAG constructed for ≥1 real pipeline | Week 4 | ABANDON — core contribution unachievable |
| PC4 | EAG diagnosis demonstrably outperforms magnitude sorting on ≥1 example | Week 5 | Pivot to Reframing A (empirical study) |
| PC5 | T4 proof completed OR explicitly abandoned with empirical substitute | Week 8 | Proceed as pure tool paper |

---

## 6. Probability Estimates (Updated for theory_bytes=0)

| Outcome | Prior Round | Updated | Δ | Rationale |
|---------|-------------|---------|---|-----------|
| Publication at strong venue (SC/OOPSLA/FSE/ISSTA) | 45-55% | 30-40% | -15% | Zero artifacts + tight timeline |
| Best-paper at any venue | 5-10% | 3-6% | -4% | Need both execution + showstopper result |
| Publication at any venue (incl. workshop/tool track) | 65-75% | 50-60% | -15% | Reframings A/C are viable fallbacks |
| Project abandoned | 15-25% | 25-35% | +10% | PC1/PC3 failure more likely without momentum |
| ≥5 real bugs found (BC4) | 55-65%* | 45-55% | -10% | Unvalidated; no scouting done |
| T4 proof completed | 85%* | 60-70% | -20% | Was "85% achievable" but no attempt was made |
| EAG demonstrably > magnitude sorting | 65-75%* | 55-65% | -10% | Conceptually sound but unvalidated |
| Merged SciPy PR | 25-35%* | 15-25% | -10% | Requires finding bug + building tool + writing PR |

*Prior estimates from depth_check and final_approach.md, before theory_bytes=0 signal.

---

## 7. The One Thing That Would Change Everything

> **A working prototype that finds a real, previously-unknown numerical error in SciPy and produces a correct diagnosis that a developer confirms is actionable.**

Not a proof. Not an optimality theorem. Not a complete tool. A single concrete demonstration:

1. `penumbra trace scipy_pipeline.py` runs successfully
2. The EAG identifies an operation where error amplifies unexpectedly
3. The diagnosis says "catastrophic cancellation at line X of `scipy.special.Y`"
4. A SciPy developer says "huh, that's a real bug, I didn't know about that"

This single result would:
- Validate BC4 (real bugs exist and are findable)
- Validate the EAG (it found something magnitude-sorting might miss)
- Validate the diagnosis taxonomy (it correctly classified the root cause)
- Provide the "showstopper result" for the paper
- Make all reframings (A, B, C) viable simultaneously
- Raise every axis by 1-2 points

**Without this result**, Penumbra is an interesting idea with nice architecture diagrams. **With it**, Penumbra is a tool that finds real bugs. The gap between these two states is the gap between 5.6/10 and 7.5/10.

**The most efficient path to this result** is NOT to build the full 50K+ LoC system. It is:
1. Week 1: Prototype shadow tracer for 10 critical ufuncs (add, subtract, multiply, divide, exp, log, sqrt, sum, dot, matmul)
2. Week 1-2: Run on 3 known SciPy precision issues to validate tracing
3. Week 2-3: Add EAG construction for these 10 ufuncs
4. Week 3: Run on 5 *un*-known SciPy functions suspected of precision issues
5. Week 3-4: If anything surfaces → pursue it immediately

Build the minimum viable tracer, not the maximum viable system. The showstopper result is worth more than 80% of the planned features.

---

## 8. Summary for the Verification Committee

**The crown jewels survive the theory_bytes=0 signal:**
- The EAG as a novel program representation ✓ (doesn't need proofs to be novel)
- The diagnosis-first paradigm ✓ (conceptual contribution, not a theorem)
- The Python-native FP diagnosis gap ✓ (market reality, not a claim)
- T4 submodularity ✓ (still 60-70% achievable — it was never attempted, not falsified)
- Treewidth measurements ✓ (empirical, needs only implementation)

**What's damaged:**
- "Proportionate math" positioning (no math exists)
- Timeline credibility (14 weeks from zero is aggressive)
- Execution confidence (theory stage produced nothing)

**What's destroyed:**
- Nothing, actually. No claim has been falsified. No proof has failed. The absence of work is not evidence of impossibility — it's evidence of non-execution.

**My honest assessment as the Scavenging Synthesizer:** This is a project with a 7/10 idea and 0/10 execution to date. The idea deserves another chance, but the next stage must demonstrate execution, not more planning. If PC1 (working tracer in 2 weeks) fails, abandon. If PC1 succeeds, this project has a 50-60% chance of producing a publication and a 15-25% chance of producing something genuinely impactful.

The rubble contains crown jewels. But someone needs to actually pick them up.
