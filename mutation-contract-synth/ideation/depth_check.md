# Depth Check: MutSpec — Mutation-Driven Contract Synthesis

| Field | Value |
|---|---|
| **Slug** | `mutation-contract-synth` |
| **Area** | 009 — Programming Languages and Formal Methods |
| **Date** | 2025-07-18 |
| **Lead** | Team Lead (Synthesis) |
| **Panel** | Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer |
| **Auditor Score** | 5.375/10 — CONDITIONAL CONTINUE |
| **Skeptic Score** | 3.5/10 — ABANDON |
| **Synthesizer Score** | 7.0/10 — CONDITIONAL CONTINUE |
| **Synthesized Verdict** | **CONDITIONAL CONTINUE (2-1, Skeptic dissents)** |
| **Composite Score** | **6.0/10** |

---

## Executive Summary

MutSpec proposes to bridge mutation testing and specification inference by treating the mutation-survival boundary as a constructive specification source: killed mutants yield contract clauses, surviving mutants expose specification gaps. The theory introduces a completeness result (Theorem 3) for QF-LIA contracts over loop-free code and a Gap Theorem grounding surviving non-equivalent mutants as provable latent bugs or test-suite deficiencies. The engineering artifact is a SyGuS-based synthesis engine that takes PIT mutation results and emits SMT-verified JML contracts.

Three panelists independently evaluated and cross-critiqued the proposal. The central tension is between genuine theoretical novelty (all three panelists confirm the mutation-specification duality is constructively unexploited in prior literature) and severe practical unknowns (SyGuS scalability on mutation-derived grammars is completely unvalidated, Theorem 3 covers ~10–20% of real functions, and nobody has demonstrated developers want mutation-derived specifications). The panel converges on a **conditional continue** with binding empirical gates, dissent from the Skeptic who argues the unknowns are fatal.

---

## Pillar 1: EXTREME AND OBVIOUS VALUE

**Synthesized Score: 6/10** · Amendment Required

### Evidence

The three panelists score Value at 3 (Skeptic), 5 (Auditor), and 6→7 with amendment (Synthesizer), yielding a wide 3–7 range that reflects genuine uncertainty about who wants this tool and why.

**The demand problem is real.** The Skeptic's core objection — "nobody wants mutation-derived contracts" — cannot be dismissed. The proposal originates from a supply-side insight (mutation testing implicitly contains specifications) rather than a demand-side observation (developers asking for better spec inference). The Auditor reinforces this: "value is still hypothetical until someone runs the tool, finds bugs, and reports that the contracts were useful." No user interviews, no feature requests, no adoption signals beyond the theoretical elegance of the duality. The Skeptic's "$10K consultant refining Daikon output" challenge is deliberately provocative but highlights a genuine gap: the proposal provides no evidence that MutSpec's output is worth the engineering investment relative to cheaper alternatives.

**The LLM competition is under-addressed.** All three panelists note that LLM-based specification generation (Copilot, CodeWhisperer, GPT-4 with prompting) offers a radically cheaper alternative for "good enough" specifications. The proposal's counter — that LLM specs are statistically plausible but not test-grounded — is theoretically sound but practically untested. If LLM-generated specs satisfy 90% of developer needs at 1% of the cost, MutSpec's formal grounding is a distinction without a practical difference for most users.

**The bug-finding reframe is the strongest value proposition.** This is where the Synthesizer's analysis is most compelling. The Gap Theorem establishes that every non-equivalent surviving mutant violating the inferred contract is provably either a latent bug or a test-suite gap. This transforms MutSpec from "a specification generator nobody asked for" into "a formally grounded bug finder that extracts value from mutation testing runs teams already perform." The marginal-cost argument is key: teams running PIT in CI already pay the computational cost of mutation analysis and currently extract only a scalar (mutation score). MutSpec recovers structured specification data implicit in every kill/survive decision.

**However, the equivalent mutant problem contaminates the bug-finding story.** At 5–25% raw false positives (8–12% after TCE and symbolic filtering), a non-trivial fraction of gap reports will be false alarms. The Skeptic calls this FATAL; the Auditor compares it to SpotBugs (20–40% FP), Infer (15–25% FP), and Coverity (>15% FP) — all commercially viable despite comparable noise. The Lead weights the Auditor's comparison most heavily: 8–12% FP is within the range that industry static-analysis tools survive in, though at the lower end of what developers tolerate. It is serious but not disqualifying.

**Three plausible early adopters exist** (Synthesizer's identification): (1) security-sensitive teams already running mutation testing (Google, Diffblue, safety-critical domains); (2) verification tool builders consuming JML/ACSL annotations as input; (3) open-source maintainers of high-assurance libraries (Apache Commons Math, Guava, Bouncy Castle). These are narrow but real niches where the value proposition is defensible.

### Lead's Assessment

The Auditor's 5/10 is the most evidence-based score: the insight is genuinely novel but the value remains hypothetical. The Synthesizer's bug-finding reframe lifts the score to 6 by identifying a concrete, defensible use case (latent bug detection for teams already running mutation testing) that sidesteps the demand problem for specifications per se. The Skeptic's 3/10 applies a standard — "name a user who asked for this" — that would reject Daikon, SpotBugs, and most static-analysis research at the proposal stage. I weight the Skeptic's concerns as constraints to satisfy, not as evidence the project is worthless.

**Score: 6/10.** Below the 7/10 threshold. Amendment required.

### Required Amendment (VALUE-1)

> **Reframe primary deliverable as latent-bug detection, not contract synthesis.** The Gap Theorem is the product; contracts are the mechanism. The paper's title, abstract, introduction, and evaluation must lead with "formally grounded bug detection from mutation testing" and position contracts as a byproduct useful for teams that want them. The demand problem for contracts is real and unsolved; the demand for bug detection is established. This is not cosmetic reframing — it changes what RQ1 measures (bugs found and confirmed, not contract precision/recall against JML ground truth) and what the evaluation section emphasizes.

> **Validate marginal cost for existing PIT users.** The value proposition depends on MutSpec being cheap *on top of* mutation testing already being run. The evaluation must report MutSpec's marginal overhead (SyGuS + SMT time) separately from mutation analysis time, and demonstrate it fits within a CI window for realistic codebases.

---

## Pillar 2: GENUINE DIFFICULTY

**Synthesized Score: 6/10** · Amendment Required

### Evidence

Scores range from 4 (Skeptic) to 5 (Auditor) to 7 (Synthesizer). The disagreement centers on how much of the ~155K proposed LoC is genuinely novel research contribution versus engineering integration of existing tools.

**The LoC decomposition is settled.** All three panelists agree the novel core is approximately 21–25K LoC. The Auditor estimates 25–35K; the Skeptic estimates 21K; the Synthesizer counts ~25K across four genuinely hard components: mutation-directed grammar construction (~5K), CEGIS with mutation counterexamples (~6K), WP differencing engine (~4K), and SyGuS problem encoder (~6K). The remaining ~130K is multi-language support (MuIR unified IR for Java/C/Python), language front-ends, test infrastructure, and PIT integration — engineering that the panel unanimously agrees should be dropped for the paper (Java-only scope). This reduces the honest artifact to ~65K LoC total, ~25K novel.

**The "just a wrapper" characterization is too dismissive.** The Skeptic's framing — "PIT + CVC5 + Z3 wrapper" — undersells three genuinely hard design problems that the Synthesizer identifies:

1. **Mutation-directed grammar construction** has no precedent in SyGuS literature. The grammar must be expressive enough to capture mutation-derived specifications yet restrictive enough for CVC5 to converge. Getting this wrong means CVC5 either times out (grammar too large) or produces trivially weak specs (grammar too restrictive). No existing SyGuS benchmark addresses this search-space design problem.

2. **CEGIS with mutation counterexamples** is a structural variant of standard CEGIS. Standard CEGIS uses SMT-generated input *values* as counterexamples; MutSpec uses surviving mutant *programs*. These are structurally different objects, and the convergence behavior of the CEGIS loop changes when counterexamples are programs rather than points. This is a genuine research question, not an API call.

3. **WP differencing for batch mutation analysis** requires computing weakest-precondition differences between original and mutated programs efficiently. The formula per-mutant is textbook; batching across hundreds of mutants with incremental Z3 solving to share common sub-expressions is not.

**But the difficulty is not paradigm-shifting.** The Auditor's counterpoint is well-taken: a strong PhD student with expertise in PIT and CVC5 could build the core encoding in 6–9 months. The CEGIS variant is a *variant*, not a new algorithmic paradigm. The WP differencing shares structure with existing tools (CBMC, Frama-C). The comparison to KLEE (33K LoC novel symbolic execution with memory modeling, constraint solving, environment modeling, and search heuristics — each an open problem) is instructive: KLEE's 33K required solving multiple open problems simultaneously. MutSpec's 25K requires solving one primary open problem (grammar construction and SyGuS encoding) while competently integrating solutions to well-understood subproblems.

**The Skeptic's "6-month PhD student" estimate is probably right for a prototype.** Building a prototype that runs on 50 functions is a 6-month task. Building a tool that runs on 50K functions reliably within an 8-hour window, with graceful degradation, incremental solving, and sound verification — that is harder. But the difficulty delta between prototype and production is engineering, not research.

### Lead's Assessment

The Synthesizer's 7/10 overweights the design challenges relative to their novelty in the broader PL/FM landscape. The Skeptic's 4/10 underweights them by treating SyGuS encoding as a solved problem (it is not — no prior work encodes mutation boundaries as SyGuS constraints). The Auditor's 5/10 is close but slightly low: the grammar construction problem is genuinely open and the CEGIS variant is a real research contribution, even if bounded. I score 6/10: one genuinely novel research problem (SyGuS encoding of mutation boundaries) plus competent integration of well-understood components.

**Score: 6/10.** Below the 7/10 threshold. Amendment required.

### Required Amendment (DIFF-1)

> **Clearly delineate research contributions from engineering contributions.** The paper must explicitly identify: (a) what is *novel* (mutation-directed grammar construction, CEGIS-with-programs variant), (b) what is *competent integration* (PIT adapter, Z3 verification, JML emission), and (c) what is *future work* (multi-language, MuIR, higher-order mutations). Reviewers who perceive the paper as claiming 155K LoC of difficulty will reject it. Reviewers who see 25K LoC addressing one genuinely open problem with clean theory will evaluate it fairly.

> **Drop multi-language scope entirely.** Java-only for the paper. MuIR is an open design problem orthogonal to the core contribution. Mentioning it signals scope creep.

---

## Pillar 3: BEST-PAPER POTENTIAL

**Synthesized Score: 5/10** · Amendment Required

### Evidence

Scores: 3 (Skeptic), 5 (Auditor), 5.5→7 with amendment (Synthesizer). This is the most contested pillar and the one where the theoretical restrictions bite hardest.

**Theorem 3's restrictions are the central obstacle.** All three panelists agree Theorem 3 (specification completeness) is restricted to QF-LIA contracts, loop-free code, first-order mutants, and four mutation operators ({AOR, ROR, LCR, UOI}). The Skeptic's critique is sharpest: "Show one PLDI best paper with main theorem restricted to loop-free code." The comparison to KLEE (found bugs in GNU Coreutils), CompCert (verified real C compiler), and Alive (proved LLVM optimizations correct) is devastating. These best papers proved results about real-world artifacts; Theorem 3 proves a result about `int max(int a, int b)`.

The Synthesizer attempts to reframe the restriction as a strength ("restricted-but-proven worth more than general-but-conjectured") and argues QF-LIA is not as restrictive as it sounds (citing Ernst et al.'s finding that 70%+ of practically useful invariants are QF-LIA expressible). This defense is partially effective: the restriction is honest and the proof technique may illuminate the general case. But the gap between "70% of useful invariants are QF-LIA" and "Theorem 3 covers 10–20% of real functions" (the Auditor's estimate, accounting for loop-free requirement) is substantial.

**The "three-legged stool" is the best positioning available.** The Synthesizer proposes structuring the contribution as: (1) a surprising completeness result (Theorem 3), (2) a formally grounded bug detector (Gap Theorem), and (3) real bugs in real code. This structure is sound — each leg alone is publishable (theory paper, tool paper, empirical paper), and together they demonstrate the duality is both theoretically grounded and practically useful. But all three legs must work:

- **Leg 1 (Completeness):** Theorem 3 works but is restricted. Reviewers will ask "does this extend?" and the answer is "we conjecture yes, but proof only covers loop-free QF-LIA." This is a visible crack.
- **Leg 2 (Gap Theorem):** Clean formalism, but practical utility depends on equivalent-mutant filtering achieving ≤12% FP. Unvalidated.
- **Leg 3 (Real bugs):** Zero bugs found to date. The entire empirical story is projected, not demonstrated.

**SpecFuzzer overlap is real.** The Auditor rates SpecFuzzer closeness at 7/10. SpecFuzzer (ICSE 2022) already uses mutation as a *filter* on specification candidates. MutSpec's advance is using mutation to *construct* the SyGuS grammar rather than filtering a fixed grammar. This is a genuine technical advance, but the user-visible difference may be small if SpecFuzzer + post-hoc Z3 verification achieves comparable contract quality. The Skeptic argues this comparison must be run before investing further; the Lead agrees.

**The "between chairs" problem is unresolved.** The Skeptic identifies that the paper is too theoretical for ICSE (no user study, formal proofs), too restricted for POPL (QF-LIA/loop-free), and too incomplete as engineering for PLDI tools track (no evidence of scalability). The best target is PLDI or OOPSLA with the three-legged-stool framing, but this requires all legs to be solid.

**The Galois connection (A6) is cosmetic.** All three panelists agree: the lattice-theoretic framework connecting mutation subsumption to specification refinement is intellectually tidy but adds no practical insight or tool capability. The paper should lead with Theorem 3 and the Gap Theorem, not the Galois connection.

### Lead's Assessment

The Skeptic's 3/10 reflects legitimate concerns about Theorem 3's scope and the absence of empirical validation, but underweights the novelty of the mutation-specification duality itself — a constructive formalization nobody achieved in 40+ years of parallel work. The Synthesizer's 7/10 (with amendment) is optimistic: the three-legged stool has the right structure but all three legs are currently wobbly (restricted theorem, unvalidated Gap Theorem, zero bugs found). The Auditor's 5/10 is the most calibrated: genuine novelty with serious execution risk. The Auditor's probability estimate (5–10% best-paper) is more realistic than the Synthesizer's (15–25%).

The "40 years of disconnected fields" narrative must be softened. FormaliSE 2021, SpecFuzzer, EvoSpex, and IronSpec demonstrate the fields are under-connected, not disconnected. The amended framing: "No prior work has formalized the constructive duality between mutation adequacy and specification strength, though several systems have exploited aspects heuristically."

**Score: 5/10.** Below the 7/10 threshold. Substantial amendment required.

### Required Amendment (BEST-1)

> **Adopt three-legged stool structure with honest scope claims.** Leg 1: Theorem 3 as "lighthouse theorem" — proves the duality is real in a clean fragment, motivating heuristic generalization. Leg 2: Gap Theorem as main practical contribution — formally grounded bug detection. Leg 3: Empirical bugs in well-tested codebases — the evidence that makes or breaks the paper.

> **Prove quantitative degradation bound for sub-adequate suites.** The Auditor's suggestion is crucial: prove that a suite with mutation score *s* determines a specification capturing at least *g(s)* fraction of the strongest spec. This transforms Theorem 3 from an all-or-nothing result (requires 100% mutation adequacy) into a smooth function connecting mutation adequacy to specification quality. If this bound is achievable, it substantially strengthens Leg 1.

> **Extend Theorem 3 to QF-LIA + bounded loops.** Even partial extension (e.g., bounded loop unrolling with verification to depth *k*) would address the most frequent criticism. If full extension is intractable, provide extensive empirical evidence that the restricted result generalizes heuristically.

> **Run SpecFuzzer + Z3 baseline comparison before submission.** If MutSpec contracts are not measurably superior to SpecFuzzer + post-hoc Z3 on the evaluation benchmarks, the SyGuS machinery is unnecessary and the paper collapses.

> **Soften the "disconnected fields" narrative.** Replace with: "No prior work formalizes the constructive duality, though several systems exploit heuristic connections."

---

## Pillar 4: LAPTOP CPU + NO HUMANS

**Synthesized Score: 8/10** · Near-threshold, but SyGuS scalability is the critical unknown

### Evidence

Scores: 4 (Skeptic), 7.5 (Auditor), 8→9 with amendment (Synthesizer). The panel agrees the computational profile is CPU-only (no GPUs, no cloud, no human interaction) but disagrees sharply on whether the scalability arithmetic works.

**The CPU-only profile is genuinely clean.** PIT runs on JVM. CVC5 and Z3 are CPU-bound solvers. The entire pipeline — mutation analysis → SyGuS encoding → synthesis → SMT verification → JML emission — runs on commodity hardware without network access, GPU acceleration, or human intervention. All three panelists confirm this. There are no API calls, no training loops, no human-in-the-loop steps.

**The 8-hour claim does not survive arithmetic.** The Auditor's calculation: 50K functions × (mutation analysis + SyGuS + SMT) yields ~17 hours on 8 cores, not 8. The Skeptic's worst-case: 43 days serial, assuming every function is loopy with 50 dominator mutants and full test-suite execution per mutant. The Synthesizer's response: the Skeptic assumes worst-case everywhere (all functions loopy, 50 dominators each, no test selection), while reality is 40–60% loop-free with subsumption reducing 50 raw mutants to 5–10 dominators and test selection reducing per-mutant cost by 10–100×.

The truth is between the Auditor's 17 hours and the Synthesizer's 6–8 hours. For the paper's evaluation benchmarks (likely 500–5,000 functions, not 50K), the budget is comfortable. For production-scale codebases (50K+ functions), overnight CI windows (8–12 hours) are tight but potentially achievable with the three-tier strategy.

**The three-tier synthesis strategy is the Synthesizer's strongest contribution.** This is a genuinely useful engineering design:

- **Tier 1 (Full SyGuS):** Mutation-directed grammar decomposed per-mutation-site, composed via conjunction. Target: 80% of functions. Each sub-problem has O(k) atoms; CVC5 handles in seconds.
- **Tier 2 (Coarsened SyGuS):** For functions exceeding 120s timeout, progressively coarsen grammar by dropping redundant predicates. Target: 15% of functions. Emits strongest contract within budget, labeled with coarsening level.
- **Tier 3 (Template fallback):** Houdini-style conjoin-and-filter for functions where SyGuS fails entirely. Target: 5% of functions. Guarantees output for every function, though at Daikon-quality rather than SyGuS-quality.

All contracts are SMT-verified regardless of tier. Tier level is recorded in output metadata for downstream consumers to make trust decisions.

**The tier distribution is assumed, not measured.** The Auditor's critique is sharp: if 50% of functions push to Tier 3 (template fallback), half the system's output is Daikon-quality and the SyGuS machinery is overhead. The 80/15/5 distribution is a hypothesis that must be validated by the SyGuS feasibility experiment.

**SyGuS scalability on mutation-derived grammars is the Achilles heel.** No SyGuS benchmark includes grammars constructed from mutation analysis. CVC5's known performance envelope: reliable on ≤15 atoms and ≤50 constraints; >50% timeout rate at ≥30 atoms or ≥100 constraints (Auditor, citing SyGuS-Comp data). A typical 200-line function with 20 mutation sites × 5 operators produces ~100 data points. Even with 90% subsumption reduction, this yields 20–30 atoms — right at CVC5's reliability boundary. Whether the per-site decomposition (Tier 1 strategy) keeps individual sub-problems below this boundary is the single most important empirical question for the project.

### Lead's Assessment

The Skeptic's 4/10 is too pessimistic: it assumes worst-case everywhere and ignores the decomposition strategies that keep per-problem complexity manageable. The Synthesizer's 9/10 (with amendment) is slightly optimistic: the three-tier strategy is sound but the tier distribution is assumed. The Auditor's 7.5/10 is well-calibrated. I score 8/10: the CPU-only profile is clean, the three-tier design is pragmatically sound, and the scalability concern is real but addressable through the per-site decomposition. The 8-hour claim should be revised to "overnight CI window (8–12 hours)" to be honest.

**Score: 8/10.** Meets the ≥8 threshold but does not reach 9.

### What Must Change to Reach 9/10 (LAPTOP-1)

> **Validate SyGuS scalability empirically.** The 4-week feasibility study (50 Apache Commons Math functions) is the binding gate. Report: solve rate, time distribution (avg/median/p95), tier distribution (what fraction goes to Tier 1/2/3), and comparison to Daikon contract quality. If CVC5 solve rate is <70% within 120s per function, the three-tier strategy's Tier 1 assumption is invalid and the system degrades to an expensive Daikon.

> **Revise the 8-hour claim.** State "overnight CI window (8–12 hours on 8-core laptop)" and provide arithmetic showing this is achievable for the evaluation benchmarks. The 8-hour figure does not survive honest arithmetic and will undermine reviewer trust.

---

## Fatal Flaws

Seven flaws identified across the panel, ranked by severity. Severities reflect the Lead's synthesis, not any single panelist's rating.

### Flaw 1: SyGuS Scalability on Mutation-Derived Grammars Is Unvalidated

**Severity: CRITICAL (near-fatal) · Fixable with 4-week empirical validation**

No SyGuS benchmark includes grammars constructed from mutation data. CVC5's reliability boundary (~15 atoms, ~50 constraints) may be exceeded by typical mutation-derived problems. The three-tier synthesis strategy mitigates but does not eliminate this risk: if Tier 1 handles only 40% of functions (not 80%), the system's value proposition is substantially weakened. This is the single highest-risk item because it determines whether the core technical mechanism works at all.

**Fix:** Run the 4-week feasibility study on 50 Apache Commons Math functions. Report solve rate, time distribution, and tier distribution. Pass criterion: ≥70% solve rate within 120s. Fail criterion: <50% solve rate = ABANDON.

### Flaw 2: Equivalent Mutant Problem Contaminates Gap Theorem

**Severity: SERIOUS · Partially fixable**

5–25% of surviving mutants are equivalent (semantically identical to original); this is undecidable in general. TCE + symbolic equivalence filtering reduces false positives to 8–12%, but the system cannot identify *which* gap reports are false positives. The Skeptic rates this FATAL; the Auditor compares to SpotBugs/Infer FP rates and rates it SERIOUS-FIXABLE; the Synthesizer rates it comparable to industry tools.

**Lead resolution:** SERIOUS-FIXABLE. The 8–12% FP rate is within the range industry static-analysis tools survive in (SpotBugs 20–40%, Infer 15–25%, Coverity >15%). The paper must: (a) report FP rates honestly, (b) implement multi-tier confidence classification so downstream consumers can filter, (c) not claim "zero false positives" or "provably correct bug reports."

**Fix:** Implement TCE + symbolic equivalence detection. Report false-positive rates transparently. Label gap reports with confidence levels. Include equivalent-mutant filtering as explicit component in evaluation.

### Flaw 3: Theorem 3's Restricted Scope (QF-LIA, Loop-Free)

**Severity: SERIOUS · Partially fixable**

Theorem 3 (specification completeness) covers only QF-LIA contracts over loop-free code with first-order mutants and four operators. The Auditor estimates this covers 10–20% of real functions. For the remaining 80%+, the system falls back to heuristic synthesis without formal guarantees, undermining the formal-methods value proposition.

**Fix:** (a) Attempt extension to QF-LIA + bounded loops (even partial results help). (b) Prove quantitative degradation bound relating mutation score to specification quality. (c) Position Theorem 3 as "lighthouse theorem" illuminating the general principle, not necessary condition for tool utility. (d) Provide extensive empirical evidence that the restricted result generalizes heuristically.

### Flaw 4: Mutation-Adequate Test Suites Don't Exist in Practice

**Severity: SERIOUS · Fixable with degradation bound**

Theorem 3 requires mutation-adequate suites (killing all killable mutants). Real projects achieve ~70% mutation scores, meaning 30% of mutants survive — of which 22–28% are killable but unkilled. The Skeptic rates this FATAL because the theorem's precondition is vacuous in practice.

**Fix:** The Auditor's suggestion resolves this: prove a quantitative degradation bound showing that a suite with mutation score *s* determines a specification capturing at least *g(s)* fraction of the strongest specification. This transforms the all-or-nothing result into a smooth function. If this bound is provable, Flaw 4 is fully resolved. If not, the paper must be explicit that Theorem 3 is an asymptotic ceiling result, not a practical guarantee.

### Flaw 5: SpecFuzzer + Z3 Baseline Not Compared

**Severity: SERIOUS · Fixable with empirical comparison**

SpecFuzzer (ICSE 2022) already uses mutation as a filter on specification candidates. Adding post-hoc Z3 verification to SpecFuzzer output is a plausible simpler alternative that nobody has tested. If SpecFuzzer + Z3 achieves comparable contract quality to MutSpec, the SyGuS machinery is unnecessary overhead.

**Fix:** Run SpecFuzzer + Z3 on the same evaluation benchmarks. Demonstrate MutSpec contracts are measurably superior (tighter, fewer false clauses, catches bugs SpecFuzzer misses) or acknowledge parity and reposition the contribution.

### Flaw 6: Evaluation Ground Truth Availability

**Severity: SERIOUS · Fixable with logistics work**

RQ1 depends on JML-annotated subsets of DaCapo and community JML specs for Guava. The availability and quality of these ground-truth annotations is unverified. If ground truth doesn't exist in usable form, the evaluation plan collapses.

**Fix:** Verify JML ground-truth availability within 2 weeks. If insufficient, either (a) create ground-truth annotations as part of the project (adds ~4 weeks), or (b) redesign RQ1 around bug-finding (Gap Theorem validation) rather than contract precision/recall.

### Flaw 7: "Between Chairs" Venue Problem

**Severity: MODERATE · Fixable with positioning**

The paper is simultaneously too theoretical for ICSE, too restricted for POPL, and too incomplete as engineering for PLDI tools track. The three-legged stool structure (theory + tool + bugs) doesn't have a natural home if any leg is weak.

**Fix:** Target PLDI or OOPSLA with the three-legged stool. Lead with Gap Theorem (practical bug-finding), support with Theorem 3 (theoretical grounding), validate with empirical bugs. This framing fits PLDI's emphasis on "theory that enables practice." If Theorem 3 cannot be extended, FormaliSE is a realistic fallback venue for the theory alone.

---

## Amendments Summary

All amendments required before proceeding to implementation. Grouped by pillar.

### Value Amendments
- **VALUE-1:** Reframe primary deliverable as latent-bug detection via Gap Theorem; contracts are byproduct
- **VALUE-2:** Validate marginal cost for existing PIT users (report MutSpec overhead separately from mutation analysis time)
- **VALUE-3:** Identify and engage at least one early-adopter team (security-sensitive, verification tool builder, or high-assurance library maintainer) for feedback on gap-report actionability

### Difficulty Amendments
- **DIFF-1:** Clearly delineate research contributions (~25K LoC novel) from engineering contributions (~40K LoC integration) in the paper
- **DIFF-2:** Drop multi-language scope entirely; Java-only for paper and initial artifact
- **DIFF-3:** Position MuIR as future work, not contribution

### Best-Paper Amendments
- **BEST-1:** Adopt three-legged stool structure (completeness + Gap Theorem + real bugs)
- **BEST-2:** Prove quantitative degradation bound for sub-adequate suites, or explicitly characterize Theorem 3 as asymptotic ceiling
- **BEST-3:** Attempt extension of Theorem 3 to QF-LIA + bounded loops
- **BEST-4:** Run SpecFuzzer + Z3 baseline comparison before submission
- **BEST-5:** Replace "40 years of disconnected fields" with "No prior work formalizes the constructive duality, though several systems exploit heuristic connections"
- **BEST-6:** Drop Galois connection (A6) as centerpiece; lead with Theorem 3 and Gap Theorem

### Laptop CPU Amendments
- **LAPTOP-1:** Validate SyGuS scalability on 50 Apache Commons Math functions (4-week study)
- **LAPTOP-2:** Revise 8-hour claim to "overnight CI window (8–12 hours on 8-core laptop)" with supporting arithmetic
- **LAPTOP-3:** Implement and document three-tier synthesis strategy with graceful degradation
- **LAPTOP-4:** Report tier distribution empirically (what fraction of functions go to each tier)

---

## Composite Score and Verdict

| Pillar | Score | Weight | Weighted |
|--------|-------|--------|----------|
| Value | 6/10 | 25% | 1.50 |
| Difficulty | 6/10 | 20% | 1.20 |
| Best-Paper Potential | 5/10 | 25% | 1.25 |
| Laptop CPU + No Humans | 8/10 | 15% | 1.20 |
| Fatal Flaw Severity | 5/10 | 15% | 0.75 |
| **Composite** | | **100%** | **5.90 → 6.0/10** |

### Verdict: **CONDITIONAL CONTINUE** (2-1)

The composite score of 6.0/10 places the project above the ABANDON threshold (≤4) but below unconditional CONTINUE (≥7). The mutation-specification duality is a genuinely novel insight that all three panelists confirm has not been constructively formalized in prior work. The Gap Theorem offers a defensible, formally grounded bug-detection capability. However, every critical claim — SyGuS scalability, Gap Theorem practical utility, superiority over SpecFuzzer + Z3 — is currently hypothetical.

The project proceeds to a **4-week empirical validation phase** with binding pass/fail criteria. If validation succeeds, the project is greenlit for full implementation (Java-only, ~65K LoC, postconditions-first). If validation fails, the project is ABANDONED or pivoted to a theory-only paper at FormaliSE.

The Skeptic's ABANDON vote is recorded as a formal dissent. The Skeptic's four non-negotiable requirements are incorporated into the binding conditions below. The Lead notes that the Skeptic's standards, applied consistently, would have rejected Daikon (no demand evidence at proposal stage), SpotBugs (20–40% FP rate), and most static-analysis tools that subsequently achieved adoption. However, the Skeptic's insistence on empirical evidence *before* full investment is sound methodology, and the 4-week validation phase addresses this directly.

---

## Binding Conditions for CONTINUE

The following conditions are **mandatory gates**. Failure on any condition triggers the specified consequence.

### Gate 1: SyGuS Feasibility Study (4 weeks)

**Protocol:**
- Select 50 functions from Apache Commons Math (diverse: arithmetic, utility, complex mathematical operations)
- Run PIT to generate mutation data
- Encode mutation boundaries as SyGuS constraints
- Run CVC5 with 120s per-function timeout
- Record: solve rate, time distribution (avg/median/p95), tier distribution, contract quality

**Pass Criteria (ALL required):**
- CVC5 solve rate ≥ 70% within 120s (Tier 1 feasibility)
- Contracts measurably tighter than Daikon output on ≥ 60% of functions
- p95 solve time ≤ 120s (SyGuS budget compatible with overnight CI window)

**Fail Criteria:**
- Solve rate < 50%: **ABANDON** (SyGuS mechanism fundamentally incompatible with mutation-derived grammars)
- Solve rate 50–70%: **PIVOT** to template-based synthesis; reassess contribution novelty
- Contracts not measurably tighter than Daikon on ≥ 60%: **ABANDON** (no advantage over existing tools)

### Gate 2: Gap Theorem Validation (6 weeks, overlaps with Gate 1)

**Protocol:**
- Run gap analysis on the same 50 functions
- Manually classify every flagged surviving mutant: confirmed bug / test-suite gap / equivalent mutant / unclear
- Report false-positive rate with and without TCE + symbolic filtering

**Pass Criteria:**
- Confirmed bug + test-suite gap rate ≥ 75% of flagged mutants (FP ≤ 25% after filtering)
- At least 1 previously unknown, developer-confirmable defect or test-suite gap in a maintained library

**Fail Criteria:**
- FP rate > 40% after filtering: **ABANDON** (Gap Theorem "theoretically clean and practically useless")
- FP rate 25–40%: **CONDITIONAL** — reassess with additional filtering heuristics; do not proceed to full implementation without further FP reduction

### Gate 3: SpecFuzzer + Z3 Baseline (concurrent with Gates 1–2)

**Protocol:**
- Run SpecFuzzer on the same 50 functions
- Verify SpecFuzzer output with Z3
- Compare contract quality (tightness, false clause rate, bug detection) against MutSpec output

**Pass Criteria:**
- MutSpec contracts measurably superior on at least one dimension (tighter, fewer false clauses, catches bugs SpecFuzzer misses)

**Fail Criteria:**
- SpecFuzzer + Z3 matches MutSpec quality: **ABANDON** or pivot to theory-only paper (SyGuS machinery unnecessary)

### Gate 4: One Real Bug

**Protocol:**
- From gap analysis on evaluation benchmarks, identify at least 1 previously unknown defect in a maintained Java library
- Defect must be confirmable by library maintainers (not Defects4J replay, not synthetic)

**Pass Criteria:**
- ≥ 1 confirmed real bug

**Outcome if Zero Bugs Found:**
- Does not trigger automatic ABANDON, but significantly reduces best-paper potential
- Project may proceed if Gates 1–3 pass, but paper repositions as tool/theory contribution without empirical bug-finding claim

### Gate 5: Evaluation Ground Truth Verification (2 weeks, immediate)

**Protocol:**
- Verify availability of JML-annotated benchmarks for RQ1
- Identify specific files, annotation density, and quality

**Pass Criteria:**
- Sufficient JML annotations exist for ≥ 30 functions with non-trivial contracts

**Fail Criteria:**
- Insufficient annotations: redesign RQ1 around bug-finding (Gap Theorem) rather than contract precision/recall
- This is not a project-level ABANDON trigger; it changes evaluation design

---

## Panel Disagreements and Lead Resolution

### Disagreement 1: Equivalent Mutant Severity

| Panelist | Rating | Reasoning |
|----------|--------|-----------|
| Skeptic | **FATAL** | 8–12% FP destroys trust; developers run tool once, see false positives, never return |
| Auditor | **SERIOUS-FIXABLE** | Comparable to SpotBugs (20–40% FP), Infer (15–25% FP); industry tools survive at this noise level |
| Synthesizer | **TOLERABLE** | Standard for static analysis; multi-tier confidence classification mitigates |

**Lead Resolution: SERIOUS-FIXABLE (Auditor's position).** The Skeptic applies a zero-tolerance standard for false positives that no static-analysis tool in production meets. SpotBugs, Infer, and Coverity all operate at comparable or higher FP rates and are commercially viable. The key mitigation is honest reporting: the paper must report FP rates transparently, implement confidence-level classification, and not claim formal correctness of gap reports. The Skeptic's concern is valid as a *usability risk* but not as a *feasibility barrier*.

### Disagreement 2: Overall Difficulty

| Panelist | Score | Reasoning |
|----------|-------|-----------|
| Skeptic | **4/10** | "PIT + CVC5 + Z3 wrapper"; 21K LoC novel; 6-month PhD project |
| Auditor | **5/10** | Competent integration engineering; not paradigm-shifting; CEGIS variant is a variant |
| Synthesizer | **7/10** | Grammar construction is open problem; CEGIS-with-programs structurally novel; WP batching non-trivial |

**Lead Resolution: 6/10 (between Auditor and Synthesizer).** The Skeptic's "wrapper" characterization is too dismissive — no prior work encodes mutation boundaries as SyGuS constraints, and the grammar construction problem is genuinely open. But the Synthesizer overweights the difficulty relative to the PL/FM landscape: the CEGIS variant is a variant (not a new paradigm), and WP differencing has well-understood structure. The honest assessment is one genuinely novel research problem plus competent integration — harder than a wrapper, easier than a paradigm shift.

### Disagreement 3: SpecFuzzer Differentiation

| Panelist | Position | Reasoning |
|----------|----------|-----------|
| Skeptic | "SpecFuzzer + Z3 might match MutSpec" | SpecFuzzer already mutation-based; Z3 verification is straightforward addition |
| Auditor | "Comparison needed; cannot claim equivalence without testing" | Different synthesis mechanisms may produce different results; untested claim |
| Synthesizer | "SpecFuzzer has narrower expressiveness ceiling" | Fixed grammar vs. data-driven grammar is fundamental architectural difference |

**Lead Resolution: Empirical comparison required (binding condition).** The Synthesizer's theoretical argument (fixed vs. data-driven grammar) is sound but the practical difference is unknown. The Skeptic's suspicion may be correct — SpecFuzzer + Z3 may suffice for most functions. This must be tested, not argued. Gate 3 resolves this disagreement empirically.

### Disagreement 4: Best-Paper Probability

| Panelist | Estimate | Reasoning |
|----------|----------|-----------|
| Skeptic | ~0% | "Decent workshop contribution inflated by ambition" |
| Auditor | 5–10% | Genuine novelty but execution risk; restricted theorem visible crack |
| Synthesizer | 15–25% | Three-legged stool, if all legs hold; "obvious in hindsight" = hallmark of strong results |

**Lead Resolution: 5–10% (Auditor's estimate).** The base rate for best paper at PLDI is ~4% of accepted papers. The Auditor's 5–10% appropriately reflects that MutSpec has a genuinely novel insight (mutation-specification duality) that modestly exceeds base rate, but Theorem 3's restrictions, the absence of empirical results, and SpecFuzzer overlap prevent the 3–6× multiplier the Synthesizer claims. Best-paper potential depends almost entirely on whether the empirical validation (Gates 1–4) produces compelling results, particularly finding real bugs.

### Disagreement 5: Project Viability

| Panelist | Verdict | Core Argument |
|----------|---------|---------------|
| Skeptic | **ABANDON** | Fatal flaws in equivalent mutants and mutation adequacy; no evidence anyone wants this; theory restricted to toy fragment |
| Auditor | **CONDITIONAL CONTINUE** | Genuine novelty; hypothetical value must be validated; if bugs found, 7+ paper |
| Synthesizer | **CONDITIONAL CONTINUE** | Constructive amendments address all flaws; salvage value even if full project fails |

**Lead Resolution: CONDITIONAL CONTINUE with Skeptic's empirical gates.** The Skeptic's ABANDON is based on standards that would reject most static-analysis research at the proposal stage. However, the Skeptic's *specific requirements* (SyGuS feasibility, Gap Theorem validation, SpecFuzzer comparison, one real bug) are exactly the right empirical gates. The project proceeds with these gates as binding conditions. The 4-week validation phase is cheap relative to full implementation and provides the evidence all three panelists agree is missing.

---

## Salvage Value (if ABANDONED after validation)

If the empirical validation fails and the project is abandoned, the following artifacts retain independent value:

1. **Theory paper at FormaliSE or ICSE-NIER:** Theorems A1–A3 (mutation-specification duality, completeness for QF-LIA/loop-free) as a standalone 4-page contribution. Publication probability: ~60%. Value: establishes priority on the duality insight.

2. **MutGap-Lite:** Standalone gap-analysis tool (~10K LoC) that takes PIT output and classifies surviving mutants as likely-bug / test-gap / equivalent using heuristic filtering. No SyGuS dependency. Useful as teaching tool and research prototype.

3. **WP differencing engine:** Reusable component for batch weakest-precondition computation with incremental Z3 solving. Applicable to other mutation-analysis and program-equivalence problems.

4. **SyGuS-from-mutations benchmark:** Even if CVC5 fails on mutation-derived grammars, the benchmark itself is a contribution to the SyGuS community (novel problem class).

---

## Minimal Viable Version

If all gates pass, the minimal artifact for a publishable paper is:

- **Scope:** Java-only, postconditions-only (preconditions as stretch goal)
- **Size:** ~30K LoC (25K novel + 5K infrastructure)
- **Components:** PIT adapter → mutation-directed grammar constructor → CVC5 SyGuS synthesis → Z3 verification → JML emitter → gap reporter
- **Evaluation:** 50–200 functions from Apache Commons Math + Guava; RQ1 (contract quality vs. Daikon + SpecFuzzer), RQ2 (bugs found via Gap Theorem), RQ3 (scalability/tier distribution)
- **Timeline:** 4 weeks validation + 12 weeks implementation + 4 weeks evaluation + 4 weeks writing = ~24 weeks
- **Target venue:** PLDI or OOPSLA (three-legged stool: theory + tool + bugs)

---

## Appendix: Score Reconciliation Table

| Pillar | Skeptic | Auditor | Synthesizer | Lead (Synthesized) |
|--------|---------|---------|-------------|-------------------|
| Value | 3/10 | 5/10 | 6→7/10 | **6/10** |
| Difficulty | 4/10 | 5/10 | 7/10 | **6/10** |
| Best-Paper | 3/10 | 5/10 | 5.5→7/10 | **5/10** |
| Laptop CPU | 4/10 | 7.5/10 | 8→9/10 | **8/10** |
| Fatal Flaw Severity | 2/10 | 5/10 | 7/10 | **5/10** |
| **Composite** | **3.5/10** | **5.375/10** | **7.0/10** | **6.0/10** |
| **Verdict** | ABANDON | COND. CONTINUE | COND. CONTINUE | **COND. CONTINUE** |

*Weighting: Auditor weighted most heavily (evidence-based, moderate). Skeptic concerns incorporated as binding gates. Synthesizer amendments adopted where substantive.*

---

*End of depth check. Next action: begin Gate 1 (SyGuS feasibility study on 50 Apache Commons Math functions). Timeline: 4 weeks. If Gate 1 fails, project terminates without further investment.*
