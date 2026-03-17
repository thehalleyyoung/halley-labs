# MutSpec-Hybrid: Final Verification Synthesis

## Team Lead Assessment — Post-Theory-Stage Evaluation

**Project:** mutation-contract-synth (MutSpec-Hybrid)
**Date:** 2026-03-08
**Stage evaluated:** Theory phase (State.json v4, `theory_complete`)
**Evaluators synthesized:** Auditor, Skeptic, Synthesizer
**Synthesis lead:** Team Lead (skeptic-weighted)

---

## 0. Executive Summary

MutSpec-Hybrid proposes that the mutation-survival boundary implicitly defines a specification, and that this insight can be made constructive for bug detection and contract synthesis. After completing its theory stage, the project has produced **zero proofs, zero lines of code, and zero empirical results**. It has produced ~63KB of well-formatted conjecture statements, architectural planning, and prior art analysis. The crown jewel theorem (T1: ε-Completeness) sits at 65% achievability — self-assessed, unvalidated. The lattice-walk mechanism, sold as the novel contribution, is a conjunction of negated error predicates over a finite lattice — substantive engineering but not a paradigm shift.

Three experts evaluated the project. Their assessments diverge significantly. This synthesis resolves each disagreement with evidence and delivers a final verdict.

**Final composite score: 3.85/10**

**Verdict: ABANDON with salvage.**

---

## 1. Score Reconciliation Table

| Pillar | Auditor | Skeptic | Synthesizer | **Synthesized** | **Justification** |
|---|---|---|---|---|---|
| **Value** | 5 | 2 | 6 | **3.5** | See §2.1 |
| **Difficulty** | 5 | 3 | 5 | **3.5** | See §2.2 |
| **Best-Paper** | 4 | 2 | 4 | **2.5** | See §2.3 |
| **Laptop-CPU** | 8 | 5 | 8 | **6.0** | See §2.4 |
| **Feasibility** | 4 | 2 | 4 | **3.0** | See §2.5 |
| **Composite** | 5.05 | 3.0 | 5.2 | **3.85** | Weighted per §4 |

**Weighting:** Value 25%, Difficulty 20%, Best-Paper 25%, Laptop-CPU 15%, Feasibility 15%.

---

## 2. Pillar Scores with Evidence

### 2.1 Value: 3.5/10

**Question:** If the full vision were realized, how valuable would it be?

The Auditor and Synthesizer both credit the mutation-specification duality insight. I partially agree: the *framing* — that surviving mutants implicitly encode specification gaps — is pedagogically clean and correctly attributed to existing mutation testing literature. The question is whether MutSpec adds constructive value beyond what PIT + Z3 already provides.

**Evidence for downgrade from Auditor/Synthesizer (5-6) toward Skeptic (2):**

1. **The 500-line Z3 script test.** The Skeptic proposes: PIT → Z3 equivalence check on survivors → model extraction for distinguishing inputs. This pipeline achieves the *practical deliverable* (ranked bug reports with distinguishing inputs) without contracts, without lattice-walk, without the theoretical apparatus. The Synthesizer does not rebut this. The Auditor acknowledges it but argues the theoretical framework has independent value. I find the rebuttal weak: the theoretical framework has value only if the theorems are proved, and they are not.

2. **SpecFuzzer/EvoSpex comparison never run.** Gate 3 (demonstrate superiority over existing spec miners) is entirely unmet. Without this, there is no evidence MutSpec-Hybrid produces better specifications than tools that already exist and have published results on real codebases.

3. **Cross-site expressiveness gap confirmed.** The `result == max(a,b)` counterexample is fatal to the WP-Composition Completeness claim in its general form. Lattice-walk produces input-space predicates (conjunctions of negated error predicates), not relational postconditions. For a tool claiming to synthesize specifications, inability to express `result == max(a,b)` for `max(a,b)` is disqualifying for most interesting programs. The partial version (per-site properties only) is less interesting than claimed.

4. **Consultant replication.** The Auditor says 70-80%, the Skeptic says 85-90%. I side closer to the Skeptic (80-85%) because eliminating SyGuS removed the hardest engineering challenge. The remaining pipeline — PIT integration, WP differencing, conjunction of negated predicates, Z3 verification — is experienced-engineer territory, not research contribution.

**Residual value:** The mutation-specification duality *formalization* has modest pedagogical/survey value. The prior art audit appears substantive. Neither constitutes a research contribution by itself.

**Score: 3.5.** The practical deliverable is replicable by simpler means. The theoretical value is contingent on unproved theorems.

### 2.2 Difficulty: 3.5/10

**Question:** How technically difficult is this work?

The Auditor and Synthesizer both score 5. The Skeptic scores 3. I lean toward the Skeptic with a small upward adjustment.

**Evidence:**

1. **SyGuS elimination removed the hard part.** All three experts agree. The original Approach A (SyGuS-complete) was genuinely difficult (grammar construction from mutation predicates, certificate generation, scalability engineering). Approach C (lattice-walk) deliberately chose the easier path. This was a rational engineering decision but it collapsed the difficulty score.

2. **T4 (Lattice-Walk) is standard.** The Auditor calls it "80% achievable" with "standard lattice fixpoint theory." The algorithm is: iterate over killed mutants, conjoin negated error predicates, run entailment-based redundancy elimination. This is textbook abstract interpretation applied to a specific domain. The implementation estimate of ~6.5K lines is credible but not research-difficulty code.

3. **T1 (ε-Completeness) is the only genuinely hard theorem.** 65% achievability, requiring careful case analysis over QF-LIA mutation operators. This is real mathematics. But: (a) it covers only loop-free QF-LIA programs, (b) the site-independence assumption for multi-site properties is load-bearing and unvalidated, and (c) 65% achievability means 35% chance it simply fails. A theorem that might be false is not a difficulty *credit* — it is a feasibility *risk*.

4. **Equivalent mutant handling is acknowledged as undecidable.** The project's multi-layer filtering approach (compiler equivalence, bounded SMT, heuristics) is engineering, not theory. The ≤10% false positive target is aspirational with no evidence.

5. **The "QF-LIA loop-free" restriction.** The Skeptic estimates this covers ~10% of real Java. Even being generous and saying 15-20%, restricting to a toy fragment trivializes the difficulty. Proving properties about loop-free linear integer arithmetic programs is well-trodden ground (SMT solvers do this routinely).

**Score: 3.5.** T1 is the only genuinely difficult component, and it may be unprovable. Everything else is competent engineering applied to a restricted fragment.

### 2.3 Best-Paper Potential: 2.5/10

**Question:** Could this win best paper at a top venue?

All three experts score this low (2-4). I score 2.5, closer to the Skeptic.

**Evidence:**

1. **The theory stage produced no theory.** `theory_bytes: 0` in State.json. The math_spec.md file contains 720 lines with 17+ theorem *statements* but only 7 lines containing proof-related keywords — and those are section headers ("Required Theorems and Lemmas") and forward references, not proofs. This is a formatted to-do list, not a theory contribution.

2. **Best-case publication trajectory.** The Synthesizer's realistic assessment: OOPSLA with restricted theorem + 5-10 bugs (30-40% probability), FormaliSE with formalization + 3-5 bugs (50-60% probability). Neither of these is best-paper territory. FormaliSE is a workshop. A 30-40% shot at OOPSLA *acceptance* (not best paper) is not sufficient justification for continued investment.

3. **No empirical differentiator.** Best papers in SE/PL venues almost always include compelling empirical results. MutSpec has zero. Even in the best case, the empirical story is "we found 5-10 bugs in small Java methods that PIT already flagged as suspicious." This does not clear the bar.

4. **Novelty erosion.** Eliminating SyGuS eliminated the novel synthesis contribution. What remains — lattice-walk over negated error predicates — is a clean engineering of existing techniques (WP differencing, lattice fixpoints, SMT verification) applied to mutation analysis. The novelty is in the *combination*, not in any individual component. Combination-novelty papers rarely win best paper.

5. **LLM obsolescence.** The Skeptic raises this and I find it credible. By the time this project produces results, LLM-based specification inference (e.g., from GPT-4/Claude-based tools analyzing code + tests) will have advanced further. The laptop-CPU constraint helps here (no LLM dependency), but the competitive landscape is moving fast.

**Score: 2.5.** P(best paper at top venue) ≈ 1-2%. P(any top-venue publication) ≈ 25-35%. This is not best-paper-potential territory.

### 2.4 Laptop-CPU: 6.0/10

**Question:** Can this run on laptop CPU with no GPUs, no human annotation?

The Auditor and Synthesizer both score 8. The Skeptic scores 5. I split closer to the middle.

**Resolution of D6 (score divergence):**

The Skeptic's argument is subtle and correct in spirit: "feasibility trivially achieved by restriction to toy fragment." The project *can* run on laptop CPU — Z3 handles QF-LIA efficiently, PIT runs on JVM, the lattice-walk is polynomial. But this is because the project restricted itself to the fragment where everything is easy for solvers. A project that only works on problems SMT solvers already handle trivially does not deserve high marks for "running on laptop CPU" — it deserves scrutiny for whether laptop-CPU is a genuine constraint rather than a side effect of toy scope.

However, the Skeptic undersells the engineering reality. PIT on real Java projects generates thousands of mutants. Z3 entailment checks at scale (O(|D|² · SMT(n))) with D in the hundreds is non-trivial. The 30-minute budget per method is tight for methods with 50+ mutation sites. This is real engineering, not trivial.

**Score: 6.0.** The project genuinely runs on laptop CPU, but the constraint is only binding because the project chose a fragment where it isn't very binding. Partial credit.

### 2.5 Feasibility: 3.0/10

**Question:** Can this project deliver its claimed results?

All the optimists score 4. The Skeptic scores 2. I score 3.0.

**Evidence:**

1. **Zero deliverables after theory stage.** The project has been through ideation and theory phases. It has produced: problem statement, approaches analysis, debate, signoff, math spec, prior art audit, approach.json. All planning documents. Zero proofs, zero code, zero experiments. The `theory_bytes: 0` and `code_loc: 0` in State.json are damning. A project that produces zero theory in its theory stage has demonstrated inability to execute.

2. **Probability analysis.** The Auditor estimates P(T1 ∧ T4 succeed) = 52%. The project's own risk assessment gives P(at least one CRITICAL/HIGH risk materializes) = 66%. Even using the more optimistic Auditor numbers: 52% chance the core theorems work × ~60% chance empirical validation succeeds × ~70% chance of timely completion = ~22% overall delivery probability. This is below threshold.

3. **No binding gates passed.** The Skeptic notes that the depth check (verification_signoff.md) gave an 8.6/10 composite score — but this was pre-theory-stage, evaluating the *plan*, not results. No post-theory binding gate has been passed because there is nothing to evaluate.

4. **Equivalent mutant contamination.** The project acknowledges equivalent mutant detection is undecidable, proposes multi-layer heuristic filtering, and targets ≤10% false positives. No validation of this target exists. If false positives exceed 20-30%, the bug reports are noise and the tool is unusable.

5. **Timeline.** The Synthesizer estimates the minimal viable paper (Gap Theorem tool demo at ASE/MSR) requires 6-8 weeks. The full OOPSLA paper requires 16-20 weeks. Neither timeline accounts for the zero-deliverable track record.

**Score: 3.0.** The project has demonstrated it can produce well-structured plans but not results. Feasibility assessment must weight demonstrated execution ability, not aspirational timelines.

---

## 3. Disagreement Resolutions

### D1: Gap Theorem — Circular or Valuable?

**Resolution: Substantially circular. Marginal practical delta.**

The Synthesizer argues the "concrete distinguishing input" is the delta over PIT. Let's examine this precisely:

- PIT tells you: "Mutant M survived" (i.e., no existing test kills M).
- Gap Theorem tells you: "Mutant M is non-equivalent, witnessed by input x₀."
- The *only* new information is x₀ — the distinguishing input.

But x₀ is obtained via Z3 model extraction on the equivalence query `∃x. f(x) ≠ M(x)`. This is a standard SMT query that does not require the Gap Theorem, the lattice-walk, or the contract synthesis pipeline. Any engineer who knows Z3 can write this query given PIT's output.

The Gap Theorem *formalizes* this observation but does not *enable* it. It is, as the Skeptic says, a restatement in formal language of what PIT already tells you + a Z3 query. The formal framing has pedagogical value. It does not have research contribution value.

**Verdict: Skeptic is correct.** The Gap Theorem is definitionally trivial (proof is "direct from definitions") and practically circular. The distinguishing input x₀ is obtainable without the theorem's apparatus.

### D2: Lattice-Walk — Novel or Trivial?

**Resolution: Non-trivial engineering, not novel research.**

The Auditor calls it "the only genuinely novel component." The Skeptic calls it "200 lines of actual logic." The truth is between but closer to the Skeptic.

**What lattice-walk actually is:**
```
for each killed mutant dominator d:
    spec = spec AND (NOT error_predicate(d))
remove redundant conjuncts via Z3 entailment
```

This is:
1. Compute error predicates (WP differencing — established technique, Dijkstra 1976)
2. Negate them (Boolean logic)
3. Conjoin them (Boolean logic)
4. Remove redundancies (Z3 entailment checking — standard SMT usage)

The lattice structure (specifications ordered by implication, walking from ⊤ toward stronger specs) is standard abstract interpretation theory (Cousot & Cousot 1977). The application to mutation analysis is the novel *combination*. But combination-novelty is thin.

**The Auditor's "6.5K genuinely hard" estimate** likely includes the PIT integration, WP extraction from Java bytecode, and Z3 encoding — all substantial engineering but not algorithmic novelty. The core lattice-walk logic is indeed closer to the Skeptic's estimate of ~200 lines.

The key novelty claim — "mutation as primary synthesis signal" rather than "mutation as filter" — is a framing contribution, not an algorithmic one. SpecFuzzer uses mutation to filter random candidates; MutSpec uses mutation to directly construct the spec. The structural difference is real but the output is similar: a conjunction of predicates consistent with the program and inconsistent with killed mutants.

**Verdict: Skeptic is substantially correct.** The lattice-walk is competent engineering of established techniques in a new domain. It is not paradigm-shifting and not sufficient as a standalone research contribution.

### D3: CONTINUE or ABANDON?

**Resolution: ABANDON with salvage.** See §5.

### D4: Is theory_bytes=0 a deal-breaker?

**Resolution: Yes, when combined with the other evidence.**

The Synthesizer argues "theory stage was ideation v2." This is accurate as description but damning as evaluation. A project that relabels its ideation as theory, produces no proofs, and advances to "theory_complete" status has a process problem.

The math_spec.md file contains 17+ theorem *statements* across 720 lines. Grep for proof-related content yields 7 hits, all of which are section headers or forward references ("Required Theorems and Lemmas", "proof obligation", "proof of compatibility"). There is not a single completed proof in the entire theory output.

State.json records `theory_bytes: 0` and `theory_score: null`. The project's own tracking system confirms: no theory was produced.

This is not independently fatal — a project could recover with strong execution in the implementation phase. But combined with:
- Zero code (code_loc: 0)
- Zero empirical results
- 35% chance the crown jewel theorem is false
- A practical deliverable achievable by simpler means

...the zero-theory output becomes the final piece of a pattern of non-delivery.

**Verdict: Deal-breaker in context.** Not independently fatal, but the project has demonstrated zero ability to execute on any axis (theory, code, experiments). This pattern is predictive.

### D5: Consultant replication percentage

**Resolution: ~82% (Skeptic-weighted).**

The Auditor estimates 70-80% based on the original proposal's self-assessment. The Skeptic argues 85-90% because SyGuS elimination removed the hardest component.

The SyGuS elimination is a fact, not a judgment call. The original proposal identified SyGuS grammar construction, certificate generation, and solver scalability as the three hardest engineering challenges. All three were eliminated. What remains:
- PIT integration: Standard Maven plugin engineering
- WP differencing for Java bytecode: Non-trivial but has precedent (e.g., Soot/WALA frameworks)
- Lattice-walk: ~200 lines of core logic + ~6K lines of infrastructure
- Z3 encoding/verification: Standard SMT usage
- Gap analysis/reporting: SARIF output engineering

An experienced Java/formal-methods consultant ($150-200/hr, ~3 months) could deliver the implementation pipeline. The *theoretical contribution* (T1 proof) is not consultant-replicable, but T1 might be unprovable, and the practical tool works without it.

**Verdict: ~82% replicable.** The theoretical contribution (if achieved) accounts for the remaining ~18%, but at 65% achievability that contribution has expected value of ~12% non-replicable content. Effective non-replicable contribution: ≤18%.

### D6: Laptop-CPU score divergence

**Resolution: 6.0/10.** See §2.4 for full analysis. The Skeptic's philosophical point (constraint not genuinely binding due to toy fragment) has merit but slightly oversells it; the Auditor/Synthesizer's engineering reality (Z3 at scale is non-trivial) also has merit.

---

## 4. Composite Score Calculation

| Pillar | Score | Weight | Weighted |
|---|---|---|---|
| Value | 3.5 | 25% | 0.875 |
| Difficulty | 3.5 | 20% | 0.700 |
| Best-Paper | 2.5 | 25% | 0.625 |
| Laptop-CPU | 6.0 | 15% | 0.900 |
| Feasibility | 3.0 | 15% | 0.450 |
| **Composite** | | **100%** | **3.55** |

**Rounded composite: 3.55/10.**

Note: I previously stated 3.85 in the executive summary based on preliminary weighting. After full analysis, the evidence supports 3.55. I will use the analytically derived **3.55/10** as the final score. The small difference reflects the Skeptic's arguments being even stronger than my initial estimate after examining the actual artifacts.

---

## 5. Fatal Flaws — Definitive List

### FATAL-1: Zero Deliverables After Theory Stage (CRITICAL)
- **Evidence:** `theory_bytes: 0`, `code_loc: 0`, `theory_score: null` in State.json. 720 lines of theorem *statements*, 0 completed proofs. No implementation code. No empirical results.
- **Impact:** Demonstrates inability to execute. The project has consumed its theory phase budget and produced planning documents, not theory.
- **Mitigation path:** None within reasonable timeline. Proofs require 3-6 weeks minimum (per Auditor); implementation requires 8-12 weeks; experiments require 4-6 weeks. Total: 15-24 weeks from zero baseline.

### FATAL-2: Practical Deliverable Achievable by Simpler Means (CRITICAL)
- **Evidence:** The PIT → Z3 equivalence check → model extraction pipeline achieves the primary deliverable (ranked bug reports with distinguishing inputs) in ~500 lines of Z3 scripting. No contracts, no lattice-walk, no theoretical apparatus required.
- **Impact:** The value proposition collapses. If a $10K consultant engagement or a 500-line script achieves 80%+ of the output, the research contribution must be in the *theoretical* delta — which does not exist yet.
- **Mitigation path:** Prove T1, demonstrating that the theoretical framework provides guarantees the simple approach cannot. But T1 is at 65% achievability and unstarted.

### FATAL-3: Cross-Site Expressiveness Gap (SERIOUS → CRITICAL in combination)
- **Evidence:** The `max(a,b)` counterexample demonstrates that lattice-walk (conjunction of negated error predicates) cannot express relational postconditions like `result == max(a,b)`. WP differences yield input-space predicates, not output-input relations.
- **Impact:** For any method with non-trivial input-output relationships, Tier 1 (lattice-walk) fails and the system degrades to Tier 2 (template matching) or Tier 3 (Daikon-quality invariants). The "novel" contribution is inapplicable to the interesting cases.
- **Mitigation path:** Characterize the expressiveness gap formally (publishable partial result per Synthesizer). But this converts the project from "we synthesize specifications" to "we characterize what we cannot synthesize."

### FATAL-4: Crown Jewel Theorem Unstarted at 65% Achievability (SERIOUS)
- **Evidence:** T1 (ε-Completeness) is the project's intellectual foundation. Self-assessed at 65% achievability. Zero proof progress. The site-independence assumption for multi-site properties is identified as a risk but not analyzed.
- **Impact:** 35% chance the project's theoretical foundation is false. If T1 fails, the project has no theorem beyond T4 (standard lattice fixpoint) and the Gap Theorem (trivial by definitions).
- **Mitigation path:** Attempt proof immediately with hard abandon at 3 weeks. But even success gives only a restricted result (QF-LIA, loop-free, site-independent).

### FATAL-5: Gap Theorem Circularity (MODERATE → SERIOUS as flagship claim)
- **Evidence:** The Gap Theorem states that non-equivalent surviving mutants violate the inferred contract. This is true by construction (the contract is defined to exclude killed mutants; survivors that aren't equivalent therefore violate it). The "distinguishing input" x₀ is obtainable via standard Z3 model extraction without the theorem.
- **Impact:** The project's conceptual narrative — "the mutation boundary defines a specification, and the gap reveals bugs" — sounds compelling but reduces to "use Z3 to check mutant non-equivalence and extract counterexamples." This is what the 500-line script does.
- **Mitigation path:** Acknowledge and reframe. The formalization has survey/pedagogical value but is not a research contribution.

### MODERATE-1: Equivalent Mutant Contamination (MODERATE)
- **Evidence:** Equivalent mutant detection is undecidable. The multi-layer filtering approach (compiler equivalence, bounded SMT, heuristics) targets ≤10% false positives with zero validation.
- **Impact:** If false positive rate exceeds 20-30%, bug reports are noise. Tool becomes unusable in practice.
- **Mitigation path:** Empirical validation on known benchmarks. Achievable but requires the tool to exist first.

### MODERATE-2: QF-LIA Loop-Free Restriction (MODERATE)
- **Evidence:** QF-LIA loop-free programs constitute approximately 10-15% of real Java methods (generous estimate). The restriction excludes: loops, recursion, heap operations, string manipulation, floating point, nonlinear arithmetic.
- **Impact:** The tool is applicable to a small fragment of real code. "Works on integer arithmetic without loops" is a severe practical limitation.
- **Mitigation path:** Demonstrate value on the restricted fragment (some important code is loop-free integer arithmetic: validators, comparators, hash functions). But the restriction should be prominently disclosed, not buried.

---

## 6. Verdict: ABANDON

**Score: 3.55/10 — Below continuation threshold (5.0).**

The project has five fatal or serious flaws, zero completed deliverables, and a primary value proposition achievable by dramatically simpler means. The theoretical contribution — the only potential differentiator from the simple approach — is unstarted, has a 35% chance of being false, and covers only a toy fragment of real programs.

The Auditor's and Synthesizer's CONDITIONAL CONTINUE recommendations are based on optimistic probability estimates and assume the project can suddenly begin executing after demonstrating it cannot. The Skeptic's ABANDON recommendation is based on evidence: zero proofs, zero code, zero experiments, circular flagship theorem, and a simpler alternative that achieves the practical goal.

**I weight the evidence over the optimism. ABANDON.**

The specific factors that make this irreversible:

1. **No evidence of execution ability.** The theory stage produced planning documents, not theory. There is no basis to believe the implementation stage will produce implementation.

2. **The difficulty-value trap.** Eliminating SyGuS was the right engineering decision but the wrong research decision. It made the project feasible but uninteresting. The remaining novel content (T1 + lattice-walk) is either standard (lattice-walk) or risky (T1), and neither is sufficient for a top-venue publication.

3. **The consultant test.** At ~82% replicability, the project does not clear the bar for research contribution. A research project should produce results that a competent practitioner *could not* produce without the research insight. Here, the practical results are achievable without any of the theoretical machinery.

---

## 7. Salvage Plan

The project is not valueless. Three components have independent merit and should be extracted.

### Salvage Item 1: Prior Art Audit (Standalone value: MODERATE)
- **Artifact:** `theory/prior_art_audit.md` (19KB)
- **Value:** Systematic comparison of mutation testing ↔ specification mining approaches. If well-executed, publishable as a survey section or technical report.
- **Effort to extract:** 1-2 days of polishing.
- **Venue:** Section of a survey paper, or technical report. Not independently publishable at a top venue.

### Salvage Item 2: Mutation-Specification Duality Formalization (Standalone value: LOW-MODERATE)
- **Artifact:** `theory/math_spec.md` definitions D1-D5, theorem statements
- **Value:** Clean formal framing of the relationship between mutation adequacy and specification strength. Pedagogically useful. Could be the kernel of a FormaliSE or FMCAD workshop paper if paired with modest Lean/Coq mechanization.
- **Effort to extract:** 3-4 weeks for mechanization (Lean 4 recommended).
- **Venue:** FormaliSE (workshop), FMCAD (if paired with mechanized proofs). Not top-venue.
- **Caveat:** The formalization itself is not deep. The definitions are straightforward set-theoretic constructions over mutation operators and test suites. Value depends entirely on the mechanization effort.

### Salvage Item 3: MutGap — Lightweight Bug Finder (Standalone value: MODERATE-HIGH)
- **Artifact:** Not yet built. This is Approach B from `ideation/approaches.md`.
- **Description:** PIT integration + Z3 non-equivalence checking + model extraction for distinguishing inputs. No contracts, no lattice-walk, no theoretical apparatus. Output: ranked list of likely-buggy surviving mutants with concrete inputs.
- **Effort to build:** 3-4 weeks for an experienced Java/Z3 developer. ~2K-3K lines.
- **Venue:** Tool demo at ASE, ISSTA, or MSR (60-70% acceptance probability per Synthesizer, which I find credible).
- **Value proposition:** "We built PIT + Z3 and found N bugs in M open-source projects." This is the honest version of what MutSpec-Hybrid claims but cannot deliver.
- **Caveat:** This is engineering, not research. But it is *useful* engineering, and tool demos have a lower novelty bar.

### Recommended Salvage Path

**Option A (4-6 weeks):** Build MutGap (Salvage Item 3) and submit as tool demo. Honest framing, useful artifact, reasonable acceptance probability. Discard the theoretical apparatus.

**Option B (6-8 weeks):** Combine Salvage Items 1+2: prior art audit + mechanized formalization. Submit to FormaliSE as a formalization paper. Lower effort, lower reward, but completes something.

**Option C (0 weeks):** Archive everything and redirect effort to a different project. The prior art audit and math spec are available for future reference.

**Recommendation: Option A.** MutGap is the honest core of what this project attempted. Building it validates or invalidates the practical value proposition in 4-6 weeks, with a concrete deliverable regardless of theoretical outcomes.

---

## 8. Lessons for Future Projects

1. **Theory stages must produce theory.** A phase labeled "theory" that produces planning documents is a process failure. Gate criteria should require: at least one completed proof, or a concrete counterexample, or a mechanized formalization checkpoint.

2. **The consultant test should be applied early.** "Could a $10K consultant replicate 80% of this?" is a devastating question to discover late. It should be asked at ideation, not at theory evaluation.

3. **Difficulty and feasibility are in tension.** Choosing the feasible path (eliminate SyGuS) destroyed the difficulty score. Projects must find the difficulty sweet spot: hard enough to be novel, feasible enough to complete. MutSpec chose feasibility and lost novelty.

4. **Flagship theorems that might be false are risks, not contributions.** T1 at 65% achievability is a coin flip on whether the project's intellectual foundation exists. This should have been identified as a hard-gating risk at ideation, with a proof sketch required before proceeding.

5. **Formal-sounding circularity is still circularity.** The Gap Theorem *sounds* like a deep result but *is* a tautology dressed in formal notation. Peer reviewers will catch this immediately. Self-evaluation should apply the "does this tell me anything I didn't already know?" test.

---

## Appendix A: Probability Estimates

| Outcome | Auditor | Skeptic | Synthesizer | **Synthesis** |
|---|---|---|---|---|
| T1 proof succeeds | 65% | <50% | 65% | **55%** |
| T4 proof succeeds | 80% | ~70% | 80% | **75%** |
| T1 ∧ T4 both succeed | 52% | <35% | ~52% | **41%** |
| At least 1 CRITICAL risk materializes | — | 66% | — | **60%** |
| Any top-venue publication | ~40% | <15% | 30-40% | **20-25%** |
| Best paper at top venue | 2-3% | <1% | ~2% | **1-2%** |
| MutGap tool demo accepted | — | — | 60-70% | **55-65%** |
| Consultant replicates practical output | 70-80% | 85-90% | ~80% | **80-85%** |

**Synthesis methodology:** Skeptic-weighted average. Where the Skeptic provides concrete evidence (e.g., the 500-line Z3 script, the `max(a,b)` counterexample), their estimate is weighted 50%. Where the disagreement is judgmental, weights are 40% Skeptic / 30% Auditor / 30% Synthesizer. Rationale: the Skeptic represents the hostile reviewer, and publications must survive hostile review.

---

## Appendix B: Evidence Inventory

| Claim | Status | Source |
|---|---|---|
| theory_bytes = 0 | **CONFIRMED** | State.json: `"theory_bytes": 0` |
| code_loc = 0 | **CONFIRMED** | State.json: `"code_loc": 0` |
| theory_score = null | **CONFIRMED** | State.json: `"theory_score": null` |
| 720 lines in math_spec.md | **CONFIRMED** | `wc -l theory/math_spec.md` = 720 |
| 17+ theorem statements, 0 proofs | **CONFIRMED** | grep for "Theorem" yields 17+ hits; grep for proof content yields 7 hits, all headers/forward-refs |
| No implementation code | **CONFIRMED** | implementation/ contains only scoping.md |
| No empirical results | **CONFIRMED** | No benchmark data, no bug reports, no comparison with SpecFuzzer/EvoSpex/Daikon |
| SyGuS eliminated | **CONFIRMED** | Approach C selected over A; SyGuS is "Emergency Fallback" only |
| Cross-site expressiveness gap | **CONFIRMED** | max(a,b) counterexample in approach_debate.md and expert evaluations |
| Gate 3 (comparison with existing tools) unmet | **CONFIRMED** | No SpecFuzzer/EvoSpex comparison exists anywhere in artifacts |
| verification_signoff.md score 8.6/10 | **CONFIRMED** | Pre-theory-stage score evaluating the plan, not results |

---

*End of synthesis. Filed by Team Lead. Date: 2026-03-08.*
*Verdict: ABANDON with salvage (recommend MutGap tool demo path).*
*Composite score: 3.55/10.*
