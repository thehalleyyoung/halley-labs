# Community Expert Evaluation: MutSpec-Hybrid (mutation-contract-synth)

**Evaluator:** PL/FM Community Expert Panel (Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer, with Lead synthesis and Independent Verifier signoff)
**Date:** 2026-03-08
**Phase:** Post-theory verification (theory_complete, impl_loc=0, theory_bytes=0 in State.json, ~50KB theory content in theory/ directory)
**Prior Depth Check:** Composite 6.0/10, CONDITIONAL CONTINUE (2-1, Skeptic dissented ABANDON at 3.5)

---

## Executive Summary

Three independent expert evaluators assessed MutSpec-Hybrid across five dimensions, producing scores ranging from 3.2/10 (Skeptic: ABANDON) to 6.75/10 (Synthesizer: CONTINUE). After adversarial cross-critique, disagreement resolution, amendment evaluation, and independent verifier signoff, the panel converges on **CONDITIONAL CONTINUE at 5.2/10** — a downward revision from the prior depth check's 6.0/10, reflecting zero empirical validation at an advanced project stage. The honest expected outcome is an **ASE/ISSTA tool paper** (P ≈ 40–50%), with a 15–20% shot at OOPSLA and ≤10% at PLDI. The project must produce an executable prototype finding real bugs within 4 weeks or be killed.

---

## Panel Scores

| Dimension | Skeptic | Auditor | Synthesizer | **Synthesized** | Weight |
|-----------|---------|---------|-------------|-----------------|--------|
| Extreme Value | 3/10 | 5/10 | 7/10 | **5/10** | 25% |
| Genuine Software Difficulty | 4/10 | 5/10 | 7/10 | **5/10** | 20% |
| Best-Paper Potential | 2/10 | 4/10 | 6/10 | **4/10** | 20% |
| Laptop-CPU Feasibility | 5/10 | 7/10 | 8/10 | **7/10** | 15% |
| Feasibility | 3/10 | 5/10 | 6/10 | **5/10** | 20% |
| **Composite** | **3.2** | **5.1** | **6.75** | **5.15** | |

Weighted composite: (5 × 0.25) + (5 × 0.20) + (4 × 0.20) + (7 × 0.15) + (5 × 0.20) = 1.25 + 1.00 + 0.80 + 1.05 + 1.00 = **5.10**

---

## 1. Extreme Value — Score: 5/10

### What the Panel Agrees On

The mutation-specification duality — that killed mutants witness what tests enforce and survived mutants witness what tests permit — is a genuinely novel insight confirmed by the prior art audit (no prior system makes this duality constructive). The bug-finding reframe is the correct paper strategy: the primary deliverable is actionable bug reports, not JML annotations.

### What the Panel Disagrees On

**The demand problem is real and unresolved.** The Skeptic correctly identifies zero evidence of user demand — no user interviews, no feature requests, no forum threads. The proposal originates from a supply-side insight ("we have a clever way to extract specs from mutation data") rather than demand-side observation ("developers asked for better spec inference from their PIT runs"). The $10K consultant test is accepted by the project itself: a competent person with PIT + Daikon + Z3 + Python scripts replicates 70–80% of the practical output.

**The bug-finding reframe helps but doesn't solve the problem.** The Gap Theorem establishes that surviving non-equivalent mutants violating the inferred contract are formal bug witnesses with distinguishing inputs. However, zero bugs have been found. The entire value proposition is projection. The Synthesizer's killer-app scenario (financial services team finding boundary bugs in `TransferService.applyFee`) is compelling fiction until demonstrated.

**The marginal-cost-on-PIT argument is the strongest value play.** Teams already running PIT compute kill matrices. MutSpec adds WP differencing + gap analysis on the ~15% of mutants that survive — the expensive operations run on O(n) dominators, not O(nk) raw mutants. If wall-clock overhead is 1–3 hours on 8 cores, this fits a CI window. But this arithmetic is unvalidated.

**LLM competition is real but not immediately fatal.** LLMs generate specs from code context, not from test behavior. MutSpec's signal — what the test suite *actually enforces* — is fundamentally unavailable to an LLM without test-execution feedback. AutoSpec (CAV 2024) demonstrates LLM + verifier feedback loops; MutSpec's deterministic, reproducible output remains a structural advantage for CI/CD. However, the window for this advantage is narrowing.

**Score rationale:** Genuine insight, defensible bug-finding reframe, plausible narrow adopter niches (PIT users, verification bootstrapping) — but entirely hypothetical. Zero demonstrated demand, zero bugs found, 70–80% replicable by a consultant.

---

## 2. Genuine Software Difficulty — Score: 5/10

### Novel Core After SyGuS Removal: ~16K LoC

The final approach eliminated SyGuS in favor of the lattice-walk algorithm. This was the correct engineering decision (eliminates fatal scalability risk), but it also removed the most technically impressive novel component. The remaining 16K novel LoC:

| Component | LoC | Novelty Assessment |
|-----------|-----|--------------------|
| Source-level mutation replay (PIT ↔ symbolic bridge) | 3,500 | Engineering — mapping bytecode mutations to AST transformations |
| Batch WP differencing with incremental Z3 sharing | 3,500 | **Genuine systems novelty** — no prior tool computes WP differences for hundreds of mutants with shared encodings |
| Lattice-walk contract synthesis | 4,000 | **Novel algorithm** — but the Skeptic's characterization ("a loop over SMT queries with entailment-based redundancy removal") is not entirely unfair |
| Entailment-based simplification | 2,500 | Standard SMT preprocessing applied to a new setting |
| Gap analysis engine | 2,000 | Z3-based equivalence checking + distinguishing input generation — competent integration |

### The Difficulty Debate

The Skeptic calls this "PIT + Z3 wrapper" and estimates a 6-month PhD project. The Synthesizer argues the batch WP differencing and full-pipeline integration ("making formal methods disappear into a developer tool") are underappreciated. The Auditor lands in between: one genuinely novel research problem (lattice-walk synthesis from WP differences) plus competent integration — harder than a wrapper, easier than a paradigm shift.

**Comparison to best-paper artifacts is instructive:**
- **KLEE:** ~33K novel LoC solving multiple open problems (memory modeling, constraint optimization, search heuristics). Found bugs in GNU Coreutils.
- **CompCert:** ~90K Coq + ~20K OCaml. Verified entire C compiler.
- **Alive:** ~10K LoC, but proved correctness of 334 LLVM optimizations. Found 8 bugs in LLVM.

MutSpec's 16K novel LoC is in Alive's range, but Alive had devastating empirical results. MutSpec has zero.

**The SyGuS removal paradox:** The depth check rated SyGuS encoding of mutation boundaries as the only genuinely novel research problem. Removing SyGuS was correct for feasibility but reduced both difficulty and novelty. The lattice-walk is elegant but deterministic — it doesn't require solving any open problem.

**Score rationale:** The batch WP differencing engine and lattice-walk are genuine contributions. The PIT↔symbolic bridge is hard engineering. But 16K novel LoC with zero open problems solved is below the bar for top-venue difficulty claims.

---

## 3. Best-Paper Potential — Score: 4/10

### Theorem 1's Restrictions Are Severe

The crown jewel — ε-completeness of {AOR, ROR, LCR, UOI} for QF-LIA over loop-free code — is a real theorem with genuine technical content. The operator exhaustiveness lemma, piecewise-linear region partitioning, and site-independence analysis are non-trivial.

However, the restrictions are crippling for best-paper consideration:

| Restriction | Impact |
|-------------|--------|
| Loop-free only | Excludes majority of real Java methods |
| QF-LIA only | No strings, heap, floating-point, nonlinear arithmetic |
| Four operators only | {AOR, ROR, LCR, UOI} — excludes method-call, exception, object mutations |
| First-order mutants only | No higher-order interactions |
| Requires 100% mutation adequacy | Real suites achieve ~70%; Theorem 1 is an asymptotic ceiling |
| Site-independence assumption | The Skeptic's `clamp()` counterexample shows multi-site properties fail |

The Auditor estimates Theorem 1 covers **10–20% of real functions** in a typical Java codebase. Even with the three-tier degradation (Tiers 2–3 handling the rest), the formal guarantee applies to a narrow fragment. The "lighthouse theorem" framing — prove it for a clean fragment, motivate heuristic generalization — is legitimate in PL, but requires compelling empirical evidence of generalization. That evidence does not exist.

### The Gap Theorem Is Closer to a Definition Than a Theorem

The panel converges: the Gap Theorem ("every surviving mutant is either equivalent or a specification gap") is a trichotomy by exhaustion. The math_spec.md itself classifies it as "Framework definition" with "straightforward" difficulty. **Rename to "Gap Characterization" in the paper.** The practical value is in the machinery (concrete witnesses, inferred postcondition context, distinguishing inputs), not the logical statement.

### The "Between Chairs" Problem

- **Too theoretical for ICSE:** No user study, formal proofs, SyGuS/lattice-walk machinery.
- **Too restricted for POPL:** QF-LIA/loop-free theorem with no Galois connection deep enough for POPL reviewers.
- **Too incomplete for PLDI tools track:** Zero empirical evidence, no scalability data.

The three-legged stool (theorem + tool + bugs) is the correct positioning — but all three legs are currently wobbly. Theorem 1 at 65% achievability, zero bugs found, tool not built.

### The "40 Years of Disconnected Fields" Narrative Is Overstated

FormaliSE 2021, SpecFuzzer (ICSE 2022), EvoSpex (ICSE 2021), IronSpec (OSDI 2024), and AutoSpec (CAV 2024) all connect mutation to specifications in some form. The fields are under-connected, not disconnected. The honest framing: "No prior work formalizes the constructive duality, though several systems exploit heuristic connections."

### Reviewer Expectations

**PLDI reviewers** increasingly expect mechanized proofs (Coq, Lean, Isabelle) for algorithm-correctness claims. Pen-and-paper proofs of Theorems 1 and 4 will draw skepticism. The paper should either mechanize the core lemma or explicitly scope claims as pen-and-paper with mechanization as future work.

**The "Daikon + PIT with extra steps" kill-sentence** is the single most dangerous reviewer line. The defense is real (different signal: test enforcement vs. execution observation; mutation provenance; completeness guarantee) but only credible with empirical evidence showing measurably better contracts.

**Score rationale:** Genuinely novel core idea nowhere near best-paper readiness. P(best paper at PLDI) ≈ 3–5%. P(best paper at OOPSLA) ≈ 5–8%.

---

## 4. Laptop-CPU Feasibility & No Humans — Score: 7/10

### Clean Computational Profile

The pipeline is purely CPU-bound: PIT (JVM), Z3 (C++), lattice-walk (Java/SMT), SARIF generation. No GPUs, no cloud, no API calls, no training loops, no human-in-the-loop. All three panelists confirm this.

### Wall-Clock Estimates

| Scenario | Estimate | Source |
|----------|----------|--------|
| 10K functions, 8 cores | 2–4 hours | Auditor (optimistic) |
| 50K functions, 8 cores | 8–17 hours | Range across panelists |
| Changed-function CI mode | <30 minutes per commit | Synthesizer |
| Serial worst-case, 50K functions | ~43 days | Skeptic |

The realistic scenario for evaluation (200–500 functions from Defects4J + Apache Commons Math) is comfortable: 30–90 minutes on commodity hardware.

### Unresolved: QF-LIA vs QF-BV Soundness Boundary

Java integers are 32-bit twos-complement; QF-LIA models unbounded integers. MutSpec contracts are **unsound** near overflow boundaries (e.g., `Integer.MAX_VALUE + 1`). For a tool marketing "formally grounded" bug detection, this is a credibility risk. The paper must document the soundness boundary and report the fraction of target functions where QF-LIA diverges from Java semantics.

**Score rationale:** Genuinely clean CPU-only profile. The incremental/CI mode makes production viability realistic. Main concern is unvalidated scalability arithmetic and the QF-LIA/QF-BV soundness gap.

---

## 5. Feasibility — Score: 5/10

### Publication Probability Estimates (Panel Consensus)

| Outcome | Skeptic | Auditor | Synthesizer | **Panel** |
|---------|---------|---------|-------------|-----------|
| PLDI | 3% | 10–15% | 20–25% | **8–12%** |
| OOPSLA/FSE | 8% | 20–25% | 35–40% | **18–25%** |
| ASE/ISSTA | 15% | 35–40% | 55–60% | **40–50%** |
| FormaliSE | 25% | 55–60% | 70–75% | **60–70%** |
| No publication | 60% | 15–20% | 5–10% | **15–25%** |

### The Skeptic's Compounding-Probability Argument

The Skeptic's claim: the project's own failure probabilities (Theorem 1 fails: 35%, WP engine overruns: 25%, bug yield low: 30%, FP too high: 25%, SpecFuzzer matches: 20%) compound to ~80% chance of at least one critical failure, yielding P(PLDI) ≈ 3%.

**This is directionally correct but methodologically flawed.** The risks are not independent (if Theorem 1 succeeds, WP engine difficulty drops; if WP works, bug yield improves). And not all failures are fatal — the three-tier architecture provides degradation paths. Corrected estimate: P(PLDI) ≈ 8–12%. The expected outcome is an ASE/ISSTA paper.

### Top 3 Execution Risks

1. **Theorem 1 fails or requires site-independence premise covering <30% of functions (P=35%).** Project loses its intellectual core. Fallback: tool paper without theoretical claims.
2. **Bug yield underwhelming — <10 real bugs across all benchmarks (P=30%).** Three-legged stool loses empirical leg. Fallback: theory + prototype paper at FormaliSE.
3. **WP engine for Java QF-LIA fragment is 2x harder than budgeted (P=25%).** Java's autoboxing, null semantics, and widening conversions leak into every expression. JBMC's encoding is ~15K LoC of C++; even the restricted fragment is non-trivial.

### The theory_bytes=0 Anomaly

State.json records theory_bytes=0 despite ~50KB of mathematical specification in theory/. This is either a pipeline measurement bug or evidence that no formalized proofs exist (the 50KB is planning, not completed proofs). Either way, it is a process flag. Gate 2 requires theory_bytes > 0.

**Score rationale:** Well-designed phased plan with gates, but everything is unvalidated. 22-week timeline from zero code is aggressive. Realistic scope is 15–20K LoC, not 48K.

---

## 6. Fatal Flaws

### Flaw 1: Zero Empirical Validation (ALL PANELISTS AGREE — HIGHEST PRIORITY)

The project is at theory_complete phase with:
- Zero lines of code written
- Zero bugs found
- Zero functions analyzed
- Zero SpecFuzzer comparisons run
- Zero lattice-walk feasibility data

150KB of planning documents and 0 bytes of executable artifact. The empirical hypothesis — that mutation gaps correspond to real bugs in real codebases — has never been tested. This is the existential vulnerability.

### Flaw 2: Site-Independence Breaks Theorem 1 for Non-Trivial Functions

The Skeptic's `clamp(x, lo, hi)` counterexample is concrete: the postcondition `result >= lo && result <= hi` is a multi-site property that no single-site mutation can witness. The achievability estimate (65%) is aspirational; with site-independence issues, realistic achievability for the clean theorem is 45–55%. A version with additional premises (site-independence) is achievable at ~75% but is a weaker result.

### Flaw 3: Equivalent Mutant FP Rate Understated

The project estimates 8–12% FP after multi-layer filtering. The panel consensus: **10–20% is more realistic**, depending on the project. For arithmetic-heavy code (Commons Math), closer to 10%. For OOP-heavy code (Jackson), closer to 20%. The "formally grounded" framing sets higher expectations than the tool can deliver — developers who see "specification gap" and investigate to find an equivalent mutant will compare MutSpec to a liar, not to SpotBugs.

### Flaw 4: SpecFuzzer Differentiation Is Narrower Than Claimed

After the SyGuS→lattice-walk pivot, MutSpec's core algorithm is: compute error predicates for killed mutants, conjoin their negations, remove redundancies. SpecFuzzer: generate candidates from a grammar, filter by mutation consistency, rank. Both use mutation data to shape specifications. The architectural difference is real (construction vs. filtering), but the user-visible output may converge. **The mandatory SpecFuzzer + Z3 comparison (Gate 3 equivalent) will resolve this empirically.**

### Flaw 5: Mutation Adequacy Assumption Is Vacuous in Practice

Theorem 1 requires mutation-adequate test suites (100% of killable mutants killed). Real suites achieve ~70% mutation scores. The degradation bound conjecture ("a suite with score *s* captures ≥g(s) fraction of specification") is acknowledged as "almost certainly false as stated." Without a degradation bound, Theorem 1 is an asymptotic ceiling result: "if you had perfect tests, you'd get perfect specs." Practical test suites get an unknown fraction.

---

## 7. Amendments Evaluated

| # | Amendment | Verdict | Rationale |
|---|-----------|---------|-----------|
| 1 | MutGap-Lite as Gate 0 (early empirical validation) | **ADOPT** | Single most important amendment. Provides first empirical signal within 4 weeks. All panelists converge. |
| 2 | Bug-finding reframe as paper strategy | **ADOPT** | Already consensus. Lead with bugs, not theorems. "Your tests specify less than you think." |
| 3 | Three-tier degradation as engineering contribution | **PARTIAL** | Present as robustness mechanism, not headline contribution. Tier 2 is incremental over SpecFuzzer; Tier 3 is Daikon. |
| 4 | CI/CD spec-drift monitoring | **DEFER** | Speculative product vision, not paper contribution. Mention in future work. |
| 5 | Distributional uniformity for sub-adequate suites | **REJECT** | Assumption-stacking without validation. The honest response to site-independence is: characterize the fragment, report coverage, let Tiers 2–3 handle the rest. |

---

## 8. Binding Conditions

### Gate 0 — MutGap-Lite Prototype (Week 4 hard deadline)

**Deliverable:** Concrete-execution prototype using PIT + Daikon + surviving-mutant gap analysis (no WP engine, no lattice-walk).

**Pass criteria:**
- ≥3 specification gaps confirmed by matching known Defects4J faults or by manual inspection producing a concrete failing test, across ≥2 Defects4J subjects
- False positive rate ≤30% (honestly measured before filtering)

**Fail → ABANDON.** No extensions. The empirical hypothesis must be validated or the project has no foundation.

### Gate 1 — WP Differencing Engine + Lattice-Walk Feasibility (Week 8)

**Deliverable:** Symbolic WP computation for ≥50 functions from Apache Commons Math.

**Pass criteria:**
- Error predicates match concrete execution on ≥90% of tested inputs
- Batch WP sharing demonstrates ≥3x speedup over naive per-mutant computation
- Lattice-walk produces contracts for ≥40% of loop-free QF-LIA functions with ≥1 conjunct not derivable from Daikon templates
- QF-LIA soundness boundary for integer overflow documented

**Fail → DESCOPE** to MutGap-Lite tool paper (ASE target). Do not attempt lattice-walk.

### Gate 2 — Theorem 1 Formalization (Week 10)

**Deliverable:** Complete proof of ε-completeness for single-site QF-LIA atoms (not conjunctions).

**Pass criteria:**
- theory_bytes > 0 (formalized in the project's theory directory)
- Honest characterization of site-independence failure modes with empirical coverage estimate

**Fail → PIVOT** to empirical paper without theoretical claims. Target FSE/ASE.

### Gate 3 — Bug Finding + Baseline Comparison (Week 14)

**Deliverable:** ≥10 genuine specification gaps across Defects4J + Apache Commons Math, with ≥5 confirmed as real bugs (developer-acknowledged or matching known Defects4J faults).

**Pass criteria:**
- MutSpec output measurably superior to SpecFuzzer + Z3 on ≥1 dimension
- MutSpec output measurably superior to Daikon on ≥50% of target functions
- Secondary comparison against GPT-4 + test-feedback baseline on same benchmarks

**Fail → DESCOPE** to theory + prototype paper (OOPSLA/FSE at best, FormaliSE likely).

---

## 9. Community Expert Perspective: How Would Reviewers React?

### PLDI (P(accept) ≈ 8–12%)

**Likely reaction: Mixed-to-negative.** PLDI values theoretical depth with strong mechanization or empirical validation. A narrow theorem over loop-free QF-LIA without mechanized proof, plus a tool that degrades to Daikon-quality on 40–75% of real functions, would face skepticism. The "Daikon + PIT with extra steps" reviewer sentence is the kill shot to avoid.

### OOPSLA (P(accept) ≈ 18–25%)

**Likely reaction: Cautiously positive if bugs are found.** OOPSLA's artifact evaluation track rewards "making formal methods practical." The bug-finding angle, three-tier degradation, and CI/CD story play well. Mandatory: head-to-head comparison against SpecFuzzer and Daikon.

### ASE/ISSTA (P(accept) ≈ 40–50%)

**Likely reaction: Positive if the tool works.** This is the natural home. Bug-finding tools with empirical validation are ASE/ISSTA's bread and butter. MutGap-Lite alone could produce a solid paper here.

### FormaliSE (P(accept) ≈ 60–70%)

**Likely reaction: Positive.** The mutation-specification duality formalization alone is a FormaliSE contribution. This is the guaranteed floor.

---

## 10. Scope Compression

### Drop from Plan
- Multi-language generalization (confirm stays deferred)
- Full 48K LoC target → target 15–20K LoC
- CI/CD integration layer (future work)
- Tier 3 custom Daikon integration (use Daikon output directly)
- Certificate generation chain (multi-year project, defer)

### Elevate in Plan
- MutGap-Lite prototype (highest priority, Gate 0)
- WP differencing for QF-LIA fragment (core novelty)
- Bug-finding evaluation on Defects4J (headline empirical result)
- SpecFuzzer + Daikon baseline comparison (mandatory, non-negotiable)

---

## 11. Salvage Value (if ABANDONED)

| Artifact | Value | P(publication) | Venue |
|----------|-------|----------------|-------|
| FormaliSE theory paper (Theorem 1 + duality formalization) | Establishes priority on the mutation-specification duality | ~60% | FormaliSE |
| MutGap-Lite standalone tool (~10–16K LoC) | Bug-finding from PIT data, no formal machinery | ~50% | ASE/ISSTA demo track |
| WP differencing engine (reusable component) | Batch WP computation with incremental Z3 | Reusable in other projects | — |
| Negative result on WP-Composition Completeness | Precise characterization of when WP suffices | Workshop-level | FormaliSE/FMCAD |

---

## 12. Score Reconciliation

| Dimension | Prior Depth Check | This Evaluation | Change | Reason |
|-----------|-------------------|-----------------|--------|--------|
| Value | 6/10 | 5/10 | −1 | Bug-finding reframe accepted but zero empirical validation at a later project stage |
| Difficulty | 6/10 | 5/10 | −1 | SyGuS removal eliminated the hardest novel component; lattice-walk is simpler |
| Best-Paper | 5/10 | 4/10 | −1 | Zero empirical results, "between chairs" problem unresolved, all three legs wobbly |
| Laptop-CPU | 8/10 | 7/10 | −1 | QF-LIA/QF-BV soundness gap; scalability unvalidated; lattice-walk has no validation gate |
| Feasibility | 6/10 | 5/10 | −1 | 22 weeks from zero code is aggressive; all critical assumptions unvalidated |
| **Composite** | **6.0** | **5.15** | **−0.85** | **Downward revision reflects zero artifacts at theory-complete** |

---

## VERDICT: CONDITIONAL CONTINUE

**Composite Score: 5.15/10**

**Decision: CONDITIONAL CONTINUE with compressed scope and hard kill-gates.**

The mutation-specification duality is a genuinely novel insight that deserves empirical validation. It does not deserve six more months of planning documents. Gate 0 (MutGap-Lite finding real bugs in 4 weeks) is the existential test. If mutation gaps correspond to real bugs in real codebases, the project has a future. If they don't, no amount of theory will save it.

**The clock starts now. Gate 0 in 4 weeks or kill.**

### What the Skeptic Got Right
- Gap Theorem is definitional, not a deep theorem — rename to "Gap Characterization"
- Site-independence limits Theorem 1's practical reach to ~15–25% of real functions
- Zero artifacts at theory-complete is a red flag
- P(PLDI) < 15% is honest
- The compounding-probability argument is directionally correct

### What the Skeptic Got Wrong
- ABANDON is premature before the empirical hypothesis has been tested
- The lattice-walk is more than "a loop over SMT queries" (dominator sets, subsumption, irredundancy)
- "Daikon + PIT with extra steps" ignores the formally different signal (test enforcement vs. execution observation)
- P(PLDI) ≈ 3% understates due to independence assumption on correlated risks

### What the Synthesizer Got Right
- Bug-finding reframe is the correct paper strategy
- MutGap-Lite as Gate 0 is the single most important amendment
- Batch WP differencing with incremental Z3 is underappreciated novel systems work
- The project has a clear floor (ASE) and high ceiling (PLDI)

### What the Synthesizer Got Wrong
- Composite 7.3/10 is overgenerous given zero artifacts
- Distributional uniformity is assumption-stacking, not a fix
- CI/CD monitoring is speculative product vision, not a paper contribution
- P(PLDI) ≈ 20–25% is aspirational; 8–12% is honest

---

## Panel Sign-Off

| Role | Score | Verdict | Sign-Off |
|------|-------|---------|----------|
| Independent Auditor | 5.1/10 | CONDITIONAL CONTINUE | ✓ |
| Fail-Fast Skeptic | 3.2/10 | ABANDON | ✗ (dissents; accepts Gate 0 as compromise) |
| Scavenging Synthesizer | 6.75/10 | CONTINUE | ✓ (accepts compressed scope) |
| **Lead (Synthesized)** | **5.15/10** | **CONDITIONAL CONTINUE** | **✓** |
| Independent Verifier | — | APPROVE WITH CHANGES | ✓ (six minor changes incorporated) |

**Final vote: 3-1 CONDITIONAL CONTINUE (Skeptic dissents ABANDON but accepts Gate 0 kill-deadline as compromise).**

---

*Evaluated by PL/FM Community Expert Panel. All scores reflect post-theory, pre-implementation status with zero empirical validation. Scores are expected to shift significantly (upward if Gate 0 passes, downward to ABANDON if Gate 0 fails) within 4 weeks.*
