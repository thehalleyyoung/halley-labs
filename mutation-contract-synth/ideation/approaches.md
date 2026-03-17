# MutSpec: Three Competing Approaches

**Project:** `mutation-contract-synth`
**Date:** 2025-07-18
**Context:** Composite score 6.0/10 · CONDITIONAL CONTINUE (2-1) · Java-only 65K LoC scope · Bug-finding reframe · Five binding gates

---

## Approach A: MutSpec-Complete — Full Mutation-Directed SyGuS with Verified Certificates

**One-line pitch:** Prove that mutation-adequate test suites determine unique specifications, then build a certified SyGuS engine that extracts them.

### 1. Extreme Value and Adopters

**Value delivered:** The first tool that takes a Java codebase with tests and emits *SMT-verified* contracts with machine-checkable certificates, plus formally grounded bug reports where every flagged surviving mutant comes with a proof witness. The certificate means downstream tools (KeY, CBMC, OpenJML) can consume MutSpec output without trusting MutSpec's internals — they verify the certificate independently.

**Who desperately needs this:**
- **Verification tool builders** (KeY team at KIT, OpenJML developers, Frama-C/ACSL consumers) who need bootstrap contracts to start modular verification. Today they require weeks of manual annotation per module. MutSpec-Complete gives them verified starting points for free.
- **Safety-critical Java shops** (avionics middleware on ARINC 653, automotive ECU stacks in Java, medical device firmware using Java Card) where regulatory frameworks (DO-178C, IEC 62304) require evidence of specification coverage. MutSpec certificates count as machine-checkable evidence.
- **Google's mutation testing infrastructure** (processes millions of mutants/day via their internal PIT variant), which currently extracts only a scalar mutation score. MutSpec-Complete recovers structured specification data from the same kill matrices at marginal cost.

### 2. Why This Is Genuinely Hard

**Hard subproblem 1 — Mutation-directed grammar construction has no precedent.** SyGuS grammars are typically hand-authored or derived from type signatures. Constructing a grammar from mutation error predicates — where each terminal is a WP-differencing result and the grammar must be simultaneously expressive enough to capture all mutation-derivable contracts and restrictive enough for CVC5 to converge — is an open design problem. The grammar's structure directly determines whether synthesis takes seconds or diverges.

**Hard subproblem 2 — CEGIS with program counterexamples.** Standard CEGIS refines candidates using *input* counterexamples (points in the input space). MutSpec-Complete uses *surviving mutant programs* as counterexamples — structurally different objects. The counterexample must be translated into a SyGuS constraint that tightens the grammar, requiring a feedback loop between program-level mutation semantics and formula-level SyGuS constraints. The convergence properties of this loop are unknown and may differ qualitatively from standard CEGIS.

**Hard subproblem 3 — Certificate generation for the full reasoning chain.** The certificate must witness: mutation data extraction → grammar construction → SyGuS solution → SMT verification. No existing contract inference tool produces certificates. The certificate format must be compact enough for practical use yet rich enough for independent verification — a non-trivial design problem when the underlying reasoning involves both SyGuS synthesis and SMT solving.

**Architectural challenge:** Three heavyweight symbolic reasoning systems (PIT mutation engine, CVC5 SyGuS solver, Z3 SMT verifier) must share a common intermediate representation and communicate constraints across fundamentally different abstraction levels (bytecode mutations ↔ logical formulas ↔ grammar terms). Integration bugs at these boundaries are the primary engineering risk.

### 3. New Math Required

**Theorem 3 Extension (QF-LIA + bounded loops).** Extend the crown-jewel completeness result from loop-free to bounded-loop code by proving ε-completeness for {AOR, ROR, LCR, UOI} operators over programs with loops bounded by a statically determinable constant *k*. The proof technique: unroll loops to depth *k*, apply the loop-free result to the unrolled program, and show that the specification gap between the unrolled and original program is bounded by a function of *k* and the loop body's mutation profile. This is load-bearing because it determines what fraction of real functions receive the formal guarantee (Auditor estimates 10–20% loop-free → potentially 40–50% with bounded loops).

**Quantitative degradation bound.** Prove that a test suite with mutation score *s* (fraction of killable mutants killed) determines a specification capturing at least *g(s)* fraction of the strongest mutation-derivable specification, where *g* is monotone and *g(1) = 1*. Conjecture: *g(s) ≥ s* (linear degradation). This transforms Theorem 3 from an all-or-nothing result requiring 100% mutation adequacy into a smooth function applicable to real suites (typically 60–85% mutation score). Load-bearing because without it, Theorem 3's precondition is vacuously unsatisfied on every real codebase.

**Certificate soundness theorem.** Prove that any contract accepted by the certificate verifier is (a) satisfied by the original program on all inputs within the bounded verification scope and (b) violated by every killed mutant on at least one input. This is the formal warrant for trusting MutSpec output without trusting MutSpec's implementation.

### 4. Best-Paper Potential

A PLDI committee would select this because it delivers the "three-legged stool" in its strongest form:
- **Leg 1 (Surprise):** Theorem 3 extended to bounded loops — the first formal proof that mutation testing determines specifications for a practically relevant fragment, contradicting the intuition that mutation kills are "too coarse" for specification inference.
- **Leg 2 (Mechanism):** The Gap Theorem with certificates — not just "we found bugs" but "here is a machine-checkable proof that this is a bug, verifiable independently of our tool."
- **Leg 3 (Impact):** Real bugs found in Apache Commons Math / Guava with certified witnesses, filed as bug reports with maintainer confirmation.

The certificate angle is the differentiator from prior specification inference work: Daikon produces "likely" invariants, SpecFuzzer produces unverified assertions, LLMs produce plausible guesses. MutSpec-Complete produces *certified* contracts. This distinction matters for safety-critical adoption and gives reviewers a concrete technical advance to point to.

### 5. Hardest Technical Challenge and Mitigation

**Challenge:** SyGuS scalability on mutation-derived grammars is completely unvalidated. CVC5 is reliable on ≤15 atoms and ≤50 constraints (SyGuS-Comp data). A typical 200-line function with 20 mutation sites × 5 operators yields ~100 raw data points; even after 90% subsumption reduction, this produces 20–30 atoms — at CVC5's reliability boundary. If the per-site decomposition strategy (treating each mutation site as an independent sub-problem, composing via conjunction) fails, synthesis degrades from seconds to timeouts.

**Mitigation:** The three-tier synthesis strategy provides guaranteed output regardless of CVC5 performance: Tier 1 (full SyGuS, target 80%) → Tier 2 (coarsened grammar, target 15%) → Tier 3 (Daikon-style template fallback, target 5%). All tiers produce SMT-verified output. Additionally, the 4-week Gate 1 feasibility study on 50 Apache Commons Math functions provides early kill signal: if solve rate < 50%, the project pivots to Approach B or C rather than sinking 12 weeks into implementation.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8/10 | Certified contracts fill a real gap for verification tool builders and safety-critical shops; bug reports with proof witnesses are uniquely compelling |
| **Difficulty** | 8/10 | Three novel subproblems (grammar construction, CEGIS-with-programs, certificate generation) plus integration of three heavyweight systems |
| **Potential** | 7/10 | Three-legged stool with certificates is strong positioning; Theorem 3 extension would be genuinely surprising; risk is restricted scope of formal results |
| **Feasibility** | 5/10 | SyGuS scalability is the Achilles heel; bounded-loop extension may be harder than projected; 24-week timeline is tight for all three novel subproblems |

---

## Approach B: MutGap — Lightweight Mutation-Guided Bug Finder

**One-line pitch:** Skip SyGuS entirely, use fast template-based specification inference with mutation-boundary filtering, and focus every engineering hour on maximizing bugs found per CPU hour.

### 1. Extreme Value and Adopters

**Value delivered:** A mutation-testing add-on that plugs into existing PIT/Stryker pipelines and transforms surviving mutants into ranked, actionable bug reports — each with a concrete distinguishing input and a human-readable explanation of what specification the test suite fails to enforce. No formal contracts unless you ask for them; the default output is a bug list sorted by confidence.

**Who desperately needs this:**
- **Any team already running PIT** (10,000+ monthly downloads, integrated into Maven/Gradle at major enterprises). Today they get a scalar mutation score and a list of surviving mutants. MutGap tells them *which* surviving mutants are bugs, *why*, and gives them a test case that triggers the bug. The pitch is: "Run `mvn mutgap:report` after your existing `mvn pitest:mutationCoverage` and get a bug list for free."
- **Open-source maintainers of high-assurance libraries** (Apache Commons Math, Guava, Bouncy Castle, BouncyCastle) who already invest heavily in testing but have no formal specs. MutGap surfaces the bugs their tests miss without asking them to learn JML.
- **CI/CD pipeline owners** who want "one more check" that catches bugs conventional testing misses. MutGap runs as a Maven plugin, produces a SARIF report, and integrates with GitHub Code Scanning / SonarQube / CodeQL dashboards. Zero new toolchain to adopt.

### 2. Why This Is Genuinely Hard

**Hard subproblem 1 — Equivalent mutant triage at scale.** The Gap Theorem says every surviving non-equivalent mutant violating the inferred spec is a bug. But 5–25% of survivors are equivalent (undecidable in general). MutGap must achieve ≤10% false positive rate to be usable, requiring a multi-layered filtering stack: Trivial Compiler Equivalence (bytecode identity), symbolic equivalence via bounded SMT, heuristic detectors (trivially dead code, idempotent mutations), and statistical confidence scoring from test-execution traces. Building a filtering stack that is simultaneously sound (never filters a real bug), effective (catches most equivalents), and fast (runs within CI window) is a research-grade filtering problem.

**Hard subproblem 2 — Distinguishing input generation for surviving mutants.** For each flagged surviving mutant, MutGap must produce a concrete input that causes the mutant to behave differently from the original — a *witness* to the specification gap. This requires directed symbolic execution or constraint solving to find inputs that satisfy the mutation error predicate. For non-trivial functions (complex control flow, object-heavy inputs), witness generation is as hard as test generation, and naive approaches produce trivial inputs that don't demonstrate the bug clearly.

**Hard subproblem 3 — Template-based spec inference that isn't just Daikon.** MutGap uses Daikon-style template matching as its synthesis backend, but templates are filtered and ranked by mutation-boundary consistency. The hard part: designing a template set that is expressive enough to catch non-trivial specification violations yet restrictive enough that the combinatorial explosion of template instantiation remains tractable. If the templates are too weak, MutGap degenerates into "Daikon + PIT" — adding nothing. If too expressive, inference becomes as slow as SyGuS.

**Architectural challenge:** The entire tool must be fast enough to run as a CI step. The budget is minutes, not hours. This constrains every design decision: no heavyweight SyGuS solving, no full WP computation, no certificate generation. The engineering discipline of *not* building the theoretically optimal system in favor of the practically fast one is itself a design challenge.

### 3. New Math Required

**Gap Theorem with confidence quantification.** Extend Theorem 4 to assign a *confidence score* to each gap report, defined as:

*conf(m) = P(m is non-equivalent | observable features of m)*

where observable features include: (a) TCE result (bytecode identity), (b) number of test cases that nearly distinguish m (coverage proximity), (c) syntactic distance of the mutation from the nearest assertion in the test suite, (d) historical false-positive rate for the mutation operator type. Prove that the confidence score is *calibrated* — i.e., among gap reports with confidence *c*, approximately fraction *c* are genuine bugs. This is load-bearing because it determines the ranking order of bug reports. Developers will look at the top 10 reports; if those 10 are all false positives, the tool is dead.

**Compositional gap analysis.** Prove that if function *g* calls function *f*, and MutGap infers a contract for *f* with confidence *c_f*, then gap analysis at *g*'s call site to *f* inherits confidence *c_f · c_g* (multiplicative composition). This enables *interprocedural* gap analysis without re-analyzing callee functions, and the confidence decay quantifies how quickly trust degrades across call chains. Load-bearing because most real bugs involve interactions between functions, not single-function errors.

**Subsumption-aware filtering theorem.** Prove that equivalent-mutant filtering applied to dominator sets (Theorem 5) preserves the gap analysis — i.e., filtering a dominator does not cause a non-dominator to "inherit" a false gap report. This is load-bearing because MutGap's scalability depends on analyzing only dominators, and unsound filtering would produce both false positives and false negatives.

### 4. Best-Paper Potential

A PLDI committee would select this if MutGap finds *a lot* of real bugs. The paper's argument:
- "We built a tool that takes 30 seconds on top of your existing PIT run and found 47 previously unknown bugs across 8 maintained Java libraries, including 12 confirmed by maintainers."
- The theory (Gap Theorem, confidence calibration) is clean but secondary — it's the *engine* that produces the bugs, not the *pitch*.
- The positioning against existing bug-finding tools (SpotBugs, Error Prone, Infer) is concrete: "MutGap finds bugs in well-tested code that SpotBugs and Infer miss, because it exploits mutation data that these tools ignore."
- Strong ablation study showing that mutation-boundary filtering is the key ingredient: remove it and the bug-finding rate drops 60%.

The risk: if MutGap finds only 5–10 bugs, the paper is an incremental tool contribution (FSE/ASE, not PLDI). The upside: if MutGap finds 30+, with maintainer confirmations, it's a potential best paper because it *demonstrably changes practice* for a large user base.

### 5. Hardest Technical Challenge and Mitigation

**Challenge:** Equivalent mutant filtering must achieve ≤10% FP rate to be usable, but the problem is undecidable in general. TCE catches only 30–40% of equivalents; symbolic equivalence adds another 20–30%; but 30–40% of equivalent mutants remain undetectable by any automated method. If the residual FP rate exceeds 15%, developers will distrust MutGap's reports.

**Mitigation:** Multi-pronged: (1) TCE + symbolic filtering as baseline. (2) Statistical confidence scoring learned from a labeled dataset of equivalent/non-equivalent mutants (train on Defects4J where ground truth is available, apply to new codebases). (3) Conservative reporting: only flag mutants with confidence ≥ 0.8, accepting lower recall for higher precision. (4) SARIF integration with "likely equivalent" tags so developers can configure their own threshold. The confidence calibration theorem ensures the thresholding is principled, not ad hoc.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9/10 | Directly solves the "what do I do with surviving mutants?" pain point for 10K+ PIT users; Maven plugin integration means zero adoption friction |
| **Difficulty** | 5/10 | No SyGuS, no certificates, no WP differencing; primary novelty is the filtering stack and confidence calibration; template inference is well-understood |
| **Potential** | 6/10 | Best-paper requires a large bug count; the theory (confidence calibration, compositional gap analysis) is solid but not deep enough for POPL; sweet spot is PLDI/OOPSLA tool track or top-tier FSE |
| **Feasibility** | 9/10 | ~20K LoC; no SyGuS scalability risk; template inference always terminates; 12-week build is realistic; PIT integration is well-documented |

---

## Approach C: MutSpec-Δ — Compositional Contract Construction via WP Differencing

**One-line pitch:** Bypass SyGuS entirely by directly composing contracts from weakest-precondition differences between original and mutant programs, using a novel lattice-walk algorithm.

### 1. Extreme Value and Adopters

**Value delivered:** A contract synthesis tool that produces SMT-verified specifications without ever invoking a SyGuS solver. Instead, it constructs contracts *directly* from the symbolic differences between original and mutant programs, composing them via a lattice-theoretic algorithm that walks the specification lattice guided by the mutation boundary. The result is deterministic (no solver non-determinism), predictable (time proportional to number of dominator mutants), and produces specifications whose structure directly mirrors the mutation data — making contracts human-readable because each clause corresponds to a specific mutation the test suite kills.

**Who desperately needs this:**
- **Developers who want to understand what their tests enforce.** MutSpec-Δ contracts are not opaque formulas — each conjunct is labeled with the mutation it was derived from: "This clause ensures that `<=` on line 47 cannot be weakened to `<`, as enforced by test `testBoundaryCase`." This traceability makes contracts *documentation*, not just verification artifacts.
- **Specification auditors** in safety-critical domains who must review contracts for adequacy. MutSpec-Δ contracts are auditable by construction: every clause has a provenance trace back to a specific mutation and test.
- **Researchers building on mutation–specification duality** who need a clean, deterministic, reproducible synthesis engine without the non-determinism and scalability unpredictability of SyGuS solvers. MutSpec-Δ is the research platform; MutSpec-Complete is the production system.

### 2. Why This Is Genuinely Hard

**Hard subproblem 1 — WP differencing at scale with sharing.** Computing wp(prefix, wp(s', Post) ⊕ wp(s, Post)) per mutant is textbook for a single mutation. For 500 dominator mutants across a 10K-function codebase, the naive approach generates 500 independent WP computations per function. The hard part is *incremental WP computation with maximal sharing*: mutations at the same program point share the same prefix WP; mutations with the same operator type share structural similarity in their error predicates. Designing a WP engine that exploits this sharing to achieve sub-linear scaling (rather than linear in mutant count) requires a novel incremental symbolic computation architecture.

**Hard subproblem 2 — Lattice-walk contract construction.** Given a set of WP-difference formulas {Δ₁, ..., Δₙ} (one per dominator mutant), the naive contract is ⋀ᵢ ¬Δᵢ — the conjunction of negated error predicates. This is correct but produces contracts with hundreds of conjuncts, most of which are redundant or subsumable. The lattice-walk algorithm must: (a) identify redundant conjuncts via entailment checking, (b) strengthen the contract by finding *simpler* formulas that imply the conjunction (abstraction step), (c) preserve soundness throughout (never emit a formula the original violates), and (d) terminate in polynomial time in the number of dominators. This is a novel algorithm combining elements of abstract interpretation (widening/narrowing) with lattice-theoretic specification construction.

**Hard subproblem 3 — Bridging the expressiveness gap without SyGuS.** SyGuS can synthesize specifications *not directly expressible* as Boolean combinations of WP differences — for example, relational postconditions like `\result == a + b` that emerge from the interaction of multiple mutation sites. The lattice-walk approach is limited to the Boolean closure of WP differences. Proving that this closure is sufficient for the QF-LIA fragment (or precisely characterizing the gap) is a novel theoretical result that determines whether MutSpec-Δ is a complete alternative to SyGuS or a sound but incomplete approximation.

**Architectural challenge:** The WP engine must produce formulas in a normal form compatible with both the lattice-walk algorithm and the Z3 verification backend. Three different consumers of the same symbolic representation, with different requirements on formula structure — the engine must serve all three without exponential blowup from normalization.

### 3. New Math Required

**WP-Composition Completeness Theorem.** Prove that for QF-LIA contracts over loop-free code with standard mutation operators, the Boolean closure of WP-difference formulas B(Δ) = Bool({¬Δᵢ | mᵢ ∈ MKill}) is *specification-complete* — i.e., it contains every contract that the SyGuS approach with mutation-directed grammars could synthesize. Formally: for every φ ∈ Grammar_M (the mutation-directed grammar of Theorem 7), there exists ψ ∈ B(Δ) such that ψ ≡ φ. This is load-bearing because it proves MutSpec-Δ loses nothing relative to the SyGuS approach in the QF-LIA fragment, eliminating the need for a SyGuS solver entirely.

**Lattice-walk termination and optimality.** Prove that the lattice-walk algorithm terminates in O(n² · SMT(n)) time (where *n* is the number of dominator mutants and SMT(n) is the cost of one entailment query on formulas of size O(n)), and produces a contract that is: (a) sound (satisfied by the original), (b) complete w.r.t. killed mutants (violates every killed mutant), and (c) *irredundant* (no conjunct can be removed without losing completeness). Load-bearing because it gives predictable performance guarantees — unlike SyGuS, where solving time is unpredictable and may diverge.

**Simplification-soundness theorem for the abstraction step.** The lattice walk's abstraction step replaces a conjunction of WP differences with a simpler formula that entails it. Prove that the simplification preserves the mutation–specification correspondence: if the simplified contract φ' is derived from φ by abstracting WP-difference conjuncts, then every surviving mutant violating φ also violates φ', and the provenance trace (which mutation each clause derives from) is preserved up to subsumption. Load-bearing for auditability — without this, simplification could obscure which mutations justify which contract clauses, destroying the traceability value proposition.

### 4. Best-Paper Potential

A PLDI committee would select this because it presents a *cleaner theoretical story* than the SyGuS approach, with a surprising equivalence result:
- **Surprise:** "You don't need SyGuS for mutation-directed contract synthesis. WP differencing + lattice walking produces identical contracts with deterministic, predictable performance." This is the kind of simplification result that reviewers love — it shows the problem is *easier than expected*.
- **Elegance:** The lattice-walk algorithm is a clean formalism — pure lattice theory driving contract construction — versus SyGuS's "encode and hope the solver works." The algorithm is analyzable, its complexity is bounded, and its output structure mirrors the mutation data.
- **Traceability:** Every contract clause has a provenance trace to a specific mutation and test, making MutSpec-Δ the first contract inference tool whose output is *auditable by construction*. This is a qualitative advance over Daikon ("likely invariant, source unknown") and SyGuS ("synthesized formula, derivation opaque").
- **Practical:** No SyGuS solver dependency means no scalability cliff. Performance is deterministic and proportional to mutant count — teams can predict synthesis time from mutation analysis results.

The risk: if the WP-Composition Completeness Theorem fails (Boolean closure of WP differences is strictly weaker than SyGuS grammars), MutSpec-Δ becomes an incomplete tool. The paper can still work if the gap is precisely characterized and empirically small, but the clean story becomes a qualified one.

### 5. Hardest Technical Challenge and Mitigation

**Challenge:** The WP-Composition Completeness Theorem may fail. There may exist QF-LIA specifications synthesizable by the SyGuS grammar that are *not* expressible as Boolean combinations of WP differences. For example, relational postconditions like `\result == max(a, b)` might require SyGuS to combine information across multiple mutation sites in ways that pure Boolean combination of per-site WP differences cannot capture. If the theorem fails, MutSpec-Δ is strictly weaker than MutSpec-Complete, and the "you don't need SyGuS" pitch collapses.

**Mitigation:** Three-pronged:
1. **Prove it for the restricted fragment first.** Start with single-expression functions (no control flow), where WP differences are arithmetic formulas and Boolean closure is Boolean algebra. This covers ~30% of functions and establishes the proof technique.
2. **Characterize the gap precisely if full completeness fails.** If some specifications require SyGuS, identify exactly which structural patterns cause the gap (e.g., cross-site relational invariants). This turns a negative result into a positive characterization: "WP-composition is complete for per-site properties; SyGuS is needed only for cross-site relations, which we characterize as..."
3. **Hybrid fallback.** For functions where WP-composition produces specifications measurably weaker than the mutation data supports (detectable by checking whether surviving mutants violate a strictly stronger but unexpressible contract), invoke SyGuS as a targeted fallback on just those functions. This preserves the performance benefits of WP-composition for 80%+ of functions while maintaining completeness via selective SyGuS invocation for the rest.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 7/10 | Traceable, auditable contracts are a unique differentiator; deterministic performance eliminates SyGuS unpredictability; but narrower adopter base than MutGap |
| **Difficulty** | 7/10 | WP-Composition Completeness Theorem is genuinely novel math; lattice-walk algorithm is a new synthesis paradigm; incremental WP sharing is hard systems work |
| **Potential** | 8/10 | "You don't need SyGuS" simplification result is the kind of surprise PLDI loves; clean theory + deterministic tool + auditable output is a compelling package; risk is completeness theorem failure |
| **Feasibility** | 7/10 | No SyGuS scalability risk; WP differencing is well-understood for loop-free code; lattice walk is implementable; main risk is theoretical (completeness theorem), not engineering |

---

## Comparative Summary

| Dimension | A: MutSpec-Complete | B: MutGap | C: MutSpec-Δ |
|-----------|-------------------|-----------|-------------|
| **Core mechanism** | Mutation-directed SyGuS + CEGIS | Template inference + mutation filtering | WP differencing + lattice walk |
| **Primary output** | Certified contracts + bug reports | Ranked bug reports (contracts optional) | Traceable contracts + bug reports |
| **Novel math** | Theorem 3 extension, degradation bound, certificate soundness | Confidence calibration, compositional gap analysis | WP-Composition Completeness, lattice-walk optimality |
| **Key risk** | SyGuS scalability cliff | Equivalent mutant FP rate | Completeness theorem failure |
| **Estimated LoC** | ~65K (25K novel) | ~20K (12K novel) | ~40K (20K novel) |
| **Timeline** | 24 weeks | 16 weeks | 20 weeks |
| **Value** | 8 | 9 | 7 |
| **Difficulty** | 8 | 5 | 7 |
| **Potential** | 7 | 6 | 8 |
| **Feasibility** | 5 | 9 | 7 |

### Trade-off Analysis

**Approach A (MutSpec-Complete)** is the maximally ambitious version: deepest theory, strongest guarantees, highest risk. It produces the most formally compelling artifact (certified contracts) but depends entirely on SyGuS scalability — an unvalidated empirical question. If Gate 1 passes (CVC5 solve rate ≥70%), this is the strongest paper. If Gate 1 fails, 24 weeks are wasted.

**Approach B (MutGap)** is the pragmatist's bet: fastest to build, highest chance of producing a large bug count, and directly addresses the depth check's "bug-finding reframe" amendment. It sidesteps SyGuS entirely, eliminating the project's highest risk. The downside is thinner theory — confidence calibration and compositional gap analysis are solid but not deep enough for PLDI's theory track. This is the safest choice if the goal is "publishable paper with real impact" rather than "best paper with formal novelty."

**Approach C (MutSpec-Δ)** is the creative synthesis: it achieves most of Approach A's formal rigor (verified contracts, clean theory) while eliminating Approach A's key risk (SyGuS scalability). The lattice-walk algorithm is a novel synthesis paradigm — not just "SyGuS without the solver" but a fundamentally different approach with unique properties (deterministic, traceable, analyzable). The WP-Composition Completeness Theorem, if it holds, is a genuine surprise result that could reframe how the community thinks about specification synthesis. If it fails, the hybrid fallback preserves most of the value. This is the best choice if the goal is "novel contribution to synthesis methodology" and the team has strong lattice-theory skills.

### Recommended Strategy

**Start with Approach B's infrastructure** (PIT integration, equivalent-mutant filtering, bug report generation) as the foundation — this is needed by all three approaches and produces value fastest. **Attempt Approach C's completeness theorem in parallel** — if it holds, pivot to MutSpec-Δ as the primary contribution. **Defer Approach A's SyGuS engine** until Gate 1 empirically validates scalability.

This ordering ensures:
- Week 4: Working bug-finding pipeline (Approach B core) — satisfies Gate 4 (one real bug)
- Week 8: Completeness theorem proved or disproved — determines whether C or A is the paper's synthesis story
- Week 12: Full tool with synthesis backend (either lattice-walk or SyGuS) — ready for evaluation
- Week 16–20: Evaluation and writing

The portfolio risk is minimal: even if both C's theorem and A's SyGuS scalability fail, Approach B produces a publishable tool paper with real bugs found.
