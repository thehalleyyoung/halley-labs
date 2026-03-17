# MutSpec-Hybrid: Mutation-Boundary Contract Synthesis via WP Differencing with Lightweight Bug Detection

**Project:** `mutation-contract-synth`
**Date:** 2025-07-18
**Status:** Synthesized from Approaches A, B, C + debate + depth check (composite 6.0/10, CONDITIONAL CONTINUE 2-1)

---

## 1. Approach Name and Summary

**One-line pitch:** Build a mutation-boundary bug finder on B's infrastructure, synthesize contracts via C's lattice-walk algorithm, and ground both in Theorem 3's completeness result — delivering bugs first, contracts second, theory throughout.

MutSpec-Hybrid combines the strongest elements of three competing approaches into a single phased design. The foundation is Approach B's lightweight bug-detection pipeline: a PIT-integrated tool that transforms surviving mutants into ranked, actionable bug reports with concrete distinguishing inputs. This infrastructure — PIT integration, equivalent-mutant filtering, gap analysis, SARIF report generation — is needed by every variant and produces standalone value within weeks. The primary deliverable is bug reports, not contracts. Developers running `mvn mutspec:report` after their existing PIT run receive a ranked list of specification gaps, each backed by a concrete counterexample, without learning JML or any specification language.

On top of this foundation, MutSpec-Hybrid attempts Approach C's key theoretical contribution: the WP-Composition Completeness question. For loop-free QF-LIA code, we compute weakest-precondition differences between original and mutant programs, then compose contracts directly via a lattice-walk algorithm — bypassing SyGuS entirely. If the completeness theorem (or a useful partial version) holds, this gives us deterministic, predictable contract synthesis with bounded complexity. If it fails, we characterize the gap precisely, which is publishable either way, and fall back to template-based synthesis (Daikon-style with mutation-boundary filtering) for functions where WP composition is insufficient.

From Approach A, we take the crown jewel — Theorem 3 (ε-completeness of standard mutation operators for QF-LIA specifications over loop-free code) — as the theoretical foundation that all synthesis depends on. We also adopt A's three-tier degradation concept, adapted: Tier 1 is the lattice-walk (replacing SyGuS), Tier 2 is template-based synthesis with mutation filtering, and Tier 3 is Daikon-quality fallback. All tiers produce SMT-verified output. We explicitly defer A's certificate generation chain and bounded-loop extension to future work.

The synthesis logic is straightforward. B provides the infrastructure everyone needs and the delivery mechanism developers will actually use. C provides the synthesis algorithm that eliminates A's fatal SyGuS scalability risk while preserving formal rigor. A provides the mathematical foundation that elevates the work above "Daikon + PIT with extra steps." The ordering — B first, C's theory in parallel, A's SyGuS only as emergency fallback — reflects unanimous reviewer consensus.

---

## 2. Value Proposition

### Who Desperately Needs This

**Teams already running PIT in CI/CD** — the immediate adopter base. PIT (pitest.org) is integrated into Maven/Gradle workflows at organizations including Google, major financial institutions, and safety-conscious open-source foundations. These teams already pay the computational cost of mutation generation, execution, and kill-matrix computation. They currently extract a single scalar: mutation score. MutSpec-Hybrid adds structured bug detection at marginal cost on top of infrastructure they already run. The pitch is not "adopt formal methods" — it is "run `mvn mutspec:report` after `mvn pitest:mutationCoverage` and get a bug list from the data you already compute."

**Open-source maintainers of high-assurance libraries** — Apache Commons Math (53K tests, zero JML annotations), Guava, Bouncy Castle, Jackson. These projects invest heavily in testing but have no formal specs. MutSpec-Hybrid surfaces bugs their tests miss without asking them to learn JML. A concrete example: "function `Math.clamp` has a surviving mutant that changes `<=` to `<` on line 47; this violates the inferred postcondition and is reachable via input `(x=5, lo=5, hi=10)`." Maintainers understand this report. The formal contract is metadata, not the deliverable.

**Verification tool builders** — the KeY team at KIT, OpenJML developers, Frama-C/ACSL consumers — who need bootstrap contracts to start modular verification. Today they require weeks of manual annotation per module. MutSpec-Hybrid gives them verified starting points as a byproduct of bug detection.

### What Becomes Possible That Wasn't Before

1. **Formally grounded bug detection from existing mutation data.** Every surviving non-equivalent mutant that violates the inferred contract is a concrete witness to a latent defect or test-suite gap, accompanied by a distinguishing input. No existing tool provides this: PIT gives a score, Daikon gives unverified invariants, SpotBugs gives pattern-matched warnings. MutSpec-Hybrid gives formally justified bug reports grounded in what the test suite actually enforces.

2. **Automated specification bootstrapping.** A codebase with 10,000 tested functions receives SMT-verified contracts for each (at varying tightness depending on synthesis tier), with no human effort. These contracts feed directly into KeY or CBMC for modular reasoning.

3. **Quantitative test-suite diagnostics.** The gap between inferred contracts and surviving mutants quantifies exactly how much specification strength the test suite is missing. This is strictly more informative than mutation scores, which count kills but say nothing about what the kills mean.

### The $10K Consultant Test

A competent consultant with PIT + Z3 + Daikon + Python can replicate approximately 70-80% of the practical output in 2–3 months. The recipe: PIT → Z3 equivalence check → Daikon invariants → mutation-boundary filter → ranked bug reports. About 2,000–3,000 lines of scripting. We accept this.

What the consultant *cannot* replicate:
- **Theorem 3** — the formal proof that mutation-adequate test suites determine unique minimal specifications for QF-LIA over loop-free code. This is the intellectual contribution that elevates the work from engineering to science.
- **The WP-Composition Completeness result** (or its precise negation) — novel regardless of outcome.
- **The lattice-walk algorithm** — a new synthesis paradigm with deterministic, bounded complexity that produces contracts whose structure mirrors mutation data.
- **SMT-verified contracts with mutation provenance** — each clause traced to specific mutations and tests, not opaque synthesized formulas.

The paper must center on these unreplicable contributions. The practical tool is the vehicle; the theory is the payload.

### LLM Obsolescence

In 2 years, LLMs will generate "good enough" JML annotations for most developer use cases. We address this head-on:
- **LLM-generated specs are not test-grounded.** An LLM produces specs conditioned on code and training data, not on what the test suite enforces. It can produce a spec weaker or stronger than what tests support, with no way to detect the discrepancy. MutSpec's mutation boundary provides a formal signal LLMs cannot replicate.
- **MutSpec's deterministic, reproducible output is a structural advantage for CI/CD.** An LLM produces different specs on different runs. MutSpec produces identical output for identical inputs — a requirement for CI/CD infrastructure.
- **Bug detection survives LLM competition.** Even if LLMs generate contracts, they cannot perform the gap analysis that identifies which surviving mutants violate those contracts. The bug-finding pipeline is complementary to, not replaced by, LLM-generated specs.

---

## 3. Technical Architecture

### Subsystem Breakdown

| Layer | Subsystem | LoC | Origin | Novel? |
|-------|-----------|-----|--------|--------|
| **1. PIT Integration & Mutation Engine** | | **12,000** | B+shared | |
| | PIT adapter (kill matrix extraction, bytecode-to-source mapping) | 4,000 | B | Integration |
| | Source-level mutation replay (symbolic extraction from PIT data) | 3,500 | Shared | **Novel** — bridges PIT ↔ symbolic gap |
| | Contract-directed mutation operators (20+) | 2,500 | A | Competent integration |
| | Test harness integration (JUnit/TestNG) | 2,000 | B | Integration |
| **2. WP Differencing Engine** | | **8,000** | C | |
| | Java source → IR lowering (loop-free QF-LIA fragment) | 3,000 | A | Integration |
| | WP computation with incremental sharing | 3,500 | C | **Novel** — batch WP with sub-linear scaling |
| | Error predicate extraction (Δᵢ formulas) | 1,500 | C | Competent integration |
| **3. Contract Synthesis** | | **10,000** | C+B | |
| | Lattice-walk algorithm (Tier 1) | 4,000 | C | **Novel** — new synthesis paradigm |
| | Entailment-based redundancy removal + algebraic simplification | 2,500 | C | **Novel** — scoped to avoid reintroducing SyGuS |
| | Template-based synthesis with mutation filtering (Tier 2) | 2,000 | B | Incremental over Daikon |
| | Daikon-quality fallback (Tier 3) | 1,500 | B | Known technique |
| **4. Verification & Gap Analysis** | | **9,000** | A+B | |
| | Bounded SMT verification (Z3) | 3,000 | A | Competent integration |
| | Equivalent mutant filtering (TCE + bounded symbolic) | 2,500 | B | Known techniques, novel combination |
| | Gap analysis engine (surviving mutant classification) | 2,000 | B | **Novel** — Gap Theorem implementation |
| | Bug report generation with witnesses (SARIF output) | 1,500 | B | Integration |
| **5. Subsumption & Scaling** | | **4,000** | Shared | |
| | Dominator-set computation (Theorem 5) | 2,000 | Shared | Competent integration |
| | Subsumption-aware filtering | 2,000 | B | Load-bearing applied result |
| **6. Infrastructure** | | **5,000** | B | |
| | Maven plugin, CLI, configuration | 2,000 | B | Integration |
| | Benchmarking infrastructure (Defects4J integration) | 1,500 | Shared | Integration |
| | Comparison framework (Daikon, SpecFuzzer baselines) | 1,500 | Shared | Integration |
| **TOTAL** | | **~48,000** | | **~16K novel** |

### Total LoC: ~48K (realistic range: 40–58K)

The 48K estimate incorporates the Difficulty Assessor's corrections: the WP engine is budgeted at 8K (not 3K), the PIT integration layer accounts for the symbolic-extraction gap all reviewers flagged, and infrastructure is sized conservatively. The novel core is approximately 16K LoC across four components: source-level mutation replay, batch WP differencing, the lattice-walk algorithm, and the gap analysis engine. The remaining 32K is competent integration of known techniques (PIT, Z3, Daikon templates, SARIF formatting).

### The PIT ↔ Symbolic Reasoning Integration Gap

All three reviewers identified this as the #1 engineering risk. PIT operates at the bytecode level and produces kill matrices as `(mutant_id, test_id, killed/survived)` triples. MutSpec needs symbolic error predicates — formulas characterizing how each mutant diverges from the original at the source level. There is no existing bridge.

Our approach: build a source-level mutation replay layer (3,500 LoC) that re-derives each PIT mutation at the source level using PIT's mutation descriptors (operator type, class, method, line number). This avoids reverse-engineering bytecode transformations while leveraging PIT's execution infrastructure for kill/survive classification. The replay layer maps each PIT mutant to a source-level AST transformation, which the WP engine then consumes. This is the first component built, because everything downstream depends on it.

Risk: PIT's mutation descriptors may not contain sufficient information for precise source-level replay (e.g., when multiple mutation sites exist on the same line). Mitigation: fall back to concrete execution for ambiguous cases, accepting loss of symbolic precision for those mutants.

### Three-Tier Synthesis Architecture

| Tier | Mechanism | Target Coverage | Contract Quality | Performance |
|------|-----------|----------------|-----------------|-------------|
| **Tier 1** | Lattice-walk over WP differences | ~60% of functions | Strongest: irredundant, mutation-provenance traced | Deterministic, O(n² · SMT(n)) |
| **Tier 2** | Template-based with mutation-boundary filtering | ~25% of functions | Moderate: Daikon templates filtered by mutation consistency | Fast, always terminates |
| **Tier 3** | Daikon-quality conjoin-and-filter | ~15% of functions | Weakest: standard Daikon invariants | Fastest, always terminates |

All tiers produce SMT-verified output. Tier level is recorded in metadata. Tier 1 applies to loop-free QF-LIA functions where WP differencing succeeds. Tier 2 handles functions with loops, complex control flow, or WP engine limitations. Tier 3 catches everything else. The 60/25/15 distribution is a hypothesis to be validated empirically — if Tier 1 handles only 30%, the system still produces useful output via Tiers 2 and 3, though the formal contribution is weaker.

### Delineation: Novel vs. Integration vs. Future Work

| Category | Components |
|----------|-----------|
| **NOVEL** (~16K LoC) | Source-level mutation replay; batch WP differencing with sharing; lattice-walk contract synthesis; scoped entailment-based simplification; Gap Theorem implementation |
| **COMPETENT INTEGRATION** (~27K LoC) | PIT adapter; Z3 verification backend; equivalent mutant filtering (TCE + symbolic); Daikon template fallback; JML emission; SARIF bug reports; Maven plugin |
| **FUTURE WORK** (not built) | Full certificate generation chain; bounded-loop Theorem 3 extension; multi-language support (MuIR); higher-order mutations; SyGuS-based synthesis (Approach A's engine); compositional interprocedural analysis |

---

## 4. New Mathematics Required

The base theory presentation is compressed per the Math Assessor's recommendation. Theorems 1 (Duality), 2 (Lattice Embedding), and 4 (Gap Theorem) are presented as **Definitions and Framework** — they are load-bearing as formalism but not mathematically deep. Theorem numbering is reserved for results that require proof.

### Definitions and Framework (formerly Theorems 1, 2, 4)

**Mutation-Induced Specification.** Given function *f*, test suite *T*, and killed mutant set *MKill*, define φ\_M(f,T) = ⋀\_{m ∈ MKill} ¬ℰ(m), where ℰ(m) is the error predicate for mutant *m*. This specification is (a) sound (*f* satisfies it), (b) complete w.r.t. killed mutants (every killed mutant violates it), and (c) weakest among specifications separating *f* from all killed mutants. These are direct consequences of the definitions and require only a Craig interpolation argument for (c).

**Specification Lattice.** The map σ: ℘(MKill) → Spec is a lattice homomorphism. Its image is a finite sub-lattice. Standard order theory; stated as a remark, not a theorem.

**Gap Characterization.** Every surviving mutant is either equivalent (semantically identical to *f*) or distinguishable-but-unkilled (a specification gap). The equivalent mutant barrier (undecidable) is the fundamental limit. This is the *framework* for bug detection; the mathematics is straightforward.

### Theorem 1: ε-Completeness for QF-LIA (Crown Jewel)

**Statement.** The mutation operator set {AOR, ROR, LCR, UOI} is ε-complete for the specification class QF-LIA over loop-free first-order programs: for every non-tautological QF-LIA property φ satisfied by *f*, if φ is violated by some semantic change at an arithmetic, relational, logical, or unary operator site, then at least one mutant from the corresponding operator family witnesses this violation. Consequently, if *T* is mutation-adequate (kills all killable mutants in {AOR, ROR, LCR, UOI}(f)), then φ\_M(f,T) equals the strongest QF-LIA specification satisfied by *f*.

**Why load-bearing.** This is the first formal bridge between mutation adequacy (a testing concept) and specification strength (a verification concept). Without it, MutSpec is "heuristic contract inference with mutation data" — Daikon with extra steps. With it, the contracts have a formal warrant: they are provably the strongest that the mutation data supports.

**Achievability: 65%.** The proof requires case analysis over QF-LIA formula structure and mutation operator families. The restricted setting (loop-free, first-order, four operator families) makes the combinatorics tractable. The risk is multi-site property interactions: a QF-LIA property that requires *simultaneous* changes at two sites to violate may not be witnessed by any single first-order mutant. If this gap exists, the theorem requires an additional "site-independence" premise.

**Risk assessment.** If the theorem fails outright, the project loses its intellectual core. If it requires an additional premise (site-independence), the result is weaker but still publishable — the premise is empirically testable and likely holds for most practical functions. Theorem 1 is the highest-priority proof effort.

### Theorem 2: Subsumption–Implication Correspondence

**Statement.** Mutant m₁ subsumes m₂ iff ¬ℰ(m₁) ⊑ ¬ℰ(m₂) in the specification lattice. The dominator set D yields the same specification as the full killed set. |D| = O(n) vs. |MKill| = O(n·k).

**Why load-bearing.** Reduces the SyGuS/lattice-walk constraint set by a factor of k (number of mutation operators). Without this, synthesis is intractable for real codebases.

**Achievability: 90%.** Standard lattice argument applied to a new setting.

### Theorem 3: WP-Composition Completeness (Restricted Fragment)

**Statement.** For QF-LIA contracts over single-expression functions (no control flow) with standard mutation operators, the Boolean closure of WP-difference formulas B(Δ) = Bool({¬Δᵢ | mᵢ ∈ MKill}) contains every contract that a mutation-directed SyGuS grammar could synthesize. That is, WP differencing + Boolean combination loses nothing relative to SyGuS for this fragment.

**Why load-bearing.** If this holds, MutSpec-Hybrid's lattice-walk produces contracts identical to what SyGuS would produce — without the solver scalability risk. This is the theoretical justification for C's approach and the "you don't need SyGuS" pitch.

**Achievability: 60-70% for the single-expression fragment; 40% for full loop-free QF-LIA.**

The full version likely fails for cross-site relational postconditions (e.g., `\result == max(a, b)`) because WP differences are precondition-space formulas and cannot express input-output relations directly. The Skeptic's counterexample (integer max function) is compelling. Our strategy:

1. **Prove the single-expression fragment first** (~30% of functions). This establishes the proof technique.
2. **Characterize the gap precisely** for multi-expression functions. Identify exactly which structural patterns require SyGuS (cross-site relational invariants). Even a negative result is publishable: "WP-composition is complete for per-site properties; SyGuS is needed only for cross-site relations, which we characterize as..."
3. **Fall back to Tier 2 (template-based)** for functions where the gap bites, preserving soundness.

**Risk assessment.** If only the single-expression fragment holds, MutSpec-Hybrid's Tier 1 covers ~30% of functions (not 60%). The system still works via Tiers 2 and 3, but the "you don't need SyGuS" pitch becomes "you need SyGuS less often than you think." The paper adapts to report the partial result + gap characterization.

### Theorem 4: Lattice-Walk Termination and Irredundancy

**Statement.** The lattice-walk algorithm terminates in O(n² · SMT(n)) time and produces a contract that is sound, complete w.r.t. killed mutants, and irredundant (no conjunct removable without losing completeness).

**Why load-bearing.** Gives predictable performance guarantees — unlike SyGuS, where solving time is unpredictable. This is a practical advantage and a formal one.

**Achievability: 80%.** Standard lattice-fixpoint theory. Monotone decrease in a finite lattice guarantees termination. The irredundancy proof requires care with the entailment-checking step.

### The Degradation Bound Problem (Reformulated)

The original conjecture — g(s) ≥ s, where a test suite with mutation score *s* captures at least fraction *s* of the strongest specification — is **almost certainly false as stated**. The Math Assessor explains: mutation score measures *fraction* of mutants killed, but specification strength depends on *which* mutants are killed. Killing 70% of mutants that all correspond to the same specification clause gives one clause, not 70% of the specification.

**Reformulation.** We pursue a weaker but honest result: under a *distributional uniformity* assumption (killed mutants are not concentrated on a single specification clause), a test suite with mutation score *s* captures at least g(s) fraction of the specification, where g is monotone and g(1) = 1. The uniformity assumption is empirically testable on real codebases.

If even this reformulation proves intractable, we accept that Theorem 1 is an asymptotic ceiling result (requires 100% mutation adequacy) and provide extensive empirical evidence that specification quality degrades gracefully with mutation score in practice. The evaluation measures this directly.

### What Is a Definition vs. What Is a Real Theorem

| Item | Status | Why |
|------|--------|-----|
| Mutation-induced specification φ\_M(f,T) | Definition | Direct construction from error predicates |
| Specification lattice embedding | Remark | Standard order theory, two-line observation |
| Gap characterization | Framework definition | Partition of survivors into equivalent/unkilled is definitional; equivalent-mutant undecidability is classical |
| **Theorem 1 (ε-completeness)** | **Real theorem** | Requires proof by case analysis over QF-LIA formulas and mutation operators |
| **Theorem 2 (subsumption-implication)** | **Applied theorem** | Known concept (subsumption) connected to new setting (specification lattice) |
| **Theorem 3 (WP-composition completeness)** | **Open question** | May prove or disprove; either outcome is publishable |
| **Theorem 4 (lattice-walk properties)** | **Real theorem** | Novel algorithm requiring termination + optimality proof |

---

## 5. Hard Problems and Risk Mitigation

### Problem 1: PIT ↔ Symbolic Reasoning Bridge (Risk: 8/10)

**Why hard.** PIT operates at JVM bytecode level. MutSpec needs source-level symbolic formulas. There is no existing bridge. Reverse-engineering PIT's bytecode transformations is fragile; building a parallel source-level mutation engine is expensive (duplicates PIT).

**Mitigation.** Build a source-level mutation replay layer using PIT's mutation descriptors (operator, class, method, line). This is the first component built (Weeks 1-4). If PIT's descriptors prove insufficient for precise replay, fall back to concrete execution for ambiguous mutants, accepting partial loss of symbolic precision.

**If mitigation fails.** Build a standalone source-level mutation engine (~4K additional LoC, ~3 weeks). This is expensive but not fatal — it replaces PIT's execution with our own, losing the "marginal cost on top of PIT" pitch but preserving all synthesis and analysis.

### Problem 2: Theorem 1 (ε-Completeness) May Fail (Risk: 7/10)

**Why hard.** The proof requires showing that every QF-LIA property violated by a semantic change at an operator site is witnessed by a first-order mutant from {AOR, ROR, LCR, UOI}. Multi-site property interactions (properties requiring simultaneous changes at two sites) may create blind spots.

**Mitigation.** Prioritize this proof above all other math. Begin in Week 1. If the full result requires a site-independence premise, add the premise and validate empirically. The Math Assessor gives 65% achievability, which we treat as realistic.

**If mitigation fails.** The project loses its crown jewel. The lattice-walk algorithm and gap analysis still work heuristically, but without formal grounding. The paper repositions as a tool/empirical contribution targeting OOPSLA/FSE rather than PLDI. Salvage value: the *question* is still novel, and a negative result with precise counterexample is publishable at a workshop (FormaliSE).

### Problem 3: WP Engine for Java Semantics (Risk: 6/10)

**Why hard.** Java's reference semantics, autoboxing, exception propagation, and array bounds make WP computation harder than textbook Dijkstra calculus. JBMC's Java-to-SMT encoding is ~15K LoC of C++. We are encoding a much smaller fragment (QF-LIA, loop-free), but even this requires handling method dispatch, primitive widening, and null checks.

**Mitigation.** Budget 8K LoC (not 3K). Scope strictly to the QF-LIA loop-free fragment: integer/long arithmetic, comparisons, boolean logic, simple array access. Exclude: heap reasoning, generics, lambdas, exception handlers beyond null checks. Test against JBMC on the overlapping fragment for validation.

**If mitigation fails.** The WP engine handles only a subset of the QF-LIA fragment (e.g., scalar arithmetic only, no arrays). Tier 1 coverage drops from ~60% to ~30%. Tiers 2 and 3 cover the rest. The formal story is weaker but the bug-finding pipeline is unaffected.

### Problem 4: Equivalent Mutant False Positives (Risk: 5/10)

**Why hard.** 5–25% of surviving mutants are equivalent (undecidable in general). TCE catches ~35% of equivalents; bounded symbolic equivalence adds ~25%; ~40% remain undetectable. The Skeptic estimates a 20% FP rate before filtering, 8-12% after.

**Mitigation.** Multi-layer filtering: (1) TCE (bytecode identity), (2) bounded symbolic equivalence via Z3, (3) heuristic detectors (dead code, idempotent mutations), (4) conservative reporting (flag only mutants with high confidence). Report FP rates transparently. Label gap reports with confidence levels. Do not claim "zero false positives."

**If mitigation fails.** FP rate exceeds 15%. The tool becomes SpotBugs-quality (20-40% FP) rather than precision-oriented. Still usable with developer-configurable thresholds, but the "formally grounded" pitch is undermined for the bug-detection output. Contracts remain sound regardless of FP rate (equivalent mutants don't affect contract correctness, only gap analysis precision).

### Problem 5: Lattice-Walk Abstraction Reintroduces Synthesis (Risk: 4/10)

**Why hard.** The Skeptic correctly identified that the lattice-walk's abstraction step — "find a simpler formula that implies the conjunction" — is a synthesis problem. If open-ended, it's SyGuS without the solver.

**Mitigation.** Strictly scope abstraction to two operations: (1) entailment-based redundancy removal (drop conjuncts implied by others — requires only SMT queries, not synthesis), and (2) known algebraic simplifications (distribute, factor, simplify linear inequalities — template-based, not open-ended). Do not attempt arbitrary simplification. Accept that contracts may have more conjuncts than the "ideal" formulation. Traceability is many-to-few (each simplified clause maps to a set of mutations), not 1:1.

**If mitigation fails.** Contracts are correct but verbose (many redundant-looking conjuncts). This is a usability issue, not a correctness issue. The paper reports contract size alongside quality metrics.

---

## 6. Phased Execution Plan

### Phase 0: Shared Foundation (Weeks 1–4) — GATE CHECK

**Build:**
- PIT integration adapter (kill matrix extraction)
- Source-level mutation replay layer (the PIT ↔ symbolic bridge)
- TCE-based equivalent mutant filter
- Dominator-set computation

**Prove:**
- Begin Theorem 1 (ε-completeness) proof effort

**Gate 1 (Week 4): PIT Bridge Validation**
- Can we extract symbolic mutation data for ≥80% of PIT mutants on 50 Apache Commons Math functions?
- Pass: ≥80% symbolic extraction rate. Fail: <50% → build standalone mutation engine (add 3 weeks).

**P(completion): 85%**

### Phase 1: Bug-Finding Pipeline (Weeks 5–8) — GATE CHECK

**Build:**
- Gap analysis engine (surviving mutant classification)
- Bounded symbolic equivalence filtering (Z3)
- Distinguishing input generation for flagged mutants
- Bug report generation (SARIF output)
- Maven plugin shell

**Validate:**
- Run gap analysis on 50 Commons Math functions
- Classify surviving mutants: bug / test-gap / equivalent / unclear

**Gate 2 (Week 8): Gap Theorem Validation**
- Confirmed bug + test-gap rate ≥75% of flagged mutants (FP ≤25% after filtering)
- ≥1 previously unknown defect in a maintained library
- Pass: both criteria. Fail: FP >40% → ABANDON gap-analysis as primary output; pivot to theory-only.

**Gate 3 (Week 8): SpecFuzzer Baseline**
- Run SpecFuzzer + Z3 on same 50 functions
- MutSpec output must be measurably superior on ≥1 dimension
- Pass: superiority demonstrated. Fail: parity → ABANDON tool contribution; pivot to theory paper.

**P(completion): 70%**

### Phase 2: Contract Synthesis (Weeks 9–14)

**Build:**
- WP differencing engine (batch computation with sharing)
- Lattice-walk algorithm (Tier 1 synthesis)
- Template-based synthesis with mutation filtering (Tier 2)
- Daikon fallback (Tier 3)
- SMT verification backend (Z3)
- JML emitter

**Prove:**
- Complete Theorem 1 proof (or identify required additional premises)
- Attempt Theorem 3 (WP-Composition Completeness) for single-expression fragment
- Prove Theorem 4 (lattice-walk termination and irredundancy)

**Priority ordering within Phase 2 (if time runs short):**
1. **Must-build (Weeks 9-11):** WP differencing engine + Tier 2 template synthesis + SMT verification backend. These produce verified contracts without the lattice-walk, enabling an OOPSLA/FSE paper with Theorem 1 + template-based contracts + bugs.
2. **Should-build (Weeks 11-13):** Lattice-walk algorithm (Tier 1). This is the stretch goal that elevates the paper to PLDI. If Phase 2 runs 2 weeks over, the lattice-walk is the first thing cut.
3. **Nice-to-have (Weeks 13-14):** Daikon fallback (Tier 3), JML emitter polish. Useful for completeness but not publication-critical.

**Theorem proof priority:** Theorem 1 (ε-completeness) is attempted throughout; Theorem 3 (WP-composition) is attempted only after must-build components are stable; Theorem 4 (lattice-walk) is proved only if the algorithm is implemented.

**Gate 4 (Week 14): Synthesis Quality**
- Tier 1 handles ≥40% of loop-free QF-LIA functions
- Contracts measurably tighter than Daikon on ≥50% of Tier 1 functions
- Pass: both criteria. Fail: Tier 1 <20% → drop lattice-walk, use template-only (weaker paper targeting OOPSLA/FSE).

**P(completion): 50%**

### Phase 3: Evaluation and Writing (Weeks 15–22)

**Run:**
- Full evaluation on Defects4J + live libraries (RQs 1-7)
- Ablation study
- Scalability benchmarks

**Write:**
- Paper targeting PLDI (if Theorems 1 + 3 hold + 30+ bugs) or OOPSLA/FSE (if partial results + 15+ bugs)

**P(completion of publishable paper): 45%**

### Timeline Summary

| Week | Milestone | Gate | Fail Action |
|------|-----------|------|-------------|
| 4 | PIT bridge working | Gate 1 | Build standalone mutation engine |
| 8 | Bug-finding pipeline + SpecFuzzer comparison | Gates 2, 3 | ABANDON or pivot to theory |
| 14 | Contract synthesis operational | Gate 4 | Drop lattice-walk, template-only |
| 22 | Paper draft complete | — | Adjust venue target |

**Honest P(completion) for each outcome:**
- PLDI-quality paper (theory + bugs + tool): 20-25%
- OOPSLA/FSE paper (partial theory + bugs + tool): 35-40%
- ASE/ISSTA paper (tool + empirical results): 55-60%
- Publishable artifact at any venue: 65-70%

---

## 7. Best-Paper Argument

### Why a PLDI Committee Would Select This

A PLDI committee selects papers that reveal surprising connections, prove non-obvious theorems, and demonstrate practical impact. MutSpec-Hybrid offers three contributions, each individually strong:

**Contribution 1: Theorem 1 (ε-Completeness).** The first formal proof that standard mutation operators determine unique minimal specifications for QF-LIA over loop-free code. Most PL researchers would expect mutation testing to be "too coarse" for specification determination — that killed mutants provide only fragmentary behavioral information. Theorem 1 proves otherwise for a practical fragment. The restriction to QF-LIA and loop-free code is precisely characterized: users know exactly when the guarantee applies. This is the kind of "obvious in hindsight" result that characterizes strong PLDI contributions.

**Contribution 2: The lattice-walk synthesis algorithm.** A new approach to contract synthesis that is deterministic, has bounded complexity, and produces contracts whose structure mirrors mutation data. This is not "SyGuS without the solver" — it is a fundamentally different paradigm where contracts are *composed from semantic differences* rather than *searched for in a grammar*. The WP-Composition Completeness result (even partial) connects this paradigm to the SyGuS approach with a precise characterization of what is lost.

**Contribution 3: Real bugs found in real code.** A result of the form "MutSpec identified 35 previously unknown latent bugs across 8 maintained Java libraries, including 12 confirmed by maintainers" transforms a strong paper into a best paper. This is the leg that makes reviewers care about the theory.

### What Makes This NOT "Daikon + PIT with Extra Steps"

1. **Daikon has no mutation signal.** Daikon observes execution traces and guesses invariants. MutSpec uses the mutation boundary — what the test suite *enforces* vs. *permits* — as a constructive specification source. These are fundamentally different signals.

2. **PIT has no contracts.** PIT produces a scalar score and a list of survivors. MutSpec transforms the kill matrix into SMT-verified specifications and formally grounded bug reports.

3. **The combination is more than the sum.** The mutation-directed lattice walk produces contracts that neither Daikon (no mutation data) nor PIT (no synthesis) can generate. The Gap Theorem connects the two — surviving mutants that violate inferred contracts — in a way that is formally justified, not heuristic.

4. **SpecFuzzer uses mutation as a filter; MutSpec uses it as a construction.** SpecFuzzer generates candidate specs from a fixed grammar, then filters by mutation consistency. MutSpec constructs contracts directly from WP differences. The grammar is data-driven, not hand-authored.

### Target Venue Hierarchy

| Venue | Condition | Probability |
|-------|-----------|-------------|
| **PLDI** | Theorem 1 proved + WP-Composition partial result + 30+ bugs confirmed | 20-25% |
| **OOPSLA** | Theorem 1 proved + lattice-walk algorithm + 15+ bugs confirmed | 35-40% |
| **FSE** | Lattice-walk algorithm + 15+ bugs + strong empirical results | 40-45% |
| **ASE/ISSTA** | Working tool + 10+ bugs + SpecFuzzer comparison | 55-60% |
| **FormaliSE** | Theorem 1 alone (theory paper, no tool) | 70-75% |

---

## 8. Evaluation Plan

Adapted from the problem statement with reviewer feedback incorporated. All evaluation is fully automated.

### RQ1: Bug Detection via Gap Analysis (Primary RQ)

**Setup.** Run MutSpec on Defects4J (835 real bugs across 17 Java projects). For each buggy version, run on pre-fix code with pre-fix test suite.

**Metrics.** Bug detection rate (fraction of known bugs flagged by gap analysis), precision of gap reports (fraction corresponding to actual bugs vs. equivalent mutants), comparison to standalone PIT surviving-mutant reports, Daikon invariant violations, and SpecFuzzer + Z3 violations.

**Addressing benchmark selection bias (per reviewer feedback).** Defects4J includes string-heavy (JFreeChart), OOP-heavy (Closure compiler), and complex-control-flow code (JxPath) — not just arithmetic functions where QF-LIA excels. We explicitly partition results by code category (arithmetic/loop-free vs. string/OOP vs. complex-control-flow) and report MutSpec's performance on each.

### RQ2: Contract Quality

**Setup.** Compare inferred contracts against ground truth on functions with existing JML annotations (community JML specs for Commons Math, Guava subsets). If JML ground truth is insufficient (<30 functions), use Defects4J pre-fix/post-fix pairs as implicit contracts.

**Metrics.** Precision (fraction of inferred contract clauses that are sound), recall (fraction of ground-truth clauses captured), specificity (how tight — lattice distance from ⊤).

**Baselines.** Daikon, SpecFuzzer + Z3 verification, Houdini, Descartes (PIT plugin for pseudo-tested method detection), and comparison against Google's internal mutation triage methodology (ICSE-SEIP 2018). Descartes addresses a specific subset of surviving-mutant classification; MutSpec must demonstrate superiority on Descartes' home turf (methods with unchecked return values) and broader coverage beyond it. Google's published approach uses ML-based triage on internal data; we compare methodology and, where public data permits, replication.

### RQ3: Live Bug Detection on Current Code

**Setup.** Run gap analysis on current HEAD of 5–10 maintained Java libraries (Apache Commons Math, Guava, Commons Lang, Joda-Time, Jackson-core). File bug reports for specification gaps with concrete witnesses.

**Metrics.** Number of gaps identified, bug reports filed, reports confirmed by maintainers.

### RQ4: Scalability

**Setup.** Codebases of increasing size: 1K → 10K → 50K → 100K LoC.

**Metrics.** Wall-clock time (per phase), tier distribution, memory peak, impact of subsumption reduction and WP differencing.

**Target.** 10K-function codebase in overnight CI window (8-12 hours on 8-core laptop). We provide arithmetic supporting this claim, not just assertion.

### RQ5: Ablation Study

Remove each key component: mutation signal, lattice-walk synthesis, SMT verification, subsumption reduction, WP differencing, three-tier strategy. Measure impact on RQ1 and RQ2.

**Critical ablation:** mutation-boundary filtering removed (testing the "just Daikon?" critique). If bug-finding rate drops <30% when mutation signal is removed, the mutation-directed approach adds marginal value. We need ≥50% drop to justify the approach.

### RQ6: Equivalent Mutant Impact

**Setup.** Classify surviving mutants via TCE + symbolic equivalence on a subset. Compare gap analysis with and without filtering.

**Metrics.** False positive rate, effectiveness of filtering stack, impact on contract quality.

### RQ7: Restriction Boundary Impact

**Setup.** Partition functions into loop-free QF-LIA, loop-free non-QF-LIA, and loopy. Compare all metrics across partitions.

**Addressing reviewer concern.** This RQ directly measures what happens outside Theorem 1's scope. If loopy/non-QF-LIA functions still yield useful bugs and reasonable contracts (via Tiers 2-3), the restriction is practically tolerable. If they degrade sharply, we report this honestly and identify the restriction boundary as the priority for future work.

---

## 9. Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Value** | 6/10 | Real adopters exist (PIT users, verification bootstrapping) but the base is narrower than initially claimed; bug-finding reframe is the viable pitch. |
| **Difficulty** | 6/10 | One genuinely novel research problem (lattice-walk synthesis from WP differences) plus competent integration; harder than a wrapper, easier than a paradigm shift. |
| **Potential** | 7/10 | Theorem 1 + lattice-walk + real bugs is a strong three-legged stool for PLDI; the WP-Composition question adds novelty even if it fails; 20-25% P(best paper) is realistic. |
| **Feasibility** | 6/10 | Phased approach with gates eliminates A's fatal SyGuS risk; B's infrastructure provides floor; but WP engine and Theorem 1 proof are genuine uncertainties; 45% P(OOPSLA-quality paper). |

These scores reflect post-debate reality. Value is lower than B's pre-debate 9 because "10,000 PIT users" is inflated and competition (Google's internal triage, Descartes) is real. Difficulty is lower than A's pre-debate 8 because the CEGIS variant is a variant and certificate generation is deferred. Potential is higher than B's pre-debate 6 because the lattice-walk and WP-Composition question add genuine novelty. Feasibility is higher than A's post-debate 3 because we eliminate SyGuS as the primary mechanism and have clear fallback paths.

---

## 10. Honest Risks and Limitations

### Top 5 Things That Could Make This Project Fail

1. **Theorem 1 fails and no useful weakening exists.** Without ε-completeness, the entire framework is heuristic. The project becomes "Daikon + PIT with a lattice-walk" — a tool paper, not a theory paper. Probability: 35%.

2. **WP engine for Java is 2x harder than budgeted.** Java's semantics are complex even for the loop-free QF-LIA fragment. If the engine takes 16 weeks instead of 8, the project timeline collapses. Probability: 25%.

3. **Bug yield is underwhelming.** If MutSpec finds <10 real bugs across all maintained libraries, the three-legged stool loses its empirical leg. The paper becomes theory-only (FormaliSE, not PLDI). Probability: 30%.

4. **Equivalent mutant FP rate exceeds 20%.** If filtering fails to bring FP below 20%, developers won't trust the output. The "formally grounded bug detection" pitch is undermined. Probability: 25%.

5. **SpecFuzzer + Z3 achieves comparable quality.** If post-hoc Z3 verification of SpecFuzzer output matches MutSpec's contract quality, the lattice-walk and WP differencing are unnecessary overhead. The paper loses its tool contribution. Probability: 20%.

### Worst Plausible Case

Theorem 1 requires a site-independence premise that holds for only 40% of functions. The WP engine handles only scalar arithmetic (no arrays), limiting Tier 1 to 20% of functions. MutSpec finds 5-8 bugs, of which 2-3 are confirmed. The equivalent-mutant FP rate is 18%. SpecFuzzer + Z3 matches MutSpec quality on 60% of benchmarks.

**Outcome:** A modest ASE/ISSTA tool paper with the lattice-walk algorithm as the contribution, the qualified Theorem 1 as supporting theory, and a few confirmed bugs as validation. The WP-Composition Completeness question and its answer (likely negative, with gap characterization) generate a companion FormaliSE theory paper. Combined value: two B-grade publications establishing priority on the mutation-specification duality.

### Best Plausible Case

Theorem 1 proves cleanly for the full QF-LIA loop-free fragment. The WP-Composition Completeness Theorem holds for per-site properties and the cross-site gap is precisely characterized. The lattice-walk handles 65% of functions at Tier 1 quality. MutSpec finds 40+ bugs across 8 libraries, with 15+ maintainer confirmations, including 3+ in released versions. FP rate is 8% after filtering. Contracts are measurably tighter than SpecFuzzer + Z3 on 70% of benchmarks.

**Outcome:** A strong PLDI submission with all three legs of the stool solid. Theorem 1 is the surprise result ("mutation determines specification"), the lattice-walk is the mechanism, the bugs are the evidence. 20-25% chance of best-paper consideration. The tool is open-sourced as a Maven plugin and adopted by 2-3 open-source projects within a year.

### Concerns from the Skeptic (All Addressed)

| Skeptic Concern | Response |
|----------------|----------|
| "$10K consultant replicates 80%" | Accepted. The paper centers on unreplicable contributions: Theorem 1, WP-Composition question, lattice-walk algorithm. The tool is the vehicle, not the payload. |
| "Nobody wants mutation-derived contracts" | Reframed: primary deliverable is bug reports, not contracts. Contracts are a byproduct for teams that want them. |
| "SyGuS scalability is fatal" | Eliminated. Lattice-walk replaces SyGuS as primary mechanism. SyGuS is not used. |
| "Theorem 3 [now 1] covers only loop-free QF-LIA" | Accepted. We do not extend to loops. We position Theorem 1 as a lighthouse result that proves the duality is real in a clean fragment and motivate heuristic generalization empirically. |
| "g(s) ≥ s is probably false" | Accepted. Reformulated under distributional uniformity or abandoned in favor of empirical evidence of graceful degradation. |
| "Equivalent mutant FP rate defeats usability" | Addressed via multi-layer filtering + honest reporting + confidence levels. We target 8-12% FP, comparable to industry static analysis tools. We do not claim zero FP. |
| "LLMs will obsolete this in 2 years" | Partially accepted. LLMs will generate contracts but cannot perform gap analysis or produce deterministic CI/CD output. Bug-finding pipeline and formal theory survive LLM competition. |
| "0-1 real adopters" | Partially accepted. We claim PIT users and verification tool builders as narrow but real niches. We do not claim mass adoption. |
| "Certificate generation is a multi-year project" | Accepted. Deferred to future work. Not attempted in this project. |

---

*Synthesized from Approaches A (MutSpec-Complete), B (MutGap), C (MutSpec-Δ), the compiled expert debate, and the depth check. Reviewed against all Skeptic concerns, Math Assessor recommendations, and Difficulty Assessor corrections.*

---

## Independent Verification Report

**Verifier:** Independent Verifier
**Date:** 2025-07-18
**Verdict: APPROVED WITH CHANGES**

---

### Structural Completeness

**1. All 10 required sections present? YES.**
The document contains: (1) Name/Summary, (2) Value Proposition, (3) Technical Architecture, (4) New Mathematics Required, (5) Hard Problems and Risk Mitigation, (6) Phased Execution Plan, (7) Best-Paper Argument, (8) Evaluation Plan, (9) Scores, (10) Honest Risks and Limitations. All sections are substantive and well-developed.

**2. LoC breakdown realistic? YES, with minor note.**
The total is revised to ~48K (range 40–58K), down from the original 65K. The WP engine is budgeted at 8K (incorporating the Difficulty Assessor's correction from 3K). The PIT integration layer (4K) accounts for the symbolic-extraction gap. The 16K novel / 32K integration split is honest. One concern: the "Source-level mutation replay" at 3,500 LoC may be optimistic given that PIT's mutation descriptors may be ambiguous (acknowledged in the risk section), but the fallback (standalone engine at +4K LoC) is priced in.

**3. Math claims honest? YES.**
Theorems 1, 2, and 4 from the original numbering are correctly compressed to "Definitions and Framework" per the Math Assessor's recommendation. The four numbered theorems (ε-completeness, subsumption-implication, WP-composition completeness, lattice-walk termination) are all genuine proof obligations. The degradation bound is honestly reformulated as an open problem with empirical fallback. The document does not dress definitions as theorems.

**4. Execution plan with clear gates and fail actions? YES.**
Four gates (Weeks 4, 8, 8, 14) with explicit pass/fail thresholds and concrete fail actions (build standalone engine, ABANDON/pivot, drop lattice-walk). Fail actions are not vague — they name specific pivots with timeline implications.

**5. Scores consistent with debate's revised scores? MOSTLY.**
Debate revised scores: A=(6,7,5,3), B=(7,4,6,7), C=(6,6.5,7,5). The final approach's scores (Value=6, Difficulty=6, Potential=7, Feasibility=6) are a reasonable weighted blend of primarily C with B's infrastructure. Value=6 matches C's post-debate 6. Difficulty=6 is between C's 6.5 and B's 4 — reasonable given the hybrid takes C's hard components plus B's easier infrastructure. Potential=7 matches C's post-debate 7. Feasibility=6 is above C's 5, justified by B's infrastructure providing a floor and the elimination of SyGuS risk. Internally consistent.

### Debate Fidelity

**6. Math Assessor's key recommendations incorporated? YES.**
- Theorems 1, 2, 4 compressed to definitions: ✓ (Section 4, "Definitions and Framework")
- Degradation bound reformulated: ✓ (Section 4, "The Degradation Bound Problem (Reformulated)" — explicitly states the linear bound is "almost certainly false" and reformulates under distributional uniformity)
- Theorem 3 proof prioritized: ✓ (Section 5, Problem 2; Section 6, Phase 0 starts proof in Week 1)

**7. Difficulty Assessor's corrections incorporated? YES.**
- WP engine 6–10K LoC: ✓ (budgeted at 8K in the architecture table)
- PIT integration gap: ✓ (Section 3, dedicated subsection "The PIT ↔ Symbolic Reasoning Integration Gap" with 3,500 LoC replay layer)
- Realistic timelines: ✓ (22 weeks total; P(completion) estimates range 85% down to 45% across phases; honest P(PLDI paper) = 20-25%)

**8. Skeptic's attacks addressed? YES — all nine concerns have explicit responses.**
- $10K consultant test: ✓ (Section 2, accepted; paper centers on unreplicable contributions)
- LLM obsolescence: ✓ (Section 2, three-point response: not test-grounded, not deterministic, can't do gap analysis)
- SyGuS scalability: ✓ (Eliminated entirely — lattice-walk replaces SyGuS)
- Equivalent mutant FP rate: ✓ (Section 5, Problem 4; targets 8-12% with multi-layer filtering; doesn't claim zero FP)
- Section 10 table addresses all nine Skeptic concerns with concrete responses.

**9. Any Skeptic concerns dismissed without adequate response? ONE MINOR GAP.**
The Skeptic's concern about "0-1 real adopters" for Approach C's traceable contracts is addressed only with "Partially accepted. We claim PIT users and verification tool builders as narrow but real niches." The Skeptic also raised competition blindspots (Google's internal mutation triage, Descartes, PIT's own evolution) — these are not explicitly addressed in the final approach despite being flagged as "SEVERE" in the debate. The evaluation plan does include SpecFuzzer comparison (Gate 3) but does not mention Descartes or Google's ICSE-SEIP 2018 work as baselines.

**REQUIRED CHANGE 1:** Add Descartes and Google's internal mutation triage (ICSE-SEIP 2018) to the related work comparison in the evaluation plan (Section 8), or add a note in Section 2 acknowledging these as competitors and explaining differentiation.

### Technical Soundness

**10. Three-tier architecture internally consistent? YES.**
Tier 1 (lattice-walk, ~60%) → Tier 2 (template + mutation filtering, ~25%) → Tier 3 (Daikon fallback, ~15%). All tiers produce SMT-verified output. The tier boundaries align with the WP engine's scope (loop-free QF-LIA for Tier 1). The architecture table's LoC numbers are internally consistent — the synthesis layer (10K) is split logically across the three tiers. The Tier 1 coverage estimate of ~60% is acknowledged as a hypothesis (with fallback to 30% if WP-Composition only holds for single-expression functions), which is honest.

**11. WP-Composition Completeness fallback if theorem fails? YES.**
Three-pronged strategy: (1) prove for single-expression fragment, (2) characterize gap precisely for multi-expression, (3) fall back to Tier 2 templates. The Skeptic's concrete counterexample (integer max function) is acknowledged. The degradation from 60% Tier 1 to 30% is quantified. The "either outcome is publishable" claim is well-justified — a precise gap characterization is a genuine contribution.

**12. Phased execution plan achievable? P(completion) estimates honest? MOSTLY.**
- Phase 0 P(completion)=85%: Reasonable for PIT integration + filter + dominator computation.
- Phase 1 P(completion)=70%: Reasonable — gap analysis and bug-report generation are well-understood.
- Phase 2 P(completion)=50%: Honest and reflects genuine uncertainty about WP engine + lattice-walk + theorem proofs in 6 weeks.
- Phase 3 P(publishable paper)=45%: Slightly optimistic given the dependencies, but within range.
- Overall outcome probabilities (20-25% PLDI, 35-40% OOPSLA, 55-60% ASE, 65-70% any venue): Internally consistent and honest. These compound correctly — the venue hierarchy degrades gracefully, which is a good sign.

One concern: Phase 2 crams the WP engine (8K LoC), lattice-walk (4K), template synthesis (2K), Daikon fallback (1.5K), SMT backend (3K), JML emitter, AND three theorem proof efforts into 6 weeks. This is ~18.5K LoC + three proofs in 6 weeks. Even at generous productivity assumptions (500 LoC/week for systems code), this is tight. The P(completion)=50% partially accounts for this but should be more explicit about what gets cut if the phase runs long.

**REQUIRED CHANGE 2:** Add a paragraph to Phase 2 specifying priority ordering within the phase: which components are built first if time runs short (presumably WP engine + Tier 2 template synthesis, with lattice-walk as the stretch goal), and how this affects the paper's positioning.

**13. Risk probabilities internally consistent? YES.**
The five risk probabilities (35%, 25%, 30%, 25%, 20%) sum to 135%, which is correct — these are not mutually exclusive events. The worst-case scenario described is plausible and maps to an ASE/ISSTA paper, which aligns with the 55-60% probability assigned to that outcome. The best-case scenario is optimistic but not fantastical (40+ bugs, 15+ confirmed, 8% FP, clean Theorem 1) and maps to the 20-25% PLDI probability. The probabilities tell a coherent story.

### Portfolio Differentiation

**14. Distinctly different from existing portfolio projects? YES.**
The specifically named comparators (cross-lang-verifier, synbio-verifier, dp-verify-repair) do not exist in this portfolio. Among the actual portfolio neighbors:
- **certified-min-sandbox-synth** (area-009): Different — focused on sandbox synthesis, not mutation testing or specification inference.
- **modular-diff-summaries-upgrade-certs** (area-009): Different — focused on upgrade certificates for module summaries, not mutation-specification duality.
- **behavioral-semver-verifier** (area-049): Different — focused on semantic versioning verification, not mutation-guided contract synthesis.
- **sql-contract-verifier** (area-086): Different — focused on SQL pipeline data contracts via abstract interpretation, entirely different domain and technique.
- **certified-leakage-contracts** (area-048): Different — security/leakage domain, not specification inference from mutation testing.

MutSpec-Hybrid's core mechanism (mutation-boundary → WP differencing → lattice-walk → contracts + gap analysis) is unique in the portfolio. No other project operates at the intersection of mutation testing and specification inference.

### Summary of Required Changes

| # | Change | Section | Severity |
|---|--------|---------|----------|
| 1 | Add Descartes and Google's ICSE-SEIP 2018 mutation triage as explicit comparators in evaluation or related work | §8 or §2 | Minor |
| 2 | Add priority ordering within Phase 2 specifying which components to build first if time runs short, and how truncation affects paper positioning | §6 Phase 2 | Minor |

### Overall Assessment

The final approach is a well-synthesized hybrid that faithfully incorporates the debate's key outputs. The Math Assessor's recommendations are fully reflected (compressed definitions, reformulated degradation bound, prioritized Theorem 3). The Difficulty Assessor's corrections are incorporated (WP engine at 8K, PIT integration gap priced in, realistic timelines). The Skeptic's attacks are addressed systematically, with each concern receiving a concrete response in Section 10's table. The three-tier architecture is technically coherent, the phased plan has meaningful gates with actionable fail paths, the scores are honest and consistent with the debate's revised numbers, and the risk analysis is sober.

The two required changes are minor and do not affect the approach's viability. The document is ready for implementation after these additions.

**VERDICT: APPROVED WITH CHANGES** (2 minor changes required, listed above)
