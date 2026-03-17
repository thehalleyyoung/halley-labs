# MutSpec: Approach Debate — Compiled Expert Review

**Date:** 2025-07-18
**Compiled by:** Team Lead
**Inputs:**
- Math Depth Assessment (Math Depth Assessor)
- Difficulty Assessment (Difficulty Assessor)
- Adversarial Critique (Adversarial Skeptic)
- Three Approaches Document (ideation/approaches.md)

**Purpose:** Faithful record of the expert debate across all three approaches, including cross-expert disagreements, surviving strengths, and fatal issues. This document feeds the Synthesis Editor.

---

## Preamble: The Shared Foundation

Before evaluating the three approaches, the reviewers assessed the base theorems (Theorems 1–7) that all approaches depend on. Their consensus is decisive and must inform every downstream decision.

**The honest math content of the shared foundation is Theorem 3 + Theorem 5. Everything else is definitions, standard techniques, or framing.**

| Theorem | Verdict | Notes |
|---------|---------|-------|
| Thm 1 (Duality) | Definition dressed as theorem | "A PLDI reviewer with PL theory background will see through the packaging." |
| Thm 2 (Lattice) | Ornamental | Trivial order theory; could be a two-line remark. |
| **Thm 3 (Completeness)** | **Crown jewel — genuinely novel and deep** | First formal bridge between mutation adequacy and specification strength. 65% achievability. All approaches depend on it. |
| Thm 4 (Gap) | Load-bearing as framework, mathematically shallow | The Skeptic adds: "practically vacuous" because equivalent-mutant detection is undecidable, so you can never compute the set the theorem guarantees things about. |
| Thm 5 (Subsumption) | Solid applied result | Not deep, but necessary and clean. |
| Thm 6 (WP) | Engineering prerequisite | Textbook WP calculus applied to a new domain. |
| Thm 7 (Grammar) | Necessary for Approach A, not deep | Close to tautological given the definitions. |

**Critical shared risk:** If Theorem 3 fails, the entire project loses its crown jewel regardless of approach. All reviewers agree this proof must be prioritized. The Math Assessor gives 65% achievability; the Skeptic raises a specific counter-scenario involving multi-site property interactions in loop-free code that the "first-order mutations" restriction may not handle.

---

## Approach A: MutSpec-Complete — Full Mutation-Directed SyGuS with Verified Certificates

### Summary

Approach A is the maximally ambitious version: build a certified SyGuS engine that extracts SMT-verified contracts from mutation data, with machine-checkable certificates. It extends Theorem 3 to bounded loops, proves a quantitative degradation bound for partial mutation adequacy, and produces proof witnesses for every bug report. The estimated scope is ~65K LoC over 24 weeks.

### Math Depth Assessment

**Grade: B+ (adequate, partially novel, partially load-bearing)**

The Math Assessor identifies three new math claims:

1. **Theorem 3 Extension (bounded loops) — 50% achievable.** Conceptually sound (unroll to depth k, apply loop-free result) but technically tricky. Correlated mutation sites across loop copies are the core difficulty. The restriction needed is likely much narrower than "bounded by constant k" — probably limited to loops with no loop-carried dependencies that create new specification-relevant properties. *Overreach risk: HIGH.*

2. **Quantitative Degradation Bound g(s) ≥ s — 40% achievable.** The Math Assessor calls this **almost certainly false as stated**. The core problem: mutation score measures *fraction* of mutants killed, but specification strength depends on *which* mutants are killed. Killing 70% of mutants that all correspond to the same specification clause gives one clause, not 70% of the specification. A correct version would require distributional uniformity assumptions, making it less useful as a formal guarantee. *This is the math blocker — without some degradation bound, Theorem 3 is vacuously unsatisfied on all real codebases.*

3. **Certificate Soundness — 90% achievable.** Mostly engineering. "If the SMT solver says SAT, and the certificate records the proof, then the contract holds." Not mathematically novel; a PLDI reviewer would see it as expected, not surprising.

**Bottom line:** "The real math is Theorem 3 (ε-completeness for QF-LIA, loop-free, {AOR, ROR, LCR, UOI}). Everything else is either standard, definitional, or aspirational."

### Difficulty Assessment

**Rating: 7/10**

The Difficulty Assessor deflates several "hard subproblems":

| Subproblem | Claimed Difficulty | Assessed Difficulty | Why |
|---|---|---|---|
| Grammar construction | Hard | 5/10 | Engineering design problem, not open research. CVC5 sensitivity is empirical tuning. |
| CEGIS with programs | Fundamentally different | 6/10 | Novel encoding, but CEGIS loop mechanics are standard. |
| Certificate generation | Novel | 4/10 | Systems integration of existing certificate infrastructure (Z3 LFSC, CVC5 Alethe). |
| PIT + CVC5 + Z3 integration | — | **7/10** | **The real hard problem.** Three tools with fundamentally different IRs. |

**Timeline: P(24 weeks) = 25%.** WP engine + CVC5 integration alone is a 12-week effort based on comparable projects. The SyGuS feasibility gate at Week 4 may reveal CVC5 solves <50% of benchmarks, causing a pivot where 4 weeks and 8–10K LoC of SyGuS-specific code are wasted.

**LoC audit:** Total realistic range 49K–77K. The 65K estimate is in range but hides real distribution issues. WP differencing engine underestimated (3K claimed → 5–8K realistic). Grammar construction underestimated by 2–3K. Infrastructure and front-end overestimated.

### Adversarial Critique

**Verdict: DOOMED**

The Skeptic identifies three fatal flaws:

1. **SyGuS grammar intractability.** CVC5's reliability boundary is ≤15 atoms, ≤50 constraints. The proposal's own math shows 20–30 atoms after subsumption — at or above the cliff. Per-site decomposition assumes semantic independence between mutation sites, which fails when sites share variables or affect each other's reachability. "Survivorship bias is baked into the gate" because the feasibility study targets exactly the kind of small, arithmetic, loop-free functions where decomposition works.

2. **Certificate generation is a multi-year research project, not 3K LoC.** CVC5's proof-producing mode is experimental and incomplete. Z3's proof format changed three times between 4.8 and 4.12. Composing proofs across the full chain (mutation → grammar → SyGuS → SMT verification) has defeated multiple PhD theses. The "certificates" are probably just "we re-run Z3 and it said SAT."

3. **Theorem 3 extension to bounded loops is probably false.** Loop unrolling generates k correlated error predicates per loop-body mutation. For loops with heap effects or aliasing, the specification gap grows *combinatorially* with k. The "10–20% → 40–50%" coverage estimate is fantasy.

**$10K Consultant Test: PASS (consultant replicates 80%).** Recipe: PIT → Z3 equivalence check → Daikon invariants → mutation-boundary filter → ranked bug reports. ~2,000 lines of Python, $10K for 3 months. Only the formal certificates and Theorem 3 are unreplicable. "The question is whether anyone pays for those."

**Adopter Analysis: 0–1 real adopters.** KeY team needs unbounded verification, not bounded certificates. Google processes millions of mutants/day — SyGuS per-function is computationally absurd. Safety-critical regulators require human-reviewed specs, not auto-generated ones.

**LLM Obsolescence: HIGH for contracts, LOW for certificates.** In 2 years, LLMs will generate "good enough" JML annotations. Certificates survive but demand is unproven.

### Cross-Expert Disagreements

| Issue | Math Assessor | Difficulty Assessor | Skeptic | Adjudication |
|---|---|---|---|---|
| **Certificate generation difficulty** | 90% achievable, engineering | 4/10 difficulty, systems integration | Multi-year research project, may require reimplementing CVC5 proofs | **Skeptic is likely right for full certificates; Math/Difficulty are right for "re-check" certificates.** The real question is what "certificate" means. If it means "Z3 re-verifies and logs the result" → easy. If it means "independently checkable proof chain" → very hard. The proposal conflates these. |
| **Degradation bound achievability** | 40% as stated, needs reformulation | Not directly assessed | Agrees it's vulnerable | **Math Assessor's 40% is generous.** The linear bound g(s) ≥ s is almost certainly false. But the *question* is novel. A weaker result under distributional assumptions may still be publishable. |
| **Overall viability** | B+ math, viable with caveats | 25% P(timeline), high integration risk | DOOMED | **The Skeptic and Difficulty Assessor are more credible here.** The Math Assessor evaluates the math in isolation; the Skeptic and Difficulty Assessor evaluate the whole system. A 25% timeline probability with three unvalidated load-bearing assumptions is effectively doomed for a small team. |

### Surviving Strengths

1. **Theorem 3 itself is real and novel** — all reviewers agree this is the crown jewel regardless of approach.
2. **The certificate concept is a genuine differentiator** from all prior contract inference work (Daikon, SpecFuzzer, LLMs), even if full implementation is harder than acknowledged.
3. **Three-tier fallback strategy** ensures the tool always produces *some* output.

### Fatal or Near-Fatal Issues

1. **FATAL: SyGuS scalability is unvalidated at the grammar sizes this approach requires.** Two of three reviewers view this as likely to fail.
2. **NEAR-FATAL: The degradation bound g(s) ≥ s is almost certainly false**, making Theorem 3 vacuously unsatisfied on all real codebases unless reformulated.
3. **NEAR-FATAL: The 24-week timeline at 25% probability** means this approach almost certainly cannot be completed as scoped.
4. **SERIOUS: No credible adopter base** beyond PL researchers.

---

## Approach B: MutGap — Lightweight Mutation-Guided Bug Finder

### Summary

Approach B skips SyGuS entirely, using fast template-based specification inference with mutation-boundary filtering to transform surviving mutants into ranked, actionable bug reports. It's positioned as a PIT/Maven plugin that runs in CI in minutes. The estimated scope is ~20K LoC over 16 weeks.

### Math Depth Assessment

**Grade: C+ (shallow, mostly ornamental, partially removable)**

1. **Confidence Calibration — 25% achievable.** The Math Assessor is blunt: "This is machine learning dressed as theorem proving." You cannot *prove* calibration — you can only measure it on a test set and hope it generalizes. The "theorem" requires distributional assumptions (training distribution matches deployment) that are empirical bets, not formal guarantees. *Calling this a "theorem" is misleading.*

2. **Compositional Gap Analysis — 55% achievable (trivial version).** The multiplicative composition c_f · c_g assumes independence between callee and caller confidence, which rarely holds. The achievable version (assume independence) is mathematically trivial: P(A ∩ B) = P(A) · P(B). The useful version (account for dependency) is hard and unaddressed.

3. **Subsumption-Aware Filtering — 85% achievable.** The only genuinely load-bearing result. Clean lattice-theoretic argument. Tractable.

**Bottom line:** "Approach B's 'math' is largely ornamental. The paper stands or falls on bug count. Calling the math 'new math required' is misleading; it's 'math we can add to the paper if reviewers want a theory section.'"

### Difficulty Assessment

**Rating: 4/10**

| Subproblem | Claimed Difficulty | Assessed Difficulty | Why |
|---|---|---|---|
| Equivalent mutant triage | Hard | 4/10 | Well-characterized problem with known partial solutions. |
| Distinguishing input generation | Hard | 4/10 | Directed symbolic execution for reachability is solved for QF-LIA. |
| Template-based inference | Hard | 3/10 | Incremental over Daikon + SpecFuzzer. "Mutation-boundary filtering is a nice engineering contribution but not genuinely hard." |

**Timeline: P(16 weeks) = 65%.** Tight but achievable. Most likely overrun: PIT's API doesn't expose fine-grained mutation data, requiring 2–3 extra weeks for a wrapper layer.

**LoC audit:** 20K is plausible and realistic. The 12K "novel" estimate assumes significant PIT API reuse, which may not hold if PIT's internal APIs prove unsuitable.

### Adversarial Critique

**Verdict: MARGINAL**

The Skeptic finds no true fatal flaws but identifies severe mediocrity risks:

1. **Equivalent mutant FP rate will likely exceed the ≤10% target.** TCE + symbolic catches ~60% of equivalents; ~40% remain. On a project with 1,000 survivors and 15% equivalents, that's ~60 false positives in 300 flagged — a 20% FP rate. The statistical confidence scoring is a classifier trained on Defects4J that won't transfer to dissimilar codebases.

2. **"Template-based spec inference that isn't just Daikon" may literally be just Daikon.** Mutation-boundary filtering is a post-hoc filter: `daikon_output | grep -v "violated by no killed mutant"`. The 60% ablation claim is projected, not measured.

**$10K Consultant Test: PASS (consultant replicates 80% in 8 weeks).** ~3,000 lines of Python + Kotlin. The consultant doesn't build confidence calibration or compositional gap analysis — but those are the thin novelty claims.

**Adopter Analysis: 1–2 real adopters.** Teams already running PIT (genuine). Open-source library maintainers (genuine but narrow). CI/CD platform vendors (speculative — "one more scanner").

**Competition Blindspots (SEVERE):**
- **Google's internal mutation triage** — published at ICSE-SEIP 2018, does exactly this with orders of magnitude more training data. The paper *must* cite and compare.
- **Descartes (STAMP project)** — existing PIT plugin classifying surviving mutants as "pseudo-tested." MutGap must demonstrate superiority on Descartes' home turf.
- **PIT's own evolution** — if PIT adds built-in survivor ranking, MutGap's value proposition is absorbed by the tool it depends on.

**LLM Obsolescence: MODERATE.** An LLM-based PIT triager is trivial to build. MutGap survives only if positioned as deterministic CI/CD infrastructure, not as a developer-facing tool.

### Cross-Expert Disagreements

| Issue | Math Assessor | Difficulty Assessor | Skeptic | Adjudication |
|---|---|---|---|---|
| **Is the math necessary?** | No — "the paper stands or falls on bug count" | Not assessed directly | Agrees — "contribution is thin" | **All agree.** The math is ornamental. The paper should lean into empirical results and not oversell the theory. |
| **Is this too easy for PLDI?** | Implicit (C+ grade) | 4/10 difficulty, "half the claimed hard subproblems are well-understood engineering" | "Difficulty at 5/10 confirms this is not hard enough for a top venue" | **Consensus: Yes.** This is FSE/ASE, not PLDI, unless bug count is extraordinary (30+). |
| **Equivalent mutant FP rate** | Not directly assessed | 4/10 — "empirical tuning problem" | 20% FP rate estimate, defeating the ≤10% target | **Skeptic's analysis is more concrete and persuasive.** The 20% FP estimate is grounded in actual numbers. The ≤10% target is aspirational, not achievable with known techniques. |

### Surviving Strengths

1. **Highest feasibility of all three approaches** — all reviewers agree this is most likely to produce a working artifact.
2. **Real adopter base** — PIT users are genuine, and the Maven plugin integration path is credible.
3. **No math blockers** — the tool works with or without its claimed theorems.
4. **Shortest path to publishable result** — if it finds 20+ bugs with maintainer confirmations, the paper justifies itself empirically regardless of theory depth.
5. **Deterministic, reproducible output** — structural advantage over LLM-based alternatives for CI/CD integration.

### Fatal or Near-Fatal Issues

1. **NEAR-FATAL: Contribution may be too thin for a top venue.** All three reviewers converge on this. The $10K consultant replicates 80%. The difficulty is 4/10. The novelty over Daikon + SpecFuzzer + PIT is incremental.
2. **SERIOUS: Competition blindspots are severe.** Google's published approach, Descartes, and PIT's own evolution all threaten to make MutGap redundant. The paper must address all three head-on.
3. **SERIOUS: Equivalent mutant FP rate of ~20% exceeds the usability target.** This doesn't kill the tool but degrades it from "precision-oriented" to "SpotBugs-quality."
4. **RISK: Bug yield may be underwhelming.** If MutGap finds 5–10 bugs instead of 30+, the paper is a B-grade tool paper.

---

## Approach C: MutSpec-Δ — Compositional Contract Construction via WP Differencing

### Summary

Approach C bypasses SyGuS entirely by constructing contracts directly from weakest-precondition differences between original and mutant programs, composed via a novel lattice-walk algorithm. The central claim is that Boolean closure of WP differences is expressively equivalent to mutation-directed SyGuS grammars — meaning "you don't need SyGuS." The estimated scope is ~40K LoC over 20 weeks.

### Math Depth Assessment

**Grade: A- (deep, novel, mostly load-bearing)**

The Math Assessor identifies this as the approach with the most genuine math:

1. **WP-Composition Completeness Theorem — 55% achievable.** The core issue: WP differences give per-site formulas; Boolean closure gives Boolean combinations of per-site formulas; but some QF-LIA specs are cross-site relational (e.g., `\result == a + b`). The Math Assessor suspects "yes for per-site properties, no for cross-site relational invariants." Full completeness: 40%. Per-site completeness + gap characterization: 60–70%. *Even the partial result is novel and interesting.*

2. **Lattice-Walk Termination and Optimality — 80% achievable.** Standard lattice-fixpoint theory. Monotone decrease in a finite lattice guarantees termination. O(n² · SMT(n)) complexity is analyzable. "The algorithm is novel; the proof techniques are standard."

3. **Simplification-Soundness — 75% achievable.** The key subtlety: abstraction (replacing clauses with simpler entailing formulas) may not preserve 1:1 mutation provenance. The "derived-from" relation must be carefully defined for abstracted clauses. Tractable but requires care.

**Bottom line:** "If the WP-Composition Completeness Theorem holds, it's the strongest mathematical contribution of the three approaches. Even if it needs qualification, the approach introduces a genuinely different way to do contract synthesis."

### Difficulty Assessment

**Rating: 6.5/10**

| Subproblem | Claimed Difficulty | Assessed Difficulty | Why |
|---|---|---|---|
| WP differencing at scale | Hard | 6/10 | Sharing/incrementality is real engineering difficulty; base WP for loop-free is textbook. |
| Lattice-walk algorithm | Novel | 5/10 | Algorithmically novel but implementable by anyone with lattice theory + SMT API comfort. |
| WP-Composition Completeness | Open research | **8/10** | "The only genuinely open research question across all three approaches." |

**Timeline: P(20 weeks) = 45%.** Achievable if the completeness theorem resolves by Week 6. Most likely failure mode: WP engine for Java (handling value/reference semantics, autoboxing, exception propagation) is harder than expected. JayHorn, which does WP for a Java subset, is ~30K LoC and took years.

**LoC audit:** 40K is reasonable and in line with comparable tools. The WP engine (like Approach A) is underestimated — budget 6–10K, not 3K.

### Adversarial Critique

**Verdict: MARGINAL (leaning DOOMED)**

The Skeptic's attacks are the sharpest here:

1. **WP-Composition Completeness is almost certainly false.** The Skeptic provides a concrete counterexample: for `f(a,b) = if (a > b) then a else b` (integer max), WP differences give precondition-space formulas like `{a == b}` and `{a > b}`. Their Boolean closure gives input predicates like `a < b`. But what you *want* is `result == max(a, b)` — a relational postcondition connecting output to inputs. "WP differences are precondition-space formulas, not relational postconditions. To get relational postconditions, you need to synthesize a function — which is exactly what SyGuS does." If the completeness theorem fails, MutSpec-Δ becomes a strictly weaker version of Approach A.

2. **The lattice walk's abstraction step IS SyGuS in disguise.** The abstraction step asks "is there a simpler formula that implies this conjunction?" — this is a synthesis problem. The proposal claims to bypass SyGuS, but the abstraction step reintroduces it without calling CVC5. Skip abstraction → get correct but unreadable contracts. Include abstraction → reintroduce the SyGuS problem you claimed to avoid.

3. **Traceability breaks under simplification.** Full simplification produces many-to-many mappings between clauses and mutations. Traceability survives only if simplification is identity (no simplification) or restricted to subsumption elimination (which doesn't actually simplify). "You can't have both."

**$10K Consultant Test: PARTIALLY PASSES (consultant replicates 70%).** Recipe: PIT → JBMC for WP computation → negate WP differences → conjoin → Z3 simplify → verify → output JML. The consultant can't build the lattice-walk or prove the completeness theorem. But if Z3's simplifier produces comparable-quality contracts, the lattice walk is "academic overhead."

**Adopter Analysis: 0 real adopters.** Safety-critical auditors don't use any automated contract inference. "MutSpec-Δ solves a technical problem for users who don't exist yet." Researchers citing your work are not adopters.

**Competition Blindspots:**
- **JBMC** already computes WP for Java bytecode and interfaces with Z3. Building a mutation-to-contract pipeline on top of JBMC is an obvious integration project that achieves MutSpec-Δ's core pipeline without a custom WP engine.
- **Infer's Pulse** handles heap, aliasing, and interprocedural analysis more scalably than WP differencing.
- **Strongest postcondition computation** (KLEE, SAGE, Pex) computes the same input-output relationships from the opposite direction.

**LLM Obsolescence: LOW for theory, HIGH for tool.** The theorems survive indefinitely; the practical "give me contracts from mutations" tool is vulnerable.

### Cross-Expert Disagreements

| Issue | Math Assessor | Difficulty Assessor | Skeptic | Adjudication |
|---|---|---|---|---|
| **WP-Composition Completeness achievability** | 55% (partial version 60–70%) | 8/10 difficulty (open research) | "Almost certainly false" — concrete counterexample provided | **The Skeptic's counterexample is compelling but may not be decisive.** The max(a,b) example shows WP differences can't capture relational postconditions like `result == max(a,b)`. But the Math Assessor's nuanced view (per-site completeness holds, cross-site relational invariants are the gap) may be the correct framing. The partial result — completeness for per-site properties + precise gap characterization — is likely achievable and still novel. **Adjudication: The full theorem probably fails. A valuable partial result probably holds.** |
| **Is the lattice walk hiding SyGuS?** | Not flagged | Rates abstraction step 5/10 | "The abstraction step IS SyGuS wearing different clothes" | **The Skeptic raises a genuine point that neither other reviewer caught.** However, the severity depends on what "simplification" means. If restricted to entailment-based redundancy removal + known algebraic simplifications (not open-ended synthesis), it's not truly SyGuS. If it requires finding arbitrary simpler formulas, it is. **Adjudication: The abstraction step must be carefully scoped to avoid reintroducing synthesis. The proposal underspecifies this.** |
| **Traceability survivability** | Simplification-soundness 75% achievable | Not directly assessed | "You can't have both" traceability and simplification | **The Skeptic is largely right.** Full simplification destroys 1:1 traceability. The Math Assessor acknowledges the "derived-from" relation for abstracted clauses needs careful definition. The practical compromise: many-to-few traceability (each simplified clause traced to a *set* of mutations) is achievable and still useful, but weaker than the proposal claims. **Adjudication: Traceability survives in degraded form. The proposal overstates it.** |
| **JBMC as competition** | Not assessed | Not assessed | "Why not just use JBMC for WP computation?" | **The Skeptic is right that this deserves a direct answer.** JBMC operates on bytecode, not source-level IR — the representation mismatch may make it unsuitable for mutation-to-WP bridging. But the proposal should explicitly justify building a custom WP engine vs. adapting JBMC. |

### Surviving Strengths

1. **The WP-Composition Completeness *question* is genuinely novel** — even if the theorem fails, the precise characterization of the gap is publishable. All reviewers agree attempting it generates value either way.
2. **The lattice-walk algorithm is a new synthesis paradigm** — deterministic, analyzable, with bounded complexity. Novel even if the completeness theorem fails.
3. **Deterministic performance** — no solver non-determinism, time proportional to mutant count. A real practical advantage over SyGuS.
4. **The mitigations are sensible** — three-pronged: restricted fragment first, gap characterization, hybrid fallback. The Skeptic acknowledges the mitigations but argues they transform C into "a weaker version of A."
5. **No SyGuS scalability cliff** — eliminates Approach A's most fatal risk.

### Fatal or Near-Fatal Issues

1. **NEAR-FATAL: WP-Composition Completeness likely fails for relational postconditions.** The Skeptic's counterexample is concrete: precondition-space formulas cannot express input-output relations. If the gap is large and uncharacterizable, the approach loses its central pitch.
2. **SERIOUS: The abstraction step may reintroduce synthesis.** The proposal underspecifies what "finding a simpler entailing formula" means. If it's open-ended, it's SyGuS without the solver.
3. **SERIOUS: Traceability is overstated.** Full simplification destroys 1:1 provenance. The practical version is weaker than claimed.
4. **SERIOUS: No credible adopter base.** The Skeptic estimates "dozens, not thousands" of potential users.
5. **RISK: WP engine underestimation.** All reviewers agree the WP engine for Java semantics is undercosted (3K → 6–10K LoC). This is the schedule bottleneck.

---

## Synthesis of Cross-Cutting Themes

These issues affect ALL approaches and must be addressed regardless of which is chosen.

### 1. The PIT ↔ Symbolic Reasoning Gap (Integration Risk 8/10)

All three approaches need mutation error predicates — symbolic formulas characterizing behavioral divergence. PIT produces kill matrices as `(mutant, test, killed/survived)` triples. **There is no existing bridge.** The choice is: reverse-engineer PIT's bytecode mutations for symbolic extraction (fragile), or build a parallel source-level mutation engine (expensive). This decision alone determines 3–6 weeks of schedule. No approach adequately prices it in.

### 2. WP Engine LoC Estimates Are Universally Too Low

All three reviewers agree: budget 6–10K LoC for a correct WP engine over even the loop-free Java QF-LIA fragment, not 3K. Java's reference semantics, autoboxing, exception propagation, and array bounds make WP computation far harder than textbook Dijkstra calculus. JBMC's Java-to-SMT encoding is ~15K LoC of C++. Either MutSpec is encoding a much smaller fragment (plausible for QF-LIA loop-free) or this is severely underestimated.

### 3. "Mutation Testing Is Already Niche"

PIT's "10,000+ monthly downloads" is misleading — CI systems re-download on every build. Actual industry adoption of mutation testing is thin. If only ~500 teams worldwide seriously use mutation testing in CI, the adopter pool for *any* MutSpec variant is small. All approaches' value propositions assume a larger base than exists.

### 4. Theorem 3 Is the Shared Crown Jewel (and Shared Risk)

All approaches depend on Theorem 3 (ε-completeness). If it fails, the project loses its intellectual core. The Math Assessor gives 65% achievability. The Skeptic raises a specific boundary-case attack (multi-site property interactions). Both agree: prioritize this proof above all else.

### 5. The Evaluation Design Is Potentially Circular

The evaluation plan measures MutSpec on benchmarks chosen to favor its strengths (QF-LIA, loop-free, arithmetic code). A fair evaluation must include string-heavy, OOP-heavy, and complex-control-flow code where MutSpec's restrictions bite hardest. No reviewer found evidence that benchmark selection bias was addressed.

### 6. The Base Theory Presentation Is Inflated

The Math Assessor recommends compressing Theorems 1, 2, and 4 into "Definitions and Framework" and saving theorem numbering for Theorems 3, 5, and approach-specific results. Presenting definitional content as "theorems" wastes reviewer goodwill.

### 7. LLM Threat Is Real but Differentiable

All approaches face LLM competition for the "generate contracts/specs" use case. In 2 years, LLMs will produce "good enough" annotations for most developer use cases. What survives: formal certificates (Approach A), deterministic reproducible CI/CD integration (Approach B), and novel theoretical contributions (Approach C). Each approach must lean into what LLMs *can't* do.

### 8. The $10K Consultant Replicates 70–80% of All Approaches

This is damning. A competent grad student with PIT + Z3 + Daikon + Python replicates most practical value in 2–3 months. The *unreplicable* contributions are: Theorem 3 and its extensions, the WP-Composition Completeness question, the lattice-walk algorithm, and formal certificates. The paper must center on these, not on the tool's practical output.

---

## Revised Scores Post-Debate

After incorporating all three reviewers' assessments and adjudicating disagreements, the following table updates the original scores.

| Dimension | A: MutSpec-Complete | B: MutGap | C: MutSpec-Δ |
|---|---|---|---|
| **Value** | 8 → **6** | 9 → **7** | 7 → **6** |
| **Difficulty** | 8 → **7** | 5 → **4** | 7 → **6.5** |
| **Potential** | 7 → **5** | 6 → **6** | 8 → **7** |
| **Feasibility** | 5 → **3** | 9 → **7** | 7 → **5** |

**Rationale for changes:**

- **Approach A Value 8→6:** Adopter base collapses under scrutiny (Skeptic). Certificate demand is undemonstrated. SyGuS scalability at relevant grammar sizes is unvalidated.
- **Approach A Feasibility 5→3:** 25% P(24-week timeline) per Difficulty Assessor. Three unvalidated load-bearing assumptions. Certificate generation severely underscoped.
- **Approach B Value 9→7:** "10,000 PIT users" is inflated (Skeptic). Competition from Google's internal triage, Descartes, and PIT's own evolution. Template inference is "just Daikon" risk.
- **Approach B Feasibility 9→7:** Equivalent mutant FP rate likely exceeds target (Skeptic's 20% estimate). PIT API may not expose fine-grained data. Still the most feasible, but not a sure thing.
- **Approach C Potential 8→7:** WP-Composition Completeness likely fails for the full fragment (Skeptic's counterexample). Traceability overstated. But partial results are still novel.
- **Approach C Feasibility 7→5:** WP engine is undercosted by 2–3x. 45% P(20-week timeline). Completeness theorem may not resolve quickly.

---

## Recommendation for the Synthesis Editor

### What to Build

The synthesis should combine elements from all three approaches in a specific order, governed by the reviewers' consensus:

**1. Shared Foundation (non-negotiable, from all approaches):**
- PIT integration layer with symbolic mutation data extraction — address the PIT ↔ symbolic reasoning gap *first*, as all reviewers identify this as the #1 engineering risk.
- Equivalent-mutant filtering stack (TCE + bounded symbolic equivalence via Z3) — needed by all approaches.
- Theorem 3 proof effort — prioritize this above all approach-specific math. If it fails, pivot the project.

**2. Primary Approach: Start with B's infrastructure, attempt C's theory (recommended strategy confirmed)**

All three reviewers independently validate the approaches document's recommended strategy:
- The Math Assessor recommends "attempt C's theorem in parallel with B's infrastructure."
- The Difficulty Assessor says "B first, then C, defer A" is "sound."
- The Skeptic says "Plan for B; hope for C; don't bet on A."

**3. From Approach B, take:**
- The bug-finding pipeline as the guaranteed deliverable (ranked SARIF reports from surviving mutants).
- The Maven plugin integration path (zero-adoption-friction delivery).
- Drop: the "confidence calibration theorem" framing — present it as "principled heuristic scoring" instead (per Math Assessor). Don't dress ML as formal methods.

**4. From Approach C, take:**
- The WP-Composition Completeness *question* as the primary theoretical investigation. Attempt the restricted fragment (single-expression functions) first. If the full theorem fails, characterize the gap — this is publishable either way.
- The lattice-walk algorithm as the synthesis backend *if* the completeness theorem (or a useful partial version) holds.
- Carefully scope the abstraction step to avoid reintroducing synthesis (per Skeptic's critique). Restrict to entailment-based redundancy removal + algebraic simplification, not open-ended "find a simpler formula."
- Downgrade the traceability claim to many-to-few mapping (clause → set of mutations), not 1:1.

**5. From Approach A, take:**
- Nothing in the first phase. Defer SyGuS entirely until Gate 1 empirically validates scalability.
- If C's completeness theorem fails *and* the gap is large, consider SyGuS as a targeted fallback (C's hybrid mitigation) — but only for the specific functions where WP composition is insufficient.
- The certificate *concept* (not the implementation) as future work. Do not attempt full certificate generation in the initial project.
- The degradation bound *question* (not the g(s) ≥ s conjecture) — reformulate under distributional assumptions before attempting.

**6. What to cut entirely:**
- Theorem 3 extension to bounded loops — the Skeptic and Math Assessor both consider it likely false as stated. Defer to future work.
- Full certificate generation chain — "multi-year research project" per Skeptic.
- Compositional gap analysis (multiplicative confidence composition) — trivial under independence, hard without it, practically vacuous for deep call chains.
- The inflated base theory presentation — compress Theorems 1, 2, 4 to definitions.

### Target Venue

- If bug count ≥ 30 with maintainer confirmations AND the completeness theorem (or partial version) holds: **PLDI** (theory + practice).
- If bug count ≥ 20 but completeness theorem fails: **OOPSLA/FSE** (tool + empirical).
- If bug count < 15: **ASE/ISSTA** (tool paper).

### The Honest Summary

The project's real contributions, stripped of marketing, are:
1. **Theorem 3** — if proved, this alone justifies a publication.
2. **The WP-Composition Completeness question** — novel regardless of answer.
3. **The lattice-walk algorithm** — new synthesis paradigm.
4. **Real bugs found in well-tested Java libraries** — the empirical contribution that makes reviewers care.

Everything else is either known techniques applied to a new domain, engineering, or ornamental theory. The synthesis should center on these four contributions and be honest about everything else.

---

*Compiled from reviews by: Math Depth Assessor, Difficulty Assessor, Adversarial Skeptic*
*Adjudication by: Team Lead*
