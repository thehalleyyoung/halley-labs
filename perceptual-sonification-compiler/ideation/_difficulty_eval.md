# Difficulty Evaluation: Perceptual Sonification Compiler

**Evaluator**: Difficulty Assessor  
**Date**: 2026-03-08  
**Inputs**: `approaches.md` (3 approaches), `depth_check.md` (V4/D6/BP5/L7.5, CONDITIONAL CONTINUE)

---

## Approach A: Graded Comonadic Sonification Calculus

### 1. Hard Subproblem Audit

**Claimed hard subproblem 1: Designing the resource semiring.**
- **Is it actually hard?** YES — genuinely hard. The core tension is real: psychoacoustic masking interactions are context-dependent (the "cost" of stream A changes when stream B enters), which fundamentally conflicts with the semiring requirement that ⊕ be a binary function of its operands alone. This is not a routine implementation problem; it's an open question whether the algebraic structure even exists for this domain. A competent engineer *cannot* solve this in a week because it's a mathematical existence question, not an engineering task.
- **Hardness in the right place?** YES. This IS the novel contribution. If the semiring works, the paper writes itself. If it doesn't, Approach A fails.
- **Hidden problems?** The mitigation strategy (ε-approximate semiring with conservative over-approximation fallback) is the real likely outcome. The hidden difficulty is that the ε-approximate version may be so conservative it rejects >50% of valid configurations, making the type system practically useless. The gap between "theoretically sound" and "practically useful" approximate semiring is under-examined.

**Claimed hard subproblem 2: Non-local constraint propagation under composition.**
- **Is it actually hard?** MODERATE. This is hard in the context of graded types (because the whole point of graded types is that costs compose algebraically, and here they don't cleanly). But if you step back, it's a constraint propagation problem over a small graph (≤16 nodes, 24 Bark bands). A competent systems engineer could build a working constraint propagator in 2–3 weeks. The hardness is in making it *fit the type-theoretic framework*, not in making it *work*.
- **Hardness in the right place?** PARTIALLY. It's hard because of the commitment to graded types. Under a different formulation (Approach B's refinement types), this becomes routine SMT encoding. The difficulty is self-imposed by the approach choice—which is fine if the type-theoretic insight is the contribution, but it's worth noting.

**Claimed hard subproblem 3: Subtyping with non-convex feasibility regions.**
- **Is it actually hard?** NO, not really. Non-convex feasibility regions in 4–6 dimensions with ≤16 streams are computationally tractable by brute evaluation. The claim that "standard refinement-type subtyping is syntactic or SMT-decidable over linear arithmetic; here, feasibility checking involves non-linear, piecewise functions" overstates the problem. With 24 Bark bands and ≤16 streams, you can evaluate the spreading function in microseconds. The subtyping check is just: evaluate the piecewise function, compare thresholds. A competent engineer solves this in days, not weeks.
- **Hidden problems?** The real hidden difficulty: making the subtyping *compositional*. It's not that checking one configuration is hard—it's that the subtyping relation must be a preorder that interacts correctly with the resource semiring's ⊕. Getting the algebra right is hard; evaluating the functions is not.

**Not mentioned — hidden hard problems:**
1. **Proof engineering for the soundness theorem.** The logical-relations argument adapted to the resource semiring is mentioned as a one-liner. In practice, logical relations proofs for novel type systems take weeks-to-months of careful work, especially when the denotational semantics involves continuous domains (real-valued psychoacoustic parameters). This is easily the most labor-intensive formal task and it's barely discussed.
2. **The DSL design itself.** Making a DSL that is (a) expressive enough for real sonification scenarios, (b) simple enough to type-check efficiently, and (c) natural enough that users actually write programs in it, is a design problem that can easily consume 3–4 weeks of iteration. It's mentioned as "routine" but DSL design is notoriously fiddly.
3. **Empirical calibration of the resource budget ceiling r_max.** The type system needs a concrete "perceptual capacity" vector. Where does this come from? From psychoacoustic literature? From model predictions? The choice of r_max determines whether the type system is useful or pathologically permissive/restrictive. This is a tricky domain-knowledge problem hiding inside a PL artifact.

### 2. Architecture Assessment

- **Sound?** Conditionally. The architecture is beautiful *if* the semiring works. If it doesn't (40% probability by their own estimate), the architecture degenerates to Approach B with wasted effort on comonadic machinery.
- **Riskiest integration point?** The interface between the algebraic type system and the actual psychoacoustic model evaluations. The semiring operates over abstract resource vectors; the model evaluations operate over concrete audio parameters. Ensuring these two layers agree (the resource vector is a faithful abstraction of the model's predictions) is a subtle soundness concern that could eat weeks of debugging.
- **Simplification?** Drop the comonadic framing entirely and use a simpler resource-indexed type system (e.g., bounded linear types à la Dal Lago & Hofmann). This loses the "perception as coeffect" narrative but keeps compositional resource tracking. Whether the narrative is worth the 40% failure risk is a judgment call.
- **Minimum viable system?** A type checker for a flat (non-compositional) stream specification language with masking/JND checking. This proves the psychoacoustic model integration but not the compositionality contribution. Add single-level composition (merge two checked specs) for the compositional story. ~25K novel LoC.

### 3. Build-vs-Buy Analysis

| Subsystem | Build or Buy | Novel % | Realistic Novel LoC |
|-----------|-------------|---------|---------------------|
| DSL parser + AST | Buy (pest PEG parser) + custom AST | 40% novel (AST design) | ~4,000 |
| Graded type checker | BUILD — this is the paper | 90% novel | ~8,000 |
| Resource semiring impl | BUILD — core math | 95% novel | ~2,000 |
| Psychoacoustic models | Implement from published formulas | 20% novel (integration, not formulas) | ~1,500 |
| Optimizer (I_ψ) | Moderate novelty, domain-specific B&B | 50% novel | ~5,000 |
| Audio renderer | BUY (cpal + standard patterns) | 5% novel | ~500 |
| Evaluation framework | Standard eval harness + domain metrics | 30% novel (metrics) | ~6,000 |
| Standard library | Routine | 5% novel | ~500 |
| CLI / tooling | Routine | 5% novel | ~400 |
| **TOTAL** | | | **~28,000 genuinely novel** |

The 48–60K "novel LoC" claim is inflated by ~50%. Realistic novel LoC: ~28K. Total with glue: ~65–75K.

### 4. "Can a PhD Student Build This?" Test

- **Person-months**: 8–12 months for a strong PL PhD student with some audio background. The semiring formalization alone could take 2–3 months of false starts.
- **Specialized expertise**: (a) Graded/coeffect type theory (niche PL subfield, ~20–30 active researchers worldwide), (b) Psychoacoustics at the level of understanding masking models (not a standard PL skill), (c) Rust systems programming for the audio pipeline.
- **Biggest "I'm stuck" risk**: Months 1–3, discovering the semiring doesn't work. The student would need to recognize this early and pivot to the ε-approximate version before sinking too much time into exact formalization. A student without strong algebra intuition could waste 4+ months here.

### 5. Difficulty Score: **7/10**

The resource semiring design is genuinely research-level. The type-theoretic formalization (soundness proof, logical relations) is hard and novel. But the 40% probability of needing to fall back to Approach B, and the fact that most of the codebase is routine audio/compiler engineering, pull the score down. The crown jewel is hard; the system around it is not.

---

## Approach B: Liquid Sonification — SMT-Backed Refinement Types

### 1. Hard Subproblem Audit

**Claimed hard subproblem 1: Custom SMT theory for psychoacoustic arithmetic.**
- **Is it actually hard?** MODERATE-TO-HARD. Building a Z3/CVC5 theory plugin is genuinely non-trivial engineering — the API is poorly documented, the interaction protocol between theories is subtle, and debugging theory solvers is painful. But the *mathematical* content (Bark-scale conversion, spreading functions) is all published formulas. The hard part is not the psychoacoustics or the math; it's the SMT engineering. An experienced SMT developer could do this in 3–4 weeks. A PL PhD student with no SMT internals experience: 6–8 weeks with significant debugging.
- **Hardness in the right place?** PARTIALLY. The custom theory is a necessary enabler, not the intellectual crown jewel. The crown jewel is the refinement type system itself. Spending 6–8 weeks on SMT plumbing is a lot of time on infrastructure vs. contribution.
- **Hidden problems?** The piecewise-linear approximation fallback (Section "Mitigation strategy" point 1) is so practical that it arguably makes the custom theory unnecessary. If you approximate Bark conversion with 24 linear segments, you get QF_LRA, which Z3 handles in milliseconds with zero custom theory code. The question is whether the paper *needs* the custom theory or whether the piecewise-linear version is sufficient. If piecewise-linear works, the "custom SMT theory" contribution evaporates — it's engineering for the sake of engineering.

**Claimed hard subproblem 2: Non-local constraint re-checking.**
- **Is it actually hard?** NO. For k ≤ 16 streams and 24 Bark bands, the O(k²) conjunctive SMT query is tiny. Z3 solves thousands of QF_LRA constraints in milliseconds. The "incremental checking" optimization (Result B3) is algorithmically clean but practically unnecessary — the brute-force approach is fast enough. This is presented as hard but is straightforward engineering.
- **Hardness in the right place?** NO. The incremental algorithm is a nice optimization, not a hard problem. The paper could simply use the conjunctive query.

**Claimed hard subproblem 3: Actionable diagnostics from UNSAT cores.**
- **Is it actually hard?** MODERATE. Extracting minimal UNSAT cores is a built-in Z3 feature. Mapping SMT variables back to domain-specific explanations ("streams 3 and 5 overlap in Bark band 12") requires bookkeeping but is not algorithmically hard. The UX challenge (making error messages actually useful) is real but it's a design problem, not a research problem.
- **Hardness in the right place?** NO. This is UX engineering, not PL contribution.

**Claimed hard subproblem 4: OMT for I_ψ maximization.**
- **Is it actually hard?** MODERATE. Branch-and-bound with an SMT feasibility oracle is a known technique (e.g., νZ, OptiMathSAT). The domain-specific instantiation requires custom bounding functions based on psychoacoustic models. This is competent optimization engineering — a strong student handles it in 2–3 weeks.
- **Hardness in the right place?** PARTIALLY. The optimizer matters for practical value but is not the paper's PL contribution.

**Not mentioned — hidden hard problems:**
1. **The refinement type soundness proof is harder than it looks.** The proof sketch ("standard Liquid-type technique") glosses over a key difficulty: Liquid Haskell's soundness relies on the decidability and soundness of the underlying SMT theory. Here, the underlying theory is custom and involves transcendental functions. The soundness argument must chain through: (a) δ-soundness of the custom theory, (b) propagation of δ into the refinement type soundness, (c) showing δ is small enough to not matter perceptually. This chain is non-trivial and not discussed.
2. **Performance engineering for interactive use.** The 2-second target for 8-stream type-checking is aggressive if using the full non-linear theory. The piecewise-linear fallback makes it trivial, but then the custom theory claim weakens. There's a tension between "impressive custom theory" and "actually fast."

### 2. Architecture Assessment

- **Sound?** YES. Refinement types + SMT is a proven architecture (Liquid Haskell, Flux for Rust). The domain-specific extension is well-motivated.
- **Riskiest integration point?** The custom SMT theory's interaction with Z3's internals. Theory plugins are fragile, and bugs in theory solvers are notoriously hard to diagnose. The piecewise-linear fallback de-risks this significantly.
- **Simplification?** Use piecewise-linear approximation from day one, skip the custom theory, and frame the contribution as "refinement types for psychoacoustic verification" rather than "custom SMT theory for psychoacoustics." This loses a claimed contribution but gains 6+ weeks of development time and removes the riskiest component.
- **Minimum viable system?** A type checker that takes a flat stream specification, encodes constraints in QF_LRA (piecewise-linear Bark approximation), and calls Z3. Report SAT/UNSAT with basic diagnostics. ~18K novel LoC. This already proves the concept.

### 3. Build-vs-Buy Analysis

| Subsystem | Build or Buy | Novel % | Realistic Novel LoC |
|-----------|-------------|---------|---------------------|
| DSL parser + AST | Buy (pest) + custom | 40% | ~4,000 |
| Refinement type checker | BUILD — but largely adapts Liquid patterns | 60% novel | ~6,000 |
| Custom SMT theory | BUILD — if needed | 70% novel | ~3,000 |
| SMT integration layer | Moderate (Z3 bindings exist) | 30% | ~2,000 |
| Psychoacoustic models | Published formulas | 20% | ~1,500 |
| Optimizer (OMT hybrid) | Moderate | 40% | ~3,500 |
| UNSAT diagnostic extraction | Moderate | 50% | ~2,000 |
| Audio renderer | BUY (cpal) | 5% | ~500 |
| Evaluation framework | Standard + domain metrics | 30% | ~6,000 |
| Standard library / CLI | Routine | 5% | ~800 |
| **TOTAL** | | | **~29,000 genuinely novel** |

Similar novel LoC to Approach A, but with lower risk. The custom SMT theory (3K LoC) is the only component that might not be needed.

### 4. "Can a PhD Student Build This?" Test

- **Person-months**: 5–8 months for a strong PL PhD student familiar with SMT solvers. The refinement type framework is well-charted territory; the domain application is the novelty.
- **Specialized expertise**: (a) SMT solver internals (for the custom theory; avoidable with piecewise-linear fallback), (b) Refinement type systems (well-taught in PL PhD programs), (c) Basic psychoacoustics (learnable in 2 weeks from Moore's textbook).
- **Biggest "I'm stuck" risk**: Getting the custom SMT theory to be both correct AND fast. Debugging theory solvers is a black art. Mitigation: the piecewise-linear fallback means the student is never truly stuck — just building a less impressive version.

### 5. Difficulty Score: **5/10**

The refinement type framework is well-understood machinery applied to a new domain. The custom SMT theory is genuine engineering difficulty but likely unnecessary (piecewise-linear fallback works). The non-local constraints are claimed as hard but are computationally trivial at the relevant scale (k ≤ 16). The novelty is in the *application* — "nobody has type-checked sonifications before" — not in the *technique*. This is a solid integration project with moderate novelty, not a research frontier.

I score this lower than the approaches document's self-assessment (7/10) because the depth check already noted that "each domain contribution is at moderate depth" and I agree — once you accept the piecewise-linear approximation, most claimed hard problems dissolve.

---

## Approach C: SoniSynth — Program Synthesis from Perceptual Specifications

### 1. Hard Subproblem Audit

**Claimed hard subproblem 1: Enormous synthesis search space.**
- **Is it actually hard?** NO — or rather, the hardness is standard and well-solved. Searching a 10¹²–10³⁰ space with constraint propagation and branch-and-bound is what every constraint solver does. The psychoacoustic cost functions provide excellent pruning heuristics (if two streams are in the same Bark band and too close in frequency, prune immediately). With 24 Bark bands providing natural decomposition and k ≤ 12, the effective search space after constraint propagation is manageable. This is routine constraint satisfaction programming (CSP), not research.
- **Hardness in the right place?** NO. Claiming "the space is 10³⁰" is technically true but misleading — the *effective* space after constraint propagation is orders of magnitude smaller. Any CSP textbook covers this.

**Claimed hard subproblem 2: The synthesis specification language.**
- **Is it actually hard?** MODERATE. The language design is a creative challenge — mapping qualitative perceptual goals ("distinguishable," "not fatiguing") to quantitative constraints requires domain expertise and careful formalization. But this is a design exercise, not a research problem. Once you decide that "distinguishable" means d'_model ≥ 2.0, the specification language is a straightforward formalization of psychoacoustic constraints. The hard part (which psychoacoustic thresholds to use) is a domain question, not a PL question.
- **Hardness in the right place?** PARTIALLY. The spec language is the user-facing contribution, which matters for value, but it's not deep PL or synthesis theory.

**Claimed hard subproblem 3: Correct-by-construction guarantees.**
- **Is it actually hard?** NO. Option (b) — generate-and-verify with a sound verifier — is trivially correct if the verifier is correct. This reduces to the type-checking problem of Approaches A/B, which Approach C explicitly delegates. Option (a) — constructive synthesis — is claimed as "more novel" but the greedy packing algorithm described in Result C2 is a standard greedy algorithm with a standard approximation guarantee. The proof technique (submodularity + matroid constraints) is textbook.
- **Hardness in the right place?** NO. The "correct-by-construction" framing is packaging standard constraint satisfaction as something deeper.

**Claimed hard subproblem 4: Pareto-optimal synthesis.**
- **Is it actually hard?** NO. Multi-objective optimization over a discrete parameter space with clear cost functions is a well-solved problem (NSGA-II, MOEA/D, or even weighted-sum scalarization for small k). With k ≤ 12 streams and 4–5 parameters per stream, even exact Pareto front enumeration is feasible via structured enumeration. This is an afternoon of coding with an existing multi-objective optimization library, not a research problem.

**Not mentioned — hidden hard problems:**
1. **The specification language's expressiveness vs. decidability tradeoff.** If users can specify arbitrary relationships between streams ("stream A should sound 'warmer' than stream B"), the spec language needs a semantics for subjective timbral qualities. This is genuinely hard and under-examined. The approach implicitly restricts to quantitative specs (d' thresholds, frequency ranges), which sidesteps the hard problem.
2. **The "negotiation" UX.** When a spec is unrealizable, the specification lattice (Result C3) proposes computing the nearest realizable relaxation. This is a set-cover/minimum-relaxation problem that could be NP-hard depending on the lattice structure. The lattice construction itself is under-specified — how do you define the partial order over all possible spec relaxations? This is likely harder than claimed.

### 2. Architecture Assessment

- **Sound?** YES. Synthesis-from-spec is a clean architecture. The separation between specification, synthesis, and verification is well-motivated.
- **Riskiest integration point?** The quality of the synthesized output. The greedy packing algorithm may produce configurations that are technically feasible but perceptually unpleasant (e.g., all streams clustered at frequency extremes). There's no aesthetic quality model, and "perceptually optimal" ≠ "aesthetically acceptable." This is a gap the evaluation may not catch because d'_model doesn't measure pleasantness.
- **Simplification?** Remove the specification lattice and Pareto optimization. Make it a single-objective synthesizer: maximize minimum pairwise d'_model subject to constraints. This handles 80% of use cases with 40% of the code.
- **Minimum viable system?** A greedy stream placer that takes (k, d'_min) and outputs a stream parameter assignment. Verify with a simple constraint checker. ~12K novel LoC. This is a weekend hackathon project at the core.

### 3. Build-vs-Buy Analysis

| Subsystem | Build or Buy | Novel % | Realistic Novel LoC |
|-----------|-------------|---------|---------------------|
| Specification language parser | Build (small DSL) | 50% | ~2,000 |
| Realizability checker | Build, but it's CSP | 30% | ~2,500 |
| Greedy packing synthesizer | Build — domain-specific algorithm | 60% | ~3,000 |
| Specification lattice | Build — moderate novelty | 50% | ~2,500 |
| Pareto front computation | BUY (existing MO libraries) or simple impl | 20% | ~1,000 |
| Verification backend (type-checking) | Reuse from Approach B or simple checker | 20% | ~2,000 |
| Psychoacoustic models | Published formulas | 20% | ~1,500 |
| Audio renderer | BUY (cpal) | 5% | ~500 |
| Evaluation framework | Standard + synthesis-specific metrics | 30% | ~5,000 |
| Standard library / CLI | Routine | 5% | ~800 |
| **TOTAL** | | | **~21,000 genuinely novel** |

Lowest novel LoC of the three. Much of the "synthesis" is standard constraint solving with domain-specific heuristics.

### 4. "Can a PhD Student Build This?" Test

- **Person-months**: 3–5 months for a strong PL PhD student. The algorithms are standard; the domain application is the main work. A student with constraint programming experience could move very fast.
- **Specialized expertise**: (a) Basic constraint satisfaction / optimization (standard CS), (b) Basic psychoacoustics (learnable in 2 weeks), (c) No deep PL theory required.
- **Biggest "I'm stuck" risk**: Discovering that the greedy algorithm produces perceptually bad results for realistic non-uniform specs, and the local search refinement doesn't help. This is an empirical quality concern, not a theoretical stuck-point. The student can always ship *something*; the question is whether it's *good enough*.

### 5. Difficulty Score: **4/10**

The NP-completeness of realizability is a clean theoretical result, but the reduction from graph coloring is straightforward (the approaches document basically gives the proof in one sentence). The greedy packing algorithm with submodularity guarantees is textbook. The specification language is a design exercise. The Pareto optimization is off-the-shelf. The only genuinely novel aspect is applying synthesis to the psychoacoustic domain — and even there, the techniques are standard CSP/optimization in a new costume.

This is a well-motivated engineering project with a nice framing, not a research-difficulty challenge.

---

## 6. Cross-Approach Comparison

### Difficulty Ranking

| Rank | Approach | Difficulty Score | Nature of Difficulty |
|------|----------|-----------------|---------------------|
| 1 | **A: Graded Comonadic** | **7/10** | Mathematical/type-theoretic: does the semiring exist? |
| 2 | **B: Liquid Sonification** | **5/10** | Engineering/integration: can you build a good SMT theory? |
| 3 | **C: SoniSynth** | **4/10** | Domain application: can you package CSP for psychoacoustics? |

### Honest Assessment of "What's Actually Hard"

Across all three approaches, only **one** genuinely research-hard problem exists: **whether psychoacoustic masking interactions admit any compositional algebraic structure** (Approach A's semiring question). Everything else is either:

1. **Standard techniques in a new domain** (refinement types, SMT, constraint satisfaction, greedy approximation, branch-and-bound)
2. **Engineering integration** (custom SMT theory, audio pipeline, CLI tooling)
3. **Domain formalization** (encoding psychoacoustic models, defining specification languages)

The depth check was correct: "no single component pushes the frontier, and 61% of the codebase is routine engineering." I'd revise upward slightly — the resource semiring in Approach A does push a frontier, but it's the *only* frontier-pushing component across all three approaches.

### Difficulty-to-Contribution Ratio

| Approach | Difficulty | Expected Contribution | Ratio (Impact/Difficulty) |
|----------|-----------|----------------------|--------------------------|
| A | 7 | High if semiring works, zero if not | **Volatile** — best case excellent, worst case wasted |
| B | 5 | Moderate-to-solid (guaranteed publishable) | **Best ratio** — predictable, sufficient novelty for OOPSLA |
| C | 4 | Moderate (nice paper, shallow theory) | **Decent ratio** — easy to execute but limited upside |

**Approach B has the best difficulty-to-contribution ratio.** It's hard enough to be a genuine OOPSLA paper but not so hard that it risks catastrophic failure. The refinement type + SMT architecture is proven technology; the domain application is novel; the evaluation is concrete and achievable.

**Approach A has the best *ceiling* but the worst risk-adjusted ratio.** If you're optimizing for expected value (probability × impact), B dominates A. If you're optimizing for maximum possible impact (best-paper shot), A is the play — but with a 40% chance of falling back to B anyway.

**Approach C is under-difficult for the venue.** OOPSLA reviewers will likely see through the "synthesis" framing and recognize it as constraint solving with a nice API. The theoretical contributions (NP-completeness of realizability, greedy approximation) are clean but shallow. This is more naturally an ICAD or UIST paper than an OOPSLA paper.

### The Depth Check's "61% Routine Engineering" Claim

**VALIDATED.** Across all three approaches, my novel LoC estimates are:
- A: ~28K novel out of ~70K total (40% novel)
- B: ~29K novel out of ~75K total (39% novel)  
- C: ~21K novel out of ~60K total (35% novel)

The depth check's estimate of "~45–50K novel core" was optimistic. The genuinely novel code — code that couldn't be written by a competent engineer following specifications — is closer to **21–29K LoC** depending on the approach. The rest is domain-specific but routine: implementing published formulas, building parsers with off-the-shelf tools, wiring up evaluation harnesses, writing tests.

### Final Recommendation

The approaches document recommends "Lead with B, keep A as stretch." **I agree**, but for a sharper reason: B is the only approach where the difficulty is well-calibrated for the venue. A is too risky (may not produce a type system at all); C is too easy (may not impress OOPSLA reviewers). B occupies the productive middle ground where the engineering is hard enough to be interesting and the theory is sound enough to be publishable.

The team should be honest that this project's difficulty is primarily **integration difficulty** (making four domains work together correctly) rather than **frontier difficulty** (inventing new theory). That's fine for OOPSLA, which values elegant engineering and domain-specific languages. But the paper framing should emphasize the *insight* (perception as refinement predicates) over the *difficulty* (which is moderate).

---

*Evaluation complete. Summary scores: A=7/10, B=5/10, C=4/10. Best difficulty-to-contribution ratio: B. Genuinely hard component across all approaches: only the resource semiring (A). Everything else is competent engineering in a novel domain.*
