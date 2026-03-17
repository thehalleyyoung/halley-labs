# Math Depth Evaluation: Perceptual Sonification Compiler

**Evaluator**: Math Depth Assessor  
**Date**: 2026-03-08  
**Input**: `approaches.md` (3 approaches, 15 claimed results), `depth_check.md` (7 binding amendments)

---

## Approach A: Graded Comonadic Sonification Calculus

### 1. Math Inventory

**Result A1: Psychoacoustic Resource Semiring — Claimed Grade A**

- **Genuinely novel?** Partially. Graded type systems with resource semirings exist (Granule, Brunel et al., Gaboardi et al.). The novelty is instantiating the resource algebra with psychoacoustic quantities. The *structure* (graded comonads, resource semirings) is known; the *instantiation* is new. This is "known technique, novel domain" — solidly B+ territory, not A.
- **Load-bearing?** Yes — critically so. Without associativity and commutativity of ⊕, the graded typing rules have no metatheoretic foundation. The entire Approach A collapses to Approach B if this fails.
- **Proof sketch plausible?** **No — this is where the approach is in serious trouble.** The composition operator includes the term $S(b', b) \cdot r_{1,b'} \cdot \mathbb{1}[r_{2,b} > 0]$. The indicator function $\mathbb{1}[r_{2,b} > 0]$ makes ⊕ depend on the *support* of $r_2$, not just its magnitude. Associativity requires $(r_1 \oplus r_2) \oplus r_3 = r_1 \oplus (r_2 \oplus r_3)$, but the indicator creates asymmetry: when computing $r_1 \oplus r_2$ first, the spreading activation from $r_1$ is gated by $r_2$'s occupied bands; when computing $r_2 \oplus r_3$ first, different bands gate different spreading terms. This is not an epsilon issue — it's a structural asymmetry. The mitigation (worst-case upper bound replacement) recovers semiring laws but makes the operator so pessimistic it may reject >50% of valid configurations, gutting practical utility.
- **Grade adjustment**: **A → B+**. Novel instantiation of a known framework. The hard part (making it actually work) is an open question with ~40% failure probability, acknowledged by the authors themselves.

**Result A2: Coeffect Typing for Perceptual Composition — Claimed Grade A**

- **Genuinely novel?** The typing rule presented is standard graded comonadic typing (cf. Orchard et al., Granule). The novelty is entirely parasitic on A1 — if A1's semiring works, A2 follows by instantiation. The logical-relations soundness proof is also standard once the semiring is in hand.
- **Load-bearing?** Yes, but only as a downstream consequence of A1. A2 is not an independent contribution — it's "plug A1's semiring into the standard graded comonadic framework."
- **Proof sketch plausible?** Conditional on A1, yes. Logical relations for graded comonadic calculi are well-understood. The only subtlety is that the "perceptual semantics" side of the logical relation (relating typed terms to psychoacoustically-feasible renderings) needs careful setup, but this is engineering, not innovation.
- **Grade adjustment**: **A → B+**. This is not a co-crown jewel — it's the predictable consequence of A1. Claiming two Grade A results where the second is mechanically derived from the first is inflation.

**Result A3: Bark-Band Decomposition with Interaction Correction — Claimed Grade B**

- **Genuinely novel?** The $(1-1/e)(1-\epsilon_S)$ approximation bound is a perturbation of the standard submodular-maximization result (Nemhauser-Wolsey-Fisher 1978). The $\epsilon_S$ correction term is domain-specific but the proof technique is textbook: bound the interaction term, absorb it into the approximation ratio.
- **Load-bearing?** Moderately. Without it the optimizer still works — it just lacks a quality guarantee. The system functions; you lose a theorem in the paper.
- **Proof sketch plausible?** Yes. This is the most straightforward result in Approach A. Standard perturbation analysis of greedy submodular maximization.
- **Grade adjustment**: **B — confirmed.** Honest grading.

**Result A4: Monotonicity of Discriminability Under Resource Weakening — Claimed Grade B**

- **Genuinely novel?** **No. This is essentially a tautology.** If "uses fewer resources" is defined via a pre-order where less energy in each band means less masking, and discriminability is defined as inversely related to masking, then monotonicity follows by definition. The only way this could be non-trivial is if the discriminability metric has a non-monotone dependence on resource consumption, but the $d'_{\text{model}}$ metric used is explicitly constructed to be monotonically decreasing in masking energy. This is unwinding a definition, not proving a theorem.
- **Load-bearing?** Needed for subtyping soundness in theory, but any reasonable definition of the pre-order would make this true. If it weren't true, the pre-order would be wrong, not the theorem.
- **Proof sketch plausible?** Trivially so — that's the problem.
- **Grade adjustment**: **B → C.** This is a lemma that verifies the definitions are self-consistent. Flagged as **ornamental**.

**Result A5: Decidability of Feasibility with Spreading Functions — Claimed Grade C**

- **Genuinely novel?** No. For fixed $k \leq 16$ and $B = 24$, the parameter space is finite (after discretization to JND granularity). Decidability of a finite problem is trivially true. The only question is complexity, and polynomial in fixed-parameter quantities is not a result.
- **Load-bearing?** No — the authors themselves acknowledge brute force works at practical scales.
- **Grade adjustment**: **C — confirmed**, but could arguably be dropped entirely. Decidability of a finite computation is not a theorem.

### 2. Hardest Proof

**A1: Associativity of the psychoacoustic resource semiring.**

- **Probability of successful proof**: **25%** for exact associativity, **60%** for an ε-approximate version with useful (non-vacuous) ε.
- **Time to prove**: 3–4 weeks for the exploration + proof/disproof. The empirical phase (testing associativity violations) is fast; the theoretical phase (proving ε-approximate semiring laws suffice for soundness) requires extending the graded type theory metatheory, which has not been done for approximate semirings.
- **If proof fails**: The entire approach collapses to Approach B. This is a single point of failure. The conservative over-approximation fallback may technically work but produces a semiring so pessimistic that the paper's evaluation story ("catches real bugs without excessive false positives") fails. A type system that rejects 50%+ of valid configurations is not useful.

### 3. Math Depth Score: **6/10**

Novel application of known graded type theory to a new domain. The semiring design is genuinely interesting as an open problem, but the framework itself (graded comonads, coeffect tracking) is established. If the exact semiring works, this rises to 7. The approximate-semiring extension would be mildly novel PL theory (nobody has studied approximate semirings in graded type systems), but it's a natural extension, not a deep new result.

### 4. Ornamental Math Detection

- **A4 (Monotonicity)**: **Tautology alert.** This "theorem" restates the definition of the pre-order. The pre-order is *defined* so that less resource → more discriminability. Proving monotonicity from this definition is circular.
- **A5 (Decidability)**: **Trivial decidability.** Any bounded finite problem is decidable. The polynomial-time claim for fixed parameters is a routine application of parameterized complexity.
- **A2 (Coeffect typing)**: **Not ornamental, but not independent.** It's presented as a co-crown jewel (Grade A) but is really just the mechanical application of A1 to a known framework. Conflating "we plugged our semiring into existing theory" with "we proved a new theorem" inflates the contribution count.

---

## Approach B: Liquid Sonification — SMT-Backed Refinement Types

### 1. Math Inventory

**Result B1: Refinement Type System for Psychoacoustic Predicates — Claimed Grade A**

- **Genuinely novel?** The typing rules are standard Liquid-type rules with domain-specific predicates. The *form* $\text{Stream}\{v \mid \phi(v, \Gamma)\}$ is textbook Liquid Haskell. The *content* of the predicates (masking, JND, segregation) is new. The genuinely novel element is the **environment-dependent refinement predicates** — $\phi_{\text{mask}}(v, \Gamma)$ depends on the typing environment Γ (other streams), which is non-standard for refinement types (typically predicates depend only on the refined value, not the context). This is a real, if incremental, contribution to refinement type theory.
- **Load-bearing?** Yes. Without formal typing rules + soundness, the system is (as the depth check flagged) "constraint checking with PL vocabulary." The typing rules ARE the PL contribution.
- **Proof sketch plausible?** Mostly yes. The progress/preservation proof via SMT soundness is standard for Liquid-type systems. The subtlety is the environment-dependent predicates: you need SMT queries that include all of Γ, which means the proof must handle the fact that adding a binding to Γ changes the satisfying set of existing bindings' predicates. This is where the non-locality bites, and the proof will need a novel lemma about "environment extension preserving satisfiability under masking monotonicity." This is doable but requires care.
- **Grade adjustment**: **A → A-/B+**. The environment-dependent predicate extension is genuinely novel but incremental. It's the strongest "new PL result" across all three approaches, which says more about the other approaches than about this one.

**Result B2: Soundness of Custom SMT Theory Solver — Claimed Grade A**

- **Genuinely novel?** Custom SMT theory solvers are published regularly (CVC5 theories, Z3 plugins for floating-point, strings, etc.). The novelty is the domain (Bark-scale arithmetic), not the technique. The δ-soundness framing (bounding rounding error from fixed-point approximation of transcendental functions) is standard numerical analysis applied to SMT.
- **Load-bearing?** Absolutely. An unsound theory solver makes the type system broken. But "we verified our implementation is correct" is an engineering contribution, not a mathematical one.
- **Proof sketch plausible?** Yes. Bounding rounding error of piecewise-linear approximations to arctan and the spreading function is textbook interval arithmetic. The 0.1 dB bound is conservative given that JND thresholds are 1-3 dB.
- **Grade adjustment**: **A → B+**. This is careful engineering with a correctness proof, not novel mathematics. Every verified SMT theory solver does this. Calling it a "co-crown jewel" (Grade A) is inflation.

**Result B3: Incremental Constraint Propagation — Claimed Grade B**

- **Genuinely novel?** Incremental constraint re-checking after adding a variable is well-studied in constraint programming (AC-3 and variants). The domain-specific insight (only O(k) new pairwise constraints + O(B) threshold updates) is a straightforward analysis of the constraint graph structure.
- **Load-bearing?** Partially. Without it, full re-checking is O(k²) per addition — still polynomial, still fast for k ≤ 16. This is a constant-factor improvement, not an asymptotic one for practical inputs.
- **Proof sketch plausible?** Yes. The equivalence between incremental and full re-checking follows from the structure of the masking model: only streams sharing Bark bands interact.
- **Grade adjustment**: **B → B-**. Useful engineering lemma, not a deep result.

**Result B4: OMT Approximation for I_ψ Maximization — Claimed Grade B**

- **Genuinely novel?** No. This is explicitly acknowledged as "a standard branch-and-bound convergence result applied to the psychoacoustic domain." The $O(|\mathcal{P}|^{O(\log(1/\epsilon))})$ bound is textbook.
- **Load-bearing?** Weakly. The optimizer works empirically without a formal guarantee. The bound justifies enumeration, which is what you'd do anyway for small parameter spaces.
- **Proof sketch plausible?** Trivially so — it's a known result.
- **Grade adjustment**: **B → C+**. Applying a known convergence result to a new domain is worth mentioning but not worth calling a "Grade B" result.

**Result B5: Diagnostic Extraction from UNSAT Cores — Claimed Grade C**

- **Genuinely novel?** UNSAT core extraction is a standard feature of SAT/SMT solvers. The domain-specific mapping to human-readable diagnostics is UX engineering.
- **Load-bearing?** No — the type system works without it.
- **Grade adjustment**: **C — confirmed.** Honest grading.

### 2. Hardest Proof

**B1: Soundness of refinement types with environment-dependent predicates.**

- **Probability of successful proof**: **75%**. The standard Liquid-type proof structure applies; the only new piece is handling the environment-dependent predicates, which requires a monotonicity lemma for masking (adding energy to the environment only increases masking thresholds, never decreases them). This is physically motivated and likely true, but needs verification for all combinations of the Schroeder spreading function.
- **Time to prove**: 2–3 weeks.
- **If proof fails**: The approach degrades but doesn't collapse. You can restrict composition to a "re-check everything" mode that's still sound (just less elegant). The paper loses the "compositional type checking" angle but retains "SMT-backed verification."

### 3. Math Depth Score: **5/10**

Solid application of known refinement-type machinery to a genuinely new domain. The environment-dependent predicate extension is the one piece of real PL novelty and it's incremental. The SMT theory solver is engineering, not math. The overall impression is "Liquid Haskell, but for audio" — which is a perfectly valid OOPSLA paper but not mathematically deep.

### 4. Ornamental Math Detection

- **B2 (SMT Solver Soundness)**: **Inflated to Grade A.** This is a verification-of-implementation result, not a theorem about a mathematical structure. Every custom SMT theory needs a soundness argument; calling it a "co-crown jewel" is resume-padding.
- **B4 (OMT Approximation)**: **Known result applied to new domain.** The authors acknowledge this themselves. Shouldn't be presented as a "new math result" in the paper — mention it in passing.
- **B5 (Diagnostic Extraction)**: Not ornamental per se, but not math. It's software engineering.

---

## Approach C: SoniSynth — Program Synthesis

### 1. Math Inventory

**Result C1: Psychoacoustic Realizability — Claimed Grade A**

- **Genuinely novel?** The NP-completeness reduction from graph coloring is straightforward: assign each "color" a region of perceptual space, constraint = different colors. This is a **trivial reduction** from a well-known NP-complete problem. Any constraint-satisfaction problem over a finite domain with pairwise constraints is NP-complete by reduction from graph coloring — this is one of the first exercises in a graduate complexity course. The polynomial special case (independent Bark bands + uniform thresholds) is also routine: independent constraints decompose into independent subproblems, each solvable in polynomial time. The practical algorithm (constraint propagation + backtracking in <10s for k ≤ 12) is standard CSP engineering.
- **Load-bearing?** Partially. Knowing that realizability is NP-complete tells you exhaustive search is likely necessary (modulo P≠NP). But you'd use constraint propagation + backtracking regardless — the complexity result doesn't change the algorithm.
- **Proof sketch plausible?** Yes, because it's trivial.
- **Grade adjustment**: **A → C+**. This is the most egregious grade inflation across all three approaches. An NP-completeness result via trivial reduction from graph coloring is a homework exercise, not a crown jewel. The depth check's assessment that "Grade A theorems are B/B+ formalizations" was *too generous* for this one.

**Result C2: Constructive Synthesis via Greedy Perceptual Packing — Claimed Grade A**

- **Genuinely novel?** Greedy packing to maximize minimum pairwise distance in a metric space is a well-studied problem in computational geometry (farthest-point insertion). The $(1/\alpha)$-approximation for max-min distance in a metric space follows from the greedy algorithm's known guarantees for k-center or dispersion problems. The domain-specific adaptation (perceptual space with masking constraints) adds some complexity, but the core algorithm and proof technique are textbook.
- **Load-bearing?** Yes — without an efficient algorithm, synthesis is brute force. But calling a known algorithm applied to a new metric space a "Grade A crown jewel" is not honest.
- **Proof sketch plausible?** Yes. The approximation guarantee follows from the metric properties of perceptual space (triangle inequality for $d'_{\text{model}}$, if it holds — and this is a hidden assumption that needs verification). If perceptual distance doesn't satisfy the triangle inequality, the greedy guarantee breaks.
- **Grade adjustment**: **A → B-**. Known algorithm, known proof technique, new domain. The hidden triangle-inequality assumption is a risk.

**Result C3: Specification Refinement Lattice — Claimed Grade B**

- **Genuinely novel?** Defining a lattice over constraint specifications where "stricter is lower" is standard in abstract interpretation and constraint-based program analysis. The lattice structure follows immediately from the subset ordering on satisfying configurations. "Greatest realizable specification below the user's input" is a standard greatest-lower-bound computation in the lattice.
- **Load-bearing?** Useful for UX (explaining why synthesis failed) but not for correctness. Without it, the synthesizer says "failed" instead of "failed, but relaxing X fixes it."
- **Proof sketch plausible?** Yes — defining the lattice and computing GLBs in a finite discretized space is routine.
- **Grade adjustment**: **B → C+**. This is a definition dressed up as a theorem.

**Result C4: Information-Theoretic Optimality of Synthesized Configurations — Claimed Grade B**

- **Genuinely novel?** The $(1-1/e)$ bound from submodular maximization under matroid constraints is Calinescu et al. (2011) / Vondrák (2008). Applying it to mutual information in the perceptual setting requires showing $I_\psi$ is submodular, which may or may not hold. If it holds, the result follows immediately from existing theory. If it doesn't hold, the claim is false.
- **Load-bearing?** Weakly. The synthesizer produces good results empirically; the theoretical guarantee is nice-to-have.
- **Proof sketch plausible?** Conditional on submodularity of $I_\psi$, which is asserted but not proven. Mutual information is not submodular in general. It is submodular in $I(X; Y_S)$ when $Y_S$ is a subset of independent sources, but streams are not independent (they interact via masking). This is a **hidden gap** that could invalidate the result.
- **Grade adjustment**: **B → B-** (if submodularity is proven) or **B → D** (if submodularity doesn't hold). Flagged as a **hidden difficulty**.

**Result C5: Synthesis Completeness for Uniform Specifications — Claimed Grade C**

- **Genuinely novel?** The symmetry argument (for uniform specs, optimal = maximally separated = greedy) is clean but straightforward. It's a corollary of the greedy algorithm's optimality for the k-center problem on symmetric instances.
- **Load-bearing?** No — the approximation guarantee already covers the general case.
- **Grade adjustment**: **C — confirmed.**

### 2. Hardest Proof

**C4: Submodularity of $I_\psi$ under masking interactions.**

- **Probability of successful proof**: **30%**. Mutual information is not generally submodular when sources interact. Masking creates exactly the kind of interaction that breaks submodularity: adding stream C can reduce the information contributed by stream A (if C masks A). This means $I_\psi(D; A_{S \cup \{C\}}) - I_\psi(D; A_S)$ can *increase* as S grows (when a new stream masks a previously-masking stream, unmasking the target — a "masking release" effect). Diminishing returns fails.
- **Time to prove**: 2 weeks to attempt, likely ending in disproof.
- **If proof fails**: The $(1-1/e)$ optimality guarantee disappears. The greedy algorithm still works heuristically but has no quality guarantee. The paper loses its theoretical optimization contribution, leaving only the NP-completeness result (which is trivial) and the greedy algorithm (which is known).

### 3. Math Depth Score: **3/10**

The honest assessment: Approach C's mathematical contributions are routine. The NP-completeness reduction is trivial. The greedy algorithm is known. The specification lattice is a definition. The submodularity claim is likely false. The only genuine mathematical content is the domain-specific adaptation of known algorithms, which is competent but not novel.

### 4. Ornamental Math Detection

- **C1 (Realizability NP-completeness)**: **Trivial reduction alert.** This is the most ornamental result in the entire document. Reduction from graph coloring to pairwise-constraint satisfaction over a finite domain is a textbook exercise. Presenting it as a "Grade A crown jewel" is deeply misleading.
- **C3 (Specification Lattice)**: **Definition dressed as theorem.** Defining a partial order on specifications and noting it forms a lattice is not a theorem — it's a design choice.
- **C4 (Submodularity claim)**: **Likely false.** Masking interactions break the diminishing-returns property required for submodularity. This isn't ornamental — it's wrong (or at least unsubstantiated).

---

## 5. Cross-Approach Comparison

### Deepest Genuine Math

**Approach A** has the deepest genuine mathematical ambition: the psychoacoustic resource semiring (A1) is a real open problem that could yield a genuinely interesting structure. But the risk of failure is high, and most of the remaining results (A2, A4, A5) are either mechanical consequences of A1 or trivial.

**Approach B** has the best "math that matters" ratio. B1's environment-dependent refinement predicates are a real (if incremental) contribution to PL theory. B2's solver soundness is engineering but essential. The mathematical content is modest but load-bearing.

**Approach C** has the shallowest math. Every claimed result is either a trivial reduction (C1), a known algorithm (C2), a definition (C3), or likely false (C4).

### Best Ratio of "Math That Matters" to "Math That Impresses"

| Approach | Total claimed results | Genuinely novel & load-bearing | Ratio |
|----------|----------------------|-------------------------------|-------|
| A | 5 (2A, 2B, 1C) | 1 (A1, partially) | 20% |
| B | 5 (2A, 2B, 1C) | 1.5 (B1 fully, B2 partially) | 30% |
| C | 5 (2A, 2B, 1C) | 0.5 (C2, partially) | 10% |

**Approach B wins on ratio.** It claims less and delivers more honestly. Approach A's impressive-sounding graded comonadic framing masks the fact that only one result (A1) is genuinely uncertain/interesting, and it has a 60-75% chance of failing in its strong form. Approach C's claims are the most inflated relative to actual mathematical content.

### Adjusted Depth Scores

| Approach | Claimed Difficulty | Adjusted Math Depth | Delta |
|----------|-------------------|---------------------|-------|
| A | 8/10 | 6/10 | -2 |
| B | 7/10 | 5/10 | -2 |
| C | 6/10 | 3/10 | -3 |

---

## 6. Recommendations

### For the Winning Approach (Likely B, per recommended strategy)

**Prioritize:**
1. **B1: Environment-dependent refinement predicates.** This is the one genuinely novel PL result. Invest heavily in the soundness proof, particularly the environment-extension lemma. Make this the paper's centerpiece theorem.
2. **The custom SMT theory solver.** Not as a "Grade A theorem" but as a carefully-engineered, verified component. Present the δ-soundness as a correctness guarantee, not a research contribution. Reviewers will appreciate honesty.
3. **Non-local constraint propagation (B3).** Demote from "theorem" to "algorithmic contribution." Present the incremental algorithm with empirical performance data, not a formal proof of equivalence (which is trivial).

**Drop or demote:**
1. **B4 (OMT Approximation):** Move to appendix. It's a known result and adds nothing to the paper's novelty claim.
2. **B2 as "co-crown jewel":** Rename to "Implementation Correctness" or "Solver Verification." Presenting verified engineering as a Grade A theorem damages credibility with PL reviewers who know what Grade A type theory looks like.

### If Upgrading to Approach A

**The semiring is everything.** If the week-4 empirical test shows associativity violations < 5%, invest in the ε-approximate semiring theory. This would be genuinely novel — nobody has studied approximate semirings in graded type systems. But be honest: the contribution is "approximate resource semirings for graded types" (a PL theory contribution), not "psychoacoustic resource tracking" (a domain contribution). The psychoacoustics is the motivation; the PL theory is the result.

**Drop A4 entirely.** It's a tautology. Drop A5 as well — decidability of finite problems is not a theorem.

### If Approach C Elements Are Incorporated as Frontend

**Do not claim C1 (NP-completeness) as a contribution.** It's a trivial reduction that will embarrass the paper if a reviewer recognizes it. Instead, simply state: "realizability of perceptual specifications is NP-complete (by standard reduction from graph coloring)" as a one-sentence justification for using heuristic search. No proof, no theorem number, no fanfare.

**The greedy packing algorithm (C2) is useful as engineering** but should not be presented as a theoretical contribution. Present it as "our synthesis algorithm" with empirical evaluation, not as "Theorem: greedy achieves $(1/\alpha)$-approximation."

### Validation of the Depth Check's Assessment

The depth check stated: "The 4 'Grade A' theorems are B/B+ formalizations—novel in domain application, not in mathematical depth." 

**I partially agree and partially disagree:**
- For B1 (environment-dependent refinement predicates): the depth check was slightly too harsh. This is a genuine, if incremental, PL contribution. **B+ is right.**
- For A1 (resource semiring): the depth check was approximately right. **B+ if the semiring works; B if it requires aggressive over-approximation.**
- For C1 (realizability NP-completeness): the depth check was **far too generous**. This is a **C+** at best — a trivial reduction that any graduate student could produce in an afternoon.
- For C2 (greedy packing): the depth check was approximately right. **B- is fair.**

The overall pattern: every approach inflates its contribution count by 1-2 "theorems" that are either tautologies, trivial reductions, or known results. The honest theorem count per approach is:

| Approach | Claimed contributions | Real contributions | Genuine novelty |
|----------|----------------------|-------------------|-----------------|
| A | 5 | 2 (A1, A3) | 1 (A1, if it works) |
| B | 5 | 2 (B1, B2-as-engineering) | 1 (B1) |
| C | 5 | 1 (C2-as-engineering) | 0 |

**Bottom line:** Approach B has the best risk-adjusted mathematical integrity. Approach A has the highest ceiling but the thinnest margin for error. Approach C should not be selected as the primary approach on mathematical grounds — its claimed contributions don't survive scrutiny.

---

*This evaluation is intentionally harsh. The purpose is to prevent the team from committing to mathematical claims that will not survive peer review at OOPSLA, where reviewers are expert PL theorists who will identify trivial reductions and inflated theorem counts immediately.*
