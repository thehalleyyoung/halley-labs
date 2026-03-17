# Math Depth Review: BiCut — Three Approaches

**Reviewer:** Math Depth Assessor  
**Date:** 2026-03-08  
**Mandate:** Brutal honesty about what is genuinely new vs. routine application of known theory.

---

## Approach A: "Cuts-First" — Deep Cutting Plane Theory

### 1. Is the math load-bearing?

**Bilevel-infeasible set polyhedrality theorem (Difficulty C).** *Verdict: Load-bearing, but partially folklore.* That the bilevel-infeasible set for an MIBLP with LP lower level decomposes as a finite union of polyhedra is essentially a consequence of the fact that the lower-level optimal partition (the mapping from leader decision x to the set of optimal bases of the follower LP) is a polyhedral complex. This is known from parametric linear programming (Gal 1995, Adler & Monteiro 1992). The "theorem" amounts to: (1) enumerate critical regions of the parametric LP, (2) in each region, bilevel feasibility is a linear condition (ŷ optimal ⟺ ŷ lies in the optimal face for that basis). Characterizing the *facets* of this union in terms of the lower-level dual vertex set is new, but the polyhedrality itself is not. **The result is necessary for the cut framework to have a foundation, but calling it Difficulty C overstates the novelty of the polyhedrality claim itself. The facet characterization within each critical region is the genuinely hard part.**

**Facet-defining conditions (Difficulty C).** *Verdict: Load-bearing and genuinely hard — IF it can be proved in useful generality.* This is the real mathematical contribution. Proving that bilevel intersection cuts define facets of conv(S_BF) requires constructing dim(P) affinely independent feasible points on the hyperplane — and the dimension of the bilevel-feasible polyhedron is itself nontrivial to establish because bilevel feasibility is not described by a fixed set of linear inequalities. The closest precedent is Balas's 1979 facet proof for disjunctive cuts, but Balas had a clean disjunction (x ∈ S₁ ∪ S₂ where S₁, S₂ are polyhedra). Here the disjunction is optimality-based, meaning the "pieces" of the infeasible set depend on x through the parametric LP's basis partition. **This is genuinely hard. The difficulty grade C is honest for the general case. However, the fatal risk is that the result may only be provable for trivial cases (1 leader variable, non-degenerate lower level), and the general case may require handling dual degeneracy that makes the combinatorial structure explode.**

**Separation complexity — polynomial for fixed follower dimension (Difficulty B).** *Verdict: Load-bearing, but the "fixed follower dimension" qualifier does most of the work.* For fixed follower dimension d, the number of lower-level bases is O(m^d) where m is the number of follower constraints. The separation oracle enumerates critical regions and tests ray intersections — polynomial in m for fixed d. This is a direct application of parametric LP complexity bounds. The result is essentially: "the separation is polynomial by the same argument that parametric LP has polynomial complexity for fixed dimension." **Difficulty B is slightly generous; this is closer to A+ (known techniques applied with care). The community would recognize this as a parameterized complexity argument, not a deep result.**

**Finite convergence (Difficulty B).** *Verdict: Partially load-bearing, largely expected.* Finite convergence of cutting-plane methods under non-degeneracy is a standard result in polyhedral combinatorics. The bilevel setting introduces the complication that the feasible set is not a polyhedron but a union of polyhedra; however, finite convergence for unions of polyhedra under non-degeneracy follows from the finite number of facets. **Difficulty B is honest but the result is expected — no reviewer would be surprised. The contribution is in the careful bookkeeping, not in new ideas.**

**Value-function lifting via Gomory–Johnson (Difficulty C).** *Verdict: Load-bearing and novel in concept, but the instantiation may be vacuous.* The idea of extending Gomory–Johnson subadditivity to value-function epigraphs is genuinely creative — it connects two literatures that have not interacted. However, the Gomory–Johnson theory is developed for functions on the integers (or lattice points), where subadditivity has a clean algebraic characterization. The value function V(x) of an LP is piecewise linear and continuous — the subadditivity theory applies but may reduce to a triviality (every piecewise-linear convex function is its own maximal subadditive minorant). For MILP lower levels, V(x) is discontinuous and evaluating it is NP-hard, making the lifting theory impossible to instantiate computationally in all but the smallest cases. **Difficulty C is honest for the theoretical development but the practical value is uncertain. The risk is that the "Gomory–Johnson extension" sounds impressive but collapses to a restatement of LP duality for the continuous case and is computationally intractable for the integer case.**

**Sampling-based MILP value-function approximation (Difficulty B).** *Verdict: Partially load-bearing, straightforward.* Piecewise-linear approximation of functions with bounded variation on compact domains is well-studied in approximation theory. The O(h²) error bound for grid-based approximation is standard. The bilevel-specific content is: (a) the domain is the leader's feasible set (a polyhedron, so standard), (b) the function is the MILP value function (discontinuous, so the O(h²) bound holds only in an integral/average sense, not pointwise). **Difficulty B is slightly generous. This is a competent application of known approximation theory to a specific setting. The bilevel community would recognize it as useful engineering, not new mathematics.**

### 2. Is the difficulty grading accurate?

| Result | Claimed | Assessed | Honest? |
|--------|---------|----------|---------|
| Polyhedrality theorem | C (novel) | B+ (new formalization of known structure) | **Inflated by ~half a grade** |
| Facet-defining conditions | C (novel) | C (genuinely novel, high risk) | **Honest** |
| Separation complexity | B (hard, precedented) | A+/B− (parameterized complexity argument) | **Slightly inflated** |
| Finite convergence | B (hard, precedented) | B (correct grade) | **Honest** |
| Value-function lifting | C (novel) | C (novel concept, uncertain instantiation) | **Honest but risks vacuity** |
| MILP approximation | B (hard, precedented) | A+/B− (known approximation theory) | **Slightly inflated** |

**Overall: The grading is slightly inflated on the more routine results but honest on the genuinely hard ones.** The community would push back on the polyhedrality theorem being Difficulty C — anyone who has worked with parametric LP knows this structure.

### 3. What's the real mathematical novelty?

**(a) Genuinely new theorems requiring novel proof techniques:**
- Facet-defining conditions for bilevel intersection cuts. This is real. No one has done this.
- Value-function lifting concept (connecting Gomory–Johnson to bilevel). Novel framing, uncertain depth.

**(b) Straightforward extensions of known results:**
- Separation complexity (parameterized complexity of parametric LP applied to cut separation).
- Finite convergence (standard cutting-plane convergence adapted to union-of-polyhedra structure).
- Sampling-based approximation (standard approximation theory on a specific function class).

**(c) Formalization of folklore:**
- Bilevel-infeasible set polyhedrality. The parametric LP community has known informally that the bilevel feasibility structure decomposes into critical regions for decades.

### 4. Fatal mathematical risks

1. **The facet-defining proof may only work for non-degenerate LP lower levels with a single leader variable.** The approach document explicitly acknowledges this as a fallback, but if the general result fails, the contribution shrinks from "new facet theory for bilevel polyhedra" to "a facet result for a special case that covers some BOBILib instances." The bilevel community has seen many results that work beautifully for the LP-lower-level case and collapse for anything harder.

2. **Value-function lifting may be trivial for LP lower levels.** If V(x) is piecewise linear and convex (which it is for LP lower levels with unique optimal solutions), the Gomory–Johnson lifting function may reduce to the concave envelope of V, which is V itself if V is already concave, or a known construction otherwise. The "Gomory–Johnson extension" may add no cutting power beyond what the value-function reformulation already provides.

3. **The gap closure target has no empirical anchor.** The 15–25% claim is calibrated against GMI cuts for pure IPs. But GMI cuts exploit integrality — a combinatorial structure. Bilevel infeasibility is optimality-based — a continuous structure. There is no reason to believe the same gap closure magnitudes transfer. If gap closure is 2–5%, the mathematics is correct but the contribution is "we proved facet-defining conditions for cuts that don't help much in practice."

4. **Dual degeneracy may make the polyhedral description exponentially complex.** The number of critical regions in a parametric LP can be exponential in the number of constraints even for fixed dimension (when the lower-level LP is highly degenerate). This could make the separation oracle impractical despite being "polynomial for fixed dimension."

### 5. Math contribution grade

**Grade: B+**

*Justification:* The facet-defining conditions for bilevel intersection cuts, if provable in useful generality, would be a genuine contribution to polyhedral combinatorics — the first extension of Balas's framework to optimality-defined infeasible sets. This is a real conceptual advance. The value-function lifting idea is creative and connects two disjoint literatures. However, the supporting results (polyhedrality, separation complexity, finite convergence, sampling approximation) range from known-with-new-wrapping to standard. The portfolio is top-heavy: one genuinely hard result (facets), one creative but uncertain result (lifting), and four routine results dressed up with difficulty grades that are slightly inflated. If the facet-defining proof works in full generality, this is an A. If it only works for the trivial case, this drops to B−.

---

## Approach B: "Compiler-First" — Full Compiler Architecture

### 1. Is the math load-bearing?

**Compiler soundness theorem (Difficulty B).** *Verdict: Load-bearing as a correctness guarantee, but mathematically routine.* The theorem states: if typechecking succeeds and structural analysis certifies preconditions Φ, then opt(emit(R)) = opt(P). The proof proceeds by cases over reformulation type — KKT soundness under LICQ+convexity (Dempe 2002, textbook), strong duality under LP+boundedness (Fortuny-Amat & McCarl 1981, textbook), value-function unconditionally (Outrata et al. 1998, textbook). Each case is a known correctness result; the "theorem" is their conjunction, guarded by a type system that checks preconditions. **This is engineering formalism, not mathematics. The individual correctness results are textbook; the type-system framing is a contribution to software design, not to optimization theory. Difficulty B is generous — this is Difficulty A (known results) organized in a new way. A bilevel optimization researcher would recognize every piece.**

**Compilability decision procedure (Difficulty B).** *Verdict: Technically load-bearing for completeness, but the result is likely trivial.* The decision procedure amounts to: check if any reformulation strategy's preconditions are satisfied by the problem's structural signature. Since there are finitely many reformulation strategies (KKT, strong duality, value function, C&CG) and each has explicit, efficiently checkable preconditions (convexity, CQ status, LP structure, boundedness), the decision procedure is polynomial — it's a lookup in a finite table. The "NP-hard via graph coloring" conjecture for optimal selection is more interesting but is explicitly labeled as a conjecture and is not proved. **Difficulty B is inflated. The decision procedure itself is trivially polynomial (finite enumeration over known strategies). The NP-hardness conjecture for optimal selection would be Difficulty B if proved, but it's a conjecture, not a result.**

**Structure-dependent selection soundness (Difficulty A).** *Verdict: Load-bearing but trivial.* The document itself acknowledges this is "a straightforward case analysis." Proving that a lookup table returns correct entries is not mathematics. **Difficulty A is correctly graded — this is indeed known/trivial.**

**Compositional error bounds (Difficulty B).** *Verdict: Partially load-bearing, standard.* The bound Π(1 + κᵢεᵢ) − 1 is a standard condition-number-based error propagation from numerical analysis. The bilevel-specific content is identifying the condition numbers κᵢ for each reformulation pass. **Difficulty B is slightly generous for what amounts to applying standard perturbation theory.**

**Bilevel intersection cuts — basic, no facet theory (Difficulty B).** *Verdict: Partially load-bearing, correctly downgraded.* Without facet theory, this is indeed "adapting Balas's procedure" to a new infeasible set — a competent application, not a theoretical contribution. **Difficulty B is honest for the implementation; the mathematical novelty is nil.**

**QP complementarity linearization (Difficulty A).** *Verdict: Load-bearing for QP extension, entirely routine.* McCormick envelopes for bilinear terms are textbook (McCormick 1976, Al-Khayyal & Falk 1983). Applying them to KKT-derived bilinear terms is a direct application. **Difficulty A is honest.**

### 2. Is the difficulty grading accurate?

| Result | Claimed | Assessed | Honest? |
|--------|---------|----------|---------|
| Compiler soundness | B (hard, precedented) | A+ (known results in new packaging) | **Inflated by a full grade** |
| Compilability decision | B (hard, precedented) | A (trivial for current scope) | **Inflated by a full grade** |
| Selection soundness | A (known) | A (trivial) | **Honest** |
| Compositional error bounds | B (hard, precedented) | A+/B− (standard numerical analysis) | **Slightly inflated** |
| Bilevel intersection cuts (basic) | B (hard, precedented) | B (fair for implementation work) | **Honest** |
| QP linearization | A (known) | A (textbook) | **Honest** |

**Overall: The grading is systematically inflated on the two flagship results (compiler soundness and compilability).** The bilevel optimization community would immediately recognize the compiler soundness theorem as a reorganization of known correctness results, not a new theorem. The "hard but precedented" label disguises what is essentially: "we checked that four textbook results apply under the conditions our type system verifies."

### 3. What's the real mathematical novelty?

**(a) Genuinely new theorems requiring novel proof techniques:**
- None. There is no result in Approach B that would be publishable as a standalone mathematical contribution.

**(b) Straightforward extensions of known results:**
- Compositional error bounds (standard perturbation theory applied to reformulation chains).
- Bilevel intersection cuts without facet theory (Balas's procedure on a new set).

**(c) Formalization of folklore:**
- Compiler soundness theorem (known correctness results for KKT, strong duality, value function organized as a type-system theorem).
- Compilability decision (trivial enumeration dressed up as a decision procedure).
- Selection soundness (a lookup table is correct by construction).

### 4. Fatal mathematical risks

1. **There is no mathematical risk because there is no genuinely new mathematics.** Every "theorem" in Approach B is either a known result or a trivial consequence of known results. This is both a strength (no risk of falsity) and a weakness (no mathematical contribution).

2. **The "just engineering" criticism is mathematically valid.** A reviewer at Mathematical Programming would correctly observe that Approach B proves no new mathematical results. The compilation pipeline is a software contribution, not a mathematical one. IJOC would be more receptive, but even there, the mathematical bar requires at least one non-trivial theorem.

3. **The NP-hardness conjecture for optimal reformulation selection is unproved and could be wrong.** The conjecture "NP-hard via reduction from graph coloring on reformulation compatibility graphs" is stated without evidence. If the problem is actually polynomial (which is plausible given the small number of reformulation strategies and the monotonicity of precondition satisfaction), the conjecture collapses and the theoretical contribution shrinks further.

### 5. Math contribution grade

**Grade: D+**

*Justification:* Approach B contains no genuinely new mathematical results. The compiler soundness theorem is a conjunction of textbook correctness results guarded by a type system. The compilability decision is a trivial enumeration. The selection soundness is a tautology. The compositional error bounds are standard numerical analysis. The "basic" intersection cuts without facet theory are an application of Balas. This is excellent *engineering* formalized with mathematical language, but it is not a mathematical contribution. The D+ (rather than D) reflects that the formalization is clean and the compositional structure of the correctness argument has pedagogical value — but a bilevel optimization expert would learn nothing mathematically new from this approach.

---

## Approach C: "Hybrid" — Compiler + Cuts as Co-Primary Contributions

### 1. Is the math load-bearing?

**Bilevel intersection cuts — full theory (Difficulty C).** *Same assessment as Approach A.* The facet-defining conditions are genuinely novel; the polyhedrality theorem is partially folklore. The difference from Approach A is that the cuts are co-primary rather than the sole contribution, meaning partial success (separation without facets) still yields a viable paper. **Load-bearing for the mathematical component, with graceful degradation.**

**Value-function lifting (Difficulty C).** *Same assessment as Approach A.* Novel concept, uncertain instantiation. **Load-bearing for cut strength, with the same risk of vacuity.**

**Compiler soundness theorem — extended (Difficulty B).** *Verdict: Load-bearing, marginally harder than Approach B's version.* The extension to cover the cut pass adds genuine content: proving that the added cuts are valid inequalities (not just any hyperplane) requires verifying that the separation oracle produces valid cuts, which requires the polyhedral characterization from the cut theory. This creates a real dependency between the compiler soundness and the cut theory — making the proof non-trivially harder than Approach B's case analysis. **Difficulty B is honest for this extended version. The compiler-soundness-with-cuts theorem is a genuine integration result, not just a conjunction of known correctness results.**

**Certificate composition — three-way (Difficulty B).** *Verdict: Load-bearing and mildly novel.* The composition of structural + reformulation + cut certificates is a real formalism challenge. The closest precedent (proof-carrying code, Necula 1997) addresses program transformations, not optimization reformulations — so the bilevel-specific content is genuine. However, the actual mathematical depth is modest: the composition is a chain of implications (Φ_structural ⟹ Φ_reformulation is valid; Φ_reformulation + cut validity ⟹ end-to-end correctness). **Difficulty B is fair. The result is load-bearing for the system's trustworthiness guarantee and has mild novelty in the optimization-specific composition structure.**

**Compilability decision (Difficulty B).** *Same assessment as Approach B — inflated.* The decision procedure is trivially polynomial for the current finite set of strategies.

**Selection soundness (Difficulty A).** *Same as Approach B — trivial and correctly graded.*

**Sampling-based value-function approximation (Difficulty B).** *Same as Approach A — known approximation theory applied to a specific setting.*

### 2. Is the difficulty grading accurate?

| Result | Claimed | Assessed | Honest? |
|--------|---------|----------|---------|
| Bilevel intersection cuts (full) | C (novel) | C (honest for facets; B+ for polyhedrality alone) | **Honest on the flagship** |
| Value-function lifting | C (novel) | C (novel concept, uncertain depth) | **Honest** |
| Compiler soundness (extended) | B (hard, precedented) | B (fair — harder than B's version) | **Honest** |
| Certificate composition | B (hard, precedented) | B− (mild novelty, modest depth) | **Slightly generous** |
| Compilability decision | B (hard, precedented) | A (trivially polynomial) | **Inflated** |
| Selection soundness | A (known) | A (trivial) | **Honest** |
| Sampling approximation | B (hard, precedented) | A+/B− (known approximation theory) | **Slightly inflated** |

**Overall: The grading is the most honest of the three approaches on the hard results (intersection cuts, lifting, extended soundness) and inherits the inflation from Approach B on the softer results (compilability, selection soundness).** The net effect is a portfolio where the top 2–3 results are honestly graded and the bottom 3–4 are slightly padded. A reviewer would notice the padding on the soft results but would not feel deceived on the headline contributions.

### 3. What's the real mathematical novelty?

**(a) Genuinely new theorems requiring novel proof techniques:**
- Facet-defining conditions for bilevel intersection cuts (same as Approach A).
- Value-function lifting concept (same as Approach A).
- Extended compiler soundness that integrates cut validity (mildly novel).
- Certificate composition for optimization reformulations (mildly novel).

**(b) Straightforward extensions of known results:**
- Separation complexity, finite convergence, sampling approximation (same as Approach A).
- Compilability decision, compositional error bounds (same as Approach B).

**(c) Formalization of folklore:**
- Bilevel-infeasible set polyhedrality (same as Approach A).
- Selection soundness (same as Approach B).

### 4. Fatal mathematical risks

1. **Inherits all risks from Approach A** (facet proof may be trivial-case-only, value-function lifting may be vacuous for LP lower levels, gap closure may be negligible, dual degeneracy may explode complexity).

2. **Depth dilution risk.** By pursuing both contributions co-primarily, the approach risks delivering neither at full depth. The facet-defining proof is the hardest result in the entire project; it requires sustained mathematical effort. If implementation effort on the compiler consumes the time budget, the facet-defining proof may end up as a conjecture or a special-case result. **This is the specific risk of the hybrid: you cannot simultaneously write a deep polyhedral theory paper and build a 121K-LoC compiler system at the quality bar both audiences demand.**

3. **The "synergy" argument may not withstand scrutiny.** The claim that the compiler enables the cuts and the cuts validate the compiler is narratively appealing but mathematically hollow. The cuts do not *need* a compiler — they need a reformulated MILP and a callback API. The compiler does not *need* the cuts — it provides value via reformulation selection and certificates. The synergy is a software-architecture argument, not a mathematical one. Reviewers at mathematical venues will see through this.

4. **Certificate composition may be trivial upon closer examination.** If each reformulation pass correctly preserves bilevel optimality (proved individually), and the cuts are valid inequalities (proved by the cut theory), then the end-to-end guarantee is just transitivity of implication. The "three-way composition" may not require any novel proof technique — it may just be: "Reformulation is sound (by Theorem X); cuts are valid (by Theorem Y); therefore reformulation + cuts is sound." That is a two-line proof, not a Difficulty B result.

### 5. Math contribution grade

**Grade: B+**

*Justification:* Approach C's mathematical portfolio is essentially Approach A's math (grade B+) plus Approach B's math (grade D+), with modest integration novelty. The extended compiler soundness and certificate composition add mild depth, but the core mathematical contribution remains the facet-defining conditions and value-function lifting from the cut theory. The hybrid structure does not deepen the mathematics — it broadens the setting in which the mathematics is deployed. This is valuable for a paper but does not change the mathematical grade. The B+ reflects: if the facet-defining proof works, the math is strong (A−); if it doesn't, the math is a collection of B-level results and known theory, and the grade drops to B−.

---

## Cross-Approach Comparison

### Which approach has the deepest genuine mathematical contribution?

**Approach A**, unambiguously. By concentrating all effort on the cut theory, Approach A has the best chance of actually proving the facet-defining conditions in useful generality, completing the Gomory–Johnson extension with genuine depth, and producing a self-contained polyhedral theory result. The minimal compiler means mathematical effort is not diluted by engineering. If the math works, Approach A produces the kind of result that gets cited for 20 years. If it doesn't, there is nothing — but the question was about depth, not risk.

Approach C has the same *potential* math depth but lower *likely* depth because engineering demands will compete for effort. Approach B has no genuine mathematical depth whatsoever.

### Which approach has the most honest difficulty grading?

**Approach C** is the most honest overall, despite inheriting some inflation from Approach B on the soft results. The flagship results (intersection cuts at Difficulty C, value-function lifting at Difficulty C, extended soundness at Difficulty B) are honestly graded. The approach is transparent about risks and has go/no-go gates.

Approach A is honest on the hard results but inflates the polyhedrality theorem and separation complexity by roughly half a grade each. This is a minor sin — the hard results carry the portfolio.

Approach B is the least honest. The compiler soundness theorem and compilability decision are each inflated by a full grade. The "Difficulty B" label on what is essentially textbook correctness results reorganized into a type-system framework would not survive peer review at a mathematical venue. This is not malicious inflation — it reflects a genuine confusion between *engineering difficulty* (building a correct system is hard) and *mathematical difficulty* (proving a new theorem is hard). They are not the same thing, and the bilevel optimization community knows the difference.

### Which approach's math is most likely to actually work out?

**Approach B**, trivially — because its "math" is already known to be true. There is nothing to prove that hasn't been proved.

Among the approaches with genuine mathematical content, **Approach C** is more likely to produce *some* mathematical results because it has a fallback hierarchy: (1) try full facet-defining conditions; (2) if that fails, prove separation + finite convergence without facets; (3) if that fails, present basic bilevel intersection cuts as a computational contribution; (4) if cuts fail entirely, fall back to compiler-only. Each level is weaker but still publishable. Approach A has no fallback — if the math fails, the approach fails.

However, the probability that the *deepest* results (facet-defining conditions in full generality) work out is the same for A and C — the mathematical difficulty doesn't change based on how much compiler you build around it.

### Recommendation: Which approach's math portfolio is strongest?

**Approach A has the strongest mathematical portfolio by depth. Approach C has the strongest mathematical portfolio by risk-adjusted expected value.**

Here is my honest assessment for each scenario:

| Scenario | Best Math Portfolio | Why |
|----------|-------------------|-----|
| Facet proof succeeds (full generality) | A > C > B | A produces the purest, deepest result. C dilutes it with engineering. |
| Facet proof succeeds (special case only) | C > A > B | C's compiler contribution compensates. A's special-case-only facet result is thin for a standalone paper. |
| Facet proof fails, cuts work (≥10% gap closure) | C > B ≈ A | C has computational cut results + compiler. A has computational results but no paper venue. B has no cuts. |
| Facet proof fails, cuts fail (<5% gap closure) | B > C > A | B's compiler stands alone. C's hybrid is weakened. A has nothing. |

**My recommendation: Approach C's math portfolio is the strongest choice under uncertainty.** The expected mathematical depth is comparable to Approach A (they share the same hard results), but the variance is dramatically lower. The bilevel optimization community is small and pragmatic — a paper that proves separation complexity + finite convergence + demonstrates 15% gap closure + provides a compiler is more impactful than a paper that proves facet-defining conditions for a special case and provides a minimal implementation.

However, I want to be direct about one thing: **none of these approaches contains field-defining mathematics.** The facet-defining conditions for bilevel intersection cuts, if proved, would be a solid contribution to polyhedral combinatorics — but they extend a 50-year-old framework to a new domain rather than introducing new techniques. The value-function lifting connects two existing theories. The compiler soundness is a formalization of known correctness results. At their best, these are A-level results (excellent contributions to a specialized area). They are not A+ results that reshape how the field thinks. The bilevel optimization community will appreciate this work; the broader mathematical programming community will note it and move on.

**Final portfolio grades:**

| Approach | Math Grade | Conditional on facets proved | Conditional on facets failed |
|----------|-----------|-------|--------|
| A | **B+** | A (excellent) | C (insufficient for standalone paper) |
| B | **D+** | N/A (no facets attempted) | D+ (no math regardless) |
| C | **B+** | A− (excellent, slightly diluted) | B− (adequate for systems-oriented venue) |

---

## Appendix: Specific Claims That Warrant Scrutiny

1. **"Bilevel feasibility is defined by optimality of a subproblem — a qualitatively different structure."** This is the paper's key marketing claim. The community should push back: for LP lower levels, optimality is equivalent to complementary slackness, which IS an algebraic disjunction (xᵢsᵢ = 0). The "qualitatively different" claim is strongest for MILP lower levels where optimality has no KKT characterization. For LP lower levels — which are the only cases where the math is tractable — bilevel infeasibility reduces to a union of complementarity-defined polyhedra, which is structurally similar to split disjunctions. The novelty is real but not as dramatic as presented.

2. **"Bridges two communities that have never interacted."** (Gomory–Johnson and bilevel optimization.) This is true as a sociological observation but overstates the mathematical gulf. The Gomory–Johnson theory applies to any subadditive function; the bilevel value function is subadditive under certain conditions. The "bridge" may amount to: "we observed that V(x) satisfies the hypotheses of Theorem 3 in Johnson (1974)."

3. **"15–25% root gap closure comparable to GMI cuts."** This calibration is unreliable. GMI cuts exploit integrality structure that has no analogue in bilevel infeasibility. A bilevel intersection cut removes a point that violates follower optimality — but the LP relaxation may already nearly satisfy follower optimality for most instances, making the cut shallow. Until empirical evidence exists, any gap closure number is speculation.

4. **"The compiler soundness theorem is Difficulty B (hard but precedented)."** (Approaches B and C.) For Approach B's version, this is flatly incorrect. Case analysis over four known correctness results is Difficulty A. For Approach C's extended version (including cut validity), Difficulty B is defensible because the integration with the cut theory introduces genuine dependencies.

5. **"Certificate composition is novel — closest precedent is proof-carrying code (Necula 1997)."** The comparison to PCC is flattering but misleading. PCC handles arbitrary program transformations with potentially undecidable correctness properties. BiCut's certificate composition handles a small fixed set of optimization reformulations with well-characterized correctness conditions. The composition is an engineering contribution, not a theoretical one on par with PCC.
