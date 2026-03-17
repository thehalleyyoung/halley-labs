# BiCut Approach Debate: Skeptic and Mathematician Critiques

## Panel

- **Adversarial Skeptic:** Designated devil's advocate — attacks assumptions, identifies fatal flaws, and estimates kill probabilities for each approach. Simulates hostile peer reviews.
- **Math Depth Assessor:** Evaluates genuine mathematical novelty vs. inflated difficulty claims. Grades each approach's math contribution and identifies what is truly new vs. routine application of known theory.
- **Difficulty Assessor:** Provides honest difficulty scores, estimates genuinely novel lines of code, identifies the hardest subproblems, and assesses architectural and schedule risk.

---

## Approach A: "Cuts-First" Critiques

### Skeptic's Critique

**Fatal Flaw:** The bilevel-infeasible set characterization requires vertex enumeration of the lower-level optimal face as a function of leader decision x. For degenerate lower-level LPs — which are the norm in BOBILib network interdiction and facility location instances — the number of optimal bases at a single parametric value of x can be combinatorial. The separation oracle collapses from "<50ms per call" to intractable on precisely the hard instances where cuts matter most.

**Hidden Assumptions:**
1. The bilevel-infeasible set B̄ has a *compact* polyhedral representation — never established and likely false for non-trivial instances.
2. LP relaxation points violating bilevel feasibility are "interior" to B̄ in a useful geometric sense — boundary/degeneracy issues that would make cuts shallow are never discussed.
3. A single SCIP backend suffices for credible evaluation — the OR community expects cross-solver validation.
4. The Gomory-Johnson lifting to value-function epigraphs is tractable — extending subadditivity from Z^n to arbitrary polyhedral domains may require completely new theory with no tractability guarantee.

**Hostile Review Summary:** Facet-defining conditions proved only for non-degenerate LP lower levels (excluding the hard instances). Single-solver evaluation suspect. Value-function lifting under-developed. Recommendation: major revision.

**Kill Probability: 65%.** Zero fallback. Every component must succeed simultaneously — gap closure ≥15%, facet proof, oracle performance, lifting tractability, reviewer acceptance of single-solver results. Threading all needles has ~35% probability.

### Mathematician's Critique

**What's Genuinely Novel:**
- Facet-defining conditions for bilevel intersection cuts — the first extension of Balas's framework to optimality-defined infeasible sets. Genuinely Difficulty C, high risk.
- Value-function lifting connecting Gomory-Johnson to bilevel — creative concept, but may be vacuous for LP lower levels (reducing to the concave envelope of V) and intractable for MILP lower levels.

**What's Inflated:**
- Polyhedrality theorem (claimed C, assessed B+→B) — essentially a consequence of known parametric LP structure (Gal 1995, Adler & Monteiro 1992). The facet characterization within critical regions is the genuinely hard part.
- Separation complexity (claimed B, assessed A+/B−→B−) — a parameterized complexity argument, not a deep result. The "fixed follower dimension" qualifier does most of the work.
- Sampling-based MILP approximation — the claimed O(h²) error bound is **false** for discontinuous MILP value functions. Requires correction to L¹ norm or restriction to LP lower levels.

**Math Contribution Grade: B+.** Top-heavy portfolio: one genuinely hard result (facets), one creative-but-uncertain result (lifting), four routine results dressed up with inflated difficulty grades. If facets proved in full generality → A. If only trivial cases → B−.

### Difficulty Assessment

**Honest Difficulty: 8/10** (claimed 9/10). Difficulty is extremely concentrated in exactly two places: the facet-defining proof (9/10 alone) and the separation oracle design. If you remove the proof requirement and settle for computational cuts, drops to 6. Novel LoC honestly ~7,000–8,500 (claimed ~14,000 — about half is standard optimization infrastructure).

**Key Risks:** Low architectural risk but high mathematical risk. SCIP-only lock-in couples oracle performance to SCIP's branch-and-bound. Timeline: 6–12 months for a strong 2-person team, variance dominated by proof work. On-time delivery probability: ~35%.

### Point-Counterpoint

**Is the polyhedrality theorem "computationally vacuous"?**
- *Skeptic:* The exponential number of critical regions makes it useless as a computational foundation. The separation oracle collapses to intractable.
- *Mathematician:* **Wrong.** The oracle operates *locally* — finding the critical region for a specific point via a single LP solve — not globally enumerating all regions. This is exactly how GMI cut separation works. The Skeptic's argument proves too much.

**Does the Gomory-Johnson extension have genuine content?**
- *Skeptic:* It "may not be an extension at all" — the lattice structure of Z^n is fundamental and doesn't generalize.
- *Mathematician:* **Wrong on existence, right on risk.** The mathematical generalization is valid (subadditivity generalizes beyond lattices). But the LP-case instantiation may reduce to known constructions, making it a theoretical curiosity. The risk is triviality, not invalidity.

**Is the O(h²) error bound valid?**
- *Skeptic:* Invalid for discontinuous MILP value functions.
- *Mathematician:* **Concedes entirely.** The bound as stated is false. Requires L¹ norm or LP-only restriction.

**Single-solver evaluation:**
- *Skeptic:* Gap closure may be an artifact of SCIP's weak default cuts.
- *Mathematician:* **Partially concedes.** Single-solver is acceptable for a *theoretical* paper at Math Programming (precedent: Balas, Conforti–Cornuéjols–Zambelli). But if positioned as a *computational* contribution, multi-solver is essential. Resolution depends on whether the facet proof succeeds.

---

## Approach B: "Compiler-First" Critiques

### Skeptic's Critique

**Fatal Flaw:** BilevelJuMP.jl already covers ~80% of BiCut's functionality — bilevel modeling language, KKT/SOS1/indicator reformulation, strong duality, multiple solver backends via JuMP. BiCut's delta reduces to: automatic reformulation selection (a lookup table), correctness certificates (useful but incremental), value-function reformulation, and QP extension. This is a feature update to an existing tool, not a new software category. The CVXPY analogy is structurally dishonest.

**Hidden Assumptions:**
1. Automatic reformulation selection provides meaningful speedup — the success bar of "≥5 instances with ≥2× speedup" out of 2,600+ is embarrassingly low (0.2%).
2. Correctness certificates catch "real bugs" — they warn about textbook errors (KKT on integer lower levels) that competent users already know about. The 10% figure is manufactured by deliberately misapplying KKT.
3. Four-backend emission is a research contribution — it is integration engineering, table stakes for any modeling tool.
4. The QP extension adds meaningful scope — McCormick envelopes and SOCP reformulation for QP lower levels are well-studied.

**Hostile Review Summary:** Incremental over BilevelJuMP. Reformulation selection helps on 0.3% of instances. Certificates verify textbook conditions. Unfair comparison with BilevelJuMP's mature 25+ solver infrastructure. Recommendation: reject — reposition as a BilevelJuMP extension package.

**Kill Probability: 30%.** Buildable and testable with no existential risks. The risk is entirely "so what?" — reviewer rejection for insufficient novelty over existing tools. ~70% chance of JOC acceptance.

### Mathematician's Critique

**What's Genuinely Novel:**
- **Nothing.** No result in Approach B would be publishable as a standalone mathematical contribution.

**What's Inflated:**
- Compiler soundness theorem (claimed B, assessed A) — a case analysis over four known correctness results (Dempe 2002, Fortuny-Amat & McCarl 1981, Outrata 1998, Bard 1998) organized in a type-system framework. Engineering formalism, not mathematics.
- Compilability decision (claimed B, assessed A) — trivially polynomial: finite enumeration over known strategies with checkable preconditions.
- Compositional error bounds (claimed B, assessed A+/B−) — standard condition-number perturbation theory from numerical analysis.

**Math Contribution Grade: D+.** Excellent engineering formalized with mathematical language, but a bilevel optimization expert would learn nothing mathematically new. The D+ (rather than D) reflects clean formalization with pedagogical value.

### Difficulty Assessment

**Honest Difficulty: 4/10** (claimed 6/10). No single subproblem requires a novel algorithm. Every component builds on known techniques. Difficulty is in scale and correctness, not novelty. Genuinely novel LoC: ~6,000–7,200 (claimed ~19,300 — about a third). Hardest single subproblem: four-backend emission at 5/10. Zero genuinely hard subproblems.

**Key Risks:** Moderate architectural risk from coupling (reformulation ↔ emission leaky abstraction, certificate ↔ everything transitive dependency). On-time delivery probability: ~75%. Best-paper probability if delivered: ~10%.

### Point-Counterpoint

**Is Approach B's kill probability really only 30%?**
- *Skeptic:* The system can be built and evaluated. 70% acceptance at JOC.
- *Mathematician:* **Too optimistic.** The Skeptic's own simulated hostile review recommends outright rejection as "incremental over BilevelJuMP." If that review is realistic, at least one reviewer in a typical panel will write it. JOC papers with one reject face an uphill battle. Revised estimate: 40–50% kill probability.

**Is the "CVXPY for bilevel" framing legitimate?**
- *Skeptic:* No — CVXPY was first-in-class with millions of downstream users. BiCut is the 4th–5th bilevel tool for 500–1,000 users.
- *Mathematician:* **Concedes entirely.** The analogy fails on market size, competitive landscape, and formalism novelty.

**Is the engineering difficulty conflated with mathematical difficulty?**
- *Mathematician:* Yes. Building a correct system is hard (engineering). Proving a new theorem is hard (mathematics). They are not the same thing, and the bilevel community knows the difference. Approach B's difficulty grading systematically confuses the two.

---

## Approach C: "Hybrid" Critiques

### Skeptic's Critique

**Fatal Flaw:** The two contributions compete for resources rather than creating genuine synergy. The compiler could evaluate any cutting-plane strategy — bilevel intersection cuts aren't uniquely enabled by it. The real relationship: both share implementation time and page count, diluting each. A specific new failure mode: the cut engine must operate in reformulated MILP space while reasoning about bilevel infeasibility in original space. The bidirectional mapping through KKT reformulation is lossy (non-injective forward mapping through big-M linearization), potentially producing invalid cuts.

**Hidden Assumptions:**
1. Co-primary contributions are acceptable at top venues — OR and Math Programming have no precedent for a combined cutting-plane theory + compiler-engineering paper. "Two papers stapled together."
2. 80,000+ experimental configurations are manageable — ~289 days serial computation, ~36 days with 8-core parallelism, infeasible on laptop hardware.
3. The 2-week prototype gate adequately de-risks the cuts — it validates the math but not the engineering integration.
4. Reformulation-aware cut selection is tractable and useful — no empirical or theoretical support for the hypothesis that cut effectiveness varies significantly across reformulations.

**Hostile Review Summary:** Neither contribution done justice. Cut theory compressed (incomplete facets). Compiler evaluation compressed (incremental over BilevelJuMP). Paper too long (45+ pages). "Synergy" is editorial, not technical — cuts could be a standalone SCIP plugin. Recommendation: major revision, split into two papers.

**Kill Probability: 40%.** More failure modes than either standalone approach. Cuts can fail (~40%), integration can introduce correctness bugs (~15%), paper too unfocused for any venue (~25%). But the compiler fallback provides a safety net — if cuts fail, a compiler-only paper is still producible.

### Mathematician's Critique

**What's Genuinely Novel:**
- Same facet-defining conditions and value-function lifting as Approach A (the core mathematical content).
- Extended compiler soundness integrating cut validity — mildly novel, creates a real dependency between compiler correctness and cut theory (harder than Approach B's version).
- Certificate composition across three layers — mildly novel formalism, though may reduce to transitivity of implication upon closer examination.

**What's Inflated:**
- Certificate composition compared to proof-carrying code (Necula 1997) — the PCC comparison is "flattering but misleading." PCC handles arbitrary program transformations; BiCut handles a small fixed set of reformulations. Revised from B− down to A+.
- Compilability decision (same inflation as Approach B).
- The "synergy" argument — narratively appealing but mathematically hollow. The cuts don't need the compiler; the compiler doesn't need the cuts.

**Math Contribution Grade: B+ → B (revised after Skeptic pressure).** The portfolio is essentially Approach A's math (B+) plus Approach B's math (D+) with modest integration novelty. The hybrid structure broadens the deployment setting but does not deepen the mathematics. If facets proved → A−. If facets fail → B−.

### Difficulty Assessment

**Honest Difficulty: 7/10** (claimed 8/10). Inherits Approach A's hard math (8/10) diluted with Approach B's engineering (4/10). The integration challenges (bidirectional mapping, certificate composition, reformulation-aware selection) are genuinely novel but bounded (~500–1,200 LoC each). Novel LoC: ~11,000–13,800 — more than either A or B alone because the integration layer itself is novel.

**Key Risks:** Highest architectural risk of all three (leaky BilevelAnnotation abstraction, transitive certificate fragility, reformulation-dependent cut performance). Timeline: 10–16 months for 4–5 person team. On-time delivery: ~45%. The critical path runs through the facet-defining proof and the integration phase — integration cannot begin until both compiler and cut engine are substantially complete.

### Point-Counterpoint

**Is the backward mapping a "correctness bug waiting to happen"?**
- *Skeptic:* The KKT backward mapping is non-injective — multiple bilevel (x,y) pairs may map to the same MILP point. The bilevel feasibility check is unreliable.
- *Mathematician:* **Mathematically imprecise.** The backward mapping is a *projection* (x,y,λ,z) → (x,y), which is always well-defined and surjective. The bilevel feasibility check depends only on (x,y), not auxiliary variables. Cuts derived in (x,y)-space are trivially valid when embedded with zero coefficients on auxiliaries. **However**, the Skeptic correctly identifies a *performance* concern: cuts that ignore auxiliary variables may be weak in the full MILP space.

**Does depth dilution fatally undermine Approach C?**
- *Skeptic:* "You cannot simultaneously write a deep polyhedral theory paper and build a 121K-LoC compiler system."
- *Mathematician:* The prototype gate converts Approach C from a fixed 50/50 split to an *adaptive* strategy — strong gap closure → prioritize facets; moderate gap closure → lean on compiler. The Skeptic treats the hybrid as rigid, but the gating mechanism enables dynamic resource allocation.

**Is the "co-primary" narrative viable at any venue?**
- *Skeptic:* OR publishes theory, JOC publishes software. Neither wants "two papers stapled together."
- *Mathematician:* Concedes the venue risk is real. A 45-page paper combining polyhedral theory and compiler architecture will challenge any single reviewer pool. This is a publication risk, not a mathematical flaw.

---

## Cross-Approach Verdict

### Consensus Points (All Reviewers Agree)

1. **The facet-defining proof is the single most important and riskiest mathematical contribution** across all approaches. Its success or failure determines whether the project produces a landmark paper or a routine one.

2. **Approach B has no genuine mathematical novelty.** The Skeptic, Mathematician, and Difficulty Assessor all agree: the compiler soundness theorem is known results in new packaging (D+ math grade, 4/10 difficulty). The systematic inflation of engineering difficulty as mathematical difficulty would not survive peer review.

3. **The CVXPY analogy is untenable.** CVXPY was first-in-class for millions of users; BiCut is the 4th–5th bilevel tool for ~500–1,000 users.

4. **The 2-week prototype gate is essential.** All reviewers agree it is the single most important decision point — gap closure ≥15% validates the cut theory direction; <10% kills it.

5. **The O(h²) sampling error bound is false for MILP lower levels.** Must be corrected to L¹ norm or restricted to LP lower levels.

6. **The claimed novel LoC figures are inflated** across all approaches (roughly 50% inflation for A, 67% for B, 30% for C).

### Disputed Points (Reviewers Disagree)

| Dispute | Skeptic | Mathematician | Resolution |
|---------|---------|---------------|------------|
| Is the polyhedrality theorem computationally vacuous? | Yes — exponential critical regions make it useless | No — the separation oracle operates locally, not globally | **Mathematician's position is stronger** (GMI analogy) |
| Does the Gomory-Johnson extension have genuine content? | May not be an extension at all | Valid generalization, but LP case may be trivial | **Both partially right** — the concept is real but the instantiation is uncertain |
| Approach B kill probability | 30% | 40–50% | **Mathematician's estimate likely more accurate** given the Skeptic's own hostile review |
| Is the backward mapping a correctness bug? | Yes — non-injective, could produce invalid cuts | No — projection is trivially correct; concern is performance, not correctness | **Mathematician's position is stronger** on correctness; Skeptic raises valid performance concern |
| Can depth dilution in C be managed? | No — can't do deep theory and large system simultaneously | Yes — prototype gating enables adaptive resource allocation | **Unresolvable a priori** — depends on team composition and discipline |

### Kill Probability Comparison

| Approach | Skeptic | Mathematician (implied) | Difficulty Assessor (on-time delivery) |
|----------|---------|------------------------|---------------------------------------|
| A: Cuts-First | **65%** | ~60% (agrees within band) | ~65% fail (35% on-time) |
| B: Compiler-First | **30%** | 40–50% (argues higher) | ~25% fail (75% on-time) |
| C: Hybrid | **40%** | ~35–40% | ~55% fail (45% on-time) |

### Final Recommendation from Each Reviewer

**Adversarial Skeptic → Approach B** (reluctantly). The boring, safe, buildable option. A 70% chance of a competent JOC paper. But: "If the 2-week prototype gate shows gap closure ≥15%, I would immediately switch my bet to Approach A."

**Math Depth Assessor → Approach C.** The strongest portfolio under uncertainty. Same hard results as A (facets, lifting) but with a fallback hierarchy. B is a "safe bet on a D+ math paper — still a D+ math paper." Under any reasonable valuation where theoretical contributions are worth ≥2.5× engineering contributions, C dominates B in expected value.

**Difficulty Assessor → Approach C** (with aggressive prototype gating). Optimal difficulty profile for a best-paper artifact: depth from cut theory, breadth from compiler, novel integration contribution. Best papers need both depth and breadth. The critical caveat: requires a balanced team with both a strong theorist and strong systems engineers.
