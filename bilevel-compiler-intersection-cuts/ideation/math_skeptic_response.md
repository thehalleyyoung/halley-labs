# Math Depth Assessor — Response to Adversarial Skeptic

**Role:** Math Depth Assessor (responding to Skeptic critique)  
**Date:** 2026-03-08  
**Purpose:** Point-by-point engagement with the Skeptic's mathematical critiques. Concessions where earned, pushback where warranted.

---

## Agreements (Where the Skeptic is Right)

### 1. The polyhedrality theorem's computational vacuity concern is partially valid (Approach A, §6 bullet 1)

The Skeptic says: "The theorem is technically correct but computationally vacuous — it establishes structure that cannot be efficiently exploited."

**I concede the core point.** In my prior review I called this result "partially folklore" and downgraded it from the claimed Difficulty C to B+. The Skeptic goes further and argues that the exponential number of critical regions renders the polyhedrality claim useless *as a computational foundation*. This is a fair sharpening of my own critique. The polyhedrality theorem is necessary as a *theoretical* foundation — it guarantees the intersection cut framework is well-defined — but the Skeptic is correct that the number of polyhedra in the decomposition of B̄ can be exponential in degenerate cases, and the proposal does not adequately address how the separation oracle handles this. I should have been harder on this in my initial review.

### 2. The O(h²) error bound is invalid for discontinuous MILP value functions (Approach A, §6 bullet 4)

The Skeptic says: "The O(h²) error bound assumes V(x) is Lipschitz continuous, which it is not for integer lower levels."

**I concede this entirely.** This is a genuine mathematical error in the proposal's claims. Piecewise-linear approximation with O(h²) pointwise error bounds requires at minimum Lipschitz continuity, and more typically C² smoothness for the quadrature-style error analysis to apply. The MILP value function V(x) can have discontinuities at points where the optimal integer solution changes combinatorially. At best you get an L¹ error bound (in an integral sense, averaging over the domain), not a pointwise O(h²) bound. I flagged this in my review as "the bilevel-specific content is that V(x) is discontinuous, so the O(h²) bound holds only in an integral/average sense," but I should have been more explicit: **the claimed error bound as stated is false for MILP lower levels.** The correct statement would require: (a) restricting to LP lower levels (where V(x) is piecewise linear and continuous, making the bound trivial — the piecewise linear overestimator is exact on a sufficiently fine grid), or (b) stating the error in total variation or L¹ norm, not pointwise.

### 3. The single-solver evaluation concern is legitimate (Approach A, §2 bullet 3)

The Skeptic says: "The OR community expects cross-solver validation."

**I concede this as a practical matter.** While this is not a *mathematical* critique (the theorems are true regardless of which solver evaluates them), the Skeptic is correct that a reviewer at Mathematical Programming or IPCO will ask about solver dependence. Gap closure percentages are heavily influenced by the baseline solver's default cuts. SCIP's default cut pool is weaker than Gurobi's on many instance classes, which could inflate the apparent contribution of bilevel-specific cuts. This is a valid concern about the experimental methodology, not about the mathematics — but it materially affects whether the paper is publishable.

### 4. Finite convergence degrades after cuts create degeneracy (Approach A, §6 bullet 3)

The Skeptic says: "Non-degeneracy fails for LP relaxations of MILPs after adding cuts."

**I concede this is a real subtlety.** Adding cutting planes to an LP relaxation generically creates degenerate bases (the new constraint is initially tight at the current LP optimum, creating a degenerate vertex). The finite convergence theorem "under non-degeneracy" therefore applies to the initial configuration but may not guarantee termination of the full iterative cut-add-resolve loop. This is a known issue in cutting-plane theory — Gomory's finite convergence for GMI cuts required lexicographic pivoting or perturbation to handle the degeneracy that cuts themselves introduce. The proposal should have addressed this explicitly. The result as stated (Difficulty B) is correct for the non-degenerate case, but the non-degeneracy assumption is violated by the algorithm's own operation, which is a real gap.

### 5. BilevelJuMP covers substantial ground (Approach B, §1)

The Skeptic says: "BiCut's delta over BilevelJuMP reduces to automatic reformulation selection, certificates, value-function reformulation, and the QP extension."

**I largely concede this.** In my prior review I gave Approach B a math grade of D+ precisely because there is no new mathematics. The Skeptic's characterization of the delta is accurate. I disagree only on the *significance* of the delta (see Disagreements below), not on the *description* of what is new.

### 6. The CVXPY analogy is overstated (Approach B, §4)

The Skeptic says: "Calling it 'the first bilevel compiler' requires defining 'compiler' narrowly enough to exclude BilevelJuMP, PAO, and GAMS EMP."

**I concede this.** The CVXPY analogy fails on all three dimensions the Skeptic identifies: market size, competitive landscape, and formalism novelty. CVXPY was genuinely first-in-class; BiCut is an improvement on existing tools. The "compiler" framing is marketing, not substance. I should have flagged this more aggressively in my initial review — I was focused on the mathematical content and let the framing claims pass without sufficient scrutiny.

### 7. The certificate composition may be trivial (Approach C, my prior review §4.4)

The Skeptic says: "The 'three-way composition' may not require any novel proof technique — it may just be transitivity of implication."

**I concede this is a serious risk.** I raised this concern myself in my prior review: "If each reformulation pass correctly preserves bilevel optimality and the cuts are valid inequalities, then the end-to-end guarantee is just transitivity of implication." The Skeptic is right that the Necula (1997) comparison flatters the result. Proof-carrying code handles arbitrary program transformations; BiCut handles a fixed finite set of reformulations with known correctness proofs. The composition is a conjunction of certified preconditions, not a general proof framework. I maintain that there is *mild* novelty in formalizing the composition for optimization reformulations (it has not been done before in this domain), but the Skeptic is right that it is not Difficulty B. I revise my assessment to A+/B−.

### 8. The "co-primary" framing has venue risk (Approach C, §2 bullet 1)

The Skeptic says: "The paper must be submitted to one venue, and either venue's reviewers will find half the paper outside their expertise."

**I concede this is a real publication risk**, though I note it is not a *mathematical* critique. The Skeptic is correct that OR, Math Programming, and JOC each have strong venue-specific expectations. A 45-page paper combining polyhedral theory and compiler architecture will challenge any single reviewer pool. This does not affect my assessment of the mathematical *content*, but it does affect the risk-adjusted value of the mathematical portfolio.

### 9. The Skeptic's kill probability for Approach A (65%) is defensible

I will not argue that the Skeptic is wrong on the overall risk assessment for Approach A. My initial review gave Approach A a feasibility score of 4/10, which translates to roughly 60% failure probability. The Skeptic's 65% is within the uncertainty band of my own assessment. The zero-fallback structure is the dominant risk factor, and the Skeptic correctly identifies it.

---

## Disagreements (Where the Skeptic is Wrong or Unfair)

### 1. The polyhedrality theorem is NOT "computationally vacuous" — the Skeptic confuses enumeration with exploitation (Approach A, §6 bullet 1)

The Skeptic claims the polyhedrality theorem "establishes structure that cannot be efficiently exploited" because the number of critical regions is exponential. **This conflates two distinct operations: enumerating all critical regions vs. finding the critical region containing a specific point.**

The separation oracle does NOT need to enumerate all critical regions of the parametric LP. It needs to:
(a) Given a specific LP relaxation point (x̂, ŷ), determine which critical region x̂ belongs to — this is a single LP solve (find the optimal basis of the follower LP at x̂), which is polynomial.
(b) Compute the boundary of B̄ along rays from (x̂, ŷ) — this requires solving O(d) auxiliary LPs where d is the follower dimension, each polynomial.

The exponential number of total critical regions is irrelevant because the separation oracle operates *locally*, not globally. This is exactly the same situation as GMI cut separation: the number of lattice-free sets is infinite, but the separation oracle only needs the one containing the current fractional point. The Skeptic's argument proves too much — by the same logic, GMI cuts would be "computationally vacuous" because the number of possible Gomory cuts is combinatorial in the number of rows.

**The polyhedrality theorem is not vacuous; it is the theoretical guarantee that the local computation (ray-boundary intersection) is well-defined and produces a valid cutting plane.** Without it, you cannot prove the cut is valid. With it, the separation oracle need never enumerate the full decomposition.

I do concede that the *global* polyhedral description of B̄ is computationally intractable. But the proposal never claims to construct it globally — the separation procedure is inherently local.

### 2. The Skeptic's vertex enumeration complexity argument is misleading (Approach A, §1)

The Skeptic writes: "a lower-level LP with 50 constraints and 20 variables can have O(C(50,20)) ≈ 4.7 × 10^13 vertices of the dual feasible region."

This is technically correct but **misleadingly applied**. The separation oracle does not enumerate all dual vertices. It solves a single LP to find the *current optimal basis* at x̂, then performs ray-tracing against the boundaries of the critical region defined by that basis. The critical region is defined by the optimality conditions for that basis — a polyhedron described by O(m) inequalities, not O(C(m,n)) vertices.

The Skeptic's argument about degeneracy is more relevant: if the lower-level LP at x̂ has multiple optimal bases (degenerate), the separation oracle must handle the union of critical regions sharing the same optimal value. This is a real complication — but it means the separation oracle needs to solve a few additional LPs to identify adjacent critical regions at a degenerate point, not that it must enumerate all C(50,20) bases.

**The correct complexity concern is:** for degenerate lower levels, the separation oracle may need to test multiple adjacent critical regions to find the deepest cut. The number of adjacent regions at a degenerate point is bounded by the degree of degeneracy, which for typical BOBILib instances is much smaller than the total number of bases. The Skeptic's worst-case bound is real but practically irrelevant for the same reason that simplex method worst-case exponential complexity is practically irrelevant — the algorithm's behavior on structured instances is far better than the worst case.

### 3. The Skeptic is wrong that Gomory-Johnson lifting "may not be an extension at all" (Approach A, §2 bullet 4)

The Skeptic says: "The Gomory-Johnson theory works because the domain is Z^n — a lattice with clean algebraic structure. Extending it to arbitrary polyhedral domains... is a fundamentally different problem."

This mischaracterizes the lifting operation. The Gomory-Johnson theory does not *require* a lattice domain — it requires a group structure, which the original papers exploit for integer programming. But the *mathematical content* of lifting (constructing maximal valid functions from subadditivity conditions) generalizes beyond lattices. Specifically:

- For LP lower levels, V(x) is piecewise linear and convex. The epigraph epi(V) is a polyhedron. Lifting valid inequalities for the bilevel-feasible set from the value-function reformulation amounts to constructing maximal valid inequalities for the intersection of a polyhedron (the leader's feasible region) with the epigraph of a piecewise-linear function — a well-posed polyhedral problem.
- The "extension" is applying the *structural insight* of Gomory-Johnson (that maximal valid functions are characterized by subadditivity on specific subsets) to a piecewise-linear setting rather than an integer lattice setting.

**I agree with the Skeptic that this is genuinely harder than it sounds** — the piecewise-linear setting lacks the algebraic regularity of lattices, and the characterization of maximal valid lifting functions in this setting is open. But the Skeptic's claim that the extension "may not be an extension at all" is wrong. It is a meaningful mathematical generalization. My concern (stated in my prior review) is different: the extension may be *trivial* for LP lower levels (reducing to the concave envelope of V, which is a known construction) while being *intractable* for MILP lower levels. The risk is triviality or intractability, not non-existence.

### 4. The Skeptic overstates the backward mapping problem for KKT reformulation (Approach C, §1)

The Skeptic identifies a specific failure mode: "Mapping an LP relaxation point back through the KKT reformulation requires inverting the big-M linearization — which is lossy (multiple bilevel (x,y) pairs may correspond to the same MILP point via different big-M encodings)."

This is **mathematically imprecise**. The KKT reformulation introduces dual variables λ and complementarity binaries z. The forward mapping is:
- (x, y) → (x, y, λ*(x,y), z*(x,y)) where λ* is the KKT multiplier and z* encodes complementarity.

The backward mapping is:
- (x, y, λ, z) → (x, y) [simply project out the auxiliary variables].

The backward mapping is a *projection*, which is always well-defined and surjective. The Skeptic's concern about non-injectivity goes the wrong direction: the forward mapping may be non-injective (multiple (λ, z) values for the same (x, y) in degenerate cases), but the backward mapping (projection) is always well-defined.

The bilevel feasibility check operates as: given MILP LP relaxation point (x̂, ŷ, λ̂, ẑ), check whether (x̂, ŷ) is bilevel-feasible by evaluating V(x̂) and comparing ĉᵀŷ to V(x̂). This check depends only on (x̂, ŷ), not on (λ̂, ẑ), so the backward mapping's non-injectivity in the auxiliary variables is irrelevant.

**Where the Skeptic has a point** is that the *cut*, once derived in the original (x, y) space, must be lifted to the full MILP space (x, y, λ, z) for the solver callback. This lifting is valid if the cut c·(x, y) ≤ c₀ is extended to c·(x, y) + 0·λ + 0·z ≤ c₀ — which is trivially valid because the cut does not reference auxiliary variables. The projection-safety concern the Skeptic raises in §6 bullet 3 is real only if the cut *references* dual variables or binaries, which a properly implemented bilevel intersection cut does not — it cuts in the original bilevel variable space.

### 5. The Skeptic's hostile review for Approach A is unfair on the "single solver" point

The Skeptic's simulated reviewer writes: "The authors must demonstrate the gap closure persists on Gurobi."

While cross-solver validation is good practice, the Mathematical Programming / IPCO reviewer community routinely accepts computational results on a single solver when the contribution is primarily theoretical. Balas's original intersection cut papers used a single solver. The recent Conforti–Cornuéjols–Zambelli monograph on cutting planes reports computational results on CPLEX alone. The SCIP ecosystem is specifically designed for custom cutting-plane research (via its constraint handler API), making it the natural choice for this work. A reviewer who demands Gurobi results for a *theoretical* cutting-plane paper is applying JOC standards to a Math Programming submission.

That said — the Skeptic's underlying concern is valid: if the paper is positioned as a *computational* contribution (gap closure numbers as the main selling point), then multi-solver validation is essential. The resolution is in the positioning: if the facet-defining proof succeeds, gap closure is supporting evidence for a theoretical paper, and single-solver results suffice. If the facet proof fails and gap closure is the main result, then the Skeptic is correct that multi-solver validation becomes necessary.

### 6. The Skeptic undervalues the separation complexity result (Approach A, §6 bullet 2)

The Skeptic says: "For BOBILib instances with 20+ follower variables, 'fixed follower dimension' is not fixed."

This critique misunderstands the purpose of parameterized complexity results. The claim "polynomial in m for fixed d" is standard FPT-style analysis. The Skeptic is correct that d is not small for all instances — but the result characterizes the *computational regime* where the cuts are viable. This is precisely what complexity results are for: they tell you when an algorithm is practical and when it is not.

The Skeptic's implicit alternative — that the result is worthless unless it is polynomial in *all* parameters simultaneously — would disqualify most parameterized complexity results in the literature. The integer programming community routinely uses fixed-parameter results (e.g., Lenstra's algorithm for ILP in fixed dimension) as theoretical foundations, even though the dependence on the fixed parameter is exponential. The separation complexity result is valuable because it delineates: for low-dimensional followers (d ≤ 5-8), separation is practical; for high-dimensional followers, it may not be. This is useful information.

**I maintain my assessment: Difficulty A+/B−.** It is a correct, useful result that applies known techniques carefully. The Skeptic's critique does not make it less useful — it identifies the regime where the result's conclusions apply.

### 7. The Skeptic's claim that reformulation-aware cut selection is "unnecessary" is unsupported (Approach C, §2 bullet 4)

The Skeptic says: "The reason there is no precedent may be that it is unnecessary."

This is speculation without mathematical backing. Different reformulations produce MILPs with fundamentally different LP relaxation polyhedra:
- KKT reformulation: the LP relaxation includes continuous relaxations of complementarity binaries and big-M constraints, creating a large feasible region with many fractional extreme points.
- Strong duality reformulation: the LP relaxation includes the dual feasible region directly, producing a tighter relaxation for certain problem structures.
- Value-function reformulation: the LP relaxation may directly encode V(x) ≤ t, producing a different relaxation geometry.

The *depth* of a cutting plane (the amount of the LP relaxation it removes) depends on the geometry of the relaxation. It is mathematically obvious that a cut optimized for one geometry will not be optimal for another. Whether this difference is *large enough to matter computationally* is an empirical question — but the Skeptic cannot dismiss it as "unnecessary" without evidence.

The absence of precedent is because no prior system could even *test* the hypothesis (no bilevel compiler with multiple reformulations and cuts existed). The absence of evidence is not evidence of absence.

### 8. The Skeptic's kill probability for Approach B (30%) is too low

The Skeptic gives Approach B a 30% kill probability, arguing "the system can be built, evaluated, and written up." But the kill mechanism is not technical failure — it is reviewer rejection. The Skeptic's own simulated hostile review for Approach B recommends **"Reject. The contribution is incremental over BilevelJuMP."** If the Skeptic believes this review is realistic (and they wrote it as a plausible hostile review), the kill probability should be higher.

In my assessment, the probability that JOC reviewers find BiCut's delta over BilevelJuMP insufficient is 40-50%, not 30%. The Skeptic underweights the reviewer-rejection risk for Approach B while overweighting it for Approach C.

### 9. The Skeptic's expected-value calculation is rigged

The Skeptic computes:
- A: 35% × 10 = 3.5
- B: 70% × 5 = 3.5
- C: 60% × 7 = 4.2

And then recommends B anyway, arguing for low variance. But the "points" are arbitrary. Why is a Math Programming paper worth exactly 2× a JOC paper? If we value publication venue impact more conservatively (Math Programming = 8, JOC = 5, OR = 7):
- A: 35% × 8 = 2.8
- B: 70% × 5 = 3.5
- C: 60% × 7 = 4.2

Or if we value breakthrough impact (Math Programming = 15, JOC = 4, OR = 8):
- A: 35% × 15 = 5.25
- B: 70% × 4 = 2.8
- C: 60% × 8 = 4.8

The Skeptic chose point values that make A and B tie, which supports the "pick B for safety" conclusion. But this is a modeling choice, not a mathematical result. **Under any reasonable valuation where a theoretical contribution is worth ≥ 2.5× an engineering contribution, C dominates B in expected value, and A may dominate B if the reader values breakthrough potential.**

---

## Revised Mathematical Assessment

### Revised Difficulty Grades

| Result | My Prior Grade | Skeptic Pressure | Revised Grade | Rationale |
|--------|---------------|-------------------|---------------|-----------|
| **Bilevel-infeasible set polyhedrality** | B+ | Valid concern about enumeration complexity | **B** | The theorem is real and necessary but the proof is a careful formalization of known parametric LP structure, not a genuinely new technique. The Skeptic's "computationally vacuous" claim is wrong (see Disagreement §1), but the result is less novel than I initially credited. |
| **Facet-defining conditions** | C | Valid concern about restricted-class fallback | **C** | No change. This is genuinely hard. The Skeptic's concern that it may only work for restricted classes is real but does not reduce the difficulty — it increases the risk. A restricted-class result is still C-difficulty; it is just a smaller C-difficulty result. |
| **Separation complexity** | A+/B− | Valid concern about fixed-dimension qualifier | **B−** | I upgrade slightly. The Skeptic's criticism pushed me to reconsider: the result is not deep, but the careful treatment of the bilevel structure within the parametric LP framework is a competent piece of work, and the regime characterization is useful. |
| **Finite convergence** | B | Valid concern about cut-induced degeneracy | **B−** | Slight downgrade. The standard non-degeneracy assumption is weaker than I credited because the algorithm itself violates it. The result needs a perturbation or lexicographic argument to be fully rigorous, which is achievable but adds genuine content. |
| **Value-function lifting** | C | Valid concern about LP-case triviality | **C/B+** | Split grade: C for MILP lower levels (genuinely hard), B+ for LP lower levels (may reduce to known constructions). The Skeptic is right that the LP case may be vacuous; I was right that the concept is novel. The truth is in between. |
| **Sampling approximation** | A+/B− | Skeptic's O(h²) criticism is correct | **A** (with corrected statement) | Downgrade. The claimed error bound is false for discontinuous functions. With a corrected statement (L¹ error or LP-only restriction), the result is straightforward approximation theory. |
| **Compiler soundness (Approach B)** | A+ | Skeptic agrees it's inflated | **A** | No change from my prior assessment. Textbook results in a new packaging. |
| **Compiler soundness (Approach C, extended)** | B | Skeptic argues still inflated | **B−** | Slight downgrade. The Skeptic is partially right that the extension to cover cuts is modest. But it is not trivial — the cut validity depends on the polyhedral characterization, creating a real dependency chain. |
| **Certificate composition** | B− | Skeptic argues it may be trivial | **A+** | Downgrade from my prior assessment. The Skeptic's point about transitivity of implication is well-taken. The composition may be non-trivial only if the certificate format must support *independent verification* (i.e., a verifier who does not trust BiCut's analysis), which adds encoding challenges. But the mathematics is modest. |
| **Compilability decision** | A | Agreement it's trivially polynomial | **A** | No change. |

### Revised Mathematical Risk Assessment

| Approach | Prior Math Grade | Revised Grade | Key Risk Change |
|----------|-----------------|---------------|-----------------|
| **A: Cuts-First** | B+ | **B+** | Unchanged. The Skeptic's valid critiques (O(h²) bound, single-solver, cut-induced degeneracy) are offset by the Skeptic's invalid critiques (polyhedrality is not vacuous, separation oracle does not require global enumeration). The portfolio remains top-heavy: one genuinely hard result (facets), one creative-but-uncertain result (lifting), and supporting results that range from B− to A. |
| **B: Compiler-First** | D+ | **D+** | Unchanged. The Skeptic and I agree completely on the mathematical content: there is none. |
| **C: Hybrid** | B+ | **B** | Slight downgrade. The Skeptic successfully argued that certificate composition is less novel than I credited, and the Necula comparison is indeed misleading. The extended compiler soundness (B−) and certificate composition (A+) are less impressive upon scrutiny. The portfolio is carried entirely by the facet-defining conditions (C) and value-function lifting (C/B+), which are shared with Approach A. |

### Updated Recommendation

**My recommendation is unchanged: Approach C's math portfolio is the strongest choice under uncertainty.**

The Skeptic's critique sharpened my understanding of the risks but did not change the fundamental calculus:

1. **Approach A** has the deepest potential math but the Skeptic correctly identifies that the "narrow viability corridor" is real. The separation oracle's practical performance on degenerate instances is a genuine concern I previously underweighted.

2. **Approach B** has no mathematical contribution. The Skeptic and I are in complete agreement on this point. The Skeptic recommends B as the "safe bet," but a safe bet on a D+ math paper is still a D+ math paper.

3. **Approach C** maintains access to the same hard results as A (facets, lifting) while providing the fallback hierarchy I identified in my prior review. The Skeptic's valid critiques reduced C's grade from B+ to B — but B is still dramatically better than D+, and the fallback structure means the *expected* mathematical contribution is higher than A's despite A's higher ceiling.

**The Skeptic's strongest argument** is that Approach C risks depth dilution — the facet-defining proof gets less attention because engineering demands consume time. This is real. But the Skeptic's recommendation (pick B) does not solve this problem — it eliminates the mathematical contribution entirely. If the goal is to produce a paper with genuine mathematical content, Approach C with aggressive prototype gating is still the optimal strategy.

---

## Key Disputed Points That Affect Approach Selection

### Dispute 1: Is the polyhedrality theorem "computationally vacuous"?

**Skeptic's position:** The exponential number of critical regions makes the polyhedrality theorem useless as a computational foundation. The separation oracle "collapses from <50ms per call to intractable."

**My position:** The separation oracle operates *locally* (finding the critical region for a specific point), not *globally* (enumerating all regions). Local critical region identification is a single LP solve, polynomial in all parameters. The exponential total count is irrelevant to the separation procedure's complexity.

**Why my position is better supported:** The Skeptic's argument conflates global enumeration with local computation. Every cutting-plane separation oracle in the literature operates locally — GMI cuts do not enumerate all lattice-free sets, split cuts do not enumerate all split disjunctions, and bilevel intersection cuts need not enumerate all critical regions. The Difficulty Assessor's review supports my position: the separation oracle's complexity is bounded by the local parametric LP solve, not the global critical region count. The Skeptic's own cite (Khachiyan et al. 2008) on #P-hardness of vertex enumeration is about *counting all vertices*, not about *finding the vertex corresponding to a given basis* — a fundamentally different problem.

**Impact on approach selection:** If the Skeptic is right, Approach A and C are dead (the cuts are impractical). If I am right, Approach A and C remain viable. This is the single most important mathematical dispute. I am confident in my position, but the 2-week prototype gate would resolve it empirically.

### Dispute 2: Does the Gomory-Johnson extension have genuine content for LP lower levels?

**Skeptic's position:** "The 'Gomory-Johnson extension' sounds impressive but collapses to a restatement of LP duality for the continuous case."

**My position:** The extension is a genuine mathematical generalization (from lattice domains to polyhedral domains), but the LP-case instantiation may indeed reduce to known constructions (concave envelope of V). The risk is triviality, not invalidity.

**Why my position is better supported:** The Skeptic claims the extension "may not be an extension at all." This is wrong — the mathematical formulation is a strict generalization. However, I concede the Skeptic's underlying concern: if the LP-case lifting function reduces to the concave envelope, the "Gomory-Johnson extension" adds no cutting power beyond the value-function reformulation itself. This would make the lifting a theoretical curiosity rather than a practical tool. The resolution is empirical: do lifted cuts close more gap than unlifted cuts on LP-lower-level instances?

**Impact on approach selection:** If lifting is vacuous for LP lower levels, Approach A loses one of its two headline mathematical contributions, making the paper thinner. Approach C is less affected because the compiler carries weight independently. This dispute favors C over A.

### Dispute 3: Is Approach B's kill probability 30% or higher?

**Skeptic's position:** Approach B has a 30% kill probability because "the system can be built, evaluated, and written up."

**My position:** The kill probability is 40-50% because the reviewer-rejection risk from "incremental over BilevelJuMP" is higher than the Skeptic credits, and the Skeptic's own simulated hostile review recommends outright rejection.

**Why my position is better supported:** The Skeptic wrote a plausible hostile review that concludes "Reject. The contribution is incremental over BilevelJuMP. The paper should be repositioned as a BilevelJuMP extension package." If this review is realistic (and the Skeptic wrote it to be realistic), then at least one reviewer in a typical three-reviewer panel will write something similar. JOC papers with one "reject" review face an uphill battle. The Skeptic's 70% acceptance estimate implies that this hostile review would be overruled by two positive reviews — possible, but optimistic for a paper whose mathematical contribution the Skeptic and I agree is D+.

**Impact on approach selection:** If B's kill probability is 40-50% rather than 30%, the expected-value gap between B and C narrows (B: 55% × 5 = 2.75; C: 60% × 7 = 4.2). Under my estimates, C dominates B by a wider margin, and the Skeptic's "safe bet on B" argument weakens.

### Dispute 4: Is the backward mapping for KKT reformulation a "correctness bug waiting to happen"?

**Skeptic's position:** The KKT backward mapping is non-injective, making the bilevel feasibility check unreliable and potentially producing invalid cuts.

**My position:** The backward mapping is a projection (x, y, λ, z) → (x, y), which is always well-defined. The bilevel feasibility check depends only on (x, y), not on auxiliary variables. Cuts derived in (x, y)-space are trivially valid when embedded in the full MILP space with zero coefficients on auxiliary variables.

**Why my position is better supported:** The Skeptic's concern confuses non-injectivity of the forward mapping (multiple auxiliary variable values for the same bilevel solution) with problems in the backward mapping (projection). Projection is a linear operation that does not introduce correctness errors. The cut c·(x,y) ≤ c₀ extended to c·(x,y) + 0·(λ,z) ≤ c₀ is valid in the MILP if and only if c·(x,y) ≤ c₀ is valid for all bilevel-feasible (x,y), which is exactly what the bilevel intersection cut procedure guarantees. No additional "projection safety" argument is needed beyond the trivial observation that adding zero-coefficient variables to an inequality preserves validity.

**However:** The Skeptic raises a legitimate *performance* concern that I did not initially address. The cut c·(x,y) + 0·(λ,z) ≤ c₀, while valid, may be *weak* in the full MILP space because it does not constrain the auxiliary variables. A stronger cut would also constrain (λ, z), exploiting the KKT structure. Deriving such strengthened cuts is a genuinely harder problem and is where "reformulation-aware cut selection" could add value. So the Skeptic is wrong about correctness but points toward a real performance question.

**Impact on approach selection:** If the Skeptic is right about a correctness bug, Approach C has a fatal integration flaw. If I am right (the projection is trivially correct), the concern reduces to a performance optimization question, and Approach C's integration challenge is hard but tractable. This dispute is decisive: a correctness bug kills C, a performance concern merely shapes its implementation.

### Dispute 5: Does depth dilution fatally undermine Approach C's mathematical contribution?

**Skeptic's position:** "You cannot simultaneously write a deep polyhedral theory paper and build a 121K-LoC compiler system at the quality bar both audiences demand."

**My position:** The depth dilution risk is real but manageable through the prototype gate. If the prototype shows strong gap closure (≥15%), the facet-defining proof becomes the priority and the compiler is descoped to essential components. If gap closure is moderate (10-15%), the compiler carries more weight and the facet proof is pursued as a bonus. The hybrid structure enables *adaptive* resource allocation, not rigid splitting.

**Why my position is better supported:** The Skeptic treats the hybrid as a fixed 50/50 split of effort. But the proposal explicitly includes a 2-week prototype gate that determines resource allocation. This gate converts Approach C from a fixed split to an *adaptive* strategy. The Difficulty Assessor supports this: "Approach C with aggressive prototype gating" is their recommendation. The Skeptic's depth-dilution argument assumes the team cannot prioritize after the gate — an organizational critique, not a mathematical one.

**Impact on approach selection:** If depth dilution is fatal, Approach A is better for math depth and Approach B is better for safety. If it is manageable (via gating), Approach C dominates both. This is ultimately a project management question, not a mathematical one, but it determines whether C's mathematical portfolio can be realized.
