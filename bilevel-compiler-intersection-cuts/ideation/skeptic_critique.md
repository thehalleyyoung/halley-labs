# Adversarial Skeptic Critique: BiCut Approaches

**Role:** Designated devil's advocate — attacking assumptions and identifying fatal flaws  
**Reviewed documents:** `approaches.md`, `problem_statement.md`, `depth_check.md`  
**Date:** 2026-03-08

---

## Approach A: "Cuts-First" — Deep Cutting Plane Theory

### 1. Fatal Flaw

The bilevel-infeasible set characterization (the polyhedrality theorem) requires vertex enumeration of the lower-level optimal face as a function of the leader's decision x. For LP lower levels, the number of dual vertices is bounded by C(m, n) — binomial in the number of constraints m and variables n — which is exponential even for moderate-sized followers. The proposal hand-waves this with "structured (dual degeneracy determines the combinatorial complexity)," but **BOBILib instances with degenerate lower levels comprise a large fraction of the library** (lower-level problems with redundant constraints, near-parallel hyperplanes, or integer restrictions that create flat optimal faces are the norm, not the exception, in network interdiction and facility location bilevel models). In degenerate cases, the number of optimal bases at a single parametric value of x can be combinatorial, and the parametric partition of x-space into critical regions has complexity exponential in the number of degenerate constraints. Concretely: a lower-level LP with 50 constraints and 20 variables can have O(C(50,20)) ≈ 4.7 × 10^13 vertices of the dual feasible region. Even restricting to optimal vertices at a given x, degeneracy means the vertex enumeration step inside the separation oracle is not "an auxiliary LP" — it is a vertex enumeration problem, which is #P-hard in degenerate settings (Khachiyan et al. 2008). The entire separation oracle collapses from "<50ms per call" to "intractable" on precisely the instances where cuts matter most (hard instances with large integrality+bilevel gaps tend to have degenerate lower levels).

### 2. Hidden Assumptions

- **Assumption: the bilevel-infeasible set B̄ decomposes into a "manageable" number of polyhedra.** The proposal asserts B̄ is a finite union of polyhedra via vertex enumeration. This is technically true but useless if the number of polyhedra is exponential. The practical utility of intersection cuts depends on B̄ having a *compact* polyhedral representation — a property that is never established and may fail for most non-trivial instances.

- **Assumption: LP relaxation points that violate bilevel feasibility are "interior" to B̄ in a useful geometric sense.** Balas's original intersection cut requires a point strictly interior to the convex set S from which rays are cast. If the LP relaxation solution lies on or near the boundary of B̄ (which happens when the LP relaxation is already nearly bilevel-feasible), the ray intersection distances become numerically degenerate and the resulting cut has negligible depth. The proposal never discusses this boundary/degeneracy issue.

- **Assumption: a single SCIP backend is sufficient for credible computational evaluation.** The OR community expects cross-solver validation. A result that only works on SCIP — known for having slower LP solves than Gurobi on many instance classes — will be questioned. Reviewers will ask: "Is the gap closure an artifact of SCIP's weak default cuts?"

- **Assumption: the Gomory-Johnson lifting extension to value-function epigraphs is tractable.** The proposal frames this as "extending subadditivity theory to piecewise-linear functions defined on polyhedral domains rather than the integers." But the Gomory-Johnson theory works because the domain is Z^n — a lattice with clean algebraic structure. Extending it to arbitrary polyhedral domains where the function V(x) itself requires solving an LP to evaluate is a fundamentally different problem. The "extension" may not be an extension at all but a completely new theory with no guarantee of tractability.

### 3. Competitor Response (Hostile Review)

> **Review of "Bilevel Intersection Cuts: Extending the Balas Framework to Optimality-Defined Infeasible Sets"**
>
> The paper proposes to extend Balas's 1971 intersection cut framework to bilevel optimization. While the idea is interesting in principle, the execution has critical gaps.
>
> **Major concern 1:** The facet-defining conditions (Theorem 3) are proved only for LP lower levels with unique optimal follower response (non-degenerate lower level). This restriction excludes precisely the hard instances where cuts are needed — degenerate lower levels are the norm in network interdiction models (which constitute ~35% of BOBILib). The "general" theorem is actually a restricted-class result dressed up as a general framework.
>
> **Major concern 2:** The computational results are on a single solver (SCIP) with no cross-validation. The 17% geometric mean gap closure is encouraging, but I cannot rule out that this is an artifact of SCIP's relatively weak default cuts on bilevel reformulations. The authors must demonstrate the gap closure persists on Gurobi, which has stronger default cutting planes that may already capture some of the bilevel structure implicitly.
>
> **Major concern 3:** The value-function lifting (Section 5) is under-developed. The connection to Gomory-Johnson is stated but the subadditivity conditions are verified only computationally on small examples (≤5 follower variables). A theoretical characterization of when the lifting function is maximal valid is missing. This section reads as a promising direction, not a result.
>
> **Recommendation: Major revision.** Extend the facet proof to degenerate lower levels or honestly restrict the scope. Add Gurobi experiments. Develop the lifting theory or remove it.

### 4. Value Inflation

The proposal claims the cuts would be "cited for decades as the first intersection cut extension to bilevel." This dramatically overstates the citation impact. Bilevel optimization papers rarely exceed 200 citations over their lifetime; the entire bilevel optimization literature is a small community. The Balas 1971 paper has ~2000 citations accumulated over 55 years — partly because integer programming is a massive field. An extension to bilevel, serving 100-200 researchers, will not achieve comparable impact. A realistic citation trajectory is 50-150 citations over 10 years, which is a solid paper but not "cited for decades" in any meaningful sense.

The claim that "the integer programming community (IPCO, MIP workshops) has been searching for new families of valid inequalities" conflates interest in new cut families for IP with interest in bilevel-specific cuts. The IP community wants cuts that apply broadly (split, cross, multi-row). Cuts that only apply to bilevel reformulations — a niche structure — will generate curiosity, not adoption.

### 5. Difficulty Inflation or Deflation

**Inflated:** The difficulty score of 9/10 counts the facet-defining proof at Difficulty C and the value-function lifting at Difficulty C. But the proposal's own fallback plan for the facet proof is "prove it for a restricted class" (non-degenerate LP lower levels). If the restricted-class proof is the actual deliverable, the difficulty drops to B+ at most — it becomes Balas's proof technique applied to a cleaner-than-general setting. The Difficulty C rating assumes the general result is achieved, which the proposal itself doubts.

**Deflated:** The engineering challenge of making the separation oracle fast enough is underrated. The proposal allocates 11K LoC for the intersection cut engine but the >90% cache hit rate assumption has no empirical precedent in any cutting-plane implementation. Standard GMI cut separation does not use parametric sensitivity caching because LP basis changes after branching are typically large enough to invalidate the cache. The claim that "consecutive LP relaxation points differ by small perturbations after branching" is true in early rounds but false deep in the branch-and-bound tree where basis changes are radical.

### 6. Math Critique

- **The polyhedrality theorem** (B̄ is a finite union of polyhedra) relies on the lower-level LP having finitely many bases, which is true but the number of bases is exponential. The theorem is technically correct but computationally vacuous — it establishes structure that cannot be efficiently exploited.

- **The separation complexity claim** ("polynomial in the number of lower-level constraints for fixed follower dimension") hides an exponential dependence on follower dimension in the "fixed" qualifier. For BOBILib instances with 20+ follower variables (common in interdiction models), "fixed follower dimension" is not fixed — it is the problem's primary scaling dimension. The complexity result is polynomial in m for fixed n, but n is not fixed in practice.

- **Finite convergence** "under non-degeneracy" is a standard qualifier, but non-degeneracy fails for LP relaxations of MILPs after adding cuts (cutting planes generically create degenerate bases). The convergence theorem applies to the first few rounds of cuts but cannot guarantee termination of the full cut loop.

- **The Gomory-Johnson lifting** requires evaluating V(x) to construct the subadditive function. For MILP lower levels, V(x) is discontinuous and evaluating it at a single point is NP-hard. The "sampling-based approximation with provable error bounds O(h²)" assumes V(x) is Lipschitz continuous, which it is not for integer lower levels (where V(x) can jump discontinuously). The O(h²) error bound is invalid for discontinuous functions.

### 7. Kill Probability

**65% probability of failing to produce a publishable paper at any venue.**

Justification: The approach has zero fallback. The narrow viability corridor identified by the depth check panel is real and unresolved. The 2-week prototype gate is necessary but insufficient — even if gap closure passes 10%, the facet-defining proof may fail for general instances, the value-function lifting may be intractable, and the single-solver evaluation may not survive reviewer scrutiny. A 35% chance of threading all needles simultaneously (gap closure ≥ 15%, facet proof for at least the restricted class, acceptable computational overhead, reviewer acceptance of single-solver results) seems generous.

---

## Approach B: "Compiler-First" — Full Compiler Architecture

### 1. Fatal Flaw

**BilevelJuMP.jl already exists and covers 80% of this system's functionality.** BilevelJuMP (Dias Garcia et al., INFORMS JOC 2023) provides: a bilevel modeling language in Julia, KKT reformulation with per-constraint mode selection (big-M, SOS1, indicator), strong duality reformulation, multiple solver backends via JuMP's solver-agnostic interface (which covers Gurobi, SCIP, HiGHS, CPLEX, and more). BiCut's delta over BilevelJuMP reduces to: (1) automatic reformulation selection (a table lookup + cost model — not a research contribution), (2) correctness certificates (useful but incremental), (3) value-function reformulation (the one strategy BilevelJuMP doesn't do well), and (4) the QP extension. This is a *feature update* to an existing tool, not a new software category. The CVXPY analogy is structurally dishonest: CVXPY was the first DCP compiler; BiCut is the fourth or fifth bilevel reformulation tool. Calling it "the first bilevel compiler" requires defining "compiler" narrowly enough to exclude BilevelJuMP, PAO, and GAMS EMP — a definition that exists only to inflate novelty.

### 2. Hidden Assumptions

- **Assumption: automatic reformulation selection provides meaningful speedup.** The proposal claims "≥5 instances with ≥2× speedup" for selection over default KKT. This is an embarrassingly low bar — 5 out of 2600+ instances. If the selection engine only helps on 0.2% of instances, it is not a research contribution; it is a convenience feature. The hidden assumption is that problem structure is sufficiently diverse across BOBILib that different reformulations dominate on different instance classes. But in practice, KKT with SOS1 encoding is the default for good reason — it works well on the vast majority of LP-lower-level instances, and strong duality is only preferable for a small subclass with very tight dual bounds.

- **Assumption: correctness certificates catch "real bugs" on ≥10% of instances.** This conflates "CQ violation on integer lower levels" (which is a known theoretical issue, not a "bug") with actual practical errors. Practitioners who work with integer lower levels already know KKT doesn't apply — they use enumeration or MibS directly. The certificate is warning users about a mistake that competent users don't make. The 10% figure is manufactured by deliberately applying KKT to instances where it is known to be invalid.

- **Assumption: four-backend emission is a research contribution rather than integration work.** Writing adapters for four solver APIs is engineering, full stop. AMPL has been emitting to 50+ solvers for 30 years. JuMP has a solver-agnostic backend layer. Pyomo dispatches to dozens of solvers. The fact that BiCut emits to four backends is not novel — it is table stakes for an optimization modeling tool. Dressing this up as a "genuinely difficult engineering breakthrough" is padding.

- **Assumption: the QP extension adds meaningful scope.** QP lower levels with McCormick envelopes for bilinear terms are well-studied (Fortuny-Amat & McCarl, McCormick 1976). The SOCP reformulation for convex QP lower levels is standard. Adding 12K LoC for known theory applied to a known setting does not increase the system's intellectual contribution — it increases its engineering surface area.

### 3. Competitor Response (Hostile Review)

> **Review of "BiCut: A Solver-Agnostic Bilevel Optimization Compiler with Correctness Certificates"**
>
> The paper presents a bilevel optimization compiler with automatic reformulation selection, correctness certificates, and four solver backends. While the system is competently engineered, I struggle to identify a research contribution that goes beyond what BilevelJuMP.jl (Dias Garcia et al., JOC 2023) already provides.
>
> **Major concern 1:** The "automatic reformulation selection" is a lookup table mapping structural features to reformulation strategies. Table 3 shows it matches expert choice on 94% of instances and beats default-KKT on exactly 7 instances (0.3% of BOBILib). This is a convenience feature, not a contribution to optimization methodology.
>
> **Major concern 2:** The correctness certificates are sound but trivially so — they verify preconditions that are textbook knowledge. The claim that certificates "catch real bugs" is manufactured: the authors deliberately apply KKT reformulation to integer lower levels (where it is known to be invalid) and then count the certificate's rejection as a "caught bug." A researcher who applies KKT to an integer lower level has made a textbook error, not a subtle bug that requires machine verification to detect.
>
> **Major concern 3:** The comparison with BilevelJuMP is unfair. BilevelJuMP uses JuMP's mature solver-agnostic infrastructure (supporting 25+ solvers) while BiCut supports four. BiCut claims novelty for "solver-agnostic emission" as if JuMP's MathOptInterface doesn't exist. The appropriate comparison is: BiCut adds reformulation selection and certificates to BilevelJuMP's existing capabilities. Is that sufficient for a JOC paper?
>
> **Recommendation: Reject.** The contribution is incremental over BilevelJuMP. The paper should be repositioned as a BilevelJuMP extension package, not a standalone system.

### 4. Value Inflation

The "CVXPY for bilevel optimization" framing is the most egregious value inflation across all three approaches. CVXPY succeeded because:
1. Convex optimization has millions of downstream users (ML, control, finance, signal processing).
2. No comparable tool existed when CVXPY launched (CVX was MATLAB-only).
3. DCP rules were a genuinely new formalism that expanded the set of problems users could express.

None of these conditions hold for BiCut:
1. Bilevel optimization has 500-1000 users at ceiling.
2. BilevelJuMP, PAO, GAMS EMP, and MibS all exist.
3. BiCut's type system is not a new formalism — it is an engineering implementation of known bilevel structure theory.

The "reproducibility layer" reframing is more honest but still overstated. Reviewers who evaluate bilevel papers can run MibS on submitted instances directly. They do not need a compiler to verify results — they need the bilevel model file and a solver. BiCut adds convenience, not capability.

### 5. Difficulty Inflation or Deflation

**Massively inflated.** The 121K LoC figure includes ~8K for CLI/logging/config, ~5K for Python bindings, ~5.5K for a benchmark loader, ~5K for testing infrastructure — standard boilerplate that is present in every optimization tool. The "~19K lines of genuinely novel algorithmic logic" is the honest figure, and even that includes the intersection cuts (which are "best-effort" in this approach and may contribute zero novelty). The genuinely novel engineering — the selection cost model calibration, the CQ verification hierarchy, the big-M bound tightening — totals perhaps 5-8K lines of moderately difficult code. This is a solid software project, not a research artifact requiring "genuine engineering breakthroughs."

The four "engineering breakthroughs" listed in §3.3 of the problem statement are: (1) intersection cut separation — not load-bearing in Approach B, (2) value-function oracle — same, (3) sound CQ verification — a three-tier approximation of a known problem, (4) four-backend emission — integration work. Calling integration work a "breakthrough" is dishonest.

### 6. Math Critique

- **The compiler soundness theorem** (T1.3) is described as "Difficulty B (hard but precedented)." In reality, it is a case analysis over four reformulation strategies, each of which has a known correctness proof in the literature (Dempe 2002 for KKT, Fortuny-Amat & McCarl 1981 for strong duality, Outrata et al. 1998 for value function, Bard 1998 for C&CG). The "theorem" is a union of known results organized into a type-theoretic framework. The type-theoretic framing is mildly novel but does not constitute hard mathematics — it is a software verification argument, not an optimization theory result.

- **The compilability decision procedure** (T2.2) claims polynomial-time decidability, but this is trivial: given a finite set of reformulation strategies with known preconditions, checking whether any strategy's preconditions are satisfied is a finite disjunction of polynomial-time-checkable conditions. There is no NP-hardness barrier because the set of reformulations is fixed and small.

- **The compositional error bounds** (T2.3) are claimed only for approximate reformulations that are "explicit non-goals" in the scoped system. This result is not needed and its inclusion inflates the mathematical content of a paper that has essentially no novel math.

### 7. Kill Probability

**30% probability of failing to produce a publishable paper at any venue.**

Justification: Approach B is buildable and testable with no existential risks. The 30% kill probability reflects the risk that reviewers at JOC or CPAIOR find the contribution incremental over BilevelJuMP and reject the paper. The system can be built, evaluated, and written up — the question is whether the delta over existing tools is sufficient for publication. A well-written paper with thorough evaluation (demonstrating selection benefits, certificate value, cross-solver consistency) has a ~70% chance of acceptance at JOC. The risk is in the "so what?" question, not in technical failure.

---

## Approach C: "Hybrid" — Compiler + Cuts as Co-Primary Contributions

### 1. Fatal Flaw

**The two contributions are not synergistic — they are competing for resources, and the integration between them introduces a new failure mode that neither standalone approach faces.** The proposal claims "the compiler enables systematic deployment and fair evaluation of the cuts; the cuts validate the compiler's extensibility." But this synergy is superficial. The compiler could evaluate any cutting-plane strategy (including GMI, split, or Chvátal-Gomory cuts from the MIP solver's default) — the bilevel intersection cuts are not uniquely enabled by the compiler. And the compiler's extensibility is demonstrated by the reformulation passes and backend emission, not by the cuts. The real relationship is: both contributions share implementation time and page count, diluting each.

The integration introduces a specific new failure mode: **the cut engine must operate in the reformulated MILP space while reasoning about bilevel infeasibility in the original space.** The bidirectional mapping between these spaces (described in §Hardest Technical Challenge) depends on the reformulation strategy — and the proposal acknowledges that for KKT reformulation, it involves "extracting primal-dual pairs from the MILP solution and checking complementarity." But KKT reformulation introduces big-M auxiliary variables and complementarity-linearization binaries that are not part of the original bilevel program. Mapping an LP relaxation point back through the KKT reformulation requires inverting the big-M linearization — which is lossy (multiple bilevel (x,y) pairs may correspond to the same MILP point via different big-M encodings). If the backward mapping is lossy, the bilevel feasibility check is unreliable, and the cut may be invalid. **This is not a theoretical concern — it is a correctness bug waiting to happen.** The proposal's "BilevelAnnotation interface" is the right idea but its correctness for each reformulation pass must be separately proved, and the proposal does not acknowledge that the KKT backward mapping is non-injective.

### 2. Hidden Assumptions

- **Assumption: co-primary contributions are acceptable at top venues.** Operations Research and Mathematical Programming have no precedent for a paper that is simultaneously a cutting-plane theory paper and a compiler-engineering paper. OR publishes theoretical contributions with computational validation, not software systems. JOC publishes software with computational contributions, not polyhedral theory. The paper must be submitted to one venue, and either venue's reviewers will find half the paper outside their expertise and uninteresting. The "strongest narrative" is actually the most confusing narrative — it is two papers stapled together.

- **Assumption: 80,000+ experimental configurations are manageable on a laptop.** 2600 instances × 4 reformulations × 4 solvers × 2 (with/without cuts) = 83,200 configurations. At a conservative 5 minutes per configuration (including compilation, cut generation, solving, and verification), this is 416,000 minutes ≈ 289 days of serial computation. Even with 8-core parallelism, this is 36 days of continuous computation. The proposal's "parallelizable across instances" hand-wave ignores that Gurobi's academic license is typically single-instance, SCIP is single-threaded, and HiGHS's parallel mode is experimental. The full evaluation matrix is infeasible on laptop hardware within any reasonable timeline.

- **Assumption: the 2-week prototype gate de-risks the cut contribution adequately.** The prototype tests a bare-bones separation oracle on LP-lower-level instances with no compiler integration, no certificate composition, no multi-backend emission. Even if gap closure passes 10%, the integration challenges (reformulation-aware cut selection, certificate composition, backward mapping correctness) can still fail. The prototype validates the math; it does not validate the engineering.

- **Assumption: "reformulation-aware cut selection" (adapting cut strategy to reformulation type) is tractable and useful.** This is described as "novel algorithm" with "no precedent," which is accurate — but the reason there is no precedent may be that it is unnecessary. If the bilevel intersection cut is derived in the original bilevel space and then mapped to the MILP space, it is valid regardless of the reformulation. Reformulation-awareness would only help if the *effectiveness* of cuts varies significantly across reformulations — a hypothesis with no empirical or theoretical support.

### 3. Competitor Response (Hostile Review)

> **Review of "BiCut: A Bilevel Optimization Compiler with Intersection Cuts"**
>
> This paper attempts to combine two contributions — a bilevel compiler and a new family of cutting planes — into a single unified system. While ambitious, the result is a paper that does neither contribution justice.
>
> **Major concern 1:** The cutting-plane theory (Sections 4-5) is promising but incomplete. The facet-defining conditions are proved only for the restricted class of non-degenerate LP lower levels. The value-function lifting section presents a framework but no concrete strength results (e.g., how much stronger are lifted cuts vs. unlifted?). A stand-alone cutting-plane paper at Mathematical Programming would demand deeper theoretical development. Here, the theory is compressed to make room for compiler architecture discussion that Mathematical Programming readers do not need.
>
> **Major concern 2:** The compiler architecture (Sections 2-3) is solid engineering but not novel relative to BilevelJuMP.jl. The reformulation selection engine selects the optimal strategy on only 7 additional instances beyond the default. The correctness certificates are sound but verify textbook conditions. A standalone compiler paper at JOC would need stronger empirical evidence of the compiler's practical value. Here, the compiler evaluation is compressed to make room for cutting-plane theory that JOC readers may not appreciate.
>
> **Major concern 3:** The paper is too long. At its current level of detail, it is 45+ pages — exceeding the typical length for OR (30 pages) or JOC (25 pages). Substantial material would need to move to appendices, weakening both contributions.
>
> **Minor concern:** The claim of "synergy" between the compiler and cuts is asserted but not demonstrated. The cuts could be implemented as a standalone SCIP plugin without any compiler infrastructure. The compiler provides convenience for deploying cuts but is not necessary for their evaluation. The synergy is editorial, not technical.
>
> **Recommendation: Major revision.** Split into two papers (one on cuts for Math Programming, one on the compiler for JOC), or significantly deepen one contribution at the expense of the other.

### 4. Value Inflation

The "research platform" argument — that BiCut becomes a platform for future bilevel cutting-plane research — is speculative and unlikely. Research platforms succeed when they have large user communities that contribute extensions (SCIP, JuMP, PyTorch). BiCut's user base of 500-1000 people will not generate a contributor community. The realistic outcome is that BiCut is used by the 3-5 PhD students in the author's group and cited by the 50-100 researchers who work directly on bilevel cutting planes. The "platform for bilevel cutting-plane research" framing implies a contributor ecosystem that will never materialize.

The "strongest narrative of the three approaches" self-assessment is also inflated. The narrative is only strong if both contributions land at sufficient depth. If the facet-defining proof fails and only computational cut results are available, the narrative becomes "we built a compiler and also tried some cuts that help modestly" — which is a weaker narrative than Approach B's clean "we built the first bilevel compiler with certificates."

### 5. Difficulty Inflation or Deflation

**Inflated integration difficulty, deflated individual contribution difficulty.** The proposal scores difficulty at 8/10 by claiming the hybrid "inherits all difficulty from both A and B, plus the integration challenges." But the hybrid does not achieve the full depth of either A or B. The cuts are not developed to Approach A's depth (facet-defining conditions have a fallback to restricted class). The compiler is not developed to Approach B's thoroughness (the QP extension and fourth solver backend are inherited but the attention is split). The integration challenges — the BilevelAnnotation interface, reformulation-aware cut selection, certificate composition — are real but are software design problems, not research problems. A well-designed interface with good test coverage handles them. The difficulty of Approach C is approximately max(A, B) + 1 for integration, not A + B.

### 6. Math Critique

- **Certificate composition** (T1.3, extended) is framed as "novel formalism" comparable to proof-carrying code (Necula 1997). This comparison is absurd. Proof-carrying code involves generating machine-checkable proofs for arbitrary program transformations in a Turing-complete language. BiCut's certificate composition checks that three specific types of preconditions (structural, reformulation, cut validity) are conjunctively satisfied — a conjunction of known conditions, not a novel formal system. Invoking Necula to inflate the novelty of a precondition checker is intellectually dishonest.

- **The "reformulation-aware cut selection" has no formal statement.** It is listed as a "novel algorithm" but no theorem characterizes what it means for a cut to be "effective for a given reformulation." Without a formal definition of effectiveness and a proof that the selection algorithm optimizes it, this is a heuristic, not an algorithm with guarantees. Calling it a "novel algorithm" is premature.

- **The end-to-end correctness argument** requires proving that the bilevel intersection cut, derived in the original space and mapped through the reformulation, is a valid inequality for the reformulated MILP. This requires showing that the forward mapping is an embedding (preserves all bilevel-feasible solutions) — which is true by the reformulation's correctness guarantee — but also that the cut does not exclude any MILP-feasible point that maps back to a bilevel-feasible solution. For KKT reformulation, the MILP has additional variables (duals, binaries) that have no bilevel counterpart. The cut, expressed in MILP variables, must be "projection-safe" — valid when projected onto the original bilevel variables. This projection safety is non-trivial for cuts that reference dual variables and is not addressed in the proposal.

### 7. Kill Probability

**40% probability of failing to produce a publishable paper at any venue.**

Justification: The hybrid has more failure modes than either standalone approach. The cuts can fail (same risk as A, ~40% probability). The integration can introduce correctness bugs (the backward mapping issue, ~15% probability of a showstopper discovered late). The paper can be too long and unfocused for any single venue (~25% probability of desk rejection or "split into two papers" reviews). However, the compiler fallback provides a safety net: if the cuts fail entirely, a compiler-only paper (Approach B) is still producible, dropping the kill probability from ~55% (cuts fail AND no fallback) to ~40% (cuts fail AND the resulting compiler-only paper is also rejected as incremental). The 40% accounts for the possibility that the pivot to compiler-only occurs too late to produce a polished paper, or that the reviewer pool has already seen the BilevelJuMP paper and finds BiCut's delta insufficient.

---

## Cross-Approach Attack

### Most likely to fail entirely: Approach A (Cuts-First)

Approach A has zero fallback and the narrowest viability corridor. Every component — the polyhedrality theorem, the facet-defining proof, the separation oracle performance, the value-function lifting, the single-solver evaluation — must succeed simultaneously. The probability that all five land at publishable quality is the product of their individual success probabilities, which is very low. A single failure (the facet proof is too hard, the gap closure is <10%, the oracle is too slow, the lifting is intractable) kills the entire paper with no recovery path.

### Most likely to produce a mediocre paper: Approach B (Compiler-First)

Approach B will almost certainly produce *a* paper — but the "just engineering" and "incremental over BilevelJuMP" risks are severe. The most likely outcome is a competent JOC paper with thorough evaluation that receives reviews like "useful contribution to the bilevel optimization community" and "recommended for publication with minor revisions" — but also "this is an engineering contribution, not a research contribution" and "I would have liked to see deeper mathematical analysis of the reformulation selection." It gets published, receives 30-60 citations over 10 years (mostly from the bilevel optimization community), and is forgotten. This is fine for a PhD student's second or third paper; it is not a career-defining contribution.

### Most honest self-assessment: Approach A (Cuts-First)

Ironically, the highest-risk approach has the most honest self-assessment. It scores its own feasibility at 4/10 and explicitly states "if the math doesn't work out, there is essentially no paper." It does not pretend to be something it isn't. The other approaches inflate their novelty (B's "new software category" claim) or their synergy (C's "co-primary contributions" frame). A gives itself a value of 4 and a feasibility of 4 — that's honest.

### If I had to bet: Approach B (Compiler-First)

**Rationale:** I am betting on a *publishable paper at any venue*, not on the best possible paper. Approach B has a 70% chance of producing a publishable JOC paper. The other approaches have higher ceilings but much lower floors. In expected value terms:
- **A:** 35% × (Math Programming paper worth 10 points) = 3.5 expected points
- **B:** 70% × (JOC paper worth 5 points) = 3.5 expected points  
- **C:** 60% × (OR/JOC paper worth 7 points) = 4.2 expected points

By expected value, C wins — but the variance is highest, and the 40% failure probability includes scenarios where significant time is wasted on cuts that don't work and a compiler that is half-baked because attention was split. If I had to bet my own career on one approach, I would bet on B — the boring, safe, buildable option — and invest the saved risk budget in writing the paper extremely well and running the most thorough evaluation in the bilevel optimization literature. A boring paper with the best-ever computational evaluation of bilevel reformulation strategies is more publishable than an ambitious paper with incomplete theory and a mediocre system.

**However:** If the 2-week prototype gate shows gap closure ≥ 15%, I would immediately switch my bet to Approach A. A 15% gap closure result with even partial facet-defining conditions is a Mathematical Programming paper that the bilevel community will remember. The prototype gate is the single most important decision point — run it first, then choose.

---

## Summary of Kill Probabilities

| Approach | Kill Probability | Primary Kill Mechanism |
|----------|-----------------|----------------------|
| A: Cuts-First | **65%** | Cuts don't work (gap closure < 10%) or facet proof fails, with zero fallback |
| B: Compiler-First | **30%** | "Incremental over BilevelJuMP" rejection at review |
| C: Hybrid | **40%** | Cuts fail + integration bugs + "split into two papers" review + too late to pivot cleanly |

*Written as adversarial skeptic. Every flaw identified is intended to be specific, falsifiable, and actionable. The goal is not to kill the project but to ensure that whichever approach is chosen has survived the hardest possible scrutiny.*
