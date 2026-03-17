# BiCut: Three Competing Approaches

**Project:** BiCut — Bilevel Optimization Compiler with Intersection Cuts  
**Slug:** `bilevel-compiler-intersection-cuts`  
**Date:** 2026-03-08

---

## Approach A: "Cuts-First" — Deep Cutting Plane Theory

### Summary

Approach A treats BiCut as a vehicle for a first-class mathematical contribution in polyhedral theory: bilevel intersection cuts and value-function lifting. The compiler infrastructure is deliberately minimal — a thin parser, a single reformulation strategy (value-function), and bare-bones MPS emission to one solver (SCIP, chosen for its open constraint handler API). All engineering effort concentrates on the intersection cut engine (S5), value-function oracle (S6), and the supporting polyhedral theory. The key deliverable is a new family of cutting planes with provable facet-defining conditions, a separation oracle with demonstrated effectiveness on BOBILib, and a finite convergence theorem. The compiler exists only to make the cuts testable. If the cuts work, this is a *Mathematical Programming* or *IPCO* paper with a software supplement; if they don't, there is no fallback.

### Extreme Value Delivered

**Who needs it:** The ~100–200 researchers working on mixed-integer bilevel linear programs (MIBLP) who are stuck with root LP relaxations that have massive integrality + bilevel gaps. MibS — the only dedicated MIBLP solver — uses branch-and-cut with no bilevel-specific cutting planes. The integer programming community (IPCO, MIP workshops) has been searching for new families of valid inequalities beyond the Chvátal–Gomory / split / cross / multi-row hierarchy; bilevel-infeasible sets are an entirely unexplored source of cuts. Additionally, the parametric optimization community (Pistikopoulos, Oberdieck) needs efficient value-function evaluation to make bilevel reformulations practical — the oracle technology developed here has standalone value for any application requiring repeated lower-level solves under varying parameters.

**Why desperate:** Current bilevel solvers treat the integrality gap and the bilevel gap independently. Integrality is handled by branching; bilevel feasibility is handled by feasibility checks and lazy constraints that enforce optimality of the follower. No existing tool generates cutting planes that simultaneously exploit both structures. This means root relaxations are weak, branch-and-bound trees are deep, and instances with >200 variables routinely time out. A 15–25% root gap closure — if achievable — would be the first improvement in bilevel LP relaxation quality in over a decade.

### Genuine Difficulty as Software Artifact

**Hard subproblems:**

1. **Bilevel-infeasible set characterization (novel algorithm).** The core challenge: given an LP relaxation point (x̂, ŷ) that violates bilevel feasibility (ŷ is suboptimal for the follower at x̂), characterize the maximal convex bilevel-infeasible set containing (x̂, ŷ). For LP lower levels, this requires enumerating vertices of the lower-level optimal face as a function of x, building a polyhedral description of the region where ŷ is dominated, and computing ray intersections. The vertex enumeration is exponential in the worst case but structured (dual degeneracy determines the combinatorial complexity). No off-the-shelf algorithm exists.

2. **Separation oracle with high cache hit rates (novel engineering).** Each cut round requires multiple calls to the separation oracle, each involving an auxiliary LP solve. For the cuts to be practical (overhead <50ms per call), the oracle must exploit parametric sensitivity: consecutive LP relaxation points differ by small perturbations after branching, so the optimal basis of the auxiliary LP is likely unchanged. Implementing warm-started parametric LP with basis tracking, perturbation detection, and cache invalidation at >90% hit rates is a systems challenge with no precedent in cutting-plane implementations.

3. **Value-function oracle for MILP lower levels (novel algorithm).** Exact value-function evaluation for integer lower levels requires solving a parametric MILP — NP-hard for each query. The sampling-based approximation (evaluate V(x) at a grid of points, construct a piecewise-linear overestimator with provable error bounds) requires careful treatment of discontinuities in the integer value function. The Gomory–Johnson lifting step — extracting valid inequalities from the overestimator's structure — requires extending subadditivity theory to piecewise-linear functions defined on polyhedral domains rather than the integers.

4. **Facet-defining proof (novel mathematics).** The hardest purely mathematical task: prove that bilevel intersection cuts define facets of the bilevel-feasible set's convex hull under explicit, verifiable conditions. This requires (i) establishing the dimension of the bilevel-feasible polyhedron (nontrivial because bilevel feasibility is not a simple linear constraint), (ii) constructing dim(P) affinely independent bilevel-feasible points on the cut hyperplane, and (iii) proving the conditions are tight (the cut is not facet-defining when the conditions fail). The closest precedent is Balas's facet proof for disjunctive cuts (1979), but the bilevel structure introduces optimality-based disjunctions that are significantly more complex.

**What's routine:** Parser (~3K LoC, standard recursive descent), value-function reformulation pass (~4K LoC, known theory from Outrata et al. 1998), MPS emission (~2K LoC, standard format), SCIP callback integration (~3K LoC, well-documented API), BOBILib loader (~2K LoC, standard file parsing).

**Architectural challenge:** The system is deliberately narrow — one solver, one reformulation, no selection engine. This makes the architecture simple but fragile: if SCIP's constraint handler performance is poor, there is no fallback to Gurobi's lazy constraints without significant rework.

### New Math Required

| Result | Description | Difficulty | Load-Bearing? |
|--------|------------|------------|---------------|
| **Bilevel-infeasible set polyhedrality theorem** | For MIBLP with LP lower level, B̄ is a finite union of polyhedra whose facets are characterized by the lower-level dual vertex set | **C** (novel) | Yes — without this, the intersection cut framework has no foundation; the entire separation procedure depends on being able to compute ray-boundary intersections against a polyhedral set |
| **Facet-defining conditions** | Complete characterization of when bilevel intersection cuts define facets of conv(bilevel-feasible set) | **C** (novel) | Yes — this is the difference between "we applied Balas to bilevel" (incremental) and "we proved new polyhedral structure" (a contribution to cutting-plane theory). Without it, the paper is computational, not theoretical |
| **Separation complexity** | Prove separation is polynomial in the number of lower-level constraints for fixed follower dimension | **B** (hard, precedented) | Yes — if separation is exponential in general, the cuts are impractical; the complexity result establishes the computational regime where cuts are viable |
| **Finite convergence** | Under non-degeneracy, the bilevel cut loop terminates in finitely many rounds | **B** (hard, precedented) | Partially — convergence is expected (standard for cutting-plane methods) but the proof requires handling the non-standard bilevel feasibility structure |
| **Value-function lifting via Gomory–Johnson** | Extend subadditivity theory to value-function epigraphs; construct maximal valid lifting functions from dual vertex enumeration | **C** (novel) | Yes — this is the second mathematical contribution; lifting produces stronger cuts than raw intersection cuts by exploiting value-function structure |
| **Sampling-based MILP value-function approximation** | Piecewise-linear overestimator of integer value function with provable error bounds O(h²) where h is grid spacing | **B** (hard, precedented) | Partially — only needed for MILP lower levels; LP lower levels use exact parametric LP |

### Best-Paper Potential

**Target venue:** Mathematical Programming Series A or IPCO.

**Why best-paper:** This would be the first extension of Balas's intersection cut framework to optimality-defined infeasible sets. The original Balas 1971 paper introduced intersection cuts for sets defined by integrality constraints; subsequent extensions (Cornuéjols, Bienstock, Conforti–Cornuéjols–Zambelli) addressed split disjunctions, multi-row cuts, and cross cuts — all defined by algebraic constraints on variables. Bilevel infeasibility is defined by *optimality of a subproblem*, a qualitatively different structure. The facet-defining conditions, if provable, would constitute a genuine advance in polyhedral combinatorics. The value-function lifting result bridges two communities (Gomory–Johnson theory and bilevel optimization) that have never interacted in the literature. Mathematical Programming's readership would recognize this as extending a 50-year-old framework to a fundamentally new domain.

**Risk:** The viability corridor is narrow. If the bilevel-infeasible set has a trivial polyhedral description for LP lower levels (reducing to a known disjunction), the math is incremental. If the polyhedral description is too complex for practical separation, the cuts are impractical. The depth check panel identified this as the critical risk: "no empirical evidence validates that this corridor is navigable." A failed prototype gate kills the entire approach.

### Hardest Technical Challenge

**The facet-defining proof for bilevel intersection cuts.**

This is the hardest challenge because it requires simultaneously understanding (i) the combinatorial structure of the bilevel-feasible set's convex hull (whose facets are determined by the interaction between leader constraints, follower optimality, and integrality), (ii) the geometry of Balas-type cuts in this setting (ray intersections with a non-standard infeasible set), and (iii) the algebraic conditions under which the resulting inequality is tight on enough points to be facet-defining.

**How to address it:**

1. **Start with the simplest nontrivial case.** Consider a bilevel program with 1 leader variable, 2 follower variables, LP lower level. In this setting, V(x) is piecewise linear with at most O(m) breakpoints (m = number of lower-level constraints), B̄ has a closed-form polyhedral description, and the facet-defining conditions can be verified by direct computation. Prove the theorem for this case first.

2. **Lift to general dimensions via induction on follower dimension.** The key observation is that adding a follower variable refines the partition of leader-variable space induced by V(x). The facet-defining conditions should extend inductively if the new variable's effect on the lower-level optimal face is "generic" (non-degenerate).

3. **Use computational algebra as a proof tool.** For small instances (≤5 follower variables), enumerate all vertices of conv(bilevel-feasible set) using Polymake or CDD, compute all facets, and verify the conjectured conditions computationally before attempting the general proof.

4. **Fallback: prove facet-defining conditions for a restricted class.** If the general proof is too hard, prove it for LP lower levels with unique optimal follower response (non-degenerate lower level). This covers the majority of BOBILib instances and is still a significant theoretical result.

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **4/10** | The user base is tiny (100–200 MIBLP researchers). Industrial adoption is negligible. The cuts have no application outside bilevel optimization. Even within the community, value is contingent on the cuts actually working — and there is zero empirical evidence. The depth check panel scored value at 4; this approach narrows the user base further by omitting the compiler's convenience features. |
| **Difficulty** | **9/10** | The facet-defining proof is genuinely hard novel mathematics (Difficulty C). The separation oracle engineering is unprecedented. The value-function lifting extends Gomory–Johnson in a nontrivial direction. The compiler infrastructure is minimal, but the mathematical and algorithmic core is extremely challenging. The difficulty is concentrated in ~15K lines of novel code plus the mathematical proofs. |
| **Potential** | **8/10** | If the facet-defining conditions are provable and the cuts achieve ≥15% gap closure, this is a strong Mathematical Programming paper and a plausible IPCO best paper. The result would be cited for decades as the first intersection cut extension to bilevel. However, the potential is binary: if the math doesn't work out, there is essentially no paper. |
| **Feasibility** | **4/10** | The narrow viability corridor is the dominant risk. The facet-defining proof may be intractable for general instances. The value-function oracle for MILP lower levels is speculative. There is no fallback — the minimal compiler has no independent value without the cuts. The depth check panel's "kill condition" (gap closure <5% AND cache hit <70%) would terminate this approach entirely. The 2-week prototype gate is essential, but even passing it doesn't guarantee the full theoretical results. |

---

## Approach B: "Compiler-First" — Full Compiler Architecture

### Summary

Approach B treats BiCut as a compiler-engineering contribution: the first disciplined bilevel programming framework with a typed intermediate representation, automatic reformulation selection, machine-checkable correctness certificates, and solver-agnostic emission to four backends. Intersection cuts are included as one optional optimization pass — useful if they work, but not load-bearing for the paper's contribution. The primary investment is in the IR design (S1), structural analysis (S2), reformulation selection engine (S3), all four reformulation passes (S4, S7), the four-backend emission layer (S8, E5), correctness certificates (S9), and the QP lower-level extension (E1). The intersection cut engine (S5) and value-function oracle (S6) are implemented as best-effort modules: functional but without facet-defining proofs or gap closure guarantees. The key deliverable is a system that makes bilevel optimization as accessible as CVXPY made convex optimization, with formal verification that no existing tool provides.

### Extreme Value Delivered

**Who needs it:** Three distinct user groups who are currently underserved:

1. **Applied bilevel practitioners (200–500 people)** in energy markets, supply chain, and infrastructure planning who prototype bilevel models but are blocked by the weeks-long cycle of manual reformulation derivation, big-M debugging, and solver-specific code. Today, a researcher who wants to compare KKT reformulation vs. strong duality on the same bilevel model must hand-derive both, implement both in solver-specific code, and manually verify correctness. BiCut reduces this to changing a configuration flag.

2. **OR educators (50–100 instructors)** who teach bilevel optimization conceptually but cannot assign computational exercises because the barrier to entry is prohibitive. A student who can write `bilevel.compile(model, strategy="auto")` can experiment with bilevel optimization in a homework assignment — currently impossible without weeks of background in complementarity theory.

3. **Computational optimization reviewers and editors** who need tools to verify bilevel results in submitted papers. BiCut's certificates provide an independent verification pathway: recompile the bilevel model, check the certificate, reproduce on a different solver. This is a qualitative advance in bilevel research reproducibility.

**Why desperate:** The fragmentation is real and costly. BilevelJuMP.jl supports KKT with per-constraint mode selection but has no automatic strategy selection, no certificates, and no support for integer lower levels beyond enumeration. PAO offers FA, PCCG, and REG solvers but is Python-only with no solver-agnostic emission. GAMS EMP does specification→reformulation→dispatch but is commercial, closed-source, and limited to GAMS-supported solvers. MibS is a direct solver, not a reformulation tool. No existing tool answers the question "which reformulation should I use for this problem?" or "is this reformulation correct?" BiCut is the first to provide both answers, automatically, with formal guarantees.

### Genuine Difficulty as Software Artifact

**Hard subproblems:**

1. **Reformulation selection engine with sound cost model (novel engineering).** The selection engine must map structural signatures (convexity, CQ status, integrality, coupling type, dimension) to ranked reformulation strategies. The soundness requirement is non-trivial: the engine must never select a reformulation whose preconditions are violated. The cost model — predicting which valid reformulation will produce the fastest-to-solve MILP — requires calibrating against empirical performance data from BOBILib, handling multi-objective tradeoffs (formulation tightness vs. formulation size vs. big-M magnitude), and composing strategies for problems that benefit from chained passes (e.g., value-function + intersection cuts). No existing tool attempts automatic reformulation selection for bilevel programs.

2. **Correctness certificates with sound CQ verification (novel algorithm + formal methods).** Each certificate is a conjunction of verified structural preconditions: LICQ holds at the lower level (verified via LP-based rank test), Slater's condition holds (verified via strict feasibility LP), the lower level is bounded (verified via dual feasibility), integrality structure matches the reformulation's requirements. The CQ verification itself is co-NP-hard; BiCut implements a conservative three-tier approximation (syntactic → LP-based → sampling-based) that must be *sound* (never certify a false CQ) while rejecting <20% of valid instances. The certificate format must be machine-checkable: a third-party tool (or reviewer) can verify the certificate without re-running BiCut's analysis.

3. **Four-backend solver emission with semantic preservation (hard systems engineering).** Gurobi, SCIP, HiGHS, and CPLEX expose fundamentally different APIs for the constructs BiCut generates. Indicator constraints: Gurobi-native, SCIP via specialized constraint handlers, CPLEX-native, HiGHS requires big-M linearization. Lazy constraints for cut callbacks: Gurobi via `cbLazy()`, SCIP via `CONSHDLR` plugins (C callbacks with memory management and thread-safety requirements), CPLEX via Benders callbacks, HiGHS via iterative MPS fallback (no true callback API). SOS1 sets: varying performance and encoding across solvers. The emission layer must select solver-optimal encodings while preserving the invariants guaranteed by correctness certificates — and must be *tested* by verifying that the same bilevel model produces the same optimal value (within tolerance) across all four backends.

4. **Automatic big-M computation with numerical soundness (hard engineering).** Big-M values for complementarity linearization must be tight (large M degrades LP relaxation) and valid (too-small M cuts off feasible solutions). BiCut computes M via auxiliary bound-tightening LPs, which must handle: unbounded dual rays (indicating unbounded primal, requiring fallback to value-function reformulation), near-zero reduced costs (numerical instability in M computation), and solver-dependent feasibility tolerances (a constraint satisfied at 10⁻⁸ in Gurobi may be violated at 10⁻⁶ in HiGHS). A single tolerance mistake corrupts all downstream results.

5. **QP lower-level reformulation (moderately novel).** Extending KKT reformulation to quadratic objectives introduces bilinear complementarity terms (λᵢ · gᵢ(x,y) = 0 where gᵢ is linear but λᵢ multiplies it, creating bilinear terms in the KKT stationarity conditions). Linearization via McCormick envelopes introduces auxiliary variables and constraints that interact with big-M computation. For convex QP lower levels, an alternative SOCP reformulation avoids complementarity entirely but requires SOCP-capable solvers (Gurobi, CPLEX, Mosek — not HiGHS or SCIP for mixed-integer SOCP).

**What's routine:** Parser (standard), expression DAG (standard), MPS/LP file writing (standard), BOBILib instance loader (standard), CLI and configuration (standard), Python bindings via pybind11 (standard), logging and error reporting (standard).

**Architectural challenges:** The compiler pipeline must be modular enough that adding a new reformulation pass or a new solver backend does not require modifying existing passes. This requires a clean pass manager interface, typed IR with stable semantics, and a backend abstraction layer. The design is precedented (LLVM's pass infrastructure, CVXPY's reductions) but requires careful upfront design to avoid coupling between passes.

### New Math Required

| Result | Description | Difficulty | Load-Bearing? |
|--------|------------|------------|---------------|
| **Compiler soundness theorem** | If Typecheck(P) succeeds and structural analysis certifies Φ, then for every selected reformulation R, opt(emit(R)) = opt(P) and solutions map back correctly | **B** (hard, precedented) | Yes — this is the central theoretical result; it guarantees that the compiler does not introduce errors. Without it, the certificates are heuristic checks rather than formal guarantees |
| **Compilability decision procedure** | Given bilevel program P and solver capability profile S, decide in polynomial time whether a valid compilation exists | **B** (hard, precedented) | Yes — this determines the compiler's coverage; users need to know whether their problem can be compiled, and the boundary of compilability is a useful theoretical characterization |
| **Structure-dependent selection soundness** | Prove that the selection function ρ(σ) always selects reformulations whose preconditions are satisfied by σ | **A** (known) | Yes — soundness of selection is essential, but the proof is a straightforward case analysis over the (finite) set of structural signatures and reformulation preconditions |
| **Compositional error bounds** | For approximate passes (regularization, penalty, cut truncation), prove composed error ≤ Π(1 + κᵢεᵢ) − 1 | **B** (hard, precedented) | Partially — only needed if approximate reformulations are included; exact reformulations (KKT, strong duality, value function) have zero error |
| **Bilevel intersection cuts (basic)** | Separation procedure for bilevel-infeasible sets without facet-defining conditions | **B** (hard, precedented) | Partially — the cuts are a nice-to-have optimization pass, not the paper's main contribution. Basic separation (without facet theory) is achievable by adapting Balas's procedure |
| **QP complementarity linearization** | McCormick envelopes for bilinear KKT terms with tightness guarantees | **A** (known) | Yes — standard McCormick theory applied to KKT-derived bilinear terms; the novelty is in the automated tightening within the compiler pipeline |

### Best-Paper Potential

**Target venue:** INFORMS Journal on Computing (JOC) or CPAIOR.

**Why best-paper:** IJOC explicitly values software contributions with rigorous evaluation. BiCut would be the first bilevel optimization compiler — a new software category analogous to CVXPY for convex optimization. The paper would demonstrate: (i) the typed IR captures all MIBLP structures in BOBILib, (ii) automatic reformulation selection matches or beats expert-chosen strategies on ≥80% of instances, (iii) correctness certificates catch real bugs (KKT applied to integer lower levels where CQs fail) on ≥10% of instances, (iv) solver-agnostic emission achieves ≥95% cross-solver success rate, and (v) the QP extension broadens applicability beyond MIBLPs. The CVXPY JOC paper (Diamond & Boyd, 2016) won a software award by demonstrating that a well-designed compiler lowers barriers and enables new applications; BiCut makes the same argument for a harder problem class.

**Risk:** The "just engineering" criticism. Mathematical Programming and IPCO reviewers would find the compiler soundness theorem (Difficulty B) insufficient for a theoretical contribution. Even at IJOC, the paper needs to demonstrate that the compiler enables *new computational results* — not just convenience. If the benchmark evaluation shows that BiCut's reformulation selection and certificates provide marginal benefit over BilevelJuMP's KKT mode, the paper reduces to "we built a more general version of BilevelJuMP." The intersection cuts, even without facet theory, provide insurance: any measurable gap closure is a computational contribution beyond pure engineering.

### Hardest Technical Challenge

**Four-backend solver emission with semantic preservation across solver-specific encodings.**

This is the hardest challenge because it requires deep knowledge of four different solver APIs (Gurobi Python, SCIP C/Python, HiGHS C++, CPLEX Python), each with different capabilities, numerical tolerances, and callback mechanisms. The fundamental difficulty is that the same mathematical construct (e.g., a complementarity constraint x·s = 0 with x, s ≥ 0) has radically different optimal encodings across solvers:

- **Gurobi:** Indicator constraint (`x = 0 → s ≥ 0` and `s = 0 → x ≥ 0`) via native API, or SOS1 set {x, s}, or big-M linearization. Indicator is fastest but interacts with Gurobi's presolve differently than big-M.
- **SCIP:** Indicator constraint via `SCIPcreateConsIndicator()` (C API), or SOS1 via `SCIPcreateConsSOS1()`, or big-M. SCIP's constraint handler architecture means indicator constraints participate in propagation and separation differently than in Gurobi.
- **HiGHS:** No native indicator constraints; must use big-M linearization or SOS1 (limited support). HiGHS's LP relaxation may be weaker for big-M encodings, requiring tighter M values.
- **CPLEX:** Indicator constraints via `cplex.indicator_constraints.add()`, SOS1 via `cplex.SOS.add()`, or big-M. CPLEX's Benders decomposition callback is the preferred mechanism for lazy constraints, different from Gurobi's `cbLazy()`.

The emission layer must select the best encoding for each solver while guaranteeing that all four produce the same optimal value (within solver-dependent tolerances). Testing this across 2600+ BOBILib instances × 4 solvers × multiple encoding strategies is a combinatorial testing challenge.

**How to address it:**

1. **Define a backend capability profile.** Each solver declares its supported constructs (indicator constraints, SOS1, lazy constraints, warm-starting) and numerical tolerances. The emission layer queries the profile to select encodings.

2. **Implement encoding as a strategy pattern.** Each mathematical construct (complementarity, big-M, indicator, SOS1) has an encoder per backend. New encoders can be added without modifying the emission pipeline.

3. **Cross-solver verification harness.** For every BOBILib instance, compile to all four backends, solve, and verify that optimal values agree within max(solver tolerances). Flag discrepancies for manual investigation.

4. **Start with MPS-only emission (all solvers support MPS), then add solver-specific API emission incrementally.** MPS emission is a common denominator that works everywhere; solver-specific emission (indicator constraints, callbacks) is layered on top for performance.

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **6/10** | Broader user base than Approach A (practitioners + educators + reviewers, not just cutting-plane researchers). The compiler convenience, certificates, and solver-agnostic emission provide value even without novel cuts. However, the user base is still small (500–1000 people at ceiling) and industrial adoption is unlikely. The depth check panel's value score of 4 was for the full system; the compiler-only framing is slightly more valuable to a wider audience but still niche. |
| **Difficulty** | **6/10** | The compiler engineering is genuinely challenging (four-backend emission, sound CQ verification, automatic big-M, QP extension) but largely precedented. No single subproblem requires novel mathematical ideas. The ~19K lines of novel code are real engineering, not research. The difficulty is distributed across many medium-hard subproblems rather than concentrated in a few hard ones. |
| **Potential** | **5/10** | A solid IJOC paper, possibly with a software award. Not a Mathematical Programming paper — the math is Difficulty A–B throughout. The "just engineering" risk is real: reviewers may argue that BilevelJuMP already provides KKT reformulation and BiCut is an incremental improvement with more backends and certificates. The QP extension and reformulation selection engine are differentiators but not groundbreaking. Best-paper at IJOC is possible if the evaluation is exceptional; best-paper at a theory venue is not. |
| **Feasibility** | **8/10** | High feasibility. All subproblems are precedented. The four-backend emission is painful but tractable (each backend is independent, so work parallelizes). The CQ verification and certificate system build on known theory. The BOBILib evaluation infrastructure is mature. The main risk is schedule — 121K LoC is a large system — but there are no "narrow viability corridors" or existential risks. If the intersection cuts don't work, the compiler stands on its own. |

---

## Approach C: "Hybrid" — Compiler + Cuts as Co-Primary Contributions

### Summary

Approach C pursues both the compiler architecture and the cutting-plane theory as co-primary contributions, with neither subordinate to the other. The compiler enables systematic deployment and fair evaluation of the cuts; the cuts validate the compiler's extensibility and provide the mathematical depth that pure compiler papers lack. Investment is balanced: S1–S4 and S8–S9 for the compiler backbone, S5–S6 for the cutting-plane engine, E1 and E5 for breadth. The key insight is that the two contributions are synergistic — the compiler's reformulation selection can route problems to the reformulation that produces the best LP relaxation for cut generation, and the cuts provide the clearest computational evidence that the compiler's abstraction layer works. The deliverable is a full-pipeline system evaluated end-to-end on BOBILib with the 2-week prototype gate as an early risk mitigator. This is the approach described in the crystallized problem statement.

### Extreme Value Delivered

**Who needs it:** All three user groups from Approach B (practitioners, educators, reviewers) *plus* the cutting-plane research community. The hybrid value proposition is:

1. **For MIBLP researchers:** A complete pipeline from bilevel specification to tightened MILP with formal guarantees. No manual reformulation, no ad-hoc cut implementation, no solver-specific code. The reformulation selection engine routes each problem to its best strategy, and the intersection cuts tighten the resulting formulation — all verified by certificates.

2. **For the cutting-plane community:** A reusable infrastructure for experimenting with bilevel cuts. The compiler's extensible pass architecture means new cut families (e.g., bilevel split cuts, bilevel Chvátal–Gomory cuts) can be added as passes without re-implementing the IR, reformulation, or emission layers. BiCut becomes a *platform* for bilevel cutting-plane research, not just a one-off implementation.

3. **For reproducibility:** The strongest version of the reproducibility argument. A compiled MILP with attached certificate and embedded cuts can be independently verified on any solver. The full pipeline — parse, analyze, reformulate, cut, emit — is deterministic and auditable.

**Why desperate:** The fundamental problem is that bilevel cutting planes cannot be fairly evaluated without a compiler. Today, a researcher who wants to test bilevel intersection cuts must: (a) hand-derive a reformulation for each test instance, (b) implement the cut separation oracle in solver-specific callback code, (c) debug big-M values and complementarity encoding by hand, and (d) repeat for each solver. This makes systematic evaluation of bilevel cuts practically impossible — which is why no bilevel-specific cutting planes have been published despite decades of bilevel optimization research. BiCut's compiler removes barriers (a)–(d), making bilevel cutting-plane research feasible for the first time.

### Genuine Difficulty as Software Artifact

**Hard subproblems (inheriting from both A and B):**

1. **Intersection cut engine with compiler integration (novel algorithm + systems).** All of Approach A's algorithmic challenges for the separation oracle and value-function oracle, *plus* the integration challenge: the cut engine must interact with the compiler's IR (reading variable types, constraint structure, reformulation metadata), the emission layer (injecting cuts as lazy constraints via solver-specific callbacks), and the certificate system (the cut's validity certificate must compose with the reformulation's certificate to produce an end-to-end guarantee). This integration is harder than building the cut engine in isolation because the cut engine cannot make assumptions about the reformulation strategy — it must work with KKT, strong duality, value-function, and C&CG reformulations.

2. **Reformulation-aware cut selection (novel algorithm).** Different reformulations produce MILPs with different LP relaxation geometries. The cuts most effective for a KKT reformulation (which has complementarity-induced big-M constraints) may be different from those effective for a strong duality reformulation (which has a dual feasibility polyhedron). The cut engine should adapt its separation strategy to the reformulation, exploiting reformulation-specific structure. This reformulation-aware cut selection has no precedent.

3. **End-to-end certificate composition (novel formalism).** The certificate system must compose three independent guarantees: (i) structural analysis correctly identified the problem's properties, (ii) the selected reformulation preserves bilevel optimality under those properties, and (iii) the added cuts are valid inequalities that do not exclude bilevel-feasible solutions. The composition must handle the interaction between reformulation-introduced variables (dual variables, big-M binaries) and cut-introduced constraints (which reference these variables). No existing certificate system handles this three-way composition.

4. **All of Approach B's engineering challenges:** Four-backend emission, automatic big-M, sound CQ verification, QP extension.

5. **Scalable benchmark evaluation (systems engineering).** The full evaluation requires: 2600+ BOBILib instances × 4+ reformulation strategies × 4 solver backends × with/without cuts = 80,000+ experimental configurations. Managing this evaluation (parallelization, failure recovery, result aggregation, statistical analysis) at laptop scale requires careful engineering of the benchmark harness.

**What's routine:** Same as Approach B — parser, expression DAG, MPS writing, instance loading, CLI, Python bindings, logging.

**Architectural challenges:** The hybrid approach requires the most careful architecture because the cut engine must be both (a) modular (pluggable as an optional pass) and (b) deeply integrated (aware of reformulation structure, solver callback APIs, and certificate composition). These requirements are in tension. The resolution is a well-defined interface between the reformulation pass and the cut pass: the reformulation pass annotates its output IR with metadata (which variables are dual, which constraints encode complementarity, what the original bilevel structure is) that the cut pass consumes without needing to understand the reformulation's internal logic.

### New Math Required

| Result | Description | Difficulty | Load-Bearing? |
|--------|------------|------------|---------------|
| **Bilevel intersection cuts (full theory)** | Polyhedrality theorem + facet-defining conditions + separation complexity + finite convergence | **C** (novel) | Yes — the mathematical crown jewel. If facet-defining conditions are proved, this is a top-tier theoretical result. If only separation is achieved (without facet theory), the contribution is computational but still significant. The 2-week prototype gate de-risks this. |
| **Value-function lifting** | Gomory–Johnson extension to value-function epigraphs | **C** (novel) | Yes — the second mathematical contribution, providing stronger cuts than raw intersection cuts. Partially de-risked by the fact that LP lower-level value functions are piecewise linear, making the subadditivity theory more tractable. |
| **Compiler soundness theorem** | Full pipeline correctness: parse → analyze → reformulate → cut → emit preserves bilevel optimality | **B** (hard, precedented) | Yes — the central guarantee. Harder than Approach B's version because it must also cover the cut pass (cuts must be valid inequalities, not just any hyperplane). |
| **Certificate composition** | Three-way composition of structural, reformulation, and cut certificates with soundness guarantee | **B** (hard, precedented) | Yes — the novelty is in the composition, not in the individual certificates. Closest precedent is proof-carrying code (Necula 1997), but applied to optimization reformulations rather than program transformations. |
| **Compilability decision** | Polynomial-time decision procedure for the existence of a valid compilation | **B** (hard, precedented) | Yes — same as Approach B |
| **Selection soundness** | Proof that ρ(σ) always selects valid reformulations | **A** (known) | Yes — straightforward case analysis |
| **Sampling-based value-function approximation** | Error bounds for piecewise-linear overestimation of MILP value functions | **B** (hard, precedented) | Partially — needed only for MILP lower levels |

### Best-Paper Potential

**Target venue:** Operations Research or INFORMS Journal on Computing (JOC).

**Why best-paper:** The hybrid approach has the strongest narrative: "We built the first bilevel optimization compiler, proved it correct, used it to discover and deploy a new family of cutting planes, and demonstrated 10–30% gap closure on 2600+ benchmark instances across four solvers." This is a complete story — theory (cuts), systems (compiler), and experiments (BOBILib evaluation) — that fits OR's tradition of papers that combine methodological innovation with computational evidence. The CVXPY analogy is strongest here: CVXPY's contribution was not just the DCP rules (theory) or the software (systems) or the experiments, but their combination into a coherent whole that lowered barriers and enabled new research. BiCut makes the same argument for bilevel optimization: the compiler enables systematic cut deployment, and the cuts validate the compiler.

Operations Research values papers that open new research directions. If BiCut demonstrates that bilevel-specific cutting planes are effective and that a compiler makes them easy to deploy, it invites follow-up work on new bilevel cut families, new reformulation strategies, and applications to new bilevel problem classes — a research program, not just a paper.

**Risk:** Jack-of-all-trades, master-of-none. Mathematical Programming reviewers may find the cut theory insufficiently deep (no facet-defining proof if that result is too hard, only separation and computational results). IJOC reviewers may find the compiler insufficiently novel (BilevelJuMP exists). The paper must thread the needle: enough math to satisfy theory reviewers, enough engineering to satisfy systems reviewers, enough experiments to satisfy computational reviewers. This requires exceptional writing and a clear story about why the two contributions are synergistic rather than independent.

### Hardest Technical Challenge

**Bridging the cut engine with the compiler's reformulation pipeline while maintaining correctness certificates.**

This is the hardest challenge because it sits at the intersection of three difficult subproblems: (1) the cut engine must understand the reformulated MILP's structure well enough to identify bilevel-infeasible points (which requires "looking through" the reformulation to the original bilevel program), (2) the emission layer must translate cut callbacks into four different solver APIs with different callback mechanisms and timing guarantees, and (3) the certificate must compose the reformulation guarantee with the cut validity guarantee to produce an end-to-end correctness statement.

The fundamental tension is: the cuts operate in the *reformulated* space (the MILP's LP relaxation), but bilevel infeasibility is defined in the *original* space (the bilevel program's leader-follower structure). The cut engine must maintain a bidirectional mapping between these spaces — forward (bilevel → MILP) to emit cuts in MILP variables, backward (MILP → bilevel) to check bilevel feasibility of LP relaxation points. This mapping depends on the reformulation strategy: for KKT reformulation, it involves extracting primal-dual pairs from the MILP solution and checking complementarity; for strong duality reformulation, it involves reconstructing the follower's primal solution from the dual; for value-function reformulation, it involves evaluating V(x) at the leader's solution.

**How to address it:**

1. **Define a `BilevelAnnotation` interface in the IR.** Each reformulation pass annotates its output with a mapping from MILP variables back to bilevel variables, a method to extract the leader's decision from an MILP solution, and a method to check bilevel feasibility. The cut engine programs against this interface, not against specific reformulations.

2. **Implement the cut engine as a two-phase system.** Phase 1 (reformulation-agnostic): given an MILP LP relaxation point, use the `BilevelAnnotation` to extract the leader's decision x̂ and check bilevel feasibility. Phase 2 (reformulation-aware): if bilevel-infeasible, compute the intersection cut in the original bilevel space, then map it forward to the MILP space using the reformulation's forward mapping.

3. **Test certificate composition incrementally.** Start with the simplest case (value-function reformulation + intersection cuts on LP lower levels), where the mapping is identity and the certificate composition is trivial. Then extend to KKT and strong duality reformulations, where the mapping involves dual variables and big-M auxiliaries.

4. **Use the BOBILib evaluation as an integration test.** Each instance compiled with each reformulation + cuts combination must produce the same optimal value (within tolerance) as MibS. Discrepancies indicate a bug in the mapping, the cut engine, or the certificate system.

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **6/10** | Same user base as Approach B (practitioners + educators + reviewers + cutting-plane researchers). The hybrid provides strictly more value than either A or B alone — the compiler is useful without cuts, and the cuts are more useful within a compiler. However, the total user base remains small (500–1000 at ceiling). The depth check panel's value score of 4 reflected the niche user base; the hybrid framing as a reproducibility layer and research platform pushes this to 6 by broadening appeal. |
| **Difficulty** | **8/10** | Inherits all difficulty from both A and B, plus the integration challenges. The co-primary structure means neither contribution can be descoped without significantly weakening the paper. The facet-defining proof (Difficulty C), value-function lifting (Difficulty C), four-backend emission, certificate composition, and reformulation-aware cut selection collectively represent an extremely challenging artifact. The ~19K lines of novel code span both cutting-plane theory and compiler engineering. |
| **Potential** | **7/10** | The strongest narrative of the three approaches. If both contributions land, this is a top Operations Research or IJOC paper with a clear shot at best paper. The synergy argument (compiler enables cuts, cuts validate compiler) is compelling and novel. However, the potential is capped by the user base size and the risk that neither contribution reaches the depth required for the top theory venues. Best-paper at OR/IJOC is realistic; best-paper at Math Programming is unlikely unless the facet-defining conditions are fully proved. |
| **Feasibility** | **5/10** | The highest technical risk of the three approaches. The system must deliver on both the compiler *and* the cuts — failure of either significantly weakens the paper. The narrow viability corridor for cuts (from Approach A) is still present, though partially mitigated by the compiler fallback. The integration challenges (cut-compiler bridge, certificate composition, reformulation-aware cut selection) add difficulty that neither A nor B faces alone. The 2-week prototype gate mitigates the existential cut risk, but schedule risk for the full 121K LoC system with both contributions at full depth is substantial. The depth check's "conditional continue" verdict applies most directly to this approach. |

---

## Comparative Summary

| Criterion | A: Cuts-First | B: Compiler-First | C: Hybrid |
|-----------|---------------|-------------------|-----------|
| **Value** | 4 | 6 | 6 |
| **Difficulty** | 9 | 6 | 8 |
| **Potential** | 8 | 5 | 7 |
| **Feasibility** | 4 | 8 | 5 |
| **Total** | **25** | **25** | **26** |
| **Best venue** | Math Programming / IPCO | INFORMS JOC / CPAIOR | Operations Research / JOC |
| **Risk profile** | Binary (cuts work or nothing) | Low (no existential risk) | High but hedgeable (compiler fallback) |
| **Fallback if cuts fail** | None — approach collapses | Cuts descoped, paper unaffected | Compiler-only paper (weaker but publishable) |
| **Key differentiator** | Deepest math; new polyhedral theory | Broadest system; most users served | Strongest narrative; synergistic contributions |
| **Biggest weakness** | Zero fallback; tiny audience | "Just engineering" criticism | Jack-of-all-trades risk |

### Recommendation for Prototype Gate

All three approaches benefit from the 2-week prototype validation gate identified in the depth check. However, the gate's outcome should influence approach selection:

- **Gap closure ≥15%:** Approach A or C are viable. The cuts are the story.
- **Gap closure 10–15%:** Approach C is optimal. The cuts contribute but need the compiler to carry weight.
- **Gap closure 5–10%:** Approach B is safest. The cuts are a minor optimization pass, not a headline.
- **Gap closure <5%:** Approach B is the only viable option. The cuts are descoped entirely.

The prototype gate should be executed before committing to any approach.
