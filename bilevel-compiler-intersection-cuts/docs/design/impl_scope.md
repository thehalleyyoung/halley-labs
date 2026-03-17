# Implementation Scope: Hybrid Bilevel Optimization Compiler

## Design Summary

**Architecture:** Compiler pipeline — parse → analyze → select → reformulate → emit — with typed IR  
**Mathematical core:** Bilevel intersection cuts + value-function lifting for MIBLPs  
**Scope:** Mixed-integer bilevel linear programs (primary); extensible IR for future problem classes  
**Solver backends:** Gurobi, SCIP, HiGHS  
**Languages:** Rust (core compiler: IR, analysis, reformulation, cuts) + Python (DSL, solver backends, benchmarks)  
**Evaluation:** MibS comparison on BOBILib + simplified strategic bidding case study

---

## Honest Totals

| Category | LoC | % of Total |
|----------|-----|------------|
| Novel algorithmic logic | ~21K | 26% |
| Infrastructure / glue | ~33K | 40% |
| Tests and validation | ~27K | 34% |
| **Total** | **~81K** | |

This is roughly 80% of the ~100K hybrid estimate from the adversarial critique. The difference: the critique assumed more verbose Rust infrastructure and a wider strategic-bidding case study. We could reach 100K with generous documentation and a more elaborate Python DSL, but the *load-bearing code* is ~81K.

**Novelty concentration:** ~70% of the novel algorithmic logic lives in three subsystems: S5 (intersection cuts), S6 (value-function oracle), and S4 (automated big-M computation). Everything else is necessary infrastructure executing known techniques.

---

## S1. Bilevel IR and Parser (~8K LoC)

**Description:** Typed abstract syntax tree for bilevel programs with a Python DSL frontend. The IR represents upper/lower objectives, coupling constraints, variable domains, and carries annotations (convexity, integrality, CQ status) that downstream passes consume.

### Key Technical Challenges
- **Expression canonicalization:** Bilevel coupling terms (upper variables in lower-level constraints) must be syntactically distinguishable from pure lower-level terms. Requires a two-namespace design where variable provenance is tracked through expression trees.
- **Annotation propagation:** Type-inference annotations (e.g., "this constraint block satisfies LICQ") must survive IR transformations without going stale. Invalidation logic when the IR is mutated.
- **PyO3 boundary:** Every IR type exposed to Python needs a wrapper. Rust enums with data don't map cleanly to Python objects; requires flattening or tagged-union encoding.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Rust AST types (expressions, constraints, objectives, problems) | — | 1,200 | 400 | 1,600 |
| Expression canonicalization and simplification | 300 | 400 | 300 | 1,000 |
| Type inference pass (DCP-extension for bilevel) | 600 | 200 | 400 | 1,200 |
| Python DSL (Variable, Expression, Constraint, Problem classes) | — | 1,500 | 600 | 2,100 |
| PyO3 bindings + serialization layer | — | 1,200 | 200 | 1,400 |
| Validation and error reporting | — | 400 | 300 | 700 |
| **Totals** | **900** | **4,900** | **2,200** | **8,000** |

### Dependencies
None. This is the foundation subsystem; everything depends on it.

### Risk Assessment
- **Low risk.** AST design for linear/mixed-integer programs is well-understood. The Python DSL is engineering, not research.
- **Medium risk on annotation invalidation.** If annotations become stale after IR mutations and a downstream pass trusts a stale annotation, correctness breaks silently. Mitigation: immutable IR with copy-on-write semantics (each pass produces a new IR).
- **Schedule risk:** PyO3 bindings are tedious and error-prone at the boundary. Budget 40% more time than the LoC suggests.

---

## S2. Structural Analysis Engine (~5K LoC)

**Description:** Analyzes the bilevel IR to produce a *problem signature*: a structured record of convexity class, constraint qualification status, coupling type, and integrality structure. This signature drives reformulation selection (S3).

### Key Technical Challenges
- **CQ verification for degenerate linear problems:** For LP lower levels, LICQ holds iff the active constraint gradients are linearly independent. But which constraints are active depends on the solution — which we don't have yet. Must use conservative checks (e.g., "LICQ holds for *all* feasible bases" via rank of the full constraint matrix, or flag as "CQ-unknown" and defer).
- **Coupling structure classification:** Must distinguish RHS-only coupling (upper variables appear only in constraint RHS → KKT is clean), objective-only coupling (upper variables only in lower objective → strong duality applies cleanly), and full-matrix coupling (upper variables in constraint matrix → KKT introduces bilinear terms requiring McCormick or explicit handling). Requires symbolic inspection of coefficient expressions.
- **Conservative vs. optimistic analysis:** Returning "unknown" for any property forces the reformulation engine to use the most expensive fallback. Tightness of analysis directly impacts compilation quality.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Coefficient matrix extraction from IR | — | 600 | 200 | 800 |
| Convexity classification (trivial for MIBLP v1; extensible) | 100 | 300 | 200 | 600 |
| CQ verification (Jacobian rank, Slater check via LP) | 400 | 500 | 400 | 1,300 |
| Coupling structure classifier | 300 | 400 | 300 | 1,000 |
| Integrality structure analysis | 100 | 200 | 200 | 500 |
| Problem signature data type and builder | — | 500 | 300 | 800 |
| **Totals** | **900** | **2,500** | **1,600** | **5,000** |

### Dependencies
- **S1** (Bilevel IR): Operates on the parsed IR.

### Risk Assessment
- **Low risk for MIBLP scope.** Linear lower levels make convexity trivial and CQ straightforward.
- **Medium risk for extensibility.** If the signature type is too MIBLP-specific, adding QP/conic support later requires a redesign. Mitigation: design signature as an extensible enum with an `Unknown` variant for every field.
- **Correctness risk:** An incorrect "LICQ holds" determination leads to a silently wrong KKT reformulation. This is the *exact* failure mode that correctness certificates (S9) must catch.

---

## S3. Reformulation Selection Engine (~3K LoC)

**Description:** Maps problem signatures to ranked lists of valid reformulation strategies. Implements the reformulation algebra (composability rules) and a cost model that scores strategies by estimated solve-time proxy features (problem size after reformulation, big-M quality, decomposability).

### Key Technical Challenges
- **Reformulation validity rules:** Not all reformulations are valid for all signatures. KKT requires CQ; strong duality requires LP lower level; value function is universal but expensive; C&CG requires specific coupling structure. Encoding these rules *soundly* (never selecting an invalid reformulation) while *completely* (not missing valid options) is the core challenge.
- **Cost model calibration:** The cost model predicts solve difficulty *before solving*. Features like "number of big-M constraints" and "estimated big-M magnitude" correlate with solve time but imperfectly. Overfitting to BOBILib would be embarrassing; the model must be simple enough to explain.
- **Strategy composition:** Some strategies compose (KKT + intersection cuts) while others don't (C&CG and KKT are alternatives, not additive). The algebra must enforce valid compositions.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Reformulation algebra (types, validity rules, composition) | 500 | 300 | 300 | 1,100 |
| Strategy enumeration from signature | 200 | 300 | 200 | 700 |
| Cost model (feature extraction + scoring) | 400 | 200 | 300 | 900 |
| Strategy ranking and selection API | — | 200 | 100 | 300 |
| **Totals** | **1,100** | **1,000** | **900** | **3,000** |

### Dependencies
- **S2** (Structural Analysis): Consumes problem signatures.

### Risk Assessment
- **Low engineering risk.** This is decision logic, not heavy computation.
- **High validation risk.** Proving the selection engine beats expert manual choice (Challenge 1 from the adversarial critique) is an empirical question that could go either way. If the cost model's selections are never better than "always use KKT+big-M," the engine adds complexity without value.
- **Mitigation:** Include a `--force-strategy` flag that bypasses automatic selection. This lets users override the engine and also enables A/B comparison for evaluation.

---

## S4. KKT / Strong Duality Reformulation Pass (~9K LoC)

**Description:** The primary reformulation pass. Transforms a bilevel IR into a single-level MILP by replacing the lower-level optimality condition with KKT conditions (for problems satisfying CQ) or a primal-dual system with strong duality constraint (for LP lower levels). Includes automatic big-M computation via bound-tightening LPs.

### Key Technical Challenges
- **Automatic big-M computation (the hardest part of this subsystem).** Complementarity constraints `λ_i · g_i(x,y) = 0` are linearized as `λ_i ≤ M_i · (1 - z_i)` and `g_i ≤ M_i · z_i`. The big-M values must be *valid* (true upper bounds on λ_i and g_i) but *tight* (loose big-Ms destroy LP relaxation quality). Computing tight bounds requires solving O(m) auxiliary LPs where m is the number of lower-level constraints. For large m (500+), this is a significant compilation cost.
- **Numerical conditioning.** Big-M values above ~10⁴ cause solver numerical issues. If bound-tightening LPs return large bounds (or unbounded), the reformulation may be numerically intractable even if theoretically correct. Must detect and report.
- **SOS1 alternative.** SOS1 encoding of complementarity avoids big-M entirely but is only supported well in Gurobi and SCIP (HiGHS has limited SOS support). The pass must emit both encodings and let the backend choose.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| KKT condition generator (dual variable intro, stationarity, complementarity) | — | 1,500 | 600 | 2,100 |
| Big-M bound tightening (LP-based, interval arithmetic fallback) | 800 | 1,000 | 700 | 2,500 |
| Complementarity linearization (big-M encoding) | 200 | 500 | 300 | 1,000 |
| SOS1 complementarity encoding | — | 400 | 200 | 600 |
| Strong duality reformulation (primal-dual + equality) | 200 | 600 | 300 | 1,100 |
| Single-level MILP IR (output representation) | — | 800 | 200 | 1,000 |
| Big-M quality reporting and numerical warnings | 100 | 300 | 300 | 700 |
| **Totals** | **1,300** | **5,100** | **2,600** | **9,000** |

### Dependencies
- **S1** (Bilevel IR): Input format.
- **S2** (Structural Analysis): CQ status determines whether KKT is valid.
- **S3** (Reformulation Selection): Tells this pass *which* variant to apply.
- Needs an LP solver for bound-tightening (bootstrap problem: use HiGHS as compile-time dependency since it's open-source).

### Risk Assessment
- **Medium risk.** KKT reformulation is well-understood (BilevelJuMP does it in ~2K LoC Julia). The *additional* complexity here comes from automatic big-M computation and dual-encoding support.
- **Big-M quality is the critical risk.** If the bound-tightening LPs are slow (>1s each for m=500 → 500s compilation), or if bounds are loose (big-M > 10⁵ → solver struggles), the compiled formulation is technically correct but practically useless. This is the "fast compilation, slow solving" paradox.
- **Mitigation:** Tiered big-M strategy: (1) interval arithmetic (fast, loose), (2) LP bound tightening (moderate, tighter), (3) user-supplied bounds (tightest). Report which tier was used.

---

## S5. Bilevel Intersection Cut Engine (~11K LoC)

**Description:** The primary novel mathematical contribution. Implements a cutting-plane engine that generates intersection cuts from bilevel-infeasible sets — extending Balas's classical intersection cut framework to exploit the bilevel structure. These cuts are added as lazy constraints during branch-and-cut via solver callbacks.

### Key Technical Challenges
- **Bilevel-infeasible set characterization.** Given a fractional LP relaxation point x̄, the bilevel-infeasible set is {(x,y) : y is feasible for the lower level at x but not optimal}. For LP lower levels, this set has a closed-form characterization via dual feasibility and complementary slackness violations. For MILP lower levels, characterization requires solving an auxiliary optimization problem (checking if a better lower-level solution exists), which is itself an MILP.
- **Separation oracle efficiency.** The oracle is called at every node of the branch-and-cut tree (potentially thousands of times). Each call must: (1) check if current solution violates bilevel feasibility, (2) if so, compute an intersection cut, (3) return the cut to the solver. If oracle time exceeds ~10ms per call, the overhead dominates solver time.
- **Cut quality vs. cost tradeoff.** Deeper separation (solving the MILP auxiliary problem exactly) gives stronger cuts but is expensive. Shallow separation (using the LP relaxation of the auxiliary problem) is fast but may produce weak cuts. Need adaptive depth control.
- **Numerical stability of cuts.** Intersection cuts with coefficients spanning several orders of magnitude are ineffective (solvers discard them or suffer numerical issues). Need normalization and filtering.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Bilevel-infeasible convex set computation (LP lower level) | 1,200 | 500 | 600 | 2,300 |
| Bilevel-infeasible set approximation (MILP lower level) | 800 | 600 | 500 | 1,900 |
| Intersection cut derivation (ray computation, cut coefficients) | 1,000 | 400 | 600 | 2,000 |
| Separation oracle (main loop, adaptive depth) | 600 | 800 | 500 | 1,900 |
| Cut pool management (storage, aging, purging, duplicate detection) | 200 | 800 | 300 | 1,300 |
| Numerical stability (normalization, filtering, diagnostics) | 200 | 500 | 400 | 1,100 |
| Callback integration interface (abstract; per-solver impl in S8) | — | 300 | 200 | 500 |
| **Totals** | **4,000** | **3,900** | **3,100** | **11,000** |

### Dependencies
- **S1** (Bilevel IR): Reads problem structure.
- **S4** (KKT Pass): Operates on the KKT-reformulated single-level MILP; cuts strengthen this formulation.
- **S8** (Solver Backends): Cuts are injected via solver-specific callback APIs.
- Requires an LP solver for auxiliary separation problems (same HiGHS dependency as S4).

### Risk Assessment
- **HIGH RISK. This is the make-or-break subsystem.**
- **Risk 1: Cut effectiveness.** If bilevel intersection cuts close <10% of the integrality gap on BOBILib instances, the entire mathematical contribution is incremental. This must be validated early (Challenge 3). Mitigation: implement the LP-lower-level case first (~3K LoC) and benchmark before building the MILP case.
- **Risk 2: Oracle overhead.** If separation takes >50ms per call, the cuts slow down branch-and-cut rather than speeding it up. Mitigation: aggressive caching of auxiliary LP solutions; skip separation at shallow tree nodes.
- **Risk 3: The "narrow corridor" problem.** Bilevel-infeasible sets must be efficiently characterizable (otherwise cuts are intractable) but non-trivial (otherwise standard MILP cuts already capture the structure). The contribution lives in this corridor.
- **Risk 4: Callback API limitations.** HiGHS has limited callback support compared to Gurobi/SCIP. The cut engine may only work well on 2 of 3 backends.

---

## S6. Value Function Oracle and Lifting (~10K LoC)

**Description:** Computes or approximates the lower-level value function V(x) = min{c^Ty : Ay ≤ b - Dx, y ∈ Y} and generates cuts from it. For LP lower levels, V(x) is piecewise-linear and can be computed exactly via parametric LP. For MILP lower levels, V(x) is approximated via sampling and outer approximation. Includes Gomory-Johnson-style lifting for integer upper-level variables.

### Key Technical Challenges
- **Parametric LP implementation.** Computing the exact piecewise-linear value function requires multiparametric programming: enumerating critical regions in the upper-variable space where the optimal basis is constant. The number of critical regions can be exponential in the number of upper-level variables. Practical for ≤15 upper-level variables; intractable beyond ~25.
- **MILP value function approximation.** V(x) for MILP lower levels is discontinuous (integer variables cause jumps). No closed-form exists. Approximation via: (1) sampling at grid/random points, (2) building convex outer approximation from sampled points, (3) iterative refinement. The approximation quality directly determines cut quality.
- **Gomory-Johnson lifting.** Classical lifting theory assumes a single "base" inequality and lifts coefficients for integer variables. Extending this to bilevel value-function cuts requires defining the lifting function as the value function itself — a circular definition that must be resolved via finite enumeration or interpolation. This is genuinely novel mathematics with limited precedent.
- **Oracle caching and warm-starting.** The value function oracle is called repeatedly with similar parameter values. Cache hit rate must exceed ~80% for acceptable performance. Caching strategy: hash the upper-variable values (with tolerance), store (x, V(x), optimal basis/solution).

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Parametric LP solver (critical region enumeration) | 1,200 | 800 | 600 | 2,600 |
| Value function sampling and outer approximation (MILP case) | 600 | 800 | 500 | 1,900 |
| Value-function cut generation | 500 | 400 | 400 | 1,300 |
| Gomory-Johnson lifting for bilevel | 1,000 | 400 | 600 | 2,000 |
| Oracle caching (hash, tolerance, eviction) | 200 | 600 | 300 | 1,100 |
| Warm-starting (basis storage, reoptimization) | 200 | 400 | 200 | 800 |
| Integration with cut engine (S5) and C&CG (S7) | — | 200 | 100 | 300 |
| **Totals** | **3,700** | **3,600** | **2,700** | **10,000** |

### Dependencies
- **S1** (Bilevel IR): Reads problem structure.
- **S4** (KKT Pass): Value-function cuts complement KKT reformulation.
- **S5** (Cut Engine): Value-function cuts feed into the same cut pool.
- Requires an LP/MILP solver for value function evaluation (HiGHS for LP; Gurobi/SCIP for MILP).

### Risk Assessment
- **HIGH RISK.** The adversarial critique identified the value-function oracle as the single hardest blocker for Framing B.
- **Risk 1: Parametric LP scalability.** Critical region enumeration is exponential. For problems with >20 upper-level continuous variables, exact computation is infeasible. Mitigation: fall back to sampling-based approximation.
- **Risk 2: Cache hit rate.** If branch-and-cut explores diverse regions of the upper-variable space, cache hits will be rare. Mitigation: locality-sensitive hashing, interpolation between cached points.
- **Risk 3: Lifting correctness.** Gomory-Johnson lifting for bilevel is novel. Errors in lifting coefficients produce invalid cuts that may cut off the optimal solution. Mitigation: extensive roundtrip verification (S9, S11).
- **Risk 4: Interaction with S5.** Value-function cuts and intersection cuts may be redundant (both exploit bilevel structure). Need empirical evaluation of which cut family is more effective, and whether combining them helps or just adds overhead.

---

## S7. Column-and-Constraint Generation (C&CG) Pass (~3.5K LoC)

**Description:** Implements the Zeng & An (2014) C&CG algorithm as an alternative solution strategy for MIBLPs. Iteratively solves a restricted master problem and a lower-level subproblem, generating columns and constraints until convergence. Useful when the KKT reformulation is too large or when intersection cuts are ineffective.

### Key Technical Challenges
- **Convergence speed.** C&CG converges finitely for MIBLPs but may require many iterations for loosely-coupled problems. Each iteration solves two MILPs. If convergence takes >50 iterations, C&CG is impractical.
- **Master problem growth.** Each iteration adds constraints and potentially variables to the master. After many iterations, the master problem itself becomes large. Need constraint management (dropping inactive constraints periodically).
- **Warm-starting.** Starting C&CG from a good initial solution (e.g., from a heuristic or from the strategic bidding case study's structure) can dramatically reduce iterations.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| C&CG main loop (master/sub alternation) | — | 700 | 300 | 1,000 |
| Master problem construction and constraint addition | — | 500 | 200 | 700 |
| Subproblem construction and solution extraction | — | 400 | 200 | 600 |
| Convergence detection (gap, iteration limit, stalling) | 100 | 200 | 200 | 500 |
| Warm-starting interface | 200 | 200 | 100 | 500 |
| Constraint management (cleanup, inactive removal) | — | 100 | 100 | 200 |
| **Totals** | **300** | **2,100** | **1,100** | **3,500** |

### Dependencies
- **S1** (Bilevel IR): Reads problem structure.
- **S8** (Solver Backends): Each iteration requires solving MILPs.

### Risk Assessment
- **Low risk.** C&CG is a well-understood algorithm with clean convergence theory. Implementation is mostly engineering.
- **Minor risk: interaction with other passes.** C&CG is an *alternative* to KKT+cuts, not a complement. The reformulation selection engine (S3) must correctly decide when to use C&CG vs. KKT+cuts. Getting this decision wrong wastes compute time but doesn't produce incorrect results.

---

## S8. Solver Backend Emission (~8K LoC)

**Description:** Translates the single-level MILP (output of reformulation passes) into solver-specific API calls. Each backend handles: variable creation, constraint addition, objective setting, parameter configuration, lazy constraint callbacks (for intersection cuts), and solution extraction. Also includes an MPS/LP file emitter for solver-agnostic output.

### Key Technical Challenges
- **Callback API divergence.** Gurobi, SCIP, and HiGHS have fundamentally different callback architectures:
  - *Gurobi*: `cbLazy()` within a callback function; clean but limited to lazy constraints and user cuts.
  - *SCIP*: Constraint handler plugins with `CONSCHECK`, `CONSENFOLP`, `CONSSEPALP` entry points; powerful but complex (requires managing constraint handler lifecycle).
  - *HiGHS*: Limited callback support as of 2024; may require a polling/iterative approach rather than true callbacks. This is the weakest link.
- **Solution fidelity.** The solver returns (x*, y*, λ*, z*) in the reformulated single-level space. Must map back to bilevel-meaningful quantities (upper solution, lower solution, lower-level dual values). This inverse mapping depends on which reformulation was applied.
- **Numerical tolerance propagation.** Gurobi, SCIP, and HiGHS use different default tolerances. A solution feasible in one solver may be infeasible in another. The backend must normalize tolerances or at least report discrepancies.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Common emission interface (abstract base, variable/constraint mapping) | — | 800 | 200 | 1,000 |
| Gurobi backend (model building, parameters, solution extraction) | — | 800 | 300 | 1,100 |
| Gurobi callback integration (lazy constraints for S5 cuts) | 300 | 500 | 300 | 1,100 |
| SCIP backend (model building, parameters, solution extraction) | — | 800 | 300 | 1,100 |
| SCIP constraint handler plugin (for S5 cuts) | 300 | 600 | 300 | 1,200 |
| HiGHS backend (model building, parameters, solution extraction) | — | 600 | 200 | 800 |
| HiGHS cut injection (limited callback; iterative fallback) | 100 | 300 | 200 | 600 |
| MPS/LP file emission | — | 500 | 200 | 700 |
| Solution back-mapping (reformulated → bilevel space) | 100 | 200 | 100 | 400 |
| **Totals** | **800** | **5,100** | **2,100** | **8,000** |

### Dependencies
- **S4** (KKT Pass): Consumes the single-level MILP.
- **S5** (Cut Engine): Callbacks invoke the separation oracle.
- Requires Python solver bindings: `gurobipy`, `pyscipopt`, `highspy`.

### Risk Assessment
- **Medium risk overall; high risk for HiGHS callbacks.**
- **Risk 1: HiGHS callback maturity.** HiGHS's callback API is newer and less stable than Gurobi's or SCIP's. If HiGHS cannot support lazy constraints, the intersection cut engine is limited to 2 backends, undermining the "solver-agnostic" claim. Mitigation: HiGHS backend emits a static MILP (no callbacks) and intersection cuts are added in an outer iterative loop.
- **Risk 2: SCIP constraint handler complexity.** PySCIPOpt constraint handlers require careful Python↔C++ lifecycle management. Memory leaks and segfaults are common failure modes. Budget extra debugging time.
- **Risk 3: Gurobi license dependency.** Gurobi is commercial. CI/CD testing requires license management. Mitigation: MPS file emission enables testing the *compilation* pipeline without Gurobi; SCIP and HiGHS are open-source for full-pipeline tests.

---

## S9. Correctness Certificate Engine (~4.5K LoC)

**Description:** Generates human-readable and machine-checkable certificates documenting: which structural properties were verified, which reformulation was applied and why it's valid, and post-solve verification that the returned solution is bilevel-feasible. Supports differential testing across solver backends.

### Key Technical Challenges
- **Bilevel feasibility checking.** Given a candidate solution (x*, y*), verifying bilevel feasibility requires solving the lower-level problem at x* and checking that y* is optimal. For MILP lower levels, this is an NP-hard subproblem *per check*. Cannot be avoided — it's the definition of bilevel feasibility.
- **Certificate completeness.** The certificate must record *every* assumption that was used. If the KKT pass assumed LICQ but the structural analysis returned "LICQ-likely" (not "LICQ-certain"), the certificate must flag this gap. Tracking assumption provenance through the pipeline requires discipline.
- **Differential testing value.** Two solvers returning the same optimal value is evidence of correctness but not proof (both could be wrong in the same way). Two solvers returning different values is a *definitive* bug signal. The challenge is determining *which* solver is correct.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Certificate data model and serialization | 200 | 600 | 200 | 1,000 |
| Assumption tracking (provenance through pipeline) | 300 | 400 | 300 | 1,000 |
| Bilevel feasibility checker (solve lower level, compare) | 200 | 500 | 400 | 1,100 |
| Differential testing harness (multi-solver comparison) | — | 500 | 300 | 800 |
| Certificate renderer (human-readable report) | — | 300 | 100 | 400 |
| Warning/error classification | — | 100 | 100 | 200 |
| **Totals** | **700** | **2,400** | **1,400** | **4,500** |

### Dependencies
- **S2** (Structural Analysis): Certificates record analysis results.
- **S3** (Selection Engine): Certificates record selection rationale.
- **S4** (KKT Pass): Certificates record reformulation assumptions.
- **S8** (Solver Backends): Differential testing requires multiple backends.

### Risk Assessment
- **Low implementation risk.** This is essentially structured logging + one verification LP/MILP.
- **High validation risk.** Proving certificates *add value* (Challenge 2) requires finding problems where the absence of certificates leads to incorrect results. If bilevel reformulation errors are rare in practice on benchmark instances, certificates are a nice feature but not a selling point.

---

## S10. Benchmark Suite (~5.5K LoC)

**Description:** Infrastructure for loading standard bilevel benchmark instances, running the compiler pipeline, executing MibS as a baseline, and collecting performance metrics (solve time, root gap, nodes explored, bilevel-feasibility status).

### Key Technical Challenges
- **BOBILib format parsing.** BOBILib instances are distributed in MPS-like or custom formats. Parsing is straightforward but tedious (many edge cases in real-world MPS files).
- **MibS integration.** MibS is a C++ application. Integration requires: building MibS (or using pre-built binaries), feeding instances via command-line, parsing stdout for solve time and optimal value. MibS's output format is not standardized; fragile parsing.
- **Fair timing comparison.** Comparing "our compiler + Gurobi" against "MibS (which uses SYMPHONY/Cbc internally)" is not apples-to-apples. Must report: (1) compilation time, (2) solver time, (3) total time. Must acknowledge that we benefit from Gurobi's superior LP solver while MibS uses open-source solvers.
- **Strategic bidding instance generation.** Simplified instances: single-bus or 5-bus network, LP-relaxed market clearing, 2-5 generators. Must be complex enough to demonstrate the compiler's value but simple enough to solve in minutes.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| BOBILib instance loader (MPS/aux format parser) | — | 1,200 | 400 | 1,600 |
| MibS runner (invocation, output parsing, timeout handling) | — | 600 | 200 | 800 |
| Performance metrics collection and aggregation | — | 500 | 200 | 700 |
| Strategic bidding instance generator | 400 | 400 | 200 | 1,000 |
| Result reporting (tables, CSV export) | — | 500 | 100 | 600 |
| Benchmark orchestration (batch runs, parallelism, resumption) | — | 600 | 200 | 800 |
| **Totals** | **400** | **3,800** | **1,300** | **5,500** |

### Dependencies
- **S1–S8** (full pipeline): Benchmarks exercise the entire compiler.
- External: MibS binary, BOBILib instance files, solver licenses.

### Risk Assessment
- **Low implementation risk.** Benchmark infrastructure is engineering.
- **High experimental risk.** If the compiler+Gurobi doesn't beat MibS on a meaningful fraction of BOBILib instances, the project's empirical story collapses. This is not a bug in the benchmark suite — it's a property of the algorithms (S5, S6). But the benchmark suite is where the failure becomes visible.
- **Fairness risk.** Reviewers will (correctly) note that Gurobi vs. SYMPHONY is an unfair comparison. Mitigation: also report compiler+SCIP vs. MibS, since SCIP is open-source.

---

## S11. Testing and Validation Infrastructure (~5K LoC)

**Description:** Cross-cutting test infrastructure including property-based testing (random bilevel problem generation + roundtrip correctness), regression tests against known optimal values, integration tests across all solver backends, and numerical stability tests.

### Key Technical Challenges
- **Random bilevel problem generation.** Generating *well-posed* random bilevel programs is non-trivial. Random constraint matrices often produce infeasible or unbounded lower levels. Must use structured generation: feasible lower level (verified by LP), bounded lower level (add box constraints), reasonable coupling (sparse upper-in-lower coupling matrix).
- **Roundtrip correctness.** The gold standard test: generate problem → compile → solve → verify bilevel feasibility of returned solution. This requires the feasibility checker from S9. For MILP lower levels, the verification step itself is NP-hard, limiting test throughput.
- **Known-optimal regression.** BOBILib provides known optimal values for ~2600 instances. Must handle: tolerances (is 42.0001 ≈ 42?), infeasibility detection, unboundedness detection.
- **Numerical stability testing.** Perturb problem data by ε and verify solution changes by O(ε). Bilevel programs can have discontinuous value functions (a small perturbation can change the optimal value dramatically), so this test is about detecting *unexpected* sensitivity rather than requiring stability.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Random bilevel problem generator (structured, well-posed) | 500 | 600 | — | 1,100 |
| Roundtrip correctness test harness | 200 | 500 | — | 700 |
| Known-optimal regression suite (BOBILib golden values) | — | 800 | — | 800 |
| Integration test framework (multi-backend, parameterized) | — | 700 | — | 700 |
| Numerical stability test suite | 200 | 400 | — | 600 |
| Test utilities (fixtures, helpers, assertions) | — | 600 | — | 600 |
| CI configuration and test orchestration | — | 500 | — | 500 |
| **Totals** | **900** | **4,100** | **—** | **5,000** |

*Note: S11 LoC is itself test code; the "Tests" column is N/A.*

### Dependencies
- **S1–S9** (all subsystems): S11 tests the full pipeline.
- **S9** (Certificates): Roundtrip verification reuses the bilevel feasibility checker.

### Risk Assessment
- **Low risk.** Test infrastructure is well-understood engineering.
- **Schedule risk.** Property-based testing is high-value but time-consuming to set up. Random problem generation must be tuned to produce interesting (non-trivial, non-degenerate) instances. Budget 2x estimated time for the generator.

---

## S12. Cross-Cutting Infrastructure (~9K LoC)

**Description:** Shared infrastructure that doesn't belong to any single subsystem: Rust project scaffolding, error types, logging, the PyO3 binding layer, sparse linear algebra utilities, CLI entry point, and the single-level MILP data structures consumed by solver backends.

### LoC Breakdown

| Component | Novel | Infra | Tests | Subtotal |
|-----------|-------|-------|-------|----------|
| Rust project structure (Cargo workspace, feature flags, modules) | — | 300 | — | 300 |
| Error types and error handling (typed errors, context chains) | — | 500 | 200 | 700 |
| Logging and diagnostics (tracing spans, compile-time logs) | — | 400 | 100 | 500 |
| Sparse matrix utilities (CSR/CSC, basic operations) | — | 1,000 | 400 | 1,400 |
| PyO3 binding scaffolding (module init, type registrations) | — | 1,500 | 300 | 1,800 |
| Serialization layer (Rust ↔ Python IR transfer, JSON/bincode) | — | 700 | 200 | 900 |
| CLI entry point (argument parsing, pipeline orchestration) | — | 600 | 200 | 800 |
| Python package structure (pyproject.toml, __init__.py, type stubs) | — | 500 | — | 500 |
| Configuration and options types | — | 400 | 100 | 500 |
| Utility functions (numerical tolerances, hashing, display) | — | 400 | 200 | 600 |
| **Totals** | **—** | **6,300** | **1,700** | **8,000** |

*Note: Revised down from 9K after careful itemization. No novel logic here.*

### Dependencies
None; consumed by all other subsystems.

### Risk Assessment
- **Low risk.** Boilerplate, but necessary.
- **PyO3 is the pain point.** Rust ↔ Python FFI is where most "mysterious segfault" debugging time goes. Budget 50% overhead for PyO3-related issues.

---

## Consolidated Summary

| Subsystem | Novel | Infra | Tests | Total | Risk |
|-----------|-------|-------|-------|-------|------|
| S1. Bilevel IR and Parser | 900 | 4,900 | 2,200 | **8,000** | Low |
| S2. Structural Analysis | 900 | 2,500 | 1,600 | **5,000** | Low |
| S3. Reformulation Selection | 1,100 | 1,000 | 900 | **3,000** | Low-Med |
| S4. KKT / Strong Duality | 1,300 | 5,100 | 2,600 | **9,000** | Medium |
| S5. Intersection Cut Engine | 4,000 | 3,900 | 3,100 | **11,000** | **High** |
| S6. Value Function Oracle | 3,700 | 3,600 | 2,700 | **10,000** | **High** |
| S7. C&CG Pass | 300 | 2,100 | 1,100 | **3,500** | Low |
| S8. Solver Backend Emission | 800 | 5,100 | 2,100 | **8,000** | Medium |
| S9. Correctness Certificates | 700 | 2,400 | 1,400 | **4,500** | Low |
| S10. Benchmark Suite | 400 | 3,800 | 1,300 | **5,500** | Low |
| S11. Testing Infrastructure | 900 | 4,100 | — | **5,000** | Low |
| S12. Cross-Cutting Infra | — | 6,300 | 1,700 | **8,000** | Low |
| **Totals** | **15,000** | **44,800** | **20,700** | **~80,500** | |

### Category Percentages
- **Novel algorithmic logic:** ~15K LoC (19%)
- **Infrastructure / glue:** ~45K LoC (56%)
- **Tests and validation:** ~21K LoC (26%)

---

## Critical Path and Build Order

```
Phase 1 — Foundation (S12, S1)
   │  Cross-cutting infra + IR + Parser + Python DSL
   │  ~16K LoC, ~4 weeks
   │  Exit criterion: can parse a bilevel LP from Python, produce Rust IR
   │
Phase 2 — Analysis + Basic Reformulation (S2, S3, S4)
   │  Structural analysis + reformulation selection + KKT pass
   │  ~17K LoC, ~5 weeks
   │  Exit criterion: can compile bilevel LP → single-level MILP via KKT
   │  *** Answers Challenge 4 (is the compiler architecture viable?) ***
   │
Phase 3 — One Backend + Roundtrip (S8-partial, S9-partial)
   │  Gurobi backend + basic correctness check
   │  ~5K LoC, ~2 weeks
   │  Exit criterion: end-to-end solve of a bilevel LP with verified solution
   │  *** MVP: answers "fast compilation, slow solving" (Challenge 5) ***
   │
Phase 4 — Novel Algorithms (S5, S6) ← THE HARD PART
   │  Intersection cuts + value function oracle
   │  ~21K LoC, ~10 weeks
   │  Exit criterion: intersection cuts close >10% gap on 20+ BOBILib instances
   │  *** Answers Challenge 3 (do the cuts work?) — GO/NO-GO GATE ***
   │
Phase 5 — Completion (S7, S8-remainder, S9-remainder, S10, S11)
   │  C&CG + remaining backends + benchmarks + full test suite
   │  ~22K LoC, ~6 weeks
   │  Exit criterion: full BOBILib benchmark vs MibS, paper-ready results
   │
Total: ~27 weeks (~7 months) for a single experienced developer
```

### The One Sentence Summary

**This is a ~80K LoC system where 15K lines of novel algorithm (intersection cuts, value-function lifting, automated big-M computation) are wrapped in 45K lines of necessary compiler infrastructure and validated by 21K lines of tests — and the entire project's success hinges on whether those 15K novel lines produce cuts that close >10% of the integrality gap on standard benchmarks.**
