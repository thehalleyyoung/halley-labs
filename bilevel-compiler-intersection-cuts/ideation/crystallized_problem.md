# BiCut: A Solver-Agnostic Bilevel Optimization Compiler with Automated Reformulation Selection, Correctness Certificates, and Bilevel Intersection Cuts

**Slug:** `bilevel-compiler-intersection-cuts`

---

## 1. Problem Statement

Bilevel optimization — in which one optimization problem is nested as a constraint within another — is the natural modeling language for interdiction, competitive facility location, strategic bidding in electricity markets, Stackelberg security games, and adversarial robustness certification. Despite two decades of algorithmic progress on mixed-integer bilevel linear programs (MIBLPs), the software landscape remains deeply fragmented. A practitioner who models a bilevel program today must manually choose among fundamentally different reformulation paradigms (KKT complementarity, strong duality, value-function replacement, column-and-constraint generation), hand-derive the resulting single-level surrogate, debug complementarity encoding or big-M calibration by inspection, and hard-wire the output to a single solver's API. Every step is error-prone, none is formally verified, and switching solvers requires rewriting the reformulation pipeline from scratch. Existing tools — BilevelJuMP.jl, PAO, GAMS EMP, YALMIP — each support one or two strategies with no automatic selection, no correctness guarantees, and no principled handling of mixed-integer lower levels where KKT conditions are inapplicable.

The root cause is a missing abstraction layer: bilevel optimization lacks a *compiler*. In single-level optimization, disciplined convex programming (CVXPY, CVX) demonstrated that a typed intermediate representation (IR) coupled with structural analysis and automatic transformation selection can make an entire class of problems accessible without sacrificing mathematical rigor. No analogous system exists for bilevel programs. No tool today reasons about *which* reformulation to apply for a given problem structure, *whether* the chosen reformulation preserves bilevel optimality, or *how* to emit the resulting single-level program to an arbitrary MIP solver with formal equivalence certificates. This compiler gap forces researchers to treat reformulation as a craft rather than a discipline, inhibits systematic comparison of reformulation strategies, and blocks reproducibility — since bilevel-feasible solutions cannot be independently verified from a solver's MILP output without the original bilevel model.

BiCut closes this gap for MIBLPs. It introduces a typed bilevel IR that captures leader/follower variable scoping, integrality, constraint qualification status, and coupling structure as first-class type annotations. A structural analysis pass infers convexity, constraint regularity, boundedness, and integrality properties, feeding a reformulation selection engine that maps problem *signatures* to the provably valid subset of available reformulations — KKT/MPEC (when CQs hold and the lower level is continuous), strong duality replacement (for LP lower levels), value-function reformulation (always valid but computationally expensive), and column-and-constraint generation (for problems with complicating upper-level constraints). Each reformulation is implemented as a semantics-preserving compiler pass with a machine-checkable correctness certificate: a conjunction of verified preconditions (LICQ, Slater, boundedness, integrality structure) that guarantees the output MILP is bilevel-equivalent to the input. On the cutting plane side, BiCut introduces *bilevel intersection cuts* — a new family of valid inequalities derived by extending Balas's 1971 intersection cut framework to bilevel-infeasible sets — and *value-function lifting*, a Gomory–Johnson-style strengthening procedure that exploits the value-function structure of bilevel programs. These cuts tighten the root LP relaxation of the reformulated MILP, targeting 10–30% root gap closure on hard instances. The compiled output is a standard `.mps` or `.lp` file emittable to Gurobi, SCIP, HiGHS, or CPLEX — four backends spanning commercial, academic, and open-source solvers — meaning any downstream user can reproduce the solution without installing BiCut.

BiCut targets MIBLPs and QP lower levels, the problem classes where the mathematical contribution is deepest. For MIBLPs, the benchmark infrastructure (BOBILib, 2600+ instances) is mature, and the baseline solver (MibS) provides a clean apples-to-apples comparison. The typed IR is *extensible* to conic bilevel programs, pessimistic formulations, and multi-follower games, but these extensions are explicit non-goals for the initial system. BiCut does not target nonlinear bilevel programs (no Ipopt backend), adversarial robustness certification (requires GPU-accelerated neural network verification), or full-fidelity energy market modeling (multi-year domain expertise). The system compiles, cuts, and emits — it does not solve.

---

## 2. Value Proposition

**Who needs this desperately:**

- **MIBLP researchers** (50–200 active worldwide) who currently hand-code reformulations for each paper, cannot systematically compare strategies, and lack tools for mixed-integer lower levels beyond brute-force enumeration or MibS's direct branch-and-cut. BiCut gives them a reproducible reformulation pipeline with correctness certificates and novel cutting planes.
- **Applied bilevel practitioners** in energy markets, supply chain, and infrastructure planning who prototype bilevel models but are blocked by the weeks-long cycle of reformulation derivation, debugging, and solver integration. BiCut reduces this to a 20-line problem specification with verified, solver-agnostic output.
- **OR educators and students** who teach bilevel optimization conceptually but cannot assign computational exercises because the barrier to entry (manual KKT derivation, big-M calibration, solver-specific code) is prohibitive. BiCut makes bilevel optimization as accessible as `cvxpy.Problem(...).solve()`.
- **Computational optimization reviewers and editors** who need tools to verify bilevel results in submitted papers. BiCut's compiler, certificates, and solver-agnostic emission provide an independent verification pathway: reviewers can recompile a bilevel model, confirm reformulation correctness via certificates, and reproduce solutions on a different solver — all without access to the original authors' code.

BiCut provides value even when intersection cuts yield modest gap closure — the compiler, certificates, and solver-agnostic emission constitute a reproducibility layer for bilevel optimization research.

**What becomes possible:**

- *Systematic reformulation comparison:* For the first time, researchers can evaluate KKT vs. strong duality vs. value function vs. C&CG on the same instance set under identical solver configurations, with correctness certificates ensuring each comparison is valid.
- *Stronger MIBLP formulations:* Bilevel intersection cuts and value-function lifting provide root relaxation tightening unavailable in any existing tool, enabling faster convergence on hard instances where the integrality gap dominates solve time.
- *Reproducible bilevel optimization:* Compiled MILPs are standard `.mps` files; reviewers, competitors, and practitioners can verify solutions without access to BiCut or the original bilevel model.
- *Solver-agnostic performance:* A single bilevel specification compiles to Gurobi, SCIP, HiGHS, or CPLEX with backend-specific encoding optimizations (indicator constraints, SOS1 sets, lazy constraints), enabling fair cross-solver benchmarking.

**Quantified impact targets:**

| Metric | Target | Calibration |
|--------|--------|-------------|
| Root gap closure (bilevel intersection cuts) | 10–30% (with 10% as the go/no-go floor) | Comparable to GMI cuts for pure IPs, though the bilevel setting introduces structural differences (optimality-defined infeasibility, parametric dependence) that widen the expected range and limit direct analogy |
| Geometric mean speedup vs. MibS (medium instances, 50–500 vars) | 2–10× | Via tighter formulations + reformulation selection |
| Previously unsolved BOBILib instances | ≥ 10 | Instances where MibS times out at 3600s |
| Solver-agnostic success rate | ≥ 95% | Same reformulation solves on ≥4/4 backends |
| Reformulation selection beats expert-default | ≥ 5 instances with ≥2× speedup | Where strong duality dominates KKT big-M |
| Certificate catches real bugs | ≥ 10% of integer-lower-level instances | KKT applied where CQs fail |

---

## 3. Technical Difficulty

### 3.1. Subsystem Breakdown (Scoped System: ~121K LoC)

| ID | Subsystem | LoC | Novel | Risk | Hardest Subproblem |
|----|-----------|-----|-------|------|--------------------|
| S1 | Bilevel IR & Parser | 8,000 | 900 | Low | Expression canonicalization; leader/follower annotation propagation |
| S2 | Structural Analysis | 5,000 | 900 | Low | CQ verification for degenerate LPs; coupling variable classification |
| S3 | Reformulation Selection Engine | 3,000 | 1,100 | Low–Med | Cost model calibration; strategy composition rules for chained passes |
| S4 | KKT / Strong Duality Passes | 9,000 | 1,300 | **Medium** | **Automatic big-M computation via bound-tightening LPs; SOS1 vs. indicator encoding** |
| **S5** | **Intersection Cut Engine** | **11,000** | **4,000** | **HIGH** | **Bilevel-infeasible set characterization; separation oracle with >90% cache hit rate** |
| **S6** | **Value Function Oracle** | **10,000** | **3,700** | **HIGH** | **Parametric LP for exact V(x); MILP approximation via sampling; Gomory lifting** |
| S7 | Column-and-Constraint Generation | 3,500 | 300 | Low | Convergence speed; master problem growth management |
| S8 | Solver Backend Emission (×3) | 8,000 | 800 | Medium | Callback API divergence (Gurobi lazy constraints vs. SCIP constraint handlers vs. HiGHS row generation) |
| S9 | Correctness Certificates | 4,500 | 700 | Low | Bilevel feasibility verification (NP-hard per solution) |
| S10 | BOBILib Benchmark Harness | 5,500 | 400 | Low | MibS integration; fair timing; instance classification |
| S11 | Testing & Validation | 5,000 | 900 | Low | Random bilevel instance generation (ensuring well-posedness); roundtrip verification |
| S12 | Cross-Cutting Infrastructure | 8,000 | 0 | Low | Logging, configuration, CLI, Python bindings |
| E1 | QP Lower Levels | 12,000 | 4,000 | Medium | Quadratic complementarity in KKT pass (bilinear λ·g terms); McCormick envelopes for non-separable quadratics; SOCP reformulation for convex QP lower levels |
| E5 | CPLEX Backend | 4,000 | 300 | Low | CPLEX-specific indicator constraint encoding; Benders decomposition callback integration |
| | **Scoped total** | **~96,500** | **~19,300** | | |
| | **With test/bench/infra overhead** | **~121,000** | | | |

### 3.2. Full Bilevel Compiler Vision (~149K LoC)

The scoped system is designed for extension. The typed IR already represents conic and multi-follower structures; the reformulation algebra accepts new passes without modifying existing ones. The following four extensions bring the full system to ~149K LoC:

| ID | Extension Module | LoC | Novel | Risk | Key Challenge |
|----|-----------------|-----|-------|------|---------------|
| E2 | Conic Lower Levels | 10,000 | 3,500 | High | Conic duality reformulation (replacing LP strong duality with conic dual); self-concordance-based CQ verification; SDP relaxation for non-convex conic lower levels; limited solver support (only SCIP handles mixed-integer SOCP natively) |
| E3 | Pessimistic Formulations | 6,000 | 2,000 | Medium | Reformulating max-min inner problem via robust optimization (Wiesemann et al. 2013); semi-infinite constraint handling; discretization schemes with convergence guarantees |
| E4 | Multi-Follower / EPEC Games | 8,000 | 3,000 | High | Equilibrium computation among multiple followers (EPEC: equilibrium problem with equilibrium constraints); existence and uniqueness conditions; decomposition into parallel single-follower subproblems; Nash equilibrium verification |
| E6 | Advanced Regularization | 5,000 | 2,000 | Medium | Scholtes/Steffensen relaxation passes with adaptive parameter selection; convergence rate certificates (O(δ) approximation bounds); warm-starting between regularization levels |
| | **Extension total** | **~29,000** | **~10,500** | | |
| | **Full vision total** | **~149,000** | **~30,000** | | |

These extensions are architecturally supported but deferred from primary evaluation — the IR and reformulation algebra are designed to accommodate them without modifying the scoped subsystems (S1–S12, E1, E5).

### 3.3. Why Genuine Engineering Breakthroughs Are Required

BiCut cannot be assembled from known pieces. Four subproblems require novel algorithmic solutions:

1. **Intersection cut separation (S5).** The standard Balas separation procedure assumes a single-level feasible set characterized by a simplex tableau. Bilevel feasibility is defined by optimality of the lower level — a set with no closed-form description in the space of leader variables. Characterizing the bilevel-infeasible set as a finite union of polyhedra (via vertex enumeration of the lower-level value function) and deriving facet-defining conditions for the resulting intersection cuts — establishing when these cuts define facets of the bilevel-feasible set's convex hull — requires new polyhedral theory that elevates the contribution from a computational heuristic to a structural result. The separation oracle must solve auxiliary LPs; achieving >90% cache hit rates through parametric sensitivity analysis is an engineering challenge with no off-the-shelf solution.

2. **Value-function oracle performance (S6).** Evaluating the lower-level value function V(x) exactly requires solving a parametric LP for each leader decision x. For MILP lower levels, this becomes a parametric MILP — a problem with no polynomial-time algorithm in general. BiCut must implement a hybrid oracle: exact parametric LP for continuous lower levels, sampling-based MILP approximation with error bounds for integer lower levels, and Gomory–Johnson lifting to extract valid inequalities from the value-function structure. Balancing exactness, speed, and numerical stability across thousands of oracle calls per cut round is nontrivial.

3. **Sound static analysis under co-NP-hard completeness (S2, S9).** Verifying that a constraint qualification (LICQ, Slater) holds for a parametric lower-level program is co-NP-hard in general. BiCut's structural analysis must be *sound* (never certify a CQ that fails) while remaining *useful* (not rejecting most instances). This requires a conservative approximation hierarchy: syntactic checks → LP-based verification → sampling-based probabilistic certificates, with formal soundness guarantees at each level.

4. **Reformulation-aware solver emission (S8, E5).** The four target solvers (Gurobi, SCIP, HiGHS, CPLEX) expose fundamentally different APIs for the constructs BiCut generates: indicator constraints (Gurobi-native, SCIP via constraint handlers, CPLEX-native, HiGHS unsupported), lazy constraints for cut callbacks (Gurobi via cbLazy, SCIP via CONSHDLR, CPLEX via Benders callbacks, HiGHS via limited row generation), and SOS1 sets (varying performance across solvers). The emission layer must select solver-optimal encodings while preserving the semantic invariants guaranteed by correctness certificates — a nontrivial code generation problem.

---

## 4. New Mathematics Required

### Tier 1: Crown Jewels (Genuinely New Results)

**T1.1. Bilevel Intersection Cuts** — *Difficulty C (requires novel ideas)*

Extend Balas's 1971 intersection cut framework from single-level integer programming to bilevel optimization. Given a bilevel program min{F(x,y) : G(x,y) ≤ 0, y ∈ argmin{f(x,y) : g(x,y) ≤ 0}}, define the *bilevel-infeasible set* B̄ = {(x,y) : y ∉ argmin f(x,·) over {y : g(x,y) ≤ 0}} — the set of (leader, follower) pairs where the follower's response is suboptimal. For MIBLPs with LP lower levels, B̄ can be characterized as a union of polyhedra via the vertices of the lower-level optimal face. An intersection cut is derived by finding a point x̂ in the LP relaxation's interior that violates bilevel feasibility, computing the intersection of rays from x̂ with the boundary of B̄, and constructing a hyperplane separating x̂ from the bilevel-feasible set. Key results required: (i) bilevel-infeasible set polyhedrality theorem, (ii) facet-defining conditions for bilevel intersection cuts — a complete characterization of when the intersection cut defines a facet of the bilevel-feasible set's convex hull, establishing the cut's theoretical strength and elevating the contribution from a computational heuristic to new polyhedral theory, (iii) separation complexity (polynomial in the number of lower-level constraints for fixed follower dimension), (iv) finite convergence of the cut loop under non-degeneracy. The extension is nontrivial because bilevel feasibility is defined by *optimality* of a subproblem, not by integrality or simple disjunction.

**T1.2. Value-Function Lifting** — *Difficulty C (requires novel ideas)*

Extend the Gomory–Johnson theory of valid inequalities to exploit the value-function structure of bilevel programs. For a bilevel program with lower-level value function V(x) = min{f(x,y) : g(x,y) ≤ 0}, the bilevel constraint f(x,y) ≤ V(x) defines a nonconvex set in (x,y)-space. A *lifting function* π: ℝⁿ → ℝ is valid if π(x) ≥ V(x) for all feasible x and the inequality f(x,y) ≤ π(x) preserves all bilevel-feasible solutions. The Gomory–Johnson framework characterizes maximal valid lifting functions via subadditivity and symmetry conditions on the value function's epigraph. Key results required: (i) characterization of the value-function epigraph as a union of polyhedra (for LP lower levels), (ii) construction of subadditive lifting functions from the dual LP's vertex set, (iii) strength ordering of lifted cuts relative to standard value-function cuts, (iv) extension to MILP lower levels via sampling-based approximation with provable error bounds. This bridges two communities — Gomory–Johnson cutting plane theory and bilevel optimization — that have not previously interacted.

**T1.3. Compiler Soundness Theorem** — *Difficulty B (hard but precedented)*

Prove that BiCut's compilation pipeline preserves bilevel optimality: if Typecheck(P) succeeds and the structural analysis certifies preconditions Φ, then for every reformulation R selected by the selection engine, the emitted MILP M satisfies opt(M) = opt(P) and every optimal solution of M maps back to a bilevel-optimal solution of P. The proof proceeds by cases over the reformulation type: KKT soundness under LICQ + convexity, strong duality soundness under LP structure + boundedness, value-function soundness unconditionally (at the cost of computational difficulty), C&CG soundness under finite convergence conditions. The type-theoretic framing — treating reformulations as semantics-preserving program transformations with typed preconditions — is moderately novel; the closest precedent is the DCP completeness proof in CVXPY, but bilevel programs introduce leader/follower scoping and optimality-based constraints that have no single-level analogue.

### Tier 2: New Formalizations of Known Ideas

**T2.1. Structure-Dependent Reformulation Selection.** Define a function ρ mapping structural signatures σ = (convexity, CQ status, integrality, coupling type, dimension) to the set of valid reformulations with predicted performance ordering. Prove that ρ is sound (every selected reformulation is valid for the given signature) and analyze the complexity of optimal selection (conjectured NP-hard via reduction from graph coloring on reformulation compatibility graphs).

**T2.2. Compilability Decision.** Given a bilevel program P and a target solver capability profile S, decide in polynomial time whether there exists a valid reformulation R such that emit(R, S) produces a well-formed solver input. Prove decidability and characterize the boundary of compilability (e.g., MILP lower levels with unbounded feasible regions are not compilable via KKT or strong duality but are compilable via value-function reformulation).

**T2.3. Compositional Error Bounds.** For approximate reformulations (regularization, penalty methods, cut-loop truncation), prove that composing k approximate passes with individual error εᵢ yields total error bounded by Π(1 + κᵢεᵢ) − 1 where κᵢ is the condition number of the i-th reformulation's inverse map. Establish conditions under which the bound is tight and identify problem structures where composition is numerically stable.

### Tier 3: Known Results Applied

Standard results deployed as compiler passes with verified preconditions:
- **KKT exactness:** Convex lower level + LICQ/MFCQ ⟹ KKT conditions are necessary and sufficient (Dempe 2002)
- **Strong duality reformulation:** LP lower level + boundedness ⟹ primal-dual pairing eliminates complementarity (Fortuny-Amat & McCarl 1981)
- **Value-function equivalence:** Always valid; V(x) computable via parametric LP for continuous lower levels (Outrata, Kočvara & Zowe 1998)
- **Integer lower-level theory:** Bounded integer lower levels admit reformulation via vertex enumeration; unbounded case requires disjunctive programming (DeNegre 2011, Xu & Wang 2014)
- **Regularization convergence:** Tikhonov regularization yields O(δ)-approximate bilevel solutions (Dempe & Dutta 2012)
- **Big-M validity:** Bound-tightening via auxiliary LP gives smallest valid big-M values (Pineda & Morales 2019)

---

## 5. Best Paper Argument

**BiCut merits best-paper consideration at a top OR/optimization venue (Mathematical Programming, Operations Research, INFORMS JOC) for six reasons:**

1. **Creates a new software category.** Disciplined convex programming (CVXPY/CVX) revolutionized convex optimization by interposing a compiler between problem specification and solver. BiCut does the same for bilevel optimization — a problem class that is harder (Σ₂ᵖ-complete in general), less standardized, and more error-prone. The typed IR, structural analysis, and reformulation selection engine constitute a fundamentally new abstraction for bilevel programming.

2. **Bilevel intersection cuts are a genuine mathematical contribution.** Extending Balas's 1971 framework to optimality-defined infeasible sets bridges the cutting plane theory community (IPCO, integer programming) and the bilevel optimization community (MPB, bilevel-specific venues). The facet-defining conditions and separation complexity analysis are new polyhedral results, not routine applications of known theory.

3. **Clean, falsifiable empirical story.** BOBILib provides 2600+ standardized MIBLP instances with known optimal values. MibS is the established baseline. The primary metrics — root gap closure, solve time, node count — are standard in the integer programming literature. The experimental design admits no ambiguity: either BiCut's cuts close 10–30% of root gap on hard instances, or they do not.

4. **Correctness certificates are a first.** No existing bilevel optimization tool provides machine-checkable proofs that a reformulation preserves bilevel optimality. BiCut's certificates — verified conjunctions of structural preconditions — are a qualitative advance in bilevel software reliability. The ability to catch real bugs (KKT applied where CQs fail on integer lower levels) is empirically testable on BOBILib.

5. **Substantial, non-trivial artifact.** At ~121K total LoC with ~19K lines of genuinely novel algorithmic logic (intersection cut separation, value-function lifting, parametric oracle, sound structural analysis, QP lower-level handling), BiCut is not a prototype or proof-of-concept. The novel logic alone exceeds the total codebase size of most optimization research software.

6. **Immediately usable outputs.** Compiled MILPs are standard `.mps`/`.lp` files solvable by any MIP solver. Reviewers can verify results without installing BiCut. Practitioners can distribute compiled formulations to collaborators using different solvers. This "compile once, solve anywhere" property is unique among bilevel optimization tools.

---

## 6. Evaluation Plan

All evaluation is fully automated with zero human involvement at runtime.

### 6.0. Prototype Validation Gate

Before full implementation, implement the intersection cut separation oracle on ≥50 BOBILib instances with LP lower levels. Measure root gap closure using only the separation oracle and a basic LP relaxation — no full compiler pipeline required. If geometric mean gap closure is <5%, descope the cutting-plane contribution and reposition BiCut as a pure compiler/verification tool (the compiler, certificates, and solver-agnostic emission still constitute a significant contribution as a reproducibility layer; see §2). This gate must be passed before proceeding to full implementation of the intersection cut engine (S5) and value-function oracle (S6). The prototype gate is expected to require ≤2 weeks of focused implementation effort and provides an early signal on the viability of the cutting-plane contribution.

### 6.1. Primary: MIBLP Benchmark (BOBILib vs. MibS)

- **Dataset:** BOBILib (2600+ instances), filtered to MIBLP class, stratified by size (small: <50 vars, medium: 50–500 vars, large: >500 vars) and structure (LP lower level, MILP lower level, pure integer).
- **Baselines:** MibS (direct bilevel branch-and-cut), BiCut-KKT (KKT reformulation only), BiCut-SD (strong duality only), BiCut-VF (value function only), BiCut-Full (automatic selection + intersection cuts).
- **Metrics:**
  - *Root gap closure:* (z_LP − z_LP+cuts) / (z_LP − z*) as percentage; target 10–30% geometric mean on medium instances (with 10% as the go/no-go floor; see §6.5).
  - *Solve time:* Wall-clock seconds to optimality or 3600s timeout; report geometric mean speedup of BiCut-Full vs. MibS.
  - *Node count:* Branch-and-bound nodes to optimality; measures formulation tightness independent of solver implementation.
  - *Solver agnosticism:* Fraction of instances where BiCut-Full achieves optimality on all four backends (Gurobi, SCIP, HiGHS, CPLEX) within 2× of the fastest.
  - *Previously unsolved:* Count of instances where MibS times out at 3600s but BiCut-Full solves to optimality.

### 6.2. Secondary: Reformulation Selection & Certificates

- **Selection evaluation:** On instances admitting multiple valid reformulations, compare BiCut's automatic selection vs. default-KKT and vs. expert-chosen reformulation (strong duality for LP lower levels, value function for integer lower levels). Report instances where selection achieves ≥2× speedup over default.
- **Certificate evaluation:** Apply KKT reformulation to all integer-lower-level instances in BOBILib; report the fraction where BiCut's certificate system correctly rejects the reformulation (CQ violation). Verify by constructing bilevel-feasible solutions that are infeasible in the KKT MILP. Construct 5 synthetic CQ-violation test fixtures during development for targeted testing.

### 6.3. Tertiary: Strategic Bidding Case Study

- **Setup:** Single strategic generator bidding into an IEEE 14-bus market with LP-relaxed market clearing (DC optimal power flow). Bilevel model: leader maximizes profit by choosing offer prices; follower (ISO) minimizes dispatch cost.
- **Purpose:** Demonstrate BiCut's applicability beyond pure benchmarks. Not a full energy market study — simplified to a scale solvable on laptop CPU in minutes.
- **Metrics:** Reformulation comparison (KKT vs. strong duality vs. value function); compilation time as fraction of total solve time; solution interpretability (bid curves, market prices).

### 6.4. Infrastructure

- **Automated runner:** Script orchestrating compilation → cut generation → MILP solve → result collection across all instances, backends, and configurations. Parallelizable across instances.
- **Bilevel feasibility verification:** For every solution reported optimal, verify bilevel feasibility by re-solving the lower level at the leader's solution and checking optimality tolerance (|f(x*,y*) − V(x*)| < ε).
- **Statistical reporting:** Geometric means with shifted ratios (Achterberg shift = 10s), performance profiles (Dolan–Moré), Wilcoxon signed-rank tests for pairwise comparisons.

### 6.5. Go/No-Go Thresholds

Explicit thresholds that determine whether the cutting-plane contribution is viable as a research result:

| Threshold | Criterion | Scope |
|-----------|-----------|-------|
| Gap closure | ≥10% geometric mean | On ≥30% of LP-lower-level BOBILib instances |
| Separation oracle overhead | <50ms | On 90% of callbacks |
| Cache hit rate | ≥80% | Across cut rounds |

If any threshold is missed after full implementation, the pivot strategy is: fall back to positioning BiCut as a compiler/verification tool without novel cuts. The compiler, certificates, reformulation selection engine, and solver-agnostic emission remain a significant contribution even without the cutting-plane component (see §2, reproducibility layer framing). Document the negative cutting-plane result as an empirical finding — failed cut strategies are informative to the bilevel optimization community.

---

## 7. Laptop CPU Feasibility

BiCut is designed for standard academic hardware — a laptop with a multi-core CPU, 16–32 GB RAM, no GPU.

| Phase | Computational Character | Expected Time |
|-------|------------------------|---------------|
| **Parsing & IR construction** | String processing, AST construction | < 100ms per instance |
| **Structural analysis** | LP feasibility checks, convexity verification, CQ testing | < 1s per instance (auxiliary LPs are tiny) |
| **Reformulation selection** | Table lookup + cost model evaluation | < 10ms per instance |
| **Reformulation lowering** | Symbolic transformation (constraint rewriting, variable introduction) | < 500ms per instance |
| **Intersection cut generation** | Auxiliary LP solves for separation oracle; cached across rounds | < 10ms per cut with >90% cache hit; ~1–5 min per instance for full cut loop |
| **Value-function oracle** | Parametric LP solves; MILP approximation for integer lower levels | 1–60s per oracle call; amortized via caching |
| **Solver emission** | Code generation to `.mps`/`.lp` format | < 200ms per instance |
| **MILP solving** | CPU-native branch-and-bound (Gurobi/SCIP/HiGHS/CPLEX) | Seconds to minutes for BOBILib instances ≤ 500 vars; hours for large instances |

**Key feasibility arguments:**

- Compilation (parse → analyze → select → reformulate → emit) is entirely symbolic — no numerical optimization until solver invocation. Total compilation time is milliseconds to low seconds per instance.
- Cut generation requires solving auxiliary LPs, but these are small (lower-level dimension, typically < 100 constraints) and heavily cacheable. The parametric sensitivity structure means consecutive oracle calls differ by small perturbations, enabling warm-starting.
- Downstream MILP solving is CPU-native and is actually *helped* by BiCut's tighter formulations — stronger cuts mean fewer branch-and-bound nodes, reducing total computation.
- BOBILib instances range from trivial (< 10 vars) to challenging (> 1000 vars). Instances up to ~500 variables are solvable in minutes on a modern laptop. Larger instances may require hours but remain CPU-feasible.
- No GPU computation is required at any stage. No human involvement is required at any stage.

---

## 8. Scope and Non-Goals

**In scope (scoped system, ~121K LoC):**
- Optimistic bilevel linear programs with continuous, integer, or mixed-integer lower levels
- QP lower levels (KKT extension to quadratic complementarity, McCormick envelopes, SOCP reformulation)
- Reformulation strategies: KKT/MPEC, strong duality, value function, column-and-constraint generation
- Novel cutting planes: bilevel intersection cuts, value-function lifting
- Four solver backends: Gurobi (commercial), SCIP (academic), HiGHS (open-source), CPLEX (commercial)
- Correctness certificates for all reformulations
- BOBILib benchmark evaluation + simplified strategic bidding case study

**Explicit non-goals (deferred to future work / full ~149K LoC vision):**
- Pessimistic bilevel formulations
- Conic or NLP lower levels (IR supports them; evaluation does not)
- Multi-follower / EPEC / stochastic bilevel
- Neural network verification / adversarial robustness
- Full-fidelity energy market modeling (PTDF, N-1 contingencies, unit commitment)
- GPU-accelerated computation of any kind

**Known limitations (stated honestly):**
- Structural analysis for CQ verification is co-NP-hard in general; BiCut uses conservative approximations that may reject valid instances.
- Bilevel intersection cuts have a *narrow viability corridor:* they require that the bilevel-infeasible set admits an efficient polyhedral characterization, which holds for LP lower levels but degrades for large MILP lower levels.
- Value-function oracle scalability is limited to ~20 upper-level variables for exact parametric LP evaluation; beyond this, sampling-based approximation introduces controlled but nonzero error.
- Big-M computation via bound-tightening may produce large constants (> 10⁴) on poorly scaled instances, degrading LP relaxation quality despite tighter formulations.
- The reformulation selection cost model is calibrated on BOBILib instance features; its predictive accuracy on out-of-distribution problems (non-standard structure, extreme dimensions) is unknown.

---

## 9. Summary

BiCut is a bilevel optimization compiler that transforms a bilevel program specification into a solver-ready MILP with formal correctness guarantees — the first system to provide automatic reformulation selection, machine-checkable correctness certificates, and novel bilevel intersection cuts for MIBLPs and QP lower levels. The mathematical contributions (bilevel intersection cuts extending Balas 1971 with facet-defining conditions, value-function lifting extending Gomory–Johnson, compiler soundness theorem) are genuinely new results at difficulty levels B–C. The engineering artifact (~121K LoC, ~19K novel algorithmic logic) is substantial and immediately usable. The evaluation plan — including an early prototype validation gate with explicit go/no-go thresholds (BOBILib vs. MibS, four solver backends, automated bilevel feasibility verification) — is clean, falsifiable, and fully automated on laptop CPU. BiCut creates a new software category — disciplined bilevel programming — and serves as both a research tool and a reproducibility layer for the bilevel optimization community.
