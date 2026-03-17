# Prior Art Audit: A Solver-Agnostic Bilevel Optimization Compiler

**Audit Date:** 2025-07-18
**Auditor Role:** Prior Art Auditor (Crystallization Stage)
**Target Community:** Operations Research and Optimization

---

## Executive Summary

The proposed concept is a **compiler** that takes high-level bilevel optimization specifications and **automatically selects and generates the best reformulation strategy** for any given solver backend. After exhaustive survey, the novelty gap is **moderate-to-wide**: while many bilevel optimization *tools* exist, **no existing system performs automatic, structure-aware reformulation selection with correctness guarantees across heterogeneous solver backends.** Existing tools either hard-wire a single reformulation (usually KKT/MPEC), target a single solver ecosystem, or require the user to manually choose the reformulation strategy. The "compiler" framing—reasoning about reformulation equivalence, emitting solver-specific code, and providing formal guarantees—is genuinely novel. However, the gap narrows significantly when considering that BilevelJuMP.jl's MixedMode and PAO's automatic model transformation already perform *partial* reformulation selection, and the BiOpt toolbox offers three switchable reformulation strategies. The key differentiator must be: *principled, automated reformulation selection with provable correctness across solver backends*, not merely "yet another bilevel modeling tool."

---

## 1. Existing Bilevel Optimization Frameworks & Tools

### 1.1 BilevelJuMP.jl (Julia/JuMP Ecosystem)

**What it does:**
- Models bilevel problems using JuMP's algebraic syntax
- Reformulates lower-level via KKT conditions into an MPEC
- Supports MixedMode MPEC: selects the best reformulation *per constraint* (big-M, SOS1, or product complementarity)
- Upper level: NLP, MIP, conic, QP. Lower level: conic, QP, dual variables
- Interfaces with CPLEX, Gurobi, Ipopt, etc. via JuMP's solver abstraction
- Published: INFORMS Journal on Computing (2023), Dias Garcia et al.

**What it cannot do:**
- No automatic *strategy-level* selection (always KKT→MPEC; never value function or CCG)
- Integer variables in the lower level are problematic (KKT conditions invalid for non-convex lower levels)
- No formal correctness certificates for the reformulation
- No reasoning about when KKT reformulation is valid vs. when an alternative is needed
- Performance degrades at scale (~1000+ variables)
- Single ecosystem (Julia/JuMP only)

**Critical gap relative to proposal:** BilevelJuMP's MixedMode is the *closest* existing work to automatic reformulation selection, but it operates *within* a single strategy family (MPEC variants). It never considers value-function, CCG, or branch-and-bound alternatives.

### 1.2 PAO (Pyomo Adversarial Optimization, Python)

**What it does:**
- Extends Pyomo with `SubModel` for bilevel/multilevel modeling
- Algebraic and compact (matrix) representations
- Automatic conversion from Pyomo models to compact formats
- Bundled solvers: FA (Fortuny-Amat big-M), PCCG (Projected Column-Constraint Generation), REG (Regularization)
- Can interface with GLPK, CBC, MibS
- Supports min/max objectives, equalities, inequalities, nested SubModels for k-level problems

**What it cannot do:**
- No automatic selection among FA, PCCG, and REG—user must choose
- Limited to linear bilevel problems for most solvers
- No formal correctness guarantees for reformulations
- No solver-agnostic code generation (tightly coupled to Pyomo/COIN-OR)
- Nonlinear bilevel support is minimal

**Critical gap:** PAO *has* multiple reformulation strategies but no automatic selection logic. The user must know which method to use.

### 1.3 GAMS EMP (Extended Mathematical Programming)

**What it does:**
- Bilevel problems specified via EMP annotations in an info file
- JAMS solver automatically reformulates to MPEC via KKT conditions
- Routes to appropriate subsolver (CPLEX, CONOPT, PATH, etc.)
- Rich model library (emplib) with bilevel examples
- Supports large-scale models through mature solver backends

**What it cannot do:**
- Always uses KKT→MPEC reformulation (no alternative strategies)
- Lower level must be convex for correctness (returns stationary points otherwise)
- No formal verification that reformulation preserves optimality
- Commercial/proprietary ecosystem
- No reasoning about reformulation applicability conditions

**Critical gap:** GAMS EMP is the closest to a "compiler" in spirit (specification → automatic reformulation → solver dispatch), but it is locked to a single reformulation strategy and provides no correctness guarantees when assumptions are violated.

### 1.4 YALMIP (MATLAB)

**What it does:**
- `solvebilevel(UpConstr, UpObj, LoConstr, LoObj, LoVar)` — high-level bilevel interface
- Default: branch on complementarity of KKT duals/slacks (avoids big-M)
- Alternative: external KKT reformulation with big-M via `sdpsettings`
- `kkt()` function for manual KKT derivation
- Integrates with Gurobi, CPLEX

**What it cannot do:**
- Small-to-moderate problems only
- Lower level must be convex QP
- Only two algorithmic options (internal branching vs. external big-M), no value-function or CCG
- MATLAB-only ecosystem
- No formal guarantees, no automatic strategy selection based on problem analysis

### 1.5 MibS (Mixed Integer Bilevel Solver, COIN-OR)

**What it does:**
- Purpose-built branch-and-cut for mixed-integer bilevel linear programs (MIBLPs)
- Both upper and lower levels can have integer variables
- Specialized cutting planes and valid inequalities for bilevel structure
- Open-source (C++, Eclipse Public License)
- The only mature solver that natively handles integer lower levels in a principled way

**What it cannot do:**
- Linear constraints and objectives only (no nonlinear/NLP)
- No multilevel support
- Barebones interface; not a modeling language
- No reformulation—it is a direct solver
- Limited scalability for large instances
- No interoperability with other solver backends

### 1.6 BiOpt Toolbox (MATLAB)

**What it does:**
- Three reformulation-based solvers:
  - **SNLLVF**: Semi-smooth Newton on lower-level value function
  - **SNQVI**: Semi-smooth Newton on quasi-variational inequality reformulation
  - **SNKKT**: Semi-smooth Newton on KKT reformulation
- BOLIB test library (173 problems)
- Derivative computation utilities (1st, 2nd, 3rd order)
- Value function plotting

**What it cannot do:**
- User must manually select among SNLLVF, SNQVI, SNKKT
- No automatic problem analysis to determine which reformulation is appropriate
- MATLAB-only; no integration with commercial MIP/NLP solvers
- Continuous problems only (no integer variables)
- No formal correctness certificates

**Critical gap:** BiOpt is the *closest existing work* to the proposed compiler in terms of supporting multiple reformulation paradigms (value function, QVI, KKT). However, it completely lacks automatic selection logic and solver-agnostic code generation.

### 1.7 BASBL (Branch-And-Sandwich BiLevel Solver)

**What it does:**
- Deterministic global optimization for nonconvex bilevel problems
- Single branch-and-bound tree exploring both upper and lower simultaneously
- Guarantees globally optimal solutions (or ε-globally optimal)
- Handles nonconvex, nonlinear lower levels—unique among solvers
- Built on MINOTAUR framework; benchmarked on BASBLib test set

**What it cannot do:**
- Requires twice continuously differentiable functions
- Computationally expensive; limited scalability
- No reformulation component—direct solver approach
- No integration with standard modeling languages

### 1.8 BOAT (Bi-Level Optimization Algorithm Toolbox)

**What it does:**
- Python library for differentiable bilevel optimization in ML contexts
- Modular: composable dynamic and hyper-gradient operations
- Supports PyTorch, Jittor, MindSpore backends
- Task-agnostic design for meta-learning, hyperparameter optimization

**What it cannot do:**
- ML-focused (gradient-based); not for classical OR bilevel problems
- No support for integer variables, linear programs, or combinatorial bilevel problems
- No reformulation in the classical sense (KKT, value function, etc.)

### 1.9 Other Tools & Libraries

| Tool | Type | Key Feature | Key Limitation |
|------|------|-------------|----------------|
| **AMPL + MPEC** | Modeling language | KKT/MPEC reformulation via `complementarity` | Single strategy, manual |
| **TorchOpt** (Meta) | Library | Differentiable bilevel in PyTorch | ML-only, no OR problems |
| **Theseus** (Meta) | Library | Differentiable nonlinear optimization | Robotics focus |
| **Plasmo.jl** | Julia | Graph-based decomposition modeling | Not bilevel-specific |
| **BOBILib** | Benchmark | 2,600+ MIBLP test instances | Library, not a solver |
| **BOLIB** | Benchmark | 173 continuous bilevel test problems | Library, not a solver |
| **BASBLib** | Benchmark | Nonconvex bilevel test problems | Library, not a solver |
| **near** | Solver | Bilevel-specific heuristics | Limited documentation, niche |

---

## 2. Key Reformulation Techniques in Literature

### 2.1 KKT-Based Reformulation
- **Idea:** Replace the lower-level optimization with its KKT conditions, yielding an MPEC.
- **When valid:** Lower level is convex, constraint qualification holds (e.g., LICQ, MFCQ).
- **Limitations:** (a) Invalid when lower level has integer variables; (b) complementarity constraints are inherently non-convex; (c) may yield stationary points that are not bilevel-optimal; (d) big-M reformulations of complementarity introduce numerical issues.
- **Used by:** BilevelJuMP, GAMS EMP, YALMIP, AMPL, BiOpt (SNKKT).

### 2.2 Strong Duality Reformulation
- **Idea:** Replace lower level with primal-dual feasibility + strong duality equality (primal obj = dual obj).
- **When valid:** Lower level is a convex program satisfying Slater's condition.
- **Limitations:** Introduces bilinear terms (primal × dual variables); requires lower-level convexity; may be weaker than KKT in some cases but avoids complementarity.
- **Used by:** Some manual formulations in GAMS; not automated in any existing tool.

### 2.3 Value Function Reformulation
- **Idea:** Add constraint that lower-level objective ≤ optimal value function, eliminating the lower-level optimization.
- **When valid:** General; does not require convexity of lower level.
- **Limitations:** Value function may be non-smooth, non-convex, and hard to evaluate; requires lower-level parametric optimization.
- **Key paper:** Lozano & Smith (2017) — exact finite algorithm for BMIP via value function.
- **Used by:** BiOpt (SNLLVF), but not automatically selected.

### 2.4 Regularization Approaches
- **Scholtes (2001):** Relax complementarity G(x)ᵀH(x) = 0 to G(x)ᵀH(x) ≤ t, t → 0. Converges to C-stationary points. Simple and robust but weak stationarity guarantee.
- **Steffensen & Ulbrich (2010):** Selective relaxation that achieves M-stationarity or strong stationarity under suitable conditions. More nuanced but more complex.
- **Kanzow & Schwartz:** Further regularization methods with convergence analysis.
- **Limitations:** All produce approximate solutions with varying stationarity guarantees; no tool automatically selects the appropriate regularization.

### 2.5 Branch-and-Bound / Branch-and-Cut
- **Idea:** Directly solve the bilevel problem via spatial branching, leveraging bilevel-specific cuts and bounding.
- **Key implementations:** MibS (linear), BASBL (nonconvex), Fischetti et al. (2017) solver.
- **Limitations:** Computationally expensive; does not produce a reformulation (direct solve).

### 2.6 Column-and-Constraint Generation (CCG)
- **Idea:** Iteratively generate variables and constraints by solving master and subproblems; particularly effective for robust and two-stage settings.
- **Key paper:** Zeng & An (2014).
- **Limitations:** Struggles with mixed-integer second-stage; convergence can be slow for non-linear problems.
- **Used by:** PAO (PCCG variant), but user must manually select it.

### 2.7 Penalty-Based Methods
- **Idea:** Penalize lower-level optimality gap in the upper-level objective.
- **Limitations:** Penalty parameter tuning; approximate solutions; no formal equivalence.

### 2.8 Summary: Reformulation Selection Landscape

| Technique | Lower-Level Convexity Required? | Integer Lower Level? | Formal Equivalence? | Automated in Any Tool? |
|-----------|-------------------------------|---------------------|---------------------|----------------------|
| KKT/MPEC | Yes | No | Conditional | Partially (GAMS, BilevelJuMP) |
| Strong Duality | Yes (Slater's) | No | Conditional | No |
| Value Function | No | Yes (in theory) | Yes (exact) | No |
| Regularization | No | No | Approximate | No |
| Branch-and-Cut | No | Yes (linear) | N/A (direct) | N/A |
| CCG | No | Limited | Conditional | No |
| Penalty | No | No | Approximate | No |

**No existing tool automatically reasons about which row of this table to use for a given problem.**

---

## 3. Key Academic Papers

### 3.1 Foundational Monographs

- **Dempe (2002):** *Foundations of Bilevel Programming.* Kluwer. The original comprehensive treatment of bilevel optimization theory, covering optimality conditions, solution algorithms, and connections to game theory. Established the mathematical foundations that all subsequent work builds on.

- **Dempe & Zemkoho (2020):** *Bilevel Optimization: Advances and Next Challenges.* Springer. Updated survey covering: optimistic/pessimistic formulations, MPEC approaches, metaheuristics, applications in energy/ML, BOLIB test library. Identifies key open problems including algorithmic scalability and robust formulations.

- **Bard (1998):** *Practical Bilevel Programming.* Kluwer. Applications-oriented treatment covering Stackelberg games, network design, and algorithmic approaches (vertex enumeration, penalty function, descent methods).

### 3.2 Computational Advances

- **Fischetti, Ljubić, Monaci & Sinnl (2017):** "A New General-Purpose Algorithm for Mixed-Integer Bilevel Linear Programs." *Operations Research* 65(6), 1615–1637. Branch-and-cut with new intersection cuts and bilevel-specific preprocessing. Solved previously unsolved benchmarks. The most complete general-purpose MIBLP solver algorithm.

- **Kleinert, Labbé, Ljubić & Schmidt (2021):** "A Survey on Mixed-Integer Programming Techniques in Bilevel Optimization." *EURO J. Computational Optimization* 9, 100007. Comprehensive survey establishing: (a) bilevel linear programs are NP-hard; (b) mixed-integer bilevel is ΣP₂-complete; (c) taxonomy of solution techniques; (d) open problems in scalability and formulation tightness.

- **Lozano & Smith (2017):** "A Value-Function-Based Exact Approach for the Bilevel Mixed-Integer Programming Problem." *Operations Research* 65(3), 768–786. First exact finite algorithm for general BMIP using the value function approach. Handles integer variables in both levels. Outperforms prior methods on benchmarks.

- **Zeng & An (2014):** "Solving Two-Stage Robust Optimization Problems Using a Column-and-Constraint Generation Method." *Operations Research Letters* 41(5), 457–461. CCG algorithm for robust/bilevel settings. Efficient when second-stage is continuous; struggles with mixed-integer recourse.

### 3.3 MPEC & Regularization Theory

- **Scholtes (2001):** "Convergence Properties of a Regularization Scheme for MPCCs." *SIAM J. Optimization* 11(4), 918–936. The foundational regularization for complementarity; guarantees C-stationarity.

- **Steffensen & Ulbrich (2010):** "A New Relaxation Scheme for Mathematical Programs with Equilibrium Constraints." *SIAM J. Optimization* 20(5), 2504–2539. Achieves M-stationarity/strong stationarity under suitable conditions.

- **Ralph & Wright (2004):** "Some Properties of Regularization and Penalization Schemes for MPECs." *Optimization Methods and Software* 19(5), 527–556.

### 3.4 Differentiable Bilevel (ML Community)

- **Petrulionytė et al. (NeurIPS 2024):** "Functional Bilevel Optimization for Machine Learning." Shift from parametric to functional view; new algorithms (FuncID) with convergence guarantees for non-convex neural networks.

- **Ji et al. (ICML 2021):** "Bilevel Optimization: Convergence Analysis and Enhanced Design." Improved complexity results for stochastic bilevel via hypergradient estimation.

- **Implicit Bilevel Optimization (BIGRAD, 2023):** End-to-end differentiable bilevel layers for deep learning.

- **Tools:** TorchOpt (Meta), Theseus (Meta), betty (ML bilevel library).

**Note:** The ML differentiable bilevel community is largely disjoint from the OR bilevel community. They share the mathematical structure but differ fundamentally in solution methodology (gradient-based vs. reformulation-based). The proposed compiler targets the OR reformulation paradigm, not the ML gradient paradigm.

### 3.5 Modeling Language & Compiler Theory

- **Dias Garcia, Guennebaud & Leclère (2023):** "BilevelJuMP.jl: Modeling and Solving Bilevel Optimization in Julia." *INFORMS J. on Computing.* The closest to a bilevel "compiler" in the modeling-language sense.

- **Fourer, Gay & Kernighan (2003):** *AMPL: A Modeling Language for Mathematical Programming.* Established the paradigm of algebraic modeling language → solver-independent formulation.

- **No paper exists on bilevel reformulation as a formal compiler problem** with intermediate representations, optimization passes, and correctness-preserving transformations. This is the core novelty gap.

---

## 4. Genuine Novelty Gap Analysis

### 4.1 Does any tool do automatic reformulation selection?

**No.** This is the single clearest novelty gap.

- BilevelJuMP's MixedMode selects among MPEC *variants* (big-M, SOS1, product) per constraint—but this is within a single strategy family (KKT→MPEC), not across fundamentally different approaches.
- PAO offers FA, PCCG, and REG solvers but requires the user to choose.
- BiOpt offers SNLLVF, SNQVI, and SNKKT but requires manual selection.
- GAMS EMP always uses KKT→MPEC via JAMS.
- No tool analyzes the problem structure (convexity of lower level, presence of integer variables, constraint qualifications) to automatically select from {KKT, value function, CCG, regularization, direct solve}.

### 4.2 Does any tool handle mixed-integer lower levels universally?

**No.** This is a significant gap.

- MibS handles integer lower levels but only for linear problems, and is a direct solver (not a reformulation tool).
- BilevelJuMP applies KKT reformulation even when the lower level has integers—which is theoretically invalid and may produce incorrect results.
- Lozano & Smith's value function approach handles integers in both levels but is an algorithm paper, not a tool/framework.
- BASBL handles nonconvex lower levels (continuous) but not integer.
- **No existing tool detects the presence of integer lower-level variables and automatically routes to an appropriate strategy (value function, branch-and-cut, or decomposition).**

### 4.3 Does any tool provide correctness guarantees for reformulations?

**No.** This is perhaps the deepest novelty gap.

- All existing tools apply reformulations *silently*—they do not verify or certify that the conditions under which the reformulation is valid actually hold for the user's problem.
- BilevelJuMP does not check lower-level convexity before applying KKT.
- GAMS EMP does not verify Slater's condition or LICQ.
- BiOpt does not verify that the chosen reformulation is appropriate.
- **No tool emits a formal proof or certificate that the reformulated single-level problem is equivalent to the original bilevel problem.**
- The concept of a "reformulation correctness certificate"—analogous to solver optimality certificates or proof logging in SAT/MIP—does not exist in the bilevel optimization literature.

### 4.4 Is there a true "compiler" that reasons about reformulation equivalence?

**No.** The compiler metaphor is genuinely novel in this domain.

- Existing tools are **translators** (one fixed reformulation) or **libraries** (multiple reformulations, user-selected).
- A compiler would: (a) parse the bilevel specification; (b) analyze problem structure via an intermediate representation; (c) determine valid reformulation strategies; (d) select the best strategy for the target solver; (e) emit solver-specific code; (f) provide correctness guarantees.
- This compiler pipeline does not exist. The closest analog is GAMS EMP's JAMS, which does (a), (c) partially, and (e), but hard-wires step (c) to "always KKT" and omits (f).

### 4.5 Where do existing tools fail?

| Failure Mode | Which Tools | Proposed Compiler Addresses? |
|-------------|-------------|------------------------------|
| KKT applied to non-convex lower level → wrong answer | BilevelJuMP, GAMS EMP, YALMIP | Yes (structure analysis) |
| Integer lower level → KKT invalid | BilevelJuMP, GAMS EMP, YALMIP | Yes (route to value function/B&C) |
| User must choose reformulation → wrong choice | PAO, BiOpt | Yes (automatic selection) |
| Single solver ecosystem → can't use best solver | All tools | Yes (solver-agnostic emission) |
| No correctness certificate → silent failures | All tools | Yes (reformulation certificates) |
| No intermediate representation → no optimization passes | All tools | Yes (compiler IR) |

### 4.6 Novelty Gap Width Assessment

| Dimension | Gap Width | Evidence |
|-----------|-----------|----------|
| Automatic reformulation selection across strategy families | **WIDE** | No tool does this |
| Correctness certificates for reformulations | **WIDE** | No tool or paper addresses this |
| Compiler IR for bilevel problems | **WIDE** | No precedent in bilevel literature |
| Solver-agnostic code generation | **MODERATE** | JuMP/Pyomo are partially solver-agnostic already, but not reformulation-aware |
| Multiple reformulation strategies in one tool | **NARROW** | BiOpt has 3, PAO has 3, but no auto-selection |
| Handling mixed-integer lower levels | **MODERATE** | MibS exists for linear; value function theory exists; no unified tool |

**Overall Assessment: The novelty gap is WIDE on the most important dimensions (automatic selection, correctness, compiler framing) and NARROW-TO-MODERATE on secondary dimensions (multi-strategy, solver-agnostic).**

---

## 5. Portfolio Differentiation

### 5.1 Sibling Project: mip-auto-reformulate (area-013)

**Description:** "A solver-agnostic engine that automatically mines MIP constraint-matrix substructure via hypergraph decomposition, applies provably-tight extended-formulation reformulations (knapsack, set-packing, network-flow, symmetry groups) for each detected motif, and certifies integrality-gap improvement."

**Overlap Risk: MODERATE — requires explicit differentiation.**

| Dimension | mip-auto-reformulate | Bilevel Compiler (this project) |
|-----------|---------------------|-------------------------------|
| Problem class | Single-level MIP | Bilevel optimization |
| Reformulation target | Tighter LP relaxation | Single-level equivalent |
| Detection logic | Constraint-matrix substructure | Bilevel problem structure (convexity, integrality, CQ) |
| Correctness concept | Integrality-gap certificates | Reformulation equivalence certificates |
| Solver role | Solve strengthened MIP | Solve reformulated single-level |

**Verdict:** Complementary, not overlapping. mip-auto-reformulate strengthens single-level MIPs. This project transforms bilevel problems into single-level problems. They could even compose: bilevel compiler emits a single-level MIP, which mip-auto-reformulate then strengthens.

### 5.2 Related Portfolio Projects

| Project | Area | Overlap? | Differentiation |
|---------|------|----------|-----------------|
| **mip-decomp-compiler** (area-033, 073) | MIP decomposition | LOW | Decomposes single-level MIPs; does not handle bilevel structure |
| **robust-counterpart-compiler** (area-033) | Robust optimization | LOW | Robust ≠ bilevel; different reformulation theory |
| **robust-opt-compiler** (area-073) | Robust optimization | LOW | Same as above |
| **mip-reformulation-compiler** (area-093) | MIP reformulation | MODERATE | Single-level MIP reformulation; shares the "compiler" metaphor but different problem class |
| **solver-portfolio-engine** (area-033) | Solver selection | LOW-MODERATE | Selects among solvers, not among reformulations; complementary |

### 5.3 Differentiation from ML Bilevel Tools

The ML community's bilevel tools (TorchOpt, BOAT, betty, Theseus) are entirely gradient-based and target neural network training. The proposed compiler targets classical OR bilevel problems with reformulation-based approaches. There is zero technical overlap despite shared mathematical terminology.

---

## 6. Risks and Honest Assessment

### 6.1 Risk: Incremental Over BiOpt + BilevelJuMP

**Concern:** If BiOpt added automatic selection among its three strategies, and BilevelJuMP extended MixedMode to strategy-level selection, the novelty gap would narrow substantially.

**Mitigation:** The compiler framing (IR, correctness certificates, solver-agnostic emission) is architecturally distinct from extending existing tools. The correctness guarantee dimension has no precedent even in principle.

### 6.2 Risk: Theoretical vs. Practical

**Concern:** Automatic reformulation selection requires solving meta-optimization problems (which reformulation is "best"?) that may themselves be hard.

**Mitigation:** The selection logic can be rule-based for many cases (e.g., "if lower level is LP → use strong duality; if lower level has integers → use value function approach; if lower level is convex QP → use KKT"). Perfect selection is not required; outperforming manual selection is sufficient.

### 6.3 Risk: Overlap with mip-reformulation-compiler

**Concern:** Both projects are "reformulation compilers" in OR.

**Mitigation:** Completely different problem classes. MIP reformulation operates on single-level MIPs; bilevel compilation transforms hierarchical problems into single-level ones. The reformulation theory is entirely disjoint (extended formulations vs. KKT/value function/CCG).

---

## 7. Recommended Novelty Claims (Ranked by Strength)

1. **STRONGEST:** First system providing formal correctness certificates for bilevel reformulations—proving that the emitted single-level problem is equivalent to the input bilevel problem under verified conditions.

2. **STRONG:** First automatic reformulation selection engine that reasons about problem structure (lower-level convexity, integrality, constraint qualifications) to choose among fundamentally different reformulation paradigms (KKT, value function, CCG, regularization, direct solve).

3. **STRONG:** First compiler intermediate representation (IR) for bilevel optimization that enables optimization passes (e.g., detecting exploitable structure, tightening formulations) before emission to a target solver.

4. **MODERATE:** Solver-agnostic code generation from bilevel specifications, going beyond the solver-abstraction of JuMP/Pyomo to reformulation-aware emission.

5. **MODERATE:** Unified handling of mixed-integer lower levels via automatic routing to value-function or branch-and-cut strategies when KKT is invalid.

---

## 8. Conclusion

**The novelty gap is real and wide on the critical dimensions.** No existing tool, paper, or framework combines:
- Automatic structure-aware reformulation selection
- Formal correctness guarantees
- Compiler-style intermediate representation
- Solver-agnostic code generation

The individual components exist in isolation (BiOpt has multiple reformulations, JuMP has solver abstraction, Lozano & Smith have value function theory), but their principled integration under a compiler framework with correctness certificates is genuinely novel. The strongest version of this project should emphasize dimensions 1–3 above and clearly differentiate from the portfolio's single-level MIP reformulation projects.
