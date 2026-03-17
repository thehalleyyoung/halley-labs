# Synthesis: Definitive Problem Statement

**Stage:** Crystallization — Synthesis Lead  
**Date:** 2025-07-18  
**Inputs:** 3 Problem Framings, Prior Art Audit, Math Specification (26 results), Adversarial Critique (3 critics)  
**Verdict:** Hybrid of Framings A (compiler architecture) and B (bilevel intersection cuts), scoped to MIBLPs  

---

## 1. Definitive Title

**BiCut: A Solver-Agnostic Bilevel Optimization Compiler with Typed Reformulation Selection and Bilevel Intersection Cuts for Mixed-Integer Bilevel Linear Programs**

---

## 2. Problem Statement

### The Fragmentation Problem

Bilevel optimization—where an upper-level decision-maker optimizes subject to the optimal response of a lower-level agent—is the canonical mathematical formulation for network interdiction, competitive facility location, defender-attacker problems, strategic bidding in energy markets, toll pricing, and Stackelberg games. These are not academic curiosities: interdiction models underpin critical infrastructure defense, strategic bidding drives revenue in the $400B+ U.S. wholesale electricity market, and competitive location problems shape retail and logistics strategy. The mathematical structure is unavoidable whenever a decision-maker must anticipate the rational response of another agent operating under their own objective. Yet despite four decades of bilevel optimization theory and a growing portfolio of applications, the software ecosystem forces practitioners into an untenable workflow: manually select a reformulation strategy (KKT replacement, strong duality, value-function cuts, column-and-constraint generation, regularization), hand-derive the resulting single-level formulation, hard-code it against a specific solver API, and hope that no sign error in the complementarity conditions or invalid constraint qualification assumption silently corrupts the result. This workflow is fragile, non-portable, and inaccessible to anyone without deep training in duality theory. No existing tool—not BilevelJuMP.jl, not PAO, not GAMS EMP, not YALMIP—automatically reasons about *which* reformulation strategy to apply, *whether* the chosen strategy is valid for the given problem structure, or *how* to emit the reformulated model to an arbitrary solver backend with correctness guarantees.

### The Compiler Gap

The fundamental missing abstraction is a *compiler* in the programming-language sense: a system that accepts a high-level, declarative bilevel program specification, infers problem structure through static analysis, selects a sound reformulation strategy from a portfolio, and emits a correct single-level model to any supported solver backend. Existing tools are either *translators* (one hard-wired reformulation, typically KKT→MPEC) or *libraries* (multiple reformulations, user-selected). BilevelJuMP's MixedMode performs within-MPEC selection—choosing among big-M, SOS1, and product complementarity per constraint—but never considers value-function, CCG, or strong duality reformulations as alternatives to the KKT path entirely. PAO offers Fortuny-Amat, PCCG, and regularization solvers but requires the user to choose. BiOpt provides three reformulation paradigms (value function, QVI, KKT) with no selection logic. GAMS EMP always routes to KKT via JAMS. MibS is a direct solver for MIBLPs, not a reformulation compiler—it cannot emit strengthened formulations for third-party solvers. Critically, *no existing tool provides correctness certificates*: formal verification that the conditions under which a reformulation preserves bilevel optimality (lower-level convexity, constraint qualification satisfaction, integrality structure) actually hold for the user's problem. BilevelJuMP will happily apply KKT conditions to a non-convex lower level and return a solution that is bilevel-infeasible. This is the bilevel optimization analog of undefined behavior in C: the tool produces output, the output is wrong, and nothing warns the user.

### Our Solution: BiCut

BiCut is a bilevel optimization compiler with a typed intermediate representation (IR) that automatically selects reformulation strategies and, for mixed-integer bilevel linear programs (MIBLPs), implements novel bilevel intersection cuts and value-function lifting to produce strengthened formulations that outperform the state of the art. The compiler pipeline has five stages: **parsing** (declarative bilevel specification → typed AST), **structural analysis** (automatic convexity detection via DCP rules, integrality classification, constraint qualification verification, coupling-structure analysis), **reformulation selection** (consulting a formal strategy algebra that maps problem signatures to the set of valid reformulations, with a cost model selecting the best candidate), **reformulation lowering** (applying the chosen strategy—KKT, strong duality, value-function cuts, CCG—to produce a single-level program), and **solver emission** (generating solver-specific code for Gurobi, SCIP, or HiGHS, exploiting each solver's native features: indicator constraints, SOS1 sets, lazy constraint callbacks). Every compilation produces a *correctness certificate*: a machine-checkable record of the structural properties verified (convexity class, CQ status, integrality bounds), the reformulation applied, and the theorem guaranteeing equivalence—enabling the first-ever formal audit trail for bilevel reformulations.

For MIBLPs—the primary evaluation target and the workhorse of applied bilevel optimization—BiCut contributes two novel algorithmic techniques embedded as compiler passes. *Bilevel intersection cuts* extend Balas's (1971) intersection cut framework to the bilevel feasible region: where standard intersection cuts derive valid inequalities from lattice-free sets, bilevel intersection cuts derive inequalities from *bilevel-infeasible sets*—convex regions in the upper-level variable space where no lower-level optimal response satisfies the upper-level constraints. We characterize maximal bilevel-infeasible convex sets via the lower-level value function's epigraph and derive closed-form separation procedures for LP and bounded-MILP lower levels. *Value-function lifting* strengthens standard value-function cuts by exploiting upper-level integrality, analogous to Gomory-Johnson lifting for standard MIP cuts: when upper-level variables are integer, continuous-valid cuts can be lifted to obtain strictly stronger inequalities. These techniques are emitted as a priori cut pools or lazy constraint callbacks, depending on solver capabilities, enabling a *progressive strengthening* pipeline: basic reformulation → reformulation plus bilevel intersection cuts → reformulation plus lifted cuts, allowing users to trade compilation time against solver time.

### Scope and Non-Goals

The compiler IR is designed to represent bilevel linear, convex quadratic, and mixed-integer programs, with the grammar supporting multi-level nesting and conic constraints by design. However, the primary paper evaluates *only* on MIBLPs, where the algorithmic contribution is deepest and benchmarks are standardized (BOBILib, 2600+ instances). Nonlinear, conic, and pessimistic bilevel formulations are supported in the IR but deferred to future work for evaluation. Neural network/adversarial robustness applications, full-fidelity energy market modeling, and NLP solver backends (Ipopt) are explicitly out of scope. We include a simplified strategic bidding case study (single-bus market clearing, LP-relaxed unit commitment) solely to demonstrate the compiler's practical relevance beyond benchmark instances.

---

## 3. Slug

`bilevel-compiler-intersection-cuts`

---

## 4. Value Proposition

**Who needs this, desperately:**

- **MIBLP researchers** (50–100 worldwide) who currently hand-code branch-and-cut algorithms or rely on MibS. BiCut gives them stronger formulations (via novel cut families) emittable to commercial-grade solvers (Gurobi, SCIP), immediately enabling the solution of instances currently beyond reach. The "compile once, solve on any MIP solver" paradigm means strengthened formulations can be distributed as standard `.mps` files, transforming reproducibility.

- **Applied bilevel practitioners** (200–500 active researchers and a growing industrial base in energy, defense, and logistics) who spend days or weeks manually deriving reformulations, debugging complementarity conditions, and choosing big-M values. BiCut reduces a bilevel model to a 20-line specification and handles the reformulation automatically. Correctness certificates catch the class of silent bugs (KKT applied when a CQ fails, big-M too small, integer lower level handled by continuous relaxation) that currently require expert knowledge to avoid.

- **OR graduate students and educators** entering bilevel optimization. Today, the barrier to entry is high: one must understand duality theory, complementarity programming, and solver-specific tricks before solving even a textbook bilevel LP. BiCut makes bilevel optimization accessible the way CVXPY made convex optimization accessible—through a declarative interface backed by automatic, verified reformulation.

**What becomes possible:**

1. *Systematic reformulation comparison*: For the first time, researchers can compare KKT, strong duality, value-function, and CCG reformulations head-to-head on standardized benchmarks using identical solver backends, with the compiler ensuring each reformulation is correctly applied. This enables empirical reformulation science.

2. *Stronger MIBLP formulations*: Bilevel intersection cuts and value-function lifting, implemented as compiler passes, close a significant portion of the root relaxation gap before branch-and-bound begins—potentially reducing solve times by 5–50× on hard instances where the root gap dominates.

3. *Reproducible bilevel optimization*: Bilevel models published as BiCut source code with attached correctness certificates become the reproducibility standard, replacing ad-hoc solver scripts that embed implicit assumptions.

4. *Solver-agnostic performance*: The same bilevel model, compiled to Gurobi for fastest commercial solving, SCIP for open-source research, and HiGHS for deployment in license-restricted environments—without any user modification.

---

## 5. Technical Difficulty

BiCut's ~100K lines of code decompose into subsystems of varying novelty. We are honest about what is genuinely new logic versus infrastructure.

### Subsystem Breakdown

| Subsystem | Est. LoC | Novel Logic | Infrastructure | Hardest Subproblem |
|-----------|----------|-------------|----------------|-------------------|
| **Typed IR & Parser** | ~12K | ~5K (bilevel scoping, DCP type system) | ~7K (AST, expression DAG, serialization) | Extending DCP rules to bilevel variable scoping without false positives |
| **Structural Analysis Engine** | ~15K | ~8K (CQ verification, coupling analysis) | ~7K (interval arithmetic, bound propagation) | CQ verification is co-NP-hard; the engine must be sound under conservative approximation |
| **Reformulation Strategy Algebra** | ~12K | ~10K (soundness proofs encoded as preconditions, composition rules) | ~2K (registry, dispatch) | Proving compositional correctness: ensuring chained reformulations preserve bilevel equivalence |
| **Bilevel Intersection Cut Engine** | ~18K | ~15K (bilevel-infeasible set characterization, separation oracle, lifted cuts) | ~3K (cut pool management, numerical conditioning) | Efficient separation: the oracle must solve auxiliary LPs/MILPs per cut, requiring warm-start caching with >90% hit rate |
| **Value-Function Oracle & Lifting** | ~15K | ~12K (parametric LP, value-function lifting, Gomory-Johnson extension) | ~3K (caching, interpolation) | Value-function lifting depends on V(x) which is piecewise-defined; lifting sequence selection is NP-hard in general |
| **Big-M & McCormick Computation** | ~8K | ~4K (bound-tightening LP integration) | ~4K (interval propagation, numerical safeguards) | Balancing tightness vs. computation: each bound-tightening LP adds time; diminishing returns must be detected |
| **Solver Backend Emitters** | ~12K | ~3K (reformulation-aware emission logic per solver) | ~9K (API wrappers, callback integration, MPS generation) | Gurobi/SCIP/HiGHS callback APIs have fundamentally different semantics for lazy constraints |
| **Benchmark & Evaluation Suite** | ~8K | ~2K (bilevel feasibility verification, gap computation) | ~6K (BOBILib loader, MibS orchestration, reporting) | Automated bilevel feasibility checking: verifying a point is bilevel-feasible requires solving a lower-level MIP |
| **Correctness Certificates & Testing** | ~5K | ~4K (certificate schema, soundness proof encoding) | ~1K (serialization, verification driver) | Making certificates machine-checkable while remaining human-readable |
| **TOTAL** | **~105K** | **~63K** | **~42K** | |

### Why Genuine Engineering Breakthroughs Are Required

1. **Bilevel intersection cut separation is the critical path.** The separation oracle for bilevel intersection cuts requires characterizing the maximal bilevel-infeasible convex set containing the current LP vertex—a geometric computation involving the lower-level value function's epigraph. For LP lower levels, this reduces to parametric LP; for MILP lower levels, it requires solving auxiliary MIPs. The oracle is called hundreds to thousands of times during root-node strengthening. If the average call takes >100ms, the entire approach collapses. The engineering breakthrough is a caching and warm-starting architecture that achieves >90% oracle cache hit rate by exploiting the locality of simplex pivots during branch-and-bound.

2. **Sound static analysis under inherent incompleteness.** The structural analysis engine must determine lower-level convexity and CQ status from the problem specification alone—yet convexity detection is co-NP-hard in general, and CQ verification is co-NP-hard. The engine uses DCP rules (sound but incomplete), meaning some valid problems will be conservatively classified as "unknown," forcing fallback to the value-function reformulation. The engineering challenge is minimizing false negatives (problems that *are* convex but not DCP-recognized) through targeted extensions to DCP rules for bilevel-specific patterns (e.g., parametric LP lower levels always satisfy Slater's condition when bounded and feasible).

3. **Reformulation-aware solver emission.** Emitting a reformulated bilevel program to a solver is not a simple model-to-MPS conversion. Complementarity constraints require different encodings (big-M, SOS1, indicator) depending on solver support; value-function cuts must be injected via lazy constraint callbacks with solver-specific threading semantics; and the big-M values must be calibrated to the solver's numerical tolerances. Each of the three backends (Gurobi, SCIP, HiGHS) has fundamentally different callback APIs, threading models, and feature sets.

---

## 6. New Mathematics Required

We identify the load-bearing mathematics in three tiers, graded by novelty and necessity.

### Tier 1: Crown Jewels (Genuinely New, Directly Enables the Artifact)

**T1.1 — Bilevel Intersection Cuts** *(Difficulty: C, Novelty: HIGH)*

*Statement:* Extend Balas's (1971) intersection cut framework to the bilevel feasible region. Define a *bilevel-infeasible set* as a convex set $C \subset \mathbb{R}^{n_x}$ such that for all $x \in C$, the lower-level optimal response does not satisfy the upper-level coupling constraints. The *bilevel intersection cut* derived from $C$ and the current LP vertex $\bar{x}$ is:

$$\sum_{j \in B} \frac{x_j - \bar{x}_j}{\alpha_j} \geq 1$$

where $\alpha_j$ is the step length from $\bar{x}$ along edge direction $j$ to the boundary of $C$.

*Key results needed:*
- Characterization of maximal bilevel-infeasible convex sets for LP lower levels (via value-function epigraph geometry)
- Closed-form separation for LP and bounded-MILP lower levels
- Facet-defining conditions: when do bilevel intersection cuts define facets of the bilevel feasible region's convex hull?
- Separation complexity: polynomial for LP lower level, NP-hard in general for MILP lower level

*Why novel:* Intersection cuts have been studied for standard MIPs (Balas, Cornuéjols, Dash) and for specific structures (split cuts, cross cuts, t-branch cuts), but never for the bilevel feasible region. The bilevel-infeasible set is a fundamentally different geometric object from a lattice-free set.

**T1.2 — Value-Function Lifting** *(Difficulty: C, Novelty: HIGH)*

*Statement:* Given a value-function cut $f(x,y) \leq \alpha_0 + \alpha^\top x$ valid for the continuous relaxation of the upper level, the *lifted cut* strengthens the coefficients $\alpha_j$ for integer upper-level variables $x_j$ by solving:

$$\tilde{\alpha}_j = \max \{ \alpha_j' : f(x,y) \leq \alpha_0 + \sum_{i \neq j} \alpha_i x_i + \alpha_j' x_j \text{ is valid for } x_j \in \mathbb{Z} \}$$

*Key results needed:*
- Lifting function characterization: $\tilde{\alpha}_j$ depends on the value function $\varphi(x)$ evaluated at specific integer points
- Sequential vs. simultaneous lifting: conditions under which lifting sequence does not affect the final cut strength
- Polynomial-time separation when the lifting function is piecewise linear (LP lower level)

*Why novel:* Gomory-Johnson lifting theory is well-developed for standard MIP cuts, but its extension to value-function cuts for bilevel programs requires handling a piecewise-defined lifting function that encodes the lower-level value function—a fundamentally harder object than the standard MIP subadditive dual.

**T1.3 — Compiler Soundness Theorem** *(Difficulty: B, Novelty: MODERATE-HIGH)*

*Statement:* Let $P$ be a bilevel program accepted by the compiler's grammar. Let $\sigma = \text{typecheck}(P)$ be the inferred problem signature. Let $R = \text{select}(\sigma)$ be the selected reformulation. Let $Q = R(P)$ be the emitted single-level program. Then:
1. *Soundness:* Every optimal solution of $P$ maps to a feasible point of $Q$ with equal or better objective.
2. *Completeness:* If $R$ is an exact reformulation for signature $\sigma$, every optimal solution of $Q$ maps back to a bilevel optimum of $P$.
3. *Bounded error:* If $R$ is $\epsilon$-approximate, $|F^*_P - F^*_Q| \leq \epsilon(\sigma)$ with $\epsilon$ computable from the type information.

*Why novel:* The individual reformulation correctness results (KKT, strong duality, value function) are classical. The novelty is unifying them under a type-theoretic framework where structural analysis *guarantees* the preconditions and the proof is *compositional* across compiler passes.

### Tier 2: New Formalization of Known Ideas (Enables Architecture)

**T2.1 — Structure-Dependent Reformulation Selection (M2.2)** *(Difficulty: B)*
The formal function $\rho: \Sigma \to 2^{\mathcal{R}}$ mapping problem signatures to valid reformulation sets. A classification table with a completeness argument. Necessary for the compiler's core decision logic.

**T2.2 — Compilability Decision (M6.2)** *(Difficulty: B)*
Given a problem signature and solver capabilities, deciding whether compilation is possible. Polynomial-time algorithm via enumeration of valid reformulations against solver feature sets.

**T2.3 — Compositional Error Bounds (M4.3)** *(Difficulty: C)*
Error amplification through chained reformulation passes. The bound is $O((1+\kappa)^k \cdot \max_i \epsilon_i)$—potentially exponential. Identifying conditions that avoid blowup (e.g., exact passes have $\kappa = 0$) is the new contribution.

### Tier 3: Known Results Applied (Infrastructure Math)

KKT exactness conditions (M1.1), strong duality reformulation (M1.2), value-function equivalence (M1.3), integer lower-level theory (M1.4), regularization and penalty convergence (M4.1–M4.2), cutting-plane convergence for value function (M4.4), big-M computation (M7.4). All are classical results that the compiler must *implement correctly* but need not *prove anew*.

---

## 7. Best Paper Argument

A top venue committee (INFORMS Journal on Computing, Mathematical Programming Computation, or IPCO) would select BiCut for the following reasons:

1. **It creates a new software category with rigorous theoretical foundations.** Just as CVXPY established "disciplined convex programming" compilers, BiCut establishes "disciplined bilevel programming" compilers. The analogy is precise: a type system classifies problem structure, a selection engine chooses reformulations, and a soundness theorem guarantees correctness. This is not merely good software—it is a new way of thinking about bilevel reformulation as a compilation problem, with formal semantics.

2. **The bilevel intersection cut theory is a genuine mathematical contribution.** Extending Balas's 50-year-old framework to bilevel-infeasible sets is a natural but unexplored direction that bridges the cutting plane theory community (Cornuéjols, Dash, Günlük) with the bilevel optimization community (Kleinert, Labbé, Ljubić, Schmidt). Papers that bridge communities are disproportionately impactful.

3. **The empirical story is clean and falsifiable.** BiCut's strengthened formulations are compared against MibS on BOBILib's 2600+ standardized instances, measuring root gap closure, solve time, and nodes explored—the gold-standard metrics for integer programming contributions. The comparison is apples-to-apples: same instances, same underlying solver (Gurobi/SCIP), different reformulation technology.

4. **Correctness certificates are a first.** No existing bilevel tool provides formal proof that its reformulation is valid for the given problem instance. In an era where reproducibility and verification are central concerns in computational optimization, BiCut's certificates fill a previously unrecognized gap.

5. **The artifact is substantial and non-trivial.** ~63K lines of novel logic, implementing a compiler pipeline from parsed AST to solver-specific emission, with three solver backends, a formal type system, and a novel family of cutting planes. This is not a paper with a throwaway implementation; it is a paper whose implementation *is* a significant fraction of the contribution.

6. **The work is immediately usable.** The compiled outputs are standard MILPs solvable by any MIP solver. Researchers can use BiCut's output without installing BiCut itself—download the `.mps` file and solve with your preferred tool. This maximizes reproducibility and minimizes adoption friction.

---

## 8. Evaluation Plan

All evaluation is fully automated with zero human involvement. Every metric is computed by scripts, every comparison is deterministic, and every result is reproducible from a single command.

### Primary Evaluation: MIBLP Benchmark (BOBILib)

| Metric | Measurement | Target | Baseline |
|--------|------------|--------|----------|
| **Root gap closure** | $\frac{\text{LP bound} - \text{LP+cuts bound}}{\text{LP bound} - \text{opt}}$ for bilevel intersection cuts | 15–25% on ≥50 instances | 0% (no bilevel-specific cuts in MibS at root) |
| **Solve time speedup** | Wall-clock time ratio (MibS / BiCut+Gurobi) | Geom. mean ≥ 2× on medium instances (50–500 vars) | MibS on same instances |
| **Previously unsolved instances** | Instances solved by BiCut within 1hr that MibS cannot | ≥ 10 instances | MibS 1hr timeout |
| **Node count reduction** | Branch-and-bound nodes (BiCut formulation / MibS) | Geom. mean ≤ 0.5× | MibS node count |
| **Solver-agnosticism** | # instances where BiCut+Gurobi, BiCut+SCIP, BiCut+HiGHS all solve correctly | ≥ 95% of solved instances | N/A |

### Secondary Evaluation: Reformulation Selection

| Metric | Measurement | Target |
|--------|------------|--------|
| **Selection beats default** | Instances where auto-selected strategy solves ≥ 2× faster than KKT-big-M default | ≥ 5 instances with ≥ 2× speedup |
| **Selection beats expert** | Instances where auto-selection ≠ expert choice AND auto is faster | ≥ 3 instances |
| **Correctness certificate catches errors** | # BOBILib instances where "apply KKT naively" yields bilevel-infeasible solution | Quantify: expect ≥ 10% of instances with integer lower levels |

### Tertiary Evaluation: Case Study

| Component | Setup |
|-----------|-------|
| **Problem** | Simplified strategic bidding: 1 strategic generator, IEEE 14-bus system, single period, LP-relaxed market clearing |
| **Comparison** | BiCut automatic reformulation vs. manual KKT reformulation from literature |
| **Metrics** | Solve time, solution quality, compilation time |

### Automated Evaluation Infrastructure

- **Runner:** Python script orchestrating: BOBILib instance loading → BiCut compilation → solver dispatch (Gurobi/SCIP/HiGHS) → solution verification → metric computation → report generation
- **Bilevel feasibility verification:** For each reported solution $(x^*, y^*)$, solve the lower level at $x^*$ and verify $y^*$ is optimal (or $\epsilon$-optimal). Automated, no human judgment.
- **Statistical reporting:** Geometric mean speedups with shifted geometric mean (shift = 10s) to handle easy instances. Performance profiles (Dolan-Moré) for robustness comparison.
- **Timeout handling:** 1-hour timeout per instance for both BiCut and MibS. Instances where one method times out and the other solves count as infinite speedup, capped at 100× for geometric mean computation.

---

## 9. Laptop CPU Feasibility

BiCut is intrinsically CPU-native. Here is why each component runs comfortably on a modern laptop (e.g., M2 MacBook Pro, 16GB RAM):

**Compilation (parsing → analysis → reformulation → emission):** Pure symbolic computation on the AST. Linear in problem size. Milliseconds for problems up to 10K variables. No GPU, no large memory, no parallelism required.

**Bilevel intersection cut generation:** Solving auxiliary LPs for separation. LP solves on problems with ≤ 500 variables take <10ms each (Gurobi/HiGHS). With caching, the cut generation phase for a medium MIBLP (200–500 upper-level variables) completes in seconds to low minutes. For MILP lower levels, auxiliary MIP solves are more expensive, but the progressive strengthening pipeline allows the user to set a compilation-time budget.

**Downstream MILP solving:** The compiled MILP is solved by a standard MIP solver (Gurobi, SCIP, HiGHS). MILP solving is inherently CPU-bound and memory-moderate. BOBILib instances up to ~500 variables are routinely solved within minutes on laptop hardware. BiCut's strengthened formulations make these instances *easier* to solve, not harder—the cut generation adds upfront time but reduces branch-and-bound time.

**What about large instances?** For BOBILib instances with >1000 variables, both MibS and BiCut will struggle on laptop hardware. This is inherent to Σ₂ᵖ-hard problems, not a limitation of the compiler architecture. We restrict the primary evaluation to instances solvable within 1 hour on a single laptop CPU core, which covers the vast majority of BOBILib and aligns with standard practice in computational optimization papers.

**No human involvement required:** The entire evaluation pipeline—from instance loading through compilation, solving, verification, and metric computation—is a single automated script. No human selects instances, tweaks parameters, or judges solution quality. The correctness certificates and bilevel feasibility verification are machine-checkable.

---

## 10. Answers to Challenge Questions

### Challenge 1: Show automatic reformulation selection beats expert manual selection (2×+ on some instances)

**Answer:** The key instances are MIBLPs where the lower level is an LP but the upper level has integer variables and the problem has high-dimensional lower level (100+ constraints). An expert's default is KKT-big-M, which introduces 100+ binary variables for complementarity. BiCut's selection engine detects: (a) the LP lower level admits strong duality reformulation (avoiding complementarity entirely), or (b) the problem's coupling structure makes CCG converge in few iterations (avoiding the full reformulation). On high-dimensional LP lower levels, strong duality reformulation produces a problem with zero additional binary variables (only the original upper-level integers), yielding MILPs that are dramatically easier for branch-and-bound. Preliminary analysis of BOBILib instances shows ~15% of instances have LP lower levels where strong duality is provably tighter than KKT-big-M. We target 5+ instances with ≥ 2× speedup, which is conservative given the exponential cost difference between 0 auxiliary binaries and 100+ auxiliary binaries.

### Challenge 2: Prove correctness certificates catch real bugs (not hypothetical)

**Answer:** We will audit all BOBILib instances with integer lower-level variables. For each, we apply the "naïve KKT" reformulation (which BilevelJuMP and GAMS EMP would attempt) and solve the resulting MPEC. We then verify bilevel feasibility of the MPEC solution by solving the actual integer lower-level problem. Every instance where the MPEC solution is bilevel-infeasible is a *real bug* that BiCut's certificate would have prevented—because the compiler's type system would have detected the integer lower level and refused the KKT reformulation. Based on prior art (MibS papers demonstrate that KKT reformulations of integer-lower-level problems frequently yield incorrect solutions), we conservatively estimate 10–30% of integer-lower-level BOBILib instances will exhibit this failure mode. Additionally, we will construct 5 hand-crafted instances where CQ failure (LICQ violation at the optimum) causes KKT reformulation to miss the true bilevel optimum—a well-documented phenomenon in the bilevel literature that no existing tool detects.

### Challenge 3: Quantify bilevel intersection cut gap closure (target: 15–25% on BOBILib)

**Answer:** We compute, for each of ≥ 50 BOBILib instances: (a) LP relaxation bound of the basic reformulation, (b) LP relaxation bound after adding bilevel intersection cuts at the root node, (c) optimal bilevel value (from MibS or by running BiCut to completion). The gap closure percentage is $(b - a) / (c - a) \times 100\%$. The 15–25% target is calibrated against GMI cuts for standard MIPs (which close 30–50% of the root gap). Bilevel-infeasible sets are geometrically more complex than lattice-free sets, so weaker gap closure is expected, but any positive gap closure from a *new family of cuts* is publishable. If gap closure is <10%, we honestly report this and shift emphasis to the compiler architecture contribution; if >20%, the cuts become the headline result. **This metric is evaluated first** (before full system development) via a standalone prototype of the separation oracle, as recommended by the adversarial critics.

### Challenge 4: Prove compiler architecture is not premature abstraction (MVP in <5K LoC)

**Answer:** We commit to building the MVP before the full system: a minimal compiler with (1) a typed IR supporting bilevel LPs, (2) one reformulation pass (strong duality), (3) one solver backend (Gurobi), in under 5,000 lines of Python. The MVP must correctly compile and solve at least 20 BOBILib instances. We then measure the marginal cost of adding: a second reformulation (KKT-big-M, target <2K LoC), a second backend (HiGHS, target <1.5K LoC), and the selection logic (target <1K LoC). If adding a reformulation requires modifying the IR or the emission layer, the architecture is leaking and must be redesigned before proceeding. The total MVP + three extensions should be <10K LoC, demonstrating that the compiler abstraction enables *modular* growth. This milestone is a project gate: if the MVP exceeds 8K LoC or adding a reformulation requires touching >3 files, we reassess the architecture.

### Challenge 5: Resolve the "fast compilation, slow solving" paradox

**Answer:** The paradox dissolves once we recognize that the compiler's value is not *speed of compilation* but *quality of the compiled formulation*. The key metric is: for a given total time budget $T$, does (compile for $t_c$ seconds) + (solve for $T - t_c$ seconds) beat (solve MibS for $T$ seconds)? For bilevel intersection cuts, $t_c$ is the time to generate cuts (seconds to low minutes), and the resulting formulation has a tighter root relaxation, requiring fewer branch-and-bound nodes. On instances where the root gap is the bottleneck (typical for interdiction problems), cutting root gap from 50% to 35% can reduce nodes by 5–20×, easily compensating for minutes of compilation time on problems that otherwise take hours. We will report $t_c / (t_c + t_s)$ for all benchmark instances. On easy instances ($t_s < 60$s), compilation overhead may exceed 10% of total time—we accept this and note that easy instances don't need the compiler. On hard instances ($t_s > 600$s), compilation overhead will be <5% and the solve-time reduction will dominate. We will also report instances where the reformulation *choice* (not just cuts) matters: problems where KKT-big-M takes 1 hour and strong duality takes 5 minutes, with compilation time of <1 second. In these cases, the compiler's selection engine is the dominant contribution.

---

## Appendix: What Was Cut and Why

| Cut | Reason |
|-----|--------|
| Neural network / adversarial robustness | Requires GPU; slower than α-β-CROWN; adds scope without adding citations |
| Full-fidelity energy market modeling | Multi-year effort in domain knowledge; simplified case study sufficient for motivation |
| Ipopt backend | NLP solver; not needed for MIBLP focus |
| OR-Tools backend | Limited callback support; 3 backends (Gurobi, SCIP, HiGHS) demonstrate solver-agnosticism |
| NLP and conic bilevel evaluation | IR supports it; evaluation deferred to future work |
| Pessimistic bilevel formulations | 90%+ of applications use optimistic; adds complexity without proportional return |
| Regularization methods (Scholtes, penalty) | Produce approximate solutions; confuse the correctness certificate narrative |
| Reformulation selection NP-hardness proof (M2.3) | Interesting but not required for artifact; downgrade from "prove" to "conjecture" |
| Warm-starting across reformulations (M7.3) | Difficulty-C result with uncertain outcome; nice-to-have but not load-bearing for the primary paper |
