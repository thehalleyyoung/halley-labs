# Brainstorm: Competing Problem Framings for a Solver-Agnostic Bilevel Optimization Compiler

**Stage:** Crystallization — Problem Architect  
**Community:** Operations Research and Optimization  
**Date:** 2026-03-08  

---

## Framing A: Maximum Breadth — *A Universal Bilevel Compiler*

### Title

**BiCL: A Solver-Agnostic Compilation Framework for General Bilevel Optimization via Automated Reformulation Selection and Decomposition Synthesis**

### Problem Statement

Bilevel optimization—where an upper-level decision-maker optimizes subject to the optimal response of a lower-level agent—is the canonical mathematical formulation for Stackelberg games, mechanism design, adversarial robustness, toll pricing, network interdiction, facility location under competition, and dozens of other hierarchical decision problems. Despite the proliferation of bilevel models across OR, machine learning, and economics, the software ecosystem remains fragmented: practitioners must manually choose between KKT-based single-level reformulations, value-function reformulations, strong-duality replacements, Benders-like decomposition, column-and-constraint generation (C&CG), penalty-based relaxations, or trust-region descent methods—then hand-code the chosen strategy against a specific solver API (Gurobi, CPLEX, HiGHS, SCIP, Ipopt, OR-Tools). This manual reformulation step is error-prone (sign errors in complementarity conditions are legendary), solver-specific (a reformulation tuned for Gurobi's indicator constraints may be infeasible in HiGHS), and non-portable.

The fundamental gap is the absence of a *compiler* in the programming-language sense: a system that accepts a high-level, declarative bilevel specification and emits solver-ready reformulations, just as GCC accepts C and emits x86 or ARM. Existing algebraic modeling languages (Pyomo, JuMP, GAMS, AMPL) provide *syntax* for bilevel problems but not *automated reformulation selection*. Pyomo.Bilevel and the nascent `BilevelJuMP.jl` require the user to specify which reformulation to apply. MibS (Mixed-Integer Bilevel Solver) is a solver, not a compiler—it implements a single algorithm (branch-and-cut with value-function cuts) and cannot emit models for third-party solvers. There is no system that (i) parses a declarative bilevel program, (ii) infers problem structure (convexity of lower level, integrality, linearity, coupling structure), (iii) selects or synthesizes the best reformulation strategy from a portfolio, and (iv) emits a correct, solver-ready single-level model to any of several solver backends.

BiCL (Bilevel Compilation Language) fills this gap. It defines a typed intermediate representation (IR) for bilevel programs—capturing upper/lower objectives, coupling constraints, variable domains, and structural annotations—and implements a multi-pass compilation pipeline: **parsing → structural analysis → reformulation selection → reformulation lowering → solver emission**. The structural analysis pass performs automatic convexity detection (via disciplined convex programming rules à la CVX/CVXPY), integrality detection, constraint qualification verification (LICQ, Slater, Mangasarian-Fromovitz), and coupling-structure analysis (is the lower level parametrized only through the objective, only through the right-hand side, or through the constraint matrix?). The reformulation selection pass consults a *strategy algebra*—a formal calculus of reformulation transformations with soundness proofs—to enumerate feasible strategies, estimates their computational footprint (number of auxiliary variables, big-M tightness, constraint count), and selects via a cost model trained on problem features. The lowering pass applies the chosen reformulation (KKT replacement, strong duality, value-function cutting planes, regularized penalty) and emits a concrete single-level model to a backend-specific code generator targeting Gurobi (via gurobipy), HiGHS (via highspy), SCIP (via PySCIPOpt), CPLEX (via docplex), Ipopt (via CasADi), and OR-Tools (via ortools).

The scientific contribution is not merely software engineering but a formal *reformulation algebra* that unifies known reformulation techniques into a common framework, proves soundness (i.e., equivalence of the bilevel feasible set before and after reformulation under stated assumptions), and enables *automated composition*—e.g., applying strong duality to the lower level's continuous relaxation, then layering C&CG for the remaining integer variables. This algebraic framework yields reformulation strategies that, to our knowledge, have never been manually constructed in the literature.

### Extreme Value

**Who needs this?** Every OR practitioner, operations engineer, and ML researcher who encounters bilevel structure. Today, a PhD student modeling electricity market clearing with strategic generators must spend weeks manually deriving KKT conditions, debugging complementarity formulations, choosing big-M values, and then reimplementing everything when switching from Gurobi to an open-source solver. BiCL reduces this to a 20-line specification. Energy companies modeling strategic bidding, defense analysts solving network interdiction, ML researchers certifying adversarial robustness via bilevel formulations, supply-chain teams modeling vendor competition—all benefit.

**What becomes possible?** (a) *Reproducibility*: bilevel models are published as BiCL source, not ad-hoc solver scripts. (b) *Solver portability*: the same model runs on any backend, enabling fair solver benchmarking. (c) *Reformulation exploration*: researchers can systematically compare reformulation strategies on standard benchmarks rather than guessing. (d) *Accessibility*: bilevel optimization becomes usable by engineers without deep knowledge of duality theory.

### Genuine Difficulty

This is a *hard software artifact* because it requires solving several intertwined subproblems, each nontrivial:

1. **IR Design for Bilevel Programs (~15K LoC).** Defining a typed, validated intermediate representation that captures linear, quadratic, conic, and general nonlinear bilevel programs with mixed-integer variables at both levels, multiple lower-level problems (multi-follower), and optimistic/pessimistic variants. Must support parametric right-hand sides, bilinear coupling, and indicator constraints. Requires a full expression DAG with automatic differentiation support for KKT Jacobian computation.

2. **Structural Analysis Engine (~25K LoC).** Automated detection of: convexity (extending DCP rules to bilevel-specific structures), constraint qualification satisfaction (checking LICQ generically requires symbolic Jacobian rank analysis; checking Slater requires solving a feasibility LP), integrality structure, and coupling topology. This is algorithmically hard—convexity detection in the general case is co-NP-hard, so the engine must be conservative (sound but incomplete).

3. **Reformulation Strategy Algebra (~20K LoC).** Formalizing ~15 known reformulation techniques as composable transformations on the IR, with machine-checkable soundness conditions. Strategies include: (a) KKT replacement (requires convex lower level + CQ), (b) strong-duality replacement (requires LP or convex-QP lower level), (c) value-function reformulation with cutting planes (general but requires solving lower-level subproblems), (d) regularization/penalization (Scholtes, Steffensen, Kadrani relaxations of complementarity), (e) column-and-constraint generation (for integer lower level), (f) Benders decomposition for bilevel (when upper level has specific structure), (g) parametric programming (when lower level is an LP and upper variables enter only the RHS). Each strategy has preconditions (what problem classes it applies to) and postconditions (what the resulting single-level program looks like).

4. **Big-M and McCormick Bound Computation (~15K LoC).** KKT-based reformulations require big-M constants for complementarity linearization. Naïve big-M values yield weak relaxations. The compiler must solve bound-tightening LPs, apply interval arithmetic propagation, and exploit problem structure (network constraints, variable bounds) to compute tight big-M values automatically. Similarly, bilinear terms arising from complementarity products require McCormick envelope computation.

5. **Cost Model and Strategy Selection (~10K LoC).** Given the set of feasible strategies, selecting the best one requires predicting solver performance. This involves a feature extractor (problem size, density, integrality ratio, constraint-type histogram) and a trained cost model (random forest or lightweight model trainable on CPU). The training data comes from an automated benchmark suite.

6. **Solver Backend Code Generators (~30K LoC, ~5K per backend).** Each backend has idiosyncratic API constraints: Gurobi supports indicator constraints and SOS1 sets natively (useful for complementarity); HiGHS does not support indicators but handles LP relaxations efficiently; SCIP supports constraint handlers for lazy constraint generation; CPLEX supports Benders decomposition natively; Ipopt requires smooth NLP formulations (no integrality). The code generators must exploit each solver's strengths and work around its limitations.

7. **Automated Benchmark Suite and Oracle Evaluator (~20K LoC).** A comprehensive benchmark library (BOLIB instances, randomly generated bilevel LPs/QPs/MILPs, structured instances from network interdiction, toll pricing, adversarial ML) with automated correctness verification (checking bilevel feasibility, comparing objective values across reformulations, verifying that KKT conditions are satisfied at reported solutions).

8. **Correctness Infrastructure (~15K LoC).** Roundtrip property testing (reformulate → solve → verify bilevel feasibility of solution on original), differential testing across solver backends, and regression testing against known optimal values from the literature.

**Total estimated complexity: ~150K LoC** across IR, analysis, reformulation, code generation, benchmarking, and testing.

### Best Paper Argument

A top venue (Mathematical Programming, Operations Research, INFORMS Journal on Computing, or CPAIOR) would select this because:

- **It creates a new software category.** Just as CVXPY created "disciplined convex programming" compilers and made convex optimization accessible, BiCL creates "disciplined bilevel programming" compilers. The analogy is precise and powerful.
- **It unifies fragmented theory.** The reformulation algebra provides the first formal framework connecting KKT, strong duality, value-function, and decomposition approaches as elements of a common calculus, with machine-checked soundness.
- **It enables systematic empirical study.** For the first time, researchers can compare reformulation strategies head-to-head on standardized benchmarks using identical solver backends—currently impossible without manual reimplementation.
- **It lowers the barrier to bilevel optimization.** The practical impact on the OR community mirrors CVXPY's impact on convex optimization.
- **The artifact is massive and non-trivial.** The 150K+ LoC artifact demonstrates engineering at a scale that commands respect.

### Fatal Flaws

1. **Breadth may sacrifice depth.** By covering LP, QP, conic, MILP, and nonlinear bilevel problems, the compiler may produce reformulations that are correct but not competitive with hand-tuned, problem-specific implementations. Reviewers may ask: "Is the auto-selected strategy ever actually *better* than what an expert would choose?"
2. **Convexity/CQ detection is inherently incomplete.** The structural analysis engine will fail to classify some problems, forcing a fallback to conservative (slow) reformulations. This could undermine the "automatic" claim.
3. **Big-M computation is NP-hard in general.** Weak big-M values can make the reformulation orders of magnitude slower than a hand-tuned version. The bound-tightening procedure may itself become a bottleneck.
4. **Solver API instability.** Supporting 6 solver backends means 6 maintenance targets. API changes in any solver can break the corresponding code generator.
5. **Risk of "jack of all trades, master of none."** The breadth story is compelling conceptually but may produce underwhelming experimental results if no single reformulation strategy dominates across all problem classes.

---

## Framing B: Maximum Depth — *Breakthrough Solver Technology for Mixed-Integer Bilevel*

### Title

**MIBiL-Cut: A Branch-and-Cut Compiler for Mixed-Integer Bilevel Linear Programs with Automatic Intersection Cut Generation and Value-Function Strengthening**

### Problem Statement

Mixed-integer bilevel linear programs (MIBLPs)—bilevel problems where both upper and lower levels are linear and either or both levels contain integer variables—are among the hardest problems in optimization. They are Σ₂ᵖ-hard in general (i.e., harder than NP-hard, sitting at the second level of the polynomial hierarchy), and even checking whether a given point is bilevel feasible is NP-hard when the lower level contains integers. Despite this extreme worst-case complexity, MIBLPs arise naturally and unavoidably in critical applications: interdiction problems (which edges to destroy in a network to maximally disrupt an adversary's shortest path), defender-attacker problems in homeland security, competitive facility location (where to open stores given a competitor's rational response), and bilevel knapsack problems in resource allocation.

The state of the art for solving MIBLPs is MibS (Mixed-Integer Bilevel Solver), a branch-and-bound framework that uses value-function cuts—hyperplanes derived from the lower-level value function—to enforce the bilevel feasibility condition. While MibS represents a significant advance, it has fundamental limitations: (a) it uses only a basic class of intersection cuts derived from the lower-level LP relaxation, missing the rich structure available from the lower-level integer hull; (b) its branching strategy is generic (most-fractional or reliability branching) and does not exploit bilevel-specific structure; (c) it cannot leverage modern cut-generation paradigms like multi-row cutting planes, split cuts, or lattice-based cuts applied to the bilevel feasible region; and (d) it is a monolithic solver, not a compiler—it cannot emit strengthened formulations for use by other solvers.

MIBiL-Cut addresses these limitations through a *compilation approach to MIBLP solving*. Rather than implementing a standalone solver, it implements a *reformulation-and-strengthen compiler* that takes a high-level MIBLP specification and emits a progressively strengthened single-level MILP that can be solved by any MILP solver (Gurobi, CPLEX, SCIP, HiGHS). The strengthening is achieved through three novel algorithmic contributions:

**First**, we develop *bilevel intersection cuts*—a new family of cutting planes derived from the geometry of the bilevel feasible region. Standard intersection cuts (Balas, 1971) are derived from a lattice-free set containing the LP relaxation's optimal vertex. We extend this theory to the bilevel setting, where the "lattice-free set" is replaced by a *bilevel-infeasible set*—a convex region in the upper-level variable space where no lower-level optimal response exists that satisfies the upper-level constraints. We show that maximal bilevel-infeasible convex sets can be characterized via the lower-level value function's epigraph, and we derive closed-form intersection cuts for the case where the lower level is a pure LP, a bounded MILP, and a network flow problem.

**Second**, we introduce *value-function lifting*—a technique that strengthens value-function cuts by exploiting the integrality of upper-level variables. Standard value-function cuts are valid for the continuous relaxation of the upper level. When upper-level variables are integer, these cuts can be lifted (in the sense of Gomory-Johnson lifting) to obtain stronger inequalities. We develop a general lifting framework and efficient separation algorithms for the resulting cuts.

**Third**, we design *bilevel-aware branching rules* that exploit the hierarchical structure. Instead of branching on the most fractional variable (which ignores bilevel structure), we branch on variables that maximally change the lower-level optimal response. This requires solving auxiliary lower-level problems at each branching decision, but we show that warm-starting and dual recycling make this overhead manageable.

The compiler emits not just a single strengthened MILP but a *progressive strengthening sequence*: first the basic KKT/value-function reformulation, then the reformulation plus bilevel intersection cuts, then with lifted cuts, packaged as lazy constraint callbacks for solvers that support them or as a priori cut pools for those that don't. This allows the user to trade off compilation time against solver time.

### Extreme Value

**Who needs this?** Researchers and practitioners solving interdiction, defender-attacker, and competitive location problems—the bread and butter of military OR, homeland security optimization, and competitive strategy. These communities currently rely on MibS or hand-coded branch-and-cut implementations. MIBiL-Cut offers: (a) *stronger formulations* via novel cut families, directly translating to faster solve times; (b) *solver portability*—use Gurobi's commercial-grade branch-and-cut rather than MibS's academic-grade implementation; (c) *reproducibility*—the compiled MILP is a standard `.mps` file that any solver can read.

**What becomes possible?** Solving MIBLPs that are currently out of reach. The gap between the best LP relaxation bound and the optimal bilevel value is often enormous (50%+ for interdiction problems). Bilevel intersection cuts and value-function lifting can close a significant portion of this gap at the root node, reducing the branch-and-bound tree by orders of magnitude. We estimate 10-100x speedups on standard MIBLP benchmarks (BOLIB, Zeng-An instances, Fischetti-Ljubić-Monaci-Sinnl instances).

### Genuine Difficulty

1. **Bilevel Intersection Cut Theory and Implementation (~25K LoC).** Deriving the bilevel-infeasible convex sets requires computing the lower-level value function (piecewise linear for LP lower levels, but with exponentially many pieces). The implementation must efficiently enumerate the relevant pieces, compute the maximal bilevel-infeasible set containing the current LP relaxation vertex, and derive the intersection cut. For MILP lower levels, this requires Gomory-style arguments applied to the bilevel feasible region. The separation problem (finding a violated bilevel intersection cut) involves solving a sequence of LPs/MILPs, and the implementation must manage warm-starting, caching, and cut management (aging, purging, parallelism).

2. **Value-Function Computation and Approximation (~20K LoC).** The lower-level value function V(x) maps upper-level decisions x to the optimal lower-level objective. For LP lower levels, V(x) is piecewise linear and concave (for minimization). Computing V(x) exactly requires parametric LP algorithms. For MILP lower levels, V(x) is discontinuous and computing it exactly is intractable; the implementation must use approximation schemes (sampling, interpolation, outer approximation). The value-function oracle is called thousands of times during cut generation and must be extremely fast.

3. **Lifting Framework for Value-Function Cuts (~15K LoC).** Lifting a value-function cut with respect to integer upper-level variables requires solving a sequence of subproblems (one per variable to be lifted), each involving evaluation of V(x) at specific points. The lifting sequence matters (as in standard Gomory-Johnson theory), and finding a good sequence requires heuristics. The implementation must handle both sequential and simultaneous lifting, and must detect when lifting yields no improvement (to avoid wasted computation).

4. **Bilevel-Aware Branching (~15K LoC).** Computing the "bilevel impact" of branching on a variable requires solving two lower-level problems (one for each child node's bound change) and comparing the resulting lower-level responses. This must be integrated with the solver's branching callback (Gurobi's `cbLazy`, SCIP's branching plugins, CPLEX's branching callbacks). The implementation must balance the information gain against the computational cost of the auxiliary solves.

5. **Progressive Strengthening Pipeline (~20K LoC).** The compilation pipeline must manage: (a) initial reformulation (KKT or value-function), (b) root-node cut generation (bilevel intersection cuts + value-function lifting), (c) cut pool management and emission (a priori cuts vs. lazy callbacks), (d) cut coefficient scaling and numerical conditioning. The pipeline must detect when to stop adding cuts (diminishing returns) and must handle degenerate cases (unbounded lower level, infeasible lower level for some x values, nonunique lower-level optima—the optimistic vs. pessimistic distinction).

6. **MIBLP Benchmark Infrastructure (~20K LoC).** A comprehensive benchmark suite with: BOLIB instances, Fischetti-Ljubić-Monaci-Sinnl instances, randomly generated instances with controlled structure (density, integrality ratio, coupling type), and structured instances from interdiction (network interdiction, shortest-path interdiction, maximum-flow interdiction) and competitive facility location. Automated comparison against MibS, Gurobi solving the KKT reformulation directly, and CPLEX solving the strong-duality reformulation.

7. **Solver Callback Integration (~15K LoC).** Deep integration with solver callbacks for Gurobi (generic callbacks and traditional callbacks), SCIP (constraint handler plugins, separator plugins, brancher plugins), and CPLEX (lazy constraint callbacks, user cut callbacks). Each solver's callback API has different semantics (when cuts can be added, thread safety, incumbent management).

8. **Correctness and Numerical Stability (~10K LoC).** Bilevel problems are numerically treacherous—complementarity conditions create near-singular constraint matrices. The implementation must handle: big-M selection with automatic tightening, numerical tolerance management (when is a complementarity condition "satisfied"?), and solution verification (checking that a reported solution is genuinely bilevel feasible, not just an artifact of numerical tolerance).

**Total estimated complexity: ~140K–160K LoC.**

### Best Paper Argument

- **Novel theory.** Bilevel intersection cuts and value-function lifting are genuinely new contributions to integer programming theory. The intersection cut theory extends Balas's 1971 framework to the bilevel setting—a natural but unexplored direction.
- **Dramatic empirical impact.** If bilevel intersection cuts close even 20-30% of the root gap (comparable to what GMI cuts do for standard MILPs), this would represent a transformative advance in MIBLP solving. The compilation approach makes this testable at scale.
- **Bridges two communities.** This work connects the cutting plane theory community (Cornuéjols, Dash, Günlük) with the bilevel optimization community (Kleinert, Labbé, Ljubić, Schmidt). Best papers often bridge communities.
- **Reproducibility via compilation.** The compiled MILPs can be distributed as standard `.mps` files, enabling reproduction without installing specialized software—a significant practical advantage over monolithic solvers like MibS.
- **The "compiler" framing is novel.** Viewing cut generation as a *compilation* step (done offline, before handing the model to a solver) rather than as an *online* solver component is a fresh perspective that enables new algorithmic designs (e.g., spending 10 minutes on root-node compilation to save hours in branch-and-bound).

### Fatal Flaws

1. **Cut generation may be too expensive.** Bilevel intersection cuts require solving lower-level problems, which are themselves MILPs. If the lower level is hard, cut generation becomes a bottleneck. The "progressive strengthening" framing mitigates this, but reviewers may question whether the compilation overhead pays off.
2. **Limited to linear bilevel.** The restriction to MIBLPs (linear objectives and constraints at both levels) excludes quadratic, conic, and nonlinear bilevel problems. Reviewers may view this as too narrow. Counter-argument: MIBLPs are the workhorse of applied bilevel optimization and the linear case is where exact methods are most impactful.
3. **Comparison with MibS is tricky.** MibS is actively developed and may incorporate similar ideas. If MibS adds bilevel intersection cuts independently, the novelty claim weakens. Mitigation: the *compilation* approach (emitting to third-party solvers) is fundamentally different from MibS's monolithic approach.
4. **Numerical issues with complementarity.** Big-M reformulations are notoriously numerically fragile. Even with automatic bound tightening, there will be instances where the compiled MILP is numerically ill-conditioned. This could undermine the "solver-agnostic" claim if some solvers handle the numerics better than others.
5. **Scalability ceiling.** For large instances (10K+ variables), even root-node cut generation may be prohibitively expensive on a laptop CPU. The hard constraint of CPU-only execution limits the instance sizes that can be tackled.

---

## Framing C: Maximum Practical Impact — *Domain-Driven Bilevel Compilation for Energy Markets and Adversarial Robustness*

### Title

**BiCompile: A Domain-Aware Bilevel Optimization Compiler with Specialized Reformulation Strategies for Strategic Energy Bidding and Certified Adversarial Robustness**

### Problem Statement

Bilevel optimization is not merely a theoretical construct—it is the *correct* mathematical formulation for two of the most economically significant optimization problems of the current decade: **strategic bidding in electricity markets** and **certified adversarial robustness of machine learning models**. In both domains, the bilevel structure arises naturally and unavoidably, and in both domains, the current practice of manual reformulation is a critical bottleneck limiting the scale and reliability of deployed systems.

**In electricity markets**, generators submit bids to an independent system operator (ISO) who clears the market by solving a unit commitment / economic dispatch problem (the lower level). A strategic generator (the upper level) seeks to maximize its profit by choosing bid prices and quantities, anticipating the ISO's clearing decision. This bilevel model—known as the Mathematical Program with Equilibrium Constraints (MPEC) formulation of strategic bidding—is the standard framework used by energy companies worldwide (EPRI, ISO-NE, PJM). However, the lower-level market clearing problem is a large-scale mixed-integer program (unit commitment with binary on/off decisions, ramping constraints, network flow constraints, contingency constraints), making the bilevel problem extraordinarily hard. Current practice: energy analysts manually derive KKT conditions for a *simplified* (LP-relaxed, single-bus, no-contingency) version of the market clearing problem, losing critical fidelity. No existing tool can automatically reformulate the full-fidelity bilevel strategic bidding problem.

**In adversarial robustness**, certifying that a neural network's prediction is robust to input perturbations within an ℓ_p ball requires solving a bilevel problem: the inner (lower-level) problem finds the worst-case perturbation (maximizing loss), and the outer (upper-level) problem trains or evaluates the model. For piecewise-linear networks (ReLU activations), the inner problem is a mixed-integer linear program (via big-M formulations of ReLU). The bilevel problem thus has a continuous upper level (model weights or simply evaluation) and a mixed-integer lower level. Current practice: researchers use specialized verifiers (α-β-CROWN, MILP-based complete verification) that are tightly coupled to specific network architectures and perturbation models. There is no general bilevel compiler that can handle the adversarial robustness problem as a bilevel program and automatically apply domain-specific reformulations.

BiCompile is a *domain-aware bilevel compiler* that combines the generality of a compiler architecture (high-level specification → automated reformulation → solver emission) with *domain-specific reformulation strategies* that exploit the mathematical structure of energy markets and adversarial robustness. The key insight is that these two domains, despite their apparent dissimilarity, share deep structural commonalities amenable to unified algorithmic treatment: both involve a lower-level problem with *network structure* (power networks / neural network layers), both involve *mixed-integer lower levels* (unit commitment binaries / ReLU activation indicators), and both benefit from *decomposition along the network* (bus-level or layer-level decomposition).

BiCompile implements three domain-specific reformulation innovations:

1. **Network-Exploiting Decomposition.** For energy markets, the lower-level unit commitment problem has a network structure (buses connected by transmission lines). BiCompile automatically detects this structure and applies a *bus-level Benders decomposition* of the bilevel problem, decomposing the monolithic bilevel program into a master problem (the strategic generator's decisions) and subproblems (per-bus or per-region market clearing). For adversarial robustness, the neural network has a *layer structure*. BiCompile applies *layer-level decomposition*, reformulating the inner MILP as a sequence of layer-wise subproblems connected by linking constraints (analogous to the bound propagation used in α-β-CROWN, but derived systematically from Benders decomposition theory).

2. **Adaptive Relaxation Cascades.** Both domains involve lower-level MILPs that are too hard to reformulate exactly (via KKT) for realistic sizes. BiCompile implements *relaxation cascades*: a sequence of progressively tighter relaxations of the lower-level problem, from the LP relaxation (cheapest to reformulate) through Lagrangian relaxation (exploiting decomposable structure) to partial integrality enforcement (fixing the "easy" binaries and relaxing the "hard" ones). The compiler automatically determines which variables to relax based on sensitivity analysis of the lower-level dual values.

3. **Warm-Started Column-and-Constraint Generation.** For the full-fidelity bilevel problem with an integer lower level, BiCompile implements column-and-constraint generation (C&CG), the algorithm of Zeng and An (2014), with domain-specific warm-starting: for energy markets, the initial lower-level solutions are drawn from historical market clearing outcomes; for adversarial robustness, the initial adversarial examples are generated by fast heuristics (PGD, AutoAttack). This dramatically reduces the number of C&CG iterations needed for convergence.

### Extreme Value

**Who needs this?**

*Energy market participants:* Every generator, load-serving entity, and energy trading firm that participates in wholesale electricity markets. The U.S. wholesale electricity market alone is worth $400B+ annually. Strategic bidding optimization directly impacts revenue. Current tools (manual MPEC reformulations of simplified models) leave significant money on the table because they cannot model full network constraints, unit commitment binaries, or contingency requirements. BiCompile enables full-fidelity strategic bidding optimization for the first time.

*ML robustness researchers and practitioners:* Certified adversarial robustness is a prerequisite for deploying ML in safety-critical systems (autonomous vehicles, medical diagnosis, financial fraud detection). Current verification tools are architecture-specific monoliths. BiCompile provides a *general* framework: specify the network and perturbation model as a bilevel program, and the compiler handles the reformulation. This enables rapid experimentation with new architectures, perturbation models, and certification criteria without reimplementing verification algorithms.

**What becomes possible?** (a) Energy companies can optimize bids against the *actual* market clearing formulation, not a simplified proxy—potentially worth millions in additional annual revenue per firm. (b) ML researchers can certify robustness of novel architectures (transformers, graph neural networks) by specifying the bilevel problem, without developing custom verifiers. (c) The unified framework reveals structural parallels between energy market optimization and adversarial robustness, potentially leading to cross-pollination of algorithmic ideas.

### Genuine Difficulty

1. **Energy Market Lower-Level Modeling (~25K LoC).** Faithfully modeling ISO market clearing requires: multi-period unit commitment (binary on/off decisions with minimum up/down time constraints), economic dispatch (quadratic cost curves, piecewise linear approximations), DC optimal power flow (bus angle differences, line flow limits, PTDF-based or B-theta formulations), contingency constraints (N-1 security, generator/line outage scenarios), reserve requirements (spinning, non-spinning, regulation), and demand response. The lower-level model alone is a large-scale MILP with tens of thousands of variables and constraints for realistic systems (IEEE 118-bus, 300-bus, Polish 2383-bus). The compiler must parse a high-level specification of the market rules and generate the full lower-level model automatically.

2. **Neural Network Lower-Level Modeling (~20K LoC).** Encoding a neural network as a MILP requires big-M formulations of ReLU activations (one binary variable per neuron per layer), with bound propagation to tighten the big-M constants (interval arithmetic, CROWN bounds, α-CROWN bounds). The compiler must support: fully connected layers, convolutional layers (via unrolling), residual connections, batch normalization (folded into linear layers), and various activation functions (ReLU, leaky ReLU, sigmoid via piecewise linear approximation). For a moderately sized network (e.g., ResNet-18 on CIFAR-10), the MILP has millions of variables. The compiler must apply bound propagation and neuron stability analysis (identifying provably active/inactive neurons) to reduce the MILP size.

3. **Network-Exploiting Benders Decomposition (~20K LoC).** The Benders decomposition for the bilevel problem requires: (a) identifying the network structure in the lower-level MILP (graph partitioning of the constraint matrix), (b) formulating the Benders master problem with the upper-level variables and complicating linking variables, (c) generating Benders optimality and feasibility cuts by solving lower-level subproblems, (d) managing the cut pool (adding, aging, purging), (e) implementing Magnanti-Wong-style cut strengthening, and (f) handling the integer lower level (where standard Benders cuts are not valid and integer Benders cuts must be derived). The decomposition must work for both the energy and adversarial robustness domains, requiring a generic graph-partitioning-based decomposition engine.

4. **Adaptive Relaxation Cascade Engine (~15K LoC).** Determining which integer variables to relax (and in what order) requires: (a) solving the LP relaxation of the lower level and computing dual values, (b) performing sensitivity analysis to rank variables by their impact on the bilevel objective, (c) implementing a cascade of relaxations (LP → partial-integer → full-integer) with automated convergence checking (is the current relaxation tight enough for the upper-level decision?), (d) warm-starting each cascade level from the previous level's solution. The cascade must be adaptive: if the LP relaxation is tight, skip to full integrality; if it's loose, go through multiple intermediate relaxation levels.

5. **C&CG with Domain-Specific Warm-Starting (~15K LoC).** Implementing C&CG requires: (a) the basic C&CG loop (master problem → lower-level oracle → add lower-level solution as column/constraint → repeat), (b) convergence detection (checking optimality gap), (c) domain-specific warm-starting (parsing historical market clearing data for energy markets, implementing PGD/AutoAttack for adversarial robustness), (d) solution pool management (storing and reusing lower-level solutions across iterations), (e) parallelization of lower-level solves (solving multiple subproblems independently). The C&CG implementation must handle the case where the lower-level problem has multiple optima (optimistic vs. pessimistic bilevel optimization).

6. **Solver Backend Emission with Callback Integration (~20K LoC).** The compiled models must be emitted to at least three solver backends (Gurobi, SCIP, HiGHS), with callback support for lazy constraint generation (Benders cuts, C&CG cuts). Each solver's callback API differs significantly. The emission layer must also handle model size: for neural network verification, the MILP can have millions of variables, and the model must be generated incrementally (lazy variable/constraint creation) rather than all at once.

7. **Benchmark and Evaluation Suite (~20K LoC).** Domain-specific benchmarks: (a) Energy: IEEE test systems (14, 30, 57, 118, 300-bus), Polish system (2383-bus), strategic bidding scenarios with 1-5 strategic generators. (b) Adversarial robustness: MNIST/CIFAR-10 networks of varying sizes (small fully-connected to medium CNNs), ℓ_∞ and ℓ_2 perturbation models, comparison against α-β-CROWN and MILP-based verifiers. Automated evaluation: solve time, optimality gap, number of C&CG iterations, quality of relaxation cascade bounds.

8. **Correctness and Validation (~10K LoC).** Cross-domain validation: for energy markets, compare strategic bidding decisions against brute-force enumeration on small instances and against known optimal solutions from the literature. For adversarial robustness, compare certification results against α-β-CROWN (the state-of-the-art verifier) on standard benchmarks.

**Total estimated complexity: ~145K–165K LoC.**

### Best Paper Argument

- **Massive practical impact.** Energy markets and adversarial robustness are two of the highest-impact application domains in optimization today. A tool that makes bilevel optimization accessible in both domains simultaneously would have enormous practical value.
- **Novel algorithmic contributions.** Network-exploiting decomposition, adaptive relaxation cascades, and domain-specific warm-starting for C&CG are genuine algorithmic innovations, not just software engineering.
- **Cross-domain unification.** The insight that energy market optimization and adversarial robustness share deep structural commonalities (network structure, mixed-integer lower levels, decomposability) is itself a contribution. Best papers often reveal unexpected connections.
- **Compelling narrative.** "A single compiler that optimizes billion-dollar energy markets and certifies safety-critical ML models" is a narrative that resonates with broad audiences—from OR to ML to power systems.
- **Directly comparable to existing systems.** The evaluation can directly compare against state-of-the-art tools in both domains (MibS/custom MPEC solvers for energy; α-β-CROWN for adversarial robustness), providing concrete evidence of impact.

### Fatal Flaws

1. **Two-domain focus may seem ad hoc.** Reviewers may ask: "Why these two domains? Why not transportation, or supply chain, or healthcare?" The unifying structural argument (network structure + integer lower level) must be made very compelling, or the work risks appearing as two unrelated case studies stitched together.
2. **Full-fidelity energy models may be unsolvable.** Realistic unit commitment problems (thousands of buses, hundreds of generators, 24-48 time periods) produce lower-level MILPs with hundreds of thousands of variables. Even with decomposition, solving the bilevel problem on a laptop CPU may be infeasible for realistic scales. The evaluation may need to use simplified (but still significantly more detailed than current practice) models.
3. **Neural network MILP encoding is a known hard problem.** The adversarial robustness community has spent years optimizing MILP encodings of neural networks. BiCompile's general-purpose encoding may be significantly slower than specialized tools (α-β-CROWN) that exploit GPU acceleration and custom bound propagation. The CPU-only constraint is particularly limiting here.
4. **Scope creep risk.** Covering two complex domains with domain-specific optimizations is an enormous engineering effort. There is a real risk that neither domain receives sufficient depth, resulting in a tool that is worse than specialized alternatives in both domains.
5. **Domain expertise requirements.** Building a faithful energy market model requires deep knowledge of power systems engineering; building an effective neural network verifier requires deep knowledge of ML verification. The team may lack sufficient expertise in both domains, leading to modeling errors or suboptimal design choices.

---

## Comparative Summary

| Dimension | Framing A (Breadth) | Framing B (Depth) | Framing C (Impact) |
|---|---|---|---|
| **Scope** | All bilevel programs (LP, QP, conic, NLP, MI) | Mixed-integer bilevel linear only | Two domains: energy + adversarial ML |
| **Key Innovation** | Reformulation algebra + strategy selection | Bilevel intersection cuts + value-function lifting | Network decomposition + relaxation cascades |
| **Theory Depth** | Medium (soundness of strategy algebra) | High (new cutting plane family) | Medium (decomposition + warm-starting) |
| **Engineering Scale** | ~150K LoC | ~150K LoC | ~155K LoC |
| **Primary Risk** | Jack of all trades | Too narrow for broad audience | Two domains feel stitched together |
| **Best Venue** | INFORMS J. Computing / Math Programming | Math Programming / IPCO | Operations Research / NeurIPS |
| **Analogy** | "CVXPY for bilevel" | "Gomory cuts for bilevel" | "TensorFlow Serving for bilevel" |
| **Impact Width** | Very broad (all bilevel practitioners) | Narrow (MIBLP researchers) | Moderate (energy + ML communities) |
| **Impact Depth** | Moderate (correct but possibly slow) | Deep (10-100x speedups possible) | Deep (enables new applications) |
| **CPU-Only Feasibility** | High (compilation is cheap) | High (MILP solving is CPU-native) | Medium (NN verification suffers without GPU) |

## Recommendation for Selection

**If the goal is maximum best-paper probability at an OR venue:** Framing B. The cutting plane theory is the most novel, the empirical results would be the most dramatic (closing root gaps), and the math programming community rewards deep theoretical contributions with strong computational evidence. The "compiler" framing adds novelty over a standard solver paper.

**If the goal is maximum long-term impact on the OR community:** Framing A. A true bilevel compiler would be transformative infrastructure, but the risk of underwhelming benchmarks is high. This is a higher-variance bet.

**If the goal is maximum breadth of audience:** Framing C. The energy + ML angle appeals to applied OR, power systems, and ML communities simultaneously. However, the risk of insufficient depth in either domain is significant, and the CPU-only constraint particularly hurts the adversarial robustness component.

**Overall recommendation:** Framing B has the highest expected value for a best-paper at a top OR venue, with Framing A as the high-variance alternative. Framing C is best if targeting a broader venue (AAAI, NeurIPS) or if the team has strong domain expertise in both energy and ML.
