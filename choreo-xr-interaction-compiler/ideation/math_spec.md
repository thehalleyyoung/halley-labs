# Mathematical Specification: XR Event Calculus Compiler

**Artifact**: DSL compiler + incremental runtime → R-tree-backed event automata for MR interaction patterns, with reachability/deadlock verification.

**Assessment methodology**: For each component I identify the load-bearing mathematical content, state what must be proved, cite the nearest prior art, and grade novelty honestly.

---

## 1. Event Calculus Formalization for Spatial-Temporal XR

### 1.1 Background: Classical Event Calculus

The Event Calculus (EC) of Kowalski & Sergot (1986) operates over:
- A sorted domain with sorts **event**, **fluent**, **timepoint** (linearly ordered).
- Core predicates: `Happens(e, t)`, `Initiates(e, f, t)`, `Terminates(e, f, t)`, `HoldsAt(f, t)`, `Clipped(t₁, f, t₂)`.
- Axioms that derive `HoldsAt` from `Happens`, `Initiates`, `Terminates`, and the absence of clipping.

The Discrete Event Calculus (DEC) variant (Mueller, 2006) restricts timepoints to integers and replaces circumscription with completion, making it amenable to SAT-based reasoning. This is the correct starting point for a compiler-oriented formalization.

### 1.2 Required Spatial Extensions

The DSL needs spatial predicates over objects in ℝ³ (or SE(3) for full pose). The minimal set:

| Predicate | Signature | Semantics |
|-----------|-----------|-----------|
| `Proximity(a, b, r)` | Entity × Entity × ℝ⁺ → fluent | ‖pos(a,t) − pos(b,t)‖ < r |
| `Inside(a, V)` | Entity × Volume → fluent | pos(a,t) ∈ V |
| `GazeAt(a, b, θ)` | Entity × Entity × ℝ⁺ → fluent | angle(gaze_dir(a,t), pos(b,t)−pos(a,t)) < θ |
| `Grasping(a, b)` | Entity × Entity → fluent | contact(a,t) ∧ grip_closed(a,t) ∧ Proximity(a,b,ε) |

**Key issue**: These are *continuous* predicates—they hold or fail at every instant based on continuous spatial state. Classical EC fluents are *inertial*: once initiated, they persist until terminated. Spatial predicates are *non-inertial*; their truth value is determined entirely by the spatial configuration at each moment.

### 1.3 Handling Continuous Predicates in Discrete EC

There are three known approaches:

**(a) Sampling + Event Generation (the engineering approach)**: Discretize time at some rate Δt. At each tick, evaluate spatial predicates against the current scene state. When a predicate transitions from false→true, generate a synthetic `Happens(enter_proximity(a,b,r), t)` event; true→false generates `Happens(exit_proximity(a,b,r), t)`. Inertial fluents then track these transitions normally.

**(b) Continuous Event Calculus (Mueller, 2004; Shanahan, 1990)**: Extend EC with trajectory axioms: `Trajectory(f, t, g, d)` means "if f is initiated at t, then g holds at t+d." This handles continuous change within EC proper. However, the trajectory axiom is designed for *derived* continuous quantities, not for externally-driven spatial state.

**(c) Hybrid approach (what this artifact actually needs)**: Treat the spatial scene as an *oracle*—an external function `σ: T → Scene` where `Scene` gives the full spatial configuration. Spatial predicates are *derived* fluents: `HoldsAt(Proximity(a,b,r), t) ≡ ‖σ(t).pos(a) − σ(t).pos(b)‖ < r`. Only *transitions* of these derived fluents feed events into the EC machinery.

**What must be formalized**:
- **Definition 1.1** (Spatial Event Calculus — SEC): A tuple ⟨E, F_inertial, F_spatial, A, σ, Δt⟩ where E is a set of event types, F_inertial are standard EC fluents, F_spatial are derived spatial fluents defined by computable predicates over Scene, A is the EC axiom set (DEC variant), σ is a scene trace, and Δt is the sampling resolution.
- **Theorem 1.1** (Sampling soundness): For any SEC theory and Lipschitz-continuous spatial trajectories with constant L, if Δt < ε/(L·√3) for threshold ε, then every spatial predicate transition in the continuous trace is detected within one tick. (This is a straightforward application of the Lipschitz condition—the proof is routine.)
- **Theorem 1.2** (Conservative extension): The SEC axiom set, restricted to inertial fluents only, reduces exactly to the DEC of Mueller (2006). (Needed for correctness but essentially trivial.)

### 1.4 Novelty Assessment

The idea of combining Event Calculus with spatial reasoning is not new. Shanahan (1996) considered EC with spatial domains. Mueller (2006, Ch. 14) discusses EC with continuous change. The "sample and generate events" pattern is standard in game engines and robotics middleware (ROS).

**What might be novel**: The specific *formalization* of SEC as a conservative extension of DEC with a spatial oracle and a sampling-soundness theorem tied to Lipschitz bounds. However, the mathematical content of this theorem is elementary analysis (intermediate value theorem + Lipschitz bound).

**Grade: C+** — Straightforward application of known EC theory and basic real analysis. The formalization has value for the artifact but does not advance the mathematical state of the art. The "spatial oracle" framing is a clean engineering contribution, not a mathematical one.

---

## 2. R-tree-Backed Event Automata

### 2.1 The Formal Model

The compiled artifact is an automaton. The question is: what kind?

**Definition 2.1** (Spatial Event Automaton — SEA): A tuple ⟨Q, Σ, δ, q₀, F, R, guard⟩ where:
- Q is a finite set of control states.
- Σ is an alphabet of event types (both explicit user events and synthetic spatial-transition events).
- δ: Q × Σ → Q is the (partial) transition function.
- q₀ ∈ Q is the initial state; F ⊆ Q is the set of accepting states.
- R is an R-tree index over spatial entities in the scene.
- guard: (Q × Σ) → SpatialQuery is a function mapping each enabled transition to an R-tree query (range query, k-NN, containment test) that must succeed for the transition to fire.

This is essentially a guarded finite automaton where guards are spatial queries. The R-tree is not part of the automaton's formal state—it is an *acceleration structure* for evaluating guards efficiently.

### 2.2 Product Structure

For a scene with multiple interacting entities, the system state is:

**S = Q₁ × Q₂ × ... × Qₖ × Scene**

where Qᵢ is the automaton state for the i-th interaction pattern and Scene is the spatial configuration. This is a *synchronous product* of finite automata composed with a continuous (but externally-driven) spatial state.

If we abstract the spatial state to a finite set of *spatial predicates* that are currently true (which is valid since there are finitely many entities and finitely many predicates of interest), then the full state space is:

**S_abstract = Q₁ × Q₂ × ... × Qₖ × 2^P**

where P is the set of spatial predicates. This is finite and thus amenable to model checking.

### 2.3 R-tree Integration and Complexity

The R-tree accelerates guard evaluation. For n spatial entities in d dimensions:

| Operation | R-tree complexity | Naive complexity |
|-----------|------------------|-----------------|
| Range query (proximity) | O(n^(1−1/d) + k) expected | O(n) |
| Containment test | O(n^(1−1/d) + k) expected | O(n) |
| k-NN (gaze target) | O(n^(1−1/d) · log n) | O(n log n) |
| Insert/Delete (scene update) | O(log n) amortized | — |

For d=3 and moderate n (typical XR scenes: n ≤ 10⁴), the R-tree provides a constant-factor speedup, not an asymptotic one. The R-tree becomes load-bearing only if n is large (e.g., particle systems, dense environments).

**Theorem 2.1** (Guard evaluation completeness): For any spatial guard expressible as a Boolean combination of range queries, containment tests, and k-NN queries over a set of entities, the R-tree-backed evaluation returns the same result as brute-force evaluation.

This is immediate from the correctness of R-tree query algorithms (Guttman, 1984; Beckmann et al., 1990). No new proof is needed.

### 2.4 Complexity of State-Space Exploration

For the abstract state space S_abstract:
- |S_abstract| = |Q₁| × ... × |Qₖ| × 2^|P|
- |P| is O(n² · m) where n is the entity count and m is the number of predicate templates (proximity, containment, etc.).
- For k interaction patterns each with at most q states: |S_abstract| = q^k · 2^(O(n²·m)).

This is exponential in both k and n²·m. **This is the standard state-explosion problem of model checking.** Nothing about the spatial domain changes this fundamental complexity.

### 2.5 Novelty Assessment

R-trees are a well-understood spatial index (Guttman 1984). Guarded automata are standard (Alur & Dill 1994 for timed guards; Bouyer et al. for weighted extensions). The combination of spatial guards with finite automata is conceptually clean but mathematically straightforward—the guards do not interact with the automaton structure in any way that requires new theory.

**Grade: C** — This is an engineering artifact, not a mathematical contribution. The R-tree is an implementation choice for performance; the formal model is a standard guarded finite automaton. The product construction is textbook. No new theorems are needed.

---

## 3. Reachability and Deadlock Verification

### 3.1 State Space Characterization

From §2.4, the abstract state space is S_abstract = Q₁ × ... × Qₖ × 2^P.

**Deadlock** in this context means: a state s ∈ S_abstract from which no transition is enabled (all guards fail, no events can fire). Formally:

**Definition 3.1** (Deadlock): A state s = (q₁,...,qₖ, P_true) is a deadlock state iff for all i ∈ {1,...,k}, for all events e ∈ Σ such that δᵢ(qᵢ,e) is defined, guard(qᵢ,e) is not satisfiable by any spatial configuration consistent with P_true.

**Reachability**: Given a target state (or set of states) T ⊆ S_abstract, is there a sequence of events and spatial configurations that drives the system from the initial state to some s ∈ T?

### 3.2 Decidability

**Theorem 3.1** (Decidability): Reachability and deadlock detection for the abstract state space S_abstract are decidable.

*Proof sketch*: S_abstract is finite. Reachability reduces to graph reachability in the finite transition graph. Deadlock detection reduces to checking for sink nodes. Both are decidable. □

This is immediate and uninteresting. The question is not decidability but *tractability*.

### 3.3 Complexity and Applicable Algorithms

The state-space size is exponential (§2.4), so naive exploration is PSPACE-hard in general (reachability in product automata). Applicable techniques:

**(a) Explicit-state model checking** (à la SPIN): BFS/DFS over the reachable state space. Effective when the reachable space is much smaller than S_abstract (common in practice). Space: O(|Reachable|); Time: O(|Reachable| × |Σ|).

**(b) Symbolic model checking with BDDs**: Represent state sets as BDDs. Effective when the state space has regular structure. The spatial predicate dimensions add Boolean variables; if |P| is moderate (say ≤ 50), this is feasible. Standard algorithms (fixed-point computation for reachability, CTL model checking) apply directly.

**(c) SAT-based bounded model checking** (BMC): Encode k-step reachability as a propositional formula. Effective for bug-finding in deep state spaces. The encoding is standard: the spatial predicates become additional Boolean variables with constraints on legal combinations (not all subsets of P are geometrically realizable).

**(d) Abstraction-refinement (CEGAR)**: Abstract the spatial predicate space by merging predicates. Refine on spurious counterexamples. This could help if P is large. The theory is well-established (Clarke et al., 2000).

**What's potentially interesting**: The spatial predicate space has geometric structure that generic model checkers ignore. Specifically, not all elements of 2^P are *geometrically realizable*—you cannot have Proximity(a,b,1) ∧ ¬Proximity(a,b,2) if 1 < 2, and triangle inequality constrains proximity predicates jointly. Exploiting this structure to prune the state space is potentially novel.

**Theorem 3.2** (Geometric consistency pruning): Let C ⊆ 2^P be the set of geometrically consistent predicate valuations (those realizable by some spatial configuration in ℝ³). The reachable state space is contained in Q₁ × ... × Qₖ × C, and |C| can be exponentially smaller than 2^|P|.

*Proof obligation*: Characterize C. For proximity predicates with thresholds r₁ < r₂ < ... < rₘ, the consistency constraints are:
1. Monotonicity: Proximity(a,b,rᵢ) → Proximity(a,b,rⱼ) for rᵢ ≤ rⱼ.
2. Triangle inequality: Proximity(a,b,r₁) ∧ Proximity(b,c,r₂) → Proximity(a,c,r₁+r₂).
3. Containment consistency: Inside(a,V₁) ∧ V₁ ⊆ V₂ → Inside(a,V₂).

Computing |C| and its structure is related to the *theory of metric spaces with distance predicates*, which connects to the satisfiability of metric constraints (studied in constraint satisfaction and computational geometry). The upper bound on |C| depends on the specific predicate vocabulary. For m proximity thresholds over n entities, monotonicity alone reduces 2^(n²·m) to O(m^(n²))—still large but a meaningful reduction.

### 3.4 Novelty Assessment

Reachability and deadlock verification are completely standard problems in model checking. The algorithms (BFS, BDD, SAT/BMC, CEGAR) are off-the-shelf. Decidability is trivial (finite state space).

The one potentially interesting contribution is the geometric consistency pruning (Theorem 3.2)—exploiting metric-space constraints to reduce the abstract state space. However, similar ideas appear in the timed automata literature (clock zone pruning in UPPAAL) and in spatial constraint databases. The novelty would depend on whether the specific geometric consistency characterization for XR spatial predicates yields meaningfully better bounds than generic approaches.

**Grade: B−** — The verification infrastructure is entirely standard (Grade C), but the geometric consistency pruning could be a modest mathematical contribution (Grade B) if the characterization of C leads to tight bounds and an efficient membership test. This needs to be worked out carefully to determine if it's truly non-trivial.

---

## 4. Incremental Compilation

### 4.1 The Update Model

When a DSL specification changes (e.g., a developer modifies an interaction pattern), the compiler should not rebuild the entire automaton product from scratch. The incremental model:

**Definition 4.1** (Delta): A delta Δ is a pair (Δ⁻, Δ⁺) where Δ⁻ is a set of removed DSL clauses and Δ⁺ is a set of added DSL clauses.

**Definition 4.2** (Incremental compilation): Given a compiled automaton A corresponding to specification S, and a delta Δ, produce A' corresponding to S' = (S \ Δ⁻) ∪ Δ⁺ without full recompilation.

### 4.2 Dependency Tracking

Each automaton state and transition in the compiled output depends on a subset of DSL clauses. This dependency is computed during initial compilation and maintained as a bipartite graph:

**D: Components(A) → 2^Clauses(S)**

When Δ⁻ removes clauses, the affected components are D⁻¹(Δ⁻). When Δ⁺ adds clauses, new components are synthesized and connected.

This is standard incremental computation (Demers et al., 1981; Acar et al., 2002 for self-adjusting computation). The specific data structure (bipartite dependency graph) is the simplest instance of the general framework.

### 4.3 Correctness

**Theorem 4.1** (Incremental correctness): If compile(S) = A and incremental_update(A, Δ) = A', then A' = compile(S'), where S' = (S \ Δ⁻) ∪ Δ⁺.

*Proof approach*: By structural induction on the compilation pipeline. Each phase (parsing → intermediate representation → automaton construction → R-tree initialization) must be shown to be incrementally correct. The key invariant: the dependency graph D is complete (every actual dependency is tracked). If D is complete, then recompiling exactly the affected components produces the same result as full recompilation.

This is a standard correctness argument for incremental systems. The proof obligation is real (one must actually verify the dependency tracking is complete), but the technique is well-established.

### 4.4 Amortized Complexity

Let the specification have n clauses and the compiled automaton have m states.

- Full compilation: O(f(n)) for some function f determined by the compilation algorithm.
- Incremental update for delta of size |Δ|: O(|Δ| × g(n)) where g accounts for propagating changes through dependencies.

In the best case (no cascading dependencies), incremental update is O(|Δ| × polylog(n)). In the worst case (a single clause change affects the entire automaton), it degenerates to O(f(n)). The average case depends on the dependency structure of typical XR interaction patterns.

**Theorem 4.2** (Amortized bound): For a sequence of k single-clause edits applied to a specification of n clauses, the total incremental compilation cost is O(k × n^α + f(n)) for some α < 1 determined by the dependency structure, versus O(k × f(n)) for full recompilation.

This is a standard amortized analysis. The value of α depends on the specific compilation algorithm and DSL structure, and would need to be established empirically or via a structural argument about typical XR interaction patterns.

### 4.5 Novelty Assessment

Incremental compilation is thoroughly studied. The theory of self-adjusting computation (Acar, Blelloch, Harper 2002) provides a general framework that subsumes this use case. The dependency tracking approach is used in build systems (Make, Bazel), incremental parsers (tree-sitter), and incremental type checkers (Salsa/Roslyn).

**Grade: C** — Entirely standard application of known incremental computation techniques. The correctness proof is a proof obligation (someone must do it), but the proof technique is routine. No new mathematical ideas are needed.

---

## 5. Compiler Correctness

### 5.1 What Correctness Means

The compiler translates a DSL specification S (a set of declarative interaction patterns expressed in the SEC formalism from §1) into an executable automaton A (the SEA from §2). Correctness means the automaton faithfully simulates the specification.

**Definition 5.1** (Specification semantics): The denotational semantics of a specification S is a function ⟦S⟧: Trace → {accept, reject} where a Trace is a sequence of (event, scene) pairs.

**Definition 5.2** (Automaton semantics): The operational semantics of a compiled automaton A is a function ⟦A⟧: Trace → {accept, reject} defined by running the automaton on the trace.

**Definition 5.3** (Compiler correctness): The compiler is correct iff for all specifications S and all traces τ: ⟦compile(S)⟧(τ) = ⟦S⟧(τ).

### 5.2 Soundness

**Theorem 5.1** (Soundness): If ⟦compile(S)⟧(τ) = accept, then ⟦S⟧(τ) = accept.

*Meaning*: If the compiled automaton accepts a trace, the specification also accepts it. Equivalently, the compiled automaton does not accept traces that violate the specification. This is the critical safety property.

*Proof approach*: By structural induction on the compilation phases. Define an intermediate semantics for each IR, and show a simulation relation is preserved across each translation step.

1. **Parsing correctness**: The AST faithfully represents the concrete syntax. (Standard; follows from parser generator correctness or hand-proof for recursive descent.)
2. **Desugaring correctness**: Syntactic sugar reductions preserve semantics. (By case analysis on each desugaring rule.)
3. **EC-to-automaton translation**: The core step. Must show that the automaton transitions correspond exactly to the EC axiom firings. The proof structure is a bisimulation between the EC model (set of currently-holding fluents + event history) and the automaton state.
4. **Spatial guard compilation**: Must show that R-tree queries correctly evaluate the spatial predicates. (Follows from R-tree correctness; see §2.3.)

The bisimulation in step 3 is the non-trivial part. It requires showing:
- **State correspondence**: Every reachable automaton state corresponds to a consistent EC fluent valuation, and vice versa.
- **Transition correspondence**: Every automaton transition corresponds to an EC axiom application (Initiates/Terminates + frame axiom), and vice versa.

### 5.3 Completeness

**Theorem 5.2** (Completeness): If ⟦S⟧(τ) = accept, then ⟦compile(S)⟧(τ) = accept.

*Meaning*: The compiled automaton accepts every trace that the specification accepts. No valid behaviors are lost in compilation.

*Caveat*: Completeness may fail if the compilation introduces over-approximations (e.g., merging automaton states for efficiency). If the compiler is an exact translation, completeness follows from the same bisimulation argument as soundness. If it uses abstractions, completeness may not hold, and the failure modes must be characterized.

### 5.4 Full Correctness = Soundness + Completeness

**Theorem 5.3** (Full correctness): If the compiler performs an exact translation (no abstractions), then ⟦compile(S)⟧ = ⟦S⟧.

This follows from Theorems 5.1 and 5.2. The full proof is a compilation correctness proof in the style of Leroy (CompCert, 2006) or Chlipala (2010), but for a much simpler source and target language.

### 5.5 Novelty Assessment

Compiler correctness is a mature field. The proof technique (bisimulation across compilation phases) is standard. CompCert (Leroy 2006, 2009) proved full correctness for a C compiler; CakeML did it for ML. This DSL compiler is vastly simpler than either.

The one wrinkle is the spatial guard compilation step: showing that the continuous-to-discrete sampling (§1.3) composes correctly with the EC-to-automaton translation. Specifically, the compiler must ensure that the sampling-induced event generation (Theorem 1.1) and the automaton guard evaluation (Theorem 2.1) are jointly correct—that no spurious events are introduced and no real events are missed (up to the Lipschitz sampling bound).

**Theorem 5.4** (End-to-end correctness under Lipschitz sampling): For any specification S, scene trace σ with Lipschitz constant L, and sampling rate Δt satisfying the bound from Theorem 1.1, the compiled automaton's behavior on the sampled trace agrees with the specification's semantics on the continuous trace, up to a timing error of at most Δt.

This theorem *composes* several individually-routine results (sampling soundness, EC-to-automaton bisimulation, R-tree correctness) but the composition requires care. It is not deep mathematics, but it is a real proof obligation that someone must discharge.

**Grade: B−** — The individual proof steps are all standard, but the end-to-end composition (Theorem 5.4) that threads the sampling bound through the compiler correctness argument is mildly non-trivial. It is not a new proof *technique*, but it is a new proof *instance* that requires care to get right. The formal statement itself (correctness up to timing error Δt under Lipschitz assumption) is a useful contribution to the artifact.

---

## 6. Summary: Load-Bearing Mathematics

| # | Component | Key Theorem(s) | Grade | Rationale |
|---|-----------|----------------|-------|-----------|
| 1 | Spatial Event Calculus | Sampling soundness (Thm 1.1); Conservative extension (Thm 1.2) | **C+** | Elementary analysis; clean formalization but no new math |
| 2 | R-tree Event Automata | Guard completeness (Thm 2.1) | **C** | Standard guarded automata + standard R-tree |
| 3 | Reachability/Deadlock | Decidability (Thm 3.1); Geometric consistency pruning (Thm 3.2) | **B−** | Verification is off-the-shelf; pruning via metric constraints is potentially interesting |
| 4 | Incremental Compilation | Incremental correctness (Thm 4.1); Amortized bound (Thm 4.2) | **C** | Standard incremental computation |
| 5 | Compiler Correctness | Soundness (Thm 5.1); End-to-end under Lipschitz (Thm 5.4) | **B−** | Individual steps routine; composition through sampling requires care |

### What is Genuinely Novel?

**Honest assessment**: No component in isolation constitutes a new mathematical result. The nearest candidates:

1. **Geometric consistency pruning (§3.2, Thm 3.2)**: Exploiting metric-space structure to prune the abstract state space of spatial event automata. This could yield a genuinely useful bound if the characterization of geometrically consistent predicate valuations C is worked out tightly. The connection to metric constraint satisfaction and clock-zone techniques in timed automata is real but the XR-specific instantiation hasn't been studied. **This is the most promising direction for a novel mathematical contribution.**

2. **End-to-end compiler correctness under Lipschitz sampling (§5.4, Thm 5.4)**: This is not a new technique, but it is a new theorem that must be proved for this specific system. The composition of continuous-to-discrete sampling with discrete compilation correctness is a real proof obligation.

### What is Routine?

Everything else. The Event Calculus extension is a clean application. The R-tree integration is an engineering choice. The verification algorithms are off-the-shelf. The incremental compilation is standard. The individual compiler correctness steps are standard.

### Recommendation for the Crystallization Team

The mathematical contribution of this artifact is primarily in the **integration**—combining spatial indexing, event calculus, automata, and verification into a coherent formal framework—rather than in any single component. The artifact is an **engineering contribution with formal foundations**, not a mathematical contribution per se.

If the team wants to elevate the mathematical novelty:
1. **Pursue the geometric consistency pruning** (Thm 3.2) seriously. Characterize C precisely, prove tight bounds on |C|/|2^P|, and show this yields practical speedups in verification. This could be a legitimate B+ contribution.
2. **Mechanize the compiler correctness proof** (e.g., in Coq or Lean). While the proof content is not novel, a mechanized proof for a spatial-temporal DSL compiler would be a notable artifact in the verified compilation community.
3. **Identify a decidability boundary**: Find the maximal fragment of the DSL for which verification is polynomial-time (rather than PSPACE-complete). If the spatial structure yields a tractability result (analogous to how timed automata with bounded clocks remain decidable), that would be genuinely interesting.

Without these, the artifact is a solid **B-grade engineering contribution** with **C-grade mathematical content**.
