# Choreo XR Interaction Compiler — Algorithm Design Proposal

**Document type**: Algorithm Design Proposal  
**Project**: Choreo — DSL Compiler & Verifier for Spatial-Temporal Interaction Choreographies  
**Stage**: Theory  
**Date**: 2025-07-18  
**Author role**: Algorithm Designer  
**Related documents**: `theory/verification_proposal.md`, `theory/eval_proposal.md`

---

## 0. Overview and Notation

Choreo verifies XR interaction choreographies modelled as **Spatial-Event Automata (SEA)**. A SEA is a tuple:

```
A = (Q, q₀, Σ, P, δ, I, F)
```

where:
- `Q`  — finite set of control states
- `q₀ ∈ Q` — initial state
- `Σ`  — XR event alphabet (gesture, gaze, haptic, proximity, etc.)
- `P`  — finite set of spatial predicates (proximity, containment, orientation)
- `δ ⊆ Q × Guard × Σ × Q` — transition relation; guards are MTL formulas over `P`
- `I`  — MTL invariant labelling (per-state timing constraints)
- `F`  — accepting states or liveness condition

A **guard** is a Boolean combination of:
- **Prox(a, b, r)**: objects a, b are within distance r (Euclidean ℝ³)
- **Inside(a, V)**: object a is inside convex volume V ⊆ ℝ³
- **Orient(a, θ, ε)**: principal axis of a lies within ε of direction θ ∈ S²
- **MTL atoms**: `◻[l,u] φ`, `◇[l,u] φ` with φ a propositional combination of P

A **spatial configuration** is a valuation `v: P → {0,1}`. A configuration is **geometrically realizable** iff there exists a placement of all rigid bodies in ℝ³ consistent with v.

A **spatial-event automata network** is a product `A₁ ‖ A₂ ‖ … ‖ Aₙ` with shared spatial predicates; synchronisation is on shared event labels.

**Key complexity parameters used throughout:**
| Symbol | Meaning |
|--------|---------|
| `n`    | number of rigid-body entities in the scene |
| `k`    | number of spatial predicates `|P|` |
| `m`    | number of distinct distance thresholds |
| `d`    | ambient dimension of spatial configuration space (≤ 6n for rigid bodies in ℝ³) |
| `q`    | maximum number of states per automaton component |
| `N`    | total number of components in the network |
| `w`    | treewidth of the spatial interference graph |
| `b`    | branching factor of the containment DAG |
| `D`    | depth of the containment DAG |

---

## 1. Spatial CEGAR

### 1.1 Background and Motivation

Counterexample-guided abstraction refinement (CEGAR) [Clarke et al. 2003] is the workhorse of software model checking, but it has never been adapted to a **geometric** abstract domain where counterexamples can be refuted by *spatial infeasibility* rather than just predicate precision. The classical CEGAR loop over Boolean predicate abstraction does not distinguish between a counterexample that is temporally spurious (fixed by adding a new Boolean predicate) and one that is *geometrically unrealizable* (fixed by splitting a spatial partition cell). Spatial CEGAR unifies both refinement axes in a single loop using GJK/EPA as the geometric oracle.

### 1.2 Abstract Domain

Let `S = ℝᵈ` be the (6n-dimensional) rigid-body configuration space of the scene, `d = 6n`.

A **spatial partition** is a finite set of closed convex cells `Π = {C₁, …, Cₜ}` such that:
1. `∪ᵢ Cᵢ = S`  (coverage)
2. `int(Cᵢ) ∩ int(Cⱼ) = ∅` for `i ≠ j`  (disjointness of interiors)
3. Each `Cᵢ` is a convex polytope (initially: axis-aligned bounding boxes)

An **abstract state** is a pair `(q, Cᵢ)` where `q ∈ Q` is a control location and `Cᵢ ∈ Π` is a spatial cell. The abstract transition relation over-approximates the concrete one: `(q, Cᵢ) →ᵃ (q', Cⱼ)` whenever the guard of the concrete transition `q →ᵍ q'` *could* hold somewhere in `Cᵢ` (i.e., the guard's polytope and `Cᵢ` are not disjoint).

Each abstract cell `Cᵢ` is assigned a fresh **BDD variable** `xᵢ`. The reachability set is maintained as a BDD over `{xᵢ} × Q`.

### 1.3 GJK/EPA as the Refinement Oracle

Given a spurious abstract counterexample path `π = (q₀,C₀), e₁, (q₁,C₁), …, eₜ, (qₜ,Cₜ)`, the geometric oracle answers:

> **Q**: Is there a concrete trajectory through `C₀, C₁, …, Cₜ` consistent with the guards and the rigid-body kinematics?

The oracle calls GJK (Gilbert–Johnson–Keerthi distance algorithm) between pairs of rigid bodies to compute the minimum separating distance for each guard atom. If the minimum distance reported by GJK exceeds the threshold `r` in a `Prox(a,b,r)` guard that must be satisfied, the counterexample is spurious. EPA (Expanding Polytope Algorithm) then returns the **minimum translation vector** (MTV) that identifies the separating half-space; this half-space is used to split the offending abstract cell.

**Cell splitting**: given cell `Cᵢ` and separating half-space `H = {x | aᵀx ≤ b}` returned by EPA:
```
Cᵢ⁺ = Cᵢ ∩ H
Cᵢ⁻ = Cᵢ ∩ Hᶜ
```
Both halves are convex (intersection of convex sets). The old BDD variable `xᵢ` is replaced by two fresh variables `xᵢ⁺`, `xᵢ⁻`.

### 1.4 Termination Argument

**Lattice of spatial partitions**: Let `𝕃(Π₀)` be the set of all partitions obtainable by refining the initial partition `Π₀` by half-space splits. Each split strictly increases the number of cells by 1. The initial partition has `t₀` cells; after `s` splits it has `t₀ + s` cells.

**Well-founded ordering**: Order partitions by refinement: `Π ≺ Π'` iff `Π'` is obtained from `Π` by at least one split. This is well-founded because:
1. Each cell `Cᵢ` is bounded (XR scenes have finite extent ≤ 10m radius in practice; we fix a bounding box `B ⊂ ℝᵈ`)
2. The half-space returned by EPA has normal derived from the scene geometry, which has finitely many combinatorially distinct orientations (the arrangement of `k` hyperplanes in `ℝᵈ` has at most `O(kᵈ)` cells)
3. Therefore the maximum number of distinct cells reachable by splitting is `|P| · 2^d` (each of the `k ≤ |P|` guard polytopes can be split at most `2^d` times before each cell is contained entirely in or out of every guard)

**Termination bound**: `O(|P| · 2^d)` refinement iterations. For typical XR scenes: `|P| ≤ 200`, `d = 6n` with `n ≤ 5` objects yields `d = 30`; in practice `d` is much smaller because spatial predicates partition a low-dimensional subspace.

### 1.5 BDD Integration

The BDD over `{xᵢ} × Q` represents the abstract reachability set. After a split of cell `Cᵢ` into `Cᵢ⁺` and `Cᵢ⁻`:
1. Every BDD node labelled `xᵢ` is replaced by a node with two children `xᵢ⁺` and `xᵢ⁻`.
2. The abstract transition relation BDD is updated: edges from `xᵢ` are split according to which sub-cell each guard polytope intersects.
3. If a sub-cell `Cᵢ⁺` has an empty intersection with *every* active guard, its variable is pruned (permanently false literal in all BDD clauses).

**Memory**: each BDD variable represents one convex cell; with at most `|P|·2^d` cells, the BDD has at most `O(|P|·2^d)` variables. In practice the BDD is far smaller because most cells are pruned by geometric consistency (§2).

### 1.6 SAT/BMC Encoding

For bounded model checking (BMC) up to depth `L`, the geometric constraints are encoded as clauses:
- **Monotonicity unit propagation**: `Prox(a,b,r₁) ∧ r₁ ≤ r₂ ⇒ Prox(a,b,r₂)` becomes the Horn clause `¬p_{a,b,r₁} ∨ p_{a,b,r₂}` for each pair `r₁ ≤ r₂`.
- **Triangle inequality**: `Prox(a,b,r₁) ∧ Prox(b,c,r₂) ⇒ Prox(a,c,r₁+r₂)` becomes `¬p_{a,b,r₁} ∨ ¬p_{b,c,r₂} ∨ p_{a,c,r₁+r₂}`.
- **Containment**: `Inside(a,V₁) ∧ V₁⊆V₂ ⇒ Inside(a,V₂)` becomes `¬ins_{a,V₁} ∨ ins_{a,V₂}`.

All these clauses are Horn; unit propagation on them runs in `O(k²)` time and can eliminate large portions of the search space before the SAT solver's CDCL loop begins.

### 1.7 Pseudocode

```
Algorithm 1: SpatialCEGAR(A: SEA_network, φ: property) → {verified | counterexample(π)}

Input:
  A       — SEA network A₁ ‖ … ‖ Aₙ
  φ       — safety or MTL reachability property
  Π₀      — initial convex partition of ℝᵈ (e.g., single bounding-box cell)
  budget  — max refinement iterations (default: |P|·2^d)

Output:
  verified          — φ holds on all concrete executions of A
  counterexample(π) — a concrete spurious-free path violating φ

Invariants maintained throughout the loop:
  (I1) The abstract system α(A,Π) over-approximates the concrete system A
  (I2) Every pruned cell has been proved geometrically empty or unreachable
  (I3) The BDD represents the exact abstract reachability set under the current Π

 1: Π ← Π₀
 2: C ← ComputeConsistentSet(P, scene_constraints)     // §2 geometric pruning
 3: bdd ← BuildAbstractBDD(A, Π, C)
 4: loop
 5:   result ← BDD_Reachability(bdd, φ)               // exact BDD fixpoint
 6:   if result = SAFE then
 7:     return verified                                // (I1) ⇒ concrete also safe
 8:   end if
 9:   π_abs ← ExtractAbstractCEX(bdd, φ)              // shortest abstract path
10:   // Check concrete realizability via GJK/EPA oracle
11:   (realizable, refutation) ← GeometricOracle(π_abs, A, Π)
12:   if realizable then
13:     return counterexample(Concretize(π_abs))       // genuine bug
14:   end if
15:   // Spurious: refine the partition
16:   (Cᵢ, H) ← refutation                           // offending cell + separating half-space
17:   (Cᵢ⁺, Cᵢ⁻) ← Split(Cᵢ, H)
18:   Π ← (Π \ {Cᵢ}) ∪ {Cᵢ⁺, Cᵢ⁻}
19:   bdd ← RefineBDD(bdd, xᵢ, xᵢ⁺, xᵢ⁻, Cᵢ⁺, Cᵢ⁻, A, C)
20:   // Prune newly-empty cells
21:   for each C' ∈ {Cᵢ⁺, Cᵢ⁻} do
22:     if GeometricConsistency(C', C) = EMPTY then
23:       bdd ← PruneCell(bdd, C')
24:       Π ← Π \ {C'}
25:     end if
26:   end for
27:   if |Π| > budget then
28:     return inconclusive("refinement budget exceeded")
29:   end if
30: end loop

Subroutine: GeometricOracle(π_abs, A, Π) → (bool, refutation?)
 1: for each step (qᵢ, Cᵢ) →ᵍⁱ (qᵢ₊₁, Cᵢ₊₁) in π_abs do
 2:   for each guard atom gᵢⱼ = Prox(aⱼ, bⱼ, rⱼ) in the negation of guard gᵢ do
 3:     dist, MTV ← GJK_EPA(aⱼ, bⱼ, Cᵢ)
 4:     if dist > rⱼ then
 5:       return (false, (Cᵢ, half-space defined by MTV))
 6:     end if
 7:   end for
 8:   for each guard atom Inside(aⱼ, Vⱼ) in ¬gᵢ do
 9:     sep ← SupportFunctionSeparation(aⱼ, Vⱼ, Cᵢ)   // via LP
10:     if sep ≠ ∅ then
11:       return (false, (Cᵢ, sep))
12:     end if
13:   end for
14: end for
15: return (true, ∅)
```

### 1.8 Correctness Argument

**Theorem 1.1 (Soundness of Spatial CEGAR)**: If `SpatialCEGAR` returns `verified`, then `A ⊨ φ`.

*Proof sketch*: By (I1), the abstract system over-approximates A. If `BDD_Reachability` reports SAFE, no abstract execution reaches a φ-violating state. Since the abstraction is an over-approximation, no concrete execution can either.

**Theorem 1.2 (Refutation Soundness)**: If `GeometricOracle` returns `(false, (Cᵢ, H))`, then no concrete configuration in `Cᵢ` satisfies the relevant guard atom.

*Proof sketch*: GJK computes the minimum distance between two convex sets exactly (over ℝᵈ; numerical issues addressed in §6). If `dist > r`, the distance constraint cannot be satisfied anywhere in `Cᵢ`. The separating half-space `H` returned by EPA correctly partitions `Cᵢ`.

**Theorem 1.3 (Progress)**: Each iteration either terminates (returns verified or counterexample) or strictly refines Π (|Π| increases by 1 or decreases by pruning, net effect: at least one new cell is strictly geometrically smaller than its parent).

*Proof sketch*: Each refinement splits a cell using a half-space strictly separating a realized point and the boundary of a guard polytope; no split is trivial (the half-space passes through the interior of `Cᵢ`).

### 1.9 Complexity

| Operation | Complexity |
|-----------|-----------|
| BDD reachability fixpoint | `O(|Q|² · |Π|² · |P|)` per BDD operation; `O(q²)` iterations |
| GeometricOracle (one CEX) | `O(|π| · k · GJK)` = `O(L · k · d · log(1/ε))` |
| RefineBDD (one split) | `O(|P| · |Π|)` BDD node replacements |
| Total (worst case) | `O(|P|·2^d · (q² · |P|·2^d + L·k·d))` |
| Practical (d≤10, |P|≤200) | `O(2^10 · 200 · q²)` ≈ tractable |

---

## 2. Geometric Pruning

### 2.1 Purpose

Before CEGAR begins (and as a preprocessing step for any analysis), we compute the **geometrically consistent** subset `C ⊆ 2^P` of spatial-predicate valuations. Most of the `2^k` Boolean combinations of `k` spatial predicates are geometrically unrealizable (e.g., `Prox(a,b,1) ∧ ¬Prox(a,b,2)` is impossible since proximity is monotone in radius). Pruning these configurations up front reduces the BDD size and the state space explored by CEGAR.

### 2.2 Inference Rules

**Monotonicity (M)**: Proximity predicates are monotone increasing in radius:
```
Prox(a,b,r₁) ∧ (r₁ ≤ r₂)  ⟹  Prox(a,b,r₂)
¬Prox(a,b,r₂) ∧ (r₁ ≤ r₂) ⟹  ¬Prox(a,b,r₁)
```

**Triangle Inequality (T)**: Euclidean distance satisfies the triangle inequality:
```
Prox(a,b,r₁) ∧ Prox(b,c,r₂)  ⟹  Prox(a,c, r₁+r₂)
```
Negation form (for pruning):
```
¬Prox(a,c,r₃) ∧ Prox(b,c,r₂) ∧ (r₁+r₂ ≥ r₃) ⟹  ¬Prox(a,b,r₁)
```

**Containment (C)**: The spatial containment relation `V₁ ⊆ V₂` (precomputed from scene graph) gives:
```
Inside(a,V₁) ∧ (V₁ ⊆ V₂)  ⟹  Inside(a,V₂)
¬Inside(a,V₂) ∧ (V₁ ⊆ V₂) ⟹  ¬Inside(a,V₁)
```

**Disjointness (D)**: If volumes `V₁` and `V₂` are spatially disjoint:
```
Inside(a,V₁) ∧ Inside(a,V₂)  ⟹  ⊥   (contradiction → prune)
```

**Self-distance (S)**:
```
Prox(a,a,0)  ≡  ⊤   (always true; can be simplified away)
¬Prox(a,b,r) ∧ Prox(b,a,r) ⟹ ⊥  (symmetry: predicates are symmetric)
```

### 2.3 Constraint Propagation Algorithm

The algorithm runs worklist-based unit propagation on the above Horn-clause encoding. Each predicate `p ∈ P` has a ternary state: `{TRUE, FALSE, UNKNOWN}`.

```
Algorithm 2: ComputeConsistentSet(P: predicate_set, Γ: scene_constraints) → C ⊆ 2^P

Input:
  P   — set of spatial predicates {Prox(a,b,r), Inside(a,V), Orient(a,θ,ε)}
  Γ   — scene constraints: volume containment DAG, body radii, initial placement

Output:
  C   — set of geometrically consistent valuations (as a compact BDD)

Data structures:
  val: P → {TRUE, FALSE, UNKNOWN}  // current propagated value
  Q: queue of (predicate, value) pairs to propagate

 1: C ← {}
 2: val ← all UNKNOWN
 3: // Apply scene-graph invariants
 4: for each (V₁,V₂) ∈ Γ.containment_edges do      // V₁ ⊆ V₂
 5:   AddContainmentConstraints(P, V₁, V₂)
 6: end for
 7: for each (V₁,V₂) ∈ Γ.disjoint_pairs do
 8:   AddDisjointnessConstraints(P, V₁, V₂)
 9: end for
10: // Main propagation loop over all 2^|P| candidate valuations (implicit via BDD)
11: bdd ← AllTrue_BDD(P)                            // start with full 2^|P|
12: // Apply monotonicity arcs as unit propagation
13: for each pair (r₁ < r₂) and entities (a,b) do
14:   clause ← (¬Prox(a,b,r₁) ∨ Prox(a,b,r₂))    // monotonicity Horn clause
15:   bdd ← bdd ∧ clause
16: end for
17: // Apply triangle inequality arcs
18: for each triple (a,b,c) and thresholds (r₁,r₂,r₃) with r₁+r₂ ≤ r₃ do
19:   clause ← (¬Prox(a,b,r₁) ∨ ¬Prox(b,c,r₂) ∨ Prox(a,c,r₃))
20:   bdd ← bdd ∧ clause
21: end for
22: // Apply containment arcs
23: for each (a, V₁ ⊆ V₂) do
24:   clause ← (¬Inside(a,V₁) ∨ Inside(a,V₂))
25:   bdd ← bdd ∧ clause
26: end for
27: // Apply disjointness
28: for each (a, V₁ ⊥ V₂) do
29:   clause ← (¬Inside(a,V₁) ∨ ¬Inside(a,V₂))
30:   bdd ← bdd ∧ clause
31: end for
32: C ← {v ∈ 2^P | bdd(v) = 1}                     // consistent set (BDD-encoded)
33: return C

Subroutine: AddContainmentConstraints(P, V₁, V₂)
 1: for each entity a in scene do
 2:   Enqueue(Q, (Inside(a,V₂), implies-by: Inside(a,V₁)))
 3: end for
```

### 2.4 Cardinality Bound

**Theorem 2.1**: `|C| ≤ O(b^D · m^(n²))` where `b` is the branching factor of the containment DAG, `D` its depth, `n` the number of entities, and `m` the number of distinct proximity thresholds.

*Proof sketch*:
- **Proximity degrees of freedom**: For `n` entities and `m` thresholds, the proximity predicates form an order-theoretic structure. By the monotonicity rule, each pair `(a,b)` can only be in one of `m+1` intervals `[r₀,r₁), [r₁,r₂), …, [rₘ,∞)`. The number of consistent proximity valuations for one pair is `m+1`, so for `n²/2` pairs it is `(m+1)^(n²/2) = O(m^(n²))`.
- **Containment degrees of freedom**: The containment DAG has at most `b^D` maximal antichains (Dilworth's theorem); each entity can occupy at most one node in each antichain layer. The containment pruning reduces the inside-predicate space from `2^(n·|V|)` to `b^(D·n)`.
- **Combined**: `|C| ≤ b^(D·n) · m^(n²)`. For typical XR scenes (`n≤5, m≤5, b≤3, D≤4`): `|C| ≤ 3^20 · 5^25 ≈ 3.5·10⁹ · 3·10¹⁷` — but the BDD representation is polynomial in `|P|` (number of BDD nodes, not number of valuations), so the BDD size is `O(|P|²)` after propagation.

**Corollary 2.1 (State-space reduction)**: The geometric pruning reduces the automaton state space from `|Q| · 2^k` to at most `|Q| · b^(D·n) · m^(n²)`. In practice, the BDD-encoded `C` is compressed further by shared structure.

### 2.5 Correctness

**Theorem 2.2 (Soundness)**: Every valuation eliminated by `ComputeConsistentSet` is geometrically unrealizable.

*Proof*: Each inference rule is a valid logical consequence of the Euclidean metric axioms or the scene-graph geometry. Horn clause resolution on valid clauses yields only valid consequences; the falsified valuations violate at least one clause that is a sound geometric axiom. ∎

**Theorem 2.3 (Incompleteness)**: `ComputeConsistentSet` may return valuations that are geometrically unrealizable.

*Proof by example*: Consider three entities `a,b,c` in ℝ¹ with `Prox(a,b,1) ∧ Prox(b,c,1) ∧ ¬Prox(a,c,3)`. The triangle inequality gives `Prox(a,c,2)`, which is consistent with `¬Prox(a,c,3)`. But in ℝ¹ only, the configuration forces `dist(a,c) ≤ 2 < 3`, so the valuation is realizable. In higher-dimensional configurations this is more complex; polynomial arithmetic checks would be needed for completeness, which we do not perform. ∎

Incompleteness is acceptable: false positives (keeping unrealizable valuations) lead to more abstract states explored by CEGAR, not to missed bugs.

---

## 3. Spatial Type Checking

### 3.1 Overview

The Choreo type system assigns **spatial types** to interaction expressions. A spatial type `τ` describes the set of spatial configurations under which an expression is safe to execute. The central operation is **spatial subtyping**: `τ₁ ≤ τ₂` (read: "τ₁ is more specific than τ₂", or equivalently "τ₁ is safe wherever τ₂ is safe"). We need this to be decidable for the common case (convex polytopes) and at worst NP-complete for CSG scenes.

### 3.2 Spatial Type Grammar

```
τ ::= ⊤                         -- universal type (any spatial configuration)
    | ⊥                         -- empty type (no configuration)
    | Prox(a, b, r)             -- proximity constraint
    | Inside(a, V)              -- containment constraint
    | Orient(a, θ, ε)           -- orientation constraint
    | τ₁ ∧ τ₂                  -- intersection (conjunction)
    | τ₁ ∨ τ₂                  -- union (disjunction)
    | ¬τ                        -- complement
    | μX.τ(X)                   -- least fixed point (for recursive types)
    | νX.τ(X)                   -- greatest fixed point (coinductive)
    | τ₁ ⊗ τ₂                  -- spatial product (both a and b satisfy resp. types)
    | ∃x.τ                      -- spatial existential (some configuration satisfies τ)
```

The **ground types** `Prox`, `Inside`, `Orient` denote convex polytopes in `ℝ^(6n)` (the rigid-body configuration space): each ground-type atom is a half-space or intersection of half-spaces.

### 3.3 Coinductive Subtyping

Subtyping for recursive types is defined **coinductively**: `τ₁ ≤ τ₂` holds if we can establish it by *unfolding* the definitions without finding a counterexample.

The formal subtyping judgment `Γ ⊢ τ₁ ≤ τ₂` uses an assumption set `Γ` of pairs `(σ₁, σ₂)` already assumed to be in the subtype relation (to handle coinduction).

**Key inference rules**:
```
——————————————————————    (Refl)
Γ ⊢ τ ≤ τ

(τ₁,τ₂) ∈ Γ
———————————    (Assume)
Γ ⊢ τ₁ ≤ τ₂

Γ ⊢ τ₁ ≤ σ    Γ ⊢ σ ≤ τ₂
——————————————————————————    (Trans)
Γ ⊢ τ₁ ≤ τ₂

LP_FEASIBLE({¬τ₁} ∪ {τ₂}) = UNSAT    (ground types as LP constraints)
——————————————————————————————————————    (LP)
Γ ⊢ τ₁ ≤ τ₂

Γ,(τ₁,τ₂) ⊢ τ₁[μX.τ₁/X] ≤ τ₂[νX.τ₂/X]
——————————————————————————————————————    (Unfold-μν)
Γ ⊢ μX.τ₁ ≤ νX.τ₂
```

### 3.4 Quantifier Elimination

For mixed spatial-temporal types (types that carry MTL guards), the subtyping check requires quantifier elimination over the reals. We use **Lasserre's SOS relaxation** for polynomial inequalities (covering the common case of ellipsoidal proximity zones) and **Fourier-Motzkin elimination** for the strictly linear case (axis-aligned bounding boxes, half-space containment).

The QE step threads through the coinduction: at each unfolding step, the accumulated LP/SOS system gains new constraints from unfolding the recursive type. Decidability follows because the accumulated system's size grows by at most `|τ₁| + |τ₂|` constraints per unfolding step, and the system is infeasible (i.e., subtyping holds) when the dimension of the feasible polytope reaches zero.

### 3.5 Pseudocode

```
Algorithm 3: SpatialSubtype(τ₁: type, τ₂: type) → bool

Input:
  τ₁, τ₂ — spatial types (possibly recursive)

Output:
  true iff τ₁ ≤ τ₂ (τ₁ is a spatial subtype of τ₂)

Γ: assumption set (pairs of types), initially ∅
lp: LP/SOS solver instance

 1: function SpatialSubtype(τ₁, τ₂):
 2:   return Sub(τ₁, τ₂, ∅)

 3: function Sub(τ₁, τ₂, Γ):
 4:   // Base cases
 5:   if τ₁ = ⊥ then return true          // empty type is subtype of everything
 6:   if τ₂ = ⊤ then return true          // everything is subtype of universal
 7:   if τ₁ = τ₂ then return true          // reflexivity
 8:   if (τ₁, τ₂) ∈ Γ then return true    // coinductive assumption
 9:   // Ground type case: reduce to LP feasibility
10:   if τ₁ and τ₂ are both ground-type conjunctions then
11:     lp_system ← EncodeGroundTypes(τ₁) ∪ NegateGroundType(τ₂)
12:     return LP_INFEASIBLE(lp_system)    // τ₁ ∧ ¬τ₂ = ∅ iff τ₁ ≤ τ₂
13:   end if
14:   // Structural cases
15:   match (τ₁, τ₂):
16:   | (τ₁ₐ ∧ τ₁ᵦ, _) →
17:       return Sub(τ₁ₐ, τ₂, Γ) ∨ Sub(τ₁ᵦ, τ₂, Γ)    // only one branch needed
18:   | (_, τ₂ₐ ∧ τ₂ᵦ) →
19:       return Sub(τ₁, τ₂ₐ, Γ) ∧ Sub(τ₁, τ₂ᵦ, Γ)
20:   | (τ₁ₐ ∨ τ₁ᵦ, _) →
21:       return Sub(τ₁ₐ, τ₂, Γ) ∧ Sub(τ₁ᵦ, τ₂, Γ)
22:   | (_, τ₂ₐ ∨ τ₂ᵦ) →
23:       return Sub(τ₁, τ₂ₐ, Γ) ∨ Sub(τ₁, τ₂ᵦ, Γ)
24:   | (μX.σ₁, νY.σ₂) →
25:       Γ' ← Γ ∪ {(τ₁, τ₂)}
26:       // Unfold: substitute μX.σ₁ for X in σ₁, νY.σ₂ for Y in σ₂
27:       τ₁' ← Unfold_μ(σ₁)    // σ₁[μX.σ₁/X]
28:       τ₂' ← Unfold_ν(σ₂)    // σ₂[νY.σ₂/Y]
29:       return Sub(τ₁', τ₂', Γ')
30:   | (μX.σ₁, τ₂) →
31:       return Sub(σ₁[μX.σ₁/X], τ₂, Γ ∪ {(τ₁,τ₂)})
32:   | (τ₁, νY.σ₂) →
33:       return Sub(τ₁, σ₂[νY.σ₂/Y], Γ ∪ {(τ₁,τ₂)})
34:   | (∃x.σ₁, τ₂) →
35:       // QE: project out x from σ₁, then check subtyping
36:       σ₁' ← FourierMotzkin_Eliminate(σ₁, x)
37:       return Sub(σ₁', τ₂, Γ)
38:   | _ → return false    // no structural rule applies

Subroutine: EncodeGroundTypes(τ) → LP constraints
 1: constraints ← {}
 2: for each atom a ∈ τ do
 3:   match a:
 4:   | Prox(u, v, r)  → add ‖pos(u)−pos(v)‖² ≤ r² (linearised as supporting half-spaces)
 5:   | Inside(u, V)   → add the H-representation of V applied to pos(u)
 6:   | Orient(u,θ,ε)  → add cos(ε) ≤ axis(u)·θ (linearised for small ε)
 7: end for
 8: return constraints
```

### 3.6 Termination

**Theorem 3.1**: `SpatialSubtype` terminates.

*Proof*: Define the **measure** `μ(τ₁, τ₂, Γ)` = (number of type constructors in `τ₁` + `τ₂` not yet in `Γ`) + (dimension of accumulated LP feasible set). Each recursive call either:
- (a) Reduces the number of unresolved type constructors by adding `(τ₁, τ₂)` to `Γ` (coinductive step), or
- (b) Reduces the LP dimension by adding a new constraint (QE step), or
- (c) Terminates immediately (base case).

The number of type constructors is finite (types are finitely presented). The LP dimension is at most `6n` (the dimension of the configuration space), so decreases at most `6n` times. Together: at most `(|τ₁|+|τ₂|) · 6n` recursive calls before termination. ∎

### 3.7 Complexity

| Case | Complexity |
|------|------------|
| Ground types only (LP) | `O(|τ|^{1.5})` per LP call (interior-point method) |
| Convex polytope types | `O((|τ₁|+|τ₂|) · 6n · LP)` = `O(|τ|² · n^{1.5})` |
| Recursive types (μ/ν) | `O(|τ|² · n^{1.5})` unfoldings before fixpoint |
| CSG (non-convex) | NP-complete (reduction to 3-SAT via bounded-depth CSG SAT [Hoffmann 1989]) |

---

## 4. Compositional Verification

### 4.1 Motivation

Verifying the full product `A₁ ‖ A₂ ‖ … ‖ Aₙ` directly is `O(qᴺ)` in the number of states, which is intractable for `N > 5`. We exploit the **sparse spatial interference** structure: in a typical XR scene, automaton `Aᵢ` spatially interferes with `Aⱼ` only if they share a spatial predicate involving a common rigid body. The **spatial interference graph** `G_I` is sparse (planar in practice), and its treewidth is small.

### 4.2 Spatial Interference Graph

**Definition**: The spatial interference graph `G_I = (V, E)` has:
- `V = {A₁, …, Aₙ}` — one node per automaton component
- `(Aᵢ, Aⱼ) ∈ E` iff Aᵢ and Aⱼ share at least one spatial predicate atom

**Observation**: For `k` total spatial predicates and `n` entities, the average degree of `G_I` is `O(k/N)`. For XR scenes with local interaction patterns (each interaction involves ≤ 3 entities), `G_I` is nearly planar with treewidth `w ≤ O(√N)` (by the planar separator theorem).

### 4.3 Tree Decomposition and Bag Verification

**Tree decomposition** of `G_I`: a tree `T = (V_T, E_T)` where each node `t ∈ V_T` has a bag `B_t ⊆ V` such that:
1. Each `Aᵢ ∈ V` appears in at least one bag
2. Each edge `(Aᵢ,Aⱼ) ∈ E` has a bag containing both
3. For each `Aᵢ`, the bags containing `Aᵢ` form a connected subtree

Treewidth `w` = max bag size − 1. We use the Bodlaender-Kloks `O(f(w)·N)` algorithm to compute a width-`w` decomposition.

**Bag verification**: For each bag `B_t`, verify `‖_{Aᵢ ∈ B_t} Aᵢ` independently against a bag-local property derived from the global property by **abstraction over separator predicates**.

**Assume-Guarantee at separators**: For each edge `(t, t')` in `T`, the separator is `B_t ∩ B_{t'}`. The component in `B_t \ B_{t'}` makes *assumptions* about the behaviour of components in `B_{t'} \ B_t` (via over-approximating summaries) and *guarantees* its own behaviour. The circular A-G reasoning of [Giannakopoulou & Pasareanu 2012] is applied at each separator.

### 4.4 Pseudocode

```
Algorithm 4: CompositionalVerify(A: SEA_network, φ: property) → {verified | counterexample | unknown}

Input:
  A = A₁ ‖ … ‖ Aₙ   — SEA network
  φ                   — global property (safety or reachability)

Output:
  verified              — φ holds on A
  counterexample(π)     — concrete counterexample
  unknown               — A-G iteration did not converge

 1: G_I ← BuildInterferenceGraph(A)
 2: (T, {B_t}) ← TreeDecomposition(G_I)            // width-w decomposition
 3: C ← ComputeConsistentSet(P, scene_constraints)  // geometric pruning (§2)
 4: // Bottom-up pass: compute summaries
 5: for each leaf node t in T (post-order) do
 6:   result_t ← VerifyBag(B_t, φ_t, C, ∅)         // φ_t = projection of φ onto B_t
 7:   if result_t = counterexample(π) then
 8:     return counterexample(π)                     // early exit
 9:   end if
10:   summary_t ← ExtractSummary(result_t, B_t)     // interface automaton
11: end for
12: // Bottom-up: propagate summaries
13: for each internal node t in T (post-order) do
14:   Σ_t ← {summary_{t'} | t' is child of t}
15:   result_t ← VerifyBag(B_t, φ_t, C, Σ_t)
16:   if result_t = counterexample(π) then
17:     π_concrete ← TryConcreteRealization(π, A)
18:     if π_concrete ≠ ∅ then return counterexample(π_concrete)
19:     else /* spurious at composition boundary */ RefineAssumptions(t)
20:   end if
21:   summary_t ← ExtractSummary(result_t, B_t)
22: end for
23: // Root check
24: result_root ← VerifyBag(B_root, φ, C, Σ_root)
25: if result_root = SAFE then return verified
26: return unknown

Subroutine: VerifyBag(B, φ_B, C, Σ: assumptions) → result
 1: // Build product automaton for this bag, restricted to consistent set C
 2: A_bag ← RestrictedProduct({Aᵢ | Aᵢ ∈ B}, C)
 3: // Incorporate assumptions as invariant constraints
 4: for each assumption (Aⱼ, inv_j) ∈ Σ do
 5:   A_bag ← A_bag ∧ inv_j    // conjoin invariant automaton
 6: end for
 7: // Local CEGAR or direct BDD reachability
 8: return SpatialCEGAR(A_bag, φ_B)    // Algorithm 1

Subroutine: ExtractSummary(result, B_t) → summary
 1: if result = verified then
 2:   return InvariantAutomaton(B_t ∩ B_parent)  // project reachability onto separator
 3: else
 4:   return OverApproximation(B_t ∩ B_parent)   // conservative: all states reachable
 5: end if
```

### 4.5 Complexity

**Theorem 4.1**: `CompositionalVerify` runs in time `O(k · q^(w+1) · |C|)` where `w` is the treewidth, `q` the maximum per-component state count, and `k` the number of spatial predicates.

*Proof sketch*:
- Each bag has at most `w+1` components, so `VerifyBag` works on a product automaton of at most `q^(w+1)` states, intersected with the consistent set `C`.
- There are `O(N)` bags in the tree decomposition.
- The A-G iteration at each separator converges in `O(q²)` steps (fixed-point of the assumption lattice, bounded by the state count of the separator automaton).
- Total: `O(N · q^(w+1) · |C|)`. Including predicate dependency: multiply by `k` for guard evaluation.

**Corollary 4.1**: For XR scenes with `w ≤ 3` (empirically observed for ≤ 8-zone scenes), `q ≤ 50`, `|C| ≤ 10^6`: total states `≤ 50^4 · 10^6 ≈ 6 · 10^{12}`, which is large but manageable with BDD representation (BDD node count is far smaller due to shared structure).

---

## 5. EC→Automata Compilation

### 5.1 Overview

The Choreo DSL uses **Event Choreography (EC) expressions** — a regular-expression-like language with spatial-temporal guards. Compilation proceeds in three stages:
1. **Thompson construction** (EC → NFA-with-guards)
2. **On-the-fly product** (NFA-with-guards × spatial-guard R-tree → guarded NFA)
3. **Guard compilation** (spatial guards → R-tree query plans)

### 5.2 EC Grammar

```
ec ::= ε                        -- empty choreography
     | e                        -- single event e ∈ Σ
     | ec₁ · ec₂               -- sequential composition
     | ec₁ | ec₂               -- choice
     | ec*                      -- Kleene star
     | ec₁ ‖ ec₂               -- parallel composition
     | [φ] ec                  -- guard (spatial-temporal precondition φ)
     | ec [φ]                  -- post-condition assertion
     | ec within [l, u]         -- timing constraint (MTL)
     | ec @ zone(V)            -- spatial context (must occur inside volume V)
```

### 5.3 Thompson-Style Construction with Spatial-Temporal Guards

The Thompson construction [Thompson 1968] builds an NFA from an EC expression by structural recursion, augmented with guard transitions.

```
Algorithm 5a: Thompson_EC(ec) → NFA-with-guards

Input:  ec — an EC expression
Output: (q_start, q_accept, δ) — NFA fragment

 1: match ec:
 2: | ε →
 3:   q_s, q_a ← fresh states
 4:   δ ← {(q_s, ε, q_a)}
 5:   return (q_s, q_a, δ)
 6:
 7: | e  →   // single event
 8:   q_s, q_a ← fresh states
 9:   δ ← {(q_s, e, q_a)}
 10:  return (q_s, q_a, δ)
11:
12: | ec₁ · ec₂ →    // sequential
13:  (s₁, a₁, δ₁) ← Thompson_EC(ec₁)
14:  (s₂, a₂, δ₂) ← Thompson_EC(ec₂)
15:  δ ← δ₁ ∪ δ₂ ∪ {(a₁, ε, s₂)}    // ε-merge
16:  return (s₁, a₂, δ)
17:
18: | ec₁ | ec₂ →    // choice
19:  (s₁, a₁, δ₁) ← Thompson_EC(ec₁)
20:  (s₂, a₂, δ₂) ← Thompson_EC(ec₂)
21:  q_s, q_a ← fresh states
22:  δ ← δ₁ ∪ δ₂ ∪ {(q_s,ε,s₁),(q_s,ε,s₂),(a₁,ε,q_a),(a₂,ε,q_a)}
23:  return (q_s, q_a, δ)
24:
25: | ec* →    // Kleene star
26:  (s₁, a₁, δ₁) ← Thompson_EC(ec₁)
27:  q_s, q_a ← fresh states
28:  δ ← δ₁ ∪ {(q_s,ε,s₁),(q_s,ε,q_a),(a₁,ε,s₁),(a₁,ε,q_a)}
29:  return (q_s, q_a, δ)
30:
31: | [φ] ec →    // guarded prefix
32:  (s₁, a₁, δ₁) ← Thompson_EC(ec)
33:  q_s ← fresh state
34:  δ ← δ₁ ∪ {(q_s, ε[φ], s₁)}    // ε-transition labelled with guard φ
35:  return (q_s, a₁, δ)
36:
37: | ec within [l, u] →    // timing
38:  (s₁, a₁, δ₁) ← Thompson_EC(ec)
39:  Annotate(s₁, clock_reset)
40:  Annotate(a₁, invariant: clock ≤ u, guard_on_entry: clock ≥ l)
41:  return (s₁, a₁, δ₁)
42:
43: | ec @ zone(V) →    // spatial context
44:  (s₁, a₁, δ₁) ← Thompson_EC(ec)
45:  for each transition (q, e, q') ∈ δ₁ do
46:    AddGuardConjunct(q → q', Inside(actor, V))    // actor = focal entity
47:  end for
48:  return (s₁, a₁, δ₁)
```

### 5.4 On-the-Fly Product Composition

After Thompson construction, the NFA-with-guards is composed with the **spatial consistency BDD** via lazy (on-the-fly) product construction. States are enumerated only as they are reached during exploration; unreachable states are never constructed.

```
Algorithm 5b: OnTheFlyProduct(nfa, bdd_C) → guarded_DFA

Input:
  nfa   — NFA from Thompson_EC (with ε-transitions and guard labels)
  bdd_C — BDD of consistent spatial configurations (from §2)

Output:
  guarded_DFA — deterministic automaton with spatial guards compiled to BDD queries

 1: q₀ ← ε-closure({nfa.start}) ⊗ {bdd_C.root}
 2: worklist ← {q₀}
 3: visited ← {q₀}
 4: dfa ← {}
 5: while worklist ≠ ∅ do
 6:   (Q_nfa, bdd_node) ← pop(worklist)
 7:   for each event e ∈ Σ do
 8:     // Compute successor NFA states reachable on e
 9:     Q_nfa' ← ε-closure(Post(Q_nfa, e))
10:     // Compute guard conjunction for this transition
11:     φ_e ← ∧{φ | (q,e[φ],q') is a transition used to reach some q'∈Q_nfa'}
12:     // Intersect spatial guard with consistent set
13:     bdd_node' ← BDD_AND(bdd_node, Encode(φ_e))
14:     if bdd_node' ≠ FALSE then
15:       q' ← (Q_nfa', bdd_node')
16:       dfa.AddTransition(current=(Q_nfa,bdd_node), on=e, to=q')
17:       if q' ∉ visited then
18:         visited ← visited ∪ {q'}
19:         worklist ← worklist ∪ {q'}
20:       end if
21:     end if
22:     // Prune: BDD_AND = FALSE means guard is inconsistent → dead transition
23:   end for
24: end while
25: return dfa
```

**Lazy enumeration**: The product automaton is never fully materialised. States are created on demand; the BDD intersection at step 13 immediately prunes dead branches via the geometric consistency BDD.

### 5.5 Guard Compilation to R-tree Query Plans

Spatial guards of the form `Prox(a, b, r)` or `Inside(a, V)` are compiled to **R-tree range queries** at runtime, enabling O(log n) spatial query evaluation instead of O(n) linear scan.

```
Algorithm 5c: CompileGuards(guarded_dfa) → r_tree_indexed_dfa

Input:  guarded_dfa — DFA with spatial guard BDDs
Output: DFA with guards replaced by R-tree query plans

 1: rtree ← RTree(max_entries = 8)
 2: for each transition t = (q, e[φ], q') in guarded_dfa do
 3:   for each atom a ∈ φ do
 4:     match a:
 5:     | Prox(u, v, r) →
 6:       bbox ← BoundingBox(u.center, r)             // AABB in ℝ³
 7:       rtree.Insert(bbox, (t, a))
 8:     | Inside(u, V) →
 9:       bbox ← V.aabb                               // AABB of volume V
10:       rtree.Insert(bbox, (t, a))
11:     | Orient(u, θ, ε) →
12:       bbox ← OrientationAABB(θ, ε)               // bounding box in SO(3)
13:       rtree.Insert(bbox, (t, a))
14:   end for
15: end for
16: // Replace each guard with an R-tree query plan
17: for each state q with pending transitions do
18:   q.query_plan ← rtree.RangeQuery(q.spatial_context_bbox)
19: end for
20: return dfa with rtree

Runtime guard evaluation (called on each XR frame):
 1: function EvalGuard(q, scene_snapshot):
 2:   candidates ← rtree.Search(scene_snapshot.aabb)  // O(log n + k)
 3:   for each (t, a) ∈ candidates do
 4:     if EvalAtom(a, scene_snapshot) then
 5:       AddEnabledTransition(q, t)
 6:   end for
```

**Complexity**: Thompson construction runs in `O(|ec|)` states and transitions. On-the-fly product exploration is `O(2^|Q_nfa| · |bdd_C|)` in the worst case; `|bdd_C|` is bounded by the BDD size from §2, typically `O(|P|²)`. Guard compilation to R-tree: `O(|P| log |P|)` for insertion; query: `O(log |P| + k)` per frame.

---

## 6. Implementation Strategy

### 6.1 Library Stack

| Function | Library | Rationale |
|----------|---------|-----------|
| BDD operations | **CUDD 3.0** (C, via Rust FFI) | Industry standard; supports BDD/ADD/ZDD; 64-bit node pool |
| SAT solving | **CaDiCaL 1.9** (C++, via Rust FFI) | State-of-the-art CDCL; supports DRAT proofs for soundness validation |
| R-tree spatial index | **rstar 0.12** (pure Rust) | Zero-copy; supports bulk-loading; pluggable distance metrics |
| GJK/EPA geometry | **parry3d 0.17** (Rust) | Production-quality; ncollide lineage; supports convex decomposition |
| LP solver (type checking) | **clarabel 0.7** (Rust) | Interior-point; handles SOS constraints; sparse matrix support |
| Tree decomposition | **treewidth 0.3** (Rust) | Bodlaender-Kloks O(f(w)N) algorithm |
| MTL monitoring | **rtamt** (Python, offline analysis) + custom Rust runtime | Offline validation of timing constraints |

### 6.2 Memory Budget for 20-Zone Verification

Target: verify a 20-zone XR scene (n=5 entities, k=200 predicates, N=10 components, q=50 states each) within **2 GB RAM**.

| Component | Estimated memory | Notes |
|-----------|----------------|-------|
| CUDD BDD node pool | 512 MB | `2^23` nodes × 16 bytes; CUDD default is 2^23 |
| Abstract state cache | 256 MB | `q^(w+1)` = `50^4` = 6.25M states × 40 bytes |
| R-tree spatial index | 64 MB | 200 predicates × 8 MB per level (3 levels) |
| CaDiCaL clause db | 128 MB | k² = 40k clauses × 3 literals × 4 bytes |
| Consistent-set BDD | 128 MB | After geometric pruning; empirically ≤ 2^16 nodes |
| GJK/EPA working set | 32 MB | 5 bodies × 10k face convex hulls × 64 bytes/face |
| Geometric partition Π | 64 MB | ≤ 10^4 cells × 6KB per polytope H-representation |
| Stack + misc | 128 MB | Recursion depth ≤ 1000; type-checker BFS queue |
| **Total** | **~1.3 GB** | Headroom ≈ 700 MB for BDD explosion |

**BDD explosion mitigation**: If CUDD's node count exceeds `2^22`, trigger garbage collection (CUDD's ref-count GC) and compress the consistent-set BDD via cofactor minimisation. If after GC the BDD still exceeds budget, fall back to explicit-state BFS with a hash-set frontier (linear in visited states, uses remaining headroom).

### 6.3 Critical Paths

#### Path 1: CEGAR Refinement Loop

The dominant cost is the BDD reachability fixpoint (Algorithm 1, line 5). Each fixpoint iteration touches every BDD node; with `2^23` nodes and 50 iterations per CEGAR round, and 100 refinement rounds:

```
50 (fixpoint iters) × 100 (refinement rounds) × 2^23 (BDD nodes) × 10ns/op
= 50 × 100 × 8.4M × 10ns ≈ 420 seconds  [worst case]
```

**Mitigation**: 
- Incremental BDD update (line 19): on split, only the sub-BDD rooted at `xᵢ` needs recomputation; saves `1 - 1/|Π|` fraction of BDD work per round.
- Early termination: if the post-image BDD stabilises within 5 iterations (common for safety properties), cut the fixpoint early.
- CEGAR order heuristic: prioritise splits that prune the largest number of abstract transitions (measured by BDD node count reduction).

Target: ≤ 60 seconds per CEGAR round on a 2023 laptop CPU; ≤ 100 rounds for 95th percentile benchmark.

#### Path 2: BDD Reachability Fixpoint

The reachability fixpoint `R_{i+1} = R_i ∪ Post(R_i)` converges in at most `|Q|²` iterations (bound by state count in the abstract product). Each `Post` operation is one BDD relational product, costing `O(|BDD|²)` in the worst case but `O(|BDD| log |BDD|)` with CUDD's dynamic variable reordering (Sifting algorithm).

**Variable ordering heuristic**: Group BDD variables by spatial cell proximity (cells sharing a boundary variable adjacent in the BDD ordering). This exploits locality in the transition relation and reduces BDD size by empirically 3–10× [Burch et al. 1992].

#### Path 3: GJK/EPA Oracle

Called once per abstract counterexample path per CEGAR iteration. Path length ≤ BMC bound `L ≤ 50`; guard atoms per step ≤ `k = 200`; GJK per call ≤ `O(d log(1/ε))` = O(30) iterations for ε = 10^{-6}:

```
50 (path length) × 200 (atoms) × 30 (GJK iters) × 1µs/iter ≈ 300ms per CEX check
```

Target: ≤ 500ms per counterexample check; parallelisable across CEGAR iterations.

### 6.4 Module Dependency DAG

```
[EC Parser]
    │
    ▼
[Thompson Compiler] ──────────────────────────┐
    │                                          │
    ▼                                          ▼
[Guard Compiler → R-tree]          [Geometric Pruning §2]
    │                                          │
    ▼                                          ▼
[On-the-Fly Product §5.4] ◄──── [Consistent Set BDD]
    │
    ▼
[SEA Network]
    │
    ├──────────────────────────┐
    ▼                          ▼
[Spatial Type Checker §3]    [Compositional Decomposer §4]
    │                          │
    ▼                          ▼
[LP/SOS Solver]            [Bag Verifier → SpatialCEGAR §1]
                                │
                                ▼
                            [BDD Reachability]
                                │
                                ▼
                            [GeometricOracle → GJK/EPA]
```

**Build order** (respecting dependencies):
1. Geometric Pruning (standalone, no dependencies)
2. Guard Compiler + R-tree (depends only on EC parser)
3. Thompson Compiler (depends on EC parser)
4. On-the-Fly Product (depends on 1, 2, 3)
5. Spatial Type Checker (depends on 1; LP solver)
6. BDD Reachability (depends on 1, 4)
7. Spatial CEGAR (depends on 6; GJK/EPA)
8. Compositional Verifier (depends on 7)

**Graceful degradation** (if a theorem fails to hold):
- If T1 (type checking) fails: the compiler emits a warning but proceeds; spatial type errors become runtime assertions; estimated 15% of bugs caught by T1 alone (rest are temporal).
- If T4 (compositional separability) fails: fall back to direct product verification (T3 only); tractable for N ≤ 6 components.
- If T3 (Spatial CEGAR) terminates with budget exceeded: report `unknown`, fall back to random simulation with 10^6 traces.
- T2 (geometric pruning) cannot fail in a soundness sense; worst case it prunes nothing (|C| = 2^k); system still correct, just slower.

### 6.5 Interface Contracts (Key Modules)

```rust
// Geometric Pruning
fn compute_consistent_set(
    predicates: &[SpatialPredicate],  // P
    scene: &SceneGraph,               // containment DAG, body radii
) -> ConsistentSet;                   // BDD-encoded C ⊆ 2^P
// Pre:  predicates derived from scene (no free bodies)
// Post: ∀v ∉ C. v is geometrically unrealizable (soundness)

// Spatial CEGAR
fn spatial_cegar(
    network: &SeaNetwork,             // A₁ ‖ … ‖ Aₙ
    property: &Property,             // φ (safety or reachability)
    budget: usize,                   // max refinement iterations
) -> CegarResult;                    // {Verified, Counterexample(Trace), Inconclusive}
// Pre:  network is well-formed; consistent_set pre-computed
// Post: Verified ⇒ network ⊨ property (under T3 correctness)
//       Counterexample(t) ⇒ t is a concrete violating trace

// Spatial Type Checker
fn spatial_subtype(tau1: &SpatialType, tau2: &SpatialType) -> bool;
// Pre:  types well-formed over shared entity vocabulary
// Post: true iff τ₁ ≤ τ₂ (sound; complete for convex polytope types)

// Compositional Verifier
fn compositional_verify(
    network: &SeaNetwork,
    property: &Property,
) -> VerificationResult;
// Pre:  treewidth of interference graph ≤ w (checked at entry)
// Post: complexity O(k · q^(w+1) · |C|)
```

---

## 7. Cross-Cutting Correctness Properties

### 7.1 Theorem Dependency Summary

| Theorem | Depends on | Provides to |
|---------|-----------|------------|
| T1 (Decidable type checking) | LP/SOS solver; coinductive subtyping | User-facing static error reporting |
| T2 (Geometric pruning) | Metric space axioms; scene graph | T3 (as preprocessing); T4 (as preprocessing) |
| T3 (Spatial CEGAR) | T2; GJK/EPA; BDD reachability | T4 (as bag verifier); top-level API |
| T4 (Compositional) | T3; tree decomposition; A-G framework | Top-level API for large networks |

T1 and T2 are **independent** of each other. T3 uses T2 but is still correct (just slower) without it. T4 uses T3 but can fall back to direct product.

### 7.2 Soundness Invariants Across the Pipeline

Throughout the compilation and verification pipeline, the following global invariants must hold:

**(S1) Abstraction Soundness**: At every point in the CEGAR loop, the abstract system over-approximates the concrete system. Maintained by: (a) starting with an over-approximate partition, (b) only refining (never coarsening).

**(S2) Consistency Soundness**: Every state `(q, v)` in the product automaton with `v ∉ C` is pruned. Maintained by: the BDD intersection with `bdd_C` at product construction time.

**(S3) Guard Monotonicity**: Compiled R-tree guards evaluate identically to the BDD-encoded guards for every concrete scene snapshot. Maintained by: R-tree AABB over-approximation is corrected by exact guard evaluation at candidates.

**(S4) Coinduction Soundness**: The assumption set `Γ` in Algorithm 3 never contains a false assumption. Maintained by: `Γ` is only extended with `(τ₁, τ₂)` when we are committed to proving it in the current proof attempt; if the attempt fails, `Γ` is unwound.

### 7.3 Known Limitations and Mitigations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Floating-point GJK | Separating plane may be slightly wrong | Use exact arithmetic (CGAL exact predicates) for certification runs; floating-point for search |
| Polynomial spatial predicates | LP infeasibility check is incomplete | Use SOS relaxation (Lasserre hierarchy level 2); sound over-approximation |
| Non-convex volumes | Type checking is NP-complete | Convex decomposition preprocessing (parry3d); warn user if decomposition introduces approximation |
| Unbounded MTL | MTL over unbounded time is undecidable | Restrict to bounded MTL with explicit horizon; noted in paper as assumption |
| Large treewidth | Compositional verification exponential in w | Fall back to CEGAR on full product for w > 5; document threshold |

---

## 8. Summary Table

| Algorithm | Inputs | Output | Time Complexity | Space | Theorem |
|-----------|--------|--------|----------------|-------|---------|
| SpatialCEGAR | SEA network, φ | verified/CEX | `O(|P|·2^d · q² · |P|·2^d)` worst | `O(|P|·2^d)` BDD nodes | T3 |
| ComputeConsistentSet | P, scene | C ⊆ 2^P | `O(k² · |P|)` Horn propagation | `O(|P|²)` BDD nodes | T2 |
| SpatialSubtype | τ₁, τ₂ | bool | `O(|τ|² · n^{1.5})` convex | `O(|τ|² · n)` LP constraints | T1 |
| CompositionalVerify | SEA network, φ | result | `O(k · q^(w+1) · |C|)` | `O(q^(w+1))` per bag | T4 |
| Thompson_EC | EC expr | NFA | `O(|ec|)` | `O(|ec|)` states | — |
| OnTheFlyProduct | NFA, BDD_C | DFA | `O(2^|Q| · |BDD_C|)` | `O(2^|Q| · |BDD_C|)` | — |
| CompileGuards | DFA | R-tree DFA | `O(|P| log |P|)` | `O(|P|)` R-tree nodes | — |

---

*End of Algorithm Design Proposal*
