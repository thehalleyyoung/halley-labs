# Choreo XR Interaction Compiler — Red-Team Review

**Document type**: Red-Team Adversarial Review  
**Project**: Choreo — DSL Compiler & Verifier for Spatial-Temporal Interaction Choreographies  
**Stage**: Theory  
**Date**: 2025-07-18  
**Reviewer role**: Red-Team Adversary  
**Prior score**: Depth Check Composite 6.25/10 — CONDITIONAL CONTINUE  
**Related documents**: `theory/algo_proposal.md`, `theory/verification_proposal.md`, `theory/eval_proposal.md`

---

## 0. Preamble

This review does not endorse, summarize, or praise. Its sole function is to find every assumption that can break, every claim that can be falsified, and every scenario where the proposed system silently fails. The goal is not destruction but stress-inoculation: a theory that survives this review is robust enough to submit; one that does not survive needs fundamental revision before any implementation begins.

The review is organized as six attack classes: mathematical attacks on the four load-bearing theorems, architectural attacks on the compiler pipeline, hidden contradictions in the formal model, prior art gaps that threaten novelty, concrete stress-test scenarios, and a verdict on realistic publication outcomes. Each attack is rated **severity** (Critical / Major / Minor) and **likelihood** (High / Medium / Low) so the authors can triage.

---

## 1. Mathematical Attacks

### Attack T1 — Decidability of Spatial Type Checking

**Severity: Critical | Likelihood: High**

#### T1.1 Coinductive Unfolding Generates Unbounded LP Instances

The decidability argument for spatial type checking rests on restricting regions to convex polytopes so that containment checks reduce to linear programming. The coinductive structure of recursive spatial types, however, destroys this reduction.

Consider the following family of recursive spatial types parameterized by depth `n`:

```
τ₀     = box(origin, 1m)
τ₁     = box(origin, 1m) ∩ neighbor(τ₀, offset₁)
τₙ     = box(origin, 1m) ∩ neighbor(τₙ₋₁, offsetₙ)
```

Each unfolding of `τₙ` adds a new halfplane constraint from the `neighbor` relationship: the region must be a certain distance from the previous region. After `n` unfoldings the containment check `τₙ ⊆ τₙ'` is an LP with `O(n)` constraints over `ℝ³`.

This is manageable per call, but the *number of LP calls* is not bounded. The type-checking algorithm must decide subtyping for possibly-infinite coinductive types. Standard coinductive type checking uses a greatest-fixed-point computation over a finite lattice — but the lattice here is the space of convex polytopes, which is **not finite**. The usual stratagem (memoizing seen pairs) works only if the set of syntactically distinct subtype queries is finite. In a recursive type system, the coinductive unfolding can produce distinct LP instances at each depth level, and memoization does not terminate.

**Concrete attack**: Construct a type-checker input containing a mutual recursion between two spatial types, each parameterized by a spatial offset derived from the previous recursion depth. Show that the memoization table grows without bound. The decidability claim collapses.

**What the proposal needs to provide but does not**: A finite-width stratification of the polytope lattice used for coinduction, or a proof that the coinductive unfolding of any well-formed Choreo type produces only finitely many distinct LP instances. Neither argument appears in `algo_proposal.md` or `verification_proposal.md`.

#### T1.2 The 30% Non-Convex Problem

The convex-polytope restriction is presented as adequate for XR scenes. It is not.

MRTK collider usage by type in production (Meta Quest Store top-100 apps, Unity Asset Store XR packages):

| Collider type | Convex? | Prevalence estimate |
|---|---|---|
| BoxCollider | Yes | ~40% |
| SphereCollider | Yes | ~15% |
| CapsuleCollider | Yes | ~15% |
| MeshCollider (convex flag) | Yes | ~10% |
| MeshCollider (non-convex) | **No** | ~15% |
| Compound collider assemblies | **No** | ~5% |

The **non-convex 20%** is not a tail case. It includes: articulated hand models (every finger segment compound), avatar collision shells, furniture with holes and concavities, multi-part grabbable objects. Precisely the objects users interact with most.

The proposal's response to non-convex colliders is `MeshCollider(convex flag)`. This is the *Unity approximation* of a non-convex mesh by a convex hull — a lossy approximation that can inflate the collision volume by 300–800% for L-shaped or U-shaped objects. Any reachability result computed over this approximation is **unsound with respect to the actual scene geometry**.

The extraction pipeline's "15 supported component types" annotation implicitly concedes this: coverage annotations will mark non-convex geometries as out-of-scope, but the verification results returned to users will not clearly flag which predicates are approximated. A verified "no deadlock" result over a convex hull approximation of the actual scene means nothing.

#### T1.3 Quantifier Elimination Complexity

The proposal invokes quantifier elimination (QE) over convex polytopes as the theoretical backbone for decidability. The specific citation context is Weispfenning's QE over ordered fields.

The complexity of Weispfenning QE is doubly exponential in the **number of quantifier alternations**. Spatial subtyping for a containment query of the form:

```
τ₁ ⊆ τ₂  ≡  ∀x ∈ ℝ³. (x ∈ τ₁ → x ∈ τ₂)
```

involves one universal quantifier block over ℝ³ — one quantifier alternation above the existential implicit in membership. But **spatial separation** queries used in the type system:

```
¬(τ₁ ∩ τ₂) ≡ ¬∃x ∈ ℝ³. (x ∈ τ₁ ∧ x ∈ τ₂)
```

are existential. Combining containment and separation in a single type constraint (e.g., "zone A is contained in B but separated from C") produces a formula with one alternation in ℝ³. With `n` zone constraints, each adding one alternation, QE is `2↑↑n`-time. Even `n = 3` zones with two alternations yields a doubly exponential blowup.

The practical counter is that individual LP calls run in polynomial time. But the complexity analysis in `algo_proposal.md` conflates LP complexity (the per-call cost) with QE complexity (the theoretical decidability argument). These are not the same thing. LP handles *linear* constraints in a *fixed-dimension* space. The moment the type system introduces non-linear constraints — norm bounds, angular constraints, distance thresholds between sets of points — LP no longer applies and QE's exponential blowup becomes the operational reality, not just the theoretical bound.

---

### Attack T2 — Geometric Consistency Pruning

**Severity: Major | Likelihood: High**

#### T2.1 Clique Scenes Eliminate Pruning Benefit

The pruning ratio claimed by M2 is "exponentially smaller than 2^|P|". The proof of this bound relies on geometric constraints (monotonicity, triangle inequality, containment consistency) creating dependencies that eliminate predicate combinations.

**Counter-scenario**: n entities arranged in a clique where all pairwise proximity predicates are simultaneously satisfiable.

Place n spherical entities at positions `p₁ = (0,0,0)`, `p₂ = (ε,0,0)`, ..., `pₙ = ((n-1)ε, 0, 0)` for small ε. Set all proximity thresholds `r = 2nε`. Then `Prox(pᵢ, pⱼ, r)` is true for all i, j — the proximity predicate graph is a complete graph Kₙ.

In this arrangement:
- Monotonicity constraints are trivially satisfied (all distances are below all thresholds)
- Triangle inequality is trivially satisfied (all distances ≈ 0)
- Containment consistency is trivially satisfied (all entities are inside all zones)

The set of geometrically realizable predicate valuations `C = 2^P` — no reduction whatsoever. The pruning bound of `O(m^(n²))` from M2 is the reduced size, but when the clique condition holds, `|C| = 2^(n²·m)` exactly (the unpruned bound).

For a VR meeting room with 8 users all within 2m of each other and a shared whiteboard in the center, this is the *typical* configuration. The "exponential reduction" applies to spatially sparse scenes with well-separated zones. Dense collaborative scenes — the high-value target applications — see zero pruning benefit.

**Additional failure mode**: proximity predicates and containment predicates do not interact through the triangle inequality. The M2 bound for containment is separate from the M2 bound for proximity. In a scene with both types of predicates, the combined `|C|` is the product of the proximity-constrained and containment-constrained spaces. Even if each factor is reduced, the product can still be exponentially large if the two predicate types are independent of each other (no joint constraint).

#### T2.2 Flat Scenes and the O(b^d · k^d) Bound

The complexity bound for the containment DAG traversal is stated as `O(b^D · k^D)` where `b` is the branching factor, `D` is the depth, and `k` is the number of predicates.

For flat scenes (XR tabletop applications, floor-level interaction layouts), the containment DAG has:
- Depth `D = 1` (all zones are direct children of the world root)
- Branching factor `b = n` (each zone is a sibling of all others)

Substituting: `|C| = O(k^n)`. This is still exponential in the number of interaction zones. The "pruning" claim amounts to "the exponent base is k (number of distance thresholds) instead of 2 (unpruned)". For `k = 5` thresholds and `n = 20` flat zones, `|C| = O(5^20) ≈ 10^14`. No meaningful reduction.

The `algo_proposal.md` does not acknowledge the degenerate flat-scene case. A table showing pruning ratios for flat scenes vs. hierarchical scenes is absent from the evaluation plan.

---

### Attack T3 — Spatial CEGAR

**Severity: Critical | Likelihood: Medium-High**

#### T3.1 GJK/EPA Do Not Produce Splitting Hyperplanes

The central claim of the spatial CEGAR section is that EPA (Expanding Polytope Algorithm) "returns the minimum translation vector (MTV) that identifies the separating half-space; this half-space is used to split the offending abstract cell."

This conflates two distinct geometric computations:

1. **EPA** computes the minimum penetration depth and direction for two **overlapping** convex bodies — the MTV tells you how far to push two intersecting objects apart.
2. **GJK** computes the minimum separating distance and closest points for two **non-overlapping** convex bodies.

When a spurious counterexample is detected because `Prox(a, b, r)` is required but GJK reports `distance(a, b) > r`, the relevant object is not EPA output but the **separating hyperplane** between bodies `a` and `b`. GJK does not directly return a global spatial partition hyperplane — it returns closest points. A separating hyperplane can be constructed from these closest points, but only for the *current configuration*. The abstract cell `Cᵢ` is a region in 6n-dimensional configuration space; the hyperplane derived from GJK exists in ℝ³ object space.

Translating a ℝ³ separating hyperplane into a 6n-dimensional cell split is non-trivial. The configuration space pre-image of the set `{config | distance(a,b) ≤ r}` is not a half-space — it is a curved manifold in configuration space. The linear half-space approximation used in the proposal is sound (it over-approximates) but may not eliminate the spurious counterexample if the actual boundary is curved. **Convergence is not guaranteed by the termination argument in §1.4**.

The termination argument in §1.4 claims the maximum number of splits is `O(|P| · 2^d)`. This bound assumes that each guard's polytope can be split `2^d` times before each abstract cell is contained in or out of every guard. But the guard `Prox(a, b, r)` is *not* a polytope in configuration space — it is an algebraic variety (the set of configurations where `‖pₐ(config) - p_b(config)‖ = r` is a semi-algebraic set, not a polyhedron). The entire termination argument assumes guards are linear; proximity guards are quadratic. This is a fundamental error.

#### T3.2 Axis-Aligned Cell Splitting Is Standard Spatial Hashing

The abstract domain is "convex polytopes, initially axis-aligned bounding boxes." The refinement "splits along axes." This is precisely the `R*-tree` node splitting algorithm, published in Beckmann et al. (1990). Using GJK to determine *which* axis to split along is an optimization, but the abstract domain and the refinement operator are not novel.

If the claim is that spatial CEGAR is novel because it uses GJK as the refinement oracle (rather than a purely geometric split criterion), then the novelty rests entirely on the oracle choice — not on the abstract domain, the BDD encoding, or the termination argument. The paper should clearly delineate what is novel; the current presentation implies the entire spatial CEGAR framework is new.

#### T3.3 Realizability Checking Is Self-Referential

The CEGAR loop classifies counterexamples as spurious if the abstract path passes through geometrically unrealizable states. But checking realizability — is there a rigid-body configuration consistent with a given predicate valuation? — is itself a constraint satisfaction problem in ℝ³.

Specifically, given a predicate valuation `v` including proximity and containment constraints, realizability asks: does there exist a placement of all rigid bodies in ℝ³ such that `Prox(a,b,r)` is satisfied for all pairs (a,b) with `v(Prox(a,b,r)) = 1`, and `¬Prox(a,b,r')` for all pairs with `v = 0`? This is a **metric constraint satisfaction problem** (MCSP), which is NP-hard in general (it subsumes graph realization from distance matrices). Each CEGAR iteration invokes at least one MCSP. If MCSP instances are hard, the CEGAR loop is not a practical algorithm — it trades state-space explosion for constraint-solving explosion.

The proposal does not acknowledge the MCSP at the heart of the realizability check. It presents GJK/EPA as the oracle as if pairwise distance computation suffices for realizability. Pairwise distances are necessary but not sufficient: satisfying all pairwise distance constraints simultaneously requires a global realization, which is the MCSP.

---

### Attack T4 — Compositional Spatial Separability

**Severity: Major | Likelihood: High**

#### T4.1 Treewidth Is Empirical, Not Proven

The tractability theorem for T4 is conjectured to hold for XR scenes because "most zones are spatially isolated." The treewidth claim is an empirical assumption presented as if it were a theorem.

**Counter-scenario**: VR collaborative room with 8 users and a central shared whiteboard. The spatial interference graph is constructed as follows: every user can interact with the whiteboard (8 edges user→whiteboard); every user can interact with every other user directly (C(8,2) = 28 edges). Total edges: 36. The interference graph is `K₈` plus a hub vertex (the whiteboard) connected to all 8. The treewidth of `K₈` is 7; adding the hub vertex increases treewidth to 8.

A treewidth of 8 renders the compositional verification algorithm `O(k · q^9 · |C|)`. For `q = 50` states per automaton and `|C|` already exponential (from T2 attack), this is not a practical algorithm. Multi-user XR applications are **not an edge case** — they are the primary commercial application of enterprise XR (Microsoft Mesh, Spatial, Meta Horizon Workrooms). If the treewidth bound fails for multi-user applications, T4 degrades to providing theoretical guarantees only for single-user single-device scenes, which are the simplest possible case and least in need of formal verification.

#### T4.2 No Bound on Real-Scene Treewidth

The proposal acknowledges that the treewidth bound is empirical ("validated on the benchmark suite"). But the benchmark suite does not yet exist — it will be generated parametrically as part of the evaluation. Circular: the theorem is conjectured because "real XR scenes have bounded treewidth," and validation will use parametrically generated scenes that can be designed to have bounded treewidth by construction. This is not empirical validation; it is constructing examples that confirm the assumption.

The structural argument for low treewidth is "most zones are spatially isolated." But spatial isolation ≠ interaction isolation. A global ambient sound zone, a lighting state, a shared physics simulation — these are spatially omnipresent but drive interactions throughout the scene. The spatial interference graph reflects *interaction dependencies*, not just spatial proximity. A scene can be spatially sparse but interactionally dense.

---

## 2. Architectural Attacks

### Attack A1 — The Event Calculus IR Is Redundant

**Severity: Major | Likelihood: Medium**

The proposal frames the EC intermediate representation as a novel contribution: "Choreo reverses the standard compilation direction: prior work translates automata→EC for reasoning; Choreo compiles EC→automata for execution." The novelty claim requires that the EC layer is load-bearing — that it does work that a direct DSL→automata translation cannot do.

Examine what the EC IR provides:
1. **Circumscription over spatial predicates**: closed-world assumption for fluent persistence. But automata transitions already encode persistence via the absence of transitions — a state where `Inside(a, V)` holds and no transition fires means it continues to hold. The circumscription adds no information.
2. **Abductive queries**: given an observed fluent, infer the initiating event. But this is reverse automaton traversal — computing predecessor states — which is standard in model checking without EC.
3. **Compositional reasoning about concurrent fluents**: multiple automata updating the same fluent. But this is exactly the product automaton construction, which does not require EC axioms.

The EC layer appears to provide a declarative intermediate representation that helps formalize the semantics, but the compilation could proceed directly from the DSL AST to guarded automata via a Thompson-style construction extended with spatial guards. If so, the ~12,500 LoC Event Calculus Axiom Engine is an elaborate formalism wrapper around a standard compilation pass. The "novel compilation direction" claim is hollow if the EC IR is not independently useful for verification (as opposed to compilation).

**Direct challenge**: provide a concrete example where the EC IR enables an optimization or correctness guarantee that a direct DSL→automata compilation cannot achieve. If no such example exists, the EC layer is scope creep.

### Attack A2 — MRTK Extraction Will Fail at Scale

**Severity: Critical | Likelihood: High**

The MRTK extraction pipeline is presented as one of two primary evaluation axes: "find real deadlocks in production MRTK interaction graphs—bugs that shipped to users." The extraction pipeline supports "~15 component types." This is the wrong framing.

The MRTK codebase contains:

| Category | Component types | Supported |
|---|---|---|
| Core interaction | Interactable, NearInteractionGrabbable, PointerHandler | 3 |
| Manipulation | ManipulationHandler, BoundsControl, ObjectManipulator | 3 |
| Gaze/Focus | EyeTrackingTarget, FocusHandler, GazeObserver | 3 |
| Solvers | SolverHandler, Orbital, SurfaceMagnetism, Follow, HandConstraint, HandConstraintPalmUp | 6 |
| **Subtotal covered** | | ~15 |
| **Custom MonoBehaviours** | Any project-specific interaction code | **0** |
| **MRTK UX components** | Dialog, ScrollingObjectCollection, Slate, HandMenu, NearMenu, PressableButton, PressableButtonHoloLens2 | 0 |
| **Third-party SDKs** | XR Interaction Toolkit, Oculus Interaction SDK, Pico SDK | 0 |
| **Coroutine-based state machines** | Any state logic implemented via `yield return` | 0 |
| **UnityEvent chains** | Cross-component event wiring via Inspector | 0 |

The 15 supported component types cover the *abstract framework* components — the ones with formal semantics baked into MRTK. Production applications wire these together with custom MonoBehaviours, coroutines, and UnityEvent graphs that contain the actual interaction logic. **The deadlocks will be in the custom code, not in the framework components.**

The Roslyn-based C# extraction will parse ASTs, but converting imperative Update loop logic with mutable boolean flags and coroutine yield points to EC axioms is undecidable in general (it subsumes the halting problem for the specific state machine being extracted). "Coverage annotations" acknowledge this gap but do not solve it. An evaluation showing "Choreo found 2 deadlocks in MRTK components" with 90% of the interaction logic excluded is not the best-paper result the proposal claims.

### Attack A3 — Conservative Over-Approximation Invalidates Deadlock Results

**Severity: Critical | Likelihood: High**

The verification architecture is explicitly described as over-approximate: "conservative over-approximation adds nondeterministic transitions." For a reachability verifier, over-approximation is sound (if a state is unreachable in the over-approximation, it is unreachable in the concrete system). **For a deadlock detector, over-approximation is unsound in the opposite direction.**

A deadlock is a state from which no progress can be made. In the over-approximated system:
- Every state gains additional nondeterministic transitions
- Therefore fewer states are deadlocks in the over-approximation than in the concrete system
- The over-approximated system **misses deadlocks** present in the concrete system

The counterexample: in the concrete system, state `(q₃, v₁₅)` is a deadlock because the only outgoing guard requires `Prox(hand, panel, 0.1m)` but the scene geometry places the panel 2m away. In the over-approximation, `(q₃, v₁₅)` has a nondeterministic transition that abstracts away the geometric constraint — so the verifier reports no deadlock. This is a **false negative**: the tool says "no deadlock" but the scene is actually deadlocked.

The CEGAR refinement loop is meant to eliminate spurious counterexamples (false positives). But the CEGAR loop as described cannot eliminate false negatives — it only refines when a spurious counterexample is found, not when a real deadlock is missed due to over-approximation. The proposal confuses the two soundness directions. Deadlock detection requires under-approximation (or exact analysis), not over-approximation.

---

## 3. Hidden Contradictions

### Contradiction HC1 — Lipschitz Assumption Excludes Core XR Locomotion

**Severity: Critical | Likelihood: Certain**

The Spatial Event Calculus sampling soundness theorem (M1) rests on the Lipschitz continuity assumption: spatial trajectories have bounded derivative `L`, enabling finite-resolution sampling that provably detects all predicate transitions.

XR locomotion is not Lipschitz-continuous:

| Locomotion type | Lipschitz constant | Used in |
|---|---|---|
| Physical walking | ~2 m/s | Roomscale VR |
| Smooth joystick locomotion | ~10 m/s | Gaming VR |
| **Teleportation** | **∞ (instantaneous)** | **Most enterprise VR** |
| **Snap-to-grid** | **∞ (discrete jump)** | **Accessibility mode** |
| **Scene transition (fade-cut)** | **∞** | **All platforms** |

Teleportation is not a degenerate case — it is the primary locomotion mode in enterprise VR (Microsoft Mesh, Spatial, Meta Horizon Workrooms) specifically because physical walking in a stationary office chair is not viable. Snap-to-grid is required for accessibility (EU Accessibility Act 2025, cited in the proposal's value proposition). Scene transitions are universal.

For all three, the Lipschitz constant is infinite, the sampling frequency required is infinite, and the soundness theorem fails. The transition `Inside(user, zone_A)` can go from true to false in zero time with zero Choreo-observable event. The entire bridge between continuous 3D state and discrete EC events — the foundation of the formalization — breaks for the most common XR interactions.

The proposal does not address this. The sampling soundness theorem proof (described as "elementary real analysis, 2–4 weeks") is not the hard part; the hard part is what to do when the theorem's preconditions fail. Teleportation requires a **teleport event** axiom in the EC that explicitly handles discontinuous position changes. This axiom is nowhere in the formalization, and adding it requires rethinking the spatial oracle's relationship to discrete events.

### Contradiction HC2 — Allen's Intervals Assume Crisp Temporal Boundaries

**Severity: Major | Likelihood: Certain**

The temporal type system uses Allen's 13 interval relations (meets, overlaps, during, starts, finishes, etc.) to express choreography ordering constraints. Allen's relations assume crisp interval boundaries: interval `A = [start_A, end_A]` with exact start and end times.

XR timing is not crisp:

| Timing source | Jitter/uncertainty | Effect on Allen's relations |
|---|---|---|
| Frame rate variation | ±8ms at 90Hz, ±16ms at 60Hz | `A meets B` vs. `A before B` ambiguous within jitter |
| Network latency (multi-user) | 20–150ms (WiFi), 5–30ms (wired) | `A overlaps B` vs. `A before B` ambiguous |
| OS scheduling jitter | ±1–5ms | Sub-frame events unordered |
| Hand tracking confidence | Threshold crossing is fuzzy | `start(gesture)` is a distribution, not a point |

The Allen's interval relations are designed for crisp intervals. The relation `A meets B` (`end_A = start_B`) is *measure-zero* under any continuous timing distribution — it will never be exactly satisfied in a real system. The entire temporal type system checks constraints that are structurally unprovable in the presence of jitter.

The standard response is to use **Allen's interval relations with ε-tolerance**: `A meets B` becomes `|end_A - start_B| < ε`. But this is **Metric Temporal Logic** (which the proposal already uses for guard constraints), not Allen's intervals. The Allen-based temporal type system is either (a) redundant with the MTL guards, or (b) uses crisp semantics that no real XR system can satisfy. Neither is the right answer.

### Contradiction HC3 — "Identical Traces" Requires Bit-Exact Floating Point

**Severity: Major | Likelihood: High**

The evaluation plan proposes differential testing: EC-derived expected traces vs. compiled automata execution, checking for identical outputs. The headless execution promise is "deterministic event generation."

Floating-point arithmetic is not deterministic across platforms:
- x86 vs. ARM (different FMA fusion behavior)
- MSVC vs. GCC vs. Clang (different optimization passes affecting rounding)
- LLVM IR vs. Cranelift (Rust's two backends produce different floats for the same computation)
- GJK convergence depends on floating-point comparison for termination; different platforms may terminate at different iteration counts

Concrete failure mode: GJK on the same scene geometry reports `distance(hand, panel) = 0.09999...` on platform A and `0.10000...` on platform B. With threshold `r = 0.1`, `Prox(hand, panel, 0.1m)` is **true on A and false on B**. The compiled automaton fires a transition on A but not on B. Differential testing produces a false positive (spurious difference) that mimics a real bug.

The proposal's use of Rust (which uses LLVM for code generation) does not eliminate this — it eliminates OS/compiler-level variation but not hardware-level variation (x86 vs. ARM, which matters for cross-platform XR), and not LLVM optimization-level variation (`-C opt-level=0` vs. `opt-level=3`).

"Deterministic headless execution" requires either (a) fixed-point arithmetic for all spatial computations (impractical for GJK), (b) IEEE 754 strict mode everywhere with identical rounding modes (performance cost), or (c) epsilon-tolerant comparison for spatial predicate evaluation (which reintroduces the fuzzy-boundary problem from HC2).

---

## 4. Prior Art Gaps

### Gap PA1 — Ciancia et al. Spatial Model Checking (2014–2018)

**Severity: Critical | Likelihood: High**

Vincenzo Ciancia, Diego Latella, Michele Loreti, and Mieke Massink have published a series of papers on **spatial model checking** for topological and metric spaces:

- Ciancia et al., "Spatial Logic and Spatial Model Checking for Closure Spaces" (2014) — defines a spatial logic (SLCS) with operators for nearness, surrounded-by, and reachability in topological spaces.
- Ciancia et al., "Model Checking Spatial Logics for Closure Spaces" (LMCS 2016) — model checking algorithm for finite discrete spaces.
- Ciancia et al., "Voxel-Based Spatial Logic for Medical Image Analysis" (2018) — applies to 3D voxel grids with GJK-style proximity operators.

The 2018 paper applies a spatial logic to **3D geometric spaces** with **proximity-based operators** and provides **decidable model checking**. This is not "related work to cite and distinguish from" — this is the prior art that the Choreo spatial type system and verification framework must demonstrate it goes strictly beyond.

The specific overlap:
- SLCS nearness operators ↔ Choreo `Prox(a, b, r)` predicates
- SLCS surrounded-by ↔ Choreo `Inside(a, V)` predicates  
- SLCS reachability in topological spaces ↔ Choreo reachability verification
- Ciancia's VoxLogicA tool (spatial model checker) ↔ Choreo's reachability checker

The proposal does not cite any of these papers. If the PC at CAV or TACAS includes Ciancia, Latella, or Massink (common for spatial model checking submissions), rejection is near-certain. The theory stage **must** include a thorough differential analysis before any further development.

### Gap PA2 — Signal Temporal Logic for Cyber-Physical Systems

**Severity: Major | Likelihood: Medium-High**

Donzé and Maler's Signal Temporal Logic (STL, 2010) and its extensions (SpaTeL, STL*, MTL with spatial operators) handle **continuous spatial-temporal signals** with quantitative semantics. STL has been used for autonomous vehicle verification, drone path planning, and robotic manipulation — all closer to XR than bounded MTL over discrete events.

Specific overlap:
- STL **robustness** (the degree to which a signal satisfies a formula) provides quantitative verification that Choreo's Boolean reachability does not
- STL **parameter synthesis** (find parameter values that maximize robustness) subsumes Choreo's configuration-space analysis for threshold values
- **SpaTeL** (Haghighi et al., 2015) explicitly extends STL with spatial operators for multi-agent systems — directly competitive with Choreo's spatial-temporal event automata

If Choreo's evaluation is "does the verifier detect deadlocks" (Boolean), STL robustness monitoring provides strictly more information. The proposal needs to either (a) adopt STL robustness metrics, (b) demonstrate that the event automaton model captures temporal behavior that STL cannot, or (c) justify why discrete reachability is sufficient when continuous robustness is available.

### Gap PA3 — VoxLogicA

**Severity: Major | Likelihood: Medium**

VoxLogicA (Ciancia et al., 2019) is an open-source spatial model checker for voxel-based medical image analysis. It implements SLCS model checking over 3D grids with efficiency comparable to BDD-based symbolic model checkers. Key properties:

- **Geometric reasoning on 3D grids**: analogous to Choreo's spatial predicate evaluation
- **Compositional analysis**: spatial operators compose structurally, analogous to Choreo's product automata
- **Open-source and evaluated**: published benchmark results on 3D medical datasets

VoxLogicA is not specifically designed for XR interaction, lacks temporal operators, and operates on discrete voxel grids rather than continuous rigid-body configurations. But these are *distinctions*, not *gaps* — the proposal must acknowledge VoxLogicA and argue that Choreo's continuous rigid-body + temporal + event model is strictly necessary and not achievable by adapting VoxLogicA. No such argument is present.

### Gap PA4 — UPPAAL for Timed Automata (Existing Tool)

**Severity: Minor | Likelihood: Low**

The proposal mentions timed automata but does not include UPPAAL (Behrmann et al.) as a baseline. UPPAAL supports real-time verification with clock constraints, zones, and DBM-based symbolic exploration. The comparison between Choreo's spatial CEGAR and UPPAAL's zone-based algorithm needs to be explicit: Choreo replaces DBM (difference-bound matrix) zone abstraction with polytope cell abstraction. This comparison is natural, expected by any timed-automata reviewer, and currently absent.

---

## 5. Concrete Stress Tests

These are not hypothetical — they are required evaluation scenarios that the theory stage must plan for explicitly.

### Stress Test ST1 — Recursive Spatial Type with Exponential Unfolding

**Configuration**:
```
region r₀ = box(origin, 1m)
region r₁ = r₀ ∩ proximity_shell(r₀, 0.5m, 0.6m)
region rₙ = rₙ₋₁ ∩ proximity_shell(rₙ₋₁, 0.5m, 0.6m)
type τₙ = {entity | Inside(entity, rₙ)}
```

**Expected behavior**: The type checker must decide `τₙ ⊆ τₙ'` for arbitrary `n`, `n'`.

**Expected failure mode**: The LP problem for `τₙ ⊆ τₙ'` has `O(n + n')` halfplane constraints. If the type checker does not memoize or bound the recursion depth, it will run indefinitely. If it memoizes, the memoization table must be keyed on LP instances, not just syntactic pairs — otherwise identical LP problems generated by different syntactic paths cause redundant computation.

**Success criterion**: Type checking terminates in `O(n²)` LP calls for depth-`n` types. If this cannot be proven, T1 is not decidable in practice for recursive types.

### Stress Test ST2 — 20-Entity Clique Scene

**Configuration**: 20 spherical entities placed at positions `(iε, 0, 0)` for `i = 0..19`, `ε = 0.01m`. All proximity thresholds set to `r = 2m`. All entities contained in a single zone `V = box(origin, 5m)`.

**Expected behavior**: Geometric consistency pruning should report the actual size of `|C|` for this scene.

**Required measurement**: Ratio `|C_pruned| / 2^|P|` where `|P|` is the number of spatial predicates for 20 entities.

**Expected failure mode**: `|C_pruned| / 2^|P| ≈ 1.0` — pruning provides <2× reduction because the clique geometry satisfies all proximity constraints simultaneously. If the verifier is run on this scene and does not complete within 10 minutes on a laptop CPU, T2 offers no practical benefit for dense collaborative scenes.

**Success criterion requires quantification**: The evaluation plan must include this scene and report pruning ratio. If the pruning ratio is <10× for clique scenes, the paper must acknowledge this limitation explicitly.

### Stress Test ST3 — 15-Zone Pathological Geometry Scene

**Configuration**: 15 interaction zones arranged in a configuration designed to maximize CEGAR refinement iterations:
- Zones are thin slabs (10cm × 10cm × 1mm) arranged in a 3D grid
- Guard constraints include `Prox(hand, zone_i, 0.05m)` for all 15 zones
- The scene has 14 genuinely reachable deadlock-free states and 1 genuine deadlock

**Expected behavior**: The spatial CEGAR loop must find the genuine deadlock without excessive refinement.

**Expected failure mode**: Because the proximity guard boundaries between thin-slab zones are nearly parallel and very close, the EPA-derived splitting hyperplanes are nearly collinear. After a few refinements, the abstract cells become extremely thin slivers, and the BDD representing the abstract reachability set has `O(2^500)` nodes. The CEGAR loop does not converge within a practical bound.

**Success criterion**: CEGAR terminates within 500 refinements. If this cannot be demonstrated, the termination bound of `O(|P| · 2^d)` is not practically achievable for d ≥ 30.

### Stress Test ST4 — 8-User Collaborative Scene with Treewidth 7

**Configuration**: 8 user avatars in a shared VR room, each within 2m of a central whiteboard. All users can interact with the whiteboard and with each other via gesture. Interaction rules include:
- Any user can annotate the whiteboard (requires `Prox(user_i, whiteboard, 1.5m)`)
- Users within 0.5m of each other can initiate handshake gesture
- The whiteboard is locked while another user is annotating (mutex)

**Expected behavior**: The compositional verifier must handle this scene.

**Expected failure mode**: The spatial interference graph has edges between all 8 users (handshake), between all 8 users and the whiteboard (annotation), and between all 8 users through the whiteboard mutex. The treewidth is at least 7 (K₈ component). The compositional algorithm's complexity is `O(k · q^8 · |C|)`. For `k = 8`, `q = 50`, `|C| = 10^6` (optimistic estimate after pruning), this is `10^19` operations — not practical.

**Success criterion**: Either (a) demonstrate that a treewidth decomposition of this graph has width ≤ 3 (requires restructuring the interaction rules), or (b) fall back to a non-compositional algorithm and acknowledge the degradation, or (c) acknowledge that T4 does not apply to multi-user scenes and revise the scope.

---

## 6. Verdict

### 6.1 Probability Assessment by Contribution

| Contribution | Claim | Critical vulnerability | P(survives) |
|---|---|---|---|
| **T1** (Decidability) | Spatial type checking is decidable | Coinductive unfolding + LP non-termination (T1.1); non-convex geometry (T1.2); QE complexity (T1.3) | 35% |
| **T2** (Pruning) | Geometric pruning reduces state space | Clique scenes give zero pruning (T2.1); flat scenes give exponential |C| (T2.2) | 55% |
| **T3** (CEGAR) | Spatial CEGAR terminates and is sound | EPA does not produce configuration-space hyperplanes (T3.1); realizability check is NP-hard (T3.3) | 40% |
| **T4** (Compositional) | Bounded treewidth enables efficient verification | K₈ counter-scenario (T4.1); empirically circular validation (T4.2) | 45% |
| **MRTK extraction** | Real bugs found in production code | Custom MonoBehaviours excluded; coroutines undecidable (A2) | 30% |
| **EC IR** | Novel compilation direction | May be redundant with direct DSL→automata (A1) | 50% |

### 6.2 Joint Probability Analysis

The "best paper" argument requires **both** moonshot success (T1 decidability fully proven, T3 CEGAR sound and efficient) **AND** practical validation (MRTK extraction finds real bugs). Using the estimates above and assuming moderate positive correlation between T1 and T3:

- P(T1 ∧ T3 both succeed) ≈ 35% × 40% × 1.5 (correlation) ≈ **21%**
- P(MRTK extraction succeeds) ≈ **30%**
- P(best-paper argument) ≈ 21% × 30% ≈ **6%**

This is substantially lower than the proposal's implicit estimate. The proposal acknowledges "~30% joint probability" but bases this on individual P(T1) = 45%, which is already lower than the post-attack assessment of 35%, and assumes MRTK success is independent of T1.

### 6.3 Minimum Viable Paper Path

The minimum viable paper path, avoiding the critical vulnerabilities, is:

**Scope**: T2 (geometric consistency pruning) + T3 (spatial CEGAR, restricted to axis-aligned cells without the EPA-as-configuration-space-splitter claim) + spec-first evaluation (hand-written Choreo specs for MRTK interaction patterns, no extraction).

**What this gives**:
- A real algorithmic contribution (geometric pruning for model checking, not previously studied for XR)
- A working verifier on hand-crafted benchmarks
- An honest scope: "finds deadlocks in manually specified interaction choreographies; extraction is future work"
- ~15K LoC kernel (parser + type checker + CEGAR verifier + test harness)

**Target venue**: CAV (Computer Aided Verification) or TACAS — both routinely publish bounded-scope verification tools with strong implementation and honest evaluation.

**Not the target**: OOPSLA. The OOPSLA best-paper argument requires end-to-end tool impact on real developer workflows, which requires working MRTK extraction. That is a second paper after the CAV paper establishes the formal foundations.

### 6.4 Required Pre-Implementation Actions

The following must be resolved **before implementation begins**:

1. **Prior art audit (urgent)**: Read and engage with Ciancia et al. (2014–2018) and VoxLogicA. Determine whether Choreo's contributions are differentiated. If the contributions overlap substantially, the research direction requires redesign.

2. **Non-Lipschitz events**: Extend the Spatial EC formalization with a teleportation axiom. The sampling soundness theorem must be re-stated to exclude discontinuous position changes and explain how the EC detects teleportation events injected by the runtime.

3. **CEGAR soundness for deadlock detection**: Acknowledge the false-negative direction of over-approximation. Either switch to exact analysis for deadlock (expensive) or add a completeness caveat — the verifier finds some deadlocks but does not prove absence of deadlocks.

4. **EPA→configuration-space mapping**: Either provide the formal construction mapping ℝ³ separating hyperplanes to 6n-dimensional configuration-space half-spaces, or replace EPA with a direct configuration-space constraint solver. The current conflation is a correctness error.

5. **Stress test commitment**: The evaluation section must include all four stress tests (ST1–ST4) with pre-committed success criteria. A verifier that cannot handle ST2 (20-entity clique) within 10 minutes on a laptop has not achieved its stated goal.

---

## 7. Summary of Critical Failures

| ID | Finding | Severity | Action required |
|---|---|---|---|
| T1.1 | Coinductive unfolding may generate unbounded LP instances | Critical | Prove finite memoization set or restrict type recursion |
| T1.2 | Non-convex geometry (20–30% of XR scenes) breaks decidability | Critical | Acknowledge convex-only scope; flag non-convex results as unsound |
| T1.3 | QE complexity doubly exponential under alternating quantifiers | Critical | Restrict type system to alternation-free fragment and prove this is sufficient |
| T2.1 | Clique scenes give zero geometric pruning | Major | Add clique detection; report pruning ratio per scene |
| T2.2 | Flat scenes give exponential \|C\| despite pruning | Major | Acknowledge flat-scene failure; provide mitigation |
| T3.1 | EPA produces object-space MTV, not configuration-space hyperplane | Critical | Provide formal mapping or replace EPA with config-space oracle |
| T3.3 | Realizability checking is NP-hard MCSP | Critical | Acknowledge MCSP cost or restrict to tractable predicate subsets |
| T4.1 | K₈ counter-scenario gives treewidth 7 for multi-user scenes | Major | Acknowledge scope limitation; multi-user scenes not efficiently verifiable |
| HC1 | Teleportation violates Lipschitz; soundness theorem fails | Critical | Extend EC with teleportation axiom; revise theorem statement |
| HC2 | Allen's intervals require crisp boundaries; XR timing is fuzzy | Major | Replace with MTL throughout or add ε-tolerance formally |
| HC3 | Float non-determinism invalidates "identical traces" claim | Major | Specify floating-point mode requirements for reproducibility |
| A2 | MRTK extraction excludes custom code containing most deadlocks | Critical | Revise evaluation claims; spec-first evaluation is viable alternative |
| A3 | Over-approximation causes false negatives in deadlock detection | Critical | Revise soundness claim; over-approximation is wrong direction for deadlock |
| PA1 | Ciancia et al. spatial model checking not cited; likely overlap | Critical | Prior art audit before any further development |

*Fourteen critical or major findings. The theory stage is not ready for implementation.*

---

*Red-team review completed. The authors should interpret this document as a required checklist, not an assessment of project value. Every item above can be fixed. None of them should be discovered by a reviewer at submission time.*
