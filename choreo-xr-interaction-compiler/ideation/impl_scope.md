# Implementation Scope Breakdown: XR Event Calculus Compiler

**Project**: Domain-specific language compiler and incremental runtime for spatial-temporal
interaction patterns in mixed-reality scenes, with R-tree-backed event automata and formal
verification (reachability + deadlock), running entirely on CPU.

**Languages**: Rust (core) + Python (evaluation/tooling)

**Date**: 2025-07-17
**Role**: Implementation Scope Lead — Crystallization Phase

---

## Executive Honesty Statement

This document is a brutally honest assessment. I will not pad estimates to hit a target.
Where subsystems are standard compiler/runtime engineering, I say so. Where genuine novelty
exists, I explain exactly what makes it hard. The 150K LoC target is assessed at the end
with a clear verdict.

---

## Subsystem Breakdown

### 1. DSL Parser and Type Checker

**Purpose**: Lex, parse, and type-check the spatial-temporal interaction DSL. Produces a
typed AST for downstream compilation. The DSL must express spatial regions (volumes,
surfaces, rays), temporal intervals and constraints, event patterns with guards, and
interaction choreographies between entities.

**Estimated LoC**: 13,000–16,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Token definitions + lexer | 1,800 | Unicode-aware, spatial literal syntax (e.g., `sphere(0,0,0; r=1.5)`) |
| AST node definitions | 2,500 | ~60 node types for spatial, temporal, event, and choreography constructs |
| Recursive-descent parser | 4,000 | Operator precedence climbing for spatial set algebra (`∩`, `∪`, `\`) |
| Name resolution + scoping | 2,000 | Lexical scoping with entity-qualified names |
| Type checker core | 3,000 | Bidirectional type checking with spatial-temporal constraint propagation |
| Pretty-printer / AST dump | 1,200 | Round-trippable for incremental compilation cache validation |

**Key Technical Challenges**:
- Designing a concrete syntax that is expressive enough for spatial-temporal patterns without
  becoming a visual programming language manqué. Users must specify 3D regions, temporal
  windows, and event sequencing in text.
- Parsing spatial set-algebra expressions with correct precedence and associativity while
  maintaining good error recovery.
- Type-checking temporal constraints (e.g., "event A must occur within 500ms of event B
  while entity is inside region R") requires unifying spatial and temporal dimensions in the
  type system.

**Genuine Complexity**: ~25% novel. Parsing and type-checking are well-understood compiler
problems. The novelty is in the DSL design itself—inventing a textual syntax for
spatial-temporal choreography that is both ergonomic and formally precise. The type checker
must reject programs with spatially unrealizable constraints at compile time, which requires
geometric reasoning during type checking—this is not standard.

**Dependencies**: None (foundational subsystem). Feeds into all downstream subsystems.

---

### 2. Spatial-Temporal Type System

**Purpose**: Define and enforce a type discipline over spatial regions, temporal intervals,
and their compositions. This is the formal backbone that lets the compiler reject
physically impossible interaction patterns statically.

**Estimated LoC**: 9,000–11,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Spatial type definitions (regions, volumes, surfaces) | 1,500 | Parametric types over dimension (2D/3D) |
| Temporal type definitions (intervals, instants, durations) | 1,200 | Allen's interval algebra as types |
| Spatial-temporal product types | 1,500 | 4D space-time regions with projections |
| Subtyping lattice and joins/meets | 2,000 | Spatial containment ⊆ as subtyping |
| Constraint solver for spatial predicates | 2,000 | SAT-backed geometric constraint checking |
| Type inference engine | 1,500 | Local type inference with spatial constraint propagation |

**Key Technical Challenges**:
- Spatial subtyping is undecidable in general (reduces to geometric containment). Must
  restrict to a decidable fragment (convex polytopes, ellipsoids, CSG combinations of
  primitives) while remaining useful.
- Allen's 13 interval relations must compose correctly with spatial containment, producing
  a product lattice that has non-trivial join and meet operations.
- The constraint solver must be fast enough for interactive use (incremental compilation)
  yet precise enough to catch real bugs.

**Genuine Complexity**: ~60% novel. Spatial type systems exist in PL research but are rare
in practice. Combining spatial and temporal types into a single lattice with decidable
subtyping is a genuine research contribution. The constraint solver is non-trivial—it must
handle 3D geometric predicates (convex hull containment, intersection non-emptiness) which
are computational geometry problems, not standard type theory.

**Dependencies**: Feeds into (1) Parser/Type Checker for enforcement, (3) Event Calculus
for fluent typing, (5) Automata Compiler for transition guards.

---

### 3. Event Calculus Axiom Engine

**Purpose**: Implement the Event Calculus (EC) formalism—specifically a variant of the
Discrete Event Calculus—extended with spatial predicates. This engine evaluates whether
fluents (time-varying properties) hold at given timepoints, computes event effects, and
handles the frame problem via circumscription.

**Estimated LoC**: 11,000–14,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Fluent representation (spatial + propositional) | 1,500 | Fluents over 3D state, not just booleans |
| Event representation and effect axioms | 1,500 | Initiates/Terminates/Releases predicates |
| Domain-closure + circumscription | 2,500 | Implements CWA and minimization of change |
| Temporal reasoning core (Happens, HoldsAt, etc.) | 2,500 | Discrete Event Calculus axioms |
| Spatial-event integration | 2,000 | Events triggered by spatial predicates (enter/exit region) |
| Narrative construction and query evaluation | 2,000 | Forward chaining + backward chaining query modes |
| Caching and memoization layer | 1,500 | Avoid recomputation of stable fluent states |

**Key Technical Challenges**:
- The Event Calculus with circumscription is computationally expensive. Naive evaluation of
  HoldsAt queries requires examining all events up to the query timepoint. Need indexing
  strategies to make this tractable.
- Extending EC with spatial fluents (e.g., "entity X is inside region R") means fluent
  state is geometric, not boolean. Effect axioms must describe geometric transformations.
- The frame problem becomes harder with spatial fluents because "nothing changes unless
  caused" must account for continuous spatial motion, not just discrete state flips.
- Must support both deductive (given events, what holds?) and abductive (given desired
  state, what events are needed?) reasoning modes.

**Genuine Complexity**: ~70% novel. The Discrete Event Calculus itself is well-studied
(Mueller, Shanahan), but extending it with first-class spatial predicates and geometric
fluents is genuinely novel. Implementing circumscription efficiently for a spatial domain
is a hard problem—you cannot just use an off-the-shelf SAT solver because the constraints
are geometric, not propositional. This is arguably the most intellectually demanding
subsystem.

**Dependencies**: (2) Spatial-Temporal Type System for fluent types, (4) R-tree for
spatial predicate evaluation.

---

### 4. R-tree Spatial Index

**Purpose**: Maintain a spatial index over all scene entities, regions, and interaction
zones. Supports efficient spatial queries (intersection, containment, k-nearest-neighbor)
that are used by the event calculus engine, the runtime, and the verifier.

**Estimated LoC**: 7,000–9,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| R*-tree core (insert, delete, search, split) | 3,000 | R*-tree variant for better query performance |
| Bulk-loading (STR packing) | 800 | For initial scene loading |
| Spatial predicates library | 1,500 | Intersection, containment, distance, ray-cast |
| Temporal extension (time-parameterized entries) | 1,500 | Entries valid over time intervals, enabling historical queries |
| Incremental update with versioning | 1,200 | Copy-on-write nodes for snapshot isolation |

**Key Technical Challenges**:
- A standard R-tree library (e.g., `rstar` crate) handles static spatial indexing well, but
  this project needs temporal parameterization—entries exist over time intervals, and queries
  like "what was entity X intersecting at time T?" require historical access.
- Incremental updates with versioning (for the incremental compilation engine) add
  significant complexity to node splitting and rebalancing.
- The spatial predicates must handle not just AABBs but oriented bounding boxes, spheres,
  convex hulls, and CSG combinations—this requires a predicate dispatch system.

**Genuine Complexity**: ~35% novel. R-trees are textbook data structures with excellent
existing Rust implementations. The novelty is: (a) temporal parameterization turning it
into a quasi-4D index, (b) versioned/persistent structure for incremental compilation
snapshots, and (c) the rich predicate library for non-AABB geometry. An honest engineer
could use `rstar` as a starting point and build ~4,000 LoC of extensions on top, but the
versioning and temporal aspects require substantial custom work.

**Dependencies**: None directly (utility subsystem). Consumed by (3) Event Calculus,
(7) Runtime, (8) Headless Simulator, (9) Reachability Checker.

---

### 5. Event Automata Compiler (DSL → Automata)

**Purpose**: Lower the typed AST from the front-end into a network of communicating event
automata. Each automaton represents one interaction pattern; the network represents the full
choreography. This is the core compilation pass.

**Estimated LoC**: 13,000–16,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| IR definition (mid-level representation) | 2,000 | Between AST and automata; explicit control flow |
| AST → IR lowering | 3,000 | Desugaring, pattern expansion, choreography flattening |
| IR → automata construction | 3,500 | Thompson-style construction extended with spatial guards |
| Automata optimization passes | 3,000 | ε-elimination, state minimization, guard simplification |
| Automata composition (product, sync) | 2,000 | Composing automata via shared events/fluents |
| Serialization / bytecode emission | 1,500 | Compact representation for runtime consumption |

**Key Technical Challenges**:
- Standard regular-expression-to-NFA construction (Thompson's) must be extended with
  spatial and temporal guards on transitions. A transition fires not just on an event symbol
  but on "event E occurs while entity is in region R and fluent F holds." These guards are
  not finite-alphabet symbols—they are predicates over continuous state.
- Automata composition for choreographies requires synchronization on shared events, which
  can cause exponential state-space blowup. Must implement on-the-fly composition with
  symbolic state representation.
- Optimization passes must reason about spatial guard satisfiability—two guards are
  "equivalent" only if they describe the same spatial region, which is a computational
  geometry problem.

**Genuine Complexity**: ~50% novel. The compilation pipeline (AST → IR → automata →
optimized automata) is standard compiler architecture. The novelty is entirely in the
nature of the automata: transitions guarded by spatial-temporal predicates, composition
over geometric state, and optimization requiring geometric reasoning. This is not a
standard finite automaton compiler—it is closer to a hybrid automaton compiler, which is
a genuine research topic.

**Dependencies**: (1) Parser for typed AST, (2) Type System for guard types,
(3) Event Calculus for fluent semantics.

---

### 6. Incremental Compilation Engine

**Purpose**: Avoid full recompilation when the DSL source changes. Track dependencies at
fine granularity and recompute only affected compilation artifacts.

**Estimated LoC**: 9,000–12,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Dependency graph representation | 1,500 | Fine-grained: expression-level, not just file-level |
| Change detection (source diffing) | 1,500 | Structural diff on AST, not text |
| Query system (Salsa-inspired) | 3,500 | Memoized, demand-driven recomputation |
| Cache persistence (on-disk) | 1,500 | Serialize/deserialize compilation artifacts |
| Invalidation propagation | 1,500 | Transitive invalidation through dependency graph |
| Incremental automata re-linking | 1,500 | Patch compiled automata network without full rebuild |

**Key Technical Challenges**:
- The Salsa framework (used by rust-analyzer) provides a model, but spatial-temporal
  compilation has non-standard invalidation patterns. Changing a spatial region definition
  can invalidate type-checking, event calculus evaluation, R-tree index, and compiled
  automata simultaneously. The dependency graph is denser than in a standard language.
- Incremental automata re-linking is especially hard: modifying one automaton in a
  synchronized network may require recomputing the product with all neighbors.
- On-disk cache serialization must handle the R-tree index, which has complex internal
  structure.

**Genuine Complexity**: ~30% novel. Incremental compilation is well-studied (Salsa,
Build Systems à la Carte). The engineering is substantial but the ideas are known. The
novelty is in applying incremental computation to a domain with geometric dependencies,
where "has this changed?" is a geometric predicate, not a hash comparison.

**Dependencies**: All compilation subsystems (1, 2, 3, 5). This wraps them.

---

### 7. Runtime Execution Engine

**Purpose**: Execute the compiled automata network against a stream of events from the
headless simulator. Manages automata state, evaluates spatial-temporal guards, dispatches
events, and reports interaction pattern matches/violations.

**Estimated LoC**: 11,000–14,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Automata interpreter (state machine stepper) | 2,500 | Handles non-determinism via token passing |
| Guard evaluator (spatial + temporal predicates) | 2,500 | Queries R-tree, evaluates fluents |
| Event dispatch and routing | 2,000 | Route events to relevant automata by spatial locality |
| State snapshot and rollback | 2,000 | For speculative execution and backtracking |
| Timing and scheduling (virtual clock) | 1,500 | Deterministic virtual time for reproducibility |
| Execution trace recording | 1,500 | Full trace for debugging and verification |
| Runtime error handling | 1,000 | Spatial constraint violations, temporal deadline misses |

**Key Technical Challenges**:
- Non-deterministic automata with spatial guards require exploring multiple possible
  transitions simultaneously. This is token-passing NFA execution, but each "token"
  carries spatial context (entity positions), making it much more expensive than standard
  NFA simulation.
- Guard evaluation involves R-tree queries on every step, which must be fast. The runtime
  must maintain the R-tree incrementally as entities move.
- Deterministic virtual time is essential for reproducible testing, but spatial guards
  create implicit timing dependencies (entity reaches region at time T). Must handle
  "continuous" events (enter/exit) from discrete time steps.

**Genuine Complexity**: ~45% novel. Automata execution is standard, but the spatial guard
evaluation, R-tree maintenance, and continuous-event detection from discrete steps add
genuine complexity. The snapshot/rollback mechanism for speculative execution of
non-deterministic spatial automata is non-trivial.

**Dependencies**: (4) R-tree, (5) Automata Compiler output, (3) Event Calculus for
fluent evaluation.

---

### 8. Headless Scene Simulator

**Purpose**: Simulate a mixed-reality scene without any GPU or display. Manages entities
with positions, orientations, and bounding volumes. Generates the event stream that drives
the runtime engine. Must be deterministic and reproducible.

**Estimated LoC**: 10,000–13,000 (Rust + Python)

| Component | LoC | Notes |
|-----------|-----|-------|
| Scene graph (entity hierarchy, transforms) | 2,500 | Rust; hierarchical transforms, lazy world-space |
| Entity physics (bounding volume movement) | 2,000 | Rust; linear interpolation, scripted paths, simple collision |
| Collision detection (narrow phase) | 2,000 | Rust; GJK/EPA for convex shapes, sweep-and-prune broad phase |
| Event generation (enter/exit/proximity) | 1,500 | Rust; derived from spatial predicate changes between steps |
| Scene description format + loader | 1,500 | Rust; declarative scene files (TOML/custom format) |
| Scripted scenario playback | 1,500 | Python; programmatic scene manipulation for testing |
| Deterministic RNG and time control | 500 | Rust; seeded PRNG for reproducible stochastic scenarios |

**Key Technical Challenges**:
- "Physics-lite" is deceptively hard. We don't need full rigid-body dynamics, but we do
  need correct collision detection for the spatial primitives the DSL supports (spheres,
  boxes, convex hulls, CSG). GJK/EPA implementation is non-trivial.
- Event generation from discrete time steps requires detecting when a spatial predicate
  *changes* between steps (entity enters/exits region). This is a temporal coherence
  problem—must detect zero-crossings of signed distance functions.
- The simulator must be fast enough to run thousands of scenarios for evaluation without
  a GPU, so spatial queries must be carefully optimized.

**Genuine Complexity**: ~35% novel. Headless scene simulation exists (e.g., game engine
headless modes), but purpose-built for event calculus evaluation with formal properties
(determinism, reproducibility, complete event generation) is somewhat novel. The collision
detection and event generation are standard computational geometry—hard to implement but
well-understood.

**Dependencies**: (4) R-tree for broad-phase spatial queries, (7) Runtime for event
consumption. Feeds events into (7).

---

### 9. Reachability Checker

**Purpose**: Given a compiled automata network and a set of spatial constraints, determine
whether a target configuration (set of automata states + spatial predicate) is reachable
from the initial configuration. Used to verify that interaction patterns can actually occur.

**Estimated LoC**: 12,000–15,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| State space representation (symbolic) | 2,500 | BDD or zone-based for temporal, polyhedra for spatial |
| Exploration engine (BFS/DFS + heuristic) | 2,500 | Configurable strategy; supports bounded model checking |
| Spatial abstraction and refinement (CEGAR) | 3,000 | Counterexample-guided abstraction refinement for spatial state |
| Counterexample generation and validation | 2,000 | Concrete trace extraction from abstract counterexample |
| State space reduction techniques | 2,000 | Partial-order reduction, symmetry breaking for spatial symmetries |
| Query language for reachability properties | 1,500 | Subset of CTL with spatial predicates |

**Key Technical Challenges**:
- The state space is enormous: it is the product of all automata states × spatial
  configuration space (which is continuous). Direct enumeration is impossible.
- Must use abstraction: represent spatial state symbolically (e.g., which regions are
  occupied, not exact coordinates). But abstraction can be too coarse—CEGAR loop refines
  it by examining spurious counterexamples.
- Spatial CEGAR is novel: the abstraction is geometric (merge nearby regions), and
  refinement splits geometric regions—this is not standard predicate abstraction.
- Partial-order reduction for spatial automata is non-trivial because spatial guards create
  implicit dependencies between otherwise independent automata.

**Genuine Complexity**: ~65% novel. Model checking is well-studied, but reachability
checking over hybrid state spaces (discrete automata × continuous spatial state) with
CEGAR is an active research area. The spatial abstraction-refinement loop is genuinely
novel—most CEGAR implementations work over predicate abstractions, not geometric ones.
This subsystem is where the project contributes most to the research frontier.

**Dependencies**: (5) Automata Compiler for automata network, (2) Type System for spatial
constraints, (4) R-tree for spatial predicate evaluation in concrete execution.

---

### 10. Deadlock Detector

**Purpose**: Detect situations where the automata network reaches a state from which no
progress is possible—all automata are waiting for events that can never occur given the
current spatial configuration.

**Estimated LoC**: 6,000–8,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Wait-for graph construction | 1,500 | Automata waiting for events guarded by spatial predicates |
| Cycle detection (Tarjan's + spatial feasibility) | 1,500 | Cycles are deadlocks only if spatial guards are satisfiable |
| Spatial feasibility checker | 1,500 | Can the wait cycle actually occur given spatial constraints? |
| Deadlock characterization and reporting | 1,000 | Minimal deadlock set, causal explanation |
| Livelock detection (bounded progress) | 1,000 | Detect infinite loops without progress via state repetition |

**Key Technical Challenges**:
- Standard deadlock detection (cycle in wait-for graph) is necessary but not sufficient.
  A cycle is a real deadlock only if the spatial guards on the waiting transitions are
  simultaneously satisfiable—i.e., there exists a spatial configuration where all entities
  are stuck. This requires solving a spatial constraint satisfaction problem.
- Livelock detection (the system moves but makes no progress) is harder than deadlock
  and requires defining "progress" in terms of the interaction choreography.
- Must integrate with the reachability checker: a deadlock is reachable only if the
  deadlock configuration is reachable from the initial state.

**Genuine Complexity**: ~55% novel. Deadlock detection for concurrent systems is textbook.
The novelty is the spatial feasibility check—a cycle is spurious if the required spatial
configuration is geometrically impossible. This requires solving a conjunction of spatial
predicates, which is a computational geometry problem layered on top of standard concurrency
analysis.

**Dependencies**: (5) Automata Compiler, (9) Reachability Checker (for reachable deadlock
detection), (2) Type System for spatial constraint representation.

---

### 11. Cross-Platform Abstraction Layer

**Purpose**: Abstract over platform-specific concerns to enable the same compiled
choreographies to target multiple XR platforms (OpenXR, WebXR, proprietary SDKs) via
platform adapter plugins. Since everything runs on CPU, this is about event format
translation and coordinate system normalization, not rendering.

**Estimated LoC**: 6,000–8,000 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Platform trait definitions | 1,000 | Abstract interfaces for event sources, coordinate systems |
| Coordinate system normalization | 1,200 | Left/right-handed, Y-up/Z-up, unit conversions |
| Event format adapters | 2,000 | Translate platform-specific events to canonical EC events |
| OpenXR adapter | 1,500 | Map OpenXR interaction profiles to DSL event types |
| WebXR adapter | 1,200 | Map WebXR input sources and spaces |
| Platform capability negotiation | 500 | Query what spatial primitives a platform supports |

**Key Technical Challenges**:
- XR platforms differ significantly in coordinate conventions, event models, and interaction
  paradigms. OpenXR uses action sets and interaction profiles; WebXR uses input sources and
  reference spaces. The abstraction must be rich enough to preserve semantic content.
- Coordinate system normalization sounds trivial but involves subtle issues with
  handedness, units, and reference frame origins that cause real bugs.

**Genuine Complexity**: ~15% novel. This is standard platform abstraction engineering.
The coordinate normalization and event translation are well-understood problems with
well-understood solutions. The main difficulty is getting the details right across
platforms, which is tedious but not intellectually novel.

**Dependencies**: (7) Runtime for event consumption, (8) Simulator for platform-specific
scene loading.

---

### 12. Test Harness and Oracle

**Purpose**: Automated testing infrastructure. The oracle generates expected outcomes from
the formal Event Calculus specification—given a scenario, the EC engine computes what
*should* happen, and the test harness verifies the runtime produces the same result.

**Estimated LoC**: 8,000–11,000 (Rust + Python)

| Component | LoC | Notes |
|-----------|-----|-------|
| Test runner / execution framework | 2,000 | Rust; parallel test execution with deterministic scheduling |
| Oracle generator (EC-derived expectations) | 2,500 | Rust; forward-chain EC to compute expected trace |
| Trace comparator (actual vs. expected) | 1,500 | Rust; semantic comparison with spatial/temporal tolerance |
| Property-based test generation | 2,000 | Rust; QuickCheck-style generation of random scenes/choreographies |
| Fuzzing harness (for parser, compiler) | 1,000 | Rust; structured fuzzing with `cargo-fuzz` / `libfuzzer` |
| Test report generation | 1,000 | Python; HTML/JSON reports with spatial visualizations |

**Key Technical Challenges**:
- Oracle generation from EC is the key idea: the EC axioms define a "ground truth" that
  the compiled automata must agree with. Any disagreement is a compiler bug. This is a
  powerful testing approach but requires the EC engine to be correct (or at least
  independently validated).
- Trace comparison with spatial tolerance is non-trivial: events may occur at slightly
  different times or positions due to floating-point differences. Must define "close enough"
  formally—probably using spatial ε-balls and temporal δ-windows.
- Property-based generation of random scenes and choreographies requires generators that
  produce *valid* inputs (well-typed, spatially realizable), which is hard.

**Genuine Complexity**: ~40% novel. Test frameworks are standard. The oracle-from-EC
approach is genuinely novel and powerful—it gives you a differential testing oracle for
free from the formal specification. The spatial-tolerant trace comparison and typed random
generation are non-trivially different from standard testing.

**Dependencies**: (3) Event Calculus for oracle computation, (7) Runtime for execution,
(8) Simulator for scenario execution.

---

### 13. CLI and Configuration

**Purpose**: Command-line interface for the compiler, runtime, and verification tools.
Configuration management for project settings, platform targets, and verification
parameters.

**Estimated LoC**: 4,000–5,500 (Rust + Python)

| Component | LoC | Notes |
|-----------|-----|-------|
| CLI argument parsing (clap-based) | 1,000 | Rust; subcommands for compile, run, verify, test |
| Configuration file handling (TOML) | 800 | Rust; project config, platform targets, solver params |
| REPL / interactive mode | 1,500 | Rust; step-through execution, query automata state |
| Progress reporting and output formatting | 800 | Rust; colored terminal output, progress bars for verification |
| Python CLI wrapper | 400 | Python; thin wrapper for evaluation scripts |

**Key Technical Challenges**:
- The REPL must support stepping through automata execution, querying spatial state at any
  timepoint, and visualizing (in text) the current automata configuration. This is a
  non-trivial interactive debugger.
- Configuration must handle verification parameters (search depth, abstraction granularity)
  that significantly affect correctness and performance.

**Genuine Complexity**: ~10% novel. This is standard CLI/config engineering. The REPL adds
some complexity but is not fundamentally novel. Mostly plumbing.

**Dependencies**: All subsystems (this is the user-facing entry point).

---

### 14. Evaluation and Benchmark Infrastructure

**Purpose**: Fully automated evaluation pipeline. Generate benchmark scenes, run
compilation and verification, measure performance, and produce publication-ready results.
No human annotation or studies.

**Estimated LoC**: 10,000–13,000 (Python + some Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Benchmark scene generators | 3,000 | Python; parametric generation of interaction scenarios |
| Scalability benchmarks (entity count, pattern complexity) | 1,500 | Python; sweep parameters, measure compile/verify/run time |
| Correctness evaluation (oracle-based) | 1,500 | Python; run test oracle over benchmark suite, aggregate results |
| Verification benchmark (reachability/deadlock) | 1,500 | Python; measure state space size, verification time |
| Statistical analysis and visualization | 2,000 | Python; confidence intervals, regression detection, matplotlib plots |
| Performance regression CI | 1,000 | Python; compare against baseline, flag regressions |
| Report generation (LaTeX tables, plots) | 1,000 | Python; automated paper-ready output |

**Key Technical Challenges**:
- Generating realistic benchmark scenes requires a model of what real XR interactions look
  like. Must parameterize: number of entities, spatial complexity, temporal density, pattern
  nesting depth, choreography concurrency.
- Scalability evaluation must isolate variables—changing entity count while holding
  interaction complexity constant, and vice versa.
- Verification benchmarks must include known-hard instances (exponential state spaces) and
  known-easy instances to characterize the verifier's practical performance.

**Genuine Complexity**: ~25% novel. Benchmark infrastructure is standard software
engineering. The novelty is in generating realistic XR interaction benchmarks—there is no
established benchmark suite for spatial-temporal interaction choreographies, so we must
create one. The parametric scene generators must produce scenes that are complex enough to
stress the system but semantically meaningful.

**Dependencies**: (8) Simulator, (7) Runtime, (9) Reachability, (10) Deadlock, (1) Parser
(for compiling benchmark DSL files).

---

### 15. Error Reporting and Diagnostics

**Purpose**: Rich, user-friendly error messages for all stages of the pipeline: parse
errors, type errors, spatial constraint violations, verification failures, runtime errors.
Modeled after rustc/elm-style diagnostics.

**Estimated LoC**: 5,500–7,500 (Rust)

| Component | LoC | Notes |
|-----------|-----|-------|
| Error type hierarchy | 1,000 | Enum-based error types per pipeline stage |
| Source span tracking | 800 | Byte-offset spans, line/column computation |
| Diagnostic renderer (codespan-style) | 2,000 | Annotated source snippets, underlines, colors |
| Spatial error visualization (text-based) | 1,200 | ASCII art for spatial constraint violations |
| Fix suggestions | 800 | "Did you mean region R₂ instead of R₁?" |
| Verification counterexample rendering | 1,000 | Step-by-step trace of how a deadlock/unreachability occurs |

**Key Technical Challenges**:
- Spatial errors are inherently geometric and hard to explain in text. "Region R₁ does not
  intersect region R₂" needs a spatial visualization, but we cannot use a GPU. Must produce
  informative ASCII/Unicode art or structured text descriptions.
- Verification counterexamples can be long traces through the automata network. Must
  produce a minimal, human-understandable explanation, not just a raw state dump.

**Genuine Complexity**: ~20% novel. Diagnostic rendering is well-understood (codespan,
ariadne crates provide foundations). The novelty is in spatial error visualization and
verification counterexample explanation—translating geometric and temporal concepts into
text is a genuine UX challenge.

**Dependencies**: All subsystems (each must produce structured errors).

---

## Summary Table

| # | Subsystem | LoC (mid-est) | Novel % | Key Challenge |
|---|-----------|---------------|---------|---------------|
| 1 | DSL Parser & Type Checker | 14,500 | 25% | Textual syntax for spatial-temporal choreography |
| 2 | Spatial-Temporal Type System | 10,000 | 60% | Decidable spatial subtyping + Allen's intervals |
| 3 | Event Calculus Axiom Engine | 12,500 | 70% | EC with geometric fluents and circumscription |
| 4 | R-tree Spatial Index | 8,000 | 35% | Temporal parameterization + versioned persistence |
| 5 | Event Automata Compiler | 14,500 | 50% | Hybrid automata with spatial-temporal guards |
| 6 | Incremental Compilation | 10,500 | 30% | Geometric dependency invalidation |
| 7 | Runtime Execution Engine | 12,500 | 45% | NFA token-passing with spatial guard evaluation |
| 8 | Headless Scene Simulator | 11,500 | 35% | Deterministic event generation from discrete steps |
| 9 | Reachability Checker | 13,500 | 65% | Spatial CEGAR over hybrid state spaces |
| 10 | Deadlock Detector | 7,000 | 55% | Spatial feasibility of wait-for cycles |
| 11 | Cross-Platform Abstraction | 7,000 | 15% | Coordinate/event normalization across XR platforms |
| 12 | Test Harness & Oracle | 9,500 | 40% | Differential testing via EC-derived oracle |
| 13 | CLI & Configuration | 4,750 | 10% | Interactive REPL with spatial state inspection |
| 14 | Evaluation/Benchmarks | 11,500 | 25% | Parametric XR scene generation (no existing suite) |
| 15 | Error Reporting & Diagnostics | 6,500 | 20% | Spatial error visualization in text |
| | **TOTAL (source code)** | **~153,750** | | |

---

## Honest Assessment

### Does it genuinely reach 150K+ LoC?

**Verdict: Borderline yes, but only with disciplined engineering.**

The mid-range estimates sum to ~154K LoC of source code (Rust + Python). This is within
the range of honest estimates (low end: ~127K, high end: ~172K). Here is my candid
breakdown of confidence:

**High confidence (±15%)**: Subsystems 1, 5, 6, 7, 8, 13, 14, 15 — These are well-scoped
with clear analogues in existing systems. I can estimate them accurately.

**Medium confidence (±25%)**: Subsystems 2, 3, 4, 9, 10, 12 — These involve genuine
research problems where scope can expand or contract based on how hard the problems turn
out to be. The spatial CEGAR loop (subsystem 9) could easily be 18K or could be simplified
to 8K depending on the chosen abstraction.

**What about tests?** The 154K figure is *source code only*. For a project emphasizing
automated evaluation and formal verification, test code is substantial:

| Category | Estimated LoC |
|----------|---------------|
| Unit tests (Rust, embedded in modules) | ~25,000 |
| Integration tests (Rust) | ~15,000 |
| Property-based test generators | ~5,000 |
| Fuzz harnesses | ~3,000 |
| Python evaluation/test scripts | ~8,000 |
| Benchmark DSL files | ~5,000 |
| **Total test + eval code** | **~61,000** |

**Grand total (source + tests + benchmarks): ~215K LoC.**

With tests, 150K+ is comfortably achieved. Without tests, it is achievable but tight.

### What fraction is genuinely novel?

Weighted by LoC:

| Category | LoC | % of Total |
|----------|-----|------------|
| Genuinely novel (research-grade) | ~52,000 | ~34% |
| Non-trivial engineering (adapted from known techniques) | ~60,000 | ~39% |
| Standard plumbing (CLI, config, serialization, formatting) | ~42,000 | ~27% |

**~34% of the code is genuinely novel.** This is a healthy ratio. For comparison:
- A typical production compiler is ~10-15% novel (most is well-understood engineering).
- A research prototype is ~50-70% novel (but is usually small and fragile).
- This project sits between production and research, which is appropriate.

### Is the complexity genuine or inflated?

**The complexity is genuine, with two caveats:**

1. **The Event Calculus + spatial extensions is the real intellectual core.** Subsystems
   2, 3, 9, and 10 (the type system, EC engine, reachability, deadlock) account for ~43K
   LoC and ~60% average novelty. If these subsystems are implemented seriously, the project
   is a real contribution. If they are implemented as toy versions, the project collapses
   into a fancy DSL with a spatial library bolted on.

2. **The cross-platform layer (subsystem 11) is the weakest justification.** At 7K LoC
   with 15% novelty, it is mostly plumbing. It exists to justify "cross-platform" in the
   project title, but an honest evaluation could defer it to a future version and lose
   little scientific value.

### What would make 150K+ unambiguously genuine?

If the current scope feels tight, here are expansions that add *genuine* complexity (not
padding):

| Expansion | Additional LoC | Why It's Genuine |
|-----------|---------------|------------------|
| **Partial evaluation / specialization** — specialize automata for known-static spatial configurations | +8,000 | Genuinely hard compiler optimization; spatial partial evaluation is novel |
| **Probabilistic model checking** — extend reachability with stochastic spatial motion models | +12,000 | Stochastic hybrid systems verification is cutting-edge research |
| **LSP server for the DSL** — IDE support with spatial-aware completions and inline diagnostics | +10,000 | Non-trivial engineering; spatial-aware completion is novel |
| **Visual trace debugger (TUI)** — terminal-based visualization of automata execution over time | +6,000 | ratatui-based; genuine UX challenge for spatial-temporal data |
| **Compositional verification** — verify subsystems independently and compose proofs | +10,000 | Assume-guarantee reasoning for spatial-temporal properties is research-grade |

Adding any two of these would push the project to 170K+ with no honesty concerns.

### Recommended scope for maximum genuine novelty

If I were designing this project to maximize the ratio of genuine novelty to total LoC,
I would:

1. **Keep subsystems 1–10, 12, 14, 15** as specified (the compiler, verification, and
   evaluation core).
2. **Replace subsystem 11** (cross-platform abstraction) with **compositional
   verification** — this swaps 7K LoC of plumbing for 10K LoC of research.
3. **Add the probabilistic model checking extension** to subsystem 9 — this adds 12K LoC
   of cutting-edge research.
4. **Add an LSP server** — this adds 10K LoC and makes the project practically usable.

This yields ~176K source LoC, ~40% novel, with every subsystem pulling its weight.

---

## Dependency Graph (ASCII)

```
                    ┌─────────────┐
                    │  13. CLI &  │
                    │   Config    │
                    └──────┬──────┘
                           │ uses all
              ┌────────────┼────────────────┐
              │            │                │
     ┌────────▼──────┐  ┌─▼──────────┐  ┌──▼──────────────┐
     │  6. Incremental│  │ 14. Eval & │  │ 15. Error       │
     │  Compilation   │  │ Benchmarks │  │  Reporting      │
     └───────┬────────┘  └─────┬──────┘  └────────┬────────┘
             │ wraps            │ drives           │ consumed by all
    ┌────────┼────────┐        │
    │        │        │   ┌────▼─────┐
┌───▼──┐ ┌──▼───┐ ┌──▼──▼──┐  ┌────▼─────┐
│1.    │ │5.    │ │12. Test │  │8. Head-  │
│Parser│ │Auto- │ │Harness &│  │less      │
│& Type│ │mata  │ │Oracle   │  │Simulator │
│Check │ │Comp. │ └────┬────┘  └──┬───────┘
└──┬───┘ └──┬───┘      │         │
   │        │      ┌───▼─────────▼───┐
   │        │      │  7. Runtime     │
   │        │      │  Execution      │
   │        │      └───┬─────────────┘
   │        │          │
┌──▼────────▼──────────▼──┐
│  3. Event Calculus      │
│  Axiom Engine           │
└────────┬────────────────┘
         │
┌────────▼────────┐    ┌────────────────┐
│  2. Spatial-    │    │  4. R-tree     │
│  Temporal Types │    │  Spatial Index │
└─────────────────┘    └────────────────┘
         │                     │
         └─────────┬───────────┘
                   │
         ┌─────────▼──────────┐
         │  9. Reachability   │
         │  Checker           │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  10. Deadlock      │
         │  Detector          │
         └────────────────────┘

    ┌─────────────────────────┐
    │  11. Cross-Platform     │◄── Weakest link;
    │  Abstraction            │    consider replacing
    └─────────────────────────┘
```

---

## Final Verdict

**The project genuinely warrants ~154K LoC of source code** (215K+ with tests), provided
the formal verification subsystems (reachability, deadlock, event calculus) are implemented
with full rigor rather than as toy proofs-of-concept. The intellectual core—spatial-temporal
type theory, event calculus with geometric fluents, and spatial CEGAR—is genuinely
novel and hard. The project is not padded; it is an ambitious integration of
compiler construction, computational geometry, and formal methods. The 150K target is
achievable without dishonesty, but the honest range is 127K–172K depending on
implementation depth, so it should be treated as a target to validate during implementation
rather than a guarantee.
