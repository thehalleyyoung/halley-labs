# Parametric Accessibility Verification for Mixed Reality Scenes via Pose-Guarded Hybrid Automata (Amended)

**Slug:** `xr-affordance-verifier`

> **Amendment notice.** This is the amended crystallized problem statement, revised after
> a 3-expert verification panel scored the original at Value 4/10, Difficulty 7/10,
> Best-Paper 4/10, Laptop CPU 5/10. Seven binding amendments (A1–A7) are incorporated
> throughout. Each is tagged inline with `[A#]` for traceability.

---

## Problem

Mixed reality is the first user-interface paradigm where a user's **physical body** is
the input device. A 5'0" user with limited shoulder mobility cannot reach the same
virtual elements as a 6'2" user with full range of motion. A Quest 3 user with 6DOF hand
tracking navigates fundamentally different interaction paths than a PSVR2 user with wand
controllers. Today, XR developers discover these spatial accessibility failures only
through expensive manual testing with physically diverse testers—or worse, from user
complaints post-launch. No tool exists that can formally **verify** that an XR scene is
accessible across the population of human bodies and the ecosystem of target devices.

We propose a formal verification engine for XR spatial accessibility, targeting enterprise
XR domains where accessibility is already legally mandated: manufacturing training
(Boeing, Lockheed Martin), surgical simulation, and industrial maintenance under
Section 508 and ADA Title I employment accommodations.

[A6] The system ingests a Unity XR scene description (.unity YAML, .prefab, C#
MonoBehaviour scripts); WebXR and OpenXR format support is deferred to future work.

[A2] The system extracts common interaction patterns automatically via heuristic pattern
matching for standard Unity interaction idioms (OnClick, OnGrab, simple FSMs); complex
interactions require declarative annotations via a domain-specific language (DSL). It
does NOT claim to "extract interaction state machines implicit in scene components"—it
extracts common interaction patterns automatically; complex interactions require
declarative annotations.

From this extraction, the system constructs **Pose-Guarded Hybrid Automata (PGHA)**—a
formalism where automaton transitions are guarded by semialgebraic predicates over the
SE(3) pose space of parameterized human kinematic chains. The engine model-checks this
PGHA against accessibility specifications, verifying that every interaction is reachable
by every body parameterization in a target population distribution, on every target
device—or produces concrete counterexample traces identifying exactly which body type,
on which device, encounters which failure.

The fundamental computational challenge is the product space: human pose is a continuous
manifold (SE(3) for the torso × joint-angle tori for each limb), interaction state is a
discrete automaton potentially exponential in scene complexity, and parameterization by
body dimensions and device capabilities creates a further continuous parameter space.
Naively, this is undecidable. Our key technical contribution is a **semialgebraic zone
abstraction on SE(3)** that reduces the continuous pose space to finite regions while
preserving reachability, combined with a **compositional assume-guarantee framework**
that decomposes large scenes into independently verifiable interaction clusters.

A three-persona analysis architecture maps to concrete deployment scenarios:

- **Tier 1 — "XR Accessibility Linter":** Runs in Unity editor in <5 seconds, visual
  annotations on every interactable. Developer-facing.
- **Tier 2 — "CI/CD Accessibility Gate":** Runs in build pipeline in <10 minutes, blocks
  deployment on violations. Platform/team-facing.
- **Tier 3 — "Certification Engine":** Runs overnight for full formal guarantees.
  Enterprise compliance-facing.

[A1] A Month 2 prototype of the Zone Abstraction Engine on SE(3) is a mandatory
feasibility gate: if cylindrical algebraic decomposition produces >10⁵ cells or requires
>10 minutes for a single 4-joint interaction element, the approach pivots to
interval-based over-approximation.

This work opens a new application domain for hybrid systems verification: **verified XR
accessibility**. It bridges robotics workspace analysis (COMPAS FAB, Reuleaux
reachability maps) and CPS verification methodology (SpaceEx, UPPAAL) with
domain-specific adaptations for the XR interaction model. The novelty is this
domain-specific adaptation bridging both traditions, not a claim to define a new subfield.

The EU Accessibility Act (effective June 2025) and evolving ADA digital-space
interpretations will likely extend to XR as regulatory scope is interpreted, though the
EAA does not currently name XR explicitly. Enterprise XR domains—manufacturing training,
surgical simulation, industrial maintenance—already face legally mandated accessibility
requirements under Section 508 and ADA Title I, making the need immediate rather than
speculative. Over 1.3 billion people worldwide live with some form of disability;
parametric body diversity is not an edge case but the norm. Our verifier transforms XR
accessibility from a manual, expensive, incomplete process into an automated, exhaustive,
formal guarantee—the same revolution that static analysis brought to software security.

---

## Value Proposition

**Who needs this.** Enterprise XR developers building applications for diverse workforces
in regulated industries: manufacturing training (Boeing, Lockheed Martin), surgical
simulation, industrial maintenance—domains where accessibility is already legally mandated
under Section 508 and ADA Title I employment accommodations. Platform holders (Meta,
Apple, Sony, Microsoft) who must screen accessibility across app ecosystems. Accessibility
compliance teams facing EU Accessibility Act mandates (effective 2025, with XR-specific
implementing guidance still evolving—note that the EAA does not currently name XR
explicitly, though its scope will likely be interpreted to include immersive interfaces).
The 1.3 billion people with disabilities currently locked out of XR experiences by
invisible spatial barriers.

**Why desperately.** Current XR accessibility testing requires: (a) physical testers with
diverse body types and abilities, (b) every target device, (c) manual exploration of
every interaction sequence. This is prohibitively expensive, provably incomplete (you
cannot test all body types), and too slow for iterative development. In enterprise XR,
accessibility is already legally mandated—a single accessibility lawsuit costs more than a
year of development. Meta alone reviews 10,000+ Quest apps per year—manual accessibility
testing does not scale.

**What becomes possible.**
- Developers obtain automated accessibility audit reports **before** deployment.
- Platform holders automatically screen submissions for accessibility violations.
- Designers explore the **accessibility frontier**—seeing exactly which body-parameter
  ranges a design choice excludes.
- Compliance teams generate automated accessibility audit reports for regulatory review.
- The substantial fraction of XR accessibility bugs that are spatial rather than
  perceptual—unreachable buttons, occluded targets, impossible gestures—become
  automatically detectable.

[A7] **Demand validation gate.** By Month 3, structured feedback from ≥20 XR developers
(mix of indie, enterprise, platform) is required. If <25% express concrete interest in
adopting the tool, the value proposition is reconsidered before further investment. If
<10% express any interest, the project is terminated. See Kill-Chain Stage 2.

---

## Technical Difficulty

### Subsystem Breakdown

[A6] **Scope decisions.** Unity-only scene parsing; WebXR and OpenXR format support
deferred to future work. Cross-Device Compatibility Analyzer (originally a separate
subsystem) is merged into the Compositional Decomposition Engine—device parameters become
compositional assumptions rather than a separate analysis pass. Visualization is
simplified to text reports + 2D SVG diagrams; WebGL 3D visualization deferred to future
work.

**1. Scene Parser & Unity Format Adapter** (~14K LoC)
[A6] Parse Unity scene files (.unity YAML, .prefab, C# MonoBehaviour scripts for
interaction logic). Extract spatial layout, collision volumes, interaction trigger zones,
and wiring between UI elements. Unity's format has undocumented quirks; interaction logic
embedded in imperative code requires pattern-based extraction via domain-specific
heuristics (recognizing Unity's OnClick/OnGrab callback patterns, animation state machine
descriptors); scene hierarchies involve complex transform chains with non-uniform scaling
and nested coordinate frames. [Reduced from ~18K by dropping WebXR and OpenXR adapters.]

**2. Scene Graph Intermediate Representation** (~10K LoC)
Typed scene graph with spatial relationship algebra (contains, overlaps,
occludes-from-pose), interaction zones with parametric geometry, state machine attachment
points, and device-capability annotations. Must faithfully represent the *semantics* of
spatial interaction rather than raw geometry, support efficient spatial queries via R-tree
indexing, and maintain sound abstractions during coordinate transforms and scene
mutations. [A6: reduced from ~12K by narrowing to Unity-only IR.]

**3. Parameterized Kinematic Body Model** (~18K LoC)
Anthropometric database (ANSUR-II, CAESAR) encoded as parameterized kinematic chains.
Joint limit models vary continuously with body parameters. Reach envelope computation via
forward kinematics over joint-angle ranges. Device-specific movement constraints (seated
VR, standing AR, room-scale). Kinematic chains with joint coupling produce non-convex
reachable volumes in SE(3); parameterization by continuous body dimensions requires
symbolic interval computation; device constraints create complex geometric intersections
with the reachable workspace. [A6: reduced from ~20K by simplifying device constraint
models.]

**4. Interaction State Machine Extractor** (~12K LoC)
[A2] Two-mode extraction approach:

- **Automated mode:** Heuristic pattern-matched extraction for standard Unity interaction
  idioms (OnClick, OnGrab, simple FSMs, slider drag, dial rotation, proximity triggers).
  Produces over-approximate state machines only. Covers common patterns sufficient for
  Tier 1 analysis.
- **Annotated mode:** Developer-provided interaction state machine specifications via DSL
  annotations. Required for Tiers 2–3 soundness guarantees. Handles complex multi-step
  unlock sequences, conditional visibility/activation, and non-standard interaction logic.

Does NOT claim to extract arbitrary interaction state machines implicit in scene
components. Complex interactions require declarative annotations. [A2: reduced from ~15K
by scoping automated mode to recognized idioms only.]

**5. PGHA Constructor** (~20K LoC)
Build the product automaton: body pose dynamics (continuous) × interaction state
(discrete) × scene state (discrete). Transition guards are semialgebraic predicates
expressing "body in pose region R can activate interaction element E." The product-space
construction must be lazy (eager materialization is impossible); guard computation
requires geometric intersection of reach envelopes with interaction zones in SE(3); the
automaton structure must support efficient model-checking traversal.
[A6: reduced from ~22K.]

**6. Semialgebraic Zone Abstraction Engine** (~22K LoC)
The mathematical core. Implements cylindrical algebraic decomposition (CAD) adapted for
the SE(3) manifold structure. Computes zone partitions of pose space where reachability
predicates are invariant. Supports refinement (zone splitting for precision) and
coarsening (zone merging for performance). CAD is doubly-exponential in dimension in
general; the key research contribution exploits the kinematic structure of human bodies to
achieve tractable decompositions via the interaction-locality theorems (M3a/M3b).

[A1] **Feasibility gate:** A Month 2 prototype must demonstrate CAD producing ≤10⁵ cells
in ≤10 minutes for a single 4-joint interaction element. Failure triggers pivot to
interval-based over-approximation. [A6: reduced from ~25K.]

**7. Model Checker Core** (~24K LoC)
Three personas:
- **Tier 1 — "XR Accessibility Linter"** (<5 seconds): Interval arithmetic on bounding
  volumes, linear in scene size. Visual annotations on every interactable element in the
  Unity editor. Developer-facing.
- **Tier 2 — "CI/CD Accessibility Gate"** (<10 minutes): BDD-based symbolic model
  checking over zone graph. Blocks deployment on violations. Memory-bound at ~16GB for
  50-object scenes. Platform/team-facing.
- **Tier 3 — "Certification Engine"** (hours/overnight): Counterexample-guided
  abstraction refinement (CEGAR) bridging continuous geometry and discrete logic. Full
  formal guarantees. Enterprise compliance-facing.

Each tier requires distinct algorithms and data structures; the CEGAR loop must bridge
continuous geometry and discrete logic; BDD encoding of zone graphs demands careful
variable ordering heuristics. [A6: reduced from ~28K.]

**8. Compositional Decomposition Engine (with Device Compatibility)** (~16K LoC)
Assume-guarantee reasoning: decompose a scene into interaction clusters, verify each
independently under assumptions about neighbors, confirm mutual consistency of
assumptions. [A6] Device capabilities (tracking volume, controller geometry, input
modalities, interaction paradigm) are modeled as compositional assumptions—device
parameters enter the assume-guarantee framework as constraints on the pose-space interface
predicates rather than as a separate analysis pass. Automatic decomposition of scene
graphs into good clusters is itself NP-hard graph partitioning; assumption inference must
be fully automated; soundness of compositional reasoning over PGHA requires novel proof
obligations involving spatial interface predicates. [A6: original subsystem 10
(Cross-Device Compatibility Analyzer) merged here; net ~16K vs. original
15K + 10K = 25K.]

**9. Counterexample Engine** (~10K LoC)
Given an abstract counterexample trace, concretize it: find a specific body
parameterization, device configuration, and pose sequence witnessing the accessibility
failure. Generate interpolated pose trajectories that are biomechanically plausible.
Concretization requires solving nonlinear constraint satisfaction in high-dimensional pose
space; plausibility demands respecting joint velocity limits, collision avoidance, and
postural stability constraints. [A6: reduced from ~12K.]

**10. Property Specification DSL & Annotation Language** (~10K LoC)
[A2] Domain-specific language serving two purposes:

(a) XR accessibility property specifications: "all buttons reachable from seated position
for 5th–95th percentile arm length," "no interaction sequence requires simultaneous
two-handed reach exceeding shoulder width," "grab targets within cone of comfortable
vision for all HMDs."

(b) Interaction state machine annotations for complex interactions not covered by
automated extraction (required by A2 annotated mode).

Includes parser, type checker, and compiler to PGHA verification conditions. [Increased
from ~8K to accommodate A2 annotation language.]

**11. Benchmark & Evaluation Framework** (~10K LoC)
Procedural XR scene generator with controlled complexity parameters. [A4] Non-procedural
real XR scene corpus (≥5 scenes: 3 from open-source XR applications, 2 hand-crafted—see
Evaluation Plan). Bug injector planting known accessibility violations across 8
categories × 5 severity levels. [A3] Monte Carlo baseline harness (1M
body-parameterization samples × 1K interaction-sequence samples, stratified sampling).
Automated ground-truth labeling. Scalability measurement harness.

**12. Visualization & Reporting** (~4K LoC)
[A6] Text-based counterexample reports, annotated 2D SVG scene diagrams highlighting
accessibility failures, accessibility heat maps over body-parameter space (SVG), and
machine-readable compliance reports. WebGL 3D visualization deferred to future work.
[Reduced from ~8K by dropping WebGL.]

### LoC Summary

**Total: ~130–150K LoC** — Rust core (~105–120K), Python evaluation/benchmarks
(~15–20K), DSL tooling (~10–15K).

- Of this, **~85–110K is genuinely novel algorithmic work** (zone abstraction, model
  checking, PGHA construction, compositional reasoning, counterexample concretization).
- Remaining **~45–55K is important engineering** (Unity parsing, visualization,
  benchmarks, DSL frontend).
- Estimates exclude test code (~30–45K additional LoC for unit, integration, and
  property-based testing).

[A6: reduced from 175–210K. Unity-only parsing, merged device analyzer, simplified
visualization.]

---

## New Mathematics Required

**Honest assessment.** 2 strong contributions (M2, M3b at A−), 1 moderate (M3a at B+),
3 incremental (M1, M4, M7 at B to B−). Original M5 (body-capability monotonicity) and
M6 (device-capability subsumption lattice) are merged into M4 as parameter-reduction
lemmas, reducing the total from 8 to 6 distinct contributions.

**Antecedents acknowledged.** Robotics workspace analysis (COMPAS FAB, Reuleaux
reachability maps) is a clear conceptual antecedent for parameterized reach envelope
computation. CPS model checkers (SpaceEx, UPPAAL) provide the verification methodology
for hybrid automata. The novelty here is the domain-specific adaptation bridging both
traditions to the XR interaction model—not a fundamentally new verification paradigm.

### M1: PGHA Formalism and Operational Semantics
**Difficulty: B. (Incremental contribution.)**

Define Pose-Guarded Hybrid Automata: a hybrid automaton where the continuous state
inhabits a Lie group product (SE(3) × joint-angle torus T^n), discrete states represent
interaction configurations, and guards are semialgebraic sets in pose space parameterized
by body dimensions. Give operational semantics handling the interplay between continuous
body motion and discrete interaction state transitions, including invariants on dwell-time
and simultaneous guard satisfaction.

**Load-bearing:** every subsequent result depends on this formalism being well-defined and
expressive enough to capture real XR interactions.

**Novelty:** the specific combination of Lie group dynamics with semialgebraic guards over
parameterized kinematic chains is new; existing hybrid automata formalize different
continuous dynamics (typically R^n with polynomial ODEs). Draws on SpaceEx-style hybrid
automata semantics adapted to the Lie group setting.

### M2: Soundness of SE(3) Zone Abstraction
**Difficulty: A−. (Strong contribution.)**

Prove that cylindrical algebraic decomposition, adapted to the Lie group structure of
SE(3), produces a finite zone partition that is a sound over-approximation for
reachability analysis. Specifically: if a concrete pose trajectory witnesses a
reachability property, the corresponding abstract zone-graph trajectory preserves it. The
converse need not hold (over-approximation), but any reported violation must be
concretizable or refinable.

**Load-bearing:** without this theorem, the model checker has no soundness guarantee—
reported accessibility violations could be phantoms.

**Novelty:** CAD exists for R^n; adaptation to SE(3) with its non-Euclidean topology (the
rotation group SO(3) is non-contractible) requires handling chart transitions and
topological wrapping that have no precedent in the CAD literature. The difficulty is
elevated because SO(3) is non-contractible, requiring careful handling of chart
transitions in the CAD partition to ensure no reachability paths are lost at chart
boundaries. Conceptually related to Reuleaux reachability map discretization, but with
formal soundness guarantees that go beyond the robotics literature.

[A1] **Feasibility gate.** If the Month 2 prototype shows CAD producing >10⁵ cells or
>10 minutes for a single 4-joint interaction element, pivot to interval-based
over-approximation and weaken M2 to soundness of the interval-based scheme.

### M3a: Continuous-Dimension Tractability
**Difficulty: B+. (Moderate contribution.)**

For k-local interactions, the semialgebraic zone computation for each interaction element
operates in dimension O(k), independent of total body DOF n. Zone complexity is
O(d^(2^O(k))) per element rather than O(d^(2^n)).

**Load-bearing:** makes zone abstraction practical for realistic bodies.

**Novelty:** exploiting kinematic locality for CAD dimension reduction. Builds on robotics
kinematic chain decomposition techniques (cf. COMPAS FAB workspace analysis) but with
formal complexity guarantees.

### M3b: Discrete-State Tractability via Bounded Treewidth
**Difficulty: A−. (Strong contribution.)**

For scenes whose interaction-dependency graph has treewidth ≤ w, PGHA reachability over
the discrete interaction-state space is solvable in time O(|Z|^(w+1) · poly(m)) where
|Z| is the zone count and m is the number of interaction elements.

**Load-bearing:** controls discrete-state explosion for well-structured scenes.

**Novelty:** connects scene-graph treewidth (an empirical property of XR scenes) to
model-checking tractability; real XR scenes exhibit low treewidth because interactions are
spatially clustered. Note: pathological scenes with high treewidth remain exponential—the
theorem provides a structural condition, not a universal guarantee.

### M4: Compositional Assume-Guarantee for PGHA (with Parameter-Reduction Lemmas)
**Difficulty: B. (Incremental contribution; enriched by absorbing M5 and M6.)**

Extend assume-guarantee reasoning to PGHA: if a scene decomposes into spatial clusters
C₁…Cₙ with interface predicates over shared pose-space boundaries, and each Cᵢ is
verified under assumptions about neighbor behavior, then the whole-scene property holds.
Prove soundness via a circular assume-guarantee rule with spatial discharge conditions.

**Load-bearing:** without compositional reasoning, verification of scenes with >20
interactable objects is infeasible (product-state explosion).

**Novelty:** assume-guarantee rules exist for hybrid automata (Henzinger et al.) but not
with semialgebraic spatial guards; the spatial discharge conditions are new proof
obligations.

**Parameter-reduction sub-lemmas (formerly M5 and M6):**

- **M4a — Body-capability monotonicity (formerly M5, B−):** The accessibility relation
  is monotone in a natural partial order on body capabilities: if body B₁ has a strictly
  larger reach envelope than B₂ at every configuration, every single-step interaction
  accessible to B₂ is accessible to B₁. Extends to multi-step interactions with a
  trap-freedom condition. Reduces parametric verification from reasoning over a continuous
  body-parameter space to verification at finitely many boundary parameterizations. The
  subtlety lies in multi-step interactions where greater reach can enable intermediate
  states that become geometric traps.
- **M4b — Device-capability subsumption lattice (formerly M6, B):** Device capabilities
  formalized as elements of a bounded lattice (L, ⊑). Subsumption preserves
  accessibility properties. Reduces cross-device verification from |devices| × |body
  params| to verification at lattice-minimal elements. [A6: device parameters now enter
  as compositional assumptions in the assume-guarantee framework, tightening the
  connection between M4b and the core M4 result.]

### M7: Counterexample Concretization Completeness
**Difficulty: B. (Incremental contribution.)**

Prove that if the abstract model checker reports a violation, the concretization procedure
either produces a concrete biomechanically-plausible witness trajectory OR identifies a
spurious region and refines the zone abstraction to eliminate it. Termination: the CEGAR
loop terminates because each refinement strictly increases zone count, which is bounded by
the (finite) algebraic complexity of the scene.

**Load-bearing:** without this, users cannot trust that reported violations are real.

**Novelty:** adaptation of CEGAR to the PGHA setting with geometric concretization in
SE(3) and biomechanical plausibility constraints.

---

## Best Paper Argument

[A5] **Two-paper strategy** replaces the original single-paper "defines a new subfield"
framing. All "defines a new subfield" language is dropped; we instead claim this work
"opens a new application domain for hybrid systems verification" and "bridges robotics
workspace analysis and CPS verification."

### Theory Paper Target: CAV or TACAS

**Core contributions:** M2 (SE(3) zone abstraction soundness) + M3b (bounded-treewidth
tractability). Pure formal methods contribution demonstrating that hybrid systems
verification methodology can be adapted to a new application domain with non-trivial
topological complications.

**Why it merits acceptance:**

1. **Opens a new application domain for hybrid systems verification.** The adaptation of
   CAD to SE(3) with formal soundness guarantees (M2) is a genuine technical
   contribution—SO(3) non-contractibility creates chart-transition subtleties absent from
   standard R^n CAD.
2. **Structural tractability with practical relevance.** M3b connects scene-graph
   treewidth to model-checking complexity, providing a formal explanation for why real XR
   scenes (spatially clustered) admit tractable verification.
3. **Novel formalism with real teeth.** PGHA is not a trivial instantiation of hybrid
   automata. The SE(3) zone abstraction and interaction-locality results are genuinely new
   contributions to the formal methods literature.
4. **Bridges robotics workspace analysis and CPS verification.** Explicitly positions the
   contribution relative to COMPAS FAB/Reuleaux (geometry) and SpaceEx/UPPAAL
   (verification), demonstrating that the combination yields results neither tradition
   achieves alone.

### Tool Paper Target: UIST, OOPSLA, or ASE

**Core contributions:** Full Tier 1 + Tier 2 pipeline, DSL for accessibility properties
and interaction annotations, case studies on real XR scenes, developer walkthrough
scenario.

**Why it merits acceptance:**

1. **Immediately useful artifact.** Produces a tool that XR developers can deploy in
   production pipelines—from Unity editor linting (Tier 1, <5 seconds) to CI/CD gating
   (Tier 2, <10 minutes).
2. **Regulatory timeliness.** The EU Accessibility Act (effective June 2025) will likely
   extend to XR as regulatory scope is interpreted. Enterprise XR in manufacturing,
   surgery, and maintenance already faces Section 508 and ADA Title I mandates—positioning
   this tool for immediate adoption.
3. **Real-scene evaluation.** [A4] Case studies on ≥5 non-procedural XR scenes, including
   developer walkthrough scenarios, demonstrate practical utility beyond synthetic
   benchmarks.
4. **Bridges communities.** Substantive contributions to formal methods (PGHA
   verification), accessibility (parametric body modeling), and HCI (XR interaction
   formalization).
5. **Scale of implementation.** A ~130–150K LoC verification engine grounded in practical
   evaluation demonstrates full-stack research ambition.

---

## Evaluation Plan

All evaluation is **fully automated**—zero human annotation, zero user studies. (Developer
feedback in [A7] is structured interviews for demand validation, not a user study.)

### Benchmark Suite

- **500+ procedurally generated XR scenes** spanning: simple (5–10 objects), medium
  (20–50 objects), complex (100+ objects).
- **Scene types:** virtual menus, control panels, room-scale environments, collaborative
  workspaces.
- Each scene generated with controlled parameters: object density, interaction depth,
  spatial layout randomization, body-parameter sensitivity.
- [A4] **≥5 non-procedural real XR scenes** (procedural benchmarks alone are
  insufficient):
  - **3 from open-source XR applications:** Unity Asset Store free assets (e.g., VR
    Interaction Framework samples), A-Frame example scenes (converted to Unity IR for this
    evaluation), Meta Interaction SDK demo scenes.
  - **2 hand-crafted to represent realistic complexity:** (a) VR settings menu with nested
    panels, scroll views, toggles, and sliders; (b) industrial control panel with
    multi-step procedures, safety interlocks, and bimanual operations.
- **Bug injection:** 8 categories × 5 severity levels = 40 bug templates, randomly
  instantiated per procedural scene. Real scenes evaluated for naturally occurring
  accessibility issues.

### Bug Categories

1. **Unreachable elements** — placed beyond 5th-percentile reach envelope.
2. **Occlusion deadlocks** — element A occluded by element B, which requires interacting
   with A first.
3. **Pose-impossible gestures** — requiring simultaneous reaches exceeding human
   capability.
4. **Cross-device failures** — interaction requiring hand tracking on wand-only device.
5. **Sequential traps** — multi-step sequence leading to geometrically locked state.
6. **Anthropometric exclusion** — accessible to 50th but not 5th percentile.
7. **Seated-mode failures** — standing-optimized layout unusable when seated.
8. **Bimanual impossibilities** — two-handed interactions with incompatible spatial
   constraints.

### Metrics

- **Detection rate** (true positive rate) per bug category.
- **False positive rate** (reported bugs that are actually accessible).
- **Verification time** vs. scene complexity (scalability curves, log-log plots).
- **Counterexample precision** (concretized witnesses that are biomechanically valid).
- **Tier agreement rates** (fraction of Tier-3 bugs caught by Tier-1 and Tier-2).
- [A3] **Marginal detection rate over Monte Carlo baseline** — the percentage of bugs
  detected by formal verification that are missed by Monte Carlo sampling. This is the
  primary justification metric for the formal verification approach.

### Baselines

- [A3] **Monte Carlo baseline (strong, primary):** 1M body-parameterization samples × 1K
  interaction-sequence samples, with stratified sampling over anthropometric percentiles
  (5th, 25th, 50th, 75th, 95th) and device configurations. **Gate criterion:** formal
  verification must detect ≥10% of bugs missed by Monte Carlo to justify its
  computational cost. If marginal detection rate is <10%, the formal verification approach
  does not justify its cost over sampling (see Kill-Chain Stage 4).
- **Geometric heuristic checker** (bounding-box reachability, no state machines).
- **Ablated verifier variants** (no composition, no zone abstraction, no tiered analysis).
- **Automated exploration agent** (simulated manual testing via random walk over
  interaction graph).

### Ablation Studies

- Impact of each mathematical contribution (M1–M4, M7) on detection rate and
  verification time.
- Zone granularity vs. precision/recall tradeoff curves.
- Compositional decomposition quality vs. monolithic verification (speedup and precision).
- Body parameter resolution vs. bug detection sensitivity.
- [A2] Automated-mode vs. annotated-mode state machine extraction: coverage gap analysis
  on real XR scenes.

---

## Feasibility Gates and Kill-Chain

Four-stage kill-chain with explicit deadlines and gate criteria. Failure at any gate
triggers the specified pivot or project termination. These gates are binding commitments,
not aspirational milestones.

### Stage 1 — Zone Abstraction Feasibility (Month 2)

[A1] **Objective:** Prototype the Semialgebraic Zone Abstraction Engine on SE(3) for a
single 4-joint interaction element (shoulder, elbow, wrist, finger).

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Pass** | CAD produces ≤10⁵ cells AND completes in ≤10 min on laptop CPU | Proceed to Stage 2 |
| **Soft fail** | CAD exceeds thresholds but interval-based over-approximation succeeds within 30 min | Pivot to interval-based scheme; weaken M2 accordingly |
| **Hard fail** | Interval-based fallback also fails to produce meaningful results within 30 min | Project terminated |

**Rationale:** The entire verification approach rests on the tractability of zone
abstraction. If a single 4-joint element is intractable, scenes with dozens of elements
are hopeless. This gate must come early enough to avoid sunk cost.

### Stage 2 — Demand Validation (Month 3)

[A7] **Objective:** Structured interviews/surveys with ≥20 XR developers (mix of indie,
enterprise, platform).

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Pass** | ≥25% express concrete interest in adopting tool | Proceed to Stage 3 |
| **Soft fail** | 10–25% express interest | Narrow scope (e.g., reach-envelope visualization only) or pivot target audience |
| **Hard fail** | <10% express any interest | Project terminated |

**Rationale:** Technical feasibility without demand produces shelf-ware. The 25% threshold
is intentionally low—early-stage developer interest is noisy, and this is a novel tool
category. But <10% signals a fundamental value-proposition problem.

### Stage 3 — End-to-End Pipeline on Real Scene (Month 5)

**Objective:** Tier 1 + Tier 2 pipeline runs end-to-end on at least one real XR scene
from the A4 benchmark corpus.

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Pass** | Produces ≥1 true positive finding in <10 min | Proceed to Stage 4 |
| **Soft fail** | Pipeline completes but produces only false positives | Return to Zone Abstraction Engine for precision improvements |
| **Hard fail** | Pipeline cannot complete on any real scene within 1 hour | Reassess architectural approach |

**Rationale:** Procedural scenes are designed to be tractable. Real scenes exercise
parsing robustness, heuristic extraction coverage, and geometric complexity that
procedural scenes cannot replicate. This gate validates the full stack, not just the
mathematical core.

### Stage 4 — Monte Carlo Marginal Value (Month 8)

[A3] **Objective:** Run formal verification and Monte Carlo baseline (1M × 1K stratified
samples) on full benchmark suite. Measure marginal detection rate.

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Pass** | Formal verification detects ≥10% of bugs missed by MC | Full paper submission proceeds |
| **Soft fail** | Marginal detection rate 5–10% | Reframe as "certification-grade guarantees"; target Tier 3 (enterprise compliance) only |
| **Hard fail** | Marginal detection rate <5% | Formal verification does not justify cost; pivot to publishing MC baseline tool as primary contribution |

**Rationale:** Formal verification is expensive to build and expensive to run. If Monte
Carlo sampling with sufficient budget catches nearly everything formal verification
catches, the marginal value of formal guarantees is too small to justify the engineering
cost. The 10% threshold ensures formal verification provides meaningful value beyond what
cheaper methods achieve.

---

## Laptop CPU Feasibility

**Why no GPU is needed.** The computation is fundamentally symbolic and algebraic, not
numerical. Zone abstraction is symbolic geometry (CAD), model checking is BDD-based graph
exploration, and constraint solving uses branch-and-bound. None of these benefit
significantly from GPU parallelism—they exhibit irregular control flow, pointer-chasing
data structures, and high branch divergence.

**Computational strategy by persona:**

- **Tier 1 — "XR Accessibility Linter"** (<5 seconds): Interval arithmetic on bounding
  volumes. Linear in scene size. Trivially CPU-bound. Runs inside Unity editor during
  development.
- **Tier 2 — "CI/CD Accessibility Gate"** (<10 minutes): BDD-based zone-graph
  reachability. Memory-bound, not compute-bound. 16GB RAM sufficient for scenes up to
  ~50 objects. Runs in CI/CD pipeline, blocks deployment on violations.
- **Tier 3 — "Certification Engine"** (hours/overnight): CEGAR loop. CPU-intensive but
  inherently sequential (each refinement depends on the previous counterexample). Benefits
  from multi-core parallelism for independent zone splitting operations. Runs as overnight
  batch job for enterprise compliance.

**Target performance:** Verify a 50-object XR scene across the 5th–95th percentile
body-parameter range × 3 target devices in under 4 hours on an 8-core laptop with 16GB
RAM. This target is contingent on theorems M3a and M3b yielding practical constants and on
real XR scenes exhibiting low interaction-graph treewidth—both hypotheses validated or
falsified by the Stage 1 feasibility gate (Month 2). [A1] If the feasibility gate forces
a pivot to interval-based over-approximation, Tier 2 and Tier 3 timing targets may need
revision.

**Key enabler:** The interaction-locality theorems (M3a/M3b). M3a ensures that real XR
interactions are *k*-local with k ≤ 4—each interaction element involves at most 4 body
joints (e.g., shoulder, elbow, wrist, finger for a button press)—bounding the
continuous-dimension exponential to f(4) regardless of total body complexity. M3b ensures
that scenes with low interaction-graph treewidth (typical of real XR scenes, where
interactions are spatially clustered) avoid discrete-state explosion. Together, k-locality
of interactions AND bounded treewidth of the scene interaction graph make laptop-scale
verification feasible for realistic scenes.

---

## Slug

`xr-affordance-verifier`
