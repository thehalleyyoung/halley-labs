# Architect Framings — xr-event-calculus-compiler

> **Community**: area-060-graphics-simulation-and-mixed-reality  
> **Venues**: SIGGRAPH, IEEE VR, ISMAR, UIST, CHI  
> **Hard constraints**: CPU-only · no human annotation/studies · fully automated evaluation  

---

## Framing 1 — Testing / Verification Angle

### Title

**Choreo-Check: Headless Formal Verification of Spatial-Temporal Interaction Protocols in Mixed-Reality Scenes**

### Problem and Approach

Mixed-reality applications encode complex multi-party interaction protocols—a user reaches toward a holographic panel, a virtual agent responds with a gesture sequence, a spatial anchor constrains a shared annotation to a tabletop region, and a timeout resets the scene if no gaze fixation is detected within two seconds. Today, testing these choreographies requires running the full engine loop (Unity, Unreal, or a WebXR runtime), strapping on a headset, and manually exercising every path. When an interaction deadlocks—say, two grab affordances compete for the same hand—the developer discovers it only in a live demo. There is no `pytest` for XR interaction logic. CI pipelines cannot cover spatial-temporal behavior, so regressions ship silently.

Choreo-Check introduces a formally grounded headless testing framework purpose-built for XR interaction choreographies. The core insight is that the *interaction-relevant* subset of a mixed-reality scene can be abstracted into a network of **spatial-temporal event automata**: finite-state machines whose transitions are guarded by spatial predicates (containment, proximity, gaze-cone intersection) evaluated over an R-tree index, and whose timing constraints are expressed in a fragment of Metric Temporal Logic (MTL). The developer writes interaction protocols in a declarative DSL—`when hand enters zone "panel" within 500ms of gaze hitting zone "panel" → activate "menu"`—and Choreo-Check compiles these declarations into a product automaton. A CPU-side model checker then exhaustively explores the reachable state space, reporting deadlocks (states from which no accepting configuration is reachable), race conditions (nondeterministic transitions on simultaneous spatial events), and liveness violations (an expected feedback event is never produced). For scenes too large for exhaustive exploration, a bounded symbolic checker using BDD-encoded spatial intervals provides sound over-approximation.

Evaluation is fully automated: we harvest open-source XR interaction graphs from the Unity Asset Store, Microsoft MRTK sample scenes, and WebXR A-Frame examples, automatically extract their implicit interaction protocols, compile them, and run the checker. Metrics include bugs found (deadlocks, unreachable states), state-space size versus checking time, and coverage comparison against random simulation. No headsets, no humans—just a laptop CPU and a CI badge.

### Who Desperately Needs This and Why

- **XR platform teams at Meta, Apple, Microsoft, and Qualcomm** ship interaction toolkits (MRTK, Interaction SDK, visionOS gestures) with hundreds of cross-cutting interaction rules. Every release risks regressions that surface only during QA playtest weeks. A headless checker that runs in CI would cut weeks from their release cycle.
- **Accessibility engineers** need to verify that alternative input modalities (eye-tracking, switch scanning, voice) reach the same set of scene states as hand tracking. This is a reachability question—exactly what a model checker answers.
- **Multi-user XR developers** (Spatial, Microsoft Mesh) face combinatorial explosion of interaction interleavings between co-located users. Manual testing cannot cover these; formal methods can.
- **Regulatory bodies** evaluating XR medical training and industrial-safety applications need evidence that critical UI paths cannot deadlock. A verification certificate is far more compelling than a test log.

### Why It's Genuinely Hard

1. **Spatial predicate abstraction.** Mapping continuous 3D geometry (meshes, colliders, ray casts) into discrete spatial predicates that are sound for verification requires a carefully designed abstract domain. Over-approximate too aggressively and every scene reports false deadlocks; under-approximate and real bugs slip through. The R-tree must support *interval-valued* bounding boxes, not just point queries, to represent sets of possible hand positions.

2. **Metric temporal logic model checking on CPU.** MTL model checking is PSPACE-complete in the general case. The system needs a practical fragment (bounded-horizon, bounded-clock-count) plus aggressive partial-order reduction to handle scenes with 50+ concurrent interaction zones at interactive checking speeds. Building an efficient zone-graph or region-automaton construction that exploits the spatial structure (most zones are far apart and independent) is a research problem in its own right.

3. **Automatic protocol extraction from existing engines.** Unity, Unreal, and WebXR encode interaction logic in radically different ways—C# MonoBehaviour event callbacks, Blueprints, A-Frame component wiring. Building robust extractors that faithfully capture the interaction-relevant fragment without importing the entire engine semantics is a systems engineering marathon (easily 30K+ LoC per engine).

4. **Product automaton state-space explosion.** A scene with N interaction zones, M input modalities, and K users has O((S^N)^(M·K)) reachable states in the worst case, where S is the per-zone state count. Practical reduction via spatial independence partitioning (zones that cannot physically overlap can be checked compositionally) requires integrating the R-tree geometry index directly into the state-space exploration loop—a novel combination of computational geometry and model checking.

5. **Counterexample visualization without a headset.** When the checker finds a deadlock, it must produce a human-readable *spatial trace*—a sequence of 3D snapshots showing hand/gaze positions that trigger the bug. Generating a minimal reproducing trace from a BDD-encoded counterexample, and rendering it to a 2D timeline diagram (no GPU), is a non-trivial UX and algorithmic problem.

6. **Incremental re-checking.** Developers change one interaction rule and expect instant feedback. The system must support incremental product-automaton updates—splicing in a modified sub-automaton without rebuilding the entire state space—which requires a dependency-tracking layer over the zone graph.

### Best-Paper Argument

No formal verification tool exists for XR interaction logic—period. The XR community currently has zero CI-integrated correctness guarantees for spatial-temporal behavior. Choreo-Check would be the first system to bring model-checking methodology, proven successful in hardware verification and protocol analysis, into the mixed-reality interaction domain. This parallels the impact of tools like SPIN and TLA+ in distributed systems: once developers can *check* interaction protocols instead of *hoping* they work, the entire development methodology shifts. A UIST or CHI best paper that demonstrates real deadlock bugs found automatically in Microsoft MRTK and Meta Interaction SDK samples—bugs that shipped in production—would be a field-defining result. The formal methods + spatial computing intersection is completely unoccupied.

### Fatal Flaws or Weaknesses

- **Abstraction fidelity gap.** The discrete automaton model necessarily discards physics, animation blending, and rendering-order effects that can influence real interaction outcomes. Reviewers may argue that the abstraction misses the bugs that matter most.
- **Adoption barrier.** Developers must either write protocols in the DSL or trust automatic extraction. If extraction is lossy, the tool checks a *model* of the interaction, not the *actual* interaction—a classic model-checking critique.
- **Scalability ceiling.** Even with spatial partitioning, complex multi-user scenes may blow up the state space. If the practical limit is 10-15 interaction zones, reviewers will question real-world applicability.
- **Evaluation without human studies is limiting.** The tool's ultimate value is developer productivity, but measuring that requires a user study—which is forbidden. Bug-count and state-space metrics are proxies at best.

---

## Framing 2 — Compiler / DSL Angle

### Title

**Choreo-Lang: A Spatial-Temporal Choreography Language That Compiles XR Interaction Patterns to Portable Event Automata**

### Problem and Approach

Programming mixed-reality interactions today is like writing distributed systems in assembly language. Every XR framework provides low-level event callbacks—`OnTriggerEnter`, `OnGazeHit`, `OnHandPinch`—and the developer manually weaves these into state machines using ad-hoc boolean flags, coroutines, and timers scattered across dozens of scripts. The resulting code is (a) platform-locked (a Unity interaction cannot run in WebXR), (b) untestable without the full engine runtime, (c) unanalyzable (no tool can determine whether two interactions conflict), and (d) unreviewable (the interaction *design* is buried in imperative spaghetti). The field has spatial scene-graph languages (USD, glTF) and shading languages (GLSL, HLSL, Slang), but no *interaction* language. This is the gap.

Choreo-Lang is a new domain-specific language for declaring spatial-temporal interaction choreographies, together with a compiler that lowers those declarations into a portable intermediate representation—a network of **R-tree-backed event automata (REA)**—which can be interpreted by a thin runtime on any platform (Unity C#, Unreal C++, WebXR JavaScript, native visionOS Swift). The language captures the *what* and *when* of interaction, not the *how*: `region panel = box(table, 0.3m)` defines a spatial zone; `interaction menu_open = gaze(panel) & reach(panel, <500ms) → activate(menu)` declares a pattern; `conflict menu_open, object_grab priority menu_open` resolves ambiguity. The compiler performs spatial type-checking (are referenced zones geometrically consistent?), temporal well-formedness checking (are timeout constraints satisfiable?), determinism analysis (can two patterns fire simultaneously on the same input?), and dead-pattern elimination (does a pattern's spatial guard overlap with any reachable configuration?). The output REA graph is a self-contained artifact—a JSON/binary bundle of automaton tables plus R-tree node descriptors—that any compliant runtime can execute without reimplementation.

The compiler is structured as a multi-phase pipeline: parsing → spatial type elaboration (resolving zone references against a scene graph, computing bounding hierarchies) → temporal constraint compilation (translating MTL-fragment guards into clock automata) → product construction (composing per-interaction automata into a global choreography graph with conflict resolution) → optimization (merging equivalent states, hoisting shared spatial queries into the R-tree bulk-load phase) → code generation (emitting platform-specific runtime bindings or the portable REA bundle). Each phase is independently testable. Evaluation uses a benchmark suite of 200+ interaction patterns extracted from published XR research prototypes and commercial samples, measuring compilation throughput, REA graph size, cross-platform execution fidelity (does the same REA produce identical event traces on Unity vs. WebXR?), and performance overhead versus hand-coded interactions.

### Who Desperately Needs This and Why

- **XR interaction designers** who prototype in one framework and must rewrite from scratch when switching platforms. A portable interaction language eliminates this rewrite tax, which currently costs weeks per interaction set.
- **Standards bodies (OpenXR, WebXR, Khronos)** working toward XR interoperability have no way to specify interaction *behavior* portably. USD describes scenes, glTF describes assets, but nothing describes how a user interacts with them. Choreo-Lang fills a critical gap in the XR content pipeline.
- **XR middleware vendors (Ultraleap, Tobii, ManusVR)** ship hand-tracking and eye-tracking SDKs with per-engine integration code. A single interaction language would let them ship platform-neutral interaction definitions, dramatically reducing their integration matrix.
- **Research labs** publishing interaction techniques at CHI and UIST currently share videos and pseudocode. A Choreo-Lang program would be a reproducible, executable artifact—transforming how interaction research is disseminated.

### Why It's Genuinely Hard

1. **Spatial type system design.** The language must type-check spatial relationships: is `zone A` always inside `zone B`? Can `zone C` and `zone D` overlap given the scene graph constraints? This requires a type system that embeds geometric reasoning—effectively, a decidable fragment of the theory of bounded real arithmetic operating over scene-graph-relative coordinate frames. Designing this so it is expressive enough to capture real interactions yet decidable enough for fast checking is a deep PL + computational geometry problem.

2. **Temporal constraint compilation.** Translating user-written temporal patterns (`within 500ms`, `after gaze dwells for 1s`, `before hand exits zone`) into clock automata is well-studied for simple cases but becomes combinatorially complex when multiple overlapping temporal constraints interact. The compiler must detect unsatisfiable constraint combinations at compile time (e.g., "within 200ms of event A AND after 500ms of event B, where A causes B" is vacuously false).

3. **Cross-platform semantic equivalence.** Each target runtime has different event-delivery semantics: Unity's physics triggers fire on FixedUpdate, WebXR's `select` events are frame-aligned, visionOS gesture events are asynchronous. The runtime layer must reconcile these timing models so that the *observable interaction behavior* is identical up to a bounded timing skew. Defining this equivalence formally and testing it automatically is a research contribution in itself.

4. **Conflict resolution and determinism.** When two interaction patterns can fire on the same spatial-temporal event, the language must provide a principled resolution mechanism (priority, mutual exclusion, merge). This is analogous to ambiguity resolution in parser generators but in a spatial-temporal domain where "ambiguity" means two overlapping 3D regions with simultaneous temporal guards—requiring geometric intersection tests at compile time.

5. **R-tree-aware optimization.** The compiler must lay out the REA graph's spatial queries so they are bulk-loadable into an R-tree with minimal query overlap, minimizing runtime spatial-query cost. This is an NP-hard bin-packing variant; the compiler needs heuristics that produce near-optimal R-tree layouts for the common case of axis-aligned interaction zones on planar surfaces (tables, walls, floors).

6. **Incremental compilation.** Interaction designers iterate rapidly. When one pattern changes, the compiler must recompile only the affected sub-automaton and re-check only the affected conflict pairs, not rebuild the entire product graph. This requires a fine-grained dependency graph over spatial zones, temporal constraints, and automaton states—analogous to incremental type-checking in language servers but with spatial and temporal dimensions.

7. **Extensibility for novel input modalities.** New input devices (EMG armbands, neural interfaces, full-body tracking) arrive yearly. The language must support user-defined spatial predicates and event types without modifying the compiler core—requiring a principled extension mechanism (type classes, traits, or a plugin protocol) that maintains all compiler guarantees.

### Best-Paper Argument

The XR field has standardized *content* (glTF, USD), *rendering* (Vulkan, WebGPU), and *tracking* (OpenXR), but has conspicuously failed to standardize *interaction*. Every new XR framework reinvents hand menus, gaze-dwell buttons, and grab mechanics from scratch. Choreo-Lang would be the first language to give interaction choreographies a formal syntax, static semantics, and a portable compilation target. This is a "new programming model" paper in the tradition of Cg (GPU shading), Halide (image processing), and TVM (ML compilation)—each of which won best-paper awards by revealing that an entire domain had been programming at the wrong abstraction level. The compiler artifact itself (parser, spatial type checker, temporal compiler, cross-platform runtime, R-tree optimizer) is a 150K+ LoC system that demonstrates genuine software-engineering ambition. A SIGGRAPH or UIST best paper that introduces a new language for XR interaction—accompanied by a working compiler and cross-platform runtime—would reshape how the community thinks about interaction portability.

### Fatal Flaws or Weaknesses

- **Language adoption risk.** New DSLs face a steep adoption curve. If the language is too restrictive, developers abandon it; if too expressive, it loses analyzability. Finding the sweet spot requires iteration with real users—which we cannot do (no human studies).
- **Expressiveness ceiling.** Many real XR interactions involve physics simulation (throwing, stacking), animation state machines, and AI agent behavior. If Choreo-Lang cannot express these, reviewers will argue it only covers "the easy part" of XR programming.
- **Comparison baseline ambiguity.** There is no prior XR interaction language to compare against. Reviewers may demand comparison with general-purpose visual scripting (Unity Visual Scripting, Unreal Blueprints), which are more expressive even if less analyzable—an apples-to-oranges comparison that is hard to win.
- **Runtime overhead skepticism.** An interpreted REA graph adds an indirection layer. Reviewers may worry about latency in the interaction loop (XR demands <20ms input-to-photon). Demonstrating negligible overhead requires careful benchmarking on constrained mobile-class CPUs.

---

## Framing 3 — Simulation / Runtime Angle

### Title

**Choreo-Engine: An Incremental Spatial-Temporal Event Engine for Headless Mixed-Reality Scene Simulation at Scale**

### Problem and Approach

Running a mixed-reality scene today requires a game engine. Testing whether a particular interaction sequence leads to a desired outcome requires *running that engine in real time*, with a headset connected, a human performing the actions, and the entire rendering pipeline active—even when rendering is irrelevant to the question being asked. Researchers studying multi-user XR collaboration must physically co-locate participants or set up networked rigs. Developers evaluating whether a redesigned hand menu works across 1,000 usage scenarios must run 1,000 manual sessions. The bottleneck is not computation—it is the coupling of *interaction simulation* to *rendering infrastructure*. If we could simulate the interaction-relevant dynamics of an XR scene without the engine, without the headset, and without the GPU, we could run thousands of scenario variations on a laptop overnight.

Choreo-Engine is a CPU-side incremental execution engine for spatial-temporal XR interaction logic. It takes as input (a) a scene graph with spatial zones and object affordances, (b) a set of interaction rules expressed as event automata (compiled from a DSL or extracted from engine projects), and (c) a scenario script specifying a sequence of synthetic input events (hand positions over time, gaze directions, voice commands). It produces as output a complete event trace: which automaton transitions fired, which scene state changes occurred, and which temporal deadlines were met or violated. The engine's key architectural contribution is an **R-tree-backed incremental spatial event dispatcher**: as synthetic input positions update frame-by-frame, the R-tree index incrementally identifies which spatial zones are entered, exited, or dwelled upon, and dispatches only the *changed* events to the affected automata. This avoids re-evaluating the entire scene graph every frame—critical when simulating scenes with hundreds of interaction zones across thousands of scenario frames.

The engine supports three execution modes: (1) **trace replay**, where recorded 6-DOF hand/head trajectories from OpenXR log files drive the simulation deterministically; (2) **stochastic exploration**, where a Monte Carlo sampler generates random-but-physically-plausible input trajectories (hand positions follow a biomechanical reachability model, gaze follows a saccade-fixation model) and the engine collects statistical coverage over the interaction state space; (3) **guided search**, where a lightweight fitness function (e.g., maximize automaton state coverage, or reach a specific target state) drives an evolutionary or beam-search exploration of input trajectories. Evaluation is fully automated: we build a benchmark of 50 XR scenes (extracted from open-source projects and synthetic generators), measure simulation throughput (scenes × frames per second on a single CPU core), incremental update efficiency (speedup over full re-evaluation), state-space coverage achieved by stochastic exploration versus random replay, and bugs detected (deadlocks, unreachable states, timing violations) across the benchmark suite.

### Who Desperately Needs This and Why

- **XR CI/CD pipelines** at every major XR company currently have a blind spot: they can build the app, run unit tests on non-spatial logic, and perhaps verify rendering via screenshot comparison—but they cannot test *interaction behavior* without a human in the loop. Choreo-Engine plugs directly into CI as a headless interaction simulator, enabling automated regression testing of spatial-temporal behavior.
- **XR interaction researchers** who publish new techniques at CHI, UIST, and IEEE VR need to evaluate robustness across diverse usage patterns. Currently, they recruit 12-20 participants for a user study. With Choreo-Engine, they can simulate 10,000 synthetic users with biomechanically-sampled hand trajectories and report statistical coverage metrics—not as a replacement for human studies, but as a complement that reveals edge cases no 20-person study would find.
- **XR accessibility auditors** need to verify that all interaction paths are reachable via alternative input modalities. Choreo-Engine can simulate switch-scanning, eye-tracking-only, and voice-only input streams and verify that every scene state remains reachable—an exhaustive check impossible with manual testing.
- **Digital twin / industrial XR teams** building training simulations for factory floors and surgical procedures need to stress-test interaction sequences under timing pressure (e.g., "if the operator doesn't press the emergency stop within 2 seconds of the alarm, does the simulation branch correctly?"). Choreo-Engine can sweep over timing parameters and report boundary cases.

### Why It's Genuinely Hard

1. **Incremental R-tree maintenance under streaming spatial updates.** Classic R-tree implementations are optimized for bulk-load-then-query. Choreo-Engine must support high-frequency incremental updates (hand position changes at 90 Hz per simulated user) while maintaining query performance for zone-entry/exit detection. This requires a carefully tuned incremental R-tree variant (e.g., R*-tree with deferred reinsertion, or a bespoke temporal-coherence-exploiting structure that amortizes rebalancing over multiple frames). Getting this to sustain >10K simulated frames/second on a single CPU core is a systems engineering challenge.

2. **Biomechanically plausible input synthesis.** Generating synthetic hand trajectories that are physically plausible (respecting joint limits, reach envelopes, movement speed distributions) and behaviorally plausible (following Fitts's Law for targeting, realistic gaze-hand coordination delays) requires a generative model grounded in motor control literature. Building this model without training on human data (no annotation constraint) means deriving it from published biomechanical parameters and kinematic equations—a substantial modeling effort.

3. **Temporal event ordering under simulation.** In a real XR runtime, events are delivered in frame order with real-time timestamps. In headless simulation, the engine must maintain a consistent temporal model: if two spatial events are generated in the same simulated frame, their relative ordering must be deterministic and consistent with the real runtime's event-delivery semantics. Defining and implementing a portable temporal execution model that is faithful across Unity, Unreal, and WebXR frame models is non-trivial.

4. **Guided search over continuous input spaces.** The input space for an XR interaction is continuous and high-dimensional (6-DOF hand pose × 2 hands × head pose × gaze direction × time). Guided search toward a target interaction state (e.g., "find an input sequence that triggers the deadlock") is a planning problem in a hybrid discrete-continuous space. The engine needs a search strategy that efficiently navigates this space—likely combining discrete automaton-state heuristics with continuous trajectory optimization. This is closely related to hybrid systems reachability, a known hard problem.

5. **Scaling to multi-user scenes.** Simulating K concurrent users multiplies the input dimensionality by K and introduces inter-user spatial interactions (occlusion, shared-zone contention). The R-tree dispatcher must efficiently handle cross-user spatial events, and the automaton product must model synchronization points between users. Keeping this tractable for K=4-8 users (a realistic collaboration scenario) is a major scalability challenge.

6. **Scene extraction and abstraction.** To run headless, the engine must ingest scene descriptions from diverse sources: Unity scene files (.unity YAML), Unreal level assets (.umap), glTF with extensions, and USD. Each format encodes colliders, trigger volumes, and interaction components differently. Building robust importers that extract the interaction-relevant scene subset—spatial zones, affordances, and wiring—is 20-30K LoC of tedious but essential format-wrangling code.

7. **Deterministic replay fidelity.** When replaying recorded OpenXR input logs, the engine must produce *exactly* the same event trace as the original runtime session. Achieving this requires precisely matching the original runtime's spatial-query thresholds, event-debouncing logic, and frame-timing model—details that are often undocumented and discovered only through painstaking reverse-engineering.

### Best-Paper Argument

The XR community has invested enormous effort in rendering engines, tracking systems, and content pipelines—but has no *interaction simulation* infrastructure. Choreo-Engine would be the first system that decouples XR interaction evaluation from rendering hardware, enabling a "headless XR testing" paradigm analogous to headless browser testing (Puppeteer, Playwright) that revolutionized web development. The technical contributions—incremental R-tree event dispatch, biomechanical input synthesis, guided hybrid-space search—each stand alone as solid systems contributions; combined, they constitute a platform. A SIGGRAPH or IEEE VR best paper that demonstrates finding real interaction bugs in production XR applications, at 10,000× the throughput of manual testing, on a laptop CPU, would have enormous impact. The "headless testing" framing is immediately legible to every XR developer, making the value proposition self-evident. Moreover, the artifact is a substantial runtime system (150K+ LoC) that the community can build upon—not a one-off research prototype.

### Fatal Flaws or Weaknesses

- **Simulation fidelity gap.** By stripping out physics and rendering, the engine may miss interaction bugs that arise from physics-dependent behavior (e.g., a thrown object's trajectory determining which zone it lands in) or rendering-dependent behavior (e.g., occlusion affecting what the user can see and therefore interact with). Reviewers will ask: "how much of real interaction behavior do you actually capture?"
- **Input synthesis validity.** Without validating synthetic trajectories against real human data (no human studies), the claim that stochastic exploration finds "realistic" bugs is weakened. The biomechanical model may be plausible yet unrepresentative.
- **Performance bar is high.** Claiming "thousands of scenarios overnight" requires sustained simulation throughput of >10K frames/second. If the incremental R-tree or automaton engine has even modest constant-factor overhead, throughput drops and the value proposition weakens. CPU-only operation compounds this—there is no GPU fallback.
- **Engine extraction brittleness.** Unity and Unreal scene formats are complex and version-dependent. Extractors may break with each engine update, creating a maintenance burden that undermines the tool's reliability.
- **Overlap with existing game-testing tools.** Tools like Unity Test Framework and Unreal Automation Framework support limited headless testing. Reviewers may argue that Choreo-Engine's advantage over extending these existing tools is incremental rather than fundamental—though these tools lack spatial-temporal reasoning entirely.

---

## Comparative Summary

| Dimension | Framing 1: Choreo-Check (Verification) | Framing 2: Choreo-Lang (Compiler/DSL) | Framing 3: Choreo-Engine (Simulation/Runtime) |
|---|---|---|---|
| **Primary contribution** | First formal verification tool for XR interaction protocols | First portable interaction language and compiler for XR | First headless interaction simulation engine for XR |
| **Core technical novelty** | Spatial-temporal model checking with R-tree-backed abstract domains | Spatial type system + temporal constraint compiler + cross-platform codegen | Incremental R-tree event dispatch + biomechanical input synthesis + guided search |
| **Value pitch** | "Find deadlocks in your XR interactions before users do" | "Write interactions once, run on every XR platform" | "Test 10,000 interaction scenarios overnight on a laptop" |
| **Extreme value** | ★★★★★ (zero existing tools in this space) | ★★★★☆ (clear need, but adoption risk is higher) | ★★★★★ (every XR team wants CI for interactions) |
| **Genuine difficulty** | ★★★★★ (PSPACE model checking + spatial abstraction) | ★★★★★ (full compiler pipeline + spatial type theory) | ★★★★☆ (hard systems engineering, less theoretical depth) |
| **Best-paper potential** | ★★★★★ at UIST/CHI (formal methods × HCI is hot) | ★★★★★ at SIGGRAPH/UIST (new language = field-defining) | ★★★★☆ at IEEE VR/ISMAR (strong systems paper, less novelty) |
| **Biggest risk** | Abstraction too lossy to find real bugs | Language never adopted outside the paper | Simulation too unfaithful to replace real testing |
| **LoC complexity** | ~120-160K (extractors + checker + counterexample viz) | ~150-180K (parser + type checker + compiler + 4 runtimes) | ~140-170K (engine + importers + input synthesis + search) |
| **Best venue** | UIST 2026 or CHI 2026 | SIGGRAPH 2026 or UIST 2026 | IEEE VR 2026 or ISMAR 2026 |

### Recommendation for Next Phase

All three framings address a genuine gap (no formal interaction tooling for XR), but they optimize for different audiences. **Framing 2 (Choreo-Lang)** has the highest ceiling because a new *language* reshapes thinking more durably than a tool or engine—but carries the highest adoption risk. **Framing 1 (Choreo-Check)** is the safest bet for a best-paper because the "bugs found" metric is unambiguous and the formal-methods angle is differentiated. **Framing 3 (Choreo-Engine)** is the most immediately useful to industry but faces the toughest "so what's new?" scrutiny. A strong strategy is to build the compiler core (Framing 2) as the foundation, then deliver verification (Framing 1) and simulation (Framing 3) as applications of the compiled representation—yielding a three-paper arc from a single codebase.
