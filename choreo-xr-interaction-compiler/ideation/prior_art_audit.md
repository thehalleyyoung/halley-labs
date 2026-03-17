# Prior Art Audit: XR Event Calculus Compiler

**Seed Idea**: A domain-specific language compiler and incremental runtime that transforms declarative spatial-temporal interaction patterns for mixed-reality scenes into R-tree-backed event automata, enabling headless cross-platform choreography testing with formal reachability and deadlock verification—all on CPU.

**Date**: 2025-07-18
**Auditor Role**: Prior Art Auditor (crystallization team)

---

## Executive Summary

The seed idea combines six distinct technical threads: (1) a spatial-temporal DSL, (2) Event Calculus semantics, (3) R-tree-backed spatial indexing for interaction dispatch, (4) compilation to event automata, (5) headless XR testing, and (6) formal reachability/deadlock verification. Each thread has significant prior art individually. The **genuinely novel contribution** lies in their *specific integration*: no existing system compiles a declarative spatial-temporal interaction DSL grounded in Event Calculus into R-tree-backed automata with built-in formal verification, targeting headless XR choreography testing on CPU. However, several sub-combinations are closer to existing work than they may initially appear, and the novelty is best characterized as **architectural synthesis** rather than fundamental breakthrough in any single dimension.

---

## 1. XR/VR Testing Frameworks

### Key Prior Art

| Tool / Project | Description | Limitations |
|---|---|---|
| **Unity Test Framework (UTF)** | Unity-integrated unit/integration tests (PlayMode, EditMode). CI-compatible. | Limited spatial/gestural input realism; tests game logic, not interaction choreography. |
| **XR Interaction Toolkit Simulator** | Editor-based simulation of XR interactions without hardware. | Still requires Unity Editor; no formal verification; platform-coupled. |
| **GameDriver** | Commercial automation for Unity/Unreal XR apps. Device and input simulation for CI/CD. | Proprietary; no formal specification of interaction patterns. |
| **Arium XR (Thoughtworks)** | Open-source, XR-focused test automation for Unity. Platform-agnostic integration with Unity's event system. | Still evolving; procedural test scripts, not declarative patterns. |
| **INTENXION** (academic) | Research framework for automated validation of XR user interactions. Taxonomy of interaction types. | Academic prototype; not formally verified; no published runtime. |
| **iv4XR** (EU Horizon 2020) | Agent-based intelligent V&V for XR. Formal assertions. Agents explore virtual worlds autonomously. | Agents are exploratory (not choreography-centric); no spatial-temporal DSL; verification is assertion-based, not model-checking. |
| **xr.sdk.functionaltests** (Unity) | Example project for multi-platform XR automated tests. | Maintenance-heavy for platform diversity; imperative test scripts. |

### Gap Analysis

All existing XR testing tools are **imperative and procedural**: you write test scripts that drive simulated inputs and assert outcomes. None offer:
- A **declarative language** for specifying spatial-temporal interaction patterns
- **Formal verification** (reachability, deadlock) of interaction specifications
- Truly **headless execution** decoupled from any rendering engine
- **Cross-platform choreography testing** as a first-class concern

**iv4XR** is the closest competitor. It uses agent-based exploration with formal assertions, but its agents discover states empirically rather than exhaustively verifying a declared specification. The seed idea's approach of compiling declared patterns to verifiable automata is categorically different from iv4XR's agent-based exploration.

**Verdict**: The gap is **genuine and significant**. No existing XR testing tool combines declarative pattern specification with formal verification.

---

## 2. Spatial DSLs and Languages

### Key Prior Art

| Language / Framework | Description | Limitations |
|---|---|---|
| **Apple RealityKit / SwiftUI** | Declarative entity-component system for visionOS. Scene graphs, spatial layouts, gesture handlers. | Apple-only; describes scenes, not interaction *patterns* or *choreographies*; no formal semantics. |
| **Microsoft MRTK / OpenXR** | Cross-platform toolkit with scene understanding, spatial anchors, interaction abstractions. | Imperative C# API, not a DSL; no formal verification. |
| **SpatiaLang** (academic) | DSL for AR/VR spatial interactions. Compiles to Unity/ARCore code. Declarative gestures, proximity triggers. | **Closest prior art for the DSL aspect.** However: no Event Calculus semantics, no R-tree backing, no formal verification, no headless execution. |
| **Scenic** (UC Berkeley) | Probabilistic scenario description language for cyber-physical systems. Compiles to simulator code. Supports spatio-temporal constraints. | Targets autonomous vehicles/robots, not XR interaction choreography. No Event Calculus; no R-tree; no interaction automata. Focuses on *scene generation*, not *interaction verification*. |
| **USD (Universal Scene Description)** | Pixar/Apple standard for scene interchange. Declarative scene composition. | Describes static/animated scenes, not interaction logic. |
| **A-Frame / WebXR** | Declarative HTML-like markup for VR scenes. Entity-component-system. | Describes scene structure, not temporal interaction patterns. |
| **Meta Spatial SDK** | Meta's declarative spatial API for Quest. | Platform-locked; no formal semantics. |

### Gap Analysis

**SpatiaLang** is the most relevant prior art. It shares the goal of a high-level declarative language for spatial interactions compiled to engine code. However, it differs from the seed idea in critical ways:
- SpatiaLang has no temporal reasoning (Event Calculus or otherwise)
- No formal verification of specified patterns
- Generates imperative code, not verifiable automata
- No headless testing capability

**Scenic** is relevant as a precedent for a DSL compiler targeting formal verification of spatial scenarios, but it operates in a fundamentally different domain (autonomous driving scene generation) and does not model interactive multi-party choreographies.

**Verdict**: The DSL component is **incrementally novel**. SpatiaLang demonstrates the concept of a spatial interaction DSL compiled to engine code. The seed idea's novelty is in the *semantics* (Event Calculus), the *target* (verifiable automata rather than imperative code), and the *purpose* (headless choreography testing with formal guarantees).

---

## 3. Event Calculus in Software

### Key Prior Art

| Work | Description | Limitations |
|---|---|---|
| **Kowalski & Sergot (1986)** | Original Event Calculus formalism. Logic-based reasoning about events, fluents, and temporal change. | Foundational theory; not directly applied to spatial or XR domains. |
| **Reactive Event Calculus (REC)** | Extension for monitoring distributed agent systems. Runtime event processing. | Targets distributed monitoring, not spatial interaction patterns. |
| **Event Calculus + Prolog implementations** | Various Prolog libraries for EC reasoning. | Performance-limited; no spatial indexing; no compilation to automata. |
| **EC Formalization of Timed Automata** (Artikis et al., CEUR-WS) | Translates timed automata representations into Event Calculus predicates for reasoning in Prolog. | Goes the *opposite direction* from the seed idea (automata→EC, not EC→automata). Academic; no spatial dimension. |
| **RTEC (Run-Time Event Calculus)** | Incremental Event Calculus engine for complex event recognition. Used in maritime surveillance, city transport. | **Close to the incremental runtime aspect.** No spatial indexing (R-tree); targets stream processing, not XR; no formal verification. |
| **Reasoning about Space, Actions and Change** (Bhatt et al., 2010) | Qualitative spatial and temporal reasoning integrating spatial logic with event/situation calculus. | Theoretical framework; no compiler, no runtime, no XR application. |

### Gap Analysis

Event Calculus has been applied to monitoring, activity recognition, and temporal reasoning, but **never to XR/spatial-computing interaction choreography**. The key missing links:
- No existing work compiles EC specifications **into** automata (the reverse direction exists)
- No existing work couples EC with spatial indexing (R-trees) for efficient interaction dispatch
- RTEC is conceptually adjacent for its incremental runtime, but lacks spatial awareness and formal verification

**Verdict**: Applying Event Calculus to XR interaction patterns and compiling *from* EC to automata (rather than the reverse) is **genuinely novel**. The integration with spatial indexing is unprecedented.

---

## 4. R-trees and Spatial Indexing for Interaction

### Key Prior Art

| Application | Description | Limitations |
|---|---|---|
| **GIS/Database spatial queries** | R-trees for range, containment, nearest-neighbor queries. Standard in PostGIS, SQLite, etc. | Purely geometric queries; no event or interaction semantics. |
| **Moving object tracking** | 3D R-trees (space+time) for trajectory intersection detection. | Detects spatial events but has no formal interaction semantics or verification. |
| **Geofencing / spatial triggers** | R-trees for enter/exit detection in IoT, logistics, location-based gaming. | Trigger-based but not pattern-based; no automata; ad-hoc implementations. |
| **Game engines (collision detection)** | Bounding volume hierarchies (related to R-trees) for physics and collision. | Built into engines; no formal semantics; not exposed as programmable interaction dispatch. |
| **GeoT-Rex** | Geospatial complex event processing for smart cities. Spatial extensions to CEP. | Targets IoT streams, not XR interaction choreography. No formal verification. |

### Gap Analysis

R-trees are universally used for spatial queries, but their use as the **backing data structure for event automata transitions** in an interaction system is novel. Existing systems use R-trees to answer "what is near X?" but not to drive "what interaction state transitions are spatially enabled?"

**Verdict**: Using R-trees as the spatial index *behind* event automata state transitions is a **novel architectural choice**. The individual components (R-trees, automata) are well-understood, but their integration for interaction dispatch is new.

---

## 5. Formal Verification of Interactive Systems

### Key Prior Art

| Tool / Approach | Description | Limitations |
|---|---|---|
| **UPPAAL** | Model checker for timed automata. Exhaustive verification of real-time systems (safety, liveness, reachability). TCTL specifications. | Powerful but models must be manually constructed. No spatial dimension. No XR-specific tooling. |
| **SPIN / Promela** | Model checker for concurrent protocols. LTL properties. Deadlock, race condition detection. | Untimed; no spatial reasoning; general-purpose protocol verification. |
| **TLA+ / TLC** | Formal specification and model checking for concurrent/distributed systems. | No spatial or temporal (real-time) primitives. Manual specification. |
| **Alloy** | Lightweight specification analyzer. Structural constraints, counterexample finding. | No temporal logic; not suited for reactive/real-time systems. |
| **XState / Statecharts** | Statechart library for UI state machines. Model-based testing. | No formal verification (exhaustive); no spatial dimension; 2D UI-focused. |
| **Contract Automata / CARE** (Basile et al., 2025) | Formal analysis of contract automata runtime environments with UPPAAL. | Close in spirit—translates interaction contracts to verifiable automata—but for service orchestration, not spatial/XR. |

### Gap Analysis

Formal verification of interactive systems is well-established for:
- Timed systems (UPPAAL)
- Communication protocols (SPIN)
- Concurrent state machines (TLA+)
- UI flows (XState + manual verification)

What is **missing**:
- No tool applies formal verification to **spatial-temporal** interaction patterns
- No tool targets **XR-specific** interaction verification
- No tool combines **spatial indexing** with **automata verification**
- The seed idea's proposition—compiling a DSL to automata that are *both* executable at runtime *and* formally verifiable—is the unique value proposition. UPPAAL models are manually constructed and separate from runtime; the seed idea proposes a single artifact that serves both purposes.

**Verdict**: The application of formal verification to XR spatial-temporal interactions is **novel**. The dual-use artifact (executable + verifiable) is a genuinely new contribution, though the underlying verification techniques are standard.

---

## 6. Headless XR Testing

### Key Prior Art

| Tool / Approach | Description | Limitations |
|---|---|---|
| **OpenXR XR_MND_headless** (Monado) | OpenXR extension for headless sessions. Tracking data without GPU rendering. | Linux/Monado-only; low-level API; no interaction semantics; no test framework. |
| **OpenXR-Simulator** | Desktop VR app emulation without headset. Mouse/keyboard input. | Still renders to a window; not truly headless. |
| **Unity XR Device Simulator** | Editor-based XR simulation. | Requires Unity Editor; minimal GPU still needed; not headless. |
| **Meta XR Simulator** | Quest app testing in Unity Editor without headset. | Platform-specific (Meta); requires Editor rendering. |
| **AR Foundation XR Simulation** | Unity AR logic testing in Editor. | Editor-dependent; not headless. |

### Gap Analysis

"Headless" in existing XR tools means "without a physical headset," **not** "without rendering." True headless testing—no GPU, no rendering pipeline, pure logic execution on CPU—is essentially absent from the XR ecosystem. The only exception is OpenXR's `XR_MND_headless` extension, which is low-level, platform-limited, and provides no test framework.

The seed idea's proposition of **fully headless, CPU-only choreography testing** where the rendering engine is entirely bypassed is a **significant gap in the market**. This is perhaps the most commercially and practically compelling aspect of the proposal.

**Verdict**: True headless XR testing (no rendering, CPU-only, with interaction semantics) is **genuinely novel** and addresses a real pain point in XR CI/CD pipelines.

---

## 7. Choreography / Interaction Pattern Languages

### Key Prior Art

| Formalism / Tool | Description | Limitations |
|---|---|---|
| **BPMN 2.0 Choreography Diagrams** | Visual notation for multi-party message exchange. Semi-formal. | Designed for business processes, not spatial interactions. No spatial or temporal semantics. |
| **WS-CDL** (W3C) | XML-based global protocol description for web service choreographies. | Web services only; no spatial dimension; abandoned W3C spec. |
| **Multiparty Session Types (MPST)** | Type-theoretic framework for multi-party protocol verification. Deadlock freedom, progress guarantees. Scribble language. | **Closest formal precedent for choreography verification.** But: purely message-based (no spatial semantics); targets distributed systems, not XR. |
| **Colored Petri Nets (CPN)** | Mathematically precise modeling of concurrent systems. Used for protocol analysis. | General-purpose; no spatial primitives; manual model construction. |
| **Netcode for GameObjects (Unity)** | Multiplayer networking framework. State sync, RPCs. | Implementation framework, not a specification language. No formal verification. |
| **Photon / Mirror (Unity)** | Multiplayer networking libraries. | Same as above—implementation, not specification. |

### Gap Analysis

**Multiparty Session Types** are the strongest theoretical precedent. MPST provide:
- Global choreography specification projected to local types
- Static verification of deadlock freedom and progress
- Composability guarantees

However, MPST are fundamentally **message-based** (actors exchanging typed messages) and have **no spatial semantics**. The seed idea's choreographies are **spatially grounded**—interaction patterns are triggered and constrained by spatial relationships (proximity, containment, gaze direction, etc.), not just message ordering.

No existing choreography formalism incorporates:
- Spatial predicates as first-class choreography elements
- Temporal constraints tied to physical (3D) space
- R-tree-backed spatial evaluation during choreography execution

**Verdict**: Spatially-grounded choreography specification with formal verification is **genuinely novel**. The extension of choreography concepts from message-based to spatial-temporal interaction patterns has no direct precedent.

---

## Cross-Cutting Prior Art: Complex Event Processing (CEP)

CEP engines (Apache Flink, Esper, GeoT-Rex) deserve mention as a cross-cutting concern:

| Aspect | CEP Systems | Seed Idea |
|---|---|---|
| **Event patterns** | Temporal sequences, windowed aggregation | Spatial-temporal interaction patterns |
| **Spatial support** | Some (GeoT-Rex); geofencing | R-tree-backed, 3D spatial predicates |
| **Formal verification** | None (runtime pattern matching only) | Compile-time reachability and deadlock checking |
| **Target domain** | IoT, finance, logistics | XR/mixed-reality |
| **Execution model** | Stream processing | Event automata with incremental state |

CEP systems share the goal of detecting complex patterns in event streams but lack formal verification, are not designed for interactive (bidirectional) systems, and have no XR-specific affordances.

---

## Novelty Assessment Matrix

| Dimension | Novelty Level | Justification |
|---|---|---|
| Spatial-temporal DSL for XR interactions | **Incremental** | SpatiaLang exists; novelty is in EC semantics and compilation target |
| Event Calculus applied to XR | **Novel** | No prior work applies EC to XR interaction choreography |
| R-tree-backed event automata | **Novel** | No prior work uses R-trees as automata transition enablers |
| Compilation from EC to verifiable automata | **Novel** | Prior work goes the reverse direction (automata→EC) |
| Headless CPU-only XR testing | **Novel** | No existing tool provides true headless XR choreography testing |
| Formal verification of XR interactions | **Novel** | iv4XR uses agents, not model checking; UPPAAL has no spatial/XR support |
| Spatially-grounded choreography language | **Novel** | MPST/BPMN choreographies are message-based, not spatially grounded |
| Incremental runtime for EC | **Incremental** | RTEC provides incremental EC; novelty is in spatial coupling and XR context |

---

## Honest Assessment

### What IS genuinely novel
1. **The integration architecture**: Compiling a spatial-temporal DSL via Event Calculus semantics into R-tree-backed automata that are both executable and formally verifiable. No existing system combines these components.
2. **Spatially-grounded choreography verification**: Extending choreography concepts from message-passing to spatial-temporal interaction with formal guarantees.
3. **True headless XR testing**: CPU-only execution of interaction logic divorced from any rendering pipeline, with formal verification of interaction properties.

### What is NOT novel
1. Each individual component (DSLs, Event Calculus, R-trees, timed automata, model checking, headless testing) is well-established.
2. The idea of compiling a DSL to a formally verifiable intermediate representation is standard in programming languages research.
3. Spatial-temporal event processing exists in CEP systems.

### Risks to Novelty
1. **SpatiaLang** could evolve to add formal verification, narrowing the gap.
2. **iv4XR** follow-on projects could add model checking to their agent-based framework.
3. **UPPAAL** could be extended with spatial primitives (there is active research on spatial model checking).
4. A sufficiently motivated team could combine Scenic + UPPAAL + R-trees to achieve similar goals, though this has not been done.

### Overall Verdict

**The seed idea is novel as an integrated system**, representing a genuine synthesis that no existing work provides. The novelty is best described as **architectural innovation**—a carefully chosen combination of well-understood components applied to an underserved domain (XR interaction verification). This is a legitimate and defensible form of novelty, similar to how React was novel not because of virtual DOMs or component models individually, but because of their specific integration.

The strongest novelty claims are:
1. Event Calculus → automata compilation (reversal of prior direction)
2. R-tree-backed automata transitions (no precedent)
3. Headless spatial choreography verification (no precedent)

The weakest novelty claim is the DSL itself, given SpatiaLang's existence. The seed idea should clearly differentiate its language design from SpatiaLang and emphasize the formal-semantic foundation (Event Calculus) and the verification-first compilation target.

---

## Key References

### XR Testing
- Gu, R. et al. "A Test Automation Framework for User Interaction in Extended Reality" (INTENXION). VaRSE 2024.
- Prasetya, I.S.W.B. et al. "iv4XR: Intelligent Verification/Validation for XR Based Systems." EU Horizon 2020 (2020–2022). https://iv4xr.testar.org
- "Software testing for extended reality applications: a systematic mapping study." Automated Software Engineering (2025). https://link.springer.com/article/10.1007/s10515-025-00523-7
- Arium XR. Thoughtworks. https://www.thoughtworks.com/insights/topic/open-source/arium

### Spatial DSLs
- "SpatiaLang: A DSL for AR/VR Interaction." (academic project, SRM University)
- Fremont, D. et al. "Scenic: A Language for Scenario Specification and Scene Generation." PLDI 2019. https://people.eecs.berkeley.edu/~sseshia/pubdir/pldi19-scenic.pdf
- Apple RealityKit Documentation. https://developer.apple.com/documentation/realitykit

### Event Calculus
- Kowalski, R. & Sergot, M. "A Logic-Based Calculus of Events." New Generation Computing, 1986. https://link.springer.com/article/10.1007/BF03037383
- Artikis, A. et al. "Reactive Event Calculus for Monitoring Global Computing Applications." 2012. https://link.springer.com/chapter/10.1007/978-3-642-29414-3_8
- Artikis, A. et al. "An Event Calculus Formalization of Timed Automata." CEUR-WS Vol-2156. https://ceur-ws.org/Vol-2156/paper5.pdf
- Bhatt, M. et al. "Reasoning about Space, Actions and Change." 2010. https://codesign-lab.org/select-papers/pdfs/Spatial_Reasoning/QSTR-RSAC-Book-Web-2010.pdf

### R-trees
- Guttman, A. "R-trees: A Dynamic Index Structure for Spatial Searching." SIGMOD 1984.
- "Geospatial complex event processing in smart city applications." Simulation Modelling Practice and Theory (2022). https://www.sciencedirect.com/science/article/pii/S1569190X22001459

### Formal Verification
- Behrmann, G. et al. "UPPAAL: Model Checking for Real-Time Systems." https://uppaal.org
- Holzmann, G. "The SPIN Model Checker." Addison-Wesley, 2003.
- Basile, D. et al. "Formal Analysis of the Contract Automata Runtime Environment with UPPAAL." 2025. https://arxiv.org/html/2501.12932
- "An integrated framework of formal methods for interaction behaviors of cyber-physical systems." Microprocessors and Microsystems (2015). https://www.sciencedirect.com/science/article/pii/S0141933115001313

### Headless XR
- OpenXR XR_MND_headless extension. Monado runtime. https://community.khronos.org/t/openxr-without-rendering-loop/108658

### Choreography Languages
- Honda, K. et al. "Multiparty Asynchronous Session Types." POPL 2008.
- Hu, R. & Yoshida, N. "Programming Language Implementations with Multiparty Session Types." https://mrg.cs.ox.ac.uk/publications/programming_languages_implementations_with_mpst/main.pdf
- "Mapping BPMN2 Service Choreographies to Colored Petri Nets." 2020. https://link.springer.com/chapter/10.1007/978-3-030-57506-9_8
- W3C. "Web Services Choreography Description Language Version 1.0." https://www.w3.org/TR/ws-cdl-10/

### Complex Event Processing
- Luckham, D. "The Power of Events." Addison-Wesley, 2002.
- "Geospatial complex event processing in smart city applications." Simulation Modelling (2022).
- Apache Flink, Esper (open-source CEP engines).
