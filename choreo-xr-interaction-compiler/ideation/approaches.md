# Competing Approaches: Choreo XR Interaction Compiler

**Project**: choreo-xr-interaction-compiler  
**Phase**: Ideation  
**Date**: 2026-03-08  

---

## Approach A: Spatial Refinement Session Types ("Choreo-Types")

### Philosophy: Type Theory First — Make Verification a Type-Checking Problem

#### 1. Extreme Value and Who Needs It

**Core value**: The first type system that guarantees deadlock-freedom and spatial realizability for multi-party XR interaction protocols *at compile time*. No model checker, no state-space explosion, no scalability ceiling — if your choreography type-checks, it is deadlock-free by construction.

**Who desperately needs this**:
- **Accessibility compliance teams** facing the EU Accessibility Act 2025 and US Section 508: they need to *prove* that every scene state is reachable via eye-tracking, switch scanning, or voice alone. A type system that encodes modality-indexed session types (`session<gaze> ≤ session<hand>`) turns accessibility compliance into a type-checking problem — the first tool that can issue a machine-checkable certificate that an XR interaction is accessible.
- **Multi-user XR platform teams** (Microsoft Mesh, Meta Horizons) shipping collaborative experiences where N co-located users interact with shared spatial objects. The combinatorial explosion of interleavings is exactly the problem multiparty session types were designed to solve — but existing MPST frameworks (Scribble, mpst-rs) operate on message channels, not spatial predicates. These teams have no formal tool for reasoning about concurrent spatial interaction.
- **Standards bodies** (OpenXR, W3C Immersive Web) seeking a portable behavioral specification format: a choreography type is a platform-independent contract that any conforming runtime must satisfy.

**Value upgrade over depth-check baseline (5→7)**: Elevating accessibility auditing from a bullet point to the primary use case with regulatory framing. The type-level guarantee ("well-typed ⇒ accessible") is a stronger value proposition than post-hoc model checking because it integrates into the development workflow, not a separate verification step.

#### 2. Why This Is Genuinely Difficult

**Hard subproblem 1: Spatial session subtyping.** Standard MPST subtyping checks compatibility of message types on channels. Spatial session subtyping must check compatibility of *geometric predicates*: can a `reach(panel, <500ms)` interaction be safely replaced by a `gaze(panel, <800ms)` interaction while preserving protocol properties? This requires embedding geometric constraint satisfaction into the subtyping algorithm. The subtyping relation is no longer syntactic — it requires solving LP feasibility problems at every subtyping check, and the interaction between geometric feasibility and session type duality is unexplored.

**Hard subproblem 2: Decidability at the spatial-session boundary.** Classical binary session type checking is decidable (Gay & Hole 2005); multiparty session type checking is decidable for the Scribble fragment (Honda et al. 2016). Adding spatial refinement types threatens decidability: refinement types over arithmetic are undecidable in general (Freeman & Pfenning 1991). The key challenge is identifying a *spatial fragment* (convex polytope constraints + bounded temporal intervals) where session subtyping remains decidable while being expressive enough for real XR interactions.

**Hard subproblem 3: Choreography projection with spatial awareness.** In MPST, a global choreography type is projected onto local types for each participant. When participants are spatial entities (user hands, virtual agents, spatial anchors), projection must account for geometric reachability: a local type is only well-formed if the participant can physically reach the spatial configurations required by its projected protocol. This requires a novel projection algorithm that threads geometric feasibility through the standard endpoint projection.

**Hard subproblem 4: Modality-indexed types for accessibility.** Encoding input modalities (hand, gaze, voice, switch) as type-level indices and proving that modality substitution preserves session compatibility requires a form of parametric polymorphism over spatial predicate families — a novel combination of session types, refinement types, and indexed type families.

**Architectural challenge**: ~85K LoC. Type checker core (~20K, 65% novel), spatial constraint solver integration (~12K, 50% novel), choreography projection engine (~15K, 55% novel), modality indexing layer (~8K, 70% novel), runtime monitor generation (~15K, 40% novel), accessibility certification module (~8K, 60% novel), CLI/diagnostics (~7K, 15% novel).

#### 3. New Math Required

**M-A1: Decidability of Spatial Session Subtyping (Target: A−)**

The central theorem: for multiparty session types refined with convex-polytope spatial predicates and bounded MTL temporal constraints, subtyping is decidable in EXPTIME. The proof proceeds by reducing spatial session subtyping to a combination of (1) standard MPST subtyping (decidable, coinductive), (2) LP feasibility for convex polytope containment (polynomial), and (3) a novel *spatial compatibility check* that verifies geometric co-realizability of dual session endpoints. The key insight is that convex polytope constraints admit a *quantifier elimination* procedure (Weispfenning 1988) that can be lifted through the session type structure, preserving decidability. For the bounded-CSG extension, the problem becomes coNP-complete, admitting practical SAT encodings.

This result would be the first decidability theorem at the intersection of session type theory and computational geometry. No prior work has studied this combination. The result is surprising because adding arithmetic refinements to session types typically destroys decidability — the geometric structure (convexity, LP tractability) is what rescues it.

**M-A2: Modality-Parametric Session Soundness (Target: B+)**

Theorem: if a choreography type-checks under modality parameter M₁ and M₁ is a spatial refinement of M₂ (i.e., every spatial predicate expressible in M₂ is expressible in M₁ with relaxed thresholds), then the choreography also type-checks under M₂. Contrapositive: if the choreography fails to type-check under modality M₂ (e.g., gaze-only), then there exist scene states unreachable via M₂ — a precise accessibility violation certificate. The proof uses a simulation relation between modality-indexed spatial predicate lattices.

**M-A3: Spatial Projection Completeness (Target: B)**

Theorem: the spatial-aware endpoint projection is *complete* for the convex-polytope fragment — every collection of spatially-realizable local types that satisfies the standard MPST compatibility condition is the projection of some global choreography type. The proof constructs the global type via a geometric synthesis procedure (Minkowski sum composition of local spatial constraints). This guarantees no expressiveness is lost in the choreography → endpoint decomposition.

#### 4. Why This Has Best-Paper Potential

This approach addresses the depth check's most damning critique ("no A-grade mathematical contribution") head-on. M-A1 (decidability of spatial session subtyping) is a genuine novelty at the intersection of two mature fields (session types, computational geometry) that have never been combined. A decidability result of this kind — showing that geometric structure rescues decidability where arithmetic destroys it — would surprise PL theorists and is the type of result that earns best-paper distinction at OOPSLA or POPL.

The accessibility certification angle gives the paper a compelling societal impact narrative that reviewers increasingly value. "The first system that can issue a machine-checkable certificate that an XR interaction is accessible" is a one-sentence pitch that resonates with both PL and HCI audiences.

The approach **directly neutralizes the MPST/Scribble reviewer attack vector**: instead of ignoring session types (the original proposal's weakness), this approach *extends* them with spatial reasoning, positioning Choreo as the natural next step in the MPST research program.

#### 5. Hardest Technical Challenge

**The decidability boundary for spatial session subtyping.** The EXPTIME upper bound requires showing that quantifier elimination over convex polytopes can be threaded through the coinductive subtyping algorithm without blowing up. The risk is that the composition of these two tractable problems produces an intractable combined problem. 

**How to address it**: (1) Prove decidability first for the restricted binary session type case (2 parties, convex polytopes, no temporal constraints) — this is likely straightforward and establishes the core technique. (2) Lift to multiparty via the standard projection approach. (3) Add temporal constraints last, using the bounded MTL decidability result (Ouaknine & Worrell 2007) as a modular extension. If the full multiparty + temporal version exceeds EXPTIME, report the decidability frontier honestly (binary: EXPTIME, multiparty-convex: open, multiparty-CSG: undecidable) — even a partial decidability result is publishable.

**Extraction risk mitigation**: This approach *sidesteps* the extraction problem entirely. Developers write choreography types directly (spec-first), rather than extracting from C# code. The evaluation shifts from "find bugs in MRTK" to "express 50 canonical XR interaction patterns in the type system and verify that the type checker correctly accepts well-formed and rejects ill-formed choreographies." A secondary evaluation encodes 10 known MRTK interaction bugs as type-level violations and demonstrates that the type checker catches them.

#### 6. Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 7/10 | Accessibility certification is a real regulatory need; type-level guarantees integrate into dev workflow; multi-user XR is a growing pain point |
| **Difficulty** | 8/10 | Novel type theory + computational geometry intersection; spatial session subtyping algorithm has no precedent; multi-domain (PL theory + geometry + XR) |
| **Potential** | 8/10 | A−grade decidability result; accessibility narrative; neutralizes MPST attack vector; POPL/OOPSLA target |
| **Feasibility** | 5/10 | Decidability proof carries genuine risk (may not hold for full multiparty+temporal); ~85K LoC; no extraction but must demonstrate expressiveness on real patterns; 18-24 months |

---

## Approach B: Geometric CEGAR Bug Hunter ("Choreo-CEGAR")

### Philosophy: Verification First — Build the Best Spatial Model Checker and Find Real Bugs

#### 1. Extreme Value and Who Needs It

**Core value**: A practical bug-finding tool that extracts interaction protocols from existing XR codebases, compiles them through Event Calculus semantics into spatial event automata, and exhaustively verifies them — finding deadlocks, unreachable states, and race conditions that shipped in production XR frameworks. The spatial CEGAR loop, which refines geometric abstractions using GJK/EPA collision detection, is the technical crown jewel that makes this tractable.

**Who desperately needs this**:
- **XR platform QA teams** at Microsoft (MRTK), Meta (Interaction SDK), and Apple (RealityKit) who currently rely on manual playtest sessions with headsets to find interaction regressions. A single missed deadlock in a hand menu interaction can ship to millions of devices. A headless verifier running in GitHub Actions CI would catch protocol-level bugs before they reach QA.
- **XR interaction researchers** publishing at CHI/UIST who share interaction techniques as videos and pseudocode. Choreo programs would be reproducible, executable, and verifiable artifacts — elevating the reproducibility standard for the field.
- **Any team using XR for safety-critical applications** (surgical planning, industrial maintenance, flight simulation) where an unresponsive interaction state is not merely annoying but hazardous. These teams need formal guarantees, not test coverage percentages.

**Value upgrade over depth-check baseline (5→7)**: This approach addresses the Skeptic's "no evidence of demand" critique by front-loading the bug-finding evaluation as a gate. The first milestone is extracting and verifying 3 MRTK sample scenes. If this finds ≥1 real bug, the value proposition is empirically validated before the full system is built.

#### 2. Why This Is Genuinely Difficult

**Hard subproblem 1: Spatial CEGAR (~14K LoC, 65% novel).** Standard predicate-abstraction CEGAR refines by adding Boolean predicates to eliminate spurious counterexamples. Spatial CEGAR must refine by *splitting geometric regions*: when a counterexample is spurious because two zones cannot physically overlap, the refinement step solves a spatial constraint satisfaction problem in ℝ³ using GJK/EPA. This requires a novel refinement operator that maps geometric infeasibility witnesses to abstract domain splits. No published CEGAR implementation performs geometric refinement.

**Hard subproblem 2: EC→automata compilation (~15K LoC, 50% novel).** All prior Event Calculus work translates automata *into* EC for reasoning. Choreo reverses this: compiling EC *out* into executable automata. The Thompson-style construction must be extended with spatial-temporal transition guards, and product composition for multi-pattern choreographies must use on-the-fly symbolic construction to avoid exponential blowup. The key innovation is *spatial guard compilation*: translating high-level spatial predicates (gaze-cone intersection, proximity with hysteresis) into efficient R-tree query plans that serve as automaton transition guards.

**Hard subproblem 3: Extraction from MRTK canonical components (~10K LoC, 30% novel).** Parsing Unity .prefab YAML and C# source (via Roslyn) to recover interaction state machines from ~15 canonical MRTK components (Interactable, NearInteractionGrabbable, ManipulationHandler, SolverHandler, BoundsControl, EyeTrackingTarget, etc.). Each component has a documented state machine, but the actual implementation is scattered across MonoBehaviour callbacks, coroutines, and event handlers. The extraction must be *conservative*: over-approximating the possible behaviors to ensure the verifier's bug reports are not false positives.

**Hard subproblem 4: Geometric consistency pruning (M2 upgrade).** Tightening the bounds on the feasible predicate set C ⊆ 2^P from the current B− characterization to a B+ result by proving that for XR-typical predicate vocabularies (proximity at k thresholds, containment in hierarchical zones, gaze-cone intersection), |C| ≤ O(poly(n,k)) rather than O(m^(n²)). This requires exploiting the *hierarchical containment structure* of XR scenes (rooms contain tables contain panels) — a constraint absent from generic metric spaces.

**Architectural challenge**: ~110K LoC total (reduced from original 147K by dropping M3 and limiting the type system). Spatial CEGAR verifier (~14K), EC→automata compiler (~15K), extraction pipeline (~10K), DSL parser + lightweight type checker (~12K), R-tree spatial index (~8.5K), runtime engine (~13K), headless simulator (~11.5K), deadlock detector (~7.5K), test harness (~10K), CLI/diagnostics (~9K).

#### 3. New Math Required

**M-B1: Tight Geometric Pruning for Hierarchical Spatial Scenes (Target: B+)**

Upgrade M2 from B− to B+ by exploiting hierarchical containment. Define the *containment DAG* D of an XR scene: nodes are spatial regions, edges are containment relations. Theorem: for scenes with containment DAG of depth d and branching factor b, the feasible predicate set satisfies |C| ≤ O(b^d · k^d) where k is the number of proximity thresholds, versus the naive 2^(n²·k). For typical XR scenes (d ≤ 4, b ≤ 6, k ≤ 3), this yields |C| ≈ 10³–10⁴ versus 2^|P| ≈ 10⁶–10⁹. The proof uses a layer-by-layer constraint propagation argument: containment constraints at each DAG level eliminate exponentially many predicate valuations at deeper levels.

The connection to *zone graphs* in timed automata (Bengtsson & Yi 2004) provides proof structure, but the geometric instantiation — operating over hierarchical spatial containment rather than clock zones — is novel. The key lemma (containment monotonicity amplification) has no analog in the timed automata literature.

**M-B2: Soundness of Conservative Extraction (Target: B)**

Theorem: for MRTK canonical components whose interaction state machines conform to the documented transition schemas, the extraction produces an over-approximation — every behavior of the real Unity runtime is a behavior of the extracted Choreo model. Contrapositive: any deadlock detected in the extracted model is a *possible* deadlock in the real system (no false positives under the documented schema assumption). The proof is by structural induction over the ~15 supported component types, with each component's extraction validated against its documented state machine.

This is not deep mathematics, but it is a crucial engineering theorem that addresses the depth check's most dangerous fatal flaw (F1: extraction fidelity gap). The theorem's *conditional* nature (sound under documented schemas) must be reported honestly.

**M-B3: Spatial CEGAR Termination and Precision (Target: B)**

Theorem: the spatial CEGAR loop terminates in at most |P|·2^d refinement steps (where d is the containment DAG depth), and the final abstraction is the coarsest geometric abstraction that eliminates all spurious counterexamples reachable within the BMC bound. The proof uses a well-founded ordering on geometric partitions (finer partitions are strictly smaller in a lattice ordered by spatial refinement). The termination bound is novel — standard CEGAR termination arguments (Henzinger et al. 2004) do not apply because the refinement domain is geometric, not propositional.

#### 4. Why This Has Best-Paper Potential

This approach leans into the depth check's recommendation: "lead with spatial CEGAR as the most technically impressive contribution" and "the bug-finding evaluation is the stronger leg." The strategy is to maximize the probability of the P(bug-finding) × P(novel-verification) conjunction:

- **The spatial CEGAR loop is genuinely novel.** GJK/EPA collision detection inside a CEGAR refinement loop has zero published precedent. This alone is a systems-verification contribution worth a strong accept.
- **The EC→automata reversal is architecturally novel.** Compiling *out of* Event Calculus (rather than *into* it) opens a new direction for the EC community.
- **If bug-finding delivers (≥5 anomalies, ≥2 corroborated), the paper is undeniable.** A tool that finds real bugs in Microsoft's MRTK — the most-used XR interaction framework — is a practical impact story that reviewers cannot dismiss.
- **Honest scalability reporting builds credibility.** "We verify scenes up to 20 zones in seconds; here is exactly where and why our approach hits its ceiling" is more convincing than inflated claims.

The math portfolio (B+, B, B) doesn't have an A-grade result, but the total package — novel verification algorithm + novel compiler direction + real bugs found — is a strong OOPSLA systems paper. Best-paper probability: ~12-18% (higher than baseline because the two independent success criteria are multiplicative).

#### 5. Hardest Technical Challenge

**Extraction fidelity from MRTK C# code.** This is the project's existential risk (depth check F1). If the extracted models don't correspond to real runtime behavior, every subsequent result is built on sand.

**How to address it**: 
1. **Gate milestone (weeks 1-3)**: Extract 3 MRTK sample scenes, manually verify extraction against documented state machines, run verifier. If extraction detects ≥1 of the ~20 known interaction bugs in MRTK's issue tracker, proceed. If extraction produces models with >30% false positive rate, pivot to Approach A (spec-first, no extraction).
2. **Scope ruthlessly**: Support only 15 canonical MRTK component types with well-defined YAML schemas. Reject scenes using unsupported components rather than producing unsound extractions.
3. **Conservative over-approximation**: When extraction encounters ambiguous control flow (coroutines, async callbacks), add nondeterministic transitions that over-approximate possible behaviors. This may increase false positives but eliminates false negatives.
4. **Validation oracle**: For each supported component type, build a reference Choreo model from MRTK's documentation. Compare extraction output against the reference model using trace equivalence checking. Report coverage: what fraction of each component's documented behavior is captured.

#### 6. Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 7/10 | Real bug-finding in production frameworks; CI/CD integration for XR teams; front-loaded empirical validation |
| **Difficulty** | 7/10 | Spatial CEGAR is novel; EC→automata reversal is novel; extraction is hard engineering; but individual components are well-understood |
| **Potential** | 6/10 | No A-grade math; strength is systems contribution + empirical impact; OOPSLA strong-accept target, best-paper only if bug-finding over-delivers |
| **Feasibility** | 7/10 | ~110K LoC; extraction risk is mitigated by gate milestone; drops M3 dependency; 15-20 months; honest about scalability ceiling |

---

## Approach C: Spatial-Temporal Reactive Synthesis ("Choreo-Synth")

### Philosophy: Synthesis First — Don't Verify Broken Code, Synthesize Correct Controllers

#### 1. Extreme Value and Who Needs It

**Core value**: Instead of extracting interaction logic from buggy imperative code and then verifying it (Approach B's risky path), *synthesize provably correct interaction controllers directly from spatial-temporal specifications*. The developer writes what the interaction *should do* (a spatial-temporal specification in a fragment of Signal Temporal Logic extended with geometric predicates); the synthesizer produces a reactive controller that is *correct by construction* — guaranteed to satisfy the specification for all spatial configurations within declared scene constraints.

**Who desperately needs this**:
- **XR framework architects** designing the next generation of interaction toolkits. Instead of hand-coding state machines for each interaction pattern (the current practice that produces bugs), they specify the desired behavior declaratively and let the synthesizer produce a provably correct implementation. This is the difference between writing assembly and writing in a high-level language with a correct compiler.
- **Safety-critical XR application developers** (surgical guidance, industrial AR, military training) who need formal guarantees that interaction controllers never enter hazardous states (e.g., a surgical overlay never becomes unresponsive during a procedure). Reactive synthesis provides the strongest guarantee possible: correctness *for all possible environments*, not just tested scenarios.
- **XR interaction researchers** who want to explore the design space of interaction techniques. Given a spatial-temporal specification (e.g., "the menu activates within 300ms of gaze fixation, and deactivates if gaze leaves for >500ms, and never activates during a grab gesture"), the synthesizer either produces a correct controller or reports that the specification is unrealizable — giving immediate feedback on whether a proposed interaction technique is even implementable.

**Value upgrade over depth-check baseline (5→8)**: Synthesis completely sidesteps the extraction problem (depth check F1) — there is nothing to extract. It also sidesteps the scalability ceiling (depth check F2) because synthesis operates per-controller rather than on monolithic scene graphs. And it addresses the LLM context shift: LLMs can generate plausible interaction code but cannot guarantee correctness for all inputs. Synthesis provides the guarantee that LLMs cannot.

#### 2. Why This Is Genuinely Difficult

**Hard subproblem 1: Decidability of spatial reactive synthesis.** Classical reactive synthesis (Church's problem) for LTL specifications is 2EXPTIME-complete (Pnueli & Rosner 1989). Extending to *spatial* specifications — where the environment's moves include changing spatial configurations (entity positions, gaze directions) and the controller must respond with correct interaction state transitions — introduces continuous state into the synthesis game. The key challenge is identifying a decidable fragment: spatial predicates over convex polytopes with bounded MTL temporal constraints. The synthesis game must be *finitely abstracted* via geometric discretization while preserving realizability.

**Hard subproblem 2: Spatial environment modeling.** In reactive synthesis, the environment is the adversary. For XR interactions, the "environment" is the user's physical behavior — hand movements, gaze shifts, body position. The synthesizer must handle *all possible* user behaviors within physical constraints (reachability envelopes, maximum hand velocity, gaze saccade limits). Encoding these constraints as a spatial environment model that is tight enough to enable synthesis (not overly permissive) but sound (doesn't exclude real user behaviors) requires a novel *spatial environment abstraction* grounded in human kinematic models.

**Hard subproblem 3: Controller code generation.** The synthesized reactive controller must be compiled into efficient executable code — not a BDD representation of a winning strategy (the standard synthesis output) but actual event-handling code that integrates with XR runtime event loops. This requires a novel *spatial strategy compilation* pass that translates BDD-encoded strategies into R-tree-backed event automata with efficient guard evaluation.

**Hard subproblem 4: Compositional synthesis for multi-pattern scenes.** Synthesizing a single interaction controller is tractable for small specifications. Composing multiple controllers for a full scene (menu + grab + gaze-dwell + timeout) while preserving non-interference requires *assume-guarantee synthesis*: each controller is synthesized under assumptions about the others' behavior, and the composition is verified to be consistent. The spatial dimension adds complexity because controllers that are logically independent may interfere spatially (overlapping activation zones).

**Architectural challenge**: ~95K LoC. Specification parser + spatial STL frontend (~10K, 30% novel), spatial environment model (~12K, 60% novel), synthesis game construction (~18K, 70% novel), BDD-based synthesis solver (~12K, 40% novel — extends existing synthesis tools), spatial strategy compiler (~15K, 65% novel), R-tree runtime backend (~8.5K, 35% novel), assume-guarantee composition engine (~10K, 60% novel), realizability checker (~5K, 55% novel), CLI/diagnostics (~4.5K, 15% novel).

#### 3. New Math Required

**M-C1: Decidability Frontier for Spatial Reactive Synthesis (Target: A−)**

The central theorem: reactive synthesis for specifications in Spatial Signal Temporal Logic (S-STL) — STL extended with convex-polytope spatial predicates — is decidable when (1) spatial predicates are restricted to convex polytope containment and proximity with finitely many thresholds, (2) temporal operators use bounded intervals, and (3) the spatial environment is constrained by a polyhedral kinematic envelope. The proof constructs a finite *spatial game arena* by discretizing the continuous spatial state space into a polyhedral partition induced by the predicate thresholds and kinematic bounds, then reduces to a standard parity game on the finite arena.

The decidability is surprising because STL synthesis over continuous signals is generally undecidable (Raman et al. 2015). The geometric structure of convex polytopes — specifically, that polyhedral partitions are finite and computable — is what rescues decidability. The complexity is doubly exponential in the number of spatial predicates and exponential in the number of temporal operators, but for typical XR specifications (5-15 predicates, 3-8 temporal operators), the game arena is manageable (~10⁴–10⁶ states).

This result directly extends the classical Pnueli-Rosner synthesis framework to spatial domains for the first time. It opens a new research direction: spatial reactive synthesis. The result would be of independent interest to the formal methods and robotics communities.

**M-C2: Spatial Environment Abstraction Soundness (Target: B+)**

Theorem: for a polyhedral kinematic envelope K that over-approximates the reachable set of human hand/gaze positions, a controller synthesized against the K-abstracted environment is correct for all physically realizable user behaviors. The proof uses a simulation relation between the continuous human kinematic model (bounded velocity, joint angle constraints) and the polyhedral over-approximation, showing that every continuous trajectory has a corresponding abstract trajectory. The key technical challenge is proving the over-approximation is *tight enough* for synthesis to succeed — an overly conservative envelope makes specifications unrealizable that should be realizable. The bound relates the approximation error to the minimum spatial predicate threshold, giving a constructive criterion for choosing the envelope granularity.

**M-C3: Assume-Guarantee Spatial Synthesis Compositionality (Target: B)**

Theorem: if k interaction controllers are individually synthesized under spatial assume-guarantee contracts (each controller assumes non-interference from others outside its declared spatial scope), and the spatial scopes are *geometrically separable* (the convex hulls of their activation zones are pairwise disjoint or have bounded overlap), then the parallel composition of the k controllers satisfies all k specifications simultaneously. The proof uses a spatial variant of the Abadi-Lamport assume-guarantee rule, where the induction is over a spatial decomposition tree rather than a temporal ordering. The geometric separability condition is the key innovation — it turns a hard compositional reasoning problem into a tractable geometric test (convex hull intersection, computable in O(n log n)).

#### 4. Why This Has Best-Paper Potential

**This is the highest-ceiling approach.** If M-C1 delivers, it establishes a new subfield: spatial reactive synthesis. The result connects three major research communities (reactive synthesis, computational geometry, XR/HCI) in a way that none of them have explored. A decidability theorem for spatial reactive synthesis would be the first result of its kind — a genuinely novel contribution that experts in all three fields would find surprising.

**The narrative is compelling**: "XR interaction developers hand-write buggy state machines. We show that these state machines can be *automatically synthesized* from declarative spatial-temporal specifications, with a proof that the synthesized controller is correct for all possible user behaviors. The key enabling result is a new decidability theorem showing that spatial reactive synthesis is tractable when spatial constraints are convex." This is a "new capability" paper — something that was literally impossible before — rather than an "incremental improvement" paper.

**Completely sidesteps the extraction problem.** There is no extraction from C#. The evaluation compares synthesized controllers against hand-written MRTK controllers on behavioral equivalence and performance. If synthesized controllers are functionally equivalent to hand-written ones (for the canonical interaction patterns), the argument is airtight: "why hand-write what you can synthesize?"

**Addresses the LLM concern definitively.** LLMs can generate plausible XR interaction code, but they cannot guarantee correctness for adversarial inputs. Synthesis provides a *formal guarantee* that LLMs fundamentally cannot match, positioning this work as complementary to (not competing with) LLM-based code generation.

**Best-paper probability: ~15-22%** — higher ceiling than Approach B but more risk. If M-C1 holds, this is a potential POPL/PLDI best paper. If M-C1 fails, the fallback is a weaker "bounded synthesis for XR" systems paper (P ≈ 5%).

#### 5. Hardest Technical Challenge

**The synthesis game arena may be too large for practical XR specifications.** The doubly-exponential worst-case complexity of reactive synthesis means that even with the spatial decidability result, synthesis may time out on specifications with >10 spatial predicates. This is the make-or-break scalability question.

**How to address it**:
1. **Exploit spatial locality for arena reduction.** Most XR interactions are spatially local — a hand menu interaction doesn't depend on entities across the room. Define *spatial influence cones* for each specification clause and partition the synthesis game into independent sub-games for spatially separated clauses. This reduces the effective number of predicates per sub-game from n to O(1)–O(5) for typical XR scenes.
2. **Bounded synthesis as a practical fallback.** If full reactive synthesis is too expensive, use *bounded synthesis* (Schewe & Finkbeiner 2007): search for controllers with at most k states. For XR interaction controllers (typically 3-15 states), bounded synthesis with k ≤ 20 is tractable. This sacrifices the optimality guarantee (smallest correct controller) but preserves the correctness guarantee.
3. **Incremental synthesis for iterative design.** Cache and reuse sub-strategies when the specification changes incrementally (e.g., adjusting a timeout threshold). The spatial partitioning enables efficient reuse because spatially-separated sub-strategies are independent.
4. **Empirical calibration (weeks 1-4)**: Encode 10 canonical MRTK interaction patterns as S-STL specifications, measure synthesis time. If >7/10 synthesize in <60 seconds on a laptop, the approach is feasible. If <3/10 synthesize, pivot to Approach B.

#### 6. Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 8/10 | Correct-by-construction controllers eliminate entire bug classes; sidesteps extraction risk; addresses safety-critical XR; LLM-complementary |
| **Difficulty** | 9/10 | Novel decidability result at synthesis × geometry boundary; spatial environment modeling; compositional synthesis with geometric separability; multi-domain expertise (synthesis + geometry + XR + type theory) |
| **Potential** | 9/10 | A−grade decidability result; opens new subfield (spatial reactive synthesis); POPL/PLDI target; highest ceiling of all three approaches |
| **Feasibility** | 4/10 | Decidability theorem carries high risk (~40% chance of failure); synthesis scalability uncertain; ~95K LoC; 20-28 months; doubly-exponential worst case may limit practical scope; no published precedent for spatial synthesis |

---

## Comparative Summary

| Dimension | A: Spatial Session Types | B: Geometric CEGAR | C: Reactive Synthesis |
|-----------|--------------------------|--------------------|-----------------------|
| **Core abstraction** | Multiparty session types + spatial refinements | Event automata + R-tree guards | Reactive games + spatial strategies |
| **Verification strategy** | Type-level guarantees (well-typed ⇒ safe) | Model checking with geometric CEGAR | Correct by construction (synthesis) |
| **Compilation target** | Monitor automata from projected session types | EC-compiled spatial event automata | BDD strategies → R-tree controllers |
| **Primary math** | Decidability of spatial session subtyping (A−) | Tight geometric pruning bounds (B+) | Spatial reactive synthesis decidability (A−) |
| **Evaluation strategy** | Expressiveness + type-error quality + accessibility certification | Bug-finding on 50+ real XR projects | Synthesized vs. hand-written controllers |
| **Extraction risk** | None (spec-first) | High (mitigated by gate milestone) | None (synthesis, not extraction) |
| **Scalability model** | Per-protocol type checking (linear) | Per-scene model checking (~15-20 zones) | Per-controller synthesis (5-15 predicates) |
| **LLM resilience** | Moderate (type systems still valuable in LLM era) | Low-moderate (LLMs absorb bug-finding) | High (synthesis guarantees > LLM heuristics) |
| **Prior art differentiation** | Extends MPST to spatial domain | Extends CEGAR to geometric domain | Extends reactive synthesis to spatial domain |
| **Risk profile** | Medium (decidability may not extend to full fragment) | Medium (extraction may fail) | High (synthesis may not scale) |
| **Value** | 7 | 7 | 8 |
| **Difficulty** | 8 | 7 | 9 |
| **Potential** | 8 | 6 | 9 |
| **Feasibility** | 5 | 7 | 4 |
| **Composite (equal weight)** | **7.0** | **6.75** | **7.5** |

### Recommended Strategy

**Lead with Approach C (Reactive Synthesis)** — highest ceiling, addresses all depth-check concerns, and opens a genuinely new research direction. However, the feasibility risk is substantial.

**Hedge with Approach B (Geometric CEGAR)** as the fallback — most feasible, lowest risk, and still delivers a strong systems contribution. Begin extraction prototyping in parallel with M-C1 decidability investigation.

**Keep Approach A (Session Types)** as a pivot option — if M-C1 fails and extraction also fails, the session types approach requires neither and has a strong independent math contribution (M-A1).

**Concrete decision gate (week 4)**: 
- If M-C1 proof sketch is viable AND 7/10 canonical patterns synthesize in <60s → commit to Approach C
- If M-C1 stalls BUT extraction prototype finds ≥1 MRTK bug → commit to Approach B  
- If both fail → commit to Approach A (spec-first, no extraction, no synthesis)
