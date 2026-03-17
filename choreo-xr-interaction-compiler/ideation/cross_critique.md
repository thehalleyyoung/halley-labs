# Phase 2: Adversarial Cross-Critique

**Project**: XR Event Calculus Compiler  
**Community**: area-060-graphics-simulation-and-mixed-reality  
**Date**: 2025-07-18  
**Status**: Crystallization — Adversarial Cross-Critique

---

## Part 1: Architecture Critique (from Math, Scope, and Prior Art Perspectives)

### Framing 1 — Choreo-Check (Verification)

**Math Lead's Critique: Is the verification math genuinely novel or routine model checking?**

It is routine model checking. The Math Spec (§3) is unambiguous: reachability and deadlock detection over the abstract state space S_abstract are decidable because S_abstract is finite. The proof is one line ("graph reachability in a finite graph"). The applicable algorithms—BFS, BDD-based symbolic model checking, SAT-based BMC, CEGAR—are all off-the-shelf and have been implemented in SPIN, UPPAAL, and NuSMV for decades. The Math Spec grades the verification infrastructure at **C** (entirely standard) with a bump to **B−** only if the geometric consistency pruning (Theorem 3.2) is worked out tightly. But Theorem 3.2 is speculative—the Math Spec does not prove it yields practically meaningful bounds, only that the monotonicity and triangle-inequality constraints *exist*. The hardest theorem needed is the characterization of geometrically consistent predicate valuations C ⊆ 2^P with tight cardinality bounds, and this is a metric constraint satisfaction problem, not a model-checking contribution. A UIST reviewer who knows UPPAAL will see through the "novel verification" claim in five minutes.

**Scope Lead's Critique: Does verification alone justify 150K+ LoC?**

No. The Reachability Checker (subsystem 9) is estimated at 12–15K LoC. The Deadlock Detector (subsystem 10) is 6–8K LoC. Together they are ~20K LoC. Even adding the R-tree (8K), the Event Calculus engine (12.5K), and the spatial-temporal type system (10K) as verification prerequisites, the verification-specific novel code tops out around 50K LoC. The remaining ~100K LoC is compiler frontend, runtime, simulator, cross-platform plumbing, CLI, benchmarks, and diagnostics—none of which is verification-novel. The Scope Lead's own numbers say the Reachability Checker is 65% novel (the highest of any subsystem), but 65% of 13.5K is only ~9K genuinely novel LoC. Choreo-Check as a pure verification story is a 50K LoC project wearing a 150K LoC trench coat. The padding is visible.

**Prior Art's Critique: How close is iv4XR or UPPAAL+spatial extensions?**

Closer than the Architect Framings admit. iv4XR (EU Horizon 2020) already performs agent-based intelligent V&V for XR with formal assertions. It lacks model checking, but the conceptual distance from "agents explore and assert" to "exhaustive state-space search" is one PhD student's worth of work on an existing codebase, not a field-defining gap. UPPAAL already handles timed automata with zone-based abstractions; adding spatial predicates as additional Boolean guards is an engineering extension, not a theoretical leap. The Prior Art Audit rates "formal verification of XR interactions" as **Novel**, but qualifies that "the underlying verification techniques are standard." The real novelty is the *application domain*, not the *technique*. A reviewer familiar with both UPPAAL and XR will ask: "Why didn't you just encode your spatial predicates in UPPAAL and use its existing model checker?" The answer ("because UPPAAL doesn't have R-trees") is an implementation convenience argument, not a research argument.

**Synthesis**: Choreo-Check's value proposition ("find deadlocks in XR interactions") is compelling to practitioners, but its *research* contribution is thin. The math is routine, the LoC is padded, and the novelty gap over UPPAAL+spatial is narrower than claimed. This is a strong *tool paper*, not a best paper.

---

### Framing 2 — Choreo-Lang (Compiler/DSL)

**Math Lead's Critique: Is the spatial type system math novel or applied PL theory?**

Applied PL theory with a geometric twist. The Math Spec (§2) grades the spatial-temporal type system at the subsystem level but doesn't assign it a standalone theorem. The Scope Lead (subsystem 2) claims 60% novelty for the type system, citing "decidable spatial subtyping + Allen's intervals." But Allen's 13 interval relations are 1983 textbook material, and spatial subtyping via containment is a restriction of computational geometry (polytope containment is decidable for convex polytopes; undecidable in general). The Math Spec's closest theorem is Theorem 3.2 (geometric consistency pruning), which is about verification, not the type system itself. The type system needs a soundness theorem: "if a program type-checks, then the compiled automata are spatially realizable." This theorem is *stated nowhere in the Math Spec*. Can you prove the type system sound? Maybe—if the spatial fragment is restricted to convex polytopes, soundness reduces to linear programming feasibility, which is polynomial-time decidable. But the Architect Framings mention "CSG combinations of primitives" and non-convex regions, which pushes soundness checking into NP-hardness territory. The type system soundness proof is the missing theorem that could elevate this framing, but only if the language is restricted enough to make the proof interesting rather than trivially "call an SMT solver."

**Scope Lead's Critique: Is a DSL compiler alone 150K+ LoC of genuine complexity?**

The honest answer: a DSL compiler without the runtime and verification backends is ~60K LoC (subsystems 1, 2, 3, 5, 6 = parser + types + EC + automata compiler + incremental compilation ≈ 61.5K LoC at mid-estimates). This is a substantial compiler but not 150K+. The Architect Framings get to 150–180K by adding "4 runtimes" (cross-platform targets), but the Scope Lead rates the cross-platform layer at 15% novelty—it's the weakest subsystem. A DSL compiler that targets *one* platform is ~68K LoC (adding a single runtime at 12.5K). To hit 150K+ honestly, Choreo-Lang must include the verification backends (reachability + deadlock = ~20K), the headless simulator (~11.5K), the test harness (~9.5K), the evaluation infrastructure (~11.5K), and the diagnostics (~6.5K). At that point, it's no longer "a compiler"—it's the full system. The framing as "just a compiler/DSL" is misleading about what actually gets built. The genuine novel compiler LoC is roughly: type system (6K novel) + EC engine (8.75K novel) + automata compiler (7.25K novel) ≈ 22K LoC. That's a solid PLDI paper, but it's not 150K+.

**Prior Art's Critique: How close is SpatiaLang?**

Uncomfortably close on the surface. SpatiaLang is a DSL for AR/VR spatial interactions that compiles to Unity/ARCore code. The Prior Art Audit rates the DSL component as **Incrementally Novel** given SpatiaLang's existence—the harshest rating in the audit. The differentiators are: (1) Event Calculus semantics, (2) compilation to verifiable automata, and (3) headless execution. These are real, but they also show that Choreo-Lang's novelty comes from the *backend* (verification + headless testing), not the *language itself*. A harsh reviewer could argue: "Take SpatiaLang, add temporal operators, and compile to UPPAAL timed automata instead of Unity code. What's left that's new about your *language*?" The answer must be the spatial type system with formal soundness guarantees—but as noted above, that theorem doesn't exist yet. The strongest differentiator would be a proof that the type system rejects all spatially unrealizable programs while accepting all realizable ones (decidable completeness), but this requires restricting the geometry fragment severely.

**Synthesis**: Choreo-Lang has the highest ceiling (new languages reshape fields) but the shakiest formal foundation. The spatial type system soundness theorem is the make-or-break contribution, and it doesn't exist yet. Without it, Choreo-Lang is "SpatiaLang plus some temporal operators and a fancier backend"—incrementally novel, not field-defining. The 150K+ claim only works if you count the full system, at which point the framing as "a compiler" is dishonest.

---

### Framing 3 — Choreo-Engine (Simulation/Runtime)

**Math Lead's Critique: What's the hardest math in the runtime?**

There isn't much. The Math Spec (§4) grades incremental compilation (the closest analogue to runtime incremental execution) at **C**—"entirely standard application of known incremental computation techniques." The runtime execution engine (subsystem 7 in the Scope doc) involves NFA token-passing with spatial guard evaluation, which is well-understood. The biomechanical input synthesis mentioned in the Architect Framings is a modeling contribution, not a mathematical one—deriving hand trajectories from Fitts's Law and kinematic equations is applied motor control, not novel mathematics. The guided search over continuous input spaces is the most mathematically interesting component: it's a planning problem in hybrid discrete-continuous space, related to hybrid systems reachability. But the Math Spec doesn't even mention this as a theorem candidate, which suggests it's either out of scope or treated as heuristic engineering. Is incremental correctness non-trivial? The Math Spec's Theorem 4.1 (incremental correctness) is described as "a standard correctness argument for incremental systems." So no: it is not non-trivial in the mathematical sense, only in the engineering sense.

**Scope Lead's Critique: Is a runtime engine 150K+ LoC without becoming middleware bloat?**

The runtime-specific subsystems are: Runtime Execution Engine (12.5K), Headless Scene Simulator (11.5K), R-tree (8K), Cross-Platform Abstraction (7K) ≈ 39K LoC. At 45% average novelty, that's ~17.5K novel LoC. This is not 150K+ by any stretch. To reach 150K+, the Engine framing must include the full compiler pipeline, the verification backends, the test harness, and the evaluation infrastructure—at which point it's the same system as the other two framings, just with a different pitch deck. The Architect Framings estimate 140–170K for Choreo-Engine, but this includes "importers" (scene extraction from Unity/Unreal/WebXR = 20–30K LoC per the Architect's own estimate). The Scope Lead doesn't break out importers separately; they're implicitly in the Headless Simulator (subsystem 8) and Cross-Platform Abstraction (subsystem 11). If scene extraction is 20–30K LoC, the Scope Lead's estimates are understated—but the Scope Lead also rates this work at 15–35% novelty. Adding 25K LoC of format-wrangling code to reach 150K+ is the definition of middleware bloat.

**Prior Art's Critique: How is this different from a game engine with formal semantics bolted on?**

This is the killer question, and the Architect Framings partially acknowledge it: "Tools like Unity Test Framework and Unreal Automation Framework support limited headless testing. Reviewers may argue that Choreo-Engine's advantage over extending these existing tools is incremental rather than fundamental." The Prior Art Audit is kinder—it rates "true headless XR testing" as **Novel**—but the novelty is in the *concept* (CPU-only, no rendering), not in the *implementation*. A game engine in headless mode (Unity has one; Unreal has one) already runs interaction logic without rendering. Choreo-Engine's claim is that it adds formal semantics, but the formal semantics come from the compiler and verification backends—not the runtime itself. Strip those away and you have a scene graph + an R-tree + an automaton interpreter, which is... a game engine's interaction subsystem. The differentiation comes entirely from the formal foundations, which means the Engine framing is parasitic on the Compiler and Verification framings for its novelty. It has no standalone research contribution.

**Synthesis**: Choreo-Engine is the most immediately useful framing but the weakest research contribution. The math is shallow, the LoC is inflated by format-wrangling, and the core novelty belongs to the compiler and verifier, not the runtime. This is a systems/engineering paper, and a strong one—but "strong systems paper" is not "best paper" at most venues.

---

## Part 2: Math vs. Engineering Assessment

### The Hard Truth

The Math Spec Lead found **no Grade-A mathematical contributions**. The best grades are two **B−** results:

1. **Geometric consistency pruning** (Theorem 3.2): Exploiting metric-space structure to prune the spatial abstract state space. Potentially interesting but unproven—the Math Spec says the characterization of C "needs to be worked out carefully to determine if it's truly non-trivial."

2. **End-to-end compiler correctness under Lipschitz sampling** (Theorem 5.4): Composing sampling soundness with compiler correctness. The Math Spec calls it "mildly non-trivial" but "not a new proof technique."

Everything else—the Spatial Event Calculus formalization (C+), the R-tree automata model (C), incremental compilation (C)—is routine application of known theory.

### Question 1: Can ANY angle produce genuinely new mathematics?

**Three candidates, in decreasing order of plausibility:**

1. **Decidability boundary for spatial-temporal verification.** The Math Spec suggests (§3 recommendation) finding the maximal fragment of the DSL for which verification is polynomial-time rather than PSPACE-complete. If the spatial structure yields a tractability result analogous to how bounded clocks keep timed automata decidable, this could be a genuine **B+/A−** contribution. The key question: does the geometry of XR scenes (most interaction zones are far apart, spatially independent, and axis-aligned) yield structural sparsity that makes the product automaton state space tractable? If yes, this is a tight, publishable theorem. If no, you've just reproduced the standard PSPACE-completeness result with extra variables.

2. **Spatial type system soundness with tight complexity bounds.** If the type system can be proved sound (well-typed programs always compile to spatially realizable automata) and the checking algorithm has an interesting complexity profile (e.g., NP-complete for general CSG but polynomial for convex regions with a practical SAT encoding), this would be a legitimate PL+geometry contribution. But no one has even *stated* this theorem yet, let alone proved it.

3. **Mechanized compiler correctness in Lean/Coq.** Not new mathematics per se, but a mechanized proof of Theorem 5.4 (end-to-end correctness with continuous-to-discrete sampling) would be a notable verification artifact. The Math Spec explicitly suggests this. However, mechanized proofs are the deliverable, not the discovery—reviewers want new *theorems*, not new *proofs of known theorems*.

### Question 2: Is this fundamentally an engineering/systems contribution?

**Yes.** The Prior Art Audit's verdict is "architectural innovation—a carefully chosen combination of well-understood components applied to an underserved domain." The Math Spec's verdict is "engineering contribution with formal foundations, not a mathematical contribution per se." The Scope Lead's analysis shows ~34% genuinely novel code, but "novel engineering" ≠ "novel mathematics." The intellectual contribution is in the *integration*: no one has previously combined Event Calculus + R-trees + automata + spatial types + headless XR testing into a coherent system. This is how React, Halide, and TVM were novel—but React didn't win best papers at PL theory conferences. It won adoption.

### Question 3: Which venues reward systems contributions as best papers?

| Venue | Systems-paper friendliness | Fit for this project | Notes |
|-------|---------------------------|---------------------|-------|
| **OOPSLA** | ★★★★★ | ★★★★★ | Best fit. Rewards DSL compilers with solid engineering. Halide, TVM-lineage papers win here. "New programming model for XR interaction" is an OOPSLA pitch. |
| **SIGGRAPH** | ★★★★☆ | ★★★☆☆ | Rewards systems that advance graphics/interaction practice. But expects visual/demo wow factor, which CPU-only headless testing lacks. |
| **UIST** | ★★★★☆ | ★★★★☆ | Rewards novel interaction tools. "Headless XR testing that finds real bugs" is compelling. But typically expects a user study for best paper. |
| **PLDI** | ★★★☆☆ | ★★★☆☆ | Rewards compilers with formal foundations. Needs the type-system soundness theorem to be competitive. Without it, it's a tool paper. |
| **IEEE VR / ISMAR** | ★★★★☆ | ★★★★☆ | Rewards XR systems innovation. Strong fit for the domain, but smaller community = less best-paper prestige. |
| **CHI** | ★★★☆☆ | ★★☆☆☆ | Demands user studies for best paper. The "no human studies" constraint is fatal at CHI. |
| **ICSE / FSE** | ★★★★☆ | ★★★☆☆ | Software engineering venues. "Formal testing for XR" fits, but may seem too niche. |

**Recommendation**: Target **OOPSLA** (DSL compiler + formal semantics + evaluation on real codebases) or **UIST** (interaction tool that finds real bugs). CHI is out without human studies. SIGGRAPH wants visuals. PLDI needs the type-soundness theorem.

### Question 4: What would need to change to produce Grade-A math?

To get genuinely new mathematics (Grade A), the project must find a **structural result that surprises experts in at least one field** (model checking, PL theory, or computational geometry). Three possible paths:

1. **Prove a spatial tractability theorem.** Show that for XR scenes satisfying a natural structural property (e.g., bounded spatial independence number—interaction zones form a graph of bounded treewidth), the verification problem drops from PSPACE-complete to polynomial time. This would surprise the model-checking community (treewidth-based tractability is studied for message-passing, not spatial systems) and the XR community (nobody knew their scenes had exploitable structure). This requires identifying the right structural parameter and proving the tight bound. Estimated difficulty: high (6–12 months of focused theoretical work).

2. **Prove a spatial type soundness theorem with a non-trivial decidability result.** Show that type-checking the spatial-temporal DSL (determining if a program's spatial constraints are jointly satisfiable) is decidable for a fragment strictly larger than convex polytopes—e.g., star-shaped regions or bounded CSG depth—with a complexity that is interestingly lower than the general case. This requires a new decision procedure, not just "call Z3." Estimated difficulty: medium-high (3–6 months if the right fragment exists).

3. **Establish a new connection between Event Calculus and automata theory.** The Prior Art Audit notes that prior work translates automata→EC, while this project does EC→automata. If the translation preserves interesting properties (e.g., if the spatial EC fragment has a Büchi-automaton characterization that enables novel liveness verification), this could be a genuine automata-theory contribution. Estimated difficulty: uncertain (could be trivial or deep, depending on the fragment).

**Brutal honesty**: None of these is guaranteed to work, and all require theoretical expertise that may not be on the team. If the team is primarily systems/engineering, forcing a math contribution risks producing a mediocre theorem that weakens the paper rather than strengthening it. It may be better to own the engineering contribution and target a systems venue.

---

## Part 3: Fatal Flaw Inventory

Flaws ranked by severity (1 = most severe):

### Flaw 1: No Grade-A Mathematical Contribution
- **Description**: The Math Spec found no component with genuinely new mathematics. All theorems are routine applications of known techniques. The best candidates (geometric consistency pruning, compiler correctness composition) are graded B−.
- **Framings affected**: All three. Each framing claims to be "the first" something, but "first application of known techniques to a new domain" is architectural novelty, not mathematical novelty.
- **Mitigable?** Partially. Pursuing the spatial tractability theorem (Part 2, Q4, Path 1) could produce Grade-A math, but it's a research bet with uncertain payoff. Alternatively, reframe as a systems contribution and target OOPSLA/UIST instead of theory venues.
- **Project killer?** Not if the venue is chosen correctly. At OOPSLA or UIST, strong engineering with formal foundations can win best paper. At POPL or LICS, this project has no chance.

### Flaw 2: Abstraction Fidelity Gap
- **Description**: All three framings abstract away physics, rendering, and animation. The Architect Framings acknowledge this: "the discrete automaton model necessarily discards physics, animation blending, and rendering-order effects." The Prior Art Audit doesn't address this because it's an evaluation flaw, not a novelty flaw. But reviewers will ask: does the tool find *real* bugs or just bugs in the *model*?
- **Framings affected**: All three, but worst for Choreo-Check (verification) and Choreo-Engine (simulation), where the claim is "find bugs before users do." If the bugs found are model artifacts that never manifest in real XR runtimes, the contribution collapses.
- **Mitigable?** Partially. Validate by extracting real interaction logic from open-source MRTK/Interaction SDK samples, running the checker, and confirming that flagged bugs correspond to known issues in those projects' bug trackers. If you can show 5+ real bugs found in production XR toolkits, the abstraction fidelity concern is addressed empirically even if not formally.
- **Project killer?** Yes, if the empirical validation fails. If the checker finds only model-level bugs that never manifest in practice, the paper is DOA.

### Flaw 3: SpatiaLang Proximity (DSL Novelty Erosion)
- **Description**: SpatiaLang already exists as a spatial interaction DSL compiled to engine code. The Prior Art Audit rates the DSL component as **Incrementally Novel**—the weakest novelty rating in the audit. The Architect Framings never mention SpatiaLang by name.
- **Framings affected**: Primarily Choreo-Lang (the DSL framing). Also affects Choreo-Check and Choreo-Engine insofar as their DSL is a contribution.
- **Mitigable?** Yes, but it requires effort. The differentiators (Event Calculus semantics, formal verification, headless execution) are real and should be foregrounded. The paper must explicitly compare against SpatiaLang and demonstrate what Choreo-Lang can express or verify that SpatiaLang cannot. A table showing "SpatiaLang cannot express temporal constraints, cannot verify deadlock freedom, cannot run headless" would defuse this concern.
- **Project killer?** No, but failure to address it invites a devastating reviewer comment: "How is this different from SpatiaLang with temporal extensions?"

### Flaw 4: No Human Studies (Evaluation Ceiling)
- **Description**: The hard constraint "no human annotation/studies" means the project cannot demonstrate developer productivity gains, usability, or adoption feasibility. The Architect Framings note this for all three framings. For a tool whose value proposition is "help XR developers," not measuring developer experience is a major gap.
- **Framings affected**: All three, but worst for Choreo-Lang (language adoption is fundamentally a human question) and Choreo-Check (tool value is measured by developer time saved).
- **Mitigable?** Partially. Automated metrics (bugs found, state-space coverage, compilation throughput, cross-platform trace fidelity) are valid proxies. For Choreo-Check, "found N real bugs in MRTK that shipped to production" is a stronger argument than any user study. For Choreo-Lang, "same interaction definition runs identically on Unity, WebXR, and visionOS" is measurable without humans. But the lack of usability evaluation will be noted by reviewers at CHI and UIST.
- **Project killer?** At CHI: yes. At OOPSLA/PLDI/IEEE VR: no (these venues accept automated evaluation).

### Flaw 5: State-Space Explosion Scalability Ceiling
- **Description**: The Math Spec (§2.4) shows the abstract state space is exponential: |S_abstract| = q^k × 2^(O(n²·m)) for k interaction patterns with q states each, n entities, and m predicate templates. The Architect Framings acknowledge that "even with spatial partitioning, complex multi-user scenes may blow up the state space."
- **Framings affected**: Primarily Choreo-Check (verification requires full exploration) and Choreo-Engine (guided search over exponential spaces). Less severe for Choreo-Lang (compilation is per-pattern, not over the product).
- **Mitigable?** Partially. Spatial independence partitioning (zones that can't physically overlap are checked compositionally) and the geometric consistency pruning (Theorem 3.2) can reduce the effective state space. But the Scope Lead estimates only ~13.5K LoC for the entire reachability checker, and the Math Spec says the pruning "needs to be worked out carefully." If the practical limit is 10–15 interaction zones, the tool is a toy.
- **Project killer?** Not if the benchmark scenes are chosen to demonstrate practical utility within the scalability limits. But claiming "scales to production XR scenes" requires evidence that production scenes typically have <20 interaction zones, which may or may not be true.

### Flaw 6: Cross-Platform Semantic Equivalence Is Unresolvable
- **Description**: The Architect Framings (Framing 2) claim that "the same REA produces identical event traces on Unity vs. WebXR." But Unity's physics triggers fire on FixedUpdate (deterministic 50Hz), WebXR's events are frame-aligned (variable rate), and visionOS gestures are asynchronous. These are fundamentally different timing models. The Architect Framings acknowledge this but understate the difficulty: "defining this equivalence formally and testing it automatically is a research contribution in itself."
- **Framings affected**: Primarily Choreo-Lang (cross-platform portability is the core value proposition) and Choreo-Engine (simulation must match real runtime behavior).
- **Mitigable?** Partially. Define equivalence "up to bounded timing skew ε" and demonstrate empirically that real interactions are insensitive to ε-bounded timing differences. But this requires knowing what ε is acceptable, which is an empirical question that may vary per application.
- **Project killer?** No, but it weakens the "write once, run everywhere" claim significantly. If the cross-platform story is "write once, run everywhere *approximately*," it's less compelling.

### Flaw 7: Engine Extraction Brittleness (Maintenance Burden)
- **Description**: The Architect Framings estimate 20–30K LoC per engine for extracting interaction protocols from Unity, Unreal, and WebXR. These scene formats are complex, version-dependent, and undocumented. The Scope Lead rates the cross-platform layer at 15% novelty—it's grunt work, not research.
- **Framings affected**: All three (all need scene input), but worst for Choreo-Check and Choreo-Engine (which claim to check *existing* interactions, requiring extraction from existing engines).
- **Mitigable?** Yes. Target only one engine (Unity, which is best-documented and most open) for the paper. Defer Unreal and WebXR to future work. This cuts 40–60K LoC of non-novel format-wrangling.
- **Project killer?** No. This is a scope management problem, not a research problem.

### Flaw 8: Biomechanical Input Model Validity (Choreo-Engine Only)
- **Description**: Choreo-Engine's stochastic exploration mode generates synthetic hand/gaze trajectories from a biomechanical model. Without validating against real human data (no human studies), the claim that these trajectories are "plausible" is unsubstantiated. The Architect Framings acknowledge: "the biomechanical model may be plausible yet unrepresentative."
- **Framings affected**: Choreo-Engine only.
- **Mitigable?** Partially. Use published biomechanical parameters (Fitts's Law coefficients, reach envelopes from ergonomics literature) and show that synthetic trajectories match the distributional statistics reported in published studies. This is a literature-validation approach, not a user study.
- **Project killer?** No, but it limits claims about "finding realistic bugs."

---

## Part 4: The Strongest Synthesis Direction

### Recommendation: Hybrid Compiler+Verifier Targeting OOPSLA

**The single strongest direction is a spatial-temporal choreography compiler with integrated verification, pitched as a new programming model for XR interaction—not as a pure verification tool, not as a standalone DSL, and not as a runtime engine.**

**Why this synthesis, and not a single framing:**

1. **Choreo-Check alone is a tool paper.** The verification math is routine (Math Spec: C to B−). The LoC is ~50K of genuine verification code. It cannot sustain a best-paper argument at a theory venue.

2. **Choreo-Lang alone is a language paper without its strongest theorem.** The spatial type system soundness proof doesn't exist yet (Math Spec doesn't state it; Scope Lead estimates 60% novelty for the type system but the Math Spec gives no theorem). Without a formal soundness guarantee, it's "SpatiaLang++" (Prior Art: incrementally novel).

3. **Choreo-Engine alone is a systems paper with no standalone novelty.** Its research contributions (formal semantics, verification) are borrowed from the compiler and verifier. Strip those away and it's a game engine subsystem.

4. **The synthesis is stronger than any part.** The Prior Art Audit's strongest novelty claims are: (a) EC→automata compilation (reversal of prior direction), (b) R-tree-backed automata transitions, (c) headless spatial choreography verification. These span all three framings. The Scope Lead's highest-novelty subsystems are: Reachability Checker (65%), Spatial-Temporal Type System (60%), Deadlock Detector (55%), Event Automata Compiler (50%). These span the compiler and verifier, not the runtime. The Architect Framings' own recommendation is "build the compiler core as the foundation, then deliver verification and simulation as applications of the compiled representation." This is the right architecture—the paper should match it.

**Specific synthesis: "Choreo: A Compiler for Spatial-Temporal Interaction Choreographies with Decidable Verification"**

- **Core contribution 1** (Language + Compiler): A spatial-temporal DSL compiled via Event Calculus semantics into R-tree-backed event automata. This subsumes Choreo-Lang's frontend with Choreo-Check's backend. (~60K LoC: parser + types + EC engine + automata compiler + incremental compilation.)

- **Core contribution 2** (Verification): Reachability and deadlock verification over the compiled automata with geometric consistency pruning. This is Choreo-Check's verification plus the Math Spec's most promising theorem (3.2). (~20K LoC: reachability + deadlock.)

- **Core contribution 3** (Headless Testing): CPU-only execution of compiled automata against synthetic scenarios, with EC-derived test oracles. This is a slimmed Choreo-Engine without the biomechanical model or guided search. (~20K LoC: runtime + simulator + test harness.)

- **Evaluation**: Extract interaction protocols from 50+ open-source XR projects (MRTK, Meta Interaction SDK, A-Frame examples). Compile, verify, and report: (a) real bugs found (deadlocks, unreachable states), (b) compilation throughput and automata size, (c) verification time vs. state-space size, (d) cross-platform trace fidelity. (~15K LoC: evaluation infrastructure.)

- **Total**: ~115–130K LoC of source code, ~38% novel. Reaches 150K+ with tests. Drops the cross-platform abstraction layer (the weakest subsystem) and the biomechanical input model (unvalidatable without human studies). Every remaining subsystem pulls its weight.

**The critical addition**: Invest 3–6 months in proving the spatial tractability theorem (Part 2, Q4, Path 1). If XR scenes with bounded spatial independence number admit polynomial-time verification, this is a genuine Grade-A theorem that elevates the paper from "strong engineering" to "engineering with a surprising theoretical insight." If the theorem doesn't work out, the paper is still a strong OOPSLA submission as a systems+PL contribution.

**Target venue**: OOPSLA 2026 (primary) or UIST 2026 (secondary). OOPSLA rewards exactly this kind of paper: a new programming model, a compiler, formal semantics, and an empirical evaluation on real-world artifacts. The comparison to Halide (PLDI 2013 → OOPSLA lineage) and TVM is apt and deliberate.

**What makes this best-paper caliber**: No one has formalized XR interaction choreography. No one has compiled it. No one has verified it. The field has standardized content (glTF, USD), rendering (Vulkan, WebGPU), and tracking (OpenXR), but *interaction* remains ad-hoc imperative code. This paper would be the first to bring the programming-languages methodology—formal semantics, compilation, static analysis, verification—to XR interaction. That's a field-defining contribution, even if the individual theorems are B-grade, because the *integration* is what matters at OOPSLA. The "bugs found in production MRTK samples" evaluation seals the practical impact argument.

**The one thing that could kill this synthesis**: If the geometric consistency pruning (Theorem 3.2) turns out to be trivial (the pruning is negligible for real scenes) AND the abstraction fidelity gap (Flaw 2) means no real bugs are found in practice. In that scenario, the paper has neither theoretical depth nor practical impact, and it's dead at any venue. This risk must be assessed empirically in the first 3 months of implementation.
