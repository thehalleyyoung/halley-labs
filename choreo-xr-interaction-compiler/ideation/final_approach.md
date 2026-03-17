# Final Approach: Choreo — Spatial CEGAR Verification with Decidable Spatial Type Checking

**Status**: Synthesized from Approaches A, B, C + expert debate + assessments  
**Date**: 2025-07-18  
**Synthesis verdict**: Approach B backbone + Approach A moonshot + Approach C compositional technique

---

## 1. Title and One-Sentence Pitch

**Choreo: A Compiler and Verifier for XR Interaction Choreographies via Geometric CEGAR and Decidable Spatial Types**

> A spatial-temporal DSL compiler that finds real interaction-protocol deadlocks in production XR frameworks using a novel geometric CEGAR refinement loop — backed by a decidable spatial type system that guarantees compiled automata are spatially realizable.

---

## 2. Approach Philosophy

**Core design choice: Verification-first with a type-theoretic moonshot.**

This approach takes Approach B's architecture (CEGAR verifier + extraction + bug-finding evaluation) as the survivable foundation, grafts Approach A's strongest mathematical contribution (decidable spatial type checking) as the moonshot that elevates the paper from a systems contribution to a potential best paper, and incorporates Approach C's compositional technique (geometric separability for assume-guarantee decomposition) as a scalability mechanism.

**What is taken from each approach and why:**

| Source | Element Taken | Why |
|--------|--------------|-----|
| **Approach B** | Spatial CEGAR verifier with GJK/EPA refinement | All experts agreed this is the most genuinely novel algorithm with zero published precedent. It has the best difficulty/risk ratio. |
| **Approach B** | MRTK extraction pipeline with gated validation | Only approach with empirical grounding in real codebases. The gate milestone provides a week-3 reality check. |
| **Approach B** | EC→automata compilation | Architecturally novel reversal. Used as an engineering choice for clean formal semantics, not claimed as a math contribution. |
| **Approach B** | Bug-finding evaluation | "If you find a real bug in MRTK, no reviewer can dismiss the paper" (Skeptic). The only evaluation strategy that produces undeniable results. |
| **Approach A** | Decidable spatial type checking (M-A1) | Math Assessor's recommended moonshot. Lower risk than M-C1 (~45% vs ~50%), more concrete proof strategy (QE threaded through coinductive subtyping), naturally integrates as a compilation phase that feeds into CEGAR verification. |
| **Approach C** | Compositional spatial separability (M-C3) | Clean, testable structural property that enables verification to scale beyond ~15 zones. Geometric separability condition has no precedent in verification literature. |

**What is explicitly rejected and why:**

| Source | Element Rejected | Why |
|--------|-----------------|-----|
| **Approach A** | Full multiparty session type infrastructure | Too much scope. The 85K LoC at 5/10 feasibility is a death sentence (Skeptic). We take the decidability *result* but implement a lighter spatial type system, not full MPST. |
| **Approach A** | Accessibility certification as primary narrative | Skeptic correctly identified this as unsubstantiated — no regulatory body requires formal proofs, no accessibility team has been consulted. Demote to "future application," not primary contribution. |
| **Approach A** | Spec-first-only evaluation | Zero empirical grounding. "You invented a type system and showed it type-checks things you wrote for it" (Skeptic). We need extraction-based evaluation. |
| **Approach C** | Reactive synthesis engine | 36 years of scaling failure. 4/10 self-assessed feasibility. Doubly-exponential complexity. We cannot bet the project on a paradigm that has never delivered practical tools. |
| **Approach C** | Spatial environment modeling (kinematic envelopes) | Unsolved biomechanics masquerading as a subroutine (Skeptic). Out of scope. |
| **All** | R-tree indexing as a research contribution | Standard infrastructure from 1984. Good engineering, not research. Stop emphasizing it. |
| **All** | EC formalization as a mathematical contribution | Rated C+ by Math Assessor, confirmed by all experts. Keep EC as IR; don't claim it as math. |

---

## 3. Extreme Value Delivered

### Who needs this

**Primary audience (realistic): XR platform QA teams and interaction framework maintainers.**

These teams at Microsoft (MRTK), Meta (Interaction SDK), and Apple (RealityKit) maintain interaction toolkits used by tens of thousands of developers. Every release risks regressions that surface only during manual headset-based playtesting. A headless verifier running in GitHub Actions CI catches protocol-level bugs (deadlocks, unreachable states, race conditions) before they reach QA.

**Secondary audience: XR interaction researchers.**

~200–400 active researchers publishing at CHI/UIST who share interaction techniques as videos and pseudocode. Choreo programs are reproducible, executable, verifiable artifacts.

**Tertiary audience: Safety-critical XR developers.**

Teams building surgical guidance, industrial maintenance AR, and training simulators where an unresponsive interaction state is hazardous. Small but high-value segment where formal guarantees justify adoption cost.

### Honest market size

The Skeptic's numbers are approximately correct:

- **XR platform teams who would directly use Choreo**: ~50–80 engineers at 4 major companies
- **XR researchers who would adopt a new tool**: ~100–200 (optimistic)
- **XR developers who would use a CI verifier**: ~500–2,000 (generous, contingent on ecosystem support)
- **Realistic total addressable users at launch**: ~50–200

This is small. We do not pretend otherwise. The value proposition is not market size — it is **filling a genuine gap in the XR toolchain** (the interaction layer has no formal tooling) and **demonstrating a novel verification technique** (spatial CEGAR) that has applications beyond XR.

### Addressing the Skeptic's demand-evidence critique

The Skeptic is right that we have zero user interviews, zero surveys, and zero evidence of demand. This is a genuine weakness. The approach mitigates it through two mechanisms:

1. **The gate milestone is a demand-evidence proxy.** If the extraction pipeline finds ≥1 real bug in MRTK that correlates with a known issue-tracker report, that is empirical evidence that the problem exists and the tool addresses it. Not a user interview, but better than speculation.

2. **Plan for ≥3 informal conversations with industry XR engineers within the first 2 months.** Not as formal user studies (which are out of scope) but as reality checks on whether the "interaction protocol bugs in CI" framing resonates. If all 3 engineers say "this doesn't solve a problem I have," trigger reassessment.

### The LLM obsolescence argument

By 2027–2028, LLMs will absorb much of the practical value of XR bug-finding at lower adoption cost. This is a valid concern. The approach addresses it directly:

**Choreo provides guarantees LLMs fundamentally cannot.** An LLM generates test cases; Choreo proves the *absence* of deadlocks. This is the difference between "we ran 10,000 tests and found no deadlocks" and "no deadlock is reachable from any spatial configuration in the declared scene constraints." The former provides confidence; the latter provides a proof. For safety-critical XR applications, this distinction matters.

**Choreo is complementary to LLMs, not competing.** The framing is: "Choreo is to LLM-generated XR tests what a type checker is to LLM-generated code — a correctness backstop that catches classes of bugs stochastic methods miss by construction." LLMs generate candidate interactions; Choreo verifies them.

**The formal-methods contribution outlasts the XR application.** Even if the XR market never materializes, the spatial CEGAR algorithm is a genuine verification contribution applicable to any domain with geometric constraints (robotics, autonomous driving, smart buildings). The math stands independently.

---

## 4. Technical Architecture

### Core abstraction

Spatial-temporal event automata — finite-state machines whose transitions are guarded by spatial predicates (containment, proximity, gaze-cone intersection) evaluated over an R-tree index, with timing constraints in bounded Metric Temporal Logic.

### Compilation pipeline

```
Choreo DSL source
    │
    ▼
[Parser + Spatial Type Checker]  ←── M-A1 decidability result (moonshot)
    │  Convex-polytope spatial subtyping
    │  Temporal well-formedness
    │  Determinism checking
    ▼
[Event Calculus IR]  ←── Engineering choice, not claimed as math
    │  Spatial oracle: σ: T → Scene
    │  Derived spatial fluents
    ▼
[Automata Compiler]  ←── Thompson-style + spatial-temporal guards
    │  On-the-fly product composition
    │  Spatial guard compilation to R-tree query plans
    ▼
[Spatial Event Automata]
    │
    ├──► [Headless Execution Engine]  ←── CPU-only, CI/CD compatible
    │       NFA token-passing + incremental R-tree guard evaluation
    │
    └──► [Spatial CEGAR Verifier]  ←── M-B3 termination (guaranteed)
            │                           M-B1 geometric pruning (guaranteed)
            │                           M-C3 compositional separability (stretch)
            ▼
         [Bug Reports]
            Deadlocks, unreachable states, race conditions
```

### What's in scope

- Spatial-temporal DSL with convex-polytope spatial predicates + bounded MTL temporal constraints
- Spatial type checker with decidability result for convex-polytope fragment
- EC IR → spatial event automata compiler
- Spatial CEGAR verifier with geometric refinement (GJK/EPA)
- Geometric consistency pruning exploiting hierarchical containment
- Compositional verification via spatial separability
- MRTK extraction pipeline for ~15 canonical component types
- Bug-finding evaluation on open-source XR projects
- Headless simulation and runtime execution engine

### What's explicitly out of scope

- **Reactive synthesis / correct-by-construction controllers** — too high risk, unscalable
- **Full multiparty session type infrastructure** — take the decidability result, not the full MPST system
- **Non-convex geometry in the decidable fragment** — bounded CSG is NP-complete; acknowledge the limitation
- **Human kinematic/biomechanical modeling** — unsolved problem, not our contribution
- **Runtime integration with Unity/Unreal** — verify the *model*, not the *runtime*
- **Physics, animation, rendering simulation** — the verifier operates on interaction protocol logic only
- **Accessibility certification as a primary contribution** — unsubstantiated demand, future work only

### Subsystem breakdown with HONEST LoC and Novel%

All novelty percentages are deflated by ~1.8× from approaches' claims per Difficulty Assessor feedback. "Novel%" means genuinely new algorithms/formalizations with no off-the-shelf implementation; it excludes domain adaptation of known techniques.

| Subsystem | Total LoC | Novel LoC | Novel% | Key Challenge |
|---|---|---|---|---|
| DSL Parser + Frontend | 8,000 | 1,500 | 19% | Spatial-temporal choreography syntax; standard recursive descent |
| Spatial Type Checker | 14,000 | 6,000 | 43% | **Spatial subtyping with LP oracle (M-A1 moonshot)**; decidability boundary for convex polytopes |
| Event Calculus IR Engine | 10,000 | 2,500 | 25% | Spatial oracle integration; fluent derivation. Known formalism, novel domain application |
| Automata Compiler | 12,000 | 4,000 | 33% | Spatial guard compilation; on-the-fly product composition with symbolic state |
| R-tree Spatial Index | 6,000 | 1,500 | 25% | Temporal parameterization, COW versioning. Mostly adapting rstar crate |
| Spatial CEGAR Verifier | 14,000 | 5,500 | 39% | **GJK/EPA geometric refinement operator (M-B3)**; BDD + BMC backends |
| Geometric Pruning Module | 5,000 | 2,500 | 50% | **Hierarchical containment exploitation (M-B1)**; layer-by-layer constraint propagation |
| Compositional Verifier | 6,000 | 2,500 | 42% | **Spatial separability decomposition (M-C3)**; assume-guarantee with geometric conditions |
| Deadlock Detector | 5,000 | 1,500 | 30% | Spatial feasibility of wait-for cycles; standard graph analysis with geometric guards |
| Runtime Execution Engine | 10,000 | 2,000 | 20% | NFA token-passing with R-tree guard evaluation; snapshot/rollback |
| Headless Simulator | 8,000 | 1,500 | 19% | CPU-only GJK/EPA collision; deterministic event generation |
| MRTK Extraction Pipeline | 8,000 | 2,000 | 25% | Roslyn C# parsing; MRTK component state machine recovery (~15 types) |
| Test Harness + Validation | 6,000 | 1,000 | 17% | Differential testing: EC oracle vs. compiled automata |
| CLI, REPL, Diagnostics | 5,000 | 500 | 10% | Counterexample trace rendering; spatial error visualization |
| Evaluation Infrastructure | 5,000 | 500 | 10% | Parametric scene generation; benchmark scripting |
| **Total** | **~122,000** | **~35,500** | **~29%** | |

**Honest summary**: ~122K total LoC (Rust core + Python evaluation tooling), of which ~35K is genuinely novel and ~25K is non-trivial domain adaptation. Including embedded unit tests and property-based generators, source files total ~150K. The intellectual core — type checker, CEGAR verifier, pruning module, compositional verifier — accounts for ~39K LoC at ~42% novelty: the research frontier.

**Note on LoC inflation**: The Difficulty Assessor found all approaches inflated novelty by ~1.8×. The figures above are post-deflation. The ~29% novel fraction is consistent with the cross-approach target of ~23-28% genuinely novel LoC. Some readers may consider ~35K generous; a harsh reading puts genuinely novel LoC at ~27K (the Skeptic's estimate). We report 35K as the upper bound and acknowledge 27K as the floor.

---

## 5. Mathematical Contributions

### Portfolio summary

| # | Result | Grade (Assessed) | Load-Bearing? | Proof Risk | Role |
|---|--------|-----------------|---------------|------------|------|
| T1 | Decidability of spatial type checking (convex-polytope fragment) | **B+ to A−** | Yes | ~45% | **Moonshot** — elevates paper from systems to theory+systems |
| T2 | Tight geometric pruning for hierarchical spatial scenes | **B to B+** | Yes | ~20% | **Guaranteed** — makes verification tractable beyond ~10 zones |
| T3 | Spatial CEGAR termination and precision | **B** | Yes | ~25% | **Guaranteed** — the novel verification algorithm's correctness |
| T4 | Compositional spatial separability | **B** | Yes | ~30% | **Stretch** — enables scaling to production-size scenes |

**If moonshot succeeds**: A−/B+, B+, B, B → average B+/A− → strong OOPSLA paper with best-paper potential (~12-18%)  
**If moonshot fails**: B+, B, B → average B/B+ → solid OOPSLA systems paper with bug-finding evaluation (~3-5% best-paper, ~65-75% strong accept)

Both tracks are publishable. The project does not depend on the moonshot.

### T1: Decidability of Spatial Type Checking (Moonshot)

**What it is.** For a spatial type system with convex-polytope spatial predicates and bounded MTL temporal constraints, subtyping is decidable. The proof reduces spatial subtyping to (1) coinductive subtyping over interaction protocol structure (known decidable), (2) LP feasibility for convex-polytope containment (polynomial), and (3) a novel spatial compatibility check via quantifier elimination (Weispfenning 1988) threaded through the coinductive algorithm. The binary session type case (2 parties) is likely straightforward; the multiparty extension carries the real risk.

**Why it's the moonshot.** This is the only result that could surprise PL theorists — showing that geometric structure (convexity, LP tractability) rescues decidability where arithmetic refinements destroy it. If the composition reveals unexpected structural obstacles, the result reaches A−. If the threading is smooth, B+.

**Why M-A1 over M-C1.** The Math Assessor recommends M-A1 because: (a) lower risk (~45% vs ~50%), (b) more concrete proof strategy, (c) the decidable fragment is more naturally scoped, (d) it integrates as a compilation phase rather than requiring an entirely separate synthesis engine. The Difficulty Assessor prefers M-C1 for its higher ceiling, but the ceiling comes with 4/10 feasibility and 36 years of scaling failure in reactive synthesis. We choose the lower-risk path that still reaches A−.

**Load-bearing?** Yes. Without decidability, the type checker may diverge on some inputs, and the "well-typed ⇒ spatially realizable" guarantee is vacuous. However, the CEGAR verifier (T3) works independently of the type system, so the bug-finding evaluation survives even if T1 fails entirely.

**Proof risk and mitigation.** ~45% failure probability. The biggest risk is that coinductive unfolding generates unboundedly many polytope constraints at each step. Mitigation cascade:
- **Best case**: Full multiparty spatial subtyping decidable in EXPTIME → A−
- **Middle case**: Binary spatial subtyping decidable → B+
- **Worst case**: Spatial type soundness for convex-polytope fragment (every guard in compiled automaton is satisfiable) without a decidability result → B (this is the original M4)

The worst case is still a publishable result that serves the system.

### T2: Tight Geometric Pruning for Hierarchical Spatial Scenes (Guaranteed)

**What it is.** The feasible spatial predicate set C ⊆ 2^P satisfies |C| ≤ O(b^d · k^d) where d is containment DAG depth, b is branching factor, k is number of proximity thresholds. For typical XR scenes (d ≤ 4, b ≤ 6, k ≤ 3), this yields |C| ≈ 10³–10⁴ versus 2^|P| ≈ 10⁶–10⁹.

**Load-bearing?** Yes. Without pruning, verification hits state-space explosion at ~10 zones.

**Proof risk.** ~20%. Layer-by-layer constraint propagation over the containment DAG is well-understood from zone graphs in timed automata. The geometric instantiation is novel but the proof structure is available. Main risk: bounds may be loose (correct but not tight), reducing intellectual interest. The *practical* pruning is unlikely to fail.

### T3: Spatial CEGAR Termination and Precision (Guaranteed)

**What it is.** The spatial CEGAR loop terminates in at most |P|·2^d refinement steps. The final abstraction is the coarsest geometric abstraction eliminating all spurious counterexamples within the BMC bound. The termination proof uses a well-founded ordering on geometric partitions — a novel argument because standard CEGAR termination (Henzinger et al. 2004) doesn't apply to geometric refinement domains.

**Load-bearing?** Yes. Without termination, the verifier may diverge. Without precision, diagnostic quality is poor.

**Proof risk.** ~25%. Risk: GJK/EPA refinement steps may produce incomparable partitions, breaking the well-founded ordering. Mitigation: impose a canonical splitting strategy (always split along the longest axis of the infeasible region), accepting some precision loss.

**This is the core novel algorithm.** GJK/EPA collision detection inside a CEGAR refinement loop has zero published precedent. Every expert endorsed this as the most genuinely novel contribution.

### T4: Compositional Spatial Separability (Stretch)

**What it is.** If k interaction controllers have geometrically separable activation zones (convex hulls pairwise disjoint or with bounded overlap), parallel composition preserves individual specifications. The proof uses a spatial variant of Abadi-Lamport assume-guarantee, with induction over a spatial decomposition tree. Geometric separability is checked in O(n log n) via convex hull intersection.

**Load-bearing?** Important for scaling but not essential. Without it, verification handles ~15-20 zones. With it, ~50-100+ zones.

**Proof risk.** ~30%. The disjoint case is likely provable. The bounded-overlap extension is harder and may require quantitative reasoning about overlap volume. If only the disjoint case works, report that honestly — many real XR scenes have spatially separated interaction zones.

### What is NOT claimed as a mathematical contribution

- **EC formalization (original M1, C+)**: Load-bearing for the compiler's correctness argument, but the math is elementary (IVT under Lipschitz). Used as engineering foundation, not claimed as a theorem.
- **Extraction soundness (original M-B2, B−/C+)**: Structural induction over 15 enumerated component types is engineering validation. Report extraction fidelity empirically (bug detection rate on known issues), not as a theorem.
- **End-to-end compiler correctness (original M5, B−)**: Important for trust but routine composition (CompCert methodology). Defer to a follow-up paper or report as engineering validation.

---

## 6. Hardest Technical Challenges

Rank-ordered by difficulty.

### Challenge 1: Extraction Fidelity from MRTK (Existential Risk)

**What makes it hard.** Recovering interaction state machines from Unity C# code that uses MonoBehaviour callbacks, coroutines, async/await, and scriptable objects is a hard program analysis problem. The Skeptic correctly identifies this as the project's existential risk.

**How to address it.** 
1. **Scope ruthlessly**: Support only ~15 canonical MRTK component types with well-defined state machine patterns (Interactable, NearInteractionGrabbable, ManipulationHandler, SolverHandler, BoundsControl, EyeTrackingTarget, etc.). Reject scenes using unsupported components.
2. **Gate milestone (weeks 1-3)**: Extract 3 MRTK sample scenes, manually verify against documented state machines. If extraction detects ≥1 known bug from MRTK's issue tracker, proceed. If >30% false positive rate, pivot to spec-first evaluation.
3. **Conservative over-approximation**: When encountering ambiguous control flow, add nondeterministic transitions. This increases false positives but eliminates false negatives.

**What happens if it fails.** Pivot to spec-first evaluation: encode ≥20 canonical MRTK interaction protocols manually in Choreo DSL, verify them, and demonstrate that the type checker catches known interaction bugs when encoded as type-level violations. The paper becomes "novel verifier demonstrated on hand-authored specifications" rather than "bug-finding tool for existing code." Weaker but publishable.

### Challenge 2: The Decidability Proof (T1) (Mathematical Risk)

**What makes it hard.** Threading quantifier elimination through coinductive subtyping is unexplored. The coinductive unfolding may generate unboundedly many polytope constraints, breaking EXPTIME termination.

**How to address it.**
1. Prove decidability for binary session types first (2 parties, convex polytopes, no temporal constraints).
2. Add temporal constraints modularly using bounded MTL decidability (Ouaknine & Worrell 2007).
3. Attempt multiparty lift via standard projection.
4. Budget 4-6 months. If no proof sketch by month 4, fall back to spatial type soundness (B grade).

**What happens if it fails.** The paper proceeds with T2+T3+T4 math portfolio (B+, B, B) plus the bug-finding evaluation. This is the "solid OOPSLA systems paper" track. The type system still provides spatial realizability guarantees (every compiled guard is satisfiable), just without the decidability novelty. Publication probability drops from ~65-75% to ~50-60%; best-paper probability drops from ~12-18% to ~3-5%.

### Challenge 3: Spatial CEGAR Performance (Engineering Risk)

**What makes it hard.** GJK/EPA is fast (~microseconds per query), but CEGAR can require many refinement iterations. The termination bound (|P|·2^d) could be large for complex scenes. If each iteration requires spatial constraint solving, the loop may be too slow for interactive use.

**How to address it.**
1. Geometric pruning (T2) reduces the abstract state space before CEGAR begins.
2. Compositional decomposition (T4) limits the effective number of predicates per sub-problem.
3. Implement incremental refinement: cache and reuse spatial partitions from previous CEGAR iterations.
4. Benchmark on 5 scenes by month 3. If pruning < 3× on any scene, investigate BDD variable ordering heuristics guided by the containment DAG.

**What happens if it fails.** The practical scalability ceiling drops to ~10 zones (below the ~15-20 target). The paper must frame this honestly: "We verify small interaction subgraphs, which is where most bugs live." If even 10-zone verification is too slow, the CEGAR algorithm is still publishable as a standalone algorithm paper at CAV/TACAS (the Skeptic's recommendation), but the XR evaluation is weaker.

### Challenge 4: Scalability Ceiling (Structural Limitation)

**What makes it hard.** Without compositional verification (T4), the practical limit is ~15-20 interaction zones. Most MRTK sample scenes fall within this range (median ~8-12 zones), but production XR applications have 50-200 zones.

**How to address it.** T4 (compositional spatial separability) is specifically designed for this. If T4 proves only the disjoint case, it still helps for many real scenes where interaction zones are spatially separated. The bounded-overlap extension is the stretch goal.

**What happens if it fails.** Report the ceiling honestly. Frame: "Most interaction *bugs* live in local neighborhoods of 5-15 zones, and our verifier covers this regime. Production-scale compositional verification is future work." Reviewers at OOPSLA will accept an honest ceiling; they will not accept an inflated one.

### Challenge 5: False Positive Rate (Practical Risk)

**What makes it hard.** Conservative over-approximation in extraction means the verifier may report deadlocks that never manifest in practice. If >50% of reports are false positives, the tool is useless.

**How to address it.** Measure false positive rate explicitly in the evaluation. Target: ≤50% false positive rate (i.e., ≥50% of reported anomalies correspond to real or plausible bugs). If exceeded, tighten extraction for the most common false-positive patterns.

**What happens if it fails.** If FP rate exceeds 50%, the bug-finding evaluation is weakened. The paper must either (a) report honestly and argue that the remaining true positives are high-value, or (b) pivot to spec-first evaluation where FP rate is zero (because the specification IS the truth).

---

## 7. Evaluation Plan

All evaluation is fully automated. No headsets, no human participants.

### Experiment 1: Bug-Finding on Real XR Projects

Extract interaction protocols from MRTK canonical interaction components across ≥30 open-source XR projects (reduced from the original 50+ to be honest about extraction scope). Compile to Choreo automata, run verifier. Report:
- Total deadlock states found
- Unreachable interaction states
- Race conditions (nondeterministic transitions on simultaneous events)
- Temporal deadline violations
- False positive rate (manual triage of top-20 reports)

Cross-reference against each project's GitHub issue tracker.

**Success threshold**: ≥5 interaction protocol anomalies found, ≥2 corroborated by existing issue tracker reports.  
**Failure threshold**: <3 anomalies, or >70% false positive rate.

**Honest caveat**: The verifier operates on an abstraction of interaction logic that omits physics, animation, and rendering. Flagged anomalies are interaction *protocol* defects, not physics or rendering bugs. Frame them as "protocol anomalies," not "bugs."

### Experiment 2: Verification Scalability

Sweep parameters: number of interaction zones (5-50), concurrent patterns (2-20), spatial predicate templates. Compare:
1. Naive explicit-state BFS
2. BDD symbolic exploration without geometric pruning
3. BDD + geometric consistency pruning (T2)
4. Spatial CEGAR (T3)
5. Compositional verification (T4, if available)

Report pruning ratio |C|/|2^P| for each benchmark scene.

**Honest scalability ceiling**: Without compositional verification: ~15-20 zones in seconds to low minutes. With compositional verification: ~50-100+ zones (target, contingent on T4). Report the observed ceiling explicitly.

### Experiment 3: Type Checker Evaluation

Encode ≥20 canonical XR interaction patterns in the Choreo type system. For each:
- Does the type checker accept well-formed patterns?
- Does the type checker reject ill-formed patterns (with useful error messages)?
- Does the type checker detect known spatial-realizability violations?

Encode ≥5 known MRTK interaction bugs as ill-typed Choreo programs. Verify that the type checker catches them. Report any patterns that are expressible in MRTK but inexpressible in the decidable fragment.

**Honest caveat**: This is a self-referential evaluation (we wrote the specs and the checker). It demonstrates expressiveness and error quality, not real-world adoption. Frame accordingly.

### Experiment 4: Compilation Throughput

Benchmark patterns compiled per second across the parametric benchmark suite (200+ patterns scaled from trivial to complex). Report compilation time breakdown by phase.

### Experiment 5: Comparison Baselines

- **iv4XR**: Agent-based exploration on the same extracted interaction graphs. Compare coverage and bugs found.
- **Manual UPPAAL encoding**: 10 representative protocols as UPPAAL timed automata (no spatial predicates). Compare verification time, expressiveness, bugs found.
- **Random simulation**: 10,000 random scenario traces per benchmark. Quantify the coverage gap between testing and verification.

### What constitutes success vs. failure

| Outcome | Classification |
|---------|---------------|
| ≥5 anomalies, ≥2 corroborated + T1 moonshot lands | **Best paper track** (~12-18% best paper at OOPSLA) |
| ≥5 anomalies, ≥2 corroborated + T1 fails | **Strong accept track** (solid OOPSLA systems paper) |
| 3-4 anomalies + T1 lands | **Moderate paper** (publishable but not distinguished) |
| <3 anomalies + T1 fails | **Pivot or abandon** — salvage spatial CEGAR as standalone CAV/TACAS paper |

---

## 8. Kill Gates and Contingency

### Gate 0: Convexity Coverage Audit (Week 1)

**Action**: Audit ≥20 MRTK interaction volumes (bounding boxes, colliders, trigger zones) from open-source sample scenes. Classify each as convex polytope, bounded-depth CSG (≤3 operations), or non-convex.

**Pass**: ≥50% of interaction volumes fall within the convex-polytope fragment (T1's decidable domain).  
**Fail**: <50% are convex polytopes.

**If fail**: T1's practical relevance is limited. Adjust the paper narrative to emphasize the CEGAR contribution (T3) over the type system (T1). Downgrade T1 from "moonshot that elevates the paper" to "theoretical contribution with restricted practical scope." The type system still provides spatial realizability guarantees for the convex fragment, but the paper leads with bug-finding + spatial CEGAR. This gate determines the paper's *framing*, not its *viability* — the CEGAR verifier works regardless of convexity coverage.

### Gate 1: Extraction Validation (Week 3)

**Action**: Extract 3 MRTK sample scenes. Run verifier. Compare against known issue-tracker bugs.

**Pass**: Extraction produces models for ≥2/3 scenes. Verifier detects ≥1 known issue-tracker bug.  
**Fail**: Extraction produces unfaithful models for ≥2/3 scenes, or verifier detects 0 known bugs.

**If fail**: Pivot to spec-first evaluation. Drop extraction pipeline. Evaluate on ≥20 hand-authored specifications. The paper becomes a "novel verifier for spatial-temporal interaction specifications" paper rather than a "bug-finding tool."

### Gate 2: Treewidth Validation (Week 2)

**Action**: Compute treewidth of ≥20 real MRTK interaction graphs.

**Pass**: Treewidth ≤ 5 for ≥80% of scenes.  
**Fail**: Treewidth > 10 for ≥50% of scenes.

**If fail**: Deprioritize compositional verification (T4). Accept ~15-20 zone scalability ceiling. Redirect effort to strengthening CEGAR (T3) and pruning (T2).

### Gate 3: Moonshot Proof Sketch (Month 4)

**Action**: Produce a proof sketch for binary spatial subtyping decidability (the simplest case of T1).

**Pass**: Binary case proof sketch is complete and convincing.  
**Fail**: Fundamental obstacle identified (e.g., coinductive unfolding generates unbounded constraints).

**If fail**: Fall back to spatial type soundness (B grade). Redirect theoretical effort to tightening T2 bounds or extending T4 to bounded overlap. The paper proceeds on Track B (systems contribution + bug-finding).

### Gate 4: CEGAR Pruning Validation (Month 3)

**Action**: Run spatial CEGAR on ≥5 benchmark scenes. Measure pruning ratio |C|/|2^P|.

**Pass**: Pruning ≥ 3× on ≥4/5 scenes.  
**Fail**: Pruning < 3× on ≥3/5 scenes.

**If fail**: Investigate BDD variable ordering and CEGAR refinement strategy. If still insufficient by month 5, the verifier's novelty claim weakens. Consider whether the CEGAR algorithm alone (without XR evaluation) is a CAV/TACAS paper.

### Gate 5: Bug-Finding Checkpoint (Month 8)

**Action**: ≥3 interaction protocol anomalies found in real XR projects, with preliminary FP triage.

**Pass**: ≥3 anomalies, FP rate ≤ 60%.  
**Fail**: <3 anomalies or FP rate > 70%.

**If fail**: Reassess the evaluation strategy. Options: (a) expand to Meta Interaction SDK projects, (b) lower threshold to "anomalies in hand-authored specs," (c) pivot to the standalone CAV/TACAS paper with CEGAR algorithm + geometric pruning, no XR evaluation.

### Fallback Paper

If all moonshots fail and bug-finding underdelivers, the minimum publishable unit is:

**"Geometric CEGAR: Counterexample-Guided Abstraction Refinement with Spatial Constraint Solving"**

A standalone verification paper at CAV or TACAS presenting:
- The spatial CEGAR refinement operator (T3)
- Geometric pruning with hierarchical containment (T2)
- Evaluation on parametric benchmarks (no XR extraction needed)

This ~15K LoC kernel has ~60% publication probability at a verification venue (Skeptic's estimate) and contributes to the formal methods community regardless of the XR application's success. It sacrifices the "first XR verifier" narrative but preserves the novel algorithm.

---

## 9. Honest Scoring

All scores use assessed (not claimed) grades, incorporating deflation from all three expert assessments.

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | **6/10** | Real CI/CD pain point for XR teams, but tiny market (~50-200 users). Bug-finding evaluation provides empirical grounding. No demand evidence beyond architectural argument. LLMs absorb some value by 2027-2028. |
| **Difficulty** | **7/10** | One genuinely novel algorithm (spatial CEGAR, zero precedent) + one moonshot theorem (spatial type decidability, ~45% risk) + multi-domain integration (PL + geometry + verification). Deflated from claimed ~8.3 average across approaches. Honest: "strong systems PhD, not multiple-breakthrough research." |
| **Potential** | **6.5/10** | If moonshot + bug-finding both deliver: 7.5-8/10. If only bug-finding delivers: 5.5-6/10. If neither: 4/10. Expected value ~6.5. Best-paper probability ~12-18% (conditional on both succeeding, ~30% joint probability). |
| **Feasibility** | **6.5/10** | B's architecture is most feasible (7/10). Adding A's moonshot lowers to ~6/10. The gate milestones and cascading fallbacks bring it back to ~6.5. Timeline: 18-20 months. ~122K LoC with ~35K novel is achievable for a strong builder. |

**Composite: 6.5/10** (equal weight)

**Comparison to approaches' self-assessments:**

| Axis | Approach A (self) | Approach B (self) | Approach C (self) | This Synthesis (honest) |
|------|-------------------|-------------------|-------------------|------------------------|
| Value | 7 | 7 | 8 | **6** |
| Difficulty | 8 | 7 | 9 | **7** |
| Potential | 8 | 6 | 9 | **6.5** |
| Feasibility | 5 | 7 | 4 | **6.5** |
| Composite | 7.0 | 6.75 | 7.5 | **6.5** |

The synthesis scores lower than every approach's self-assessment. This is by design — the experts caught inflation everywhere. A 6.5 composite with honest scores is worth more than a 7.5 with inflated ones. A reviewer who encounters honest scoring trusts the paper; a reviewer who detects inflation discounts everything.

---

## 10. Differentiation

### From existing portfolio projects

| Project | How Choreo Differs |
|---------|-------------------|
| **spatial-hash-compiler** | Spatial hashing is a data-structure compilation problem; Choreo is an interaction-protocol verification problem. Shared interest in spatial computation, but the type systems, verification algorithms, and application domains are distinct. |
| **cross-lang-verifier** | Cross-language verification targets inter-language FFI safety; Choreo targets intra-application interaction safety across spatial-temporal event automata. Different abstraction level (language boundary vs. interaction protocol). |
| **synbio-verifier** | Synthetic biology verification shares the "CEGAR for domain-specific constraints" pattern. The differentiation is the *domain of constraints*: genetic regulatory networks vs. geometric spatial predicates. The CEGAR refinement operators are fundamentally different (Boolean predicate refinement vs. GJK/EPA geometric refinement). |

### From prior art

| System | How Choreo Differs |
|--------|-------------------|
| **SpatiaLang** | SpatiaLang compiles spatial interactions to engine code but has no temporal reasoning, no formal verification, no headless execution, and no compilation to verifiable automata. Choreo compiles through EC semantics into formally verifiable automata with spatial type soundness guarantees. SpatiaLang is a code generator; Choreo is a verification-enabled compiler. |
| **MPST / Scribble** | MPST formalizes choreographic communication protocols over message channels. Choreo choreographies are spatially-grounded: guard semantics evaluate geometric predicates (containment, proximity, gaze-cone) over R-tree indices, not message-passing over typed channels. The type-theoretic machinery differs (spatial subtyping via LP feasibility, not channel subtyping). |
| **Scenic (PLDI 2019)** | Scenic targets spatial-temporal scenario *generation* for autonomous driving — synthesizing environments for testing. Choreo targets interactive multi-party choreography *verification* for XR — proving properties of interaction protocols. Different direction (generation vs. verification), different domain (driving vs. XR), different properties (scenario coverage vs. deadlock freedom). |
| **UPPAAL** | UPPAAL verifies timed automata with clock constraints but has no spatial predicates, no XR-domain integration, no geometric pruning, and requires manual model construction. Choreo automates model extraction and adds spatial-geometric reasoning to the verification loop. UPPAAL is a general-purpose tool; Choreo is a domain-specific compiler+verifier. |
| **iv4XR** | iv4XR uses agent-based exploration with runtime assertions — empirical discovery, not exhaustive verification. Choreo performs exhaustive reachability analysis with formal guarantees. iv4XR can find bugs by stumbling upon them; Choreo can prove the absence of deadlocks. |

### From LLM-based approaches

LLMs generate plausible code and test cases but cannot prove the *absence* of bugs. An LLM might generate 10,000 XR interaction test scenarios and miss the one spatial configuration that triggers a deadlock. Choreo proves that no deadlock exists for *any* spatial configuration within declared scene constraints. This is a fundamentally different capability — exhaustive proof vs. stochastic coverage.

**However**: the Skeptic is right that this distinction matters to ~0.1% of the market. The practical differentiator is CI/CD integration: Choreo runs in GitHub Actions as a static check; LLM-based testing requires simulation infrastructure. For teams that already have CI pipelines, adding a static verifier is cheaper than adding LLM-driven simulation.

**The honest framing**: Choreo is complementary to LLMs, not competing. LLMs lower the floor (making it easy to generate decent interaction code); Choreo raises the ceiling (proving the code is correct). Both are needed; neither is sufficient.

---

## 11. Why This Approach Wins the Debate

### What it takes from each approach

**From Approach B (foundation — ~70% of the architecture):**
- The spatial CEGAR verifier — universally endorsed as the most genuinely novel algorithm
- The extraction pipeline with gated validation — only path to empirical grounding
- The EC→automata compilation — solid engineering that provides clean formal semantics
- The bug-finding evaluation strategy — only evaluation that produces undeniable results
- The honest scalability reporting — "here is where we hit our ceiling" builds reviewer trust

**From Approach A (moonshot — ~20% of the architecture):**
- The decidability result for spatial type checking — elevates the paper from systems to theory+systems
- The spatial type system — provides compile-time guarantees that feed cleaner models into the verifier
- The cascading proof strategy (multiparty → binary → soundness) — ensures a publishable result even if the moonshot fails

**From Approach C (technique — ~10% of the architecture):**
- Compositional spatial separability — the scaling mechanism that extends verification beyond ~15 zones
- The geometric disjointness condition — a clean, testable structural property with zero precedent

### What it explicitly rejects

**From Approach A:**
- The accessibility certification narrative — unsubstantiated demand, no regulatory basis
- The full MPST infrastructure — too much scope for the feasibility budget
- Spec-first-only evaluation — circular without extraction-based grounding

**From Approach B:**
- Extraction soundness (M-B2) as a math contribution — it's engineering validation, not a theorem
- The original 157K LoC scope — inflated by ~2×
- The claim that B+/B/B math is sufficient for best-paper — it's not; the moonshot is needed

**From Approach C:**
- The reactive synthesis engine — 36 years of scaling failure, 4/10 feasibility
- The kinematic envelope modeling — unsolved biomechanics, not our problem
- The "correct by construction" narrative — impedance mismatch with real runtimes is too large
- The 9/10 difficulty claim — honest difficulty is 8/10, and even that's only if M-C1 lands

### How it addresses the Skeptic's hardest question

> "If you showed a working demo to the lead XR engineer at Microsoft/Meta/Apple, would they integrate it?"

**Honest answer**: Probably not immediately. But this is the wrong question. The right question is: "Would they add it to their CI pipeline if it caught a real bug their QA missed?"

If the gate milestone succeeds — if extraction + verification finds ≥1 MRTK bug corroborated by an issue-tracker report — then the answer is "maybe, for the same reason teams adopted ESLint and CBMC: it catches bugs automatically that humans miss occasionally." Not a workflow revolution; an incremental improvement to an existing workflow.

If the gate milestone fails, the honest answer is "no," and we pivot to the standalone verification contribution (spatial CEGAR at CAV/TACAS) or spec-first evaluation.

**The Skeptic's deeper critique — that the market is too small and LLMs will absorb the value — is partially valid.** We do not refute it. We mitigate it:

1. The formal guarantee narrative (exhaustive proof vs. stochastic testing) has a durable audience in safety-critical applications, even if small.
2. The spatial CEGAR algorithm is domain-independent and contributes to the verification community regardless of XR adoption.
3. The research contribution (novel verification algorithm + decidability result + real bugs found) is evaluated by publication standards, not market standards. The venue is OOPSLA, not a startup pitch.

**The Skeptic gives this approach ~35-40% survival probability.** We agree this is in the credible range. The project is risky. Every gate milestone is designed to detect failure early and preserve publishable fallbacks. The minimum publishable unit (spatial CEGAR algorithm at CAV/TACAS, ~15K LoC) is achievable within 6-8 months and has ~60% publication probability. Every month of work beyond that adds upside but does not increase the sunk cost beyond the fallback.

---

## Appendix: Timeline

| Phase | Months | Milestones |
|-------|--------|------------|
| **Foundation** | 1-3 | Gate 1 (extraction, week 3); Gate 2 (treewidth, week 2); DSL parser prototype; CEGAR refinement operator design; Gate 4 (pruning, month 3) |
| **Core Build** | 4-8 | Spatial type checker with LP oracle; EC→automata compiler; CEGAR verifier + BDD backend; Gate 3 (decidability proof sketch, month 4); Gate 5 (bug-finding checkpoint, month 8) |
| **Theory** | 4-10 | T1 decidability proof (overlaps with core build); T2 pruning bounds; T3 termination proof; T4 compositional separability |
| **Evaluation** | 9-14 | Extraction at scale (≥30 projects); bug-finding sweep; scalability benchmarks; comparison baselines |
| **Paper** | 15-18 | Writing; additional experiments; revision buffer |

**Total: 18-20 months** (Difficulty Assessor's recommended range: 18-22 months)

---

*This synthesis was produced as binding input to the implementation phase. All scores, grades, and risk estimates use assessed (not claimed) values from the Math Depth Assessor, Difficulty Assessor, and Adversarial Skeptic. Inflation has been systematically corrected. The approach is designed to be survivable — every failure mode triggers a pivot to a publishable fallback, and the minimum viable publication (spatial CEGAR at CAV/TACAS) is achievable within 8 months.*

---

## Verification Signoff

**Verifier**: Independent Verifier (not a member of the expert team)
**Date**: 2026-03-08

### Checklist Results

**1. Internal Consistency: PASS WITH CAVEATS**

Scores, LoC estimates, and risk assessments are broadly consistent between the debate and final approach, with three notable deviations:

- **LoC overshoot**: The debate consensus target (I6) was ~25K novel LoC in ~100–115K total. The final approach reports ~35K novel in ~122K total — exceeding both targets. The final approach acknowledges this ("Skeptic's estimate" floor of 27K) but doesn't explain why the upper bound drifts 40% above the debate target. This is not fatal — the 27K floor is within range — but the 35K headline number is inconsistent with the debate consensus.
- **Risk inflation for T3 and T4**: The debate assessed M-B3 (CEGAR termination) and M-C3 (spatial separability) as "Low" risk. The final approach assigns ~25% and ~30% respectively. This is more conservative than the debate's assessment. The inflation may reflect combined mathematical + engineering risk (the debate assessed pure mathematical risk), but this distinction is not stated. The increased conservatism is defensible but should be acknowledged as a deviation.
- **Moonshot selection overrides I6**: The debate's I6 explicitly recommends "M-C1 (spatial reactive synthesis decidability) — highest ceiling — with M-A1 as fallback." The final approach inverts this, choosing M-A1 as the primary moonshot and rejecting M-C1 entirely. The final approach provides justification (lower risk, more concrete proof strategy, avoids reactive synthesis scaling failure), but this is a material departure from the debate's binding recommendation. The justification is reasonable — the team correctly identified that grafting M-C1 requires an entire synthesis engine at 4/10 feasibility — but the override should be explicitly flagged as a deliberate departure from I6, not silently adopted.

**2. Grade Honesty: PASS**

All math grades in the final approach use the Math Depth Assessor's assessed grades, not the self-claimed grades from the approaches document:
- T1 (from M-A1): "B+ to A−" — matches assessed grade (claimed was A−)
- T2 (from M-B1): "B to B+" — matches assessed grade (claimed was B+)
- T3 (from M-B3): "B" — matches assessed grade (claimed was B)
- T4 (from M-C3): "B" — matches assessed grade (claimed was B)

M-B2 correctly excluded as a math contribution per the Math Assessor's B−/C+ rating and the debate ruling. EC formalization (C+) correctly excluded. The final approach does not smuggle inflated grades anywhere. This is one of the strongest aspects of the document.

**3. Skeptic Concerns Addressed: PASS WITH ONE GAP**

The final approach addresses or explicitly acknowledges every major Skeptic critique:
- Market size (~50–200 users): Acknowledged honestly, not inflated (§3)
- No demand evidence: Partially addressed via gate milestones as demand proxy + plan for 3 informal conversations (§3). However, the 3 conversations are not a kill gate — there are no pass/fail criteria, and "trigger reassessment" is vague. This is the weakest mitigation.
- MRTK extraction risk: Addressed with scope restriction (15 types) and Gate 1 (§8)
- Over-approximation false positives: Addressed with FP rate measurement (§7)
- Bug-finding as extraordinary claim: Reframed as "protocol anomalies" (§7)
- LLM obsolescence: Addressed with complementary framing (§3)
- Accessibility narrative: Correctly demoted to future work (§4)
- B-grade math insufficient for best paper: Addressed by adding T1 moonshot (§5)
- "Would an XR engineer integrate it?": Addressed honestly (§11)
- **Gap — convexity coverage not gated**: The debate's I3 explicitly requires quantifying "what fraction of real XR interaction logic falls within the convex-polytope fragment." The final approach has no early gate for this. Gate 2 measures treewidth, not convexity coverage. If <50% of real interaction volumes are convex, the decidability result (T1) is a theoretical curiosity. This empirical question was flagged as a first-month task and was dropped.

**4. Kill Gates Are Concrete: PASS**

All five gates are time-bounded, have measurable pass/fail criteria, and specify concrete failure actions:
- Gate 1 (Week 3): ≥2/3 scenes extracted, ≥1 known bug → else pivot to spec-first
- Gate 2 (Week 2): Treewidth ≤5 for ≥80% → else deprioritize T4
- Gate 3 (Month 4): Binary proof sketch complete → else fall back to B-grade
- Gate 4 (Month 3): Pruning ≥3× on ≥4/5 scenes → else investigate alternatives
- Gate 5 (Month 8): ≥3 anomalies, FP ≤60% → else reassess strategy

Gate 3's pass/fail criterion ("proof sketch complete and convincing" vs. "fundamental obstacle identified") is the most subjective — an incomplete sketch with no clear obstacle is an ambiguous state. This is an inherent difficulty with theoretical milestones and is acceptable.

**5. Fallback Is Publishable: PASS**

The minimum publishable unit — "Geometric CEGAR: Counterexample-Guided Abstraction Refinement with Spatial Constraint Solving" at CAV/TACAS — is genuinely viable. The spatial CEGAR refinement operator (T3) has zero published precedent (all experts agreed). Geometric pruning (T2) provides the scalability story. CAV/TACAS regularly accepts papers on novel CEGAR refinement operators with parametric benchmarks. The ~15K LoC kernel at ~60% publication probability is a credible estimate. The fallback does not depend on the XR narrative, extraction pipeline, or the type system moonshot — it stands on algorithmic novelty alone.

**6. Differentiation Is Clear: PASS WITH CAVEAT**

Differentiation from prior art (SpatiaLang, MPST, Scenic, UPPAAL, iv4XR) is well-articulated with a concrete feature comparison table. Each system is distinguished on a specific technical axis, not vague hand-waving.

Portfolio differentiation has one soft spot: **synbio-verifier** shares the "CEGAR for domain-specific constraints" pattern. The final approach acknowledges this and distinguishes by refinement domain (Boolean predicates vs. GJK/EPA geometric refinement), which is valid — the refinement operators are fundamentally different algorithms. However, a skeptical portfolio reviewer may still see "yet another domain-specific CEGAR verifier." The differentiation is adequate but should be sharpened during paper writing.

**7. Feasibility Check: PASS (TIGHT)**

18–20 months for ~122K LoC with ~35K novel:
- ~6.1K LoC/month total, ~1.75K novel LoC/month — achievable for a strong builder
- However, months 4–10 overlap core build + theory + proofs. If theory absorbs 6 months of focused time, the remaining 12–14 months must produce ~122K LoC of implementation + evaluation — ~8.7–10K LoC/month, which is aggressive
- The 18-month lower bound is at the aggressive end of the Difficulty Assessor's 18–22 month recommendation
- The cascading fallbacks provide pressure relief — if T1 or T4 fail, effort redirects from theory to engineering, easing the timeline

Verdict: feasible but with minimal slack. A 20-month plan is more realistic than 18. Any setback beyond those anticipated by the kill gates (e.g., Rust toolchain issues, BDD library integration surprises) could push to 22+ months.

**8. Math Portfolio Is Load-Bearing: PASS WITH NUANCE**

- **T2 (geometric pruning)**: Genuinely load-bearing — without it, verification hits state-space explosion at ~10 zones, rendering the tool impractical for any real scene.
- **T3 (CEGAR termination)**: Genuinely load-bearing — without it, the core verification algorithm has no termination guarantee. This is the heartbeat of the system.
- **T4 (compositional separability)**: Load-bearing for production scale (~50+ zones) but the system functions without it on smaller scenes (~15–20 zones). More accurately "load-enhancing" than "load-bearing."
- **T1 (decidable spatial types)**: The final approach is honest that the CEGAR verifier works independently of T1. T1 is load-bearing for the type system subsystem but not for the overall artifact. Calling it "load-bearing" in the portfolio table (§5) while simultaneously saying "bug-finding evaluation survives even if T1 fails entirely" is a mild tension. T1 is better described as "load-bearing for the type system; enhancement for the overall system."

No result is ornamental — every claimed math result directly enables or significantly enhances a system component. The portfolio is lean and functional.

**9. Evaluation Is Falsifiable: PASS**

The evaluation has explicit success/failure thresholds that could produce negative results:
- Bug-finding: ≥5 anomalies with ≥2 corroborated (success) vs. <3 anomalies or >70% FP (failure)
- Scalability: explicit ceiling reporting with honest caveats
- Type checker: acknowledged as self-referential ("we wrote the specs and the checker")
- Classification matrix maps outcomes to paper quality (best-paper → pivot/abandon)
- The "pivot or abandon" row explicitly exists — the team has committed to a falsification threshold

The type checker evaluation (Experiment 3) is the weakest — self-referential by admission — but this is noted and framed appropriately.

**10. No Fatal Flaw Missed: PASS**

I find no missed fatal flaw that would kill the project. The team has correctly identified the two existential risks (extraction fidelity, decidability proof) and built kill gates around both. Potential concerns not rising to "fatal":

- **Convexity coverage gap** (raised above): If most real XR interaction volumes are non-convex, the decidable fragment is a theoretical curiosity. This is an empirical question the debate flagged but the final approach doesn't gate on. Not fatal because the CEGAR verifier doesn't depend on convexity — but it weakens the T1 contribution's practical relevance.
- **No proof mechanization**: For a PL/verification venue, pen-and-paper proofs of novel type system properties (T1) may face reviewer skepticism. Lean/Coq mechanization was mentioned in the problem statement for M5 but dropped. Not fatal, but a reviewer risk.
- **Industry validation is aspirational**: The 3 informal conversations have no binding criteria. If all 3 engineers say "this doesn't solve my problem," the plan says "trigger reassessment" with no defined action. Not fatal because the research contribution stands on technical merit at OOPSLA, but the practical impact narrative remains speculative.

### Issues Found

1. **[BINDING] Missing convexity coverage gate**: Add a Gate 0 or fold into Gate 1: "Audit ≥20 MRTK interaction volumes and report fraction within the convex-polytope fragment. If <50%, flag T1's practical relevance as limited and adjust the paper narrative to emphasize the CEGAR contribution over the type system." This was explicitly required by the debate (I3) and dropped.

2. **[ADVISORY] Moonshot override should be explicit**: Section 5's explanation of "Why M-A1 over M-C1" is good, but should explicitly note this departs from the debate's I6 recommendation. A single sentence — "The debate recommended M-C1 as the primary moonshot; we override this because [reasons already given]" — would close the loop.

3. **[ADVISORY] Risk inflation for T3/T4 should be explained**: Note that the ~25%/~30% risk figures for T3/T4 represent combined mathematical + engineering risk, versus the debate's "Low" mathematical risk assessment. Currently the discrepancy is silent.

4. **[ADVISORY] Strengthen demand validation**: The ≥3 informal conversations should have pass/fail criteria (e.g., "If 0/3 engineers identify interaction protocol bugs as a top-5 pain point, reassess the CI/CD framing and lead with the formal-methods-contribution narrative instead").

5. **[ADVISORY] LoC headline should use conservative estimate**: Report ~27–35K novel LoC rather than ~35K as the point estimate, consistent with the acknowledged Skeptic floor.

### Final Verdict

**APPROVE WITH CONDITIONS**

**Binding condition**: Add a convexity coverage gate (Issue #1 above) before implementation begins. The debate explicitly required this (I3), and the final approach dropped it. This is a first-week empirical check that could significantly alter the T1 narrative.

**Non-binding recommendations**: Address Issues #2–5 during the first week of implementation. None are blockers, but all strengthen the document's internal coherence and the project's intellectual honesty.

**Overall assessment**: This is a well-constructed synthesis that correctly selects the survivable B-backbone architecture with a justified moonshot, uses assessed (not claimed) grades throughout, builds cascading fallbacks with concrete gates, and honestly acknowledges its small market and speculative demand. The team's systematic inflation-correction is the strongest aspect — the 6.5 composite score with honest grades is more trustworthy than any approach's self-assessed 7.0–7.5. The spatial CEGAR algorithm (T3) alone justifies the project as a verification contribution; everything above that is upside. The project is ready for implementation pending the binding condition.