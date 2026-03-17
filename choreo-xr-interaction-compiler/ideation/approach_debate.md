# Approach Debate: Choreo XR Interaction Compiler

**Date:** 2025-07-18  
**Participants:** Math Depth Assessor · Difficulty Assessor · Adversarial Skeptic · Team Lead (moderator)  
**Status:** Complete — feeds into final synthesis

---

## 1. Debate Summary

Three expert reviewers subjected Approaches A (Spatial Session Types), B (Geometric CEGAR), and C (Reactive Synthesis) to adversarial scrutiny across mathematical depth, engineering difficulty, and real-world viability. The debate surfaced a central tension that no approach resolves: all three share a pattern of *one genuinely novel contribution surrounded by well-understood engineering*, yet each claims a portfolio that inflates difficulty and math grades by 1–2 notches. The Math Depth Assessor finds no guaranteed A-grade result in any approach; the Difficulty Assessor deflates claimed difficulty ratings by 1–2.5 points across the board and identifies ~2× LoC inflation in every proposal; the Adversarial Skeptic argues all three may be solving a problem nobody has — with an addressable market of ~50–200 users and an LLM obsolescence horizon of 2027–2028. The debate produces no winner, but yields a rank ordering: Approach C has the highest ceiling and highest risk, Approach A has the best type-theoretic narrative but the worst feasibility, and Approach B is the most honest and most survivable but has the weakest math. The Skeptic's forced choice is B at ~35–40% survival probability, with the actual recommendation to strip the spatial CEGAR algorithm and publish it as a domain-independent verification contribution at CAV/TACAS — abandoning the XR framing entirely.

---

## 2. Per-Approach Debate

### 2.1 Approach A: Spatial Refinement Session Types ("Choreo-Types")

#### Math Assessment (Math Depth Assessor)

| Result | Claimed | Assessed | Risk | Notes |
|--------|---------|----------|------|-------|
| M-A1: Decidability of Spatial Session Subtyping | A− | B+ to A− | ~45% proof failure | Domain transfer: quantifier elimination over polytopes lifted through session type structure. Each ingredient is known; novelty is the combination. Not a genuinely new technique — it is a (nontrivial) engineering of existing decision procedures into a new index domain. |
| M-A2: Modality-Parametric Session Soundness | B+ | B | Low | Standard parametric reasoning applied to a new index domain. Modality refinement is a routine preservation argument — reviewers will recognize this as known technique. |
| M-A3: Spatial Projection Completeness | B | B | Low | Honest self-assessment. Minkowski sum for choreographic endpoint projection is genuinely novel application. Clean, solid, unglamorous. |

**Portfolio verdict:** B+ average. If M-A1 delivers (55% probability), portfolio is B+/A−. If M-A1 fails, collapses to B/B average — indistinguishable from a competent course project in session types. The entire approach lives or dies on one theorem.

#### Difficulty Assessment (Difficulty Assessor)

| Metric | Claimed | Honest |
|--------|---------|--------|
| Difficulty rating | 8/10 | 6.5/10 |
| Total LoC | ~44K | ~22K genuinely novel |
| LoC inflation factor | — | 2× |
| Genuinely novel subproblems | Multiple | 1 (spatial session subtyping) |
| Consultant replicability | — | Moderate — session types + computational geometry cross-expertise is rare but not unprecedented |

The Difficulty Assessor notes that only spatial session subtyping is genuinely novel. Modality parametricity and endpoint projection are standard PL techniques applied to new domain. The claimed difficulty of 8/10 assumes the decidability proof is hard, but the *engineering* around it is routine compiler construction.

#### Adversarial Critique (Skeptic)

1. **Accessibility compliance is fantasy.** The pitch positions Choreo-Types as enabling formal accessibility proofs ("well-typed ⇒ accessible"). But accessibility regulations (WCAG, ADA Section 508, EN 301 549) do not require or recognize formal proofs. No accessibility standards body has requested this. No XR accessibility team has been interviewed. The compliance narrative is aspirational worldbuilding, not a real value proposition.

2. **Decidability for convex polytopes serves nobody.** M-A1 restricts to convex polytopes, but real XR interaction volumes are non-convex: L-shaped rooms, furniture-occluded zones, hand meshes. A decidability result that applies only to convex polytopes is a theoretical curiosity, not a practical tool. The moment you need a non-convex region, you fall outside the decidable fragment.

3. **Spec-first means zero empirical validation.** Choreo-Types requires developers to *write specifications from scratch*. No extraction, no connection to existing codebases. This means the evaluation will consist entirely of hand-authored examples — which proves the compiler works on specs the authors wrote, not that it applies to real XR interaction logic.

4. **85K LoC + 5/10 feasibility = death sentence.** The approach claims the largest codebase with the lowest feasibility score of any approach. This is a multi-year effort with a coin-flip on the central theorem. If M-A1 fails at month 12, the entire investment is stranded.

#### Defense

1. **On accessibility:** The defense is weaker than it appears. The strongest counter-argument is *not* that regulations require formal proofs today, but that formal guarantees create *new capabilities* — e.g., proving that every interactive state is reachable via eye-tracking alone, which no existing tool can verify. This is forward-looking research, not compliance tooling. However, the Skeptic is correct that without a single interview or letter of support, this remains speculative.

2. **On convex polytopes:** Convex polytopes are not as restrictive as the Skeptic implies. Real XR interaction regions are typically defined as axis-aligned bounding boxes, spheres, and convex hulls in all major frameworks (MRTK InteractionVolume, Meta SDK CollisionShape). Non-convex cases are decomposed into convex sub-volumes at the framework level. The decidable fragment genuinely covers the majority of *programmatic* interaction volumes, even if physical spaces are non-convex. That said, the proposal should explicitly quantify what fraction of MRTK interaction volumes are convex.

3. **On spec-first:** Spec-first is a *feature*, not a bug — the entire session types paradigm is spec-first, and this community accepts that. The valid critique is not "no extraction" but "no evidence the spec language is expressive enough to capture real protocols." The defense should commit to encoding ≥10 real MRTK interaction protocols in Choreo-Types to demonstrate coverage.

4. **On feasibility:** Genuine weakness. No strong defense exists. The approach is high-risk-high-reward by design. The only mitigation is defining a clear M-A1 failure contingency that pivots to the B-grade portfolio (M-A2 + M-A3) as a PL workshop paper.

#### Verdict

| Dimension | Score | Notes |
|-----------|-------|-------|
| Math ceiling | A− | Highest *type-theoretic* ceiling, but only if M-A1 delivers |
| Math floor | B | If M-A1 fails, portfolio is indistinguishable from routine PL work |
| Difficulty (honest) | 6.5/10 | One genuinely novel subproblem; rest is known technique |
| Feasibility | 5/10 | Lowest of all approaches. Multi-year commitment with coin-flip outcome. |
| Practical impact | Low | Spec-first with no extraction means no path to finding bugs in existing code |
| Narrative strength | High | "Well-typed ⇒ deadlock-free AND spatially accessible" is a compelling one-liner |
| Survival probability | ~25–30% | Per Skeptic. Conditional on M-A1 success AND strong narrative framing. |

**Strengths:** Best theoretical narrative. Type-system approach integrates into developer workflow. Accessibility angle, if substantiated, provides unique societal impact story. M-A3 (Minkowski sum for projection) is clean novel result regardless.

**Weaknesses:** Worst feasibility. Entire approach collapses if one theorem fails. No empirical connection to real XR code. LoC most inflated (2×). Accessibility claim unsubstantiated.

---

### 2.2 Approach B: Geometric CEGAR Bug Hunter ("Choreo-CEGAR")

#### Math Assessment (Math Depth Assessor)

| Result | Claimed | Assessed | Risk | Notes |
|--------|---------|----------|------|-------|
| M-B1: Tight Geometric Pruning | B+ | B to B+ | ~20% | Zone graphs adapted to spatial containment hierarchies. Solid but incremental — exploiting structure in predicate space is standard in model checking. |
| M-B2: Soundness of Conservative Extraction | B | **B−/C+** | Low | **Most inflated result across all approaches.** Structural induction over 15 enumerated MRTK component types is engineering validation, not mathematics. "Soundness" here means "we checked all cases" — this is a lookup table, not a theorem. |
| M-B3: Spatial CEGAR Termination | B | B | Low | Honest. Geometric refinement domain for CEGAR termination *is* genuinely novel — no prior CEGAR loop uses collision detection primitives (GJK/EPA) for abstraction refinement. |

**Portfolio verdict:** B average. No standout result. The strength of this approach is systems/empirical, not mathematical. M-B2 is the weakest claimed contribution across all three approaches — the Math Assessor specifically flags it as "not mathematics." If the paper is positioned as a theory contribution, it dies. If positioned as a systems contribution with M-B3 as the novel algorithm, it survives.

#### Difficulty Assessment (Difficulty Assessor)

| Metric | Claimed | Honest |
|--------|---------|--------|
| Difficulty rating | 7/10 | 6/10 |
| Total LoC | ~40K | ~25.5K genuinely novel |
| LoC inflation factor | — | 1.7× (lowest inflation) |
| Genuinely novel subproblems | Multiple | 1 (spatial CEGAR operator) |
| Consultant replicability | — | **High** — most reproducible approach; well-defined algorithms, clear specification |

The Difficulty Assessor notes Approach B as the *most honestly self-assessed* across all dimensions. The 1.7× inflation factor is the smallest. The spatial CEGAR operator is the sole genuinely novel algorithm, but it is clearly defined and has a clean boundary with the surrounding engineering. A strong verification engineer could replicate the novel contribution in 6–8 months.

#### Adversarial Critique (Skeptic)

1. **MRTK extraction is a pipe dream.** Recovering state machines from Unity C# is itself a research problem — unsound in general, and MRTK uses Unity-specific patterns (coroutines, MonoBehaviour lifecycles, ScriptableObject event channels) that defeat standard static analysis. The extraction pipeline is ~8–12K LoC of custom tooling that may not produce faithful models. If extraction is unfaithful, every "bug found" is a model artifact, not a real bug.

2. **Conservative over-approximation means every result is a false positive.** M-B2 claims "soundness" — but soundness of over-approximation means: if the tool reports a deadlock, the deadlock is *possible in the model*, not that it *occurs in reality*. Every result requires manual triage. With real XR timing, physics, animation, and frame ordering abstracted away, the gap between "model says possible" and "actually happens" may be vast.

3. **Finding bugs in MRTK is an extraordinary claim.** Microsoft has maintained MRTK for 5+ years with professional QA, unit tests, integration tests, and a public issue tracker. Claiming that an academic tool finds bugs Microsoft missed requires extraordinary evidence. The more likely outcome: the tool finds model artifacts that don't correspond to real bugs, or it rediscovers already-known issues.

4. **B+/B/B math isn't publishable as theory.** At PLDI, POPL, or CAV, the math portfolio is below threshold. At OOPSLA or UIST, the paper survives on systems contribution — but "best paper" requires either surprising theory or overwhelming empirical impact. A B-average math portfolio with 5 bugs found in MRTK is a solid paper, not a best paper.

#### Defense

1. **On extraction:** The Skeptic's critique of *general* extraction from Unity C# is valid, but Approach B explicitly scopes to 15 canonical MRTK interactable component types. These are well-documented, have stable APIs, and follow a known state machine pattern (Idle → Focus → Select → Activated). Targeted extraction from ~15 types with known schemas is fundamentally different from arbitrary program analysis. The defense should commit to a concrete extraction gate: extract 3 scenes, validate against ≥3 known issue-tracker bugs before proceeding.

2. **On over-approximation:** Over-approximation is standard in model checking — SLAM, BLAST, and CEGAR all work this way. The critique that "every result is a false positive" mischaracterizes how verification tools are used in practice. The relevant metric is *false positive rate*, which the proposal commits to measuring. If the tool finds 20 potential deadlocks and 5 correspond to known issues, that's a 25% true positive rate — standard for research verification tools and useful in practice.

3. **On MRTK bugs:** The claim is better framed as "protocol anomalies" rather than "bugs." MRTK has known interaction protocol issues documented in its GitHub issue tracker (e.g., race conditions between near and far interaction, focus lock during teleportation). The tool doesn't need to find *unknown* bugs — demonstrating that it automatically detects *known* issues validates the approach. Finding even 1–2 *previously unknown* anomalies would be a bonus.

4. **On math grade:** Conceded. Approach B is not a theory paper. The defense is that OOPSLA and UIST reward systems contributions with novel algorithms. M-B3 (spatial CEGAR with GJK/EPA refinement) has zero published precedent. It is a genuine algorithmic contribution even if it's B-grade math, and its novelty is in the *domain* (geometric abstraction refinement), not the *technique* (CEGAR).

#### Verdict

| Dimension | Score | Notes |
|-----------|-------|-------|
| Math ceiling | B+ | No path to A-grade. Ceiling is a solid B+ with M-B1 + M-B3. |
| Math floor | B−/C+ | If extraction fails, M-B2 is the only "result" and it's barely math |
| Difficulty (honest) | 6/10 | Most honestly assessed. One genuinely novel algorithm. |
| Feasibility | 7/10 | Highest feasibility. Clear milestones, well-defined failure modes. |
| Practical impact | **High** | Only approach that connects to existing XR codebases and finds real bugs |
| Narrative strength | Medium | "We found bugs in MRTK" is compelling but not field-defining |
| Survival probability | ~35–40% | Per Skeptic. Highest of all approaches. |

**Strengths:** Most feasible. Most honest self-assessment. Only approach with empirical grounding in real codebases. Spatial CEGAR operator is genuine algorithmic novelty with zero precedent. Highest consultant replicability (clean boundary between novel and engineering). Best match for OOPSLA/UIST venue expectations.

**Weaknesses:** Weakest math (B average). Extraction fidelity is existential risk. "Bug-finding in professionally QA'd framework" is extraordinary claim requiring extraordinary evidence. No A-grade theorem possible. Over-approximation semantics mean manual triage burden.

---

### 2.3 Approach C: Spatial-Temporal Reactive Synthesis ("Choreo-Synth")

#### Math Assessment (Math Depth Assessor)

| Result | Claimed | Assessed | Risk | Notes |
|--------|---------|----------|------|-------|
| M-C1: Decidability of Spatial Reactive Synthesis | A− | B+ to A− | **~50%** | Higher ceiling than M-A1 because reactive synthesis is a harder baseline problem. But the meta-technique — using geometric finiteness to reduce continuous problems to finite games — is known from hybrid systems (Henzinger, Alur). The contribution is the *specific* reduction for spatial predicates, not the reduction strategy itself. |
| M-C2: Spatial Environment Abstraction Soundness | B+ | B | Low | Standard simulation relation argument for hybrid systems. Polyhedral kinematic envelope is the new ingredient, but the proof technique is textbook. |
| M-C3: Assume-Guarantee Spatial Compositionality | B | B | Low | Honest. Geometric separability for assume-guarantee decomposition is genuinely novel — spatial disjointness as a compositional principle has no precedent in reactive synthesis literature. |

**Portfolio verdict:** B+ average. Highest ceiling of all three approaches — if M-C1 delivers, this is the only approach that could produce a genuinely field-defining result. But also highest risk: 50% probability of M-C1 failure, at which point the portfolio drops to B/B and the "synthesis" narrative collapses (you can't claim "correct by construction" without the decidability result).

#### Difficulty Assessment (Difficulty Assessor)

| Metric | Claimed | Honest |
|--------|---------|--------|
| Difficulty rating | 9/10 | **8/10** |
| Total LoC | ~49K | ~26.2K genuinely novel |
| LoC inflation factor | — | 1.9× |
| Genuinely novel subproblems | Multiple | M-C1 decidability genuinely justifies high difficulty rating |
| Consultant replicability | — | **Low** — reactive synthesis + computational geometry + XR domain = extreme cross-expertise requirement |

The Difficulty Assessor confirms Approach C is genuinely the hardest. M-C1 alone justifies an 8/10 honest difficulty rating — the decidability proof requires simultaneously reasoning about game-theoretic synthesis, polyhedral geometry, and temporal logic. The 1-point deflation (from 9 to 8) reflects that M-C2 and M-C3, while solid, use standard techniques. Consultant replicability is the lowest of all approaches: very few researchers have the cross-domain expertise to replicate this work.

#### Adversarial Critique (Skeptic)

1. **Reactive synthesis has never scaled in 36 years.** Church's problem was stated in 1957. Reactive synthesis was solved (Büchi-Landweber, 1969) and has been refined for decades (Pnueli-Rosner, Piterman-Pnueli-Sa'ar). Yet the technique has *never* been adopted in industrial practice for systems of meaningful complexity. The doubly-exponential complexity of M-C1 is not a theoretical curiosity — it means the approach cannot handle specifications beyond ~10–15 spatial predicates. Every "reactive synthesis for X" paper shows 5–10 toy examples. This will be no different.

2. **"Correct by construction" has impedance mismatch with real systems.** Synthesized controllers must interface with real XR runtimes (Unity, Unreal, WebXR) that do not obey the formal model. Frame timing, physics simulation, rendering pipeline, and input latency introduce behaviors the synthesis cannot anticipate. The synthesized controller is "correct" with respect to the formal model but may behave incorrectly in the real system — the same false-precision problem the Skeptic identifies in Approach B, but worse because users *trust* the output as provably correct.

3. **Spatial environment modeling is unsolved biomechanics.** M-C2 claims "controller is correct for all physically realizable user behaviors" given a polyhedral kinematic envelope. But modeling the kinematic envelope of a human user — range of motion, seated vs. standing, disability accommodations — is itself an unsolved problem in biomechanics. The approach assumes a critical input (the envelope) that doesn't exist and is extremely hard to obtain accurately.

4. **Nobody wants synthesized controllers.** XR designers are artists. They tune "feel" — the haptic feedback curve, the dwell-time threshold, the snap-to behavior. A synthesized controller that is "correct" but feels wrong is useless. The approach solves a problem XR developers don't have (formal correctness) while ignoring the problem they do have (tunability, expressiveness, designer control).

#### Defense

1. **On scalability:** The 36-year scalability critique is the strongest attack and has no clean refutation. The strongest counter-argument is that XR interaction specifications are *structurally simple* compared to general reactive systems — typically 5–15 spatial predicates with local interaction patterns. The doubly-exponential worst case may not be reached in practice. M-C3's compositional decomposition is specifically designed to address scalability: decompose into independent spatial zones, synthesize each separately. But the defense must honestly commit to benchmarking scalability limits.

2. **On impedance mismatch:** Valid concern, but this is a concern for *all* formal methods applied to real systems, not specific to synthesis. The counter-argument is that Choreo's Event Calculus IR already abstracts away physics/rendering, and the synthesized controller operates at the *event* level, not the frame level. The controller says "when gaze enters region AND reach detected, activate menu" — it does not control physics simulation. The gap between event-level specification and frame-level execution exists but is the same gap all approaches share.

3. **On kinematic envelopes:** Partially conceded. Full biomechanical modeling is out of scope. The defense is that the approach uses *conservative* envelopes — bounding boxes derived from OpenXR tracking space specifications (e.g., seated play area: 1m × 1m × 1.5m). These are coarse but sound: the synthesized controller is correct for all behaviors within the envelope, even if the envelope is an over-approximation of what humans actually do. The approach trades precision for soundness.

4. **On "nobody wants this":** The Skeptic's strongest argument. The best counter is that reactive synthesis targets a *different user* than the Skeptic imagines: not XR designers tuning feel, but safety-critical XR applications (surgical guidance, industrial maintenance AR, accessibility-constrained interfaces) where formal guarantees matter. This narrows the market even further, but it's a market where correctness is genuinely valued. The defense must honestly acknowledge the narrow audience.

#### Verdict

| Dimension | Score | Notes |
|-----------|-------|-------|
| Math ceiling | **A−** | Highest of all approaches, if M-C1 delivers |
| Math floor | B | If M-C1 fails, B/B portfolio — "synthesis" narrative collapses entirely |
| Difficulty (honest) | **8/10** | Highest genuine difficulty. Cross-domain expertise requirement is extreme. |
| Feasibility | **4/10** | Lowest of all approaches. 50% coin-flip on central theorem. |
| Practical impact | Low–Medium | Correct-by-construction is valuable in principle but narrow audience |
| Narrative strength | **Highest** | "Synthesize correct XR controllers from specs" is field-defining if it works |
| Survival probability | ~20–25% | Per Skeptic. Highest ceiling, lowest floor. |

**Strengths:** Highest mathematical ceiling. Only approach that could produce a genuinely field-defining result. M-C1 decidability, if proven, establishes a new subfield (spatial reactive synthesis). Completely sidesteps extraction problem. Compositional decomposition (M-C3) with geometric separability is novel. Lowest consultant replicability = highest moat. Directly addresses LLM-obsolescence concern (synthesis provides guarantees LLMs cannot).

**Weaknesses:** Worst feasibility. 36 years of scaling failure in reactive synthesis. Impedance mismatch between formal model and real XR runtimes. Kinematic envelope is an unsolved input. Narrow audience (safety-critical XR only). If M-C1 fails, the entire "synthesis" narrative is dead and the remaining portfolio is weaker than B's.

---

## 3. Cross-Cutting Debate

### 3.1 The Market Problem

**Skeptic's claim:** The addressable market for formal XR interaction verification is ~50–200 researchers and maybe ~5,000 specialized developers globally. No evidence of demand exists — zero user interviews, zero surveys, zero industry letters of support.

**Math Assessor's response:** Market size is irrelevant to mathematical contribution. A field-defining theorem (M-A1 or M-C1) stands on its own regardless of how many people use the tool.

**Difficulty Assessor's response:** The difficulty is genuine regardless of market. But the Assessor concedes that a 25K-LoC novel system with 50 users is harder to justify at an applied venue (OOPSLA, UIST) than at a theory venue (POPL, LICS).

**Unresolved:** The Skeptic is correct that no demand evidence exists. The Math Assessor is correct that mathematical novelty transcends market size. The question is which *venue* is targeted: at POPL, market doesn't matter; at OOPSLA, it matters enormously. This tension is unresolved and must be addressed in the synthesis.

### 3.2 The LLM Obsolescence Horizon

**Skeptic's claim:** By 2027–2028, LLM-based test generation and code review will absorb the practical value of headless XR testing and bug-finding. The verification guarantees that distinguish Choreo from LLM-based approaches are valued by ~0.1% of the market.

**Defense across all approaches:** Formal guarantees are *complementary* to LLM-generated tests, not substitutes. LLMs generate test cases; Choreo proves properties exhaustively. An LLM cannot prove the *absence* of deadlocks. Approach C (synthesis) is the strongest defense: synthesized correct-by-construction controllers are a capability LLMs fundamentally cannot provide.

**Unresolved:** The Skeptic's underlying point stands — even if formal methods are complementary to LLMs, the *perceived need* for formal guarantees may be absorbed by "good enough" LLM-generated testing. The research value is not threatened (formal methods are published for their theoretical contribution), but the practical impact narrative weakens.

### 3.3 The "One Novel Contribution" Pattern

**Difficulty Assessor's finding:** "All three share a pattern: one genuinely novel contribution surrounded by well-understood engineering."

- Approach A: Spatial session subtyping is novel. Everything else is standard.
- Approach B: Spatial CEGAR operator is novel. Everything else is standard.
- Approach C: M-C1 decidability is novel. Everything else is standard.

**Implication:** The winning approach should *center* its one novel contribution and be honest about the engineering. Don't claim 3 novel results when you have 1. A paper with 1 clean novel result + solid engineering is stronger than a paper claiming 3 results where 2 are routine.

### 3.4 The LoC Inflation Problem

**Difficulty Assessor's finding:** All three approaches inflate LoC by 1.7×–2×.

| Approach | Claimed LoC | Genuinely Novel LoC | Inflation |
|----------|------------|---------------------|-----------|
| A | ~44K | ~22K | 2.0× |
| B | ~40K | ~25.5K | 1.7× |
| C | ~49K | ~26.2K | 1.9× |

**Cross-approach target:** ~25K genuinely novel LoC in ~100–115K total system. 18–22 months timeline.

### 3.5 The Event Calculus and R-tree Distractions

**Skeptic's claim:** Event Calculus (EC) formalization and R-tree emphasis are padding. EC is a known formalism (Kowalski & Sergot, 1986) being applied to a new domain — this is domain adaptation, not research. R-tree indexing is standard computational geometry infrastructure.

**Math Assessor's response:** Partially agrees. EC formalization (M1 in original proposal) was rated C+. R-tree automata are infrastructure. Neither contributes to mathematical depth.

**Difficulty Assessor's response:** EC→automata compilation reversal is genuinely novel *engineering* (~50% novel LoC in that component), even if the math is routine. R-tree emphasis is pure padding.

**Consensus:** EC is a reasonable *engineering choice* but should not be claimed as a research contribution. R-tree is infrastructure — remove from contributions list.

### 3.6 The Extraction vs. Specification Dilemma

All three approaches must obtain formal models of XR interaction logic. They differ fundamentally in how:

- **Approach A (spec-first):** Developer writes specifications. No extraction. Clean formally but ungrounded empirically.
- **Approach B (extraction-first):** Automatically extracts from existing MRTK code. Empirically grounded but extraction fidelity is existential risk.
- **Approach C (synthesis-first):** Developer writes specifications, system synthesizes controllers. Same spec-first problem as A, but with a stronger payoff if it works.

**The dilemma:** Extraction (B) provides empirical grounding but may be unfaithful. Specification (A, C) provides formal cleanliness but may be ungrounded. No approach solves both problems.

### 3.7 The Convexity Constraint

Both A and C restrict their decidability results to convex polytopes. The Skeptic's critique that "real XR uses non-convex geometry" applies to both.

**Status:** Partially addressed by the defense (frameworks already decompose into convex sub-volumes), but no approach quantifies what fraction of real XR interaction volumes fall within the decidable fragment. This is an empirical question that should be answered early.

---

## 4. Teammate Disagreements

### Disagreement 1: Is M-B2 a mathematical contribution?

- **Math Depth Assessor:** **No.** M-B2 is rated B−/C+ — "structural induction over 15 enumerated cases is not mathematics." It is the most inflated result across all three approaches.
- **Difficulty Assessor:** Implicitly treats it as engineering validation, consistent with the Math Assessor.
- **Skeptic:** Agrees M-B2 is not math, but goes further: even calling it "soundness" is misleading. It's a completeness argument over a lookup table.
- **Approach B proponents:** M-B2 establishes a formal guarantee that extraction is an over-approximation for the 15 supported component types. Even if the proof technique is simple, the *result* matters for users.
- **Ruling:** M-B2 should be reframed as an "engineering validation" or "coverage guarantee," not a theorem. It does not contribute to the math portfolio.

### Disagreement 2: Can Approach C scale?

- **Math Depth Assessor:** Agnostic — the decidability result (M-C1) is mathematically valid regardless of practical scalability. The doubly-exponential complexity is *intrinsic* to the problem.
- **Difficulty Assessor:** Rates C's difficulty at 8/10, implicitly acknowledging the scaling challenge is genuinely hard.
- **Skeptic:** **Absolutely not.** 36 years of reactive synthesis have produced zero industrial deployments. Doubly-exponential complexity with ~10⁴–10⁶ arena states is already optimistic; real specifications will exceed this. "Scalable synthesis" is an oxymoron.
- **Defense:** Compositional decomposition (M-C3) is specifically designed to address this. Spatial separability allows independent per-zone synthesis.
- **Ruling:** Unresolved. The scalability question can only be answered empirically. The synthesis should require early benchmarking of arena sizes on realistic XR specifications.

### Disagreement 3: How much does the market matter?

- **Math Depth Assessor:** Market size is irrelevant. A decidability theorem stands on mathematical merit.
- **Difficulty Assessor:** Market matters for applied venues. 25K novel LoC with 50 users is a hard sell at OOPSLA.
- **Skeptic:** Market is *everything*. Without demand, the system is an academic exercise. "First system to X" is necessary but not sufficient — you must show someone *wants* X.
- **Ruling:** Depends on venue strategy. For POPL/LICS: market irrelevant. For OOPSLA/UIST: must demonstrate either (a) bugs found in real systems, or (b) developer interest via interviews/surveys. The synthesis must commit to a venue and address market accordingly.

### Disagreement 4: Is Approach A's accessibility narrative legitimate?

- **Math Depth Assessor:** The mathematical content (modality-parametric soundness, M-A2) is real but routine. The *framing* as accessibility is narrative packaging around a standard preservation theorem.
- **Difficulty Assessor:** No opinion on narrative legitimacy; rates the underlying math at B.
- **Skeptic:** **Illegitimate.** Accessibility compliance is governed by regulations and standards bodies, not type systems. No one in the accessibility community has asked for formal proofs of spatial reachability. This is "ethics-washing" a type theory paper.
- **Defense:** The capability is real — proving that every interactive state is reachable via alternative input modalities is a genuine computation. Whether the accessibility community *wants* it is a separate question from whether the tool *provides* it.
- **Ruling:** The capability is real but the narrative is unsubstantiated. If Approach A is selected, the accessibility angle must be validated (≥2 interviews with XR accessibility researchers or developers) or demoted from a primary contribution to a "future application."

### Disagreement 5: What is the honest difficulty target?

- **Math Depth Assessor:** Primarily concerned with proof difficulty, not engineering difficulty. Rates M-C1 as genuinely hard (50% failure risk) and M-A1 as moderately hard (45% failure risk). Silent on engineering.
- **Difficulty Assessor:** Honest difficulty is 6–8/10 across approaches. Target 6.5–7.5/10 for the final synthesis. ~25K genuinely novel LoC.
- **Skeptic:** Claims real difficulty is lower than the Difficulty Assessor states, because most "novel" LoC is domain adaptation of known algorithms. True novelty is <15K LoC across any approach.
- **Ruling:** The Difficulty Assessor's methodology is the most rigorous (they decompose LoC into novel/adapted/standard with explicit criteria). Accept 6.5–7.5/10 as the honest target. The Skeptic's <15K claim is too aggressive — it conflates "domain adaptation" with "standard engineering" when the adaptation itself requires novel algorithmic thinking (e.g., GJK/EPA inside a CEGAR loop is not trivially adapting CEGAR).

---

## 5. Consensus Points

Despite sharp disagreements, all three experts converge on the following:

### C1: No approach has a guaranteed A-grade mathematical result.

All three assessors agree. The Math Assessor explicitly states this. The Difficulty Assessor's deflation of difficulty ratings is consistent. The Skeptic argues that without A-grade math, no approach justifies a top-venue theory paper. The best available path is M-C1 or M-A1, both at ~45–50% proof risk.

### C2: Every approach has exactly one genuinely novel contribution.

- A: Spatial session subtyping
- B: Spatial CEGAR with geometric refinement
- C: Spatial reactive synthesis decidability

Everything else in every approach is engineering (good engineering, sometimes novel engineering, but not novel math). The winning approach should be honest about this.

### C3: LoC claims are inflated by ~2× across the board.

The genuinely novel LoC is ~22–26K in each approach, embedded in ~40–49K of claimed LoC. The honest system size is ~100–115K total with ~25K genuinely novel.

### C4: Approach B is the most survivable; Approach C has the highest ceiling.

The Skeptic rates B at ~35–40% survival and C at ~20–25%. The Math Assessor rates C's ceiling highest (A−) and B's ceiling lowest (B+). The Difficulty Assessor rates B as most replicable and C as least. All agree on this relative ordering.

### C5: Empirical validation gates should come before theoretical investment.

All experts agree that early empirical gates — extraction validation (B), convexity coverage quantification (A, C), scalability benchmarking (C) — should precede heavy theoretical investment. Don't spend 12 months on M-C1 before verifying the approach can handle 10 real XR specifications.

### C6: The Event Calculus formalization and R-tree indexing are not research contributions.

EC is a known formalism applied to a new domain (C+). R-tree is standard infrastructure. Neither should appear in a contributions list.

### C7: The target venue determines the strategy.

Theory venues (POPL, LICS, CAV) need A-grade math. Systems venues (OOPSLA, UIST) need engineering novelty + empirical impact. The approaches are not equally suited to all venues: A targets POPL/LICS, B targets OOPSLA/UIST, C targets either (if M-C1 delivers) or neither (if M-C1 fails). All experts agree the venue choice is a binding strategic decision.

---

## 6. Implications for Synthesis

The debate produces clear constraints on what the winning approach — or the synthesized hybrid — must address:

### I1: The synthesis must contain exactly one high-risk/high-reward theorem and 2–3 solid B-grade results.

No approach's portfolio is publishable as-is. The synthesis should pick ONE moonshot decidability result (M-A1 or M-C1) and back it with safe, clean B-grade results (M-B3's CEGAR termination, M-A3's projection completeness, M-C3's spatial separability). The Math Assessor's recommended portfolio: (1) CEGAR termination from B, (2) geometric pruning from B, (3) spatial separability from C, (4) one moonshot from A or C.

### I2: The synthesis must have an empirical grounding that Approaches A and C lack.

The Skeptic's attacks on A and C both target the lack of empirical connection to real code. The synthesis *must* include either extraction (from B) or a commitment to encoding ≥10 real MRTK interaction protocols. "We wrote specs and verified them" is not sufficient for an applied venue.

### I3: The synthesis must honestly scope the decidable/tractable fragment.

Both decidability results (M-A1, M-C1) restrict to convex polytopes with bounded temporal operators. The synthesis must quantify what fraction of real XR interaction logic falls within this fragment. If the answer is <50%, the decidability result is a theoretical curiosity. If >80%, it's practically relevant. This is an empirical question that must be answered in the first month.

### I4: The synthesis must define failure contingencies.

Given ~45–50% risk on the moonshot theorem:
- **Track A (moonshot succeeds):** Theory paper at POPL/LICS + systems paper at OOPSLA. Field-defining contribution.
- **Track B (moonshot fails):** Systems paper at OOPSLA with B-grade math portfolio (CEGAR termination + pruning + separability) + strong bug-finding evaluation. Strong paper, not best paper.

Both tracks must be viable. The synthesis cannot be a plan that only works if the moonshot lands.

### I5: The synthesis must address the Skeptic's hardest question.

> "If you showed a working demo to the lead XR engineer at Microsoft/Meta/Apple, would they integrate it?"

Until this question has a credible answer — supported by at least informal conversations with industry practitioners — the practical impact narrative remains speculative. The synthesis should include a plan for ≥3 informal industry conversations in the first 2 months, not as formal user studies (which violate the project constraints), but as reality checks on the problem framing.

### I6: The honest target profile for the synthesis is:

| Dimension | Target |
|-----------|--------|
| Difficulty | 6.5–7.5/10 honest |
| Genuinely novel LoC | ~25K in ~100–115K total |
| Math portfolio | 1× A−/B+ moonshot + 3× B-grade solid results |
| Timeline | 18–22 months |
| Primary venue | OOPSLA (systems + PL) or CAV/TACAS (verification, if stripped to CEGAR) |
| Survival probability | ~35–45% for best-paper; ~65–75% for solid publication |
| Moonshot | M-C1 (spatial reactive synthesis decidability) — highest ceiling — with M-A1 (spatial session subtyping) as fallback |
| Safe base | M-B3 (CEGAR termination) + M-C3 (spatial separability) + geometric pruning from B |
| Empirical base | ≥5 protocol anomalies in MRTK, ≥2 corroborated by issue tracker |

---

*End of debate. The synthesis phase should use this document as binding input to the final approach design.*
