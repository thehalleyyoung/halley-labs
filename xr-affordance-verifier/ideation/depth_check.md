# Depth Check: xr-affordance-verifier

## Panel: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer

**Project:** Parametric Accessibility Verification for Mixed Reality Scenes via Pose-Guarded Hybrid Automata
**Phase:** crystallize → verification
**Date:** Final consensus document

---

### Summary

This document constitutes the binding depth-check assessment for the xr-affordance-verifier project, produced by a three-expert panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer). The project proposes a formal verification framework for spatial accessibility in mixed reality (XR) scenes, built around pose-guarded hybrid automata (PGHA) and a tiered verification architecture spanning interval arithmetic (Tier 1), BDD-based zone reachability (Tier 2), and CEGAR-driven full model checking (Tier 3). The mathematical core — particularly the SE(3) zone abstraction (M2) and the bounded-treewidth compositional decomposition (M3b) — represents genuinely novel contributions at the intersection of formal methods, computational algebraic geometry, and spatial computing. No prior system attempts formal verification of kinematic reachability predicates over parameterized human body models in XR interaction graphs.

However, the panel finds the project carries exceptionally high compound risk (~85–90% probability of at least one fatal flaw), driven primarily by unvalidated computational feasibility of cylindrical algebraic decomposition (CAD) at problem-relevant dimensions (7–10 DOF), zero evidence of developer demand, and the absence of any prototype code. The claimed 175–210K lines of code is inflated; the genuinely novel algorithmic core is estimated at 85–110K LoC, with the remainder constituting important but non-novel engineering (scene parsers, visualization, benchmarks, DSL tooling). The "8 mathematical contributions" framing overstates the work: 2 contributions are strong (M2, M3b at A-grade), 1 is moderate (M3a at B+), and 5 are incremental (M1, M4–M7 at B to B-grade), sufficient for one solid publication but not the field-defining contribution the project claims.

The panel recommends **CONDITIONAL CONTINUE** with seven binding amendments (A1–A7) and a four-stage kill-chain that ensures early detection and termination at each major risk gate. The worst-case outcome (CAD proves infeasible at scale) still yields a publishable theory contribution at CAV or TACAS. The best-case outcome yields a novel, practically useful verification tool for XR accessibility — a genuine first in the literature. The composite score is 20/40 (V4/D7/BP4/L5), reflecting strong technical novelty offset by uncertain value and constrained publication prospects.

---

### Axis 1: Extreme and Obvious Value — 4/10

#### The Problem Is Real but Narrow

Spatial accessibility in XR environments is a genuinely under-served problem. When a developer places an interactive button at coordinates that a wheelchair user cannot physically reach, or designs a grab interaction requiring a range of motion that excludes users with limited shoulder mobility, the resulting application silently excludes a population. There is no existing tool — static or dynamic — that checks XR scenes against parameterized kinematic models of diverse human bodies. The status quo is manual testing with a small number of testers, or no testing at all. This is a real gap.

However, the *magnitude* of this gap must be assessed honestly:

#### Population and Market Size

- The global XR developer population is estimated at 300–500K individuals, concentrated in gaming, enterprise training, and social platforms.
- The subset building accessibility-critical applications (healthcare, education, public-facing enterprise) is a fraction — conservatively 30–50K developers.
- The subset of *those* developers who would adopt a formal-methods-based tool with a DSL requirement (as opposed to a simple linting plugin) is smaller still.
- The "1.3 billion people with disabilities" framing used in the project proposal is misleading. The relevant intersection is people with *motor* disabilities who also *use XR headsets*. Current XR headset penetration is ~30–40M units globally; the motor-disability subset of headset owners is vanishingly small today, likely in the tens of thousands.

#### Regulatory Landscape

- The EU Accessibility Act (EAA, effective June 2025) mandates accessibility for digital products and services, but does **not** explicitly name XR or mixed reality. The Act targets web content, mobile applications, and e-commerce — areas with mature accessibility standards (WCAG 2.1/2.2).
- XR-specific accessibility standards do not exist in any major regulatory framework. The W3C XR Accessibility group has produced requirements documents but no testable conformance criteria.
- Framing the tool as regulatory-compliance-driven is therefore speculative. No XR developer today faces legal liability for spatial inaccessibility in the way web developers face WCAG litigation.

#### The Simpler-Alternative Problem

This is the panel's central value concern. A Monte Carlo approach — sampling 10,000–100,000 body parameterizations from anthropometric distributions and checking reachability via forward kinematics — can be implemented in ~2,000 lines of code, runs in seconds on a laptop, and would catch >90% of real spatial accessibility bugs (buttons placed behind walls, grab targets above 95th-percentile reach envelopes, interaction zones below wheelchair-seated eye height).

The formal verification tiers (2–3) provide *provable* guarantees — they can certify that *no* parameterization in the space violates the accessibility predicate. This is mathematically elegant and practically valuable for safety-critical applications (medical XR, industrial training). But the marginal detection rate over Monte Carlo sampling is likely small (estimated 5–15% additional bugs caught), and the computational cost is 100–1,000× higher. For the vast majority of XR developers, the simpler tool would suffice.

#### No Evidence of Developer Demand

- Zero developer surveys have been conducted.
- Zero developer interviews are documented.
- No feature requests from XR platform holders (Meta, Apple, Microsoft, Qualcomm) reference formal accessibility verification.
- No XR developer forum threads, blog posts, or conference talks identify formal verification of spatial accessibility as a desired capability.
- The DSL requirement (developers must specify accessibility predicates in a domain-specific language) is a significant adoption barrier for a population that primarily works in visual editors (Unity, Unreal, A-Frame).

#### What Would Raise This Score

- Evidence of platform-holder interest: If Meta's Accessibility team or Apple's Accessibility engineering group expressed interest in formal verification tooling, the value proposition changes entirely.
- XR-specific accessibility litigation: A lawsuit or regulatory action specifically targeting spatial inaccessibility in XR would create immediate demand.
- Reframing as "accessibility linter": If Tier 1 (interval arithmetic, seconds-fast) were packaged as a Unity/Unreal plugin with zero DSL requirement and one-click operation, the adoption barrier drops dramatically. This is the highest-value pivot the project could make.
- Demonstrated Monte Carlo insufficiency: If the evaluation shows specific, important bug classes that Monte Carlo reliably misses (e.g., narrow reachability gaps at specific joint configurations), the value of formal guarantees increases.

**Score: 4/10.** The problem is real, the solution is novel, but the addressable market is small, simpler alternatives exist, regulatory urgency is speculative, and no demand signal has been validated.

---

### Axis 2: Genuine Difficulty as Software Artifact — 7/10

#### Novel LoC Analysis

The project claims 175–210K total lines of code. The panel's independent assessment, based on subsystem-level analysis of algorithmic novelty versus engineering integration, estimates the genuinely novel core at **85–110K LoC**, with the remainder being important but non-novel engineering:

| Subsystem | Claimed LoC | Novel LoC (est.) | Novelty Assessment |
|-----------|-------------|-------------------|-------------------|
| Zone Abstraction Engine | 25,000 | 18,000–22,000 | **High.** CAD adapted for SE(3) manifold structure. No off-the-shelf library computes cylindrical algebraic decomposition over Lie group configuration spaces. The chart-based decomposition of SO(3) into coordinate patches, combined with cell complex construction in each chart, is without precedent in the CAD literature. |
| Model Checker Core | 28,000 | 15,000–20,000 | **High.** Zone-graph encoding into BDD variables with CEGAR bridge. While BDD model checking is mature (NuSMV, CUDD), the zone-graph abstraction layer and the domain-specific variable ordering heuristics for spatial predicates are novel. |
| PGHA Constructor | 22,000 | 15,000–18,000 | **High.** Lazy product-space construction with semialgebraic guard predicates. The PGHA formalism itself is new — hybrid automata with guards expressed as semialgebraic sets over SE(3)-valued configuration variables. |
| Compositional Engine | 15,000 | 8,000–12,000 | **Moderate-High.** Spatial assume-guarantee reasoning with automated interface decomposition. The compositional verification principle (verify subsystems independently, compose guarantees) is well-known, but the spatial decomposition strategy (exploiting bounded treewidth of interaction graphs) is novel. |
| Counterexample Engine | 12,000 | 7,000–9,000 | **Moderate-High.** CEGAR concretization in SE(3). The core CEGAR loop is standard, but concretizing abstract counterexamples as physical body configurations in SE(3) — and checking feasibility against joint limits and self-collision — is novel. |
| Scene Parsers (Unity, WebXR, OpenXR) | 20,000 | 3,000–5,000 | **Low.** Standard AST traversal and scene graph extraction. Unity's serialization format is well-documented. The novelty is in the state machine extraction heuristics, not the parsing. |
| Kinematic Model Library | 15,000 | 2,000–4,000 | **Low.** Human kinematic chain modeling is mature (Pinocchio, RBDL, Drake). The novelty is in the parameterization interface (anthropometric distributions → kinematic parameters), not the kinematics itself. |
| DSL & Frontend | 12,000 | 3,000–5,000 | **Low-Moderate.** Language design is non-trivial but not algorithmically novel. |
| Visualization | 10,000 | 1,000–2,000 | **Low.** Standard 3D rendering of counterexamples. |
| Benchmarks & Evaluation | 10,000 | 2,000–3,000 | **Low.** Procedural scene generation and measurement infrastructure. |
| Cross-device Module | 8,000 | 2,000–3,000 | **Low-Moderate.** Device-specific interaction models. |
| **Total** | **177,000** | **76,000–103,000** | |

Rounding and accounting for integration complexity (which is itself non-trivial), the panel estimates **85–110K genuinely novel LoC**.

#### Why This Is Hard

The core difficulty lies in three algorithmic challenges that have no known solutions:

1. **SE(3) Zone Abstraction (M2).** Cylindrical algebraic decomposition (CAD) is a fundamental tool in real algebraic geometry, but it operates over Euclidean spaces (ℝⁿ). The configuration space of a human arm is SE(3) — or more precisely, a product of SO(2) and SO(3) factors constrained by joint limits. Constructing a zone abstraction (finite partition of the configuration space into cells where accessibility predicates are uniform) requires either: (a) embedding SE(3) into ℝⁿ via charts and handling chart transitions, or (b) developing an intrinsic CAD for Lie groups. Option (a) is the proposed approach, and it is genuinely novel — no published algorithm handles the chart-transition soundness problem for CAD on manifolds.

2. **Bounded-Treewidth Exploitation (M3b).** The compositional verification strategy depends on the observation that XR interaction graphs (where nodes are interactable objects and edges represent spatial proximity or sequential dependencies) have bounded treewidth in practice. If treewidth is bounded by k, the model checking problem decomposes into O(n) subproblems of size O(exp(k)), yielding tractability. This is a well-known technique in graph algorithms, but its application to spatial accessibility verification — and the proof that it preserves soundness of the zone abstraction — is novel.

3. **PGHA Semantics.** Pose-guarded hybrid automata combine continuous dynamics (body motion along kinematic chains) with discrete transitions (interaction state changes: idle → reaching → grasping → manipulating). The guard conditions are semialgebraic sets in SE(3). The semantics of this formalism — particularly the interplay between continuous reachability (can the body reach a configuration?) and discrete reachability (can the interaction state machine reach a target state?) — require careful formalization. No existing hybrid automaton framework handles SE(3)-valued continuous state with semialgebraic guards.

#### Library Reuse

Significant components can leverage existing libraries:

- **QEPCAD** (or QEPCAD-B): Quantifier elimination over the reals via CAD. Core algorithmic substrate for zone computation.
- **CUDD**: BDD manipulation library. Core substrate for model checker.
- **Pinocchio**: Rigid-body dynamics and kinematics. Core substrate for kinematic model.
- **Z3/CVC5**: SMT solvers for feasibility checking in CEGAR loop.
- **ANTLR/tree-sitter**: Parser generation for DSL.

These libraries reduce from-scratch LoC substantially, but the *integration* novelty is real. Making QEPCAD operate on chart-decomposed SE(3) manifolds, or encoding zone graphs into CUDD BDDs with domain-specific variable ordering, requires deep understanding of both the libraries and the problem domain.

**Score: 7/10.** The novel algorithmic core is substantial (85–110K LoC), the SE(3) zone abstraction is a genuinely hard open problem, and no existing tool or library solves this. The score is not higher because ~40% of the claimed LoC is standard engineering, and the difficulty is concentrated in a few key algorithms rather than distributed across the system.

---

### Axis 3: Best-Paper Potential — 4/10

#### Venue Analysis

The project's interdisciplinary nature — spanning formal methods, computational geometry, XR systems, and accessibility — is simultaneously its greatest strength and its greatest publication obstacle. No single venue naturally accommodates this work.

**CAV / TACAS (Formal Methods, A-tier)**

- **Strengths:** M2 (SE(3) zone abstraction, A-grade) and M3b (bounded-treewidth compositional decomposition, A-grade) are the project's strongest contributions and align well with CAV's interest in novel verification techniques for new domains.
- **Weaknesses:** CAV wants *depth* — a single, sharp contribution with a tight proof and convincing experiments. The project's breadth (8 claimed mathematical contributions) works against it. Reviewers will ask: "Which of these is the real contribution?" and penalize the paper for trying to do too much.
- **Conditional assessment:** A focused paper on M2 + M3b, with proofs and experiments on zone abstraction scalability, has a plausible path to acceptance. But two conditions are currently unmet: (1) the chart-transition soundness problem for M2 is flagged as unresolved, and (2) the bounded-treewidth assumption for M3b is unvalidated on real scenes. If either fails, the paper collapses.
- **Probability of acceptance (if submitted):** 25–35% (conditional on resolving chart transitions and validating treewidth).

**IEEE VR / ISMAR (XR Systems, A-tier for the field)**

- **Strengths:** The application domain (XR accessibility) is timely, and these venues welcome systems contributions.
- **Weaknesses:** Papers at IEEE VR and ISMAR are overwhelmingly empirical — user studies with 20–40 participants, quantitative performance evaluations on real applications, qualitative assessments of user experience. A formal methods paper with zero user studies, no real XR application evaluations, and procedural-only benchmarks will be mis-assigned to reviewers who lack formal methods expertise, or desk-rejected for methodological mismatch.
- **Probability of acceptance (as currently scoped):** 5–10%.

**CHI / ASSETS (HCI / Accessibility, A-tier)**

- **Fatal issue:** Zero user studies. ASSETS in particular — the premier accessibility venue — requires genuine engagement with people with disabilities. A paper that claims to "formally verify accessibility" without involving a single person with a disability in its design, evaluation, or validation will be viewed as tone-deaf at best and appropriative at worst. The CHI/ASSETS community has explicit norms around participatory design and "nothing about us without us."
- **Probability of acceptance:** <5%.

**UIST (User Interface Software and Technology, A-tier)**

- **Possible as tool paper.** UIST accepts tool papers that demonstrate novel interaction techniques or developer tools with clear impact. A submission focusing on Tier 1 + Tier 2 as a practical "accessibility linter" for XR, with a DSL demo, case studies on real scenes, and a developer evaluation (even a small one, n=10–15), could be competitive.
- **Weaknesses:** UIST wants demonstrated developer impact, not just procedural benchmarks. The paper would need to show that real XR developers find the tool useful, which requires the demand validation (A7) and real scene evaluation (A4).
- **Probability of acceptance (if properly scoped):** 15–25%.

**OOPSLA / ASE (Software Engineering, A-tier)**

- **Possible for tool paper.** These venues accept verification tools for new domains (e.g., smart contract verification at OOPSLA). A paper on the full Tier 1–2 pipeline with DSL, focusing on the software engineering novelty (DSL design, state machine extraction, tiered verification architecture), could work.
- **Probability of acceptance (if properly scoped):** 15–20%.

#### Mathematical Contributions Assessment

| ID | Contribution | Grade | Assessment |
|----|-------------|-------|------------|
| M1 | PGHA formalism definition | B | Clean formalization but incremental over existing hybrid automata literature. The SE(3) guards are the novelty, but the formalism itself is a straightforward extension. |
| M2 | SE(3) zone abstraction via chart-decomposed CAD | A- | Genuinely novel. No published algorithm computes CAD over Lie group configuration spaces. The chart-transition soundness problem is the key open challenge. |
| M3a | Kinematic locality (k=4 effective dimension) | B+ | Useful observation that kinematic chains induce spatial locality, reducing effective CAD dimension. But the "k=4" claim is optimistic — real effective dimension is 7–10 when joint DOFs are counted. |
| M3b | Bounded-treewidth compositional decomposition | A- | Novel application of treewidth-based decomposition to spatial verification. Strong theoretical contribution if the bounded-treewidth assumption holds on real scenes. |
| M4 | Zone-graph BDD encoding | B | Standard BDD encoding technique adapted to zone graphs. The variable ordering heuristic is the novelty, but it's an engineering contribution, not a theoretical one. |
| M5 | CEGAR for SE(3) zone refinement | B | Standard CEGAR loop adapted to zone abstraction. The concretization step (checking feasibility of abstract counterexamples in SE(3)) is non-trivial but not a major theoretical advance. |
| M6 | Spatial assume-guarantee rule | B- | Compositional verification rule for spatial predicates. Straightforward application of assume-guarantee reasoning with spatial decomposition. |
| M7 | Counterexample animation/reconstruction | B- | Engineering contribution. Reconstructing physical body configurations from abstract counterexamples is useful but not algorithmically deep. |

**Summary:** 2 strong contributions (M2, M3b), 1 moderate (M3a), 5 incremental (M1, M4–M7). This is sufficient material for **1 solid paper** (theory: M2+M3b at CAV) and **1 good tool paper** (Tier 1+2 at UIST/OOPSLA/ASE). It is **not** sufficient for the "8 mathematical contributions defining a new subfield" framing, which will invite hostility from reviewers.

#### Publication Strategy

The only viable path is a **two-paper strategy**:

1. **Theory paper (CAV/TACAS):** M2 (SE(3) zone abstraction) + M3b (bounded-treewidth decomposition). Focused proofs, scalability experiments on zone computation, comparison with naive CAD. No XR framing beyond motivation — present as "verification of kinematic reachability on Lie groups."
2. **Tool paper (UIST/OOPSLA/ASE):** Tier 1 + Tier 2 pipeline. DSL, case studies on real scenes, developer feedback (even preliminary). MC baseline comparison. Present as "first accessibility linter for XR."

The "defines a new subfield" framing must be dropped entirely. It is a red flag for reviewers — it signals overclaiming and invites hostile scrutiny. Let the work speak for itself.

**Score: 4/10.** Two strong mathematical contributions (M2, M3b) but no natural venue home, zero user studies (fatal at HCI/accessibility venues), and the multi-contribution breadth works against CAV's depth preference. A two-paper strategy is viable but not guaranteed. The "new subfield" framing actively hurts.

---

### Axis 4: Laptop CPU + No Humans — 5/10

#### Tier-by-Tier Computational Feasibility

**Tier 1: Interval Arithmetic — FEASIBLE (High Confidence)**

- Interval arithmetic over kinematic chains is well-understood and computationally cheap.
- For a scene with N interaction elements, Tier 1 computes a bounding box of the reachable workspace for each body parameterization interval and checks intersection with each element's activation volume.
- Complexity: O(N × J) interval evaluations, where J is the number of joints (~20–30 for a full-body model).
- Runtime: Milliseconds to seconds for scenes up to 1,000 objects.
- Memory: Negligible (<100MB).
- **No concerns.** This tier is solidly laptop-feasible and requires no extraordinary engineering.

**Tier 2: BDD-Based Zone Reachability — FEASIBLE WITH CAVEATS (Moderate Confidence)**

- The zone abstraction partitions the configuration space into cells; the zone graph captures adjacency and transitions. BDD model checking operates on a Boolean encoding of zone-graph states.
- For a scene with N=30 objects and ~100 zones per object, the zone graph has ~3,000 nodes. The BDD encoding requires ~12 Boolean variables per zone (log₂ 3,000 ≈ 12), plus state bits for the interaction state machines.
- At this scale, BDD model checking (using CUDD) is comfortably feasible on a laptop: seconds to minutes, <1GB memory.
- **Scaling concern at N=50:** The claimed target of 50 objects with 16GB memory is unsubstantiated. If zone counts grow (e.g., 500–1,000 zones per object for complex geometry), the BDD encoding may require 50,000+ Boolean variables. BDD size without domain-specific variable ordering is unknowable a priori — it could be 10MB or 10GB depending on the structure.
- **Variable ordering:** The project proposes domain-specific variable ordering heuristics (spatial clustering, kinematic chain proximity). These are reasonable but unvalidated. Poor variable ordering can cause exponential BDD blowup.
- **Verdict:** Feasible for moderate scenes (~30 objects) with high confidence. Feasible for large scenes (~50 objects) only if zone counts are controlled and variable ordering is effective — this requires prototype validation.

**Tier 3: CEGAR Full Model Checking — ASPIRATIONAL (Low Confidence)**

- CEGAR operates by iteratively refining the zone abstraction until either a real counterexample is found or the property is verified.
- Each refinement step requires recomputing CAD in the refined region, re-encoding into BDDs, and re-checking.
- The claimed "4 hours for 50-object scene" target has **zero prototype measurements** supporting it. This is entirely aspirational.
- **CEGAR convergence:** The number of refinement iterations is unbounded in the worst case. For well-structured problems (where the initial abstraction is close to sufficient), CEGAR may converge in 5–10 iterations. For adversarial or complex geometry, it may never converge within a time budget.
- **Individual refinement cost:** Each CAD recomputation in the refined region has complexity doubly exponential in the number of variables. Even with the locality argument (M3a, effective dimension 4), the real effective dimension is 7–10 (each joint contributes 1–3 DOF). CAD at dimension 7–10 is at the absolute frontier of computational feasibility.
- **Verdict:** Tier 3 feasibility is the project's highest technical risk. Without prototype measurements, the panel cannot assess whether Tier 3 is feasible on real scenes within any reasonable time budget.

#### The CAD Dimension Problem

This is the single most important computational concern, and it deserves detailed analysis.

Cylindrical algebraic decomposition has complexity doubly exponential in the number of variables: for n variables and polynomials of degree d, the number of cells is O((2d)^(2^n)). The project's key insight (M3a) is that kinematic locality reduces the effective number of variables: a button-press reachability predicate depends only on the joints in the kinematic chain from shoulder to fingertip, not on the entire body.

The claim is that this reduces the effective dimension to k=4. But this is optimistic:

- A human arm kinematic chain from shoulder to fingertip has **7 DOF** (3 shoulder, 1 elbow, 3 wrist). If the predicate involves hand orientation (not just position), all 7 DOF are relevant.
- With joint limits (semialgebraic constraints on each DOF), the effective polynomial degree increases.
- Even at k=7, CAD complexity is O((2d)^128) — astronomical unless the polynomials have special structure.
- The ONLY hope for tractability is that kinematic chain polynomials (products of rotation matrices, sines and cosines) yield practically sparse CAD cell complexes. This is a plausible but entirely unvalidated hypothesis.

At k=4 (the optimistic case), CAD is tractable: QEPCAD-B handles 4-variable problems routinely. At k=7–10 (the realistic case), feasibility is uncertain and depends entirely on problem structure. This is why Amendment A1 (prototype zone abstraction by Month 2) is critical.

#### BDD Memory Analysis

The BDD encoding of a zone graph with Z total zones requires ceil(log₂ Z) Boolean variables for each state component plus additional variables for interaction state machines.

| Scene Size | Zones/Object | Total Zones | BDD Variables | Estimated BDD Size |
|------------|-------------|-------------|---------------|-------------------|
| 10 objects | 100 | 1,000 | ~100 | <10MB |
| 30 objects | 100 | 3,000 | ~350 | 100MB–1GB |
| 50 objects | 200 | 10,000 | ~700 | 1–10GB (uncertain) |
| 50 objects | 1,000 | 50,000 | ~800+ | 10GB+ (likely infeasible without reordering) |

These estimates assume reasonable variable ordering. With poor ordering, BDD sizes can be exponentially larger. The 16GB budget for 50-object scenes is plausible only if zone counts per object are controlled (<200) and variable ordering is effective.

#### The Procedural Benchmark Problem

All performance claims are based on procedural benchmarks — algorithmically generated scenes with controlled properties (treewidth, spatial distribution, object complexity). The panel has three concerns:

1. **Artificial structure:** Procedural generation may produce scenes with favorable structure (regular spatial clustering, controlled treewidth, uniform object complexity) that real XR scenes lack. Real scenes have irregular spatial distributions, varying object complexity, and potentially unbounded local interaction density.
2. **Circular evaluation:** If the benchmarks are designed by the tool's authors to exercise the tool's strengths, the evaluation is circular. External validity requires testing on scenes the authors did not design.
3. **No real XR scenes exist in the evaluation plan.** Amendment A4 requires at least 5 non-procedural scenes by Month 6, which partially addresses this concern.

#### The No-Code Reality

`code_loc: 0` in State.json. Every performance claim, every feasibility assertion, every scalability argument in this project is based on theoretical analysis, not measurement. The gap between theoretical feasibility and engineering reality is often 10–100× — algorithms that are "polynomial" in theory may have constants that make them impractical, memory access patterns may defeat cache hierarchies, and integration overhead may dominate algorithmic cost.

The panel does not penalize a project for being pre-implementation — that's the nature of the ideation phase. But we flag that **every computational claim in this document is unvalidated**, and Amendment A1 exists precisely to ground-truth the most critical one.

#### No GPU Required — Correct

The project correctly notes that symbolic and algebraic computation (CAD, BDD manipulation, SMT solving) does not benefit from GPU parallelism. These are branching, memory-bound computations that run best on CPUs with large caches. A modern laptop (Apple M-series or Intel i7/i9 with 16–32GB RAM) is the appropriate platform.

**Score: 5/10.** Tier 1 is solidly feasible. Tier 2 is feasible for moderate scenes but uncertain at scale. Tier 3 is entirely aspirational with zero supporting evidence. The CAD dimension problem (7–10 effective DOF vs. claimed 4) is the critical uncertainty. No code exists to validate any claim.

---

### Axis 5: Fatal Flaws

| # | Flaw | P(fatal) | Impact | Testable Early? |
|---|------|----------|--------|-----------------|
| F1 | CAD on SE(3) intractable at problem dimension (7–10) | 40% | Total — Tiers 2–3 collapse | **YES** — prototype by Month 2 |
| F2 | State machine extraction intractable on real XR code | 35% | Severe — tool requires manual annotation for everything | **YES** — test 10 real Unity projects |
| F3 | Zero validated user demand | 35% | Total — tool works but nobody uses it | **YES** — developer survey by Month 3 |
| F4 | Monte Carlo sampling dominates in practice | 30% | Total — formal verification adds insufficient marginal value | **YES** — implement MC baseline |
| F5 | Bounded treewidth assumption fails for real scenes | 25% | Moderate — Tier 3 infeasible for complex scenes | **YES** — measure treewidth on 50 real scenes |
| F6 | CEGAR non-convergence within time budget | 20% | Moderate — Tier 3 unusable | **PARTIAL** — test on simple scenes |
| F7 | Circular evaluation (procedural benchmarks only) | 15% | Moderate — results lack external validity | **YES** — add real scenes |
| F8 | Chart transition soundness on SO(3) unsolvable | 25% | Total — M2 collapses | **PARTIAL** — proof attempt |

#### Detailed Analysis

**F1: CAD Intractability at Problem Dimension (P=40%, Impact=Total)**

This is the highest-risk flaw. CAD complexity is doubly exponential in dimension. The project claims effective dimension k=4 via kinematic locality (M3a), but the panel's analysis shows realistic effective dimension of 7–10 for arm-based interactions (7 DOF for shoulder-to-fingertip chain). At dimension 7, even with optimized implementations (QEPCAD-B, RegularChains), CAD may produce >10⁶ cells for a single reachability predicate, taking hours or days. At dimension 10, CAD is likely infeasible on any hardware.

The mitigation is M3a's locality argument: not all joints participate in every predicate. For a simple button press (position-only), the effective dimension may indeed be 3–4 (shoulder pitch, shoulder yaw, elbow flexion, with wrist locked). But for a grasp interaction requiring specific hand orientation, all 7 DOF participate. The project must demonstrate CAD tractability for the *hard* cases (7-DOF grasps), not just the easy ones (3-DOF reaches).

**Kill criterion (A1):** Zone computation for a single button-press reachability predicate on a 4-joint chain must complete in <10 minutes with <10⁵ cells. If this gate fails, the project must pivot to an interval-only approach (Tier 1 only), abandoning the formal verification tiers that constitute the core novelty.

**F2: State Machine Extraction Intractability (P=35%, Impact=Severe)**

The tool requires a finite-state representation of the XR application's interaction logic (which objects can be interacted with, in what order, under what conditions). The project proposes automated extraction from Unity/WebXR source code via static analysis. But XR interaction logic is typically implemented in procedural C# or JavaScript with:

- Event-driven callbacks (OnTriggerEnter, OnGrab, OnRelease)
- Coroutines and async/await patterns
- State stored in mutable fields, often across multiple scripts
- Dynamic object instantiation and destruction
- Physics-engine-mediated interactions (collision detection, raycasting)

Extracting a finite-state machine from this code is at least as hard as general program analysis — and the general problem is undecidable. The project's claim of "automated extraction" is almost certainly overstated. Realistic options are:

- **Pattern-matched extraction** for standard idioms (Unity XR Interaction Toolkit callbacks), yielding an over-approximation. This is feasible but limited to well-structured code using standard interaction patterns.
- **Developer-provided annotations** in the DSL, where the developer explicitly specifies the state machine. This shifts the burden to the developer and requires DSL expertise.

Amendment A2 mandates honest scoping: automated extraction for standard idioms only, with developer annotation required for non-standard code.

**F3: Zero Validated User Demand (P=35%, Impact=Total)**

Building a tool that nobody wants is the most common failure mode in research software. The project has zero evidence of developer demand:

- No surveys of XR developers about accessibility pain points
- No interviews with accessibility specialists
- No feature requests from platform holders
- No analysis of accessibility-related bug reports in open-source XR projects

The tool may be technically excellent and still fail to achieve any impact if XR developers don't perceive spatial accessibility verification as a problem worth solving, or if they perceive it as a problem better solved by simpler means (manual testing, heuristic checks).

Amendment A7 requires structured feedback from ≥20 XR developers by Month 3. If <25% express interest, the project should reconsider its value proposition — potentially pivoting to a theory-only contribution or reframing as a general kinematic reachability verifier (which has applications beyond accessibility).

**F4: Monte Carlo Dominance (P=30%, Impact=Total)**

If a 1M-sample stratified Monte Carlo simulation detects all or nearly all of the accessibility bugs that formal verification detects, the value proposition of Tiers 2–3 collapses. The Monte Carlo approach is:

- 1,000–10,000× cheaper computationally
- Zero DSL requirement (just run the simulation)
- Trivially parallelizable
- Easy to understand and debug

The formal verification approach adds *provable guarantees* — it can certify that *no* parameterization violates the predicate, not just that none of the sampled parameterizations do. But if the practical marginal detection rate (bugs found by formal verification that Monte Carlo misses) is <5%, the provable guarantee is an academic distinction.

Amendment A3 requires a Monte Carlo baseline in the evaluation, with explicit measurement of marginal detection rate. Gate criterion: formal verification must detect ≥10% of bugs missed by Monte Carlo to justify its cost.

**F5: Bounded Treewidth Assumption Failure (P=25%, Impact=Moderate)**

The compositional decomposition (M3b) depends on the interaction graph having bounded treewidth. The project assumes k≤6 based on informal arguments about spatial proximity. But:

- Dense XR scenes (e.g., a virtual control panel with 50 buttons in a grid) may have high-treewidth interaction graphs (all buttons are mutually reachable, creating clique-like structures).
- Multi-step interaction sequences (e.g., "pick up tool A, then use it on objects B, C, D in order") create long-range dependencies that increase treewidth.
- No measurement of treewidth on real XR scenes exists.

If treewidth is unbounded (or effectively >10–12), the compositional decomposition provides no speedup, and Tier 3 scales as O(exp(N)) — infeasible for large scenes.

Amendment A4 (real scenes) and a supplementary analysis of treewidth on 50 real scenes (proposed as part of F5 testing) would ground-truth this assumption.

**F6: CEGAR Non-Convergence (P=20%, Impact=Moderate)**

CEGAR (counterexample-guided abstraction refinement) is not guaranteed to converge in finite time for all problem instances. The refinement loop may:

- Oscillate between over- and under-refinement
- Produce refinements that don't eliminate the spurious counterexample
- Converge but only after an impractical number of iterations

For well-structured problems, CEGAR typically converges in 5–20 iterations. But the SE(3) zone abstraction introduces geometric refinement decisions (which cell to split, along which dimension) that may not have good heuristics. Without prototype measurements, convergence behavior is unknown.

**F7: Circular Evaluation (P=15%, Impact=Moderate)**

Procedural benchmarks are generated by the tool's authors and may inadvertently (or deliberately) favor the tool's strengths. This is a standard concern in verification tool evaluation but is particularly acute here because:

- No standard benchmark suite for XR accessibility verification exists (the field is new)
- The procedural generation parameters (treewidth, spatial distribution, object complexity) are chosen by the authors
- Without real-scene benchmarks, reviewers have no way to assess external validity

This is the least likely to be "fatal" but would significantly weaken any publication. Amendment A4 (real scenes) directly addresses this.

**F8: Chart Transition Soundness (P=25%, Impact=Total)**

The SE(3) zone abstraction (M2) requires decomposing SO(3) into coordinate charts (e.g., Euler angle charts covering different regions of the rotation group) and computing CAD in each chart independently. The soundness of the overall abstraction depends on correct handling of transitions between charts — configurations that lie on chart boundaries must be consistently abstracted.

This is a mathematical problem, not an engineering one. If the chart-transition soundness proof fails (e.g., because chart overlaps create ambiguities in cell membership), M2 — the project's strongest mathematical contribution — collapses. A single-chart approach (e.g., using quaternion parameterization) avoids the transition problem but introduces algebraic complications (quaternion constraints are degree-2 polynomials that must be maintained throughout the CAD).

This is flagged as "PARTIAL — proof attempt" because soundness may be provable with sufficient mathematical effort, but the proof has not been completed.

#### Compound Probability

The flaws are partially correlated:

- F1 and F6 are correlated (both relate to computational feasibility of the core algorithms)
- F3 and F4 are correlated (both relate to value proposition)
- F1 and F8 are correlated (both relate to the zone abstraction)

Using a conservative dependence model (pairwise correlation ρ ≈ 0.3 for correlated pairs, ρ ≈ 0 for uncorrelated pairs), the compound probability of **at least one** fatal flaw materializing is:

P(≥1 fatal) ≈ 1 - ∏(1 - Pᵢ) × correction ≈ 85–90%

This is very high but is characteristic of ambitious research projects. The staged kill-chain (below) is designed to detect fatal flaws early and minimize wasted effort.

---

### Binding Amendments

These amendments are **required conditions** for the CONDITIONAL CONTINUE recommendation. Failure to execute any amendment by its deadline triggers reassessment.

#### A1: Prototype Zone Abstraction First

**Deadline:** Month 2.

**Requirement:** Implement the SE(3) zone abstraction for a single 4-joint kinematic chain (shoulder pitch, shoulder yaw, elbow flexion, wrist rotation) operating on a single interaction element (a button with a cylindrical activation volume).

**Gate criterion:** Zone computation for a single button-press reachability predicate must complete in <10 minutes on a laptop (Apple M-series or equivalent, 16GB RAM) and produce <10⁵ cells.

**Rationale:** This is the minimum viable test of the project's core algorithmic hypothesis. If CAD on a 4-DOF kinematic chain with a single semialgebraic predicate produces >10⁶ cells or takes >1 hour, the zone abstraction approach is infeasible at any scale, and Tiers 2–3 cannot work.

**If gate fails:** Pivot to interval-only approach (Tier 1). Abandon zone abstraction. Reframe project as fast interval-based accessibility linter. Pursue UIST/ASE tool paper only.

#### A2: Honest State Machine Extraction Scope

**Deadline:** Immediate.

**Requirement:** Define two explicit modes of operation:

- **(a) Automated mode:** Pattern-matched extraction for standard Unity XR Interaction Toolkit idioms (OnSelectEntered, OnSelectExited, OnActivated, OnHoverEntered, etc.). Produces over-approximate state machines (may include infeasible transitions). Limited to single-script interaction components with standard callback patterns.
- **(b) Annotated mode:** Developer-provided state machine specifications in the DSL. Required for: multi-script interactions, custom interaction patterns, Tiers 2–3 soundness guarantees, any non-standard XR framework.

**Rationale:** Claiming fully automated state machine extraction from arbitrary XR codebases is overclaiming. The general problem is undecidable (Rice's theorem). Honest scoping prevents reviewer backlash and sets realistic user expectations.

**If violated:** Reviewers will attack the extraction claim as the weakest link, undermining otherwise strong algorithmic contributions.

#### A3: Monte Carlo Baseline Required

**Deadline:** Included in evaluation (Month 6–8).

**Requirement:** Implement a stratified Monte Carlo baseline:

- Sample 1M body parameterizations from anthropometric distributions (ANSUR-II database for military population, CAESAR for civilian)
- For each sample, compute forward kinematics and check reachability of all interaction elements
- Report: detection rate (fraction of accessibility bugs found), false positive rate, runtime

**Gate criterion:** Formal verification (Tiers 2–3) must detect ≥10% of bugs missed by Monte Carlo. Specifically: there must be at least 10% more unique accessibility violations detected by formal verification that are not detected by any of the 1M Monte Carlo samples.

**Rationale:** Without a Monte Carlo baseline, the paper cannot demonstrate that formal verification adds value over a simple, fast, scalable alternative. This is the most predictable reviewer objection.

**If gate fails:** Marginal detection rate <5% → ABANDON Tiers 2–3. Focus on Tier 1 (which is faster than MC and useful as a quick filter) and the theory contributions (M2, M3b) as standalone results.

#### A4: Real XR Scenes Required

**Deadline:** Month 6.

**Requirement:** Include at least 5 non-procedural XR scenes in the evaluation:

- **3 from open-source applications:** Candidates include Mozilla Hubs (social VR), A-Frame examples (web XR), Unity XR Interaction Toolkit samples, OpenXR conformance test scenes.
- **2 hand-crafted for realistic complexity:** Designed to stress-test the tool with features common in real applications (dense control panels, multi-step interaction sequences, dynamic objects, physics-mediated interactions).

**Gate criterion:** The tool must successfully process all 5 scenes through at least Tier 1 and produce meaningful accessibility reports. At least 3 scenes must complete Tier 2 analysis within the time budget.

**If gate fails:** <5 real scenes obtainable, or tool fails on all real scenes → Downscope to theory-only contribution. Publish M2/M3b at CAV/TACAS without tool evaluation.

#### A5: Two-Paper Strategy

**Deadline:** Immediate.

**Requirement:** Plan and structure all work toward two independent publications:

1. **Theory paper (target: CAV 2026 or TACAS 2026):**
   - Focus: M2 (SE(3) zone abstraction) + M3b (bounded-treewidth decomposition)
   - Content: Formal definitions, proofs, zone computation experiments, scalability analysis
   - Framing: "Verification of kinematic reachability on Lie groups" — general formal methods contribution, XR as motivating application
   - No tool evaluation, no user studies, no DSL

2. **Tool paper (target: UIST 2026, OOPSLA 2026, or ASE 2026):**
   - Focus: Tier 1 + Tier 2 pipeline, DSL, case studies on real scenes
   - Content: System architecture, DSL design, MC baseline comparison, developer feedback
   - Framing: "First accessibility linter for mixed reality" — practical tool contribution

**Critical:** Drop the "defines a new subfield" framing entirely. This invites hostile reviewer scrutiny and signals overclaiming. Let the novelty of the work speak for itself.

#### A6: Scope Reduction

**Deadline:** Immediate.

**Requirement:** Reduce total implementation scope to ~130–150K LoC (from claimed 175–210K):

- **Unity-only scene parsing.** Drop WebXR and OpenXR parsers. Unity dominates the XR development market (>60% market share for non-gaming XR). WebXR and OpenXR parsing adds ~15K LoC of engineering with minimal incremental value for a research prototype.
- **Merge cross-device module into compositional engine.** Cross-device interaction (e.g., hand tracking + controller) is a special case of compositional verification. A separate module is unnecessary.
- **Simplify visualization.** Replace 3D counterexample rendering with text reports + 2D SVG diagrams of body configurations. Saves ~8K LoC and removes a WebGL/Three.js dependency.
- **Defer benchmark infrastructure.** Minimal procedural generation only. Full benchmark suite after core algorithms are validated.

**Rationale:** Scope reduction reduces implementation risk, accelerates time-to-prototype, and focuses engineering effort on the algorithmically novel components. Every LoC not written is a LoC that doesn't need debugging.

#### A7: Demand Signal

**Deadline:** Month 3.

**Requirement:** Conduct structured outreach to XR developers:

- Identify ≥50 XR developers (via Unity forums, XR Developer Discord, Reddit r/virtualreality, LinkedIn XR developer groups)
- Present a 1-page description of the proposed tool (what it does, what input it requires, what output it produces)
- Collect structured feedback: (a) Is spatial accessibility a problem you encounter? (b) Would you use a tool like this? (c) What would make it useful/unusable?
- Target: ≥20 responses with structured data

**Gate criterion:** ≥25% of respondents express concrete interest (would try the tool, would integrate into workflow). <25% → Reconsider value proposition.

**If gate fails:** <25% interest → Options: (a) pivot to theory-only contribution, (b) reframe as general kinematic reachability verifier, (c) abandon project.

---

### Staged Kill-Chain

The kill-chain is a sequence of go/no-go gates designed to detect fatal flaws early and minimize sunk cost. Each gate has a clear deadline, criterion, and kill action.

| Stage | Deadline | Gate | Criterion | Kill Action |
|-------|----------|------|-----------|-------------|
| 1 | Month 2 | A1: Zone Abstraction Prototype | CAD on 4-joint chain, single predicate: <10 min, <10⁵ cells | If >10⁶ cells or >1 hour → **PIVOT** to interval-only (Tier 1). Abandon Tiers 2–3. |
| 2 | Month 3 | A7: Demand Signal | ≥25% of ≥20 surveyed XR developers express interest | If <25% interest → **ABANDON** tool track, or pivot to theory-only |
| 3 | Month 6 | A3: MC Comparison | Formal verification detects ≥10% of bugs missed by 1M-sample Monte Carlo | If marginal detection <5% → **ABANDON** Tiers 2–3. Publish Tier 1 + theory only. |
| 4 | Month 6 | A4: Real Scenes | ≥5 real XR scenes processed; ≥3 complete Tier 2 | If <5 real scenes or tool fails → **DOWNSCOPE** to theory-only contribution |

**Stage 1 (Month 2)** is the most critical gate. If the zone abstraction prototype fails, the entire algorithmic core of the project is invalidated, and the project reduces to a Tier 1 interval-based linter — useful, but not a major research contribution. The remaining gates (Stages 2–4) can be pursued in parallel with ongoing development.

**Stage 2 (Month 3)** is the most *impactful* gate. If developers don't want the tool, even a technically successful project will have no impact. This gate should run concurrently with Stage 1 development.

**Stages 3–4 (Month 6)** are evaluation gates that determine whether the full system or a reduced version is publishable. By Month 6, sufficient code should exist to run real experiments on real scenes with a real Monte Carlo baseline.

**Escape hatches:** At every kill point, the project retains value:
- After Stage 1 kill: Publishable theory on SE(3) zone abstraction (M2) even if infeasible at scale — negative results are publishable at CAV workshops.
- After Stage 2 kill: Theory paper (M2 + M3b) at CAV/TACAS is independent of developer demand.
- After Stage 3 kill: Tier 1 tool + theory paper. Tier 1 has value as a fast accessibility linter even without formal guarantees.
- After Stage 4 kill: Theory paper with procedural benchmarks only. Weaker but still publishable.

---

### Overall Recommendation

## CONDITIONAL CONTINUE

The panel recommends **conditional continuation** of the xr-affordance-verifier project, subject to all seven binding amendments (A1–A7) and the four-stage kill-chain.

**Rationale for continuation:**

1. **Genuine mathematical novelty.** The SE(3) zone abstraction (M2) and bounded-treewidth compositional decomposition (M3b) are real contributions that advance the state of the art in formal verification. These results have value independent of the tool's practical success.

2. **Staged risk management.** The kill-chain ensures that the highest risks (CAD infeasibility, zero demand) are tested within the first 2–3 months, before the majority of implementation effort is invested. The worst-case sunk cost is ~2 months of prototyping.

3. **Valuable failure modes.** Even if the project fails at its most ambitious goals (Tier 3 verification of complex scenes), it can still produce: (a) a publishable theory paper on SE(3) zone abstraction, (b) a fast Tier 1 accessibility linter with practical utility, (c) a formal framework (PGHA) that others can build on.

4. **Real problem.** Spatial accessibility in XR is genuinely under-served. Even a partial solution (Tier 1 interval linter) would be the first automated tool for this problem domain.

**Rationale for conditionality:**

1. **Compound risk is ~85–90%.** The probability that at least one fatal flaw materializes is very high. Unconditional commitment to the full scope would be reckless.

2. **Zero code, zero evidence.** Every performance claim is aspirational. The project has not earned trust — it must build it through the staged gates.

3. **Value proposition is unvalidated.** The tool may work perfectly and still have zero impact if developers don't want it or Monte Carlo sampling suffices.

4. **Publication path is narrow.** The two-paper strategy is viable but not guaranteed. Both papers face conditional acceptance: the theory paper needs resolved chart transitions, and the tool paper needs real scenes and developer feedback.

**What "conditional continue" means in practice:**

- Months 1–2: Prototype zone abstraction (A1). Begin demand survey (A7). Implement honest extraction scope (A2). Adopt two-paper strategy (A5) and scope reduction (A6).
- Month 2 gate: If A1 fails → PIVOT immediately. Do not invest further in Tiers 2–3.
- Month 3 gate: If A7 fails → ABANDON tool track or pivot to theory-only.
- Months 3–6: If gates passed, implement full Tier 1 + Tier 2 pipeline. Collect real scenes (A4). Implement MC baseline (A3).
- Month 6 gates: If A3 or A4 fail → DOWNSCOPE to theory contribution.
- Months 6–12: If all gates passed, write and submit both papers.

---

### Score Summary

| Axis | Score | Assessment |
|------|-------|------------|
| Value (V) | 4/10 | Real problem, small market, simpler alternatives exist, zero demand evidence |
| Difficulty (D) | 7/10 | 85–110K novel LoC, SE(3) zone abstraction is genuinely hard, no existing solution |
| Best-Paper (BP) | 4/10 | 2 strong math results (M2, M3b), no natural venue home, zero user studies |
| Laptop CPU (L) | 5/10 | Tier 1 feasible, Tier 2 uncertain at scale, Tier 3 aspirational, zero measurements |
| **Composite** | **20/40** | **CONDITIONAL CONTINUE with staged kill-chain** |

---

*This depth check is binding. Amendments A1–A7 and the staged kill-chain are required conditions for continuation. Deviation from these conditions without panel reassessment constitutes an unmitigated risk acceptance.*
