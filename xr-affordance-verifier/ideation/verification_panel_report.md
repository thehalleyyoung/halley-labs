# Independent Verification Panel Report
## XR Affordance Verifier — Crystallized Problem Review

**Date:** 2025-07-18
**Reviewers:** Math Verifier, Prior Art Verifier
**Document under review:** `ideation/crystallized_problem.md`

---

# PART I: MATH VERIFICATION

## M1: PGHA Formalism and Operational Semantics — Difficulty: B

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **PASS** |
| Difficulty rating honest? | **PASS** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **CONDITIONAL PASS** |
| Mathematician could poke a hole? | Minor concerns only |

**Analysis:** The formalism is well-specified. SE(3) × T^n is a concrete manifold, semialgebraic guards are a well-defined class, and operational semantics for hybrid automata are well-studied (Henzinger et al., 1996). Adapting to Lie group continuous state is a genuine wrinkle but not a deep open problem. B is honest.

**Novelty concern:** The claim that "existing hybrid automata formalize different continuous dynamics (typically R^n with polynomial ODEs)" understates prior work. Hybrid automata on manifolds have been explored in the geometric control community (e.g., Tabuada & Pappas, 2004, on bisimulation for systems on manifolds). The specific combination with semialgebraic guards over parameterized kinematic chains is plausibly new, but the novelty claim should acknowledge the geometric hybrid systems literature.

**Fix required:** Add explicit positioning against geometric hybrid systems literature (Tabuada, Pappas, Belta, etc.) to justify what is new beyond "hybrid automata on manifolds."

**Verdict: CONDITIONAL PASS** — Needs sharper novelty positioning.

---

## M2: Soundness of SE(3) Zone Abstraction — Difficulty: B+

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **CONDITIONAL PASS** |
| Difficulty rating honest? | **CONDITIONAL PASS — likely underrated** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **PASS** |
| Mathematician could poke a hole? | **YES — serious topological concern** |

**Analysis:** This is the contribution I'm most concerned about. The document correctly identifies that SO(3) is non-contractible, but underestimates the consequences.

**The topological problem:** CAD operates on semialgebraic subsets of R^n. SE(3) is not R^n. To apply CAD, you must either:

1. **Embed SE(3) in R^{12}** (3×3 rotation matrix + 3D translation) with 6 orthogonality constraints (R^T R = I, det R = 1). This inflates dimension from 6 to 12 with 6 polynomial constraints. CAD complexity is doubly exponential in the number of *free* variables, so this is doubly exponential in 6 (the effective dimension after constraint propagation) — barely tractable, but vastly more expensive than naive dimension counting suggests.

2. **Use local charts** (e.g., Euler angles, quaternions). But SO(3) cannot be covered by a single chart without singularity. Euler angles have gimbal lock; quaternions are a double cover (S^3 → SO(3)). Any chart-based approach requires proving that the zone partition is consistent across chart transitions. This is a non-trivial topological soundness obligation with no precedent in CAD literature.

**The dimension explosion:** Even with the embedding approach, the effective CAD dimension includes joint angles. For a human arm with 7 DOF (shoulder 3 + elbow 1 + wrist 3), plus SE(3) torso = 13 continuous dimensions. CAD at dimension 13 is 2^{2^{13}} in the worst case — astronomically infeasible. The document relies on M3 (interaction locality) to rescue this, claiming you only need CAD over k ≤ 4 joints at a time. But M2's soundness proof cannot depend on M3's tractability result — soundness must be proven for the general case first, then tractability shows it's practically computable.

**Difficulty assessment:** B+ is plausibly underrated. The chart transition soundness problem alone could be A- level. If the embedding approach is used, the polynomial constraint handling in CAD adds substantial complexity. I'd rate this A- to be safe.

**Fix required:**
1. Specify whether the approach uses embedding or charts, and address the consequences of that choice explicitly.
2. Separate the soundness proof (general, any dimension) from the tractability argument (requires M3).
3. Consider upgrading difficulty to A-.

**Verdict: CONDITIONAL PASS** — Needs topological approach specified and difficulty potentially upgraded.

---

## M3: Interaction-Locality FPT — Difficulty: A-

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **FAIL** |
| Difficulty rating honest? | **CONDITIONAL PASS** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **PASS** |
| Mathematician could poke a hole? | **YES — fundamental scoping issue** |

**Analysis:** This is the crown jewel claim and it has a critical precision problem.

**The undecidability barrier:** Hybrid automata reachability is undecidable for linear hybrid automata (Henzinger et al., 1998). PGHA, with continuous dynamics on SE(3) × T^n, is strictly more expressive than linear hybrid automata. Therefore, PGHA reachability is undecidable. The FPT claim CANNOT be about concrete PGHA reachability — it must be about the abstract zone-graph reachability after applying the M2 abstraction. The document does not make this distinction.

**What the claim likely means:** After the zone abstraction of M2 produces a finite discrete graph, the *abstract* reachability problem on this graph is FPT in k. Specifically, if each interaction involves at most k joints, then the zone graph has O(f(k) · poly(n)) zones (because you only need fine granularity in k dimensions, with coarse partitions for the remaining n−k), and graph reachability on a polynomial-size graph is polynomial.

**The discrete state explosion problem:** Even granting the continuous-dimension argument, the claim asserts O(f(k) · poly(n, **m**)) where m is scene size. Being polynomial in m means the *discrete* state explosion (m interactable objects, each with state, giving up to 2^m product states) is also tamed by k-locality. This is the stronger claim and requires a separate argument: that the interaction state machines decompose by locality (object A's state only affects objects within spatial proximity). This is essentially the M4 compositional argument, not an M3 locality argument. Conflating them weakens both.

**Specific hole a mathematician would exploit:** An adversary constructs a scene where m objects have a chain of k-local interactions (A₁ triggers A₂ triggers ... triggers Aₘ) where each trigger is 2-local but the *sequential* dependency creates an m-length chain requiring 2^m state exploration. This is k-local (k=2) but not FPT in k with poly(m) — it's poly(n)·2^m.

**Fix required:**
1. State explicitly that the FPT claim is about *abstract* (zone-graph) reachability, not concrete PGHA reachability.
2. Separate the continuous-dimension locality argument (f(k)·poly(n) zone count) from the discrete-state tractability argument (poly(m) — which requires compositional assumptions from M4 or additional structural hypotheses about interaction graphs).
3. Address the sequential-dependency counterexample: under what structural conditions on the interaction graph does the poly(m) claim hold? (Likely: bounded treewidth or bounded interaction depth.)
4. Consider whether this is really two theorems: M3a (continuous tractability via k-locality) and M3b (discrete tractability via interaction-graph structure), with different difficulty ratings.

**Verdict: FAIL** — The claim conflates abstract/concrete reachability and continuous/discrete tractability. Needs fundamental restructuring.

---

## M4: Compositional Assume-Guarantee for PGHA — Difficulty: B

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **PASS** |
| Difficulty rating honest? | **CONDITIONAL PASS — possibly underrated** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **CONDITIONAL PASS** |
| Mathematician could poke a hole? | Minor concerns |

**Analysis:** Assume-guarantee reasoning for hybrid automata exists (Henzinger et al., 2001; SpaceEx AGAR). The adaptation to semialgebraic spatial guards is genuinely novel but the document undersells the difficulty.

**Novelty concern:** The document says "assume-guarantee rules exist for hybrid automata (Henzinger et al.) but not with semialgebraic spatial guards." This is partially true — SpaceEx's AGAR framework handles affine dynamics with semialgebraic-like guards. The claim should be more specific: what is new is the *spatial interface predicates* (assumptions about neighbor behavior expressed as constraints on shared pose-space boundaries). This is a genuine contribution but should be positioned more carefully.

**Difficulty concern:** The circular assume-guarantee discharge conditions are notoriously subtle (the Namjoshi-Trefler completeness problem). Getting the proof rule both sound and useful (not so restrictive that it never applies) is harder than B. Consider B+.

**Unacknowledged dependency:** The *automatic* decomposition of scenes into clusters is NP-hard graph partitioning, acknowledged in subsystem 8's description but not reflected in the mathematical contribution. The quality of decomposition determines the practical utility of M4. This should at least be noted as an important algorithmic challenge even if it's not a theorem.

**Verdict: CONDITIONAL PASS** — Sharpen novelty positioning against SpaceEx AGAR; consider B+ difficulty.

---

## M5: Body-Capability Monotonicity — Difficulty: B-

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **CONDITIONAL PASS** |
| Difficulty rating honest? | **PASS** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **PASS** |
| Mathematician could poke a hole? | **YES — the trap-freedom condition** |

**Analysis:** The single-step monotonicity is essentially trivial (larger reach envelope ⊇ smaller reach envelope, so any zone reachable by the smaller body is reachable by the larger). The real content is in the multi-step extension.

**The trap-freedom condition is a real subtlety, not hand-waving.** The robotics literature confirms this: a longer arm can reach behind an obstacle but then cannot retract without collision. This is a well-known phenomenon in manipulation planning (non-monotonic workspace accessibility). The document correctly identifies it as the key difficulty.

**However, the condition needs precise definition.** The current formulation ("extend to multi-step interactions with a trap-freedom condition") is too vague. Specifically:
- Is trap-freedom a property of the scene (static, checkable once)?
- Is it a property of the body+scene pair (varies with parameterization)?
- Is checking trap-freedom itself decidable/tractable?

If trap-freedom is as hard to verify as the original reachability problem, then M5 provides no practical benefit — you'd need to solve the hard problem to know whether the easy reduction applies.

**Fix required:** Formally define the trap-freedom condition and bound the complexity of checking it. If it's co-NP or undecidable to check, acknowledge this and explain how it's used in practice (e.g., as a sufficient condition that covers common cases).

**Verdict: CONDITIONAL PASS** — Trap-freedom needs formal definition and complexity characterization.

---

## M6: Device-Capability Subsumption Lattice — Difficulty: B

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **PASS** |
| Difficulty rating honest? | **PASS** |
| Genuinely load-bearing? | **CONDITIONAL PASS** |
| Novelty claim honest? | **PASS** |
| Mathematician could poke a hole? | Minor concerns |

**Analysis:** Clean formalization. The lattice structure is well-defined and the subsumption theorem is straightforward.

**Load-bearing concern:** The current XR device landscape has substantial incomparability (Quest 3: hand tracking + controllers; PSVR2: eye tracking + wands; Apple Vision Pro: eye + hand + no controllers). If most device pairs are incomparable (neither subsumes the other), the lattice has few non-trivial subsumption relationships, and the verification reduction is minimal. The claim "reduces cross-device verification from |devices| × |body params| to verification at lattice-minimal elements" is only useful if the lattice has significantly fewer minimal elements than total devices.

**Fix required:** Provide an honest assessment of the current XR device lattice structure. How many meet-irreducible elements exist? If the answer is "nearly as many as devices," acknowledge limited practical benefit.

**Verdict: CONDITIONAL PASS** — Needs honest assessment of lattice utility for the actual device ecosystem.

---

## M7: Counterexample Concretization Completeness — Difficulty: B

| Criterion | Verdict |
|-----------|---------|
| Precise enough to prove/disprove? | **PASS** |
| Difficulty rating honest? | **PASS** |
| Genuinely load-bearing? | **PASS** |
| Novelty claim honest? | **PASS** |
| Mathematician could poke a hole? | Minor concern on termination |

**Analysis:** Standard CEGAR adaptation with two novel aspects: (1) geometric concretization in SE(3), (2) biomechanical plausibility constraints. Both are well-motivated.

**Termination concern:** The termination argument ("each refinement strictly increases zone count, which is bounded by the finite algebraic complexity of the scene") is sound in principle. But the bound could be astronomically large (doubly exponential in dimension, per M2). Termination is guaranteed but practical convergence is a separate question. The document should distinguish "terminates in theory" from "converges in practice within the hour-scale Tier 3 budget."

**Minor concern:** Biomechanical plausibility checking during concretization involves nonlinear constraint satisfaction (joint velocity limits, collision avoidance, postural stability). This is NP-hard in general. The document acknowledges this in subsystem 9's description but doesn't bound concretization time. This is acceptable — CEGAR concretization is often expensive — but should be noted.

**Verdict: PASS**

---

# PART II: PRIOR ART VERIFICATION

## Portfolio Overlap

### With xr-ergonomic-layout-solver (area-060)

**Overlap level: MODERATE — genuinely concerning**

Both projects:
- Verify XR spatial accessibility
- Model parametric human body diversity (5th-95th percentile)
- Reason about reach envelopes and ergonomic constraints
- Target the same users (XR developers, platform holders)

**Differentiation (genuine):**
- xr-ergonomic-layout-solver uses CVXPY constraint satisfaction with MUS explanation — it's a *design-time layout optimizer*, not a state-machine verifier. It has 0 novel theorems and ~2-5K novel LoC.
- xr-affordance-verifier uses hybrid automata model checking with CEGAR — it's a *runtime reachability verifier* that handles multi-step interaction sequences.
- The ergonomic solver checks static spatial constraints; the affordance verifier checks dynamic sequential reachability.

**Verdict: CONDITIONAL PASS** — The differentiation is real but must be explicitly stated in the crystallized problem. Add a paragraph distinguishing from static-constraint approaches (which the ergonomic solver represents) and explain why dynamic sequential verification is strictly harder and not subsumed.

### With xr-interaction-grammar-compiler (area-060)

**Overlap level: MODERATE-STRONG — conceptual, not technical**

Both projects:
- Formalize XR spatial interactions (grammars vs. state machines)
- Work with 6DOF spatial predicates
- Target automated XR testing/verification

**Differentiation (genuine):**
- The grammar compiler produces *testing oracles* (does this execution trace match the expected interaction grammar?). It tests behavior, not accessibility.
- The affordance verifier tests *spatial reachability* (can a given body type physically perform this interaction?). It tests capability, not behavior.
- Different mathematical cores: timed context-free grammars vs. hybrid automata.

**Verdict: PASS** — Genuinely complementary. One tests "did the user do the right thing?", the other tests "can the user do the thing at all?"

### With spatial-hash-compiler, bounded-rational-usability-oracle, cross-lang-verifier

**These projects do not exist in the actual portfolio.** The 69-project portfolio contains no projects with these names. Cannot evaluate overlap with nonexistent projects.

**Verdict: N/A**

---

## Novelty Claim: "First formal verification for XR accessibility"

**Assessment: CONDITIONAL PASS — defensible but needs qualification**

**What exists in robotics:**
- Workspace reachability analysis is a mature field (COMPAS FAB, scikit-robot, DLR Reuleaux). These compute reach envelopes for robot manipulators via discretized forward kinematics.
- These are *numerical sampling methods* (grid-based, Monte Carlo), NOT formal verification with soundness guarantees.
- They do not handle multi-step interaction sequences, interaction state machines, or parametric body diversity.

**What exists in CPS verification:**
- UPPAAL, SpaceEx, and other hybrid system model checkers handle cyber-physical reachability. None have been applied to XR accessibility specifically.
- SpaceEx handles affine dynamics with polyhedral guards — not SE(3) dynamics with semialgebraic guards.
- No existing tool combines body parameterization, interaction state machines, and spatial reachability in a single framework.

**The claim holds up, but needs caveats:**
1. Robotics workspace analysis is a clear antecedent — the conceptual framework (kinematic chain + reach envelope + reachability) comes from this community. Not citing it would be academically dishonest.
2. UPPAAL/SpaceEx provide the verification methodology — the contribution is the *domain-specific adaptation*, not invention of model checking from scratch.

**Fix required:** Reframe from "no tool exists" to "existing robotics workspace tools lack formal guarantees, existing CPS model checkers lack body-parameterized spatial semantics — this work bridges both."

---

## EU Accessibility Act Angle

**Assessment: CONDITIONAL PASS — legitimate but overstated**

**What the EAA actually requires:**
- The EAA (Directive 2019/882, effective June 28, 2025) mandates accessibility for digital products and services.
- It does NOT explicitly name XR, VR, or MR as product categories.
- It applies to "consumer general purpose computer hardware systems" — XR headsets likely fall under scope but this hasn't been tested.
- It requires that products be *perceivable, operable, understandable, and robust* per EN 301 549 / WCAG 2.1 AA.
- It does NOT mandate any specific testing methodology, let alone formal verification.

**What the document claims:**
- "The EU Accessibility Act (effective June 2025) and evolving ADA digital-space interpretations are creating legal mandates for accessible XR" — **partially true**, but XR is not explicitly named.
- "No automated compliance tools exist" — **mostly true** for spatial XR accessibility, but automated web accessibility scanners (Axe, WAVE) exist for web-based XR (WebXR).
- "Machine-checkable certificates" — the EAA does not require or recognize machine-checkable certificates. Compliance is demonstrated via technical documentation and declarations of conformity.

**The regulatory angle is LEGITIMATE as motivation** — accessibility regulations are expanding, XR will eventually be covered, and automated tools will be needed. But the document overstates the specificity and urgency of XR-specific mandates.

**Fix required:** Temper the regulatory language:
- Change "creating legal mandates for accessible XR" to "will likely extend to XR products as the regulatory scope is interpreted"
- Drop "machine-checkable certificates" language unless grounded in actual regulatory framework
- Acknowledge that the EAA does not currently name XR explicitly

---

# OVERALL VERDICTS

## Summary Table

| Item | Verdict | Fix Required? |
|------|---------|---------------|
| M1 (PGHA formalism) | CONDITIONAL PASS | Sharpen novelty against geometric hybrid systems literature |
| M2 (SE(3) zone abstraction) | CONDITIONAL PASS | Specify topological approach; consider A- difficulty |
| M3 (interaction-locality FPT) | **FAIL** | Fundamental restructuring needed (see detailed fixes) |
| M4 (compositional A-G) | CONDITIONAL PASS | Position against SpaceEx AGAR; consider B+ |
| M5 (monotonicity) | CONDITIONAL PASS | Formally define trap-freedom; bound its complexity |
| M6 (device lattice) | CONDITIONAL PASS | Honest assessment of lattice utility |
| M7 (CEGAR completeness) | PASS | — |
| Portfolio overlap (ergonomic solver) | CONDITIONAL PASS | Explicit differentiation paragraph |
| Portfolio overlap (grammar compiler) | PASS | — |
| Novelty claim | CONDITIONAL PASS | Acknowledge robotics & CPS antecedents |
| EU Accessibility Act | CONDITIONAL PASS | Temper regulatory specificity claims |

---

## OVERALL VERDICT: CONDITIONAL SIGNOFF

The crystallized problem describes a genuinely novel and ambitious project with real mathematical depth. The formalism is creative, the application domain is compelling, and most mathematical contributions are honestly scoped. However, **signoff is conditional on the following mandatory changes:**

### BLOCKING (must fix before signoff):

1. **M3 must be restructured.** The FPT claim conflates three distinct issues:
   - (a) Abstract vs. concrete reachability (the claim is about the abstract zone graph, not the undecidable concrete problem — say so).
   - (b) Continuous-dimension tractability (the zone count is f(k)·poly(n) when interactions are k-local — this is the real M3).
   - (c) Discrete-state tractability (poly(m) for m objects — this requires structural assumptions on the interaction graph, e.g., bounded treewidth, not just k-locality. This is either part of M4 or a separate theorem M3b).
   - Address the sequential-dependency counterexample: m objects with a chain of 2-local interactions creating 2^m state exploration.

### NON-BLOCKING (should fix, but signoff can proceed):

2. M1: Add positioning against geometric hybrid systems literature (Tabuada & Pappas).
3. M2: Specify whether the approach uses embedding or charts; consider upgrading to A-.
4. M4: Acknowledge SpaceEx AGAR as closest prior work; consider B+ difficulty.
5. M5: Formally define the trap-freedom condition and characterize its verification complexity.
6. M6: Provide honest lattice analysis for the current XR device ecosystem.
7. Portfolio: Add explicit differentiation paragraph against xr-ergonomic-layout-solver.
8. Novelty: Reframe as bridging robotics workspace analysis and CPS verification, not "nothing exists."
9. EU Accessibility Act: Temper claims to reflect that XR is not explicitly named in the EAA.

Once the M3 blocking issue is resolved, this project merits signoff. The underlying idea is sound and genuinely novel — it just needs more precise mathematical scoping.
