# Depth Check: Choreo — XR Interaction Compiler

**Verification Panel**: 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) + Team Lead  
**Date**: 2026-03-08  
**Slug**: choreo-xr-interaction-compiler  
**Verdict**: **CONDITIONAL CONTINUE** (2-1, Skeptic dissents ABANDON)

---

## Scoring Summary

| Axis | Auditor | Skeptic | Synthesizer | **Lead Ruling** |
|------|---------|---------|-------------|-----------------|
| 1. Extreme and Obvious Value | 4 | 3 | 7 | **5** |
| 2. Genuine Difficulty | 7 | 4 | 8 | **7** |
| 3. Best-Paper Potential | 5 | 2 | 6 | **5** |
| 4. Laptop CPU + No Humans | 8 | 5 | 9 | **8** |
| **Composite (equal weight)** | 6.0 | 3.5 | 7.5 | **6.25** |

**Fatal Flaws**: 2 genuine, 3 serious risks  
**Amendments Required**: 8 (5 must-have, 3 should-have)

---

## Axis 1: Extreme and Obvious Value — **5/10** ⛔

### What the problem gets right

The structural argument is sound: the XR stack has standardized content (glTF), rendering (Vulkan/WebGPU), and tracking (OpenXR), but interaction choreography — what users *do* in a scene — remains imperative spaghetti. This "missing layer" observation is the kind of clean architectural gap that resonates with PL reviewers. The headless CI/CD angle ("runs on a $0.008/min GitHub Actions runner, no GPU, no headset") solves a real operational pain for XR development teams who currently require expensive GPU-provisioned test infrastructure.

### What's missing

**No evidence of demand.** The crystallized problem asserts that XR developers need formal verification but provides zero evidence: no surveys, no interviews, no GitHub issue analysis quantifying interaction-logic bugs as a category. The addressable population for multi-party spatial-temporal interaction choreographies is ~30K–80K developers globally (generous estimate; the Skeptic argues 5K). Platform teams at Meta, Apple, and Microsoft have internal QA pipelines and are unlikely to adopt an external academic tool.

**LLM context shift.** By the time Choreo ships (~2027), LLM-based test generation will absorb much of the practical value proposition (finding common interaction bugs) at a fraction of the adoption cost. LLMs cannot replace model checkers for exhaustive verification, but the marginal value of exhaustive verification over high-coverage AI-generated testing in a domain with small state spaces (<15 zones) is thin.

**The accessibility auditing use case is compelling but unsubstantiated.** "Can every scene state be reached via eye-tracking alone?" is a powerful reachability question with growing regulatory relevance (EU Accessibility Act 2025, US Section 508). But no accessibility team has ever requested this capability. Elevating this from a bullet point to a named contribution — with evidence of compliance demand — would strengthen the value proposition significantly.

### What would raise this to 7

1. Concrete demand evidence: even 5 informal interviews with XR interaction developers about testing pain points.
2. Elevate accessibility auditing to a co-equal contribution with regulatory framing.
3. Demonstrate the "missing layer" argument with a concrete example: extract one MRTK interaction protocol, show it cannot be specified in any existing format (glTF, USD, OpenXR), then show it in Choreo.

---

## Axis 2: Genuine Difficulty as Software Artifact — **7/10** ✅

### Where the genuine difficulty lives

**The spatial CEGAR loop (~14K LoC, 65% novel)** is the crown jewel. Standard CEGAR refines by adding Boolean predicates; spatial CEGAR refines by splitting geometric regions. When a counterexample is spurious because two zones can't physically overlap, the refinement step solves a spatial constraint satisfaction problem in ℝ³, not a propositional formula. GJK/EPA collision detection inside a CEGAR refinement loop has no published precedent.

**EC→automata compilation (~15K LoC, 50% novel)** reverses all prior work. Existing EC systems (Mueller, Shanahan, Artikis) translate automata *into* EC for reasoning. Choreo compiles EC *into* automata for execution. Thompson-style construction extended with spatial-temporal guards plus on-the-fly product composition with symbolic state representation is a genuine compiler engineering challenge.

**The spatial type system (~10.5K LoC, 60% novel)** — decidable spatial subtyping for convex polytopes (via LP feasibility) composed with Allen's 13 interval relations creates a product lattice with non-trivial join/meet operations. The decidability boundary question (convex polytopes: P-time; bounded CSG: NP-complete; unbounded CSG: undecidable) is a genuine PL × computational geometry open problem.

**Multi-domain expertise requirement** — simultaneous fluency in PL theory (type systems, soundness), computational geometry (GJK/EPA, R-trees, CSG), formal verification (CEGAR, BDD encoding, compositional model checking), and Event Calculus (circumscription, abduction). Finding one person with depth in 3+ of these is genuinely rare.

### Honest decomposition

| Category | LoC | % of Total |
|----------|-----|-----------|
| Research-grade novel | ~50,000 | 34% |
| Non-trivial domain adaptation | ~57,000 | 39% |
| Standard plumbing | ~40,000 | 27% |
| **Total (honest)** | **~147,000** | **100%** |

The Skeptic correctly notes that individual components (Thompson's construction, BDD model checking, R-tree indexing, Salsa incremental compilation) are standard and have existing libraries. The difficulty is in the *interfaces* between fields, not in any single component. This is "integration difficulty" — harder than an MS thesis, comparable to a strong systems PhD.

### Assessment

Replication time for the research core (~50K LoC): 10–15 months for a multi-domain expert, 18–24 months for a typical PhD student. Not a 6-month project. The ~100K LoC of supporting infrastructure is competent engineering but not where the research contribution lives. The paper should lead with spatial CEGAR as its most technically impressive contribution.

---

## Axis 3: Best-Paper Potential — **5/10** ⛔

### The math portfolio is B-grade

| Contribution | Grade | Risk | Note |
|---|---|---|---|
| M1: Spatial EC Formalization | C+ | Low | Clean but elementary (IVT under Lipschitz) |
| M2: Geometric Consistency Pruning | B− | Medium | Bounds may not be tight for real scenes |
| M3: Spatial Tractability | B+conditional | **High** | Unproven conjecture, 6–12 month risk |
| M4: Type System Soundness | B | Medium | LP core is standard; CSG boundary adds interest |
| M5: End-to-End Correctness | B− | Low | Routine composition (CompCert methodology) |

**No A-grade mathematical contribution.** The best result (M3) is an unproven conjecture that the project's own math reviewer downgraded from A−conditional to B+conditional, noting "the novelty is the domain identification, not the technique." OOPSLA best papers typically feature at least one result that makes experts say "I didn't know that was possible." None of M1–M5 reaches that bar without M3 delivering.

### The Halide/TVM analogy is aspirational

| Property | Halide | TVM | Choreo |
|----------|--------|-----|--------|
| Domain size | Millions (image processing) | Millions (ML deployment) | ~50K (XR interaction) |
| Math novelty | A (optimal scheduling) | B+ (auto-tuning cost model) | B− average, B+cond best |
| Empirical impact | 5–10× speedup demonstrated | 10× speedup demonstrated | Unknown (0 LoC built) |
| Industry adoption | Google, Adobe, Qualcomm | Amazon, Facebook | Speculative |

The analogy is aspirational, not earned. Best-paper probability: **~10–15%** (conditional on M3 succeeding AND bug-finding exceeding 5 confirmed anomalies). Without both: **~3–5%**.

### The bug-finding evaluation is the stronger leg

If Choreo finds ≥5 interaction protocol anomalies in open-source MRTK/Meta SDK projects, with ≥2 corroborated by existing issue tracker reports, the practical impact argument becomes defensible regardless of M3. This is the more reliable best-paper path. But the target must be calibrated honestly: MRTK has been maintained by Microsoft's XR team for 5+ years with professional QA. The claim that a model-level abstraction (which omits physics, animation, rendering) will find bugs Microsoft missed is extraordinary. Probability of ≥5 corroborated anomalies: ~35–45%.

### What would raise this to 7

1. **M3 delivers** — a tight polynomial-time bound for bounded-treewidth spatial interference graphs, empirically validated on ≥20 real scenes. This alone pushes best-paper potential to 6–7.
2. **Bug-finding exceeds expectations** — ≥8 anomalies, ≥3 acknowledged by platform teams. This pushes to 7 regardless of M3.
3. **Both M3 + strong bug-finding** — pushes to 8. This is the only credible best-paper path.

### Realistic target

Target a **strong accept with distinguished artifact potential** at OOPSLA. Best-paper only if both M3 and bug-finding over-deliver. Prepare a "Choreo without M3" contingency paper as a systems contribution.

---

## Axis 4: Laptop CPU + No Humans — **8/10** ✅

### The pipeline is naturally CPU-native

| Component | Why CPU Is Natural | Performance |
|---|---|---|
| Parsing + type checking | String/graph operations | Milliseconds |
| EC compilation | Graph transformation (Thompson construction) | Linear in spec size |
| R-tree queries | Spatial indexing, cache-friendly STR bulk-loading | Microseconds per query |
| BDD verification | CUDD/BuDDy — decades of CPU optimization | Seconds for ≤20 zones |
| SAT-based BMC | CaDiCaL/MiniSat — mature CPU solvers | Minutes for bounded checking |
| GJK/EPA collision | Purely geometric, no rendering | >10,000 fps single core |
| Headless simulation | Bounding volumes + event dispatch | 1,000-scenario sweep overnight |

This is a genuine architectural strength. The system processes *interaction logic*, not *rendered pixels*. CI/CD integration on standard cloud VMs ($0.008/min, no GPU provisioning) is a real operational advantage.

### The extraction weak point

Automated extraction from Unity C# is hard in the general case (MonoBehaviours, coroutines, ScriptableObjects, visual scripting). But extraction from **MRTK canonical components** — which use standardized patterns with well-defined YAML schemas — is tractable with ~8–12K LoC of targeted extraction tooling, scoped to ~15 common interaction components. This extraction pipeline is not currently in the 147K LoC estimate and must be added.

### Scalability ceiling

Without M3: **~15–20 interaction zones**, verification in seconds to minutes. This covers most MRTK sample scenes (median ~8–12 zones). With M3 (if it delivers): **50–100+ zones**, scaling to production applications. The paper must report this ceiling honestly.

### What keeps this from 9

The extraction pipeline is unscoped and unprototyped. "No human annotation" requires that extraction from MRTK canonical components actually works end-to-end. Until prototyped, this is an assumption, not a fact.

---

## Fatal Flaws

### Genuine Fatal Flaws (2)

**F1: Extraction Fidelity Gap (Severity: HIGH)**  
The entire bug-finding evaluation rests on the assumption that interaction protocols can be faithfully extracted from MRTK C# code into Choreo specifications. If extraction produces models that don't correspond to actual runtime behavior (because Unity's physics, animation, and frame ordering are abstracted away), then "bugs found" may be model artifacts that never manifest in practice. This is the project's existential risk.

*Mitigation*: Scope extraction to MRTK canonical components with well-defined semantics. Validate against ≥10 known bugs from MRTK's GitHub issue tracker. If ≥7/10 known bugs are detected by the verifier, extraction fidelity is empirically validated.

**F2: Scalability Ceiling Without M3 (Severity: MEDIUM-HIGH)**  
Without the spatial tractability theorem (M3, unproven, B+conditional), verification hits a wall at ~15–20 interaction zones. If the paper claims to verify "XR interaction choreographies" broadly, reviewers will ask about production scenes with 50–200 zones. The paper must either (a) prove M3, (b) demonstrate useful bugs exist in small scenes, or (c) honestly scope the contribution.

*Mitigation*: Empirically validate that MRTK sample scenes fall within the tractable range (likely true). Report the ceiling honestly. Implement compositional preprocessing (spatial independence partitioning) as a practical extension even without M3's full theorem.

### Serious Risks (3)

**R1: M3 Tractability Theorem May Fail (~60% risk).** If XR interference graphs don't have bounded treewidth, the theoretical contribution drops to B-grade. Paper survives as systems contribution but best-paper argument evaporates.

**R2: SpatiaLang Differentiation Not Addressed.** A reviewer who knows SpatiaLang could argue Choreo's DSL is incremental. The differentiation (EC semantics, formal verification, headless execution, compilation to verifiable automata) is real but must be made explicit with a comparison table.

**R3: Bug-Finding Target May Be Unreachable.** MRTK has been maintained with professional QA for 5+ years. Finding ≥5 interaction protocol anomalies corroborated by issue tracker reports is achievable but not certain (~35–45% probability). The target must be framed as "anomalies" not "bugs."

---

## Required Amendments

### Must-Have (5)

| # | Amendment | Addresses |
|---|-----------|-----------|
| A1 | **Scope extraction explicitly**: define "supported components" list for MRTK extraction (~15 canonical interaction components). Add extraction pipeline as a subsystem (~8–12K LoC). Prototype on ≥3 MRTK scenes before full build. | F1 |
| A2 | **Reframe bug-finding target**: "≥5 interaction protocol anomalies in open-source XR projects, with ≥2 corroborated by existing issue tracker reports." | R3 |
| A3 | **Add SpatiaLang/MPST/Scenic comparison table** in the best-paper argument / related work. Differentiate on 4 axes: temporal reasoning, formal verification, headless execution, verifiable automata compilation. | R2 |
| A4 | **Report scalability ceiling honestly**: "Without compositional techniques, verification handles scenes with up to ~15–20 interaction zones in minutes. Most MRTK sample scenes fall within this range." Frame M3 as optional upside. | F2 |
| A5 | **Correct LoC and Novel% claims**: Total ~147K (not 153K). ~34% research-novel, ~42% including domain adaptation. EC Engine Novel% corrected from 70% to 50%. | Inflation |

### Should-Have (3)

| # | Amendment | Addresses |
|---|-----------|-----------|
| A6 | **Front-load M3 feasibility**: compute treewidth of 20 real MRTK interaction graphs in weeks 1–2. If treewidth ≤ 5, pursue M3 aggressively. If > 10, redirect to strengthening M4 (type system soundness). | R1 |
| A7 | **Elevate accessibility auditing**: promote from bullet point to named contribution. Frame as "first tool that can prove an XR interaction is accessible via alternative input modalities." | Axis 1 |
| A8 | **Define M3-failure contingency paper**: Track A (M3 succeeds) → "Decidable Verification" narrative. Track B (M3 fails) → "Bug-Finding via Spatial EC Compilation" systems paper. | R1 |

---

## Expert Signoff

| Expert | Verdict | Score | Key Concern |
|--------|---------|-------|-------------|
| Independent Auditor | CONDITIONAL PASS | 6.0/10 | Extraction unscoped, evaluation speculative |
| Fail-Fast Skeptic | **ABANDON** (dissent) | 2.8/10 | No market, no A-grade math, speculative eval |
| Scavenging Synthesizer | PROCEED | 7.5/10 | All flaws have actionable amendments |
| **Team Lead** | **CONDITIONAL CONTINUE** | **6.25/10** | 2 fatal flaws mitigable; 8 amendments required |

**Consensus**: 2-1 CONDITIONAL CONTINUE (Auditor + Synthesizer approve with conditions; Skeptic dissents).

### Skeptic's Dissent (recorded)

"The graveyard of academic tools is full of 'firsts' that nobody asked for. This project requires ~24 months to deliver ~147K LoC of which ~33K is genuinely novel algorithms. The expected return — a paper with medium theory and speculative impact — does not justify the investment. P(best-paper) < 5%. P(any publication) ≈ 40%. P(abandonment mid-development) ≈ 20%. Salvage path: 15K-LoC geometric pruning library + hand-written verification of 10 MRTK samples."

### Lead's Response to Dissent

The Skeptic's probability estimates are harsh but within the credible range for a pessimistic prior. However, the Skeptic undervalues three factors: (1) the spatial CEGAR loop is genuinely novel and would survive independent scrutiny, (2) the CPU-only architecture is a real engineering insight that enables CI/CD integration, and (3) the "missing layer" structural argument is compelling to PL reviewers even if the XR market is small. The project is risky but not reckless. CONDITIONAL CONTINUE with the 8 amendments above.

---

## Binding Conditions for Implementation

1. **Before writing compiler code**: Prototype extraction on 3 MRTK sample scenes. If extraction produces models that detect ≥1 known bug from MRTK's issue tracker, proceed. If extraction fails to produce faithful models, reassess the evaluation strategy.

2. **Within first 2 weeks**: Compute treewidth of ≥20 real MRTK interaction graphs. This determines whether M3 is worth pursuing (treewidth ≤ 5) or should be redirected to M4 (treewidth > 10).

3. **Within first 3 months**: Prototype the spatial CEGAR loop and measure pruning ratio |C|/|2^P| on ≥5 benchmark scenes. If pruning < 3×, the verification backend's novelty claim weakens and the paper must lean harder on bug-finding.

4. **At month 6 checkpoint**: ≥3 interaction protocol anomalies found in real XR projects, OR demonstrable M3 progress. If neither, trigger reassessment.

---

*Signed: Verification Panel Lead*  
*Assessment method: 3-expert independent proposals → adversarial cross-critique → lead synthesis with rulings on 5 disputed axes*  
*Total expert-hours: ~12 (3 independent proposals + 1 cross-critique round + 1 synthesis)*
