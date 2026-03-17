# Final Approach: Coverage-Certified XR Accessibility Verifier

## One-Sentence Summary

A phased system that ships an interval-arithmetic XR accessibility linter first (Months 1–2), then layers a sampling-guided symbolic engine with formally grounded coverage certificates (Months 2–6), producing the first tool that combines real-time developer feedback with probabilistic formal guarantees of spatial accessibility across parameterized human bodies.

## Architecture Overview

The system has four subsystems arranged in a pipeline with two verification tiers:

1. **Scene Ingestion Layer.** Parses Unity `.unity` YAML, `.prefab` hierarchies, and C# MonoBehaviour scripts. Extracts interaction elements (position, activation volumes, interaction type) via pattern-matching against standard Unity XR Interaction Toolkit idioms (OnSelectEntered, OnActivated, OnHoverEntered). Complex multi-step interactions require developer-provided DSL annotations. No claim of general-purpose state machine extraction—Rice's theorem makes that undecidable.

2. **Kinematic Model Layer.** Wraps Pinocchio for forward-kinematics evaluation. Parameterizes a 7-DOF arm model (3-DOF shoulder, 1-DOF elbow, 3-DOF wrist) by ANSUR-II/CAESAR anthropometric distributions (stature, arm length, shoulder breadth, ROM limits). Exposes two interfaces: (a) point evaluation `FK(θ, body_params) → end-effector pose` for the sampler, and (b) affine-arithmetic evaluation `FK_interval(θ_range, body_range) → enclosure` for the linter.

3. **Tier 1: Interval-Arithmetic Linter.** Computes conservative reachability envelopes via affine arithmetic over the kinematic chain. For each scene element, checks whether the envelope intersects the activation volume. Runs in <2 seconds in the Unity editor. Produces immediate visual annotations: green (definitely reachable for target population), red (definitely unreachable for some subpopulation with percentage estimate), yellow (inconclusive—envelope intersects boundary). This is B's core contribution, shipped first.

4. **Tier 2: Sampling-Symbolic Engine with Coverage Certificates.** Runs in CI/CD or on-demand (budget: 5–15 minutes). Phase one: adaptive stratified sampling over the anthropometric × joint-angle parameter space, concentrating samples near the accessibility frontier estimated from Tier 1's interval analysis. Phase two: targeted SMT verification (Z3/CVC5 on linearized kinematics) of boundary regions where sampling is inconclusive. Phase three: certificate assembly—combines sampled regions (with verdicts) and symbolically verified regions (with proofs) into a coverage certificate ⟨S, V, ε, δ⟩ bounding P(undetected bug) ≤ ε with confidence 1−δ.

**Data flow:** Unity Scene → Scene Ingestion → (interaction graph, element positions, activation volumes) → Kinematic Model → Tier 1 (fast, conservative) → Tier 2 (thorough, certificate-producing). Tier 1 results seed Tier 2's adaptive sampling by identifying frontier regions. The certificate is serialized as a machine-readable JSON artifact for CI/CD integration.

## Value Proposition

**Primary users:** XR developers building accessibility-sensitive applications—enterprise training (Boeing, Siemens), surgical simulation (Osso VR), education, and public-facing XR experiences subject to Section 508/ADA Title I or the EU Accessibility Act.

**Secondary users:** Platform holders (Meta Accessibility Engineering, Apple Accessibility) who could mandate the linter as a store submission gate.

**Honest market assessment:** The addressable developer population is 30–50K. The intersection of "motor disability" and "owns XR headset" is currently tens of thousands of humans. No XR-specific accessibility regulation exists today—the EU Accessibility Act targets web/mobile, not XR explicitly. Zero developer demand has been validated. The value bet is that: (a) the XR accessibility tooling market is completely empty, making even a modest tool infinitely better than the status quo; (b) zero-configuration, 2-second linting in the editor has adoption dynamics similar to ESLint—developers discover it, not seek it; (c) regulatory trajectory is toward XR inclusion, and first-mover advantage matters.

**The demand risk is existential across all approaches.** Gate D7 tests it cheaply at Month 3.

## Technical Strategy

### Phase 1: Interval-Arithmetic Linter (Months 1–2)

This is Approach B's substrate, built first because it has 85% 6-month prototype probability, validates demand, and produces a useful artifact regardless of research outcomes.

**What ships:** A Unity editor plugin that, on scene load or element movement, computes affine-arithmetic reachability envelopes for each interactable element across the 5th–95th percentile anthropometric range. Visual annotations in the editor. Population-fraction estimates for flagged elements using Chebyshev bounds on ANSUR-II marginal distributions.

**Key technical decisions:**
- **Language:** C++ native plugin for the affine-arithmetic engine (performance-critical path), C# for Unity editor integration and visualization. Not Rust—Unity's native plugin ecosystem is C/C++-centric and the FFI is better documented.
- **Affine arithmetic, not naïve intervals.** Naïve interval arithmetic produces 10–100× over-approximation on 7-joint chains. Affine arithmetic with noise-symbol tracking reduces wrapping to 3–5× for typical XR joint ranges (<60° per joint). The 2× wrapping bound claimed in Approach B is optimistic for 7-joint chains (Math Assessor flagged this; Merlet 2004 benchmarks suggest 3–5×). We target <5× wrapping factor, not <2×.
- **Single-step interactions only in Phase 1.** Multi-step envelope composition (B2) is deferred to Phase 2. This keeps Phase 1 scope tight (~12–15K LoC novel) and avoids the wrapping-compounding problem.
- **Incremental recomputation.** Dependency graph tracks which envelopes are affected by scene modifications. Only dirty envelopes are recomputed on change.

**Kill gate D1 (Month 1):** Measure affine-arithmetic wrapping factor on a 4-joint revolute chain with ±30° joint ranges. If wrapping factor >5×, switch to Taylor-model propagation immediately. If >10× with Taylor models, the linter's false-positive rate will exceed 20% and the tool is dead—pivot to lookup-table approach (500 LoC, catches obvious violations only, no publication story).

### Phase 2: Sampling-Guided Symbolic Engine (Months 2–4)

This is Approach C's core. It builds on the Tier 1 infrastructure (kinematic model, scene parser) and adds the sampling + SMT hybrid.

**What ships:** A command-line/CI tool that takes a parsed Unity scene and produces a coverage certificate.

**Key technical decisions:**
- **Adaptive stratified sampling.** Initial uniform sampling (100K samples, ~10s) over the anthropometric parameter space identifies the accessibility frontier. Subsequent samples concentrate within a Δ-neighborhood of the frontier, where Δ is determined by the local Lipschitz estimate. Uses Latin hypercube sampling within strata for low-discrepancy coverage.
- **Frontier seeding from Tier 1.** The interval linter's "yellow" (inconclusive) regions directly identify where the frontier lies in parameter space. This eliminates the cold-start problem for frontier-adaptive sampling—a concrete advantage of building on B's substrate.
- **Linearized-kinematics SMT encoding.** Taylor-expand FK around each sample center. Approximation error bounded by C2: ‖f(θ) − f_lin(θ)‖ ≤ C · Δ² · L_max^k. Each SMT query verifies a hypercube of radius Δ_max around the linearization point, where Δ_max is the largest Δ for which the error bound keeps the verification sound. For a 7-DOF arm with L_max = 0.4m, Δ_max ≈ 5–15° per joint depending on chain geometry.
- **SMT solver:** Z3 on QF_LRA (quantifier-free linear real arithmetic) after linearization, not QF_NRA. Linearization converts the nonlinear kinematic problem into a linear one within each soundness envelope. Individual queries should complete in <100ms. Budget: ~1000 SMT queries per scene within the 10-minute budget.
- **Timeout-and-skip for SMT.** Queries exceeding 2s are killed. The corresponding region remains "sampled only" (not symbolically verified), which loosens ε but preserves certificate soundness.
- **Multi-step restriction: ≤3 steps initially (Amendment D5).** For k-step interactions, the trajectory space is k×d dimensional. At k=3, d=7, dimension is 21—manageable with ~500K stratified samples. At k=5, dimension is 35—certificates will be loose (ε > 0.1). Defer 5+-step support to future work.

**Implementation:** Python orchestrator + C++ kinematic evaluation (shared with Tier 1) + Z3 via Python bindings (pyz3). ~15–20K novel LoC.

### Phase 3: Coverage Certificate Framework (Months 3–6)

This is C's crown jewel—the novel formal contribution that makes or breaks the research story.

**What ships:** The certificate engine that aggregates sampling and SMT results into a formally grounded ⟨S, V, ε, δ⟩ certificate.

**The formal object.** A coverage certificate for scene Ω with parameter space Θ consists of:
- S ⊂ Θ: sampled region with pointwise verdicts (accessible/inaccessible)
- V ⊂ Θ: symbolically verified region with SMT proofs
- ε ∈ [0,1]: upper bound on P(∃ undetected bug in Θ \ (S ∪ V))
- δ ∈ [0,1]: confidence level (probability that the ε bound holds)

The soundness theorem states: if the accessibility frontier ∂F in Θ satisfies L-Lipschitz continuity (no isolated-point failures), and if the sampling density in Θ \ V exceeds n_min(L, ε, δ), then P(bug in Θ \ (S ∪ V)) ≤ ε with probability ≥ 1−δ. The proof combines stratified-sampling Hoeffding bounds with volume accounting for symbolically eliminated regions.

**Addressing the Lipschitz problem (Skeptic's Attack 1 on C).** The Skeptic correctly identifies that knife-edge failures (button exactly at a reachability boundary, grasp requiring exactly 88° wrist pronation) violate Lipschitz and the certificate is vacuous for these cases. Our response:

1. **Lipschitz violations are not pathological but are detectable.** Joint-limit boundaries create predictable non-Lipschitz frontiers at known parameter values. The certificate engine identifies these boundaries from the kinematic model and excludes them from the Lipschitz-dependent bound, reporting them separately as "boundary-condition warnings" that require explicit verification.
2. **The certificate reports Lipschitz violations, not ignores them.** For each element, the tool checks whether the activation volume intersects the kinematic singularity surface or joint-limit boundary in parameter space. Intersections are flagged as "Lipschitz-violation regions" with explicit descriptions. The certificate's ε bound applies only to the Lipschitz-regular portion; the violation regions are enumerated.
3. **Practically, measure-zero failures affect measure-zero populations.** A button at the exact boundary of a reachability envelope is inaccessible for a set of body parameterizations with measure zero (or near-zero). The practical concern is ε-neighborhoods of boundaries, which *are* Lipschitz-regular. We handle these by construction.
4. **Amendment D4 validates this.** By Month 3, we formally characterize which scene configurations produce Lipschitz violations and measure their frequency on 50+ benchmark scenes. If >20% of scenes have violations that escape the boundary-detection heuristic, the certificate's applicability claim is weakened and we downscope.

**Addressing the Clopper-Pearson benchmark (Skeptic's Attack 2 on C).** The Skeptic asks: how is this better than "we sampled a lot"? The answer has two parts:

1. **Quantitative:** The coverage certificate's ε bound incorporates symbolically verified volume, which Clopper-Pearson does not. If 30% of the frontier is SMT-verified, the effective sample space is 70% of the original, tightening ε by ~1.4×. With targeted SMT on the highest-uncertainty regions, the improvement over Clopper-Pearson should be 3–10× (tested at gate D3).
2. **Structural:** The certificate provides a spatial map of verification status—which parameter regions are sampled, which are symbolically proved, which are unverified. Clopper-Pearson gives a single global number. The spatial map enables targeted follow-up: "increase your verification budget by X minutes to close the gap in this region."

**Amendment D3 is binding.** If the certificate doesn't improve ε by ≥5× over Clopper-Pearson from the same sample count at Month 2, the theoretical contribution is marginal. Downscope to tool paper.

### Phase 4: Integration and Evaluation (Months 6–9)

**Combined system evaluation.** The full Tier 1 + Tier 2 pipeline runs on:
- 500 procedurally generated scenes (controlled treewidth, spatial distribution, object count)
- 10+ real Unity XR scenes (5 from open-source projects, 5 hand-crafted for complexity)
- Monte Carlo baseline comparison (1M stratified samples via Pinocchio)

**Developer study (Months 7–9).** 15–22 XR developers use the linter-equipped Unity editor vs. unassisted development on matched accessibility tasks. Measure: accessibility violations found per hour, time to first fix, subjective usability (SUS). IRB approval required—budget 6–8 weeks for recruitment, IRB, sessions, analysis.

**Two-paper preparation.** Structure all evaluation data to serve both papers (see Best-Paper Strategy).

## Subsystem Breakdown with LoC Estimates

| Subsystem | Estimated LoC | Genuinely Difficult LoC | Novelty Assessment |
|-----------|--------------|------------------------|-------------------|
| **Affine-arithmetic FK engine** (C++) | 5–8K | 3–5K | **Moderate-hard.** Affine arithmetic with noise-symbol tracking through revolute-chain composition. The wrapping-reduction strategy for >4 joints is the novel part. Known technique (Stolfi & de Figueiredo) in new domain. |
| **Unity scene parser** (C#) | 4–6K | 1–2K | **Low.** YAML parsing + XR Interaction Toolkit pattern matching. State machine extraction limited to standard idioms. |
| **Unity editor plugin** (C#/C++) | 4–6K | 1–2K | **Low.** Native plugin bridge, incremental recomputation, visual annotations. Solid engineering, not algorithmic novelty. |
| **Kinematic model + anthropometric DB** | 3–5K | 1–2K | **Low.** Pinocchio wrapping + ANSUR-II parameterization interface. |
| **Adaptive stratified sampler** (Python/C++) | 5–8K | 3–5K | **Moderate.** Frontier-adaptive Latin hypercube sampling. Novel frontier-seeding from Tier 1 results. |
| **Linearized-kinematics SMT encoding** | 5–8K | 2–4K | **Moderate-hard.** Taylor expansion, soundness envelope computation (C2), SMT-LIB generation. Well-understood techniques, careful engineering. |
| **Coverage certificate engine** (Python) | 6–10K | 5–8K | **Hard—crown jewel.** Certificate data structure, Lipschitz estimation, volume accounting, soundness proof realization, certificate serialization/verification. Novel formal object. |
| **Sampling-symbolic handoff controller** | 3–5K | 2–4K | **Hard.** Budget allocation, frontier-region identification, SMT dispatch, timeout management. Novel control logic. |
| **Population-fraction reporting** | 2–3K | 0.5–1K | **Low.** Chebyshev bounds on marginal distributions. Textbook. |
| **Counterexample reporting + certificate UX** | 3–4K | 1–2K | **Low-moderate.** Plain-language translation of certificates. UX engineering. |
| **Benchmarks + procedural scene generation** | 3–5K | 1–2K | **Low.** Infrastructure. |
| **Total** | **43–68K** | **21–37K** | |

The Difficulty Assessor estimated ~28–40K genuinely novel LoC for Approach C alone and ~15–22K for Approach B. Our combined B+C estimate of 43–68K total (21–37K difficult) is consistent: the shared infrastructure (kinematic model, scene parser) is counted once, and Phase 1 (B's substrate) contributes ~12–15K to the total.

## New Mathematics Required

| ID | Contribution | Difficulty | Load-Bearing? | Novelty Type | Risk of Failure |
|----|-------------|-----------|---------------|--------------|-----------------|
| **C1** | **Coverage certificate soundness theorem.** Proves ⟨S, V, ε, δ⟩ certificate is sound under Lipschitz assumption: P(bug in Θ\(S∪V)) ≤ ε with prob ≥ 1−δ. Combines stratified-sampling Hoeffding bounds with SMT-verified volume elimination. | **B+** | **Maximally.** Without this, the system is "Monte Carlo with SMT on the side"—no publication story, no formal guarantee. | **(a)/(b) boundary.** Genuinely new formal object assembled from known components. The individual pieces (stratified sampling bounds, SMT region elimination) exist; their composition into a coverage certificate for parameter-space verification is new. Conceptual distance from statistical model checking (Younes & Simmons) is shorter than originally claimed—C1 adapts the paradigm from stochastic temporal properties to deterministic parameter-space verification. The Math Assessor grades this B+, not A−. We accept this grading. | **30–35%.** Certificate will likely be *correct* but may not be *tight* enough (ε > 0.05 on typical scenes). Graceful degradation: tool works as MC+SMT without the formal guarantee. |
| **C2** | **Linearization soundness envelope.** Bounds ‖f(θ)−f_lin(θ)‖ ≤ C·Δ²·L_max^k for k-joint revolute chains. Explicit constant C as function of chain geometry. | **C+** | **Supporting.** Determines SMT query granularity and thus total SMT cost. If envelope is too small, need exponentially many queries. | **(b) Known Taylor-remainder analysis** applied to kinematic chains. Standard technique, specific constant is useful but not deep. | **10%.** This is a careful computation, not a hard proof. Risk is that the bound is looser than hoped, requiring more SMT queries. |
| **C3** | **Optimal sampling-symbolic budget allocation.** Given compute budget T, derives (n_samples, n_smt_regions) minimizing ε. Convex when frontier is Lipschitz, admitting closed-form solution. | **C** | **Mild.** Guides internal resource allocation. Without it, use a heuristic—certificate still works, just looser. | **(b) Known optimization techniques** applied to a new resource-allocation problem. Belongs in an appendix, not a theorem statement. | **5%.** Straightforward convex optimization. |
| **C4** | **Tier 1 completeness gap bound.** Characterizes the maximum undetectable bug radius r_max as a function of wrapping factor w: any accessibility violation with spatial margin ≤ r_max/w is guaranteed detectable by interval methods; violations with margin > r_max/w may be missed. Provides a sharp boundary between bugs interval methods can and cannot catch. | **B−** | **Moderate.** Gives the UIST paper a precise theoretical characterization of Tier 1's limitations, preempting "what can't your linter find?" reviewer attacks. | **(b) Known interval-analysis error bounds** applied to the kinematic reachability domain. The relationship between wrapping factor and detectable spatial margin is new. | **10%.** Requires careful bounding of FK error propagation, but the structure is well-understood. |
| **B1** | **Wrapping-factor analysis for affine-arithmetic FK.** Bounds over-approximation factor of affine arithmetic through k-joint revolute chains as function of joint-range width and chain length. | **C+** | **Supporting for Tier 1 quality.** Determines false-positive rate. Without it, Tier 1 works but we can't predict its precision. | **(b) Known technique** (affine arithmetic) with **domain-specific wrapping analysis.** A lemma in a paper, not a theorem. The flat 2× bound originally claimed is too optimistic; the real bound is f(k) where f grows sublinearly in chain length k. | **20%.** Risk is the bound is worse than 5× for 7-joint chains, making Tier 1 imprecise. Tested at gate D1. |

**Crown jewel: C1.** Honest grade: B+, not A−. The generalizability argument (coverage certificates apply to any parameterized verification domain—robotics, drug dosage, autonomous vehicles) is the strongest aspect and the CAV paper's primary hook. The XR application is the case study; the framework is the contribution.

## Best-Paper Strategy

### Two-Paper Strategy (preserved from depth check, adapted for C-on-B)

**Paper 1: Theory paper — CAV 2026 or TACAS 2026**

- **Title framing:** "Coverage Certificates for Parameter-Space Verification: Combining Statistical Sampling with Symbolic Proofs"
- **Core contribution:** C1 (coverage certificate soundness), presented as a general framework for verifying properties across parameterized families of systems. XR accessibility is the motivating application and primary case study, but the framework applies to any domain with parameterized verification (robotic workspace certification, medical device safety, autonomous driving scenario coverage).
- **Key sections:** Formal definition of coverage certificates, soundness proof under Lipschitz assumption, comparison with statistical model checking (Younes & Simmons, Legay et al.), experimental evaluation showing ε improvement over Clopper-Pearson, scalability analysis across parameter-space dimension.
- **Reviewer anticipation:** (1) "Lipschitz is too strong"—address with boundary-detection + explicit violation reporting. (2) "How much better than just sampling?"—D3 benchmark data showing ≥5× improvement. (3) "Linearization makes SMT unsound at scale"—C2 bounds + empirical query-count data.
- **Acceptance probability:** 20–30% at CAV (conditional on ε being meaningfully tight). Coverage certificates are a new enough concept to attract interest, but CAV's formal-methods community will scrutinize the Lipschitz conditioning.

**Paper 2: Tool paper — UIST 2026 or ASSETS 2026**

- **Title framing:** "AccessLint-XR: Real-Time Spatial Accessibility Verification for Mixed Reality Development"
- **Core contribution:** First XR accessibility linter. Developer study showing measurable improvement in accessibility violations detected per hour. The coverage certificate is presented as a CI/CD feature, not the intellectual centerpiece—the tool's usability and impact are the story.
- **Key sections:** System architecture, Unity editor integration, developer study (15–22 participants), comparison with unassisted development, real-scene case studies, false-positive rate analysis.
- **Reviewer anticipation:** (1) "Where are the disabled participants?"—supplement ANSUR-II data story with explicit limitations section; collaborate with disability advocacy org for feedback if possible (Amendment D6). (2) "False-positive rate?"—empirical measurement on real scenes. (3) "Math is thin"—correct; this is a systems paper, not a theory paper.
- **Acceptance probability:** 25–35% at UIST, 15–25% at ASSETS (ASSETS requires deeper disability-community engagement).

### Why not A's venue (CAV for SE(3) zone abstraction)?

The depth check's A5 proposed a CAV paper on M2 (chart-decomposed CAD on SE(3)). This is abandoned because: (a) M2 is an unproven conjecture with 35–40% failure probability, (b) even if proved, CAD at dimension 7–10 is likely intractable, making the result a beautiful theorem with no practical consequence, (c) the 2-month time-box for M2 research produces either a standalone algebraic geometry result (publishable independently, outside this project's scope) or nothing. We do not carry M2 as a project dependency.

## Hardest Technical Challenges

### Challenge 1: Coverage Certificate Tightness

**The problem:** The coverage certificate's ε bound depends on the Lipschitz constant L of the accessibility frontier, which is unknown and must be estimated. Conservative L (large) → useless ε (>0.1). Aggressive L (small) → unsound certificate. This is the fundamental chicken-and-egg: you need dense frontier samples to estimate L, but L tells you where to sample densely.

**Mitigation:**
1. **Local Lipschitz estimation.** Estimate L locally in each stratum from nearby sample pairs, rather than globally. Use the maximum local L across strata as a conservative-but-tighter global bound.
2. **Frontier seeding from Tier 1.** The interval linter identifies frontier regions before any sampling begins, breaking the chicken-and-egg by providing an independent frontier estimate.
3. **Cross-validation.** Hold out 20% of samples. Estimate L from 80%, compute predicted verdicts for the 20%, measure prediction error. If prediction error exceeds the certificate's claimed bound, the L estimate is too aggressive—inflate and re-certify.
4. **Gate D2 tests this empirically at Month 2.** If ε > 0.1 on a 10-object benchmark with 5-minute budget, the certificate is not useful enough. Kill the certificate framework; publish B as standalone tool.

### Challenge 2: SMT Solver Performance Variability

**The problem:** Z3 on linearized kinematic constraints has unpredictable runtime—0.01s to 100s depending on constraint structure. If median query time exceeds 1s, the 10-minute budget supports only ~600 queries, and frontier coverage is sparse.

**Mitigation:**
1. **Linearization converts NRA to LRA.** Within the soundness envelope, all constraints are linear. Z3 on QF_LRA is fast and predictable (<100ms per query in typical cases).
2. **Aggressive timeouts (2s).** Queries exceeding 2s are killed. Their regions remain sampling-only, loosening ε but keeping the certificate sound.
3. **Query prioritization.** Allocate SMT budget to the highest-uncertainty frontier regions first (those with the most conflicting sample verdicts). Diminishing returns: the first 100 queries eliminate more uncertainty than the next 1000.

### Challenge 3: False-Positive Rate of Tier 1 Linter

**The problem:** If affine-arithmetic wrapping produces >5× over-approximation on 7-joint chains, the linter flags too many elements as "potentially inaccessible" (false positives >15%), and developers disable it.

**Mitigation:**
1. **Subdivision on critical parameters.** When envelope width exceeds a threshold, subdivide the anthropometric parameter space into 2–8 sub-ranges per dimension. At 3 subdivided dimensions × 4 sub-ranges = 64 sub-problems, total compute remains <2s.
2. **Joint-range sensitivity.** Shoulder rotation ranges span 120°+, which is where affine arithmetic breaks down. For wide-range joints, switch to piecewise-affine evaluation (split the joint range into 2–3 sub-intervals, evaluate each, take union).
3. **Confidence levels in annotations.** Instead of binary red/green, report "inaccessible for X% of population with Y confidence." Developers tolerate "might be inaccessible for 2% of users" better than a flat red flag.
4. **Gate D1 at Month 1.** If wrapping >5× on 4-joint chain after mitigations, switch to Taylor models.

## Evaluation Plan

### Benchmarks

1. **Procedural benchmark suite (500 scenes).** Parameterized by: object count (5, 10, 30, 50), spatial distribution (clustered, uniform, adversarial), interaction type (single-step reach, 2-step grasp-then-reach, 3-step sequential), anthropometric range (5th–95th percentile). Bug injection: 10% of scenes have injected spatial accessibility bugs at varying severity (obvious: button at 2.5m; subtle: grasp requiring 89° pronation where 90° is the 5th-percentile limit).

2. **Real XR scenes (10+).** 5 from open-source projects (Mozilla Hubs, Unity XR Interaction Toolkit samples, A-Frame examples). 5 hand-crafted for realistic complexity (dense control panel with 50 buttons, multi-step medical procedure, collaborative assembly task).

3. **Anthropometric data: ANSUR-II + supplementary.** ANSUR-II is the primary data source. Limitation acknowledged: this is a US military dataset of primarily able-bodied adults. Supplement with published disability-specific ROM data (Boone & Azen 1979, Soucie et al. 2011 for normative ROM; disability-specific data from rehabilitation literature). Document the gap explicitly—no tool can solve the data-availability problem, but it must not pretend the problem doesn't exist (Amendment D6).

### Baselines

1. **Stratified Monte Carlo (1M samples).** The mandatory baseline. Samples from ANSUR-II distributions, FK evaluation via Pinocchio, reachability check per element. Report: detection rate, runtime, false-positive rate.
2. **Clopper-Pearson confidence intervals from MC.** For each element, compute the 99% Clopper-Pearson upper bound on failure rate from the MC sample. Compare directly with the coverage certificate's ε.
3. **Lookup table (20 discrete percentiles).** The Skeptic's 500-LoC alternative. Pre-compute reachability for 20 anthropometric percentiles, check membership. Report detection rate to quantify how much the interval arithmetic and coverage certificates actually add. **Explicitly measure Tier 1's marginal detection rate over lookup-table baseline** — if Tier 1 catches <10% more bugs than the lookup table, the engineering complexity of affine arithmetic is harder to justify in the UIST paper. Not a kill gate, but must be reported honestly.

### Metrics

- **Detection rate:** fraction of injected bugs detected (target: >95% for Tier 1, >97% for Tier 2)
- **False-positive rate:** fraction of accessible elements flagged as inaccessible (target: <15% for Tier 1, <5% for Tier 2)
- **Coverage certificate ε:** upper bound on P(undetected bug) (target: <0.01 for single-step on 30-object scene in 10 minutes)
- **ε improvement over Clopper-Pearson:** coverage certificate ε divided by MC Clopper-Pearson bound from same sample count (target: ≥5× improvement)
- **Runtime:** Tier 1 <2s, Tier 2 <10 minutes on a laptop (Apple M-series, 16GB RAM)
- **Developer study:** accessibility violations found per hour (target: ≥2× improvement over unassisted)

## Feasibility Gates and Kill-Chain

The kill-chain merges the original depth check amendments (A1–A7) with the debate amendments (D1–D7). Where they overlap, the stricter criterion applies.

| Gate | Deadline | Criterion | Kill Action |
|------|----------|-----------|-------------|
| **D1** (from A1, adapted) | **Month 1** | Affine-arithmetic wrapping factor on 4-joint chain with ±30° ranges ≤ 5×. If >5×, Taylor-model wrapping ≤ 5×. | If wrapping >10× with Taylor models → **ABANDON** linter approach. Fall back to lookup-table (500 LoC). No publication story. |
| **D2** | **Month 2** | Coverage certificate prototype demonstrates ε < 0.05 on a 10-object benchmark with 5-minute compute budget. | If ε > 0.1 → **ABANDON** certificate framework. Publish Tier 1 linter as standalone UIST tool paper. |
| **D3** | **Month 2** | Coverage certificate ε improves over Clopper-Pearson bound from same sample count by ≥ 5×. | If improvement < 3× → Certificate adds marginal theoretical value. **DOWNSCOPE** to tool-only paper; certificate becomes a supplementary result, not the main contribution. |
| **D4** | **Month 3** | Lipschitz violation characterization: formally identify which scene configurations violate Lipschitz; measure frequency on 50+ benchmark scenes. | If >20% of scenes have undetectable Lipschitz violations → **DOWNSCOPE** certificate applicability claim. Report as conditional result with explicit scope limitations. |
| **D7** (from A7) | **Month 3** | Publish Tier 1 linter as free Unity Asset Store plugin. Track downloads and GitHub issues for 8 weeks. Survey ≥20 XR developers with structured feedback. | If <100 downloads AND <25% of surveyed developers express interest → Document as known limitation. Proceed with theory paper regardless (demand is not required for the CAV contribution). |
| **D5** | **Month 4** | Multi-step coverage certificates (≤3 steps) achieve ε < 0.1 on benchmark scenes within 15-minute budget. | If ε > 0.2 for 3-step interactions → **RESTRICT** to single-step verification in the paper. Defer multi-step to future work. |
| **A3** (from depth check) | **Month 6** | Formal verification (Tier 2) detects ≥10% of bugs missed by 1M-sample MC. | If marginal detection <5% → The verification adds insufficient value over MC. **DOWNSCOPE** to theory paper (certificates as paradigm) + Tier 1 tool paper. |
| **A4** (from depth check) | **Month 6** | ≥10 real XR scenes processed; ≥5 complete Tier 2 analysis within budget. | If <5 real scenes or tool fails on all → **DOWNSCOPE** to procedural benchmarks only. Acknowledge as limitation. |
| **D6** | **Month 6** | Anthropometric data audit complete. Disability-specific ROM data incorporated or limitation documented. | If no disability-specific data available → **DOCUMENT** limitation explicitly in both papers. Do not claim the tool "verifies accessibility for disabled users" without disability-specific kinematic data. |

**Escape hatches at each gate:**
- After D1 kill: No publication. Lookup table is a utility, not research.
- After D2 kill: UIST tool paper on Tier 1 linter. Viable but thin.
- After D3 kill: UIST tool paper (strong) + downscoped CAV paper (certificates as supplementary result, not headline).
- After A3/A4 kill: CAV theory paper on coverage certificates (domain-independent) + UIST tool paper on Tier 1 linter. Two modest papers instead of one strong one.

## Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 6/10 | Fills a completely empty niche (zero existing XR accessibility tools). Zero-config Unity linter has realistic adoption path. Coverage certificates provide formal-ish guarantees compliance teams can use. Loses points: market is small (30–50K developers), no demand signal validated, simpler alternatives capture most value. The lookup table gets you 70% of the way at 1% of the cost—our value add is in the remaining 30% plus the formal guarantee. |
| **Difficulty** | 6/10 | Combined B+C has ~43–68K total LoC, ~21–37K genuinely difficult. Coverage certificate theory (C1) is a real intellectual contribution. No open mathematical problems (unlike A), but the certificate-tightness question is genuinely uncertain. The Difficulty Assessor rated B at 4.5 and C at 6.5; the combined approach is closer to C's difficulty because the certificate is the hard part. |
| **Potential** | 6/10 | Coverage certificates generalize beyond XR—genuine paradigm contribution. But: Math Assessor grades C1 at B+ not A−, Lipschitz limitation invites reviewer pushback, and the XR application domain is niche. Two-paper strategy gives two shots at publication. Best-paper probability at CAV: ~5–8% (the concept is elegant but the execution must be flawless and ε must be meaningfully tight). |
| **Feasibility** | 7/10 | 6-month prototype probability: ~60–65% (B substrate at 85% × C prototype at ~75%, adjusted downward for shared infrastructure coupling—Pinocchio integration, ANSUR-II formatting, and scene parser issues can stall both tracks). 12-month full system: ~45%. No open math problems. Primary risk: certificate looseness (D2, D3 gates). Laptop-feasible for single-step and 2–3 step interactions. 16GB constraint is not fatal but is binding for complex multi-step scenes. |
| **Composite** | **25/40** | Improvement over original 20/40, driven by higher feasibility (7 vs. 5) and maintained difficulty. Honest: not a slam-dunk project, but the risk-adjusted expected value is positive with the kill-chain enforced. |

## What We're NOT Doing (and Why)

1. **Chart-decomposed CAD on SE(3) (Approach A's M2).** Abandoned because: (a) unproven conjecture with 35–40% failure probability on the core theorem, (b) CAD at dimension 7–10 is likely intractable (doubly exponential complexity, QEPCAD can't handle it), (c) the entire Tier 2–3 formal verification architecture depends on this single open problem, (d) engineering scope is 85–110K novel LoC requiring rare dual expertise in computational algebraic geometry + formal verification, (e) compound fatal-flaw probability ~85%. If someone wants to pursue M2 as a standalone algebraic geometry result, that's a separate project.

2. **BDD-based zone-graph model checking.** Without the zone abstraction from CAD, there is no zone graph to encode in BDDs. The BDD layer, CUDD integration, and variable-ordering heuristics are all downstream of M2 and are cut with it.

3. **CEGAR loop over SE(3).** Same dependency on M2. Also: SE(3) counterexample concretization requires SMT over nonlinear real arithmetic with trigonometric constraints—Z3/CVC5 choke on these, and dReal is slow. The CEGAR convergence question is unresolved.

4. **Full PGHA formalism as verification target.** The PGHA semantics (M1) were designed for the CAD-based approach. Our system verifies parameter-space accessibility properties directly via sampling + SMT, without constructing an explicit hybrid automaton. The PGHA formalism may appear in the background section of the theory paper for context, but it is not a computational dependency.

5. **WebXR and OpenXR parser support.** Unity dominates XR development (>60% market share). Multi-format parsing adds ~15K LoC with minimal incremental value for a research prototype. Deferred to future work (Amendment A6).

6. **Multi-step interactions beyond 3 steps.** The curse of dimensionality in trajectory space (35D+ for k=5) makes tight coverage certificates unachievable at practical compute budgets. Honestly deferred rather than promised and under-delivered (Amendment D5).

7. **3D counterexample visualization.** Replaced with text reports + 2D SVG body-configuration diagrams. Saves ~8K LoC and a WebGL dependency (Amendment A6).

8. **"Defines a new subfield" framing.** Dropped entirely. This invites hostile reviewer scrutiny and signals overclaiming. One novel formal object (coverage certificates) and one practical tool (XR accessibility linter) are the contributions. Let them speak for themselves.
