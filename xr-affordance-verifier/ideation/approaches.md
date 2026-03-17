# Competing Approaches for xr-affordance-verifier

> **Context.** The xr-affordance-verifier aims to formally verify that XR scenes are spatially accessible across the population of human bodies and target devices. The crystallized problem (V4/D7/BP4/L5, composite 20/40) identified 85–90% compound fatal-flaw probability. Below are three genuinely competing strategies—each a plausible "best path"—that differ in mathematical foundation, engineering architecture, and risk profile.

---

## Approach A: Pose-Guarded Hybrid Automata with Chart-Decomposed CAD

**One-line summary.** Full formal verification of XR spatial accessibility via hybrid automata on SE(3), using cylindrical algebraic decomposition (CAD) adapted for Lie-group chart transitions to produce sound, finite zone abstractions.

### Value

**Who desperately needs it:** Enterprise XR compliance teams at Boeing, Lockheed Martin, Siemens, and surgical-simulation vendors (Osso VR, Fundamental Surgery) who face Section 508 / ADA Title I obligations and, imminently, the EU Accessibility Act. A platform holder like Meta's Accessibility Engineering group could use the Tier 3 certification engine to screen Quest Store submissions, converting an internal audit cost into a developer-facing gate. The extreme value is *provable guarantees*: not "we sampled 100K bodies and found no failures" but "no body parameterization in the 5th–95th percentile can fail to reach this button." For regulated industries where litigation risk attaches to accessibility failures, this is the difference between "best effort" and "certified compliant."

**Specific use case.** A surgical-training XR app must be usable by trainees ranging from a 5th-percentile-height woman with limited shoulder abduction (70° vs. typical 150°) to a 95th-percentile-height man, across Quest 3 hand-tracking and PSVR2 wand controllers. The certification engine produces a machine-checkable proof that every interaction step in a 12-step procedure is reachable for every point in the anthropometric × device parameter space—or returns a concrete body configuration and interaction step that fails.

### Difficulty

**Why this is genuinely hard as a software artifact:**

1. **CAD on SE(3) (the core open problem).** Cylindrical algebraic decomposition operates on ℝⁿ. SE(3) = SO(3) ⋊ ℝ³ is a non-contractible 6-manifold. The two viable embeddings—(a) ℝ¹² with 6 orthogonality constraints inflating effective dimension, (b) local charts (Euler angles / unit quaternions) with chart-transition soundness proofs—are both at the frontier. No published CAD implementation handles chart transitions on a Lie group. Effective dimension for a full arm kinematic chain is 7–10 (shoulder 3-DOF + elbow 1-DOF + wrist 3-DOF + translation 3-DOF), and CAD doubly-exponential complexity (2^{2^{O(n)}}) makes dimension 10 potentially intractable.

2. **Lazy PGHA product construction.** The product automaton of SE(3) pose space × interaction state machine × scene state can explode combinatorially. The compositional assume-guarantee framework (M4) requires spatial discharge conditions that are novel and NP-hard to compute optimally; the bounded-treewidth hypothesis (M3b) is unvalidated on real scenes.

3. **Unity scene extraction.** Parsing .unity YAML, .prefab hierarchies, and C# MonoBehaviour scripts to extract interaction state machines is a 14K+ LoC engineering effort with no clean API; C# analysis requires Roslyn integration or conservative pattern-matching.

4. **CEGAR loop over SE(3).** Counterexample-guided abstraction refinement must concretize abstract zone-graph traces into physical body configurations on SE(3), then re-refine the zone abstraction when spurious—requiring geometric feasibility solvers at each iteration.

**Hard subproblems:** Chart-transition soundness proof, CAD cell-count scaling at dimension 7–10, treewidth measurement of real XR interaction graphs, SE(3) counterexample concretization.

### New Mathematics

| ID | Contribution | Type | Load-Bearing Role |
|----|-------------|------|-------------------|
| **M2** | **SE(3) zone abstraction soundness theorem.** Proves that chart-decomposed CAD on SO(3) × ℝ³, with explicit chart-overlap consistency conditions, yields a sound finite abstraction of the continuous reachable set. Requires handling the non-contractibility of SO(3) (antipodal identification on S³) and chart-transition regions. | **(a) Genuinely new.** CAD on non-contractible manifolds with chart-transition soundness has no precedent in the algebraic geometry or formal verification literature. | Enables the entire Tier 2 and Tier 3 pipeline. Without it, zone abstraction is heuristic, not sound. |
| **M3b** | **Bounded-treewidth compositional tractability.** If the interaction graph of an XR scene has treewidth w, abstract zone-graph reachability is O(\|Z\|^{w+1} · poly(m)) where \|Z\| = zones per object, m = scene size. | **(b) Known technique (tree decomposition for model checking) applied to a new domain** (PGHA zone graphs). The novelty is in proving that the spatial locality of XR interactions induces bounded treewidth and defining the correct notion of "interaction graph." | Makes Tier 2 verification tractable for scenes with spatially clustered interactions (the common case). Without it, BDD explosion at m > 30 objects. |
| **M3a** | **k-locality zone-count bound.** For k-local interactions (each guard involves ≤ k joints), zone count is O(f(k) · poly(n)) where n = total DOF. | **(b) Known CAD projection techniques applied to kinematic locality structure.** | Bounds Tier 2 zone construction time. |
| **M1** | **PGHA operational semantics on Lie groups.** Formal definition of hybrid automata with SE(3) continuous state, semialgebraic guards, and identity reset maps. | **(c) Straightforward extension** of existing hybrid automata theory (Henzinger, Alur) to manifold-valued continuous state. | Provides the formal object that the rest of the math reasons about. |
| **M4** | **Spatial assume-guarantee for PGHA.** Compositional verification where spatial non-interference serves as the side condition for assume-guarantee decomposition. | **(c) Straightforward application** of Henzinger et al. assume-guarantee framework with spatial discharge conditions that are domain-specific but not mathematically deep. | Enables scaling to large scenes by verifying clusters independently. |
| **M7** | **CEGAR completeness for finite zone graphs.** The CEGAR loop over finite zone-graph abstractions terminates because the lattice of refinements is finite (bounded by CAD cell count). | **(c) Straightforward application** of standard CEGAR theory. The SE(3)-specific concretization subroutine is novel engineering but not novel math. | Guarantees Tier 3 termination. |

**Crown jewel: M2.** If chart-decomposed CAD soundness on SE(3) is proven and the resulting cell count is tractable at dimension 7–10, this is an A-grade contribution to computational algebraic geometry with implications beyond XR.

### Best-Paper Argument

**Venue: CAV (Computer-Aided Verification) or TACAS.**

The argument: "This paper opens a new application domain for hybrid-systems verification—XR spatial accessibility—by solving the previously open problem of constructing sound finite abstractions of semialgebraic predicates on SE(3)." CAV rewards sharp technical contributions that open new territory. M2 is exactly that: a soundness theorem for CAD on a non-contractible Lie group, which has implications for any CPS application involving orientation (robotics, aerospace, biomechanics). The bounded-treewidth result (M3b) provides the tractability story that reviewers demand. If the chart-transition problem is solved cleanly, the paper has genuine best-paper potential because it bridges computational algebraic geometry and formal verification in a way that hasn't been done.

**Risk:** CAV reviewers may find the XR application domain niche. The chart-transition soundness proof must be airtight—a gap will sink the paper. If CAD cell counts at dimension 7 are 10⁸+, the "tractability" claim is undermined.

### Hardest Challenge

**CAD cell-count scaling at dimension 7–10 on SE(3).**

CAD complexity is doubly exponential in dimension: O((2d)^{2^{n}}) where d = max polynomial degree and n = effective dimension. At n = 4 (2-joint planar chain), this may be ~10⁴–10⁵ cells. At n = 7 (3-DOF shoulder + 1-DOF elbow + 3-DOF wrist), it could be 10⁸–10¹² cells, far beyond tractability.

**Mitigation strategy:**
1. **k-locality projection (M3a):** Most guards involve ≤ 4 joints → project CAD to 4-dimensional subspaces, avoiding full-dimensional decomposition.
2. **Lazy cell enumeration:** Only construct cells reachable from the initial zone; unreachable cells are never computed.
3. **Degree reduction via kinematic simplifications:** Replace exact joint-limit polynomials with lower-degree approximations that are conservative (over-approximate the reachable set).
4. **Month-2 kill gate (A1):** If a 4-joint chain produces > 10⁵ cells or takes > 10 minutes, pivot to Approach B immediately.

### Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 5/10 | Provable guarantees are the gold standard for regulated industries, but the addressable market is small (~30–50K developers in accessibility-critical XR), demand is unvalidated, and the DSL adoption barrier is real. Score rises to 7 if a platform holder commits. |
| **Difficulty** | 9/10 | SE(3) zone abstraction is a genuine open problem in computational algebraic geometry. Full PGHA + CEGAR + compositional decomposition at ~85–110K novel LoC is an enormous artifact. Subsystem integration (CAD + BDD + SMT + Unity parser) is exceptionally challenging. |
| **Potential** | 6/10 | M2 is an A-grade result if solved; CAV best-paper is plausible but not likely (~8–12% given venue competition). Two-paper strategy (CAV + UIST) is viable but demands resolved chart-transition proof AND real-scene evaluation. |
| **Feasibility** | 3/10 | 40% probability that CAD at dimension 7–10 is intractable. 35% probability of zero user demand. 25% probability that chart-transition soundness is harder than expected. Compound fatal-flaw probability ~85%. Kill gates help but the base-rate is grim. |

---

## Approach B: Interval-Arithmetic Accessibility Linter with Conservative Reachability Envelopes

**One-line summary.** Skip CAD and hybrid automata entirely; verify XR spatial accessibility using interval arithmetic over parameterized forward-kinematics to compute conservative reachability envelopes, delivering a fast Unity linter (Tier 1) and CI/CD gate (Tier 2) that catch 85–95% of spatial accessibility bugs with zero false negatives.

### Value

**Who desperately needs it:** Every XR developer, not just enterprises. The core insight is that most spatial accessibility bugs are *easy*: a button placed 2.3 m high is unreachable for the 10th-percentile-height population regardless of subtle joint interactions. An interval-arithmetic linter running in < 2 seconds inside the Unity editor—no DSL, no annotations, zero configuration—catches these immediately. The CI/CD gate (< 5 minutes) handles multi-step interactions with conservative envelope intersection.

**Specific use case.** An indie developer building a Quest 3 fitness app drags a "start workout" button in the Unity editor. The linter instantly highlights it red: "Unreachable for 23% of the population (height < 165 cm, shoulder flexion < 120°). Suggested placement: lower by 15 cm or add gaze-activation alternative." This is *developer-experience-first* accessibility tooling: think ESLint for spatial layout.

**Why the value is extreme:** The XR accessibility tooling market is completely empty. Even a conservative, imprecise tool that catches obvious violations has near-infinite marginal value over the status quo (nothing). The tool can ship as a free Unity Asset Store plugin, build community adoption, and create the demand signal that Approach A needs but lacks. Platform holders (Meta, Apple) could mandate it as a submission requirement—the low integration cost (< 2s, no annotations) makes this politically feasible.

### Difficulty

**Why this is genuinely hard (despite simpler math):**

1. **Tight interval enclosures for forward kinematics.** Naïve interval arithmetic on the composition of rotation matrices suffers catastrophic wrapping: the enclosure of R₁(θ₁) · R₂(θ₂) · ... · Rₖ(θₖ) · p can be 10–100× wider than the true reachable set. Getting enclosures tight enough to be *useful* (not just sound) requires affine arithmetic, Taylor-model propagation, or joint-range subdivision—each with nontrivial engineering.

2. **Multi-step interaction reasoning without state machines.** Tier 2 must verify sequential interactions (e.g., "grab handle, pull drawer, reach item inside drawer") where body configuration at step i constrains reachability at step i+1. Without explicit state machines, this requires conservative composition of reachability envelopes across steps—an over-approximation that compounds and can produce excessive false positives for chains > 3 steps.

3. **Anthropometric distribution integration.** Computing what *fraction* of the population fails—not just whether anyone fails—requires integrating interval enclosures against ANSUR-II/CAESAR anthropometric distributions. This is tractable for marginal distributions but hard for correlated body parameters (arm length correlates with height, shoulder mobility anticorrelates with age).

4. **Unity editor integration.** Real-time (< 2s) feedback inside the Unity editor requires a native C++ plugin (not managed C#) for interval arithmetic, careful memory management, and incremental recomputation when scene elements move.

5. **False-positive management.** Conservative over-approximation means the linter will flag some accessible elements as potentially inaccessible. If the false-positive rate exceeds ~15%, developers will disable it. Balancing soundness against usability is a hard engineering problem.

**Novel LoC estimate:** ~35–50K total, of which ~20–30K is genuinely novel (interval-arithmetic engine, envelope composition, anthropometric integration, Unity plugin architecture).

### New Mathematics

| ID | Contribution | Type | Load-Bearing Role |
|----|-------------|------|-------------------|
| **B1** | **Affine-arithmetic forward-kinematics enclosure with wrapping-reduction.** Bound the over-approximation factor of affine-arithmetic propagation through k-joint revolute chains as a function of joint-range width. Prove that for typical XR joint ranges (< 60° per joint for constrained populations), the enclosure is within 2× of the true reachable set. | **(b) Known technique (affine arithmetic, Stolfi & de Figueiredo 1997) applied to kinematic chains.** The novelty is in the wrapping-factor analysis specific to revolute-joint composition, which doesn't exist in the interval-arithmetic literature. | Determines whether Tier 1 enclosures are tight enough to be useful. Without this bound, we can't predict false-positive rates. |
| **B2** | **Conservative sequential-envelope composition.** For a k-step interaction sequence where step i's reachable set R_i depends on the body configuration at step i−1, define an over-approximate composition operator R₁ ⊕ R₂ ⊕ ... ⊕ Rₖ that is (a) sound (contains the true sequential reachable set) and (b) grows at most quadratically in k (not exponentially). Achieves this by tracking a correlated affine form across steps rather than independent intervals. | **(a) Genuinely new, but modest.** Correlated affine forms for kinematic-chain composition across sequential constraints haven't been formalized. The math is not deep—it's careful bookkeeping of affine-form dependencies—but it's new and directly useful. | Enables Tier 2 multi-step verification without state-machine extraction. Without it, envelope composition is exponentially loose and Tier 2 is useless beyond 2-step interactions. |
| **B3** | **Population-fraction computation via interval-distribution convolution.** Given an interval enclosure of the failure region in anthropometric space and a multivariate distribution (ANSUR-II body dimensions), compute a sound lower bound on the fraction of the population affected. Uses Chebyshev-type concentration bounds when the distribution is not analytically tractable. | **(b) Known probabilistic techniques applied to the anthropometric-interval setting.** Standard but useful. | Converts a binary "reachable/unreachable" verdict into a quantitative "X% of population affected" report, which is what compliance teams actually need. |

**No crown jewel—but all three contributions are directly load-bearing.** None is individually A-grade, but together they form a coherent and publishable story.

### Best-Paper Argument

**Venue: UIST (User Interface Software and Technology) or CHI (Systems track).**

The argument: "We present the first real-time accessibility linter for mixed-reality development environments, running at < 2 seconds in the Unity editor with zero developer annotations. On a benchmark of 500 scenes with injected spatial-accessibility bugs, the linter detects 91% of bugs with a 12% false-positive rate, compared to 0% detection by existing tools (which don't exist). A 22-developer study shows that linter-assisted developers fix 3.4× more accessibility violations per hour than unaided developers."

UIST and CHI reward *working systems that change developer behavior*. The mathematical depth is modest but the systems contribution—first-of-its-kind real-time accessibility feedback for XR—is compelling. Best-paper potential rests on the developer study: if we can demonstrate measurable accessibility improvement in a controlled study, this is exactly what UIST rewards.

**Risk:** CHI/ASSETS reviewers may demand engagement with disability communities (co-design, participatory methods). UIST reviewers may find the math too shallow for a technical contribution. The tool must actually work on real Unity projects, not just benchmarks.

### Hardest Challenge

**Interval-wrapping explosion in multi-step sequential interactions.**

For a 5-step interaction (e.g., open door → walk through → reach panel → press button → confirm), naïve interval composition produces envelopes 50–200× wider than the true reachable set, making Tier 2 useless (everything is flagged as potentially inaccessible).

**Mitigation strategy:**
1. **Correlated affine forms (B2):** Track first-order dependencies between steps so that the envelope width grows quadratically, not exponentially.
2. **Subdivision on critical parameters:** When envelope width exceeds a threshold, subdivide the anthropometric parameter space into 2–8 sub-ranges and verify each independently. At k = 3 subdivision dimensions × 4 sub-ranges each = 64 sub-problems, still tractable at < 5 minutes total.
3. **Interaction-step clustering:** Group sequential steps that share no joints (e.g., "reach with left hand" then "reach with right hand") and verify independently, avoiding cross-step wrapping.
4. **Graceful degradation:** For interactions > 5 steps where over-approximation is too loose, report "unable to verify—consider Tier 3 analysis" rather than a false positive. This bounds the false-positive rate at the cost of completeness.

### Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 7/10 | Fills a completely empty niche (zero existing XR accessibility tools). Zero-configuration Unity plugin has realistic adoption path. Quantitative population-fraction reports serve compliance teams. Loses points because guarantees are conservative (not exact) and multi-step reasoning is limited. |
| **Difficulty** | 5/10 | Tight interval enclosures for kinematic chains are nontrivial but not frontier-research-hard. Unity plugin engineering is significant but well-understood. ~35–50K LoC is substantial but manageable. No open mathematical problems. |
| **Potential** | 5/10 | UIST/CHI best paper requires strong developer study (which is feasible but adds scope). Math contributions are B-grade (useful, new-to-domain, but not deep). No CAV/TACAS shot. Could be a solid UIST paper but unlikely to be remembered 5 years later. |
| **Feasibility** | 8/10 | No unsolved mathematical problems. All algorithmic components have known solutions (interval/affine arithmetic, forward kinematics, BDD-free). Month-2 prototype is realistic. Primary risk is false-positive rate—if > 20%, tool is unusable. Kill gate: if wrapping factor on a 4-joint chain exceeds 5× at Month 1, switch to Taylor-model propagation. |

---

## Approach C: Sampling-Guided Symbolic Verification with Probabilistic Coverage Certificates

**One-line summary.** Use stratified Monte Carlo sampling over the anthropometric × device parameter space to identify probable failure regions, then apply targeted SMT-based symbolic verification (over linearized kinematics) only in inconclusive zones, producing a *coverage certificate* that formally bounds the probability of undetected bugs.

### Value

**Who desperately needs it:** The same compliance teams as Approach A, but with a realistic path to deployment. The key insight is that most of the body-parameter space is *easy*—either clearly accessible (tall, flexible users) or clearly inaccessible (extreme anthropometric outliers). The hard cases lie on the *boundary* of the accessibility frontier in parameter space. Approach C allocates computational effort adaptively: cheap sampling for the easy 90%, expensive symbolic verification for the hard 10%.

**Specific use case.** A Meta Quest Store reviewer runs the tool on a submitted app. In 8 minutes, the tool reports: "With 99.7% confidence (coverage certificate), all interactions are accessible for body parameterizations covering 95.2% of the ANSUR-II distribution. Two interactions have inconclusive accessibility for the 4.8% boundary population [details]. One interaction is provably inaccessible for 1.3% of the population [counterexample: height 152 cm, shoulder flexion < 85°, wrist pronation < 40°]." This combines statistical assurance (the 99.7% confidence) with symbolic proof (the provable inaccessibility) in a single report.

**Why the value is extreme:** The coverage certificate is the novel artifact. Monte Carlo alone gives point estimates with no formal guarantees. SMT alone is too expensive for full-space verification. The hybrid approach gives *formally grounded probabilistic guarantees* at practical computational cost—a middle ground that doesn't exist in any accessibility domain.

### Difficulty

**Why this is genuinely hard as a software artifact:**

1. **Adaptive sampling with formal coverage guarantees.** The coverage certificate must formally bound P(missed bug) ≤ ε given the sampling strategy. This requires (a) a formal model of the bug distribution in parameter space (which we don't know a priori), (b) concentration inequalities that account for adaptive sampling (samples are not i.i.d. once we refine based on results), and (c) soundness of the certificate even when the sampling strategy is adversarially unlucky. This is related to but distinct from PAC-learning theory.

2. **Linearized-kinematics SMT encoding.** To verify boundary regions symbolically, we encode forward kinematics as SMT constraints. Exact trigonometric kinematics are undecidable in general (theory of reals with sine). Linearization (first-order Taylor expansion around the sampling center) is decidable but introduces approximation error that must be formally bounded. The *soundness envelope* around the linearization point determines how large a region each SMT query covers—too small and we need exponentially many queries, too large and the approximation is unsound.

3. **Sampling-symbolic handoff.** Deciding when to stop sampling and start symbolic verification in a region is a non-trivial decision problem. Too early: wasted SMT calls. Too late: wasted sampling budget. The optimal strategy depends on the local geometry of the accessibility frontier in parameter space, which is unknown.

4. **Counterexample witness quality.** When the SMT solver finds a symbolic counterexample in a linearized region, we must verify it against the exact (nonlinear) kinematics. If the linearization error exceeds the margin, the counterexample may be spurious. A refinement loop (re-linearize at the candidate, re-check with tighter bounds) is needed and may not converge.

5. **Multi-step sequential verification under sampling.** For k-step interactions, sampling must explore the space of *trajectories* (body configurations at each step), not just static configurations. The dimension of the trajectory space grows as k × d where d = body DOF, making coverage exponentially harder.

**Novel LoC estimate:** ~55–75K total, of which ~35–50K is genuinely novel (adaptive sampler, coverage-certificate engine, linearized SMT encoding, sampling-symbolic handoff, trajectory-space sampling).

### New Mathematics

| ID | Contribution | Type | Load-Bearing Role |
|----|-------------|------|-------------------|
| **C1** | **Adaptive-sampling coverage certificate.** Define a certificate structure ⟨S, V, ε, δ⟩ where S = sampled regions (with verdicts), V = symbolically verified regions (with proofs), ε = upper bound on P(missed bug), δ = confidence level. Prove that the certificate is sound: if the true bug set B satisfies a Lipschitz-continuity condition in parameter space (bugs don't appear at isolated points with zero measure), then P(B ∩ (Ω \ S ∪ V)) ≤ ε with probability ≥ 1−δ. Uses a novel combination of stratified-sampling concentration bounds and SMT-verified region coverage. | **(a) Genuinely new.** The combination of statistical coverage guarantees with symbolic verification certificates doesn't exist. Related work in statistical model checking (Younes & Simmons 2006, Legay et al. 2010) provides guarantees for *temporal properties of stochastic systems*, not *spatial coverage of deterministic parameter spaces*. The Lipschitz-continuity assumption is the key enabling condition and is novel in this setting. | The entire value proposition rests on this certificate. Without it, Approach C is just "Monte Carlo with some SMT on the side"—no formal guarantees, no publication story. |
| **C2** | **Linearization soundness envelope for revolute-chain kinematics.** For a k-joint revolute chain with joint angles θ ∈ [θ₀ − Δ, θ₀ + Δ], the linearized forward-kinematics f_lin(θ) = f(θ₀) + J(θ₀)(θ − θ₀) has approximation error \|f(θ) − f_lin(θ)\| ≤ C · Δ² · L_max^k where L_max = max link length, C = explicit constant depending on chain geometry. This bound determines the maximum Δ for which a single SMT query is sound. | **(b) Known Taylor-remainder analysis applied to kinematic chains.** The specific constant and its dependence on chain geometry is new but the technique is standard. | Determines SMT query granularity—hence total SMT cost. If the envelope is too small, we need O(1/Δ^d) queries per boundary region, killing tractability. |
| **C3** | **Optimal sampling-symbolic budget allocation.** Given total compute budget T, sampling cost c_s per sample, and SMT cost c_v per region, derive the allocation (n_samples, n_smt_regions) that minimizes ε in the coverage certificate. Show this is a convex optimization when the accessibility frontier is Lipschitz, admitting a closed-form solution parameterized by the estimated frontier complexity. | **(b) Known optimization techniques applied to a new resource-allocation problem.** Related to optimal experiment design in statistics. The novelty is in the specific problem formulation and the connection to the coverage certificate. | Guides the tool's internal resource allocation. Without it, the sampling-vs-SMT tradeoff is heuristic, undermining the certificate's tightness. |

**Crown jewel: C1.** The coverage certificate is genuinely novel and conceptually appealing. It provides a formal bridge between statistical testing and symbolic verification—a topic of independent interest beyond XR.

### Best-Paper Argument

**Venue: CAV (as a hybrid verification technique) or TACAS (tool track), with a secondary UIST submission for the tool.**

The argument: "We introduce *coverage certificates* for parameter-space verification: formal objects that combine statistical sampling evidence with targeted symbolic proofs to bound the probability of undetected property violations. Applied to XR spatial accessibility, coverage certificates enable verification of body-parameterized reachability across the full anthropometric distribution in minutes, with formally guaranteed miss rates below user-specified thresholds. On a benchmark of 500 XR scenes, coverage-certified verification detects 97% of injected bugs (vs. 91% for pure sampling, 100% for full symbolic at 47× the cost) with ε ≤ 0.003 certificates achievable in < 10 minutes."

CAV has published hybrid statistical-symbolic approaches (statistical model checking, runtime verification). Coverage certificates extend this paradigm to *parameter-space verification* of deterministic systems—a genuinely new concept. The theoretical contribution (C1) is sharp enough for a CAV technical paper, and the practical tool (detecting XR bugs faster than pure symbolic) provides the empirical story reviewers want. Best-paper potential exists if the coverage-certificate framework generalizes beyond XR (which it clearly does—any parameterized verification problem can use it).

**Risk:** Reviewers may see the Lipschitz-continuity assumption as too strong (pathological scenes could violate it). The linearization-based SMT encoding may be seen as too approximate compared to CAD-based exact verification. If the coverage certificate's ε bound is loose (e.g., ε = 0.1 for realistic compute budgets), the formal guarantee is underwhelming.

### Hardest Challenge

**Making the coverage certificate tight enough to be useful.**

The coverage certificate's ε bound depends on: (a) sampling density in unexplored regions, (b) SMT-verified region volume, and (c) the Lipschitz constant of the accessibility frontier. If the frontier is highly irregular (many small inaccessible pockets scattered through parameter space), the Lipschitz constant is large and ε degrades.

**Mitigation strategy:**
1. **Frontier-adaptive stratification.** Use initial uniform sampling to estimate the accessibility frontier's location, then concentrate subsequent samples near the frontier. Standard stratified sampling reduces variance by O(1/k²) vs. O(1/k) for uniform sampling—frontier-adaptive stratification can do better when the frontier is smooth.
2. **SMT verification of frontier neighborhoods.** Instead of verifying arbitrary regions, target SMT queries at thin shells around the estimated frontier. Each verified shell eliminates the highest-uncertainty region, maximally tightening ε.
3. **Hierarchical linearization.** If the linearization envelope at a boundary point is too small (< 1° joint range), re-linearize at multiple points along the frontier and combine their envelopes. This trades SMT queries for better coverage per query.
4. **Graceful degradation with certificate reporting.** The tool always produces a certificate, even if ε is large. For easy scenes (95% of cases), ε < 0.001 in seconds. For hard scenes with complex frontiers, the tool reports "ε = 0.05 (95% confidence) with current budget; increase budget to X for ε = 0.01." This is honest and useful.
5. **Empirical Lipschitz estimation.** From the sampling data, estimate the local Lipschitz constant of the frontier and use it to tighten the certificate. If the estimate is unreliable (insufficient data), fall back to conservative global bounds.

### Scores

| Axis | Score | Justification |
|------|-------|---------------|
| **Value** | 7/10 | Coverage certificates offer formal guarantees at practical cost—the best of both worlds. The 99.7%-confidence + provable-counterexample reporting is exactly what compliance teams need. Loses points because the guarantee is probabilistic (not absolute like Approach A) and the Lipschitz assumption may not hold universally. |
| **Difficulty** | 7/10 | The coverage certificate (C1) is a genuine research contribution. Adaptive sampling with formal guarantees in high-dimensional spaces is hard. SMT encoding of linearized kinematics and the sampling-symbolic handoff are nontrivial. ~55–75K LoC is significant. Not as hard as Approach A because no open algebraic-geometry problems, but harder than Approach B because the certificate theory is genuinely new. |
| **Potential** | 7/10 | Coverage certificates generalize beyond XR—any parameterized verification problem benefits. This generality gives the CAV paper a broader impact story than Approach A (which is XR-specific). The concept is elegant and memorable. Loses a point because the XR application is still niche and the Lipschitz assumption invites reviewer pushback. |
| **Feasibility** | 6/10 | No open mathematical problems (C1 is new but tractable—the proof technique is a combination of known tools). SMT encoding of linearized kinematics is well-understood. Primary risk: coverage certificate ε may be too loose for practical use (> 0.05 for complex scenes). Secondary risk: linearization error compounds over multi-step interactions. Kill gate: if ε > 0.1 on a 30-object scene with 10-minute budget at Month 3, the certificate is not useful enough to publish. |

---

## Comparative Summary Table

| Dimension | **A: PGHA + CAD** | **B: Interval Linter** | **C: Sampling-Symbolic Hybrid** |
|-----------|-------------------|----------------------|-------------------------------|
| **Core Math** | SE(3) zone abstraction via chart-decomposed CAD | Affine-arithmetic kinematic enclosures | Coverage certificates (statistical + symbolic) |
| **Guarantee Type** | Sound & complete (for abstraction) | Sound, conservative (over-approximate) | Probabilistic with formal bound |
| **Tier Coverage** | Tier 1 + 2 + 3 | Tier 1 + 2 only | Tier 1 + 2 (Tier 3 aspirational) |
| **Strongest Math** | M2: A-grade (genuinely new, open problem) | B2: B-grade (new, modest) | C1: A−grade (genuinely new, generalizable) |
| **Primary Venue** | CAV (theory) + UIST (tool) | UIST or CHI (tool + study) | CAV (theory) + UIST (tool) |
| **Novel LoC** | ~85–110K | ~20–30K | ~35–50K |
| **Time to First Useful Output** | Month 6+ (requires CAD prototype) | Month 2 (interval linter ships early) | Month 3 (sampler + basic certificate) |
| **Fatal-Flaw Probability** | ~85% (CAD intractability, demand, MC dominance) | ~25% (false-positive rate, limited multi-step) | ~45% (certificate looseness, Lipschitz violation) |
| **Value** | 5 | 7 | 7 |
| **Difficulty** | 9 | 5 | 7 |
| **Potential** | 6 | 5 | 7 |
| **Feasibility** | 3 | 8 | 6 |
| **Composite (sum)** | **23** | **25** | **27** |

### Recommended Strategy

**Primary: Approach C.** The coverage-certificate concept is genuinely novel, generalizable beyond XR, and achievable within a realistic timeline. It delivers strong value (formal probabilistic guarantees at practical cost), targets the same venues as Approach A with better feasibility, and produces a more memorable intellectual contribution (coverage certificates as a paradigm) than either alternative.

**Hedge: Begin with Approach B's Tier 1 linter (Month 1–2)** as the engineering substrate. The interval-arithmetic linter is useful regardless of which higher-tier approach succeeds, ships earliest, and validates demand (gate A7). At Month 2, if the coverage-certificate prototype (C1) shows ε < 0.05 on benchmark scenes, commit to Approach C for the research contribution. If C1 fails, Approach B's linter is already a viable UIST submission.

**Abandon Approach A unless:** Month-2 CAD prototype (gate A1) shows ≤ 10⁵ cells for a 4-joint chain AND a platform holder expresses concrete interest (gate A7). The compound probability of both gates passing is ~15–20%.
