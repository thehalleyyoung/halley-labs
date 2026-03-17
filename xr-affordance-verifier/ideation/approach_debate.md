# Adversarial Debate: xr-affordance-verifier Approaches

**Panel:** Skeptic (maximally hostile), Math Assessor (novelty auditor), Difficulty Assessor (engineering realist)
**Format:** Five rounds per approach, then cross-approach comparison and final verdict.

---

## Approach A: Pose-Guarded Hybrid Automata with Chart-Decomposed CAD

### Round 1 — Skeptic's Attack

**Attack 1: You're betting the farm on an open problem.** The entire project—Tiers 2 and 3, the CAV paper, the formal guarantees that differentiate this from Monte Carlo—rests on M2, an unproven theorem about CAD on a non-contractible Lie group. You haven't solved the chart-transition soundness problem. Nobody has. You're proposing to solve it as a *side task* while simultaneously building an 85–110K LoC system. If the proof has a gap, the crown jewel evaporates and you've spent 12 months on a broken tool. The document's own feasibility score is 3/10 and the compound fatal-flaw probability is ~85%. Those numbers should end this conversation.

**Attack 2: The simpler alternative destroys you.** 2,000 lines of Python running 1M stratified Monte Carlo samples over ANSUR-II body parameters catches >90% of bugs in seconds. Your 110K LoC system adds 5–15% marginal detection for a 1,000× increase in complexity. Nobody asked for provable guarantees—there is no XR-specific accessibility regulation, zero developer surveys, zero platform-holder interest. You're building a cathedral in a desert.

**Attack 3: The "so what?" test.** Even in the miracle scenario where every open problem is solved, CAD scales, the proof is airtight, and the tool ships—who writes the check? The addressable market is ~30–50K developers. The intersection of "motor disability" and "owns XR headset" is tens of thousands of humans. The DSL adoption barrier means your actual users are the tiny fraction of those 30–50K developers willing to learn a formal specification language. The formal guarantee is the difference between an A in a graduate seminar and a product anyone deploys.

### Round 2 — Math Assessor's Critique

**Novelty audit:** The project claims 8 mathematical contributions. The honest count is **1.5**. M2 (chart-decomposed CAD on SE(3)) is genuinely unprecedented—no published CAD implementation handles chart transitions on Lie groups. This is real. M3b (bounded treewidth for XR scenes) is an *empirical hypothesis*, not a provable theorem—if real scenes have bounded treewidth, the result is useful but routine; if not, there's nothing to publish. M1, M3a, M4, M7 are formalism housekeeping: definitions, straightforward extensions of Henzinger et al., and the observation that finite lattices have no infinite descending chains. Listing these as "contributions" is padding.

**Crown jewel conditional:** M2 is grade-A *if proved*. But an unproven theorem with a 35–40% chance of failure is a conjecture, not a contribution. The chart-transition soundness proof requires merging CAD cell complexes across algebraically unrelated coordinate charts on a space with π₁ = ℤ/2—this has no precedent. Getting it wrong doesn't produce imprecision; it produces *unsoundness*. Grade the actual state: **A if proved, D if not, with no middle ground.**

**Risk assessment:** P(math contribution fails or must be substantially weakened) ≈ 55–65%. Chart-transition failure (35%), CAD intractability at dim 7–10 (40%), treewidth failure on real scenes (25%). These risks are correlated. If the math fails, the fallback is a Tier 1 interval linter—which is Approach B with less engineering maturity.

### Round 3 — Difficulty Assessor's Reality Check

**LoC realism:** Claimed 85–110K novel LoC. Genuinely *difficult* algorithmic code: ~45–55K. The difficulty is concentrated in the Zone Abstraction Engine (18–22K, every line load-bearing) and CEGAR (7–9K, dense geometric reasoning). The remaining ~40–55K is real but not intellectually hard. The problem isn't volume—it's that those 45K difficult lines require simultaneous deep expertise in computational algebraic geometry *and* formal verification, a rare combination.

**Integration nightmare:** Five heterogeneous computational substrates (QEPCAD, CUDD, Z3/CVC5, Pinocchio, Unity) with incompatible data representations. QEPCAD is 1990s-era C maintained by one retired person, with no parallel implementation. A sign error in a chart-transition boundary polynomial silently produces an unsound abstraction. Integration testing this end-to-end pipeline is itself a multi-week effort.

**Laptop feasibility:** Tier 1 trivially feasible. Tier 2 on edge at 50+ objects (BDD memory 1–10GB). Tier 3 **almost certainly infeasible**: CAD at dimension 7 consumes 4–8GB and hours of CPU time *per cell computation*, CEGAR with 10+ iterations on a 30-object scene requires 32GB+ and days. The 16GB constraint is fatal for Tier 3.

**Timeline:** 6-month prototype probability ~35%. 12-month full system for a single developer: ~15%. This is a multi-year, multi-expert project compressed into a solo timeline.

### Round 4 — Defense

The defense of Approach A is not that it will likely succeed—it is that the *expected value* of the attempt, properly managed, justifies the risk.

**On the open problem:** Yes, M2 is unproven. But the Month-2 kill gate (A1) exists precisely for this: if a 4-joint chain produces >10⁵ cells or takes >10 minutes, we pivot to B immediately. The worst case is 2 months of focused research that either produces a publishable algebraic geometry result or terminates cleanly. The *standalone* M2 result—a sound finite abstraction of semialgebraic predicates on a non-contractible Lie group—has implications for robotics, aerospace, and biomechanics verification far beyond XR. Even if the tool dies, the math paper lives.

**On the simpler alternative:** Monte Carlo catches *almost all* bugs. Formal verification catches *all* bugs. For surgical training simulators where a missed accessibility failure means a trainee can't complete a life-critical procedure, "almost all" isn't enough. The 5–15% marginal detection rate includes the *hardest* bugs—boundary cases where a specific body parameterization fails at a specific joint configuration. These are precisely the bugs that cause ADA litigation. The market is small today, but the EU Accessibility Act's trajectory toward XR-specific regulation makes "nobody asked for it yet" an argument against building the Brooklyn Bridge before Brooklyn.

**On scope management:** The tiered architecture provides graceful degradation. If only Tier 1 ships, it's still the first XR accessibility linter. If Tier 2 works but Tier 3 doesn't, you have a sound-but-incomplete verifier that publishes at UIST. The all-or-nothing framing is a Skeptic's construction.

### Round 5 — Cross-Team Challenge

**Math Assessor vs. Skeptic:** The Skeptic says "kill it." But I rated M2 as grade-A—the only genuinely A-grade math across all three approaches. If the Skeptic's 2,000-line Python script catches 90% of bugs, it produces *zero* publishable math. This project's charter requires both a tool and a research contribution. Approach A is the only one where solving the core problem advances the field of computational algebraic geometry. The Skeptic evaluates A as a product; I evaluate it as a research bet. At 35% success probability, the expected value of an A-grade result justifies the attempt—provided kill gates are enforced.

**Skeptic's rebuttal:** Expected value requires accounting for opportunity cost. Those 2 months could prototype Approach B *and* start Approach C. A 35% chance at an A-grade result vs. an 85% chance at a working tool + B-grade paper is not a close call. And the "math paper without the tool" scenario gives you a CAV submission in computational algebraic geometry—which is great, but it's not what this project is for.

**Difficulty Assessor vs. Math Assessor:** Even if M2 is proved, the *engineering* of making it practical at dimension 7–10 is uncertain. QEPCAD has no parallel implementation, no modern build system, and may have undocumented limitations at high dimension. The math being correct does not make the tool feasible. The gap between "theorem proved" and "code that runs on real scenes in finite time" is enormous.

### Verdict: **WEAK**
An A-grade mathematical bet buried under engineering intractability, zero demand signal, and an 85% compound failure probability. Valuable only as a time-boxed research sprint on M2 with strict kill gates.

---

## Approach B: Interval-Arithmetic Accessibility Linter

### Round 1 — Skeptic's Attack

**Attack 1: The false-positive death spiral.** Interval arithmetic over kinematic chains suffers catastrophic wrapping—the approach itself admits 10–100× over-approximation for naïve propagation. Affine arithmetic reduces but doesn't eliminate this. If the false-positive rate exceeds ~15%, developers disable the linter and it dies the death of every ignored warning. For multi-step interactions (>3 steps), wrapping compounds and the tool flags *everything* as potentially inaccessible. At that point it's equivalent to no tool at all.

**Attack 2: An even simpler alternative exists.** Pre-compute lookup tables of reachability envelopes for 20 discrete anthropometric percentiles. For each interactive element, check membership. This is ~500 lines of code, runs in milliseconds, catches the same obvious bugs, and requires no affine arithmetic engine, no native C++ plugin. The interval arithmetic machinery only adds value for borderline cases near the reachability boundary—exactly where over-approximation produces false positives. You're adding 20–30K LoC of engineering for a capability that undermines itself on the hard cases.

**Attack 3: The research contribution is thin.** B1 is known technique applied to a known domain. B2 is "correlated affine forms"—standard in the affine arithmetic literature (Messine 2002, Kashiwagi 2003). B3 is a textbook exercise. The UIST paper story is "first XR accessibility linter," which is a systems contribution, not a research contribution. If the developer study falls flat (possible—no demand signal), the publication story evaporates.

### Round 2 — Math Assessor's Critique

**Novelty audit:** Zero genuinely new results. B1 applies affine arithmetic to kinematic chains—the wrapping-factor analysis for revolute chains is a useful calculation but a *lemma*, not a theorem. B2 (sequential envelope composition) is claimed as "genuinely new, but modest"—I downgrade to "known technique, modestly adapted." Correlated affine forms with dependency tracking are standard (Stolfi & de Figueiredo). B3 (Chebyshev bounds on interval-distribution convolution) is a graduate-level exercise.

**Crown jewel:** There is none. Grade: C+. The value of Approach B is entirely in the *system*, not the math. This is fine for UIST—many impactful tools have no novel math—but the math depth assessment must call it what it is.

**Key observation:** A math-free version (naïve intervals, single-step, binary yes/no) would still be the first XR accessibility linter and capture ~70% of the value. The math makes it *better*, not *possible*. Even if all of B1–B3 fail, the tool still works—just with higher false-positive rates. This is Approach B's superpower: **graceful degradation.** Math risk: 15–20%.

### Round 3 — Difficulty Assessor's Reality Check

**LoC realism:** Claimed ~20–30K novel. Genuinely difficult: ~10–14K. The hardest part is the affine-arithmetic FK engine with wrapping reduction (~5–8K), of which only ~3K is genuinely hard. The rest is solid engineering.

**Integration complexity: LOW.** A straightforward pipeline (scene → parse → envelope → check → report) with two language boundaries (C# ↔ C++), both well-supported by Unity's native plugin system. No BDDs, no SMT solvers, no CAD. This is the simplest architecture across all three approaches.

**Laptop feasibility: FULLY FEASIBLE.** Affine-arithmetic FK for 7-joint chain: ~0.1ms per evaluation. 10K evaluations for 100 objects × 100 sub-ranges ≈ 1 second. Multi-step with subdivision: ~5 seconds for 5-step interactions. Memory negligible (~10MB). Even old 4-core laptops handle this.

**Timeline:** 6-month prototype probability ~85%. 12-month full system probability ~65% for a single developer. No rare expertise combination required—a strong C++/numerical-methods engineer can build this.

### Round 4 — Defense

Approach B's defense is the most straightforward because its value proposition is the most honest: *build the thing people can actually use.*

**On the false-positive concern:** The kill gate is clear—if wrapping factor on a 4-joint chain exceeds 5× at Month 1, switch to Taylor-model propagation. Subdivision on critical parameters (2–8 sub-ranges per anthropometric dimension) keeps false-positive rates manageable. Graceful degradation for >5-step interactions (report "unable to verify" rather than a false positive) bounds the worst case. ESLint also has false positives; developers configure rule severity, not disable the tool.

**On the simpler alternative:** The lookup-table approach catches obvious violations but misses interactions where body proportions *correlate* (arm length with torso height). Interval arithmetic handles these correlations through affine forms. The lookup table also provides no population-fraction estimates—it gives 20 binary verdicts, not "23% of the population is affected." Compliance teams need quantitative reports, not percentile spot-checks.

**On research depth:** This approach never claimed to be a math breakthrough. The research contribution is the *tool*—the first real-time accessibility linter for XR, with a developer study showing measurable accessibility improvement. UIST has awarded best paper to systems with modest math (e.g., Pointing devices, scrolling techniques) when the *impact on practice* was demonstrated. If the developer study shows 3.4× more accessibility fixes per hour, that's the paper.

**On demand:** The XR accessibility tooling market is empty. "Nobody has asked for it" and "nobody would use it" are different claims. Developers don't ask for linters—they discover them via their IDE, adopt them because they're zero-configuration, and come to depend on them. A 2-second, zero-annotation Unity plugin doesn't need evangelism; it needs a Unity Asset Store listing and word of mouth.

### Round 5 — Cross-Team Challenge

**Skeptic vs. Math Assessor:** The Skeptic says "proceed with conditions" and the Math Assessor rates this C+ for math. But they diverge on the *implications*. The Math Assessor notes a missed opportunity: a rigorous analysis of the *completeness gap* (what fraction of bugs can interval methods *never* catch?) would elevate B from "engineering with math support" to "engineering with a sharp theoretical bound." This would give Approach B a B+ result at modest additional effort, making the UIST submission significantly stronger. The Skeptic should *encourage* this addition, not dismiss B's math as entirely ornamental.

**Math Assessor's concession:** The Skeptic is right that B doesn't need its math to exist. But the math determines whether the tool is *useful* vs. *useless*. A wrapping factor of 2× gives a 12% false-positive rate; a wrapping factor of 10× gives an 80% false-positive rate. The difference between "widely adopted linter" and "abandoned experiment" is entirely determined by B1's wrapping-factor bound. So calling the math "ornamental" is wrong—it's load-bearing for *quality*, even if not for *existence*.

**Difficulty Assessor vs. Skeptic:** The Skeptic's 500-line lookup table is an underestimate of what catches obvious bugs but an overestimate of what produces a publishable tool paper. The engineering gap between "script that flags high buttons" and "polished Unity editor plugin with real-time feedback, population-fraction reports, and multi-step reasoning" is precisely what makes B a genuine 4.5/10 difficulty project rather than a weekend hack. The Skeptic collapses this gap to dismiss B; I argue the gap is where the real value lives.

### Verdict: **VIABLE**
No research breakthroughs, but the highest probability of shipping a useful artifact (85% 6-month prototype). The research contribution is thin but honestly scoped. Strongest approach as an engineering substrate and demand validator.

---

## Approach C: Sampling-Guided Symbolic Verification with Coverage Certificates

### Round 1 — Skeptic's Attack

**Attack 1: The Lipschitz assumption is self-defeating.** The entire coverage certificate—the crown jewel, the publication story, the value proposition—rests on the assumption that accessibility bugs satisfy Lipschitz continuity in parameter space. But isolated-point bugs are *exactly what formal verification exists to catch*. A button at the exact boundary of a reachability envelope creates a discontinuous frontier. A grasping interaction requiring precisely 88° of wrist pronation creates a knife-edge failure. **Your guarantee is strongest where it's least needed, and void where it matters most.**

**Attack 2: How is this different from "we sampled a lot"?** A 1M-sample Monte Carlo with zero failures gives >99.999% confidence that the bug rate is <0.001% via Clopper-Pearson. Your coverage certificate adds a Lipschitz-conditioned formal bound. But compliance teams don't understand Lipschitz conditions. They want PASS/FAIL (which MC gives) or PROVEN SAFE (which Approach A gives). "Probably safe with Lipschitz conditions" falls in an uncanny valley: too complicated for practitioners, too weak for formal-methods purists.

**Attack 3: The simpler alternative, again.** Stratified MC + SMT spot-checks on flagged boundaries, *without* the coverage certificate. Report "sampled X bodies, found Y failures, verified Z boundary cases." This gives 95%+ of the practical value without the certificate theory. The 55–75K LoC and the novel math exist primarily to produce a number (ε) that nobody outside the formal methods community will understand or trust.

### Round 2 — Math Assessor's Critique

**Novelty audit:** C1 (coverage certificates) is a genuinely novel formal object assembled from known components. Stratified-sampling concentration bounds + SMT-verified region elimination is a new *combination*, not a new *technique*. The conceptual distance from statistical model checking (Younes & Simmons, Legay et al.) is shorter than claimed—they did this for stochastic systems; C1 adapts it for deterministic parameter-space verification. The self-assessment of "A−" is generous by one notch. **Grade: B+.**

**The Lipschitz problem in math terms:** The certificate is sound *conditional on* the Lipschitz assumption. But the Lipschitz constant L is unknown, must be estimated from samples, and estimation errors flow directly into the certificate's ε bound. Conservative estimation (large L) makes ε uselessly large; aggressive estimation (small L) makes the certificate unsound. This is a fundamental chicken-and-egg problem—you need dense samples near the frontier to estimate L, but estimating L is supposed to tell you *where* to sample densely. The document treats this as an engineering detail; it is actually a foundational limitation.

**Tightness risk:** The certificate will *work* formally. The question is whether it's *meaningfully tighter* than vanilla Clopper-Pearson confidence intervals from the same sample count. If 1M samples and the coverage certificate both give ~99.9% confidence on the same scene, the certificate adds nothing. P(certificate doesn't improve on naïve sampling by ≥5×) ≈ 30%. Math risk overall: 30–35%, with graceful degradation (tool still works as MC + SMT without the formal guarantee).

### Round 3 — Difficulty Assessor's Reality Check

**LoC realism:** Claimed 35–50K novel. Genuinely difficult: ~22–30K. The coverage certificate engine (6–10K, the crown jewel in code form) and the sampling-symbolic handoff controller (3–5K, genuine control-theoretic optimization) are the hard parts. SMT encoding of linearized kinematics is moderate-to-hard with well-understood techniques.

**Trajectory-space curse of dimensionality:** For 5-step interactions with 7-DOF body, the trajectory space is 35-dimensional. Meaningful sample density in 35D requires astronomically many samples. The coverage certificate's ε will be very loose (>0.1) for multi-step interactions unless effective dimension can be reduced. This is an open question—the mitigations (exploit sequential structure, step independence) may not work.

**SMT solver unpredictability:** Z3/CVC5 on quantifier-free nonlinear real arithmetic is NP-hard and practically erratic. Individual queries for a linearized 7-DOF chain may take 0.01s or 100s. If median query time exceeds 1s, the 10-minute compute budget is blown. Must implement timeout-and-skip, degrading the certificate.

**Laptop feasibility:** Single-step: feasible (~10s sampling + ~4 min SMT). Multi-step: tight—10M+ trajectory samples for 5-step interactions takes ~15 minutes and ~1GB for sample storage. The 16GB constraint is binding but not fatal for ≤3-step interactions.

**Timeline:** 6-month prototype probability ~65%. 12-month full system ~40% for a solo developer. The coverage certificate theory requires 2–3 months of proof development interleaved with coding.

### Round 4 — Defense

**On the Lipschitz assumption:** The Lipschitz condition fails at measure-zero sets (exact kinematic singularities, exact boundary points). These edge cases are real but *rare* in practice—they require pathological scene layouts where interactive elements are placed at the mathematical boundary of reachability. The tool can *detect* Lipschitz violations (large gradient estimates from nearby samples with opposite verdicts) and report them explicitly: "Lipschitz violation detected at these body configurations—manual review required." The certificate degrades gracefully: it covers the smooth 95% of parameter space and explicitly flags the non-smooth 5%. This is more honest and more useful than Monte Carlo, which silently misses the boundary cases without knowing they exist.

**On the "just sample more" attack:** The Skeptic's Clopper-Pearson argument conflates two different guarantees. MC gives: "We sampled N bodies and found no failures → the failure rate is probably < ε." The coverage certificate gives: "We have verified (sampled + symbolically proved) all body parameterizations except a region of measure ≤ ε → the *unverified* region has provably bounded volume." The MC guarantee degrades in high-dimensional parameter spaces (curse of dimensionality), while the certificate combines sampling coverage with symbolic proofs that cover entire *regions*, not just points. For a 30-dimensional parameter space, the coverage certificate provably outperforms MC by reducing the effective dimensionality of the unverified region.

**On generalizability:** The coverage certificate framework is genuinely domain-independent. Any parameterized verification problem—robotic workspace certification, drug dosage safety verification, autonomous vehicle scenario coverage—can use the same ⟨S, V, ε, δ⟩ structure. This is the CAV paper's strongest argument: it introduces a new *paradigm* (parameter-space coverage certification), not just a new tool. The XR application is the case study; the framework is the contribution.

**On engineering scope:** At ~55–75K LoC with 65% 6-month prototype probability, this is ambitious but realistic for a researcher with probability/statistics and SMT background. The modular architecture (sampler, SMT engine, certificate engine) allows incremental development and early validation.

### Round 5 — Cross-Team Challenge

**Math Assessor vs. Skeptic:** The Skeptic rates C as "proceed with conditions" and I rate C1 as B+. We agree on the direction but disagree on the magnitude of the Lipschitz problem. The Skeptic says the coverage certificate might be "barely better than Clopper-Pearson." I argue the certificate's value isn't in the *numerical* improvement of ε but in the *structural* guarantee: the certificate identifies *which regions* are verified vs. unverified, enabling targeted follow-up. MC gives a global confidence number; the certificate gives a spatial map of verification status. This structural information is independently valuable even if the headline ε number is similar.

**Skeptic's counter:** If the structural map is the value, you don't need the formal certificate theory—just report "sampled these regions, verified these regions, gap here." The formal ε bound adds nothing practitioners will use. The formal guarantee is for the CAV reviewers; the spatial map is for the users. These are different products, and C1 serves the former, not the latter.

**Difficulty Assessor vs. Math Assessor:** The Math Assessor gives C1 a B+ and says the proof is achievable. I agree on correctness but worry about *practical tightness*. The theoretical certificate will work; the question is whether ε < 0.01 is achievable on a real 30-object scene in 10 minutes on a laptop. If the answer is "ε = 0.08" on typical scenes, the certificate is formally correct but practically disappointing. The gap between "theorem proved" and "theorem useful" is the key implementation risk—and it's not resolvable by mathematical analysis alone; it requires empirical evaluation on real scenes.

### Verdict: **STRONG**
Genuinely novel formal object (coverage certificates) with broad applicability, moderate math risk with graceful degradation, and a realistic engineering path. The Lipschitz limitation and tightness uncertainty are real but manageable. Best risk-adjusted research bet.

---

## Head-to-Head Comparison

| Dimension | **A: PGHA + CAD** | **B: Interval Linter** | **C: Coverage Certificates** |
|---|---|---|---|
| **Value (who cares?)** | 3rd — provable guarantees nobody asked for | 1st — zero-config linter fills an empty niche | 2nd — formal-ish guarantees at practical cost |
| **Math Depth** | 1st — M2 is the only A-grade math (if proved) | 3rd — no novel math (C+) | 2nd — B+ novel formal object with generality |
| **Engineering Difficulty** | 1st — 8/10, frontier open problems | 3rd — 4.5/10, bounded engineering | 2nd — 6.5/10, research + engineering |
| **Feasibility** | 3rd — 15% 12-month probability, 85% fatal-flaw | 1st — 65% 12-month, 25% fatal-flaw | 2nd — 40% 12-month, 45% fatal-flaw |
| **Best-Paper Potential** | 2nd — CAV best paper if M2 works (~8% probability) | 3rd — solid UIST tool paper, not memorable | 1st — generalizable framework with broad impact |
| **Consensus Verdict** | **WEAK** | **VIABLE** | **STRONG** |

## Recommended Winner

**Approach C (Sampling-Guided Symbolic Verification with Coverage Certificates), built on Approach B's engineering substrate.**

The recommended strategy is a phased hybrid:

1. **Months 1–2:** Build Approach B's Tier 1 interval-arithmetic linter as the engineering substrate. This ships earliest, validates demand, and provides a working tool regardless of research outcomes. If wrapping factor >5× at Month 1, switch to Taylor models.

2. **Month 2–3:** Prototype C1 (coverage certificate) on top of the sampling infrastructure. If ε < 0.05 on a 10-object benchmark within a 5-minute budget, commit to Approach C for the research contribution.

3. **Month 3–6:** Develop the full coverage certificate framework, targeting a CAV submission on the generalizable parameter-space verification paradigm with XR as the case study.

4. **Months 6–12:** Polish the combined B+C tool for a UIST submission, including developer study.

**Justification:** Approach C offers the best risk-adjusted combination of novel math (B+ coverage certificates), practical value (formal probabilistic guarantees at practical cost), and generalizability (the framework applies beyond XR). Building on B's substrate ensures a useful artifact ships early and validates demand. Approach A is too risky as the primary path but its M2 result could be pursued as a parallel 2-month time-boxed research sprint if resources allow.

## Key Amendments

| ID | Amendment | Rationale |
|---|---|---|
| **D1** | **Month-1 wrapping gate.** Measure affine-arithmetic wrapping factor on a 4-joint revolute chain. If >5×, switch to Taylor-model propagation. | Validates Approach B substrate feasibility before building on it. |
| **D2** | **Month-2 certificate tightness gate.** Demonstrate ε < 0.05 on a 10-object benchmark with 5-minute budget. If ε > 0.1, abandon the certificate framework; publish B as a standalone UIST tool paper. | Prevents investing 10 months in a certificate that's no better than Monte Carlo. |
| **D3** | **Month-2 Clopper-Pearson benchmark.** Compare coverage certificate ε against vanilla Clopper-Pearson bounds from the same sample count. If the certificate doesn't improve ε by ≥5×, the theoretical contribution is marginal—downscope to tool paper. | Ensures the math actually earns its keep vs. the trivial baseline. |
| **D4** | **Lipschitz violation characterization by Month 3.** Formally characterize which scene configurations violate Lipschitz and measure their frequency on 50+ benchmark scenes. If >20% of scenes have violations, the certificate's general applicability claim dies—restrict scope or add explicit violation-detection and reporting. | The Lipschitz assumption is the certificate's Achilles' heel; it must be empirically validated. |
| **D5** | **Restrict multi-step to ≤3 steps initially.** The trajectory-space curse of dimensionality (35D for 5-step interactions) makes tight certificates unachievable for complex sequences. Defer 5+-step support to future work. | Avoids promising what the math can't deliver at practical compute budgets. |
| **D6** | **Anthropometric data audit.** Supplement ANSUR-II with disability-specific kinematic data (reduced ROM, asymmetric capabilities). If suitable data is unavailable, document the limitation explicitly. | ASSETS/CHI reviewers will ask why able-bodied military data is used for disability-accessibility verification. |
| **D7** | **Month-3 demand signal.** Publish the Tier 1 linter as a free Unity Asset Store plugin. Track downloads and GitHub issues for 8 weeks. If <100 downloads, document this as a known limitation. | Zero demand signal is the existential threat across all approaches; this is the cheapest way to test it. |
