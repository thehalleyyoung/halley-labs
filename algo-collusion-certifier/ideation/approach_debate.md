# CollusionProof: Approach Debate

---

## Part 1: Adversarial Skeptic Critique
I am the Adversarial Skeptic: a composite of a cynical EC program committee member who has desk-rejected three "paradigm-shifting" papers this cycle, a Rust systems engineer who has shipped a proof checker and knows every line-count inflation trick, and a mathematical statistician who has reviewed (and rejected) sloppy applications of empirical process theory. My job is to find the way each approach dies. Every approach has one.

---

### Critique of Approach A: "The Algorithmic Price-Fixing Fire Alarm" (Domain Visionary)

#### Fatal Flaws

**1. The H₀-broad distribution-freeness theorem will not prove in useful form.** Approach A's entire scientific claim rests on M1: uniform Type-I error control over all Lipschitz demand systems × all independent learners. The approach acknowledges this is "the hardest challenge" but then waves at "covering-number arguments over Lipschitz function spaces" as though this were a homework exercise. Here is the reality: the function class F_{H₀} is indexed by *pairs* (demand function, algorithm tuple) where the demand function is infinite-dimensional (Lipschitz on ℝ^N) and the algorithm factor has no standard parameterization — "independent learners" is not a well-defined function class in empirical process theory. You cannot compute metric entropy of a class you cannot formally define. The approach proposes to bound sup_{D ∈ Lip(L)} Corr(π₁(D), π₂(D)), but this supremum depends on *which learning algorithms* are being used — two independent Q-learners on highly correlated demand produce very different correlation structures than two independent bandit algorithms. The supremum over algorithms is over an unspecified, possibly non-compact class. This is not a technical gap; it is a missing definition that, once pinned down, may yield a bound so loose that H₀-broad tests have zero practical power for any T < 10^10.

**Probability this theorem proves in publishable form: ~35%.** It will likely prove for H₀-narrow (linear demand × Q-learning) and H₀-medium (parametric demand × no-regret learners). The "broad" claim will survive as a conjecture or as an asymptotic result with unknown convergence rate — which is exactly what the depth check already flagged and Approach A claims to have resolved.

**2. The proof checker is a soundness liability, not a selling point.** Approach A proposes a de novo proof language with ~15 axiom schemas and ~25 inference rules, all designed from scratch for game-theoretic collusion properties. Every axiom is a potential soundness hole. The approach claims the 2,500 LoC trusted kernel is "small enough to audit by hand." I have audited proof-checker kernels. A 2,500 LoC kernel with 15 axiom schemas covering statistical inference, rational-arithmetic verification, and game-theoretic properties is *not* auditable in an afternoon. It is a month-long verification effort for an expert. Any axiom that is subtly unsound — e.g., an axiom that assumes independence where conditional independence holds, or an axiom that elides a measurability condition — produces certificates that pass verification but assert false conclusions. The approach has zero plan for formal verification of the axiom system itself. "Iterate on the axiom system" is not a soundness argument; it is an invitation for bugs to survive into production certificates.

**3. Layer 0 alone is a statistical screening tool, not a "certifier."** The entire framing — "fire alarm," "proof-carrying certificates," "evidence bundle" — implies Layer 0 produces something qualitatively different from existing screening. It does not. Layer 0 is a battery of hypothesis tests composed with Holm-Bonferroni. The "certificate" wrapping does not make the statistical conclusion stronger; it makes it *auditable*, which is valuable but not revolutionary. A competition economist who runs a well-designed bootstrap test with documented code already produces "auditable" evidence. The marginal value of Approach A's Layer 0 over a well-implemented battery of existing tests (say, Assad et al.'s methodology with formal error control) is incremental, not categorical. The "fire alarm" metaphor conceals this.

#### Over-Optimistic Claims

| Claim | Self-Score | Honest Score | Why |
|-------|-----------|-------------|-----|
| Value | 8/10 | 6/10 | Layer 0 is a screening tool, not courtroom evidence. Regulatory adoption requires trust-building the approach waves away. The three stakeholder groups are real but the urgency is overstated — DG-COMP's actual bottleneck is legal authority, not statistical tools. |
| Difficulty | 7/10 | 6/10 | Layer 0 avoids the hardest subproblems. The proof checker at ~5K LoC (with ~2,500 trusted) is small for a proof assistant, period. Bertrand/Cournot 2-player is textbook. The "hard" part (M1 distribution-freeness) is genuinely hard, but the rest is competent engineering, not research difficulty. |
| Potential | 8/10 | 6/10 | M1 is formulation novelty — the depth check already downgraded this to B+. The conditional completeness story (C3-dependent) is a liability, not a feature. EC reviewers who study mechanism design will notice that "sound but maybe not complete" is the default for any conservative statistical test. The PCC framing is novel but the actual mathematical contribution is a composite hypothesis test. |
| Feasibility | 7/10 | 6/10 | Realistic if scoped to H₀-narrow and H₀-medium only. The 30-minute smoke mode claim is plausible. But "achievable" and "will produce a best paper" are different — the approach is achievable *as an engineering artifact*. |

#### Hidden Risks

1. **The axiom system will require 3+ design iterations**, each invalidating all previously generated certificates. Approach A budgets zero time for this. Realistic axiom-system stabilization: 6-8 weeks of focused work, not the implicit "design it and move on."
2. **The Berry-Esseen finite-sample corrections for H₀-medium may have constants so large that they dominate the test statistic.** Berry-Esseen gives O(1/√T) convergence with constants that depend on third moments of the test statistic under the null — these third moments are *unknown* for the composite null and must themselves be bounded, creating a regress.
3. **The 8 competitive validation scenarios will not catch systematic α-inflation.** To detect α-inflation from 0.05 to 0.08 with 80% power, you need ~1,500 runs per scenario (binomial test calculation). 50 seeds × 8 scenarios = 400 runs — insufficient. The empirical Type-I validation will have wide confidence intervals and prove nothing about whether α-control actually holds.
4. **The "perfect timing" argument cuts both ways.** If the RealPage case is decided before the paper is published, the regulatory motivation evaporates from "defining the standard" to "providing an alternative to the standard the court already adopted."

#### Score Corrections

| Criterion | Self-Score | Corrected | Justification |
|-----------|-----------|-----------|---------------|
| Value | 8 | 6 | Screening tool, not certifier. Regulatory adoption is years away. |
| Difficulty | 7 | 6 | Layer 0 is the easy part of the project. |
| Potential | 8 | 6 | Formulation novelty ≠ mathematical novelty. C3 dependency weakens the story. |
| Feasibility | 7 | 6 | Achievable but H₀-broad will be a conjecture. |
| **Composite** | **7.5** | **6.0** | |

---

### Critique of Approach B: "Resolving the Collusion Detection Barrier" (Math Depth)

#### Fatal Flaws

**1. Five load-bearing theorems is a suicide pact.** Approach B stakes *everything* on proving five new theorems (C3′ deterministic, C3′ stochastic, M4′ upper bound, M4′ lower bound, M8 impossibility), each declared "load-bearing" — meaning "removing any one collapses either soundness, completeness, optimality, or the model's necessity." This is not a strength; it is a single point of failure replicated five times. The approach itself admits feasibility is 5/10 and the realistic outcome is "3 of 5 theorems proved in full strength." But the entire *narrative* of the paper — the "Collusion Detection Barrier Theorem" as a clean dichotomy — requires *all five*. With only 3/5, you have: a completeness theorem for deterministic automata (good but not a barrier theorem), a probably-correct upper bound (fine), and maybe the impossibility result (which is actually the easiest of the five). You do *not* have: the stochastic extension (which is the practically relevant case for ε-greedy Q-learning, the dominant deployed algorithm class), or the minimax lower bound (which is the "optimality" claim). The paper that ships is "completeness for deterministic automata with an upper bound" — a solid EC contribution but not a best paper, and certainly not the "Collusion Detection Barrier Theorem" as advertised.

**Probability that all five theorems prove: ~15%.** Probability that 4/5 prove: ~30%. The stochastic C3′ and the minimax lower bound are the two most likely to fail. The stochastic C3′ fails because the mixing-time dependence τ_mix may be exponential for natural strategy classes (Boltzmann Q-learning with low temperature has τ_mix = Θ(exp(1/T))), rendering the bound Δ_P ≥ η/(M·N·τ_mix) vacuous. The minimax lower bound fails because the Le Cam construction requires embedding a collusive trajectory in the competitive null, which requires constructing a Lipschitz demand function that mimics an M-state automaton's output distribution — this construction may not exist for all M when the price grid is finite and small.

**2. The coupling argument for stochastic C3′ has a gap.** The proof sketch says: "couple the on-path chain with the post-deviation chain; since the chains diverge and the state space is finite, they must re-couple within O(τ_mix) steps." This is wrong as stated. Two Markov chains on the same state space with *different* transition kernels (which is the case here — the deviating player's kernel changes) do not necessarily re-couple within O(τ_mix) of *either* chain. The coupling time between chains with different kernels is bounded by the mixing time of the *slower* chain, which is the post-deviation chain — and the post-deviation chain's mixing time is unknown and potentially much larger than the on-path chain's mixing time. The deviating player's new strategy may create absorbing states or near-absorbing traps in the joint chain. This is not a minor technical issue; it is a gap in the proof strategy that may require an entirely different technique.

**3. The metric entropy computation for the algorithm factor is undefined.** M1′ requires computing the entropy H_algo(ε) of the class of "independent no-regret learners." But "independent no-regret learners" is not a well-defined function class in any formal sense. Which regret definition? Over what time horizon? What information structure? A Q-learner with ε-greedy exploration is not a no-regret learner in the standard sense (it converges to a fixed policy, not a Hannan-consistent sequence). If you restrict to Hannan-consistent learners, the class is enormous and may have infinite metric entropy at any resolution ε. If you restrict to specific algorithm families, you lose the "any independent learner" universality claim. The approach hand-waves this with "exploit the independence structure" but independence alone does not bound entropy — independent algorithms can produce arbitrarily complex trajectory distributions.

**4. The engineering artifact is an afterthought.** Approach B's mathematical ambition is inversely proportional to its engineering feasibility. The proof checker kernel expands from ~2,500 to ~3,500 LoC to encode "automaton decomposition lemmas, coupling arguments for stochastic strategies, and entropy integral computations." Encoding Markov chain coupling arguments in a domain-specific proof language is a research project in its own right — existing proof assistants (Lean, Coq) struggle with probabilistic coupling proofs. The approach gives zero architectural detail for how the proof checker handles these new proof objects. The 60K LoC implementation "must track the evolving theorem statements" — meaning the code cannot stabilize until the math is done, and the math is uncertain. This is a recipe for a prototype that demonstrates 60% of the theory on 40% of the cases.

#### Over-Optimistic Claims

| Claim | Self-Score | Honest Score | Why |
|-------|-----------|-------------|-----|
| Value | 7/10 | 6/10 | Theory-concentrated value. Regulators don't need minimax-optimal bounds; they need a tool that works. The "data budget" framing is nice but the constants in T* = Θ̃(M²σ²/(η²Δ_P²)) will be too loose for practical guidance. |
| Difficulty | 9/10 | 8/10 | Genuinely hard math, but the 9/10 assumes all five theorems are attempted. If 2/5 are weakened to conjectures, difficulty drops because you're proving less. |
| Potential | 9/10 | 7/10 | A 3/5 theorem paper is a strong EC submission, not a best paper. The "dichotomy result" narrative requires the impossibility theorem + the stochastic completeness, and both may fail. Without the dichotomy, the paper is "completeness for deterministic automata + an upper bound" — Grade A but not award-caliber. |
| Feasibility | 5/10 | 4/10 | Honestly, the approach is candid about this. But 5/10 still implies a coin-flip; the reality is worse because the five theorems are *correlated* in their difficulty — if the stochastic coupling fails, the minimax lower bound's construction (which uses similar coupling ideas) likely fails too. |

#### Hidden Risks

1. **The Roughgarden smoothness analogy is misleading.** Roughgarden's smoothness framework (EC 2009) was a *simplification* — a single clean condition that unified a zoo of prior price-of-anarchy results. Approach B's "Collusion Detection Barrier Theorem" is a *complication* — it requires five new theorems to state one dichotomy. EC best papers simplify; Approach B complexifies.
2. **The M8 impossibility theorem may be trivially true.** The stealth-collusion strategy construction — "mimic competitive behavior for T rounds, then punish starting at round T+1" — is a standard delayed-punishment argument that experienced game theorists will find obvious. The "novelty" claim (Grade A) for formalizing something the community already informally knows is generous. Reviewers may dismiss M8 as "clearly true, not novel."
3. **The 3,500 LoC proof checker is no longer auditable.** The approach's own selling point — "the checker is small enough to audit" — is undermined by expanding it 40% to handle stochastic coupling proofs. At 3,500 LoC with 22 axiom schemas, the checker is entering the territory where soundness bugs become likely and audit becomes impractical in a single afternoon.
4. **Math-engineering coupling creates schedule chaos.** When the stochastic C3′ proof fails (or weakens), the proof checker's axiom system must be revised, the certificate format must change, the evaluation scenarios must be updated, and the paper narrative must be restructured. This coupling means a late-stage mathematical setback cascades into weeks of engineering rework.

#### Score Corrections

| Criterion | Self-Score | Corrected | Justification |
|-----------|-----------|-----------|---------------|
| Value | 7 | 6 | Theory value is real but concentrated. Regulatory applicability is diluted. |
| Difficulty | 9 | 8 | Genuinely hard but self-score assumes full 5-theorem program. |
| Potential | 9 | 7 | 3/5 theorems ≠ best paper. Dichotomy requires all components. |
| Feasibility | 5 | 4 | Correlated theorem failures. Math-engineering coupling. |
| **Composite** | **7.5** | **6.25** | |

---

### Critique of Approach C: "The Compositional Certification Kernel" (Difficulty Assessor)

#### Fatal Flaws

**1. Engineering difficulty is not mathematical novelty, and EC rewards the latter.** Approach C's thesis is that the *composition* of seven subsystems across four paradigms is the core difficulty. This is true — and completely irrelevant to an EC program committee. EC is a theory venue with a systems-appreciation streak, not a systems venue with a theory-appreciation streak. A paper whose main contribution is "we made f64 → ℚ conversion work across a Rust-Python boundary with phantom-type segment isolation" will be reviewed by game theorists and economists who do not care about phantom types. The "compositional soundness challenge is publishable in its own right" claim (from the Best-Paper Potential section) is delusional — it is publishable at a software engineering venue (ICSE, FSE), not at EC. Approach C undersells the math and oversells the engineering.

**2. The math is the weakest of all three approaches.** Approach C lists four "new math" contributions: (1) correlation supremum bound, (2) peeling argument for adaptive sampling, (3) automaton C3 proof, and (4) error propagation algebra. Items (1) and (3) are shared with Approaches A and B. Item (4) is straightforwardly interval arithmetic composition — a "novel combination" at Grade C, meaning "not novel." Item (2) is the only distinctive mathematical contribution, and it is a Layer 1-2 result that doesn't apply to Layer 0. For the paper artifact (Layer 0 focus), Approach C has *zero distinctive mathematical contributions beyond what Approaches A and B already claim.* The engineering is the differentiator, and the engineering cannot carry an EC paper.

**3. "Silent bugs" are a risk, not a feature.** Approach C repeatedly emphasizes that bugs in the compositional pipeline are "statistically invisible" — they don't crash the system but produce certificates with inflated false positive rates. This framing is intended to show engineering difficulty, but it actually reveals a *fundamental validation problem*: how do you know the system is correct? The proposed solutions — dual-path verification, statistical meta-testing, phantom-type enforcement — are reasonable engineering practices, but they cannot prove the absence of all subtle composition bugs. The approach's own admission that a single rounding error in one comparison out of thousands can invalidate a certificate, and that this bug manifests in 1/10,000 certificates, means the evaluation suite (30 scenarios × 10 seeds = 300 certificates) has a ~3% chance of encountering such a bug and no ability to detect it statistically. The system may ship with undetected soundness bugs and the evaluation cannot catch them.

**4. The performance analysis reveals a schedule risk the approach ignores.** Challenge 5 calculates that full 3-player counterfactual analysis takes ~100 minutes per scenario. With 15 standard-mode scenarios, that's 25 hours just for counterfactual analysis — and this assumes the >100K rounds/sec Rust target is hit. But Challenge 3 reveals that the PyO3 FFI overhead may reduce throughput to ~500K rounds/sec *before any algorithm computation*, and with Python algorithms (the RL agents), actual throughput will be lower. If throughput drops to 30K rounds/sec (realistic with GIL contention and Python algorithm evaluation), the 100-minute per-scenario estimate becomes 330 minutes = 5.5 hours per scenario. Standard-mode evaluation balloons from 4 days to ~10 days. Development iteration with the --standard tier becomes impractical, forcing exclusive reliance on --smoke mode, which tests only 5 scenarios and cannot validate compositional correctness across the full suite.

#### Over-Optimistic Claims

| Claim | Self-Score | Honest Score | Why |
|-------|-----------|-------------|-----|
| Value | 7/10 | 6/10 | Same value as Approaches A and B — the engineering framing doesn't add value to *regulators*. It adds value to systems researchers, who are not the target audience. |
| Difficulty | 8/10 | 7/10 | The compositional challenge is real but not 8/10 for an experienced Rust+Python engineer. The individual subsystems are well-understood. The integration is hard but not *research* hard — it's *engineering* hard. The distinction matters for EC. |
| Potential | 7/10 | 5/10 | Engineering-forward papers do not win EC best paper awards. The math is the weakest of the three approaches. The "artifact is the proof of concept" argument works only if the theory is already strong — which it isn't in this approach. |
| Feasibility | 6/10 | 6/10 | Fair self-assessment. The schedule risk from PyO3 performance is underweighted but the overall feasibility is realistic. |

#### Hidden Risks

1. **Phantom-type segment isolation sounds elegant but is brittle.** The Rust type system can prevent cross-segment data access, but it cannot prevent a developer from accidentally assigning two sub-tests to the same segment tag. The tag assignment is a human decision, not a type-level guarantee. One mistagged segment silently inflates α.
2. **The Merkle-integrity evidence bundle requires reproducible Rust builds.** Reproducible builds in Rust are possible but require pinned toolchain versions, stripped binaries, and careful handling of build metadata. In practice, achieving byte-identical builds across different developer machines and CI environments takes weeks of CI engineering. The approach budgets ~1,500 LoC for this — realistic for the Merkle infrastructure, but the CI/build engineering is unbudgeted and easily 2-3 weeks of DevOps work.
3. **The "dual-path verification" (f64 + exact rational) doubles the computation time of certificate generation.** If certificate generation currently takes 1-5 minutes (per the crystallized problem), dual-path verification makes it 2-10 minutes. For the standard evaluation (15 scenarios × seeds), this adds 30-150 minutes. Tolerable, but the approach presents it without acknowledging the cost.
4. **Approach C has no distinctive evaluation advantage.** All three approaches use the same 30-scenario evaluation suite. Approach C's emphasis on the engineering pipeline means it should invest *more* in integration testing — but the evaluation plan is identical to the other approaches. Where are the integration-specific tests? Where are the cross-boundary regression tests? The approach identifies the right risks but proposes the same evaluation as everyone else.

#### Score Corrections

| Criterion | Self-Score | Corrected | Justification |
|-----------|-----------|-----------|---------------|
| Value | 7 | 6 | Engineering framing doesn't add stakeholder value. |
| Difficulty | 8 | 7 | Engineering hard ≠ research hard. EC cares about the latter. |
| Potential | 7 | 5 | Weakest math of all three. Engineering cannot carry an EC paper. |
| Feasibility | 6 | 6 | Fair self-assessment. |
| **Composite** | **7.0** | **6.0** | |

---

### Cross-Approach Comparison

#### Where Each Fails and the Others Succeed

| Dimension | Approach A Fails | Approach B Succeeds | Approach C Succeeds |
|-----------|-----------------|--------------------|--------------------|
| **Mathematical depth** | M1 is formulation novelty; techniques are borrowed from semiparametric statistics. The depth check downgraded this to B+. | C3′ + M4′ lower bound + M8 impossibility are genuine new theorems. Even if only 3/5 prove, the math is deeper. | Weakest math, but correctly identifies that engineering composition is an underappreciated challenge. |
| **Completeness story** | C3-conditional. No plan to resolve C3 beyond restricted classes. | Explicit plan to resolve C3 with staged difficulty and honest fallbacks. | Same C3 dependency as A, with no additional resolution plan. |
| **Feasibility** | Most feasible of the three. Layer 0 is self-contained and achievable. | Least feasible. Five load-bearing theorems with 60% success probability. | Middle ground. Engineering-focused but schedule risks from PyO3 performance. |

| Dimension | Approach B Fails | Approach A Succeeds | Approach C Succeeds |
|-----------|-----------------|--------------------|--------------------|
| **Deliverable certainty** | May ship a 3/5 paper that doesn't match its own narrative. The "barrier theorem" requires all components. | Layer 0 ships regardless. The contribution is modular and degrades gracefully. | The engineering pipeline is the contribution; it ships or it doesn't, but there's no "partial narrative" problem. |
| **Practical relevance** | Minimax bounds with unknown constants don't help regulators decide how much data to collect. | Layer 0 is directly usable as a screening tool today. | Equally usable but adds no practical value beyond A. |
| **Engineering architecture** | Engineering is an afterthought. The 3,500 LoC checker encoding coupling proofs is hand-waved. | Honest about the trust boundary and proof checker scope. | Best engineering design of the three. Phantom-type isolation, dual-path verification, and interval arithmetic are well-motivated. |

| Dimension | Approach C Fails | Approach A Succeeds | Approach B Succeeds |
|-----------|-----------------|--------------------|--------------------|
| **EC best-paper narrative** | "We built a compositionally sound pipeline" is not a theory contribution. | "First composite hypothesis test with game-theoretic null" is a clear theory contribution. | "New theorem in repeated game theory + impossibility result" is the strongest theory narrative. |
| **Novel mathematics** | Zero distinctive math beyond what A and B already claim. The peeling argument (M2) is Layer 1+ only. | M1 is novel in formulation if not technique. | C3′, M4′, M8 are each novel. |
| **Researcher impact** | Generates follow-up work in software engineering (compositional verification), not in game theory or statistics. | Generates follow-up in statistics (composite testing with structured nulls). | Generates follow-up in game theory (converse folk theorems), statistics (minimax detection), and complexity (detection barriers). |

#### The Uncomfortable Truth All Three Share

All three approaches share the same fundamental vulnerability: **the composite hypothesis test M1 may have trivial statistical power against realistic collusion alternatives when tested against H₀-broad.** If the correlation bound over Lipschitz demand × independent learners is so loose that the test cannot reject H₀-broad for any practical T, then:

- Approach A's "screening tool" detects only the most flagrant collusion patterns (H₀-narrow rejection only), which existing ad-hoc methods already catch.
- Approach B's "minimax-optimal" bound proves optimality of a test with trivial power — a theoretically beautiful but practically useless result ("your test is provably as good as possible; unfortunately, 'as good as possible' is terrible").
- Approach C's "compositionally sound pipeline" composes correct sub-tests into a correct system that never rejects.

None of the three approaches honestly confronts the possibility that H₀-broad is too broad for useful testing. The tiered null hierarchy is the right structural response, but all three approaches treat H₀-narrow and H₀-medium as fallbacks rather than potentially being the *ceiling* of the contribution.

---

### Skeptic's Ranking

#### Rank 1: Approach A (Domain Visionary) — Corrected Composite: 6.0

**Why first despite being "boring."** Approach A is the only approach that can ship a complete, honest paper with a working artifact under realistic conditions. Layer 0 is achievable. H₀-narrow and H₀-medium proofs are tractable. The PCC framing is genuinely novel even if the math is formulation-level. The regulatory timing is real. The approach's weakness — lack of deep mathematical novelty — is also its strength: it doesn't promise theorems it can't deliver. An honest EC submission with M1 (composite test), conditional M4 for restricted classes, and M6 (certificate verification) with a working 60K LoC artifact is a *solid accept* at EC and a *possible* best paper if the presentation is excellent. The ceiling is lower, but the floor is much higher.

#### Rank 2: Approach B (Math Depth) — Corrected Composite: 6.25

**Why second despite the highest ceiling.** Approach B has the best paper at EC if everything works. But the probability that everything works is ~15%. The expected outcome — 3/5 theorems with the stochastic C3′ and minimax lower bound as conjectures — produces a paper that is stronger than Approach A in theory but weaker in narrative coherence (the "barrier theorem" narrative falls apart). The math-engineering coupling means late-stage theorem failures cascade into engineering rework. If I were betting on best-paper probability × probability of completion, Approach B has: 0.15 × 0.4 (best paper | all theorems) + 0.3 × 0.15 (best paper | 4/5 theorems) + 0.35 × 0.05 (best paper | 3/5 theorems) ≈ 12% expected best-paper probability. Approach A has maybe 8% with higher completion certainty. The risk-adjusted value favors B slightly on the upside, but A on the downside — and in research, the downside matters because you have to actually ship the paper.

#### Rank 3: Approach C (Difficulty Assessor) — Corrected Composite: 6.0

**Why last despite sound engineering.** Approach C is the most architecturally honest and the least publishable at EC. The engineering contributions are real — phantom-type segment isolation, dual-path verification, the numerical-to-formal bridge — but they are systems contributions at a theory venue. The math is the weakest of the three, with zero distinctive contributions beyond what A and B already claim. The approach would rank first if the target venue were ICSE or FSE; it ranks last for EC. The irony: Approach C correctly diagnoses the hardest *engineering* problem in the project, but the project's success is determined by *mathematical* contributions at a *theory* venue.

#### Final Assessment

The honest recommendation is **Approach A's scope with Approach B's mathematical ambition as a stretch goal**: ship Layer 0 with H₀-narrow/H₀-medium proofs and the PCC framework as the guaranteed contribution, while investing available mathematical bandwidth in the deterministic C3′ proof (Approach B's Stage 1) and the M8 impossibility theorem (the easiest of B's five theorems). Approach C's engineering insights (phantom-type isolation, dual-path verification) should be adopted as *implementation practices* within whichever approach is selected, not as a paper-level contribution. This hybrid avoids A's ceiling problem, B's floor problem, and C's venue-mismatch problem.

---

## Part 2: Mathematical Depth Critique
**Evaluator role**: Math Depth Assessor performing cross-critique duty
**Method**: Theorem-by-theorem achievability, load-bearing, grade audit, difficulty calibration, and risk assessment across all three approaches

---

### Approach A: Math Audit

Approach A proposes four mathematical contributions: M1, M4/C3, M6, and M7. The math program is deliberately conservative — lead with formulation novelty, prove restricted cases, defer hard generalizations.

#### M1: Composite Hypothesis Test over Game-Algorithm Pairs (Self-graded: A)

**Achievability.**
- *Full generality (H₀-broad, Lipschitz demand × independent learners)*: 35–45% probability of a complete proof. The supremum of cross-firm correlation over all Lipschitz demand functions is a well-posed infinite-dimensional optimization problem, but bounding it tightly enough for practical power (not just a vacuous bound) is the real challenge. Covering-number arguments for Lipschitz classes are standard (Kolmogorov-Tikhomirov), but the game-theoretic structure — where demand mediates between firms' independent learning dynamics — adds genuine complication. The Berry-Esseen finite-sample correction for T in the 10⁵–10⁷ range may yield remainder terms that dominate the signal.
- *Restricted form (H₀-narrow, linear demand × Q-learning)*: 90%+ probability. This is a concrete finite-dimensional calculation. The cross-firm correlation under linear demand with independent Q-learners can be bounded analytically via the closed-form best-response dynamics.
- *H₀-medium (parametric demand × no-regret learners)*: 70% probability. Standard but requires careful handling of the parametric family's dimensionality.

**Load-bearing test.** Remove M1 and Layer 0 has no formal Type-I error guarantee — it becomes a heuristic screen indistinguishable from existing ad-hoc methods. M1 is **fully load-bearing**. The artifact's entire claim to scientific rigor over existing correlation screens depends on this theorem.

**Grade audit.** Grade A is generous. The *formulation* (testing against a game-theoretic null) is novel, but the *techniques* required (covering numbers, empirical process theory, Berry-Esseen bounds) are well-established in semiparametric statistics. The depth check correctly identified this: "formulation novelty, not mathematical novelty." **Honest grade: B+ for H₀-narrow/medium, A− for H₀-broad if the bound is tight and non-vacuous.** The gap between "apply known techniques to a new domain" (B+) and "the domain structure forces genuinely new techniques" (A) depends on whether the game-theoretic correlation structure requires ideas beyond standard covering arguments. My assessment: it probably doesn't for the upper bound, but a tight characterization (matching lower bound on achievable correlation) would push it to A.

**Difficulty calibration.** H₀-narrow: 1–2 person-months. H₀-medium: 2–3 person-months. H₀-broad with tight bound: 4–6 person-months, with significant risk of vacuous constants.

**Risk assessment.** Medium risk. The restricted forms (narrow/medium) will almost certainly succeed, giving Layer 0 meaningful guarantees. H₀-broad failure means the contribution is "a principled test for markets where you know the demand family" rather than "a universal screen." This is still publishable but significantly weaker — it reintroduces the expert-judgment dependency the system claims to eliminate.

#### M4/C3: Folk Theorem Converse for Bounded-Recall Strategies (Self-graded: A*)

**Achievability.**
- *Deterministic automata (grim-trigger, tit-for-tat, bounded-state Mealy machines)*: 85% probability. The argument via cycle detection in the product automaton state space and contradiction through profitable deviation is clean and well-motivated. The bound Δ_P ≥ η/(M·N) follows from a counting argument. This is a hard but tractable theorem.
- *General bounded-recall strategies (including stochastic)*: 30–40% probability. The stochastic extension introduces mixing time dependencies that may make the bound vacuous (Δ_P ≥ η/(M·N·τ_mix) with τ_mix potentially exponential in M). Approach A wisely leaves this as an open conjecture.

**Load-bearing test.** Remove M4/C3 and the system retains soundness but loses completeness entirely. The contribution becomes: "if we say it's collusion, we're right; if we say it's not, we don't know." This is still valuable (one-sided guarantees are standard in hypothesis testing), but it dramatically weakens the paper's narrative. M4/C3 is **conditionally load-bearing** — it converts the contribution from a sound screening tool to a sound-and-complete certification framework.

**Grade audit.** Grade A* (conditional on C3) is honest for the full statement. The unconditional result for deterministic automata is **Grade A** — a genuinely new theorem connecting automaton theory to collusion detectability. This is the strongest mathematical claim in Approach A.

**Difficulty calibration.** Deterministic case: 2–3 person-months. Stochastic case: open-ended, possibly 6+ person-months with uncertain outcome. Approach A's strategy of proving restricted cases and conjecturing the rest is the correct risk management.

**Risk assessment.** Low risk for the restricted result, high risk for the general conjecture. If the stochastic extension fails, the paper still has a clean theorem for deterministic automata covering all practical pricing algorithms (Q-learning with discretized tables, lookup strategies, etc.). The honest framing — "unconditional for deterministic bounded-recall, conjectured for stochastic" — is defensible at EC.

#### M6: Certificate Verification Soundness (Self-graded: C)

**Achievability.** 80–90% probability. This is a metatheorem about the proof checker: if the axiom system is sound and the checker correctly implements the inference rules, then accepted certificates are valid. The hard part is *verifying* axiom soundness (each of ~15 axioms must be a true statement about the game-theoretic domain), not proving the metatheorem itself.

**Load-bearing test.** Remove M6 and certificates are unverified data bundles — the entire PCC paradigm collapses. **Fully load-bearing** for the certification framing, though not for the statistical testing contribution (M1 works without certificates).

**Grade audit.** Grade C is honest and refreshingly accurate. This is a careful verification exercise, not a mathematical innovation. The novelty is in the *domain* (game-theoretic axioms), not the *technique* (standard proof-checker soundness arguments).

**Difficulty calibration.** 2–3 person-months, dominated by the iterative axiom design process (getting the axiom system expressive enough yet sound). The metatheorem itself is straightforward once the axioms are fixed.

**Risk assessment.** Low-medium risk. The main danger is a subtle unsound axiom that enables vacuous proofs. Mitigation: extensive testing with adversarial certificate attempts that should fail. This is engineering verification, not mathematical risk.

#### M7: Directed Closed Testing (Self-graded: D)

**Achievability.** 95%+ probability. Holm-Bonferroni with a collusion-informed ordering is textbook multiple testing with a domain-specific twist.

**Load-bearing test.** Remove M7 and the composite test uses a generic omnibus procedure with lower power. The system still works but is less practically useful. **Partially load-bearing** — improves power but isn't structurally necessary.

**Grade audit.** Grade D is honest. This is engineering application of standard methods.

**Difficulty calibration.** 0.5–1 person-month. The difficulty is in choosing the right ordering (requires economic intuition), not in the math.

**Risk assessment.** Negligible.

#### Approach A Summary Verdict

The math program contains 2 load-bearing theorems (M1, M4/C3), 1 verification exercise (M6), and 1 engineering contribution (M7). Total expert effort: ~6–10 person-months for the achievable components. The program is **honest and achievable but mathematically thin for EC best paper.** The depth check's concern that M1 is formulation novelty is valid — M1 alone, even with H₀-broad, is unlikely to win best paper. The combination of M1 + restricted M4/C3 + M6 is a solid paper but relies heavily on the novelty of the *artifact category* (PCC for economics) rather than deep theorems. This is a risk: EC program committees vary year to year in how much they weight artifact novelty vs. theorem depth.

**Probability of EC acceptance: 55–65%. Probability of EC best paper: 10–15%.**

---

### Approach B: Math Audit

Approach B proposes five new theorems: M1′, C3′, M4′, M8, M5′. This is the maximally ambitious math program — every component is load-bearing, every theorem is novel, and the total represents a substantial research agenda.

#### M1′: Non-Asymptotic Uniform Testing via Empirical Process Theory (Self-graded: A)

**Achievability.**
- *Full statement (computable, non-asymptotic remainder R(T,L,N))*: 50–60% probability. The metric entropy computation for the product class (Lipschitz demand × independent learners) is novel. The demand-function factor has standard entropy bounds; the learning-algorithm factor is the open question. Independent no-regret learners produce trajectory distributions whose entropy depends on the algorithm class — and "all independent learners" is an enormous class. The entropy may be infinite without additional structural assumptions (e.g., bounded regret rate, Lipschitz value functions).
- *Restricted (parametric demand families)*: 90%+ probability, with R = 0 by standard parametric theory.
- *Semi-restricted (Lipschitz demand × bounded-regret learners)*: 65% probability. Adding a regret bound constrains the algorithm class enough for finite entropy, but the resulting R(T,L,N) may have impractical constants.

**Load-bearing test.** Remove M1′ and the composite test is valid only asymptotically — finite-sample certificates have no formal guarantee. **Fully load-bearing** for the non-asymptotic certification claim. However, Layer 0 still "works" with asymptotic validity and empirical calibration; the formal finite-sample guarantee is scientifically important but not operationally essential for a screening tool.

**Grade audit.** Grade A is justified *if* the metric entropy computation for the algorithm factor is genuinely new and non-trivial. My assessment: it is novel (no prior entropy computation for independent learner trajectory distributions), but the result may be technically straightforward once the right structural assumption is identified (e.g., bounded regret implies bounded Rademacher complexity of trajectory distributions, which implies bounded entropy via standard chaining). **Honest grade: A− if the computation reveals new structure; B+ if it reduces to standard chaining after the right definition.**

**Difficulty calibration.** 3–5 person-months. The novel part (algorithm-factor entropy) is a genuine research question. The composition via Dudley's integral and Talagrand's inequality is technically demanding but follows established patterns.

**Risk assessment.** Medium-high. The full non-asymptotic bound may have constants too large for practical T (requiring T > 10⁸ for the remainder to be negligible). This is a common failure mode in non-asymptotic statistics — the bound is correct but vacuous for all practically relevant sample sizes. Mitigation: report the bound honestly and show empirically that the test is well-calibrated at practical T.

#### C3′: Folk Theorem Converse for Finite-State Strategies (Self-graded: A+)

**Achievability.**
- *Deterministic automata*: 85% probability (same as Approach A's assessment — this is the same theorem).
- *Stochastic automata with τ_mix ≤ poly(M)*: 50–60% probability. The coupling argument between on-path and post-deviation Markov chains is well-motivated, and the bound Δ_P ≥ η/(M³·N) for polynomial-mixing chains is plausible. The subtlety is that the deviating player's transition kernel changes, which may create coupling difficulties that require chain-specific analysis.
- *Full characterization of the C3′ frontier*: 15–25% probability. This is an open problem in Markov chain theory. Characterizing exactly when C3′ holds (the τ_mix < exp(M)/N conjecture) would be a major result in its own right.

**Load-bearing test.** Remove C3′ and the system has no completeness guarantee for any strategy class beyond grim-trigger/tit-for-tat. **Maximally load-bearing** — this is the theorem the entire approach is structured around.

**Grade audit.** Grade A+ is defensible for the full statement (deterministic + stochastic). The deterministic case alone is Grade A — connecting automaton cycle structure to collusion detectability via the profitable-deviation contradiction is genuinely novel in repeated game theory. The stochastic extension using Markov chain coupling in a game-theoretic context is a legitimate methodological innovation. However, A+ implies "field-defining breakthrough" — this is a strong new theorem, not a paradigm shift. **Honest grade: A for deterministic, A+ only if the stochastic extension reveals deep structure about the detectability/mixing-time tradeoff.**

**Difficulty calibration.** Deterministic: 2–3 person-months. Stochastic (polynomial mixing): 3–5 person-months. Full frontier characterization: 6–12 person-months with uncertain outcome.

**Risk assessment.** The deterministic result is low risk and high value. The stochastic extension is medium risk — the coupling argument may work cleanly for ε-greedy exploration but break down for adversarial mixing constructions. The full frontier characterization is high risk and should be treated as aspirational.

#### M4′: Tight Sample Complexity for Collusion Detection (Self-graded: A)

**Achievability.**
- *Upper bound*: 75% probability. The upper bound follows from C3′ by standard arguments (once you know the punishment signal magnitude, sample complexity follows from standard detection theory). The dependence on M², σ², η², and τ_mix is natural. The dependence on K (number of sub-tests) through log(K/α) is standard from Bonferroni.
- *Lower bound*: 40–50% probability. This is the hard part. Constructing a competitive distribution P₀ that matches an M-state collusive distribution P₁ in total variation through T* rounds requires finding a Lipschitz demand function whose induced trajectory distribution mimics the automaton's output. The claim that "the demand function has enough degrees of freedom to mimic the automaton" is plausible but not obvious — the demand function maps prices to quantities continuously, while the automaton has discrete state transitions. The embedding may fail for automata whose state transitions create trajectory distributions with discrete support or sharp jumps that no Lipschitz demand function can reproduce.

**Load-bearing test.** The upper bound is load-bearing for practical guidance (tells regulators how much data they need). The lower bound is load-bearing for the *optimality* claim — without it, the system's sample complexity could be dismissed as an artifact of poor algorithm design. **The upper bound is practically load-bearing; the lower bound is intellectually load-bearing.**

**Grade audit.** Grade A is justified for the package (matching upper and lower bounds). The upper bound alone is Grade B (follows from C3′ by standard methods). The lower bound, if correct, is Grade A — the construction technique (embedding collusive automaton trajectories in the competitive null via demand function design) is novel. **Honest grade: A if both bounds are proved; B+ for the upper bound alone.**

**Difficulty calibration.** Upper bound: 1–2 person-months (conditional on C3′). Lower bound: 3–5 person-months, with significant risk of technical failure in the embedding construction.

**Risk assessment.** High risk concentrated in the lower bound. The embedding construction is the most technically fragile claim in Approach B. If it fails, Approach B loses its optimality story and the tight sample complexity degrades to "upper bound only" — still useful but much less impressive. The paper would need restructuring.

#### M8: Impossibility of Detection Without Bounded Recall (Self-graded: A)

**Achievability.** 90%+ probability. The construction of "stealth collusion" strategies (sustain supra-competitive payoffs but defer punishment beyond the observation horizon T) is conceptually clean and technically straightforward. The strategy that mimics competitive behavior for T rounds and only punishes after T+1 rounds produces a T-round marginal distribution identical to competitive play by construction. The formal argument via total variation distance is immediate.

**Load-bearing test.** Remove M8 and the bounded-recall restriction looks like a design limitation rather than a fundamental barrier. M8 transforms the restriction into a *theorem about the problem's structure.* **Load-bearing for the narrative** — not for the system's operation, but for the intellectual contribution's completeness. Without M8, a reviewer can ask "why not handle unbounded strategies?" and the answer is "we chose not to." With M8, the answer is "it's provably impossible."

**Grade audit.** Grade A is slightly generous. The impossibility result is clean and well-motivated, but the construction is not deep — it's essentially "design a strategy that hides its punishment mechanism beyond the observation horizon." Similar hiding arguments appear in adversarial learning theory and steganography. **Honest grade: A− for the result in the game-theoretic context; B+ if viewed as a straightforward adaptation of known impossibility constructions.**

**Difficulty calibration.** 1–2 person-months. The construction is conceptually straightforward; the work is in making the formal details precise (ensuring the stealth strategy's T-round marginal exactly matches a member of H₀, not just approximately).

**Risk assessment.** Low. This is the safest theorem in Approach B's program.

#### M5′: Minimax-Optimal Collusion Premium Estimation (Self-graded: B+)

**Achievability.** 70–80% probability. The three-term error decomposition is natural and the individual minimax rates for each sub-problem (demand estimation, NE approximation, finite-sample averaging) are known or derivable from standard theory. The novelty is the composition and the game-theoretic structure of the demand estimation problem.

**Load-bearing test.** Remove M5′ and the Collusion Premium estimate has no formal error guarantee — but the system still produces certificates via M1's testing. M5′ is load-bearing for Layer 2's quantitative collusion measurement but **not load-bearing for Layer 0's binary screening.** This makes it the lowest-priority theorem in Approach B's program.

**Grade audit.** Grade B+ is honest and accurate. This is a novel combination of known techniques with genuine game-theoretic structure in the composition. Not a deep theorem, but a useful and publishable result.

**Difficulty calibration.** 2–3 person-months. Mostly careful calculation and verification of rates.

**Risk assessment.** Low-medium. The main risk is that the composition introduces interaction terms between the three error sources that are not minimax-separable, complicating the clean three-term story.

#### Approach B Summary Verdict

The math program contains 5 load-bearing theorems totaling **12–20 person-months of expert effort** for the achievable components. Probability of completing all 5 in full generality: **5–10%.** Probability of completing 3 of 5 in full strength with restricted versions of the others: **35–45%.** The most likely outcome is: deterministic C3′ (full), M8 (full), M1′ (restricted to parametric + semi-parametric), M4′ upper bound only, M5′ (full). This is still a very strong paper — but it's not the paper Approach B describes.

**The fundamental tension:** Approach B's narrative requires *all five theorems* to support its "resolving the collusion detection barrier" framing. Deterministic C3′ + M8 + M4′ upper bound is already a strong EC paper. But the minimax lower bound and full non-asymptotic M1′ are the claims that elevate it from "strong paper" to "best paper." The probability that both the lower bound and full M1′ succeed is ~20–30%.

**Approach B is too ambitious by a factor of ~1.5.** Three theorems (C3′ deterministic, M8, M4′ upper bound) are the reliable core. The remaining two (M4′ lower bound, full M1′) are stretch goals that should be honestly labeled as such.

**Probability of EC acceptance: 50–60% (assuming 3/5 theorems proved). Probability of EC best paper: 15–25% (if 4+/5 theorems land).**

---

### Approach C: Math Audit

Approach C proposes four mathematical contributions: correlation supremum bound, peeling argument, automaton C3 proof, and error propagation algebra. The approach deliberately subordinates math to engineering — the artifact is the primary contribution, and the math is load-bearing for the artifact rather than independently interesting.

#### Contribution 1: Correlation Supremum Bound over Lipschitz Function Spaces (No explicit grade given)

**Achievability.** This is essentially the core of M1 from Approach A / M1′ from Approach B, stated more narrowly: bound sup_{D ∈ Lip(L)} Corr(π₁(D), π₂(D)). The achievability analysis is identical to M1. For the restricted case (parametric demand): 90%+. For full Lipschitz: 35–45%.

**Load-bearing test.** Same as M1 — **fully load-bearing** for Layer 0's formal guarantee.

**Grade audit.** Approach C calls this a "novel combination: the function-space optimization is standard; its application to a game-theoretic null is new." This is more honest than Approach A's Grade A. **Honest grade: B+ (novel application of known techniques).** Approach C's self-assessment is the most accurate of the three.

**Difficulty calibration.** 2–4 person-months (same problem as M1).

**Risk assessment.** Same as M1.

#### Contribution 2: Peeling Argument for Selection-Bias-Free Adaptive Sampling

**Achievability.** 70–80% probability. Peeling arguments are well-established in bandit theory (Auer et al., 2002; Lattimore & Szepesvári, 2020). The application to game-theoretic deviation payoff surfaces is new but structurally similar. The key question is whether the deviation payoff surface's Lipschitz structure is sufficient for the peeling decomposition to yield tight bounds — and it should be, since Lipschitz continuity is exactly the structural assumption peeling arguments exploit.

**Load-bearing test.** Remove the peeling argument and the deviation oracle (M2) cannot guarantee selection-bias-free adaptive sampling. The oracle still works but without formal (ε, α)-correctness guarantee — it becomes a heuristic bandit algorithm. **Load-bearing for Layer 1-2's formal guarantees, not for Layer 0.**

**Grade audit.** Approach C calls this "novel: the peeling decomposition applied to game-theoretic deviation payoff surfaces is new; peeling arguments exist in bandit theory but not in this domain." **Honest grade: C+ (novel application of a known technique to a new domain).** The peeling argument is a well-understood tool; applying it to deviation payoff surfaces is useful but not mathematically deep.

**Difficulty calibration.** 1–2 person-months. The technique is known; the work is adapting it and verifying the Lipschitz assumptions hold for game-theoretic payoff surfaces.

**Risk assessment.** Low. The main risk is that the deviation payoff surface is not globally Lipschitz (e.g., has kinks at strategy boundaries), requiring localized peeling that complicates the analysis without adding insight.

#### Contribution 3: Automaton C3 Proof (Deterministic Bounded-Recall)

**Achievability.** 85% probability (identical assessment to approaches A and B — this is the same theorem for the deterministic case).

**Load-bearing test.** Same as in Approaches A and B — **load-bearing for completeness** but not for soundness.

**Grade audit.** Approach C calls this "novel: the connection between automaton structure and statistical detectability of punishment is new." This is consistent with Grade A (new theorem). **Honest grade: A for the deterministic case.** Approach C does not claim the stochastic extension, which is the honest move.

**Difficulty calibration.** 2–3 person-months.

**Risk assessment.** Low-medium. Same as the other approaches' deterministic C3.

#### Contribution 4: Error Propagation Algebra for the Collusion Premium

**Achievability.** 85–90% probability. Interval arithmetic through a formula is well-understood. The zero-profit boundary handling (switching to absolute margin when NE profit → 0) is a technical detail, not a mathematical challenge.

**Load-bearing test.** Remove this and the Collusion Premium estimate has no formal error bounds. The system still produces a point estimate. **Partially load-bearing** — important for certificate rigor but not for the system's core screening function.

**Grade audit.** Approach C calls this a "novel combination: individual error bounds are standard; their composition through the game-theoretic quantity CP with the zero-profit boundary handling is new." **Honest grade: C (novel combination of entirely standard techniques).** The zero-profit boundary handling is a definitional fix, not a mathematical contribution.

**Difficulty calibration.** 1–2 person-months. Mostly careful bookkeeping.

**Risk assessment.** Negligible.

#### Approach C Summary Verdict

The math program contains 1 genuine theorem (correlation supremum / C3), 1 technique adaptation (peeling argument), and 2 standard calculations (correlation bound, error propagation). Total expert effort: **6–11 person-months.** All components are achievable. The math is honestly positioned as load-bearing infrastructure for the artifact rather than independent contributions.

**Could the engineering stand alone without the math?** Partially. Without the correlation supremum bound, Layer 0 has no formal Type-I error guarantee but still functions as a heuristic screen. Without C3, completeness is conjectural. Without the peeling argument, the deviation oracle is a heuristic. Without error propagation, the CP has no error bars. The system *operates* without the math but loses all formal guarantees — at which point it's a well-engineered heuristic tool, not a certification framework. **The math is necessary for the artifact to be what it claims to be** (a certifier), but unnecessary for it to be *useful* (a screening tool).

**Approach C's math is the most honest of the three but also the thinnest.** The engineering contribution is genuine and substantial. The question is whether EC rewards engineering-dominant papers — historically, the answer is "sometimes, if the engineering demonstrates something theoretically interesting." The artifact-as-proof-of-concept framing is viable but riskier than a theorem-first paper.

**Probability of EC acceptance: 45–55%. Probability of EC best paper: 5–10%.**

---

### Cross-Approach Math Comparison

#### Which math program is strongest?

**Approach B**, by a wide margin, proposes the deepest mathematics. C3′ (deterministic + stochastic), M4′ (matching bounds), and M8 (impossibility) are genuine theorems that advance repeated game theory. If completed, this is a landmark paper. But "if completed" is doing enormous load-bearing work in that sentence.

#### Which math program is most honest?

**Approach C.** Every claim is accurately graded, the techniques are correctly identified as known-technique applications, and the approach doesn't pretend the math is deeper than it is. The "novel combination" language is precise and accurate. Approach A is mostly honest (M7 at Grade D is refreshing) but over-grades M1 by half a grade. Approach B over-grades M8 by half a grade and the A+ on C3′ is premature — justified only if the stochastic extension reveals deep structure, which is uncertain.

#### Which math program is most achievable?

**Approach C > Approach A > Approach B.** Approach C's math is ~85% achievable in full. Approach A's is ~70% achievable for the restricted forms (H₀-narrow M1 + deterministic C3). Approach B's full program is ~5–10% achievable, with the realistic outcome being 3/5 theorems.

#### Specific comparative observations

1. **All three approaches share the same core math.** M1/M1′/Correlation-supremum are the same problem at different ambition levels. C3/C3′/Automaton-C3 are the same theorem at different generality levels. The approaches diverge in what they *add* beyond the shared core:
   - A adds M6 (verification soundness) and M7 (test ordering) — both low-novelty
   - B adds M4′ (tight bounds), M8 (impossibility), M5′ (optimal estimation) — high-novelty, high-risk
   - C adds peeling argument and error propagation — moderate-novelty, low-risk

2. **The shared core (restricted M1 + deterministic C3) is the reliable foundation.** All three approaches can build on this. The question is what additional math justifies the research investment.

3. **Approach B's lower bound (M4′) is the highest-risk, highest-reward component across all approaches.** If the embedding construction works (collusive automaton trajectory ↪ competitive null via demand function design), it's a genuinely new technique. If it fails, Approach B's optimality story collapses.

4. **Approach A's M6 (certificate soundness) is the only meta-theorem across all approaches.** It's low-novelty but uniquely important for the PCC framing. Neither B nor C adequately addresses whether their certificates are formally sound — they assume it. A's explicit treatment is more rigorous.

5. **Approach C's peeling argument is unnecessary for the core contribution.** It enables Layer 1-2 formal guarantees, which the depth check already identified as limited by the oracle-rewind assumption. For a Layer 0-focused paper, this math is not load-bearing.

#### What's missing from all three?

- **No approach adequately addresses the finite-sample gap for realistic T.** All three acknowledge that distribution-freeness may be asymptotic, but only B (M1′) directly attacks the finite-sample problem. A and C rely on empirical calibration and parametric fallbacks — defensible but theoretically unsatisfying.
- **No approach proves a positive result about power.** All three control Type-I error (soundness) but none proves that the test has non-trivial power against specific collusion alternatives at realistic sample sizes. A power analysis theorem — "against M-state grim-trigger with η ≥ 0.1, the test rejects with probability ≥ 0.9 at T = 10⁶" — would be more practically valuable than tight minimax bounds.
- **No approach addresses robustness to model misspecification.** What happens when the true demand system is not Lipschitz, or the agents have external state dependencies? All three assume the null family is correctly specified.

---

### Math Depth Ranking

#### Rank 1: Approach B

**Mathematical quality: 8.5/10 (if completed), 6.5/10 (realistic achievable portion)**

Approach B's math program is the most ambitious and, in its complete form, the most impressive. The C3′ + M4′ + M8 trifecta — a completeness theorem, a tight minimax rate, and an impossibility result — is the kind of comprehensive theoretical treatment that wins best papers. The problem is that the program's ambition exceeds its probability of completion. The realistic version (deterministic C3′ + M8 + upper bound only for M4′ + parametric M1′) is still very strong — arguably the best achievable paper across all three approaches.

**Key risk:** The math program has correlated failure modes. C3′ stochastic failure degrades M4′ (the sample complexity depends on punishment detection), and M1′ failure degrades M4′ (the upper bound requires the uniform testing result). A single theorem's failure cascades.

#### Rank 2: Approach A

**Mathematical quality: 5.5/10**

Approach A's math is honest, achievable, and load-bearing — but thin. Two genuine theorems (M1 restricted, C3 restricted) plus a verification exercise (M6) and an engineering contribution (M7) is a solid paper but not a best-paper contender on math alone. The approach compensates with artifact novelty and policy relevance. This is the **safest** approach: lowest variance, most achievable, but also lowest ceiling. The depth check's 5.0/10 for best-paper potential reflects this ceiling accurately.

**Key advantage:** The math will almost certainly get proved. The paper can be written with certainty about what it contains. No contingency planning needed.

#### Rank 3: Approach C

**Mathematical quality: 4.5/10**

Approach C's math is the thinnest and most derivative — it largely recapitulates the shared core (correlation bound, deterministic C3) with two additions (peeling argument, error propagation) that are technique applications rather than theorems. The approach is forthright about this: the engineering *is* the contribution. For a venue that rewards engineering demonstrations of theoretical ideas, this can work. For a venue that rewards theorems, it cannot. At EC, engineering-forward papers are accepted but rarely win best paper.

**Key advantage:** The math is almost entirely achievable. The key disadvantage is that there is not enough of it to justify the "certified" framing independently of the artifact.

---

#### Bottom Line

| Approach | Math Depth | Achievability | Honesty | Best Paper P(math alone) | Recommended Strategy |
|----------|-----------|---------------|---------|--------------------------|---------------------|
| **A** | Medium | High (70%+) | Good (minor over-grading on M1) | 10–15% | Lead with artifact novelty, math supports |
| **B** | Very High | Low (5–10% full, 35–45% partial) | Good (A+ premature, M8 slightly over-graded) | 15–25% (if 4+ theorems) | Prove core 3, label rest as stretch |
| **C** | Low-Medium | Very High (85%+) | Excellent | 5–10% | Lead with engineering, math as infrastructure |

**If forced to choose one approach for maximum expected math quality × achievability:** Approach A with B's deterministic C3′ argument and M8 impossibility theorem grafted in. This gives: restricted M1 (B+), deterministic C3′ (A), M8 (A−), M6 (C), M7 (D) — a more complete theoretical story than A alone, without B's over-ambitious program. Expected effort: ~8–12 person-months, ~70% probability of completing the full revised program.

---

## Synthesis of Critiques

### Points of Agreement

1. **A+B hybrid is the optimal strategy.** Both critics independently converge on the same recommendation: combine Approach A's achievable scope and artifact-first framing with Approach B's strongest mathematical results (deterministic C3′ and M8 impossibility). The Skeptic explicitly recommends "Approach A's scope with Approach B's mathematical ambition as a stretch goal." The Math Assessor recommends "Approach A with B's deterministic C3′ argument and M8 impossibility theorem grafted in."

2. **Approach C ranks last for EC.** Both critics agree that Approach C's engineering-forward framing is mismatched with EC as a theory venue. The Skeptic notes "engineering difficulty is not mathematical novelty, and EC rewards the latter." The Math Assessor gives C the lowest math depth score (4.5/10) and lowest best-paper probability (5–10%).

3. **Deterministic C3′ is the reliable mathematical core.** Both assign ~85% probability to the deterministic automaton completeness theorem and identify it as the highest-value achievable result across all approaches.

4. **H₀-broad may have trivial power.** Both flag the shared vulnerability: if the correlation bound over Lipschitz demand × independent learners is too loose, all three approaches produce tests with negligible statistical power against realistic collusion. Neither approach adequately confronts this possibility.

5. **Approach B's full 5-theorem program is over-ambitious.** The Skeptic gives ~15% probability to all five theorems proving; the Math Assessor gives 5–10%. Both agree the realistic outcome is 3/5 theorems, which is still strong but breaks the "Collusion Detection Barrier Theorem" narrative.

6. **Approach C's engineering insights should be adopted as implementation practices.** Both critics see value in phantom-type segment isolation, dual-path verification, and interval arithmetic — but as engineering practices within the chosen approach, not as paper-level contributions.

### Points of Disagreement

1. **Ranking of A vs. B.** The Skeptic ranks A first (corrected composite 6.0) and B second (6.25), favoring A's higher floor and deliverable certainty. The Math Assessor ranks B's math first (8.5/10 if completed, 6.5/10 realistic) but ultimately recommends the A+B hybrid — implicitly acknowledging that B alone is too risky. The disagreement is about risk tolerance: the Skeptic is more risk-averse (ship certainty matters most), while the Math Assessor weights mathematical ceiling higher.

2. **Severity of M8's novelty.** The Skeptic calls M8 "potentially trivially true" and suggests experienced game theorists will find the stealth-collusion construction obvious, downgrading it significantly. The Math Assessor gives M8 an honest grade of A− (B+ if viewed as adaptation of known constructions) and 90%+ achievability, treating it as a genuine if not deep contribution. The Skeptic is harsher on M8's impressiveness.

3. **Approach A's value score.** The Skeptic corrects A's value from 8 to 6, arguing Layer 0 is "a screening tool, not a certifier" and that regulatory adoption is years away. The Math Assessor doesn't directly re-score value but treats the artifact-novelty framing more favorably, noting the PCC paradigm for economics is genuinely new.

4. **The role of the proof checker.** The Skeptic views the de novo proof checker as a "soundness liability" requiring month-long expert audit, while the Math Assessor treats M6 (certificate verification soundness) as a legitimate if low-novelty contribution that is "uniquely important for the PCC framing."

### Strongest Conclusion

**Build Approach A's system with Approach B's deterministic C3′ and M8 grafted in, using Approach C's engineering practices.** This consensus delivers:

- **Restricted M1** (composite test with H₀-narrow/H₀-medium soundness) — ~70–90% achievable
- **Deterministic C3′** (completeness for all finite-state deterministic automata) — ~85% achievable
- **M8** (impossibility without bounded recall) — ~90% achievable
- **M6** (certificate verification soundness) — ~80–90% achievable
- **M7** (directed closed testing) — ~95% achievable
- **Working 60K LoC artifact** with Layer 0 certificates in <30 minutes

This hybrid has ~70% probability of full completion, yields a paper with both formulation novelty (M1) and genuine theorem depth (C3′ + M8), and ships a working artifact that demonstrates the theory. Expected effort: ~8–12 person-months. Estimated EC acceptance probability: 60–70%. Estimated EC best-paper probability: 12–20%.
