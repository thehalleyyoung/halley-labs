# CollusionProof: Three Competing Approaches

---

## Approach A: The Algorithmic Price-Fixing Fire Alarm
### Overview

CollusionProof ships as the world's first **statistically rigorous, machine-verifiable screening tool for algorithmic price coordination** — positioned not as a courtroom weapon but as the fire alarm that tells regulators *which buildings to inspect*. The approach leads with Layer 0 (passive observation on price trajectory data alone) as the complete, self-contained, independently publishable contribution, because Layer 0 is the only layer that works *today* against real proprietary algorithms without requiring any cooperation from the firms under investigation. Layers 1 and 2 are architected as strict capability extensions activated only in cooperative sandbox settings. The composite hypothesis testing framework (M1) — where the null family is parameterized by demand systems × learning algorithms — is a genuinely new problem instance in statistics, and the proof-carrying certificate format (M6) brings the PCC paradigm into economic regulation for the first time. The system targets EC as the primary venue, structured as a theory contribution (M1 + conditional M4 + M6) demonstrated through a working ~60K LoC artifact that produces verifiable certificates on 2-player Bertrand/Cournot markets with tabular RL agents in under 30 minutes on a laptop.

---

### Extreme Value Delivered

**Who needs this and why they can't sleep at night.**

Three groups face an *acute* crisis — not a theoretical concern, but active legal and regulatory emergencies with deadlines:

**1. EU DG-COMP enforcement teams (deadline: NOW).** The Digital Markets Act entered full enforcement March 2024. Article 5 and 6 obligations apply to designated gatekeepers *today*. DG-COMP must investigate algorithmic pricing practices but has **zero formal tools** — their economists run ad-hoc Stata scripts computing price correlations that any competent defense attorney dismantles in discovery. Assad et al. (2024, *JPE*) proved algorithmically-managed German gas stations show 9% margin inflation. DG-COMP knows algorithmic collusion is real. They cannot prove it to the standard required for enforcement action. Every month without formal tools means another market where algorithmic coordination calcifies into the status quo.

**2. DOJ Antitrust Division lawyers on the RealPage case (deadline: trial preparation).** The DOJ filed the first-ever federal antitrust action targeting algorithmic pricing coordination in 2024. Their theory of harm: RealPage's revenue management software enables tacit coordination among landlords who never communicate. The defense will argue these are *independent optimization algorithms responding to identical market signals* — the exact alternative hypothesis that Layer 0's composite test is designed to distinguish from coordinated behavior. DOJ economists currently lack any tool that produces evidence with formal statistical guarantees. Their expert witnesses present bespoke analyses that the defense can (and will) attack assumption-by-assumption. A CollusionProof Layer 0 certificate — machine-verifiable, with quantified Type-I error bounds, tested against a game-theoretic null — is not admissible evidence today, but it is exactly the kind of "generally accepted, testable, peer-reviewed methodology" that satisfies the Daubert standard for expert testimony. The methodology is what the DOJ needs, not the certificate format.

**3. Competition economists who build the analyses that regulators rely on.** These are the 200-500 PhD economists at firms like Compass Lexecon, Charles River Associates, and NERA who actually write the expert reports. They currently have: variance screens (known to produce false positives on correlated demand shocks), Granger causality tests (no game-theoretic foundation), and bespoke simulation comparisons (not reproducible, not verifiable). CollusionProof's Layer 0 gives them a principled replacement: a composite test with formal Type-I error control over a null that *actually means* "competitive behavior under any plausible demand system." The tiered null hierarchy (narrow/medium/broad) lets them calibrate the strength of their conclusions. This is not incremental — it replaces the methodological foundation of algorithmic collusion screening.

**Why they can't wait.** The regulatory window has a closing date. Once the RealPage case establishes precedent — whatever methodology the DOJ's experts use — that methodology becomes the *de facto* standard for algorithmic collusion evidence. If the precedent is set by ad-hoc correlation analysis, every future case will be litigated on the same fragile foundation. CollusionProof's contribution is defining the evidentiary standard *before* it calcifies. After the first major ruling, the window closes for a decade.

**Why existing tools fail completely:**
- Calvano et al.'s simulation approach: works only for tabular Q-learning, no certification, no error bounds, not reproducible as an audit tool
- Gambit: cannot accept black-box algorithms, no concept of collusion, no certificates
- PRISM-games: requires explicit state-space models — impossible for proprietary algorithms
- Variance/correlation screens: no game-theoretic null, rampant false positives from correlated demand shocks
- EGTA: extracts empirical game models but produces no collusion formalization or certified evidence

The gap is not "we need a slightly better tool." The gap is "no tool exists in the category."

---

### Genuine Difficulty as Software Artifact

The ~60K LoC MVP (CollusionProof-Lite) contains several subproblems that are individually hard and collectively require non-trivial architectural integration:

**Hard Subproblem 1: Compositional soundness across heterogeneous probability spaces.** The composite test (M1) combines multiple sub-tests — excess price correlation, punishment response asymmetry, supra-competitive persistence, convergence anomalies — each operating on different probability spaces with different conditioning events. Composing these via directed closed testing (M7) with Holm-Bonferroni FWER control requires proving that the composition preserves end-to-end α-control *uniformly* over the infinite-dimensional null family. This is not a standard application of multiple testing correction — the sub-tests share data and have complex dependency structures. Getting this wrong means the entire system's soundness guarantee is invalid.

**Hard Subproblem 2: Bounding maximum cross-firm correlation under competitive behavior.** The core technical challenge in M1 is proving distribution-freeness: the test statistic's null distribution must be bounded *uniformly* over all Lipschitz demand systems in each null tier. This requires bounding the maximum achievable cross-firm price correlation when firms use independent learning algorithms on a shared demand system — an optimization-over-function-spaces problem that must be solved for each null tier (linear demand, parametric demand, full Lipschitz family). The bound for H₀-narrow is tractable; the bound for H₀-broad requires novel covering-number arguments over Lipschitz function classes.

**Hard Subproblem 3: The numerical-to-formal bridge.** The simulation engine runs in f64 for performance (>100K rounds/sec target). The proof checker requires exact rational arithmetic for certificate verification. Every ordinal comparison that feeds into a proof derivation must be re-verified with rational arithmetic, with formal encoding of f64-to-rational conversion error bounds. This is not a rounding-error afterthought — it is a core architectural constraint that propagates through every subsystem boundary.

**Hard Subproblem 4: A de novo proof checker for a de novo domain.** The certificate DSL and proof checker (S6, ~5K LoC with ~2,500 LoC trusted kernel) must encode ~15 axiom schemas and ~25 inference rules that are *sound* for game-theoretic collusion properties. No existing proof checker handles this domain. The axiom system must be expressive enough to certify real collusion scenarios while small enough to audit by hand. Every axiom is a potential soundness hole. This is the hardest per-line-of-code subsystem in the project.

**Hard Subproblem 5: PPAD avoidance on the critical path.** Layer 0 must function entirely without Nash equilibrium computation (PPAD-complete for N > 2). The composite test's accept/reject decision must be independent of equilibrium — only the Collusion Premium benchmark (M5, Layers 1-2) requires NE computation, and only for structured games with analytical solutions (Bertrand/Cournot, n ≤ 4). Ensuring PPAD-avoidance is maintained under all code paths, including edge cases and fallbacks, is a persistent architectural constraint.

**Architectural challenge: The trust boundary.** The system has an explicit trust boundary: the proof-checker kernel (~2,500 LoC Rust, zero external dependencies) is the only code that must be correct for soundness. Everything else — simulation, statistical computation, certificate construction — can be buggy without compromising the guarantee that *verified certificates are valid*. Maintaining this trust boundary through development, ensuring no subsystem can inject unverified claims into the certificate chain, is a systems-level challenge that requires disciplined API design and continuous integration testing against the trust boundary.

---

### New Math Required

Only load-bearing mathematics — contributions that the artifact cannot function without:

**M1: Composite Hypothesis Test over Game-Algorithm Pairs (Grade A, Layer 0).** The competitive null H₀ is parameterized by (demand system, learning algorithm tuple) — an infinite-dimensional space. The tiered null hierarchy (H₀-narrow, H₀-medium, H₀-broad) provides tractability. The core mathematical contribution is proving that the test statistic's null distribution can be bounded uniformly over each tier, yielding distribution-free Type-I error control. For H₀-narrow (linear demand × Q-learning), this requires bounding cross-firm correlation analytically. For H₀-broad (Lipschitz demand × independent learners), this requires covering-number arguments over Lipschitz function spaces with Berry-Esseen finite-sample corrections. The result is: for any demand system and any independent learning algorithms in the null tier, the probability of false rejection is ≤ α. *This is the theorem that makes Layer 0 a scientific contribution rather than a heuristic.*

**M4/C3: Folk Theorem Converse for Bounded-Recall Strategies (Grade A*, conditional).** The completeness theorem: sustained supra-competitive pricing by strategies with recall M *necessarily* produces punishment responses detectable within O(M) rounds. We prove C3 unconditionally for: (a) grim-trigger strategies, (b) tit-for-tat with bounded memory, (c) any deterministic automaton with ≤ M states. These cover all practically deployed algorithmic pricing strategies. The general conjecture remains open — if false for exotic strategies, completeness degrades but soundness is unaffected. *This theorem connects the economic definition of collusion to statistical detectability.* Without it, we can only say "the test is sound"; with it, we can say "the test will find collusion if it exists (for strategies in the proven class)."

**M6: Certificate Verification Soundness (Grade C, all layers).** A formal proof that the certificate verification procedure is sound: if the proof checker accepts a certificate, the claimed statistical conclusions hold (with probability ≥ 1−α over the original randomness). This requires: (a) soundness of the axiom system (every axiom is a true statement about the game-theoretic domain), (b) correctness of the rational arithmetic re-verification, (c) proof that the f64-to-rational conversion preserves all ordinal relations used in proof derivations. *Without this, certificates are just data — with it, they are evidence.*

**M7: Directed Closed Testing with Collusion-Structured Ordering (Grade D, Layer 0).** Collusion-informed test ordering (supra-competitive pricing → punishment → correlation → convergence) that improves statistical power against typical collusion alternatives while maintaining FWER control via standard Holm-Bonferroni. Technically straightforward but essential for practical detection power. *This is what makes Layer 0 competitive with bespoke expert analysis rather than a conservative omnibus test.*

---

### Best-Paper Potential

CollusionProof targets an EC best paper through four reinforcing strengths:

**1. It introduces a genuinely new problem instance in statistics.** The composite hypothesis test where H₀ is parameterized by demand systems × learning algorithms has no precedent. While composite testing over infinite-dimensional nuisance parameters exists in semiparametric statistics (Andrews & Shi, 2013; Chernozhukov, Chetverikov & Kato, 2014), no existing framework tests against a null family whose structure is *game-theoretic*. This is not an application of known techniques to a new domain — the game-theoretic structure of the null family changes the mathematical problem (the nuisance parameters interact through market equilibrium, not independently). EC program committees reward exactly this kind of formulation novelty.

**2. It creates a category of artifact that does not exist.** Machine-checkable collusion certificates have zero precedent in any community. The prior art audit is exhaustive: formal methods has PCC for memory/type safety (Necula, 1997); economics has econometric screening without certificates; game theory has equilibrium computation without certification; antitrust law has expert testimony without machine verification. CollusionProof is the first system at the center of this triangle. Category creation — not incremental improvement — is the hallmark of best papers.

**3. The soundness/completeness separation is elegant and honest.** "Soundness is unconditional. Completeness is conditional on C3, with unconditional completeness for all deterministic bounded-recall automata." This is the kind of clean theoretical statement that committees remember. It echoes the structure of conditional results in cryptography (security under hardness assumptions) while being more practically satisfying — the unconditional completeness for deterministic automata covers every known deployed pricing algorithm.

**4. The timing is perfect.** EC 2025/2026 will have a program committee intensely interested in algorithmic pricing. The EU DMA, the RealPage case, and Assad et al.'s empirical evidence have made algorithmic collusion the most policy-relevant topic in market design. A paper that provides the *first formal framework* for certification — with a working artifact — will be irresistible to a committee that values relevance alongside theory.

**Why this beats a pure theory paper:** The artifact demonstrates that the theory *works*. Layer 0 certificates produced in under 30 minutes on a laptop, verified by an independent 2,500-LoC checker, distinguishing known-collusive from known-competitive scenarios with empirically validated Type-I error control. The theory says it's possible; the system proves it's practical.

---

### Hardest Technical Challenge

**The hardest challenge is proving distribution-free Type-I error control for M1's composite test over H₀-broad (full Lipschitz demand × independent learners).**

This is the load-bearing theorem for Layer 0. Without it, Layer 0's soundness guarantee applies only to restrictive parametric demand families (H₀-narrow), and the contribution reduces from "principled screening tool for arbitrary markets" to "a test that works when you already know the demand structure." With it, Layer 0 produces valid certificates *regardless of the true demand system*, which is the entire point for regulators who cannot observe demand.

**Why it's hard:** The cross-firm price correlation under the competitive null depends on the demand system — two independent learners on a highly correlated demand system can produce price trajectories that look coordinated. Bounding the *maximum* achievable cross-firm correlation over all L-Lipschitz demand functions requires:

1. Characterizing the supremum of a correlation functional over a Lipschitz function class — an infinite-dimensional optimization problem
2. Proving that the bound is tight enough for practical statistical power (a loose bound makes the test useless)
3. Handling the finite-sample gap between asymptotic distribution-freeness and the actual test distribution for realistic T (10⁵-10⁷ rounds)

**How to address it:**

*Tier the contribution honestly.* H₀-narrow (linear demand × Q-learning) has a closed-form correlation bound — prove this first, get a working Layer 0 with narrow soundness. H₀-medium (parametric demand × no-regret learners) requires optimization over a finite-dimensional parameter space — tractable with standard techniques. H₀-broad is the stretch goal — attempt the covering-number argument over Lipschitz function spaces, but if it requires T > 10⁸ for practical power, report it honestly and present H₀-medium as the practical ceiling.

*Provide finite-sample fallbacks.* For any T where asymptotic distribution-freeness hasn't kicked in, provide: (a) Berry-Esseen finite-sample correction terms bounding the gap, (b) a parametric sub-family of demand systems where exact (non-asymptotic) distribution-freeness holds for all T, and (c) permutation-based sub-tests that provide exact finite-sample validity at the cost of testing a narrower alternative.

*Validate empirically.* Run the test on 8+ known-competitive scenarios across 50+ seeds and verify that the empirical false-positive rate is ≤ α at each null tier. Empirical validation does not replace the theorem, but it calibrates whether finite-sample corrections are practically necessary and provides evidence that the theoretical bounds are not vacuous.

---

### Scores

**Value: 8/10.** The regulatory timing (EU DMA enforcement, DOJ RealPage litigation, Assad et al. empirical evidence) creates genuine urgency. Three distinct stakeholder groups (regulators, competition economists, CS/GT researchers) need this tool, and the closest existing alternative is ad-hoc Stata scripts. The gap is categorical, not incremental. Docked from 10 because: (a) Layer 0 is a screening tool, not courtroom evidence — regulators need it, but it doesn't directly win cases; (b) the full value proposition (Layers 1-2) depends on cooperative access models that don't yet exist in most jurisdictions; (c) adoption requires trust-building with a conservative legal community that has never used machine-checkable certificates.

**Difficulty: 7/10.** The ~60K LoC MVP contains genuine research difficulty: M1's distribution-freeness proof, the compositional soundness argument, the de novo proof checker, and the numerical-to-formal bridge are each individually hard. The Rust/Python split is architecturally motivated and adds integration complexity. The trust boundary maintenance is a persistent engineering challenge. Docked from 10 because: (a) Layer 0 alone avoids the hardest subproblems (deviation oracle, punishment detection, PPAD equilibrium); (b) the proof checker's axiom system is small (~15 schemas) compared to general-purpose proof assistants; (c) the market models (Bertrand/Cournot, 2-player) are well-understood.

**Potential: 8/10.** EC best-paper potential is real: genuinely new problem formulation (M1), category-creating artifact (PCC for economic properties), uncontested triple intersection, and perfect policy timing. The conditional completeness story (unconditional for practical strategy classes, C3-conditional generally) is elegant. Docked from 10 because: (a) M1 is formulation novelty — the semiparametric testing machinery exists, the application is new; (b) C3-conditional completeness is a mild liability for program committees unfamiliar with conditional results; (c) the artifact demonstration needs to be compelling enough that reviewers who don't run the code still find the paper convincing.

**Feasibility: 7/10.** Layer 0 completes in <30 minutes on a laptop (smoke mode). The ~60K LoC MVP is achievable with the core/extended split. The three-tier evaluation budget (smoke/standard/full) makes development iteration tractable. Rust + Python is the right technology choice. Docked from 10 because: (a) the M1 distribution-freeness proof for H₀-broad may require T beyond practical evaluation range; (b) 800 CPU-hours for standard evaluation is 4 days on 8 cores — tolerable but not comfortable for iterative development; (c) C3 proofs for restricted strategy classes are mathematically non-trivial and could take longer than estimated; (d) the proof checker's axiom soundness must be verified with extreme care — a single unsound axiom invalidates all certificates.

---

## Approach B: Resolving the Collusion Detection Barrier — Automaton-Theoretic Completeness with Tight Information-Theoretic Bounds
The depth check identified a composite score of 5.6/10, with the central weakness being that M1 offers formulation novelty rather than deep mathematical novelty, and M4's completeness depends on the unproved Conjecture C3. Approach B restructures CollusionProof so that **resolving C3 and establishing tight sample complexity bounds are the primary mathematical contributions**, with the certification system serving as the constructive witness of these theorems. Instead of building a system with a conjectural guarantee and hoping C3 is true, we make the proof of C3 (for finite-state strategies) and the proof that bounded recall is *necessary* for any detection scheme the twin pillars of the paper. This transforms the contribution from "a certification framework that works if C3 holds" into "a new theorem in repeated game theory (the collusion detection barrier theorem) whose constructive proof *is* the certification algorithm." Every mathematical component is load-bearing: remove any theorem and either the soundness guarantee, the completeness guarantee, the finite-sample validity, or the optimality claim collapses.

---

### Extreme Value Delivered

**Theoretical CS and game theory community (primary).** The Folk Theorem (Aumann & Shapley, 1976; Rubinstein, 1979; Fudenberg & Maskin, 1986) is one of the most celebrated results in game theory, but its *converse*—characterizing when collusion is detectable—has received no formal treatment for finite-state strategies. We deliver the first converse folk theorem for bounded-recall automata: if an M-state strategy profile sustains η-collusion, then punishment responses of magnitude ≥ (π_collusion − π_minimax)/M are detectable within M rounds. This is a standalone theorem in repeated game theory, independent of any software artifact, publishable at EC, Games and Economic Behavior, or Theoretical Economics.

**Competition economists and regulators (secondary).** The tight sample complexity result T* = Θ̃(M²σ²/(η²Δ_P²)) tells regulators *exactly* how much data they need to certify collusion at a given confidence level—and proves no method, however clever, can do better. The impossibility result for unbounded-recall strategies tells policymakers precisely where the formal detection boundary lies, informing regulatory sandbox design (e.g., "mandate that submitted algorithms use at most M states of internal memory").

**Statistical testing community (tertiary).** Computing the metric entropy of the function class {trajectory distributions indexed by Lipschitz demand × independent learning algorithms} and deriving non-asymptotic uniform concentration bounds for this class is a novel computation in empirical process theory. The function class has product structure (demand × algorithm) with Lipschitz constraints on one factor and algorithmic constraints on the other—a structure not previously analyzed.

---

### Genuine Difficulty as Software Artifact

#### Hard Subproblems

1. **Proving C3 for stochastic finite-state automata.** The deterministic case admits a clean Myhill-Nerode-style argument, but stochastic automata require Markov chain coupling techniques to bound the expected punishment response. The transition from "punishment state exists in the automaton graph" (deterministic) to "punishment is reachable with sufficient probability" (stochastic) introduces mixing-time dependencies that interact with the memory bound M.

2. **Information-theoretic lower bound construction.** Proving T* = Ω(M²σ²/(η²Δ_P²)) requires constructing a pair of distributions—one competitive, one η-collusive—that are indistinguishable from T < T* samples. The construction must respect the game-theoretic structure: the competitive distribution must arise from *some* Lipschitz demand × independent learner pair, and the collusive distribution must arise from a bounded-recall automaton with M states. Standard Le Cam / Fano constructions don't directly apply because the hypothesis classes are game-theoretically constrained.

3. **Metric entropy computation for the composite null.** The function class F_H₀ = {P_θ : θ ∈ (Lipschitz demand) × (independent learners)} is infinite-dimensional with product structure. Computing the covering number N(ε, F_H₀, d_TV) requires bounding entropy contributions from both the demand-function factor (standard in nonparametric statistics) and the learning-algorithm factor (novel—no prior entropy computation for the class of independent no-regret learners).

4. **End-to-end composition of probabilistic guarantees.** The certificate must compose: (a) finite-sample uniform testing from M1, (b) deviation bounds from M2, (c) punishment detection from M3, and (d) collusion premium estimation from M5—each with their own error terms—into a single end-to-end guarantee. The dependency structure between these components (M3 uses trajectory segments disjoint from M1; M5 depends on M2's output) requires a careful union-bound / closed-testing composition that preserves the overall α level.

5. **Rational arithmetic verification of irrational bounds.** The tight sample complexity involves σ², Δ_P², and η²—all estimated from data in floating point. The certificate must encode these as rational intervals and verify the ordinal comparisons in exact arithmetic. The gap between f64 simulation and rational verification is load-bearing: any rounding error that flips an inequality invalidates the certificate.

#### Architectural Challenges

The system architecture is unchanged from the crystallized problem (CollusionProof-Lite at ~60K LoC), but the proof checker kernel (S6) must now encode and verify substantially deeper proof objects: automaton decomposition lemmas for C3, coupling arguments for stochastic strategies, and entropy integral computations for the uniform testing bound. The axiom system expands from ~15 to ~22 schemas, and the trusted kernel may push to ~3,500 LoC—still auditable but at higher complexity.

---

### New Math Required

#### M1′: Non-Asymptotic Uniform Testing via Empirical Process Theory

**Statement.** For a trajectory of length T from N algorithms in a Lipschitz(L) demand market, the composite test statistic S_T satisfies:

sup_{θ ∈ H₀} P_θ(S_T > c_α) ≤ α + R(T, L, N)

where the remainder R(T, L, N) = O(L^d · N² · T^{−1/2} · log T) is *computable* and *non-asymptotic*. For the parametric sub-families (H₀-narrow, H₀-medium), R = 0 exactly.

**Technique.** Compute the metric entropy H(ε, F_H₀, d_TV) of the null family. The demand-function factor has entropy H_demand(ε) = O((L/ε)^d) by standard Lipschitz covering number bounds (Kolmogorov & Tikhomirov). The learning-algorithm factor has entropy H_algo(ε) bounded by exploiting the *independence* structure: independent learners produce trajectories where cross-firm correlations are bounded by a function of demand elasticity alone. Apply Dudley's entropy integral to the product class. Obtain a maximal inequality via Talagrand's concentration for suprema of empirical processes, yielding the uniform finite-sample bound.

**Load-bearing justification.** Without M1′, the composite test is valid only asymptotically—certificates produced from finite data have no formal Type-I error guarantee. M1′ makes every Layer 0 certificate valid at any T, with an explicit, computable error term. The depth check flagged this ("distribution-freeness may be asymptotic only") as a MODERATE flaw; M1′ resolves it completely.

**Novelty: Grade A.** Computing metric entropy for a function class indexed by (Lipschitz demand functions × independent learning algorithms) is a new computation. The product structure and the independence constraint on the algorithm factor distinguish this from standard nonparametric testing problems.

#### C3′: Folk Theorem Converse for Finite-State Strategies (The Collusion Detection Barrier Theorem)

**Statement (Deterministic).** Let σ = (σ₁, …, σ_N) be a profile of deterministic finite-state automata, each with at most M states, playing a repeated pricing game Γ with discount factor δ. If the average payoff profile π̄(σ) satisfies π̄_i(σ) ≥ π^NE_i + η for all i (η-collusion), then for every player i there exists a unilateral deviation d_i such that:

(a) d_i is detectable: the opponents' response within M rounds of the deviation produces a payoff drop of at least Δ_P ≥ η/(M · N), and

(b) d_i is identifiable: the post-deviation trajectory differs from the on-path trajectory in at least ⌈η·M/(π̄_max − π_minimax)⌉ of the M rounds following deviation.

**Proof sketch (Deterministic).** Model each σ_i as a Mealy machine (Q_i, Σ_i, δ_i, λ_i) where Q_i is the state set, Σ_i = P is the price set, δ_i is the transition function, and λ_i is the output function. The joint automaton σ operates on the product state space Q = Q₁ × ⋯ × Q_N with |Q| ≤ M^N. The on-path play traces a cycle C in Q of length ≤ M^N. If σ sustains η-collusion, every state on C must produce payoffs above π^NE + η. Consider player i deviating to the static Nash price p^NE_i. Since opponents are deterministic automata, their response is deterministic given the deviation. The joint state exits C and enters a transient path of length ≤ M (bounded by player i's state space). If no punishment occurs along this path (payoff drop < η/(M·N) in every round), then player i's average payoff over M rounds is at least π̄_i(σ) − η/N > π^NE_i + η(1 − 1/N)—strictly above Nash. But if *no player* can trigger punishment by deviating, the strategy profile is not self-enforcing (it offers profitable deviations with no deterrent), contradicting the assumption that σ is an equilibrium sustaining η-collusion. This contradiction establishes that at least one deviation triggers detectable punishment. A counting argument over all N players and M rounds yields the bound Δ_P ≥ η/(M·N).

**Statement (Stochastic).** For stochastic finite-state automata where transitions are Markov (each σ_i has transition kernel P_i(q'|q, a_{-i})), the conclusion holds in expectation: E[Δ_P] ≥ η/(M · N · τ_mix) where τ_mix is the mixing time of the joint Markov chain on Q. The probability of observing punishment ≥ η/(2M·N·τ_mix) within M + τ_mix rounds is at least 1 − exp(−Ω(η²/(M²·σ²_P))).

**Proof technique (Stochastic).** Use Markov chain coupling: couple the on-path chain (no deviation) with the post-deviation chain (player i deviates). Since the chains diverge at the deviation point and the state space is finite, they must re-couple within O(τ_mix) steps. The expected total payoff difference over [0, M + τ_mix] is bounded below by η · M / (M + τ_mix) minus the coupling cost. Concentration via Azuma-Hoeffding on the bounded-difference martingale yields the high-probability bound.

**Load-bearing justification.** C3′ converts M4 from conditional to unconditional for all finite-state strategies—the practical universe of pricing algorithms. Without C3′, the system can only guarantee completeness for grim-trigger and tit-for-tat; with C3′, completeness holds for any algorithm implementable as a finite-state machine (which includes all practical pricing algorithms with bounded memory). This is the single highest-impact mathematical contribution.

**Novelty: Grade A+.** A new theorem in repeated game theory. The deterministic case connects automaton theory (Myhill-Nerode, cycle structure) to strategic incentive analysis in a way not previously exploited. The stochastic extension uses Markov chain coupling in a game-theoretic context—bridging probability theory and strategic reasoning.

#### M4′: Tight Sample Complexity for Collusion Detection

**Statement (Upper bound).** Under C3′, the hybrid certifier detects η-collusion against M-state automaton strategies with probability ≥ 1 − β from

T* = O(M² · N² · σ² · τ²_mix · log(K/α) / (η² · β))

rounds, where K is the number of sub-tests in the composite battery.

**Statement (Lower bound).** For any test Ψ that is α-sound against H₀-broad (all Lipschitz(L) demand × independent learners):

inf_{M-state η-collusion} P(Ψ rejects) ≤ 1 − β whenever T ≤ c · M² · σ² / (η² · log(1/α))

for a universal constant c depending only on L and the price grid.

**Proof technique (Lower bound).** Construct a pair (P₀, P₁) where P₀ ∈ H₀-broad is a competitive distribution and P₁ is an η-collusive distribution generated by an M-state automaton, such that the total variation distance d_TV(P₀^T, P₁^T) ≤ 1/3 for T below the threshold. The construction exploits the richness of H₀-broad: any trajectory generated by an M-state collusive automaton with noise injection at level σ can be matched—in finite-dimensional marginals up to length T*—by a competitive process arising from a carefully chosen Lipschitz demand function with correlated shocks. The matching is possible because the demand function has enough degrees of freedom (Lipschitz functions on [0,1]^N → R^N_+) to mimic the M-state automaton's trajectory distribution over windows of length < M. Apply Le Cam's two-point method to the (P₀, P₁) pair.

**Load-bearing justification.** The upper bound tells regulators exactly how much data they need; the lower bound proves no algorithm can do better. Without the lower bound, a critic could argue "your system is slow because it's poorly designed." With it, slowness at small T is proved fundamental. This converts a potential weakness (high sample complexity) into a strength (optimality).

**Novelty: Grade A.** Minimax-optimal detection bounds for game-theoretic hypothesis testing. The lower bound construction—embedding a collusive automaton's trajectory distribution within the competitive null via demand function design—is a new technique connecting nonparametric statistics to game theory.

#### M8: Impossibility of Detection Without Bounded Recall (The No-Free-Lunch Theorem)

**Statement.** For any finite sample size T and any test Ψ that is α-sound against H₀-broad:

sup_{σ ∈ Σ_∞(η)} P_σ(Ψ accepts) ≥ 1 − α − 2exp(−T/2)

where Σ_∞(η) is the class of *all* (not necessarily finite-state) strategies sustaining η-collusion. In words: without the bounded-recall assumption, any sound test has trivial power against some collusive strategy.

**Proof sketch.** Construct a "stealth collusion" strategy that: (i) sustains average payoffs at π^NE + η, (ii) stores the entire history of play in an infinite-state register, and (iii) only triggers punishment after observing a deviation sequence of length ≥ T+1 (which never occurs in T rounds of observation). The strategy's T-round marginal distribution is identical to a competitive process (independent play with demand-driven correlation), so no T-round test can distinguish it. Formally, for any T, define σ^T_stealth using a strategy that mimics competitive best-response for the first T rounds after any deviation, then punishes starting at round T+1. The resulting T-round trajectory distribution is exactly equal to a member of H₀ (by construction), so d_TV(P_{σ^T_stealth}^T, H₀) = 0 and Ψ must accept with probability ≥ 1 − α.

**Load-bearing justification.** M8 proves that the bounded-recall restriction in the problem formulation is *necessary*, not merely convenient. Without it, CollusionProof's design choice to restrict to bounded-recall strategies looks like a limitation. With M8, it's a fundamental barrier: *any* detection system must make this restriction (or an equivalent one). This transforms a weakness into a deep structural insight about the problem.

**Novelty: Grade A.** An impossibility theorem for economic certification. The construction of stealth-collusion strategies as a formal mathematical object is new. The result connects to the broader impossibility literature in learning theory (no-free-lunch theorems, adversarial examples) but in a game-theoretic context.

#### M5′: Minimax-Optimal Collusion Premium Estimation

**Statement.** Under C3′ and Layer 2 access, the Collusion Premium estimator CP̂ satisfies:

E[(CP̂ − CP_true)²] ≤ C · (σ²_demand / T_demand + σ²_NE / |P|^N + σ²_play / T_play)

where the three terms correspond to demand estimation error, Nash equilibrium approximation error, and finite-sample averaging error, respectively. Each term is minimax-optimal over the respective estimation sub-problem.

**Load-bearing justification.** The error decomposition into three independent sources with minimax-optimal rates for each tells regulators exactly which component limits their certificate precision—and proves each component is as good as possible. For zero-profit equilibria, the estimator smoothly transitions to the absolute-margin metric δ_p with analogous optimal rates.

**Novelty: Grade B+.** The individual estimation sub-problems have known optimal rates; the composition and the game-theoretic structure of the demand estimation problem (where the demand function is a nuisance parameter in a game) add genuine novelty.

---

### Best-Paper Potential

This approach targets a **best paper at EC** by delivering three results that, individually, would be strong workshop papers, and collectively constitute a major contribution:

1. **The Collusion Detection Barrier Theorem (C3′ + M8).** A new theorem in repeated game theory with a clean, quotable statement: "Collusion by M-state automata is detectable in Θ̃(M²/η²) rounds; collusion by unrestricted strategies is undetectable at any finite horizon." This is the kind of dichotomy result that wins awards—compare Roughgarden's smoothness framework (EC 2009 best paper), which gave a clean characterization separating games where equilibria are efficient from those where they aren't. Our result separates detectable collusion from undetectable collusion, with the boundary at bounded recall.

2. **Minimax-optimal detection (M4′).** Matching upper and lower bounds for a natural statistical problem. Papers establishing tight minimax rates are perennially celebrated (Donoho & Johnstone, Tsybakov, etc.). Our lower bound construction—embedding collusive trajectories in the competitive null via demand function design—is a new technique that the community can extend.

3. **Constructive proof via certification system.** The proof of C3′ is constructive: the detection algorithm *is* the proof that detection is possible. The certification system produces machine-verifiable witnesses of the theorem's conclusion. This "theorem-as-software" structure is distinctive and appeals to EC's dual identity as both a theory and a systems venue.

**Why the math quality wins the award.** The depth check criticized the original M1 as "formulation novelty, not mathematical novelty" (Grade A generous, Grade B+ more accurate). Approach B addresses this by making every mathematical contribution prove something *hard*: C3′ is a new theorem in repeated game theory, M4′ lower bound requires a non-trivial construction in statistics, M8 is an impossibility result, and M1′ resolves the finite-sample gap completely. No component is ornamental—removing any one collapses either soundness (M1′), completeness (C3′), optimality (M4′), or the necessity of the model (M8).

---

### Hardest Technical Challenge

**Proving C3′ for stochastic finite-state automata.** The deterministic case admits a clean graph-theoretic argument (cycle detection in the product automaton, contradiction via profitable deviation). The stochastic case is fundamentally harder because:

1. **No deterministic cycle structure.** Stochastic automata visit states according to a Markov chain, so the "on-path play" is a distribution over paths, not a single path. The graph-theoretic argument must be replaced by a probabilistic one.

2. **Mixing time interaction.** The punishment bound involves the mixing time τ_mix of the joint Markov chain. If τ_mix is large (near M^N), the punishment signal is washed out by the stochastic transitions and the detection horizon grows beyond M. The bound Δ_P ≥ η/(M·N·τ_mix) may be vacuous for poorly mixing chains.

3. **Coupling argument subtlety.** The Markov chain coupling between the on-path and post-deviation chains must account for the fact that the deviating player's transition kernel changes, which may *accelerate or decelerate* convergence to the stationary distribution.

**How to address it.** We adopt a three-stage strategy:

*Stage 1 (Certain):* Prove C3′ for deterministic automata (all M-state deterministic Mealy machines). This is achievable via the graph-theoretic argument outlined above and covers Q-learning with discretized Q-tables, grim-trigger, tit-for-tat, and any lookup-table strategy. Estimated difficulty: hard but tractable (2–3 weeks of focused work). This alone yields unconditional completeness for the most practically relevant strategy class.

*Stage 2 (Probable):* Extend to stochastic automata with τ_mix ≤ poly(M). Most practical pricing algorithms (ε-greedy Q-learning, Boltzmann exploration) have polynomial mixing times because the exploration parameter ensures rapid mixing. The coupling argument works cleanly when τ_mix = O(M²), giving Δ_P ≥ η/(M³·N)—weaker but still non-vacuous. Estimated difficulty: significant (3–4 weeks). Requires careful Markov chain analysis.

*Stage 3 (Aspirational):* Characterize the exact frontier: for which stochastic automata is C3′ true, and what is the optimal dependence on τ_mix? Conjecture: C3′ holds whenever τ_mix < exp(M)/N, which covers all "reasonable" stochastic strategies but excludes pathological constructions. If this boundary can be characterized, it becomes a contribution to the Markov chain theory literature as well. Estimated difficulty: open research problem (may not resolve within project timeline).

**Fallback plan.** If Stage 2 fails, the paper presents C3′ for deterministic automata (Stage 1) as the main theorem and states the stochastic extension as a conjecture with the Stage 2 partial result as supporting evidence. This is still a major advance over the original proposal (which left C3 entirely as a conjecture) and gives the community a concrete open problem with a clear attack strategy.

---

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Value** | **7/10** | Resolving C3 and proving impossibility without bounded recall delivers a permanent contribution to repeated game theory. The tight sample complexity bounds give regulators a rigorous "data budget" and prove optimality. Value is somewhat concentrated in the theory community rather than immediate regulatory deployment, but the theorems inform regulatory sandbox design (memory bounds on submitted algorithms). |
| **Difficulty** | **9/10** | C3′ for stochastic automata is genuinely hard and uncertain. The minimax lower bound construction (embedding collusive trajectories in the competitive null) requires a novel technique. The metric entropy computation for the composite null class is a new problem in empirical process theory. The end-to-end composition of all guarantees into a sound certificate adds integration difficulty. Only the deterministic C3′ and M1′ are "routine-hard"; the rest is "research-hard." |
| **Potential** | **9/10** | A paper that resolves an open conjecture in repeated game theory, proves a minimax lower bound for a natural statistical problem, establishes an impossibility result for unrestricted strategies, and demonstrates the theory via a working certification system is a best-paper-caliber contribution at EC. The dichotomy result (bounded recall → detectable; unbounded → undetectable) is clean and quotable. Multiple theorems generate independent follow-up directions. |
| **Feasibility** | **5/10** | Stage 1 (deterministic C3′) is tractable. Stage 2 (stochastic with polynomial mixing) is probable but uncertain. The minimax lower bound requires a specific construction that may have subtle technical issues. M1′'s metric entropy computation is novel and may reveal that the non-asymptotic bound has worse constants than hoped. The total mathematical program—five new theorems, each load-bearing—is ambitious for a single project. Realistic outcome: 3 of 5 theorems proved in full strength, remaining 2 proved for restricted cases with conjectures stated. The 60K LoC implementation must track the evolving theorem statements, creating coupling between math progress and engineering work. |

---

## Approach C: The Compositional Certification Kernel — Engineering a Trusted Pipeline Across Four Hostile Abstraction Boundaries
### Overview

CollusionProof's genuine software artifact difficulty is not that any one subsystem is impossibly hard—it is that **seven subsystems spanning four programming paradigms, three numerical precision regimes, and two trust domains must compose into a single pipeline where a bug in any stage silently corrupts every downstream certificate**. This approach reframes the project around the engineering challenge of building a *compositionally sound certification pipeline*: a system where statistical computations in f64, game simulations in Rust, hypothesis tests orchestrated in Python, and proof terms verified in rational arithmetic must all agree on a single chain of evidence—with the property that an independently-built 2,500-LoC checker can validate the entire chain without trusting the pipeline that produced it. The hard part is not building a game simulator, nor writing a hypothesis test, nor designing a proof language. The hard part is building all three so they interoperate with formally tracked error propagation across every boundary, under performance constraints that rule out the easy architectural choices (everything in Python, everything in exact arithmetic, everything in one process).

### Extreme Value Delivered

**Who needs it**: Antitrust regulators (EC DG-COMP, FTC, DOJ) facing the first generation of algorithmic pricing investigations need formal evidence that withstands adversarial legal scrutiny. Competition economists need certified measurement replacing ad-hoc index comparisons. Algorithm auditors in regulatory sandboxes need black-box testing methodology with reproducible, machine-checkable outputs.

**Why it matters now**: The EU DMA entered full enforcement March 2024. The DOJ RealPage case (filed 2024) is the first federal antitrust action targeting algorithmic pricing. Assad et al. (2024, *JPE*) documents empirical margin inflation from algorithmic duopolies. Yet no tool produces a self-contained, independently verifiable evidence bundle demonstrating that observed pricing satisfies game-theoretic collusion conditions. The triple intersection of formal verification × game theory × competition law has zero active research groups producing tools. CollusionProof defines the first machine-checkable evidentiary standard for algorithmic collusion—before ad-hoc approaches calcify into legal precedent.

**Why an expert can't build this in a weekend**: An expert can build a game simulator in a weekend (Bertrand in ~500 lines of Rust). An expert can implement a battery of statistical tests in a weekend (correlation screens in ~300 lines of Python). An expert can write a toy proof checker in a weekend (~800 lines for propositional logic). But no expert can *compose* these into a system where the simulator's f64 round-step outputs feed statistical tests whose p-values feed proof derivations verified in exact rational arithmetic—all with formally tracked error propagation across every boundary—in anything less than months of careful architectural work. The composition is the artifact.

### Genuine Difficulty as Software Artifact

#### Challenge 1: The Numerical-to-Formal Bridge (The Hardest Per-Line-of-Code Problem)

The simulation engine runs in f64 for performance (>100K rounds/sec requirement). The proof checker verifies in exact rational arithmetic (soundness requirement). Every ordinal comparison that appears in a proof derivation—"price p₁ exceeded threshold τ at round t"—must be re-verified in rational arithmetic with formal encoding of the f64→ℚ conversion error bound.

**Why this is genuinely hard**: IEEE 754 f64 arithmetic is not monotone under conversion. Two f64 values that satisfy `a > b` in floating-point may fail `rat(a) > rat(b)` after rational conversion, depending on rounding mode and ULP distance. The pipeline must identify every comparison in the proof's dependency DAG, compute conservative error intervals, and verify that the rational-arithmetic re-check confirms the same ordering. This requires:

- A comparison-tracking layer in the simulation engine that logs every comparison used downstream (not all comparisons—only those that influence the certificate). Estimated ~2,000 LoC of annotation infrastructure in Rust.
- Interval arithmetic wrappers for all statistical computations that contribute to proof terms. Every p-value, every bootstrap CI bound, every deviation estimate must carry an error interval. ~1,500 LoC.
- A rational-arithmetic verification pass in the proof checker that re-derives every numerical claim from the raw data using exact arithmetic, failing the certificate if any ordering flips. ~1,000 LoC in the trusted kernel.

The total is ~4,500 LoC of load-bearing bridge code—small in absolute terms, but every line must be correct or the entire certification pipeline is unsound. This is the highest difficulty-per-LoC subsystem in the project.

#### Challenge 2: Compositional α-Control Across Heterogeneous Probability Spaces

The composite hypothesis test (M1) runs K sub-tests across different probability spaces. The deviation oracle (M2) performs adaptive sampling with coarse-to-fine refinement. The punishment detector (M3) injects perturbations on separate trajectory segments. The directed closed testing procedure (M7) composes all of these with family-wise error rate (FWER) control. End-to-end, the system must guarantee Type-I error ≤ α.

**Why this is genuinely hard as software**: Each sub-test consumes a fraction of the α budget. The M7 composition uses Holm-Bonferroni, which requires the sub-tests to be valid p-values under the null—but each sub-test operates on different data segments, different probability spaces, and different conditioning events. A single data-reuse bug (e.g., using the same trajectory segment for both M1's correlation test and M3's perturbation injection) silently inflates the false positive rate without any runtime error. The software must enforce:

- **Segment isolation**: A trajectory partitioning system that provably prevents data reuse across sub-tests. Not a soft convention—a type-level enforcement in Rust where trajectory segments carry phantom-type tags that make cross-segment access a compile-time error. ~3,000 LoC.
- **α-budget accounting**: A resource-linear type system tracking how much α budget each sub-test has consumed, preventing over-allocation at compile time (or, more practically, a runtime budget tracker that aborts certificate construction if α is overdrawn). ~1,000 LoC.
- **Conditional validity bookkeeping**: Each sub-test's p-value is valid only under specific conditioning assumptions. The proof term must encode which conditioning events each p-value depends on. ~1,500 LoC in the certificate DSL.

Getting this wrong doesn't crash the system—it produces certificates that appear valid but have inflated false positive rates. This class of bug is undetectable by standard testing because it manifests only statistically, over thousands of runs.

#### Challenge 3: The Sandboxed Black-Box Execution Environment

The system must execute arbitrary pricing algorithms (Q-learning, DQN, PPO, grim trigger, custom strategies) as black-box oracles inside a controlled sandbox that provides: (a) deterministic replay from checkpoints, (b) memory isolation so algorithms cannot observe the auditor's strategy, (c) bounded resource consumption (CPU time, memory), and (d) a clean FFI boundary between Rust's simulation engine and Python's ML ecosystem.

**Why this is genuinely hard**: The PyO3 Rust↔Python FFI layer must manage GIL acquisition/release across ~10⁸ round-step calls without serialization overhead destroying the >100K rounds/sec performance target. Naively acquiring the GIL per round-step costs ~500ns overhead, reducing throughput to ~2M/sec before any work—but with 3-4 Python algorithm calls per round-step, this drops to ~500K rounds/sec with zero algorithm computation. Solutions:

- **Batched oracle calls**: Buffer N round-steps in Rust, acquire GIL once, batch-evaluate N steps in Python, release GIL. Requires careful buffer management with backpressure. ~2,000 LoC.
- **Checkpoint/restore for deterministic replay**: Algorithm state must be serializable to bytes for Layer 1-2 oracle operations. PyTorch model state, optimizer state, and RNG state must all be captured. Custom serialization for each algorithm class. ~3,000 LoC.
- **Resource sandboxing**: Process-level isolation (forking + resource limits) for untrusted algorithm implementations. Must handle OOM, infinite loops, and non-determinism from uncontrolled RNG seeds. ~2,000 LoC.

#### Challenge 4: The Proof Language Design Problem

The certificate DSL must be expressive enough to encode real collusion arguments (involving statistical significance, deviation bounds, punishment evidence, and economic quantities) while keeping the trusted checker kernel ≤ 2,500 LoC with zero external dependencies.

**Why this is genuinely hard**: Existing proof assistants (Lean, Coq, Isabelle) have >100K LoC kernels with decades of development. CollusionProof needs a *domain-specific* proof language that is:

- Expressive enough to encode: "Sub-test k rejected H₀-narrow at significance α_k, the deviation oracle certified a lower bound on deviation payoff, the collusion premium exceeds threshold τ with confidence 1-β, and these facts compose into a valid certificate at overall significance α."
- Restrictive enough that soundness is auditable: ~15 axiom schemas, ~25 inference rules, no polymorphism, no dependent types, no tactic language.
- Efficient enough for O(T·n) verification time—the checker must stream through proof terms without loading the entire proof into memory.

The design tension is between expressiveness (can the language state real collusion arguments?) and auditability (can a skeptical reviewer verify the axiom system is sound in an afternoon?). Getting the axiom system wrong—including an axiom that is "obviously" true but subtly unsound—undermines the entire project. This requires iterative design: implement certificate generation, discover which proof terms are needed, refine the axiom system, verify that no axiom combination enables vacuous proofs.

#### Challenge 5: O(N·D·S·T) Counterfactual Analysis Under Laptop Constraints

For N=3 players, D=20 deviations, S=100 seeds, T=100K rounds: ~600M round-steps. At the >100K rounds/sec performance target, this is ~6,000 seconds ≈ 100 minutes for a single scenario. The system must complete the full evaluation suite (15 standard-mode scenarios, each with counterfactual analysis) in ~4 days on 8 cores.

**Why this is genuinely hard**: The counterfactual simulation is *not* embarrassingly parallel at the round-step level—each round depends on the previous round's prices and algorithm states. Parallelism must exploit the (player, deviation, seed) decomposition: N×D×S independent simulation trajectories, each sequential in T. With N=3, D=20, S=100 this yields 6,000 independent jobs. On 8 cores, each job must complete in ~(4 days × 8 cores × 3600 sec/hr × 24 hr/day) / (15 scenarios × 6,000 jobs) ≈ 30 seconds per job. At T=100K rounds per job, this requires >3,300 rounds/sec per job—achievable in Rust but impossible in Python.

The scheduling challenge: 6,000 jobs with heterogeneous runtimes (some deviations trigger early termination, others run full T) across 8 cores with checkpoint/restart for crash recovery, memory-bounded queuing (each job holds ~50MB of algorithm state), and progress reporting. This is a custom work-stealing scheduler. ~2,500 LoC.

#### Challenge 6: Merkle-Integrity Evidence Bundles

Certificates must be self-contained evidence bundles that any party can independently verify. The bundle includes: raw price trajectory data, algorithm configuration, simulation parameters, all intermediate statistical results, proof terms, and the checker binary. A Merkle tree binds the entire bundle so that tampering with any component invalidates the root hash.

**Why this is genuinely hard as integration**: The Merkle tree must hash heterogeneous data types (f64 arrays, protobuf-encoded proof terms, configuration TOML, binary executables) with a canonical serialization that is deterministic across platforms. The checker binary must be reproducibly built (same source → same binary on different machines) or the hash verification fails. Reproducible builds in Rust require pinned toolchain versions, deterministic linking, and stripped binaries. ~1,500 LoC for the Merkle infrastructure + substantial CI/build engineering.

### New Math Required

Only math that is **load-bearing**—directly enables the artifact and cannot be replaced by existing results:

1. **Correlation supremum bound over Lipschitz function spaces** (enables M1). To prove that the composite test has Type-I error ≤ α *uniformly* over the infinite-dimensional competitive null H₀, we must bound sup_{D ∈ Lip(L)} Corr(π₁(D), π₂(D)) where π_i are profit trajectories under demand system D. This is an optimization over a function space. The bound uses covering number arguments for Lipschitz classes (entropy integral bounds from empirical process theory) combined with a symmetrization argument specific to the game structure. **Novel combination**: the function-space optimization is standard; its application to a game-theoretic null is new.

2. **Peeling argument for selection-bias-free adaptive sampling** (enables M2). The deviation oracle uses coarse-to-fine Lipschitz-aware refinement—querying more densely in price regions where the deviation payoff surface has high curvature. This adaptive sampling must not introduce selection bias. The peeling argument decomposes the adaptive procedure into resolution levels and proves that the bias at each level is bounded, yielding an overall (ε, α)-correctness guarantee with query complexity O(n · polylog|P_δ| · log(n/α)/ε²). **Novel**: the peeling decomposition applied to game-theoretic deviation payoff surfaces is new; peeling arguments exist in bandit theory but not in this domain.

3. **Automaton C3 proof** (enables unconditional completeness for practical cases). For deterministic bounded-recall automata with ≤ M states sustaining supra-competitive pricing, we prove that punishment responses are detectable within M rounds. The proof constructs the deviation that triggers the worst-case (fastest) punishment by exhaustive search over the automaton's state transition graph. **Novel**: the connection between automaton structure and statistical detectability of punishment is new.

4. **Error propagation algebra for the Collusion Premium** (enables M5). The certified CP composes: (a) demand estimation error (from finite-sample regression), (b) Nash equilibrium approximation tolerance (from analytical solver or numerical solver with convergence certificate), and (c) finite-sample averaging error (from bootstrap). The error propagation tracks interval arithmetic through the CP = (π_obs - π_NE) / π_NE formula, with the special-case switch to absolute margin δ_p when π_NE → 0 (homogeneous Bertrand). **Novel combination**: the individual error bounds are standard; their composition through the game-theoretic quantity CP with the zero-profit boundary handling is new.

### Best-Paper Potential

CollusionProof earns a best-paper award at EC not because of any single mathematical theorem, but because the *engineering quality of the artifact demonstrates the theory's feasibility in a way that no theorem alone can*. The key insight:

**The artifact is the proof of concept for a new category of formal object.** Machine-checkable collusion certificates do not exist in any community. A theorem proving they *could* exist (M4) is interesting. A working system that *produces* them—with a 2,500 LoC checker that anyone can audit, running on a laptop in 30 minutes for Layer 0—is transformative. The gap between "theoretically possible" and "here is a certificate, verify it yourself" is the gap this artifact bridges, and it can only be bridged by building the system.

**Engineering honesty wins reviewers.** The tiered oracle model (Layer 0/1/2), the explicit C3 conditionality, the honest LoC accounting (~60K core, not 168K), the three-tier evaluation budget—these signal a project that understands its own limitations. EC reviewers are sophisticated enough to reward this.

**The compositional soundness challenge is publishable in its own right.** The problem of maintaining end-to-end Type-I error control across heterogeneous sub-tests operating on different probability spaces, with formally tracked error propagation from f64 simulation through to rational-arithmetic verification, is a novel systems-meets-statistics challenge. No existing framework addresses it.

### Hardest Technical Challenge

**The numerical-to-formal bridge with compositional α-control.**

This is the intersection of Challenges 1 and 2: ensuring that the proof checker's rational-arithmetic verification of statistical claims agrees with the f64 computations that produced those claims, *and* that the composition of K sub-tests across different probability spaces maintains FWER ≤ α—with the additional constraint that a bug in either direction is *statistically invisible* in normal testing.

**Why it's hardest**: A bug in the game simulator crashes or produces obviously wrong prices. A bug in the CLI produces a bad error message. A bug in the numerical-to-formal bridge produces certificates that look correct, pass the checker, but have inflated false positive rates due to a subtle f64→ℚ rounding issue in one comparison out of thousands. This class of bug is:
- **Silent**: No runtime error, no crash, no obviously wrong output.
- **Rare**: Manifests only when a comparison is within ULP distance of the threshold—perhaps 1 in 10,000 certificates.
- **Consequential**: A false collusion certificate presented in antitrust proceedings could cause billions in damages.

**How to address it**:

1. **Conservative interval arithmetic everywhere**. Every f64 computation that feeds a proof term carries an explicit [lo, hi] interval. The proof checker verifies that the *worst-case* endpoint of every interval satisfies the required ordering. Overly conservative intervals reduce statistical power but never produce false positives. ~1,500 LoC.

2. **Dual-path verification**. The certificate generation pipeline runs every proof-relevant computation twice: once in f64 (fast, for the actual certificate) and once in exact rational arithmetic (slow, for verification). The two paths must agree on every ordinal claim. Disagreements abort certificate construction. ~2,000 LoC.

3. **Statistical meta-testing**. The evaluation suite includes a dedicated Type-I error validation: run 1,000+ seeds on known-competitive scenarios and verify that the empirical false positive rate does not exceed α + ε for a chosen tolerance ε. This catches systematic α-inflation bugs that manifest only statistically. Requires ~200 CPU-hours of dedicated compute in --standard mode.

4. **Segment isolation via type-level enforcement**. Trajectory data segments are tagged with phantom types in Rust. A sub-test that receives `Segment<Tag3>` cannot access data from `Segment<Tag1>`. This prevents the most common source of α-inflation (data reuse) at compile time rather than relying on programmer discipline.

### Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Value** | **7/10** | Addresses a real, urgent regulatory gap (EU DMA, DOJ RealPage) at an uncontested triple intersection with no competing tools. Score limited by the oracle-rewind assumption restricting Layers 1-2 to sandbox settings and the reality that regulators have not yet demanded machine-checkable certificates—but defining the standard before demand crystallizes is how paradigm-setting tools are built. Layer 0 alone surpasses every existing statistical screening method. |
| **Difficulty** | **8/10** | The compositional challenge—making seven subsystems across two languages, three numerical precision regimes, and two trust domains produce certificates that a 2,500 LoC checker can independently verify—is genuinely hard and has no off-the-shelf solution. The numerical-to-formal bridge, α-budget tracking, PyO3 FFI performance, and proof language design are each independently challenging; their *composition* is the real difficulty. A strong Rust+Python systems engineer with game theory expertise would need 4-6 months for the core ~60K LoC. The 600M round-step counterfactual analysis imposes real performance constraints that rule out naive implementations. |
| **Potential** | **7/10** | Strong EC best-paper candidate due to the new artifact category (machine-checkable collusion certificates), the uncontested intersection, and the immediate policy relevance. The working system demonstrating the theory is worth more than the theory alone. Limited by M1's formulation-level (not technique-level) novelty and the C3 conditionality on completeness, though unconditional soundness and restricted-class completeness proofs mitigate this. |
| **Feasibility** | **6/10** | The ~60K LoC core is achievable for an expert team in 4-6 months. Rust + Python is architecturally justified and the team likely has both skills. The three-tier evaluation budget makes iteration feasible (30-min smoke mode). Risks: (1) the proof language design may require 2-3 iterations to get the axiom system right, each requiring rework of certificate generation; (2) the C3 proof for general automata may be harder than expected, though restricted-class proofs are achievable; (3) the 800 CPU-hour standard evaluation imposes a 4-day turnaround for milestone validation, which is tight but manageable. The --full evaluation at 3,800 CPU-hours is a one-shot event. |
