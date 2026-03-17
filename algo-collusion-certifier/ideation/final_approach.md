# Final Approach: The Collusion Detection Barrier — Certified Algorithmic Audit via Compositional Testing with Automaton-Theoretic Completeness

## Overview

CollusionProof is the first **algorithmic audit framework** that produces machine-checkable collusion certificates for black-box pricing algorithms. It unifies composite statistical hypothesis testing, automaton-theoretic game analysis, and proof-carrying code into a certification pipeline with three tiers of oracle access. The system targets **EC** (Economics and Computation) as a theory contribution with a working artifact demonstration.

The approach synthesizes three design philosophies: (A) a pragmatic Layer 0 focus that ships a complete screening tool on passive price trajectory data alone; (B) two deep mathematical contributions — a **Folk Theorem converse for deterministic bounded-recall automata** (C3') and an **impossibility theorem proving bounded recall is necessary** (M8) — that elevate the work from formulation novelty to genuine theoretical contribution; and (C) compositional engineering practices — phantom-type segment isolation, dual-path rational verification, and interval arithmetic — that make the trusted proof-checker kernel defensible under adversarial scrutiny.

The guaranteed contribution is a **sound** certification framework: if CollusionProof issues a collusion certificate, the certificate is valid with quantified statistical confidence, regardless of which layer produced it, which strategy class the algorithms belong to, or whether any conjecture is true. **Completeness** — the guarantee that collusion will be detected if it exists — is proved unconditionally for all deterministic bounded-recall automata (covering every known deployed pricing algorithm) and is conditional on C3 for stochastic strategies. The M8 impossibility theorem proves this restriction to bounded recall is *necessary*: no detection scheme of any kind can identify collusion by unrestricted-memory strategies at any finite observation horizon.

## Extreme Value Delivered

### Who Needs This and Why It's Desperate

**EU DG-COMP enforcement teams (deadline: NOW).** The Digital Markets Act entered full enforcement March 2024. DG-COMP must investigate algorithmic pricing but has zero formal tools — their economists run ad-hoc Stata scripts computing price correlations that competent defense attorneys dismantle in discovery. Assad et al. (2024, *JPE*) proved algorithmically-managed German gas stations show 9% margin inflation. DG-COMP knows algorithmic collusion is real. They cannot prove it to the standard required for enforcement. CollusionProof's Layer 0 gives them the first principled statistical test with formal Type-I error control over a game-theoretic null — replacing ad-hoc correlation screens that any expert can dispute.

**DOJ Antitrust Division on the RealPage case (deadline: trial preparation).** The first federal antitrust action targeting algorithmic pricing coordination (filed 2024). The defense argument: these are independent optimization algorithms responding to identical market signals. Layer 0's composite test is designed to distinguish exactly this from coordinated behavior, with a null hypothesis that *means* "competitive behavior under any plausible demand system." The methodology — generally accepted, testable, peer-reviewed — is what satisfies the Daubert standard for expert testimony. The certificate format is forward-looking infrastructure; the statistical methodology is immediately useful.

**Competition economists (200–500 PhD economists at Compass Lexecon, CRA, NERA).** They write expert reports with variance screens (known false-positive liability from correlated demand shocks), Granger causality (no game-theoretic foundation), and bespoke simulations (not reproducible). CollusionProof's tiered null hierarchy lets them calibrate conclusions: rejection at H₀-narrow means "collusion likely under linear demand/Q-learning"; rejection at H₀-medium means "collusion likely under any parametric demand/no-regret learning." This replaces the methodological foundation, not just a tool.

### Why Existing Tools Fail Categorically

| Tool | What It Does | What It Cannot Do |
|------|-------------|-------------------|
| Calvano et al. (2020) | Simulates Q-learning collusion | No certification, no error bounds, no general algorithms |
| Gambit | Computes Nash equilibria | Cannot accept black-box algorithms, no collusion concept |
| PRISM-games | Model-checks stochastic games | Requires explicit state-space models (infeasible for proprietary algorithms) |
| PrimeNash (2025) | LLM-derived analytical equilibria | Targets described games, not empirical behavior certification |
| EGTA (Wellman) | Extracts empirical game models | No collusion formalization, no certified evidence |
| Variance/correlation screens | Detects price parallelism | No game-theoretic null, rampant false positives |

The gap is not "we need a better tool." The gap is "no tool exists in the category."

### Regulatory Timing

The window for defining the evidentiary standard for algorithmic collusion is open. Once the RealPage case establishes precedent — whatever methodology the DOJ's experts use — that methodology becomes the *de facto* standard. If precedent is set by ad-hoc correlation analysis, every future case is litigated on a fragile foundation. CollusionProof aims to define the standard *before* it calcifies. This window closes with the first major ruling.

**Critical framing**: CollusionProof is an **algorithmic audit framework** — research infrastructure for the emerging field of algorithmic competition analysis. It operates in the regulatory sandbox paradigm: certifying algorithm behavior under controlled conditions, analogous to crash-testing vehicles rather than monitoring highway driving. It is not a courtroom enforcement tool today.

## Technical Architecture

### System Design Philosophy

The architecture enforces a strict **trust boundary**: a proof-checker kernel (≤ 2,500 LoC Rust, zero external dependencies) is the only code that must be correct for certificate soundness. Everything else — simulation, statistics, certificate construction — can be buggy without compromising the guarantee that verified certificates are valid. This separation is maintained by:

- **Phantom-type segment isolation** (from Approach C): Trajectory data segments carry Rust phantom-type tags preventing cross-segment data reuse at compile time, eliminating the most dangerous class of α-inflation bugs.
- **Dual-path rational verification** (from Approach C): Every proof-relevant f64 computation is independently re-verified in exact rational arithmetic. Disagreements abort certificate construction.
- **Interval arithmetic propagation**: All statistical quantities feeding proof terms carry explicit [lo, hi] error intervals. The checker verifies worst-case endpoints.

### Tiered Oracle Architecture

| Layer | Access Model | What It Adds | Setting |
|-------|-------------|-------------|---------|
| **Layer 0** (Primary) | Passive observation only | M1 composite test + M7 directed closed testing | Public/regulator-collected price data |
| **Layer 1** | Periodic state checkpoints | + M2 deviation oracle + partial M5 Collusion Premium | Regulatory sandbox with checkpointing |
| **Layer 2** | Full rewind oracle | + M3 punishment detection + tight M5 bounds | Sandbox/voluntary audit |

Layer 0 is self-contained and independently publishable. Each layer has its own soundness guarantees. Layers 1–2 apply only in cooperative audit or sandbox settings — the sandbox owns the algorithms.

### Subsystem Breakdown: CollusionProof-Lite (Core MVP ~60K LoC)

| # | Subsystem | Language | LoC | Description |
|---|-----------|----------|-----|-------------|
| S1 | Game Simulation Engine | Rust | ~12,000 | 2 market models (Bertrand, Cournot) × linear demand × 2-player orchestration. Hot loop >100K rounds/sec. |
| S2 | Black-Box Algorithm Interface | Rust + Python | ~8,000 | Sandboxed execution, 3 algorithms (Q-learning, grim trigger, DQN), PyO3 bindings with batched GIL management. |
| S3 | Equilibrium Computation | Rust | ~4,000 | Analytical Bertrand/Cournot solvers only. No general solver needed for 2-player structured games. |
| S4 | Counterfactual Deviation Analysis | Rust | ~10,000 | Deviation enumeration, re-simulation, punishment detection (M3), deviation oracle (M2). Active for Layers 1–2; minimal stub for Layer 0. |
| S5 | Statistical Testing & Collusion Premium | Rust + Python | ~9,000 | M1 composite test battery, M7 directed closed testing, CP computation (M5) with bootstrap CIs, tiered null hierarchy. For zero-profit competitive equilibria (homogeneous Bertrand), reports absolute supra-competitive margin δ_p instead of relative CP, avoiding undefined quantities. |
| S6 | Certificate DSL & Proof Checker | Rust | ~5,000 | Certificate language, proof-term language (~15 axiom schemas, ~25 inference rules), trusted kernel ≤ 2,500 LoC, rational arithmetic verification. |
| S7 | Evidence Bundle Pipeline | Rust + Python | ~2,500 | Protobuf schema, Merkle-tree integrity, standalone verifier. |
| S8 | Evaluation Framework | Python | ~6,500 | 15 ground-truth scenarios (--standard), automated metrics, sensitivity analysis. |
| S9 | CLI & Orchestration | Rust + Python | ~3,000 | CLI, config, multiprocessing with checkpointing. |
| | **Core Total** | **Rust ~42K / Python ~18K** | **~60,000** | |

**Honest breakdown**: ~50K novel research code + ~10K essential infrastructure. The ~60K estimate is calibrated against the Skeptic's LoC deflation and the depth check's honest accounting. Every subsystem serves a distinct pipeline function; removing any breaks the certification chain.

### Language Choice Justification

**Rust** for: (1) the proof-checker kernel (memory-safe, deterministic, auditable — Python's runtime is non-deterministic and too large to audit); (2) game simulation hot loop (at Python speeds of ~1–5K rounds/sec, evaluation would require 76K–380K CPU-hours — infeasible); (3) the numerical-to-formal bridge (interval arithmetic, phantom-type enforcement).

**Python** for: (1) statistical/ML ecosystem (scipy, numpy, PyTorch for DQN); (2) RL algorithm implementations; (3) evaluation and visualization.

**PyO3 FFI**: Batched oracle calls (buffer N round-steps in Rust, acquire GIL once, batch-evaluate in Python, release) to maintain >50K rounds/sec with Python algorithms. ~2,000 LoC of bridge code.

## Mathematical Contributions

### Design Principle: Unconditional Soundness, Conditional Completeness

**Soundness** (Type-I error control) is **UNCONDITIONAL** — it holds without any assumption on C3, oracle access level, or strategy class. A certificate that passes verification is valid regardless of how it was produced.

**Completeness** is **conditional**: proved unconditionally for deterministic bounded-recall automata (covering all known deployed pricing algorithms), conditional on C3 for stochastic strategies. The M8 impossibility theorem proves that the bounded-recall restriction is *necessary*, not merely convenient.

### CORE Contributions (Guaranteed — Must-Prove)

---

#### M1: Composite Hypothesis Test over Game-Algorithm Pairs

**Statement.** The competitive null H₀ is parameterized by (demand system, learning algorithm tuple). The tiered null hierarchy provides tractability:
- **H₀-narrow**: Lipschitz-bounded linear demand × independent Q-learning
- **H₀-medium**: Lipschitz-bounded parametric demand (CES/logit) × independent no-regret learners
- **H₀-broad**: Lipschitz demand × independent learners (full family)

The composite test statistic S_T satisfies: sup_{θ ∈ H₀-tier} P_θ(S_T > c_α) ≤ α for each tier, with distribution-free Type-I error control holding exactly for permutation-based sub-tests and asymptotically for composite correlation tests.

**Proof strategy.** For H₀-narrow: bound cross-firm correlation analytically via closed-form best-response dynamics under linear demand (finite-dimensional calculation). For H₀-medium: optimization over finite-dimensional parametric family with standard covering arguments. For H₀-broad: covering-number arguments over Lipschitz function spaces with Berry-Esseen finite-sample corrections; parametric sub-family provides exact distribution-freeness as fallback for all T.

**Load-bearing justification.** Without M1, Layer 0 is a heuristic screen indistinguishable from existing ad-hoc methods. M1 is fully load-bearing — it is what makes Layer 0 a scientific contribution.

**Achievability.** H₀-narrow: 90%+ (1–2 person-months). H₀-medium: 70% (2–3 person-months). H₀-broad: 35–45% (4–6 person-months, with risk of vacuous constants). **The tiered null hierarchy is the practical power strategy, not a fallback.** Rejection at H₀-narrow already exceeds every existing screening method. Rejection at H₀-medium is the practical ceiling for most applications. H₀-broad is a stretch goal.

**Honest grade: B+ for H₀-narrow/medium (novel application of established techniques), A− for H₀-broad if tight and non-vacuous.**

**Person-months: 3–5 (narrow + medium guaranteed; broad attempted).**

---

#### C3': Folk Theorem Converse for Deterministic Bounded-Recall Automata (The Collusion Detection Theorem)

**Statement.** Let σ = (σ₁, …, σ_N) be a profile of deterministic finite-state automata, each with at most M states, playing a repeated pricing game Γ. If the average payoff profile satisfies π̄_i(σ) ≥ π^NE_i + η for all i (η-collusion), then for every player i there exists a unilateral deviation d_i such that:

(a) **Detectable**: opponents' response within M rounds produces payoff drop ≥ η/(M·N).

(b) **Identifiable**: post-deviation trajectory differs from on-path in ≥ ⌈η·M/(π̄_max − π_minimax)⌉ of the M rounds following deviation.

**Proof strategy.** Model each σ_i as a Mealy machine (Q_i, Σ_i, δ_i, λ_i). Joint automaton operates on product state space Q with |Q| ≤ M^N. On-path play traces a cycle C in Q of length ≤ M^N. If σ sustains η-collusion, every state on C yields payoffs above π^NE + η. Deviation to static Nash price p^NE_i causes exit from C into a transient of length ≤ M. If no punishment occurs (payoff drop < η/(M·N) every round), player i's M-round average exceeds π^NE_i + η(1 − 1/N) — strictly above Nash. If no player triggers punishment, the profile is not self-enforcing (profitable deviations with no deterrent), contradicting the equilibrium assumption. Counting over N players and M rounds yields Δ_P ≥ η/(M·N).

**Load-bearing justification.** Without C3', completeness is entirely conjectural. With C3', the system has unconditional completeness for deterministic automata — covering Q-learning with discretized Q-tables, grim-trigger, tit-for-tat, and all lookup-table strategies. This converts the contribution from "sound screening tool" to "sound-and-complete certification framework for the practically relevant strategy class."

**Achievability.** 85% (2–3 person-months). The graph-theoretic argument is clean and well-motivated. This is hard but tractable — the Myhill-Nerode connection to strategic incentive analysis is genuinely novel.

**Honest grade: A.** A new theorem in repeated game theory connecting automaton structure to collusion detectability.

**Person-months: 2–3.**

---

#### M8: Impossibility of Detection Without Bounded Recall (The No-Free-Lunch Theorem)

**Statement.** For any finite sample size T and any test Ψ that is α-sound against H₀-broad:

sup_{σ ∈ Σ_∞(η)} P_σ(Ψ accepts) ≥ 1 − α − 2exp(−T/2)

where Σ_∞(η) is the class of all (not necessarily finite-state) strategies sustaining η-collusion. Without the bounded-recall assumption, any sound test has trivial power against some collusive strategy.

**Proof strategy.** Construct a "stealth collusion" strategy σ^T_stealth that: (i) sustains payoffs at π^NE + η, (ii) stores entire history in an infinite-state register, (iii) only punishes after a deviation sequence of length ≥ T+1 (never occurring in T rounds). The T-round marginal is identical to a competitive process by construction, so d_TV(P_{σ^T_stealth}^T, H₀) = 0 and any α-sound test must accept with probability ≥ 1 − α.

**Load-bearing justification.** Without M8, the bounded-recall restriction in the problem formulation looks like a design limitation. With M8, it is a **fundamental barrier**: any detection system must make this restriction or an equivalent one. This transforms a potential weakness into a structural insight about the problem.

**Achievability.** 90%+ (1–2 person-months). The construction is conceptually clean; the formalization requires care to ensure the stealth strategy's T-round marginal exactly matches a member of H₀.

**Honest grade: A−.** An impossibility theorem for economic certification. The delayed-punishment construction is straightforward but the formalization in this context is novel. Slightly over-graded at A in Approach B — experienced game theorists will find the construction "obviously true," reducing perceived novelty.

**Person-months: 1–2.**

---

#### M6: Certificate Verification Soundness

**Statement.** If the proof checker accepts a certificate, the claimed statistical conclusions hold (with probability ≥ 1−α over the original randomness). Requires: (a) soundness of the axiom system (every axiom is a true statement about the game-theoretic domain), (b) correctness of rational arithmetic re-verification, (c) proof that f64-to-rational conversion preserves all ordinal relations used in proof derivations.

**Proof strategy.** Structural induction over proof terms, verifying each axiom schema against the domain semantics. The axiom system is deliberately small (~15 schemas, ~25 inference rules) to keep soundness verification tractable.

**Load-bearing justification.** Without M6, certificates are unverified data bundles — the PCC paradigm collapses. Fully load-bearing for the certification framing.

**Achievability.** 80–90% (2–3 person-months). The hard part is iterative axiom design — getting the system expressive enough yet sound. Budget 3+ design iterations, each potentially invalidating previously generated certificates.

**Honest grade: C.** Careful verification exercise, not mathematical innovation. The novelty is in the domain, not the technique.

**Person-months: 2–3.**

---

#### M7: Directed Closed Testing for Collusion Signatures

**Statement.** Collusion-structured rejection ordering (supra-competitive pricing → punishment → correlation → convergence) improves power against typical collusion alternatives while maintaining FWER control via Holm-Bonferroni.

**Load-bearing justification.** Without M7, the composite test uses generic omnibus ordering with lower practical power. Partially load-bearing — improves power but is not structurally necessary.

**Achievability.** 95%+ (0.5–1 person-month).

**Honest grade: D.** Engineering application of standard methods with domain-specific ordering.

**Person-months: 0.5–1.**

---

### CORE Summary

| ID | Name | Grade | Achievability | Person-Months | Layer |
|----|------|-------|--------------|---------------|-------|
| M1 | Composite Hypothesis Test (narrow + medium) | B+ | 85% | 3–5 | 0 |
| C3' | Folk Theorem Converse (deterministic) | A | 85% | 2–3 | 0–2 |
| M8 | Impossibility without bounded recall | A− | 90% | 1–2 | 0 |
| M6 | Certificate verification soundness | C | 85% | 2–3 | 0–2 |
| M7 | Directed closed testing | D | 95% | 0.5–1 | 0 |
| **Total Core** | | | **~75% all succeed** | **9–14** | |

**Probability all CORE theorems proved: ~75%.** This is the reliable foundation. The paper can be written with high confidence about what it contains.

### STRETCH Contributions (Attempted — Honestly Labeled)

---

#### M1-broad: H₀-broad Distribution-Freeness

**Statement.** Extend M1's Type-I error guarantee to the full Lipschitz demand × independent learner null family with computable, non-asymptotic remainder R(T, L, N).

**Why it's hard.** "Independent learners" is not a well-defined function class in empirical process theory. The metric entropy of the algorithm factor is an open question. The correlation bound over the full Lipschitz family may be so loose that H₀-broad tests have zero practical power for T < 10^10.

**Achievability.** 35–45% (4–6 additional person-months). May yield a vacuous bound. Honest assessment: H₀-medium is likely the practical ceiling.

**What happens if it fails.** The contribution is "a principled test for markets where you know the demand family" rather than "a universal screen." This reintroduces expert judgment on demand specification — weaker but still far superior to existing methods.

---

#### C3'-stochastic: Stochastic Automaton Extension

**Statement.** For stochastic finite-state automata with τ_mix ≤ poly(M), E[Δ_P] ≥ η/(M³·N) and punishment is observable with probability ≥ 1 − exp(−Ω(η²/(M²·σ²_P))) within M + τ_mix rounds.

**Why it's hard.** The coupling argument between on-path and post-deviation Markov chains must handle different transition kernels (the deviating player's kernel changes). Two chains with different kernels do not necessarily re-couple within O(τ_mix) of either chain. The post-deviation chain's mixing time is unknown and potentially much larger. Boltzmann Q-learning with low temperature has τ_mix = Θ(exp(1/T_temp)), which may render the bound vacuous.

**Achievability.** 50–60% for polynomial-mixing chains (3–5 additional person-months). Full frontier characterization: 15–25%, likely beyond project timeline.

**What happens if it fails.** The paper presents C3' for deterministic automata as the main theorem with the stochastic extension as a conjecture and partial evidence. This is still a major advance over the original problem statement (which left C3 entirely as a conjecture).

---

#### M4'-lower: Minimax Lower Bound for Collusion Detection

**Statement.** For any α-sound test Ψ: inf_{M-state η-collusion} P(Ψ rejects) ≤ 1 − β whenever T ≤ c · M² · σ² / (η² · log(1/α)).

**Why it's hard.** Requires constructing a competitive distribution P₀ that matches an M-state collusive distribution P₁ in total variation through T* rounds. The embedding of a collusive automaton's trajectory in the competitive null via Lipschitz demand function design may fail for automata with discrete-support trajectory distributions that no continuous demand function can reproduce.

**Achievability.** 40–50% (3–5 additional person-months).

**What happens if it fails.** The system's sample complexity could be dismissed as an artifact of algorithm design rather than a fundamental limit. We lose the "optimality" narrative but retain all practical results. The upper bound T* = O(M²N²σ²τ²_mix·log(K/α)/(η²β)) still provides data-budget guidance for regulators.

---

### STRETCH Summary

| ID | Name | Grade | Achievability | Addl. Person-Months |
|----|------|-------|--------------|---------------------|
| M1-broad | H₀-broad distribution-freeness | A− | 35–45% | 4–6 |
| C3'-stoch | Stochastic automaton extension | A+ (if deep structure found) | 50–60% | 3–5 |
| M4'-lower | Minimax lower bound | A | 40–50% | 3–5 |

**Probability at least one STRETCH succeeds: ~75%.** Probability all three succeed: ~10%.

## Genuine Difficulty as Software Artifact

### Hard Subproblem 1: Compositional Soundness Across Heterogeneous Probability Spaces

The composite test (M1) combines K sub-tests over different probability spaces with different conditioning events. Composing via directed closed testing (M7) with FWER ≤ α requires proving composition preserves α-control *uniformly* over the infinite-dimensional null. The sub-tests share data and have complex dependencies. A single data-reuse bug silently inflates false positive rates without runtime errors.

**Engineering response (from Approach C)**: Phantom-type segment isolation in Rust. Trajectory segments carry phantom-type tags making cross-segment access a compile-time error. α-budget accounting tracks consumption per sub-test, aborting certificate construction on overdraft. ~4,000 LoC of segment isolation + budget enforcement.

### Hard Subproblem 2: The Numerical-to-Formal Bridge

Simulations run in f64 for performance. Certificates require exact rational arithmetic. IEEE 754 arithmetic is not monotone under conversion — two f64 values satisfying `a > b` may fail `rat(a) > rat(b)` after conversion.

**Engineering response (from Approach C)**: (1) Comparison-tracking layer logging proof-relevant comparisons (~2,000 LoC annotation infrastructure). (2) Interval arithmetic wrappers on all statistical computations contributing to proof terms (~1,500 LoC). (3) Dual-path verification: every proof-relevant computation runs twice (f64 fast path + rational slow path); disagreements abort. (4) Rational re-derivation in the trusted kernel (~1,000 LoC). Total: ~4,500 LoC of load-bearing bridge code — highest difficulty-per-LoC in the project.

### Hard Subproblem 3: De Novo Proof Checker for a De Novo Domain

~15 axiom schemas, ~25 inference rules, ≤ 2,500 LoC trusted kernel, zero dependencies. Must encode statistical inference, rational arithmetic, and game-theoretic properties while remaining auditable. Every axiom is a potential soundness hole.

**Risk mitigation**: (1) Extensive adversarial testing with certificates that *should* fail. (2) Budget 3+ design iterations for axiom stabilization (6–8 weeks, not "design and move on"). (3) Axiom schema documentation with explicit domain-semantic justification for each. (4) No encoding of complex probabilistic arguments (coupling, Markov chains) in the checker — these are established by the math proofs and enter as atomic axioms about their conclusions, not their proof structure.

### Hard Subproblem 4: PPAD Avoidance on the Critical Path

Layer 0 must operate entirely without Nash equilibrium computation (PPAD-complete for N > 2). NE computation needed only for Collusion Premium benchmarking (M5, Layers 1–2) using analytical Bertrand/Cournot solvers for 2-player structured games. For homogeneous Bertrand markets where the competitive equilibrium has zero profit, the relative Collusion Premium CP = (π_obs − π_NE)/π_NE is undefined; the system reports an absolute supra-competitive margin δ_p that smoothly reduces to the relative CP when competitive profits are strictly positive, with a normalized Collusion Index CI = CP/(1+CP) ∈ [0,1] for intuitive regulatory reporting.

### Hard Subproblem 5: PyO3 FFI Performance Under GIL Contention

Naive per-round GIL acquisition costs ~500ns/call, reducing throughput below targets. Batched oracle calls (buffer N steps in Rust, single GIL acquisition, batch evaluation in Python, release) are mandatory. ~2,000 LoC of bridge code with backpressure management.

### Architectural Challenge: Trust Boundary Maintenance

The proof-checker kernel is the only code that must be correct for soundness. Maintaining this boundary through development — ensuring no subsystem injects unverified claims into the certificate chain — requires disciplined API design and continuous integration testing. The Skeptic's concern about axiom system iterations is valid: each iteration potentially invalidates all previously generated certificates, requiring coordinated rework across S5 (certificate construction) and S6 (checker).

## Best-Paper Argument

CollusionProof targets an **EC best paper** through four reinforcing strengths:

**1. A genuinely new problem instance in statistics.** The composite hypothesis test where H₀ is parameterized by demand systems × learning algorithms is unprecedented. The game-theoretic structure of the null changes the mathematical problem — nuisance parameters interact through market equilibrium, not independently. EC rewards formulation novelty, and the tiered null hierarchy is a principled approach to the power-generality tradeoff.

**2. A new theorem in repeated game theory.** The deterministic C3' proves the first Folk Theorem converse for bounded-recall automata — connecting automaton cycle structure to collusion detectability. Combined with M8 (impossibility for unbounded recall), this yields a clean **dichotomy**: collusion by bounded-recall automata is certifiably detectable; collusion by unrestricted strategies is provably undetectable. This is the kind of structural characterization EC committees reward — compare Roughgarden's smoothness framework (EC 2009 best paper), which separated efficient from inefficient equilibria.

**3. Category creation.** Machine-checkable collusion certificates have zero precedent. The prior art audit is exhaustive: formal methods has PCC for memory/type safety; economics has screening without certificates; game theory has equilibrium computation without certification; antitrust has testimony without machine verification. CollusionProof is the first system at the center. Category creation — not incremental improvement — is the hallmark of best papers.

**4. Perfect timing + working artifact.** EC 2025/2026 program committees will be intensely interested in algorithmic pricing given the DMA, RealPage, and Assad et al. A paper with the first formal framework *plus* a working ~60K LoC artifact producing verifiable certificates on a laptop in <30 minutes is the theory-meets-practice demonstration EC values.

**Why this hybrid beats each pure approach:**
- Pure Approach A: too thin mathematically — formulation novelty alone is B+, not best-paper caliber.
- Pure Approach B: too risky — 5 load-bearing theorems with ~15% probability of full completion; the "barrier theorem" narrative requires all components.
- Pure Approach C: wrong venue — engineering difficulty cannot carry an EC paper.
- **This hybrid**: A's achievable scope + B's two strongest theorems (deterministic C3' + M8) + C's engineering practices = higher floor than B, higher ceiling than A, better venue fit than C.

## Hardest Technical Challenge

**Proving distribution-free Type-I error control for M1 over H₀-medium (parametric demand × no-regret learners).**

H₀-narrow (linear demand × Q-learning) has a closed-form correlation bound — tractable. H₀-medium is the practical ceiling for the contribution's reach and requires bounding the maximum achievable cross-firm correlation over a parametric demand family where firms use independent no-regret learners.

**Why it's hard**: The cross-firm price correlation under the competitive null depends on the demand system. Two independent learners on a highly correlated demand system produce trajectories that look coordinated. Bounding the supremum over parametric demand families requires:
1. Characterizing the optimization over a finite-dimensional parameter space where demand parameters interact with learning dynamics
2. Proving the bound is tight enough for practical power (a loose bound makes the test useless)
3. Berry-Esseen finite-sample corrections with constants depending on unknown third moments under the null

**How to address it:**
1. **Tier honestly.** Prove H₀-narrow first (guaranteed). Attack H₀-medium second. Present H₀-broad as a stretch with honest uncertainty.
2. **Finite-sample fallbacks.** (a) Berry-Esseen corrections bounding the gap; (b) parametric sub-family with exact non-asymptotic distribution-freeness for all T; (c) permutation-based sub-tests with exact finite-sample validity at narrower alternative cost.
3. **Validate empirically.** Run on 10+ known-competitive scenarios across 200+ seeds (not 50 — the Skeptic correctly notes that 50 seeds × 8 scenarios = 400 runs is insufficient to detect α-inflation from 0.05 to 0.08). Target ≥1,500 runs per scenario for adequate Type-I validation power.

## Evaluation Plan

### Fully Automated — No Human Annotation

All evaluation is automated end-to-end. Zero human annotation, zero subjective judgments.

### Three-Tier Compute Budget

| Mode | Scenarios | Algorithms | Rounds | Seeds | CPU-Hours | Wall-Clock (8 cores) | Purpose |
|------|-----------|-----------|--------|-------|-----------|---------------------|---------|
| `--smoke` | 5 (2 collusive, 2 competitive, 1 boundary) | Tabular RL only | 100K | 2 | < 0.5 | **< 30 min** | CI and development |
| `--standard` | 15 | Tabular + bandit, 2–3 players | 1M | 10 | ~800 | **~4 days** | Milestone validation |
| `--full` | 30 | All algorithms incl. DQN | 10M | 20 | ~3,800 | **~20 days** | Camera-ready only |

Development uses `--smoke` exclusively. `--standard` at milestones. `--full` once for camera-ready.

### Ground-Truth Benchmark Suite (30 scenarios, full mode)

| Category | Count | Ground Truth | Key Algorithms |
|----------|-------|-------------|----------------|
| Known-collusive | 10 | CP > threshold, punishment detected | Grim trigger, tit-for-tat, Q-learning (Calvano replication), DQN |
| Known-competitive | 8 | CP ≈ 0, no punishment | Myopic best-response, ε-greedy bandit, competitive strategies |
| Boundary/hard | 8 | Variable — tests discrimination | Mixed pairs, partial collusion, Edgeworth cycles, asymmetric players |
| Adversarial red-team | 4 | Known-collusive but designed to evade | Collusion with injected noise, randomized punishment timing, correlation-mimicking strategies |

### Metrics

- **Classification**: Precision, recall, F1 for COLLUSIVE/COMPETITIVE/INCONCLUSIVE trichotomy
- **CP calibration**: Does Collusion Premium correctly rank scenarios by collusion severity?
- **ROC analysis**: AUC from 100+ parametric sweep evaluation points (5 anchor scenarios × 20 parameter variations)
- **Type-I error validation**: ≤ α false positive rate on 8 competitive scenarios × 200+ seeds per scenario, separately per null tier
- **Statistical power**: Fraction of collusive scenarios correctly certified at α = 0.05, per null tier
- **Adversarial evasion rate**: Fraction of red-team scenarios evading detection per tier
- **Certificate verification rate**: Fraction passing independent checker verification
- **Layer comparison**: Certificate strength across Layers 0/1/2 for scenarios tested at multiple access levels

### Baselines

| Baseline | What It Represents |
|----------|-------------------|
| Price correlation screen | Standard regulatory Pearson/Spearman screening |
| Variance screen | Low-variance-as-collusion heuristic (competition authority practice) |
| Granger causality test | Time-series causal screening |
| Gambit equilibrium check | NE computation + comparison to observed prices (no certification) |
| Calvano-style comparison | Reproduce Calvano et al. methodology |

### Statistical Validation

- Bootstrap CIs on all aggregate metrics (1,000 samples)
- Benjamini-Hochberg correction across scenario families
- Power analysis with effect size estimation validating that scenario × seed counts are sufficient
- Sensitivity analysis via Latin hypercube over: round count, discount factor, demand parameters, algorithm hyperparameters

## Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation | What If It Happens |
|------|------------|--------|-----------|-------------------|
| **M1 H₀-broad bound is vacuous** | 55–65% | Medium | Present tiered hierarchy as primary strategy; H₀-narrow/medium are the practical contribution | Paper is "principled test for markets with known demand family" — still beats all existing tools |
| **Deterministic C3' proof fails** | 15% | High | Retain restricted proofs (grim-trigger, tit-for-tat); present general deterministic case as conjecture with strong evidence | Lose unconditional completeness for general automata. Completeness for specific strategy classes still exceeds prior art. **Kill gate**: if the basic automaton argument has a gap by Phase 2 end, restructure paper around M1 + M8 only |
| **Proof checker axiom unsoundness** | 20% | Critical | Adversarial testing, 3+ design iterations, external review; budget 6–8 weeks for axiom stabilization | Single unsound axiom invalidates all certificates. Detection via adversarial certificates that should fail. Mitigation by keeping axiom system minimal |
| **Berry-Esseen constants dominate signal** | 40% | Medium | Parametric fallback sub-family with exact validity; empirical calibration | H₀-medium reverts to asymptotic validity with empirical evidence. Honest reporting |
| **PyO3 throughput < 50K rounds/sec** | 25% | Low-Medium | Pure-Rust algorithm implementations for core strategies; Python only for DQN | Development iteration slows; --standard evaluation extends from 4 to ~8 days |
| **M8 dismissed as "trivially true"** | 30% | Low | Frame as necessary structural result completing the dichotomy, not as standalone novelty | M8 supports C3' narrative. If both C3' and M8 stand, the dichotomy is the contribution, not M8 alone |
| **Stochastic C3' coupling argument breaks** | 40–50% | Low (stretch) | Present deterministic C3' as main theorem; stochastic as open problem with partial results | Stated as stretch goal from the start. No narrative damage |
| **Regulatory window closes (RealPage precedent set)** | 30% (within project timeline) | Medium | Framework remains valuable as methodological contribution independent of specific cases | Timing advantage reduced but not eliminated. Assad et al. + DMA ensure ongoing relevance |

### Kill Gates

| Gate | Condition | Decision |
|------|----------|----------|
| **KG1: Phase 1 end (Month 3)** | M1 H₀-narrow proof fails OR basic automaton C3' argument has a fundamental gap | **Kill the project.** If the restricted M1 or deterministic C3' are not achievable, the contribution is insufficiently novel for EC. |
| **KG2: Phase 2 end (Month 5)** | Proof checker cannot stabilize after 3+ axiom iterations AND certificate verification fails on >5% of test cases | **Pivot to theory-only paper.** Drop the artifact and PCC framing. Submit M1 + C3' + M8 as a pure theory paper. |
| **KG3: Phase 3 midpoint (Month 7)** | Empirical Type-I error rate exceeds 2α on competitive scenarios at any null tier | **Debug or descope.** Either a statistical bug exists (fix it) or the finite-sample corrections are inadequate (honestly report the tier where validity holds empirically). |

## Timeline

### Phase 1: Mathematical Foundation (Months 1–3)

| Milestone | Target | Deliverable |
|-----------|--------|------------|
| 1.1 | Month 1 | M1 H₀-narrow proof complete. Closed-form cross-firm correlation bound under linear demand × Q-learning. |
| 1.2 | Month 2 | Deterministic C3' proof complete. Automaton cycle-structure argument with Δ_P ≥ η/(M·N) bound. |
| 1.3 | Month 2.5 | M8 impossibility proof complete. Stealth-collusion construction with formal d_TV argument. |
| 1.4 | Month 3 | M1 H₀-medium proof complete or clear assessment of feasibility. M7 test ordering validated. **Kill Gate KG1 evaluation.** |

### Phase 2: Core Infrastructure (Months 2–5, overlapping with Phase 1)

| Milestone | Target | Deliverable |
|-----------|--------|------------|
| 2.1 | Month 3 | S1 (game engine), S2 (algorithm interface), S3 (equilibrium solvers) functional. >100K rounds/sec demonstrated. |
| 2.2 | Month 4 | S5 (statistical testing) implementing M1 for H₀-narrow. First end-to-end Layer 0 certificate on a Bertrand scenario (even if M6 checker is a stub). |
| 2.3 | Month 4.5 | S6 proof checker v1 with initial axiom system. First verified certificate. |
| 2.4 | Month 5 | S4 (counterfactual analysis) for Layers 1–2. S6 axiom stabilization iteration 2. **Kill Gate KG2 evaluation.** |

### Phase 3: Integration & Evaluation (Months 5–8)

| Milestone | Target | Deliverable |
|-----------|--------|------------|
| 3.1 | Month 5.5 | S7 evidence bundles + S9 CLI complete. `--smoke` evaluation passing (5 scenarios, all certificates verified). |
| 3.2 | Month 6 | S6 axiom stabilization iteration 3. `--standard` evaluation initiated. Type-I validation on competitive scenarios. |
| 3.3 | Month 7 | `--standard` evaluation complete. Empirical results compiled. **Kill Gate KG3 evaluation.** Begin stretch goals if bandwidth allows. |
| 3.4 | Month 8 | `--full` evaluation run. Paper writing. Stretch goal results (if any) incorporated. Camera-ready. |

### MVP vs Extended Scope

**In the MVP (~60K LoC):**
- Layer 0 + Layer 1 (Layer 2 as extension)
- 2-player Bertrand/Cournot with linear demand
- Tabular RL (Q-learning, grim trigger) + DQN
- Analytical equilibria only
- M1 (narrow + medium), C3' (deterministic), M8, M6, M7
- 15 evaluation scenarios (--standard)

**In the extended scope (~110–130K LoC):**
- Layer 2 full rewind
- 3–4 player support, CES/logit demand
- PPO, SARSA, bandits, Edgeworth, additional strategies
- General equilibrium solvers (support enumeration, Lemke-Howson)
- Stretch theorems (M1-broad, C3'-stochastic, M4'-lower)
- 30 evaluation scenarios (--full) + report generator

## Scores

### Value: 7/10

The regulatory timing is genuine and urgent (EU DMA, DOJ RealPage, Assad et al. empirical evidence). Three stakeholder groups need this tool, and the gap is categorical. Layer 0 alone surpasses every existing deployable screening method. **Docked from 10 because**: (a) Layer 0 is a screening tool, not courtroom evidence — regulators need it but it doesn't directly win cases today; (b) Layers 1–2 require cooperative access models that don't yet exist in most jurisdictions; (c) adoption by conservative legal institutions takes years; (d) DG-COMP's actual bottleneck is legal authority, not statistical tools — though both are needed.

### Difficulty: 7/10

The ~60K LoC MVP contains genuine research difficulty: M1's distribution-freeness proof, C3' automaton-theoretic argument, M8 impossibility construction, the de novo proof checker, the numerical-to-formal bridge, and compositional α-control with phantom-type enforcement. The Rust/Python split is architecturally motivated and adds integration complexity. **Docked from 10 because**: (a) Layer 0 avoids the hardest subproblems (deviation oracle, punishment detection in the stochastic setting); (b) the proof checker's axiom system is deliberately small (~15 schemas); (c) Bertrand/Cournot 2-player are well-understood markets; (d) the Skeptic correctly notes that much of the artifact is "competent engineering, not research difficulty."

### Potential: 7/10

EC best-paper potential is real through the combination of: genuinely new problem formulation (M1), new theorem in repeated game theory (C3'), impossibility result (M8), category-creating artifact (PCC for economics), uncontested triple intersection, and perfect policy timing. The clean dichotomy — bounded recall enables detection, unbounded recall defeats it — is quotable and structurally appealing. **Docked from 10 because**: (a) M1 is formulation novelty — the semiparametric testing techniques exist, the application is new; (b) C3' for deterministic automata alone, while novel, is narrower than the full collusion detection barrier of Approach B's vision; (c) M8 may be perceived as "obviously true" by experienced game theorists; (d) the artifact must be compelling enough to reviewers who won't run the code.

**Probability of EC acceptance: 55–65%. Probability of EC best paper: 15–20%.** If a stretch goal (especially C3'-stochastic or M4'-lower) succeeds, best-paper probability rises to 25–30%.

### Feasibility: 6/10

The core math program (M1 narrow/medium + C3' deterministic + M8 + M6 + M7) has ~75% probability of full completion. The ~60K LoC artifact is achievable in 6–8 months with the core/extended split. The three-tier evaluation budget makes iteration tractable. **Docked from 10 because**: (a) proof checker axiom stabilization requires 3+ iterations over 6–8 weeks, each potentially requiring rework; (b) the Berry-Esseen constants for H₀-medium may have impractical magnitude; (c) 800 CPU-hours for --standard evaluation is 4 days on 8 cores — tight for iterative development; (d) any late-stage C3' gap cascades into proof checker and certificate rework; (e) the Skeptic's valid concern that math-engineering coupling creates schedule risk, though the CORE/STRETCH separation mitigates this.

### Score Justification Summary

| Dimension | Self-Score | Skeptic's Correction (range) | Final | Rationale |
|-----------|-----------|------------------------------|-------|-----------|
| Value | 8 (A) / 7 (B,C) | 6 (all) | **7** | Genuine urgency and categorical gap, but regulatory adoption is realistically years away |
| Difficulty | 7 (A) / 9 (B) / 8 (C) | 6 (A) / 8 (B) / 7 (C) | **7** | Core is engineering-hard + research-hard at C3'/M8; honest about Layer 0 avoiding the hardest problems |
| Potential | 8 (A) / 9 (B) / 7 (C) | 6 (A) / 7 (B) / 5 (C) | **7** | C3' + M8 elevate beyond A's ceiling; stretch goals could push to 8+ |
| Feasibility | 7 (A) / 5 (B) / 6 (C) | 6 (A) / 4 (B) / 6 (C) | **6** | CORE math is ~75% achievable; kill gates prevent death spirals; stretch goals are honestly optional |
| **Composite** | | | **6.75** | Bounded below by A's floor (6.0), above by B's realistic achievable ceiling (~6.5), lifted by the hybrid's better risk profile |
