# CollusionProof: Proof-Carrying Collusion Certificates via Compositional Statistical Testing and Counterfactual Deviation Analysis for Black-Box Algorithmic Pricing Markets

## Problem Statement

Antitrust regulators worldwide face a crisis of formal evidence. When pricing algorithms deployed by competing firms independently converge to supra-competitive prices—as documented by Calvano et al. (2020, *AER*) for Q-learning agents and by Assad et al. (2024, *JPE*) showing significant margin increases in German gasoline duopolies—no formal, reproducible, machine-checkable method exists to distinguish tacit algorithmic collusion from competitive equilibrium. The EU Digital Markets Act entered full enforcement in March 2024. The U.S. DOJ's RealPage rental-pricing investigation (filed 2024) is the first federal antitrust action targeting algorithmic pricing coordination. The FTC's scrutiny of dynamic pricing shares the same bottleneck: regulators can detect suspicious price correlations, but cannot produce a *certificate*—a self-contained, independently verifiable evidence bundle—demonstrating that observed pricing outcomes satisfy the game-theoretic conditions for self-enforcing collusion. Current expert testimony relies on ad-hoc econometric analysis that opposing counsel can dispute at every assumption. There is no machine-checkable evidentiary standard for algorithmic collusion. The window for defining such a standard is open—before ad-hoc approaches calcify into legal precedent.

Existing tools fail at every layer of this problem. **Calvano et al.** demonstrated collusion in simulation but only for tabular Q-learning, with no formal certification, no counterfactual deviation testing, and no machine-readable evidence output. **Gambit** (McKelvey, McLennan & Turocy) computes Nash equilibria for finite games but cannot accept black-box algorithms as input, has no concept of collusion, and produces no proof objects. **PRISM-games** (Kwiatkowska et al.) model-checks stochastic multiplayer games but requires explicit state-space models—infeasible for black-box pricing algorithms—and encodes no antitrust-relevant properties. **PrimeNash** (Zhang et al., 2025) uses an LLM agent to derive analytical equilibria with machine-checkable proofs, but targets game-solving for analytically described games, not empirical collusion certification of observed behavior. EGTA (Wellman et al.) extracts empirical game models from multi-agent simulations but does not formalize collusion detection or produce certified evidence. Statistical screening methods used by competition authorities (variance screens, Granger causality, price parallelism tests) lack game-theoretic foundations and produce no independently verifiable evidence. Adjacent theoretical work—Horner et al. on repeated games with imperfect monitoring, Abreu (1988) on the repeated-game folk theorem, Harrington (2008) on econometric collusion screening, Klein (2021) and Johnson et al. (2023) on algorithmic pricing simulation—addresses individual facets of the problem. None produces certificates. The triple intersection of formal verification, game theory, and competition law has **no known active research groups** producing tools.

**CollusionProof** fills this gap by providing an algorithmic audit framework—research infrastructure for the emerging field of algorithmic competition analysis—that unifies complementary mathematical frameworks into a certification pipeline with three explicit layers of oracle access:

**Layer 0 — Passive Observation** (no oracle access). The primary deployable contribution. Working entirely on publicly available or regulator-collected price trajectory data, the system implements a composite hypothesis testing framework (M1) where the null hypothesis H_0 encompasses the entire family of competitive behaviors: any Lipschitz demand system paired with any tuple of independent learning algorithms. A battery of tests—targeting excess price correlation, punishment response asymmetry, supra-competitive price persistence, and convergence pattern anomalies—is composed via directed closed testing (M7) with Holm-Bonferroni FWER control. This layer requires *no equilibrium computation* (avoiding the PPAD-hardness barrier), *no oracle access*, and *no algorithm cooperation*. It produces statistical evidence with formal Type-I error control—soundness is unconditional. Layer 0 is self-contained and independently publishable.

**Layer 1 — Replay Oracle** (periodic state checkpoints). In cooperative audit or regulatory discovery settings where algorithm state snapshots are available at periodic intervals (not arbitrary prefixes), Layer 1 adds the deviation oracle (M2) with certified bounds and partial Collusion Premium quantification (M5) with wider confidence intervals reflecting checkpoint granularity. This setting corresponds to the regulatory sandbox paradigm: competition authorities require firms to submit pricing algorithms into a controlled test environment where periodic checkpointing is a design feature. Layer 1 substantially increases evidential power over Layer 0 while remaining feasible in realistic regulatory contexts.

**Layer 2 — Full Rewind Oracle** (arbitrary history prefix restart). In sandbox or voluntary audit settings where algorithms can be replayed from arbitrary history prefixes, Layer 2 adds punishment detection via controlled perturbation (M3) and tight Collusion Premium bounds (M5). Full interactive oracle access with rollback enables the strongest certificates. This layer applies in sandbox/voluntary audit only—the sandbox owns the algorithms.

CollusionProof operates in the regulatory sandbox paradigm: competition authorities require firms to submit pricing algorithms into a controlled test environment. In this setting, oracle rewind is a design feature—the sandbox owns the algorithms. CollusionProof does not monitor live markets; it certifies algorithm behavior under controlled conditions, analogous to crash-testing vehicles rather than monitoring highway driving. Each layer has its own soundness guarantees, contribution claims, and implementation scope.

**Soundness and completeness guarantees.** Soundness (Type-I error control) is UNCONDITIONAL—it holds without any assumption on Conjecture C3, oracle access level, or strategy class. This is a critical design choice: a certificate that passes verification is valid regardless of which layer produced it or whether C3 is true. Completeness—the guarantee that the system will detect collusion when it exists—is conditional on Conjecture C3 (the Folk Theorem converse for bounded-recall strategies). For restricted strategy classes (deterministic bounded-recall automata with at most M states), we prove C3 directly, yielding unconditional completeness in those cases. If C3 is false for some exotic strategy class, "stealth collusion" strategies may exist that evade detection, but the system never produces false positives. The system functions coherently if C3 is false: it retains full soundness and partial completeness for all strategies where C3 holds.

The system addresses the folk theorem's core challenge—that any feasible, individually rational payoff can be sustained as an equilibrium in sufficiently patient repeated games—by restricting analysis to bounded-recall strategies and measuring collusion on a continuous scale rather than as a binary classification. A Collusion Premium CP(p_bar) quantifies the proportional excess profit above the most favorable competitive equilibrium; for zero-profit competitive equilibria (e.g., homogeneous Bertrand), the system reports an absolute supra-competitive margin delta_p rather than the relative CP, avoiding undefined quantities. For intuitive regulatory reporting, a normalized Collusion Index CI = CP/(1+CP) in [0,1] locates observed outcomes on the spectrum from perfect competition (CI ~ 0) to extreme collusion (CI -> 1). The composite test employs a tiered null hierarchy: H_0-narrow (Lipschitz-bounded linear demand x independent Q-learning), H_0-medium (Lipschitz-bounded parametric demand x independent no-regret learners), and H_0-broad (Lipschitz demand x independent learners). Certificates report rejection at each tier—rejection at H_0-broad is strongest evidence; rejection at H_0-narrow alone still exceeds current screening methods.

Distribution-free Type-I error control holds exactly for permutation-based sub-tests (M3 punishment detection) and asymptotically for the composite correlation tests (M1). For finite T, we provide (a) finite-sample correction terms via Berry-Esseen bounds, and (b) a fallback parametric sub-family where exact distribution-freeness holds for all T.

A domain-specific certificate language and a small, auditable proof-checker kernel (at most 2,500 LoC, zero external dependencies) verify the logical chain from raw data to verdict. This brings the proof-carrying code paradigm—previously applied only to memory safety and type safety (Necula, 1997)—into economic certification for the first time.

CollusionProof enables capabilities that do not currently exist. Regulators gain an audit tool that accepts arbitrary pricing algorithms as black-box oracles, pits them against each other in controlled market simulations, and produces machine-checkable evidence with quantified statistical confidence. Competition economists gain certified error bounds replacing ad-hoc index comparisons. CS researchers gain a new problem domain—composite hypothesis testing where the null is parameterized by games x learning algorithms—generating follow-up work at the intersection of computational game theory and mathematical statistics. The contribution opens a new direction at the intersection of computational game theory and formal verification.

---

## Value Proposition

| Stakeholder | Need | What CollusionProof Provides |
|---|---|---|
| **Antitrust regulators** (EC DG-COMP, FTC, DOJ) | Formal evidence for algorithmic pricing investigations | Machine-checkable collusion certificates with quantified statistical confidence; independently verifiable by any party; three oracle tiers from passive observation to sandbox audit |
| **Competition economists** | Rigorous measurement of algorithmic collusion | Continuous Collusion Premium with bootstrap confidence intervals, tiered null hypothesis testing (narrow/medium/broad), and sensitivity analysis across demand specifications |
| **Algorithm auditors** | Black-box testing methodology for proprietary pricing systems | Sandboxed execution framework supporting three levels of access: passive price data, periodic checkpoints, and full algorithm replay |
| **CS/GT researchers** | New formal problems at the intersection of verification and game theory | Novel composite hypothesis testing framework, certified deviation oracles, proof-carrying economic certificates, C3 conjecture as open problem |

**Why now**: The EU DMA entered full enforcement in March 2024. The DOJ's RealPage case (filed 2024) is the first federal antitrust action targeting algorithmic pricing coordination. Assad et al.'s empirical evidence of margin inflation in algorithmic duopolies provides economic urgency. Every major competition authority seeks algorithmic pricing audit tools, but no formal framework exists. CollusionProof targets the regulatory sandbox paradigm: algorithmic audit in controlled environments, not courtroom enforcement today. The window for defining the evidentiary standard is open—before ad-hoc approaches calcify into precedent.

---

## Technical Difficulty

### Hard Subproblems

1. **Composite null hypothesis over infinite-dimensional nuisance parameters**: The competitive null H_0 is indexed by *all* Lipschitz demand systems x *all* independent learning algorithms—an infinite-dimensional parameter space. The tiered null hierarchy (H_0-narrow, H_0-medium, H_0-broad) provides tractability: H_0-narrow restricts to linear demand x Q-learning, H_0-medium to parametric demand x no-regret learners, H_0-broad to the full Lipschitz family. Proving distribution-freeness of test statistics over each tier requires bounding the maximum achievable cross-firm correlation under any demand system in that tier, a novel optimization-over-function-spaces argument. Distribution-freeness holds exactly for permutation-based sub-tests and asymptotically for the composite correlation tests; finite-sample corrections via Berry-Esseen bounds handle the gap.

2. **Black-box deviation oracle with selection-bias-free adaptive sampling** (Layer 1-2 only): Computing certified deviation bounds from black-box algorithm queries with logarithmic (not linear) dependence on price-grid size, while proving the Lipschitz-aware adaptive refinement does not introduce selection bias via a peeling argument over resolution levels. Layer 1 requires periodic state checkpoints (not arbitrary prefixes), yielding wider confidence intervals; Layer 2 assumes full rewind capability for tight bounds.

3. **Folk Theorem converse for bounded-recall strategies (Conjecture C3)**: Proving that sustained supra-competitive pricing with recall M *necessarily* produces punishment responses detectable within M rounds—connecting the economic definition of collusion to statistical detectability. This is an open conjecture, believed true from standard repeated-game theory but not yet formally proven for finite-state pricing algorithms. We commit to proving C3 unconditionally for: (a) grim-trigger strategies, (b) tit-for-tat with bounded memory, and (c) any deterministic automaton strategy with at most M states. These cover the practically relevant cases. The general conjecture remains open; if false for exotic strategies, completeness degrades but soundness is unaffected.

4. **Numerical-to-formal bridge**: Simulations run in f64; certificates require exact verification. All ordinal comparisons used in proof derivations must be re-verified in rational arithmetic, with formal encoding of f64-to-rational conversion error bounds.

5. **Proof checker axiom soundness**: Designing a small (~15 axiom schemas, ~25 inference rules), provably sound axiom system for game-theoretic certificates that is expressive enough to certify real collusion scenarios while keeping the trusted kernel under 2,500 LoC.

6. **PPAD avoidance for the critical path**: The Layer 0 statistical testing layer must function entirely without equilibrium computation (PPAD-complete for N > 2). Nash equilibrium computation is needed only for the Collusion Premium benchmark (M5, Layer 1-2), not for the composite test's accept/reject decision. For structured games (Bertrand/Cournot with n <= 4), analytical equilibria exist.

7. **Scalable counterfactual simulation** (Layer 1-2 only): For N players x D deviations x S Monte Carlo seeds x T rounds, cost is O(N*D*S*T). With N=3, D=20, S=100, T=100K: ~600M round-steps. Must execute at >50K rounds/sec with truncated horizons, deviation pruning, and embarrassingly parallel decomposition.

8. **Bertrand CP for zero-profit equilibria**: For homogeneous Bertrand markets with zero-profit competitive equilibria, the relative Collusion Premium CP is undefined. Defining an absolute-margin deviation metric delta_p that smoothly handles this boundary case while remaining comparable to the relative CP in positive-profit settings.

### Subsystem Breakdown

#### CollusionProof-Lite (Core): ~60K LoC

The paper artifact. Implements all mathematical contributions for 2-player Bertrand/Cournot markets with tabular RL algorithms. Produces end-to-end collusion certificates.

| # | Subsystem | Language | LoC | Scope |
|---|-----------|----------|-----|-------|
| S1 | **Game Simulation Engine** | Rust | ~15,000 | Two market models (Bertrand, Cournot) x linear demand x 2-player orchestration with synchronous timing. Hot loop sustains >100K rounds/sec. |
| S2 | **Black-Box Algorithm Interface** | Rust + Python | ~10,000 | Sandboxed execution, 3 core algorithm implementations (Q-learning, grim trigger, DQN), PyO3 bindings with GIL management. |
| S3 | **Equilibrium Computation Engine** | Rust | ~5,000 | Analytical Bertrand/Cournot solvers only. No general-purpose solver required for 2-player structured games. |
| S4 | **Counterfactual Deviation Analysis** | Rust | ~12,000 | Core research subsystem. Deviation strategy enumeration, counterfactual re-simulation, statistical punishment detection (M3), deviation oracle (M2). ~12K LoC if Layer 1+ available; minimal stub if Layer 0 only. |
| S5 | **Statistical Testing & Collusion Premium** | Rust + Python | ~10,000 | M1 composite hypothesis test battery, M7 directed closed testing, Collusion Premium computation (M5) with bootstrap CIs, tiered null hierarchy (narrow/medium/broad). |
| S6 | **Certificate DSL & Proof Checker** | Rust | ~5,000 | Certificate language (parser/AST), proof-term language with ~15 axiom schemas, auditable checker kernel (at most 2,500 LoC trusted core), rational arithmetic verification. Implements M6. |
| S7 | **Evidence Bundle Pipeline** | Rust + Python | ~3,000 | Protobuf schema, Merkle-tree integrity, standalone bundle verifier. |
| S9 | **Evaluation Framework** | Python | ~7,000 | 15 ground-truth scenarios (--standard mode), automated metrics pipeline, sensitivity analysis. |
| S10 | **CLI & Orchestration** | Rust + Python | ~3,000 | Minimal CLI, experiment configuration, multiprocessing with checkpointing. |
| | **Core Total** | Rust ~42K / Python ~18K | **~60,000** | |

**Honest breakdown**: ~55K novel research code + ~5K essential infrastructure. Every subsystem serves a distinct pipeline function; removing any breaks the certification chain.

#### CollusionProof-Full (Extended): ~110-130K LoC

Generalizes the core to N-player markets, deep RL algorithms, three market models, and full evaluation. Adds:

| Extension | Additional LoC | What It Adds |
|-----------|---------------|--------------|
| N-player generalization (S1, S3, S4) | ~20,000 | 3-4 player support, Cournot/posted-price additional market models, CES/logit demand |
| Deep RL algorithms (S2) | ~11,000 | DQN, PPO, SARSA, bandits, Edgeworth, tit-for-tat; sandboxed GPU-optional training |
| General equilibrium solvers (S3) | ~12,000 | Support enumeration, Lemke-Howson, iterated best response for N > 2 |
| Full counterfactual analysis (S4) | ~8,000 | Multi-period deviation via dynamic programming, importance sampling |
| Report generator (S8) | ~10,000 | LaTeX/HTML reports, 4 visualization types, template-based NLG for regulatory summaries |
| Full evaluation (S9) | ~10,000 | 30 scenarios, adversarial red-team, parametric sweeps, Latin hypercube |
| Extended infrastructure (S7, S10) | ~7,000 | Full evidence bundles, experiment DSL, structured logging |
| **Extended Total** | **~78,000** | |
| **Grand Total (Core + Extended)** | **~138,000** | |

The original 168K estimate was inflated. Honest breakdown: ~55K core novel research code + ~55-75K essential infrastructure and generalization + ~15-23K support/evaluation.

**Why the Core/Extended split**: The Core (~60K) is the paper artifact—it demonstrates every mathematical contribution (M1-M7) on the practically most important case (2-player Bertrand/Cournot with tabular RL). The Extended adds generality and evaluation depth for a full system paper. The Rust/Python split is architecturally motivated: Rust for the performance-critical simulation loop (>100K rounds/sec) and the trusted proof-checker kernel; Python for the statistical/ML ecosystem and RL algorithm library.

---

## New Mathematics Required

| ID | Name | Description | Novelty | Role in Artifact | Layer |
|---|---|---|---|---|---|
| **M1** | Composite Hypothesis Test over Game-Algorithm Pairs | Defines the competitive null family H_0 as all trajectory distributions induced by any Lipschitz demand system paired with any tuple of independent learners, organized into a tiered null hierarchy: H_0-narrow (linear demand x Q-learning), H_0-medium (parametric demand x no-regret learners), H_0-broad (full Lipschitz family). Proves alpha-sound testing in O(T*n^2*K) time with Type-I error control *uniform* over infinite-dimensional nuisance parameters in each tier. Distribution-free Type-I error control holds exactly for permutation-based sub-tests and asymptotically for composite correlation tests; finite-sample correction terms via Berry-Esseen bounds bridge the gap, with a fallback parametric sub-family providing exact distribution-freeness for all T. Certificates report rejection at each tier. | **Grade A** (deep novel) | Logical backbone of every verdict. Determines what "competitive" means formally. Primary Layer 0 contribution. | 0 |
| **M2** | Black-Box Deviation Oracle with Certified Bounds | Adaptive procedure computing (epsilon,alpha)-correct deviation bound certificates with query complexity O(n*polylog(abs(P_delta))*log(n/alpha)/epsilon^2)—logarithmic in grid size via Lipschitz-aware coarse-to-fine refinement. Peeling argument proves selection-bias-freeness. Layer 1 variant uses periodic state checkpoints (not arbitrary prefixes), yielding wider confidence intervals proportional to checkpoint spacing. Layer 2 variant uses full rewind for tight bounds. Requires known or estimated Lipschitz constant L_D. | **Grade B** (significant novel) | Engine for counterfactual analysis (S4). Called O(n*K) times per certificate construction. | 1-2 |
| **M3** | Punishment Detection via Controlled Perturbation | First provably powerful statistical test for punishment detection against bounded-recall strategies: J = O(sigma^2*log(1/beta)/Delta_P^2) injections suffice. Permutation framework provides distribution-free p-values under the competitive null—exact Type-I error control for any sample size. Injection-based testing must run on separate trajectory segments from passive M1 sub-tests to preserve validity. | **Grade B** (significant novel) | Active testing component (S4). Produces the "we deviated and observed retaliation" evidence most directly visible to regulators. | 2 |
| **M4** | Completeness of the Hybrid Certifier | **Conditional on Conjecture C3 (Folk Theorem converse for bounded-recall strategies)***: the hybrid certifier detects eta-collusion with probability >= 1-beta from T* = O_tilde(n^2*M^2*sigma^2/(eta^2*alpha*beta) + n*polylog(abs(P_delta))/epsilon^2) rounds. The 1/(alpha*beta) dependence reflects a union-bound upper bound, likely not tight. **Unconditionally for restricted strategy classes (deterministic automata with at most M states)**: C3 holds and completeness follows. Specifically, we prove C3 for: (a) grim-trigger strategies, (b) tit-for-tat with bounded memory, and (c) any deterministic automaton strategy with at most M states. These cover the practically relevant cases. If C3 is false for some exotic strategy class, undetectable "stealth collusion" strategies may exist for that class—a fundamental barrier—but soundness is entirely unaffected and the system functions coherently. | **Grade A*** (deep novel, C3-conditional) | Tells users how much simulation to run (S4-S5). Connects economic collusion definition to statistical detectability. | 0-2 |
| **M5** | Certified Collusion Premium with Error Propagation | End-to-end error bound composing demand estimation error, NE approximation tolerance, and finite-sample averaging into a certified CP = delta +/- epsilon_CP. For zero-profit competitive equilibria (e.g., homogeneous Bertrand), the system reports an absolute supra-competitive margin delta_p rather than the relative Collusion Premium CP, avoiding undefined quantities. The absolute-margin metric smoothly reduces to the relative CP when competitive profits are strictly positive. Layer 1 provides wider CIs (checkpoint granularity); Layer 2 provides tight bounds. Demand estimation complexity is O(abs(P_delta)^n), manageable for n <= 4 with structural demand assumptions. | **Grade C** (novel combination) | The headline number in every certificate (S5). Most immediately relevant to regulators. | 1-2 |
| **M6** | Certificate Format & Verification (Polynomial for Fixed n) | Defines the proof-carrying collusion certificate tuple and a deterministic verifier running in O(T*n + K*n*abs(P_delta) + K_NE*abs(P_delta)^n). Polynomial in (T, abs(P_delta), K) for any fixed n; the abs(P_delta)^n term is exponential in the number of firms. Soundness is probabilistic (probability >= 1-alpha over original query randomness), not absolute. Each layer produces certificates at its own evidential level; the verifier validates any layer's certificates identically. | **Grade C** (novel combination) | The output format (S6-S7). What regulators, auditors, and researchers receive and independently verify. | 0-2 |
| **M7** | Directed Closed Testing for Collusion Signatures | Collusion-structured rejection ordering (supra-competitive pricing -> punishment -> correlation -> convergence) that improves power against typical collusion alternatives while maintaining FWER control under arbitrary dependence via standard Holm-Bonferroni. Power improvement is qualitatively significant for collusion alternatives exhibiting all four signatures. | **Grade D** (engineering application) | Determines test evaluation order (S5). Detects collusion faster in common cases. Primary Layer 0 contribution alongside M1. | 0 |

**Crown jewels**: **M1** (composite null over games x algorithms—a genuinely new *problem instance* in statistics, the first where the null family's structure is game-theoretic, though the technique of composite testing over infinite-dimensional nuisance parameters exists in semiparametric statistics) and **M4** (finite-sample certifiability of black-box collusion, conditional on C3, with unconditional completeness for restricted strategy classes including all deterministic bounded-recall automata). Together, these establish that algorithmic collusion is *formally certifiable*—a non-obvious statement given the infinite-dimensional null and game-theoretic nature of the alternative—subject to the bounded-recall restriction and (for general strategies) C3. The contribution is at the intersection of adjacent theoretical traditions: Horner et al. on repeated games with imperfect monitoring, Abreu (1988) on repeated-game folk theorems, Harrington (2008) on econometric collusion screening, Klein (2021) and Johnson et al. (2023) on algorithmic pricing simulation. None individually produces certificates; CollusionProof unifies them into a certification framework.

---

## Best Paper Argument

CollusionProof is a strong candidate for a best paper award at **EC** (Economics and Computation, primary venue) or **AAMAS** (secondary venue) because it simultaneously:

**Introduces a new mathematical problem instance.** The composite hypothesis test where H_0 is parameterized by demand systems x learning algorithms (M1) is a genuinely new *problem instance* in statistics—the first where the null family's structure is game-theoretic. While composite testing over infinite-dimensional nuisance parameters exists in semiparametric statistics (e.g., Andrews & Shi, 2013), no existing framework tests against a null family parameterized by games and learning algorithms. The tiered null hierarchy (H_0-narrow, H_0-medium, H_0-broad) provides a principled approach to the power-versus-generality tradeoff inherent in such composite testing. This formulation will generate follow-up work in both computational game theory and mathematical statistics.

**Creates a new category of artifact.** Proof-carrying certificates for economic properties bring the PCC paradigm (Necula, 1997) and certifying-algorithms paradigm (McConnell et al., 2011) into an entirely new domain. The prior art audit confirms **no precedent**: the concept of a machine-checkable collusion certificate does not exist in any community (formal methods, game theory, antitrust, legal tech). This is a paradigm-level contribution, not an incremental improvement.

**Occupies an uncontested triple intersection.** Formal verification x game theory x competition law has no known active research groups producing tools. The closest work—PrimeNash (analytical equilibrium derivation with machine-checkable proofs for described games), PRISM-games (model-based verification), Calvano et al. (simulation without certification), EGTA (empirical game model extraction without collusion formalization), Harrington (econometric screening without certificates), Klein/Johnson (simulation without formal certification)—each covers at most one edge of the triangle. CollusionProof is the first system at the center.

**Directly addresses the most urgent policy question in digital market regulation.** The EU DMA, the DOJ RealPage case, and Assad et al.'s empirical evidence of harm all converge on one need: formal tools for auditing algorithmic pricing. CollusionProof operates as a regulatory sandbox audit tool—certifying algorithm behavior under controlled conditions—which is precisely the paradigm competition authorities are building toward.

**Clear soundness/completeness separation.** The unconditional soundness guarantee (no false positives regardless of C3) combined with conditional completeness (with unconditional completeness for practically relevant strategy classes) is a mature, honest theoretical contribution. The explicit identification of C3 as an open conjecture—with proofs for restricted cases—gives the community a concrete research direction.

**Paper structure**: Theory contribution (M1 + conditional M4 + M6) demonstrated through a working system producing verifiable certificates. The paper leads with Layer 0 (fully passive, no oracle access) as the primary contribution, with Layers 1-2 adding evidential power in more controlled settings.

**Comparison to prior best papers.** EC best papers have rewarded new equilibrium concepts (Roughgarden's smoothness), new mechanism design paradigms, and new computational results for markets. CollusionProof offers a new *application domain* for formal methods with immediate practical impact, backed by novel mathematics—opening a new direction at the intersection of computational game theory and formal verification rather than contributing incrementally to an existing one.

---

## Evaluation Plan

### Fully Automated — No Human Annotation

All evaluation is automated end-to-end. No human annotation, no human studies, no subjective judgments.

### Three-Tier Evaluation Budget

Development and evaluation use a three-tier compute profile:

| Mode | Scenarios | Algorithms | Rounds | Seeds | CPU-Hours | Wall-Clock (8 cores) | Purpose |
|---|---|---|---|---|---|---|---|
| **`--smoke`** | 5 (2 collusive, 2 competitive, 1 boundary) | Tabular RL only | 100K | 2 | < 0.5 | **< 30 minutes** | CI and development iteration |
| **`--standard`** | 15 | Tabular + bandit, 2-3 players | 1M | 5 | ~800 | **~4 days** | Milestone validation and paper review |
| **`--full`** | 30 | All algorithms incl. deep RL | 10M | 10 | ~3,800 | **~20 days** | Camera-ready evaluation only |

Development uses `--smoke` exclusively. Development iteration: < 30 minutes (smoke mode). Paper-review evaluation: ~800 CPU-hours (~4 days on 8 cores). Camera-ready full evaluation: ~3,800 CPU-hours (~20 days on 8 cores).

### Ground-Truth Benchmark Suite

**30 scenarios** (full mode) spanning known-collusive, known-competitive, boundary, and adversarial configurations:

| Category | Scenarios | Ground Truth | Key Algorithms |
|---|---|---|---|
| **Known-collusive** (10) | Grim-trigger pairs, tit-for-tat with punishment, Q-learning replicating Calvano et al., DQN collusion | CP > threshold, punishment detected | Q-learning, SARSA, DQN, grim trigger |
| **Known-competitive** (8) | Static Nash play, myopic best response, independent bandits, competitive Bertrand equilibrium | CP ~ 0, no punishment | Myopic BR, epsilon-greedy bandit, competitive strategies |
| **Boundary/hard** (8) | Partial collusion, asymmetric players, mixed strategies, Edgeworth cycles, noisy punishment | Variable — tests system discrimination | Mixed algorithm pairs |
| **Adversarial red-team** (4) | Strategies specifically designed to evade M1 detection: collusion with injected noise to mask correlation, punishment strategies with randomized timing, strategies mimicking competitive correlation structure while sustaining supra-competitive prices, gradual collusion emergence below detection threshold | Known-collusive but designed to evade | Custom adversarial implementations |

**Operational collusion definition**: prices exceed static Nash equilibrium AND unilateral deviation to Nash price is deterred by punishment. Multiple alternative operationalizations pre-computed for robustness.

### Parametric Sweeps

Each of 5 anchor scenarios (2 collusive, 2 competitive, 1 boundary) is run with 20 parameter variations (discount factor, demand elasticity, learning rate, exploration rate, memory length), yielding 100+ evaluation points for ROC analysis. Parametric sweeps validate that detection power degrades gracefully with parameter perturbation and provide continuous ROC curves rather than point estimates.

### Tiered Null Hypothesis Evaluation

Certificates report rejection at each null tier:
- **H_0-narrow**: Lipschitz-bounded linear demand x independent Q-learning. Highest power, most restrictive.
- **H_0-medium**: Lipschitz-bounded parametric demand (CES/logit) x independent no-regret learners. Moderate power.
- **H_0-broad**: Lipschitz demand x independent learners (full M1 null). Lowest power, broadest coverage.

Evaluation reports rejection rates at each tier separately, enabling analysis of the power-generality tradeoff.

### Metrics

- **Classification accuracy**: Precision, recall, F1 for the COLLUSIVE/COMPETITIVE/INCONCLUSIVE trichotomy against ground truth labels
- **CP calibration**: Does the Collusion Premium correctly rank scenarios by known collusion severity?
- **ROC analysis**: Area under the ROC curve for CP threshold sweeps, computed from the 100+ parametric sweep evaluation points
- **Type-I error empirical validation**: Verify <= alpha false positive rate on the 8 known-competitive scenarios across all seeds, separately for each null tier
- **Statistical power**: Fraction of known-collusive scenarios correctly certified at significance alpha = 0.05, separately for each null tier
- **Adversarial evasion rate**: Fraction of adversarial red-team scenarios that successfully evade detection at each null tier
- **Certificate verification rate**: Fraction of produced certificates that pass independent verification
- **Layer comparison**: For scenarios tested at multiple oracle access levels, compare certificate strength across Layers 0, 1, and 2

### Baselines

| Baseline | What It Represents |
|---|---|
| **Price correlation screen** | Standard regulatory screening (Pearson/Spearman on price time series) |
| **Variance screen** | Low-variance-as-collusion heuristic used by competition authorities |
| **Granger causality test** | Time-series causal screening |
| **Gambit equilibrium check** | Compute NE via Gambit; compare observed prices to NE (no certification, no counterfactual) |
| **Calvano-style simulation comparison** | Reproduce Calvano et al.'s methodology: compare to pre-computed equilibrium benchmarks |

### Statistical Validation

- **Bootstrap confidence intervals** on all aggregate metrics (1,000 bootstrap samples)
- **Multiple comparison correction** (Benjamini-Hochberg) across scenario families
- **Power analysis** with effect size estimation to validate that 30 scenarios x 10 seeds provides sufficient statistical power
- **Sensitivity analysis** via Latin hypercube sampling over: round count (100K-10M), discount factor (0.9-0.999), player count (2-3 for core, 2-4 for extended), demand parameters, algorithm hyperparameters

---

## Laptop CPU Feasibility

### Target Hardware

Standard developer laptop: **8-core CPU, 32GB RAM, no GPU**. All computation is CPU-bound.

### Why GPU Is Not Needed

CollusionProof's critical path—game simulation, statistical testing, proof checking—is entirely CPU-bound arithmetic. Deep RL *training* (DQN, PPO) runs on CPU via PyTorch; slow (~2-6 hours per agent) but embarrassingly parallel and only required during evaluation (extended mode), not certification.

### Three-Tier Compute Profile

| Tier | Wall-Clock (8 cores) | Peak RAM | Use Case |
|---|---|---|---|
| **`--smoke`** | < 30 minutes | 2 GB | Development iteration, CI |
| **`--standard`** | ~4 days | 4 GB | Paper review, milestone validation |
| **`--full`** | ~20 days | 8 GB | Camera-ready only |

### Per-Scenario Compute Profile (Core / 2-Player)

| Operation | Time | Peak RAM |
|---|---|---|
| 2-player Bertrand simulation (1M rounds) | 10 sec | 50 MB |
| 2-player Cournot simulation (1M rounds) | 15 sec | 60 MB |
| Q-learning training (2-player, 10M rounds) | 3 min | 200 MB |
| DQN training (2-player, 10M rounds) | 2-4 hrs | 2 GB |
| Analytical equilibrium (Bertrand/Cournot) | < 1 sec | 10 MB |
| Counterfactual analysis (2-player, Layer 1) | 30-60 min | 500 MB |
| Counterfactual analysis (2-player, Layer 2) | 1-2 hrs | 1 GB |
| Certificate generation + rational verification | 1-5 min | 500 MB |
| Certificate verification (checker only) | < 10 sec | 50 MB |

### Per-Scenario Compute Profile (Extended / N-Player)

| Operation | Time | Peak RAM |
|---|---|---|
| 3-player Cournot simulation (1M rounds) | 25 sec | 80 MB |
| Equilibrium computation (3-player, 100 actions) | 2-10 min | 500 MB |
| Counterfactual analysis (3-player, Layer 2) | 4-8 hrs | 2 GB |
| PPO training (3-player, 10M rounds) | 4-8 hrs | 4 GB |

### End-to-End Certification Time

| Scenario | Layer | Wall-Clock (8 cores) | Peak RAM |
|---|---|---|---|
| Tabular RL, 2-player Bertrand, Layer 0 | 0 | **15-30 minutes** | 1 GB |
| Tabular RL, 2-player Bertrand, Layer 2 | 2 | **1.5-3 hours** | 2 GB |
| Tabular RL, 3-player Cournot, Layer 2 | 2 | 5-10 hours | 3 GB |
| DQN, 2-player Bertrand, Layer 2 | 2 | 4-8 hours | 4 GB |

### What's Expensive and How It's Managed

The computational bottleneck is **counterfactual deviation analysis** (S4, Layers 1-2): O(N*D*S*T) round-steps. Managed via: (1) truncated horizons (100K-round windows), (2) deviation pruning, (3) importance sampling on high-variance regions, (4) embarrassingly parallel decomposition across (player, deviation, seed) triples. Peak concurrent memory: **<= 8 GB**, well within 32 GB limit.

Layer 0 (passive observation) is dramatically cheaper: no counterfactual simulation required. A Layer 0 certificate for a 2-player tabular RL market completes in under 30 minutes on 8 cores—suitable for interactive development.

---

## Slug

`algo-collusion-certifier`
