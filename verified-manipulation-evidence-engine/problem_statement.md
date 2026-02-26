# Formally Certified Manipulation Evidence via Causal-Bayesian Inference and Adversarial Evaluation

**Slug:** `verified-manipulation-evidence-engine`

**Amendment Status:** Revised after depth-check verification (2026-02-25). Changes address value grounding, theorem reclassification, treewidth justification, RL scoping, memory/runtime honesty, evaluation calibration, and venue targeting.

---

## Core Description

Market surveillance systems detect anomalies but do not produce evidence. Exchange surveillance desks generate thousands of alerts daily—flagging suspicious order patterns, unusual cancellation ratios, transient price dislocations—yet the gap between an alert and an actionable evidentiary artifact remains vast. Analysts manually cross-reference alerts against regulatory definitions, applying subjective judgment at every stage. The bottleneck is not detection sensitivity. It is that no existing system transforms a statistical anomaly into a machine-verifiable proof artifact that formally certifies: (1) a causal mechanism linking observed order-flow patterns to manipulative intent under stated identifiability assumptions, (2) quantified evidence strength via exact Bayesian posterior probability over latent intent models, and (3) a verified compliance violation against temporal-logic regulatory specifications.

This project constructs a **Verified Manipulation Evidence Engine (VMEE)**—a CPU-native pipeline integrating three core capabilities and one stress-testing layer, never jointly realized in a single system. First, a **causal discovery and identification engine** learns DAG structures over order-flow microstructure variables and applies do-calculus to identify causal effects of trader actions on price, distinguishing manipulation from legitimate market-making under stated faithfulness and sufficiency assumptions. Second, an **exact Bayesian inference engine** compiles the identified causal model into arithmetic circuits (exploiting bounded-treewidth decompositions of the market's factor graph) to compute exact posterior probabilities over latent manipulation-intent variables—eliminating the approximation error that renders MCMC and variational posteriors unsuitable for evidentiary claims. Third, a **temporal logic monitoring and SMT proof generation layer** encodes regulatory compliance specifications in first-order metric temporal logic (FO-MTL), reduces quantitative monitoring queries to quantifier-free linear real arithmetic (QF_LRA), and produces independently checkable SMT proofs (via Z3 and CVC5) that a detected pattern violates the specification. As a stress-testing layer, an **adversarial reinforcement learning probe** trains manipulation agents against the detection pipeline in a limit-order-book simulator to discover evasion strategies and surface detection blind spots—functioning as a systematic stress test, not a comprehensive robustness guarantee.

The primary output is an **evidence bundle**: a self-contained proof artifact comprising a causal subgraph with identified effects, exact posterior probabilities over intent variables with Bayes factors, and temporal logic violation proofs checkable by any SMT solver. When adversarial stress-testing has been performed, the bundle includes a coverage annotation indicating which RL-discovered strategy classes the detected pattern was tested against, along with a strategy-space coverage estimate. Each component is individually verifiable; the Soundness Framework (M5.3) establishes that their composition is sound under explicit measure-theoretic compatibility conditions. Evidence bundles are machine-verifiable end-to-end: no human judgment is required at any stage of generation or validation.

The integration is more than the sum of its parts. Causal identification without exact inference yields directional claims without calibrated evidence strength. Exact inference without causal structure conflates correlation with manipulation. Formal verification without causal-Bayesian grounding certifies violations that may have innocent explanations. Only the three-way core integration produces artifacts that are simultaneously causally grounded, probabilistically calibrated, and formally certified. The adversarial stress-testing layer adds systematic (but not exhaustive) coverage testing. No existing system—academic or commercial—achieves even two of the core three properties jointly.

**Scope limitations.** VMEE is evidence infrastructure, not a prosecution engine, regulatory submission system, or legal evidence package. Causal claims hold only under stated identifiability assumptions (faithfulness, causal sufficiency, no unmodeled latent confounders). Bayesian posteriors are exact given the model but conditional on model correctness. RL adversarial stress-testing covers the strategy space reachable within compute budget; it does not guarantee exhaustive coverage of all evasion strategies and reports an explicit strategy-space coverage bound. The system targets three canonical manipulation types (spoofing, layering, wash trading) in equity limit-order-book markets. All evaluation uses synthetic data calibrated against published empirical LOB statistics; real-market deployment claims require further validation.

---

## Value Proposition

**Who needs this.** Exchange surveillance teams review thousands of alerts daily, with senior analysts manually adjudicating which constitute genuine manipulation. Regulatory bodies (SEC Market Abuse Unit, FCA, ESMA) require evidence meeting evidentiary standards but receive statistical scores demanding re-analysis. Quantitative trading firms need defensive tools demonstrating their strategies do not constitute manipulation—certified non-violation proofs are as valuable as violation proofs.

**Scale of the problem.** SEC enforcement actions for market manipulation averaged $1.2B in penalties annually (2019–2023). FINRA's surveillance teams process approximately 500K alerts/day across U.S. equities alone, with >99% dismissed after manual review—an estimated $200M+ annual cost in analyst time. EU MAR Article 16 mandates that trading venues maintain "effective arrangements, systems and procedures" for detecting manipulation, with ESMA explicitly requiring systems that can "reconstruct the order book" and "identify patterns." VMEE's evidence bundles directly address the reconstruction and pattern-certification requirements of MAR RTS 25, even though legal admissibility remains out of scope.

**What becomes possible.** VMEE fills a specific capability gap: the transition from detection (alerts requiring human judgment) to certification (machine-verifiable evidence artifacts). A surveillance system can output a self-contained artifact that any independent party verifies by running an SMT solver, inspecting causal identification conditions, and checking Bayesian computation traces. This eliminates the bottleneck of human expert review for evidence assessment—not for legal judgment, which remains a human responsibility. Adversarial stress-testing provides systematic discovery of detection blind spots within explored strategy-space regions, replacing ad hoc red-teaming with structured RL-driven strategy search.

**What this is not — and what it enables.** VMEE does not replace human legal judgment, regulatory expertise, or prosecutorial discretion. It does not produce legally admissible evidence. It produces formally certified, machine-verifiable evidence artifacts whose properties are mathematically established under stated assumptions. However, these artifacts are **precisely the technical substrate** that a regulatory submission system would require—the gap between VMEE output and a regulatory filing is jurisdictional and procedural, not technical. Whether such artifacts meet any jurisdiction's evidentiary standards is a legal question outside scope.

**Comparison to alternative approaches.** LLM-based surveillance tools can generate natural-language case narratives and cross-reference regulatory definitions, providing useful analyst-facing workflows. However, LLM outputs are not machine-verifiable, cannot provide exact posterior probabilities with formal guarantees, and cannot produce independently checkable proofs of temporal-logic violations. VMEE occupies a complementary niche: where LLMs provide interpretable explanations for analyst workflows, VMEE provides formally verifiable evidence artifacts for automated compliance checking and audit trails. The two approaches are complementary, not competing.

---

## Technical Difficulty

### Five Hard Subproblems (Revised from Seven)

1. **Exact inference at scale via arithmetic circuit compilation.** Compiling Bayesian networks with 50–200 nodes (order-flow microstructure models) into arithmetic circuits supporting exact marginal and MAP queries. Requires exploiting bounded treewidth via tree decomposition, and engineering efficient evaluation over circuits with 10⁶–10⁷ edges on CPU. **Treewidth justification:** LOB microstructure models over canonical order-flow features (bid-ask spread, queue imbalance, order arrival rate, cancellation rate, trade-through rate) exhibit sparse conditional independence structure: each feature depends on a bounded neighborhood of temporally and structurally adjacent features. Preliminary analysis of representative LOB factor graphs over 50–100 variables yields treewidth 8–14, consistent with the sparse structure observed in time-series factor graphs (Koller & Friedman 2009, §9.4). We set a treewidth bound of 15 and define explicit graceful degradation: if treewidth exceeds 15 for a given model, the system falls back to bounded-cutset conditioning (exact inference over a reduced model) with a documented approximation bound, preserving the probabilistic-to-logical proof bridge for the conditioned submodel.

2. **Probabilistic → logical proof bridge (CORE CONTRIBUTION).** Translating Bayesian posterior probabilities and Bayes factors into SMT-checkable proof obligations. The encoding must faithfully represent probabilistic semantics—not merely threshold the posterior—and produce independently verifiable proofs. Requires a formal reduction from quantitative evidence claims to QF_LRA satisfiability with an equisatisfiability guarantee. This is the system's most novel capability: no existing tool or publication provides a verified reduction from Bayesian evidence to SMT-checkable proofs.

3. **Causal discovery under non-stationarity in financial time series.** Financial microstructure exhibits regime shifts (volatility clustering, liquidity withdrawal) that violate stationarity assumptions of standard causal discovery algorithms. Requires windowed discovery with structural change detection and robustness guarantees (Theorem M7.4) bounding posterior degradation under DAG misspecification.

4. **First-order metric temporal logic monitoring at event-stream scale.** Monitoring FO-MTL specifications over high-frequency event streams (10⁴–10⁵ events per trading day per instrument). Requires an incremental monitoring algorithm processing events in O(1) amortized time per event per formula, with bounded memory proportional to the specification's temporal horizon. Extends prior qualitative monitoring (MonPoly, DejaVu) to the quantitative setting required for financial signal monitoring.

5. **Heterogeneous proof composition.** Composing evidence from three fundamentally different formal systems—causal DAGs (graphical models), Bayesian posteriors (probability theory), and temporal logic violations (formal verification)—into a single evidence bundle with end-to-end soundness. The key technical difficulty is measure-theoretic consistency: causal identification operates over interventional distributions (truncated factorizations over σ-algebras generated by do-operations), Bayesian posteriors operate over parameter spaces (Euclidean with Lebesgue base measure), and temporal logic violations operate over discrete event traces (counting measure). The Soundness Framework (M5.3) constructs a product measurable space and establishes that natural projections preserve individual guarantees—a non-trivial condition when causal and Bayesian components share random variables (order-flow observables) requiring consistent marginalization.

### Supporting Engineering Subproblems

6. **CPU-feasible adversarial RL stress-testing in LOB environments.** Training manipulation agents in a limit-order-book simulator on CPU only. Requires small policy networks (<200K parameters), sample-efficient RL (PPO with domain-specific reward shaping), and a simulator generating 10⁶+ episodes within 24 hours. Scoped as a systematic stress-test discovering evasion strategies within the explored strategy space, not a comprehensive adversarial guarantee. An explicit **strategy-space coverage metric** reports what fraction of the parameterized strategy space was explored.

7. **Sim-to-real calibration.** Calibrating the synthetic LOB simulator against published empirical LOB statistics (queue length distributions, cancellation rates, inter-arrival times from LOBSTER data and published results in Huang & Polak 2011, Cont et al. 2014). Reports calibration KS-statistics between synthetic and empirical marginals for all market microstructure features used by the causal discovery engine.

### Subsystem Breakdown (~120K LoC)

| Subsystem | Est. LoC | Languages | Category |
|---|---|---|---|
| Causal Discovery & Inference Engine | 22K | Rust, Python | Core algorithmic |
| Exact Bayesian Inference Engine (Arithmetic Circuits) | 20K | Rust, Python | Core algorithmic |
| Temporal Logic Specification & Monitoring | 14K | Rust, DSL | Core algorithmic |
| SMT Proof Generation & Verification | 22K | Rust, SMT-LIB | Core algorithmic |
| Order Book Simulator & Microstructure Engine | 15K | Rust | Supporting |
| RL Adversarial Stress-Testing Environment | 12K | Python, Rust | Supporting |
| Evidence Assembly & Reporting | 6K | Python | Infrastructure |
| Evaluation & Benchmarking | 6K | Python | Infrastructure |
| Infrastructure (FFI, CLI, Config) | 3K | Rust, Python | Infrastructure |
| **Total** | **~120K** | | |
| *Core algorithmic* | *78K* | | |
| *Supporting* | *27K* | | |
| *Infrastructure* | *15K* | | |

---

## New Mathematics Required

### Tier 1: Novel Theorem

**M7.4 — DAG Robustness Under Adversarial Shift.**
Proves quantitative bounds on posterior degradation when the assumed causal DAG $G$ differs from the true data-generating DAG $G^*$. If $G$ and $G^*$ differ by at most $k$ edge additions/deletions (SHD$(G, G^*) \leq k$), then for any identified causal effect $\tau$ and its exact posterior $\pi(\tau \mid D, G)$: the total variation distance $d_{TV}(\pi(\tau \mid D, G),\ \pi(\tau \mid D, G^*)) \leq f(k, n, \kappa)$ where $n$ is sample size and $\kappa$ bounds the condition number of implied covariance matrices. **Tightness target:** the bound must be non-vacuous (< 0.5) for k ≤ 3, n ≥ 10³, κ ≤ 10². If the bound is vacuous for realistic parameters, the theorem will be reported as a theoretical result with limited practical applicability—not hidden. This bound is critical because adversarial manipulation strategies may shift the true causal structure, and the evidence engine must quantify how its guarantees degrade under such shifts. Related work: Uhler et al. (2013) on Gaussian DAG perturbation bounds, Peters & Bühlmann (2015) on structural intervention distance. M7.4 extends these to the posterior-over-effects setting with explicit sample-size dependence.

### Tier 2: Novel Formalizations

**M5.3 — Soundness Framework for Heterogeneous Evidence Composition.**
Let an evidence bundle $\mathcal{B} = (G, \pi, \varphi)$ consist of a causal subgraph $G$ with identified effects satisfying identification conditions $\mathcal{I}$, exact Bayesian posteriors $\pi$ over latent intent variables computed from an arithmetic circuit satisfying correctness specification $\mathcal{C}$, and a temporal logic violation proof $\varphi$ verified against monitoring specification $\mathcal{S}$. The framework establishes: if $G \models \mathcal{I}$, $\pi \models \mathcal{C}$, and $\varphi \models \mathcal{S}$, then $\mathcal{B} \models \Phi$ where $\Phi$ is the overall compliance claim. The proof constructs a product measurable space $(\Omega_C \times \Omega_B \times \Omega_T, \mathcal{F}_C \otimes \mathcal{F}_B \otimes \mathcal{F}_T)$ and establishes that natural projections preserve individual component guarantees—a non-trivial condition when causal and Bayesian components share random variables (order-flow observables) that must be consistently marginalized. This framework is necessary because naive conjunction of heterogeneous evidence can produce unsound bundles when components make incompatible distributional or structural assumptions. Classification: novel formalization of compositional reasoning (analogous to assume-guarantee reasoning in concurrent verification), not a deep theorem.

**M7.1 — Causal-Bayesian Composition Conditions.**
Establishes sufficient conditions under which causal identification results (interventional distributions $P(Y \mid do(X))$ identified from a DAG $G$) can be consistently composed with Bayesian posterior probabilities $P(\theta \mid D)$ computed from an arithmetic circuit encoding the observational model $P(D \mid \theta, G)$. The key technical condition is *identification-inference compatibility*: the arithmetic circuit must encode a likelihood function consistent with the truncated factorization implied by do-calculus on $G$. Under this condition, the composed evidence measure $E(Y, \theta \mid do(X), D) = P(Y \mid do(X), \theta) \cdot P(\theta \mid D)$ is a valid posterior predictive distribution over manipulated outcomes given observed data. Classification: useful formalization of consistency conditions between two well-understood frameworks, ensuring the system's causal and Bayesian components are provably compatible.

**4. SMT encoding of heterogeneous evidence (M5.3 implementation).** A formal encoding scheme representing causal identification conditions as linear arithmetic constraints, Bayesian posterior thresholds as rational inequalities, and temporal logic violations as LRA formulas, within a unified QF_LRA theory. The encoding is polynomial in the size of the evidence bundle.

**5. Quantitative MTL to QF_LRA reduction (CORE CONTRIBUTION).** A constructive reduction from quantitative metric temporal logic monitoring queries (with real-valued signals and timing constraints) to QF_LRA satisfiability, with an equisatisfiability proof. Extends prior qualitative MTL-to-SAT reductions to the quantitative setting required for financial signal monitoring. This is the second core novel formalization alongside the probabilistic→logical proof bridge.

**6. Generative intent model with manipulation-phase HMM.** A hidden Markov model of manipulation intent where latent states correspond to manipulation phases (setup → execution → withdrawal → profit-taking), emissions are order-flow observables, and the HMM structure derives from the causal DAG's temporal unrolling. Enables exact inference over manipulation phase via arithmetic circuit compilation of the HMM.

**7. Adversarial MDP with Bayesian detector belief in state space.** Formalizes the adversarial stress-testing problem as an MDP where the manipulation agent's state includes the detector's current Bayesian belief (posterior over intent). This creates a POMDP-like structure that the RL agent must solve to discover evasion strategies within the explored strategy space. Optimal evasion strategies in this MDP correspond to worst-case inputs for the evidence engine within the reachable strategy class.

---

## Best Paper Argument

**Primary venue: UAI 2026** (Causal Inference / Probabilistic Reasoning track). Secondary targets: AAAI 2026 (AI Safety and Robustness track), ACM CCS (Financial Security track).

**Core contribution framing.** The paper leads with the **probabilistic→logical proof bridge**: the first verified reduction from Bayesian evidence (exact posteriors, Bayes factors) to SMT-checkable proofs with an equisatisfiability guarantee. This is the needle that threads the narrative—it bridges the probabilistic and formal methods communities with a novel formalization that neither has produced independently.

**Integration novelty.** No prior system jointly implements causal identification, exact Bayesian inference via arithmetic circuits, and SMT-certified compliance verification. Each pair has been explored (causal + Bayesian, Bayesian + formal), but the three-way core integration creates capabilities impossible with any subset: causally grounded, probabilistically calibrated, formally certified evidence. The adversarial RL layer adds systematic stress-testing with explicit coverage bounds.

**Standalone mathematical contributions.** Theorem M7.4 (DAG robustness under adversarial shift) addresses an active gap in robust causal inference: quantitative degradation bounds under structural misspecification with explicit sample-size dependence, extending Uhler et al. (2013) and Peters & Bühlmann (2015). The Soundness Framework M5.3 and Composition Conditions M7.1, while formalizations rather than deep theorems, provide the first rigorous foundation for composing heterogeneous evidence from causal, probabilistic, and temporal-logic components.

**Evaluation narrative.** The evaluation tells a four-stage story: (1) detect and certify known manipulation patterns in calibrated synthetic markets with known ground truth, demonstrating zero false certifications under model assumptions; (2) deploy RL adversarial agents as stress-testing probes to discover evasion strategies within explored strategy-space; (3) produce certified evidence bundles for both known and RL-discovered manipulation, demonstrating generalization; (4) stress-test the full pipeline under adversarial conditions, measuring graceful degradation rather than claiming invulnerability. Calibration against published empirical LOB statistics (Cont et al. 2014) demonstrates that synthetic scenarios are not trivially distinguishable from real markets.

**Qualitative comparison to commercial systems.** VMEE is compared against published capability descriptions of Nasdaq SMARTS, NICE Actimize, and Bloomberg SSEOMS on a capability-class matrix: causal grounding (none provide), exact probabilistic calibration (none provide), formal verification with SMT proofs (none provide), adversarial robustness testing (none provide). This is a capability-class comparison, not a head-to-head performance benchmark.

---

## Evaluation Plan

All metrics are fully automated. No human annotation, subjective rating, or manual review at any stage.

1. **Soundness verification.** Generate calibrated synthetic market data with known ground-truth manipulation labels. Verify that the evidence engine produces zero false certifications: every evidence bundle certifying a manipulation violation corresponds to a true positive. Metric: false certification rate = 0 on the synthetic benchmark suite. A single false certification is a system failure. **Note:** this metric validates self-consistency under model assumptions, not real-world correctness. The synthetic data generator uses independent parameterization from the detection engine to reduce circularity.

2. **Detection coverage.** Measure the fraction of known manipulation types (spoofing with 3+ subtypes, layering with 2+ subtypes, wash trading with 2+ subtypes) correctly detected and certified across 1,000+ synthetic scenarios per type. Metric: per-type and aggregate detection rates with 95% confidence intervals.

3. **Evidence strength.** Compare Bayes factors produced by exact arithmetic circuit inference against three baselines: (a) Gibbs sampling with 10⁴ iterations, (b) mean-field variational inference, (c) belief propagation on loopy graphs. Metric: relative error in Bayes factors, calibration curves, and coverage of exact vs. approximate credible intervals.

4. **Proof validity.** Every SMT proof is independently checked by both Z3 and CVC5. Metric: proof acceptance rate by both solvers (target: 100%). Any rejected proof triggers investigation and counts as a system defect.

5. **Adversarial stress-testing.** Train RL manipulation agents for 24 hours (CPU), then measure certified detection rate against RL-discovered strategies. Metric: detection rate on novel strategies (target: ≥70% for known-type variants, with measured degradation for genuinely novel strategies). Report the strategy taxonomy discovered by RL agents and the **strategy-space coverage bound** (fraction of parameterized strategy space explored).

6. **Causal accuracy.** Compare discovered causal DAGs against ground-truth DAGs using structural Hamming distance (SHD), structural intervention distance (SID), and F1 on edge recovery. Evaluate under stationary and non-stationary conditions separately.

7. **Latency benchmarks.** End-to-end evidence generation time per case (single instrument, single trading day) on a laptop CPU (8-core, 16 GB RAM). Target: <5 minutes per case for the core pipeline (causal discovery + exact inference + monitoring + SMT proofs). Report per-subsystem breakdown.

8. **Comparison baselines.** (a) Rule-based detection (SMARTS-style threshold alerts): no formal certification, no causal grounding. (b) ML anomaly detection (isolation forest, autoencoder): probabilistic scores without causal or formal guarantees. (c) Uncertified Bayesian detection (MCMC posterior, no SMT proofs): approximate posteriors without formal evidence status. VMEE is evaluated against all three on detection rate, false positive rate, and evidence verifiability (binary: produces machine-checkable proof or does not).

9. **Calibrated synthetic markets.** Calibrate the synthetic LOB generator against publicly available empirical LOB statistics (queue length distributions, cancellation rates, inter-arrival times from LOBSTER data and published summaries in Huang & Polak 2011, Cont et al. 2014). Report calibration KS-statistics between synthetic and empirical marginals for all market microstructure features used by the causal discovery engine. This does not constitute real-data validation but demonstrates that synthetic scenarios are not trivially distinguishable from real markets.

10. **Capability-class comparison.** Qualitative comparison against Nasdaq SMARTS, NICE Actimize, and Bloomberg SSEOMS on formal capability matrix: causal grounding, probabilistic calibration, formal verification, adversarial testing. Based on published documentation and capability descriptions.

---

## Laptop CPU Feasibility

**Hardware specification.** All runtime and memory estimates target a laptop with an 8-core CPU (Intel i7/AMD Ryzen 7 or Apple M-series) and **16 GB RAM**. Peak memory of ~8 GB requires pipeline stages to execute sequentially, not in parallel.

| Component | Runtime | Memory | Notes |
|---|---|---|---|
| Synthetic data generation | ~20 min | ~2 GB | 10 instruments × 1 trading day each |
| Causal discovery (per instrument) | ~15 min | ~1 GB | Windowed PC algorithm + score-based refinement |
| Arithmetic circuit compilation | ~10 min | ~3 GB | One-time per model structure; 10⁶–10⁷ edges |
| Exact Bayesian inference (per case) | ~2 min | ~2 GB | Circuit evaluation: additions and multiplications |
| Temporal logic monitoring (per case) | ~1 min | ~0.5 GB | Incremental, bounded memory |
| SMT proof generation (per case) | ~5 min | ~2 GB | Z3 + CVC5 cross-validation |
| Evidence assembly (per case) | ~1 min | ~0.5 GB | Serialization and bundling |
| **Core pipeline (10 cases)** | **~4 hours** | **~8 GB peak** | Sequential execution; causal + Bayesian + monitoring + SMT |
| RL adversarial stress-testing | ~24 hours | ~4 GB | Offline, one-time; small MLP policies (<200K params) |
| **Total including RL** | **~28 hours** | **~8 GB peak** | RL runs separately as offline stress-test |

**Treewidth and circuit scale.** Arithmetic circuits for LOB models with treewidth 8–14 and 50–200 variables produce circuits in the 10⁶–10⁷ edge range. At 10⁷ edges with ~40 bytes per edge (weights + topology + metadata), raw circuit storage is ~400 MB, well within budget. If treewidth exceeds 15, the system falls back to bounded-cutset conditioning with documented approximation bounds.

**No GPU dependency.** All neural networks are small MLPs (2–3 hidden layers, 64–128 units, <200K parameters). Training uses CPU-native PyTorch. Arithmetic circuit evaluation is pure arithmetic over real-valued edge weights. SMT solving is inherently CPU-native. Causal discovery algorithms are combinatorial/statistical, not gradient-based.

**No human involvement.** The entire pipeline—from synthetic data generation through evidence bundle production through evaluation—runs without human input at any stage. Configuration is specified in declarative TOML files. Results are output as structured JSON with all metrics computed automatically.

---

## Explicit Non-Claims

To pre-empt predictable reviewer objections, the following are **not claimed**:

- **Not a production surveillance system.** VMEE is a research prototype demonstrating formal properties. Production deployment requires engineering beyond scope (real-time streaming, regulatory API integration, multi-asset support).
- **Not a legal evidence system.** "Evidence" refers to formal proof artifacts with mathematically established properties, not evidence in any legal or regulatory sense. Whether VMEE outputs meet any jurisdiction's evidentiary standards is a legal question this work does not address. The output is the technical substrate a regulatory system would build upon.
- **Not exhaustive adversarial coverage.** RL stress-testing covers the strategy space reachable within compute budget and reports explicit coverage bounds. It does not guarantee discovery of all possible evasion strategies. The adversarial component is a systematic stress-test, not a comprehensive robustness certificate.
- **Not assumption-free causal claims.** All causal identification is conditional on stated assumptions (faithfulness, causal sufficiency, correct adjustment sets). Violations invalidate the causal component. Theorem M7.4 quantifies degradation under DAG misspecification but does not eliminate it.
- **Not a claim of superiority over human analysts.** VMEE automates evidence generation, not judgment. It produces artifacts that analysts can verify and use but does not replace domain expertise in interpreting market context.
- **Not real-data validated.** All evaluation uses synthetic data calibrated against published empirical LOB statistics. Generalization to real market data requires further work on simulator calibration and domain adaptation. Calibration KS-statistics are reported to quantify the gap between synthetic and empirical distributions.
- **Not a replacement for LLM-based surveillance tools.** VMEE provides formally verifiable evidence artifacts; LLM-based tools provide interpretable analyst-facing narratives. The approaches are complementary, addressing different needs in the surveillance workflow.

---

`verified-manipulation-evidence-engine`
