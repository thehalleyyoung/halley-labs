# MARACE: Multi-Agent Race Analysis and Certification Engine — Final Synthesized Approach

## 1. Title and Summary

**Interaction Race Detection for Asynchronous Multi-Agent Systems via HB-Constrained Abstract Interpretation and Guided Schedule Search**

We introduce the *interaction race*, a new formal bug class for asynchronous multi-agent systems with independently deployed policies, and build MARACE, a verification tool that detects them. An interaction race occurs when two agents' actions are causally unordered (incomparable under a happens-before relation) and there exists a valid schedule permutation leading to a safety violation—a class of failure that is provably invisible to single-agent verification and exponentially unlikely to surface under random simulation. MARACE combines three analysis engines: (1) an HB-constrained zonotope abstract interpreter that soundly over-approximates reachable joint states under valid schedules, providing race-absence certificates where its fixpoint converges; (2) a guided adversarial schedule search (MCTS + CMA-ES + optional Gumbel-Softmax heuristic) that concretely finds race-triggering interleavings in regions flagged by the abstract interpreter; and (3) an importance-weighted probability estimator that bounds race occurrence frequency under deployment conditions. The system decomposes multi-agent verification compositionally via assume-guarantee contracts over interaction groups derived from the HB graph, reducing cost from exponential in the total agent count to polynomial in the interaction group size. We target domain-agnostic deployment with evaluation across traffic, warehouse robotics, and algorithmic trading benchmarks, and present the interaction race formalism and separation theorem (including a PSPACE-hardness result) as a standalone theoretical contribution.

---

## 2. Extreme Value

### Who Needs This

**Primary users:** Safety engineers and verification teams deploying multi-agent systems where independently developed policies must coexist without a central coordinator.

- **Mixed-autonomy traffic.** Vehicles from different OEMs (Tesla, Waymo, Mobileye) share roads with no shared coordinator. Each OEM verifies its own stack independently. No existing tool verifies their *composition* under asynchronous perception-to-actuation timing (20ms–200ms across platforms). The V2X standardization effort (SAE J3016) assumes cooperative behavior but provides no formal verification.

- **Decentralized warehouse robotics.** Multi-vendor robot fleets (Locus Robotics, 6 River Systems) deploy learned reactive policies on robots with heterogeneous sensor update rates, creating effective asynchrony. Corridor deadlocks and near-collisions arise from timing-dependent interleavings that no single-robot verification catches.

- **Algorithmic trading.** Multiple market-making agents from different firms submit orders to the same exchange with microsecond-scale timing dependencies. While IP constraints prevent full production deployment (firms will not share strategy checkpoints), the domain is valuable for (a) intra-firm cross-strategy verification (a single fund running 10–50 strategies through shared risk limits), and (b) exchange-level systemic risk research using stylized agent models.

### Honest Domain Grounding

We do **not** claim that exchanges or regulators will provide proprietary strategy checkpoints. The realistic near-term finance use case is *intra-firm*: a quantitative fund verifying its own strategies' interactions through shared infrastructure. The traffic and warehouse domains have no such IP barrier and are the strongest near-term deployment targets. The theoretical contribution (interaction race formalism + separation theorem) is domain-independent and stands alone.

**Current practice gap:** All three domains currently rely on simulation testing, which provides coverage estimates but no guarantees and systematically misses schedule-dependent failures (Theorem 2 quantifies this). Formal methods for multi-agent systems (PRISM-games, Storm, shield synthesis, multi-agent CBFs) either require explicit finite-state models, restrict to two-player zero-sum structure, or provide only runtime enforcement without pre-deployment diagnosis. MARACE fills the pre-deployment static/dynamic analysis gap for learned policies.

---

## 3. Architecture

MARACE is structured as a pipeline of four engines plus supporting infrastructure, designed for single-machine CPU execution.

```
┌─────────────────────────────────────────────────────────────────┐
│                        MARACE Pipeline                          │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │  Policy   │    │  HB Partial  │    │   Compositional     │   │
│  │ Ingestion │───▶│  Order       │───▶│   Decomposition     │   │
│  │ (ONNX)    │    │  Engine      │    │   (Interaction      │   │
│  └──────────┘    └──────────────┘    │    Groups + A/G)     │   │
│                                      └──────┬──────────────┘   │
│                                             │                   │
│                          ┌──────────────────┼────────────┐      │
│                          ▼                  ▼            ▼      │
│                  ┌──────────────┐  ┌─────────────┐ ┌────────┐  │
│                  │  HB-Aware    │  │  Adversarial │ │  SIS   │  │
│                  │  Abstract    │  │  Schedule    │ │Prob.   │  │
│                  │  Interpreter │  │  Search      │ │Estim.  │  │
│                  │  (Zonotopes) │  │  (MCTS/CMA)  │ │        │  │
│                  └──────┬───────┘  └──────┬──────┘ └───┬────┘  │
│                         │                 │            │        │
│                         ▼                 ▼            ▼        │
│                  ┌─────────────────────────────────────────┐    │
│                  │         Race Catalog Output              │    │
│                  │  • Absence certificates (where converged)│    │
│                  │  • Adversarial replay traces              │    │
│                  │  • Probability bounds                     │    │
│                  │  • Causal fix suggestions (min-cut)       │    │
│                  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Engine 1: HB Partial-Order Engine

Constructs the happens-before relation over multi-agent execution from three sources:
- **Observation dependencies:** Agent B observes state modified by Agent A.
- **Communication events:** Explicit messages, shared-memory writes.
- **Environment-mediated causal chains:** Agent A's action changes a physical quantity entering Agent B's observation within bounded delay.

The HB graph's connected components define **interaction groups**—sets of agents with potential causal coupling. This is the input to compositional decomposition.

### Engine 2: HB-Aware Abstract Interpreter

Performs joint reachability analysis per interaction group using zonotope abstract domains with HB-consistency constraints. The concretization maps abstract elements to sets of *HB-consistent* joint states (not arbitrary Cartesian products). The abstract transfer function prunes states violating causal ordering constraints, yielding sound over-approximation of only the reachable states under valid schedules.

- **Convergence for ReLU policies:** A formally defined widening operator guaranteeing finite ascending chains (see §4, M3).
- **Non-ReLU policies:** Sound per-iteration over-approximation without fixpoint guarantee. The abstract interpreter provides useful race-region identification even without convergence—guiding the adversarial search engine.
- **Output:** For converged fixpoints: sound race-absence certificates. For all runs: safety-margin maps identifying high-risk schedule-state regions.

### Engine 3: Adversarial Schedule Search

Searches for concrete race-triggering interleavings, guided by the abstract interpreter's safety-margin estimates. Uses three complementary search strategies:

- **k-bounded MCTS** over the schedule space with UCB1 using abstract safety margins as value estimates. AI-guided pruning reduces branching from O(N!) to O(k!).
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) as a derivative-free optimizer over the continuous schedule parameterization—robust, well-understood, no differentiable surrogate needed.
- **Gumbel-Softmax gradient search** (optional heuristic): when a differentiable environment proxy is available, gradient-based schedule optimization via Sinkhorn-parameterized doubly-stochastic matrices. Used as one search heuristic among several, not the core method.

A **verify-then-refine loop** validates all candidate races in the true (non-differentiable) simulator. Only verified races enter the catalog.

### Engine 4: Sequential Importance Sampling (SIS)

Provides statistically rigorous probability bounds on race occurrence under the deployment schedule distribution:
- Cross-entropy proposal adaptation centered on abstract-interpreter-identified race regions.
- Importance-weighted probability estimation with confidence intervals.
- Effective sample size monitoring to bound estimation quality.

### Post-Hoc Analysis: Causal Fix Suggestions

For confirmed races, construct a lightweight causal race graph from re-simulation under schedule interventions (not neural counterfactual prediction—direct re-simulation is more reliable for v1). The graph captures which timing dependencies are causally necessary for the violation. Compute minimum edge cuts to suggest minimal coordination interventions (delays, batching windows, synchronization points) that eliminate the race class. This is a *heuristic* post-hoc explainer—not a formal causal identification procedure—and does not assume faithfulness.

### Compositional Decomposition

The system partitions agents into interaction groups based on HB-graph connected components, analyzes each group independently, and composes results via assume-guarantee contracts on shared state predicates. The decomposition is **static** for a given HB graph (addressing the adaptive-vs-fixed contradiction identified in the debate). If analysis of a group reveals previously unknown causal dependencies, the HB graph is updated and decomposition is recomputed from scratch—a new analysis pass, not mid-iteration mutation.

### Subsystem Breakdown (~40K LoC, 6-month target)

| ID  | Subsystem | LoC | Language | Notes |
|-----|-----------|-----|----------|-------|
| S01 | Trace Infrastructure | 2K | Rust | HB-stamped execution traces |
| S02 | HB & Partial-Order Engine | 5K | Rust | Vector clocks, causal chain inference |
| S03 | Abstract Interpretation Core | 12K | Rust | Zonotope domains, HB-aware transfer, widening, fixpoint |
| S04 | Compositional Decomposition | 4K | Rust | Interaction graph, contracts, A/G composition |
| S05 | Adversarial Search (MCTS + CMA-ES) | 6K | Rust | Schedule-space search, AI-guided pruning |
| S06 | SIS Probability Estimator | 3K | Rust | Cross-entropy proposals, importance weighting |
| S07 | Policy Ingestion | 3K | Rust | ONNX import, ReLU abstract transformers |
| S08 | Environment Adapters | 3K | Python | PettingZoo/Gymnasium async stepping |
| S09 | Orchestration & Reporting | 2K | Python | Pipeline control, race catalog output |
| | **Total** | **~40K** | **~35K Rust + ~5K Python** | |

The 110K LoC estimate from the original Approach 1 was unrealistic for 6 months. This scope targets a working prototype with the core engines functional.

---

## 4. Load-Bearing Mathematics

We present only the mathematics essential to the system's correctness, with gaps identified in the debate explicitly addressed.

### M1. Interaction Race Formalism

**Definition.** Given an execution trace τ over agents A₁, …, Aₙ with happens-before partial order ≺_HB derived from observation dependencies, communication, and environment-mediated causal chains, an *interaction race* at joint state s is a pair of action events (eᵢ, eⱼ) from distinct agents such that:
1. eᵢ ∦ eⱼ (incomparable under ≺_HB), and
2. There exists a valid schedule permutation σ consistent with ≺_HB under which step(s, σ) ⊨ φ (safety violation).

### M2. Separation Theorem

Three results establishing that interaction races occupy a formal blind spot:

**Theorem 1 (Non-Projectability).** There exist interaction races whose projection onto every individual agent's trace satisfies that agent's local safety specification. *Proof by construction* over piecewise-linear policies on 2D continuous state spaces.

**Theorem 2 (Statistical Intractability).** For interaction races whose occurrence depends on schedule orderings within a timing window Δt, the probability that uniform random scheduling triggers the race scales as O((Δt/T)^k) where T is episode length and k is the number of involved agents—exponential in k.

**Theorem 3 (PSPACE-Hardness).** The interaction race detection problem is PSPACE-hard by reduction from timed automata reachability with N clocks, even when each individual policy is verifiable in polynomial time (deterministic, memoryless, piecewise-linear).

**Role of Theorem 3:** This result motivates the need for approximation (abstract interpretation + search) rather than exact verification. It is a complexity-theoretic ceiling, not a claim of algorithmic novelty. It establishes a genuine complexity separation: single-agent verification is in P, but the composition problem is PSPACE-hard. The reduction from timed automata reachability requires encoding clock constraints as schedule-timing constraints and is non-trivial, but we present it as motivation, not as a core algorithmic contribution.

### M3. HB-Constrained Zonotope Domain with Corrected Convergence

**Definition.** An HB-constrained zonotope is a pair (Z, C) where Z = (c, G) is a standard zonotope (center c ∈ ℝ^d, generator matrix G ∈ ℝ^{d×m}) and C = {Ax ≤ b} is a fixed conjunction of linear constraints encoding HB-consistency. The concretization is:

γ(Z, C) = {c + Gξ : ξ ∈ [-1,1]^m, A(c + Gξ) ≤ b}

**Abstract transfer function for ReLU layer** f(x) = max(Wx + b, 0): computed via DeepZ-style zonotope propagation. HB constraints C are *not* propagated through Fourier-Motzkin elimination (which was correctly identified in the debate as causing unbounded constraint growth). Instead, C is maintained as a fixed set of linear constraints over the *input-space* variables, and each abstract element carries a sound linear relaxation relating its variables back to the input-space coordinates. This avoids constraint blowup entirely: |C| is fixed throughout iteration.

**Widening operator (formally defined, addressing debate gap):**

The widening operator ∇ is defined as follows. Given two HB-constrained zonotopes (Z₁, C) ⊑ (Z₂, C) in the ascending chain:

∇((Z₁, C), (Z₂, C)) = (Z∇, C)

where Z∇ is the standard zonotope widening of Girard (generators of Z₁ that grew between iterations are extrapolated; new generators are added). The HB constraint set C is *invariant*—it is never widened, relaxed, or modified during iteration. This eliminates the non-monotonicity problem: since C is fixed and zonotope widening is monotone over the generator lattice, the combined operator is monotone.

**Convergence theorem (for ReLU networks).** For policies with ReLU activations, depth D, and width W, with fixed HB-constraint set C, the widening operator ∇ produces an ascending chain of length at most O(D · W · log(R/ε)) where R is the initial abstract diameter and ε is the widening threshold. The zonotope component stabilizes in bounded iterations because each widening step either doubles a generator coefficient (at most log(R/ε) times per generator) or adds a new generator (at most D · W across the network). Generator count is bounded by Girard's reduction (periodically reduce to 2d generators via PCA-based merging).

**Soundness without convergence.** For non-ReLU architectures, each iteration of the abstract interpreter produces a sound over-approximation of the states reachable in that many steps. The system can use any finite number of iterations for race detection (flagging regions where abstract elements intersect the violation predicate) even if the fixpoint does not converge.

### M4. ε-Race Calibration

For continuous state spaces, define an *ε-race*: two agents are in an ε-race if a safety violation is reachable under any HB-consistent schedule within an ε-ball of the current joint state. The calibration procedure:

1. Initialize ε₀ = L⁻¹ · δ₀ where L is the maximum policy Lipschitz constant and δ₀ is the coarse global safety margin.
2. Run abstract interpretation with current ε.
3. Compute refined safety margin δ₁ from the abstract interpreter's output.
4. Update ε₁ = L⁻¹ · δ₁.
5. Repeat until |εₖ₊₁ - εₖ| < tolerance.

The iteration is monotone on a complete lattice of safety margins (since enlarging ε can only increase the set of detected races, which can only decrease δ) and converges in O(log(δ₀/δ_min)) steps.

### M5. Compositional Soundness

**Theorem.** For a static partition of N agents into interaction groups G₁, …, Gₘ with interface contracts Φᵢⱼ over shared state predicates: if each group Gᵢ is race-free under its contracts, and every contract Φᵢⱼ is discharged by the abstract interpreter, then the full N-agent system is race-free.

*Proof sketch (by contraposition):* Any global race trace contains a sub-trace projecting onto some group Gᵢ that violates Gᵢ's local race-freedom, via the causal closure property of HB-graph connected components. The key requirement—that the decomposition is **static** during each analysis pass—ensures the assume-guarantee proof is valid.

### M6. Importance-Weighted Probability Estimation

Given a discovered race R, estimate P_μ(R) under deployment schedule distribution μ using importance sampling with cross-entropy-adapted proposal q:

P̂_μ(R) = (1/N) Σᵢ 𝟙[σᵢ ∈ R] · μ(σᵢ)/q(σᵢ)

The proposal q is a mixture distribution constructed from the adversarial search engine's trajectory data, adapted via cross-entropy minimization. Confidence intervals are derived from the effective sample size N_eff = (Σ wᵢ)² / (Σ wᵢ²).

---

## 5. Best-Paper Case

**Target venue: CAV (Computer-Aided Verification)**

The best-paper case rests on three pillars:

**Pillar 1: A new formal bug class with a complexity-theoretic separation.** The interaction race is precisely defined over asynchronous multi-agent execution with happens-before partial order. The separation theorem (non-projectability + statistical intractability + PSPACE-hardness) establishes that this is a genuinely new class of failure that existing paradigms cannot detect. This theoretical contribution stands alone as a publishable result independent of the tool.

**Pillar 2: A novel abstract domain combining concurrency theory and neural network verification.** HB-constrained zonotopes are a new point in the abstract interpretation design space, combining happens-before reasoning from concurrent systems verification with zonotope propagation from neural network verification. The corrected convergence proof for ReLU policies and the sound-without-convergence guarantee for general architectures are technically clean contributions.

**Pillar 3: An end-to-end tool that finds real bugs.** MARACE takes trained policy checkpoints and a safety specification, and produces either race-absence certificates (where convergence occurs) or concrete adversarial replays with probability bounds (everywhere else). The evaluation demonstrates both planted-bug detection and open-ended discovery across multiple domains.

**Why CAV over ML venues:** CAV values the combination of theory (new bug class + complexity result), technique (novel abstract domain), and systems (working tool)—exactly the profile of this work. ML venues (NeurIPS/ICML) would require stronger empirical scaling results that may not materialize in 6 months.

**Backup venue:** TACAS or FM, where the theoretical contribution and tool would both be competitive.

---

## 6. Hardest Challenge + Mitigation

**The single biggest risk: The abstract interpreter's false-positive rate may be too high for practical use.**

Zonotope over-approximation, combined with HB-consistency constraints, may flag large regions of the state-schedule space as potential races when no true race exists. If the false-positive rate exceeds ~30%, the adversarial search engine drowns in spurious leads, the race catalog becomes noise, and the system provides no practical value beyond simulation.

**Mitigation strategy (three layers):**

1. **Precision engineering.** Use Girard's generator reduction judiciously (reduce only when generator count exceeds 4d, not the aggressive 2d). This costs memory and iteration time but preserves precision where it matters. Budget 2–4× slower iterations for 3–5× fewer false positives.

2. **Search-based validation.** Every abstract-interpreter-flagged race region is validated by the adversarial search engine (MCTS + CMA-ES) attempting to find a concrete witnessing schedule. Unvalidated flags are demoted to "potential races" with lower confidence. This means the tool *always* produces useful output: confirmed races with replays, potential races with probability estimates, and (where convergence occurs) certified-safe regions.

3. **Graceful degradation.** If the abstract interpreter fails to converge or produces vacuously imprecise results for a given interaction group, the system falls back to pure adversarial search (without AI guidance). This degrades from "sound verification + targeted bug-finding" to "untargeted but systematic bug-finding"—still more valuable than random simulation, since MCTS/CMA-ES explores the schedule space more efficiently than uniform sampling.

---

## 7. Honest Feasibility Assessment

### What can actually be built in 6 months (1–2 researchers, 8-core laptop)

**Month 1–2: Core infrastructure + abstract interpreter prototype**
- HB-graph construction from execution traces
- Zonotope abstract domain with ReLU transfer functions (leverage existing DeepZ/CROWN implementations)
- Basic fixpoint engine with the corrected widening operator
- 2-agent prototype on a toy gridworld environment

**Month 3: Adversarial search + compositional decomposition**
- MCTS over schedule space with abstract-margin guidance
- CMA-ES alternative search strategy
- Interaction group decomposition and contract checking

**Month 4: Integration + first real benchmark**
- Policy ingestion (ONNX → zonotope transformers)
- PettingZoo environment adapters (Highway-Env, warehouse gridworld)
- SIS probability estimator
- End-to-end pipeline on Highway-Env 4-agent intersection

**Month 5: Evaluation**
- Planted-bug benchmarks across all three domains
- Baseline comparisons (per-agent CROWN, brute-force simulation)
- Ablation studies

**Month 6: Paper writing + polish**
- Discovery mode runs on pre-trained MARL policies
- Causal fix suggestion (min-cut) implementation
- Performance optimization, reproducibility artifacts

### Realistic deliverables

| Component | Status at 6 months |
|-----------|-------------------|
| Interaction race formalism + separation theorem | Complete (theory paper-ready) |
| HB-constrained zonotope abstract interpreter | Working for ReLU policies, 2–4 agents per group |
| Adversarial schedule search (MCTS + CMA-ES) | Working for up to 6–8 agents |
| Compositional decomposition | Basic version (static groups, linear contracts) |
| SIS probability estimation | Working |
| Causal fix suggestions | Prototype (re-simulation-based, not neural) |
| Gumbel-Softmax gradient search | Optional stretch goal |

### What will NOT be ready

- Non-ReLU policy support (LSTM, Transformer) — abstract transformers are research problems
- 50+ agent scalability — realistic is 4–8 agents per interaction group, up to ~12 total with decomposition
- Differentiable environment surrogates — dropped from v1 in favor of direct simulation
- Neural counterfactual prediction — replaced with re-simulation
- Partial identification under latent confounders — full observability assumed

### Target hardware: 8-core CPU, 16–32 GB RAM

| Scenario | Agents | Groups | Per-group Time | Total Time |
|----------|--------|--------|----------------|------------|
| Highway-Env intersection | 4 | 1–2 | 15–30 min | <1 hour |
| Warehouse gridworld | 8 | 2–4 | 20–40 min | <3 hours |
| ABIDES trading (stylized) | 4 | 1–2 | 10–20 min | <1 hour |

---

## 8. Evaluation Plan

### Benchmarks (3 domains, all with explicit asynchronous stepping)

**Domain 1 — Traffic (Primary).** Highway-Env multi-vehicle intersection and merging scenarios (2–6 agents) with asynchronous action timing. Policies trained with MAPPO. Both planted races at controlled frequencies (10⁻², 10⁻⁴ per episode) and open discovery mode.

**Domain 2 — Warehouse Robotics.** Continuous gridworld with corridor constraints (4–8 agents) with heterogeneous stepping rates. Multi-agent PPO policies. Planted deadlock-inducing corridor configurations.

**Domain 3 — Algorithmic Trading.** ABIDES limit-order-book simulator (3–4 agents) with microsecond-scale asynchronous order submission. Multi-agent DQN policies. Planted correlated-liquidation scenarios. We acknowledge that ABIDES is a simplified approximation of real exchange semantics; results demonstrate the methodology, not production-grade financial verification.

### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Race detection recall | Fraction of planted races detected | >90% for ε-races |
| False positive rate | Fraction of reported races not confirmed by replay | <20% |
| Sound coverage | Fraction of state-schedule space certified race-free (ReLU, N≤4) | >50% |
| Time to first race | Wall-clock seconds on 8-core CPU | <600s for N≤4 |
| Probability accuracy | SIS estimate vs. empirical frequency ratio | Within 10× for p < 10⁻³ |
| Scalability | Verification time as N scales 2→8 | Sub-exponential growth |

### Baselines

1. **Per-agent verification (CROWN/α-β-CROWN).** Verifies each policy independently. Expected to miss all interaction races (validating Theorem 1).

2. **Brute-force simulation (10⁷ steps, uniform scheduling).** Standard Monte Carlo. Expected to miss races rarer than ~10⁻⁵ (validating Theorem 2).

3. **CMA-ES schedule search without AI guidance.** Isolates the contribution of abstract-interpreter guidance.

4. **Ablation: MARACE without compositional decomposition.** Measures decomposition's scaling benefit.

5. **Ablation: MARACE without HB-aware pruning.** Measures HB constraint's precision contribution.

### Ground Truth Construction

- **Planted races:** Race probability known by design (controlled injection into deterministic scenarios with calibrated noise).
- **Empirical validation:** Targeted 10⁶–10⁷ step simulation focused on AI-identified race regions, using SIS proposal distribution for efficient ground-truth estimation. This is laptop-feasible (hours, not days).

### Minimum-Viable Evaluation (Degradation Plan)

| Priority | Benchmark | If time-limited |
|----------|-----------|-----------------|
| P0 | Highway-Env 4-agent planted | Must complete |
| P1 | Warehouse 4-agent planted | Reduce to 2-agent |
| P2 | ABIDES 4-agent planted | Drop domain |
| P3 | Discovery mode (Highway-Env) | Run only on P0 |
| P4 | Ablation studies | HB-pruning ablation only |

---

## 9. Scores (Post-Debate, Honest)

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8/10 | Addresses a real verification gap in multi-agent deployment. Reduced from 9–10 because near-term users are in traffic/warehouse (not finance), and the gap is not yet as acute as concurrent-software race detection. |
| **Difficulty** | 8/10 | HB-constrained zonotope abstract interpretation with convergence proofs is genuinely novel. Reduced from 9 because scope is narrowed to ReLU policies and modest agent counts. |
| **Potential** | 8/10 | New bug class + complexity separation + novel abstract domain + working tool is a strong CAV submission. Reduced from 9–10 because the tool's practical impact depends on false-positive rates that are uncertain pre-implementation. |
| **Feasibility** | 5/10 | Honest assessment: the abstract interpreter may not achieve useful precision in practice; zonotope over-approximation may be too coarse for real policies; 40K LoC of research-grade Rust in 6 months is ambitious. The adversarial search fallback ensures the system produces *something* useful even if AI convergence fails, preventing a total loss. |

### Risk-Adjusted Expected Outcome

- **70% probability:** Working adversarial schedule search (MCTS + CMA-ES) with basic AI guidance, finding planted races in 2–4 agent scenarios. Publishable as a systems paper with the theoretical contribution.
- **40% probability:** Abstract interpreter converges with useful precision for ReLU policies on 2-agent groups, enabling sound race-absence certificates for non-trivial state regions. Upgrades to a strong CAV paper.
- **15% probability:** Full pipeline works at 6–8 agents with compositional decomposition, abstract certification, and probability bounds. Best-paper contender.

---

## Appendix: Key Design Decisions and Their Rationale

### Why not a pure differentiable approach (Approach 2)?
The differentiable surrogate introduces two layers of approximation (Gumbel-Softmax relaxation + learned environment model) whose fidelity is uncertain. CMA-ES and MCTS over the true simulator provide the same bug-finding capability with fewer assumptions. We keep Gumbel-Softmax as an optional heuristic but do not build the architecture around it.

### Why not a pure causal approach (Approach 3)?
The faithfulness assumption fails for deterministic neural policies (the exact systems we target). ACE averages away rare catastrophic events. We adopt the *interventional race definition* as an elegant secondary characterization and the *min-cut fix synthesis* as a practical output, but use re-simulation rather than neural counterfactual prediction, and treat causal analysis as a post-hoc explainer rather than the primary detection mechanism.

### Why static decomposition, not adaptive?
The debate correctly identified that adaptive group merging during analysis invalidates the assume-guarantee proof. We use static decomposition per analysis pass. If new dependencies are discovered, we recompute the decomposition and restart—sound but potentially expensive. In practice, the initial HB graph captures most causal structure, so restarts are rare.

### Why Rust?
The core analysis engines (zonotope arithmetic, MCTS, CMA-ES, fixpoint iteration) are compute-intensive inner loops. Rust provides C-level performance with memory safety and fearless concurrency (Rayon for parallel group analysis). Python is used only for environment adapters and orchestration.

### Why not "ThreadSanitizer for multi-agent AI"?
ThreadSanitizer achieves near-zero false-positive rates in production. We cannot honestly claim this. MARACE provides soundness where convergence occurs and useful bug-finding everywhere else. The analogy is aspirational, not descriptive, and we avoid it in the paper.
