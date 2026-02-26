# MARACE: Interaction Race Detection via Happens-Before–Aware Abstract Interpretation for Multi-Agent RL Safety Verification

## Problem Statement and Approach

Multi-agent systems with independently trained or independently deployed control policies exhibit a fundamental safety gap: each policy may be individually verified—collision-free in isolation, profitable in expectation—yet their *composition* produces emergent catastrophic behaviors that no single-agent analysis can detect. This gap is not hypothetical. In mixed-autonomy traffic, vehicles from different manufacturers running independently developed planning stacks must negotiate unsignaled intersections with no shared coordinator—and the relative timing of their perception-to-actuation pipelines (varying from 20ms to 200ms across hardware platforms) determines whether they merge safely or collide. In multi-vendor drone airspace, FAA UTM corridors require provable deconfliction among swarm operators using independent path-planning policies with heterogeneous compute latencies and communication delays. In warehouse robotics, decentralized multi-robot systems (Locus Robotics, 6 River Systems) deploy learned obstacle-avoidance policies on robots with variable sensor update rates, creating effective asynchrony even in nominally synchronized fleets. In algorithmic trading, multiple market-making agents from different firms submit orders to the same exchange with microsecond-scale timing dependencies that determine whether their joint order flow stabilizes or destabilizes the book. These failures arise specifically from *asynchronous composition*—the absence of a global coordinator means agents' actions interleave in timing-dependent ways that neither agent anticipates.

We introduce the concept of an **interaction race**: a formal bug class defined over the joint execution of multiple independently deployed policies in asynchronous multi-agent systems. An interaction race occurs when two or more agents can reach a joint state satisfying a safety-violating predicate, and the relative ordering of their actions is unconstrained by any happens-before dependency—meaning the violation arises precisely because no coordination protocol governs their interleaving at that point. The formal definition is precise: given an execution trace τ over agents A₁, …, Aₙ with a happens-before partial order ≺_HB derived from observation dependencies, communication events, and environment-mediated causal chains, an interaction race at joint state s is a pair of action events (e_i, e_j) from distinct agents such that e_i ∦ e_j (incomparable under ≺_HB) and there exists a valid schedule permutation σ consistent with ≺_HB under which the resulting joint state s' = step(s, σ) satisfies a safety-violating predicate φ.

We prove a **separation theorem** establishing that interaction races occupy a formal blind spot of existing verification paradigms:

1. **Non-projectability (Theorem 1).** There exist interaction races whose projection onto every individual agent's trace satisfies that agent's local safety specification. Proof is by explicit construction for a class of piecewise-linear policies over 2D continuous state spaces, demonstrating that the race-inducing joint states lie in the *intersection* of per-agent safe regions while violating a joint predicate.

2. **Statistical intractability (Theorem 2).** For any finite sample budget B, any race with occurrence probability p < B⁻¹ · (1 − δ) is missed with probability ≥ 1 − δ. More substantively, for interaction races whose occurrence depends on schedule orderings within a timing window of width Δt in a continuous-time system, the probability that uniform random scheduling hits the race window scales as O((Δt/T)^k) where T is the episode length and k is the number of agents involved—making schedule-dependent races exponentially harder to find by sampling than state-dependent failures.

3. **Complexity-theoretic separation (Theorem 3, NEW).** We prove that the interaction race detection problem—given N policies, an environment transition function, and a safety predicate, does there exist an HB-consistent schedule leading to a safety violation?—is PSPACE-hard by reduction from the reachability problem in timed automata with N clocks, even when each individual policy is verifiable in polynomial time (deterministic, memoryless, piecewise-linear). This establishes a genuine complexity separation: single-agent verification is in P, but interaction race detection is PSPACE-hard, even for the simplest non-trivial policy class.

**MARACE** (Multi-Agent Race Analysis and Certification Engine) is a verification system that takes as input N policy checkpoints (PyTorch or ONNX), an environment simulator with explicit asynchronous stepping semantics, and a safety specification in a temporal logic over joint predicates, and produces as output a **coordination race catalog**: for every detected interaction race, a probability bound on its occurrence under the deployment schedule distribution, an adversarial replay trace demonstrating the race, and—where the abstract interpretation fixpoint converges—a sound over-approximation certifying race absence for verified regions of the joint state-schedule space.

The system's architecture fuses four analysis engines:

1. **Happens-before partial-order engine.** Constructs a causal ordering over multi-agent execution traces from three sources: (a) observation dependencies (agent B observes state modified by agent A), (b) explicit communication events (messages, shared-memory writes), and (c) environment-mediated causal chains (agent A's action changes a physical quantity that enters agent B's observation within a bounded delay). The HB graph's connected components define **interaction groups**—sets of agents with potential causal dependencies.

2. **HB-aware abstract interpretation framework** (primary technical novelty). Performs joint reachability analysis using zonotope abstract domains equipped with a concretization function that maps abstract elements to sets of *HB-consistent* joint states—not arbitrary Cartesian products. The key insight is a novel abstract transfer function that prunes abstract elements violating causal ordering constraints derived from the HB graph, yielding sound over-approximation of only the reachable states under valid schedules. This reduces the abstract state space from exponential in N (full Cartesian product) to polynomial in the interaction group size k (typically 2–3 agents). For convergence: we prove that for ReLU networks with bounded depth D and width W, the HB-aware widening operator stabilizes in at most O(D · W · |schedule_windows|) iterations, yielding a finite ascending chain. For general architectures, the framework provides sound over-approximation at each iteration even without fixpoint convergence—yielding race *detection* with formal guarantees, with absence certification as a bonus when convergence occurs.

3. **Adversarial search engine.** Uses k-bounded Monte Carlo tree search over the schedule space (not the action space) to find worst-case interleavings that drive the joint state toward safety-violation boundaries identified by the abstract interpreter. Abstract-interpretation–guided pruning reduces effective branching from O(N!) to O(k!) where k is the interaction group size.

4. **Sequential importance sampling engine.** Provides statistically rigorous probability bounds on race occurrence by importance-sampling over the schedule distribution conditioned on abstract-interpretation–identified race regions, with cross-entropy adaptation for proposal construction.

The key architectural insight is **compositional decomposition**: MARACE partitions agents into interaction groups based on the HB graph's connected components, analyzes each group independently, and composes results via assume-guarantee contracts on shared state predicates. This decomposition is sound—we prove that any interaction race in the full system projects onto at least one interaction group—and reduces verification cost from O(|S|^N) to O(Σ |S|^{k_i}) where k_i is the size of the i-th interaction group.

## Value Proposition

MARACE addresses a verification gap at the intersection of formal methods and multi-agent AI deployment. The gap is already real in three concrete settings:

1. **Mixed-autonomy traffic.** Vehicles from different OEMs (Tesla, Waymo, Mobileye, comma.ai) will share roads with no central coordinator. Each OEM verifies its own stack independently. No existing tool verifies their *composition*. The V2X standardization effort (SAE J3016, ETSI ITS) explicitly assumes cooperative behavior but provides no formal verification of cooperation.

2. **Multi-vendor drone airspace.** FAA's UTM (UAS Traffic Management) framework requires deconfliction among operators with independent flight controllers. Current deconfliction relies on pre-planned 4D trajectories, which break down when reactive RL-based controllers deviate from plans.

3. **Decentralized warehouse robotics.** Multi-vendor fleet deployments (multiple robot types from different manufacturers sharing warehouse corridors) increasingly use learned reactive policies rather than centralized MAPF planners, creating exactly the independently-deployed-policy composition that MARACE targets.

Current practice relies on simulation testing (billions of steps), which provides coverage estimates but no guarantees and systematically misses schedule-dependent failures. Formal methods for multi-agent systems (PRISM-games, Storm, shield synthesis, multi-agent CBFs) either require explicit finite-state models incompatible with neural policies, restrict to two-player zero-sum structure, or provide runtime enforcement without pre-deployment diagnosis. MARACE is the first tool to provide *pre-deployment, static analysis* of independently deployed learned policies with formal soundness guarantees, filling the role that ThreadSanitizer and RacerD fill for concurrent software—but for concurrent *learned behaviors* operating over continuous state spaces with asynchronous timing.

## Technical Difficulty

The system requires solving four genuinely novel subproblems and two engineering challenges:

### Novel Algorithmic Contributions

**H1. Interaction Race Semantics over Asynchronous Continuous Systems.** Defining races for systems with continuous state, stochastic transitions, asynchronous stepping, and partial observability requires extending Lamport's happens-before from discrete message-passing to continuous-time event spaces with environment-mediated causality. The happens-before relation must be constructed from three causal sources (observation dependencies, communication events, physics-mediated chains), each requiring formal definition over continuous dynamics. The ε-race formulation—two agents are in an ε-race if a safety violation is reachable under any HB-consistent schedule permutation within an ε-ball of the current joint state—uses an automated, iterative ε-calibration procedure: initialize ε₀ = L⁻¹ · δ_global (where L is the maximum policy Lipschitz constant and δ_global is the coarse global safety margin), run abstract interpretation, compute refined safety margin δ₁, update ε₁ = L⁻¹ · δ₁, and repeat. The iteration is monotone in a complete lattice and converges in at most O(log(δ_global/δ_min)) steps.

**H2. HB-Aware Abstract Interpretation with Convergence Guarantees.** Standard abstract interpretation over the joint state space is intractable (exponential in N). Our framework restricts the abstract domain to HB-consistent states via a novel transfer function that prunes abstract elements violating causal ordering constraints. The central technical challenge is designing widening operators that preserve HB-awareness while guaranteeing fixpoint convergence. **We provide a convergence proof for ReLU networks with bounded depth and width:** the HB-consistency predicate is a conjunction of linear constraints over the abstract state, and widening over zonotopes with linear side constraints produces a finite ascending chain when the constraint set is fixed. For general architectures, we provide sound per-iteration guarantees without requiring convergence, and characterize the convergence conditions (Lipschitz bound on the abstract transfer function < 1 in the Hausdorff metric on zonotopes).

**H3. Compositional Assume-Guarantee Decomposition.** Partitioning the agent interaction graph, generating interface contracts over continuous predicates, and proving that per-group verification composes into whole-system guarantees. The contract language uses linear arithmetic over shared state variables, which is decidable and amenable to SMT-based automated checking. The decomposition is adaptive: as the abstract interpreter discovers new interaction patterns (causal dependencies not visible in the initial HB graph), groups are merged and contracts are strengthened.

**H4. Complexity-Theoretic Separation.** The PSPACE-hardness proof for interaction race detection via reduction from timed automata reachability. This is genuine theoretical work requiring careful encoding of clock constraints as schedule-timing constraints and policy state as automaton configurations.

### Engineering Contributions

**H5. Schedule-Space Adversarial Search.** MCTS over the combinatorial schedule space, guided by abstract-interpretation safety-margin estimates. The algorithm is standard MCTS with a novel UCB1 variant that uses abstract safety margins as value estimates. Engineering challenge: effective pruning to reduce branching from O(N!) to tractable levels.

**H6. Policy Ingestion and Abstract Transformers.** Extracting Lipschitz bounds and implementing abstract transformers for ReLU and Tanh network layers. We restrict to ReLU/Tanh architectures (covering >90% of deployed RL policies) and build on existing abstract transformer implementations from the neural network verification literature (DeepZ, CROWN), adapting them for the HB-aware framework.

### Subsystem Breakdown (~110K LoC)

| ID  | Subsystem | LoC | Language | Notes |
|-----|-----------|-----|----------|-------|
| S01 | Trace Infrastructure | 3K | Rust | HB-stamped execution traces, serialization |
| S02 | HB & Partial-Order Engine | 10K | Rust | Vector clocks, transitive closure, connected components, causal chain inference |
| S03 | Abstract Interpretation Framework | 25K | Rust | Zonotope domains, HB-aware transfer functions, widening with convergence proofs, fixpoint engine |
| S04 | Compositional Decomposition Engine | 10K | Rust | Interaction graph, contract generation/checking, assume-guarantee composition |
| S05 | SIS Engine | 10K | Rust | Cross-entropy proposal adaptation, schedule-space sampling, confidence intervals |
| S06 | Adversarial Search Engine | 14K | Rust | k-bounded MCTS over schedules, AI-guided pruning |
| S07 | Integration & Orchestration | 10K | Rust+Python | Pipeline control, Rayon parallelism, caching |
| S08 | Policy Ingestion | 7K | Rust | ONNX import, Lipschitz extraction, ReLU/Tanh abstract transformers |
| S09 | Environment Adapters | 6K | Python | PettingZoo/Gymnasium interfaces with async stepping semantics |
| S10 | Specification Language & Checker | 6K | Rust | Temporal logic parser, joint-predicate evaluation, contract DSL |
| S11 | Evaluation Harness | 5K | Python | Benchmark suite, metric collection, baseline runners |
| S12 | Reporting | 4K | Python | Race catalog output, adversarial replay traces, proof certificates |
| | **Total** | **~110K** | **~85K Rust + ~25K Python** | |

## New Mathematics Required

**M1. Interaction Race Formalism and Separation Theorem.** Define interaction races as a predicate over joint execution traces equipped with a happens-before partial order derived from observation dependencies, communication events, and environment-mediated causal chains in continuous-time asynchronous systems. Prove three separation results: (1) non-projectability by explicit construction over piecewise-linear policies; (2) statistical intractability with schedule-dependent exponential scaling; (3) PSPACE-hardness of interaction race detection by reduction from timed automata reachability, even when single-agent verification is in P.

**M2. ε-Race Detection with Iterative Calibration.** Extend discrete race detection to continuous domains via ε-races. Define the iterative ε-calibration as a monotone map on a complete lattice of safety margins and prove convergence. Bound the false-positive volume as a function of ε and the environment's Lipschitz constant.

**M3. HB-Aware Abstract Interpretation with Convergence.** Construct the HB-restricted abstract domain with a concretization function mapping to HB-consistent joint states. Prove soundness of the abstract transfer functions. For ReLU networks with bounded depth D and width W: prove that widening over zonotopes with HB-consistency constraints (a finite conjunction of linear inequalities) produces a strictly ascending chain of bounded length O(D · W · |HB_windows|), guaranteeing fixpoint convergence. For general architectures: prove sound per-iteration over-approximation and characterize convergence conditions via contraction in the Hausdorff metric.

**M4. Compositional Soundness.** For a partition of N agents into interaction groups G₁, …, Gₘ with interface contracts Φᵢⱼ: if each group Gᵢ is race-free under its contracts, and every contract Φᵢⱼ is discharged by the abstract interpreter, then the full N-agent system is race-free. Proof by contraposition: any global race trace contains a sub-trace projecting onto some group Gᵢ that violates Gᵢ's local race-freedom, via the causal closure property of the HB graph's connected components.

## Best Paper Argument

The strongest best-paper case rests on three pillars, each now substantiated:

*First*, the interaction race is a genuinely new formal bug class—precisely defined over asynchronous multi-agent execution with a happens-before partial order—with a separation theorem that includes a *complexity-theoretic* gap: interaction race detection is PSPACE-hard even when single-agent verification is in P. This is not three obvious observations stapled together; the PSPACE-hardness proof via timed automata reduction requires non-trivial encoding and establishes that the problem is fundamentally harder than its single-agent counterpart.

*Second*, the HB-aware abstract interpretation framework is a novel contribution to abstract interpretation theory: restricting abstract domains by a concurrency-theoretic partial order, with proved convergence for ReLU networks. No prior work combines zonotope abstract domains with happens-before consistency constraints, and the tractability gain (exponential to compositional) is both theoretically clean and practically necessary.

*Third*, the system is end-to-end and finds bugs: it takes trained policy checkpoints and a safety spec, and produces either a sound race-absence certificate or a concrete adversarial replay. The evaluation includes both planted-bug detection (validating detection capability) and open-ended discovery on standard MARL benchmarks (demonstrating practical value).

The target venue is **CAV** (Computer-Aided Verification), where the combination of a new bug class, a complexity-theoretic separation, a novel abstract interpretation framework, and a working verification tool matches the profile of recent best papers.

## Evaluation Plan

All evaluation is fully automated with no human-in-the-loop steps.

**Benchmarks (3 domains, all with explicit asynchronous stepping):**

- *Autonomous Driving*: Highway-Env multi-vehicle intersection and merging scenarios (2–6 agents) with asynchronous action timing. Policies trained with MAPPO and QMIX. (a) Planted race conditions at controlled frequencies (10⁻², 10⁻⁴, 10⁻⁶ per episode) to measure detection capability. (b) Discovery mode: run on standard pre-trained policies without injection to find naturally occurring interaction races.

- *Warehouse Robotics*: Lightweight continuous gridworld with corridor constraints (4–12 agents) with heterogeneous stepping rates. Policies from multi-agent PPO. Inject deadlock-inducing corridor configurations and run discovery mode.

- *Algorithmic Trading*: ABIDES limit-order-book simulator (3–6 agents) with microsecond-scale asynchronous order submission. Policies from multi-agent DQN. Inject correlated-liquidation race conditions.

**Metrics:**
1. *Race detection recall*: fraction of planted races detected (target: >95% for ε-races, 100% for exact races).
2. *False positive rate*: fraction of reported races that do not correspond to actual safety violations in ground-truth simulation (target: <10%, accepting higher FPR for conservative ε-calibration).
3. *Sound coverage*: fraction of joint state-schedule space for which race absence is formally certified via converged fixpoint (target: >70% for N ≤ 4 with ReLU policies).
4. *Time to detection*: wall-clock seconds on a single 8-core CPU to find the first race (target: <600s for N ≤ 4).
5. *Probability bound accuracy*: ratio of SIS-estimated race probability to empirical race frequency from ground-truth simulation (target: within 10× for races rarer than 10⁻⁴).
6. *Scalability*: verification time as N scales from 2 to 12, demonstrating compositional decomposition's sub-exponential scaling.
7. *Discovery yield*: number and nature of previously unknown interaction races found in standard pre-trained MARL policies.

**Ground Truth Construction (laptop-feasible):**
Rather than brute-force 10⁹-step simulation, ground truth is established by construction:
- For planted races: race probability is *known by design* (controlled injection rate into deterministic base scenarios with calibrated noise).
- For empirical validation: run 10⁶–10⁷ step targeted simulation (hours, not days) focused on abstract-interpreter–identified race regions, using the SIS proposal distribution for efficient ground-truth estimation.

**Baselines:**
- *Per-agent verification*: individual policy verification via CROWN/α-β-CROWN (expected to miss all interaction races, validating Theorem 1).
- *Brute-force simulation*: 10⁷-step Monte Carlo with uniform scheduling (expected to miss races rarer than ~10⁻⁵).
- *PRISM-games*: finite-state model checking of discretized policies (expected to miss continuous-state races, validating Theorem 3).
- *Ablation*: MARACE without compositional decomposition, without HB-aware pruning, and without adversarial search, to isolate each component's contribution.

## Laptop CPU Feasibility

MARACE is explicitly designed for single-machine, CPU-only execution. Four design decisions make this feasible:

*First*, compositional decomposition reduces the verification problem from the full N-agent joint space to independent analysis of interaction groups of size 2–3, each tractable on a laptop. For a 12-agent warehouse scenario, the HB graph typically decomposes into 4–6 groups of 2–3 agents.

*Second*, the abstract interpretation framework uses zonotope domains (polynomial-time affine arithmetic) with Girard's generator reduction method to bound memory: generators are periodically reduced to a maximum of 2d generators (where d is the state dimension) via PCA-based merging, capping per-element memory at O(d²) regardless of trace length.

*Third*, the convergence guarantee for ReLU networks bounds the number of fixpoint iterations, providing predictable wall-clock time: for a 2-agent group with 3-layer ReLU policies (256 hidden units) over a 10-dimensional joint state space, the fixpoint converges in ~50–200 iterations, each taking ~1–5 seconds on 8 cores, for a total of ~2–15 minutes per group.

*Fourth*, the Rust implementation of core analysis engines (S01–S06, S08, S10) provides C-level performance with safe concurrency via Rayon.

**Target hardware:** 8-core CPU, 16–32 GB RAM.

| Scenario | Agents | Groups | Per-group Time | Per-group RAM | Total Time |
|----------|--------|--------|----------------|---------------|------------|
| Highway-Env intersection | 4 | 1-2 | 15-30 min | 2-4 GB | <1 hour |
| Highway-Env merging | 6 | 2-3 | 15-30 min | 2-4 GB | <2 hours |
| Warehouse gridworld | 12 | 4-6 | 20-40 min | 2-6 GB | <4 hours |
| ABIDES trading | 6 | 2-3 | 10-20 min | 1-3 GB | <1 hour |

**Minimum-viable evaluation table (degradation plan):**

| Priority | Benchmark | Target | If time-limited |
|----------|-----------|--------|-----------------|
| P0 | Highway-Env 4-agent planted | Full metrics | Must complete |
| P1 | Highway-Env 6-agent planted | Full metrics | Reduce to 4 agents |
| P2 | Warehouse 8-agent planted | Detection + scalability | Reduce to 4 agents |
| P3 | ABIDES 4-agent planted | Detection only | Drop domain |
| P4 | Discovery mode (all domains) | Report any findings | Run on P0 only |
| P5 | Ablation studies | All ablations | AI-pruning ablation only |

---

**Slug:** `marl-interaction-race-verifier`
