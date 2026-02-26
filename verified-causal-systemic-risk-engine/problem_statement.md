# CausalBound: Verified Worst-Case Systemic Risk Bounds via Decomposed Causal Polytope Inference with Adversarial Search

---

## PROBLEM DESCRIPTION

Systemic financial risk assessment remains one of the most consequential unsolved problems in computational economics. Current regulatory stress testing frameworks—the Federal Reserve's CCAR/DFAST, the EBA's EU-wide stress tests, and the Bank of England's ACS—operate by propagating a small number of hand-crafted macroeconomic scenarios (typically 3–5) through institution-level models and aggregating losses. These frameworks suffer from three fundamental deficiencies. First, they are *scenario-dependent*: they evaluate risk conditional on specific narratives chosen by regulators, providing no guarantee that the worst-case scenario has been considered. The 2008 Global Financial Crisis, the 2010 European sovereign debt crisis, the 2020 COVID liquidity shock, and the 2023 UK gilt crisis each involved contagion pathways that fell outside the scenario sets of contemporaneous stress tests. Second, they treat the financial network as a black box: contagion propagates through bilateral exposures (derivatives, repo, securities lending) whose nonlinear interactions create emergent systemic effects invisible to institution-level analysis. Models like DebtRank (Battiston et al. 2012) and its extensions capture network topology but provide only point estimates or Monte Carlo confidence intervals—never provable bounds. Third, no existing framework provides machine-checkable certificates that its risk assessments are sound: model risk is assessed via qualitative expert review (OCC SR 11-7), not formal verification.

This artifact addresses all three deficiencies through a novel computational pipeline. The core insight is that financial contagion networks, while globally complex (treewidth 15–30+ for full interbank networks), can be decomposed into overlapping bounded-treewidth subgraphs via graph decomposition algorithms (tree decomposition with separator-based partitioning). On each subgraph (typically 30–50 nodes, treewidth ≤ 12), we formulate and solve a *causal polytope linear program*: the set of all joint distributions consistent with the observed network's structural causal model (SCM) and the constraints imposed by do-calculus interventional semantics. The LP optimizes over this polytope to produce provable worst-case bounds on contagion loss—not point estimates, but intervals [L, U] such that no distribution consistent with the causal structure can produce losses outside these bounds. A novel *bound composition theorem* then aggregates subgraph bounds into global network bounds, with a characterized composition gap that quantifies the price of decomposition. All inferential claims—every bound, every conditional independence assertion, every causal identification result—are incrementally certified by a streaming SMT co-routine that emits machine-checkable proof certificates during junction-tree message passing.

The causal semantics in this pipeline are structurally essential, not decorative. Standard Bayesian network inference computes P(Loss | Evidence), which answers "what loss do we expect given what we've observed?" Systemic risk assessment requires the interventional query P(Loss | do(Shock)), which answers "what loss results if we *force* a shock, regardless of why that shock might occur?" This distinction is the difference between correlation and causation: observing that Lehman Brothers' CDS spreads widened does not tell us what happens if Lehman defaults, because the observation is confounded by market-wide factors. The do-calculus (Pearl 2009) provides the formal machinery to answer interventional queries from observational data when the causal graph satisfies identifiability conditions. Our SCMs encode the causal mechanisms of financial contagion—counterparty credit exposure, margin spirals, fire-sale externalities, funding liquidity withdrawal—as structural equations, and the causal polytope LP bounds are computed over the interventional distributions induced by do-operator applications. When identifiability fails (because of latent confounders such as unobserved correlated positions), the polytope bounds automatically widen to reflect genuine causal uncertainty, producing honest intervals rather than false precision.

Finally, the pipeline deploys Monte Carlo Tree Search (MCTS) with a novel *causal UCB* acquisition function to adversarially search for worst-case cross-boundary shock scenarios. Rather than searching the full combinatorial shock space (intractable), MCTS operates over the *interface variables* at subgraph boundaries—the separator sets from graph decomposition, typically 5–15 variables per boundary. Each MCTS rollout triggers a memoized junction-tree inference pass (~5ms base, ~2ms amortized with subtree caching), and the causal UCB criterion uses d-separation structure to prune provably irrelevant branches. This scoping makes adversarial search tractable: 100K rollouts × 5ms ≈ 8 minutes on a single laptop CPU, sufficient for PAC-optimal convergence guarantees on the interface shock space. The entire pipeline—decomposition, LP solving, junction-tree inference, SMT verification, MCTS search—is CPU-native, requiring no GPU, no cloud, no proprietary data feeds, and no human annotation.

---

## VALUE PROPOSITION

**Central bank supervisors (Federal Reserve, ECB, Bank of England, MAS)** currently commission stress tests that cost millions of dollars per cycle and take 6–9 months to execute, yet provide no guarantee of worst-case coverage. CausalBound would give supervisors a tool that produces *provable* worst-case contagion bounds with machine-checkable soundness certificates, runs in minutes on commodity hardware, and automatically discovers adversarial scenarios that human scenario designers miss. The SMT certificates directly address the model risk governance requirements of SR 11-7 and SS1/23 by providing formal evidence that inferential claims are sound with respect to the stated causal model—something no existing stress testing tool can offer.

**Central counterparties (LCH, CME, Eurex, ICE)** manage default waterfalls that must withstand the simultaneous default of their two largest clearing members (Cover 2 requirement under EMIR/Dodd-Frank). Current CCP stress tests use historical and hypothetical scenarios with no formal coverage guarantee. CausalBound's worst-case bounds on multi-entity contagion loss, computed via causal polytope LP, would provide CCPs with the first *provably conservative* default fund sizing methodology. The adversarial MCTS search specifically targets cross-CCP contagion—the systemic risk regulators fear most but cannot currently model.

**Risk model validators (OCC, PRA model validation teams, internal validation at G-SIBs)** spend thousands of person-hours per model assessing conceptual soundness, typically through qualitative review. CausalBound's SMT certificates provide machine-checkable proofs that every inferential step is sound—transforming model validation from subjective expert judgment to automated formal verification. This does not replace validators (they must still assess model scope and fitness-for-purpose), but it eliminates an entire class of model risk: logical errors in inference.

**Target researcher alignment:** *Sung-Ho Kim* (causal inference, financial stability): the causal polytope LP formulation extends his work on causal bounds to network settings. *Justin P. Moore* (formal methods, cyber-physical systems): the streaming SMT verification of probabilistic inference is a novel formal methods application. *Evan T. Fields* (RL, systematic trading): the causal UCB MCTS variant bridges RL exploration theory with causal inference. *Pedro F. Silva* (Bayesian networks, derivatives pricing): the junction-tree engine with SIMD vectorization and exact inference on financial instrument CPDs directly extends his expertise.

---

## TECHNICAL DIFFICULTY

### Hard Subproblem 1: Graph Decomposition with Bounded Treewidth for Financial Networks

Real interbank networks (e.g., the Fedwire payment network with ~7,000 participants, or OTC derivatives networks with ~1,000 significant counterparties) have global treewidth estimated at 15–30+. Exact treewidth computation is NP-hard. The difficulty is not merely computing a tree decomposition—heuristics like min-fill and min-degree exist—but producing a decomposition where (a) every subgraph has tw ≤ 12 (required for tractable junction-tree inference), (b) separator sets are small (5–15 variables, required for tractable MCTS), and (c) the decomposition respects causal semantics (cutting an edge in the SCM must be handled by introducing boundary conditions that preserve interventional semantics). This requires adapting graph decomposition algorithms to work with directed acyclic graphs (DAGs) while preserving the moral graph structure needed for junction-tree inference. When tw ≤ 12 decomposition is not achievable for a subgraph, the system falls back to approximate inference (loopy belief propagation or variational methods) with convergence certificates that bound the approximation error, ensuring graceful degradation rather than pipeline failure.

### Hard Subproblem 2: Causal Polytope LP on Subgraphs with Column Generation

The causal polytope for a subgraph with n binary variables has up to 2^n vertices—exponential in the number of variables. For n = 50, this is intractable by direct enumeration. The difficulty is formulating a column generation scheme where (a) the pricing subproblem can be solved efficiently by exploiting the DAG structure of the SCM, (b) observational and interventional constraints (conditional independences from d-separation, do-calculus identification results) are encoded as LP constraints without exponential blowup, and (c) the LP optimum provides valid worst-case bounds on contagion loss that are tight (the gap between the LP bound and the true worst case over all consistent distributions is small). We restrict subgraphs to 30–50 nodes and exploit the bounded treewidth (tw ≤ 12) to structure the column generation, keeping the number of active columns polynomial in n for fixed treewidth.

### Hard Subproblem 3: Subgraph Bound Composition Theorem (Novel Mathematics)

Given worst-case bounds [L_i, U_i] computed independently on overlapping subgraphs G_1, ..., G_k, how do these compose into global bounds [L, U] on the full network? The difficulty is that subgraph bounds are computed under potentially inconsistent marginal distributions on shared separator variables. The composition theorem must (a) characterize the composition gap—how much tighter the global bound could be if subgraphs were solved jointly, (b) provide conditions under which the gap is bounded (e.g., when separator sets have bounded size and the contagion function has bounded Lipschitz constant across boundaries), and (c) be computationally verifiable (the SMT co-routine must be able to certify the composition). This is genuinely novel mathematics: no existing work in causal inference or probabilistic graphical models provides a composition theorem for interventional bounds over graph decompositions.

### Hard Subproblem 4: Junction-Tree Exact Inference with SIMD Vectorization

Junction-tree inference (Lauritzen-Spiegelhalter) on discrete Bayesian networks with treewidth tw requires O(n · k^tw) operations where k is the variable cardinality. For tw = 12 and k = 8 (representing discretized financial exposures), a single clique potential table has 8^12 ≈ 68 billion entries—far too large. The difficulty is (a) choosing discretization granularity that balances accuracy against table size (adaptive discretization with k = 4–6 per variable, varying by variable type), (b) exploiting sparsity in CPDs (most financial instrument payoff functions have structured sparsity: CDS payoffs are piecewise linear, option payoffs have known functional form), (c) implementing SIMD-vectorized message passing that achieves near-peak throughput on modern x86 CPUs (AVX-512 for 512-bit vector operations on potential tables), and (d) handling the interventional semantics correctly (do-operator application requires mutilating the CPDs, which changes the junction tree structure).

### Hard Subproblem 5: Streaming SMT Co-routine for Incremental Verification

The SMT solver (Z3 or CVC5) must verify each inference step *as it happens*, not post-hoc. The difficulty is (a) formulating junction-tree message passing as a sequence of SMT assertions (each message is a claim about conditional distributions that must be consistent with the CPDs and the d-separation structure), (b) maintaining an incremental assertion stack that grows linearly (not quadratically) in the number of messages, (c) handling the interaction between LP bounds (which are real-arithmetic claims) and graphical model structure (which involves combinatorial constraints on graph topology), requiring the theory combination of QF_LRA and graph-theoretic predicates, and (d) ensuring that the verification overhead is bounded—target ≤ 3× wall-clock overhead compared to uncertified inference. The streaming protocol must be *sound*: if the SMT solver accepts all incremental assertions, the final inference result is guaranteed correct with respect to the stated SCM.

### Hard Subproblem 6: MCTS Adversarial Search with Causal UCB

Standard UCB1 treats all arms as exchangeable. In our setting, the "arms" are shock configurations on interface variables, and the causal structure implies that some shocks are d-separated from the loss variable given the current partial assignment—meaning they provably cannot affect the outcome. The difficulty is (a) defining a causal UCB acquisition function that uses d-separation to prune provably irrelevant branches (reducing the effective branching factor), (b) proving PAC-optimal convergence (the MCTS finds a near-worst-case scenario with probability ≥ 1-δ within a bounded number of rollouts), (c) handling the non-stationarity introduced by the composition theorem (the global bound changes as different subgraphs are evaluated), and (d) ensuring that the MCTS rollouts use memoized junction-tree inference efficiently (cache invalidation when the shock configuration changes only boundary variables).

### Hard Subproblem 7: Financial Instrument Exposure Models

The conditional probability distributions (CPDs) in the SCM must faithfully encode the payoff functions and risk characteristics of real financial instruments: credit default swaps (CDS) with accrual, upfront fees, and recovery rate uncertainty; interest rate swaps (IRS) with day-count conventions and CSA margining; repurchase agreements with haircut dynamics and right-of-substitution; equity options with discrete dividends and early exercise. The difficulty is discretizing these continuous payoff functions into the finite-cardinality CPDs required by junction-tree inference while preserving the tail behavior that drives systemic risk. Each instrument type requires a bespoke discretization strategy that captures the nonlinear payoff structure (e.g., CDS protection leg is a discontinuous function of default time).

### Hard Subproblem 8: SCM Construction from Partially-Observed Data

Real financial networks are partially observed: bilateral OTC derivatives positions are reported to trade repositories but with lags and inconsistencies; interbank lending relationships are inferred from payment system data; common asset holdings are partially disclosed via regulatory filings. Constructing an SCM requires (a) causal discovery under latent confounders (FCI algorithm or variants), (b) incorporating domain knowledge as orientation rules (e.g., margin calls are caused by price moves, not vice versa), (c) handling selection bias (reported data is not a random sample of all positions), and (d) quantifying structural uncertainty (multiple DAGs may be Markov-equivalent given the data). The causal polytope bounds must account for this structural uncertainty by optimizing over the set of distributions consistent with *any* DAG in the Markov equivalence class.

---

## HONEST SUBSYSTEM BREAKDOWN

| Subsystem | Algorithmic LoC | Test LoC | Supporting/Infra LoC | Description |
|---|---|---|---|---|
| **Graph decomposition engine** | 6,500 | 4,200 | 2,800 | Tree decomposition (min-fill, metaheuristic refinement), separator extraction, causal-aware partitioning, bounded-tw enforcement |
| **Causal polytope LP solver** | 8,200 | 5,500 | 3,000 | Column generation master/pricing, d-separation constraint encoding, interventional polytope construction, bound extraction |
| **Bound composition engine** | 4,800 | 3,500 | 1,500 | Composition theorem implementation, gap estimation, separator consistency checking, monotone bound propagation |
| **Junction-tree inference engine** | 9,500 | 6,000 | 4,000 | Clique tree construction, SIMD-vectorized message passing, adaptive discretization, do-operator mutilation, memoization cache |
| **Streaming SMT co-routine** | 7,200 | 4,800 | 3,500 | Incremental assertion protocol, QF_LRA encoding of LP bounds, graph-theoretic predicate encoding, certificate emission, Z3/CVC5 bindings |
| **MCTS adversarial search** | 6,800 | 4,500 | 2,200 | Causal UCB implementation, d-separation pruning, interface shock enumeration, rollout scheduling, PAC convergence monitoring |
| **SCM construction** | 5,500 | 3,800 | 2,500 | FCI causal discovery, domain orientation rules, Markov equivalence class enumeration, partial observability handling |
| **Financial instrument models** | 7,000 | 5,000 | 2,000 | CDS, IRS, repo, equity option CPD encoders, adaptive discretization per instrument type, tail-preserving quantization |
| **Network topology generator** | 3,500 | 2,200 | 1,500 | Erdős–Rényi, scale-free, core-periphery, empirical topology loaders, calibration to BIS/ECB statistics |
| **Evaluation & benchmarking harness** | 4,500 | 2,000 | 5,500 | Monte Carlo ground truth engine, historical crisis topology reconstruction, automated metric computation, regression testing |
| **Data pipeline & serialization** | 2,500 | 1,500 | 3,000 | Network serialization (protobuf), SCM I/O, LP solution caching, checkpoint/restart |
| **CLI & orchestration** | 2,000 | 1,200 | 3,500 | Pipeline orchestration, configuration management, logging, progress reporting |
| **Integration & end-to-end tests** | 5,000 | 3,000 | — | Cross-subsystem integration tests, full-pipeline scenario tests, regression suites |

**Totals:** ~73,000 algorithmic LoC + ~47,200 test LoC + ~35,000 supporting/infrastructure LoC = **~155,200 total LoC**

Every algorithmic line serves a purpose: the graph decomposition engine implements genuine NP-hard optimization heuristics; the LP solver implements column generation with a non-trivial pricing subproblem; the junction-tree engine requires careful numerical implementation to avoid underflow in high-dimensional potential tables. The test infrastructure is substantial because correctness is a first-class requirement—the SMT certificates are only as trustworthy as the assertion encoding, and every encoding must be tested against known-correct inference results.

---

## NEW MATHEMATICS REQUIRED

### 1. Subgraph Causal Polytope Bound Computation

**Statement.** Given a sub-DAG G_i = (V_i, E_i) of the full SCM with treewidth tw(G_i) ≤ w, a target variable Y ∈ V_i, and an intervention do(X = x) for X ⊆ V_i, define the *causal polytope* P_i as the set of all joint distributions P(V_i) consistent with (a) the DAG structure (Markov factorization), (b) observed marginal constraints, and (c) interventional constraints derived from do-calculus. The *subgraph bound* is B_i = [min_{P ∈ P_i} E_P[Y | do(X=x)], max_{P ∈ P_i} E_P[Y | do(X=x)]].

**Required result.** An LP formulation of P_i whose constraint matrix has dimension polynomial in |V_i| for fixed treewidth w, together with a column generation scheme where the pricing subproblem decomposes along the cliques of the junction tree of G_i's moral graph. The key technical challenge is encoding the interventional (mutilated graph) constraints without enumerating all joint configurations.

### 2. Bound Composition Theorem

**Statement.** Given subgraph bounds B_1, ..., B_k computed independently on overlapping sub-DAGs G_1, ..., G_k that cover the full DAG G = (V, E), with separator sets S_{ij} = V_i ∩ V_j, define the *composed bound* B = f(B_1, ..., B_k, {S_{ij}}).

**Required result.** (a) A composition function f that produces valid (conservative) global bounds: for any distribution P consistent with the full DAG G, E_P[Y | do(X=x)] ∈ B. (b) A characterization of the *composition gap* Δ = |B| - |B*| where B* is the bound obtained by solving the LP on the full DAG. (c) Conditions under which Δ → 0 (e.g., when the contagion loss function has bounded Lipschitz constant L across separator variables and |S_{ij}| ≤ s, then Δ ≤ O(k · L · s · ε) where ε is the discretization granularity). This theorem is the intellectual core of the contribution—it enables the decomposition strategy that makes the pipeline tractable.

### 3. Causal UCB for MCTS

**Statement.** Define a multi-armed bandit over shock configurations σ ∈ Σ on interface variables, where the reward of arm σ is the contagion loss L(σ) computed via junction-tree inference. Standard UCB1 requires O(|Σ| log T) regret. When the causal structure implies that certain shocks are d-separated from L given the current partial assignment, the effective action space is reduced.

**Required result.** A *causal UCB* acquisition function that (a) uses d-separation queries to identify and prune irrelevant arms, achieving regret O(|Σ_eff| log T) where |Σ_eff| ≤ |Σ| is the number of causally relevant arms, (b) provides a PAC guarantee: with probability ≥ 1-δ, the best arm found after T rollouts satisfies L(σ̂) ≥ max_σ L(σ) - ε, where T = O(|Σ_eff| / ε² · log(|Σ_eff|/δ)), and (c) is compatible with the non-stationary setting where bound estimates improve as more subgraphs are evaluated.

### 4. Streaming SMT Certificate Soundness

**Statement.** The streaming SMT protocol emits a sequence of incremental assertions a_1, a_2, ..., a_m during junction-tree message passing, where each a_t encodes a claim about a single message computation. The SMT solver maintains a satisfiability context C_t = C_{t-1} ∧ a_t.

**Required result.** A soundness theorem: if the SMT solver returns SAT for every prefix context C_1, C_2, ..., C_m, then the final inference result R is correct with respect to the SCM—formally, R = E_P[Y | do(X=x)] where P is the unique distribution satisfying the SCM's structural equations (or R ∈ B where B is the causal polytope bound when P is not unique). The key technical challenge is proving that the incremental protocol is equivalent to batch verification: that checking assertions one-at-a-time with incremental push/pop is sound even though intermediate contexts may be satisfiable for reasons unrelated to inference correctness. This requires showing that the assertion sequence is *monotone* in the sense that satisfiability of C_m implies correctness of the inference chain.

---

## BEST PAPER ARGUMENT

**Intellectual novelty.** No single prior work combines (a) causal polytope bounds on interventional queries in graphical models, (b) graph decomposition with a formal bound composition theorem, (c) streaming formal verification of probabilistic inference, and (d) causally-informed adversarial search. Each component draws on substantial existing theory—Balke-Pearl bounds (1997), Mauá-de Campos (credal network LP, 2012), Lauritzen-Spiegelhalter junction-tree (1988), de Moura-Bjørner (Z3, 2008), Kocsis-Szepesvári (UCT, 2006)—but their *composition* is novel and creates capabilities none of them achieves alone. The bound composition theorem (Section: New Mathematics #2) is a genuinely new mathematical result that enables tractable verified causal inference on networks too large for any single-subgraph approach.

**Community bridging.** This work sits at the intersection of four research communities that rarely interact: (1) *causal inference* (Pearl, Bareinboim, Tian—partial identification, do-calculus), (2) *formal verification* (SMT solving, certified computation, proof-carrying code), (3) *reinforcement learning* (MCTS, bandit theory, adversarial search), and (4) *computational finance* (systemic risk, network models, stress testing). A best paper in any one of these communities would require deep contribution to that community's core concerns. This work contributes to all four: a new composition theorem for causal bounds, a new application of streaming SMT to probabilistic inference, a new causal UCB variant with PAC guarantees, and the first formally verified systemic risk assessment tool. The bridging creates a "1+1+1+1 > 4" effect: the causal structure makes verification tractable (d-separation reduces the assertion space), verification makes the causal bounds trustworthy (certificates catch encoding errors), adversarial search makes the bounds practically relevant (finding worst cases), and the financial application motivates all three (systemic risk is too important for unverified point estimates).

**Compelling evaluation.** Unlike most systems papers in these communities, the evaluation requires zero human annotation and is fully reproducible. All metrics—bound tightness vs. Monte Carlo ground truth, contagion pathway recall on known-structure synthetic networks, MCTS discovery power vs. random/grid baselines, SMT overhead ratios, scalability curves—are computed automatically. The historical crisis similarity analysis (comparing discovered adversarial scenarios to structural features of the 2008, 2010, 2020, and 2023 crises) uses publicly available network topology data (BIS consolidated banking statistics, ECB money market surveys). No proprietary data, no human-labeled ground truth, no subjective quality assessments.

**Follow-on research.** This work opens at least three new research directions: (1) *verified causal inference* as a subfield—applying streaming SMT to other causal inference algorithms (IDA, do-calculus transport, causal fairness), (2) *decomposition-based causal bounds* as a methodology—extending the composition theorem to continuous variables, dynamic causal models, and non-DAG causal structures (cycles, feedback), and (3) *adversarial causal discovery*—using MCTS-style search to find DAG structures that maximize worst-case risk, rather than taking the DAG as given. Each of these is a multi-paper research program.

---

## EVALUATION PLAN

### Bound Tightness (Primary Metric)

Generate synthetic financial networks with known ground-truth SCMs (100 networks, 50–500 nodes, treewidth 5–25). For each, compute (a) the true causal polytope bound via brute-force LP on the full DAG (feasible only for networks with ≤ 80 nodes), (b) Monte Carlo estimates with 10M samples as approximate ground truth for larger networks, and (c) CausalBound's decomposed bounds. Report the *bound ratio* = |CausalBound interval| / |true interval|. Target: bound ratio ≤ 1.5 for tw ≤ 12, ≤ 2.5 for tw ≤ 20 (after decomposition). The gap directly measures the composition theorem's cost.

### Contagion Pathway Recall

On synthetic networks with planted contagion pathways (directed paths from shock source to loss variable with known intermediate amplification mechanisms), measure whether CausalBound's adversarial search discovers all planted pathways. Recall = (planted pathways found) / (total planted pathways). Target: recall ≥ 0.95 for networks with ≤ 200 nodes and ≤ 5 planted pathways. This validates that MCTS explores the relevant subspace.

### Adversarial Discovery Power

Compare MCTS-discovered worst-case loss against three baselines: (a) random shock sampling (uniform over interface variable space), (b) grid search over discretized interface variable space, (c) CMA-ES (covariance matrix adaptation evolution strategy, a strong black-box optimizer). Report the *discovery ratio* = MCTS worst-case loss / baseline worst-case loss. Target: discovery ratio ≥ 1.2 vs. random, ≥ 1.05 vs. CMA-ES. Also measure wall-clock time: MCTS should achieve comparable loss to CMA-ES in ≤ 50% of the time due to causal pruning.

### Verification Overhead

Measure wall-clock time for junction-tree inference with and without streaming SMT co-routine. Report the *overhead ratio* = verified time / unverified time. Target: overhead ratio ≤ 3.0× for networks with ≤ 200 nodes, ≤ 5.0× for networks with 200–500 nodes. Also report: number of SMT assertions per inference pass, assertion generation time vs. SMT solving time breakdown, and incremental vs. batch verification time comparison.

### Scalability Benchmarks

Measure wall-clock time for the full pipeline (decomposition + LP + inference + verification + MCTS) as a function of: (a) number of nodes (50, 100, 200, 300, 500), (b) treewidth before decomposition (5, 10, 15, 20, 25), (c) number of subgraphs after decomposition, (d) number of MCTS rollouts (1K, 10K, 100K). All on a single laptop CPU (target: Apple M2 or Intel i7-13700, 32GB RAM). Report wall-clock time, peak memory, and cache hit rates for memoized inference.

### Historical Crisis Structural Similarity

Reconstruct approximate network topologies for four historical crises using publicly available data: (a) 2008 GFC (CDS network among top 16 dealers, using DTCC aggregate data), (b) 2010 EU sovereign debt (sovereign-bank exposure network from EBA disclosures), (c) 2020 COVID (Treasury market network from TRACE data), (d) 2023 UK gilt (LDI-gilt-repo network from BoE Financial Stability Reports). Run CausalBound on each topology and compare discovered adversarial scenarios to known crisis mechanisms. Report structural similarity metrics: Jaccard similarity of contagion pathway edges, rank correlation of institution-level losses. This is not a prediction exercise (we do not claim to predict crises) but a validation that the tool discovers structurally similar contagion mechanisms when given similar network topologies.

### All Evaluations Fully Automated

Every metric above is computed by deterministic scripts with no human judgment. The evaluation harness generates synthetic networks, runs the pipeline, computes metrics, and produces tables and figures. Historical topology reconstruction uses fixed data extraction scripts applied to publicly available datasets. Zero human annotation at any point. Full reproducibility: random seeds fixed, all dependencies pinned, single `make eval` command.

---

## LAPTOP CPU FEASIBILITY

### Junction-Tree Inference

Junction-tree message passing is inherently sequential: messages propagate inward to a root and then outward, with each message depending on its children. GPU parallelism helps only within a single clique potential table multiplication, but for tw ≤ 12 with adaptive discretization (k = 4–6), each table has ≤ 6^12 ≈ 2.2B entries in the worst case. In practice, structured sparsity in financial instrument CPDs (CDS protection legs are step functions, IRS have piecewise-linear profiles) reduces effective table sizes by an estimated 10–100×, validated empirically during evaluation. SIMD vectorization (AVX-512 on x86, NEON on ARM) achieves near-peak throughput for the element-wise multiply-and-sum operations in message passing. A single inference pass on a subgraph with 50 nodes and tw = 10 completes in ~5ms on an M2 CPU. Memoization of unchanged subtree messages during MCTS (where only boundary variables change between rollouts) reduces amortized cost to ~2ms per rollout.

### LP Solving

The simplex method is fundamentally sequential: each pivot operation depends on the previous basis. Column generation adds an outer loop (also sequential: generate column, add to master, re-solve). Interior point methods offer some parallelism but are less effective for the sparse, structured LPs arising from causal polytope formulations. For subgraphs with 30–50 nodes and tw ≤ 12, the master LP has ~500–2,000 constraints and converges in ~200–1,000 column generation iterations. Each iteration takes ~1ms (simplex pivot) + ~2ms (pricing subproblem via dynamic programming on the junction tree). Total LP solve time per subgraph: ~1–3 seconds. With ~10–20 subgraphs for a 500-node network, total LP time: ~15–60 seconds.

### SMT Solving

The DPLL(T)/CDCL architecture of modern SMT solvers (Z3, CVC5) is inherently sequential: unit propagation, conflict analysis, and backtracking are sequential operations. The incremental assertion protocol (push/assert/check-sat/pop) amortizes solver state across assertions. For the QF_LRA theory (quantifier-free linear real arithmetic) used to encode LP bound claims, each check-sat call takes ~0.1–1ms for the small assertion sets generated by a single message passing step. With ~200–500 assertions per inference pass, total SMT time per pass: ~50–250ms. The streaming protocol ensures that the solver never faces the full assertion set at once, keeping individual queries small.

### MCTS Search

Tree search is sequential by nature: each rollout selects a path, evaluates a leaf, and backpropagates. The causal UCB acquisition function adds ~0.01ms per node (d-separation query on precomputed reachability tables). With memoized inference (~5ms per rollout including SMT verification, ~2ms amortized with subtree caching), 100K rollouts complete in ~200 seconds (~3.3 minutes amortized). The PAC guarantee for causal UCB requires T = O(|Σ_eff| / ε² · log(|Σ_eff|/δ)) rollouts; for |Σ_eff| ≈ 1,000 (after d-separation pruning from ~10,000 interface configurations), ε = 0.05, δ = 0.05, this gives T ≈ 100K—exactly our budget.

### Total Pipeline Budget (500-node network)

| Stage | Time | Memory |
|---|---|---|
| Graph decomposition | ~10s | ~500MB |
| LP solving (20 subgraphs) | ~60s | ~2GB peak |
| MCTS (100K rollouts) | ~200s | ~4GB (memoization cache) |
| SMT verification (amortized) | ~included in MCTS | ~1GB (assertion stack) |
| **Total** | **~270s (~4.5 min)** | **~8GB peak** |

For a 200-node network (more realistic for current regulatory topology data): ~90 seconds total, ~4GB peak memory. Comfortably within laptop CPU constraints (target: M2 MacBook Pro with 16GB RAM for 200-node, 32GB for 500-node networks).

---

## SLUG

`verified-causal-systemic-risk-engine`
