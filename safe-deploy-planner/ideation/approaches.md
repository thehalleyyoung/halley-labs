# SafeStep: Three Competing Technical Approaches

> **Project:** SafeStep — Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters
>
> **Problem Core:** Pre-compute rollback safety envelopes — a map of which intermediate deployment states admit safe rollback and which are irreversible points of no return — by modeling multi-service version-upgrade orchestration as bounded model checking over a version-product graph.
>
> **Context:** Problem crystallized 2026-03-07. Depth-check score 6.0/10 (CONDITIONAL CONTINUE). Oracle accuracy unvalidated (critical). Treewidth DP infeasible at tw≥5 (serious). ~115K realistic LoC.

---

## Approach A: SAT/BMC-First with Interval Compression

### A.1 Extreme Value Delivered

Every week, platform engineering teams at organizations running 20+ microservices face the same nightmare: they need to roll out a coordinated version upgrade across a constellation of interdependent services, and they have no formal guarantee that any intermediate state is safe — or that they can retreat if something goes wrong. Today's state of the art is a hand-drawn dependency graph on a whiteboard, a topological sort heuristic, and a prayer. Canary deployments catch forward bugs (the new version crashes) but are completely blind to *retreat blockage* — the scenario where rolling back service A requires first rolling back service B, which has already been upgraded past the point of compatibility with A's old version.

SafeStep Approach A delivers a **provably-safe minimum-step deployment schedule** together with a complete **rollback safety envelope** — a per-state annotation telling operators exactly when retreat becomes impossible and why. The envelope is the killer feature: no existing tool (Kubernetes, ArgoCD, Flux, Istio, Spinnaker) computes anything like it. For a 50-service cluster with 20 version candidates per service, the system produces this in 3–17 minutes on a laptop.

**Who desperately needs this:**
- **Platform engineering teams** at mid-to-large organizations (Shopify, Stripe, Airbnb scale) who manage 30–200 microservices and perform coordinated upgrades weekly. A single cross-service incompatibility incident costs $50K–$500K in engineering time and customer impact.
- **SRE incident commanders** who need to know, mid-incident, whether rollback is still a viable option or whether they must push forward to a known-safe state.
- **Regulated industries** (fintech, healthcare) where deployment safety must be auditable and explainable — SafeStep's stuck-configuration witnesses provide formal evidence for compliance.

### A.2 Why This Is Genuinely Difficult

This is not a straightforward application of existing model checking. The difficulty arises from four interacting hard subproblems:

**Subproblem 1: Exponential State Space with Implicit Representation.** The version-product graph for n services with L versions each has L^n states. For n=50, L=20, this is 20^50 ≈ 10^65. The entire graph can never be materialized. The BMC encoding must work with an *implicit* representation where states are bit-vector tuples and transitions are constraint conjunctions. The architectural challenge is building an incremental BMC unroller that adds one step at a time without re-encoding the entire formula, while maintaining the solver's learned clauses across increments.

**Subproblem 2: Constraint Oracle Construction.** The system's correctness is relative to an oracle that classifies pairwise version compatibility. This oracle must be automatically derived from schema artifacts (OpenAPI 3.x specs, Protocol Buffer definitions, GraphQL schemas, Avro schemas) by detecting breaking changes: removed fields, type changes, renamed endpoints, altered cardinality. The fundamental difficulty is that schema analysis captures *structural* incompatibility but misses *behavioral* incompatibility (semantic shifts, performance cliffs, error-handling changes). The depth check flagged this as the critical gate criterion: if <40% of real outages involve structural failures, the tool's value proposition collapses. The oracle must be both precise enough to avoid false positives (<5%) and complete enough to catch ≥60% of real incompatibilities.

**Subproblem 3: Interval Encoding Efficiency.** The key insight enabling tractability is that >92% of real compatibility predicates have *interval structure*: if service A at version v is compatible with service B at versions w₁ and w₃ where w₁ < w₃, then it is also compatible with all versions w₂ where w₁ ≤ w₂ ≤ w₃. This allows encoding compatibility as O(log² L) clauses per service pair per step (using binary-encoded interval predicates) instead of the naive O(L²). But the encoding is delicate: the binary arithmetic constraints must interact correctly with the solver's unit propagation, and non-interval predicates (the remaining ~8%) must be handled via BDD fallback without blowing up clause count.

**Subproblem 4: Bidirectional Reachability for Envelope Computation.** The rollback safety envelope requires computing, for each state on the deployment plan, whether a safe path exists *back* to the starting configuration. This is a second reachability problem (backward from current state to start) that must be solved for every state on the forward plan. Naively, this multiplies the solving time by the plan length. The algorithmic challenge is amortizing backward reachability across plan states using incremental solving with assumption literals, so that adding one more plan state to the backward check costs only marginal solver work.

**Architectural challenges:**
- **Solver integration:** CaDiCaL for pure SAT (incremental mode with assumption-based activation), Z3 for resource constraints requiring linear integer arithmetic. The two solvers must cooperate via a CEGAR (counterexample-guided abstraction refinement) loop: CaDiCaL finds candidate plans, Z3 checks resource feasibility, conflicts feed back as blocking clauses.
- **Kubernetes integration:** Parsing real Helm charts and Kustomize overlays to extract version constraints and resource requirements. The depth check mandated using `helm template` as a subprocess (not reimplementing Helm's Go templating in Rust), which introduces subprocess management complexity and output parsing brittleness.
- **Treewidth-guided fast path:** For service dependency graphs with treewidth ≤ 3 and L ≤ 15, a dynamic-programming algorithm on the tree decomposition yields >10× speedup over flat BMC. Computing the tree decomposition and detecting when to use this fast path adds algorithmic complexity. The depth check noted that the original claim of feasibility at tw≤5 was overstated; the realistic cutoff is tw≤3 due to the L^{2(w+1)} factor.

### A.3 New Math Required

**Theorem (Monotone Sufficiency).** *If the compatibility predicate is downward-closed (i.e., if version v of service i is compatible with version w of service j and v' ≤ v, then v' is also compatible with w), then every safe deployment plan can be transformed into a monotone plan (one that never downgrades any service) of equal or shorter length.*

This theorem is **load-bearing**: it collapses the search space from all possible transition sequences (including rollbacks as search moves) to only monotonically increasing sequences. Without it, the BMC horizon would need to account for arbitrary back-and-forth, making the completeness bound exponential rather than linear in ∑ᵢ(goalᵢ − startᵢ).

**Proof sketch:** Given a safe plan π with a downgrade step (service i goes from v to v' < v), construct π' by deleting the downgrade and all subsequent re-upgrades of i back to v. Downward closure ensures every state visited by π' is safe if the corresponding state in π was safe. The plan length decreases by at least 2 per eliminated downgrade.

**Theorem (Interval Encoding Compression).** *If the compatibility predicate C(i,j,v,w) has interval structure for all service pairs (i,j), then the BMC transition relation for one step can be encoded in O(n² · log² L) clauses, where n is the number of services and L is the maximum version count.*

This is **load-bearing** because it determines whether the SAT instance fits in solver memory. The naive encoding produces O(n² · L²) clauses per step; for n=50, L=20, k=200, that's 2 billion clauses (infeasible). The interval encoding reduces this to ~9.3M clauses (feasible for CaDiCaL).

**Construction:** Encode each service's version as a ⌈log₂ L⌉-bit binary variable. For interval-structured compatibility C(i,j,v,·) = [lowᵢⱼ(v), highᵢⱼ(v)], encode the constraint "w ∈ [low, high]" as two binary comparisons: w ≥ low ∧ w ≤ high. Each binary comparison requires O(log L) clauses via standard bit-vector encoding. The conjunction over all service pairs yields O(n² · log² L) total clauses.

**Theorem (BMC Completeness Bound).** *The minimum plan length is at most k* = ∑ᵢ(goalᵢ − startᵢ), and if no plan of length ≤ k* exists, then no plan exists at all.*

This converts BMC from a semi-decision procedure to a complete decision procedure. Without it, the system would need to search to an unknown depth or declare "timeout" without certainty.

**Theorem (Treewidth Tractability).** *If the service dependency graph has treewidth w, then deployment plan existence can be decided in O(n · L^{2(w+1)} · k*) time via dynamic programming on a tree decomposition.*

This is load-bearing for the fast-path optimization, but the depth check correctly noted that L^{2(w+1)} becomes infeasible at w≥4 for L=20: 20^10 = 10^13, which is beyond practical DP table size. The theorem remains valid but the practical applicability window is narrow (w≤3, L≤15).

### A.4 Best-Paper Potential

If pulled off completely, Approach A has best-paper potential for three reasons:

1. **The rollback safety envelope is a genuinely new operational concept.** No prior work in deployment orchestration, SDN consistent updates, or configuration synthesis computes anything equivalent. The envelope transforms deployment from a "hope for the best" exercise into a formally characterized risk landscape. This is the kind of conceptual contribution that reviewers at SOSP/OSDI value — a new way of thinking about a real problem, not just a faster algorithm for an old formulation.

2. **The BMC encoding combines multiple non-obvious reductions.** Monotone sufficiency, interval compression, treewidth decomposition, and replica symmetry reduction each contribute independently to tractability, but their *composition* is what makes the system work at scale. Showing that these reductions compose cleanly (i.e., monotonicity doesn't break interval encoding, treewidth decomposition doesn't break replica symmetry) requires careful formal argument.

3. **The evaluation connects formal methods to production incidents.** Reconstructing 15 real postmortem incidents as SafeStep inputs and showing that 13+/15 would have been caught is a compelling empirical story. If the oracle validation experiment shows ≥60% structural coverage, the paper bridges the gap between formal methods and systems practice that most FM papers fail to cross.

**Estimated publication probability (with all amendments):** SOSP/OSDI 35–40%, EuroSys/NSDI 60–65%, best-paper at any top venue 8–12%.

### A.5 Hardest Technical Challenge

The hardest challenge is **oracle validation and the model-reality gap**. The entire system's value depends on the constraint oracle being accurate enough that the computed safety envelopes reflect reality. If the oracle misses a critical incompatibility (false negative), the "safe" plan may lead to an outage. If the oracle is too conservative (false positive), it may declare no plan exists when one does.

**How to address it:**
1. **Incident classification experiment:** Take 15 real postmortem reports involving multi-service version incompatibility. For each, classify the root cause as structural (detectable by schema analysis) vs. behavioral (requires runtime observation). If ≥60% are structural, the oracle is viable; otherwise, reposition the work.
2. **Oracle confidence levels:** Instead of binary compatible/incompatible, assign confidence levels to each compatibility judgment. Schema removal = high confidence incompatible. Type widening = medium confidence compatible. Unknown behavioral change = low confidence. Feed confidence levels into the BMC encoding as soft constraints, producing plans that are robust to oracle uncertainty.
3. **Runtime validation feedback loop:** After deployment, observe actual compatibility outcomes and feed them back to refine the oracle. This creates a learning loop that improves over time, but the first deployment must rely on schema analysis alone.
4. **Conservative mode:** Default to assuming incompatibility for any version pair where the oracle lacks evidence. This maximizes safety (no false negatives) at the cost of potentially rejecting valid plans (more false positives). Operators can selectively override with manual annotations for specific version pairs they have tested.

### A.6 Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9 | Rollback safety envelopes are genuinely novel and practically critical; direct industry application |
| **Difficulty** | 8 | Multiple interacting hard subproblems; oracle validation is a genuine research question; encoding efficiency requires careful engineering |
| **Potential** | 7 | Strong conceptual novelty (envelope) but oracle limitation caps the impact ceiling; P(SOSP)≈35–40% |
| **Feasibility** | 7 | Core BMC machinery is well-understood; interval encoding and monotone reduction are tractable; oracle validation is the main risk |

---

## Approach B: Abstract Interpretation + Symbolic Deployment Semantics

### B.1 Extreme Value Delivered

Approach A's BMC engine is fundamentally a *path-finding* approach: it searches for a specific sequence of deployment steps that satisfies all constraints. This works well for finding *one* safe plan, but it gives limited insight into the *structure* of the safe deployment space. Operators don't just want a single plan — they want to understand the *topology of safety*: which regions of the version-product space are universally safe, which are universally unsafe, and where the boundaries lie. They want compositional reasoning: if I know service A can safely upgrade from v1 to v3 in isolation, and service B can safely upgrade from v2 to v5 in isolation, can I compose these into a joint plan?

Approach B delivers a **complete safety map** of the deployment state space, computed via abstract interpretation. Instead of finding paths in a concrete graph, it computes fixpoints of abstract transfer functions that over-approximate the set of reachable safe states (forward) and the set of states from which rollback is safe (backward). The rollback safety envelope emerges naturally as the intersection of the forward and backward fixpoints.

**Who desperately needs this:**
- **Platform teams managing continuous deployment pipelines** who upgrade multiple times daily and need *pre-computed* safety maps that can be queried in milliseconds at deployment time, rather than re-running a solver for each upgrade.
- **Multi-tenant platform providers** (e.g., cloud PaaS, managed Kubernetes) who manage thousands of customer clusters with similar but not identical service topologies. Abstract interpretation's compositional properties allow computing safety maps for service *archetypes* and instantiating them per-tenant.
- **Chaos engineering teams** who want to understand the *shape* of the unsafe region: not just "this plan is unsafe" but "these 47 states form an unsafe pocket surrounded by safe states, and here's the minimal boundary."

**Extreme value:** Approach B computes a **deployment safety atlas** — a symbolic representation of all safe, unsafe, and envelope-boundary states — that can be queried in O(1) per state after an upfront fixpoint computation. This enables real-time deployment decisions: at each step, the operator queries the atlas to check whether the next proposed action keeps them inside the envelope. The atlas also supports *what-if* queries: "If I add a new service C at version v, which existing plans become unsafe?"

### B.2 Why This Is Genuinely Difficult

Abstract interpretation over deployment states is a novel application of a well-studied framework, and the difficulty lies in designing abstract domains that are simultaneously precise enough to be useful and abstract enough to converge quickly.

**Subproblem 1: Abstract Domain Design.** The concrete domain is the powerset of V = ∏ᵢ Vᵢ (sets of version-product states). A naive abstract domain (e.g., intervals per service) loses all relational information: it cannot express "service A is at v3 AND service B is at v5" as distinct from "service A is at v3 OR service B is at v5." A fully relational domain (e.g., polyhedra over version indices) has exponential cost. The challenge is designing a *deployment-specific* abstract domain that captures the essential relational structure of compatibility constraints while remaining tractable. The key insight is that compatibility constraints are *pairwise* — they relate exactly two services at a time — which suggests a relational domain structured as a conjunction of per-pair abstractions.

**Proposed abstract domain: Pairwise Interval Zonotopes (PIZ).** Represent the abstract state as a conjunction of per-service interval constraints (service i's version ∈ [lᵢ, hᵢ]) augmented with per-pair compatibility zones. Each compatibility zone is a 2D region in the (vᵢ, vⱼ) plane representing compatible version combinations. The PIZ domain is closed under intersection and join (convex hull in each 2D projection), and widening is per-pair interval widening. The precision loss from per-pair projection is bounded by the structure of real compatibility constraints, which are empirically pairwise.

**Subproblem 2: Widening for Convergence.** The ascending chain in the abstract domain must converge in finite steps, which requires widening operators. Standard interval widening (jump to ±∞ on the second iteration) is too aggressive for version spaces that are bounded [1, L]. A deployment-specific widening operator must respect the discrete, bounded nature of version spaces while still guaranteeing convergence. The proposed approach: *threshold widening* with thresholds at observed version boundaries in the constraint oracle.

**Subproblem 3: Backward Transfer Functions for Rollback.** Computing the rollback envelope requires a *backward* abstract interpretation: starting from the set of all states, compute the greatest fixpoint of the backward transfer function (predecessor states from which the current abstract set is reachable). Backward abstract interpretation is more delicate than forward because the backward transfer function is the *inverse* of the forward function, which may not be expressible precisely in the abstract domain. The PIZ domain's pairwise structure helps: backward transfer through a pairwise compatibility constraint is still a pairwise operation.

**Subproblem 4: Compositional Safety Analysis.** The major architectural advantage of abstract interpretation is *compositionality*: analyze each service pair independently, then combine results. But compositionality introduces a precision gap: the independent per-pair analysis may certify a state as safe when the *joint* constraints render it unsafe. This is the classic "relational vs. independent attribute" problem in abstract interpretation. The challenge is bounding this precision gap and detecting when it matters. The proposed approach: compute the per-pair fixpoints, combine them, and then run a *reduced product* refinement pass that iteratively tightens each pair's constraints using information from other pairs until convergence.

**Subproblem 5: Constraint Oracle Integration (Same as Approach A).** The oracle problem is identical: schema analysis provides the compatibility predicate, and its accuracy determines the system's value. However, abstract interpretation has a natural advantage for handling oracle uncertainty: the abstract domain can encode *may-compatible* and *must-compatible* states separately, providing sound over-approximation (every genuinely safe state is marked safe) at the cost of possible imprecision (some actually-safe states may be marked "unknown").

**Architectural challenges:**
- **Fixpoint engine:** Must handle domains with O(n²) pairwise components, each a 2D discrete region. The fixpoint iteration must converge in O(L) iterations (bounded by version space size), with each iteration costing O(n² · L²) for pairwise constraint propagation.
- **Kubernetes integration:** Same as Approach A — `helm template` subprocess, OpenAPI/protobuf parsing, CRD extraction.
- **Query interface:** The computed safety atlas must support constant-time membership queries (is state s in the envelope?) and efficient what-if queries (how does adding service C affect the atlas?).

### B.3 New Math Required

**Definition (Pairwise Interval Zonotope Domain).** Let S = {1, ..., n} be the set of services, Vᵢ = {1, ..., Lᵢ} the version set for service i. Define the PIZ abstract element as:

$$a = \langle (I_i)_{i \in S}, (Z_{ij})_{(i,j) \in E} \rangle$$

where $I_i = [l_i, h_i] \subseteq V_i$ is a per-service interval, $E$ is the set of dependent service pairs, and $Z_{ij} \subseteq V_i \times V_j$ is the compatible zone for pair $(i,j)$.

The concretization function is:

$$\gamma(a) = \{ v \in \prod_i V_i \mid \forall i: v_i \in I_i \wedge \forall (i,j) \in E: (v_i, v_j) \in Z_{ij} \}$$

This is **load-bearing** because it defines the precision of the analysis. The per-pair zones $Z_{ij}$ capture exactly the relational information that pairwise compatibility constraints encode, without the exponential cost of a fully relational domain.

**Theorem (PIZ Galois Connection).** *The pair $(\alpha, \gamma)$ where $\alpha(S) = \langle (\text{proj}_i(S))_{i}, (\text{proj}_{ij}(S))_{(i,j)} \rangle$ forms a Galois connection between $\mathcal{P}(\prod_i V_i)$ and the PIZ domain, ordered by componentwise inclusion.*

This is **load-bearing** because it guarantees soundness: any property proved in the abstract domain holds in the concrete domain. The Galois connection ensures that the abstract fixpoint over-approximates the concrete fixpoint, so every state marked "safe" by the abstract analysis is genuinely safe.

**Theorem (Bounded Convergence).** *With threshold widening at version boundaries, the ascending Kleene iteration in the PIZ domain converges in at most $2 \cdot |E| \cdot \max_i L_i$ steps.*

This is **load-bearing** because it guarantees termination and bounds the computation time. Without a convergence bound, the fixpoint computation might not terminate or might take exponentially many iterations.

**Proof sketch:** Each widening step increases at least one interval endpoint or zone boundary by at least one version. There are $2n$ interval endpoints and $2|E| \cdot \max_i L_i$ zone boundaries. Each can increase at most $\max_i L_i$ times. The product $2|E| \cdot \max_i L_i$ bounds the total iterations.

**Theorem (Compositional Envelope Soundness).** *Let $F^{\uparrow}$ be the forward fixpoint (reachable safe states from start) and $B^{\downarrow}$ be the backward fixpoint (states from which start is reachable through safe states). Then $\gamma(F^{\uparrow}) \cap \gamma(B^{\downarrow}) \subseteq \text{Envelope}_{\text{concrete}}$. Moreover, $\gamma(F^{\uparrow}) \setminus \gamma(B^{\downarrow})$ over-approximates the set of points of no return.*

This is the **central soundness theorem** for the envelope computation. It says the abstract envelope is a sound over-approximation: every state the abstract analysis identifies as "inside the envelope" is genuinely inside (no false positives for safety). Points of no return may have false positives (states marked as points of no return that actually admit rollback), which is the safe direction for conservative analysis.

### B.4 Best-Paper Potential

Approach B has best-paper potential for different reasons than Approach A:

1. **Novel application of abstract interpretation to deployment safety.** Abstract interpretation has been applied to program analysis (Astrée, SPARTA, Infer), network verification (Batfish, Minesweeper), and hardware verification, but never to deployment orchestration. The PIZ domain is specifically designed for deployment constraints and has no analog in the AI literature. A well-crafted POPL/PLDI submission could position this as "abstract interpretation meets DevOps" — a new application domain for a mature theoretical framework.

2. **Compositional analysis enables scalability claims that BMC cannot match.** If the per-pair analysis is fast and the reduced product converges quickly, Approach B could handle n=500+ services by analyzing O(n²) pairs independently and combining results. BMC (Approach A) is fundamentally limited by solver scalability; abstract interpretation's scaling is determined by domain size, not formula size.

3. **The safety atlas concept is richer than a single plan + envelope.** Instead of answering "is this plan safe?" (a yes/no question), the atlas answers "what is the shape of the safe region?" This enables downstream applications (chaos engineering, capacity planning, multi-tenant platform management) that BMC-first cannot support.

4. **The precision-vs-speed tradeoff is tunable.** By adjusting the abstract domain (e.g., adding 3-way zones for service triples, or projecting to per-service intervals for speed), operators can choose their precision/performance tradeoff. This parameter-space exploration is attractive for an evaluation section.

**Estimated publication probability:** POPL/PLDI 25–35% (as PL venue), SOSP/OSDI 25–30% (less mature systems story), EuroSys/NSDI 50–55%, best-paper 5–10%.

### B.5 Hardest Technical Challenge

The hardest challenge is **precision of the pairwise abstraction**. The PIZ domain captures pairwise relationships but loses higher-order correlations: it cannot express "services A, B, C must all be at version ≥3 simultaneously." In practice, most compatibility constraints are pairwise (service A's API compatibility with service B), but resource constraints (total CPU/memory across all services ≤ cluster capacity) are inherently n-ary. The PIZ domain will over-approximate resource constraints, potentially marking some actually-safe states as "unknown."

**How to address it:**
1. **Reduced product with resource domain.** Add a second abstract domain specifically for resource constraints: a polyhedron or octagon domain over resource sums. The reduced product of PIZ (for compatibility) and octagon (for resources) captures both pairwise compatibility and aggregate resource bounds. The reduced product refinement iteratively tightens both domains.
2. **Targeted concretization.** When the abstract analysis produces "unknown" regions, extract the corresponding concrete states and check them with the BMC engine (Approach A's solver). This hybrid uses abstract interpretation for the easy 90% and BMC for the hard 10%.
3. **Empirical precision measurement.** On small instances (n≤15) where exact computation is feasible, measure the gap between abstract and concrete envelopes. If the gap is <10% of states, the precision is adequate for practice.

### B.6 Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Safety atlas is richer than single plan + envelope; compositional analysis enables new use cases; but operators may prefer a concrete plan over an abstract map |
| **Difficulty** | 9 | Novel abstract domain design; convergence proofs; reduced product engineering; less established than BMC for this problem class |
| **Potential** | 8 | Novel AI × systems crossover; compositional scalability is a strong differentiator; but PL reviewers may find the domain simple and systems reviewers may find the abstraction unfamiliar |
| **Feasibility** | 5 | Abstract domain design is high-risk: precision may be inadequate for real constraints; reduced product convergence is uncertain; no existing codebase to build on (unlike SAT solvers for Approach A) |

---

## Approach C: Game-Theoretic Adversarial Planning

### C.1 Extreme Value Delivered

Approaches A and B share a fundamental assumption: the constraint oracle is correct. If the oracle says versions (A:v3, B:v5) are compatible, the plan treats them as compatible. But the depth check identified oracle accuracy as the critical gate criterion, and real-world experience teaches us that compatibility oracles are always incomplete. Schema analysis misses behavioral incompatibilities, performance cliffs, and subtle semantic shifts. What operators truly need is not a plan that's safe *assuming the oracle is perfect*, but a plan that's safe *even when the oracle is wrong about some fraction of its judgments*.

Approach C reframes deployment planning as a **two-player game**:
- **The Deployer** (player 1) chooses which service to upgrade at each step, with the goal of reaching the target configuration.
- **The Adversary** (player 2, representing the environment/reality) chooses which oracle-uncertain compatibility judgments to violate, within a budget of k disruptions.

A deployment plan is *robustly safe* if it is a **winning strategy for the Deployer** against all possible Adversary moves — i.e., it reaches the target and maintains rollback safety regardless of which k oracle judgments are wrong. The rollback safety envelope becomes a *game-theoretic* concept: a state is inside the envelope if the Deployer has a winning rollback strategy against the Adversary from that state.

**Who desperately needs this:**
- **Organizations that have been burned by oracle failures.** Every company that has experienced an outage caused by a compatibility issue that "should have been caught" by schema analysis. The game-theoretic formulation explicitly models oracle uncertainty and provides robustness guarantees.
- **High-stakes deployment environments** (healthcare systems, financial infrastructure, air traffic control) where the cost of a single unsafe deployment is catastrophic and "the oracle is probably right" is not an acceptable safety argument. These environments need worst-case guarantees, which is exactly what game-theoretic planning provides.
- **Teams adopting SafeStep incrementally.** The Adversary budget k parameterizes robustness: k=0 reduces to Approach A (oracle is trusted), k=1 provides robustness against a single oracle error, and k=3 provides robustness against three simultaneous errors. Teams can start with k=0 and increase as they gain confidence in the tool.

**Extreme value:** A **robust deployment plan with quantified oracle uncertainty tolerance.** Instead of "this plan is safe (assuming the oracle is correct)," SafeStep Approach C says "this plan is safe even if up to 3 of the oracle's compatibility judgments are wrong." This is a fundamentally stronger guarantee that directly addresses the depth check's most critical concern.

### C.2 Why This Is Genuinely Difficult

Game-theoretic deployment planning introduces computational challenges qualitatively different from Approaches A and B.

**Subproblem 1: Game Tree Explosion.** The deployment game tree has branching factor O(n) for the Deployer (choice of which service to upgrade) and O(m) for the Adversary (choice of which compatibility judgment to violate), where m is the number of uncertain oracle entries. At each of the k* deployment steps, both players make a move, yielding a game tree of size O((n · m)^{k*}). For n=50, m=200 (uncertain entries), k*=100, this is astronomically large. The tree cannot be explicitly enumerated.

**Subproblem 2: Information Structure.** The game's information structure matters critically. If the Deployer can observe which compatibility judgments the Adversary has violated (i.e., failed deployments are detected immediately), the game is one of *perfect information* and admits minimax solutions. If violations are *silent* (a service appears to work but will fail under load), the game has *imperfect information* and requires more sophisticated solution concepts (behavioral strategies, belief updates). The realistic model is somewhere between: some incompatibilities are immediately observable (service crashes on startup) while others are latent (performance degradation under load). The proposed approach: model the game as **partial-information** with a detection delay parameter δ (incompatibilities detected within δ steps), and show that the minimax solution under perfect information provides a sound over-approximation of the partial-information solution.

**Subproblem 3: Oracle Uncertainty Quantification.** The Adversary's budget k must be justified: how many oracle judgments can realistically be wrong? This requires a probabilistic model of oracle accuracy. The proposed approach: assign each compatibility judgment a *confidence score* based on the evidence strength (schema removal = high confidence, type widening = medium, no evidence = low). The Adversary's budget is drawn from the pool of low-confidence judgments. This transforms the game from worst-case (Adversary can violate anything) to *distributional* (Adversary is constrained by evidence strength).

**Subproblem 4: Efficient Game Solving.** Even with perfect information, solving the deployment game exactly is EXPTIME-complete (it's a reachability game on an exponential state space). The system needs efficient approximation algorithms:
- **Alpha-beta pruning** for exact solving of small game trees (n≤15, k≤2).
- **Monte Carlo Tree Search (MCTS)** for approximate solutions at scale: randomly sample Adversary moves according to the confidence-weighted distribution, build a search tree incrementally, and use UCB1 (Upper Confidence Bound) for exploration-exploitation balance.
- **Strategy summarization:** Instead of storing the full strategy (exponential size), represent the Deployer's strategy as a *policy*: a function from current state × observed violations → next action. Parameterize the policy as a decision tree or neural network trained on MCTS rollouts.

**Subproblem 5: Robust Envelope Computation.** The rollback safety envelope under adversarial uncertainty is: state s is inside the robust envelope if the Deployer has a winning rollback strategy from s against the Adversary with remaining budget k' (where k' ≤ k minus violations already observed). Computing this requires solving a *family* of games parameterized by remaining budget, which multiplies the computation by the budget parameter k.

**Architectural challenges:**
- **Game engine architecture:** The game solver must maintain a game tree (or MCTS tree) alongside the version-product graph. Memory management is critical: the game tree can easily exceed available RAM. The proposed approach: store only the MCTS tree (compact) and reconstruct game tree nodes on-the-fly from the implicit version-product graph.
- **Parallelization:** MCTS rollouts are embarrassingly parallel. The game solver should use a root-parallel MCTS architecture with virtual loss to avoid thread contention, targeting 8-16 core utilization.
- **Integration with BMC (hybrid).** For the inner loop (checking whether a specific Deployer action is safe given a specific Adversary move), use the BMC engine from Approach A. This creates a *hierarchical solver*: MCTS at the outer level selects moves, BMC at the inner level verifies safety of individual states. The BMC engine must be extremely fast for single-state safety checks (target: <100ms per check).

### C.3 New Math Required

**Definition (Deployment Game).** A deployment game is a tuple $\mathcal{G} = (S, s_0, t, \text{Act}_D, \text{Act}_A, \delta, \text{Safe}, k)$ where:
- $S = \prod_i V_i$ is the state space (version-product states)
- $s_0$ is the initial configuration, $t$ is the target
- $\text{Act}_D = \{(i, v) \mid i \in [n], v \in V_i\}$ is the Deployer's action set (upgrade service i to version v)
- $\text{Act}_A \subseteq E \times V \times V$ is the Adversary's action set (declare that edge $(i,j)$ with versions $(v,w)$ is actually incompatible)
- $\delta: S \times \text{Act}_D \times \text{Act}_A \to S$ is the transition function
- $\text{Safe} \subseteq S$ is the safety predicate (under current oracle beliefs minus adversary violations)
- $k$ is the Adversary's disruption budget

This is **load-bearing** because it formally defines the problem that all subsequent algorithms solve. The key design choice is making the Adversary's budget $k$ explicit and parameterizable.

**Theorem (Robust Safety Reduction).** *The Deployer has a winning strategy in the deployment game $\mathcal{G}$ with budget $k$ if and only if there exists a deployment plan that remains safe under every subset of $\leq k$ oracle compatibility judgments being simultaneously wrong.*

This is **load-bearing** because it connects the game-theoretic formulation to the operational meaning of "robust deployment plan." It justifies the game model as a sound abstraction of oracle uncertainty.

**Proof:** Forward direction: a winning strategy, by definition, handles all Adversary moves (each being a choice of which judgments to violate). Backward direction: given a plan safe under all violation subsets, the Deployer strategy that follows this plan wins regardless of Adversary moves, because each Adversary move corresponds to choosing a violation subset.

**Theorem (MCTS Convergence for Deployment Games).** *Under the UCB1 selection policy with exploration parameter $c = \sqrt{2}$ and uniformly random rollouts, MCTS applied to the deployment game converges to the minimax value as the number of rollouts $N \to \infty$. The convergence rate is $O((\log N / N)^{1/2})$ for the root value estimate.*

This is **load-bearing** because it guarantees that the MCTS approximation improves with computation time and eventually finds the correct answer. The convergence rate determines how many rollouts are needed for a given accuracy guarantee.

**Proof sketch:** Follows from the standard MCTS convergence proof (Kocsis & Szepesvári, 2006), adapted to the finite-horizon, deterministic, perfect-information deployment game. The key adaptation is showing that the deployment game satisfies the bounded-reward and finite-branching conditions required by the convergence theorem.

**Theorem (Adversary Budget Bound from Oracle Confidence).** *If each oracle judgment has independent error probability $p_e(i,j,v,w)$ derived from schema evidence strength, and we set the budget $k = \lceil \sum_{(i,j,v,w) \in \text{relevant}} p_e(i,j,v,w) + z_\alpha \cdot \sqrt{\sum p_e(1-p_e)} \rceil$ (one-sided $\alpha$-confidence bound on total errors), then the robust plan is safe with probability $\geq 1 - \alpha$ under the independent-error model.*

This is **load-bearing** because it provides a principled way to set the Adversary budget from empirical oracle accuracy data, connecting the game-theoretic abstraction to measurable quantities. Without this theorem, the budget $k$ is an arbitrary parameter.

### C.4 Best-Paper Potential

Approach C has best-paper potential for reasons distinct from both A and B:

1. **Directly addresses the depth check's most critical concern.** Oracle accuracy was flagged as the critical gate criterion. Approaches A and B treat the oracle as a fixed input and punt on accuracy. Approach C *incorporates oracle uncertainty into the model itself* and provides quantifiable robustness guarantees. This turns a weakness into a strength: instead of "our system assumes the oracle is correct (and we hope it is)," the paper says "our system is provably robust to k oracle errors, and here's how to set k from your confidence in the oracle."

2. **Game-theoretic modeling of deployment is novel in systems.** Game theory has been applied to security (attack-defense games), networking (congestion games), and mechanism design, but not to deployment orchestration. The framing of deployment-as-a-game is a conceptual contribution that may inspire follow-on work.

3. **The robust envelope concept is strictly stronger than the basic envelope.** A robust envelope (safe under adversarial oracle errors) subsumes the basic envelope (safe under correct oracle) as the special case k=0. This gives the paper a clean "generalization" story: Approach A is a special case of Approach C with zero adversarial budget.

4. **MCTS + BMC hybrid is an interesting algorithmic contribution.** Using MCTS for outer-level move selection and BMC for inner-level safety verification is a novel architecture that could be applied to other planning-under-uncertainty problems.

**Estimated publication probability:** SOSP/OSDI 30–35% (strong systems framing + novel formulation), AAAI/IJCAI 35–40% (as AI planning contribution), EuroSys/NSDI 45–50%, best-paper 10–15% (high novelty, high risk).

### C.5 Hardest Technical Challenge

The hardest challenge is **computational cost of the game-theoretic approach at scale**. The game tree's branching factor is multiplicative in the Deployer's and Adversary's action counts, and MCTS convergence is slow for games with large branching factors. For n=50 services with m=200 uncertain oracle entries and budget k=3, a single MCTS rollout touches O(k* · (n + C(m,k))) ≈ 100 · (50 + 1.3M) ≈ 130M nodes. Even with fast inner-loop BMC checks (100ms each), a single rollout takes ~3.6 hours. This is infeasible.

**How to address it:**

1. **Adversary action pruning.** Most oracle entries have high confidence (schema removal, type change). Only low-confidence entries (say, 20 out of 200) are realistic Adversary targets. This reduces C(m,k) from C(200,3) ≈ 1.3M to C(20,3) = 1,140 — a 1,000× reduction.

2. **Monotone reduction for the Deployer.** Apply Theorem 2 (Monotone Sufficiency) to restrict the Deployer's moves to monotonically increasing upgrades, reducing branching factor from O(n·L) to O(n) (choose which service to upgrade by one version).

3. **Lazy BMC evaluation.** Don't run full BMC for every inner-loop safety check. Instead, maintain a *safety cache*: a set of states already known to be safe or unsafe. Only invoke BMC for cache misses. Since MCTS rollouts share many states (they diverge late in the game), the cache hit rate should be high (>90% after warm-up).

4. **Stratified game solving.** Solve a hierarchy of games with increasing budget: first solve for k=0 (standard planning, fast), then use the k=0 solution to warm-start k=1, then k=2, etc. Each level inherits the previous level's strategy as a starting point, dramatically reducing MCTS exploration.

5. **Parallel MCTS.** Distribute rollouts across 8–16 cores with root parallelism and virtual loss. Target: 10,000 rollouts in <30 minutes for n=50, pruned adversary (20 entries), k=3.

6. **Budget-k abstraction.** For large k, approximate the game by clustering adversary actions into "disruption classes" (e.g., "any one of the 5 database-layer compatibility judgments is wrong") and solving the game over classes rather than individual judgments. This reduces branching factor at the cost of some conservatism (the robust plan may be overly cautious).

### C.6 Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 10 | Directly addresses the oracle accuracy problem (the project's critical weakness); robust guarantees are strictly stronger than basic safety; parameterizable risk tolerance |
| **Difficulty** | 10 | Game tree explosion; MCTS convergence at scale; hybrid MCTS+BMC architecture; information structure modeling; oracle confidence calibration |
| **Potential** | 9 | Novel formulation (deployment-as-game); directly addresses reviewer concern (oracle accuracy); generalizes Approach A; high novelty across both systems and AI venues |
| **Feasibility** | 4 | Computational cost is severe; MCTS convergence may be too slow for practical deployment timelines (3–17 min target); game-theoretic machinery is complex to implement and debug; high risk of delivering a theoretically interesting but practically unusable system |

---

## Comparative Summary

| Dimension | A: SAT/BMC | B: Abstract Interp. | C: Game-Theoretic |
|-----------|-----------|---------------------|-------------------|
| **Core Engine** | CaDiCaL/Z3 SAT solver | Fixpoint iteration on PIZ domain | MCTS + inner BMC |
| **Output** | Single optimal plan + envelope | Complete safety atlas | Robust plan + robust envelope |
| **Oracle Treatment** | Trusted input | Trusted input (may/must for uncertainty) | Adversarial model of uncertainty |
| **Scalability Limit** | n≈200 (solver-limited) | n≈500 (compositional) | n≈50 (game tree-limited) |
| **Time Budget** | 3–17 min (laptop) | 1–10 min (fixpoint fast) | 30–120 min (MCTS slow) |
| **Precision** | Exact (complete for BMC bound) | Sound over-approximation | Exact for solved games; approx for MCTS |
| **Novel Math** | Monotone sufficiency, interval encoding, completeness bound | Galois connection for PIZ, convergence bound, compositional soundness | Robust safety reduction, MCTS convergence, budget bound from oracle confidence |
| **Best Venue** | SOSP/OSDI (systems) | POPL/PLDI (PL) or SOSP (systems) | SOSP/OSDI (systems) or AAAI (AI) |
| **Value** | 9 | 8 | 10 |
| **Difficulty** | 8 | 9 | 10 |
| **Potential** | 7 | 8 | 9 |
| **Feasibility** | 7 | 5 | 4 |
| **Composite** | **7.75** | **7.50** | **8.25** |

### Recommendation

**Approach A** is the highest-feasibility path with strong value and well-understood algorithmic foundations. It should be the primary implementation target.

**Approach C** has the highest potential payoff because it directly addresses the oracle accuracy problem — the project's Achilles' heel. If computational challenges can be solved (via pruning, caching, and stratified solving), it yields a strictly stronger result. Consider implementing the game-theoretic layer as an *extension* of Approach A's BMC engine, activated when operators request robust plans.

**Approach B** is the most intellectually distinctive (abstract interpretation in a new domain) and scales best, but its precision-vs-abstraction tradeoff is the riskiest unknown. Consider it for a follow-on publication targeting PL venues, using Approach A's BMC engine as the ground-truth comparator.

**Hybrid strategy:** Implement Approach A as the core engine. Add Approach C's adversarial layer as an optional robust-planning mode. Use Approach B's compositional framework for pre-computation of safety atlases in continuous-deployment pipelines. This maximizes both feasibility and paper portfolio.
