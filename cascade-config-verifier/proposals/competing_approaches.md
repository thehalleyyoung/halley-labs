# CascadeVerify: Three Competing Approaches

**Phase:** Ideation — Domain Visionary Output  
**Target Venue:** NSDI  
**Core Problem:** Static detection and repair of retry amplification cascades and timeout chain violations in microservice configurations  
**Scope:** v1 restricts to retry + timeout primitives only (circuit breakers deferred to v2 due to non-monotonicity)

---

## Approach A: BMC-MUS — Bounded Model Checking with MUS-Guided Cascade Discovery and MaxSAT Repair

### One-Line Summary

Encode the Retry-Timeout Interaction Graph as a QF_LIA bounded model checking problem, discover minimal cascading failure sets via MUS enumeration exploiting monotonicity-based antichain pruning, and synthesize Pareto-optimal repairs via weighted partial MaxSAT.

### Extreme Value Delivered

**Who desperately needs it:** Platform engineering teams at any organization running >20 microservices where retry storms and timeout misconfigurations are the #1 and #2 cause of cascading outages. Every major cloud post-mortem (AWS S3 2017, Google 2019, Cloudflare 2019) identifies these exact failure modes. SRE teams currently have zero pre-deployment tools that reason across service boundaries about retry-timeout interactions — existing linters (kube-score, Istio Analyze, OPA/Rego) inspect individual resources without cross-service reasoning.

**Why it's extreme value:** CascadeVerify as a CI/CD gate means the configuration that would have taken down production on Thursday at 2am is caught on Wednesday in the PR review. The tool doesn't just detect — it synthesizes minimal-deviation repairs, meaning the fix changes as few parameters as possible from what operators intended. This converts a 4-hour incident response into a 30-second PR annotation.

### Technical Approach

**Architecture.** The system has four core phases: (1) Multi-format config ingestion parses Kubernetes, Istio, and Envoy YAML/JSON manifests, resolves Helm/Kustomize templates, resolves cross-resource references, and infers control-plane defaults to build a complete Retry-Timeout Interaction Graph (RTIG) — a directed dependency graph G = (V, E, κ, ρ) where V is the service set, E is the call dependency relation, κ: V → ℕ assigns capacity descriptors, and ρ: E → (retries, timeout_per_attempt, timeout_total) annotates each edge with its resilience policy. (2) SMT encoding translates the RTIG into a QF_LIA bounded model checking formula Φ_BMC(G, k, d) with Boolean failure-injection variables per service, integer load counters per service per step, and integer timeout-budget counters per edge per step. Retry amplification is encoded as multiplicative load propagation: the effective load arriving at service v from predecessor u at step t is the load at u multiplied by the retry count on edge (u,v), expressed as a linear constraint via bounded integer multiplication. Timeout chain feasibility is encoded as additive constraints: the sum of per-hop timeouts along any dependency path, expanded by retry counts, must not exceed the upstream deadline. (3) Cascade analysis uses the MARCO algorithm for MUS (Minimal Unsatisfiable Subset) enumeration over the negated cascade-freedom formula to discover minimal failure sets. The central theoretical result — monotonicity of CB-free retry-timeout networks (Theorem B6) — guarantees that antichain pruning is sound: if failure set F causes a cascade, every superset F' ⊇ F also causes one, so supersets of known cascading sets and subsets of known safe sets are safely pruned from enumeration. Portfolio solving dispatches to multiple Z3 configurations with shared clause databases. Incremental solving reuses learned clauses across related queries for successive values of k. (4) Repair synthesis encodes the repair problem as weighted partial MaxSAT: hard clauses assert cascade-freedom under all verified failure scenarios up to bound k, soft clauses penalize deviation from the operator's original parameter values weighted by operational impact (retry count changes weighted higher than timeout adjustments, reflecting operational sensitivity). The solver produces the Pareto frontier of repairs trading off minimality against robustness margin.

**Key algorithms.** BMC unrolling with cone-of-influence reduction to eliminate variables unreachable from the cascade target. Symmetry breaking constraints to prune equivalent failure sets (e.g., swapping two failed leaf services with identical topology). MARCO-based MUS enumeration augmented with monotonicity-aware antichain pruning. Weighted partial MaxSAT with iterative Pareto enumeration.

**Formalism.** QF_LIA (quantifier-free linear integer arithmetic) for the verification core. Weighted partial MaxSAT for repair. The completeness bound d* = diameter(G) × max_retries is tight: in the monotone model, load propagation reaches a fixed point within one full traversal of the graph scaled by maximum retry fan-out.

### Genuine Difficulty

**(1) SMT encoding scalability.** A 50-service topology with unrolling depth 60 and 3 constraint variables per service per step yields ~9,000 SMT variables and ~27,000 constraints. While within Z3's capability, the constant factors matter: naive encoding of retry multiplication as repeated addition creates constraint bloat, requiring domain-specific optimizations (logarithmic decomposition for bounded integer products, variable elimination for single-predecessor services). **(2) Faithful config parsing.** Istio's configuration precedence rules span 40+ pages across 4+ schema versions with version-specific semantics. Helm template rendering involves Go templating with dynamic values. Missing manifests require default-value inference that matches control-plane behavior. **(3) MaxSAT repair operational sensibility.** The solver can produce repairs that are formally optimal but operationally absurd (e.g., setting all retry counts to 0). Encoding operator-acceptable parameter ranges as additional hard constraints requires understanding domain conventions. **(4) MUS enumeration completeness.** Enumerating all minimal cascading failure sets is #P-hard (Theorem D2); the practical question is whether the first few MUS cores found correspond to the most operationally relevant cascades.

### New Math Required

**(M1) Monotonicity Lemma (B6 — central contribution).** Retry-timeout networks without circuit breakers are monotone: for failure sets F ⊆ F', if F induces a cascade then F' induces a cascade. Proof is by structural induction showing every operator in the load propagation equation — addition of retry-amplified loads from predecessors, comparison against capacity — is monotonically non-decreasing in the number of failed services. This is the key enabling result: it guarantees soundness of antichain pruning, converting exponential MUS enumeration to a tractable antichain traversal.

**(M2) Completeness Bound (B3).** d* = diameter(G) × max_retries suffices for BMC completeness. Proof: in the monotone model, load at every service is non-decreasing across steps, so a fixed point is reached within one full graph traversal scaled by maximum fan-out. The bound is tight (constructive lower bound via chain topology).

**(M3) BMC Encoding Soundness (B2).** The QF_LIA formula Φ_BMC(G, k, d) is satisfiable iff there exists a failure set F with |F| ≤ k causing cascade within d steps. Proof by structural induction on unrolling depth with case analysis for retry and timeout encodings.

**(M4) Repair Optimality (B5).** The weighted partial MaxSAT formulation guarantees that returned repairs minimize total weighted parameter deviation subject to cascade-freedom. Follows from standard MaxSAT optimality applied to the specific encoding, but requires proving that the hard-clause encoding of cascade-freedom is equivalent to the universally-quantified safety property.

### Best-Paper Potential

This approach has best-paper potential because it formalizes a universally-recognized problem that has no existing formal treatment. Every SRE knows about retry storms; nobody has a formal definition of cascade-safety for configurations or a complexity characterization of the verification problem. The monotonicity lemma is clean, memorable, and immediately useful — it's the kind of structural insight that a reviewer can verify in 20 minutes and appreciate for the rest of their career. The tool operates on real configs with zero specification burden, bridges formal methods and systems exactly as NSDI rewards, and the explicit scoping to the monotone case with circuit breakers as a stated open problem demonstrates intellectual honesty. The evaluation plan — semi-synthetic topologies with full ground truth, real open-source configs, graph-analysis baseline comparison, and post-mortem case studies — is comprehensive and honest about its methodology.

### Hardest Technical Challenge

**Scaling BMC to 50-service topologies within 90-second laptop budgets.** The naive encoding hits Z3's limits around 30 services due to the multiplicative blowup from retry amplification across multiple hops. The solution strategy is three-pronged: (a) cone-of-influence reduction eliminates all variables not on any path from failed services to the cascade target, typically removing 40-60% of variables; (b) symmetry breaking constraints reduce the search space by canonicalizing equivalent failure sets; (c) incremental solving reuses the clause database from the k-failure analysis when checking k+1, amortizing Z3's learning across queries. If these optimizations are insufficient, compositional decomposition — verifying strongly-connected components independently with interface contracts — provides a fallback that sacrifices exact minimality guarantees for scalability to 100+ services.

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 9 | Fills a critical gap with zero existing solutions; directly prevents production outages |
| **Difficulty** | 7 | Novel encoding + faithful parsing is hard; core SMT/MaxSAT techniques are mature |
| **Potential** | 8 | Clean theoretical contribution + practical tool is NSDI's sweet spot |
| **Feasibility** | 8 | ~50K novel LoC in Rust is ambitious but Z3 does the heavy lifting; monotone model avoids CB complexity |

---

## Approach B: AmpDom — Abstract Interpretation with Amplification Domains for Lightweight Cascade Analysis

### One-Line Summary

Propagate retry amplification factors and timeout budgets through the service graph using a novel abstract domain (the *amplification-budget lattice*), detect cascades as abstract fixpoint violations in polynomial time, and synthesize repairs via a SAT-encoded constraint propagation over the abstract state.

### Extreme Value Delivered

**Who desperately needs it:** The same SRE and platform engineering teams as Approach A, but AmpDom targets a different operational profile — organizations with 100–500 microservices where SMT-based verification is infeasible. Companies like Uber (4,000+ microservices), Netflix (1,000+), and Airbnb (500+) need cascade verification that scales to their full topology, not just a 50-service critical subgraph. For these organizations, the choice is between AmpDom's sound-but-approximate analysis of the full mesh and no analysis at all.

**Why it's extreme value:** AmpDom runs in seconds on topologies where BMC would time out. It integrates as a sub-second linting pass in CI/CD, fast enough to run on every commit, not just release candidates. The trade-off — some false positives in exchange for zero false negatives (soundness) and 100× better scalability — is exactly the trade-off production teams prefer. Better to have a sound alarm that occasionally cries wolf than a precise alarm that can only watch 5% of your services.

### Technical Approach

**Architecture.** AmpDom replaces the SMT solver with a custom abstract interpreter operating over a purpose-built abstract domain. The system has three phases: (1) RTIG extraction is identical to Approach A — parse K8s/Istio/Envoy configs into a Retry-Timeout Interaction Graph. (2) Abstract cascade analysis defines the *amplification-budget lattice* L = (AmpFactor × TimeoutBudget × LoadBound, ⊑) where AmpFactor tracks worst-case retry amplification as a multiplicative interval [lo, hi] ⊆ ℕ, TimeoutBudget tracks remaining timeout as an interval [lo, hi] ⊆ ℝ≥0, and LoadBound tracks worst-case request multiplier at each service. The abstract transfer function for an edge (u, v) with retry count r and per-attempt timeout t is:

- AmpFactor(v) := AmpFactor(v) ⊔ (AmpFactor(u) ⊗ [r, r])  (multiplicative join)
- TimeoutBudget(v) := TimeoutBudget(v) ⊓ (TimeoutBudget(u) ⊖ [r × t, r × t])  (subtractive meet)
- LoadBound(v) := LoadBound(v) ⊔ (LoadBound(u) ⊗ [r, r])  (multiplicative join)

where ⊗ is interval multiplication, ⊖ is interval subtraction, and ⊔/⊓ are join/meet in the lattice. A cascade alarm fires when LoadBound(v) exceeds the capacity interval for any service v, or when TimeoutBudget(v) goes negative (timeout chain violation). The analysis is a standard Kleene iteration to least fixpoint on the product lattice, guaranteed to terminate because AmpFactor is bounded by the product of all retry counts in the graph (finite), TimeoutBudget is bounded below by 0, and the lattice has finite height (retry counts are small integers, typically ≤ 5, so the product lattice height is bounded by Πᵢ rᵢ).

(3) Repair synthesis. When the abstract analysis reports a cascade alarm, the repair problem is: find a minimal perturbation of retry counts and timeout values such that the abstract fixpoint no longer violates any bound. This is encoded as a pseudo-Boolean optimization (PBO) problem — equivalent to 0-1 integer linear programming — where Boolean variables represent whether each parameter is modified, integer variables represent the new parameter values, and constraints encode the abstract transfer functions as linear inequalities. The PBO solver minimizes the number of changed parameters (primary objective) and total deviation magnitude (secondary objective). This is a SAT-modulo-linear-arithmetic problem, solvable by off-the-shelf PBO solvers (e.g., RoundingSat, Open-WBO) in seconds for realistic sizes.

**Key algorithms.** Worklist-based Kleene iteration with widening on the amplification intervals for cyclic topologies (apply widening after 3 iterations to ensure convergence in the presence of back-edges). Narrowing pass after widening to recover precision. PBO-based repair with iterative constraint tightening: the first repair pass ensures cascade-freedom under single failures, then re-analyzes under double failures, iterating until the k-failure bound is met.

**Formalism.** Galois connection between the concrete load propagation semantics (integer load values at each service under each failure scenario) and the abstract amplification-budget domain. Soundness: if the abstract analysis reports no cascade alarm, then no concrete failure set of size ≤ k triggers a cascade. The Galois connection is defined by the abstraction function α that maps each concrete load vector to its bounding intervals, and the concretization function γ that maps each abstract state to the set of concrete states it represents.

### Genuine Difficulty

**(1) Designing a domain that is both precise enough to avoid spurious alarms and abstract enough to analyze in polynomial time.** The amplification-budget lattice must capture the multiplicative nature of retry amplification (not just additive load accumulation) while remaining tractable. Naive interval arithmetic over-approximates badly when retry amplification compounds across multiple hops — the abstract amplification at depth d grows as [Πrᵢ, Πrᵢ] while the concrete amplification depends on which specific services fail. The key design challenge is incorporating failure-set sensitivity into the abstract domain without reverting to per-failure-set analysis (which would be exponential). **(2) Handling fan-in convergence.** When multiple upstream services retry against the same downstream service, the abstract domain must soundly combine their amplification factors. Simple join over-approximates because it assumes worst-case from all predecessors simultaneously; a relational domain (tracking correlations between different predecessors' failure states) would be more precise but breaks polynomial-time analysis. **(3) False positive rate management.** Abstract interpretation is inherently over-approximate: it will report cascade risks that don't exist in any concrete execution. If the false positive rate exceeds ~20%, the tool becomes useless because operators ignore its warnings. Achieving ≤10% false positive rate while maintaining soundness requires careful domain design and may require path-sensitive refinements (analyzing critical paths individually rather than joining over all paths).

### New Math Required

**(M1) Amplification-Budget Lattice (Novel).** Define the product lattice L = AmpInterval × BudgetInterval × LoadInterval with operations ⊗ (multiplicative transfer), ⊖ (timeout consumption), ⊔/⊓ (join/meet). Prove: (a) L is a complete lattice of finite height bounded by H = (Πᵢ rᵢ) × Tmax × Cmax, where rᵢ are retry counts, Tmax is maximum timeout, and Cmax is maximum capacity. (b) The abstract transfer function F♯ is monotone on L. (c) Kleene iteration converges in at most H steps (without widening) and in O(|E| × W) steps with widening, where W is the widening chain length.

**(M2) Soundness of the Galois Connection (Novel).** Prove that the abstraction α and concretization γ form a Galois insertion (L_concrete, ⊆) ⇆ (L_abstract, ⊑), and that the abstract transfer function F♯ soundly over-approximates the concrete transfer function: γ(F♯(a)) ⊇ F(γ(a)) for all abstract states a. The concrete domain here is the powerset of (failure-set, load-vector) pairs, which requires showing that the interval abstraction soundly captures the multiplicative interactions across all failure sets of size ≤ k simultaneously.

**(M3) Precision Bound (Novel).** Characterize the false-positive rate as a function of topology structure. For tree topologies (no fan-in convergence), prove the analysis is exact (zero false positives). For DAGs, bound the over-approximation factor by the maximum fan-in degree. For general graphs, characterize the additional imprecision introduced by widening.

**(M4) PBO Repair Correctness.** Prove that the PBO repair encoding is sound: if the PBO solver returns a parameter assignment, the modified configuration passes the abstract cascade analysis. This requires showing that the PBO constraints faithfully encode the abstract transfer functions.

### Best-Paper Potential

AmpDom has best-paper potential as a *scalability story* — it demonstrates that the cascade verification problem admits a polynomial-time sound approximation that is precise enough for practical use. The amplification-budget lattice is a novel abstract domain tailored to the retry-timeout interaction semantics; the precision-versus-scalability trade-off, characterized formally as a function of topology structure, gives reviewers a clean intellectual contribution to evaluate. The result that tree topologies admit exact polynomial-time analysis while general topologies require approximation with bounded imprecision is a clean dichotomy. If the false positive rate on real-world topologies (which are often close to trees) is ≤5%, the practical story is extremely compelling: sound cascade verification in sub-second time for topologies 10× larger than any SMT-based approach can handle.

### Hardest Technical Challenge

**Achieving ≤10% false positive rate on realistic topologies while maintaining soundness.** The multiplicative nature of retry amplification means that interval arithmetic compounds imprecision exponentially across hops. A chain of 5 services each with retry count 3 has concrete worst-case amplification 3⁵ = 243, which the abstract domain computes exactly. But when paths merge (fan-in), the abstract domain assumes all paths contribute their worst case simultaneously, which may not correspond to any single failure scenario. The solution strategy is a *demand-driven refinement*: when the abstract analysis reports an alarm, trace back the abstract cascade path and selectively refine the domain along that path using disjunctive completion (tracking a small number of separate abstract states for different failure scenarios). This refines away spurious alarms without abandoning the polynomial-time baseline. The risk is that refinement devolves into enumerating failure scenarios, losing the scalability advantage.

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Same problem as A but at 10× scale; slightly less value because less precise |
| **Difficulty** | 7 | Novel abstract domain design is genuinely hard; balancing precision vs. scalability is an art |
| **Potential** | 7 | Clean contribution but abstract interpretation for configs is a harder sell at NSDI than BMC |
| **Feasibility** | 9 | No external solver dependency; the analysis is a graph algorithm; repair via off-the-shelf PBO |

---

## Approach C: RetryAlg — Semiring-Algebraic Cascade Verification via Algebraic Path Problems

### One-Line Summary

Model the microservice retry-timeout interaction graph as a weighted graph over a custom *cascade semiring*, compute worst-case amplification and timeout consumption via algebraic path problem (generalized shortest path) algorithms, and synthesize repairs via integer linear programming over the semiring valuations.

### Extreme Value Delivered

**Who desperately needs it:** The same audience — but RetryAlg's unique value proposition is *mathematical transparency*. SREs don't trust black-box SMT solvers that say "this config is unsafe" without explanation. RetryAlg expresses every cascade risk as a concrete algebraic expression: "the amplification factor along path A→B→C→D is 3×2×4 = 24×, which exceeds D's capacity of 20× baseline load." The entire verification result is a *human-readable algebraic certificate* — a product of retry counts along a path, a sum of timeouts along a chain — that any engineer can verify with mental arithmetic. This is extreme value because it converts a formal verification result into something an on-call SRE can reason about at 3am without understanding SAT solvers.

**Why it's extreme value:** The algebraic formulation also enables *compositional reasoning by design*. The semiring structure means that verification results compose: if team A verifies their subgraph and team B verifies theirs, the combined result follows algebraically from the interface. This matches how microservice organizations actually operate — independent teams owning independent services — and provides a path to verification at organizational scale without centralized analysis.

### Technical Approach

**Architecture.** RetryAlg treats cascade verification as an instance of the algebraic path problem (APP) — a generalization of shortest paths to arbitrary semirings. The system has three phases:

(1) **RTIG extraction** is shared with Approaches A and B.

(2) **Algebraic cascade analysis.** Define the *cascade semiring* S = (D, ⊕, ⊗, 0̄, 1̄) where the domain D = (AmpFactor ∈ ℕ, TimeoutCost ∈ ℝ≥0, CriticalPathLen ∈ ℕ) is a triple encoding retry amplification factor, cumulative timeout cost, and path length. The semiring operations are:

- **⊕ (choice/join):** takes the component-wise maximum of amplification and timeout cost:
  (a₁, t₁, l₁) ⊕ (a₂, t₂, l₂) = (max(a₁, a₂), max(t₁, t₂), max(l₁, l₂))
  This selects the worst-case path at fan-in points.

- **⊗ (sequential composition):** multiplies amplification, adds timeout cost:
  (a₁, t₁, l₁) ⊗ (a₂, t₂, l₂) = (a₁ × a₂, t₁ + a₁ × t₂, l₁ + l₂)
  The key insight: timeout cost at the second hop is *multiplied* by the amplification factor of the first hop, because retries re-incur the downstream timeout. This captures the retry-timeout interaction that simple additive timeout analysis misses.

- **0̄ = (0, 0, 0)** (no path), **1̄ = (1, 0, 0)** (identity/self-loop).

The algebraic path problem computes, for each pair of services (s, v), the semiring "distance" d(s, v) — the worst-case amplification factor, cumulative timeout cost, and critical path length from entry point s to service v across all paths. For DAGs, this is computed by a topological-order traversal in O(|V| + |E|) time. For general graphs (cycles from mutual dependencies), the algebraic path problem is solved via Gaussian elimination on the semiring (O(|V|³)) or, equivalently, by computing the Kleene star (closure) of the adjacency matrix over the cascade semiring.

A **cascade alarm** fires when: (a) d(s, v).AmpFactor > κ(v) for some entry point s and service v (retry amplification exceeds capacity — a retry storm), or (b) d(s, v).TimeoutCost > Deadline(s) for some entry point s (timeout chain violation — the cumulative timeout along the critical path exceeds the caller's deadline).

The analysis also computes the *critical cascade path* — the specific sequence of edges achieving the worst-case amplification — by standard algebraic path problem backtracking.

(3) **ILP repair synthesis.** When alarms fire, the repair problem is: find a minimal perturbation of edge retry counts and timeout values such that all semiring distances satisfy the safety bounds. This is a mixed-integer linear program (MILP): integer variables for retry counts (small domain: 0–5), continuous variables for timeout values (bounded by operator-specified ranges), and the objective minimizes total weighted parameter deviation. The constraints encode the semiring distance computation as linear inequalities (amplification products are linearized using logarithms for bounded integer domains, or directly encoded as products of small integers via standard MILP techniques). Off-the-shelf solvers (COIN-OR CBC, HiGHS, or Gurobi's free academic license) solve this in milliseconds for realistic sizes because the number of decision variables equals the number of tunable parameters (typically ≤ 200 for a 50-service topology).

**Key algorithms.** Topological-order APP for DAGs (O(|V| + |E|)). Kleene star computation via Floyd-Warshall generalization for cyclic graphs (O(|V|³)). MILP for repair. Bellman-Ford-style iteration as a fallback for very large sparse graphs (O(|V| × |E|)).

**Formalism.** Algebraic path problems over semirings, generalizing Dijkstra/Floyd-Warshall. The cascade semiring is a bounded lattice-ordered semiring (since ⊕ = max and the domain is bounded), guaranteeing convergence of Kleene star computation.

### Genuine Difficulty

**(1) The cascade semiring is not a true semiring for cyclic graphs with fan-in.** When multiple paths converge on the same service, the worst-case amplification depends on whether the paths' failure triggers are correlated. The ⊕ = max operation assumes the worst case from each path occurs independently, which is sound (over-approximate) but may be imprecise. The deeper problem: for cyclic graphs, the semiring "distance" is the Kleene star of the adjacency matrix, and this star is only well-defined if the semiring is *closed* (the infinite sum ⊕ᵢ aⁱ converges). The cascade semiring is closed because amplification factors are bounded by the product of all retry counts, but computing the closure requires careful handling of cycles with amplification > 1 (which represent unbounded retry storms — themselves a cascade). **(2) Multiplicative-additive interaction.** The ⊗ operation's timeout component — t₁ + a₁ × t₂ — mixes multiplication (from retry amplification) with addition (from timeout accumulation). This means the semiring is not commutative, which complicates the algebraic path computation (the left-to-right composition order matters). Standard Floyd-Warshall assumes commutativity; the non-commutative generalization requires tracking directed path composition, increasing the constant factor. **(3) Multi-source cascade interactions.** The semiring APP computes worst-case paths from a single entry point. When multiple entry points contribute load to the same downstream service, the total load is the *sum* of per-source amplifications, not the max. This requires a second pass that aggregates per-source results — easy for DAGs but requires care for cyclic topologies to avoid double-counting load through cycles. **(4) Repair linearization.** The MILP encoding must linearize products of integer variables (retry counts along a path). For bounded integers (0–5), this is standard (McCormick envelopes or binary expansion) but adds auxiliary variables proportional to path length × variable bit-width.

### New Math Required

**(M1) Cascade Semiring Definition and Closure (Novel).** Formally define S = (D, ⊕, ⊗, 0̄, 1̄) and prove: (a) S satisfies semiring axioms (associativity, distributivity of ⊗ over ⊕, identity/annihilator elements). (b) S is ω-continuous: for any ascending chain d₁ ⊑ d₂ ⊑ ···, the supremum exists in D. (c) The Kleene star a* = ⊕ᵢ≥₀ aⁱ exists for every element a ∈ D and is computable in O(log H) semiring operations via repeated squaring, where H is the lattice height. The non-trivial part is proving distributivity given the mixed multiplicative-additive structure of ⊗.

**(M2) Soundness of Algebraic Cascade Analysis (Novel).** Prove that for every concrete failure set F of size ≤ k, the concrete load at service v under F is bounded above by the algebraic distance d(s, v).AmpFactor × BaseLoad(s). This requires showing that the semiring ⊕ (max) soundly over-approximates the concrete load aggregation (sum) for the worst-case failure set — which it does for single-source analysis but requires a correction factor for multi-source convergence. State the correction theorem: the total concrete load at v is bounded by Σₛ d(s, v).AmpFactor × BaseLoad(s), computable in O(|Sources| × (|V| + |E|)) time.

**(M3) Characterization of Exact vs. Approximate Cases (Novel).** Prove that for tree topologies (no fan-in), the algebraic analysis is exact — the reported amplification factors equal the true worst-case. For DAGs with fan-in, characterize the over-approximation ratio as a function of the maximum fan-in degree and the failure budget k. Specifically: the algebraic amplification at service v over-approximates the true worst case by at most a factor of min(k, fan-in(v)), where fan-in(v) is the number of distinct predecessors. This bound is tight.

**(M4) ILP Repair Correctness and Optimality (Adaptation).** Prove that the MILP encoding of the repair problem is equivalent to the algebraic safety condition: a parameter assignment satisfies the MILP constraints iff the resulting algebraic distances satisfy all cascade-freedom bounds. The linearization of retry-count products via McCormick envelopes is exact for binary and bounded-integer variables (standard result, but must be verified for the specific cascade semiring structure).

### Best-Paper Potential

RetryAlg has best-paper potential as an *elegance story*. The insight that cascade verification is an algebraic path problem — the same mathematical structure underlying shortest paths, transitive closure, and dataflow analysis — is unifying and beautiful. The cascade semiring is a concrete, novel algebraic object that captures the retry-timeout interaction in a single clean abstraction. The O(|V|³) verification time for arbitrary topologies and O(|V| + |E|) for DAGs dominates both BMC (exponential worst-case) and abstract interpretation (polynomial but with domain-specific overhead). The human-readable algebraic certificates ("amplification along path X is 3×2×4 = 24×") are a powerful practical differentiator. If the semiring structure extends naturally to circuit breakers in v2 (via tropical semiring extensions with thresholded idempotent elements), the algebraic framework becomes the foundation for a comprehensive cascade theory.

### Hardest Technical Challenge

**Proving the cascade semiring axioms hold, particularly distributivity, given the non-commutative mixed multiplicative-additive structure.** The ⊗ operation — (a₁, t₁, l₁) ⊗ (a₂, t₂, l₂) = (a₁ × a₂, t₁ + a₁ × t₂, l₁ + l₂) — is not commutative because timeout cost depends on composition order. Left-distributivity of ⊗ over ⊕ requires: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c). Expanding the timeout component: a.t + a.a × max(b.t, c.t) vs. max(a.t + a.a × b.t, a.t + a.a × c.t). These are equal because max distributes over addition of a non-negative constant — but the proof requires careful treatment of the interaction between max and the multiplicative timeout expansion. Right-distributivity requires analogous analysis. The solution is to prove distributivity component-by-component, leveraging the non-negativity of all domain elements and the distributivity of max over addition in the non-negative reals. If full semiring axioms fail (e.g., distributivity breaks for some edge cases), the fallback is to work with a *pre-semiring* (dropping one distributivity law) and use path-sensitive APP algorithms that don't require the dropped axiom.

### Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Value** | 8 | Same core value; algebraic certificates add transparency; compositionality adds organizational value |
| **Difficulty** | 6 | Algebraic path problems are well-studied; novelty is in the semiring definition, not the algorithms |
| **Potential** | 7 | Elegant unifying framework; slightly lower NSDI potential because it's more "theory" than "systems" |
| **Feasibility** | 9 | O(|V|³) worst-case is trivially laptop-feasible; MILP repair is fast; no SMT solver dependency |

---

## Comparative Summary

| Dimension | A: BMC-MUS | B: AmpDom | C: RetryAlg |
|-----------|-----------|-----------|-------------|
| **Core formalism** | QF_LIA BMC + MaxSAT | Abstract interpretation | Semiring algebraic path problem |
| **Where novelty lives** | Monotonicity theorem + MUS-cascade correspondence | Novel abstract domain design | Cascade semiring definition |
| **Verification complexity** | NP (SAT query) | Polynomial (fixpoint) | O(\|V\|³) (matrix closure) |
| **Repair method** | Weighted partial MaxSAT | Pseudo-Boolean optimization | Mixed-integer LP |
| **Precision** | Exact (no false positives) | Sound over-approx (some FPs) | Sound over-approx (fewer FPs than B) |
| **Scalability** | ~50 services | ~500 services | ~500 services (faster than B) |
| **External solver needed** | Z3 (SMT/MaxSAT) | Optional PBO solver | MILP solver (lightweight) |
| **Explainability** | SMT model → failure trace | Abstract alarm → path | Algebraic expression → path product |
| **Math weight** | Medium (structural lemma) | Medium (Galois connection) | Medium-High (semiring theory) |
| **NSDI fit** | Best (systems + theory balance) | Good (scalability story) | Good (elegance story) |
| **Value** | 9 | 8 | 8 |
| **Difficulty** | 7 | 7 | 6 |
| **Potential** | 8 | 7 | 7 |
| **Feasibility** | 8 | 9 | 9 |

### Recommended Strategy

**Lead with Approach A (BMC-MUS)** as the primary proposal — it has the strongest NSDI fit, the cleanest novel theorem (monotonicity), and exact precision. **Implement Approach C (RetryAlg) as the "graph-analysis baseline"** required by the depth-check feedback — the semiring APP is exactly the "simple path analysis" baseline the evaluation plan calls for, but formalized with enough rigor to stand on its own if BMC struggles at scale. **Hold Approach B (AmpDom) as a scalability fallback** — if BMC hits a hard wall at 30 services despite optimizations, pivot to abstract interpretation as the primary analysis with BMC as a precision refinement for flagged subgraphs.

This three-approach strategy provides: a strong primary contribution (A), a principled baseline that doubles as independent contribution (C), and a scalability escape hatch (B).
