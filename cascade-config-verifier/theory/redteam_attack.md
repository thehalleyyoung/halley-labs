# Red-Team Attack Report: CascadeVerify Theory Stage

**Status:** Pre-theory adversarial review
**Objective:** Find every brittle assumption, hidden contradiction, and potential failure point before the full theory is written.
**Stance:** Adversarial. The goal is to strengthen the theory by trying to break it.

---

## 1. Attack on Theorem B6 (Monotonicity) — HIGHEST PRIORITY

**Claimed theorem:** For an RTIG G without circuit breakers, if failure set F induces a cascade, then every superset F' ⊇ F also induces a cascade.

**Claimed proof strategy:** Show load_F(v,t) ≤ load_F'(v,t) for all v, t by induction on t, using monotonicity of the load propagation equation L(v, t+1) = Σ_{pred} L(pred, t) × (1 + retry_factor(pred → v)).

### Attack 1a: Can blocked-path load decrease exceed retry increase?

**Concrete RTIG counterexample attempt:**

```
Topology:
  A → w → C → D    (A calls w, w calls C, C calls D)
  A → B → C         (A also calls B, B also calls C)

Parameters:
  retry(A→w) = 1, retry(A→B) = 1
  retry(w→C) = 5, retry(B→C) = 1
  retry(C→D) = 3
  capacity(C) = 20, capacity(D) = 50
  Baseline load at A = 10

Scenario F  = {D fails}
Scenario F' = {D fails, w fails}
```

**Analysis under F (D fails only):**
- A sends 10 to w and 10 to B (each after initial + 1 retry = ×2)
- w receives 20 requests, forwards to C with retry factor 5 → w sends 20 × (1+5) = 120 to C
- B receives 20 requests, forwards to C with retry factor 1 → B sends 20 × (1+1) = 40 to C
- C receives 120 + 40 = 160 requests
- C calls D (failed) with retry factor 3 → 160 × 4 = 640 requests at D → D is overwhelmed
- Load at C: 160. Load at D: 640.

**Analysis under F' (D fails AND w fails):**
- A sends 10 to w (which fails immediately → errors returned) and 10 to B
- A retries against w: retry(A→w) = 1 → A sends 10 × (1+1) = 20 to w, all of which fail
- A sends 10 × (1+1) = 20 to B (retry(A→B) = 1)
- w is failed → forwards ZERO to C (this is the blocked-path load reduction)
- B receives 20, forwards to C with retry factor 1 → B sends 20 × 2 = 40 to C
- C receives 0 + 40 = 40 requests (was 160 under F)
- C calls D with retry 3 → 40 × 4 = 160 at D (was 640 under F)

**Result:** Load at C dropped from 160 to 40. Load at D dropped from 640 to 160.

**Verdict: This looks like a monotonicity violation — but is it?**

The critical question is: does A's retry against w generate load that *compensates* for the loss at C? Under the current proof sketch, A's retries against w produce 20 requests *at w* — but w is dead, so these go nowhere downstream. The "load increase at w's predecessors" (A retries more) does NOT translate to increased load at w's successors (C, D) because the additional retries are directed *at the dead service w*, not rerouted to C.

**The proof sketch's claim that "predecessors, now retrying, may route through alternative paths" is WRONG for the RTIG model.** An RTIG is a static dependency graph. Service A has a fixed edge A→w and a fixed edge A→B. A's retries against w go to w, period. A does not "reroute" retries from w to B. The retries against w produce errors, which increase load at A, but this increased load at A does NOT increase A's calls to B — A calls B with its own independent load, not in response to w's failure.

**CRITICAL FINDING:** The proof sketch conflates two distinct phenomena:
1. A retries against w (increases load at A and on edge A→w — but w is dead, so no downstream propagation)
2. A's independent calls to B (unchanged by w's failure)

The blocked-path effect CAN reduce load at downstream services without compensation.

### Attack 1a: Resolution / Why This Attack May Fail

The attack above depends on the load propagation equation. Re-examine:

L(v, t+1) = Σ_{pred u} L(u, t) × (1 + retry_factor(u → v))

The key question is: what is L(u, t) for a predecessor u of a failed service w?

If w fails, u sends requests to w, gets errors, and retries. The retry factor on edge u→w means u sends (1 + retry(u→w)) × L(u,t) requests *along edge u→w*. All fail. Does u's load L(u, t) itself increase?

**In the BMC encoding as written**, L(u, t) is the "effective request count reaching u at step t." If u=A and A receives 10 requests from external sources, then L(A, t) = 10 regardless of whether w fails. A's retries against w don't increase L(A, t) — they increase the number of requests *leaving* A on edge A→w.

So in the load propagation equation for C:
- L(C, t+1) = L(w, t) × (1+retry(w→C)) + L(B, t) × (1+retry(B→C))

If w is failed, what is L(w, t)? The model says failed services start unavailable and presumably have load 0 (they're not processing requests). So L(w, t) = 0, and the first term vanishes. The load at C genuinely decreases.

**The compensation mechanism in the proof sketch doesn't work as stated.** The retries at A increase load on edge A→w, but since w is dead, this load never reaches C. The load at C is strictly determined by the incoming load from its live predecessors.

**SEVERITY: POTENTIALLY FATAL for B6.**

The proof needs to be restructured. One possible rescue:

**Rescue attempt:** Redefine the model so that when service w fails, the error responses from w cause w's predecessors to *themselves* become degraded/unavailable (because they timeout or exhaust retries). This predecessor failure then creates a *new* failure, which by induction increases load elsewhere. But this only works if the cascade definition includes the predecessors eventually failing — and it assumes the predecessor failure propagates upward in a way that eventually reaches the entry point through some path. This is plausible but is a much more complex argument than the current proof sketch suggests.

**Alternative rescue:** The claim should be reformulated: monotonicity holds for the *cascade predicate* (does cascade occur: yes/no), not necessarily for pointwise load at every node. It's possible that Cascade(F) ⟹ Cascade(F') even though some individual nodes have lower load under F' — as long as the cascade at the critical node (entry point) still occurs. But this requires a completely different proof strategy than pointwise load monotonicity.

### Attack 1b: Indirect effects through shared dependencies

```
Topology:
  A → w → C    (A calls w, w calls C)
  B → C         (B calls C, independently)
  C → D         (C calls D)

Parameters:
  retry(A→w) = 2, retry(w→C) = 4
  retry(B→C) = 1, retry(C→D) = 3
  capacity(D) = 100
  Baseline: L(A)=5, L(B)=10
```

Under F = {D fails}:
- w receives ~15 from A, forwards to C: 15×5 = 75
- B forwards to C: 20×2 = 40
- C total: 75+40 = 115 → C calls D: 115×4 = 460

Under F' = {D fails, w fails}:
- w is dead → contributes 0 to C
- B unchanged → C gets 40
- C calls D: 40×4 = 160

Load at D dropped from 460 to 160. B's contribution to C is unchanged (B has no dependency on w). The only effect of w's failure is removing load from C. The cascade at D may or may not still occur, but the load is strictly lower.

**This attack strengthens 1a:** when the failed service w is a high-amplification intermediary, blocking it removes the dominant load source from its downstream services. No compensating mechanism exists.

### Attack 1c: "Alternative paths" argument is invalid for RTIGs

**Claim in proof sketch:** "w's predecessors, now retrying, may route through alternative paths."

**Attack:** An RTIG is a static dependency graph, not a dynamic routing mesh. The edges represent fixed call dependencies. If A depends on w and B, A's calls to w always go to w. There is no mechanism for A to "reroute" w-bound traffic to B upon w's failure. In real systems:
- Envoy/Istio routing rules are static per request type
- Service A calls service w because it needs w's specific functionality
- If w is unavailable, A retries against w, then eventually gives up and returns an error upstream
- A does NOT redirect w's traffic to B (B provides a different service)

**Even if A has a fallback path** (e.g., a circuit breaker fallback route), this would require circuit breaker logic — which is explicitly excluded from the model scope.

**Verdict:** The "alternative paths" argument in the proof sketch is invalid for the RTIG model. It confuses static dependency edges with dynamic routing. This is a **gap in the proof**, not necessarily a gap in the theorem — but the proof sketch as stated does not close it.

### Attack 1d: Fast-failure timing changes load dynamics

When service w is alive but slow, it consumes timeout budget — w's predecessor A waits for w's response until timeout, then retries. Total time: retry_count × timeout.

When w is dead (in F'), w returns errors immediately (connection refused). A retries immediately without waiting. Total time: retry_count × ~0.

**Implication:** Under F' with w dead, A exhausts retries *faster* and becomes unavailable *sooner*. In a time-step model, this means A's failure propagates upstream in fewer time steps. But the BMC model uses discrete steps, not wall-clock time. Each step advances the load propagation equation.

**In the discrete-step model:** A sends (1+retry(A→w)) requests to w in one step, all fail, and A may become unavailable in the same step. This is the same whether w is slow-and-failing or dead — the BMC model doesn't distinguish timeout latency from immediate failure at the granularity of time steps.

**Verdict:** This attack is **DEFLECTED by the model abstraction.** The discrete time-step model doesn't capture the timing difference between fast failure (dead service) and slow failure (timeout). This is both a feature (simplifies the model) and a limitation (misses timing-sensitive cascades). But it doesn't break monotonicity within the model.

### Attack 1 Summary

| Attack | Verdict | Severity |
|--------|---------|----------|
| 1a: Blocked-path decrease exceeds retry increase | **Potentially valid counterexample** | FATAL |
| 1b: Indirect shared-dependency effects | Strengthens 1a | FATAL |
| 1c: "Alternative paths" argument invalid | **Proof gap confirmed** | SEVERE |
| 1d: Fast-failure timing | Deflected by model abstraction | MINOR |

**Recommended action for B6:** The proof CANNOT rely on pointwise load monotonicity at every node. Either:
1. **Reformulate the theorem** to claim monotonicity of the cascade predicate at the entry point, using a global argument about total system load or reachability, not pointwise load comparison.
2. **Strengthen the model** so that failed intermediary services still contribute load (e.g., error responses count as load on downstream services), making the propagation equation truly monotone.
3. **Restrict the topology** to DAGs where every node is on a path from a failure source to the cascade target, eliminating the blocked-path scenario.
4. **Abandon pointwise load monotonicity** and prove cascade monotonicity directly by showing that any cascade trace under F can be "embedded" into a cascade trace under F' ⊇ F, even if the traces differ at individual nodes.

---

## 2. Attack on B1 (NP-Completeness Reduction)

**Claimed:** Cascade reachability is NP-complete via reduction from SUBSET-SUM.

### Step-by-step walkthrough of the claimed reduction

**SUBSET-SUM instance:** Given integers a₁, ..., aₙ and target T, does there exist S ⊆ {1,...,n} with Σᵢ∈S aᵢ = T?

**Claimed reduction strategy:** "Encode item values as retry counts and target sum as capacity threshold."

**Attempted reconstruction:**
- Create n+1 services: s₁, ..., sₙ, and a target service t
- For each item i, create edge sᵢ → t with retry_count = aᵢ
- Set capacity(t) = T
- Each service sᵢ has baseline load 1
- Failure budget k = n (all can fail)
- Cascade condition: load(t) > T

**Problem 1: Load accumulation is multiplicative, not additive.**

The load propagation equation is: L(t) = Σᵢ L(sᵢ) × (1 + retry(sᵢ → t))

If L(sᵢ) = 1 for all i, then L(t) = Σᵢ (1 + aᵢ). This is additive in (1 + aᵢ), not in aᵢ. To encode SUBSET-SUM, we need the load at t to equal Σᵢ∈S aᵢ for a chosen subset S.

**But which services are "failed"?** The failure set F determines which services start as unavailable. If sᵢ fails, it doesn't send load to t. We want: load(t) = Σᵢ∉F (1 + aᵢ) and cascade iff this exceeds T.

So we're encoding: does there exist F with |F| ≤ k such that Σᵢ∉F (1+aᵢ) > T? This is a complement-of-SUBSET-SUM variant, which is still NP-hard, but the encoding has off-by-one issues with the "+1" from the retry formula.

**Problem 2: The "failure set" semantics are backwards.**

In cascade reachability, failed services are *bad* — they cause the cascade. But in this reduction, we want the non-failed services to contribute load. The failed services *reduce* load at the target (they don't send requests). This is an inverted semantics from the natural interpretation.

**Possible fix:** Make the failure set be the set that does NOT contribute. Ask: is there F with |F| ≤ k such that the *remaining* services cascade the target? This works but is the complement question — asking "can at most k failures cause a cascade" means asking "does removing ≤ k services from the load sources still produce a cascade?" This encodes SUBSET-SUM on the complement.

**Problem 3: Retry counts must be positive integers in realistic configs.**

SUBSET-SUM allows arbitrary integers (including zero and negative). If aᵢ = 0 or aᵢ < 0, we cannot encode this as a retry count. Restriction to positive integers is a constraint on the reduction.

**This is NOT fatal:** SUBSET-SUM restricted to positive integers is still NP-complete.

**Problem 4: The reduction doesn't use the cascade propagation dynamics.**

The "reduction" creates a flat star topology (all services connect directly to target). No multi-hop propagation, no retry amplification chains, no timeout interactions. The NP-hardness comes from subset selection, not from cascade dynamics. A reviewer could argue this is an artificial encoding that doesn't exercise the RTIG model's distinctive features.

**Verdict:** The reduction is **plausible but under-specified** and has semantic issues that need careful resolution. The "+1" offset in (1 + retry_count) and the inverted failure semantics require explicit handling. The reduction is also **uninteresting** because it reduces to a flat topology — the hardness doesn't come from cascade propagation. Consider reducing from a more relevant problem (e.g., network reliability with multiplicative weights) to make the reduction more natural.

**SEVERITY: MODERATE.** The theorem is likely true (NP-completeness of subset-selection problems is robust), but the specific reduction needs careful construction and the current sketch is too vague to verify.

---

## 3. Attack on B3 (Completeness Bound)

**Claimed:** d* = diameter(G) × max_retries is a tight completeness bound.

### Attack 3a: Is the bound actually tight?

**Topology where cascade takes fewer than d* steps (bound is loose):**

```
Star topology: A → B, A → C, A → D (diameter = 2)
retry(all edges) = 5, max_retries = 5
d* = 2 × 5 = 10
```

If B, C, D all fail simultaneously, A receives errors on all edges in step 1, retries in step 2, exhausts retries by step 5, and is overwhelmed by step 6. The cascade completes in ~6 steps, well under d* = 10.

This is expected — d* is an upper bound, not an exact count.

### Attack 3b: Can we construct a topology requiring exactly d* steps?

```
Chain: s₁ → s₂ → s₃ → ... → sₖ (diameter = k-1)
retry(each edge) = R (max_retries = R)
Only sₖ fails.
```

Step-by-step: sₖ₋₁ sends requests to sₖ (failed), retries R times. After R steps, sₖ₋₁ becomes unavailable. Then sₖ₋₂ discovers sₖ₋₁ is unavailable, retries R times. This takes another R steps. Propagation through the entire chain: (k-1) × R = D × R = d* steps.

**This confirms the bound is tight for chains.** Good — the bound is not wastefully loose.

### Attack 3c: What about cycles?

**The problem statement says "dependency graph" but does NOT require a DAG.**

Real service meshes can have cycles: A → B → C → A (e.g., A is an API gateway, B is auth service, C is user service that calls back to A for some operation).

**How do cycles affect the completeness bound?**

If G has cycles, the "diameter" is the longest shortest path between any two nodes. In a cycle A → B → C → A, diameter = 2 (for 3 nodes). But a cascade can circulate around the cycle: A overloads B, B overloads C, C overloads A, which further overloads B, etc. Each cycle iteration multiplies the load.

**The proof sketch argues:** "After d* steps with no new state change, the system has reached a fixpoint." But in a cycle, every step *can* change state — the load increases on each circuit around the cycle. The system doesn't reach a fixpoint; it diverges.

**In the monotone lattice argument:** The lattice height is claimed to be D × R. But with cycles, the load at a node is unbounded (it grows each time the cascade circles back). The lattice must be bounded by the capacity values. The load at each node ranges from 0 to some maximum (beyond which the service is "unavailable" and stops propagating). If services have finite capacity, the lattice height is bounded by Σᵥ capacity(v) — which could be much larger than D × R.

**Potential counterexample with cycles:**

```
Cycle: A → B → A
retry(A→B) = 3, retry(B→A) = 3
capacity(A) = 1000, capacity(B) = 1000
d* = diameter × max_retries = 1 × 3 = 3
Baseline load at A = 10
```

Step 0: A has load 10, sends 10 × 4 = 40 to B
Step 1: B has load 40, sends 40 × 4 = 160 to A
Step 2: A has load 10 + 160 = 170, sends 170 × 4 = 680 to B
Step 3 (= d*): B has load 40 + 680 = 720, sends 720 × 4 = 2880 to A

The load is still growing at step d* = 3. It hasn't reached a fixpoint. The cascade might not trigger until step 5 or 6 when load finally exceeds capacity.

**SEVERITY: SEVERE.** The completeness bound d* = D × R is incorrect for graphs with cycles. The bound needs to account for cyclic amplification. A corrected bound might be d* = D × R × log(max_capacity / baseline_load) / log(max_retry_factor) or similar, capturing the number of cycle iterations needed for load to exceed capacity.

**Recommended action:** Either (a) restrict to DAGs (which excludes some real topologies), (b) prove a corrected bound for cyclic graphs, or (c) acknowledge this as a known limitation and use a conservative over-estimate.

---

## 4. Attack on D3 (Σ₂ᴾ-completeness of Repair)

### Attack 4a: Does MaxSAT over discovered MFS guarantee safety against ALL failure sets?

**Claimed:** The repair is formulated as: for each discovered minimal failure set F, add hard clauses ensuring cascade is unreachable under F with repaired parameters.

**The gap:** MaxSAT only adds hard clauses for *discovered* minimal failure sets. If the MinUnsat enumeration misses any MFS (due to solver timeouts, incomplete search, or bugs), the repair does not protect against the undiscovered failure sets.

**But wait —** the MinUnsat enumeration is claimed to be *complete* (finds ALL minimal failure sets). If completeness holds (via antichain search + monotonicity), then the repair does cover all MFS. And by monotonicity, protection against all MFS implies protection against all failure sets (since every dangerous set contains an MFS as subset).

**The real vulnerability:** The completeness of enumeration depends on:
1. Theorem B6 (monotonicity) — which we've attacked above
2. Theorem B3 (completeness bound) — which we've attacked above
3. The MARCO algorithm terminating within resource limits

If B6 is wrong, the enumeration may miss failure sets. If B3 is wrong, the BMC may miss cascades that require more steps. In either case, the MaxSAT repair is incomplete.

**Second vulnerability:** Even if ALL MFS are discovered, the repair only guarantees cascade-freedom under the repaired config for the *same* failure budget k. It does not guarantee safety for failure sets of size k+1. An operator who sees "all cascades repaired" might assume complete safety, when in fact larger failure sets could still cascade.

**SEVERITY: MODERATE.** The Σ₂ᴾ-completeness claim itself is likely correct (the ∀∃ structure is genuine). But the *practical guarantee* of the MaxSAT repair depends on the correctness of B6, B3, and complete enumeration — creating a chain of dependencies where a flaw in any link breaks the repair guarantee.

### Attack 4b: The quantifier structure is unusual

Standard Σ₂ᴾ formulation: ∃ repair R. ∀ failure set F. ¬Cascade(R, F).

But the actual implementation does: enumerate all MFS {F₁,...,Fₘ}, then solve ∃ R. ∧ᵢ ¬Cascade(R, Fᵢ). This is an NP problem (existential only), not Σ₂ᴾ. The Σ₂ᴾ-completeness characterizes the theoretical worst case, but the practical algorithm avoids the ∀ quantifier by pre-computing the universe of dangerous failure sets.

**This is fine as theory** (showing the general problem is hard), but could mislead a reader into thinking the implementation faces Σ₂ᴾ hardness. It doesn't — it faces NP hardness for repair plus whatever the enumeration cost is.

**SEVERITY: MINOR.** A presentation/framing issue, not a correctness issue.

---

## 5. Attack on D4 (FPT for Bounded Treewidth)

### Attack 5a: Is the treewidth claim provable?

**Claimed:** Cascade reachability is FPT parameterized by treewidth: O(n · 2^O(w) · d · r_max).

**The claim requires:** A dynamic programming algorithm on a tree decomposition of the RTIG that solves cascade reachability.

**Standard approach:** For each bag in the tree decomposition, enumerate all possible states of the services in the bag (2^O(w) possibilities per bag), and propagate constraints along the tree. This works for problems expressible in monadic second-order logic (MSO) via Courcelle's theorem.

**Is cascade reachability MSO-expressible?** It involves:
- Selecting a subset F (MSO: existential quantification over a set)
- Simulating load propagation over d steps (requires arithmetic on loads — NOT MSO)
- Checking if load exceeds capacity (arithmetic comparison — NOT MSO)

**Courcelle's theorem does NOT apply** because cascade reachability involves integer arithmetic (load values), not just graph structure. MSO can express connectivity, set membership, coloring — but not "sum of retry counts along a path exceeds threshold."

**A custom DP is needed.** The DP must track, for each bag, the load profile (load at each service in the bag at each time step). Since loads can be exponentially large (multiplicative amplification), the DP state space may include load values up to capacity, making it pseudo-polynomial, not FPT.

**SEVERITY: SEVERE.** The FPT claim is hand-wavy. A rigorous proof requires either (a) showing that load values can be bounded polynomially in n (doubtful — multiplicative amplification creates exponential loads), or (b) working in a model where loads are bounded by some parameter, making the bound FPT in treewidth × log(max_capacity). The O(n · 2^O(w) · d · r_max) bound suspiciously omits any dependence on capacity or load values.

### Attack 5b: Do real microservice topologies have bounded treewidth?

**Claimed:** "Real topologies typically have treewidth ≤ 5-8."

**Evidence provided:** None. This is an unsupported assertion.

**Counter-evidence:**
- A fully connected mesh of k services has treewidth k-1
- API gateways that connect to all backend services create cliques
- Service meshes with sidecar proxies create complex topologies
- Real graphs from Google's Borg paper show dense interconnections

**Plausible but unverified.** Microservice architectures that follow the "strangler fig" pattern or layered architectures may indeed have low treewidth. But meshes with cross-cutting concerns (auth, logging, tracing) create additional edges that increase treewidth.

**SEVERITY: MODERATE.** The claim is plausible but needs empirical evidence from real service mesh topologies. Without evidence, it's an unsupported assumption that weakens the D4 result's practical relevance.

### Attack 5c: Is O(n · 2^O(w) · d · r_max) practical?

For w = 8 (claimed upper bound), 2^O(w) could be 2^8 = 256 or 2^(8×c) for some constant c depending on state encoding. If each service has 3 state variables with ranges up to max_capacity, the exponent is much larger.

With d = 60 (from B3 for a 30-service topology), r_max = 5, n = 30:
- 30 × 256 × 60 × 5 = 2,304,000 — if 2^O(w) = 2^w. This is fine.
- 30 × 2^24 × 60 × 5 = ~150 billion — if 2^O(w) = 2^(3w). This is impractical.

**The constant in the exponent matters enormously** and is not specified.

**SEVERITY: MODERATE.** The FPT claim needs a precise exponent, not just O(w). The practical applicability depends entirely on the constant.

---

## 6. Hidden Assumptions

### 6a: Load is integer-valued

**Model:** L(v,t) ∈ ℕ (natural numbers)
**Reality:** Services handle request *rates* (requests/second), which are continuous. Load also fluctuates stochastically.

**Impact on theory:** Using integers enables QF_LIA encoding (efficient SMT solving). Using reals would require QF_LRA (also decidable but different solver characteristics) or nonlinear real arithmetic (undecidable in general).

**Verdict:** Reasonable modeling choice. Discretizing request rates into integer counts is standard. Does NOT break the theory. The over-approximation is conservative (ceiling of real rate ≥ actual rate).

### 6b: Retries are instantaneous

**Model:** All retries in a step complete within that step. No backoff delays.
**Reality:** Exponential backoff means retry 1 fires at ~1s, retry 2 at ~2s, retry 3 at ~4s. Total time: ~7s vs. 3 × 0s = 0s in the model.

**Impact on theory:** The model over-approximates the speed of cascade propagation (cascades propagate faster in the model than in reality). This means the model is *conservative* — if the model says no cascade exists, reality won't produce one. But the model may report cascades that real backoff prevents (false positives).

**Impact on B3:** The completeness bound d* may be tighter than necessary — real cascades may need more wall-clock time but fewer discrete steps. This is fine (conservative).

**Impact on B6:** Monotonicity is unaffected — backoff is a delay, not a load-changing mechanism.

**Verdict:** Reasonable modeling choice. Explicitly acknowledge that the model over-approximates cascade speed. False positives are acceptable for a safety tool.

### 6c: Timeouts are exact

**Model:** timeout_remaining decrements by exact amounts. Service becomes unavailable when timeout_remaining ≤ 0.
**Reality:** Timeouts have jitter (±10-50ms). Network latency varies. Clock skew exists.

**Impact:** Minimal. Jitter can be modeled conservatively by reducing timeout values by the maximum jitter bound. Does not break any theorem.

**Verdict:** Reasonable. MINOR limitation.

### 6d: Services have fixed capacity

**Model:** capacity(v) is a constant.
**Reality:** Service capacity varies with CPU load, memory pressure, garbage collection, autoscaling, etc.

**Impact:** The model assumes worst-case capacity (minimum capacity). If capacity is set to the minimum, the model is conservative. If set to average, the model may miss cascades that occur during capacity dips.

**Impact on B6:** Monotonicity relies on capacity being fixed. If capacity decreases when load increases (realistic — high load reduces throughput), the propagation equation changes and monotonicity may break in subtle ways.

**Verdict:** MODERATE concern. The fixed-capacity assumption is stated but its interaction with monotonicity should be discussed.

### 6e: Failures are permanent

**Model:** Once a service is in the failure set F, it remains unavailable for all time steps.
**Reality:** Failures are often transient. A service may fail, restart (30-60s), and recover. Recovery changes the cascade dynamics mid-propagation.

**Impact on B6:** If failed services can recover, monotonicity may break: a larger failure set F' ⊇ F has more services unavailable at t=0, but some may recover by t=5, potentially reducing load below what F produces at t=5.

**Impact on B3:** The completeness bound assumes failures persist long enough for the cascade to propagate. If failures are transient and shorter than the propagation time, the model over-approximates.

**Verdict:** MODERATE. Permanent failure is a standard worst-case assumption for safety analysis. Should be acknowledged explicitly.

### 6f: RTIG accurately represents the service mesh

**Model:** The RTIG is constructed from Kubernetes/Istio/Envoy configs.
**Reality:** The actual runtime behavior may differ from the declared config due to:
- Runtime overrides and feature flags
- Service mesh control plane propagation delays
- Configuration drift between declared and actual state
- Sidecar injection failures
- Ambient mesh vs. sidecar proxy behavior differences

**Impact:** The tool's value proposition is "config-time analysis." If configs don't accurately represent runtime behavior, the analysis is unsound. This is acknowledged in the non-claims (Appendix B item 6: "post-mortem case studies use plausible reconstructions, not ground-truth configs").

**Verdict:** MODERATE. Fundamental limitation of static analysis. Should be front-and-center in the paper's limitations section.

### 6g: No load balancing

**Model:** A single instance of each service. All load goes to one instance.
**Reality:** Services have multiple replicas. Load is distributed across replicas. A single replica's failure doesn't take down the service.

**Impact:** Massive false positive risk. The model says "load(C) = 160, exceeds capacity(C) = 100, CASCADE!" But if C has 3 replicas, the per-replica load is ~53, well within capacity.

**The model must account for replica counts.** If it doesn't, every analysis is off by a factor of replica_count. This is not mentioned anywhere in the formalization.

**SEVERITY: SEVERE if not addressed.** The load propagation equation must divide by replica count, or the capacity must be multiplied by replica count. Either way, this must be explicit in the model.

### 6h: Synchronous call model

**Model:** Service A calls service B and waits for response. Retries are sequential.
**Reality:** Some calls are asynchronous (message queues, event streams). Async calls don't produce retry amplification in the same way.

**Impact:** The model applies to synchronous RPC calls (gRPC, HTTP). Async messaging has different failure semantics. Should be explicit that the tool targets synchronous service meshes.

**Verdict:** MINOR. Reasonable scope restriction. Should be stated.

### Hidden Assumptions Summary

| Assumption | Severity | Theory-Breaking? |
|------------|----------|-------------------|
| Integer loads | MINOR | No — conservative |
| Instantaneous retries | MINOR | No — conservative |
| Exact timeouts | MINOR | No — small perturbation |
| Fixed capacity | MODERATE | Potentially, if capacity degrades under load |
| Permanent failures | MODERATE | No — worst-case assumption |
| Config = runtime | MODERATE | Fundamental limitation of approach |
| No load balancing / replicas | SEVERE | Yes — load calculations wrong by factor of replica count |
| Synchronous calls | MINOR | No — reasonable scope |

---

## 7. Contradictions Between Documents

### 7.1 LoC Estimates

| Source | Total LoC | Novel Core |
|--------|-----------|------------|
| problem_statement.md | 105,000 | 50,000 (48%) |
| final_approach.md | 60,000 | 30,000 (50%) |

**Contradiction:** 75% inflation in the problem statement. The problem_statement.md has not been updated.

**Specific subsystem contradictions:**
- S1 (Config Ingestion): 15,000 → 12,000
- S3 (Semantics Model): 12,000 → merged into BMC encoding
- S7 (Z3 Integration): 6,000 → merged into BMC component
- S9 (Visualization): 8,000 → dropped or reduced

**SEVERITY: MODERATE.** Cosmetic but creates credibility risk. The problem_statement.md should be updated or explicitly superseded.

### 7.2 Variable Count Contradiction

| Source | Claim |
|--------|-------|
| depth_check.md (flagged) | "15K variables" vs "100K+ variables" for 50-service topology |
| final_approach.md | ~9,000 variables for 50 services at 3 vars/service/step |

**Resolution:** final_approach.md resolves with the concrete model (~3 variables per service per step). But the prior "100K+" claim from technical_excerpts.txt (which mentioned 5 variables/service/step for a model including CB state) has not been explicitly retracted. The 3-variable model drops `retry_remaining` and `timeout_remaining` from the per-step variable list but still lists them in the variable definition (lines 86-92 of final_approach.md list 5 variables: state, load, retry_remaining, timeout_remaining, failed).

**Wait — counting the actual variables listed:**
1. state[v,t] — per service per step
2. load[v,t] — per service per step
3. retry_remaining[v,t] — per service per step
4. timeout_remaining[v,t] — per service per step
5. failed[v] — per service (NOT per step)

That's 4 variables per service per step + 1 per service. For 50 services, d*=60:
- Per-step variables: 50 × 60 × 4 = 12,000
- Per-service variables: 50 × 1 = 50
- Total: ~12,050

The claim of "~3 variables per service per step" producing "~9,000 for 50 services" is inconsistent with the 4-per-step variables listed. 50 × 60 × 3 = 9,000, but that requires dropping one of the four per-step variables.

**SEVERITY: MODERATE.** The variable count is either 3/step (if one variable is eliminated or two are merged) or 4/step (as listed). The difference matters: 9,000 vs 12,000 variables. Not fatal but sloppy.

### 7.3 Scalability Claims

| Source | Claim |
|--------|-------|
| Debate majority | 25-35 services ceiling |
| final_approach.md | 30-50 services ceiling |
| problem_statement.md | "n ≤ 500, d ≤ 20, c ≤ 5" (from technical_excerpts) |

The "n ≤ 500" from technical_excerpts refers to a different model (Timed Automata), not the final BMC approach. But it's in the same document and could confuse readers.

**SEVERITY: MINOR.** The final_approach.md is clear about its own claims.

### 7.4 "Zero Human Involvement" vs. Manual Verification

| Source | Claim |
|--------|-------|
| problem_statement.md line 99 | "All evaluation is fully automated with zero human involvement." |
| final_approach.md §7.2 | "manually verify each reported risk" |

**Direct contradiction.** The problem_statement.md claims zero human involvement; the evaluation plan requires manual verification of real-config findings.

**SEVERITY: MODERATE.** The problem_statement.md is the outdated document, but this contradiction should be explicitly resolved.

### 7.5 CEGIS vs. MaxSAT

| Source | Claim |
|--------|-------|
| technical_excerpts.txt | Repair via CEGIS loop |
| final_approach.md | Repair via MaxSAT (CEGIS explicitly rejected) |

**Resolution:** The final_approach explicitly discusses why MaxSAT was chosen over CEGIS. This is a design evolution, not a contradiction. But technical_excerpts.txt should be marked as superseded.

**SEVERITY: MINOR.** Resolved by design decision.

### 7.6 Are V/D/P/F Scores Still Justified?

Current scores: V=6, D=5, P=5, F=7.

**V=6 challenge:** The value score was already deflated from the pre-debate estimate. Given the circuit breaker exclusion and the "47-line script covers 60-70%" finding, V=6 may still be generous. If graph analysis catches 70-85% of bugs, the *marginal* value of the formal approach (BMC+MaxSAT) might warrant V=4-5 for the *incremental* contribution.

**D=5 challenge:** If B6 has the proof gaps identified in §1 above, the difficulty increases — a correct proof of monotonicity (or a corrected theorem) would be harder than the current sketch suggests. Revised D=5-6 if the proof requires restructuring.

**P=5 challenge:** Hinges entirely on evaluation. No change to assessment.

**F=7 challenge:** If the monotonicity theorem needs restructuring, the feasibility of monotonicity-aware pruning is at risk. Without pruning, the MinUnsat enumeration is exponential, making the 30-50 service target infeasible. Revised F=5-6 if B6 requires significant rework.

**SEVERITY: MODERATE.** Scores should be reassessed after resolving the B6 proof issues.

---

## 8. "Why Not a 50-Line Script?" Challenge

### 8a: False negative — script misses a real cascade

**Scenario: Fan-in load accumulation**

```
Topology:
  A → C (retry=2)
  B → C (retry=2)
  C → D (retry=3)

  capacity(C) = 30
  capacity(D) = 100
  baseline: L(A) = 5, L(B) = 5
```

**Script analysis (per-path):**
- Path A→C→D: amplification = (1+2)×(1+3) = 12. Load at D from this path: 5×12 = 60
- Path B→C→D: amplification = (1+2)×(1+3) = 12. Load at D from this path: 5×12 = 60
- Max single-path load at D: 60. D's capacity is 100. Script says: SAFE ✓

**BMC analysis (fan-in):**
- Load at C = L(A)×(1+2) + L(B)×(1+2) = 15 + 15 = 30
- Load at D = L(C)×(1+3) = 30×4 = 120 > 100 = capacity(D). CASCADE ✗

**The script misses this because it analyzes paths independently.** The combined load from two safe paths exceeds capacity at the fan-in point.

**Counter-argument:** A slightly smarter script (sum amplifications at fan-in nodes) would catch this. It's about 60 lines instead of 47. This is the strongest argument for BMC's marginal value — but it can be incrementally patched.

### 8b: False positive that BMC avoids

**Scenario: Over-conservative path analysis**

```
Topology:
  A → B → C → D (chain)
  retry(A→B) = 3, retry(B→C) = 3, retry(C→D) = 3
  capacity(D) = 100
  baseline: L(A) = 1
```

**Script analysis:**
- Path amplification: (1+3)^3 = 64. Load at D: 1 × 64 = 64.
- Script says: 64 < 100 = capacity(D). SAFE ✓
- Hmm, that's not a false positive.

**Revised scenario with conservative fan-in:**

```
Topology:
  A → C (retry=2)
  B → C (retry=2)
  C → D (retry=2)
  capacity(D) = 50
  baseline: L(A) = 5, L(B) = 5
  Only A fails (not B) — k=1 failure budget
```

**Script (worst-case, ignoring failure semantics):**
- Assumes both A and B contribute load (doesn't model failure conditions)
- Path A→C→D: 5×3×3 = 45
- Path B→C→D: 5×3×3 = 45
- Total at D (sum): 45+45 = 90 > 50. Script says: CASCADE ✗

**BMC analysis (failure-conditional):**
- A fails → A does NOT send load to C. A's load is 0.
- Only B is active: L(C) = L(B)×(1+2) = 15. L(D) = 15×3 = 45 < 50.
- With B as the single failure: L(C) = L(A)×3 = 15. Same result.
- BMC says: SAFE for k=1 ✓

**The script reports a false alarm because it doesn't model which services are failed.** It computes worst-case amplification assuming all load sources are active, but the failure set determines which sources contribute. BMC distinguishes between "A is the failed service (not sending load)" and "B is the failed service."

### 8c: BMC provides genuinely new information

**Genuinely new capabilities:**

1. **Failure set identification:** "Services {B, E} failing simultaneously cascades D" — the script can't discover *which* failure combinations are dangerous.

2. **Minimal failure sets:** "The smallest set that cascades D is {B} alone, but {A,C} also cascades D through a different mechanism" — compositional failure analysis.

3. **Counterexample traces:** "At step 3, load on C exceeds capacity because B retried 3× and A retried 2×, producing combined load of 47" — step-by-step cascade reconstruction.

4. **Repair synthesis:** "Reduce retry(B→C) from 3 to 2 (minimum change to eliminate all discovered cascades)" — optimization over the failure set space.

5. **Completeness guarantee:** "No failure set of size ≤ 2 cascades D" — exhaustive verification, not sampling.

**Honest assessment:** Items 1-3 provide genuine new information that a script cannot easily replicate. Item 4 requires optimization machinery. Item 5 is the strongest theoretical contribution but is useful only if the scaling ceiling is high enough to cover realistic topologies.

**The script's real weakness:** It conflates "which services are sending load" with "which services have failed." A failed service doesn't send load — it *receives* retries and generates errors. The script treats all services as simultaneously active and all as potential failure points, producing both false positives (all sources active when some should be failed) and false negatives (paths analyzed independently when fan-in combines them).

---

## 9. Summary: Kill List

### FATAL Vulnerabilities

| ID | Vulnerability | Description |
|----|---------------|-------------|
| **K1** | **B6 monotonicity: blocked-path counterexample** | The proof sketch's claim that retry-generated load compensates for blocked-path load reduction is not supported by the load propagation equation. When an intermediary service w fails, its downstream services see strictly less load (w forwards nothing), and w's predecessors' retries don't create compensating downstream load. Pointwise load monotonicity (load_F(v,t) ≤ load_F'(v,t) for ALL v) appears false for downstream services of a newly failed intermediary. |
| **K2** | **B6 proof: "alternative paths" argument invalid** | The proof sketch claims predecessors "route through alternative paths" — but RTIGs are static dependency graphs. Services don't dynamically reroute. This is a fundamental gap in the proof's key argument. |

### SEVERE Vulnerabilities

| ID | Vulnerability | Description |
|----|---------------|-------------|
| **K3** | **B3 completeness bound wrong for cyclic graphs** | d* = D × R doesn't account for cyclic load amplification. A cycle can amplify load exponentially per circuit, requiring more steps than D × R to reach fixpoint. |
| **K4** | **D4 FPT claim unsubstantiated** | Courcelle's theorem doesn't apply (cascade reachability involves arithmetic). Custom DP likely pseudo-polynomial, not FPT. The 2^O(w) bound hides potentially impractical constants. |
| **K5** | **Hidden assumption: no replica/load-balancing modeling** | The load propagation equation doesn't account for replica counts. All load calculations are off by a factor of replica_count, producing massive false positive rates. |

### MODERATE Vulnerabilities

| ID | Vulnerability | Description |
|----|---------------|-------------|
| **K6** | **B1 reduction under-specified** | SUBSET-SUM reduction sketch has semantic issues (+1 offset, inverted failure semantics). Plausible but needs careful construction. |
| **K7** | **Variable count inconsistency** | 4 per-step variables listed but "~3" claimed. Difference: 12,000 vs 9,000 for 50-service topology. |
| **K8** | **problem_statement.md outdated** | 105K→60K LoC, "zero human involvement" contradicts evaluation plan. Not updated after depth check. |
| **K9** | **MaxSAT repair completeness depends on B6+B3** | If monotonicity or completeness bound is wrong, enumeration is incomplete, and repairs don't guarantee cascade-freedom. |
| **K10** | **Treewidth claim "typically ≤ 5-8" unsupported** | No empirical evidence. Real topologies with cross-cutting concerns may have higher treewidth. |
| **K11** | **Fixed capacity interacts with monotonicity** | If capacity degrades under load (realistic), the propagation equation changes, potentially breaking monotonicity. |
| **K12** | **Config-to-runtime gap** | Static analysis of configs may not reflect runtime behavior. Fundamental limitation that needs prominent discussion. |

### MINOR Vulnerabilities

| ID | Vulnerability | Description |
|----|---------------|-------------|
| **K13** | **D3 Σ₂ᴾ characterization vs NP implementation** | The implementation avoids the ∀ quantifier. Σ₂ᴾ is theoretical worst case only. Framing issue. |
| **K14** | **Instantaneous retries overestimate cascade speed** | Conservative but may produce false positives for time-sensitive cascades. |
| **K15** | **Synchronous call model assumption** | Async messaging has different semantics. Should be explicit scope restriction. |
| **K16** | **CEGIS→MaxSAT evolution not reconciled in excerpts** | technical_excerpts.txt still mentions CEGIS. Should be marked superseded. |
| **K17** | **Fast-failure timing unmodeled** | Dead services return errors faster than slow services. Model doesn't distinguish. |

---

## 10. Recommended Priority Actions

1. **FIX B6 IMMEDIATELY.** The monotonicity proof is the load-bearing beam of the entire project. Either (a) find a correct proof strategy that handles blocked paths, (b) reformulate the theorem to something provable (e.g., monotonicity of the cascade predicate via a different argument, not pointwise load comparison), or (c) restrict the topology class to eliminate the counterexample (e.g., require that every service is on a path from a potential failure source to the cascade target).

2. **FIX B3 for cycles.** Either restrict to DAGs, prove a corrected bound, or acknowledge the limitation with a conservative over-estimate.

3. **Address replica counts** in the formal model. The load propagation equation must include division by replica count or the capacity must be scaled.

4. **Make the B1 reduction explicit.** Write out the full reduction with all details, handling the +1 offset and failure semantics.

5. **Substantiate or retract D4.** Either provide a rigorous DP algorithm with precise complexity, or retract the FPT claim and state it as a conjecture.

6. **Reconcile documents.** Update problem_statement.md or mark it as superseded. Fix the variable count (3 vs 4 per step).
