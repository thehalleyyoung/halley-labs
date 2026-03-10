# Competing Approaches: Bounded-Rational Usability Oracle

Three competing approaches for building a bounded-rational usability oracle, each targeting different points in the theory–practicality trade-off space.

---

## Approach A: Full-Theory Bisimulation Machine — "The Cognitive Type Checker"

**One-line summary:** A formally verified consistency oracle that reduces usability regression detection to bounded-rational bisimulation quotients over UI-state MDPs, with a compositional cost algebra whose soundness is proved against information-theoretic ground truth.

### 1. Extreme Value Delivered

Enterprise design-system teams maintaining 500+ components where usability failures carry regulatory or safety consequences (healthcare EHR, financial trading platforms, government digital services). Every PR receives a formally grounded cognitive cost diff with provable error bounds. The paired-comparison theorem proves that even if absolute cost predictions are 50% off, relative ordering between UI versions is preserved within O(ε) under shared bisimulation abstraction. Enables calibrated confidence thresholds — teams set regression gates to their risk tolerance with meaningful error bounds.

### 2. Genuine Difficulty

- **Bisimulation quotient at scale**: Computing d\_cog(s₁, s₂) = sup\_{β' ≤ β} d\_TV(π\_{β'}(·|s₁), π\_{β'}(·|s₂)) requires a novel partition-refinement algorithm for β-parameterized metrics. No existing algorithm. O(|S|²·K) naive complexity.
- **Accessibility-tree-to-MDP reduction**: Real trees have implicit state (collapsed menus, modals, scroll containers). Must handle dynamic state with conditional MDPs. Graveyard of prior automated usability tools.
- **Paired-comparison error cancellation**: Requires lazy union construction over heterogeneous UI changes. Union MDP can be exponentially larger.
- **Architecture**: Full pipeline in ≤60s on 4-core laptop for ≤500 elements.

### 3. New Math Required (Load-Bearing Only)

**Theorem 1 (Paired-Comparison Tightness):** For bounded-rational ε-bisimulation φ over union S\_A ∪ S\_B:

> |(Ĉ\_B − Ĉ\_A) − (C\_B − C\_A)| ≤ 2ε·L\_R

where L\_R is the reward Lipschitz constant. For ε = 0.005, c\_max = 2s → error ≤ 20ms. This is the formal heart of the consistency-oracle claim.

**Theorem 2 (Cost Algebra Soundness):** For simulations with Markov transitions, conditionally independent sensory channels, and bounded interference degree k:

> C\_alg(G) ≥ Σ\_t I(S\_t; A\_t)

Proof by induction on task graph using the data-processing inequality.

**Theorem 3 (Bottleneck Exhaustiveness):** Any decision point with cost exceeding the free-energy baseline by δ exhibits at least one of 5 bottleneck signatures with exceedance ≥ f(δ, β) = δ / (1 + β⁻¹·log 5).

**Math Assessment:** Paired-comparison theorem is genuinely load-bearing and probably true under reasonable assumptions. Cost algebra soundness has dimensional mismatch issues (time vs bits) and may be false as stated. Bottleneck exhaustiveness is tautological. ~60% of proposed math is ornamental.

### 4. Best-Paper Potential

Wins at UIST/CHI by establishing usability regression testing as formal verification. The paired-comparison theorem is the "aha" moment — a loose TV bound becomes tight for regression detection. First formal proof that relative usability comparison is easier than absolute prediction.

### 5. Hardest Technical Challenge

Accessibility-tree-to-MDP reduction. Real trees have implicit state, cross-cutting concerns, and platform-specific semantics. Address via incremental parser with MDP debugger, restricted UI grammar covering ~60% of flows, and property-based testing.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 8 | Formal CI gate with calibrated thresholds; regulated-industry need |
| Difficulty | 9 | Novel bisimulation algorithm, MDP reduction, proofs all genuinely hard |
| Potential | 8 | Novel framing + surprising tightness result + practical impact |
| Feasibility | 5 | 12+ months; MDP parser is research project; many proof obligations |

---

## Approach B: The Differential Cost Profiler — "Cognitive git diff"

**One-line summary:** A pragmatic, zero-theory usability regression detector that diffs accessibility trees directly, applies calibrated cognitive cost functions to the structural delta, and reports regressions with bottleneck annotations — no MDPs, no bisimulation, no free energy.

### 1. Extreme Value Delivered

Front-end teams at any scale shipping daily. Zero-config:

```
npm install @usability-oracle/cli && npx usability-diff --before v2.3 --after HEAD --task "checkout-flow"
```

Produces output like:

> ⚠ Step 3: +2.1 bits Hick cost (4→18 options), Bottleneck: Choice paralysis, Suggested fix: Progressive disclosure, Overall: +18% expected task cost

Catches 85% of real regressions (obvious structural changes). 10,000 teams using it imperfectly beats 10 teams with formal guarantees.

### 2. Genuine Difficulty

- **Semantic tree alignment**: Not DOM diffing — semantic alignment where identity is role + label + position + relationships. Must classify changes (additions, removals, reorderings, nestings, visibility changes), each with different cost implications. Adapted tree-edit-distance (RTED) with domain-specific costs.
- **Task-flow extraction without MDP**: Task specs as annotated paths via recording, YAML DSL, or automatic inference from common patterns. Automatic inference is hardest.
- **Robust cost functions under parameter uncertainty**: Interval arithmetic over published parameter ranges. Regression flagged only if entire after-interval exceeds entire before-interval. Conservative but trustworthy.

### 3. New Math Required (Load-Bearing Only)

**Interval Cost Comparison:** C(step) = \[c(θ\_low), c(θ\_high)\]. For Fitts' law, regression reduces to checking whether the structural change log₂(1 + D/W) is positive — parameter-independent for dominant failure modes. This eliminates the calibration problem entirely for common regressions.

**Compositional Interval Propagation:**

> C\_total = ⊕ᵢ C(stepᵢ) where \[a, b\] ⊕ \[c, d\] = \[a + c, b + d\]
>
> Regression test: inf(C\_after) > sup(C\_before)

Conservative but no false positives.

**Math Assessment:** All math is genuinely load-bearing but trivial. No theorems to prove. The parameter-independence insight for dominant failure modes is elegant simplicity. Missing: β sensitivity, monotonicity, error model for additive approximation.

### 4. Best-Paper Potential

Wins at ICSE/FSE by demonstrating practical regression detection at scale — 1,000+ real PRs, F1 ≥ 0.85, zero false positives on benign refactors. Surprising result: you don't need MDPs/bisimulation for 85% of regressions. Explicitly characterizes the detection boundary where Approach A's theory becomes necessary.

### 5. Hardest Technical Challenge

Semantic alignment of accessibility trees across versions. Refactors change tree structure dramatically while preserving semantics. Three-pass alignment: exact match on (role, name, desc), fuzzy match via weighted bipartite matching, then classify unmatched as adds/removes.

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 9 | Immediately usable; zero-config; covers 85% of real regressions |
| Difficulty | 5 | Tree alignment hard; rest is careful engineering |
| Potential | 6 | Strong ICSE paper but unlikely best-paper |
| Feasibility | 9 | 6–8 week build; known algorithms; interval arithmetic trivial |

---

## Approach C: The Adversarial Cognitive Fuzzer — "Usability Chaos Engineering"

**One-line summary:** Synthesize worst-case bounded-rational users — adversarial task strategies that maximize cognitive cost within human-plausible capacity bounds — and use the gap between best-case and worst-case as a robustness metric for UI designs.

### 1. Extreme Value Delivered

Accessibility teams and inclusive design practitioners who care about worst-case users, not average ones. Generates usability stress test reports identifying "cognitive cliffs" — points where small capacity decreases cause large difficulty increases. Transforms usability from mean-estimation to tail-risk analysis. Solves evaluation circularity: "did the UI become more fragile to user variation?" is self-contained and empirically testable, independent of absolute cost calibration.

**Cognitive Fragility metric:**

> F(M) = max\_{β ∈ B\_human} E\_{π\_β}\[C(τ)\] − min\_{β ∈ B\_human} E\_{π\_β}\[C(τ)\]

A model-independent robustness measure comparing a UI to itself across capacity space.

### 2. Genuine Difficulty

- **Adversarial policy synthesis**: β\* = argmax\_{β ∈ B\_human} E\_{π\_β}\[C(τ)\] is non-convex, high-dimensional minimax. Hybrid Bayesian optimization + analytical bounds.
- **Cognitive cliff detection**: Find β where ∂C/∂βᵢ is discontinuous (policy phase transitions). Analytical for small MDPs, binary search for larger ones.
- **Fragility surface in CI/CD budgets**: Need worst-case point (via BO, ~50–100 MDP solves) and cliff locations (binary search). Total ~200 MDP evaluations.
- **Novel algorithm — Capacity-Adversarial MCTS (CA-MCTS)**: Nature player adversarially selects capacity per step; user player selects actions under resulting softmax. Produces worst-case trajectories without full minimax solve.

### 3. New Math Required (Load-Bearing Only)

**Cliff Location Theorem:** For softmax policies, cliffs occur where β\_i·(Q\_i(s, a\*) − Q\_i(s, a')) = 0 for state s on the task-optimal path. Severity ∝ |C(τ\_{a\*}) − C(τ\_{a'})|. Enables analytical cliff detection without exhaustive search.

**Regression via Fragility Theorem:** Fragility regression if F(M\_B) > F(M\_A) + ε\_frag. Strictly weaker than cost regression but more robust — independent of absolute calibration.

**Fragility Decomposition:**

> F(M) ≈ Σ\_t F\_t(M) with correction O(γ·F\_max²)

Enables per-step attribution.

**Math Assessment:** Cliff location theorem is genuinely novel and load-bearing — enables the core algorithm. Fragility definition is clean and self-contained. Fisher metric variant (information geometry) could subsume d\_cog but adds β² sensitivity. Missing: convergence guarantees for BO over non-convex cost landscape.

### 4. Best-Paper Potential

Wins at CHI by reframing usability as inclusive robustness analysis. Connects to hot topics (adversarial robustness, chaos engineering, inclusive design). Cognitive cliff concept is vivid and actionable. Reveals problems no existing method can find — traditional testing with median users misses cliffs entirely.

### 5. Hardest Technical Challenge

Efficient adversarial optimization over capacity parameter space. Cost function is non-convex, potentially discontinuous at cliffs, and expensive to evaluate. Address via: (1) analytical cliff pre-computation from Q-value crossings, (2) adaptive BO seeded with analytical candidates, (3) hierarchical evaluation (cheap MC for exploration, expensive for identified worst-case).

### 6. Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Value | 8 | Unique tail-risk capability; strong inclusive design angle |
| Difficulty | 8 | Novel adversarial optimization + cliff detection + CA-MCTS |
| Potential | 9 | Novel framing + hot topics + vivid results = strong best-paper |
| Feasibility | 6 | Requires MDP infrastructure; adversarial optimization may exceed CI/CD budgets |

---

## Comparative Summary

| Dimension | A: Full Theory | B: Lean Profiler | C: Adversarial Fuzzer |
|-----------|---------------|-----------------|----------------------|
| Value | 8 | 9 | 8 |
| Difficulty | 9 | 5 | 8 |
| Potential | 8 | 6 | 9 |
| Feasibility | 5 | 9 | 6 |
| **Total** | **30** | **29** | **31** |
