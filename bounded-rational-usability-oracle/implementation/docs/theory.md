# Mathematical Foundations

## Bounded-Rational Usability Oracle — Theory Reference

This document describes the mathematical framework underlying the Bounded-Rational
Usability Oracle: the free-energy formulation of bounded rationality, its connection to
classical cognitive laws, and the three theoretical contributions (bisimulation, cost
algebra, bottleneck taxonomy).

---

## Table of Contents

- [Information-Theoretic Bounded Rationality](#information-theoretic-bounded-rationality)
- [Free-Energy Formulation](#free-energy-formulation)
- [Recovering Classical Cognitive Laws](#recovering-classical-cognitive-laws)
- [Contribution 1: Bounded-Rational Bisimulation](#contribution-1-bounded-rational-bisimulation)
- [Contribution 2: Compositional Cost Algebra](#contribution-2-compositional-cost-algebra)
- [Contribution 3: Bottleneck Taxonomy](#contribution-3-bottleneck-taxonomy)
- [Consistency Oracle Guarantee](#consistency-oracle-guarantee)
- [References](#references)

---

## Information-Theoretic Bounded Rationality

The system models UI users as **bounded-rational agents** in the sense of Ortega & Braun
(2013). A fully rational agent would select the action minimizing expected cost at each
decision point. A bounded-rational agent faces an information-processing constraint: the
mutual information between the agent's internal state and its action is limited by a
capacity parameter β.

### The Decision Problem

At each state `s` in a task-state MDP, the agent selects an action `a` from available
actions `A(s)`. The interaction channel is:

```
  s → [cognitive processing, capacity β] → a
```

The agent's policy `π(a|s)` is the conditional distribution over actions given the state.
A fully rational agent selects:

```
  π*(a|s) = argmin_a cost(s, a)
```

A bounded-rational agent instead solves a constrained optimization:

```
  minimize   E_π[cost(s, a)]
  subject to I(S; A) ≤ C
```

where `I(S; A)` is the mutual information between state and action, and `C` is the
agent's information-processing capacity.

---

## Free-Energy Formulation

The constrained optimization above is equivalent (via Lagrange duality) to minimizing
the **free energy**:

```
  F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)
```

where:

- `E_π[cost]` is the expected task cost under policy π
- `D_KL(π ‖ p₀)` is the KL divergence from the prior policy p₀
- `β` is the inverse temperature (rationality parameter)
- `p₀(a|s)` is the prior (default, uninformed) policy

The optimal bounded-rational policy takes the **softmax** form:

```
  π_β(a|s) = p₀(a|s) · exp(−β · Q(s, a)) / Z(s)
```

where `Q(s, a)` is the state-action value function and `Z(s)` is the normalizing
partition function:

```
  Z(s) = Σ_a p₀(a|s) · exp(−β · Q(s, a))
```

### Interpretation of β

| β | Behavior | Analogy |
|---|----------|---------|
| β → 0 | Random (prior) behavior | Completely overwhelmed user |
| β = 1–5 | Moderate rationality | Typical user under cognitive load |
| β = 5–20 | High rationality | Expert or focused user |
| β → ∞ | Fully rational (argmin) | Optimal performance |

The system analyzes behavior across a **range** of β values to account for population
variability (novice through expert users).

### Implementation

The `FreeEnergyComputer` class in `policy/free_energy.py` implements:

- `compute(policy, mdp, beta, prior) → float` — evaluates F(π)
- `decompose(policy, mdp, beta, prior) → FreeEnergyDecomposition` — separates
  expected cost from information cost per state
- `optimal_policy(mdp, beta, prior) → Policy` — finds the optimal π_β via
  soft value iteration
- `rate_distortion_curve(mdp, betas, prior) → list[(info_cost, expected_cost)]` —
  traces the Pareto frontier

The `SoftmaxPolicy` class in `policy/softmax.py` implements the softmax form:

- `from_q_values(q_values, beta, prior) → Policy`
- `kl_divergence(policy, prior, state) → float`
- `mutual_information(policy, prior) → float`

---

## Recovering Classical Cognitive Laws

The bounded-rational framework unifies classical HCI laws as capacity-constrained
channels:

### Fitts' Law (Motor Channel)

```
  MT = a + b · log₂(1 + D/W)
```

Motor pointing is a capacity-constrained channel from intended target to motor output.
The index of difficulty `ID = log₂(1 + D/W)` is the mutual information required for
accurate pointing. Fitts' slope `b` is the reciprocal of motor throughput (bits/s).

**Implementation:** `cognitive/fitts.py` — `FittsLaw.predict(distance, width)`

### Hick–Hyman Law (Choice Channel)

```
  RT = a + b · log₂(n)
```

Choice reaction time reflects the information that must be transmitted through the
decision channel: `log₂(n)` bits for `n` equiprobable alternatives. For unequal
probabilities: `RT = a + b · H(p)` where `H(p)` is Shannon entropy.

**Implementation:** `cognitive/hick.py` — `HickHymanLaw.predict(n_alternatives)` and
`predict_unequal_probabilities(probabilities)`

### Visual Search (Perceptual Channel)

Serial search time scales linearly with set size because each item requires a fixation
that transmits a bounded amount of information. Guided search reduces the effective set
size by a guidance factor.

**Implementation:** `cognitive/visual_search.py` — `VisualSearchModel.predict_serial()`,
`predict_parallel()`, `predict_guided()`

### Working Memory (Temporal Channel)

Information decays over time: `P(recall) ∝ exp(−decay_rate · delay)`. The capacity
limit (≈4 chunks) reflects the temporal channel's bandwidth. Proactive interference
further degrades recall.

**Implementation:** `cognitive/working_memory.py` —
`WorkingMemoryModel.predict_recall_probability(items, delay)`

---

## Contribution 1: Bounded-Rational Bisimulation

### Motivation

Production UIs generate MDPs with 10⁴–10⁶ raw states. Exact inference is intractable.
Classical bisimulation merges states with identical transition structure, but this ignores
the agent's capacity constraints. A bounded-rational agent may be unable to distinguish
states that are technically different, making further merging possible.

### Definition

Given an MDP (S, A, T, R) and bounded-rational policy family π_β, define the **cognitive
distance metric**:

```
  d_cog(s₁, s₂) = sup_{β' ≤ β} d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))
```

An abstraction φ: S → Ŝ is a **bounded-rational ε-bisimulation** if:

```
  φ(s₁) = φ(s₂)  ⟹  d_cog(s₁, s₂) ≤ ε
```

### Trajectory Approximation Theorem

If φ is a bounded-rational ε-bisimulation, then for any task of horizon H:

```
  d_TV(T, T̂) ≤ 1 − (1 − βε)^H
```

For typical parameters (β=5, ε=0.005, H=30): d_TV ≤ 0.53.

### Paired-Comparison Theorem (Key Result)

When both UI versions A and B are abstracted under the **same** partition φ:

```
  |(Ĉ_B − Ĉ_A) − (C_B − C_A)| ≤ O(ε)
```

This is much tighter than the naive O(Hβε) bound. Correlated abstraction errors cancel
in the cost difference because both versions share the same partition.

### Coarsening Corollary

Abstract MDP size scales as:

```
  |Ŝ| = O(β^d · N_ε(F))
```

where d is the metric doubling dimension and N_ε(F) is the ε-covering number of the
perceptual feature space. Empirically: |Ŝ| ≤ 10⁴ for production UIs.

### Implementation

- `bisimulation/cognitive_distance.py` — `CognitiveDistanceComputer` computes d_cog
  via soft value iteration and total variation distance
- `bisimulation/partition.py` — `PartitionRefinement.refine(mdp, beta, epsilon)`
  iteratively refines partitions
- `bisimulation/quotient.py` — `QuotientMDPBuilder.build(mdp, partition)` constructs
  the abstract MDP

---

## Contribution 2: Compositional Cost Algebra

See [cost_algebra.md](cost_algebra.md) for the full specification.

### Summary

Three operators compose `CostElement(μ, σ², κ, λ)` values:

- **Sequential (⊕):** μ₁₊₂ = μ₁ + μ₂ + γ·κ₁·μ₂
- **Parallel (⊗):** μ₁ₓ₂ = max(μ₁,μ₂) + α·λ₁λ₂·min(μ₁,μ₂)
- **Context (Δ_θ):** scales (μ, σ²) by context parameters

### Soundness Theorem

For any task graph G with n operations:

```
  C_alg(G) ≥ Σ_{t=1}^{n} I(S_t; A_t)
```

The algebra provides a **conservative upper bound** on true information cost.

---

## Contribution 3: Bottleneck Taxonomy

See [bottleneck_taxonomy.md](bottleneck_taxonomy.md) for the full specification.

### Summary

Five bottleneck types classified by information-theoretic signatures:

| Type | Condition |
|------|-----------|
| Perceptual | Display entropy exceeds perceptual capacity |
| Choice | Action entropy far exceeds transmitted information |
| Motor | Information reaches decision but not motor execution |
| Memory | Temporal information transmission decays |
| Interference | Spurious dependence between independent channels |

### Exhaustiveness Theorem

Any decision point exceeding the free-energy baseline by δ must exhibit at least one
bottleneck signature with threshold exceedance ≥ f(δ, β).

### Distinguishability Theorem

For generic UI configurations, the five signatures are locally separable (linearly
independent gradients w.r.t. UI parameters).

---

## Consistency Oracle Guarantee

The system makes a **consistency oracle** claim, not a fidelity oracle claim:

1. **What it guarantees:** If UI version B is genuinely worse than A for a given task
   (i.e., a bounded-rational agent incurs higher cognitive cost), the oracle detects
   this with probability ≥ 1 − α, subject to the approximation error ε.

2. **What it does not guarantee:** Absolute prediction of human task completion time.

3. **Formal statement:** Under shared ε-bisimulation with N trajectory samples and
   significance level α:

   ```
   P(detect regression | true regression of magnitude δ) ≥ 1 − α
   provided δ > O(ε) + O(1/√N)
   ```

4. **Validation:** Ordinal agreement (Kendall τ ≥ 0.6) between model cost orderings
   and published human performance orderings.

---

## References

1. Ortega, P.A. & Braun, D.A. (2013). Thermodynamics as a theory of decision-making
   with information-processing costs. *Proc. Royal Society A*, 469(2153).
2. MacKenzie, I.S. (1992). Fitts' law as a research and design tool in HCI.
   *Human-Computer Interaction*, 7(1), 91–139.
3. Wickens, C.D. (2002). Multiple resources and performance prediction.
   *Theoretical Issues in Ergonomics Science*, 3(2), 159–177.
4. Tishby, N., Pereira, F.C. & Bialek, W. (2000). The information bottleneck method.
   *Proc. 37th Allerton Conference*.
5. John, B.E. & Salvucci, D.D. (2005). Multipurpose prototypes for assessing user
   interfaces. *IEEE Pervasive Computing*, 4(4), 27–34.
6. Hick, W.E. (1952). On the rate of gain of information. *Quarterly Journal of
   Experimental Psychology*, 4(1), 11–26.
7. Fitts, P.M. (1954). The information capacity of the human motor system in
   controlling the amplitude of movement. *J. Experimental Psychology*, 47(6), 381–391.
8. Miller, G.A. (1956). The magical number seven, plus or minus two. *Psychological
   Review*, 63(2), 81–97.
