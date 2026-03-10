# Cost Algebra Specification

## Compositional Cognitive Cost Algebra — Formal Reference

This document specifies the compositional cost algebra used by the Bounded-Rational
Usability Oracle to model how cognitive costs combine across sequential and parallel
task operations.

---

## Table of Contents

- [Overview](#overview)
- [Cost Elements](#cost-elements)
- [Sequential Composition (⊕)](#sequential-composition-)
- [Parallel Composition (⊗)](#parallel-composition-)
- [Context Modulation (Δ)](#context-modulation-Δ)
- [Expression Trees](#expression-trees)
- [Algebraic Properties](#algebraic-properties)
- [Soundness Theorem](#soundness-theorem)
- [Parameter Calibration](#parameter-calibration)
- [Implementation Reference](#implementation-reference)
- [Examples](#examples)

---

## Overview

The cost algebra provides a principled way to compose cognitive costs from individual
operations into aggregate task costs. It replaces naive summation with operators that
model real cognitive phenomena:

- **Prior cognitive load amplifies subsequent operations** (sequential coupling)
- **Concurrent operations interfere via shared resources** (parallel interference)
- **Context (fatigue, practice, stress) modulates all costs** (context modulation)

The algebra operates on **cost elements** and produces **cost expressions** (trees) that
can be evaluated, optimized, and verified for soundness.

---

## Cost Elements

A cost element is a 4-tuple `c = (μ, σ², κ, λ)`:

| Component | Symbol | Description | Range | Unit |
|-----------|--------|-------------|-------|------|
| Mean cost | μ | Expected time for the operation | [0, ∞) | seconds |
| Variance | σ² | Variability in completion time | [0, ∞) | seconds² |
| Capacity utilization | κ | Fraction of cognitive capacity consumed | [0, 1] | dimensionless |
| Interference susceptibility | λ | Sensitivity to concurrent operations | [0, 1] | dimensionless |

### Construction from Cognitive Laws

| Cognitive Law | μ | σ² | κ | λ |
|---------------|---|-----|---|---|
| Fitts' Law (motor) | `a + b·log₂(1+D/W)` | `(0.1·μ)²` | `ID / ID_max` | 0.1 (motor-only) |
| Hick–Hyman (choice) | `a + b·log₂(n)` | `(0.15·μ)²` | `log₂(n) / log₂(n_max)` | 0.3 (moderate) |
| Visual search (perceptual) | `slope · n_items` | `(slope · n/4)²` | `n / n_max` | 0.5 (high) |
| Working memory (memory) | `load_cost(items)` | varies | `items / capacity` | 0.6 (very high) |

### Implementation

```python
from usability_oracle.algebra.models import CostElement

# Create from raw values
click = CostElement(mu=0.3, sigma_sq=0.01, kappa=0.2, lam=0.1)

# Properties
click.expected_cost()           # 0.3
click.std_dev()                 # 0.1
click.coefficient_of_variation() # 0.333
click.to_interval()             # Interval(0.2, 0.4) approx ±1σ
click.tail_probability(0.5)     # P(cost > 0.5)

# Special elements
zero = CostElement.zero()       # (0, 0, 0, 0) — identity for ⊕
```

---

## Sequential Composition (⊕)

Sequential composition models the cost of performing operation `b` after operation `a`.
The key insight: **prior cognitive load amplifies subsequent costs**.

### Definition

```
a ⊕ b = (μ_ab, σ²_ab, κ_ab, λ_ab)
```

where:

```
μ_ab  = μ_a + μ_b + γ · κ_a · μ_b
σ²_ab = σ²_a + σ²_b + γ² · κ²_a · σ²_b
κ_ab  = max(κ_a, κ_b · (1 + γ · κ_a))   capped at 1.0
λ_ab  = max(λ_a, λ_b)
```

### Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Sequential coupling | γ | 0.15 | [0, 0.5] | How much prior load amplifies subsequent cost |

### Intuition

When `γ = 0`, the operator reduces to simple addition (the MVP baseline). When
`γ > 0`, performing a high-capacity operation (high κ_a) before another operation
inflates the second operation's cost. This models cognitive load propagation:
after a demanding visual search (high κ), a subsequent choice reaction is slower.

### Chain Composition

For a sequence of n operations:

```python
composer = SequentialComposer()
total = composer.compose_chain(
    elements=[step1, step2, step3, step4],
    couplings=[0.15, 0.10, 0.20],  # per-transition coupling
)
```

### Interval-Valued Composition

When coupling γ is uncertain:

```python
total = composer.compose_interval(
    a=step1,
    b=step2,
    coupling_interval=Interval(0.10, 0.20),
)
# Returns CostElement with interval-valued μ
```

### Sensitivity Analysis

```python
partials = composer.sensitivity(a=step1, b=step2, coupling=0.15, delta=0.01)
# Returns partial derivatives ∂μ/∂μ_a, ∂μ/∂μ_b, ∂μ/∂γ, etc.
```

---

## Parallel Composition (⊗)

Parallel composition models the cost of two operations performed concurrently on
different cognitive channels. Based on Wickens' Multiple Resource Theory (MRT).

### Definition

```
a ⊗ b = (μ_ab, σ²_ab, κ_ab, λ_ab)
```

where:

```
μ_ab  = max(μ_a, μ_b) + α · λ_a · λ_b · min(μ_a, μ_b)
σ²_ab = max(σ²_a, σ²_b) + α² · λ²_a · λ²_b · min(σ²_a, σ²_b)
κ_ab  = κ_a + κ_b − κ_a · κ_b    (probability-union formula)
λ_ab  = max(λ_a, λ_b)
```

### Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Parallel interference | α | 0.3 | [0, 1] | Degree of cross-channel interference |

### Multiple Resource Theory Integration

The `ParallelComposer` incorporates Wickens' MRT channel taxonomy:

| Channel Dimension | Values |
|-------------------|--------|
| Processing stage | Perceptual, Cognitive, Response |
| Perceptual modality | Visual, Auditory |
| Visual processing | Focal, Ambient |
| Processing code | Spatial, Verbal |
| Response modality | Manual, Vocal |

The `INTERFERENCE_MATRIX` provides pre-computed interference factors for channel pairs:

```python
# Same-channel interference (high)
ParallelComposer.interference_factor("visual_focal", "visual_focal")  # 0.8

# Cross-modality (low)
ParallelComposer.interference_factor("visual_focal", "auditory")      # 0.1

# Different processing codes (medium)
ParallelComposer.interference_factor("spatial", "verbal")             # 0.2
```

### Channel-Aware Composition

```python
par = ParallelComposer()

# Automatic interference lookup by channel
result = par.compose_with_channels(
    elements=[visual_search, motor_click],
    channels=["visual_focal", "manual"],
)
```

### N-ary Parallel Composition

```python
# Multiple concurrent operations
result = par.compose_group(
    elements=[search, read, click],
    interference=0.25,
)
```

---

## Context Modulation (Δ)

Context modulation adjusts costs based on the user's cognitive state.

### Definition

```
Δ_θ(c) = (μ · m(θ), σ² · m(θ)², κ · min(1, m(θ)), λ)
```

where `m(θ)` is the total modulation multiplier computed from context factors.

### Context Factors

| Factor | Symbol | Range | Effect |
|--------|--------|-------|--------|
| Fatigue | f | [0, 1] | Increases μ by up to 50% |
| Working memory load | w | [0, 1] | Super-linear increase when w > 0.75 |
| Practice (trials) | p | [0, ∞) | Power-law decrease: μ · p^{-0.4} |
| Stress | s | [0, 1] | Inverted-U (Yerkes-Dodson): optimal at s ≈ 0.4 |
| Age factor | a | [0, 1] | U-shaped: fastest at a ≈ 0.35 (young adult) |

### Modulation Functions

```
m_fatigue(f)   = 1 + 0.5 · f
m_memory(w)    = 1 + 0.3·w + 0.7·w⁴           (super-linear near capacity)
m_practice(p)  = max(0.5, p^{-0.4})            (power law of practice)
m_stress(s)    = 1 + 0.2·(2s−1)² − 0.2         (Yerkes-Dodson inverted U)
m_age(a)       = 1 + 0.3·|a − 0.35|            (optimal at young adult)

m(θ) = m_fatigue · m_memory · m_practice · m_stress · m_age
```

### Implementation

```python
from usability_oracle.algebra.context import CognitiveContext, ContextModulator

context = CognitiveContext(
    fatigue=0.3,        # 30% fatigue
    wm_load=0.6,        # 60% working memory utilization
    practice=50,         # 50 prior trials
    stress=0.4,          # moderate (optimal) stress
    age=0.35,            # young adult
)

modulator = ContextModulator()
adjusted = modulator.modulate(click_cost, context)
multiplier = modulator.total_multiplier(context)
print(f"Total modulation: {multiplier:.2f}x")
```

---

## Expression Trees

Cost compositions build expression trees that can be evaluated, optimized, and inspected.

### Node Types

| Type | Class | Description |
|------|-------|-------------|
| Leaf | `Leaf(element)` | A single cost element |
| Sequential | `Sequential(left, right)` | left ⊕ right |
| Parallel | `Parallel(left, right)` | left ⊗ right |
| Context | `ContextMod(child, context)` | Δ_θ(child) |

### Example Tree

```
      Sequential(⊕)
      /            \
  Leaf(search)   Parallel(⊗)
                 /          \
           Leaf(read)    ContextMod(Δ)
                             |
                         Leaf(click)
```

### Evaluation

```python
expr = Sequential(
    Leaf(search),
    Parallel(
        Leaf(read),
        ContextMod(Leaf(click), context),
    ),
)

result = expr.evaluate()  # Returns CostElement
```

### Optimization

The `AlgebraicOptimizer` simplifies expression trees:

```python
from usability_oracle.algebra.optimizer import AlgebraicOptimizer

opt = AlgebraicOptimizer()
simplified = opt.optimize(expr)

# Rewrites:
# - Flatten nested Sequential chains
# - Flatten nested Parallel groups
# - Eliminate zero-cost leaves
# - Factor common sub-expressions
# - Reorder commutative Parallel operands
```

### Task Graph Composition

For DAG-structured tasks, `TaskGraphComposer` handles the composition:

```python
from usability_oracle.algebra.composer import TaskGraphComposer

composer = TaskGraphComposer()
total = composer.compose(task_graph, cost_map)
critical = composer.critical_path_cost()
bottlenecks = composer.bottleneck_nodes()
```

---

## Algebraic Properties

### Verified Properties

The `SoundnessVerifier` checks these axioms:

| Property | ⊕ (Sequential) | ⊗ (Parallel) |
|----------|-----------------|---------------|
| Associativity | ✓ (a⊕b)⊕c ≈ a⊕(b⊕c) | ✓ (a⊗b)⊗c ≈ a⊗(b⊗c) |
| Identity | ✓ a⊕0 = a | ✓ a⊗0 = a |
| Monotonicity | ✓ a≤a' ⟹ a⊕b ≤ a'⊕b | ✓ a≤a' ⟹ a⊗b ≤ a'⊗b |
| Commutativity | ✗ (order matters) | ✓ a⊗b = b⊗a |
| Triangle ineq. | ✓ μ(a⊕b) ≤ μ(a) + μ(b) + γ·μ(b) | ✓ |

### Verification

```python
from usability_oracle.algebra.soundness import SoundnessVerifier

verifier = SoundnessVerifier()

# Verify specific properties
r1 = verifier.verify_monotonicity(a, b, a_prime)
r2 = verifier.verify_commutativity(a, b)  # For ⊗ only
r3 = verifier.verify_identity(a)

# Verify entire expression tree
results = verifier.verify_all(expression)
for r in results:
    print(f"{r.property}: {r.status}")  # PASS / FAIL / WARN
```

---

## Soundness Theorem

### Statement

For any task graph G with n operations, the composed cost satisfies:

```
C_alg(G) ≥ Σ_{t=1}^{n} I(S_t; A_t)
```

where I(S_t; A_t) is the mutual information at step t in the full discrete-event
cognitive simulation. The algebra provides a **conservative upper bound** on the true
information cost.

### Tighter Bound (Tree-Structured Graphs)

For tree-structured task graphs with bounded interference degree k:

```
C_alg(G) ≤ (1 + O(kε/β)) · Σ_t I(S_t; A_t)
```

### Proof Sketch

1. **Base case:** For a single operation, I(S_t; A_t) ≤ μ by the rate-distortion
   principle (each cognitive channel is a noisy channel from state to action).

2. **Sequential (⊕):** The coupling term γ·κ_a·μ_b bounds I(S_{t+1}; A_{t+1} | S_t, A_t)
   by the data-processing inequality applied to the cascade S_t → A_t → S_{t+1} → A_{t+1}.

3. **Parallel (⊗):** The interference term α·λ_a·λ_b·min(μ_a, μ_b) bounds the mutual
   information I(A^(1); A^(2) | S) between concurrent channels sharing a resource.

4. **Context (Δ):** Scaling preserves the bound by monotonicity of mutual information
   under channel degradation.

### Assumptions

The theorem requires:

1. **Markov transitions** — state at t+1 depends only on state and action at t
2. **Conditionally independent channels** — given world state, sensory observations
   across modalities are independent (Multiple Resource Theory assumption)
3. **Bounded interference degree k** — each operation interferes with at most k others

---

## Parameter Calibration

### Component-Level Parameters (Published)

| Law | Parameter | Value | Source |
|-----|-----------|-------|--------|
| Fitts | a (intercept) | 0.050 s | MacKenzie 1992 |
| Fitts | b (slope) | 0.150 s/bit | MacKenzie 1992 |
| Hick–Hyman | a (intercept) | 0.200 s | Hick 1952 |
| Hick–Hyman | b (slope) | 0.155 s/bit | Hyman 1953 |
| Working memory | capacity | 4 chunks | Cowan 2001 |
| Working memory | decay rate | 0.077 /s | Barrouillet et al. 2004 |
| Visual search | serial slope | 0.025 s/item | Wolfe 1998 |

### Composition Parameters (Novel — Calibrated)

| Parameter | Symbol | Default | Calibration Strategy |
|-----------|--------|---------|---------------------|
| Sequential coupling | γ | 0.15 | Ordinal stability across [0, 0.5] |
| Parallel interference | α | 0.30 | MRT-based interference matrix |

Calibration proceeds via:

1. **Sensitivity analysis:** verify ordinal ranking stability across γ ∈ [0, 0.5], α ∈ [0, 1]
2. **Retrospective fit:** minimize rank-order disagreement vs. published data
3. **Ablation baseline:** γ = α = 0 (additive model) as parameter-free reference

---

## Implementation Reference

| Class | Module | Key Methods |
|-------|--------|-------------|
| `CostElement` | `algebra/models.py` | `expected_cost()`, `to_interval()`, `zero()` |
| `SequentialComposer` | `algebra/sequential.py` | `compose()`, `compose_chain()`, `sensitivity()` |
| `ParallelComposer` | `algebra/parallel.py` | `compose()`, `compose_with_channels()` |
| `ContextModulator` | `algebra/context.py` | `modulate()`, `total_multiplier()` |
| `TaskGraphComposer` | `algebra/composer.py` | `compose()`, `critical_path_cost()` |
| `SoundnessVerifier` | `algebra/soundness.py` | `verify_all()`, `verify_monotonicity()` |
| `AlgebraicOptimizer` | `algebra/optimizer.py` | `optimize()`, `pretty_print()` |

---

## Examples

### Example 1: Simple Login Flow

```python
from usability_oracle.algebra import CostElement, SequentialComposer

# Individual operations
find_username = CostElement(mu=0.8, sigma_sq=0.04, kappa=0.4, lam=0.3)
click_username = CostElement(mu=0.2, sigma_sq=0.005, kappa=0.15, lam=0.1)
type_username = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.5, lam=0.2)
find_password = CostElement(mu=0.5, sigma_sq=0.02, kappa=0.3, lam=0.3)
type_password = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.5, lam=0.2)
click_submit = CostElement(mu=0.3, sigma_sq=0.01, kappa=0.2, lam=0.1)

# Compose sequentially
seq = SequentialComposer()
total = seq.compose_chain(
    [find_username, click_username, type_username,
     find_password, type_password, click_submit],
    couplings=[0.15] * 5,
)

print(f"Expected total cost: {total.mu:.2f}s")
print(f"Std deviation: {total.std_dev():.2f}s")
```

### Example 2: Dashboard with Concurrent Tasks

```python
from usability_oracle.algebra import ParallelComposer

# Reading a chart while monitoring a notification
read_chart = CostElement(mu=2.0, sigma_sq=0.3, kappa=0.7, lam=0.5)
notice_alert = CostElement(mu=0.5, sigma_sq=0.02, kappa=0.2, lam=0.4)

par = ParallelComposer()
concurrent = par.compose_with_channels(
    elements=[read_chart, notice_alert],
    channels=["visual_focal", "visual_ambient"],
)
# Focal-ambient interference is low → minimal cost increase
```
