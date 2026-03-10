# Bottleneck Taxonomy

## Information-Theoretic Bottleneck Classification — Reference

This document describes the five-type bottleneck taxonomy used by the Bounded-Rational
Usability Oracle to classify *why* a usability regression occurred, enabling targeted
repair recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Information-Theoretic Signatures](#information-theoretic-signatures)
- [Bottleneck Type 1: Perceptual Overload](#bottleneck-type-1-perceptual-overload)
- [Bottleneck Type 2: Choice Paralysis](#bottleneck-type-2-choice-paralysis)
- [Bottleneck Type 3: Motor Difficulty](#bottleneck-type-3-motor-difficulty)
- [Bottleneck Type 4: Memory Decay](#bottleneck-type-4-memory-decay)
- [Bottleneck Type 5: Cross-Channel Interference](#bottleneck-type-5-cross-channel-interference)
- [Theoretical Properties](#theoretical-properties)
- [Detection Pipeline](#detection-pipeline)
- [Repair Mapping](#repair-mapping)
- [Severity Scoring](#severity-scoring)
- [Implementation Reference](#implementation-reference)
- [Examples](#examples)

---

## Overview

When the oracle detects a usability regression (cost increase), it classifies the
failure mode using **information-theoretic signatures** at each decision point in the
task MDP. This transforms the system from a binary regression detector into a
**diagnostic tool** that tells developers *what* went wrong and *how* to fix it.

### The Five Types

| # | Type | One-Line Description |
|---|------|---------------------|
| 1 | **Perceptual** | Too much visual information to process |
| 2 | **Choice** | Too many options to decide among |
| 3 | **Motor** | Target is too small or far away |
| 4 | **Memory** | User must remember too much across steps |
| 5 | **Interference** | Concurrent cognitive channels conflict |

### Classification Flow

```
  Trajectory Statistics
         │
         ▼
  ┌─────────────────────┐
  │  SignatureComputer   │  Compute info-theoretic signatures per state
  │  (signatures.py)     │
  └──────────┬──────────┘
             │  BottleneckSignature per state
             ▼
  ┌─────────────────────┐
  │ BottleneckClassifier │  Classify signatures into bottleneck types
  │  (classifier.py)     │
  └──────────┬──────────┘
             │  list[BottleneckResult]
             ▼
  ┌─────────────────────┐
  │  BottleneckReport    │  Aggregate, rank by severity, summarize
  │  (models.py)         │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    repair_map.py     │  Map bottleneck types to repair strategies
  └─────────────────────┘
```

---

## Information-Theoretic Signatures

At each decision point (state `s`, action `a`) in the task MDP, the `SignatureComputer`
extracts an information-theoretic `BottleneckSignature`:

### Signature Components

| Component | Symbol | Computation | Meaning |
|-----------|--------|-------------|---------|
| Display entropy | H(S\|display) | Entropy of visible elements | Visual complexity |
| Action entropy | H(A\|s) | Entropy of action distribution | Decision difficulty |
| Mutual information | I(S;A) | MI between state and action | Information transmitted |
| Channel utilization | I/C | MI divided by channel capacity | How much capacity is used |
| Motor information | I(A;target) | MI between action and target | Motor precision required |
| Temporal MI | I(S_t; S_{t-k}) | MI between current and past state | Memory retention |
| Cross-channel MI | I(A^(1); A^(2)\|S) | MI between concurrent channels | Interference level |

### Implementation

```python
from usability_oracle.bottleneck.signatures import SignatureComputer

computer = SignatureComputer()
signature = computer.compute(mdp, policy, state_id)

# Access components
print(f"Display entropy: {signature.display_entropy:.3f}")
print(f"Action entropy: {signature.action_entropy:.3f}")
print(f"Mutual information: {signature.mutual_information:.3f}")
print(f"Channel utilization: {signature.channel_utilization:.3f}")

# Quick classification
btype = computer.classify_signature(signature)
```

---

## Bottleneck Type 1: Perceptual Overload

### Signature

```
H(S_t | display) > τ_p
```

Display entropy exceeds the perceptual channel's capacity. The user cannot efficiently
scan and parse the visible information.

### Detection Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Display entropy | > τ_p (default: 4.0 bits) | Too many distinguishable elements |
| Effective set size | > 20 items | Visual search becomes slow |
| Visual density | > 0.7 | Elements are densely packed |
| Saliency variance | < 0.1 | Nothing stands out — flat saliency |

### Common Causes

- Too many interactive elements visible simultaneously
- Poor visual hierarchy (all elements same size/color)
- Missing semantic grouping (ARIA landmarks, headings)
- Information-dense tables without clear structure

### Cognitive Impact

Perceptual overload increases visual search time linearly:

```
search_time = slope × effective_set_size
```

With slope ≈ 25ms/item for serial search, 50 items → 1.25s per search.

### Implementation

The `perceptual.py` module detects this bottleneck by:

1. Computing entropy of element-type distribution in visible viewport
2. Measuring effective visual set size (elements competing for attention)
3. Checking saliency distribution (flat = overload, peaked = good)
4. Comparing against calibrated threshold τ_p

---

## Bottleneck Type 2: Choice Paralysis

### Signature

```
log|A_t| − I(S_t; A_t) > τ_c
```

The action space entropy far exceeds the information transmitted from state to action.
The user knows what state they're in but cannot efficiently select the right action.

### Detection Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Choice gap | > τ_c (default: 2.0 bits) | Large gap between options and decisions |
| Number of alternatives | > 7 (Miller's limit) | Too many choices |
| Policy entropy | > 3.0 bits | Highly uncertain about which action |
| Hick–Hyman RT | > 1.5s | Predicted reaction time is high |

### Common Causes

- Flat navigation with too many top-level options
- Forms with many similar options (e.g., 50 countries without search)
- Unclear labeling (user can't distinguish actions)
- Missing progressive disclosure

### Cognitive Impact

Choice paralysis increases reaction time logarithmically:

```
RT = a + b × log₂(n)
```

With typical parameters: 20 options → 1.07s choice time.

### Implementation

The `choice.py` module detects this bottleneck by:

1. Counting available actions at the state
2. Computing policy entropy H(π(·|s))
3. Computing mutual information I(S;A) under the bounded-rational policy
4. Checking if the gap log|A| − I(S;A) exceeds τ_c

---

## Bottleneck Type 3: Motor Difficulty

### Signature

```
I(A_t; target) < τ_m  given  I(S_t; A_t) ≥ τ_c
```

The user has enough information to decide (high I(S;A)) but cannot execute the motor
action accurately (low I(A;target)). The motor channel is the bottleneck.

### Detection Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Motor information | < τ_m (default: 1.0 bit) | Motor channel is strained |
| Fitts' index of difficulty | > 5.0 bits | Target is small/distant |
| Target width | < 20px | Touch/click target too small |
| Distance-to-width ratio | > 10 | Long reach for small target |
| Fitts' movement time | > 0.8s | Predicted motor time is high |

### Common Causes

- Small click/tap targets (buttons, links)
- Targets far from current cursor/focus position
- Dense layouts where targets are adjacent (risk of mis-clicks)
- Inconsistent target sizes across similar elements

### Cognitive Impact

Motor difficulty increases movement time per Fitts' Law:

```
MT = a + b × log₂(1 + D/W)
```

With D=500px, W=10px → ID=5.7 bits → MT ≈ 0.90s.

### Implementation

The `motor.py` module detects this bottleneck by:

1. Computing Fitts' index of difficulty for target acquisition
2. Checking that decision-level information I(S;A) is sufficient
3. Comparing motor execution cost against threshold τ_m
4. Flagging states where motor cost dominates total cost

---

## Bottleneck Type 4: Memory Decay

### Signature

```
I(S_t; S_{t−k}) < τ_μ  for required k
```

Temporal mutual information between the current state and a state k steps ago has
decayed below the threshold. The user must remember information across steps but
the working memory channel has insufficient bandwidth.

### Detection Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Temporal MI | < τ_μ (default: 0.5 bits) | Past information is lost |
| Working memory load | > 4 chunks | Exceeds typical WM capacity |
| Delay between reference and use | > 10s | Decay has eroded memory |
| Recall probability | < 0.7 | Likely to forget |
| Proactive interference | > 0.3 | Similar prior items cause confusion |

### Common Causes

- Multi-step wizards where earlier inputs affect later decisions
- Reference information on different pages (no persistent cues)
- Confirmation screens that don't show what was entered
- Forms requiring data from external sources (e.g., account numbers)

### Cognitive Impact

Memory decay follows an exponential law:

```
P(recall) = exp(−decay_rate × delay)
```

With decay_rate = 0.077/s: after 10s, P(recall) ≈ 0.46 (worse than chance for some items).

### Implementation

The `memory.py` module detects this bottleneck by:

1. Tracking working memory load across task steps
2. Computing temporal mutual information I(S_t; S_{t-k})
3. Estimating recall probability given delay and chunk count
4. Flagging states where required recall exceeds available WM bandwidth

---

## Bottleneck Type 5: Cross-Channel Interference

### Signature

```
I(A_t^{(1)}; A_t^{(2)} | S_t) > τ_ι
```

Spurious dependence between nominally independent cognitive channels. Two operations
that should be independent are interfering because they share a cognitive resource.

### Detection Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Cross-channel MI | > τ_ι (default: 0.3 bits) | Channels are coupled |
| Interference factor | > 0.5 | MRT predicts high interference |
| Resource overlap | > 0.6 | Shared cognitive resources |
| Dual-task cost | > 20% of single-task cost | Significant dual-task overhead |

### Common Causes

- Reading text while performing spatial motor actions (visual-spatial conflict)
- Listening to audio while reading (auditory-verbal-visual conflict)
- Complex gesture input while maintaining spatial awareness
- Real-time updates competing with user's focused task

### Cognitive Impact

Interference adds a cost proportional to the product of susceptibilities:

```
interference_cost = α × λ₁ × λ₂ × min(μ₁, μ₂)
```

With α=0.3, two high-interference operations (λ=0.6 each): 10.8% overhead.

### Implementation

The `interference.py` module detects this bottleneck by:

1. Identifying concurrent cognitive operations at each state
2. Looking up interference factors from the MRT interference matrix
3. Computing conditional mutual information between action channels
4. Flagging states where interference exceeds τ_ι

---

## Theoretical Properties

### Exhaustiveness Theorem

**Statement:** Any decision point whose total cognitive cost exceeds the free-energy
baseline by more than δ must exhibit at least one of the five bottleneck signatures
with threshold exceedance ≥ f(δ, β).

**Implication:** No regression goes unclassified. If the oracle detects a cost increase,
it will always identify at least one bottleneck type responsible.

### Distinguishability Theorem

**Statement:** For generic UI configurations (outside a measure-zero set), the five
bottleneck signatures have linearly independent gradients with respect to UI parameters.

**Implication:** The taxonomy can usually identify a single dominant bottleneck type.
In degenerate cases (multiple simultaneous bottlenecks), the classifier provides a
soft distribution (posterior weights over types).

### Intervention-Specificity Theorem

**Statement:** Each bottleneck type maps to a distinct repair class, and applying the
corresponding repair reduces that bottleneck's signature while leaving others within
O(ε).

**Implication:** Repairs are targeted — fixing a perceptual bottleneck doesn't
accidentally worsen a motor bottleneck.

---

## Detection Pipeline

### Step-by-Step Process

1. **Compute signatures** for each state in the task MDP:
   ```python
   sig_computer = SignatureComputer()
   signatures = {s: sig_computer.compute(mdp, policy, s) for s in mdp.states}
   ```

2. **Classify each state** using per-type detectors:
   ```python
   classifier = BottleneckClassifier()
   results = classifier.classify(mdp, policy, trajectory_stats, cost_breakdown)
   ```

3. **Aggregate across states** to produce a report:
   ```python
   report = classifier.classify_to_report(mdp, policy, stats, breakdown)
   ```

4. **Rank by severity** and generate actionable summaries:
   ```python
   print(report.generate_summary())
   for bn in report.by_severity(Severity.ERROR):
       print(f"  [{bn.bottleneck_type}] at state {bn.state_id}")
   ```

### Confidence Scoring

Each `BottleneckResult` includes a confidence score [0, 1] based on:

- How far the signature exceeds the threshold
- Consistency across nearby states
- Number of trajectory samples supporting the classification

---

## Repair Mapping

The `repair_map.py` module maps each bottleneck type to concrete repair strategies:

| Bottleneck | Primary Repair | Secondary Repairs |
|------------|---------------|-------------------|
| **Perceptual** | Group related elements (add landmarks, headings) | Increase contrast, add whitespace, reduce density |
| **Choice** | Progressive disclosure (collapse, paginate, search) | Better labeling, group by category, add defaults |
| **Motor** | Increase target size (min 44×44px touch, 24×24px click) | Reduce distance, improve placement, add keyboard shortcuts |
| **Memory** | Add persistent state cues (breadcrumbs, summaries) | Reduce steps, show context, avoid mode-dependent behavior |
| **Interference** | Separate modalities (visual vs. auditory) | Serialize instead of parallelize, reduce concurrency |

### Repair Synthesis Integration

When the repair synthesizer is enabled, the repair map constrains the Z3 search space:

```python
# In repair/strategies.py
def strategy_for_bottleneck(bn_type: BottleneckType) -> list[MutationType]:
    return REPAIR_MAP[bn_type]

# In repair/synthesizer.py
def synthesize(mdp, bottlenecks, constraints, timeout):
    for bn in bottlenecks:
        strategies = strategy_for_bottleneck(bn.bottleneck_type)
        # Encode mutation constraints for Z3
        ...
```

---

## Severity Scoring

### Severity Levels

| Level | Score Range | Description |
|-------|------------|-------------|
| ERROR | ≥ 0.8 | Severe bottleneck — likely to cause task failure |
| WARNING | 0.4 – 0.8 | Moderate bottleneck — degrades performance |
| INFO | < 0.4 | Minor bottleneck — noticeable but manageable |

### Scoring Formula

```
severity_score = threshold_exceedance × frequency × impact_weight
```

where:

- `threshold_exceedance` = how far the signature exceeds its threshold (normalized)
- `frequency` = fraction of trajectories encountering this bottleneck
- `impact_weight` = per-type weight based on typical user impact

### Impact Score

The `impact_score` property estimates the total cost contribution:

```
impact_score = severity_score × cost_contribution / total_cost
```

---

## Implementation Reference

| Class | Module | Key Methods |
|-------|--------|-------------|
| `SignatureComputer` | `bottleneck/signatures.py` | `compute()`, `classify_signature()` |
| `BottleneckClassifier` | `bottleneck/classifier.py` | `classify()`, `classify_to_report()` |
| `BottleneckSignature` | `bottleneck/models.py` | Frozen dataclass with signature fields |
| `BottleneckResult` | `bottleneck/models.py` | `severity_score`, `impact_score`, `to_dict()` |
| `BottleneckReport` | `bottleneck/models.py` | `generate_summary()`, `by_type()`, `type_distribution()` |
| Perceptual detector | `bottleneck/perceptual.py` | Display entropy analysis |
| Choice detector | `bottleneck/choice.py` | Action space analysis |
| Motor detector | `bottleneck/motor.py` | Fitts' difficulty analysis |
| Memory detector | `bottleneck/memory.py` | Working memory load analysis |
| Interference detector | `bottleneck/interference.py` | MRT-based interference analysis |
| Repair mapping | `bottleneck/repair_map.py` | Bottleneck → repair strategy |

---

## Examples

### Example: Detecting Choice Paralysis

```python
from usability_oracle.bottleneck import BottleneckClassifier

classifier = BottleneckClassifier()
report = classifier.classify_to_report(mdp, policy, stats, breakdown)

choice_bottlenecks = report.by_type(BottleneckType.CHOICE)
for bn in choice_bottlenecks:
    print(f"Choice paralysis at state {bn.state_id}")
    print(f"  Alternatives: {bn.metadata.get('n_alternatives', '?')}")
    print(f"  Policy entropy: {bn.metadata.get('policy_entropy', '?'):.2f} bits")
    print(f"  Severity: {bn.severity_score:.2f}")
    print(f"  Suggested repair: progressive disclosure")
```

### Example: Full Report

```python
report = classifier.classify_to_report(mdp, policy, stats, breakdown)
print(report.generate_summary())

# Output:
# Bottleneck Report: 3 bottlenecks detected
#   [ERROR] Choice paralysis at state s_42 (severity: 0.85)
#   [WARNING] Perceptual overload at state s_17 (severity: 0.62)
#   [INFO] Motor difficulty at state s_31 (severity: 0.38)
#
# Type distribution: Choice=1, Perceptual=1, Motor=1

dist = report.type_distribution()
# {'CHOICE': 1, 'PERCEPTUAL': 1, 'MOTOR': 1, 'MEMORY': 0, 'INTERFERENCE': 0}
```
