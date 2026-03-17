# Mathematical Foundations

This document presents the mathematical foundations of the Certified Leakage
Contracts framework: abstract interpretation with Galois connections, the
reduced product domain, the composition theorem, and the soundness argument.

## 1. Abstract Interpretation

### 1.1 Concrete and Abstract Domains

Let **(C, ⊆)** be the concrete domain (power set of concrete program states)
and **(A, ⊑)** be an abstract domain.  An **abstraction** is defined by a
Galois connection:

```
(C, ⊆) ⇄[α, γ] (A, ⊑)
```

where:
- **α : C → A** (abstraction function) maps concrete sets to abstract elements
- **γ : A → C** (concretization function) maps abstract elements to concrete sets
- **Soundness:** ∀ c ∈ C, S ∈ A : c ⊆ γ(S) ⟹ α(c) ⊑ S
- **Optimality:** ∀ c ∈ C : α(c) is the smallest abstract element covering c

### 1.2 Transfer Functions

For each instruction `i`, the concrete semantics is a transformer
`⟦i⟧ : C → C`.  The abstract transfer function `⟦i⟧♯ : A → A` must be
**sound**:

```
∀ S ∈ A : ⟦i⟧(γ(S)) ⊆ γ(⟦i⟧♯(S))
```

That is, the abstract transformer covers all concrete behaviors.

### 1.3 Fixpoint Computation

The analysis computes the least fixpoint of the abstract transfer function
over the CFG.  By Tarski's theorem, this fixpoint exists because A is a
complete lattice and the transfer functions are monotone.

**Widening** (∇) accelerates convergence by extrapolating:
```
S ∇ T ⊒ S ⊔ T
```
with the guarantee that any ascending chain stabilizes in finite steps.

**Narrowing** (△) recovers precision lost by widening:
```
S ⊒ S △ T ⊒ T    (when S ⊒ T)
```

## 2. The Three Abstract Domains

### 2.1 D_spec — Speculative Reachability

**Concrete domain:** Sets of (program point, speculation status) pairs.
A pair (p, σ) means program point p is reachable with speculation state σ
(either architectural or speculative with remaining window w).

**Abstract domain:** A\_spec = P(SpecTag) per program point, where each
SpecTag identifies a speculation scenario.

**Galois connection:**
```
α_spec(C) = { tag(σ) | (p, σ) ∈ C }
γ_spec(S) = { (p, σ) | tag(σ) ∈ S }
```

**Transfer function:**
At a branch instruction with condition c:
- Architectural successor: propagate with σ = arch
- Speculative successor (mispredicted): propagate with σ = spec(w)
  where w is the remaining speculation window
- Kill speculative paths when w = 0

### 2.2 D_cache — Tainted Abstract Cache

**Concrete domain:** Sets of (cache configuration, taint map) pairs.
A cache configuration assigns each set/way to a cache line with an LRU age.
The taint map records whether each line's presence depends on secret data.

**Abstract domain:** A\_cache = ∏ᵢ AbstractCacheSet(i), where each
abstract cache set is:

```
AbstractCacheSet = Way → (AbstractAge × TaintAnnotation)
```

- **AbstractAge** ∈ {definite(n) | 0 ≤ n < W} ∪ {range(lo, hi)} ∪ {⊤_age}
- **TaintAnnotation** ∈ {untainted, tainted(source), ⊤_taint}

**Galois connection:**
```
α_cache(C) = ⊔ { abstract(config, taint) | (config, taint) ∈ C }
γ_cache(S) = { (config, taint) | config ∈ concretize_ages(S),
                                  taint ∈ concretize_taints(S) }
```

**Transfer function:**
For a memory access to address a with taint t:
1. Compute cache set index: set = (a / line_size) mod num_sets
2. Compute cache tag: tag = a / (line_size × num_sets)
3. **Hit:** If tag is present in abstract set, update ages (promote to MRU)
4. **Miss:** Evict the line with maximum abstract age, insert new line at
   age 0 with taint annotation t
5. **Unknown:** If hit/miss cannot be determined, join both outcomes

### 2.3 D_quant — Quantitative Channel Capacity

**Concrete domain:** Partitions of secret values into equivalence classes
based on observable cache behavior.

**Abstract domain:** A\_quant = ℚ≥0 (rational number representing log₂ of
the number of distinguishable cache configurations).

**Galois connection:**
```
α_quant(P) = log₂(|P|)        where P is the partition
γ_quant(b) = { P : |P| ≤ 2^b }
```

**Transfer function:**
After a tainted memory access:
1. Determine the number of distinct cache set indices that could be accessed
   (depends on the abstract value of the tainted address)
2. Each distinct index creates a distinguishable observation
3. New leakage = log₂(number of new distinguishable observations)
4. Total leakage = previous leakage + new leakage (additive in log space
   corresponds to multiplicative growth in observation count)

## 3. The Reduced Product

### 3.1 Definition

The **reduced product** of D_spec, D_cache, and D_quant is:

```
D = D_spec ⊗ D_cache ⊗ D_quant
  = { (s, c, q) ∈ A_spec × A_cache × A_quant | ρ(s, c, q) = (s, c, q) }
```

where ρ is the reduction operator.  The product is "reduced" because ρ
removes elements that are inconsistent across domains.

### 3.2 Reduction Operator

**Definition.** ρ : A\_spec × A\_cache × A\_quant → A\_spec × A\_cache × A\_quant
is the greatest fixpoint of the following reduction rules:

**Rule 1 (Spec → Cache):**
```
If spec(p) = ∅ then cache(p) := ⊥_cache
```
Unreachable points have empty cache state.

**Rule 2 (Cache → Quant):**
```
If ∀ set : tainted_lines(cache, set) = ∅ then quant := 0
```
No tainted lines means zero leakage.

**Rule 3 (Spec → Quant):**
```
quant := quant + Σ_{σ ∈ spec_only(p)} additional_observations(σ, cache)
```
Speculatively-only reachable points may introduce additional observations.

**Rule 4 (Quant → Spec):**
```
If observations(σ₁, cache) = observations(σ₂, cache)
then merge σ₁ and σ₂ in spec
```
Observation-equivalent speculative paths can be merged.

**Rule 5 (Cache → Spec):**
```
If ∀ access on path σ : taint(access) = untainted
then deprioritize σ in spec
```

**Theorem 3.1 (Reduction Correctness).** The reduction operator ρ is:
1. **Sound:** γ(ρ(s, c, q)) ⊇ γ(s) ∩ γ(c) ∩ γ(q)
2. **Reductive:** ρ(s, c, q) ⊑ (s, c, q)
3. **Idempotent:** ρ(ρ(s, c, q)) = ρ(s, c, q)

*Proof sketch.* Each rule removes only abstract elements that correspond
to no concrete element (inconsistency across domains).  Since we only
remove impossible combinations, soundness is preserved.  Reductiveness
follows from the fact that we only tighten.  Idempotence follows from
the fixpoint formulation.  □

## 4. Leakage Contracts

### 4.1 Definition

A **leakage contract** for function f is a tuple:

```
Contract(f) = (τ_f, B_f)
```

where:
- **τ\_f : A\_cache → A\_cache** is the abstract cache transformer
- **B\_f : A\_cache → ℚ≥0** is the leakage bound function

**Interpretation:** If the cache state before calling f is s, then:
- After f returns, the cache state is (at most) τ\_f(s)
- The information leaked by f is at most B\_f(s) bits of min-entropy

### 4.2 Contract Extraction

Given the fixpoint analysis result for function f:

```
τ_f(s) = cache component of fixpoint at the exit block, starting from s
B_f(s) = quant component of fixpoint at the exit block, starting from s
```

## 5. Composition Theorem

### 5.1 Statement

**Theorem 5.1 (Additive Composition).** Let f, g be functions with
contracts (τ\_f, B\_f) and (τ\_g, B\_g).  Suppose:

1. **Monotonicity:** τ\_f and τ\_g are monotone: s ⊑ s' ⟹ τ(s) ⊑ τ(s')
2. **Independence:** The cache observations of f and g are conditionally
   independent given the intermediate cache state τ\_f(s).

Then the sequential composition f ; g has contract:

```
τ_{f;g}(s) = τ_g(τ_f(s))
B_{f;g}(s) ≤ B_f(s) + B_g(τ_f(s))
```

### 5.2 Proof

**Proof.** We show that B\_{f;g}(s) ≤ B\_f(s) + B\_g(τ\_f(s)).

Let S be the secret, O\_f be the cache observation of f, and O\_g be the
cache observation of g.  The total observation is O = (O\_f, O\_g).

The min-entropy leakage of the composed function is:

```
L(f;g) = H_∞(S) − H_∞(S | O_f, O_g)
```

By the chain rule for min-entropy (with a correction for non-Shannon
entropy):

```
L(f;g) ≤ L(f) + L(g | intermediate state)
```

More precisely, using the independence assumption:

```
H_∞(S | O_f, O_g) ≥ H_∞(S | O_f) − log₂|O_g|
```

where |O\_g| is the number of distinguishable observations of g.  Since:

```
L(f) = H_∞(S) − H_∞(S | O_f) ≤ B_f(s)
log₂|O_g| ≤ B_g(τ_f(s))
```

we get:

```
L(f;g) = H_∞(S) − H_∞(S | O_f, O_g)
       ≤ H_∞(S) − (H_∞(S | O_f) − B_g(τ_f(s)))
       = L(f) + B_g(τ_f(s))
       ≤ B_f(s) + B_g(τ_f(s))
```

**Note:** The independence assumption is critical.  Without it, the
observations of g could be correlated with those of f, potentially
leaking more than the sum.  The `IndependenceChecker` in `leak-contract`
verifies this condition statically.  □

### 5.3 Parallel Composition

**Theorem 5.2 (Parallel Composition).** If f and g operate on disjoint
cache sets, then:

```
B_{f||g}(s) ≤ B_f(s) + B_g(s)
```

*Proof.* Disjoint cache sets imply independent observations.  The bound
follows by the same argument as sequential composition, with the
intermediate state being the input state s (since the functions do not
interfere).  □

### 5.4 Conditional Composition

**Theorem 5.3 (Conditional Composition).** For `if c then f else g`:

```
B_{if}(s) ≤ max(B_f(s), B_g(s)) + B_branch(s)
```

where B\_branch(s) is the leakage from the branch condition itself (at most
1 bit if the branch depends on secret data, 0 if it does not).

## 6. Min-Entropy Leakage

### 6.1 Definition

For a secret S with prior distribution π and an observation channel
O : S → Obs, the **min-entropy leakage** is:

```
L_∞ = H_∞(S) − H_∞(S | O)
```

where:
```
H_∞(S) = −log₂(max_s π(s))
H_∞(S | O) = −log₂(Σ_o max_s P(s, o))
```

### 6.2 Operational Interpretation

Min-entropy leakage measures the multiplicative increase in an attacker's
probability of guessing the secret in one try, after observing the cache
behavior:

```
2^{L_∞} = P_guess(S | O) / P_guess(S)
```

A leakage bound of B bits means the attacker's one-guess success probability
increases by at most a factor of 2^B.

### 6.3 Why Min-Entropy?

We use min-entropy rather than Shannon entropy because:

1. **Worst-case guarantee:** Min-entropy captures the attacker's best
   single guess, which is the most relevant security metric.
2. **Compositionality:** Min-entropy leakage composes additively under
   independence (as shown in the composition theorem).
3. **Conservative:** H\_∞(S) ≤ H(S) (min-entropy is always at most
   Shannon entropy), so bounds on min-entropy leakage are at least as
   strong as Shannon entropy bounds.

### 6.4 Counting Abstraction

In practice, computing min-entropy leakage exactly requires enumerating
all distinguishable observations.  Our counting domain abstracts this by:

1. **Taint-restricted counting:** Only count observations due to
   tainted (secret-dependent) cache accesses.
2. **Per-set decomposition:** Count distinguishable configurations per
   cache set, then combine: total ≤ ∏ᵢ count(set\_i).
3. **Logarithmic representation:** Store bounds as log₂(count) for
   efficient arithmetic.

**Theorem 6.1 (Counting Soundness).** If the counting domain reports
k distinguishable configurations, then the min-entropy leakage is at most
log₂(k) bits.

*Proof.* The number of distinguishable observations is at most k.  By
definition of min-entropy leakage:

```
L_∞ ≤ log₂(|Obs|) ≤ log₂(k)
```

where the first inequality is tight when the prior is uniform.  □

## 7. Soundness of the Framework

### 7.1 Main Theorem

**Theorem 7.1 (Framework Soundness).** For any function f and initial
concrete cache state c:

```
L_∞(f, c) ≤ B_f(α_cache(c))
```

where L\_∞(f, c) is the true min-entropy leakage of f starting from cache
state c, and B\_f is the leakage bound from the contract.

### 7.2 Proof Outline

The proof proceeds by structural induction:

1. **Per-instruction soundness:** Each abstract transfer function is sound
   with respect to the concrete semantics (Section 1.2).

2. **Fixpoint soundness:** The fixpoint over-approximates all concrete
   executions (by Tarski's theorem and soundness of widening).

3. **Reduction soundness:** The reduction operator only removes impossible
   abstract states (Theorem 3.1).

4. **Counting soundness:** The counting domain correctly bounds the number
   of distinguishable observations (Theorem 6.1).

5. **Composition soundness:** The additive composition rule is valid under
   the independence condition (Theorem 5.1).

6. **Certificate soundness:** The certificate checker independently verifies
   each step, providing a second line of defense.

Combining these, the claimed bound B\_f(s) is a valid upper bound on the
true min-entropy leakage for any concrete state in γ(s).  □

## 8. Speculative Execution Model

### 8.1 Spectre-PHT Model

We model **Spectre-PHT** (branch direction misprediction):

- At each conditional branch, the attacker can cause the processor to
  speculatively execute the wrong direction for up to W μops.
- During speculative execution, memory accesses affect the cache but
  are "rolled back" architecturally.
- The **cache effects persist** even after the speculation is resolved.

### 8.2 Speculative Leakage

The speculative leakage bound accounts for additional observations created
by transiently executed instructions:

```
B_spec(f, s) = B_arch(f, s) + B_transient(f, s, W)
```

where B\_arch is the architectural (non-speculative) bound and B\_transient
accounts for additional cache effects from speculative paths within
window W.

### 8.3 Window Parameter

The speculation window W bounds the number of micro-operations that can
execute speculatively before the branch resolves.  Typical values:

| CPU Family       | Approximate W |
|------------------|---------------|
| Intel Skylake    | ~30 μops      |
| Intel Alder Lake | ~40 μops      |
| AMD Zen 3       | ~25 μops      |
| Conservative     | 50 μops       |

The default W = 20 balances precision (smaller window = tighter bounds)
with coverage (larger window = more speculation scenarios).

## References

1. Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified
   lattice model for static analysis of programs. POPL.
2. Smith, G. (2009). On the foundations of quantitative information flow.
   FoSSaCS.
3. Alvim, M. S., et al. (2020). The Science of Quantitative Information
   Flow. Springer.
4. Doychev, G., et al. (2013). CacheAudit: A tool for the static analysis
   of cache side channels. USENIX Security.
5. Kocher, P., et al. (2019). Spectre attacks: Exploiting speculative
   execution. S&P.
6. Guarnieri, M., et al. (2020). Spectector: Principled detection of
   speculative information flows. S&P.
