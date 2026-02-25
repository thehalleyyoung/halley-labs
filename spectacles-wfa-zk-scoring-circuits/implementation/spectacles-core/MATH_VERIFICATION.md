# Mathematical Rigor Verification — Spectacles

Verification pass performed on the core mathematical claims in the Spectacles
WFA-ZK scoring circuit implementation.

---

## 1. Circuit Soundness (circuit/compiler.rs)

### Algebraic compilation (`compile_algebraic`)

**Transition encoding as degree-2 constraints — ✅ Correct**

The transition constraint for each target state `s` is:

```
state'[s] = Σ_a selector_a · (Σ_q M_a[s][q] · state[q])
```

Each `selector_a · state[q]` product is degree 2 (product of two witness
columns). The outer sum over symbols and inner sum over source states are
linear combinations of these degree-2 terms, so the overall constraint
polynomial has degree exactly 2. This correctly encodes the WFA
matrix-vector transition `v' = M(σ) · v` selected by the input symbol σ.

**Trace width is 2|Q| + O(1) — ✅ Correct**

From `compute_trace_layout` (line 1899):

| Column group       | Count          |
|---------------------|----------------|
| state variables     | \|Q\|          |
| input symbol        | 1              |
| symbol selectors    | \|Σ\|          |
| matmul auxiliaries   | \|Q\|          |
| output              | 1              |
| step counter        | 1              |
| **Total**           | **2\|Q\| + \|Σ\| + 3** |

This matches the claim of 2|Q| + O(1) when |Σ| is treated as a constant
of the metric (alphabet size is fixed per metric specification).

### Gadget-assisted compilation (`compile_gadget_assisted`)

**Bit decomposition for tropical min — ✅ Correct**

For non-algebraic semirings (tropical), the compiler introduces:
- Candidate variables: `cand = weight + state[from]` (tropical ⊗ = field +)
- Comparison gadgets with `num_bits` auxiliary bit columns per comparison
- Bit decomposition enforces `diff = Σ_k 2^k · bit_k` with boolean constraints
  on each bit, proving the minimum selection is correct

The tropical multiplication (⊕ = min, ⊗ = +) is correctly mapped: the
"candidate" value uses field addition (encoding tropical ⊗), and the
minimum selection uses comparison gadgets (encoding tropical ⊕).

---

## 2. Kleene Semiring Axioms (wfa/semiring.rs)

### Tropical semiring (ℝ ∪ {+∞}, min, +) — ✅ Correct

| Axiom                | Implementation                     | Status |
|----------------------|-------------------------------------|--------|
| zero (⊕ identity)   | `+∞` (`f64::INFINITY`)             | ✅     |
| one  (⊗ identity)   | `0.0`                               | ✅     |
| ⊕ (add)             | `min(a, b)`                         | ✅     |
| ⊗ (mul)             | `a + b` (real addition)             | ✅     |
| zero annihilates ⊗  | `∞ + x = ∞`                        | ✅     |
| ⊗ distributes over ⊕| `a + min(b,c) = min(a+b, a+c)`     | ✅     |

### Kleene star — ✅ Correct

```rust
fn star(&self) -> Self {
    if self.value >= 0.0 { Self::one() }    // a* = 0
    else { Self::new(f64::NEG_INFINITY) }   // divergent
}
```

For a ≥ 0: `a* = 1 ⊕ a ⊗ a* = min(0, a + 0) = min(0, a) = 0` ✓

The unfolding `a* = 1 ⊕ a ⊗ a*` is satisfied since
`min(0, a + a*) = min(0, a + 0) = min(0, a) = 0 = a*` when a ≥ 0.

### Boolean semiring ({false, true}, ∨, ∧) — ✅ Correct

`a* = true` for all a. Unfolding: `a* = true ∨ (a ∧ true) = true` ✓

### Counting semiring (ℕ, +, ·) — ✅ Correct

Standard natural number addition and multiplication with saturation
arithmetic to prevent overflow.

---

## 3. WFA Compilation Correctness (evalspec/compiler.rs)

### Metric-to-WFA semantics preservation — ✅ Correct

| Metric      | WFA construction                                          |
|-------------|-----------------------------------------------------------|
| ExactMatch  | 2-state Boolean WFA (match/fail sink)                     |
| TokenF1     | Pair of 1-state counting WFAs (precision/recall)          |
| RegexMatch  | Thompson NFA → subset construction → Boolean DFA          |
| BLEU        | n-gram counters per order + length counter + post-process  |
| ROUGE-N     | n-gram counter pair + F-measure post-process              |
| ROUGE-L     | Tropical WFA for LCS via Viterbi (weight −1 on match)     |

Each construction correctly uses the appropriate semiring for its semantics.

### N-gram WFA construction (`build_ngram_counter`) — ✅ Correct

Two strategies based on alphabet/n-gram size:

1. **Flat** (large alphabet or n > 4): Chain of n states; first n−1 are
   "warming" states, state n−1 self-loops. Every transition at full depth
   completes an n-gram.

2. **Full trie** (small alphabet): Enumerates all Σ^k contexts for
   k = 0..n−1. At depth n−1, transitions shift the context window:
   `new_context = old_context[1..] ++ [symbol]`, correctly computed as
   `(local % Σ^{n-2}) * Σ + a`.

Both constructions are deterministic and correctly structure the state
space for n-gram windowing.

---

## 4. Contamination Detection Bounds (psi/protocol.rs)

### Contamination score computation — ✅ Correct

Two computation paths exist:

- `compute_contamination(result, total_benchmark_ngrams)`:
  `|A ∩ B| / total_benchmark_ngrams` — used for external scoring with
  a known benchmark size.

- `compute_contamination_from_counts(intersection, set_a_size)`:
  `|A ∩ B| / |A|` — used internally for containment-based scoring.

Both correctly handle the zero-denominator case.

### Threshold comparison — ✅ Correct (with minor comment inconsistency)

| Function             | Comparison         | Semantics              |
|----------------------|--------------------|------------------------|
| `check_threshold`    | `score ≤ threshold`| passes if at-or-below  |
| `is_contaminated`    | `score > threshold`| contaminated if above  |
| `threshold_satisfied`| `bound ≤ threshold`| attestation passes     |

All three are mutually consistent. The `PSIMode::Threshold` doc comment
says "overlap < threshold" (strict), but the implementation uses `≤`
(non-strict). The implementation convention (≤) is standard and the
comment is slightly imprecise but the code logic is correct and consistent.

---

## 5. Goldilocks Field Arithmetic (circuit/goldilocks.rs)

### Modulus — ✅ Correct

```
p = 2^64 − 2^32 + 1 = 0xFFFFFFFF00000001
```

Verified: `0xFFFFFFFF00000001 = 18446744069414584321 = 2^64 − 2^32 + 1`.

### Modular reduction — ✅ Correct

**`reduce(v)`**: Single conditional subtraction. Correct since inputs are
< 2p.

**`reduce_u128(x)`**: Uses `2^64 ≡ 2^32 − 1 (mod p)` to reduce
`x = x_hi·2^64 + x_lo` to `x_lo + x_hi·(2^32 − 1)`. Handles up to two
levels of carry propagation, each applying the same identity. Final
conditional subtraction ensures result ∈ [0, p). Verified correct for the
full range of u128 inputs arising from multiplication of two elements
< p.

**`add_elem`**: Overflow case adds `2^32 − 1 = 0xFFFFFFFF` (since
`2^64 ≡ 2^32 − 1 mod p`). Verified no double overflow is possible since
max sum < 2p.

### NTT parameters — ✅ Correct

| Parameter                 | Value              | Verified |
|---------------------------|--------------------|----------|
| Two-adicity               | 32                 | ✅ `p−1 = 2^32 · (2^32 − 1)` |
| 2^32-th root of unity     | `0x185629DCDA58878C` | ✅ `7^((p−1)/2^32) mod p` |
| Primitive                 | Yes                | ✅ `ω^{2^31} ≠ 1 mod p` |
| Montgomery R = 2^64 mod p | `0xFFFFFFFF`       | ✅ `2^64 − p = 2^32 − 1` |
| Montgomery R² mod p       | `0xFFFFFFFE00000001` | ✅ Independently computed |

### 🐛 BUG FIXED: Montgomery inverse constant

**`MONTGOMERY_INV`** was `0xFFFFFFFF` — **incorrect**.

The correct value is `−p^{−1} mod 2^{64}`:
- `p^{−1} mod 2^{64} = 1 + 2^{32} = 0x100000001`
  (since `(1 − 2^{32})(1 + 2^{32}) = 1 − 2^{64} ≡ 1 mod 2^{64}`)
- `−p^{−1} mod 2^{64} = 2^{64} − (1 + 2^{32}) = 0xFFFFFFFEFFFFFFFF`

**Fix applied**: Changed `MONTGOMERY_INV` from `0xFFFFFFFF` to
`0xFFFFFFFEFFFFFFFF`.

**Impact**: The Montgomery multiplication path (`montgomery_mul`,
`montgomery_reduce`, `to_montgomery`, `from_montgomery`) would produce
incorrect results. The primary arithmetic path (`mul_elem`, `add_elem`)
uses direct `reduce_u128` and is unaffected.

### 🐛 BUG FIXED: Lagrange interpolation constant-term computation

In `lagrange_interpolation`, the update to `basis[0]` when multiplying
the basis polynomial by `(x − x_j)` had a dead-code line that corrupted
the intermediate value:

```rust
// BEFORE (buggy):
basis[0] = basis[0].neg_elem().mul_elem(xs[j]).neg_elem();  // = basis[0] * xs[j]  (WRONG)
// Actually: ...
basis[0] = Self::ZERO.sub_elem(basis[0].mul_elem(xs[j]));   // uses corrupted basis[0]
```

Line 497 computes `−(−b₀ · xⱼ) = b₀ · xⱼ` instead of the needed
`−b₀ · xⱼ`. Line 499 then applies the formula to the already-corrupted
value, yielding `−b₀ · xⱼ²` instead of `−b₀ · xⱼ`.

**Verification**: For `(x−3)(x−5)`, the buggy code produces constant term
225 instead of the correct 15.

**Fix applied**: Removed line 497, keeping only the correct line 499:
```rust
basis[0] = Self::ZERO.sub_elem(basis[0].mul_elem(xs[j]));
```

**Impact**: Lagrange interpolation (used in FRI protocol, polynomial
commitment verification, and constraint interpolation) would produce
incorrect polynomials, causing proof verification failures.

---

## Summary

| Area                        | Status                  |
|-----------------------------|-------------------------|
| Algebraic compilation       | ✅ Verified correct     |
| Gadget-assisted compilation | ✅ Verified correct     |
| Trace width claim           | ✅ 2\|Q\| + \|Σ\| + 3  |
| Tropical semiring axioms    | ✅ Verified correct     |
| Boolean semiring axioms     | ✅ Verified correct     |
| Counting semiring axioms    | ✅ Verified correct     |
| Kleene star unfolding       | ✅ Verified correct     |
| EvalSpec→WFA compilation    | ✅ Verified correct     |
| N-gram WFA construction     | ✅ Verified correct     |
| Contamination score         | ✅ Verified correct     |
| Threshold comparison        | ✅ Consistent (≤)       |
| Goldilocks prime            | ✅ Verified correct     |
| Modular reduction           | ✅ Verified correct     |
| NTT root of unity           | ✅ Verified correct     |
| Montgomery INV constant     | 🐛 **Fixed** (was 0xFFFFFFFF → 0xFFFFFFFEFFFFFFFF) |
| Lagrange interpolation      | 🐛 **Fixed** (dead code corrupting basis[0])         |
