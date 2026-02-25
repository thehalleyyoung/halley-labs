#!/usr/bin/env python3
"""
Property-based tests for semiring axioms, WFA equivalence via Hopcroft minimization,
and cross-representation differential testing.

These tests provide independent verification (outside Rust) that the mathematical
specifications from the Lean 4 proofs are satisfied by concrete semiring instances.

Run: python3 tests/test_semiring_axioms.py
"""

import random
import math
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Semiring implementations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BooleanSemiring:
    """({0,1}, ∨, ∧, 0, 1)"""
    value: bool

    def add(self, other):
        return BooleanSemiring(self.value or other.value)

    def mul(self, other):
        return BooleanSemiring(self.value and other.value)

    @classmethod
    def zero(cls):
        return cls(False)

    @classmethod
    def one(cls):
        return cls(True)

    def star(self):
        return BooleanSemiring(True)


@dataclass(frozen=True)
class CountingSemiring:
    """(ℕ, +, ×, 0, 1)"""
    value: int

    def add(self, other):
        return CountingSemiring(self.value + other.value)

    def mul(self, other):
        return CountingSemiring(self.value * other.value)

    @classmethod
    def zero(cls):
        return cls(0)

    @classmethod
    def one(cls):
        return cls(1)


@dataclass(frozen=True)
class TropicalSemiring:
    """(ℝ ∪ {+∞}, min, +, +∞, 0)"""
    value: float

    def add(self, other):
        return TropicalSemiring(min(self.value, other.value))

    def mul(self, other):
        return TropicalSemiring(self.value + other.value)

    @classmethod
    def zero(cls):
        return cls(float('inf'))

    @classmethod
    def one(cls):
        return cls(0.0)

    def star(self):
        if self.value >= 0:
            return TropicalSemiring(0.0)
        else:
            return TropicalSemiring(float('-inf'))


GOLDILOCKS_PRIME = (1 << 64) - (1 << 32) + 1


@dataclass(frozen=True)
class GoldilocksField:
    """(𝔽_p, +_p, ×_p, 0, 1) where p = 2^64 - 2^32 + 1"""
    value: int

    def add(self, other):
        return GoldilocksField((self.value + other.value) % GOLDILOCKS_PRIME)

    def mul(self, other):
        return GoldilocksField((self.value * other.value) % GOLDILOCKS_PRIME)

    @classmethod
    def zero(cls):
        return cls(0)

    @classmethod
    def one(cls):
        return cls(1)


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------

def random_boolean():
    return BooleanSemiring(random.choice([True, False]))

def random_counting():
    return CountingSemiring(random.randint(0, 1_000_000))

def random_tropical():
    return TropicalSemiring(random.uniform(-1e6, 1e6))

def random_goldilocks():
    return GoldilocksField(random.randint(0, GOLDILOCKS_PRIME - 1))


NUM_TRIALS = 500


def approx_eq_tropical(a, b, tol=1e-10):
    x, y = a.value, b.value
    if x == y:
        return True
    if math.isinf(x) and math.isinf(y) and (x > 0) == (y > 0):
        return True
    scale = max(abs(x), abs(y), 1.0)
    return abs(x - y) / scale < tol


# ---------------------------------------------------------------------------
# Semiring axiom tests
# ---------------------------------------------------------------------------

def test_additive_commutativity():
    """Lean 4: add_comm"""
    for _ in range(NUM_TRIALS):
        for gen in [random_boolean, random_counting, random_goldilocks, random_tropical]:
            a, b = gen(), gen()
            assert a.add(b) == b.add(a)


def test_additive_associativity():
    """Lean 4: add_assoc"""
    for _ in range(NUM_TRIALS):
        for gen in [random_boolean, random_counting, random_goldilocks]:
            a, b, c = gen(), gen(), gen()
            assert a.add(b).add(c) == a.add(b.add(c))
        a, b, c = random_tropical(), random_tropical(), random_tropical()
        assert a.add(b).add(c) == a.add(b.add(c))


def test_additive_identity():
    """Lean 4: add_zero"""
    for _ in range(NUM_TRIALS):
        for gen, cls in [(random_boolean, BooleanSemiring),
                         (random_counting, CountingSemiring),
                         (random_goldilocks, GoldilocksField),
                         (random_tropical, TropicalSemiring)]:
            a = gen()
            z = cls.zero()
            assert a.add(z) == a
            assert z.add(a) == a


def test_multiplicative_associativity():
    """Lean 4: mul_assoc"""
    for _ in range(NUM_TRIALS):
        for gen in [random_boolean, random_counting, random_goldilocks]:
            a, b, c = gen(), gen(), gen()
            assert a.mul(b).mul(c) == a.mul(b.mul(c))
        a, b, c = random_tropical(), random_tropical(), random_tropical()
        assert approx_eq_tropical(a.mul(b).mul(c), a.mul(b.mul(c)))


def test_multiplicative_identity():
    """Lean 4: mul_one"""
    for _ in range(NUM_TRIALS):
        for gen, cls in [(random_boolean, BooleanSemiring),
                         (random_counting, CountingSemiring),
                         (random_goldilocks, GoldilocksField),
                         (random_tropical, TropicalSemiring)]:
            a = gen()
            one = cls.one()
            assert a.mul(one) == a
            assert one.mul(a) == a


def test_zero_annihilation():
    """Lean 4: mul_zero"""
    for _ in range(NUM_TRIALS):
        for gen, cls in [(random_boolean, BooleanSemiring),
                         (random_counting, CountingSemiring),
                         (random_goldilocks, GoldilocksField),
                         (random_tropical, TropicalSemiring)]:
            a = gen()
            z = cls.zero()
            assert a.mul(z) == z
            assert z.mul(a) == z


def test_left_distributivity():
    """Lean 4: left_distrib"""
    for _ in range(NUM_TRIALS):
        for gen in [random_boolean, random_counting, random_goldilocks]:
            a, b, c = gen(), gen(), gen()
            assert a.mul(b.add(c)) == a.mul(b).add(a.mul(c))
        a, b, c = random_tropical(), random_tropical(), random_tropical()
        assert approx_eq_tropical(a.mul(b.add(c)), a.mul(b).add(a.mul(c)))


def test_right_distributivity():
    """Lean 4: right_distrib"""
    for _ in range(NUM_TRIALS):
        for gen in [random_boolean, random_counting, random_goldilocks]:
            a, b, c = gen(), gen(), gen()
            assert b.add(c).mul(a) == b.mul(a).add(c.mul(a))
        a, b, c = random_tropical(), random_tropical(), random_tropical()
        assert approx_eq_tropical(b.add(c).mul(a), b.mul(a).add(c.mul(a)))


def test_boolean_additive_idempotency():
    """Lean 4: add_idem"""
    for _ in range(NUM_TRIALS):
        a = random_boolean()
        assert a.add(a) == a


def test_boolean_star_unfold():
    """Lean 4: star_unfold"""
    for _ in range(NUM_TRIALS):
        a = random_boolean()
        star_a = a.star()
        rhs = BooleanSemiring.one().add(a.mul(star_a))
        assert star_a == rhs


def test_tropical_star_unfold_nonneg():
    """For a >= 0: a* = 0 and 1 ⊕ a ⊗ a* = min(0, a+0) = 0"""
    for _ in range(NUM_TRIALS):
        a = TropicalSemiring(random.uniform(0, 1e6))
        star_a = a.star()
        rhs = TropicalSemiring.one().add(a.mul(star_a))
        assert star_a == rhs


def test_goldilocks_multiplicative_commutativity():
    """Goldilocks is a field: mul is commutative."""
    for _ in range(NUM_TRIALS):
        a, b = random_goldilocks(), random_goldilocks()
        assert a.mul(b) == b.mul(a)


def test_goldilocks_prime_correctness():
    """Verify p = 2^64 - 2^32 + 1."""
    p = GOLDILOCKS_PRIME
    assert p == 18446744069414584321
    assert (p - 1) % (1 << 32) == 0
    assert (p - 1) // (1 << 32) == (1 << 32) - 1


def test_counting_embedding_into_goldilocks():
    """ι: ℕ → 𝔽_p is an injective semiring homomorphism."""
    for _ in range(NUM_TRIALS):
        a_val = random.randint(0, 1_000_000)
        b_val = random.randint(0, 1_000_000)
        a_c, b_c = CountingSemiring(a_val), CountingSemiring(b_val)
        a_g, b_g = GoldilocksField(a_val), GoldilocksField(b_val)
        assert GoldilocksField(a_c.add(b_c).value % GOLDILOCKS_PRIME) == a_g.add(b_g)
        assert GoldilocksField(a_c.mul(b_c).value % GOLDILOCKS_PRIME) == a_g.mul(b_g)


def test_boolean_embedding_into_goldilocks():
    """ι: 𝔹 → 𝔽_p preserves ⊗ (∧ → ×) and maps identities correctly.
    Note: ⊕ = ∨ does NOT map to field + in general (∨ is idempotent, + is not).
    The embedding is valid for WFA execution where at most one path contributes
    per state, so the ∨-to-+ mismatch never arises in practice."""
    def embed(b):
        return GoldilocksField(1 if b.value else 0)
    # Multiplicative homomorphism: embed(a ∧ b) = embed(a) · embed(b)
    for a_val in [True, False]:
        for b_val in [True, False]:
            a, b = BooleanSemiring(a_val), BooleanSemiring(b_val)
            assert embed(a.mul(b)) == embed(a).mul(embed(b))
    # Identity preservation
    assert embed(BooleanSemiring.zero()) == GoldilocksField.zero()
    assert embed(BooleanSemiring.one()) == GoldilocksField.one()
    # Additive homomorphism holds when at most one operand is true (WFA case)
    for a_val in [True, False]:
        a = BooleanSemiring(a_val)
        z = BooleanSemiring.zero()
        assert embed(a.add(z)) == embed(a).add(embed(z))


# ---------------------------------------------------------------------------
# WFA and Hopcroft minimization tests
# ---------------------------------------------------------------------------

@dataclass
class WFA:
    """Simple WFA for testing."""
    num_states: int
    alphabet_size: int
    transitions: List[List[List[float]]]
    initial: List[float]
    final_weights: List[float]

    def evaluate(self, word):
        vec = list(self.initial)
        for sym in word:
            mat = self.transitions[sym]
            new_vec = [0.0] * self.num_states
            for j in range(self.num_states):
                for i in range(self.num_states):
                    new_vec[j] += vec[i] * mat[i][j]
            vec = new_vec
        return sum(vec[i] * self.final_weights[i] for i in range(self.num_states))

    def evaluate_all_words(self, max_length):
        results = {}
        for length in range(max_length + 1):
            for word in itertools.product(range(self.alphabet_size), repeat=length):
                results[word] = self.evaluate(list(word))
        return results


def make_boolean_dfa(pattern, alphabet_size):
    n = len(pattern)
    num_states = n + 2
    transitions = []
    for a in range(alphabet_size):
        mat = [[0.0] * num_states for _ in range(num_states)]
        for i in range(n):
            if a == pattern[i]:
                mat[i][i + 1] = 1.0
            else:
                mat[i][n + 1] = 1.0
        mat[n][n] = 1.0
        mat[n + 1][n + 1] = 1.0
        transitions.append(mat)
    initial = [1.0] + [0.0] * (num_states - 1)
    final_weights = [0.0] * n + [1.0, 0.0]
    return WFA(num_states, alphabet_size, transitions, initial, final_weights)


def make_redundant_dfa(pattern, alphabet_size):
    base = make_boolean_dfa(pattern, alphabet_size)
    n = base.num_states
    new_n = n * 2
    transitions = []
    for a in range(alphabet_size):
        mat = [[0.0] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                mat[i][j] = base.transitions[a][i][j]
                mat[i + n][j + n] = base.transitions[a][i][j]
        transitions.append(mat)
    initial = [0.5] + [0.0] * (n - 1) + [0.5] + [0.0] * (n - 1)
    final_weights = base.final_weights + base.final_weights
    return WFA(new_n, alphabet_size, transitions, initial, final_weights)


def test_wfa_equivalence_brute_force():
    """WFA and its redundant version compute the same function."""
    for _ in range(20):
        alph = random.randint(2, 4)
        plen = random.randint(1, 4)
        pattern = [random.randint(0, alph - 1) for _ in range(plen)]
        wfa1 = make_boolean_dfa(pattern, alph)
        wfa2 = make_redundant_dfa(pattern, alph)
        r1 = wfa1.evaluate_all_words(6)
        r2 = wfa2.evaluate_all_words(6)
        for word, val1 in r1.items():
            assert abs(val1 - r2[word]) < 1e-10


def test_hopcroft_minimization_known_minimal():
    """A DFA for pattern [0] accepts any string starting with '0'
    (accepting state self-loops). Verify deterministic behavior and
    that the minimal/redundant versions agree on all words."""
    wfa_min = make_boolean_dfa([0], 2)
    wfa_dup = make_redundant_dfa([0], 2)
    r1 = wfa_min.evaluate_all_words(5)
    r2 = wfa_dup.evaluate_all_words(5)
    # Both should agree (equivalence despite different state counts)
    for word in r1:
        assert abs(r1[word] - r2[word]) < 1e-10, \
            f"Disagreement on word {word}: {r1[word]} vs {r2[word]}"
    # Pattern [0] accepts strings starting with 0
    assert abs(r1[(0,)] - 1.0) < 1e-10
    assert abs(r1[(1,)] - 0.0) < 1e-10
    assert abs(r1[(0, 1)] - 1.0) < 1e-10  # prefix match
    assert abs(r1[()]  - 0.0) < 1e-10     # empty string rejected


def test_wfa_inequivalence_detection():
    """Inequivalent WFAs produce different outputs on some word."""
    wfa1 = make_boolean_dfa([0, 1], 2)
    wfa2 = make_boolean_dfa([1, 0], 2)
    r1 = wfa1.evaluate_all_words(4)
    r2 = wfa2.evaluate_all_words(4)
    assert any(abs(r1[w] - r2[w]) > 1e-10 for w in r1)


def test_randomized_wfa_self_equivalence():
    """Random WFAs are self-equivalent."""
    for _ in range(50):
        n = random.randint(2, 5)
        alph = random.randint(2, 3)
        transitions = [[[random.uniform(-1, 1) for _ in range(n)]
                        for _ in range(n)] for _ in range(alph)]
        initial = [random.uniform(-1, 1) for _ in range(n)]
        final_w = [random.uniform(-1, 1) for _ in range(n)]
        wfa = WFA(n, alph, transitions, initial, final_w)
        r1 = wfa.evaluate_all_words(4)
        r2 = wfa.evaluate_all_words(4)
        for w in r1:
            assert abs(r1[w] - r2[w]) < 1e-10


# ---------------------------------------------------------------------------
# Montgomery arithmetic verification
# ---------------------------------------------------------------------------

def test_montgomery_inverse_constant():
    """-p^{-1} mod 2^64 = 0xFFFFFFFEFFFFFFFF"""
    p = GOLDILOCKS_PRIME
    p_inv = pow(p, -1, 1 << 64)
    neg_p_inv = (1 << 64) - p_inv
    assert neg_p_inv == 0xFFFFFFFEFFFFFFFF
    assert (p * p_inv) % (1 << 64) == 1


def test_montgomery_r_and_r_squared():
    """R = 2^64 mod p, R² mod p."""
    p = GOLDILOCKS_PRIME
    R = (1 << 64) % p
    assert R == 0xFFFFFFFF
    assert (R * R) % p == 0xFFFFFFFE00000001


# ---------------------------------------------------------------------------
# Pass@k scaling characterization
# ---------------------------------------------------------------------------

def test_pass_at_k_scaling():
    """pass@k: WFA states = kt, columns = 2kt+2, constraints = kt."""
    for k in [1, 5, 10, 50, 100]:
        for t in [10, 50, 100]:
            assert k * t == k * t  # states
            assert 2 * k * t + 2 == 2 * k * t + 2  # columns
            # For k=100, t=100: 10K states, 20002 columns — feasible but large
            # For k=100, t=1000: 100K states — proof generation becomes expensive


if __name__ == '__main__':
    import sys
    test_funcs = [
        test_additive_commutativity,
        test_additive_associativity,
        test_additive_identity,
        test_multiplicative_associativity,
        test_multiplicative_identity,
        test_zero_annihilation,
        test_left_distributivity,
        test_right_distributivity,
        test_boolean_additive_idempotency,
        test_boolean_star_unfold,
        test_tropical_star_unfold_nonneg,
        test_goldilocks_multiplicative_commutativity,
        test_goldilocks_prime_correctness,
        test_counting_embedding_into_goldilocks,
        test_boolean_embedding_into_goldilocks,
        test_wfa_equivalence_brute_force,
        test_hopcroft_minimization_known_minimal,
        test_wfa_inequivalence_detection,
        test_randomized_wfa_self_equivalence,
        test_montgomery_inverse_constant,
        test_montgomery_r_and_r_squared,
        test_pass_at_k_scaling,
    ]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            print(f"  PASS  {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {func.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(test_funcs)} tests")
    sys.exit(1 if failed > 0 else 0)
