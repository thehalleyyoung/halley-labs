"""
Comprehensive tests for state abstraction soundness.

Tests interval arithmetic, abstract MDP overapproximation, refinement,
safety transfer, independent verification, and certificate round-trips.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Ensure the implementation package is importable
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.verification.interval_arithmetic import (
    Interval,
    IntervalMatrix,
    IntervalVector,
    interval_matmul,
    interval_matmul_matrix,
    interval_polyval,
    _EPS,
    _TINY,
)
from causal_trading.verification.state_abstraction import (
    AbstractState,
    AbstractionFunction,
    ConcreteState,
    ConservativeOverapproximation,
    OverapproximationResult,
    SoundnessCertificate,
    compute_abstract_transitions,
    discretize_state_space,
    interval_safety_probability,
    refinement_step,
    verify_overapproximation,
)
from causal_trading.verification.model_checking import (
    MDP,
    MDPTransition,
    Specification,
    SpecKind,
    SymbolicModelChecker,
    build_mdp_from_matrix,
)
from causal_trading.verification.independent_verifier import (
    IndependentVerifier,
    VerificationReport,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(seed=12345)


@pytest.fixture
def small_mdp():
    """3-state, 2-action MDP with known transition structure."""
    T0 = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.15, 0.8],
    ])
    T1 = np.array([
        [0.3, 0.3, 0.4],
        [0.1, 0.3, 0.6],
        [0.05, 0.05, 0.9],
    ])
    transitions = {}
    for s in range(3):
        for a, T in enumerate([T0, T1]):
            row = T[s]
            nz = np.where(row > 1e-15)[0]
            transitions[(s, a)] = MDPTransition(
                next_states=nz.copy(),
                probs=row[nz].copy(),
                reward=0.0,
            )
    return MDP(n_states=3, n_actions=2, transitions=transitions, initial_state=0)


@pytest.fixture
def simple_transition_matrix():
    return np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.05, 0.15, 0.8],
    ])


# ===================================================================
# 1. Interval Arithmetic Soundness
# ===================================================================

class TestIntervalArithmetic:
    """Test that interval operations produce sound enclosures."""

    def test_point_interval_contains_value(self):
        x = 3.14159
        iv = Interval.point(x)
        assert iv.contains(x)

    def test_add_soundness(self, rng):
        """For random a ∈ A, b ∈ B, a + b must be in A + B."""
        for _ in range(200):
            a_lo, a_hi = sorted(rng.uniform(-10, 10, 2))
            b_lo, b_hi = sorted(rng.uniform(-10, 10, 2))
            A = Interval(a_lo, a_hi)
            B = Interval(b_lo, b_hi)
            C = A + B
            a = rng.uniform(a_lo, a_hi)
            b = rng.uniform(b_lo, b_hi)
            assert C.contains(a + b), f"{a}+{b}={a+b} not in {C}"

    def test_sub_soundness(self, rng):
        for _ in range(200):
            a_lo, a_hi = sorted(rng.uniform(-10, 10, 2))
            b_lo, b_hi = sorted(rng.uniform(-10, 10, 2))
            A = Interval(a_lo, a_hi)
            B = Interval(b_lo, b_hi)
            C = A - B
            a = rng.uniform(a_lo, a_hi)
            b = rng.uniform(b_lo, b_hi)
            assert C.contains(a - b), f"{a}-{b}={a-b} not in {C}"

    def test_mul_soundness(self, rng):
        for _ in range(200):
            a_lo, a_hi = sorted(rng.uniform(-10, 10, 2))
            b_lo, b_hi = sorted(rng.uniform(-10, 10, 2))
            A = Interval(a_lo, a_hi)
            B = Interval(b_lo, b_hi)
            C = A * B
            a = rng.uniform(a_lo, a_hi)
            b = rng.uniform(b_lo, b_hi)
            assert C.contains(a * b), f"{a}*{b}={a*b} not in {C}"

    def test_div_soundness(self, rng):
        for _ in range(200):
            a_lo, a_hi = sorted(rng.uniform(-10, 10, 2))
            # Avoid division by zero
            b_lo = rng.uniform(0.1, 5)
            b_hi = rng.uniform(b_lo, 10)
            A = Interval(a_lo, a_hi)
            B = Interval(b_lo, b_hi)
            C = A / B
            a = rng.uniform(a_lo, a_hi)
            b = rng.uniform(b_lo, b_hi)
            assert C.contains(a / b), f"{a}/{b}={a/b} not in {C}"

    def test_div_by_zero_interval(self):
        A = Interval(1.0, 2.0)
        B = Interval(-1.0, 1.0)
        C = A / B
        assert C.lo == -math.inf
        assert C.hi == math.inf

    def test_sqrt_soundness(self, rng):
        for _ in range(100):
            lo = rng.uniform(0, 10)
            hi = rng.uniform(lo, 20)
            A = Interval(lo, hi)
            S = A.sqrt()
            x = rng.uniform(lo, hi)
            assert S.contains(math.sqrt(x)), f"sqrt({x}) not in {S}"

    def test_exp_soundness(self, rng):
        for _ in range(100):
            lo, hi = sorted(rng.uniform(-5, 5, 2))
            A = Interval(lo, hi)
            E = A.exp()
            x = rng.uniform(lo, hi)
            assert E.contains(math.exp(x)), f"exp({x}) not in {E}"

    def test_log_soundness(self, rng):
        for _ in range(100):
            lo = rng.uniform(0.01, 5)
            hi = rng.uniform(lo, 10)
            A = Interval(lo, hi)
            L = A.log()
            x = rng.uniform(lo, hi)
            assert L.contains(math.log(x)), f"log({x}) not in {L}"

    def test_negation(self):
        A = Interval(1.0, 3.0)
        B = -A
        assert B.lo == -3.0
        assert B.hi == -1.0

    def test_abs(self):
        assert Interval(-3, -1).abs() == Interval(1.0, 3.0)
        assert Interval(1, 5).abs() == Interval(1.0, 5.0)
        mixed = Interval(-2, 5).abs()
        assert mixed.lo == 0.0
        assert mixed.hi == 5.0

    def test_hull_and_intersect(self):
        A = Interval(1, 3)
        B = Interval(2, 5)
        h = A.hull(B)
        assert h.lo == 1.0
        assert h.hi == 5.0
        i = A.intersect(B)
        assert i is not None
        assert i.lo == 2.0
        assert i.hi == 3.0

    def test_disjoint_intersect(self):
        A = Interval(1, 2)
        B = Interval(3, 4)
        assert A.intersect(B) is None

    def test_contains_interval(self):
        A = Interval(0, 10)
        B = Interval(2, 5)
        assert A.contains_interval(B)
        assert not B.contains_interval(A)

    def test_polyval_soundness(self, rng):
        """Test polynomial evaluation over intervals."""
        coeffs = [1.0, -2.0, 3.0]  # x^2 - 2x + 3
        for _ in range(100):
            lo, hi = sorted(rng.uniform(-5, 5, 2))
            X = Interval(lo, hi)
            result = interval_polyval(coeffs, X)
            x = rng.uniform(lo, hi)
            val = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
            assert result.contains(val), f"p({x})={val} not in {result}"


# ===================================================================
# 2. Interval Vector and Matrix
# ===================================================================

class TestIntervalVectorMatrix:

    def test_interval_vector_contains_point(self, rng):
        lo = np.array([0.0, -1.0, 2.0])
        hi = np.array([1.0, 1.0, 5.0])
        iv = IntervalVector.from_bounds(lo, hi)
        for _ in range(100):
            pt = rng.uniform(lo, hi)
            assert iv.contains_point(pt)

    def test_interval_vector_rejects_outside(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        iv = IntervalVector.from_bounds(lo, hi)
        assert not iv.contains_point(np.array([1.5, 0.5]))

    def test_interval_vector_add(self):
        a = IntervalVector([Interval(1, 2), Interval(3, 4)])
        b = IntervalVector([Interval(0.5, 1.5), Interval(1, 2)])
        c = a + b
        assert c[0].contains(2.0)
        assert c[1].contains(5.0)

    def test_interval_matrix_from_matrix(self, rng):
        M = rng.uniform(0, 1, (4, 4))
        im = IntervalMatrix.from_matrix(M)
        assert im.contains_matrix(M)

    def test_interval_matrix_contains(self, rng):
        M = rng.uniform(0, 1, (3, 3))
        lo = M - 0.01
        hi = M + 0.01
        im = IntervalMatrix.from_bounds(lo, hi)
        assert im.contains_matrix(M)
        assert not im.contains_matrix(M + 0.02)

    def test_interval_matmul_soundness(self, rng):
        """A_iv * v_iv should contain A * v for any A in A_iv, v in v_iv."""
        n = 5
        for _ in range(50):
            M = rng.uniform(-2, 2, (n, n))
            v = rng.uniform(-2, 2, n)
            M_iv = IntervalMatrix.from_matrix(M)
            v_iv = IntervalVector.from_point(v)
            result = interval_matmul(M_iv, v_iv)
            exact = M @ v
            assert result.contains_point(exact), (
                f"M@v not contained: exact={exact}, result_lo={result.lo_array()}, "
                f"result_hi={result.hi_array()}"
            )

    def test_interval_matmul_wide_intervals(self, rng):
        """Test with wider interval inputs."""
        n = 3
        M_lo = rng.uniform(-1, 0, (n, n))
        M_hi = M_lo + rng.uniform(0.5, 2, (n, n))
        v_lo = rng.uniform(-1, 0, n)
        v_hi = v_lo + rng.uniform(0.5, 2, n)

        M_iv = IntervalMatrix.from_bounds(M_lo, M_hi)
        v_iv = IntervalVector.from_bounds(v_lo, v_hi)
        result = interval_matmul(M_iv, v_iv)

        # Sample random M, v from intervals and check containment
        for _ in range(200):
            M = rng.uniform(M_lo, M_hi)
            v = rng.uniform(v_lo, v_hi)
            exact = M @ v
            assert result.contains_point(exact)

    def test_interval_matmul_matrix(self, rng):
        """Test matrix-matrix multiplication soundness."""
        n = 3
        A = rng.uniform(-2, 2, (n, n))
        B = rng.uniform(-2, 2, (n, n))
        A_iv = IntervalMatrix.from_matrix(A)
        B_iv = IntervalMatrix.from_matrix(B)
        C_iv = interval_matmul_matrix(A_iv, B_iv)
        C = A @ B
        assert C_iv.contains_matrix(C)


# ===================================================================
# 3. State Abstraction
# ===================================================================

class TestStateAbstraction:

    def test_abstract_state_contains(self):
        s = AbstractState(lo=np.array([0.0, 0.0]), hi=np.array([1.0, 1.0]))
        assert s.contains_point(np.array([0.5, 0.5]))
        assert not s.contains_point(np.array([1.5, 0.5]))

    def test_abstract_state_bisect(self):
        s = AbstractState(lo=np.array([0.0, 0.0]), hi=np.array([2.0, 1.0]))
        left, right = s.bisect()
        # Should split along dim 0 (widest)
        assert left.hi[0] == pytest.approx(1.0)
        assert right.lo[0] == pytest.approx(1.0)
        # Check coverage
        assert left.contains_point(np.array([0.5, 0.5]))
        assert right.contains_point(np.array([1.5, 0.5]))

    def test_abstract_state_volume(self):
        s = AbstractState(lo=np.array([0.0, 0.0, 0.0]), hi=np.array([2.0, 3.0, 4.0]))
        assert s.volume == pytest.approx(24.0)

    def test_discretize_state_space(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        abst = discretize_state_space(lo, hi, n_bins=5)
        assert abst.n_abstract == 25  # 5x5

    def test_discretize_covers_space(self, rng):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        abst = discretize_state_space(lo, hi, n_bins=4)
        for _ in range(200):
            pt = rng.uniform(lo, hi)
            assert abst.covers_point(pt)

    def test_abstraction_function_maps(self):
        lo = np.array([0.0])
        hi = np.array([1.0])
        abst = discretize_state_space(lo, hi, n_bins=4)
        cs = ConcreteState(values=(0.3,))
        abs_s = abst.abstract(cs)
        assert abs_s.contains_concrete(cs)

    def test_concrete_state_roundtrip(self):
        cs = ConcreteState.from_array(np.array([1.0, 2.0, 3.0]))
        assert cs.dim == 3
        arr = cs.to_array()
        assert np.allclose(arr, [1.0, 2.0, 3.0])

    def test_partition_verification(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        abst = discretize_state_space(lo, hi, n_bins=3)
        assert abst.verify_partition(lo, hi)


# ===================================================================
# 4. Abstract MDP Overapproximation
# ===================================================================

class TestAbstractMDP:

    def test_abstract_mdp_construction(self, small_mdp):
        """Abstract MDP should have correct dimensions."""
        state_vecs = np.array([[0.1], [0.5], [0.9]])
        abst = discretize_state_space(
            np.array([0.0]), np.array([1.0]), n_bins=3
        )
        abs_mdp = compute_abstract_transitions(small_mdp, abst, state_vecs)
        assert abs_mdp.n_states == 3
        assert abs_mdp.n_actions == 2

    def test_overapproximation_on_known_mdp(self, small_mdp):
        """Verify overapproximation holds for the constructed abstract MDP."""
        state_vecs = np.array([[0.1], [0.5], [0.9]])
        abst = discretize_state_space(
            np.array([0.0]), np.array([1.0]), n_bins=3
        )
        abs_mdp = compute_abstract_transitions(small_mdp, abst, state_vecs)
        result = verify_overapproximation(
            small_mdp, abs_mdp, abst, state_vecs
        )
        assert result.sound, f"Violations: {result.violations}"

    def test_overapproximation_random_mdp(self, rng):
        """Test on randomly generated MDPs."""
        n_states = 6
        n_actions = 2
        for _ in range(10):
            transitions = {}
            for s in range(n_states):
                for a in range(n_actions):
                    probs = rng.dirichlet(np.ones(n_states))
                    transitions[(s, a)] = MDPTransition(
                        next_states=np.arange(n_states),
                        probs=probs,
                    )
            mdp = MDP(n_states=n_states, n_actions=n_actions,
                      transitions=transitions)

            state_vecs = np.linspace(0, 1, n_states).reshape(-1, 1)
            abst = discretize_state_space(
                np.array([0.0]), np.array([1.0]), n_bins=n_states
            )
            abs_mdp = compute_abstract_transitions(mdp, abst, state_vecs)
            result = verify_overapproximation(mdp, abs_mdp, abst, state_vecs)
            assert result.sound, f"Failed on random MDP: {result.violations[:3]}"

    def test_abstract_transitions_normalised(self, small_mdp):
        """Abstract transition rows should sum to 1."""
        state_vecs = np.array([[0.1], [0.5], [0.9]])
        abst = discretize_state_space(
            np.array([0.0]), np.array([1.0]), n_bins=3
        )
        abs_mdp = compute_abstract_transitions(small_mdp, abst, state_vecs)
        for (s, a), tr in abs_mdp.transitions.items():
            assert abs(tr.probs.sum() - 1.0) < 1e-10, (
                f"Row ({s},{a}) sums to {tr.probs.sum()}"
            )


# ===================================================================
# 5. Refinement
# ===================================================================

class TestRefinement:

    def test_refinement_increases_states(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        abst = discretize_state_space(lo, hi, n_bins=2)
        assert abst.n_abstract == 4
        refined = refinement_step(abst, target_index=0)
        assert refined.n_abstract == 5

    def test_refinement_preserves_coverage(self, rng):
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        abst = discretize_state_space(lo, hi, n_bins=3)
        refined = refinement_step(abst, target_index=4)
        for _ in range(100):
            pt = rng.uniform(lo, hi)
            assert refined.covers_point(pt)

    def test_refinement_reduces_max_volume(self):
        lo = np.array([0.0])
        hi = np.array([1.0])
        abst = discretize_state_space(lo, hi, n_bins=2)
        max_vol_before = max(s.volume for s in abst.abstract_states())
        refined = refinement_step(abst, target_index=0)
        max_vol_after = max(s.volume for s in refined.abstract_states())
        assert max_vol_after <= max_vol_before


# ===================================================================
# 6. Safety Transfer
# ===================================================================

class TestSafetyTransfer:

    def test_abstract_safety_implies_concrete(self, small_mdp):
        """If abstract MDP is safe, concrete MDP should be safe."""
        state_vecs = np.array([[0.1], [0.5], [0.9]])
        abst = discretize_state_space(
            np.array([0.0]), np.array([1.0]), n_bins=3
        )
        abs_mdp = compute_abstract_transitions(small_mdp, abst, state_vecs)

        spec = Specification(
            kind=SpecKind.SAFETY,
            safe_states=frozenset([0, 1, 2]),
            horizon=5,
        )

        coa = ConservativeOverapproximation()
        cert = coa.certify_soundness(
            small_mdp, abs_mdp, abst, spec, state_vecs
        )
        assert cert.partition_covers
        assert cert.overapproximation_holds

    def test_certify_full_pipeline(self, rng):
        """Full verify-and-refine pipeline on a simple MDP."""
        n_states = 4
        transitions = {}
        for s in range(n_states):
            for a in range(2):
                probs = rng.dirichlet(np.ones(n_states) * 5)
                transitions[(s, a)] = MDPTransition(
                    next_states=np.arange(n_states),
                    probs=probs,
                )
        mdp = MDP(n_states=n_states, n_actions=2, transitions=transitions)

        spec = Specification(
            kind=SpecKind.SAFETY,
            safe_states=frozenset(range(n_states)),
            horizon=3,
        )
        state_vecs = np.linspace(0, 1, n_states).reshape(-1, 1)

        coa = ConservativeOverapproximation()
        cert, abst, abs_mdp = coa.verify_and_refine(
            mdp, spec,
            bounds_lo=np.array([0.0]),
            bounds_hi=np.array([1.0]),
            initial_bins=n_states,
            state_vectors=state_vecs,
        )
        assert cert.partition_covers
        assert cert.overapproximation_holds
        assert cert.n_abstract_states >= n_states

    def test_interval_safety_probability(self):
        """Interval safety probability should contain the exact value."""
        T = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ])
        T_iv = IntervalMatrix.from_matrix(T)
        safe = frozenset([0, 1])

        prob_iv = interval_safety_probability(T_iv, safe, initial_state=0, horizon=5)
        assert prob_iv.contains(1.0)  # All states safe

    def test_interval_safety_with_unsafe_state(self):
        """With an unsafe state, safety probability < 1."""
        T = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
        T_iv = IntervalMatrix.from_matrix(T)
        safe = frozenset([0])  # Only state 0 is safe

        prob_iv = interval_safety_probability(T_iv, safe, initial_state=0, horizon=3)
        # Should contain the exact probability
        assert prob_iv.lo >= 0.0
        assert prob_iv.hi <= 1.0
        # Exact: prob of staying in state 0 for 3 steps = 0.8^3 = 0.512
        assert prob_iv.contains(0.8**3)


# ===================================================================
# 7. Independent Verifier
# ===================================================================

class TestIndependentVerifier:

    def test_pac_bayes_recomputation(self, rng):
        """PAC-Bayes bound recomputation should be consistent."""
        K = 3
        prior = np.ones((K, K))
        # Generate transitions
        n_trans = 500
        transitions = np.column_stack([
            rng.integers(0, K, n_trans),
            rng.integers(0, K, n_trans),
        ])
        # Compute posterior
        observed = np.zeros((K, K))
        for s, sp in transitions:
            observed[s, sp] += 1
        posterior = prior + observed

        verifier = IndependentVerifier()
        ok, bound, disc = verifier.verify_pac_bayes_bound(
            transitions, prior, posterior, delta=0.05,
        )
        assert ok
        assert bound > 0
        assert bound < 1.0  # Should be a reasonable bound

    def test_pac_bayes_detects_discrepancy(self, rng):
        """Verifier should detect incorrect claimed bounds."""
        K = 2
        prior = np.ones((K, K))
        transitions = np.column_stack([
            rng.integers(0, K, 200),
            rng.integers(0, K, 200),
        ])
        observed = np.zeros((K, K))
        for s, sp in transitions:
            observed[s, sp] += 1
        posterior = prior + observed

        verifier = IndependentVerifier(tolerance=1e-6)
        ok, bound, disc = verifier.verify_pac_bayes_bound(
            transitions, prior, posterior, delta=0.05,
            claimed_bound=0.0001,  # Way too small
        )
        assert not ok or disc > 1e-4

    def test_composition_verification(self):
        verifier = IndependentVerifier()
        ok, composed = verifier.verify_composition(0.05, 0.03, claimed_composed=0.08)
        assert ok
        assert composed == pytest.approx(0.08)

    def test_composition_detects_error(self):
        verifier = IndependentVerifier()
        ok, composed = verifier.verify_composition(
            0.05, 0.03, claimed_composed=0.01  # Too small!
        )
        assert not ok

    def test_abstraction_verification_sound(self):
        """Sound abstract MDP should pass verification."""
        # Use a concrete T where aggregated transitions don't conflict
        # so the abstract T can be a valid overapproximation.
        concrete_T = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.0, 0.2, 0.8],
        ])
        c2a = np.array([0, 0, 1])
        # s=0 (abs=0): probs_to_abs = [0.9, 0.1]
        # s=1 (abs=0): probs_to_abs = [0.9, 0.1]
        # s=2 (abs=1): probs_to_abs = [0.2, 0.8]
        # Per-entry max for abs=0: [0.9, 0.1] -> sum=1.0, perfect
        abstract_T = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ])

        verifier = IndependentVerifier()
        ok, warnings = verifier.verify_state_abstraction(
            concrete_T, abstract_T, c2a
        )
        assert ok, f"Unexpected warnings: {warnings}"

    def test_abstraction_verification_unsound(self):
        """Underapproximated abstract MDP should fail."""
        concrete_T = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        c2a = np.array([0, 1])
        abstract_T = np.array([
            [0.9, 0.1],  # Under-approximates 0.5 → state 1
            [0.1, 0.9],
        ])
        verifier = IndependentVerifier()
        ok, warnings = verifier.verify_state_abstraction(
            concrete_T, abstract_T, c2a
        )
        assert not ok
        assert len(warnings) > 0

    def test_full_audit(self, rng):
        K = 2
        prior = np.ones((K, K))
        transitions = np.column_stack([
            rng.integers(0, K, 300),
            rng.integers(0, K, 300),
        ])
        observed = np.zeros((K, K))
        for s, sp in transitions:
            observed[s, sp] += 1
        posterior = prior + observed

        verifier = IndependentVerifier()
        _, bound, _ = verifier.verify_pac_bayes_bound(
            transitions, prior, posterior, delta=0.05
        )

        certificate = {
            "pac_bayes_bound": bound,
            "delta": 0.05,
            "bound_type": "catoni",
            "causal_bound": 0.03,
            "shield_bound": 0.02,
            "composed_bound": 0.05,
        }
        raw_data = {
            "transitions": transitions,
            "prior_counts": prior,
            "posterior_counts": posterior,
        }

        report = verifier.full_audit(certificate, raw_data)
        assert report.pac_bayes_bound_verified
        assert report.composition_verified
        assert report.abstraction_sound
        assert report.all_verified

    def test_verification_report_summary(self):
        report = VerificationReport(
            pac_bayes_bound_verified=True,
            pac_bayes_bound_value=0.05,
            pac_bayes_bound_discrepancy=1e-8,
            composition_verified=True,
            abstraction_sound=True,
        )
        summary = report.summary()
        assert "PASS" in summary

    def test_verification_report_fail_summary(self):
        report = VerificationReport(
            pac_bayes_bound_verified=False,
            pac_bayes_bound_value=0.05,
            pac_bayes_bound_discrepancy=0.1,
            composition_verified=True,
            abstraction_sound=True,
            warnings=["Bound mismatch"],
        )
        summary = report.summary()
        assert "FAIL" in summary

    def test_bound_types(self, rng):
        """All bound types should produce valid results."""
        K = 2
        prior = np.ones((K, K))
        transitions = np.column_stack([
            rng.integers(0, K, 100),
            rng.integers(0, K, 100),
        ])
        observed = np.zeros((K, K))
        for s, sp in transitions:
            observed[s, sp] += 1
        posterior = prior + observed

        verifier = IndependentVerifier()
        for bt in ["catoni", "mcallester", "maurer"]:
            ok, bound, _ = verifier.verify_pac_bayes_bound(
                transitions, prior, posterior, delta=0.05, bound_type=bt,
            )
            assert ok
            assert 0 < bound < 10  # Reasonable range


# ===================================================================
# 8. Property-Based Tests
# ===================================================================

class TestPropertyBased:

    def test_abstraction_always_covers_concrete(self, rng):
        """For random states in bounds, abstraction must cover them."""
        dims = [1, 2, 3]
        for dim in dims:
            lo = np.zeros(dim)
            hi = np.ones(dim)
            abst = discretize_state_space(lo, hi, n_bins=5)
            for _ in range(100):
                pt = rng.uniform(lo, hi)
                cs = ConcreteState.from_array(pt)
                abs_s = abst.abstract(cs)
                assert abs_s.contains_concrete(cs)

    def test_interval_operations_always_contain_point(self, rng):
        """Any scalar op on points in intervals must stay in result interval."""
        ops = [
            ("add", lambda a, b: a + b, lambda a, b: a + b),
            ("sub", lambda a, b: a - b, lambda a, b: a - b),
            ("mul", lambda a, b: a * b, lambda a, b: a * b),
        ]
        for name, iv_op, scalar_op in ops:
            for _ in range(100):
                a_lo, a_hi = sorted(rng.uniform(-5, 5, 2))
                b_lo, b_hi = sorted(rng.uniform(-5, 5, 2))
                A = Interval(a_lo, a_hi)
                B = Interval(b_lo, b_hi)
                C = iv_op(A, B)
                a = rng.uniform(a_lo, a_hi)
                b = rng.uniform(b_lo, b_hi)
                assert C.contains(scalar_op(a, b)), (
                    f"{name}({a},{b})={scalar_op(a,b)} not in {C}"
                )

    def test_matmul_monotonicity(self, rng):
        """Wider input intervals should produce wider output intervals."""
        n = 3
        M = rng.uniform(0, 1, (n, n))
        v = rng.uniform(0, 1, n)

        # Narrow intervals
        M_narrow = IntervalMatrix.from_bounds(M - 0.01, M + 0.01)
        v_narrow = IntervalVector.from_bounds(v - 0.01, v + 0.01)
        res_narrow = interval_matmul(M_narrow, v_narrow)

        # Wider intervals
        M_wide = IntervalMatrix.from_bounds(M - 0.1, M + 0.1)
        v_wide = IntervalVector.from_bounds(v - 0.1, v + 0.1)
        res_wide = interval_matmul(M_wide, v_wide)

        # Wide should contain narrow
        assert res_wide.contains_vector(res_narrow)


# ===================================================================
# 9. Certificate Round-Trip
# ===================================================================

class TestCertificateRoundTrip:

    def test_generate_and_verify_certificate(self, rng):
        """Generate a certificate and independently verify it."""
        # Build a simple MDP
        K = 3
        n_states = K
        transitions = {}
        T_matrix = np.zeros((n_states, n_states))
        for s in range(n_states):
            probs = rng.dirichlet(np.ones(n_states) * 3)
            T_matrix[s] = probs
            transitions[(s, 0)] = MDPTransition(
                next_states=np.arange(n_states),
                probs=probs.copy(),
            )
        mdp = MDP(n_states=n_states, n_actions=1, transitions=transitions)

        # Generate abstract MDP
        state_vecs = np.linspace(0, 1, n_states).reshape(-1, 1)
        abst = discretize_state_space(
            np.array([0.0]), np.array([1.0]), n_bins=n_states
        )
        abs_mdp = compute_abstract_transitions(mdp, abst, state_vecs)

        # Certify
        spec = Specification(
            kind=SpecKind.SAFETY,
            safe_states=frozenset(range(n_states)),
            horizon=5,
        )
        coa = ConservativeOverapproximation()
        cert = coa.certify_soundness(mdp, abs_mdp, abst, spec, state_vecs)
        assert cert.sound

        # Build abstract transition matrix
        abs_T = np.zeros((abst.n_abstract, abst.n_abstract))
        for (s, a), tr in abs_mdp.transitions.items():
            if a == 0:
                for ns, p in zip(tr.next_states, tr.probs):
                    abs_T[s, int(ns)] = p

        c2a = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            cs = ConcreteState.from_array(state_vecs[s])
            c2a[s] = abst.abstract(cs).index

        # Independent verification
        verifier = IndependentVerifier()
        ok, warnings = verifier.verify_state_abstraction(
            T_matrix, abs_T, c2a
        )
        assert ok, f"Independent verification failed: {warnings}"

    def test_end_to_end_with_audit(self, rng):
        """Full end-to-end: build MDP, abstract, certify, audit."""
        K = 2
        prior = np.ones((K, K))
        n_trans = 200
        raw_transitions = np.column_stack([
            rng.integers(0, K, n_trans),
            rng.integers(0, K, n_trans),
        ])
        observed = np.zeros((K, K))
        for s, sp in raw_transitions:
            observed[s, sp] += 1
        posterior = prior + observed

        verifier = IndependentVerifier()
        ok, bound, disc = verifier.verify_pac_bayes_bound(
            raw_transitions, prior, posterior, delta=0.05,
        )
        assert ok

        certificate = {
            "pac_bayes_bound": bound,
            "delta": 0.05,
            "bound_type": "catoni",
        }
        raw_data = {
            "transitions": raw_transitions,
            "prior_counts": prior,
            "posterior_counts": posterior,
        }
        report = verifier.full_audit(certificate, raw_data)
        assert report.pac_bayes_bound_verified
        assert report.audit_timestamp != ""


# ===================================================================
# 10. Edge Cases
# ===================================================================

class TestEdgeCases:

    def test_single_state_mdp(self):
        """Single-state MDP should work."""
        transitions = {(0, 0): MDPTransition(
            next_states=np.array([0]), probs=np.array([1.0])
        )}
        mdp = MDP(n_states=1, n_actions=1, transitions=transitions)
        abst = discretize_state_space(np.array([0.0]), np.array([1.0]), n_bins=1)
        abs_mdp = compute_abstract_transitions(
            mdp, abst, np.array([[0.5]])
        )
        assert abs_mdp.n_states == 1

    def test_empty_interval_rejected(self):
        with pytest.raises(ValueError):
            Interval(5.0, 3.0)

    def test_zero_width_interval(self):
        iv = Interval(2.0, 2.0)
        assert iv.width == 0.0
        assert iv.mid == 2.0
        assert iv.contains(2.0)

    def test_interval_vector_from_point(self):
        pt = np.array([1.0, 2.0, 3.0])
        iv = IntervalVector.from_point(pt)
        assert iv.contains_point(pt)
        assert iv.dim == 3

    def test_high_dimensional_discretization(self):
        """3D discretization with 2 bins per dim = 8 states."""
        lo = np.zeros(3)
        hi = np.ones(3)
        abst = discretize_state_space(lo, hi, n_bins=2)
        assert abst.n_abstract == 8

    def test_abstract_state_sample(self, rng):
        s = AbstractState(lo=np.array([0.0, 0.0]), hi=np.array([1.0, 1.0]))
        samples = s.sample_uniform(rng, 50)
        assert samples.shape == (50, 2)
        for pt in samples:
            assert s.contains_point(pt)
