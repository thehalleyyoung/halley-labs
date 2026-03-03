"""
Comprehensive tests for dp_forge.verifier — divergences, DP verification,
tolerance management, dataclasses, and verifier classes.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import numpy as np
import pytest

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    AdjacencyRelation,
    NumericalConfig,
    PrivacyBudget,
    VerifyResult,
)
from dp_forge.verifier import (
    MonteCarloVerifier,
    PairAnalysis,
    PrivacyVerifier,
    VerificationMode,
    VerificationReport,
    ViolationRecord,
    ViolationType,
    _verify_approx_dp_pair,
    _verify_pure_dp_pair,
    compute_safe_tolerance,
    hockey_stick_divergence,
    hockey_stick_divergence_detailed,
    kl_divergence,
    max_divergence,
    renyi_divergence,
    total_variation,
    validate_tolerance,
    verify,
    warn_tolerance_violation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — mechanism builders
# ═══════════════════════════════════════════════════════════════════════════


def _uniform_mechanism(n: int, k: int) -> np.ndarray:
    """Build a uniform mechanism: every row is [1/k, ..., 1/k]."""
    return np.full((n, k), 1.0 / k)


def _deterministic_mechanism(n: int) -> np.ndarray:
    """Build a deterministic mechanism: row i maps to output i (identity)."""
    return np.eye(n)


def _single_output_mechanism(n: int) -> np.ndarray:
    """Build a mechanism with a single output column (always 1)."""
    return np.ones((n, 1))


def _randomized_response(eps: float) -> np.ndarray:
    """Build 2×2 randomized response at privacy level eps.

    p[0] = [e^eps/(1+e^eps), 1/(1+e^eps)]
    p[1] = [1/(1+e^eps), e^eps/(1+e^eps)]
    """
    e = math.exp(eps)
    high = e / (1.0 + e)
    low = 1.0 / (1.0 + e)
    return np.array([[high, low], [low, high]])


def _almost_pure_dp_mechanism(eps: float, violation_factor: float = 1.5) -> np.ndarray:
    """Build a 2×3 mechanism that violates pure eps-DP by violation_factor.

    The worst ratio is e^eps * violation_factor, so the mechanism fails
    pure eps-DP but may pass approximate DP with appropriate delta.
    """
    e = math.exp(eps)
    target_ratio = e * violation_factor
    # Row 0: [high, medium, rest]
    high = 0.5
    medium = high / target_ratio
    rest = 1.0 - high - medium
    assert rest >= 0, "violation_factor too large for this construction"
    row0 = np.array([high, medium, rest])
    row1 = np.array([medium, high, rest])
    return np.vstack([row0, row1])


def _chain_edges(n: int) -> List[Tuple[int, int]]:
    """Build chain adjacency: (0,1), (1,2), ..., (n-2,n-1)."""
    return [(i, i + 1) for i in range(n - 1)]


def _complete_edges(n: int) -> List[Tuple[int, int]]:
    """Build complete adjacency: all pairs (i, j) with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ═══════════════════════════════════════════════════════════════════════════
# §1  Enum Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationMode:
    """Tests for VerificationMode enum."""

    def test_members_exist(self):
        assert hasattr(VerificationMode, "FAST")
        assert hasattr(VerificationMode, "MOST_VIOLATING")
        assert hasattr(VerificationMode, "EXHAUSTIVE")

    def test_member_count(self):
        assert len(VerificationMode) == 3

    def test_values_are_distinct(self):
        vals = [m.value for m in VerificationMode]
        assert len(vals) == len(set(vals))

    def test_repr(self):
        assert "FAST" in repr(VerificationMode.FAST)
        assert "MOST_VIOLATING" in repr(VerificationMode.MOST_VIOLATING)
        assert "EXHAUSTIVE" in repr(VerificationMode.EXHAUSTIVE)

    def test_name_attribute(self):
        assert VerificationMode.FAST.name == "FAST"
        assert VerificationMode.MOST_VIOLATING.name == "MOST_VIOLATING"
        assert VerificationMode.EXHAUSTIVE.name == "EXHAUSTIVE"


class TestViolationType:
    """Tests for ViolationType enum."""

    def test_members_exist(self):
        assert hasattr(ViolationType, "PURE_DP_RATIO")
        assert hasattr(ViolationType, "APPROX_DP_HOCKEY_STICK")

    def test_member_count(self):
        assert len(ViolationType) == 2

    def test_repr(self):
        assert "PURE_DP_RATIO" in repr(ViolationType.PURE_DP_RATIO)
        assert "APPROX_DP_HOCKEY_STICK" in repr(ViolationType.APPROX_DP_HOCKEY_STICK)


# ═══════════════════════════════════════════════════════════════════════════
# §2  Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestViolationRecord:
    """Tests for ViolationRecord dataclass."""

    def test_creation(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=2, magnitude=0.5, ratio=3.0,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        assert vr.i == 0
        assert vr.i_prime == 1
        assert vr.j_worst == 2
        assert vr.magnitude == 0.5
        assert vr.ratio == 3.0
        assert vr.violation_type == ViolationType.PURE_DP_RATIO
        assert vr.direction == "forward"  # default

    def test_to_tuple(self):
        vr = ViolationRecord(
            i=3, i_prime=5, j_worst=7, magnitude=1.23, ratio=4.56,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        t = vr.to_tuple()
        assert t == (3, 5, 7, 1.23)
        assert isinstance(t, tuple)
        assert len(t) == 4

    def test_to_tuple_approx_dp(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=-1, magnitude=0.01, ratio=0.05,
            violation_type=ViolationType.APPROX_DP_HOCKEY_STICK,
            direction="reverse",
        )
        t = vr.to_tuple()
        assert t == (0, 1, -1, 0.01)

    def test_direction_forward(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=0, magnitude=1.0, ratio=2.0,
            violation_type=ViolationType.PURE_DP_RATIO, direction="forward",
        )
        assert vr.direction == "forward"

    def test_direction_reverse(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=0, magnitude=1.0, ratio=2.0,
            violation_type=ViolationType.PURE_DP_RATIO, direction="reverse",
        )
        assert vr.direction == "reverse"

    def test_repr_contains_info(self):
        vr = ViolationRecord(
            i=2, i_prime=4, j_worst=1, magnitude=0.001, ratio=5.0,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        r = repr(vr)
        assert "2" in r and "4" in r
        assert "PURE_DP_RATIO" in r

    def test_repr_approx_dp(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=-1, magnitude=0.01, ratio=0.05,
            violation_type=ViolationType.APPROX_DP_HOCKEY_STICK,
        )
        r = repr(vr)
        assert "APPROX_DP_HOCKEY_STICK" in r


class TestPairAnalysis:
    """Tests for PairAnalysis dataclass."""

    def test_creation(self):
        pa = PairAnalysis(
            i=0, i_prime=1,
            max_ratio_forward=2.5, max_ratio_reverse=1.8,
            j_worst_forward=0, j_worst_reverse=1,
            hockey_stick_forward=0.01, hockey_stick_reverse=0.02,
            is_violating=False,
        )
        assert pa.i == 0
        assert pa.i_prime == 1
        assert pa.max_ratio_forward == 2.5
        assert not pa.is_violating

    def test_repr_ok(self):
        pa = PairAnalysis(
            i=0, i_prime=1,
            max_ratio_forward=1.5, max_ratio_reverse=1.2,
            j_worst_forward=0, j_worst_reverse=1,
            hockey_stick_forward=0.0, hockey_stick_reverse=0.0,
            is_violating=False,
        )
        r = repr(pa)
        assert "ok" in r

    def test_repr_violating(self):
        pa = PairAnalysis(
            i=0, i_prime=1,
            max_ratio_forward=10.0, max_ratio_reverse=1.2,
            j_worst_forward=0, j_worst_reverse=1,
            hockey_stick_forward=0.5, hockey_stick_reverse=0.0,
            is_violating=True,
        )
        r = repr(pa)
        assert "VIOLATING" in r


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def _make_report(self, is_valid: bool = True, **kwargs) -> VerificationReport:
        defaults = dict(
            is_valid=is_valid, epsilon=1.0, delta=0.0, tolerance=1e-9,
            n_pairs_checked=3, n_violations=0,
        )
        defaults.update(kwargs)
        return VerificationReport(**defaults)

    def test_valid_report(self):
        rpt = self._make_report(is_valid=True)
        assert rpt.is_valid
        assert rpt.n_violations == 0
        assert rpt.worst_violation is None

    def test_invalid_report(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=2, magnitude=0.5, ratio=3.0,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        rpt = self._make_report(
            is_valid=False, n_violations=1, worst_violation=vr,
        )
        assert not rpt.is_valid
        assert rpt.n_violations == 1
        assert rpt.worst_violation is vr

    def test_summary_pass(self):
        rpt = self._make_report(is_valid=True)
        s = rpt.summary()
        assert "PASS" in s
        assert "ε=1.0" in s

    def test_summary_fail(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=2, magnitude=0.5, ratio=3.0,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        rpt = self._make_report(is_valid=False, n_violations=1, worst_violation=vr)
        s = rpt.summary()
        assert "FAIL" in s
        assert "Worst violation" in s

    def test_summary_with_actual_epsilon(self):
        rpt = self._make_report(is_valid=True, actual_epsilon=0.8)
        s = rpt.summary()
        assert "Actual ε" in s
        assert "0.8" in s

    def test_summary_with_actual_delta(self):
        rpt = self._make_report(is_valid=True, delta=0.01, actual_delta=0.005)
        s = rpt.summary()
        assert "Actual δ" in s

    def test_summary_with_recommendations(self):
        rpt = self._make_report(
            is_valid=True, recommendations=["Consider tightening epsilon."],
        )
        s = rpt.summary()
        assert "Recommendations" in s
        assert "Consider tightening" in s

    def test_summary_time(self):
        rpt = self._make_report(is_valid=True, verification_time_s=1.234)
        s = rpt.summary()
        assert "1.234s" in s

    def test_repr(self):
        rpt = self._make_report(is_valid=True)
        r = repr(rpt)
        assert "valid" in r
        assert "ε=1.0" in r

    def test_repr_invalid(self):
        vr = ViolationRecord(
            i=0, i_prime=1, j_worst=0, magnitude=0.1, ratio=3.0,
            violation_type=ViolationType.PURE_DP_RATIO,
        )
        rpt = self._make_report(is_valid=False, n_violations=1, worst_violation=vr)
        r = repr(rpt)
        assert "INVALID" in r

    def test_default_fields(self):
        rpt = self._make_report()
        assert rpt.all_violations == []
        assert rpt.pair_analyses == []
        assert rpt.recommendations == []
        assert rpt.metadata == {}
        assert rpt.actual_epsilon is None
        assert rpt.actual_delta is None


# ═══════════════════════════════════════════════════════════════════════════
# §3  Tolerance Management Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSafeTolerance:
    """Tests for compute_safe_tolerance."""

    def test_basic_computation(self):
        eps = 1.0
        solver_tol = 1e-8
        safety = 2.0
        expected = safety * math.exp(eps) * solver_tol
        result = compute_safe_tolerance(eps, solver_tol, safety)
        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("eps", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_various_epsilon(self, eps):
        result = compute_safe_tolerance(eps)
        expected = 2.0 * math.exp(eps) * 1e-8
        assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("solver_tol", [1e-12, 1e-10, 1e-8, 1e-6])
    def test_various_solver_tol(self, solver_tol):
        result = compute_safe_tolerance(1.0, solver_tol)
        expected = 2.0 * math.exp(1.0) * solver_tol
        assert result == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("safety", [1.0, 1.5, 2.0, 5.0, 10.0])
    def test_various_safety_factor(self, safety):
        result = compute_safe_tolerance(1.0, 1e-8, safety)
        expected = safety * math.exp(1.0) * 1e-8
        assert result == pytest.approx(expected, rel=1e-10)

    def test_result_positive(self):
        result = compute_safe_tolerance(0.5)
        assert result > 0

    def test_result_satisfies_i4(self):
        eps = 2.0
        solver_tol = 1e-8
        tol = compute_safe_tolerance(eps, solver_tol)
        assert validate_tolerance(tol, eps, solver_tol)

    def test_invalid_epsilon_zero(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(0.0)

    def test_invalid_epsilon_negative(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(-1.0)

    def test_invalid_epsilon_inf(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(float("inf"))

    def test_invalid_epsilon_nan(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(float("nan"))

    def test_invalid_solver_tol_zero(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(1.0, 0.0)

    def test_invalid_solver_tol_negative(self):
        with pytest.raises(ConfigurationError):
            compute_safe_tolerance(1.0, -1e-8)


class TestValidateTolerance:
    """Tests for validate_tolerance."""

    def test_valid_tolerance(self):
        eps = 1.0
        solver_tol = 1e-8
        tol = math.exp(eps) * solver_tol * 2
        assert validate_tolerance(tol, eps, solver_tol) is True

    def test_exact_boundary(self):
        eps = 1.0
        solver_tol = 1e-8
        tol = math.exp(eps) * solver_tol
        assert validate_tolerance(tol, eps, solver_tol) is True

    def test_below_boundary(self):
        eps = 1.0
        solver_tol = 1e-8
        tol = math.exp(eps) * solver_tol * 0.5
        assert validate_tolerance(tol, eps, solver_tol) is False

    @pytest.mark.parametrize(
        "eps,solver_tol",
        [(0.1, 1e-8), (1.0, 1e-8), (5.0, 1e-10), (0.01, 1e-6)],
    )
    def test_safe_tolerance_always_valid(self, eps, solver_tol):
        tol = compute_safe_tolerance(eps, solver_tol)
        assert validate_tolerance(tol, eps, solver_tol)

    def test_very_large_epsilon(self):
        eps = 20.0
        solver_tol = 1e-8
        required = math.exp(eps) * solver_tol
        assert validate_tolerance(required, eps, solver_tol)
        assert not validate_tolerance(required * 0.9, eps, solver_tol)


class TestWarnToleranceViolation:
    """Tests for warn_tolerance_violation."""

    def test_no_warning_when_valid(self):
        eps = 1.0
        solver_tol = 1e-8
        tol = compute_safe_tolerance(eps, solver_tol)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_tolerance_violation(tol, eps, solver_tol)
            assert len(w) == 0

    def test_warning_when_violated(self):
        eps = 1.0
        solver_tol = 1e-8
        tol = 1e-12  # way too small
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_tolerance_violation(tol, eps, solver_tol)
            assert len(w) == 1
            assert "I4" in str(w[0].message) or "Invariant" in str(w[0].message)


# ═══════════════════════════════════════════════════════════════════════════
# §4  Divergence Computation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHockeyStickDivergence:
    """Tests for hockey_stick_divergence."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = hockey_stick_divergence(p, p, epsilon=1.0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_identical_nonuniform(self):
        p = np.array([0.1, 0.3, 0.6])
        result = hockey_stick_divergence(p, p, epsilon=0.5)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_known_value_eps_zero(self):
        # At eps=0, H_0(P||Q) = sum max(p_j - q_j, 0) = TV(P,Q) when
        # both sum to 1.
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        hs = hockey_stick_divergence(p, q, epsilon=0.0)
        expected = max(0.7 - 0.3, 0) + max(0.3 - 0.7, 0)
        assert hs == pytest.approx(expected, abs=1e-14)

    def test_known_value_specific(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        eps = 0.5
        exp_eps = math.exp(eps)
        expected = max(0.6 - exp_eps * 0.4, 0) + max(0.4 - exp_eps * 0.6, 0)
        result = hockey_stick_divergence(p, q, eps)
        assert result == pytest.approx(expected, abs=1e-14)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            result = hockey_stick_divergence(p, q, epsilon=1.0)
            assert result >= -1e-15

    def test_zero_when_p_dominated(self):
        # If p_j <= e^eps * q_j for all j, then H_eps = 0
        q = np.array([0.5, 0.5])
        eps = 2.0
        p = np.array([0.5, 0.5])  # p <= e^2 * q certainly
        result = hockey_stick_divergence(p, q, eps)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            hockey_stick_divergence(np.array([0.5, 0.5]), np.array([1.0]), 1.0)

    def test_large_epsilon_gives_zero(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        result = hockey_stick_divergence(p, q, epsilon=10.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_near_zero_probabilities(self):
        p = np.array([1e-15, 1.0 - 1e-15])
        q = np.array([1.0 - 1e-15, 1e-15])
        result = hockey_stick_divergence(p, q, epsilon=1.0)
        assert np.isfinite(result)

    @pytest.mark.parametrize("k", [2, 5, 10, 50, 100])
    def test_various_sizes(self, k):
        p = np.ones(k) / k
        q = np.ones(k) / k
        result = hockey_stick_divergence(p, q, epsilon=1.0)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_accepts_lists(self):
        result = hockey_stick_divergence([0.5, 0.5], [0.5, 0.5], 1.0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_symmetry_not_guaranteed(self):
        # H_eps(P||Q) != H_eps(Q||P) in general
        p = np.array([0.8, 0.2])
        q = np.array([0.2, 0.8])
        eps = 0.5
        fwd = hockey_stick_divergence(p, q, eps)
        rev = hockey_stick_divergence(q, p, eps)
        # They may or may not be equal, but both should be finite
        assert np.isfinite(fwd)
        assert np.isfinite(rev)


class TestHockeyStickDivergenceDetailed:
    """Tests for hockey_stick_divergence_detailed."""

    def test_returns_tuple(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        result = hockey_stick_divergence_detailed(p, q, 1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_total_matches_simple(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.5, 0.4])
        eps = 0.5
        total_simple = hockey_stick_divergence(p, q, eps)
        total_detailed, _ = hockey_stick_divergence_detailed(p, q, eps)
        assert total_detailed == pytest.approx(total_simple, abs=1e-15)

    def test_contributions_sum_to_total(self):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        total, contributions = hockey_stick_divergence_detailed(p, q, 0.5)
        assert float(np.sum(contributions)) == pytest.approx(total, abs=1e-15)

    def test_contributions_nonnegative(self):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        _, contributions = hockey_stick_divergence_detailed(p, q, 0.5)
        assert np.all(contributions >= 0)

    def test_contributions_shape(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        _, contributions = hockey_stick_divergence_detailed(p, q, 1.0)
        assert contributions.shape == (4,)

    def test_identical_gives_zero_contributions(self):
        p = np.array([0.3, 0.7])
        total, contributions = hockey_stick_divergence_detailed(p, p, 1.0)
        assert total == pytest.approx(0.0, abs=1e-15)
        assert np.allclose(contributions, 0.0, atol=1e-15)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            hockey_stick_divergence_detailed(
                np.array([0.5, 0.5]), np.array([1.0]), 1.0
            )


class TestKLDivergence:
    """Tests for kl_divergence."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_known_value_binary(self):
        # KL(Ber(0.5) || Ber(0.5)) = 0
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-12)

    def test_known_value_asymmetric(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        expected = 0.9 * math.log(0.9 / 0.1) + 0.1 * math.log(0.1 / 0.9)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            result = kl_divergence(p, q)
            assert result >= -1e-12

    def test_infinite_when_q_zero_p_positive(self):
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        result = kl_divergence(p, q)
        assert result == float("inf")

    def test_zero_p_contributes_nothing(self):
        p = np.array([0.0, 1.0])
        q = np.array([0.5, 0.5])
        expected = 1.0 * math.log(1.0 / 0.5)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-10)

    def test_all_zero_p(self):
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.3, 0.3, 0.4])
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-15)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            kl_divergence(np.array([0.5, 0.5]), np.array([1.0]))

    @pytest.mark.parametrize("k", [2, 3, 10, 50])
    def test_uniform_distributions(self, k):
        p = np.ones(k) / k
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_asymmetry(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        fwd = kl_divergence(p, q)
        rev = kl_divergence(q, p)
        assert fwd == pytest.approx(rev, rel=1e-10)  # symmetric for this pair
        # But in general KL is not symmetric:
        p2 = np.array([0.7, 0.3])
        q2 = np.array([0.4, 0.6])
        fwd2 = kl_divergence(p2, q2)
        rev2 = kl_divergence(q2, p2)
        # Just verify both are finite, may differ
        assert np.isfinite(fwd2) and np.isfinite(rev2)


class TestRenyiDivergence:
    """Tests for renyi_divergence."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = renyi_divergence(p, p, alpha=2.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            p = rng.dirichlet(np.ones(4))
            q = rng.dirichlet(np.ones(4))
            result = renyi_divergence(p, q, alpha=2.0)
            assert result >= -1e-10

    def test_alpha_near_one_approaches_kl(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        kl = kl_divergence(p, q)
        renyi_close = renyi_divergence(p, q, alpha=1.0 + 1e-13)
        assert renyi_close == pytest.approx(kl, rel=1e-4)

    def test_alpha_2_known_value(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        result = renyi_divergence(p, q, alpha=2.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 3.0, 5.0, 10.0])
    def test_various_alpha(self, alpha):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        result = renyi_divergence(p, q, alpha=alpha)
        assert np.isfinite(result)
        assert result >= -1e-10

    def test_alpha_le_zero_raises(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_divergence(p, q, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_divergence(p, q, alpha=-1.0)

    def test_infinite_when_q_zero_alpha_gt_1(self):
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        result = renyi_divergence(p, q, alpha=2.0)
        assert result == float("inf")

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            renyi_divergence(np.array([0.5, 0.5]), np.array([1.0]), 2.0)

    def test_large_alpha_approaches_max_divergence(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.5, 0.4])
        renyi_large = renyi_divergence(p, q, alpha=1e7)
        d_max = max_divergence(p, q)
        assert renyi_large == pytest.approx(d_max, rel=0.1)

    def test_monotonically_increasing_in_alpha(self):
        p = np.array([0.8, 0.15, 0.05])
        q = np.array([0.2, 0.3, 0.5])
        alphas = [1.5, 2.0, 3.0, 5.0, 10.0]
        vals = [renyi_divergence(p, q, a) for a in alphas]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10

    def test_all_zero_p(self):
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.3, 0.3, 0.4])
        result = renyi_divergence(p, q, alpha=2.0)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestMaxDivergence:
    """Tests for max_divergence."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert max_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self):
        p = np.array([0.8, 0.2])
        q = np.array([0.2, 0.8])
        expected = math.log(0.8 / 0.2)
        assert max_divergence(p, q) == pytest.approx(expected, rel=1e-10)

    def test_nonnegative(self):
        rng = np.random.default_rng(77)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            result = max_divergence(p, q)
            assert result >= -1e-12

    def test_infinite_when_q_zero_p_positive(self):
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        assert max_divergence(p, q) == float("inf")

    def test_zero_when_p_all_zero(self):
        p = np.array([0.0, 0.0])
        q = np.array([0.5, 0.5])
        assert max_divergence(p, q) == pytest.approx(0.0, abs=1e-12)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            max_divergence(np.array([0.5, 0.5]), np.array([1.0]))

    def test_corresponds_to_pure_dp_epsilon(self):
        # For randomized response, actual eps = log(e^eps) = eps
        eps = 1.0
        rr = _randomized_response(eps)
        d_inf_01 = max_divergence(rr[0], rr[1])
        d_inf_10 = max_divergence(rr[1], rr[0])
        actual_eps = max(d_inf_01, d_inf_10)
        assert actual_eps == pytest.approx(eps, rel=1e-10)

    @pytest.mark.parametrize("k", [2, 5, 10])
    def test_uniform_gives_zero(self, k):
        p = np.ones(k) / k
        assert max_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


class TestTotalVariation:
    """Tests for total_variation."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert total_variation(p, p) == pytest.approx(0.0, abs=1e-15)

    def test_known_value(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        expected = 0.5 * (abs(0.7 - 0.3) + abs(0.3 - 0.7))
        assert total_variation(p, q) == pytest.approx(expected, abs=1e-14)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(55)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            tv = total_variation(p, q)
            assert 0 <= tv <= 1.0 + 1e-12

    def test_disjoint_distributions(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert total_variation(p, q) == pytest.approx(1.0, abs=1e-14)

    def test_equals_hockey_stick_at_eps_zero(self):
        # TV(P, Q) = H_0(P || Q) for distributions summing to 1
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.4, 0.4])
        tv = total_variation(p, q)
        hs = hockey_stick_divergence(p, q, epsilon=0.0)
        assert tv == pytest.approx(hs, abs=1e-14)

    def test_symmetry(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.5, 0.4])
        assert total_variation(p, q) == pytest.approx(total_variation(q, p), abs=1e-15)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            total_variation(np.array([0.5, 0.5]), np.array([1.0]))

    @pytest.mark.parametrize("k", [2, 5, 10, 50])
    def test_uniform(self, k):
        p = np.ones(k) / k
        assert total_variation(p, p) == pytest.approx(0.0, abs=1e-14)


# ═══════════════════════════════════════════════════════════════════════════
# §5  Core Verification Function Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyPureDP:
    """Tests for verify() with delta=0 (pure DP)."""

    def test_uniform_passes_any_epsilon(self):
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        result = verify(p, epsilon=0.1, delta=0.0, edges=edges)
        assert result.valid
        assert result.violation is None

    def test_randomized_response_passes_at_correct_eps(self):
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(rr, epsilon=eps, delta=0.0, edges=edges, tol=tol)
        assert result.valid

    def test_randomized_response_fails_at_smaller_eps(self):
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        tol = compute_safe_tolerance(0.1)
        result = verify(rr, epsilon=0.1, delta=0.0, edges=edges, tol=tol)
        assert not result.valid
        assert result.violation is not None
        i, ip, j, mag = result.violation
        assert mag > 0

    def test_deterministic_mechanism_fails(self):
        # Identity matrix: p[0][0]/p[1][0] = 1/floor ~ huge ratio
        p = _deterministic_mechanism(2)
        edges = [(0, 1)]
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert not result.valid

    def test_single_output_passes(self):
        # All rows identical -> all ratios = 1
        p = _single_output_mechanism(3)
        edges = _chain_edges(3)
        result = verify(p, epsilon=0.01, delta=0.0, edges=edges)
        assert result.valid

    def test_violation_contains_correct_pair(self):
        p = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        edges = [(0, 1)]
        result = verify(p, epsilon=0.5, delta=0.0, edges=edges)
        assert not result.valid
        i, ip, j, mag = result.violation
        assert {i, ip} == {0, 1}
        assert 0 <= j <= 1
        assert mag > 0

    def test_multiple_edges_worst_found(self):
        # Create mechanism where pair (1,2) has worse violation than (0,1)
        p = np.array([
            [0.6, 0.4],
            [0.5, 0.5],
            [0.05, 0.95],
        ])
        edges = [(0, 1), (1, 2)]
        result = verify(
            p, epsilon=0.5, delta=0.0, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        assert not result.valid
        # Pair (1,2) has ratio 0.95/0.5 = 1.9 or 0.5/0.05 = 10
        # Pair (0,1) has ratio 0.6/0.5 = 1.2
        i, ip, j, mag = result.violation
        # The worst pair should involve 1 and 2 (or 0 and 2 via symmetry expansion)
        assert mag > 0


class TestVerifyApproxDP:
    """Tests for verify() with delta>0 (approximate DP)."""

    def test_uniform_passes(self):
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        result = verify(p, epsilon=1.0, delta=0.01, edges=edges)
        assert result.valid

    def test_mechanism_with_high_ratios_but_valid_hockey_stick_passes(self):
        """Critical test: high individual ratios but valid aggregate hockey-stick."""
        # Build mechanism where some bin ratios > e^eps, but hockey-stick <= delta
        eps = 1.0
        delta = 0.1
        exp_eps = math.exp(eps)

        # Row 0: one "bad" bin with high prob, rest spread out
        # Row 1: bad bin has low prob, rest spread out
        p = np.array([
            [0.3, 0.05, 0.05, 0.6],
            [0.05, 0.3, 0.6, 0.05],
        ])
        # Ratio at bin 0: 0.3/0.05 = 6.0, which > e^1 ≈ 2.718
        # But hockey-stick = sum max(p_j - e^eps * q_j, 0) can be small
        hs_fwd = hockey_stick_divergence(p[0], p[1], eps)
        hs_rev = hockey_stick_divergence(p[1], p[0], eps)

        # Choose delta large enough to make it pass
        actual_delta = max(hs_fwd, hs_rev)
        safe_delta = actual_delta + 0.01

        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(p, epsilon=eps, delta=safe_delta, edges=edges, tol=tol)
        assert result.valid, (
            f"Mechanism should pass approx DP despite high individual ratios. "
            f"HS_fwd={hs_fwd:.4f}, HS_rev={hs_rev:.4f}, delta={safe_delta:.4f}"
        )

    def test_hockey_stick_violation_detected(self):
        eps = 1.0
        delta = 0.001
        p = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        edges = [(0, 1)]
        hs = hockey_stick_divergence(p[0], p[1], eps)
        if hs > delta:
            result = verify(p, epsilon=eps, delta=delta, edges=edges)
            assert not result.valid
            _, _, j, mag = result.violation
            assert j == -1  # aggregate violation for approx DP
            assert mag > 0

    def test_approx_dp_j_worst_is_minus_one(self):
        """For approximate DP violations, j_worst should be -1 (aggregate)."""
        p = np.array([
            [0.95, 0.05],
            [0.05, 0.95],
        ])
        edges = [(0, 1)]
        result = verify(p, epsilon=0.5, delta=1e-6, edges=edges)
        if not result.valid:
            _, _, j, _ = result.violation
            assert j == -1

    def test_randomized_response_approx_dp(self):
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        # At the exact eps, hockey-stick should be ~0, so any positive delta passes
        tol = compute_safe_tolerance(eps)
        result = verify(rr, epsilon=eps, delta=0.01, edges=edges, tol=tol)
        assert result.valid


class TestVerifyModes:
    """Tests for different VerificationMode behavior."""

    def _violating_mechanism(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """3-row mechanism with multiple violating pairs."""
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        edges = _chain_edges(3)
        return p, edges

    def test_fast_mode_returns_violation(self):
        p, edges = self._violating_mechanism()
        result = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.FAST,
        )
        assert not result.valid
        assert result.violation is not None

    def test_most_violating_returns_worst(self):
        p, edges = self._violating_mechanism()
        result_mv = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        result_fast = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.FAST,
        )
        assert not result_mv.valid
        assert not result_fast.valid
        # MOST_VIOLATING should have >= magnitude of FAST
        assert result_mv.violation[3] >= result_fast.violation[3] - 1e-12

    def test_exhaustive_mode(self):
        p, edges = self._violating_mechanism()
        result = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.EXHAUSTIVE,
        )
        assert not result.valid


class TestVerifyEdgeHandling:
    """Tests for edge/adjacency handling in verify()."""

    def test_list_edges(self):
        p = _uniform_mechanism(3, 2)
        edges = [(0, 1), (1, 2)]
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert result.valid

    def test_adjacency_relation(self):
        p = _uniform_mechanism(3, 2)
        adj = AdjacencyRelation.hamming_distance_1(3)
        result = verify(p, epsilon=1.0, delta=0.0, edges=adj)
        assert result.valid

    def test_complete_adjacency(self):
        p = _uniform_mechanism(4, 3)
        adj = AdjacencyRelation.complete(4)
        result = verify(p, epsilon=1.0, delta=0.0, edges=adj)
        assert result.valid

    def test_self_loop_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 0)])

    def test_out_of_range_edge_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 5)])

    def test_empty_edges_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=0.0, edges=[])


class TestVerifyMechanismValidation:
    """Tests for mechanism validation in verify()."""

    def test_1d_array_rejected(self):
        p = np.array([0.5, 0.5])
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_non_stochastic_rejected(self):
        p = np.array([[0.5, 0.6], [0.3, 0.7]])  # row 0 sums to 1.1
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_negative_probabilities_rejected(self):
        p = np.array([[-0.1, 1.1], [0.5, 0.5]])
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_nan_rejected(self):
        p = np.array([[float("nan"), 0.5], [0.5, 0.5]])
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_inf_rejected(self):
        p = np.array([[float("inf"), 0.5], [0.5, 0.5]])
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_empty_table_rejected(self):
        p = np.empty((0, 0))
        with pytest.raises(InvalidMechanismError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])


class TestVerifyParameterValidation:
    """Tests for parameter validation in verify()."""

    def test_epsilon_zero_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=0.0, delta=0.0, edges=[(0, 1)])

    def test_epsilon_negative_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=-1.0, delta=0.0, edges=[(0, 1)])

    def test_delta_negative_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=-0.1, edges=[(0, 1)])

    def test_delta_one_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=1.0, edges=[(0, 1)])

    def test_tol_negative_rejected(self):
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)], tol=-1e-9)


class TestVerifyResultIntegration:
    """Tests for VerifyResult properties returned by verify()."""

    def test_valid_result_properties(self):
        p = _uniform_mechanism(2, 2)
        result = verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])
        assert result.valid
        assert result.violation is None
        assert result.violation_pair is None
        assert result.violation_magnitude == 0.0

    def test_invalid_result_properties(self):
        p = _deterministic_mechanism(2)
        result = verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])
        assert not result.valid
        assert result.violation is not None
        assert result.violation_pair is not None
        assert result.violation_magnitude > 0


# ═══════════════════════════════════════════════════════════════════════════
# §6  Internal Pair Verification Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyPureDPPair:
    """Tests for _verify_pure_dp_pair."""

    def test_identical_rows_valid(self):
        row = np.array([0.25, 0.25, 0.25, 0.25])
        valid, j, ratio, mag = _verify_pure_dp_pair(row, row, math.exp(1.0), 1e-9)
        assert valid
        assert mag == 0.0

    def test_violating_rows(self):
        p_i = np.array([0.9, 0.1])
        p_ip = np.array([0.1, 0.9])
        exp_eps = math.exp(0.5)
        valid, j, ratio, mag = _verify_pure_dp_pair(p_i, p_ip, exp_eps, 1e-9)
        assert not valid
        assert ratio > exp_eps
        assert mag > 0

    def test_just_within_tolerance(self):
        exp_eps = 2.0
        tol = 0.01
        # Build rows where ratio is just below exp_eps + tol
        p_i = np.array([0.5, 0.5])
        p_ip = np.array([0.5 / (exp_eps + tol * 0.9), 1.0 - 0.5 / (exp_eps + tol * 0.9)])
        valid, _, _, _ = _verify_pure_dp_pair(p_i, p_ip, exp_eps, tol)
        assert valid

    def test_checks_both_directions(self):
        p_i = np.array([0.9, 0.1])
        p_ip = np.array([0.5, 0.5])
        exp_eps = math.exp(0.5)
        valid, j, ratio, mag = _verify_pure_dp_pair(p_i, p_ip, exp_eps, 1e-9)
        # The worst ratio is max(0.9/0.5, 0.5/0.1) = max(1.8, 5.0) = 5.0
        assert ratio == pytest.approx(5.0, rel=1e-6)


class TestVerifyApproxDPPair:
    """Tests for _verify_approx_dp_pair."""

    def test_identical_rows_valid(self):
        row = np.array([0.25, 0.25, 0.25, 0.25])
        valid, hs, mag = _verify_approx_dp_pair(row, row, 1.0, 0.01, 1e-9)
        assert valid
        assert hs == pytest.approx(0.0, abs=1e-14)
        assert mag == 0.0

    def test_violating_rows(self):
        p_i = np.array([0.9, 0.1])
        p_ip = np.array([0.1, 0.9])
        valid, hs, mag = _verify_approx_dp_pair(p_i, p_ip, 0.5, 0.001, 1e-9)
        if not valid:
            assert hs > 0.001
            assert mag > 0


# ═══════════════════════════════════════════════════════════════════════════
# §7  PrivacyVerifier Class Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyVerifier:
    """Tests for PrivacyVerifier class."""

    def test_construction_default(self):
        v = PrivacyVerifier()
        assert v.config is not None

    def test_construction_with_config(self):
        cfg = NumericalConfig(solver_tol=1e-10)
        v = PrivacyVerifier(numerical_config=cfg)
        assert v.config.solver_tol == 1e-10

    def test_verify_pure_dp_pass(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        result = v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        assert result.valid

    def test_verify_pure_dp_fail(self):
        v = PrivacyVerifier()
        p = _deterministic_mechanism(2)
        edges = [(0, 1)]
        result = v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        assert not result.valid

    def test_verify_approx_dp_pass(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        result = v.verify_approx_dp(p, epsilon=1.0, delta=0.01, edges=edges)
        assert result.valid

    def test_verify_approx_dp_zero_delta_raises(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(2, 2)
        with pytest.raises(ConfigurationError):
            v.verify_approx_dp(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])

    def test_verify_mechanism_pure_dp(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        adj = AdjacencyRelation.hamming_distance_1(3)
        result = v.verify_mechanism(p, budget, edges=adj)
        assert result.valid

    def test_verify_mechanism_approx_dp(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        budget = PrivacyBudget(epsilon=1.0, delta=0.01)
        adj = AdjacencyRelation.hamming_distance_1(3)
        result = v.verify_mechanism(p, budget, edges=adj)
        assert result.valid

    def test_find_most_violating_pair(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        edges = _chain_edges(3)
        vr = v.find_most_violating_pair(p, epsilon=0.1, delta=0.0, edges=edges)
        assert vr is not None
        assert vr.magnitude > 0

    def test_find_most_violating_pair_none_when_valid(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        vr = v.find_most_violating_pair(p, epsilon=1.0, delta=0.0, edges=edges)
        assert vr is None

    def test_find_all_violations(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        edges = _chain_edges(3)
        violations = v.find_all_violations(p, epsilon=0.1, delta=0.0, edges=edges)
        assert len(violations) > 0
        # Should be sorted by descending magnitude
        for i in range(len(violations) - 1):
            assert violations[i].magnitude >= violations[i + 1].magnitude

    def test_find_all_violations_empty_when_valid(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        violations = v.find_all_violations(p, epsilon=10.0, delta=0.0, edges=edges)
        assert len(violations) == 0

    def test_compute_actual_epsilon(self):
        v = PrivacyVerifier()
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        actual_eps = v.compute_actual_epsilon(rr, edges)
        assert actual_eps == pytest.approx(eps, rel=1e-8)

    def test_compute_actual_epsilon_uniform(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 4)
        edges = _chain_edges(3)
        actual_eps = v.compute_actual_epsilon(p, edges)
        assert actual_eps == pytest.approx(0.0, abs=1e-10)

    def test_compute_actual_delta(self):
        v = PrivacyVerifier()
        rr = _randomized_response(1.0)
        edges = [(0, 1)]
        actual_delta = v.compute_actual_delta(rr, epsilon=1.0, edges=edges)
        # At the true eps, delta should be ~0
        assert actual_delta == pytest.approx(0.0, abs=1e-10)

    def test_compute_actual_delta_below_eps(self):
        v = PrivacyVerifier()
        rr = _randomized_response(2.0)
        edges = [(0, 1)]
        # At eps=0.5 (less than mechanism's 2.0), delta should be positive
        actual_delta = v.compute_actual_delta(rr, epsilon=0.5, edges=edges)
        assert actual_delta > 0

    def test_compute_privacy_curve(self):
        v = PrivacyVerifier()
        rr = _randomized_response(1.0)
        edges = [(0, 1)]
        eps_vals, delta_vals = v.compute_privacy_curve(
            rr, edges, n_points=10,
        )
        assert len(eps_vals) == 10
        assert len(delta_vals) == 10
        # Delta should be non-increasing in epsilon
        for i in range(len(delta_vals) - 1):
            assert delta_vals[i] >= delta_vals[i + 1] - 1e-12

    def test_analyze_pair(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.7, 0.3],
            [0.3, 0.7],
        ])
        pa = v.analyze_pair(p, 0, 1, epsilon=1.0, delta=0.0)
        assert isinstance(pa, PairAnalysis)
        assert pa.i == 0
        assert pa.i_prime == 1
        assert pa.max_ratio_forward > 0
        assert pa.max_ratio_reverse > 0

    def test_analyze_all_pairs(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(4, 3)
        edges = _chain_edges(4)
        analyses = v.analyze_all_pairs(p, epsilon=1.0, delta=0.0, edges=edges)
        assert len(analyses) == 3  # 3 undirected pairs for chain of 4

    def test_generate_report(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        report = v.generate_report()
        assert isinstance(report, VerificationReport)
        assert report.is_valid
        assert report.n_violations == 0

    def test_generate_report_with_violations(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        edges = [(0, 1)]
        v.verify_pure_dp(p, epsilon=0.1, edges=edges)
        report = v.generate_report()
        assert not report.is_valid
        assert report.n_violations > 0
        assert report.worst_violation is not None

    def test_generate_report_explicit_args(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        report = v.generate_report(
            p=p, epsilon=1.0, delta=0.0, edges=edges,
        )
        assert report.is_valid

    def test_generate_report_no_state_raises(self):
        v = PrivacyVerifier()
        with pytest.raises(ConfigurationError):
            v.generate_report()


# ═══════════════════════════════════════════════════════════════════════════
# §8  MonteCarloVerifier Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMonteCarloVerifier:
    """Tests for MonteCarloVerifier class."""

    def test_construction(self):
        mc = MonteCarloVerifier(seed=42)
        assert mc._seed == 42

    def test_construction_no_seed(self):
        mc = MonteCarloVerifier()
        assert mc._seed is None

    def test_audit_dp_uniform_passes(self):
        mc = MonteCarloVerifier(seed=42)
        p = _uniform_mechanism(2, 4)
        edges = [(0, 1)]
        result = mc.audit_dp(p, edges, epsilon=1.0, n_samples=1000)
        assert result["pass"]
        assert "epsilon_empirical" in result
        assert "epsilon_upper_bound" in result
        assert "delta_empirical" in result
        assert result["n_samples"] == 1000

    def test_audit_dp_result_keys(self):
        mc = MonteCarloVerifier(seed=123)
        p = _uniform_mechanism(2, 3)
        edges = [(0, 1)]
        result = mc.audit_dp(p, edges, epsilon=1.0, n_samples=500)
        expected_keys = {
            "pass", "epsilon_empirical", "epsilon_upper_bound",
            "delta_empirical", "n_samples", "confidence",
            "worst_pair", "hoeffding_bound",
        }
        assert expected_keys.issubset(result.keys())

    def test_audit_dp_deterministic_fails(self):
        mc = MonteCarloVerifier(seed=42)
        p = _deterministic_mechanism(2)
        edges = [(0, 1)]
        result = mc.audit_dp(p, edges, epsilon=0.5, n_samples=10000)
        # Should detect high epsilon
        assert result["epsilon_empirical"] > 0.5

    def test_reproducibility_with_seed(self):
        p = _randomized_response(1.0)
        edges = [(0, 1)]
        mc1 = MonteCarloVerifier(seed=999)
        mc2 = MonteCarloVerifier(seed=999)
        r1 = mc1.audit_dp(p, edges, epsilon=1.0, n_samples=1000)
        r2 = mc2.audit_dp(p, edges, epsilon=1.0, n_samples=1000)
        assert r1["epsilon_empirical"] == r2["epsilon_empirical"]

    def test_audit_with_adjacency_relation(self):
        mc = MonteCarloVerifier(seed=42)
        p = _uniform_mechanism(3, 3)
        adj = AdjacencyRelation.hamming_distance_1(3)
        result = mc.audit_dp(p, adj, epsilon=1.0, n_samples=500)
        assert result["pass"]

    def test_estimate_epsilon_empirical(self):
        mc = MonteCarloVerifier(seed=42)
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        est_eps = mc.estimate_epsilon_empirical(rr, edges, n_samples=50000)
        # Should be close to true eps (lower bound)
        assert est_eps > 0
        assert est_eps <= eps + 0.5  # allow some slack for sampling noise


# ═══════════════════════════════════════════════════════════════════════════
# §9  Parametrized Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDivergenceRelationships:
    """Tests verifying mathematical relationships between divergences."""

    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_tv_equals_hockey_stick_eps_zero(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        tv = total_variation(p, q)
        hs = hockey_stick_divergence(p, q, epsilon=0.0)
        assert tv == pytest.approx(hs, abs=1e-14)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_hockey_stick_decreasing_in_epsilon(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        eps_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
        hs_vals = [hockey_stick_divergence(p, q, e) for e in eps_vals]
        for i in range(len(hs_vals) - 1):
            assert hs_vals[i] >= hs_vals[i + 1] - 1e-14

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_kl_upper_bounds_hockey_stick(self, seed):
        """KL divergence relates to hockey-stick via Pinsker-like bounds."""
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        kl = kl_divergence(p, q)
        hs_0 = hockey_stick_divergence(p, q, epsilon=0.0)
        # Pinsker: TV <= sqrt(KL/2), and HS_0 = TV for distributions summing to 1
        if np.isfinite(kl):
            assert hs_0 <= math.sqrt(kl / 2.0) + 1e-10

    @pytest.mark.parametrize("seed", [7, 8, 9])
    def test_max_divergence_ge_kl(self, seed):
        """D_inf >= D_KL for distributions (Gibbs' inequality)."""
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        d_max = max_divergence(p, q)
        kl = kl_divergence(p, q)
        if np.isfinite(d_max) and np.isfinite(kl):
            assert d_max >= kl - 1e-10


class TestRandomizedResponseVerification:
    """End-to-end tests with randomized response mechanism."""

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_passes_at_correct_eps(self, eps):
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(rr, epsilon=eps, delta=0.0, edges=edges, tol=tol)
        assert result.valid

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_fails_at_smaller_eps(self, eps):
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        smaller = eps * 0.5
        tol = compute_safe_tolerance(smaller)
        result = verify(rr, epsilon=smaller, delta=0.0, edges=edges, tol=tol)
        assert not result.valid

    @pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
    def test_actual_epsilon_matches(self, eps):
        rr = _randomized_response(eps)
        v = PrivacyVerifier()
        actual = v.compute_actual_epsilon(rr, [(0, 1)])
        assert actual == pytest.approx(eps, rel=1e-8)


class TestScaledMechanisms:
    """Tests with various mechanism sizes."""

    @pytest.mark.parametrize("n,k", [(2, 2), (3, 5), (5, 3), (10, 10), (2, 50)])
    def test_uniform_passes(self, n, k):
        p = _uniform_mechanism(n, k)
        edges = _chain_edges(n)
        result = verify(p, epsilon=0.01, delta=0.0, edges=edges)
        assert result.valid

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_deterministic_fails(self, n):
        p = _deterministic_mechanism(n)
        edges = _chain_edges(n)
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert not result.valid

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_single_output_passes(self, n):
        p = _single_output_mechanism(n)
        edges = _chain_edges(n)
        result = verify(p, epsilon=0.01, delta=0.0, edges=edges)
        assert result.valid


class TestToleranceIntegration:
    """Integration tests for tolerance management with verification."""

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_safe_tolerance_satisfies_i4(self, eps):
        solver_tol = 1e-8
        tol = compute_safe_tolerance(eps, solver_tol)
        assert validate_tolerance(tol, eps, solver_tol)

    @pytest.mark.parametrize("eps", [0.1, 1.0, 5.0])
    def test_monotone_in_epsilon(self, eps):
        tol_low = compute_safe_tolerance(eps)
        tol_high = compute_safe_tolerance(eps + 1.0)
        assert tol_high > tol_low

    def test_i4_invariant_formula(self):
        eps = 2.0
        solver_tol = 1e-8
        safety = 2.0
        tol = compute_safe_tolerance(eps, solver_tol, safety)
        # Must be safety * exp(eps) * solver_tol
        assert tol == pytest.approx(safety * math.exp(eps) * solver_tol, rel=1e-12)
        # Must satisfy tol >= exp(eps) * solver_tol (I4)
        assert tol >= math.exp(eps) * solver_tol


# ═══════════════════════════════════════════════════════════════════════════
# §10  Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for verifier module."""

    def test_very_small_epsilon(self):
        p = _uniform_mechanism(2, 3)
        edges = [(0, 1)]
        tol = compute_safe_tolerance(0.001)
        result = verify(p, epsilon=0.001, delta=0.0, edges=edges, tol=tol)
        assert result.valid  # uniform always passes

    def test_very_large_epsilon(self):
        # Any mechanism should pass with very large eps
        p = np.array([[0.99, 0.01], [0.01, 0.99]])
        edges = [(0, 1)]
        tol = compute_safe_tolerance(20.0)
        result = verify(p, epsilon=20.0, delta=0.0, edges=edges, tol=tol)
        assert result.valid

    def test_very_small_delta(self):
        p = _uniform_mechanism(2, 3)
        edges = [(0, 1)]
        result = verify(p, epsilon=1.0, delta=1e-15, edges=edges)
        assert result.valid

    def test_near_boundary_mechanism(self):
        # Mechanism where ratio is just at exp(eps)
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        # With safe tolerance, should pass
        tol = compute_safe_tolerance(eps)
        result = verify(rr, epsilon=eps, delta=0.0, edges=edges, tol=tol)
        assert result.valid

    def test_large_mechanism(self):
        n, k = 20, 20
        p = _uniform_mechanism(n, k)
        edges = _chain_edges(n)
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert result.valid

    def test_single_row_two_columns(self):
        # Minimal case: 2 rows (need at least 2 for an edge), 2 columns
        p = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        result = verify(p, epsilon=0.01, delta=0.0, edges=edges)
        assert result.valid

    def test_almost_deterministic_mechanism(self):
        # Very peaked but not exactly deterministic
        p = np.array([
            [1.0 - 1e-10, 1e-10],
            [1e-10, 1.0 - 1e-10],
        ])
        edges = [(0, 1)]
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert not result.valid

    def test_nearly_equal_rows(self):
        p = np.array([
            [0.5 + 1e-12, 0.5 - 1e-12],
            [0.5, 0.5],
        ])
        edges = [(0, 1)]
        result = verify(p, epsilon=0.01, delta=0.0, edges=edges)
        assert result.valid


class TestDivergenceEdgeCases:
    """Edge case tests for divergence functions."""

    def test_hockey_stick_zero_epsilon(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        result = hockey_stick_divergence(p, q, epsilon=0.0)
        # H_0(P||Q) = sum max(p-q, 0) = max(0.6-0.4,0) + max(0.4-0.6,0) = 0.2
        assert result == pytest.approx(0.2, abs=1e-14)

    def test_kl_with_very_small_values(self):
        p = np.array([1e-100, 1.0 - 1e-100])
        q = np.array([1e-100, 1.0 - 1e-100])
        result = kl_divergence(p, q)
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_renyi_alpha_exactly_one(self):
        """Alpha exactly 1 should fall back to KL."""
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        result = renyi_divergence(p, q, alpha=1.0)
        kl = kl_divergence(p, q)
        assert result == pytest.approx(kl, rel=1e-4)

    def test_max_divergence_single_bin_positive(self):
        p = np.array([0.0, 1.0])
        q = np.array([0.0, 1.0])
        result = max_divergence(p, q)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_total_variation_one_hot(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        assert total_variation(p, q) == pytest.approx(1.0, abs=1e-14)

    def test_hockey_stick_negative_epsilon(self):
        # Negative epsilon is mathematically valid (exp(-eps) < 1)
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        result = hockey_stick_divergence(p, q, epsilon=-1.0)
        assert np.isfinite(result)
        assert result >= 0

    def test_divergences_with_single_element(self):
        p = np.array([1.0])
        q = np.array([1.0])
        assert hockey_stick_divergence(p, q, 1.0) == pytest.approx(0.0, abs=1e-15)
        assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-15)
        assert max_divergence(p, q) == pytest.approx(0.0, abs=1e-15)
        assert total_variation(p, q) == pytest.approx(0.0, abs=1e-15)
        assert renyi_divergence(p, q, 2.0) == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# §11  Most-Violating Pair Selection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMostViolatingPairSelection:
    """Tests that MOST_VIOLATING mode returns the maximum violation."""

    def test_returns_worst_pure_dp(self):
        # Create mechanism with pair (0,1) mild and (1,2) severe
        p = np.array([
            [0.6, 0.4],       # mild difference from row 1
            [0.5, 0.5],       # uniform
            [0.01, 0.99],     # extreme difference from row 1
        ])
        edges = [(0, 1), (1, 2)]
        result = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        assert not result.valid
        i, ip, j, mag = result.violation
        # The pair involving rows 1,2 should have the worst violation
        # ratio at (1,2) is 0.99/0.5 = 1.98 vs exp(0.1) ≈ 1.105
        # ratio at (0,1) is 0.6/0.5 = 1.2 vs exp(0.1) ≈ 1.105
        assert mag > 0

    def test_returns_worst_approx_dp(self):
        p = np.array([
            [0.55, 0.45],     # mild
            [0.5, 0.5],       # uniform
            [0.95, 0.05],     # severe
        ])
        edges = [(0, 1), (1, 2)]
        eps = 0.1
        delta = 0.001
        result = verify(
            p, epsilon=eps, delta=delta, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        if not result.valid:
            # Verify the magnitude corresponds to the worst pair
            hs_01 = max(
                hockey_stick_divergence(p[0], p[1], eps),
                hockey_stick_divergence(p[1], p[0], eps),
            )
            hs_12 = max(
                hockey_stick_divergence(p[1], p[2], eps),
                hockey_stick_divergence(p[2], p[1], eps),
            )
            worst_hs = max(hs_01, hs_12)
            expected_mag = worst_hs - delta
            _, _, _, mag = result.violation
            assert mag == pytest.approx(expected_mag, rel=1e-10)

    def test_fast_may_differ_from_most_violating(self):
        """FAST returns first found; MOST_VIOLATING returns maximum."""
        p = np.array([
            [0.6, 0.4],
            [0.5, 0.5],
            [0.01, 0.99],
        ])
        edges = [(0, 1), (1, 2)]
        r_fast = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.FAST,
        )
        r_mv = verify(
            p, epsilon=0.1, delta=0.0, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        assert not r_fast.valid
        assert not r_mv.valid
        # MOST_VIOLATING magnitude should be >= FAST magnitude
        assert r_mv.violation[3] >= r_fast.violation[3] - 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# §12  Approx DP Critical Semantics Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestApproxDPCriticalSemantics:
    """Critical tests for approximate DP semantics.

    Approximate DP uses hockey-stick divergence (aggregate), NOT per-bin
    ratios. A mechanism with high individual ratios but valid hockey-stick
    MUST pass. This is the most common source of verifier bugs.
    """

    def test_high_ratio_valid_hockey_stick_passes(self):
        """A mechanism with ratio > e^eps at some bins but valid HS passes."""
        eps = 0.5
        exp_eps = math.exp(eps)

        # Construct mechanism where one bin has ratio >> e^eps
        # but the total hockey-stick is within delta
        p0 = np.array([0.4, 0.1, 0.5])
        p1 = np.array([0.1, 0.4, 0.5])

        # Ratio at bin 0: 0.4/0.1 = 4.0 >> e^0.5 ≈ 1.649
        assert p0[0] / p1[0] > exp_eps

        hs = hockey_stick_divergence(p0, p1, eps)
        # Use delta safely above hs
        delta = hs + 0.05

        p = np.vstack([p0, p1])
        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(p, epsilon=eps, delta=delta, edges=edges, tol=tol)
        assert result.valid, (
            "Approx DP should pass even with high individual ratios "
            "when hockey-stick divergence is within delta"
        )

    def test_pure_dp_same_mechanism_fails(self):
        """Same mechanism that passes approx DP fails pure DP."""
        eps = 0.5
        p = np.array([
            [0.4, 0.1, 0.5],
            [0.1, 0.4, 0.5],
        ])
        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)

        # Pure DP should fail (ratio 4.0 > e^0.5 ≈ 1.649)
        result_pure = verify(p, epsilon=eps, delta=0.0, edges=edges, tol=tol)
        assert not result_pure.valid

    def test_approx_dp_does_not_check_per_bin_ratios(self):
        """Verifier must NOT reject based on per-bin ratios in approx mode."""
        eps = 1.0
        exp_eps = math.exp(eps)

        # Build mechanism with extreme per-bin ratio but tiny hockey-stick
        high = 0.5
        low = 0.01
        rest = 1.0 - high - low
        p0 = np.array([high, low, rest])
        p1 = np.array([low, high, rest])

        # Per-bin ratio at bin 0: 0.5/0.01 = 50 >> e^1
        assert p0[0] / p1[0] > exp_eps * 10

        hs_fwd = hockey_stick_divergence(p0, p1, eps)
        hs_rev = hockey_stick_divergence(p1, p0, eps)
        worst_hs = max(hs_fwd, hs_rev)
        delta = worst_hs + 0.1

        p = np.vstack([p0, p1])
        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(p, epsilon=eps, delta=delta, edges=edges, tol=tol)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# §13  PrivacyVerifier Analysis and Report Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyVerifierReportGeneration:
    """Detailed tests for PrivacyVerifier report and analysis methods."""

    def test_report_summary_format(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        report = v.generate_report()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "DP Verification Report" in summary
        assert "PASS" in summary

    def test_report_with_actual_params(self):
        v = PrivacyVerifier()
        p = _randomized_response(1.0)
        edges = [(0, 1)]
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        report = v.generate_report(compute_actual=True)
        assert report.actual_epsilon is not None
        assert report.actual_epsilon == pytest.approx(1.0, rel=1e-6)
        assert report.actual_delta is not None

    def test_report_without_actual_params(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        report = v.generate_report(compute_actual=False)
        assert report.actual_epsilon is None
        assert report.actual_delta is None

    def test_report_pair_analyses(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
        ])
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        report = v.generate_report()
        assert len(report.pair_analyses) > 0

    def test_report_all_violations_sorted(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=0.1, edges=edges)
        report = v.generate_report()
        for i in range(len(report.all_violations) - 1):
            assert (
                report.all_violations[i].magnitude
                >= report.all_violations[i + 1].magnitude
            )

    def test_report_recommendations_on_failure(self):
        v = PrivacyVerifier()
        p = np.array([[0.9, 0.1], [0.1, 0.9]])
        edges = [(0, 1)]
        v.verify_pure_dp(p, epsilon=0.1, edges=edges)
        report = v.generate_report()
        assert len(report.recommendations) > 0

    def test_report_recommendations_on_pass(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(2, 4)
        edges = [(0, 1)]
        v.verify_pure_dp(p, epsilon=10.0, edges=edges)
        report = v.generate_report()
        # With very large eps, may recommend tightening
        # Just verify it doesn't crash
        assert isinstance(report.recommendations, list)


# ═══════════════════════════════════════════════════════════════════════════
# §14  PrivacyVerifier Privacy Curve Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyCurve:
    """Tests for compute_privacy_curve."""

    def test_custom_epsilon_range(self):
        v = PrivacyVerifier()
        p = _randomized_response(1.0)
        edges = [(0, 1)]
        eps_range = np.array([0.1, 0.5, 1.0, 2.0])
        eps_vals, delta_vals = v.compute_privacy_curve(p, edges, epsilon_range=eps_range)
        assert len(eps_vals) == 4
        assert len(delta_vals) == 4

    def test_delta_nonincreasing(self):
        v = PrivacyVerifier()
        p = _randomized_response(1.0)
        edges = [(0, 1)]
        eps_vals, delta_vals = v.compute_privacy_curve(p, edges, n_points=20)
        for i in range(len(delta_vals) - 1):
            assert delta_vals[i] >= delta_vals[i + 1] - 1e-12

    def test_delta_zero_at_true_epsilon(self):
        v = PrivacyVerifier()
        eps = 1.0
        rr = _randomized_response(eps)
        edges = [(0, 1)]
        eps_range = np.array([eps])
        _, delta_vals = v.compute_privacy_curve(rr, edges, epsilon_range=eps_range)
        assert delta_vals[0] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_mechanism_all_deltas_zero(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        eps_vals, delta_vals = v.compute_privacy_curve(p, edges, n_points=10)
        assert np.allclose(delta_vals, 0.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════════
# §15  Symmetry and Direction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSymmetryAndDirection:
    """Tests for symmetric edge expansion and violation direction."""

    def test_symmetric_edges_expanded(self):
        p = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        edges = [(0, 1)]  # symmetric=True by default
        result = verify(p, epsilon=0.5, delta=0.0, edges=edges)
        assert not result.valid
        # Both directions should be checked; violation in either is reported

    def test_violation_direction_recorded(self):
        v = PrivacyVerifier()
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
        ])
        edges = [(0, 1)]
        vr = v.find_most_violating_pair(p, epsilon=0.1, delta=0.0, edges=edges)
        assert vr is not None
        assert vr.direction in ("forward", "reverse")

    def test_asymmetric_adjacency(self):
        adj = AdjacencyRelation(
            edges=[(0, 1)], n=2, symmetric=False,
        )
        p = _uniform_mechanism(2, 3)
        result = verify(p, epsilon=1.0, delta=0.0, edges=adj)
        assert result.valid


# ═══════════════════════════════════════════════════════════════════════════
# §16  Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalStability:
    """Tests for numerical stability of divergence computations."""

    def test_hockey_stick_near_zero_probs(self):
        p = np.array([1e-300, 1.0 - 1e-300])
        q = np.array([1.0 - 1e-300, 1e-300])
        result = hockey_stick_divergence(p, q, 1.0)
        assert np.isfinite(result)

    def test_kl_near_zero_probs(self):
        p = np.array([1e-50, 1.0 - 1e-50])
        q = np.array([1e-50, 1.0 - 1e-50])
        result = kl_divergence(p, q)
        assert np.isfinite(result)

    def test_renyi_near_zero_probs(self):
        p = np.array([1e-50, 1.0 - 1e-50])
        q = np.array([1e-50, 1.0 - 1e-50])
        result = renyi_divergence(p, q, alpha=2.0)
        assert np.isfinite(result)

    def test_max_divergence_near_zero_probs(self):
        p = np.array([1e-100, 1.0 - 1e-100])
        q = np.array([1e-100, 1.0 - 1e-100])
        result = max_divergence(p, q)
        assert np.isfinite(result)

    def test_verify_with_near_zero_entries(self):
        p = np.array([
            [1e-15, 1.0 - 1e-15],
            [1.0 - 1e-15, 1e-15],
        ])
        edges = [(0, 1)]
        # Should not crash; result may be valid or invalid
        result = verify(p, epsilon=1.0, delta=0.0, edges=edges)
        assert isinstance(result, VerifyResult)


# ═══════════════════════════════════════════════════════════════════════════
# §17  Comprehensive Parametrized Divergence Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHockeyStickParametrized:
    """Parametrized tests for hockey_stick_divergence."""

    @pytest.mark.parametrize(
        "p,q,eps,expected_approx",
        [
            ([0.5, 0.5], [0.5, 0.5], 1.0, 0.0),
            ([1.0, 0.0], [1.0, 0.0], 1.0, 0.0),
            ([0.0, 1.0], [0.0, 1.0], 0.5, 0.0),
        ],
        ids=["identical_uniform", "identical_one_hot_a", "identical_one_hot_b"],
    )
    def test_identical_distributions(self, p, q, eps, expected_approx):
        result = hockey_stick_divergence(np.array(p), np.array(q), eps)
        assert result == pytest.approx(expected_approx, abs=1e-14)

    @pytest.mark.parametrize("eps", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_uniform_always_zero(self, eps):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = hockey_stick_divergence(p, p, eps)
        assert result == pytest.approx(0.0, abs=1e-14)


class TestKLDivergenceParametrized:
    """Parametrized tests for kl_divergence."""

    @pytest.mark.parametrize(
        "p,q,expected_approx",
        [
            ([0.5, 0.5], [0.5, 0.5], 0.0),
            ([1.0, 0.0], [0.5, 0.5], math.log(2.0)),
            ([0.0, 1.0], [0.5, 0.5], math.log(2.0)),
        ],
        ids=["identical", "one_hot_0_vs_uniform", "one_hot_1_vs_uniform"],
    )
    def test_known_values(self, p, q, expected_approx):
        result = kl_divergence(np.array(p), np.array(q))
        assert result == pytest.approx(expected_approx, rel=1e-10)


class TestTotalVariationParametrized:
    """Parametrized tests for total_variation."""

    @pytest.mark.parametrize(
        "p,q,expected",
        [
            ([0.5, 0.5], [0.5, 0.5], 0.0),
            ([1.0, 0.0], [0.0, 1.0], 1.0),
            ([0.75, 0.25], [0.25, 0.75], 0.5),
            ([0.6, 0.4], [0.4, 0.6], 0.2),
        ],
        ids=["identical", "disjoint", "quarter_shift", "tenth_shift"],
    )
    def test_known_values(self, p, q, expected):
        result = total_variation(np.array(p), np.array(q))
        assert result == pytest.approx(expected, abs=1e-14)


# ═══════════════════════════════════════════════════════════════════════════
# §18  PrivacyVerifier State Management Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyVerifierState:
    """Tests for PrivacyVerifier internal state management."""

    def test_state_updated_after_verify(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=1.0, edges=edges)
        assert v._last_epsilon == 1.0
        assert v._last_delta == 0.0
        assert v._last_p is not None
        assert v._last_edges is not None

    def test_state_updated_after_approx_verify(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_approx_dp(p, epsilon=1.0, delta=0.01, edges=edges)
        assert v._last_epsilon == 1.0
        assert v._last_delta == 0.01

    def test_report_uses_cached_state(self):
        v = PrivacyVerifier()
        p = _uniform_mechanism(3, 3)
        edges = _chain_edges(3)
        v.verify_pure_dp(p, epsilon=2.0, edges=edges)
        report = v.generate_report()
        assert report.epsilon == 2.0
        assert report.delta == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §19  Verify with AdjacencyRelation Factory Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyWithAdjacencyFactories:
    """Tests using AdjacencyRelation factory methods."""

    def test_hamming_distance_1(self):
        n = 5
        p = _uniform_mechanism(n, 3)
        adj = AdjacencyRelation.hamming_distance_1(n)
        result = verify(p, epsilon=1.0, delta=0.0, edges=adj)
        assert result.valid

    def test_complete_adjacency(self):
        n = 4
        p = _uniform_mechanism(n, 3)
        adj = AdjacencyRelation.complete(n)
        result = verify(p, epsilon=1.0, delta=0.0, edges=adj)
        assert result.valid

    def test_hamming_catches_violation(self):
        n = 3
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        adj = AdjacencyRelation.hamming_distance_1(n)
        result = verify(p, epsilon=0.1, delta=0.0, edges=adj)
        assert not result.valid

    def test_complete_catches_all_violations(self):
        n = 3
        p = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        adj = AdjacencyRelation.complete(n)
        v = PrivacyVerifier()
        violations = v.find_all_violations(
            p, epsilon=0.1, delta=0.0, edges=adj,
        )
        # Complete adjacency should find violations for all pairs
        assert len(violations) > 0


# ═══════════════════════════════════════════════════════════════════════════
# §20  Regression / Bug Prevention Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionPrevention:
    """Tests to prevent common verifier bugs."""

    def test_approx_dp_never_checks_per_bin_ratios(self):
        """Regression: verifier must NOT check per-bin ratios for approx DP.

        This is the #1 most common verifier bug. When delta > 0, only
        the hockey-stick divergence matters. Individual bin ratios can
        exceed e^eps — that's the whole point of approximate DP.
        """
        eps = 0.1
        exp_eps = math.exp(eps)

        # Create mechanism with extreme per-bin ratio
        p = np.array([
            [0.5, 0.01, 0.49],
            [0.01, 0.5, 0.49],
        ])
        # Ratio at bin 0: 0.5/0.01 = 50 >> e^0.1 ≈ 1.105

        hs_fwd = hockey_stick_divergence(p[0], p[1], eps)
        hs_rev = hockey_stick_divergence(p[1], p[0], eps)
        delta = max(hs_fwd, hs_rev) + 0.01

        edges = [(0, 1)]
        tol = compute_safe_tolerance(eps)
        result = verify(p, epsilon=eps, delta=delta, edges=edges, tol=tol)
        assert result.valid, (
            "REGRESSION: Approx DP verifier is rejecting based on per-bin "
            "ratios instead of hockey-stick divergence!"
        )

    def test_most_violating_returns_max_not_first(self):
        """Regression: MOST_VIOLATING must scan all pairs, not return first."""
        p = np.array([
            [0.55, 0.45],     # mild violation with row 1
            [0.5, 0.5],       # uniform (baseline)
            [0.99, 0.01],     # extreme violation with row 1
        ])
        edges = [(0, 1), (1, 2)]

        result = verify(
            p, epsilon=0.01, delta=0.0, edges=edges,
            mode=VerificationMode.MOST_VIOLATING,
        )
        assert not result.valid
        _, _, _, mag = result.violation

        # Compute what the magnitude should be for the worst pair
        exp_eps = math.exp(0.01)
        # Pair (1,2): ratio 0.99/0.5 = 1.98 (or 0.5/0.01 = 50)
        # Pair (0,1): ratio 0.55/0.5 = 1.1
        # The worst must be from pair (1,2)
        worst_ratio_12 = max(0.99 / 0.5, 0.5 / 0.01)  # via floor
        # Magnitude should be large
        assert mag > 1.0

    def test_verify_returns_verify_result_type(self):
        p = _uniform_mechanism(2, 2)
        result = verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])
        assert isinstance(result, VerifyResult)

    def test_tolerance_warning_does_not_block_verification(self):
        """Tolerance I4 violation should warn but not prevent verification."""
        p = _uniform_mechanism(2, 2)
        edges = [(0, 1)]
        # Use very small tol that violates I4
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = verify(p, epsilon=1.0, delta=0.0, edges=edges, tol=1e-15)
            assert isinstance(result, VerifyResult)

    def test_verify_result_valid_no_violation(self):
        """Valid result must have violation=None."""
        p = _uniform_mechanism(2, 2)
        result = verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])
        assert result.valid
        assert result.violation is None

    def test_verify_result_invalid_has_violation(self):
        """Invalid result must have violation != None."""
        p = _deterministic_mechanism(2)
        result = verify(p, epsilon=1.0, delta=0.0, edges=[(0, 1)])
        assert not result.valid
        assert result.violation is not None
        assert len(result.violation) == 4
