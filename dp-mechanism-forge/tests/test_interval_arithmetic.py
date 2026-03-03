"""
Tests for dp_forge.robust.interval_arithmetic — Interval arithmetic.

Covers:
    - Basic operations: add, subtract, multiply, divide
    - Monotone functions: exp, log (interval must contain true result)
    - Soundness: for random inputs, interval always contains exact result
    - Interval matrix-vector product: result interval contains true product
    - Comparison operators: certainly_less, possibly_equal
    - Special cases: zero-width intervals, negative intervals, division by
      interval containing zero
    - Property-based testing with hypothesis: arithmetic soundness
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.robust.interval_arithmetic import (
    Interval,
    IntervalMatrix,
    interval_verify_dp,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def unit_interval():
    return Interval(0.0, 1.0)


@pytest.fixture
def point_interval():
    return Interval(3.0, 3.0)


@pytest.fixture
def neg_interval():
    return Interval(-2.0, -1.0)


@pytest.fixture
def crossing_interval():
    return Interval(-1.0, 2.0)


# =========================================================================
# Construction and properties
# =========================================================================


class TestIntervalConstruction:
    """Tests for Interval construction and basic properties."""

    def test_exact(self):
        iv = Interval.exact(5.0)
        assert iv.lo == 5.0
        assert iv.hi == 5.0
        assert iv.is_point

    def test_from_value_and_error(self):
        iv = Interval.from_value_and_error(1.0, 0.1)
        assert iv.lo == pytest.approx(0.9)
        assert iv.hi == pytest.approx(1.1)

    def test_from_value_negative_error(self):
        with pytest.raises(ValueError, match="abs_error"):
            Interval.from_value_and_error(1.0, -0.1)

    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lower"):
            Interval(2.0, 0.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            Interval(float("nan"), 1.0)
        with pytest.raises(ValueError, match="NaN"):
            Interval(0.0, float("nan"))

    def test_mid(self, unit_interval):
        assert unit_interval.mid == pytest.approx(0.5)

    def test_width(self, unit_interval):
        assert unit_interval.width == pytest.approx(1.0)

    def test_width_point(self, point_interval):
        assert point_interval.width == 0.0

    def test_is_point(self, point_interval, unit_interval):
        assert point_interval.is_point
        assert not unit_interval.is_point

    def test_contains(self, unit_interval):
        assert unit_interval.contains(0.5)
        assert unit_interval.contains(0.0)
        assert unit_interval.contains(1.0)
        assert not unit_interval.contains(1.5)
        assert not unit_interval.contains(-0.1)

    def test_overlaps(self):
        a = Interval(0.0, 2.0)
        b = Interval(1.0, 3.0)
        c = Interval(3.0, 4.0)
        assert a.overlaps(b)
        assert not a.overlaps(c)

    def test_repr(self, unit_interval):
        r = repr(unit_interval)
        assert "Interval" in r

    def test_eq(self):
        assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
        assert Interval(1.0, 2.0) != Interval(1.0, 3.0)


# =========================================================================
# Arithmetic operations
# =========================================================================


class TestArithmeticOperations:
    """Tests for interval arithmetic operations."""

    # --- Addition ---
    def test_add_intervals(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a + b
        assert c.lo == pytest.approx(4.0)
        assert c.hi == pytest.approx(6.0)

    def test_add_scalar(self):
        a = Interval(1.0, 2.0)
        c = a + 5.0
        assert c.lo == pytest.approx(6.0)
        assert c.hi == pytest.approx(7.0)

    def test_radd_scalar(self):
        a = Interval(1.0, 2.0)
        c = 5.0 + a
        assert c.lo == pytest.approx(6.0)
        assert c.hi == pytest.approx(7.0)

    # --- Subtraction ---
    def test_sub_intervals(self):
        a = Interval(3.0, 5.0)
        b = Interval(1.0, 2.0)
        c = a - b
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(4.0)

    def test_sub_scalar(self):
        a = Interval(3.0, 5.0)
        c = a - 1.0
        assert c.lo == pytest.approx(2.0)
        assert c.hi == pytest.approx(4.0)

    def test_rsub_scalar(self):
        a = Interval(1.0, 2.0)
        c = 5.0 - a
        assert c.lo == pytest.approx(3.0)
        assert c.hi == pytest.approx(4.0)

    # --- Negation ---
    def test_neg(self):
        a = Interval(1.0, 3.0)
        c = -a
        assert c.lo == pytest.approx(-3.0)
        assert c.hi == pytest.approx(-1.0)

    # --- Multiplication ---
    def test_mul_both_positive(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a * b
        assert c.lo == pytest.approx(3.0)
        assert c.hi == pytest.approx(8.0)

    def test_mul_crossing_zero(self, crossing_interval):
        a = crossing_interval  # [-1, 2]
        b = Interval(3.0, 4.0)
        c = a * b
        assert c.lo == pytest.approx(-4.0)  # -1 * 4
        assert c.hi == pytest.approx(8.0)   # 2 * 4

    def test_mul_both_negative(self, neg_interval):
        a = neg_interval  # [-2, -1]
        b = Interval(-4.0, -3.0)
        c = a * b
        assert c.lo == pytest.approx(3.0)   # -1 * -3
        assert c.hi == pytest.approx(8.0)   # -2 * -4

    def test_mul_positive_scalar(self):
        a = Interval(1.0, 3.0)
        c = a * 2.0
        assert c.lo == pytest.approx(2.0)
        assert c.hi == pytest.approx(6.0)

    def test_mul_negative_scalar(self):
        a = Interval(1.0, 3.0)
        c = a * (-2.0)
        assert c.lo == pytest.approx(-6.0)
        assert c.hi == pytest.approx(-2.0)

    def test_rmul_scalar(self):
        a = Interval(1.0, 2.0)
        c = 3.0 * a
        assert c.lo == pytest.approx(3.0)
        assert c.hi == pytest.approx(6.0)

    # --- Division ---
    def test_div_positive(self):
        a = Interval(2.0, 6.0)
        b = Interval(1.0, 3.0)
        c = a / b
        assert c.lo == pytest.approx(2.0 / 3.0)
        assert c.hi == pytest.approx(6.0)

    def test_div_by_scalar(self):
        a = Interval(2.0, 6.0)
        c = a / 2.0
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(3.0)

    def test_div_by_negative_scalar(self):
        a = Interval(2.0, 6.0)
        c = a / (-2.0)
        assert c.lo == pytest.approx(-3.0)
        assert c.hi == pytest.approx(-1.0)

    def test_div_by_zero_scalar(self):
        a = Interval(1.0, 2.0)
        with pytest.raises(ZeroDivisionError):
            a / 0.0

    def test_div_by_interval_containing_zero(self):
        a = Interval(1.0, 2.0)
        b = Interval(-1.0, 1.0)
        with pytest.raises(ZeroDivisionError):
            a / b

    # --- Absolute value ---
    def test_abs_positive(self):
        a = Interval(1.0, 3.0)
        c = abs(a)
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(3.0)

    def test_abs_negative(self):
        a = Interval(-3.0, -1.0)
        c = abs(a)
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(3.0)

    def test_abs_crossing_zero(self, crossing_interval):
        c = abs(crossing_interval)
        assert c.lo == pytest.approx(0.0)
        assert c.hi == pytest.approx(2.0)

    # --- Power ---
    def test_pow_0(self):
        a = Interval(2.0, 3.0)
        c = a ** 0
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(1.0)

    def test_pow_1(self):
        a = Interval(2.0, 3.0)
        c = a ** 1
        assert c.lo == pytest.approx(2.0)
        assert c.hi == pytest.approx(3.0)

    def test_pow_2_positive(self):
        a = Interval(2.0, 3.0)
        c = a ** 2
        assert c.lo == pytest.approx(4.0)
        assert c.hi == pytest.approx(9.0)

    def test_pow_2_crossing_zero(self, crossing_interval):
        c = crossing_interval ** 2  # [-1, 2]^2
        assert c.lo == pytest.approx(0.0)
        assert c.hi == pytest.approx(4.0)

    def test_pow_2_negative(self, neg_interval):
        c = neg_interval ** 2  # [-2, -1]^2
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(4.0)

    def test_pow_3_monotone(self):
        a = Interval(2.0, 3.0)
        c = a ** 3
        assert c.lo == pytest.approx(8.0)
        assert c.hi == pytest.approx(27.0)

    def test_pow_negative_exponent(self):
        a = Interval(2.0, 3.0)
        with pytest.raises(ValueError):
            a ** (-1)


# =========================================================================
# Transcendental functions
# =========================================================================


class TestTranscendental:
    """Tests for exp, log, sqrt on intervals."""

    def test_exp_positive(self):
        a = Interval(0.0, 1.0)
        c = a.exp()
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(math.e)

    def test_exp_negative(self):
        a = Interval(-1.0, 0.0)
        c = a.exp()
        assert c.lo == pytest.approx(math.exp(-1.0))
        assert c.hi == pytest.approx(1.0)

    def test_exp_contains_exact(self):
        """For any x in [lo, hi], exp(x) must be in the result interval."""
        a = Interval(0.5, 1.5)
        c = a.exp()
        for x in np.linspace(a.lo, a.hi, 20):
            assert c.contains(math.exp(x))

    def test_exp_overflow_capping(self):
        """Very large exponent should be capped without error."""
        a = Interval(1000.0, 1000.0)
        c = a.exp()
        assert math.isfinite(c.lo)
        assert math.isfinite(c.hi)

    def test_log_positive(self):
        a = Interval(1.0, math.e)
        c = a.log()
        assert c.lo == pytest.approx(0.0)
        assert c.hi == pytest.approx(1.0)

    def test_log_contains_exact(self):
        a = Interval(0.5, 2.0)
        c = a.log()
        for x in np.linspace(a.lo, a.hi, 20):
            assert c.contains(math.log(x))

    def test_log_non_positive_raises(self):
        a = Interval(-1.0, 1.0)
        with pytest.raises(ValueError):
            a.log()

    def test_log_zero_lower_raises(self):
        a = Interval(0.0, 1.0)
        with pytest.raises(ValueError):
            a.log()

    def test_sqrt(self):
        a = Interval(4.0, 9.0)
        c = a.sqrt()
        assert c.lo == pytest.approx(2.0)
        assert c.hi == pytest.approx(3.0)

    def test_sqrt_contains_exact(self):
        a = Interval(1.0, 10.0)
        c = a.sqrt()
        for x in np.linspace(a.lo, a.hi, 20):
            assert c.contains(math.sqrt(x))

    def test_sqrt_negative_raises(self):
        a = Interval(-1.0, 4.0)
        with pytest.raises(ValueError):
            a.sqrt()


# =========================================================================
# Comparison predicates
# =========================================================================


class TestComparisons:
    """Tests for comparison predicates."""

    def test_certainly_less(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert a.certainly_less(b)
        assert not b.certainly_less(a)

    def test_certainly_less_overlapping(self):
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 4.0)
        assert not a.certainly_less(b)

    def test_certainly_leq(self):
        a = Interval(1.0, 2.0)
        b = Interval(2.0, 3.0)
        assert a.certainly_leq(b)

    def test_certainly_greater(self):
        a = Interval(5.0, 6.0)
        b = Interval(1.0, 2.0)
        assert a.certainly_greater(b)

    def test_possibly_equal(self):
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 4.0)
        assert a.possibly_equal(b)

    def test_possibly_equal_disjoint(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert not a.possibly_equal(b)

    def test_certainly_positive(self):
        assert Interval(1.0, 2.0).certainly_positive()
        assert not Interval(-1.0, 2.0).certainly_positive()
        assert not Interval(0.0, 1.0).certainly_positive()

    def test_certainly_nonneg(self):
        assert Interval(0.0, 1.0).certainly_nonneg()
        assert not Interval(-0.1, 1.0).certainly_nonneg()


# =========================================================================
# Set operations
# =========================================================================


class TestSetOperations:
    """Tests for interval set operations."""

    def test_intersect(self):
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 4.0)
        c = a.intersect(b)
        assert c is not None
        assert c.lo == pytest.approx(2.0)
        assert c.hi == pytest.approx(3.0)

    def test_intersect_disjoint(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert a.intersect(b) is None

    def test_hull(self):
        a = Interval(1.0, 3.0)
        b = Interval(5.0, 7.0)
        c = a.hull(b)
        assert c.lo == pytest.approx(1.0)
        assert c.hi == pytest.approx(7.0)

    def test_widen(self):
        a = Interval(1.0, 3.0)
        c = a.widen(0.5)  # width = 2, widen by 0.5 * 2 / 2 = 0.5
        assert c.lo == pytest.approx(0.5)
        assert c.hi == pytest.approx(3.5)


# =========================================================================
# IntervalMatrix tests
# =========================================================================


class TestIntervalMatrix:
    """Tests for IntervalMatrix and interval matvec."""

    def test_from_matrix(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = IntervalMatrix.from_matrix(A, abs_error=0.1)
        assert im.shape == (2, 2)
        assert im.m == 2
        assert im.n == 2
        np.testing.assert_allclose(im.lo, A - 0.1)
        np.testing.assert_allclose(im.hi, A + 0.1)

    def test_from_matrix_exact(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = IntervalMatrix.from_matrix(A)
        np.testing.assert_allclose(im.lo, A)
        np.testing.assert_allclose(im.hi, A)

    def test_matvec_point_matrix(self):
        """Point matrix × point vector = point result."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        im = IntervalMatrix.from_matrix(A)
        x_lo = np.array([2.0, 3.0])
        x_hi = np.array([2.0, 3.0])
        r_lo, r_hi = im.matvec(x_lo, x_hi)
        np.testing.assert_allclose(r_lo, [2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(r_hi, [2.0, 3.0], atol=1e-10)

    def test_matvec_contains_true_product(self):
        """Result interval must contain the true A·x for any A, x in intervals."""
        np.random.seed(42)
        A = np.random.randn(3, 4)
        err = 0.05
        im = IntervalMatrix.from_matrix(A, abs_error=err)

        x = np.random.randn(4)
        x_err = 0.02
        x_lo = x - x_err
        x_hi = x + x_err

        r_lo, r_hi = im.matvec(x_lo, x_hi)
        true_result = A @ x

        for i in range(3):
            assert r_lo[i] <= true_result[i] + 1e-10
            assert r_hi[i] >= true_result[i] - 1e-10

    def test_matvec_identity(self):
        """I × [x_lo, x_hi] = [x_lo, x_hi]."""
        n = 3
        im = IntervalMatrix.from_matrix(np.eye(n))
        x_lo = np.array([1.0, 2.0, 3.0])
        x_hi = np.array([4.0, 5.0, 6.0])
        r_lo, r_hi = im.matvec(x_lo, x_hi)
        np.testing.assert_allclose(r_lo, x_lo, atol=1e-10)
        np.testing.assert_allclose(r_hi, x_hi, atol=1e-10)

    def test_entry(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = IntervalMatrix.from_matrix(A, abs_error=0.1)
        entry = im.entry(0, 1)
        assert isinstance(entry, Interval)
        assert entry.lo == pytest.approx(1.9)
        assert entry.hi == pytest.approx(2.1)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            IntervalMatrix(np.ones((2, 3)), np.ones((3, 2)))

    def test_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            IntervalMatrix(np.ones(3), np.ones(3))

    def test_lo_gt_hi(self):
        with pytest.raises(ValueError, match="lo must be"):
            IntervalMatrix(np.ones((2, 2)) * 2, np.ones((2, 2)))

    def test_repr(self):
        im = IntervalMatrix.from_matrix(np.eye(2))
        assert "IntervalMatrix" in repr(im)


# =========================================================================
# Soundness: property-based testing with hypothesis
# =========================================================================


# Strategy for finite floats in a reasonable range
reasonable_float = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False)


class TestArithmeticSoundness:
    """Property-based tests: interval arithmetic must always contain the exact result."""

    @given(
        a=reasonable_float,
        b=reasonable_float,
        ea=st.floats(min_value=0.0, max_value=1.0),
        eb=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_add_soundness(self, a, b, ea, eb):
        """For x in [a-ea, a+ea] and y in [b-eb, b+eb], x+y must be in result."""
        ia = Interval(a - ea, a + ea)
        ib = Interval(b - eb, b + eb)
        result = ia + ib
        assert result.contains(a + b)

    @given(
        a=reasonable_float,
        b=reasonable_float,
        ea=st.floats(min_value=0.0, max_value=1.0),
        eb=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_sub_soundness(self, a, b, ea, eb):
        ia = Interval(a - ea, a + ea)
        ib = Interval(b - eb, b + eb)
        result = ia - ib
        assert result.contains(a - b)

    @given(
        a=reasonable_float,
        b=reasonable_float,
        ea=st.floats(min_value=0.0, max_value=1.0),
        eb=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_mul_soundness(self, a, b, ea, eb):
        ia = Interval(a - ea, a + ea)
        ib = Interval(b - eb, b + eb)
        result = ia * ib
        assert result.contains(a * b)

    @given(
        a=reasonable_float,
        b=positive_float,
        ea=st.floats(min_value=0.0, max_value=0.5),
        eb=st.floats(min_value=0.0, max_value=0.005),
    )
    @settings(max_examples=80)
    def test_div_soundness(self, a, b, ea, eb):
        """Division soundness: b > 0 and eb small enough that interval doesn't contain 0."""
        assume(b - eb > 0)
        ia = Interval(a - ea, a + ea)
        ib = Interval(b - eb, b + eb)
        result = ia / ib
        exact = a / b
        # Allow 1 ULP of floating-point slack since Interval uses reciprocal-multiply
        # rather than direct division, which can introduce ±1 ULP error
        eps = max(abs(exact) * 1e-14, 1e-300)
        assert result.lo <= exact + eps and result.hi >= exact - eps

    @given(
        a=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        ea=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_exp_soundness(self, a, ea):
        """exp(x) for x in [a-ea, a+ea] must be in exp result."""
        ia = Interval(a - ea, a + ea)
        result = ia.exp()
        assert result.contains(math.exp(a))

    @given(
        a=positive_float,
        ea=st.floats(min_value=0.0, max_value=0.005),
    )
    @settings(max_examples=50)
    def test_log_soundness(self, a, ea):
        """log(x) for x in [a-ea, a+ea] must be in log result (if a-ea > 0)."""
        assume(a - ea > 0)
        ia = Interval(a - ea, a + ea)
        result = ia.log()
        assert result.contains(math.log(a))

    @given(
        a=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ea=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_sqrt_soundness(self, a, ea):
        """sqrt(x) for x in [a-ea, a+ea] must be in sqrt result (if a-ea >= 0)."""
        assume(a - ea >= 0)
        ia = Interval(a - ea, a + ea)
        result = ia.sqrt()
        assert result.contains(math.sqrt(a))

    @given(
        a=reasonable_float,
        ea=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_pow2_soundness(self, a, ea):
        """x² for x in [a-ea, a+ea] must be in result."""
        ia = Interval(a - ea, a + ea)
        result = ia ** 2
        assert result.contains(a ** 2)

    @given(
        a=reasonable_float,
        ea=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_neg_soundness(self, a, ea):
        ia = Interval(a - ea, a + ea)
        result = -ia
        assert result.contains(-a)


# =========================================================================
# IntervalMatrix soundness with hypothesis
# =========================================================================


class TestMatrixSoundness:
    """Property-based tests for interval matrix-vector product soundness."""

    @given(
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30)
    def test_matvec_soundness(self, seed):
        """For random A, x in their intervals, A@x must be in the result."""
        rng = np.random.RandomState(seed)
        m, n = 3, 4
        A = rng.randn(m, n)
        x = rng.randn(n)
        a_err = 0.05
        x_err = 0.03

        im = IntervalMatrix.from_matrix(A, abs_error=a_err)
        x_lo = x - x_err
        x_hi = x + x_err

        r_lo, r_hi = im.matvec(x_lo, x_hi)
        true_result = A @ x

        for i in range(m):
            assert r_lo[i] <= true_result[i] + 1e-10, (
                f"Row {i}: r_lo={r_lo[i]}, true={true_result[i]}"
            )
            assert r_hi[i] >= true_result[i] - 1e-10, (
                f"Row {i}: r_hi={r_hi[i]}, true={true_result[i]}"
            )


# =========================================================================
# Zero-width and special interval tests
# =========================================================================


class TestSpecialIntervals:
    """Tests for zero-width, very small, and very large intervals."""

    def test_zero_width_add(self):
        a = Interval.exact(2.0)
        b = Interval.exact(3.0)
        c = a + b
        assert c.lo == pytest.approx(5.0)
        assert c.hi == pytest.approx(5.0)

    def test_zero_width_mul(self):
        a = Interval.exact(2.0)
        b = Interval.exact(3.0)
        c = a * b
        assert c.lo == pytest.approx(6.0)
        assert c.hi == pytest.approx(6.0)

    def test_very_small_interval(self):
        a = Interval(1.0 - 1e-15, 1.0 + 1e-15)
        b = Interval(2.0 - 1e-15, 2.0 + 1e-15)
        c = a + b
        assert c.contains(3.0)

    def test_negative_interval_operations(self, neg_interval):
        a = neg_interval  # [-2, -1]
        c = a + a
        assert c.lo == pytest.approx(-4.0)
        assert c.hi == pytest.approx(-2.0)

    def test_interval_containing_zero_mul(self, crossing_interval):
        """[-1, 2] × [-1, 2] should give [-2, 4]."""
        a = crossing_interval
        c = a * a
        assert c.lo <= 0  # Contains 0
        assert c.hi >= 4  # Contains 4
