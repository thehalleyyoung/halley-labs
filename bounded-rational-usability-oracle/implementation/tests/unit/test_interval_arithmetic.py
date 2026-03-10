"""Tests for interval arithmetic operations and IntervalArithmetic class.

This module covers the dunder arithmetic operators on
usability_oracle.interval.interval.Interval (__add__, __sub__, __mul__,
__truediv__, __pow__, __neg__, __abs__) as well as the functional API
exposed by usability_oracle.interval.arithmetic.IntervalArithmetic
(add, subtract, multiply, divide, power, sqrt, log, exp, min_interval,
max_interval, sum_intervals, dot_product, weighted_sum).
"""

from __future__ import annotations

import math
import pytest

from usability_oracle.interval.interval import Interval
from usability_oracle.interval.arithmetic import IntervalArithmetic


# ===================================================================
# Dunder arithmetic operators on Interval
# ===================================================================


class TestIntervalAdd:
    """Tests for Interval.__add__ and __radd__."""

    def test_add_two_intervals(self):
        """[1,2] + [3,4] should equal [4,6]."""
        result = Interval(1, 2) + Interval(3, 4)
        assert result == Interval(4, 6)

    def test_add_scalar_and_radd(self):
        """[1,2] + 5 and 5 + [1,2] should both equal [6,7]."""
        assert Interval(1, 2) + 5 == Interval(6, 7)
        assert 5 + Interval(1, 2) == Interval(6, 7)

    def test_add_negative_intervals(self):
        """[-3,-1] + [-2,0] should equal [-5,-1]."""
        result = Interval(-3, -1) + Interval(-2, 0)
        assert result == Interval(-5, -1)


class TestIntervalSub:
    """Tests for Interval.__sub__ and __rsub__."""

    def test_sub_two_intervals(self):
        """[3,5] - [1,2] should equal [1,4]."""
        assert Interval(3, 5) - Interval(1, 2) == Interval(1, 4)

    def test_sub_scalar_and_rsub(self):
        """[3,5] - 1 should equal [2,4]; 10 - [3,5] should equal [5,7]."""
        assert Interval(3, 5) - 1 == Interval(2, 4)
        assert 10 - Interval(3, 5) == Interval(5, 7)


class TestIntervalMul:
    """Tests for Interval.__mul__ and __rmul__."""

    def test_mul_positive(self):
        """[2,3] * [4,5] should equal [8,15]."""
        result = Interval(2, 3) * Interval(4, 5)
        assert result == Interval(8, 15)

    def test_mul_negative_times_positive(self):
        """[-3,-2] * [4,5] should equal [-15,-8]."""
        result = Interval(-3, -2) * Interval(4, 5)
        assert result == Interval(-15, -8)

    def test_mul_both_negative(self):
        """[-3,-2] * [-5,-4] should equal [8,15]."""
        result = Interval(-3, -2) * Interval(-5, -4)
        assert result == Interval(8, 15)

    def test_mul_straddling_zero(self):
        """[-1,2] * [-3,4] should use the four-product method."""
        result = Interval(-1, 2) * Interval(-3, 4)
        # min(3,-4,-6,8)=-6, max(3,-4,-6,8)=8
        assert result.low == pytest.approx(-6)
        assert result.high == pytest.approx(8)

    def test_mul_scalar_and_rmul(self):
        """[2,3] * 5 and 5 * [2,3] should both equal [10,15]."""
        assert Interval(2, 3) * 5 == Interval(10, 15)
        assert 5 * Interval(2, 3) == Interval(10, 15)

    def test_mul_by_zero_scalar(self):
        """[2,3] * 0 should equal [0,0]."""
        assert Interval(2, 3) * 0 == Interval(0, 0)


class TestIntervalDiv:
    """Tests for Interval.__truediv__ and __rtruediv__."""

    def test_div_positive(self):
        """[4,6] / [2,3] should equal [4/3, 3]."""
        result = Interval(4, 6) / Interval(2, 3)
        assert result.low == pytest.approx(4.0 / 3.0)
        assert result.high == pytest.approx(3.0)

    def test_div_by_zero_raises(self):
        """Division by [0,0] or an interval straddling zero should raise."""
        with pytest.raises(ZeroDivisionError):
            Interval(1, 2) / Interval(0, 0)
        with pytest.raises(ZeroDivisionError):
            Interval(1, 2) / Interval(-1, 1)

    def test_div_scalar_and_rtruediv(self):
        """[4,6] / 2 should equal [2,3]; 6 / [2,3] should equal [2,3]."""
        assert Interval(4, 6) / 2 == Interval(2, 3)
        assert 6 / Interval(2, 3) == Interval(2, 3)


class TestIntervalPow:
    """Tests for Interval.__pow__."""

    def test_pow_zero(self):
        """Any interval raised to 0 should give [1,1]."""
        assert Interval(2, 5) ** 0 == Interval(1, 1)

    def test_pow_one(self):
        """Raising to 1 should return the same interval."""
        assert Interval(2, 5) ** 1 == Interval(2, 5)

    def test_pow_odd(self):
        """Odd power of [-2,3] → [-8,27] (monotone)."""
        result = Interval(-2, 3) ** 3
        assert result.low == pytest.approx(-8)
        assert result.high == pytest.approx(27)

    def test_pow_even_positive(self):
        """Even power of [2,3] → [4,9]."""
        result = Interval(2, 3) ** 2
        assert result == Interval(4, 9)

    def test_pow_even_straddling(self):
        """Even power of [-3,2] should be [0, 9] (straddles zero)."""
        result = Interval(-3, 2) ** 2
        assert result.low == 0.0
        assert result.high == 9.0

    def test_pow_negative_exponent_raises(self):
        """Negative exponents are not supported and should raise."""
        with pytest.raises(ValueError):
            Interval(1, 2) ** (-1)


class TestIntervalUnary:
    """Tests for __neg__, __abs__, __pos__."""

    def test_neg(self):
        """-[1,3] should be [-3,-1]."""
        assert -Interval(1, 3) == Interval(-3, -1)

    def test_abs_cases(self):
        """|[2,5]| = [2,5]; |[-5,-2]| = [2,5]; |[-3,5]| = [0,5]."""
        assert abs(Interval(2, 5)) == Interval(2, 5)
        assert abs(Interval(-5, -2)) == Interval(2, 5)
        assert abs(Interval(-3, 5)) == Interval(0, 5)


# ===================================================================
# IntervalArithmetic functional API
# ===================================================================


class TestIntervalArithmeticBasic:
    """Tests for the named static methods on IntervalArithmetic."""

    def test_add(self):
        """IntervalArithmetic.add delegates to __add__."""
        result = IntervalArithmetic.add(Interval(1, 2), Interval(3, 4))
        assert result == Interval(4, 6)

    def test_subtract(self):
        """IntervalArithmetic.subtract delegates to __sub__."""
        result = IntervalArithmetic.subtract(Interval(5, 8), Interval(1, 2))
        assert result == Interval(3, 7)

    def test_multiply(self):
        """IntervalArithmetic.multiply delegates to __mul__."""
        result = IntervalArithmetic.multiply(Interval(2, 3), Interval(4, 5))
        assert result == Interval(8, 15)

    def test_divide(self):
        """IntervalArithmetic.divide delegates to __truediv__."""
        result = IntervalArithmetic.divide(Interval(4, 6), Interval(2, 3))
        assert result.low == pytest.approx(4 / 3)
        assert result.high == pytest.approx(3.0)

    def test_power(self):
        """IntervalArithmetic.power delegates to __pow__."""
        result = IntervalArithmetic.power(Interval(2, 3), 2)
        assert result == Interval(4, 9)


class TestIntervalArithmeticElementary:
    """sqrt, log, exp via IntervalArithmetic."""

    def test_sqrt_and_sqrt_negative_raises(self):
        """sqrt([4,9]) should be [2,3]; negative interval should raise."""
        result = IntervalArithmetic.sqrt(Interval(4, 9))
        assert result.low == pytest.approx(2.0)
        assert result.high == pytest.approx(3.0)
        with pytest.raises(ValueError):
            IntervalArithmetic.sqrt(Interval(-4, 9))

    def test_log_and_log_non_positive_raises(self):
        """log([1, e]) should be [0, 1]; non-positive interval should raise."""
        result = IntervalArithmetic.log(Interval(1, math.e))
        assert result.low == pytest.approx(0.0)
        assert result.high == pytest.approx(1.0)
        with pytest.raises(ValueError):
            IntervalArithmetic.log(Interval(0, 5))

    def test_exp(self):
        """exp([0, 1]) should be [1, e]."""
        result = IntervalArithmetic.exp(Interval(0, 1))
        assert result.low == pytest.approx(1.0)
        assert result.high == pytest.approx(math.e)


class TestIntervalArithmeticMinMax:
    """min_interval and max_interval."""

    def test_min_interval(self):
        """min_interval([1,5], [3,7]) should be [1,5];
        min_interval([10,20], [1,5]) should be [1,5]."""
        assert IntervalArithmetic.min_interval(Interval(1, 5), Interval(3, 7)) == Interval(1, 5)
        assert IntervalArithmetic.min_interval(Interval(10, 20), Interval(1, 5)) == Interval(1, 5)

    def test_max_interval(self):
        """max_interval([1,5], [3,7]) should be [3,7];
        max_interval([10,20], [1,5]) should be [10,20]."""
        assert IntervalArithmetic.max_interval(Interval(1, 5), Interval(3, 7)) == Interval(3, 7)
        assert IntervalArithmetic.max_interval(Interval(10, 20), Interval(1, 5)) == Interval(10, 20)


class TestIntervalArithmeticAggregate:
    """sum_intervals, dot_product, weighted_sum."""

    def test_sum_intervals_and_empty_raises(self):
        """sum_intervals of [1,2], [3,4], [5,6] should be [9,12];
        empty sequence should raise ValueError."""
        result = IntervalArithmetic.sum_intervals([
            Interval(1, 2), Interval(3, 4), Interval(5, 6),
        ])
        assert result == Interval(9, 12)
        with pytest.raises(ValueError, match="empty"):
            IntervalArithmetic.sum_intervals([])

    def test_dot_product(self):
        """Dot product of ([1,2],[3,4]) · ([5,6],[7,8]) should be
        [1*5+3*7, 2*6+4*8] = [26,44]."""
        a = [Interval(1, 2), Interval(3, 4)]
        b = [Interval(5, 6), Interval(7, 8)]
        result = IntervalArithmetic.dot_product(a, b)
        assert result.low == pytest.approx(26)
        assert result.high == pytest.approx(44)

    def test_dot_product_length_mismatch_and_empty_raises(self):
        """Mismatched vector lengths and empty vectors should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            IntervalArithmetic.dot_product(
                [Interval(1, 2)],
                [Interval(3, 4), Interval(5, 6)],
            )
        with pytest.raises(ValueError, match="empty"):
            IntervalArithmetic.dot_product([], [])

    def test_weighted_sum(self):
        """weighted_sum should be mathematically equivalent to dot_product."""
        vals = [Interval(1, 2), Interval(3, 4)]
        weights = [Interval(0.5, 0.5), Interval(0.5, 0.5)]
        ws = IntervalArithmetic.weighted_sum(vals, weights)
        dp = IntervalArithmetic.dot_product(weights, vals)
        assert ws == dp


# ===================================================================
# Containment property — result encloses all point evaluations
# ===================================================================


class TestContainmentProperty:
    """Every interval operation should produce a result that contains
    the corresponding point-arithmetic result for every pair of
    endpoints sampled from the operands."""

    @staticmethod
    def _sample_points(iv: Interval) -> list[float]:
        return [iv.low, iv.midpoint, iv.high]

    def test_add_containment(self):
        """Addition result must contain a+b for all a in A, b in B."""
        a, b = Interval(1, 3), Interval(2, 5)
        result = a + b
        for x in self._sample_points(a):
            for y in self._sample_points(b):
                assert result.contains(x + y)

    def test_sub_containment(self):
        """Subtraction result must contain a-b for all a in A, b in B."""
        a, b = Interval(1, 5), Interval(2, 3)
        result = a - b
        for x in self._sample_points(a):
            for y in self._sample_points(b):
                assert result.contains(x - y)

    def test_mul_containment(self):
        """Multiplication result must contain a*b for all a in A, b in B."""
        a, b = Interval(-2, 3), Interval(-1, 4)
        result = a * b
        for x in self._sample_points(a):
            for y in self._sample_points(b):
                assert result.contains(x * y)

    def test_pow_containment(self):
        """Power result must contain a^n for all a in A."""
        a = Interval(-2, 3)
        result = a ** 2
        for x in self._sample_points(a):
            assert result.contains(x ** 2)
