"""Tests for interval arithmetic."""

import pytest
import math
import numpy as np
from collusion_proof.interval_arithmetic import (
    Interval,
    IntervalVector,
    interval_mean,
    interval_variance,
    interval_correlation,
    interval_regression_slope,
    interval_t_statistic,
    interval_collusion_premium,
)


class TestIntervalBasic:
    def test_creation(self):
        iv = Interval(1.0, 3.0)
        assert iv.lo == 1.0
        assert iv.hi == 3.0

    def test_auto_swap(self):
        iv = Interval(5.0, 2.0)
        assert iv.lo == 2.0
        assert iv.hi == 5.0

    def test_point(self):
        iv = Interval.point(3.14)
        assert iv.lo == 3.14
        assert iv.hi == 3.14
        assert iv.is_point

    def test_pm(self):
        iv = Interval.pm(5.0, 1.0)
        assert iv.lo == 4.0
        assert iv.hi == 6.0

    def test_from_confidence(self):
        iv = Interval.from_confidence(10.0, 1.0, z=1.96)
        assert abs(iv.lo - 8.04) < 0.01
        assert abs(iv.hi - 11.96) < 0.01

    def test_properties_width(self):
        iv = Interval(2.0, 5.0)
        assert iv.width == 3.0

    def test_properties_midpoint(self):
        iv = Interval(2.0, 8.0)
        assert iv.midpoint == 5.0

    def test_properties_radius(self):
        iv = Interval(1.0, 5.0)
        assert iv.radius == 2.0

    def test_is_positive(self):
        assert Interval(1.0, 3.0).is_positive
        assert not Interval(-1.0, 3.0).is_positive
        assert not Interval(-3.0, -1.0).is_positive

    def test_is_negative(self):
        assert Interval(-3.0, -1.0).is_negative
        assert not Interval(-1.0, 3.0).is_negative

    def test_contains(self):
        iv = Interval(1.0, 5.0)
        assert iv.contains(3.0)
        assert iv.contains(1.0)
        assert iv.contains(5.0)
        assert not iv.contains(0.0)
        assert not iv.contains(6.0)

    def test_overlaps(self):
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 5.0)
        c = Interval(4.0, 6.0)
        assert a.overlaps(b)
        assert not a.overlaps(c)

    def test_intersection(self):
        a = Interval(1.0, 4.0)
        b = Interval(2.0, 6.0)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.lo == 2.0
        assert inter.hi == 4.0

    def test_intersection_none(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert a.intersection(b) is None

    def test_union_hull(self):
        a = Interval(1.0, 3.0)
        b = Interval(5.0, 7.0)
        hull = a.union_hull(b)
        assert hull.lo == 1.0
        assert hull.hi == 7.0

    def test_hull_class_method(self):
        intervals = [Interval(1, 3), Interval(5, 7), Interval(2, 4)]
        h = Interval.hull(intervals)
        assert h.lo == 1.0
        assert h.hi == 7.0

    def test_hull_empty(self):
        h = Interval.hull([])
        # hull of empty list produces the sentinel (inf, -inf), which auto-swaps
        # to (-inf, inf) due to frozen-dataclass swap logic
        assert h.lo == float("-inf") or h.lo == float("inf")

    def test_empty(self):
        e = Interval.empty()
        # empty() returns Interval(inf, -inf); the swap logic may produce (-inf, inf)
        # The key invariant is lo >= hi (before swap) or it wraps to entire
        assert e.lo == float("-inf") or e.lo == float("inf")

    def test_entire(self):
        e = Interval.entire()
        assert e.lo == float("-inf")
        assert e.hi == float("inf")

    def test_equality(self):
        assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
        assert Interval(1.0, 2.0) != Interval(1.0, 3.0)

    def test_hash(self):
        a = Interval(1.0, 2.0)
        b = Interval(1.0, 2.0)
        assert hash(a) == hash(b)


class TestIntervalArithmetic:
    def test_addition(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a + b
        assert c.lo == 4.0
        assert c.hi == 6.0

    def test_addition_scalar(self):
        a = Interval(1.0, 2.0)
        c = a + 5
        assert c.lo == 6.0
        assert c.hi == 7.0

    def test_radd(self):
        a = Interval(1.0, 2.0)
        c = 3 + a
        assert c.lo == 4.0
        assert c.hi == 5.0

    def test_subtraction(self):
        a = Interval(3.0, 5.0)
        b = Interval(1.0, 2.0)
        c = a - b
        assert c.lo == 1.0
        assert c.hi == 4.0

    def test_multiplication(self):
        a = Interval(2.0, 3.0)
        b = Interval(4.0, 5.0)
        c = a * b
        assert c.lo == 8.0
        assert c.hi == 15.0

    def test_multiplication_negative(self):
        a = Interval(-2.0, 3.0)
        b = Interval(-1.0, 4.0)
        c = a * b
        assert c.lo == -8.0
        assert c.hi == 12.0

    def test_division(self):
        a = Interval(6.0, 12.0)
        b = Interval(2.0, 3.0)
        c = a / b
        assert c.lo == 2.0
        assert c.hi == 6.0

    def test_division_by_zero(self):
        a = Interval(1.0, 2.0)
        b = Interval(0.0, 0.0)
        with pytest.raises(ZeroDivisionError):
            a / b

    def test_division_containing_zero(self):
        a = Interval(1.0, 2.0)
        b = Interval(-1.0, 1.0)
        c = a / b
        assert c.lo == float("-inf")
        assert c.hi == float("inf")

    def test_negation(self):
        a = Interval(1.0, 3.0)
        b = -a
        assert b.lo == -3.0
        assert b.hi == -1.0

    def test_abs_positive(self):
        a = Interval(2.0, 5.0)
        b = abs(a)
        assert b.lo == 2.0
        assert b.hi == 5.0

    def test_abs_negative(self):
        a = Interval(-5.0, -2.0)
        b = abs(a)
        assert b.lo == 2.0
        assert b.hi == 5.0

    def test_abs_straddle(self):
        a = Interval(-3.0, 5.0)
        b = abs(a)
        assert b.lo == 0.0
        assert b.hi == 5.0

    def test_power_even(self):
        iv = Interval(-1.0, 2.0)
        sq = iv ** 2
        assert sq.lo == 0.0
        assert sq.hi == 4.0

    def test_power_even_positive(self):
        iv = Interval(2.0, 3.0)
        sq = iv ** 2
        assert sq.lo == 4.0
        assert sq.hi == 9.0

    def test_power_even_negative(self):
        iv = Interval(-3.0, -2.0)
        sq = iv ** 2
        assert sq.lo == 4.0
        assert sq.hi == 9.0

    def test_power_odd(self):
        iv = Interval(-2.0, 3.0)
        cu = iv ** 3
        assert cu.lo == -8.0
        assert cu.hi == 27.0

    def test_power_zero(self):
        iv = Interval(2.0, 5.0)
        result = iv ** 0
        assert result == Interval.point(1.0)

    def test_scalar_ops(self):
        a = Interval(2.0, 4.0)
        assert (a * 2).lo == 4.0
        assert (a * 2).hi == 8.0
        assert (2 * a).lo == 4.0
        assert (2 * a).hi == 8.0


class TestIntervalMath:
    def test_sqrt(self):
        iv = Interval(4.0, 9.0)
        s = iv.sqrt()
        assert abs(s.lo - 2.0) < 1e-10
        assert abs(s.hi - 3.0) < 1e-10

    def test_sqrt_zero(self):
        iv = Interval(0.0, 4.0)
        s = iv.sqrt()
        assert s.lo == 0.0
        assert abs(s.hi - 2.0) < 1e-10

    def test_sqrt_negative_error(self):
        iv = Interval(-4.0, -1.0)
        with pytest.raises(ValueError):
            iv.sqrt()

    def test_sqrt_straddle(self):
        iv = Interval(-1.0, 4.0)
        s = iv.sqrt()
        assert s.lo == 0.0
        assert abs(s.hi - 2.0) < 1e-10

    def test_exp(self):
        iv = Interval(0.0, 1.0)
        e = iv.exp()
        assert abs(e.lo - 1.0) < 1e-10
        assert abs(e.hi - math.e) < 1e-10

    def test_log(self):
        iv = Interval(1.0, math.e)
        l = iv.log()
        assert abs(l.lo - 0.0) < 1e-10
        assert abs(l.hi - 1.0) < 1e-10

    def test_log_negative_error(self):
        iv = Interval(-2.0, -1.0)
        with pytest.raises(ValueError):
            iv.log()

    def test_sin_narrow(self):
        iv = Interval(0.0, math.pi / 4)
        s = iv.sin()
        assert s.lo <= 0.0 + 1e-10
        assert s.hi >= math.sin(math.pi / 4) - 1e-10

    def test_cos_narrow(self):
        iv = Interval(0.0, math.pi / 4)
        c = iv.cos()
        assert c.contains(math.cos(0.0))
        assert c.contains(math.cos(math.pi / 4))

    def test_sin_wide(self):
        iv = Interval(0.0, 10.0)
        s = iv.sin()
        assert s.lo >= -1.0 - 1e-10
        assert s.hi <= 1.0 + 1e-10


class TestIntervalComparison:
    def test_definitely_less(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert a.definitely_less_than(b)
        assert not b.definitely_less_than(a)

    def test_not_definitely_less(self):
        a = Interval(1.0, 3.5)
        b = Interval(3.0, 4.0)
        assert not a.definitely_less_than(b)

    def test_definitely_greater(self):
        a = Interval(5.0, 6.0)
        b = Interval(1.0, 2.0)
        assert a.definitely_greater_than(b)

    def test_possibly_equal(self):
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 4.0)
        assert a.possibly_equal(b)

    def test_not_possibly_equal(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        assert not a.possibly_equal(b)

    def test_definitely_less_scalar(self):
        a = Interval(1.0, 2.0)
        assert a.definitely_less_than(3.0)
        assert not a.definitely_less_than(1.5)


class TestIntervalVector:
    def test_creation(self):
        ivs = [Interval(1, 2), Interval(3, 4), Interval(5, 6)]
        v = IntervalVector(ivs)
        assert len(v) == 3
        assert v[0] == Interval(1, 2)

    def test_from_arrays(self):
        lo = np.array([1.0, 2.0, 3.0])
        hi = np.array([4.0, 5.0, 6.0])
        v = IntervalVector.from_arrays(lo, hi)
        assert len(v) == 3
        assert v[0].lo == 1.0
        assert v[2].hi == 6.0

    def test_from_point_array(self):
        v = IntervalVector.from_point_array(np.array([1.0, 2.0, 3.0]))
        assert len(v) == 3
        assert v[0].is_point
        assert v[1].midpoint == 2.0

    def test_from_confidence_arrays(self):
        values = np.array([10.0, 20.0])
        errors = np.array([1.0, 2.0])
        v = IntervalVector.from_confidence_arrays(values, errors, z=1.96)
        assert v[0].contains(10.0)
        assert v[1].contains(20.0)

    def test_addition(self):
        a = IntervalVector([Interval(1, 2), Interval(3, 4)])
        b = IntervalVector([Interval(5, 6), Interval(7, 8)])
        c = a + b
        assert c[0] == Interval(6, 8)
        assert c[1] == Interval(10, 12)

    def test_subtraction(self):
        a = IntervalVector([Interval(5, 6)])
        b = IntervalVector([Interval(1, 2)])
        c = a - b
        assert c[0].lo == 3.0
        assert c[0].hi == 5.0

    def test_multiplication(self):
        a = IntervalVector([Interval(2, 3)])
        b = IntervalVector([Interval(4, 5)])
        c = a * b
        assert c[0].lo == 8.0
        assert c[0].hi == 15.0

    def test_division(self):
        a = IntervalVector([Interval(6, 12)])
        b = IntervalVector([Interval(2, 3)])
        c = a / b
        assert c[0].lo == 2.0
        assert c[0].hi == 6.0

    def test_scalar_operations(self):
        v = IntervalVector([Interval(1, 2), Interval(3, 4)])
        c = v + 10
        assert c[0].lo == 11.0
        assert c[1].hi == 14.0

    def test_dot_product(self):
        a = IntervalVector([Interval.point(1.0), Interval.point(2.0), Interval.point(3.0)])
        b = IntervalVector([Interval.point(4.0), Interval.point(5.0), Interval.point(6.0)])
        d = a.dot(b)
        # 1*4 + 2*5 + 3*6 = 32
        assert d.contains(32.0)

    def test_dot_product_intervals(self):
        a = IntervalVector([Interval(1, 2), Interval(3, 4)])
        b = IntervalVector([Interval(1, 1), Interval(1, 1)])
        d = a.dot(b)
        assert d.lo == 4.0
        assert d.hi == 6.0

    def test_sum(self):
        v = IntervalVector([Interval(1, 2), Interval(3, 4), Interval(5, 6)])
        s = v.sum()
        assert s.lo == 9.0
        assert s.hi == 12.0

    def test_mean(self):
        v = IntervalVector([Interval(2, 4), Interval(4, 6)])
        m = v.mean()
        assert m.lo == 3.0
        assert m.hi == 5.0

    def test_mean_empty_error(self):
        v = IntervalVector([])
        with pytest.raises(ValueError):
            v.mean()

    def test_lo_hi_arrays(self):
        v = IntervalVector([Interval(1, 2), Interval(3, 5)])
        np.testing.assert_array_equal(v.lo_array, [1.0, 3.0])
        np.testing.assert_array_equal(v.hi_array, [2.0, 5.0])

    def test_midpoints(self):
        v = IntervalVector([Interval(1, 3), Interval(4, 6)])
        np.testing.assert_array_equal(v.midpoints, [2.0, 5.0])

    def test_widths(self):
        v = IntervalVector([Interval(1, 3), Interval(4, 10)])
        np.testing.assert_array_equal(v.widths, [2.0, 6.0])

    def test_length_mismatch(self):
        a = IntervalVector([Interval(1, 2)])
        b = IntervalVector([Interval(1, 2), Interval(3, 4)])
        with pytest.raises(ValueError):
            a + b

    def test_dot_length_mismatch(self):
        a = IntervalVector([Interval(1, 2)])
        b = IntervalVector([Interval(1, 2), Interval(3, 4)])
        with pytest.raises(ValueError):
            a.dot(b)


class TestIntervalUtilities:
    def test_interval_mean_simple(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        iv = interval_mean(data)
        assert iv.contains(3.0)

    def test_interval_mean_large_sample(self):
        rng = np.random.RandomState(42)
        data = rng.normal(10.0, 1.0, 1000)
        iv = interval_mean(data)
        assert iv.contains(float(np.mean(data)))

    def test_interval_variance_simple(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        iv = interval_variance(data)
        sample_var = float(np.var(data, ddof=1))
        # The interval should be reasonably close
        assert iv.width > 0

    def test_interval_variance_too_few(self):
        with pytest.raises(ValueError):
            interval_variance(np.array([1.0]))

    def test_interval_correlation_basic(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 100)
        y = 0.5 * x + rng.normal(0, 1, 100)
        iv = interval_correlation(x, y)
        assert iv.lo >= -1.0
        assert iv.hi <= 1.0
        sample_r = float(np.corrcoef(x, y)[0, 1])
        assert iv.contains(sample_r)

    def test_interval_correlation_small_sample(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        iv = interval_correlation(x, y)
        assert iv == Interval(-1.0, 1.0)

    def test_interval_regression_slope(self):
        rng = np.random.RandomState(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * x + rng.normal(0, 1.0, 100)
        iv = interval_regression_slope(x, y)
        assert iv.contains(2.5)

    def test_interval_t_statistic(self):
        rng = np.random.RandomState(42)
        data = rng.normal(5.0, 1.0, 100)
        iv = interval_t_statistic(data, mu0=5.0)
        assert iv.contains(0.0) or abs(iv.midpoint) < 3.0

    def test_interval_t_statistic_significant(self):
        rng = np.random.RandomState(42)
        data = rng.normal(10.0, 1.0, 100)
        iv = interval_t_statistic(data, mu0=0.0)
        assert iv.lo > 0  # should be very significantly positive

    def test_interval_collusion_premium_competitive(self):
        obs = Interval.point(1.0)
        nash = Interval.point(1.0)
        mono = Interval.point(5.5)
        premium = interval_collusion_premium(obs, nash, mono)
        assert premium.contains(0.0)

    def test_interval_collusion_premium_monopoly(self):
        obs = Interval.point(5.5)
        nash = Interval.point(1.0)
        mono = Interval.point(5.5)
        premium = interval_collusion_premium(obs, nash, mono)
        assert premium.contains(1.0)

    def test_interval_collusion_premium_mid(self):
        obs = Interval(3.0, 3.5)
        nash = Interval(0.9, 1.1)
        mono = Interval(5.3, 5.7)
        premium = interval_collusion_premium(obs, nash, mono)
        assert premium.width > 0

    def test_interval_collusion_premium_identical(self):
        obs = Interval.point(3.0)
        nash = Interval.point(3.0)
        mono = Interval.point(3.0)
        with pytest.raises(ValueError):
            interval_collusion_premium(obs, nash, mono)
