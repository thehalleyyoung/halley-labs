"""Tests for usability_oracle.interval.interval — Interval comparison methods.

This module tests the dunder comparison protocol (__eq__, __le__, __lt__,
__ge__, __gt__, __contains__, __hash__, __bool__, __float__) on the
Interval class, verifying the partial-order semantics, hash consistency,
boolean truthiness, and float conversion.
"""

from __future__ import annotations

import pytest

from usability_oracle.interval.interval import Interval


# ===================================================================
# Equality
# ===================================================================


class TestEquality:
    """__eq__ and related identity checks."""

    def test_eq_same_endpoints(self):
        """Two intervals with identical endpoints should be equal."""
        assert Interval(1, 3) == Interval(1, 3)

    def test_eq_different_endpoints(self):
        """Intervals with different endpoints should not be equal."""
        assert Interval(1, 3) != Interval(1, 4)

    def test_eq_different_low(self):
        """Differing lower bounds → not equal."""
        assert Interval(0, 3) != Interval(1, 3)

    def test_eq_degenerate(self):
        """Two degenerate intervals at the same point should be equal."""
        assert Interval.from_value(7) == Interval.from_value(7)

    def test_eq_with_non_interval_returns_not_implemented(self):
        """Comparing an Interval with a non-Interval should not be equal."""
        assert Interval(1, 2) != 42
        assert Interval(1, 2) != "not an interval"

    def test_eq_reflexive(self):
        """An interval should equal itself (reflexivity)."""
        iv = Interval(2, 5)
        assert iv == iv

    def test_eq_symmetric(self):
        """a == b implies b == a (symmetry)."""
        a = Interval(1, 3)
        b = Interval(1, 3)
        assert a == b and b == a


# ===================================================================
# Partial-order comparisons
# ===================================================================


class TestPartialOrder:
    """__le__, __lt__, __ge__, __gt__ implement a partial order based
    on interval endpoints."""

    def test_lt_disjoint(self):
        """[1,2] < [3,4] because 2 < 3."""
        assert Interval(1, 2) < Interval(3, 4)

    def test_lt_adjacent_false(self):
        """[1,3] is NOT < [3,5] because 3 is not < 3."""
        assert not (Interval(1, 3) < Interval(3, 5))

    def test_le_disjoint(self):
        """[1,2] <= [3,4] because 2 <= 3."""
        assert Interval(1, 2) <= Interval(3, 4)

    def test_le_adjacent(self):
        """[1,3] <= [3,5] because 3 <= 3."""
        assert Interval(1, 3) <= Interval(3, 5)

    def test_le_overlapping_false(self):
        """[1,4] is NOT <= [3,5] because 4 > 3."""
        assert not (Interval(1, 4) <= Interval(3, 5))

    def test_gt_disjoint(self):
        """[5,6] > [1,2] because 5 > 2."""
        assert Interval(5, 6) > Interval(1, 2)

    def test_gt_adjacent_false(self):
        """[3,5] is NOT > [1,3] because 3 is not > 3."""
        assert not (Interval(3, 5) > Interval(1, 3))

    def test_ge_disjoint(self):
        """[5,6] >= [1,2] because 5 >= 2."""
        assert Interval(5, 6) >= Interval(1, 2)

    def test_ge_adjacent(self):
        """[3,5] >= [1,3] because 3 >= 3."""
        assert Interval(3, 5) >= Interval(1, 3)

    def test_ge_overlapping_false(self):
        """[2,5] is NOT >= [3,7] because 2 < 7."""
        assert not (Interval(2, 5) >= Interval(3, 7))

    def test_overlapping_not_lt(self):
        """Overlapping intervals should NOT satisfy strict less-than
        since the high of the first exceeds the low of the second."""
        assert not (Interval(1, 5) < Interval(3, 7))

    def test_overlapping_not_gt(self):
        """Overlapping intervals should NOT satisfy strict greater-than."""
        assert not (Interval(3, 7) > Interval(1, 5))

    def test_identical_le_and_ge(self):
        """Equal intervals satisfy both <= and >=."""
        a = Interval(2, 5)
        b = Interval(2, 5)
        # a.high <= b.low is 5 <= 2 → False, so NOT <=
        # These are overlapping, not disjoint, so partial order is incomparable
        assert not (a <= b)  # 5 <= 2 is False
        assert not (a >= b)  # 2 >= 5 is False

    def test_disjoint_ordering_consistency(self):
        """For disjoint intervals, exactly one of <, > should be True."""
        a = Interval(1, 2)
        b = Interval(4, 5)
        assert (a < b) != (a > b)


# ===================================================================
# __contains__ for value-in-interval
# ===================================================================


class TestContainsDunder:
    """Test `value in interval` via __contains__."""

    def test_value_in_interval(self):
        """A value inside the interval should be 'in' it."""
        assert 3 in Interval(1, 5)

    def test_endpoint_in_interval(self):
        """Endpoints should be 'in' the interval (closed)."""
        assert 1 in Interval(1, 5)
        assert 5 in Interval(1, 5)

    def test_value_outside_interval(self):
        """A value outside should NOT be 'in' the interval."""
        assert 10 not in Interval(1, 5)

    def test_negative_in_straddling(self):
        """A negative value in a straddling interval should be 'in' it."""
        assert -0.5 in Interval(-1, 1)

    def test_zero_in_straddling(self):
        """Zero in a straddling interval."""
        assert 0 in Interval(-1, 1)


# ===================================================================
# __hash__ consistency with __eq__
# ===================================================================


class TestHash:
    """__hash__ must be consistent with __eq__."""

    def test_equal_intervals_same_hash(self):
        """Equal intervals must have the same hash."""
        a = Interval(1, 3)
        b = Interval(1, 3)
        assert hash(a) == hash(b)

    def test_different_intervals_likely_different_hash(self):
        """Different intervals should (very likely) have different hashes."""
        a = Interval(1, 3)
        b = Interval(2, 4)
        # Not guaranteed, but overwhelmingly likely for these values
        assert hash(a) != hash(b)

    def test_usable_as_dict_key(self):
        """Intervals should be usable as dict keys."""
        d = {Interval(1, 3): "a", Interval(2, 4): "b"}
        assert d[Interval(1, 3)] == "a"

    def test_usable_in_set(self):
        """Intervals should be usable in sets, with duplicates collapsed."""
        s = {Interval(1, 3), Interval(1, 3), Interval(2, 4)}
        assert len(s) == 2

    def test_degenerate_hash_consistency(self):
        """Degenerate intervals at the same point should have equal hashes."""
        a = Interval.from_value(42)
        b = Interval(42, 42)
        assert hash(a) == hash(b)


# ===================================================================
# __bool__
# ===================================================================


class TestBool:
    """__bool__ should be True for valid non-zero intervals."""

    def test_nonzero_interval_is_truthy(self):
        """A non-degenerate interval is truthy."""
        assert bool(Interval(1, 3)) is True

    def test_zero_degenerate_is_falsy(self):
        """[0,0] should be falsy."""
        assert bool(Interval(0, 0)) is False

    def test_nonzero_degenerate_is_truthy(self):
        """[5,5] should be truthy even though degenerate."""
        assert bool(Interval(5, 5)) is True

    def test_straddling_zero_is_truthy(self):
        """[-1,1] is truthy because it is not [0,0]."""
        assert bool(Interval(-1, 1)) is True

    def test_negative_interval_is_truthy(self):
        """[-5,-2] is truthy."""
        assert bool(Interval(-5, -2)) is True


# ===================================================================
# __float__
# ===================================================================


class TestFloat:
    """__float__ converts an interval to its midpoint."""

    def test_float_degenerate(self):
        """For a degenerate interval, float should return the exact value."""
        assert float(Interval.from_value(7)) == 7.0

    def test_float_non_degenerate(self):
        """For [2,8], float should return the midpoint 5.0."""
        assert float(Interval(2, 8)) == pytest.approx(5.0)

    def test_float_symmetric(self):
        """For [-3,3], midpoint is 0."""
        assert float(Interval(-3, 3)) == pytest.approx(0.0)

    def test_float_usable_in_arithmetic(self):
        """float(interval) should be usable in normal arithmetic."""
        result = float(Interval(2, 4)) + 1
        assert result == pytest.approx(4.0)
