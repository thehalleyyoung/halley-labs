"""Tests for usability_oracle.interval.interval — Interval class.

This module exercises construction, properties, predicates, set operations,
and string representation of the core Interval type used throughout the
bounded-rational usability oracle for rigorous uncertainty propagation.
"""

from __future__ import annotations

import math
import pytest

from usability_oracle.interval.interval import Interval


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    """Verify the various ways to create Interval instances."""

    def test_basic_construction(self):
        """Interval(1, 3) should store low=1.0 and high=3.0."""
        iv = Interval(1, 3)
        assert iv.low == 1.0
        assert iv.high == 3.0

    def test_from_value(self):
        """from_value(5) should create the degenerate interval [5, 5]."""
        iv = Interval.from_value(5)
        assert iv.low == 5.0
        assert iv.high == 5.0

    def test_from_center_radius(self):
        """from_center_radius(10, 2) should give [8, 12]."""
        iv = Interval.from_center_radius(10, 2)
        assert iv.low == pytest.approx(8.0)
        assert iv.high == pytest.approx(12.0)

    def test_from_center_radius_zero(self):
        """A zero radius should produce a degenerate interval."""
        iv = Interval.from_center_radius(5, 0)
        assert iv.low == iv.high == 5.0

    def test_from_center_radius_negative_raises(self):
        """A negative radius must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Interval.from_center_radius(5, -1)

    def test_invalid_low_gt_high_raises(self):
        """Constructing with low > high should raise ValueError."""
        with pytest.raises(ValueError, match="must not exceed"):
            Interval(5, 3)

    def test_nan_low_raises(self):
        """NaN as a bound should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            Interval(float("nan"), 1)

    def test_nan_high_raises(self):
        """NaN as the upper bound should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            Interval(1, float("nan"))

    def test_int_coercion(self):
        """Integer arguments should be silently promoted to float."""
        iv = Interval(2, 4)
        assert isinstance(iv.low, float)
        assert isinstance(iv.high, float)

    def test_negative_bounds(self):
        """Negative bounds should be perfectly valid."""
        iv = Interval(-5, -2)
        assert iv.low == -5.0
        assert iv.high == -2.0

    def test_infinite_bounds(self):
        """Infinite bounds should be accepted."""
        iv = Interval(-math.inf, math.inf)
        assert iv.low == -math.inf
        assert iv.high == math.inf


# ===================================================================
# Properties
# ===================================================================


class TestProperties:
    """Verify computed properties on Interval instances."""

    def test_width(self):
        """width should be high - low."""
        assert Interval(2, 7).width == pytest.approx(5.0)

    def test_width_degenerate(self):
        """A degenerate interval has width 0."""
        assert Interval.from_value(3).width == 0.0

    def test_midpoint(self):
        """midpoint should be the average of low and high."""
        assert Interval(2, 8).midpoint == pytest.approx(5.0)

    def test_is_degenerate_true(self):
        """from_value produces a degenerate interval."""
        assert Interval.from_value(42).is_degenerate is True

    def test_is_degenerate_false(self):
        """A non-trivial interval is not degenerate."""
        assert Interval(1, 2).is_degenerate is False


# ===================================================================
# Predicates
# ===================================================================


class TestPredicates:
    """Test containment and comparison predicates."""

    def test_contains_interior(self):
        """A value strictly inside the interval should be contained."""
        assert Interval(1, 5).contains(3)

    def test_contains_low_endpoint(self):
        """The lower endpoint should be contained (closed interval)."""
        assert Interval(1, 5).contains(1)

    def test_contains_high_endpoint(self):
        """The upper endpoint should be contained (closed interval)."""
        assert Interval(1, 5).contains(5)

    def test_not_contains_outside(self):
        """A value outside should not be contained."""
        assert not Interval(1, 5).contains(6)

    def test_overlaps_true(self):
        """Two intervals sharing a region should overlap."""
        assert Interval(1, 5).overlaps(Interval(3, 7))

    def test_overlaps_single_point(self):
        """Intervals touching at a single point should still overlap."""
        assert Interval(1, 3).overlaps(Interval(3, 5))

    def test_overlaps_false(self):
        """Disjoint intervals should not overlap."""
        assert not Interval(1, 2).overlaps(Interval(3, 4))

    def test_includes_zero_positive(self):
        """A strictly positive interval does not include zero."""
        assert not Interval(1, 5).includes_zero()

    def test_includes_zero_straddling(self):
        """An interval straddling zero includes zero."""
        assert Interval(-1, 1).includes_zero()

    def test_includes_zero_at_boundary(self):
        """An interval with 0 as an endpoint includes zero."""
        assert Interval(0, 5).includes_zero()

    def test_is_positive(self):
        """is_positive is True when low > 0."""
        assert Interval(0.1, 5).is_positive()
        assert not Interval(-1, 5).is_positive()

    def test_is_negative(self):
        """is_negative is True when high < 0."""
        assert Interval(-5, -0.1).is_negative()
        assert not Interval(-5, 1).is_negative()

    def test_is_subset_of_true(self):
        """[2,3] is a subset of [1,5]."""
        assert Interval(2, 3).is_subset_of(Interval(1, 5))

    def test_is_subset_of_false(self):
        """[1,5] is NOT a subset of [2,3]."""
        assert not Interval(1, 5).is_subset_of(Interval(2, 3))

    def test_is_subset_of_equal(self):
        """An interval is a subset of itself."""
        iv = Interval(1, 3)
        assert iv.is_subset_of(iv)


# ===================================================================
# Set operations
# ===================================================================


class TestSetOperations:
    """Union and intersection."""

    def test_union_overlapping(self):
        """Union of overlapping intervals gives the hull."""
        u = Interval(1, 5).union(Interval(3, 7))
        assert u.low == 1.0
        assert u.high == 7.0

    def test_union_disjoint(self):
        """Union of disjoint intervals gives the hull (not set union)."""
        u = Interval(1, 2).union(Interval(4, 5))
        assert u.low == 1.0
        assert u.high == 5.0

    def test_intersection_overlapping(self):
        """Intersection of overlapping intervals returns the shared region."""
        inter = Interval(1, 5).intersection(Interval(3, 7))
        assert inter is not None
        assert inter.low == 3.0
        assert inter.high == 5.0

    def test_intersection_disjoint(self):
        """Intersection of disjoint intervals returns None."""
        assert Interval(1, 2).intersection(Interval(3, 4)) is None

    def test_intersection_single_point(self):
        """Intervals touching at a single point should have a degenerate
        intersection."""
        inter = Interval(1, 3).intersection(Interval(3, 5))
        assert inter is not None
        assert inter.is_degenerate
        assert inter.low == 3.0


# ===================================================================
# String representation
# ===================================================================


class TestRepr:
    """Test __repr__ and __str__."""

    def test_repr(self):
        """__repr__ should be Interval(low, high)."""
        iv = Interval(1.0, 2.0)
        assert repr(iv) == "Interval(1.0, 2.0)"

    def test_str(self):
        """__str__ should be [low, high]."""
        iv = Interval(1.0, 2.0)
        assert str(iv) == "[1.0, 2.0]"

    def test_to_tuple(self):
        """to_tuple should return a (low, high) pair."""
        assert Interval(1, 3).to_tuple() == (1.0, 3.0)
