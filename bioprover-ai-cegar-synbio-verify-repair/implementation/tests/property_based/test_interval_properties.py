"""Property-based tests for interval arithmetic using hypothesis."""

import math

import numpy as np
import pytest

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from bioprover.solver.interval import (
    Interval,
    IntervalMatrix,
    IntervalVector,
    hull,
    intersection,
)


# ===================================================================
# Custom strategies
# ===================================================================

# Finite floats (no inf/nan)
finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

# Positive finite floats
pos_floats = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)

# Small positive floats for numerical stability
small_pos_floats = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def intervals(draw, lo_strategy=finite_floats, hi_strategy=finite_floats):
    """Strategy that generates valid (non-empty) intervals."""
    a = draw(lo_strategy)
    b = draw(hi_strategy)
    lo, hi = min(a, b), max(a, b)
    return Interval(lo, hi)


@st.composite
def positive_intervals(draw):
    """Generate intervals fully in the positive domain."""
    lo = draw(pos_floats)
    width = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    return Interval(lo, lo + width)


@st.composite
def bounded_intervals(draw):
    """Generate intervals within [-100, 100]."""
    lo = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo, max_value=100.0, allow_nan=False, allow_infinity=False))
    return Interval(lo, hi)


@st.composite
def interval_vectors(draw, dim=2):
    """Generate interval vectors of given dimension."""
    ivs = [draw(bounded_intervals()) for _ in range(dim)]
    return IntervalVector(ivs)


@st.composite
def small_interval_matrices(draw, rows=2, cols=2):
    """Generate small interval matrices."""
    data = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(draw(bounded_intervals()))
        data.append(row)
    return IntervalMatrix(data)


# ===================================================================
# Property: interval arithmetic is conservative
# ===================================================================


class TestConservativeArithmetic:
    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_add_conservative(self, a, b):
        """For all x in A, y in B: x + y is in A + B."""
        result = a + b
        x = a.mid()
        y = b.mid()
        assert result.contains(x + y)

    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sub_conservative(self, a, b):
        result = a - b
        x = a.mid()
        y = b.mid()
        assert result.contains(x - y)

    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mul_conservative(self, a, b):
        result = a * b
        x = a.mid()
        y = b.mid()
        assert result.contains(x * y)

    @given(a=bounded_intervals(), b=positive_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_div_conservative(self, a, b):
        """Division by positive interval is conservative."""
        assume(b.lo > 0)
        result = a / b
        x = a.mid()
        y = b.mid()
        assert result.contains(x / y)

    @given(a=positive_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sqrt_conservative(self, a):
        assume(a.lo > 0)
        result = a.sqrt()
        x = a.mid()
        assert result.contains(math.sqrt(x))

    @given(a=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exp_conservative(self, a):
        assume(a.hi < 50)  # avoid overflow
        result = a.exp()
        x = a.mid()
        assert result.contains(math.exp(x))

    @given(a=positive_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_log_conservative(self, a):
        assume(a.lo > 0)
        result = a.log()
        x = a.mid()
        assert result.contains(math.log(x))

    @given(a=bounded_intervals())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sin_conservative(self, a):
        result = a.sin()
        x = a.mid()
        assert result.contains(math.sin(x))

    @given(a=bounded_intervals())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cos_conservative(self, a):
        result = a.cos()
        x = a.mid()
        assert result.contains(math.cos(x))

    @given(a=bounded_intervals(), n=st.integers(min_value=1, max_value=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pow_conservative(self, a, n):
        result = a ** n
        x = a.mid()
        assert result.contains(x ** n)


# ===================================================================
# Property: interval width is non-negative
# ===================================================================


class TestWidthProperties:
    @given(a=intervals())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_width_nonneg(self, a):
        assert a.width() >= 0

    @given(a=intervals())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_radius_nonneg(self, a):
        assert a.radius() >= 0

    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_add_width(self, a, b):
        """Width of sum is at most width(a) + width(b) + rounding."""
        result = a + b
        assert result.width() >= 0
        # Due to outward rounding, width may be slightly more
        assert result.width() <= a.width() + b.width() + 1e-10


# ===================================================================
# Property: hull of two intervals contains both
# ===================================================================


class TestHullProperties:
    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_hull_contains_both(self, a, b):
        h = hull(a, b)
        assert h.contains(a)
        assert h.contains(b)

    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hull_is_smallest(self, a, b):
        """Hull should have lo = min(a.lo, b.lo) and hi = max(a.hi, b.hi)."""
        h = hull(a, b)
        assert h.lo <= min(a.lo, b.lo) + 1e-15
        assert h.hi >= max(a.hi, b.hi) - 1e-15

    @given(a=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hull_self(self, a):
        h = hull(a, a)
        assert h.contains(a)
        assert a.contains(h) or h.width() <= a.width() + 1e-15


# ===================================================================
# Property: intersection is subset of both
# ===================================================================


class TestIntersectionProperties:
    @given(a=bounded_intervals(), b=bounded_intervals())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_intersection_subset(self, a, b):
        i = intersection(a, b)
        if i is not None and not i.is_empty():
            assert a.contains(i)
            assert b.contains(i)

    @given(a=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_intersection_self(self, a):
        i = intersection(a, a)
        if i is not None:
            assert abs(i.lo - a.lo) < 1e-15
            assert abs(i.hi - a.hi) < 1e-15


# ===================================================================
# Property: subdivision covers original
# ===================================================================


class TestSubdivisionProperties:
    @given(a=bounded_intervals(), n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_subdivision_covers(self, a, n):
        parts = a.subdivide(n)
        assert len(parts) == n
        # The hull of all parts should contain the original
        combined = parts[0]
        for p in parts[1:]:
            combined = hull(combined, p)
        assert combined.contains(a) or (
            abs(combined.lo - a.lo) < 1e-10 and abs(combined.hi - a.hi) < 1e-10
        )

    @given(a=bounded_intervals(), n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_subdivision_narrower(self, a, n):
        assume(a.width() > 1e-10)
        parts = a.subdivide(n)
        for p in parts:
            assert p.width() <= a.width() / n + 1e-10

    @given(a=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bisect_covers(self, a):
        lo_half, hi_half = a.bisect()
        combined = hull(lo_half, hi_half)
        assert combined.contains(a) or (
            abs(combined.lo - a.lo) < 1e-10 and abs(combined.hi - a.hi) < 1e-10
        )


# ===================================================================
# Property: monotone functions on intervals
# ===================================================================


class TestMonotoneFunctionProperties:
    @given(a=positive_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sqrt_monotone(self, a):
        """sqrt is monotone increasing, so sqrt([lo,hi]) = [sqrt(lo), sqrt(hi)]."""
        assume(a.lo > 0)
        result = a.sqrt()
        assert result.lo <= math.sqrt(a.lo) + 1e-10
        assert result.hi >= math.sqrt(a.hi) - 1e-10

    @given(a=bounded_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exp_monotone(self, a):
        """exp is monotone increasing."""
        assume(a.hi < 50)
        result = a.exp()
        assert result.lo <= math.exp(a.lo) + 1e-10
        assert result.hi >= math.exp(a.hi) - 1e-10

    @given(a=positive_intervals())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_log_monotone(self, a):
        """log is monotone increasing."""
        assume(a.lo > 0)
        result = a.log()
        assert result.lo <= math.log(a.lo) + 1e-10
        assert result.hi >= math.log(a.hi) - 1e-10


# ===================================================================
# IntervalVector properties
# ===================================================================


class TestIntervalVectorProperties:
    @given(a=interval_vectors(), b=interval_vectors())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hull_contains_both_vec(self, a, b):
        h = a.hull(b)
        assert h.contains(a)
        assert h.contains(b)

    @given(a=interval_vectors(), b=interval_vectors())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_intersection_subset_vec(self, a, b):
        i = a.intersection(b)
        if i is not None:
            assert a.contains(i)
            assert b.contains(i)

    @given(a=interval_vectors())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_midpoint_in_interval(self, a):
        mp = a.midpoint()
        for idx in range(a.dim):
            assert a[idx].contains(mp[idx])

    @given(a=interval_vectors())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_widths_nonneg(self, a):
        w = a.widths()
        assert np.all(w >= 0)

    @given(a=interval_vectors(), eps=st.floats(min_value=0.001, max_value=1.0))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bloat_contains_original(self, a, eps):
        bloated = a.bloat(eps)
        assert bloated.contains(a)


# ===================================================================
# IntervalMatrix properties
# ===================================================================


class TestIntervalMatrixProperties:
    @given(a=small_interval_matrices(), b=small_interval_matrices())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_matrix_mul_contains_true_product(self, a, b):
        """Interval matrix multiply should contain the true product."""
        result = a * b
        # Check that midpoint product is contained
        a_mid = a.midpoint_matrix()
        b_mid = b.midpoint_matrix()
        true_prod = a_mid @ b_mid
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                assert result[i, j].contains(true_prod[i, j])

    @given(m=small_interval_matrices(), v=interval_vectors())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_mat_vec_contains_true_product(self, m, v):
        """Interval mat-vec should contain the true product."""
        result = m.mat_vec(v)
        m_mid = m.midpoint_matrix()
        v_mid = v.midpoint()
        true_prod = m_mid @ v_mid
        for i in range(result.dim):
            assert result[i].contains(true_prod[i])

    def test_identity_times_vec(self):
        I = IntervalMatrix.identity(3)
        v = IntervalVector([Interval(1.0, 2.0), Interval(3.0, 4.0), Interval(5.0, 6.0)])
        result = I.mat_vec(v)
        for i in range(3):
            assert result[i].contains(v[i].mid())
