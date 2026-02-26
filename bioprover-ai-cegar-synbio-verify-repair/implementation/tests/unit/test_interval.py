"""Unit tests for interval arithmetic — Interval, IntervalVector, IntervalMatrix."""

import math

import numpy as np
import pytest

from bioprover.solver.interval import (
    Interval,
    IntervalMatrix,
    IntervalVector,
    hausdorff_distance,
    hull,
    intersection,
    midpoint,
    pow_interval,
    radius,
    subdivision,
    width,
)

# ===================================================================
# Interval — construction
# ===================================================================


class TestIntervalConstruction:
    def test_point_interval(self):
        iv = Interval(3.0)
        assert iv.lo == 3.0
        assert iv.hi == 3.0

    def test_proper_interval(self):
        iv = Interval(1.0, 5.0)
        assert iv.lo == 1.0
        assert iv.hi == 5.0

    def test_reversed_bounds_still_stored(self):
        """If lo > hi the interval is considered empty."""
        iv = Interval(5.0, 1.0)
        assert iv.is_empty()

    def test_repr_contains_bounds(self):
        iv = Interval(1.0, 2.0)
        r = repr(iv)
        assert "1" in r and "2" in r


# ===================================================================
# Interval — predicates
# ===================================================================


class TestIntervalPredicates:
    def test_is_thin_point(self, thin_interval):
        assert thin_interval.is_thin()

    def test_is_thin_tolerance(self):
        iv = Interval(1.0, 1.0 + 1e-16)
        assert iv.is_thin(tol=1e-15)

    def test_contains_zero(self, symmetric_interval):
        assert symmetric_interval.contains_zero()

    def test_not_contains_zero(self, positive_interval):
        assert not positive_interval.contains_zero()

    def test_is_positive(self, positive_interval):
        assert positive_interval.is_positive()

    def test_is_negative(self):
        iv = Interval(-5.0, -1.0)
        assert iv.is_negative()

    def test_contains_float(self, unit_interval):
        assert unit_interval.contains(0.5)
        assert not unit_interval.contains(1.5)

    def test_contains_interval(self, unit_interval):
        inner = Interval(0.2, 0.8)
        assert unit_interval.contains(inner)
        assert not inner.contains(unit_interval)

    def test_overlaps(self):
        a = Interval(0.0, 2.0)
        b = Interval(1.0, 3.0)
        assert a.overlaps(b) and b.overlaps(a)

    def test_no_overlap(self):
        a = Interval(0.0, 1.0)
        b = Interval(2.0, 3.0)
        assert not a.overlaps(b)

    def test_strictly_contains(self, unit_interval):
        inner = Interval(0.1, 0.9)
        assert unit_interval.strictly_contains(inner)
        assert not unit_interval.strictly_contains(unit_interval)


# ===================================================================
# Interval — scalar queries
# ===================================================================


class TestIntervalScalarQueries:
    def test_mid(self, unit_interval):
        assert unit_interval.mid() == pytest.approx(0.5)

    def test_width(self, unit_interval):
        assert unit_interval.width() == pytest.approx(1.0)

    def test_radius(self, unit_interval):
        assert unit_interval.radius() == pytest.approx(0.5)

    def test_mignitude_contains_zero(self, symmetric_interval):
        assert symmetric_interval.mignitude() == pytest.approx(0.0)

    def test_mignitude_positive(self, positive_interval):
        assert positive_interval.mignitude() == pytest.approx(1.0)

    def test_magnitude(self, symmetric_interval):
        assert symmetric_interval.magnitude() == pytest.approx(1.0)


# ===================================================================
# Interval — arithmetic with known results
# ===================================================================


class TestIntervalArithmetic:
    def test_add(self):
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a + b
        assert c.lo <= 4.0 and c.hi >= 6.0

    def test_add_scalar(self):
        a = Interval(1.0, 2.0)
        c = a + 5.0
        assert c.lo <= 6.0 and c.hi >= 7.0

    def test_radd(self):
        a = Interval(1.0, 2.0)
        c = 3.0 + a
        assert c.lo <= 4.0 and c.hi >= 5.0

    def test_sub(self):
        a = Interval(3.0, 5.0)
        b = Interval(1.0, 2.0)
        c = a - b
        assert c.lo <= 1.0 and c.hi >= 3.0

    def test_neg(self, positive_interval):
        neg = -positive_interval
        assert neg.lo <= -3.0 and neg.hi >= -1.0

    def test_mul_positive(self):
        a = Interval(2.0, 3.0)
        b = Interval(4.0, 5.0)
        c = a * b
        assert c.lo <= 8.0 and c.hi >= 15.0

    def test_mul_mixed(self):
        a = Interval(-1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a * b
        assert c.lo <= -4.0 and c.hi >= 8.0

    def test_div(self):
        a = Interval(2.0, 6.0)
        b = Interval(1.0, 3.0)
        c = a / b
        assert c.lo <= 2.0 / 3.0 and c.hi >= 6.0

    def test_div_by_interval_containing_zero_raises(self):
        a = Interval(1.0, 2.0)
        b = Interval(-1.0, 1.0)
        with pytest.raises((ZeroDivisionError, ValueError)):
            _ = a / b

    def test_pow_even(self):
        a = Interval(-2.0, 3.0)
        c = a ** 2
        # [0, 9] because even power
        assert c.lo <= 0.0 and c.hi >= 9.0

    def test_pow_odd(self):
        a = Interval(-2.0, 3.0)
        c = a ** 3
        assert c.lo <= -8.0 and c.hi >= 27.0


# ===================================================================
# Interval — elementary functions
# ===================================================================


class TestIntervalElementary:
    def test_sqrt(self, positive_interval):
        s = positive_interval.sqrt()
        assert s.lo <= 1.0 and s.hi >= math.sqrt(3.0)

    def test_exp(self, unit_interval):
        e = unit_interval.exp()
        assert e.lo <= 1.0 and e.hi >= math.e

    def test_log(self, positive_interval):
        lg = positive_interval.log()
        assert lg.lo <= 0.0 and lg.hi >= math.log(3.0)

    def test_sin(self, unit_interval):
        s = unit_interval.sin()
        assert s.lo <= 0.0 and s.hi >= math.sin(1.0)

    def test_cos(self, unit_interval):
        c = unit_interval.cos()
        # cos(0) = 1, cos(1) ≈ 0.54 → [cos(1), 1]
        assert c.lo <= math.cos(1.0) and c.hi >= math.cos(1.0)


# ===================================================================
# Interval — outward rounding property
# ===================================================================


class TestOutwardRounding:
    """Result interval must contain the true value for all inputs in the operand intervals."""

    @pytest.mark.parametrize("a_val", [1.0, 1.5, 2.0])
    @pytest.mark.parametrize("b_val", [3.0, 3.5, 4.0])
    def test_add_contains_true(self, a_val, b_val):
        A = Interval(1.0, 2.0)
        B = Interval(3.0, 4.0)
        result = A + B
        assert result.contains(a_val + b_val)

    @pytest.mark.parametrize("a_val", [1.0, 1.5, 2.0])
    @pytest.mark.parametrize("b_val", [3.0, 3.5, 4.0])
    def test_mul_contains_true(self, a_val, b_val):
        A = Interval(1.0, 2.0)
        B = Interval(3.0, 4.0)
        result = A * B
        assert result.contains(a_val * b_val)

    @pytest.mark.parametrize("x", [1.0, 2.0, 3.0])
    def test_sqrt_contains_true(self, x):
        iv = Interval(1.0, 3.0)
        result = iv.sqrt()
        assert result.contains(math.sqrt(x))

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0])
    def test_exp_contains_true(self, x):
        iv = Interval(0.0, 1.0)
        result = iv.exp()
        assert result.contains(math.exp(x))


# ===================================================================
# Free functions — hull, intersection, midpoint, width
# ===================================================================


class TestFreeFunctions:
    def test_hull(self):
        a = Interval(1.0, 3.0)
        b = Interval(5.0, 7.0)
        h = hull(a, b)
        assert h.lo <= 1.0 and h.hi >= 7.0

    def test_intersection_overlap(self):
        a = Interval(0.0, 3.0)
        b = Interval(2.0, 5.0)
        i = intersection(a, b)
        assert i is not None
        assert i.lo <= 2.0 and i.hi >= 3.0

    def test_intersection_disjoint(self):
        a = Interval(0.0, 1.0)
        b = Interval(2.0, 3.0)
        i = intersection(a, b)
        assert i is None or i.is_empty()

    def test_midpoint_fn(self, unit_interval):
        assert midpoint(unit_interval) == pytest.approx(0.5)

    def test_width_fn(self, unit_interval):
        assert width(unit_interval) == pytest.approx(1.0)

    def test_radius_fn(self, unit_interval):
        assert radius(unit_interval) == pytest.approx(0.5)

    def test_pow_interval(self):
        base = Interval(2.0, 3.0)
        exp = Interval(1.0, 2.0)
        result = pow_interval(base, exp)
        # Must contain 2^1 = 2 and 3^2 = 9
        assert result.lo <= 2.0 and result.hi >= 9.0


# ===================================================================
# Interval — subdivision
# ===================================================================


class TestIntervalSubdivision:
    def test_bisect(self, unit_interval):
        lo_half, hi_half = unit_interval.bisect()
        assert lo_half.hi >= hi_half.lo - 1e-15  # overlap or touch
        assert lo_half.lo == pytest.approx(0.0)
        assert hi_half.hi == pytest.approx(1.0)

    def test_subdivide_n(self, unit_interval):
        parts = unit_interval.subdivide(4)
        assert len(parts) == 4
        # Each part is narrower
        for p in parts:
            assert p.width() <= unit_interval.width() / 4 + 1e-15
        # Union covers original
        combined = parts[0]
        for p in parts[1:]:
            combined = hull(combined, p)
        assert combined.contains(unit_interval)


# ===================================================================
# IntervalVector
# ===================================================================


class TestIntervalVector:
    def test_dim(self, interval_vector_2d):
        assert interval_vector_2d.dim == 2
        assert len(interval_vector_2d) == 2

    def test_getitem(self, interval_vector_2d):
        assert interval_vector_2d[0].lo == pytest.approx(0.0)
        assert interval_vector_2d[1].hi == pytest.approx(3.0)

    def test_midpoint(self, interval_vector_2d):
        mp = interval_vector_2d.midpoint()
        np.testing.assert_allclose(mp, [0.5, 2.5])

    def test_widths(self, interval_vector_2d):
        w = interval_vector_2d.widths()
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_from_bounds(self):
        iv = IntervalVector.from_bounds(
            np.array([0.0, 1.0]), np.array([2.0, 3.0])
        )
        assert iv.dim == 2
        assert iv[0].lo == pytest.approx(0.0) and iv[0].hi == pytest.approx(2.0)

    def test_from_midpoint_radius(self):
        iv = IntervalVector.from_midpoint_radius(
            np.array([1.0, 2.0]), np.array([0.5, 0.5])
        )
        assert iv[0].lo <= 0.5 and iv[0].hi >= 1.5

    def test_constant(self):
        iv = IntervalVector.constant(np.array([1.0, 2.0]))
        assert iv[0].is_thin() and iv[1].is_thin()

    def test_add(self, interval_vector_2d):
        b = IntervalVector([Interval(1.0, 2.0), Interval(0.0, 1.0)])
        c = interval_vector_2d + b
        assert c[0].lo <= 1.0 and c[0].hi >= 3.0
        assert c[1].lo <= 2.0 and c[1].hi >= 4.0

    def test_sub(self, interval_vector_2d):
        b = IntervalVector([Interval(0.0, 0.5), Interval(0.0, 0.5)])
        c = interval_vector_2d - b
        assert c[0].lo <= -0.5 and c[0].hi >= 1.0

    def test_contains(self, interval_vector_2d):
        inner = IntervalVector([Interval(0.2, 0.8), Interval(2.2, 2.8)])
        assert interval_vector_2d.contains(inner)
        assert not inner.contains(interval_vector_2d)

    def test_hull(self, interval_vector_2d):
        other = IntervalVector([Interval(3.0, 4.0), Interval(0.0, 1.0)])
        h = interval_vector_2d.hull(other)
        assert h[0].lo <= 0.0 and h[0].hi >= 4.0
        assert h[1].lo <= 0.0 and h[1].hi >= 3.0

    def test_intersection(self, interval_vector_2d):
        other = IntervalVector([Interval(0.5, 1.5), Interval(2.5, 3.5)])
        i = interval_vector_2d.intersection(other)
        assert i is not None
        assert i[0].lo <= 0.5 and i[0].hi >= 1.0

    def test_bloat(self, interval_vector_2d):
        bloated = interval_vector_2d.bloat(0.1)
        assert bloated[0].lo <= -0.1 and bloated[0].hi >= 1.1

    def test_project(self, interval_vector_2d):
        proj = interval_vector_2d.project([1])
        assert proj.dim == 1
        assert proj[0].lo == pytest.approx(2.0)


# ===================================================================
# IntervalMatrix
# ===================================================================


class TestIntervalMatrix:
    def test_identity(self, identity_interval_matrix):
        assert identity_interval_matrix.shape == (2, 2)
        assert identity_interval_matrix[0, 0].lo == pytest.approx(1.0)
        assert identity_interval_matrix[0, 1].lo == pytest.approx(0.0)

    def test_from_numpy(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        im = IntervalMatrix.from_numpy(m)
        assert im[0, 0].lo == pytest.approx(1.0)
        assert im[1, 1].hi == pytest.approx(4.0)

    def test_zeros(self):
        z = IntervalMatrix.zeros(3, 2)
        assert z.shape == (3, 2)
        assert z[0, 0].lo == pytest.approx(0.0)

    def test_add(self, identity_interval_matrix):
        s = identity_interval_matrix + identity_interval_matrix
        assert s[0, 0].lo <= 2.0 and s[0, 0].hi >= 2.0

    def test_sub(self, identity_interval_matrix):
        d = identity_interval_matrix - identity_interval_matrix
        assert d[0, 0].contains(0.0)

    def test_mul(self, identity_interval_matrix):
        """I * I = I."""
        prod = identity_interval_matrix * identity_interval_matrix
        assert prod[0, 0].contains(1.0)
        assert prod[0, 1].contains(0.0)

    def test_mat_vec(self, identity_interval_matrix, interval_vector_2d):
        result = identity_interval_matrix.mat_vec(interval_vector_2d)
        assert result[0].lo == pytest.approx(interval_vector_2d[0].lo)
        assert result[1].hi == pytest.approx(interval_vector_2d[1].hi)

    def test_scalar_mul(self, identity_interval_matrix):
        scaled = identity_interval_matrix.scalar_mul(Interval(2.0, 3.0))
        assert scaled[0, 0].lo <= 2.0 and scaled[0, 0].hi >= 3.0

    def test_transpose(self):
        data = [[Interval(1.0, 2.0), Interval(3.0, 4.0)]]
        m = IntervalMatrix(data)
        t = m.transpose()
        assert t.shape == (2, 1)
        assert t[0, 0].lo == pytest.approx(1.0)
        assert t[1, 0].lo == pytest.approx(3.0)

    def test_midpoint_matrix(self, identity_interval_matrix):
        mp = identity_interval_matrix.midpoint_matrix()
        np.testing.assert_allclose(mp, np.eye(2))

    def test_spectral_radius_bound(self, identity_interval_matrix):
        sr = identity_interval_matrix.spectral_radius_bound()
        assert sr >= 1.0

    def test_max_width(self, identity_interval_matrix):
        assert identity_interval_matrix.max_width() == pytest.approx(0.0)


# ===================================================================
# Module free functions — hausdorff, subdivision
# ===================================================================


class TestModuleFunctions:
    def test_hausdorff_distance_same(self, interval_vector_2d):
        assert hausdorff_distance(interval_vector_2d, interval_vector_2d) == pytest.approx(0.0)

    def test_hausdorff_distance_shifted(self):
        a = IntervalVector([Interval(0.0, 1.0)])
        b = IntervalVector([Interval(1.0, 2.0)])
        d = hausdorff_distance(a, b)
        assert d == pytest.approx(1.0)

    def test_subdivision_covers(self, interval_vector_2d):
        parts = subdivision(interval_vector_2d, n_per_dim=2)
        assert len(parts) == 4  # 2^2
        # The hull of all parts must contain the original
        combined = parts[0]
        for p in parts[1:]:
            combined = combined.hull(p)
        assert combined.contains(interval_vector_2d)


# ===================================================================
# ValidatedInterval
# ===================================================================


class TestValidatedInterval:
    """Tests for mpmath-backed ValidatedInterval."""

    def test_construction_point(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(3.0)
        assert vi.lo == pytest.approx(3.0)
        assert vi.hi == pytest.approx(3.0)

    def test_construction_range(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(1.0, 2.0)
        assert vi.lo <= 1.0
        assert vi.hi >= 2.0

    def test_add(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        a = ValidatedInterval(1.0, 2.0)
        b = ValidatedInterval(3.0, 4.0)
        c = a + b
        assert c.lo <= 4.0 and c.hi >= 6.0

    def test_sub(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        a = ValidatedInterval(3.0, 5.0)
        b = ValidatedInterval(1.0, 2.0)
        c = a - b
        assert c.lo <= 1.0 and c.hi >= 3.0

    def test_mul(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        a = ValidatedInterval(2.0, 3.0)
        b = ValidatedInterval(4.0, 5.0)
        c = a * b
        assert c.lo <= 8.0 and c.hi >= 15.0

    def test_div(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        a = ValidatedInterval(2.0, 6.0)
        b = ValidatedInterval(1.0, 3.0)
        c = a / b
        assert c.lo <= 2.0 / 3.0 and c.hi >= 6.0

    def test_sqrt(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(1.0, 4.0)
        s = vi.sqrt()
        assert s.lo <= 1.0 and s.hi >= 2.0

    def test_exp(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(0.0, 1.0)
        e = vi.exp()
        assert e.lo <= 1.0 and e.hi >= math.e

    def test_log(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(1.0, 3.0)
        lg = vi.log()
        assert lg.lo <= 0.0 and lg.hi >= math.log(3.0)

    def test_sin(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(0.0, 1.0)
        s = vi.sin()
        assert s.lo <= 0.0 and s.hi >= math.sin(1.0)

    def test_cos(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(0.0, 1.0)
        c = vi.cos()
        assert c.lo <= math.cos(1.0) and c.hi >= math.cos(1.0)

    def test_to_interval(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        vi = ValidatedInterval(1.0, 2.0)
        iv = vi.to_interval()
        assert iv.lo <= 1.0 and iv.hi >= 2.0

    def test_interval_to_validated(self):
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        iv = Interval(1.0, 3.0)
        vi = iv.to_validated()
        assert isinstance(vi, ValidatedInterval)
        assert vi.lo <= 1.0 and vi.hi >= 3.0

    def test_enclosure_property(self):
        """ValidatedInterval must enclose the true result for all inputs."""
        from bioprover.solver.interval import ValidatedInterval, _HAS_MPMATH
        if not _HAS_MPMATH:
            pytest.skip("mpmath not installed")
        a = ValidatedInterval(1.0, 2.0)
        b = ValidatedInterval(3.0, 4.0)
        result = a * b
        for x in [1.0, 1.5, 2.0]:
            for y in [3.0, 3.5, 4.0]:
                assert result.contains(x * y)
